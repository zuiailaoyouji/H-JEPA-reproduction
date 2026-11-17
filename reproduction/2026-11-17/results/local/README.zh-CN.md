# H-JEPA 本机环境（仅本地，不提交 GitHub）

配置日期：2026-10-25。环境目录：`/home/songxw/miniconda3/envs/hjepa`。

## 激活

```bash
source /home/songxw/H-JEPA/local_env/activate.sh
python --version
which python
```

激活脚本会退出当前 Python venv，然后激活 Conda `hjepa`，设置默认
`HJEPA_HOME=/home/songxw/hjepa-data`、`MUJOCO_GL=egl` 和 4 个 OpenMP 线程。
已有的 HJEPA_HOME 和 OMP_NUM_THREADS 会保留。也可使用 `conda activate hjepa`，
再自行设置这些变量。数据目录已创建，目前没有下载数据集或模型。

## 设备及版本选择

- Ubuntu 22.04.5；双 AMD EPYC 7452，64 个物理核心 / 128 线程；约 503 GiB 内存。
- 本机可见 7 张 NVIDIA A100 PCIe 40GB；驱动 570.211.01。
- Python 3.11.13；PyTorch 2.7.1+cu128；torchvision 0.22.1+cu128；torchaudio 2.7.1+cu128。
- stable-pretraining 0.1.7；Gymnasium 1.2.3；MuJoCo 3.14.0；dm_control 1.0.47；OGBench 1.2.1。
- 已安装官方 `[train,env]` 全部依赖。完整版本见 `requirements-frozen.txt`。
- Python 来自 conda-forge，CUDA 12.8 运行库来自 PyTorch wheel 依赖。没有更改系统驱动或 CUDA Toolkit。
- JAX 按官方普通 `jax` 依赖安装，使用 CPU；H-JEPA 模型训练由 PyTorch 使用 GPU。

## 安装与重建记录

首次从已有 conda-forge 缓存创建完整 Python 环境，避免 Anaconda 默认通道的条款提示：

```bash
conda create -n hjepa --override-channels -c conda-forge python=3.11.13 --offline -y
```

安装命令使用 uv 指向该 Conda 环境，保持官方核心版本和两个 extras。
本次直连下载比已有代理线路更快，所以仅对安装命令临时设置 NO_PROXY：

```bash
cd /home/songxw/H-JEPA
NO_PROXY='*' no_proxy='*' UV_CONCURRENT_DOWNLOADS=8 UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 \
  uv pip install --python /home/songxw/miniconda3/envs/hjepa/bin/python \
  -e '.[train,env]' --extra-index-url https://download.pytorch.org/whl/cu128
```

`decord==0.6.0` 官方 Linux wheel 文件名为 `py3-none`，但包内 WHEEL 元数据标成
`cp36-cp36m`，导致 pip/uv check 误报平台不兼容。已验证其 ctypes 导入及实际视频解码。
仅对本地 wheel 的平台标签作了修正并重算 RECORD，未修改 Python 代码或共享库；
来源、原始 SHA256 和修复后的 SHA256 在 `decord-metadata-fix.json`。
随后安装 `wheels/decord-0.6.0-py3-none-manylinux2010_x86_64.whl`，两种依赖检查均通过。

`conda-explicit.txt` 保存底层 Conda 包；`requirements-frozen.txt` 保存 Python 包（含本地
decord wheel 和 editable 项目绝对路径）；`environment.yml` 为完整环境导出。
这些文件用于本机重建，迁移到别的路径时须更新本地路径，不能直接当作跨机器锁文件。
`environment.yml` 仅为清单；重建应先使用 Conda explicit 清单创建环境，再通过
`requirements-frozen.txt` 和 PyTorch cu128 索引安装 Python 包，以保留本地修复的 wheel。

此前未完成且引用 Conda 包缓存解释器的 `.venv` 已在目录清理时删除。
当前使用上面的 Conda `hjepa` 环境。

## 验证范围

- Python SSL、SQLite 及训练/仿真依赖导入成功；pip check 和 uv pip check 通过。
- 官方训练入口 `main_hjepa.py --config-name ant_hjepa_l3 --cfg job` 与评估入口
  `eval.py --config-name ant_l3 --cfg job` 成功。
- 一张空闲 A100 上，用原始 `ant_hjepa_l3` 模型配置和 batch=2 合成输入，完成三层
  BF16 前向、损失、反向传播和 AdamW 更新；三层梯度均有限。
- 测试模型约 3476 万参数，CUDA 峰值已分配显存约 695 MiB。这是小批量功能测试，
  不能据此宣称完整训练所需显存或吞吐量。
- Push-T、FourRoom、Visual AntMaze、Cube 均成功 reset、step、render；MuJoCo 使用 EGL。
  部分原仓库环境会触发 Gymnasium 的观测类型/空间警告；本次未修改仿真源码，
  此检查只验证能运行和渲染，不等同于完整 Gymnasium API 一致性测试。
- decord 成功解码本地生成的四帧测试视频。
- 验证脚本为 `smoke_test.py`，可在空闲 GPU 上重新执行；脚本会生成
  `validation.json` 和 `smoke-video.avi`。

安装和验证日志、依赖检查输出、训练/评估配置快照及测试输出已清理。
保留激活脚本、验证脚本、环境重建清单，以及修复后的 decord wheel 和修复记录。

## 后续训练

先按项目 README 准备数据、probing 文件及 evaluation tasks。开始前用 `nvidia-smi`
选择当时空闲的 GPU，当前编号 5 只是本次验证使用的设备，不代表以后始终空闲。
例如，确认 GPU 5 可用并完成数据准备后：

```bash
source /home/songxw/H-JEPA/local_env/activate.sh
cd /home/songxw/H-JEPA/h_jepa
CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
  python main_hjepa.py --config-name ant_hjepa_l3 seed=42 \
  trainer.devices=1 trainer.precision=bf16-mixed
```

保留官方 batch_size=128、num_workers=6 作为起点；本次未对完整 batch 训练做容量测试。
如需降低每卡 batch，应同时核对总 batch 与论文设置的差异。原配置 `devices: auto`
可能使用所有可见 GPU，因此共享机器应显式指定 CUDA_VISIBLE_DEVICES 和设备数。
环境安装后磁盘剩余约 62 GiB；大数据集和训练输出需要另外核对容量。

## Git 范围

`local_env/` 与 `.venv/` 被项目 `.gitignore` 排除；Conda 环境、uv 缓存、数据目录均
位于仓库外。本次没有执行 git add、commit 或 push。共享 uv 缓存未清理。
