# 资产与环境恢复

原环境使用 Python 3.11.13、PyTorch 2.7.1、MuJoCo 3.14.0、dm_control 1.0.47 和 stable-pretraining 0.1.7。安装入口 `setup.sh` 使用移除机器路径后的冻结依赖；原始 `requirements-frozen.txt`、`environment.yml` 和 conda 显式清单保留为来源记录。不同 GPU/驱动可能需要调整 PyTorch wheel 源。该环境安装步骤依据原环境快照整理，未在另一台机器全新安装验证。

## 固定任务与独有结果

从同仓库的 `reproduction-2026-11-17` Release 下载所有附件，再运行 `scripts/restore_assets.py`。脚本验证每个附件哈希，恢复逐任务输出、固定任务和 31 个独有本地 checkpoint，随后验证解压后的逐文件哈希。checkpoint 归档还包括本地训练的 optimizer/resume 状态；这些权重仅支持报告中注明的单 seed L2 训练诊断。

## 官方模型

官方权重使用固定版本，模型文件不重复存入此归档。安装 huggingface_hub 后，可从 `jepa-world-models/h-jepa` 下载以下版本到 `$HJEPA_HOME/ckpts`：

```bash
hf download jepa-world-models/h-jepa \
  --revision 08fc8a21208671a87741f2960aeba9b5db4420e0 \
  --local-dir "$HJEPA_HOME/ckpts"

python code/h_jepa/scripts/verify_official_weights.py \
  --root "$HJEPA_HOME/ckpts" \
  --tree-json provenance/assets/model_tree_08fc8a21208671a87741f2960aeba9b5db4420e0.json \
  --revision 08fc8a21208671a87741f2960aeba9b5db4420e0 --write-manifest
```

聚合已有规划结果只需要恢复脚本放置的官方 manifest，不需要重新下载权重。重跑 GPU 评测则需要真实权重。

## 数据

原始 HDF5 和 DROID clip 不重复上传 GitHub。DROID 的 16 个评测 clip 来自数据仓库 `jepa-world-models/h-jepa` 的 `cc05cb8d2aaf9aec0d5364d073c6369ff760b60b` 版本，其中 `droid_raw_eval16.tar` 的 SHA-256 为 `3f77463d316b325b0827b8acc75095db46feac9e0c3d3637cae2c8b740c6074b`。解压后位于 `$HJEPA_HOME/droid/droid_raw/1.0.1`。

仿真数据的名称、大小、结构审计和已记录的 SHA-256 位于 `results/local/dataset-audit-20261026.json`、`results/local/dataset-sha256-20261026.json` 和 `provenance/assets/dataset_SHA256SUMS`。Cube 采用 9,800 / 200 episode 的训练/验证切分；数据采集和切分脚本位于 `code/scripts/data/`。原始仓库 README 中保留公开数据获取说明。全部数据的原始下载 revision 未完整记录，不能将这些数据描述为全部具有已固定的源版本；应优先使用已记录的内容哈希核对文件。

不要把 `results/local/` 的旧协议、单 seed 或未完成训练数据与 `results/official/` 的最终公开权重结果合并。远端原始 ZIP 的 probe 取消记录是来源审计；最终 18/18 probe 以本机已完成结果为准。
