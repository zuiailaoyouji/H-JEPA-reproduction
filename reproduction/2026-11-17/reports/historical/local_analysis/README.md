# 训练期间 CPU 复现分析结果

训练日志采集开始时间：2026-10-27T13:41:39+08:00（北京时间，各源文件依次读取）。

本目录完成四环境的数据计数、频率统计、固定投影 SIGReg 分布式验证与新训练日志整理。分析仅用 CPU；未启动模型训练，未改动在训代码或配置。

## Table 3：16/16 项有效 clips 与论文一致

调用当前仓库 HDF5Dataset 与 transition-budget episode 选择逻辑，按 `sum(max(ep_len - loader_span + 1, 0))` 计数。下表是每个 epoch 可用 clips，并非原始 transitions 数量。

| 环境 | L1 | L2（当前训练） | L3 | L4 |
|---|---:|---:|---:|---:|
| fourroom | 952,708 | 915,298 | 840,478 | 690,838 |
| ant | 9,575,000 | 9,200,000 | 8,450,000 | 6,950,000 |
| pusht | 1,981,721 | 1,701,446 | 1,143,623 | 285,931 |
| cube | 1,783,600 | 1,636,600 | 1,342,600 | 754,600 |

四种深度分别含 4/7/13/25 个 L1 帧；观测首尾的最小 inclusive span 为 16/31/61/121，但 loader 预留的 half-open transition span 为 20/35/65/125。混用两者会多计 clips。FourRoom 按默认 subset_seed=0 选择约 100 万 transitions（实际 2,494 episodes、1,000,094 transitions）；Ant 使用全部 25,000 episodes（10,050,000 transitions），Push-T/Cube 使用完整训练文件。详见 [计数 CSV](table3_clip_counts.csv)。

## Table 11：实际 CPU Gloo 分片验证通过

固定同一批 Gaussian 数据（n=256、D=32）、64 个投影方向和 17 个积分点，使用 float64。W=1/2/4 均实际初始化 Gloo process group；调用仓库 SIGReg 的 ECF all_reduce。主指标采用与论文一致的输入表示梯度 `∂TEP/∂Z`：各 rank 的局部输入梯度除以 W 后拼接，并与未分片参考逐元素对比。单位线性层参数梯度仅作为 DDP 辅助校验。

| 分片 | 输入梯度范数 `‖∂TEP/∂Z‖` | TEP | 输入梯度最大绝对误差 | 单位线性层参数梯度范数（辅助） | 参数梯度最大绝对误差（辅助） |
|---|---:|---:|---:|---:|---:|
| 1 x 256 | 0.280711511056 | 0.977725579145 | 0 | 2.162668056770 | 0 |
| 2 x 128 | 0.280711511056 | 0.977725579145 | 6.94e-18 | 2.162668056770 | 2.5e-16 |
| 4 x 64 | 0.280711511056 | 0.977725579145 | 7.81e-18 | 2.162668056770 | 3.89e-16 |

取消 ECF 同步后，三种分片的平均局部 TEP 为 0.977725579 / 0.951421246 / 0.910388510，出现预期的分片漂移。这里复现的是不变性结论；论文未提供相同随机矩阵/seed，因此不宣称复现论文的 TEP=1.372619178543。

PyTorch 对仓库的非 autograd collective 发出兼容性 warning；当前数值验证覆盖其 local backward + DDP 平均语义。积分 buffer 沿用生产实现的 float32 初始化，再转 float64；更换 PyTorch 版本需重跑数值验证。详见 [完整结果](sigreg_sharding.json)、[表格](sigreg_sharding.csv) 与 [执行日志](sigreg_gloo_run.log)。

## Figure 5/12：四环境频率子集，数值尚未对齐论文

使用训练 split 内完整 episode，每 5 个环境步采样一次；每维在 episode 内减均值、除标准差。逐维计算 spectral centroid，再按有效 episode-dimension 等权平均到实体。主结果采用功率 `|rFFT|²` 权重，同时保留幅度 `|rFFT|` 权重作为敏感性分析。无额外平滑、窗函数或线性去趋势；常数维不参与平均。频率单位为 cycles / L1 frame，gap 为最大实体质心 / 最小实体质心。

| 环境 | Episodes | 功率谱 gap | 幅度谱 gap | 最快实体 | 最慢实体 |
|---|---:|---:|---:|---|---|
| fourroom | 2,494 | 2.2772× | 1.5637× | distractor_xy_fast | ego_xy_slow |
| ant | 25,000 | 6.2661× | 2.1725× | body_state_fast | global_xy_slow |
| pusht | 18,685 | 2.3967× | 1.5301× | agent_velocity | block_orientation |
| cube | 9,800 | 2.7290× | 1.7194× | arm_joint_velocities | cube_position |

实体按论文 Table 6：FourRoom 为 ego xy 与 distractor xy；Ant 为 global xy 与 qpos[2:]+qvel（27维）；Push-T 为 agent xy、block xy、block orientation（sin/cos）、agent velocity；Cube 为 block position/orientation、effector position/yaw、gripper、arm joint position/velocity，共7项。训练中的额外诊断 target 不并入 gap。

Ant 的功率谱逐 episode 比值中位数为 6.435×，IQR 为 5.684–7.382×；论文报告 6.4×、4.9–7.8×，dataset gap 为 8.3×。论文还报告导航类 3.1–12.7×、操作类 1.2–1.6×。本次两种权重下均未完全对齐，尚需作者的频谱处理/实体聚合实现确认原因，不能将这些本地数值视为论文数值复现成功。

按论文的“前64条中，episode gap 最接近 dataset gap”规则，选择本地 Ant episode 17，共 81 帧；27维 gap=6.277×，图中13维 qpos gap=5.879×。两者在本地数据也有差别，均已保留，未强制改成论文示例的8.3×。

Push-T 有41个 episode 的 block xy 与 orientation 为常数（每实体82个维度-episode项被排除）；其余常数/非有限值诊断保存在 JSON。四环境扫描未发现非有限实体序列。

图：[实体频率 PNG](frequency_entity_centroids.png) / [PDF](frequency_entity_centroids.pdf)；[Ant 单 episode PNG](ant_frequency_example.png) / [PDF](ant_frequency_example.pdf)。每个实体的质心与四分位数见 [CSV](frequency_entity_centroids.csv)，协议/数据 SHA-256 见 [JSON](frequency_summary.json)，Ant 候选64条与所选轨迹也已导出 CSV。

这些结果覆盖 Figure 5/12 的数据频率部分及 Figure 12a 的本地示例。Figure 5/12b 的 L1/L2 probe NMSE 仍需等待新模型的结束回调；当前只有 seed42，也无法提供论文3个训练种子的 SE。

## 新训练 loss 与结果 manifest

四个 run 都限定为 2026-10-27 09:32 重启的 L2/seed42/protocol2。每个 SPT session 的原始日志先单独复制、哈希，再按启动时间合并；后启动 session 从其首个已记录 global step 起替换旧 session 的整个后缀，避免把恢复前已放弃的分支混进曲线。不完整的末尾 CSV 行丢弃并记录字节数。step 为 logger 的0起始 optimizer step，采样点数不等于训练更新数。

| 环境 | Sessions | 被替换点 | Loss采样点 | 最新step | Epoch（0起始） | Total loss | Pred L1 / L2 | SIGReg L1 / L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pusht | 1 | 0 | 507 | 25349 | 1.0 | 0.34498 | 0.021343 / 0.048734 | 1.2422 / 1.8125 |
| cube | 2 | 38 | 445 | 22249 | 1.0 | 0.38548 | 0.016701 / 0.052378 | 1.4375 / 2.0781 |
| fourroom | 1 | 0 | 518 | 25899 | 3.0 | 0.53642 | 0.011448 / 0.024976 | 1.2344 / 1.5391 |
| ant | 1 | 0 | 577 | 28849 | 0.0 | 0.42838 | 0.028825 / 0.055804 | 1.5859 / 2.2344 |

[Loss 曲线 PNG](training_loss_curves.png) / [PDF](training_loss_curves.pdf) 分开显示 prediction、SIGReg、total 与 LR；loss 使用对数纵轴，保留原始点，无平滑。另存 [逐步数据](training_loss_points.csv) 与 [最新摘要](training_loss_summary.csv)。这是仍在训练的单 seed L2 快照，不能直接作为论文四层训练图或最终收敛结果。

[results_manifest.json](results_manifest.json) 记录源代码/配置哈希、训练数据身份、日志快照、实际 batch/累积、更新预算、学习率、checkpoint 与后续50-task规划配置（seed42、默认chunk_size=1）、probe/decoder配置及现有结果文件。逐 epoch 的 `*_epoch_*_object.ckpt` 单列为中间权重；仅同名 `*_hjepa_l2_object.ckpt` 计为最终权重，最终权重存在也不代表回调已完成。

当前采集环境看不到宿主机训练 PID，PID 不可见不代表进程停止；manifest 另列日志更新时间作为文件层面的进展证据。不会将其他旧 run 的成功率或 NMSE 混入本次结果。

## 重跑

完整分析（CPU，低调度/IO优先级）：

```bash
cd /home/songxw/H-JEPA
nice -n 15 ionice -c 3 /home/songxw/miniconda3/envs/hjepa/bin/python local_env/analysis_20261027/analyze_local_reproduction.py
```

只刷新训练曲线/manifest/报告，不重扫数据：

```bash
/home/songxw/miniconda3/envs/hjepa/bin/python local_env/analysis_20261027/analyze_local_reproduction.py --refresh-logs
```

Gloo 数值验证单独运行 `verify_sigreg_sharding.py`。本目录不注册后台任务；已有数据 SHA-256 来自前一轮完整文件校验，本轮不重复读取约145GB文件。所有内容保存在被 Git 忽略的 local_env 下。
