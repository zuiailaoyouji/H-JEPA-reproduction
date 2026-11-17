# H-JEPA 公开权重与本地训练复现实验报告

**实验报告日期：** 2026-10-29

**整理归档日期：** 2026-11-17。本次仅整理既有实验并重新校验归档，未重跑完整 GPU 评测。实验范围、数值差异和限定条件保留如下。统一入口见 [归档说明](../README.md)。

**论文：** *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning*（arXiv:2610.06805）

**论文副本：** [ppt_assets/hjepa_paper.pdf](https://arxiv.org/abs/2610.06805)

**论文参考数值：** [docs/RESULTS.md](../code/docs/RESULTS.md)

**模型版本：** `jepa-world-models/h-jepa@08fc8a21208671a87741f2960aeba9b5db4420e0`

## 1. 汇报摘要

本次工作完成了公开模型与公开数据能够支持的 H-JEPA 实验复现。最终结果覆盖 84/84 个层数对比单元、108/108 个 cost-ladder 单元、27/27 个 DROID 规划单元和 18/18 个 probe/decoder 单元，共 237 个实验单元；所有聚合结果均来自实际运行，缺失项没有用论文数值回填。

最稳妥的总体结论是：**论文的主要定性结论大部分得到复现，但不能宣称全部论文数值被精确复现。** 具体而言：

1. **完全对齐：** Table 3 的 16 个训练 clip 计数全部与论文一致。
2. **性质复现：** Table 11 的 SIGReg 在 1/2/4 卡等总 batch 分片下保持全局统计量和梯度不变；由于论文没有公开固定随机样本与投影方向，绝对 TEP 数值不应相同。
3. **主要表征结论复现：** Figure 4 中，Ant 身体状态和 FourRoom 干扰物位置随层级升高越来越难解码，而智能体全局位置仍可恢复；解码图也显示腿部姿态/干扰物逐层模糊。
4. **主要规划趋势复现：** Figure 6(bottom) 中，FourRoom、Ant 和 Cube 的层级规划总体优于平坦 LeWM，Push-T 在三、四层明显退化；但若干成功率没有逐项对齐论文。
5. **cost-space 机制复现：** Table 1/9 的核心结论得到复现：Ant 和 FourRoom 使用高层 latent cost 可改善平坦 L1 规划，而 Cube 和 Push-T 没有稳定收益。
6. **部分复现：** Figure 9(b) 的三个公开模型保持 `LeWM+IDM < HWM < H-JEPA` 的均值排序，但 H-JEPA/HWM 的差距小于论文且误差区间重叠；无 IDM 的崩溃模型没有公开，无法复现第一根柱。
7. **未数值对齐：** Figure 5/12 的 probe 趋势成立，但本地推断的频谱聚合给出 Ant `6.266x`、Cube `2.729x`、Push-T `2.671x`，与论文 Ant `8.3x`、操作环境 `1.2-1.6x` 的描述不一致，因此只能作为限定性复现。
8. **不能作为论文复现：** 本地 L2、单 seed 训练 loss 只能说明训练过程正常，不能替代论文 Figure 10 的四层、三 seed 完整曲线。

## 2. 结果来源与可信度边界

### 2.1 两组结果目录

| 目录 | 内容 | 本报告中的用途 |
|---|---|---|
| [local_env/analysis_20261027](../results/local/analysis_20261027) | 本地数据计数、CPU Gloo SIGReg 验证、频率分析、单 seed L2 训练日志、早期旧协议规划汇总 | Table 3、Table 11、Figure 12(a) 的本地诊断；Figure 10-style 训练趋势仅作补充 |
| [latest_complete](../results/official) | 公开权重的最终完整聚合：Figure 4/5/6/7/9(b)/12(b)、Table 1/9 | 本报告的主要公开权重复现结果 |

`latest_complete` 是符号链接，当前指向 `full_merged_20261029`。其 [MERGE_REPORT.md](../results/official/MERGE_REPORT.md) 和 [combined_coverage.json](../results/official/combined_coverage.json) 记录了来源与完整性：DROID 和 Probe/Decoder 来自本机完成的运行，Depth 和 Cost-ladder 来自校验后的结果归档；历史归档共 11,346 个文件，其中 manifest 列出的 11,344 个载荷在 2026-11-17 整理时重新通过 SHA-256 检查；其余 2 个为 README 和 manifest 本身。

### 2.2 状态定义

| 状态 | 含义 |
|---|---|
| 数值复现 | 协议一致，公开数值能够逐项对齐 |
| 结论复现 | 数值有差异，但论文要验证的排序、趋势或机制结论成立 |
| 限定性复现 | 缺少论文未公开的随机实例、参数或部分数据，只能验证性质/定性图像 |
| 未复现 | 当前公开资产不足，或现有结果与论文设定不等价 |

## 3. 原论文图表与复现结果总对应

| 原论文图表 | 当前结果文件 | 复现状态 | 核心判断 | 主要代码 |
|---|---|---|---|---|
| Figure 4 left | [figure_4_left.png](../results/official/figure_4_left.png)、[probe_summary.csv](../results/official/probe_summary.csv) | 结论复现 | 快变量随层级丢失，位置变量保留 | [main_probing_decoding_eval.py](../code/h_jepa/main_probing_decoding_eval.py)、[analyze_probes.py](../code/h_jepa/scripts/analyze_probes.py) |
| Figure 4 right | [figure_4_right.png](../results/official/figure_4_right.png) | 结论复现 | Ant 腿部姿态、FourRoom 蓝色干扰物逐层模糊 | 同上，decoder 使用 seed 42 |
| Figure 5 | [figure_5.png](../results/official/figure_5.png)、[figure_5.csv](../results/official/figure_5.csv) | 部分/限定性 | selective abstraction 趋势成立；频率 gap 未数值对齐，且缺 Humanoid、DROID | [analyze_frequency.py](../code/h_jepa/scripts/analyze_frequency.py)、[analyze_probes.py](../code/h_jepa/scripts/analyze_probes.py) |
| Figure 6 bottom | [figure_6_bottom.png](../results/official/figure_6_bottom.png)、[depth_summary.csv](../results/official/depth_summary.csv) | 结论复现 | 层级规划有效；Push-T 深层退化；部分均值偏离论文 | [eval_depth.sh](../code/h_jepa/scripts/eval_depth.sh)、[eval.py](../code/h_jepa/eval.py)、[analyze_official_release.py](../code/h_jepa/scripts/analyze_official_release.py) |
| Figure 7 | [qualified_figure7/figure_7.png](../results/official/qualified_figure7/figure_7.png) | 限定性复现 | 高层 cost 沿迷宫走廊更平滑、更有全局结构 | [analyze_cost_geometry.py](../code/h_jepa/scripts/analyze_cost_geometry.py) |
| Figure 9(b) | [figure_9b_released_subset.png](../results/official/figure_9b_released_subset.png)、[droid_summary.csv](../results/official/droid_summary.csv) | 部分复现 | 三个公开模型均值排序一致，但数值和优势幅度未完全对齐；缺无 IDM 模型 | [eval_droid.sh](../code/h_jepa/scripts/eval_droid.sh)、[eval.py](../code/h_jepa/eval.py)、[analyze_official_release.py](../code/h_jepa/scripts/analyze_official_release.py) |
| Figure 10 | [training_loss_curves.png](../results/local/analysis_20261027/training_loss_curves.png) | 未复现，仅诊断 | 当前是 L2、seed 42、训练中快照；论文是 L4 三 seed 和 DROID 100 epochs | [analyze_local_reproduction.py](../scripts/historical/analysis_20261027/analyze_local_reproduction.py)、通用脚本 [analyze_training_dynamics.py](../code/h_jepa/scripts/analyze_training_dynamics.py) |
| Figure 12(a) | [ant_frequency_example.png](../results/local/analysis_20261027/ant_frequency_example.png) | 限定性、未数值对齐 | 可见身体状态比全局位置变化快，但本地 gap `6.28x/5.88x`，论文为 `8.3x` | [analyze_local_reproduction.py](../scripts/historical/analysis_20261027/analyze_local_reproduction.py)、通用脚本 [analyze_frequency.py](../code/h_jepa/scripts/analyze_frequency.py) |
| Figure 12(b) | [figure_12b.png](../results/official/figure_12b.png)、[figure_12b.csv](../results/official/figure_12b.csv) | 部分复现 | 四个仿真环境 probe 结果齐全；缺 Humanoid、DROID，频率 gap 未对齐 | [analyze_frequency.py](../code/h_jepa/scripts/analyze_frequency.py)、[analyze_probes.py](../code/h_jepa/scripts/analyze_probes.py) |
| Table 1 | [tables_1_9_cost_ladder.pdf](../results/official/tables_1_9_cost_ladder.pdf)、[table_1_ant.csv](../results/official/table_1_ant.csv) | 结论复现 | Ant 每个深度至少一个高层 cost 优于 native L1 | [eval_cost_ladder.sh](../code/h_jepa/scripts/eval_cost_ladder.sh)、[analyze_official_release.py](../code/h_jepa/scripts/analyze_official_release.py) |
| Table 3 | [table3_clip_counts.csv](../results/local/analysis_20261027/table3_clip_counts.csv) | **数值复现** | 4 环境 x 4 深度，共 16/16 项完全一致 | [analyze_local_reproduction.py](../scripts/historical/analysis_20261027/analyze_local_reproduction.py)、通用脚本 [analyze_protocol_tables.py](../code/h_jepa/scripts/analyze_protocol_tables.py) |
| Table 9 | [tables_1_9_cost_ladder.pdf](../results/official/tables_1_9_cost_ladder.pdf)、[table_9_other_envs.csv](../results/official/table_9_other_envs.csv) | 结论复现 | FourRoom 有高层 cost 收益，Cube/Push-T 无稳定收益 | [eval_cost_ladder.sh](../code/h_jepa/scripts/eval_cost_ladder.sh)、[analyze_official_release.py](../code/h_jepa/scripts/analyze_official_release.py) |
| Table 11 | [sigreg_sharding.csv](../results/local/analysis_20261027/sigreg_sharding.csv)、[sigreg_sharding.json](../results/local/analysis_20261027/sigreg_sharding.json) | **性质复现** | 全局 SIGReg 和梯度对 1/2/4 分片不变 | [verify_sigreg_sharding.py](../scripts/historical/analysis_20261027/verify_sigreg_sharding.py)、通用脚本 [analyze_protocol_tables.py](../code/h_jepa/scripts/analyze_protocol_tables.py) |

## 4. 逐项结果分析

### 4.1 Table 3：训练样本时序布局与每 epoch clip 数

Table 3 是本次最严格的数值复现。复现代码直接读取公开 HDF5 的 episode 长度，并使用仓库 loader 的递归时间跨度和 half-open transition 预留逻辑计算有效 clips；16 个计数全部等于论文。

| 环境 | LeWM/L1 | H-JEPA L2 | H-JEPA L3 | H-JEPA L4 |
|---|---:|---:|---:|---:|
| FourRoom | 952,708 | 915,298 | 840,478 | 690,838 |
| AntMaze | 9,575,000 | 9,200,000 | 8,450,000 | 6,950,000 |
| Push-T | 1,981,721 | 1,701,446 | 1,143,623 | 285,931 |
| Cube | 1,783,600 | 1,636,600 | 1,342,600 | 754,600 |

四种深度需要 4/7/13/25 个 L1 帧，对应论文展示的最短观测跨度 16/31/61/121 个环境步；loader 实际预留 20/35/65/125 个 transition span。这个区别解释了为什么简单使用观测端点跨度会多计 clips。

**对应代码：** 实际产物由 `local_env/analysis_20261027/analyze_local_reproduction.py::table3_counts` 生成；`h_jepa/scripts/analyze_protocol_tables.py::table3` 是可重复运行的通用入口。

### 4.2 Table 11：分布式 SIGReg 分片不变性

本地实际初始化了 CPU Gloo process group，并在固定高斯数据、64 个投影方向、17 个积分点和 float64 下比较 1/2/4 个分片。下表采用与论文一致的输入表示梯度范数 `‖∂TEP/∂Z‖`；先前列出的 `2.162668056770` 是额外单位线性层的参数梯度，不应与论文的输入梯度直接比较。

| 分片 | 输入梯度范数 `‖∂TEP/∂Z‖` | 全局 TEP | 无 ECF 同步的平均局部 TEP |
|---|---:|---:|---:|
| 1 × 256 | 0.280711511056 | 0.977725579145 | 0.977725579145 |
| 2 × 128 | 0.280711511056 | 0.977725579145 | 0.951421246410 |
| 4 × 64 | 0.280711511056 | 0.977725579145 | 0.910388509876 |

三种分片的全局 TEP 完全相同，重建输入梯度的最大绝对误差不超过 `7.81e-18`。论文 Table 11 的另一固定随机实例给出 `TEP=1.372619178543`、输入梯度范数 `0.349154178012`；论文没有公开对应的高斯矩阵、投影方向或 seed，因此绝对值不应相同，可严格声称的是**分片不变性被精确复现**。单位线性层参数梯度 `2.162668056770` 仍保留在原始结果中，作为 DDP 参数梯度平均的辅助验证。

**对应代码：** [verify_sigreg_sharding.py](../scripts/historical/analysis_20261027/verify_sigreg_sharding.py)；日志为 [sigreg_gloo_run.log](../results/local/analysis_20261027/sigreg_gloo_run.log)。

### 4.3 Figure 4：层级表征的选择性抽象

Figure 4 的核心定性结论得到清楚复现。公开四层模型的三 seed probe 结果如下：

| 环境/实体 | L1 NMSE | L2 NMSE | L3 NMSE | L4 NMSE | 结论 |
|---|---:|---:|---:|---:|---|
| Ant 身体状态 | 0.0606 | 0.4308 | 0.6853 | 0.7806 | 随层级快速丢失 |
| Ant 全局位置 | 0.000050 | 0.000099 | 0.000228 | 0.000516 | 始终接近可完全恢复 |
| FourRoom 干扰物位置 | 0.0539 | 0.1711 | 0.3487 | 0.5686 | 随层级逐步丢失 |
| FourRoom 智能体位置 | 0.000107 | 0.000155 | 0.000193 | 0.000222 | 始终保留 |

右侧 post-hoc decoder 图也与论文描述一致：Ant 腿部细节在 L2/L3 逐渐模糊，FourRoom 蓝色干扰物在高层消失，而红色智能体仍保留。论文没有提供 Figure 4 每个点的精确表格，因此这里应表述为**三 seed 协议下的结论复现**，而不是逐点数值核验。

### 4.4 Figure 5 与 Figure 12：频率分离和 selective abstraction

Probe 部分支持论文的表征结论，但频率 gap 没有数值对齐。当前 Figure 5 数据为：

| 环境 | 复现 frequency gap | 慢实体 `L2-L1` NMSE | 快实体 `L2-L1` NMSE | 判断 |
|---|---:|---:|---:|---|
| FourRoom | 2.294x | `+0.000020`（ego xy） | `+0.2141`（distractor） | 明显选择性抽象 |
| AntMaze | 6.266x | `+0.000046`（xy） | `+0.3696`（body） | 明显选择性抽象 |
| Cube | 2.729x | `+0.000154`（block pos） | `+0.0529`（arm joint vel） | 层间差异很小 |
| Push-T | 2.671x | `+0.00127`（block xy） | `-0.0559`（agent vel） | 未出现选择性丢失 |

这组结果复现了“Ant/FourRoom 的快实体更容易被 L2 丢弃，而慢位置保留；Cube/Push-T 两层接近”的 probe 结论。但它没有复现论文的频率横坐标：论文把 Ant 数据集 gap 写为 `8.3x`，并把 Push-T/Cube/DROID 描述为 `1.2-1.6x`；当前周期图聚合得到 Ant `6.266x`、Push-T `2.671x`、Cube `2.729x`。

Figure 12(a) 的本地 episode 也只复现视觉趋势：81 帧中身体状态振荡显著快于全局位置，但 27 维/13 维 gap 分别为 `6.28x/5.88x`，没有达到论文示例的 `8.3x`。造成差异的关键限制是论文没有公开 FFT 权重与跨 episode pooling 的完整公式；当前方法记录在 [frequency_method.json](../results/official/frequency_method.json)，不能把推断协议当成作者原始实现。

此外，Figure 5/12(b) 目前只覆盖 FourRoom、Ant、Cube 和 Push-T，论文中的 Humanoid 与 DROID 频率/probe 输入没有公开，因此是 released subset，而不是完整图。

### 4.5 Figure 6(bottom)：规划成功率与层级深度

完整复现实验使用模型 seed 42/43/44、每个模型 50 个固定任务，并在训练 seed 与 planner seed 配对后计算均值和标准误。下表给出“复现均值 / 论文均值”（单位：成功率 `%`）：

| 环境 | LeWM | H-JEPA L2 | H-JEPA L3 | H-JEPA L4 | HWM L2 | HWM L3 | HWM L4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AntMaze | 20.0 / 18.0 | 43.3 / 39.3 | 65.3 / 73.3 | 46.0 / 63.3 | 33.3 / 34.0 | 34.7 / 38.7 | 16.7 / 14.0 |
| FourRoom | 39.3 / 40.7 | 81.3 / 81.3 | 96.0 / 96.0 | 86.0 / 80.0 | 70.0 / 75.3 | 98.7 / 99.3 | 94.7 / 92.7 |
| Cube | 24.7 / 32.0 | 44.0 / 47.3 | 46.7 / 60.0 | 54.0 / 55.3 | 48.7 / 46.7 | 56.7 / 53.3 | 35.3 / 34.0 |
| Push-T | 40.0 / 40.0 | 42.0 / 45.3 | 13.3 / 17.3 | 0.0 / 0.7 | 53.3 / 42.0 | 9.3 / 12.0 | 0.0 / 0.0 |

可以支持的结论包括：

- FourRoom 最接近论文，H-JEPA L2/L3 均值分别精确复现为 `81.3%/96.0%`。
- Ant 仍在 L3 达到 H-JEPA 峰值，且明显优于同深度 HWM，但复现值 `65.3%` 低于论文 `73.3%`。
- Cube 的层级模型总体优于 LeWM，H-JEPA L4 `54.0%` 接近论文 `55.3%`，但 L3 偏低。
- Push-T 仍呈现 L2 尚可、L3/L4 迅速退化的论文趋势。

因此 Figure 6(bottom) 应汇报为**完整协议下的趋势/结论复现，不是全表数值复现**。规划结果的合并报告还说明，远端运行冻结了世界模型参数梯度但保留动作梯度，并使用原 float32 math-SDPA；基准动作最大差异约 `1.746e-4`，所以结果被标为 protocol-aligned 而非 bitwise identical。该差异可能影响优化轨迹，但现有证据不足以把全部数值偏差唯一归因于此。

### 4.6 Table 1 与 Table 9：高层 latent cost 是否改善平坦规划

合并展示结果见 [tables_1_9_cost_ladder.pdf](../results/official/tables_1_9_cost_ladder.pdf)（适合汇报投屏）及 [PNG 版本](../results/official/tables_1_9_cost_ladder.png)；底层数值分别保存在 `table_1_ant.csv` 和 `table_9_other_envs.csv`。

Table 1 的机制结论在 Ant 上得到复现。每个深度都至少存在一个 projected cost 优于 native L1：

| Ant 模型 | 复现 Native L1 | 复现最佳 projected cost | 论文 Native L1 | 论文最佳 projected cost |
|---|---:|---:|---:|---:|
| H-JEPA L2 | 20.7 | 30.7（L2） | 23.3 | 31.3（L2） |
| H-JEPA L3 | 15.3 | 27.3（L2） | 16.7 | 22.0（L3） |
| H-JEPA L4 | 5.3 | 25.3（L2） | 10.0 | 26.7（L4） |

最优层级不总与论文相同，但“高层表征能够重塑 cost，使 L1 平坦规划改善”的核心结论成立。完整 hierarchical planner 仍明显更强，说明 cost abstraction 不能单独解释全部层级规划收益。

Table 9 的跨环境结论也基本成立：

- **FourRoom：** L2 模型 `12.0 -> 42.0`、L3 模型 `42.0 -> 74.0`；L4 的最佳 projected cost `74.7` 略高于 native `73.3`。高层 cost 有收益。
- **Cube：** L2 projection 使 L2 模型 `24.0 -> 13.3`；L3/L4 的深层 projection 也普遍下降。没有稳定收益。
- **Push-T：** L2/L3 的差异处于较小幅度，L4 所有 cost 与 hierarchical planner 都接近 0。没有清晰、稳健收益。

由于各单元的复现成功率与论文仍有差异，Table 1/9 应称为**机制结论复现**。

### 4.7 Figure 7：AntMaze cost geometry

Figure 7 使用公开 `ant_hjepa_l3/seed42` 权重，对全部 502,500 个验证状态编码，并选取最接近 `(8, 8)` 的状态作为 anchor。图中 L1 的低 cost 区域主要集中在 anchor 周围，高层 cost 沿走廊形成更宽、更平滑的梯度，与论文“高层 cost 更符合绕墙最短路径结构”的定性结论一致。

该图必须标记为限定性复现：论文没有公开 anchor 坐标、状态采样集或 0-1 颜色归一化方式。本次使用每层距离除以本层最大值，完整参数记录在 [cost_geometry_manifest.json](../results/official/qualified_figure7/cost_geometry_manifest.json)。

### 4.8 Figure 9(b)：DROID 离线规划

Figure 9(b) 使用 16 个固定 DROID clip，三个训练 seed（1/1000/10000），每个训练 seed 再平均三个 planner seed（1/2/3）。结果如下：

| 模型 | 本次复现 Fréchet fidelity (%) | 论文 (%) | 差值 |
|---|---:|---:|---:|
| LeWM + IDM | 33.57 ± 1.52 | 33.57 ± 1.52 | 0.00 |
| HWM L2 | 34.74 ± 3.56 | 35.65 ± 2.22 | -0.91 |
| H-JEPA L2 | 35.86 ± 1.55 | 38.42 ± 0.94 | -2.56 |

三模型的均值排序与论文一致，但本次 `H-JEPA-HWM=1.12` 个百分点，远小于两个估计的不确定性，不能据此声称复现了论文中的“clear advantage”。此外，公开 release 不含无 IDM 的 slow-feature-collapse 权重，因此 Figure 9(b) 的第一根接近零的柱无法复现；当前图是明确标注的 three-model released subset。

### 4.9 Figure 10-style 本地训练曲线

[training_loss_curves.png](../results/local/analysis_20261027/training_loss_curves.png) 显示四个环境的 L1/L2 prediction、SIGReg、total loss 与学习率。曲线支持“训练没有发生常量表征式崩溃、总 loss 持续下降、L2 prediction/SIGReg 通常高于 L1”的健康训练判断。

但是这些日志是 `seed42_paper_v2_20261027` 的两层、单 seed、训练中快照，而论文 Figure 10 的仿真部分是四层模型、三训练 seed 的均值与范围，DROID 部分又是 100 epochs。因此该图只能作为本地训练诊断，不能列为 Figure 10 复现成果。早期 [AGGREGATED_RESULTS.md](../reports/historical/local_analysis/AGGREGATED_RESULTS.md) 中的单 seed、旧协议成功率同样不应与最终公开权重结果混用。

## 5. 复现代码位置与执行链路

### 5.1 总控、完整性与说明

| 功能 | 文件 |
|---|---|
| 论文图表与脚本总说明 | [h_jepa/scripts/ANALYSIS.md](../code/h_jepa/scripts/ANALYSIS.md) |
| 公开 release 可复现范围 | [REPRODUCTION_SCOPE.md](../reports/historical/REPRODUCTION_SCOPE.md) |
| 动态枚举 237 个评测单元 | [run_official_reproduction.py](../code/h_jepa/scripts/run_official_reproduction.py) |
| 等待完整覆盖、统一聚合和测试 | [finalize_official_reproduction.py](../code/h_jepa/scripts/finalize_official_reproduction.py) |
| 校验官方权重 manifest 与 hash | [verify_official_weights.py](../code/h_jepa/scripts/verify_official_weights.py) |
| 公共分析工具与 provenance 检查 | [analysis_utils.py](../code/h_jepa/scripts/analysis_utils.py) |

### 5.2 各图表执行链路

| 图表 | 评测/原始指标生成 | 聚合与绘图 | 最终输出 |
|---|---|---|---|
| Figure 4 | `main_probing_decoding_eval.py` 调用 frozen MLP probe 和 post-hoc decoder | `scripts/analyze_probes.py` | `figure_4_left/right.*`、`probe_*.csv` |
| Figure 5/12 | 数据 HDF5 + Figure 4/5 probe 指标 | `scripts/analyze_frequency.py` | `figure_5.*`、`figure_12a/b.*`、`frequency_method.json` |
| Figure 6 bottom | `scripts/eval_depth.sh` -> `eval.py` | `scripts/analyze_official_release.py` | `depth_raw/summary.csv`、`figure_6_bottom.*` |
| Figure 7 | 公开 L3 checkpoint + 验证集 | `scripts/analyze_cost_geometry.py` | `qualified_figure7/cost_geometry.csv`、`figure_7.*` |
| Figure 9(b) | `scripts/eval_droid.sh` -> `eval.py` | `scripts/analyze_official_release.py` | `droid_raw/summary.csv`、`figure_9b_released_subset.*` |
| Table 1/9 | `scripts/eval_cost_ladder.sh` -> `eval.py` | `scripts/analyze_official_release.py` | `tables_1_9_cost_ladder.pdf/.png`、`table_1_ant.csv`、`table_9_other_envs.csv` |
| Table 3 | 公开 HDF5 episode metadata | `local_env/.../analyze_local_reproduction.py` 或 `scripts/analyze_protocol_tables.py` | `table3_clip_counts.*` |
| Table 11 | CPU Gloo + 生产 SIGReg 路径 | `local_env/.../verify_sigreg_sharding.py` | `sigreg_sharding.*` |
| Figure 10-style | SPT `metrics.csv` 多 session 日志 | `local_env/.../analyze_local_reproduction.py` | `training_loss_curves.*`、`training_loss_points.csv` |

### 5.3 主要重跑命令

从 `/home/songxw/H-JEPA/h_jepa` 执行公开权重评测：

```bash
python scripts/run_official_reproduction.py \
  --experiments droid probe depth cost --gpus 0,1,2,3,4,6
```

单独执行官方 release 聚合：

```bash
python scripts/analyze_official_release.py
```

重算 probe 与 Figure 4/5/12：

```bash
python scripts/analyze_probes.py "$HJEPA_HOME/ckpts" \
  --model hjepa_l2 --model hjepa_l4 --require-seeds 3 \
  --output-dir "$HJEPA_HOME/analyse"

python scripts/analyze_frequency.py \
  --centroids-csv "$HJEPA_HOME/analyse/spectral_centroids.csv" \
  --probe-csv "$HJEPA_HOME/analyse/probe_summary.csv" \
  --output-dir "$HJEPA_HOME/analyse"
```

重算 Table 3/11 通用分析：

```bash
python scripts/analyze_protocol_tables.py \
  --output-dir "$HJEPA_HOME/analyse/protocol_tables"
```

本地 CPU 分析和日志刷新从仓库根目录执行：

```bash
/home/songxw/miniconda3/envs/hjepa/bin/python \
  local_env/analysis_20261027/analyze_local_reproduction.py

/home/songxw/miniconda3/envs/hjepa/bin/python \
  local_env/analysis_20261027/analyze_local_reproduction.py --refresh-logs
```

注意：`analyze_protocol_tables.py` 的 Table 11 是同一数学性质的新随机实例；本报告引用的实际多进程 Gloo 结果由 `verify_sigreg_sharding.py` 生成。

## 6. 公开 release 尚不能精确复现的论文结果

| 论文结果 | 当前不能精确复现的原因 |
|---|---|
| Figure 6 top | 未公开 compute-sweep 任务集、完整候选数网格和实测 planner FLOPs |
| Figure 8 / Table 10 | 论文使用单独训练的 `s2=3` 两层模型；公开仿真权重为 `s2=2` |
| Figure 9(a)、9(c) | 缺无 IDM/anti-collapse 权重和精确候选数/TFLOP sweep |
| Figure 10 | 官方权重不包含逐 step/epoch 训练日志 |
| Figure 11 | 未公开论文选择的任务 ID 和执行录像 |
| Figure 13、16 | 未公开 stagewise Ant/DROID checkpoint |
| Figure 14、15 | 未公开 Cube/RoboCasa outcome manifest 与 RoboCasa 资产 |
| Figure 17、18 / Table 12 | 未公开 anti-collapse、系数网格和 collapse-ladder checkpoint/log |

Figures 1-3 是方法示意图；Tables 2、4-8、13 主要是符号、超参数、协议或文献比较，不属于需要通过当前评测输出复现的性能结果。

## 7. 建议向老师汇报时使用的结论

可以概括为：

> 我使用固定版本的官方公开权重和公开数据完成了 H-JEPA 可公开复现部分的全覆盖评测，共 237 个实验单元。Table 3 的 16 个数据计数完全一致，Table 11 的 SIGReg 分片不变性在实际 Gloo 分布式运行中得到验证。Figure 4 的选择性抽象、Figure 6(bottom) 的层级规划趋势、Table 1/9 的高层 cost 机制以及 Figure 9(b) 的模型均值排序总体复现。但规划成功率并非每项都与论文相同；Figure 5/12 的频率 gap 由于作者未公开完整频谱聚合协议而未数值对齐；DROID 中 H-JEPA 相对 HWM 的优势也弱于论文。因此本次结果支持论文的主要定性结论，但不支持“所有数值精确复现”的表述。

汇报时建议重点展示：

1. `figure_4_left/right.png`：最直观地说明层级表征在保留位置的同时丢弃快变量。
2. `figure_6_bottom.png`：展示主要规划趋势，并在旁边列 Ant/FourRoom 的论文与复现值。
3. [tables_1_9_cost_ladder.pdf](../results/official/tables_1_9_cost_ladder.pdf)：合并展示 Table 1/9，说明“表征层级”和“时间分解”是两个不同收益来源。
4. `figure_9b_released_subset.png`：说明 DROID 排序复现，但主动标注差距与缺失的 no-IDM bar。
5. Table 3 和 Table 11：作为最严格、最可验证的复现成果。

## 8. 结论—证据映射与自查

| 主要结论 | 直接证据 | 状态 |
|---|---|---|
| 最终公开权重评测覆盖完整 | `combined_coverage.json`：84/84、108/108、27/27、18/18 | 支持 |
| Table 3 数值完全一致 | `table3_clip_counts.csv` 的 16 个 `matches_paper_table3=True` | 支持 |
| SIGReg 对等总 batch 分片不变 | `sigreg_sharding.json/csv` 中三组全局 TEP 与梯度一致 | 支持；绝对论文随机实例不可核验 |
| 高层丢弃快变量并保留位置 | `probe_summary.csv` 和 `figure_4_left/right` | 支持 |
| 层级规划总体改善长时程任务 | `depth_summary.csv` 与论文 `docs/RESULTS.md` 对照 | 支持主要趋势；数值部分偏离 |
| Ant/FourRoom 的高层 cost 改善平坦规划 | `table_1_ant.csv`、`table_9_other_envs.csv` | 支持 |
| DROID 中 H-JEPA 明显优于 HWM | `droid_summary.csv` | **证据不足**；均值更高但误差重叠 |
| 频率 gap 数值复现 | `figure_5.csv`、`frequency_method.json` 与论文对照 | **不支持** |
| Figure 10 完整复现 | 单 seed L2 训练日志与论文协议对照 | **不支持**，仅训练诊断 |

五维自查结果：

- **实验贡献：** 公开可支持范围已完整枚举，并明确区分官方权重、本地训练和旧协议结果。
- **表述清晰：** 每项结论均给出原论文图表号、结果路径和代码入口。
- **实验强度：** 主要公开评测使用论文要求的三训练 seed；Table 3/11 另有确定性/数值性质验证。
- **评估完整性：** 237 个可支持单元完整；未公开资产导致的缺项逐项列出。
- **方法可靠性：** provenance、hash、任务数和 seed 聚合顺序均有记录；频率协议与远端规划的限定条件没有隐藏。
