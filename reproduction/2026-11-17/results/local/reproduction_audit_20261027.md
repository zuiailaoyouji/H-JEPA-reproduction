# H-JEPA 全论文复现审计与 48 小时计划

审计日期：2026-10-27，北京时间；最后运行状态快照为 **09:13**。

对象：论文 arXiv:2610.06805；本地 `/home/songxw/H-JEPA`，Git HEAD `97054ec`；现有数据与模型 `/home/songxw/hjepa-data`（指向 `/mnt/disk2/songxw/hjepa-data`）。

用户资源：稳定可用 4 张 A100 PCIe 40GB，最多 5 张；从本次排期起还有约 48 小时。以 2026-10-27 09:00 开始为例，截止时间为 **2026-10-29 09:00（北京时间）**；这只是排期基准，不是用户另行确认的提交时间。

本次只做审计、读取公开文件清单、写本地报告。未下载模型/大数据，未启动、停止或修改训练/评估任务，未修改项目训练代码，未提交或上传 GitHub。本报告和状态采集脚本位于已被 `.gitignore` 忽略的 `local_env/`。

## 1. 结论与复现口径

**当前项目可以直接运行论文的核心规划实验，但不是全论文的完整实验发行包。** 现成批量入口覆盖图 6 下半部分、表 1/9、图 9b 中的三个模型；世界模型训练、冻结模型后的探针和解码器拟合也有实现。频率分析、FLOPs/Pareto、代价空间分析、分阶段训练和多项 DROID 消融仍缺完整实验流水线。

**48 小时内从头训练并复现论文全部结果不可行。** 仅模拟环境深度比较就是 4 环境 × 7 模型 × 3 seeds = 84 次训练；DROID 核心又有 9 次 100-epoch 训练，附录还包含大量额外训练网格、RoboCasa 校准等。4 卡的总预算是 192 GPU·小时，5 卡全部可用才是 240 GPU·小时；第五卡不能计入保底。

建议分开报告三种证据，禁止混在同一 mean±SE 中：

1. **自训复现**：本地从头训练；保留实际 batch、梯度累积、调度器、训练种子等信息。目前属于资源受限、训练协议有差异的单 seed 复现。
2. **官方权重评估复现**：使用公开训练好的权重，在本机重跑论文的评估任务与指标；验证规划和评估结果，不等于复现了训练过程。
3. **探索性/部分复现**：单 seed、不同 stride、缺少基线或只画了部分环境的分析，明确标出与论文的差异。

## 2. 已有成果：可以交付什么

以下均为 **seed=42，50 个固定任务**，数值来自真实 `metrics.yaml`。右侧论文参照来自仓库 `docs/RESULTS.md`，是三训练 seed 的均值±标准误，不是单次运行的合格区间。

| 模型 | 自训/规划状态 | 实测成功率 | 论文三 seed 参照 | 后处理状态 |
|---|---|---:|---:|---|
| FourRoom H-JEPA L2 | 完成 7 epochs，规划完成 | **100%（50/50）** | 81.3±6.7% | 探针/解码器完成；解码图有明显问题 |
| FourRoom H-JEPA L3 | 完成 7 epochs，规划完成 | **92%（46/50）** | 96.0±1.1% | 08:48 起运行探针/解码器；审计时未落最终文件 |
| Push-T H-JEPA L2 | 完成 9 epochs，规划完成 | **30%（15/50）** | 45.3±1.3% | 探针/解码器完成 |
| Cube H-JEPA L2 | 完成 8 epochs，规划完成 | **30%（15/50）** | 47.3±2.7% | 探针/解码器完成 |
| Ant H-JEPA L2 | 训练约 1.62/5 epochs | 暂无 | 39.3±3.7% | 尚无最终规划与探针结果 |
| Push-T H-JEPA L3 | 第 1/9 epoch 中 | 暂无 | 17.3±3.5% | 未开始 |
| Cube H-JEPA L3 | 第 1/8 epoch 中 | 暂无 | 60.0±5.0% | 未开始 |

这些已覆盖图 6 下半部分中的 **4/84 个独立模型-seed 评估条目**，而不是完整复现了 4 个图表或全部实验。当前没有 LeWM/HWM、四层模型、seed43/44、Ant L3、DROID 训练成果。

可立即整理为交付物：上述四项规划成功率、逐任务成功/失败、现有训练损失曲线、三个模型的探针 NMSE 表，以及配置/数据/模型来源说明。FourRoom L2→L3 的单 seed 成功率反而下降，不能由此声称已经重现论文的总体深度结论。

不需要重训世界模型、但还需要补跑的实验：

- FourRoom L2：native L1、project-to-L2，共 **2** 次规划评估。
- FourRoom L3：native L1、project-to-L2、project-to-L3，共 **3** 次。
- Push-T L2：native L1、project-to-L2，共 **2** 次。
- Cube L2：native L1、project-to-L2，共 **2** 次。
- 合计 **9 次**评估，可组成表 9 的单 seed 子集；当前尚未运行这些消融。

现有探针并不自动意味着图 4 已复现。例如 FourRoom L2 的 distractor NMSE 为 L1=0.9641、L2=0.9695，L1 已经接近不含预测信息，不能展示论文所需的清晰逐层选择性抽象；论文图 4 本身还要求四层模型及三 seed。

## 3. 全论文实验逐项映射

“有基础实现”与“有完整论文实验脚本”不同。这里按本地代码实际可执行程度分类。

| 论文结果 | 当前代码能做什么 | 仍需要的复现代码/资源 | 两天优先级 |
|---|---|---|---|
| 图 1 | 复用层级规划、解码器和计算比较组件 | teaser 拼图/解码规划序列/Pareto 绘图；不是一组完全独立的新实验 | 低；由相关实验派生 |
| 图 2、3 | 方法示意图 | 不需要额外训练实验；示意图重绘可选 | 不占 GPU 预算 |
| **图 4**：逐层抽象、NMSE、重建 | `main_probing_decoding_eval.py` 支持四个仿真环境的任意已训练深度 | FourRoom/Ant 四层权重与三 seed；修正解码学习率；跨 seed 聚合及论文版面图 | 中；先输出已有模型的明确标注子集 |
| **图 5、图 12a/b**：频率差与抽象相关性 | 有四个环境状态标签和探针组件；有 Humanoid 环境生成基础 | 谱质心、按维标准化、episode 聚合、fast/slow ratio、NMSE 差与误差条；Humanoid、DROID 的对应数据/探针入口缺失 | 四环境频率统计可做；六环境完整图延期 |
| **图 6 下半部分**：深度/基线成功率 | `scripts/train_all.sh`、`scripts/eval_depth.sh`；28 组仿真配置 | 所有模型/seed 的权重；汇总/绘图；当前只有 4 条本地结果 | **最高**：优先关键基线、三 seed |
| **图 6 上半部分**：成功率–计算量 | 规划候选数可配置，保存 `planning_compute.json` | 独立 sweep 任务集、候选数网格、真实 FLOPs 计数与反传统计、统一每 episode 计量、Pareto/置信区间代码 | 中低；先单环境小范围 |
| **表 1、表 9**：只改变代价空间 | `scripts/eval_cost_ladder.sh` 及 `*_lN_project.yaml` | 权重、运行消融与聚合；保持 L1 动力学和其余规划设置固定 | **高**：先 FourRoom，再 Cube/Ant |
| **图 7**：Ant 代价几何热图 | 编码器/层级模块可调用 | 固定其他状态、扫描 xy 位置、渲染/编码、distance-to-anchor、统一色标/绘图 | 中；需要 Ant L3 权重 |
| **图 8、表 10**：目标代价/子目标单调性 | 可编码专家轨迹、可构造层级规划代价 | 沿轨迹计算 goal/subgoal cost、−Spearman 统计、固定子目标取法、跨 seed 聚合；图 8 指定 **stride s2=3**，现有仿真发布配置为 2 | 中低；先做当前模型的探索性版本，不能冒充原图 |
| **图 9a/b**：DROID 场景与离线保真度 | `scripts/eval_droid.sh`、`droid_eval.py`、`traj_metrics.py`；LeWM+IDM/HWM/H-JEPA | 9 个官方权重和 16 clips；核对 S=(16,4) 与论文 (16,16) 差异；图中 no-IDM LeWM 不在这 9 个权重内 | 若接受官方权重则高；完整图仍缺 no-IDM |
| **图 9c**：DROID 计算 Pareto | 候选数参数和离线评估基本实现已有 | 候选数 sweep/FLOPs/Pareto；JEPA-WM、V-JEPA2-AC 的冻结编码器基线适配、1fps/15 clips 协议 | 低；完整图超过这两天的稳妥范围 |
| **表 3**：各深度有效训练 clips | 现有 dataset 和 episode 长度可读取 | CPU 统计各深度最短 clip 长度与有效起点数的脚本、核对表格 | 高；无需训练 |
| **图 10**：训练损失曲线 | SPT `metrics.csv` 有预测、SIGReg、总损失 | 曲线提取/对齐/聚合；论文是四层仿真模型及 DROID 100 epochs 的三 seed，现有 L2/L3 只能替代性展示 | 已有日志整理为高；完整原图低 |
| **图 11**：成功执行序列 | 仿真规划与环境可运行 | 执行轨迹帧采集、固定成功 episode 选择规则、拼图；当前 metrics/compute 文件不是视频 | 中；只补少量可视化任务 |
| **图 13 / 附录 G**：数据专门化 | 有 Ant explore/stitch/union 训练数据基础 | **stagewise lower-level freeze 训练入口**、数据路由、3×3 分配、端到端对照、FLOPs 匹配；当前冻结功能用于探针而非阶段训练 | 低；需要多次新训练 |
| **图 14**：Cube Fréchet 与抓取校准 | Fréchet 核心函数、Cube 环境已有 | 轨迹动作记录/坐标语义、抓取 outcome、9 配置×3 planner seeds×50 episodes、相关性/散点图 | 低；现有 success YAML 不足以计算 |
| **图 15**：RoboCasa CloseDrawer 校准 | 无对应完整流水线 | RoboCasa 环境/数据/模型、graded drawer closure、99 配置×32 episodes、分层相关性分析 | 很低；两天延期 |
| **图 16**：DROID 解码规划 | 规划可保存 `actions.pt`，已有通用解码器 | 论文用 stagewise seed1 模型、DROID decoder 训练/归一化适配、16 clips 分层计划解码与拼图；主结果端到端权重不能直接视为同一实验 | 低 |
| **图 17**：SIGReg×IDM×target detach | SIGReg 与 IDM loss 和系数可配 | target-detach 分支、16 cells 的训练/恢复/调度、effective rank 统计、3 planner seeds 的网格分析 | 低；现有权重不能补齐 |
| **图 18**：两级系数网格 | 有相关 loss 与模型组件 | L1 16 cells×3 seeds；stagewise L2 9 cells×3 seeds；合计 75 个 cell-seed 训练条目及网格聚合 | 很低；两天延期 |
| **表 11**：分布式 SIGReg 不变性 | `loss.py` 已有跨 rank ECF 均值和全局 n | 固定随机投影、float64、1×256/2×128/4×64 的值/梯度误差数值验证；模拟分片只能作预检，真实 distributed 验证应单列 | 高；通常 CPU/Gloo 即可，无需模型训练 |
| **图 19**：episode 内/间潜变量方差 | Cube 模型可直接抽取 latents，DROID 权重可取得 | 按 episode 分组的 within/between 分解、χ 指标、跨 seed 聚合 | 中；Cube 子集成本低 |
| **表 12**：no-IDM 塌缩、时间排斥 | 可关闭 IDM，已有 SIGReg | 时间并入样本轴的 SIGReg grouping、time repulsion loss、action-sensitivity 指标、跨 epoch 评估/早停/λ ladder；大量 100-epoch 训练 | 很低；两天延期 |
| 表 2/4/5/6/7/8/13；附录理论推导 | 符号、配置、协议、相关方法或理论 | 不应计为独立需要训练的新结果；应核对配置与实现 | 文档工作 |

全量成本量级：`eval_depth.sh` 是 **84** 次规划评估，`eval_cost_ladder.sh` 是 **108** 次（4×3×(2+3+4)），DROID 主脚本是 **27** 次（3 models×3 train seeds×3 planner seeds），后者每次 16 clips。不能把“下载全部权重”误当成“已经完成全部评估”。

## 4. 当前必须披露/修正的协议问题

### 4.1 梯度累积没有恢复真实 batch=128 的 SIGReg 训练

本地四个环境的 base config 从 `loader.batch_size: 128` 改为 32，并设置 `accumulate_grad_batches: 4`。普通逐样本损失的梯度累积不意味着 SIGReg 和 BatchNorm 统计等价。

`h_jepa/loss.py:28` 每次 forward 对实际 B 做 ECF 均值，再计算非线性的平方偏差并乘 n；每次只看到 32 个样本。`h_jepa/models/module.py` 的投影头还用了 BatchNorm1d。故当前模型应该标明“microbatch32/accum4”，不能只写 effective batch128 并宣称原始协议一致。

后续新训练如要严格对齐，应先验证真实全局 batch128 的实现与显存：单卡可容纳时不累积；多卡方案要核对全局 SIGReg 随机投影/统计、BN 行为和实际 optimizer step 数，不能只把 accumulate 改成 DDP 就认为自动等价。

### 4.2 已证实：累积训练的学习率调度长度错配

从真实 `lightning_resume/last.ckpt` 读到：

| 模型 | scheduler.max_steps | 实际 scheduler.last_epoch / optimizer global_step | 最终 L1 学习率 |
|---|---:|---:|---:|
| FourRoom L2 | 200221 | 50050 | 4.30947e-5 |
| FourRoom L3 | 183848 | 45962 | 4.30932e-5 |
| Cube L2 | 409144 | 102280 | 4.30941e-5 |

这些模型已经结束训练，但 cosine 仅前进到约 **25%**；末尾 LR 仍约为峰值 5e-5 的 **86.2%**。这不是仅凭日志推测。

原因链条：安装的 `stable_pretraining/utils/lightning_patch.py:61` 将 accumulation 保存到 `accumulate_grad_batches_`，把 Lightning 原属性复位为 1；默认 scheduler 用 `estimated_stepping_batches` 配置完整长度，而手动 optimizer/scheduler 每 4 个 microbatches 才 step。新训练前必须让调度器采用实际 optimizer 更新次数，并核验 warmup 与最后 LR。现有 checkpoint 不能通过“改一下配置”追溯变成原协议训练；也不能在进行中的同一 run 中无说明地改调度后继续汇总。

### 4.3 探针/解码器学习率与论文不同；FourRoom 解码已明显失败

论文附录 D 要求 probes=1e-3、decoders=1e-4（Ant 的 decoder 例外为 1e-3）。当前 `final_probing_decoding_eval.py:196` 给 probe 和 decoder 参数组共同使用 `cfg.optimizer.lr`，四环境 probing yaml 都是 0.001，导致 FourRoom/Push-T/Cube decoder 与论文不一致。

已实际查看 FourRoom L2 `decodings/eval_level1_embed_unroll.png`：解码结果几乎都是相同的灰色棋盘，无法重建场景；L1/L2 decoder MSE≈2.2576。应先给解码器独立 LR，再检查初始化、训练曲线、归一化及输出；不能只凭文件已生成就宣称复现成功。学习率差异是已确认的问题，但尚未证明它是解码失败的唯一原因。

这类后处理修复通常 **不需要重训世界模型**。修复后的 probe/decoder 输出应另存目录，避免覆盖已经报告的结果。

### 4.4 规划任务分块/随机种子

当前四项结果都记录 `chunk_size: 1`。`planning_eval.py:455` 默认逐任务分块；chunk 内 planner generator 用原 seed + task start index 重新设种子。它与把 50 个任务放在一个 World 中、只设一次 seed 的随机数路径不等价。

50 个任务内容一致，但不能保证逐 seed 数值严格等同论文原运行。比较自身模型和官方模型时固定并报告分块方式；若目标是原始 seed 级数值对照，需要确认官方运行使用的分块协议。不要混用不同分块方式的结果。

### 4.5 DROID 和论文目标值还有两处待核对

- `config/eval/droid_l2.yaml` 的候选数是 **(16,4)**；论文表 8 是 **(16,16)**。先复现发行脚本的配置，再单列 paper-setting 配置核对；不能悄悄更改后称与发行结果同一协议。
- `docs/RESULTS.md` 与 PDF 的部分层级参考数值不同。例如 Ant H-JEPA4 的层级参考，仓库图 6 对照为 **67.3±2.4%**，PDF 表 1 为 **63.3±7.7%**；FourRoom4 88±2.3 vs 表9 80±4；Cube4 62.7±2.4 vs 表9 55.3±4.4。报告应给明各自来源，不能只选更接近本地结果的一组。
- DROID 发行脚本采用 train seeds 1/1000/10000、planner seeds 1/2/3，按训练种子先平均 planner 再求跨训练种子的 mean±SE。附录其他分析有不同 planner seeds/聚合表述；每张图应使用自己的协议，不能拿单次或 median 替代脚本的 headline mean。

## 5. 官方预训练权重具体能补什么

已通过 Hugging Face API 实查 `jepa-world-models/h-jepa`，模型 revision `08fc8a21208671a87741f2960aeba9b5db4420e0`。这只是读取清单，尚未下载权重。

| 权重/数据 | 公开规模 | 能补齐的实验 |
|---|---:|---|
| 四模拟环境 LeWM、H-JEPA2/3/4、HWM2/3/4 × seeds42/43/44 | **84 模型，约 9.71 GB**（十进制，含 normalizer） | 图6下半部分所缺的基线、深度和随机种子；表1/9代价消融 |
| DROID LeWM+IDM、HWM2、H-JEPA2 × seeds1/1000/10000 | **9 模型，约 2.73 GB** | 图9b三个模型的离线规划 fidelity；部分后续分析 |
| `droid_raw_eval16.tar` | **252,037,120 bytes（约252 MB）** | 16条DROID评估 clips；不用91GB训练集即可评估 |

官方模型能进一步作为图4/7/19等分析的输入，但 probes/decoders 仍须拟合或补分析脚本；不自动提供出图结果。图8的stride3模型、图13阶段式数据组合、图16的阶段式模型、图17/18/表12训练消融、RoboCasa、外部冻结编码器基线 **不在这批权重覆盖之内**。

当前 DROID 本地目录存在但为空，不能当作已下载完成；四个模拟环境的训练/探针 HDF5 和50-task评估文件已有，之前校验记录在 `local_env/dataset-audit-20261026.json` 及 `eval-task-validation-*.json`。

存储建议：官方模型放单独根目录，例如项目忽略区 `local_env/official_checkpoints/`，每个模型连同配套 normalizer 下载，输出也单独存。**不能把官方 seed42 文件下载到当前自训 seed42 目录覆盖它。** 数据环境、权重和输出都不上传 GitHub。

## 6. 现有任务耗时与 GPU 安排

GPU 与任务经 PID/UUID 核对，以下是当时的物理卡号：

| GPU | 正在运行 | 近期速度推算剩余纯训练 | 预计纯训练结束（北京时间） | 后续 |
|---|---|---:|---|---|
| 0 | Push-T H-JEPA L3 | 点估计约17h；预留17–21h | 10月28日约02:00–06:00 | 规划评估+探针/解码另计 |
| 1 | Cube H-JEPA L3 | 点估计约19h；预留19–24h | 10月28日约04:00–09:00 | 规划评估+探针/解码另计 |
| 2 | FourRoom H-JEPA L3 后处理 | 训练/规划已完成 | 08:48开始后处理，暂不报精确剩余 | 参考L2后处理约5.6h，L3应预留约6–9h总时长 |
| 3 | Ant H-JEPA L2 | 点估计约35h；预留35–40h | 10月28日约20:30至10月29日01:00 | 规划/探针尚需额外时间，可能逼近截止 |

点估计用最近 checkpoint 日志的 microbatch 吞吐与数据 clips 数计算，不含后续 validation、I/O争用、失败重试和自动评估；不是承诺结束时间。可用 `collect_reproduction_status.py` 刷新。第五张卡仅在确实可用时加入评估队列，不依赖某张目前空闲卡长期可用。

本机已测到的50-task评估时长：FourRoom L2 **65.3min**、L3 **71.1min**；Push-T L2 **45.8min**；Cube L2 **10.9min**。它们只代表对应层级规划配置，不能直接外推 flat/projected/Ant。尤其 FourRoom flat horizon=23、hierarchical L1 horizon=2，代价消融可能明显更慢。

3项在训任务继续跑完约占 **71–85 GPU·小时**；FourRoom现有后处理以及后续三个模型的评估/探针还要另占资源。4卡192GPU·小时里，最后6–8小时建议留出24–32GPU·小时的收尾窗口。剩余预算优先投评估，不建议再排新的DROID、RoboCasa或成套四层/多seed长训练。

## 7. 推荐的 48 小时执行次序

这里的阶段预算是**资源上限/取舍规则**，不是未经测量的任务耗时保证。是否采用官方权重路线仍由用户选择；本次没有执行这些队列。

### 0–3小时：固定证据、解决复现口径

- 固定代码/配置/数据版本，保存现有四项结果、学习率诊断和解码样例。
- 补只读汇总器与图表代码；先实现 Table3 的clip统计、现有loss/NMSE整理、Table11的数值验证。
- 在独立输出目录修 decoder 的参数组学习率，做小规模可视化验证；不覆盖进行中的训练。
- 若接受官方权重，先取关键基线及配套 normalizer。对 Ant、flat/projected、DROID 各做少量独立试跑测吞吐，**试跑目录与50-task最终评估分开**。

### 3–24小时：主结果和最有辨别力的对照

保持当前4个任务的计划不变时，GPU2应最先从FourRoom后处理释放；如第五卡可用，让它提前做评估。

1. **优先模型基线**：FourRoom/Ant 的 LeWM、HWM3、H-JEPA3，各3训练seeds，共18个官方评估条目；它们能区分平面模型、仅时间层级和学习抽象。Ant先测速再决定批量规模。
2. **优先机制消融**：自己训练的 FourRoom L2/L3 native/projected 共5次，先选L2两次作为最小完整对子；Cube L2两次次之。必须成对完成，不只挑表现好的投影列。
3. 有预算再加 FourRoom/Ant 官方 H-JEPA2（三seeds共6次），得到更完整的深度对照。

### 24–40小时：补操控环境、DROID，按测得速度裁剪

- Push-T/Cube L3完成后，先收取其自动评估结果；不要只等训练权重而漏掉最终指标。
- 官方 Push-T/Cube 的 LeWM、HWM2、H-JEPA2，各3seeds，共18条评估；有余力再加各自H-JEPA3共6条。前三步扩展后总目标是 **48/84** 条有针对性的官方仿真评估，未完成部分明确列缺口。
- DROID先1个模型×1个planner seed完成16clips测时；若估计可在预算内，再扩展9模型×3planner seeds共27jobs。逐训练seed先平均planner结果，再求3训练seed的mean±SE。该指标是离线路径保真度，**不是任务成功率**。
- 已有latent分析优先图7或Cube图19子集；完整FLOPs sweep、六环境频率图、RoboCasa和训练消融不挤占主结果预算。

### 40–48小时：收尾与提交

- 不再开长训练/大sweep；保证任务、种子、配置、输出文件一一对应，聚合所有完整运行。
- 画单seed自训表、独立的官方权重mean±SE表、cost-ladder对子、可用的NMSE/曲线/执行样例。
- 报告各图“完整/部分/未复现”和协议差异；失败结果照实保留，不能用官方seed42替换自己的seed42后混称自训。
- Ant L2若仍未结束，报告最新checkpoint与训练进度；优先保留规划评估时间。可选的高成本探针是否延期，需要在明确任务范围后决定，不能默默把未完成后处理写成完成。

### 第五卡的用途

第五卡最适合官方权重评估/代价消融的小作业队列，可增加最多48GPU·小时；其时段不稳定时可用现有chunk恢复功能。不要为它开启另一个可能超过截止的新Ant/DROID训练。

### 若只接受自己从头训练

优先整理现有4个结果并完成正在跑的3个模型，做FourRoom/Cube已有权重的代价消融。之后的新训练应先修正scheduler和batch协议，再在一个环境补LeWM/HWM的最小对照，而非同时加多个环境/深度。两天交付应明确称“部分、单seed复现”，不能承诺完整的三seed结论。

## 8. 需要补的代码：建议按此顺序实施

| 顺序 | 建议新增/修改模块（本次尚未实施） | 验收标准 |
|---|---|---|
| P0 | 调度器总optimizer步数；probe/decoder独立LR | accumulation1/4下scheduler都完整结束；解码器能恢复非恒定场景；旧结果不覆盖 |
| P0 | 显式实验manifest、`aggregate_results.py` | 模型来源/训练seed/planner seed/任务文件/配置/实际成功数齐全；缺失不当作0；mean±SE按训练seed而非50tasks计算 |
| P1 | 选择性job runner和安全断点 | ENVS/MODELS/SEEDS可过滤，缺权重先预检，completed跳过；主脚本目前缺MODELS过滤，直接跑会遇缺模型退出 |
| P1 | `plot_training_and_probes.py`、`count_valid_clips.py` | 现有日志/指标可追溯，Table3统计来源明确 |
| P1 | `check_sigreg_sharding.py` | 固定投影，value/gradient双核验；浮点精度/分片方式明确 |
| P1 | `cost_ladder_manifest.py` / 聚合工具 | 同一模型各列只变cost空间，规划设置和任务一致 |
| P2 | `frequency_analysis.py`、`latent_geometry.py`、`trajectory_cost_analysis.py` | 频率/位置热图/−Spearman定义与论文一致；不合协议的模型明确标注 |
| P2 | `planning_compute_sweep.py` / FLOPs profiler | 覆盖实际规划前向和反向；独立sweep任务；不是用显卡峰值×时间假算FLOPs |
| P2 | `latent_variance_analysis.py`、rollout帧记录/拼图 | episode内外方差及执行视频可复核 |
| P3 | stagewise训练、target-detach/time-repulsion分支、DROID网格runner | 严格冻结低层、正确优化器/数据路由、每个cell可追踪 |
| P3 | RoboCasa与JEPA-WM/V-JEPA2-AC适配、Cube/DROID校准 | 新环境/动作语义/数据率/指标定义逐项验证 |

## 9. 审计产物和证据入口

- `local_env/reproduction_status_20261027.json`：全部7个run的快照、checkpoint、规划/探针指标、训练ETA及估计方法。
- `local_env/reproduction_results_20261027.csv`：已完成4项规划评估的可读结果表。
- `local_env/collect_reproduction_status.py`：只读采集任务状态，写上述小报告；不启动训练。
- 原始规划证据：`ckpts/<env>/<env>_hjepa_lN/seed42/planning_eval/epoch_*/metrics.yaml`。
- 原始探针证据：同run下 `final_probing_decoding_eval/metrics.yaml` 和 `decodings/*.png`。
- 调度器证据：同run下 `lightning_resume/last.ckpt` 的 `lr_schedulers`。
- 代码入口：`h_jepa/main_hjepa.py`、`planning_eval.py`、`final_probing_decoding_eval.py`、`loss.py`、`scripts/eval_*.sh`、`config/eval/*.yaml`。
- 论文：arXiv:2610.06805，已读取全文（本地暂存 `/tmp/hjepa-2610.06805.pdf` / `.txt`）；目标值另见 `docs/RESULTS.md`。
- 官方模型清单：Hugging Face `jepa-world-models/h-jepa`，上述固定revision；每个发布目录均有模型与normalizer。

刷新快照（CPU读文件，不接触GPU任务）：

```bash
cd /home/songxw/H-JEPA
/home/songxw/miniconda3/envs/hjepa/bin/python local_env/collect_reproduction_status.py
```

**推荐交付目标：先形成可核验的主结论与关键消融，再扩展图表覆盖；把“从头训练复现”和“官方权重评估复现”始终分开。当前全部结果尚未复现，完整缺口如本报告逐项列明。**
