# H-JEPA 三项修复与四环境 L2 重训记录

记录时间：2026-10-27 09:42，北京时间。项目：`/home/songxw/H-JEPA`。

## 执行结果

按用户要求，原四个任务及所属旧调度进程已经停止；四个环境均于 **2026-10-27 09:32:00** 从头训练 H-JEPA L2，seed=42。新的两层世界模型共同随机初始化，没有加载旧 L1/L2/L3 权重。旧权重、日志、配置和已完成评估均保留在原目录。

| GPU | 新任务 | PID | 09:42 已完成更新 | 训练总更新预算 | 该卡显存快照 |
|---|---|---:|---:|---:|---:|
| 0 | Push-T H-JEPA L2 | 679338 | 1,000 | 119,628 | 6,701 MiB |
| 1 | Cube H-JEPA L2 | 679339 | 1,050 | 102,280 | 6,701 MiB |
| 2 | FourRoom H-JEPA L2 | 679340 | 1,000 | 50,050 | 7,401 MiB |
| 3 | Ant H-JEPA L2 | 679341 | 1,100 | 359,375 | 4,575 MiB |

四个 PID 与命令已核验；当前全部已记录训练指标均为有限值，日志没有 CUDA OOM 或异常退出。GPU 4 的修复验证任务已结束并释放显存。

此次启动的是四个 L2/seed42 任务。每个任务保留原有训练结束后的规划评估和探针/解码器回调；本次没有附加 L3/L4 队列。训练完成不等于评估已完成。

新输出目录统一为：

```text
/home/songxw/hjepa-data/ckpts/<env>/<env>_hjepa_l2/seed42_paper_v2_20261027/
```

`env` 为 `pusht`、`cube`、`fourroom`、`ant`。完整命令、PID、GPU 和日志路径见同目录 `restarted_l2_jobs.json`；最近状态见 `live_status.json`。

## 1. batch=128 的显存问题

旧实现直接保留整段视频全部图像帧的 ViT 中间激活，batch=128 会出现显存问题。新实现采用**图像编码器按 128 帧分块 + 激活检查点重计算**：

- 数据加载器仍一次提供 128 条训练 clip，`accumulate_grad_batches=1`。
- 仅对逐帧 ViT 编码部分分块并保存重计算所需输入。
- 所有分块得到的 CLS 特征先拼接，再经过原投影头；BatchNorm 在完整批次的全部帧（B×T）上计算一次统计。
- SIGReg 仍对真实 B=128 计算统计。它没有变成 microbatch32/accum4。
- 不改变模型层数、输入分辨率或 SIGReg 样本定义；代价是反向传播需要重算图像编码器，增加计算时间。

修改：`h_jepa/models/jepa.py`、`h_jepa/training_runtime.py` 和四个仿真环境 base config。旧 pickled 模型不含新增属性时仍走原路径；冻结/推理路径不启用检查点。

验证包括：完整批次与分块检查点的输出、损失、参数梯度及 BatchNorm 统计对比；真实数据训练与验证；多种真实网络结构的两次前向/反向/AdamW 更新。显存压力测试结果如下，数值为 PyTorch 峰值 allocated/reserved，不等同于上表整张卡的瞬时占用：

| 配置 | allocated GiB | reserved GiB | SIGReg 实际 B |
|---|---:|---:|---:|
| FourRoom L2 | 4.73 | 5.46 | 128 |
| Push-T L3 | 5.88 | 6.74 | 128 |
| Cube L3 | 5.88 | 6.74 | 128 |
| Ant L2 | 3.42 | 3.96 | 128 |
| Cube L4 | 7.35 | 8.75 | 128 |

这些压力测试使用合成输入，不是训练质量结果。正式四环境 L2 已使用真实数据运行约千步，当前约占 4.5–7.3 GiB/卡；这不代表所有后续配置的显存峰值均已测完。

## 2. 学习率调度与梯度累积

已将 warmup/cosine 的长度统一为**实际 optimizer 更新数**。安装的 SPT 会将累积因子保存在 `accumulate_grad_batches_`，本修复读取这个真实值，按实际训练批数、epoch 和 max_steps 计算调度长度。不能完整累积的尾批会明确报错，避免悄悄漏掉更新。

同时修正 SPT 手动 backward 路径未自动除以累积因子的行为；只有反传 loss 按累积因子缩放，展示的原损失保持原单位。当前正式任务累积因子为 1。

验证使用本机安装的 Lightning/SPT 真实训练循环：

| 累积因子 | optimizer 更新数 | scheduler 最终步数/预算 | 最后 LR |
|---|---:|---:|---:|
| 1 | 64 | 64/64 | 0 |
| 4 | 16 | 16/16 | 0 |

额外 FourRoom 真实数据 8 次更新 + 验证的完整 smoke run 正常结束：checkpoint global_step=8，scheduler=8/8，两组最终 LR 均为 0。

新 checkpoint 写入协议版本、实际 batch 和累积因子；遇到旧协议 checkpoint 或旧协议目录时拒绝静默续训/覆盖。旧训练历史不会因代码修复被重新解释为新协议。

## 3. 解码器学习率与 FourRoom 灰图

按照论文附录 D，将 probe 与 decoder 分成独立 AdamW 参数组：

- 全部 probe：LR=1e-3。
- FourRoom / Push-T / Cube decoder：LR=1e-4。
- Ant decoder：LR=1e-3。
- weight decay 仍为 1e-4；解码器架构、图像归一化、MSE 定义保持原实现。

新的正式后处理目录为 `final_probing_decoding_eval_v2`，不会覆盖旧后处理结果。四个运行中的任务在启动时已经加载此修复代码，文件哈希核对一致，因此其训练结束回调会采用独立学习率。

### 实际对照结果

使用同一个冻结的旧 FourRoom L2 世界模型，固定 seed=42、同一组 1,024 个 probing 训练帧和 128 个独立 probing_val 帧，重新初始化解码器；batch=128。两组对照只改变 decoder LR，训练数据、初始化、批次顺序和架构相同。

| 设置 | 更新次数 | 留出集 L1 MSE | 留出集 L2 MSE |
|---|---:|---:|---:|
| 已保存旧 60-epoch 解码器 | 原历史训练 | 2.25658 | 2.25657 |
| 新初始化，旧 LR=1e-3 | 1,000 | 2.25692 | 2.25692 |
| 新初始化，论文 LR=1e-4 | 1,000 | 0.18397 | 0.17703 |
| 同一 LR=1e-4 继续训练 | 3,000 | 0.02301 | 0.02126 |

已查看重建图：旧解码器和高 LR 对照停留在灰色重复纹理，论文 LR 的解码器能重建墙体和门口。在这个受控实验中，降低 LR 足以脱离灰图平台；高 LR 对照的 L1 query 梯度在第 1,000 步约为 1.7e-9，低 LR 为约 0.0243。不能据此宣称它是所有环境和随机种子下唯一可能的失败原因。

开始的 150 步测试不够长：低 LR 要到约 500 步才明显脱离平台，因此不能用那个短测试否定修复。

**边界：这是修复验证，不是论文完整 60-epoch 后处理复现。** 小测试图中红/蓝移动实体仍重建不清晰；MSE 下降和墙体恢复不意味着层级抽象或图 4 全部结论已复现。正式新世界模型完成后仍需完整 probing/decoding 评估。

证据文件：

- `decoder_long/decoder_validation_results.json`：论文 LR，3,000 步曲线。
- `decoder_lr_control/decoder_validation_results.json`：旧 LR，1,000 步对照。
- `decoder_long/decoder_validation.png`：输入/旧 L1/新 L1/新 L2 可视化。
- 两组诊断权重都标记为 `repair_validation_only`，与正式结果分开保存。

## 旧结果的保存

停止记录见 `stopped_legacy_jobs.json`。旧模型目录、已保存完整训练状态及已有规划结果保留原位，没有删除或覆盖。

已逐一 `torch.load` 核验旧 `lightning_resume/last.ckpt` 可读：

| 旧任务 | 已保存 epoch（0-based） | 已保存 global_step |
|---|---:|---:|
| Ant L2 | 1 | 118,500 |
| Push-T L3 | 0 | 5,500 |
| Cube L3 | 0 | 3,000 |
| FourRoom L3 | 6 | 45,962 |

停止时未写入 checkpoint 的内存更新不属于已保存结果。FourRoom L3 已完成的世界模型和规划评估保留；被中止的后处理不标记为完成。

旧 FourRoom L2/L3、Push-T L2、Cube L2 已有规划结果仍可作为明确标注旧协议的结果使用，不能与新协议结果合并统计。

## 时间影响与复现范围

09:42 早期实测速率约为 Push-T 1.8、Cube 1.9、FourRoom 1.8、Ant 2.0 更新/秒。按剩余更新数粗略外推，**仅训练**剩余约为：

| 任务 | 仅训练剩余时间粗估 |
|---|---:|
| FourRoom L2 | 7.6 小时 |
| Cube L2 | 14.8 小时 |
| Push-T L2 | 18.3 小时 |
| Ant L2 | 49.8 小时 |

这是训练开始约十分钟的估计，不包含 epoch 验证、保存、规划评估及后处理，吞吐也会变化。**Ant 本次从头训练很可能超过原两天预算，其他三个 L2 更有机会在预算内进入评估。** 当前尚无新协议最终成功率，不能保证修复后具体成功率上升。

## 验证与代码留档

- `h_jepa/tests/test_training_repairs.py`：8 项测试通过。
- `unit_tests.log`、`lightning_results.json`、`gpu_memory_results.json`、`checkpoint_validation.json` 保存验证证据。
- `before/` 保存修复前文件；`source_changes.patch` / `source_sha256.json` 是启动阶段快照。
- `final_source_changes.patch` 包含当前差异和新增训练辅助代码/测试；`final_source_sha256.json` 记录最终源代码与配置哈希。差异中包含先前已有 `.gitignore` 修改，不全是本次新增。
- 本次没有提交或上传 GitHub。虚拟环境、数据、权重和本报告均留在本机，`local_env/` 已被 Git 忽略。

本报告更新了先前 09:13 审计的运行状态；先前审计的历史结果仍保留。另更正该审计一处代码能力描述：`h_jepa/hjepa_utils.py` 已支持 `detach_pred_target`，需要补的是对应消融网格流水线，不是新增 target-detach 分支。
