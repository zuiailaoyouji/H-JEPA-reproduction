# H-JEPA 复现归档（2026-11-17）

本目录将已有复现脚本、真实实验结果和中文报告统一保存。实验完成时间为 2026-10-29；2026-11-17 进行了目录整理和完整性复核，没有重新运行完整 GPU 评测。

**完整报告：** [reproduction_report.zh-CN.md](reports/reproduction_report.zh-CN.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `code/` | 实际使用的完整代码、配置和本地修复，保留 MIT 许可 |
| `scripts/reproduce.sh` | 校验、任务枚举、聚合、评测、测试的统一入口 |
| `scripts/restore_assets.py` | 校验并恢复大文件归档 |
| `scripts/historical/` | 当时使用的本地分析、训练诊断和修复辅助脚本；含历史机器路径，作为来源记录保存 |
| `environment/` | 依赖快照、可迁移依赖清单和安装入口 |
| `results/official/` | 最终完整聚合结果、CSV、图表和覆盖率 |
| `results/local/` | 单 seed L2 训练诊断、数据计数、SIGReg、频率分析；与公开权重结果分开 |
| `reports/` | 中文复现报告、历史说明和原汇报 PPT |
| `provenance/` | 源码差异、固定资产版本、逐文件哈希、远端运行来源和取消记录 |

## 结论及范围

| 实验组 | 完成单元 |
|---|---:|
| 层数对比 | 84 / 84 |
| cost-ladder | 108 / 108 |
| DROID 规划 | 27 / 27 |
| probe / decoder | 18 / 18 |
| 合计 | 237 / 237 |

上述完整性指固定公开资产能够支持的评测范围。Table 3 的 16 个 clip 计数一致，SIGReg 分片不变性得到验证。规划与表征的主要趋势得到支持，但规划成功率并非全部数值对齐；频率 gap 未数值对齐；DROID 的均值排序一致但误差区间重叠。单 seed L2 训练曲线只用于诊断，不是论文 Figure 10 的完整复现。详见报告中的逐项证据和限定条件。

- 代码基点：`97054ecaff83e13839195ec4b92d74c08a0127c5`，实际修复见 `code/` 和 `provenance/local_source_changes.patch`。
- 官方模型：`jepa-world-models/h-jepa@08fc8a21208671a87741f2960aeba9b5db4420e0`，189 个文件、93 个 checkpoint；本地保存其逐文件哈希清单。
- DROID 数据：`jepa-world-models/h-jepa@cc05cb8d2aaf9aec0d5364d073c6369ff760b60b`，16 个固定 clip。
- 仿真规划：每单元 50 个任务，模型 seed 42/43/44。
- DROID：训练 seed 1/1000/10000，与 planner seed 1/2/3 交叉。

## 大文件与恢复

逐任务原始输出、固定评测任务张量、Figure 7 状态数据和独有本地训练/decoder 权重放在同仓库的 [reproduction-2026-11-17 Release](https://github.com/zuiailaoyouji/H-JEPA-reproduction/releases/tag/reproduction-2026-11-17) 附件中。`provenance/large_assets.json` 记录文件名、字节数和 SHA-256。checkpoint 归档分卷以适应附件大小限制；恢复脚本会先验证每卷和合并流的哈希。

| 附件 | 用途 |
|---|---|
| `cost_geometry.csv.gz` | Figure 7 的 502,500 状态 CSV，无损压缩 |
| `raw_results.tar.zst` | 当前数据树中的全部非官方资产输出：逐任务结果、完整 DROID/probe 详情、日志、动作张量 |
| `fixed_eval_tasks.tar.zst` | 实际使用的固定仿真任务，恢复到 `code/h_jepa/assets/` |
| `local_checkpoints.tar.zst.part*` | 31 个独有本地训练、decoder 和验证 checkpoint；保留恢复训练所需状态 |
| `remote_planning_results_20261029.zip` | 原始远端结果归档，含运行日志、性能和取消记录；此归档的 probe 曾取消，最终结果采用本机已完成的 probe |

公开原始数据集、官方预训练权重、可重建环境和旧迁移备份不重复上传。下载/生成方式及固定版本见 [资产恢复说明](environment/ASSETS.md)。

```bash
# 从仓库根目录执行，Python 3.11。
bash reproduction/2026-11-17/scripts/reproduce.sh verify

# 下载所有 Release 附件（含 21 个 checkpoint 分卷）；需要 gh 和 zstd。
gh release download reproduction-2026-11-17 \
  --repo zuiailaoyouji/H-JEPA-reproduction --dir /path/to/downloaded-release

# 校验并恢复附件。
python3 reproduction/2026-11-17/scripts/restore_assets.py \
  --assets-dir /path/to/downloaded-release --data-dir /path/to/hjepa-data

# 安装到自己新建的 Python 3.11 环境。
bash reproduction/2026-11-17/environment/setup.sh

# 枚举完整 237 个任务，不启动 GPU 工作。
bash reproduction/2026-11-17/scripts/reproduce.sh dry-run

# 从已归档的原始结果重新聚合规划数据；输出至指定数据目录。
HJEPA_HOME=/path/to/hjepa-data \
  bash reproduction/2026-11-17/scripts/reproduce.sh aggregate

# 下载公开数据及官方权重后才可重跑评测。明确指定可用 GPU。
HJEPA_HOME=/path/to/hjepa-data \
  bash reproduction/2026-11-17/scripts/reproduce.sh evaluate --gpus 0,1
```

恢复附件后，`results/official/qualified_figure7/cost_geometry.csv.gz` 是无损压缩的 502,500 状态数据，读取时可用 pandas 的自动解压，或以 `gzip -dk` 恢复。历史报告和日志中的绝对路径只表示原始来源；新入口通过 `HJEPA_HOME` 指定数据位置。

## 完整性

`provenance/MANIFEST.sha256` 覆盖本目录的发布文件；`verify_archive.py` 检查哈希、实验覆盖率与 CSV 行数。大文件另有压缩包哈希和解压后逐文件清单。历史远端归档共 11,346 个文件，其中 manifest 列出的 11,344 个载荷在本次整理中重新通过 SHA-256 检查，其余 2 个为 README 和 manifest 本身。
