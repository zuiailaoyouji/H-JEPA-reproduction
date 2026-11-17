# H-JEPA 数据补齐记录

日期：2026-10-26。

下载顺序：FourRoom → Push-T → AntMaze → Cube。

## 新增规划评估任务文件

| 环境 | 文件（相对 h_jepa/assets/eval_trajs） | 字节数 | 任务数 | 校验 |
| --- | --- | ---: | ---: | --- |
| FourRoom | `fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt` | 573,644,209 | 50 | 官方 SHA256 及 CPU 读取通过 |
| Push-T | `pusht/goal_offset_75_val.pt` | 564,814,221 | 50 | 官方 SHA256 及 CPU 读取通过 |
| AntMaze | `ant/expert_grid_d3_n50.pt` | 155,119,789 | 50 | 官方 SHA256 及 CPU 读取通过 |
| Cube | `ogbench/goal_offset_20_pickup_val.pt` | 152,425,457 | 50 | 官方 SHA256 及 CPU 读取通过 |

共 1,446,003,676 字节（1.35 GiB），共 200 个规划评估任务。

来源：Hugging Face 数据集 `jepa-world-models/h-jepa`。固定版本：`cc05cb8d2aaf9aec0d5364d073c6369ff760b60b`。校验清单保存在 `h_jepa/assets/eval_trajs/SHA256SUMS`。

28 份模拟环境评估配置均能找到对应任务文件。无残留 `.part` 或 `.assembling` 文件。

## 已有训练、验证与 probing 数据

12 份必需 HDF5 文件已存在，本次未重复下载；均通过结构、episode 偏移/计数与抽样读取检查。

| 环境 | 文件 | episodes | 官方 SHA256 |
| --- | --- | ---: | --- |
| FourRoom | `fourroom_tp35_d1.h5` | 3,750 | 通过 |
| FourRoom | `fourroom_tp35_d1_val.h5` | 150 | 通过 |
| FourRoom | `fourroom_tp35_d1_probing.h5` | 600 | 通过 |
| FourRoom | `fourroom_tp35_d1_probing_val.h5` | 150 | 通过 |
| Push-T | `pusht_expert_train.h5` | 18,685 | 本次未作官方哈希校验（结构/抽样通过） |
| Push-T | `pusht_expert_val.h5` | 21 | 通过 |
| AntMaze | `visual_antmaze_medium_explore_stitch_train.h5` | 25,000 | 通过 |
| AntMaze | `visual_antmaze_medium_stitch_val.h5` | 1,250 | 通过 |
| AntMaze | `visual_antmaze_medium_probing_train.h5` | 3,106 | 通过 |
| AntMaze | `visual_antmaze_medium_probing_eval.h5` | 310 | 通过 |
| Cube | `cube_single_expert_train.h5` | 9,800 | 本次未作官方哈希校验（结构/抽样通过） |
| Cube | `cube_single_expert_val.h5` | 200 | 本次未作官方哈希校验（结构/抽样通过） |

数据目录：`/home/songxw/hjepa-data`，指向 `/mnt/disk2/songxw/hjepa-data`。

## 读取时观察到的兼容性提示

最后一次 AntMaze/Cube CPU 读取命令成功退出，但出现 scikit-learn `InconsistentVersionWarning`：序列化的 StandardScaler 来自 1.7.2，当前环境为 1.9.1。文件 SHA256 校验和任务读取均通过；完整评估的数值兼容性尚未验证。本次未为此变更依赖或启动训练/评估。

## 本地保存

`local_env/` 和 `h_jepa/assets/` 已通过 git check-ignore 确认被忽略。未执行 Git 提交或 GitHub 上传。原有配置修改保持不变。

详细记录：`dataset-audit-20261026.json`、`dataset-sha256-20261026.json`、`eval-task-config-references.json`、`eval-task-validation-fourroom.json`、`eval-task-validation-pusht.json`、`eval-task-validation-ant-cube.json`。
