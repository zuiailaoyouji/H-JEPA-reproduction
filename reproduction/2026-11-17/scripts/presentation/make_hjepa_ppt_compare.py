#!/usr/bin/env python3
"""Build a compact experiment-by-experiment H-JEPA reproduction deck."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from make_hjepa_ppt import (
    AMBER,
    ASSETS,
    BG,
    BLUE,
    FONT,
    INK,
    LIGHT,
    LINE,
    MUTED,
    PALE_AMBER,
    PALE_BLUE,
    PALE_RED,
    PALE_TEAL,
    PURPLE,
    RED,
    ROOT,
    SLIDE_H,
    SLIDE_W,
    TEAL,
    WHITE,
    add_image_contain,
    add_line,
    add_rect,
    add_text,
    rgb,
)


DATA = Path("/home/songxw/hjepa-data")
OUTPUT = ROOT / "H-JEPA复现汇报_实验对照初版_20261028.pptx"


def crop_paper_figures():
    crops = [
        (ASSETS / "paper_page9.png", ASSETS / "paper_fig9.png", (215, 135, 1055, 520)),
        (ASSETS / "paper_page21.png", ASSETS / "paper_fig10.png", (225, 290, 1065, 1140)),
    ]
    for source, target, box in crops:
        with Image.open(source) as im:
            im.crop(box).save(target)


def chart_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#9AA5AE",
            "axes.labelcolor": "#18232D",
            "xtick.color": "#18232D",
            "ytick.color": "#18232D",
        }
    )


def make_probe_chart():
    source = DATA / "analyse/official_release_08fc8a21/probe_summary.csv"
    rows = list(csv.DictReader(source.open(newline="")))
    lookup = {(row["entity"], int(row["level"])): row for row in rows if row["env"] == "fourroom"}
    entities = ["xy", "distractor0_xy"]
    labels = ["ego xy（慢变量）", "distractor xy（快变量）"]
    x = np.arange(2)
    width = 0.33
    l1 = [float(lookup[(e, 1)]["nmse_mean"]) for e in entities]
    l2 = [float(lookup[(e, 2)]["nmse_mean"]) for e in entities]
    e1 = [float(lookup[(e, 1)]["nmse_se"]) for e in entities]
    e2 = [float(lookup[(e, 2)]["nmse_se"]) for e in entities]

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=180)
    b1 = ax.bar(x - width / 2, l1, width, yerr=e1, capsize=4, color="#AAB3BA", label="Level 1")
    b2 = ax.bar(x + width / 2, l2, width, yerr=e2, capsize=4, color="#D83A34", label="Level 2")
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("Probe NMSE（越高越难恢复）")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#D9E0E5", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            label = f"{val:.3f}" if val >= 0.001 else f"{val:.1e}"
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, label, ha="center", fontsize=9, fontweight="bold")
    ax.set_title("FourRoom L2｜官方权重本机冻结探针｜3 seeds", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "local_fourroom_probe.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_frequency_chart():
    source = ROOT / "local_env/analysis_20261027/frequency_summary.json"
    data = json.loads(source.read_text())
    order = ["fourroom", "ant", "pusht", "cube"]
    labels = ["FourRoom", "AntMaze", "Push-T", "Cube"]
    gaps = [data[key]["frequency_gap_power_spectrum_fastest_over_slowest"] for key in order]
    colors = ["#2E6F95", "#D83A34", "#E7A13A", "#2A9D8F"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=180)
    bars = ax.bar(labels, gaps, color=colors, width=0.60)
    ax.axhline(2.0, color="#6D7780", linestyle="--", linewidth=1.4, label="论文讨论的 2× 阈值")
    ax.set_ylim(0, 7.2)
    ax.set_ylabel("最快 / 最慢实体谱质心比")
    ax.set_title("公开训练数据重算｜periodogram-weighted centroid", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#D9E0E5", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.16, f"{val:.2f}×", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "local_frequency_gaps.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_planning_chart():
    metric_path = DATA / "ckpts/fourroom/fourroom_hjepa_l2/seed42_paper_v2_20261027/planning_eval/epoch_0008/metrics.yaml"
    metric = yaml.safe_load(metric_path.read_text())
    outcomes = [bool(x) for x in metric["episode_successes"]]
    success = sum(outcomes)
    rate = float(metric["success_rate"])

    fig = plt.figure(figsize=(7.2, 3.9), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar([0], [rate], width=0.55, color="#D83A34")
    ax1.bar([0], [100 - rate], bottom=[rate], width=0.55, color="#D9E0E5")
    ax1.set_ylim(0, 100)
    ax1.set_xticks([0], ["H-JEPA L2\nseed 42"])
    ax1.set_ylabel("规划任务占比 (%)")
    ax1.text(0, rate / 2, f"成功\n{rate:.1f}%", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax1.text(0, rate + (100 - rate) / 2, f"失败\n{100-rate:.1f}%", ha="center", va="center", color="#586574", fontsize=10, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    for idx, ok in enumerate(outcomes):
        row, col = divmod(idx, 10)
        ax2.scatter(col, 4 - row, s=260, marker="s", color="#D83A34" if ok else "#C8D0D6", edgecolor="white", linewidth=0.8)
    ax2.set_xlim(-0.8, 9.8)
    ax2.set_ylim(-0.8, 4.8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"固定任务逐项结果：{success}/50", fontsize=12, fontweight="bold")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.suptitle("FourRoom 从头训练｜protocol-v2｜7 epochs", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "local_fourroom_planning.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_droid_chart():
    source = DATA / "analyse/official_release_08fc8a21/droid_summary.csv"
    rows = {row["model"]: row for row in csv.DictReader(source.open(newline=""))}
    models = ["lewm", "hwm_l2", "hjepa_l2"]
    labels = ["LeWM + IDM", "HWM L2", "H-JEPA L2"]
    vals = [float(rows[m]["mean"]) for m in models]
    errs = [float(rows[m]["se"]) for m in models]
    colors = ["#9AA5AE", "#E7A13A", "#D83A34"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=180)
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors, width=0.58)
    ax.set_ylim(25, 41)
    ax.set_ylabel("Fréchet fidelity (%)")
    ax.set_title("官方权重本机重评估｜16 clips｜3 train seeds", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#D9E0E5", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.65, f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "local_droid_only.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_training_chart():
    source = ROOT / "local_env/analysis_20261027/training_loss_points.csv"
    grouped: dict[str, list[tuple[float, float]]] = {key: [] for key in ("fourroom", "ant", "pusht", "cube")}
    with source.open(newline="") as f:
        for row in csv.DictReader(f):
            env = row["environment"]
            if env not in grouped or not row.get("fit/loss"):
                continue
            grouped[env].append((float(row["step"]), float(row["fit/loss"])))

    names = {"fourroom": "FourRoom", "ant": "AntMaze", "pusht": "Push-T", "cube": "Cube"}
    colors = {"fourroom": "#2A9D8F", "ant": "#7B5BA7", "pusht": "#D83A34", "cube": "#2E6F95"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.9), dpi=180)
    for ax, env in zip(axes.flat, ("fourroom", "ant", "pusht", "cube")):
        points = grouped[env]
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        if len(xs) > 0:
            ax.plot(xs, ys, color=colors[env], linewidth=1.3)
        ax.set_yscale("log")
        ax.set_title(names[env], fontsize=10, fontweight="bold")
        ax.grid(color="#D9E0E5", alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("total loss")
    axes[1, 0].set_ylabel("total loss")
    axes[1, 0].set_xlabel("optimizer step")
    axes[1, 1].set_xlabel("optimizer step")
    fig.suptitle("本地 H-JEPA L2 / seed 42 / protocol-v2 日志", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "local_training_dynamics.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_assets():
    ASSETS.mkdir(exist_ok=True)
    crop_paper_figures()


def add_panel_header(slide, text, x, y, w, color, sub=None):
    add_rect(slide, x, y, w, 0.38, fill=color, line=color, radius=True)
    add_text(slide, text, x + 0.12, y + 0.08, w - 0.24, 0.20, size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    if sub:
        add_text(slide, sub, x, y + 0.43, w, 0.22, size=9.5, color=MUTED, align=PP_ALIGN.CENTER)


def add_bottom_note(slide, title, body, x, color, pale, code=None):
    add_rect(slide, x, 5.18, 5.93, 1.64, fill=WHITE, line=LINE, radius=True)
    add_rect(slide, x, 5.18, 0.14, 1.64, fill=color, line=color)
    add_text(slide, title, x + 0.34, 5.38, 2.35, 0.29, size=15, color=color, bold=True)
    add_text(slide, body, x + 0.34, 5.78, 5.25, 0.70, size=11.6, color=INK, bold=True, line_spacing=1.08)
    if code:
        add_rect(slide, x + 0.32, 6.49, 5.28, 0.20, fill=pale, line=pale, radius=True)
        add_text(slide, code, x + 0.41, 6.53, 5.10, 0.13, size=7.8, color=MUTED, bold=True, align=PP_ALIGN.CENTER)


def add_compare_slide(
    prs,
    *,
    page,
    title,
    paper_image,
    local_image,
    local_label,
    comparison,
    how,
    code,
    why,
    source,
):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    add_rect(slide, 0, 0, SLIDE_W, 0.07, fill=RED, line=RED)
    add_text(slide, title, 0.55, 0.25, 12.20, 0.52, size=25, color=INK, bold=True)
    add_line(slide, 6.665, 0.92, 6.665, 4.55, color=LINE, width=1.2)
    add_panel_header(slide, "原论文结果", 0.55, 0.91, 1.42, INK, "paper")
    add_panel_header(slide, "我的复现结果", 6.88, 0.91, 1.62, RED, local_label)
    add_image_contain(slide, paper_image, 0.55, 1.38, 5.92, 2.88, bg=WHITE, border=True)
    if local_image is None:
        add_rect(slide, 6.88, 1.38, 5.90, 2.88, fill=WHITE, line="C8D0D6")
    else:
        add_image_contain(slide, local_image, 6.88, 1.38, 5.90, 2.88, bg=WHITE, border=True)
    add_rect(slide, 0.55, 4.46, 12.23, 0.48, fill=INK, line=INK, radius=True)
    add_text(slide, comparison, 0.82, 4.59, 11.68, 0.23, size=12.3, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_bottom_note(slide, "我是怎么复现的", how, 0.55, BLUE, PALE_BLUE, code=code)
    add_bottom_note(slide, "为什么论文要这样设计", why, 6.88, RED, PALE_RED)
    add_line(slide, 0.55, 7.08, 12.78, 7.08, color=LINE, width=0.7)
    add_text(slide, source, 0.58, 7.14, 11.35, 0.16, size=8.1, color=MUTED)
    add_text(slide, f"{page:02d}", 12.20, 7.14, 0.55, 0.16, size=8.1, color=MUTED, align=PP_ALIGN.RIGHT)
    return slide


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    prs.core_properties.title = "H-JEPA 复现：逐实验论文/本地结果对照"
    prs.core_properties.subject = "每页左论文结果，右复现结果，下方为代码路径与实验动机"
    prs.core_properties.author = "[姓名]"

    add_compare_slide(
        prs,
        page=1,
        title="Fig. 4｜高层表示是否真的选择性抽象？",
        paper_image=ASSETS / "paper_depth_probes.png",
        local_image=None,
        local_label="留空待插入你的结果图",
        comparison="对比重点：快变量的 probe NMSE 是否随层级上升；任务相关的慢变量是否仍保持可恢复。",
        how="冻结 H-JEPA 编码器，只训练 MLP probe；以目标方差归一化 MSE。使用 FourRoom L2 官方权重，seeds 42/43/44，探针训练 60 epochs 后聚合 mean±SE。",
        code="config/probing/fourroom.yaml → final_probing_decoding_eval.py:273–286, 516–529 → analyze_probes.py",
        why="我理解到：规划成功率只能说明“能用”，不能证明层级里学到了什么。通过独立移动的 distractor 与任务相关 ego 位置，论文构造了一个可控对照，专门检验高层是否丢快细节、留慢状态。",
        source="论文：arXiv:2610.06805v1 Fig.4；本地：analyse/official_release_08fc8a21/probe_summary.csv。",
    )

    add_compare_slide(
        prs,
        page=2,
        title="Fig. 5 / 12｜选择性抽象在什么数据条件下出现？",
        paper_image=ASSETS / "paper_frequency_abstraction.png",
        local_image=None,
        local_label="留空待插入你的结果图",
        comparison="对比重点：你的频率 gap 排序是否与论文一致；gap 较大的环境是否也出现更强的选择性抽象。",
        how="对每个 episode、每个状态维度做零均值/单位方差；去掉 FFT 的 DC 项，用 |rFFT|² 加权谱质心，再按实体聚合并计算 fastest/slowest ratio。",
        code="scripts/analyze_frequency.py → frequency_summary.json / spectral_centroids.csv（公开 HDF5 数据）",
        why="我理解到：高层并不会必然产生抽象。只有快慢变量的时间尺度明显分离时，长时域预测才会把不可预测的快变量当成负担。这个实验是在给方法划定适用条件，而不是只展示好案例。",
        source="论文：arXiv:2610.06805v1 Fig.5/12；本地方法属于 qualified reproduction，详见 frequency_method.json。",
    )

    add_compare_slide(
        prs,
        page=3,
        title="Fig. 6｜层级规划是否真正提升成功率？",
        paper_image=ROOT / "assets/depth_planning_sr.png",
        local_image=None,
        local_label="留空待插入你的结果图",
        comparison="对比重点：同环境、同模型深度、同任务数下比较成功率，并明确你的 seed 数与论文 mean±SE 的口径差异。",
        how="按 fourroom_hjepa_l2.yaml 从头训练 7 epochs（batch=128, accum=1, stride₂=2）；训练结束后用同一 fourroom_l2 配置在 50 个固定 start/goal 任务上层级规划。",
        code="main_hjepa.py → models/hjepa.py / hjepa_utils.py → hierarchical_solver.py:90–225 → planning_eval.py",
        why="我理解到：LeWM、HWM、H-JEPA 的对照不是简单“换模型”。LeWM 检验是否需要层级；HWM 只保留时间层级；H-JEPA 再加入独立潜空间，因此可以把时间分解与表示抽象的贡献拆开。",
        source="论文：arXiv:2610.06805v1 Fig.6；本地：seed42_paper_v2_20261027/planning_eval/epoch_0008/metrics.yaml。",
    )

    add_compare_slide(
        prs,
        page=4,
        title="Fig. 9｜多场景 DROID 上，为什么必须加入逆动力学损失？",
        paper_image=ASSETS / "paper_fig9.png",
        local_image=None,
        local_label="留空待插入你的结果图",
        comparison="对比重点：LeWM+IDM、HWM、H-JEPA 的排序是否一致，并逐项报告 Fréchet fidelity 的绝对差值。",
        how="对 3 个模型 × 3 个训练 seed × 3 个 planner seed 重跑 16 个 5fps clips；每 clip 计算累计 xyz 轨迹的 Fréchet fidelity，先平均 planner seeds，再跨训练 seeds 求 mean±SE。",
        code="eval.py → droid_eval.py:70–86 → traj_metrics.py:19–53 → analyze_official_release.py:116–178",
        why="我理解到：DROID 的场景身份本身很容易预测，纯 JEPA 会把容量用在静态背景上。IDM 强迫相邻表示包含动作相关变化；Fréchet 评估整条路径而非终点，也更适合非贪心操作任务。",
        source="论文：arXiv:2610.06805v1 Fig.9；本地：analyse/official_release_08fc8a21/droid_summary.csv。",
    )

    add_compare_slide(
        prs,
        page=5,
        title="Fig. 10｜训练曲线为什么也是论文结果的一部分？",
        paper_image=ASSETS / "paper_fig10.png",
        local_image=None,
        local_label="留空待插入你的结果图",
        comparison="对比重点：prediction / SIGReg / total loss 的下降趋势、各层差异，以及你的深度、seed 数和训练 epoch 是否与论文一致。",
        how="训练时逐 step 记录 prediction、SIGReg、total loss；恢复训练后按 global step 合并日志片段，后续 session 覆盖重叠后缀，避免把重跑区间重复计数。",
        code="main_hjepa.py:155–167 → training_runtime.py → scripts/analyze_training_dynamics.py",
        why="我理解到：最终成功率无法区分“模型学错了”还是“优化没收敛”。逐层 prediction/SIGReg 曲线用于检查高层是否得到有效梯度、是否发生 collapse，并解释不同环境为何需要不同 epoch 预算。",
        source="论文：arXiv:2610.06805v1 Fig.10；本地图为 protocol-v2 单 seed 部分趋势，源自 training_loss_points.csv。",
    )

    prs.save(OUTPUT)
    print(OUTPUT)


def main():
    make_assets()
    build_deck()


if __name__ == "__main__":
    main()
