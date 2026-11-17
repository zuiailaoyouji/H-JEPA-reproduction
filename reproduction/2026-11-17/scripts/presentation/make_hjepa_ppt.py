#!/usr/bin/env python3
"""Generate the first-draft H-JEPA reproduction presentation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
DATA = Path("/home/songxw/hjepa-data")
ASSETS = ROOT / "ppt_assets"
OUTPUT = ROOT / "H-JEPA复现汇报_初版_20261028.pptx"

SLIDE_W = 13.333
SLIDE_H = 7.5

FONT = "Microsoft YaHei"
FONT_MONO = "DejaVu Sans Mono"

BG = "F7F8FA"
WHITE = "FFFFFF"
INK = "18232D"
MUTED = "5C6976"
LINE = "D9E0E5"
LIGHT = "EAF0F3"
RED = "D83A34"
BLUE = "2E6F95"
TEAL = "2A9D8F"
AMBER = "E7A13A"
PURPLE = "7B5BA7"
PALE_RED = "F8E8E6"
PALE_BLUE = "E6F0F5"
PALE_TEAL = "E5F2EF"
PALE_AMBER = "F9F0DC"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def set_cell_border(cell, color=LINE, width="12700"):
    from lxml import etree
    from pptx.oxml.ns import qn

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = tcPr.find(qn(edge))
        if ln is None:
            ln = etree.SubElement(tcPr, qn(edge))
        ln.set("w", width)
        solid = ln.find(qn("a:solidFill"))
        if solid is None:
            solid = etree.SubElement(ln, qn("a:solidFill"))
        for child in list(solid):
            solid.remove(child)
        srgb = etree.SubElement(solid, qn("a:srgbClr"))
        srgb.set("val", color)


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    *,
    size=18,
    color=INK,
    bold=False,
    font=FONT,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin=0,
    line_spacing=1.0,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    run = p.add_run()
    run.text = str(text)
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_rich_text(slide, segments, x, y, w, h, *, size=18, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    for seg in segments:
        run = p.add_run()
        run.text = seg[0]
        run.font.name = seg[4] if len(seg) > 4 else FONT
        run.font.size = Pt(seg[3] if len(seg) > 3 else size)
        run.font.bold = bool(seg[2])
        run.font.color.rgb = rgb(seg[1])
    return box


def add_bullets(slide, bullets, x, y, w, h, *, size=17, color=INK, gap=8, bullet_color=RED):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for idx, item in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.08
        r1 = p.add_run()
        r1.text = "●  "
        r1.font.name = FONT
        r1.font.size = Pt(max(9, size - 5))
        r1.font.color.rgb = rgb(bullet_color)
        r2 = p.add_run()
        r2.text = item
        r2.font.name = FONT
        r2.font.size = Pt(size)
        r2.font.color.rgb = rgb(color)
    return box


def add_rect(slide, x, y, w, h, *, fill=WHITE, line=LINE, radius=False, transparency=0):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.fill.transparency = transparency
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(0.8)
    if radius:
        try:
            shape.adjustments[0] = 0.08
        except Exception:
            pass
    return shape


def add_line(slide, x1, y1, x2, y2, *, color=LINE, width=1.2, arrow=False):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    if arrow:
        from pptx.oxml import parse_xml
        from pptx.oxml.ns import nsdecls

        line.line._ln.append(parse_xml(f"<a:tailEnd {nsdecls('a')} type=\"triangle\"/>"))
    return line


def add_image_contain(slide, path, x, y, w, h, *, bg=None, border=False):
    path = Path(path)
    if bg:
        add_rect(slide, x, y, w, h, fill=bg, line=bg)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    pic = slide.shapes.add_picture(
        str(path),
        Inches(x + (w - dw) / 2),
        Inches(y + (h - dh) / 2),
        Inches(dw),
        Inches(dh),
    )
    if border:
        pic.line.color.rgb = rgb(LINE)
        pic.line.width = Pt(0.8)
    return pic


def add_image_cover(slide, path, x, y, w, h):
    path = Path(path)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = max(w / iw, h / ih)
    sw, sh = iw * scale, ih * scale
    crop_left = max(0, (sw - w) / sw / 2)
    crop_right = crop_left
    crop_top = max(0, (sh - h) / sh / 2)
    crop_bottom = crop_top
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic.crop_left = crop_left
    pic.crop_right = crop_right
    pic.crop_top = crop_top
    pic.crop_bottom = crop_bottom
    return pic


def add_title(slide, title, section=None, page=None):
    add_rect(slide, 0, 0, SLIDE_W, 0.08, fill=RED, line=RED)
    add_text(slide, title, 0.55, 0.30, 10.8, 0.55, size=27, bold=True)
    if section:
        add_text(slide, section, 11.1, 0.34, 1.65, 0.28, size=10, color=RED, bold=True, align=PP_ALIGN.RIGHT)
    if page is not None:
        add_text(slide, f"{page:02d}", 12.3, 7.12, 0.45, 0.2, size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def add_footer(slide, text, page):
    add_line(slide, 0.55, 7.05, 12.78, 7.05, color=LINE, width=0.7)
    add_text(slide, text, 0.58, 7.11, 11.35, 0.18, size=8.5, color=MUTED)
    add_text(slide, f"{page:02d}", 12.15, 7.11, 0.6, 0.18, size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)


def add_metric(slide, value, label, x, y, w, *, color=RED, sub=None):
    add_text(slide, value, x, y, w, 0.62, size=31, color=color, bold=True)
    add_text(slide, label, x, y + 0.58, w, 0.43, size=14, color=INK, bold=True)
    if sub:
        add_text(slide, sub, x, y + 0.99, w, 0.48, size=10.5, color=MUTED)


def add_label(slide, text, x, y, w, *, fill=PALE_RED, color=RED):
    add_rect(slide, x, y, w, 0.30, fill=fill, line=fill, radius=True)
    add_text(slide, text, x, y + 0.03, w, 0.20, size=9.5, color=color, bold=True, align=PP_ALIGN.CENTER)


def style_table(table, *, header_fill=INK, header_color=WHITE, body_fill=WHITE, body_size=11.5):
    for r in range(len(table.rows)):
        for c in range(len(table.columns)):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(header_fill if r == 0 else body_fill)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            set_cell_border(cell)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                p.font.name = FONT
                p.font.size = Pt(10.5 if r == 0 else body_size)
                p.font.bold = r == 0
                p.font.color.rgb = rgb(header_color if r == 0 else INK)
                p.line_spacing = 1.0


def load_results():
    fourroom_path = DATA / "ckpts/fourroom/fourroom_hjepa_l2/seed42_paper_v2_20261027/planning_eval/epoch_0008/metrics.yaml"
    with fourroom_path.open() as f:
        fourroom = yaml.safe_load(f)

    droid_path = DATA / "analyse/official_release_08fc8a21/droid_summary.csv"
    droid = {}
    with droid_path.open(newline="") as f:
        for row in csv.DictReader(f):
            droid[row["model"]] = {
                "mean": float(row["mean"]),
                "se": float(row["se"]),
                "n": int(row["n"]),
            }

    status_path = DATA / "analyse/official_release_08fc8a21/runner_status_droid_probe_depth_cost.json"
    with status_path.open() as f:
        status = json.load(f)

    return {
        "fourroom_success": float(fourroom["success_rate"]),
        "fourroom_n": len(fourroom["episode_successes"]),
        "droid": droid,
        "status": status,
    }


def make_charts(results):
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#98A3AD",
            "axes.labelcolor": "#18232D",
            "xtick.color": "#18232D",
            "ytick.color": "#18232D",
        }
    )

    # FourRoom: protocol-v2 local training versus the three-seed paper reference.
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=180)
    vals = [81.3, results["fourroom_success"]]
    err = [6.7, 0]
    bars = ax.bar([0, 1], vals, yerr=err, capsize=5, color=["#AAB3BA", "#D83A34"], width=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel("规划成功率 (%)")
    ax.set_xticks([0, 1], ["论文 H-JEPA L2\n3 seeds, mean ± SE", "本地 protocol-v2\nseed 42, 40/50"])
    ax.grid(axis="y", color="#D9E0E5", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fourroom_reproduction.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # DROID: paper versus local official-weight re-evaluation.
    paper = {"lewm": (33.57, 1.52), "hwm_l2": (35.65, 2.22), "hjepa_l2": (38.42, 0.94)}
    models = ["lewm", "hwm_l2", "hjepa_l2"]
    labels = ["LeWM + IDM", "HWM L2", "H-JEPA L2"]
    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=180)
    pvals = [paper[m][0] for m in models]
    perr = [paper[m][1] for m in models]
    lvals = [results["droid"][m]["mean"] for m in models]
    lerr = [results["droid"][m]["se"] for m in models]
    b1 = ax.bar(x - width / 2, pvals, width, yerr=perr, capsize=4, color="#AAB3BA", label="论文")
    b2 = ax.bar(x + width / 2, lvals, width, yerr=lerr, capsize=4, color="#2E6F95", label="本机重评估")
    ax.set_ylim(25, 42)
    ax.set_ylabel("Fréchet fidelity (%)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#D9E0E5", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.45, f"{bar.get_height():.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "droid_reproduction.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def new_deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    prs.core_properties.title = "H-JEPA 论文复现汇报（初版）"
    prs.core_properties.subject = "论文实验目的、论文结果、本地复现结果与代码映射"
    prs.core_properties.author = "[姓名]"
    return prs


def add_blank(prs, fill=BG):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(fill)
    return slide


def build_deck(results):
    prs = new_deck()
    paper_url = "arXiv:2610.06805v1"

    # 1. Cover
    slide = add_blank(prs, WHITE)
    add_rect(slide, 0, 0, 4.65, 7.5, fill=INK, line=INK)
    add_rect(slide, 0.55, 0.55, 0.65, 0.08, fill=RED, line=RED)
    add_text(slide, "H-JEPA", 0.55, 1.05, 3.45, 0.72, size=35, color=WHITE, bold=True)
    add_text(slide, "复现任务汇报", 0.55, 1.78, 3.45, 0.62, size=28, color=WHITE, bold=True)
    add_text(slide, "论文结果 × 本地复现 × 代码证据", 0.58, 2.62, 3.4, 0.55, size=16, color="CBD5DD")
    add_text(slide, "初版｜2026-10-28", 0.58, 5.87, 2.9, 0.28, size=11, color="CBD5DD")
    add_text(slide, "汇报人：[姓名]", 0.58, 6.27, 2.9, 0.28, size=11, color=WHITE, bold=True)
    add_image_contain(slide, ROOT / "assets/teaser.png", 4.83, 0.78, 8.10, 5.62, bg=WHITE)
    add_text(slide, "End-to-End Learning of Hierarchical World Models for Visual Planning", 5.0, 6.70, 7.75, 0.34, size=11, color=INK, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, "Zhang et al. · arXiv:2610.06805v1", 5.0, 7.05, 7.75, 0.20, size=8.5, color=MUTED, align=PP_ALIGN.RIGHT)

    # 2. Executive summary
    slide = add_blank(prs)
    add_title(slide, "先给结论：核心链路已跑通，但完整矩阵仍在进行", "SUMMARY")
    add_metric(slide, "80.0%", "FourRoom L2 自训", 0.75, 1.22, 2.7, color=RED, sub="protocol-v2 · seed 42 · 40/50")
    add_metric(slide, "81.3±6.7%", "论文对应结果", 3.65, 1.22, 2.8, color=INK, sub="3 seeds · mean ± SE")
    add_metric(slide, "35.86%", "DROID H-JEPA 重评估", 6.67, 1.22, 2.9, color=BLUE, sub="官方权重 · 3 train seeds")
    status = results["status"]
    add_metric(slide, f"{status['complete']}/{status['total']}", "官方权重评估矩阵", 9.85, 1.22, 2.7, color=TEAL, sub=f"进行中 · failures={len(status['failures'])}")
    add_line(slide, 0.75, 3.10, 12.55, 3.10, color=LINE)
    add_text(slide, "当前可以对老师明确说明的三件事", 0.75, 3.42, 5.7, 0.42, size=19, bold=True)
    add_bullets(
        slide,
        [
            "方法复现不是只“跑起来”：训练、层级规划、冻结探针与严格聚合均已映射到代码。",
            "本地自训已得到一个与论文均值接近的有效点；单 seed 不能替代论文三 seed 统计。",
            "官方权重重评估保留了 DROID 上 H-JEPA > HWM > LeWM 的排序，但 H-JEPA 绝对值低 2.56 个百分点。",
        ],
        0.78,
        3.95,
        7.15,
        2.35,
        size=16,
    )
    add_rect(slide, 8.45, 3.48, 4.0, 2.55, fill=PALE_AMBER, line=PALE_AMBER, radius=True)
    add_label(slide, "汇报口径", 8.75, 3.78, 1.2, fill=AMBER, color=WHITE)
    add_text(slide, "论文结果", 8.78, 4.34, 1.15, 0.28, size=13, bold=True)
    add_text(slide, "公开三种子均值 ± SE", 9.95, 4.34, 2.05, 0.28, size=12, color=MUTED)
    add_text(slide, "本地自训", 8.78, 4.82, 1.15, 0.28, size=13, bold=True)
    add_text(slide, "单 seed，单独标注", 9.95, 4.82, 2.05, 0.28, size=12, color=MUTED)
    add_text(slide, "官方重评", 8.78, 5.30, 1.15, 0.28, size=13, bold=True)
    add_text(slide, "验证评估链路，不等于重训", 9.95, 5.30, 2.15, 0.45, size=12, color=MUTED)
    add_footer(slide, "本地结果截至 2026-10-28；官方权重 revision 08fc8a21。", 2)

    # 3. Problem and method
    slide = add_blank(prs)
    add_title(slide, "为什么需要 H-JEPA：单一潜空间同时承担两种冲突目标", "METHOD")
    add_rect(slide, 0.65, 1.12, 3.35, 2.15, fill=PALE_RED, line=PALE_RED, radius=True)
    add_label(slide, "失败模式 1", 0.93, 1.40, 1.05)
    add_text(slide, "长时域滚动", 0.93, 1.90, 2.2, 0.34, size=19, bold=True)
    add_text(slide, "细粒度步数多 → 误差累积\n动作搜索空间快速膨胀", 0.93, 2.35, 2.72, 0.65, size=14, color=MUTED, line_spacing=1.15)
    add_rect(slide, 0.65, 3.60, 3.35, 2.15, fill=PALE_BLUE, line=PALE_BLUE, radius=True)
    add_label(slide, "失败模式 2", 0.93, 3.88, 1.05, fill=BLUE, color=WHITE)
    add_text(slide, "目标匹配不抽象", 0.93, 4.38, 2.4, 0.34, size=19, bold=True)
    add_text(slide, "低层姿态细节主导距离\n“到达位置”被误当成“匹配姿态”", 0.93, 4.83, 2.82, 0.65, size=14, color=MUTED, line_spacing=1.15)

    add_text(slide, "H-JEPA 的回答", 4.55, 1.12, 3.2, 0.40, size=20, bold=True)
    levels = [
        ("L3", "更长时域", "保留慢变量 / 规划全局目标", PURPLE, 1.78),
        ("L2", "中等时域", "产生子目标 / 塑造抽象代价", RED, 3.17),
        ("L1", "短时域", "输出原始动作 / 精细控制", BLUE, 4.56),
    ]
    for lab, horizon, desc, color, yy in levels:
        add_rect(slide, 4.55, yy, 7.55, 0.98, fill=WHITE, line=LINE, radius=True)
        add_rect(slide, 4.55, yy, 0.95, 0.98, fill=color, line=color, radius=True)
        add_text(slide, lab, 4.55, yy + 0.25, 0.95, 0.34, size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, horizon, 5.78, yy + 0.19, 1.55, 0.28, size=14, color=color, bold=True)
        add_text(slide, desc, 7.25, yy + 0.19, 4.35, 0.46, size=14, color=INK)
    add_line(slide, 11.35, 2.75, 11.35, 3.15, color=RED, width=2.0, arrow=True)
    add_line(slide, 11.35, 4.14, 11.35, 4.54, color=RED, width=2.0, arrow=True)
    add_text(slide, "上层预测 → 下层子目标", 7.18, 5.86, 4.82, 0.34, size=15, color=RED, bold=True, align=PP_ALIGN.RIGHT)
    add_footer(slide, f"论文依据：{paper_url}，§1–2，Fig. 2–3。", 3)

    # 4. Reproduction protocol
    slide = add_blank(prs)
    add_title(slide, "复现口径：把“自训”和“官方权重重评估”分开", "PROTOCOL")
    columns = [
        ("A", "本地从头训练", RED, PALE_RED, ["验证训练链路", "真实 checkpoint", "单 seed 42", "论文 protocol-v2"]),
        ("B", "官方权重重评估", BLUE, PALE_BLUE, ["验证规划/指标链路", "hash 固定权重", "三训练 seed", "固定评估任务"]),
        ("C", "探索性/部分结果", AMBER, PALE_AMBER, ["频率分析", "旧协议单 seed", "不并入论文均值", "用于诊断机制"]),
    ]
    for i, (tag, title, color, pale, items) in enumerate(columns):
        x = 0.68 + i * 4.14
        add_rect(slide, x, 1.22, 3.72, 4.85, fill=WHITE, line=LINE, radius=True)
        add_rect(slide, x, 1.22, 3.72, 0.72, fill=color, line=color, radius=True)
        add_text(slide, tag, x + 0.18, 1.42, 0.42, 0.30, size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, title, x + 0.67, 1.42, 2.75, 0.30, size=17, color=WHITE, bold=True)
        yy = 2.24
        for item in items:
            add_rect(slide, x + 0.28, yy, 3.16, 0.58, fill=pale, line=pale, radius=True)
            add_text(slide, item, x + 0.50, yy + 0.17, 2.75, 0.24, size=13, color=INK, bold=True)
            yy += 0.75
    add_rect(slide, 0.68, 6.32, 12.0, 0.48, fill=INK, line=INK, radius=True)
    add_text(slide, "比较原则：同一页必须同时写清 seeds、任务数、指标、权重来源；缺失项保持空白。", 0.98, 6.45, 11.45, 0.23, size=12.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, "本地审计：local_env/official_reproduction_20261027/REPRODUCTION_SCOPE.md。", 4)

    # 5. Experiment map
    slide = add_blank(prs)
    add_title(slide, "论文为什么做这些实验：从“有效”追问到“为何有效”", "EXPERIMENT MAP")
    rows = [
        ("Fig. 4", "层级表征是否真的更抽象？", "冻结探针 + 像素解码", "表征机制"),
        ("Fig. 5 / 12", "什么时候会出现选择性抽象？", "频率间隔 vs. NMSE 变化", "适用边界"),
        ("Fig. 6", "是否比平坦模型更准、更省算力？", "LeWM / HWM / H-JEPA × 深度", "主效果"),
        ("Table 1/9 + Fig. 7", "增益是否来自更好的目标代价空间？", "固定 L1 动力学，仅替换 cost space", "表征归因"),
        ("Fig. 8 + Table 10", "增益是否来自时间分解？", "全局目标代价 vs. 子目标跟踪", "时间归因"),
        ("Fig. 9", "多场景真实机器人视频上是否仍成立？", "DROID + IDM + Fréchet fidelity", "外部有效性"),
        ("Fig. 10", "训练是否稳定、各项损失如何收敛？", "prediction / SIGReg / total loss", "优化诊断"),
        ("Fig. 11", "成功率背后的行为是否合理？", "成功执行序列", "定性验证"),
    ]
    table_shape = slide.shapes.add_table(len(rows) + 1, 4, Inches(0.60), Inches(1.13), Inches(12.14), Inches(5.55))
    table = table_shape.table
    widths = [1.15, 4.10, 4.25, 2.64]
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    headers = ["论文位置", "问题 / 做实验的原因", "控制变量与证据", "回答层级"]
    for c, val in enumerate(headers):
        table.cell(0, c).text = val
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            table.cell(r, c).text = val
    style_table(table, body_size=11.4)
    for r in range(1, len(rows) + 1):
        table.cell(r, 0).fill.fore_color.rgb = rgb(PALE_RED if r in (3, 4, 5, 6) else LIGHT)
        table.cell(r, 0).text_frame.paragraphs[0].font.bold = True
        table.cell(r, 3).text_frame.paragraphs[0].font.color.rgb = rgb(RED if r in (3, 4, 5, 6) else BLUE)
        table.cell(r, 3).text_frame.paragraphs[0].font.bold = True
    add_footer(slide, f"实验目的根据 {paper_url} §3–4 与附录整理。", 5)

    # 6. Selective abstraction
    slide = add_blank(prs)
    add_title(slide, "Fig. 4：验证高层潜变量是否丢弃快细节、保留任务状态", "REPRESENTATION")
    add_image_contain(slide, ASSETS / "paper_depth_probes.png", 0.58, 1.16, 8.25, 3.35, bg=WHITE)
    add_rect(slide, 9.10, 1.18, 3.58, 3.28, fill=WHITE, line=LINE, radius=True)
    add_label(slide, "为什么做", 9.42, 1.47, 1.05)
    add_text(slide, "规划性能提升并不自动证明\n“学到了抽象层级”。", 9.42, 1.99, 2.85, 0.75, size=17, bold=True, line_spacing=1.15)
    add_text(slide, "因此冻结世界模型，用 MLP probe 测量各层能否恢复真实状态；NMSE 越大代表信息越难恢复。", 9.42, 2.93, 2.85, 1.06, size=13.5, color=MUTED, line_spacing=1.12)
    add_rect(slide, 0.70, 4.82, 11.98, 1.38, fill=INK, line=INK, radius=True)
    add_text(slide, "论文结果", 0.98, 5.06, 1.25, 0.30, size=14, color="F4B6B2", bold=True)
    add_text(slide, "Ant 的 body、FourRoom 的 distractor 随层级上升而 NMSE 增大；位置/ego 始终接近 0。", 2.20, 5.04, 9.95, 0.37, size=16, color=WHITE, bold=True)
    add_text(slide, "含义：高层选择性抛弃快速、难预测的细节，同时保留慢速、与目标相关的状态。解码图仅作可视化，不是定量指标。", 2.20, 5.51, 9.95, 0.44, size=12.5, color="CBD5DD")
    add_footer(slide, f"来源：{paper_url}，Fig. 4。图像来自 arXiv HTML。", 6)

    # 7. Frequency gap
    slide = add_blank(prs)
    add_title(slide, "Fig. 5 / 12：解释“何时抽象”——关键是时间频率分离", "MECHANISM")
    add_image_contain(slide, ASSETS / "paper_frequency_abstraction.png", 0.58, 1.16, 8.40, 4.54, bg=WHITE)
    add_rect(slide, 9.25, 1.18, 3.43, 4.52, fill=WHITE, line=LINE, radius=True)
    add_label(slide, "实验逻辑", 9.55, 1.46, 1.05, fill=PURPLE, color=WHITE)
    add_text(slide, "1", 9.53, 2.07, 0.35, 0.30, size=17, color=PURPLE, bold=True)
    add_text(slide, "计算各实体的谱质心，得到最快/最慢频率比。", 9.97, 2.03, 2.28, 0.72, size=13.5, bold=True)
    add_text(slide, "2", 9.53, 2.98, 0.35, 0.30, size=17, color=PURPLE, bold=True)
    add_text(slide, "比较 L1→L2 的 probe NMSE 变化。", 9.97, 2.94, 2.28, 0.56, size=13.5, bold=True)
    add_text(slide, "3", 9.53, 3.77, 0.35, 0.30, size=17, color=PURPLE, bold=True)
    add_text(slide, "检验频率差越大，快变量是否越容易被抛弃。", 9.97, 3.73, 2.28, 0.72, size=13.5, bold=True)
    add_text(slide, "论文结论", 9.55, 4.68, 1.05, 0.28, size=13, color=RED, bold=True)
    add_text(slide, ">2× 的环境出现明显选择性抽象；Push-T/Cube/DROID 频率差小，抽象不明显。", 9.55, 5.03, 2.70, 0.64, size=13, color=MUTED)
    add_rect(slide, 0.72, 5.94, 11.94, 0.70, fill=PALE_TEAL, line=PALE_TEAL, radius=True)
    add_text(slide, "本地探索性结果：公开四环境数据的频率 gap 已重算；Ant 6.27× 最大，但精确 FFT 聚合规则未公开，因此标为 qualified reproduction。", 1.02, 6.14, 11.32, 0.28, size=12.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, f"来源：{paper_url}，Fig. 5/12；本地：analyse/spectral_centroids.csv。", 7)

    # 8. Planning performance
    slide = add_blank(prs)
    add_title(slide, "Fig. 6：主实验回答“层级规划是否更准，并且更省算力”", "PLANNING")
    add_image_contain(slide, ROOT / "assets/depth_planning_sr.png", 0.60, 1.13, 8.65, 3.58, bg=WHITE)
    add_rect(slide, 9.48, 1.16, 3.16, 3.50, fill=WHITE, line=LINE, radius=True)
    add_label(slide, "对照为什么这样设", 9.75, 1.44, 1.55)
    add_text(slide, "LeWM", 9.76, 2.05, 0.82, 0.26, size=13, color=MUTED, bold=True)
    add_text(slide, "平坦 JEPA 基线", 10.61, 2.05, 1.55, 0.26, size=12.5)
    add_text(slide, "HWM", 9.76, 2.54, 0.82, 0.26, size=13, color=BLUE, bold=True)
    add_text(slide, "只有时间层级；共享潜空间", 10.61, 2.54, 1.70, 0.48, size=12.5)
    add_text(slide, "H-JEPA", 9.76, 3.22, 0.82, 0.26, size=13, color=RED, bold=True)
    add_text(slide, "时间层级 + 独立抽象潜空间", 10.61, 3.22, 1.70, 0.48, size=12.5)
    add_text(slide, "因此 H-JEPA vs HWM 可以隔离“表示层级”的额外贡献。", 9.76, 3.96, 2.30, 0.52, size=12, color=MUTED)
    add_rect(slide, 0.72, 5.03, 11.93, 1.31, fill=INK, line=INK, radius=True)
    add_text(slide, "最强论文信号", 0.98, 5.30, 1.62, 0.28, size=14, color="F4B6B2", bold=True)
    add_text(slide, "Visual AntMaze：LeWM 18.0% → H-JEPA L3 73.3%，且优于 HWM L3 的 38.7%。", 2.60, 5.23, 9.25, 0.38, size=18, color=WHITE, bold=True)
    add_text(slide, "边界：Push-T 在 L3/L4 退化，论文归因于短 episode 导致高层可用训练 clip 只剩 58%/14%。", 2.60, 5.73, 9.25, 0.35, size=12.5, color="CBD5DD")
    add_footer(slide, f"来源：{paper_url}，Fig. 6；数值见 docs/RESULTS.md。", 8)

    # 9. Local fresh-training result
    slide = add_blank(prs)
    add_title(slide, "本地从头训练：FourRoom L2 单种子结果接近论文均值", "REPRODUCTION")
    add_image_contain(slide, ASSETS / "fourroom_reproduction.png", 0.65, 1.15, 7.25, 4.60, bg=WHITE)
    add_rect(slide, 8.18, 1.17, 4.50, 4.58, fill=WHITE, line=LINE, radius=True)
    add_label(slide, "可比条件", 8.50, 1.48, 1.05, fill=TEAL, color=WHITE)
    add_bullets(
        slide,
        [
            "H-JEPA L2，seed=42",
            "7 epochs，batch=128，accumulation=1",
            "50 个固定 start/goal 任务",
            "同一 fourroom_l2 层级规划配置",
            "结果：40/50 = 80.0%",
        ],
        8.52,
        1.97,
        3.62,
        2.75,
        size=14,
        bullet_color=TEAL,
    )
    add_rect(slide, 8.48, 4.88, 3.88, 0.58, fill=PALE_AMBER, line=PALE_AMBER, radius=True)
    add_text(slide, "注意：论文是 3 seeds 均值；单 seed 不能判定统计复现。", 8.68, 5.04, 3.48, 0.26, size=11.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.70, 6.02, 11.96, 0.58, fill=PALE_BLUE, line=PALE_BLUE, radius=True)
    add_text(slide, "训练后 planning 指标已完整落盘；随后 post-hoc probe 的 DataLoader worker 被终止，不影响该 planning 结果，但探针需单独补跑。", 0.98, 6.18, 11.42, 0.28, size=12.2, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, "本地证据：.../seed42_paper_v2_20261027/planning_eval/epoch_0008/metrics.yaml。", 9)

    # 10. Cost-space ablation
    slide = add_blank(prs)
    add_title(slide, "Table 1/9 + Fig. 7：固定动力学，只改变目标代价空间", "ABLATION")
    add_image_contain(slide, ASSETS / "paper_cost_geometry.png", 0.60, 1.18, 6.64, 3.90, bg=WHITE)
    add_rect(slide, 7.53, 1.18, 5.08, 3.90, fill=WHITE, line=LINE, radius=True)
    add_label(slide, "为什么做", 7.83, 1.47, 1.05)
    add_text(slide, "完整层级规划同时改变：", 7.84, 2.00, 3.80, 0.30, size=16, bold=True)
    add_text(slide, "① 时间分解   ② 目标距离的潜空间", 7.84, 2.45, 4.10, 0.30, size=15, color=RED, bold=True)
    add_text(slide, "为隔离第②项，论文始终使用同一个 L1 world model 和同一 planner，只把 rollout 与 goal 投影到 L2/L3/L4 再计算距离。", 7.84, 2.96, 4.10, 1.15, size=13.5, color=MUTED, line_spacing=1.15)
    add_text(slide, "AntMaze 论文结果", 7.84, 4.34, 2.0, 0.28, size=13, color=BLUE, bold=True)
    add_text(slide, "L3 model：Native L1 16.7 → L3 cost 22.0 → Hierarchical 73.3", 7.84, 4.67, 4.18, 0.45, size=13.2, color=INK, bold=True)
    add_rect(slide, 0.72, 5.43, 11.89, 1.00, fill=PALE_AMBER, line=PALE_AMBER, radius=True)
    add_text(slide, "结论：抽象代价可以单独改善平坦规划，但无法解释全部增益；剩余差距来自时间分解。", 1.02, 5.65, 11.30, 0.31, size=16, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "本地状态：108 个 cost cells 已进入官方权重队列，初版不填未完成数值。", 1.02, 6.05, 11.30, 0.24, size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
    add_footer(slide, f"来源：{paper_url}，Table 1/9、Fig. 7；代码：hierarchical_solver.py:142–153, 412–465。", 10)

    # 11. Temporal decomposition
    slide = add_blank(prs)
    add_title(slide, "Fig. 8 / Table 10：时间分解让局部子目标更单调、更易优化", "ABLATION")
    add_text(slide, "同一条专家轨迹上，论文比较两种代价", 0.73, 1.20, 5.0, 0.35, size=19, bold=True)
    add_rect(slide, 0.73, 1.82, 5.52, 1.54, fill=PALE_RED, line=PALE_RED, radius=True)
    add_label(slide, "平坦目标", 1.02, 2.10, 1.05)
    add_text(slide, "直接从当前位置匹配最终 goal", 1.02, 2.59, 4.30, 0.30, size=16, bold=True)
    add_text(slide, "长时域代价可能停滞或上升", 1.02, 2.95, 4.30, 0.24, size=12.5, color=MUTED)
    add_rect(slide, 0.73, 3.72, 5.52, 1.54, fill=PALE_TEAL, line=PALE_TEAL, radius=True)
    add_label(slide, "层级子目标", 1.02, 4.00, 1.15, fill=TEAL, color=WHITE)
    add_text(slide, "每段只跟踪上层给出的下一个 subgoal", 1.02, 4.49, 4.55, 0.30, size=16, bold=True)
    add_text(slide, "搜索时域更短，代价更一致地下降", 1.02, 4.85, 4.55, 0.24, size=12.5, color=MUTED)
    add_line(slide, 6.70, 1.52, 6.70, 5.51, color=LINE, width=1.2)
    add_text(slide, "论文量化结果", 7.22, 1.25, 4.5, 0.35, size=19, bold=True)
    add_metric(slide, "0.89–1.00", "子目标跟踪单调性", 7.22, 1.96, 2.80, color=TEAL, sub="-Spearman(cost, time)")
    add_metric(slide, "0.48–0.75", "全时域 goal cost", 10.18, 1.96, 2.20, color=RED, sub="四个仿真环境")
    add_text(slide, "为什么 HWM 也会变好？", 7.22, 3.73, 4.0, 0.30, size=16, bold=True)
    add_text(slide, "HWM 没有独立的高层潜空间，但仍具备时间层级；它优于 LeWM，正是时间分解贡献的对照证据。", 7.22, 4.18, 4.95, 0.88, size=14, color=MUTED, line_spacing=1.15)
    add_rect(slide, 7.22, 5.36, 4.96, 0.65, fill=PALE_AMBER, line=PALE_AMBER, radius=True)
    add_text(slide, "精确复现受限：论文该实验使用 s₂=3，公开仿真权重为 s₂=2。", 7.48, 5.55, 4.44, 0.26, size=11.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, f"来源：{paper_url}，Fig. 8 / Table 10。", 11)

    # 12. DROID
    slide = add_blank(prs)
    add_title(slide, "Fig. 9：DROID 检验多场景泛化，并揭示 IDM 的必要性", "DROID")
    add_image_contain(slide, ASSETS / "paper_scene_diversity.png", 0.62, 1.13, 3.35, 2.44, bg=WHITE)
    add_text(slide, "为什么加入 IDM？", 0.73, 3.90, 2.35, 0.30, size=17, bold=True)
    add_text(slide, "DROID 的场景/光照/物体跨 episode 变化，表示容易只编码“场景身份”而忽略动作相关变化。IDM 强迫相邻状态可恢复动作，抑制 slow-feature collapse。", 0.73, 4.35, 3.15, 1.38, size=13.3, color=MUTED, line_spacing=1.12)
    add_image_contain(slide, ASSETS / "droid_reproduction.png", 4.25, 1.13, 8.35, 4.75, bg=WHITE)
    add_rect(slide, 4.42, 6.02, 8.05, 0.54, fill=PALE_BLUE, line=PALE_BLUE, radius=True)
    add_text(slide, "本机重评估：LeWM +0.00，HWM −0.91，H-JEPA −2.56 个百分点；排序仍为 H-JEPA > HWM > LeWM。", 4.68, 6.17, 7.54, 0.26, size=12, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, "本地证据：analyse/official_release_08fc8a21/droid_summary.csv；16 clips × 3 planner seeds × 3 train seeds。", 12)

    # 13. End-to-end pipeline
    slide = add_blank(prs)
    add_title(slide, "我是怎么复现的：从输入校验到严格聚合的完整链路", "PIPELINE")
    steps = [
        ("01", "数据与任务", "HDF5 + 50 固定任务\nSHA256 / manifest", BLUE),
        ("02", "配置固化", "Hydra config\nseed / stride / budget", PURPLE),
        ("03", "从头训练", "多层 end-to-end\npred + SIGReg (+IDM)", RED),
        ("04", "层级规划", "top-down subgoals\ngradient solver", TEAL),
        ("05", "后处理", "probe / decoder\nDROID Fréchet", AMBER),
        ("06", "严格聚合", "50 tasks / 16 clips\nmean ± SE", INK),
    ]
    for i, (num, title, desc, color) in enumerate(steps):
        x = 0.50 + i * 2.13
        add_rect(slide, x, 1.58, 1.72, 3.38, fill=WHITE, line=LINE, radius=True)
        add_rect(slide, x, 1.58, 1.72, 0.66, fill=color, line=color, radius=True)
        add_text(slide, num, x + 0.13, 1.78, 0.40, 0.25, size=14, color=WHITE, bold=True)
        add_text(slide, title, x + 0.50, 1.77, 1.06, 0.28, size=13.5, color=WHITE, bold=True, align=PP_ALIGN.RIGHT)
        add_text(slide, desc, x + 0.18, 2.62, 1.36, 1.15, size=13, color=INK, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, line_spacing=1.18)
        if i < len(steps) - 1:
            add_line(slide, x + 1.76, 3.26, x + 2.04, 3.26, color=RED, width=2.0, arrow=True)
    add_rect(slide, 0.70, 5.42, 11.94, 0.92, fill=INK, line=INK, radius=True)
    add_text(slide, "两条执行路径共享评估代码", 0.98, 5.67, 2.45, 0.27, size=14, color="F4B6B2", bold=True)
    add_text(slide, "本地自训产生 checkpoint；官方权重重评估直接从 checkpoint 起步。两者都进入同一规划、指标和聚合链路。", 3.38, 5.62, 8.76, 0.37, size=14.5, color=WHITE, bold=True)
    add_footer(slide, "关键入口：main_hjepa.py、eval.py、main_probing_decoding_eval.py、scripts/run_official_reproduction.py。", 13)

    # 14. Code mapping
    slide = add_blank(prs)
    add_title(slide, "论文方法与代码一一对照", "CODE MAP")
    rows = [
        ("逐层编码", "models/hjepa.py:240–294", "L1 图像编码后，逐层生成 embed/action"),
        ("逐层预测损失", "hjepa_utils.py:55–179", "每层 teacher forcing / rollout + SIGReg + IDM"),
        ("端到端训练", "main_hjepa.py:170–257", "Hydra 配置、seed、数据、模型与优化器"),
        ("顶向下规划", "hierarchical_solver.py:90–225", "上层 prediction 作为下层 subgoal"),
        ("固定任务评估", "planning_eval.py:300–545", "50 episode outcomes 与规划计算记录"),
        ("DROID 指标", "traj_metrics.py:19–53", "离散 Fréchet / zero-motion floor"),
        ("探针与解码", "final_probing_decoding_eval.py:217–529", "冻结表示、NMSE、post-hoc decoder"),
        ("完整性检查", "scripts/run_official_reproduction.py:148–171", "50 tasks / 16 clips / required metrics"),
    ]
    table_shape = slide.shapes.add_table(len(rows) + 1, 3, Inches(0.58), Inches(1.12), Inches(12.16), Inches(4.96))
    table = table_shape.table
    for i, width in enumerate([2.1, 4.0, 6.06]):
        table.columns[i].width = Inches(width)
    for c, val in enumerate(["论文概念", "代码位置", "复现中的作用"]):
        table.cell(0, c).text = val
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            table.cell(r, c).text = val
    style_table(table, body_size=11.1)
    for r in range(1, len(rows) + 1):
        table.cell(r, 1).text_frame.paragraphs[0].font.name = FONT_MONO
        table.cell(r, 1).text_frame.paragraphs[0].font.size = Pt(9.7)
        table.cell(r, 0).text_frame.paragraphs[0].font.bold = True
        table.cell(r, 0).text_frame.paragraphs[0].font.color.rgb = rgb(RED if r <= 4 else BLUE)
    add_rect(slide, 0.70, 6.30, 11.93, 0.42, fill=LIGHT, line=LIGHT, radius=True)
    add_text(slide, "复现不是重写算法：优先复用论文仓库入口，只在运行时、完整性校验和分析聚合处补齐工程缺口。", 0.95, 6.41, 11.46, 0.22, size=11.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, "以上行号对应当前工作区版本；PPT 生成时已复核。", 14)

    # 15. Engineering corrections
    slide = add_blank(prs)
    add_title(slide, "为保证结果可解释，我修正了哪些工程问题", "ENGINEERING")
    fixes = [
        ("训练协议", "恢复论文 batch=128、accumulation=1；调度器按 optimizer update 计数。", RED, PALE_RED),
        ("显存与恢复", "编码器 checkpointing / frame chunk；每 2000 steps 保存完整恢复状态。", BLUE, PALE_BLUE),
        ("结果完整性", "仿真必须 50 outcomes；DROID 必须 16 clips；探针必须包含全部 paper target。", TEAL, PALE_TEAL),
        ("权重来源", "官方权重 revision 固定并做 manifest/hash 校验；不把论文数字回填为本地结果。", AMBER, PALE_AMBER),
        ("结果隔离", "旧协议 b32×accum4、自训 protocol-v2、官方权重重评估分别存放和标注。", PURPLE, "EEEAF5"),
        ("失败处理", "训练后的 probe 中断不会覆盖 planning 指标；不完整 cell 自动重启而非“视为成功”。", INK, LIGHT),
    ]
    for i, (title, desc, color, pale) in enumerate(fixes):
        col = i % 2
        row = i // 2
        x = 0.70 + col * 6.20
        y = 1.24 + row * 1.67
        add_rect(slide, x, y, 5.78, 1.35, fill=WHITE, line=LINE, radius=True)
        add_rect(slide, x, y, 0.18, 1.35, fill=color, line=color)
        add_label(slide, title, x + 0.38, y + 0.23, 1.22, fill=pale, color=color)
        add_text(slide, desc, x + 1.72, y + 0.22, 3.76, 0.88, size=11.6, color=INK, bold=True, valign=MSO_ANCHOR.MIDDLE, line_spacing=1.05)
    add_rect(slide, 0.70, 6.37, 12.0, 0.35, fill=INK, line=INK, radius=True)
    add_text(slide, "目标不是“让数字更好看”，而是让每个数字都能追溯到配置、checkpoint、任务和聚合规则。", 0.98, 6.44, 11.45, 0.21, size=11.4, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, "工程验证记录：local_env/fixes_20261027/ 与 local_env/official_reproduction_20261027/。", 15)

    # 16. Current status
    slide = add_blank(prs)
    add_title(slide, "当前完成度与可交付边界", "STATUS")
    add_metric(slide, "1", "自训 protocol-v2 完整 planning cell", 0.75, 1.20, 3.2, color=RED, sub="FourRoom H-JEPA L2 / seed 42")
    add_metric(slide, "27/27", "DROID 官方权重 cells", 4.15, 1.20, 2.7, color=BLUE, sub="3 models × 3 train × 3 planner")
    add_metric(slide, f"{status['complete']}/{status['total']}", "官方发布矩阵总进度", 7.15, 1.20, 2.7, color=TEAL, sub="depth + cost + droid + probe")
    add_metric(slide, str(len(status["failures"])), "当前失败 cells", 10.35, 1.20, 2.0, color=AMBER, sub="动态 runner 自动续跑")
    add_line(slide, 0.73, 3.05, 12.55, 3.05, color=LINE)
    categories = [
        ("可精确复现", "Fig. 4 左/右、Fig. 6 下、Table 1/9、Fig. 9b 三个公开模型", TEAL, PALE_TEAL),
        ("有条件复现", "Fig. 7 anchor 未公开；Fig. 12a FFT 聚合规则未公开", AMBER, PALE_AMBER),
        ("公开材料不足", "Fig. 6 上、Fig. 8/10/11、部分附录消融缺 task/log/checkpoint", RED, PALE_RED),
    ]
    for i, (title, desc, color, pale) in enumerate(categories):
        y = 3.44 + i * 0.99
        add_rect(slide, 0.78, y, 11.82, 0.72, fill=pale, line=pale, radius=True)
        add_text(slide, title, 1.05, y + 0.21, 1.65, 0.25, size=13.5, color=color, bold=True)
        add_text(slide, desc, 2.88, y + 0.19, 9.22, 0.32, size=13, color=INK, bold=True)
    add_rect(slide, 0.78, 6.50, 11.82, 0.30, fill=INK, line=INK, radius=True)
    add_text(slide, "下一版优先更新：Fig. 6 深度矩阵 → Table 1/9 cost ladder → Fig. 4 三种子 probe。", 1.02, 6.55, 11.34, 0.20, size=10.8, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide, "当前状态读取自 analyse/official_release_08fc8a21/runner_status_droid_probe_depth_cost.json。", 16)

    # 17. Takeaways
    slide = add_blank(prs, WHITE)
    add_rect(slide, 0, 0, 13.333, 7.5, fill=INK, line=INK)
    add_rect(slide, 0.68, 0.72, 0.72, 0.08, fill=RED, line=RED)
    add_text(slide, "阶段性结论", 0.68, 1.15, 4.0, 0.56, size=31, color=WHITE, bold=True)
    takeaways = [
        ("01", "方法层面", "H-JEPA 的增益来自两个互补机制：抽象目标代价 + 时间分解。"),
        ("02", "复现层面", "FourRoom L2 自训 80.0%，与论文 81.3±6.7% 接近；仍需补三种子。"),
        ("03", "评估层面", "DROID 重评估保留模型排序，但 H-JEPA 与论文相差 −2.56pp，需要继续定位。"),
        ("04", "证据层面", "所有结果都按权重来源、seeds、任务数和完整性状态分层报告。"),
    ]
    for i, (num, title, desc) in enumerate(takeaways):
        y = 2.12 + i * 1.02
        add_text(slide, num, 0.72, y, 0.55, 0.36, size=18, color=RED, bold=True)
        add_text(slide, title, 1.43, y, 1.28, 0.34, size=16, color=WHITE, bold=True)
        add_text(slide, desc, 2.90, y, 8.90, 0.52, size=14.5, color="D6DEE4", bold=True)
    add_text(slide, "Q & A", 9.85, 6.36, 2.15, 0.45, size=24, color=WHITE, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, "附录：论文完整深度结果", 0.70, 6.75, 3.05, 0.24, size=10.5, color="8EA0AD")

    # 18. Appendix numeric table
    slide = add_blank(prs)
    add_title(slide, "附录：论文 Fig. 6 深度比较完整数值", "APPENDIX")
    headers = ["环境", "LeWM", "H-JEPA 2", "H-JEPA 3", "H-JEPA 4", "HWM 2", "HWM 3", "HWM 4"]
    data = [
        ("Visual AntMaze", "18.0±3.5", "39.3±3.7", "73.3±3.5", "63.3±7.7", "34.0±1.2", "38.7±3.5", "14.0±2.0"),
        ("FourRoom", "40.7±6.8", "81.3±6.7", "96.0±1.1", "80.0±4.0", "75.3±3.3", "99.3±0.7", "92.7±4.4"),
        ("OGBench Cube", "32.0±6.1", "47.3±2.7", "60.0±5.0", "55.3±4.4", "46.7±2.9", "53.3±5.2", "34.0±3.1"),
        ("Push-T", "40.0±2.3", "45.3±1.3", "17.3±3.5", "0.7±0.7", "42.0±6.4", "12.0±2.0", "0.0±0.0"),
    ]
    table_shape = slide.shapes.add_table(5, 8, Inches(0.48), Inches(1.22), Inches(12.36), Inches(3.45))
    table = table_shape.table
    widths = [2.05] + [1.47] * 7
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    for c, val in enumerate(headers):
        table.cell(0, c).text = val
    for r, row in enumerate(data, start=1):
        for c, val in enumerate(row):
            table.cell(r, c).text = val
    style_table(table, body_size=10.5)
    for r in range(1, 5):
        table.cell(r, 0).text_frame.paragraphs[0].font.bold = True
        for c in range(1, 8):
            if (r, c) in [(1, 3), (2, 6), (3, 3), (4, 2)]:
                table.cell(r, c).fill.fore_color.rgb = rgb(PALE_RED)
                table.cell(r, c).text_frame.paragraphs[0].font.color.rgb = rgb(RED)
                table.cell(r, c).text_frame.paragraphs[0].font.bold = True
    add_rect(slide, 0.70, 5.10, 11.94, 1.18, fill=WHITE, line=LINE, radius=True)
    add_text(slide, "读表要点", 0.98, 5.37, 1.30, 0.28, size=14, color=RED, bold=True)
    add_text(slide, "三层通常是甜点位；但最佳深度依赖数据。Push-T 的高层退化提醒我们：层级更深并非单调更好，clip 覆盖率是训练有效性的关键约束。", 2.28, 5.32, 9.76, 0.53, size=14, color=INK, bold=True)
    add_text(slide, "单位：规划成功率 %，mean ± SE，seeds 42/43/44；每环境 50 固定任务。", 2.28, 5.91, 9.76, 0.25, size=11, color=MUTED)
    add_footer(slide, "来源：docs/RESULTS.md（转录自论文）。", 18)

    prs.save(OUTPUT)
    return OUTPUT


def main():
    results = load_results()
    make_charts(results)
    output = build_deck(results)
    print(output)


if __name__ == "__main__":
    main()
