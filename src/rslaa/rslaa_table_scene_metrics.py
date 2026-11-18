# -*- coding: utf-8 -*-
"""方案 B 口径下逐场景感知指标统计，并输出可直接粘贴到论文的 Word 表格。

方案 B 指标（逐个体记录汇总，TP = matched==1 且 IoU3D > 0.15）
---------------------------------------------------------------
IoU  : TP 上 IoU3D 的均值
MR   : 1 - N_TP / N_matched（无效框率，越低越好）
Dev  : TP 上底面中心 XY 偏差 dev_m 的中位数 (m)

场景级聚合：把该场景内该方法组全部 run、全部交通条件的 TP 记录合并后统一计算
（IoU 按 TP 数加权、Dev 取合并后的中位数、MR 按总计数相除）。

用法：
    python -m rslaa.rslaa_table_scene_metrics --root "...\\Results"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rslaa.rslaa_plot_score_trends_n import (  # noqa: E402
    TP_IOU, RAW_NAME, parse_run, iter_eval_dirs,
)

GROUPS = ["static", "mpc", "marl_match", "marl_mismatch"]
GROUP_LABEL = {
    "static": "Static",
    "mpc": "MPC",
    "marl_match": "MARL_m",
    "marl_mismatch": "MARL_mm",
}


def scene_metrics(scene_dir: Path) -> dict[str, dict]:
    """统计单个场景内各方法组的方案 B 指标。"""
    acc = {g: dict(n_gt=0, n_matched=0, n_tp=0, sum_iou=0.0,
                   devs=[], runs=0) for g in GROUPS}

    for eval_dir in iter_eval_dirs(scene_dir):
        method, _w, _pos, _ang = parse_run(eval_dir)
        if method not in acc:
            continue
        raw = eval_dir / RAW_NAME
        try:
            df = pd.read_csv(raw, encoding="utf-8-sig",
                             usecols=["iou3d", "dev_m", "matched"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if df.empty:
            continue
        iou = df["iou3d"].to_numpy(dtype=float)
        dev = df["dev_m"].to_numpy(dtype=float)
        matched = df["matched"].to_numpy(dtype=float) > 0
        tp = matched & (iou > TP_IOU) & np.isfinite(iou)

        a = acc[method]
        a["runs"] += 1
        a["n_gt"] += int(iou.size)
        a["n_matched"] += int(matched.sum())
        a["n_tp"] += int(tp.sum())
        a["sum_iou"] += float(np.nansum(iou[tp]))
        if tp.any():
            a["devs"].append(dev[tp])

    out = {}
    for g in GROUPS:
        a = acc[g]
        if a["runs"] == 0:
            out[g] = dict(iou=np.nan, mr=np.nan, dev=np.nan, n_tp=0, runs=0)
            continue
        n_tp, n_matched = a["n_tp"], a["n_matched"]
        iou = a["sum_iou"] / n_tp if n_tp else np.nan
        mr = 1.0 - n_tp / max(n_matched, 1)
        dev_vals = np.concatenate(a["devs"]) if a["devs"] else np.array([])
        dev_vals = dev_vals[np.isfinite(dev_vals)]
        dev = float(np.median(dev_vals)) if dev_vals.size else np.nan
        out[g] = dict(iou=iou, mr=mr, dev=dev, n_tp=n_tp, runs=a["runs"])
    return out


# ---------------------------------------------------------------- Word 输出
def _set_cell_borders(cell, **edges):
    """给单元格设置边框，edges 形如 top={"sz": 12}，sz 单位为 1/8 pt。"""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, spec in edges.items():
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), spec.get("val", "single"))
        el.set(qn("w:sz"), str(spec.get("sz", 8)))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), spec.get("color", "000000"))


def _table_three_line(table):
    """三线表：去掉全部内框与竖线，只保留顶线与底线。"""
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "12")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for edge in ("left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto")
        borders.append(el)
    tblPr.append(borders)


def _write(cell, text, bold=False, size=9.0, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _fmt(v, nd):
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}"


def build_docx(rows: list[tuple[str, dict]], out_path: Path,
               n_frames_note: str = ""):
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)

    head = doc.add_paragraph()
    r = head.add_run("Per-scenario perception performance under static and "
                     "adaptive LiDAR configurations")
    r.bold = True
    r.font.size = Pt(11)
    r.font.name = "Times New Roman"

    # 表头两行：三大列 + 四个子列
    n_cols = 1 + 3 * len(GROUPS)
    table = doc.add_table(rows=2 + len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    _table_three_line(table)

    table.cell(0, 0).merge(table.cell(1, 0))
    _write(table.cell(0, 0), "Scenario", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)

    groups = [("IoU", 0), ("MR (%)", 1), ("Dev (m)", 2)]
    for title, gi in groups:
        c0 = gi * len(GROUPS) + 1
        merged = table.cell(0, c0).merge(table.cell(0, c0 + len(GROUPS) - 1))
        _write(merged, title, bold=True)
        for j, g in enumerate(GROUPS):
            _write(table.cell(1, c0 + j), GROUP_LABEL[g], bold=True)

    for k, (scene, m) in enumerate(rows):
        r_i = 2 + k
        _write(table.cell(r_i, 0), scene, align=WD_ALIGN_PARAGRAPH.LEFT)
        for gi, nd in enumerate((3, 2, 3)):
            for j, g in enumerate(GROUPS):
                v = m[g]
                val = {"iou": v["iou"], "mr": v["mr"] * (100.0 if gi == 1 else 1.0),
                       "dev": v["dev"]}[("iou", "mr", "dev")[gi]]
                _write(table.cell(r_i, gi * len(GROUPS) + 1 + j), _fmt(val, nd))

    # 表头下加一条细线
    for c in range(n_cols):
        _set_cell_borders(table.cell(1, c), bottom={"sz": 6})

    doc.add_paragraph()
    note = doc.add_paragraph()
    note.paragraph_format.space_after = Pt(4)
    txt = (
        "Notes: TP = a matched object with 3D IoU > 0.15. IoU is the mean 3D "
        "IoU over TPs; Dev is the median planar (XY) centroid deviation over "
        "TPs; MR = 1 - N_TP / N_matched is the invalid-box rate (lower is "
        "better). All values are pooled over every run and every traffic "
        "condition within a scene, with equal treatment of all frames"
        + (f" ({n_frames_note})" if n_frames_note else "")
        + ". Static denotes the fixed LiDAR configuration; MPC the "
        "model-predictive adaptive policy; MARL_m and MARL_mm the MARL agents "
        "whose training traffic profile matches and mismatches the evaluated "
        "scene, respectively."
    )
    run = note.add_run(txt)
    run.font.size = Pt(8)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Scheme-B per-scene metrics and Word table")
    parser.add_argument("--root", default="Results")
    parser.add_argument(
        "--out",
        default="RSLaA_scene_metrics_table.docx")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    scenes = sorted(p for p in root.iterdir()
                    if p.is_dir() and re.fullmatch(r"scene_\d+", p.name))
    if not scenes:
        raise SystemExit(f"{root} 下未找到 scene_xxx 目录")

    rows = []
    print(f"{'scene':10s} {'method':12s} {'runs':>5s} {'N_TP':>10s} "
          f"{'IoU':>7s} {'MR':>7s} {'Dev':>7s}")
    for scene_dir in scenes:
        m = scene_metrics(scene_dir)
        rows.append((scene_dir.name, m))
        for g in GROUPS:
            v = m[g]
            print(f"{scene_dir.name:10s} {g:12s} {v['runs']:5d} {v['n_tp']:10d} "
                  f"{v['iou']:7.4f} {v['mr']:7.4f} {v['dev']:7.4f}")
        print()

    out = Path(args.out)
    build_docx(rows, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
