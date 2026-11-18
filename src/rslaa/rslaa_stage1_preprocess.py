# -*- coding: utf-8 -*-
"""RSLaA 第二阶段 —— 脚本 1：道路静态环境点云预处理与静态参数提取.

功能
----
1) 语义清洗 (原地覆盖源文件)
   - whu_urban3D (.h5): 删除 semantics ∈ {100500, 100600} 的点
   - kitti360    (.ply): 不做任何删除 (用户已离线处理完毕, DROP_LABELS 为空)
2) 静态参数提取 (每个地点一份 Results/static_params/scene_xxx.json)
   - 道路: 车道数、车道宽、道路宽度、曲率/最小半径
   - 交通包络: 轨迹覆盖宽度、中心线长度
   - 路侧障碍物: whu {100300,100100,101600,101500,105400},
                 kitti360 {11,17,21,44,37}
                 平均高度、沿道路纵向离散程度
   估算不到的量一律写 null, 由人工后续补充 (notes 中给出原因).

用法
----
    python -m rslaa.rslaa_stage1_preprocess --all
    python -m rslaa.rslaa_stage1_preprocess --sites scene_001,scene_006
    python -m rslaa.rslaa_stage1_preprocess --all --skip-clean      # 跳过语义清洗
    python -m rslaa.rslaa_stage1_preprocess --all --skip-static     # 仅清洗
    python -m rslaa.rslaa_stage1_preprocess --all --dry-run         # 只统计不写盘
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# 全局配置 (三份脚本各自内置一份, 便于单独拷走运行)
# ---------------------------------------------------------------------------
from rslaa.paths import MODELS_DIR, REPO_ROOT  # noqa: E402

RESULTS_DIR = os.environ.get("RSLAA_RESULTS", str(REPO_ROOT / "Results"))
PC_ROOT = os.environ.get("RSLAA_POINTCLOUD", str(REPO_ROOT / "data" / "pointclouds"))
TRAJ_ROOT = os.environ.get("RSLAA_TRAJECTORIES", str(REPO_ROOT / "data" / "trajectories"))

# 10 个地点: 前 5 个 whu_urban3D, 后 5 个 kitti360
SITES = [
    dict(id="scene_001", name="whu_0940", family="whu",
         env=os.path.join(PC_ROOT, r"whu_urban3D\mls\h5\0940.h5"),
         traj_dir=os.path.join(TRAJ_ROOT, r"whu_3D\0940"), stem="参数 ({i})"),
    dict(id="scene_002", name="whu_2421", family="whu",
         env=os.path.join(PC_ROOT, r"whu_urban3D\mls\h5\2421.h5"),
         traj_dir=os.path.join(TRAJ_ROOT, r"whu_3D\2421"), stem="参数 ({i})"),
    dict(id="scene_003", name="whu_4629", family="whu",
         env=os.path.join(PC_ROOT, r"whu_urban3D\mls\h5\4629.h5"),
         traj_dir=os.path.join(TRAJ_ROOT, r"whu_3D\4629"), stem="参数 ({i})"),
    dict(id="scene_004", name="whu_5627", family="whu",
         env=os.path.join(PC_ROOT, r"whu_urban3D\mls\h5\5627.h5"),
         traj_dir=os.path.join(TRAJ_ROOT, r"whu_3D\5627"), stem="参数 ({i})轨迹数据",
         # 参数 (6) 轨迹数据位于另一路段 (与点云重合度 0%), 该交通场景放弃
         skip=[6]),
    dict(id="scene_005", name="whu_8048", family="whu",
         env=os.path.join(PC_ROOT, r"whu_urban3D\mls\h5\8048.h5"),
         traj_dir=os.path.join(TRAJ_ROOT, r"whu_3D\8048"), stem="参数 ({i})轨迹数据"),
    dict(id="scene_006", name="kitti_0000", family="kitti360",
         env=os.path.join(PC_ROOT, r"kitti360\2013_05_28_drive_0000_sync"
                          r"\0000001740_0000001991.ply"),
         traj_dir=os.path.join(TRAJ_ROOT, r"kitti360\2013_05_28_drive_0000_sync"
                               r"\0000001740_0000001991"), stem="交通量 ({i})"),
    dict(id="scene_007", name="kitti_0002", family="kitti360",
         env=os.path.join(PC_ROOT, r"kitti360\2013_05_28_drive_0002_sync"
                          r"\0000006069_0000006398.ply"),
         traj_dir=os.path.join(TRAJ_ROOT, r"kitti360\2013_05_28_drive_0002_sync"
                               r"\0000006069_0000006398"), stem="交通量 ({i})"),
    dict(id="scene_008", name="kitti_0003", family="kitti360",
         env=os.path.join(PC_ROOT, r"kitti360\2013_05_28_drive_0003_sync"
                          r"\0000000731_0000000893.ply"),
         traj_dir=os.path.join(TRAJ_ROOT, r"kitti360\2013_05_28_drive_0003_sync"
                               r"\0000000731_0000000893"), stem="交通量 ({i})"),
    dict(id="scene_009", name="kitti_0004", family="kitti360",
         env=os.path.join(PC_ROOT, r"kitti360\2013_05_28_drive_0004_sync"
                          r"\0000003570_0000003975.ply"),
         traj_dir=os.path.join(TRAJ_ROOT, r"kitti360\2013_05_28_drive_0004_sync"
                               r"\0000003570_0000003975"), stem="交通量 ({i})"),
    dict(id="scene_010", name="kitti_0005", family="kitti360",
         env=os.path.join(PC_ROOT, r"kitti360\2013_05_28_drive_0005_sync"
                          r"\0000003501_0000003711.ply"),
         traj_dir=os.path.join(TRAJ_ROOT, r"kitti360\2013_05_28_drive_0005_sync"
                               r"\0000003501_0000003711"), stem="交通量 ({i})"),
]


def traffic_indices(site: dict) -> list:
    """该地点实际使用的交通场景序号 (排除 skip 中登记的问题数据)."""
    skip = set(site.get("skip", []) or [])
    return [k for k in range(1, 10) if k not in skip]

# 语义 label 配置 (空列表 = 该数据集不做任何删除)
DROP_LABELS = {
    "whu": [100500, 100600],
    "kitti360": [],          # kitti360 点云已由用户离线处理, 不再删除任何点
}
OBSTACLE_LABELS = {
    "whu": [100300, 100100, 101600, 101500, 105400],
    "kitti360": [11, 17, 21, 44, 37],
}
# 道路/地面 label (仅用于辅助判断, 不参与清洗)
ROAD_LABELS = {"whu": [102000], "kitti360": [13, 21]}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. 语义清洗
# ---------------------------------------------------------------------------
def clean_h5(path: str, drop_labels, dry_run=False) -> dict:
    """删除 semantics 属于 drop_labels 的点, 原地覆盖 (先写 .tmp 再替换)."""
    import h5py

    with h5py.File(path, "r") as f:
        if "semantics" not in f:
            raise RuntimeError(f"{path} 中未找到 semantics 数据集")
        sem = f["semantics"][:]
        n0 = int(sem.size)
        keep = ~np.isin(sem, np.asarray(drop_labels, np.int64))
        n1 = int(keep.sum())
        removed = n0 - n1
        info = dict(path=path, n_before=n0, n_after=n1, n_removed=removed,
                    drop_labels=list(drop_labels))
        log(f"  [h5] {os.path.basename(path)}: {n0} -> {n1} (-{removed})")
        if dry_run or removed == 0:
            return info

        tmp = path + ".tmp"
        with h5py.File(tmp, "w") as g:
            for k in f.keys():
                d = f[k]
                data = d[:][keep] if d.shape[:1] == (n0,) else d[:]
                kwargs = {}
                if d.compression:
                    kwargs["compression"] = d.compression
                    kwargs["compression_opts"] = d.compression_opts
                    kwargs["chunks"] = True
                g.create_dataset(k, data=data, dtype=d.dtype, **kwargs)
    os.replace(tmp, path)
    return info


_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "double": "f8", "float32": "f4", "float64": "f8",
}


def clean_ply(path: str, drop_labels, dry_run=False) -> dict:
    """按 semantic 字段过滤 PLY, 原地覆盖; 完整保留其余属性与头部格式."""
    with open(path, "rb") as f:
        head = f.read(65536)
    hend = head.find(b"end_header")
    if hend < 0:
        raise RuntimeError(f"PLY 头部解析失败: {path}")
    lines = head[:hend].decode("ascii", "ignore").strip().splitlines()
    fmt, nvert, props, header_tail = None, 0, [], []
    for ln in lines:
        tk = ln.split()
        if not tk:
            continue
        if tk[0] == "format":
            fmt = tk[1]
        elif tk[0] == "element" and len(tk) >= 3 and tk[1] == "vertex":
            nvert = int(tk[2])
        elif tk[0] == "property" and len(tk) >= 3:
            props.append((tk[2], _PLY_TYPES[tk[1]]))
        header_tail.append(ln)
    if fmt != "binary_little_endian":
        raise RuntimeError(f"仅支持 binary_little_endian PLY: {fmt}")
    names = [p[0] for p in props]
    if "semantic" not in names:
        raise RuntimeError(f"{path} 中未找到 semantic 属性")

    off = hend + len(b"end_header")
    while off < len(head) and head[off:off + 1] in (b"\n", b"\r"):
        off += 1
    dt = np.dtype(props)
    with open(path, "rb") as f:
        f.seek(off)
        arr = np.fromfile(f, dtype=dt, count=nvert)

    keep = ~np.isin(arr["semantic"].astype(np.int64),
                    np.asarray(drop_labels, np.int64))
    n1 = int(keep.sum())
    removed = nvert - n1
    info = dict(path=path, n_before=nvert, n_after=n1, n_removed=removed,
                drop_labels=list(drop_labels))
    log(f"  [ply] {os.path.basename(path)}: {nvert} -> {n1} (-{removed})")
    if dry_run or removed == 0:
        return info

    new_tail = []
    for ln in header_tail:
        tk = ln.split()
        if tk[0] == "element" and len(tk) >= 3 and tk[1] == "vertex":
            new_tail.append(f"element vertex {n1}")
        else:
            new_tail.append(ln)
    new_head = ("\n".join(new_tail) + "\nend_header\n").encode("ascii")

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new_head)
        arr[keep].tofile(f)
    os.replace(tmp, path)
    return info


def do_clean(site: dict, dry_run=False, backup=False) -> dict:
    fam = site["family"]
    drop = DROP_LABELS[fam]
    env = site["env"]
    if not os.path.isfile(env):
        raise FileNotFoundError(env)
    if not drop:
        log(f"[清洗] {site['id']} ({site['name']}): 该数据集不做语义删除 "
            f"(用户已离线处理), 跳过且不改写文件")
        return dict(path=env, n_before=None, n_after=None, n_removed=0,
                    drop_labels=[], skipped=True)
    if backup:
        bak = env + ".bak"
        if not os.path.exists(bak):
            log(f"  备份 -> {os.path.basename(bak)}")
            shutil.copy2(env, bak)
    log(f"[清洗] {site['id']} ({site['name']}) drop={drop}")
    if env.lower().endswith(".h5"):
        return clean_h5(env, drop, dry_run=dry_run)
    return clean_ply(env, drop, dry_run=dry_run)


# ---------------------------------------------------------------------------
# 2. 静态参数提取所需的几何工具
# ---------------------------------------------------------------------------
def _load_traj_records(path: str) -> np.ndarray:
    rec = np.loadtxt(path, encoding="utf-8-sig")
    if rec.ndim == 1:
        rec = rec.reshape(1, -1)
    return rec


def _longest_track_centerline(records: np.ndarray, ds: float = 0.5):
    """取采样点最多的车辆轨迹作为参考中心线, 并按 ds 重采样."""
    t, vid = records[:, 0], records[:, 3].astype(np.int64)
    fx, fy = records[:, 6], records[:, 7]
    best, best_n = None, -1
    for v in np.unique(vid):
        m = vid == v
        if m.sum() > best_n:
            best_n, best = int(m.sum()), v
    m = vid == best
    order = np.argsort(t[m], kind="stable")
    pts = np.stack([fx[m][order], fy[m][order]], 1)
    # 去掉重复点
    keep = np.ones(len(pts), bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3
    pts = pts[keep]
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / ds), 2)
    sq = np.linspace(0, s[-1], n)
    cx = np.interp(sq, s, pts[:, 0])
    cy = np.interp(sq, s, pts[:, 1])
    cl = np.stack([cx, cy], 1)
    return dict(points=cl, s=sq, length=float(s[-1]), ref_vid=int(best))


def _centerline_frame(cl: dict):
    """返回 (点, 单位切线, 单位法向) 供横向/纵向投影."""
    p = cl["points"]
    seg = np.gradient(p, axis=0)
    nrm = np.linalg.norm(seg, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    tang = seg / nrm
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    return p, tang, normal


def _project_lateral(points_xy: np.ndarray, cl: dict):
    """把 (M,2) 点投到中心线: 返回 (有符号横向偏移 d, 纵向弧长 s)."""
    from scipy.spatial import cKDTree
    p = cl["points"]
    _, _, normal = _centerline_frame(cl)
    tree = cKDTree(p)
    _, idx = tree.query(points_xy, k=1)
    diff = points_xy - p[idx]
    d = np.einsum("ij,ij->i", diff, normal[idx])
    s = cl["s"][idx]
    return d, s


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    ker = np.ones(k, np.float64) / k
    return np.convolve(x, ker, mode="same")


def _estimate_curvature(cl: dict) -> dict:
    p = cl["points"]
    ds = float(cl["s"][1] - cl["s"][0]) if len(cl["s"]) > 1 else 0.5
    d = np.gradient(p, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kappa = np.gradient(heading) / max(ds, 1e-6)
    kappa = _smooth(kappa, max(int(3.0 / max(ds, 1e-6)), 1))
    absk = np.abs(kappa)
    with np.errstate(divide="ignore"):
        radius = np.where(absk > 1e-6, 1.0 / absk, np.inf)
    length = float(cl["length"])
    seg_len = np.gradient(cl["s"])
    curved = radius < 200.0
    curved_frac = float(seg_len[curved].sum() / max(seg_len.sum(), 1e-9))
    finite = radius[np.isfinite(radius)]
    out = dict(
        radius_min_m=None, radius_median_m=None,
        curved_fraction=round(curved_frac, 4),
        is_curved=bool(curved_frac > 0.30),
        length_m=round(length, 2),
    )
    if finite.size:
        out["radius_min_m"] = round(float(np.percentile(finite, 5)), 2)
        out["radius_median_m"] = round(float(np.median(finite)), 2)
    return out


def _detect_lanes(d: np.ndarray) -> dict:
    """由横向偏移直方图峰位估计车道中心、车道数与车道宽."""
    if d.size < 200:
        return dict(lane_count=None, lane_width_m=None, lane_centers=None,
                    note="轨迹横向样本不足")
    lo, hi = -30.0, 30.0
    dd = d[(d >= lo) & (d <= hi)]
    if dd.size < 200:
        return dict(lane_count=None, lane_width_m=None, lane_centers=None,
                    note="横向偏移超出 ±30m 的样本过多")
    binw = 0.25
    bins = np.arange(lo, hi + binw, binw)
    hist, edges = np.histogram(dd, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = _smooth(hist.astype(np.float64), 5)
    thr = 0.15 * hs.max()
    peaks = []
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] > thr:
            peaks.append(i)
    # 合并过近峰 (间距 < 2.5m)
    merged = []
    for i in peaks:
        if merged and centers[i] - centers[merged[-1]] < 2.5:
            if hs[i] > hs[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    if len(merged) < 2:
        return dict(lane_count=None, lane_width_m=None,
                    lane_centers=[round(float(centers[i]), 2) for i in merged],
                    note="峰位不足 2 个, 无法判定车道结构")
    lc = centers[merged]
    gaps = np.diff(lc)
    lane_w = float(np.median(gaps))
    centers_out = [round(float(x), 2) for x in lc]
    if not (2.2 <= lane_w <= 5.2):
        return dict(lane_count=None, lane_width_m=None, lane_centers=centers_out,
                    note=f"车道宽估计 {lane_w:.2f} m 超出合理范围, 已弃用")
    note = ""
    if len(lc) > 8:
        note = f"检测到 {len(lc)} 条车道中心, 疑似交叉口, 道路宽度仅供参考"
    return dict(lane_count=int(len(lc)), lane_width_m=round(lane_w, 2),
                lane_centers=centers_out, note=note)


def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def compute_static_params(site: dict) -> dict:
    """估算静态参数; 不可估算项为 null, 并在 notes 中说明."""
    from lidarsim.pointcloud import load_env, GroundModel

    notes = []
    env = load_env(site["env"])
    pts = np.asarray(env.points, np.float64)
    sem = env.semantics
    log(f"  环境点云: {len(pts)} 点, semantics={'有' if sem is not None else '无'}")

    sub = pts[:: max(1, len(pts) // 1_500_000)]
    gm = GroundModel(sub, cell=1.0)
    gz = gm.height(pts[:, :2])
    h_above = pts[:, 2] - gz

    # ---- 轨迹 ----
    files = []
    for i in traffic_indices(site):
        p = os.path.join(site["traj_dir"], site["stem"].format(i=i) + ".txt")
        if os.path.isfile(p):
            files.append(p)
        else:
            notes.append(f"缺少轨迹文件 参数 ({i})")
    if not files:
        raise FileNotFoundError(f"{site['traj_dir']} 下未找到轨迹文件")
    log(f"  轨迹文件: {len(files)} 个")
    if site.get("skip"):
        notes.append(f"按配置跳过交通场景序号 {site['skip']} (数据与点云不匹配)")

    rec0 = _load_traj_records(files[0])
    cl = _longest_track_centerline(rec0)
    if cl is None:
        raise RuntimeError("无法构建参考中心线")

    # 多文件横向偏移抽样 (用于车道识别)
    lat_samples = []
    for p in files:
        rec = _load_traj_records(p)
        m = (rec[:, 0] % 5) < 1e-6          # 每 5 s 取一帧
        xy = rec[m][:, 6:8]
        if len(xy) == 0:
            continue
        d, _ = _project_lateral(xy, cl)
        lat_samples.append(d[np.abs(d) < 30])
    d_all = np.concatenate(lat_samples) if lat_samples else np.array([])

    lanes = _detect_lanes(d_all)
    curv = _estimate_curvature(cl)

    # ---- 交通包络 ----
    env_width = None
    if d_all.size:
        env_width = round(float(np.percentile(np.abs(d_all), 98) * 2.0), 2)

    # ---- 道路宽度 ----
    road_width = None
    width_source = "none"
    if lanes.get("lane_count") and lanes.get("lane_width_m"):
        road_width = round(lanes["lane_count"] * lanes["lane_width_m"], 2)
        width_source = "lane_count_x_lane_width"
    elif env_width:
        road_width = env_width
        width_source = "traffic_envelope_p98"

    # ---- 点云地面横向范围 (辅助验证) ----
    env_ground_width = None
    if sem is not None:
        gmask = h_above < 0.25
        gp = pts[gmask]
        if len(gp) > 300_000:
            gp = gp[:: max(1, len(gp) // 300_000)]
        if len(gp):
            dg, sg = _project_lateral(gp[:, :2], cl)
            sel = (np.abs(dg) < 40) & (sg > 0) & (sg < cl["length"])
            if sel.sum() > 1000:
                env_ground_width = round(
                    float(np.percentile(dg[sel], 98) -
                          np.percentile(dg[sel], 2)), 2)

    # ---- 路侧障碍物 ----
    obs_labels = OBSTACLE_LABELS[site["family"]]
    obstacle = dict(labels=obs_labels, count=None, mean_height_m=None,
                    height_p90_m=None, longitudinal_std_m=None,
                    longitudinal_iqr_m=None, longitudinal_range_m=None,
                    density_per_100m=None)
    present = []
    if sem is not None:
        mask = np.isin(sem.astype(np.int64), np.asarray(obs_labels, np.int64))
        present = sorted(set(sem[mask].astype(np.int64).tolist()))
        idx = np.flatnonzero(mask)
        if idx.size:
            op = pts[idx]
            oh = h_above[idx]
            good = oh > 0.2                  # 去掉贴地残余点
            op, oh = op[good], oh[good]
            if oh.size:
                os_xy = op[:, :2]
                _, os_s = _project_lateral(os_xy, cl)
                obs_s = os_s[(os_s > 0) & (os_s < cl["length"])]
                obstacle.update(
                    count=int(oh.size),
                    mean_height_m=round(float(np.mean(oh)), 3),
                    height_p90_m=round(float(np.percentile(oh, 90)), 3),
                    density_per_100m=round(
                        float(oh.size) / max(cl["length"] / 100.0, 1e-6), 2),
                )
                if obs_s.size > 10:
                    obstacle.update(
                        longitudinal_std_m=round(float(np.std(obs_s)), 2),
                        longitudinal_iqr_m=round(
                            float(np.percentile(obs_s, 75) -
                                  np.percentile(obs_s, 25)), 2),
                        longitudinal_range_m=round(
                            float(obs_s.max() - obs_s.min()), 2),
                    )
    else:
        notes.append("环境点云缺少语义字段, 未提取路侧障碍物")

    if not present:
        notes.append("指定障碍物 label 在当前点云中未出现")
    if lanes.get("note"):
        notes.append("车道识别: " + lanes["note"])
    if curv["curved_fraction"] < 0.05:
        notes.append("道路近似直线, 曲率半径参考意义有限")
    if road_width is None:
        notes.append("道路宽度无法估算, 需人工补充")

    out = dict(
        scene=site["id"], site_name=site["name"], family=site["family"],
        env_path=site["env"], traj_example=files[0],
        road=dict(
            road_width_m=road_width, width_source=width_source,
            lane_count=lanes.get("lane_count"),
            lane_width_m=lanes.get("lane_width_m"),
            lane_centers_m=lanes.get("lane_centers"),
            env_ground_width_m=env_ground_width,
            radius_min_m=curv["radius_min_m"],
            radius_median_m=curv["radius_median_m"],
            curved_fraction=curv["curved_fraction"],
            is_curved=curv["is_curved"],
            centerline_length_m=curv["length_m"],
        ),
        traffic_envelope=dict(width_m=env_width,
                              centerline_length_m=curv["length_m"]),
        roadside_obstacle=obstacle,
        ground=dict(z_min=round(float(gz.min()), 3),
                    z_max=round(float(gz.max()), 3),
                    z_mean=round(float(gz.mean()), 3)),
        env=dict(n_points=int(len(pts)),
                 x_min=round(float(pts[:, 0].min()), 2),
                 x_max=round(float(pts[:, 0].max()), 2),
                 y_min=round(float(pts[:, 1].min()), 2),
                 y_max=round(float(pts[:, 1].max()), 2),
                 labels_present=[int(x) for x in
                                 np.unique(sem.astype(np.int64)).tolist()]
                 if sem is not None else None),
        notes=notes,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="RSLaA 第一阶段: 预处理 + 静态参数")
    ap.add_argument("--sites", default="", help="逗号分隔, 如 scene_001,scene_006")
    ap.add_argument("--all", action="store_true", help="处理全部 10 个地点")
    ap.add_argument("--skip-clean", action="store_true", help="跳过语义清洗")
    ap.add_argument("--skip-static", action="store_true", help="跳过静态参数提取")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--backup", action="store_true",
                    help="清洗前生成 .bak 备份 (占磁盘)")
    ap.add_argument("--force", action="store_true", help="已有静态参数也重算")
    return ap.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(argv)
    if args.all and not args.sites:
        sites = SITES
    elif args.sites:
        want = [s.strip() for s in args.sites.split(",") if s.strip()]
        sites = [s for s in SITES if s["id"] in want or s["name"] in want]
        if not sites:
            raise SystemExit(f"未匹配到地点: {want}")
    else:
        raise SystemExit("请指定 --all 或 --sites")

    out_root = os.path.join(args.results_dir, "static_params")
    os.makedirs(out_root, exist_ok=True)
    clean_log, static_log = [], []

    for site in sites:
        log("=" * 70)
        log(f"{site['id']}  {site['name']}  [{site['family']}]")
        if not args.skip_clean:
            try:
                clean_log.append(do_clean(site, dry_run=args.dry_run,
                                          backup=args.backup))
            except Exception as e:
                log(f"  [清洗失败] {e}")
                clean_log.append(dict(path=site["env"], error=str(e)))

        if args.skip_static:
            continue
        dst = os.path.join(out_root, site["id"] + ".json")
        if os.path.isfile(dst) and not args.force:
            log(f"  已存在静态参数, 跳过: {dst}")
            with open(dst, "r", encoding="utf-8") as f:
                static_log.append(json.load(f))
            continue
        try:
            info = compute_static_params(site)
            static_log.append(info)
            if not args.dry_run:
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                log(f"  已保存: {dst}")
        except Exception as e:
            log(f"  [静态参数失败] {e}")
            static_log.append(dict(scene=site["id"], error=str(e)))

    if args.dry_run:
        log("\n--dry-run: 未写入任何文件")
        return

    # 清洗记录按点云路径合并 (只跑部分站点时不丢其他站点的历史记录)
    rep_path = os.path.join(out_root, "_clean_report.json")
    merged = {}
    if os.path.isfile(rep_path):
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                for e in json.load(f):
                    merged[e.get("path")] = e
        except Exception:
            merged = {}
    for e in clean_log:
        old = merged.get(e.get("path"))
        # 历史记录里确实删过点, 而本次复检为 0 时保留历史
        if old and old.get("n_removed") and not e.get("n_removed"):
            continue
        merged[e.get("path")] = e
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump([merged[k] for k in sorted(merged)], f,
                  ensure_ascii=False, indent=2)

    # 汇总 CSV
    rows = []
    for s in static_log:
        if "road" not in s:
            continue
        r, o = s["road"], s["roadside_obstacle"]
        rows.append([
            s["scene"], s["site_name"], s["family"],
            r.get("road_width_m"), r.get("lane_count"), r.get("lane_width_m"),
            r.get("radius_min_m"), r.get("radius_median_m"),
            r.get("curved_fraction"),
            o.get("mean_height_m"), o.get("longitudinal_std_m"),
            o.get("count"),
        ])
    csv_path = os.path.join(out_root, "static_params_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["scene", "site_name", "family", "road_width_m", "lane_count",
                    "lane_width_m", "radius_min_m", "radius_median_m",
                    "curved_fraction", "obstacle_mean_height_m",
                    "obstacle_long_std_m", "obstacle_points"])
        w.writerows(rows)
    log(f"\n汇总已保存: {csv_path}")
    log("阶段 1 完成。")


if __name__ == "__main__":
    main()
