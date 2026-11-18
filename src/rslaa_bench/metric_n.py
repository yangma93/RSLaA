# -*- coding: utf-8 -*-
"""替代指标 v2: rslaa_bench/metric_n.py (修复版 + 批量高效计算).

保持 rslaa_bench/metrics.py **不变**, 仅在此文件重新定义 5 个指标 + 注册表.

根因修复
--------
  (1) ROI 掩码改用"任务区域"(轨迹走廊), 而非"历史 POG 几何可观测区域"(近全盘).
  (2) 体素级权重 w 也带 FoV 掩码, 与 q_vox 有效集严格对齐.

效率优化 (针对 benchmark 一次评多个候选位姿的场景)
--------------------------------------------------
  * compute_all_fast(): 单次 soft_visibility + 一次 task_weights, 同时算
    出 5 个指标, 比逐指标调用节省 4x 的可见度计算.
  * prep_context(): 在多候选场景下预计算与位姿无关的量 (H, d, corridors,
    roi_flat, col_index 等), 仅在每个候选上跑可见度 + 权重.
"""
from __future__ import annotations

import numpy as np
from rslaa_bench.metrics import (
    OccGrid,
    soft_visibility as _soft_visibility_orig,
    union_q,
    bern_entropy,
    _dyn_weight as _dyn_weight_orig,
    extract_corridors,
)

ROAD_HALF_M = 5.0       # 任务 ROI 道路半宽 (m)


# ---------------------------------------------------------------------------
# 任务 ROI 构造 (BEV 柱级掩码)
# ---------------------------------------------------------------------------
def build_task_roi(grid: OccGrid, traj_front: np.ndarray,
                   road_half_m: float = ROAD_HALF_M) -> np.ndarray:
    """由轨迹构造 BEV 任务区域 (nx,ny) bool: 轨迹散点 + 横向膨胀."""
    traj_front = np.asarray(traj_front, np.float64)
    traj_xy = traj_front[:, :2] if traj_front.ndim == 2 \
        else traj_front[:2].reshape(1, -1)
    xi = ((traj_xy[:, 0] - grid.origin[0]) / grid.voxel).astype(np.int64)
    yi = ((traj_xy[:, 1] - grid.origin[1]) / grid.voxel).astype(np.int64)
    nx, ny, _ = grid.shape
    mask = np.zeros((nx, ny), bool)
    ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    mask[xi[ok], yi[ok]] = True
    k_half = max(1, int(np.round(road_half_m / grid.voxel)))
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, np.ones((2 * k_half + 1, 2 * k_half + 1), bool))
    return mask


def task_weights_v2(grid: OccGrid, roi_mask_col: np.ndarray,
                     poses_user, specs) -> np.ndarray:
    """w_vox = 1{柱在 ROI} AND 1{至少一台 LiDAR 体素级 FoV 内}."""
    nx, ny, nz = grid.shape
    w = np.repeat(roi_mask_col[:, :, None].astype(np.float64), nz, axis=2)
    ctr = grid.centers()
    mask_fov = np.zeros(len(ctr), bool)
    for pu, spec in zip(poses_user, specs):
        o = np.asarray(pu[:3], np.float64)
        d = ctr - o
        r2 = np.hypot(d[:, 0], d[:, 1])
        r3 = np.linalg.norm(d, axis=1)
        el = np.degrees(np.arctan2(d[:, 2], np.maximum(r2, 1e-6)))
        m = (el >= spec.vfov_min_deg) & (el <= spec.vfov_max_deg) \
            & (r3 >= spec.range_min) & (r3 <= spec.range_max)
        if spec.fov_h_deg < 360:
            az = np.degrees(np.arctan2(d[:, 1], d[:, 0]) - pu[5])
            az = (az + 180.0) % 360.0 - 180.0
            m &= np.abs(az) <= spec.fov_h_deg / 2
        mask_fov |= m
    return w.ravel() * mask_fov.astype(np.float64)


def soft_visibility_v2(poses_user, specs, grid, p_hist_col=None,
                        n_ray=48, occ_alpha=0.9):
    return _soft_visibility_orig(poses_user, specs, grid, p_hist_col=p_hist_col,
                                  n_ray=n_ray, occ_alpha=occ_alpha)


# ---------------------------------------------------------------------------
# 五个修复版指标 (逐指标调用, 接口兼容)
# ---------------------------------------------------------------------------
def j_eig_n(poses_user, specs, grid, hist, roi_mask_col, n_ray=48) -> float:
    ref = hist[-1]
    q_vox, _, _ = soft_visibility_v2(poses_user, specs, ref,
                                      p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    w = task_weights_v2(grid, roi_mask_col, poses_user, specs)
    H = bern_entropy(ref.p).ravel()
    return float((w * q * H).sum() / (w.sum() + 1e-12))


def j_cov_n(poses_user, specs, grid, hist, roi_mask_col, n_ray=48) -> float:
    ref = hist[-1]
    q_vox, _, _ = soft_visibility_v2(poses_user, specs, ref,
                                      p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    w = task_weights_v2(grid, roi_mask_col, poses_user, specs)
    return float((w * q).sum() / (w.sum() + 1e-12))


def j_mv_n(poses_user, specs, grid, hist, roi_mask_col,
            lam: float = 0.5, n_ray=48) -> float:
    ref = hist[-1]
    _, q_col, col_ij = soft_visibility_v2(poses_user, specs, ref,
                                            p_hist_col=ref.column_max(), n_ray=n_ray)
    w_col = roi_mask_col.ravel().astype(np.float64)
    q_u = union_q(q_col)
    base = (w_col * q_u).sum() / (w_col.sum() + 1e-12)
    mv = 0.0
    col_xy = grid.origin[:2] + (col_ij + 0.5) * grid.voxel
    N = len(poses_user)
    for a in range(N):
        for b in range(a + 1, N):
            va = col_xy - np.asarray(poses_user[a][:2])
            vb = col_xy - np.asarray(poses_user[b][:2])
            na = np.hypot(va[:, 0], va[:, 1])
            nb = np.hypot(vb[:, 0], vb[:, 1])
            sin = np.abs(va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]) \
                / np.maximum(na * nb, 1e-6)
            mv += (w_col * q_col[:, a] * q_col[:, b] * sin).sum()
    mv /= (w_col.sum() + 1e-12)
    return float(base + lam * mv)


def j_dyn_n(poses_user, specs, grid, hist, roi_mask_col,
            eta: float = 0.8, n_ray=48) -> float:
    ref = hist[-1]
    q_vox, _, _ = soft_visibility_v2(poses_user, specs, ref,
                                      p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    d = _dyn_weight_orig(hist, eta).ravel()
    if d.max() < 1e-3:
        return float("nan")
    w = task_weights_v2(grid, roi_mask_col, poses_user, specs) * d
    if w.sum() < 1e-9:
        return float("nan")
    return float((w * q).sum() / w.sum())


def j_traj_n(poses_user, specs, grid, hist, roi_mask_col,
             mu: float = 0.5, seg_len: float = 5.0,
             q_low: float = 0.3, n_ray=48) -> float:
    ref = hist[-1]
    _, q_col, col_ij = soft_visibility_v2(poses_user, specs, ref,
                                            p_hist_col=ref.column_max(), n_ray=n_ray)
    q_u = union_q(q_col)
    corridors = extract_corridors(ref)
    if not corridors:
        return float("nan")
    nx, ny, _ = grid.shape
    col_index = col_ij[:, 0] * ny + col_ij[:, 1]
    roi_flat = roi_mask_col.ravel()
    num = 0.0; den = 0.0
    for cor in corridors:
        s1v, ij = cor["s1"]
        cells = ij[:, 0] * ny + ij[:, 1]
        in_roi = roi_flat[np.clip(cells, 0, len(roi_flat) - 1)]
        keep = in_roi & (s1v >= cor["s1_range"][0]) & (s1v < cor["s1_range"][1])
        if keep.sum() < 5:
            continue
        lo, hi = cor["s1_range"]
        n_seg = max(int(np.ceil((hi - lo) / seg_len)), 1)
        edges = np.linspace(lo, hi, n_seg + 1)
        Q = np.zeros(n_seg)
        for b in range(n_seg):
            m = keep & (s1v >= edges[b]) & (s1v < edges[b + 1])
            if not m.any():
                continue
            c = cells[m]
            pos = np.searchsorted(col_index, c)
            pos = np.clip(pos, 0, len(col_index) - 1)
            hit = col_index[pos] == c
            Q[b] = q_u[pos[hit]].mean() if hit.any() else 0.0
        low = Q < q_low
        L = 0; run = 0
        for v in low:
            if v: run += 1
            else:
                if run >= 2: L += run
                run = 0
        if run >= 2: L += run
        num += cor["conf"] * (Q.mean() - mu * (L / n_seg))
        den += cor["conf"]
    return float(num / (den + 1e-12))


SURROGATES_N = [
    ("J_EIG",  j_eig_n),
    ("J_cov",  j_cov_n),
    ("J_mv",   j_mv_n),
    ("J_dyn",  j_dyn_n),
    ("J_traj", j_traj_n),
]
METRIC_FUNCS_N = {name: fn for name, fn in SURROGATES_N}


def compute_all_n(poses_user, specs, grid, hist, roi_mask_col, n_ray=48) -> list[float]:
    """v2 指标集合, 签名与 rslaa_bench.metrics.compute_all 一致 (逐指标调用)."""
    return [float(fn(poses_user, specs, grid, hist, roi_mask_col, n_ray=n_ray))
            for _, fn in SURROGATES_N]


# ---------------------------------------------------------------------------
# 高效批量计算: soft_visibility 只算一次, 5 个指标共用
# ---------------------------------------------------------------------------
def prep_context(grid: OccGrid, hist: list[OccGrid], roi_mask_col: np.ndarray,
                 eta: float = 0.8) -> dict:
    """预计算与位姿无关的量, 供 compute_all_fast 在多候选间复用."""
    ref = hist[-1]
    nx, ny, nz = grid.shape
    H = bern_entropy(ref.p).ravel()
    d = _dyn_weight_orig(hist, eta).ravel()
    corridors = extract_corridors(ref)
    roi_flat = roi_mask_col.ravel()
    w_col = roi_flat.astype(np.float64)
    return dict(ref=ref, H=H, d=d, corridors=corridors,
                roi_flat=roi_flat, w_col=w_col, nx=nx, ny=ny, eta=eta)


def compute_all_fast(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
                     roi_mask_col: np.ndarray, ctx: dict | None = None,
                     n_ray: int = 48, lam: float = 0.5,
                     mu: float = 0.5, seg_len: float = 5.0,
                     q_low: float = 0.3) -> list[float]:
    """单次 soft_visibility 同时计算 5 个指标.

    相比 compute_all_n (5 次 soft_visibility) 节省约 4x 可见度计算.
    多候选时配合 prep_context() 进一步复用 H/d/corridors 等静态量.
    """
    if ctx is None:
        ctx = prep_context(grid, hist, roi_mask_col)
    ref = ctx["ref"]
    q_vox, q_col, col_ij = soft_visibility_v2(
        poses_user, specs, ref, p_hist_col=ref.column_max(), n_ray=n_ray)
    q_v = union_q(q_vox)
    q_c = union_q(q_col)
    w_vox = task_weights_v2(grid, roi_mask_col, poses_user, specs)
    w_col = ctx["w_col"]
    H = ctx["H"]; d = ctx["d"]; roi_flat = ctx["roi_flat"]
    nx, ny = ctx["nx"], ctx["ny"]

    # J_EIG
    j_eig = float((w_vox * q_v * H).sum() / (w_vox.sum() + 1e-12))
    # J_cov
    j_cov = float((w_vox * q_v).sum() / (w_vox.sum() + 1e-12))
    # J_mv
    base = (w_col * q_c).sum() / (w_col.sum() + 1e-12)
    mv = 0.0
    col_xy = grid.origin[:2] + (col_ij + 0.5) * grid.voxel
    N = len(poses_user)
    for a in range(N):
        for b in range(a + 1, N):
            va = col_xy - np.asarray(poses_user[a][:2])
            vb = col_xy - np.asarray(poses_user[b][:2])
            na = np.hypot(va[:, 0], va[:, 1])
            nb = np.hypot(vb[:, 0], vb[:, 1])
            sin = np.abs(va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]) \
                / np.maximum(na * nb, 1e-6)
            mv += (w_col * q_col[:, a] * q_col[:, b] * sin).sum()
    j_mv = float(base + lam * mv / (w_col.sum() + 1e-12))
    # J_dyn
    if d.max() < 1e-3:
        j_dyn = float("nan")
    else:
        wd = w_vox * d
        j_dyn = float((wd * q_v).sum() / wd.sum()) if wd.sum() >= 1e-9 \
            else float("nan")
    # J_traj
    corridors = ctx["corridors"]
    if not corridors:
        j_traj = float("nan")
    else:
        col_index = col_ij[:, 0] * ny + col_ij[:, 1]
        num = 0.0; den = 0.0
        for cor in corridors:
            s1v, ij = cor["s1"]
            cells = ij[:, 0] * ny + ij[:, 1]
            in_roi = roi_flat[np.clip(cells, 0, len(roi_flat) - 1)]
            keep = in_roi & (s1v >= cor["s1_range"][0]) \
                & (s1v < cor["s1_range"][1])
            if keep.sum() < 5:
                continue
            lo, hi = cor["s1_range"]
            n_seg = max(int(np.ceil((hi - lo) / seg_len)), 1)
            edges = np.linspace(lo, hi, n_seg + 1)
            Q = np.zeros(n_seg)
            for b in range(n_seg):
                m = keep & (s1v >= edges[b]) & (s1v < edges[b + 1])
                if not m.any():
                    continue
                c = cells[m]
                pos = np.searchsorted(col_index, c)
                pos = np.clip(pos, 0, len(col_index) - 1)
                hit = col_index[pos] == c
                Q[b] = q_c[pos[hit]].mean() if hit.any() else 0.0
            low = Q < q_low
            L = 0; run = 0
            for v in low:
                if v: run += 1
                else:
                    if run >= 2: L += run
                    run = 0
            if run >= 2: L += run
            num += cor["conf"] * (Q.mean() - mu * (L / n_seg))
            den += cor["conf"]
        j_traj = float(num / (den + 1e-12)) if den > 0 else float("nan")

    return [j_eig, j_cov, j_mv, j_dyn, j_traj]
