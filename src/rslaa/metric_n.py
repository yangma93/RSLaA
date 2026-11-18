# -*- coding: utf-8 -*-
"""替代指标 v2: metric_n.py (修复版).

保持 rslaa_bench/metrics.py **不变**, 仅在此文件重新定义 5 个指标 + 注册表.

根因修复
--------
诊断发现 (diag_metrics.py):
  * J_cov(metrics 原版) 在 3 组位姿对 A/B/C 上排序 B>A>C, 与真实 IoU+Comp 排序 A>B>C
    完全反向. 根因:
    (1) ROI 掩码取自"历史 POG 几何可观测区域"(全盘 87.69%), 不是"任务区域"
        (轨迹包围盒仅 2.25%). 候选位姿只要 FoV 覆盖住历史 POG 曾被观测过的区域
        (对 360° 雷达来说几乎 trivially 成立), 就能拿高分 —— 与真实"能不能看到
        车"无关.
    (2) J_EIG/J_cov/J_dyn 体素级: w = task_weights(grid, roi_mask_col) 把柱级
        roi_mask_col 广播到 z 各层, 但 soft_visibility 的 q_vox 还乘了**体素级
        俯仰 FoV 掩码 m**. 结果分母 sum(w) 被"FoV 外体素"稀释, 分母 ≈ 全盘
        体素数, 指标恒接近 0 且区分度弱.

修复措施 (保持 metrics.py 不动)
--------------------------------
  1. **ROI 改用"任务区域"**: 由轨迹包围盒 + ROIMARGIN 在 OccGrid 上投影得到
     BEV 掩码; 可由 build_task_roi() 生成或由外部直接传入.
  2. **体素级 w 也带 FoV 掩码**: 每个候选位姿先算"至少一台 LiDAR 在体素级 FoV 内"
     的体素掩码 m_fov, 再把 w *= m_fov. 分母 sum(w) 变为"任务区域 × FoV 内"的
     体素数, 与分子 sum(w*q) 有效集严格对齐.
  3. **遮挡模型的"空区域"衰减**: 复用 metrics._column_visibility, 历史占用为空
     时 ray-march 全透光, 是合理的 (空区域本来就不该被遮挡惩罚). 但需要任务
     ROI 正确后才能让"FoV 是否覆盖任务区"成为主导效应.
  4. **保留柱级 J_mv/J_traj**: 它们本来就是柱级, 没有体素对齐问题, 仅需修 ROI.

兼容
----
SURROGATES_N 列表注册 5 个新指标, compute_all_n() 同 compute_all() 接口.
rslaa.common、mpc_controller.py 等调用处只需把 compute_all / METRIC_FUNCS
改为本文件的版本即可 (不要求, 本文件作为替代独立使用).
"""
from __future__ import annotations

import numpy as np
from rslaa_bench.metrics import (
    OccGrid,
    _column_visibility,
    soft_visibility as _soft_visibility_orig,
    union_q,
    bern_entropy,
    _dyn_weight as _dyn_weight_orig,
    extract_corridors,
)

# 默认轨迹 ROI 外扩 (与 rslaa.common 中 ROIMARGIN 一致)
ROIMARGIN = 10.0


# ---------------------------------------------------------------------------
# 任务 ROI 构造 (BEV 柱级掩码)
# ---------------------------------------------------------------------------
def build_task_roi(grid: OccGrid, traj_front: np.ndarray,
                   margin: float = ROIMARGIN, road_half_m: float = 5.0) -> np.ndarray:
    """由轨迹构造 BEV 任务区域 (nx,ny) bool.

    语义: 轨迹散点标记的柱 + 沿轨迹横向膨胀 road_half_m (默认 5m,
    对应常规车道/道路半宽) —— 这样 ROI 只覆盖"有车会走的道路", 不会
    退化为全盘 (网格本身就是按轨迹包围盒 + margin 定的, 如果再按包围盒
    填充等于全盘).
    """
    traj_front = np.asarray(traj_front, np.float64)
    if traj_front.ndim == 2:
        traj_xy = traj_front[:, :2]
    else:
        traj_xy = traj_front[:2].reshape(1, -1)

    xi = ((traj_xy[:, 0] - grid.origin[0]) / grid.voxel).astype(np.int64)
    yi = ((traj_xy[:, 1] - grid.origin[1]) / grid.voxel).astype(np.int64)
    nx, ny, _ = grid.shape
    mask = np.zeros((nx, ny), bool)
    ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    mask[xi[ok], yi[ok]] = True

    # 沿轨迹散点膨胀 (代替包围盒填充, 让 ROI 保持"道路走廊"形状)
    k_half = max(1, int(np.round(road_half_m / grid.voxel)))
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * k_half + 1, 2 * k_half + 1), bool)
    mask = binary_dilation(mask, struct)
    return mask


def task_weights_v2(grid: OccGrid, roi_mask_col: np.ndarray,
                     poses_user, specs) -> np.ndarray:
    """体素级任务权重 v2 (关键修复).

    w_vox = 1{柱在 ROI 内} AND 1{至少一台 LiDAR 在该体素的俯仰 FoV 内}

    这样 w 的有效体素集与 q_vox 完全一致 (q_vox 在 FoV 外也是 0).
    """
    nx, ny, nz = grid.shape
    w = np.repeat(roi_mask_col[:, :, None].astype(np.float64), nz, axis=2)
    # 体素级 FoV 掩码 (逐体素, 至少一台 LiDAR 覆盖)
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
    w = w.ravel() * mask_fov.astype(np.float64)
    return w


# ---------------------------------------------------------------------------
# 软可见度 (代理, 方便本文件内引用)
# ---------------------------------------------------------------------------
def soft_visibility_v2(poses_user, specs, grid, p_hist_col=None,
                        n_ray=48, occ_alpha=0.9):
    """直接调用 metrics.soft_visibility, 避免重复实现."""
    return _soft_visibility_orig(poses_user, specs, grid, p_hist_col=p_hist_col,
                                  n_ray=n_ray, occ_alpha=occ_alpha)


# ---------------------------------------------------------------------------
# 五个修复版指标
# ---------------------------------------------------------------------------
def j_eig_n(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
            roi_mask_col, n_ray=48) -> float:
    """J_EIG v2: 预期占用不确定性下降 (任务 ROI × FoV 内)."""
    ref = hist[-1]
    q_vox, _, _ = soft_visibility_v2(poses_user, specs, ref,
                                      p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    w = task_weights_v2(grid, roi_mask_col, poses_user, specs)
    H = bern_entropy(ref.p).ravel()
    denom = w.sum() + 1e-12
    return float((w * q * H).sum() / denom)


def j_cov_n(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
            roi_mask_col, n_ray=48) -> float:
    """J_cov v2: 任务区域 × FoV 内覆盖.

    分母是"任务区域中至少一台雷达几何上能看到的体素数", 分子是它们的
    联合软可见度加权和. 一个"好位姿"应该既覆盖任务区域又让遮挡衰减小,
    这正对应真实 IoU/Comp 的高值.
    """
    ref = hist[-1]
    q_vox, _, _ = soft_visibility_v2(poses_user, specs, ref,
                                      p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    w = task_weights_v2(grid, roi_mask_col, poses_user, specs)
    return float((w * q).sum() / (w.sum() + 1e-12))


def j_mv_n(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
            roi_mask_col, lam: float = 0.5, n_ray=48) -> float:
    """J_mv v2: 任务区域覆盖 + 多视角互补 (柱级, ROI 修正)."""
    ref = hist[-1]
    _, q_col, col_ij = soft_visibility_v2(poses_user, specs, ref,
                                            p_hist_col=ref.column_max(), n_ray=n_ray)
    w_col = roi_mask_col.ravel().astype(np.float64)
    q_u = union_q(q_col)
    base = (w_col * q_u).sum() / (w_col.sum() + 1e-12)
    mv = 0.0
    nx, ny, _ = grid.shape
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


def _dyn_weight_n(hist: list[OccGrid], eta: float = 0.8) -> np.ndarray:
    """复用 metrics._dyn_weight (逻辑正确, 仅在此代理)."""
    return _dyn_weight_orig(hist, eta)


def j_dyn_n(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
            roi_mask_col, eta: float = 0.8, n_ray=48) -> float:
    """J_dyn v2: 历史动态区域覆盖 (体素级 w 修复)."""
    ref = hist[-1]
    q_vox, _, _ = soft_visibility_v2(poses_user, specs, ref,
                                      p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    d = _dyn_weight_n(hist, eta).ravel()
    if d.max() < 1e-3:
        return float("nan")
    w = task_weights_v2(grid, roi_mask_col, poses_user, specs) * d
    if w.sum() < 1e-9:
        return float("nan")
    return float((w * q).sum() / w.sum())


def j_traj_n(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
             roi_mask_col, mu: float = 0.5, seg_len: float = 5.0,
             q_low: float = 0.3, n_ray=48) -> float:
    """J_traj v2: 预测走廊连续可见性 (ROI 修正).

    只在任务 ROI 内的通行柱上提取走廊 (旧版本用全盘 0.05<p<0.95).
    """
    ref = hist[-1]
    _, q_col, col_ij = soft_visibility_v2(poses_user, specs, ref,
                                            p_hist_col=ref.column_max(), n_ray=n_ray)
    q_u = union_q(q_col)
    corridors = extract_corridors(ref)
    if not corridors:
        return float("nan")
    # 任务 ROI 掩码叠加: 只保留走廊中落在任务 ROI 的柱
    nx, ny, _ = grid.shape
    col_index = col_ij[:, 0] * ny + col_ij[:, 1]
    roi_flat = roi_mask_col.ravel()
    num = 0.0
    den = 0.0
    for cor in corridors:
        s1v, ij = cor["s1"]
        cells = ij[:, 0] * ny + ij[:, 1]
        # col_mask 是 roi_mask_col (nx,ny), cells 的 flat index 可直接索引
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
            if v:
                run += 1
            else:
                if run >= 2:
                    L += run
                run = 0
        if run >= 2:
            L += run
        score = Q.mean() - mu * (L / n_seg)
        num += cor["conf"] * score
        den += cor["conf"]
    return float(num / (den + 1e-12))


# ---------------------------------------------------------------------------
# 注册表 (独立, 不覆盖 rslaa_bench.metrics.SURROGATES)
# ---------------------------------------------------------------------------
SURROGATES_N = [
    ("J_EIG",  j_eig_n),
    ("J_cov",  j_cov_n),
    ("J_mv",   j_mv_n),
    ("J_dyn",  j_dyn_n),
    ("J_traj", j_traj_n),
]

METRIC_FUNCS_N = {name: fn for name, fn in SURROGATES_N}


def compute_all_n(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
                  roi_mask_col, n_ray=48) -> list[float]:
    """v2 指标集合, 签名与 rslaa_bench.metrics.compute_all 完全一致."""
    return [float(fn(poses_user, specs, grid, hist, roi_mask_col, n_ray=n_ray))
            for _, fn in SURROGATES_N]
