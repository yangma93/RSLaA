# -*- coding: utf-8 -*-
"""多 LiDAR 位姿候选的无真值替代评价指标 (surrogate metrics).

对应《问题定义》第 5 节的五类指标, 全部仅使用决策时刻可用信息:
  H_k = 历史窗口的点云/占用概率图/观测掩码/位姿, 候选位姿 U_k(a), 传感器模型, 道路 ROI.

位姿约定: 用户序 [x, y, z, roll, pitch, yaw] (弧度); 内部 LidarPose 序
为 [x, y, z, roll, yaw, pitch], 由 pose_internal() 转换.

实现说明 (在文档公式基础上的具体化):
  * 软可见性 q_i,n 分两步: (1) BEV 柱级可见性 —— FoV/距离衰减/沿视线对历史
    占用图做 ray-march 透光率; (2) 体素级 —— 柱可见性 x 俯仰角 FoV 掩码
    x 3D 距离衰减. 遮挡仅由历史占用图估计, 不使用未来观测.
  * EIG 采用命中/漏检模型的闭式近似: I(X_i;Z_i) ≈ q_i * H(p_i),
    H 为伯努利熵 (观测命中后占用状态基本确定).
  * 多视角收益 g(theta) = |sin(theta)|, 90 度最大 (三角测量几何最优), 在柱级计算.
  * 动态权重 d_i: 历史相邻窗占用图的 JSD(Bern(p_r-1), Bern(p_r)) 按观测掩码
    与时间衰减加权; 历史不足 2 帧地图时退化为伯努利熵 (首决策点的变化代理).
  * 轨迹走廊: 由历史占用柱图提取 (0.05<p_col<0.95 的通行区域), 主方向 PCA,
    横向 3.75 m 分车道为走廊, 沿主方向 5 m 分段; Q 为段内联合可见度均值,
    L 为连续低可见段惩罚.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 位姿约定转换
# ---------------------------------------------------------------------------
def pose_internal(pose_user) -> list:
    """用户序 [x y z roll pitch yaw] -> 内部序 [x y z roll yaw pitch]."""
    x, y, z, roll, pitch, yaw = [float(v) for v in pose_user]
    return [x, y, z, roll, yaw, pitch]


# ---------------------------------------------------------------------------
# 占用网格容器
# ---------------------------------------------------------------------------
class OccGrid:
    """稠密体素占用概率图 + 观测掩码."""

    def __init__(self, p: np.ndarray, o: np.ndarray, origin: np.ndarray,
                 voxel: float):
        self.p = np.asarray(p, np.float64)          # (nx,ny,nz) [0,1]
        self.o = np.asarray(o, np.float64)          # (nx,ny,nz) [0,1]
        self.origin = np.asarray(origin, np.float64)
        self.voxel = float(voxel)

    @property
    def shape(self):
        return self.p.shape

    def centers(self) -> np.ndarray:
        """全部体素中心 (V,3)."""
        nx, ny, nz = self.shape
        i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                              indexing="ij")
        ij = np.stack([i.ravel(), j.ravel(), k.ravel()], 1)
        return self.origin + (ij + 0.5) * self.voxel

    def column_max(self, p=None) -> np.ndarray:
        """BEV 柱级占用 (nx,ny): 沿 z 取 max."""
        return (self.p if p is None else p).max(axis=2)

    def save(self, path: str, pose_user=None, t_range=None):
        kw = dict(p=self.p.astype(np.float32), o=self.o.astype(np.float32),
                  origin=self.origin, voxel=np.float64(self.voxel))
        if pose_user is not None:
            kw["pose_user"] = np.asarray(pose_user, np.float64)
        if t_range is not None:
            kw["t_range"] = np.asarray(t_range, np.float64)
        np.savez_compressed(path, **kw)

    @classmethod
    def load(cls, path: str) -> "OccGrid":
        d = np.load(path)
        g = cls(d["p"], d["o"], d["origin"], float(d["voxel"]))
        g.pose_user = d["pose_user"] if "pose_user" in d else None
        g.t_range = d["t_range"] if "t_range" in d else None
        return g


def bern_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def bern_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """伯努利分布间的 Jensen-Shannon 散度 (逐体素)."""
    m = 0.5 * (p + q)
    return np.clip(bern_entropy(m) - 0.5 * bern_entropy(p)
                   - 0.5 * bern_entropy(q), 0.0, None)


# ---------------------------------------------------------------------------
# 软可见性 q
# ---------------------------------------------------------------------------
def _column_visibility(pose_u, spec, col_xy: np.ndarray, p_col: np.ndarray,
                       origin, voxel, n_ray: int, occ_alpha: float) -> np.ndarray:
    """单个 LiDAR 对全部 BEV 柱的软可见性 (C,).

    pose_u: 用户序位姿; col_xy: (C,2) 柱中心; p_col: (nx,ny) 历史柱占用.
    """
    o = np.asarray(pose_u[:2], np.float64)
    vec = col_xy - o                                     # (C,2)
    r = np.hypot(vec[:, 0], vec[:, 1])
    q = np.zeros(len(col_xy), np.float64)
    m = (r >= spec.range_min) & (r <= spec.range_max)
    if spec.fov_h_deg < 360:
        az = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]) - pose_u[5])
        az = (az + 180.0) % 360.0 - 180.0
        m &= np.abs(az) <= spec.fov_h_deg / 2
    if not m.any():
        return q
    # 距离衰减 (接收功率近似)
    qr = 1.0 / (1.0 + (r / spec.range_ref) ** 2)
    # 遮挡: 沿 2D 视线对历史占用柱图 ray-march (不含目标柱自身, 避免自遮挡)
    idx = np.flatnonzero(m)
    rr = r[idx]
    u = vec[idx] / rr[:, None]
    ts = np.linspace(0.02, 0.98, n_ray)[None, :]                  # 归一化弧长
    samp = o[None, None, :] + u[:, None, :] * (rr[:, None, None] * ts[:, :, None])
    ij = np.floor((samp.reshape(-1, 2) - origin[:2]) / voxel).astype(np.int64)
    nx, ny = p_col.shape
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    pv = np.where(ok, p_col[np.clip(ij[:, 0], 0, nx - 1),
                            np.clip(ij[:, 1], 0, ny - 1)], 0.0)
    T = np.prod(1.0 - occ_alpha * pv.reshape(len(idx), -1), axis=1)
    q[idx] = qr[idx] * np.clip(T, 0.0, 1.0)
    return q


def soft_visibility(poses_user, specs, grid: OccGrid, p_hist_col=None,
                    n_ray: int = 48, occ_alpha: float = 0.9) -> tuple:
    """候选联合位姿下, 各体素被各 LiDAR 有效观测的软概率.

    返回 (q_vox (V,N), q_col (C,N), col_ij (C,2)).
    """
    nx, ny, nz = grid.shape
    ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    col_ij = np.stack([ci.ravel(), cj.ravel()], 1)
    col_xy = grid.origin[:2] + (col_ij + 0.5) * grid.voxel
    p_col = grid.column_max() if p_hist_col is None else p_hist_col

    N = len(poses_user)
    q_col = np.zeros((len(col_ij), N), np.float64)
    for n, (pu, spec) in enumerate(zip(poses_user, specs)):
        q_col[:, n] = _column_visibility(pu, spec, col_xy, p_col,
                                         grid.origin, grid.voxel,
                                         n_ray, occ_alpha)
    # 体素级: 柱可见性 x 俯仰 FoV x 3D 距离衰减
    ctr = grid.centers()
    V = len(ctr)
    q_vox = np.zeros((V, N), np.float64)
    vij = np.floor((ctr - grid.origin) / grid.voxel).astype(np.int64)
    col_of_v = vij[:, 0] * ny + vij[:, 1]
    for n, (pu, spec) in enumerate(zip(poses_user, specs)):
        o = np.asarray(pu[:3], np.float64)
        d = ctr - o
        r2 = np.hypot(d[:, 0], d[:, 1])
        r3 = np.linalg.norm(d, axis=1)
        el = np.degrees(np.arctan2(d[:, 2], np.maximum(r2, 1e-6)))
        m = (el >= spec.vfov_min_deg) & (el <= spec.vfov_max_deg) \
            & (r3 >= spec.range_min) & (r3 <= spec.range_max)
        q3r = 1.0 / (1.0 + (r3 / spec.range_ref) ** 2)
        q_col_n = q_col[col_of_v, n]
        q2r = 1.0 / (1.0 + (np.maximum(r2, 1e-6) / spec.range_ref) ** 2)
        q_vox[:, n] = q_col_n * np.where(m, q3r / np.maximum(q2r, 1e-6), 0.0)
    return q_vox, q_col, col_ij


def union_q(q: np.ndarray) -> np.ndarray:
    """q_i = 1 - prod_n (1 - q_i,n) (近似独立)."""
    return 1.0 - np.prod(1.0 - np.clip(q, 0.0, 1.0), axis=1)


# ---------------------------------------------------------------------------
# 任务权重与 ROI
# ---------------------------------------------------------------------------
def task_weights(grid: OccGrid, roi_mask_col: np.ndarray) -> np.ndarray:
    """体素任务权重: ROI 柱内 w=1, 其余 0 (可按历史占用频率细化)."""
    nx, ny, nz = grid.shape
    w = np.repeat(roi_mask_col[:, :, None], nz, axis=2).astype(np.float64)
    return w.ravel()


# ---------------------------------------------------------------------------
# 五个替代指标
# ---------------------------------------------------------------------------
def j_eig(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
          roi_mask_col, n_ray=48) -> float:
    """J_EIG: 预期占用不确定性下降 ≈ sum_i w_i q_i H(p_i) / sum_i w_i."""
    ref = hist[-1]
    q_vox, _, _ = soft_visibility(poses_user, specs, ref,
                                  p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    w = task_weights(grid, roi_mask_col)
    H = bern_entropy(ref.p).ravel()
    denom = w.sum() + 1e-12
    return float((w * q * H).sum() / denom)


def j_cov(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
          roi_mask_col, n_ray=48) -> float:
    """J_cov: 任务加权覆盖 = sum_i w_i q_i / sum_i w_i."""
    ref = hist[-1]
    q_vox, _, _ = soft_visibility(poses_user, specs, ref,
                                  p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    w = task_weights(grid, roi_mask_col)
    return float((w * q).sum() / (w.sum() + 1e-12))


def j_mv(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
         roi_mask_col, lam: float = 0.5, n_ray=48) -> float:
    """J_mv: 覆盖 + 多视角互补 (柱级, g=|sin 视线夹角|)."""
    ref = hist[-1]
    _, q_col, col_ij = soft_visibility(poses_user, specs, ref,
                                       p_hist_col=ref.column_max(), n_ray=n_ray)
    w_col = roi_mask_col.ravel()
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


def _dyn_weight(hist: list[OccGrid], eta: float = 0.8) -> np.ndarray:
    """历史动态权重 d_i (体素级): 相邻窗占用图 JSD, 观测掩码与时间衰减加权.

    历史地图不足 2 帧时退化为伯努利熵 (占用不确定性高的区域更可能动态).
    """
    if len(hist) < 2:
        return bern_entropy(hist[-1].p)
    num = np.zeros_like(hist[-1].p)
    den = 0.0
    R = len(hist)
    for r in range(1, R):
        decay = eta ** (R - 1 - r)
        om = hist[r].o * hist[r - 1].o
        num = num + decay * om * bern_jsd(hist[r - 1].p, hist[r].p)
        den = den + decay * om
    return num / (den + 1e-12)


def j_dyn(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
          roi_mask_col, eta: float = 0.8, n_ray=48) -> float:
    """J_dyn: 历史动态区域覆盖 = sum w d q / sum w d.

    退化策略:
      * 历史地图 < 2 帧 (首个决策点): d_i 退化为伯努利熵 H(p_i)
        —— 占用概率接近 0.5 的体素历史占用不稳定, 更可能动态;
      * 动态权重总和 ≈ 0 (历史中无可辨识的动态区域, 如全空闲/全占用):
        指标失去判别力, 返回 NaN, 由上层记录为"不适用"决策点.
    """
    ref = hist[-1]
    q_vox, _, _ = soft_visibility(poses_user, specs, ref,
                                  p_hist_col=ref.column_max(), n_ray=n_ray)
    q = union_q(q_vox)
    d = _dyn_weight(hist, eta).ravel()
    # 退化: 所有体素动态权重均低于阈值 (p 全 ≈0/1, 无不确定性/变化)
    if d.max() < 1e-3:
        return float("nan")
    w = task_weights(grid, roi_mask_col) * d
    if w.sum() < 1e-9:
        return float("nan")
    return float((w * q).sum() / w.sum())


def extract_corridors(grid: OccGrid, n_lanes: int = 8, seg_len: float = 5.0,
                        min_conf: float = 0.05):
    """由历史占用柱图提取带置信度的预测轨迹走廊.

    返回 list[dict]: lanes (横向分带) x segments (沿主方向分段).
    剔除置信度 < min_conf 的走廊, 保留高置信度预测.
    """
    p_col = grid.column_max()
    nx, ny, _ = grid.shape
    ci, cj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    col_ij = np.stack([ci.ravel(), cj.ravel()], 1)
    col_xy = grid.origin[:2] + (col_ij + 0.5) * grid.voxel
    act = (p_col.ravel() > 0.05) & (p_col.ravel() < 0.95)
    if act.sum() < 20:
        return []
    xy = col_xy[act]
    # 主方向 (PCA 第一主成分)
    xm = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(xm, full_matrices=False)
    u1, u2 = vt[0], vt[1]
    s1, s2 = xy @ u1, xy @ u2
    p_act = p_col.ravel()[act]
    # 横向分带 = 走廊
    edges = np.linspace(s2.min(), s2.max(), n_lanes + 1)
    corridors = []
    for l in range(n_lanes):
        m = (s2 >= edges[l]) & (s2 < edges[l + 1])
        if m.sum() < 10:
            continue
        conf = float(np.clip(p_act[m].mean() * 2, 0, 1))
        if conf < min_conf:
            continue
        corridors.append(dict(u1=u1, u2=u2, s1=(s1[m], col_ij[act][m]),
                              conf=conf,
                              s1_range=(float(s1[m].min()), float(s1[m].max()))))
    return corridors


def j_traj(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
           roi_mask_col, mu: float = 0.5, seg_len: float = 5.0,
           q_low: float = 0.3, n_ray=48) -> float:
    """J_traj: 预测走廊连续可见性 = sum a_t (mean Q - mu L) / sum a_t.

    退化策略:
      * 历史占用图提取不出可靠走廊 (通行柱数不足 20, 或全部走廊置信度过低)
        时, 说明历史运动预测不可靠 -> 返回 NaN, 由上层记录为"不适用"决策点
        (对应问题定义 5.5 节: 明确报告该指标不适用的决策点比例).
    """
    ref = hist[-1]
    _, q_col, col_ij = soft_visibility(poses_user, specs, ref,
                                       p_hist_col=ref.column_max(), n_ray=n_ray)
    q_u = union_q(q_col)
    nx, ny, _ = grid.shape
    corridors = extract_corridors(ref)
    if not corridors:
        return float("nan")
    num = 0.0
    den = 0.0
    col_index = col_ij[:, 0] * ny + col_ij[:, 1]
    for cor in corridors:
        s1v, ij = cor["s1"]
        lo, hi = cor["s1_range"]
        n_seg = max(int(np.ceil((hi - lo) / seg_len)), 1)
        edges = np.linspace(lo, hi, n_seg + 1)
        Q = np.zeros(n_seg)
        for b in range(n_seg):
            m = (s1v >= edges[b]) & (s1v < edges[b + 1])
            if not m.any():
                continue
            cells = ij[m, 0] * ny + ij[m, 1]
            pos = np.searchsorted(col_index, cells)
            pos = np.clip(pos, 0, len(col_index) - 1)
            hit = col_index[pos] == cells
            Q[b] = q_u[pos[hit]].mean() if hit.any() else 0.0
        # 连续低可见段惩罚: 长度 >=2 的低可见游程总长
        low = Q < q_low
        L = 0
        run = 0
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


# 指标注册表: 顺序对应 data 矩阵的 指标1..指标5
SURROGATES = [
    ("J_EIG",  j_eig),
    ("J_cov",  j_cov),
    ("J_mv",   j_mv),
    ("J_dyn",  j_dyn),
    ("J_traj", j_traj),
]


def compute_all(poses_user, specs, grid: OccGrid, hist: list[OccGrid],
                roi_mask_col, n_ray=48) -> list[float]:
    """计算 5 个替代指标, 返回 [J1..J5] (统一为越大越好)."""
    return [float(fn(poses_user, specs, grid, hist, roi_mask_col, n_ray=n_ray))
            for _, fn in SURROGATES]
