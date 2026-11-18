# -*- coding: utf-8 -*-
"""Ray-casting 求交内核: 环境点云 (CSR 体素网格加速) + 车辆三角网格 (Möller–Trumbore).

环境点云与车辆目标的求交物理构成不同:
  - 环境: 离散点集, 每点视为有限半径散射体, 命中条件为光束轴线到点垂距 <= 束半径;
  - 车辆: 连续表面三角网格, 用射线-三角形精确求交, 可得面元法向/入射角用于强度计算.

后端: CuPy (GPU 并行, 每条光线一个线程束单元批量处理) 或 NumPy (CPU 回退).
"""
import numpy as np

try:
    import cupy as cp
    HAS_CP = True
except Exception:
    cp = None
    HAS_CP = False


def get_xp(device: str = "auto"):
    if device == "gpu" and not HAS_CP:
        raise RuntimeError("需要 CuPy 才能使用 GPU (--device cpu 可回退)")
    if device == "cpu":
        return np, False
    if HAS_CP:
        try:
            cp.cuda.runtime.getDeviceCount()
            return cp, True
        except Exception:
            if device == "gpu":
                raise
    return np, False


class EnvGrid:
    """环境点云的 CSR 体素网格 (均匀 cell, 按点排序构建)."""

    def __init__(self, points: np.ndarray, cell: float, xp, on_gpu: bool):
        self.xp, self.gpu = xp, on_gpu
        self.cell = float(cell)
        pts = np.ascontiguousarray(points, np.float32)
        self.gmin = pts.min(0) - cell
        ij = np.floor((pts - self.gmin) / cell).astype(np.int64)
        nx, ny, nz = ij.max(0).astype(np.int64) + 1
        self.dims = (nx, ny, nz)
        ncell = nx * ny * nz
        if ncell > 300_000_000:
            raise ValueError(f"环境网格过大 ({ncell} cells), 请增大 grid_cell")
        flat = ((ij[:, 0] * ny) + ij[:, 1]) * nz + ij[:, 2]
        order = np.argsort(flat, kind="stable")
        f_s = flat[order]
        counts = np.bincount(f_s, minlength=ncell).astype(np.int64)
        offsets = np.zeros(ncell + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        arrs = dict(points=pts, sorted_idx=order.astype(np.int32),
                    offsets=offsets.astype(np.int32))
        self.arrs = {k: xp.asarray(v) for k, v in arrs.items()}
        # numpy 副本 (净空检查等 CPU 用途)
        self.points_np = pts
        self.sorted_idx_np = order.astype(np.int32)
        self.offsets_np = offsets.astype(np.int32)

    def points_in_cell(self, xyz: np.ndarray) -> np.ndarray:
        """返回与 xyz 同一体素内的环境点 (可能为空), 用于净空检查."""
        nx, ny, nz = self.dims
        ij = np.floor((np.asarray(xyz, np.float64) - self.gmin) / self.cell).astype(np.int64)
        if np.any(ij < 0) or np.any(ij >= np.array([nx, ny, nz])):
            return np.zeros((0, 3), np.float32)
        flat = int(((ij[0] * ny) + ij[1]) * nz + ij[2])
        s, e = int(self.offsets_np[flat]), int(self.offsets_np[flat + 1])
        return self.points_np[self.sorted_idx_np[s:e]]

    # ---- 单次发射: 光线束 vs 环境点 ----
    def raycast(self, origins, dirs, r_min, t_max, beam_tan, chunk=4096):
        """origins (R,3) f4, dirs (R,3) f4 单位向量, t_max (R,).

        返回 t_env (R,) float32 (inf=未命中), pt_idx (R,) int32 (-1=未命中).
        """
        xp = self.xp
        R = len(origins)
        O = xp.asarray(origins); D = xp.asarray(dirs)
        TMAX = xp.asarray(t_max)
        step = self.cell * 0.5
        S = int(np.ceil(float(t_max.max()) / step)) + 1
        t_off = (xp.arange(S, dtype=xp.float32) * step)
        t_env = xp.full(R, np.inf, xp.float32)
        pt_idx = xp.full(R, -1, xp.int32)
        pts = self.arrs["points"]; sidx = self.arrs["sorted_idx"]; offs = self.arrs["offsets"]
        gmin = xp.asarray(self.gmin, xp.float32)
        nx, ny, nz = self.dims
        for s0 in range(0, R, chunk):
            s1 = min(s0 + chunk, R)
            row = xp.arange(s0, s1, dtype=xp.int32)[:, None]
            o = O[s0:s1, None, :]                       # (r,1,3)
            d = D[s0:s1, None, :]
            sp = o + t_off[None, :, None] * d           # (r,S,3) 采样点
            c = xp.floor((sp - gmin) / self.cell).astype(xp.int32)
            c = xp.clip(c, 0, xp.asarray([nx - 1, ny - 1, nz - 1], xp.int32))
            flat = ((c[..., 0].astype(xp.int64) * ny) + c[..., 1]) * nz + c[..., 2]
            cnt = offs[flat + 1] - offs[flat]           # (r,S)
            ts = xp.broadcast_to(t_off[None, :], flat.shape)
            sel = (cnt > 0) & (ts >= r_min) & (ts < TMAX[s0:s1, None])
            if not bool(sel.any()):
                continue
            ray_sel = xp.broadcast_to(row, flat.shape)[sel]        # (k,)
            f_sel = flat[sel]
            c_sel = cnt[sel]
            total = int(c_sel.sum())
            if total == 0:
                continue
            ends = xp.cumsum(c_sel)
            starts = ends - c_sel
            pos = xp.arange(total, dtype=xp.int64)
            # 按 c_sel 展开成逐点副本 (用 searchsorted 代替 repeat, CuPy 不支持数组 repeats)
            cell_rep = xp.searchsorted(ends, pos, side="right")
            off_in = pos - starts[cell_rep]
            pair_pt = sidx[offs[f_sel[cell_rep]] + off_in]         # (total,)
            pair_ray = ray_sel[cell_rep]                           # (total,)
            # 精确光线-点测试
            P = pts[pair_pt]; O2 = O[pair_ray]; D2 = D[pair_ray]
            v = P - O2
            along = (v * D2).sum(1)
            perp2 = (v * v).sum(1) - along * along
            rb = xp.maximum(0.03, beam_tan * along)                # 束半径
            ok = (along > r_min) & (along < TMAX[pair_ray]) & (perp2 <= rb * rb)
            if not bool(ok.any()):
                continue
            rr = pair_ray[ok]; aa = along[ok]; pp = pair_pt[ok]
            order = _lexsort2(xp, aa, rr)
            rr2, aa2, pp2 = rr[order], aa[order], pp[order]
            _, first = xp.unique(rr2, return_index=True)
            t_env[rr2[first]] = aa2[first]
            pt_idx[rr2[first]] = pp2[first]
        return t_env, pt_idx


def _lexsort2(xp, secondary, primary):
    """按 primary 主序、secondary 次序排序索引 (numpy/cupy 兼容)."""
    return xp.lexsort(xp.stack([secondary, primary]))


def ray_aabb(origins, dirs, bb_min, bb_max, t_cap):
    """slab 测试. origins (R,3), dirs (R,3), bb (V,3), t_cap (R,) -> (R,V) bool."""
    xp = cp if HAS_CP and isinstance(origins, cp.ndarray) else np
    o = origins[:, None, :]; d = dirs[:, None, :]
    inv = 1.0 / xp.where(xp.abs(d) < 1e-12, 1e-12, d)
    t1 = (bb_min[None] - o) * inv
    t2 = (bb_max[None] - o) * inv
    tlo = xp.minimum(t1, t2).max(-1)
    thi = xp.maximum(t1, t2).min(-1)
    return (tlo <= thi) & (thi > 0) & (tlo < t_cap[:, None])


_VEH_CUDA_SRC = r"""
#include <math_constants.h>

// Per-vehicle ray/triangle intersection (Moller-Trumbore).
// One thread per (ray, triangle) pair; per-ray vehicle pose (motion
// distortion) rotation/translation is applied inside the kernel.
// Output: hit distance per pair (R x T), +inf = miss.
// NOTE: keep this source pure ASCII -- CuPy writes it to a temp file
// using the system locale encoding (GBK on zh-CN Windows).
extern "C" __global__
void mt_hit(const float* __restrict__ o, const float* __restrict__ d,
            const float* __restrict__ pos, const float* __restrict__ yaw,
            const float* __restrict__ V0, const float* __restrict__ E1,
            const float* __restrict__ E2, const float* __restrict__ cap,
            const float r_min, const int T, const int R,
            float* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= R) return;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const float c = cosf(yaw[r]);
    const float s = sinf(yaw[r]);
    const float px = pos[3 * r + 0];
    const float py = pos[3 * r + 1];
    const float pz = pos[3 * r + 2];

    const float v0x = V0[9 * t + 0], v0y = V0[9 * t + 1], v0z = V0[9 * t + 2];
    const float e1x0 = E1[3 * t + 0], e1y0 = E1[3 * t + 1], e1z0 = E1[3 * t + 2];
    const float e2x0 = E2[3 * t + 0], e2y0 = E2[3 * t + 1], e2z0 = E2[3 * t + 2];

    // 绕 z 旋转 + 平移到世界系
    const float ax = c * v0x - s * v0y + px;
    const float ay = s * v0x + c * v0y + py;
    const float az = v0z + pz;
    const float e1x = c * e1x0 - s * e1y0;
    const float e1y = s * e1x0 + c * e1y0;
    const float e1z = e1z0;
    const float e2x = c * e2x0 - s * e2y0;
    const float e2y = s * e2x0 + c * e2y0;
    const float e2z = e2z0;

    const float ox = o[3 * r + 0], oy = o[3 * r + 1], oz = o[3 * r + 2];
    const float dx = d[3 * r + 0], dy = d[3 * r + 1], dz = d[3 * r + 2];

    const float pvx = dy * e2z - dz * e2y;
    const float pvy = dz * e2x - dx * e2z;
    const float pvz = dx * e2y - dy * e2x;
    const float det = e1x * pvx + e1y * pvy + e1z * pvz;

    float val = CUDART_INF_F;
    if (fabsf(det) > 1e-9f) {
        const float inv = 1.0f / det;
        const float tvx = ox - ax, tvy = oy - ay, tvz = oz - az;
        const float qvx = tvy * e1z - tvz * e1y;
        const float qvy = tvz * e1x - tvx * e1z;
        const float qvz = tvx * e1y - tvy * e1x;
        const float u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv;
        const float w = (dx * qvx + dy * qvy + dz * qvz) * inv;
        const float tt = (e2x * qvx + e2y * qvy + e2z * qvz) * inv;
        if (u >= 0.f && w >= 0.f && (u + w) <= 1.f &&
            tt > r_min && tt < cap[r] && tt < 1e9f)
            val = tt;
    }
    out[r * T + t] = val;
}
"""

_veh_kernel = None


def _get_veh_kernel():
    global _veh_kernel
    if _veh_kernel is None:
        _veh_kernel = cp.RawKernel(_VEH_CUDA_SRC, "mt_hit")
    return _veh_kernel


def _raycast_vehicle_gpu(origins, dirs, t_cap, tri_verts, pos, yaw, r_min):
    """GPU 融合版: 单 kernel 完成旋转 + Möller-Trumbore, 再逐光线取最近命中."""
    R = len(origins)
    T = len(tri_verts)
    t_hit = cp.full(R, np.inf, cp.float32)
    tri_idx = cp.full(R, -1, cp.int32)
    if R == 0 or T == 0:
        return t_hit, tri_idx

    o = cp.ascontiguousarray(origins, cp.float32)
    d = cp.ascontiguousarray(dirs, cp.float32)
    p = cp.ascontiguousarray(pos, cp.float32)
    yw = cp.ascontiguousarray(yaw, cp.float32)
    cap = cp.ascontiguousarray(t_cap, cp.float32)
    V0 = cp.ascontiguousarray(tri_verts, cp.float32).reshape(T, 3, 3)
    e1 = cp.ascontiguousarray(V0[:, 1, :] - V0[:, 0, :])
    e2 = cp.ascontiguousarray(V0[:, 2, :] - V0[:, 0, :])

    kern = _get_veh_kernel()
    block = 256
    rc = max(1, 8_000_000 // T)          # 控制 (r,T) 命中距离缓冲 <= 32 MB
    for s0 in range(0, R, rc):
        r = min(rc, R - s0)
        buf = cp.empty((r, T), cp.float32)
        grid = ((T + block - 1) // block, r)
        kern(grid, (block, 1, 1),
             (o[s0:], d[s0:], p[s0:], yw[s0:], V0, e1, e2, cap[s0:],
              np.float32(r_min), np.int32(T), np.int32(r), buf))
        tt = buf.min(axis=1)
        ti = cp.where(cp.isfinite(tt), buf.argmin(axis=1).astype(cp.int32),
                      cp.int32(-1))
        t_hit[s0:s0 + r] = tt
        tri_idx[s0:s0 + r] = ti
    return t_hit, tri_idx


def raycast_vehicle(origins, dirs, t_cap, tri_verts, tri_nrm, pos, yaw,
                    r_min):
    """单个车辆: 逐光线位姿 (运动伪影) 的光线-三角形求交.

    origins/dirs (R,3) 已是该车辆候选光线; t_cap (R,) 当前最佳命中距离;
    pos (R,3), yaw (R,) 为各光线时刻的车辆位姿 (插值);
    tri_verts (T,3,3) 车辆系三角网格 (车底中心原点), tri_nrm (T,3).
    返回 (t_hit (R,) inf=未命中, tri_idx (R,) -1=未命中).
    """
    if HAS_CP and isinstance(origins, cp.ndarray):
        return _raycast_vehicle_gpu(origins, dirs, t_cap, tri_verts, pos, yaw,
                                    r_min)
    xp = np
    R, T = len(origins), len(tri_verts)
    t_hit = xp.full(R, np.inf, xp.float32)
    tri_idx = xp.full(R, -1, xp.int32)
    V0 = xp.asarray(tri_verts)                    # (T,3,3)
    v0 = V0[:, 0, :]; e1 = V0[:, 1, :] - v0; e2 = V0[:, 2, :] - v0
    n0 = xp.asarray(tri_nrm)
    rc = max(8, int(2_000_000 // max(T, 1)))      # 控制 r*T 内存
    for s0 in range(0, R, rc):
        s1 = min(s0 + rc, R)
        o = xp.asarray(origins[s0:s1])[:, None, :]        # (r,1,3)
        d = xp.asarray(dirs[s0:s1])[:, None, :]
        p = xp.asarray(pos[s0:s1])[:, None, :]
        # 逐光线绕 z 旋转 (直接三角公式, 避免 matmul/cublas 依赖)
        c = xp.cos(xp.asarray(yaw[s0:s1]))[:, None]         # (r,1)
        s = xp.sin(xp.asarray(yaw[s0:s1]))[:, None]
        v0r = xp.asarray(v0)[None]                          # (1,T,3)
        a = xp.stack([c * v0r[..., 0] - s * v0r[..., 1] + p[..., 0],
                      s * v0r[..., 0] + c * v0r[..., 1] + p[..., 1],
                      v0r[..., 2] + p[..., 2]], -1)         # (r,T,3)
        e1_ = xp.asarray(e1)[None]; e2_ = xp.asarray(e2)[None]

        def _rot(v3):
            vx = c * v3[..., 0] - s * v3[..., 1]
            vy = s * v3[..., 0] + c * v3[..., 1]
            vz = xp.broadcast_to(v3[..., 2], vx.shape)
            return vx, vy, vz

        ax_, ay_, az_ = _rot(v0r)
        a = xp.stack([ax_ + p[..., 0], ay_ + p[..., 1], az_ + p[..., 2]], -1)
        e1r = xp.stack(_rot(e1_), -1)
        e2r = xp.stack(_rot(e2_), -1)
        pv = xp.cross(xp.broadcast_to(d, (s1 - s0, T, 3)), e2r)
        det = (e1r * pv).sum(-1)
        tv = o - a
        qv = xp.cross(tv, e1r)
        eps = 1e-9
        adet = xp.abs(det)
        safe = xp.where(adet > eps, det, 1.0)
        u = (tv * pv).sum(-1) / safe            # Möller–Trumbore: u = tvec·pvec/det
        w = (xp.broadcast_to(d, (s1 - s0, T, 3)) * qv).sum(-1) / safe
        t = (e2r * qv).sum(-1) / safe
        cap = xp.asarray(t_cap[s0:s1])[:, None]
        hit = (adet > eps) & (u >= 0) & (w >= 0) & (u + w <= 1) & \
              (t > r_min) & (t < cap) & (t < 1e9)
        if not bool(hit.any()):
            continue
        rr = xp.broadcast_to(xp.arange(s1 - s0, dtype=xp.int32)[:, None], hit.shape)[hit]
        tt = t[hit]; tidx = xp.broadcast_to(xp.arange(T, dtype=xp.int32)[None, :], hit.shape)[hit]
        order = _lexsort2(xp, tt, rr)
        rr2, tt2, ti2 = rr[order], tt[order], tidx[order]
        _, first = xp.unique(rr2, return_index=True)
        t_hit[s0 + rr2[first]] = tt2[first]
        tri_idx[s0 + rr2[first]] = ti2[first]
    return t_hit, tri_idx
