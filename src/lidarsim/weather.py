# -*- coding: utf-8 -*-
"""雨 / 雪天气下激光雷达点云仿真 (基于物理机制).

模型要点
--------
1) 粒子尺寸分布 (单位体积内直径 D 的数密度谱 N(D), m^-3 mm^-1):
   - 雨: Marshall-Palmer  N(D) = N0 * exp(-Lambda D),
         N0 = 8000,  Lambda = 4.1 * R^(-0.21)   (R: 降雨率 mm/h)
   - 雪: Gunn-Marshall  N0 = 3.8e3 * S^(-0.87), Lambda = 2.55 * S^(-0.48)
         (S: 降雪率, 水当量 mm/h)

2) 消光 (对真实目标回波的双程衰减):
   905nm 波长下雨滴/雪花直径 D >> lambda, 处于几何光学区, Q_ext -> 2:
       sigma_ext = (pi/4) * Q_ext * ∫ D^2 N(D) dD
                 = pi * N0 / Lambda^3 * 1e-6        [1/m]
   回波功率再乘 exp(-2 * sigma_ext * R) (双程).

3) 降水杂波 (虚假回波):
   单位体积后向散射系数 beta = (Q_b/2) * sigma_ext,
   Q_b 为几何光学区平均后向散射效率 (雨 0.06 / 雪 0.10).
   每束光产生杂波为沿光束的非齐次泊松过程, 线密度
       lambda(z) = clutter_factor * beta * exp(-2*sigma_ext*z),
   深度 z 按上述强度包络采样; 杂波强度复用主仿真的漫反射雷达方程,
   等效反射率 rho_rain/rho_snow 很小, 因此杂波多为低强度点.

杂波点 label = -1 (与 0=环境 / vid>0=车辆 区分).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class WeatherConfig:
    kind: str = "none"          # none | rain | snow
    rate: float = 10.0          # 降雨率 mm/h; 降雪率 (水当量) mm/h
    clutter_range: float = 30.0     # 杂波最远出现距离 (m)
    clutter_factor: float = 60.0    # 杂波线密度系数 (含接收孔径/阈值综合因子)

    def active(self) -> bool:
        return self.kind in ("rain", "snow") and self.rate > 0

    # ---- 粒子谱参数 ----
    def _mp_params(self):
        if self.kind == "rain":
            n0 = 8000.0
            lam = 4.1 * self.rate ** (-0.21)
        else:  # snow, Gunn-Marshall
            n0 = 3.8e3 * self.rate ** (-0.87)
            lam = 2.55 * self.rate ** (-0.48)
        return n0, max(lam, 1e-6)

    # ---- 派生物理量 ----
    def sigma_ext(self) -> float:
        """消光系数 [1/m] (几何光学极限 Q_ext=2)."""
        if not self.active():
            return 0.0
        n0, lam = self._mp_params()
        return float(np.pi * n0 / lam ** 3 * 1e-6)

    def beta_back(self) -> float:
        """体后向散射系数 [1/m/sr]."""
        q_b = 0.06 if self.kind == "rain" else 0.10
        return 0.5 * q_b * self.sigma_ext()

    def rho_clutter(self) -> float:
        """杂波等效反射率 (进入强度模型)."""
        return 0.02 if self.kind == "rain" else 0.03


def generate_clutter(cfg: WeatherConfig, spec, o_np, d_np, tf_np, r_hit, rng,
                     max_points: int = 300000):
    """沿光束采样降水杂波.

    cfg: WeatherConfig; spec: LidarSpec
    o_np/d_np: (R,3) 光束原点/单位方向; tf_np: (R,) 光束相对时刻
    r_hit: (R,) 各光束真实命中距离 (np.inf 表示无命中)
    返回 (xyz(N,3) f32, inten(N,) f32, t_beam(N,) f32, ranges(N,) f32) 或 None
    """
    if not cfg.active():
        return None
    sigma = cfg.sigma_ext()
    lam = cfg.clutter_factor * cfg.beta_back()
    r_avail = np.where(np.isfinite(r_hit), r_hit, spec.range_max)
    r_avail = np.clip(r_avail, 0.0, cfg.clutter_range).astype(np.float64)
    # 每束光杂波数 ~ Poisson(lambda * r_avail)  (弱消光近似, 常数线密度)
    counts = rng.poisson(lam * r_avail)
    total = int(counts.sum())
    if total == 0:
        return None
    if total > max_points:                    # 极端参数下的安全上限
        keep_idx = np.random.default_rng(0).choice(total, max_points, replace=False)
    ray_idx = np.repeat(np.arange(len(counts)), counts)
    if total > max_points:
        ray_idx = ray_idx[np.sort(keep_idx)]
        total = max_points
    # 深度采样: 线密度包络 ∝ exp(-2*sigma*z), 逆变换采样
    u = rng.random(total)
    ra = r_avail[ray_idx]
    if sigma > 1e-12:
        k = 2.0 * sigma
        z = -np.log1p(-u * (1.0 - np.exp(-k * ra))) / k
    else:
        z = u * ra
    z = np.clip(z, spec.range_min, None)
    xyz = o_np[ray_idx] + z[:, None].astype(np.float32) * d_np[ray_idx]
    # 杂波强度: 复用漫反射雷达方程 (各向同性散射, cos=1)
    atm = np.exp(-2.0 * (spec.alpha_atm + sigma) * z)
    p = cfg.rho_clutter() * atm / (1.0 + (z / spec.range_ref) ** 2)
    dn = np.clip(spec.gain * p, 0.0, 1.0) * 255.0
    dn += rng.normal(0.0, spec.noise_intensity, total)
    keep = (p > spec.dropout_thr) & (rng.random(total) > spec.dropout_random)
    if not keep.any():
        return None
    return (xyz[keep].astype(np.float32),
            np.clip(dn[keep], 0, 255).astype(np.float32),
            tf_np[ray_idx][keep].astype(np.float32),
            z[keep].astype(np.float32))
