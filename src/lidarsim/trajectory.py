# -*- coding: utf-8 -*-
"""PTV VISSIM 车辆轨迹记录 (15 列空白分隔) 解析与逐帧/亚帧插值.

列定义 (与 Vissim Vehicle Record 对应):
  1 模拟时间 [s]      2 O-D 累计距离 [m]   3 occupancy        4 车辆编号
  5 车辆类型 vType    6 路段 link          7-9 前端点 x,y,z   10-12 后端点 x,y,z
  13 速度 [km/h]      14 加速度 [m/s^2]    15 车宽 [m]
文件可能带 UTF-8 BOM.
"""
from dataclasses import dataclass
import numpy as np

COLS = dict(time=0, odom=1, occ=2, vid=3, vtype=4, link=5,
            fx=6, fy=7, fz=8, rx=9, ry=10, rz=11,
            speed_kmh=12, accel=13, width=14)

# SUMO vClass/type -> (长, 宽) 默认尺寸 [m] (FCD 未含尺寸时按类别取默认)
_SUMO_DIMS = {
    "car": (5.0, 1.8), "sedan": (5.0, 1.8), "passenger": (5.0, 1.8),
    "suv": (4.8, 1.9), "hatchback": (4.2, 1.75),
    "bus": (12.0, 2.5), "coach": (12.0, 2.55),
    "truck": (7.5, 2.5), "trailer": (10.0, 2.55), "delivery": (6.0, 2.2),
    "van": (5.5, 2.0),
    "motorcycle": (2.2, 0.8), "moped": (2.0, 0.8),
    "bicycle": (1.8, 0.65), "pedestrian": (0.5, 0.5),
    "emergency": (5.5, 1.9), "taxi": (5.0, 1.8),
}


def _sumo_dims(vtype: str, attrs: dict) -> tuple[float, float]:
    """SUMO 车辆 (长, 宽): 优先取 FCD 属性 length/width, 否则按类别默认."""
    if "length" in attrs and "width" in attrs:
        return float(attrs["length"]), float(attrs["width"])
    if "length" in attrs:
        L = float(attrs["length"])
        return L, _SUMO_DIMS.get(str(vtype).lower(), (5.0, 1.8))[1]
    return _SUMO_DIMS.get(str(vtype).lower(), (5.0, 1.8))


@dataclass
class Track:
    """单车轨迹 (按时间升序). 前后端点插值 -> 任意时刻位姿."""
    vid: int
    vtype: int
    t: np.ndarray            # (n,)
    front: np.ndarray        # (n,2) x,y
    rear: np.ndarray         # (n,2)
    width: float

    def length(self) -> float:
        return float(np.median(np.hypot(*(self.front - self.rear).T)))

    def pose_at(self, tq: np.ndarray):
        """tq: (m,) 任意时刻 -> pos(m,3 前后中点, z=0 占位), yaw(m,), front/rear(m,2).

        yaw 为行进方向 (rear->front) 与 +x 轴夹角, 已展开避免 2pi 跳变.
        """
        tq = np.clip(tq, self.t[0], self.t[-1])
        # 角度展开
        yaw_raw = np.arctan2(self.front[:, 1] - self.rear[:, 1],
                             self.front[:, 0] - self.rear[:, 0])
        yaw_u = np.unwrap(yaw_raw)
        fu = np.stack([np.interp(tq, self.t, self.front[:, i]) for i in range(2)], 1)
        ru = np.stack([np.interp(tq, self.t, self.rear[:, i]) for i in range(2)], 1)
        yaw = np.interp(tq, self.t, yaw_u)
        return (fu + ru) * 0.5, yaw, fu, ru


class Trajectories:
    def __init__(self, records: np.ndarray):
        c = COLS
        self.time = records[:, c["time"]].astype(np.float64)
        self.vid = records[:, c["vid"]].astype(np.int64)
        self.vtype = records[:, c["vtype"]].astype(np.int64)
        self.link = records[:, c["link"]].astype(np.int64)
        self.width = records[:, c["width"]].astype(np.float64)
        self.front = records[:, [c["fx"], c["fy"]]].astype(np.float64)
        self.rear = records[:, [c["rx"], c["ry"]]].astype(np.float64)
        self.speed = records[:, c["speed_kmh"]].astype(np.float64)
        self.source_format = "vissim"
        self.vid_map: dict[int, str] = {}      # vid -> 原始字符串 id (SUMO)
        self.vtype_map: dict[int, str] = {}    # vtype -> 原始类型名 (SUMO)
        self._build_tracks()

    @classmethod
    def load(cls, path: str) -> "Trajectories":
        ext = path.lower().rsplit(".", 1)[-1]
        if ext == "xml":
            return cls.load_sumo(path)
        rec = np.loadtxt(path, encoding="utf-8-sig")
        if rec.ndim == 1:
            rec = rec[None, :]
        if rec.shape[1] < 15:
            raise ValueError(f"期望 15 列 Vissim 轨迹记录, 实际 {rec.shape[1]} 列: {path}")
        return cls(rec)

    @classmethod
    def load_sumo(cls, path: str) -> "Trajectories":
        """SUMO FCD 导出 XML (--fcd-output).

        逐 timestep 读取 <vehicle id x y angle speed type [z] [length] [width]>;
        SUMO angle 单位为度, 0=北(+y), 顺时针 -> 航向 dir=(sin,cos).
        前后端点 = 中心 +/- dir*L/2; 速度 m/s -> km/h.
        字符串 id/type 分别枚举为整数 vid/vtype, 映射存于 vid_map/vtype_map.
        """
        import xml.etree.ElementTree as ET

        rows = []
        vid_map: dict[str, int] = {}
        vt_map: dict[str, int] = {}
        cur_t = 0.0
        for ev, elem in ET.iterparse(path, events=("start", "end")):
            tag = elem.tag.rsplit("}", 1)[-1]
            if ev == "start" and tag == "timestep":
                cur_t = float(elem.attrib.get("time", 0.0))
            elif ev == "end" and tag == "vehicle":
                a = elem.attrib
                sid = a.get("id", "0")
                stp = a.get("type", "car")
                if sid not in vid_map:
                    vid_map[sid] = len(vid_map)
                if stp not in vt_map:
                    vt_map[stp] = len(vt_map)
                ang = np.deg2rad(float(a.get("angle", 0.0)))
                d = np.array([np.sin(ang), np.cos(ang)])   # 0=北, 顺时针
                x, y = float(a.get("x", 0.0)), float(a.get("y", 0.0))
                z = float(a.get("z", 0.0))
                L, W = _sumo_dims(stp, a)
                spd = float(a.get("speed", 0.0)) * 3.6
                c = np.array([x, y])
                fr = c + d * (L / 2.0)
                rr = c - d * (L / 2.0)
                rows.append([cur_t, 0.0, 0.0, vid_map[sid], vt_map[stp], 0,
                             fr[0], fr[1], z, rr[0], rr[1], z, spd, 0.0, W])
                elem.clear()
        if not rows:
            raise ValueError(f"SUMO FCD 中未解析到车辆记录: {path}")
        rec = np.asarray(rows, np.float64)
        obj = cls(rec)
        obj.source_format = "sumo"
        obj.vid_map = {v: k for k, v in vid_map.items()}
        obj.vtype_map = {v: k for k, v in vt_map.items()}
        return obj

    def _build_tracks(self):
        order = np.lexsort((self.time, self.vid))
        vid_s, t_s = self.vid[order], self.time[order]
        # 分组边界
        bounds = np.flatnonzero(np.diff(vid_s)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(vid_s)]])
        self.tracks: dict[int, Track] = {}
        for s, e in zip(starts, ends):
            v = int(vid_s[s])
            idx = order[s:e]
            o2 = np.argsort(self.time[idx], kind="stable")
            idx = idx[o2]
            self.tracks[v] = Track(
                vid=v, vtype=int(self.vtype[idx[0]]), t=self.time[idx],
                front=self.front[idx], rear=self.rear[idx],
                width=float(np.median(self.width[idx])))

    # ---- 查询 ----
    @property
    def frame_times(self) -> np.ndarray:
        return np.unique(self.time)

    def active_at(self, t: float) -> np.ndarray:
        return self.vid[self.time == t]

    def vtype_dims(self) -> dict[int, tuple[float, float]]:
        """每个 vType 的中位 (长度, 宽度), 用于网格模型匹配."""
        out = {}
        for v, tr in self.tracks.items():
            L = tr.length()
            out.setdefault(tr.vtype, []).append((L, tr.width))
        return {k: (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v])))
                for k, v in out.items()}
