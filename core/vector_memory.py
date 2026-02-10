"""
R3000 向量记忆体
按市场状态 × 方向 × 入场/离场 存储 (A,B,C) 点云
提供偏离指数（相似度）计算

结构：
  memory[regime][direction][phase] = [(A, B, C), ...]
  
  regime    = "强多头" | "弱多头" | ... | "未知"
  direction = "LONG" | "SHORT"
  phase     = "ENTRY" | "EXIT"

偏离指数算法：
  1. 计算新点到点云中最近 k 个点的平均欧氏距离 d_avg
  2. 偏离指数 = max(0, 1 - d_avg / D_spread) * 100%
     其中 D_spread 是该点云的平均点间距
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PointCloud:
    """单个点云（一个 regime + direction + phase 的所有点）"""
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    _spread: float = -1.0  # 缓存的离散度
    _points_array: Optional[np.ndarray] = None  # 缓存的 numpy 数组

    def add(self, a: float, b: float, c: float):
        """添加一个坐标点"""
        self.points.append((a, b, c))
        self._spread = -1.0  # 清除缓存
        self._points_array = None

    def count(self) -> int:
        return len(self.points)

    def as_array(self) -> np.ndarray:
        """获取 numpy 数组 (n, 3)"""
        if self._points_array is None or len(self._points_array) != len(self.points):
            if not self.points:
                self._points_array = np.zeros((0, 3))
            else:
                self._points_array = np.array(self.points, dtype=np.float64)
        return self._points_array

    @property
    def spread(self) -> float:
        """
        计算点云的平均离散度（所有点到质心的平均距离）
        用于归一化偏离指数
        """
        if self._spread >= 0:
            return self._spread

        arr = self.as_array()
        if len(arr) < 2:
            self._spread = 1.0  # 默认值，避免除零
            return self._spread

        centroid = arr.mean(axis=0)
        dists = np.sqrt(np.sum((arr - centroid) ** 2, axis=1))
        self._spread = max(float(dists.mean()), 1e-9)
        return self._spread

    def centroid(self) -> Tuple[float, float, float]:
        """点云质心"""
        arr = self.as_array()
        if len(arr) == 0:
            return (0.0, 0.0, 0.0)
        c = arr.mean(axis=0)
        return (round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3))


class VectorMemory:
    """
    向量记忆体

    使用方式：
        memory = VectorMemory()
        memory.add_point("强多头", "LONG", "ENTRY", 1.234, -0.567, 0.891)
        similarity = memory.similarity("强多头", "LONG", "ENTRY", 1.200, -0.550, 0.880)
    """

    def __init__(self, k_neighbors: int = 5, min_points: int = 3):
        """
        Args:
            k_neighbors: 计算偏离指数时取最近的 k 个点
            min_points: 点云最少多少个点才计算偏离指数
        """
        self.k = k_neighbors
        self.min_points = min_points
        self._memory: Dict[str, Dict[str, Dict[str, PointCloud]]] = {}

    def _ensure_path(self, regime: str, direction: str, phase: str) -> PointCloud:
        """确保路径存在"""
        if regime not in self._memory:
            self._memory[regime] = {}
        if direction not in self._memory[regime]:
            self._memory[regime][direction] = {}
        if phase not in self._memory[regime][direction]:
            self._memory[regime][direction][phase] = PointCloud()
        return self._memory[regime][direction][phase]

    def add_point(self, regime: str, direction: str, phase: str,
                  a: float, b: float, c: float):
        """添加一个坐标点到记忆体"""
        cloud = self._ensure_path(regime, direction, phase)
        cloud.add(a, b, c)

    def get_cloud(self, regime: str, direction: str, phase: str) -> PointCloud:
        """获取指定的点云"""
        return self._ensure_path(regime, direction, phase)

    def similarity(self, regime: str, direction: str, phase: str,
                   a: float, b: float, c: float) -> float:
        """
        计算偏离指数（相似度百分比）

        Args:
            regime, direction, phase: 记忆体键
            a, b, c: 新点坐标

        Returns:
            0~100 的相似度百分比
            100 = 完全重叠于点云中心
            0   = 远离点云
            -1  = 数据不足，无法计算
        """
        cloud = self.get_cloud(regime, direction, phase)
        if cloud.count() < self.min_points:
            return -1.0

        arr = cloud.as_array()
        new_point = np.array([a, b, c], dtype=np.float64)

        # 计算到所有点的距离
        dists = np.sqrt(np.sum((arr - new_point) ** 2, axis=1))

        # 取最近 k 个点的平均距离
        k = min(self.k, len(dists))
        nearest_k = np.sort(dists)[:k]
        d_avg = float(nearest_k.mean())

        # 归一化
        spread = cloud.spread
        similarity = max(0.0, 1.0 - d_avg / spread) * 100.0

        return round(similarity, 1)

    def distance(self, regime: str, direction: str, phase: str,
                 a: float, b: float, c: float) -> float:
        """
        计算到记忆体的绝对距离（最近k点平均距离）

        Returns:
            绝对距离值，-1 表示数据不足
        """
        cloud = self.get_cloud(regime, direction, phase)
        if cloud.count() < self.min_points:
            return -1.0

        arr = cloud.as_array()
        new_point = np.array([a, b, c], dtype=np.float64)
        dists = np.sqrt(np.sum((arr - new_point) ** 2, axis=1))
        k = min(self.k, len(dists))
        nearest_k = np.sort(dists)[:k]
        return round(float(nearest_k.mean()), 3)

    # ── 统计 ───────────────────────────────────────────
    def get_stats(self) -> Dict:
        """获取记忆体统计概览"""
        stats = {}
        for regime, dirs in self._memory.items():
            stats[regime] = {}
            for direction, phases in dirs.items():
                stats[regime][direction] = {}
                for phase, cloud in phases.items():
                    stats[regime][direction][phase] = {
                        "count": cloud.count(),
                        "spread": round(cloud.spread, 3),
                        "centroid": cloud.centroid(),
                    }
        return stats

    def get_all_points_for_plot(self, regime: str = None) -> Dict:
        """
        获取用于3D绘图的所有点数据

        Args:
            regime: 可选，筛选特定市场状态

        Returns:
            {
                "LONG_ENTRY": np.ndarray (n, 3),
                "LONG_EXIT": np.ndarray (n, 3),
                "SHORT_ENTRY": np.ndarray (n, 3),
                "SHORT_EXIT": np.ndarray (n, 3),
            }
        """
        result = {
            "LONG_ENTRY": [],
            "LONG_EXIT": [],
            "SHORT_ENTRY": [],
            "SHORT_EXIT": [],
        }

        regimes_to_scan = [regime] if regime else list(self._memory.keys())

        for r in regimes_to_scan:
            if r not in self._memory:
                continue
            for direction in ["LONG", "SHORT"]:
                if direction not in self._memory[r]:
                    continue
                for phase in ["ENTRY", "EXIT"]:
                    if phase not in self._memory[r][direction]:
                        continue
                    cloud = self._memory[r][direction][phase]
                    key = f"{direction}_{phase}"
                    if cloud.count() > 0:
                        result[key].append(cloud.as_array())

        # 合并
        for key in result:
            if result[key]:
                result[key] = np.vstack(result[key])
            else:
                result[key] = np.zeros((0, 3))

        return result

    def get_regime_list(self) -> List[str]:
        """获取已有记录的市场状态列表"""
        return list(self._memory.keys())

    def clear(self):
        """清空记忆体"""
        self._memory.clear()

    def total_points(self) -> int:
        """总点数"""
        total = 0
        for dirs in self._memory.values():
            for phases in dirs.values():
                for cloud in phases.values():
                    total += cloud.count()
        return total
