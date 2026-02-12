"""
R3000 轨迹匹配器
混合匹配策略：余弦相似度快速粗筛 → DTW精确匹配

三阶段匹配：
  1. 入场匹配：当前60根K线 vs 历史入场前60根
  2. 持仓监控：持仓轨迹 vs 历史持仓轨迹（偏离度监控）
  3. 离场匹配：当前30根K线 vs 历史离场前30根
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAJECTORY_CONFIG
from core.trajectory_engine import TrajectoryTemplate

# 尝试导入高性能DTW库
try:
    from dtaidistance import dtw as dtw_lib
    from dtaidistance import dtw_ndim
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False
    print("[TrajectoryMatcher] dtaidistance未安装, 使用numpy简化版DTW")


@dataclass
class MatchResult:
    """匹配结果"""
    matched: bool               # 是否匹配成功
    best_template: Optional[TrajectoryTemplate]  # 最佳匹配模板
    cosine_sim: float           # 余弦相似度 (0~1)
    dtw_distance: float         # DTW距离（归一化后）
    dtw_similarity: float       # DTW相似度 (0~1)
    top_k_indices: List[int]    # 余弦粗筛的Top-K模板索引
    all_dtw_results: List[Tuple[int, float]]  # 所有DTW结果 (idx, distance)


class TrajectoryMatcher:
    """
    轨迹匹配器

    使用方式：
        matcher = TrajectoryMatcher()
        result = matcher.match_entry(current_window, candidates, cosine_th, dtw_th)
        if result.matched:
            # 开仓
    """

    def __init__(self, cosine_top_k: int = None, dtw_radius: int = None):
        self.cosine_top_k = cosine_top_k or TRAJECTORY_CONFIG["COSINE_TOP_K"]
        self.dtw_radius = dtw_radius or TRAJECTORY_CONFIG["DTW_RADIUS"]

    # ── 入场匹配 ───────────────────────────────────────
    def match_entry(self, current_window: np.ndarray,
                    candidates: List[TrajectoryTemplate],
                    cosine_threshold: float = 0.6,
                    dtw_threshold: float = 0.5) -> MatchResult:
        """
        入场匹配：当前轨迹 vs 历史入场前轨迹

        Args:
            current_window: (window_size, 32) 当前K线窗口
            candidates: 候选模板列表
            cosine_threshold: 余弦相似度阈值 (0~1)
            dtw_threshold: DTW归一化距离阈值（越小越严格）

        Returns:
            MatchResult 匹配结果
        """
        if len(candidates) == 0 or current_window.size == 0:
            return MatchResult(
                matched=False, best_template=None,
                cosine_sim=0.0, dtw_distance=999.0, dtw_similarity=0.0,
                top_k_indices=[], all_dtw_results=[]
            )

        # 1. z-score标准化并展平当前窗口
        current_flat = self._zscore_flatten(current_window).astype(np.float32)
        q_norm = np.linalg.norm(current_flat)
        if q_norm < 1e-12:
            return MatchResult(
                matched=False, best_template=None,
                cosine_sim=0.0, dtw_distance=999.0, dtw_similarity=0.0,
                top_k_indices=[], all_dtw_results=[]
            )

        # 2. 向量化余弦粗筛 (NumPy 矩阵运算)
        # 预先检查是否有缓存的矩阵，如果没有则根据长度分组计算
        # 注意：此处简化实现，优先处理长度一致的情况
        q_len = len(current_flat)
        
        # 将相同长度的候选模板提取出来做批处理
        potential_indices = []
        templates_to_process = []
        for i, t in enumerate(candidates):
            if t.pre_entry_flat.size == q_len:
                potential_indices.append(i)
                templates_to_process.append(t.pre_entry_flat)
        
        cosine_scores = []
        if templates_to_process:
            # 转换为 (N, L) 矩阵
            tpl_matrix = np.stack(templates_to_process).astype(np.float32)
            # 批量计算点积
            dots = np.dot(tpl_matrix, current_flat)
            # 批量计算范数
            tpl_norms = np.linalg.norm(tpl_matrix, axis=1)
            # 计算余弦相似度
            denom = tpl_norms * q_norm
            valid = denom > 1e-12
            sims = np.zeros(len(dots))
            sims[valid] = dots[valid] / denom[valid]
            
            for idx_in_batch, sim in enumerate(sims):
                cosine_scores.append((potential_indices[idx_in_batch], float(sim)))
        
        # 处理异常长度（回退到慢速循环，通常很少）
        if len(cosine_scores) < len(candidates):
            processed_indices = set(potential_indices)
            for i, template in enumerate(candidates):
                if i in processed_indices or template.pre_entry_flat.size == 0:
                    continue
                t_flat = self._align_vectors(current_flat, template.pre_entry_flat)
                sim = self._cosine_similarity(current_flat, t_flat)
                cosine_scores.append((i, sim))

        if not cosine_scores:
            return MatchResult(
                matched=False, best_template=None,
                cosine_sim=0.0, dtw_distance=999.0, dtw_similarity=0.0,
                top_k_indices=[], all_dtw_results=[]
            )

        # 按余弦相似度排序，取Top-K
        cosine_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = cosine_scores[:self.cosine_top_k]
        top_k_indices = [x[0] for x in top_k]

        # 过滤低于阈值的
        top_k = [(idx, sim) for idx, sim in top_k if sim >= cosine_threshold]
        if not top_k:
            best_cos = cosine_scores[0] if cosine_scores else (0, 0.0)
            return MatchResult(
                matched=False, best_template=None,
                cosine_sim=best_cos[1], dtw_distance=999.0, dtw_similarity=0.0,
                top_k_indices=top_k_indices, all_dtw_results=[]
            )

        # 3. DTW精筛
        dtw_results = []
        for idx, cosine_sim in top_k:
            template = candidates[idx]
            dtw_dist = self._compute_dtw_distance(current_window, template.pre_entry)
            dtw_results.append((idx, dtw_dist, cosine_sim))

        # 按DTW距离排序
        dtw_results.sort(key=lambda x: x[1])
        all_dtw = [(idx, dist) for idx, dist, _ in dtw_results]

        # 检查最佳匹配是否满足阈值
        best_idx, best_dtw_dist, best_cosine = dtw_results[0]
        dtw_similarity = max(0.0, 1.0 - best_dtw_dist)

        if best_dtw_dist <= dtw_threshold:
            return MatchResult(
                matched=True,
                best_template=candidates[best_idx],
                cosine_sim=best_cosine,
                dtw_distance=best_dtw_dist,
                dtw_similarity=dtw_similarity,
                top_k_indices=top_k_indices,
                all_dtw_results=all_dtw
            )
        else:
            return MatchResult(
                matched=False,
                best_template=candidates[best_idx],
                cosine_sim=best_cosine,
                dtw_distance=best_dtw_dist,
                dtw_similarity=dtw_similarity,
                top_k_indices=top_k_indices,
                all_dtw_results=all_dtw
            )

    # ── 持仓监控 ───────────────────────────────────────
    def monitor_holding(self, holding_so_far: np.ndarray,
                        matched_template: TrajectoryTemplate,
                        divergence_limit: float = 0.7) -> Tuple[float, bool]:
        """
        持仓监控：计算当前持仓轨迹与模板持仓轨迹的偏离度

        Args:
            holding_so_far: (n, 32) 从入场到当前的持仓轨迹
            matched_template: 匹配的模板
            divergence_limit: 偏离度上限，超过则建议离场

        Returns:
            (divergence, should_exit): 偏离度(0~1) 和是否建议离场
        """
        if holding_so_far.size == 0 or matched_template.holding.size == 0:
            return 0.0, False

        # 取模板持仓轨迹的对应长度部分
        current_len = len(holding_so_far)
        template_len = len(matched_template.holding)

        if current_len <= template_len:
            # 当前持仓时间未超过模板，比较对应部分
            template_segment = matched_template.holding[:current_len]
        else:
            # 当前持仓时间超过模板，只比较模板长度
            template_segment = matched_template.holding
            holding_so_far = holding_so_far[:template_len]

        # 计算DTW距离作为偏离度
        dtw_dist = self._compute_dtw_distance(holding_so_far, template_segment)
        divergence = min(1.0, dtw_dist)

        should_exit = divergence > divergence_limit
        return divergence, should_exit

    # ── 离场匹配 ───────────────────────────────────────
    def match_exit(self, recent_window: np.ndarray,
                   candidates: List[TrajectoryTemplate],
                   threshold: float = 0.5) -> Tuple[bool, float]:
        """
        离场匹配：当前轨迹 vs 历史离场前轨迹

        Args:
            recent_window: (window_size, 32) 最近的K线窗口
            candidates: 候选模板列表
            threshold: DTW距离阈值

        Returns:
            (should_exit, best_similarity): 是否应该离场，最佳匹配相似度
        """
        if len(candidates) == 0 or recent_window.size == 0:
            return False, 0.0

        best_dist = 999.0
        for template in candidates:
            if template.pre_exit.size == 0:
                continue
            # 对齐长度
            min_len = min(len(recent_window), len(template.pre_exit))
            current_seg = recent_window[-min_len:]
            template_seg = template.pre_exit[-min_len:]

            dtw_dist = self._compute_dtw_distance(current_seg, template_seg)
            best_dist = min(best_dist, dtw_dist)

        similarity = max(0.0, 1.0 - best_dist)
        should_exit = best_dist <= threshold
        return should_exit, similarity

    # ── 辅助方法 ───────────────────────────────────────
    @staticmethod
    def _zscore_flatten(matrix: np.ndarray) -> np.ndarray:
        """z-score标准化后展平"""
        flat = matrix.flatten()
        std = flat.std()
        if std < 1e-9:
            return flat - flat.mean()
        return (flat - flat.mean()) / std

    @staticmethod
    def _align_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """对齐两个向量长度（截取较长的）"""
        if len(v1) == len(v2):
            return v2
        min_len = min(len(v1), len(v2))
        return v2[:min_len]

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        if len(v1) != len(v2):
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        计算多变量DTW距离（归一化到0~1范围）

        Args:
            seq1, seq2: (time, features) 多变量时间序列

        Returns:
            归一化的DTW距离
        """
        if seq1.size == 0 or seq2.size == 0:
            return 1.0

        if HAS_DTAIDISTANCE:
            return self._dtw_dtaidistance(seq1, seq2)
        else:
            return self._dtw_numpy(seq1, seq2)

    def _dtw_dtaidistance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """使用dtaidistance库计算DTW"""
        try:
            # 多变量DTW
            distance = dtw_ndim.distance(seq1, seq2, window=self.dtw_radius)
            # 归一化
            max_len = max(len(seq1), len(seq2))
            n_features = seq1.shape[1] if seq1.ndim > 1 else 1
            # 估算最大可能距离（用于归一化）
            max_dist = max_len * np.sqrt(n_features) * 2  # 粗略估计
            return min(1.0, distance / max_dist)
        except Exception as e:
            # 回退到numpy版本
            return self._dtw_numpy(seq1, seq2)

    def _dtw_numpy(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        简化版DTW（numpy实现）
        使用欧氏距离，带Sakoe-Chiba约束加速
        """
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return 1.0

        # 初始化代价矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        # Sakoe-Chiba带约束
        window = max(self.dtw_radius, abs(n - m))

        for i in range(1, n + 1):
            j_start = max(1, i - window)
            j_end = min(m, i + window) + 1
            for j in range(j_start, j_end):
                cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # 插入
                    dtw_matrix[i, j - 1],      # 删除
                    dtw_matrix[i - 1, j - 1]   # 匹配
                )

        distance = dtw_matrix[n, m]

        # 归一化
        max_len = max(n, m)
        n_features = seq1.shape[1] if seq1.ndim > 1 else 1
        max_dist = max_len * np.sqrt(n_features) * 2
        return min(1.0, distance / max_dist)


# ── 快速批量余弦相似度 ─────────────────────────────────
def batch_cosine_similarity(query: np.ndarray,
                             templates: List[TrajectoryTemplate],
                             top_k: int = 20) -> List[Tuple[int, float]]:
    """
    批量计算余弦相似度并返回Top-K

    Args:
        query: (window, 32) 查询轨迹
        templates: 模板列表
        top_k: 返回前K个

    Returns:
        [(template_idx, similarity), ...]
    """
    query_flat = TrajectoryMatcher._zscore_flatten(query)
    query_norm = np.linalg.norm(query_flat)
    if query_norm < 1e-9:
        return []

    scores = []
    for i, template in enumerate(templates):
        if template.pre_entry_flat.size == 0:
            continue

        t_flat = template.pre_entry_flat
        if len(t_flat) != len(query_flat):
            min_len = min(len(t_flat), len(query_flat))
            t_flat = t_flat[:min_len]
            q_flat = query_flat[:min_len]
        else:
            q_flat = query_flat

        t_norm = np.linalg.norm(t_flat)
        if t_norm < 1e-9:
            continue

        sim = np.dot(q_flat, t_flat) / (np.linalg.norm(q_flat) * t_norm)
        scores.append((i, float(sim)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
