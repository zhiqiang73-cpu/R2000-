"""
R3000 模板聚类引擎
将大量碎片化模板聚类成少量"原型"，提高匹配效率和统计可靠性

核心思路：
  - 每个模板的完整交易 (pre_entry + holding + pre_exit) 压缩成一个统计向量
  - 使用 K-Means 聚类得到若干原型
  - 每个原型代表一类反复出现的交易模式
  - 原型自带历史胜率、平均收益等统计信息

存储：
  - 原型保存在 data/prototypes/ 目录
  - 支持增量更新（重新聚类）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os
import pickle
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAJECTORY_CONFIG, CONFIDENCE_CONFIG, SIMILARITY_CONFIG, FEATURE_WEIGHTS_CONFIG

# 默认原型存储路径
DEFAULT_PROTOTYPE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prototypes"
)

# ══════════════════════════════════════════════════════════════════════════
# 原型库版本控制
# ══════════════════════════════════════════════════════════════════════════
# 版本历史：
#   "1.0" - 初始版本，基础原型结构
#   "2.0" - 添加置信度系统 (confidence, profit_std, regime_purity)
#   "3.0" - 指纹3D图匹配系统，添加多维相似度支持:
#           - weighted_mean: 加权均值向量（从 pre_entry_centroid 迁移）
#           - trend_vector: 趋势特征向量
#           - time_segments: 时间分段特征 (early/mid/late)
#           - volatility: 波动率向量
PROTOTYPE_LIBRARY_VERSION = "3.0"

# 版本迁移支持的最低版本
PROTOTYPE_MIN_SUPPORTED_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════════════════
# 多维相似度计算器 (Multi-Similarity Calculator)
# ══════════════════════════════════════════════════════════════════════════

class MultiSimilarityCalculator:
    """
    多维相似度计算器
    
    将三种相似度度量融合，解决单一余弦相似度在高维空间中区分度不足的问题：
    
    1. 余弦相似度 (Cosine): 捕捉特征变化方向是否一致
    2. 欧氏距离 (Euclidean): 捕捉特征数值是否接近
    3. 动态时间规整 (DTW): 捕捉时间序列形态是否匹配
    
    融合公式：
        综合分数 = w1 * cos_sim + w2 * (1 - norm_euclidean) + w3 * dtw_sim
        最终分数 = 综合分数 * 置信度系数
    
    使用方式：
        calculator = MultiSimilarityCalculator()
        result = calculator.compute_similarity(current_window, prototype)
    """
    
    def __init__(self, 
                 cosine_weight: float = None,
                 euclidean_weight: float = None,
                 dtw_weight: float = None,
                 dtw_radius: int = None,
                 apply_feature_weights: bool = True):
        """
        初始化多维相似度计算器
        
        Args:
            cosine_weight: 余弦相似度权重，默认从配置读取
            euclidean_weight: 欧氏距离权重，默认从配置读取
            dtw_weight: DTW权重，默认从配置读取
            dtw_radius: DTW Sakoe-Chiba band 半径，默认从配置读取
            apply_feature_weights: 是否应用特征层权重（Layer A/B/C）
        """
        # 从配置读取默认值
        self.cosine_weight = cosine_weight if cosine_weight is not None else SIMILARITY_CONFIG.get("COSINE_WEIGHT", 0.30)
        self.euclidean_weight = euclidean_weight if euclidean_weight is not None else SIMILARITY_CONFIG.get("EUCLIDEAN_WEIGHT", 0.40)
        self.dtw_weight = dtw_weight if dtw_weight is not None else SIMILARITY_CONFIG.get("DTW_WEIGHT", 0.30)
        self.dtw_radius = dtw_radius if dtw_radius is not None else SIMILARITY_CONFIG.get("DTW_RADIUS", 10)
        
        # 归一化权重确保总和为1
        total_weight = self.cosine_weight + self.euclidean_weight + self.dtw_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.cosine_weight /= total_weight
            self.euclidean_weight /= total_weight
            self.dtw_weight /= total_weight
        
        # 特征权重
        self.apply_feature_weights = apply_feature_weights
        self._feature_weights = None
        self._default_feature_weights = None   # 默认权重备份（用于 reset 和 L2 正则化基准）
        self._dynamic_weights_active = False   # 是否正在使用动态注入权重
        if apply_feature_weights:
            self._build_feature_weights()
        
        # 欧氏距离归一化上限（兼容多种键名格式）
        self.euclidean_max_dist = SIMILARITY_CONFIG.get("EUCLIDEAN_MAX_DISTANCE",
                                   SIMILARITY_CONFIG.get("EUCLIDEAN_MAX_DIST", 10.0))
        self.dtw_max_seq_len = SIMILARITY_CONFIG.get("DTW_MAX_SEQ_LEN", 100)
        # DTW 距离归一化上限（累积距离较大，典型值约5000-8000）
        self.dtw_max_dist = SIMILARITY_CONFIG.get("DTW_MAX_DISTANCE", 8000.0)
    
    def _build_feature_weights(self):
        """
        构建 32 维特征权重向量
        
        根据配置中的层权重，构建完整的特征权重向量：
        - Layer A (前16维): 即时信号，权重 1.5x
        - Layer B (中间10维): 动量变化，权重 1.2x
        - Layer C (后6维): 结构位置，权重 1.0x
        """
        n_a = FEATURE_WEIGHTS_CONFIG.get("N_A", 16)
        n_b = FEATURE_WEIGHTS_CONFIG.get("N_B", 10)
        n_c = FEATURE_WEIGHTS_CONFIG.get("N_C", 6)
        
        w_a = FEATURE_WEIGHTS_CONFIG.get("LAYER_A_WEIGHT", 1.5)
        w_b = FEATURE_WEIGHTS_CONFIG.get("LAYER_B_WEIGHT", 1.2)
        w_c = FEATURE_WEIGHTS_CONFIG.get("LAYER_C_WEIGHT", 1.0)
        
        # 构建权重向量 (32,)
        weights = np.concatenate([
            np.full(n_a, w_a),
            np.full(n_b, w_b),
            np.full(n_c, w_c),
        ])
        
        # 归一化使权重向量的 L2 范数为 sqrt(32)（保持原始尺度）
        weights = weights / weights.mean()
        self._feature_weights = weights
    
    def set_dynamic_weights(self, weights: np.ndarray):
        """
        动态注入特征权重（覆盖默认的 Layer A/B/C 权重）
        
        由 WF Evolution 引擎调用，使用进化后的 32 维权重向量替换
        默认的 Layer A(1.5) / B(1.2) / C(1.0) 分层权重。
        
        Args:
            weights: 32 维特征权重向量（numpy array）。
                     会被归一化处理（保持均值为1、整体尺度不变）。
        
        Raises:
            ValueError: 当 weights 不是 32 维向量时抛出。
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != 32:
            raise ValueError(
                f"set_dynamic_weights 需要 32 维权重向量，收到 {len(w)} 维"
            )
        # 确保所有权重为正
        w = np.clip(w, 0.01, None)
        # 归一化：保持均值为1，与 _build_feature_weights 逻辑一致
        w = w / w.mean()
        self._feature_weights = w
        self.apply_feature_weights = True
    
    def apply_weights(self, vector: np.ndarray) -> np.ndarray:
        """
        对特征向量应用层权重
        
        Args:
            vector: 原始特征向量 (32,) 或 (N, 32)
        
        Returns:
            加权后的特征向量
        """
        if not self.apply_feature_weights or self._feature_weights is None:
            return vector
        
        if vector.ndim == 1:
            if len(vector) != len(self._feature_weights):
                return vector
            return vector * self._feature_weights
        elif vector.ndim == 2:
            if vector.shape[1] != len(self._feature_weights):
                return vector
            return vector * self._feature_weights[np.newaxis, :]
        
        return vector
    
    # ══════════════════════════════════════════════════════════════════════
    # 1. 余弦相似度 (Cosine Similarity)
    # ══════════════════════════════════════════════════════════════════════
    
    def compute_cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        捕捉特征变化的方向是否一致，对尺度不敏感。
        
        Args:
            v1, v2: 输入向量，可以是 (D,) 或 (T, D) 形状
                    如果是 2D，先沿时间轴取均值
        
        Returns:
            相似度值，范围 [-1, 1]，通常映射到 [0, 1]
        """
        # 如果是序列，先取均值得到统计向量
        if v1.ndim == 2:
            v1 = v1.mean(axis=0)
        if v2.ndim == 2:
            v2 = v2.mean(axis=0)
        
        if v1.size == 0 or v2.size == 0:
            return 0.0
        
        # 应用特征权重
        v1_weighted = self.apply_weights(v1)
        v2_weighted = self.apply_weights(v2)
        
        # 对齐长度
        min_len = min(len(v1_weighted), len(v2_weighted))
        v1_weighted = v1_weighted[:min_len]
        v2_weighted = v2_weighted[:min_len]
        
        # 计算余弦相似度
        norm1 = np.linalg.norm(v1_weighted)
        norm2 = np.linalg.norm(v2_weighted)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        cos_sim = float(np.dot(v1_weighted, v2_weighted) / (norm1 * norm2))
        
        # 将 [-1, 1] 映射到 [0, 1]（假设我们关心正相关）
        return max(0.0, cos_sim)
    
    # ══════════════════════════════════════════════════════════════════════
    # 2. 欧氏距离 (Euclidean Distance)
    # ══════════════════════════════════════════════════════════════════════
    
    def compute_euclidean_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        计算基于欧氏距离的相似度
        
        捕捉特征数值是否接近。距离越小，相似度越高。
        
        Args:
            v1, v2: 输入向量，可以是 (D,) 或 (T, D) 形状
                    如果是 2D，先沿时间轴取均值
        
        Returns:
            相似度值，范围 [0, 1]
        """
        # 如果是序列，先取均值得到统计向量
        if v1.ndim == 2:
            v1 = v1.mean(axis=0)
        if v2.ndim == 2:
            v2 = v2.mean(axis=0)
        
        if v1.size == 0 or v2.size == 0:
            return 0.0
        
        # 应用特征权重
        v1_weighted = self.apply_weights(v1)
        v2_weighted = self.apply_weights(v2)
        
        # 对齐长度
        min_len = min(len(v1_weighted), len(v2_weighted))
        v1_weighted = v1_weighted[:min_len]
        v2_weighted = v2_weighted[:min_len]
        
        # 计算欧氏距离
        euclidean_dist = np.linalg.norm(v1_weighted - v2_weighted)
        
        # 归一化到 [0, 1]：使用 sqrt(D) * max_feature_range 作为归一化因子
        # 这里使用配置的最大距离
        normalized_dist = min(1.0, euclidean_dist / self.euclidean_max_dist)
        
        # 转换为相似度：距离越小，相似度越高
        return 1.0 - normalized_dist
    
    # ══════════════════════════════════════════════════════════════════════
    # 3. 动态时间规整 DTW (Dynamic Time Warping)
    # ══════════════════════════════════════════════════════════════════════
    
    def compute_dtw_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        计算两个多变量时间序列的 DTW 相似度
        
        捕捉时间序列的形态是否匹配，允许时间轴上的弹性对齐。
        
        Args:
            seq1, seq2: 输入序列，形状 (T, D) 或 (D,)
        
        Returns:
            相似度值，范围 [0, 1]
        """
        if seq1.size == 0 or seq2.size == 0:
            return 0.0
        
        # 确保是 2D
        if seq1.ndim == 1:
            seq1 = seq1.reshape(1, -1)
        if seq2.ndim == 1:
            seq2 = seq2.reshape(1, -1)
        
        # 应用特征权重
        seq1_weighted = self.apply_weights(seq1)
        seq2_weighted = self.apply_weights(seq2)
        
        # 截断过长序列
        if len(seq1_weighted) > self.dtw_max_seq_len:
            seq1_weighted = seq1_weighted[-self.dtw_max_seq_len:]
        if len(seq2_weighted) > self.dtw_max_seq_len:
            seq2_weighted = seq2_weighted[-self.dtw_max_seq_len:]
        
        n, m = len(seq1_weighted), len(seq2_weighted)
        if n == 0 or m == 0:
            return 0.0
        
        # 初始化 DTW 代价矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Sakoe-Chiba band 约束（加速计算）
        window = max(self.dtw_radius, abs(n - m))
        
        # 填充 DTW 矩阵
        for i in range(1, n + 1):
            j_start = max(1, i - window)
            j_end = min(m, i + window) + 1
            for j in range(j_start, j_end):
                # 计算当前点对的距离
                cost = np.linalg.norm(seq1_weighted[i - 1] - seq2_weighted[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # 插入
                    dtw_matrix[i, j - 1],      # 删除
                    dtw_matrix[i - 1, j - 1]   # 匹配
                )
        
        dtw_distance = dtw_matrix[n, m]
        
        # 归一化：使用配置的最大距离值
        # 注意：DTW累积距离随序列长度和特征维度增加而增加
        # 典型值约为 5000-8000（60步 * 80-100每步距离）
        if self.dtw_max_dist <= 0:
            return 0.0
        
        normalized_dist = min(1.0, dtw_distance / self.dtw_max_dist)
        
        # 转换为相似度
        return 1.0 - normalized_dist
    
    # ══════════════════════════════════════════════════════════════════════
    # 融合计算
    # ══════════════════════════════════════════════════════════════════════
    
    def compute_similarity(self, 
                           current_window: np.ndarray,
                           prototype: 'Prototype',
                           include_dtw: bool = True) -> Dict:
        """
        计算当前窗口与原型的多维融合相似度
        
        Args:
            current_window: 当前特征窗口 (T, 32) 或 (32,)
            prototype: Prototype 实例
            include_dtw: 是否包含 DTW 计算（可关闭以加速）
        
        Returns:
            {
                "combined_score": float,      # 融合相似度 [0, 1]
                "cosine_similarity": float,   # 余弦相似度
                "euclidean_similarity": float,# 欧氏相似度
                "dtw_similarity": float,      # DTW 相似度
                "confidence": float,          # 原型置信度
                "final_score": float,         # 最终分数（含置信度）
            }
        """
        result = {
            "combined_score": 0.0,
            "cosine_similarity": 0.0,
            "euclidean_similarity": 0.0,
            "dtw_similarity": 0.0,
            "confidence": 1.0,
            "final_score": 0.0,
        }
        
        if current_window.size == 0 or prototype is None:
            return result
        
        # 获取原型的 pre_entry 中心向量
        proto_centroid = prototype.pre_entry_centroid
        if proto_centroid.size == 0:
            return result
        
        # 1. 计算余弦相似度
        cosine_sim = self.compute_cosine_similarity(current_window, proto_centroid)
        result["cosine_similarity"] = cosine_sim
        
        # 2. 计算欧氏相似度
        euclidean_sim = self.compute_euclidean_similarity(current_window, proto_centroid)
        result["euclidean_similarity"] = euclidean_sim
        
        # 3. 计算 DTW 相似度（如果启用且原型有代表序列）
        dtw_sim = 0.5  # 默认中性值
        if include_dtw and prototype.representative_pre_entry.size > 0:
            dtw_sim = self.compute_dtw_similarity(current_window, prototype.representative_pre_entry)
        result["dtw_similarity"] = dtw_sim
        
        # 4. 加权融合
        if include_dtw:
            combined = (
                self.cosine_weight * cosine_sim +
                self.euclidean_weight * euclidean_sim +
                self.dtw_weight * dtw_sim
            )
        else:
            # 不使用 DTW 时，重新归一化权重
            cos_w = self.cosine_weight / (self.cosine_weight + self.euclidean_weight)
            euc_w = self.euclidean_weight / (self.cosine_weight + self.euclidean_weight)
            combined = cos_w * cosine_sim + euc_w * euclidean_sim
        
        result["combined_score"] = combined
        
        # 5. 计算置信度
        confidence = self.compute_confidence(prototype)
        result["confidence"] = confidence
        
        # 6. 最终分数（综合分数 × 置信度）
        result["final_score"] = combined * confidence
        
        return result
    
    def compute_confidence(self, prototype: 'Prototype') -> float:
        """
        计算原型的置信度系数
        
        置信度 = 
            w1 * 历史胜率 +
            w2 * 样本稳定性 +
            w3 * min(样本量/目标量, 1) +
            w4 * 市场状态一致性
        
        Args:
            prototype: Prototype 实例
        
        Returns:
            置信度系数，范围 [0, 1]
        """
        if prototype is None:
            return 0.5
        
        # 获取置信度配置（兼容多种键名格式）
        w_win_rate = CONFIDENCE_CONFIG.get("WIN_RATE_WEIGHT", 
                     CONFIDENCE_CONFIG.get("WEIGHT_WIN_RATE", 0.40))
        w_stability = CONFIDENCE_CONFIG.get("STABILITY_WEIGHT",
                      CONFIDENCE_CONFIG.get("WEIGHT_STABILITY", 0.30))
        w_sample = CONFIDENCE_CONFIG.get("SAMPLE_COUNT_WEIGHT",
                   CONFIDENCE_CONFIG.get("WEIGHT_SAMPLE_COUNT", 0.20))
        w_regime = CONFIDENCE_CONFIG.get("REGIME_WEIGHT",
                   CONFIDENCE_CONFIG.get("WEIGHT_REGIME_CONSISTENCY", 0.10))
        sample_target = CONFIDENCE_CONFIG.get("SAMPLE_COUNT_FULL",
                        CONFIDENCE_CONFIG.get("SAMPLE_COUNT_TARGET", 10))
        
        # 1. 历史胜率（直接使用）
        win_rate = prototype.win_rate if prototype.win_rate > 0 else 0.5
        
        # 2. 样本稳定性（基于成员交易收益率的变异系数）
        stability = 1.0
        if prototype.member_trade_stats and len(prototype.member_trade_stats) > 1:
            profits = [stat[0] for stat in prototype.member_trade_stats]
            mean_profit = np.mean(profits)
            std_profit = np.std(profits)
            
            # 使用配置的稳定性阈值
            std_threshold = CONFIDENCE_CONFIG.get("STABILITY_STD_THRESHOLD", 0.15)
            if std_profit <= std_threshold:
                stability = 1.0
            else:
                # 超过阈值时按比例降低稳定性分数
                stability = max(0.0, 1.0 - (std_profit - std_threshold) / std_threshold)
        
        # 3. 样本量充足度
        sample_count = prototype.member_count
        sample_sufficiency = min(1.0, sample_count / sample_target)
        
        # 4. 市场状态一致性（如果有 WF 验证结果）
        regime_consistency = 0.5  # 默认中性
        if prototype.verified and prototype.wf_grade == "合格":
            regime_consistency = 0.9
        elif prototype.wf_grade == "待观察":
            regime_consistency = 0.6
        elif prototype.wf_grade == "淘汰":
            regime_consistency = 0.3
        
        # 加权综合
        confidence = (
            w_win_rate * win_rate +
            w_stability * stability +
            w_sample * sample_sufficiency +
            w_regime * regime_consistency
        )
        
        return min(1.0, max(0.0, confidence))
    
    def batch_compute_similarity(self, 
                                  current_window: np.ndarray,
                                  prototypes: List['Prototype'],
                                  include_dtw: bool = False) -> List[Dict]:
        """
        批量计算当前窗口与多个原型的相似度
        
        Args:
            current_window: 当前特征窗口 (T, 32)
            prototypes: Prototype 列表
            include_dtw: 是否包含 DTW（批量时默认关闭以加速）
        
        Returns:
            相似度结果列表，按 final_score 降序排列
        """
        results = []
        for proto in prototypes:
            sim_result = self.compute_similarity(current_window, proto, include_dtw=include_dtw)
            sim_result["prototype"] = proto
            results.append(sim_result)
        
        # 按最终分数降序排列
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    
    def vectorized_cosine_match(self, 
                                 current_vec: np.ndarray,
                                 centroid_matrix: np.ndarray) -> np.ndarray:
        """
        【性能优化】向量化余弦相似度计算
        
        一次性计算当前向量与所有原型中心的余弦相似度。
        
        Args:
            current_vec: 当前特征向量 (32,)，已归一化
            centroid_matrix: 原型中心矩阵 (N, 32)，已归一化
        
        Returns:
            相似度数组 (N,)
        """
        if current_vec.size == 0 or centroid_matrix.size == 0:
            return np.array([])
        
        # 应用特征权重
        current_weighted = self.apply_weights(current_vec)
        centroids_weighted = self.apply_weights(centroid_matrix)
        
        # 归一化
        c_norm = np.linalg.norm(current_weighted)
        if c_norm < 1e-9:
            return np.zeros(len(centroid_matrix))
        current_unit = current_weighted / c_norm
        
        # 矩阵运算
        norms = np.linalg.norm(centroids_weighted, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        centroids_unit = centroids_weighted / norms
        
        # 批量点积
        similarities = centroids_unit @ current_unit
        
        return np.maximum(0.0, similarities)
    
    def get_weights_info(self) -> Dict:
        """获取当前权重配置信息"""
        return {
            "cosine_weight": self.cosine_weight,
            "euclidean_weight": self.euclidean_weight,
            "dtw_weight": self.dtw_weight,
            "dtw_radius": self.dtw_radius,
            "apply_feature_weights": self.apply_feature_weights,
            "feature_weights": self._feature_weights.tolist() if self._feature_weights is not None else None,
            "dynamic_weights_active": self._dynamic_weights_active,
            "default_feature_weights": self._default_feature_weights.tolist() if self._default_feature_weights is not None else None,
        }
    
    # ══════════════════════════════════════════════════════════════════════
    # 动态权重注入（供 WF Evolution 引擎使用）
    # ══════════════════════════════════════════════════════════════════════
    
    # 特征语义组定义：将 32 维特征分为 8 个语义组
    # 每组中的特征在交易含义上高度相关，共享一个优化权重
    # 组顺序: rsi, macd, volatility, momentum, volume, trend, structure, adx
    #
    # 索引参照 (Layer A: 0-15, Layer B: 16-25, Layer C: 26-31):
    #   0:rsi_14  1:rsi_6  2:macd_hist  3:kdj_k  4:kdj_j  5:roc
    #   6:atr_ratio  7:boll_width  8:boll_position  9:upper_shadow  10:lower_shadow
    #   11:ema12_dev  12:ema26_dev  13:adx  14:volume_ratio  15:obv_slope
    #   16:d_rsi3  17:d_rsi5  18:d_macd3  19:d_macd5  20:d_atr  21:d_boll_w
    #   22:d_ema12_slope  23:d_ema26_slope  24:d_vol_ratio  25:d_adx
    #   26:price_in_range  27:dist_high  28:dist_low  29:amp_ratio
    #   30:price_vs_20h  31:price_vs_20l
    FEATURE_GROUPS = {
        "rsi":        [0, 1, 16, 17],               # rsi_14, rsi_6, delta_rsi_3, delta_rsi_5
        "macd":       [2, 18, 19],                   # macd_hist, delta_macd_3, delta_macd_5
        "volatility": [6, 7, 8, 20, 21],             # atr_ratio, boll_width, boll_position, delta_atr_rate, delta_boll_width
        "momentum":   [3, 4, 5],                     # kdj_k, kdj_j, roc
        "volume":     [14, 15, 24],                  # volume_ratio, obv_slope, delta_volume_ratio
        "trend":      [9, 10, 11, 12, 22, 23],       # shadow_ratios, ema_devs, ema_slope_changes
        "structure":  [26, 27, 28, 29, 30, 31],      # Layer C: 价格结构位置特征
        "adx":        [13, 25],                      # adx, delta_adx
    }
    
    # 有序组名列表（用于 group 模式下的索引映射）
    FEATURE_GROUP_ORDER = ["rsi", "macd", "volatility", "momentum", "volume", "trend", "structure", "adx"]
    
    def set_dynamic_weights(self, weights: np.ndarray, mode: str = "direct") -> None:
        """
        注入动态特征权重（供 WF Evolution 引擎使用）
        
        支持两种模式：
        1. "direct": 直接提供 32 维权重向量，替换当前特征权重
        2. "group":  提供 8 维语义组乘数，与默认 Layer A/B/C 权重逐元素相乘
        
        group 模式下的计算过程：
            expanded[i] = group_multiplier[group_of_feature_i]   (32维)
            final_weights = default_weights * expanded           (32维)
            final_weights /= mean(final_weights)                 (归一化保持尺度)
        
        Args:
            weights: 权重向量
                - mode="direct": shape (32,), 每个特征的独立权重
                - mode="group":  shape (8,), 每个语义组的乘数，典型范围 [0.1, 3.0]
                  组顺序: [rsi, macd, volatility, momentum, volume, trend, structure, adx]
            mode: "direct" 或 "group"
        
        Raises:
            ValueError: 权重维度不匹配或模式无效
        """
        weights = np.asarray(weights, dtype=np.float64)
        
        if mode == "direct":
            if weights.shape != (32,):
                raise ValueError(f"direct 模式需要 32 维权重向量，收到 shape={weights.shape}")
            
            # 保存默认权重（仅在首次调用时）
            self._ensure_default_weights_saved()
            self._feature_weights = weights.copy()
            self._dynamic_weights_active = True
            
        elif mode == "group":
            if weights.shape != (8,):
                raise ValueError(
                    f"group 模式需要 8 维组乘数向量，收到 shape={weights.shape}。"
                    f"组顺序: {self.FEATURE_GROUP_ORDER}"
                )
            
            # 保存默认权重（仅在首次调用时）
            self._ensure_default_weights_saved()
            
            # 扩展为 32 维并与默认权重相乘
            expanded = self.expand_group_weights(weights)
            self._feature_weights = self._default_feature_weights * expanded
            
            # 归一化使权重均值为 1（保持原始尺度不膨胀）
            mean_w = self._feature_weights.mean()
            if mean_w > 1e-9:
                self._feature_weights = self._feature_weights / mean_w
            
            self._dynamic_weights_active = True
        else:
            raise ValueError(f"无效模式: '{mode}'，支持 'direct' 或 'group'")
    
    def reset_to_default_weights(self) -> None:
        """
        恢复为配置文件中的默认 Layer A/B/C 权重
        
        如果之前保存了默认权重，则从备份恢复；否则重新构建。
        """
        if self._default_feature_weights is not None:
            self._feature_weights = self._default_feature_weights.copy()
        else:
            self._build_feature_weights()
        self._dynamic_weights_active = False
    
    def get_default_weights(self) -> np.ndarray:
        """
        获取默认权重向量（用于 L2 正则化基准比较）
        
        WF Evolution 的 L2 正则化项：
            penalty = lambda * ||w_current - w_default||^2
        
        Returns:
            默认 32 维特征权重向量（副本，修改不影响内部状态）
        """
        if self._default_feature_weights is not None:
            return self._default_feature_weights.copy()
        
        # 未设置过动态权重，当前权重即为默认
        if self._feature_weights is not None:
            return self._feature_weights.copy()
        
        # 兜底：重新构建
        self._build_feature_weights()
        return self._feature_weights.copy()
    
    def _ensure_default_weights_saved(self) -> None:
        """确保默认权重已备份（在首次注入动态权重前调用）"""
        if self._default_feature_weights is not None:
            return  # 已保存
        
        if self._feature_weights is not None:
            self._default_feature_weights = self._feature_weights.copy()
        else:
            self._build_feature_weights()
            self._default_feature_weights = self._feature_weights.copy()
    
    @staticmethod
    def expand_group_weights(group_multipliers: np.ndarray) -> np.ndarray:
        """
        将 8 维语义组乘数扩展为 32 维特征权重向量
        
        每个组内的所有特征共享同一乘数值。
        
        Args:
            group_multipliers: shape (8,), 各组乘数
                组顺序: [rsi, macd, volatility, momentum, volume, trend, structure, adx]
        
        Returns:
            shape (32,) 的扩展权重向量
        """
        group_multipliers = np.asarray(group_multipliers, dtype=np.float64)
        if group_multipliers.shape != (8,):
            raise ValueError(f"需要 8 维组乘数，收到 shape={group_multipliers.shape}")
        
        expanded = np.ones(32, dtype=np.float64)
        for i, group_name in enumerate(MultiSimilarityCalculator.FEATURE_GROUP_ORDER):
            indices = MultiSimilarityCalculator.FEATURE_GROUPS[group_name]
            expanded[indices] = group_multipliers[i]
        
        return expanded
    
    @staticmethod
    def get_feature_group_definitions() -> Dict[str, List[int]]:
        """
        获取特征语义组定义
        
        Returns:
            字典，key=组名, value=该组包含的特征索引列表
            共 8 组覆盖全部 32 维特征
        """
        return {k: list(v) for k, v in MultiSimilarityCalculator.FEATURE_GROUPS.items()}
    
    @property
    def is_dynamic_weights_active(self) -> bool:
        """当前是否使用动态权重（由 WF Evolution 注入）"""
        return self._dynamic_weights_active


@dataclass
class Prototype:
    """
    交易原型 - 代表一类反复出现的交易模式
    
    每个原型属于特定的 (方向, 市场状态) 组合
    
    置信度系统 (Confidence System):
    - confidence: 综合置信度评分 (0~1)
    - profit_std: 收益率标准差（样本稳定性指标）
    - regime_purity: 市场状态一致性（1.0 = 所有成员来自同一 regime）
    
    置信度计算公式：
    confidence = (
        0.4 * historical_win_rate +      # 历史胜率
        0.3 * sample_stability +          # 样本稳定性 (1 - normalized_std)
        0.2 * min(sample_count/10, 1) +   # 样本量充足度
        0.1 * regime_consistency          # 市场状态一致性
    )
    """
    prototype_id: int                   # 原型ID
    direction: str                      # "LONG" / "SHORT"
    regime: str = ""                    # 市场状态 (强多头/弱多头/震荡偏多/震荡偏空/弱空头/强空头)
    
    # 聚类中心向量（用于匹配）
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))  # 统计向量
    
    # 分段统计向量（用于三阶段匹配）
    pre_entry_centroid: np.ndarray = field(default_factory=lambda: np.array([]))  # 入场前特征中心 (32,)
    holding_centroid: np.ndarray = field(default_factory=lambda: np.array([]))    # 持仓特征中心 (64,) = mean + std
    pre_exit_centroid: np.ndarray = field(default_factory=lambda: np.array([]))   # 出场前特征中心 (32,)
    
    # 统计信息（来自聚类成员）
    member_count: int = 0               # 成员模板数量
    win_count: int = 0                  # 盈利交易数
    total_profit_pct: float = 0.0       # 总收益率
    avg_profit_pct: float = 0.0         # 平均收益率
    win_rate: float = 0.0               # 胜率
    avg_hold_bars: float = 0.0          # 平均持仓K线数
    
    # 【置信度系统】
    confidence: float = 0.0             # 综合置信度评分 (0~1)
    profit_std: float = 0.0             # 收益率标准差（稳定性指标）
    regime_purity: float = 1.0          # 市场状态一致性 (0~1)
    
    # 成员指纹（用于追溯）
    member_fingerprints: List[str] = field(default_factory=list)
    
    # Walk-Forward 验证状态
    verified: bool = False              # 是否通过 WF 验证
    wf_grade: str = ""                  # WF 评级: "合格"/"待观察"/"淘汰"/""
    wf_match_count: int = 0             # WF 中累计匹配次数
    wf_win_rate: float = 0.0            # WF 中累计胜率
    
    # 用于快速余弦匹配的展平向量
    pre_entry_flat: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 【DTW二次过滤】代表序列 - 存储最接近质心的模板的原始 pre_entry 序列
    # 形状: (window_size, 32)，用于DTW距离计算
    representative_pre_entry: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 【概率扇形图】成员交易的收益率和持仓时长分布
    # 列表 of (profit_pct, hold_bars)，用于构建真实价格路径的概率分布
    member_trade_stats: List[tuple] = field(default_factory=list)
    
    # ══════════════════════════════════════════════════════════════════════════
    # 【指纹3D图匹配系统】 - 版本 3.0 新增字段
    # ══════════════════════════════════════════════════════════════════════════
    # 加权均值向量 (32,) - 从 pre_entry_centroid 迁移，应用层级权重
    weighted_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    # 趋势特征向量 (32,) - 每个特征的时间趋势斜率
    trend_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    # 时间分段特征 - 捕获早中晚期变化
    time_segments: Dict = field(default_factory=lambda: {'early': np.array([]), 'mid': np.array([]), 'late': np.array([])})
    # 波动率向量 (32,) - 每个特征的时间标准差
    volatility: np.ndarray = field(default_factory=lambda: np.array([]))
    # 序列特征版本标记 - 用于判断是否需要迁移
    sequence_features_version: str = ""
    
    def __post_init__(self):
        """初始化时预计算展平向量和迁移旧格式"""
        if self.pre_entry_centroid.size > 0 and self.pre_entry_flat.size == 0:
            self.pre_entry_flat = self._zscore_normalize(self.pre_entry_centroid)
        
        # 【版本迁移】如果有 pre_entry_centroid 但没有 weighted_mean，自动迁移
        if self.pre_entry_centroid.size > 0 and self.weighted_mean.size == 0:
            self._migrate_to_v3()
    
    @staticmethod
    def _zscore_normalize(vec: np.ndarray) -> np.ndarray:
        """z-score 标准化"""
        if vec.size == 0:
            return vec
        std = vec.std()
        if std < 1e-9:
            return vec - vec.mean()
        return (vec - vec.mean()) / std
    
    def _migrate_to_v3(self):
        """
        【版本迁移】从 v1.0/v2.0 格式迁移到 v3.0 指纹3D图格式
        
        迁移内容：
        - pre_entry_centroid → weighted_mean（应用层级权重）
        - 生成默认 trend_vector（基于代表序列或零向量）
        - 生成默认 time_segments（基于代表序列或零向量）
        - 生成默认 volatility（基于代表序列或零向量）
        """
        from core.feature_vector import apply_weights, extract_trends
        
        # 1. 将 pre_entry_centroid 转换为 weighted_mean
        if self.pre_entry_centroid.size == 32:
            self.weighted_mean = apply_weights(self.pre_entry_centroid.copy())
        elif self.pre_entry_centroid.size > 0:
            # 非标准维度，直接复制
            self.weighted_mean = self.pre_entry_centroid.copy()
        
        # 2. 如果有代表序列，从中提取序列特征
        if self.representative_pre_entry.size > 0 and self.representative_pre_entry.ndim == 2:
            window = self.representative_pre_entry
            n_time, n_features = window.shape
            
            # 提取趋势向量
            if n_features == 32:
                self.trend_vector = extract_trends(window)
            else:
                self.trend_vector = np.zeros(32)
            
            # 提取时间分段特征
            seg_size = max(1, n_time // 3)
            early_end = seg_size
            mid_end = 2 * seg_size
            
            early_mean = window[:early_end].mean(axis=0) if early_end > 0 else np.zeros(n_features)
            mid_mean = window[early_end:mid_end].mean(axis=0) if mid_end > early_end else np.zeros(n_features)
            late_mean = window[mid_end:].mean(axis=0) if n_time > mid_end else np.zeros(n_features)
            
            # 应用权重
            if n_features == 32:
                self.time_segments = {
                    'early': apply_weights(early_mean),
                    'mid': apply_weights(mid_mean),
                    'late': apply_weights(late_mean),
                }
            else:
                self.time_segments = {
                    'early': early_mean,
                    'mid': mid_mean,
                    'late': late_mean,
                }
            
            # 提取波动率
            self.volatility = window.std(axis=0)
        else:
            # 没有代表序列，使用默认值
            self.trend_vector = np.zeros(32)
            self.time_segments = {
                'early': np.zeros(32),
                'mid': np.zeros(32),
                'late': np.zeros(32),
            }
            self.volatility = np.zeros(32)
        
        # 标记版本
        self.sequence_features_version = PROTOTYPE_LIBRARY_VERSION
    
    def get_sequence_features(self) -> Dict:
        """
        获取序列特征字典（用于多维相似度计算）
        
        兼容 _window_to_sequence_features 返回的格式
        
        Returns:
            {
                'raw_sequence': np.ndarray,     # 代表序列
                'weighted_mean': np.ndarray,    # 加权均值
                'trend_vector': np.ndarray,     # 趋势向量
                'time_segments': dict,          # 时间分段
                'volatility': np.ndarray,       # 波动率
            }
        """
        # 如果尚未迁移，先迁移
        if self.weighted_mean.size == 0 and self.pre_entry_centroid.size > 0:
            self._migrate_to_v3()
        
        return {
            'raw_sequence': self.representative_pre_entry if self.representative_pre_entry.size > 0 else np.zeros((0, 32)),
            'weighted_mean': self.weighted_mean if self.weighted_mean.size > 0 else np.zeros(32),
            'trend_vector': self.trend_vector if self.trend_vector.size > 0 else np.zeros(32),
            'time_segments': self.time_segments if self.time_segments else {'early': np.zeros(32), 'mid': np.zeros(32), 'late': np.zeros(32)},
            'volatility': self.volatility if self.volatility.size > 0 else np.zeros(32),
        }
    
    def calculate_confidence(self, 
                             member_profits: List[float] = None,
                             member_regimes: List[str] = None) -> float:
        """
        计算原型的置信度评分
        
        置信度公式：
        confidence = (
            w1 * historical_win_rate +      # 历史胜率 (40%)
            w2 * sample_stability +          # 样本稳定性 (30%)
            w3 * sample_adequacy +           # 样本量充足度 (20%)
            w4 * regime_consistency          # 市场状态一致性 (10%)
        )
        
        Args:
            member_profits: 成员交易的收益率列表（可选，用于计算稳定性）
            member_regimes: 成员交易的市场状态列表（可选，用于计算一致性）
        
        Returns:
            置信度评分 (0~1)
        """
        # 获取配置权重
        w1 = CONFIDENCE_CONFIG.get("WIN_RATE_WEIGHT", 0.40)
        w2 = CONFIDENCE_CONFIG.get("STABILITY_WEIGHT", 0.30)
        w3 = CONFIDENCE_CONFIG.get("SAMPLE_COUNT_WEIGHT", 0.20)
        w4 = CONFIDENCE_CONFIG.get("REGIME_WEIGHT", 0.10)
        
        sample_full = CONFIDENCE_CONFIG.get("SAMPLE_COUNT_FULL", 10)
        std_baseline = CONFIDENCE_CONFIG.get("STABILITY_STD_THRESHOLD", 0.15)
        
        # ══════════════════════════════════════════════════════════
        # 1. 历史胜率分数 (0~1)
        # ══════════════════════════════════════════════════════════
        win_rate_score = self.win_rate  # 已经是 0~1
        
        # ══════════════════════════════════════════════════════════
        # 2. 样本稳定性分数 (0~1)
        # 基于收益率的标准差：标准差越低，稳定性越高
        # ══════════════════════════════════════════════════════════
        if member_profits is not None and len(member_profits) > 1:
            self.profit_std = float(np.std(member_profits))
        elif self.member_trade_stats and len(self.member_trade_stats) > 1:
            profits = [p for p, _ in self.member_trade_stats]
            self.profit_std = float(np.std(profits))
        
        # 标准差越小，稳定性越高
        # stability = 1 - min(profit_std / baseline, 1)
        normalized_std = min(self.profit_std / std_baseline, 1.0) if std_baseline > 0 else 0.0
        stability_score = 1.0 - normalized_std
        
        # ══════════════════════════════════════════════════════════
        # 3. 样本量充足度分数 (0~1)
        # 样本量达到 sample_full 时为满分
        # ══════════════════════════════════════════════════════════
        sample_adequacy_score = min(self.member_count / sample_full, 1.0)
        
        # ══════════════════════════════════════════════════════════
        # 4. 市场状态一致性分数 (0~1)
        # 如果所有成员都来自同一 regime，一致性为 1
        # ══════════════════════════════════════════════════════════
        if member_regimes is not None and len(member_regimes) > 0:
            # 计算主要 regime 的占比
            regime_counts = {}
            for r in member_regimes:
                regime_counts[r] = regime_counts.get(r, 0) + 1
            max_count = max(regime_counts.values())
            self.regime_purity = max_count / len(member_regimes)
        
        regime_consistency_score = self.regime_purity
        
        # ══════════════════════════════════════════════════════════
        # 综合置信度
        # ══════════════════════════════════════════════════════════
        self.confidence = (
            w1 * win_rate_score +
            w2 * stability_score +
            w3 * sample_adequacy_score +
            w4 * regime_consistency_score
        )
        
        # 确保在 0~1 范围内
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        return self.confidence
    
    def get_confidence_level(self) -> str:
        """
        获取置信度等级描述
        
        Returns:
            "高" / "中" / "低" / "极低"
        """
        high_threshold = CONFIDENCE_CONFIG.get("HIGH_CONFIDENCE", 0.70)
        min_threshold = CONFIDENCE_CONFIG.get("MIN_CONFIDENCE", 0.30)
        mid_threshold = (high_threshold + min_threshold) / 2  # 0.50
        
        if self.confidence >= high_threshold:
            return "高"
        elif self.confidence >= mid_threshold:
            return "中"
        elif self.confidence >= min_threshold:
            return "低"
        else:
            return "极低"
    
    def get_confidence_details(self) -> dict:
        """
        获取置信度的详细分解
        
        Returns:
            {
                "confidence": float,      # 综合置信度
                "level": str,             # 置信度等级
                "win_rate": float,        # 历史胜率
                "profit_std": float,      # 收益率标准差
                "member_count": int,      # 样本数量
                "regime_purity": float,   # 市场状态一致性
                "breakdown": dict,        # 各维度分数
            }
        """
        w1 = CONFIDENCE_CONFIG.get("WIN_RATE_WEIGHT", 0.40)
        w2 = CONFIDENCE_CONFIG.get("STABILITY_WEIGHT", 0.30)
        w3 = CONFIDENCE_CONFIG.get("SAMPLE_COUNT_WEIGHT", 0.20)
        w4 = CONFIDENCE_CONFIG.get("REGIME_WEIGHT", 0.10)
        
        sample_full = CONFIDENCE_CONFIG.get("SAMPLE_COUNT_FULL", 10)
        std_baseline = CONFIDENCE_CONFIG.get("STABILITY_STD_THRESHOLD", 0.15)
        
        # 计算各维度原始分数
        win_rate_score = self.win_rate
        normalized_std = min(self.profit_std / std_baseline, 1.0) if std_baseline > 0 else 0.0
        stability_score = 1.0 - normalized_std
        sample_adequacy_score = min(self.member_count / sample_full, 1.0)
        regime_consistency_score = self.regime_purity
        
        return {
            "confidence": self.confidence,
            "level": self.get_confidence_level(),
            "win_rate": self.win_rate,
            "profit_std": self.profit_std,
            "member_count": self.member_count,
            "regime_purity": self.regime_purity,
            "breakdown": {
                "win_rate_score": win_rate_score,
                "win_rate_weighted": w1 * win_rate_score,
                "stability_score": stability_score,
                "stability_weighted": w2 * stability_score,
                "sample_adequacy_score": sample_adequacy_score,
                "sample_adequacy_weighted": w3 * sample_adequacy_score,
                "regime_consistency_score": regime_consistency_score,
                "regime_consistency_weighted": w4 * regime_consistency_score,
            }
        }
    
    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "prototype_id": self.prototype_id,
            "direction": self.direction,
            "regime": self.regime,
            "centroid": self.centroid.tolist() if self.centroid.size > 0 else [],
            "pre_entry_centroid": self.pre_entry_centroid.tolist() if self.pre_entry_centroid.size > 0 else [],
            "holding_centroid": self.holding_centroid.tolist() if self.holding_centroid.size > 0 else [],
            "pre_exit_centroid": self.pre_exit_centroid.tolist() if self.pre_exit_centroid.size > 0 else [],
            "member_count": self.member_count,
            "win_count": self.win_count,
            "total_profit_pct": self.total_profit_pct,
            "avg_profit_pct": self.avg_profit_pct,
            "win_rate": self.win_rate,
            "avg_hold_bars": self.avg_hold_bars,
            # 置信度系统
            "confidence": self.confidence,
            "profit_std": self.profit_std,
            "regime_purity": self.regime_purity,
            # 成员信息
            "member_fingerprints": self.member_fingerprints,
            "verified": self.verified,
            "wf_grade": self.wf_grade,
            "wf_match_count": self.wf_match_count,
            "wf_win_rate": self.wf_win_rate,
            # DTW 代表序列
            "representative_pre_entry": self.representative_pre_entry.tolist() if self.representative_pre_entry.size > 0 else [],
            # 概率扇形图数据
            "member_trade_stats": self.member_trade_stats,
            # ═══ 指纹3D图匹配系统 (v3.0) ═══
            "weighted_mean": self.weighted_mean.tolist() if self.weighted_mean.size > 0 else [],
            "trend_vector": self.trend_vector.tolist() if self.trend_vector.size > 0 else [],
            "time_segments": {
                k: v.tolist() if isinstance(v, np.ndarray) and v.size > 0 else []
                for k, v in self.time_segments.items()
            } if self.time_segments else {},
            "volatility": self.volatility.tolist() if self.volatility.size > 0 else [],
            "sequence_features_version": self.sequence_features_version,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Prototype':
        """从字典还原"""
        proto = cls(
            prototype_id=d["prototype_id"],
            direction=d["direction"],
            regime=d.get("regime", ""),  # 兼容旧版本
            centroid=np.array(d.get("centroid", [])),
            pre_entry_centroid=np.array(d.get("pre_entry_centroid", [])),
            holding_centroid=np.array(d.get("holding_centroid", [])),
            pre_exit_centroid=np.array(d.get("pre_exit_centroid", [])),
            member_count=d.get("member_count", 0),
            win_count=d.get("win_count", 0),
            total_profit_pct=d.get("total_profit_pct", 0.0),
            avg_profit_pct=d.get("avg_profit_pct", 0.0),
            win_rate=d.get("win_rate", 0.0),
            avg_hold_bars=d.get("avg_hold_bars", 0.0),
            # 置信度系统（兼容旧版本，旧原型没有这些字段时重新计算）
            confidence=d.get("confidence", 0.0),
            profit_std=d.get("profit_std", 0.0),
            regime_purity=d.get("regime_purity", 1.0),
            # 成员信息
            member_fingerprints=d.get("member_fingerprints", []),
            verified=d.get("verified", False),
            wf_grade=d.get("wf_grade", ""),
            wf_match_count=d.get("wf_match_count", 0),
            wf_win_rate=d.get("wf_win_rate", 0.0),
            # DTW 代表序列（兼容旧版本）
            representative_pre_entry=np.array(d.get("representative_pre_entry", [])),
            # 概率扇形图数据（兼容旧版本）
            member_trade_stats=d.get("member_trade_stats", []),
            # ═══ 指纹3D图匹配系统 (v3.0) ═══
            weighted_mean=np.array(d.get("weighted_mean", [])),
            trend_vector=np.array(d.get("trend_vector", [])),
            time_segments=cls._parse_time_segments(d.get("time_segments", {})),
            volatility=np.array(d.get("volatility", [])),
            sequence_features_version=d.get("sequence_features_version", ""),
        )
        
        # 兼容性处理：如果旧原型没有置信度，基于现有数据计算
        if proto.confidence == 0.0 and proto.member_count > 0:
            proto.calculate_confidence()
        
        # 【v3.0 迁移】如果旧原型没有 weighted_mean 但有 pre_entry_centroid，自动迁移
        # 注：__post_init__ 会自动触发迁移，但这里显式调用以确保
        if proto.weighted_mean.size == 0 and proto.pre_entry_centroid.size > 0:
            proto._migrate_to_v3()
        
        return proto
    
    @classmethod
    def _parse_time_segments(cls, segments_dict: dict) -> Dict:
        """
        解析时间分段特征字典
        
        Args:
            segments_dict: 序列化的时间分段字典
        
        Returns:
            包含 numpy 数组的时间分段字典
        """
        if not segments_dict:
            return {'early': np.array([]), 'mid': np.array([]), 'late': np.array([])}
        
        return {
            'early': np.array(segments_dict.get('early', [])),
            'mid': np.array(segments_dict.get('mid', [])),
            'late': np.array(segments_dict.get('late', [])),
        }


@dataclass
class PrototypeLibrary:
    """原型库 - 存储所有原型（聚合指纹图）"""
    long_prototypes: List[Prototype] = field(default_factory=list)
    short_prototypes: List[Prototype] = field(default_factory=list)
    
    # 元信息
    created_at: str = ""
    source_template_count: int = 0
    clustering_params: Dict = field(default_factory=dict)
    source_symbol: str = ""
    source_interval: str = ""
    
    # 进化结果（绑定到本聚合指纹图，保存/加载时一并持久化）
    evolved_full_weights: Optional[np.ndarray] = None      # 32 维（兼容旧版单组）
    evolved_fusion_threshold: Optional[float] = None
    evolved_cosine_min_threshold: Optional[float] = None
    evolved_euclidean_min_threshold: Optional[float] = None
    evolved_dtw_min_threshold: Optional[float] = None
    # 多空分开进化：做多/做空各一套
    evolved_full_weights_long: Optional[np.ndarray] = None
    evolved_fusion_threshold_long: Optional[float] = None
    evolved_cosine_min_threshold_long: Optional[float] = None
    evolved_euclidean_min_threshold_long: Optional[float] = None
    evolved_dtw_min_threshold_long: Optional[float] = None
    evolved_full_weights_short: Optional[np.ndarray] = None
    evolved_fusion_threshold_short: Optional[float] = None
    evolved_cosine_min_threshold_short: Optional[float] = None
    evolved_euclidean_min_threshold_short: Optional[float] = None
    evolved_dtw_min_threshold_short: Optional[float] = None
    
    @property
    def total_count(self) -> int:
        return len(self.long_prototypes) + len(self.short_prototypes)
    
    @staticmethod
    def _regime_direction_matches(proto: Prototype) -> bool:
        """
        严格剔除“方向-状态”反向的原型：
        - LONG 只能出现在偏多状态
        - SHORT 只能出现在偏空状态
        - 空/未知状态不做限制（交由上层处理）
        """
        if not proto.regime:
            return True
        bull_regimes = {"强多头", "弱多头", "震荡偏多", "Strong Bull", "Weak Bull", "Range Bull"}
        bear_regimes = {"强空头", "弱空头", "震荡偏空", "Strong Bear", "Weak Bear", "Range Bear"}
        if proto.direction == "LONG":
            return proto.regime in bull_regimes
        if proto.direction == "SHORT":
            return proto.regime in bear_regimes
        return True

    def get_prototypes_by_direction(self, direction: str) -> List[Prototype]:
        """获取指定方向的原型"""
        if direction == "LONG":
            return [p for p in self.long_prototypes if self._regime_direction_matches(p)]
        elif direction == "SHORT":
            return [p for p in self.short_prototypes if self._regime_direction_matches(p)]
        else:
            return []
    
    def get_prototypes_by_direction_and_regime(self, direction: str, regime: str) -> List[Prototype]:
        """
        获取指定方向和市场状态的原型
        
        Args:
            direction: "LONG" / "SHORT"
            regime: 市场状态（强多头/弱多头/震荡偏多/震荡偏空/弱空头/强空头）
        
        Returns:
            匹配的原型列表
        """
        prototypes = self.get_prototypes_by_direction(direction)
        return [p for p in prototypes if p.regime == regime and self._regime_direction_matches(p)]
    
    def get_available_regimes(self) -> List[str]:
        """获取原型库中存在的所有市场状态"""
        regimes = set()
        for p in self.get_all_prototypes():
            if p.regime:
                regimes.add(p.regime)
        return sorted(regimes)
    
    def get_prototypes_by_confidence(self, min_confidence: float = None, 
                                      max_confidence: float = None,
                                      direction: str = None) -> List[Prototype]:
        """
        按置信度筛选原型
        
        Args:
            min_confidence: 最低置信度（>=）
            max_confidence: 最高置信度（<=）
            direction: 方向过滤 (LONG/SHORT/None)
        
        Returns:
            符合条件的原型列表
        """
        if direction:
            prototypes = self.get_prototypes_by_direction(direction)
        else:
            prototypes = self.get_all_prototypes()
        
        result = []
        for p in prototypes:
            if min_confidence is not None and p.confidence < min_confidence:
                continue
            if max_confidence is not None and p.confidence > max_confidence:
                continue
            result.append(p)
        
        return result
    
    def get_high_confidence_prototypes(self, direction: str = None) -> List[Prototype]:
        """获取高置信度原型"""
        threshold = CONFIDENCE_CONFIG.get("HIGH_CONFIDENCE", 0.70)
        return self.get_prototypes_by_confidence(min_confidence=threshold, direction=direction)
    
    def get_confidence_stats(self) -> dict:
        """
        获取原型库的置信度统计
        
        Returns:
            {
                "total_count": int,
                "avg_confidence": float,
                "high_confidence_count": int,
                "mid_confidence_count": int,
                "low_confidence_count": int,
                "by_direction": {
                    "LONG": {...},
                    "SHORT": {...},
                }
            }
        """
        high_threshold = CONFIDENCE_CONFIG.get("HIGH_CONFIDENCE", 0.70)
        min_threshold = CONFIDENCE_CONFIG.get("MIN_CONFIDENCE", 0.30)
        
        def calc_stats(protos):
            if not protos:
                return {"count": 0, "avg": 0.0, "high": 0, "mid": 0, "low": 0}
            
            confidences = [p.confidence for p in protos]
            return {
                "count": len(protos),
                "avg": float(np.mean(confidences)),
                "high": sum(1 for c in confidences if c >= high_threshold),
                "mid": sum(1 for c in confidences if min_threshold <= c < high_threshold),
                "low": sum(1 for c in confidences if c < min_threshold),
            }
        
        all_protos = self.get_all_prototypes()
        long_protos = self.long_prototypes
        short_protos = self.short_prototypes
        
        all_stats = calc_stats(all_protos)
        
        return {
            "total_count": all_stats["count"],
            "avg_confidence": all_stats["avg"],
            "high_confidence_count": all_stats["high"],
            "mid_confidence_count": all_stats["mid"],
            "low_confidence_count": all_stats["low"],
            "by_direction": {
                "LONG": calc_stats(long_protos),
                "SHORT": calc_stats(short_protos),
            }
        }
    
    def apply_wf_verification(self, prototype_stats: dict,
                               min_matches: int = 3,
                               min_win_rate: float = 0.6) -> dict:
        """
        将 Walk-Forward 验证结果回写到原型库中
        
        Args:
            prototype_stats: BatchWalkForwardEngine._prototype_stats
                格式: {fp_string: {"match_count": int, "win_count": int, "total_profit": float, ...}}
            min_matches: 最小匹配次数阈值
            min_win_rate: 最小胜率阈值
        
        Returns:
            {"qualified": int, "pending": int, "eliminated": int, "total_verified": int}
        """
        counts = {"qualified": 0, "pending": 0, "eliminated": 0, "total_verified": 0}
        
        # 构建 fp -> stats 的映射
        for proto in self.get_all_prototypes():
            fp = f"proto_{proto.direction}_{proto.prototype_id}"
            stat = prototype_stats.get(fp)
            
            if stat is None:
                # 该原型在 WF 中从未被匹配
                proto.verified = False
                proto.wf_grade = "待观察"
                proto.wf_match_count = 0
                proto.wf_win_rate = 0.0
                counts["pending"] += 1
                counts["total_verified"] += 1  # 待观察也保留
                continue
            
            match_count = stat.get("match_count", 0)
            win_count = stat.get("win_count", 0)
            win_rate = win_count / match_count if match_count > 0 else 0.0
            
            avg_profit = stat.get("total_profit", 0.0) / match_count if match_count > 0 else 0.0
            proto.wf_match_count = match_count
            proto.wf_win_rate = win_rate
            
            if match_count < min_matches:
                proto.verified = False
                proto.wf_grade = "待观察"
                counts["pending"] += 1
                counts["total_verified"] += 1
            elif self._is_qualified(win_rate, avg_profit, min_win_rate):
                proto.verified = True
                proto.wf_grade = "合格"
                counts["qualified"] += 1
                counts["total_verified"] += 1
            else:
                proto.verified = False
                proto.wf_grade = "淘汰"
                counts["eliminated"] += 1
        
        print(f"[PrototypeLibrary] WF验证结果已回写(期望收益模式): "
              f"合格={counts['qualified']}, "
              f"待观察={counts['pending']}, "
              f"淘汰={counts['eliminated']}")
        
        return counts
    
    @staticmethod
    def _is_qualified(win_rate: float, avg_profit: float, min_win_rate: float) -> bool:
        """判断原型是否合格：优先用期望收益，备用纯胜率"""
        from config import WALK_FORWARD_CONFIG
        use_expected = WALK_FORWARD_CONFIG.get("EVAL_USE_EXPECTED_PROFIT", True)
        if use_expected:
            return avg_profit > 0  # 期望收益为正即合格
        else:
            return win_rate >= min_win_rate and avg_profit >= 0.0  # 旧逻辑

    def get_all_prototypes(self) -> List[Prototype]:
        """获取所有原型"""
        return self.long_prototypes + self.short_prototypes

    def set_evolved_params(self,
                          long_weights: Optional[np.ndarray] = None,
                          long_fusion: Optional[float] = None,
                          long_cosine: Optional[float] = None,
                          long_euclidean: Optional[float] = None,
                          long_dtw: Optional[float] = None,
                          short_weights: Optional[np.ndarray] = None,
                          short_fusion: Optional[float] = None,
                          short_cosine: Optional[float] = None,
                          short_euclidean: Optional[float] = None,
                          short_dtw: Optional[float] = None,
                          # 兼容旧版：单组时传 full_weights, fusion_threshold, cosine_min_threshold等
                          full_weights: Optional[np.ndarray] = None,
                          fusion_threshold: Optional[float] = None,
                          cosine_min_threshold: Optional[float] = None,
                          euclidean_min_threshold: Optional[float] = None,
                          dtw_min_threshold: Optional[float] = None) -> None:
        """
        将 WF 进化结果绑定到本原型库。支持多空两套或单组（单组时多空共用）。
        调用方式：
          - 多空两套: set_evolved_params(long_weights=w, long_fusion=f, long_cosine=c, long_euclidean=e, long_dtw=d, short_weights=w2, ...)
          - 单组: set_evolved_params(full_weights=w, fusion_threshold=f, cosine_min_threshold=c, euclidean_min_threshold=e, dtw_min_threshold=d)
        """
        if (full_weights is not None or fusion_threshold is not None or cosine_min_threshold is not None 
            or euclidean_min_threshold is not None or dtw_min_threshold is not None):
            # 旧版单组：写入共用字段，并复制到 long/short 以便统一读取
            w = np.asarray(full_weights, dtype=np.float64) if full_weights is not None else None
            self.evolved_full_weights = w
            self.evolved_fusion_threshold = fusion_threshold or 0.65
            self.evolved_cosine_min_threshold = cosine_min_threshold or 0.70
            self.evolved_euclidean_min_threshold = euclidean_min_threshold or 0.50
            self.evolved_dtw_min_threshold = dtw_min_threshold or 0.40
            self.evolved_full_weights_long = self.evolved_full_weights_short = w
            self.evolved_fusion_threshold_long = self.evolved_fusion_threshold_short = self.evolved_fusion_threshold
            self.evolved_cosine_min_threshold_long = self.evolved_cosine_min_threshold_short = self.evolved_cosine_min_threshold
            self.evolved_euclidean_min_threshold_long = self.evolved_euclidean_min_threshold_short = self.evolved_euclidean_min_threshold
            self.evolved_dtw_min_threshold_long = self.evolved_dtw_min_threshold_short = self.evolved_dtw_min_threshold
            return
        if long_weights is not None or long_fusion is not None or long_cosine is not None or long_euclidean is not None or long_dtw is not None:
            self.evolved_full_weights_long = np.asarray(long_weights, dtype=np.float64) if long_weights is not None else self.evolved_full_weights_long
            self.evolved_fusion_threshold_long = long_fusion if long_fusion is not None else (self.evolved_fusion_threshold_long or 0.65)
            self.evolved_cosine_min_threshold_long = long_cosine if long_cosine is not None else (self.evolved_cosine_min_threshold_long or 0.70)
            self.evolved_euclidean_min_threshold_long = long_euclidean if long_euclidean is not None else (self.evolved_euclidean_min_threshold_long or 0.50)
            self.evolved_dtw_min_threshold_long = long_dtw if long_dtw is not None else (self.evolved_dtw_min_threshold_long or 0.40)
        if short_weights is not None or short_fusion is not None or short_cosine is not None or short_euclidean is not None or short_dtw is not None:
            self.evolved_full_weights_short = np.asarray(short_weights, dtype=np.float64) if short_weights is not None else self.evolved_full_weights_short
            self.evolved_fusion_threshold_short = short_fusion if short_fusion is not None else (self.evolved_fusion_threshold_short or 0.65)
            self.evolved_cosine_min_threshold_short = short_cosine if short_cosine is not None else (self.evolved_cosine_min_threshold_short or 0.70)
            self.evolved_euclidean_min_threshold_short = short_euclidean if short_euclidean is not None else (self.evolved_euclidean_min_threshold_short or 0.50)
            self.evolved_dtw_min_threshold_short = short_dtw if short_dtw is not None else (self.evolved_dtw_min_threshold_short or 0.40)
        # 若只传了一侧，另一侧保持原样或回退到共用
        if self.evolved_full_weights_long is None and self.evolved_full_weights is not None:
            self.evolved_full_weights_long = self.evolved_full_weights
            self.evolved_fusion_threshold_long = self.evolved_fusion_threshold or 0.65
            self.evolved_cosine_min_threshold_long = self.evolved_cosine_min_threshold or 0.70
            self.evolved_euclidean_min_threshold_long = self.evolved_euclidean_min_threshold or 0.50
            self.evolved_dtw_min_threshold_long = self.evolved_dtw_min_threshold or 0.40
        if self.evolved_full_weights_short is None and self.evolved_full_weights is not None:
            self.evolved_full_weights_short = self.evolved_full_weights
            self.evolved_fusion_threshold_short = self.evolved_fusion_threshold or 0.65
            self.evolved_cosine_min_threshold_short = self.evolved_cosine_min_threshold or 0.70
            self.evolved_euclidean_min_threshold_short = self.evolved_euclidean_min_threshold or 0.50
            self.evolved_dtw_min_threshold_short = self.evolved_dtw_min_threshold or 0.40
        if self.evolved_full_weights_short is None and self.evolved_full_weights_long is not None:
            self.evolved_full_weights_short = self.evolved_full_weights_long
            self.evolved_fusion_threshold_short = self.evolved_fusion_threshold_long
            self.evolved_cosine_min_threshold_short = self.evolved_cosine_min_threshold_long
            self.evolved_euclidean_min_threshold_short = self.evolved_euclidean_min_threshold_long
            self.evolved_dtw_min_threshold_short = self.evolved_dtw_min_threshold_long

    def get_evolved_params(self) -> Tuple[Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[float], Optional[float]],
                                           Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[float], Optional[float]]]:
        """
        返回 (long_tuple, short_tuple)，每侧为 (weights, fusion_threshold, cosine_min_threshold, euclidean_min_threshold, dtw_min_threshold)。
        若仅有旧版单组，则 long 与 short 返回相同值。
        """
        long_t = (
            self.evolved_full_weights_long if self.evolved_full_weights_long is not None else self.evolved_full_weights,
            self.evolved_fusion_threshold_long if self.evolved_fusion_threshold_long is not None else self.evolved_fusion_threshold,
            self.evolved_cosine_min_threshold_long if self.evolved_cosine_min_threshold_long is not None else self.evolved_cosine_min_threshold,
            self.evolved_euclidean_min_threshold_long if self.evolved_euclidean_min_threshold_long is not None else self.evolved_euclidean_min_threshold,
            self.evolved_dtw_min_threshold_long if self.evolved_dtw_min_threshold_long is not None else self.evolved_dtw_min_threshold,
        )
        short_t = (
            self.evolved_full_weights_short if self.evolved_full_weights_short is not None else self.evolved_full_weights,
            self.evolved_fusion_threshold_short if self.evolved_fusion_threshold_short is not None else self.evolved_fusion_threshold,
            self.evolved_cosine_min_threshold_short if self.evolved_cosine_min_threshold_short is not None else self.evolved_cosine_min_threshold,
            self.evolved_euclidean_min_threshold_short if self.evolved_euclidean_min_threshold_short is not None else self.evolved_euclidean_min_threshold,
            self.evolved_dtw_min_threshold_short if self.evolved_dtw_min_threshold_short is not None else self.evolved_dtw_min_threshold,
        )
        return (long_t, short_t)

    def save(self, filepath: str = None, verbose: bool = True) -> str:
        """
        保存原型库到文件
        
        版本说明：
        - v1.0: 基础原型结构
        - v2.0: 添加置信度系统
        - v3.0: 指纹3D图匹配系统（当前版本）
        """
        if filepath is None:
            os.makedirs(DEFAULT_PROTOTYPE_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(DEFAULT_PROTOTYPE_DIR, f"prototypes_{timestamp}.pkl")
        
        # 保存前确保所有原型都已迁移到 v3.0
        migrated_count = self._ensure_all_migrated()
        
        data = {
            "version": PROTOTYPE_LIBRARY_VERSION,  # 使用全局版本常量
            "created_at": self.created_at or datetime.now().isoformat(),
            "source_template_count": self.source_template_count,
            "clustering_params": self.clustering_params,
            "source_symbol": self.source_symbol,
            "source_interval": self.source_interval,
            "long_prototypes": [p.to_dict() for p in self.long_prototypes],
            "short_prototypes": [p.to_dict() for p in self.short_prototypes],
            "evolved_full_weights": self.evolved_full_weights.tolist() if self.evolved_full_weights is not None else None,
            "evolved_fusion_threshold": self.evolved_fusion_threshold,
            "evolved_cosine_min_threshold": self.evolved_cosine_min_threshold,
            "evolved_euclidean_min_threshold": self.evolved_euclidean_min_threshold,
            "evolved_dtw_min_threshold": self.evolved_dtw_min_threshold,
            "evolved_full_weights_long": self.evolved_full_weights_long.tolist() if getattr(self, "evolved_full_weights_long", None) is not None else None,
            "evolved_fusion_threshold_long": getattr(self, "evolved_fusion_threshold_long", None),
            "evolved_cosine_min_threshold_long": getattr(self, "evolved_cosine_min_threshold_long", None),
            "evolved_euclidean_min_threshold_long": getattr(self, "evolved_euclidean_min_threshold_long", None),
            "evolved_dtw_min_threshold_long": getattr(self, "evolved_dtw_min_threshold_long", None),
            "evolved_full_weights_short": self.evolved_full_weights_short.tolist() if getattr(self, "evolved_full_weights_short", None) is not None else None,
            "evolved_fusion_threshold_short": getattr(self, "evolved_fusion_threshold_short", None),
            "evolved_cosine_min_threshold_short": getattr(self, "evolved_cosine_min_threshold_short", None),
            "evolved_euclidean_min_threshold_short": getattr(self, "evolved_euclidean_min_threshold_short", None),
            "evolved_dtw_min_threshold_short": getattr(self, "evolved_dtw_min_threshold_short", None),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if verbose:
            size_kb = os.path.getsize(filepath) / 1024
            print(f"[PrototypeLibrary] 已保存: {filepath} (版本 {PROTOTYPE_LIBRARY_VERSION})")
            print(f"    LONG原型: {len(self.long_prototypes)}, "
                  f"SHORT原型: {len(self.short_prototypes)}, "
                  f"文件大小: {size_kb:.1f} KB")
            if self.evolved_full_weights is not None:
                print(f"    已含进化权重 (fusion_th={self.evolved_fusion_threshold}, cosine_min={self.evolved_cosine_min_threshold})")
            if migrated_count > 0:
                print(f"    迁移到v3.0: {migrated_count} 个原型")
        
        return filepath
    
    def _ensure_all_migrated(self) -> int:
        """
        确保所有原型都已迁移到 v3.0 格式
        
        Returns:
            迁移的原型数量
        """
        migrated = 0
        for proto in self.get_all_prototypes():
            if proto.weighted_mean.size == 0 and proto.pre_entry_centroid.size > 0:
                proto._migrate_to_v3()
                migrated += 1
        return migrated
    
    @classmethod
    def load(cls, filepath: str, verbose: bool = True) -> 'PrototypeLibrary':
        """从文件加载原型库"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        version = data.get("version", "1.0")
        
        library = cls()
        library.created_at = data.get("created_at", "")
        library.source_template_count = data.get("source_template_count", 0)
        library.clustering_params = data.get("clustering_params", {})
        library.source_symbol = (data.get("source_symbol", "") or "").upper()
        library.source_interval = (data.get("source_interval", "") or "").strip()
        library.long_prototypes = [
            Prototype.from_dict(d) for d in data.get("long_prototypes", [])
        ]
        library.short_prototypes = [
            Prototype.from_dict(d) for d in data.get("short_prototypes", [])
        ]
        # 进化结果（绑定到聚合指纹图）
        if data.get("evolved_full_weights") is not None:
            library.evolved_full_weights = np.array(data["evolved_full_weights"], dtype=np.float64)
        if data.get("evolved_fusion_threshold") is not None:
            library.evolved_fusion_threshold = float(data["evolved_fusion_threshold"])
        if data.get("evolved_cosine_min_threshold") is not None:
            library.evolved_cosine_min_threshold = float(data["evolved_cosine_min_threshold"])
        if data.get("evolved_euclidean_min_threshold") is not None:
            library.evolved_euclidean_min_threshold = float(data["evolved_euclidean_min_threshold"])
        if data.get("evolved_dtw_min_threshold") is not None:
            library.evolved_dtw_min_threshold = float(data["evolved_dtw_min_threshold"])
        if data.get("evolved_full_weights_long") is not None:
            library.evolved_full_weights_long = np.array(data["evolved_full_weights_long"], dtype=np.float64)
        if data.get("evolved_fusion_threshold_long") is not None:
            library.evolved_fusion_threshold_long = float(data["evolved_fusion_threshold_long"])
        if data.get("evolved_cosine_min_threshold_long") is not None:
            library.evolved_cosine_min_threshold_long = float(data["evolved_cosine_min_threshold_long"])
        if data.get("evolved_euclidean_min_threshold_long") is not None:
            library.evolved_euclidean_min_threshold_long = float(data["evolved_euclidean_min_threshold_long"])
        if data.get("evolved_dtw_min_threshold_long") is not None:
            library.evolved_dtw_min_threshold_long = float(data["evolved_dtw_min_threshold_long"])
        if data.get("evolved_full_weights_short") is not None:
            library.evolved_full_weights_short = np.array(data["evolved_full_weights_short"], dtype=np.float64)
        if data.get("evolved_fusion_threshold_short") is not None:
            library.evolved_fusion_threshold_short = float(data["evolved_fusion_threshold_short"])
        if data.get("evolved_cosine_min_threshold_short") is not None:
            library.evolved_cosine_min_threshold_short = float(data["evolved_cosine_min_threshold_short"])
        if data.get("evolved_euclidean_min_threshold_short") is not None:
            library.evolved_euclidean_min_threshold_short = float(data["evolved_euclidean_min_threshold_short"])
        if data.get("evolved_dtw_min_threshold_short") is not None:
            library.evolved_dtw_min_threshold_short = float(data["evolved_dtw_min_threshold_short"])
        # 若仅有 long/short 而无旧版单组，用 long 填 legacy 以便兼容
        if library.evolved_full_weights is None and library.evolved_full_weights_long is not None:
            library.evolved_full_weights = library.evolved_full_weights_long
            library.evolved_fusion_threshold = library.evolved_fusion_threshold_long or 0.65
            library.evolved_cosine_min_threshold = library.evolved_cosine_min_threshold_long or 0.70
            library.evolved_euclidean_min_threshold = library.evolved_euclidean_min_threshold_long or 0.50
            library.evolved_dtw_min_threshold = library.evolved_dtw_min_threshold_long or 0.40
        
        # ═══════════════════════════════════════════════════════════════════
        # 版本兼容性处理
        # ═══════════════════════════════════════════════════════════════════
        
        migration_stats = {"confidence": 0, "v3_features": 0}
        
        # v1.0 → v2.0: 计算置信度
        if version < "2.0":
            for proto in library.get_all_prototypes():
                if proto.confidence == 0.0 and proto.member_count > 0:
                    proto.calculate_confidence()
                    migration_stats["confidence"] += 1
        
        # v1.0/v2.0 → v3.0: 迁移指纹3D图匹配系统字段
        # 注：Prototype.from_dict() 已经会调用 _migrate_to_v3()，这里统计迁移数量
        if version < "3.0":
            for proto in library.get_all_prototypes():
                # 检查是否需要迁移（from_dict 可能已处理，这里做二次确认）
                if proto.weighted_mean.size == 0 and proto.pre_entry_centroid.size > 0:
                    proto._migrate_to_v3()
                    migration_stats["v3_features"] += 1
                elif proto.sequence_features_version == PROTOTYPE_LIBRARY_VERSION:
                    # 已在 from_dict 中迁移
                    migration_stats["v3_features"] += 1
        
        if verbose:
            print(f"[PrototypeLibrary] 已加载: {filepath}")
            print(f"    文件版本: {version}, 当前版本: {PROTOTYPE_LIBRARY_VERSION}")
            print(f"    LONG原型: {len(library.long_prototypes)}, "
                  f"SHORT原型: {len(library.short_prototypes)}")
            if library.evolved_full_weights is not None:
                print(f"    已含进化权重 (fusion_th={library.evolved_fusion_threshold}, cosine_min={library.evolved_cosine_min_threshold})，匹配时将自动使用")
            # 打印迁移统计
            if migration_stats["confidence"] > 0 or migration_stats["v3_features"] > 0:
                migration_msg = []
                if migration_stats["confidence"] > 0:
                    migration_msg.append(f"置信度计算: {migration_stats['confidence']}个")
                if migration_stats["v3_features"] > 0:
                    migration_msg.append(f"v3.0特征迁移: {migration_stats['v3_features']}个")
                print(f"    兼容性升级: {', '.join(migration_msg)}")
            
            # 打印置信度统计
            stats = library.get_confidence_stats()
            if stats["total_count"] > 0:
                print(f"    置信度统计: 平均{stats['avg_confidence']:.1%}, "
                      f"高({stats['high_confidence_count']}), "
                      f"中({stats['mid_confidence_count']}), "
                      f"低({stats['low_confidence_count']})")
        
        return library
    
    @classmethod
    def load_latest(cls, prototype_dir: str = None, 
                    verbose: bool = True) -> Optional['PrototypeLibrary']:
        """加载最新的原型库"""
        if prototype_dir is None:
            prototype_dir = DEFAULT_PROTOTYPE_DIR
        
        if not os.path.exists(prototype_dir):
            if verbose:
                print("[PrototypeLibrary] 没有找到原型库目录")
            return None
        
        files = []
        for filename in os.listdir(prototype_dir):
            if filename.endswith('.pkl') and filename.startswith('prototypes_'):
                filepath = os.path.join(prototype_dir, filename)
                stat = os.stat(filepath)
                files.append((filepath, stat.st_mtime))
        
        if not files:
            if verbose:
                print("[PrototypeLibrary] 没有找到已保存的原型库")
            return None
        
        # 按修改时间排序，取最新的
        files.sort(key=lambda x: x[1], reverse=True)
        return cls.load(files[0][0], verbose=verbose)
    
    @staticmethod
    def check_file_version(filepath: str) -> Dict:
        """
        检查原型库文件版本（不完全加载）
        
        用于诊断和版本兼容性检查
        
        Args:
            filepath: 原型库文件路径
        
        Returns:
            {
                "version": str,             # 文件版本号
                "current_version": str,     # 当前系统版本号
                "needs_migration": bool,    # 是否需要迁移
                "migration_steps": list,    # 需要的迁移步骤
                "created_at": str,          # 创建时间
                "prototype_count": int,     # 原型数量
            }
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            file_version = data.get("version", "1.0")
            long_count = len(data.get("long_prototypes", []))
            short_count = len(data.get("short_prototypes", []))
            
            # 确定需要的迁移步骤
            migration_steps = []
            if file_version < "2.0":
                migration_steps.append("v1.0→v2.0: 计算置信度评分")
            if file_version < "3.0":
                migration_steps.append("v2.0→v3.0: 迁移指纹3D图特征 (weighted_mean, trend_vector, etc.)")
            
            return {
                "version": file_version,
                "current_version": PROTOTYPE_LIBRARY_VERSION,
                "needs_migration": file_version < PROTOTYPE_LIBRARY_VERSION,
                "migration_steps": migration_steps,
                "created_at": data.get("created_at", ""),
                "prototype_count": long_count + short_count,
                "long_count": long_count,
                "short_count": short_count,
                "source_symbol": data.get("source_symbol", ""),
                "source_interval": data.get("source_interval", ""),
            }
        except Exception as e:
            return {
                "version": "unknown",
                "current_version": PROTOTYPE_LIBRARY_VERSION,
                "needs_migration": True,
                "migration_steps": [],
                "error": str(e),
            }
    
    def get_version_info(self) -> Dict:
        """
        获取当前原型库的版本信息
        
        Returns:
            {
                "library_version": str,     # 原型库保存时的版本
                "current_version": str,     # 当前系统版本
                "v3_migrated_count": int,   # 已迁移到v3.0的原型数
                "v3_pending_count": int,    # 待迁移的原型数
            }
        """
        v3_migrated = 0
        v3_pending = 0
        
        for proto in self.get_all_prototypes():
            if proto.weighted_mean.size > 0:
                v3_migrated += 1
            elif proto.pre_entry_centroid.size > 0:
                v3_pending += 1
        
        return {
            "library_version": PROTOTYPE_LIBRARY_VERSION,
            "current_version": PROTOTYPE_LIBRARY_VERSION,
            "v3_migrated_count": v3_migrated,
            "v3_pending_count": v3_pending,
            "total_prototypes": len(self.get_all_prototypes()),
        }


class TemplateClusterer:
    """
    模板聚类引擎
    
    使用方式：
        clusterer = TemplateClusterer(n_clusters=30)
        library = clusterer.fit(trajectory_memory)
        library.save()
    """
    
    def __init__(self, 
                 n_clusters_long: int = 30,
                 n_clusters_short: int = 30,
                 min_cluster_size: int = 3,
                 random_state: int = 42):
        """
        Args:
            n_clusters_long: LONG 方向的聚类数
            n_clusters_short: SHORT 方向的聚类数
            min_cluster_size: 最小簇大小（小于此的簇会被丢弃）
            random_state: 随机种子
        """
        self.n_clusters_long = n_clusters_long
        self.n_clusters_short = n_clusters_short
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
    
    def fit(self, trajectory_memory, verbose: bool = True) -> PrototypeLibrary:
        """
        对模板库进行聚类，生成原型库
        
        聚类策略：按 (方向, 市场状态) 分组，每组独立聚类
        这确保了同一原型内的模板都来自相同的市场环境
        
        Args:
            trajectory_memory: TrajectoryMemory 实例
            verbose: 是否打印过程信息
        
        Returns:
            PrototypeLibrary 原型库
        """
        from sklearn.cluster import KMeans
        from core.market_regime import MarketRegime
        
        if verbose:
            print(f"[TemplateClusterer] 开始聚类（按方向+市场状态分组）...")
            print(f"    模板总数: {trajectory_memory.total_count}")
            print(f"    每组LONG聚类数: {self.n_clusters_long}, SHORT聚类数: {self.n_clusters_short}")
        
        library = PrototypeLibrary()
        library.created_at = datetime.now().isoformat()
        library.source_template_count = trajectory_memory.total_count
        library.clustering_params = {
            "n_clusters_long": self.n_clusters_long,
            "n_clusters_short": self.n_clusters_short,
            "min_cluster_size": self.min_cluster_size,
            "cluster_by_regime": True,  # 标记新版本聚类方式
        }
        library.source_symbol = getattr(trajectory_memory, "source_symbol", "").upper()
        library.source_interval = getattr(trajectory_memory, "source_interval", "").strip()
        
        # 6个市场状态
        all_regimes = MarketRegime.ALL_REGIMES + [MarketRegime.UNKNOWN]
        
        long_prototypes = []
        short_prototypes = []
        
        # 按 (direction, regime) 分组聚类
        for direction in ["LONG", "SHORT"]:
            all_templates = trajectory_memory.get_templates_by_direction(direction)
            base_n_clusters = self.n_clusters_long if direction == "LONG" else self.n_clusters_short
            
            if len(all_templates) == 0:
                if verbose:
                    print(f"    {direction}: 无模板，跳过")
                continue
            
            # 按市场状态分组
            # 【消除 regime="未知"】将"未知"状态的模板归入方向对应的默认 regime
            regime_groups = {}
            for t in all_templates:
                regime = t.regime if t.regime else ""
                # 消除"未知"：LONG 归入震荡偏多，SHORT 归入震荡偏空
                if not regime or regime == MarketRegime.UNKNOWN:
                    regime = MarketRegime.RANGE_BULL if direction == "LONG" else MarketRegime.RANGE_BEAR
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(t)
            
            if verbose:
                print(f"    {direction}: 共{len(all_templates)}个模板，分布在{len(regime_groups)}个市场状态")
            
            direction_prototypes = []
            
            for regime, templates in regime_groups.items():
                if len(templates) == 0:
                    continue
                
                # 根据该组的模板数量调整聚类数
                # 每个状态至少1个原型，最多base_n_clusters个
                n_clusters = min(base_n_clusters, max(1, len(templates)))
                
                # 1. 提取统计向量
                vectors, template_refs = self._extract_feature_vectors(templates)
                
                if len(vectors) == 0:
                    continue
                
                if len(vectors) < n_clusters:
                    n_clusters = max(1, len(vectors))
                
                # 2. K-Means 聚类
                if len(vectors) <= n_clusters:
                    labels = np.arange(len(vectors))
                    n_clusters = len(vectors)
                else:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        n_init=10,
                        max_iter=300,
                    )
                    labels = kmeans.fit_predict(vectors)
                
                # 3. 为每个簇生成原型（传入 regime）
                prototypes = self._build_prototypes_from_clusters(
                    direction, regime, vectors, labels, template_refs, n_clusters,
                    id_offset=len(direction_prototypes)
                )
                
                direction_prototypes.extend(prototypes)
                
                if verbose and len(prototypes) > 0:
                    print(f"        {direction}/{regime}: {len(templates)}模板 → {len(prototypes)}原型")
            
            if direction == "LONG":
                long_prototypes = direction_prototypes
            else:
                short_prototypes = direction_prototypes
        
        # 【原型去重】删除过于相似的原型，提高区分度
        if verbose:
            print(f"[TemplateClusterer] 原型去重中...")
        
        long_prototypes = self._deduplicate_prototypes(
            long_prototypes, similarity_threshold=0.98, verbose=verbose
        )
        short_prototypes = self._deduplicate_prototypes(
            short_prototypes, similarity_threshold=0.98, verbose=verbose
        )
        
        library.long_prototypes = long_prototypes
        library.short_prototypes = short_prototypes
        
        if verbose:
            print(f"[TemplateClusterer] 聚类完成: "
                  f"LONG={len(library.long_prototypes)}, "
                  f"SHORT={len(library.short_prototypes)}")
            self._print_prototype_stats(library)
        
        return library
    
    def _extract_feature_vectors(self, templates: list) -> Tuple[np.ndarray, list]:
        """
        从模板列表提取统计向量
        
        每个模板的完整交易压缩为：
          - pre_entry: mean(32) 
          - holding: mean(32) + std(32) = 64
          - pre_exit: mean(32)
          - meta: [hold_bars_norm, profit_pct_norm] = 2
          总计: 32 + 64 + 32 + 2 = 130 维
        
        Returns:
            (vectors, template_refs): 向量数组和对应的模板引用列表
        """
        vectors = []
        template_refs = []
        
        for template in templates:
            try:
                vec = self._template_to_vector(template)
                if vec is not None:
                    vectors.append(vec)
                    template_refs.append(template)
            except Exception as e:
                # 跳过无效模板
                continue
        
        if len(vectors) == 0:
            return np.array([]), []
        
        return np.array(vectors), template_refs
    
    def _template_to_vector(self, template) -> Optional[np.ndarray]:
        """
        将单个模板转换为统计向量
        
        统计摘要法：
          - pre_entry: 时间维度均值 (32,)
          - holding: 时间维度均值 + 标准差 (64,)
          - pre_exit: 时间维度均值 (32,)
          - meta: 持仓时长（归一化）+ 收益率（归一化）(2,)
        """
        # pre_entry 统计
        if template.pre_entry.size == 0:
            return None
        pre_entry_mean = template.pre_entry.mean(axis=0)  # (32,)
        
        # holding 统计
        if template.holding.size == 0:
            holding_mean = np.zeros(32)
            holding_std = np.zeros(32)
        else:
            holding_mean = template.holding.mean(axis=0)  # (32,)
            holding_std = template.holding.std(axis=0)    # (32,)
        
        # pre_exit 统计
        if template.pre_exit.size == 0:
            pre_exit_mean = np.zeros(32)
        else:
            pre_exit_mean = template.pre_exit.mean(axis=0)  # (32,)
        
        # 元信息（归一化）
        hold_bars = template.holding.shape[0] if template.holding.size > 0 else 0
        hold_bars_norm = hold_bars / 200.0  # 假设最大200根K线
        profit_norm = template.profit_pct / 10.0  # 假设最大10%收益
        meta = np.array([hold_bars_norm, profit_norm])
        
        # 拼接
        vec = np.concatenate([
            pre_entry_mean,      # 32
            holding_mean,        # 32
            holding_std,         # 32
            pre_exit_mean,       # 32
            meta,                # 2
        ])  # 总计 130 维
        
        return vec
    
    def _build_prototypes_from_clusters(self, direction: str, 
                                         regime: str,
                                         vectors: np.ndarray,
                                         labels: np.ndarray,
                                         template_refs: list,
                                         n_clusters: int,
                                         id_offset: int = 0) -> List[Prototype]:
        """
        从聚类结果构建原型列表
        
        Args:
            direction: 方向 (LONG/SHORT)
            regime: 市场状态 (6态之一)
            vectors: 特征向量
            labels: 聚类标签
            template_refs: 模板引用
            n_clusters: 簇数量
            id_offset: ID偏移量（用于保证全局唯一）
        """
        prototypes = []
        
        for cluster_id in range(n_clusters):
            # 找到属于该簇的所有模板
            mask = (labels == cluster_id)
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) < self.min_cluster_size:
                # 簇太小，丢弃
                continue
            
            # 获取该簇的向量和模板
            cluster_vectors = vectors[mask]
            cluster_templates = [template_refs[i] for i in cluster_indices]
            
            # 计算聚类中心
            centroid = cluster_vectors.mean(axis=0)
            
            # 拆分中心向量为各段
            pre_entry_centroid = centroid[:32]
            holding_centroid = centroid[32:96]  # mean(32) + std(32)
            pre_exit_centroid = centroid[96:128]
            
            # 【DTW代表序列】找到最接近质心的模板，用其 pre_entry 作为代表序列
            representative_pre_entry = np.array([])
            if len(cluster_vectors) > 0:
                # 计算每个模板到质心的距离
                distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
                closest_idx = np.argmin(distances)
                closest_template = cluster_templates[closest_idx]
                if closest_template.pre_entry.size > 0:
                    representative_pre_entry = closest_template.pre_entry.copy()
            
            # 统计成员信息
            member_count = len(cluster_templates)
            profits = [t.profit_pct for t in cluster_templates]
            hold_bars = [t.holding.shape[0] if t.holding.size > 0 else 0 
                        for t in cluster_templates]
            
            total_profit = sum(profits)
            avg_profit = np.mean(profits) if profits else 0.0
            win_count = sum(1 for p in profits if p > 0)
            win_rate = win_count / member_count if member_count > 0 else 0.0
            avg_hold = np.mean(hold_bars) if hold_bars else 0.0
            
            # 【概率扇形图】收集每个成员的 (收益率, 持仓K线数)
            member_trade_stats = [
                (float(t.profit_pct), int(t.holding.shape[0]) if t.holding.size > 0 else 0)
                for t in cluster_templates
            ]
            
            # 收集成员指纹
            member_fps = [t.fingerprint() for t in cluster_templates]
            
            # 【置信度系统】收集成员的 regime 信息（用于计算一致性）
            member_regimes = [t.regime if hasattr(t, 'regime') and t.regime else regime 
                             for t in cluster_templates]
            
            prototype = Prototype(
                prototype_id=id_offset + len(prototypes),
                direction=direction,
                regime=regime,  # 新增：市场状态
                centroid=centroid,
                pre_entry_centroid=pre_entry_centroid,
                holding_centroid=holding_centroid,
                pre_exit_centroid=pre_exit_centroid,
                member_count=member_count,
                win_count=win_count,
                total_profit_pct=total_profit,
                avg_profit_pct=avg_profit,
                win_rate=win_rate,
                avg_hold_bars=avg_hold,
                member_fingerprints=member_fps,
                representative_pre_entry=representative_pre_entry,  # DTW 代表序列
                member_trade_stats=member_trade_stats,  # 概率扇形图数据
            )
            
            # 【置信度系统】计算并设置置信度
            prototype.calculate_confidence(
                member_profits=profits,
                member_regimes=member_regimes
            )
            
            prototypes.append(prototype)
        
        return prototypes
    
    def _print_prototype_stats(self, library: PrototypeLibrary):
        """打印原型统计信息"""
        for direction, protos in [("LONG", library.long_prototypes), 
                                   ("SHORT", library.short_prototypes)]:
            if not protos:
                continue
            
            total_members = sum(p.member_count for p in protos)
            avg_win_rate = np.mean([p.win_rate for p in protos])
            avg_profit = np.mean([p.avg_profit_pct for p in protos])
            
            # 置信度统计
            confidences = [p.confidence for p in protos]
            avg_confidence = np.mean(confidences)
            high_conf_count = sum(1 for c in confidences if c >= CONFIDENCE_CONFIG.get("HIGH_CONFIDENCE", 0.70))
            low_conf_count = sum(1 for c in confidences if c < CONFIDENCE_CONFIG.get("MIN_CONFIDENCE", 0.30))
            
            # 统计各市场状态的原型数量
            regime_counts = {}
            for p in protos:
                r = p.regime or "未知"
                regime_counts[r] = regime_counts.get(r, 0) + 1
            regime_str = ", ".join([f"{r}:{c}" for r, c in sorted(regime_counts.items())])
            
            print(f"    [{direction}] {len(protos)}个原型, "
                  f"覆盖{total_members}个模板, "
                  f"平均胜率{avg_win_rate:.1%}, "
                  f"平均收益{avg_profit:.2f}%")
            print(f"        市场状态分布: {regime_str}")
            print(f"        置信度: 平均{avg_confidence:.1%}, "
                  f"高({high_conf_count}), 低({low_conf_count})")

    def _deduplicate_prototypes(self, prototypes: List[Prototype], 
                                 similarity_threshold: float = 0.98,
                                 verbose: bool = True) -> List[Prototype]:
        """
        去重：删除过于相似的原型，保留更有统计意义的那个
        
        算法：
        1. 计算所有原型对之间的余弦相似度
        2. 如果两个原型相似度 > threshold，合并为一个
        3. 合并规则：保留成员数更多的原型，合并统计信息
        
        Args:
            prototypes: 原型列表
            similarity_threshold: 相似度阈值，超过则认为太像需要合并
            verbose: 是否打印去重信息
        
        Returns:
            去重后的原型列表
        """
        if len(prototypes) <= 1:
            return prototypes
        
        # 标记哪些原型被合并掉了
        merged_into = {}  # {被合并的idx: 合并目标idx}
        
        # 计算相似度矩阵
        n = len(prototypes)
        for i in range(n):
            if i in merged_into:
                continue
            
            for j in range(i + 1, n):
                if j in merged_into:
                    continue
                
                # 计算余弦相似度（使用 pre_entry_centroid）
                p1, p2 = prototypes[i], prototypes[j]
                
                # 只在同方向、同 regime 内去重
                if p1.direction != p2.direction or p1.regime != p2.regime:
                    continue
                
                if p1.pre_entry_centroid.size == 0 or p2.pre_entry_centroid.size == 0:
                    continue
                
                sim = self._cosine_similarity(p1.pre_entry_centroid, p2.pre_entry_centroid)
                
                if sim >= similarity_threshold:
                    # 选择保留：优先级 置信度 > 成员数 > 平均收益
                    # 1. 首先比较置信度（高置信度原型更可靠）
                    if abs(p1.confidence - p2.confidence) > 0.1:  # 置信度差异显著
                        if p1.confidence > p2.confidence:
                            keep_idx, drop_idx = i, j
                        else:
                            keep_idx, drop_idx = j, i
                    # 2. 置信度接近时，比较成员数
                    elif p1.member_count > p2.member_count:
                        keep_idx, drop_idx = i, j
                    elif p2.member_count > p1.member_count:
                        keep_idx, drop_idx = j, i
                    else:
                        # 3. 成员数相同，比较平均收益
                        if p1.avg_profit_pct >= p2.avg_profit_pct:
                            keep_idx, drop_idx = i, j
                        else:
                            keep_idx, drop_idx = j, i
                    
                    merged_into[drop_idx] = keep_idx
                    
                    # 合并统计信息到保留的原型
                    keep_proto = prototypes[keep_idx]
                    drop_proto = prototypes[drop_idx]
                    
                    # 合并成员数和统计
                    keep_proto.member_count += drop_proto.member_count
                    keep_proto.win_count += drop_proto.win_count
                    keep_proto.total_profit_pct += drop_proto.total_profit_pct
                    keep_proto.member_fingerprints.extend(drop_proto.member_fingerprints)
                    
                    # 重新计算平均值
                    if keep_proto.member_count > 0:
                        keep_proto.avg_profit_pct = keep_proto.total_profit_pct / keep_proto.member_count
                        keep_proto.win_rate = keep_proto.win_count / keep_proto.member_count
        
        # 过滤掉被合并的原型
        result = [p for i, p in enumerate(prototypes) if i not in merged_into]
        
        if verbose and len(merged_into) > 0:
            print(f"        [去重] 合并了 {len(merged_into)} 个相似原型 "
                  f"(阈值={similarity_threshold:.0%}), 剩余 {len(result)} 个")
        
        return result
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        if v1.size == 0 or v2.size == 0:
            return 0.0
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))


class PrototypeMatcher:
    """
    原型匹配器 - 用于实时匹配
    
    使用方式：
        matcher = PrototypeMatcher(prototype_library)
        result = matcher.match_entry(current_window, direction="LONG")
    
    多维相似度匹配系统：
        - 使用 MultiSimilarityCalculator 融合三种相似度度量
        - 余弦相似度 (30%): 捕捉特征变化方向是否一致
        - 欧氏距离 (40%): 捕捉特征数值是否接近
        - DTW (30%): 捕捉时间序列形态是否匹配
        - 置信度系统: 综合历史胜率、样本稳定性、样本量进行加权
    """
    
    def __init__(self, library: PrototypeLibrary, 
                 cosine_threshold: float = 0.5,
                 min_prototypes_agree: int = 1,
                 dtw_top_k: int = 5,
                 dtw_threshold: float = 0.5,
                 dtw_weight: float = 0.3,
                 enable_multi_similarity: bool = True,
                 include_dtw_in_match: bool = True):
        """
        Args:
            library: 原型库
            cosine_threshold: 余弦相似度阈值
            min_prototypes_agree: 最少需要多少个原型同意才开仓
            dtw_top_k: DTW二次验证的候选数量（取余弦Top-K）
            dtw_threshold: DTW距离阈值（0~1，越小越严格）
            dtw_weight: DTW分数权重（综合分 = (1-w)*cosine + w*dtw_sim）
            enable_multi_similarity: 是否启用多维相似度匹配系统
            include_dtw_in_match: 是否在匹配时包含DTW计算（可关闭以加速）
        """
        self.library = library
        self.cosine_threshold = cosine_threshold
        self.min_prototypes_agree = min_prototypes_agree
        
        # 【DTW二次过滤参数】
        self.dtw_top_k = dtw_top_k
        self.dtw_threshold = dtw_threshold
        self.dtw_weight = dtw_weight
        self.dtw_radius = 10  # Sakoe-Chiba band radius
        
        # 【多维相似度匹配系统】
        self.enable_multi_similarity = enable_multi_similarity
        self.include_dtw_in_match = include_dtw_in_match
        self._similarity_calculator = MultiSimilarityCalculator()
        
        # 从配置获取融合分数阈值
        self.fusion_threshold = SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.65)
        self.high_confidence_threshold = SIMILARITY_CONFIG.get("HIGH_CONFIDENCE_THRESHOLD", 0.80)
        
        # 【单维度阈值】多空分开进化时可各自设置
        self.euclidean_min_threshold_long = 0.50
        self.euclidean_min_threshold_short = 0.50
        self.dtw_min_threshold_long = 0.40
        self.dtw_min_threshold_short = 0.40
        
        # 【性能优化】预先堆叠原型中心向量，实现向量化匹配
        self._long_centroids = None
        self._short_centroids = None
        self._long_protos = []
        self._short_protos = []
        # 多空分开进化：做多/做空各一套权重
        self._weights_long = None
        self._weights_short = None

        if library:
            self._prepare_centroids()

    def set_feature_weights(self, weights: np.ndarray, short_weights: Optional[np.ndarray] = None):
        """
        注入进化后的 32 维特征权重。多空分开时传 short_weights；否则做多做空共用 weights。

        Args:
            weights: 32 维特征权重（做多方向，或共用）
            short_weights: 做空方向权重，None 则与 weights 共用
        """
        self._weights_long = np.asarray(weights, dtype=np.float64).ravel() if weights is not None else None
        self._weights_short = (np.asarray(short_weights, dtype=np.float64).ravel() if short_weights is not None else self._weights_long)
        if self._weights_long is not None:
            self._similarity_calculator.set_dynamic_weights(self._weights_long)
            self._prepare_weighted_centroids(self._weights_long, self._weights_short)
    
    def set_single_dimension_thresholds(self, 
                                       long_euclidean: Optional[float] = None, 
                                       long_dtw: Optional[float] = None,
                                       short_euclidean: Optional[float] = None,
                                       short_dtw: Optional[float] = None):
        """
        设置欧氏和DTW的单维度最低阈值。多空分开时可各自设置。
        
        Args:
            long_euclidean: 做多欧氏距离最低阈值
            long_dtw: 做多DTW形态最低阈值
            short_euclidean: 做空欧氏距离最低阈值
            short_dtw: 做空DTW形态最低阈值
        """
        if long_euclidean is not None:
            self.euclidean_min_threshold_long = float(long_euclidean)
        if long_dtw is not None:
            self.dtw_min_threshold_long = float(long_dtw)
        if short_euclidean is not None:
            self.euclidean_min_threshold_short = float(short_euclidean)
        if short_dtw is not None:
            self.dtw_min_threshold_short = float(short_dtw)

    def _prepare_weighted_centroids(self, long_weights: Optional[np.ndarray] = None,
                                    short_weights: Optional[np.ndarray] = None):
        """
        使用进化权重重新构建预堆叠中心向量矩阵；多空可各用一套权重。
        """
        long_w = long_weights if long_weights is not None else self._weights_long
        short_w = short_weights if short_weights is not None else self._weights_short
        if short_w is None:
            short_w = long_w

        def _norm(w):
            w = np.asarray(w, dtype=np.float64).ravel()
            w = np.clip(w, 0.01, None)
            return w / w.mean()

        def _reweight_and_stack(proto_list, w):
            if not proto_list or w is None:
                return None
            w = _norm(w)
            valid = [p for p in proto_list if p.pre_entry_centroid.size == 32]
            if not valid:
                return None
            mat = np.stack([p.pre_entry_centroid for p in valid])
            mat = mat * w[np.newaxis, :]
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            mat /= norms
            return mat

        if self._long_protos and long_w is not None:
            weighted = _reweight_and_stack(self._long_protos, long_w)
            if weighted is not None:
                self._long_centroids = weighted
        if self._short_protos and short_w is not None:
            weighted = _reweight_and_stack(self._short_protos, short_w)
            if weighted is not None:
                self._short_centroids = weighted
            
    def _prepare_centroids(self):
        """预先将原型中心向量堆叠成矩阵"""
        # LONG
        long_list = self.library.get_prototypes_by_direction("LONG")
        if long_list:
            # 过滤掉无效原型
            long_list = [p for p in long_list if p.pre_entry_centroid.size == 32]
            if long_list:
                self._long_centroids = np.stack([p.pre_entry_centroid for p in long_list])
                # 预归一化
                norms = np.linalg.norm(self._long_centroids, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._long_centroids /= norms
                self._long_protos = long_list
        
        # SHORT
        short_list = self.library.get_prototypes_by_direction("SHORT")
        if short_list:
            short_list = [p for p in short_list if p.pre_entry_centroid.size == 32]
            if short_list:
                self._short_centroids = np.stack([p.pre_entry_centroid for p in short_list])
                norms = np.linalg.norm(self._short_centroids, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._short_centroids /= norms
                self._short_protos = short_list

    # ══════════════════════════════════════════════════════════════════════════
    # DTW 二次过滤方法
    # ══════════════════════════════════════════════════════════════════════════
    
    def _compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        计算两个多变量序列的DTW距离（归一化到0~1）
        
        Args:
            seq1, seq2: (time, features) 形状的序列
        
        Returns:
            归一化的DTW距离，越小越相似
        """
        if seq1.size == 0 or seq2.size == 0:
            return 1.0
        
        # 确保是2D
        if seq1.ndim == 1:
            seq1 = seq1.reshape(-1, 1)
        if seq2.ndim == 1:
            seq2 = seq2.reshape(-1, 1)
        
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return 1.0
        
        # 初始化代价矩阵
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Sakoe-Chiba 带约束
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
        
        # 归一化：按最大可能距离估算
        max_len = max(n, m)
        n_features = seq1.shape[1] if seq1.ndim > 1 else 1
        max_dist = max_len * np.sqrt(n_features) * 2
        
        return min(1.0, distance / max_dist) if max_dist > 0 else 1.0
    
    def _dtw_filter_candidates(self, current_window: np.ndarray,
                                candidates: List[Tuple[Prototype, float]]) -> List[Tuple[Prototype, float, float, float]]:
        """
        对余弦候选进行DTW二次过滤
        
        Args:
            current_window: (time, 32) 当前K线窗口
            candidates: [(prototype, cosine_sim), ...] 余弦排序后的候选
        
        Returns:
            [(prototype, cosine_sim, dtw_sim, combined_score), ...] 按综合分排序
        """
        if not candidates:
            return []
        
        # 只对 Top-K 进行DTW验证（节省计算）
        top_k_candidates = candidates[:self.dtw_top_k]
        
        results = []
        for proto, cosine_sim in top_k_candidates:
            # 检查原型是否有代表序列
            if proto.representative_pre_entry.size == 0:
                # 没有代表序列，只用余弦
                dtw_dist = 0.5  # 中性值
                dtw_sim = 0.5
            else:
                # 计算DTW距离
                dtw_dist = self._compute_dtw_distance(
                    current_window, proto.representative_pre_entry
                )
                dtw_sim = max(0.0, 1.0 - dtw_dist)
            
            # 综合分数 = (1 - dtw_weight) * cosine + dtw_weight * dtw_sim
            combined = (1 - self.dtw_weight) * cosine_sim + self.dtw_weight * dtw_sim
            
            results.append((proto, cosine_sim, dtw_sim, combined))
        
        # 按综合分排序（高到低）
        results.sort(key=lambda x: x[3], reverse=True)
        
        return results

    def match_entry(self, current_window: np.ndarray, 
                    direction: str = None,
                    regime: str = None) -> Dict:
        """
        入场匹配：当前轨迹 vs 原型（多维相似度融合系统）
        
        【指纹3D图匹配核心】使用多维相似度融合取代单一余弦相似度：
        
        融合公式：
            综合分数 = w1*cos_sim + w2*(1 - norm_euclidean) + w3*dtw_sim
            最终分数 = 综合分数 * 置信度系数
        
        权重配置（来自 SIMILARITY_CONFIG）：
            - 余弦相似度 (30%): 捕捉特征变化方向是否一致
            - 欧氏距离 (40%): 捕捉特征数值是否接近
            - DTW (30%): 捕捉时间序列形态是否匹配
        
        置信度系统（来自 CONFIDENCE_CONFIG）：
            置信度 = 0.4*历史胜率 + 0.3*样本稳定性 + 0.2*样本量 + 0.1*市场一致性
        
        匹配逻辑：
        1. 如果指定了 regime，只在该市场状态的原型中匹配
        2. 如果指定了 direction，只匹配该方向
        3. 返回最终分数最高的匹配结果
        
        Args:
            current_window: (window_size, 32) 当前K线窗口
            direction: 指定方向，None则双向匹配
            regime: 当前市场状态，None则匹配所有状态
        
        Returns:
            {
                "matched": bool,               # 是否匹配成功
                "direction": str,              # 匹配方向
                "best_prototype": Prototype,   # 最佳匹配原型
                "similarity": float,           # 最终分数（含置信度）
                "combined_score": float,       # 融合分数（不含置信度）
                "cosine_similarity": float,    # 余弦相似度
                "euclidean_similarity": float, # 欧氏相似度
                "dtw_similarity": float,       # DTW相似度
                "confidence": float,           # 原型置信度
                "vote_long": int,
                "vote_short": int,
                "vote_ratio": float,
                "top_matches": [(prototype, similarity), ...],
                "top_matches_detail": [...],   # 完整多维相似度信息
            }
        """
        if current_window.size == 0 or self.library is None:
            return self._empty_result()
        
        # Regime-Direction 严格约束：任何偏多/偏空状态只允许对应方向
        BULL_REGIMES = {"强多头", "弱多头", "震荡偏多", "Strong Bull", "Weak Bull", "Range Bull"}
        BEAR_REGIMES = {"强空头", "弱空头", "震荡偏空", "Strong Bear", "Weak Bear", "Range Bear"}
        
        regime_str = str(regime) if regime else ""
        allowed_direction = None
        if regime_str in BULL_REGIMES:
            allowed_direction = "LONG"
        elif regime_str in BEAR_REGIMES:
            allowed_direction = "SHORT"
        
        # 确定要匹配的方向
        directions_to_match = []
        if direction is not None:
            # 如果 regime 已明确方向，则禁止反向匹配
            if allowed_direction and direction != allowed_direction:
                return self._empty_result()
            directions_to_match = [direction]
        elif allowed_direction:
            directions_to_match = [allowed_direction]
        else:
            directions_to_match = ["LONG", "SHORT"]
        
        # 收集所有候选原型
        all_candidates = []
        for d in directions_to_match:
            if regime:
                protos = self.library.get_prototypes_by_direction_and_regime(d, regime)
            else:
                protos = self.library.get_prototypes_by_direction(d)
            for proto in protos:
                if proto.pre_entry_centroid.size > 0:
                    all_candidates.append((d, proto))
        
        if not all_candidates:
            return self._empty_result()
        
        # ══════════════════════════════════════════════════════════
        # 多维相似度计算
        # ══════════════════════════════════════════════════════════
        
        # 决定是否使用多维相似度系统
        use_multi_sim = self.enable_multi_similarity
        include_dtw = self.include_dtw_in_match and use_multi_sim
        
        # 存储所有匹配结果
        match_results = []  # [(direction, proto, cos, euc, dtw, combined, confidence, final), ...]
        
        for d, proto in all_candidates:
            # 多空分开进化：匹配该方向时使用该方向的权重
            if self._weights_long is not None:
                w = self._weights_long if d == "LONG" else self._weights_short
                if w is not None:
                    self._similarity_calculator.set_dynamic_weights(w)
            if use_multi_sim:
                # 使用 MultiSimilarityCalculator 计算多维相似度
                sim_result = self._similarity_calculator.compute_similarity(
                    current_window, proto, include_dtw=include_dtw
                )
                
                cos_sim = sim_result["cosine_similarity"]
                euc_sim = sim_result["euclidean_similarity"]
                dtw_sim = sim_result["dtw_similarity"]
                combined = sim_result["combined_score"]
                confidence = sim_result["confidence"]
                final_score = sim_result["final_score"]
            else:
                # 兼容模式：仅使用余弦相似度
                current_vec = self._window_to_vector(current_window)
                cos_sim = self._cosine_similarity(current_vec, proto.pre_entry_centroid)
                euc_sim = 0.0
                dtw_sim = 0.0
                combined = cos_sim
                confidence = 1.0
                final_score = cos_sim
            
            match_results.append((d, proto, cos_sim, euc_sim, dtw_sim, combined, confidence, final_score))
        
        # 按最终分数降序排列
        match_results.sort(key=lambda x: x[7], reverse=True)
        
        # ══════════════════════════════════════════════════════════
        # 投票统计（基于融合分数阈值）
        # ══════════════════════════════════════════════════════════
        
        # 使用融合分数阈值来统计投票
        match_threshold = self.fusion_threshold if use_multi_sim else self.cosine_threshold
        
        vote_long = sum(1 for r in match_results if r[0] == "LONG" and r[7] >= match_threshold)
        vote_short = sum(1 for r in match_results if r[0] == "SHORT" and r[7] >= match_threshold)
        total_votes = vote_long + vote_short
        
        # ══════════════════════════════════════════════════════════
        # 确定最终方向和最佳匹配
        # ══════════════════════════════════════════════════════════
        
        if direction is not None:
            # 指定方向时，筛选该方向的结果
            dir_results = [r for r in match_results if r[0] == direction]
            if not dir_results:
                return self._empty_result()
            best_result = dir_results[0]
            final_direction = direction
        else:
            # 未指定方向时
            if allowed_direction:
                # regime 已限定方向
                dir_results = [r for r in match_results if r[0] == allowed_direction]
                if not dir_results:
                    return self._empty_result()
                best_result = dir_results[0]
                final_direction = allowed_direction
            elif match_results:
                # 选择最终分数最高的
                best_result = match_results[0]
                final_direction = best_result[0]
            else:
                return self._empty_result()
        
        # 提取最佳匹配信息
        _, best_proto, best_cos, best_euc, best_dtw, best_combined, best_confidence, best_final = best_result
        
        # 判断是否匹配成功
        # 需同时满足：融合分数达标 + 单维度阈值要求（欧氏、DTW）
        if use_multi_sim:
            # 获取当前方向的单维度阈值
            euc_min = self.euclidean_min_threshold_long if final_direction == "LONG" else self.euclidean_min_threshold_short
            dtw_min = self.dtw_min_threshold_long if final_direction == "LONG" else self.dtw_min_threshold_short
            
            # 融合分数达标 + 欧氏距离达标 + DTW形态达标（如果启用）
            matched = (best_final >= match_threshold 
                       and best_euc >= euc_min 
                       and (best_dtw >= dtw_min if include_dtw else True))
        else:
            matched = (best_cos >= self.cosine_threshold)
        
        # 投票比例
        vote_ratio = vote_long / total_votes if total_votes > 0 else 0.5
        
        # ══════════════════════════════════════════════════════════
        # 构建返回结果
        # ══════════════════════════════════════════════════════════
        
        # Top matches（兼容旧格式）
        top_matches = [(r[1], r[7]) for r in match_results[:5]]
        
        # Top matches detail（新格式：包含完整多维相似度信息）
        # (proto, cos, euc, dtw, combined, confidence, final)
        top_matches_detail = [
            (r[1], r[2], r[3], r[4], r[5], r[6], r[7])
            for r in match_results[:10]
        ]
        
        return {
            "matched": matched,
            "direction": final_direction,
            "best_prototype": best_proto,
            # 主分数（最终分数，含置信度调整）
            "similarity": best_final,
            # 多维相似度分解
            "combined_score": best_combined,       # 融合分数（不含置信度）
            "cosine_similarity": best_cos,         # 余弦相似度
            "euclidean_similarity": best_euc,      # 欧氏相似度
            "dtw_similarity": best_dtw,            # DTW相似度
            "confidence": best_confidence,         # 原型置信度
            # 匹配质量指标
            "is_high_confidence": best_final >= self.high_confidence_threshold,
            "match_threshold": match_threshold,
            # 投票统计
            "vote_long": vote_long,
            "vote_short": vote_short,
            "vote_ratio": vote_ratio,
            # 兼容旧格式
            "top_matches": top_matches,
            # 新格式：完整多维相似度信息
            "top_matches_detail": top_matches_detail,
        }

    def _match_direction_vectorized(self, current_unit: np.ndarray, 
                                     direction: str) -> List[Tuple[Prototype, float]]:
        """【性能核心】矩阵化匹配指定方向的所有原型"""
        centroids = self._long_centroids if direction == "LONG" else self._short_centroids
        protos = self._long_protos if direction == "LONG" else self._short_protos
        
        if centroids is None:
            return []
        
        # 计算余弦相似度：矩阵 × 向量
        sims = centroids @ current_unit
        
        results = []
        for i, sim in enumerate(sims):
            results.append((protos[i], float(sim)))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def check_holding_health(self, holding_so_far: np.ndarray,
                             matched_prototype: Prototype) -> Tuple[float, str]:
        """
        检查持仓健康度
        
        Args:
            holding_so_far: (n, 32) 从入场到当前的持仓轨迹
            matched_prototype: 匹配的原型
        
        Returns:
            (health_score, status): 健康分数(0~1) 和 状态描述
        """
        if holding_so_far.size == 0:
            return 1.0, "正常"
        
        # 计算当前持仓的统计向量
        current_mean = holding_so_far.mean(axis=0)
        current_std = holding_so_far.std(axis=0)
        current_vec = np.concatenate([current_mean, current_std])
        
        # 与原型的 holding_centroid 比较
        proto_vec = matched_prototype.holding_centroid
        
        # 余弦相似度作为健康度
        similarity = self._cosine_similarity(current_vec, proto_vec)
        
        if similarity >= 0.7:
            status = "健康"
        elif similarity >= 0.5:
            status = "正常"
        elif similarity >= 0.3:
            status = "偏离"
        else:
            status = "危险"
        
        return similarity, status
    
    def rematch_by_holding(self, holding_so_far: np.ndarray, 
                           current_prototype: Prototype,
                           direction: str,
                           regime: str = None,
                           switch_threshold: float = 0.15) -> Tuple[Optional[Prototype], float, bool]:
        """
        根据持仓轨迹重新匹配更合适的原型
        
        当持仓过程中发现有其他原型的持仓段更匹配当前轨迹时，切换到该原型。
        这允许系统在市场动态变化时调整预期。
        
        Args:
            holding_so_far: (n, 32) 从入场到当前的持仓轨迹
            current_prototype: 当前匹配的原型
            direction: 交易方向 (LONG/SHORT)
            regime: 市场状态过滤，None则匹配所有状态
            switch_threshold: 切换阈值，新原型相似度需超出当前原型此值才切换
        
        Returns:
            (best_prototype, similarity, switched): 
              - best_prototype: 最匹配的原型（可能是当前原型或新原型）
              - similarity: 与该原型的相似度
              - switched: 是否发生了切换
        """
        if holding_so_far.size == 0 or self.library is None:
            return current_prototype, 1.0, False
        
        # 计算当前持仓的统计向量 (64D = mean + std)
        current_mean = holding_so_far.mean(axis=0)
        current_std = holding_so_far.std(axis=0)
        current_vec = np.concatenate([current_mean, current_std])
        
        # 当前原型的相似度
        current_sim = self._cosine_similarity(current_vec, current_prototype.holding_centroid)
        
        # 获取同方向的原型候选
        if regime:
            candidates = self.library.get_prototypes_by_direction_and_regime(direction, regime)
        else:
            candidates = self.library.get_prototypes_by_direction(direction)
        
        # 找到持仓段最匹配的原型
        best_proto = current_prototype
        best_sim = current_sim
        
        for proto in candidates:
            if proto.prototype_id == current_prototype.prototype_id:
                continue  # 跳过当前原型
            if proto.holding_centroid.size == 0:
                continue
            
            sim = self._cosine_similarity(current_vec, proto.holding_centroid)
            if sim > best_sim + switch_threshold:  # 必须显著优于当前
                best_sim = sim
                best_proto = proto
        
        switched = (best_proto.prototype_id != current_prototype.prototype_id)
        return best_proto, best_sim, switched
    
    def check_exit_pattern(self, recent_trajectory: np.ndarray,
                           current_prototype: Prototype,
                           direction: str,
                           entry_price: float,
                           current_price: float,
                           stop_loss: float,
                           take_profit: float,
                           current_regime: str = None) -> Dict:
        """
        检查离场模式：综合 pre_exit 匹配 + 止损保护 + 市场状态 + K线空间位置
        
        这是三阶段匹配系统的最后一环：当持仓轨迹开始像原型的出场段时，
        结合风控因素决定是否触发离场。
        
        Args:
            recent_trajectory: (n, 32) 最近的轨迹（通常取持仓末尾或 pre_exit_window 根）
            current_prototype: 当前匹配的原型
            direction: 交易方向
            entry_price: 入场价格
            current_price: 当前价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            current_regime: 当前市场状态
        
        Returns:
            {
                "should_exit": bool,           # 是否建议离场
                "exit_signal_strength": float, # 离场信号强度 (0~1)
                "exit_reason": str,            # 离场原因
                "pattern_similarity": float,   # 与出场模式的相似度
                "price_position": float,       # 价格位置 (0=入场, 1=止盈, -1=止损)
                "regime_warning": bool,        # 市场状态是否变化
                "details": dict,               # 详细信息
            }
        """
        result = {
            "should_exit": False,
            "exit_signal_strength": 0.0,
            "exit_reason": "",
            "pattern_similarity": 0.0,
            "price_position": 0.0,
            "regime_warning": False,
            "details": {},
        }
        
        if recent_trajectory.size == 0 or current_prototype is None:
            return result
        
        # 【关键防御】轨迹不足3根K线时，模式匹配结果不可靠，直接跳过
        # 避免用1-2行数据做 pre_exit 匹配导致随机离场信号
        MIN_TRAJECTORY_ROWS = 3
        if recent_trajectory.ndim >= 2 and recent_trajectory.shape[0] < MIN_TRAJECTORY_ROWS:
            result["details"]["skipped"] = f"轨迹仅{recent_trajectory.shape[0]}行，需≥{MIN_TRAJECTORY_ROWS}行"
            return result
        elif recent_trajectory.ndim < 2:
            return result
        
        # ══════════════════════════════════════════════════════════
        # 1. Pre-exit 模式匹配：当前轨迹是否像原型的出场段？
        # ══════════════════════════════════════════════════════════
        current_vec = recent_trajectory.mean(axis=0)  # 32D
        proto_exit_vec = current_prototype.pre_exit_centroid
        
        if proto_exit_vec.size > 0:
            pattern_sim = self._cosine_similarity(current_vec, proto_exit_vec)
        else:
            pattern_sim = 0.0
        
        result["pattern_similarity"] = pattern_sim
        result["details"]["pre_exit_similarity"] = pattern_sim
        
        # ══════════════════════════════════════════════════════════
        # 2. K线空间位置：当前价格在入场价、止盈、止损之间的位置
        # ══════════════════════════════════════════════════════════
        # 归一化价格位置：0=入场价，1=止盈，-1=止损
        if direction == "LONG":
            tp_distance = take_profit - entry_price
            sl_distance = entry_price - stop_loss
            price_move = current_price - entry_price
            
            if price_move >= 0 and tp_distance > 0:
                price_position = min(1.0, price_move / tp_distance)
            elif price_move < 0 and sl_distance > 0:
                price_position = max(-1.0, price_move / sl_distance)
            else:
                price_position = 0.0
        else:  # SHORT
            tp_distance = entry_price - take_profit
            sl_distance = stop_loss - entry_price
            price_move = entry_price - current_price
            
            if price_move >= 0 and tp_distance > 0:
                price_position = min(1.0, price_move / tp_distance)
            elif price_move < 0 and sl_distance > 0:
                price_position = max(-1.0, price_move / sl_distance)
            else:
                price_position = 0.0
        
        result["price_position"] = price_position
        result["details"]["price_pct_to_tp"] = price_position if price_position > 0 else 0
        result["details"]["price_pct_to_sl"] = -price_position if price_position < 0 else 0
        
        # ══════════════════════════════════════════════════════════
        # 3. 市场状态变化检测
        # ══════════════════════════════════════════════════════════
        proto_regime = current_prototype.regime
        regime_warning = False
        
        if current_regime and proto_regime:
            # 检测逆向市场状态变化
            from core.market_regime import MarketRegime
            bull_regimes = {MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL, MarketRegime.RANGE_BULL}
            bear_regimes = {MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.RANGE_BEAR}
            
            proto_is_bull = proto_regime in bull_regimes
            current_is_bear = current_regime in bear_regimes
            proto_is_bear = proto_regime in bear_regimes
            current_is_bull = current_regime in bull_regimes
            
            # LONG 仓位遇到熊市信号，或 SHORT 仓位遇到牛市信号
            if direction == "LONG" and proto_is_bull and current_is_bear:
                regime_warning = True
            elif direction == "SHORT" and proto_is_bear and current_is_bull:
                regime_warning = True
        
        result["regime_warning"] = regime_warning
        result["details"]["proto_regime"] = proto_regime
        result["details"]["current_regime"] = current_regime
        
        # ══════════════════════════════════════════════════════════
        # 4. 综合离场决策
        # ══════════════════════════════════════════════════════════
        # 计算离场信号强度（加权综合）
        exit_signal = 0.0
        exit_reasons = []
        
        # 4.1 出场模式匹配 (权重 40%)
        if pattern_sim >= 0.7:
            exit_signal += 0.4 * pattern_sim
            exit_reasons.append(f"出场模式匹配 {pattern_sim:.0%}")
        elif pattern_sim >= 0.5:
            exit_signal += 0.2 * pattern_sim
        
        # 4.2 价格接近止盈 (权重 30%)
        if price_position >= 0.8:
            exit_signal += 0.3
            exit_reasons.append(f"接近止盈 {price_position:.0%}")
        elif price_position >= 0.5:
            exit_signal += 0.15
        
        # 4.3 价格接近止损 (权重 20%) - 止损优先保护
        if price_position <= -0.7:
            exit_signal += 0.3  # 接近止损时提高权重
            exit_reasons.append(f"接近止损 {-price_position:.0%}")
        elif price_position <= -0.5:
            exit_signal += 0.15
        
        # 4.4 市场状态逆转 (权重 10%)
        if regime_warning:
            exit_signal += 0.2
            exit_reasons.append("市场状态逆转")
        
        result["exit_signal_strength"] = min(1.0, exit_signal)
        
        # 离场决策阈值
        EXIT_THRESHOLD = 0.6
        
        if exit_signal >= EXIT_THRESHOLD:
            result["should_exit"] = True
            result["exit_reason"] = " + ".join(exit_reasons) if exit_reasons else "综合信号触发"
        elif price_position <= -0.9:
            # 接近止损时强制建议离场
            result["should_exit"] = True
            result["exit_reason"] = "止损保护"
        
        return result
    
    def _match_direction(self, current_vec: np.ndarray, 
                         direction: str,
                         regime: str = None) -> List[Tuple[Prototype, float]]:
        """
        匹配指定方向的原型
        
        Args:
            current_vec: 当前特征向量
            direction: 方向 (LONG/SHORT)
            regime: 市场状态过滤，None则匹配所有状态
        
        Returns:
            [(原型, 相似度), ...] 按相似度降序
        """
        if regime:
            # 只匹配指定市场状态的原型
            prototypes = self.library.get_prototypes_by_direction_and_regime(direction, regime)
        else:
            # 匹配所有原型
            prototypes = self.library.get_prototypes_by_direction(direction)
        
        results = []
        for proto in prototypes:
            if proto.pre_entry_centroid.size == 0:
                continue
            sim = self._cosine_similarity(current_vec, proto.pre_entry_centroid)
            results.append((proto, sim))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _window_to_vector(self, window: np.ndarray) -> np.ndarray:
        """
        将K线窗口转换为统计向量（只取加权均值，用于与 pre_entry_centroid 比较）
        
        【已升级】现在返回加权均值，兼容旧接口
        """
        if window.size == 0:
            return np.zeros(32)
        
        # 计算均值
        mean_vec = window.mean(axis=0)  # (32,)
        
        # 应用特征层权重
        from core.feature_vector import apply_weights
        return apply_weights(mean_vec)
    
    def _window_to_sequence_features(self, window: np.ndarray) -> Dict:
        """
        【指纹3D图匹配核心】将K线窗口转换为保留时间序列结构的特征集
        
        改造前: _window_to_vector() 只返回 window.mean(axis=0)，丢失时间信息
        改造后: 保留完整结构，支持多维相似度计算
        
        Args:
            window: (time, 32) K线窗口矩阵，如 (60, 32)
        
        Returns:
            {
                'raw_sequence': np.ndarray,     # 原始序列 (time, 32)，用于DTW形态匹配
                'weighted_mean': np.ndarray,    # 加权均值 (32,)，用于快速余弦相似度
                'trend_vector': np.ndarray,     # 趋势向量 (32,)，每个特征的变化趋势斜率
                'time_segments': dict,          # 时间分段特征（早中晚期）
                'volatility': np.ndarray,       # 波动率向量 (32,)，每个特征的标准差
            }
        """
        from core.feature_vector import apply_weights, extract_trends
        
        # 处理空窗口
        if window.size == 0 or window.ndim != 2:
            return {
                'raw_sequence': np.zeros((0, 32)),
                'weighted_mean': np.zeros(32),
                'trend_vector': np.zeros(32),
                'time_segments': {'early': np.zeros(32), 'mid': np.zeros(32), 'late': np.zeros(32)},
                'volatility': np.zeros(32),
            }
        
        n_time, n_features = window.shape
        if n_features != 32:
            # 非标准特征维度，返回空结果
            return {
                'raw_sequence': window,
                'weighted_mean': window.mean(axis=0) if n_time > 0 else np.zeros(n_features),
                'trend_vector': np.zeros(n_features),
                'time_segments': {'early': np.zeros(n_features), 'mid': np.zeros(n_features), 'late': np.zeros(n_features)},
                'volatility': window.std(axis=0) if n_time > 0 else np.zeros(n_features),
            }
        
        # 1. 原始序列（用于DTW）
        raw_sequence = window.copy()
        
        # 2. 加权均值（用于快速余弦相似度）
        mean_vec = window.mean(axis=0)  # (32,)
        weighted_mean = apply_weights(mean_vec)
        
        # 3. 趋势向量（每个特征的时间趋势斜率）
        trend_vector = extract_trends(window)
        
        # 4. 时间分段特征（捕获早中晚期变化）
        # 将窗口分为3段：早期(前1/3)、中期(中1/3)、晚期(后1/3)
        seg_size = max(1, n_time // 3)
        early_end = seg_size
        mid_end = 2 * seg_size
        
        early_mean = window[:early_end].mean(axis=0) if early_end > 0 else np.zeros(32)
        mid_mean = window[early_end:mid_end].mean(axis=0) if mid_end > early_end else np.zeros(32)
        late_mean = window[mid_end:].mean(axis=0) if n_time > mid_end else np.zeros(32)
        
        # 应用权重到分段特征
        time_segments = {
            'early': apply_weights(early_mean),
            'mid': apply_weights(mid_mean),
            'late': apply_weights(late_mean),
        }
        
        # 5. 波动率向量（每个特征的时间标准差）
        volatility = window.std(axis=0)  # (32,)
        
        return {
            'raw_sequence': raw_sequence,
            'weighted_mean': weighted_mean,
            'trend_vector': trend_vector,
            'time_segments': time_segments,
            'volatility': volatility,
        }
    
    def compute_multi_similarity(self, 
                                  current_features: Dict,
                                  proto_features: Dict,
                                  proto_raw_sequence: np.ndarray = None) -> Dict:
        """
        【指纹3D图匹配核心】计算多维相似度融合分数
        
        融合三种相似度度量:
        - 方向相似度（余弦）: 捕捉特征变化方向是否一致
        - 距离相似度（欧氏）: 捕捉特征数值是否接近
        - 形态相似度（DTW）: 捕捉时间序列形态是否匹配
        
        Args:
            current_features: 当前窗口的序列特征 (来自 _window_to_sequence_features)
            proto_features: 原型的序列特征
            proto_raw_sequence: 原型的原始序列 (用于DTW，可选)
        
        Returns:
            {
                'combined_score': float,     # 综合相似度分数 (0~1)
                'cosine_sim': float,         # 方向相似度
                'euclidean_sim': float,      # 距离相似度
                'dtw_sim': float,            # 形态相似度
                'trend_sim': float,          # 趋势相似度（额外指标）
            }
        """
        # 从配置获取权重
        w_cos = SIMILARITY_CONFIG.get("COSINE_WEIGHT", 0.30)
        w_euc = SIMILARITY_CONFIG.get("EUCLIDEAN_WEIGHT", 0.40)
        w_dtw = SIMILARITY_CONFIG.get("DTW_WEIGHT", 0.30)
        max_euc_dist = SIMILARITY_CONFIG.get("EUCLIDEAN_MAX_DIST", 10.0)
        
        # 获取加权均值
        cur_mean = current_features.get('weighted_mean', np.zeros(32))
        proto_mean = proto_features.get('weighted_mean', np.zeros(32))
        
        # 1. 余弦相似度（方向）
        cosine_sim = self._cosine_similarity(cur_mean, proto_mean)
        
        # 2. 欧氏距离相似度（归一化）
        if cur_mean.size > 0 and proto_mean.size > 0:
            euc_dist = np.linalg.norm(cur_mean - proto_mean)
            euclidean_sim = max(0.0, 1.0 - euc_dist / max_euc_dist)
        else:
            euclidean_sim = 0.0
        
        # 3. DTW 形态相似度
        dtw_sim = 0.5  # 默认中性值
        cur_seq = current_features.get('raw_sequence', np.array([]))
        proto_seq = proto_raw_sequence if proto_raw_sequence is not None else proto_features.get('raw_sequence', np.array([]))
        
        if cur_seq.size > 0 and proto_seq.size > 0:
            dtw_dist = self._compute_dtw_distance(cur_seq, proto_seq)
            dtw_sim = max(0.0, 1.0 - dtw_dist)
        
        # 4. 趋势相似度（额外指标，用于分析）
        cur_trend = current_features.get('trend_vector', np.zeros(32))
        proto_trend = proto_features.get('trend_vector', np.zeros(32))
        trend_sim = self._cosine_similarity(cur_trend, proto_trend)
        
        # 5. 融合计算综合分数
        combined_score = (
            w_cos * cosine_sim +
            w_euc * euclidean_sim +
            w_dtw * dtw_sim
        )
        
        return {
            'combined_score': float(combined_score),
            'cosine_sim': float(cosine_sim),
            'euclidean_sim': float(euclidean_sim),
            'dtw_sim': float(dtw_sim),
            'trend_sim': float(trend_sim),
        }

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        if v1.size == 0 or v2.size == 0:
            return 0.0
        
        # 对齐长度
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _empty_result(self) -> Dict:
        """返回空匹配结果（兼容多维相似度系统）"""
        return {
            "matched": False,
            "direction": None,
            "best_prototype": None,
            # 主分数
            "similarity": 0.0,
            # 多维相似度分解
            "combined_score": 0.0,
            "cosine_similarity": 0.0,
            "euclidean_similarity": 0.0,
            "dtw_similarity": 0.0,
            "confidence": 0.0,
            # 匹配质量指标
            "is_high_confidence": False,
            "match_threshold": 0.0,
            # 投票统计
            "vote_long": 0,
            "vote_short": 0,
            "vote_ratio": 0.5,
            # 兼容旧格式
            "top_matches": [],
            # 新格式
            "top_matches_detail": [],
        }


# ══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════════

def cluster_templates(trajectory_memory, 
                      n_clusters_long: int = 30,
                      n_clusters_short: int = 30,
                      save: bool = True,
                      verbose: bool = True) -> PrototypeLibrary:
    """
    便捷函数：对模板库进行聚类并保存
    
    Args:
        trajectory_memory: TrajectoryMemory 实例
        n_clusters_long: LONG 聚类数
        n_clusters_short: SHORT 聚类数
        save: 是否保存到文件
        verbose: 是否打印信息
    
    Returns:
        PrototypeLibrary
    """
    clusterer = TemplateClusterer(
        n_clusters_long=n_clusters_long,
        n_clusters_short=n_clusters_short,
    )
    
    library = clusterer.fit(trajectory_memory, verbose=verbose)
    
    if save:
        library.save(verbose=verbose)
    
    return library
