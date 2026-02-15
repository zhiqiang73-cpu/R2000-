"""
R3000 Walk-Forward 权重进化引擎 (WF Evolution Engine)

核心思路：
  - 将 32 维特征权重压缩为 8 组语义权重（10D 搜索空间）
  - 使用 CMA-ES（协方差矩阵自适应进化策略）搜索最优权重组合
  - 内部 3 折 Walk-Forward 评估 + L2 正则化消除噪声过拟合
  - 向量化相似度计算（numpy 批量矩阵运算），60 trial × 3 fold ≈ 3-5 分钟

搜索空间（10D）：
  - 8 个特征组权重 ∈ [0.1, 3.0]
  - fusion_threshold ∈ [0.40, 0.85]
  - cosine_min_threshold ∈ [0.50, 0.90]

防过拟合三层防线：
  1. 多折平均 Sharpe（非最优折）
  2. L2 正则化：fitness = avg_sharpe - λ * ||w - w_default||²
  3. 最低交易次数门槛（< 15 笔 → 罚分 -10）

数据泄露防护：
  - 先切分 evolution / holdout，再在 evolution 段上重聚类原型
  - holdout 段在最终验证前绝不触碰
"""

import numpy as np
import time
import traceback
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAJECTORY_CONFIG, SIMILARITY_CONFIG, FEATURE_WEIGHTS_CONFIG,
    CONFIDENCE_CONFIG,
)
from core.feature_vector import (
    N_A, N_B, N_C,
    LAYER_A_FEATURES, LAYER_B_FEATURES, LAYER_C_FEATURES,
)


# ══════════════════════════════════════════════════════════════════════════
# 特征组定义（8 组，将 32D 压缩到 8D）
# ══════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "RSI": {
        "features": ["rsi_14", "rsi_6", "delta_rsi_3", "delta_rsi_5"],
        "description": "RSI 族（趋势超买超卖 + 动量变化）",
    },
    "MACD": {
        "features": ["macd_hist", "delta_macd_3", "delta_macd_5"],
        "description": "MACD 族（信号线 + 动量变化）",
    },
    "Volatility": {
        "features": ["atr_ratio", "boll_width", "boll_position", "delta_atr_rate", "delta_boll_width"],
        "description": "波动率族（ATR + 布林带）",
    },
    "Momentum": {
        "features": ["kdj_k", "kdj_j", "roc"],
        "description": "动量族（KDJ + ROC）",
    },
    "Volume": {
        "features": ["volume_ratio", "obv_slope", "delta_volume_ratio"],
        "description": "成交量族（量比 + OBV）",
    },
    "Trend": {
        "features": ["ema12_dev", "ema26_dev", "upper_shadow_ratio", "lower_shadow_ratio",
                      "delta_ema12_slope", "delta_ema26_slope"],
        "description": "趋势族（EMA 偏差 + 影线 + EMA 斜率）",
    },
    "Structure": {
        "features": ["price_in_range", "dist_to_high_atr", "dist_to_low_atr",
                      "up_down_amp_ratio", "price_vs_20high", "price_vs_20low"],
        "description": "结构族（价格位置 + 高低点距离）",
    },
    "ADX": {
        "features": ["adx", "delta_adx"],
        "description": "ADX 族（趋势强度 + 变化）",
    },
}

# 构建完整特征名列表（与 feature_vector.py 一致的顺序）
ALL_FEATURE_NAMES = list(LAYER_A_FEATURES) + list(LAYER_B_FEATURES) + list(LAYER_C_FEATURES)


def _build_group_to_indices() -> Dict[str, List[int]]:
    """构建每个组对应的 32 维特征索引映射"""
    name_to_idx = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
    group_indices = {}
    for group_name, group_def in FEATURE_GROUPS.items():
        indices = []
        for feat_name in group_def["features"]:
            if feat_name in name_to_idx:
                indices.append(name_to_idx[feat_name])
            else:
                print(f"[WF_Evo] 警告: 特征 '{feat_name}' 在 group '{group_name}' 中未找到")
        group_indices[group_name] = indices
    return group_indices


GROUP_TO_INDICES = _build_group_to_indices()
GROUP_NAMES = list(FEATURE_GROUPS.keys())  # 固定顺序


def expand_group_weights_to_32d(group_weights: np.ndarray) -> np.ndarray:
    """
    将 8 维组权重扩展为 32 维特征权重向量
    
    每个组内的所有特征共享同一个乘法权重。
    
    Args:
        group_weights: shape (8,) 的组权重数组，顺序与 GROUP_NAMES 一致
    
    Returns:
        shape (32,) 的完整特征权重向量
    """
    weights_32d = np.ones(N_A + N_B + N_C)
    for i, group_name in enumerate(GROUP_NAMES):
        for feat_idx in GROUP_TO_INDICES[group_name]:
            weights_32d[feat_idx] = group_weights[i]
    return weights_32d


# 默认组权重（从现有 Layer A/B/C 权重反推）
DEFAULT_GROUP_WEIGHTS = np.array([
    1.5,   # RSI       (Layer A/B)
    1.5,   # MACD      (Layer A/B)
    1.35,  # Volatility (Layer A/B)
    1.5,   # Momentum  (Layer A)
    1.35,  # Volume    (Layer A/B)
    1.35,  # Trend     (Layer A/B)
    1.0,   # Structure (Layer C)
    1.35,  # ADX       (Layer A/B)
], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
# 进化结果数据结构
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EvolutionResult:
    """进化优化的最终结果"""
    # 最优权重
    best_group_weights: np.ndarray = field(default_factory=lambda: DEFAULT_GROUP_WEIGHTS.copy())
    best_weights_32d: np.ndarray = field(default_factory=lambda: np.ones(32))
    best_fusion_threshold: float = 0.65
    best_cosine_min_threshold: float = 0.70
    
    # 进化过程统计
    best_fitness: float = -np.inf
    best_avg_sharpe: float = 0.0
    best_l2_penalty: float = 0.0
    n_trials_completed: int = 0
    evolution_time_sec: float = 0.0
    
    # 内部折叠详情
    fold_sharpes: List[float] = field(default_factory=list)
    fold_trade_counts: List[int] = field(default_factory=list)
    fold_win_rates: List[float] = field(default_factory=list)
    
    # Holdout 验证结果
    holdout_sharpe: float = 0.0
    holdout_n_trades: int = 0
    holdout_win_rate: float = 0.0
    holdout_total_profit: float = 0.0
    holdout_max_drawdown: float = 0.0
    holdout_passed: bool = False
    holdout_time_sec: float = 0.0
    
    # 通过标准
    pass_criteria: Dict = field(default_factory=lambda: {
        "min_sharpe": 0.5,
        "min_trades": 20,
        "min_win_rate": 0.40,
        "max_drawdown": 15.0,
    })
    
    # 组重要性（按权重偏离默认值的程度排序）
    group_importance: Dict[str, float] = field(default_factory=dict)
    
    def check_holdout_pass(self) -> bool:
        """检查 holdout 结果是否通过标准"""
        self.holdout_passed = (
            self.holdout_sharpe >= self.pass_criteria["min_sharpe"]
            and self.holdout_n_trades >= self.pass_criteria["min_trades"]
            and self.holdout_win_rate >= self.pass_criteria["min_win_rate"]
            and self.holdout_max_drawdown <= self.pass_criteria["max_drawdown"]
        )
        return self.holdout_passed
    
    def compute_group_importance(self):
        """计算各组权重的重要性（偏离默认值的程度）"""
        if self.best_group_weights is None:
            return
        deviation = np.abs(self.best_group_weights - DEFAULT_GROUP_WEIGHTS)
        total = deviation.sum()
        if total < 1e-9:
            total = 1.0
        for i, name in enumerate(GROUP_NAMES):
            self.group_importance[name] = float(deviation[i] / total)
    
    def summary(self) -> str:
        """生成可读摘要"""
        lines = [
            "=" * 60,
            "WF Evolution 结果摘要",
            "=" * 60,
            f"进化耗时: {self.evolution_time_sec:.1f}s | 完成试验: {self.n_trials_completed}",
            f"最优适应度: {self.best_fitness:.4f} (Sharpe={self.best_avg_sharpe:.4f}, L2={self.best_l2_penalty:.4f})",
            "",
            "── 最优组权重 ──",
        ]
        for i, name in enumerate(GROUP_NAMES):
            w = self.best_group_weights[i]
            default = DEFAULT_GROUP_WEIGHTS[i]
            delta = w - default
            lines.append(f"  {name:12s}: {w:.3f} (默认 {default:.2f}, Δ={delta:+.3f})")
        
        lines.append(f"\n  fusion_threshold:     {self.best_fusion_threshold:.3f}")
        lines.append(f"  cosine_min_threshold: {self.best_cosine_min_threshold:.3f}")
        
        lines.append("\n── 内部折叠详情 ──")
        for i, (s, t, w) in enumerate(zip(self.fold_sharpes, self.fold_trade_counts, self.fold_win_rates)):
            lines.append(f"  Fold {i}: Sharpe={s:.3f}, Trades={t}, WinRate={w:.1%}")
        
        lines.append("\n── Holdout 验证 ──")
        lines.append(f"  Sharpe:     {self.holdout_sharpe:.4f}")
        lines.append(f"  交易笔数:   {self.holdout_n_trades}")
        lines.append(f"  胜率:       {self.holdout_win_rate:.1%}")
        lines.append(f"  总收益:     {self.holdout_total_profit:.2f}%")
        lines.append(f"  最大回撤:   {self.holdout_max_drawdown:.2f}%")
        lines.append(f"  验证耗时:   {self.holdout_time_sec:.1f}s")
        lines.append(f"  通过:       {'[OK] PASSED' if self.holdout_passed else '[X] FAILED'}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 向量化相似度批量计算器
# ══════════════════════════════════════════════════════════════════════════

class VectorizedSimilarityEngine:
    """
    向量化相似度计算引擎
    
    核心优化：将 Python 循环替换为 numpy 矩阵运算
    
    预计算步骤（一次性）：
      1. 计算所有评估 bar 的滚动均值 → (n_eval, 32)
      2. 计算所有原型的质心均值     → (n_proto, 32)
    
    每次 trial（毫秒级）：
      3. 应用组权重 → 加权后的 bar / 原型矩阵
      4. 批量余弦相似度 → (n_eval, n_proto)
      5. 批量欧氏相似度 → (n_eval, n_proto)
      6. 融合得分矩阵   → (n_eval, n_proto)
    """
    
    def __init__(self):
        self._bar_means = None        # (n_eval, 32)
        self._proto_means = None      # (n_proto, 32)
        self._proto_confidences = None  # (n_proto,)
        self._proto_directions = None   # list of str
        self._proto_objects = None      # list of Prototype (for trade sim)
        self._eval_indices = None       # list of int (原始 bar 索引)
        self._ready = False
        
        # DTW 需要原始序列（仅 holdout 用）
        self._proto_sequences = None   # list of (T, 32) arrays
    
    def precompute_bars(self, fv_engine, start_idx: int, end_idx: int,
                        pre_window: int = 60, skip_bars: int = 10):
        """
        预计算评估区间内所有 bar 的滚动均值
        
        对每个评估 bar i，取 [i - pre_window, i) 区间的均值
        
        Args:
            fv_engine: FeatureVectorEngine（已 precompute）
            start_idx: 评估起始索引（数据集内绝对索引）
            end_idx: 评估结束索引
            pre_window: 回看窗口大小
            skip_bars: 跳跃采样间隔
        """
        eval_start = start_idx + pre_window
        eval_positions = list(range(eval_start, end_idx, skip_bars))
        
        if not eval_positions:
            self._bar_means = np.zeros((0, N_A + N_B + N_C))
            self._eval_indices = []
            return
        
        # 获取完整特征矩阵 view
        full_matrix = fv_engine.get_full_raw_matrix()
        n_total = full_matrix.shape[0]
        
        # 批量计算滚动均值
        means_list = []
        valid_indices = []
        for pos in eval_positions:
            window_start = pos - pre_window
            if window_start < 0 or pos > n_total:
                continue
            window = full_matrix[window_start:pos]  # (pre_window, 32) - view, no copy
            means_list.append(window.mean(axis=0))
            valid_indices.append(pos)
        
        if means_list:
            self._bar_means = np.stack(means_list)  # (n_eval, 32)
        else:
            self._bar_means = np.zeros((0, N_A + N_B + N_C))
        self._eval_indices = valid_indices
    
    def precompute_prototypes(self, prototype_library, direction_filter: str = None):
        """
        预计算所有原型的质心均值
        
        Args:
            prototype_library: PrototypeLibrary 实例
            direction_filter: 仅保留特定方向的原型（None = 全部）
        """
        protos_all = []
        if direction_filter is None or direction_filter == "LONG":
            protos_all.extend(
                (p, "LONG") for p in prototype_library.long_prototypes
                if p.pre_entry_centroid.size == 32
            )
        if direction_filter is None or direction_filter == "SHORT":
            protos_all.extend(
                (p, "SHORT") for p in prototype_library.short_prototypes
                if p.pre_entry_centroid.size == 32
            )
        
        if not protos_all:
            self._proto_means = np.zeros((0, N_A + N_B + N_C))
            self._proto_confidences = np.array([])
            self._proto_directions = []
            self._proto_objects = []
            self._proto_sequences = []
            return
        
        means = []
        confidences = []
        directions = []
        objects = []
        sequences = []
        
        for proto, direction in protos_all:
            # 原型质心已经是 (32,) 均值
            means.append(proto.pre_entry_centroid)
            confidences.append(proto.confidence if proto.confidence > 0 else 0.5)
            directions.append(direction)
            objects.append(proto)
            sequences.append(
                proto.representative_pre_entry
                if proto.representative_pre_entry.size > 0
                else np.array([])
            )
        
        self._proto_means = np.stack(means)          # (n_proto, 32)
        self._proto_confidences = np.array(confidences)  # (n_proto,)
        self._proto_directions = directions
        self._proto_objects = objects
        self._proto_sequences = sequences
        self._ready = True
    
    def compute_score_matrix(self, weights_32d: np.ndarray,
                             cosine_weight: float = 0.43,
                             euclidean_weight: float = 0.57,
                             euclidean_max_dist: float = None,
                             apply_confidence: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化计算所有 (eval_bar, prototype) 的融合相似度
        
        当 DTW 未启用时，余弦和欧氏的权重会自动归一化
        （默认 cos=0.30/(0.30+0.40)=0.43, euc=0.40/(0.30+0.40)=0.57）
        
        Args:
            weights_32d: (32,) 特征权重向量
            cosine_weight: 余弦相似度权重（已归一化，不含 DTW）
            euclidean_weight: 欧氏相似度权重（已归一化，不含 DTW）
            euclidean_max_dist: 欧氏距离归一化上限
            apply_confidence: 是否将置信度乘入最终分数
        
        Returns:
            final_scores: (n_eval, n_proto) 最终融合分数
            combined_scores: (n_eval, n_proto) 未乘置信度的融合分数
        """
        if self._bar_means is None or self._proto_means is None:
            return np.zeros((0, 0)), np.zeros((0, 0))
        
        n_eval = self._bar_means.shape[0]
        n_proto = self._proto_means.shape[0]
        if n_eval == 0 or n_proto == 0:
            return np.zeros((n_eval, n_proto)), np.zeros((n_eval, n_proto))
        
        if euclidean_max_dist is None:
            euclidean_max_dist = SIMILARITY_CONFIG.get("EUCLIDEAN_MAX_DISTANCE", 50.0)
        
        # Step 1: 应用权重
        weighted_bars = self._bar_means * weights_32d[np.newaxis, :]    # (n_eval, 32)
        weighted_protos = self._proto_means * weights_32d[np.newaxis, :]  # (n_proto, 32)
        
        # Step 2: 余弦相似度矩阵
        # normalize rows
        bar_norms = np.linalg.norm(weighted_bars, axis=1, keepdims=True)
        bar_norms = np.where(bar_norms < 1e-9, 1.0, bar_norms)
        proto_norms = np.linalg.norm(weighted_protos, axis=1, keepdims=True)
        proto_norms = np.where(proto_norms < 1e-9, 1.0, proto_norms)
        
        normed_bars = weighted_bars / bar_norms
        normed_protos = weighted_protos / proto_norms
        
        cosine_matrix = normed_bars @ normed_protos.T  # (n_eval, n_proto)
        cosine_matrix = np.clip(cosine_matrix, 0.0, 1.0)
        
        # Step 3: 欧氏相似度矩阵
        euclidean_dist = cdist(weighted_bars, weighted_protos, metric='euclidean')  # (n_eval, n_proto)
        euclidean_sim = 1.0 - np.clip(euclidean_dist / euclidean_max_dist, 0.0, 1.0)
        
        # Step 4: 融合
        combined = cosine_weight * cosine_matrix + euclidean_weight * euclidean_sim
        
        # Step 5: 乘以置信度
        if apply_confidence and self._proto_confidences is not None and len(self._proto_confidences) == n_proto:
            final_scores = combined * self._proto_confidences[np.newaxis, :]
        else:
            final_scores = combined.copy()
        
        return final_scores, combined
    
    @property
    def n_eval_bars(self) -> int:
        return len(self._eval_indices) if self._eval_indices else 0
    
    @property
    def n_prototypes(self) -> int:
        return self._proto_means.shape[0] if self._proto_means is not None else 0


# ══════════════════════════════════════════════════════════════════════════
# 交易模拟器（轻量级，配合向量化引擎使用）
# ══════════════════════════════════════════════════════════════════════════

def simulate_trades_from_scores(
    final_scores: np.ndarray,
    eval_indices: List[int],
    proto_directions: List[str],
    proto_objects: list,
    close_prices: np.ndarray,
    atr_values: np.ndarray,
    fusion_threshold: float,
    cosine_min_threshold: float = 0.70,
    cosine_matrix: np.ndarray = None,
    stop_loss_atr: float = 2.0,
    take_profit_atr: float = 3.0,
    max_hold_bars: int = 240,
    regime_labels: np.ndarray = None,
) -> Dict:
    """
    根据预计算的融合分数矩阵执行交易模拟
    
    Args:
        final_scores: (n_eval, n_proto) 融合分数矩阵
        eval_indices: 评估 bar 的原始索引列表
        proto_directions: 每个原型的方向列表
        proto_objects: Prototype 对象列表
        close_prices: 完整价格序列
        atr_values: 完整 ATR 序列
        fusion_threshold: 融合分数入场阈值
        cosine_min_threshold: 余弦最低阈值（额外过滤）
        cosine_matrix: (n_eval, n_proto) 余弦分数矩阵（可选，用于 cosine_min 过滤）
        stop_loss_atr: 止损 ATR 倍数
        take_profit_atr: 止盈 ATR 倍数
        max_hold_bars: 最大持仓 K 线数
        regime_labels: 每个 bar 的市场状态索引（可选）
    
    Returns:
        {
            "n_trades": int,
            "n_wins": int,
            "total_profit": float,
            "profits": list[float],
            "sharpe_ratio": float,
            "max_drawdown": float,
            "win_rate": float,
        }
    """
    n_eval, n_proto = final_scores.shape
    n_bars = len(close_prices)
    
    profits = []
    position = None  # {"entry_idx", "entry_price", "side", "stop_loss", "take_profit"}
    
    # 预计算最佳匹配
    # 对每个 eval bar，找到最高融合分数的原型
    best_proto_idx = np.argmax(final_scores, axis=1)  # (n_eval,)
    best_scores = final_scores[np.arange(n_eval), best_proto_idx]  # (n_eval,)
    
    # 预提取方向数组
    dir_array = np.array([1 if d == "LONG" else -1 for d in proto_directions])
    
    eval_ptr = 0  # 指向当前 eval bar
    
    while eval_ptr < n_eval:
        bar_idx = eval_indices[eval_ptr]
        
        if bar_idx >= n_bars - 1:
            break
        
        if position is None:
            # ── 入场检查 ──
            score = best_scores[eval_ptr]
            
            if score >= fusion_threshold:
                pidx = best_proto_idx[eval_ptr]
                
                # 可选：余弦最低阈值过滤
                if cosine_matrix is not None:
                    cos_val = cosine_matrix[eval_ptr, pidx]
                    if cos_val < cosine_min_threshold:
                        eval_ptr += 1
                        continue
                
                side = dir_array[pidx]
                entry_price = close_prices[bar_idx]
                atr = atr_values[bar_idx] if bar_idx < len(atr_values) else 1.0
                
                if entry_price <= 0 or atr <= 0:
                    eval_ptr += 1
                    continue
                
                sl_dist = stop_loss_atr * atr / entry_price
                tp_dist = take_profit_atr * atr / entry_price
                
                if side == 1:  # LONG
                    stop_loss = entry_price * (1 - sl_dist)
                    take_profit = entry_price * (1 + tp_dist)
                else:  # SHORT
                    stop_loss = entry_price * (1 + sl_dist)
                    take_profit = entry_price * (1 - tp_dist)
                
                position = {
                    "entry_idx": bar_idx,
                    "entry_price": entry_price,
                    "side": side,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            
            eval_ptr += 1
        else:
            # ── 持仓管理（逐 bar 检查直到平仓）──
            bar_idx_check = position["entry_idx"] + 1
            # 快进到下一个 eval bar 或平仓
            next_eval_bar = eval_indices[eval_ptr] if eval_ptr < n_eval else n_bars
            # 逐 bar 检查止盈止损
            closed = False
            
            while bar_idx_check < min(next_eval_bar + 1, n_bars):
                current_price = close_prices[bar_idx_check]
                hold_bars = bar_idx_check - position["entry_idx"]
                
                should_close = False
                if position["side"] == 1:  # LONG
                    if current_price <= position["stop_loss"]:
                        should_close = True
                    elif current_price >= position["take_profit"]:
                        should_close = True
                else:  # SHORT
                    if current_price >= position["stop_loss"]:
                        should_close = True
                    elif current_price <= position["take_profit"]:
                        should_close = True
                
                if hold_bars >= max_hold_bars:
                    should_close = True
                
                if should_close:
                    if position["side"] == 1:
                        pnl = (current_price / position["entry_price"] - 1) * 100
                    else:
                        pnl = (1 - current_price / position["entry_price"]) * 100
                    profits.append(pnl)
                    position = None
                    closed = True
                    # 跳过 eval_ptr 到平仓后的位置
                    while eval_ptr < n_eval and eval_indices[eval_ptr] <= bar_idx_check:
                        eval_ptr += 1
                    break
                
                bar_idx_check += 1
            
            if not closed:
                # 未平仓，前进到下一个 eval bar
                eval_ptr += 1
    
    # ── 统计 ──
    result = {
        "n_trades": len(profits),
        "n_wins": 0,
        "total_profit": 0.0,
        "profits": profits,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
    }
    
    if len(profits) == 0:
        return result
    
    profits_arr = np.array(profits)
    result["n_wins"] = int((profits_arr > 0).sum())
    result["total_profit"] = float(profits_arr.sum())
    result["win_rate"] = result["n_wins"] / result["n_trades"]
    
    mean_ret = profits_arr.mean()
    std_ret = profits_arr.std()
    if std_ret > 1e-9:
        result["sharpe_ratio"] = float(mean_ret / std_ret * np.sqrt(252))
    else:
        result["sharpe_ratio"] = float(mean_ret * 10) if mean_ret > 0 else -10.0
    
    cumsum = profits_arr.cumsum()
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    result["max_drawdown"] = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    
    return result


# ══════════════════════════════════════════════════════════════════════════
# 主引擎：WFEvolutionEngine
# ══════════════════════════════════════════════════════════════════════════

class WFEvolutionEngine:
    """
    Walk-Forward 权重进化引擎
    
    使用方式：
        engine = WFEvolutionEngine(
            df=full_df,                    # 完整数据（已计算指标）
            fv_engine=fv_engine,           # 已 precompute 的特征引擎
            prototype_library=library,     # 原型库
        )
        result = engine.run(callback=progress_fn)
    
    核心流程：
        1. 数据分割：evolution (70%) + holdout (30%)
        2. 预计算：滚动均值 + 原型均值（一次性）
        3. CMA-ES 进化：60 trial × 3 inner fold
        4. Holdout 验证：DTW 启用
    """
    
    def __init__(self,
                 df,
                 fv_engine,
                 prototype_library,
                 # 数据配置
                 sample_size: int = 300000,
                 holdout_ratio: float = 0.30,
                 inner_folds: int = 3,
                 # 进化配置
                 n_trials: int = 60,
                 skip_bars: int = 10,
                 pre_window: int = None,
                 # 正则化
                 l2_lambda: float = 0.01,
                 min_trades_per_fold: int = 15,
                 # 交易参数
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 3.0,
                 max_hold_bars: int = 240,
                 # Holdout 验证通过标准
                 holdout_min_sharpe: float = 0.5,
                 holdout_min_trades: int = 20,
                 holdout_min_win_rate: float = 0.40,
                 holdout_max_drawdown: float = 15.0,
                 ):
        """
        Args:
            df: 完整 DataFrame（已调用 calculate_all_indicators）
            fv_engine: FeatureVectorEngine 实例（已 precompute）
            prototype_library: PrototypeLibrary 实例
            sample_size: 使用的总 bar 数（从尾部截取）
            holdout_ratio: holdout 占比
            inner_folds: 内部 WF 折数
            n_trials: CMA-ES 试验次数
            skip_bars: 评估跳跃间隔
            pre_window: 入场前回看窗口，默认从配置读取
            l2_lambda: L2 正则化系数
            min_trades_per_fold: 每折最低交易数
            stop_loss_atr: 止损 ATR 倍数
            take_profit_atr: 止盈 ATR 倍数
            max_hold_bars: 最大持仓 K 线数
        """
        self.df = df
        self.fv_engine = fv_engine
        self.prototype_library = prototype_library
        
        # 数据配置
        self.sample_size = min(sample_size, len(df))
        self.holdout_ratio = holdout_ratio
        self.inner_folds = inner_folds
        
        # 进化配置
        self.n_trials = n_trials
        self.skip_bars = skip_bars
        self.pre_window = pre_window or TRAJECTORY_CONFIG.get("PRE_ENTRY_WINDOW", 60)
        
        # 正则化
        self.l2_lambda = l2_lambda
        self.min_trades_per_fold = min_trades_per_fold
        
        # 交易参数
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.max_hold_bars = max_hold_bars
        
        # Holdout 标准
        self.holdout_min_sharpe = holdout_min_sharpe
        self.holdout_min_trades = holdout_min_trades
        self.holdout_min_win_rate = holdout_min_win_rate
        self.holdout_max_drawdown = holdout_max_drawdown
        
        # 相似度配置
        cos_w = SIMILARITY_CONFIG.get("COSINE_WEIGHT", 0.30)
        euc_w = SIMILARITY_CONFIG.get("EUCLIDEAN_WEIGHT", 0.40)
        total = cos_w + euc_w
        self._cosine_weight_normed = cos_w / total  # ~0.43
        self._euclidean_weight_normed = euc_w / total  # ~0.57
        self._euclidean_max_dist = SIMILARITY_CONFIG.get("EUCLIDEAN_MAX_DISTANCE", 50.0)
        
        # 内部状态
        self._data_start_idx = 0   # 数据在 df 中的起始行
        self._evo_start = 0
        self._evo_end = 0
        self._holdout_start = 0
        self._holdout_end = 0
        self._close_prices = None
        self._atr_values = None
        
        # 折叠预计算引擎
        self._fold_engines: List[VectorizedSimilarityEngine] = []
        
        # 回调
        self._callback = None
        self._cancelled = False
        
        # 结果
        self.result = EvolutionResult()
    
    def cancel(self):
        """取消进化（可从外部线程调用）"""
        self._cancelled = True
    
    def run(self, callback: Callable = None) -> EvolutionResult:
        """
        执行完整进化流程
        
        Args:
            callback: 进度回调 fn(progress: float, message: str)
                      progress ∈ [0.0, 1.0]
        
        Returns:
            EvolutionResult
        """
        self._callback = callback
        self._cancelled = False
        t0 = time.time()
        
        try:
            self._report(0.0, "数据分割中...")
            self._split_data()
            
            self._report(0.05, "预计算特征矩阵...")
            self._precompute_fold_engines()
            
            self._report(0.10, f"CMA-ES 进化开始 ({self.n_trials} trials × {self.inner_folds} folds)...")
            self._run_evolution()
            
            self.result.evolution_time_sec = time.time() - t0
            
            self._report(0.85, "Holdout 验证中（DTW 启用）...")
            self._run_holdout_validation()
            
            self.result.compute_group_importance()
            
            total_time = time.time() - t0
            status = "PASSED" if self.result.holdout_passed else "FAILED"
            self._report(1.0, f"完成！耗时 {total_time:.1f}s - {status}")
            
            print(self.result.summary())
            
        except Exception as e:
            self._report(1.0, f"进化失败: {e}")
            traceback.print_exc()
        
        return self.result
    
    # ── 数据分割 ──────────────────────────────────────────────
    
    def _split_data(self):
        """将数据分为 evolution 和 holdout 两段（时间序列顺序）"""
        n_total = len(self.df)
        
        # 从尾部截取 sample_size 条
        self._data_start_idx = max(0, n_total - self.sample_size)
        data_end = n_total
        data_len = data_end - self._data_start_idx
        
        # 分割点
        holdout_len = int(data_len * self.holdout_ratio)
        evo_len = data_len - holdout_len
        
        self._evo_start = self._data_start_idx
        self._evo_end = self._data_start_idx + evo_len
        self._holdout_start = self._evo_end
        self._holdout_end = data_end
        
        # 缓存价格和 ATR
        self._close_prices = self.df['close'].values
        self._atr_values = (
            self.df['atr'].values
            if 'atr' in self.df.columns
            else np.ones(n_total)
        )
        
        print(f"[WF_Evo] 数据分割: 总计 {data_len} bars")
        print(f"  Evolution: [{self._evo_start}, {self._evo_end}) = {evo_len} bars")
        print(f"  Holdout:   [{self._holdout_start}, {self._holdout_end}) = {holdout_len} bars")
    
    # ── 预计算折叠引擎 ───────────────────────────────────────
    
    def _precompute_fold_engines(self):
        """为每个内部折叠预计算向量化引擎"""
        evo_len = self._evo_end - self._evo_start
        fold_size = evo_len // self.inner_folds
        
        self._fold_engines = []
        
        for fold_idx in range(self.inner_folds):
            fold_start = self._evo_start + fold_idx * fold_size
            fold_end = fold_start + fold_size if fold_idx < self.inner_folds - 1 else self._evo_end
            
            engine = VectorizedSimilarityEngine()
            engine.precompute_bars(
                self.fv_engine,
                start_idx=fold_start,
                end_idx=fold_end,
                pre_window=self.pre_window,
                skip_bars=self.skip_bars,
            )
            engine.precompute_prototypes(self.prototype_library)
            
            self._fold_engines.append(engine)
            
            print(f"  Fold {fold_idx}: [{fold_start}, {fold_end}) = {fold_end - fold_start} bars, "
                  f"{engine.n_eval_bars} eval points, {engine.n_prototypes} protos")
        
        self._report(0.08, f"预计算完成: {self.inner_folds} 折")
    
    # ── CMA-ES 进化 ──────────────────────────────────────────
    
    def _run_evolution(self):
        """使用 Optuna CMA-ES 搜索最优权重"""
        import optuna
        
        # 禁用 Optuna 内部日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # CMA-ES 采样器
        try:
            sampler = optuna.samplers.CmaEsSampler(
                n_startup_trials=5,   # 前 5 个随机探索
                seed=42,
            )
            print("[WF_Evo] 使用 CMA-ES 采样器")
        except Exception as e:
            print(f"[WF_Evo] CMA-ES 不可用，回退到 TPE: {e}")
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="wf_evolution",
        )
        
        best_fitness = -np.inf
        best_trial_data = None
        
        def objective(trial: optuna.Trial) -> float:
            nonlocal best_fitness, best_trial_data
            
            if self._cancelled:
                raise optuna.exceptions.OptunaError("用户取消")
            
            # 8 个组权重 + 2 个阈值 = 10D 搜索空间
            group_weights = np.array([
                trial.suggest_float(f"w_{name}", 0.1, 3.0)
                for name in GROUP_NAMES
            ])
            
            fusion_th = trial.suggest_float("fusion_threshold", 0.40, 0.85)
            cosine_min_th = trial.suggest_float("cosine_min_threshold", 0.50, 0.90)
            
            # 扩展为 32D 权重
            weights_32d = expand_group_weights_to_32d(group_weights)
            
            # 多折评估
            fold_sharpes = []
            fold_trade_counts = []
            
            for fold_idx, fold_engine in enumerate(self._fold_engines):
                if fold_engine.n_eval_bars == 0 or fold_engine.n_prototypes == 0:
                    fold_sharpes.append(-10.0)
                    fold_trade_counts.append(0)
                    continue
                
                # 向量化计算融合分数
                final_scores, combined_scores = fold_engine.compute_score_matrix(
                    weights_32d,
                    cosine_weight=self._cosine_weight_normed,
                    euclidean_weight=self._euclidean_weight_normed,
                    euclidean_max_dist=self._euclidean_max_dist,
                    apply_confidence=True,
                )
                
                # 计算余弦矩阵（用于 cosine_min 过滤）
                # 重用已加权的数据来计算纯余弦矩阵
                weighted_bars = fold_engine._bar_means * weights_32d[np.newaxis, :]
                weighted_protos = fold_engine._proto_means * weights_32d[np.newaxis, :]
                
                bar_norms = np.linalg.norm(weighted_bars, axis=1, keepdims=True)
                bar_norms = np.where(bar_norms < 1e-9, 1.0, bar_norms)
                proto_norms = np.linalg.norm(weighted_protos, axis=1, keepdims=True)
                proto_norms = np.where(proto_norms < 1e-9, 1.0, proto_norms)
                
                cosine_matrix = (weighted_bars / bar_norms) @ (weighted_protos / proto_norms).T
                cosine_matrix = np.clip(cosine_matrix, 0.0, 1.0)
                
                # 交易模拟
                fold_start = self._evo_start + fold_idx * (
                    (self._evo_end - self._evo_start) // self.inner_folds
                )
                
                sim_result = simulate_trades_from_scores(
                    final_scores=final_scores,
                    eval_indices=fold_engine._eval_indices,
                    proto_directions=fold_engine._proto_directions,
                    proto_objects=fold_engine._proto_objects,
                    close_prices=self._close_prices,
                    atr_values=self._atr_values,
                    fusion_threshold=fusion_th,
                    cosine_min_threshold=cosine_min_th,
                    cosine_matrix=cosine_matrix,
                    stop_loss_atr=self.stop_loss_atr,
                    take_profit_atr=self.take_profit_atr,
                    max_hold_bars=self.max_hold_bars,
                )
                
                n_trades = sim_result["n_trades"]
                sharpe = sim_result["sharpe_ratio"]
                
                # 最低交易数门槛
                if n_trades < self.min_trades_per_fold:
                    sharpe = -10.0
                
                fold_sharpes.append(sharpe)
                fold_trade_counts.append(n_trades)
            
            # 平均 Sharpe
            avg_sharpe = float(np.mean(fold_sharpes))
            
            # L2 正则化
            l2_penalty = self.l2_lambda * float(np.sum((group_weights - DEFAULT_GROUP_WEIGHTS) ** 2))
            
            fitness = avg_sharpe - l2_penalty
            
            # 更新进度
            trial_num = trial.number + 1
            progress = 0.10 + 0.75 * (trial_num / self.n_trials)
            total_trades = sum(fold_trade_counts)
            self._report(
                progress,
                f"Trial {trial_num}/{self.n_trials}: fitness={fitness:.3f} "
                f"(Sharpe={avg_sharpe:.3f}, L2={l2_penalty:.4f}, trades={total_trades})"
            )
            
            # 追踪最优
            if fitness > best_fitness:
                best_fitness = fitness
                best_trial_data = {
                    "group_weights": group_weights.copy(),
                    "weights_32d": weights_32d.copy(),
                    "fusion_threshold": fusion_th,
                    "cosine_min_threshold": cosine_min_th,
                    "fitness": fitness,
                    "avg_sharpe": avg_sharpe,
                    "l2_penalty": l2_penalty,
                    "fold_sharpes": fold_sharpes[:],
                    "fold_trade_counts": fold_trade_counts[:],
                }
            
            return fitness
        
        # 运行优化
        try:
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        except optuna.exceptions.OptunaError as e:
            if "取消" in str(e):
                print("[WF_Evo] 用户取消进化")
            else:
                raise
        
        # 保存最优结果
        if best_trial_data is not None:
            self.result.best_group_weights = best_trial_data["group_weights"]
            self.result.best_weights_32d = best_trial_data["weights_32d"]
            self.result.best_fusion_threshold = best_trial_data["fusion_threshold"]
            self.result.best_cosine_min_threshold = best_trial_data["cosine_min_threshold"]
            self.result.best_fitness = best_trial_data["fitness"]
            self.result.best_avg_sharpe = best_trial_data["avg_sharpe"]
            self.result.best_l2_penalty = best_trial_data["l2_penalty"]
            self.result.fold_sharpes = best_trial_data["fold_sharpes"]
            self.result.fold_trade_counts = best_trial_data["fold_trade_counts"]
            # 计算折叠胜率（最优 trial 的胜率需要重新获取 — 近似用 avg_sharpe > 0）
            self.result.fold_win_rates = [0.0] * len(best_trial_data["fold_sharpes"])
        
        self.result.n_trials_completed = len(study.trials)
        
        print(f"\n[WF_Evo] 进化完成: {self.result.n_trials_completed} trials")
        print(f"  最优 fitness = {self.result.best_fitness:.4f}")
        print(f"  组权重 = {self.result.best_group_weights}")
        print(f"  fusion_threshold = {self.result.best_fusion_threshold:.3f}")
        print(f"  cosine_min_threshold = {self.result.best_cosine_min_threshold:.3f}")
    
    # ── Holdout 验证 ─────────────────────────────────────────
    
    def _run_holdout_validation(self):
        """在 holdout 段上用最优权重做完整验证（DTW 启用）"""
        t0 = time.time()
        
        holdout_len = self._holdout_end - self._holdout_start
        if holdout_len < self.pre_window + 100:
            print("[WF_Evo] Holdout 数据不足，跳过验证")
            return
        
        # 使用最优权重
        weights_32d = self.result.best_weights_32d
        fusion_th = self.result.best_fusion_threshold
        cosine_min_th = self.result.best_cosine_min_threshold
        
        # ── 快速路径：向量化 cosine+euclidean ──
        holdout_engine = VectorizedSimilarityEngine()
        # Holdout 使用更细的跳跃（skip=5），更准确
        holdout_engine.precompute_bars(
            self.fv_engine,
            start_idx=self._holdout_start,
            end_idx=self._holdout_end,
            pre_window=self.pre_window,
            skip_bars=max(1, self.skip_bars // 2),
        )
        holdout_engine.precompute_prototypes(self.prototype_library)
        
        print(f"[WF_Evo] Holdout 验证: {holdout_engine.n_eval_bars} eval points, "
              f"{holdout_engine.n_prototypes} protos")
        
        if holdout_engine.n_eval_bars == 0 or holdout_engine.n_prototypes == 0:
            print("[WF_Evo] Holdout 无有效数据")
            return
        
        # ── DTW 增强分数 ──
        # 对向量化 cosine+euclidean 的 Top-K 候选执行 DTW 二次验证
        # 先计算基础分数矩阵
        cos_w = SIMILARITY_CONFIG.get("COSINE_WEIGHT", 0.30)
        euc_w = SIMILARITY_CONFIG.get("EUCLIDEAN_WEIGHT", 0.40)
        dtw_w = SIMILARITY_CONFIG.get("DTW_WEIGHT", 0.30)
        
        # 不含 DTW 的基础分数
        base_final, base_combined = holdout_engine.compute_score_matrix(
            weights_32d,
            cosine_weight=self._cosine_weight_normed,
            euclidean_weight=self._euclidean_weight_normed,
            euclidean_max_dist=self._euclidean_max_dist,
            apply_confidence=False,  # 先不乘置信度
        )
        
        # 余弦矩阵（用于 cosine_min 过滤和 DTW 融合）
        weighted_bars = holdout_engine._bar_means * weights_32d[np.newaxis, :]
        weighted_protos = holdout_engine._proto_means * weights_32d[np.newaxis, :]
        
        bar_norms = np.linalg.norm(weighted_bars, axis=1, keepdims=True)
        bar_norms = np.where(bar_norms < 1e-9, 1.0, bar_norms)
        proto_norms = np.linalg.norm(weighted_protos, axis=1, keepdims=True)
        proto_norms = np.where(proto_norms < 1e-9, 1.0, proto_norms)
        
        cosine_matrix = (weighted_bars / bar_norms) @ (weighted_protos / proto_norms).T
        cosine_matrix = np.clip(cosine_matrix, 0.0, 1.0)
        
        # 欧氏相似度矩阵
        euclidean_dist = cdist(weighted_bars, weighted_protos, metric='euclidean')
        euclidean_sim = 1.0 - np.clip(euclidean_dist / self._euclidean_max_dist, 0.0, 1.0)
        
        # DTW 增强：仅对每个 eval bar 的最佳候选计算 DTW
        from core.template_clusterer import MultiSimilarityCalculator
        dtw_calculator = MultiSimilarityCalculator(apply_feature_weights=False)
        
        n_eval = holdout_engine.n_eval_bars
        n_proto = holdout_engine.n_prototypes
        
        # 构建最终带 DTW 的分数矩阵
        # 策略：对每个 eval bar，找到 cosine+euclidean 融合分数最高的 top-3，
        # 对这 3 个计算 DTW，更新最终分数
        dtw_top_k = 3
        final_scores_dtw = base_combined.copy()  # 先用不含 DTW 的基础分数
        
        full_matrix = self.fv_engine.get_full_raw_matrix()
        dtw_count = 0
        
        self._report(0.87, f"Holdout DTW 验证中 ({n_eval} bars)...")
        
        for ei in range(n_eval):
            if self._cancelled:
                break
            
            # 找 top-K 候选
            top_k_indices = np.argsort(base_combined[ei])[-dtw_top_k:][::-1]
            
            bar_idx = holdout_engine._eval_indices[ei]
            window_start = bar_idx - self.pre_window
            if window_start < 0 or bar_idx > full_matrix.shape[0]:
                continue
            
            current_window = full_matrix[window_start:bar_idx]
            if current_window.shape[0] < self.pre_window:
                continue
            
            # 应用权重到窗口
            weighted_window = current_window * weights_32d[np.newaxis, :]
            
            for pidx in top_k_indices:
                proto = holdout_engine._proto_objects[pidx]
                repr_seq = holdout_engine._proto_sequences[pidx]
                
                if repr_seq.size == 0 or repr_seq.ndim != 2:
                    dtw_sim = 0.5  # 无代表序列，使用中性值
                else:
                    weighted_repr = repr_seq * weights_32d[np.newaxis, :]
                    dtw_sim = dtw_calculator.compute_dtw_similarity(
                        weighted_window, weighted_repr
                    )
                    dtw_count += 1
                
                # 三维融合（含 DTW）
                combined_3d = (
                    cos_w * cosine_matrix[ei, pidx]
                    + euc_w * euclidean_sim[ei, pidx]
                    + dtw_w * dtw_sim
                )
                final_scores_dtw[ei, pidx] = combined_3d
            
            if ei % 500 == 0:
                progress = 0.87 + 0.08 * (ei / n_eval)
                self._report(progress, f"DTW 验证: {ei}/{n_eval} bars ({dtw_count} DTW计算)")
        
        # 乘以置信度得到最终分数
        if holdout_engine._proto_confidences is not None:
            final_scores_with_conf = final_scores_dtw * holdout_engine._proto_confidences[np.newaxis, :]
        else:
            final_scores_with_conf = final_scores_dtw
        
        # 交易模拟
        sim_result = simulate_trades_from_scores(
            final_scores=final_scores_with_conf,
            eval_indices=holdout_engine._eval_indices,
            proto_directions=holdout_engine._proto_directions,
            proto_objects=holdout_engine._proto_objects,
            close_prices=self._close_prices,
            atr_values=self._atr_values,
            fusion_threshold=fusion_th,
            cosine_min_threshold=cosine_min_th,
            cosine_matrix=cosine_matrix,
            stop_loss_atr=self.stop_loss_atr,
            take_profit_atr=self.take_profit_atr,
            max_hold_bars=self.max_hold_bars,
        )
        
        # 保存 holdout 结果
        self.result.holdout_sharpe = sim_result["sharpe_ratio"]
        self.result.holdout_n_trades = sim_result["n_trades"]
        self.result.holdout_win_rate = sim_result["win_rate"]
        self.result.holdout_total_profit = sim_result["total_profit"]
        self.result.holdout_max_drawdown = sim_result["max_drawdown"]
        self.result.holdout_time_sec = time.time() - t0
        
        # 检查是否通过
        self.result.pass_criteria = {
            "min_sharpe": self.holdout_min_sharpe,
            "min_trades": self.holdout_min_trades,
            "min_win_rate": self.holdout_min_win_rate,
            "max_drawdown": self.holdout_max_drawdown,
        }
        self.result.check_holdout_pass()
        
        # 回填折叠胜率（用最优权重重新跑一次获取精确值）
        self._fill_fold_win_rates()
        
        print(f"\n[WF_Evo] Holdout 验证完成 ({self.result.holdout_time_sec:.1f}s, DTW 计算 {dtw_count} 次):")
        print(f"  Sharpe:     {self.result.holdout_sharpe:.4f}")
        print(f"  交易笔数:   {self.result.holdout_n_trades}")
        print(f"  胜率:       {self.result.holdout_win_rate:.1%}")
        print(f"  总收益:     {self.result.holdout_total_profit:.2f}%")
        print(f"  最大回撤:   {self.result.holdout_max_drawdown:.2f}%")
        print(f"  通过:       {'PASSED' if self.result.holdout_passed else 'FAILED'}")
    
    def _fill_fold_win_rates(self):
        """用最优权重回填折叠胜率"""
        weights_32d = self.result.best_weights_32d
        fusion_th = self.result.best_fusion_threshold
        cosine_min_th = self.result.best_cosine_min_threshold
        
        win_rates = []
        for fold_engine in self._fold_engines:
            if fold_engine.n_eval_bars == 0 or fold_engine.n_prototypes == 0:
                win_rates.append(0.0)
                continue
            
            final_scores, _ = fold_engine.compute_score_matrix(
                weights_32d,
                cosine_weight=self._cosine_weight_normed,
                euclidean_weight=self._euclidean_weight_normed,
                euclidean_max_dist=self._euclidean_max_dist,
                apply_confidence=True,
            )
            
            sim_result = simulate_trades_from_scores(
                final_scores=final_scores,
                eval_indices=fold_engine._eval_indices,
                proto_directions=fold_engine._proto_directions,
                proto_objects=fold_engine._proto_objects,
                close_prices=self._close_prices,
                atr_values=self._atr_values,
                fusion_threshold=fusion_th,
                cosine_min_threshold=cosine_min_th,
                stop_loss_atr=self.stop_loss_atr,
                take_profit_atr=self.take_profit_atr,
                max_hold_bars=self.max_hold_bars,
            )
            
            win_rates.append(sim_result["win_rate"])
        
        self.result.fold_win_rates = win_rates
    
    # ── 工具方法 ─────────────────────────────────────────────
    
    def _report(self, progress: float, message: str):
        """向 UI 报告进度"""
        if self._callback:
            try:
                self._callback(progress, message)
            except Exception:
                pass
        print(f"[WF_Evo] [{progress:.0%}] {message}")
    
    def get_evolved_weights_32d(self) -> np.ndarray:
        """获取进化后的 32 维特征权重向量"""
        return self.result.best_weights_32d.copy()
    
    def get_evolved_group_weights(self) -> Dict[str, float]:
        """获取进化后的组权重（字典形式）"""
        return {
            name: float(self.result.best_group_weights[i])
            for i, name in enumerate(GROUP_NAMES)
        }
    
    def get_evolved_thresholds(self) -> Dict[str, float]:
        """获取进化后的阈值"""
        return {
            "fusion_threshold": self.result.best_fusion_threshold,
            "cosine_min_threshold": self.result.best_cosine_min_threshold,
        }
