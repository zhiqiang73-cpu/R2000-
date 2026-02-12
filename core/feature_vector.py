"""
R3000 三维特征向量引擎
将多维技术指标聚合为三根语义轴 (A, B, C)，构成交易向量空间

A轴 - 即时信号 (Instant): 当前K线所有指标的综合瞬时状态
B轴 - 动量变化 (Momentum): 各指标的变化速率，描述市场的动态加速/减速
C轴 - 结构位置 (Structure): 价格相对于摆动结构的空间位置

设计原则：
  1. 不归一化 - 保留原始精度（3位小数）
  2. 不降维(PCA) - 用加权求和做语义聚合，权重由GA优化
  3. 每个市场状态独立 - 由外部按市场状态分类后分别存储
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_CONFIG


# ── 子特征定义 ─────────────────────────────────────────
# 每根轴的子特征名称，严格对应计算顺序

LAYER_A_FEATURES = [
    "rsi_14",             # RSI(14) 原始值 0-100
    "rsi_6",              # RSI(6) 快速 0-100
    "macd_hist",          # MACD柱状图 (原始值)
    "kdj_k",              # KDJ K值 0-100
    "kdj_j",              # KDJ J值 (可超出0-100)
    "roc",                # ROC 动量 (百分比)
    "atr_ratio",          # ATR / 价格 (比率)
    "boll_width",         # 布林带宽度 (比率)
    "boll_position",      # 价格在布林带中的位置 0-1
    "upper_shadow_ratio", # 上影线占比
    "lower_shadow_ratio", # 下影线占比
    "ema12_dev",          # 价格偏离EMA12 (百分比)
    "ema26_dev",          # 价格偏离EMA26 (百分比)
    "adx",                # ADX 趋势强度 0-100
    "volume_ratio",       # 量比 (当前量/均量)
    "obv_slope",          # OBV斜率
]

LAYER_B_FEATURES = [
    "delta_rsi_3",        # RSI 3根K线变化量
    "delta_rsi_5",        # RSI 5根K线变化量
    "delta_macd_3",       # MACD柱 3根K线变化量
    "delta_macd_5",       # MACD柱 5根K线变化量
    "delta_atr_rate",     # ATR变化率 (ATR/ATR_5ago)
    "delta_boll_width",   # 布林带宽度变化
    "delta_ema12_slope",  # EMA12斜率变化
    "delta_ema26_slope",  # EMA26斜率变化
    "delta_volume_ratio", # 量比变化
    "delta_adx",          # ADX 变化量
]

LAYER_C_FEATURES = [
    "price_in_range",     # 价格在近期高低区间的位置 0-1
    "dist_to_high_atr",   # 距近期高点 (ATR单位)
    "dist_to_low_atr",    # 距近期低点 (ATR单位)
    "up_down_amp_ratio",  # 近期上涨/下跌振幅比
    "price_vs_20high",    # 价格相对20根高点位置
    "price_vs_20low",     # 价格相对20根低点位置
]

N_A = len(LAYER_A_FEATURES)  # 16
N_B = len(LAYER_B_FEATURES)  # 10
N_C = len(LAYER_C_FEATURES)  # 6


@dataclass
class AxisWeights:
    """三轴权重，初始等权，后续GA优化"""
    w_a: np.ndarray = field(default_factory=lambda: np.ones(N_A) / N_A)
    w_b: np.ndarray = field(default_factory=lambda: np.ones(N_B) / N_B)
    w_c: np.ndarray = field(default_factory=lambda: np.ones(N_C) / N_C)

    def to_flat(self) -> np.ndarray:
        """展平为一维数组（供GA搜索）"""
        return np.concatenate([self.w_a, self.w_b, self.w_c])

    @classmethod
    def from_flat(cls, flat: np.ndarray) -> 'AxisWeights':
        """从一维数组还原"""
        w = cls()
        w.w_a = flat[:N_A].copy()
        w.w_b = flat[N_A:N_A + N_B].copy()
        w.w_c = flat[N_A + N_B:N_A + N_B + N_C].copy()
        return w

    @property
    def total_dims(self) -> int:
        return N_A + N_B + N_C


class FeatureVectorEngine:
    """
    三维特征向量引擎

    使用方式：
        engine = FeatureVectorEngine(weights)
        engine.precompute(df)  # 预计算所有K线的子特征（一次性）
        abc = engine.get_abc(idx)  # 获取某根K线的 (A, B, C) 坐标
        raw = engine.get_raw(idx)  # 获取某根K线的全部子特征原始值
    """

    def __init__(self, weights: AxisWeights = None):
        self.weights = weights or AxisWeights()

        # 预计算缓存
        self._raw_a: Optional[np.ndarray] = None  # (n, N_A)
        self._raw_b: Optional[np.ndarray] = None  # (n, N_B)
        self._raw_c: Optional[np.ndarray] = None  # (n, N_C)
        self._n: int = 0
        self._precomputed: bool = False

    def set_weights(self, weights: AxisWeights):
        """设置新权重（GA优化后调用）"""
        self.weights = weights

    # ── 预计算 ─────────────────────────────────────────
    # ── 预计算 ─────────────────────────────────────────
    def precompute(self, df: pd.DataFrame):
        """
        预计算所有K线的原始子特征值（不含加权）
        要求 df 已经调用过 calculate_all_indicators()
        """
        n = len(df)
        self._n = n
        self._raw_a = self._compute_layer_a(df)
        self._raw_b = self._compute_layer_b(df)
        self._raw_c = self._compute_layer_c(df)
        
        # 【性能优化】预先拼接完整特征矩阵，避免推理时的高频 hstack 拷贝
        self._full_matrix = np.hstack([self._raw_a, self._raw_b, self._raw_c])
        
        self._precomputed = True
        # 【优化】减少日志输出，避免控制台噪音
        # print(f"[FeatureVector] 预计算完成: {n} 根K线, "
        #       f"A={N_A}维, B={N_B}维, C={N_C}维, 总={N_A+N_B+N_C}维")

    # ── 获取坐标 ───────────────────────────────────────
    def get_abc(self, idx: int) -> Tuple[float, float, float]:
        """
        获取指定K线的 (A, B, C) 三维坐标（加权求和）

        Returns:
            (A, B, C) 保留3位小数
        """
        if not self._precomputed or idx < 0 or idx >= self._n:
            return (0.0, 0.0, 0.0)

        # 直接使用全量矩阵切片，减少属性访问开销
        row = self._full_matrix[idx]
        a_val = round(float(np.dot(row[:N_A], self.weights.w_a)), 3)
        b_val = round(float(np.dot(row[N_A:N_A+N_B], self.weights.w_b)), 3)
        c_val = round(float(np.dot(row[N_A+N_B:], self.weights.w_c)), 3)
        return (a_val, b_val, c_val)

    def get_abc_batch(self, indices: List[int]) -> np.ndarray:
        """
        批量获取多根K线的 (A, B, C) 坐标

        Returns:
            np.ndarray shape (len(indices), 3)
        """
        if not self._precomputed:
            return np.zeros((len(indices), 3))

        idx_arr = np.array(indices, dtype=int)
        valid = (idx_arr >= 0) & (idx_arr < self._n)
        result = np.zeros((len(indices), 3))

        if valid.any():
            vi = idx_arr[valid]
            # 批量矩阵乘法
            result[valid, 0] = np.round(self._raw_a[vi] @ self.weights.w_a, 3)
            result[valid, 1] = np.round(self._raw_b[vi] @ self.weights.w_b, 3)
            result[valid, 2] = np.round(self._raw_c[vi] @ self.weights.w_c, 3)

        return result

    def get_raw(self, idx: int) -> Dict[str, np.ndarray]:
        """获取某根K线的全部子特征原始值"""
        if not self._precomputed or idx < 0 or idx >= self._n:
            return {"a": np.zeros(N_A), "b": np.zeros(N_B), "c": np.zeros(N_C)}
        
        row = self._full_matrix[idx]
        return {
            "a": row[:N_A].copy(),
            "b": row[N_A:N_A+N_B].copy(),
            "c": row[N_A+N_B:].copy(),
        }

    def get_raw_matrix(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        获取 [start_idx, end_idx) 区间的完整32维原始特征矩阵

        【性能关键项】直接返回预计算矩阵的 View (切片)，彻底消除内存分配和数据拷贝。
        用于轨迹匹配：直接拼接 raw_a(16) + raw_b(10) + raw_c(6) = 32维
        不做加权压缩，保留全部信息

        Args:
            start_idx: 起始索引（包含）
            end_idx: 结束索引（不包含）

        Returns:
            np.ndarray: shape (end_idx - start_idx, 32)
        """
        if not self._precomputed:
            return np.zeros((max(0, end_idx - start_idx), N_A + N_B + N_C))

        # 边界裁剪
        start_idx = max(0, start_idx)
        end_idx = min(self._n, end_idx)
        if start_idx >= end_idx:
            return np.zeros((0, N_A + N_B + N_C))

        # 直接切片返回 View
        return self._full_matrix[start_idx:end_idx]

    def get_full_raw_matrix(self) -> np.ndarray:
        """获取全部K线的32维原始特征矩阵"""
        if not self._precomputed:
            return np.zeros((0, N_A + N_B + N_C))
        return self._full_matrix

    # ── Layer A: 即时信号 ──────────────────────────────
    def _compute_layer_a(self, df: pd.DataFrame) -> np.ndarray:
        """计算 Layer A 所有子特征，保留原始精度"""
        n = len(df)
        raw = np.zeros((n, N_A))

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        bar_range = high - low + 1e-9

        # 0: RSI(14)
        if 'rsi' in df.columns:
            raw[:, 0] = np.round(df['rsi'].values, 3)

        # 1: RSI(6)
        if 'rsi_fast' in df.columns:
            raw[:, 1] = np.round(df['rsi_fast'].values, 3)

        # 2: MACD histogram
        if 'macd_hist' in df.columns:
            raw[:, 2] = np.round(df['macd_hist'].values, 3)

        # 3: KDJ K
        if 'k' in df.columns:
            raw[:, 3] = np.round(df['k'].values, 3)

        # 4: KDJ J
        if 'j' in df.columns:
            raw[:, 4] = np.round(df['j'].values, 3)

        # 5: ROC
        if 'roc' in df.columns:
            raw[:, 5] = np.round(df['roc'].values, 3)

        # 6: ATR ratio (ATR / close)
        if 'atr' in df.columns:
            raw[:, 6] = np.round(df['atr'].values / (close + 1e-9), 6)

        # 7: Bollinger bandwidth
        if 'boll_width' in df.columns:
            raw[:, 7] = np.round(df['boll_width'].values, 6)

        # 8: Bollinger position (0~1)
        if 'boll_position' in df.columns:
            raw[:, 8] = np.round(df['boll_position'].values, 3)

        # 9: Upper shadow ratio
        if 'upper_shadow' in df.columns:
            raw[:, 9] = np.round(df['upper_shadow'].values / bar_range, 3)

        # 10: Lower shadow ratio
        if 'lower_shadow' in df.columns:
            raw[:, 10] = np.round(df['lower_shadow'].values / bar_range, 3)

        # 11: EMA12 deviation (%)
        if 'ema12' in df.columns:
            raw[:, 11] = np.round((close - df['ema12'].values) / (df['ema12'].values + 1e-9) * 100, 3)

        # 12: EMA26 deviation (%)
        if 'ema26' in df.columns:
            raw[:, 12] = np.round((close - df['ema26'].values) / (df['ema26'].values + 1e-9) * 100, 3)

        # 13: ADX
        if 'adx' in df.columns:
            raw[:, 13] = np.round(df['adx'].values, 3)

        # 14: Volume ratio
        if 'volume_ratio' in df.columns:
            raw[:, 14] = np.round(df['volume_ratio'].values, 3)

        # 15: OBV slope
        if 'obv_slope' in df.columns:
            raw[:, 15] = np.round(df['obv_slope'].values, 3)

        # NaN → 0
        np.nan_to_num(raw, copy=False, nan=0.0)
        return raw

    # ── Layer B: 变化率 ────────────────────────────────
    def _compute_layer_b(self, df: pd.DataFrame) -> np.ndarray:
        """计算 Layer B 所有子特征"""
        n = len(df)
        raw = np.zeros((n, N_B))

        # 0: delta RSI 3
        if 'rsi' in df.columns:
            rsi = df['rsi'].values
            raw[3:, 0] = np.round(rsi[3:] - rsi[:-3], 3)

        # 1: delta RSI 5
        if 'rsi' in df.columns:
            rsi = df['rsi'].values
            raw[5:, 1] = np.round(rsi[5:] - rsi[:-5], 3)

        # 2: delta MACD hist 3
        if 'macd_hist' in df.columns:
            mh = df['macd_hist'].values
            raw[3:, 2] = np.round(mh[3:] - mh[:-3], 3)

        # 3: delta MACD hist 5
        if 'macd_hist' in df.columns:
            mh = df['macd_hist'].values
            raw[5:, 3] = np.round(mh[5:] - mh[:-5], 3)

        # 4: ATR change rate (ATR_now / ATR_5ago)
        if 'atr' in df.columns:
            atr = df['atr'].values
            raw[5:, 4] = np.round(atr[5:] / (atr[:-5] + 1e-9), 3)

        # 5: Bollinger width change
        if 'boll_width' in df.columns:
            bw = df['boll_width'].values
            raw[3:, 5] = np.round(bw[3:] - bw[:-3], 6)

        # 6: EMA12 slope change
        if 'ema12' in df.columns:
            ema = df['ema12'].values
            slope = np.zeros(n)
            slope[1:] = (ema[1:] - ema[:-1]) / (ema[:-1] + 1e-9) * 100
            raw[3:, 6] = np.round(slope[3:] - slope[:-3], 3)

        # 7: EMA26 slope change
        if 'ema26' in df.columns:
            ema = df['ema26'].values
            slope = np.zeros(n)
            slope[1:] = (ema[1:] - ema[:-1]) / (ema[:-1] + 1e-9) * 100
            raw[3:, 7] = np.round(slope[3:] - slope[:-3], 3)

        # 8: Volume ratio change
        if 'volume_ratio' in df.columns:
            vr = df['volume_ratio'].values
            raw[3:, 8] = np.round(vr[3:] - vr[:-3], 3)

        # 9: ADX change
        if 'adx' in df.columns:
            adx = df['adx'].values
            raw[5:, 9] = np.round(adx[5:] - adx[:-5], 3)

        np.nan_to_num(raw, copy=False, nan=0.0)
        return raw

    # ── Layer C: 空间位置 ──────────────────────────────
    def _compute_layer_c(self, df: pd.DataFrame) -> np.ndarray:
        """计算 Layer C 所有子特征"""
        n = len(df)
        raw = np.zeros((n, N_C))
        lookback = 20

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 滚动高低点
        high_roll = pd.Series(high).rolling(lookback, min_periods=1).max().values
        low_roll = pd.Series(low).rolling(lookback, min_periods=1).min().values
        range_roll = high_roll - low_roll + 1e-9

        # ATR for distance normalization
        atr = df['atr'].values if 'atr' in df.columns else np.ones(n)
        atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)

        # 0: Price position in recent range (0~1)
        raw[:, 0] = np.round((close - low_roll) / range_roll, 3)

        # 1: Distance to recent high (ATR units, positive)
        raw[:, 1] = np.round((high_roll - close) / atr, 3)

        # 2: Distance to recent low (ATR units, positive)
        raw[:, 2] = np.round((close - low_roll) / atr, 3)

        # 3: Up/Down amplitude ratio (近期上涨幅度 / 近期下跌幅度) - 向量化
        price_changes = np.zeros(n)
        price_changes[1:] = close[1:] - close[:-1]
        pos_changes = np.where(price_changes > 0, price_changes, 0.0)
        neg_changes = np.where(price_changes < 0, -price_changes, 0.0)
        sum_ups = pd.Series(pos_changes).rolling(lookback, min_periods=1).sum().values
        sum_downs = pd.Series(neg_changes).rolling(lookback, min_periods=1).sum().values
        raw[:, 3] = np.round(sum_ups / (sum_downs + 1e-9), 3)

        # 4: Price vs 20-bar high (how close to the high, 0~1)
        raw[:, 4] = np.round(1.0 - (high_roll - close) / range_roll, 3)

        # 5: Price vs 20-bar low (how close to the low, 0~1)
        raw[:, 5] = np.round(1.0 - (close - low_roll) / range_roll, 3)

        np.nan_to_num(raw, copy=False, nan=0.0)
        return raw

    # ── 元数据 ─────────────────────────────────────────
    @staticmethod
    def get_layer_names() -> Dict[str, List[str]]:
        """获取三层子特征名称"""
        return {
            "A": list(LAYER_A_FEATURES),
            "B": list(LAYER_B_FEATURES),
            "C": list(LAYER_C_FEATURES),
        }

    @property
    def is_ready(self) -> bool:
        return self._precomputed
