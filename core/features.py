"""
R3000 特征提取模块
参考 TR2000 的 44 维特征空间，扩展到 52 维
包含趋势/动量/波动/成交量/形态/多时间框架特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_CONFIG, MTF_CONFIG
from utils.indicators import calculate_all_indicators


class FeatureExtractor:
    """
    特征提取器
    
    从 K 线数据中提取 52 维非重叠技术特征：
    - 趋势类 (8 维)
    - 动量类 (10 维)
    - 波动类 (8 维)
    - 成交量类 (8 维)
    - 价格形态类 (10 维)
    - 多时间框架类 (8 维)
    """
    
    def __init__(self):
        self.num_features = FEATURE_CONFIG["NUM_FEATURES"]
        self._feature_names = self._init_feature_names()
        
        # 特征矩阵缓存
        self.feature_matrix: Optional[np.ndarray] = None
        self._df_id: Optional[int] = None
        
    def _init_feature_names(self) -> List[str]:
        """初始化特征名称列表"""
        return [
            # 趋势类 (8 维)
            "trend_ma_cross_5_10",       # MA5/MA10 交叉
            "trend_ma_cross_10_20",      # MA10/MA20 交叉
            "trend_ma_cross_20_60",      # MA20/MA60 交叉
            "trend_ema_slope_12",        # EMA12 斜率
            "trend_ema_slope_26",        # EMA26 斜率
            "trend_price_vs_ma5",        # 价格相对 MA5 位置
            "trend_price_vs_ma20",       # 价格相对 MA20 位置
            "trend_adx",                 # ADX 趋势强度
            
            # 动量类 (10 维)
            "mom_rsi",                   # RSI(14)
            "mom_rsi_fast",              # RSI(6)
            "mom_rsi_divergence",        # RSI 背离
            "mom_macd",                  # MACD
            "mom_macd_hist",             # MACD 柱状图
            "mom_macd_cross",            # MACD 交叉信号
            "mom_kdj_k",                 # KDJ K值
            "mom_kdj_j",                 # KDJ J值
            "mom_kdj_cross",             # KDJ 交叉信号
            "mom_roc",                   # ROC 动量
            
            # 波动类 (8 维)
            "vol_atr_ratio",             # ATR 比率
            "vol_boll_width",            # 布林带宽度
            "vol_boll_position",         # 布林带位置
            "vol_amplitude",             # 振幅
            "vol_expansion",             # 波动扩张
            "vol_squeeze",               # 波动收缩
            "vol_atr_percentile",        # ATR 分位数
            "vol_range_ratio",           # 波幅比率
            
            # 成交量类 (8 维)
            "volm_ratio",                # 成交量比率
            "volm_trend",                # 成交量趋势
            "volm_obv_slope",            # OBV 斜率
            "volm_price_confirm",        # 量价确认
            "volm_climax",               # 成交量高潮
            "volm_dry_up",               # 成交量萎缩
            "volm_vwap_position",        # VWAP 位置
            "volm_accumulation",         # 累积/派发
            
            # 价格形态类 (10 维)
            "pa_body_ratio",             # 实体比率
            "pa_shadow_ratio",           # 影线比率
            "pa_trend_count",            # 趋势连续数
            "pa_engulfing",              # 吞没形态
            "pa_pin_bar",                # Pin Bar
            "pa_inside_bar",             # 内包形态
            "pa_dist_support",           # 距离支撑位
            "pa_dist_resistance",        # 距离阻力位
            "pa_higher_high",            # 更高高点
            "pa_lower_low",              # 更低低点
            
            # 多时间框架类 (8 维)
            "mtf_1h_trend",              # 1H 趋势方向
            "mtf_1h_rsi",                # 1H RSI
            "mtf_1h_ema_slope",          # 1H EMA 斜率
            "mtf_1h_volatility",         # 1H 波动率
            "mtf_4h_trend",              # 4H 趋势方向
            "mtf_4h_rsi",                # 4H RSI
            "mtf_trend_align",           # 多时间框架趋势对齐
            "mtf_momentum_align",        # 多时间框架动量对齐
        ]
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self._feature_names.copy()
    
    def extract_all_features(self, df: pd.DataFrame, 
                             mtf_data: Dict[str, pd.DataFrame] = None) -> np.ndarray:
        """
        提取所有数据点的特征矩阵
        
        Args:
            df: 1分钟 K 线数据（已计算指标）
            mtf_data: 多时间框架数据字典
            
        Returns:
            特征矩阵 (n_samples, n_features)
        """
        # 检查缓存
        df_id = id(df)
        if df_id == self._df_id and self.feature_matrix is not None:
            return self.feature_matrix
        
        # 确保指标已计算
        if 'rsi' not in df.columns:
            df = calculate_all_indicators(df)
        
        n = len(df)
        features = np.zeros((n, self.num_features))
        
        # 批量提取特征
        features[:, :8] = self._extract_trend_features(df)
        features[:, 8:18] = self._extract_momentum_features(df)
        features[:, 18:26] = self._extract_volatility_features(df)
        features[:, 26:34] = self._extract_volume_features(df)
        features[:, 34:44] = self._extract_price_action_features(df)
        features[:, 44:52] = self._extract_mtf_features(df, mtf_data)
        
        # 归一化到 [-1, 1]
        features = np.clip(features, 
                          FEATURE_CONFIG["FEATURE_CLIP_MIN"],
                          FEATURE_CONFIG["FEATURE_CLIP_MAX"])
        
        # 处理 NaN
        features = np.nan_to_num(features, nan=0.0)
        
        # 缓存
        self.feature_matrix = features
        self._df_id = df_id
        
        return features
    
    def extract_at_index(self, df: pd.DataFrame, idx: int,
                         mtf_data: Dict[str, pd.DataFrame] = None) -> np.ndarray:
        """
        提取单个数据点的特征向量
        
        Args:
            df: K 线数据
            idx: 索引
            mtf_data: 多时间框架数据
            
        Returns:
            特征向量 (n_features,)
        """
        if self.feature_matrix is None or id(df) != self._df_id:
            self.extract_all_features(df, mtf_data)
        
        if 0 <= idx < len(self.feature_matrix):
            return self.feature_matrix[idx]
        return np.zeros(self.num_features)
    
    def extract_at_labels(self, df: pd.DataFrame, labels: pd.Series,
                          mtf_data: Dict[str, pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        在标注点提取特征
        
        Args:
            df: K 线数据
            labels: 标注序列
            mtf_data: 多时间框架数据
            
        Returns:
            (features, label_values) - 特征矩阵和对应的标注值
        """
        # 获取非 HOLD 的点
        labeled_mask = labels != 0
        labeled_indices = np.where(labeled_mask)[0]
        
        if len(labeled_indices) == 0:
            return np.array([]), np.array([])
        
        # 提取特征
        all_features = self.extract_all_features(df, mtf_data)
        
        features = all_features[labeled_indices]
        label_values = labels.iloc[labeled_indices].values
        
        return features, label_values
    
    def _extract_trend_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取趋势类特征 (8 维)"""
        n = len(df)
        features = np.zeros((n, 8))
        
        # MA 交叉信号
        if 'ma5' in df.columns and 'ma10' in df.columns:
            ma5 = df['ma5'].values
            ma10 = df['ma10'].values
            features[:, 0] = np.tanh((ma5 - ma10) / (ma10 + 1e-9) * 100)
        
        if 'ma10' in df.columns and 'ma20' in df.columns:
            ma10 = df['ma10'].values
            ma20 = df['ma20'].values
            features[:, 1] = np.tanh((ma10 - ma20) / (ma20 + 1e-9) * 100)
        
        if 'ma20' in df.columns and 'ma60' in df.columns:
            ma20 = df['ma20'].values
            ma60 = df['ma60'].values
            features[:, 2] = np.tanh((ma20 - ma60) / (ma60 + 1e-9) * 100)
        
        # EMA 斜率
        if 'ema12' in df.columns:
            ema12 = df['ema12'].values
            ema12_slope = np.zeros(n)
            ema12_slope[5:] = (ema12[5:] - ema12[:-5]) / (ema12[:-5] + 1e-9) * 100
            features[:, 3] = np.tanh(ema12_slope)
        
        if 'ema26' in df.columns:
            ema26 = df['ema26'].values
            ema26_slope = np.zeros(n)
            ema26_slope[5:] = (ema26[5:] - ema26[:-5]) / (ema26[:-5] + 1e-9) * 100
            features[:, 4] = np.tanh(ema26_slope)
        
        # 价格相对位置
        if 'close_vs_ma5' in df.columns:
            features[:, 5] = np.tanh(df['close_vs_ma5'].values / 5)
        
        if 'close_vs_ma20' in df.columns:
            features[:, 6] = np.tanh(df['close_vs_ma20'].values / 5)
        
        # ADX 趋势强度
        if 'adx' in df.columns:
            features[:, 7] = (df['adx'].values - 25) / 25  # 25 作为中性点
        
        return features
    
    def _extract_momentum_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取动量类特征 (10 维)"""
        n = len(df)
        features = np.zeros((n, 10))
        
        # RSI
        if 'rsi' in df.columns:
            features[:, 0] = (df['rsi'].values - 50) / 50
        
        if 'rsi_fast' in df.columns:
            features[:, 1] = (df['rsi_fast'].values - 50) / 50
        
        # RSI 背离（简化版：RSI 斜率 vs 价格斜率）
        if 'rsi' in df.columns and 'close' in df.columns:
            rsi = df['rsi'].values
            close = df['close'].values
            rsi_slope = np.zeros(n)
            price_slope = np.zeros(n)
            rsi_slope[10:] = rsi[10:] - rsi[:-10]
            price_slope[10:] = (close[10:] - close[:-10]) / (close[:-10] + 1e-9) * 100
            divergence = np.sign(rsi_slope) != np.sign(price_slope)
            features[:, 2] = divergence.astype(float) * np.sign(rsi_slope)
        
        # MACD
        if 'macd' in df.columns:
            macd = df['macd'].values
            close = df['close'].values
            features[:, 3] = np.tanh(macd / (close * 0.001 + 1e-9))
        
        if 'macd_hist' in df.columns:
            macd_hist = df['macd_hist'].values
            close = df['close'].values
            features[:, 4] = np.tanh(macd_hist / (close * 0.0005 + 1e-9))
        
        # MACD 交叉
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].values
            signal = df['macd_signal'].values
            cross = np.zeros(n)
            cross[1:] = np.sign(macd[1:] - signal[1:]) - np.sign(macd[:-1] - signal[:-1])
            features[:, 5] = np.clip(cross, -1, 1)
        
        # KDJ
        if 'k' in df.columns:
            features[:, 6] = (df['k'].values - 50) / 50
        
        if 'j' in df.columns:
            features[:, 7] = np.clip((df['j'].values - 50) / 50, -1, 1)
        
        # KDJ 交叉
        if 'k' in df.columns and 'd' in df.columns:
            k = df['k'].values
            d = df['d'].values
            cross = np.zeros(n)
            cross[1:] = np.sign(k[1:] - d[1:]) - np.sign(k[:-1] - d[:-1])
            features[:, 8] = np.clip(cross, -1, 1)
        
        # ROC
        if 'roc' in df.columns:
            features[:, 9] = np.tanh(df['roc'].values / 5)
        
        return features
    
    def _extract_volatility_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取波动类特征 (8 维)"""
        n = len(df)
        features = np.zeros((n, 8))
        
        # ATR 比率
        if 'atr' in df.columns and 'close' in df.columns:
            atr_pct = df['atr'].values / df['close'].values * 100
            atr_ma = pd.Series(df['atr'].values).rolling(20).mean().values
            features[:, 0] = np.tanh((df['atr'].values - atr_ma) / (atr_ma + 1e-9))
        
        # 布林带宽度
        if 'boll_width' in df.columns:
            boll_width_ma = pd.Series(df['boll_width'].values).rolling(20).mean().values
            features[:, 1] = np.tanh((df['boll_width'].values - boll_width_ma) / (boll_width_ma + 1e-9))
        
        # 布林带位置
        if 'boll_position' in df.columns:
            features[:, 2] = df['boll_position'].values * 2 - 1  # [0,1] -> [-1,1]
        
        # 振幅
        if 'amplitude' in df.columns:
            amp_ma = pd.Series(df['amplitude'].values).rolling(20).mean().values
            features[:, 3] = np.tanh((df['amplitude'].values - amp_ma) / (amp_ma + 1e-9))
        
        # 波动扩张
        if 'atr' in df.columns:
            atr = df['atr'].values
            atr_ma = pd.Series(atr).rolling(20).mean().values
            expansion = atr > atr_ma * 1.5
            features[:, 4] = expansion.astype(float)
        
        # 波动收缩
        if 'vol_squeeze' in df.columns:
            features[:, 5] = df['vol_squeeze'].values
        elif 'atr' in df.columns:
            atr = df['atr'].values
            atr_ma = pd.Series(atr).rolling(20).mean().values
            squeeze = atr < atr_ma * 0.7
            features[:, 5] = squeeze.astype(float)
        
        # ATR 分位数
        if 'atr' in df.columns:
            atr = df['atr'].values
            atr_rank = np.zeros(n)
            for i in range(50, n):
                atr_rank[i] = (atr[i] > atr[i-50:i]).sum() / 50
            features[:, 6] = atr_rank * 2 - 1
        
        # 波幅比率
        if 'high' in df.columns and 'low' in df.columns:
            range_ = df['high'].values - df['low'].values
            range_ma = pd.Series(range_).rolling(20).mean().values
            features[:, 7] = np.tanh((range_ - range_ma) / (range_ma + 1e-9))
        
        return features
    
    def _extract_volume_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取成交量类特征 (8 维)"""
        n = len(df)
        features = np.zeros((n, 8))
        
        # 成交量比率
        if 'volume_ratio' in df.columns:
            features[:, 0] = np.tanh((df['volume_ratio'].values - 1) * 2)
        
        # 成交量趋势
        if 'volume' in df.columns:
            vol = df['volume'].values
            vol_ma5 = pd.Series(vol).rolling(5).mean().values
            vol_ma20 = pd.Series(vol).rolling(20).mean().values
            features[:, 1] = np.tanh((vol_ma5 - vol_ma20) / (vol_ma20 + 1e-9))
        
        # OBV 斜率
        if 'obv_slope' in df.columns:
            features[:, 2] = np.tanh(df['obv_slope'].values)
        
        # 量价确认
        if 'volume' in df.columns and 'close' in df.columns:
            vol = df['volume'].values
            close = df['close'].values
            vol_change = np.zeros(n)
            price_change = np.zeros(n)
            vol_change[1:] = vol[1:] - vol[:-1]
            price_change[1:] = close[1:] - close[:-1]
            confirm = np.sign(vol_change) == np.sign(price_change)
            features[:, 3] = confirm.astype(float) * 2 - 1
        
        # 成交量高潮
        if 'volume_spike' in df.columns:
            features[:, 4] = df['volume_spike'].values
        elif 'volume' in df.columns:
            vol = df['volume'].values
            vol_ma = pd.Series(vol).rolling(20).mean().values
            climax = vol > vol_ma * 2.5
            features[:, 4] = climax.astype(float)
        
        # 成交量萎缩
        if 'volume' in df.columns:
            vol = df['volume'].values
            vol_ma = pd.Series(vol).rolling(20).mean().values
            dry_up = vol < vol_ma * 0.5
            features[:, 5] = dry_up.astype(float)
        
        # VWAP 位置
        if 'close_vs_vwap' in df.columns:
            features[:, 6] = np.tanh(df['close_vs_vwap'].values / 2)
        
        # 累积/派发
        if 'close' in df.columns and 'volume' in df.columns:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            vol = df['volume'].values
            
            clv = ((close - low) - (high - close)) / (high - low + 1e-9)
            ad = (clv * vol).cumsum()
            ad_ma = pd.Series(ad).rolling(20).mean().values
            features[:, 7] = np.tanh((ad - ad_ma) / (abs(ad_ma) + 1e-9))
        
        return features
    
    def _extract_price_action_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取价格形态类特征 (10 维)"""
        n = len(df)
        features = np.zeros((n, 10))
        
        # 实体比率
        if 'body_ratio' in df.columns:
            features[:, 0] = df['body_ratio'].values * 2 - 1
        
        # 影线比率
        if 'upper_shadow' in df.columns and 'lower_shadow' in df.columns:
            upper = df['upper_shadow'].values
            lower = df['lower_shadow'].values
            range_ = df['high'].values - df['low'].values + 1e-9
            shadow_ratio = (upper + lower) / range_
            features[:, 1] = shadow_ratio * 2 - 1
        
        # 趋势连续数
        if 'trend_count' in df.columns:
            features[:, 2] = np.tanh(df['trend_count'].values / 5)
        
        # 吞没形态
        if 'engulfing' in df.columns:
            features[:, 3] = df['engulfing'].values
        
        # Pin Bar
        if 'pin_bar' in df.columns:
            features[:, 4] = df['pin_bar'].values
        
        # 内包形态
        if 'inside_bar' in df.columns:
            features[:, 5] = df['inside_bar'].values
        
        # 距离支撑位
        if 'dist_to_support' in df.columns:
            features[:, 6] = 1 - df['dist_to_support'].values * 2
        elif 'low' in df.columns:
            low = df['low'].values
            support = pd.Series(low).rolling(50).min().values
            close = df['close'].values
            range_ = pd.Series(df['high'].values).rolling(50).max().values - support + 1e-9
            features[:, 6] = 1 - (close - support) / range_ * 2
        
        # 距离阻力位
        if 'dist_to_resistance' in df.columns:
            features[:, 7] = df['dist_to_resistance'].values * 2 - 1
        elif 'high' in df.columns:
            high = df['high'].values
            resistance = pd.Series(high).rolling(50).max().values
            close = df['close'].values
            support = pd.Series(df['low'].values).rolling(50).min().values
            range_ = resistance - support + 1e-9
            features[:, 7] = (resistance - close) / range_ * 2 - 1
        
        # 更高高点
        if 'high' in df.columns:
            high = df['high'].values
            prev_high = pd.Series(high).rolling(20).max().shift(1).values
            features[:, 8] = (high > prev_high).astype(float)
        
        # 更低低点
        if 'low' in df.columns:
            low = df['low'].values
            prev_low = pd.Series(low).rolling(20).min().shift(1).values
            features[:, 9] = (low < prev_low).astype(float)
        
        return features
    
    def _extract_mtf_features(self, df: pd.DataFrame, 
                               mtf_data: Dict[str, pd.DataFrame] = None) -> np.ndarray:
        """提取多时间框架特征 (8 维)"""
        n = len(df)
        features = np.zeros((n, 8))
        
        if mtf_data is None:
            return features
        
        # 1H 特征
        if '1h' in mtf_data and mtf_data['1h'] is not None:
            df_1h = mtf_data['1h']
            if 'rsi' not in df_1h.columns:
                df_1h = calculate_all_indicators(df_1h)
            
            # 将 1H 特征映射到 1m 索引
            ratio = 60  # 1H = 60 * 1m
            n_1h = len(df_1h)
            
            # 1H 趋势方向
            if 'ema12' in df_1h.columns and 'ema26' in df_1h.columns:
                trend_1h = np.tanh((df_1h['ema12'].values - df_1h['ema26'].values) / 
                                   (df_1h['ema26'].values + 1e-9) * 100)
                for i in range(n):
                    h_idx = min(i // ratio, n_1h - 1)
                    features[i, 0] = trend_1h[h_idx] if not np.isnan(trend_1h[h_idx]) else 0
            
            # 1H RSI
            if 'rsi' in df_1h.columns:
                rsi_1h = (df_1h['rsi'].values - 50) / 50
                for i in range(n):
                    h_idx = min(i // ratio, n_1h - 1)
                    features[i, 1] = rsi_1h[h_idx] if not np.isnan(rsi_1h[h_idx]) else 0
            
            # 1H EMA 斜率
            if 'ema12' in df_1h.columns:
                ema = df_1h['ema12'].values
                slope = np.zeros(n_1h)
                slope[3:] = (ema[3:] - ema[:-3]) / (ema[:-3] + 1e-9) * 100
                for i in range(n):
                    h_idx = min(i // ratio, n_1h - 1)
                    features[i, 2] = np.tanh(slope[h_idx]) if not np.isnan(slope[h_idx]) else 0
            
            # 1H 波动率
            if 'atr' in df_1h.columns:
                atr = df_1h['atr'].values
                atr_ma = pd.Series(atr).rolling(20).mean().values
                vol_1h = np.tanh((atr - atr_ma) / (atr_ma + 1e-9))
                for i in range(n):
                    h_idx = min(i // ratio, n_1h - 1)
                    features[i, 3] = vol_1h[h_idx] if not np.isnan(vol_1h[h_idx]) else 0
        
        # 4H 特征
        if '4h' in mtf_data and mtf_data['4h'] is not None:
            df_4h = mtf_data['4h']
            if 'rsi' not in df_4h.columns:
                df_4h = calculate_all_indicators(df_4h)
            
            ratio = 240  # 4H = 240 * 1m
            n_4h = len(df_4h)
            
            # 4H 趋势方向
            if 'ema12' in df_4h.columns and 'ema26' in df_4h.columns:
                trend_4h = np.tanh((df_4h['ema12'].values - df_4h['ema26'].values) / 
                                   (df_4h['ema26'].values + 1e-9) * 100)
                for i in range(n):
                    h_idx = min(i // ratio, n_4h - 1)
                    features[i, 4] = trend_4h[h_idx] if not np.isnan(trend_4h[h_idx]) else 0
            
            # 4H RSI
            if 'rsi' in df_4h.columns:
                rsi_4h = (df_4h['rsi'].values - 50) / 50
                for i in range(n):
                    h_idx = min(i // ratio, n_4h - 1)
                    features[i, 5] = rsi_4h[h_idx] if not np.isnan(rsi_4h[h_idx]) else 0
        
        # 多时间框架趋势对齐
        trend_1m = features[:, 0] if np.any(features[:, 0] != 0) else np.zeros(n)
        trend_1h = features[:, 0]
        trend_4h = features[:, 4]
        
        # 如果 1m 没有趋势特征，用自身数据计算
        if 'ema12' in df.columns and 'ema26' in df.columns:
            trend_1m = np.tanh((df['ema12'].values - df['ema26'].values) / 
                              (df['ema26'].values + 1e-9) * 100)
        
        align_score = (np.sign(trend_1m) == np.sign(trend_1h)).astype(float) * 0.5 + \
                      (np.sign(trend_1m) == np.sign(trend_4h)).astype(float) * 0.5
        features[:, 6] = align_score * 2 - 1
        
        # 多时间框架动量对齐
        mom_1m = features[:, 1] if np.any(features[:, 1] != 0) else np.zeros(n)
        mom_1h = features[:, 1]
        mom_4h = features[:, 5]
        
        if 'rsi' in df.columns:
            mom_1m = (df['rsi'].values - 50) / 50
        
        mom_align = (np.sign(mom_1m) == np.sign(mom_1h)).astype(float) * 0.5 + \
                    (np.sign(mom_1m) == np.sign(mom_4h)).astype(float) * 0.5
        features[:, 7] = mom_align * 2 - 1
        
        return features
    
    def reset_cache(self):
        """重置缓存"""
        self.feature_matrix = None
        self._df_id = None


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    n = 1000
    
    test_df = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(n) * 100),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(100, 10000, n),
    })
    test_df['close'] = test_df['open'] + np.random.randn(n) * 50
    test_df['high'] = test_df[['open', 'close']].max(axis=1) + abs(np.random.randn(n) * 30)
    test_df['low'] = test_df[['open', 'close']].min(axis=1) - abs(np.random.randn(n) * 30)
    
    # 计算指标
    test_df = calculate_all_indicators(test_df)
    
    # 提取特征
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(test_df)
    
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征名称: {extractor.get_feature_names()[:5]}...")
    print(f"\n特征统计:")
    print(f"  均值: {features.mean(axis=0)[:5]}")
    print(f"  标准差: {features.std(axis=0)[:5]}")
    print(f"  最小值: {features.min():.4f}")
    print(f"  最大值: {features.max():.4f}")
