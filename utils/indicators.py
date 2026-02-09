"""
R3000 技术指标计算模块
参考 TR2000 的 technical_indicators.py，提供基础技术指标计算
"""
import pandas as pd
import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_CONFIG


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        
    Returns:
        添加了所有技术指标的 DataFrame
    """
    df = df.copy()
    
    # 确保数据类型正确
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # === 移动平均线 ===
    for period in FEATURE_CONFIG["MA_PERIODS"]:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # === EMA 指数移动平均 ===
    for period in FEATURE_CONFIG["EMA_PERIODS"]:
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # === MACD ===
    fast = FEATURE_CONFIG["MACD_FAST"]
    slow = FEATURE_CONFIG["MACD_SLOW"]
    signal = FEATURE_CONFIG["MACD_SIGNAL"]
    df['macd'] = df[f'ema{fast}'] - df[f'ema{slow}']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # === RSI ===
    df['rsi'] = calculate_rsi(df['close'], period=FEATURE_CONFIG["RSI_PERIOD"])
    df['rsi_fast'] = calculate_rsi(df['close'], period=FEATURE_CONFIG["RSI_FAST_PERIOD"])
    
    # === KDJ ===
    df['k'], df['d'], df['j'] = calculate_kdj(df, period=FEATURE_CONFIG["KDJ_PERIOD"])
    
    # === 布林带 ===
    boll_period = FEATURE_CONFIG["BOLL_PERIOD"]
    boll_std = FEATURE_CONFIG["BOLL_STD"]
    df['boll_mid'] = df['close'].rolling(window=boll_period).mean()
    df['boll_std'] = df['close'].rolling(window=boll_period).std()
    df['boll_upper'] = df['boll_mid'] + boll_std * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - boll_std * df['boll_std']
    df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / (df['boll_mid'] + 1e-9)
    df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-9)
    
    # === 成交量指标 ===
    for period in FEATURE_CONFIG["VOLUME_MA_PERIODS"]:
        df[f'volume_ma{period}'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1e-9)
    
    # === ATR (平均真实波幅) ===
    df['atr'] = calculate_atr(df, period=FEATURE_CONFIG["ATR_PERIOD"])
    
    # === 动量指标 ===
    roc_period = FEATURE_CONFIG["ROC_PERIOD"]
    df['momentum'] = df['close'] - df['close'].shift(roc_period)
    df['roc'] = (df['close'] - df['close'].shift(roc_period)) / (df['close'].shift(roc_period) + 1e-9) * 100
    
    # === ADX 趋势强度 ===
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, period=FEATURE_CONFIG["ADX_PERIOD"])
    
    # === 价格变化 ===
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    # === 振幅 ===
    df['amplitude'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-9) * 100
    
    # === K线形态特征 ===
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_ratio'] = abs(df['body']) / (df['high'] - df['low'] + 1e-9)
    
    # === 趋势强度 ===
    df['ma5_slope'] = (df['ma5'] - df['ma5'].shift(5)) / (df['ma5'].shift(5) + 1e-9) * 100
    df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(5)) / (df['ma20'].shift(5) + 1e-9) * 100
    
    # === 相对位置 ===
    df['close_vs_ma5'] = (df['close'] - df['ma5']) / (df['ma5'] + 1e-9) * 100
    df['close_vs_ma20'] = (df['close'] - df['ma20']) / (df['ma20'] + 1e-9) * 100
    
    # === OBV (能量潮) ===
    df['obv'] = calculate_obv(df)
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    df['obv_slope'] = (df['obv'] - df['obv'].shift(5)) / (abs(df['obv'].shift(5)) + 1e-9)
    
    # === VWAP (成交量加权平均价) ===
    df['vwap'] = calculate_vwap(df)
    df['close_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9) * 100
    
    # === 微结构指标 ===
    df = calculate_microstructure(df)
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_kdj(df: pd.DataFrame, period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算 KDJ 指标"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100
    
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 ATR (平均真实波幅)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算 ADX 指标"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 计算方向运动
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # ATR
    atr = calculate_atr(df, period)
    
    # 平滑后的方向运动
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-9))
    
    # DX 和 ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """计算 OBV (能量潮)"""
    close = df['close']
    volume = df['volume']
    
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()
    
    return obv


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """计算 VWAP (成交量加权平均价)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    return vwap


def calculate_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场微结构指标
    """
    df = df.copy()
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    body = close - open_
    body_abs = body.abs()
    bar_range = high - low + 1e-9
    
    # 1. 吞没形态 (Engulfing Pattern) [-1, 1]
    prev_body = body.shift(1)
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    bullish_engulf = (body > 0) & (prev_body < 0) & (close >= prev_open) & (open_ <= prev_close) & (body_abs > prev_body.abs() * 1.1)
    bearish_engulf = (body < 0) & (prev_body > 0) & (close <= prev_open) & (open_ >= prev_close) & (body_abs > prev_body.abs() * 1.1)
    df['engulfing'] = np.where(bullish_engulf, 1.0, np.where(bearish_engulf, -1.0, 0.0))
    
    # 2. Pin Bar (针形反转) [-1, 1]
    upper_shadow = df['upper_shadow']
    lower_shadow = df['lower_shadow']
    bullish_pin = (lower_shadow > 2 * body_abs) & (lower_shadow > 2 * upper_shadow) & (body >= 0)
    bearish_pin = (upper_shadow > 2 * body_abs) & (upper_shadow > 2 * lower_shadow) & (body <= 0)
    df['pin_bar'] = np.where(bullish_pin, 1.0, np.where(bearish_pin, -1.0, 0.0))
    
    # 3. 内包形态 (Inside Bar)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    df['inside_bar'] = ((high <= prev_high) & (low >= prev_low)).astype(float)
    
    # 4. 趋势连续性 (连续同向K线数)
    df['trend_count'] = calculate_trend_count(body)
    
    # 5. 成交量异常 (Volume Spike)
    volume_ma = volume.rolling(window=20).mean()
    df['volume_spike'] = (volume / (volume_ma + 1e-9)) > 2.0
    df['volume_spike'] = df['volume_spike'].astype(float)
    
    # 6. 波动率收缩 (Volatility Squeeze)
    atr = df.get('atr', calculate_atr(df))
    atr_ma = atr.rolling(window=20).mean()
    df['vol_squeeze'] = (atr < atr_ma * 0.7).astype(float)
    
    return df


def calculate_trend_count(body: pd.Series) -> pd.Series:
    """计算连续同向K线数"""
    direction = np.sign(body)
    
    # 计算连续相同方向的数量
    groups = (direction != direction.shift()).cumsum()
    trend_count = direction.groupby(groups).cumcount() + 1
    trend_count = trend_count * direction  # 正数表示上涨趋势，负数表示下跌趋势
    
    return trend_count


def calculate_support_resistance(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    计算支撑阻力位距离
    """
    df = df.copy()
    
    # 简化版：使用滚动最高/最低价
    df['resistance'] = df['high'].rolling(window=lookback).max()
    df['support'] = df['low'].rolling(window=lookback).min()
    
    # 计算距离（归一化）
    price_range = df['resistance'] - df['support'] + 1e-9
    df['dist_to_resistance'] = (df['resistance'] - df['close']) / price_range
    df['dist_to_support'] = (df['close'] - df['support']) / price_range
    
    return df


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
    
    # 计算所有指标
    result = calculate_all_indicators(test_df)
    
    print(f"原始列数: {len(test_df.columns)}")
    print(f"计算后列数: {len(result.columns)}")
    print(f"\n新增指标:\n{[c for c in result.columns if c not in test_df.columns]}")
