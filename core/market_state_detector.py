"""
core/market_state_detector.py
市场状态检测工具（公共模块）

供 signal_analyzer 和 SignalLiveMonitor 共用，基于 ADX 和 MA5 斜率判断市场状态。
"""
from __future__ import annotations


def detect_state(adx: float, ma5_slope: float) -> str:
    """
    根据 ADX 强度和 MA5 斜率判断当前市场状态。

    分类规则：
      多头趋势: ADX > 25 且 ma5_slope > 0
      空头趋势: ADX > 25 且 ma5_slope < 0
      震荡市:   ADX <= 25

    Args:
        adx:       ADX 指标值（来自 calculate_all_indicators 的 'adx' 列）
        ma5_slope: MA5 斜率（来自 calculate_all_indicators 的 'ma5_slope' 列）

    Returns:
        '多头趋势' | '空头趋势' | '震荡市'
    """
    if adx > 25:
        return "多头趋势" if ma5_slope > 0 else "空头趋势"
    return "震荡市"
