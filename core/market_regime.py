"""
R3000 市场状态分类器
基于已确认的历史摆动点（交替高低点序列）来判断市场所处的环境

分类维度：
  1. direction (方向): 最近高低点是整体上移还是下移
  2. strength  (强度): 摆动幅度相对于价格水平的大小
  3. rhythm    (节奏): 摆动周期的长短（辅助参考）

6 种市场状态：
  S1 强多头: 方向明确向上 + 幅度大
  S2 弱多头: 方向向上 + 幅度小
  S3 震荡偏多: 方向微弱向上或不明确、整体偏多
  S4 震荡偏空: 方向微弱向下或不明确、整体偏空
  S5 弱空头: 方向向下 + 幅度小
  S6 强空头: 方向明确向下 + 幅度大
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


# ── 市场状态常量 ──────────────────────────────────────────
class MarketRegime:
    STRONG_BULL = "强多头"
    WEAK_BULL   = "弱多头"
    RANGE_BULL  = "震荡偏多"
    RANGE_BEAR  = "震荡偏空"
    WEAK_BEAR   = "弱空头"
    STRONG_BEAR = "强空头"
    UNKNOWN     = "未知"

    ALL_REGIMES = [STRONG_BULL, WEAK_BULL, RANGE_BULL,
                   RANGE_BEAR, WEAK_BEAR, STRONG_BEAR]

    # 用于 UI 着色
    COLORS = {
        STRONG_BULL: "#00E676",   # 亮绿
        WEAK_BULL:   "#66BB6A",   # 绿
        RANGE_BULL:  "#A5D6A7",   # 浅绿
        RANGE_BEAR:  "#EF9A9A",   # 浅红
        WEAK_BEAR:   "#EF5350",   # 红
        STRONG_BEAR: "#FF1744",   # 亮红
        UNKNOWN:     "#888888",   # 灰
        "多头趋势":   "#00E676",   # 亮绿
        "震荡市":     "#A5D6A7",   # 浅绿 (or neutral color)
        "空头趋势":   "#FF1744",   # 亮红
    }


# ── 分类器 ──────────────────────────────────────────────
class MarketRegimeClassifier:
    """
    基于交替摆动点（alternating_swings）的市场环境分类器

    使用方法：
        classifier = MarketRegimeClassifier(labeler.alternating_swings, config)
        regime = classifier.classify_at(entry_bar_index)
    """

    def __init__(self, alternating_swings: list, config: dict = None):
        """
        Args:
            alternating_swings: 严格交替的 SwingPoint 列表（高低交替）
            config: MARKET_REGIME_CONFIG 字典
        """
        self.swings = sorted(alternating_swings, key=lambda s: s.index)
        cfg = config or {}
        self.dir_strong   = cfg.get("DIR_STRONG_THRESHOLD", 0.008)
        self.str_strong   = cfg.get("STRENGTH_STRONG_THRESHOLD", 0.006)
        self.lookback     = cfg.get("LOOKBACK_SWINGS", 4)
        self.min_swings_required = 3  # 调整为 3 个更合理
        
    # ─── 核心分类 ───────────────────────────────────────
    def classify_at(self, idx: int) -> str:
        """
        对指定 K 线索引处的市场状态进行分类

        Args:
            idx: 当前 K 线索引（只使用 index < idx 的已确认摆动点）

        Returns:
            MarketRegime 常量字符串
        """
        # 取 idx 之前的已确认摆动点
        relevant = [s for s in self.swings if s.index < idx]
        if len(relevant) < self.min_swings_required:
            return MarketRegime.UNKNOWN

        recent = relevant[-self.lookback:] if len(relevant) >= self.lookback else relevant

        # ── 分离高低点 ──
        highs = [s for s in recent if s.is_high]
        lows  = [s for s in recent if not s.is_high]

        if len(highs) < 2 or len(lows) < 2:
            return MarketRegime.UNKNOWN
        
        # ── 1. 方向 (direction) ── （与上帝视角训练完全一致）
        # 高点趋势：最近的高点比早期的高点高 → 向上
        high_trend = (highs[-1].price - highs[0].price) / highs[0].price
        # 低点趋势：最近的低点比早期的低点高 → 向上
        low_trend  = (lows[-1].price - lows[0].price) / lows[0].price
        direction  = (high_trend + low_trend) / 2.0

        # ── 2. 强度 (strength) ──
        # 计算相邻摆动点之间的平均振幅 / 平均价格
        amplitudes = []
        for i in range(1, len(recent)):
            amp = abs(recent[i].price - recent[i - 1].price)
            amplitudes.append(amp)
        avg_amplitude = np.mean(amplitudes) if amplitudes else 0.0
        avg_price     = np.mean([s.price for s in recent])
        strength      = avg_amplitude / avg_price if avg_price > 0 else 0.0

        # ── 3. 分类决策 ──
        return self._decide(direction, strength)

    def _decide(self, direction: float, strength: float) -> str:
        """根据方向和强度决定市场状态"""
        is_strong = strength >= self.str_strong

        if direction > self.dir_strong:
            return MarketRegime.STRONG_BULL if is_strong else MarketRegime.WEAK_BULL
        if direction < -self.dir_strong:
            return MarketRegime.STRONG_BEAR if is_strong else MarketRegime.WEAK_BEAR

        # 中性区间：不再用弱阈值，避免轻微偏差触发趋势
        return MarketRegime.RANGE_BULL if direction >= 0 else MarketRegime.RANGE_BEAR

    # ─── 批量分类 ───────────────────────────────────────
    def classify_trades(self, trades: list) -> Dict[int, str]:
        """
        为交易列表中的每一笔交易分类市场状态

        Args:
            trades: TradeRecord 列表

        Returns:
            {trade_index: regime_string}
        """
        result = {}
        for i, trade in enumerate(trades):
            result[i] = self.classify_at(trade.entry_idx)
        return result

    # ─── 统计摘要 ───────────────────────────────────────
    @staticmethod
    def compute_regime_stats(trades: list, regime_map: Dict[int, str]) -> Dict[str, dict]:
        """
        统计每种市场状态下的交易表现

        Args:
            trades: TradeRecord 列表
            regime_map: {trade_index: regime_string}

        Returns:
            {regime: {count, long, short, wins, losses, win_rate, avg_profit_pct, total_profit}}
        """
        stats = {}
        # 为了兼容新的3态，也初始化它们
        for regime in MarketRegime.ALL_REGIMES + [MarketRegime.UNKNOWN, "多头趋势", "空头趋势", "震荡市"]:
            stats[regime] = {
                "count": 0,
                "long": 0,
                "short": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_profit_pct": 0.0,
                "total_profit": 0.0,
                "profit_pcts": [],
            }

        for i, trade in enumerate(trades):
            regime = regime_map.get(i, MarketRegime.UNKNOWN)
            if regime not in stats:
                stats[regime] = {
                    "count": 0, "long": 0, "short": 0,
                    "wins": 0, "losses": 0, "win_rate": 0.0,
                    "avg_profit_pct": 0.0, "total_profit": 0.0,
                    "profit_pcts": [],
                }

            s = stats[regime]
            s["count"] += 1
            if trade.side == 1:
                s["long"] += 1
            else:
                s["short"] += 1

            if trade.profit > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

            s["total_profit"] += trade.profit
            s["profit_pcts"].append(trade.profit_pct)

        # 计算汇总指标
        for regime, s in stats.items():
            if s["count"] > 0:
                s["win_rate"] = s["wins"] / s["count"]
                s["avg_profit_pct"] = np.mean(s["profit_pcts"]) if s["profit_pcts"] else 0.0
            # 清理临时列表
            del s["profit_pcts"]

        return stats

    @staticmethod
    def compute_direction_regime_stats(trades: list, regime_map: Dict[int, str]) -> Dict:
        """
        返回 {('long'|'short', regime): {count, wins, losses, win_rate,
                                        avg_win_pct, avg_loss_pct,
                                        avg_profit_pct, total_profit}}
        """
        stats = {}
        regimes = MarketRegime.ALL_REGIMES + [MarketRegime.UNKNOWN, "多头趋势", "空头趋势", "震荡市"]
        for direction in ("long", "short"):
            for regime in regimes:
                stats[(direction, regime)] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "avg_win_pct": 0.0,
                    "avg_loss_pct": 0.0,
                    "avg_profit_pct": 0.0,
                    "total_profit": 0.0,
                    "win_pcts": [],
                    "loss_pcts": [],
                    "profit_pcts": [],
                }

        for i, trade in enumerate(trades):
            regime = regime_map.get(i, MarketRegime.UNKNOWN)
            direction = "long" if trade.side == 1 else "short"
            key = (direction, regime)
            if key not in stats:
                stats[key] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "avg_win_pct": 0.0,
                    "avg_loss_pct": 0.0,
                    "avg_profit_pct": 0.0,
                    "total_profit": 0.0,
                    "win_pcts": [],
                    "loss_pcts": [],
                    "profit_pcts": [],
                }

            s = stats[key]
            s["count"] += 1
            if trade.profit > 0:
                s["wins"] += 1
                s["win_pcts"].append(trade.profit_pct)
            else:
                s["losses"] += 1
                s["loss_pcts"].append(trade.profit_pct)

            s["total_profit"] += trade.profit
            s["profit_pcts"].append(trade.profit_pct)

        for _, s in stats.items():
            if s["count"] > 0:
                s["win_rate"] = s["wins"] / s["count"]
                s["avg_win_pct"] = np.mean(s["win_pcts"]) if s["win_pcts"] else 0.0
                s["avg_loss_pct"] = np.mean(s["loss_pcts"]) if s["loss_pcts"] else 0.0
                s["avg_profit_pct"] = (
                    np.mean(s["profit_pcts"]) if s["profit_pcts"] else 0.0
                )
            del s["win_pcts"]
            del s["loss_pcts"]
            del s["profit_pcts"]

        return stats
