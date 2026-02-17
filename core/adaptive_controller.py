"""
R3000 统一自适应控制器
决策快照 + 反事实分析 + 特征学习 + 自动调参 + 防死亡螺旋

功能：
  - TradeContext: 交易完整上下文（订单+快照+价格历史）
  - Diagnosis: 根因诊断结果
  - CounterfactualResult: 反事实分析结果（"如果我...会怎样"）
  - FeaturePatternDB: 特征模式数据库（哪些指标范围胜率高）
  - RegimeAdapter: 市场状态分类器自适应
  - AdaptiveController: 统一控制器（自动调参+防死亡螺旋）
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict, deque

from core.paper_trader import PaperOrder, OrderSide, CloseReason
from core.trade_reasoning import DecisionSnapshot


@dataclass
class TradeContext:
    """
    交易完整上下文
    包含订单、快照、价格历史等所有回溯分析需要的数据
    """
    order: PaperOrder
    
    # 决策快照
    entry_snapshot: DecisionSnapshot
    exit_snapshot: Optional[DecisionSnapshot]
    
    # 持仓期间的价格历史（用于反事实分析）
    price_history: List[Tuple[int, float]]  # [(bar_idx, price), ...]
    
    # 持仓期间的指标快照（用于回放）
    indicator_snapshots: List[DecisionSnapshot]
    
    # 结果
    profit_pct: float
    hold_bars: int
    close_reason: CloseReason


@dataclass
class Diagnosis:
    """根因诊断结果"""
    primary_cause: str  # 主要原因
    secondary_causes: List[str]  # 次要原因
    confidence: float  # 诊断置信度 (0-1)
    
    # 具体问题标记
    regime_error: bool = False  # 市场状态误判
    entry_premature: bool = False  # 入场过早
    entry_late: bool = False  # 入场过晚
    entry_indicators_unfavorable: bool = False  # 入场指标不利
    sl_too_tight: bool = False  # 止损过紧
    sl_too_loose: bool = False  # 止损过松
    tp_too_far: bool = False  # 止盈过远
    tp_too_close: bool = False  # 止盈过近
    exit_premature: bool = False  # 出场过早
    exit_late: bool = False  # 出场过晚
    hold_too_long: bool = False  # 持仓过久
    
    # 建议调整
    suggested_adjustments: Dict[str, float] = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """反事实分析结果"""
    scenario: str  # 场景描述
    
    # "如果入场晚一点"
    better_entry_bar: Optional[int] = None
    better_entry_price: Optional[float] = None
    entry_improvement_pct: float = 0.0
    
    # "如果出场早/晚一点"
    better_exit_bar: Optional[int] = None
    better_exit_price: Optional[float] = None
    exit_improvement_pct: float = 0.0
    
    # "如果TP/SL不同"
    optimal_sl_atr: Optional[float] = None
    optimal_tp_atr: Optional[float] = None
    sl_tp_improvement_pct: float = 0.0
    
    # 总体改进空间
    total_improvement_pct: float = 0.0
    
    # 建议参数调整
    parameter_deltas: Dict[str, float] = field(default_factory=dict)


class FeaturePatternDB:
    """
    特征模式数据库
    记录哪些指标范围与胜率相关，用于特征级别的门控学习
    """
    
    def __init__(self):
        """初始化"""
        # 数据结构: {(regime, direction): {"win": {...}, "loss": {...}}}
        # 每个指标存储: {"kdj_j": [values...], "macd_slope": [values...], ...}
        self.patterns: Dict[Tuple[str, str], Dict[str, Dict[str, List[float]]]] = defaultdict(
            lambda: {"win": defaultdict(list), "loss": defaultdict(list)}
        )
        
        # 统计信息
        self.stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "wins": 0, "losses": 0}
        )
    
    def record_entry(self, snapshot: DecisionSnapshot, outcome: float, regime: str, direction: str):
        """
        记录入场指标与结果
        
        Args:
            snapshot: 入场时的决策快照
            outcome: 结果（profit_pct）
            regime: 市场状态
            direction: 方向 (LONG/SHORT)
        """
        key = (regime, direction)
        category = "win" if outcome > 0 else "loss"
        
        # 记录各指标值
        self.patterns[key][category]["kdj_j"].append(snapshot.kdj_j)
        self.patterns[key][category]["kdj_d"].append(snapshot.kdj_d)
        self.patterns[key][category]["kdj_k"].append(snapshot.kdj_k)
        self.patterns[key][category]["macd_hist"].append(snapshot.macd_hist)
        self.patterns[key][category]["macd_slope"].append(snapshot.macd_hist_slope)
        self.patterns[key][category]["rsi"].append(snapshot.rsi)
        self.patterns[key][category]["adx"].append(snapshot.adx)
        self.patterns[key][category]["boll_position"].append(snapshot.boll_position)
        self.patterns[key][category]["atr_change"].append(snapshot.atr_change_pct)
        self.patterns[key][category]["volume_ratio"].append(snapshot.volume_ratio)
        
        # 更新统计
        self.stats[key]["total"] += 1
        if outcome > 0:
            self.stats[key]["wins"] += 1
        else:
            self.stats[key]["losses"] += 1
    
    def get_feature_filters(self, regime: str, direction: str, min_samples: int = 10) -> Dict[str, Tuple[float, float]]:
        """
        获取学习到的特征范围（有利于胜率的范围）
        
        Args:
            regime: 市场状态
            direction: 方向
            min_samples: 最少样本数
        
        Returns:
            {feature_name: (min_value, max_value), ...}
        """
        key = (regime, direction)
        
        if self.stats[key]["total"] < min_samples:
            return {}  # 样本不足，返回空
        
        filters = {}
        win_data = self.patterns[key]["win"]
        loss_data = self.patterns[key]["loss"]
        
        # 对每个特征分析
        for feature in ["kdj_j", "kdj_d", "kdj_k", "macd_hist", "macd_slope", "rsi", "adx", "boll_position"]:
            if feature not in win_data or len(win_data[feature]) < 5:
                continue
            
            # 计算胜利样本的分位数范围 (25%-75%)
            win_values = win_data[feature]
            p25, p75 = np.percentile(win_values, [25, 75])
            
            # 如果有足够的亏损样本，排除亏损集中区域
            if feature in loss_data and len(loss_data[feature]) >= 5:
                loss_values = loss_data[feature]
                loss_median = np.median(loss_values)
                
                # 如果亏损中位数在胜利范围外，说明有区分度
                if loss_median < p25 or loss_median > p75:
                    filters[feature] = (float(p25), float(p75))
            else:
                filters[feature] = (float(p25), float(p75))
        
        return filters
    
    def evaluate_snapshot(self, snapshot: DecisionSnapshot, regime: str, direction: str) -> float:
        """
        评估快照的特征质量分数 (0-1)
        
        Args:
            snapshot: 决策快照
            regime: 市场状态
            direction: 方向
        
        Returns:
            质量分数 (0-1)，1表示最优
        """
        filters = self.get_feature_filters(regime, direction)
        
        if not filters:
            return 0.5  # 无数据时返回中性分数
        
        # 计算有多少特征在"好"范围内
        in_range_count = 0
        total_count = 0
        
        feature_values = {
            "kdj_j": snapshot.kdj_j,
            "kdj_d": snapshot.kdj_d,
            "kdj_k": snapshot.kdj_k,
            "macd_hist": snapshot.macd_hist,
            "macd_slope": snapshot.macd_hist_slope,
            "rsi": snapshot.rsi,
            "adx": snapshot.adx,
            "boll_position": snapshot.boll_position,
        }
        
        for feature, (min_val, max_val) in filters.items():
            if feature in feature_values:
                value = feature_values[feature]
                total_count += 1
                if min_val <= value <= max_val:
                    in_range_count += 1
        
        if total_count == 0:
            return 0.5
        
        return in_range_count / total_count
    
    def to_dict(self) -> dict:
        """转为字典（用于持久化）"""
        return {
            "patterns": {
                f"{regime}_{direction}": {
                    "win": {k: list(v) for k, v in data["win"].items()},
                    "loss": {k: list(v) for k, v in data["loss"].items()},
                }
                for (regime, direction), data in self.patterns.items()
            },
            "stats": {
                f"{regime}_{direction}": dict(stat)
                for (regime, direction), stat in self.stats.items()
            }
        }
    
    def from_dict(self, data: dict):
        """从字典加载"""
        if "patterns" in data:
            for key_str, patterns in data["patterns"].items():
                regime, direction = key_str.rsplit("_", 1)
                key = (regime, direction)
                for category in ["win", "loss"]:
                    for feature, values in patterns.get(category, {}).items():
                        self.patterns[key][category][feature] = list(values)
        
        if "stats" in data:
            for key_str, stat in data["stats"].items():
                regime, direction = key_str.rsplit("_", 1)
                key = (regime, direction)
                self.stats[key] = dict(stat)


class KellyAdapter:
    """
    凯利参数自适应学习
    跟踪不同凯利仓位的表现，自动调整 KELLY_FRACTION, KELLY_MAX, KELLY_MIN
    """
    
    def __init__(self):
        """初始化"""
        # 不同凯利分数的表现跟踪
        # {kelly_fraction: {"trades": int, "total_profit": float, "wins": int, "losses": int}}
        self.kelly_fraction_stats: Dict[float, Dict[str, Any]] = defaultdict(
            lambda: {"trades": 0, "total_profit": 0.0, "wins": 0, "losses": 0, "max_drawdown": 0.0}
        )
        
        # 仓位使用统计 (跟踪实际使用的仓位比例)
        self.position_distribution: List[float] = []  # 最近100笔交易的凯利仓位
        
        # 当前参数（会覆盖config中的值）
        self.kelly_fraction: Optional[float] = None
        self.kelly_max: Optional[float] = None
        self.kelly_min: Optional[float] = None
        
        # 杠杆自适应
        self.leverage: Optional[int] = None
        
        # 参数调整历史
        self.adjustment_history: List[Dict] = []
        
        # 性能指标
        self.recent_performance: deque = deque(maxlen=20)  # 最近20笔交易的收益
        self.drawdown_tracker: float = 0.0  # 当前回撤
        self.peak_capital: float = 0.0  # 资金峰值
    
    def record_trade(self, kelly_position_pct: float, profit_pct: float, 
                     kelly_fraction_used: float, current_capital: float):
        """
        记录交易结果
        
        Args:
            kelly_position_pct: 使用的凯利仓位（0-1）
            profit_pct: 交易盈亏（%）
            kelly_fraction_used: 使用的凯利分数
            current_capital: 当前资金
        """
        # 记录凯利分数的表现
        stats = self.kelly_fraction_stats[kelly_fraction_used]
        stats["trades"] += 1
        stats["total_profit"] += profit_pct
        if profit_pct > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        
        # 记录仓位分布
        self.position_distribution.append(kelly_position_pct)
        if len(self.position_distribution) > 100:
            self.position_distribution = self.position_distribution[-100:]
        
        # 记录最近表现
        self.recent_performance.append(profit_pct)
        
        # 更新回撤
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
            self.drawdown_tracker = 0.0
        else:
            self.drawdown_tracker = (self.peak_capital - current_capital) / self.peak_capital * 100
            stats["max_drawdown"] = max(stats["max_drawdown"], self.drawdown_tracker)
    
    def get_kelly_fraction_performance(self, kelly_fraction: float) -> Dict[str, float]:
        """
        获取某个凯利分数的表现指标
        
        Args:
            kelly_fraction: 凯利分数
        
        Returns:
            {"win_rate": float, "avg_profit": float, "sharpe": float, "max_drawdown": float}
        """
        stats = self.kelly_fraction_stats[kelly_fraction]
        
        if stats["trades"] == 0:
            return {"win_rate": 0.0, "avg_profit": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
        
        win_rate = stats["wins"] / stats["trades"]
        avg_profit = stats["total_profit"] / stats["trades"]
        
        # 简化夏普率（标准差用20%估计）
        sharpe = avg_profit / 20.0 if avg_profit != 0 else 0.0
        
        return {
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "sharpe": sharpe,
            "max_drawdown": stats["max_drawdown"],
            "trades": stats["trades"],
        }
    
    def suggest_kelly_fraction_adjustment(self, min_trades: int = 1) -> Optional[float]:
        """
        建议 KELLY_FRACTION 调整
        
        Args:
            min_trades: 最少交易数
        
        Returns:
            建议的新 kelly_fraction 值，None 表示不建议调整
        """
        # 需要足够样本
        if len(self.recent_performance) < min_trades:
            return None
        
        # 计算最近表现
        recent_profits = list(self.recent_performance)
        avg_profit = np.mean(recent_profits)
        
        # 当前 kelly_fraction 和范围
        from config import PAPER_TRADING_CONFIG
        current_fraction = self.kelly_fraction or PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
        fraction_range = PAPER_TRADING_CONFIG.get("KELLY_FRACTION_RANGE", (0.15, 0.40))
        
        # 根据表现调整
        if avg_profit > 1.0 and self.drawdown_tracker < 15.0:
            # 连续盈利且回撤小，提升凯利分数
            suggested = min(current_fraction * 1.1, fraction_range[1])
            if suggested > current_fraction + 0.02:
                return suggested
        
        elif avg_profit < -0.5 or self.drawdown_tracker > 25.0:
            # 亏损或回撤大，降低凯利分数
            suggested = max(current_fraction * 0.9, fraction_range[0])
            if suggested < current_fraction - 0.02:
                return suggested
        
        return None
    
    def suggest_kelly_max_adjustment(self) -> Optional[float]:
        """
        建议 KELLY_MAX 调整
        
        Returns:
            建议的新 kelly_max 值，None 表示不建议调整
        """
        if len(self.position_distribution) < 1:
            return None
        
        # 当前 kelly_max 和范围
        from config import PAPER_TRADING_CONFIG
        current_max = self.kelly_max or PAPER_TRADING_CONFIG.get("KELLY_MAX_POSITION", 0.8)
        max_range = PAPER_TRADING_CONFIG.get("KELLY_MAX_RANGE", (0.5, 0.9))
        
        # 检查回撤
        if self.drawdown_tracker > 30.0:
            # 回撤过大，降低最大仓位
            suggested = max(current_max - 0.1, max_range[0])
            return suggested
        
        elif self.drawdown_tracker < 10.0 and len(self.recent_performance) >= 1:
            # 回撤小且稳定，可以提升最大仓位
            avg_profit = np.mean(list(self.recent_performance))
            if avg_profit > 0.8:
                suggested = min(current_max + 0.05, max_range[1])
                if suggested > current_max:
                    return suggested
        
        return None
    
    def suggest_kelly_min_adjustment(self) -> Optional[float]:
        """
        建议 KELLY_MIN 调整
        
        Returns:
            建议的新 kelly_min 值，None 表示不建议调整
        """
        if len(self.position_distribution) < 1:
            return None
        
        # 当前 kelly_min 和范围
        from config import PAPER_TRADING_CONFIG
        current_min = self.kelly_min or PAPER_TRADING_CONFIG.get("KELLY_MIN_POSITION", 0.05)
        min_range = PAPER_TRADING_CONFIG.get("KELLY_MIN_RANGE", (0.05, 0.20))
        
        # 检查仓位分布
        avg_position = np.mean(self.position_distribution)
        
        # 如果平均仓位接近最小值，说明信号质量普遍不高
        if avg_position < current_min * 1.3:
            # 波动过小，可以提升最小仓位（更激进）
            suggested = min(current_min + 0.02, min_range[1])
            return suggested
        
        elif avg_position > current_min * 3.0:
            # 平均仓位远高于最小值，可以降低最小仓位（给低质量信号更多试探空间）
            suggested = max(current_min - 0.02, min_range[0])
            return suggested
        
        return None
    
    def adjust_leverage_after_trade(self, profit_pct: float) -> Optional[int]:
        """
        每笔开平仓后按盈亏调整杠杆：盈利则放大、亏损则缩小。
        范围 [LEVERAGE_MIN, LEVERAGE_MAX]，步长 LEVERAGE_ADJUST_STEP，基线 LEVERAGE_DEFAULT=20。
        
        Returns:
            新杠杆值（若发生调整），否则 None
        """
        from config import PAPER_TRADING_CONFIG
        if not PAPER_TRADING_CONFIG.get("LEVERAGE_ADAPTIVE", True):
            return None
        default_lev = PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20)
        min_lev = PAPER_TRADING_CONFIG.get("LEVERAGE_MIN", 5)
        max_lev = PAPER_TRADING_CONFIG.get("LEVERAGE_MAX", 100)
        step = PAPER_TRADING_CONFIG.get("LEVERAGE_ADJUST_STEP", 2)
        current = int(self.leverage) if self.leverage is not None else default_lev
        current = max(min_lev, min(max_lev, current))
        if profit_pct > 0:
            new_lev = min(current + step, max_lev)
        else:
            new_lev = max(current - step, min_lev)
        if new_lev == current:
            return None
        old_lev = self.leverage
        self.leverage = new_lev
        self.adjustment_history.append({
            "timestamp": datetime.now().isoformat(),
            "parameter": "LEVERAGE",
            "old_value": float(old_lev or current),
            "new_value": float(self.leverage),
            "delta": float(self.leverage - (old_lev or current)),
            "reason": f"每笔调整: 本笔盈亏{profit_pct:+.2f}% → {'放大' if profit_pct > 0 else '缩小'}",
        })
        if len(self.adjustment_history) > 50:
            self.adjustment_history = self.adjustment_history[-50:]
        return self.leverage

    def suggest_leverage_adjustment(self, min_trades: int = 20) -> Optional[int]:
        """
        建议 LEVERAGE 调整（基于近期多笔表现，用于 auto_adjust_parameters）。
        日常以 adjust_leverage_after_trade 每笔调整为准；此处保留兼容。
        """
        from config import PAPER_TRADING_CONFIG
        if not PAPER_TRADING_CONFIG.get("LEVERAGE_ADAPTIVE", False):
            return None
        if len(self.recent_performance) < min_trades:
            return None
        current_leverage = self.leverage or PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20)
        min_leverage = PAPER_TRADING_CONFIG.get("LEVERAGE_MIN", 5)
        max_leverage = PAPER_TRADING_CONFIG.get("LEVERAGE_MAX", 100)
        recent_profits = list(self.recent_performance)
        avg_profit = np.mean(recent_profits)
        win_rate = sum(1 for p in recent_profits if p > 0) / len(recent_profits)
        profit_threshold = PAPER_TRADING_CONFIG.get("LEVERAGE_PROFIT_THRESHOLD", 2.0)
        loss_threshold = PAPER_TRADING_CONFIG.get("LEVERAGE_LOSS_THRESHOLD", -1.0)
        drawdown_limit = PAPER_TRADING_CONFIG.get("LEVERAGE_DRAWDOWN_LIMIT", 20.0)
        if (avg_profit > profit_threshold and win_rate > 0.55 and
            self.drawdown_tracker < 10.0 and current_leverage < max_leverage):
            suggested = min(current_leverage + 2, max_leverage)
            if suggested > current_leverage:
                return suggested
        elif (avg_profit < loss_threshold or self.drawdown_tracker > drawdown_limit or win_rate < 0.40):
            reduction = 5 if self.drawdown_tracker > drawdown_limit else 2
            suggested = max(current_leverage - reduction, min_leverage)
            if suggested < current_leverage:
                return suggested
        return None
    
    def auto_adjust_parameters(self, learning_rate: float = 0.1):
        """
        自动调整凯利参数
        
        Args:
            learning_rate: 调整幅度 (0-1)
        """
        adjustments = []
        
        # 1. 调整 KELLY_FRACTION
        suggested_fraction = self.suggest_kelly_fraction_adjustment()
        if suggested_fraction is not None:
            if self.kelly_fraction is None:
                from config import PAPER_TRADING_CONFIG
                self.kelly_fraction = PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
            
            # 平滑调整
            delta_val = (suggested_fraction - self.kelly_fraction) * learning_rate
            new_fraction = self.kelly_fraction + delta_val
            avg_profit = np.mean(list(self.recent_performance)) if self.recent_performance else 0.0
            
            delta_pct = (new_fraction - self.kelly_fraction) / self.kelly_fraction * 100 if self.kelly_fraction != 0 else 0
            adjustments.append({
                "timestamp": datetime.now().isoformat(),
                "parameter": "KELLY_FRACTION",
                "old_value": float(self.kelly_fraction),
                "new_value": float(new_fraction),
                "delta": float(new_fraction - self.kelly_fraction),
                "delta_pct": float(delta_pct),
                "reason": f"基于表现调整 (平均收益: {avg_profit:.2f}%, 回撤: {self.drawdown_tracker:.1f}%)",
            })
            
            self.kelly_fraction = new_fraction
        
        # 2. 调整 KELLY_MAX
        suggested_max = self.suggest_kelly_max_adjustment()
        if suggested_max is not None:
            if self.kelly_max is None:
                from config import PAPER_TRADING_CONFIG
                self.kelly_max = PAPER_TRADING_CONFIG.get("KELLY_MAX_POSITION", 0.8)
            
            # 平滑调整
            delta_val = (suggested_max - self.kelly_max) * learning_rate
            new_max = self.kelly_max + delta_val
            
            delta_pct = (new_max - self.kelly_max) / self.kelly_max * 100 if self.kelly_max != 0 else 0
            adjustments.append({
                "timestamp": datetime.now().isoformat(),
                "parameter": "KELLY_MAX",
                "old_value": float(self.kelly_max),
                "new_value": float(new_max),
                "delta": float(new_max - self.kelly_max),
                "delta_pct": float(delta_pct),
                "reason": f"基于回撤调整 (当前回撤: {self.drawdown_tracker:.1f}%)",
            })
            
            self.kelly_max = new_max
        
        # 3. 调整 KELLY_MIN
        suggested_min = self.suggest_kelly_min_adjustment()
        if suggested_min is not None:
            if self.kelly_min is None:
                from config import PAPER_TRADING_CONFIG
                self.kelly_min = PAPER_TRADING_CONFIG.get("KELLY_MIN_POSITION", 0.05)
            
            # 平滑调整
            delta_val = (suggested_min - self.kelly_min) * learning_rate
            new_min = self.kelly_min + delta_val
            
            delta_pct = (new_min - self.kelly_min) / self.kelly_min * 100 if self.kelly_min != 0 else 0
            avg_pos = np.mean(self.position_distribution) if self.position_distribution else 0
            adjustments.append({
                "timestamp": datetime.now().isoformat(),
                "parameter": "KELLY_MIN",
                "old_value": float(self.kelly_min),
                "new_value": float(new_min),
                "delta": float(new_min - self.kelly_min),
                "delta_pct": float(delta_pct),
                "reason": f"基于仓位分布调整 (平均仓位: {avg_pos:.1%})",
            })
            
            self.kelly_min = new_min
        
        # 4. 杠杆由每笔开平仓的 adjust_leverage_after_trade 单独调整，此处不再调整
        
        # 记录调整历史
        if adjustments:
            self.adjustment_history.extend(adjustments)
            # 只保留最近50条
            if len(self.adjustment_history) > 50:
                self.adjustment_history = self.adjustment_history[-50:]
    
    def get_current_parameters(self) -> Dict[str, float]:
        """获取当前凯利参数（用于覆盖config）"""
        from config import PAPER_TRADING_CONFIG
        
        return {
            "KELLY_FRACTION": self.kelly_fraction or PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25),
            "KELLY_MAX": self.kelly_max or PAPER_TRADING_CONFIG.get("KELLY_MAX_POSITION", 0.8),
            "KELLY_MIN": self.kelly_min or PAPER_TRADING_CONFIG.get("KELLY_MIN_POSITION", 0.05),
            "LEVERAGE": self.leverage or PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20),
        }
    
    def to_dict(self) -> dict:
        """转为字典（用于持久化）"""
        return {
            "kelly_fraction_stats": {
                str(k): dict(v) for k, v in self.kelly_fraction_stats.items()
            },
            "position_distribution": list(self.position_distribution),
            "kelly_fraction": self.kelly_fraction,
            "kelly_max": self.kelly_max,
            "kelly_min": self.kelly_min,
            "leverage": self.leverage,
            "adjustment_history": self.adjustment_history[-20:],
            "recent_performance": list(self.recent_performance),
            "drawdown_tracker": self.drawdown_tracker,
            "peak_capital": self.peak_capital,
        }
    
    def from_dict(self, data: dict):
        """从字典加载"""
        if "kelly_fraction_stats" in data:
            for k_str, stats in data["kelly_fraction_stats"].items():
                k = float(k_str)
                self.kelly_fraction_stats[k] = dict(stats)
        
        self.position_distribution = data.get("position_distribution", [])
        self.kelly_fraction = data.get("kelly_fraction")
        self.kelly_max = data.get("kelly_max")
        self.kelly_min = data.get("kelly_min")
        self.leverage = data.get("leverage")
        self.adjustment_history = data.get("adjustment_history", [])
        
        recent_perf = data.get("recent_performance", [])
        self.recent_performance = deque(recent_perf, maxlen=20)
        
        self.drawdown_tracker = data.get("drawdown_tracker", 0.0)
        self.peak_capital = data.get("peak_capital", 0.0)


class RegimeAdapter:
    """
    市场状态分类器自适应学习
    跟踪每种状态的准确率，自动调整阈值
    """
    
    def __init__(self):
        """初始化"""
        # 市场状态准确率统计: {regime: {"correct": int, "wrong": int}}
        self.regime_accuracy: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "wrong": 0}
        )
        
        # 误分类模式记录
        self.misclassification_patterns: List[Dict] = []
        
        # 参数调整历史
        self.adjustment_history: List[Dict] = []
        
        # 当前参数（会覆盖config中的值）
        self.dir_strong_threshold: Optional[float] = None
        self.short_trend_threshold: Optional[float] = None
        
        # 每种状态的入场阈值调整
        self.regime_entry_thresholds: Dict[str, float] = {}  # {regime: threshold_adjustment}
    
    def record_regime_accuracy(self, entry_snapshot: DecisionSnapshot, 
                                actual_price_movement: float, side: OrderSide,
                                profit_pct: float = 0.0, hold_bars: int = 0):
        """
        记录市场状态分类准确性（增强版）
        
        Args:
            entry_snapshot: 入场快照
            actual_price_movement: 实际价格移动（%，去杠杆）
            side: 开仓方向
            profit_pct: 交易盈亏（%，含杠杆）
            hold_bars: 持仓K线数
        """
        regime = entry_snapshot.market_regime
        is_long = (side == OrderSide.LONG)
        
        # 增强版市场状态准确性判断
        correct = False
        
        if "强多头" in regime or "多头" in regime or "看涨" in regime:
            # 多头市场 → 应该做多盈利，或至少价格上涨
            if is_long:
                # 做多时：盈利或价格持续上涨视为正确
                correct = (profit_pct > 0.5) or (actual_price_movement > 0.5)
            else:
                # 做空时：如果盈利说明判断错误（市场实际下跌）
                correct = (profit_pct < -0.5)  # 做空亏损说明市场确实是多头
        
        elif "强空头" in regime or "空头" in regime or "看跌" in regime:
            # 空头市场 → 应该做空盈利，或至少价格下跌
            if not is_long:
                # 做空时：盈利或价格持续下跌视为正确
                correct = (profit_pct > 0.5) or (actual_price_movement < -0.5)
            else:
                # 做多时：如果盈利说明判断错误（市场实际上涨）
                correct = (profit_pct < -0.5)  # 做多亏损说明市场确实是空头
        
        elif "震荡" in regime or "弱" in regime:
            # 震荡市场 → 价格不应有剧烈单向移动
            # 判断标准：价格移动小于2%（去杠杆后）
            price_volatility = abs(actual_price_movement)
            correct = price_volatility < 2.0
            
            # 或者：持仓时间短（震荡市应该快进快出）
            if hold_bars < 60:  # 1小时内
                correct = True
        
        else:
            # 未分类或其他状态，不计入统计
            return
        
        # 更新统计
        if correct:
            self.regime_accuracy[regime]["correct"] += 1
        else:
            self.regime_accuracy[regime]["wrong"] += 1
            
            # 记录误分类模式（用于后续分析）
            self.misclassification_patterns.append({
                "timestamp": datetime.now().isoformat(),
                "regime": regime,
                "side": side.value,
                "actual_movement": actual_price_movement,
                "profit_pct": profit_pct,
                "hold_bars": hold_bars,
                "indicators": {
                    "rsi": entry_snapshot.rsi,
                    "macd_hist": entry_snapshot.macd_hist,
                    "adx": entry_snapshot.adx,
                } if entry_snapshot else {},
            })
            
            # 只保留最近100条
            if len(self.misclassification_patterns) > 100:
                self.misclassification_patterns = self.misclassification_patterns[-100:]
    
    def get_regime_accuracy_rate(self, regime: str) -> float:
        """获取某市场状态的准确率"""
        stats = self.regime_accuracy[regime]
        total = stats["correct"] + stats["wrong"]
        if total == 0:
            return 0.5  # 无数据时返回中性
        return stats["correct"] / total
    
    def suggest_threshold_adjustment(self) -> Dict[str, float]:
        """
        根据准确率建议阈值调整（增强版）
        
        Returns:
            {"DIR_STRONG_THRESHOLD": new_value, "SHORT_TREND_THRESHOLD": new_value}
        """
        suggestions = {}
        
        # 检查各市场状态的准确率和样本量
        all_regimes = list(self.regime_accuracy.keys())
        regime_stats = {}
        
        for regime in all_regimes:
            stats = self.regime_accuracy[regime]
            total = stats["correct"] + stats["wrong"]
            if total > 0:
                regime_stats[regime] = {
                    "accuracy": stats["correct"] / total,
                    "total": total,
                }
        
        # 如果样本量不足，不做调整
        min_samples = 15
        sufficient_samples = sum(1 for s in regime_stats.values() if s["total"] >= min_samples)
        if sufficient_samples < 2:
            return suggestions
        
        # 计算趋势和震荡的总体准确率
        trend_regimes = [r for r in regime_stats.keys() if any(kw in r for kw in ["多头", "空头", "看涨", "看跌"])]
        ranging_regimes = [r for r in regime_stats.keys() if any(kw in r for kw in ["震荡", "弱"])]
        
        trend_acc = np.mean([regime_stats[r]["accuracy"] for r in trend_regimes if r in regime_stats]) if trend_regimes else 0.5
        ranging_acc = np.mean([regime_stats[r]["accuracy"] for r in ranging_regimes if r in regime_stats]) if ranging_regimes else 0.5
        
        # 策略1: 趋势识别准确率低 → 提高阈值（更难触发趋势）
        if trend_acc < 0.45:
            if self.dir_strong_threshold is None:
                from config import MARKET_REGIME_CONFIG
                self.dir_strong_threshold = MARKET_REGIME_CONFIG.get("DIR_STRONG_THRESHOLD", 0.003)
            
            # 提高阈值5-10%
            increase_pct = 0.05 + (0.45 - trend_acc) * 0.2  # 准确率越低，提升越多
            new_threshold = min(self.dir_strong_threshold * (1 + increase_pct), 0.01)
            suggestions["DIR_STRONG_THRESHOLD"] = new_threshold
        
        # 策略2: 趋势识别准确率高但震荡识别低 → 降低阈值（更敏感）
        elif trend_acc > 0.65 and ranging_acc < 0.45:
            if self.dir_strong_threshold is None:
                from config import MARKET_REGIME_CONFIG
                self.dir_strong_threshold = MARKET_REGIME_CONFIG.get("DIR_STRONG_THRESHOLD", 0.003)
            
            # 降低阈值5%
            new_threshold = max(self.dir_strong_threshold * 0.95, 0.0008)
            suggestions["DIR_STRONG_THRESHOLD"] = new_threshold
        
        # 策略3: 震荡识别准确率低 → 调整短期趋势阈值
        if ranging_acc < 0.40 and any(regime_stats.get(r, {}).get("total", 0) >= min_samples for r in ranging_regimes):
            if self.short_trend_threshold is None:
                from config import MARKET_REGIME_CONFIG
                self.short_trend_threshold = MARKET_REGIME_CONFIG.get("SHORT_TREND_THRESHOLD", 0.0015)
            
            # 震荡识别不准，可能是短期趋势阈值不合理
            # 提高短期阈值，让更多情况被识别为震荡
            new_threshold = min(self.short_trend_threshold * 1.08, 0.005)
            suggestions["SHORT_TREND_THRESHOLD"] = new_threshold
        
        return suggestions
    
    def auto_adjust_thresholds(self, learning_rate: float = 0.1):
        """
        自动调整阈值
        
        Args:
            learning_rate: 调整幅度 (0-1)
        """
        suggestions = self.suggest_threshold_adjustment()
        
        if not suggestions:
            return
        
        # 应用调整（使用学习率）
        for param, suggested_value in suggestions.items():
            if param == "DIR_STRONG_THRESHOLD":
                if self.dir_strong_threshold is None:
                    from config import MARKET_REGIME_CONFIG
                    self.dir_strong_threshold = MARKET_REGIME_CONFIG["DIR_STRONG_THRESHOLD"]
                
                # 平滑调整
                delta_val = (suggested_value - self.dir_strong_threshold) * learning_rate
                new_value = self.dir_strong_threshold + delta_val
                
                delta_pct = (new_value - self.dir_strong_threshold) / self.dir_strong_threshold * 100 if self.dir_strong_threshold != 0 else 0
                
                # 获取准确率信息
                bullish_acc = self.get_regime_accuracy_rate("看涨") if "看涨" in self.regime_accuracy else 0
                bearish_acc = self.get_regime_accuracy_rate("看跌") if "看跌" in self.regime_accuracy else 0
                trend_acc = (bullish_acc + bearish_acc) / 2 if (bullish_acc > 0 or bearish_acc > 0) else 0
                
                # 记录调整
                self.adjustment_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "parameter": param,
                    "old_value": float(self.dir_strong_threshold),
                    "new_value": float(new_value),
                    "delta": float(new_value - self.dir_strong_threshold),
                    "delta_pct": float(delta_pct),
                    "reason": f"市场状态准确率调整 (趋势准确率: {trend_acc:.1%})",
                })
                
                self.dir_strong_threshold = new_value
        
        # 只保留最近50条历史
        if len(self.adjustment_history) > 50:
            self.adjustment_history = self.adjustment_history[-50:]
    
    def get_regime_entry_threshold_adjustment(self, regime: str) -> float:
        """
        获取某市场状态下的入场阈值调整值
        
        Returns:
            调整值（加到基础阈值上）
        """
        return self.regime_entry_thresholds.get(regime, 0.0)
    
    def to_dict(self) -> dict:
        """转为字典（用于持久化）"""
        return {
            "regime_accuracy": {k: dict(v) for k, v in self.regime_accuracy.items()},
            "misclassification_patterns": self.misclassification_patterns[-20:],  # 只保存最近20条
            "adjustment_history": self.adjustment_history[-20:],
            "dir_strong_threshold": self.dir_strong_threshold,
            "short_trend_threshold": self.short_trend_threshold,
            "regime_entry_thresholds": dict(self.regime_entry_thresholds),
        }
    
    def from_dict(self, data: dict):
        """从字典加载"""
        if "regime_accuracy" in data:
            for regime, stats in data["regime_accuracy"].items():
                self.regime_accuracy[regime] = dict(stats)
        
        self.misclassification_patterns = data.get("misclassification_patterns", [])
        self.adjustment_history = data.get("adjustment_history", [])
        self.dir_strong_threshold = data.get("dir_strong_threshold")
        self.short_trend_threshold = data.get("short_trend_threshold")
        self.regime_entry_thresholds = data.get("regime_entry_thresholds", {})


class AdaptiveController:
    """
    统一自适应控制器
    整合决策快照、反事实分析、特征学习、自动调参
    """
    
    def __init__(self, state_file: str = "data/adaptive_controller_state.json"):
        """
        初始化
        
        Args:
            state_file: 状态持久化文件路径
        """
        self.state_file = state_file
        self._created_at: float = 0.0
        self._last_save_time: float = 0.0
        
        # 子模块
        self.feature_db = FeaturePatternDB()
        self.regime_adapter = RegimeAdapter()
        self.kelly_adapter = KellyAdapter()
        
        # 参数调整状态
        self.parameters: Dict[str, float] = {}  # 当前参数值
        self.parameter_history: List[Dict] = []  # 调整历史
        
        # 诊断历史（最近100笔）
        self.diagnosis_history: deque = deque(maxlen=100)
        
        # 统计信息
        self.stats: Dict[str, Any] = {
            "total_trades": 0,
            "consecutive_losses": 0,
            "consecutive_wins": 0,
            "last_trade_time": None,
            "per_regime_stats": defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0}),
        }
        
        # 学习率
        self.base_learning_rate = 0.05
        self.current_learning_rate = 0.05
        
        # 防死亡螺旋机制
        self.exploration_counter = 0  # 探索交易计数器
        self.min_trade_frequency_hours = 2.0  # 最少2小时内应有1笔交易
        
        # 加载持久化状态
        self.load_state()
    
    def on_trade_closed(self, context: TradeContext):
        """
        交易关闭时调用（主入口）
        
        Args:
            context: 交易完整上下文
        """
        # 1. 记录到特征数据库
        if context.entry_snapshot:
            regime = context.entry_snapshot.market_regime
            direction = context.order.side.value
            self.feature_db.record_entry(
                context.entry_snapshot,
                context.profit_pct,
                regime,
                direction
            )
        
        # 2. 记录市场状态准确性（增强版）
        if context.entry_snapshot and context.order.hold_bars > 0:
            self.regime_adapter.record_regime_accuracy(
                context.entry_snapshot,
                context.profit_pct / 10,  # 转为价格百分比（去杠杆）
                context.order.side,
                profit_pct=context.profit_pct,
                hold_bars=context.order.hold_bars
            )
        
        # 3. 根因诊断
        diagnosis = self.diagnose_trade(context)
        self.diagnosis_history.append({
            "timestamp": datetime.now().isoformat(),
            "order_id": context.order.order_id,
            "diagnosis": diagnosis,
        })
        
        # 4. 反事实分析
        counterfactual = self.counterfactual_analysis(context)
        
        # 5. 自动调整参数
        self.auto_adjust_parameters(diagnosis, counterfactual, context)
        
        # 6. 记录凯利仓位表现
        if hasattr(context.order, 'kelly_position_pct') and context.order.kelly_position_pct > 0:
            from config import PAPER_TRADING_CONFIG
            kelly_fraction_used = self.kelly_adapter.kelly_fraction or PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
            # 估算当前资金（简化处理）
            current_capital = 5000.0 * (1 + context.profit_pct / 100)
            
            self.kelly_adapter.record_trade(
                kelly_position_pct=context.order.kelly_position_pct,
                profit_pct=context.profit_pct,
                kelly_fraction_used=kelly_fraction_used,
                current_capital=current_capital
            )
        
        # 7. 更新统计
        self._update_stats(context)
        
        # 8. 防死亡螺旋检查
        self._check_death_spiral()
        
        # 9. 凯利参数自适应调整
        if self.stats["total_trades"] % 10 == 0:
            self.kelly_adapter.auto_adjust_parameters(self.current_learning_rate)
        
        # 10. 定期保存状态
        if self.stats["total_trades"] % 5 == 0:
            self.save_state()
    
    def on_trade_closed_simple(self, order: PaperOrder, market_regime: str = "未知"):
        """
        简化版交易关闭回调（用于没有完整TradeContext的场景）
        
        Args:
            order: 纸单订单对象
            market_regime: 市场状态（从entry_reason提取或传入）
        """
        # 记录凯利仓位表现
        if hasattr(order, 'kelly_position_pct') and order.kelly_position_pct > 0:
            from config import PAPER_TRADING_CONFIG
            kelly_fraction_used = self.kelly_adapter.kelly_fraction or PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
            # 估算当前资金（简化处理）
            current_capital = 5000.0 * (1 + order.profit_pct / 100)
            
            self.kelly_adapter.record_trade(
                kelly_position_pct=order.kelly_position_pct,
                profit_pct=order.profit_pct,
                kelly_fraction_used=kelly_fraction_used,
                current_capital=current_capital
            )
        
        # 简单的市场状态准确性记录（无完整快照）
        if order.hold_bars > 0 and market_regime != "未知":
            # 简化版：只根据盈亏判断市场状态是否合理
            is_correct = False
            if "多头" in market_regime or "看涨" in market_regime:
                is_correct = (order.side == OrderSide.LONG and order.profit_pct > 0)
            elif "空头" in market_regime or "看跌" in market_regime:
                is_correct = (order.side == OrderSide.SHORT and order.profit_pct > 0)
            elif "震荡" in market_regime:
                is_correct = abs(order.profit_pct) < 20.0  # 杠杆后20%
            else:
                is_correct = True  # 未知状态不计算
            
            if is_correct:
                self.regime_adapter.regime_accuracy[market_regime]["correct"] += 1
            else:
                self.regime_adapter.regime_accuracy[market_regime]["wrong"] += 1
        
        # 更新统计
        self.stats["total_trades"] += 1
        self.stats["last_trade_time"] = datetime.now().isoformat()
        
        if order.profit_pct > 0:
            self.stats["consecutive_wins"] += 1
            self.stats["consecutive_losses"] = 0
        else:
            self.stats["consecutive_losses"] += 1
            self.stats["consecutive_wins"] = 0
        
        # 杠杆自适应：每笔平仓后都按盈亏调整（盈利放大、亏损缩小，5x~100x，默认20x）
        from config import PAPER_TRADING_CONFIG
        if self.kelly_adapter and PAPER_TRADING_CONFIG.get("LEVERAGE_ADAPTIVE", True):
            self.kelly_adapter.adjust_leverage_after_trade(order.profit_pct)
        
        # 仓位自适应：每笔平仓后都调整凯利参数（KELLY_FRACTION / KELLY_MAX / KELLY_MIN），下一笔开仓即生效
        if self.kelly_adapter:
            self.kelly_adapter.auto_adjust_parameters(self.current_learning_rate)
        
        # 市场状态自适应调整
        self.regime_adapter.auto_adjust_thresholds(self.current_learning_rate)
        
        # 定期保存状态
        if self.stats["total_trades"] % 5 == 0:
            self.save_state()
    
    def diagnose_trade(self, context: TradeContext) -> Diagnosis:
        """
        诊断交易失败/成功的根本原因
        
        Args:
            context: 交易上下文
        
        Returns:
            Diagnosis 诊断结果
        """
        diagnosis = Diagnosis(
            primary_cause="",
            secondary_causes=[],
            confidence=0.0,
        )
        
        # 如果盈利，简单标记为成功
        if context.profit_pct > 0:
            diagnosis.primary_cause = "successful_trade"
            diagnosis.confidence = 0.9
            return diagnosis
        
        # 亏损交易，详细诊断
        causes = []
        
        # 1. 市场状态检查
        if context.entry_snapshot and context.exit_snapshot:
            entry_regime = context.entry_snapshot.market_regime
            # 检查价格是否与状态预期相反
            actual_move = context.profit_pct / 10  # 去杠杆
            is_long = context.order.side == OrderSide.LONG
            
            if entry_regime == "看涨" and actual_move < -0.5:
                diagnosis.regime_error = True
                causes.append("regime_misclassification")
            elif entry_regime == "看跌" and actual_move > 0.5:
                diagnosis.regime_error = True
                causes.append("regime_misclassification")
        
        # 2. 入场时机检查（看入场指标是否在不利范围）
        if context.entry_snapshot:
            regime = context.entry_snapshot.market_regime
            direction = context.order.side.value
            feature_score = self.feature_db.evaluate_snapshot(
                context.entry_snapshot, regime, direction
            )
            
            if feature_score < 0.3:
                diagnosis.entry_indicators_unfavorable = True
                causes.append("entry_indicators_unfavorable")
        
        # 3. 止损检查
        if context.order.close_reason == CloseReason.STOP_LOSS:
            # 检查止损后价格是否反转
            if len(context.price_history) > context.order.exit_bar_idx:
                # 看止损后10根K线的价格
                post_exit_prices = [p for idx, p in context.price_history 
                                   if idx > context.order.exit_bar_idx 
                                   and idx <= context.order.exit_bar_idx + 10]
                if post_exit_prices:
                    avg_post_price = np.mean(post_exit_prices)
                    sl_price = context.order.stop_loss
                    
                    # 如果止损后价格回到有利区，说明止损过紧
                    is_long = context.order.side == OrderSide.LONG
                    if is_long and avg_post_price > sl_price * 1.01:
                        diagnosis.sl_too_tight = True
                        causes.append("sl_too_tight")
                    elif not is_long and avg_post_price < sl_price * 0.99:
                        diagnosis.sl_too_tight = True
                        causes.append("sl_too_tight")
        
        # 4. 出场时机检查
        if context.order.close_reason in [CloseReason.MANUAL, CloseReason.DERAIL, CloseReason.TRAILING_STOP]:
            # 看峰值利润与实际利润差距
            if context.order.peak_profit_pct > context.profit_pct + 1.0:
                diagnosis.exit_premature = True
                causes.append("exit_premature")
        
        # 5. 持仓时长检查
        if context.hold_bars > 180:  # 超过3小时
            diagnosis.hold_too_long = True
            causes.append("hold_too_long")
        
        # 确定主要原因
        if causes:
            diagnosis.primary_cause = causes[0]
            diagnosis.secondary_causes = causes[1:] if len(causes) > 1 else []
            diagnosis.confidence = 0.7
        else:
            diagnosis.primary_cause = "market_noise"
            diagnosis.confidence = 0.5
        
        # 生成建议调整
        diagnosis.suggested_adjustments = self._generate_adjustment_suggestions(diagnosis)
        
        return diagnosis
    
    def counterfactual_analysis(self, context: TradeContext) -> CounterfactualResult:
        """
        反事实分析："如果我...会怎样"
        
        Args:
            context: 交易上下文
        
        Returns:
            CounterfactualResult 反事实分析结果
        """
        result = CounterfactualResult(scenario="trade_retrospective")
        
        if not context.indicator_snapshots or not context.price_history:
            return result
        
        # 1. 寻找更好的入场点（入场后1-3根K线）
        entry_bar = context.order.entry_bar_idx
        entry_price = context.order.entry_price
        is_long = context.order.side == OrderSide.LONG
        
        for i in range(1, min(4, len(context.indicator_snapshots))):
            if i >= len(context.price_history):
                break
            
            alt_price = context.price_history[i][1]
            
            # 计算如果在这个点入场的盈亏差异
            if is_long:
                alt_profit_diff = (alt_price - entry_price) / entry_price * 100 * 10  # 10倍杠杆
            else:
                alt_profit_diff = (entry_price - alt_price) / entry_price * 100 * 10
            
            # 如果更好（亏损更少或盈利更多）
            if alt_profit_diff > result.entry_improvement_pct:
                result.better_entry_bar = entry_bar + i
                result.better_entry_price = alt_price
                result.entry_improvement_pct = alt_profit_diff
        
        # 2. 寻找更好的出场点（峰值利润点）
        if context.order.peak_profit_pct > context.profit_pct + 0.5:
            # 峰值显著好于实际，找到峰值发生的K线
            peak_profit = context.order.peak_profit_pct
            exit_improvement = peak_profit - context.profit_pct
            result.exit_improvement_pct = exit_improvement
            # 这里简化处理，实际应该从price_history中找到峰值时刻
            result.better_exit_bar = context.order.exit_bar_idx - 5  # 估计在出场前5根
        
        # 3. TP/SL 优化（简化版）
        # 如果止损触发，检查更宽松的止损是否能存活
        if context.order.close_reason == CloseReason.STOP_LOSS:
            # 建议放宽0.5倍ATR
            result.optimal_sl_atr = 2.5  # 假设当前是2.0
            result.sl_tp_improvement_pct = 2.0  # 估计改进
        
        # 计算总改进空间
        result.total_improvement_pct = (
            result.entry_improvement_pct + 
            result.exit_improvement_pct + 
            result.sl_tp_improvement_pct
        )
        
        # 生成参数调整建议
        if result.entry_improvement_pct > 0.5:
            result.parameter_deltas["ENTRY_CONFIRM_PCT"] = 0.0002  # 建议增加入场确认
        
        if result.exit_improvement_pct > 1.0:
            result.parameter_deltas["STAGED_TP_1_PCT"] = -0.5  # 建议提前分段止盈第一档
        
        if result.sl_tp_improvement_pct > 1.0:
            result.parameter_deltas["STOP_LOSS_ATR"] = 0.3  # 建议放宽止损
        
        return result
    
    def auto_adjust_parameters(self, diagnosis: Diagnosis, counterfactual: CounterfactualResult, context: TradeContext):
        """
        根据诊断和反事实分析自动调整参数
        
        Args:
            diagnosis: 诊断结果
            counterfactual: 反事实分析结果
            context: 交易上下文
        """
        adjustments = {}
        
        # 从诊断获取建议
        adjustments.update(diagnosis.suggested_adjustments)
        
        # 从反事实分析获取建议
        adjustments.update(counterfactual.parameter_deltas)
        
        # 应用调整（带学习率和边界检查）
        for param, delta in adjustments.items():
            current_value = self.parameters.get(param)
            if current_value is None:
                # 从config加载初始值
                current_value = self._get_config_value(param)
            
            # 应用调整
            new_value = current_value + delta * self.current_learning_rate
            
            # 边界检查
            new_value = self._apply_parameter_bounds(param, new_value)
            
            # 如果值有实际变化，记录调整
            if abs(new_value - current_value) > 1e-6:
                self._record_parameter_adjustment(
                    param_name=param,
                    old_value=current_value,
                    new_value=new_value,
                    reason=f"诊断触发: {diagnosis.primary_cause}"
                )
            
            self.parameters[param] = new_value
        
        # 只保留最近100条历史
        if len(self.parameter_history) > 100:
            self.parameter_history = self.parameter_history[-100:]
        
        # 调整学习率
        self._adjust_learning_rate()
        
        # 市场状态分类器自适应
        self.regime_adapter.auto_adjust_thresholds(self.current_learning_rate)
    
    def _record_parameter_adjustment(self, param_name: str, old_value: float, new_value: float, reason: str):
        """
        统一记录参数调整（标准格式）
        
        Args:
            param_name: 参数名称
            old_value: 旧值
            new_value: 新值
            reason: 调整原因
        """
        delta = new_value - old_value
        delta_pct = (delta / old_value * 100) if old_value != 0 else 0.0
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "parameter": param_name,
            "old_value": float(old_value),
            "new_value": float(new_value),
            "delta": float(delta),
            "delta_pct": float(delta_pct),
            "reason": reason,
        }
        
        self.parameter_history.append(record)
        
        # 只保留最近100条
        if len(self.parameter_history) > 100:
            self.parameter_history = self.parameter_history[-100:]
        
        # 打印调整日志
        sign = "+" if delta > 0 else ""
        print(f"[AdaptiveController] 📊 参数调整: {param_name} = {old_value:.4f} → {new_value:.4f} "
              f"({sign}{delta:.4f}, {sign}{delta_pct:.1f}%) | 原因: {reason}")
    
    def get_parameter(self, param_name: str) -> Optional[float]:
        """获取参数值（优先返回调整后的值）"""
        if param_name in self.parameters:
            return self.parameters[param_name]
        
        # 特殊处理：regime adapter的参数
        if param_name == "DIR_STRONG_THRESHOLD" and self.regime_adapter.dir_strong_threshold is not None:
            return self.regime_adapter.dir_strong_threshold
        
        # 特殊处理：kelly adapter的参数
        kelly_params = self.kelly_adapter.get_current_parameters()
        if param_name in kelly_params:
            return kelly_params[param_name]
        
        return None
    
    def get_all_adjustments(self, recent_n: int = 20) -> List[Dict]:
        """
        获取所有参数调整记录（包括子模块）
        
        Args:
            recent_n: 返回最近N条记录
        
        Returns:
            调整记录列表
        """
        all_adjustments = []
        
        # AdaptiveController的调整
        all_adjustments.extend(self.parameter_history)
        
        # KellyAdapter的调整
        all_adjustments.extend(self.kelly_adapter.adjustment_history)
        
        # RegimeAdapter的调整
        all_adjustments.extend(self.regime_adapter.adjustment_history)
        
        # 按时间排序
        all_adjustments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return all_adjustments[:recent_n]
    
    def should_relax_threshold(self) -> Tuple[bool, float]:
        """
        检查是否应该临时放宽阈值（防死亡螺旋）
        
        Returns:
            (should_relax, relax_amount)
        """
        # 检查交易频率
        if self.stats["last_trade_time"]:
            hours_since_last = (datetime.now() - datetime.fromisoformat(self.stats["last_trade_time"])).total_seconds() / 3600
            
            if hours_since_last > self.min_trade_frequency_hours:
                # 超过2小时无交易，放宽5%
                return True, 0.05
        
        # 检查连续亏损
        if self.stats["consecutive_losses"] >= 5:
            # 5连亏后，保持交易但减小仓位（由kelly公式自动处理）
            # 同时略微放宽阈值，鼓励继续交易
            return True, 0.03
        
        return False, 0.0
    
    def is_exploration_trade(self) -> bool:
        """判断是否应该进行探索交易（1/10概率）"""
        self.exploration_counter += 1
        if self.exploration_counter >= 10:
            self.exploration_counter = 0
            return True
        return False
    
    def _update_stats(self, context: TradeContext):
        """更新统计信息"""
        self.stats["total_trades"] += 1
        self.stats["last_trade_time"] = datetime.now().isoformat()
        
        # 连胜/连亏
        if context.profit_pct > 0:
            self.stats["consecutive_wins"] += 1
            self.stats["consecutive_losses"] = 0
        else:
            self.stats["consecutive_losses"] += 1
            self.stats["consecutive_wins"] = 0
        
        # 按状态统计
        regime = context.entry_snapshot.market_regime if context.entry_snapshot else "未知"
        regime_stats = self.stats["per_regime_stats"][regime]
        if context.profit_pct > 0:
            regime_stats["wins"] += 1
        else:
            regime_stats["losses"] += 1
        regime_stats["total_pnl"] += context.profit_pct
    
    def _check_death_spiral(self):
        """检查并防止死亡螺旋"""
        # 如果连亏超过3次，降低学习率（避免过度反应）
        if self.stats["consecutive_losses"] >= 3:
            self.current_learning_rate = self.base_learning_rate * 0.5
        else:
            self.current_learning_rate = self.base_learning_rate
    
    def _adjust_learning_rate(self):
        """根据连胜/连亏调整学习率"""
        if self.stats["consecutive_losses"] >= 3:
            # 连亏时加快调整
            self.current_learning_rate = min(self.base_learning_rate * (1 + self.stats["consecutive_losses"] * 0.5), 0.15)
        elif self.stats["consecutive_wins"] >= 3:
            # 连胜时减慢调整（避免过度优化）
            self.current_learning_rate = max(self.base_learning_rate * 0.5, 0.03)
        else:
            self.current_learning_rate = self.base_learning_rate
    
    def _generate_adjustment_suggestions(self, diagnosis: Diagnosis) -> Dict[str, float]:
        """根据诊断生成参数调整建议"""
        suggestions = {}
        
        if diagnosis.regime_error:
            suggestions["DIR_STRONG_THRESHOLD"] = 0.0005  # 提高阈值
        
        if diagnosis.entry_premature:
            suggestions["ENTRY_CONFIRM_PCT"] = 0.0002
        
        if diagnosis.sl_too_tight:
            suggestions["STOP_LOSS_ATR"] = 0.3  # 增加0.3倍ATR
        
        if diagnosis.tp_too_far:
            suggestions["TAKE_PROFIT_ATR"] = -0.5
        
        if diagnosis.exit_premature:
            suggestions["STAGED_TP_1_PCT"] = -0.5  # 提前分段止盈第一档
        
        return suggestions
    
    def _get_config_value(self, param_name: str) -> float:
        """从config获取参数初始值"""
        from config import PAPER_TRADING_CONFIG, MARKET_REGIME_CONFIG, VECTOR_SPACE_CONFIG
        
        # 映射参数名到config
        config_map = {
            "STOP_LOSS_ATR": (PAPER_TRADING_CONFIG, "STOP_LOSS_ATR"),
            "TAKE_PROFIT_ATR": (PAPER_TRADING_CONFIG, "TAKE_PROFIT_ATR"),
            "ENTRY_CONFIRM_PCT": (VECTOR_SPACE_CONFIG, "ENTRY_CONFIRM_PCT"),
            "DIR_STRONG_THRESHOLD": (MARKET_REGIME_CONFIG, "DIR_STRONG_THRESHOLD"),
            "SHORT_TREND_THRESHOLD": (MARKET_REGIME_CONFIG, "SHORT_TREND_THRESHOLD"),
            "STAGED_TP_1_PCT": (PAPER_TRADING_CONFIG, "STAGED_TP_1_PCT"),
            "STAGED_TP_2_PCT": (PAPER_TRADING_CONFIG, "STAGED_TP_2_PCT"),
            "STAGED_SL_1_PCT": (PAPER_TRADING_CONFIG, "STAGED_SL_1_PCT"),
            "STAGED_SL_2_PCT": (PAPER_TRADING_CONFIG, "STAGED_SL_2_PCT"),
        }
        
        if param_name in config_map:
            config, key = config_map[param_name]
            return config.get(key, 0.0)
        
        return 0.0
    
    def _apply_parameter_bounds(self, param_name: str, value: float) -> float:
        """应用参数边界限制"""
        bounds = {
            "STOP_LOSS_ATR": (0.8, 3.0),
            "TAKE_PROFIT_ATR": (1.0, 5.0),
            "ENTRY_CONFIRM_PCT": (0.0005, 0.005),
            "DIR_STRONG_THRESHOLD": (0.001, 0.01),
            "SHORT_TREND_THRESHOLD": (0.001, 0.01),
            "STAGED_TP_1_PCT": (3.0, 8.0),
            "STAGED_TP_2_PCT": (6.0, 15.0),
            "STAGED_SL_1_PCT": (2.0, 8.0),
            "STAGED_SL_2_PCT": (5.0, 15.0),
        }
        
        if param_name in bounds:
            min_val, max_val = bounds[param_name]
            return float(np.clip(value, min_val, max_val))
        
        return value
    
    def save_state(self):
        """保存状态到文件"""
        try:
            now = time.time()
            if not getattr(self, "_created_at", None) or self._created_at <= 0:
                self._created_at = now
            self._last_save_time = now
            state = {
                "created_at": self._created_at,
                "last_save_time": self._last_save_time,
                "parameters": self.parameters,
                "parameter_history": self.parameter_history[-50:],  # 只保存最近50条
                "stats": {
                    k: (dict(v) if isinstance(v, defaultdict) else v)
                    for k, v in self.stats.items()
                },
                "feature_db": self.feature_db.to_dict(),
                "regime_adapter": self.regime_adapter.to_dict(),
                "kelly_adapter": self.kelly_adapter.to_dict(),
                "diagnosis_history": list(self.diagnosis_history)[-20:],  # 只保存最近20条
                "current_learning_rate": self.current_learning_rate,
                "exploration_counter": self.exploration_counter,
            }
            
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            print(f"[AdaptiveController] 状态已保存: {self.state_file}")
        except Exception as e:
            print(f"[AdaptiveController] 保存状态失败: {e}")
    
    def load_state(self):
        """从文件加载状态"""
        if not os.path.exists(self.state_file):
            print(f"[AdaptiveController] 状态文件不存在，使用默认状态")
            return
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self._created_at = state.get("created_at", 0.0)
            self._last_save_time = state.get("last_save_time", 0.0)
            if self._created_at <= 0 and self._last_save_time > 0:
                self._created_at = self._last_save_time
            
            self.parameters = state.get("parameters", {})
            self.parameter_history = state.get("parameter_history", [])
            
            # 恢复stats
            if "stats" in state:
                for key, value in state["stats"].items():
                    if key == "per_regime_stats":
                        self.stats[key] = defaultdict(
                            lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0},
                            value
                        )
                    else:
                        self.stats[key] = value
            
            # 恢复子模块
            if "feature_db" in state:
                self.feature_db.from_dict(state["feature_db"])
            
            if "regime_adapter" in state:
                self.regime_adapter.from_dict(state["regime_adapter"])
            
            if "kelly_adapter" in state:
                self.kelly_adapter.from_dict(state["kelly_adapter"])
            
            if "diagnosis_history" in state:
                self.diagnosis_history = deque(state["diagnosis_history"], maxlen=100)
            
            self.current_learning_rate = state.get("current_learning_rate", self.base_learning_rate)
            self.exploration_counter = state.get("exploration_counter", 0)
            
            print(f"[AdaptiveController] 状态已加载: {len(self.parameters)} 个参数")
        except Exception as e:
            print(f"[AdaptiveController] 加载状态失败: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取用于UI展示的仪表板数据"""
        return {
            "total_trades": self.stats["total_trades"],
            "consecutive_losses": self.stats["consecutive_losses"],
            "consecutive_wins": self.stats["consecutive_wins"],
            "current_learning_rate": self.current_learning_rate,
            "per_regime_stats": dict(self.stats["per_regime_stats"]),
            "recent_adjustments": self.get_all_adjustments(recent_n=20),  # 统一获取所有调整
            "regime_accuracy": {
                regime: self.regime_adapter.get_regime_accuracy_rate(regime)
                for regime in list(self.regime_adapter.regime_accuracy.keys())[:5]  # 前5个状态
            },
            "current_parameters": self.parameters,
            "kelly_parameters": self.kelly_adapter.get_current_parameters(),
        }
    
    # ═══════════════════════════════════════════════════════════
    # 盈亏驱动分析（新增）
    # ═══════════════════════════════════════════════════════════
    
    def analyze_trades_by_close_reason(self, orders: List[Any], recent_n: int = 20) -> Dict[str, Any]:
        """
        按平仓原因分析交易盈亏
        
        Args:
            orders: PaperOrder列表
            recent_n: 分析最近N笔交易
        
        Returns:
            {
                "stats_by_reason": [CloseReasonStats数据...],
                "suggested_adjustments": [调整建议...],
                "summary": {...}
            }
        """
        if not orders:
            return {"stats_by_reason": [], "suggested_adjustments": [], "summary": {}}
        
        # 取最近N笔
        recent_orders = sorted(orders, key=lambda x: x.exit_time if x.exit_time else datetime.min, reverse=True)[:recent_n]
        
        # 按原因分组统计
        reason_stats = defaultdict(lambda: {
            "count": 0,
            "win_count": 0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "total_hold_bars": 0,
            "total_peak_loss": 0.0,  # 峰值流失总和
            "reversal_count": 0,     # 止损后反转计数
        })
        
        for order in recent_orders:
            reason = self._normalize_close_reason(order.close_reason)
            stats = reason_stats[reason]
            
            stats["count"] += 1
            stats["total_pnl"] += getattr(order, 'realized_pnl', 0)
            stats["total_pnl_pct"] += getattr(order, 'profit_pct', 0)
            stats["total_hold_bars"] += getattr(order, 'hold_bars', 0)
            
            if getattr(order, 'realized_pnl', 0) > 0:
                stats["win_count"] += 1
            
            # 峰值流失
            peak_pct = getattr(order, 'peak_profit_pct', 0)
            profit_pct = getattr(order, 'profit_pct', 0)
            if peak_pct > 0:
                stats["total_peak_loss"] += max(0, peak_pct - profit_pct)
            
            # 检测止损后反转（峰值利润超过止损亏损的30%）
            if reason == "止损" and peak_pct > abs(profit_pct) * 0.3:
                stats["reversal_count"] += 1
        
        # 转换为结果列表
        stats_list = []
        for reason, data in reason_stats.items():
            count = data["count"]
            avg_hold = data["total_hold_bars"] / count if count > 0 else 0
            avg_peak_loss = data["total_peak_loss"] / count if count > 0 else 0
            
            stats_list.append({
                "reason": reason,
                "count": count,
                "win_count": data["win_count"],
                "total_pnl": data["total_pnl"],
                "total_pnl_pct": data["total_pnl_pct"],
                "avg_hold_bars": avg_hold,
                "avg_peak_loss": avg_peak_loss,
                "reversal_count": data["reversal_count"],
                "suggestion": self._generate_pnl_suggestion(reason, data),
            })
        
        # 按数量排序
        stats_list.sort(key=lambda x: x["count"], reverse=True)
        
        # 生成调整建议
        suggested_adjustments = self._generate_adjustments_from_pnl(reason_stats, recent_orders)
        
        # 汇总统计
        total_trades = len(recent_orders)
        total_pnl = sum(getattr(o, 'realized_pnl', 0) for o in recent_orders)
        wins = sum(1 for o in recent_orders if getattr(o, 'realized_pnl', 0) > 0)
        
        return {
            "stats_by_reason": stats_list,
            "suggested_adjustments": suggested_adjustments,
            "summary": {
                "total_trades": total_trades,
                "total_pnl": total_pnl,
                "win_count": wins,
                "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            }
        }
    
    def _normalize_close_reason(self, close_reason) -> str:
        """标准化平仓原因"""
        if close_reason is None:
            return "未知"
        
        reason_value = close_reason.value if hasattr(close_reason, 'value') else str(close_reason)
        
        if "止损" in reason_value and "追踪" not in reason_value:
            return "止损"
        elif "止盈" in reason_value:
            return "止盈"
        elif "追踪" in reason_value or "TRAILING" in reason_value:
            return "追踪止损"
        elif "超时" in reason_value or "MAX_HOLD" in reason_value:
            return "超时离场"
        elif "脱轨" in reason_value or "DERAIL" in reason_value:
            return "脱轨"
        elif "手动" in reason_value or "MANUAL" in reason_value:
            return "手动平仓"
        elif "翻转" in reason_value:
            return "位置翻转"
        else:
            return reason_value
    
    def _generate_pnl_suggestion(self, reason: str, data: Dict) -> str:
        """根据盈亏数据生成建议"""
        count = data["count"]
        win_count = data["win_count"]
        total_pnl = data["total_pnl"]
        reversal_count = data.get("reversal_count", 0)
        avg_peak_loss = data["total_peak_loss"] / count if count > 0 else 0
        win_rate = (win_count / count * 100) if count > 0 else 0
        
        if reason == "止损":
            if count >= 3 and total_pnl < 0:
                if reversal_count >= count * 0.5:  # 超过50%止损后反转
                    return "放宽止损距离"
                elif avg_peak_loss < 10:  # 峰值流失小说明入场点不好
                    return "提高入场阈值"
            return "观察中..."
        
        elif reason == "止盈":
            if win_rate >= 80:
                if avg_peak_loss > 30:
                    return "考虑追踪止损"
                return "✓ 保持当前设置"
            return "检查止盈距离"
        
        elif reason == "追踪止损":
            if win_rate >= 60:
                if avg_peak_loss > 35:
                    return "提前启动追踪"
                return "✓ 追踪有效"
            return "调整追踪阈值"
        
        elif reason == "超时离场":
            if win_rate < 40 and count >= 2:
                return "缩短最大持仓"
            return "保持当前设置"
        
        elif reason == "脱轨":
            if win_rate < 50 and count >= 2:
                return "收紧脱轨阈值"
            return "保持当前设置"
        
        return "保持当前设置"
    
    def _generate_adjustments_from_pnl(self, reason_stats: Dict, orders: List) -> List[Dict]:
        """根据盈亏分析生成参数调整"""
        adjustments = []
        
        from config import PAPER_TRADING_CONFIG, SIMILARITY_CONFIG
        
        # 1. 止损分析
        sl_data = reason_stats.get("止损", {})
        if sl_data.get("count", 0) >= 3 and sl_data.get("total_pnl", 0) < 0:
            reversal_rate = sl_data.get("reversal_count", 0) / sl_data["count"]
            if reversal_rate >= 0.5:  # 50%以上止损后反转
                current_sl = self.get_parameter("STOP_LOSS_ATR") or PAPER_TRADING_CONFIG.get("STOP_LOSS_ATR", 2.0)
                new_sl = min(current_sl + 0.2, 3.0)
                if abs(new_sl - current_sl) > 0.01:
                    adjustments.append({
                        "parameter": "STOP_LOSS_ATR",
                        "old_value": current_sl,
                        "new_value": new_sl,
                        "reason": f"{sl_data['reversal_count']}笔止损后反转",
                    })
        
        # 2. 追踪止损/分段止盈分析
        ts_data = reason_stats.get("追踪止损", {})
        if ts_data.get("count", 0) >= 2:
            avg_peak_loss = ts_data.get("total_peak_loss", 0) / ts_data["count"]
            if avg_peak_loss > 30:  # 峰值流失超过30%
                current_ts = self.get_parameter("STAGED_TP_1_PCT") or PAPER_TRADING_CONFIG.get("STAGED_TP_1_PCT", 5.0)
                new_ts = max(current_ts - 0.5, 3.0)
                if abs(new_ts - current_ts) > 0.01:
                    adjustments.append({
                        "parameter": "STAGED_TP_1_PCT",
                        "old_value": current_ts,
                        "new_value": new_ts,
                        "reason": f"峰值利润流失{avg_peak_loss:.0f}%",
                    })
        
        # 3. 连续亏损分析
        consecutive_losses = self.stats.get("consecutive_losses", 0)
        if consecutive_losses >= 3:
            current_threshold = self.get_parameter("FUSION_THRESHOLD") or SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.65)
            adjustment = 0.02 * (consecutive_losses - 2)
            new_threshold = min(current_threshold + adjustment, 0.75)
            if abs(new_threshold - current_threshold) > 0.01:
                adjustments.append({
                    "parameter": "FUSION_THRESHOLD",
                    "old_value": current_threshold,
                    "new_value": new_threshold,
                    "reason": f"连亏{consecutive_losses}笔，收紧",
                })
        
        # 4. 超时离场分析
        timeout_data = reason_stats.get("超时离场", {})
        if timeout_data.get("count", 0) >= 2:
            win_rate = timeout_data.get("win_count", 0) / timeout_data["count"]
            if win_rate < 0.4:  # 超时胜率低于40%
                current_max_hold = self.get_parameter("MAX_HOLD_BARS") or PAPER_TRADING_CONFIG.get("MAX_HOLD_BARS", 180)
                new_max_hold = max(current_max_hold - 30, 60)
                if abs(new_max_hold - current_max_hold) > 5:
                    adjustments.append({
                        "parameter": "MAX_HOLD_BARS",
                        "old_value": current_max_hold,
                        "new_value": new_max_hold,
                        "reason": f"超时离场胜率仅{win_rate*100:.0f}%",
                    })
        
        return adjustments
    
    def apply_pnl_driven_adjustments(self, orders: List[Any]) -> List[Dict]:
        """
        应用盈亏驱动的参数调整
        
        Args:
            orders: 交易订单列表
        
        Returns:
            实际应用的调整列表
        """
        analysis = self.analyze_trades_by_close_reason(orders)
        adjustments = analysis.get("suggested_adjustments", [])
        
        applied = []
        for adj in adjustments:
            param = adj["parameter"]
            old_val = adj["old_value"]
            new_val = adj["new_value"]
            reason = adj["reason"]
            
            # 应用平滑调整
            smoothed_val = old_val + (new_val - old_val) * self.current_learning_rate
            smoothed_val = self._apply_parameter_bounds(param, smoothed_val)
            
            if abs(smoothed_val - old_val) > 1e-6:
                self.parameters[param] = smoothed_val
                self._record_parameter_adjustment(param, old_val, smoothed_val, reason)
                applied.append({
                    "parameter": param,
                    "old_value": old_val,
                    "new_value": smoothed_val,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                })
        
        return applied
