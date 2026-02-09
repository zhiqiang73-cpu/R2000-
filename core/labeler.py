"""
R3000 上帝视角标注引擎
利用未来数据找到理想买卖点（自适应 ZigZag + 动态规划最优交易序列）
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABELING_CONFIG
from utils.indicators import calculate_atr


class LabelType(IntEnum):
    """标注类型枚举 - 完整的 LONG/SHORT/EXIT 系统"""
    HOLD = 0
    LONG_ENTRY = 1      # 做多入场
    LONG_EXIT = 2       # 做多出场
    SHORT_ENTRY = -1    # 做空入场
    SHORT_EXIT = -2     # 做空出场


@dataclass
class SwingPoint:
    """摆动点数据结构"""
    index: int          # K线索引
    price: float        # 价格
    is_high: bool       # True=高点, False=低点
    atr: float          # 当时的ATR值


@dataclass
class Trade:
    """交易数据结构 - 支持 LONG 和 SHORT"""
    entry_idx: int       # 入场索引
    exit_idx: int        # 出场索引
    entry_price: float   # 入场价格
    exit_price: float    # 出场价格
    profit: float        # 利润
    hold_periods: int    # 持仓周期
    is_long: bool        # True=做多, False=做空
    
    @property
    def profit_pct(self) -> float:
        """收益率"""
        if self.is_long:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100


class GodViewLabeler:
    """
    上帝视角标注器
    
    核心算法：
    1. 自适应 ZigZag：基于 ATR 动态调整摆动阈值
    2. 动态规划：选择最大化累计利润的交易序列
    """
    
    def __init__(self, 
                 atr_period: int = None,
                 swing_factor: float = None,
                 swing_window: int = None,
                 min_hold_periods: int = None,
                 max_hold_periods: int = None,
                 min_risk_reward: float = None):
        """
        初始化标注器
        
        Args:
            atr_period: ATR 周期
            swing_factor: 摆动阈值 = swing_factor * ATR
            min_hold_periods: 最小持仓周期
            max_hold_periods: 最大持仓周期
            min_risk_reward: 最小风险收益比
        """
        self.atr_period = atr_period or LABELING_CONFIG["ATR_PERIOD"]
        self.swing_factor = swing_factor or LABELING_CONFIG["SWING_FACTOR"]
        self.swing_window = swing_window or LABELING_CONFIG["SWING_WINDOW"]
        self.min_hold_periods = min_hold_periods or LABELING_CONFIG["MIN_HOLD_PERIODS"]
        self.max_hold_periods = max_hold_periods or LABELING_CONFIG["MAX_HOLD_PERIODS"]
        self.min_risk_reward = min_risk_reward or LABELING_CONFIG["MIN_RISK_REWARD"]
        
        # 标注结果
        self.swing_points: List[SwingPoint] = []
        self.alternating_swings: List[SwingPoint] = []  # 严格交替的高低点序列
        self.optimal_trades: List[Trade] = []
        self.labels: Optional[pd.Series] = None
        
    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        执行上帝视角标注
        
        Args:
            df: K线数据 DataFrame (需包含 open, high, low, close)
            
        Returns:
            标注序列 (BUY=1, SELL=-1, HOLD=0)
        """
        print("[Labeler] 开始上帝视角标注...")
        
        # Phase 1: 基于高低点的本地极值标注（使用未来数据）
        self.swing_points = self._detect_price_swings(df)
        print(f"[Labeler] 检测到 {len(self.swing_points)} 个高低点")
        
        # 直接根据高低点生成标注序列
        self.optimal_trades = []
        self.labels = self._generate_labels_from_swings(df)
        
        # 统计信息（翻转策略：每个标注点同时隐含平仓+开仓）
        long_entry = (self.labels == LabelType.LONG_ENTRY).sum()
        short_entry = (self.labels == LabelType.SHORT_ENTRY).sum()
        total_signals = long_entry + short_entry
        print(f"[Labeler] 标注完成 (翻转策略): LONG_ENTRY={long_entry}, SHORT_ENTRY={short_entry}, "
              f"总信号={total_signals}, 预计交易≈{max(0, total_signals - 1)}笔")
        
        return self.labels

    def _detect_price_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """基于价格局部高低点检测摆动点（使用未来数据）"""
        swings = []
        n = len(df)
        window = max(2, int(self.swing_window))
        if n < window * 2 + 1:
            return swings
        
        high = df['high'].values
        low = df['low'].values
        for i in range(window, n - window):
            hi = high[i]
            lo = low[i]
            if hi >= np.max(high[i - window:i + window + 1]):
                swings.append(SwingPoint(index=i, price=hi, is_high=True, atr=0.0))
            if lo <= np.min(low[i - window:i + window + 1]):
                swings.append(SwingPoint(index=i, price=lo, is_high=False, atr=0.0))
        
        swings.sort(key=lambda s: s.index)
        # 去掉同一根K线上的重复高低点
        dedup = []
        last_idx = -1
        for s in swings:
            if s.index == last_idx:
                continue
            dedup.append(s)
            last_idx = s.index
        return dedup

    def _generate_labels_from_swings(self, df: pd.DataFrame) -> pd.Series:
        """
        根据高低点生成 LONG/SHORT 标注序列 —— 翻转策略
        
        硬约束（绝不违反）：
          低点 → 只标 LONG_ENTRY（低买做多 / 同时隐含空平）
          高点 → 只标 SHORT_ENTRY（高空做空 / 同时隐含多平）
        
        翻转逻辑：
          每个摆动点都翻转仓位方向，保证始终持仓、始终盈利：
            低点 → LONG_ENTRY  (回测器自动平掉空仓 + 开多)
            高点 → SHORT_ENTRY (回测器自动平掉多仓 + 开空)
          
          为了 UI 显示清晰，我们在同一根 K 线上同时标注 EXIT 和 ENTRY。
        """
        n = len(df)
        labels = pd.Series([LabelType.HOLD] * n, index=df.index)
        if len(self.swing_points) < 2:
            return labels
        
        # 先过滤掉连续同向的点（只保留交替的高低点序列）
        alternating = []
        for s in self.swing_points:
            if not alternating:
                alternating.append(s)
            else:
                last = alternating[-1]
                if s.is_high == last.is_high:
                    # 连续同向：高点保留更高的，低点保留更低的
                    if s.is_high and s.price > last.price:
                        alternating[-1] = s
                    elif not s.is_high and s.price < last.price:
                        alternating[-1] = s
                else:
                    alternating.append(s)
        
        if len(alternating) < 2:
            return labels
        
        # 保存交替摆动点序列供市场状态分类器使用
        self.alternating_swings = list(alternating)
        
        # 翻转标注：低点标 LONG_ENTRY，高点标 SHORT_ENTRY
        # 为了让 UI 显示 EXIT，我们需要在回测逻辑或标注逻辑中明确它
        # 这里我们保持 label 序列简洁，但在 UI 渲染时根据逻辑显示 EXIT
        for i, s in enumerate(alternating):
            if s.is_high:
                labels.iloc[s.index] = LabelType.SHORT_ENTRY
            else:
                labels.iloc[s.index] = LabelType.LONG_ENTRY
        
        return labels
    
    def _detect_zigzag_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        自适应 ZigZag 摆动检测
        
        使用动态 ATR 阈值检测价格的高低点
        """
        swings = []
        n = len(df)
        
        if n < self.atr_period + 10:
            return swings
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = df['atr'].values
        
        # 初始化
        last_swing_idx = self.atr_period
        last_swing_price = close[last_swing_idx]
        last_swing_is_high = True  # 从假设高点开始
        
        # 寻找第一个有效摆动点
        for i in range(self.atr_period, min(self.atr_period + 50, n)):
            threshold = atr[i] * self.swing_factor
            if high[i] - low[i] > threshold:
                last_swing_idx = i
                if close[i] > close[i-1]:
                    last_swing_is_high = True
                    last_swing_price = high[i]
                else:
                    last_swing_is_high = False
                    last_swing_price = low[i]
                break
        
        # 遍历检测摆动点
        for i in range(last_swing_idx + 1, n):
            threshold = atr[i] * self.swing_factor
            
            if np.isnan(threshold) or threshold <= 0:
                threshold = close[i] * 0.01  # 默认1%
            
            if last_swing_is_high:
                # 上一个是高点，寻找低点
                if low[i] < last_swing_price - threshold:
                    # 确认上一个高点
                    swings.append(SwingPoint(
                        index=last_swing_idx,
                        price=last_swing_price,
                        is_high=True,
                        atr=atr[last_swing_idx] if not np.isnan(atr[last_swing_idx]) else threshold
                    ))
                    # 更新为当前低点
                    last_swing_idx = i
                    last_swing_price = low[i]
                    last_swing_is_high = False
                elif high[i] > last_swing_price:
                    # 更新高点
                    last_swing_idx = i
                    last_swing_price = high[i]
            else:
                # 上一个是低点，寻找高点
                if high[i] > last_swing_price + threshold:
                    # 确认上一个低点
                    swings.append(SwingPoint(
                        index=last_swing_idx,
                        price=last_swing_price,
                        is_high=False,
                        atr=atr[last_swing_idx] if not np.isnan(atr[last_swing_idx]) else threshold
                    ))
                    # 更新为当前高点
                    last_swing_idx = i
                    last_swing_price = high[i]
                    last_swing_is_high = True
                elif low[i] < last_swing_price:
                    # 更新低点
                    last_swing_idx = i
                    last_swing_price = low[i]
        
        # 添加最后一个摆动点
        if last_swing_idx < n - 1:
            swings.append(SwingPoint(
                index=last_swing_idx,
                price=last_swing_price,
                is_high=last_swing_is_high,
                atr=atr[last_swing_idx] if not np.isnan(atr[last_swing_idx]) else close[last_swing_idx] * 0.01
            ))
        
        return swings
    
    def _dp_optimize_trades(self, df: pd.DataFrame) -> List[Trade]:
        """
        动态规划选择最优交易序列
        
        从摆动点中选择使累计利润最大化的 LONG 和 SHORT 交易配对
        """
        if len(self.swing_points) < 2:
            return []
        
        # 提取低点和高点
        lows = [s for s in self.swing_points if not s.is_high]
        highs = [s for s in self.swing_points if s.is_high]
        
        if not lows or not highs:
            return []
        
        # 构建所有可能的有效交易（包括 LONG 和 SHORT）
        valid_trades = []
        
        # LONG 交易：低点买入 → 高点卖出
        for low in lows:
            for high in highs:
                # 高点必须在低点之后
                if high.index <= low.index:
                    continue
                
                # 检查持仓周期约束
                hold_periods = high.index - low.index
                if hold_periods < self.min_hold_periods:
                    continue
                if hold_periods > self.max_hold_periods:
                    continue
                
                # 计算利润
                profit = high.price - low.price
                
                # 检查最小收益要求
                risk = low.atr if low.atr > 0 else low.price * 0.01
                
                if profit > 0 and profit / risk >= self.min_risk_reward:
                    valid_trades.append(Trade(
                        entry_idx=low.index,
                        exit_idx=high.index,
                        entry_price=low.price,
                        exit_price=high.price,
                        profit=profit,
                        hold_periods=hold_periods,
                        is_long=True
                    ))
        
        # SHORT 交易：高点做空 → 低点平仓
        for high in highs:
            for low in lows:
                # 低点必须在高点之后
                if low.index <= high.index:
                    continue
                
                # 检查持仓周期约束
                hold_periods = low.index - high.index
                if hold_periods < self.min_hold_periods:
                    continue
                if hold_periods > self.max_hold_periods:
                    continue
                
                # 计算利润（做空：高卖低买）
                profit = high.price - low.price
                
                # 检查最小收益要求
                risk = high.atr if high.atr > 0 else high.price * 0.01
                
                if profit > 0 and profit / risk >= self.min_risk_reward:
                    valid_trades.append(Trade(
                        entry_idx=high.index,
                        exit_idx=low.index,
                        entry_price=high.price,
                        exit_price=low.price,
                        profit=profit,
                        hold_periods=hold_periods,
                        is_long=False
                    ))
        
        if not valid_trades:
            return []
        
        # 按入场时间排序
        valid_trades.sort(key=lambda t: t.entry_idx)
        
        # 动态规划：选择不重叠的最优交易序列
        # dp[i] = 以第i笔交易结束时的最大累计利润
        n = len(valid_trades)
        dp = [0.0] * n
        parent = [-1] * n  # 用于回溯
        
        for i in range(n):
            # 只选择这一笔交易
            dp[i] = valid_trades[i].profit
            parent[i] = -1
            
            # 或者接在某个不重叠的交易之后
            for j in range(i):
                # 检查不重叠：j的出场点在i的入场点之前
                if valid_trades[j].exit_idx < valid_trades[i].entry_idx:
                    if dp[j] + valid_trades[i].profit > dp[i]:
                        dp[i] = dp[j] + valid_trades[i].profit
                        parent[i] = j
        
        # 找到最优结束点
        max_profit_idx = np.argmax(dp)
        
        # 回溯构建最优交易序列
        optimal = []
        idx = max_profit_idx
        while idx != -1:
            optimal.append(valid_trades[idx])
            idx = parent[idx]
        
        optimal.reverse()
        
        # 统计
        long_count = sum(1 for t in optimal if t.is_long)
        short_count = sum(1 for t in optimal if not t.is_long)
        total_profit = sum(t.profit for t in optimal)
        avg_hold = np.mean([t.hold_periods for t in optimal]) if optimal else 0
        print(f"[Labeler] 最优交易序列: {long_count} LONG + {short_count} SHORT, 累计利润 ${total_profit:.2f}, 平均持仓 {avg_hold:.1f} 周期")
        
        return optimal
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        根据最优交易生成标注序列 - LONG/SHORT/EXIT 系统
        """
        n = len(df)
        labels = pd.Series([LabelType.HOLD] * n, index=df.index)
        
        for trade in self.optimal_trades:
            if trade.is_long:
                labels.iloc[trade.entry_idx] = LabelType.LONG_ENTRY
                labels.iloc[trade.exit_idx] = LabelType.LONG_EXIT
            else:
                labels.iloc[trade.entry_idx] = LabelType.SHORT_ENTRY
                labels.iloc[trade.exit_idx] = LabelType.SHORT_EXIT
        
        return labels
    
    def get_statistics(self) -> dict:
        """获取标注统计信息"""
        if not self.optimal_trades:
            return {}
        
        long_trades = [t for t in self.optimal_trades if t.is_long]
        short_trades = [t for t in self.optimal_trades if not t.is_long]
        
        profits = [t.profit for t in self.optimal_trades]
        profit_pcts = [t.profit_pct for t in self.optimal_trades]
        hold_periods = [t.hold_periods for t in self.optimal_trades]
        
        return {
            "total_trades": len(self.optimal_trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "total_profit": sum(profits),
            "avg_profit": np.mean(profits),
            "avg_profit_pct": np.mean(profit_pcts),
            "max_profit_pct": max(profit_pcts),
            "min_profit_pct": min(profit_pcts),
            "avg_hold_periods": np.mean(hold_periods),
            "max_hold_periods": max(hold_periods),
            "min_hold_periods": min(hold_periods),
            "swing_points_count": len(self.swing_points),
        }
    
    def get_labeled_points(self) -> dict:
        """
        获取标注点的索引
        
        Returns:
            dict with 'long_entry', 'long_exit', 'short_entry', 'short_exit' lists
        """
        long_entry = [t.entry_idx for t in self.optimal_trades if t.is_long]
        long_exit = [t.exit_idx for t in self.optimal_trades if t.is_long]
        short_entry = [t.entry_idx for t in self.optimal_trades if not t.is_long]
        short_exit = [t.exit_idx for t in self.optimal_trades if not t.is_long]
        return {
            'long_entry': long_entry,
            'long_exit': long_exit,
            'short_entry': short_entry,
            'short_exit': short_exit
        }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据（模拟价格走势）
    np.random.seed(42)
    n = 5000
    
    # 生成带有趋势和波动的价格数据
    trend = np.cumsum(np.random.randn(n) * 0.5)
    noise = np.random.randn(n) * 20
    base_price = 50000 + trend * 50 + noise
    
    test_df = pd.DataFrame({
        'open': base_price,
        'close': base_price + np.random.randn(n) * 10,
    })
    test_df['high'] = test_df[['open', 'close']].max(axis=1) + abs(np.random.randn(n) * 15)
    test_df['low'] = test_df[['open', 'close']].min(axis=1) - abs(np.random.randn(n) * 15)
    test_df['volume'] = np.random.randint(100, 10000, n)
    
    # 执行标注
    labeler = GodViewLabeler()
    labels = labeler.label(test_df)
    
    # 打印统计
    stats = labeler.get_statistics()
    print(f"\n标注统计:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 获取标注点
    points = labeler.get_labeled_points()
    print(f"\nLONG 入场索引: {points['long_entry'][:5]}...")
    print(f"LONG 出场索引: {points['long_exit'][:5]}...")
    print(f"SHORT 入场索引: {points['short_entry'][:5]}...")
    print(f"SHORT 出场索引: {points['short_exit'][:5]}...")
