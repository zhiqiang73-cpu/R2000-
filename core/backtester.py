"""
R3000 回测引擎
用于 GA 适应度评估，支持正向递归回测（不使用未来数据）
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BACKTEST_CONFIG, PAPER_TRADING_CONFIG


class PositionSide(IntEnum):
    """持仓方向"""
    NONE = 0
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    """持仓信息"""
    side: PositionSide = PositionSide.NONE
    entry_price: float = 0.0
    entry_idx: int = 0
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    take_profit_2: float = 0.0      # 保留字段（不再使用，阶段2改为追踪止损）
    tp1_ratio: float = 0.70         # TP1 平仓比例
    stage: int = 1                  # 当前阶段：1=初始，2=TP1已触发
    liquidation_price: float = 0.0  # 强平价格
    margin: float = 0.0             # 保证金
    peak_pnl_pct: float = 0.0       # 持仓期间峰值浮盈（杠杆后%）
    ever_reached_6pct: bool = False  # 是否曾达到+BREAKEVEN_THRESHOLD_PCT
    original_stop_loss: float = 0.0  # 原始止损价（用于止损参考）
    peak_price: float = 0.0         # 阶段2追踪最高/最低价
    signal_key: str = ""            # combo_key，用于日志和追溯
    avg_hold_bars: int = 0          # 策略平均持仓，0=使用全局配置


@dataclass
class TradeRecord:
    """交易记录"""
    entry_idx: int
    exit_idx: int
    side: int              # 1=多, -1=空
    entry_price: float
    exit_price: float
    size: float
    profit: float
    profit_pct: float
    hold_periods: int
    exit_reason: str       # "tp", "sl", "signal", "timeout"
    stop_loss: float = 0.0
    take_profit: float = 0.0
    liquidation_price: float = 0.0
    margin: float = 0.0
    market_regime: str = ""  # 市场状态分类
    # ABC 向量坐标
    entry_abc: tuple = (0.0, 0.0, 0.0)  # 入场时的 (A, B, C) 坐标
    exit_abc: tuple = (0.0, 0.0, 0.0)   # 离场时的 (A, B, C) 坐标
    # 轨迹匹配字段
    pre_entry_traj: object = None       # (60, 32) 入场前轨迹 np.ndarray
    holding_traj: object = None         # (hold, 32) 持仓轨迹 np.ndarray
    pre_exit_traj: object = None        # (30, 32) 离场前轨迹 np.ndarray
    matched_template_idx: int = -1      # 匹配到的模板编号
    entry_similarity: float = 0.0       # 入场匹配度 (DTW相似度)
    # 信号回测附加信息
    signal_key: str = ""                # 累计结果编号（combo key）
    signal_rate: float = 0.0            # 综合命中率 overall_rate
    signal_score: float = 0.0          # 综合评分
    avg_hold_bars: int = 0              # 策略平均持仓（用于时间衰减）
    pool_id: str = ""                   # 来源池：pool1 / pool2


@dataclass
class BacktestResult:
    """回测结果"""
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    total_profit: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_profit_pct: float = 0.0
    avg_hold_periods: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    long_trades: int = 0
    long_win_rate: float = 0.0
    long_profit: float = 0.0
    short_trades: int = 0
    short_win_rate: float = 0.0
    short_profit: float = 0.0
    initial_capital: float = 0.0
    current_capital: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    current_pos: Optional[Position] = None  # 当前持仓信息
    trade_logs: List[dict] = field(default_factory=list)


class Backtester:
    """
    回测器
    
    特点：
    1. 严格正向递归，不使用未来数据
    2. 支持做多/做空
    3. 支持止盈止损
    4. 计算各种绩效指标
    """
    
    def __init__(self, 
                 initial_capital: float = None,
                 leverage: float = None,
                 fee_rate: float = None,
                 slippage: float = None,
                 position_size_pct: float = None,
                 stop_loss_pct: float = None,
                 take_profit_pct: float = None):
        """
        初始化回测器
        """
        self.initial_capital = initial_capital or BACKTEST_CONFIG["INITIAL_CAPITAL"]
        self.leverage = leverage or BACKTEST_CONFIG["LEVERAGE"]
        self.fee_rate = fee_rate or BACKTEST_CONFIG["FEE_RATE"]
        self.slippage = slippage or BACKTEST_CONFIG["SLIPPAGE"]
        self.position_size_pct = position_size_pct or BACKTEST_CONFIG["POSITION_SIZE_PCT"]
        self.stop_loss_pct = stop_loss_pct or BACKTEST_CONFIG["STOP_LOSS_PCT"]
        self.take_profit_pct = take_profit_pct or BACKTEST_CONFIG["TAKE_PROFIT_PCT"]
        
        # 状态
        self.capital = self.initial_capital
        self.position = Position()
        self.equity_curve = []
        self.trades: List[TradeRecord] = []
        self.trade_logs: List[dict] = []
        self._reset_stats()
        
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = Position()
        self.equity_curve = []
        self.trades = []
        self.trade_logs = []
        self._reset_stats()

    def _log_event(self, event: str, idx: int, side: Optional[PositionSide],
                   price: float, info: Optional[dict] = None) -> None:
        """记录回测关键事件（entry / tp1 / sl / trailing_sl / signal / end）"""
        side_map = {PositionSide.LONG: "LONG", PositionSide.SHORT: "SHORT"}
        side_text = side_map.get(side, "") if side is not None else ""
        self.trade_logs.append({
            "idx": idx,
            "event": event,
            "side": side_text,
            "price": price,
            "info": info or {},
        })

    def _reset_stats(self):
        """重置统计缓存"""
        self._total_trades = 0
        self._win_trades = 0
        self._loss_trades = 0
        self._total_profit = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._long_trades = 0
        self._short_trades = 0
        self._long_win_trades = 0
        self._short_win_trades = 0
        self._long_profit = 0.0
        self._short_profit = 0.0
        self._profit_pcts = []
        self._hold_periods = []
        self._peak_equity = self.initial_capital
        self._max_drawdown = 0.0
    
    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """
        执行回测
        
        Args:
            df: K线数据 (需包含 open, high, low, close)
            signals: 信号序列 (1=买入, -1=卖出, 0=持有)
            
        Returns:
            回测结果
        """
        self.reset()
        
        n = len(df)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        signals_arr = signals.values if isinstance(signals, pd.Series) else signals
        
        for i in range(n):
            current_price = close[i]
            current_high = high[i]
            current_low = low[i]
            signal = signals_arr[i]
            
            # 检查止盈止损
            if self.position.side != PositionSide.NONE:
                exit_triggered, exit_price, exit_reason = self._check_exit(
                    current_high, current_low, current_price, i
                )

                if exit_triggered:
                    self._close_position(i, exit_price, exit_reason)

            # 处理新信号
            if signal != 0:
                if self.position.side == PositionSide.NONE:
                    # 开仓
                    side = PositionSide.LONG if signal > 0 else PositionSide.SHORT
                    self._open_position(i, current_price, side, reason="signal", meta={"signal": float(signal)})
                    
                elif (signal > 0 and self.position.side == PositionSide.SHORT) or \
                     (signal < 0 and self.position.side == PositionSide.LONG):
                    # 反向信号，平仓后反向开仓
                    self._close_position(i, current_price, "signal")
                    side = PositionSide.LONG if signal > 0 else PositionSide.SHORT
                    self._open_position(i, current_price, side, reason="signal_reverse", meta={"signal": float(signal)})
            
            # 记录权益
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)
        
        # 强制平仓剩余持仓
        if self.position.side != PositionSide.NONE:
            self._close_position(n-1, close[-1], "end")
        
        # 计算统计指标
        result = self._calculate_statistics()
        
        return result

    def run_with_labels(self, df: pd.DataFrame, labels: pd.Series) -> BacktestResult:
        """
        使用标注序列回测（LONG/SHORT/EXIT）
        labels: 1=LONG_ENTRY, 2=LONG_EXIT, -1=SHORT_ENTRY, -2=SHORT_EXIT
        """
        self.reset()
        n = len(df)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        labels_arr = labels.values if isinstance(labels, pd.Series) else labels

        for i in range(n):
            label = int(labels_arr[i])

            if label == 1:  # LONG_ENTRY
                if self.position.side == PositionSide.SHORT:
                    self._close_position(i, low[i], "signal")
                if self.position.side == PositionSide.NONE:
                    self._open_position(i, low[i], PositionSide.LONG, reason="label_long_entry")
            elif label == -1:  # SHORT_ENTRY
                if self.position.side == PositionSide.LONG:
                    self._close_position(i, high[i], "signal")
                if self.position.side == PositionSide.NONE:
                    self._open_position(i, high[i], PositionSide.SHORT, reason="label_short_entry")
            elif label == 2:  # LONG_EXIT
                if self.position.side == PositionSide.LONG:
                    self._close_position(i, high[i], "signal")
            elif label == -2:  # SHORT_EXIT
                if self.position.side == PositionSide.SHORT:
                    self._close_position(i, low[i], "signal")

            equity = self._calculate_equity(close[i])
            self.equity_curve.append(equity)

        if self.position.side != PositionSide.NONE:
            self._close_position(n - 1, close[-1], "end")

        return self._calculate_statistics()

    def step_with_label(self, idx: int, close: float, high: float, low: float, label: int) -> BacktestResult:
        """逐步回测（实时统计）"""
        label = int(label)

        if label == 1:  # LONG_ENTRY
            if self.position.side == PositionSide.SHORT:
                self._close_position(idx, low, "signal")
            if self.position.side == PositionSide.NONE:
                self._open_position(idx, low, PositionSide.LONG, reason="label_long_entry")
        elif label == -1:  # SHORT_ENTRY
            if self.position.side == PositionSide.LONG:
                self._close_position(idx, high, "signal")
            if self.position.side == PositionSide.NONE:
                self._open_position(idx, high, PositionSide.SHORT, reason="label_short_entry")
        elif label == 2:  # LONG_EXIT
            if self.position.side == PositionSide.LONG:
                self._close_position(idx, high, "signal")
        elif label == -2:  # SHORT_EXIT
            if self.position.side == PositionSide.SHORT:
                self._close_position(idx, low, "signal")

        equity = self._calculate_equity(close)
        self.equity_curve.append(equity)
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, drawdown)

        return self.get_realtime_result()

    def get_realtime_result(self) -> BacktestResult:
        """获取实时统计结果"""
        result = BacktestResult()
        result.equity_curve = self.equity_curve.copy()
        result.trades = self.trades.copy()
        result.trade_logs = self.trade_logs.copy()
        result.initial_capital = self.initial_capital
        result.current_capital = self.capital
        result.current_pos = self.position if self.position.side != PositionSide.NONE else None

        result.total_trades = self._total_trades
        result.win_trades = self._win_trades
        result.loss_trades = self._loss_trades
        result.total_profit = self._total_profit
        result.total_return_pct = (self.equity_curve[-1] / self.initial_capital - 1) * 100 if self.equity_curve else 0
        result.max_drawdown = self._max_drawdown
        result.win_rate = (self._win_trades / self._total_trades) if self._total_trades > 0 else 0
        result.avg_profit_pct = float(np.mean(self._profit_pcts)) if self._profit_pcts else 0
        result.avg_hold_periods = float(np.mean(self._hold_periods)) if self._hold_periods else 0
        result.gross_profit = self._gross_profit
        result.gross_loss = self._gross_loss
        result.avg_win = (self._gross_profit / self._win_trades) if self._win_trades > 0 else 0
        result.avg_loss = (self._gross_loss / self._loss_trades) if self._loss_trades > 0 else 0
        result.profit_factor = self._gross_profit / self._gross_loss if self._gross_loss > 0 else float('inf')
        result.long_trades = self._long_trades
        result.short_trades = self._short_trades
        result.long_profit = self._long_profit
        result.short_profit = self._short_profit
        result.long_win_rate = (self._long_win_trades / self._long_trades) if self._long_trades > 0 else 0
        result.short_win_rate = (self._short_win_trades / self._short_trades) if self._short_trades > 0 else 0

        # 夏普比率（按交易收益率估算）
        if len(self._profit_pcts) > 1:
            returns = np.array(self._profit_pcts)
            if returns.std() > 0:
                result.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))
            else:
                result.sharpe_ratio = 0.0
        else:
            result.sharpe_ratio = 0.0

        return result
    
    def run_with_strategy(self, df: pd.DataFrame, features: np.ndarray,
                          strategy_weights: np.ndarray,
                          long_threshold: float, short_threshold: float,
                          max_hold: int = 60) -> BacktestResult:
        """
        使用策略权重运行回测
        
        Args:
            df: K线数据
            features: 特征矩阵 (n_samples, n_features)
            strategy_weights: 策略权重 (n_features,) 或 (n_features*2,) 包含多空权重
            long_threshold: 做多阈值
            short_threshold: 做空阈值
            max_hold: 最大持仓周期
            
        Returns:
            回测结果
        """
        self.reset()
        
        n = len(df)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        n_features = features.shape[1]
        
        # 解析权重
        if len(strategy_weights) == n_features * 2:
            long_weights = strategy_weights[:n_features]
            short_weights = strategy_weights[n_features:]
        else:
            long_weights = strategy_weights
            short_weights = -strategy_weights
        
        cfg_ptc = PAPER_TRADING_CONFIG
        lev = BACKTEST_CONFIG.get("LEVERAGE", 10)
        breakeven_thr = cfg_ptc.get("BREAKEVEN_THRESHOLD_PCT", 6.0)   # 杠杆后 %
        breakeven_sl  = cfg_ptc.get("BREAKEVEN_SL_PCT", -3.0)         # 杠杆后 %

        for i in range(n):
            current_price = close[i]
            current_high = high[i]
            current_low = low[i]
            
            # 检查持仓超时
            if self.position.side != PositionSide.NONE:
                hold_time = i - self.position.entry_idx
                if hold_time >= max_hold:
                    self._close_position(i, current_price, "timeout")
                    continue
            
            # 每根K线更新持仓浮盈峰值，并触发阶梯止损上移
            if self.position.side != PositionSide.NONE and self.position.stage == 1:
                pos = self.position
                is_long = (pos.side == PositionSide.LONG)
                price_chg_pct = (current_price - pos.entry_price) / pos.entry_price
                if not is_long:
                    price_chg_pct = -price_chg_pct
                pnl_pct = price_chg_pct * lev * 100  # 杠杆后收益率%

                if pnl_pct > pos.peak_pnl_pct:
                    pos.peak_pnl_pct = pnl_pct

                # 阶梯止损上移：浮盈首次达到阈值 → 止损移到 BREAKEVEN_SL_PCT
                if not pos.ever_reached_6pct and pos.peak_pnl_pct >= breakeven_thr:
                    pos.ever_reached_6pct = True
                    sl_price_chg = breakeven_sl / 100 / lev  # 价格变动幅度
                    new_sl = pos.entry_price * (1 + sl_price_chg) if is_long else pos.entry_price * (1 - sl_price_chg)
                    # 止损只能上移（对多头：新止损 > 当前止损；对空头：新止损 < 当前止损）
                    if is_long:
                        pos.stop_loss = max(pos.stop_loss, new_sl)
                    else:
                        pos.stop_loss = min(pos.stop_loss, new_sl)

            # 检查止盈止损
            if self.position.side != PositionSide.NONE:
                exit_triggered, exit_price, exit_reason = self._check_exit(
                    current_high, current_low, current_price, i
                )
                if exit_triggered:
                    self._close_position(i, exit_price, exit_reason)

            # 计算信号分数
            feat = features[i]
            long_score = np.dot(feat, long_weights)
            short_score = np.dot(feat, short_weights)
            
            # 判断信号
            if self.position.side == PositionSide.NONE:
                if long_score > long_threshold and long_score > short_score:
                    self._open_position(i, current_price, PositionSide.LONG,
                                        reason="score_long",
                                        meta={"long_score": float(long_score), "short_score": float(short_score),
                                              "long_threshold": float(long_threshold), "short_threshold": float(short_threshold)})
                elif short_score > short_threshold and short_score > long_score:
                    self._open_position(i, current_price, PositionSide.SHORT,
                                        reason="score_short",
                                        meta={"long_score": float(long_score), "short_score": float(short_score),
                                              "long_threshold": float(long_threshold), "short_threshold": float(short_threshold)})
            
            elif self.position.side == PositionSide.LONG:
                if short_score > short_threshold:
                    self._close_position(i, current_price, "signal")
                    self._open_position(i, current_price, PositionSide.SHORT,
                                        reason="score_short_reverse",
                                        meta={"long_score": float(long_score), "short_score": float(short_score),
                                              "short_threshold": float(short_threshold)})
            
            elif self.position.side == PositionSide.SHORT:
                if long_score > long_threshold:
                    self._close_position(i, current_price, "signal")
                    self._open_position(i, current_price, PositionSide.LONG,
                                        reason="score_long_reverse",
                                        meta={"long_score": float(long_score), "short_score": float(short_score),
                                              "long_threshold": float(long_threshold)})
            
            # 记录权益
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)
        
        # 强制平仓
        if self.position.side != PositionSide.NONE:
            self._close_position(n-1, close[-1], "end")
        
        return self._calculate_statistics()
    
    def _open_position(self, idx: int, price: float, side: PositionSide,
                       reason: str = "", meta: Optional[dict] = None,
                       signal_key: str = "", avg_hold_bars: int = 0):
        """开仓（追踪止盈止损框架：TP1=70%平仓进入阶段2，阶段2用追踪止损，SL全平）"""
        if meta:
            signal_key = meta.get("signal_key", signal_key)
            avg_hold_bars = meta.get("avg_hold_bars", avg_hold_bars)
        tp1_pct = BACKTEST_CONFIG.get("TAKE_PROFIT_PCT", 0.012)    # TP1 价格变动 1.2%（+12%杠杆后）
        sl_pct  = BACKTEST_CONFIG.get("STOP_LOSS_PCT", 0.008)      # SL  价格变动 0.8%
        tp1_ratio = BACKTEST_CONFIG.get("TP1_RATIO", 0.70)         # TP1 平仓 70%
        pos_size_pct = BACKTEST_CONFIG.get("POSITION_SIZE_PCT", self.position_size_pct)

        # 计算滑点后的入场价
        if side == PositionSide.LONG:
            entry_price = price * (1 + self.slippage)
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp1_pct)
            liquidation_price = entry_price * (1 - 1 / self.leverage * 0.9)
        else:
            entry_price = price * (1 - self.slippage)
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp1_pct)
            liquidation_price = entry_price * (1 + 1 / self.leverage * 0.9)

        # 计算仓位大小
        margin = self.capital * pos_size_pct
        position_value = margin * self.leverage
        size = position_value / entry_price

        # 扣除手续费
        fee = position_value * self.fee_rate
        self.capital -= fee

        self.position = Position(
            side=side,
            entry_price=entry_price,
            entry_idx=idx,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=0.0,
            tp1_ratio=tp1_ratio,
            stage=1,
            liquidation_price=liquidation_price,
            margin=margin,
            peak_pnl_pct=0.0,
            ever_reached_6pct=False,
            original_stop_loss=stop_loss,
            peak_price=entry_price,
            signal_key=signal_key,
            avg_hold_bars=avg_hold_bars,
        )
        self._log_event("entry", idx, side, entry_price, {
            "reason": reason,
            "meta": meta or {},
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": self.leverage,
            "size": size,
            "margin": margin,
        })
    
    def _close_position(self, idx: int, price: float, reason: str):
        """平仓"""
        if self.position.side == PositionSide.NONE:
            return
        
        # 计算滑点后的出场价
        if self.position.side == PositionSide.LONG:
            exit_price = price * (1 - self.slippage)
            profit = (exit_price - self.position.entry_price) * self.position.size
        else:
            exit_price = price * (1 + self.slippage)
            profit = (self.position.entry_price - exit_price) * self.position.size
        
        # 扣除手续费
        position_value = exit_price * self.position.size
        fee = position_value * self.fee_rate
        
        # 更新资金
        net_profit = profit - fee
        self.capital += net_profit
        
        # 记录交易
        profit_pct = (exit_price - self.position.entry_price) / self.position.entry_price * 100
        if self.position.side == PositionSide.SHORT:
            profit_pct = -profit_pct
        
        self.trades.append(TradeRecord(
            entry_idx=self.position.entry_idx,
            exit_idx=idx,
            side=int(self.position.side),
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            profit=net_profit,
            profit_pct=profit_pct,
            hold_periods=idx - self.position.entry_idx,
            exit_reason=reason,
            stop_loss=self.position.stop_loss,
            take_profit=self.position.take_profit,
            liquidation_price=self.position.liquidation_price,
            margin=self.position.margin,
            signal_key=self.position.signal_key,
            avg_hold_bars=self.position.avg_hold_bars,
        ))

        # 更新统计缓存
        self._total_trades += 1
        if net_profit > 0:
            self._win_trades += 1
            self._gross_profit += net_profit
        else:
            self._loss_trades += 1
            self._gross_loss += abs(net_profit)
        self._total_profit += net_profit
        self._profit_pcts.append(profit_pct)
        self._hold_periods.append(idx - self.position.entry_idx)

        if self.position.side == PositionSide.LONG:
            self._long_trades += 1
            self._long_profit += net_profit
            if net_profit > 0:
                self._long_win_trades += 1
        elif self.position.side == PositionSide.SHORT:
            self._short_trades += 1
            self._short_profit += net_profit
            if net_profit > 0:
                self._short_win_trades += 1

        # 记录平仓日志（sl / trailing_sl / signal / end）
        self._log_event(reason, idx, self.position.side, exit_price, {
            "profit_pct": round(profit_pct, 2),
            "hold_bars": idx - self.position.entry_idx,
            "entry_price": round(self.position.entry_price, 2),
            "stop_loss": self.position.stop_loss,
            "take_profit": self.position.take_profit,
        })

        # 清空持仓
        self.position = Position()
    
    def _check_exit(self, high: float, low: float, close: float,
                    current_idx: int = -1) -> Tuple[bool, float, str]:
        """
        追踪止盈止损检查。

        阶段1（stage=1）：
          - 动态止损：浮盈>=BREAKEVEN_THRESHOLD_PCT → 上移止损到BREAKEVEN_SL_PCT
          - 时间衰减：持仓>TIME_DECAY_BAR_2 → 止损收窄到TIME_DECAY_SL_2
                      持仓>TIME_DECAY_BAR_1 → 止损收窄到TIME_DECAY_SL_1
          - SL 触发 → 全平
          - TP1 触发（+STAGED_TP_PCT%）→ 部分平仓（tp1_ratio），止损移至入场价，进入阶段2

        阶段2（stage=2）：
          - 追踪止损 = 最高价×(1−TRAILING_STOP_PCT) [多] / 最低价×(1+TRAILING_STOP_PCT) [空]
          - 止损只升不降（多）/只降不升（空）
          - 追踪止损触发 → 全平剩余（无TP2上限，让利润奔跑）
        """
        pos = self.position
        if pos.side == PositionSide.NONE:
            return False, 0, ""

        cfg = BACKTEST_CONFIG
        lev = cfg.get("LEVERAGE", 10)
        is_long = (pos.side == PositionSide.LONG)
        old_sl = pos.stop_loss

        if pos.stage == 1:
            hold_bars = current_idx - pos.entry_idx if current_idx >= 0 else 0

            # ── 更新峰值浮盈（杠杆后%）──
            best_bar_price = high if is_long else low
            if is_long:
                bar_pnl = (best_bar_price - pos.entry_price) / pos.entry_price * lev * 100
            else:
                bar_pnl = (pos.entry_price - best_bar_price) / pos.entry_price * lev * 100
            if bar_pnl > pos.peak_pnl_pct:
                pos.peak_pnl_pct = bar_pnl

            # ── 动态止损调整（优先级：阶梯上移 > 时间衰减）──
            threshold = cfg.get("BREAKEVEN_THRESHOLD_PCT", 6.0)
            if not pos.ever_reached_6pct and pos.peak_pnl_pct >= threshold:
                # 第一次浮盈达到阈值：止损上移到 BREAKEVEN_SL_PCT 对应价格
                pos.ever_reached_6pct = True
                sl_pct = cfg.get("BREAKEVEN_SL_PCT", -3.0) / 100 / lev
                if is_long:
                    pos.stop_loss = max(pos.stop_loss, pos.entry_price * (1 + sl_pct))
                else:
                    pos.stop_loss = min(pos.stop_loss, pos.entry_price * (1 - sl_pct))
                if pos.stop_loss != old_sl:
                    old_sl = pos.stop_loss

            elif not pos.ever_reached_6pct:
                # 未触发阶梯上移时：时间衰减止损收窄（止损只能收紧，不能放宽）
                decay_enabled = cfg.get("TIME_DECAY_ENABLED", True)
                if decay_enabled:
                    if pos.avg_hold_bars >= 10:
                        bar1 = int(pos.avg_hold_bars * 0.50)
                        bar2 = int(pos.avg_hold_bars * 0.70)
                    else:
                        bar1 = cfg.get("TIME_DECAY_BAR_1", 30)
                        bar2 = cfg.get("TIME_DECAY_BAR_2", 60)
                    sl2_pct = cfg.get("TIME_DECAY_SL_2", -5.0) / 100 / lev
                    sl1_pct = cfg.get("TIME_DECAY_SL_1", -10.0) / 100 / lev
                    if hold_bars > bar2:
                        decay_sl = pos.entry_price * (1 + sl2_pct) if is_long else pos.entry_price * (1 - sl2_pct)
                        if is_long:
                            pos.stop_loss = max(pos.stop_loss, decay_sl)
                        else:
                            pos.stop_loss = min(pos.stop_loss, decay_sl)
                        if pos.stop_loss != old_sl:
                            old_sl = pos.stop_loss
                    elif hold_bars > bar1:
                        decay_sl = pos.entry_price * (1 + sl1_pct) if is_long else pos.entry_price * (1 - sl1_pct)
                        if is_long:
                            pos.stop_loss = max(pos.stop_loss, decay_sl)
                        else:
                            pos.stop_loss = min(pos.stop_loss, decay_sl)
                        if pos.stop_loss != old_sl:
                            old_sl = pos.stop_loss

            # 检查 SL 触发（全平）
            if is_long and low <= pos.stop_loss:
                return True, pos.stop_loss, "sl"
            if not is_long and high >= pos.stop_loss:
                return True, pos.stop_loss, "sl"

            # 检查 TP1 触发（部分平仓，转入阶段2）
            if is_long and high >= pos.take_profit:
                self._partial_close_position(pos.take_profit, pos.tp1_ratio, "tp1", current_idx)
                pos.stop_loss = pos.entry_price  # 止损上移至保本
                pos.stage = 2
                pos.peak_price = pos.take_profit  # 初始化追踪最高价
                return False, 0, ""
            if not is_long and low <= pos.take_profit:
                self._partial_close_position(pos.take_profit, pos.tp1_ratio, "tp1", current_idx)
                pos.stop_loss = pos.entry_price
                pos.stage = 2
                pos.peak_price = pos.take_profit  # 初始化追踪最低价
                return False, 0, ""

        elif pos.stage == 2:
            trail_pct = cfg.get("TRAILING_STOP_PCT", 0.08)

            # 更新追踪最高/最低价，计算并更新追踪止损（止损只能收紧）
            if is_long:
                if high > pos.peak_price:
                    pos.peak_price = high
                trailing_sl = pos.peak_price * (1 - trail_pct)
                pos.stop_loss = max(pos.stop_loss, trailing_sl)
                if pos.stop_loss != old_sl:
                    old_sl = pos.stop_loss
                if low <= pos.stop_loss:
                    return True, pos.stop_loss, "trailing_sl"
            else:
                if pos.peak_price == 0 or low < pos.peak_price:
                    pos.peak_price = low
                trailing_sl = pos.peak_price * (1 + trail_pct)
                pos.stop_loss = min(pos.stop_loss, trailing_sl)
                if pos.stop_loss != old_sl:
                    old_sl = pos.stop_loss
                if high >= pos.stop_loss:
                    return True, pos.stop_loss, "trailing_sl"

        return False, 0, ""

    def _partial_close_position(self, price: float, ratio: float, reason: str,
                                 idx: int = -1) -> None:
        """
        按比例部分平仓，更新仓位 size 和资金，记录部分平仓盈亏。
        剩余仓位继续持有。
        """
        pos = self.position
        if pos.side == PositionSide.NONE or ratio <= 0:
            return

        close_size = pos.size * ratio

        # 计算滑点后的出场价
        if pos.side == PositionSide.LONG:
            exit_price = price * (1 - self.slippage)
            profit = (exit_price - pos.entry_price) * close_size
        else:
            exit_price = price * (1 + self.slippage)
            profit = (pos.entry_price - exit_price) * close_size

        # 扣除手续费
        position_value = exit_price * close_size
        fee = position_value * self.fee_rate
        net_profit = profit - fee
        self.capital += net_profit

        # 更新剩余仓位 size（保证金和 entry_price 不变）
        pos.size -= close_size

        exit_idx = idx if idx >= 0 else pos.entry_idx
        hold_periods = exit_idx - pos.entry_idx if idx >= 0 else 0

        # 记录部分平仓交易
        profit_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        if pos.side == PositionSide.SHORT:
            profit_pct = -profit_pct

        # 记录 TP1 日志（部分平仓触发追踪止损阶段2）
        if reason == "tp1":
            self._log_event("tp1", exit_idx, pos.side, exit_price, {
                "profit_pct": round(profit_pct, 2),
                "ratio": round(ratio * 100),
                "entry_price": round(pos.entry_price, 2),
                "take_profit": pos.take_profit,
            })

        self.trades.append(TradeRecord(
            entry_idx=pos.entry_idx,
            exit_idx=exit_idx,
            side=int(pos.side),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=close_size,
            profit=net_profit,
            profit_pct=profit_pct,
            hold_periods=hold_periods,
            exit_reason=reason,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            liquidation_price=pos.liquidation_price,
            margin=pos.margin,
            signal_key=pos.signal_key,
            avg_hold_bars=pos.avg_hold_bars,
        ))

        # 更新统计缓存（部分平仓计为一笔独立交易）
        self._total_trades += 1
        if net_profit > 0:
            self._win_trades += 1
            self._gross_profit += net_profit
        else:
            self._loss_trades += 1
            self._gross_loss += abs(net_profit)
        self._total_profit += net_profit
        self._profit_pcts.append(profit_pct)
        self._hold_periods.append(hold_periods)

        if pos.side == PositionSide.LONG:
            self._long_trades += 1
            self._long_profit += net_profit
            if net_profit > 0:
                self._long_win_trades += 1
        elif pos.side == PositionSide.SHORT:
            self._short_trades += 1
            self._short_profit += net_profit
            if net_profit > 0:
                self._short_win_trades += 1
    
    def _calculate_equity(self, current_price: float) -> float:
        """计算当前权益"""
        equity = self.capital
        
        if self.position.side == PositionSide.LONG:
            unrealized = (current_price - self.position.entry_price) * self.position.size
            equity += unrealized
        elif self.position.side == PositionSide.SHORT:
            unrealized = (self.position.entry_price - current_price) * self.position.size
            equity += unrealized
        
        return equity
    
    def _calculate_statistics(self) -> BacktestResult:
        """计算统计指标"""
        return self.get_realtime_result()


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    n = 5000
    
    test_df = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(n) * 50),
    })
    test_df['close'] = test_df['open'] + np.random.randn(n) * 30
    test_df['high'] = test_df[['open', 'close']].max(axis=1) + abs(np.random.randn(n) * 20)
    test_df['low'] = test_df[['open', 'close']].min(axis=1) - abs(np.random.randn(n) * 20)
    
    # 生成随机信号
    signals = pd.Series(np.zeros(n))
    buy_idx = np.random.choice(n, 50, replace=False)
    sell_idx = np.random.choice([i for i in range(n) if i not in buy_idx], 50, replace=False)
    signals.iloc[buy_idx] = 1
    signals.iloc[sell_idx] = -1
    
    # 执行回测
    backtester = Backtester()
    result = backtester.run(test_df, signals)
    
    print("回测结果:")
    print(f"  总交易数: {result.total_trades}")
    print(f"  胜率: {result.win_rate:.1%}")
    print(f"  总收益: ${result.total_profit:.2f}")
    print(f"  总收益率: {result.total_return_pct:.2f}%")
    print(f"  最大回撤: {result.max_drawdown:.1%}")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  盈亏比: {result.profit_factor:.2f}")
