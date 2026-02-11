"""
R3000 虚拟订单管理模块
模拟交易的核心：管理虚拟持仓、计算盈亏、记录交易

功能：
  - 虚拟开仓/平仓
  - 实时盈亏计算
  - 止盈止损管理
  - 交易记录存储
  - 模板表现统计
"""

import json
import os
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class OrderSide(Enum):
    """订单方向"""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "PENDING"      # 待成交
    FILLED = "FILLED"        # 已成交
    CLOSED = "CLOSED"        # 已平仓
    CANCELLED = "CANCELLED"  # 已取消


class CloseReason(Enum):
    """平仓原因"""
    TAKE_PROFIT = "止盈"
    STOP_LOSS = "止损"
    DERAIL = "脱轨"          # 动态追踪脱轨
    MAX_HOLD = "超时"        # 超过最大持仓时间
    MANUAL = "手动"          # 手动平仓
    SIGNAL = "信号"          # 模板匹配离场信号


@dataclass
class PaperOrder:
    """虚拟订单"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float           # 数量 (如 0.05 BTC)
    margin_used: float        # 占用保证金 (USDT)
    entry_price: float        # 入场价
    entry_time: datetime      # 入场时间
    entry_bar_idx: int        # 入场K线索引
    
    # 止盈止损
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    original_stop_loss: Optional[float] = None  # 原始止损（警戒模式恢复用）
    
    # 状态
    status: OrderStatus = OrderStatus.FILLED
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_bar_idx: Optional[int] = None
    close_reason: Optional[CloseReason] = None
    
    # 盈亏
    unrealized_pnl: float = 0.0      # 未实现盈亏 (USDT)
    realized_pnl: float = 0.0        # 已实现盈亏 (USDT)
    profit_pct: float = 0.0          # 收益率 (%)
    
    # 模板信息
    template_fingerprint: Optional[str] = None
    entry_similarity: float = 0.0
    entry_reason: str = ""    # 开仓因果说明
    
    # 动态追踪状态
    tracking_status: str = "安全"     # "安全" / "警戒" / "脱轨"
    alert_mode: bool = False          # 是否处于警戒模式
    current_similarity: float = 0.0   # 当前相似度
    
    # 持仓时长
    hold_bars: int = 0
    
    def update_pnl(self, current_price: float, leverage: float = 10):
        """更新未实现盈亏"""
        if self.status != OrderStatus.FILLED:
            return
        
        if self.side == OrderSide.LONG:
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:
            price_change_pct = (self.entry_price - current_price) / self.entry_price
        
        self.profit_pct = price_change_pct * 100 * leverage
        self.unrealized_pnl = self.quantity * self.entry_price * price_change_pct * leverage
    
    def close(self, exit_price: float, exit_time: datetime, exit_bar_idx: int,
              reason: CloseReason, leverage: float = 10):
        """平仓"""
        self.status = OrderStatus.CLOSED
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_bar_idx = exit_bar_idx
        self.close_reason = reason
        
        if self.side == OrderSide.LONG:
            price_change_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            price_change_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.profit_pct = price_change_pct * 100 * leverage
        self.realized_pnl = self.quantity * self.entry_price * price_change_pct * leverage
        self.unrealized_pnl = 0.0
    
    def to_dict(self) -> dict:
        """转为字典（用于存储/显示）"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "margin_used": self.margin_used,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_bar_idx": self.entry_bar_idx,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "status": self.status.value,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_bar_idx": self.exit_bar_idx,
            "close_reason": self.close_reason.value if self.close_reason else None,
            "realized_pnl": self.realized_pnl,
            "profit_pct": self.profit_pct,
            "template_fingerprint": self.template_fingerprint,
            "entry_similarity": self.entry_similarity,
            "entry_reason": self.entry_reason,
            "hold_bars": self.hold_bars,
        }


@dataclass
class TemplateSimPerformance:
    """模板在模拟交易中的表现"""
    fingerprint: str
    match_count: int = 0       # 匹配次数
    win_count: int = 0         # 盈利次数
    loss_count: int = 0        # 亏损次数
    total_profit: float = 0.0  # 累计收益 (USDT)
    profits: List[float] = field(default_factory=list)  # 每次收益
    
    @property
    def win_rate(self) -> float:
        if self.match_count == 0:
            return 0.0
        return self.win_count / self.match_count
    
    @property
    def avg_profit(self) -> float:
        if not self.profits:
            return 0.0
        return sum(self.profits) / len(self.profits)
    
    def add_trade(self, profit_pct: float):
        """添加一次交易结果"""
        self.match_count += 1
        self.profits.append(profit_pct)
        self.total_profit += profit_pct
        if profit_pct > 0:
            self.win_count += 1
        else:
            self.loss_count += 1


@dataclass
class AccountStats:
    """账户统计"""
    initial_balance: float = 5000.0
    current_balance: float = 5000.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    
    max_balance: float = 5000.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # 分方向统计
    long_trades: int = 0
    long_wins: int = 0
    short_trades: int = 0
    short_wins: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_trades / self.total_trades
    
    @property
    def long_win_rate(self) -> float:
        if self.long_trades == 0:
            return 0.0
        return self.long_wins / self.long_trades
    
    @property
    def short_win_rate(self) -> float:
        if self.short_trades == 0:
            return 0.0
        return self.short_wins / self.short_trades


class PaperTrader:
    """
    虚拟交易管理器
    
    用法：
        trader = PaperTrader(
            initial_balance=5000,
            leverage=10,
        )
        
        # 开仓
        order = trader.open_position(
            side=OrderSide.LONG,
            price=97500,
            template_fingerprint="abc123",
        )
        
        # 更新价格
        trader.update_price(97800)
        
        # 平仓
        trader.close_position(order.order_id, 97800, CloseReason.TAKE_PROFIT)
    """
    
    def __init__(self,
                 symbol: str = "BTCUSDT",
                 initial_balance: float = 5000.0,
                 leverage: float = 10,
                 position_size_pct: float = 1.0,
                 fee_rate: float = 0.0004,
                 slippage: float = 0.0002,
                 on_order_update: Optional[Callable[[PaperOrder], None]] = None,
                 on_trade_closed: Optional[Callable[[PaperOrder], None]] = None):
        """
        Args:
            symbol: 交易对
            initial_balance: 初始余额 (USDT)
            leverage: 杠杆倍数
            position_size_pct: 每次开仓使用资金比例 (1.0 = 全仓)
            fee_rate: 手续费率
            slippage: 滑点
            on_order_update: 订单更新回调
            on_trade_closed: 交易关闭回调
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.on_order_update = on_order_update
        self.on_trade_closed = on_trade_closed
        
        # 账户状态
        self.balance = initial_balance
        self.stats = AccountStats(
            initial_balance=initial_balance,
            current_balance=initial_balance,
        )
        
        # 当前持仓
        self.current_position: Optional[PaperOrder] = None
        
        # 历史订单
        self.order_history: List[PaperOrder] = []
        
        # 模板表现统计
        self.template_performances: Dict[str, TemplateSimPerformance] = {}
        
        # 当前K线索引
        self.current_bar_idx: int = 0
        
        # 订单ID计数器
        self._order_counter = 0
    
    def has_position(self) -> bool:
        """是否有持仓"""
        return self.current_position is not None
    
    def get_position_side(self) -> Optional[OrderSide]:
        """获取当前持仓方向"""
        if self.current_position:
            return self.current_position.side
        return None
    
    def open_position(self,
                      side: OrderSide,
                      price: float,
                      bar_idx: int,
                      take_profit: Optional[float] = None,
                      stop_loss: Optional[float] = None,
                      template_fingerprint: Optional[str] = None,
                      entry_similarity: float = 0.0,
                      entry_reason: str = "") -> Optional[PaperOrder]:
        """
        开仓
        
        Args:
            side: 方向
            price: 入场价
            bar_idx: K线索引
            take_profit: 止盈价
            stop_loss: 止损价
            template_fingerprint: 匹配的模板指纹
            entry_similarity: 入场相似度
        
        Returns:
            PaperOrder 或 None（如果已有持仓）
        """
        if self.current_position is not None:
            print("[PaperTrader] 已有持仓，无法开仓")
            return None
        
        # 计算开仓数量
        margin = self.balance * self.position_size_pct
        
        # 考虑滑点
        if side == OrderSide.LONG:
            actual_price = price * (1 + self.slippage)
        else:
            actual_price = price * (1 - self.slippage)
        
        # 计算数量（名义价值 / 价格）
        notional = margin * self.leverage
        quantity = notional / actual_price
        
        # 扣除手续费
        fee = notional * self.fee_rate
        self.balance -= fee
        
        # 创建订单
        self._order_counter += 1
        order = PaperOrder(
            order_id=f"SIM_{self._order_counter:06d}",
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            margin_used=margin,
            entry_price=actual_price,
            entry_time=datetime.now(),
            entry_bar_idx=bar_idx,
            take_profit=take_profit,
            stop_loss=stop_loss,
            original_stop_loss=stop_loss,
            template_fingerprint=template_fingerprint,
            entry_similarity=entry_similarity,
            entry_reason=entry_reason,
        )
        
        self.current_position = order
        self.current_bar_idx = bar_idx
        
        print(f"[PaperTrader] 开仓: {side.value} {quantity:.6f} @ {actual_price:.2f}")
        
        if self.on_order_update:
            self.on_order_update(order)
        
        return order
    
    def close_position(self,
                       price: float,
                       bar_idx: int,
                       reason: CloseReason) -> Optional[PaperOrder]:
        """
        平仓
        
        Args:
            price: 平仓价
            bar_idx: K线索引
            reason: 平仓原因
        
        Returns:
            关闭的订单 或 None
        """
        if self.current_position is None:
            return None
        
        order = self.current_position
        
        # 考虑滑点
        if order.side == OrderSide.LONG:
            actual_price = price * (1 - self.slippage)
        else:
            actual_price = price * (1 + self.slippage)
        
        # 计算盈亏
        order.close(
            exit_price=actual_price,
            exit_time=datetime.now(),
            exit_bar_idx=bar_idx,
            reason=reason,
            leverage=self.leverage,
        )
        
        # 扣除手续费
        notional = order.quantity * actual_price
        fee = notional * self.fee_rate
        
        # 更新余额
        pnl = order.realized_pnl - fee
        self.balance += pnl
        
        # 更新统计
        self._update_stats(order)
        
        # 记录模板表现
        if order.template_fingerprint:
            self._record_template_performance(order)
        
        # 保存到历史
        self.order_history.append(order)
        self.current_position = None
        
        print(f"[PaperTrader] 平仓: {reason.value} @ {actual_price:.2f} | "
              f"盈亏: {order.profit_pct:+.2f}%")
        
        if self.on_trade_closed:
            self.on_trade_closed(order)
        
        return order
    
    def update_price(self, price: float, high: float = None, low: float = None,
                     bar_idx: int = None) -> Optional[CloseReason]:
        """
        更新价格，检查止盈止损
        
        Args:
            price: 当前价格
            high: 最高价（用于更精确的止盈止损判断）
            low: 最低价
            bar_idx: K线索引
        
        Returns:
            触发的平仓原因 或 None
        """
        if self.current_position is None:
            return None
        
        if bar_idx is not None:
            self.current_bar_idx = bar_idx
            self.current_position.hold_bars = bar_idx - self.current_position.entry_bar_idx
        
        order = self.current_position
        high = high or price
        low = low or price
        
        # 更新未实现盈亏
        order.update_pnl(price, self.leverage)
        
        # 检查止盈
        if order.take_profit is not None:
            if order.side == OrderSide.LONG and high >= order.take_profit:
                self.close_position(order.take_profit, bar_idx or self.current_bar_idx,
                                   CloseReason.TAKE_PROFIT)
                return CloseReason.TAKE_PROFIT
            elif order.side == OrderSide.SHORT and low <= order.take_profit:
                self.close_position(order.take_profit, bar_idx or self.current_bar_idx,
                                   CloseReason.TAKE_PROFIT)
                return CloseReason.TAKE_PROFIT
        
        # 检查止损
        if order.stop_loss is not None:
            if order.side == OrderSide.LONG and low <= order.stop_loss:
                self.close_position(order.stop_loss, bar_idx or self.current_bar_idx,
                                   CloseReason.STOP_LOSS)
                return CloseReason.STOP_LOSS
            elif order.side == OrderSide.SHORT and high >= order.stop_loss:
                self.close_position(order.stop_loss, bar_idx or self.current_bar_idx,
                                   CloseReason.STOP_LOSS)
                return CloseReason.STOP_LOSS
        
        # 回调
        if self.on_order_update:
            self.on_order_update(order)
        
        return None
    
    def update_tracking_status(self, similarity: float,
                               safe_threshold: float = 0.7,
                               alert_threshold: float = 0.5,
                               derail_threshold: float = 0.3,
                               current_price: float = None,
                               bar_idx: int = None) -> Optional[CloseReason]:
        """
        更新动态追踪状态
        
        Args:
            similarity: 当前相似度
            safe_threshold: 安全阈值
            alert_threshold: 警戒阈值
            derail_threshold: 脱轨阈值
            current_price: 当前价格（脱轨时平仓用）
            bar_idx: K线索引
        
        Returns:
            触发的平仓原因（脱轨时返回 DERAIL）或 None
        """
        if self.current_position is None:
            return None
        
        order = self.current_position
        order.current_similarity = similarity
        
        if similarity >= safe_threshold:
            # 安全区
            order.tracking_status = "安全"
            if order.alert_mode:
                # 从警戒恢复，还原止损
                order.alert_mode = False
                order.stop_loss = order.original_stop_loss
        
        elif similarity >= alert_threshold:
            # 警戒区
            order.tracking_status = "警戒"
            if not order.alert_mode:
                # 进入警戒，收紧止损到成本价
                order.alert_mode = True
                order.stop_loss = order.entry_price
        
        else:
            # 脱轨区
            order.tracking_status = "脱轨"
            if current_price is not None:
                self.close_position(current_price, bar_idx or self.current_bar_idx,
                                   CloseReason.DERAIL)
                return CloseReason.DERAIL
        
        return None
    
    def _update_stats(self, order: PaperOrder):
        """更新账户统计"""
        self.stats.total_trades += 1
        self.stats.current_balance = self.balance
        self.stats.total_pnl = self.balance - self.initial_balance
        self.stats.total_pnl_pct = (self.balance / self.initial_balance - 1) * 100
        
        if order.profit_pct > 0:
            self.stats.win_trades += 1
        else:
            self.stats.loss_trades += 1
        
        if order.side == OrderSide.LONG:
            self.stats.long_trades += 1
            if order.profit_pct > 0:
                self.stats.long_wins += 1
        else:
            self.stats.short_trades += 1
            if order.profit_pct > 0:
                self.stats.short_wins += 1
        
        # 更新最大回撤
        if self.balance > self.stats.max_balance:
            self.stats.max_balance = self.balance
        
        drawdown = self.stats.max_balance - self.balance
        if drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = drawdown
            self.stats.max_drawdown_pct = drawdown / self.stats.max_balance * 100
    
    def _record_template_performance(self, order: PaperOrder):
        """记录模板表现"""
        fp = order.template_fingerprint
        if fp not in self.template_performances:
            self.template_performances[fp] = TemplateSimPerformance(fingerprint=fp)
        
        self.template_performances[fp].add_trade(order.profit_pct)
    
    def get_profitable_templates(self, min_matches: int = 1) -> List[str]:
        """获取盈利的模板指纹列表"""
        result = []
        for fp, perf in self.template_performances.items():
            if perf.match_count >= min_matches and perf.win_rate >= 0.5:
                result.append(fp)
        return result
    
    def get_losing_templates(self, min_matches: int = 1) -> List[str]:
        """获取亏损的模板指纹列表"""
        result = []
        for fp, perf in self.template_performances.items():
            if perf.match_count >= min_matches and perf.win_rate < 0.5:
                result.append(fp)
        return result
    
    def reset(self):
        """重置账户"""
        self.balance = self.initial_balance
        self.current_position = None
        self.order_history.clear()
        self.template_performances.clear()
        self.stats = AccountStats(
            initial_balance=self.initial_balance,
            current_balance=self.initial_balance,
        )
        self._order_counter = 0
        print("[PaperTrader] 账户已重置")
    
    def save_history(self, filepath: str):
        """保存交易历史"""
        data = {
            "symbol": self.symbol,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "stats": {
                "total_trades": self.stats.total_trades,
                "win_rate": self.stats.win_rate,
                "total_pnl": self.stats.total_pnl,
                "total_pnl_pct": self.stats.total_pnl_pct,
                "max_drawdown_pct": self.stats.max_drawdown_pct,
            },
            "trades": [order.to_dict() for order in self.order_history],
            "template_performances": {
                fp: {
                    "match_count": perf.match_count,
                    "win_rate": perf.win_rate,
                    "avg_profit": perf.avg_profit,
                    "total_profit": perf.total_profit,
                }
                for fp, perf in self.template_performances.items()
            },
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[PaperTrader] 交易历史已保存: {filepath}")


# 简单测试
if __name__ == "__main__":
    trader = PaperTrader(
        initial_balance=5000,
        leverage=10,
    )
    
    # 开仓
    order = trader.open_position(
        side=OrderSide.LONG,
        price=97500,
        bar_idx=0,
        take_profit=98000,
        stop_loss=97000,
        template_fingerprint="test_fp_123",
        entry_similarity=0.85,
    )
    
    # 模拟价格变动
    trader.update_price(97600, bar_idx=1)
    print(f"  未实现盈亏: {order.unrealized_pnl:.2f} USDT ({order.profit_pct:.2f}%)")
    
    trader.update_price(97800, bar_idx=2)
    print(f"  未实现盈亏: {order.unrealized_pnl:.2f} USDT ({order.profit_pct:.2f}%)")
    
    # 触发止盈
    result = trader.update_price(98100, high=98100, bar_idx=3)
    print(f"  触发: {result}")
    
    # 查看统计
    print(f"\n账户统计:")
    print(f"  余额: {trader.balance:.2f} USDT")
    print(f"  总盈亏: {trader.stats.total_pnl:.2f} USDT")
    print(f"  胜率: {trader.stats.win_rate:.1%}")
