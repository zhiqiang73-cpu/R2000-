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
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field, replace
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
    TRAILING_STOP = "追踪止损"    # 追踪止损/保本止损触发（有盈利但未到TP）
    PARTIAL_TP = "分段止盈"       # 阶梯止盈部分平仓
    PARTIAL_SL = "分段止损"       # 阶梯止损部分平仓
    DERAIL = "脱轨"          # 动态追踪脱轨
    MAX_HOLD = "超时"        # 超过最大持仓时间
    MANUAL = "手动"          # 手动平仓
    SIGNAL = "信号"          # 模板匹配离场信号
    EXCHANGE_CLOSE = "交易所平仓"  # 交易所侧被动平仓（非本系统主动触发）
    POSITION_FLIP = "位置翻转"    # 价格到达区间极端位置，主动平仓+反手


def load_trade_history_from_file(filepath: str) -> List["PaperOrder"]:
    """
    从 JSON 文件加载历史交易记录（程序启动时调用，与 BinanceTestnetTrader 共用格式）
    
    Returns:
        已解析的 PaperOrder 列表，文件不存在或解析失败时返回空列表
    """
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        trades_data = data.get("trades", [])
        loaded = []
        for t in trades_data:
            entry_time = None
            if t.get("entry_time"):
                try:
                    entry_time = datetime.fromisoformat(t["entry_time"])
                except (ValueError, TypeError):
                    pass
            exit_time = None
            if t.get("exit_time"):
                try:
                    exit_time = datetime.fromisoformat(t["exit_time"])
                except (ValueError, TypeError):
                    pass
            close_reason = None
            if t.get("close_reason"):
                try:
                    close_reason = CloseReason(t["close_reason"])
                except ValueError:
                    pass
            
            # 【指纹3D图】加载轨迹矩阵数据
            entry_trajectory = None
            traj_data = t.get("entry_trajectory")
            if traj_data is not None:
                try:
                    entry_trajectory = np.array(traj_data, dtype=np.float32)
                except (ValueError, TypeError):
                    entry_trajectory = None
            
            order = PaperOrder(
                order_id=t.get("order_id", ""),
                symbol=t.get("symbol", ""),
                side=OrderSide(t["side"]) if t.get("side") else OrderSide.LONG,
                quantity=float(t.get("quantity", 0)),
                margin_used=float(t.get("margin_used", 0)),
                leverage=float(t.get("leverage", 0)),
                entry_price=float(t.get("entry_price", 0)),
                entry_time=entry_time,
                entry_bar_idx=int(t.get("entry_bar_idx", 0)),
                take_profit=t.get("take_profit"),
                stop_loss=t.get("stop_loss"),
                status=OrderStatus(t["status"]) if t.get("status") else OrderStatus.CLOSED,
                exit_price=t.get("exit_price"),
                exit_time=exit_time,
                exit_bar_idx=t.get("exit_bar_idx"),
                close_reason=close_reason,
                close_reason_detail=t.get("close_reason_detail", "") or "",
                realized_pnl=float(t.get("realized_pnl", 0)),
                profit_pct=float(t.get("profit_pct", 0)),
                total_fee=float(t.get("total_fee", 0)),
                template_fingerprint=t.get("template_fingerprint"),
                entry_similarity=float(t.get("entry_similarity", 0)),
                entry_reason=t.get("entry_reason", ""),
                decision_reason=t.get("decision_reason", ""),
                hold_bars=int(t.get("hold_bars", 0)),
                # === 利润追踪字段 ===
                peak_profit_pct=float(t.get("peak_profit_pct", 0)),
                partial_tp_count=int(t.get("partial_tp_count", 0)),
                partial_sl_count=int(t.get("partial_sl_count", 0)),
                # === 离场学习字段 ===
                exit_signals_triggered=t.get("exit_signals_triggered", []),
                entry_atr=float(t.get("entry_atr", 0)),
                # === 翻转单标记 ===
                is_flip_trade=bool(t.get("is_flip_trade", False)),
                flip_reason=t.get("flip_reason", ""),
                # === 指纹3D图 ===
                entry_trajectory=entry_trajectory,
                # === 自适应学习字段 ===
                similarity_history=t.get("similarity_history", []),
                reasoning_history=t.get("reasoning_history", []),
                regime_at_entry=t.get("regime_at_entry", "未知"),
                entry_snapshot=t.get("entry_snapshot"),
                exit_snapshot=t.get("exit_snapshot"),
                indicator_snapshots_during_hold=t.get("indicator_snapshots_during_hold", []),
                # 凯利动态仓位
                kelly_position_pct=float(t.get("kelly_position_pct", 0)),
                # 信号组合跟踪
                signal_combo_keys=t.get("signal_combo_keys", []) or [],
            )
            loaded.append(order)
        return loaded
    except Exception as e:
        print(f"[PaperTrader] 加载历史记录失败 {filepath}: {e}")
        return []


def save_trade_history_to_file(orders: List["PaperOrder"], filepath: str) -> None:
    """
    保存交易记录到 JSON 文件（与 load_trade_history_from_file 兼容）
    """
    try:
        data = {
            "trades": [
                o.to_dict() if hasattr(o, "to_dict") else o
                for o in (orders or [])
            ]
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[PaperTrader] 交易记录已保存: {filepath}")
    except Exception as e:
        print(f"[PaperTrader] 保存历史记录失败 {filepath}: {e}")


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
    leverage: float = 0.0     # 杠杆（用于UI显示/记录）
    
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
    close_reason_detail: str = ""   # 平仓原因补充（如 "挂单触发(交易所成交)" / "按价格推断"）

    # 盈亏
    unrealized_pnl: float = 0.0      # 未实现盈亏 (USDT)
    realized_pnl: float = 0.0        # 已实现盈亏 (USDT)
    profit_pct: float = 0.0          # 收益率 (%)
    total_fee: float = 0.0           # 总手续费 (USDT，含开仓+平仓)
    
    # 模板信息
    template_fingerprint: Optional[str] = None
    entry_similarity: float = 0.0
    entry_reason: str = ""    # 开仓因果说明
    decision_reason: str = "" # 平仓决策说明（详细原因）
    
    # 动态追踪状态
    tracking_status: str = "安全"     # "安全" / "警戒" / "脱轨"
    alert_mode: bool = False          # 是否处于警戒模式
    current_similarity: float = 0.0   # 当前相似度
    
    # 持仓时长
    hold_bars: int = 0
    
    # 利润追踪
    peak_price: float = 0.0           # 持仓期间最有利价格
    peak_profit_pct: float = 0.0      # 持仓期间峰值收益率 (%)
    partial_tp_count: int = 0         # 已执行分段止盈次数
    partial_sl_count: int = 0         # 已执行分段止损次数
    
    # 离场信号学习（用于自适应优化）
    exit_signals_triggered: List[tuple] = field(default_factory=list)  # [(signal_name, profit_at_trigger), ...]
    entry_atr: float = 0.0            # 入场时的 ATR（用于学习最优 TP 距离）
    
    # 翻转单标记（用于贝叶斯加权学习）
    is_flip_trade: bool = False       # 是否由价格位置翻转触发
    flip_reason: str = ""             # 翻转原因（"底部翻转做多"/"顶部翻转做空"）
    
    # 【指纹3D图】入场时的轨迹矩阵数据（用于增量训练）
    entry_trajectory: Optional[np.ndarray] = None  # 入场时的 (60, 32) 轨迹矩阵
    
    # 【自适应学习】决策快照和推理历史
    entry_snapshot: Optional['DecisionSnapshot'] = None  # 入场决策快照
    exit_snapshot: Optional['DecisionSnapshot'] = None   # 出场决策快照
    indicator_snapshots: List[Dict] = field(default_factory=list)  # 持仓期间的指标快照
    similarity_history: List[Tuple[int, float]] = field(default_factory=list)  # [(bar_idx, similarity), ...]
    reasoning_history: List['ReasoningResult'] = field(default_factory=list)  # 推理结果历史
    regime_at_entry: str = "未知"  # 入场时的市场状态
    
    # 限价单相关
    pending_limit_order: bool = False      # 是否有待成交限价单
    limit_order_price: Optional[float] = None  # 限价单价格
    limit_order_start_bar: Optional[int] = None  # 限价单挂单开始K线
    limit_order_max_wait: int = 5          # 最多等待5根K线
    limit_order_quantity: Optional[float] = None  # 限价单数量（支持部分平仓）
    
    # 持仓期间的指标快照（用于反事实分析）
    indicator_snapshots_during_hold: List[dict] = field(default_factory=list)  # [DecisionSnapshot.to_dict(), ...]
    
    # 凯利动态仓位（用于自适应学习）
    kelly_position_pct: float = 0.0  # 凯利公式计算的仓位比例（0-1）

    # 信号组合跟踪（用于实盘命中率统计）
    signal_combo_keys: List[str] = field(default_factory=list)  # 开仓时触发的组合key列表

    def update_pnl(self, current_price: float, leverage: float = 10):
        """更新未实现盈亏 + 追踪峰值"""
        if self.status != OrderStatus.FILLED:
            return
        
        if self.side == OrderSide.LONG:
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:
            price_change_pct = (self.entry_price - current_price) / self.entry_price
        
        self.profit_pct = price_change_pct * 100 * leverage
        self.unrealized_pnl = self.quantity * self.entry_price * price_change_pct * leverage
        
        # 追踪峰值利润（用于锁利逻辑）
        if self.side == OrderSide.LONG:
            if current_price > self.peak_price:
                self.peak_price = current_price
        else:
            if self.peak_price == 0 or current_price < self.peak_price:
                self.peak_price = current_price
        if self.profit_pct > self.peak_profit_pct:
            self.peak_profit_pct = self.profit_pct
    
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
        # 【指纹3D图】将轨迹矩阵转换为可JSON序列化的格式
        trajectory_data = None
        if self.entry_trajectory is not None and isinstance(self.entry_trajectory, np.ndarray):
            trajectory_data = self.entry_trajectory.tolist()
        
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "margin_used": self.margin_used,
            "leverage": self.leverage,
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
            "close_reason_detail": getattr(self, "close_reason_detail", "") or "",
            "realized_pnl": self.realized_pnl,
            "profit_pct": self.profit_pct,
            "total_fee": self.total_fee,
            "template_fingerprint": self.template_fingerprint,
            "entry_similarity": self.entry_similarity,
            "entry_reason": self.entry_reason,
            "decision_reason": self.decision_reason,
            "hold_bars": self.hold_bars,
            "peak_profit_pct": self.peak_profit_pct,
            "partial_tp_count": self.partial_tp_count,
            "partial_sl_count": self.partial_sl_count,
            "exit_signals_triggered": self.exit_signals_triggered,
            "entry_atr": self.entry_atr,
            "is_flip_trade": self.is_flip_trade,
            "flip_reason": self.flip_reason,
            # 【指纹3D图】轨迹矩阵数据
            "entry_trajectory": trajectory_data,
            # === 自适应学习字段 ===
            "similarity_history": self.similarity_history,
            "reasoning_history": self.reasoning_history,
            "regime_at_entry": self.regime_at_entry,
            "entry_snapshot": self.entry_snapshot,
            "exit_snapshot": self.exit_snapshot,
            "indicator_snapshots_during_hold": self.indicator_snapshots_during_hold,
            # 凯利动态仓位
            "kelly_position_pct": self.kelly_position_pct,
            # 信号组合跟踪
            "signal_combo_keys": self.signal_combo_keys,
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
    available_margin: float = 0.0
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
                 taker_fee_rate: float = 0.0004,  # Taker手续费（市价单）
                 maker_fee_rate: float = 0.0002,  # Maker手续费（限价单）
                 slippage: float = 0.0,  # 去除滑点假设
                 limit_order_offset: float = 0.0001,  # 限价单偏移0.01%
                 limit_order_max_wait: int = 5,  # 限价单最大等待K线数
                 on_order_update: Optional[Callable[[PaperOrder], None]] = None,
                 on_trade_closed: Optional[Callable[[PaperOrder], None]] = None):
        """
        Args:
            symbol: 交易对
            initial_balance: 初始余额 (USDT)
            leverage: 杠杆倍数
            position_size_pct: 每次开仓使用资金比例 (1.0 = 全仓)
            taker_fee_rate: Taker手续费率（市价单开仓）
            maker_fee_rate: Maker手续费率（限价单平仓）
            slippage: 滑点（已去除，保留参数兼容性）
            limit_order_offset: 限价单价格偏移
            limit_order_max_wait: 限价单最多等待K线数
            on_order_update: 订单更新回调
            on_trade_closed: 交易关闭回调
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.taker_fee_rate = taker_fee_rate
        self.maker_fee_rate = maker_fee_rate
        self.slippage = slippage  # 保留但不使用
        self.limit_order_offset = limit_order_offset
        self.limit_order_max_wait = limit_order_max_wait
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

        # 待成交的止损单 (Resting/Stop Orders)
        self.pending_stop_orders: List[dict] = []  # List of {side, trigger_price, qty, ...}
    
    def has_position(self) -> bool:
        """是否有持仓"""
        return self.current_position is not None
    
    def get_position_side(self) -> Optional[OrderSide]:
        """获取当前持仓方向"""
        if self.current_position:
            return self.current_position.side
        return None
    
    def has_pending_stop_orders(self, current_bar_idx: int = None) -> bool:
        """检查是否有待成交的止损单（排除已超时）"""
        if not self.pending_stop_orders:
            return False
        
        # 如果提供了 bar_idx，过滤掉已超时的订单
        if current_bar_idx is not None:
            valid_orders = [o for o in self.pending_stop_orders if current_bar_idx <= o["expire_bar"]]
            return len(valid_orders) > 0
        
        return len(self.pending_stop_orders) > 0
    
    def set_leverage(self, leverage: int):
        """运行时修改杠杆"""
        if leverage < 1 or leverage > 125:
            raise ValueError(f"杠杆倍数必须在1-125之间，当前: {leverage}")
        
        old_leverage = self.leverage
        self.leverage = leverage
        
        # 如果有持仓，需要调用交易所API修改杠杆（测试网可能不支持）
        # 这里只修改内部变量，新订单会使用新杠杆
        print(f"[PaperTrader] 杠杆已更新: {old_leverage}x -> {leverage}x")
    
    def open_position(self,
                      side: OrderSide,
                      price: float,
                      bar_idx: int,
                      take_profit: Optional[float] = None,
                      stop_loss: Optional[float] = None,
                      template_fingerprint: Optional[str] = None,
                      entry_similarity: float = 0.0,
                      entry_reason: str = "",
                      entry_trajectory: Optional[np.ndarray] = None,
                      regime_at_entry: str = "未知") -> Optional[PaperOrder]:
        """
        开仓 (市价/直接成交)
        
        Args:
            entry_trajectory: 【指纹3D图】入场时的轨迹矩阵 (60, 32)，用于增量训练
        """
        if self.current_position is not None:
            print("[PaperTrader] 已有持仓，无法开仓")
            return None
        
        # 计算开仓数量
        margin = self.balance * self.position_size_pct
        actual_price = price
        notional = margin * self.leverage
        quantity = notional / actual_price
        
        # 扣除手续费
        fee = notional * self.taker_fee_rate
        self.balance -= fee
        
        # 创建并返回订单
        return self._create_filled_order(
            side=side, price=actual_price, qty=quantity, margin=margin,
            bar_idx=bar_idx, tp=take_profit, sl=stop_loss,
            fp=template_fingerprint, sim=entry_similarity, reason=entry_reason,
            trajectory=entry_trajectory,
            regime_at_entry=regime_at_entry,
        )

    def place_stop_order(self,
                        side: OrderSide,
                        trigger_price: float,
                        bar_idx: int,
                        take_profit: Optional[float] = None,
                        stop_loss: Optional[float] = None,
                        template_fingerprint: Optional[str] = None,
                        entry_similarity: float = 0.0,
                        entry_reason: str = "",
                        timeout_bars: int = 5,
                        entry_trajectory: Optional[np.ndarray] = None,
                        position_size_pct: Optional[float] = None,
                        regime_at_entry: str = "未知") -> str:
        """
        放置条件触发单 (Stop Order)
        
        Args:
            entry_trajectory: 【指纹3D图】入场时的轨迹矩阵 (60, 32)，用于增量训练
            position_size_pct: 凯利动态仓位比例（None=使用默认固定仓位）
        """
        self._order_counter += 1
        order_id = f"STOP_{self._order_counter:06d}"
        
        stop_order = {
            "order_id": order_id,
            "side": side,
            "trigger_price": trigger_price,
            "start_bar": bar_idx,
            "expire_bar": bar_idx + timeout_bars,
            "tp": take_profit,
            "sl": stop_loss,
            "fp": template_fingerprint,
            "sim": entry_similarity,
            "reason": entry_reason,
            "trajectory": entry_trajectory,  # 【指纹3D图】轨迹矩阵
            "position_size_pct": position_size_pct,  # 凯利动态仓位
            "regime_at_entry": regime_at_entry,
        }
        
        self.pending_stop_orders.append(stop_order)
        pct_str = f", 仓位={position_size_pct:.1%}" if position_size_pct else ""
        print(f"[PaperTrader] 放置止损触发单: {side.value} @ 触发价 {trigger_price:.2f} (有效至 Bar {bar_idx + timeout_bars}{pct_str})")
        return order_id

    def cancel_stop_order(self, order_id: str):
        """撤销待处理的触发单"""
        self.pending_stop_orders = [o for o in self.pending_stop_orders if o["order_id"] != order_id]
        print(f"[PaperTrader] 已撤销触发单: {order_id}")

    def get_pending_entry_orders_snapshot(self, current_bar_idx: int = None) -> list:
        """返回挂单快照（与 BinanceTestnetTrader 同结构，含 take_profit/stop_loss 与持仓保护单供 UI 显示 TP/SL 预计盈亏）"""
        out = []
        for o in self.pending_stop_orders:
            expire_bar = int(o.get("expire_bar", -1))
            remaining_bars = None
            if current_bar_idx is not None and expire_bar >= 0:
                remaining_bars = max(0, expire_bar - int(current_bar_idx))
            side = o.get("side")
            side_str = side.value if hasattr(side, "value") else str(side)
            out.append({
                "order_id": o.get("order_id"),
                "client_id": o.get("client_id", ""),
                "side": side_str,
                "trigger_price": float(o.get("trigger_price", 0) or 0),
                "quantity": float(o.get("quantity", 0) or 0),
                "start_bar": int(o.get("start_bar", -1)),
                "expire_bar": expire_bar,
                "remaining_bars": remaining_bars,
                "template_fingerprint": o.get("fp") or o.get("template_fingerprint") or "-",
                "entry_similarity": float(o.get("sim") or o.get("entry_similarity") or 0),
                "status": "入场挂单",
                "take_profit": o.get("tp"),
                "stop_loss": o.get("sl"),
            })
        # 有持仓时追加止损/止盈保护单行，供 UI 显示「预计亏/赚 金额+百分比」
        pos = self.current_position
        if pos and pos.stop_loss is not None:
            exit_side = "BUY" if pos.side == OrderSide.SHORT else "SELL"
            out.append({
                "order_id": f"SIM_SL_{pos.order_id}",
                "client_id": "R3000_SL",
                "side": exit_side,
                "trigger_price": float(pos.stop_loss),
                "quantity": pos.quantity,
                "remaining_bars": None,
                "template_fingerprint": "止损保护",
                "status": "🛡️止损",
                "entry_price": pos.entry_price,
                "order_type": "sl",
            })
        if pos and pos.take_profit is not None:
            exit_side = "BUY" if pos.side == OrderSide.SHORT else "SELL"
            out.append({
                "order_id": f"SIM_TP_{pos.order_id}",
                "client_id": "R3000_TP",
                "side": exit_side,
                "trigger_price": float(pos.take_profit),
                "quantity": pos.quantity,
                "remaining_bars": None,
                "template_fingerprint": "止盈保护",
                "status": "🎯止盈",
                "entry_price": pos.entry_price,
                "order_type": "tp",
            })
        return out

    def _create_filled_order(self, side, price, qty, margin, bar_idx, tp, sl, fp, sim, reason,
                              trajectory: Optional[np.ndarray] = None,
                              kelly_position_pct: float = 0.0,
                              regime_at_entry: str = "未知") -> PaperOrder:
        """辅助方法：创建已成交订单对象
        
        Args:
            trajectory: 【指纹3D图】入场时的轨迹矩阵 (60, 32)，用于增量训练
            kelly_position_pct: 凯利动态仓位比例（用于自适应学习）
        """
        self._order_counter += 1
        order = PaperOrder(
            order_id=f"SIM_{self._order_counter:06d}",
            symbol=self.symbol,
            side=side,
            quantity=qty,
            margin_used=margin,
            leverage=self.leverage,
            entry_price=price,
            entry_time=datetime.now(),
            entry_bar_idx=bar_idx,
            take_profit=tp,
            stop_loss=sl,
            original_stop_loss=sl,
            template_fingerprint=fp,
            entry_similarity=sim,
            entry_reason=reason,
            peak_price=price,  # 初始峰值 = 入场价
            entry_trajectory=trajectory,  # 【指纹3D图】轨迹矩阵
            kelly_position_pct=kelly_position_pct,  # 凯利动态仓位
            regime_at_entry=regime_at_entry,
        )
        self.current_position = order
        self.current_bar_idx = bar_idx
        if self.on_order_update:
            self.on_order_update(order)
        return order
    
    def close_position(self,
                       price: float,
                       bar_idx: int,
                       reason: CloseReason,
                       use_limit_order: bool = True,
                       quantity: Optional[float] = None) -> Optional[PaperOrder]:
        """
        平仓（始终使用市价单，限价平仓单逻辑已移除）
        
        Args:
            price: 当前价格
            bar_idx: K线索引
            reason: 平仓原因
            use_limit_order: 已忽略，始终市价平仓
            quantity: 平仓数量（None 表示全平）
        
        Returns:
            关闭的订单 或 None
        """
        if self.current_position is None:
            return None
        
        order = self.current_position
        close_qty = quantity if quantity is not None else order.quantity
        
        # 始终市价平仓（限价平仓单逻辑已删除）
        return self._market_close(price, bar_idx, reason, quantity=close_qty)
    
    def update_price(self, price: float, high: float = None, low: float = None,
                     bar_idx: int = None, protection_mode: bool = False) -> Optional[CloseReason]:
        """
        更新价格，检查止盈止损和限价单成交
        
        Args:
            protection_mode: 保护期模式（True时止损暂缓触发，允许止盈）
        """
        if bar_idx is not None:
            self.current_bar_idx = bar_idx

        # 1. 检查待成交的止损入场单 (Entry Stop Orders)
        self._check_pending_stop_orders(price, high, low, bar_idx)

        if self.current_position is None:
            return None
        
        if bar_idx is not None:
            self.current_position.hold_bars = bar_idx - self.current_position.entry_bar_idx
        
        order = self.current_position
        high = high or price
        low = low or price
        
        # 更新未实现盈亏（供实时展示）
        order.update_pnl(price, self.leverage)
        
        # 仅保留分段止盈/分段止损（5%、10%），不再按价格检查硬止盈/硬止损
        # 限价平仓单成交逻辑已删除，始终市价平仓

        # 常规UI回调
        if self.on_order_update:
            self.on_order_update(order)
        
        return None

    def _check_pending_stop_orders(self, price, high, low, bar_idx):
        """检查并执行止损入场单的成交"""
        if self.current_position is not None:
            # 已有持仓，不在此处理入场单（由外部逻辑决定是否撤销）
            return
        effective_bar_idx = bar_idx if bar_idx is not None else self.current_bar_idx

        high = high or price
        low = low or price
        activated_orders = []
        
        for stop_order in self.pending_stop_orders:
            # 检查是否超时
            if effective_bar_idx > stop_order["expire_bar"]:
                print(f"[PaperTrader] 止损触发单已超时: {stop_order['order_id']}")
                continue
            
            triggered = False
            if stop_order["side"] == OrderSide.LONG:
                if high >= stop_order["trigger_price"]:
                    triggered = True
            else: # SHORT
                if low <= stop_order["trigger_price"]:
                    triggered = True
            
            if triggered:
                print(f"[PaperTrader] 🔥 止损触发单成交! Price={price} Trigger={stop_order['trigger_price']}")
                # 记录为已激活，稍后转换
                activated_orders.append(stop_order)
            else:
                # 保留未成交且未超时的单子
                pass

        # 清理已成交或超时的单子（重新构建列表）
        self.pending_stop_orders = [o for o in self.pending_stop_orders 
                                   if (o not in activated_orders) and 
                                   (effective_bar_idx <= o["expire_bar"])]
        
        # 将第一个触发的单子转换为持仓（假设同一时间只允许一个触发）
        if activated_orders:
            sides = {o["side"] for o in activated_orders}
            if len(sides) > 1:
                print("[PaperTrader] ⚠ 同一根K线多空同时触发，取消本次开仓以避免方向冲突")
                return
            # 执行开仓
            o = activated_orders[0]
            # 计算数量（优先使用凯利动态仓位，否则用默认固定仓位）
            kelly_pct = o.get("position_size_pct")
            actual_pct = kelly_pct if kelly_pct is not None else self.position_size_pct
            margin = self.balance * actual_pct
            notional = margin * self.leverage
            quantity = notional / o["trigger_price"]
            
            # 扣除手续费
            fee = notional * self.taker_fee_rate
            self.balance -= fee
            
            self._create_filled_order(
                side=o["side"], price=o["trigger_price"], qty=quantity, margin=margin,
                bar_idx=effective_bar_idx, tp=o["tp"], sl=o["sl"],
                fp=o["fp"], sim=o["sim"], reason=o["reason"],
                trajectory=o.get("trajectory"),  # 【指纹3D图】传递轨迹矩阵
                kelly_position_pct=kelly_pct or 0.0,  # 凯利仓位（用于学习）
                regime_at_entry=o.get("regime_at_entry", "未知"),
            )
    
    def update_tracking_status(self, similarity: float,
                               safe_threshold: float = 0.7,
                               alert_threshold: float = 0.5,
                               derail_threshold: float = 0.3,
                               current_price: float = None,
                               bar_idx: int = None) -> Optional[CloseReason]:
        """
        更新动态追踪状态 (与追踪止损协调，绝不回退SL)
        
        核心原则：追踪状态可以"收紧"止损，但绝不"放松"它。
        如果追踪止损已经把SL上移到比成本价更好的位置，这里不会覆盖。
        """
        if self.current_position is None:
            return None
        
        order = self.current_position
        order.current_similarity = similarity
        
        # 仅更新状态供 UI 显示，不因脱轨/警戒/危险而平仓或收紧止损（平仓仅靠分段止盈/分段止损）
        if similarity >= safe_threshold:
            order.tracking_status = "安全"
            order.alert_mode = False
        elif similarity >= alert_threshold:
            order.tracking_status = "警戒"
            order.alert_mode = True
        elif similarity >= derail_threshold:
            order.tracking_status = "危险"
            order.alert_mode = True
        else:
            order.tracking_status = "脱轨"
            order.alert_mode = True
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
    
    def _market_close(self, price: float, bar_idx: int, reason: CloseReason, quantity: Optional[float] = None) -> PaperOrder:
        """市价紧急平仓"""
        order = self.current_position
        actual_price = price
        original_qty = order.quantity
        close_qty = quantity if quantity is not None else original_qty
        close_qty = min(close_qty, original_qty)
        full_close = close_qty >= (original_qty - 1e-12)
        qty_ratio = close_qty / max(original_qty, 1e-12)

        if order.side == OrderSide.LONG:
            price_change_pct = (actual_price - order.entry_price) / order.entry_price
        else:
            price_change_pct = (order.entry_price - actual_price) / order.entry_price
        profit_pct = price_change_pct * 100 * self.leverage
        realized_pnl = close_qty * order.entry_price * price_change_pct * self.leverage

        notional = close_qty * actual_price
        fee = notional * self.taker_fee_rate
        pnl = realized_pnl - fee
        self.balance += pnl

        # 计算总手续费（开仓+平仓）
        entry_notional = order.quantity * order.entry_price
        entry_fee = entry_notional * self.taker_fee_rate
        total_fee = entry_fee + fee  # 开仓手续费 + 平仓手续费
        
        closed_order = replace(
            order,
            quantity=close_qty,
            margin_used=order.margin_used * qty_ratio,
            status=OrderStatus.CLOSED,
            exit_price=actual_price,
            exit_time=datetime.now(),
            exit_bar_idx=bar_idx,
            close_reason=reason,
            realized_pnl=pnl,  # 改为净盈亏（已扣除平仓手续费）
            profit_pct=profit_pct,
            unrealized_pnl=0.0,
            total_fee=total_fee,
        )
        self._update_stats(closed_order)
        if closed_order.template_fingerprint:
            self._record_template_performance(closed_order)
        self.order_history.append(closed_order)

        if full_close:
            self.current_position = None
            # 增强日志：显示持仓时长和分段止损/止盈次数
            hold_bars = closed_order.hold_bars
            partial_tp = getattr(closed_order, 'partial_tp_count', 0)
            partial_sl = getattr(closed_order, 'partial_sl_count', 0)
            stage_info = ""
            if partial_tp > 0 or partial_sl > 0:
                stage_info = f" | 分段止盈{partial_tp}次 分段止损{partial_sl}次"
            print(f"[PaperTrader] 市价平仓: {reason.value} @ {actual_price:.2f} | "
                  f"盈亏: {profit_pct:+.2f}% ({pnl:+.2f} USDT) | "
                  f"持仓={hold_bars}根K线 | 手续费: {total_fee:.4f}{stage_info}")
            if self.on_trade_closed:
                self.on_trade_closed(closed_order)
        else:
            remaining_qty = original_qty - close_qty
            order.quantity = remaining_qty
            order.margin_used = order.margin_used * (remaining_qty / max(original_qty, 1e-12))
            order.pending_limit_order = False
            order.limit_order_quantity = None
            order.update_pnl(actual_price, self.leverage)
            if self.on_order_update:
                self.on_order_update(order)
            # 增强部分平仓日志
            partial_tp = getattr(closed_order, 'partial_tp_count', 0)
            partial_sl = getattr(closed_order, 'partial_sl_count', 0)
            partial_count = partial_tp + partial_sl
            print(f"[PaperTrader] 市价部分平仓: {reason.value} @ {actual_price:.2f} | "
                  f"数量={close_qty:.6f} | 盈亏: {profit_pct:+.2f}% ({pnl:+.2f} USDT) | "
                  f"剩余仓位={remaining_qty:.6f} | 已分段{partial_count}次")
            if self.on_trade_closed:
                self.on_trade_closed(closed_order)
        return closed_order
    
    def _check_limit_order_fill(self, price: float, high: float, low: float) -> bool:
        """检查限价单是否成交"""
        if self.current_position is None or not self.current_position.pending_limit_order:
            return False
        order = self.current_position
        limit_price = order.limit_order_price
        if order.side == OrderSide.LONG:
            return high >= limit_price
        else:
            return low <= limit_price
    
    def _execute_limit_order_fill(self, bar_idx: int) -> PaperOrder:
        """执行限价单成交"""
        order = self.current_position
        actual_price = order.limit_order_price
        reason = order.close_reason or CloseReason.MANUAL
        original_qty = order.quantity
        close_qty = order.limit_order_quantity if order.limit_order_quantity is not None else original_qty
        close_qty = min(close_qty, original_qty)
        full_close = close_qty >= (original_qty - 1e-12)
        qty_ratio = close_qty / max(original_qty, 1e-12)

        if order.side == OrderSide.LONG:
            price_change_pct = (actual_price - order.entry_price) / order.entry_price
        else:
            price_change_pct = (order.entry_price - actual_price) / order.entry_price
        profit_pct = price_change_pct * 100 * self.leverage
        realized_pnl = close_qty * order.entry_price * price_change_pct * self.leverage

        notional = close_qty * actual_price
        fee = notional * self.maker_fee_rate
        pnl = realized_pnl - fee
        self.balance += pnl

        # 计算总手续费（开仓Taker + 平仓Maker）
        entry_notional = order.quantity * order.entry_price
        entry_fee = entry_notional * self.taker_fee_rate
        total_fee = entry_fee + fee  # 开仓手续费 + 平仓手续费
        
        closed_order = replace(
            order,
            quantity=close_qty,
            margin_used=order.margin_used * qty_ratio,
            status=OrderStatus.CLOSED,
            exit_price=actual_price,
            exit_time=datetime.now(),
            exit_bar_idx=bar_idx,
            close_reason=reason,
            realized_pnl=pnl,  # 改为净盈亏（已扣除平仓手续费）
            profit_pct=profit_pct,
            unrealized_pnl=0.0,
            pending_limit_order=False,
            limit_order_quantity=None,
            total_fee=total_fee,
        )
        self._update_stats(closed_order)
        if closed_order.template_fingerprint:
            self._record_template_performance(closed_order)
        self.order_history.append(closed_order)

        if full_close:
            self.current_position = None
            # 增强日志：显示持仓时长和分段止损/止盈次数
            hold_bars = closed_order.hold_bars
            partial_tp = getattr(closed_order, 'partial_tp_count', 0)
            partial_sl = getattr(closed_order, 'partial_sl_count', 0)
            stage_info = ""
            if partial_tp > 0 or partial_sl > 0:
                stage_info = f" | 分段止盈{partial_tp}次 分段止损{partial_sl}次"
            print(f"[PaperTrader] 限价单成交: {reason.value} @ {actual_price:.2f} | "
                  f"盈亏: {profit_pct:+.2f}% ({pnl:+.2f} USDT) | "
                  f"持仓={hold_bars}根K线 | 手续费: {total_fee:.4f}{stage_info}")
            if self.on_trade_closed:
                self.on_trade_closed(closed_order)
        else:
            remaining_qty = original_qty - close_qty
            order.quantity = remaining_qty
            order.margin_used = order.margin_used * (remaining_qty / max(original_qty, 1e-12))
            order.pending_limit_order = False
            order.limit_order_quantity = None
            order.update_pnl(actual_price, self.leverage)
            if self.on_order_update:
                self.on_order_update(order)
            # 增强部分平仓日志
            partial_count = getattr(closed_order, 'partial_tp_count', 0) + getattr(closed_order, 'partial_sl_count', 0)
            print(f"[PaperTrader] 限价部分成交: {reason.value} @ {actual_price:.2f} | "
                  f"数量={close_qty:.6f} | 盈亏: {profit_pct:+.2f}% | "
                  f"剩余仓位={remaining_qty:.6f} | 已分段{partial_count}次")
            if self.on_trade_closed:
                self.on_trade_closed(closed_order)
        return closed_order

    def _cancel_and_relist_limit_order(self, current_price: float, bar_idx: int):
        """重新挂单"""
        order = self.current_position
        old_price = order.limit_order_price
        print(f"[PaperTrader] 限价单超时 @ {old_price:.2f}，重挂...")
        if order.side == OrderSide.LONG:
            new_limit_price = current_price * (1 + self.limit_order_offset)
        else:
            new_limit_price = current_price * (1 - self.limit_order_offset)
        order.limit_order_price = new_limit_price
        order.limit_order_start_bar = bar_idx
        print(f"[PaperTrader] 重新挂限价单: @ {new_limit_price:.2f}")

    
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
