"""
R3000 Binance 测试网执行器
真实下单到 Binance Futures Testnet：
  - 入场：限价单（LIMIT + GTC，在 trigger_price 挂单等待，争取 Maker 0.02%）
  - 离场：限价单（LIMIT + IOC，reduceOnly，确保快速平仓）
  
手续费优化策略：
  - 入场距离 0.02%（约$13），有较大概率挂单等待成交（Maker）
  - 离场 IOC 使用 EXIT_IOC_BUFFER_PCT（默认 0.3%）提高成交率，减少市价降级（省 Taker 0.05%）
  - 超时 5 根K线未成交自动撤单
"""

import hashlib
import hmac
import time
import os
import json
from datetime import datetime
from dataclasses import replace
from typing import Optional, Dict, List, Callable
from urllib.parse import urlencode

import requests

from core.paper_trader import (
    AccountStats,
    CloseReason,
    OrderSide,
    OrderStatus,
    PaperOrder,
    TemplateSimPerformance,
)


class BinanceTestnetTrader:
    """Binance Futures Testnet 真实执行交易器（接口兼容 PaperTrader）"""

    def __init__(self,
                 symbol: str = "BTCUSDT",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 initial_balance: float = 5000.0,
                 leverage: float = 10,
                 position_size_pct: float = 1.0,
                 fee_rate: float = 0.0004,
                 on_order_update: Optional[Callable[[PaperOrder], None]] = None,
                 on_trade_closed: Optional[Callable[[PaperOrder], None]] = None):
        self.symbol = symbol.upper()
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.initial_balance = float(initial_balance)
        self.leverage = int(leverage)
        self.position_size_pct = float(position_size_pct)
        self.fee_rate = float(fee_rate)
        self.on_order_update = on_order_update
        self.on_trade_closed = on_trade_closed

        # Futures Testnet
        self.base_url = "https://testnet.binancefuture.com"
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        self.stats = AccountStats(
            initial_balance=self.initial_balance,
            current_balance=self.initial_balance,
        )
        self.current_position: Optional[PaperOrder] = None
        self.order_history: List[PaperOrder] = []
        self.template_performances: Dict[str, TemplateSimPerformance] = {}
        
        # 记录保存路径 (使用绝对路径避免当前工作目录切换带来的问题)
        self.history_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.history_file = os.path.join(self.history_dir, "live_trade_history.json")
        
        self.current_bar_idx: int = 0
        self._order_counter = 0

        self._qty_step = 0.001
        self._qty_min = 0.001
        self._price_tick = 0.1
        self._min_notional = 5.0
        self._last_sync_ts = 0.0
        self._sync_interval_sec = 2.0
        self._pending_close = None  # (price, bar_idx, reason) 若离场失败则记录待重试
        # [{order_id, client_id, expire_bar, take_profit, stop_loss}]
        self._entry_stop_orders: List[dict] = []
        # 记录最近一次入场的TP/SL，用于交易所同步建仓时回填
        self._last_entry_tp: Optional[float] = None
        self._last_entry_sl: Optional[float] = None
        self._last_entry_side: Optional[OrderSide] = None
        self._last_entry_price: Optional[float] = None
        self._last_entry_ts: float = 0.0
        # 交易所成交同步游标
        self._last_user_trade_id: int = 0
        self._last_user_trade_time_ms: int = 0
        
        # ═══════════════════════════════════════════════════════════════
        #  【核心改进】交易所端止盈止损保护单
        #  不再仅靠本地 Python 代码检测 TP/SL，而是在交易所挂真实的
        #  STOP_MARKET (止损) 和 TAKE_PROFIT_MARKET (止盈) 订单。
        #  即使程序崩溃、网络断开，交易所也会自动执行保护。
        # ═══════════════════════════════════════════════════════════════
        self._exchange_sl_order_id: Optional[int] = None   # 交易所止损单ID
        self._exchange_tp_order_id: Optional[int] = None   # 交易所止盈单ID
        self._exchange_sl_price: float = 0.0               # 当前交易所止损价
        self._exchange_tp_price: float = 0.0               # 当前交易所止盈价
        self._last_sl_update_ts: float = 0.0               # 上次更新止损的时间
        self._sl_update_min_interval: float = 2.0          # 止损更新最小间隔(秒)

        self._validate_credentials()
        self._load_symbol_filters()
        self._set_leverage(self.leverage)
        self._sync_from_exchange()
        # 启动时清理残留保护单（避免旧订单干扰）
        self._cleanup_orphan_tp_sl()
        # 如果启动时已有持仓且有TP/SL，立即挂保护单
        if self.current_position is not None:
            pos = self.current_position
            if pos.take_profit is not None or pos.stop_loss is not None:
                print(f"[BinanceTrader] 启动时发现持仓，挂交易所保护单...")
                self._place_exchange_tp_sl(pos)
        self._load_history()  # 【持久化】启动时加载历史记录

    def _validate_credentials(self):
        if not self.api_key or not self.api_secret:
            raise ValueError("必须提供 Binance Testnet API Key/Secret")

    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def _sign(self, params: dict) -> str:
        query = urlencode(params, doseq=True)
        return hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

    def _signed_request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        params = dict(params or {})
        params["timestamp"] = self._timestamp()
        params["recvWindow"] = 5000
        params["signature"] = self._sign(params)
        url = f"{self.base_url}{path}"
        if method == "GET":
            r = self.session.get(url, params=params, timeout=8)
        elif method == "POST":
            r = self.session.post(url, params=params, timeout=8)
        elif method == "DELETE":
            r = self.session.delete(url, params=params, timeout=8)
        else:
            raise ValueError(f"unsupported method: {method}")
        
        # 【关键】解析 Binance API 错误信息
        if r.status_code >= 400:
            try:
                error_body = r.json()
                error_code = error_body.get("code", "?")
                error_msg = error_body.get("msg", r.text[:200])
                # 显示发送的参数（隐藏签名）
                safe_params = {k: v for k, v in params.items() if k != "signature"}
                print(f"[BinanceAPI] ❌ {method} {path} 失败")
                print(f"[BinanceAPI] 错误码: {error_code} | 消息: {error_msg}")
                print(f"[BinanceAPI] 请求参数: {safe_params}")
                raise Exception(f"Binance API {error_code}: {error_msg}")
            except Exception as e:
                if "Binance API" in str(e):
                    raise
                # JSON 解析失败，回退到原始错误
                r.raise_for_status()
        
        return r.json()

    @staticmethod
    def _trade_side(trade: dict) -> Optional[str]:
        """兼容返回结构，提取成交方向 BUY/SELL"""
        if "side" in trade:
            return str(trade.get("side", "")).upper()
        if "buyer" in trade:
            return "BUY" if trade.get("buyer") else "SELL"
        return None

    @staticmethod
    def _to_float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except Exception:
            return default

    def _get_user_trades(self, start_time_ms: Optional[int] = None,
                         limit: int = 200, order_id: Optional[int] = None) -> List[dict]:
        """拉取成交明细（真实撮合）"""
        params = {"symbol": self.symbol, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if order_id is not None:
            params["orderId"] = int(order_id)
        try:
            return self._signed_request("GET", "/fapi/v1/userTrades", params)
        except Exception:
            # 兼容部分测试网不支持 orderId 的情况
            if order_id is not None:
                params.pop("orderId", None)
                return self._signed_request("GET", "/fapi/v1/userTrades", params)
            return []

    def _aggregate_trades(self, trades: List[dict], entry_side: str) -> Dict[str, float]:
        """聚合成交（真实成交均价 / 手续费 / 已实现盈亏）"""
        if not trades:
            return {"exit_price": 0.0, "exit_fee": 0.0, "entry_fee": 0.0,
                    "realized_pnl": 0.0, "last_time_ms": 0}

        entry_side = entry_side.upper()
        exit_side = "SELL" if entry_side == "BUY" else "BUY"

        entry_trades = []
        exit_trades = []
        realized_pnl = 0.0
        entry_fee = 0.0
        exit_fee = 0.0
        last_time_ms = 0

        for t in trades:
            side = self._trade_side(t) or ""
            qty = self._to_float(t.get("qty", t.get("executedQty", 0.0)))
            price = self._to_float(t.get("price", 0.0))
            commission = self._to_float(t.get("commission", 0.0))
            pnl = self._to_float(t.get("realizedPnl", 0.0))
            trade_time = int(t.get("time", 0) or 0)
            if trade_time > last_time_ms:
                last_time_ms = trade_time

            if side == entry_side:
                entry_trades.append((price, qty))
                entry_fee += commission
            elif side == exit_side:
                exit_trades.append((price, qty))
                exit_fee += commission
            # realizedPnl 通常只在减仓/平仓产生
            realized_pnl += pnl

        def _weighted_avg(rows: List[tuple]) -> float:
            total_qty = sum(q for _, q in rows)
            if total_qty <= 1e-12:
                return 0.0
            return sum(p * q for p, q in rows) / total_qty

        exit_price = _weighted_avg(exit_trades)
        if exit_price <= 0.0:
            # 若无法区分方向，退化为所有成交均价
            exit_price = _weighted_avg([(self._to_float(t.get("price", 0.0)),
                                         self._to_float(t.get("qty", t.get("executedQty", 0.0))))
                                        for t in trades])

        return {
            "exit_price": exit_price,
            "exit_fee": exit_fee,
            "entry_fee": entry_fee,
            "realized_pnl": realized_pnl,
            "last_time_ms": last_time_ms,
        }

    def _public_get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=params or {}, timeout=8)
        r.raise_for_status()
        return r.json()

    def _load_symbol_filters(self):
        data = self._public_get("/fapi/v1/exchangeInfo")
        symbol_info = None
        for s in data.get("symbols", []):
            if s.get("symbol") == self.symbol:
                symbol_info = s
                break
        if symbol_info is None:
            return
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                self._qty_step = float(f.get("stepSize", "0.001"))
                self._qty_min = float(f.get("minQty", "0.001"))
            elif f.get("filterType") == "PRICE_FILTER":
                self._price_tick = float(f.get("tickSize", "0.1"))
            elif f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                self._min_notional = float(f.get("notional", f.get("minNotional", "5.0")))

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        n = int(value / step)
        return max(step, n * step)

    def _load_history(self):
        """从 JSON 文件加载持久化的记录"""
        if not os.path.exists(self.history_file):
            return
        
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                trades_data = data.get("trades", [])
                
                # 恢复全局统计（尤其是初始资金，防止盈利率错误）
                saved_stats = data.get("stats", {})
                if saved_stats:
                    self.stats.initial_balance = float(data.get("initial_balance", self.initial_balance))
                    self.stats.max_balance = float(saved_stats.get("max_balance", self.stats.initial_balance))
                    self.stats.total_trades = int(saved_stats.get("total_trades", 0))
                
                # 转换回 PaperOrder 对象
                loaded_history = []
                for t in trades_data:
                    order = PaperOrder(
                        order_id=t["order_id"],
                        symbol=t["symbol"],
                        side=OrderSide(t["side"]),
                        quantity=t["quantity"],
                        margin_used=t["margin_used"],
                        entry_price=t["entry_price"],
                        entry_time=datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else None,
                        entry_bar_idx=t.get("entry_bar_idx", 0),
                        take_profit=t.get("take_profit"),
                        stop_loss=t.get("stop_loss"),
                        status=OrderStatus(t["status"]),
                        exit_price=t.get("exit_price"),
                        exit_time=datetime.fromisoformat(t["exit_time"]) if t.get("exit_time") else None,
                        exit_bar_idx=t.get("exit_bar_idx"),
                        close_reason=CloseReason(t["close_reason"]) if t.get("close_reason") else None,
                        realized_pnl=t.get("realized_pnl", 0.0),
                        profit_pct=t.get("profit_pct", 0.0),
                        template_fingerprint=t.get("template_fingerprint"),
                        entry_similarity=t.get("entry_similarity", 0.0),
                        entry_reason=t.get("entry_reason", ""),
                        hold_bars=t.get("hold_bars", 0)
                    )
                    loaded_history.append(order)
                    
                    # 恢复模板性能统计
                    if order.template_fingerprint:
                        self._record_template_performance(order)
                
                self.order_history = loaded_history
                print(f"[BinanceTrader] 成功从本地加载 {len(self.order_history)} 条历史交易记录 (数据底座: ${self.stats.initial_balance:.2f})")
                
                # 更新账户统计
                self._update_stats_from_exchange()
                
        except Exception as e:
            print(f"[BinanceTrader] 加载历史记录失败: {e}")
            import traceback
            traceback.print_exc()

    def _set_leverage(self, leverage: int):
        try:
            self._signed_request("POST", "/fapi/v1/leverage", {
                "symbol": self.symbol,
                "leverage": int(leverage),
            })
        except Exception:
            pass

    def _get_usdt_balance(self) -> float:
        rows = self._signed_request("GET", "/fapi/v2/balance")
        for row in rows:
            if row.get("asset") == "USDT":
                bal = float(row.get("balance", 0.0))
                self.stats.current_balance = bal
                return bal
        return 0.0

    def _get_usdt_available_balance(self) -> float:
        rows = self._signed_request("GET", "/fapi/v2/balance")
        for row in rows:
            if row.get("asset") == "USDT":
                # Binance 下单应使用可用余额，而不是总余额
                return float(row.get("availableBalance", row.get("balance", 0.0)))
        return 0.0

    def _get_mark_price(self) -> float:
        data = self._public_get("/fapi/v1/premiumIndex", {"symbol": self.symbol})
        return float(data.get("markPrice", 0.0))

    def _get_position(self) -> dict:
        rows = self._signed_request("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol})
        if isinstance(rows, list) and rows:
            return rows[0]
        return {}

    def _infer_close_reason(self, order: PaperOrder, exit_price: float) -> CloseReason:
        """
        根据平仓价格和订单的TP/SL设置，推断真正的平仓原因
        
        逻辑：
        1. 如果平仓价在止盈价附近（±0.1%），认为是止盈
        2. 如果平仓价在止损价附近（±0.1%），认为是止损
        3. 如果盈亏符合止盈方向，认为是止盈
        4. 如果盈亏符合止损方向，认为是止损
        5. 否则标记为"未知"（但用SIGNAL代替，因为可能是追踪止损等情况）
        """
        tolerance = 0.001  # 0.1% 容差
        
        # 检查是否触及止盈
        if order.take_profit is not None:
            tp = order.take_profit
            if abs(exit_price - tp) / tp < tolerance:
                return CloseReason.TAKE_PROFIT
            # LONG: 平仓价 >= TP 表示止盈触发
            # SHORT: 平仓价 <= TP 表示止盈触发
            if order.side == OrderSide.LONG and exit_price >= tp:
                return CloseReason.TAKE_PROFIT
            if order.side == OrderSide.SHORT and exit_price <= tp:
                return CloseReason.TAKE_PROFIT
        
        # 检查是否触及止损（区分追踪止损和真正止损）
        if order.stop_loss is not None:
            sl = order.stop_loss
            # 判断SL是否在盈利区（追踪止损/保本止损）
            is_profit_sl = (
                (order.side == OrderSide.LONG and sl >= order.entry_price) or
                (order.side == OrderSide.SHORT and sl <= order.entry_price)
            )
            sl_reason = CloseReason.TRAILING_STOP if is_profit_sl else CloseReason.STOP_LOSS
            
            if abs(exit_price - sl) / sl < tolerance:
                return sl_reason
            # LONG: 平仓价 <= SL 表示止损触发
            # SHORT: 平仓价 >= SL 表示止损触发
            if order.side == OrderSide.LONG and exit_price <= sl:
                return sl_reason
            if order.side == OrderSide.SHORT and exit_price >= sl:
                return sl_reason
        
        # 无法确定原因：退出价既不在TP也不在SL附近
        # 使用 EXCHANGE_CLOSE 标记，表示“交易所侧被动平仓（非本系统主动触发）”
        return CloseReason.EXCHANGE_CLOSE
    
    def _fetch_real_close_reason(self, order: PaperOrder) -> CloseReason:
        """从交易所查询真实平仓原因（最准确的诊断方法）"""
        try:
            start_time = int((time.time() - 300) * 1000)
            if order.entry_time:
                start_time = max(start_time, int(order.entry_time.timestamp() * 1000) - 1000)
            
            trades = self._signed_request("GET", "/fapi/v1/userTrades", {
                "symbol": self.symbol,
                "limit": 20,
                "startTime": start_time
            })
            
            if not trades:
                print("[平仓诊断] ⚠ 未找到成交记录，回退价格推断")
                return self._infer_close_reason(order, order.entry_price)
            
            close_trades = [t for t in trades if float(t.get("realizedPnl", "0")) != 0]
            if not close_trades:
                print("[平仓诊断] ⚠ 未找到平仓成交，回退价格推断")
                return self._infer_close_reason(order, order.entry_price)
            
            last_close = close_trades[-1]
            order_id = last_close.get("orderId")
            realized_pnl = float(last_close.get("realizedPnl", "0"))
            exit_price = float(last_close.get("price", "0"))
            
            order_info = self._signed_request("GET", "/fapi/v1/order", {
                "symbol": self.symbol,
                "orderId": order_id
            })
            
            order_type = order_info.get("type", "")
            status = order_info.get("status", "")
            orig_type = order_info.get("origType", "")
            
            print(f"[平仓诊断] 交易所详情: orderId={order_id}, type={order_type}, "
                  f"origType={orig_type}, exit_price={exit_price:.2f}, pnl={realized_pnl:+.4f}")
            
            if "STOP" in order_type or "STOP_MARKET" in order_type:
                print("[平仓诊断] ✓ 确认止损触发")
                return CloseReason.STOP_LOSS
            
            if "TAKE_PROFIT" in order_type:
                print("[平仓诊断] ✓ 确认止盈触发")
                return CloseReason.TAKE_PROFIT
            
            if "LIQUIDATION" in order_type or "LIQUIDATION" in orig_type:
                print("[平仓诊断] ⚠️ 强制平仓（爆仓）")
                return CloseReason.STOP_LOSS
            
            if order_type == "MARKET" and status == "FILLED":
                inferred = self._infer_close_reason(order, exit_price)
                if inferred in (CloseReason.TAKE_PROFIT, CloseReason.STOP_LOSS):
                    print(f"[平仓诊断] ℹ️ 市价单，价格触及{inferred.value}")
                    return inferred
                print("[平仓诊断] ℹ️ 市价单平仓（手动或ADL）")
                return CloseReason.EXCHANGE_CLOSE
            
            if order_type == "LIMIT" and status == "FILLED":
                inferred = self._infer_close_reason(order, exit_price)
                print(f"[平仓诊断] ℹ️ 限价单，推断为{inferred.value}")
                return inferred
            
            print(f"[平仓诊断] ⚠ 未识别类型{order_type}，回退价格推断")
            return self._infer_close_reason(order, exit_price)
            
        except Exception as e:
            print(f"[平仓诊断] ❌ 查询失败: {e}")
            return self._infer_close_reason(order, order.entry_price)

    def _sync_from_exchange(self, force: bool = False):
        """从交易所同步余额/持仓，确保UI与币安账户一致"""
        now = time.time()
        if (not force) and (now - self._last_sync_ts < self._sync_interval_sec):
            return
        self._last_sync_ts = now

        bal = self._get_usdt_balance()
        self.stats.current_balance = bal
        self.stats.total_pnl = bal - self.stats.initial_balance
        if self.stats.initial_balance > 0:
            self.stats.total_pnl_pct = (bal / self.stats.initial_balance - 1.0) * 100.0

        pos = self._get_position()
        amt = float(pos.get("positionAmt", 0.0)) if pos else 0.0
        if abs(amt) < 1e-12:
            # 检测"之前有仓 -> 交易所已无仓"的转变，兜底触发平仓回调
            prev_pos = self.current_position
            self.current_position = None
            # 【修复】仓位已消失，清除可能残留的 _pending_close，
            # 防止旧的平仓重试指令误杀下一个新仓位
            self._pending_close = None
            if prev_pos is not None and prev_pos.status != OrderStatus.CLOSED:
                # 使用真实成交记录计算盈亏与费用
                entry_time_ms = int(prev_pos.entry_time.timestamp() * 1000) - 1000
                entry_side = "BUY" if prev_pos.side == OrderSide.LONG else "SELL"
                trades = self._get_user_trades(start_time_ms=entry_time_ms)
                agg = self._aggregate_trades(trades, entry_side=entry_side)

                exit_price = agg["exit_price"] or prev_pos.entry_price
                exit_fee = agg["exit_fee"]
                entry_fee = prev_pos.total_fee or agg["entry_fee"]
                realized_pnl = agg["realized_pnl"]
                net_pnl = realized_pnl - exit_fee - entry_fee
                exit_time = datetime.fromtimestamp(agg["last_time_ms"] / 1000) if agg["last_time_ms"] > 0 else datetime.now()

                # 【核心改进】优先检查交易所保护单是否成交来确定平仓原因
                close_reason = self._detect_tp_sl_fill()
                if close_reason:
                    print(f"[BinanceTrader] 📍 仓位由交易所保护单平仓: {close_reason.value}")
                else:
                    # 保护单未成交，走原有诊断流程
                    close_reason = self._fetch_real_close_reason(prev_pos)

                # 诊断：检查交易所是否有残留的反向入场单成交
                exit_side_str = "SELL" if entry_side == "BUY" else "BUY"
                exit_trade_count = sum(1 for t in trades if (self._trade_side(t) or "") == exit_side_str)
                has_stale_entry = False
                stale_detail = ""
                for t in trades:
                    cid = t.get("clientOrderId", "") or ""
                    t_side = self._trade_side(t) or ""
                    if t_side == exit_side_str and ("ENTRY_LIMIT" in cid or "ENTRY_STOP" in cid):
                        has_stale_entry = True
                        stale_detail = f"残留入场单成交: {cid} side={t_side} qty={t.get('qty')}"
                        break

                # 构建详细的诊断原因
                diag_parts = []
                if has_stale_entry:
                    diag_parts.append(f"[根因] {stale_detail}")
                elif exit_trade_count > 0:
                    diag_parts.append(f"[根因] 交易所有{exit_trade_count}笔{exit_side_str}成交记录")
                else:
                    diag_parts.append("[根因] 交易所无出场成交记录，可能API返回异常或ADL")
                diag_parts.append(f"入场={prev_pos.entry_price:.2f} 出场={exit_price:.2f}")
                diag_parts.append(f"TP={prev_pos.take_profit} SL={prev_pos.stop_loss}")
                diag_parts.append(f"hold_bars={prev_pos.hold_bars}")
                decision_detail = " | ".join(diag_parts)
                
                # 清理残留保护单（另一个保护单可能还在挂着）
                self._cancel_exchange_tp_sl(silent=False)

                prev_pos.status = OrderStatus.CLOSED
                prev_pos.exit_price = exit_price
                prev_pos.exit_time = exit_time
                prev_pos.exit_bar_idx = self.current_bar_idx
                prev_pos.close_reason = close_reason
                prev_pos.realized_pnl = net_pnl
                prev_pos.unrealized_pnl = 0.0
                margin_used = prev_pos.margin_used if prev_pos.margin_used > 0 else 1.0
                prev_pos.profit_pct = (net_pnl / margin_used) * 100.0
                prev_pos.total_fee = entry_fee + exit_fee
                prev_pos.decision_reason = f"[交易所同步平仓] {decision_detail}"

                self.order_history.append(prev_pos)
                # 持久化：防止停止程序时丢记录
                self.save_history(self.history_file)
                print(f"[BinanceTrader] ⚠ 交易所仓位已消失: "
                      f"{prev_pos.side.value} PnL={prev_pos.realized_pnl:+.2f} USDT | "
                      f"{decision_detail}")
                if self.on_trade_closed:
                    self.on_trade_closed(prev_pos)
            return

        side = OrderSide.LONG if amt > 0 else OrderSide.SHORT
        qty = abs(amt)
        entry = float(pos.get("entryPrice", 0.0))
        mark = float(pos.get("markPrice", entry or 0.0))
        leverage = float(pos.get("leverage", self.leverage))
        margin = abs(entry * qty) / max(leverage, 1.0)
        pnl = (mark - entry) * qty if side == OrderSide.LONG else (entry - mark) * qty
        pnl_pct = (pnl / margin * 100.0) if margin > 1e-9 else 0.0

        # 如果已有本地持仓且方向/入场价一致，只更新行情数据，保留追踪状态
        existing = self.current_position
        if (existing is not None
                and existing.side == side
                and abs(existing.entry_price - entry) < 0.01):
            # 更新行情相关字段，保留所有追踪状态（trailing_stage, peak_price等）
            existing.quantity = qty
            existing.margin_used = margin
            existing.unrealized_pnl = pnl
            existing.profit_pct = pnl_pct
            # 若本地缺失TP/SL，则尝试回填（来自最新入场信号）
            tp_filled = False
            sl_filled = False
            if existing.take_profit is None and self._last_entry_tp is not None:
                existing.take_profit = self._last_entry_tp
                tp_filled = True
            if existing.stop_loss is None and self._last_entry_sl is not None:
                existing.stop_loss = self._last_entry_sl
                sl_filled = True
            # TP/SL回填后，如果交易所还没有保护单，立即挂上
            if (tp_filled or sl_filled):
                has_sl = self._exchange_sl_order_id and self._exchange_sl_order_id > 0
                has_tp = self._exchange_tp_order_id and self._exchange_tp_order_id > 0
                if not (has_sl and has_tp):
                    self._place_exchange_tp_sl(existing)
                if existing.original_stop_loss is None:
                    existing.original_stop_loss = self._last_entry_sl
            # 更新峰值追踪
            if side == OrderSide.LONG:
                if mark > existing.peak_price:
                    existing.peak_price = mark
            else:
                if existing.peak_price == 0 or mark < existing.peak_price:
                    existing.peak_price = mark
            if pnl_pct > existing.peak_profit_pct:
                existing.peak_profit_pct = pnl_pct
        else:
            # 新仓位（首次发现或方向变了），创建新对象
            # 回填最近一次入场的TP/SL（仅限“刚由本系统触发”的仓位）
            # 避免手动仓位错误继承旧TP/SL，导致异常快速止损
            entry_tp = None
            entry_sl = None
            entry_bar_idx = self.current_bar_idx
            entry_fp = None
            entry_sim = 0.0
            entry_reason = ""
            if self._entry_stop_orders:
                last_entry = self._entry_stop_orders[-1]
                entry_tp = last_entry.get("take_profit")
                entry_sl = last_entry.get("stop_loss")
                entry_bar_idx = int(last_entry.get("start_bar", self.current_bar_idx))
                entry_fp = last_entry.get("template_fingerprint")
                entry_sim = float(last_entry.get("entry_similarity", 0.0) or 0.0)
                entry_reason = last_entry.get("entry_reason", "")
            else:
                # 没有挂单记录时，只允许短时间内且方向/价格近似一致才回填
                recent_window_sec = 180.0
                is_recent = (time.time() - self._last_entry_ts) <= recent_window_sec
                side_match = (self._last_entry_side == side) if self._last_entry_side is not None else False
                price_match = False
                if self._last_entry_price is not None and entry > 0:
                    price_match = abs(self._last_entry_price - entry) / entry <= 0.005  # 0.5%
                if is_recent and side_match and price_match:
                    entry_tp = self._last_entry_tp
                    entry_sl = self._last_entry_sl
                    entry_bar_idx = self.current_bar_idx

            self.current_position = PaperOrder(
                order_id="EXCHANGE_SYNC",
                symbol=self.symbol,
                side=side,
                quantity=qty,
                margin_used=margin,
                entry_price=entry,
                entry_time=datetime.now(),
                entry_bar_idx=entry_bar_idx,
                take_profit=entry_tp,
                stop_loss=entry_sl,
                original_stop_loss=entry_sl,
                unrealized_pnl=pnl,
                profit_pct=pnl_pct,
                peak_price=mark,
                template_fingerprint=entry_fp,
                entry_similarity=entry_sim,
                entry_reason=entry_reason,
            )
            
            # 【核心】新仓位同步后，如果有TP/SL，立即挂交易所保护单
            if (entry_tp is not None or entry_sl is not None):
                # 检查是否已有保护单（避免重复挂）
                has_existing_sl = self._exchange_sl_order_id and self._exchange_sl_order_id > 0
                has_existing_tp = self._exchange_tp_order_id and self._exchange_tp_order_id > 0
                if not (has_existing_sl and has_existing_tp):
                    print(f"[BinanceTrader] 同步检测到新仓位，挂交易所保护单...")
                    self._place_exchange_tp_sl(self.current_position)
        
        # 若交易所已有持仓，说明入场单已成交或不再有效
        # 【关键修复】不能只清本地列表！必须同时取消交易所上的挂单
        # 否则旧的反方向入场单可能仍在交易所上，一旦成交就会平掉当前仓位
        # 注意：即使本地列表为空也要检查交易所，因为列表可能已被之前的sync清空
        try:
            open_orders = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol})
            for o in open_orders:
                client_id = o.get("clientOrderId", "")
                if "ENTRY_LIMIT" in client_id or "ENTRY_STOP" in client_id:
                    try:
                        self._signed_request("DELETE", "/fapi/v1/order", {
                            "symbol": self.symbol,
                            "orderId": o["orderId"]
                        })
                        print(f"[BinanceTrader] 持仓同步：撤销残留入场单 {client_id}")
                    except Exception as ce:
                        print(f"[BinanceTrader] 撤销残留入场单失败: {ce}")
        except Exception as e:
            print(f"[BinanceTrader] 查询残留入场单失败: {e}")
        if self._entry_stop_orders:
            self._entry_stop_orders.clear()

    def has_position(self) -> bool:
        return self.current_position is not None

    def has_pending_stop_orders(self, current_bar_idx: int = None) -> bool:
        """检查是否有活跃的入场挂单（优先本地记录，必要时查交易所）"""
        # 先用本地挂单缓存，避免频繁 API 查询
        if self._entry_stop_orders:
            if current_bar_idx is None:
                return True
            valid = [o for o in self._entry_stop_orders if current_bar_idx <= o.get("expire_bar", -1)]
            if valid:
                return True
        try:
            # 获取所有挂单
            open_orders = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol})
            # 查找带有 ENTRY_LIMIT 或 ENTRY_STOP 前缀的挂单（兼容旧版本）
            for o in open_orders:
                client_id = o.get("clientOrderId", "")
                if "ENTRY_LIMIT" in client_id or "ENTRY_STOP" in client_id:
                    return True
            return False
        except Exception as e:
            print(f"[BinanceTrader] 检查挂单失败: {e}")
            return False

    def cancel_entry_stop_orders(self):
        """取消所有挂起的入场挂单"""
        try:
            open_orders = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol})
            for o in open_orders:
                client_id = o.get("clientOrderId", "")
                # 兼容新旧版本的订单前缀
                if "ENTRY_LIMIT" in client_id or "ENTRY_STOP" in client_id:
                    print(f"[BinanceTrader] 正在撤销过期/替换入场单: {client_id}")
                    self._signed_request("DELETE", "/fapi/v1/order", {
                        "symbol": self.symbol,
                        "orderId": o["orderId"]
                    })
            self._entry_stop_orders.clear()
        except Exception as e:
            print(f"[BinanceTrader] 撤销入场单失败: {e}")
    
    def cancel_expired_entry_stop_orders(self, current_bar_idx: int):
        """超时撤销入场止损单"""
        if not self._entry_stop_orders:
            return
        remaining = []
        for o in self._entry_stop_orders:
            expire_bar = o.get("expire_bar", -1)
            if current_bar_idx <= expire_bar:
                remaining.append(o)
                continue
            order_id = o.get("order_id")
            client_id = o.get("client_id")
            try:
                print(f"[BinanceTrader] 入场单超时撤销: {client_id or order_id}")
                params = {"symbol": self.symbol}
                if order_id:
                    params["orderId"] = order_id
                elif client_id:
                    params["origClientOrderId"] = client_id
                else:
                    remaining.append(o)
                    continue
                self._signed_request("DELETE", "/fapi/v1/order", params)
            except Exception as e:
                print(f"[BinanceTrader] 撤销超时入场单失败: {e}")
        self._entry_stop_orders = remaining

    def _new_client_order_id(self, prefix: str) -> str:
        self._order_counter += 1
        return f"R3000_{prefix}_{int(time.time())}_{self._order_counter}"

    def _place_order(self, params: dict) -> dict:
        return self._signed_request("POST", "/fapi/v1/order", params)

    # ═══════════════════════════════════════════════════════════════════
    #  交易所端止盈止损保护单管理
    # ═══════════════════════════════════════════════════════════════════

    def _place_exchange_tp_sl(self, order: PaperOrder) -> None:
        """
        在交易所挂止盈止损保护单（STOP_MARKET + TAKE_PROFIT_MARKET）
        
        这是系统最重要的安全机制：
        - 即使程序崩溃、网络断开，交易所也会自动执行保护
        - 使用 closePosition=true，不需要指定数量
        - 开仓后必须立即调用此方法
        """
        if order is None:
            return
        
        exit_side = "SELL" if order.side == OrderSide.LONG else "BUY"
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        
        # ── 先清除可能残留的旧保护单 ──
        self._cancel_exchange_tp_sl(silent=True)
        
        # ── 挂止损单 STOP_MARKET ──
        if order.stop_loss is not None and order.stop_loss > 0:
            sl_price_str = f"{self._round_step(order.stop_loss, self._price_tick):.{p_prec}f}"
            try:
                sl_resp = self._place_order({
                    "symbol": self.symbol,
                    "side": exit_side,
                    "type": "STOP_MARKET",
                    "stopPrice": sl_price_str,
                    "closePosition": "true",
                    "workingType": "CONTRACT_PRICE",
                    "newClientOrderId": self._new_client_order_id("SL"),
                })
                self._exchange_sl_order_id = int(sl_resp.get("orderId", 0) or 0)
                self._exchange_sl_price = order.stop_loss
                self._last_sl_update_ts = time.time()
                print(f"[BinanceTrader] ✅ 交易所止损单已挂: {exit_side} STOP_MARKET @ {sl_price_str} "
                      f"| orderId={self._exchange_sl_order_id}")
            except Exception as e:
                print(f"[BinanceTrader] ❌ 挂止损单失败: {e}")
                self._exchange_sl_order_id = None
                self._exchange_sl_price = 0.0
        
        # ── 挂止盈单 TAKE_PROFIT_MARKET ──
        if order.take_profit is not None and order.take_profit > 0:
            tp_price_str = f"{self._round_step(order.take_profit, self._price_tick):.{p_prec}f}"
            try:
                tp_resp = self._place_order({
                    "symbol": self.symbol,
                    "side": exit_side,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": tp_price_str,
                    "closePosition": "true",
                    "workingType": "CONTRACT_PRICE",
                    "newClientOrderId": self._new_client_order_id("TP"),
                })
                self._exchange_tp_order_id = int(tp_resp.get("orderId", 0) or 0)
                self._exchange_tp_price = order.take_profit
                print(f"[BinanceTrader] ✅ 交易所止盈单已挂: {exit_side} TAKE_PROFIT_MARKET @ {tp_price_str} "
                      f"| orderId={self._exchange_tp_order_id}")
            except Exception as e:
                print(f"[BinanceTrader] ❌ 挂止盈单失败: {e}")
                self._exchange_tp_order_id = None
                self._exchange_tp_price = 0.0
        
        # ── 结果汇报 ──
        has_sl = self._exchange_sl_order_id is not None and self._exchange_sl_order_id > 0
        has_tp = self._exchange_tp_order_id is not None and self._exchange_tp_order_id > 0
        if has_sl and has_tp:
            print(f"[BinanceTrader] 🛡️ 交易所双重保护已就位: SL={order.stop_loss:.2f} TP={order.take_profit:.2f}")
        elif has_sl:
            print(f"[BinanceTrader] ⚠ 仅止损保护: SL={order.stop_loss:.2f}，止盈单挂失败")
        elif has_tp:
            print(f"[BinanceTrader] ⚠ 仅止盈保护: TP={order.take_profit:.2f}，止损单挂失败")
        else:
            print(f"[BinanceTrader] 🚨 交易所保护单全部挂失败！仓位处于裸风险状态！")

    def _cancel_exchange_tp_sl(self, silent: bool = False) -> None:
        """取消交易所上的止盈止损保护单"""
        for attr, label in [("_exchange_sl_order_id", "止损"), ("_exchange_tp_order_id", "止盈")]:
            order_id = getattr(self, attr, None)
            if order_id and order_id > 0:
                try:
                    self._signed_request("DELETE", "/fapi/v1/order", {
                        "symbol": self.symbol,
                        "orderId": order_id,
                    })
                    if not silent:
                        print(f"[BinanceTrader] 🔄 已取消交易所{label}单 orderId={order_id}")
                except Exception as e:
                    # 订单可能已被执行或已取消，忽略错误
                    if not silent:
                        print(f"[BinanceTrader] ⚠ 取消{label}单异常(可能已成交): {e}")
                setattr(self, attr, None)
        
        self._exchange_sl_price = 0.0
        self._exchange_tp_price = 0.0

    def _update_exchange_sl(self, new_sl: float) -> bool:
        """
        更新交易所止损价（追踪止损时调用）
        
        返回 True 表示更新成功
        """
        if new_sl <= 0:
            return False
        
        # 节流：避免频繁更新API
        now = time.time()
        if now - self._last_sl_update_ts < self._sl_update_min_interval:
            return False
        
        # 价格未变化或变化太小，跳过
        if abs(new_sl - self._exchange_sl_price) < self._price_tick * 0.5:
            return False
        
        order = self.current_position
        if order is None:
            return False
        
        exit_side = "SELL" if order.side == OrderSide.LONG else "BUY"
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        sl_price_str = f"{self._round_step(new_sl, self._price_tick):.{p_prec}f}"
        
        # 取消旧止损单
        if self._exchange_sl_order_id and self._exchange_sl_order_id > 0:
            try:
                self._signed_request("DELETE", "/fapi/v1/order", {
                    "symbol": self.symbol,
                    "orderId": self._exchange_sl_order_id,
                })
            except Exception:
                pass  # 可能已成交
            self._exchange_sl_order_id = None
        
        # 挂新止损单
        try:
            sl_resp = self._place_order({
                "symbol": self.symbol,
                "side": exit_side,
                "type": "STOP_MARKET",
                "stopPrice": sl_price_str,
                "closePosition": "true",
                "workingType": "CONTRACT_PRICE",
                "newClientOrderId": self._new_client_order_id("SL_UPD"),
            })
            self._exchange_sl_order_id = int(sl_resp.get("orderId", 0) or 0)
            self._exchange_sl_price = new_sl
            self._last_sl_update_ts = now
            print(f"[BinanceTrader] 🔄 交易所止损已更新: SL={sl_price_str}")
            return True
        except Exception as e:
            print(f"[BinanceTrader] ❌ 更新止损失败: {e}")
            return False

    def _update_exchange_tp(self, new_tp: float) -> bool:
        """更新交易所止盈价"""
        if new_tp <= 0:
            return False
        
        if abs(new_tp - self._exchange_tp_price) < self._price_tick * 0.5:
            return False
        
        order = self.current_position
        if order is None:
            return False
        
        exit_side = "SELL" if order.side == OrderSide.LONG else "BUY"
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        tp_price_str = f"{self._round_step(new_tp, self._price_tick):.{p_prec}f}"
        
        # 取消旧止盈单
        if self._exchange_tp_order_id and self._exchange_tp_order_id > 0:
            try:
                self._signed_request("DELETE", "/fapi/v1/order", {
                    "symbol": self.symbol,
                    "orderId": self._exchange_tp_order_id,
                })
            except Exception:
                pass
            self._exchange_tp_order_id = None
        
        # 挂新止盈单
        try:
            tp_resp = self._place_order({
                "symbol": self.symbol,
                "side": exit_side,
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": tp_price_str,
                "closePosition": "true",
                "workingType": "CONTRACT_PRICE",
                "newClientOrderId": self._new_client_order_id("TP_UPD"),
            })
            self._exchange_tp_order_id = int(tp_resp.get("orderId", 0) or 0)
            self._exchange_tp_price = new_tp
            print(f"[BinanceTrader] 🔄 交易所止盈已更新: TP={tp_price_str}")
            return True
        except Exception as e:
            print(f"[BinanceTrader] ❌ 更新止盈失败: {e}")
            return False

    def _detect_tp_sl_fill(self) -> Optional[CloseReason]:
        """
        检测交易所止盈/止损单是否已成交
        
        Returns:
            CloseReason if a protective order was filled, None otherwise
        """
        for order_id, reason, label in [
            (self._exchange_sl_order_id, CloseReason.STOP_LOSS, "止损"),
            (self._exchange_tp_order_id, CloseReason.TAKE_PROFIT, "止盈"),
        ]:
            if not order_id or order_id <= 0:
                continue
            try:
                info = self._signed_request("GET", "/fapi/v1/order", {
                    "symbol": self.symbol,
                    "orderId": order_id,
                })
                status = str(info.get("status", ""))
                if status == "FILLED":
                    print(f"[BinanceTrader] 📍 交易所{label}单已成交: orderId={order_id}")
                    
                    # 判断是否是追踪止损（SL在盈利区）
                    if reason == CloseReason.STOP_LOSS and self.current_position:
                        order = self.current_position
                        sl = self._exchange_sl_price
                        is_profit_sl = (
                            (order.side == OrderSide.LONG and sl >= order.entry_price) or
                            (order.side == OrderSide.SHORT and sl <= order.entry_price)
                        )
                        if is_profit_sl:
                            return CloseReason.TRAILING_STOP
                    
                    return reason
            except Exception:
                pass
        return None

    def _cleanup_orphan_tp_sl(self) -> None:
        """
        清除交易所上与 R3000 相关的残留保护单
        （程序重启时调用，避免旧订单干扰新仓位）
        """
        try:
            open_orders = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol})
            for o in open_orders:
                client_id = str(o.get("clientOrderId", ""))
                order_type = str(o.get("type", ""))
                # 识别 R3000 的保护单
                if any(tag in client_id for tag in ["R3000_SL", "R3000_TP", "R3K_SL", "R3K_TP"]):
                    try:
                        self._signed_request("DELETE", "/fapi/v1/order", {
                            "symbol": self.symbol,
                            "orderId": o["orderId"],
                        })
                        print(f"[BinanceTrader] 🧹 清除残留保护单: {client_id} ({order_type})")
                    except Exception as e:
                        print(f"[BinanceTrader] ⚠ 清除残留单失败: {e}")
                elif order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                    # 非 R3000 的保护单，也输出告知
                    print(f"[BinanceTrader] ℹ 发现非本系统保护单: orderId={o.get('orderId')} type={order_type}")
        except Exception as e:
            print(f"[BinanceTrader] ⚠ 检查残留保护单失败: {e}")

    def _calc_entry_quantity(self, price: float, position_size_pct: Optional[float] = None) -> float:
        """计算开仓数量（支持动态仓位比例）"""
        # 关键：按可用余额计算，避免 balance 包含被占用资金导致 -2019
        avail = self._get_usdt_available_balance()
        pct = position_size_pct if position_size_pct is not None else self.position_size_pct
        margin = avail * pct
        # 给手续费/滑点/撮合波动留缓冲，避免“刚好全仓”被拒
        safety_factor = max(0.90, 1.0 - self.fee_rate * 3 - 0.01)  # 默认约 98.88%
        effective_margin = margin * safety_factor
        notional = effective_margin * self.leverage
        raw_qty = notional / max(price, 1e-9)
        qty = self._round_step(raw_qty, self._qty_step)
        if qty < self._qty_min:
            qty = self._qty_min
        # 确保满足最小名义价值要求（通常 5 USDT）
        if qty * price < self._min_notional:
            qty = self._round_step((self._min_notional / max(price, 1e-9)) * 1.02, self._qty_step)
            if qty < self._qty_min:
                qty = self._qty_min
        return qty

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
                         position_size_pct: Optional[float] = None) -> Optional[str]:
        """
        放置限价开仓单 (LIMIT + GTC)
        在 trigger_price 挂限价单，等待价格触及成交（争取 Maker 0.02%）
        超时未成交会自动撤单
        
        Args:
            position_size_pct: 仓位比例（None=使用默认配置，凯利公式动态调整时传入）
        """
        self._sync_from_exchange(force=True)
        if self.current_position is not None:
            return None

        qty = self._calc_entry_quantity(trigger_price, position_size_pct)
        side_str = "BUY" if side == OrderSide.LONG else "SELL"
        
        # 格式化
        precision = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
        qty_str = f"{qty:.{precision}f}"
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        trigger_str = f"{trigger_price:.{p_prec}f}"

        pct_used = position_size_pct if position_size_pct is not None else self.position_size_pct
        print(f"[BinanceTrader] 放置限价开仓单: {side_str} {qty_str} @ {trigger_str} (GTC挂单, 仓位={pct_used:.1%})")

        client_id = self._new_client_order_id("ENTRY_LIMIT")
        resp = self._place_order({
            "symbol": self.symbol,
            "side": side_str,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": qty_str,
            "price": trigger_str,
            "newClientOrderId": client_id,
        })
        order_id = resp.get("orderId")
        if order_id:
            self._entry_stop_orders.append({
                "order_id": order_id,
                "client_id": client_id,
                "side": side.value,
                "trigger_price": trigger_price,
                "quantity": qty,
                "expire_bar": bar_idx + timeout_bars,
                "start_bar": bar_idx,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "template_fingerprint": template_fingerprint,
                "entry_similarity": entry_similarity,
                "entry_reason": entry_reason,
            })
            # 记录最近一次入场的TP/SL，供交易所同步建仓时回填
            self._last_entry_tp = take_profit
            self._last_entry_sl = stop_loss
            self._last_entry_side = side
            self._last_entry_price = trigger_price
            self._last_entry_ts = time.time()
        return order_id

    def get_pending_entry_orders_snapshot(self, current_bar_idx: int = None) -> List[dict]:
        """返回所有挂单快照（入场单 + 保护单，用于UI展示）"""
        snapshots: List[dict] = []
        
        # ── 入场挂单 ──
        for o in self._entry_stop_orders:
            expire_bar = int(o.get("expire_bar", -1))
            remaining_bars = None
            if current_bar_idx is not None and expire_bar >= 0:
                remaining_bars = max(0, expire_bar - int(current_bar_idx))
            snapshots.append({
                "order_id": o.get("order_id"),
                "client_id": o.get("client_id", ""),
                "side": o.get("side", "-"),
                "trigger_price": float(o.get("trigger_price", 0.0) or 0.0),
                "quantity": float(o.get("quantity", 0.0) or 0.0),
                "start_bar": int(o.get("start_bar", -1)),
                "expire_bar": expire_bar,
                "remaining_bars": remaining_bars,
                "template_fingerprint": o.get("template_fingerprint") or "-",
                "entry_similarity": float(o.get("entry_similarity", 0.0) or 0.0),
                "status": "入场挂单",
            })
        
        # ── 止损保护单 ──
        if self._exchange_sl_order_id and self._exchange_sl_order_id > 0:
            pos = self.current_position
            exit_side = "BUY" if (pos and pos.side == OrderSide.SHORT) else "SELL"
            snapshots.append({
                "order_id": self._exchange_sl_order_id,
                "client_id": f"R3000_SL",
                "side": exit_side,
                "trigger_price": self._exchange_sl_price,
                "quantity": pos.quantity if pos else 0.0,
                "start_bar": -1,
                "expire_bar": -1,
                "remaining_bars": None,
                "template_fingerprint": "止损保护",
                "entry_similarity": 0.0,
                "status": "🛡️止损",
            })
        
        # ── 止盈保护单 ──
        if self._exchange_tp_order_id and self._exchange_tp_order_id > 0:
            pos = self.current_position
            exit_side = "BUY" if (pos and pos.side == OrderSide.SHORT) else "SELL"
            snapshots.append({
                "order_id": self._exchange_tp_order_id,
                "client_id": f"R3000_TP",
                "side": exit_side,
                "trigger_price": self._exchange_tp_price,
                "quantity": pos.quantity if pos else 0.0,
                "start_bar": -1,
                "expire_bar": -1,
                "remaining_bars": None,
                "template_fingerprint": "止盈保护",
                "entry_similarity": 0.0,
                "status": "🎯止盈",
            })
        
        return snapshots

    def open_position(self,
                      side: OrderSide,
                      price: float,
                      bar_idx: int,
                      take_profit: Optional[float] = None,
                      stop_loss: Optional[float] = None,
                      template_fingerprint: Optional[str] = None,
                      entry_similarity: float = 0.0,
                      entry_reason: str = "") -> Optional[PaperOrder]:
        self._sync_from_exchange(force=True)
        if self.current_position is not None:
            print("[BinanceTrader] 交易所已有持仓，跳过开仓")
            return None

        self._set_leverage(self.leverage)
        
        # 获取余额和计算数量
        balance = self._get_usdt_balance()
        available = self._get_usdt_available_balance()
        qty = self._calc_entry_quantity(price)
        side_str = "BUY" if side == OrderSide.LONG else "SELL"
        
        # 格式化数量，确保不超过精度限制
        qty_str = f"{qty:.8f}".rstrip('0').rstrip('.')
        if '.' in qty_str:
            # 根据 _qty_step 自动判断精度
            precision = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
            qty_str = f"{qty:.{precision}f}"
        else:
            qty_str = str(int(qty))
        
        # 【调试】显示开仓参数
        print(f"[BinanceTrader] 开仓请求: {side_str} {qty_str} {self.symbol} @ ~${price:.2f}")
        print(
            f"[BinanceTrader] 账户余额: ${balance:.2f} | 可用: ${available:.2f} | "
            f"杠杆: {self.leverage}x | 数量精度: {self._qty_step} | 最小名义: ${self._min_notional:.2f}"
        )

        resp = self._place_order({
            "symbol": self.symbol,
            "side": side_str,
            "type": "MARKET",
            "quantity": qty_str,
            "newClientOrderId": self._new_client_order_id("ENTRY"),
        })

        executed_qty = float(resp.get("executedQty", qty))
        avg_price = float(resp.get("avgPrice", 0.0)) or float(resp.get("price", 0.0)) or price
        margin_used = (executed_qty * avg_price) / max(float(self.leverage), 1.0)
        self.current_bar_idx = bar_idx
        # 真实成交手续费（入场）
        entry_fee = 0.0
        try:
            order_id = int(resp.get("orderId", 0) or 0)
            if order_id > 0:
                entry_trades = self._get_user_trades(order_id=order_id)
                entry_fee = sum(self._to_float(t.get("commission", 0.0)) for t in entry_trades)
        except Exception:
            pass
        
        # 如果API获取失败，使用费率估算（限价单=Maker 0.02%，市价单=Taker 0.05%）
        if entry_fee == 0.0:
            # 检查订单类型
            order_type = resp.get("type", "MARKET")
            if order_type == "LIMIT":
                entry_fee = (executed_qty * avg_price) * 0.0002  # Maker费率
            else:
                entry_fee = (executed_qty * avg_price) * 0.0005  # Taker费率

        order = PaperOrder(
            order_id=str(resp.get("orderId", self._new_client_order_id("ENTRY_LOCAL"))),
            symbol=self.symbol,
            side=side,
            quantity=executed_qty,
            margin_used=margin_used,
            entry_price=avg_price,
            entry_time=datetime.now(),
            entry_bar_idx=bar_idx,
            take_profit=take_profit,
            stop_loss=stop_loss,
            original_stop_loss=stop_loss,
            template_fingerprint=template_fingerprint,
            entry_similarity=entry_similarity,
            entry_reason=entry_reason,
            peak_price=avg_price,  # 初始峰值 = 入场价
            total_fee=entry_fee,
        )
        # 记录最近一次入场的TP/SL，供交易所同步建仓时回填
        self._last_entry_tp = take_profit
        self._last_entry_sl = stop_loss
        self._last_entry_side = side
        self._last_entry_price = avg_price
        self._last_entry_ts = time.time()
        self.current_position = order
        
        # 【核心】开仓后立即在交易所挂止盈止损保护单
        if order.take_profit is not None or order.stop_loss is not None:
            self._place_exchange_tp_sl(order)
        
        if self.on_order_update:
            self.on_order_update(order)
        return order

    def _marketable_limit_price(self, side: OrderSide, desired_price: float) -> float:
        """计算可成交限价（更激进的缓冲提高 IOC 成交率，减少市价降级及 Taker 费）"""
        from config import PAPER_TRADING_CONFIG
        buffer = float(PAPER_TRADING_CONFIG.get("EXIT_IOC_BUFFER_PCT", 0.003))
        buffer = max(0.001, min(0.01, buffer))  # 限制在 0.1%~1% 之间

        mark = self._get_mark_price()
        if side == OrderSide.LONG:
            # 平多 = 卖出，设置更低于现价，提高成交率（sell limit 需 ≤ best bid）
            px = min(desired_price, mark * (1.0 - buffer))
        else:
            # 平空 = 买入，设置更高于现价，提高成交率（buy limit 需 ≥ best ask）
            px = max(desired_price, mark * (1.0 + buffer))
        px = self._round_step(px, self._price_tick)
        return max(self._price_tick, px)

    def _force_market_close(self, order: 'PaperOrder', exit_side: str, close_qty: float) -> dict:
        """限价单失败后，降级为市价单强制平仓"""
        print(f"[BinanceTrader] ⚠ 限价IOC未成交，降级为市价单强制平仓!")
        # 格式化数量
        precision = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
        qty_str = f"{self._round_step(close_qty, self._qty_step):.{precision}f}"

        resp = self._place_order({
            "symbol": self.symbol,
            "side": exit_side,
            "type": "MARKET",
            "reduceOnly": "true",
            "quantity": qty_str,
            "newClientOrderId": self._new_client_order_id("FORCE"),
        })
        return resp

    def close_position(self,
                       price: float,
                       bar_idx: int,
                       reason: CloseReason,
                       quantity: Optional[float] = None) -> Optional[PaperOrder]:
        """关闭持仓"""
        # 【关键】平仓前先取消交易所保护单，避免：
        # 1. 保护单和平仓单同时执行导致双重平仓
        # 2. 保护单在新仓位开后仍然存在，干扰新仓位
        self._cancel_exchange_tp_sl(silent=False)
        
        # 在操作前先强制同步一次，确保本地 current_position 与交易所一致
        self._sync_from_exchange(force=True)
        
        if self.current_position is None:
            print(f"[BinanceTrader] 尝试关闭仓位失败：交易所当前无持仓")
            return None

        order = self.current_position
        original_qty = order.quantity
        close_qty = quantity if quantity is not None else original_qty
        close_qty = min(close_qty, original_qty)
        exit_side = "SELL" if order.side == OrderSide.LONG else "BUY"

        # 第一步：尝试限价 IOC（低滑点）
        limit_price = self._marketable_limit_price(order.side, price)
        
        # 格式化精度
        q_prec = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        
        qty_str = f"{self._round_step(close_qty, self._qty_step):.{q_prec}f}"
        price_str = f"{limit_price:.{p_prec}f}"

        resp = self._place_order({
            "symbol": self.symbol,
            "side": exit_side,
            "type": "LIMIT",
            "timeInForce": "IOC",
            "reduceOnly": "true",
            "quantity": qty_str,
            "price": price_str,
            "newClientOrderId": self._new_client_order_id("EXIT"),
        })

        status = str(resp.get("status", ""))
        filled_qty = float(resp.get("executedQty", 0.0))

        # 第二步：限价失败 → 立即降级为市价单（绝不让仓位悬空！）
        if status not in ("FILLED", "PARTIALLY_FILLED") or filled_qty <= 0:
            print(f"[BinanceTrader] 限价离场未成交(status={status})，启动市价降级...")
            resp = self._force_market_close(order, exit_side, close_qty)
            status = str(resp.get("status", ""))
            filled_qty = float(resp.get("executedQty", 0.0))
            if status not in ("FILLED", "PARTIALLY_FILLED") or filled_qty <= 0:
                # 市价也失败 —— 标记为待重试
                print(f"[BinanceTrader] ❌ 市价强平也失败: status={status}")
                self._pending_close = (price, bar_idx, reason)
                return None

        exit_price = float(resp.get("avgPrice", 0.0)) or limit_price
        closed_qty = min(filled_qty, close_qty)
        if closed_qty <= 0:
            self._pending_close = (price, bar_idx, reason)
            return None

        # ===== 使用真实撮合成交计算盈亏/手续费 =====
        order_id = int(resp.get("orderId", 0) or 0)
        entry_time_ms = int(order.entry_time.timestamp() * 1000) - 1000
        entry_side = "BUY" if order.side == OrderSide.LONG else "SELL"
        trades = []
        if order_id > 0:
            trades = self._get_user_trades(order_id=order_id, start_time_ms=entry_time_ms)
        if not trades:
            trades = self._get_user_trades(start_time_ms=entry_time_ms)
        agg = self._aggregate_trades(trades, entry_side=entry_side)
        exit_price = agg["exit_price"] or exit_price
        exit_fee = agg["exit_fee"]
        entry_fee = order.total_fee or agg["entry_fee"]
        realized_pnl = agg["realized_pnl"]
        
        # 如果API获取失败，使用费率估算手续费
        if exit_fee == 0.0:
            # 先尝试限价IOC（Maker费率），如果限价单失败就是市价单（Taker费率）
            order_type = resp.get("type", "LIMIT")
            if order_type == "LIMIT":
                exit_fee = (closed_qty * exit_price) * 0.0002  # Maker费率
            else:
                exit_fee = (closed_qty * exit_price) * 0.0005  # Taker费率
        
        if entry_fee == 0.0:
            # 入场大概率是限价单成交（系统设计）
            entry_fee = (order.quantity * order.entry_price) * 0.0002
        
        # 交易所 realizedPnl 通常不含手续费，按净值计算
        net_pnl = realized_pnl - exit_fee - entry_fee
        margin_portion = order.margin_used * (closed_qty / max(original_qty, 1e-12))
        pnl_pct = (net_pnl / max(margin_portion, 1e-9)) * 100.0
        exit_time = datetime.fromtimestamp(agg["last_time_ms"] / 1000) if agg["last_time_ms"] > 0 else datetime.now()
        total_fee = entry_fee + exit_fee

        closed_order = replace(
            order,
            quantity=closed_qty,
            margin_used=margin_portion,
            status=OrderStatus.CLOSED,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_bar_idx=bar_idx,
            close_reason=reason,
            realized_pnl=net_pnl,
            unrealized_pnl=0.0,
            profit_pct=pnl_pct,
            hold_bars=max(0, bar_idx - order.entry_bar_idx),
            total_fee=total_fee,
        )

        self.order_history.append(closed_order)
        self._pending_close = None  # 清除重试标记

        full_close = closed_qty >= (original_qty - 1e-12)
        if full_close:
            self.current_position = None
        else:
            remaining_qty = original_qty - closed_qty
            order.quantity = remaining_qty
            order.margin_used = order.margin_used - margin_portion
            # 更新未实现盈亏（使用当前价格近似）
            mark_price = price
            pnl_unreal = (mark_price - order.entry_price) * remaining_qty if order.side == OrderSide.LONG else (order.entry_price - mark_price) * remaining_qty
            order.unrealized_pnl = pnl_unreal
            order.profit_pct = (pnl_unreal / max(order.margin_used, 1e-9)) * 100.0

        self._update_stats_from_exchange()
        if closed_order.template_fingerprint:
            self._record_template_performance(closed_order)
        
        # 【持久化】平仓后自动保存
        self.save_history(self.history_file)
        
        if full_close:
            if self.on_trade_closed:
                self.on_trade_closed(closed_order)
        else:
            if self.on_order_update:
                self.on_order_update(order)
        return closed_order

    def update_price(self, price: float, high: float = None, low: float = None,
                     bar_idx: int = None, protection_mode: bool = False) -> Optional[CloseReason]:
        if bar_idx is not None:
            self.current_bar_idx = bar_idx

        # ── 重试未成交的平仓（每次 tick 都检查）──
        if self._pending_close is not None and self.current_position is not None:
            p_price, p_bar, p_reason = self._pending_close
            print(f"[BinanceTrader] 🔄 重试挂起的平仓: reason={p_reason.value}")
            closed = self.close_position(price, bar_idx or self.current_bar_idx, p_reason)
            if closed:
                return p_reason
            # 仍然失败，继续等下一次tick重试

        if self.current_position is None:
            return None

        order = self.current_position
        if bar_idx is not None:
            order.hold_bars = max(0, bar_idx - order.entry_bar_idx)

        high = high if high is not None else price
        low = low if low is not None else price
        pnl = (price - order.entry_price) * order.quantity if order.side == OrderSide.LONG else (order.entry_price - price) * order.quantity
        order.unrealized_pnl = pnl
        order.profit_pct = (pnl / max(order.margin_used, 1e-9)) * 100.0

        # 追踪峰值利润（用于锁利逻辑）
        if order.side == OrderSide.LONG:
            if high > order.peak_price:
                order.peak_price = high
        else:
            if order.peak_price == 0 or low < order.peak_price:
                order.peak_price = low
        if order.profit_pct > order.peak_profit_pct:
            order.peak_profit_pct = order.profit_pct

        if order.take_profit is not None:
            if order.side == OrderSide.LONG and high >= order.take_profit:
                closed = self.close_position(order.take_profit, bar_idx or self.current_bar_idx, CloseReason.TAKE_PROFIT)
                return CloseReason.TAKE_PROFIT if closed else None
            if order.side == OrderSide.SHORT and low <= order.take_profit:
                closed = self.close_position(order.take_profit, bar_idx or self.current_bar_idx, CloseReason.TAKE_PROFIT)
                return CloseReason.TAKE_PROFIT if closed else None

        if order.stop_loss is not None:
            # 如果止损价已进入盈利区（如追踪止盈），则按“止盈”记录
            is_profit_sl = (
                (order.side == OrderSide.LONG and order.stop_loss >= order.entry_price) or
                (order.side == OrderSide.SHORT and order.stop_loss <= order.entry_price)
            )
            sl_reason = CloseReason.TRAILING_STOP if is_profit_sl else CloseReason.STOP_LOSS
            
            # 保护期内：只允许盈利止损，阻止亏损止损
            if protection_mode and not is_profit_sl:
                # 保护期内止损暂缓，不触发
                pass
            else:
                if order.side == OrderSide.LONG and low <= order.stop_loss:
                    closed = self.close_position(order.stop_loss, bar_idx or self.current_bar_idx, sl_reason)
                    return sl_reason if closed else None
                if order.side == OrderSide.SHORT and high >= order.stop_loss:
                    closed = self.close_position(order.stop_loss, bar_idx or self.current_bar_idx, sl_reason)
                    return sl_reason if closed else None

        # ═══════════════════════════════════════════════════════════════
        # 【核心】检测本地 TP/SL 变化，自动同步到交易所保护单
        # 追踪止损、保护期收紧等操作会修改 order.stop_loss，
        # 这里检测到变化后立即更新交易所订单，保持一致。
        # ═══════════════════════════════════════════════════════════════
        if order.stop_loss is not None and order.stop_loss > 0:
            if abs(order.stop_loss - self._exchange_sl_price) >= self._price_tick * 0.5:
                self._update_exchange_sl(order.stop_loss)
        
        if order.take_profit is not None and order.take_profit > 0:
            if abs(order.take_profit - self._exchange_tp_price) >= self._price_tick * 0.5:
                self._update_exchange_tp(order.take_profit)
        
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
        三级追踪状态：
          similarity >= safe_threshold  (0.7) → 安全（恢复原始止损）
          similarity >= alert_threshold (0.5) → 警戒（止损移至成本价）
          similarity >= derail_threshold(0.3) → 危险（加紧止损但不立刻平仓）
          similarity <  derail_threshold(0.3) → 脱轨（立刻强制平仓）
        """
        if self.current_position is None:
            return None
        order = self.current_position
        order.current_similarity = similarity

        if similarity >= safe_threshold:
            # 安全区：不回退止损，追踪止损可能已经更好
            order.tracking_status = "安全"
            order.alert_mode = False
        elif similarity >= alert_threshold:
            # 警戒区：收紧到成本价（但不回退已上移的SL）
            order.tracking_status = "警戒"
            order.alert_mode = True
            if order.side == OrderSide.LONG:
                order.stop_loss = max(order.stop_loss or 0, order.entry_price)
            else:
                order.stop_loss = min(order.stop_loss or float('inf'), order.entry_price)
        elif similarity >= derail_threshold:
            # 危险区：收紧到成本价+微利
            order.tracking_status = "危险"
            order.alert_mode = True
            if order.side == OrderSide.LONG:
                danger_sl = order.entry_price * 1.001
                order.stop_loss = max(order.stop_loss or 0, danger_sl)
            else:
                danger_sl = order.entry_price * 0.999
                order.stop_loss = min(order.stop_loss or float('inf'), danger_sl)
        else:
            # 脱轨：立即强制平仓
            order.tracking_status = "脱轨"
            if current_price is not None:
                closed = self.close_position(current_price, bar_idx or self.current_bar_idx, CloseReason.DERAIL)
                return CloseReason.DERAIL if closed else None
        return None

    def _update_stats_from_exchange(self):
        bal = self._get_usdt_balance()
        available = self._get_usdt_available_balance()
        self.stats.current_balance = bal
        self.stats.available_margin = available
        self.stats.total_pnl = bal - self.stats.initial_balance
        if self.stats.initial_balance > 0:
            self.stats.total_pnl_pct = (bal / self.stats.initial_balance - 1.0) * 100.0
        self.stats.total_trades = len(self.order_history)
        wins = sum(1 for t in self.order_history if t.profit_pct > 0)
        losses = max(0, len(self.order_history) - wins)
        self.stats.win_trades = wins
        self.stats.loss_trades = losses

        self.stats.long_trades = sum(1 for t in self.order_history if t.side == OrderSide.LONG)
        self.stats.short_trades = sum(1 for t in self.order_history if t.side == OrderSide.SHORT)
        self.stats.long_wins = sum(1 for t in self.order_history if t.side == OrderSide.LONG and t.profit_pct > 0)
        self.stats.short_wins = sum(1 for t in self.order_history if t.side == OrderSide.SHORT and t.profit_pct > 0)

        if bal > self.stats.max_balance:
            self.stats.max_balance = bal
        dd = self.stats.max_balance - bal
        if dd > self.stats.max_drawdown:
            self.stats.max_drawdown = dd
            if self.stats.max_balance > 1e-9:
                self.stats.max_drawdown_pct = dd / self.stats.max_balance * 100.0

    def _record_template_performance(self, order: PaperOrder):
        fp = order.template_fingerprint
        if not fp:
            return
        if fp not in self.template_performances:
            self.template_performances[fp] = TemplateSimPerformance(fingerprint=fp)
        self.template_performances[fp].add_trade(order.profit_pct)

    def get_profitable_templates(self, min_matches: int = 1) -> List[str]:
        out = []
        for fp, perf in self.template_performances.items():
            if perf.match_count >= min_matches and perf.win_rate >= 0.5:
                out.append(fp)
        return out

    def get_losing_templates(self, min_matches: int = 1) -> List[str]:
        out = []
        for fp, perf in self.template_performances.items():
            if perf.match_count >= min_matches and perf.win_rate < 0.5:
                out.append(fp)
        return out

    def reset(self):
        # 不重置交易所账户，只清空本地展示缓存
        self.current_position = None
        self.order_history.clear()
        self.template_performances.clear()
        self.stats = AccountStats(
            initial_balance=self.initial_balance,
            current_balance=self._get_usdt_balance(),
        )
        self._order_counter = 0
        self._sync_from_exchange(force=True)

    def sync_from_exchange(self, force: bool = False):
        """供外部主动触发同步（UI刷新前调用）"""
        self._sync_from_exchange(force=force)

    def save_history(self, filepath: str):
        # 确保路径存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "symbol": self.symbol,
            "save_time": datetime.now().isoformat(),
            "initial_balance": self.stats.initial_balance,
            "leverage": self.leverage,
            "stats": {
                "total_trades": self.stats.total_trades,
                "win_rate": self.stats.win_rate,
                "total_pnl": self.stats.total_pnl,
                "total_pnl_pct": self.stats.total_pnl_pct,
                "max_drawdown_pct": self.stats.max_drawdown_pct,
                "max_balance": self.stats.max_balance,
            },
            "trades": [o.to_dict() for o in self.order_history],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BinanceTrader] 交易记录已保存至: {filepath}")