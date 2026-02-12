"""
R3000 Binance 测试网执行器
真实下单到 Binance Futures Testnet：
  - 入场：市价单
  - 离场：限价单（reduceOnly + IOC，确保按限价语义且尽量即时成交）
"""

import hashlib
import hmac
import time
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Callable
from urllib.parse import urlencode

import requests

from core.paper_trader import (
    AccountStats,
    CloseReason,
    OrderSide,
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
        
        # 记录保存路径
        self.history_dir = os.path.join(os.getcwd(), "data")
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

        self._validate_credentials()
        self._load_symbol_filters()
        self._set_leverage(self.leverage)
        self._sync_from_exchange()
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
                trades = data.get("trades", [])
                
                # 转换回 PaperOrder 对象
                loaded_history = []
                for t in trades:
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
                    
                    # 同时更新模板统计
                    if order.template_fingerprint:
                        self._record_template_performance(order)
                
                self.order_history = loaded_history
                print(f"[BinanceTrader] 成功从本地加载 {len(self.order_history)} 条历史交易记录")
                
                # 更新账户统计
                self._update_stats_from_exchange()
                
        except Exception as e:
            print(f"[BinanceTrader] 加载历史记录失败: {e}")

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
            self.current_position = None
            return

        side = OrderSide.LONG if amt > 0 else OrderSide.SHORT
        qty = abs(amt)
        entry = float(pos.get("entryPrice", 0.0))
        mark = float(pos.get("markPrice", entry or 0.0))
        leverage = float(pos.get("leverage", self.leverage))
        margin = abs(entry * qty) / max(leverage, 1.0)
        pnl = (mark - entry) * qty if side == OrderSide.LONG else (entry - mark) * qty
        pnl_pct = (pnl / margin * 100.0) if margin > 1e-9 else 0.0

        self.current_position = PaperOrder(
            order_id="EXCHANGE_SYNC",
            symbol=self.symbol,
            side=side,
            quantity=qty,
            margin_used=margin,
            entry_price=entry,
            entry_time=datetime.now(),
            entry_bar_idx=self.current_bar_idx,
            unrealized_pnl=pnl,
            profit_pct=pnl_pct,
        )

    def has_position(self) -> bool:
        return self.current_position is not None

    def _new_client_order_id(self, prefix: str) -> str:
        self._order_counter += 1
        return f"R3000_{prefix}_{int(time.time())}_{self._order_counter}"

    def _place_order(self, params: dict) -> dict:
        return self._signed_request("POST", "/fapi/v1/order", params)

    def _calc_entry_quantity(self, price: float) -> float:
        # 关键：按可用余额计算，避免 balance 包含被占用资金导致 -2019
        avail = self._get_usdt_available_balance()
        margin = avail * self.position_size_pct
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
        )
        self.current_position = order
        if self.on_order_update:
            self.on_order_update(order)
        return order

    def _marketable_limit_price(self, side: OrderSide, desired_price: float) -> float:
        mark = self._get_mark_price()
        # 使用更大的价格缓冲（0.1%），提高IOC成交概率
        if side == OrderSide.LONG:
            # 平多 = 卖出，设置略低于现价保证可成交
            px = min(desired_price, mark * 0.999)
        else:
            # 平空 = 买入，设置略高于现价保证可成交
            px = max(desired_price, mark * 1.001)
        px = self._round_step(px, self._price_tick)
        return max(self._price_tick, px)

    def _force_market_close(self, order: 'PaperOrder', exit_side: str) -> dict:
        """限价单失败后，降级为市价单强制平仓"""
        print(f"[BinanceTrader] ⚠ 限价IOC未成交，降级为市价单强制平仓!")
        # 格式化数量
        precision = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
        qty_str = f"{self._round_step(order.quantity, self._qty_step):.{precision}f}"

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
                       reason: CloseReason) -> Optional[PaperOrder]:
        """关闭持仓"""
        # 在操作前先强制同步一次，确保本地 current_position 与交易所一致
        self._sync_from_exchange(force=True)
        
        if self.current_position is None:
            print(f"[BinanceTrader] 尝试关闭仓位失败：交易所当前无持仓")
            return None

        order = self.current_position
        exit_side = "SELL" if order.side == OrderSide.LONG else "BUY"

        # 第一步：尝试限价 IOC（低滑点）
        limit_price = self._marketable_limit_price(order.side, price)
        
        # 格式化精度
        q_prec = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        
        qty_str = f"{self._round_step(order.quantity, self._qty_step):.{q_prec}f}"
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
            resp = self._force_market_close(order, exit_side)
            status = str(resp.get("status", ""))
            filled_qty = float(resp.get("executedQty", 0.0))
            if status not in ("FILLED", "PARTIALLY_FILLED") or filled_qty <= 0:
                # 市价也失败 —— 标记为待重试
                print(f"[BinanceTrader] ❌ 市价强平也失败: status={status}")
                self._pending_close = (price, bar_idx, reason)
                return None

        exit_price = float(resp.get("avgPrice", 0.0)) or limit_price
        pnl = (exit_price - order.entry_price) * filled_qty if order.side == OrderSide.LONG else (order.entry_price - exit_price) * filled_qty
        fee = (order.entry_price * order.quantity + exit_price * filled_qty) * self.fee_rate
        net_pnl = pnl - fee
        pnl_pct = (net_pnl / max(order.margin_used, 1e-9)) * 100.0

        order.exit_price = exit_price
        order.exit_time = datetime.now()
        order.exit_bar_idx = bar_idx
        order.close_reason = reason
        order.realized_pnl = net_pnl
        order.unrealized_pnl = 0.0
        order.profit_pct = pnl_pct
        order.status = order.status.__class__.CLOSED
        order.hold_bars = bar_idx - order.entry_bar_idx

        self.order_history.append(order)
        self.current_position = None
        self._pending_close = None  # 清除重试标记

        self._update_stats_from_exchange()
        if order.template_fingerprint:
            self._record_template_performance(order)
        
        # 【持久化】平仓后自动保存
        self.save_history(self.history_file)
        
        if self.on_trade_closed:
            self.on_trade_closed(order)
        return order

    def update_price(self, price: float, high: float = None, low: float = None,
                     bar_idx: int = None) -> Optional[CloseReason]:
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
            order.hold_bars = bar_idx - order.entry_bar_idx

        high = high if high is not None else price
        low = low if low is not None else price
        pnl = (price - order.entry_price) * order.quantity if order.side == OrderSide.LONG else (order.entry_price - price) * order.quantity
        order.unrealized_pnl = pnl
        order.profit_pct = (pnl / max(order.margin_used, 1e-9)) * 100.0

        if order.take_profit is not None:
            if order.side == OrderSide.LONG and high >= order.take_profit:
                closed = self.close_position(order.take_profit, bar_idx or self.current_bar_idx, CloseReason.TAKE_PROFIT)
                return CloseReason.TAKE_PROFIT if closed else None
            if order.side == OrderSide.SHORT and low <= order.take_profit:
                closed = self.close_position(order.take_profit, bar_idx or self.current_bar_idx, CloseReason.TAKE_PROFIT)
                return CloseReason.TAKE_PROFIT if closed else None

        if order.stop_loss is not None:
            if order.side == OrderSide.LONG and low <= order.stop_loss:
                closed = self.close_position(order.stop_loss, bar_idx or self.current_bar_idx, CloseReason.STOP_LOSS)
                return CloseReason.STOP_LOSS if closed else None
            if order.side == OrderSide.SHORT and high >= order.stop_loss:
                closed = self.close_position(order.stop_loss, bar_idx or self.current_bar_idx, CloseReason.STOP_LOSS)
                return CloseReason.STOP_LOSS if closed else None

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
            order.tracking_status = "安全"
            if order.alert_mode:
                order.alert_mode = False
                order.stop_loss = order.original_stop_loss
        elif similarity >= alert_threshold:
            order.tracking_status = "警戒"
            if not order.alert_mode:
                order.alert_mode = True
                order.stop_loss = order.entry_price
        elif similarity >= derail_threshold:
            # 危险区间：收紧止损到成本价，但还不立刻强平
            order.tracking_status = "危险"
            if not order.alert_mode:
                order.alert_mode = True
                order.stop_loss = order.entry_price
        else:
            # 真正脱轨：similarity < derail_threshold → 强制平仓
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
            "leverage": self.leverage,
            "stats": {
                "total_trades": self.stats.total_trades,
                "win_rate": self.stats.win_rate,
                "total_pnl": self.stats.total_pnl,
                "total_pnl_pct": self.stats.total_pnl_pct,
                "max_drawdown_pct": self.stats.max_drawdown_pct,
            },
            "trades": [o.to_dict() for o in self.order_history],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BinanceTrader] 交易记录已保存至: {filepath}")