"""
R3000 Binance 测试网执行器
真实下单到 Binance Futures Testnet：
  - 入场：市价单
  - 离场：限价单（reduceOnly + IOC，确保按限价语义且尽量即时成交）
"""

import hashlib
import hmac
import time
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
        self.current_bar_idx: int = 0
        self._order_counter = 0

        self._qty_step = 0.001
        self._qty_min = 0.001
        self._price_tick = 0.1
        self._last_sync_ts = 0.0
        self._sync_interval_sec = 2.0

        self._validate_credentials()
        self._load_symbol_filters()
        self._set_leverage(self.leverage)
        self._sync_from_exchange()

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

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        n = int(value / step)
        return max(step, n * step)

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
        bal = self._get_usdt_balance()
        margin = bal * self.position_size_pct
        notional = margin * self.leverage
        raw_qty = notional / max(price, 1e-9)
        qty = self._round_step(raw_qty, self._qty_step)
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
        qty = self._calc_entry_quantity(price)
        side_str = "BUY" if side == OrderSide.LONG else "SELL"
        resp = self._place_order({
            "symbol": self.symbol,
            "side": side_str,
            "type": "MARKET",
            "quantity": f"{qty:.8f}",
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
        # 离场必须限价；使用可成交限价 + IOC，提高成交概率
        if side == OrderSide.LONG:
            # 平多 = 卖出，设置略低于现价保证可成交
            px = min(desired_price, mark * 0.9995)
        else:
            # 平空 = 买入，设置略高于现价保证可成交
            px = max(desired_price, mark * 1.0005)
        px = self._round_step(px, self._price_tick)
        return max(self._price_tick, px)

    def close_position(self,
                       price: float,
                       bar_idx: int,
                       reason: CloseReason) -> Optional[PaperOrder]:
        if self.current_position is None:
            return None

        order = self.current_position
        exit_side = "SELL" if order.side == OrderSide.LONG else "BUY"
        limit_price = self._marketable_limit_price(order.side, price)
        resp = self._place_order({
            "symbol": self.symbol,
            "side": exit_side,
            "type": "LIMIT",
            "timeInForce": "IOC",
            "reduceOnly": "true",
            "quantity": f"{order.quantity:.8f}",
            "price": f"{limit_price:.8f}",
            "newClientOrderId": self._new_client_order_id("EXIT"),
        })

        status = str(resp.get("status", ""))
        filled_qty = float(resp.get("executedQty", 0.0))
        if status not in ("FILLED", "PARTIALLY_FILLED") or filled_qty <= 0:
            print(f"[BinanceTrader] 限价离场未成交: status={status}")
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

        self._update_stats_from_exchange()
        if order.template_fingerprint:
            self._record_template_performance(order)
        if self.on_trade_closed:
            self.on_trade_closed(order)
        return order

    def update_price(self, price: float, high: float = None, low: float = None,
                     bar_idx: int = None) -> Optional[CloseReason]:
        if bar_idx is not None:
            self.current_bar_idx = bar_idx
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
        else:
            order.tracking_status = "脱轨"
            if current_price is not None:
                closed = self.close_position(current_price, bar_idx or self.current_bar_idx, CloseReason.DERAIL)
                return CloseReason.DERAIL if closed else None
        return None

    def _update_stats_from_exchange(self):
        bal = self._get_usdt_balance()
        self.stats.current_balance = bal
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
        # 复用原有接口，保持主流程不报错
        import json
        import os
        data = {
            "symbol": self.symbol,
            "stats": {
                "total_trades": self.stats.total_trades,
                "win_rate": self.stats.win_rate,
                "total_pnl": self.stats.total_pnl,
                "total_pnl_pct": self.stats.total_pnl_pct,
                "max_drawdown_pct": self.stats.max_drawdown_pct,
            },
            "trades": [o.to_dict() for o in self.order_history],
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
