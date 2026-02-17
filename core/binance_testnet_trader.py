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
from typing import Optional, Dict, List, Callable, Any
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
        #  【三档阶梯式止盈止损委托单系统】
        #  支持两种模式：一次性挂6单 或 逐档挂单（第1档触发→第2档→第3档，可跳过第2档）
        # ═══════════════════════════════════════════════════════════════
        self._staged_orders: List[Dict[str, Any]] = []     # 当前挂出的分段委托单
        self._staged_config: Optional[Dict[str, Any]] = None  # 逐档模式下预计算的全部档位配置
        self._stage3_close_recorded: bool = False  # 标记第3档是否已按分段记账
        # 每个元素包含: {
        #   "order_id": int,          # 交易所订单ID
        #   "type": str,              # "TP" 或 "SL"
        #   "stage": int,             # 档位 1/2/3
        #   "price": float,           # 委托价格
        #   "quantity": float,        # 委托数量
        #   "filled": bool,           # 是否已成交
        # }
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
        
        # 【关键】解析 Binance API 错误信息（含 HTTP 200 但 body 里 code!=0 的情况）
        data = r.json()
        if r.status_code >= 400:
            try:
                error_code = data.get("code", "?")
                error_msg = data.get("msg", r.text[:200])
                safe_params = {k: v for k, v in params.items() if k != "signature"}
                print(f"[BinanceAPI] ❌ {method} {path} 失败")
                print(f"[BinanceAPI] 错误码: {error_code} | 消息: {error_msg}")
                print(f"[BinanceAPI] 请求参数: {safe_params}")
                raise Exception(f"Binance API {error_code}: {error_msg}")
            except Exception as e:
                if "Binance API" in str(e):
                    raise
                r.raise_for_status()
        # Binance 常返回 HTTP 200 但 body 中 code 非 0（如 -1013 价格/数量过滤失败）
        if isinstance(data, dict) and "code" in data and data["code"] != 0:
            error_code = data.get("code")
            error_msg = data.get("msg", "unknown")
            safe_params = {k: v for k, v in params.items() if k != "signature"}
            print(f"[BinanceAPI] ❌ {method} {path} 业务错误(HTTP 200)")
            print(f"[BinanceAPI] 错误码: {error_code} | 消息: {error_msg}")
            print(f"[BinanceAPI] 请求参数: {safe_params}")
            raise Exception(f"Binance API {error_code}: {error_msg}")
        
        return data

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
    
    def set_leverage(self, leverage: int):
        """运行时修改杠杆"""
        if leverage < 1 or leverage > 125:
            raise ValueError(f"杠杆倍数必须在1-125之间，当前: {leverage}")
        
        old_leverage = self.leverage
        self.leverage = leverage
        
        # 调用交易所API修改杠杆
        try:
            self._signed_request("POST", "/fapi/v1/leverage", {
                "symbol": self.symbol,
                "leverage": int(leverage),
            })
            print(f"[BinanceTestnet] 杠杆已更新: {old_leverage}x -> {leverage}x")
        except Exception as e:
            # 如果API调用失败，回滚本地值
            self.leverage = old_leverage
            raise Exception(f"更新杠杆失败: {e}")

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
        根据平仓价格和订单的TP/SL设置，推断真正的平仓原因；尽量返回止盈/止损/追踪止损以便参与 TP/SL 学习。
        会设置 order.close_reason_detail 供日志展示（如 "挂单触发(交易所成交)"）。
        """
        tolerance_tight = 0.001   # 0.1% 严格容差
        tolerance_loose = 0.005   # 0.5% 宽松容差（交易所滑点时用）
        setattr(order, "close_reason_detail", "")

        def set_detail(msg: str):
            setattr(order, "close_reason_detail", msg)

        # 检查是否触及止盈
        if order.take_profit is not None:
            tp = order.take_profit
            if abs(exit_price - tp) / tp < tolerance_tight:
                set_detail("挂单触发(止盈)")
                return CloseReason.TAKE_PROFIT
            if order.side == OrderSide.LONG and exit_price >= tp:
                set_detail("挂单触发(止盈)")
                return CloseReason.TAKE_PROFIT
            if order.side == OrderSide.SHORT and exit_price <= tp:
                set_detail("挂单触发(止盈)")
                return CloseReason.TAKE_PROFIT

        # 检查是否触及止损（追踪止损已关闭，统一视为止损）
        if order.stop_loss is not None:
            sl = order.stop_loss
            sl_reason = CloseReason.STOP_LOSS
            detail_sl = "挂单触发(止损)"

            if abs(exit_price - sl) / sl < tolerance_tight:
                set_detail(detail_sl)
                return sl_reason
            if order.side == OrderSide.LONG and exit_price <= sl:
                set_detail(detail_sl)
                return sl_reason
            if order.side == OrderSide.SHORT and exit_price >= sl:
                set_detail(detail_sl)
                return sl_reason

        # 宽松容差再判一次（交易所滑点可能导致成交价略偏离设定）
        if order.take_profit is not None:
            tp = order.take_profit
            if abs(exit_price - tp) / tp < tolerance_loose:
                set_detail(f"挂单触发(止盈,按价格推断; 出场{exit_price:.2f} vs TP{tp:.2f})")
                return CloseReason.TAKE_PROFIT
        if order.stop_loss is not None:
            sl = order.stop_loss
            sl_reason = CloseReason.STOP_LOSS
            if abs(exit_price - sl) / sl < tolerance_loose:
                set_detail(f"挂单触发(止损,按价格推断; 出场{exit_price:.2f} vs SL{sl:.2f})")
                return sl_reason

        # 仍无法精确匹配：按“更接近 TP 还是 SL”推断，便于参与 TP/SL 学习；详情中写出出场价与设定价，避免误解
        if order.take_profit is not None and order.stop_loss is not None:
            dist_tp = abs(exit_price - order.take_profit) / order.take_profit
            dist_sl = abs(exit_price - order.stop_loss) / order.stop_loss
            if dist_tp <= dist_sl:
                set_detail(f"挂单触发(止盈,按价格推断; 出场{exit_price:.2f} vs TP{order.take_profit:.2f})")
                return CloseReason.TAKE_PROFIT
            set_detail(f"挂单触发(止损,按价格推断; 出场{exit_price:.2f} vs SL{order.stop_loss:.2f})")
            return CloseReason.STOP_LOSS

        set_detail("交易所平仓(原因不明)")
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
                setattr(order, "close_reason_detail", "挂单触发(交易所STOP单)")
                return CloseReason.STOP_LOSS

            if "TAKE_PROFIT" in order_type:
                print("[平仓诊断] ✓ 确认止盈触发")
                setattr(order, "close_reason_detail", "挂单触发(交易所止盈单)")
                return CloseReason.TAKE_PROFIT

            if "LIQUIDATION" in order_type or "LIQUIDATION" in orig_type:
                print("[平仓诊断] ⚠️ 强制平仓（爆仓）")
                setattr(order, "close_reason_detail", "强制平仓(爆仓)")
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
            # 无仓位时本地阶梯单缓存必须清空，避免下一笔误判“已有保护单”
            self._staged_config = None
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
                self._stage3_close_recorded = False
                close_reason = self._detect_tp_sl_fill(order_for_detail=prev_pos)
                if close_reason:
                    print(f"[BinanceTrader] 📍 仓位由交易所保护单平仓: {close_reason.value}")
                    # 若第3档已按分段记账，跳过整笔重复记录
                    if self._stage3_close_recorded:
                        print(f"[BinanceTrader] 第3档已按分段记录，跳过整笔同步")
                        # 关键：提前返回前必须清空本地阶梯缓存，避免遗留脏状态影响下一笔
                        self._staged_orders.clear()
                        self._staged_config = None
                        return
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
                if not getattr(prev_pos, "close_reason_detail", ""):
                    prev_pos.close_reason_detail = "挂单触发(交易所保护单)" if (close_reason and close_reason != CloseReason.EXCHANGE_CLOSE) else ""
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
            else:
                # 无前置持仓但交易所也无仓，兜底清理本地阶梯缓存
                self._staged_orders.clear()
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
            # 更新行情相关字段，保留 peak_price 等
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
                has_staged = len(self._staged_orders) > 0
                if not has_staged:
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
            # 有持仓时也轮询检测阶梯单是否部分成交（第一/二档），否则价格过了 TP1 仍显示挂单中
            if self._staged_orders:
                self._detect_tp_sl_fill()
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
            entry_kelly_pct = 0.0
            if self._entry_stop_orders:
                last_entry = self._entry_stop_orders[-1]
                entry_tp = last_entry.get("take_profit")
                entry_sl = last_entry.get("stop_loss")
                entry_bar_idx = int(last_entry.get("start_bar", self.current_bar_idx))
                entry_fp = last_entry.get("template_fingerprint")
                entry_sim = float(last_entry.get("entry_similarity", 0.0) or 0.0)
                entry_reason = last_entry.get("entry_reason", "")
                psp = last_entry.get("position_size_pct")
                if psp is not None:
                    entry_kelly_pct = float(psp)
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
                kelly_position_pct=entry_kelly_pct,
            )
            
            # 【核心】新仓位同步后，如果有TP/SL，立即挂交易所保护单
            if (entry_tp is not None or entry_sl is not None):
                self._ensure_exchange_tp_sl_protection(self.current_position, source="sync_new_position")
        
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

        # 已有持仓时，执行一次保护单自检，防止“本地有缓存但交易所无挂单”导致裸仓
        if self.current_position is not None:
            self._ensure_exchange_tp_sl_protection(self.current_position, source="sync_tail_check")

    def _has_active_local_staged_orders(self) -> bool:
        """本地是否存在未成交的阶梯保护单缓存"""
        for so in self._staged_orders:
            if so.get("filled"):
                continue
            if int(so.get("order_id", 0) or 0) > 0:
                return True
        return False

    def _has_active_exchange_staged_orders(self) -> bool:
        """交易所是否存在本系统的阶梯保护单"""
        try:
            open_orders = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol})
            for o in open_orders:
                cid = str(o.get("clientOrderId", "") or "")
                # 兼容旧前缀，避免历史单导致误判
                if any(tag in cid for tag in ("R3000_TP", "R3000_SL", "R3K_TP", "R3K_SL")):
                    return True
            return False
        except Exception as e:
            print(f"[BinanceTrader] ⚠ 检查交易所保护单失败: {e}")
            return False

    def _ensure_exchange_tp_sl_protection(self, order: Optional[PaperOrder], source: str = "") -> None:
        """
        保护单一致性自检+自愈：
        - 本地缓存与交易所状态不一致时给出诊断日志
        - 两侧都没有保护单时自动补挂，避免裸仓
        """
        if order is None:
            return
        if order.take_profit is None and order.stop_loss is None:
            return

        # 先清理本地“幽灵单”
        if self._staged_orders:
            self._verify_staged_orders_on_exchange()

        has_local = self._has_active_local_staged_orders()
        has_exchange = self._has_active_exchange_staged_orders()

        if has_local != has_exchange:
            print(
                f"[BinanceTrader] ⚠ 保护单状态不一致: local={has_local} exchange={has_exchange} | "
                f"source={source or '-'}"
            )

        if not has_local and not has_exchange:
            print(
                f"[BinanceTrader] 🚑 检测到持仓无保护单，自动补挂阶梯止盈止损 | "
                f"source={source or '-'}"
            )
            self._place_exchange_tp_sl(order)

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
        【阶梯基准止盈止损系统】
        
        核心理念：
        - 止盈分三档锁定利润（TP1→TP2→TP3），每档 +7%
        - 止损不分档，始终全平剩余仓位，每档 -5%
        - 阶梯基准：TP1基于入场价，TP2基于TP1成交价，TP3基于TP2成交价
        - 止损跟随：SL1基于入场价，SL2基于TP1成交价，SL3基于TP2成交价
        
        开仓时只挂：
        - TP1: +7%（基于入场价），平仓 50%
        - SL:  -5%（基于入场价），全平 100%
        
        后续挂单由 _place_next_stage_orders 处理
        """
        if order is None:
            return
        
        from config import PAPER_TRADING_CONFIG as _ptc
        
        # 获取配置：统一的止盈/止损收益率
        lev = max(1, int(getattr(self, "leverage", 1)))
        tp_return = _ptc.get("STAGED_TP_PCT", 7.0) / 100   # 每档止盈收益率 7%
        sl_return = _ptc.get("STAGED_SL_PCT", 5.0) / 100   # 止损收益率 5%
        
        # 价格变动 = 收益率 / 杠杆
        tp_pct = tp_return / lev
        sl_pct = sl_return / lev
        
        # 仓位分配
        ratio1 = _ptc.get("STAGED_TP_RATIO_1", 0.50)  # 第1档 50%
        
        entry_price = order.entry_price
        total_qty = order.quantity
        is_long = (order.side == OrderSide.LONG)
        exit_side = "SELL" if is_long else "BUY"
        p_prec = len(str(self._price_tick).split('.')[-1]) if '.' in str(self._price_tick) else 0
        q_prec = len(str(self._qty_step).split('.')[-1]) if '.' in str(self._qty_step) else 0
        
        # 清除旧的保护单
        self._cancel_exchange_tp_sl(silent=True)
        
        # 计算第1档价格（基于入场价）
        if is_long:
            tp1_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp1_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
        
        # 第1档仓位 = 总仓位 × 50%
        qty1 = self._round_step(total_qty * ratio1, self._qty_step)
        # 止损仓位 = 全部剩余（开仓时 = 100%）
        sl_qty = self._round_step(total_qty, self._qty_step)
        
        # 保存阶梯配置（用于后续挂单）
        self._staged_config = {
            "entry_price": entry_price,
            "current_base_price": entry_price,  # 当前阶梯基准价（会随TP成交更新）
            "total_qty": total_qty,
            "is_long": is_long,
            "exit_side": exit_side,
            "p_prec": p_prec,
            "q_prec": q_prec,
            "tp_return": tp_return,
            "sl_return": sl_return,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "current_tier": 1,  # 当前档位
            "leverage": lev,
        }
        
        print(f"[BinanceTrader] 【阶梯基准系统】杠杆={lev}x | 每档TP=+{tp_return*100:.0f}% | SL=-{sl_return*100:.0f}%")
        print(f"[BinanceTrader] 开仓挂单: TP1={tp1_price:.{p_prec}f} (平{ratio1*100:.0f}%) | SL={sl_price:.{p_prec}f} (全平)")
        
        # 挂 TP1（限价单，平仓 50%）
        n = self._place_tiered_tp_sl(
            tp_price=tp1_price,
            tp_qty=qty1,
            sl_price=sl_price,
            sl_qty=sl_qty,
            tier=1,
            tp_pct=tp_return * 100,
            sl_pct=sl_return * 100,
        )
        
        if n == 2:
            print(f"[BinanceTrader] 🎯 第1档就位: TP1(平50%) + SL(全平) | TP成交后将基于成交价挂第2档")
        elif n > 0:
            print(f"[BinanceTrader] ⚠ 第1档部分挂单成功 ({n}/2)")
        else:
            print(f"[BinanceTrader] 🚨 第1档挂单失败！")
    
    def _place_tiered_tp_sl(self, tp_price: float, tp_qty: float, sl_price: float, sl_qty: float,
                            tier: int, tp_pct: float, sl_pct: float) -> int:
        """挂指定档位的 TP + SL，返回成功数"""
        from config import PAPER_TRADING_CONFIG as _ptc
        
        cfg = self._staged_config
        if not cfg:
            return 0
        
        is_long = cfg["is_long"]
        exit_side = cfg["exit_side"]
        p_prec = cfg["p_prec"]
        q_prec = cfg["q_prec"]
        mark_price = self._get_mark_price()
        band_pct = _ptc.get("LIMIT_PRICE_BAND_PCT", 5.0) / 100
        min_sell_limit = mark_price * (1 - band_pct) if mark_price > 0 else 0.0
        max_buy_limit = mark_price * (1 + band_pct) if mark_price > 0 else float("inf")
        
        success_count = 0
        
        # 挂 TP（限价单）
        if tp_qty > 0:
            tp_price_rounded = self._round_step(tp_price, self._price_tick)
            tp_price_str = f"{tp_price_rounded:.{p_prec}f}"
            tp_qty_str = f"{tp_qty:.{q_prec}f}"
            
            # 判断是否需要用 STOP_MARKET（价格偏离过大）
            use_stop_market = False
            if is_long and tp_price_rounded < min_sell_limit and min_sell_limit > 0:
                use_stop_market = True
            elif not is_long and tp_price_rounded > max_buy_limit:
                use_stop_market = True
            
            try:
                if use_stop_market:
                    params = {
                        "symbol": self.symbol, "side": exit_side, "type": "STOP_MARKET",
                        "stopPrice": tp_price_str, "quantity": tp_qty_str, "reduceOnly": "true",
                        "workingType": "CONTRACT_PRICE", "newClientOrderId": self._new_client_order_id(f"TP{tier}"),
                    }
                else:
                    params = {
                        "symbol": self.symbol, "side": exit_side, "type": "LIMIT",
                        "price": tp_price_str, "quantity": tp_qty_str, "reduceOnly": "true",
                        "timeInForce": "GTC", "newClientOrderId": self._new_client_order_id(f"TP{tier}"),
                    }
                
                print(f"[BinanceTrader] 📤 TP{tier} 下单请求: {params}")
                resp = self._place_order(params)
                oid = int(resp.get("orderId", 0) or 0)
                status = str(resp.get("status", "")).upper()
                print(f"[BinanceTrader] 📥 TP{tier} 响应: orderId={oid} status={status}")
                
                if oid > 0:
                    self._staged_orders.append({
                        "order_id": oid, "type": "TP", "stage": tier, "price": tp_price_rounded,
                        "quantity": tp_qty, "pct": tp_pct, "filled": (status == "FILLED")
                    })
                    success_count += 1
                    print(f"[BinanceTrader] ✅ 止盈第{tier}档: {exit_side} @ {tp_price_str} | 数量={tp_qty_str} (+{tp_pct:.0f}%)")
            except Exception as e:
                print(f"[BinanceTrader] ❌ 止盈第{tier}档挂单失败: {e}")
        
        # 挂 SL（STOP_MARKET，全平剩余）
        if sl_qty > 0:
            sl_price_rounded = self._round_step(sl_price, self._price_tick)
            sl_price_str = f"{sl_price_rounded:.{p_prec}f}"
            sl_qty_str = f"{sl_qty:.{q_prec}f}"
            
            try:
                params = {
                    "symbol": self.symbol, "side": exit_side, "type": "STOP_MARKET",
                    "stopPrice": sl_price_str, "quantity": sl_qty_str, "reduceOnly": "true",
                    "workingType": "CONTRACT_PRICE", "newClientOrderId": self._new_client_order_id(f"SL{tier}"),
                }
                
                print(f"[BinanceTrader] 📤 SL{tier} 下单请求: {params}")
                resp = self._place_order(params)
                oid = int(resp.get("orderId", 0) or 0)
                status = str(resp.get("status", "")).upper()
                print(f"[BinanceTrader] 📥 SL{tier} 响应: orderId={oid} status={status}")
                
                if oid > 0:
                    self._staged_orders.append({
                        "order_id": oid, "type": "SL", "stage": tier, "price": sl_price_rounded,
                        "quantity": sl_qty, "pct": sl_pct, "filled": (status == "FILLED"),
                        "is_full_close": True  # 标记为全平止损
                    })
                    success_count += 1
                    print(f"[BinanceTrader] ✅ 止损第{tier}档: {exit_side} STOP @ {sl_price_str} | 数量={sl_qty_str} (-{sl_pct:.0f}%) [全平]")
            except Exception as e:
                print(f"[BinanceTrader] ❌ 止损第{tier}档挂单失败: {e}")
        
        if success_count > 0:
            self._verify_staged_orders_on_exchange()
        
        return success_count

    def _place_stage_orders(self, cfg: dict, tiers: List[int]) -> int:
        """挂指定档位的委托单，返回成功数"""
        from config import PAPER_TRADING_CONFIG as _ptc
        
        is_long = cfg["is_long"]
        exit_side = cfg["exit_side"]
        p_prec = cfg["p_prec"]
        q_prec = cfg["q_prec"]
        mark_price = self._get_mark_price()
        band_pct = _ptc.get("LIMIT_PRICE_BAND_PCT", 5.0) / 100
        min_sell_limit = mark_price * (1 - band_pct) if mark_price > 0 else 0.0
        max_buy_limit = mark_price * (1 + band_pct) if mark_price > 0 else float("inf")
        
        stages = []
        for t in tiers:
            stages.append({"type": "TP", "stage": t, "price": cfg[f"tp{t}_price"], "quantity": cfg[f"qty{t}"], "pct": cfg[f"tp{t}_pct"]})
            stages.append({"type": "SL", "stage": t, "price": cfg[f"sl{t}_price"], "quantity": cfg[f"qty{t}"], "pct": cfg[f"sl{t}_pct"]})
        
        success_count = 0
        for stage_info in stages:
            stype, snum = stage_info["type"], stage_info["stage"]
            price = self._round_step(stage_info["price"], self._price_tick)
            qty = stage_info["quantity"]
            pct = stage_info["pct"]
            if qty <= 0:
                continue
            price_str = f"{price:.{p_prec}f}"
            qty_str = f"{qty:.{q_prec}f}"
            
            use_stop_market = False
            if stype == "SL":
                # 止损单始终用 STOP_MARKET，避免 LIMIT 直接吃单导致“看不到止损”
                use_stop_market = True
            elif stype == "TP":
                # 止盈单优先 LIMIT，若价格偏离过大则退化为 STOP_MARKET
                if is_long and price < min_sell_limit and min_sell_limit > 0:
                    use_stop_market = True
                elif not is_long and price > max_buy_limit:
                    use_stop_market = True
            
            try:
                if use_stop_market:
                    params = {
                        "symbol": self.symbol, "side": exit_side, "type": "STOP_MARKET",
                        "stopPrice": price_str, "quantity": qty_str, "reduceOnly": "true",
                        "workingType": "CONTRACT_PRICE", "newClientOrderId": self._new_client_order_id(f"{stype}{snum}"),
                    }
                    resp = self._place_order(params)
                    ot = "STOP_MARKET"
                else:
                    params = {
                        "symbol": self.symbol, "side": exit_side, "type": "LIMIT",
                        "price": price_str, "quantity": qty_str, "reduceOnly": "true",
                        "timeInForce": "GTC", "newClientOrderId": self._new_client_order_id(f"{stype}{snum}"),
                    }
                    resp = self._place_order(params)
                    ot = "LIMIT"
                oid = int(resp.get("orderId", 0) or 0)
                status = str(resp.get("status", "")).upper()
                lbl = "止盈" if stype == "TP" else "止损"
                print(f"[BinanceTrader] 📤 下单请求: {params}")
                print(f"[BinanceTrader] 📥 下单响应: orderId={oid} status={status} code={resp.get('code','')} msg={resp.get('msg','')}")
                if oid <= 0:
                    print(f"[BinanceTrader] ⚠ {lbl}第{snum}档挂单无效(无 orderId)，已忽略")
                    continue
                filled = (status == "FILLED")
                self._staged_orders.append({
                    "order_id": oid, "type": stype, "stage": snum, "price": price,
                    "quantity": qty, "pct": pct, "filled": filled
                })
                success_count += 1
                extra = " | 已成交" if filled else ""
                print(f"[BinanceTrader] ✅ {lbl}第{snum}档: {exit_side} {ot} @ {price_str} | 数量={qty_str} ({pct:.0f}%) | orderId={oid}{extra}")
            except Exception as e:
                lbl = "止盈" if stype == "TP" else "止损"
                print(f"[BinanceTrader] ❌ {lbl}第{snum}档挂单失败: {e}")
        if success_count > 0:
            self._verify_staged_orders_on_exchange()
        return success_count

    def _verify_staged_orders_on_exchange(self) -> None:
        """校验阶梯单是否真实存在于交易所，若不存在则从本地移除，避免显示「假委托」"""
        if not self._staged_orders:
            return
        try:
            open_orders = self._signed_request("GET", "/fapi/v1/openOrders", {"symbol": self.symbol})
            exchange_ids = {int(o["orderId"]) for o in open_orders}
        except Exception as e:
            print(f"[BinanceTrader] ⚠ 校验委托单失败(无法拉取 openOrders): {e}")
            return
        to_remove = []
        for so in self._staged_orders:
            if so.get("filled"):
                continue
            oid = so.get("order_id") or 0
            if oid in exchange_ids:
                continue
            try:
                info = self._signed_request("GET", "/fapi/v1/order", {"symbol": self.symbol, "orderId": oid})
                status = str(info.get("status", "")).upper()
                if status == "FILLED":
                    so["filled"] = True
                    lbl = "止盈" if so.get("type") == "TP" else "止损"
                    print(f"[BinanceTrader] 📍 校验发现{lbl}第{so.get('stage')}档已成交: orderId={oid}")
                    continue
            except Exception:
                pass
            to_remove.append(so)
            lbl = "止盈" if so.get("type") == "TP" else "止损"
            print(f"[BinanceTrader] ⚠ 委托单在交易所不存在(已从本地移除): {lbl}第{so.get('stage')}档 orderId={oid} | 请确认使用【币安合约测试网】并核对 API")
        for so in to_remove:
            self._staged_orders.remove(so)
        if to_remove:
            print(f"[BinanceTrader] 提示: 本程序使用 testnet.binancefuture.com，请在测试网网页/APP 查看委托单，勿看主网。")

    def _cancel_other_stage_order(self, filled_type: str, filled_stage: int) -> None:
        """第一档或第二档一方成交后，取消同档另一侧未成交单"""
        other = None
        for so in self._staged_orders:
            if so.get("filled"):
                continue
            if so.get("type") != filled_type and so.get("stage") == filled_stage:
                other = so
                break
        if other:
            oid = other.get("order_id")
            if oid and oid > 0:
                try:
                    self._signed_request("DELETE", "/fapi/v1/order", {"symbol": self.symbol, "orderId": oid})
                    lbl = "止盈" if other.get("type") == "TP" else "止损"
                    print(f"[BinanceTrader] 🔄 已取消同档{lbl}单 orderId={oid}")
                except Exception:
                    pass
            self._staged_orders.remove(other)


    def _place_next_stage_orders(self, from_tier: int, filled_type: str, tier_fill_price: Optional[float] = None) -> None:
        """
        【阶梯基准系统】TP成交后挂下一档
        
        核心逻辑：
        - TP成交：基于TP成交价计算新的 TP + SL，挂下一档
        - SL成交：全平剩余，交易结束，无需挂单
        
        阶梯基准：
        - TP1成交后：TP2 = TP1成交价 + 7%，SL2 = TP1成交价 - 5%
        - TP2成交后：TP3 = TP2成交价 + 7%，SL3 = TP2成交价 - 5%
        """
        cfg = self._staged_config
        if not cfg or not self.current_position:
            return
        
        # 取消同档的另一侧单（TP成交取消SL，SL成交取消TP）
        self._cancel_other_stage_order(filled_type, from_tier)
        
        # 如果是 SL 成交，全平剩余，交易结束
        if filled_type == "SL":
            print(f"[BinanceTrader] 🛑 止损第{from_tier}档成交，全平剩余仓位，交易结束")
            self._staged_config = None
            self._staged_orders.clear()
            return
        
        # TP 成交，准备挂下一档
        pos = self._get_position()
        amt = float(pos.get("positionAmt", 0.0)) if pos else 0.0
        if abs(amt) < 1e-12:
            print(f"[BinanceTrader] ⚠ TP{from_tier}成交后仓位为0，无需挂下一档")
            self._staged_config = None
            return
        
        # 下一档
        next_tier = from_tier + 1
        if next_tier > 3:
            print(f"[BinanceTrader] ✅ 所有3档止盈已完成")
            self._staged_config = None
            return
        
        # 获取配置参数
        is_long = cfg["is_long"]
        tp_pct = cfg["tp_pct"]
        sl_pct = cfg["sl_pct"]
        tp_return = cfg["tp_return"]
        sl_return = cfg["sl_return"]
        p_prec = cfg["p_prec"]
        q_prec = cfg["q_prec"]
        
        # 新的阶梯基准价 = 上一档 TP 成交价
        if tier_fill_price and tier_fill_price > 0:
            new_base_price = tier_fill_price
        else:
            # 如果没有成交价，用当前 mark price 估算
            new_base_price = self._get_mark_price()
        
        # 更新阶梯基准
        cfg["current_base_price"] = new_base_price
        cfg["current_tier"] = next_tier
        
        # 计算新的 TP 和 SL 价格（基于新基准）
        if is_long:
            next_tp_price = new_base_price * (1 + tp_pct)
            next_sl_price = new_base_price * (1 - sl_pct)
        else:
            next_tp_price = new_base_price * (1 - tp_pct)
            next_sl_price = new_base_price * (1 + sl_pct)
        
        # 计算仓位
        from config import PAPER_TRADING_CONFIG as _ptc
        remaining_qty = self._round_step(abs(amt), self._qty_step)
        
        if next_tier == 2:
            ratio2 = _ptc.get("STAGED_TP_RATIO_2", 0.50)  # 剩余的 50%
            tp_qty = self._round_step(remaining_qty * ratio2, self._qty_step)
        else:  # tier 3
            tp_qty = remaining_qty  # 全平剩余
        
        sl_qty = remaining_qty  # 止损始终全平剩余
        
        print(f"[BinanceTrader] 📊 第{next_tier}档阶梯基准: TP{from_tier}成交价={new_base_price:.{p_prec}f}")
        print(f"[BinanceTrader] 📊 计算: TP{next_tier}={next_tp_price:.{p_prec}f} (+{tp_return*100:.0f}%) | "
              f"SL{next_tier}={next_sl_price:.{p_prec}f} (-{sl_return*100:.0f}%)")
        print(f"[BinanceTrader] 📊 仓位: TP{next_tier}平{tp_qty:.{q_prec}f} | SL{next_tier}全平{sl_qty:.{q_prec}f}")
        
        # 挂新的 TP + SL
        n = self._place_tiered_tp_sl(
            tp_price=next_tp_price,
            tp_qty=tp_qty,
            sl_price=next_sl_price,
            sl_qty=sl_qty,
            tier=next_tier,
            tp_pct=tp_return * 100,
            sl_pct=sl_return * 100,
        )
        
        if n == 2:
            print(f"[BinanceTrader] 🎯 第{next_tier}档就位: TP(平{'全部' if next_tier == 3 else '50%'}) + SL(全平)")
        elif n > 0:
            print(f"[BinanceTrader] ⚠ 第{next_tier}档部分挂单成功 ({n}/2)")
        else:
            print(f"[BinanceTrader] 🚨 第{next_tier}档挂单失败！")

    def _cancel_exchange_tp_sl(self, silent: bool = False) -> None:
        """取消交易所上的所有阶梯式止盈止损委托单"""
        if not self._staged_orders:
            return
        
        for stage_order in self._staged_orders:
            order_id = stage_order.get("order_id")
            if not order_id or order_id <= 0:
                continue
            
            stage_type = stage_order.get("type", "")
            stage_num = stage_order.get("stage", 0)
            label = f"{'止盈' if stage_type == 'TP' else '止损'}第{stage_num}档"
            
            if stage_order.get("filled", False):
                # 已成交的订单不需要取消
                continue
            
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
        
        # 清空委托单列表
        self._staged_orders.clear()

    def _update_exchange_sl(self, new_sl: float, force: bool = False) -> bool:
        """
        【已禁用】阶梯式委托单系统不支持动态更新止损
        
        阶梯式系统在开仓时一次性挂好所有委托单，不支持后续修改。
        如需调整止损，需要取消所有委托单后重新挂单。
        """
        return False  # 禁用动态止损更新

        now = time.time()
        if not force:
            # 节流：避免频繁更新API
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
        """
        【已禁用】阶梯式委托单系统不支持动态更新止盈
        
        阶梯式系统在开仓时一次性挂好所有委托单，不支持后续修改。
        """
        return False  # 禁用动态止盈更新
        
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

    def _detect_tp_sl_fill(self, order_for_detail: Optional[PaperOrder] = None) -> Optional[CloseReason]:
        """
        检测交易所阶梯式止盈/止损委托单是否已成交
        
        Returns:
            CloseReason if a staged order was filled, None otherwise
        """
        if not self._staged_orders:
            return None
        target_order = order_for_detail or self.current_position
        
        for stage_order in self._staged_orders:
            if stage_order.get("filled", False):
                # 已处理过的成交，跳过
                continue
            
            order_id = stage_order.get("order_id")
            if not order_id or order_id <= 0:
                continue
            
            stage_type = stage_order.get("type", "")
            stage_num = stage_order.get("stage", 0)
            pct = stage_order.get("pct", 0)
            
            try:
                info = self._signed_request("GET", "/fapi/v1/order", {
                    "symbol": self.symbol,
                    "orderId": order_id,
                })
                status = str(info.get("status", ""))
                
                if status == "FILLED":
                    stage_order["filled"] = True
                    label = "止盈" if stage_type == "TP" else "止损"
                    detail_msg = f"挂单触发({label}第{stage_num}档)"
                    print(f"[BinanceTrader] 📍 交易所{label}第{stage_num}档已成交: orderId={order_id} | 档位={pct:.0f}%")
                    filled_price = float(info.get("avgPrice", 0.0) or 0.0) or float(stage_order.get("price", 0.0) or 0.0)
                    
                    if target_order:
                        if stage_type == "TP":
                            target_order.partial_tp_count = getattr(target_order, 'partial_tp_count', 0) + 1
                        else:
                            target_order.partial_sl_count = getattr(target_order, 'partial_sl_count', 0) + 1
                        target_order.close_reason_detail = detail_msg
                    
                    from config import PAPER_TRADING_CONFIG as _ptc
                    # 阶梯基准系统：TP成交后挂下一档（传递成交价），SL成交后全平结束
                    if stage_num == 3:
                        # 第3档成交，交易完全结束
                        self._cancel_other_stage_order(stage_type, 3)
                        self._staged_config = None
                    elif stage_type == "SL":
                        # 止损成交：全平剩余仓位，交易结束（由 _place_next_stage_orders 处理）
                        self._place_next_stage_orders(
                            from_tier=stage_num,
                            filled_type=stage_type,
                            tier_fill_price=filled_price
                        )
                    elif _ptc.get("STAGED_ORDERS_SEQUENTIAL", False):
                        # 止盈成交：挂下一档（传递TP成交价作为新基准）
                        self._place_next_stage_orders(
                            from_tier=stage_num,
                            filled_type=stage_type,
                            tier_fill_price=filled_price  # 所有TP档都传递成交价
                        )
                    
                    # 分段成交（含第3档）写入交易记录，统计按「当前档」口径
                    if stage_num in (1, 2, 3) and target_order is not None:
                        try:
                            closed_qty = float(stage_order.get("quantity", 0.0))
                            # 第3档时，closed_qty 就是全部剩余，从交易所同步确保精准
                            if stage_num == 3:
                                pos_final = self._get_position()
                                qty_before_tier3 = abs(float(pos_final.get("positionAmt", 0.0))) if pos_final else 0.0
                                closed_qty = max(closed_qty, qty_before_tier3)
                            # 本档保证金按当前杠杆实际计算
                            lev_now = max(float(getattr(self, "leverage", 1.0) or 1.0), 1.0)
                            margin_portion = (closed_qty * float(target_order.entry_price)) / lev_now
                            
                            entry_time_ms = int(target_order.entry_time.timestamp() * 1000) - 1000
                            entry_side = "BUY" if target_order.side == OrderSide.LONG else "SELL"
                            trades = []
                            if order_id > 0:
                                trades = self._get_user_trades(order_id=order_id, start_time_ms=entry_time_ms)
                            if not trades:
                                trades = self._get_user_trades(start_time_ms=entry_time_ms)
                            agg = self._aggregate_trades(trades, entry_side=entry_side)
                            exit_price = agg["exit_price"] or float(stage_order.get("price", 0.0)) or self._get_mark_price()
                            exit_fee = agg["exit_fee"]
                            # 本档手续费：按本档数量占比
                            original_qty = max(float(target_order.quantity), 1e-12)
                            entry_fee_total = target_order.total_fee or agg["entry_fee"]
                            entry_fee = entry_fee_total * (closed_qty / original_qty)
                            realized_pnl = agg["realized_pnl"]
                            if realized_pnl == 0.0 and exit_price > 0:
                                if target_order.side == OrderSide.LONG:
                                    realized_pnl = (exit_price - target_order.entry_price) * closed_qty
                                else:
                                    realized_pnl = (target_order.entry_price - exit_price) * closed_qty
                            net_pnl = realized_pnl - exit_fee - entry_fee
                            pnl_pct = (net_pnl / max(margin_portion, 1e-9)) * 100.0
                            exit_time = datetime.fromtimestamp(agg["last_time_ms"] / 1000) if agg["last_time_ms"] > 0 else datetime.now()
                            # 阶梯基准系统：SL始终全平（不是PARTIAL），TP分档
                            stage_reason = (
                                CloseReason.TAKE_PROFIT if (stage_type == "TP" and stage_num == 3)
                                else CloseReason.STOP_LOSS if stage_type == "SL"  # SL任何档都是全平
                                else CloseReason.PARTIAL_TP  # TP1/TP2 是分段止盈
                            )
                            closed_order = replace(
                                target_order,
                                quantity=closed_qty,
                                margin_used=margin_portion,
                                status=OrderStatus.CLOSED,
                                exit_price=exit_price,
                                exit_time=exit_time,
                                exit_bar_idx=self.current_bar_idx,
                                close_reason=stage_reason,
                                close_reason_detail=detail_msg,
                                realized_pnl=net_pnl,
                                unrealized_pnl=0.0,
                                profit_pct=pnl_pct,
                                hold_bars=max(0, self.current_bar_idx - target_order.entry_bar_idx),
                                total_fee=entry_fee + exit_fee,
                            )
                            self.order_history.append(closed_order)
                            self.save_history(self.history_file)
                            if stage_num == 3:
                                self._stage3_close_recorded = True
                            if self.on_trade_closed:
                                self.on_trade_closed(closed_order)
                        except Exception as e:
                            print(f"[BinanceTrader] ⚠ 分段成交记录写入失败: {e}")
                    
                    # 阶梯基准系统：SL任何档都是全平（返回STOP_LOSS），TP分档
                    if stage_type == "TP":
                        return CloseReason.TAKE_PROFIT if stage_num == 3 else CloseReason.PARTIAL_TP
                    else:
                        return CloseReason.STOP_LOSS  # SL始终全平，不分档
                            
            except Exception as e:
                # 忽略查询错误，继续检查下一个
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
                # 识别 R3000 的保护单（包括旧版和新的阶梯式委托单）
                if any(tag in client_id for tag in ["R3000_SL", "R3000_TP", "R3K_SL", "R3K_TP"]):
                    try:
                        self._signed_request("DELETE", "/fapi/v1/order", {
                            "symbol": self.symbol,
                            "orderId": o["orderId"],
                        })
                        print(f"[BinanceTrader] 🧹 清除残留保护单: {client_id} ({order_type})")
                    except Exception as e:
                        print(f"[BinanceTrader] ⚠ 清除残留单失败: {e}")
                elif order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "LIMIT"):
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
                         position_size_pct: Optional[float] = None,
                         entry_trajectory=None) -> Optional[str]:
        """
        放置限价开仓单 (LIMIT + GTC)
        在 trigger_price 挂限价单，等待价格触及成交（争取 Maker 0.02%）
        超时未成交会自动撤单
        
        Args:
            position_size_pct: 仓位比例（None=使用默认配置，凯利公式动态调整时传入）
            entry_trajectory: 【指纹3D图】入场轨迹矩阵，仅 PaperTrader 使用，本实现忽略
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
                "position_size_pct": position_size_pct,  # 凯利仓位，同步建仓时回填到 order 供学习
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
                "take_profit": o.get("take_profit"),
                "stop_loss": o.get("stop_loss"),
            })
        
        # ── 阶梯式止盈/止损保护单（_staged_orders，交易所实际挂单）──
        pos = self.current_position
        if pos and self._staged_orders:
            exit_side = "SELL" if pos.side == OrderSide.LONG else "BUY"
            for so in self._staged_orders:
                if so.get("filled"):
                    continue
                stype = so.get("type", "")
                snum = so.get("stage", 0)
                lbl = "止盈" if stype == "TP" else "止损"
                snapshots.append({
                    "order_id": so.get("order_id"),
                    "client_id": f"R3000_{stype}{snum}",
                    "side": exit_side,
                    "trigger_price": float(so.get("price", 0.0) or 0.0),
                    "quantity": float(so.get("quantity", 0.0) or 0.0),
                    "start_bar": -1,
                    "expire_bar": -1,
                    "remaining_bars": None,
                    "template_fingerprint": f"{lbl}第{snum}档",
                    "entry_similarity": 0.0,
                    "status": f"🎯{lbl}" if stype == "TP" else f"🛡️{lbl}",
                    "entry_price": pos.entry_price,
                    "order_type": "tp" if stype == "TP" else "sl",
                })
        else:
            # ── 旧版单档保护单（兼容）──
            pos = self.current_position
            if self._exchange_sl_order_id and self._exchange_sl_order_id > 0:
                exit_side = "BUY" if (pos and pos.side == OrderSide.SHORT) else "SELL"
                snapshots.append({
                    "order_id": self._exchange_sl_order_id,
                    "client_id": "R3000_SL",
                    "side": exit_side,
                    "trigger_price": self._exchange_sl_price,
                    "quantity": pos.quantity if pos else 0.0,
                    "start_bar": -1, "expire_bar": -1, "remaining_bars": None,
                    "template_fingerprint": "止损保护",
                    "entry_similarity": 0.0,
                    "status": "🛡️止损",
                    "entry_price": pos.entry_price if pos else None,
                    "order_type": "sl",
                })
            if self._exchange_tp_order_id and self._exchange_tp_order_id > 0:
                exit_side = "BUY" if (pos and pos.side == OrderSide.SHORT) else "SELL"
                snapshots.append({
                    "order_id": self._exchange_tp_order_id,
                    "client_id": "R3000_TP",
                    "side": exit_side,
                    "trigger_price": self._exchange_tp_price,
                    "quantity": pos.quantity if pos else 0.0,
                    "start_bar": -1, "expire_bar": -1, "remaining_bars": None,
                    "template_fingerprint": "止盈保护",
                    "entry_similarity": 0.0,
                    "status": "🎯止盈",
                    "entry_price": pos.entry_price if pos else None,
                    "order_type": "tp",
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
                      entry_reason: str = "",
                      position_size_pct: Optional[float] = None) -> Optional[PaperOrder]:
        self._sync_from_exchange(force=True)
        if self.current_position is not None:
            print("[BinanceTrader] 交易所已有持仓，跳过开仓")
            return None

        self._set_leverage(self.leverage)
        
        # 获取余额和计算数量（凯利仓位传入时用其计算数量）
        balance = self._get_usdt_balance()
        available = self._get_usdt_available_balance()
        qty = self._calc_entry_quantity(price, position_size_pct)
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

        kelly_pct = position_size_pct if position_size_pct is not None else 0.0
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
            kelly_position_pct=kelly_pct,
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
        
        # 每笔平仓（含分段止盈/分段止损）都通知 UI 写入交易记录，避免“15:43 以后记录缺失”
        if self.on_trade_closed:
            self.on_trade_closed(closed_order)
        if full_close:
            pass  # 已通知
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

        # 仅保留分段止盈/分段止损（5%、10%），不再按价格触发硬止盈/硬止损
        # ═══════════════════════════════════════════════════════════════
        # 【阶梯式委托单系统】不支持动态更新 TP/SL
        # 所有委托单在开仓时一次性挂好，无需后续同步
        # ═══════════════════════════════════════════════════════════════
        # 注释掉旧的动态更新逻辑
        # if order.stop_loss is not None and order.stop_loss > 0:
        #     if abs(order.stop_loss - self._exchange_sl_price) >= self._price_tick * 0.5:
        #         self._update_exchange_sl(order.stop_loss)
        # 
        # if order.take_profit is not None and order.take_profit > 0:
        #     if abs(order.take_profit - self._exchange_tp_price) >= self._price_tick * 0.5:
        #         self._update_exchange_tp(order.take_profit)
        
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