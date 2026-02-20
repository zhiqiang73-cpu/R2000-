"""
R3000 实时数据接收模块
通过 Binance WebSocket 接收实时K线数据

支持：
  - 公开接口（无需API Key）
  - 可配置API Key以获得更稳定连接
  - 自动重连机制
  - K线数据缓存
"""

import json
import time
import threading
import queue
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# UTC+8 上海时区
_TZ_SHANGHAI = timezone(timedelta(hours=8))

# WebSocket 库
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    print("[LiveDataFeed] 警告: websocket-client 未安装, 请运行: pip install websocket-client")

# REST API 库
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# SSL 配置
import ssl


@dataclass
class KlineData:
    """K线数据结构"""
    timestamp: int          # 毫秒时间戳
    open_time: datetime     # 开盘时间
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int         # 收盘时间戳
    is_closed: bool         # 是否已收盘（完整K线）
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }
    
    def __repr__(self):
        return f"Kline({self.open_time.strftime('%H:%M:%S')} O={self.open:.2f} H={self.high:.2f} L={self.low:.2f} C={self.close:.2f})"


class LiveDataFeed:
    """
    实时数据接收器
    
    用法：
        feed = LiveDataFeed(
            symbol="BTCUSDT",
            interval="1m",
            on_kline=my_callback,
        )
        feed.start()
        ...
        feed.stop()
    """
    
    # 默认端点（兼容旧逻辑）
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    REST_BASE_URL = "https://api.binance.com"
    
    def __init__(self,
                 symbol: str = "BTCUSDT",
                 interval: str = "1m",
                 on_kline: Optional[Callable[[KlineData], None]] = None,
                 on_price: Optional[Callable[[float, int], None]] = None,
                 on_connected: Optional[Callable[[], None]] = None,
                 on_disconnected: Optional[Callable[[str], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 history_limit: int = 500,
                 use_testnet: bool = False,
                 market_type: str = "spot",
                 http_proxy: Optional[str] = None,
                 socks_proxy: Optional[str] = None,
                 rest_poll_seconds: float = 1.0,
                 emit_realtime: bool = False,
                 realtime_emit_interval: float = 0.2):
        """
        Args:
            symbol: 交易对 (如 "BTCUSDT")
            interval: K线周期 (如 "1m", "5m", "1h")
            on_kline: K线回调函数
            on_connected: 连接成功回调
            on_disconnected: 断开连接回调
            on_error: 错误回调
            api_key: API Key (可选，用于更稳定连接)
            api_secret: API Secret (可选)
            history_limit: 历史K线缓存数量
            use_testnet: 是否使用测试网
            market_type: 市场类型 ("spot" / "futures")
            http_proxy: HTTP代理 (如 "http://127.0.0.1:7890")
            socks_proxy: SOCKS5代理 (如 "socks5://127.0.0.1:7891")
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.on_kline = on_kline
        self.on_price = on_price
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.on_error = on_error
        self.api_key = api_key
        self.api_secret = api_secret
        self.history_limit = history_limit
        self.use_testnet = use_testnet
        self.market_type = (market_type or "spot").lower()
        self.http_proxy = http_proxy
        self.socks_proxy = socks_proxy
        self.rest_poll_seconds = max(0.05, float(rest_poll_seconds))
        self.emit_realtime = bool(emit_realtime)
        self.realtime_emit_interval = max(0.01, float(realtime_emit_interval))
        
        # 解析端点（主网/测试网 + 现货/合约）
        self._ws_base_url, self._rest_base_url, self._rest_path_prefix = self._resolve_endpoints()
        
        # 状态
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._price_ws: Optional[websocket.WebSocketApp] = None
        self._price_ws_thread: Optional[threading.Thread] = None
        self._rest_poll_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._max_reconnects = 30
        self._reconnect_delay = 3  # 秒
        
        # K线缓存
        self._kline_buffer: List[KlineData] = []
        self._current_kline: Optional[KlineData] = None
        self._lock = threading.Lock()
        
        # 上一个已触发回调的收线K线时间戳（用于避免WebSocket/REST重复触发）
        self._last_emitted_closed_ts: int = 0
        self._last_emitted_realtime_ts: float = 0.0
        
        # 消息队列（用于线程安全的回调）
        self._msg_queue = queue.Queue()
    
    def _resolve_endpoints(self) -> tuple:
        """
        解析数据端点
        Returns:
            (ws_base_url, rest_base_url, rest_path_prefix)
        """
        # Futures
        if self.market_type == "futures":
            if self.use_testnet:
                # Binance Futures Testnet
                return (
                    "wss://stream.binancefuture.com/ws",
                    "https://testnet.binancefuture.com",
                    "/fapi/v1",
                )
            # Binance Futures Mainnet
            return (
                "wss://fstream.binance.com/ws",
                "https://fapi.binance.com",
                "/fapi/v1",
            )
        
        # Spot（默认）
        if self.use_testnet:
            # Binance Spot Testnet
            return (
                "wss://stream.testnet.binance.vision/ws",
                "https://testnet.binance.vision",
                "/api/v3",
            )
        
        # Binance Spot Mainnet
        return (
            self.WS_BASE_URL,
            self.REST_BASE_URL,
            "/api/v3",
        )
        
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def start(self):
        """启动数据接收"""
        if not HAS_WEBSOCKET:
            if self.on_error:
                self.on_error("websocket-client 未安装")
            return False
        
        if self._running:
            return True
        
        self._running = True
        self._reconnect_count = 0
        
        # 先获取历史K线
        self._fetch_history()
        
        # 启动 kline 回调分发线程（将 on_kline 调用从 WS 线程解耦）
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="kline-dispatch"
        )
        self._dispatch_thread.start()

        # 启动 WebSocket
        self._start_websocket()
        # 启动逐笔成交价通道（仅用于低延迟价格显示）
        self._start_price_websocket()
        # 启动 1s 级 REST 快照轮询（仅更新当前未收线K，保证UI持续流动）
        self._start_rest_poller()
        
        return True
    
    def stop(self):
        """停止数据接收"""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._price_ws:
            self._price_ws.close()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3)
        if self._price_ws_thread and self._price_ws_thread.is_alive():
            self._price_ws_thread.join(timeout=3)
        if self._rest_poll_thread and self._rest_poll_thread.is_alive():
            self._rest_poll_thread.join(timeout=3)
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            self._dispatch_thread.join(timeout=3)
        self._connected = False
    
    def get_history_df(self, include_current: bool = True) -> pd.DataFrame:
        """获取历史K线DataFrame（可选包含当前未收线K）"""
        with self._lock:
            if not self._kline_buffer:
                if include_current and self._current_kline is not None:
                    return pd.DataFrame([{
                        'timestamp': self._current_kline.timestamp,
                        'open': self._current_kline.open,
                        'high': self._current_kline.high,
                        'low': self._current_kline.low,
                        'close': self._current_kline.close,
                        'volume': self._current_kline.volume,
                    }])
                return pd.DataFrame()
            
            data = []
            for k in self._kline_buffer:
                data.append({
                    'timestamp': k.timestamp,
                    'open': k.open,
                    'high': k.high,
                    'low': k.low,
                    'close': k.close,
                    'volume': k.volume,
                })
            
            df = pd.DataFrame(data)
            # 追加当前未收线K，保证图表持续流动（尤其15m/1h）
            if include_current and self._current_kline is not None:
                ck = self._current_kline
                if len(df) == 0 or int(df.iloc[-1]['timestamp']) != ck.timestamp:
                    df = pd.concat([df, pd.DataFrame([{
                        'timestamp': ck.timestamp,
                        'open': ck.open,
                        'high': ck.high,
                        'low': ck.low,
                        'close': ck.close,
                        'volume': ck.volume,
                    }])], ignore_index=True)
                else:
                    df.iloc[-1, df.columns.get_loc('open')] = ck.open
                    df.iloc[-1, df.columns.get_loc('high')] = ck.high
                    df.iloc[-1, df.columns.get_loc('low')] = ck.low
                    df.iloc[-1, df.columns.get_loc('close')] = ck.close
                    df.iloc[-1, df.columns.get_loc('volume')] = ck.volume
            return df
    
    def get_latest_klines(self, n: int = 100) -> List[KlineData]:
        """获取最近N根K线"""
        with self._lock:
            return self._kline_buffer[-n:].copy()
    
    def get_current_kline(self) -> Optional[KlineData]:
        """获取当前未完成的K线"""
        return self._current_kline
    
    def _fetch_history(self):
        """获取历史K线数据"""
        if not HAS_REQUESTS:
            print("[LiveDataFeed] requests 未安装，跳过历史数据获取")
            return
        
        try:
            url = f"{self._rest_base_url}{self._rest_path_prefix}/klines"
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": self.history_limit,
            }
            
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            
            # 设置代理
            proxies = {}
            if self.http_proxy:
                proxies = {"http": self.http_proxy, "https": self.http_proxy}
            elif self.socks_proxy:
                proxies = {"http": self.socks_proxy, "https": self.socks_proxy}
            
            response = requests.get(url, params=params, headers=headers, timeout=15, proxies=proxies or None)
            response.raise_for_status()
            
            data = response.json()
            
            with self._lock:
                self._kline_buffer.clear()
                self._current_kline = None
                for item in data:
                    kline = self._parse_rest_kline(item)
                    if not kline:
                        continue
                    # 仅把已收线K放入历史缓存；未收线K放入 current
                    if kline.is_closed:
                        self._kline_buffer.append(kline)
                    else:
                        self._current_kline = kline
                if self._kline_buffer:
                    self._last_emitted_closed_ts = self._kline_buffer[-1].timestamp
            
            print(f"[LiveDataFeed] 获取历史K线: {len(self._kline_buffer)} 根")
            
        except Exception as e:
            print(f"[LiveDataFeed] 获取历史数据失败: {e}")
            if self.on_error:
                self.on_error(f"获取历史数据失败: {e}")
    
    def _parse_rest_kline(self, item: list) -> Optional[KlineData]:
        """解析 REST API 返回的K线数据"""
        try:
            return KlineData(
                timestamp=int(item[0]),
                open_time=datetime.fromtimestamp(int(item[0]) / 1000, tz=_TZ_SHANGHAI),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                close_time=int(item[6]),
                is_closed=(int(time.time() * 1000) >= int(item[6])),
            )
        except Exception as e:
            print(f"[LiveDataFeed] 解析K线失败: {e}")
            return None

    def _start_rest_poller(self):
        """启动REST轮询线程，每秒刷新当前未收线K线快照"""
        if not HAS_REQUESTS:
            return
        if self._rest_poll_thread and self._rest_poll_thread.is_alive():
            return
        self._rest_poll_thread = threading.Thread(
            target=self._rest_poll_loop,
            daemon=True,
        )
        self._rest_poll_thread.start()

    def _rest_poll_loop(self):
        """REST轮询循环（只更新缓存，不触发交易回调）"""
        while self._running:
            try:
                self._refresh_current_kline_from_rest()
            except Exception:
                pass
            time.sleep(self.rest_poll_seconds)

    def _refresh_current_kline_from_rest(self):
        """
        从REST拉取最新1根K线快照，更新 `_current_kline`。
        目的：即使WebSocket节流/抖动，UI仍可按1s展示K线变化。
        """
        if not HAS_REQUESTS:
            return
        url = f"{self._rest_base_url}{self._rest_path_prefix}/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": 1,
        }
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        proxies = {}
        if self.http_proxy:
            proxies = {"http": self.http_proxy, "https": self.http_proxy}
        elif self.socks_proxy:
            proxies = {"http": self.socks_proxy, "https": self.socks_proxy}

        response = requests.get(url, params=params, headers=headers, timeout=3, proxies=proxies or None)
        response.raise_for_status()
        data = response.json()
        if not data:
            return

        item = data[-1]
        open_ts = int(item[0])
        close_ts = int(item[6])
        now_ms = int(time.time() * 1000)
        is_closed = now_ms >= close_ts

        kline = KlineData(
            timestamp=open_ts,
            open_time=datetime.fromtimestamp(open_ts / 1000, tz=_TZ_SHANGHAI),
            open=float(item[1]),
            high=float(item[2]),
            low=float(item[3]),
            close=float(item[4]),
            volume=float(item[5]),
            close_time=close_ts,
            is_closed=is_closed,
        )

        emit_kline = None
        with self._lock:
            if kline.is_closed:
                if self._kline_buffer and self._kline_buffer[-1].timestamp == kline.timestamp:
                    self._kline_buffer[-1] = kline
                elif (not self._kline_buffer) or self._kline_buffer[-1].timestamp < kline.timestamp:
                    self._kline_buffer.append(kline)
                    if len(self._kline_buffer) > self.history_limit:
                        self._kline_buffer = self._kline_buffer[-self.history_limit:]
                self._current_kline = None
                # 【备用触发】按“是否已发出收线事件”去重，而不是按buffer插入去重
                if kline.timestamp > self._last_emitted_closed_ts:
                    self._last_emitted_closed_ts = kline.timestamp
                    emit_kline = kline
            else:
                self._current_kline = kline
                # 允许REST实时触发（用于秒级决策/预览）
                if self.emit_realtime and self.on_kline:
                    now = time.time()
                    if now - self._last_emitted_realtime_ts >= self.realtime_emit_interval:
                        self._last_emitted_realtime_ts = now
                        emit_kline = kline
        if emit_kline is not None and self.on_kline:
            if emit_kline.is_closed:
                print(f"[REST] 备用触发K线收线: {emit_kline.open_time} | 收盘={emit_kline.close:.2f}")
            try:
                self.on_kline(emit_kline)
            except Exception as e:
                print(f"[LiveDataFeed] REST备用回调异常: {e}")
    
    def _start_websocket(self):
        """启动 WebSocket 连接"""
        stream_name = f"{self.symbol.lower()}@kline_{self.interval}"
        ws_url = f"{self._ws_base_url}/{stream_name}"
        
        print(f"[LiveDataFeed] 正在连接WebSocket: {ws_url}")
        if self.http_proxy:
            print(f"[LiveDataFeed] 使用HTTP代理: {self.http_proxy}")
        if self.socks_proxy:
            print(f"[LiveDataFeed] 使用SOCKS代理: {self.socks_proxy}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'k' in data:
                    kline = self._parse_ws_kline(data['k'])
                    if kline:
                        if kline.is_closed:
                            print(f"[WebSocket] K线收线: {kline.open_time} | 收盘={kline.close:.2f}")
                        self._handle_kline(kline)
            except Exception as e:
                print(f"[LiveDataFeed] 解析消息失败: {e}")
        
        def on_error(ws, error):
            error_str = str(error)
            print(f"[LiveDataFeed] WebSocket 错误: {error_str}")
            
            # 检查常见错误类型
            if "10054" in error_str or "Connection refused" in error_str:
                error_msg = "网络连接被阻断 - 可能需要VPN或代理才能访问Binance WebSocket"
            elif "10060" in error_str or "timed out" in error_str.lower():
                error_msg = "连接超时 - 请检查网络连接或代理设置"
            elif "10061" in error_str:
                error_msg = "连接被拒绝 - 代理设置可能不正确"
            elif "SSL" in error_str or "certificate" in error_str.lower():
                error_msg = "SSL证书错误 - 请检查系统时间是否正确"
            elif "Handshake" in error_str:
                error_msg = "WebSocket握手失败 - 网络可能被防火墙阻断"
            else:
                error_msg = error_str
                
            if self.on_error:
                self.on_error(error_msg)
        
        def on_close(ws, close_status_code, close_msg):
            self._connected = False
            print(f"[LiveDataFeed] WebSocket 断开: code={close_status_code} msg={close_msg}")
            if self.on_disconnected:
                self.on_disconnected(f"断开: {close_msg}")
            
            # 自动重连
            if self._running and self._reconnect_count < self._max_reconnects:
                self._reconnect_count += 1
                delay = min(self._reconnect_delay * self._reconnect_count, 30)  # 指数退避，最大30秒
                print(f"[LiveDataFeed] {delay}秒后重连... (第{self._reconnect_count}/{self._max_reconnects}次)")
                time.sleep(delay)
                if self._running:
                    self._start_websocket()
            elif self._running:
                print(f"[LiveDataFeed] 已达最大重连次数({self._max_reconnects})，停止重连")
                if self.on_error:
                    self.on_error(f"WebSocket已断开，{self._max_reconnects}次重连均失败。请检查网络/代理设置。")
        
        def on_open(ws):
            self._connected = True
            self._reconnect_count = 0
            print(f"[LiveDataFeed] WebSocket 已连接: {self.symbol} {self.interval}")
            if self.on_connected:
                self.on_connected()
        
        def on_ping(ws, message):
            pass  # websocket-client 自动回复pong
        
        def on_pong(ws, message):
            pass
        
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
            on_ping=on_ping,
            on_pong=on_pong,
        )
        
        # SSL配置 - 宽松模式，避免证书验证失败
        ssl_opt = {"cert_reqs": ssl.CERT_NONE}
        
        # 代理配置
        proxy_host = None
        proxy_port = None
        proxy_type = None
        proxy_auth = None
        
        if self.http_proxy:
            # 解析 http://host:port 格式
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.http_proxy)
                proxy_host = parsed.hostname
                proxy_port = parsed.port
                proxy_type = "http"
                if parsed.username and parsed.password:
                    proxy_auth = (parsed.username, parsed.password)
                print(f"[LiveDataFeed] WebSocket使用HTTP代理: {proxy_host}:{proxy_port}")
            except Exception as e:
                print(f"[LiveDataFeed] 解析HTTP代理失败: {e}")
        elif self.socks_proxy:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.socks_proxy)
                proxy_host = parsed.hostname
                proxy_port = parsed.port
                proxy_type = "socks5"
                if parsed.username and parsed.password:
                    proxy_auth = (parsed.username, parsed.password)
                print(f"[LiveDataFeed] WebSocket使用SOCKS5代理: {proxy_host}:{proxy_port}")
            except Exception as e:
                print(f"[LiveDataFeed] 解析SOCKS代理失败: {e}")
        
        # run_forever 参数
        ws_kwargs = {
            "sslopt": ssl_opt,
            "ping_interval": 20,  # 每20秒发送ping
            "ping_timeout": 10,  # ping超时10秒
            "reconnect": 0,      # 不使用内置重连（我们自己管理）
        }
        
        if proxy_host and proxy_port:
            ws_kwargs["http_proxy_host"] = proxy_host
            ws_kwargs["http_proxy_port"] = proxy_port
            if proxy_type:
                ws_kwargs["proxy_type"] = proxy_type
            if proxy_auth:
                ws_kwargs["http_proxy_auth"] = proxy_auth
        
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs=ws_kwargs,
            daemon=True
        )
        self._ws_thread.start()

    def _start_price_websocket(self):
        """启动逐笔成交价通道（低延迟价格，仅用于UI显示）"""
        if self.on_price is None:
            return

        stream_name = f"{self.symbol.lower()}@aggTrade"
        ws_url = f"{self._ws_base_url}/{stream_name}"
        print(f"[LiveDataFeed] 正在连接Price WS: {ws_url}")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                # aggTrade: p=price, T=trade time
                if "p" in data:
                    price = float(data["p"])
                    ts_ms = int(data.get("T", int(time.time() * 1000)))
                    try:
                        self.on_price(price, ts_ms)
                    except Exception as cb_e:
                        print(f"[LiveDataFeed] Price回调异常: {cb_e}")
            except Exception as e:
                print(f"[LiveDataFeed] 解析Price消息失败: {e}")

        def on_error(ws, error):
            print(f"[LiveDataFeed] Price WS错误: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"[LiveDataFeed] Price WS断开: code={close_status_code} msg={close_msg}")
            if self._running:
                time.sleep(1.0)
                if self._running:
                    self._start_price_websocket()

        def on_open(ws):
            print(f"[LiveDataFeed] Price WS已连接: {self.symbol}")

        self._price_ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )

        ssl_opt = {"cert_reqs": ssl.CERT_NONE}
        ws_kwargs = {
            "sslopt": ssl_opt,
            "ping_interval": 20,
            "ping_timeout": 10,
            "reconnect": 0,
        }

        proxy_host = None
        proxy_port = None
        proxy_type = None
        proxy_auth = None
        if self.http_proxy:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.http_proxy)
                proxy_host = parsed.hostname
                proxy_port = parsed.port
                proxy_type = "http"
                if parsed.username and parsed.password:
                    proxy_auth = (parsed.username, parsed.password)
            except Exception:
                pass
        elif self.socks_proxy:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.socks_proxy)
                proxy_host = parsed.hostname
                proxy_port = parsed.port
                proxy_type = "socks5"
                if parsed.username and parsed.password:
                    proxy_auth = (parsed.username, parsed.password)
            except Exception:
                pass

        if proxy_host and proxy_port:
            ws_kwargs["http_proxy_host"] = proxy_host
            ws_kwargs["http_proxy_port"] = proxy_port
            if proxy_type:
                ws_kwargs["proxy_type"] = proxy_type
            if proxy_auth:
                ws_kwargs["http_proxy_auth"] = proxy_auth

        self._price_ws_thread = threading.Thread(
            target=self._price_ws.run_forever,
            kwargs=ws_kwargs,
            daemon=True,
        )
        self._price_ws_thread.start()
    
    def _parse_ws_kline(self, k: dict) -> Optional[KlineData]:
        """解析 WebSocket K线数据"""
        try:
            return KlineData(
                timestamp=int(k['t']),
                open_time=datetime.fromtimestamp(int(k['t']) / 1000, tz=_TZ_SHANGHAI),
                open=float(k['o']),
                high=float(k['h']),
                low=float(k['l']),
                close=float(k['c']),
                volume=float(k['v']),
                close_time=int(k['T']),
                is_closed=k['x'],
            )
        except Exception as e:
            print(f"[LiveDataFeed] 解析WebSocket K线失败: {e}")
            return None
    
    def _handle_kline(self, kline: KlineData):
        """处理收到的K线数据"""
        emit_callback = True
        with self._lock:
            if kline.is_closed:
                # 完整K线 → 加入缓存
                # 检查是否已存在（避免重复）
                if self._kline_buffer and self._kline_buffer[-1].timestamp == kline.timestamp:
                    self._kline_buffer[-1] = kline
                else:
                    self._kline_buffer.append(kline)
                
                # 限制缓存大小
                if len(self._kline_buffer) > self.history_limit:
                    self._kline_buffer = self._kline_buffer[-self.history_limit:]
                
                self._current_kline = None
                if kline.timestamp > self._last_emitted_closed_ts:
                    self._last_emitted_closed_ts = kline.timestamp
                    emit_callback = True
                else:
                    # 同一根收线K线已触发过（例如REST先到、WS后到），跳过重复回调
                    emit_callback = False
            else:
                # 未完成K线 → 更新当前K线
                self._current_kline = kline
                emit_callback = True
        
        # 入队交给 dispatch 线程调用，避免阻塞 WS 线程
        if emit_callback and self.on_kline:
            self._msg_queue.put(kline)
    
    def _dispatch_loop(self):
        """分发线程：从 _msg_queue 取出 kline 并调用 on_kline 回调。
        将重IO/计算的回调与 WebSocket 线程完全解耦，防止 WS Ping 超时断线。"""
        while self._running:
            try:
                kline = self._msg_queue.get(timeout=1.0)
                try:
                    self.on_kline(kline)
                except Exception as e:
                    print(f"[LiveDataFeed] dispatch回调异常: {e}")
            except queue.Empty:
                pass

    def test_connection(self) -> tuple:
        """
        测试API连接
        
        Returns:
            (success: bool, message: str)
        """
        if not HAS_REQUESTS:
            return False, "requests 库未安装"
        
        # 设置代理
        proxies = {}
        if self.http_proxy:
            proxies = {"http": self.http_proxy, "https": self.http_proxy}
        elif self.socks_proxy:
            proxies = {"http": self.socks_proxy, "https": self.socks_proxy}
        
        try:
            # 测试公开接口
            url = f"{self._rest_base_url}{self._rest_path_prefix}/ping"
            response = requests.get(url, timeout=5, proxies=proxies or None)
            response.raise_for_status()
            
            # 获取服务器时间
            url = f"{self._rest_base_url}{self._rest_path_prefix}/time"
            response = requests.get(url, timeout=5, proxies=proxies or None)
            server_time = response.json().get('serverTime', 0)
            
            # 获取交易对信息
            url = f"{self._rest_base_url}{self._rest_path_prefix}/ticker/price"
            params = {"symbol": self.symbol}
            response = requests.get(url, params=params, timeout=5, proxies=proxies or None)
            price_data = response.json()
            
            price = float(price_data.get('price', 0))
            
            proxy_info = ""
            if self.http_proxy:
                proxy_info = f" (HTTP代理)"
            elif self.socks_proxy:
                proxy_info = f" (SOCKS代理)"
            
            env = "Testnet" if self.use_testnet else "Mainnet"
            market = "Futures" if self.market_type == "futures" else "Spot"
            return True, f"连接成功{proxy_info} | {env} {market} | {self.symbol}: ${price:,.2f}"
            
        except requests.exceptions.Timeout:
            return False, "连接超时 - 请检查网络或代理设置"
        except requests.exceptions.ConnectionError as e:
            error_str = str(e)
            if "10054" in error_str or "10060" in error_str:
                return False, "网络连接失败 - 可能需要配置代理"
            return False, f"网络连接失败: {error_str[:100]}"
        except Exception as e:
            return False, f"连接失败: {str(e)}"


# 简单测试
if __name__ == "__main__":
    def on_kline(k: KlineData):
        status = "✓" if k.is_closed else "..."
        print(f"[{status}] {k}")
    
    def on_connected():
        print("=== 已连接 ===")
    
    def on_error(msg):
        print(f"!!! 错误: {msg}")
    
    feed = LiveDataFeed(
        symbol="BTCUSDT",
        interval="1m",
        on_kline=on_kline,
        on_connected=on_connected,
        on_error=on_error,
    )
    
    # 测试连接
    success, msg = feed.test_connection()
    print(f"连接测试: {msg}")
    
    if success:
        feed.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            feed.stop()
            print("已停止")
