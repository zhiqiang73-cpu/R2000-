"""
R3000 实时交易引擎
整合数据接收、模板匹配、虚拟交易的核心引擎

功能：
  - 接收实时K线数据
  - 计算32维特征向量
  - 匹配入场模板
  - 动态追踪持仓
  - 智能离场管理
"""

import threading
import time
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_data_feed import LiveDataFeed, KlineData
from config import PAPER_TRADING_CONFIG
from core.paper_trader import PaperOrder, OrderSide, CloseReason
from core.binance_testnet_trader import BinanceTestnetTrader
from core.market_regime import MarketRegimeClassifier, MarketRegime
from core.labeler import SwingPoint


@dataclass
class EngineState:
    """引擎状态"""
    is_running: bool = False
    is_connected: bool = False
    current_price: float = 0.0
    current_time: Optional[datetime] = None
    total_bars: int = 0
    
    # 当前匹配状态
    matching_phase: str = "等待"  # "等待" / "匹配入场" / "持仓中" / "匹配离场"
    best_match_similarity: float = 0.0
    best_match_template: Optional[str] = None
    
    # 追踪状态
    tracking_status: str = "-"   # "安全" / "警戒" / "脱轨"
    
    # 交易解释状态
    market_regime: str = "未知"
    fingerprint_status: str = "待匹配"
    decision_reason: str = ""
    hold_reason: str = ""        # 为何继续持仓
    danger_level: float = 0.0    # 风险度 (0-100%)
    exit_reason: str = ""        # 预估平仓理由
    position_side: str = "-"
    swing_points_count: int = 0       # 已识别的摆动点数量
    last_event: str = ""              # 最新事件（用于UI日志显示）
    entry_threshold: float = 0.7      # 运行时真实开仓阈值
    macd_ready: bool = False          # MACD 指标对齐
    kdj_ready: bool = False           # KDJ 指标对齐
    


class LiveTradingEngine:
    """
    实时交易引擎
    
    用法：
        engine = LiveTradingEngine(
            trajectory_memory=memory,
            on_state_update=my_callback,
        )
        engine.start()
        ...
        engine.stop()
    """
    
    def __init__(self,
                 trajectory_memory,
                 prototype_library=None,
                 symbol: str = "BTCUSDT",
                 interval: str = "1m",
                 initial_balance: float = 5000.0,
                 leverage: float = 10,
                 # 匹配参数
                 cosine_threshold: float = 0.7,
                 dtw_threshold: float = 0.5,
                 min_templates_agree: int = 1,
                 # 止盈止损参数
                 stop_loss_atr: float = 2.0,
                 take_profit_atr: float = 3.0,
                 max_hold_bars: int = 240,
                 # 动态追踪参数
                 hold_safe_threshold: float = 0.7,
                 hold_alert_threshold: float = 0.5,
                 hold_derail_threshold: float = 0.3,
                 hold_check_interval: int = 3,
                 # 模板筛选
                 use_qualified_only: bool = True,
                 qualified_fingerprints: Optional[set] = None,
                 qualified_prototype_fingerprints: Optional[set] = None,
                 # API配置
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 use_testnet: bool = True,
                 market_type: str = "futures",
                 # 代理配置
                 http_proxy: Optional[str] = None,
                 socks_proxy: Optional[str] = None,
                 # 风控
                 max_drawdown_pct: Optional[float] = None,
                 # 回调
                 on_state_update: Optional[Callable[[EngineState], None]] = None,
                 on_kline: Optional[Callable[[KlineData], None]] = None,
                 on_trade_opened: Optional[Callable[[PaperOrder], None]] = None,
                 on_trade_closed: Optional[Callable[[PaperOrder], None]] = None,
                 on_error: Optional[Callable[[str], None]] = None):
        """
        Args:
            trajectory_memory: TrajectoryMemory 模板记忆库
            prototype_library: PrototypeLibrary 原型库（优先于模板）
            symbol: 交易对
            interval: K线周期
            initial_balance: 初始余额
            leverage: 杠杆
            cosine_threshold: 余弦相似度阈值
            dtw_threshold: DTW阈值
            min_templates_agree: 最少匹配模板数
            stop_loss_atr: 止损ATR倍数
            take_profit_atr: 止盈ATR倍数
            max_hold_bars: 最大持仓K线数
            hold_safe_threshold: 安全阈值
            hold_alert_threshold: 警戒阈值
            hold_derail_threshold: 脱轨阈值
            hold_check_interval: 追踪检查间隔
            use_qualified_only: 是否只用合格模板
            qualified_fingerprints: 合格模板指纹集合
            qualified_prototype_fingerprints: 合格原型指纹集合（proto_LONG_x / proto_SHORT_x）
            api_key: API Key
            api_secret: API Secret
            use_testnet: 是否使用测试网
            market_type: 市场类型 ("spot" / "futures")
            on_state_update: 状态更新回调
            on_kline: K线回调
            on_trade_opened: 开仓回调
            on_trade_closed: 平仓回调
            on_error: 错误回调
        """
        self.trajectory_memory = trajectory_memory
        self.prototype_library = prototype_library
        self.use_prototypes = prototype_library is not None
        self.symbol = symbol
        self.interval = interval
        
        # 风控
        if max_drawdown_pct is None:
            try:
                from config import LIVE_RISK_CONFIG
                max_drawdown_pct = LIVE_RISK_CONFIG.get("MAX_DRAWDOWN_PCT")
            except Exception:
                max_drawdown_pct = None
        self.max_drawdown_pct = max_drawdown_pct
        
        # 匹配参数
        self.cosine_threshold = cosine_threshold
        self.dtw_threshold = dtw_threshold
        self.min_templates_agree = min_templates_agree
        
        # 止盈止损
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.max_hold_bars = max_hold_bars
        
        # 动态追踪
        self.hold_safe_threshold = hold_safe_threshold
        self.hold_alert_threshold = hold_alert_threshold
        self.hold_derail_threshold = hold_derail_threshold
        self.hold_check_interval = hold_check_interval
        
        # 模板筛选
        self.use_qualified_only = use_qualified_only
        self.qualified_fingerprints = qualified_fingerprints or set()
        self.qualified_prototype_fingerprints = qualified_prototype_fingerprints or set()
        self.use_qualified_prototypes = bool(self.qualified_prototype_fingerprints)
        
        # 回调
        self.on_state_update = on_state_update
        self.on_kline = on_kline
        self.on_trade_opened = on_trade_opened
        self.on_trade_closed = on_trade_closed
        self.on_error = on_error

        # 实时决策频率（秒）
        self._realtime_decision_interval = float(
            PAPER_TRADING_CONFIG.get("REALTIME_DECISION_SEC", 0.05)
        )
        self._realtime_entry_enabled = bool(
            PAPER_TRADING_CONFIG.get("REALTIME_ENTRY_ENABLED", True)
        )
        self._last_realtime_decision_ts = 0.0
        
        # 数据接收器
        self._data_feed = LiveDataFeed(
            symbol=symbol,
            interval=interval,
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet,
            market_type=market_type,
            on_kline=self._on_kline_received,
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_error=self._on_feed_error,
            http_proxy=http_proxy,
            socks_proxy=socks_proxy,
            rest_poll_seconds=PAPER_TRADING_CONFIG.get("REALTIME_REST_POLL_SEC", 0.05),
            emit_realtime=True,
            realtime_emit_interval=PAPER_TRADING_CONFIG.get("REALTIME_DECISION_SEC", 0.05),
        )
        
        # 执行参数固定：每次开仓 50% 仓位，杠杆 10x
        self.fixed_position_size_pct = 0.5
        self.fixed_leverage = 10

        # Binance 测试网真实执行器（不再使用本地虚拟模式）
        self._paper_trader = BinanceTestnetTrader(
            symbol=symbol,
            api_key=api_key,
            api_secret=api_secret,
            initial_balance=initial_balance,
            leverage=self.fixed_leverage,
            position_size_pct=self.fixed_position_size_pct,
            on_order_update=self._on_order_update,
            on_trade_closed=self._on_trade_closed_internal,
        )
        
        # 引擎状态
        self.state = EngineState()
        self.state.entry_threshold = self.cosine_threshold
        
        # 特征引擎和匹配器（延迟初始化）
        self._fv_engine = None
        self._matcher = None
        self._proto_matcher = None
        self._active_prototype_library = None
        self._df_buffer = None
        self._current_bar_idx = 0
        
        # 当前匹配的模板
        self._current_template = None
        self._current_prototype = None
        
        # 市场状态分类（6态上帝视角）
        self._swing_points: List[SwingPoint] = []  # 实时检测的摆动点
        self._regime_classifier: Optional[MarketRegimeClassifier] = None
        # 摆动点检测窗口（从配置读取，应与训练一致）
        from config import LABELING_CONFIG
        self._swing_window = LABELING_CONFIG.get("SWING_WINDOW", 5)
        
        # 线程控制
        self._running = False
        self._lock = threading.Lock()
        
        # 实时挂单信号 (待价格确认)
        self.pending_signal = None  # Dict with {side, trigger_price, expire_idx, fingerprint, similarity, reason}
        # 上一根K线是否有持仓（用于检测新开仓）
        self._last_had_position = False
        
        # 日志节流：避免高频决策刷屏
        self._last_log_messages: dict = {}   # key -> last_message
        self._last_log_times: dict = {}      # key -> last_time
    
    def _throttled_print(self, key: str, msg: str, interval: float = 5.0):
        """节流打印：同一 key 的相同内容在 interval 秒内只打印一次"""
        now = time.time()
        last_msg = self._last_log_messages.get(key)
        last_t = self._last_log_times.get(key, 0)
        if msg != last_msg or (now - last_t) >= interval:
            print(msg)
            self._last_log_messages[key] = msg
            self._last_log_times[key] = now

    @property
    def paper_trader(self) -> BinanceTestnetTrader:
        return self._paper_trader
    
    @property
    def data_feed(self) -> LiveDataFeed:
        return self._data_feed
    
    def test_connection(self) -> tuple:
        """测试连接"""
        return self._data_feed.test_connection()
    
    def start(self) -> bool:
        """启动引擎"""
        if self._running:
            return True
        
        print(f"[LiveEngine] 启动引擎: {self.symbol} {self.interval}")
        print(
            f"[LiveEngine] 执行参数固定: 杠杆={self.fixed_leverage}x | "
            f"单次仓位={self.fixed_position_size_pct:.0%}"
        )
        if self.use_prototypes:
            proto_count = self._active_prototype_library.total_count if self._active_prototype_library is not None else 0
            print(f"[LiveEngine] 模式: 聚合指纹图（原型）")
            print(f"[LiveEngine] 原型库: {proto_count} 个原型")
            if self.use_qualified_prototypes:
                print(f"[LiveEngine] 使用已验证原型: {len(self.qualified_prototype_fingerprints)} 个")
        else:
            tpl_count = self.trajectory_memory.total_count if self.trajectory_memory is not None else 0
            print(f"[LiveEngine] 模板库: {tpl_count} 个模板")
            if self.use_qualified_only and self.qualified_fingerprints:
                print(f"[LiveEngine] 使用合格模板: {len(self.qualified_fingerprints)} 个")
        
        # 初始化特征引擎和匹配器
        self._init_engines()
        
        # 启动数据接收
        self._running = True
        self.state.is_running = True
        
        success = self._data_feed.start()
        if not success:
            self._running = False
            self.state.is_running = False
            return False
        
        return True
    
    def stop(self):
        """停止引擎"""
        print("[LiveEngine] 停止引擎...")
        self._running = False
        self.state.is_running = False
        
        # 先强制同步一次，避免本地状态滞后
        try:
            self._paper_trader.sync_from_exchange(force=True)
        except Exception:
            pass
        # 如果有持仓，按当前价格平仓
        if self._paper_trader.has_position():
            close_price = self.state.current_price
            if close_price <= 0:
                try:
                    close_price = self._paper_trader._get_mark_price()
                except Exception:
                    close_price = self._paper_trader.current_position.entry_price
            self._paper_trader.close_position(
                close_price,
                self._current_bar_idx,
                CloseReason.MANUAL,
            )
        
        self._data_feed.stop()
        print("[LiveEngine] 引擎已停止")
    
    def reset(self):
        """重置引擎"""
        self._paper_trader.reset()
        self._current_bar_idx = 0
        self._current_template = None
        self._current_prototype = None
        self.state = EngineState()
    
    def _init_engines(self):
        """初始化特征引擎和匹配器"""
        try:
            from core.feature_vector import FeatureVectorEngine
            from core.trajectory_matcher import TrajectoryMatcher
            from core.template_clusterer import PrototypeMatcher, PrototypeLibrary
            
            self._fv_engine = FeatureVectorEngine()
            self._matcher = None
            self._proto_matcher = None

            if self.use_prototypes:
                # 构建去重后的可用原型库；若有WF验证结果则只使用验证子集
                def _filter_and_dedup(protos):
                    seen = set()
                    out = []
                    for p in protos:
                        fp = f"proto_{p.direction}_{p.prototype_id}"
                        if self.use_qualified_prototypes and fp not in self.qualified_prototype_fingerprints:
                            continue
                        if fp in seen:
                            continue
                        seen.add(fp)
                        out.append(p)
                    return out

                src = self.prototype_library
                if src is None:
                    raise ValueError("原型模式下 prototype_library 不能为空")

                self._active_prototype_library = PrototypeLibrary(
                    long_prototypes=_filter_and_dedup(src.long_prototypes),
                    short_prototypes=_filter_and_dedup(src.short_prototypes),
                    created_at=src.created_at,
                    source_template_count=src.source_template_count,
                    clustering_params=src.clustering_params,
                    source_symbol=getattr(src, "source_symbol", ""),
                    source_interval=getattr(src, "source_interval", ""),
                )
                self._proto_matcher = PrototypeMatcher(
                    library=self._active_prototype_library,
                    cosine_threshold=self.cosine_threshold,
                    min_prototypes_agree=self.min_templates_agree,
                    dtw_weight=0.1,  # 降低DTW权重（余弦90% + DTW10%）
                )
            else:
                self._matcher = TrajectoryMatcher()
            
            print("[LiveEngine] 特征引擎和匹配器已初始化")
        except Exception as e:
            print(f"[LiveEngine] 初始化失败: {e}")
            if self.on_error:
                self.on_error(f"初始化失败: {e}")
    
    def _on_connected(self):
        """连接成功回调"""
        self.state.is_connected = True
        print("[LiveEngine] 数据连接成功")
        
        # 获取历史数据并预计算特征
        self._init_features_from_history()
        
        if self.on_state_update:
            self.on_state_update(self.state)
    
    def _on_disconnected(self, msg: str):
        """断开连接回调"""
        self.state.is_connected = False
        print(f"[LiveEngine] 连接断开: {msg}")
        
        if self.on_state_update:
            self.on_state_update(self.state)
    
    def _on_feed_error(self, msg: str):
        """数据错误回调"""
        print(f"[LiveEngine] 数据错误: {msg}")
        if self.on_error:
            self.on_error(msg)
    
    def _init_features_from_history(self):
        """从历史数据初始化特征"""
        if self._fv_engine is None:
            return
        
        df = self._data_feed.get_history_df()
        if df.empty:
            print("[LiveEngine] 历史数据为空")
            return
        
        try:
            from utils.indicators import calculate_all_indicators
            
            # 添加必要的列
            df = df.rename(columns={'timestamp': 'open_time'})
            
            # 计算指标
            df = calculate_all_indicators(df)
            
            # 预计算特征
            self._fv_engine.precompute(df)
            self._df_buffer = df
            self._current_bar_idx = len(df) - 1
            # 与交易器对齐bar索引，避免同步仓位entry_bar_idx错乱
            self._paper_trader.current_bar_idx = self._current_bar_idx
            
            print(f"[LiveEngine] 历史特征计算完成: {len(df)} 根K线")
            
            # 【新增】从历史数据预先检测摆动点，避免冷启动等待
            self._init_swing_points_from_history()
            
        except Exception as e:
            print(f"[LiveEngine] 特征计算失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_swing_points_from_history(self):
        """从历史数据预先检测摆动点（避免冷启动等待）"""
        if self._df_buffer is None or len(self._df_buffer) < 20:
            return
        
        try:
            import numpy as np
            
            high = self._df_buffer['high'].values
            low = self._df_buffer['low'].values
            window = self._swing_window
            n = len(high)
            
            # 清空现有摆动点
            self._swing_points = []
            
            # 从头到尾扫描历史数据，检测所有可确认的摆动点
            # 从 window 开始，到 n - window 结束（需要前后各 window 个K线确认）
            for i in range(window, n - window):
                start = i - window
                end = i + window + 1
                
                hi = high[i]
                lo = low[i]
                
                # 检测高点
                if hi >= np.max(high[start:end]):
                    self._swing_points.append(SwingPoint(
                        index=i,
                        price=hi,
                        is_high=True,
                        atr=0.0
                    ))
                # 检测低点
                elif lo <= np.min(low[start:end]):
                    self._swing_points.append(SwingPoint(
                        index=i,
                        price=lo,
                        is_high=False,
                        atr=0.0
                    ))
            
            # 按时间排序
            self._swing_points.sort(key=lambda s: s.index)
            
            raw_count = len(self._swing_points)
            
            # 过滤为交替序列
            self._swing_points = self._filter_alternating_swings(self._swing_points)
            
            # 只保留最近的若干个摆动点（避免过多历史数据干扰）
            if len(self._swing_points) > 10:
                self._swing_points = self._swing_points[-10:]
            
            print(f"[LiveEngine] 历史摆动点预检测: {len(self._swing_points)} 个 (原始: {raw_count})")
            
            if self._swing_points:
                # 显示最近的摆动点
                recent = self._swing_points[-4:] if len(self._swing_points) >= 4 else self._swing_points
                seq = [('H' if s.is_high else 'L') + f'@{s.index}' for s in recent]
                print(f"[LiveEngine] 最近摆动点序列: {seq}")
            
        except Exception as e:
            print(f"[LiveEngine] 历史摆动点检测失败: {e}")
    
    def _on_kline_received(self, kline: KlineData):
        """K线数据回调"""
        if not self._running:
            return
        
        with self._lock:
            # 始终同步给交易器，避免交易所同步路径使用过时bar索引
            self._paper_trader.current_bar_idx = self._current_bar_idx
            # 更新状态
            self.state.current_price = kline.close
            self.state.current_time = kline.open_time
            if self._paper_trader.current_position is not None:
                self.state.position_side = self._paper_trader.current_position.side.value
            else:
                self.state.position_side = "-"
            
            # 更新动能指标状态
            self._update_indicator_state()
            
            # 回调
            if self.on_kline:
                self.on_kline(kline)
            
            # 只处理完整K线（入场/持仓决策）
            if kline.is_closed:
                self._process_closed_kline(kline)
            else:
                # 实时 tick：只更新界面显示，不做任何平仓决策
                # 所有 TP/SL/信号离场 统一在 K 线收线时处理（有完整保护期）
                if self._paper_trader.has_position():
                    order = self._paper_trader.current_position
                    if order is not None:
                        # 仅更新浮动盈亏供 UI 展示，不触发任何平仓
                        pnl = ((kline.close - order.entry_price) * order.quantity
                               if order.side == OrderSide.LONG
                               else (order.entry_price - kline.close) * order.quantity)
                        order.unrealized_pnl = pnl
                        order.profit_pct = (pnl / max(order.margin_used, 1e-9)) * 100.0
                else:
                    # 未持仓时，按秒级决策频率做预匹配/入场判断
                    now = time.time()
                    if (self._realtime_entry_enabled
                            and now - self._last_realtime_decision_ts >= self._realtime_decision_interval):
                        self._last_realtime_decision_ts = now
                        atr = self._get_current_atr()
                        self._process_entry(kline, atr)
                    else:
                        self._preview_match(kline)
            
            if self.on_state_update:
                self.on_state_update(self.state)
    
    def _process_closed_kline(self, kline: KlineData):
        """处理完整K线"""
        self.state.total_bars += 1
        self._current_bar_idx += 1
        self._paper_trader.current_bar_idx = self._current_bar_idx
        
        # 同步交易所状态（检测止损单成交等）
        if hasattr(self._paper_trader, "sync_from_exchange"):
            self._paper_trader.sync_from_exchange(force=False)
        if hasattr(self._paper_trader, "cancel_expired_entry_stop_orders"):
            self._paper_trader.cancel_expired_entry_stop_orders(self._current_bar_idx)
        
        has_pos = self._paper_trader.has_position()
        # 检测新开仓：上一根无持仓，本根有持仓 → 触发开仓回调（含止损单成交、手动开仓等）
        if has_pos and not self._last_had_position and self.on_trade_opened:
            self._ensure_position_tp_sl()
            order = self._paper_trader.current_position
            if order:
                try:
                    self.on_trade_opened(order)
                except Exception as e:
                    print(f"[LiveEngine] 开仓回调异常: {e}")
        self._last_had_position = has_pos
        
        print(f"[LiveEngine] K线收线: {kline.open_time} | 价格={kline.close:.2f} | 持仓={has_pos}")
        self.state.last_event = f"K线收线 {kline.open_time.strftime('%H:%M')} | ${kline.close:,.2f}"
        
        # 更新DataFrame和特征
        if not self._update_features(kline):
            print("[LiveEngine] 特征更新失败，跳过本K线")
            return
        
        # 获取当前ATR
        atr = self._get_current_atr()
        
        # 检查持仓状态
        if self._paper_trader.has_position():
            self._process_holding(kline, atr)
        else:
            self._throttled_print("try_entry", "[LiveEngine] 尝试入场匹配...", interval=10.0)
            self._process_entry(kline, atr)
    
    def _update_features(self, kline: KlineData) -> bool:
        """更新特征"""
        if self._df_buffer is None or self._fv_engine is None:
            return False
        
        try:
            from utils.indicators import calculate_all_indicators
            
            # 添加新K线
            new_row = pd.DataFrame([{
                'open_time': kline.timestamp,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume,
            }])
            
            self._df_buffer = pd.concat([self._df_buffer, new_row], ignore_index=True)
            
            # 限制长度
            if len(self._df_buffer) > 1000:
                self._df_buffer = self._df_buffer.iloc[-1000:].reset_index(drop=True)
            
            # 重新计算指标
            self._df_buffer = calculate_all_indicators(self._df_buffer)
            
            # 重新预计算特征
            self._fv_engine.precompute(self._df_buffer)
            
            return True
            
        except Exception as e:
            print(f"[LiveEngine] 更新特征失败: {e}")
            return False
    
    def _get_current_atr(self) -> float:
        """获取当前ATR"""
        if self._df_buffer is None or 'atr' not in self._df_buffer.columns:
            return 0.0
        
        atr = float(self._df_buffer['atr'].iloc[-1])
        if np.isnan(atr) or atr <= 0:
            high = float(self._df_buffer['high'].iloc[-1])
            low = float(self._df_buffer['low'].iloc[-1])
            atr = max(high - low, high * 0.001)
        return atr

    def _ensure_position_tp_sl(self):
        """确保当前持仓拥有 TP/SL（兼容手动开仓/交易所同步仓位）"""
        order = self._paper_trader.current_position
        if order is None:
            return
        if order.take_profit is not None and order.stop_loss is not None:
            return
        try:
            direction = order.side.value
            atr = self._get_current_atr()
            if atr <= 0:
                atr = max(order.entry_price * 0.001, 1.0)
            tp, sl = self._calculate_dynamic_tp_sl(
                entry_price=order.entry_price,
                direction=direction,
                prototype=self._current_prototype if self.use_prototypes else None,
                atr=atr,
            )
            if order.take_profit is None:
                order.take_profit = tp
            if order.stop_loss is None:
                order.stop_loss = sl
                if getattr(order, "original_stop_loss", None) is None:
                    order.original_stop_loss = sl
            self.state.last_event = (
                f"[风控补全] TP/SL已补全 | TP={order.take_profit:.2f} SL={order.stop_loss:.2f}"
            )
        except Exception as e:
            print(f"[LiveEngine] 补全TP/SL失败: {e}")
    
    def _process_entry(self, kline: KlineData, atr: float):
        """处理入场逻辑：实现 Ready-Aim-Fire 三重过滤 (已优化：支持信号动态替换)"""
        if self._risk_limit_reached():
            if self._paper_trader.has_pending_stop_orders():
                self._paper_trader.cancel_entry_stop_orders()
            self.state.matching_phase = "等待"
            self.state.fingerprint_status = "风控暂停"
            self.state.decision_reason = "风控触发：最大回撤已达阈值，暂停开仓。"
            self.state.last_event = "⚠ 风控暂停开仓"
            return
        # 实盘/测试网：只在收线时决策，避免实时tick疯狂下单
        if hasattr(self._paper_trader, "sync_from_exchange") and not kline.is_closed:
            self.state.last_event = "[入场跳过] 实盘仅在收线决策"
            return
        try:
            # 准备阶段
            self.state.matching_phase = "匹配入场"
            self.state.market_regime = self._infer_market_regime()
        except Exception as e:
            print(f"[LiveEngine] 入场前置流程失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        if self._fv_engine is None:
            self.state.last_event = "[入场跳过] 特征引擎未就绪"
            return
        if self.use_prototypes and self._proto_matcher is None:
            self.state.last_event = "[入场跳过] 原型匹配器未就绪"
            return
        if (not self.use_prototypes) and self._matcher is None:
            self.state.last_event = "[入场跳过] 模板匹配器未就绪"
            return
        
        try:
            from config import TRAJECTORY_CONFIG
            pre_entry_window = TRAJECTORY_CONFIG.get("PRE_ENTRY_WINDOW", 60)
            
            # 获取入场前轨迹
            start_idx = max(0, self._current_bar_idx - pre_entry_window)
            pre_entry_traj = self._fv_engine.get_raw_matrix(start_idx, self._current_bar_idx + 1)
            
            if pre_entry_traj.size == 0:
                self.state.matching_phase = "等待"
                self.state.last_event = "[入场跳过] 轨迹为空"
                return
            
            direction = None
            similarity = 0.0
            chosen_fp = ""

            if self.use_prototypes:
                # 关键：传入当前市场状态
                current_regime = self.state.market_regime
                
                # 【严格市场状态过滤】
                # 用户要求：regime 必须一致，不允许 UNKNOWN 状态下开仓
                match_regime = current_regime
                if current_regime == MarketRegime.UNKNOWN:
                    # UNKNOWN 状态下，不进行入场匹配，等待市场状态明确
                    self.state.decision_reason = "[等待] 市场状态未明确 (需 ≥4 个摆动点)，暂不入场。"
                    self.state.fingerprint_status = "状态未知"
                    self.state.last_event = "[入场跳过] 市场状态未知"
                    return
                
                long_result = self._proto_matcher.match_entry(
                    pre_entry_traj, direction="LONG", regime=match_regime
                )
                short_result = self._proto_matcher.match_entry(
                    pre_entry_traj, direction="SHORT", regime=match_regime
                )

                chosen_proto = None
                long_sim = long_result.get("similarity", 0.0)
                short_sim = short_result.get("similarity", 0.0)
                if long_result.get("matched") and short_result.get("matched"):
                    if long_sim >= short_sim:
                        direction, chosen_proto, similarity = "LONG", long_result.get("best_prototype"), long_sim
                    else:
                        direction, chosen_proto, similarity = "SHORT", short_result.get("best_prototype"), short_sim
                elif long_result.get("matched"):
                    direction, chosen_proto, similarity = "LONG", long_result.get("best_prototype"), long_sim
                elif short_result.get("matched"):
                    direction, chosen_proto, similarity = "SHORT", short_result.get("best_prototype"), short_sim

                # 提取匹配信息（DTW已移除）
                long_votes = long_result.get("vote_long", 0)
                short_votes = short_result.get("vote_short", 0)
                long_cosine = long_result.get("cosine_similarity", long_sim)
                short_cosine = short_result.get("cosine_similarity", short_sim)
                self._throttled_print("proto_match",
                    f"[LiveEngine] 原型匹配结果: "
                    f"LONG={long_sim:.1%}(余弦{long_cosine:.1%},投票{long_votes}) | "
                    f"SHORT={short_sim:.1%}(余弦{short_cosine:.1%},投票{short_votes})")
                
                if direction is not None and chosen_proto is not None:
                    # 显示包含市场状态的原型名称
                    regime_short = chosen_proto.regime[:2] if chosen_proto.regime else ""
                    chosen_fp = f"proto_{chosen_proto.direction}_{chosen_proto.prototype_id}_{regime_short}"
                    self._current_prototype = chosen_proto
                    self._current_template = None
                    self._throttled_print("proto_matched",
                        f"[LiveEngine] 匹配成功! 方向={direction} | 原型={chosen_fp} | 相似度={similarity:.2%}")
                    self.state.last_event = f"匹配成功 {direction} | {chosen_fp} | {similarity:.1%}"
                    
            else:
                long_candidates = self.trajectory_memory.get_templates_by_direction("LONG")
                short_candidates = self.trajectory_memory.get_templates_by_direction("SHORT")
                
                long_result = self._matcher.match_entry(
                    pre_entry_traj,
                    long_candidates,
                    cosine_threshold=self.cosine_threshold,
                    dtw_threshold=self.dtw_threshold,
                )
                short_result = self._matcher.match_entry(
                    pre_entry_traj,
                    short_candidates,
                    cosine_threshold=self.cosine_threshold,
                    dtw_threshold=self.dtw_threshold,
                )
                
                # 合格模板过滤
                if self.use_qualified_only and self.qualified_fingerprints:
                    if long_result.best_template and long_result.best_template.fingerprint() not in self.qualified_fingerprints:
                        long_result.matched = False
                    if short_result.best_template and short_result.best_template.fingerprint() not in self.qualified_fingerprints:
                        short_result.matched = False
                
                template = None
                if long_result.matched and short_result.matched:
                    if long_result.dtw_similarity >= short_result.dtw_similarity:
                        direction, template, similarity = "LONG", long_result.best_template, long_result.dtw_similarity
                    else:
                        direction, template, similarity = "SHORT", short_result.best_template, short_result.dtw_similarity
                elif long_result.matched:
                    direction, template, similarity = "LONG", long_result.best_template, long_result.dtw_similarity
                elif short_result.matched:
                    direction, template, similarity = "SHORT", short_result.best_template, short_result.dtw_similarity

                if direction is not None and template is not None:
                    chosen_fp = template.fingerprint()
                    self._current_template = template
                    self._current_prototype = None

            if direction is not None and chosen_fp:
                # 【新增：动态信号管理】
                # 如果已经有挂单，检查是否需要“更新”或“撤销”
                # 连续开仓冷却（避免信号抖动造成频繁挂单）
                from config import VECTOR_SPACE_CONFIG
                cooldown = float(VECTOR_SPACE_CONFIG.get("ENTRY_COOLDOWN_SEC", 8))
                last_ts = getattr(self._paper_trader, "_last_entry_ts", 0.0) or 0.0
                if cooldown > 0 and (time.time() - last_ts) < cooldown:
                    self.state.last_event = f"[入场跳过] 冷却中({cooldown:.0f}s)"
                    return

                has_pending = self._paper_trader.has_pending_stop_orders(current_bar_idx=self._current_bar_idx)
                if has_pending:
                    # 只有当指纹变化，或者相似度显著提升（>1%）时，才重新布防
                    is_different = (chosen_fp != self.state.best_match_template)
                    sim_improved = (similarity > (self.state.best_match_similarity + 0.01))
                    
                    if is_different or sim_improved:
                        print(f"[LiveEngine] 检测到更佳或更符合当下的信号，正在替换挂单: {self.state.best_match_template} -> {chosen_fp}")
                        self._paper_trader.cancel_entry_stop_orders()
                    else:
                        # 维持原样，不重复下单
                        self.state.last_event = "[入场跳过] 挂单未变化"
                        return

                price = kline.close
                side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
                take_profit, stop_loss = self._calculate_dynamic_tp_sl(
                    entry_price=price,
                    direction=direction,
                    prototype=chosen_proto if self.use_prototypes else None,
                    atr=atr
                )

                # 构建详细的开仓原因说明
                tp_pct = ((take_profit / price) - 1) * 100 if direction == "LONG" else ((price / take_profit) - 1) * 100
                sl_pct = ((price / stop_loss) - 1) * 100 if direction == "LONG" else ((stop_loss / price) - 1) * 100
                
                proto_info = ""
                if self.use_prototypes and chosen_proto and getattr(chosen_proto, 'member_count', 0) >= 10:
                    proto_info = (
                        f"原型={chosen_fp}(胜率={chosen_proto.win_rate:.1%}, "
                        f"平均收益={chosen_proto.avg_profit_pct:.2f}%, "
                        f"样本={chosen_proto.member_count}笔)"
                    )
                else:
                    proto_info = f"原型={chosen_fp}"
                
                reason = (
                    f"[开仓] 市场={self.state.market_regime} | {direction} | "
                    f"{proto_info} | 相似度={similarity:.1%} | "
                    f"TP={take_profit:.2f}(+{tp_pct:.1f}%) SL={stop_loss:.2f}(-{sl_pct:.1f}%)"
                )
                
                # 【三重确认逻辑】
                from config import VECTOR_SPACE_CONFIG
                confirm_pct = VECTOR_SPACE_CONFIG.get("ENTRY_CONFIRM_PCT", 0.001)
                timeout = VECTOR_SPACE_CONFIG.get("TRIGGER_TIMEOUT_BARS", 5)
                
                # A. 检查指标闸门 (Aim 瞄准)
                if not self._check_indicator_gate(self._df_buffer, direction):
                    if has_pending: 
                        # 如果指标变了不再满足，撤掉之前的单子
                        self._paper_trader.cancel_entry_stop_orders()
                    self.state.decision_reason = (
                        f"[等待瞄准] 指纹匹配成功({similarity:.1%}), 但 MACD/KDJ 动能未对齐。"
                        f"(MACD={self.state.macd_ready}, KDJ={self.state.kdj_ready})"
                    )
                    self.state.last_event = f"[门控] 未通过 | MACD={self.state.macd_ready} KDJ={self.state.kdj_ready}"
                    self.state.best_match_similarity = similarity
                    self.state.best_match_template = chosen_fp
                    return
                
                # B. 计算挂单价格（限价单入场价）
                limit_price = price * (1 + confirm_pct) if side == OrderSide.LONG else price * (1 - confirm_pct)
                self.state.last_event = (
                    f"[门控] 通过 | MACD={self.state.macd_ready} KDJ={self.state.kdj_ready} | "
                    f"限价={limit_price:.2f}"
                )
                
                # C. 直接向交易器下达“预埋开火单” (Exchange-side Stop Order)
                order_id = self._paper_trader.place_stop_order(
                    side=side,
                    trigger_price=limit_price,
                    bar_idx=self._current_bar_idx,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    template_fingerprint=chosen_fp,
                    entry_similarity=similarity,
                    entry_reason=reason,
                    timeout_bars=timeout
                )
                
                print(f"[LiveEngine] 🎯 挂限价单入场: {direction} @ {limit_price:.2f} "
                      f"(当前价={price:.2f}, 需涨跌{abs(limit_price-price):.2f})")
                
                self.state.best_match_similarity = similarity
                self.state.best_match_template = chosen_fp
                self.state.matching_phase = "待定执行"
                self.state.fingerprint_status = "等待成交"
                self.state.decision_reason = (
                    f"[🎯挂单中] 限价单已挂({similarity:.1%}) @ {limit_price:.2f} "
                    f"(MACD={self.state.macd_ready}, KDJ={self.state.kdj_ready})"
                )
                self.state.last_event = f"🎯限价单 {direction} | 挂单价 {limit_price:.2f}"
                return
            else:
                # 如果当前没有匹配到任何符合门槛的信号，但手里还有挂单
                if self._paper_trader.has_pending_stop_orders():
                    print(f"[LiveEngine] 信号已失效或走势变坏，主动撤销挂单。")
                    self._paper_trader.cancel_entry_stop_orders()
                    self.state.best_match_template = None
                    self.state.best_match_similarity = 0.0
                    self.state.matching_phase = "等待"
                    self.state.fingerprint_status = "待匹配"
                    self.state.decision_reason = "之前的指纹信号已消失或不再符合相似度要求，重回扫描模式。"
                    self.state.last_event = "[入场取消] 信号失效，撤销挂单"
                    return
                
                self.state.matching_phase = "等待"
                self.state.fingerprint_status = "未匹配"
                self.state.best_match_similarity = 0.0
                self.state.best_match_template = None
                self.state.last_event = "[入场跳过] 未匹配到信号"
            
            # 没有匹配
            self.state.matching_phase = "等待"
            self.state.fingerprint_status = "未匹配"
            self.state.best_match_similarity = 0.0
            self.state.best_match_template = None
            
            if self.use_prototypes:
                self.state.decision_reason = self._build_no_entry_reason(
                    regime=self.state.market_regime,
                    long_sim=long_result.get("similarity", 0.0),
                    short_sim=short_result.get("similarity", 0.0),
                    long_votes=long_result.get("vote_long", 0),
                    short_votes=short_result.get("vote_short", 0),
                    threshold=self.cosine_threshold,
                    min_agree=self.min_templates_agree,
                )
            else:
                self.state.decision_reason = self._build_no_entry_reason(
                    regime=self.state.market_regime,
                    long_sim=long_result.dtw_similarity,
                    short_sim=short_result.dtw_similarity,
                )
            
        except Exception as e:
            print(f"[LiveEngine] 入场匹配失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_indicator_state(self):
        """实时更新动能指标对齐状态"""
        if self._df_buffer is None or len(self._df_buffer) < 5:
            self.state.macd_ready = False
            self.state.kdj_ready = False
            return

        direction = None
        is_exit_gate = False
        
        # 1. 确定当前关注的方向
        if self._paper_trader.has_position():
            direction = self._paper_trader.current_position.side.value # LONG / SHORT
            is_exit_gate = True
        elif self.state.best_match_template:
            # 从当前原型或匹配中的模板推断方向
            if self._current_prototype:
                direction = self._current_prototype.direction
            elif "LONG" in self.state.best_match_template:
                direction = "LONG"
            elif "SHORT" in self.state.best_match_template:
                direction = "SHORT"

        if not direction:
            self.state.macd_ready = False
            self.state.kdj_ready = False
            return

        # 2. 计算指标状态
        df = self._df_buffer
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        if not is_exit_gate:
            # 入场逻辑 (Aim 瞄准阶段) - 趋势一致性判定（非单根动能）
            if direction == "LONG":
                # MACD 多头趋势：柱体在零轴上方，或至少近两根在回升
                self.state.macd_ready = (curr['macd_hist'] > 0) or (curr['macd_hist'] > prev['macd_hist'])
                # KDJ 多头趋势：J/K 站上 D 且动能向上（不再用 90 上限硬限制）
                self.state.kdj_ready = (
                    ((curr['j'] >= curr['d']) or (curr['k'] >= curr['d'])) and
                    ((curr['j'] >= prev['j']) or (curr['k'] >= prev['k']))
                )
            else: # SHORT
                # MACD 空头趋势：柱体在零轴下方，或至少近两根在走弱
                self.state.macd_ready = (curr['macd_hist'] < 0) or (curr['macd_hist'] < prev['macd_hist'])
                # KDJ 空头趋势：J/K 位于 D 下方且动能向下（不再用 10 下限硬限制）
                self.state.kdj_ready = (
                    ((curr['j'] <= curr['d']) or (curr['k'] <= curr['d'])) and
                    ((curr['j'] <= prev['j']) or (curr['k'] <= prev['k']))
                )
        else:
            # 离场逻辑 (Ready 表示门控已打开，允许平仓)
            if direction == "LONG":
                self.state.macd_ready = curr['macd_hist'] < prev['macd_hist'] or curr['macd_hist'] < 0
                self.state.kdj_ready = curr['j'] < prev['j']
            else: # SHORT
                self.state.macd_ready = curr['macd_hist'] > prev['macd_hist'] or curr['macd_hist'] > 0
                self.state.kdj_ready = curr['j'] > prev['j']

    def _preview_match(self, kline: KlineData):
        """K线未收线时的预匹配展示（不下单，仅更新UI状态供用户参考）"""
        # 【关键】更新市场状态，确保UI始终显示最新市场状态
        self.state.market_regime = self._infer_market_regime()
        
        if self._fv_engine is None:
            return
        if self.use_prototypes and self._proto_matcher is None:
            return
        if (not self.use_prototypes) and self._matcher is None:
            return

        try:
            from config import TRAJECTORY_CONFIG
            pre_entry_window = TRAJECTORY_CONFIG.get("PRE_ENTRY_WINDOW", 60)
            start_idx = max(0, self._current_bar_idx - pre_entry_window)
            pre_entry_traj = self._fv_engine.get_raw_matrix(start_idx, self._current_bar_idx + 1)
            if pre_entry_traj.size == 0:
                return

            best_sim = 0.0
            best_fp = ""
            best_dir = ""
            long_sim = 0.0 
            short_sim = 0.0

            if self.use_prototypes:
                match_regime = self.state.market_regime
                if match_regime == MarketRegime.UNKNOWN:
                    # 与 _process_entry 保持一致：UNKNOWN 状态不匹配
                    self.state.fingerprint_status = "状态未知"
                    self.state.decision_reason = "[等待] 市场状态未明确 (需 ≥4 个摆动点)，暂不入场。"
                    return
                lp = self._proto_matcher.match_entry(pre_entry_traj, direction="LONG", regime=match_regime)
                sp = self._proto_matcher.match_entry(pre_entry_traj, direction="SHORT", regime=match_regime)
                long_sim = lp.get("similarity", 0.0)
                short_sim = sp.get("similarity", 0.0)
                if long_sim > short_sim:
                    best_sim, best_dir = long_sim, "LONG"
                    p = lp.get("best_prototype")
                    best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                else:
                    best_sim, best_dir = short_sim, "SHORT"
                    p = sp.get("best_prototype")
                    best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
            
            self.state.best_match_similarity = best_sim
            self.state.best_match_template = best_fp
            
            # 【新增】实时决策说明
            if best_sim >= self.cosine_threshold:
                self.state.fingerprint_status = "匹配达标"
                # 检查指标状态
                macd_ok = self.state.macd_ready
                kdj_ok = self.state.kdj_ready
                if macd_ok and kdj_ok:
                    self.state.decision_reason = f"匹配成功({best_sim:.1%})，动能已对齐。等待本K线收线确认开仓..."
                else:
                    missing = []
                    if not macd_ok: missing.append("MACD")
                    if not kdj_ok: missing.append("KDJ")
                    self.state.decision_reason = f"指纹匹配达标({best_sim:.1%})，正在等待 {' & '.join(missing)} 动能对齐。"
            elif best_sim > 0.3:
                self.state.fingerprint_status = "扫描中"
                self.state.decision_reason = f"正在扫描潜在信号({best_sim:.1%})..."
            else:
                self.state.fingerprint_status = "待匹配"
                self.state.decision_reason = "扫描市场中，寻找符合历史特征的极值点走势..."
            
        except Exception as e:
            print(f"[LiveEngine] 预匹配失败: {e}")

    def _check_indicator_gate(self, df: pd.DataFrame, direction: str) -> bool:
        """
        第二层确认：技术指标共振 (Aim)
        检查 MACD 和 KDJ 是否配合指纹图方向
        """
        if df is None or len(df) < 5:
            return False
            
        # 获取最新两根数据进行对比（趋势一致口径）
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if direction == "LONG":
            # MACD 多头趋势：零轴上方或近两根回升
            macd_ok = (curr['macd_hist'] > 0) or (curr['macd_hist'] > prev['macd_hist'])
            self.state.macd_ready = macd_ok
            
            # KDJ 多头趋势：J/K 站上 D 且动能向上（不使用90上限硬限制）
            kdj_ok = (
                ((curr['j'] >= curr['d']) or (curr['k'] >= curr['d'])) and
                ((curr['j'] >= prev['j']) or (curr['k'] >= prev['k']))
            )
            self.state.kdj_ready = kdj_ok
            
            return macd_ok and kdj_ok
            
        elif direction == "SHORT":
            # MACD 空头趋势：零轴下方或近两根走弱
            macd_ok = (curr['macd_hist'] < 0) or (curr['macd_hist'] < prev['macd_hist'])
            self.state.macd_ready = macd_ok
            
            # KDJ 空头趋势：J/K 位于 D 下方且动能向下（不使用10下限硬限制）
            kdj_ok = (
                ((curr['j'] <= curr['d']) or (curr['k'] <= curr['d'])) and
                ((curr['j'] <= prev['j']) or (curr['k'] <= prev['k']))
            )
            self.state.kdj_ready = kdj_ok
            
            return macd_ok and kdj_ok
            
        return False

    def _check_price_trigger(self, current_price: float) -> bool:
        """
        第三层确认：价格突破 (Fire)
        """
        if not self.pending_signal:
            return False
            
        sig = self.pending_signal
        
        # 检查是否超时
        if self._current_bar_idx > sig['expire_idx']:
            print(f"[LiveEngine] 信号超时过期: {sig['side']} {sig['fingerprint']}")
            self.pending_signal = None
            return False
            
        # 检查价格是否触碰到触发价格
        if sig['side'] == OrderSide.LONG:
            if current_price >= sig['trigger_price']:
                return True
        else: # SHORT
            if current_price <= sig['trigger_price']:
                return True
                
        return False
    # ══════════════════════════════════════════════════════════════════
    #  统一平仓出口 — 交易大师的铁律：所有退出必经同一扇门
    # ══════════════════════════════════════════════════════════════════

    def _reset_position_state(self, reason_text: str = ""):
        """
        平仓后统一重置所有持仓相关状态 — 唯一清理出口

        无论平仓由谁触发（TP/SL、信号、脱轨、超时），都经过此处，
        确保不会遗留任何"幽灵状态"影响下一次决策。
        """
        self.state.matching_phase = "等待"
        self.state.tracking_status = "-"
        self.state.position_side = "-"
        self.state.hold_reason = ""
        self.state.exit_reason = ""
        self.state.danger_level = 0.0
        self.state.fingerprint_status = "待匹配"
        self._current_template = None
        self._current_prototype = None
        if reason_text:
            self.state.decision_reason = reason_text

    def _close_and_reset(self, price: float, bar_idx: int, reason: CloseReason,
                          order=None, reason_text: str = "") -> bool:
        """
        主动平仓 + 状态重置 — 唯一的主动平仓出口

        所有由引擎主动触发的平仓（超时、信号离场等）必须走此方法，
        确保平仓动作与状态清理原子化执行。

        Returns:
            True 平仓成功, False 平仓失败（需等待重试）
        """
        closed = self._paper_trader.close_position(price, bar_idx, reason)
        if not closed:
            self.state.exit_reason = "下单失败，等待重试。"
            return False
        # close_position 内部会触发 _on_trade_closed_internal → _reset_position_state()
        # 这里再次调用以设置具体的 decision_reason
        if not reason_text and order:
            reason_text = self._build_exit_reason(reason.value, order)
        elif not reason_text:
            reason_text = f"[平仓] {reason.value}"
        
        # 将详细原因写入最近关闭的订单记录
        if self._paper_trader.order_history:
            self._paper_trader.order_history[-1].decision_reason = reason_text
        
        self._reset_position_state(reason_text)
        return True

    # ══════════════════════════════════════════════════════════════════
    #  持仓管理主流程
    # ══════════════════════════════════════════════════════════════════

    def _process_holding(self, kline: KlineData, atr: float):
        """处理持仓逻辑"""
        self.state.matching_phase = "持仓中"
        self.state.fingerprint_status = "持仓追踪中"
        
        order = self._paper_trader.current_position
        if order is None:
            return

        # 刚开仓到首次相似度巡检前，避免UI显示“未持仓/0%”
        if not self.state.hold_reason:
            self.state.hold_reason = "已开仓，等待下一次持仓相似度巡检。"
        if self.state.danger_level <= 0:
            default_danger = {"安全": 5.0, "警戒": 55.0, "危险": 80.0, "脱轨": 100.0}
            self.state.danger_level = default_danger.get(order.tracking_status, 5.0)
        if not self.state.exit_reason:
            self.state.exit_reason = "形态配合良好，暂无平仓预兆。"
        
        # 【新增】更新详细的离场/持有说明
        pnl_pct = order.profit_pct
        self.state.decision_reason = f"[持仓中] {order.side.value} | 相似度={order.current_similarity:.1%} | 收益={pnl_pct:+.2f}%"
        
        # 新开仓保护期：8秒内止损暂缓触发，允许止盈
        hold_seconds = 0.0
        try:
            hold_seconds = max(0.0, (datetime.now() - order.entry_time).total_seconds())
        except Exception:
            hold_seconds = 0.0
        
        in_protection = hold_seconds < 8
        
        # 更新价格，检查止盈止损（保护期内禁止止损）
        close_reason = self._paper_trader.update_price(
            kline.close,
            high=kline.high,
            low=kline.low,
            bar_idx=self._current_bar_idx,
            protection_mode=in_protection  # 传递保护期状态
        )
        
        if close_reason:
            # 详细的平仓决策日志
            reason_detail = self._get_tp_sl_trigger_reason(order, close_reason, kline)
            reason_text = self._build_exit_reason(reason_detail, order)
            print(f"[LiveEngine] 💰 {reason_text}")
            # 将详细原因写入最近关闭的订单记录
            if self._paper_trader.order_history:
                self._paper_trader.order_history[-1].decision_reason = reason_text
            self._reset_position_state(reason_text)
            return

        # 保护期内跳过相似度检查和追踪止损调整
        if in_protection:
            self.state.hold_reason = "新开仓保护期(8秒)，止损暂缓、允许止盈。"
            self.state.exit_reason = "保护期内不执行相似度离场和追踪止损调整。"
            return
        
        # ══════════════════════════════════════════════════════════
        # 【新增】市场因素一致性监控：市场状态 + MACD + KDJ
        # 如果三者一致反转，主动触发离场
        # ══════════════════════════════════════════════════════════
        self.state.market_regime = self._infer_market_regime()  # 持仓期间持续更新市场状态
        
        if order.hold_bars >= 3:  # 与信号离场保护一致
            market_reversal = self._check_market_reversal(order)
            if market_reversal["should_exit"]:
                # 输出详细的决策依据
                d = market_reversal["details"]
                curr = self._df_buffer.iloc[-1]
                prev = self._df_buffer.iloc[-2]
                print(f"[LiveEngine] 🔄 市场因素一致反转触发:")
                print(f"  ├─ 方向: {order.side.value} | 持仓: {order.hold_bars}根K线")
                print(f"  ├─ 市场状态: {d['regime']}")
                print(f"  ├─ MACD柱: {prev['macd_hist']:.2f} → {curr['macd_hist']:.2f} "
                      f"({'转空✓' if d['macd_bearish'] else '转多✓' if d['macd_bullish'] else '中性✗'})")
                print(f"  ├─ KDJ-J: {prev['j']:.1f} → {curr['j']:.1f} "
                      f"({'转空✓' if d['kdj_bearish'] else '转多✓' if d['kdj_bullish'] else '中性✗'})")
                print(f"  └─ 结论: {market_reversal['reason']}")
                
                reason_text = self._build_exit_reason(f"市场反转({market_reversal['reason']})", order)
                if self._close_and_reset(kline.close, self._current_bar_idx,
                                         CloseReason.SIGNAL, order, reason_text):
                    return
        
        # 三阶段追踪止损 + 追踪止盈
        self._update_trailing_stop(order, kline, atr)
        
        # 分段减仓：阶梯式落袋为安
        self._check_staged_partial_tp(kline)
        
        # 最大持仓安全网（防止相似度一直在0.5~0.7之间缓慢失血）
        max_hold = getattr(self, 'max_hold_bars', 240)
        if max_hold > 0 and order.hold_bars >= max_hold:
            print(f"[LiveEngine] 超过最大持仓时间 {max_hold} 根K线，强制平仓")
            self._close_and_reset(kline.close, self._current_bar_idx, CloseReason.MAX_HOLD, order)
            return
        
        # 动态追踪检查：首根K线必检（早期脱轨检测）+ 每N根定期检查
        should_check = (
            order.hold_bars == 1  # 首根K线必检
            or (order.hold_bars > 0 and order.hold_bars % self.hold_check_interval == 0)
        )
        if should_check:
            self._check_holding_similarity(kline)
    
    def _check_holding_similarity(self, kline: KlineData):
        """
        检查持仓相似度（动态追踪）
        
        三阶段匹配系统的第二、三阶段：
        1. 持仓健康度监控 - 与当前原型的持仓段对比
        2. 持仓重匹配 - 如果有更匹配的原型则切换
        3. 离场模式检测 - 检查是否开始像原型的出场段
        """
        if self._fv_engine is None:
            return
        if self.use_prototypes and self._current_prototype is None:
            return
        if (not self.use_prototypes) and self._current_template is None:
            return
        
        order = self._paper_trader.current_position
        if order is None:
            return
        
        try:
            # 获取持仓轨迹
            holding_traj = self._fv_engine.get_raw_matrix(
                order.entry_bar_idx, self._current_bar_idx + 1
            )
            
            if holding_traj.size == 0:
                return
            
            direction = "LONG" if order.side == OrderSide.LONG else "SHORT"
            
            if self.use_prototypes:
                # ══════════════════════════════════════════════════════════
                # 阶段1：持仓健康度监控
                # ══════════════════════════════════════════════════════════
                similarity, health_status = self._proto_matcher.check_holding_health(
                    holding_traj, self._current_prototype
                )
                
                # ══════════════════════════════════════════════════════════
                # 阶段2：持仓重匹配 - 检查是否有更匹配的原型
                # ══════════════════════════════════════════════════════════
                # 仅在相似度下降时尝试重匹配（节省计算）
                if similarity < self.hold_safe_threshold and order.hold_bars >= 5:
                    # 获取当前市场状态用于过滤
                    current_regime = self.state.market_regime
                    if current_regime == MarketRegime.UNKNOWN:
                        current_regime = None  # 预热期不过滤
                    
                    new_proto, new_sim, switched = self._proto_matcher.rematch_by_holding(
                        holding_traj,
                        self._current_prototype,
                        direction,
                        regime=current_regime,
                        switch_threshold=0.1,  # 新原型需超出10%才切换
                    )
                    
                    if switched:
                        old_id = self._current_prototype.prototype_id
                        self._current_prototype = new_proto
                        similarity = new_sim
                        print(f"[LiveEngine] 持仓切换原型: {old_id} → {new_proto.prototype_id} "
                              f"(相似度: {new_sim:.1%})")
                        
                        # 更新订单的模板指纹
                        order.template_fingerprint = f"proto_{direction}_{new_proto.prototype_id}"

                        # 【新增】同步更新 TP/SL 目标
                        atr = self._get_current_atr()
                        new_tp, new_sl = self._calculate_dynamic_tp_sl(
                            entry_price=order.entry_price,
                            direction=direction,
                            prototype=new_proto,
                            atr=atr
                        )
                        order.take_profit = new_tp
                        order.stop_loss = new_sl
                        print(f"[LiveEngine] TP/SL 已随原型同步更新: TP={new_tp:.2f}, SL={new_sl:.2f}")
                
                # ══════════════════════════════════════════════════════════
                # 阶段3：离场模式检测
                # ══════════════════════════════════════════════════════════
                # 【关键保护】持仓不足3根K线时，轨迹数据不可靠，
                # 禁止信号离场，只允许 TP/SL 硬保护（已在 update_price 中处理）
                MIN_HOLD_BARS_FOR_SIGNAL_EXIT = 3
                if order.hold_bars < MIN_HOLD_BARS_FOR_SIGNAL_EXIT:
                    self.state.exit_reason = (
                        f"持仓{order.hold_bars}根K线，需≥{MIN_HOLD_BARS_FOR_SIGNAL_EXIT}根才启用信号离场，"
                        f"当前仅TP/SL硬保护生效。"
                    )
                else:
                    # 取最近的轨迹（持仓末尾）用于出场模式匹配
                    from config import TRAJECTORY_CONFIG
                    pre_exit_window = TRAJECTORY_CONFIG.get("PRE_EXIT_WINDOW", 10)
                    recent_traj = holding_traj[-pre_exit_window:] if len(holding_traj) >= pre_exit_window else holding_traj
                    
                    exit_check = self._proto_matcher.check_exit_pattern(
                        recent_trajectory=recent_traj,
                        current_prototype=self._current_prototype,
                        direction=direction,
                        entry_price=order.entry_price,
                        current_price=kline.close,
                        stop_loss=order.stop_loss or order.entry_price,
                        take_profit=order.take_profit or order.entry_price,
                        current_regime=self.state.market_regime,
                    )
                    
                    # 如果离场模式检测建议离场，且通过指标确认闸门
                    if exit_check["should_exit"]:
                        # 离场指标确认 (MACD + KDJ 共振)
                        gate_result = self._check_exit_indicator_gate(self._df_buffer, direction)
                        
                        # 输出详细的决策依据
                        print(f"[LiveEngine] 📊 信号离场决策分析:")
                        print(f"  ├─ 方向: {direction} | 持仓: {order.hold_bars}根K线")
                        print(f"  ├─ 形态匹配: {exit_check['pattern_similarity']:.1%} | 信号强度: {exit_check['exit_signal_strength']:.1%}")
                        print(f"  ├─ 离场原因: {exit_check['exit_reason']}")
                        
                        if "details" in gate_result and gate_result["details"]:
                            d = gate_result["details"]
                            print(f"  ├─ MACD柱: {d['macd_prev']:.2f} → {d['macd_curr']:.2f} ({d['macd_status']})")
                            print(f"  ├─ KDJ-J: {d['kdj_prev']:.1f} → {d['kdj_curr']:.1f} ({d['kdj_status']})")
                        
                        print(f"  └─ 指标闸门: {'通过✓' if gate_result['passed'] else '未通过✗'} ({gate_result['reason']})")
                        
                        if not gate_result["passed"]:
                            msg = f"形态拟出场，但指标动能支撑({gate_result['reason']})，暂缓离场。"
                            self.state.exit_reason = msg
                            self.state.decision_reason = f"[持仓中] {msg}"
                        else:
                            exit_reason_str = exit_check["exit_reason"]
                            reason_text = self._build_exit_reason(f"信号({exit_reason_str})", order)
                            if not self._close_and_reset(kline.close, self._current_bar_idx,
                                                         CloseReason.SIGNAL, order, reason_text):
                                return  # 下单失败，等待重试
                            return
                    
                    # 更新状态中的出场预估
                    if exit_check["exit_signal_strength"] > 0.3:
                        self.state.exit_reason = (
                            f"出场信号 {exit_check['exit_signal_strength']:.0%} | "
                            f"模式匹配 {exit_check['pattern_similarity']:.0%} | "
                            f"价格位置 {exit_check['price_position']:+.0%}"
                        )
                
            else:
                # 模板模式（旧逻辑）
                divergence, _ = self._matcher.monitor_holding(
                    holding_traj,
                    self._current_template,
                    divergence_limit=1.0 - self.hold_derail_threshold,
                )
                similarity = max(0.0, 1.0 - divergence)
            
            # ══════════════════════════════════════════════════════════
            # 更新追踪状态（原有逻辑）
            # ══════════════════════════════════════════════════════════
            close_reason = self._paper_trader.update_tracking_status(
                similarity,
                safe_threshold=self.hold_safe_threshold,
                alert_threshold=self.hold_alert_threshold,
                derail_threshold=self.hold_derail_threshold,
                current_price=kline.close,
                bar_idx=self._current_bar_idx,
            )
            
            self.state.tracking_status = order.tracking_status
            self.state.best_match_similarity = similarity
            
            # 填充持仓监控说明
            status_map = {"安全": "形态配合完美", "警戒": "形态轻微偏离"}
            hold_desc = status_map.get(order.tracking_status, "形态匹配中")
            self.state.hold_reason = f"相似度 {similarity:.1%} >= 警戒线 {self.hold_alert_threshold:.1%}，{hold_desc}，故继续持仓。"
            
            # 持仓风险度
            danger = max(0.0, (1.0 - similarity) / (1.0 - self.hold_derail_threshold)) * 100
            self.state.danger_level = min(100.0, danger)
            
            # 如果没有更具体的出场预估，使用默认
            if not self.state.exit_reason or similarity < self.hold_safe_threshold:
                if similarity < self.hold_safe_threshold:
                    self.state.exit_reason = f"相似度下降 ({similarity:.1%})，若跌破 {self.hold_derail_threshold:.1%} 触发【脱轨】。"
                else:
                    self.state.exit_reason = "形态配合良好，暂无平仓预兆。"
            
            if close_reason:
                self._reset_position_state(self._build_exit_reason("脱轨", order))
            
        except Exception as e:
            import traceback
            print(f"[LiveEngine] 持仓追踪失败: {e}")
            traceback.print_exc()
    
    def _calculate_dynamic_tp_sl(self, entry_price: float, direction: str,
                                  prototype, atr: float):
        """
        【三因子融合】基于原型历史表现 + ATR波动率 + 固定下限 计算止盈止损
        
        三因子设计：
        1. 原型历史表现因子：基于avg_profit_pct和win_rate
        2. ATR波动率因子：至少2.0倍ATR，适应市场波动
        3. 固定百分比下限：至少0.15%（BTC约$100），避免噪声止损
        
        止损 = max(三因子)，永远不会太紧
        止盈 >= 止损 * 风险收益比，保证盈亏比合理
        
        Args:
            entry_price: 入场价格
            direction: LONG/SHORT
            prototype: 匹配的原型（Prototype对象）
            atr: 当前ATR
        
        Returns:
            (take_profit_price, stop_loss_price)
        """
        import numpy as np
        leverage = float(self._paper_trader.leverage)
        
        # ========== 因子1: 基于原型历史表现 ==========
        if prototype and getattr(prototype, 'member_count', 0) >= 10:
            raw_profit_pct = np.clip(prototype.avg_profit_pct, 0.5, 10.0)
            price_move_pct = raw_profit_pct / leverage / 100.0  # 还原为价格百分比
            win_rate = prototype.win_rate
            
            # 根据胜率调整止盈目标（高胜率更激进）
            if win_rate >= 0.75:
                price_move_pct *= 1.2
            elif win_rate < 0.60:
                price_move_pct *= 0.8
        else:
            # 回退：使用ATR倍数
            price_move_pct = (atr * self.take_profit_atr) / entry_price
            win_rate = 0.5
        
        # ========== 因子2: ATR波动率止损（市场适应性）==========
        # 至少 2.0 倍 ATR，保证在正常波动范围外
        atr_based_sl_pct = (atr / entry_price) * 2.0
        
        # ========== 因子3: 固定百分比下限（避免噪声止损）==========
        # BTC 1分钟线，至少 0.2% 距离（约 $130+），强制保护
        # 其他币种可根据价格自动调整
        min_fixed_pct = 0.002  # 0.2% - 增强保护，防止过快止损
        
        # ========== 风险收益比（基于胜率）==========
        if win_rate >= 0.70:
            risk_reward_ratio = 2.0  # 高胜率：1:2
        elif win_rate >= 0.50:
            risk_reward_ratio = 1.5  # 中胜率：1:1.5
        else:
            risk_reward_ratio = 1.0  # 低胜率：1:1
        
        # 原型建议的止损（按风险收益比）
        prototype_sl_pct = price_move_pct / risk_reward_ratio
        
        # ========== 综合：止损取三因子最大值 ==========
        stop_loss_pct = max(prototype_sl_pct, atr_based_sl_pct, min_fixed_pct)
        
        # ========== 止盈：至少要比止损大，保证盈亏比 ==========
        take_profit_pct = max(price_move_pct, stop_loss_pct * risk_reward_ratio)
        
        # ========== 计算最终价格 ==========
        if direction == "LONG":
            take_profit = entry_price * (1 + take_profit_pct)
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - take_profit_pct)
            stop_loss = entry_price * (1 + stop_loss_pct)
        
        # ========== 最终安全检查：确保止损距离至少 0.2% ==========
        actual_sl_distance = abs(stop_loss - entry_price)
        min_sl_distance = entry_price * min_fixed_pct
        if actual_sl_distance < min_sl_distance:
            print(f"[LiveEngine] ⚠️ 止损距离过小({actual_sl_distance:.2f})，强制调整到 {min_sl_distance:.2f}")
            if direction == "LONG":
                stop_loss = entry_price * (1 - min_fixed_pct)
            else:
                stop_loss = entry_price * (1 + min_fixed_pct)
            # 重新计算实际百分比
            stop_loss_pct = min_fixed_pct
        
        # 详细日志
        print(f"[LiveEngine] 三因子TP/SL:")
        print(f"  - 原型因子: {prototype_sl_pct*100:.3f}% (收益{price_move_pct*100:.3f}% / RR={risk_reward_ratio})")
        print(f"  - ATR因子:  {atr_based_sl_pct*100:.3f}% ({atr:.2f}*2.0)")
        print(f"  - 固定下限: {min_fixed_pct*100:.3f}%")
        print(f"  → 最终SL={stop_loss_pct*100:.3f}% (${abs(stop_loss-entry_price):.2f}) | "
              f"TP={take_profit_pct*100:.3f}% (${abs(take_profit-entry_price):.2f})")
        
        return take_profit, stop_loss
    
    def _on_order_update(self, order: PaperOrder):
        """订单更新回调"""
        pass  # 由状态更新回调处理
    
    def _on_trade_closed_internal(self, order: PaperOrder):
        """交易关闭内部回调 — 安全网，确保任何平仓路径都能清理状态"""
        if self.on_trade_closed:
            self.on_trade_closed(order)
        # 统一走 _reset_position_state，不设 reason_text（由调用方设定具体原因）
        self._reset_position_state()
    
    def _infer_market_regime(self) -> str:
        """
        使用上帝视角6态市场状态分类（与训练一致）
        
        6个状态：
          - 强多头 (STRONG_BULL)
          - 弱多头 (WEAK_BULL)
          - 震荡偏多 (RANGE_BULL)
          - 震荡偏空 (RANGE_BEAR)
          - 弱空头 (WEAK_BEAR)
          - 强空头 (STRONG_BEAR)
        """
        if self._df_buffer is None or len(self._df_buffer) < 30:
            return MarketRegime.UNKNOWN
        
        try:
            # 1. 更新摆动点检测（只使用已确认的历史数据）
            self._update_swing_points()
            
            # 更新状态中的摆动点计数（供UI显示）
            self.state.swing_points_count = len(self._swing_points)
            
            # 2. 检查是否有足够的摆动点（与上帝视角一致，需要 4 个：2高2低）
            if len(self._swing_points) < 4:
                return MarketRegime.UNKNOWN
            
            # 3. 创建/更新分类器
            from config import MARKET_REGIME_CONFIG
            self._regime_classifier = MarketRegimeClassifier(
                alternating_swings=self._swing_points,
                config=MARKET_REGIME_CONFIG
            )
            
            # 4. 分类当前K线的市场状态
            current_idx = len(self._df_buffer) - 1
            regime = self._regime_classifier.classify_at(current_idx)

            # 5. 短期趋势修正（让 regime 更贴近 K 线走势）
            try:
                from config import MARKET_REGIME_CONFIG
                lookback = int(MARKET_REGIME_CONFIG.get("SHORT_TREND_LOOKBACK", 12))
                threshold = float(MARKET_REGIME_CONFIG.get("SHORT_TREND_THRESHOLD", 0.002))
                if lookback > 0 and len(self._df_buffer) >= lookback + 1:
                    start_px = float(self._df_buffer["close"].iloc[-lookback - 1])
                    end_px = float(self._df_buffer["close"].iloc[-1])
                    if start_px > 0:
                        short_trend = (end_px - start_px) / start_px
                        # 短期明显向上：偏空/弱空 → 修正为震荡偏多/弱多
                        if short_trend > threshold:
                            if short_trend > threshold * 2:
                                if regime in (MarketRegime.RANGE_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.STRONG_BEAR, MarketRegime.UNKNOWN):
                                    return MarketRegime.WEAK_BULL
                            else:
                                if regime in (MarketRegime.RANGE_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.STRONG_BEAR, MarketRegime.UNKNOWN):
                                    return MarketRegime.RANGE_BULL
                        # 短期明显向下：偏多/弱多 → 修正为震荡偏空/弱空
                        if short_trend < -threshold:
                            if short_trend < -threshold * 2:
                                if regime in (MarketRegime.RANGE_BULL, MarketRegime.WEAK_BULL, MarketRegime.STRONG_BULL, MarketRegime.UNKNOWN):
                                    return MarketRegime.WEAK_BEAR
                            else:
                                if regime in (MarketRegime.RANGE_BULL, MarketRegime.WEAK_BULL, MarketRegime.STRONG_BULL, MarketRegime.UNKNOWN):
                                    return MarketRegime.RANGE_BEAR
            except Exception:
                pass

            return regime
            
        except Exception as e:
            print(f"[LiveEngine] 市场状态分类失败: {e}")
            return MarketRegime.UNKNOWN
    
    def _update_swing_points(self):
        """
        实时更新摆动点检测（只使用已确认的历史数据）
        
        与上帝视角的区别：
          - 上帝视角在 i 位置可以看 i+window 的数据
          - 实时只能看 i-window 到 i 的数据，所以有 window 个K线的延迟
        
        检测逻辑：
          当前位置 = current_idx
          确认位置 = current_idx - swing_window
          如果确认位置是局部极值（相对于前后各 swing_window 个K线），则标记
        """
        if self._df_buffer is None:
            return
        
        n = len(self._df_buffer)
        window = self._swing_window
        
        # 需要足够的历史数据
        if n < window * 2 + 1:
            return
        
        high = self._df_buffer['high'].values
        low = self._df_buffer['low'].values
        
        # 只检测可以确认的位置（current_idx - window）
        # 因为需要前后各 window 个K线来确认极值
        confirm_idx = n - 1 - window
        if confirm_idx < window:
            return
        
        # 检查这个位置是否已经被检测过
        existing_indices = {s.index for s in self._swing_points}
        if confirm_idx in existing_indices:
            return
        
        # 检测窗口范围
        start = confirm_idx - window
        end = confirm_idx + window + 1  # exclusive
        
        hi = high[confirm_idx]
        lo = low[confirm_idx]
        
        # 检测高点
        if hi >= np.max(high[start:end]):
            self._swing_points.append(SwingPoint(
                index=confirm_idx,
                price=hi,
                is_high=True,
                atr=0.0
            ))
        # 检测低点
        elif lo <= np.min(low[start:end]):
            self._swing_points.append(SwingPoint(
                index=confirm_idx,
                price=lo,
                is_high=False,
                atr=0.0
            ))
        
        # 保持摆动点按时间排序
        self._swing_points.sort(key=lambda s: s.index)
        
        # 记录原始点位
        raw_count = len(self._swing_points)
        
        # 过滤为交替序列（与上帝视角一致）
        self._swing_points = self._filter_alternating_swings(self._swing_points)
        
        if len(self._swing_points) > 0:
             self._throttled_print("swing_points",
                 f"[LiveEngine] 当前摆动点: {len(self._swing_points)} (原始: {raw_count}) | 序列: {[('H' if s.is_high else 'L') + '@' + str(s.index) for s in self._swing_points]}")
    
    def _filter_alternating_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """过滤为严格交替的高低点序列"""
        if not swings:
            return []
        
        alternating = []
        for s in swings:
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
        
        return alternating
    
    @staticmethod
    def _sim_grade(similarity: float) -> str:
        if similarity >= 0.75:
            return "强匹配"
        if similarity >= 0.60:
            return "中匹配"
        return "弱匹配"
    
    def _build_entry_reason(self, direction: str, similarity: float,
                            regime: str, template_fp: str, atr: float) -> str:
        """交易员风格开仓因果说明"""
        grade = self._sim_grade(similarity)
        return (
            f"[开仓逻辑] 市场={regime} | 信号={direction} | "
            f"原型={template_fp} | 相似度={similarity:.2%}({grade}) | "
            f"风控=SL {self.stop_loss_atr:.1f}ATR / TP {self.take_profit_atr:.1f}ATR。"
            f" 匹配强度满足阈值且方向一致，执行{direction}开仓。"
        )
    
    def _build_no_entry_reason(self, regime: str, long_sim: float, short_sim: float,
                                 long_votes: int = 0, short_votes: int = 0,
                                 threshold: float = 0.70, min_agree: int = 1) -> str:
        """交易员风格不开仓因果说明"""
        best_side = "LONG" if long_sim >= short_sim else "SHORT"
        best_sim = max(long_sim, short_sim)
        # 判断失败原因
        reasons = []
        if best_sim < threshold:
            reasons.append(f"相似度{best_sim:.1%}<阈值{threshold:.0%}")
        
        fail_reason = "；".join(reasons) if reasons else "条件未满足"
        
        return (
            f"[观望] 市场={regime} | 最佳={best_side}({best_sim:.1%}) | ❌{fail_reason}"
        )
    
    @staticmethod
    def _build_exit_reason(reason: str, order) -> str:
        """交易员风格平仓因果说明 - 详细版"""
        if order is None:
            return f"[平仓] 触发条件={reason}"
        
        side = order.side.value
        hold = order.hold_bars
        entry = order.entry_price
        pnl_pct = order.profit_pct
        peak_pct = order.peak_profit_pct
        trailing_stage = order.trailing_stage
        sl = order.stop_loss
        tp = order.take_profit
        original_sl = order.original_stop_loss
        
        # 构建详细的决策逻辑说明
        stage_names = {0: "未启动", 1: "保本阶段", 2: "锁利阶段", 3: "紧追阶段"}
        stage_name = stage_names.get(trailing_stage, "未知")
        
        # 判断是否是追踪止损触发（SL已经移动到盈利区）
        sl_moved = False
        if sl and original_sl:
            if side == "LONG" and sl > original_sl:
                sl_moved = True
            elif side == "SHORT" and sl < original_sl:
                sl_moved = True
        
        # 生成决策逻辑
        logic_parts = [
            f"方向={side}",
            f"持仓={hold}根K线",
            f"入场价={entry:.2f}",
            f"当前盈亏={pnl_pct:+.2f}%",
            f"峰值盈利={peak_pct:.2f}%",
            f"追踪阶段={stage_name}",
        ]
        
        if sl_moved:
            logic_parts.append(f"SL已上移({original_sl:.2f}→{sl:.2f})")
        
        logic_str = " | ".join(logic_parts)
        
        return f"[平仓决策] {logic_str} | 触发={reason}"
    
    def _get_tp_sl_trigger_reason(self, order, close_reason: CloseReason, kline) -> str:
        """
        根据触发情况生成详细的平仓原因说明
        """
        if order is None:
            return close_reason.value
        
        entry = order.entry_price
        sl = order.stop_loss
        tp = order.take_profit
        original_sl = order.original_stop_loss
        trailing_stage = order.trailing_stage
        peak_pct = order.peak_profit_pct
        side = order.side.value
        
        # 判断 SL 是否已移动（追踪止损生效）
        sl_moved = False
        if sl and original_sl:
            if side == "LONG" and sl > original_sl:
                sl_moved = True
            elif side == "SHORT" and sl < original_sl:
                sl_moved = True
        
        # 判断 SL 是否在盈利区
        sl_in_profit = False
        if sl:
            if side == "LONG" and sl >= entry:
                sl_in_profit = True
            elif side == "SHORT" and sl <= entry:
                sl_in_profit = True
        
        stage_names = {0: "未启动", 1: "保本", 2: "锁利", 3: "紧追"}
        stage_name = stage_names.get(trailing_stage, "")
        
        if close_reason == CloseReason.TAKE_PROFIT:
            # 细分止盈类型
            if tp and ((side == "LONG" and kline.high >= tp) or (side == "SHORT" and kline.low <= tp)):
                return f"触及止盈价(TP={tp:.2f})"
            elif sl_moved and sl_in_profit:
                return f"追踪止盈({stage_name}阶段, SL={sl:.2f}, 峰值盈利{peak_pct:.1f}%)"
            elif sl_in_profit:
                return f"保本止盈(SL={sl:.2f}已在成本价之上)"
            else:
                return f"止盈(TP={tp:.2f})"
        
        elif close_reason == CloseReason.STOP_LOSS:
            return f"触及止损价(SL={sl:.2f}, 原始SL={original_sl:.2f})"
        
        else:
            return close_reason.value
    
    def get_history_df(self) -> pd.DataFrame:
        """获取历史K线DataFrame（含 MACD/KDJ 等指标，供图表显示）"""
        # 优先返回带指标的 _df_buffer，确保图表能显示 MACD、KDJ
        if self._df_buffer is not None and not self._df_buffer.empty:
            return self._df_buffer.copy()
        # 冷启动：_init_features 尚未完成时，临时计算指标
        df = self._data_feed.get_history_df()
        if df.empty:
            return df
        try:
            from utils.indicators import calculate_all_indicators
            if 'open_time' not in df.columns and 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'open_time'})
            df = calculate_all_indicators(df)
            return df
        except Exception:
            return df
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if hasattr(self._paper_trader, "sync_from_exchange"):
            # 节流同步，避免高频请求
            self._paper_trader.sync_from_exchange(force=False)
        stats = self._paper_trader.stats
        return {
            "initial_balance": stats.initial_balance,
            "current_balance": stats.current_balance,
            "available_margin": getattr(stats, "available_margin", 0.0),
            "total_pnl": stats.total_pnl,
            "total_pnl_pct": stats.total_pnl_pct,
            "total_trades": stats.total_trades,
            "win_rate": stats.win_rate,
            "max_drawdown_pct": stats.max_drawdown_pct,
            "long_trades": stats.long_trades,
            "long_win_rate": stats.long_win_rate,
            "short_trades": stats.short_trades,
            "short_win_rate": stats.short_win_rate,
        }
    
    def get_profitable_templates(self) -> List[str]:
        """获取盈利模板列表"""
        return self._paper_trader.get_profitable_templates()
    
    def get_losing_templates(self) -> List[str]:
        """获取亏损模板列表"""
        return self._paper_trader.get_losing_templates()
    
    def save_results(self, filepath: str):
        """保存交易结果"""
        self._paper_trader.save_history(filepath)


    def _check_staged_partial_tp(self, kline: KlineData):
        """
        分段减仓：根据利润阶梯逐步落袋为安
        
        阶梯:
          - 第1次: 峰值利润 >= 2.0% → 减仓 30%
          - 第2次: 峰值利润 >= 4.0% → 再减仓 30%
          - 剩余 40% 由追踪止损保护，让利润奔跑
        """
        if self._paper_trader is None or not self._paper_trader.has_position():
            return
        
        order = self._paper_trader.current_position
        if order is None:
            return

        # 根据已减仓次数判断下一阈值
        if order.partial_tp_count == 0 and order.peak_profit_pct >= 2.0:
            pct = 0.30  # 第1次：减30%
            label = "1/2"
            threshold = 2.0
        elif order.partial_tp_count == 1 and order.peak_profit_pct >= 4.0:
            pct = 0.30  # 第2次：再减30%（此时总减60%）
            label = "2/2"
            threshold = 4.0
        else:
            return
        
        partial_qty = self._round_to_step(order.quantity * pct)
        if partial_qty <= 0:
            return
        
        closed = self._paper_trader.close_position(
            price=kline.close,
            bar_idx=self._current_bar_idx,
            reason=CloseReason.TAKE_PROFIT,
            quantity=partial_qty
        )
        if closed:
            # 更新 current_position（partial close后对象可能变化）
            remaining = self._paper_trader.current_position
            if remaining is not None:
                remaining.partial_tp_count = order.partial_tp_count + 1
                # 继承追踪状态
                remaining.peak_price = order.peak_price
                remaining.peak_profit_pct = order.peak_profit_pct
                remaining.trailing_stage = order.trailing_stage
            msg = f"阶梯止盈 {label}: 减仓{pct:.0%} @ 峰值利润{order.peak_profit_pct:.1f}%"
            print(f"[LiveEngine] {msg}")
            self.state.last_event = f"✅{msg}"

    def _update_trailing_stop(self, order: PaperOrder, kline: KlineData, atr: float):
        """
        三阶段渐进式追踪止损 + 追踪止盈
        
        █ 阶段0（未激活）: profit < 1.5% → 保持原始止损
        █ 阶段1（保本）:   profit >= 1.5% → SL移至入场价附近（保本）
        █ 阶段2（锁利）:   profit >= 3.0% → SL锁住峰值利润的40%
        █ 阶段3（紧追）:   profit >= 5.0% → SL紧跟峰值利润的60%，追踪TP上移
        
        核心原则：
        - 止损只能往有利方向移动，永不回退
        - 止盈跟随价格上移（多）/下移（空），永不降低
        - 持仓不足3根K线时不启动追踪（让交易有发展空间）
        """
        if atr <= 0:
            return
        
        # 【关键保护】持仓不足3根K线，不启动追踪止损
        # 与信号离场保护一致：前3根K线只靠TP/SL硬保护
        if order.hold_bars < 3:
            return
        
        entry = order.entry_price
        current_sl = order.stop_loss or entry
        current_tp = order.take_profit
        profit_pct = order.profit_pct
        peak_pct = order.peak_profit_pct
        leverage = float(self._paper_trader.leverage)
        
        new_sl = current_sl
        new_tp = current_tp
        new_stage = order.trailing_stage
        
        # ── 阶段判定（阈值已提高，给交易更多发展空间）──
        if peak_pct >= 5.0:
            new_stage = max(order.trailing_stage, 3)
        elif peak_pct >= 3.0:
            new_stage = max(order.trailing_stage, 2)
        elif peak_pct >= 1.5:
            new_stage = max(order.trailing_stage, 1)
        
        # ── 阶段1：保本（杠杆感知）──
        if new_stage >= 1:
            # 根据实际峰值利润和杠杆计算合理的保本缓冲
            # peak_pct 是杠杆化利润，换算为价格百分比：peak_pct / leverage / 100
            peak_price_pct = peak_pct / leverage / 100.0
            # 保本缓冲 = 实际价格移动的 40%，但不超过 0.2%，不低于 0.03%
            breakeven_buffer = min(0.002, peak_price_pct * 0.4)
            breakeven_buffer = max(breakeven_buffer, 0.0003)
            
            if order.side == OrderSide.LONG:
                breakeven_sl = entry * (1 + breakeven_buffer)
                new_sl = max(new_sl, breakeven_sl)
            else:
                breakeven_sl = entry * (1 - breakeven_buffer)
                new_sl = min(new_sl, breakeven_sl)
        
        # ── 阶段2：锁利（锁住峰值利润的40%）──
        if new_stage >= 2:
            lock_pct = peak_pct * 0.40 / 100.0  # 锁住40%的峰值收益
            if order.side == OrderSide.LONG:
                lock_sl = entry * (1 + lock_pct / self._paper_trader.leverage)
                new_sl = max(new_sl, lock_sl)
            else:
                lock_sl = entry * (1 - lock_pct / self._paper_trader.leverage)
                new_sl = min(new_sl, lock_sl)
        
        # ── 阶段3：紧追（锁住峰值利润的60% + 追踪TP上移）──
        if new_stage >= 3:
            lock_pct = peak_pct * 0.60 / 100.0  # 锁住60%的峰值收益
            if order.side == OrderSide.LONG:
                tight_sl = entry * (1 + lock_pct / self._paper_trader.leverage)
                # 额外：ATR紧追（取两者更有利的）
                atr_sl = order.peak_price - atr * 1.2
                tight_sl = max(tight_sl, atr_sl)
                new_sl = max(new_sl, tight_sl)
            else:
                tight_sl = entry * (1 - lock_pct / self._paper_trader.leverage)
                atr_sl = order.peak_price + atr * 1.2
                tight_sl = min(tight_sl, atr_sl)
                new_sl = min(new_sl, tight_sl)
            
            # 追踪止盈：TP跟随价格上移，永不降低
            if current_tp is not None:
                tp_distance = abs(current_tp - entry)
                if order.side == OrderSide.LONG:
                    # 价格每突破旧TP的50%距离，TP上移
                    new_tp_candidate = order.peak_price + tp_distance * 0.3
                    if new_tp_candidate > current_tp:
                        new_tp = new_tp_candidate
                else:
                    new_tp_candidate = order.peak_price - tp_distance * 0.3
                    if new_tp_candidate < current_tp:
                        new_tp = new_tp_candidate
        
        # ── 应用（SL只能往有利方向移动）──
        if order.side == OrderSide.LONG:
            if new_sl > (order.stop_loss or 0):
                order.stop_loss = new_sl
            if new_tp is not None and current_tp is not None and new_tp > current_tp:
                order.take_profit = new_tp
        else:
            if new_sl < (order.stop_loss or float('inf')):
                order.stop_loss = new_sl
            if new_tp is not None and current_tp is not None and new_tp < current_tp:
                order.take_profit = new_tp
        
        # 记录阶段变化
        if new_stage > order.trailing_stage:
            stage_names = {1: "保本", 2: "锁利", 3: "紧追"}
            print(f"[LiveEngine] 追踪止损升级: 阶段{new_stage}({stage_names[new_stage]}) | "
                  f"SL={order.stop_loss:.2f} | TP={order.take_profit:.2f} | "
                  f"峰值利润={peak_pct:.1f}%")
            order.trailing_stage = new_stage

    def _check_exit_indicator_gate(self, df: pd.DataFrame, direction: str) -> dict:
        """
        离场指标确认门槛 (MACD + KDJ 共振)
        只有当指标也显示反向动能时，才允许基于形态的离场
        
        Returns:
            {"passed": bool, "reason": str, "details": dict}
        """
        if df is None or len(df) < 3:
            return {"passed": True, "reason": "指标数据不足，默认通过", "details": {}}
            
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        details = {
            "macd_prev": prev['macd_hist'],
            "macd_curr": curr['macd_hist'],
            "kdj_prev": prev['j'],
            "kdj_curr": curr['j'],
        }
        
        if direction == "LONG":
            # 1. MACD 柱状图在收缩或转负
            macd_exit = curr['macd_hist'] < prev['macd_hist'] or curr['macd_hist'] < 0
            # 2. KDJ J线不再创新高（已经掉头或走平）
            kdj_exit = curr['j'] < prev['j']
            passed = macd_exit and kdj_exit
            
            details["macd_status"] = "收缩/转负✓" if macd_exit else "仍扩张✗"
            details["kdj_status"] = "掉头✓" if kdj_exit else "仍上行✗"
        else:
            # 1. MACD 柱状图在回升或转正
            macd_exit = curr['macd_hist'] > prev['macd_hist'] or curr['macd_hist'] > 0
            # 2. KDJ J线不再创新低（已经拉升或走平）
            kdj_exit = curr['j'] > prev['j']
            passed = macd_exit and kdj_exit
            
            details["macd_status"] = "回升/转正✓" if macd_exit else "仍下行✗"
            details["kdj_status"] = "拉升✓" if kdj_exit else "仍下行✗"
        
        reason = f"MACD {details['macd_status']}, KDJ {details['kdj_status']}"
        return {"passed": passed, "reason": reason, "details": details}

    def _check_market_reversal(self, order: PaperOrder) -> dict:
        """
        检查市场因素是否一致反转（市场状态 + MACD + KDJ）
        
        做多持仓：如果市场变空 + MACD转空 + KDJ转空 → 建议离场
        做空持仓：如果市场变多 + MACD转多 + KDJ转多 → 建议离场
        
        Returns:
            {"should_exit": bool, "reason": str, "details": dict}
        """
        result = {"should_exit": False, "reason": "", "details": {}}
        
        if self._df_buffer is None or len(self._df_buffer) < 3:
            return result
        
        direction = order.side.value  # "LONG" or "SHORT"
        curr = self._df_buffer.iloc[-1]
        prev = self._df_buffer.iloc[-2]
        
        # 1. 市场状态检查
        current_regime = self.state.market_regime
        bull_regimes = {MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL, MarketRegime.RANGE_BULL}
        bear_regimes = {MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.RANGE_BEAR}
        
        regime_bullish = current_regime in bull_regimes
        regime_bearish = current_regime in bear_regimes
        
        # 2. MACD 趋势检查（连续2根K线确认）
        macd_bullish = curr['macd_hist'] > 0 and curr['macd_hist'] > prev['macd_hist']
        macd_bearish = curr['macd_hist'] < 0 and curr['macd_hist'] < prev['macd_hist']
        
        # 3. KDJ 趋势检查（J线方向）
        kdj_bullish = curr['j'] > prev['j'] and curr['j'] > 50
        kdj_bearish = curr['j'] < prev['j'] and curr['j'] < 50
        
        result["details"] = {
            "regime": str(current_regime),
            "regime_bullish": regime_bullish,
            "regime_bearish": regime_bearish,
            "macd_bullish": macd_bullish,
            "macd_bearish": macd_bearish,
            "kdj_bullish": kdj_bullish,
            "kdj_bearish": kdj_bearish,
        }
        
        # 判断是否一致反转
        if direction == "LONG":
            # 做多时，三者一致转空 → 离场
            if regime_bearish and macd_bearish and kdj_bearish:
                result["should_exit"] = True
                result["reason"] = f"市场={current_regime} + MACD转空 + KDJ转空"
        else:  # SHORT
            # 做空时，三者一致转多 → 离场
            if regime_bullish and macd_bullish and kdj_bullish:
                result["should_exit"] = True
                result["reason"] = f"市场={current_regime} + MACD转多 + KDJ转多"
        
        return result

    def _round_to_step(self, qty: float) -> float:
        """按交易所最小步进对齐数量"""
        step = getattr(self._paper_trader, '_qty_step', 0.001)
        return max(step, (qty // step) * step)

    def _risk_limit_reached(self) -> bool:
        """检查风控阈值是否触发"""
        if self.max_drawdown_pct is None:
            return False
        stats = getattr(self._paper_trader, "stats", None)
        if not stats:
            return False
        return stats.max_drawdown_pct >= self.max_drawdown_pct

# 简单测试
if __name__ == "__main__":
    print("LiveTradingEngine 测试需要 TrajectoryMemory，请在完整环境中运行")
