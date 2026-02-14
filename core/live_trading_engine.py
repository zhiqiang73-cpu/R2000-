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
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_data_feed import LiveDataFeed, KlineData
from config import PAPER_TRADING_CONFIG
from core.paper_trader import PaperOrder, OrderSide, CloseReason
from core.binance_testnet_trader import BinanceTestnetTrader
from core.market_regime import MarketRegimeClassifier, MarketRegime
from core.labeler import SwingPoint
from core.bayesian_filter import BayesianTradeFilter
from core.exit_signal_learner import ExitSignalLearner


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
    bayesian_win_rate: float = 0.0    # 贝叶斯预测胜率（0-1）
    kelly_position_pct: float = 0.0   # 凯利公式动态仓位（0-1）
    


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
                 on_price_tick: Optional[Callable[[float, int], None]] = None,
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
        self.on_price_tick = on_price_tick
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
            on_price=self._on_price_received,
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
        
        # 【新增】市场状态确认期：连续3根K线保持同向才切换状态，避免震荡市频繁切换
        self._regime_history: List[str] = []  # 存储最近3根的市场状态原始判断
        self._confirmed_regime: Optional[str] = None  # 确认后的稳定市场状态
        self._last_raw_regime: Optional[str] = None  # 最近一次原始（未确认）市场状态，供反转检测使用
        
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
        
        # ── 反手单状态 ──
        self._reverse_pending = False         # 是否有待执行的反手信号
        self._reverse_direction: Optional[str] = None  # 反手方向 "LONG"/"SHORT"
        self._reverse_price: float = 0.0      # 反手参考价（被止损的价格）
        self._reverse_count: int = 0          # 当前连续反手次数
        self._last_stoploss_side: Optional[str] = None   # 上次止损方向
        self._last_stoploss_time: float = 0.0             # 上次止损时间戳
        
        # ── 价格位置翻转状态 ──
        self._flip_pending = False            # 是否有待执行的翻转信号
        self._flip_direction: Optional[str] = None  # 翻转方向
        self._flip_price_position: float = 0.0       # 翻转时的价格位置
        self._flip_proto_fp: str = ""                # 翻转匹配的原型指纹
        self._flip_similarity: float = 0.0           # 翻转匹配相似度
        self._flip_proto = None                      # 翻转匹配的原型对象
        self._flip_template = None                   # 翻转匹配的模板对象
        self._pending_flip_mark = False       # 下一个开仓订单标记为翻转单
        
        # ── 贝叶斯交易过滤器 ──
        self._bayesian_filter: Optional[BayesianTradeFilter] = None
        self._bayesian_enabled = PAPER_TRADING_CONFIG.get("BAYESIAN_ENABLED", False)
        if self._bayesian_enabled:
            self._bayesian_filter = BayesianTradeFilter(
                prior_strength=PAPER_TRADING_CONFIG.get("BAYESIAN_PRIOR_STRENGTH", 10.0),
                min_win_rate_threshold=PAPER_TRADING_CONFIG.get("BAYESIAN_MIN_WIN_RATE", 0.50),
                thompson_sampling=PAPER_TRADING_CONFIG.get("BAYESIAN_THOMPSON_SAMPLING", True),
                decay_enabled=PAPER_TRADING_CONFIG.get("BAYESIAN_DECAY_ENABLED", True),
                decay_interval_hours=PAPER_TRADING_CONFIG.get("BAYESIAN_DECAY_HOURS", 24.0),
                decay_factor=PAPER_TRADING_CONFIG.get("BAYESIAN_DECAY_FACTOR", 0.95),
                persistence_path=PAPER_TRADING_CONFIG.get("BAYESIAN_STATE_FILE", "data/bayesian_state.json"),
            )
        
        # ── 离场信号学习器 ──
        self._exit_learner: Optional[ExitSignalLearner] = None
        self._exit_learning_enabled = PAPER_TRADING_CONFIG.get("EXIT_LEARNING_ENABLED", False)
        if self._exit_learning_enabled:
            self._exit_learner = ExitSignalLearner(
                persistence_path=PAPER_TRADING_CONFIG.get("EXIT_LEARNING_STATE_FILE", "data/exit_learning_state.json"),
                decay_enabled=PAPER_TRADING_CONFIG.get("EXIT_LEARNING_DECAY_ENABLED", True),
                decay_interval_hours=PAPER_TRADING_CONFIG.get("EXIT_LEARNING_DECAY_HOURS", 48.0),
                decay_factor=PAPER_TRADING_CONFIG.get("EXIT_LEARNING_DECAY_FACTOR", 0.95),
            )
            print(f"[LiveEngine] 贝叶斯过滤器已启用: Thompson={self._bayesian_filter.thompson_sampling}, "
                  f"最低胜率={self._bayesian_filter.min_win_rate_threshold:.0%}")
    
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
            
            # 【贝叶斯先验初始化】用原型库的历史回测胜率初始化 Beta 分布
            if self._bayesian_enabled and self._bayesian_filter and self.use_prototypes:
                if self._active_prototype_library:
                    proto_count = 0
                    for proto in self._active_prototype_library.long_prototypes + self._active_prototype_library.short_prototypes:
                        fp = f"proto_{proto.direction}_{proto.prototype_id}"
                        regime = proto.regime if proto.regime else "未知"
                        if proto.member_count >= 3:  # 至少 3 个样本才可靠
                            self._bayesian_filter.initialize_from_prototype(
                                prototype_fingerprint=fp,
                                market_regime=regime,
                                historical_win_rate=proto.win_rate,
                                historical_sample_count=proto.member_count,
                                historical_avg_profit_pct=proto.avg_profit_pct,
                            )
                            proto_count += 1
                    print(f"[LiveEngine] 贝叶斯先验初始化完成: {proto_count} 个原型×市场状态组合")
            
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
                # 实时 tick：检查 TP/SL 硬保护 + 更新界面
                # 信号离场/动量衰减等复杂决策仍在 K 线收线时处理
                if self._paper_trader.has_position():
                    order = self._paper_trader.current_position
                    if order is not None:
                        # 【关键修复】实时检查 TP/SL，防止价格快速穿透
                        # 使用 tick 的实时价格作为 high/low
                        close_reason = self._paper_trader.update_price(
                            kline.close,
                            high=kline.high,  # tick 期间的最高价
                            low=kline.low,    # tick 期间的最低价
                            bar_idx=self._current_bar_idx,
                            protection_mode=False  # tick 检查不启用保护期
                        )
                        if close_reason:
                            # TP/SL 在 tick 期间触发，重置状态
                            print(f"[LiveEngine] ⚡ Tick触发{close_reason.value} @ {kline.close:.2f}")
                            self.state.last_event = f"⚡ Tick触发{close_reason.value}"
                            self._reset_position_state()
                else:
                    # 未持仓时，tick级仅做预匹配（UI预览），不做入场决策
                    # 实盘/测试网入场决策统一在K线收线时执行，避免：
                    # 1. tick级_process_entry总被"实盘仅在收线决策"拦截，产生无意义日志
                    # 2. 这些"入场跳过"消息会覆盖真正的收线入场决策结果，导致用户看不到
                    is_exchange_mode = hasattr(self._paper_trader, "sync_from_exchange")
                    if is_exchange_mode:
                        # 交易所模式：tick级只做预匹配更新UI，不调_process_entry
                        self._preview_match(kline)
                    else:
                        # 模拟模式：保留原有的实时入场逻辑
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

    def _on_price_received(self, price: float, ts_ms: int):
        """逐笔成交价回调（低延迟显示，不触发策略决策）"""
        if not self._running:
            return
        with self._lock:
            self.state.current_price = float(price)
            try:
                self.state.current_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone(timedelta(hours=8)))
            except Exception:
                pass
        if self.on_price_tick:
            try:
                self.on_price_tick(float(price), int(ts_ms))
            except Exception as e:
                print(f"[LiveEngine] on_price_tick 回调异常: {e}")
    
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
        has_pending = self._paper_trader.has_pending_stop_orders(current_bar_idx=self._current_bar_idx)
        # 检测新开仓：上一根无持仓，本根有持仓 → 触发开仓回调（含止损单成交、手动开仓等）
        if has_pos and not self._last_had_position and self.on_trade_opened:
            self._ensure_position_tp_sl()
            order = self._paper_trader.current_position
            if order:
                # 【新增】记录入场时的 ATR（用于离场信号学习）
                if getattr(order, 'entry_atr', 0) == 0:
                    order.entry_atr = self._get_current_atr()
                    print(f"[LiveEngine] 新开仓：已记录 ATR={order.entry_atr:.2f}")
                
                # 【翻转单标记】如果这个开仓是由位置翻转触发的
                if self._pending_flip_mark:
                    order.is_flip_trade = True
                    pos_label = "底部翻转做多" if order.side == OrderSide.LONG else "顶部翻转做空"
                    order.flip_reason = pos_label
                    self._pending_flip_mark = False
                    print(f"[LiveEngine] 🔄 已标记翻转单: {pos_label}")
                
                try:
                    self.on_trade_opened(order)
                except Exception as e:
                    print(f"[LiveEngine] 开仓回调异常: {e}")
        self._last_had_position = has_pos
        
        print(f"[LiveEngine] K线收线: {kline.open_time} | 价格={kline.close:.2f} | 持仓={has_pos} | 挂单={has_pending}")
        if (not has_pos) and has_pending:
            self.state.last_event = (
                f"K线收线 {kline.open_time.strftime('%H:%M')} | ${kline.close:,.2f} | 挂单等待成交"
            )
        else:
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
            # 入场优先级：翻转单 > 反手单 > 正常入场
            if self._flip_pending and self._flip_direction:
                # 翻转单最高优先级（持仓在底部/顶部主动翻转）
                self._execute_flip_entry(kline, atr)
            elif self._reverse_pending and self._reverse_direction:
                # 反手单次优先级（止损后自动反手）
                self._execute_reverse_entry(kline, atr)
            else:
                print(f"[LiveEngine] 📌 收线入场决策: 价格={kline.close:.2f} | ATR={atr:.2f} | 市场={self.state.market_regime}")
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
            tp_changed = order.take_profit is None
            sl_changed = order.stop_loss is None
            if order.take_profit is None:
                order.take_profit = tp
            if order.stop_loss is None:
                order.stop_loss = sl
                if getattr(order, "original_stop_loss", None) is None:
                    order.original_stop_loss = sl
            self.state.last_event = (
                f"[风控补全] TP/SL已补全 | TP={order.take_profit:.2f} SL={order.stop_loss:.2f}"
            )
            
            # 【核心】TP/SL补全后，同步到交易所保护单
            if (tp_changed or sl_changed) and hasattr(self._paper_trader, '_place_exchange_tp_sl'):
                print(f"[LiveEngine] TP/SL补全完毕，同步到交易所保护单...")
                self._paper_trader._place_exchange_tp_sl(order)
        except Exception as e:
            print(f"[LiveEngine] 补全TP/SL失败: {e}")
    
    def _execute_reverse_entry(self, kline: KlineData, atr: float):
        """
        反手单入场：止损后自动反方向开仓
        
        逻辑：
        - 止损说明市场方向判断错了，顺势反手
        - 使用当前价格作为入场参考，计算新 TP/SL
        - 跳过冷却时间和指标门控（止损反手是确定性信号）
        - 反手单入场后，如果再次止损则不再继续反手（由 max_count 控制）
        """
        direction = self._reverse_direction
        price = kline.close
        side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
        
        # 清除反手待执行状态
        self._reverse_pending = False
        self._reverse_direction = None
        
        # 【regime-direction 一致性检查】反手方向必须与市场状态一致
        current_regime = self._confirm_market_regime()
        BULL_REGIMES_REV = {"强多头", "弱多头", "震荡偏多"}
        BEAR_REGIMES_REV = {"强空头", "弱空头", "震荡偏空"}
        if direction == "SHORT" and current_regime in BULL_REGIMES_REV:
            print(f"[LiveEngine] ⚠ 反手SHORT被拦截: 市场={current_regime}(偏多)，不允许做空")
            self.state.last_event = f"[反手取消] 市场{current_regime}不允许做空"
            return
        if direction == "LONG" and current_regime in BEAR_REGIMES_REV:
            print(f"[LiveEngine] ⚠ 反手LONG被拦截: 市场={current_regime}(偏空)，不允许做多")
            self.state.last_event = f"[反手取消] 市场{current_regime}不允许做多"
            return
        
        print(f"[LiveEngine] 🔄 执行反手单: {direction} @ {price:.2f} | 市场={current_regime}")
        
        # 计算限价入场价
        from config import VECTOR_SPACE_CONFIG
        confirm_pct = VECTOR_SPACE_CONFIG.get("ENTRY_CONFIRM_PCT", 0.001)
        timeout = VECTOR_SPACE_CONFIG.get("TRIGGER_TIMEOUT_BARS", 5)
        limit_price = price * (1 + confirm_pct) if side == OrderSide.LONG else price * (1 - confirm_pct)
        
        # 计算 TP/SL（基于实际入场价）
        take_profit, stop_loss = self._calculate_dynamic_tp_sl(
            entry_price=limit_price,
            direction=direction,
            prototype=None,  # 反手单不依赖原型
            atr=atr
        )
        
        tp_pct = abs(take_profit - limit_price) / limit_price * 100
        sl_pct = abs(stop_loss - limit_price) / limit_price * 100
        
        reason = (
            f"[反手单] 止损反手 | {direction} | "
            f"TP={take_profit:.2f}(+{tp_pct:.1f}%) SL={stop_loss:.2f}(-{sl_pct:.1f}%)"
        )
        
        # 下单（不检查指标门控，反手是确定性信号）
        order_id = self._paper_trader.place_stop_order(
            side=side,
            trigger_price=limit_price,
            bar_idx=self._current_bar_idx,
            take_profit=take_profit,
            stop_loss=stop_loss,
            template_fingerprint="REVERSE",
            entry_similarity=0.0,
            entry_reason=reason,
            timeout_bars=timeout
        )
        
        if order_id:
            print(f"[LiveEngine] 🔄 反手限价单已挂: {direction} @ {limit_price:.2f} "
                  f"(当前={price:.2f}, TP={take_profit:.2f}, SL={stop_loss:.2f})")
            self.state.last_event = f"[反手单] {direction} @ {limit_price:.2f}"
            self.state.decision_reason = reason
            self.state.matching_phase = "反手入场"
        else:
            print(f"[LiveEngine] ⚠ 反手单下单失败")
            self.state.last_event = "[反手单] 下单失败"
    
    def _execute_flip_entry(self, kline: KlineData, atr: float):
        """
        价格位置翻转入场：持仓触底/触顶后的智能反手开仓
        
        与止损反手的区别：
        - 止损反手：被动，价格已经到止损才触发
        - 位置翻转：主动，在有利位置（底部/顶部）主动翻转
        - 翻转单标记为 is_flip_trade=True，贝叶斯会给予更高学习权重
        """
        direction = self._flip_direction
        price = kline.close
        side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
        pos_pct = self._flip_price_position
        flip_fp = self._flip_proto_fp
        flip_sim = self._flip_similarity
        flip_proto = self._flip_proto
        flip_template = self._flip_template
        
        # 清除翻转待执行状态
        self._flip_pending = False
        self._flip_direction = None
        
        # 设置当前原型/模板（用于后续TP/SL计算）
        if flip_proto:
            self._current_prototype = flip_proto
            self._current_template = None
        elif flip_template:
            self._current_template = flip_template
            self._current_prototype = None
        
        pos_label = "底部" if direction == "LONG" else "顶部"
        print(f"[LiveEngine] 🔄🔄 执行翻转入场: {direction} @ {price:.2f} | "
              f"价格在区间{pos_label}({pos_pct:.0%}) | 原型={flip_fp}({flip_sim:.1%})")
        
        # 计算限价入场价
        from config import VECTOR_SPACE_CONFIG
        confirm_pct = VECTOR_SPACE_CONFIG.get("ENTRY_CONFIRM_PCT", 0.001)
        timeout = VECTOR_SPACE_CONFIG.get("TRIGGER_TIMEOUT_BARS", 5)
        limit_price = price * (1 + confirm_pct) if side == OrderSide.LONG else price * (1 - confirm_pct)
        
        # 计算 TP/SL
        take_profit, stop_loss = self._calculate_dynamic_tp_sl(
            entry_price=limit_price,
            direction=direction,
            prototype=flip_proto,
            atr=atr
        )
        
        tp_pct = abs(take_profit - limit_price) / limit_price * 100
        sl_pct = abs(stop_loss - limit_price) / limit_price * 100
        
        reason = (
            f"[翻转单] 价格{pos_label}({pos_pct:.0%})翻转 | {direction} | "
            f"原型={flip_fp}({flip_sim:.1%}) | "
            f"TP={take_profit:.2f}(+{tp_pct:.1f}%) SL={stop_loss:.2f}(-{sl_pct:.1f}%)"
        )
        
        # 下单（翻转单跳过MACD门控 — 已在检测时确认MACD支持）
        order_id = self._paper_trader.place_stop_order(
            side=side,
            trigger_price=limit_price,
            bar_idx=self._current_bar_idx,
            take_profit=take_profit,
            stop_loss=stop_loss,
            template_fingerprint=flip_fp or "FLIP",
            entry_similarity=flip_sim,
            entry_reason=reason,
            timeout_bars=timeout
        )
        
        if order_id:
            # 标记这是一个翻转单（通过回调在开仓时设置）
            self._pending_flip_mark = True
            
            print(f"[LiveEngine] 🔄 翻转限价单已挂: {direction} @ {limit_price:.2f} "
                  f"(当前={price:.2f}, TP={take_profit:.2f}, SL={stop_loss:.2f})")
            self.state.last_event = f"🔄 [翻转单] {direction} @ {limit_price:.2f} | {pos_label}({pos_pct:.0%})"
            self.state.decision_reason = reason
            self.state.matching_phase = "翻转入场"
            
            # 重置止损反手计数（翻转不计入连续止损反手）
            self._reverse_count = 0
            self._reverse_pending = False
        else:
            print(f"[LiveEngine] ⚠ 翻转单下单失败")
            self.state.last_event = "[翻转单] 下单失败"
    
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
            self.state.market_regime = self._confirm_market_regime()
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
                
                # 【regime-direction 一致性】只匹配与市场方向一致的原型
                BULL_REGIMES_ENTRY = {"强多头", "弱多头", "震荡偏多"}
                BEAR_REGIMES_ENTRY = {"强空头", "弱空头", "震荡偏空"}

                chosen_proto = None
                if match_regime in BULL_REGIMES_ENTRY:
                    # 偏多市场：只匹配 LONG
                    long_result = self._proto_matcher.match_entry(
                        pre_entry_traj, direction="LONG", regime=match_regime
                    )
                    long_sim = long_result.get("similarity", 0.0)
                    short_sim = 0.0
                    if long_result.get("matched"):
                        direction, chosen_proto, similarity = "LONG", long_result.get("best_prototype"), long_sim
                elif match_regime in BEAR_REGIMES_ENTRY:
                    # 偏空市场：只匹配 SHORT
                    short_result = self._proto_matcher.match_entry(
                        pre_entry_traj, direction="SHORT", regime=match_regime
                    )
                    long_sim = 0.0
                    short_sim = short_result.get("similarity", 0.0)
                    if short_result.get("matched"):
                        direction, chosen_proto, similarity = "SHORT", short_result.get("best_prototype"), short_sim
                else:
                    # 未知状态：双向匹配
                    long_result = self._proto_matcher.match_entry(
                        pre_entry_traj, direction="LONG", regime=match_regime
                    )
                    short_result = self._proto_matcher.match_entry(
                        pre_entry_traj, direction="SHORT", regime=match_regime
                    )
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

                self._throttled_print("proto_match",
                    f"[LiveEngine] 原型匹配结果: "
                    f"市场={match_regime} | LONG={long_sim:.1%} | SHORT={short_sim:.1%}")
                
                if direction is not None and chosen_proto is not None:
                    # 【修复】构建原型指纹时添加防御性检查，确保所有字段都存在
                    proto_direction = getattr(chosen_proto, 'direction', None) or "UNKNOWN"
                    proto_id = getattr(chosen_proto, 'prototype_id', None)
                    proto_regime = getattr(chosen_proto, 'regime', None) or ""
                    regime_short = proto_regime[:2] if proto_regime else "未知"
                    
                    # 构建完整指纹
                    chosen_fp = f"proto_{proto_direction}_{proto_id}_{regime_short}"
                    
                    # 防御性检查：如果有字段缺失，输出警告
                    if proto_id is None or proto_direction == "UNKNOWN" or regime_short == "未知":
                        print(f"[警告] 原型指纹构建不完整: {chosen_fp}")
                        print(f"  ├─ direction: {proto_direction} (type: {type(chosen_proto.direction) if hasattr(chosen_proto, 'direction') else 'N/A'})")
                        print(f"  ├─ prototype_id: {proto_id} (type: {type(chosen_proto.prototype_id) if hasattr(chosen_proto, 'prototype_id') else 'N/A'})")
                        print(f"  ├─ regime: {proto_regime} (type: {type(chosen_proto.regime) if hasattr(chosen_proto, 'regime') else 'N/A'})")
                        print(f"  └─ chosen_proto type: {type(chosen_proto)}")
                    
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
                # ── 反手保护：止损后禁止同方向再入场 ──
                from config import PAPER_TRADING_CONFIG as _ptc_entry
                block_same_sec = _ptc_entry.get("REVERSE_BLOCK_SAME_DIR_SEC", 300)
                if (self._last_stoploss_side == direction
                        and block_same_sec > 0
                        and (time.time() - self._last_stoploss_time) < block_same_sec):
                    remaining = block_same_sec - (time.time() - self._last_stoploss_time)
                    self.state.last_event = (
                        f"[入场跳过] {direction}方向刚止损，禁止同向入场(剩余{remaining:.0f}s)"
                    )
                    self.state.decision_reason = (
                        f"[同向封锁] 刚在{self._last_stoploss_side}方向止损，"
                        f"{block_same_sec}秒内禁止同向再入场，避免反复被扫"
                    )
                    return
                
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
                
                # 【三重确认逻辑】
                from config import VECTOR_SPACE_CONFIG
                confirm_pct = VECTOR_SPACE_CONFIG.get("ENTRY_CONFIRM_PCT", 0.001)
                timeout = VECTOR_SPACE_CONFIG.get("TRIGGER_TIMEOUT_BARS", 5)
                
                # 0. 【核心改进】价格位置智能翻转 — 震荡市底部做多/顶部做空
                #    问题根因：指纹匹配看到"价格在跌"→匹配SHORT原型→在最低点做空
                #    修复策略：
                #      震荡市：检测到价格在区间底部做空 → 翻转为做多（反向匹配）
                #      震荡市：检测到价格在区间顶部做多 → 翻转为做空（反向匹配）
                #      趋势市：只做警告（趋势中追势是合理的，但极端位置仍需谨慎）
                flip_triggered = False
                if self._df_buffer is not None and len(self._df_buffer) >= 20:
                    lookback = min(30, len(self._df_buffer))
                    recent = self._df_buffer.tail(lookback)
                    range_high = recent['high'].max()
                    range_low = recent['low'].min()
                    range_size = range_high - range_low
                    
                    if range_size > 0:
                        # 当前价格在区间中的位置 (0=底部, 1=顶部)
                        price_position = (price - range_low) / range_size
                        
                        regime = self.state.market_regime
                        is_range_market = "震荡" in regime if regime else False
                        
                        # 震荡市：底部30%做空→翻转做多，顶部30%做多→翻转做空
                        # 趋势市：底部15%做空→仅拒绝，顶部15%做多→仅拒绝
                        danger_zone = 0.30 if is_range_market else 0.15
                        
                        need_flip = False
                        flip_direction = None
                        
                        if direction == "SHORT" and price_position < danger_zone:
                            need_flip = True
                            flip_direction = "LONG"
                            print(f"[LiveEngine] 🔄 价格位置翻转: SHORT→LONG | "
                                  f"价格={price:.2f} 在区间底部({price_position:.0%}) | "
                                  f"区间={range_low:.0f}-{range_high:.0f} | 市场={regime}")
                        
                        elif direction == "LONG" and price_position > (1 - danger_zone):
                            need_flip = True
                            flip_direction = "SHORT"
                            print(f"[LiveEngine] 🔄 价格位置翻转: LONG→SHORT | "
                                  f"价格={price:.2f} 在区间顶部({price_position:.0%}) | "
                                  f"区间={range_low:.0f}-{range_high:.0f} | 市场={regime}")
                        
                        if need_flip and flip_direction:
                            if is_range_market:
                                # 【震荡市智能翻转】尝试用反方向重新匹配原型
                                # pre_entry_traj 在本次 _process_entry 调用中已构建好
                                flip_result = None
                                flip_matched = False
                                
                                # 优先用原型匹配器
                                if hasattr(self, '_proto_matcher') and self._proto_matcher:
                                    flip_result = self._proto_matcher.match_entry(
                                        pre_entry_traj,
                                        direction=flip_direction,
                                        regime=self.state.market_regime
                                    )
                                    flip_matched = flip_result and flip_result.get("matched")
                                
                                # 原型匹配失败时，尝试模板匹配器
                                if not flip_matched and hasattr(self, '_matcher') and self._matcher:
                                    from core.trajectory_memory import TrajectoryMemory
                                    if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
                                        flip_candidates = self.trajectory_memory.get_templates_by_direction(flip_direction)
                                        flip_tmpl_result = self._matcher.match_entry(
                                            pre_entry_traj,
                                            flip_candidates,
                                            cosine_threshold=self.cosine_threshold,
                                            dtw_threshold=self.dtw_threshold,
                                        )
                                        if flip_tmpl_result.matched and flip_tmpl_result.best_template:
                                            # 将模板结果转换为统一格式
                                            flip_result = {
                                                "matched": True,
                                                "best_prototype": None,
                                                "best_template": flip_tmpl_result.best_template,
                                                "similarity": flip_tmpl_result.dtw_similarity,
                                            }
                                            flip_matched = True
                                
                                if flip_matched and flip_result:
                                    # 翻转成功：找到反方向的匹配
                                    flip_proto = flip_result.get("best_prototype")
                                    flip_template = flip_result.get("best_template")
                                    flip_sim = flip_result.get("similarity", 0.0)
                                    
                                    # 更新方向
                                    direction = flip_direction
                                    similarity = flip_sim
                                    side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
                                    flip_triggered = True
                                    
                                    if flip_proto:
                                        # 原型匹配成功
                                        chosen_proto = flip_proto
                                        self._current_prototype = flip_proto
                                        self._current_template = None
                                        
                                        proto_direction = getattr(flip_proto, 'direction', None) or "UNKNOWN"
                                        proto_id = getattr(flip_proto, 'prototype_id', None)
                                        proto_regime = getattr(flip_proto, 'regime', None) or ""
                                        regime_short = proto_regime[:2] if proto_regime else "未知"
                                        chosen_fp = f"proto_{proto_direction}_{proto_id}_{regime_short}"
                                    elif flip_template:
                                        # 模板匹配成功
                                        self._current_template = flip_template
                                        self._current_prototype = None
                                        chosen_fp = flip_template.fingerprint()
                                    
                                    self.state.last_event = (
                                        f"🔄 位置翻转: {flip_direction} | {chosen_fp} | {flip_sim:.1%} | "
                                        f"价格在区间{price_position:.0%}位置"
                                    )
                                    self.state.decision_reason = (
                                        f"[智能翻转] 原始信号{('SHORT' if flip_direction == 'LONG' else 'LONG')}"
                                        f"在区间{'底部' if flip_direction == 'LONG' else '顶部'}"
                                        f"({price_position:.0%})危险，"
                                        f"翻转为{flip_direction}，匹配={chosen_fp}({flip_sim:.1%})"
                                    )
                                    print(f"[LiveEngine] ✅ 翻转匹配成功: {flip_direction} | "
                                          f"{chosen_fp} | {flip_sim:.1%}")
                                else:
                                    # 翻转失败：反方向没有匹配原型，拒绝原方向
                                    orig_dir = direction
                                    self.state.last_event = (
                                        f"[入场拒绝] {orig_dir}在区间{'底部' if orig_dir == 'SHORT' else '顶部'}"
                                        f"({price_position:.0%})，翻转{flip_direction}无匹配原型"
                                    )
                                    self.state.decision_reason = (
                                        f"[位置过滤] {orig_dir}在区间{'底部' if orig_dir == 'SHORT' else '顶部'}"
                                        f"({price_position:.0%})危险，"
                                        f"尝试翻转{flip_direction}但无匹配原型。放弃入场。"
                                        f"(区间: {range_low:.0f}-{range_high:.0f})"
                                    )
                                    print(f"[LiveEngine] ⛔ 翻转失败: {flip_direction}无匹配原型，放弃入场")
                                    self.state.best_match_similarity = similarity
                                    self.state.best_match_template = chosen_fp
                                    return
                            else:
                                # 【趋势市】不翻转，只拒绝极端位置入场
                                pos_label = "底部" if direction == "SHORT" else "顶部"
                                self.state.last_event = (
                                    f"[入场拒绝] 价格在区间{pos_label}({price_position:.0%})，"
                                    f"趋势市{direction}谨慎"
                                )
                                self.state.decision_reason = (
                                    f"[位置过滤] 趋势市中价格处于近{lookback}根K线区间"
                                    f"{pos_label}({price_position:.0%}位置)，"
                                    f"{direction}风险过高。"
                                    f"(区间: {range_low:.0f}-{range_high:.0f})"
                                )
                                print(f"[LiveEngine] ⛔ 趋势市位置过滤: {direction}被拒 | "
                                      f"价格={price:.2f} 在{pos_label}({price_position:.0%})")
                                self.state.best_match_similarity = similarity
                                self.state.best_match_template = chosen_fp
                                return
                
                # A. 检查指标闸门 (Aim 瞄准) — MACD必须通过，KDJ仅参考
                if not self._check_indicator_gate(self._df_buffer, direction):
                    if has_pending: 
                        # 如果MACD不再满足，撤掉之前的单子
                        self._paper_trader.cancel_entry_stop_orders()
                    kdj_hint = "✓" if self.state.kdj_ready else "✗"
                    self.state.decision_reason = (
                        f"[等待MACD] 指纹匹配成功({similarity:.1%}), 但 MACD 动能未对齐。"
                        f"(MACD={self.state.macd_ready}, KDJ={kdj_hint}参考)"
                    )
                    self.state.last_event = f"[门控] MACD未通过 | KDJ={kdj_hint}(参考)"
                    self.state.best_match_similarity = similarity
                    self.state.best_match_template = chosen_fp
                    return
                
                # B. 贝叶斯门控（基于实盘学习的胜率预测）
                if self._bayesian_enabled and self._bayesian_filter:
                    should_trade, predicted_wr, bay_reason = self._bayesian_filter.should_trade(
                        prototype_fingerprint=chosen_fp,
                        market_regime=self.state.market_regime,
                    )
                    if not should_trade:
                        self.state.last_event = f"[贝叶斯拒绝] {bay_reason}"
                        self.state.decision_reason = (
                            f"[贝叶斯过滤] 原型={chosen_fp} 市场={self.state.market_regime} | {bay_reason}"
                        )
                        self.state.best_match_similarity = similarity
                        self.state.best_match_template = chosen_fp
                        print(f"[LiveEngine] ⛔ 贝叶斯拒绝: {chosen_fp} | {bay_reason}")
                        return
                    else:
                        # 更新 state 中的贝叶斯胜率
                        self.state.bayesian_win_rate = predicted_wr
                        print(f"[LiveEngine] ✅ 贝叶斯通过: {chosen_fp} | {bay_reason}")
                
                # B2. 凯利仓位计算（根据贝叶斯预测的胜率和盈亏比）
                kelly_position_pct = None  # None = 使用默认仓位
                kelly_reason = ""
                from config import PAPER_TRADING_CONFIG as _ptc_kelly
                kelly_enabled = _ptc_kelly.get("KELLY_ENABLED", False)
                if kelly_enabled and self._bayesian_filter:
                    kelly_fraction = _ptc_kelly.get("KELLY_FRACTION", 0.25)
                    kelly_max = _ptc_kelly.get("KELLY_MAX_POSITION", 0.5)
                    kelly_min = _ptc_kelly.get("KELLY_MIN_POSITION", 0.05)
                    kelly_min_samples = _ptc_kelly.get("KELLY_MIN_SAMPLES", 5)
                    
                    kelly_position_pct, kelly_reason = self._bayesian_filter.calculate_kelly_fraction(
                        prototype_fingerprint=chosen_fp,
                        market_regime=self.state.market_regime,
                        kelly_fraction=kelly_fraction,
                        max_position_pct=kelly_max,
                        min_position_pct=kelly_min,
                        min_sample_count=kelly_min_samples,
                    )
                    
                    # 凯利值为 0 说明期望收益为负，拒绝交易
                    if kelly_position_pct <= 0:
                        self.state.last_event = f"[凯利拒绝] {kelly_reason}"
                        self.state.decision_reason = f"[凯利过滤] {kelly_reason}"
                        self.state.kelly_position_pct = 0.0
                        print(f"[LiveEngine] ⛔ 凯利拒绝: {chosen_fp} | {kelly_reason}")
                        return
                    
                    # 更新 state 中的凯利仓位
                    self.state.kelly_position_pct = kelly_position_pct
                    print(f"[LiveEngine] 📊 凯利仓位: {kelly_position_pct:.1%} | {kelly_reason}")
                
                # C. 计算挂单价格（限价单入场价）
                limit_price = price * (1 + confirm_pct) if side == OrderSide.LONG else price * (1 - confirm_pct)
                
                # 【修复】TP/SL 基于实际入场价（limit_price）计算，而非 kline.close
                # 否则实际 SL 距离 < 预期距离（如预期 0.2% 实际只有 0.13%），容易被扫损
                take_profit, stop_loss = self._calculate_dynamic_tp_sl(
                    entry_price=limit_price,
                    direction=direction,
                    prototype=chosen_proto if self.use_prototypes else None,
                    atr=atr
                )

                # 构建详细的开仓原因说明
                tp_pct = ((take_profit / limit_price) - 1) * 100 if direction == "LONG" else ((limit_price / take_profit) - 1) * 100
                sl_pct = ((limit_price / stop_loss) - 1) * 100 if direction == "LONG" else ((stop_loss / limit_price) - 1) * 100
                
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
                    timeout_bars=timeout,
                    position_size_pct=kelly_position_pct,  # 凯利动态仓位
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
                self.state.last_event = (
                    f"🎯限价单 {direction} | 挂单价 {limit_price:.2f} | "
                    f"当前价 {price:.2f} | 等待触发成交"
                )
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
        
        # 确保有足够的数据进行3根趋势判断
        if len(df) < 3:
            self.state.macd_ready = False
            self.state.kdj_ready = False
            return
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        # 取最近3根K线用于趋势判断
        recent_3 = df.iloc[-3:]

        if not is_exit_gate:
            # 入场逻辑 (Aim 瞄准阶段) - 趋势一致性判定（3根趋势判断）
            if direction == "LONG":
                # MACD 多头趋势：柱体在零轴上方，或至少近两根在回升
                self.state.macd_ready = (curr['macd_hist'] > 0) or (curr['macd_hist'] > prev['macd_hist'])
                
                # 【改进】KDJ 3根趋势判断：多头需要最近3根中至少2根 J > D，且 J 整体上升
                j_values = recent_3['j'].values
                d_values = recent_3['d'].values
                # 条件1：最近3根中至少2根 J > D
                j_above_d_count = sum(j_values > d_values)
                # 条件2：J 值整体上升（第3根 > 第1根）
                j_trend_up = (j_values[-1] > j_values[0])
                self.state.kdj_ready = (j_above_d_count >= 2) and j_trend_up
                
            else: # SHORT
                # MACD 空头趋势：柱体在零轴下方，或至少近两根在走弱
                self.state.macd_ready = (curr['macd_hist'] < 0) or (curr['macd_hist'] < prev['macd_hist'])
                
                # 【改进】KDJ 3根趋势判断：空头需要最近3根中至少2根 J < D，且 J 整体下降
                j_values = recent_3['j'].values
                d_values = recent_3['d'].values
                # 条件1：最近3根中至少2根 J < D
                j_below_d_count = sum(j_values < d_values)
                # 条件2：J 值整体下降（第3根 < 第1根）
                j_trend_down = (j_values[-1] < j_values[0])
                self.state.kdj_ready = (j_below_d_count >= 2) and j_trend_down
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
        self.state.market_regime = self._confirm_market_regime()
        
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

                # 【regime-direction 一致性】只匹配与市场方向一致的原型
                BULL_REGIMES = {"强多头", "弱多头", "震荡偏多"}
                BEAR_REGIMES = {"强空头", "弱空头", "震荡偏空"}
                if match_regime in BULL_REGIMES:
                    # 偏多市场：只看 LONG
                    lp = self._proto_matcher.match_entry(pre_entry_traj, direction="LONG", regime=match_regime)
                    long_sim = lp.get("similarity", 0.0)
                    if long_sim > 0 and lp.get("best_prototype"):
                        best_sim, best_dir = long_sim, "LONG"
                        p = lp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                elif match_regime in BEAR_REGIMES:
                    # 偏空市场：只看 SHORT
                    sp = self._proto_matcher.match_entry(pre_entry_traj, direction="SHORT", regime=match_regime)
                    short_sim = sp.get("similarity", 0.0)
                    if short_sim > 0 and sp.get("best_prototype"):
                        best_sim, best_dir = short_sim, "SHORT"
                        p = sp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                else:
                    # 其他状态：双向匹配，取更高的
                    lp = self._proto_matcher.match_entry(pre_entry_traj, direction="LONG", regime=match_regime)
                    sp = self._proto_matcher.match_entry(pre_entry_traj, direction="SHORT", regime=match_regime)
                    long_sim = lp.get("similarity", 0.0)
                    short_sim = sp.get("similarity", 0.0)
                    if long_sim >= short_sim and long_sim > 0 and lp.get("best_prototype"):
                        best_sim, best_dir = long_sim, "LONG"
                        p = lp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                    elif short_sim > 0 and sp.get("best_prototype"):
                        best_sim, best_dir = short_sim, "SHORT"
                        p = sp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
            
            self.state.best_match_similarity = best_sim
            self.state.best_match_template = best_fp
            
            # 【新增】实时决策说明
            if best_sim >= self.cosine_threshold:
                self.state.fingerprint_status = "匹配达标"
                # 检查指标状态（MACD是必要条件，KDJ仅参考）
                macd_ok = self.state.macd_ready
                kdj_ok = self.state.kdj_ready
                kdj_hint = "✓" if kdj_ok else "✗"
                
                # 价格位置预警 + 翻转提示
                pos_warning = ""
                if self._df_buffer is not None and len(self._df_buffer) >= 20 and best_dir:
                    _lb = min(30, len(self._df_buffer))
                    _recent = self._df_buffer.tail(_lb)
                    _rh = _recent['high'].max()
                    _rl = _recent['low'].min()
                    _rs = _rh - _rl
                    if _rs > 0:
                        _pp = (kline.close - _rl) / _rs
                        _regime = self.state.market_regime
                        _is_range = "震荡" in _regime if _regime else False
                        _dz = 0.30 if _is_range else 0.15
                        if best_dir == "SHORT" and _pp < _dz:
                            if _is_range:
                                pos_warning = f" 🔄底部({_pp:.0%})将翻转做多"
                            else:
                                pos_warning = f" ⚠️价格在区间底部({_pp:.0%})，做空危险！"
                        elif best_dir == "LONG" and _pp > (1 - _dz):
                            if _is_range:
                                pos_warning = f" 🔄顶部({_pp:.0%})将翻转做空"
                            else:
                                pos_warning = f" ⚠️价格在区间顶部({_pp:.0%})，做多危险！"
                
                if macd_ok:
                    # MACD通过即可，KDJ仅参考
                    self.state.decision_reason = f"匹配成功({best_sim:.1%})，MACD已对齐(KDJ{kdj_hint}参考)。等待收线确认...{pos_warning}"
                else:
                    self.state.decision_reason = f"指纹匹配达标({best_sim:.1%})，正在等待 MACD 动能对齐(KDJ{kdj_hint}参考)。{pos_warning}"
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
        第二层确认：技术指标门控 (Aim)
        
        改进逻辑 v2：
        - MACD：必须通过（一票否决权）
          旧逻辑：仅看最近1根K线 → 在底部做空也能通过（因为MACD柱在下降）
          新逻辑：看最近3根K线的MACD趋势 + 方向一致性
          SHORT要求：MACD柱状图至少2/3根在下降（趋势确认），且当前MACD < 0 或 MACD正在加速下行
          LONG要求：MACD柱状图至少2/3根在上升（趋势确认），且当前MACD > 0 或 MACD正在加速上行
        - KDJ：降级为参考
        
        Returns:
            True = MACD确认方向一致，允许开仓
        """
        if df is None or len(df) < 5:
            return False
            
        # 获取最新3根数据（看趋势而非单点）
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # MACD柱状图变化方向（3根K线）
        hist_values = [prev2['macd_hist'], prev['macd_hist'], curr['macd_hist']]
        hist_increasing = sum(1 for i in range(1, 3) if hist_values[i] > hist_values[i-1])  # 上升的次数
        hist_decreasing = sum(1 for i in range(1, 3) if hist_values[i] < hist_values[i-1])  # 下降的次数
        
        if direction == "LONG":
            # MACD 多头趋势（更严格）：
            # 条件1：MACD柱状图正在上升（至少2/3根在涨）
            # 条件2：当前MACD > 0（零轴上方）或 MACD正在从负值加速回升
            trend_up = hist_increasing >= 2
            above_zero = curr['macd_hist'] > 0
            accelerating_up = (curr['macd_hist'] > prev['macd_hist'] > prev2['macd_hist'])
            
            # 必须趋势向上 AND (在零轴上方 OR 加速回升中)
            macd_ok = trend_up and (above_zero or accelerating_up)
            self.state.macd_ready = macd_ok
            
            # KDJ 多头趋势：仅记录状态作为参考（不拦截开仓）
            kdj_ok = (
                ((curr['j'] >= curr['d']) or (curr['k'] >= curr['d'])) and
                ((curr['j'] >= prev['j']) or (curr['k'] >= prev['k']))
            )
            self.state.kdj_ready = kdj_ok
            
            return macd_ok
            
        elif direction == "SHORT":
            # MACD 空头趋势（更严格）：
            # 条件1：MACD柱状图正在下降（至少2/3根在跌）
            # 条件2：当前MACD < 0（零轴下方）或 MACD正在从正值加速下行
            trend_down = hist_decreasing >= 2
            below_zero = curr['macd_hist'] < 0
            accelerating_down = (curr['macd_hist'] < prev['macd_hist'] < prev2['macd_hist'])
            
            # 必须趋势向下 AND (在零轴下方 OR 加速下行中)
            macd_ok = trend_down and (below_zero or accelerating_down)
            self.state.macd_ready = macd_ok
            
            # KDJ 空头趋势：仅记录状态作为参考（不拦截开仓）
            kdj_ok = (
                ((curr['j'] <= curr['d']) or (curr['k'] <= curr['d'])) and
                ((curr['j'] <= prev['j']) or (curr['k'] <= prev['k']))
            )
            self.state.kdj_ready = kdj_ok
            
            return macd_ok
            
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
        
        # 【新增】更新详细的离场/持有说明（含价格位置）
        pnl_pct = order.profit_pct
        pos_info = ""
        if self._df_buffer is not None and len(self._df_buffer) >= 20:
            _lb = min(30, len(self._df_buffer))
            _recent = self._df_buffer.tail(_lb)
            _rh = _recent['high'].max()
            _rl = _recent['low'].min()
            _rs = _rh - _rl
            if _rs > 0:
                _pp = (kline.close - _rl) / _rs
                _regime = self.state.market_regime
                _is_range = "震荡" in _regime if _regime else False
                _flip_zone = 0.25
                if order.side == OrderSide.SHORT and _pp < _flip_zone and _is_range:
                    pos_info = f" | ⚠️底部({_pp:.0%})接近翻转"
                elif order.side == OrderSide.LONG and _pp > (1 - _flip_zone) and _is_range:
                    pos_info = f" | ⚠️顶部({_pp:.0%})接近翻转"
                else:
                    pos_info = f" | 位置={_pp:.0%}"
        self.state.decision_reason = f"[持仓中] {order.side.value} | 相似度={order.current_similarity:.1%} | 收益={pnl_pct:+.2f}%{pos_info}"
        
        # 新开仓保护期：保护期内止损暂缓触发，允许止盈
        from config import PAPER_TRADING_CONFIG as _ptc
        protection_sec = _ptc.get("SL_PROTECTION_SEC", 60)
        hold_seconds = 0.0
        try:
            hold_seconds = max(0.0, (datetime.now() - order.entry_time).total_seconds())
        except Exception:
            hold_seconds = 0.0
        
        in_protection = hold_seconds < protection_sec
        
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
            
            # ── 反手单逻辑：止损触发时，准备反手信号 ──
            stopped_side = order.side.value  # "LONG" 或 "SHORT"
            self._last_stoploss_side = stopped_side
            self._last_stoploss_time = time.time()
            
            if close_reason == CloseReason.STOP_LOSS:
                from config import PAPER_TRADING_CONFIG as _ptc
                reverse_enabled = _ptc.get("REVERSE_ON_STOPLOSS", True)
                max_reverse = _ptc.get("REVERSE_MAX_COUNT", 1)
                
                if reverse_enabled and self._reverse_count < max_reverse:
                    reverse_dir = "LONG" if stopped_side == "SHORT" else "SHORT"
                    self._reverse_pending = True
                    self._reverse_direction = reverse_dir
                    self._reverse_price = kline.close
                    self._reverse_count += 1
                    print(f"[LiveEngine] 🔄 止损反手信号: {stopped_side} 止损 → 准备 {reverse_dir} 反手"
                          f" (连续反手第{self._reverse_count}次，上限{max_reverse}次)")
                else:
                    self._reverse_pending = False
                    if self._reverse_count >= max_reverse:
                        print(f"[LiveEngine] ⛔ 连续反手已达上限({max_reverse}次)，不再反手")
            else:
                # 非止损离场（止盈、信号等），重置反手计数
                self._reverse_count = 0
                self._reverse_pending = False
            
            self._reset_position_state(reason_text)
            return

        # 保护期内跳过相似度检查和追踪止损调整
        if in_protection:
            remaining = max(0, protection_sec - hold_seconds)
            self.state.hold_reason = f"新开仓保护期({protection_sec}秒，剩余{remaining:.0f}秒)，止损暂缓、允许止盈。"
            self.state.exit_reason = "保护期内不执行相似度离场和追踪止损调整。"
            return
        
        # ══════════════════════════════════════════════════════════
        # 【新增】市场因素一致性监控：市场状态 + MACD + KDJ
        # 市场反转多数投票检测（2/3即触发）
        # ══════════════════════════════════════════════════════════
        self.state.market_regime = self._confirm_market_regime()  # 持仓期间持续更新市场状态（使用确认后的稳定状态）
        
        # ══════════════════════════════════════════════════════════
        # 【新增】前3根K线紧急止损守卫 (Early Exit Guard)
        # 问题：前3根保护期内只有硬止损保护，形成致命盲区
        # 方案：
        #   1. 亏损超阈值(默认-1.5%) → 紧急平仓
        #   2. 原始市场状态与持仓方向冲突 → 收紧止损至入场价附近
        # 注意：使用原始状态(_infer)而非确认状态(_confirm)，更灵敏
        # ══════════════════════════════════════════════════════════
        if order.hold_bars < 3:
            early_exit_pct = PAPER_TRADING_CONFIG.get("EARLY_EXIT_ADVERSE_PCT", 1.5)
            
            # ─── 检查1：亏损超过紧急阈值 → 立即平仓 ───
            # profit_pct 已包含杠杆，负值代表亏损
            if order.profit_pct <= -early_exit_pct:
                reason_text = (
                    f"[紧急止损] 保护期第{order.hold_bars}根K线, "
                    f"亏损{order.profit_pct:.1f}% 超过紧急阈值-{early_exit_pct}%"
                )
                print(f"[LiveEngine] 🚨 {reason_text}")
                self.state.last_event = f"🚨 保护期紧急止损({order.profit_pct:.1f}%)"
                if self._close_and_reset(kline.close, self._current_bar_idx,
                                         CloseReason.STOP_LOSS, order, reason_text):
                    return
            
            # ─── 检查2：原始市场状态与持仓方向冲突 → 收紧止损 ───
            raw_regime = self._infer_market_regime()
            _BULL_REGIMES = {"强多头", "弱多头", "震荡偏多"}
            _BEAR_REGIMES = {"强空头", "弱空头", "震荡偏空"}
            
            regime_conflict = (
                (order.side == OrderSide.LONG and raw_regime in _BEAR_REGIMES) or
                (order.side == OrderSide.SHORT and raw_regime in _BULL_REGIMES)
            )
            
            if regime_conflict:
                tighten_pct = PAPER_TRADING_CONFIG.get("EARLY_EXIT_TIGHTEN_PCT", 0.005)
                entry = order.entry_price
                old_sl = order.stop_loss
                old_sl_str = f"{old_sl:.2f}" if old_sl is not None else "无"
                
                if order.side == OrderSide.LONG:
                    tightened_sl = entry * (1 - tighten_pct)
                    if old_sl is None or tightened_sl > old_sl:
                        order.stop_loss = tightened_sl
                        print(f"[LiveEngine] ⚠ 保护期市场冲突(LONG vs {raw_regime}): "
                              f"收紧止损 {old_sl_str} → {tightened_sl:.2f}")
                        self.state.last_event = f"⚠ 保护期收紧止损(市场{raw_regime})"
                else:  # SHORT
                    tightened_sl = entry * (1 + tighten_pct)
                    if old_sl is None or tightened_sl < old_sl:
                        order.stop_loss = tightened_sl
                        print(f"[LiveEngine] ⚠ 保护期市场冲突(SHORT vs {raw_regime}): "
                              f"收紧止损 {old_sl_str} → {tightened_sl:.2f}")
                        self.state.last_event = f"⚠ 保护期收紧止损(市场{raw_regime})"
        
        if order.hold_bars >= 3:  # 与信号离场保护一致
            market_reversal = self._check_market_reversal(order)
            if market_reversal["should_exit"]:
                # 输出详细的决策依据
                d = market_reversal["details"]
                curr = self._df_buffer.iloc[-1]
                prev = self._df_buffer.iloc[-2]
                votes = d.get('reversal_votes', '?')
                print(f"[LiveEngine] 🔄 市场反转投票触发 ({votes}/3票):")
                print(f"  ├─ 方向: {order.side.value} | 持仓: {order.hold_bars}根K线")
                print(f"  ├─ 市场状态(原始): {d['regime_raw']} "
                      f"({'转空✓' if d['regime_bearish'] else '转多✓' if d['regime_bullish'] else '中性✗'})"
                      f" | 确认状态: {d['regime_confirmed']}")
                print(f"  ├─ MACD柱: {prev['macd_hist']:.2f} → {curr['macd_hist']:.2f} "
                      f"({'转空✓' if d['macd_bearish'] else '转多✓' if d['macd_bullish'] else '中性✗'})")
                print(f"  ├─ KDJ-J: {prev['j']:.1f} → {curr['j']:.1f} "
                      f"({'转空✓' if d['kdj_bearish'] else '转多✓' if d['kdj_bullish'] else '中性✗'})")
                print(f"  └─ 结论: {market_reversal['reason']}")
                
                reason_text = self._build_exit_reason(f"市场反转({market_reversal['reason']})", order)
                if self._close_and_reset(kline.close, self._current_bar_idx,
                                         CloseReason.SIGNAL, order, reason_text):
                    return
        
        # ══════════════════════════════════════════════════════════
        # 【核心】持仓中价格位置翻转检测
        # 震荡市中：持SHORT到底部 → 平仓+做多 / 持LONG到顶部 → 平仓+做空
        # 比止损反手更聪明：不等止损，主动在有利位置翻转
        # ══════════════════════════════════════════════════════════
        if order.hold_bars >= 3:  # 至少持仓3根K线后才翻转（避免刚开仓就翻）
            flip_result = self._check_position_flip(order, kline, atr)
            if flip_result and flip_result.get("should_flip"):
                flip_dir = flip_result["flip_direction"]
                flip_pos = flip_result["price_position"]
                pos_label = "底部" if flip_dir == "LONG" else "顶部"
                
                reason_text = (
                    f"[位置翻转] {order.side.value}→{flip_dir} | "
                    f"价格在区间{pos_label}({flip_pos:.0%}) | "
                    f"当前收益={order.profit_pct:+.1f}% | "
                    f"{flip_result.get('detail', '')}"
                )
                print(f"[LiveEngine] 🔄🔄 持仓翻转触发: {reason_text}")
                
                # 记录翻转信息到订单（平仓前）
                order.decision_reason = reason_text
                
                # 平仓当前持仓
                closed = self._close_and_reset(
                    kline.close, self._current_bar_idx,
                    CloseReason.POSITION_FLIP, order, reason_text
                )
                
                if closed:
                    # 准备反手信号（翻转单）
                    self._flip_pending = True
                    self._flip_direction = flip_dir
                    self._flip_price_position = flip_pos
                    self._flip_proto_fp = flip_result.get("flip_fp", "")
                    self._flip_similarity = flip_result.get("flip_sim", 0.0)
                    self._flip_proto = flip_result.get("flip_proto")
                    self._flip_template = flip_result.get("flip_template")
                    
                    print(f"[LiveEngine] 🔄 翻转信号已准备: {flip_dir} | "
                          f"原型={self._flip_proto_fp} | 相似度={self._flip_similarity:.1%}")
                return
        
        # 三阶段追踪止损 + 追踪止盈
        self._update_trailing_stop(order, kline, atr)
        
        # ── 价格动量衰减离场（高点检测 + 自适应响应）──
        momentum_exit = self._check_momentum_decay_exit(order, kline)
        if momentum_exit["should_exit"]:
            reason_text = f"[动量衰减] {momentum_exit['reason']} | 峰值利润={order.peak_profit_pct:.1f}% → 当前={order.profit_pct:.1f}%"
            print(f"[LiveEngine] 📉 动量衰减离场: {reason_text}")
            self.state.last_event = f"📉 动量衰减平仓"
            if self._close_and_reset(kline.close, self._current_bar_idx, CloseReason.SIGNAL, order, reason_text):
                return
        elif momentum_exit.get("should_tighten_stop", False):
            # 收紧止损到成本价以上（可信度中高，不立即平但防止回撤）
            leverage = float(self._paper_trader.leverage)
            entry = order.entry_price
            current_profit_pct = order.profit_pct / leverage / 100.0  # 转为价格百分比
            
            if order.side == OrderSide.LONG:
                # 收紧到成本价 + 当前利润的 50%
                tightened_sl = entry * (1 + current_profit_pct * 0.5)
                if tightened_sl > (order.stop_loss or 0):
                    order.stop_loss = tightened_sl
                    print(f"[LiveEngine] 🔒 动量衰减信号：收紧止损到 {tightened_sl:.2f} (锁定50%利润)")
                    self.state.last_event = f"🔒 收紧止损(动量衰减信号)"
            else:  # SHORT
                tightened_sl = entry * (1 - current_profit_pct * 0.5)
                if tightened_sl < (order.stop_loss or float('inf')):
                    order.stop_loss = tightened_sl
                    print(f"[LiveEngine] 🔒 动量衰减信号：收紧止损到 {tightened_sl:.2f} (锁定50%利润)")
                    self.state.last_event = f"🔒 收紧止损(动量衰减信号)"
        
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
                        
                        # 【修复】更新订单的模板指纹（添加防御性检查）
                        new_proto_id = getattr(new_proto, 'prototype_id', None)
                        new_proto_regime = getattr(new_proto, 'regime', None) or ""
                        regime_short = new_proto_regime[:2] if new_proto_regime else "未知"
                        
                        order.template_fingerprint = f"proto_{direction}_{new_proto_id}_{regime_short}"
                        
                        # 诊断日志
                        if new_proto_id is None or regime_short == "未知":
                            print(f"[警告] 持仓重匹配指纹不完整: {order.template_fingerprint}")
                            print(f"  ├─ new_proto_id: {new_proto_id}")
                            print(f"  └─ regime_short: {regime_short}")

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
                    
                    # 【改进】出场模式匹配：只输出警告日志，不触发实际平仓
                    # 原因：出场模式依赖原型历史数据，样本不足时误判率高
                    # 让TP/SL/追踪止损/市场反转等更可靠的机制来决定真正的离场
                    if exit_check["should_exit"]:
                        # 离场指标确认 (MACD + KDJ 共振)
                        gate_result = self._check_exit_indicator_gate(self._df_buffer, direction)
                        
                        # 输出详细的决策依据（仅作为参考警告）
                        print(f"[LiveEngine] ⚠ 出场模式预警（仅警告，不平仓）:")
                        print(f"  ├─ 方向: {direction} | 持仓: {order.hold_bars}根K线")
                        print(f"  ├─ 形态匹配: {exit_check['pattern_similarity']:.1%} | 信号强度: {exit_check['exit_signal_strength']:.1%}")
                        print(f"  ├─ 离场原因: {exit_check['exit_reason']}")
                        
                        if "details" in gate_result and gate_result["details"]:
                            d = gate_result["details"]
                            print(f"  ├─ MACD柱: {d['macd_prev']:.2f} → {d['macd_curr']:.2f} ({d['macd_status']})")
                            print(f"  ├─ KDJ-J: {d['kdj_prev']:.1f} → {d['kdj_curr']:.1f} ({d['kdj_status']})")
                        
                        print(f"  └─ 指标闸门: {'通过✓' if gate_result['passed'] else '未通过✗'} ({gate_result['reason']})")
                        
                        # 更新UI状态（仅显示预警信息，不执行平仓）
                        exit_reason_str = exit_check["exit_reason"]
                        if gate_result["passed"]:
                            self.state.exit_reason = f"⚠ 出场模式预警: {exit_reason_str}（仅参考，不自动平仓）"
                            self.state.decision_reason = f"[持仓中] 出场形态匹配+指标确认，但仅作为警告参考"
                        else:
                            self.state.exit_reason = f"出场模式轻微预警: {exit_reason_str}（指标未确认，风险低）"
                            self.state.decision_reason = f"[持仓中] 出场形态匹配但指标不支持，继续持仓"
                    
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
        【三因子融合 + 自适应学习】基于原型历史表现 + ATR波动率 + 固定下限 + 学习的最优TP
        
        三因子设计：
        1. 原型历史表现因子：基于avg_profit_pct和win_rate
        2. ATR波动率因子：至少2.0倍ATR，适应市场波动
        3. 固定百分比下限：至少0.15%（BTC约$100），避免噪声止损
        4. **自适应学习因子（新增）**：从实盘历史峰值利润学习最优 TP
        
        止损 = max(三因子)，永远不会太紧
        止盈 = 学习器建议（如果有足够样本） OR 原型建议（回退）
        
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
        
        # 【新增】获取原型指纹，用于查询学习器
        proto_fp = ""
        if prototype and getattr(prototype, 'prototype_id', None):
            regime_short = prototype.regime[:2] if prototype.regime else ""
            proto_fp = f"proto_{prototype.direction}_{prototype.prototype_id}_{regime_short}"
        
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
        
        # 【新增】因子4: 自适应学习的最优 TP（基于历史峰值利润）
        learned_tp_pct = None
        learned_reason = ""
        if self._exit_learning_enabled and self._exit_learner and proto_fp:
            min_samples = PAPER_TRADING_CONFIG.get("EXIT_LEARNING_MIN_SAMPLES", 10)
            proto_learning = self._exit_learner.prototypes.get(proto_fp)
            if proto_learning and len(proto_learning.peak_profit_history) >= min_samples:
                # 使用学习到的最优 TP（ATR 倍数）
                learned_tp_atr_mult = proto_learning.optimal_tp_atr_multiplier
                learned_tp_pct = (atr * learned_tp_atr_mult) / entry_price
                learned_reason = f"学习样本={len(proto_learning.peak_profit_history)}笔，最优{proto_learning.optimal_tp_pct:.1f}% → {learned_tp_atr_mult:.1f}×ATR"
                # 使用学习的 TP 替换原型建议
                price_move_pct = learned_tp_pct
        
        # ========== 因子2: ATR波动率止损（市场适应性）==========
        from config import PAPER_TRADING_CONFIG
        atr_multiplier = PAPER_TRADING_CONFIG.get("ATR_SL_MULTIPLIER", 3.0)
        atr_based_sl_pct = (atr / entry_price) * atr_multiplier
        
        # ========== 因子3: 固定百分比下限（避免噪声止损）==========
        # BTC 1分钟线，至少 0.5% 距离（约 $475），挡住正常波动
        min_fixed_pct = PAPER_TRADING_CONFIG.get("MIN_SL_PCT", 0.005)
        
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
        
        # ========== 最终安全检查：确保止损距离至少 min_fixed_pct ==========
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
        if learned_tp_pct is not None:
            print(f"  - 【学习因子】: {learned_reason}")
        print(f"  - 原型因子: {prototype_sl_pct*100:.3f}% (收益{price_move_pct*100:.3f}% / RR={risk_reward_ratio})")
        print(f"  - ATR因子:  {atr_based_sl_pct*100:.3f}% ({atr:.2f}*{atr_multiplier})")
        print(f"  - 固定下限: {min_fixed_pct*100:.3f}%")
        print(f"  → 最终SL={stop_loss_pct*100:.3f}% (${abs(stop_loss-entry_price):.2f}) | "
              f"TP={take_profit_pct*100:.3f}% (${abs(take_profit-entry_price):.2f})")
        
        return take_profit, stop_loss
    
    def _on_order_update(self, order: PaperOrder):
        """订单更新回调"""
        pass  # 由状态更新回调处理
    
    def _on_trade_closed_internal(self, order: PaperOrder):
        """交易关闭内部回调 — 安全网，确保任何平仓路径都能清理状态"""
        # 【贝叶斯更新】用实盘交易结果更新 Beta 分布
        if self._bayesian_enabled and self._bayesian_filter:
            # 提取原型指纹和市场状态
            proto_fp = getattr(order, 'template_fingerprint', None)
            # 从开仓时的 entry_reason 提取市场状态（如果有）
            entry_reason = getattr(order, 'entry_reason', '')
            market_regime = "未知"
            if "市场=" in entry_reason:
                # 从 "[开仓] 市场=强空头 | SHORT | ..." 中提取
                try:
                    market_regime = entry_reason.split("市场=")[1].split("|")[0].strip()
                except:
                    pass
            
            # 只更新有原型指纹的交易（反手单的 fingerprint="REVERSE" 不更新）
            # 翻转单(FLIP)也参与贝叶斯学习，且权重更高
            if proto_fp and proto_fp != "REVERSE" and proto_fp != "EXCHANGE_SYNC":
                is_win = order.profit_pct > 0
                is_flip = getattr(order, 'is_flip_trade', False)
                
                if is_flip:
                    flip_label = f" [翻转单: {getattr(order, 'flip_reason', '未知')}]"
                    print(f"[LiveEngine] 🔄 翻转单贝叶斯学习: {'盈利' if is_win else '亏损'} "
                          f"{order.profit_pct:+.1f}%{flip_label} → 加权学习")
                
                self._bayesian_filter.update_trade_result(
                    prototype_fingerprint=proto_fp,
                    market_regime=market_regime,
                    is_win=is_win,
                    profit_pct=order.profit_pct,
                    is_flip_trade=is_flip,
                )
        
        # 【离场信号学习】记录交易结果
        if self._exit_learning_enabled and self._exit_learner:
            proto_fp = getattr(order, 'template_fingerprint', None)
            if proto_fp and proto_fp not in ("REVERSE", "EXCHANGE_SYNC"):
                peak_profit_pct = order.peak_profit_pct
                actual_profit_pct = order.profit_pct
                entry_atr = getattr(order, 'entry_atr', 0.0)
                entry_price = order.entry_price
                signals_triggered = getattr(order, 'exit_signals_triggered', [])
                
                if entry_atr > 0:  # 只有有效 ATR 才记录
                    self._exit_learner.record_trade_exit(
                        prototype_fingerprint=proto_fp,
                        peak_profit_pct=peak_profit_pct,
                        actual_profit_pct=actual_profit_pct,
                        atr_at_entry=entry_atr,
                        entry_price=entry_price,
                        signals_triggered=signals_triggered,
                    )
        
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
                        bull_votes = 0
                        bear_votes = 0

                        if short_trend > threshold:
                            bull_votes += 1
                        elif short_trend < -threshold:
                            bear_votes += 1

                        # MACD 柱状图方向作为额外投票
                        if "macd_hist" in self._df_buffer.columns and len(self._df_buffer) >= 2:
                            curr_hist = float(self._df_buffer["macd_hist"].iloc[-1])
                            prev_hist = float(self._df_buffer["macd_hist"].iloc[-2])
                            if curr_hist > prev_hist:
                                bull_votes += 1
                            elif curr_hist < prev_hist:
                                bear_votes += 1

                        if bull_votes >= 2:
                            if short_trend > threshold * 2:
                                if regime in (MarketRegime.RANGE_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.STRONG_BEAR, MarketRegime.UNKNOWN):
                                    return MarketRegime.WEAK_BULL
                            else:
                                if regime in (MarketRegime.RANGE_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.STRONG_BEAR, MarketRegime.UNKNOWN):
                                    return MarketRegime.RANGE_BULL

                        if bear_votes >= 2:
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
    
    def _confirm_market_regime(self) -> str:
        """
        市场状态确认机制：连续3根K线保持同向才切换状态
        
        目的：避免震荡市中市场状态频繁切换（如：震荡偏多 ↔ 震荡偏空）
        原理：维护最近3根K线的市场状态判断历史，只有三根完全一致时才正式切换
        
        好处：
        1. 过滤短期抖动，减少原型频繁切换
        2. 避免贝叶斯key频繁变化导致采样结果不连贯
        3. 确保状态切换是真实趋势变化，而非噪声
        
        Returns:
            确认后的稳定市场状态
        """
        # 1. 获取当前K线的原始市场状态判断
        current_raw_regime = self._infer_market_regime()
        self._last_raw_regime = current_raw_regime  # 缓存原始状态，供持仓反转检测使用（更灵敏）
        
        # 2. 更新历史队列（保持最近3根）
        self._regime_history.append(current_raw_regime)
        if len(self._regime_history) > 3:
            self._regime_history.pop(0)
        
        # 3. 确认逻辑：连续3根完全相同才切换
        if len(self._regime_history) == 3:
            # 检查三根是否完全一致
            if (self._regime_history[0] == self._regime_history[1] == self._regime_history[2]):
                # 三根一致，正式切换状态
                old_regime = self._confirmed_regime
                self._confirmed_regime = current_raw_regime
                
                # 只在状态真正发生变化时输出日志，避免刷屏
                if old_regime != self._confirmed_regime:
                    print(f"[市场状态确认] 连续3根确认: {old_regime} → {self._confirmed_regime}")
            else:
                # 不一致，保持旧状态不变
                if self._confirmed_regime is None:
                    # 首次初始化，使用当前判断
                    self._confirmed_regime = current_raw_regime
                    print(f"[市场状态确认] 首次初始化: {self._confirmed_regime}")
                else:
                    # 保持旧状态，输出待确认信息（低频日志）
                    self._throttled_print(
                        "regime_pending",
                        f"[市场状态待确认] 最近3根: {self._regime_history} | 保持: {self._confirmed_regime}",
                        interval=30.0  # 30秒打印一次，避免刷屏
                    )
        else:
            # 启动阶段（少于3根历史），直接使用原始判断
            if self._confirmed_regime is None:
                self._confirmed_regime = current_raw_regime
                print(f"[市场状态确认] 启动阶段初始化: {self._confirmed_regime}")
        
        return self._confirmed_regime
    
    def _update_swing_points(self):
        """
        实时更新摆动点检测（只使用已确认的历史数据）
        
        与上帝视角的区别：
          - 上帝视角在 i 位置可以看 i+window 的数据
          - 实时只做 1 根 K 线的确认延迟（更贴近最新走势）
        
        检测逻辑：
          当前位置 = current_idx
          确认位置 = current_idx - swing_window
          如果确认位置是局部极值（相对于前后各 swing_window 个K线），则标记
        """
        if self._df_buffer is None:
            return
        
        n = len(self._df_buffer)
        # 3 根 K 线延迟确认，兼顾灵敏度和抗噪声
        window = 3
        
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
            # 真正的止盈：价格触及TP目标
            if tp and ((side == "LONG" and kline.high >= tp) or (side == "SHORT" and kline.low <= tp)):
                return f"触及止盈价(TP={tp:.2f})"
            else:
                return f"止盈(TP={tp:.2f})"
        
        elif close_reason == CloseReason.TRAILING_STOP:
            # 追踪止损/保本止损：SL已移至盈利区，有盈利但未到TP
            if sl_moved and sl_in_profit:
                return f"追踪止损({stage_name}阶段, SL={sl:.2f}, 峰值盈利{peak_pct:.1f}%)"
            elif sl_in_profit:
                return f"保本止损(SL={sl:.2f}已在成本价之上)"
            else:
                return f"追踪止损(SL={sl:.2f})"
        
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
        
        █ 阶段0（未激活）: profit < stage1 → 保持原始止损
        █ 阶段1（保本）:   profit >= stage1 → SL移至入场价附近（保本）
        █ 阶段2（锁利）:   profit >= stage2 → SL锁住峰值利润的50%
        █ 阶段3（紧追）:   profit >= stage3 → SL紧跟峰值利润的70%，追踪TP上移
        
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
        
        # 从配置读取阈值（降低阈值以更早锁定利润）
        stage1_pct = PAPER_TRADING_CONFIG.get("TRAILING_STAGE1_PCT", 1.0)
        stage2_pct = PAPER_TRADING_CONFIG.get("TRAILING_STAGE2_PCT", 2.0)
        stage3_pct = PAPER_TRADING_CONFIG.get("TRAILING_STAGE3_PCT", 3.5)
        
        # ── 阶段判定 ──
        if peak_pct >= stage3_pct:
            new_stage = max(order.trailing_stage, 3)
        elif peak_pct >= stage2_pct:
            new_stage = max(order.trailing_stage, 2)
        elif peak_pct >= stage1_pct:
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
        
        # ── 阶段2：锁利（锁住峰值利润的50%）──
        if new_stage >= 2:
            lock_ratio_stage2 = PAPER_TRADING_CONFIG.get("TRAILING_LOCK_PCT_STAGE2", 0.50)
            lock_pct = peak_pct * lock_ratio_stage2 / 100.0  # 锁住50%的峰值收益
            if order.side == OrderSide.LONG:
                lock_sl = entry * (1 + lock_pct / self._paper_trader.leverage)
                new_sl = max(new_sl, lock_sl)
            else:
                lock_sl = entry * (1 - lock_pct / self._paper_trader.leverage)
                new_sl = min(new_sl, lock_sl)
        
        # ── 阶段3：紧追（锁住峰值利润的70% + 追踪TP上移）──
        if new_stage >= 3:
            lock_ratio_stage3 = PAPER_TRADING_CONFIG.get("TRAILING_LOCK_PCT_STAGE3", 0.70)
            lock_pct = peak_pct * lock_ratio_stage3 / 100.0  # 锁住70%的峰值收益
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

    def _check_momentum_decay_exit(self, order: PaperOrder, kline: KlineData) -> dict:
        """
        价格动量衰减检测 — 识别高点并提前离场
        
        核心逻辑：
        1. 只在有一定利润时检测（防止刚开仓就被触发）
        2. 检测K线实体是否在缩小（动能衰减）
        3. 检测从峰值利润的回撤程度
        4. 结合 KDJ/MACD 确认动量减弱
        
        Returns:
            {"should_exit": bool, "reason": str, "details": dict}
        """
        result = {"should_exit": False, "reason": "", "details": {}}
        
        # 读取配置
        enabled = PAPER_TRADING_CONFIG.get("MOMENTUM_EXIT_ENABLED", True)
        if not enabled:
            return result
        
        min_profit = PAPER_TRADING_CONFIG.get("MOMENTUM_MIN_PROFIT_PCT", 1.5)
        lookback = PAPER_TRADING_CONFIG.get("MOMENTUM_LOOKBACK_BARS", 3)
        decay_threshold = PAPER_TRADING_CONFIG.get("MOMENTUM_DECAY_THRESHOLD", 0.5)
        retracement_threshold = PAPER_TRADING_CONFIG.get("MOMENTUM_PEAK_RETRACEMENT", 0.3)
        
        # 1. 利润门槛：至少有一定利润才检测
        peak_pct = order.peak_profit_pct
        current_pct = order.profit_pct
        if peak_pct < min_profit:
            return result
        
        # 2. 回撤检测：从峰值利润回撤超过阈值
        if peak_pct > 0:
            retracement = (peak_pct - current_pct) / peak_pct
        else:
            retracement = 0
        
        retracement_triggered = retracement >= retracement_threshold
        
        # 3. K线实体衰减检测（需要 df_buffer）
        body_decay = False
        if self._df_buffer is not None and len(self._df_buffer) >= lookback + 1:
            direction = order.side.value
            recent = self._df_buffer.iloc[-lookback:]
            
            # 计算最近几根K线的实体大小
            bodies = []
            for _, row in recent.iterrows():
                body = abs(row['close'] - row['open'])
                bodies.append(body)
            
            if len(bodies) >= 2:
                # 峰值实体 vs 当前实体
                peak_body = max(bodies[:-1]) if len(bodies) > 1 else bodies[0]
                current_body = bodies[-1]
                
                if peak_body > 0:
                    body_ratio = current_body / peak_body
                    body_decay = body_ratio < decay_threshold
                    
                    result["details"]["peak_body"] = peak_body
                    result["details"]["current_body"] = current_body
                    result["details"]["body_ratio"] = body_ratio
        
        # 4. KDJ/MACD 动量确认
        indicator_weak = False
        if self._df_buffer is not None and len(self._df_buffer) >= 2:
            curr = self._df_buffer.iloc[-1]
            prev = self._df_buffer.iloc[-2]
            direction = order.side.value
            
            if direction == "LONG":
                # 做多：J线下降 或 MACD柱缩小
                j_declining = curr['j'] < prev['j']
                macd_shrinking = curr['macd_hist'] < prev['macd_hist']
                indicator_weak = j_declining or macd_shrinking
            else:
                # 做空：J线上升 或 MACD柱回升
                j_rising = curr['j'] > prev['j']
                macd_rising = curr['macd_hist'] > prev['macd_hist']
                indicator_weak = j_rising or macd_rising
            
            result["details"]["j_current"] = curr['j']
            result["details"]["j_prev"] = prev['j']
            result["details"]["macd_current"] = curr['macd_hist']
            result["details"]["macd_prev"] = prev['macd_hist']
        
        # 综合判断：回撤触发 + (实体衰减 或 指标走弱)
        signal_triggered = retracement_triggered and (body_decay or indicator_weak)
        if signal_triggered:
            reasons = []
            reasons.append(f"回撤{retracement:.0%}")
            if body_decay:
                reasons.append("K线实体缩小")
            if indicator_weak:
                reasons.append("指标动量减弱")
            result["reason"] = " + ".join(reasons)
            
            # 【新增】查询信号可信度，决定响应策略
            if self._exit_learning_enabled and self._exit_learner:
                proto_fp = order.template_fingerprint
                confidence, conf_reason = self._exit_learner.get_signal_confidence(
                    proto_fp, "momentum_decay"
                )
                strategy, strategy_reason = self._exit_learner.get_response_strategy(confidence)
                
                result["confidence"] = confidence
                result["strategy"] = strategy
                result["strategy_reason"] = strategy_reason
                
                # 根据可信度决定是否立即平仓
                if strategy == "immediate_exit":
                    result["should_exit"] = True
                    result["reason"] = f"{result['reason']} | 可信度{confidence:.0%}(高) → 立即平仓"
                elif strategy == "tighten_stop":
                    result["should_exit"] = False  # 不立即平，改为收紧止损
                    result["should_tighten_stop"] = True
                    result["reason"] = f"{result['reason']} | 可信度{confidence:.0%}(中高) → 收紧止损"
                elif strategy == "monitor":
                    result["should_exit"] = False
                    result["reason"] = f"{result['reason']} | 可信度{confidence:.0%}(中性) → 监控但保持"
                else:  # ignore
                    result["should_exit"] = False
                    result["reason"] = f"{result['reason']} | 可信度{confidence:.0%}(低) → 忽略信号"
                
                # 记录信号触发（即使不立即平仓）
                if not hasattr(order, 'exit_signals_triggered'):
                    order.exit_signals_triggered = []
                order.exit_signals_triggered.append(("momentum_decay", current_pct))
            else:
                # 未启用学习，保持原有逻辑
                result["should_exit"] = True
        
        result["details"]["retracement"] = retracement
        result["details"]["retracement_triggered"] = retracement_triggered
        result["details"]["body_decay"] = body_decay
        result["details"]["indicator_weak"] = indicator_weak
        
        return result

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

    def _check_position_flip(self, order: PaperOrder, kline: KlineData, atr: float) -> dict:
        """
        持仓中价格位置翻转检测
        
        核心逻辑：
        - 持SHORT + 价格到达区间底部 → 翻转做多
        - 持LONG + 价格到达区间顶部 → 翻转做空
        - 只在震荡市生效（趋势市中追势到极端是合理的）
        - 翻转前尝试匹配反方向原型，有匹配才翻
        
        Returns:
            {"should_flip": bool, "flip_direction": str, "price_position": float, ...}
        """
        result = {"should_flip": False}
        
        if self._df_buffer is None or len(self._df_buffer) < 20:
            return result
        
        # 检查是否震荡市
        regime = self.state.market_regime
        is_range_market = "震荡" in regime if regime else False
        if not is_range_market:
            return result  # 趋势市不翻转
        
        price = kline.close
        lookback = min(30, len(self._df_buffer))
        recent = self._df_buffer.tail(lookback)
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        range_size = range_high - range_low
        
        if range_size <= 0:
            return result
        
        price_position = (price - range_low) / range_size
        
        # 翻转阈值：底部25%做空→翻多 / 顶部25%做多→翻空
        # 比入场翻转(30%)更保守，因为持仓翻转是主动平仓动作
        flip_zone = 0.25
        
        need_flip = False
        flip_direction = None
        
        if order.side == OrderSide.SHORT and price_position < flip_zone:
            need_flip = True
            flip_direction = "LONG"
        elif order.side == OrderSide.LONG and price_position > (1 - flip_zone):
            need_flip = True
            flip_direction = "SHORT"
        
        if not need_flip or not flip_direction:
            return result
        
        # 额外确认：MACD 已经转向支持翻转方向
        if self._df_buffer is not None and len(self._df_buffer) >= 3:
            curr = self._df_buffer.iloc[-1]
            prev = self._df_buffer.iloc[-2]
            prev2 = self._df_buffer.iloc[-3]
            
            if flip_direction == "LONG":
                # 做多翻转需要MACD开始回升
                macd_supports = (curr['macd_hist'] > prev['macd_hist'])
            else:
                # 做空翻转需要MACD开始下降
                macd_supports = (curr['macd_hist'] < prev['macd_hist'])
            
            if not macd_supports:
                # MACD不支持翻转方向，不翻
                return result
        
        # 尝试匹配反方向原型
        flip_proto = None
        flip_template = None
        flip_fp = ""
        flip_sim = 0.0
        flip_matched = False
        
        try:
            from config import TRAJECTORY_CONFIG
            pre_entry_window = TRAJECTORY_CONFIG.get("PRE_ENTRY_WINDOW", 60)
            start_idx = max(0, self._current_bar_idx - pre_entry_window)
            pre_entry_traj = self._fv_engine.get_raw_matrix(start_idx, self._current_bar_idx + 1)
            
            if pre_entry_traj.size > 0:
                # 优先原型匹配
                if hasattr(self, '_proto_matcher') and self._proto_matcher:
                    proto_result = self._proto_matcher.match_entry(
                        pre_entry_traj,
                        direction=flip_direction,
                        regime=regime
                    )
                    if proto_result and proto_result.get("matched"):
                        flip_proto = proto_result.get("best_prototype")
                        flip_sim = proto_result.get("similarity", 0.0)
                        if flip_proto:
                            p_dir = getattr(flip_proto, 'direction', "UNKNOWN")
                            p_id = getattr(flip_proto, 'prototype_id', "?")
                            p_regime = getattr(flip_proto, 'regime', "") or ""
                            flip_fp = f"proto_{p_dir}_{p_id}_{p_regime[:2] if p_regime else '未知'}"
                            flip_matched = True
                
                # 原型匹配失败，尝试模板匹配
                if not flip_matched and hasattr(self, '_matcher') and self._matcher:
                    if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
                        flip_candidates = self.trajectory_memory.get_templates_by_direction(flip_direction)
                        tmpl_result = self._matcher.match_entry(
                            pre_entry_traj,
                            flip_candidates,
                            cosine_threshold=self.cosine_threshold,
                            dtw_threshold=self.dtw_threshold,
                        )
                        if tmpl_result.matched and tmpl_result.best_template:
                            flip_template = tmpl_result.best_template
                            flip_sim = tmpl_result.dtw_similarity
                            flip_fp = flip_template.fingerprint()
                            flip_matched = True
        except Exception as e:
            print(f"[LiveEngine] 持仓翻转匹配异常: {e}")
        
        if not flip_matched:
            # 没有匹配到反方向原型，不翻
            return result
        
        # 相似度门槛：翻转要求稍高的匹配度（避免随意翻转）
        min_flip_sim = max(self.cosine_threshold, 0.90)
        if flip_sim < min_flip_sim:
            return result
        
        pos_label = "底部" if flip_direction == "LONG" else "顶部"
        detail = (
            f"区间={range_low:.0f}-{range_high:.0f} | "
            f"匹配={flip_fp}({flip_sim:.1%}) | MACD支持翻转"
        )
        
        return {
            "should_flip": True,
            "flip_direction": flip_direction,
            "price_position": price_position,
            "detail": detail,
            "flip_fp": flip_fp,
            "flip_sim": flip_sim,
            "flip_proto": flip_proto,
            "flip_template": flip_template,
        }

    def _check_market_reversal(self, order: PaperOrder) -> dict:
        """
        检查市场因素是否多数反转（2/3多数投票制）
        
        三个指标各投一票：市场状态、MACD、KDJ
        做多持仓：≥2个指标转空 → 建议离场
        做空持仓：≥2个指标转多 → 建议离场
        
        Returns:
            {"should_exit": bool, "reason": str, "details": dict}
        """
        result = {"should_exit": False, "reason": "", "details": {}}
        
        if self._df_buffer is None or len(self._df_buffer) < 3:
            return result
        
        direction = order.side.value  # "LONG" or "SHORT"
        curr = self._df_buffer.iloc[-1]
        prev = self._df_buffer.iloc[-2]
        
        # 1. 市场状态检查 —— 使用原始（未确认）状态，提高反转检测灵敏度
        #    确认状态（self.state.market_regime）有3根K线延迟，适合开仓决策
        #    原始状态（self._last_raw_regime）无延迟，适合持仓反转检测
        raw_regime = self._last_raw_regime or self.state.market_regime
        confirmed_regime = self.state.market_regime
        bull_regimes = {MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL, MarketRegime.RANGE_BULL}
        bear_regimes = {MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.RANGE_BEAR}
        
        regime_bullish = raw_regime in bull_regimes
        regime_bearish = raw_regime in bear_regimes
        
        # 2. MACD 趋势检查（连续2根K线确认）
        macd_bullish = curr['macd_hist'] > 0 and curr['macd_hist'] > prev['macd_hist']
        macd_bearish = curr['macd_hist'] < 0 and curr['macd_hist'] < prev['macd_hist']
        
        # 3. KDJ 趋势检查（J线方向）
        kdj_bullish = curr['j'] > prev['j'] and curr['j'] > 50
        kdj_bearish = curr['j'] < prev['j'] and curr['j'] < 50
        
        result["details"] = {
            "regime_raw": str(raw_regime),
            "regime_confirmed": str(confirmed_regime),
            "regime_bullish": regime_bullish,
            "regime_bearish": regime_bearish,
            "macd_bullish": macd_bullish,
            "macd_bearish": macd_bearish,
            "kdj_bullish": kdj_bullish,
            "kdj_bearish": kdj_bearish,
        }
        
        # 2/3 多数投票制：≥2个指标反转即触发离场
        if direction == "LONG":
            # 做多时，统计转空票数
            bearish_votes = sum([regime_bearish, macd_bearish, kdj_bearish])
            bearish_signals = []
            if regime_bearish:
                bearish_signals.append(f"市场(原始)={raw_regime}")
            if macd_bearish:
                bearish_signals.append("MACD转空")
            if kdj_bearish:
                bearish_signals.append("KDJ转空")
            
            result["details"]["reversal_votes"] = bearish_votes
            result["details"]["vote_threshold"] = 2
            
            if bearish_votes >= 2:
                result["should_exit"] = True
                result["reason"] = f"[{bearish_votes}/3票转空] " + " + ".join(bearish_signals)
        else:  # SHORT
            # 做空时，统计转多票数
            bullish_votes = sum([regime_bullish, macd_bullish, kdj_bullish])
            bullish_signals = []
            if regime_bullish:
                bullish_signals.append(f"市场(原始)={raw_regime}")
            if macd_bullish:
                bullish_signals.append("MACD转多")
            if kdj_bullish:
                bullish_signals.append("KDJ转多")
            
            result["details"]["reversal_votes"] = bullish_votes
            result["details"]["vote_threshold"] = 2
            
            if bullish_votes >= 2:
                result["should_exit"] = True
                result["reason"] = f"[{bullish_votes}/3票转多] " + " + ".join(bullish_signals)
        
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
