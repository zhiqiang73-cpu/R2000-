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
from typing import Optional, Callable, Dict, List, Any, Tuple
from dataclasses import dataclass, field, field
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_data_feed import LiveDataFeed, KlineData
from config import PAPER_TRADING_CONFIG, COLD_START_CONFIG
from core.paper_trader import PaperOrder, OrderSide, CloseReason, OrderStatus
from core.binance_testnet_trader import BinanceTestnetTrader
from core.market_regime import MarketRegimeClassifier, MarketRegime
from core.labeler import SwingPoint
from core.bayesian_filter import BayesianTradeFilter
from core.exit_signal_learner import ExitSignalLearner
from core.rejection_tracker import RejectionTracker
from core.exit_timing_tracker import ExitTimingTracker
from core.tpsl_tracker import TPSLTracker
from core.near_miss_tracker import NearMissTracker
from core.early_exit_tracker import EarlyExitTracker
from core.cold_start_manager import ColdStartManager
from core.trade_reasoning import TradeReasoning


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
    
    # 【指纹3D图匹配】多维相似度分解（用于UI展示）
    cosine_similarity: float = 0.0      # 方向相似度（余弦）
    euclidean_similarity: float = 0.0   # 距离相似度（欧氏）
    dtw_similarity: float = 0.0         # 形态相似度（DTW）
    prototype_confidence: float = 0.0   # 原型置信度
    final_match_score: float = 0.0      # 最终匹配分数（含置信度加权）
    
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
    position_score: float = 0.0        # 多维度空间位置评分（-100~+100）
    
    # 门控拒绝追踪（UI 展示用）
    rejection_history: list = field(default_factory=list)   # 最近拒绝记录 (dict list)
    gate_scores: dict = field(default_factory=dict)         # 门控准确率摘要 (dict)
    
    # 门控拒绝跟踪
    rejection_history: List[Dict] = field(default_factory=list)   # 最近拒绝记录（UI 展示）
    gate_scores: Dict[str, Dict] = field(default_factory=dict)    # 门控评分统计（UI 展示）

    # 自适应学习扩展（UI 展示）
    exit_timing_history: List[Dict] = field(default_factory=list)
    exit_timing_scores: Dict[str, Dict] = field(default_factory=dict)
    tpsl_history: List[Dict] = field(default_factory=list)
    tpsl_scores: Dict[str, Dict] = field(default_factory=dict)
    near_miss_history: List[Dict] = field(default_factory=list)
    near_miss_scores: Dict[str, Dict] = field(default_factory=dict)
    regime_history: List[Dict] = field(default_factory=list)
    regime_scores: Dict[str, Dict] = field(default_factory=dict)
    early_exit_history: List[Dict] = field(default_factory=list)
    early_exit_scores: Dict[str, Dict] = field(default_factory=dict)
    adaptive_adjustments_applied: int = 0
    
    # 冷启动系统状态（UI 展示）
    cold_start_enabled: bool = False
    cold_start_thresholds: Dict[str, float] = field(default_factory=dict)
    cold_start_frequency: Dict[str, Any] = field(default_factory=dict)

    # 持仓思维链输出（链环 1～4，供推理 Tab 展示）
    reasoning_result: Optional[Any] = None      # TradeReasoning 5层推理结果
    holding_regime_change: str = ""             # "一致" / "弱化·震荡" / "反转"
    holding_exit_suggestion: str = ""            # "继续持有" / "部分止盈" / "仅收紧止损" / "准备离场" / "立即离场"
    tpsl_action: str = "hold"                    # "hold" / "recalc" / "tighten_sl_only"
    holding_position_suggestion: str = ""       # "维持" / "建议减仓"
    position_suggestion: str = ""               # "维持" / "建议减仓"（与 holding_position_suggestion 同步）

    # 持仓中实时 DeepSeek（仅展示）
    deepseek_holding_advice: str = ""           # AI 持仓建议
    deepseek_judgement: str = ""                # AI 对系统决策的评判
    deepseek_heartbeat: bool = False            # 心跳灯：本轮是否已请求 DeepSeek


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
                 on_error: Optional[Callable[[str], None]] = None,
                 adaptive_controller: Optional[Any] = None):
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
        
        # 自适应控制器
        self.adaptive_controller = adaptive_controller

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
        
        # 执行参数固定：默认仓位比例（凯利启用时会覆盖为动态仓位）
        self.fixed_position_size_pct = float(PAPER_TRADING_CONFIG.get("POSITION_SIZE_PCT", 0.1)) (绋宠禋鐗堟湰: 闃舵鍩哄噯姝㈢泩姝㈡崯绯荤粺)
        default_lev = int(PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20))
        self.fixed_leverage = default_lev

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
        
        # 价格历史缓存（用于反事实分析）
        self._price_history: Dict[int, float] = {}  # {bar_idx: close_price}
        
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
        self._last_tpsl_atr: Optional[float] = None   # 上次重算 TP/SL 时的 ATR，用于 ATR 变化时重算
        
        # 信号组合实盘监控（懒加载）
        self._signal_live_monitor = None   # type: Optional[object]
        self._pending_signal_combos: List[str] = []

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
        
        # ── 反向下单测试模式 ──
        self._reverse_signal_mode = PAPER_TRADING_CONFIG.get("REVERSE_SIGNAL_MODE", False)
        
        # ── 价格位置翻转状态 ──
        self._flip_pending = False            # 是否有待执行的翻转信号
        self._flip_direction: Optional[str] = None  # 翻转方向
        self._flip_price_position: float = 0.0       # 翻转时的价格位置
        self._flip_proto_fp: str = ""                # 翻转匹配的原型指纹
        self._flip_similarity: float = 0.0           # 翻转匹配相似度
        self._flip_proto = None                      # 翻转匹配的原型对象
        self._flip_template = None                   # 翻转匹配的模板对象
        self._pending_flip_mark = False       # 下一个开仓订单标记为翻转单
        
        # ── 入场候选锁（避免等待MACD期间方向抖动） ──
        self._entry_candidate: Optional[Dict[str, Any]] = None
        self._entry_candidate_ttl_bars: int = int(
            PAPER_TRADING_CONFIG.get("ENTRY_CANDIDATE_TTL_BARS", 3)
        )
        
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
        
        if self._bayesian_enabled and self._bayesian_filter:
            print(f"[LiveEngine] 贝叶斯过滤器已启用: Thompson={self._bayesian_filter.thompson_sampling}, "
                  f"最低胜率={self._bayesian_filter.min_win_rate_threshold:.0%}")
        
        # ── 入场拒绝跟踪器（门控自适应学习）──
        self._rejection_tracker: Optional[RejectionTracker] = None
        self._rejection_tracker_enabled = PAPER_TRADING_CONFIG.get("REJECTION_TRACKER_ENABLED", False)
        if self._rejection_tracker_enabled:
            self._rejection_tracker = RejectionTracker(
                eval_bars=PAPER_TRADING_CONFIG.get("REJECTION_EVAL_BARS", 30),
                profit_threshold_pct=PAPER_TRADING_CONFIG.get("REJECTION_PROFIT_THRESHOLD_PCT", 0.3),
                max_history=PAPER_TRADING_CONFIG.get("REJECTION_MAX_HISTORY", 200),
                persistence_path=PAPER_TRADING_CONFIG.get("REJECTION_STATE_FILE", "data/rejection_tracker_state.json"),
            )
            print(f"[LiveEngine] 拒绝跟踪器已启用: "
                  f"评估周期={self._rejection_tracker._eval_bars}根K线, "
                  f"利润阈值={self._rejection_tracker._profit_threshold}%")

        # ── 自适应学习扩展追踪器 ──
        self._exit_timing_tracker: Optional[ExitTimingTracker] = None
        if PAPER_TRADING_CONFIG.get("EXIT_TIMING_TRACKER_ENABLED", True):
            self._exit_timing_tracker = ExitTimingTracker(
                eval_bars=PAPER_TRADING_CONFIG.get("EXIT_TIMING_EVAL_BARS", 30),
                premature_threshold_pct=PAPER_TRADING_CONFIG.get("EXIT_TIMING_PREMATURE_PCT", 0.5),
                late_retracement_pct=PAPER_TRADING_CONFIG.get("EXIT_TIMING_LATE_RETRACE_PCT", 0.5),
                max_history=PAPER_TRADING_CONFIG.get("EXIT_TIMING_MAX_HISTORY", 200),
                min_evaluations_for_suggestion=PAPER_TRADING_CONFIG.get("EXIT_TIMING_MIN_EVALS", 10),
                persistence_path=PAPER_TRADING_CONFIG.get("EXIT_TIMING_STATE_FILE", "data/exit_timing_state.json"),
            )
            print(f"[LiveEngine] 出场时机追踪器已启用: "
                  f"评估周期={self._exit_timing_tracker._eval_bars}根K线, "
                  f"过早阈值={self._exit_timing_tracker._premature_threshold}%")

        self._tpsl_tracker: Optional[TPSLTracker] = None
        if PAPER_TRADING_CONFIG.get("TPSL_TRACKER_ENABLED", False):
            self._tpsl_tracker = TPSLTracker(
                eval_bars=PAPER_TRADING_CONFIG.get("TPSL_EVAL_BARS", 30),
                move_threshold_pct=PAPER_TRADING_CONFIG.get("TPSL_MOVE_PCT", 0.5),
                min_evals_for_suggest=PAPER_TRADING_CONFIG.get("TPSL_MIN_EVALS", 20),
                persistence_path=PAPER_TRADING_CONFIG.get("TPSL_STATE_FILE", "data/tpsl_tracker_state.json"),
            )

        self._near_miss_tracker: Optional[NearMissTracker] = None
        if PAPER_TRADING_CONFIG.get("NEAR_MISS_TRACKER_ENABLED", False):
            self._near_miss_tracker = NearMissTracker(
                eval_bars=PAPER_TRADING_CONFIG.get("NEAR_MISS_EVAL_BARS", 30),
                profit_threshold_pct=PAPER_TRADING_CONFIG.get("NEAR_MISS_PROFIT_THRESHOLD_PCT", 0.3),
                near_miss_margin=PAPER_TRADING_CONFIG.get("NEAR_MISS_MARGIN", 0.10),
                max_history=PAPER_TRADING_CONFIG.get("NEAR_MISS_MAX_HISTORY", 200),
                persistence_path=PAPER_TRADING_CONFIG.get("NEAR_MISS_STATE_FILE", "data/near_miss_tracker_state.json"),
            )
            print(f"[LiveEngine] 近似信号追踪器已启用: "
                  f"评估周期={self._near_miss_tracker._eval_bars}根K线, "
                  f"捕获范围={self._near_miss_tracker._near_miss_margin:.0%}")

        self._early_exit_tracker: Optional[EarlyExitTracker] = None
        if PAPER_TRADING_CONFIG.get("EARLY_EXIT_TRACKER_ENABLED", False):
            self._early_exit_tracker = EarlyExitTracker(
                eval_bars=PAPER_TRADING_CONFIG.get("EARLY_EXIT_EVAL_BARS", 30),
                move_threshold_pct=PAPER_TRADING_CONFIG.get("EARLY_EXIT_MOVE_PCT", 0.5),
                min_evals_for_suggest=PAPER_TRADING_CONFIG.get("EARLY_EXIT_MIN_EVALS", 15),
                persistence_path=PAPER_TRADING_CONFIG.get("EARLY_EXIT_STATE_FILE", "data/early_exit_state.json"),
            )
        
        # ── 冷启动管理器 ──
        self._cold_start_manager: Optional[ColdStartManager] = None
        self._cold_start_manager = ColdStartManager(
            state_file="data/cold_start_state.json",
            on_threshold_changed=self._on_cold_start_threshold_changed,
            on_auto_relax=self._on_cold_start_auto_relax,
        )
        # 同步初始状态到 EngineState
        self._sync_cold_start_state()
        
        # ── 自适应控制器（统一参数调整）──
        from core.adaptive_controller import AdaptiveController
        self._adaptive_controller: Optional[AdaptiveController] = None
        try:
            self._adaptive_controller = AdaptiveController(
                state_file="data/adaptive_controller_state.json"
            )
            print(f"[LiveEngine] 自适应控制器已启用")
        except Exception as e:
            print(f"[LiveEngine] 自适应控制器初始化失败: {e}")
        
        # ── DeepSeek AI 复盘分析器 ──
        from config import DEEPSEEK_CONFIG
        from core.deepseek_reviewer import DeepSeekReviewer
        self._deepseek_reviewer: Optional[DeepSeekReviewer] = None
        try:
            self._deepseek_reviewer = DeepSeekReviewer(config=DEEPSEEK_CONFIG)
            if self._deepseek_reviewer.enabled:
                self._deepseek_reviewer.start_background_worker()
                print(f"[LiveEngine] DeepSeek AI 复盘分析器已启用")
        except Exception as e:
            print(f"[LiveEngine] DeepSeek AI 复盘分析器初始化失败: {e}")
        
        # 会话结束报告（stop() 时生成）
        self._session_end_report: Optional[Dict] = None
    
    def _throttled_print(self, key: str, msg: str, interval: float = 5.0):
        """节流打印：同一 key 的相同内容在 interval 秒内只打印一次"""
        now = time.time()
        last_msg = self._last_log_messages.get(key)
        last_t = self._last_log_times.get(key, 0)
        if msg != last_msg or (now - last_t) >= interval:
            print(msg)
            self._last_log_messages[key] = msg
            self._last_log_times[key] = now

    def _clear_entry_candidate(self, reason: str = ""):
        """清理入场候选锁状态。"""
        if self._entry_candidate is not None and reason:
            self._throttled_print(
                "entry_candidate_clear",
                f"[LiveEngine] 🧹 清理候选锁: {reason}",
                interval=2.0,
            )
        self._entry_candidate = None

    def _lock_entry_candidate(
        self,
        direction: str,
        fingerprint: str,
        similarity: float,
        *,
        source: str,
        prototype=None,
        template=None,
        stage: str = "matched",
        match_result: Dict = None,  # 【指纹3D图】存储匹配结果用于多维相似度
    ):
        """创建或刷新入场候选锁。"""
        ttl = max(1, int(self._entry_candidate_ttl_bars))
        self._entry_candidate = {
            "direction": direction,
            "fingerprint": fingerprint,
            "similarity": float(similarity),
            "created_bar_idx": int(self._current_bar_idx),
            "expires_bar_idx": int(self._current_bar_idx + ttl),
            "stage": stage,
            "source": source,  # "prototype" / "template"
            "prototype": prototype,
            "template": template,
            "match_result": match_result,  # 【指纹3D图】存储匹配结果
        }

    def _get_active_entry_candidate(self) -> Optional[Dict[str, Any]]:
        """返回未过期候选；过期时自动清理。"""
        if self._entry_candidate is None:
            return None
        if self._current_bar_idx > int(self._entry_candidate.get("expires_bar_idx", -1)):
            self._clear_entry_candidate("候选超时过期")
            return None
        return self._entry_candidate

    # ─── 冷启动系统回调 ─────────────────────────

    def _sync_cold_start_state(self) -> None:
        """同步冷启动状态到 EngineState（供 UI 读取）"""
        if self._cold_start_manager is None:
            return
        ui_data = self._cold_start_manager.get_state_for_ui()
        self.state.cold_start_enabled = ui_data.get("enabled", False)
        self.state.cold_start_thresholds = ui_data.get("thresholds", {})
        self.state.cold_start_frequency = ui_data.get("frequency", {})

    def _ensure_cold_start_thresholds(self) -> None:
        """冷启动启用时强制应用阈值，避免被其他流程覆盖。"""
        if not self._cold_start_manager or not self._cold_start_manager.enabled:
            return
        try:
            cs_thresholds = self._cold_start_manager.get_thresholds()
            self.cosine_threshold = cs_thresholds.get("cosine", 0.50)
            self.state.entry_threshold = cs_thresholds.get("cosine", 0.50)
            if self._proto_matcher is not None:
                self._proto_matcher.fusion_threshold = cs_thresholds.get("fusion", 0.30)
                self._proto_matcher.cosine_threshold = cs_thresholds.get("cosine", 0.50)
                self._proto_matcher.set_single_dimension_thresholds(
                    long_euclidean=cs_thresholds.get("euclidean"),
                    long_dtw=cs_thresholds.get("dtw"),
                    short_euclidean=cs_thresholds.get("euclidean"),
                    short_dtw=cs_thresholds.get("dtw"),
                )
        except Exception as e:
            print(f"[LiveEngine] 冷启动阈值强制应用失败: {e}")

    def _on_cold_start_threshold_changed(self, thresholds: Dict[str, float]) -> None:
        """冷启动门槛变化回调 - 应用新门槛到匹配器"""
        try:
            # 更新引擎级别的阈值（模板模式也需要）
            self.cosine_threshold = thresholds.get("cosine", 0.70)
            self.state.entry_threshold = thresholds.get("cosine", 0.70)

            if self._proto_matcher is not None:
                # 更新原型匹配器的门槛
                self._proto_matcher.fusion_threshold = thresholds.get("fusion", 0.65)
                self._proto_matcher.cosine_threshold = thresholds.get("cosine", 0.70)
                # 更新单维度阈值
                self._proto_matcher.set_single_dimension_thresholds(
                    long_euclidean=thresholds.get("euclidean"),
                    long_dtw=thresholds.get("dtw"),
                    short_euclidean=thresholds.get("euclidean"),
                    short_dtw=thresholds.get("dtw"),
                )

            print(f"[LiveEngine] 冷启动门槛已应用: 融合={thresholds.get('fusion'):.2f}, "
                  f"余弦={thresholds.get('cosine'):.2f}")
        except Exception as e:
            print(f"[LiveEngine] 冷启动门槛应用失败: {e}")
        
        # 同步状态到 UI
        self._sync_cold_start_state()

    def _on_cold_start_auto_relax(self, message: str) -> None:
        """冷启动自动放宽回调 - 触发 UI 通知"""
        self.state.last_event = f"🧊 {message}"
        print(f"[LiveEngine] {message}")
        # 同步状态到 UI
        self._sync_cold_start_state()
        if self.on_state_update:
            self.on_state_update(self.state)

    def set_cold_start_enabled(self, enabled: bool) -> None:
        """设置冷启动模式开关（供外部调用）"""
        if self._cold_start_manager is None:
            return
        self._cold_start_manager.set_enabled(enabled)

    def get_cold_start_state(self) -> Dict[str, Any]:
        """获取冷启动状态（供外部调用）"""
        if self._cold_start_manager is None:
            return {"enabled": False, "thresholds": {}, "frequency": {}}
        return self._cold_start_manager.get_state_for_ui()

    def _update_similarity_state(
        self,
        similarity: float,
        fingerprint: str,
        match_result: Optional[Dict] = None,
        prototype = None,
    ):
        """
        【指纹3D图匹配】更新多维相似度状态（用于UI展示）
        
        Args:
            similarity: 综合相似度（用于决策）
            fingerprint: 匹配的原型/模板指纹
            match_result: 匹配结果字典（包含分解的相似度）
            prototype: 匹配的原型对象（用于获取置信度）
        """
        self.state.best_match_similarity = similarity
        self.state.best_match_template = fingerprint
        
        # 提取多维相似度分解
        if match_result is not None:
            self.state.cosine_similarity = float(match_result.get("cosine_similarity", similarity))
            self.state.dtw_similarity = float(match_result.get("dtw_similarity", 0.0))
            # euclidean_similarity 可能不在 PrototypeMatcher 的结果中，默认为 similarity
            self.state.euclidean_similarity = float(match_result.get("euclidean_similarity", 0.0))
            self.state.final_match_score = float(match_result.get("final_score", similarity))
        else:
            # 如果没有 match_result，使用 similarity 填充
            self.state.cosine_similarity = similarity
            self.state.dtw_similarity = 0.0
            self.state.euclidean_similarity = 0.0
            self.state.final_match_score = similarity
        
        # 提取原型置信度
        if prototype is not None and hasattr(prototype, 'confidence'):
            self.state.prototype_confidence = float(prototype.confidence)
        elif prototype is not None and hasattr(prototype, 'win_rate'):
            # 旧原型兼容：使用胜率作为基础置信度
            self.state.prototype_confidence = float(prototype.win_rate)
        else:
            self.state.prototype_confidence = 0.0
    
    def _clear_similarity_state(self):
        """清空多维相似度状态"""
        self.state.best_match_similarity = 0.0
        self.state.best_match_template = None
        self.state.cosine_similarity = 0.0
        self.state.euclidean_similarity = 0.0
        self.state.dtw_similarity = 0.0
        self.state.prototype_confidence = 0.0
        self.state.final_match_score = 0.0

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
        # 退出前强制保存交易记录，避免异常关闭导致丢失
        try:
            trader = getattr(self, "_paper_trader", None)
            if trader is not None and getattr(trader, "save_history", None):
                path = getattr(trader, "history_file", None)
                if path:
                    trader.save_history(path)
        except Exception as e:
            print(f"[LiveEngine] 停止时保存交易记录失败: {e}")
        
        # 保存拒绝跟踪器状态 & 生成会话结束报告
        if self._rejection_tracker:
            try:
                # 生成会话结束报告（含调整建议）
                report = self._rejection_tracker.generate_session_report(
                    config_dict=PAPER_TRADING_CONFIG
                )
                self._session_end_report = report
                
                # 打印会话摘要
                stats = report.get("statistics", {})
                print(f"[LiveEngine] 📊 会话结束报告: "
                      f"总拒绝={stats.get('total_rejections', 0)}, "
                      f"总评估={stats.get('total_evaluations', 0)}, "
                      f"已调整={stats.get('total_adjustments_applied', 0)}, "
                      f"待评估={stats.get('pending_evaluations', 0)}")
                
                suggestions = report.get("pending_suggestions", [])
                if suggestions:
                    print(f"[LiveEngine] 💡 有 {len(suggestions)} 项门控阈值调整建议（需手动确认）")
                    for sug in suggestions:
                        print(f"  · {sug.get('param_key')}: "
                              f"{sug.get('current_value')} → {sug.get('suggested_value')} "
                              f"({sug.get('action_text', sug.get('action', ''))})")
                
                self._rejection_tracker.save()
                print("[LiveEngine] 拒绝跟踪器状态已保存")
            except Exception as e:
                print(f"[LiveEngine] 拒绝跟踪器保存/报告生成失败: {e}")

        # 保存自适应学习扩展追踪器
        for tracker in (self._exit_timing_tracker, self._tpsl_tracker,
                        self._near_miss_tracker, self._early_exit_tracker):
            try:
                if tracker:
                    tracker.save()
            except Exception:
                pass
        
        # 保存冷启动状态
        if self._cold_start_manager:
            try:
                self._cold_start_manager.save_state()
                print("[LiveEngine] 冷启动状态已保存")
            except Exception as e:
                print(f"[LiveEngine] 冷启动状态保存失败: {e}")
        
        self._data_feed.stop()
        print("[LiveEngine] 引擎已停止")
    
    def reset(self):
        """重置引擎"""
        self._paper_trader.reset()
        self._current_bar_idx = 0
        self._current_template = None
        self._current_prototype = None
        self.state = EngineState()
        self._session_end_report = None
        # 重置拒绝跟踪器的待评估队列（bar_idx 归零后旧 pending 不再有效）
        if self._rejection_tracker:
            # 清除待评估记录（因为 bar_idx 从 0 重新开始，旧记录无法正确评估）
            # 保留 gate_scores 和 history（已评估的历史仍有学习价值）
            self._rejection_tracker._pending_eval.clear()
            self._rejection_tracker.save()

        for tracker in (self._exit_timing_tracker, self._tpsl_tracker,
                        self._near_miss_tracker, self._early_exit_tracker):
            if tracker:
                if hasattr(tracker, "_pending_eval"):
                    tracker._pending_eval.clear()
                elif hasattr(tracker, "_pending"):
                    tracker._pending.clear()
                try:
                    tracker.save()
                except Exception:
                    pass
        
        # 重置后同步冷启动状态到新的 EngineState
        if self._cold_start_manager:
            self._sync_cold_start_state()

    # ─── 门控阈值调整接口 ─────────────────────────

    def get_session_end_report(self) -> Optional[Dict]:
        """
        获取会话结束报告（引擎停止后调用）。

        报告包含：
        - 各门控评分汇总
        - 待处理的阈值调整建议
        - 本次会话已应用的调整记录
        - 统计概要
        """
        return getattr(self, "_session_end_report", None)

    def apply_threshold_adjustment(self,
                                   param_key: str,
                                   new_value: float,
                                   fail_code: str = "",
                                   reason: str = "") -> Optional[Dict]:
        """
        通过拒绝跟踪器应用阈值调整（带审计日志和安全边界）。

        由 UI 确认后调用。调整会：
        1. 修改 PAPER_TRADING_CONFIG 中的运行时值
        2. 创建审计记录（含门控评分快照）
        3. 持久化调整历史

        Args:
            param_key: 参数名（如 "POS_THRESHOLD_LONG"）
            new_value: 目标新值
            fail_code: 关联的门控代码（审计用）
            reason: 调整原因

        Returns:
            审计记录 dict，或 None（未执行）
        """
        if not self._rejection_tracker:
            # 回退：直接修改配置（无审计）
            old_value = PAPER_TRADING_CONFIG.get(param_key)
            if old_value is None:
                return None
            PAPER_TRADING_CONFIG[param_key] = new_value
            print(f"[LiveEngine] 门控阈值调整（无跟踪器）: {param_key} {old_value} → {new_value}")
            return {"param_key": param_key, "old_value": old_value, "new_value": new_value}

        record = self._rejection_tracker.apply_adjustment(
            config_dict=PAPER_TRADING_CONFIG,
            param_key=param_key,
            new_value=new_value,
            fail_code=fail_code,
            reason=reason,
        )
        return record.to_dict() if record else None

    def apply_adaptive_adjustment(self, source: str, param_key: str, new_value: float,
                                  reason: str = "") -> Optional[Dict]:
        """
        自适应学习通用参数调整入口（出场时机/止盈止损/近似信号/早期出场）。
        支持融合/欧氏/DTW 开仓门槛自适应。
        """
        tracker_map = {
            "exit_timing": self._exit_timing_tracker,
            "tpsl": self._tpsl_tracker,
            "near_miss": self._near_miss_tracker,
            "early_exit": self._early_exit_tracker,
        }
        tracker = tracker_map.get(source)
        if source == "near_miss":
            # 融合/欧氏/DTW 门槛：直接更新 proto_matcher，不通过 config
            if param_key == "FUSION_THRESHOLD" and self._proto_matcher:
                old = getattr(self._proto_matcher, "fusion_threshold", 0.40)
                self._proto_matcher.fusion_threshold = float(new_value)
                if self._cold_start_manager and hasattr(self._cold_start_manager, "_state"):
                    self._cold_start_manager._state.fusion_threshold = float(new_value)
                print(f"[LiveEngine] 融合评分阈值自适应: {old:.2f} → {new_value:.2f}")
                return {"param_key": param_key, "old_value": old, "new_value": new_value}
            if param_key in ("EUCLIDEAN_MIN_THRESHOLD", "DTW_MIN_THRESHOLD") and self._proto_matcher:
                euc_val = new_value if param_key == "EUCLIDEAN_MIN_THRESHOLD" else None
                dtw_val = new_value if param_key == "DTW_MIN_THRESHOLD" else None
                self._proto_matcher.set_single_dimension_thresholds(
                    long_euclidean=euc_val, long_dtw=dtw_val,
                    short_euclidean=euc_val, short_dtw=dtw_val,
                )
                print(f"[LiveEngine] {param_key} 自适应: → {new_value:.2f}")
                return {"param_key": param_key, "new_value": new_value}
            if not tracker:
                return None
            record = tracker.apply_adjustment(PAPER_TRADING_CONFIG, param_key, new_value, reason=reason)
            if record is not None:
                if param_key == "COSINE_THRESHOLD":
                    self.cosine_threshold = float(record.new_value)
                    self.state.entry_threshold = float(record.new_value)
                    if self._proto_matcher:
                        self._proto_matcher.cosine_threshold = float(record.new_value)
                return record.to_dict()
            return None
        if not tracker:
            return None
        if source in ("exit_timing", "tpsl", "early_exit"):
            return tracker.apply_adjustment(PAPER_TRADING_CONFIG, param_key, new_value, reason=reason)
        return None
    
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
                # 【WF Evolution】注入挂载的进化权重（多空分开时两套，否则一套）
                self._using_evolved_weights = False
                pending_long = getattr(self, "_pending_evolved_weights_long", None) or getattr(self, "_pending_evolved_weights", None)
                pending_short = getattr(self, "_pending_evolved_weights_short", None)
                if pending_long is not None:
                    try:
                        self._proto_matcher.set_feature_weights(pending_long, short_weights=pending_short)
                        pending_fusion_th = getattr(self, "_pending_evolved_fusion_th", None)
                        if pending_fusion_th is not None:
                            self._proto_matcher.fusion_threshold = pending_fusion_th
                        pending_cosine_th = getattr(self, "_pending_evolved_cosine_th", None)
                        if pending_cosine_th is not None:
                            self._proto_matcher.cosine_threshold = pending_cosine_th
                            self.cosine_threshold = pending_cosine_th
                            self.state.entry_threshold = pending_cosine_th
                        # 【欧氏/DTW阈值注入】注入进化后的欧氏距离和DTW形态阈值
                        pending_euc_th_long = getattr(self, "_pending_evolved_euclidean_th_long", None)
                        pending_euc_th_short = getattr(self, "_pending_evolved_euclidean_th_short", None)
                        pending_dtw_th_long = getattr(self, "_pending_evolved_dtw_th_long", None)
                        pending_dtw_th_short = getattr(self, "_pending_evolved_dtw_th_short", None)
                        if any(x is not None for x in [pending_euc_th_long, pending_euc_th_short, pending_dtw_th_long, pending_dtw_th_short]):
                            self._proto_matcher.set_single_dimension_thresholds(
                                long_euclidean=pending_euc_th_long,
                                long_dtw=pending_dtw_th_long,
                                short_euclidean=pending_euc_th_short,
                                short_dtw=pending_dtw_th_short
                            )
                        self._using_evolved_weights = True
                        print(f"[LiveEngine] 已注入进化后的特征权重 (32维, 多空{'分开' if pending_short is not None else '共用'})")
                    except Exception as e:
                        print(f"[LiveEngine] 注入进化权重失败: {e}")
                
                # 【冷启动系统】如果冷启动启用，覆盖门槛为宽松值
                if self._cold_start_manager and self._cold_start_manager.enabled:
                    try:
                        cs_thresholds = self._cold_start_manager.get_thresholds()
                        self._proto_matcher.fusion_threshold = cs_thresholds.get("fusion", 0.30)
                        self._proto_matcher.cosine_threshold = cs_thresholds.get("cosine", 0.50)
                        self.cosine_threshold = cs_thresholds.get("cosine", 0.50)
                        self.state.entry_threshold = cs_thresholds.get("cosine", 0.50)
                        self._proto_matcher.set_single_dimension_thresholds(
                            long_euclidean=cs_thresholds.get("euclidean"),
                            long_dtw=cs_thresholds.get("dtw"),
                            short_euclidean=cs_thresholds.get("euclidean"),
                            short_dtw=cs_thresholds.get("dtw"),
                        )
                        print(f"[LiveEngine] 🧊 冷启动模式已启用: 融合={cs_thresholds.get('fusion'):.2f}, "
                              f"余弦={cs_thresholds.get('cosine'):.2f}")
                    except Exception as e:
                        print(f"[LiveEngine] 冷启动门槛注入失败: {e}")
            else:
                self._matcher = TrajectoryMatcher()
                # 【冷启动系统】模板模式也需要覆盖门槛
                if self._cold_start_manager and self._cold_start_manager.enabled:
                    try:
                        cs_thresholds = self._cold_start_manager.get_thresholds()
                        self.cosine_threshold = cs_thresholds.get("cosine", 0.50)
                        self.state.entry_threshold = cs_thresholds.get("cosine", 0.50)
                        print(f"[LiveEngine] 🧊 冷启动模式已启用(模板): 余弦={cs_thresholds.get('cosine'):.2f}")
                    except Exception as e:
                        print(f"[LiveEngine] 冷启动门槛注入失败(模板): {e}")
            
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
                            # 【regime-原型一致性】持仓时也运行预匹配，使“匹配原型”与当前市场状态一致
                            # 否则会一直显示开仓时的 LONG 原型，而市场已变为震荡偏空（应显示 SHORT）
                            self._preview_match(kline)
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
        
        # 特征更新后刷新指标灯状态，确保 UI 与后续门控使用相同数据
        self._update_indicator_state()

        # 信号组合实盘监控：检测当前K线触发哪些已知组合
        try:
            if self._signal_live_monitor is None:
                from core.signal_live_monitor import SignalLiveMonitor
                self._signal_live_monitor = SignalLiveMonitor()
            if self._df_buffer is not None and len(self._df_buffer) > 0:
                self._pending_signal_combos = self._signal_live_monitor.on_bar(
                    self._df_buffer, self._current_bar_idx
                )
        except Exception as _e:
            self._pending_signal_combos = []
        
        # 事后评估被拒绝的交易（门控自适应学习）
        if self._rejection_tracker:
            newly_evaluated = self._rejection_tracker.evaluate_pending(
                kline.close, self._current_bar_idx
            )
            if newly_evaluated:
                correct = sum(1 for r in newly_evaluated if r.was_correct)
                wrong = len(newly_evaluated) - correct
                print(f"[LiveEngine] 📊 拒绝评估完成: {len(newly_evaluated)}笔 | "
                      f"✓正确拒绝={correct} ✗错过机会={wrong}")
            # 同步最新数据到 EngineState（供 UI 读取，转为 dict 格式）
            ui_data = self._rejection_tracker.get_state_for_ui()
            self.state.rejection_history = ui_data["rejection_history"]
            self.state.gate_scores = ui_data["gate_scores"]

            # 过滤出市场状态相关的记录/评分
            regime_codes = {"BLOCK_REGIME_UNKNOWN", "BLOCK_REGIME_CONFLICT"}
            self.state.regime_history = [
                r for r in self.state.rejection_history if r.get("fail_code") in regime_codes
            ]
            self.state.regime_scores = {
                k: v for k, v in self.state.gate_scores.items() if k in regime_codes
            }

        # 事后评估出场时机
        if self._exit_timing_tracker:
            self._exit_timing_tracker.evaluate_pending(kline.close, self._current_bar_idx)
            ui_data = self._exit_timing_tracker.get_state_for_ui()
            self.state.exit_timing_history = ui_data["exit_timing_history"]
            self.state.exit_timing_scores = ui_data["exit_timing_scores"]
            self._exit_timing_tracker.save_if_dirty(min_interval_sec=60.0)

        # 事后评估止盈止损
        if self._tpsl_tracker:
            self._tpsl_tracker.evaluate_pending(kline.close, self._current_bar_idx)
            ui_data = self._tpsl_tracker.get_state_for_ui()
            self.state.tpsl_history = ui_data["records"]
            self.state.tpsl_scores = ui_data["scores"]
            self._tpsl_tracker.save_if_dirty(min_interval_sec=60.0)

        # 事后评估近似信号
        if self._near_miss_tracker:
            self._near_miss_tracker.evaluate_pending(kline.close, self._current_bar_idx)
            ui_data = self._near_miss_tracker.get_state_for_ui()
            self.state.near_miss_history = ui_data["near_miss_history"]
            self.state.near_miss_scores = ui_data["near_miss_scores"]
            self._near_miss_tracker.save_if_dirty(min_interval_sec=60.0)

        # 事后评估早期出场
        if self._early_exit_tracker:
            self._early_exit_tracker.evaluate_pending(kline.close, self._current_bar_idx)
            ui_data = self._early_exit_tracker.get_state_for_ui()
            self.state.early_exit_history = ui_data["records"]
            self.state.early_exit_scores = ui_data["scores"]
        
        # 冷启动频率监控 - 检查是否需要自动放宽门槛
        if self._cold_start_manager:
            self._cold_start_manager.check_frequency()
            self._cold_start_manager.save_if_dirty(min_interval_sec=60.0)
            self._sync_cold_start_state()
        
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
        
        # 再次同步拒绝状态：捕获本轮 _process_entry 中新记录的拒绝
        if self._rejection_tracker:
            ui_data = self._rejection_tracker.get_state_for_ui()
            self.state.rejection_history = ui_data["rejection_history"]
            self.state.gate_scores = ui_data["gate_scores"]
            regime_codes = {"BLOCK_REGIME_UNKNOWN", "BLOCK_REGIME_CONFLICT"}
            self.state.regime_history = [
                r for r in self.state.rejection_history if r.get("fail_code") in regime_codes
            ]
            self.state.regime_scores = {
                k: v for k, v in self.state.gate_scores.items() if k in regime_codes
            }
            # 周期性保存待评估记录（防止崩溃丢失 pending 数据）
            self._rejection_tracker.save_if_dirty(min_interval_sec=60.0)
        
        # 再次同步近似信号状态：捕获本轮 _process_entry 中新记录的近似信号
        if self._near_miss_tracker:
            ui_data = self._near_miss_tracker.get_state_for_ui()
            self.state.near_miss_history = ui_data["near_miss_history"]
            self.state.near_miss_scores = ui_data["near_miss_scores"]
    
    def _update_features(self, kline: KlineData) -> bool:
        """更新特征"""
        if self._df_buffer is None or self._fv_engine is None:
            return False
        
        try:
            from utils.indicators import calculate_all_indicators, calculate_support_resistance
            
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
            self._df_buffer = calculate_support_resistance(self._df_buffer)
            
            # 记录价格历史（用于反事实分析）
            self._price_history[self._current_bar_idx] = kline.close
            # 限制历史长度（保留最近500个）
            if len(self._price_history) > 500:
                oldest_keys = sorted(self._price_history.keys())[:-500]
                for k in oldest_keys:
                    del self._price_history[k]
            
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

    def _calc_position_score(self, direction: str) -> tuple:
        """
        空间位置评分：读 _df_buffer 最后一行，5维度加权，返回 (-100, +100) 评分和构成说明。
        用于入场位置检查、持仓翻转检测、TP/SL 微调。
        """
        if self._df_buffer is None or len(self._df_buffer) == 0:
            return 0.0, "无数据"
        row = self._df_buffer.iloc[-1]
        score = 0.0
        parts = []

        # 1. 布林带位置 (权重25): LONG 底部高分, SHORT 顶部高分
        bp = row.get("boll_position", 0.5)
        if pd.isna(bp):
            bp = 0.5
        bp = max(0.0, min(1.0, float(bp)))
        if direction == "LONG":
            s = (1 - bp) * 50 - 25  # 底部=+25, 顶部=-25
        else:
            s = bp * 50 - 25
        score += s
        parts.append(f"布林{s:+.0f}")

        # 2. RSI 极端 (权重20): LONG RSI<30, SHORT RSI>70
        rsi = row.get("rsi", 50)
        if pd.isna(rsi):
            rsi = 50
        rsi = float(rsi)
        if direction == "LONG" and rsi < 30:
            score += 20
            parts.append("RSI超卖+20")
        elif direction == "SHORT" and rsi > 70:
            score += 20
            parts.append("RSI超买+20")

        # 3. 支撑阻力距离 (权重20): LONG 近支撑加分, SHORT 近阻力加分
        d_sup = row.get("dist_to_support", 0.5)
        d_res = row.get("dist_to_resistance", 0.5)
        if pd.isna(d_sup):
            d_sup = 0.5
        if pd.isna(d_res):
            d_res = 0.5
        d_sup = max(0.0, min(1.0, float(d_sup)))
        d_res = max(0.0, min(1.0, float(d_res)))
        if direction == "LONG":
            s_sr = (1 - d_sup) * 40 - 20  # 近支撑=+20, 远支撑=-20
        else:
            s_sr = (1 - d_res) * 40 - 20
        score += s_sr
        parts.append(f"支撑阻力{s_sr:+.0f}")

        # 4. 量价配合 (权重15): obv_slope 与 direction 一致且 volume_ratio>1
        obv_s = row.get("obv_slope", 0)
        vr = row.get("volume_ratio", 1)
        if pd.isna(obv_s):
            obv_s = 0
        if pd.isna(vr):
            vr = 1
        obv_s = float(obv_s)
        vr = float(vr)
        if direction == "LONG" and obv_s > 0 and vr > 1:
            score += 15
            parts.append("量价配合+15")
        elif direction == "SHORT" and obv_s < 0 and vr > 1:
            score += 15
            parts.append("量价配合+15")

        # 5. 反转形态确认 (权重10): pin_bar / engulfing 在关键位置
        pin = row.get("pin_bar", 0)
        eng = row.get("engulfing", 0)
        if pd.isna(pin):
            pin = 0
        if pd.isna(eng):
            eng = 0
        pin = float(pin)
        eng = float(eng)
        if direction == "LONG" and (pin == 1 or eng == 1) and bp < 0.4:
            score += 10
            parts.append("反转形态+10")
        elif direction == "SHORT" and (pin == -1 or eng == -1) and bp > 0.6:
            score += 10
            parts.append("反转形态+10")

        # 6. 趋势强度 (权重10): ADX 震荡加分，强趋势减分
        adx_val = row.get("adx", 25)
        if pd.isna(adx_val):
            adx_val = 25
        adx_val = float(adx_val)
        adx_mod = max(-10, min(10, (25 - adx_val) * 0.5))
        score += adx_mod
        parts.append(f"ADX{adx_mod:+.0f}")

        score = max(-100.0, min(100.0, score))
        return score, " | ".join(parts)

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

    def _sync_exchange_sl_if_used(self, sl_price: float, force: bool = False):
        """
        若当前使用交易所模式（Binance 等），立即将内存中的止损价同步到交易所（可绕过节流）。
        供其他需立刻生效的 SL 更新场景使用。
        """
        if not hasattr(self._paper_trader, "_update_exchange_sl"):
            return
        if self._paper_trader.current_position is None:
            return
        try:
            ok = self._paper_trader._update_exchange_sl(sl_price, force=force)
            if ok:
                print(f"[LiveEngine] 已同步交易所止损至保本: SL={sl_price:.2f}")
        except Exception as e:
            print(f"[LiveEngine] 同步交易所止损失败: {e}")

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
            timeout_bars=timeout,
            regime_at_entry=self.state.market_regime,
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
            timeout_bars=timeout,
            regime_at_entry=self.state.market_regime,
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
        # 冷启动阈值兜底（防止被其他流程覆盖）
        self._ensure_cold_start_thresholds()
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
            chosen_proto = None
            template = None
            chosen_match_result = None  # 【指纹3D图】存储匹配结果用于多维相似度提取
            active_candidate = self._get_active_entry_candidate()

            # 锁状态与当前模式不一致时，直接清理避免脏状态
            if active_candidate is not None:
                expected_source = "prototype" if self.use_prototypes else "template"
                if active_candidate.get("source") != expected_source:
                    self._clear_entry_candidate("候选来源与当前匹配模式不一致")
                    active_candidate = None
                elif (not active_candidate.get("direction")) or (not active_candidate.get("fingerprint")):
                    self._clear_entry_candidate("候选锁字段不完整")
                    active_candidate = None

            if self.use_prototypes:
                # 关键：传入当前市场状态
                current_regime = self.state.market_regime
                long_result: Dict[str, Any] = {"similarity": 0.0, "matched": False}
                short_result: Dict[str, Any] = {"similarity": 0.0, "matched": False}
                long_sim = 0.0
                short_sim = 0.0
                
                # 【严格市场状态过滤】
                # 用户要求：regime 必须一致，不允许 UNKNOWN 状态下开仓
                match_regime = current_regime
                if current_regime == MarketRegime.UNKNOWN:
                    # UNKNOWN 状态下，不进行入场匹配，等待市场状态明确
                    self.state.decision_reason = "[等待] 市场状态未明确 (需 ≥4 个摆动点)，暂不入场。"
                    self.state.fingerprint_status = "状态未知"
                    self.state.last_event = "[入场跳过] 市场状态未知"
                    if self._rejection_tracker:
                        guess_dir = "LONG"
                        if "SHORT" in str(getattr(self.state, "best_match_template", "")).upper():
                            guess_dir = "SHORT"
                        self._rejection_tracker.record_rejection(
                            price=price,
                            direction=guess_dir,
                            fail_code="BLOCK_REGIME_UNKNOWN",
                            gate_stage="regime_filter",
                            market_regime=current_regime,
                            bar_idx=self._current_bar_idx,
                            detail={
                                "market_regime": current_regime,
                                "reason": "UNKNOWN",
                            },
                        )
                    return
                
                # 【regime-direction 一致性】只匹配与市场方向一致的原型
                BULL_REGIMES_ENTRY = {"强多头", "弱多头", "震荡偏多"}
                BEAR_REGIMES_ENTRY = {"强空头", "弱空头", "震荡偏空"}

                if active_candidate is not None:
                    cand_dir = str(active_candidate.get("direction", ""))
                    # 【regime-direction 一致性守卫】候选方向必须与当前市场状态兼容
                    regime_dir_conflict = (
                        (cand_dir == "LONG" and match_regime in BEAR_REGIMES_ENTRY) or
                        (cand_dir == "SHORT" and match_regime in BULL_REGIMES_ENTRY)
                    )
                    if regime_dir_conflict:
                        cand_sim = float(active_candidate.get("similarity", 0.0)) if active_candidate else 0.0
                        cand_fp = str(active_candidate.get("fingerprint", "")) if active_candidate else ""
                        self._clear_entry_candidate(
                            f"候选方向{cand_dir}与当前市场{match_regime}冲突，作废"
                        )
                        active_candidate = None
                        print(f"[LiveEngine] ⚠ 候选锁作废: {cand_dir} vs 市场{match_regime}")
                        if self._rejection_tracker:
                            self._rejection_tracker.record_rejection(
                                price=price,
                                direction=cand_dir,
                                fail_code="BLOCK_REGIME_CONFLICT",
                                gate_stage="regime_filter",
                                market_regime=match_regime,
                                bar_idx=self._current_bar_idx,
                                detail={
                                    "market_regime": match_regime,
                                    "candidate_dir": cand_dir,
                                    "similarity": cand_sim,
                                    "fingerprint": cand_fp,
                                },
                            )
                    else:
                        direction = cand_dir
                        similarity = float(active_candidate.get("similarity", 0.0))
                        chosen_fp = str(active_candidate.get("fingerprint", ""))
                        chosen_proto = active_candidate.get("prototype")
                        chosen_match_result = active_candidate.get("match_result")  # 【指纹3D图】复用匹配结果
                        self._current_prototype = chosen_proto
                        self._current_template = None
                        self._throttled_print(
                            "entry_candidate_reuse",
                            f"[LiveEngine] 🔒 复用候选(等待阶段): {direction} | {chosen_fp} | {similarity:.2%}",
                            interval=1.0,
                        )
                if active_candidate is None:
                    if match_regime in BULL_REGIMES_ENTRY:
                        # 偏多市场：只匹配 LONG
                        long_result = self._proto_matcher.match_entry(
                            pre_entry_traj, direction="LONG", regime=match_regime
                        )
                        long_sim = long_result.get("similarity", 0.0)
                        short_sim = 0.0
                        if long_result.get("matched"):
                            direction, chosen_proto, similarity = "LONG", long_result.get("best_prototype"), long_sim
                            chosen_match_result = long_result  # 【指纹3D图】存储匹配结果
                    elif match_regime in BEAR_REGIMES_ENTRY:
                        # 偏空市场：只匹配 SHORT
                        short_result = self._proto_matcher.match_entry(
                            pre_entry_traj, direction="SHORT", regime=match_regime
                        )
                        long_sim = 0.0
                        short_sim = short_result.get("similarity", 0.0)
                        if short_result.get("matched"):
                            direction, chosen_proto, similarity = "SHORT", short_result.get("best_prototype"), short_sim
                            chosen_match_result = short_result  # 【指纹3D图】存储匹配结果
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
                                chosen_match_result = long_result  # 【指纹3D图】存储匹配结果
                            else:
                                direction, chosen_proto, similarity = "SHORT", short_result.get("best_prototype"), short_sim
                                chosen_match_result = short_result  # 【指纹3D图】存储匹配结果
                        elif long_result.get("matched"):
                            direction, chosen_proto, similarity = "LONG", long_result.get("best_prototype"), long_sim
                            chosen_match_result = long_result  # 【指纹3D图】存储匹配结果
                        elif short_result.get("matched"):
                            direction, chosen_proto, similarity = "SHORT", short_result.get("best_prototype"), short_sim
                            chosen_match_result = short_result  # 【指纹3D图】存储匹配结果

                    # 【指纹3D图】将轨迹矩阵存入匹配结果，用于后续保存到 PaperOrder
                    if chosen_match_result is not None:
                        chosen_match_result["entry_trajectory"] = pre_entry_traj
                    
                    # 【指纹3D图】打印多维相似度分解
                    if chosen_match_result:
                        cos_sim = chosen_match_result.get("cosine_similarity", similarity)
                        dtw_sim = chosen_match_result.get("dtw_similarity", 0.0)
                        self._throttled_print("proto_match",
                            f"[LiveEngine] 原型匹配结果: "
                            f"市场={match_regime} | LONG={long_sim:.1%} | SHORT={short_sim:.1%} | "
                            f"(余弦={cos_sim:.1%}, DTW={dtw_sim:.1%})")
                    else:
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
                if active_candidate is not None:
                    direction = str(active_candidate.get("direction", ""))
                    similarity = float(active_candidate.get("similarity", 0.0))
                    chosen_fp = str(active_candidate.get("fingerprint", ""))
                    template = active_candidate.get("template")
                    chosen_match_result = active_candidate.get("match_result")  # 【指纹3D图】复用匹配结果
                    self._current_template = template
                    self._current_prototype = None
                    self._throttled_print(
                        "entry_candidate_reuse",
                        f"[LiveEngine] 🔒 复用候选(等待阶段): {direction} | {chosen_fp} | {similarity:.2%}",
                        interval=1.0,
                    )
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

            # 近似信号追踪（仅在未达阈值时记录）
            if self._near_miss_tracker and direction is None:
                near_ratio = float(PAPER_TRADING_CONFIG.get("NEAR_MISS_RATIO", 0.85))
                fp_str = ""
                if self.use_prototypes:
                    best_sim = max(long_sim, short_sim)
                    best_dir = "LONG" if long_sim >= short_sim else "SHORT"
                    best_res = long_result if long_sim >= short_sim else short_result
                    thresh = getattr(self._proto_matcher, "fusion_threshold", 0.40) if self._proto_matcher else 0.40
                    detail = {
                        "market_regime": self.state.market_regime,
                        "combined_score": best_res.get("combined_score", best_sim),
                        "cosine_sim": best_res.get("cosine_similarity", 0.0),
                        "euclidean_sim": best_res.get("euclidean_similarity", 0.0),
                        "dtw_sim": best_res.get("dtw_similarity", 0.0),
                    }
                    proto = best_res.get("best_prototype") if isinstance(best_res, dict) else None
                    if proto is not None:
                        d = getattr(proto, "direction", "") or "?"
                        i = getattr(proto, "prototype_id", "") or "?"
                        fp_str = f"proto_{d}_{i}"
                else:
                    best_sim = max(getattr(long_result, "dtw_similarity", 0.0),
                                   getattr(short_result, "dtw_similarity", 0.0))
                    best_dir = "LONG" if getattr(long_result, "dtw_similarity", 0.0) >= getattr(short_result, "dtw_similarity", 0.0) else "SHORT"
                    thresh = self.cosine_threshold
                    detail = {"market_regime": self.state.market_regime}
                    tmpl = long_result.best_template if long_sim >= short_sim else short_result.best_template
                    fp_str = tmpl.fingerprint() if tmpl else ""
                if best_sim > 0 and best_sim < thresh and best_sim >= thresh * near_ratio:
                    self._near_miss_tracker.record_near_miss(
                        price=price,
                        direction=best_dir,
                        similarity=best_sim,
                        threshold=thresh,
                        market_regime=self.state.market_regime,
                        bar_idx=self._current_bar_idx,
                        fingerprint=fp_str,
                        detail=detail,
                    )

            if direction is not None and chosen_fp:
                self._lock_entry_candidate(
                    direction=direction,
                    fingerprint=chosen_fp,
                    similarity=similarity,
                    source="prototype" if self.use_prototypes else "template",
                    prototype=chosen_proto if self.use_prototypes else None,
                    template=template if not self.use_prototypes else None,
                    stage="matched",
                    match_result=chosen_match_result,  # 【指纹3D图】传递匹配结果
                )
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

                price = kline.close
                
                # 【反向下单测试模式】
                if self._reverse_signal_mode:
                    original_direction = direction
                    direction = "SHORT" if direction == "LONG" else "LONG"
                    print(f"[LiveEngine] 🔄 [反向模式] 信号反转: {original_direction} → {direction}")
                
                side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
                
                # 【三重确认逻辑】
                from config import VECTOR_SPACE_CONFIG
                confirm_pct = VECTOR_SPACE_CONFIG.get("ENTRY_CONFIRM_PCT", 0.001)
                timeout = VECTOR_SPACE_CONFIG.get("TRIGGER_TIMEOUT_BARS", 5)
                
                # 0. 【核心改进】多维度空间位置评分 — 代替30根K线硬编码
                #    用 _calc_position_score 综合布林/RSI/支撑阻力/量价/形态/趋势，采用方向化阈值
                flip_triggered = False
                flip_degraded_position_cap = None
                if self._df_buffer is not None and len(self._df_buffer) >= 20:
                    pos_score, score_detail = self._calc_position_score(direction)
                    self.state.position_score = pos_score
                    entry_direction = direction
                    entry_pos_threshold = (
                        PAPER_TRADING_CONFIG.get("POS_THRESHOLD_LONG", -30) if direction == "LONG"
                        else PAPER_TRADING_CONFIG.get("POS_THRESHOLD_SHORT", -40)
                    )

                    if pos_score < entry_pos_threshold:  # 位置不利，触发翻转或拒绝
                        regime = self.state.market_regime
                        is_range_market = "震荡" in regime if regime else False
                        flip_direction = "LONG" if direction == "SHORT" else "SHORT"

                        # ========== 趋势市位置评分放宽 ==========
                        # 趋势市中，位置评分的"逆势"逻辑（布林高位/RSI超买 → 负分）会误杀顺势机会
                        # 解决：趋势多头中，只要位置评分 > -60（极端恶劣才拒绝），否则放行
                        is_trending_market = regime and any(kw in str(regime) for kw in ["强多头", "弱多头", "强空头", "弱空头"])
                        trend_relaxed_threshold = -60  # 趋势市放宽阈值（只拦截极端位置）
                        
                        # 检查是否是顺势交易
                        is_trend_aligned = False
                        if is_trending_market:
                            if direction == "LONG" and ("多头" in regime):
                                is_trend_aligned = True
                            elif direction == "SHORT" and ("空头" in regime):
                                is_trend_aligned = True
                        
                        if is_trend_aligned and pos_score > trend_relaxed_threshold:
                            # 趋势市顺势交易，且位置不是极端恶劣 → 放行
                            print(f"[LiveEngine] 📈 趋势市位置放宽: {direction} | 市场={regime} | "
                                  f"位置={pos_score:.0f} > {trend_relaxed_threshold}（震荡市阈值={entry_pos_threshold}）→ 豁免")
                            # 不触发翻转，继续正常流程
                        else:
                            print(f"[LiveEngine] 🔄 位置评分翻转: {direction}→{flip_direction} | "
                                  f"当前分数={pos_score:.0f}(< {entry_pos_threshold}) | "
                                  f"方向阈值={entry_direction}:{entry_pos_threshold} | {score_detail} | 市场={regime}")

                            if is_range_market:
                                from config import PAPER_TRADING_CONFIG as _ptc_flip
                                enable_template_fallback = bool(_ptc_flip.get("FLIP_FALLBACK_TEMPLATE_ENABLED", True))
                                enable_degraded_fallback = bool(_ptc_flip.get("FLIP_FALLBACK_DEGRADED_ENTRY_ENABLED", True))
                                degraded_min_score = float(_ptc_flip.get("FLIP_FALLBACK_MIN_SCORE", 35.0))
                                degraded_position_cap = float(_ptc_flip.get("FLIP_FALLBACK_DEGRADED_POSITION_PCT", 0.05))
                                degraded_position_cap = max(0.01, min(1.0, degraded_position_cap))

                                # 【震荡市智能翻转】尝试用反方向重新匹配原型
                                flip_result = None
                                flip_matched = False

                                if hasattr(self, '_proto_matcher') and self._proto_matcher:
                                    flip_result = self._proto_matcher.match_entry(
                                        pre_entry_traj,
                                        direction=flip_direction,
                                        regime=self.state.market_regime
                                    )
                                    flip_matched = flip_result and flip_result.get("matched")

                                if (not flip_matched and enable_template_fallback
                                        and hasattr(self, '_matcher') and self._matcher):
                                    if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
                                        flip_candidates = self.trajectory_memory.get_templates_by_direction(flip_direction)
                                        flip_tmpl_result = self._matcher.match_entry(
                                            pre_entry_traj,
                                            flip_candidates,
                                            cosine_threshold=self.cosine_threshold,
                                            dtw_threshold=self.dtw_threshold,
                                        )
                                        if flip_tmpl_result.matched and flip_tmpl_result.best_template:
                                            flip_result = {
                                                "matched": True,
                                                "best_prototype": None,
                                                "best_template": flip_tmpl_result.best_template,
                                                "similarity": flip_tmpl_result.dtw_similarity,
                                            }
                                            flip_matched = True

                                if flip_matched and flip_result:
                                    flip_proto = flip_result.get("best_prototype")
                                    flip_template = flip_result.get("best_template")
                                    flip_sim = flip_result.get("similarity", 0.0)

                                    direction = flip_direction
                                    similarity = flip_sim
                                    side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
                                    flip_triggered = True

                                    if flip_proto:
                                        chosen_proto = flip_proto
                                        self._current_prototype = flip_proto
                                        self._current_template = None
                                        proto_direction = getattr(flip_proto, 'direction', None) or "UNKNOWN"
                                        proto_id = getattr(flip_proto, 'prototype_id', None)
                                        proto_regime = getattr(flip_proto, 'regime', None) or ""
                                        regime_short = proto_regime[:2] if proto_regime else "未知"
                                        chosen_fp = f"proto_{proto_direction}_{proto_id}_{regime_short}"
                                    elif flip_template:
                                        self._current_template = flip_template
                                        self._current_prototype = None
                                        chosen_fp = flip_template.fingerprint()

                                    self.state.last_event = (
                                        f"🔄 位置翻转: {flip_direction} | {chosen_fp} | {flip_sim:.1%} | "
                                        f"方向阈值={entry_direction}:{entry_pos_threshold} | 当前分数={pos_score:.0f}"
                                    )
                                    self.state.decision_reason = (
                                        f"[智能翻转] 原始信号{('SHORT' if flip_direction == 'LONG' else 'LONG')}"
                                        f"位置评分不利，"
                                        f"翻转为{flip_direction}，匹配={chosen_fp}({flip_sim:.1%})，"
                                        f"方向阈值={entry_direction}:{entry_pos_threshold} | 当前分数={pos_score:.0f} | {score_detail}"
                                    )
                                    print(f"[LiveEngine] ✅ 翻转匹配成功: {flip_direction} | "
                                          f"{chosen_fp} | {flip_sim:.1%}")
                                else:
                                    degraded_used = False
                                    flip_score = 0.0
                                    flip_score_detail = ""
                                    if enable_degraded_fallback:
                                        flip_score, flip_score_detail = self._calc_position_score(flip_direction)

                                    if enable_degraded_fallback and flip_score >= degraded_min_score:
                                        direction = flip_direction
                                        side = OrderSide.LONG if direction == "LONG" else OrderSide.SHORT
                                        flip_triggered = True
                                        degraded_used = True
                                        flip_degraded_position_cap = degraded_position_cap
                                        chosen_fp = f"FLIP_FALLBACK_{flip_direction}"
                                        chosen_proto = None
                                        self._current_prototype = None
                                        self._current_template = None
                                        self.state.position_score = flip_score
                                        self.state.last_event = (
                                            f"🔄 降级翻转: {flip_direction} | flip_score={flip_score:.0f} "
                                            f"(阈值={degraded_min_score:.0f})"
                                        )
                                        self.state.decision_reason = (
                                            f"[翻转降级] 原始{entry_direction}位置评分不利，翻转{flip_direction}无匹配模板/原型；"
                                            f"但 flip_score={flip_score:.0f}>=阈值{degraded_min_score:.0f}，"
                                            f"启用小仓位降级入场(仓位上限={degraded_position_cap:.1%})。"
                                            f"原始分数={pos_score:.0f} | {score_detail} | flip细节={flip_score_detail}"
                                        )
                                        print(
                                            f"[LiveEngine] ⚠ 翻转降级入场: {flip_direction} | "
                                            f"flip_score={flip_score:.0f} | 仓位上限={degraded_position_cap:.1%}"
                                        )

                                    if not degraded_used:
                                        # 【修复】翻转失败时，回退检查原方向MACD是否支持
                                        # 避免"连坐拒绝"：原方向MACD支持时，用原方向小仓位入场
                                        orig_dir = direction
                                        orig_macd_ok, orig_macd_meta = self._eval_macd_trend_gate(
                                            self._df_buffer, orig_dir,
                                            market_regime=regime, position_score=pos_score,
                                        )
                                        
                                        if orig_macd_ok:
                                            # 原方向MACD支持，回退使用原方向（小仓位）
                                            flip_degraded_position_cap = degraded_position_cap  # 使用降级仓位
                                            self.state.last_event = (
                                                f"🔙 [翻转回退] {orig_dir} | 翻转失败但原方向MACD支持 | 小仓位入场"
                                            )
                                            self.state.decision_reason = (
                                                f"[翻转回退] 翻转{flip_direction}失败，但原方向{orig_dir}的MACD支持"
                                                f"(斜率={orig_macd_meta.get('slope', 0):+.3f})，"
                                                f"回退使用原方向，仓位上限={degraded_position_cap:.1%}。"
                                                f"原始分数={pos_score:.0f} | {score_detail}"
                                            )
                                            print(
                                                f"[LiveEngine] 🔙 翻转回退: {flip_direction}失败 → 回退{orig_dir} | "
                                                f"MACD斜率={orig_macd_meta.get('slope', 0):+.3f} | 小仓位={degraded_position_cap:.1%}"
                                            )
                                            # 不return，继续走后面的开仓流程
                                        else:
                                            # 原方向MACD也不支持，真正放弃
                                            blocked_reason = "模板兜底已禁用" if not enable_template_fallback else "无匹配模板/原型"
                                            reject_diag = self._fmt_reject_diag(
                                                candidate_dir=orig_dir,
                                                pos_score=pos_score,
                                                threshold=entry_pos_threshold,
                                                regime=regime,
                                                gate_stage="position_flip",
                                                fail_code="FLIP_NO_MATCH",
                                            )
                                            self.state.last_event = (
                                                f"[入场拒绝] {orig_dir}位置评分不利，翻转{flip_direction}{blocked_reason}，原方向MACD也不支持 | "
                                                f"{reject_diag}"
                                            )
                                            self.state.decision_reason = (
                                                f"[位置过滤] {orig_dir}位置评分危险，尝试翻转{flip_direction}但{blocked_reason}，"
                                                f"回退检查原方向MACD也不支持(斜率={orig_macd_meta.get('slope', 0):+.3f})。"
                                                f"degraded_fallback={'开' if enable_degraded_fallback else '关'}"
                                                f"{f', flip_score={flip_score:.0f}/{degraded_min_score:.0f}' if enable_degraded_fallback else ''}。"
                                                f"方向阈值={orig_dir}:{entry_pos_threshold} | 当前分数={pos_score:.0f} | "
                                                f"{score_detail} | {reject_diag}"
                                            )
                                            print(
                                                f"[LiveEngine] ⛔ 翻转失败+原方向MACD不支持: {flip_direction}无匹配，{orig_dir}斜率={orig_macd_meta.get('slope', 0):+.3f}，放弃入场"
                                            )
                                            # 记录拒绝（门控自适应学习）
                                            if self._rejection_tracker:
                                                self._rejection_tracker.record_rejection(
                                                    price=price,
                                                    direction=orig_dir,
                                                    fail_code="FLIP_NO_MATCH",
                                                    gate_stage="position_flip",
                                                    market_regime=regime,
                                                    bar_idx=self._current_bar_idx,
                                                    detail={
                                                        "pos_score": pos_score,
                                                        "threshold": entry_pos_threshold,
                                                        "flip_direction": flip_direction,
                                                        "orig_macd_slope": orig_macd_meta.get("slope", 0),
                                                        "similarity": similarity,
                                                        "fingerprint": chosen_fp,
                                                    },
                                                )
                                            # 【指纹3D图】更新多维相似度状态
                                            self._update_similarity_state(
                                                similarity, chosen_fp, chosen_match_result, chosen_proto
                                            )
                                            return
                            else:
                                # 【趋势市】不翻转，只拒绝（但现在有豁免逻辑，这里主要处理极端位置）
                                reject_diag = self._fmt_reject_diag(
                                    candidate_dir=direction,
                                    pos_score=pos_score,
                                    threshold=entry_pos_threshold,
                                    regime=regime,
                                    gate_stage="position_filter",
                                    fail_code="BLOCK_POS",
                                )
                                self.state.last_event = (
                                    f"[入场拒绝] 位置评分不利，趋势市{direction}谨慎 | {reject_diag}"
                                )
                                self.state.decision_reason = (
                                    f"[位置过滤] 趋势市中位置评分不利，{direction}风险过高。"
                                    f"方向阈值={direction}:{entry_pos_threshold} | 当前分数={pos_score:.0f} | "
                                    f"{score_detail} | {reject_diag}"
                                )
                                print(f"[LiveEngine] ⛔ 趋势市位置过滤: {direction}被拒 | {reject_diag}")
                                # 记录拒绝（门控自适应学习）
                                if self._rejection_tracker:
                                    self._rejection_tracker.record_rejection(
                                        price=price,
                                        direction=direction,
                                        fail_code="BLOCK_POS",
                                        gate_stage="position_filter",
                                        market_regime=regime,
                                        bar_idx=self._current_bar_idx,
                                        detail={
                                            "pos_score": pos_score,
                                            "threshold": entry_pos_threshold,
                                            "score_detail": score_detail,
                                            "similarity": similarity,
                                            "fingerprint": chosen_fp,
                                        },
                                    )
                                # 【指纹3D图】更新多维相似度状态
                                self._update_similarity_state(
                                    similarity, chosen_fp, chosen_match_result, chosen_proto
                                )
                                return
                
                # A. 检查指标闸门 (Aim 瞄准) — MACD必须通过，KDJ仅参考
                if not self._check_indicator_gate(self._df_buffer, direction):
                    if has_pending: 
                        # 如果MACD不再满足，撤掉之前的单子
                        self._paper_trader.cancel_entry_stop_orders()
                    kdj_hint = "✓" if self.state.kdj_ready else "✗"
                    macd_trend_diag = self._format_macd_trend_diag(self._df_buffer, direction=direction, window=5)
                    reject_diag = self._fmt_reject_diag(
                        candidate_dir=direction,
                        pos_score=self.state.position_score if self.state.position_score != 0 else None,
                        threshold=(
                            PAPER_TRADING_CONFIG.get("POS_THRESHOLD_LONG", -30) if direction == "LONG"
                            else PAPER_TRADING_CONFIG.get("POS_THRESHOLD_SHORT", -40)
                        ),
                        regime=self.state.market_regime,
                        gate_stage="indicator_gate",
                        fail_code="BLOCK_MACD",
                    )
                    self.state.decision_reason = (
                        f"[等待MACD] 指纹匹配成功({similarity:.1%}), 但 MACD 动能未对齐。"
                        f"(MACD={self.state.macd_ready}, KDJ={kdj_hint}参考) | {macd_trend_diag} | {reject_diag}"
                    )
                    self.state.last_event = (
                        f"[门控拒绝] MACD未通过 | KDJ={kdj_hint}(参考) | {macd_trend_diag} | {reject_diag}"
                    )
                    # 记录拒绝（门控自适应学习）
                    if self._rejection_tracker:
                        self._rejection_tracker.record_rejection(
                            price=price,
                            direction=direction,
                            fail_code="BLOCK_MACD",
                            gate_stage="indicator_gate",
                            market_regime=self.state.market_regime,
                            bar_idx=self._current_bar_idx,
                            detail={
                                "macd_ready": self.state.macd_ready,
                                "kdj_ready": self.state.kdj_ready,
                                "macd_trend_diag": macd_trend_diag,
                                "pos_score": self.state.position_score,
                                "similarity": similarity,
                                "fingerprint": chosen_fp,
                            },
                        )
                    # 【指纹3D图】更新多维相似度状态
                    self._update_similarity_state(
                        similarity, chosen_fp, chosen_match_result, chosen_proto
                    )
                    self._lock_entry_candidate(
                        direction=direction,
                        fingerprint=chosen_fp,
                        similarity=similarity,
                        source="prototype" if self.use_prototypes else "template",
                        prototype=chosen_proto if self.use_prototypes else None,
                        template=template if not self.use_prototypes else None,
                        stage="waiting_macd",
                        match_result=chosen_match_result,  # 【指纹3D图】传递匹配结果
                    )
                    return
                
                # B. 贝叶斯门控（可禁用，仅用于凯利数据收集）
                # 趋势市豁免：趋势延续逻辑不同于震荡反转，不适用震荡市的胜率统计
                is_trending_market = self.state.market_regime and any(
                    kw in str(self.state.market_regime) for kw in ["强多头", "弱多头", "强空头", "弱空头"]
                )
                bayes_gate_enabled = PAPER_TRADING_CONFIG.get("BAYESIAN_GATE_ENABLED", True)
                bayes_probe_enabled = PAPER_TRADING_CONFIG.get("BAYESIAN_PROBE_ENABLED", False)
                bayes_probe_position = PAPER_TRADING_CONFIG.get("BAYESIAN_PROBE_POSITION_PCT", 0.05)
                forced_position_pct = None
                if (
                    bayes_gate_enabled
                    and self._bayesian_enabled
                    and self._bayesian_filter
                    and not is_trending_market
                ):
                    should_trade, predicted_wr, bay_reason = self._bayesian_filter.should_trade(
                        prototype_fingerprint=chosen_fp,
                        market_regime=self.state.market_regime,
                    )
                    if not should_trade:
                        if bayes_probe_enabled:
                            forced_position_pct = float(bayes_probe_position or 0.0)
                            self.state.bayesian_win_rate = predicted_wr
                            self.state.last_event = f"[贝叶斯试探] {bay_reason} | 试探仓位 {forced_position_pct:.1%}"
                            self.state.decision_reason = (
                                f"[贝叶斯试探] 原型={chosen_fp} 市场={self.state.market_regime} | "
                                f"{bay_reason} | 试探仓位 {forced_position_pct:.1%}"
                            )
                            print(f"[LiveEngine] ⚠️ 贝叶斯试探放行: {chosen_fp} | {bay_reason} | 仓位={forced_position_pct:.1%}")
                        else:
                            reject_diag = self._fmt_reject_diag(
                                candidate_dir=direction,
                                pos_score=self.state.position_score if self.state.position_score != 0 else None,
                                threshold=None,
                                regime=self.state.market_regime,
                                gate_stage="bayesian_gate",
                                fail_code="BLOCK_BAYES",
                            )
                            self.state.last_event = f"[贝叶斯拒绝] {bay_reason} | {reject_diag}"
                            self.state.decision_reason = (
                                f"[贝叶斯过滤] 原型={chosen_fp} 市场={self.state.market_regime} | "
                                f"{bay_reason} | {reject_diag}"
                            )
                            # 【指纹3D图】更新多维相似度状态
                            self._update_similarity_state(
                                similarity, chosen_fp, chosen_match_result, chosen_proto
                            )
                            print(f"[LiveEngine] ⛔ 贝叶斯拒绝: {chosen_fp} | {bay_reason} | {reject_diag}")
                            # 记录拒绝（门控自适应学习）
                            if self._rejection_tracker:
                                self._rejection_tracker.record_rejection(
                                    price=price,
                                    direction=direction,
                                    fail_code="BLOCK_BAYES",
                                    gate_stage="bayesian_gate",
                                    market_regime=self.state.market_regime,
                                    bar_idx=self._current_bar_idx,
                                    detail={
                                        "predicted_wr": predicted_wr,
                                        "bay_reason": bay_reason,
                                        "pos_score": self.state.position_score,
                                        "similarity": similarity,
                                        "fingerprint": chosen_fp,
                                    },
                                )
                            return
                    else:
                        # 更新 state 中的贝叶斯胜率
                        self.state.bayesian_win_rate = predicted_wr
                        print(f"[LiveEngine] ✅ 贝叶斯通过: {chosen_fp} | {bay_reason}")
                elif is_trending_market:
                    print(f"[LiveEngine] 📈 趋势市豁免贝叶斯检查: {self.state.market_regime} | 趋势延续逻辑不同于震荡反转")
                
                # B2. 凯利仓位计算（根据贝叶斯预测的胜率和盈亏比）
                # 凯利公式改造：纯仓位管理（10%-80%），不再拒绝交易
                kelly_position_pct = None  # None = 使用默认仓位
                kelly_reason = ""
                from config import PAPER_TRADING_CONFIG as _ptc_kelly
                kelly_enabled = _ptc_kelly.get("KELLY_ENABLED", False)
                if kelly_enabled and self._bayesian_filter:
                    # 优先使用自适应控制器调整后的参数
                    if self._adaptive_controller and hasattr(self._adaptive_controller, 'kelly_adapter'):
                        kelly_params = self._adaptive_controller.kelly_adapter.get_current_parameters()
                        kelly_fraction = kelly_params.get("KELLY_FRACTION", _ptc_kelly.get("KELLY_FRACTION", 0.25))
                        kelly_max = kelly_params.get("KELLY_MAX", _ptc_kelly.get("KELLY_MAX_POSITION", 0.8))
                        kelly_min = kelly_params.get("KELLY_MIN", _ptc_kelly.get("KELLY_MIN_POSITION", 0.05))
                    else:
                        kelly_fraction = _ptc_kelly.get("KELLY_FRACTION", 0.25)
                        kelly_max = _ptc_kelly.get("KELLY_MAX_POSITION", 0.8)
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
                    
                    # 凯利公式改造：纯仓位管理，不拒绝交易
                    # 将凯利仓位限制在 [kelly_min, kelly_max] 范围内
                    kelly_position_pct = max(kelly_min, min(kelly_position_pct, kelly_max))
                    
                    if kelly_position_pct <= kelly_min:
                        kelly_reason = f"信号质量一般，使用最小仓位 {kelly_min:.1%}"
                        print(f"[LiveEngine] ⚠️ 凯利保守: {kelly_position_pct:.1%} | {kelly_reason}")
                    else:
                        print(f"[LiveEngine] 📊 凯利仓位: {kelly_position_pct:.1%} | {kelly_reason}")
                    
                    # 更新 state 中的凯利仓位
                    self.state.kelly_position_pct = kelly_position_pct

                # 贝叶斯低胜率试探：强制使用小仓位
                if forced_position_pct is not None:
                    kelly_position_pct = forced_position_pct
                    kelly_reason = f"贝叶斯低胜率试探，强制仓位 {kelly_position_pct:.1%}"
                    self.state.kelly_position_pct = kelly_position_pct

                # D. 所有门控通过后，再决定是否替换已有挂单，避免链路中途换轨
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
                    position_size_pct=(
                        min(kelly_position_pct, flip_degraded_position_cap)
                        if (kelly_position_pct is not None and flip_degraded_position_cap is not None)
                        else (flip_degraded_position_cap if flip_degraded_position_cap is not None else kelly_position_pct)
                    ),  # 凯利动态仓位 / 翻转降级仓位上限
                    # 【指纹3D图】从匹配结果中提取轨迹矩阵用于后续增量训练
                    entry_trajectory=chosen_match_result.get("entry_trajectory") if chosen_match_result else None,
                    regime_at_entry=self.state.market_regime,
                )
                
                print(f"[LiveEngine] 🎯 挂限价单入场: {direction} @ {limit_price:.2f} "
                      f"(当前价={price:.2f}, 需涨跌{abs(limit_price-price):.2f})")
                
                # 【指纹3D图】更新多维相似度状态
                self._update_similarity_state(
                    similarity, chosen_fp, chosen_match_result, chosen_proto
                )
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
                self._clear_entry_candidate("已挂单，候选锁完成使命")
                return
            else:
                self._clear_entry_candidate("未匹配到有效候选")
                # 如果当前没有匹配到任何符合门槛的信号，但手里还有挂单
                if self._paper_trader.has_pending_stop_orders():
                    print(f"[LiveEngine] 信号已失效或走势变坏，主动撤销挂单。")
                    self._paper_trader.cancel_entry_stop_orders()
                    # 【指纹3D图】清空相似度状态
                    self._clear_similarity_state()
                    self.state.matching_phase = "等待"
                    self.state.fingerprint_status = "待匹配"
                    self.state.decision_reason = "之前的指纹信号已消失或不再符合相似度要求，重回扫描模式。"
                    self.state.last_event = "[入场取消] 信号失效，撤销挂单"
                    return
                
                self.state.matching_phase = "等待"
                self.state.fingerprint_status = "未匹配"
                # 【指纹3D图】清空相似度状态
                self._clear_similarity_state()
                self.state.last_event = "[入场跳过] 未匹配到信号"
            
            # 没有匹配
            self.state.matching_phase = "等待"
            self.state.fingerprint_status = "未匹配"
            # 【指纹3D图】清空相似度状态
            self._clear_similarity_state()
            
            if self.use_prototypes:
                _l_sim = long_result.get("similarity", 0.0)
                _s_sim = short_result.get("similarity", 0.0)
                self.state.decision_reason = self._build_no_entry_reason(
                    regime=self.state.market_regime,
                    long_sim=_l_sim,
                    short_sim=_s_sim,
                    long_votes=long_result.get("vote_long", 0),
                    short_votes=short_result.get("vote_short", 0),
                    threshold=self.cosine_threshold,
                    min_agree=self.min_templates_agree,
                )
                # ── 近似信号追踪：捕获接近阈值但未达标的信号 ──
                if self._near_miss_tracker:
                    for _nm_dir, _nm_sim, _nm_res in [
                        ("LONG", _l_sim, long_result),
                        ("SHORT", _s_sim, short_result),
                    ]:
                        if _nm_sim > 0 and self._near_miss_tracker.is_near_miss(_nm_sim, self.cosine_threshold):
                            _nm_proto = _nm_res.get("best_prototype")
                            _nm_fp = ""
                            if _nm_proto:
                                _p_dir = getattr(_nm_proto, 'direction', '') or "?"
                                _p_id = getattr(_nm_proto, 'prototype_id', '') or "?"
                                _p_reg = (getattr(_nm_proto, 'regime', '') or "")[:2] or "?"
                                _nm_fp = f"proto_{_p_dir}_{_p_id}_{_p_reg}"
                            self._near_miss_tracker.record_near_miss(
                                price=kline.close,
                                direction=_nm_dir,
                                similarity=_nm_sim,
                                threshold=self.cosine_threshold,
                                market_regime=self.state.market_regime,
                                bar_idx=self._current_bar_idx,
                                fingerprint=_nm_fp,
                                detail={
                                    "votes": _nm_res.get(f"vote_{_nm_dir.lower()}", 0),
                                },
                            )
            else:
                _l_sim_t = long_result.dtw_similarity
                _s_sim_t = short_result.dtw_similarity
                self.state.decision_reason = self._build_no_entry_reason(
                    regime=self.state.market_regime,
                    long_sim=_l_sim_t,
                    short_sim=_s_sim_t,
                    threshold=self.cosine_threshold,
                    min_agree=self.min_templates_agree,
                )
                # ── 近似信号追踪（模板模式）──
                if self._near_miss_tracker:
                    for _nm_dir, _nm_sim, _nm_res in [
                        ("LONG", _l_sim_t, long_result),
                        ("SHORT", _s_sim_t, short_result),
                    ]:
                        if _nm_sim > 0 and self._near_miss_tracker.is_near_miss(_nm_sim, self.cosine_threshold):
                            _nm_tmpl = getattr(_nm_res, 'best_template', None)
                            _nm_fp = _nm_tmpl.fingerprint() if _nm_tmpl else ""
                            self._near_miss_tracker.record_near_miss(
                                price=kline.close,
                                direction=_nm_dir,
                                similarity=_nm_sim,
                                threshold=self.cosine_threshold,
                                market_regime=self.state.market_regime,
                                bar_idx=self._current_bar_idx,
                                fingerprint=_nm_fp,
                            )
            
        except Exception as e:
            print(f"[LiveEngine] 入场匹配失败: {e}")
            import traceback
            traceback.print_exc()

    def _eval_macd_trend_gate(self, df: pd.DataFrame, direction: str, window: int = None,
                               market_regime: str = None, position_score: float = None) -> Tuple[bool, Dict[str, Any]]:
        """
        统一 MACD 趋势门控（斜率法）。

        通过最近 `window` 根 `macd_hist` 做一阶线性回归，使用斜率判断趋势方向，
        并结合零轴位置冲突检查，返回可解释元信息供日志/UI复用。

        救援机制（slope_ok=True 但 zero_axis_ok=False 时）：
        - 零轴容忍度下限（MACD_ZERO_AXIS_FLOOR）：防止小斜率时容忍度过小
        - 震荡市零轴豁免（MACD_RANGE_BYPASS_ZERO_AXIS）：震荡市斜率方向正确即可
        - 高位置评分救援（MACD_POS_SCORE_RESCUE）：极佳位置补偿动能滞后
        """
        if window is None:
            window = int(PAPER_TRADING_CONFIG.get("MACD_TREND_WINDOW", 5))
        trend_meta: Dict[str, Any] = {
            "direction": direction,
            "window": int(window),
            "samples": 0,
            "slope": 0.0,
            "slope_min": float(PAPER_TRADING_CONFIG.get("MACD_SLOPE_MIN", 0.005)),
            "current_hist": None,
            "previous_hist": None,
            "zero_axis_allowance": 0.0,
            "zero_axis_ok": False,
            "zero_axis_conflict": True,
            "rescued": False,
            "rescue_reason": "",
            "reason": "",
        }

        if df is None:
            trend_meta["reason"] = "df为空"
            return False, trend_meta

        win = max(3, int(window))
        trend_meta["window"] = win
        if len(df) < win:
            trend_meta["reason"] = f"数据不足({len(df)}/{win})"
            return False, trend_meta

        if "macd_hist" not in df.columns:
            trend_meta["reason"] = "缺少macd_hist列"
            return False, trend_meta

        hist_series = pd.to_numeric(df["macd_hist"].iloc[-win:], errors="coerce")
        if hist_series.isna().any():
            trend_meta["reason"] = "macd_hist存在缺失值"
            return False, trend_meta

        y = hist_series.to_numpy(dtype=float)
        x = np.arange(win, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if win > 1 else 0.0
        curr_hist = float(y[-1])
        prev_hist = float(y[-2]) if win >= 2 else curr_hist
        slope_min = max(float(PAPER_TRADING_CONFIG.get("MACD_SLOPE_MIN", 0.005)), 1e-9)

        # 零轴容忍度：应用下限（防止小斜率时容忍度过小导致误拦）
        zero_axis_floor = max(float(PAPER_TRADING_CONFIG.get("MACD_ZERO_AXIS_FLOOR", 3.0)), 0.0)
        zero_axis_allowance = max(abs(slope) * win, zero_axis_floor)

        trend_meta["slope"] = slope
        trend_meta["slope_min"] = slope_min
        trend_meta["current_hist"] = curr_hist
        trend_meta["previous_hist"] = prev_hist
        trend_meta["samples"] = int(len(y))
        trend_meta["zero_axis_allowance"] = float(zero_axis_allowance)

        # ========== 趋势市特殊逻辑：放宽MACD检查（高位钝化不是反转） ==========
        # 当市场处于明确趋势时（强多/弱多/强空/弱空），不要求斜率加速，只看当前柱方向
        is_trending_market = market_regime and any(kw in str(market_regime) for kw in ["强多头", "弱多头", "强空头", "弱空头"])
        
        if direction == "LONG":
            if is_trending_market and market_regime and ("多头" in str(market_regime)):
                # 多头趋势市策略：
                # 1. 强多头：允许 MACD 适度回调（-20 以内），抓住回踩机会
                # 2. 弱多头：允许轻微回调（-10 以内）
                macd_pullback_tolerance = -20.0 if "强" in str(market_regime) else -10.0
                slope_ok = curr_hist > macd_pullback_tolerance
                zero_axis_ok = curr_hist >= -zero_axis_allowance
                macd_ok = slope_ok  # 趋势市不要求 zero_axis_ok
                trend_meta["trend_bypass"] = True
                print(f"[LiveEngine] 📈 多头趋势市MACD放宽: 当前柱={curr_hist:+.2f} > {macd_pullback_tolerance:.1f} "
                      f"({market_regime}) → {'✓' if macd_ok else '✗'}（斜率={slope:+.3f}可忽略）")
            else:
                slope_ok = slope > slope_min
                zero_axis_ok = curr_hist >= -zero_axis_allowance
                macd_ok = slope_ok and zero_axis_ok
        elif direction == "SHORT":
            if is_trending_market and market_regime and ("空头" in str(market_regime)):
                # 空头趋势市策略：
                # 1. 强空头：允许 MACD 适度反弹（+20 以内），抓住回踩机会
                # 2. 弱空头：允许轻微反弹（+10 以内）
                macd_pullback_tolerance = 20.0 if "强" in str(market_regime) else 10.0
                slope_ok = curr_hist < macd_pullback_tolerance
                zero_axis_ok = curr_hist <= zero_axis_allowance
                macd_ok = slope_ok  # 趋势市不要求 zero_axis_ok
                trend_meta["trend_bypass"] = True
                print(f"[LiveEngine] 📉 空头趋势市MACD放宽: 当前柱={curr_hist:+.2f} < {macd_pullback_tolerance:.1f} "
                      f"({market_regime}) → {'✓' if macd_ok else '✗'}（斜率={slope:+.3f}可忽略）")
            else:
                slope_ok = slope < -slope_min
                zero_axis_ok = curr_hist <= zero_axis_allowance
                macd_ok = slope_ok and zero_axis_ok
        else:
            trend_meta["reason"] = f"未知方向: {direction}"
            return False, trend_meta

        # ── 救援机制：斜率方向正确但零轴条件未满足 ──
        # 原理：斜率已确认动能方向，零轴滞后是因为 MACD 柱状图还在"追赶"
        # 注意：趋势市已在上面特殊处理，这里主要处理震荡市救援
        if slope_ok and not zero_axis_ok and not macd_ok and not trend_meta.get("trend_bypass", False):
            rescue_reasons = []

            # A. 震荡市零轴豁免：震荡市中 MACD 围绕零轴快速翻转，零轴检查过严
            range_bypass = bool(PAPER_TRADING_CONFIG.get("MACD_RANGE_BYPASS_ZERO_AXIS", True))
            if range_bypass and market_regime and "震荡" in str(market_regime):
                rescue_reasons.append("震荡市零轴豁免")

            # B. 高位置评分救援：极佳入场位置可以补偿动能的短暂滞后
            pos_rescue_threshold = float(PAPER_TRADING_CONFIG.get("MACD_POS_SCORE_RESCUE", 40))
            if position_score is not None and position_score >= pos_rescue_threshold:
                rescue_reasons.append(f"位置评分{position_score:.0f}≥{pos_rescue_threshold:.0f}")

            if rescue_reasons:
                macd_ok = True
                zero_axis_ok = True  # 标记为豁免通过
                trend_meta["rescued"] = True
                trend_meta["rescue_reason"] = " + ".join(rescue_reasons)

        trend_meta["zero_axis_ok"] = zero_axis_ok
        trend_meta["zero_axis_conflict"] = not zero_axis_ok
        if macd_ok:
            trend_meta["reason"] = "ok" if not trend_meta["rescued"] else f"ok(救援: {trend_meta['rescue_reason']})"
        else:
            trend_meta["reason"] = "斜率方向未对齐" if not slope_ok else "零轴位置冲突"
        return macd_ok, trend_meta
    
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
            regime = getattr(self.state, 'market_regime', None)
            pos_score = getattr(self.state, 'position_score', None)

            # 【冷启动自适应】冷启动且 MACD 豁免时，macd_ready 视为 True，便于 UI 显示"MACD已对齐"
            cold_start = getattr(self, "_cold_start_manager", None)
            cold_start_macd_bypass = (
                cold_start is not None and cold_start.enabled
                and COLD_START_CONFIG.get("MACD_BYPASS", True)
            )
            if cold_start_macd_bypass:
                self.state.macd_ready = True
            elif direction == "LONG":
                macd_ok, _ = self._eval_macd_trend_gate(
                    df, direction,
                    market_regime=regime, position_score=pos_score,
                )
                self.state.macd_ready = macd_ok
            else:  # SHORT
                macd_ok, _ = self._eval_macd_trend_gate(
                    df, direction,
                    market_regime=regime, position_score=pos_score,
                )
                self.state.macd_ready = macd_ok

            # KDJ 3根趋势判断（LONG/SHORT 共用逻辑，仅参考）
            if direction == "LONG":
                j_values = recent_3['j'].values
                d_values = recent_3['d'].values
                j_above_d_count = sum(j_values > d_values)
                j_trend_up = (j_values[-1] > j_values[0])
                self.state.kdj_ready = (j_above_d_count >= 2) and j_trend_up
            else:  # SHORT
                j_values = recent_3['j'].values
                d_values = recent_3['d'].values
                j_below_d_count = sum(j_values < d_values)
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
        # 冷启动阈值兜底（防止被其他流程覆盖）
        self._ensure_cold_start_thresholds()
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
                # 【指纹3D图】用于存储最佳匹配结果
                best_match_result = None
                best_proto = None
                
                if match_regime in BULL_REGIMES:
                    # 偏多市场：只看 LONG
                    lp = self._proto_matcher.match_entry(pre_entry_traj, direction="LONG", regime=match_regime)
                    long_sim = lp.get("similarity", 0.0)
                    if long_sim > 0 and lp.get("best_prototype"):
                        best_sim, best_dir = long_sim, "LONG"
                        p = lp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                        best_match_result = lp
                        best_proto = p
                elif match_regime in BEAR_REGIMES:
                    # 偏空市场：只看 SHORT
                    sp = self._proto_matcher.match_entry(pre_entry_traj, direction="SHORT", regime=match_regime)
                    short_sim = sp.get("similarity", 0.0)
                    if short_sim > 0 and sp.get("best_prototype"):
                        best_sim, best_dir = short_sim, "SHORT"
                        p = sp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                        best_match_result = sp
                        best_proto = p
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
                        best_match_result = lp
                        best_proto = p
                    elif short_sim > 0 and sp.get("best_prototype"):
                        best_sim, best_dir = short_sim, "SHORT"
                        p = sp.get("best_prototype")
                        best_fp = f"proto_{p.direction}_{p.prototype_id}" if p else ""
                        best_match_result = sp
                        best_proto = p
            
            # 【指纹3D图】更新多维相似度状态
            self._update_similarity_state(best_sim, best_fp, best_match_result, best_proto)
            
            # 【新增】实时决策说明
            if best_sim >= self.cosine_threshold:
                self.state.fingerprint_status = "匹配达标"
                # 检查指标状态（MACD是必要条件，KDJ仅参考）
                macd_ok = self.state.macd_ready
                kdj_ok = self.state.kdj_ready
                kdj_hint = "✓" if kdj_ok else "✗"
                
                # 空间位置评分 + 评分细项
                score_suffix = ""
                if self._df_buffer is not None and len(self._df_buffer) >= 20 and best_dir:
                    pos_score, score_detail = self._calc_position_score(best_dir)
                    self.state.position_score = pos_score
                    score_suffix = f" | 位置评分={pos_score:.0f} ({score_detail})" if score_detail else ""
                
                if macd_ok:
                    # MACD通过即可，KDJ仅参考
                    self.state.decision_reason = f"匹配成功({best_sim:.1%})，MACD已对齐(KDJ{kdj_hint}参考)。等待收线确认...{score_suffix}"
                else:
                    best_dir_text = best_dir if best_dir else "UNKNOWN"
                    macd_trend_diag = self._format_macd_trend_diag(self._df_buffer, direction=best_dir_text, window=5)
                    self.state.decision_reason = (
                        f"指纹匹配达标({best_sim:.1%})，正在等待 MACD 动能对齐(KDJ{kdj_hint}参考)。"
                        f"{score_suffix} | {macd_trend_diag}"
                    )
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

        - MACD：必须通过（一票否决权），并与 UI 的 `macd_ready` 完全复用同一门控函数
          【冷启动自适应】冷启动且启用 MACD_BYPASS 时跳过 MACD 趋势确认，便于启动学习
        - KDJ：仅参考，不拦截开仓
        - 传入 market_regime / position_score 以启用震荡市豁免和高评分救援

        Returns:
            True = MACD确认方向一致，允许开仓
        """
        if df is None or len(df) < 5:
            return False
            
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # 【冷启动自适应】冷启动时跳过 MACD 趋势确认，便于启动学习；关闭冷启动后 MACD 门控恢复
        cold_start = getattr(self, "_cold_start_manager", None)
        if (cold_start is not None and cold_start.enabled
                and COLD_START_CONFIG.get("MACD_BYPASS", True)):
            self.state.macd_ready = True  # 门控视为通过（冷启动豁免）
            if direction == "LONG":
                kdj_ok = (
                    ((curr['j'] >= curr['d']) or (curr['k'] >= curr['d'])) and
                    ((curr['j'] >= prev['j']) or (curr['k'] >= prev['k']))
                )
            else:
                kdj_ok = (
                    ((curr['j'] <= curr['d']) or (curr['k'] <= curr['d'])) and
                    ((curr['j'] <= prev['j']) or (curr['k'] <= prev['k']))
                )
            self.state.kdj_ready = kdj_ok
            return True

        # 获取上下文供 MACD 门控救援机制使用
        regime = getattr(self.state, 'market_regime', None)
        pos_score = getattr(self.state, 'position_score', None)
        
        if direction == "LONG":
            macd_ok, trend_meta = self._eval_macd_trend_gate(
                df, direction,
                market_regime=regime, position_score=pos_score,
            )
            self.state.macd_ready = macd_ok
            if trend_meta.get("rescued"):
                print(f"[LiveEngine] 🛟 MACD救援通过: {trend_meta['rescue_reason']} | "
                      f"slope={trend_meta['slope']:+.4f} hist={trend_meta['current_hist']:+.2f}")
            
            # KDJ 多头趋势：仅记录状态作为参考（不拦截开仓）
            kdj_ok = (
                ((curr['j'] >= curr['d']) or (curr['k'] >= curr['d'])) and
                ((curr['j'] >= prev['j']) or (curr['k'] >= prev['k']))
            )
            self.state.kdj_ready = kdj_ok
            
            return macd_ok
            
        elif direction == "SHORT":
            macd_ok, trend_meta = self._eval_macd_trend_gate(
                df, direction,
                market_regime=regime, position_score=pos_score,
            )
            self.state.macd_ready = macd_ok
            if trend_meta.get("rescued"):
                print(f"[LiveEngine] 🛟 MACD救援通过: {trend_meta['rescue_reason']} | "
                      f"slope={trend_meta['slope']:+.4f} hist={trend_meta['current_hist']:+.2f}")
            
            # KDJ 空头趋势：仅记录状态作为参考（不拦截开仓）
            kdj_ok = (
                ((curr['j'] <= curr['d']) or (curr['k'] <= curr['d'])) and
                ((curr['j'] <= prev['j']) or (curr['k'] <= prev['k']))
            )
            self.state.kdj_ready = kdj_ok
            
            return macd_ok
            
        return False

    @staticmethod
    def _format_macd_trend_diag(df: Optional[pd.DataFrame], direction: str, window: int = 5) -> str:
        """构建 MACD 趋势数值诊断，便于日志与 UI 解释拒绝原因。"""
        direction_map = {"LONG": "做多", "SHORT": "做空"}
        dir_text = direction_map.get(direction, direction)
        if df is None or "macd_hist" not in df.columns or len(df) == 0:
            return f"MACD趋势(方向={dir_text}, 窗口=NA, 斜率=NA, 当前柱=NA)"

        hist = pd.to_numeric(df["macd_hist"].tail(max(2, window)), errors="coerce").dropna()
        if len(hist) < 2:
            curr_hist = float(hist.iloc[-1]) if len(hist) == 1 else float("nan")
            curr_text = "NA" if np.isnan(curr_hist) else f"{curr_hist:+.6f}"
            return f"MACD趋势(方向={dir_text}, 窗口={len(hist)}, 斜率=NA, 当前柱={curr_text})"

        x = np.arange(len(hist), dtype=float)
        slope = float(np.polyfit(x, hist.to_numpy(dtype=float), 1)[0])
        curr_hist = float(hist.iloc[-1])
        return (
            f"MACD趋势(方向={dir_text}, 窗口={len(hist)}, 斜率={slope:+.6f}, "
            f"当前柱={curr_hist:+.6f})"
        )

    def _get_holding_indicator_summary(self, direction: str) -> dict:
        """
        持仓期当前K线的 KDJ/MACD/市场状态 摘要，用于持仓理由与警觉度。
        返回: kdj_j, kdj_d, kdj_trend(多/空/中性), macd_hist, macd_trend(多/空/中性),
              kdj_supports_position, macd_supports_position, regime_text
        """
        out = {
            "kdj_j": None, "kdj_d": None, "kdj_trend": "中性",
            "macd_hist": None, "macd_trend": "中性",
            "kdj_supports_position": True, "macd_supports_position": True,
            "regime_text": str(getattr(self.state, "market_regime", "") or "未知"),
        }
        df = getattr(self, "_df_buffer", None)
        if df is None or len(df) < 2:
            return out
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        # KDJ
        j_val = curr.get("j")
        d_val = curr.get("d")
        if j_val is not None and not (isinstance(j_val, float) and np.isnan(j_val)):
            out["kdj_j"] = float(j_val)
        if d_val is not None and not (isinstance(d_val, float) and np.isnan(d_val)):
            out["kdj_d"] = float(d_val)
        if out["kdj_j"] is not None and prev.get("j") is not None:
            j_prev = float(prev["j"])
            if out["kdj_j"] > j_prev:
                out["kdj_trend"] = "多头"
            elif out["kdj_j"] < j_prev:
                out["kdj_trend"] = "空头"
        # MACD
        hist = curr.get("macd_hist")
        if hist is not None and not (isinstance(hist, float) and np.isnan(hist)):
            out["macd_hist"] = float(hist)
        if out["macd_hist"] is not None and prev.get("macd_hist") is not None:
            h_prev = float(prev["macd_hist"])
            if out["macd_hist"] > h_prev:
                out["macd_trend"] = "多头"
            elif out["macd_hist"] < h_prev:
                out["macd_trend"] = "空头"
        # 与持仓方向是否一致
        if direction == "LONG":
            out["kdj_supports_position"] = out["kdj_trend"] != "空头"
            out["macd_supports_position"] = out["macd_trend"] != "空头"
        else:
            out["kdj_supports_position"] = out["kdj_trend"] != "多头"
            out["macd_supports_position"] = out["macd_trend"] != "多头"
        return out

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
        self._last_tpsl_atr = None
        # 持仓推理链与 DeepSeek 状态
        self.state.reasoning_result = None
        self.state.holding_regime_change = ""
        self.state.holding_exit_suggestion = ""
        self.state.tpsl_action = ""
        self.state.position_suggestion = ""
        self.state.deepseek_holding_advice = ""
        self.state.deepseek_judgement = ""
        self.state.deepseek_heartbeat = False
        self._clear_entry_candidate("持仓状态重置")
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
        order = self._paper_trader.current_position
        if order is None:
            return

        # 【自适应学习】每根K线存储指标快照（用于反事实分析）
        if self._df_buffer is not None and len(self._df_buffer) > 0:
            try:
                # 存储简化的指标快照（不需要完整的DecisionSnapshot对象）
                snapshot = {
                    'bar_idx': self._current_bar_idx,
                    'price': kline.close,
                    'high': kline.high,
                    'low': kline.low,
                    'kdj_j': self._df_buffer.iloc[-1].get('kdj_j', 0),
                    'macd_hist': self._df_buffer.iloc[-1].get('macd_hist', 0),
                    'rsi': self._df_buffer.iloc[-1].get('rsi_14', 50),
                    'atr': atr,
                }
                if not hasattr(order, 'indicator_snapshots'):
                    order.indicator_snapshots = []
                order.indicator_snapshots.append(snapshot)
            except Exception as e:
                pass  # 静默失败，不影响交易
        # 刚开仓到首次相似度巡检前，避免UI显示“未持仓/0%”
        if not self.state.hold_reason:
            self.state.hold_reason = "已开仓，等待下一次持仓相似度巡检。"
        if self.state.danger_level <= 0:
            default_danger = {"安全": 5.0, "警戒": 55.0, "危险": 80.0, "脱轨": 100.0}
            self.state.danger_level = default_danger.get(order.tracking_status, 5.0)
        if not self.state.exit_reason:
            self.state.exit_reason = "形态配合良好，暂无平仓预兆。"
        
        # 【新增】更新详细的离场/持有说明（含空间位置评分细项 + 止损层级监控）
        pnl_pct = order.profit_pct
        score_suffix = ""
        if self._df_buffer is not None and len(self._df_buffer) >= 20:
            pos_score, score_detail = self._calc_position_score(order.side.value)
            self.state.position_score = pos_score
            score_suffix = f" | 位置评分={pos_score:.0f} ({score_detail})" if score_detail else ""
        
        # 【止损层级监控】显示当前距离各档止损的距离
        sl_monitor = self._get_stop_loss_monitor(order, pnl_pct)
        
        self.state.decision_reason = f"[持仓中] {order.side.value} | 相似度={order.current_similarity:.1%} | 收益={pnl_pct:+.2f}%{score_suffix}{sl_monitor}"
        
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
        # 【思维链 1～4 + TradeReasoning】持仓推理链驱动 UI
        # 链环1: 市场状态变化  链环2: 止盈建议  链环3: TP/SL动作  链环4: 仓位建议
        # ══════════════════════════════════════════════════════════
        self._update_holding_reasoning_chain(order)
        
        # 仅保留分段止盈/分段止损（5%、10%），已移除：保护期紧急止损、保护期收紧、市场反转、位置翻转

        # ══════════════════════════════════════════════════════════
        # 持仓思维链 1：市场状态有没有变化
        # ══════════════════════════════════════════════════════════
        regime_at_entry = getattr(order, "regime_at_entry", "") or "未知"
        current_regime = self.state.market_regime
        regime_change = self._classify_holding_regime_change(regime_at_entry, current_regime, order.side)
        self.state.holding_regime_change = regime_change

        # ══════════════════════════════════════════════════════════
        # 持仓思维链 3：TP/SL 要不要重算或继续等（反转/ATR 变化→重算，弱化→仅收紧 SL）
        # ══════════════════════════════════════════════════════════
        # 首根持仓 K 线：记录 ATR 基线，供后续 ATR 变化重算使用
        if order.hold_bars == 1 and atr > 0:
            self._last_tpsl_atr = atr
        atr_change_pct = 0.0
        if self._last_tpsl_atr is not None and self._last_tpsl_atr > 0:
            atr_change_pct = abs(atr - self._last_tpsl_atr) / self._last_tpsl_atr
        atr_changed_significantly = atr_change_pct >= PAPER_TRADING_CONFIG.get("TPSL_ATR_CHANGE_RECALC_PCT", 0.20)

        if order.hold_bars >= 3:
            if regime_change == "反转" or (atr_changed_significantly and regime_change != "弱化·震荡"):
                self.state.tpsl_action = "recalc"
                if self._current_prototype is not None and atr > 0:
                    direction = "LONG" if order.side == OrderSide.LONG else "SHORT"
                    new_tp, new_sl = self._calculate_dynamic_tp_sl(
                        entry_price=order.entry_price,
                        direction=direction,
                        prototype=self._current_prototype,
                        atr=atr,
                    )
                    order.take_profit = new_tp
                    order.stop_loss = new_sl
                    self._last_tpsl_atr = atr
                    reason = "市场反转" if regime_change == "反转" else f"ATR变化{atr_change_pct:.0%}"
                    print(f"[LiveEngine] {reason} → 重算 TP/SL: TP={new_tp:.2f}, SL={new_sl:.2f}")
            elif regime_change == "弱化·震荡":
                # 已移除「弱化·震荡：收紧至保本」逻辑，不再把止损挪到入场价
                self.state.tpsl_action = "hold"
            else:
                self.state.tpsl_action = "hold"
        else:
            self.state.tpsl_action = "hold"

        # 分段止损（亏损时分批减仓）、分段止盈（盈利时分批减仓）
        self._check_staged_partial_sl(kline)
        self._check_staged_partial_tp(kline)

        # 已移除动量衰减离场/收紧，仅保留分段止盈与分段止损
        # 已移除市场恶化减仓（用户不需要）

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

        # 链环 2：统一止盈决策层（综合 regime/相似度/盈亏，覆盖 reasoning_chain 的结论，因 regime 已由链环 1 更新）
        exit_suggestion, position_suggestion = self._compute_holding_exit_suggestion(order)
        self.state.holding_exit_suggestion = exit_suggestion
        self.state.holding_position_suggestion = position_suggestion
        self.state.position_suggestion = position_suggestion

    def _update_holding_reasoning_chain(self, order):
        """
        更新持仓推理链（思维链1～4）并调用 TradeReasoning
        
        链环1: 市场状态变化 (一致/弱化·震荡/反转)
        链环2: 止盈建议 (继续持有/部分止盈/仅收紧止损/准备离场/立即离场)
        链环3: TP/SL 动作 (hold/recalc/tighten_sl_only)
        链环4: 仓位建议 (维持/建议减仓)
        """
        regime_at_entry = getattr(order, 'regime_at_entry', '未知')
        current_regime = self.state.market_regime
        
        # 链环1: 市场状态变化
        _BULL = {"强多头", "弱多头", "震荡偏多"}
        _BEAR = {"强空头", "弱空头", "震荡偏空"}
        if regime_at_entry == current_regime:
            self.state.holding_regime_change = "一致"
        elif current_regime == "震荡":
            self.state.holding_regime_change = "弱化·震荡"
        elif (order.side == OrderSide.LONG and current_regime in _BEAR) or \
             (order.side == OrderSide.SHORT and current_regime in _BULL):
            self.state.holding_regime_change = "反转"
        else:
            self.state.holding_regime_change = "弱化·震荡"
        
        # 链环2: 止盈建议（综合 regime/相似度/盈亏/动量 → verdict 映射）
        sim = getattr(order, 'current_similarity', 0.0) or 0.0
        profit_pct = getattr(order, 'profit_pct', 0.0) or 0.0
        peak_pct = getattr(order, 'peak_profit_pct', 0.0) or 0.0
        drawdown_from_peak = max(0, peak_pct - profit_pct) if peak_pct > 0 else 0
        
        if self.state.holding_regime_change == "反转" and profit_pct >= -0.5:
            self.state.holding_exit_suggestion = "准备离场"
        elif sim < 0.3:
            self.state.holding_exit_suggestion = "仅收紧止损"
        elif self.state.holding_regime_change in ("弱化·震荡", "反转") and profit_pct >= 1.0:
            self.state.holding_exit_suggestion = "部分止盈"
        elif self.state.holding_regime_change in ("弱化·震荡", "反转"):
            self.state.holding_exit_suggestion = "仅收紧止损"
        elif drawdown_from_peak >= peak_pct * 0.5 and peak_pct >= 1.5:
            self.state.holding_exit_suggestion = "仅收紧止损"
        else:
            self.state.holding_exit_suggestion = "继续持有"
        
        # 链环3: TP/SL 动作
        if self.state.holding_regime_change == "反转":
            self.state.tpsl_action = "recalc"
        elif self.state.holding_regime_change == "弱化·震荡":
            self.state.tpsl_action = "tighten_sl_only"
        else:
            self.state.tpsl_action = "hold"
        
        # 链环4: 仓位建议
        if self.state.holding_regime_change in ("弱化·震荡", "反转") and profit_pct >= 1.0:
            self.state.position_suggestion = "建议减仓"
        else:
            self.state.position_suggestion = "维持"
        
        # 调用 TradeReasoning 并写入 state.reasoning_result
        if self._df_buffer is not None and len(self._df_buffer) > 0:
            try:
                tr = TradeReasoning()
                rr = tr.analyze(order, self._df_buffer, self.state)
                self.state.reasoning_result = rr
            except Exception as e:
                self.state.reasoning_result = None
    
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
                        self._last_tpsl_atr = atr
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

            # 持仓期 KDJ/MACD 摘要（用于详细持仓理由与警觉度）
            ind = self._get_holding_indicator_summary(direction)
            status_map = {"安全": "形态配合完美", "警戒": "形态轻微偏离"}
            hold_desc = status_map.get(order.tracking_status, "形态匹配中")
            # 构建详细持仓理由：形态 + KDJ + MACD + 市场状态，并注明实时/秒级
            parts = [
                f"【形态】相似度 {similarity:.1%} ≥ 警戒线 {self.hold_alert_threshold:.1%}，{hold_desc}。",
            ]
            if ind["kdj_j"] is not None:
                kdj_s = f"J={ind['kdj_j']:.0f}" + (f" D={ind['kdj_d']:.0f}" if ind["kdj_d"] is not None else "")
                parts.append(f"【KDJ】{kdj_s} 趋势{ind['kdj_trend']}" + ("✓" if ind["kdj_supports_position"] else "⚠背离"))
            if ind["macd_hist"] is not None:
                parts.append(f"【MACD】柱={ind['macd_hist']:+.2f} 趋势{ind['macd_trend']}" + ("✓" if ind["macd_supports_position"] else "⚠背离"))
            parts.append(f"【市场】{ind['regime_text']}。")
            parts.append("按最新K线实时更新（秒级数据时逐秒刷新）。")
            self.state.hold_reason = " ".join(parts)

            # 持仓警觉度：趋势匹配 + KDJ匹配 + MACD匹配，三者综合
            danger_trend = max(0.0, (1.0 - similarity) / max(1e-6, 1.0 - self.hold_derail_threshold)) * 100
            danger_kdj = 0.0 if ind["kdj_supports_position"] else 20.0
            danger_macd = 0.0 if ind["macd_supports_position"] else 20.0
            danger = danger_trend + danger_kdj + danger_macd
            self.state.danger_level = min(100.0, danger)
            
            # 如果没有更具体的出场预估，使用默认
            if not self.state.exit_reason or similarity < self.hold_safe_threshold:
                if similarity < self.hold_safe_threshold:
                    self.state.exit_reason = f"相似度下降 ({similarity:.1%})，持仓进入警戒区，TP/SL硬保护继续生效。"
                else:
                    self.state.exit_reason = "形态配合良好，暂无平仓预兆。"
            
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
        from config import PAPER_TRADING_CONFIG
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
        atr_multiplier = PAPER_TRADING_CONFIG.get("ATR_SL_MULTIPLIER", 3.0)
        atr_based_sl_pct = (atr / entry_price) * atr_multiplier
        
        # ========== 因子3: 固定百分比下限（避免噪声止损）==========
        # BTC 1分钟线，至少 0.5% 距离（约 $475），挡住正常波动
        min_fixed_pct = PAPER_TRADING_CONFIG.get("MIN_SL_PCT", 0.005)
        
        # ========== 风险收益比（基于胜率）==========
        min_rr = float(PAPER_TRADING_CONFIG.get("MIN_RR_RATIO", 1.4))
        if win_rate >= 0.70:
            risk_reward_ratio = min_rr * 1.2  # 高胜率加成 20%
        else:
            risk_reward_ratio = min_rr
        
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
        
        # ========== 位置评分微调：位置极佳时 TP 拉远（取消 SL 收紧，避免破坏最小止损保护）==========
        pos_score, _ = self._calc_position_score(direction)
        if pos_score > 40:
            # TP 距离拉远 10%
            if direction == "LONG":
                take_profit = entry_price + (take_profit - entry_price) * 1.1
            else:
                take_profit = entry_price - (entry_price - take_profit) * 1.1
            print(f"[LiveEngine] 位置评分{pos_score:.0f}>40，TP拉远10%（SL保持不变，维持最小保护）")
        
        # ========== 最终安全检查：确保止损距离不低于最小值（防止任何微调破坏保护）==========
        final_sl_distance = abs(stop_loss - entry_price)
        min_sl_distance = entry_price * min_fixed_pct
        if final_sl_distance < min_sl_distance:
            print(f"[LiveEngine] ⚠️ 最终SL过小({final_sl_distance:.2f})，强制修正到 {min_sl_distance:.2f}")
            if direction == "LONG":
                stop_loss = entry_price * (1 - min_fixed_pct)
            else:
                stop_loss = entry_price * (1 + min_fixed_pct)
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
        try:
            # 入场后再次确认市场状态并回填（避免 entry 时状态漂移）
            if getattr(order, "status", None) == OrderStatus.FILLED:
                if getattr(order, "regime_at_entry", "") in ("", "未知", None):
                    current_regime = self._confirm_market_regime()
                    if current_regime and current_regime != "未知":
                        order.regime_at_entry = current_regime
                # 首次成交时打标信号组合（仅在列表为空时写入，避免重复）
                if not getattr(order, 'signal_combo_keys', None):
                    order.signal_combo_keys = list(self._pending_signal_combos)
        except Exception as e:
            print(f"[LiveEngine] 入场状态回填失败: {e}")
    
    def _on_trade_closed_internal(self, order: PaperOrder):
        """交易关闭内部回调 — 安全网，确保任何平仓路径都能清理状态"""

        # 【信号组合实盘命中率】记录本次交易结果到 signal_store
        try:
            combo_keys = getattr(order, 'signal_combo_keys', None) or []
            if combo_keys:
                from core import signal_store
                hit = (getattr(order, 'profit_pct', 0.0) or 0.0) > 0
                for _key in combo_keys:
                    signal_store.record_live_result(_key, hit)
        except Exception as _e:
            print(f"[LiveEngine] signal_store.record_live_result 失败: {_e}")

        # 【自适应学习】捕获出场决策快照
        if self._adaptive_controller and self._df_buffer is not None and len(self._df_buffer) > 0:
            try:
                from core.adaptive_controller import DecisionSnapshot
                exit_snapshot = DecisionSnapshot.from_dataframe(
                    self._df_buffer,
                    bar_idx=min(self._current_bar_idx, len(self._df_buffer) - 1),
                    market_regime=self.state.market_regime,
                    similarity=order.current_similarity if hasattr(order, 'current_similarity') else 0.0
                )
                order.exit_snapshot = exit_snapshot
            except Exception as e:
                print(f"[LiveEngine] 捕获出场快照失败: {e}")
        
        # 【自适应控制器】记录交易关闭（使用简化版）
        if self._adaptive_controller:
            try:
                # 提取市场状态（从entry_reason）
                market_regime = "未知"
                entry_reason = getattr(order, 'entry_reason', '')
                if "市场=" in entry_reason:
                    try:
                        market_regime = entry_reason.split("市场=")[1].split("|")[0].strip()
                    except:
                        pass
                
                self._adaptive_controller.on_trade_closed_simple(order, market_regime)
                
                # 更新杠杆到交易器（仅 LEVERAGE_ADAPTIVE=True 时）
                from config import PAPER_TRADING_CONFIG as _ptc_lev
                if _ptc_lev.get("LEVERAGE_ADAPTIVE", False):
                    if self._adaptive_controller.kelly_adapter:
                        new_leverage = self._adaptive_controller.kelly_adapter.leverage
                        if new_leverage and new_leverage != self._paper_trader.leverage:
                            old_leverage = self._paper_trader.leverage
                            try:
                                self._paper_trader.set_leverage(int(new_leverage))
                                print(f"[LiveEngine] 杠杆已调整: {old_leverage}x -> {new_leverage}x")
                            except Exception as e:
                                print(f"[LiveEngine] 更新杠杆失败: {e}")
                else:
                    # 固定杠杆模式：确保交易所与配置一致，平仓后重新同步
                    cfg_leverage = int(_ptc_lev.get("LEVERAGE_DEFAULT", 20))
                    if self._paper_trader.leverage != cfg_leverage:
                        try:
                            self._paper_trader.set_leverage(cfg_leverage)
                            print(f"[LiveEngine] 固定杠杆同步: {cfg_leverage}x")
                        except Exception as e:
                            print(f"[LiveEngine] 同步固定杠杆失败: {e}")
            except Exception as e:
                print(f"[LiveEngine] 自适应控制器记录失败: {e}")
        
        # 【DeepSeek AI 复盘】异步添加到复盘队列
        if self._deepseek_reviewer and self._deepseek_reviewer.enabled:
            try:
                from core.deepseek_reviewer import TradeContext
                
                # 获取反事实分析结果（从自适应控制器获取）
                cf_result = None
                if self._adaptive_controller and hasattr(self._adaptive_controller, 'get_counterfactual_result'):
                    try:
                        cf_result = self._adaptive_controller.get_counterfactual_result(order.order_id)
                    except:
                        pass
                
                # 获取原型历史表现
                proto_stats = None
                if self.prototype_library:
                    proto_fp = getattr(order, 'template_fingerprint', None)
                    if proto_fp:
                        try:
                            # 从原型库获取该原型的历史表现统计
                            if hasattr(self.prototype_library, 'get_prototype_stats'):
                                proto_stats = self.prototype_library.get_prototype_stats(proto_fp)
                        except:
                            pass
                
                # 获取特征模式统计（从自适应控制器的特征数据库）
                feature_patterns = None
                if self._adaptive_controller and hasattr(self._adaptive_controller, 'feature_db'):
                    try:
                        feature_patterns = self._adaptive_controller.feature_db.get_profitable_ranges()
                    except:
                        pass
                
                trade_ctx = TradeContext.from_order(
                    order,
                    counterfactual_result=cf_result,
                    prototype_stats=proto_stats,
                    feature_patterns=feature_patterns,
                    position_pct=getattr(order, 'position_size_pct', self.fixed_position_size_pct),
                )
                self._deepseek_reviewer.add_trade_for_review(trade_ctx)
            except Exception as e:
                print(f"[LiveEngine] DeepSeek复盘添加失败: {e}")
        
        # 【贝叶斯更新】用实盘交易结果更新 Beta 分布
        if self._bayesian_enabled and self._bayesian_filter:
            # 提取原型指纹和市场状态
            proto_fp = getattr(order, 'template_fingerprint', None)
            # 优先使用订单记录的入场市场状态，其次从 entry_reason 回退解析
            market_regime = getattr(order, "regime_at_entry", "") or "未知"
            if market_regime == "未知":
                entry_reason = getattr(order, 'entry_reason', '')
                if "市场=" in entry_reason:
                    # 从 "[开仓] 市场=强空头 | SHORT | ..." 中提取
                    try:
                        market_regime = entry_reason.split("市场=")[1].split("|")[0].strip()
                    except Exception:
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

        # ── 自适应学习：出场时机 / 止盈止损 / 早期出场 ──
        exit_price = getattr(order, "exit_price", None) or getattr(order, "entry_price", 0.0)
        direction = order.side.value if getattr(order, "side", None) else "LONG"
        bar_idx = getattr(order, "exit_bar_idx", None) or getattr(order, "entry_bar_idx", 0)
        decision_reason = getattr(order, "decision_reason", "") or ""
        close_reason = getattr(order, "close_reason", None)

        if self._exit_timing_tracker and close_reason:
            # 提取市场状态
            exit_market_regime = getattr(self.state, "market_regime", "未知")
            self._exit_timing_tracker.record_exit(
                direction=direction,
                close_reason=close_reason.value,  # CloseReason enum → 中文字符串
                entry_price=order.entry_price,
                exit_price=exit_price,
                profit_pct=order.profit_pct,
                peak_profit_pct=order.peak_profit_pct,
                hold_bars=order.hold_bars,
                trailing_stage=getattr(order, 'trailing_stage', 0),
                market_regime=exit_market_regime,
                bar_idx=bar_idx,
                template_fingerprint=getattr(order, 'template_fingerprint', '') or '',
            )

        if self._tpsl_tracker and close_reason:
            # 只记录 TP/SL/追踪止损 相关的平仓；EXCHANGE_CLOSE 时按价格推断后也参与学习
            tpsl_reason_map = {
                CloseReason.STOP_LOSS: "STOP_LOSS",
                CloseReason.TAKE_PROFIT: "TAKE_PROFIT",
                CloseReason.TRAILING_STOP: "TRAILING_STOP",
            }
            tpsl_reason = tpsl_reason_map.get(close_reason)
            if not tpsl_reason and close_reason == CloseReason.EXCHANGE_CLOSE:
                # 交易所平仓：按出场价与 TP/SL 距离推断，纳入 TP/SL 学习
                sl_price = getattr(order, 'stop_loss', 0.0) or 0.0
                tp_price = getattr(order, 'take_profit', 0.0) or 0.0
                if tp_price and sl_price:
                    dist_tp = abs(exit_price - tp_price) / tp_price if tp_price else 1.0
                    dist_sl = abs(exit_price - sl_price) / sl_price if sl_price else 1.0
                    is_long = getattr(order, 'side', None) and getattr(order.side, 'value', '') == 'LONG'
                    sl_in_profit = (is_long and sl_price >= order.entry_price) or (not is_long and sl_price <= order.entry_price)
                    if dist_tp <= dist_sl:
                        tpsl_reason = "TAKE_PROFIT"
                    else:
                        tpsl_reason = "TRAILING_STOP" if sl_in_profit else "STOP_LOSS"
            if tpsl_reason:
                entry_atr = getattr(order, 'entry_atr', 0.0)
                sl_price = getattr(order, 'stop_loss', 0.0) or 0.0
                tp_price = getattr(order, 'take_profit', 0.0) or 0.0
                original_sl = getattr(order, 'original_stop_loss', 0.0) or 0.0
                self._tpsl_tracker.record_exit(
                    direction=direction,
                    exit_price=exit_price,
                    bar_idx=bar_idx,
                    reason=tpsl_reason,
                    entry_price=order.entry_price,
                    entry_atr=entry_atr,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    profit_pct=order.profit_pct,
                    peak_profit_pct=order.peak_profit_pct,
                    hold_bars=order.hold_bars,
                    trailing_stage=getattr(order, 'trailing_stage', 0),
                    detail={
                        "close_reason": close_reason.name,
                        "original_sl": original_sl,
                        "template_fingerprint": getattr(order, 'template_fingerprint', '') or '',
                        "market_regime": getattr(self.state, "market_regime", "未知"),
                    },
                )

        if self._early_exit_tracker and close_reason == CloseReason.STOP_LOSS:
            if "紧急止损" in decision_reason:
                self._early_exit_tracker.record_early_exit(
                    direction=direction,
                    exit_price=exit_price,
                    bar_idx=bar_idx,
                    detail={"reason": decision_reason},
                )
        
        # 冷启动系统 - 记录交易（频率统计 + 结果反馈，供自动毕业判断）
        if self._cold_start_manager:
            self._cold_start_manager.record_trade(profit_pct=order.profit_pct)
            self._sync_cold_start_state()
        
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

                        # 短期趋势修正：允许同向增强 + 对立反转
                        # 条件放宽：只需 1 票 + 明确的短期趋势
                        if bull_votes >= 1 and short_trend > threshold:
                            if short_trend > threshold * 3:  # 强趋势 > 0.45%
                                if regime in (MarketRegime.RANGE_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.STRONG_BEAR, MarketRegime.UNKNOWN):
                                    return MarketRegime.WEAK_BULL
                                elif regime == MarketRegime.RANGE_BULL:
                                    return MarketRegime.WEAK_BULL  # 同向增强
                            elif short_trend > threshold * 1.5:  # 中等趋势 > 0.225%
                                if regime in (MarketRegime.RANGE_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.STRONG_BEAR, MarketRegime.UNKNOWN):
                                    return MarketRegime.RANGE_BULL

                        if bear_votes >= 1 and short_trend < -threshold:
                            if short_trend < -threshold * 3:  # 强趋势 < -0.45%
                                if regime in (MarketRegime.RANGE_BULL, MarketRegime.WEAK_BULL, MarketRegime.STRONG_BULL, MarketRegime.UNKNOWN):
                                    return MarketRegime.WEAK_BEAR
                                elif regime == MarketRegime.RANGE_BEAR:
                                    return MarketRegime.WEAK_BEAR  # 同向增强
                            elif short_trend < -threshold * 1.5:  # 中等趋势 < -0.225%
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
        市场状态确认机制：连续2根K线保持同向才切换状态
        
        目的：避免震荡市中市场状态频繁切换（如：震荡偏多 ↔ 震荡偏空）
        原理：维护最近2根K线的市场状态判断历史，只有两根完全一致时才正式切换
        
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
        
        # 2. 更新历史队列（保持最近2根）
        self._regime_history.append(current_raw_regime)
        if len(self._regime_history) > 2:
            self._regime_history.pop(0)
        
        # 3. 确认逻辑：连续2根完全相同才切换
        if len(self._regime_history) == 2:
            # 检查两根是否完全一致
            if self._regime_history[0] == self._regime_history[1]:
                # 两根一致，正式切换状态
                old_regime = self._confirmed_regime
                self._confirmed_regime = current_raw_regime
                
                # 只在状态真正发生变化时输出日志，避免刷屏
                if old_regime != self._confirmed_regime:
                    print(f"[市场状态确认] 连续2根确认: {old_regime} → {self._confirmed_regime}")
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
                        f"[市场状态待确认] 最近2根: {self._regime_history} | 保持: {self._confirmed_regime}",
                        interval=30.0  # 30秒打印一次，避免刷屏
                    )
        else:
            # 启动阶段（少于2根历史），直接使用原始判断
            if self._confirmed_regime is None:
                self._confirmed_regime = current_raw_regime
                print(f"[市场状态确认] 启动阶段初始化: {self._confirmed_regime}")
        
        return self._confirmed_regime

    def _classify_holding_regime_change(self, regime_at_entry: str, current_regime, side) -> str:
        """
        持仓思维链 1：比较入场时与当前市场状态，得到 一致 / 弱化·震荡 / 反转。
        用于驱动后续止盈建议与 TP/SL 动作。
        """
        _BULL = {"强多头", "弱多头", "震荡偏多"}
        _BEAR = {"强空头", "弱空头", "震荡偏空"}
        _RANGE = {"震荡偏多", "震荡偏空"}

        def _norm(r):
            if r is None:
                return ""
            return getattr(r, "value", r) if hasattr(r, "value") else str(r)

        entry = _norm(regime_at_entry)
        curr = _norm(current_regime)
        if not entry or not curr or curr == "未知":
            return "一致"

        is_long = (side == OrderSide.LONG) if hasattr(side, "value") else (side == "LONG")
        curr_bull = curr in _BULL
        curr_bear = curr in _BEAR
        entry_bull = entry in _BULL
        entry_bear = entry in _BEAR

        # 反转：持仓方向与当前状态相反
        if is_long and curr_bear:
            return "反转"
        if not is_long and curr_bull:
            return "反转"

        # 弱化·震荡：同向但由强/弱进入震荡，或趋势减弱
        if curr in _RANGE:
            return "弱化·震荡"
        if entry_bull and curr_bull and (
            (entry == "强多头" and curr != "强多头") or (entry == "弱多头" and curr == "震荡偏多")
        ):
            return "弱化·震荡"
        if entry_bear and curr_bear and (
            (entry == "强空头" and curr != "强空头") or (entry == "弱空头" and curr == "震荡偏空")
        ):
            return "弱化·震荡"

        return "一致"

    def _compute_holding_exit_suggestion(self, order) -> Tuple[str, str]:
        """
        持仓思维链 2：是否先止盈。综合 regime 变化、相似度、盈亏、峰值回撤，
        输出 继续持有/部分止盈/仅收紧止损/准备离场/立即离场；以及仓位建议 维持/建议减仓。
        """
        regime_change = getattr(self.state, "holding_regime_change", "") or "一致"
        sim = getattr(order, "current_similarity", None)
        if sim is None:
            sim = getattr(self.state, "best_match_similarity", 0.7)
        profit_pct = getattr(order, "profit_pct", 0.0)
        peak_pct = getattr(order, "peak_profit_pct", 0.0)
        status = getattr(order, "tracking_status", "安全")

        # 相似度极低（脱轨区）→ 仅显示警告，平仓由阶梯TP/SL硬保护负责，不强制软件平仓
        if status == "脱轨" or sim < self.hold_derail_threshold:
            return "仅收紧止损", "建议减仓"

        # 反转 + 有盈利 → 部分止盈或准备离场
        if regime_change == "反转":
            if profit_pct >= 1.0:
                return "部分止盈", "建议减仓"
            return "准备离场", "建议减仓"

        # 弱化·震荡 + 盈利达阶梯 → 部分止盈
        staged_tp2 = PAPER_TRADING_CONFIG.get("STAGED_TP_2_PCT", 10.0)
        if regime_change == "弱化·震荡" and peak_pct >= staged_tp2 and profit_pct >= 0.5:
            return "部分止盈", "建议减仓"
        if regime_change == "弱化·震荡":
            return "仅收紧止损", "维持"

        # 警戒区 + 已有一定盈利 → 仅收紧止损
        if status == "警戒" and profit_pct >= 1.0:
            return "仅收紧止损", "维持"

        # 峰值回撤较大（如超 50%）且仍有盈利 → 仅收紧止损
        if peak_pct >= 1.5 and peak_pct > 0:
            retrace = (peak_pct - profit_pct) / peak_pct if peak_pct > 0 else 0
            if retrace >= 0.5 and profit_pct >= 0.3:
                return "仅收紧止损", "维持"

        return "继续持有", "维持"

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

    @staticmethod
    def _fmt_reject_diag(candidate_dir: str,
                         gate_stage: str,
                         fail_code: str,
                         pos_score: Optional[float] = None,
                         threshold: Optional[float] = None,
                         regime: Optional[str] = None) -> str:
        """统一入场拒绝诊断字段，便于日志检索与复盘。"""
        pos_score_text = "NA" if pos_score is None else f"{pos_score:.0f}"
        threshold_text = "NA" if threshold is None else f"{threshold:.0f}"
        regime_text = regime or "未知"

        direction_map = {"LONG": "做多", "SHORT": "做空"}
        stage_map = {
            "position_flip": "位置翻转",
            "position_filter": "位置过滤",
            "indicator_gate": "指标门控",
            "bayesian_gate": "贝叶斯门控",
            "kelly_gate": "凯利门控",
        }
        code_map = {
            "FLIP_NO_MATCH": "翻转后无匹配",
            "BLOCK_POS": "位置评分拦截",
            "BLOCK_MACD": "MACD门控拦截",
            "BLOCK_BAYES": "贝叶斯过滤拦截",
            "BLOCK_KELLY_NEG": "凯利仓位拦截",
        }
        dir_text = direction_map.get(candidate_dir, candidate_dir)
        stage_text = stage_map.get(gate_stage, gate_stage)
        code_text = code_map.get(fail_code, fail_code)

        return (
            f"候选方向={dir_text} | "
            f"位置评分={pos_score_text} | "
            f"方向阈值={threshold_text} | "
            f"市场状态={regime_text} | "
            f"门控阶段={stage_text} | "
            f"失败原因={code_text}"
        )
    
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
        partial_tp = getattr(order, 'partial_tp_count', 0)
        partial_sl = getattr(order, 'partial_sl_count', 0)
        sl = order.stop_loss
        tp = order.take_profit
        original_sl = order.original_stop_loss

        # 构建详细的决策逻辑说明（分段止盈/止损次数）
        stage_name = f"分段止盈{partial_tp}次 分段止损{partial_sl}次" if (partial_tp or partial_sl) else "无"
        
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
            f"分段={stage_name}",
        ]
        
        if sl_moved:
            logic_parts.append(f"SL已上移({original_sl:.2f}→{sl:.2f})")
        
        logic_str = " | ".join(logic_parts)
        
        return f"[平仓决策] {logic_str} | 触发={reason}"
    
    def _get_stop_loss_monitor(self, order, pnl_pct: float) -> str:
        """
        生成止损层级监控信息
        
        显示当前距离各档止损的距离和已触发的分段止损次数
        
        Args:
            order: 当前订单
            pnl_pct: 当前盈亏百分比
        
        Returns:
            止损层级监控字符串
        """
        from config import PAPER_TRADING_CONFIG as _ptc
        
        # 获取分段止损配置
        stage1_pct = _ptc.get("STAGED_SL_1_PCT", 5.0)
        stage2_pct = _ptc.get("STAGED_SL_2_PCT", 10.0)
        min_sl_pct = _ptc.get("MIN_SL_PCT", 0.15) * 100  # 转为百分比
        
        # 获取已触发的分段次数
        partial_sl = getattr(order, 'partial_sl_count', 0)
        
        # 判断当前在哪个区间
        if pnl_pct >= 0:
            # 盈利状态，不显示止损层级
            return ""
        
        abs_loss = abs(pnl_pct)
        
        if abs_loss < stage1_pct:
            # 安全区：未触及第1档
            status = f"安全区(距第1档 {stage1_pct - abs_loss:.1f}%)"
        elif abs_loss < stage2_pct:
            # 第1档已触发
            status = f"⚠第1档已触(距第2档 {stage2_pct - abs_loss:.1f}%)"
        elif abs_loss < min_sl_pct:
            # 第2档已触发
            status = f"⚠⚠第2档已触(距硬止损 {min_sl_pct - abs_loss:.1f}%)"
        else:
            # 接近硬止损
            status = f"🚨危险区(硬止损{min_sl_pct:.0f}%)"
        
        # 显示已减仓次数
        if partial_sl > 0:
            status += f" 已减仓{partial_sl}次"
        
        return f" | 止损层级: {status}"
    
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

        if close_reason == CloseReason.PARTIAL_TP:
            detail = getattr(order, 'close_reason_detail', '')
            partial_count = getattr(order, 'partial_tp_count', 0)
            if detail:
                return f"{detail} (第{partial_count}次分段止盈，峰值盈利{peak_pct:.1f}%)"
            return f"分段止盈(第{partial_count}次，峰值盈利{peak_pct:.1f}%)"
        
        if close_reason == CloseReason.PARTIAL_SL:
            detail = getattr(order, 'close_reason_detail', '')
            partial_count = getattr(order, 'partial_sl_count', 0)
            from config import PAPER_TRADING_CONFIG as _ptc
            stage1_pct = _ptc.get("STAGED_SL_1_PCT", 5.0)
            stage2_pct = _ptc.get("STAGED_SL_2_PCT", 10.0)
            current_loss = abs(order.profit_pct)
            
            # 判断触发的是哪一档
            if partial_count == 1:
                stage_info = f"触发第1档(亏损{stage1_pct:.0f}%)"
            elif partial_count == 2:
                stage_info = f"触发第2档(亏损{stage2_pct:.0f}%)"
            else:
                stage_info = f"第{partial_count}次触发"
            
            if detail:
                return f"{detail} ({stage_info}，当前亏损{current_loss:.1f}%)"
            return f"分段止损({stage_info}，当前亏损{current_loss:.1f}%)"

        if close_reason == CloseReason.TAKE_PROFIT:
            # 真正的止盈：价格触及TP目标
            if tp and ((side == "LONG" and kline.high >= tp) or (side == "SHORT" and kline.low <= tp)):
                return f"触及止盈价(TP={tp:.2f})"
            else:
                return f"止盈(TP={tp:.2f})"
        
        elif close_reason == CloseReason.TRAILING_STOP:
            # 追踪止损/保本止损：SL已移至盈利区，有盈利但未到TP
            if sl_moved and sl_in_profit:
                return f"追踪止损(SL={sl:.2f}, 峰值盈利{peak_pct:.1f}%)"
            elif sl_in_profit:
                return f"保本止损(SL={sl:.2f}已在成本价之上)"
            else:
                return f"追踪止损(SL={sl:.2f})"
        
        elif close_reason == CloseReason.STOP_LOSS:
            # 显示硬止损触发信息，包含分段止损历史
            partial_sl = getattr(order, 'partial_sl_count', 0)
            from config import PAPER_TRADING_CONFIG as _ptc
            min_sl_pct = _ptc.get("MIN_SL_PCT", 0.15) * 100
            
            if partial_sl > 0:
                return f"硬止损(SL={sl:.2f}, {min_sl_pct:.0f}%全平线触发，已分段减仓{partial_sl}次)"
            else:
                return f"硬止损(SL={sl:.2f}, {min_sl_pct:.0f}%全平线触发)"
        
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


    def _check_staged_partial_sl(self, kline: KlineData):
        """
        分段止损：亏损达到阶梯时分批减仓（做多/做空共用，第1档5% 第2档10%）
        """
        if self._paper_trader is None or not self._paper_trader.has_position():
            return
        order = self._paper_trader.current_position
        if order is None:
            return
        profit_pct = getattr(order, "profit_pct", 0.0) or 0.0
        if profit_pct >= 0:
            return
        t1 = PAPER_TRADING_CONFIG.get("STAGED_SL_1_PCT", 5.0)
        t2 = PAPER_TRADING_CONFIG.get("STAGED_SL_2_PCT", 10.0)
        r1 = PAPER_TRADING_CONFIG.get("STAGED_SL_RATIO_1", 0.30)
        r2 = PAPER_TRADING_CONFIG.get("STAGED_SL_RATIO_2", 0.30)
        if order.partial_sl_count == 0 and profit_pct <= -t1:
            pct = r1
            label = "1/2"
        elif order.partial_sl_count == 1 and profit_pct <= -t2:
            pct = r2
            label = "2/2"
        else:
            return
        partial_qty = self._round_to_step(order.quantity * pct)
        if partial_qty <= 0:
            return
        order.close_reason_detail = f"分段止损({label}) 减仓{pct:.0%} 当前亏损{profit_pct:.1f}%"
        closed = self._paper_trader.close_position(
            price=kline.close,
            bar_idx=self._current_bar_idx,
            reason=CloseReason.PARTIAL_SL,
            use_limit_order=False,
            quantity=partial_qty,
        )
        if closed:
            remaining = self._paper_trader.current_position
            if remaining is not None:
                remaining.partial_sl_count = order.partial_sl_count + 1
                remaining.peak_price = order.peak_price
                remaining.peak_profit_pct = order.peak_profit_pct
            msg = f"[分段止损] {label}: 减仓{pct:.0%} @ 当前亏损{profit_pct:.1f}%"
            print(f"[LiveEngine] {msg}")
            self.state.last_event = msg

    def _check_staged_partial_tp(self, kline: KlineData):
        """
        分段止盈：峰值利润达到阶梯时分批减仓（做多/做空共用，第1档5% 第2档10%）
        """
        if self._paper_trader is None or not self._paper_trader.has_position():
            return
        order = self._paper_trader.current_position
        if order is None:
            return
        t1 = PAPER_TRADING_CONFIG.get("STAGED_TP_1_PCT", 5.0)
        t2 = PAPER_TRADING_CONFIG.get("STAGED_TP_2_PCT", 10.0)
        r1 = PAPER_TRADING_CONFIG.get("STAGED_TP_RATIO_1", 0.30)
        r2 = PAPER_TRADING_CONFIG.get("STAGED_TP_RATIO_2", 0.30)
        if order.partial_tp_count == 0 and order.peak_profit_pct >= t1:
            pct = r1
            label = "1/2"
        elif order.partial_tp_count == 1 and order.peak_profit_pct >= t2:
            pct = r2
            label = "2/2"
        else:
            return
        partial_qty = self._round_to_step(order.quantity * pct)
        if partial_qty <= 0:
            return
        order.close_reason_detail = f"分段止盈({label}) 减仓{pct:.0%} 峰值利润{order.peak_profit_pct:.1f}%"
        closed = self._paper_trader.close_position(
            price=kline.close,
            bar_idx=self._current_bar_idx,
            reason=CloseReason.PARTIAL_TP,
            use_limit_order=False,
            quantity=partial_qty,
        )
        if closed:
            remaining = self._paper_trader.current_position
            if remaining is not None:
                remaining.partial_tp_count = order.partial_tp_count + 1
                remaining.peak_price = order.peak_price
                remaining.peak_profit_pct = order.peak_profit_pct
            msg = f"[分段止盈] {label}: 减仓{pct:.0%} @ 峰值利润{order.peak_profit_pct:.1f}%"
            print(f"[LiveEngine] {msg}")
            self.state.last_event = msg

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
        - 持SHORT + 反方向(LONG)位置评分>40 → 翻转做多
        - 持LONG + 反方向(SHORT)位置评分>40 → 翻转做空
        - 使用 _calc_position_score 替代硬编码25%阈值
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
        
        # 用空间位置评分替代硬编码 price_position 阈值
        opp_dir = "SHORT" if order.side == OrderSide.LONG else "LONG"
        flip_score, flip_detail = self._calc_position_score(opp_dir)
        if flip_score <= 40:
            return result  # 反方向位置不够好，不翻转
        
        need_flip = True
        flip_direction = opp_dir
        
        # 用于显示：boll_position 近似表示区间位置 (0=底部, 1=顶部)
        row = self._df_buffer.iloc[-1]
        bp = row.get("boll_position", 0.5)
        if pd.isna(bp):
            bp = 0.5
        price_position = max(0.0, min(1.0, float(bp)))
        
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
            f"位置评分={flip_score:.0f}({flip_detail}) | "
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
