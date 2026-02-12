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
        
        # 如果有持仓，按当前价格平仓
        if self._paper_trader.has_position():
            self._paper_trader.close_position(
                self.state.current_price,
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
            # 更新状态
            self.state.current_price = kline.close
            self.state.current_time = kline.open_time
            if self._paper_trader.current_position is not None:
                self.state.position_side = self._paper_trader.current_position.side.value
            else:
                self.state.position_side = "-"
            
            # 回调
            if self.on_kline:
                self.on_kline(kline)
            
            # 只处理完整K线（入场/持仓决策）
            if kline.is_closed:
                self._process_closed_kline(kline)
            else:
                # 实时更新持仓盈亏 + TP/SL检查（含挂起重试）
                if self._paper_trader.has_position():
                    close_reason = self._paper_trader.update_price(
                        kline.close,
                        high=kline.high,
                        low=kline.low,
                    )
                    if close_reason:
                        self.state.matching_phase = "等待"
                        self.state.tracking_status = "-"
                        self._current_template = None
                        self._current_prototype = None
                        self.state.position_side = "-"
                        self.state.decision_reason = f"实时TP/SL触发: {close_reason.value}"
                else:
                    # 未持仓时，做预匹配展示（不下单，仅更新UI状态）
                    self._preview_match(kline)
            
            if self.on_state_update:
                self.on_state_update(self.state)
    
    def _process_closed_kline(self, kline: KlineData):
        """处理完整K线"""
        self.state.total_bars += 1
        self._current_bar_idx += 1
        
        print(f"[LiveEngine] K线收线: {kline.open_time} | 价格={kline.close:.2f} | 持仓={self._paper_trader.has_position()}")
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
            print(f"[LiveEngine] 尝试入场匹配...")
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
            return 100.0  # 默认值
        
        return float(self._df_buffer['atr'].iloc[-1])
    
    def _process_entry(self, kline: KlineData, atr: float):
        """处理入场逻辑"""
        self.state.matching_phase = "匹配入场"
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
            
            # 获取入场前轨迹
            start_idx = max(0, self._current_bar_idx - pre_entry_window)
            pre_entry_traj = self._fv_engine.get_raw_matrix(start_idx, self._current_bar_idx + 1)
            
            if pre_entry_traj.size == 0:
                self.state.matching_phase = "等待"
                return
            
            direction = None
            similarity = 0.0
            chosen_fp = ""

            if self.use_prototypes:
                # 关键：传入当前市场状态
                current_regime = self.state.market_regime
                
                # 【预热期降级逻辑】
                # 如果摆动点不足 4 个，市场状态为 UNKNOWN。
                # 此时 match_entry 如果传入 regime="未知"，通常会因为原型库中没有该状态而返回未匹配。
                # 我们将其改为 None，让 matcher 跳过状态过滤，直接匹配所有原型。
                match_regime = current_regime
                if current_regime == MarketRegime.UNKNOWN:
                    match_regime = None
                
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

                # 提取投票信息
                long_votes = long_result.get("vote_long", 0)
                short_votes = short_result.get("vote_short", 0)
                print(f"[LiveEngine] 原型匹配结果: LONG={long_sim:.2%}(投票{long_votes}) | SHORT={short_sim:.2%}(投票{short_votes})")
                
                if direction is not None and chosen_proto is not None:
                    # 显示包含市场状态的原型名称
                    regime_short = chosen_proto.regime[:2] if chosen_proto.regime else ""
                    chosen_fp = f"proto_{chosen_proto.direction}_{chosen_proto.prototype_id}_{regime_short}"
                    self._current_prototype = chosen_proto
                    self._current_template = None
                    print(f"[LiveEngine] 匹配成功! 方向={direction} | 原型={chosen_fp} | 相似度={similarity:.2%}")
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
                self.state.best_match_similarity = similarity
                self.state.best_match_template = chosen_fp
                
                # 【动态止盈止损】基于原型历史表现计算
                price = kline.close
                if direction == "LONG":
                    side = OrderSide.LONG
                else:
                    side = OrderSide.SHORT
                
                try:
                    # 使用原型的历史表现计算动态TP/SL
                    take_profit, stop_loss = self._calculate_dynamic_tp_sl(
                        entry_price=price,
                        direction=direction,
                        prototype=chosen_proto if self.use_prototypes else None,
                        atr=atr
                    )
                except Exception as e:
                    print(f"[LiveEngine] TP/SL计算失败: {e}，使用固定ATR")
                    if direction == "LONG":
                        take_profit = price + atr * self.take_profit_atr
                        stop_loss = price - atr * self.stop_loss_atr
                    else:
                        take_profit = price - atr * self.take_profit_atr
                        stop_loss = price + atr * self.stop_loss_atr
                
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
                
                # 【执行开仓】
                try:
                    order = self._paper_trader.open_position(
                        side=side,
                        price=price,
                        bar_idx=self._current_bar_idx,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        template_fingerprint=chosen_fp,
                        entry_similarity=similarity,
                        entry_reason=reason,
                    )
                except Exception as e:
                    # 开仓失败（网络/API错误）— 必须在UI上显示！
                    error_msg = f"开仓失败: {e}"
                    print(f"[LiveEngine] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    self.state.decision_reason = f"[❌开仓失败] {direction} {chosen_fp} | {error_msg}"
                    self.state.last_event = f"❌开仓失败 {direction} | {str(e)[:60]}"
                    return
                
                if order is None:
                    # open_position返回None（已有持仓等）
                    self.state.decision_reason = f"[❌开仓被拒] {direction} | 交易所返回空（可能已有持仓或余额不足）"
                    self.state.last_event = f"❌开仓被拒 {direction} | 交易所返回空"
                    return
                
                if self.on_trade_opened:
                    self.on_trade_opened(order)
                
                self.state.matching_phase = "持仓中"
                self.state.tracking_status = "安全"
                self.state.fingerprint_status = "匹配成功"
                self.state.decision_reason = reason
                self.state.hold_reason = "已开仓，正在按持仓轨迹持续监控。"
                self.state.danger_level = 0.0
                self.state.exit_reason = "形态配合良好，暂无平仓预兆。"
                self.state.position_side = direction
                self.state.last_event = f"✅开仓 {direction} @ ${price:,.2f} | {chosen_fp}"
                return
            
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
            long_sim = 0.0 # Initialize for the no-match message
            short_sim = 0.0 # Initialize for the no-match message

            if self.use_prototypes:
                # 预匹配时也使用当前市场状态过滤
                current_regime = self.state.market_regime
                
                # 【预热期降级逻辑】与入场逻辑一致：UNKNOWN 时跳过 regime 过滤
                match_regime = current_regime
                if current_regime == MarketRegime.UNKNOWN:
                    match_regime = None
                
                long_r = self._proto_matcher.match_entry(
                    pre_entry_traj, direction="LONG", regime=match_regime
                )
                short_r = self._proto_matcher.match_entry(
                    pre_entry_traj, direction="SHORT", regime=match_regime
                )
                long_sim = long_r.get("similarity", 0.0)
                short_sim = short_r.get("similarity", 0.0)
                long_votes = long_r.get("vote_long", 0)
                short_votes = short_r.get("vote_short", 0)
                long_matched = long_r.get("matched", False)
                short_matched = short_r.get("matched", False)
                
                if long_sim >= short_sim and long_sim > 0:
                    best_sim = long_sim
                    best_votes = long_votes
                    best_matched = long_matched
                    proto = long_r.get("best_prototype")
                    regime_short = proto.regime[:2] if proto and proto.regime else ""
                    best_fp = f"proto_{proto.direction}_{proto.prototype_id}_{regime_short}" if proto else ""
                    best_dir = "LONG"
                elif short_sim > 0:
                    best_sim = short_sim
                    best_votes = short_votes
                    best_matched = short_matched
                    proto = short_r.get("best_prototype")
                    regime_short = proto.regime[:2] if proto and proto.regime else ""
                    best_fp = f"proto_{proto.direction}_{proto.prototype_id}_{regime_short}" if proto else ""
                    best_dir = "SHORT"
                else:
                    best_votes = 0
                    best_matched = False
            else:
                # For non-prototype matching, we still need long_sim and short_sim for the decision reason
                long_candidates = self.trajectory_memory.get_templates_by_direction("LONG")
                short_candidates = self.trajectory_memory.get_templates_by_direction("SHORT")
                
                long_r = self._matcher.match_entry(
                    pre_entry_traj,
                    long_candidates,
                    cosine_threshold=self.cosine_threshold,
                    dtw_threshold=self.dtw_threshold,
                )
                short_r = self._matcher.match_entry(
                    pre_entry_traj,
                    short_candidates,
                    cosine_threshold=self.cosine_threshold,
                    dtw_threshold=self.dtw_threshold,
                )
                long_sim = long_r.dtw_similarity
                short_sim = short_r.dtw_similarity

                # The rest of the logic for best_sim, best_fp, best_dir would go here if needed for non-prototype preview
                # However, the user's instruction only modified the prototype branch for best_sim/fp/dir.
                # For the non-prototype case, the original code didn't update best_sim/fp/dir in preview.
                # So, we only calculate long_sim/short_sim here for the decision reason.


            # 更新状态（预览，不触发交易）
            self.state.matching_phase = "预匹配(等待收线)"
            self.state.fingerprint_status = "实时预览"
            self.state.best_match_similarity = best_sim
            self.state.best_match_template = best_fp if best_fp else None
            
            if self.use_prototypes and best_sim > 0 and best_fp:
                # 判断收线后是否会开仓
                will_open = best_matched
                status_icon = "✅可开仓" if will_open else "⏳待确认"
                vote_info = f"投票={best_votes}/{self.min_templates_agree}"
                threshold_info = f"阈值={self.cosine_threshold:.0%}"
                
                self.state.decision_reason = (
                    f"[预匹配] 市场={self.state.market_regime} | {best_dir} | "
                    f"相似度={best_sim:.1%} | {vote_info} | {threshold_info} | "
                    f"{status_icon} — 等K线收线"
                )
            elif best_sim > 0 and best_fp:
                self.state.decision_reason = (
                    f"[预匹配] 市场={self.state.market_regime} | 最佳={best_dir} | "
                    f"原型={best_fp} | 相似度={best_sim:.2%} — 等待K线收线后确认入场"
                )
            else:
                if self.use_prototypes:
                    self.state.decision_reason = (
                        f"[观望] 市场={self.state.market_regime} | "
                        f"LONG={long_sim:.1%}(投票{long_votes}) | SHORT={short_sim:.1%}(投票{short_votes}) | "
                        f"❌未达阈值{self.cosine_threshold:.0%}"
                    )
                else:
                    self.state.decision_reason = (
                        f"[观望] 市场={self.state.market_regime} | "
                        f"未找到匹配原型（LONG={long_sim:.1%}, SHORT={short_sim:.1%}）"
                    )

        except Exception as e:
            # 预匹配失败不影响主流程
            print(f"[LiveEngine] 预匹配失败: {e}")


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
        
        # 更新价格，检查止盈止损
        close_reason = self._paper_trader.update_price(
            kline.close,
            high=kline.high,
            low=kline.low,
            bar_idx=self._current_bar_idx,
        )
        
        if close_reason:
            self.state.matching_phase = "等待"
            self.state.tracking_status = "-"
            self._current_template = None
            self._current_prototype = None
            self.state.position_side = "-"
            self.state.decision_reason = self._build_exit_reason(close_reason.value, order)
            return
        
        # 【删除】最大持仓时间限制 - 完全依赖轨迹相似度追踪
        
        # 动态追踪检查
        if order.hold_bars > 0 and order.hold_bars % self.hold_check_interval == 0:
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
                
                # ══════════════════════════════════════════════════════════
                # 阶段3：离场模式检测
                # ══════════════════════════════════════════════════════════
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
                
                # 如果离场模式检测建议离场
                if exit_check["should_exit"]:
                    exit_reason_str = exit_check["exit_reason"]
                    print(f"[LiveEngine] 离场模式触发: {exit_reason_str} "
                          f"(信号强度: {exit_check['exit_signal_strength']:.0%})")
                    
                    # 执行信号离场
                    self._paper_trader.close_position(
                        exit_price=kline.close,
                        exit_time=datetime.now(),
                        reason=CloseReason.SIGNAL,
                        bar_idx=self._current_bar_idx,
                    )
                    
                    self.state.matching_phase = "等待"
                    self.state.tracking_status = "-"
                    self.state.hold_reason = ""
                    self.state.danger_level = 0.0
                    self.state.exit_reason = ""
                    self._current_template = None
                    self._current_prototype = None
                    self.state.position_side = "-"
                    self.state.decision_reason = self._build_exit_reason(f"信号({exit_reason_str})", order)
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
                self.state.matching_phase = "等待"
                self.state.tracking_status = "-"
                self.state.hold_reason = ""
                self.state.danger_level = 0.0
                self.state.exit_reason = ""
                self._current_template = None
                self._current_prototype = None
                self.state.position_side = "-"
                self.state.decision_reason = self._build_exit_reason("脱轨", order)
            
        except Exception as e:
            import traceback
            print(f"[LiveEngine] 持仓追踪失败: {e}")
            traceback.print_exc()
    
    def _calculate_dynamic_tp_sl(self, entry_price: float, direction: str,
                                  prototype, atr: float):
        """
        基于原型历史表现计算动态止盈止损
        
        Args:
            entry_price: 入场价格
            direction: LONG/SHORT
            prototype: 匹配的原型（Prototype对象）
            atr: 当前ATR
        
        Returns:
            (take_profit_price, stop_loss_price)
        """
        # 安全检查：如果原型数据不足，回退到固定ATR倍数
        if not prototype or getattr(prototype, 'member_count', 0) < 10:
            if direction == "LONG":
                tp = entry_price + atr * self.take_profit_atr
                sl = entry_price - atr * self.stop_loss_atr
            else:
                tp = entry_price - atr * self.take_profit_atr
                sl = entry_price + atr * self.stop_loss_atr
            return tp, sl
        
        # 1. 计算止盈目标（基于平均收益率）
        import numpy as np
        profit_pct = np.clip(prototype.avg_profit_pct, 0.5, 10.0) / 100.0
        
        # 根据胜率调整（高胜率更激进，低胜率更保守）
        win_rate = prototype.win_rate
        if win_rate >= 0.75:
            profit_pct *= 1.2  # 高胜率：提高20%
        elif win_rate < 0.60:
            profit_pct *= 0.8  # 低胜率：降低20%
        
        # 2. 计算止损幅度（基于风险收益比）
        if win_rate >= 0.70:
            risk_reward_ratio = 2.0  # 高胜率：1:2
        elif win_rate >= 0.50:
            risk_reward_ratio = 1.5  # 中胜率：1:1.5
        else:
            risk_reward_ratio = 1.0  # 低胜率：1:1
        
        stop_loss_pct = profit_pct / risk_reward_ratio
        
        # ATR保护：止损至少为1.5倍ATR
        min_stop_loss_pct = (atr / entry_price) * 1.5
        stop_loss_pct = max(stop_loss_pct, min_stop_loss_pct)
        
        # 3. 计算最终价格
        if direction == "LONG":
            take_profit = entry_price * (1 + profit_pct)
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - profit_pct)
            stop_loss = entry_price * (1 + stop_loss_pct)
        
        return take_profit, stop_loss
    
    def _on_order_update(self, order: PaperOrder):
        """订单更新回调"""
        pass  # 由状态更新回调处理
    
    def _on_trade_closed_internal(self, order: PaperOrder):
        """交易关闭内部回调"""
        if self.on_trade_closed:
            self.on_trade_closed(order)
        
        self._current_template = None
        self._current_prototype = None
        self.state.matching_phase = "等待"
        self.state.tracking_status = "-"
        self.state.fingerprint_status = "待匹配"
        self.state.position_side = "-"
    
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
             print(f"[LiveEngine] 当前摆动点: {len(self._swing_points)} (原始: {raw_count}) | 序列: {[('H' if s.is_high else 'L') + '@' + str(s.index) for s in self._swing_points]}")
    
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
        """交易员风格不开仓因果说明（含投票信息）"""
        best_side = "LONG" if long_sim >= short_sim else "SHORT"
        best_sim = max(long_sim, short_sim)
        best_votes = long_votes if long_sim >= short_sim else short_votes
        
        # 判断失败原因
        reasons = []
        if best_sim < threshold:
            reasons.append(f"相似度{best_sim:.1%}<阈值{threshold:.0%}")
        if best_votes < min_agree:
            reasons.append(f"投票{best_votes}<最低{min_agree}")
        
        fail_reason = "；".join(reasons) if reasons else "条件未满足"
        
        return (
            f"[观望] 市场={regime} | 最佳={best_side}({best_sim:.1%}) | "
            f"投票={best_votes}/{min_agree} | ❌{fail_reason}"
        )
    
    @staticmethod
    def _build_exit_reason(reason: str, order) -> str:
        """交易员风格平仓因果说明"""
        side = order.side.value if order is not None else "-"
        hold = order.hold_bars if order is not None else "-"
        return (
            f"[平仓逻辑] 方向={side} | 持仓K线={hold} | 触发条件={reason}。"
            f" 因为风险控制条件触发，所以执行平仓。"
        )
    
    def get_history_df(self) -> pd.DataFrame:
        """获取历史K线DataFrame"""
        return self._data_feed.get_history_df()
    
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


# 简单测试
if __name__ == "__main__":
    print("LiveTradingEngine 测试需要 TrajectoryMemory，请在完整环境中运行")
