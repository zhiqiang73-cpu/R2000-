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
    position_side: str = "-"


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
                 cosine_threshold: float = 0.6,
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
        
        # Binance 测试网真实执行器（不再使用本地虚拟模式）
        self._paper_trader = BinanceTestnetTrader(
            symbol=symbol,
            api_key=api_key,
            api_secret=api_secret,
            initial_balance=initial_balance,
            leverage=leverage,
            on_order_update=self._on_order_update,
            on_trade_closed=self._on_trade_closed_internal,
        )
        
        # 引擎状态
        self.state = EngineState()
        
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
            
        except Exception as e:
            print(f"[LiveEngine] 特征计算失败: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # 只处理完整K线
            if kline.is_closed:
                self._process_closed_kline(kline)
            else:
                # 实时更新持仓盈亏
                if self._paper_trader.has_position():
                    self._paper_trader.update_price(
                        kline.close,
                        high=kline.high,
                        low=kline.low,
                    )
            
            if self.on_state_update:
                self.on_state_update(self.state)
    
    def _process_closed_kline(self, kline: KlineData):
        """处理完整K线"""
        self.state.total_bars += 1
        self._current_bar_idx += 1
        
        # 更新DataFrame和特征
        if not self._update_features(kline):
            return
        
        # 获取当前ATR
        atr = self._get_current_atr()
        
        # 检查持仓状态
        if self._paper_trader.has_position():
            self._process_holding(kline, atr)
        else:
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
                long_result = self._proto_matcher.match_entry(pre_entry_traj, direction="LONG")
                short_result = self._proto_matcher.match_entry(pre_entry_traj, direction="SHORT")

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

                if direction is not None and chosen_proto is not None:
                    chosen_fp = f"proto_{chosen_proto.direction}_{chosen_proto.prototype_id}"
                    self._current_prototype = chosen_proto
                    self._current_template = None
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
                self.state.best_match_template = chosen_fp[:8]
                
                # 计算止盈止损
                price = kline.close
                if direction == "LONG":
                    side = OrderSide.LONG
                    take_profit = price + atr * self.take_profit_atr
                    stop_loss = price - atr * self.stop_loss_atr
                else:
                    side = OrderSide.SHORT
                    take_profit = price - atr * self.take_profit_atr
                    stop_loss = price + atr * self.stop_loss_atr
                
                reason = self._build_entry_reason(
                    direction=direction,
                    similarity=similarity,
                    regime=self.state.market_regime,
                    template_fp=chosen_fp,
                    atr=atr,
                )
                
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
                
                if order and self.on_trade_opened:
                    self.on_trade_opened(order)
                
                self.state.matching_phase = "持仓中"
                self.state.tracking_status = "安全"
                self.state.fingerprint_status = "匹配成功"
                self.state.decision_reason = reason
                self.state.position_side = direction
                return
            
            # 没有匹配
            self.state.matching_phase = "等待"
            self.state.fingerprint_status = "未匹配"
            self.state.best_match_similarity = 0.0
            self.state.best_match_template = None
            self.state.decision_reason = self._build_no_entry_reason(
                regime=self.state.market_regime,
                long_sim=(long_result.get("similarity", 0.0) if self.use_prototypes else long_result.dtw_similarity),
                short_sim=(short_result.get("similarity", 0.0) if self.use_prototypes else short_result.dtw_similarity),
            )
            
        except Exception as e:
            print(f"[LiveEngine] 入场匹配失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_holding(self, kline: KlineData, atr: float):
        """处理持仓逻辑"""
        self.state.matching_phase = "持仓中"
        self.state.fingerprint_status = "持仓追踪中"
        
        order = self._paper_trader.current_position
        if order is None:
            return
        
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
        
        # 检查最大持仓时间
        if order.hold_bars >= self.max_hold_bars:
            self._paper_trader.close_position(
                kline.close,
                self._current_bar_idx,
                CloseReason.MAX_HOLD,
            )
            self.state.matching_phase = "等待"
            self.state.tracking_status = "-"
            self._current_template = None
            self._current_prototype = None
            self.state.position_side = "-"
            self.state.decision_reason = self._build_exit_reason("超时", order)
            return
        
        # 动态追踪检查
        if order.hold_bars > 0 and order.hold_bars % self.hold_check_interval == 0:
            self._check_holding_similarity(kline)
    
    def _check_holding_similarity(self, kline: KlineData):
        """检查持仓相似度（动态追踪）"""
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
            
            if self.use_prototypes:
                similarity, _ = self._proto_matcher.check_holding_health(
                    holding_traj, self._current_prototype
                )
            else:
                # 计算与当前模板的相似度
                divergence, _ = self._matcher.monitor_holding(
                    holding_traj,
                    self._current_template,
                    divergence_limit=1.0 - self.hold_derail_threshold,  # 转换为偏离度
                )
                # 相似度 = 1 - 偏离度
                similarity = max(0.0, 1.0 - divergence)
            
            # 更新追踪状态
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
            
            if close_reason:
                self.state.matching_phase = "等待"
                self.state.tracking_status = "-"
                self._current_template = None
                self._current_prototype = None
                self.state.position_side = "-"
                self.state.decision_reason = self._build_exit_reason("脱轨", order)
            
        except Exception as e:
            print(f"[LiveEngine] 持仓追踪失败: {e}")
    
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
        """轻量市场状态推断（实时）"""
        if self._df_buffer is None or len(self._df_buffer) < 30:
            return "未知"
        try:
            closes = self._df_buffer["close"].values
            atr = self._df_buffer["atr"].values if "atr" in self._df_buffer.columns else None
            ret_20 = (closes[-1] / closes[-20] - 1.0) if closes[-20] != 0 else 0.0
            atr_ratio = float(np.nanmean(atr[-20:]) / closes[-1]) if atr is not None and closes[-1] != 0 else 0.0
            if ret_20 > 0.004:
                return "上涨趋势"
            if ret_20 < -0.004:
                return "下跌趋势"
            if atr_ratio > 0.004:
                return "高波动震荡"
            return "震荡"
        except Exception:
            return "未知"
    
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
        fp_short = template_fp[:10]
        return (
            f"[开仓逻辑] 市场={regime} | 信号={direction} | "
            f"指纹={fp_short} | 相似度={similarity:.2%}({grade}) | "
            f"风控=SL {self.stop_loss_atr:.1f}ATR / TP {self.take_profit_atr:.1f}ATR。"
            f" 因为匹配强度满足阈值且方向一致，所以执行{direction}开仓。"
        )
    
    def _build_no_entry_reason(self, regime: str, long_sim: float, short_sim: float) -> str:
        """交易员风格不开仓因果说明"""
        best_side = "LONG" if long_sim >= short_sim else "SHORT"
        best_sim = max(long_sim, short_sim)
        return (
            f"[观望逻辑] 市场={regime} | 候选最佳={best_side} | "
            f"最佳相似度={best_sim:.2%}。"
            f" 因为未达到有效开仓阈值/一致性要求，所以维持观望。"
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
