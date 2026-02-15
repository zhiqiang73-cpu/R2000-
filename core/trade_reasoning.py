"""
R3000 交易推理引擎
实时5层逻辑链分析 + 决策快照记录

功能：
  - DecisionSnapshot: 捕获决策时刻的完整指标状态
  - ReasoningLayer: 单层推理结果（状态+摘要+详情+指标）
  - ReasoningResult: 5层推理结果+综合判决+叙述
  - TradeReasoning: 主推理引擎（每根K线分析一次）
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class DecisionSnapshot:
    """
    决策快照：记录决策时刻（入场/出场/拒绝）的完整指标状态
    用于回溯分析、反事实推演、特征学习
    """
    timestamp: str
    price: float
    bar_idx: int
    
    # === KDJ 状态 ===
    kdj_j: float
    kdj_d: float
    kdj_k: float
    kdj_trend: str  # "rising" / "falling" / "flat" (比较最近3根K线)
    
    # === MACD 状态 ===
    macd_hist: float
    macd_hist_slope: float  # 柱状图斜率（最近3根K线）
    macd_signal: float
    
    # === 趋势指标 ===
    rsi: float
    adx: float
    
    # === 布林带位置 ===
    boll_position: float  # 0=下轨, 0.5=中轨, 1=上轨
    
    # === 波动率 ===
    atr: float
    atr_change_pct: float  # ATR扩张/收缩比例（相对前一根）
    
    # === 成交量 ===
    volume_ratio: float  # 当前成交量/MA成交量
    obv_slope: float  # OBV斜率（最近3根）
    
    # === 位置/支撑阻力 ===
    dist_to_support_pct: float  # 距离支撑位的百分比
    dist_to_resistance_pct: float  # 距离阻力位的百分比
    
    # === 上下文 ===
    market_regime: str  # 市场状态
    similarity: float  # 当前相似度（持仓中才有）
    
    def to_dict(self) -> dict:
        """转为字典（用于存储）"""
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "bar_idx": self.bar_idx,
            "kdj_j": self.kdj_j,
            "kdj_d": self.kdj_d,
            "kdj_k": self.kdj_k,
            "kdj_trend": self.kdj_trend,
            "macd_hist": self.macd_hist,
            "macd_hist_slope": self.macd_hist_slope,
            "macd_signal": self.macd_signal,
            "rsi": self.rsi,
            "adx": self.adx,
            "boll_position": self.boll_position,
            "atr": self.atr,
            "atr_change_pct": self.atr_change_pct,
            "volume_ratio": self.volume_ratio,
            "obv_slope": self.obv_slope,
            "dist_to_support_pct": self.dist_to_support_pct,
            "dist_to_resistance_pct": self.dist_to_resistance_pct,
            "market_regime": self.market_regime,
            "similarity": self.similarity,
        }


@dataclass
class ReasoningLayer:
    """单层推理结果"""
    icon: str  # 图标（emoji或符号）
    name: str  # 层名称
    status: str  # "favorable" / "neutral" / "adverse"
    color: str  # 颜色（用于UI）
    summary: str  # 一句话摘要
    detail: str  # 详细说明（多行）
    raw_metrics: Dict[str, Any] = field(default_factory=dict)  # 原始指标值


@dataclass
class ReasoningResult:
    """5层推理结果"""
    layers: List[ReasoningLayer]  # 5个层的结果
    verdict: str  # "hold_firm" / "tighten_watch" / "prepare_exit" / "exit_now"
    narrative: str  # 综合叙述（2-3句话）
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """转为字典（用于存储）"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "verdict": self.verdict,
            "narrative": self.narrative,
            "layers": [
                {
                    "icon": layer.icon,
                    "name": layer.name,
                    "status": layer.status,
                    "color": layer.color,
                    "summary": layer.summary,
                    "detail": layer.detail,
                    "raw_metrics": layer.raw_metrics,
                }
                for layer in self.layers
            ],
        }


class TradeReasoning:
    """
    交易推理引擎
    每根K线调用一次，生成5层分析结果
    """
    
    def __init__(self):
        """初始化"""
        pass
    
    def analyze(self, order: Any, df: pd.DataFrame, state: Any) -> ReasoningResult:
        """
        分析当前持仓状态
        
        Args:
            order: PaperOrder 当前持仓订单
            df: DataFrame K线数据（包含指标）
            state: EngineState 引擎状态
        
        Returns:
            ReasoningResult 5层推理结果
        """
        if df is None or len(df) == 0:
            return self._create_empty_result()
        
        current_bar = df.iloc[-1]
        
        # Layer 1: 市场态势
        layer1 = self._analyze_market_stance(order, df, state, current_bar)
        
        # Layer 2: 模式追踪
        layer2 = self._analyze_pattern_tracking(order, df, state, current_bar)
        
        # Layer 3: 动量分析
        layer3 = self._analyze_momentum(order, df, state, current_bar)
        
        # Layer 4: 盈亏评估
        layer4 = self._analyze_pnl(order, df, state, current_bar)
        
        # Layer 5: 安全检查
        layer5 = self._analyze_safety(order, df, state, current_bar)
        
        # 综合判决
        verdict = self._synthesize_verdict([layer1, layer2, layer3, layer4, layer5])
        
        # 生成叙述
        narrative = self._generate_narrative([layer1, layer2, layer3, layer4, layer5], verdict)
        
        return ReasoningResult(
            layers=[layer1, layer2, layer3, layer4, layer5],
            verdict=verdict,
            narrative=narrative,
            timestamp=datetime.now()
        )
    
    def capture_decision_snapshot(self, df: pd.DataFrame, bar_idx: int, 
                                   market_regime: str, similarity: float = 0.0) -> DecisionSnapshot:
        """
        捕获决策时刻的完整指标状态
        
        Args:
            df: K线数据（包含指标）
            bar_idx: 当前K线索引
            market_regime: 市场状态
            similarity: 相似度（持仓中才有）
        
        Returns:
            DecisionSnapshot 决策快照
        """
        if df is None or len(df) == 0 or bar_idx >= len(df):
            return self._create_empty_snapshot(market_regime, similarity)
        
        current = df.iloc[bar_idx]
        
        # 计算趋势（最近3根）
        kdj_trend = self._calculate_trend(df, bar_idx, 'kdj_j', window=3)
        
        # 计算MACD斜率
        macd_hist_slope = self._calculate_slope(df, bar_idx, 'macd_hist', window=3)
        
        # 计算ATR变化
        atr_change_pct = self._calculate_change_pct(df, bar_idx, 'atr')
        
        # 计算OBV斜率
        obv_slope = self._calculate_slope(df, bar_idx, 'obv', window=3) if 'obv' in df.columns else 0.0
        
        # 计算布林带位置
        boll_position = self._calculate_boll_position(current)
        
        # 计算支撑/阻力距离
        dist_to_support, dist_to_resistance = self._calculate_support_resistance_dist(df, bar_idx)
        
        return DecisionSnapshot(
            timestamp=str(current.get('timestamp', datetime.now())),
            price=float(current['close']),
            bar_idx=bar_idx,
            kdj_j=float(current.get('kdj_j', 50)),
            kdj_d=float(current.get('kdj_d', 50)),
            kdj_k=float(current.get('kdj_k', 50)),
            kdj_trend=kdj_trend,
            macd_hist=float(current.get('macd_hist', 0)),
            macd_hist_slope=macd_hist_slope,
            macd_signal=float(current.get('macd_signal', 0)),
            rsi=float(current.get('rsi', 50)),
            adx=float(current.get('adx', 25)),
            boll_position=boll_position,
            atr=float(current.get('atr', 0)),
            atr_change_pct=atr_change_pct,
            volume_ratio=float(current.get('volume_ratio', 1.0)),
            obv_slope=obv_slope,
            dist_to_support_pct=dist_to_support,
            dist_to_resistance_pct=dist_to_resistance,
            market_regime=market_regime,
            similarity=similarity,
        )
    
    # ==================== Layer 1: 市场态势 ====================
    
    def _analyze_market_stance(self, order, df, state, current_bar) -> ReasoningLayer:
        """Layer 1: 市场态势分析"""
        regime = state.market_regime if hasattr(state, 'market_regime') else "未知"
        regime_at_entry = order.regime_at_entry if hasattr(order, 'regime_at_entry') else "未知"
        
        # 判断市场状态是否与入场一致
        regime_consistent = (regime == regime_at_entry)
        
        # 检查市场状态变化历史（如果有）
        regime_changed = not regime_consistent
        
        if regime_consistent:
            status = "favorable"
            color = "#00E676"
            summary = f"市场保持{regime}，与入场一致"
            detail = f"入场时: {regime_at_entry}\n当前: {regime}\n状态一致，继续持仓有利"
        elif regime == "震荡":
            status = "neutral"
            color = "#FFA726"
            summary = f"市场转为{regime}，需谨慎"
            detail = f"入场时: {regime_at_entry}\n当前: {regime}\n震荡市中方向性减弱"
        else:
            status = "adverse"
            color = "#FF5252"
            summary = f"市场状态改变: {regime_at_entry} → {regime}"
            detail = f"入场时: {regime_at_entry}\n当前: {regime}\n市场环境已改变，警惕趋势反转"
        
        return ReasoningLayer(
            icon="🌐",
            name="市场态势",
            status=status,
            color=color,
            summary=summary,
            detail=detail,
            raw_metrics={
                "regime_at_entry": regime_at_entry,
                "current_regime": regime,
                "regime_consistent": regime_consistent,
            }
        )
    
    # ==================== Layer 2: 模式追踪 ====================
    
    def _analyze_pattern_tracking(self, order, df, state, current_bar) -> ReasoningLayer:
        """Layer 2: 模式追踪分析"""
        current_sim = order.current_similarity if hasattr(order, 'current_similarity') else 0.0
        entry_sim = order.entry_similarity if hasattr(order, 'entry_similarity') else 0.0
        
        # 分析相似度趋势（如果有历史记录）
        sim_history = order.similarity_history if hasattr(order, 'similarity_history') else []
        
        if len(sim_history) >= 3:
            recent_sims = [s[1] for s in sim_history[-3:]]
            sim_trend = self._analyze_similarity_trend(recent_sims)
        else:
            sim_trend = "stable"
        
        # 判断状态
        if current_sim >= 0.7:
            if sim_trend == "rising":
                status = "favorable"
                color = "#00E676"
                summary = f"模式匹配强且上升 ({current_sim:.2f})"
                detail = f"当前相似度: {current_sim:.2f}\n入场相似度: {entry_sim:.2f}\n趋势: 上升\n模式匹配良好"
            else:
                status = "favorable"
                color = "#00E676"
                summary = f"模式匹配稳定 ({current_sim:.2f})"
                detail = f"当前相似度: {current_sim:.2f}\n入场相似度: {entry_sim:.2f}\n趋势: 稳定\n保持良好匹配"
        elif current_sim >= 0.5:
            status = "neutral"
            color = "#FFA726"
            summary = f"模式开始偏离 ({current_sim:.2f})"
            detail = f"当前相似度: {current_sim:.2f}\n入场相似度: {entry_sim:.2f}\n趋势: {sim_trend}\n轻度偏离，需关注"
        elif current_sim >= 0.3:
            status = "adverse"
            color = "#FF8A65"
            summary = f"模式偏离明显 ({current_sim:.2f})"
            detail = f"当前相似度: {current_sim:.2f}\n入场相似度: {entry_sim:.2f}\n趋势: {sim_trend}\n偏离较大，警惕脱轨"
        else:
            status = "adverse"
            color = "#FF5252"
            summary = f"模式严重脱轨 ({current_sim:.2f})"
            detail = f"当前相似度: {current_sim:.2f}\n入场相似度: {entry_sim:.2f}\n趋势: {sim_trend}\n严重偏离，建议平仓"
        
        return ReasoningLayer(
            icon="📊",
            name="模式追踪",
            status=status,
            color=color,
            summary=summary,
            detail=detail,
            raw_metrics={
                "current_similarity": current_sim,
                "entry_similarity": entry_sim,
                "similarity_trend": sim_trend,
            }
        )
    
    # ==================== Layer 3: 动量分析 ====================
    
    def _analyze_momentum(self, order, df, state, current_bar) -> ReasoningLayer:
        """Layer 3: 动量分析"""
        # 计算MACD柱状图变化
        macd_delta = self._calculate_slope(df, len(df)-1, 'macd_hist', window=3)
        
        # 计算KDJ J线斜率
        kdj_j_slope = self._calculate_slope(df, len(df)-1, 'kdj_j', window=3)
        
        # 判断方向
        side = order.side.value if hasattr(order, 'side') else "LONG"
        is_long = (side == "LONG")
        
        # 判断动量状态
        if is_long:
            macd_favorable = macd_delta > 0
            kdj_favorable = kdj_j_slope > 0
        else:
            macd_favorable = macd_delta < 0
            kdj_favorable = kdj_j_slope < 0
        
        # 综合判断
        if macd_favorable and kdj_favorable:
            status = "favorable"
            color = "#00E676"
            summary = "动量增强，趋势延续"
            detail = f"MACD柱斜率: {macd_delta:+.2f}\nKDJ-J斜率: {kdj_j_slope:+.2f}\n动量指标支持持仓方向"
        elif not macd_favorable and not kdj_favorable:
            status = "adverse"
            color = "#FF5252"
            summary = "动量逆转，趋势反转风险"
            detail = f"MACD柱斜率: {macd_delta:+.2f}\nKDJ-J斜率: {kdj_j_slope:+.2f}\n动量指标与持仓方向相反"
        else:
            status = "neutral"
            color = "#FFA726"
            summary = "动量分化，信号不明"
            detail = f"MACD柱斜率: {macd_delta:+.2f}\nKDJ-J斜率: {kdj_j_slope:+.2f}\n动量指标出现分歧"
        
        return ReasoningLayer(
            icon="⚡",
            name="动量分析",
            status=status,
            color=color,
            summary=summary,
            detail=detail,
            raw_metrics={
                "macd_hist_slope": macd_delta,
                "kdj_j_slope": kdj_j_slope,
                "side": side,
            }
        )
    
    # ==================== Layer 4: 盈亏评估 ====================
    
    def _analyze_pnl(self, order, df, state, current_bar) -> ReasoningLayer:
        """Layer 4: 盈亏评估"""
        profit_pct = order.profit_pct if hasattr(order, 'profit_pct') else 0.0
        peak_profit_pct = order.peak_profit_pct if hasattr(order, 'peak_profit_pct') else 0.0
        
        # 计算回撤
        drawdown_from_peak = peak_profit_pct - profit_pct if peak_profit_pct > 0 else 0.0
        
        # 计算风险收益比
        current_price = float(current_bar['close'])
        entry_price = order.entry_price if hasattr(order, 'entry_price') else current_price
        tp = order.take_profit if hasattr(order, 'take_profit') else None
        sl = order.stop_loss if hasattr(order, 'stop_loss') else None
        
        if tp and sl:
            remaining_reward = abs(tp - current_price)
            remaining_risk = abs(current_price - sl)
            rr_ratio = remaining_reward / remaining_risk if remaining_risk > 0 else 0
        else:
            rr_ratio = 0
        
        # 计算持仓效率（利润/持仓时间）
        hold_bars = order.hold_bars if hasattr(order, 'hold_bars') else 1
        profit_per_bar = profit_pct / max(hold_bars, 1)
        
        # 判断状态
        if profit_pct >= 3.0 and drawdown_from_peak < 0.5:
            status = "favorable"
            color = "#00E676"
            summary = f"盈利丰厚 +{profit_pct:.2f}%"
            detail = f"当前利润: +{profit_pct:.2f}%\n峰值利润: +{peak_profit_pct:.2f}%\n回撤: {drawdown_from_peak:.2f}%\n持仓效率: {profit_per_bar:.3f}%/根\n剩余风险收益比: {rr_ratio:.2f}"
        elif profit_pct >= 1.0:
            status = "favorable"
            color = "#69F0AE"
            summary = f"稳健盈利 +{profit_pct:.2f}%"
            detail = f"当前利润: +{profit_pct:.2f}%\n峰值利润: +{peak_profit_pct:.2f}%\n回撤: {drawdown_from_peak:.2f}%\n持仓效率: {profit_per_bar:.3f}%/根\n剩余风险收益比: {rr_ratio:.2f}"
        elif profit_pct >= -0.5:
            status = "neutral"
            color = "#FFA726"
            summary = f"盈亏平衡 {profit_pct:+.2f}%"
            detail = f"当前利润: {profit_pct:+.2f}%\n峰值利润: +{peak_profit_pct:.2f}%\n回撤: {drawdown_from_peak:.2f}%\n持仓效率: {profit_per_bar:.3f}%/根\n剩余风险收益比: {rr_ratio:.2f}"
        else:
            status = "adverse"
            color = "#FF5252"
            summary = f"亏损扩大 {profit_pct:+.2f}%"
            detail = f"当前利润: {profit_pct:+.2f}%\n峰值利润: +{peak_profit_pct:.2f}%\n回撤: {drawdown_from_peak:.2f}%\n持仓效率: {profit_per_bar:.3f}%/根\n剩余风险收益比: {rr_ratio:.2f}"
        
        return ReasoningLayer(
            icon="💰",
            name="盈亏评估",
            status=status,
            color=color,
            summary=summary,
            detail=detail,
            raw_metrics={
                "profit_pct": profit_pct,
                "peak_profit_pct": peak_profit_pct,
                "drawdown_from_peak": drawdown_from_peak,
                "rr_ratio": rr_ratio,
                "profit_per_bar": profit_per_bar,
            }
        )
    
    # ==================== Layer 5: 安全检查 ====================
    
    def _analyze_safety(self, order, df, state, current_bar) -> ReasoningLayer:
        """Layer 5: 安全检查"""
        current_price = float(current_bar['close'])
        sl = order.stop_loss if hasattr(order, 'stop_loss') else None
        atr = float(current_bar.get('atr', 0))
        
        # 计算距离止损的ATR倍数
        if sl and atr > 0:
            dist_to_sl = abs(current_price - sl)
            atr_multiples = dist_to_sl / atr
        else:
            atr_multiples = 0
        
        # 检查ATR是否在扩张
        atr_change = self._calculate_change_pct(df, len(df)-1, 'atr')
        atr_expanding = atr_change > 5  # ATR扩张超过5%
        
        # 计算保证金利用率（假设）
        margin_utilization = 0.0  # 这里需要从实际账户获取
        
        # 判断安全状态
        if atr_multiples >= 2.0 and not atr_expanding:
            status = "favorable"
            color = "#00E676"
            summary = f"止损安全距离充足 ({atr_multiples:.1f}x ATR)"
            detail = f"距止损: {atr_multiples:.1f}x ATR\nATR变化: {atr_change:+.1f}%\n{'ATR扩张中' if atr_expanding else 'ATR稳定'}\n风险可控"
        elif atr_multiples >= 1.0:
            status = "neutral"
            color = "#FFA726"
            summary = f"止损距离适中 ({atr_multiples:.1f}x ATR)"
            detail = f"距止损: {atr_multiples:.1f}x ATR\nATR变化: {atr_change:+.1f}%\n{'ATR扩张中，注意风险' if atr_expanding else 'ATR稳定'}\n需关注价格波动"
        elif atr_multiples > 0:
            status = "adverse"
            color = "#FF8A65"
            summary = f"接近止损 ({atr_multiples:.1f}x ATR)"
            detail = f"距止损: {atr_multiples:.1f}x ATR\nATR变化: {atr_change:+.1f}%\n{'ATR扩张，风险加大' if atr_expanding else 'ATR稳定'}\n警惕被止损"
        else:
            status = "adverse"
            color = "#FF5252"
            summary = "止损数据异常"
            detail = "无法计算止损距离\n需检查止损设置"
        
        return ReasoningLayer(
            icon="🛡️",
            name="安全检查",
            status=status,
            color=color,
            summary=summary,
            detail=detail,
            raw_metrics={
                "atr_multiples_to_sl": atr_multiples,
                "atr_change_pct": atr_change,
                "atr_expanding": atr_expanding,
            }
        )
    
    # ==================== 辅助方法 ====================
    
    def _synthesize_verdict(self, layers: List[ReasoningLayer]) -> str:
        """综合5层结果得出判决"""
        # 计数各状态
        favorable_count = sum(1 for layer in layers if layer.status == "favorable")
        adverse_count = sum(1 for layer in layers if layer.status == "adverse")
        
        # Layer 5 (安全检查) 权重最高
        safety_status = layers[4].status if len(layers) >= 5 else "neutral"
        
        # Layer 4 (盈亏) 权重次之
        pnl_status = layers[3].status if len(layers) >= 4 else "neutral"
        
        # 判决逻辑
        if safety_status == "adverse" and pnl_status == "adverse":
            return "exit_now"  # 安全+盈亏双重不利 → 立即平仓
        elif adverse_count >= 3:
            return "prepare_exit"  # 3层及以上不利 → 准备平仓
        elif adverse_count >= 2:
            return "tighten_watch"  # 2层不利 → 收紧观察
        else:
            return "hold_firm"  # 多数有利/中性 → 坚定持仓
    
    def _generate_narrative(self, layers: List[ReasoningLayer], verdict: str) -> str:
        """生成综合叙述"""
        # 提取关键点
        market_summary = layers[0].summary if len(layers) > 0 else ""
        pattern_summary = layers[1].summary if len(layers) > 1 else ""
        pnl_summary = layers[3].summary if len(layers) > 3 else ""
        
        # 根据判决生成叙述
        if verdict == "exit_now":
            return f"⚠️ 建议立即平仓。{pnl_summary}，{pattern_summary}。风险已达临界点。"
        elif verdict == "prepare_exit":
            return f"🔶 准备平仓。{market_summary}，{pattern_summary}。多项指标转向不利，建议择机离场。"
        elif verdict == "tighten_watch":
            return f"👀 收紧观察。{pnl_summary}，需警惕变化。部分指标出现预警信号。"
        else:
            return f"✅ 坚定持仓。{market_summary}，{pnl_summary}。多项指标支持继续持仓。"
    
    def _calculate_trend(self, df: pd.DataFrame, bar_idx: int, column: str, window: int = 3) -> str:
        """计算趋势（rising/falling/flat）"""
        if bar_idx < window or column not in df.columns:
            return "flat"
        
        values = df[column].iloc[bar_idx-window+1:bar_idx+1].values
        if len(values) < window:
            return "flat"
        
        # 简单线性回归斜率
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.5:
            return "rising"
        elif slope < -0.5:
            return "falling"
        else:
            return "flat"
    
    def _calculate_slope(self, df: pd.DataFrame, bar_idx: int, column: str, window: int = 3) -> float:
        """计算斜率"""
        if bar_idx < window or column not in df.columns:
            return 0.0
        
        values = df[column].iloc[bar_idx-window+1:bar_idx+1].values
        if len(values) < window:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _calculate_change_pct(self, df: pd.DataFrame, bar_idx: int, column: str) -> float:
        """计算变化百分比"""
        if bar_idx < 1 or column not in df.columns:
            return 0.0
        
        current = df[column].iloc[bar_idx]
        previous = df[column].iloc[bar_idx - 1]
        
        if previous == 0:
            return 0.0
        
        return float((current - previous) / previous * 100)
    
    def _calculate_boll_position(self, bar: pd.Series) -> float:
        """计算布林带位置 (0-1)"""
        if 'boll_upper' not in bar or 'boll_lower' not in bar:
            return 0.5
        
        upper = bar.get('boll_upper', 0)
        lower = bar.get('boll_lower', 0)
        close = bar.get('close', 0)
        
        if upper == lower:
            return 0.5
        
        position = (close - lower) / (upper - lower)
        return float(np.clip(position, 0, 1))
    
    def _calculate_support_resistance_dist(self, df: pd.DataFrame, bar_idx: int) -> tuple:
        """计算距离支撑/阻力的百分比"""
        if bar_idx < 20:
            return 0.0, 0.0
        
        # 简单方法：最近20根的高低点
        recent = df.iloc[max(0, bar_idx-20):bar_idx+1]
        support = recent['low'].min()
        resistance = recent['high'].max()
        current_price = df.iloc[bar_idx]['close']
        
        dist_to_support = (current_price - support) / current_price * 100
        dist_to_resistance = (resistance - current_price) / current_price * 100
        
        return float(dist_to_support), float(dist_to_resistance)
    
    def _analyze_similarity_trend(self, recent_sims: List[float]) -> str:
        """分析相似度趋势"""
        if len(recent_sims) < 2:
            return "stable"
        
        # 简单比较最后两个值
        diff = recent_sims[-1] - recent_sims[-2]
        
        if diff > 0.05:
            return "rising"
        elif diff < -0.05:
            return "falling"
        else:
            return "stable"
    
    def _create_empty_result(self) -> ReasoningResult:
        """创建空结果（无持仓时）"""
        empty_layer = ReasoningLayer(
            icon="⏸️",
            name="无持仓",
            status="neutral",
            color="#888888",
            summary="当前无持仓",
            detail="等待入场信号",
            raw_metrics={}
        )
        
        return ReasoningResult(
            layers=[empty_layer] * 5,
            verdict="hold_firm",
            narrative="当前无持仓，等待入场信号。",
            timestamp=datetime.now()
        )
    
    def _create_empty_snapshot(self, market_regime: str, similarity: float) -> DecisionSnapshot:
        """创建空快照"""
        return DecisionSnapshot(
            timestamp=datetime.now().isoformat(),
            price=0.0,
            bar_idx=0,
            kdj_j=50.0,
            kdj_d=50.0,
            kdj_k=50.0,
            kdj_trend="flat",
            macd_hist=0.0,
            macd_hist_slope=0.0,
            macd_signal=0.0,
            rsi=50.0,
            adx=25.0,
            boll_position=0.5,
            atr=0.0,
            atr_change_pct=0.0,
            volume_ratio=1.0,
            obv_slope=0.0,
            dist_to_support_pct=0.0,
            dist_to_resistance_pct=0.0,
            market_regime=market_regime,
            similarity=similarity,
        )
