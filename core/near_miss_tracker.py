"""
近似信号追踪器 - 记录相似度刚好低于阈值的"差一点"信号，事后评估收益性
通过追踪 "差点达标的信号后来价格怎么走了" 来判断余弦阈值是否过严

工作流：
1. 入场匹配相似度 < 阈值但在 margin 范围内 → record_near_miss() 记录
2. 每根K线 → evaluate_pending() 检查已记录近似信号的价格后续走势
3. 积累足够评估 → suggest_threshold_adjustment() 给出阈值调整建议

与 RejectionTracker 互补：
- RejectionTracker: 追踪门控拒绝（匹配通过但被门控拦截）
- NearMissTracker: 追踪未匹配（相似度接近但未达阈值）→ 调整相似度阈值
"""

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ── 近似信号分类标签（按距离阈值的远近分桶）──

MISS_BAND_LABELS = {
    "VERY_CLOSE": "极接近 (差<2%)",
    "CLOSE": "较接近 (差2-5%)",
    "MARGINAL": "边缘 (差5-10%)",
}

MISS_BAND_COLORS = {
    "VERY_CLOSE": "#FF9800",   # orange
    "CLOSE": "#FFC107",        # amber
    "MARGINAL": "#9E9E9E",     # grey
}


def classify_miss_band(similarity: float, threshold: float) -> str:
    """
    根据相似度与阈值的差距分类到桶。

    Args:
        similarity: 信号的余弦相似度 (0-1)
        threshold: 当前余弦阈值 (0-1)

    Returns:
        分类标签 ("VERY_CLOSE" / "CLOSE" / "MARGINAL")
    """
    gap = threshold - similarity
    if gap <= 0.02:
        return "VERY_CLOSE"
    elif gap <= 0.05:
        return "CLOSE"
    else:
        return "MARGINAL"


# ── 可调参数定义（余弦 / 融合 / 欧氏 / DTW 阈值）──

ADJUSTABLE_NEAR_MISS: Dict[str, Dict[str, Any]] = {
    "COSINE_THRESHOLD": {
        "loosen_step": -0.01,
        "tighten_step": 0.01,
        "min": 0.80,
        "max": 0.99,
    },
    "FUSION_THRESHOLD": {
        "loosen_step": -0.03,
        "tighten_step": 0.02,
        "min": 0.25,
        "max": 0.75,
    },
    "EUCLIDEAN_MIN_THRESHOLD": {
        "loosen_step": -0.05,
        "tighten_step": 0.03,
        "min": 0.15,
        "max": 0.70,
    },
    "DTW_MIN_THRESHOLD": {
        "loosen_step": -0.05,
        "tighten_step": 0.03,
        "min": 0.10,
        "max": 0.60,
    },
}


@dataclass
class NearMissRecord:
    """
    单次近似信号记录

    记录"差一点就匹配上"的信号完整上下文，以及事后价格评估结果。
    用于判断 "如果放宽阈值让这个信号通过，会盈利还是亏损"。
    """
    id: str                              # UUID
    timestamp: float                     # time.time()
    timestamp_str: str                   # 可读时间字符串
    price_at_signal: float               # 信号出现时的价格
    direction: str                       # "LONG" / "SHORT"
    similarity: float                    # 信号的余弦相似度
    threshold: float                     # 当时的余弦阈值
    miss_band: str                       # "VERY_CLOSE" / "CLOSE" / "MARGINAL"
    market_regime: str                   # 市场状态
    bar_idx_at_signal: int               # 信号出现时的K线索引
    fingerprint: str = ""                # 最佳匹配的原型/模板指纹
    detail: Dict[str, Any] = field(default_factory=dict)

    # ── 事后评估字段 ──
    evaluated: bool = False
    price_after_eval: Optional[float] = None
    bars_waited: int = 0
    price_move_pct: Optional[float] = None   # 候选方向的价格变动百分比
    was_profitable: Optional[bool] = None    # True = 如果放行本可盈利
    evaluation_time: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "price_at_signal": self.price_at_signal,
            "direction": self.direction,
            "similarity": self.similarity,
            "threshold": self.threshold,
            "miss_band": self.miss_band,
            "market_regime": self.market_regime,
            "bar_idx_at_signal": self.bar_idx_at_signal,
            "fingerprint": self.fingerprint,
            "detail": self.detail,
            "evaluated": self.evaluated,
            "price_after_eval": self.price_after_eval,
            "bars_waited": self.bars_waited,
            "price_move_pct": self.price_move_pct,
            "was_profitable": self.was_profitable,
            "evaluation_time": self.evaluation_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'NearMissRecord':
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", 0.0),
            timestamp_str=d.get("timestamp_str", ""),
            price_at_signal=d.get("price_at_signal", 0.0),
            direction=d.get("direction", "LONG"),
            similarity=d.get("similarity", 0.0),
            threshold=d.get("threshold", 0.0),
            miss_band=d.get("miss_band", "MARGINAL"),
            market_regime=d.get("market_regime", "未知"),
            bar_idx_at_signal=d.get("bar_idx_at_signal", 0),
            fingerprint=d.get("fingerprint", ""),
            detail=d.get("detail", {}),
            evaluated=d.get("evaluated", False),
            price_after_eval=d.get("price_after_eval"),
            bars_waited=d.get("bars_waited", 0),
            price_move_pct=d.get("price_move_pct"),
            was_profitable=d.get("was_profitable"),
            evaluation_time=d.get("evaluation_time"),
        )

    def to_ui_dict(self) -> dict:
        """转换为 TrackerCard.update_records 所期望的格式"""
        gap_pct = (self.threshold - self.similarity) * 100
        if self.evaluated:
            if self.was_profitable:
                outcome = f"本可盈利(+{self.price_move_pct:+.2f}%)" if self.price_move_pct else "本可盈利"
            else:
                outcome = f"拒绝正确({self.price_move_pct:+.2f}%)" if self.price_move_pct else "拒绝正确"
        else:
            outcome = "待评估"
        return {
            "timestamp_str": self.timestamp_str,
            "direction": self.direction,
            "reason": f"{MISS_BAND_LABELS.get(self.miss_band, self.miss_band)} (差{gap_pct:.1f}%)",
            "outcome": outcome,
            "move_pct": self.price_move_pct,
            "evaluated": self.evaluated,
            "similarity": self.similarity,
            "threshold": self.threshold,
            "miss_band": self.miss_band,
            "fingerprint": self.fingerprint,
        }


@dataclass
class BandScore:
    """
    按近似信号分桶的累计评分

    追踪各距离桶中近似信号的盈利率，用于判断阈值是否应该放宽。
    - profitable_rate > 0.5 → 这些近似信号多数本可盈利，阈值可能过严
    - profitable_rate < 0.3 → 多数不盈利，阈值合理
    """
    miss_band: str
    profitable_count: int = 0     # 如果放行本可盈利的次数
    unprofitable_count: int = 0   # 放行也不会盈利的次数
    total_profit_pct: float = 0.0    # 累计正向价格变动百分比
    total_loss_pct: float = 0.0      # 累计负向价格变动百分比
    _recent_results: List[bool] = field(default_factory=list)

    RECENT_WINDOW: int = 20

    @property
    def total_count(self) -> int:
        return self.profitable_count + self.unprofitable_count

    @property
    def profitable_rate(self) -> float:
        """近似信号的潜在盈利率"""
        if self.total_count == 0:
            return 0.5
        return self.profitable_count / self.total_count

    @property
    def recent_trend(self) -> float:
        """最近 N 次评估的盈利率（滚动窗口）"""
        if not self._recent_results:
            return 0.5
        return sum(self._recent_results) / len(self._recent_results)

    def record_evaluation(self, was_profitable: bool, move_pct: float):
        """记录一次评估结果"""
        if was_profitable:
            self.profitable_count += 1
            self.total_profit_pct += abs(move_pct)
        else:
            self.unprofitable_count += 1
            self.total_loss_pct += abs(move_pct)

        self._recent_results.append(was_profitable)
        if len(self._recent_results) > self.RECENT_WINDOW:
            self._recent_results.pop(0)

    def to_dict(self) -> dict:
        return {
            "miss_band": self.miss_band,
            "profitable_count": self.profitable_count,
            "unprofitable_count": self.unprofitable_count,
            "total_profit_pct": self.total_profit_pct,
            "total_loss_pct": self.total_loss_pct,
            "recent_results": list(self._recent_results),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BandScore':
        bs = cls(
            miss_band=d.get("miss_band", ""),
            profitable_count=d.get("profitable_count", 0),
            unprofitable_count=d.get("unprofitable_count", 0),
            total_profit_pct=d.get("total_profit_pct", 0.0),
            total_loss_pct=d.get("total_loss_pct", 0.0),
        )
        bs._recent_results = [bool(x) for x in d.get("recent_results", [])]
        return bs


@dataclass
class ThresholdAdjustment:
    """余弦阈值调整审计记录"""
    id: str
    timestamp: float
    timestamp_str: str
    param_key: str
    action: str                      # "loosen" / "tighten"
    old_value: float
    new_value: float
    step_applied: float
    profitable_rate_at_time: float
    recent_trend_at_time: float
    total_evaluations: int
    reason: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "param_key": self.param_key,
            "action": self.action,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "step_applied": self.step_applied,
            "profitable_rate_at_time": self.profitable_rate_at_time,
            "recent_trend_at_time": self.recent_trend_at_time,
            "total_evaluations": self.total_evaluations,
            "reason": self.reason,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ThresholdAdjustment':
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", 0.0),
            timestamp_str=d.get("timestamp_str", ""),
            param_key=d.get("param_key", ""),
            action=d.get("action", ""),
            old_value=d.get("old_value", 0.0),
            new_value=d.get("new_value", 0.0),
            step_applied=d.get("step_applied", 0.0),
            profitable_rate_at_time=d.get("profitable_rate_at_time", 0.0),
            recent_trend_at_time=d.get("recent_trend_at_time", 0.0),
            total_evaluations=d.get("total_evaluations", 0),
            reason=d.get("reason", ""),
            session_id=d.get("session_id", ""),
        )


class NearMissTracker:
    """
    近似信号追踪器

    追踪相似度接近但未达到余弦阈值的入场信号，事后评估价格走势
    判断 "如果放宽阈值让这些信号通过，结果如何"，据此建议阈值调整。

    使用方式：
        tracker = NearMissTracker(persistence_path="data/near_miss_tracker_state.json")
        # 匹配未达标但相似度接近时调用
        tracker.record_near_miss(price=..., direction=..., similarity=..., threshold=..., ...)
        # 每根K线闭合时调用
        tracker.evaluate_pending(current_price=..., current_bar_idx=...)
    """

    DATA_VERSION = 1

    def __init__(self,
                 eval_bars: int = 30,
                 profit_threshold_pct: float = 0.3,
                 near_miss_margin: float = 0.10,
                 max_history: int = 200,
                 min_evaluations_for_suggestion: int = 15,
                 persistence_path: str = "data/near_miss_tracker_state.json"):
        """
        Args:
            eval_bars: 信号出现后等待多少根K线再评估价格走势
            profit_threshold_pct: 判定"本可盈利"的阈值（%，价格向候选方向移动超过此值）
            near_miss_margin: 近似信号捕获范围（与阈值的差距，如0.10=10%）
            max_history: 最多保留多少条已评估的历史记录
            min_evaluations_for_suggestion: 至少需要多少次评估才给出调整建议
            persistence_path: 持久化文件路径
        """
        self._eval_bars = eval_bars
        self._profit_threshold = profit_threshold_pct
        self._near_miss_margin = near_miss_margin
        self._max_history = max_history
        self._min_evaluations = min_evaluations_for_suggestion
        self.persistence_path = persistence_path

        # 已评估的历史记录
        self._history: deque = deque(maxlen=max_history)
        # 等待评估的记录
        self._pending_eval: List[NearMissRecord] = []
        # 按近似信号桶分的累计评分
        self._band_scores: Dict[str, BandScore] = {}

        # 阈值调整审计日志
        self._adjustment_history: List[ThresholdAdjustment] = []
        self._max_adjustment_history: int = 100

        # 会话标识
        self._session_id: str = (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        )

        # 统计
        self.total_near_misses_recorded: int = 0
        self.total_evaluations_done: int = 0
        self.total_adjustments_applied: int = 0

        # 持久化状态追踪
        self._dirty: bool = False
        self._last_save_time: float = 0.0
        self._created_at: float = 0.0  # 首次创建时间戳（记忆开始时间）

        # 去重：同一根K线同方向只记录一次（避免高频重复记录）
        self._last_record_key: Optional[str] = None

        # 从文件加载
        self.load()

    # ─────────────────────────────────────────────
    #  记录近似信号
    # ─────────────────────────────────────────────

    def is_near_miss(self, similarity: float, threshold: float) -> bool:
        """
        判断相似度是否属于"近似信号"范围。

        条件：similarity < threshold 且 similarity >= threshold - margin

        Args:
            similarity: 信号的余弦相似度
            threshold: 当前余弦阈值

        Returns:
            是否属于近似信号
        """
        if similarity >= threshold:
            return False  # 已达标，不是 near miss
        return similarity >= (threshold - self._near_miss_margin)

    def record_near_miss(self,
                         price: float,
                         direction: str,
                         similarity: float,
                         threshold: float,
                         market_regime: str,
                         bar_idx: int,
                         fingerprint: str = "",
                         detail: Optional[Dict[str, Any]] = None) -> Optional[NearMissRecord]:
        """
        记录一次近似信号

        Args:
            price: 信号出现时的当前价格
            direction: 候选方向 "LONG" / "SHORT"
            similarity: 信号的余弦相似度
            threshold: 当时的余弦阈值
            market_regime: 当前市场状态
            bar_idx: 当前K线索引
            fingerprint: 最佳匹配的原型/模板指纹
            detail: 额外上下文信息

        Returns:
            创建的 NearMissRecord，或 None（去重/不在范围内）
        """
        if not self.is_near_miss(similarity, threshold):
            return None

        # 去重：同一根K线、同方向只记一次
        dedup_key = f"{bar_idx}_{direction}"
        if dedup_key == self._last_record_key:
            return None
        self._last_record_key = dedup_key

        miss_band = classify_miss_band(similarity, threshold)
        now = time.time()
        rec = NearMissRecord(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            price_at_signal=price,
            direction=direction,
            similarity=similarity,
            threshold=threshold,
            miss_band=miss_band,
            market_regime=market_regime,
            bar_idx_at_signal=bar_idx,
            fingerprint=fingerprint,
            detail=detail or {},
        )

        self._pending_eval.append(rec)
        self.total_near_misses_recorded += 1
        self._dirty = True

        # 确保 band_scores 中有该桶的条目
        if miss_band not in self._band_scores:
            self._band_scores[miss_band] = BandScore(miss_band=miss_band)

        gap_pct = (threshold - similarity) * 100
        print(f"[NearMissTracker] 记录近似信号: {direction} | "
              f"相似度={similarity:.2%} (阈值={threshold:.2%}, 差{gap_pct:.1f}%) | "
              f"桶={MISS_BAND_LABELS.get(miss_band, miss_band)} | "
              f"指纹={fingerprint}")

        return rec

    # ─────────────────────────────────────────────
    #  事后评估
    # ─────────────────────────────────────────────

    def evaluate_pending(self, current_price: float, current_bar_idx: int) -> List[NearMissRecord]:
        """
        评估待处理的近似信号记录（在每根K线闭合时调用）

        对于每条已等待足够 K 线的记录，检查价格是否朝候选方向移动：
        - 移动 > profit_threshold → 信号本可盈利（阈值可能过严）
        - 移动 < profit_threshold → 信号不盈利（阈值合理）

        Args:
            current_price: 当前价格
            current_bar_idx: 当前K线索引

        Returns:
            本轮新完成评估的记录列表
        """
        newly_evaluated: List[NearMissRecord] = []
        still_pending: List[NearMissRecord] = []

        for rec in self._pending_eval:
            bars_passed = current_bar_idx - rec.bar_idx_at_signal

            if bars_passed >= self._eval_bars:
                # 计算候选方向的价格变动百分比
                if rec.direction == "LONG":
                    move_pct = ((current_price - rec.price_at_signal)
                                / rec.price_at_signal * 100)
                else:  # SHORT
                    move_pct = ((rec.price_at_signal - current_price)
                                / rec.price_at_signal * 100)

                # 判定：价格向候选方向移动超过阈值 → 信号本可盈利
                rec.price_move_pct = round(move_pct, 4)
                rec.was_profitable = (move_pct >= self._profit_threshold)
                rec.evaluated = True
                rec.price_after_eval = current_price
                rec.bars_waited = bars_passed
                rec.evaluation_time = time.time()

                # 更新桶评分
                band = self._band_scores.get(rec.miss_band)
                if band is None:
                    band = BandScore(miss_band=rec.miss_band)
                    self._band_scores[rec.miss_band] = band
                band.record_evaluation(rec.was_profitable, move_pct)

                self._history.append(rec)
                newly_evaluated.append(rec)
                self.total_evaluations_done += 1
            else:
                still_pending.append(rec)

        self._pending_eval = still_pending

        if newly_evaluated:
            self.save()
            eval_summary = ", ".join(
                f"{r.miss_band}={'盈利' if r.was_profitable else '不盈利'}({r.price_move_pct:+.2f}%)"
                for r in newly_evaluated
            )
            print(f"[NearMissTracker] 评估完成 {len(newly_evaluated)} 条: {eval_summary}")

        return newly_evaluated

    # ─────────────────────────────────────────────
    #  查询接口
    # ─────────────────────────────────────────────

    def get_band_score(self, miss_band: str) -> Optional[BandScore]:
        """获取指定桶的累计评分"""
        return self._band_scores.get(miss_band)

    def get_all_band_scores(self) -> Dict[str, BandScore]:
        """获取所有桶的评分"""
        return dict(self._band_scores)

    def get_recent_records(self, limit: int = 20) -> List[NearMissRecord]:
        """获取最近的近似信号记录（含已评估和待评估），按时间倒序"""
        all_records: List[NearMissRecord] = list(self._history) + self._pending_eval
        all_records.sort(key=lambda r: r.timestamp, reverse=True)
        return all_records[:limit]

    def get_pending_count(self) -> int:
        """待评估记录数"""
        return len(self._pending_eval)

    # ─────────────────────────────────────────────
    #  阈值调整建议
    # ─────────────────────────────────────────────

    def suggest_threshold_adjustment(self) -> Optional[Dict[str, Any]]:
        """
        基于近似信号的盈利率，建议余弦阈值调整方向。

        重点关注 VERY_CLOSE 桶（最接近阈值的信号）：
        - profitable_rate > 0.55 → 多数近似信号本可盈利，建议放宽阈值
        - profitable_rate < 0.25 → 多数不盈利，可考虑收紧（保持严格）
        - 其他 → 阈值合理，不建议调整

        Returns:
            调整建议字典，或 None（数据不足 / 无需调整）
        """
        # 优先看最接近阈值的桶
        very_close = self._band_scores.get("VERY_CLOSE")
        close = self._band_scores.get("CLOSE")

        # 合并 VERY_CLOSE + CLOSE 的评估数据
        total_evals = 0
        total_profitable = 0
        for band in [very_close, close]:
            if band:
                total_evals += band.total_count
                total_profitable += band.profitable_count

        if total_evals < self._min_evaluations:
            return None  # 数据不足

        combined_rate = total_profitable / total_evals if total_evals > 0 else 0.5

        # 近期趋势（以 VERY_CLOSE 为主）
        recent_trend = very_close.recent_trend if very_close else 0.5

        if combined_rate > 0.55:
            return {
                "action": "loosen",
                "param_key": "COSINE_THRESHOLD",
                "profitable_rate": round(combined_rate, 3),
                "recent_trend": round(recent_trend, 3),
                "total_evaluations": total_evals,
                "total_missed_profit_pct": round(
                    sum(b.total_profit_pct for b in self._band_scores.values()), 2
                ),
                "adjustable_params": ADJUSTABLE_NEAR_MISS,
                "reason": (
                    f"近似信号盈利率 {combined_rate:.0%}（近期 {recent_trend:.0%}），"
                    f"共评估 {total_evals} 次 — 阈值可能过严，建议适当放宽"
                ),
            }
        elif combined_rate < 0.25 and total_evals >= self._min_evaluations * 2:
            # 收紧建议需要更多数据支撑
            return {
                "action": "tighten",
                "param_key": "COSINE_THRESHOLD",
                "profitable_rate": round(combined_rate, 3),
                "recent_trend": round(recent_trend, 3),
                "total_evaluations": total_evals,
                "total_filtered_loss_pct": round(
                    sum(b.total_loss_pct for b in self._band_scores.values()), 2
                ),
                "adjustable_params": ADJUSTABLE_NEAR_MISS,
                "reason": (
                    f"近似信号盈利率仅 {combined_rate:.0%}（近期 {recent_trend:.0%}），"
                    f"共评估 {total_evals} 次 — 阈值过滤有效，可适当收紧"
                ),
            }

        return None  # 阈值合理

    def get_suggestions(self, config_dict: Dict[str, Any], 
                        fusion_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        计算具体的调整建议列表（TrackerCard UI 可直接使用）。
        支持融合/欧氏/DTW 门槛（需传入 fusion_threshold 等当前值）。

        Args:
            config_dict: 运行时配置字典（通常为 PAPER_TRADING_CONFIG）
            fusion_threshold: 当前融合门槛（用于 FUSION_THRESHOLD 建议）

        Returns:
            逐参数的调整条目列表
        """
        raw = self.suggest_threshold_adjustment()
        if raw is None:
            return []

        action = raw.get("action", "")
        expanded: List[Dict[str, Any]] = []

        for param_key, param_def in ADJUSTABLE_NEAR_MISS.items():
            current_val = config_dict.get(param_key)
            if current_val is None and param_key == "FUSION_THRESHOLD" and fusion_threshold is not None:
                current_val = fusion_threshold
            if current_val is None:
                continue

            step_key = "loosen_step" if action == "loosen" else "tighten_step"
            step = param_def.get(step_key, 0)
            if step == 0:
                continue

            new_value = current_val + step
            min_bound = param_def.get("min", 0.0)
            max_bound = param_def.get("max", 1.0)
            new_value = max(min_bound, min(max_bound, new_value))

            if new_value == current_val:
                continue  # 已在边界

            action_text = "放宽" if action == "loosen" else "收紧"
            labels = {
                "COSINE_THRESHOLD": "余弦相似度阈值",
                "FUSION_THRESHOLD": "融合评分阈值",
                "EUCLIDEAN_MIN_THRESHOLD": "欧氏距离阈值",
                "DTW_MIN_THRESHOLD": "DTW形态阈值",
            }
            expanded.append({
                "param_key": param_key,
                "action": action,
                "action_text": action_text,
                "label": labels.get(param_key, param_key),
                "old_value": current_val,
                "new_value": round(new_value, 4),
                "step": step,
                "min_bound": min_bound,
                "max_bound": max_bound,
                "profitable_rate": raw.get("profitable_rate", 0),
                "recent_trend": raw.get("recent_trend", 0),
                "reason": raw.get("reason", ""),
                "total_evaluations": raw.get("total_evaluations", 0),
            })

        return expanded

    # ─────────────────────────────────────────────
    #  阈值调整应用
    # ─────────────────────────────────────────────

    def apply_adjustment(self,
                         config_dict: Dict[str, Any],
                         param_key: str,
                         new_value: float,
                         reason: str = "") -> Optional[ThresholdAdjustment]:
        """
        应用余弦阈值调整到运行时配置，并记录审计日志。

        Args:
            config_dict: 运行时配置字典（原地修改）
            param_key: 参数名（"COSINE_THRESHOLD"）
            new_value: 目标新值
            reason: 调整原因

        Returns:
            审计记录，或 None（未执行）
        """
        old_value = config_dict.get(param_key)
        if old_value is None:
            print(f"[NearMissTracker] 调整失败: 参数 {param_key} 不存在于配置中")
            return None

        # 安全夹紧
        param_def = ADJUSTABLE_NEAR_MISS.get(param_key, {})
        min_bound = param_def.get("min", 0.0)
        max_bound = param_def.get("max", 1.0)
        clamped = max(min_bound, min(max_bound, new_value))

        if clamped == old_value:
            print(f"[NearMissTracker] 跳过调整: {param_key} 已在目标值 {old_value}")
            return None

        # 确定方向
        action = "loosen" if clamped < old_value else "tighten"

        # 获取评分快照
        very_close = self._band_scores.get("VERY_CLOSE")
        close = self._band_scores.get("CLOSE")
        total_evals = 0
        total_profitable = 0
        for band in [very_close, close]:
            if band:
                total_evals += band.total_count
                total_profitable += band.profitable_count
        rate = total_profitable / total_evals if total_evals > 0 else 0.5
        recent = very_close.recent_trend if very_close else 0.5

        # 实际应用
        config_dict[param_key] = clamped

        now = time.time()
        record = ThresholdAdjustment(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            param_key=param_key,
            action=action,
            old_value=old_value,
            new_value=clamped,
            step_applied=round(clamped - old_value, 6),
            profitable_rate_at_time=round(rate, 3),
            recent_trend_at_time=round(recent, 3),
            total_evaluations=total_evals,
            reason=reason,
            session_id=self._session_id,
        )

        self._adjustment_history.append(record)
        if len(self._adjustment_history) > self._max_adjustment_history:
            self._adjustment_history = self._adjustment_history[-self._max_adjustment_history:]

        self.total_adjustments_applied += 1
        self.save()

        action_text = "放宽" if action == "loosen" else "收紧"
        print(f"[NearMissTracker] 阈值调整已应用: {param_key} "
              f"{old_value} -> {clamped} ({action_text}) | "
              f"近似信号盈利率={rate:.0%} 近期={recent:.0%}")

        return record

    # ─────────────────────────────────────────────
    #  UI 数据导出
    # ─────────────────────────────────────────────

    def get_state_for_ui(self) -> Dict[str, Any]:
        """
        导出用于 UI 展示的状态摘要。

        返回结构可直接序列化到 EngineState 的 near_miss_history / near_miss_scores 字段。
        """
        recent = self.get_recent_records(20)
        return {
            "near_miss_history": [r.to_ui_dict() for r in recent],
            "near_miss_scores": {
                band: {
                    "miss_band": bs.miss_band,
                    "label": MISS_BAND_LABELS.get(bs.miss_band, bs.miss_band),
                    "correct": bs.profitable_count,      # TrackerCard 使用 "correct"
                    "total": bs.total_count,              # TrackerCard 使用 "total"
                    "profitable_count": bs.profitable_count,
                    "unprofitable_count": bs.unprofitable_count,
                    "profitable_rate": round(bs.profitable_rate, 3),
                    "recent_trend": round(bs.recent_trend, 3),
                    "total_profit_pct": round(bs.total_profit_pct, 2),
                    "total_loss_pct": round(bs.total_loss_pct, 2),
                }
                for band, bs in self._band_scores.items()
            },
            "pending_count": self.get_pending_count(),
            "total_near_misses": self.total_near_misses_recorded,
            "total_evaluations": self.total_evaluations_done,
            "total_adjustments": self.total_adjustments_applied,
            "session_adjustments": [
                adj.to_dict() for adj in self._adjustment_history
                if adj.session_id == self._session_id
            ],
        }

    # ─────────────────────────────────────────────
    #  持久化
    # ─────────────────────────────────────────────

    def save(self):
        """持久化到 JSON（原子写入）"""
        try:
            dirpath = os.path.dirname(self.persistence_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            now = time.time()
            self._last_save_time = now
            if self._created_at <= 0:
                self._created_at = now
            data = {
                "config": {
                    "eval_bars": self._eval_bars,
                    "profit_threshold_pct": self._profit_threshold,
                    "near_miss_margin": self._near_miss_margin,
                    "max_history": self._max_history,
                    "min_evaluations_for_suggestion": self._min_evaluations,
                },
                "state": {
                    "data_version": self.DATA_VERSION,
                    "created_at": self._created_at,
                    "last_save_time": self._last_save_time,
                    "total_near_misses_recorded": self.total_near_misses_recorded,
                    "total_evaluations_done": self.total_evaluations_done,
                    "total_adjustments_applied": self.total_adjustments_applied,
                },
                "band_scores": {
                    band: bs.to_dict() for band, bs in self._band_scores.items()
                },
                "history": [rec.to_dict() for rec in self._history],
                "pending_eval": [rec.to_dict() for rec in self._pending_eval],
                "adjustment_history": [adj.to_dict() for adj in self._adjustment_history],
            }

            tmp_path = self.persistence_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.persistence_path)
            self._dirty = False
        except Exception as e:
            print(f"[NearMissTracker] 保存失败: {e}")
            tmp_path = self.persistence_path + ".tmp"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save_if_dirty(self, min_interval_sec: float = 60.0):
        """条件保存：仅在有未保存变更且距上次保存超过指定间隔时才保存"""
        if not self._dirty:
            return
        if time.time() - self._last_save_time < min_interval_sec:
            return
        self.save()

    def load(self):
        """从 JSON 加载持久化数据"""
        if not os.path.exists(self.persistence_path):
            print(f"[NearMissTracker] 持久化文件不存在，使用初始状态")
            return

        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = data.get("state", {})
            self.total_near_misses_recorded = state.get("total_near_misses_recorded", 0)
            self.total_evaluations_done = state.get("total_evaluations_done", 0)
            self.total_adjustments_applied = state.get("total_adjustments_applied", 0)
            self._last_save_time = state.get("last_save_time", 0.0)
            self._created_at = state.get("created_at", 0.0)
            if self._created_at <= 0 and self._last_save_time > 0:
                self._created_at = self._last_save_time

            # 加载桶评分
            scores_data = data.get("band_scores", {})
            self._band_scores = {
                band: BandScore.from_dict(bs_dict)
                for band, bs_dict in scores_data.items()
            }

            # 加载历史记录
            history_data = data.get("history", [])
            self._history = deque(
                (NearMissRecord.from_dict(d) for d in history_data),
                maxlen=self._max_history,
            )

            # 加载待评估记录
            pending_data = data.get("pending_eval", [])
            self._pending_eval = [NearMissRecord.from_dict(d) for d in pending_data]

            # 加载调整审计日志
            adj_data = data.get("adjustment_history", [])
            self._adjustment_history = [ThresholdAdjustment.from_dict(d) for d in adj_data]

            self._dirty = False
            total_records = len(self._history) + len(self._pending_eval)
            last_save_str = (
                datetime.fromtimestamp(self._last_save_time).strftime("%m-%d %H:%M")
                if self._last_save_time > 0 else "未知"
            )
            print(f"[NearMissTracker] 加载成功 (v{state.get('data_version', 1)}): "
                  f"{len(self._band_scores)} 个桶评分, "
                  f"{total_records} 条记录 ({len(self._pending_eval)} 条待评估), "
                  f"累计 {self.total_near_misses_recorded} 次近似信号, "
                  f"上次保存: {last_save_str}")

        except json.JSONDecodeError as e:
            print(f"[NearMissTracker] JSON 解析失败: {e}，使用初始状态")
        except Exception as e:
            print(f"[NearMissTracker] 加载失败: {e}，使用初始状态")

    # ─────────────────────────────────────────────
    #  重置
    # ─────────────────────────────────────────────

    def reset(self):
        """完全重置所有数据"""
        self._history.clear()
        self._pending_eval.clear()
        self._band_scores.clear()
        self._adjustment_history.clear()
        self.total_near_misses_recorded = 0
        self.total_evaluations_done = 0
        self.total_adjustments_applied = 0
        self._dirty = False
        self._last_record_key = None
        self.save()
        print("[NearMissTracker] 已重置全部数据")
