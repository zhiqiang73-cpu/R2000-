"""
门控拒绝追踪器 - 记录被门控拦截的交易信号，事后评估门控准确性
通过追踪 "被拒绝的信号后来价格怎么走了" 来判断门控阈值是否过严/过松

工作流：
1. 信号被门控拒绝 → record_rejection() 记录
2. 每根K线 → evaluate_pending() 检查已记录拒绝的价格后续走势
3. 积累足够评估 → suggest_threshold_adjustment() 给出阈值调整建议

与贝叶斯系统互补：
- BayesianFilter: 从已完成交易学习（胜/亏）→ 调整原型可信度
- RejectionTracker: 从被拒绝交易学习（what-if）→ 调整门控阈值松紧
"""

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ── 门控标签与颜色（UI 用）──

FAIL_CODE_LABELS = {
    "BLOCK_POS": "位置评分拦截",
    "BLOCK_MACD": "MACD门控拦截",
    "BLOCK_BAYES": "贝叶斯过滤拦截",
    "BLOCK_KELLY_NEG": "凯利仓位拦截",
    "FLIP_NO_MATCH": "翻转后无匹配",
    "BLOCK_REGIME_UNKNOWN": "市场状态未知",
    "BLOCK_REGIME_CONFLICT": "市场方向冲突",
}

FAIL_CODE_COLORS = {
    "BLOCK_POS": "#FF9800",       # orange
    "BLOCK_MACD": "#F44336",      # red
    "BLOCK_BAYES": "#9C27B0",     # purple
    "BLOCK_KELLY_NEG": "#673AB7", # deep purple
    "FLIP_NO_MATCH": "#FFC107",   # yellow
    "BLOCK_REGIME_UNKNOWN": "#607D8B",  # blue gray
    "BLOCK_REGIME_CONFLICT": "#795548", # brown
}

GATE_STAGE_LABELS = {
    "position_flip": "位置翻转",
    "position_filter": "位置过滤",
    "indicator_gate": "指标门控",
    "bayesian_gate": "贝叶斯门控",
    "kelly_gate": "凯利门控",
    "regime_filter": "市场状态",
}

# ── 可调门控参数定义（阈值边界 & 调整步长）──

ADJUSTABLE_GATES: Dict[str, Dict[str, Any]] = {
    "BLOCK_POS": {
        "params": {
            "POS_THRESHOLD_LONG":  {"loosen_step": -5, "tighten_step": 5, "min": -50, "max": -10},
            "POS_THRESHOLD_SHORT": {"loosen_step": -5, "tighten_step": 5, "min": -60, "max": -20},
        },
    },
    "BLOCK_MACD": {
        "params": {
            "MACD_SLOPE_MIN": {"loosen_step": -0.001, "tighten_step": 0.001, "min": 0.001, "max": 0.02},
        },
    },
    "BLOCK_BAYES": {
        "params": {
            "BAYESIAN_MIN_WIN_RATE": {"loosen_step": -0.02, "tighten_step": 0.02, "min": 0.30, "max": 0.55},
        },
    },
    # BLOCK_KELLY_NEG 不直接调参，间接通过贝叶斯统计影响
}


@dataclass
class RejectionRecord:
    """
    单次门控拒绝记录

    记录信号被拒绝时的完整上下文，以及事后价格评估结果。
    用于判断 "这次拒绝是对的（避免了亏损）还是错的（错过了利润）"。
    """
    id: str                              # UUID
    timestamp: float                     # time.time()
    timestamp_str: str                   # 可读时间字符串
    price_at_rejection: float            # 拒绝时价格
    direction: str                       # "LONG" / "SHORT"
    fail_code: str                       # "BLOCK_POS", "BLOCK_MACD", etc.
    gate_stage: str                      # "position_filter", "indicator_gate", etc.
    market_regime: str                   # 市场状态
    bar_idx_at_rejection: int            # 拒绝时的K线索引
    detail: Dict[str, Any] = field(default_factory=dict)  # 门控特定信息

    # ── 事后评估字段 ──
    evaluated: bool = False
    price_after_eval: Optional[float] = None   # 评估时价格
    bars_waited: int = 0                       # 等待了多少根K线
    price_move_pct: Optional[float] = None     # 候选方向的价格变动百分比
    was_correct: Optional[bool] = None         # True = 拒绝正确（避免亏损）
    evaluation_time: Optional[float] = None    # 评估时间 (time.time())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "price_at_rejection": self.price_at_rejection,
            "direction": self.direction,
            "fail_code": self.fail_code,
            "gate_stage": self.gate_stage,
            "market_regime": self.market_regime,
            "bar_idx_at_rejection": self.bar_idx_at_rejection,
            "detail": self.detail,
            "evaluated": bool(self.evaluated),
            "price_after_eval": self.price_after_eval,
            "bars_waited": self.bars_waited,
            "price_move_pct": self.price_move_pct,
            "was_correct": bool(self.was_correct) if self.was_correct is not None else None,
            "evaluation_time": self.evaluation_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RejectionRecord':
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", 0.0),
            timestamp_str=d.get("timestamp_str", ""),
            price_at_rejection=d.get("price_at_rejection", 0.0),
            direction=d.get("direction", "LONG"),
            fail_code=d.get("fail_code", ""),
            gate_stage=d.get("gate_stage", ""),
            market_regime=d.get("market_regime", "未知"),
            bar_idx_at_rejection=d.get("bar_idx_at_rejection", 0),
            detail=d.get("detail", {}),
            evaluated=d.get("evaluated", False),
            price_after_eval=d.get("price_after_eval"),
            bars_waited=d.get("bars_waited", 0),
            price_move_pct=d.get("price_move_pct"),
            was_correct=d.get("was_correct"),
            evaluation_time=d.get("evaluation_time"),
        )


@dataclass
class GateScore:
    """
    单个门控的累计评分

    追踪门控拒绝的正确率，用于判断门控是"太严"还是"恰当"。
    - accuracy > 0.8 → 门控很准，可以适当收紧
    - accuracy < 0.4 → 门控拒绝大多是错的，应该放松阈值
    """
    fail_code: str
    correct_count: int = 0          # 拒绝正确（避免了亏损）
    wrong_count: int = 0            # 拒绝错误（错过了利润）
    total_saved_pct: float = 0.0    # 累计避免的亏损百分比
    total_missed_pct: float = 0.0   # 累计错过的利润百分比
    # 滚动窗口追踪近期趋势
    _recent_results: List[bool] = field(default_factory=list)

    RECENT_WINDOW: int = 20  # 滚动窗口大小

    @property
    def total_count(self) -> int:
        return self.correct_count + self.wrong_count

    @property
    def accuracy(self) -> float:
        """门控准确率"""
        if self.total_count == 0:
            return 0.5  # 无数据时中性
        return self.correct_count / self.total_count

    @property
    def recent_trend(self) -> float:
        """最近 N 次评估的准确率（滚动窗口）"""
        if not self._recent_results:
            return 0.5
        return sum(self._recent_results) / len(self._recent_results)

    def record_evaluation(self, was_correct: bool, move_pct: float):
        """记录一次评估结果"""
        if was_correct:
            self.correct_count += 1
            if move_pct < 0:
                self.total_saved_pct += abs(move_pct)
        else:
            self.wrong_count += 1
            self.total_missed_pct += abs(move_pct)

        # 更新滚动窗口
        self._recent_results.append(was_correct)
        if len(self._recent_results) > self.RECENT_WINDOW:
            self._recent_results.pop(0)

    def to_dict(self) -> dict:
        return {
            "fail_code": self.fail_code,
            "correct_count": self.correct_count,
            "wrong_count": self.wrong_count,
            "total_saved_pct": self.total_saved_pct,
            "total_missed_pct": self.total_missed_pct,
            "recent_results": [bool(x) for x in self._recent_results],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GateScore':
        gs = cls(
            fail_code=d.get("fail_code", ""),
            correct_count=d.get("correct_count", 0),
            wrong_count=d.get("wrong_count", 0),
            total_saved_pct=d.get("total_saved_pct", 0.0),
            total_missed_pct=d.get("total_missed_pct", 0.0),
        )
        gs._recent_results = [bool(x) for x in d.get("recent_results", [])]
        return gs


@dataclass
class ThresholdAdjustment:
    """
    阈值调整审计记录

    每次用户确认应用门控阈值调整时，生成一条审计记录。
    用于回溯哪些参数在何时被修改、修改依据如何。
    """
    id: str                          # UUID
    timestamp: float                 # time.time()
    timestamp_str: str               # 可读时间字符串
    fail_code: str                   # 对应的门控代码
    param_key: str                   # 被调整的参数名（如 "POS_THRESHOLD_LONG"）
    action: str                      # "loosen" / "tighten"
    old_value: float                 # 调整前的值
    new_value: float                 # 调整后的值
    step_applied: float              # 实际应用的步长
    accuracy_at_time: float          # 调整时门控准确率
    recent_trend_at_time: float      # 调整时近期准确率趋势
    total_evaluations: int           # 调整时累计评估次数
    reason: str = ""                 # 调整原因（可读描述）
    session_id: str = ""             # 会话标识（可选，用于关联同一会话的调整）

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "fail_code": self.fail_code,
            "param_key": self.param_key,
            "action": self.action,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "step_applied": self.step_applied,
            "accuracy_at_time": self.accuracy_at_time,
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
            fail_code=d.get("fail_code", ""),
            param_key=d.get("param_key", ""),
            action=d.get("action", ""),
            old_value=d.get("old_value", 0.0),
            new_value=d.get("new_value", 0.0),
            step_applied=d.get("step_applied", 0.0),
            accuracy_at_time=d.get("accuracy_at_time", 0.0),
            recent_trend_at_time=d.get("recent_trend_at_time", 0.0),
            total_evaluations=d.get("total_evaluations", 0),
            reason=d.get("reason", ""),
            session_id=d.get("session_id", ""),
        )


class RejectionTracker:
    """
    门控拒绝追踪器

    追踪被各门控（位置评分、MACD、贝叶斯、凯利）拒绝的交易信号，
    事后评估价格走势判断拒绝是否正确，并据此建议阈值调整。

    使用方式：
        tracker = RejectionTracker(persistence_path="data/rejection_tracker_state.json")
        # 在门控拒绝处调用
        tracker.record_rejection(price=..., direction=..., fail_code="BLOCK_MACD", ...)
        # 每根K线闭合时调用
        tracker.evaluate_pending(current_price=..., current_bar_idx=...)
    """

    DATA_VERSION = 1

    def __init__(self,
                 eval_bars: int = 30,
                 profit_threshold_pct: float = 0.3,
                 max_history: int = 200,
                 min_evaluations_for_suggestion: int = 20,
                 persistence_path: str = "data/rejection_tracker_state.json"):
        """
        Args:
            eval_bars: 拒绝后等待多少根K线再评估价格走势
            profit_threshold_pct: 判定"错过利润"的阈值（%，价格向候选方向移动超过此值）
            max_history: 最多保留多少条已评估的历史记录
            min_evaluations_for_suggestion: 每个门控至少需要多少次评估才给出调整建议
            persistence_path: 持久化文件路径
        """
        self._eval_bars = eval_bars
        self._profit_threshold = profit_threshold_pct  # 百分比（0.3 = 0.3%）
        self._max_history = max_history
        self._min_evaluations = min_evaluations_for_suggestion
        self.persistence_path = persistence_path

        # 已评估的历史记录（有限长度）
        self._history: deque = deque(maxlen=max_history)
        # 等待评估的记录（拒绝后还没过够 N 根K线）
        self._pending_eval: List[RejectionRecord] = []
        # 每个 fail_code 的累计评分
        self._gate_scores: Dict[str, GateScore] = {}

        # ── 阈值调整审计日志 ──
        self._adjustment_history: List[ThresholdAdjustment] = []
        self._max_adjustment_history: int = 100  # 最多保留 100 条调整记录

        # 会话标识（用于关联同一次运行中的多个调整）
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        # 统计
        self.total_rejections_recorded: int = 0
        self.total_evaluations_done: int = 0
        self.total_adjustments_applied: int = 0

        # 持久化状态追踪
        self._dirty: bool = False           # 是否有未保存的变更
        self._last_save_time: float = 0.0   # 上次保存时间
        self._created_at: float = 0.0       # 首次创建时间戳（记忆开始时间）

        # 从文件加载（如果持久化文件存在）
        self.load()

    # ─────────────────────────────────────────────
    #  记录拒绝
    # ─────────────────────────────────────────────

    def record_rejection(self,
                         price: float,
                         direction: str,
                         fail_code: str,
                         gate_stage: str,
                         market_regime: str,
                         bar_idx: int,
                         detail: Optional[Dict[str, Any]] = None) -> RejectionRecord:
        """
        记录一次门控拒绝

        Args:
            price: 拒绝时的当前价格
            direction: 候选方向 "LONG" / "SHORT"
            fail_code: 门控失败代码（BLOCK_POS, BLOCK_MACD, etc.）
            gate_stage: 门控阶段（position_filter, indicator_gate, etc.）
            market_regime: 当前市场状态
            bar_idx: 当前K线索引
            detail: 门控特定的上下文信息（如 slope, pos_score 等）

        Returns:
            创建的 RejectionRecord
        """
        now = time.time()
        rec = RejectionRecord(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            price_at_rejection=price,
            direction=direction,
            fail_code=fail_code,
            gate_stage=gate_stage,
            market_regime=market_regime,
            bar_idx_at_rejection=bar_idx,
            detail=detail or {},
        )

        self._pending_eval.append(rec)
        self.total_rejections_recorded += 1
        self._dirty = True

        # 确保 gate_scores 中有该 fail_code 的条目
        if fail_code not in self._gate_scores:
            self._gate_scores[fail_code] = GateScore(fail_code=fail_code)

        return rec

    # ─────────────────────────────────────────────
    #  事后评估
    # ─────────────────────────────────────────────

    def evaluate_pending(self, current_price: float, current_bar_idx: int) -> List[RejectionRecord]:
        """
        评估待处理的拒绝记录（在每根K线闭合时调用）

        对于每条已等待足够 K 线的拒绝记录，检查价格是否朝候选方向移动：
        - 移动 > profit_threshold → 拒绝是错的（错过了利润）
        - 移动 < profit_threshold → 拒绝是对的（避免了亏损或无利可图）

        Args:
            current_price: 当前价格
            current_bar_idx: 当前K线索引

        Returns:
            本轮新完成评估的记录列表
        """
        newly_evaluated: List[RejectionRecord] = []
        still_pending: List[RejectionRecord] = []

        for rec in self._pending_eval:
            bars_passed = current_bar_idx - rec.bar_idx_at_rejection

            if bars_passed >= self._eval_bars:
                # 计算候选方向的价格变动百分比
                if rec.direction == "LONG":
                    move_pct = ((current_price - rec.price_at_rejection)
                                / rec.price_at_rejection * 100)
                else:  # SHORT
                    move_pct = ((rec.price_at_rejection - current_price)
                                / rec.price_at_rejection * 100)

                # 判定：价格向候选方向移动超过阈值 → 拒绝是错的
                rec.price_move_pct = round(move_pct, 4)
                rec.was_correct = bool(move_pct < self._profit_threshold)
                rec.evaluated = True
                rec.price_after_eval = current_price
                rec.bars_waited = bars_passed
                rec.evaluation_time = time.time()

                # 更新门控评分
                gate = self._gate_scores.get(rec.fail_code)
                if gate is None:
                    gate = GateScore(fail_code=rec.fail_code)
                    self._gate_scores[rec.fail_code] = gate
                gate.record_evaluation(rec.was_correct, move_pct)

                # 移入历史
                self._history.append(rec)
                newly_evaluated.append(rec)
                self.total_evaluations_done += 1
            else:
                still_pending.append(rec)

        self._pending_eval = still_pending

        # 有新评估结果时自动保存
        if newly_evaluated:
            self.save()
            eval_summary = ", ".join(
                f"{r.fail_code}={'正确' if r.was_correct else '错误'}({r.price_move_pct:+.2f}%)"
                for r in newly_evaluated
            )
            print(f"[RejectionTracker] 评估完成 {len(newly_evaluated)} 条: {eval_summary}")

        return newly_evaluated

    # ─────────────────────────────────────────────
    #  查询接口
    # ─────────────────────────────────────────────

    def get_gate_score(self, fail_code: str) -> Optional[GateScore]:
        """获取指定门控的累计评分"""
        return self._gate_scores.get(fail_code)

    def get_all_gate_scores(self) -> Dict[str, GateScore]:
        """获取所有门控的评分"""
        return dict(self._gate_scores)

    def get_recent_rejections(self, limit: int = 20) -> List[RejectionRecord]:
        """
        获取最近的拒绝记录（含已评估和待评估）

        优先返回最新的，按时间倒序。
        """
        all_records: List[RejectionRecord] = list(self._history) + self._pending_eval
        all_records.sort(key=lambda r: r.timestamp, reverse=True)
        return all_records[:limit]

    def get_pending_count(self) -> int:
        """待评估记录数"""
        return len(self._pending_eval)

    # ─────────────────────────────────────────────
    #  阈值调整建议
    # ─────────────────────────────────────────────

    def suggest_threshold_adjustment(self, fail_code: str) -> Optional[Dict[str, Any]]:
        """
        基于门控评分，建议阈值调整方向

        仅在积累足够评估（≥ min_evaluations）时才给出建议。
        accuracy < 0.4 → 门控错误率 > 60%，建议放松
        accuracy > 0.8 → 门控正确率 > 80%，可选择收紧
        其他 → 门控表现合理，不建议调整

        Args:
            fail_code: 门控失败代码

        Returns:
            调整建议字典，或 None（数据不足 / 无需调整）
        """
        score = self._gate_scores.get(fail_code)
        if not score or score.total_count < self._min_evaluations:
            return None  # 数据不足

        accuracy = score.accuracy
        recent = score.recent_trend
        gate_def = ADJUSTABLE_GATES.get(fail_code)

        if accuracy < 0.4:
            return {
                "action": "loosen",
                "fail_code": fail_code,
                "accuracy": round(accuracy, 3),
                "recent_trend": round(recent, 3),
                "total_evaluations": score.total_count,
                "total_missed_pct": round(score.total_missed_pct, 2),
                "adjustable_params": gate_def["params"] if gate_def else {},
                "reason": (f"门控 {FAIL_CODE_LABELS.get(fail_code, fail_code)} 准确率仅 "
                           f"{accuracy:.0%}，近期 {recent:.0%}，建议放松阈值"),
            }
        elif accuracy > 0.8:
            return {
                "action": "tighten",
                "fail_code": fail_code,
                "accuracy": round(accuracy, 3),
                "recent_trend": round(recent, 3),
                "total_evaluations": score.total_count,
                "total_saved_pct": round(score.total_saved_pct, 2),
                "adjustable_params": gate_def["params"] if gate_def else {},
                "reason": (f"门控 {FAIL_CODE_LABELS.get(fail_code, fail_code)} 准确率 "
                           f"{accuracy:.0%}，近期 {recent:.0%}，可适当收紧阈值"),
            }

        return None  # 门控表现合理，不需要调整

    def suggest_all_adjustments(self) -> List[Dict[str, Any]]:
        """获取所有门控的调整建议（过滤掉 None）"""
        suggestions = []
        for fail_code in self._gate_scores:
            suggestion = self.suggest_threshold_adjustment(fail_code)
            if suggestion is not None:
                suggestions.append(suggestion)
        return suggestions

    # ─────────────────────────────────────────────
    #  阈值调整计算与应用
    # ─────────────────────────────────────────────

    def compute_adjustment(self, fail_code: str, param_key: str,
                           current_value: float,
                           action: str) -> Optional[Dict[str, Any]]:
        """
        计算具体参数的调整值（含安全边界夹紧），但不实际应用。

        步骤：
        1. 从 ADJUSTABLE_GATES 查找参数定义（step, min, max）
        2. 按 action 方向应用步长
        3. 夹紧到 [min, max] 安全边界
        4. 如果新值等于当前值（已在边界），返回 None

        Args:
            fail_code: 门控失败代码（如 "BLOCK_POS"）
            param_key: 参数名（如 "POS_THRESHOLD_LONG"）
            current_value: 当前参数值
            action: "loosen"（放松）或 "tighten"（收紧）

        Returns:
            计算结果 dict 或 None（已在边界无法调整）
            {
                "param_key": str,
                "action": str,
                "action_text": str,  # 中文（"放宽" / "收紧"）
                "label": str,        # 参数可读名
                "current_value": float,
                "suggested_value": float,
                "step": float,
                "min_bound": float,
                "max_bound": float,
            }
        """
        gate_def = ADJUSTABLE_GATES.get(fail_code)
        if not gate_def:
            return None

        param_def = gate_def.get("params", {}).get(param_key)
        if not param_def:
            return None

        step_key = "loosen_step" if action == "loosen" else "tighten_step"
        step = param_def.get(step_key, 0)
        if step == 0:
            return None

        min_bound = param_def.get("min", float("-inf"))
        max_bound = param_def.get("max", float("inf"))

        # 应用步长
        new_value = current_value + step
        # 夹紧到安全边界
        new_value = max(min_bound, min(max_bound, new_value))

        # 如果已在边界，无法继续调整
        if new_value == current_value:
            return None

        action_text = "放宽" if action == "loosen" else "收紧"
        return {
            "param_key": param_key,
            "action": action,
            "action_text": action_text,
            "label": param_key.replace("_", " ").title(),
            "current_value": current_value,
            "suggested_value": new_value,
            "step": step,
            "min_bound": min_bound,
            "max_bound": max_bound,
        }

    def compute_all_concrete_adjustments(self,
                                         config_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于当前建议和运行时配置，计算所有可调参数的具体调整值。

        将 suggest_all_adjustments() 的每条建议（可含多个参数）展开为
        逐参数的具体调整条目，每条包含当前值、建议值、安全边界等完整信息。

        这是 UI 对话框展示和 apply_adjustment() 前置计算的核心方法。

        Args:
            config_dict: 运行时配置字典（通常为 PAPER_TRADING_CONFIG）

        Returns:
            逐参数的调整条目列表，每条为：
            {
                "fail_code", "param_key", "action", "action_text", "label",
                "current_value", "suggested_value", "step", "min_bound", "max_bound",
                "accuracy", "recent_trend", "reason"
            }
        """
        raw_suggestions = self.suggest_all_adjustments()
        expanded: List[Dict[str, Any]] = []

        for sug in raw_suggestions:
            fail_code = sug.get("fail_code", "")
            action = sug.get("action", "")
            params = sug.get("adjustable_params", {})

            for param_key in params:
                current_val = config_dict.get(param_key)
                if current_val is None:
                    continue

                result = self.compute_adjustment(
                    fail_code=fail_code,
                    param_key=param_key,
                    current_value=current_val,
                    action=action,
                )
                if result is not None:
                    # 附加门控评分信息
                    result["fail_code"] = fail_code
                    result["accuracy"] = sug.get("accuracy", 0)
                    result["recent_trend"] = sug.get("recent_trend", 0)
                    result["reason"] = sug.get("reason", "")
                    result["total_evaluations"] = sug.get("total_evaluations", 0)
                    expanded.append(result)

        return expanded

    def apply_adjustment(self,
                         config_dict: Dict[str, Any],
                         param_key: str,
                         new_value: float,
                         fail_code: str = "",
                         reason: str = "") -> Optional[ThresholdAdjustment]:
        """
        应用单个阈值调整到运行时配置，并记录审计日志。

        安全措施：
        1. 验证参数存在于 config_dict
        2. 验证新值在安全边界内（再次夹紧）
        3. 如果新值等于旧值，不执行
        4. 记录完整审计信息（旧值、新值、门控状态、时间）

        Args:
            config_dict: 运行时配置字典（通常为 PAPER_TRADING_CONFIG），原地修改
            param_key: 要调整的参数名
            new_value: 目标新值
            fail_code: 关联的门控失败代码（审计用）
            reason: 调整原因描述（审计用）

        Returns:
            创建的审计记录，或 None（未执行调整）
        """
        old_value = config_dict.get(param_key)
        if old_value is None:
            print(f"[RejectionTracker] 调整失败: 参数 {param_key} 不存在于配置中")
            return None

        # 再次安全夹紧（防御性编程：确保即使外部传入越界值也不会突破边界）
        clamped_value = self._clamp_to_safe_bounds(fail_code, param_key, new_value)

        if clamped_value == old_value:
            print(f"[RejectionTracker] 跳过调整: {param_key} 已在目标值 {old_value}")
            return None

        # 确定 action 方向
        # 对于 loosen_step 为负数的参数（如 POS_THRESHOLD），new < old 表示 loosen
        action = "loosen" if clamped_value != old_value else "unknown"
        gate_def = ADJUSTABLE_GATES.get(fail_code, {})
        param_def = gate_def.get("params", {}).get(param_key, {})
        loosen_step = param_def.get("loosen_step", 0)
        if loosen_step != 0:
            delta = clamped_value - old_value
            # 如果 delta 与 loosen_step 符号一致，则为 loosen
            if (delta > 0 and loosen_step > 0) or (delta < 0 and loosen_step < 0):
                action = "loosen"
            else:
                action = "tighten"

        # 获取门控评分快照
        score = self._gate_scores.get(fail_code)
        accuracy = score.accuracy if score else 0.0
        recent_trend = score.recent_trend if score else 0.0
        total_evals = score.total_count if score else 0

        # 实际应用
        config_dict[param_key] = clamped_value

        # 创建审计记录
        now = time.time()
        record = ThresholdAdjustment(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            fail_code=fail_code,
            param_key=param_key,
            action=action,
            old_value=old_value,
            new_value=clamped_value,
            step_applied=round(clamped_value - old_value, 6),
            accuracy_at_time=round(accuracy, 3),
            recent_trend_at_time=round(recent_trend, 3),
            total_evaluations=total_evals,
            reason=reason,
            session_id=self._session_id,
        )

        self._adjustment_history.append(record)
        # 维护最大长度
        if len(self._adjustment_history) > self._max_adjustment_history:
            self._adjustment_history = self._adjustment_history[-self._max_adjustment_history:]

        self.total_adjustments_applied += 1

        # 持久化
        self.save()

        action_text = "放宽" if action == "loosen" else "收紧"
        print(f"[RejectionTracker] 阈值调整已应用: {param_key} "
              f"{old_value} -> {clamped_value} ({action_text}) | "
              f"门控准确率={accuracy:.0%} 近期={recent_trend:.0%}")

        return record

    def _clamp_to_safe_bounds(self, fail_code: str, param_key: str,
                              value: float) -> float:
        """
        将值夹紧到安全边界内。

        如果在 ADJUSTABLE_GATES 中找不到对应定义，原值返回（不做限制）。
        """
        gate_def = ADJUSTABLE_GATES.get(fail_code, {})
        param_def = gate_def.get("params", {}).get(param_key, {})
        if not param_def:
            # 尝试遍历所有门控查找该参数（跨门控参数查找）
            for _fc, _gd in ADJUSTABLE_GATES.items():
                if param_key in _gd.get("params", {}):
                    param_def = _gd["params"][param_key]
                    break
        if not param_def:
            return value

        min_bound = param_def.get("min", float("-inf"))
        max_bound = param_def.get("max", float("inf"))
        return max(min_bound, min(max_bound, value))

    # ─────────────────────────────────────────────
    #  会话结束报告
    # ─────────────────────────────────────────────

    def generate_session_report(self,
                                config_dict: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
        """
        生成交易会话结束报告。

        在引擎停止时调用，汇总本次会话的门控表现和调整建议。
        报告内容：
        - 各门控评分汇总（正确/错误/准确率/近期趋势）
        - 待处理的调整建议（如有足够数据）
        - 本次会话中已应用的调整记录
        - 整体统计（总拒绝数、总评估数、总调整数）

        Args:
            config_dict: 运行时配置字典（用于计算具体建议值），可选

        Returns:
            完整的会话报告 dict
        """
        # 门控评分汇总
        gate_summaries: Dict[str, Dict[str, Any]] = {}
        for code, gs in self._gate_scores.items():
            gate_summaries[code] = {
                "fail_code": code,
                "label": FAIL_CODE_LABELS.get(code, code),
                "correct_count": gs.correct_count,
                "wrong_count": gs.wrong_count,
                "total_count": gs.total_count,
                "accuracy": round(gs.accuracy, 3),
                "recent_trend": round(gs.recent_trend, 3),
                "total_saved_pct": round(gs.total_saved_pct, 2),
                "total_missed_pct": round(gs.total_missed_pct, 2),
            }

        # 待处理建议
        pending_suggestions: List[Dict[str, Any]] = []
        if config_dict is not None:
            pending_suggestions = self.compute_all_concrete_adjustments(config_dict)

        # 本次会话的调整记录
        session_adjustments = [
            adj.to_dict() for adj in self._adjustment_history
            if adj.session_id == self._session_id
        ]

        report = {
            "session_id": self._session_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": {
                "total_rejections": self.total_rejections_recorded,
                "total_evaluations": self.total_evaluations_done,
                "total_adjustments_applied": self.total_adjustments_applied,
                "pending_evaluations": len(self._pending_eval),
                "history_size": len(self._history),
            },
            "gate_summaries": gate_summaries,
            "pending_suggestions": pending_suggestions,
            "session_adjustments": session_adjustments,
        }

        return report

    # ─────────────────────────────────────────────
    #  调整历史查询
    # ─────────────────────────────────────────────

    def get_adjustment_history(self, limit: int = 50) -> List[ThresholdAdjustment]:
        """获取最近的阈值调整记录（按时间倒序）"""
        sorted_history = sorted(self._adjustment_history,
                                key=lambda a: a.timestamp, reverse=True)
        return sorted_history[:limit]

    def get_session_adjustments(self) -> List[ThresholdAdjustment]:
        """获取本次会话中的所有调整记录"""
        return [adj for adj in self._adjustment_history
                if adj.session_id == self._session_id]

    def get_param_adjustment_trail(self, param_key: str) -> List[ThresholdAdjustment]:
        """获取指定参数的全部调整轨迹（按时间正序）"""
        return [adj for adj in self._adjustment_history
                if adj.param_key == param_key]

    # ─────────────────────────────────────────────
    #  UI 数据导出
    # ─────────────────────────────────────────────

    def get_state_for_ui(self) -> Dict[str, Any]:
        """
        导出用于 UI 展示的状态摘要

        返回结构可直接序列化到 EngineState 的 rejection_history / gate_scores 字段。
        """
        return {
            "rejection_history": [r.to_dict() for r in self.get_recent_rejections(20)],
            "gate_scores": {
                code: {
                    "fail_code": gs.fail_code,
                    "correct_count": gs.correct_count,
                    "wrong_count": gs.wrong_count,
                    "accuracy": round(gs.accuracy, 3),
                    "recent_trend": round(gs.recent_trend, 3),
                    "total_saved_pct": round(gs.total_saved_pct, 2),
                    "total_missed_pct": round(gs.total_missed_pct, 2),
                    "total_count": gs.total_count,
                }
                for code, gs in self._gate_scores.items()
            },
            "pending_count": self.get_pending_count(),
            "total_rejections": self.total_rejections_recorded,
            "total_evaluations": self.total_evaluations_done,
            "total_adjustments": self.total_adjustments_applied,
            "session_adjustments": [
                adj.to_dict() for adj in self.get_session_adjustments()
            ],
        }

    # ─────────────────────────────────────────────
    #  持久化
    # ─────────────────────────────────────────────

    def save(self):
        """
        持久化到 JSON（原子写入：先写临时文件再重命名，防止崩溃导致文件损坏）

        保存内容：
        - config: 当前配置参数
        - state: 数据版本、累计计数器、最后保存时间
        - gate_scores: 各门控累计评分
        - history: 已评估的历史记录
        - pending_eval: 待评估的记录
        """
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
                    "max_history": self._max_history,
                    "min_evaluations_for_suggestion": self._min_evaluations,
                },
                "state": {
                    "data_version": self.DATA_VERSION,
                    "created_at": self._created_at,
                    "last_save_time": self._last_save_time,
                    "total_rejections_recorded": self.total_rejections_recorded,
                    "total_evaluations_done": self.total_evaluations_done,
                    "total_adjustments_applied": self.total_adjustments_applied,
                },
                "gate_scores": {
                    code: gs.to_dict() for code, gs in self._gate_scores.items()
                },
                "history": [rec.to_dict() for rec in self._history],
                "pending_eval": [rec.to_dict() for rec in self._pending_eval],
                "adjustment_history": [adj.to_dict() for adj in self._adjustment_history],
            }

            # 原子写入：先写 .tmp 再重命名，防止写入中途崩溃导致文件损坏
            tmp_path = self.persistence_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # os.replace 是原子操作（同一文件系统内），覆盖目标文件
            os.replace(tmp_path, self.persistence_path)
            self._dirty = False
        except Exception as e:
            print(f"[RejectionTracker] 保存失败: {e}")
            # 清理残留的临时文件
            tmp_path = self.persistence_path + ".tmp"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save_if_dirty(self, min_interval_sec: float = 60.0):
        """
        条件保存：仅在有未保存变更且距上次保存超过指定间隔时才保存

        用于周期性调用（如每根K线结束时），避免频繁 IO 又不丢失待评估记录。
        record_rejection() 会标记 _dirty，evaluate_pending() 内部已自动 save()，
        此方法主要覆盖 "只有新 pending 但还没到评估时间" 的场景。

        Args:
            min_interval_sec: 最小保存间隔（秒），默认 60 秒
        """
        if not self._dirty:
            return
        if time.time() - self._last_save_time < min_interval_sec:
            return
        self.save()

    def load(self):
        """
        从 JSON 加载持久化数据

        加载流程：
        1. 检查文件是否存在
        2. 解析 JSON 数据
        3. 验证数据版本（预留未来迁移接口）
        4. 恢复：状态计数器、门控评分、历史记录、待评估记录
        """
        if not os.path.exists(self.persistence_path):
            print(f"[RejectionTracker] 持久化文件不存在，使用初始状态")
            return

        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载并验证数据版本
            state = data.get("state", {})
            loaded_version = state.get("data_version", 1)
            self.total_rejections_recorded = state.get("total_rejections_recorded", 0)
            self.total_evaluations_done = state.get("total_evaluations_done", 0)
            self.total_adjustments_applied = state.get("total_adjustments_applied", 0)
            self._last_save_time = state.get("last_save_time", 0.0)
            self._created_at = state.get("created_at", 0.0)
            if self._created_at <= 0 and self._last_save_time > 0:
                self._created_at = self._last_save_time

            # 版本迁移预留（当前 v1，未来升级时在此添加迁移逻辑）
            if loaded_version < self.DATA_VERSION:
                print(f"[RejectionTracker] 数据版本 v{loaded_version} → v{self.DATA_VERSION}，执行迁移...")
                self._migrate_data(loaded_version, data)

            # 加载门控评分
            scores_data = data.get("gate_scores", {})
            self._gate_scores = {
                code: GateScore.from_dict(gs_dict)
                for code, gs_dict in scores_data.items()
            }

            # 加载历史记录
            history_data = data.get("history", [])
            self._history = deque(
                (RejectionRecord.from_dict(d) for d in history_data),
                maxlen=self._max_history,
            )

            # 加载待评估记录
            pending_data = data.get("pending_eval", [])
            self._pending_eval = [RejectionRecord.from_dict(d) for d in pending_data]

            # 加载阈值调整审计日志
            adj_data = data.get("adjustment_history", [])
            self._adjustment_history = [ThresholdAdjustment.from_dict(d) for d in adj_data]

            self._dirty = False
            total_records = len(self._history) + len(self._pending_eval)
            last_save_str = (datetime.fromtimestamp(self._last_save_time).strftime("%m-%d %H:%M")
                             if self._last_save_time > 0 else "未知")
            print(f"[RejectionTracker] 加载成功 (v{loaded_version}): "
                  f"{len(self._gate_scores)} 个门控评分, "
                  f"{total_records} 条记录 ({len(self._pending_eval)} 条待评估), "
                  f"累计 {self.total_rejections_recorded} 次拒绝, "
                  f"上次保存: {last_save_str}")

        except json.JSONDecodeError as e:
            print(f"[RejectionTracker] JSON 解析失败: {e}，使用初始状态")
        except Exception as e:
            print(f"[RejectionTracker] 加载失败: {e}，使用初始状态")

    def _migrate_data(self, from_version: int, data: dict):
        """
        数据版本迁移（预留接口）

        当 DATA_VERSION 升级时，在此添加迁移逻辑。
        例如：if from_version < 2: ... 对旧字段做转换 ...
        迁移完成后自动保存新版本。
        """
        # 当前 DATA_VERSION = 1，无需迁移
        # 未来示例：
        # if from_version < 2:
        #     # v1 → v2: 添加新字段、转换旧格式等
        #     for rec_dict in data.get("history", []):
        #         rec_dict.setdefault("new_field", default_value)
        print(f"[RejectionTracker] 迁移完成 v{from_version} → v{self.DATA_VERSION}")

    # ─────────────────────────────────────────────
    #  重置
    # ─────────────────────────────────────────────

    def reset(self):
        """完全重置所有数据（需手动确认后调用）"""
        self._history.clear()
        self._pending_eval.clear()
        self._gate_scores.clear()
        self._adjustment_history.clear()
        self.total_rejections_recorded = 0
        self.total_evaluations_done = 0
        self.total_adjustments_applied = 0
        self._dirty = False
        self.save()
        print("[RejectionTracker] 已重置全部数据")
