"""
止盈止损事后评估追踪器 - 记录每次 TP/SL 触发后的价格走势，
判断止盈止损距离是否合理，建议 ATR 倍数调整。

工作流：
1. 交易被止盈/止损平仓 → record_exit() 记录
2. 每根K线 → evaluate_pending() 检查平仓后 N 根K线价格走势
3. 积累足够评估 → get_suggestions() 给出 ATR 倍数调整建议

评估逻辑：
- STOP_LOSS 后价格反转（朝原方向走）→ SL 过紧（too_tight）
- STOP_LOSS 后价格继续下跌 → SL 正确（correct）
- TAKE_PROFIT 后价格继续上涨 → TP 过紧（too_tight）
- TAKE_PROFIT 后价格反转 → TP 正确（correct）
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ── 平仓原因标签与颜色（UI 用）──

TPSL_REASON_LABELS = {
    "STOP_LOSS": "止损触发",
    "TAKE_PROFIT": "止盈触发",
    "TRAILING_STOP": "追踪止损",
}

TPSL_REASON_COLORS = {
    "STOP_LOSS": "#F44336",     # red
    "TAKE_PROFIT": "#4CAF50",   # green
    "TRAILING_STOP": "#FF9800", # orange
}

# ── 可调参数定义（ATR 倍数边界 & 调整步长）──

ADJUSTABLE_TPSL_PARAMS: Dict[str, Dict[str, Any]] = {
    "STOP_LOSS_ATR": {
        "label": "止损ATR倍数",
        "loosen_step": 0.2,     # 放宽 = 增大ATR倍数 → 止损更远
        "tighten_step": -0.2,   # 收紧 = 减小ATR倍数 → 止损更近
        "min": 1.0,
        "max": 4.0,
    },
    "TAKE_PROFIT_ATR": {
        "label": "止盈ATR倍数",
        "loosen_step": 0.3,     # 放宽 = 增大ATR倍数 → 止盈更远
        "tighten_step": -0.3,   # 收紧 = 减小ATR倍数 → 止盈更近
        "min": 2.0,
        "max": 6.0,
    },
    "ATR_SL_MULTIPLIER": {
        "label": "ATR止损乘数",
        "loosen_step": 0.2,     # 放宽 = 增大乘数 → 止损更宽
        "tighten_step": -0.2,   # 收紧 = 减小乘数 → 止损更紧
        "min": 1.5,
        "max": 4.0,
    },
    "MIN_SL_PCT": {
        "label": "最小止损百分比",
        "loosen_step": 0.0005,   # 放宽 = 增大最小止损距离
        "tighten_step": -0.0005, # 收紧
        "min": 0.0005,
        "max": 0.005,
    },
}

# 平仓原因 → 关联可调参数
REASON_TO_PARAMS = {
    "STOP_LOSS": ["STOP_LOSS_ATR", "ATR_SL_MULTIPLIER", "MIN_SL_PCT"],
    "TAKE_PROFIT": ["TAKE_PROFIT_ATR"],
    "TRAILING_STOP": ["STOP_LOSS_ATR", "ATR_SL_MULTIPLIER"],
}


@dataclass
class TPSLRecord:
    """
    单次止盈/止损平仓记录

    记录平仓时的完整上下文（含 ATR、TP/SL 距离等），
    以及事后价格评估结果。
    """
    id: str                              # UUID
    timestamp: float                     # time.time()
    timestamp_str: str                   # 可读时间字符串
    exit_price: float                    # 平仓价格
    direction: str                       # "LONG" / "SHORT"
    reason: str                          # "STOP_LOSS" / "TAKE_PROFIT" / "TRAILING_STOP"
    bar_idx_at_exit: int                 # 平仓时的K线索引
    detail: Dict[str, Any] = field(default_factory=dict)  # 附加信息

    # ── 交易上下文 ──
    entry_price: float = 0.0             # 入场价格
    entry_atr: float = 0.0               # 入场时 ATR
    sl_price: float = 0.0                # 实际止损价
    tp_price: float = 0.0                # 实际止盈价
    sl_distance_pct: float = 0.0         # 止损距入场价百分比
    tp_distance_pct: float = 0.0         # 止盈距入场价百分比
    sl_atr_multiple: float = 0.0         # 止损 = 几倍 ATR
    tp_atr_multiple: float = 0.0         # 止盈 = 几倍 ATR
    profit_pct: float = 0.0              # 平仓收益率 (%)
    peak_profit_pct: float = 0.0         # 持仓期间峰值收益率 (%)
    hold_bars: int = 0                   # 持仓K线数
    trailing_stage: int = 0              # 追踪止损阶段 (0/1/2/3)

    # ── 事后评估字段 ──
    evaluated: bool = False
    price_after_eval: Optional[float] = None   # 评估时价格
    bars_waited: int = 0                       # 等待了多少根K线
    move_pct: Optional[float] = None           # 平仓后价格变动（原方向，正=继续走，负=反转）
    outcome: Optional[str] = None              # "too_tight" / "correct" / "neutral"
    evaluation_time: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "exit_price": self.exit_price,
            "direction": self.direction,
            "reason": self.reason,
            "bar_idx_at_exit": self.bar_idx_at_exit,
            "detail": self.detail,
            "entry_price": self.entry_price,
            "entry_atr": self.entry_atr,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "sl_distance_pct": self.sl_distance_pct,
            "tp_distance_pct": self.tp_distance_pct,
            "sl_atr_multiple": self.sl_atr_multiple,
            "tp_atr_multiple": self.tp_atr_multiple,
            "profit_pct": self.profit_pct,
            "peak_profit_pct": self.peak_profit_pct,
            "hold_bars": self.hold_bars,
            "trailing_stage": self.trailing_stage,
            "evaluated": self.evaluated,
            "price_after_eval": self.price_after_eval,
            "bars_waited": self.bars_waited,
            "move_pct": self.move_pct,
            "outcome": self.outcome,
            "evaluation_time": self.evaluation_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TPSLRecord":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", 0.0),
            timestamp_str=d.get("timestamp_str", ""),
            exit_price=d.get("exit_price", 0.0),
            direction=d.get("direction", "LONG"),
            reason=d.get("reason", ""),
            bar_idx_at_exit=d.get("bar_idx_at_exit", 0),
            detail=d.get("detail", {}),
            entry_price=d.get("entry_price", 0.0),
            entry_atr=d.get("entry_atr", 0.0),
            sl_price=d.get("sl_price", 0.0),
            tp_price=d.get("tp_price", 0.0),
            sl_distance_pct=d.get("sl_distance_pct", 0.0),
            tp_distance_pct=d.get("tp_distance_pct", 0.0),
            sl_atr_multiple=d.get("sl_atr_multiple", 0.0),
            tp_atr_multiple=d.get("tp_atr_multiple", 0.0),
            profit_pct=d.get("profit_pct", 0.0),
            peak_profit_pct=d.get("peak_profit_pct", 0.0),
            hold_bars=d.get("hold_bars", 0),
            trailing_stage=d.get("trailing_stage", 0),
            evaluated=d.get("evaluated", False),
            price_after_eval=d.get("price_after_eval"),
            bars_waited=d.get("bars_waited", 0),
            move_pct=d.get("move_pct"),
            outcome=d.get("outcome"),
            evaluation_time=d.get("evaluation_time"),
        )


@dataclass
class TPSLScore:
    """
    单个平仓原因的累计评分

    追踪该类型平仓的正确率：
    - accuracy < 0.4 → TP/SL 过紧（大多数平仓后价格继续朝原方向走）
    - accuracy > 0.8 → TP/SL 合适或过宽
    """
    reason: str
    correct: int = 0         # 平仓正确（价格反转，TP/SL 距离合适）
    too_tight: int = 0       # 平仓过紧（价格继续走，错失更多利润/SL太近被扫）
    neutral: int = 0         # 中性（价格波动不大）
    # 累计统计
    total_missed_pct: float = 0.0     # 累计因过紧错过的价格变动
    total_saved_pct: float = 0.0      # 累计因正确平仓避免的价格反转
    # 滚动窗口
    _recent_results: List[bool] = field(default_factory=list)
    RECENT_WINDOW: int = 20

    @property
    def total(self) -> int:
        return self.correct + self.too_tight + self.neutral

    @property
    def accuracy(self) -> float:
        """正确率（correct / (correct + too_tight)，中性不计入）"""
        denom = self.correct + self.too_tight
        return (self.correct / denom) if denom > 0 else 0.5

    @property
    def recent_trend(self) -> float:
        """最近 N 次评估的准确率（滚动窗口）"""
        if not self._recent_results:
            return 0.5
        return sum(self._recent_results) / len(self._recent_results)

    def record_evaluation(self, was_correct: bool, move_pct: float):
        """记录一次评估结果"""
        if was_correct:
            self.correct += 1
            if move_pct < 0:
                self.total_saved_pct += abs(move_pct)
        else:
            self.too_tight += 1
            self.total_missed_pct += abs(move_pct)
        # 更新滚动窗口
        self._recent_results.append(was_correct)
        if len(self._recent_results) > self.RECENT_WINDOW:
            self._recent_results.pop(0)

    def to_dict(self) -> dict:
        return {
            "reason": self.reason,
            "correct": self.correct,
            "too_tight": self.too_tight,
            "neutral": self.neutral,
            "total": self.total,
            "accuracy": round(self.accuracy, 3),
            "recent_trend": round(self.recent_trend, 3),
            "total_missed_pct": round(self.total_missed_pct, 2),
            "total_saved_pct": round(self.total_saved_pct, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TPSLScore":
        score = cls(
            reason=d.get("reason", ""),
            correct=d.get("correct", 0),
            too_tight=d.get("too_tight", 0),
            neutral=d.get("neutral", 0),
            total_missed_pct=d.get("total_missed_pct", 0.0),
            total_saved_pct=d.get("total_saved_pct", 0.0),
        )
        score._recent_results = [bool(x) for x in d.get("recent_results", [])]
        return score


class TPSLTracker:
    """
    止盈止损事后评估追踪器

    记录每次 TP/SL 触发后的价格走势，判断止盈止损距离是否合理。
    通过追踪 "平仓后价格是继续走还是反转" 来判断 ATR 倍数设置。

    使用方式：
        tracker = TPSLTracker(persistence_path="data/tpsl_tracker_state.json")
        # 交易被 TP/SL 平仓时调用
        tracker.record_exit(direction="LONG", exit_price=..., bar_idx=...,
                           reason="STOP_LOSS", entry_price=..., entry_atr=..., ...)
        # 每根K线闭合时调用
        tracker.evaluate_pending(current_price=..., current_bar_idx=...)
    """

    DATA_VERSION = 1

    DEFAULT_PARAMS = ADJUSTABLE_TPSL_PARAMS

    REASON_TO_PARAMS = REASON_TO_PARAMS

    def __init__(self,
                 eval_bars: int = 30,
                 move_threshold_pct: float = 0.5,
                 min_evals_for_suggest: int = 20,
                 max_history: int = 200,
                 persistence_path: Optional[str] = None):
        """
        Args:
            eval_bars: 平仓后等待多少根K线再评估价格走势
            move_threshold_pct: 判定"过紧/正确"的阈值（%，价格变动超过此值）
            min_evals_for_suggest: 每个原因至少需要多少次评估才给出调整建议
            max_history: 最多保留多少条已评估的历史记录
            persistence_path: 持久化文件路径
        """
        self._eval_bars = int(eval_bars)
        self._move_threshold = float(move_threshold_pct)
        self._min_evals = int(min_evals_for_suggest)
        self._max_history = max_history
        self._persistence_path = persistence_path

        # 已评估的历史记录
        self._history: List[TPSLRecord] = []
        # 等待评估的记录
        self._pending: List[TPSLRecord] = []
        # 每个 reason 的累计评分
        self._scores: Dict[str, TPSLScore] = {}

        # 统计
        self.total_records: int = 0
        self.total_evaluations: int = 0

        # 持久化状态
        self._dirty: bool = False
        self._last_save_time: float = 0.0
        self._created_at: float = 0.0  # 首次创建时间戳（记忆开始时间）

        if self._persistence_path:
            self.load()

    # ─────────────────────────────────────────────
    #  记录平仓
    # ─────────────────────────────────────────────

    def record_exit(self,
                    direction: str,
                    exit_price: float,
                    bar_idx: int,
                    reason: str,
                    entry_price: float = 0.0,
                    entry_atr: float = 0.0,
                    sl_price: float = 0.0,
                    tp_price: float = 0.0,
                    profit_pct: float = 0.0,
                    peak_profit_pct: float = 0.0,
                    hold_bars: int = 0,
                    trailing_stage: int = 0,
                    detail: Optional[Dict[str, Any]] = None) -> TPSLRecord:
        """
        记录一次 TP/SL 平仓

        Args:
            direction: "LONG" / "SHORT"
            exit_price: 平仓价格
            bar_idx: 平仓时K线索引
            reason: "STOP_LOSS" / "TAKE_PROFIT" / "TRAILING_STOP"
            entry_price: 入场价格
            entry_atr: 入场时 ATR
            sl_price: 实际止损价
            tp_price: 实际止盈价
            profit_pct: 平仓收益率 (%)
            peak_profit_pct: 持仓期间峰值收益率 (%)
            hold_bars: 持仓K线数
            trailing_stage: 追踪止损阶段
            detail: 附加详情
        """
        now = time.time()

        # 计算 TP/SL 距离和 ATR 倍数
        sl_distance_pct = 0.0
        tp_distance_pct = 0.0
        sl_atr_multiple = 0.0
        tp_atr_multiple = 0.0

        if entry_price > 0:
            if sl_price > 0:
                sl_distance_pct = abs(sl_price - entry_price) / entry_price * 100
            if tp_price > 0:
                tp_distance_pct = abs(tp_price - entry_price) / entry_price * 100
        if entry_atr > 0 and entry_price > 0:
            if sl_price > 0:
                sl_atr_multiple = round(abs(sl_price - entry_price) / entry_atr, 2)
            if tp_price > 0:
                tp_atr_multiple = round(abs(tp_price - entry_price) / entry_atr, 2)

        rec = TPSLRecord(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=time.strftime("%m-%d %H:%M:%S", time.localtime(now)),
            exit_price=float(exit_price),
            direction=direction,
            reason=reason,
            bar_idx_at_exit=int(bar_idx),
            detail=detail or {},
            entry_price=float(entry_price),
            entry_atr=float(entry_atr),
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            sl_distance_pct=round(sl_distance_pct, 4),
            tp_distance_pct=round(tp_distance_pct, 4),
            sl_atr_multiple=sl_atr_multiple,
            tp_atr_multiple=tp_atr_multiple,
            profit_pct=round(float(profit_pct), 2),
            peak_profit_pct=round(float(peak_profit_pct), 2),
            hold_bars=int(hold_bars),
            trailing_stage=int(trailing_stage),
        )

        self._pending.append(rec)
        self.total_records += 1
        self._dirty = True

        # 确保 scores 中有该 reason 的条目
        if reason not in self._scores:
            self._scores[reason] = TPSLScore(reason=reason)

        label = TPSL_REASON_LABELS.get(reason, reason)
        print(f"[TPSLTracker] 记录 {label}: {direction} | "
              f"入场={entry_price:.2f} 出场={exit_price:.2f} | "
              f"收益={profit_pct:+.2f}% | "
              f"SL={sl_atr_multiple:.1f}×ATR TP={tp_atr_multiple:.1f}×ATR")

        return rec

    # ─────────────────────────────────────────────
    #  事后评估
    # ─────────────────────────────────────────────

    def evaluate_pending(self, current_price: float, bar_idx: int) -> List[TPSLRecord]:
        """
        评估待处理的平仓记录（在每根K线闭合时调用）

        评估逻辑：
        - STOP_LOSS/TRAILING_STOP 后：
          - 价格朝原方向走（反转）> threshold → SL 过紧（too_tight）
          - 价格继续逆向走 → SL 正确（correct）
        - TAKE_PROFIT 后：
          - 价格继续朝原方向走 > threshold → TP 过紧（too_tight）
          - 价格反转走 → TP 正确（correct）

        Args:
            current_price: 当前价格
            bar_idx: 当前K线索引

        Returns:
            本轮新完成评估的记录列表
        """
        newly_evaluated: List[TPSLRecord] = []
        still_pending: List[TPSLRecord] = []

        for rec in self._pending:
            bars_passed = bar_idx - rec.bar_idx_at_exit

            if bars_passed >= self._eval_bars:
                # 计算平仓后沿原交易方向的价格变动
                # 正值 = 价格朝原交易方向继续移动
                # 负值 = 价格反转（朝对方向移动）
                if rec.direction == "LONG":
                    move_pct = (current_price - rec.exit_price) / rec.exit_price * 100
                else:  # SHORT
                    move_pct = (rec.exit_price - current_price) / rec.exit_price * 100

                rec.move_pct = round(move_pct, 4)
                rec.price_after_eval = current_price
                rec.bars_waited = bars_passed
                rec.evaluated = True
                rec.evaluation_time = time.time()

                # 判定逻辑
                score = self._scores.get(rec.reason)
                if score is None:
                    score = TPSLScore(reason=rec.reason)
                    self._scores[rec.reason] = score

                if rec.reason in ("STOP_LOSS", "TRAILING_STOP"):
                    # 止损后：价格朝原方向走 = SL 过紧（错过了反转利润）
                    if move_pct >= self._move_threshold:
                        rec.outcome = "too_tight"
                        score.record_evaluation(was_correct=False, move_pct=move_pct)
                    elif move_pct <= -self._move_threshold:
                        rec.outcome = "correct"
                        score.record_evaluation(was_correct=True, move_pct=move_pct)
                    else:
                        rec.outcome = "neutral"
                        score.neutral += 1
                elif rec.reason == "TAKE_PROFIT":
                    # 止盈后：价格继续朝原方向走 = TP 过紧（错过了更多利润）
                    if move_pct >= self._move_threshold:
                        rec.outcome = "too_tight"
                        score.record_evaluation(was_correct=False, move_pct=move_pct)
                    elif move_pct <= -self._move_threshold:
                        rec.outcome = "correct"
                        score.record_evaluation(was_correct=True, move_pct=move_pct)
                    else:
                        rec.outcome = "neutral"
                        score.neutral += 1

                self._history.append(rec)
                newly_evaluated.append(rec)
                self.total_evaluations += 1
            else:
                still_pending.append(rec)

        self._pending = still_pending

        # 维护历史记录最大长度
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # 有新评估结果时自动保存
        if newly_evaluated:
            self._dirty = True
            self.save()
            eval_summary = ", ".join(
                f"{r.reason}={r.outcome}({r.move_pct:+.2f}%)"
                for r in newly_evaluated
            )
            print(f"[TPSLTracker] 评估完成 {len(newly_evaluated)} 条: {eval_summary}")

        return newly_evaluated

    # ─────────────────────────────────────────────
    #  查询接口
    # ─────────────────────────────────────────────

    def get_recent_records(self, limit: int = 20) -> List[TPSLRecord]:
        """获取最近的 TP/SL 记录（含已评估和待评估），按时间倒序"""
        all_records = list(self._history) + self._pending
        all_records.sort(key=lambda r: r.timestamp, reverse=True)
        return all_records[:limit]

    def get_pending_count(self) -> int:
        """待评估记录数"""
        return len(self._pending)

    # ─────────────────────────────────────────────
    #  调整建议
    # ─────────────────────────────────────────────

    def get_suggestions(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于评分生成 ATR 倍数调整建议

        当某类平仓的准确率异常时，建议调整对应参数：
        - accuracy < 0.4 → TP/SL 过紧，建议放宽（增大 ATR 倍数）
        - accuracy > 0.8 → TP/SL 过宽或合适，可收紧（减小 ATR 倍数）

        Returns:
            建议列表，每条包含 param_key, old_value, new_value, reason 等
        """
        suggestions: List[Dict[str, Any]] = []

        for reason, score in self._scores.items():
            if score.total < self._min_evals:
                continue

            accuracy = score.accuracy
            recent = score.recent_trend
            label = TPSL_REASON_LABELS.get(reason, reason)

            if accuracy < 0.4:
                action = "loosen"
                reason_text = (f"{label} 设置过紧 (准确率 {accuracy:.0%}, "
                               f"近期 {recent:.0%}), "
                               f"累计错失 {score.total_missed_pct:.1f}%, "
                               f"建议放宽 ATR 倍数")
            elif accuracy > 0.8:
                action = "tighten"
                reason_text = (f"{label} 表现良好 (准确率 {accuracy:.0%}, "
                               f"近期 {recent:.0%}), "
                               f"可适当收紧 ATR 倍数")
            else:
                continue  # 表现合理，无需调整

            # 展开为逐参数的具体建议
            target_params = self.REASON_TO_PARAMS.get(reason, [])
            for param_key in target_params:
                param_def = self.DEFAULT_PARAMS.get(param_key, {})
                current = config.get(param_key)
                if current is None:
                    continue

                step_key = "loosen_step" if action == "loosen" else "tighten_step"
                step = param_def.get(step_key, 0)
                if step == 0:
                    continue

                new_value = current + step
                min_bound = param_def.get("min", float("-inf"))
                max_bound = param_def.get("max", float("inf"))
                new_value = max(min_bound, min(max_bound, new_value))

                if new_value == current:
                    continue  # 已在边界

                param_label = param_def.get("label", param_key)
                action_text = "放宽" if action == "loosen" else "收紧"
                suggestions.append({
                    "reason": reason,
                    "param_key": param_key,
                    "action": action,
                    "action_text": action_text,
                    "label": param_label,
                    "old_value": current,
                    "new_value": round(new_value, 4),
                    "step": step,
                    "min_bound": min_bound,
                    "max_bound": max_bound,
                    "accuracy": round(accuracy, 3),
                    "recent_trend": round(recent, 3),
                    "total_evaluations": score.total,
                    "reason_text": reason_text,
                })

        return suggestions

    # ─────────────────────────────────────────────
    #  调整应用
    # ─────────────────────────────────────────────

    def apply_adjustment(self, config: Dict[str, Any], param_key: str,
                         new_value: float, reason: str = "") -> Optional[Dict[str, Any]]:
        """
        应用单个参数调整到运行时配置

        Args:
            config: 运行时配置字典（原地修改）
            param_key: 参数名
            new_value: 目标新值
            reason: 调整原因

        Returns:
            调整记录 dict 或 None（未执行）
        """
        if param_key not in config:
            print(f"[TPSLTracker] 调整失败: 参数 {param_key} 不存在于配置中")
            return None

        old_value = config[param_key]

        # 安全夹紧
        param_def = self.DEFAULT_PARAMS.get(param_key, {})
        min_bound = param_def.get("min", float("-inf"))
        max_bound = param_def.get("max", float("inf"))
        clamped = max(min_bound, min(max_bound, new_value))

        if clamped == old_value:
            print(f"[TPSLTracker] 跳过调整: {param_key} 已在目标值 {old_value}")
            return None

        config[param_key] = clamped
        self._dirty = True
        self.save()

        action_text = "放宽" if clamped > old_value else "收紧"
        print(f"[TPSLTracker] 参数调整已应用: {param_key} "
              f"{old_value} → {clamped} ({action_text}) | {reason}")

        return {
            "param_key": param_key,
            "old_value": old_value,
            "new_value": clamped,
            "reason": reason,
        }

    # ─────────────────────────────────────────────
    #  UI 数据导出
    # ─────────────────────────────────────────────

    def get_state_for_ui(self) -> dict:
        """
        导出用于 UI 展示的状态摘要

        返回结构可直接序列化到 EngineState 的 tpsl_history / tpsl_scores 字段。
        """
        recent = self.get_recent_records(20)
        return {
            "records": [r.to_dict() for r in recent],
            "scores": {
                reason: {
                    "reason": s.reason,
                    "correct": s.correct,
                    "too_tight": s.too_tight,
                    "neutral": s.neutral,
                    "total": s.total,
                    "accuracy": round(s.accuracy, 3),
                    "recent_trend": round(s.recent_trend, 3),
                    "total_missed_pct": round(s.total_missed_pct, 2),
                    "total_saved_pct": round(s.total_saved_pct, 2),
                }
                for reason, s in self._scores.items()
            },
            "pending_count": self.get_pending_count(),
            "total_records": self.total_records,
            "total_evaluations": self.total_evaluations,
        }

    # ─────────────────────────────────────────────
    #  持久化
    # ─────────────────────────────────────────────

    def save(self):
        """持久化到 JSON（原子写入）"""
        if not self._persistence_path:
            return
        try:
            dirpath = os.path.dirname(self._persistence_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            now = time.time()
            self._last_save_time = now
            if self._created_at <= 0:
                self._created_at = now
            data = {
                "version": self.DATA_VERSION,
                "state": {
                    "created_at": self._created_at,
                    "last_save_time": self._last_save_time,
                    "total_records": self.total_records,
                    "total_evaluations": self.total_evaluations,
                },
                "history": [r.to_dict() for r in self._history],
                "pending": [r.to_dict() for r in self._pending],
                "scores": {k: v.to_dict() for k, v in self._scores.items()},
            }

            tmp_path = self._persistence_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._persistence_path)
            self._dirty = False
        except Exception as e:
            print(f"[TPSLTracker] 保存失败: {e}")
            tmp_path = self._persistence_path + ".tmp"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save_if_dirty(self, min_interval_sec: float = 60.0):
        """条件保存：有未保存变更且超过间隔时才保存"""
        if not self._dirty:
            return
        if time.time() - self._last_save_time < min_interval_sec:
            return
        self.save()

    def load(self):
        """从 JSON 加载持久化数据"""
        if not self._persistence_path or not os.path.exists(self._persistence_path):
            return
        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 加载状态
            state = data.get("state", {})
            self.total_records = state.get("total_records", 0)
            self.total_evaluations = state.get("total_evaluations", 0)
            self._last_save_time = state.get("last_save_time", 0.0)
            self._created_at = state.get("created_at", 0.0)
            if self._created_at <= 0 and self._last_save_time > 0:
                self._created_at = self._last_save_time

            # 加载历史
            self._history = [TPSLRecord.from_dict(d) for d in data.get("history", [])]
            self._pending = [TPSLRecord.from_dict(d) for d in data.get("pending", [])]

            # 加载评分
            self._scores = {}
            for k, v in data.get("scores", {}).items():
                self._scores[k] = TPSLScore.from_dict(v)

            self._dirty = False
            total = len(self._history) + len(self._pending)
            print(f"[TPSLTracker] 加载成功: "
                  f"{len(self._scores)} 个评分类别, "
                  f"{total} 条记录 ({len(self._pending)} 条待评估), "
                  f"累计 {self.total_records} 次记录")

        except json.JSONDecodeError as e:
            print(f"[TPSLTracker] JSON 解析失败: {e}，使用初始状态")
        except Exception as e:
            print(f"[TPSLTracker] 加载失败: {e}，使用初始状态")

    # ─────────────────────────────────────────────
    #  重置
    # ─────────────────────────────────────────────

    def reset(self):
        """完全重置所有数据"""
        self._history.clear()
        self._pending.clear()
        self._scores.clear()
        self.total_records = 0
        self.total_evaluations = 0
        self._dirty = False
        self.save()
        print("[TPSLTracker] 已重置全部数据")
