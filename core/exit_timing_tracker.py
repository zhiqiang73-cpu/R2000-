"""
出场时机追踪器 - 记录已平仓交易，事后追踪价格走势以评估出场决策质量

工作流：
1. 交易平仓 → record_exit() 记录出场时的完整上下文
2. 每根K线 → evaluate_pending() 追踪已记录出场的后续价格走势
3. 积累足够评估 → suggest_adjustments() 给出追踪止损/动量/脱轨阈值调整建议

评估维度：
- 出场过早（premature）: 出场后价格继续朝有利方向大幅移动 → 建议放宽追踪止损
- 出场正确（correct）: 出场后价格走平或反转 → 当前策略合理
- 出场过晚（late）: 出场时已从峰值大幅回撤 → 建议收紧追踪止损/动量衰减

与 RejectionTracker 互补：
- RejectionTracker: 评估 "没做的交易" → 门控阈值松紧
- ExitTimingTracker: 评估 "已做交易的出场点" → 出场策略参数
"""

import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ── 出场原因标签与颜色（UI 用）──

CLOSE_REASON_LABELS = {
    "止盈": "止盈 (TP)",
    "止损": "止损 (SL)",
    "追踪止损": "追踪止损",
    "分段止盈": "分段止盈",
    "分段止损": "分段止损",
    "脱轨": "脱轨离场",
    "超时": "超时离场",
    "信号": "信号离场",
    "位置翻转": "位置翻转",
    "手动": "手动平仓",
    "交易所平仓": "交易所平仓",
}

CLOSE_REASON_COLORS = {
    "止盈": "#4CAF50",        # green
    "止损": "#F44336",        # red
    "追踪止损": "#FF9800",    # orange
    "分段止盈": "#8BC34A",    # light green
    "分段止损": "#FF5722",    # deep orange
    "脱轨": "#9C27B0",        # purple
    "超时": "#607D8B",        # blue-grey
    "信号": "#2196F3",        # blue
    "位置翻转": "#FFC107",    # yellow
    "手动": "#795548",        # brown
    "交易所平仓": "#E91E63",  # pink
}

# ── 出场评价标签 ──

EXIT_VERDICT_LABELS = {
    "premature": "过早出场",
    "correct": "出场正确",
    "late": "出场过晚",
}

# ── 可调出场参数定义（阈值边界 & 调整步长）──
# 分段止盈/止损与市场状态自适应：强趋势时放宽阈值让利润跑，震荡时收紧阈值早锁利

ADJUSTABLE_EXIT_PARAMS: Dict[str, Dict[str, Any]] = {
    "STAGED_TP_1_PCT": {
        "label": "分段止盈第1档阈值(%)",
        "loosen_step": 0.5,     # 放宽 = 提高阈值（更晚触发第一档 → 让利润跑更久）
        "tighten_step": -0.5,   # 收紧 = 降低阈值（更早触发第一档 → 早锁利）
        "min": 3.0,
        "max": 12.0,
    },
    "STAGED_TP_2_PCT": {
        "label": "分段止盈第2档阈值(%)",
        "loosen_step": 0.5,
        "tighten_step": -0.5,
        "min": 6.0,
        "max": 20.0,
    },
    "STAGED_SL_1_PCT": {
        "label": "分段止损第1档阈值(%)",
        "loosen_step": -0.5,    # 放宽 = 降低绝对值（更晚触发第一档减仓）
        "tighten_step": 0.5,    # 收紧 = 提高绝对值（更早触发第一档减仓）
        "min": 2.0,
        "max": 10.0,
    },
    "STAGED_SL_2_PCT": {
        "label": "分段止损第2档阈值(%)",
        "loosen_step": -0.5,
        "tighten_step": 0.5,
        "min": 5.0,
        "max": 18.0,
    },
    "MOMENTUM_MIN_PROFIT_PCT": {
        "label": "动量离场最低利润阈值(%)",
        "loosen_step": 0.3,     # 放宽 = 提高阈值（更不容易触发动量离场）
        "tighten_step": -0.2,   # 收紧 = 降低阈值（更容易触发动量离场）
        "min": 0.5,
        "max": 3.5,
    },
    "MOMENTUM_DECAY_THRESHOLD": {
        "label": "动量衰减阈值",
        "loosen_step": 0.05,    # 放宽 = 提高阈值（更宽容的衰减判定）
        "tighten_step": -0.05,  # 收紧 = 降低阈值（更严格的衰减判定）
        "min": 0.2,
        "max": 0.8,
    },
    "HOLD_DERAIL_THRESHOLD": {
        "label": "脱轨离场阈值",
        "loosen_step": -0.05,   # 放宽 = 降低阈值（更不容易脱轨 → 持仓更久）
        "tighten_step": 0.05,   # 收紧 = 提高阈值（更容易脱轨 → 更快离场）
        "min": 0.15,
        "max": 0.50,
    },
}

# ── 出场原因到可调参数的映射 ──

CLOSE_REASON_PARAM_MAP: Dict[str, List[str]] = {
    "追踪止损": [
        "STAGED_TP_1_PCT", "STAGED_TP_2_PCT",
    ],
    "分段止盈": [
        "STAGED_TP_1_PCT", "STAGED_TP_2_PCT",
    ],
    "分段止损": [
        "STAGED_SL_1_PCT", "STAGED_SL_2_PCT",
    ],
    "信号": [
        "MOMENTUM_MIN_PROFIT_PCT", "MOMENTUM_DECAY_THRESHOLD",
    ],
    "脱轨": [
        "HOLD_DERAIL_THRESHOLD",
    ],
}


@dataclass
class ExitTimingRecord:
    """
    单次出场记录

    记录交易平仓时的完整上下文，以及事后价格追踪评估结果。
    用于判断 "这次出场是太早了（利润继续跑）还是恰好（价格反转/走平）还是太晚了（已回撤大半）"。
    """
    id: str                              # UUID
    timestamp: float                     # time.time()
    timestamp_str: str                   # 可读时间字符串
    direction: str                       # "LONG" / "SHORT"
    close_reason: str                    # CloseReason.value (如 "止盈", "追踪止损")
    entry_price: float                   # 入场价
    exit_price: float                    # 出场价
    profit_pct: float                    # 出场时收益率 (%, 含杠杆)
    peak_profit_pct: float               # 持仓峰值收益率 (%)
    hold_bars: int                       # 持仓K线数
    trailing_stage: int                  # 出场时追踪止损阶段
    market_regime: str                   # 市场状态
    bar_idx_at_exit: int                 # 出场时K线索引
    template_fingerprint: str = ""       # 原型/模板指纹

    # ── 事后评估字段 ──
    evaluated: bool = False
    price_after_eval: Optional[float] = None    # 追踪期结束时价格
    bars_tracked: int = 0                       # 已追踪K线数
    max_favorable_pct: float = 0.0              # 出场后最大有利方向移动 (%)
    max_adverse_pct: float = 0.0                # 出场后最大不利方向移动 (%)
    price_at_max_favorable: Optional[float] = None  # 最有利时的价格
    verdict: str = ""                           # "premature" / "correct" / "late"
    evaluation_time: Optional[float] = None     # 评估完成时间

    # 追踪期间数据 (每bar更新)
    _price_history: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "direction": self.direction,
            "close_reason": self.close_reason,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "profit_pct": self.profit_pct,
            "peak_profit_pct": self.peak_profit_pct,
            "hold_bars": self.hold_bars,
            "trailing_stage": self.trailing_stage,
            "market_regime": self.market_regime,
            "bar_idx_at_exit": self.bar_idx_at_exit,
            "template_fingerprint": self.template_fingerprint,
            "evaluated": self.evaluated,
            "price_after_eval": self.price_after_eval,
            "bars_tracked": self.bars_tracked,
            "max_favorable_pct": self.max_favorable_pct,
            "max_adverse_pct": self.max_adverse_pct,
            "price_at_max_favorable": self.price_at_max_favorable,
            "verdict": self.verdict,
            "evaluation_time": self.evaluation_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ExitTimingRecord':
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", 0.0),
            timestamp_str=d.get("timestamp_str", ""),
            direction=d.get("direction", "LONG"),
            close_reason=d.get("close_reason", ""),
            entry_price=d.get("entry_price", 0.0),
            exit_price=d.get("exit_price", 0.0),
            profit_pct=d.get("profit_pct", 0.0),
            peak_profit_pct=d.get("peak_profit_pct", 0.0),
            hold_bars=d.get("hold_bars", 0),
            trailing_stage=d.get("trailing_stage", 0),
            market_regime=d.get("market_regime", "未知"),
            bar_idx_at_exit=d.get("bar_idx_at_exit", 0),
            template_fingerprint=d.get("template_fingerprint", ""),
            evaluated=d.get("evaluated", False),
            price_after_eval=d.get("price_after_eval"),
            bars_tracked=d.get("bars_tracked", 0),
            max_favorable_pct=d.get("max_favorable_pct", 0.0),
            max_adverse_pct=d.get("max_adverse_pct", 0.0),
            price_at_max_favorable=d.get("price_at_max_favorable"),
            verdict=d.get("verdict", ""),
            evaluation_time=d.get("evaluation_time"),
        )


@dataclass
class ExitReasonScore:
    """
    按出场原因的累计评分

    追踪每种出场原因（追踪止损、脱轨、信号等）的出场质量：
    - premature_rate > 0.5 → 该类出场经常过早，应放宽参数
    - late_rate > 0.3 → 该类出场经常过晚，应收紧参数
    """
    close_reason: str
    premature_count: int = 0    # 过早出场次数
    correct_count: int = 0      # 出场正确次数
    late_count: int = 0         # 出场过晚次数
    total_left_on_table_pct: float = 0.0    # 累计因过早出场错过的利润 (%)
    total_excess_retracement_pct: float = 0.0  # 累计因过晚出场多承受的回撤 (%)
    _recent_verdicts: List[str] = field(default_factory=list)

    RECENT_WINDOW: int = 20

    @property
    def total_count(self) -> int:
        return self.premature_count + self.correct_count + self.late_count

    @property
    def correct_rate(self) -> float:
        """正确出场率"""
        if self.total_count == 0:
            return 0.5
        return self.correct_count / self.total_count

    @property
    def premature_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.premature_count / self.total_count

    @property
    def late_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.late_count / self.total_count

    @property
    def recent_trend(self) -> str:
        """最近评估的主要趋势"""
        if not self._recent_verdicts:
            return "insufficient"
        from collections import Counter
        counts = Counter(self._recent_verdicts)
        return counts.most_common(1)[0][0]

    @property
    def recent_correct_rate(self) -> float:
        if not self._recent_verdicts:
            return 0.5
        return sum(1 for v in self._recent_verdicts if v == "correct") / len(self._recent_verdicts)

    def record_evaluation(self, verdict: str, favorable_pct: float, adverse_pct: float):
        """记录一次评估结果"""
        if verdict == "premature":
            self.premature_count += 1
            self.total_left_on_table_pct += abs(favorable_pct)
        elif verdict == "late":
            self.late_count += 1
            self.total_excess_retracement_pct += abs(adverse_pct)
        else:
            self.correct_count += 1

        self._recent_verdicts.append(verdict)
        if len(self._recent_verdicts) > self.RECENT_WINDOW:
            self._recent_verdicts.pop(0)

    def to_dict(self) -> dict:
        return {
            "close_reason": self.close_reason,
            "premature_count": self.premature_count,
            "correct_count": self.correct_count,
            "late_count": self.late_count,
            "total": self.total_count,
            "correct": self.correct_count,
            "total_left_on_table_pct": self.total_left_on_table_pct,
            "total_excess_retracement_pct": self.total_excess_retracement_pct,
            "recent_verdicts": list(self._recent_verdicts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ExitReasonScore':
        s = cls(
            close_reason=d.get("close_reason", ""),
            premature_count=d.get("premature_count", 0),
            correct_count=d.get("correct_count", 0),
            late_count=d.get("late_count", 0),
            total_left_on_table_pct=d.get("total_left_on_table_pct", 0.0),
            total_excess_retracement_pct=d.get("total_excess_retracement_pct", 0.0),
        )
        s._recent_verdicts = [str(x) for x in d.get("recent_verdicts", [])]
        return s


class ExitTimingTracker:
    """
    出场时机追踪器

    追踪已平仓交易的出场后价格走势，判断出场时机是否合理，
    并据此建议追踪止损/动量衰减/脱轨阈值的参数调整。

    使用方式：
        tracker = ExitTimingTracker(persistence_path="data/exit_timing_state.json")
        # 在交易平仓时调用
        tracker.record_exit(order=closed_order, market_regime="强多头", bar_idx=1234)
        # 每根K线闭合时调用
        tracker.evaluate_pending(current_price=65000.0, current_bar_idx=1264)
    """

    DATA_VERSION = 1

    def __init__(self,
                 eval_bars: int = 30,
                 premature_threshold_pct: float = 0.5,
                 late_retracement_pct: float = 0.5,
                 max_history: int = 200,
                 min_evaluations_for_suggestion: int = 10,
                 persistence_path: str = "data/exit_timing_state.json"):
        """
        Args:
            eval_bars: 出场后追踪多少根K线评估后续走势
            premature_threshold_pct: 出场后价格继续朝有利方向移动超过此 % 视为 "过早出场"
            late_retracement_pct: 出场时已从峰值回撤超过此比例视为 "过晚出场"
                                  (比较 peak_profit - actual_profit vs peak_profit)
            max_history: 最多保留多少条已评估的历史记录
            min_evaluations_for_suggestion: 每种出场原因至少需要多少次评估才给出调整建议
            persistence_path: 持久化文件路径
        """
        self._eval_bars = eval_bars
        self._premature_threshold = premature_threshold_pct
        self._late_retracement_pct = late_retracement_pct
        self._max_history = max_history
        self._min_evaluations = min_evaluations_for_suggestion
        self.persistence_path = persistence_path

        # 已评估的历史记录（有限长度）
        self._history: deque = deque(maxlen=max_history)
        # 等待评估的记录（出场后还没过够 N 根K线）
        self._pending_eval: List[ExitTimingRecord] = []
        # 每个 close_reason 的累计评分
        self._reason_scores: Dict[str, ExitReasonScore] = {}

        # 统计
        self.total_exits_recorded: int = 0
        self.total_evaluations_done: int = 0
        self.total_adjustments_applied: int = 0

        # 持久化状态追踪
        self._dirty: bool = False
        self._last_save_time: float = 0.0
        self._created_at: float = 0.0  # 首次创建时间戳（记忆开始时间）

        # 会话标识
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        # 从文件加载
        self.load()

    # ─────────────────────────────────────────────
    #  记录出场
    # ─────────────────────────────────────────────

    def record_exit(self,
                    direction: str,
                    close_reason: str,
                    entry_price: float,
                    exit_price: float,
                    profit_pct: float,
                    peak_profit_pct: float,
                    hold_bars: int,
                    trailing_stage: int,
                    market_regime: str,
                    bar_idx: int,
                    template_fingerprint: str = "") -> ExitTimingRecord:
        """
        记录一次出场

        Args:
            direction: 持仓方向 "LONG" / "SHORT"
            close_reason: 平仓原因（CloseReason.value）
            entry_price: 入场价
            exit_price: 出场价
            profit_pct: 出场时收益率 (%, 含杠杆)
            peak_profit_pct: 持仓峰值收益率 (%)
            hold_bars: 持仓K线数
            trailing_stage: 出场时追踪止损阶段
            market_regime: 当前市场状态
            bar_idx: 当前K线索引
            template_fingerprint: 原型/模板指纹

        Returns:
            创建的 ExitTimingRecord
        """
        now = time.time()
        rec = ExitTimingRecord(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
            direction=direction,
            close_reason=close_reason,
            entry_price=entry_price,
            exit_price=exit_price,
            profit_pct=round(profit_pct, 4),
            peak_profit_pct=round(peak_profit_pct, 4),
            hold_bars=hold_bars,
            trailing_stage=trailing_stage,
            market_regime=market_regime,
            bar_idx_at_exit=bar_idx,
            template_fingerprint=template_fingerprint,
        )

        self._pending_eval.append(rec)
        self.total_exits_recorded += 1
        self._dirty = True

        # 确保 reason_scores 中有该 close_reason 的条目
        if close_reason not in self._reason_scores:
            self._reason_scores[close_reason] = ExitReasonScore(close_reason=close_reason)

        print(f"[ExitTimingTracker] 记录出场: {direction} | {close_reason} | "
              f"收益={profit_pct:+.2f}% | 峰值={peak_profit_pct:+.2f}% | "
              f"追踪阶段={trailing_stage}")

        return rec

    # ─────────────────────────────────────────────
    #  事后评估
    # ─────────────────────────────────────────────

    def evaluate_pending(self, current_price: float, current_bar_idx: int) -> List[ExitTimingRecord]:
        """
        评估待处理的出场记录（在每根K线闭合时调用）

        对于每条待评估记录：
        1. 更新追踪期间的价格历史和最大有利/不利移动
        2. 若已追踪满 eval_bars 根K线，执行最终判定

        判定逻辑（基于出场后价格走势 + 出场时回撤情况）：
        - premature (过早): 出场后价格朝有利方向继续移动 > premature_threshold
        - late (过晚): 出场时 peak_profit - actual_profit > peak * late_retracement_pct
                       且出场后价格未再创新高
        - correct (正确): 其他情况

        Args:
            current_price: 当前价格
            current_bar_idx: 当前K线索引

        Returns:
            本轮新完成评估的记录列表
        """
        newly_evaluated: List[ExitTimingRecord] = []
        still_pending: List[ExitTimingRecord] = []

        for rec in self._pending_eval:
            bars_passed = current_bar_idx - rec.bar_idx_at_exit

            if bars_passed <= 0:
                still_pending.append(rec)
                continue

            # 更新中间追踪数据
            rec._price_history.append(current_price)
            rec.bars_tracked = bars_passed

            # 计算出场后有利/不利方向的价格移动
            if rec.direction == "LONG":
                favorable_move_pct = ((current_price - rec.exit_price)
                                      / rec.exit_price * 100)
            else:  # SHORT
                favorable_move_pct = ((rec.exit_price - current_price)
                                      / rec.exit_price * 100)

            # 更新最大有利/不利移动
            if favorable_move_pct > rec.max_favorable_pct:
                rec.max_favorable_pct = round(favorable_move_pct, 4)
                rec.price_at_max_favorable = current_price
            if favorable_move_pct < 0 and abs(favorable_move_pct) > rec.max_adverse_pct:
                rec.max_adverse_pct = round(abs(favorable_move_pct), 4)

            # 追踪满 N 根K线，执行最终判定
            if bars_passed >= self._eval_bars:
                rec.evaluated = True
                rec.price_after_eval = current_price
                rec.evaluation_time = time.time()

                # 判定出场质量
                rec.verdict = self._judge_exit(rec)

                # 更新出场原因评分
                score = self._reason_scores.get(rec.close_reason)
                if score is None:
                    score = ExitReasonScore(close_reason=rec.close_reason)
                    self._reason_scores[rec.close_reason] = score
                score.record_evaluation(
                    rec.verdict, rec.max_favorable_pct, rec.max_adverse_pct
                )

                # 移入历史
                self._history.append(rec)
                newly_evaluated.append(rec)
                self.total_evaluations_done += 1
            else:
                still_pending.append(rec)

        self._pending_eval = still_pending

        if newly_evaluated:
            self.save()
            eval_summary = ", ".join(
                f"{r.close_reason}={EXIT_VERDICT_LABELS.get(r.verdict, r.verdict)}"
                f"(有利{r.max_favorable_pct:+.2f}%)"
                for r in newly_evaluated
            )
            print(f"[ExitTimingTracker] 评估完成 {len(newly_evaluated)} 条: {eval_summary}")

        return newly_evaluated

    def _judge_exit(self, rec: ExitTimingRecord) -> str:
        """
        判定出场质量

        三维度判定：
        1. 出场后价格是否继续朝有利方向大幅移动 → premature
        2. 出场时是否已从峰值大幅回撤（同时出场后未再创新高）→ late
        3. 其他 → correct
        """
        # 维度1: 过早出场 —— 出场后利润继续跑
        if rec.max_favorable_pct > self._premature_threshold:
            return "premature"

        # 维度2: 过晚出场 —— 出场时已经从峰值大幅回撤
        if rec.peak_profit_pct > 0:
            retracement_ratio = (rec.peak_profit_pct - rec.profit_pct) / rec.peak_profit_pct
            if retracement_ratio > self._late_retracement_pct:
                # 还需确认出场后价格未再创新高（否则只是临时回调）
                if rec.max_favorable_pct < self._premature_threshold * 0.5:
                    return "late"

        return "correct"

    # ─────────────────────────────────────────────
    #  查询接口
    # ─────────────────────────────────────────────

    def get_reason_score(self, close_reason: str) -> Optional[ExitReasonScore]:
        """获取指定出场原因的累计评分"""
        return self._reason_scores.get(close_reason)

    def get_all_reason_scores(self) -> Dict[str, ExitReasonScore]:
        """获取所有出场原因的评分"""
        return dict(self._reason_scores)

    def get_recent_records(self, limit: int = 20) -> List[ExitTimingRecord]:
        """获取最近的出场记录（含已评估和待评估），按时间倒序"""
        all_records: List[ExitTimingRecord] = list(self._history) + self._pending_eval
        all_records.sort(key=lambda r: r.timestamp, reverse=True)
        return all_records[:limit]

    def get_pending_count(self) -> int:
        return len(self._pending_eval)

    # ─────────────────────────────────────────────
    #  调整建议
    # ─────────────────────────────────────────────

    def suggest_adjustment(self, close_reason: str) -> Optional[Dict[str, Any]]:
        """
        基于出场原因评分，建议参数调整方向

        仅在积累足够评估（>= min_evaluations）时才给出建议。
        premature_rate > 0.5 → 建议放宽（loosen）出场参数（让利润跑更久）
        late_rate > 0.3 → 建议收紧（tighten）出场参数（更快锁利）
        其他 → 无需调整

        Args:
            close_reason: 出场原因（如 "追踪止损"）

        Returns:
            调整建议字典，或 None
        """
        score = self._reason_scores.get(close_reason)
        if not score or score.total_count < self._min_evaluations:
            return None

        param_keys = CLOSE_REASON_PARAM_MAP.get(close_reason, [])
        if not param_keys:
            return None

        if score.premature_rate > 0.5:
            return {
                "action": "loosen",
                "close_reason": close_reason,
                "correct_rate": round(score.correct_rate, 3),
                "premature_rate": round(score.premature_rate, 3),
                "late_rate": round(score.late_rate, 3),
                "total_evaluations": score.total_count,
                "total_left_on_table_pct": round(score.total_left_on_table_pct, 2),
                "param_keys": param_keys,
                "reason": (f"出场原因 [{CLOSE_REASON_LABELS.get(close_reason, close_reason)}] "
                           f"过早出场率 {score.premature_rate:.0%}（正确率 {score.correct_rate:.0%}），"
                           f"建议放宽出场参数以捕获更多利润"),
            }
        elif score.late_rate > 0.3:
            return {
                "action": "tighten",
                "close_reason": close_reason,
                "correct_rate": round(score.correct_rate, 3),
                "premature_rate": round(score.premature_rate, 3),
                "late_rate": round(score.late_rate, 3),
                "total_evaluations": score.total_count,
                "total_excess_retracement_pct": round(score.total_excess_retracement_pct, 2),
                "param_keys": param_keys,
                "reason": (f"出场原因 [{CLOSE_REASON_LABELS.get(close_reason, close_reason)}] "
                           f"过晚出场率 {score.late_rate:.0%}（正确率 {score.correct_rate:.0%}），"
                           f"建议收紧出场参数以减少利润回吐"),
            }

        return None

    def suggest_all_adjustments(self) -> List[Dict[str, Any]]:
        """获取所有出场原因的调整建议"""
        suggestions = []
        for close_reason in self._reason_scores:
            sug = self.suggest_adjustment(close_reason)
            if sug is not None:
                suggestions.append(sug)
        return suggestions

    def get_suggestions(self, config_dict: Dict[str, Any], current_regime: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有具体的参数调整建议（供 UI 展示）。

        将 suggest_all_adjustments() 的每条建议展开为逐参数的具体调整条目。
        若传入 current_regime，则根据市场强弱做自适应步长：
        - 强趋势/趋势市：建议放宽时步长放大 1.2 倍（让利润多跑）
        - 震荡/弱市：建议收紧时步长放大 1.2 倍（早锁利）

        Args:
            config_dict: 运行时配置字典（通常为 PAPER_TRADING_CONFIG）
            current_regime: 当前市场状态（如 "强多头"/"震荡"/"弱空头"），用于自适应步长

        Returns:
            逐参数的调整条目列表
        """
        raw = self.suggest_all_adjustments()
        expanded: List[Dict[str, Any]] = []

        # 市场状态对步长的自适应系数
        regime = (current_regime or "").strip()
        strong_trend = any(x in regime for x in ("强", "趋势", "多头", "空头")) and "震荡" not in regime
        choppy = "震荡" in regime or "弱" in regime

        for sug in raw:
            action = sug.get("action", "")
            param_keys = sug.get("param_keys", [])

            for param_key in param_keys:
                current_val = config_dict.get(param_key)
                if current_val is None:
                    continue

                param_def = ADJUSTABLE_EXIT_PARAMS.get(param_key)
                if not param_def:
                    continue

                step_key = "loosen_step" if action == "loosen" else "tighten_step"
                step = param_def.get(step_key, 0)
                if step == 0:
                    continue

                # 根据市场状态放大步长（自适应）
                if strong_trend and action == "loosen":
                    step = step * 1.2
                elif choppy and action == "tighten":
                    step = step * 1.2

                min_bound = param_def.get("min", float("-inf"))
                max_bound = param_def.get("max", float("inf"))

                new_value = current_val + step
                new_value = max(min_bound, min(max_bound, new_value))

                if new_value == current_val:
                    continue

                action_text = "放宽" if action == "loosen" else "收紧"
                expanded.append({
                    "param_key": param_key,
                    "action": action,
                    "action_text": action_text,
                    "label": param_def.get("label", param_key),
                    "old_value": current_val,
                    "new_value": round(new_value, 4),
                    "step": step,
                    "min_bound": min_bound,
                    "max_bound": max_bound,
                    "close_reason": sug.get("close_reason", ""),
                    "correct_rate": sug.get("correct_rate", 0),
                    "premature_rate": sug.get("premature_rate", 0),
                    "late_rate": sug.get("late_rate", 0),
                    "reason": sug.get("reason", ""),
                    "total_evaluations": sug.get("total_evaluations", 0),
                })

        return expanded

    def apply_adjustment(self,
                         config_dict: Dict[str, Any],
                         param_key: str,
                         new_value: float,
                         reason: str = "") -> Optional[Dict[str, Any]]:
        """
        应用单个参数调整到运行时配置

        Args:
            config_dict: 运行时配置字典，原地修改
            param_key: 要调整的参数名
            new_value: 目标新值
            reason: 调整原因

        Returns:
            调整结果 dict，或 None（未执行）
        """
        old_value = config_dict.get(param_key)
        if old_value is None:
            print(f"[ExitTimingTracker] 调整失败: 参数 {param_key} 不存在于配置中")
            return None

        # 安全夹紧
        param_def = ADJUSTABLE_EXIT_PARAMS.get(param_key, {})
        min_bound = param_def.get("min", float("-inf"))
        max_bound = param_def.get("max", float("inf"))
        clamped = max(min_bound, min(max_bound, new_value))

        if clamped == old_value:
            print(f"[ExitTimingTracker] 跳过调整: {param_key} 已在目标值 {old_value}")
            return None

        config_dict[param_key] = clamped
        self.total_adjustments_applied += 1
        self._dirty = True
        self.save()

        action_text = "放宽" if clamped > old_value else "收紧"
        print(f"[ExitTimingTracker] 参数调整已应用: {param_key} "
              f"{old_value} -> {clamped} ({action_text})")

        return {
            "param_key": param_key,
            "old_value": old_value,
            "new_value": clamped,
            "action_text": action_text,
        }

    # ─────────────────────────────────────────────
    #  UI 数据导出
    # ─────────────────────────────────────────────

    def get_state_for_ui(self) -> Dict[str, Any]:
        """导出用于 UI 展示的状态摘要"""
        records = self.get_recent_records(20)
        ui_records = []
        for r in records:
            ui_records.append({
                "timestamp_str": r.timestamp_str,
                "direction": r.direction,
                "close_reason": r.close_reason,
                "tag": r.close_reason,
                "reason": r.close_reason,
                "profit_pct": r.profit_pct,
                "peak_profit_pct": r.peak_profit_pct,
                "move_pct": r.max_favorable_pct if r.evaluated else None,
                "outcome": EXIT_VERDICT_LABELS.get(r.verdict, "追踪中") if r.evaluated else "追踪中",
                "result": EXIT_VERDICT_LABELS.get(r.verdict, "追踪中") if r.evaluated else "追踪中",
                "evaluated": r.evaluated,
                "verdict": r.verdict,
                "trailing_stage": r.trailing_stage,
                "hold_bars": r.hold_bars,
                "bars_tracked": r.bars_tracked,
            })

        scores_dict: Dict[str, Dict[str, Any]] = {}
        for reason, score in self._reason_scores.items():
            scores_dict[reason] = {
                "close_reason": score.close_reason,
                "premature_count": score.premature_count,
                "correct_count": score.correct_count,
                "correct": score.correct_count,
                "late_count": score.late_count,
                "total": score.total_count,
                "correct_rate": round(score.correct_rate, 3),
                "premature_rate": round(score.premature_rate, 3),
                "late_rate": round(score.late_rate, 3),
                "total_left_on_table_pct": round(score.total_left_on_table_pct, 2),
                "total_excess_retracement_pct": round(score.total_excess_retracement_pct, 2),
            }

        return {
            "exit_timing_history": ui_records,
            "exit_timing_scores": scores_dict,
            "pending_count": self.get_pending_count(),
            "total_exits": self.total_exits_recorded,
            "total_evaluations": self.total_evaluations_done,
            "total_adjustments": self.total_adjustments_applied,
        }

    # ─────────────────────────────────────────────
    #  会话报告
    # ─────────────────────────────────────────────

    def generate_session_report(self,
                                config_dict: Optional[Dict[str, Any]] = None
                                ) -> Dict[str, Any]:
        """生成会话结束报告"""
        reason_summaries: Dict[str, Dict[str, Any]] = {}
        for reason, score in self._reason_scores.items():
            reason_summaries[reason] = {
                "close_reason": reason,
                "label": CLOSE_REASON_LABELS.get(reason, reason),
                "premature_count": score.premature_count,
                "correct_count": score.correct_count,
                "late_count": score.late_count,
                "total_count": score.total_count,
                "correct_rate": round(score.correct_rate, 3),
                "premature_rate": round(score.premature_rate, 3),
                "late_rate": round(score.late_rate, 3),
            }

        pending_suggestions = []
        if config_dict is not None:
            pending_suggestions = self.get_suggestions(config_dict)

        return {
            "session_id": self._session_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": {
                "total_exits": self.total_exits_recorded,
                "total_evaluations": self.total_evaluations_done,
                "total_adjustments_applied": self.total_adjustments_applied,
                "pending_evaluations": len(self._pending_eval),
                "history_size": len(self._history),
            },
            "reason_summaries": reason_summaries,
            "pending_suggestions": pending_suggestions,
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
                    "premature_threshold_pct": self._premature_threshold,
                    "late_retracement_pct": self._late_retracement_pct,
                    "max_history": self._max_history,
                    "min_evaluations_for_suggestion": self._min_evaluations,
                },
                "state": {
                    "data_version": self.DATA_VERSION,
                    "created_at": self._created_at,
                    "last_save_time": self._last_save_time,
                    "total_exits_recorded": self.total_exits_recorded,
                    "total_evaluations_done": self.total_evaluations_done,
                    "total_adjustments_applied": self.total_adjustments_applied,
                },
                "reason_scores": {
                    reason: score.to_dict()
                    for reason, score in self._reason_scores.items()
                },
                "history": [rec.to_dict() for rec in self._history],
                "pending_eval": [rec.to_dict() for rec in self._pending_eval],
            }

            tmp_path = self.persistence_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.persistence_path)
            self._dirty = False
        except Exception as e:
            print(f"[ExitTimingTracker] 保存失败: {e}")
            tmp_path = self.persistence_path + ".tmp"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def save_if_dirty(self, min_interval_sec: float = 60.0):
        """条件保存"""
        if not self._dirty:
            return
        if time.time() - self._last_save_time < min_interval_sec:
            return
        self.save()

    def load(self):
        """从 JSON 加载持久化数据"""
        if not os.path.exists(self.persistence_path):
            print(f"[ExitTimingTracker] 持久化文件不存在，使用初始状态")
            return

        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = data.get("state", {})
            self.total_exits_recorded = state.get("total_exits_recorded", 0)
            self.total_evaluations_done = state.get("total_evaluations_done", 0)
            self.total_adjustments_applied = state.get("total_adjustments_applied", 0)
            self._last_save_time = state.get("last_save_time", 0.0)
            self._created_at = state.get("created_at", 0.0)
            if self._created_at <= 0 and self._last_save_time > 0:
                self._created_at = self._last_save_time

            # 加载出场原因评分
            scores_data = data.get("reason_scores", {})
            self._reason_scores = {
                reason: ExitReasonScore.from_dict(sd)
                for reason, sd in scores_data.items()
            }

            # 加载历史记录
            history_data = data.get("history", [])
            self._history = deque(
                (ExitTimingRecord.from_dict(d) for d in history_data),
                maxlen=self._max_history,
            )

            # 加载待评估记录
            pending_data = data.get("pending_eval", [])
            self._pending_eval = [ExitTimingRecord.from_dict(d) for d in pending_data]

            self._dirty = False
            total_records = len(self._history) + len(self._pending_eval)
            last_save_str = (datetime.fromtimestamp(self._last_save_time).strftime("%m-%d %H:%M")
                             if self._last_save_time > 0 else "未知")
            print(f"[ExitTimingTracker] 加载成功: "
                  f"{len(self._reason_scores)} 种出场评分, "
                  f"{total_records} 条记录 ({len(self._pending_eval)} 条待评估), "
                  f"上次保存: {last_save_str}")

        except json.JSONDecodeError as e:
            print(f"[ExitTimingTracker] JSON 解析失败: {e}，使用初始状态")
        except Exception as e:
            print(f"[ExitTimingTracker] 加载失败: {e}，使用初始状态")

    # ─────────────────────────────────────────────
    #  重置
    # ─────────────────────────────────────────────

    def reset(self):
        """完全重置所有数据"""
        self._history.clear()
        self._pending_eval.clear()
        self._reason_scores.clear()
        self.total_exits_recorded = 0
        self.total_evaluations_done = 0
        self.total_adjustments_applied = 0
        self._dirty = False
        self.save()
        print("[ExitTimingTracker] 已重置全部数据")
