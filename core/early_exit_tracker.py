import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class EarlyExitRecord:
    id: str
    timestamp: float
    timestamp_str: str
    exit_price: float
    direction: str
    bar_idx_at_exit: int
    detail: Dict[str, Any] = field(default_factory=dict)

    evaluated: bool = False
    price_after_eval: Optional[float] = None
    bars_waited: int = 0
    move_pct: Optional[float] = None
    outcome: Optional[str] = None  # "too_aggressive" / "correct" / "neutral"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "exit_price": self.exit_price,
            "direction": self.direction,
            "bar_idx_at_exit": self.bar_idx_at_exit,
            "detail": self.detail,
            "evaluated": self.evaluated,
            "price_after_eval": self.price_after_eval,
            "bars_waited": self.bars_waited,
            "move_pct": self.move_pct,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EarlyExitRecord":
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", 0.0),
            timestamp_str=d.get("timestamp_str", ""),
            exit_price=d.get("exit_price", 0.0),
            direction=d.get("direction", "LONG"),
            bar_idx_at_exit=d.get("bar_idx_at_exit", 0),
            detail=d.get("detail", {}),
            evaluated=d.get("evaluated", False),
            price_after_eval=d.get("price_after_eval"),
            bars_waited=d.get("bars_waited", 0),
            move_pct=d.get("move_pct"),
            outcome=d.get("outcome"),
        )


@dataclass
class EarlyExitScore:
    correct: int = 0
    too_aggressive: int = 0
    neutral: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.too_aggressive + self.neutral

    @property
    def accuracy(self) -> float:
        denom = self.correct + self.too_aggressive
        return (self.correct / denom) if denom > 0 else 0.5

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "too_aggressive": self.too_aggressive,
            "neutral": self.neutral,
            "total": self.total,
            "accuracy": self.accuracy,
        }


class EarlyExitTracker:
    DEFAULT_PARAMS = {
        "EARLY_EXIT_ADVERSE_PCT": {"loosen_step": 0.2, "tighten_step": -0.2, "min": 0.5, "max": 3.0},
        "SL_PROTECTION_SEC": {"loosen_step": 10, "tighten_step": -10, "min": 15, "max": 120},
    }

    def __init__(self,
                 eval_bars: int = 30,
                 move_threshold_pct: float = 0.5,
                 min_evals_for_suggest: int = 15,
                 max_history: int = 200,
                 persistence_path: Optional[str] = None):
        self._eval_bars = int(eval_bars)
        self._move_threshold = float(move_threshold_pct)
        self._min_evals = int(min_evals_for_suggest)
        self._history: List[EarlyExitRecord] = []
        self._pending: List[EarlyExitRecord] = []
        self._score = EarlyExitScore()
        self._max_history = max_history
        self._persistence_path = persistence_path
        self._dirty = False
        if self._persistence_path:
            self.load()

    def record_early_exit(self, direction: str, exit_price: float, bar_idx: int,
                          detail: Optional[Dict[str, Any]] = None) -> EarlyExitRecord:
        now = time.time()
        rec = EarlyExitRecord(
            id=str(uuid.uuid4()),
            timestamp=now,
            timestamp_str=time.strftime("%m-%d %H:%M:%S", time.localtime(now)),
            exit_price=float(exit_price),
            direction=direction,
            bar_idx_at_exit=int(bar_idx),
            detail=detail or {},
        )
        self._pending.append(rec)
        self._dirty = True
        return rec

    def evaluate_pending(self, current_price: float, bar_idx: int) -> List[EarlyExitRecord]:
        newly = []
        still = []
        for rec in self._pending:
            bars_passed = bar_idx - rec.bar_idx_at_exit
            if bars_passed >= self._eval_bars:
                if rec.direction == "LONG":
                    move_pct = (current_price - rec.exit_price) / rec.exit_price * 100
                else:
                    move_pct = (rec.exit_price - current_price) / rec.exit_price * 100
                rec.move_pct = round(move_pct, 4)
                rec.price_after_eval = current_price
                rec.bars_waited = bars_passed
                rec.evaluated = True
                if move_pct >= self._move_threshold:
                    rec.outcome = "too_aggressive"
                    self._score.too_aggressive += 1
                elif move_pct <= -self._move_threshold:
                    rec.outcome = "correct"
                    self._score.correct += 1
                else:
                    rec.outcome = "neutral"
                    self._score.neutral += 1
                self._history.append(rec)
                newly.append(rec)
            else:
                still.append(rec)
        self._pending = still
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        if newly:
            self._dirty = True
            self.save()
        return newly

    def get_state_for_ui(self) -> dict:
        return {
            "records": [r.to_dict() for r in reversed(self._history)],
            "scores": {"EARLY_EXIT": self._score.to_dict()},
        }

    def get_suggestions(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._score.total < self._min_evals:
            return []
        accuracy = self._score.accuracy
        if accuracy < 0.4:
            action = "loosen"
            reason = "早期出场过于激进，可放宽阈值"
        elif accuracy > 0.8:
            action = "tighten"
            reason = "早期出场很准，可适度收紧"
        else:
            return []
        suggestions = []
        for param_key, param_def in self.DEFAULT_PARAMS.items():
            current = config.get(param_key)
            if current is None:
                continue
            step = param_def.get("loosen_step" if action == "loosen" else "tighten_step", 0)
            new_value = current + step
            new_value = max(param_def.get("min", new_value), min(param_def.get("max", new_value), new_value))
            if new_value == current:
                continue
            suggestions.append({
                "param_key": param_key,
                "action": action,
                "old_value": current,
                "new_value": new_value,
                "reason": reason,
                "accuracy": accuracy,
            })
        return suggestions

    def apply_adjustment(self, config: Dict[str, Any], param_key: str, new_value: float,
                         reason: str = "") -> Optional[Dict[str, Any]]:
        if param_key not in config:
            return None
        old_value = config.get(param_key)
        if old_value == new_value:
            return None
        config[param_key] = new_value
        self._dirty = True
        self.save()
        return {
            "param_key": param_key,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }

    def save(self):
        if not self._persistence_path or not self._dirty:
            return
        data = {
            "history": [r.to_dict() for r in self._history],
            "pending": [r.to_dict() for r in self._pending],
            "score": self._score.to_dict(),
        }
        os.makedirs(os.path.dirname(self._persistence_path), exist_ok=True)
        with open(self._persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def load(self):
        if not self._persistence_path or not os.path.exists(self._persistence_path):
            return
        try:
            with open(self._persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._history = [EarlyExitRecord.from_dict(d) for d in data.get("history", [])]
            self._pending = [EarlyExitRecord.from_dict(d) for d in data.get("pending", [])]
            score_data = data.get("score", {})
            self._score.correct = score_data.get("correct", 0)
            self._score.too_aggressive = score_data.get("too_aggressive", 0)
            self._score.neutral = score_data.get("neutral", 0)
        except Exception:
            self._history = []
            self._pending = []
            self._score = EarlyExitScore()
