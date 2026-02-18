"""
core/signal_live_monitor.py
实时信号组合触发监控

在 LiveTradingEngine 接收到每根新K线时调用 on_bar()，
检测已知的高质量信号组合是否在最新K线触发，
返回触发的 combo_key 列表，用于在开仓时打标到 PaperOrder。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# 最少出现轮次阈值（低于此的组合视为噪声，不追踪）
_MIN_APPEAR_ROUNDS = 2


class SignalLiveMonitor:
    """
    实时检测已知信号组合是否在最新K线触发。

    设计原则：
      - 只追踪 appear_rounds >= min_appear_rounds 的组合（过滤噪声）
      - 对每根 K 线只计算一次（缓存最新 bar_idx 的结果）
      - 不修改 signal_store，只读取组合定义
    """

    def __init__(self, min_appear_rounds: int = _MIN_APPEAR_ROUNDS):
        self._min_appear_rounds = min_appear_rounds
        self._last_bar_idx: int = -1
        self._last_combos: List[str] = []
        # 条件数组缓存（按方向，避免重复构建）
        self._cond_cache: Dict[str, dict] = {}   # direction -> {cond_name: np.ndarray}
        self._cond_cache_bar: int = -1

    # ─────────────────────────────────────────────────────────────────────────

    def on_bar(self, df: pd.DataFrame, latest_bar_idx: int) -> List[str]:
        """
        检测所有已知累计组合是否在 latest_bar_idx 对应的K线触发。

        Args:
            df:             已计算全部技术指标的 DataFrame
            latest_bar_idx: 最新完成K线的行索引（0-based）

        Returns:
            触发的 combo_key 列表（可能为空）。
        """
        # 同一根K线只算一次
        if latest_bar_idx == self._last_bar_idx:
            return list(self._last_combos)

        try:
            from core import signal_store
            from core.signal_analyzer import _build_condition_arrays
        except ImportError:
            return []

        cumulative = signal_store.get_cumulative()
        if not cumulative:
            return []

        n = len(df)
        if latest_bar_idx < 0 or latest_bar_idx >= n:
            return []

        # 构建条件数组（按方向缓存，同一根K线两个方向各只建一次）
        if self._cond_cache_bar != latest_bar_idx:
            self._cond_cache = {}
            self._cond_cache_bar = latest_bar_idx

        triggered: List[str] = []

        for combo_key, entry in cumulative.items():
            if entry.get('appear_rounds', 0) < self._min_appear_rounds:
                continue

            direction  = entry.get('direction', 'long')
            conditions = entry.get('conditions', [])
            if not conditions:
                continue

            # 懒加载并缓存条件数组
            if direction not in self._cond_cache:
                try:
                    self._cond_cache[direction] = _build_condition_arrays(df, direction)
                except Exception:
                    self._cond_cache[direction] = {}

            cond_arrays = self._cond_cache[direction]

            # 检查所有条件在 latest_bar_idx 是否同时满足
            all_true = True
            for cond in conditions:
                arr = cond_arrays.get(cond)
                if arr is None or not bool(arr[latest_bar_idx]):
                    all_true = False
                    break

            if all_true:
                triggered.append(combo_key)

        self._last_bar_idx = latest_bar_idx
        self._last_combos  = triggered
        return list(triggered)

    def reset_cache(self) -> None:
        """清除条件数组缓存（换新 DataFrame 时调用）。"""
        self._cond_cache      = {}
        self._cond_cache_bar  = -1
        self._last_bar_idx    = -1
        self._last_combos     = []
