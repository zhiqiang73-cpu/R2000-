"""
core/signal_live_monitor.py
实时信号组合触发监控

在 LiveTradingEngine 接收到每根新K线时调用 on_bar()，
检测已知的高质量信号组合是否在最新K线触发，
返回触发的 combo_key 列表，用于在开仓时打标到 PaperOrder。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        self._last_pool_keys: Optional[frozenset] = None
        # 条件数组缓存（按方向，避免重复构建）
        self._cond_cache: Dict[str, dict] = {}   # direction -> {cond_name: np.ndarray}
        self._cond_cache_bar: int = -1

    # ─────────────────────────────────────────────────────────────────────────

    def on_bar(
        self,
        df: pd.DataFrame,
        latest_bar_idx: int,
        pool_keys: Optional[List[str]] = None,
    ) -> List[str]:
        """
        检测指定组合（或全部累计组合）是否在 latest_bar_idx 对应的K线触发。

        Args:
            df:             已计算全部技术指标的 DataFrame
            latest_bar_idx: 最新完成K线的行索引（0-based）
            pool_keys:      仅检测的 combo_key 列表（None 表示全部）

        Returns:
            触发的 combo_key 列表（可能为空）。
        """
        pool_key_set = frozenset(pool_keys) if pool_keys else None

        # 同一根K线+同一池只算一次
        if latest_bar_idx == self._last_bar_idx and pool_key_set == self._last_pool_keys:
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
            if pool_key_set is not None and combo_key not in pool_key_set:
                continue
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

        self._last_bar_idx   = latest_bar_idx
        self._last_pool_keys = pool_key_set
        self._last_combos    = triggered
        return list(triggered)

    def reset_cache(self) -> None:
        """清除条件数组缓存（换新 DataFrame 时调用）。"""
        self._cond_cache      = {}
        self._cond_cache_bar  = -1
        self._last_bar_idx    = -1
        self._last_combos     = []
        self._last_pool_keys  = None

    def get_best_for_state(
        self,
        triggered_keys: List[str],
        cumulative: Dict[str, dict],
        market_state: str,
        tier_map: Optional[Dict[str, str]] = None,
    ) -> Optional[Tuple[str, dict, str]]:
        """
        在已触发的组合中，根据当前市场状态选择最适合的一个。

        精品优先策略：
        1. 过滤在该市场状态下有触发记录的组合 (total_triggers > 0)
        2. 按 tier 分组（精品 / 高频），两层兜底
        3. 精品层有触发 → 只从精品层选；精品层无触发 → 从高频层选
        4. 同层内按方向偏向选最高评分：
           - 多头趋势: 优先 long，次选 short
           - 空头趋势: 优先 short，次选 long
           - 震荡市:   不分方向，取最高分

        Args:
            triggered_keys: 当前K线已触发的 combo_key 列表
            cumulative:     signal_store 的累计数据 dict
            market_state:   当前市场状态
            tier_map:       combo_key -> tier ("精品"/"高频")，None 时视全部为精品

        Returns:
            (combo_key, entry_dict, tier_str) 或 None
        """
        if not triggered_keys or not cumulative:
            return None

        _tier_map = tier_map or {}

        # 1. 过滤：该市场状态下有历史触发记录的组合
        valid_triggered = []
        for key in triggered_keys:
            entry = cumulative.get(key)
            if not entry:
                continue
            breakdown = entry.get('market_state_breakdown', {})
            state_info = breakdown.get(market_state, {})
            if state_info.get('total_triggers', 0) > 0:
                tier = _tier_map.get(key, "精品")
                valid_triggered.append((key, entry, tier))

        if not valid_triggered:
            return None

        def _pick_by_direction(pool: List[Tuple[str, dict, str]]) -> Optional[Tuple[str, dict, str]]:
            """按市场状态方向偏向从给定列表中选最高评分"""
            if not pool:
                return None
            if market_state == "多头趋势":
                longs = [x for x in pool if x[1].get('direction') == 'long']
                if longs:
                    return max(longs, key=lambda x: x[1].get('综合评分', 0.0))
                shorts = [x for x in pool if x[1].get('direction') == 'short']
                return max(shorts, key=lambda x: x[1].get('综合评分', 0.0)) if shorts else None
            elif market_state == "空头趋势":
                shorts = [x for x in pool if x[1].get('direction') == 'short']
                if shorts:
                    return max(shorts, key=lambda x: x[1].get('综合评分', 0.0))
                longs = [x for x in pool if x[1].get('direction') == 'long']
                return max(longs, key=lambda x: x[1].get('综合评分', 0.0)) if longs else None
            else:  # 震荡市
                return max(pool, key=lambda x: x[1].get('综合评分', 0.0))

        # 精品 → 高频，两层兜底
        elite_pool = [x for x in valid_triggered if x[2] == "精品"]
        result = _pick_by_direction(elite_pool)
        if result:
            return result

        high_freq_pool = [x for x in valid_triggered if x[2] == "高频"]
        return _pick_by_direction(high_freq_pool)
