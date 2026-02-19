"""
Verify signal-mode entry flow with historical data.

This script picks up to 2 premium + 2 high-frequency combos that can be
triggered in historical data, then simulates LiveTradingEngine to confirm:
1) combo triggers
2) engine places entry stop order
3) TP/SL are attached
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.data_loader import DataLoader
from core.signal_analyzer import _build_condition_arrays
from core.signal_store import get_premium_pool
from core.trajectory_engine import TrajectoryMemory
from core.live_data_feed import KlineData
from utils.indicators import calculate_all_indicators, calculate_support_resistance


@dataclass
class ComboHit:
    combo: dict
    hit_indices: List[int]


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        return df
    # Create synthetic timestamps if missing
    start = datetime(2026, 1, 1, 0, 0, 0)
    df = df.copy()
    df["timestamp"] = [
        int((start + timedelta(minutes=i)).timestamp() * 1000) for i in range(len(df))
    ]
    return df


def _kline_from_row(row: pd.Series) -> KlineData:
    ts_val = row["timestamp"]
    if isinstance(ts_val, pd.Timestamp):
        ts = int(ts_val.value // 1_000_000)
    else:
        ts = int(ts_val)
    open_time = datetime.fromtimestamp(ts / 1000.0)
    return KlineData(
        timestamp=ts,
        open_time=open_time,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        close_time=ts + 60_000,
        is_closed=True,
    )


def _find_hits_for_combo(
    combo: dict,
    cond_arrays: Dict[str, np.ndarray],
) -> List[int]:
    conditions = combo.get("conditions", []) or []
    if not conditions:
        return []
    mask = np.ones(len(next(iter(cond_arrays.values()))), dtype=bool)
    for cond in conditions:
        arr = cond_arrays.get(cond)
        if arr is None:
            return []
        mask &= arr
    return list(np.nonzero(mask)[0])


def _pick_combos_with_hits(
    combos: List[dict],
    cond_arrays_by_dir: Dict[str, Dict[str, np.ndarray]],
    limit: int,
) -> List[ComboHit]:
    picked: List[ComboHit] = []
    for combo in combos:
        direction = combo.get("direction", "long")
        cond_arrays = cond_arrays_by_dir.get(direction)
        if not cond_arrays:
            continue
        hits = _find_hits_for_combo(combo, cond_arrays)
        if hits:
            picked.append(ComboHit(combo=combo, hit_indices=hits))
        if len(picked) >= limit:
            break
    return picked


def _simulate_single_combo(
    df: pd.DataFrame,
    combo_hit: ComboHit,
    warmup_bars: int,
) -> Tuple[bool, Optional[dict]]:
    # Monkeypatch get_premium_pool to only return this combo
    import core.signal_store as _sig_store
    import core.binance_testnet_trader as _btt
    from core.paper_trader import PaperTrader

    original_get_pool = _sig_store.get_premium_pool
    original_trader = _btt.BinanceTestnetTrader

    def _only_this_combo(*args, **kwargs):
        return [combo_hit.combo]

    class _ShimTrader(PaperTrader):
        def __init__(self, **kwargs):
            super().__init__(
                symbol=kwargs.get("symbol", "BTCUSDT"),
                initial_balance=kwargs.get("initial_balance", 5000.0),
                leverage=kwargs.get("leverage", 10),
                position_size_pct=kwargs.get("position_size_pct", 0.1),
                on_order_update=kwargs.get("on_order_update"),
                on_trade_closed=kwargs.get("on_trade_closed"),
            )

    _sig_store.get_premium_pool = _only_this_combo
    _btt.BinanceTestnetTrader = _ShimTrader

    try:
        from core.live_trading_engine import LiveTradingEngine
        memory = TrajectoryMemory()
        engine = LiveTradingEngine(trajectory_memory=memory)
        engine.use_signal_mode = True
        engine._init_engines()

        # Prepare warmup buffer
        warmup_end = max(warmup_bars, combo_hit.hit_indices[0])
        warmup_df = df.iloc[:warmup_end].copy().reset_index(drop=True)
        engine._df_buffer = warmup_df
        engine._current_bar_idx = len(warmup_df) - 1
        engine._paper_trader.current_bar_idx = engine._current_bar_idx

        # Feed bars from warmup_end to first hit index
        target_idx = combo_hit.hit_indices[0]
        for i in range(warmup_end, target_idx + 1):
            kline = _kline_from_row(df.iloc[i])
            engine._process_closed_kline(kline)
            if engine._paper_trader.pending_stop_orders:
                return True, engine._paper_trader.pending_stop_orders[0]

        return False, None
    finally:
        _sig_store.get_premium_pool = original_get_pool
        _btt.BinanceTestnetTrader = original_trader


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify signal entry flow.")
    parser.add_argument("--sample-size", type=int, default=3000, help="Historical bars to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup bars before simulation.")
    args = parser.parse_args()

    # Load data
    loader = DataLoader()
    df = loader.sample_continuous(sample_size=args.sample_size, random_seed=args.seed)
    df = _ensure_timestamp(df)
    df = calculate_all_indicators(df)
    df = calculate_support_resistance(df)

    # Build condition arrays
    cond_arrays_by_dir = {
        "long": _build_condition_arrays(df, "long"),
        "short": _build_condition_arrays(df, "short"),
    }

    # Pick combos with actual hits
    pool = get_premium_pool(include_high_freq=True)
    premium = [c for c in pool if c.get("tier") == "精品"]
    highfreq = [c for c in pool if c.get("tier") == "高频"]

    premium_hits = _pick_combos_with_hits(premium, cond_arrays_by_dir, limit=2)
    highfreq_hits = _pick_combos_with_hits(highfreq, cond_arrays_by_dir, limit=2)
    selected = premium_hits + highfreq_hits

    if not selected:
        print("[Verify] No combos found with hits in sampled data.")
        return 1

    print(f"[Verify] Selected combos: {len(selected)}")
    for i, ch in enumerate(selected, 1):
        combo_key = ch.combo.get("combo_key", "")
        tier = ch.combo.get("tier", "")
        direction = ch.combo.get("direction", "")
        print(f"  {i}. {tier} {direction} | {combo_key} | hits={len(ch.hit_indices)}")

    # Simulate each combo
    for i, ch in enumerate(selected, 1):
        combo_key = ch.combo.get("combo_key", "")
        print(f"\n[Verify] Simulating combo {i}: {combo_key}")
        ok, order = _simulate_single_combo(df, ch, warmup_bars=args.warmup)
        if not ok:
            print("  -> FAIL: No pending order placed at first hit index.")
            continue
        print("  -> OK: Pending order placed.")
        tp = order.get("tp")
        sl = order.get("sl")
        trigger = order.get("trigger_price")
        print(
            f"     side={order.get('side')} trigger={trigger:.2f} "
            f"tp={tp if tp is None else format(tp, '.2f')} "
            f"sl={sl if sl is None else format(sl, '.2f')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
