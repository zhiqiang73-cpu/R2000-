import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.live_trading_engine import EngineState, LiveTradingEngine


class _PaperTraderStub:
    def __init__(self, has_position: bool = False, side: str = "LONG"):
        self._has_position = has_position
        self.current_position = SimpleNamespace(side=SimpleNamespace(value=side))

    def has_position(self) -> bool:
        return self._has_position


def _make_df(hist_values):
    n = len(hist_values)
    return pd.DataFrame(
        {
            "macd_hist": hist_values,
            "j": [50 + i for i in range(n)],
            "d": [49 + i for i in range(n)],
            "k": [50 + i for i in range(n)],
        }
    )


def _build_engine_for_entry(direction: str, hist_values):
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.state = EngineState()
    engine._paper_trader = _PaperTraderStub(False, direction)
    engine._current_prototype = None
    engine._df_buffer = _make_df(hist_values)
    engine.state.best_match_template = f"proto_{direction}_validation"
    return engine


def test_long_short_trigger_and_reject_scenarios():
    # LONG trigger: slope up and no strong zero-axis conflict
    long_trigger = _build_engine_for_entry("LONG", [-0.60, -0.45, -0.25, -0.10, 0.05])
    long_trigger._update_indicator_state()
    assert long_trigger.state.macd_ready is True
    assert long_trigger._check_indicator_gate(long_trigger._df_buffer, "LONG") is True

    # LONG reject: slope down
    long_reject = _build_engine_for_entry("LONG", [0.30, 0.20, 0.10, 0.00, -0.10])
    long_reject._update_indicator_state()
    assert long_reject.state.macd_ready is False
    assert long_reject._check_indicator_gate(long_reject._df_buffer, "LONG") is False

    # SHORT trigger: slope down and no strong zero-axis conflict
    short_trigger = _build_engine_for_entry("SHORT", [0.50, 0.35, 0.20, 0.05, -0.10])
    short_trigger._update_indicator_state()
    assert short_trigger.state.macd_ready is True
    assert short_trigger._check_indicator_gate(short_trigger._df_buffer, "SHORT") is True

    # SHORT reject: slope up
    short_reject = _build_engine_for_entry("SHORT", [-0.30, -0.15, -0.05, 0.05, 0.15])
    short_reject._update_indicator_state()
    assert short_reject.state.macd_ready is False
    assert short_reject._check_indicator_gate(short_reject._df_buffer, "SHORT") is False


def test_ui_macd_ready_matches_entry_gate_across_cases():
    long_cases = [
        [-0.60, -0.40, -0.20, -0.05, 0.10],
        [0.30, 0.25, 0.20, 0.15, 0.10],
        [-0.40, -0.35, -0.32, -0.28, -0.27],
        [0.05, 0.10, 0.08, 0.12, 0.15],
    ]
    short_cases = [
        [0.60, 0.45, 0.30, 0.10, -0.05],
        [-0.30, -0.25, -0.20, -0.15, -0.10],
        [0.40, 0.35, 0.33, 0.30, 0.29],
        [-0.05, -0.10, -0.08, -0.12, -0.15],
    ]

    for hist_values in long_cases:
        engine = _build_engine_for_entry("LONG", hist_values)
        engine._update_indicator_state()
        ui_ready = engine.state.macd_ready
        gate_ready = engine._check_indicator_gate(engine._df_buffer, "LONG")
        assert ui_ready == gate_ready

    for hist_values in short_cases:
        engine = _build_engine_for_entry("SHORT", hist_values)
        engine._update_indicator_state()
        ui_ready = engine.state.macd_ready
        gate_ready = engine._check_indicator_gate(engine._df_buffer, "SHORT")
        assert ui_ready == gate_ready


def test_macd_reject_diag_contains_numeric_trend_fields():
    df = _make_df([0.40, 0.35, 0.30, 0.25, 0.20])
    diag = LiveTradingEngine._format_macd_trend_diag(df, direction="SHORT", window=5)
    assert "窗口=5" in diag
    assert "斜率=" in diag
    assert "当前柱=" in diag
