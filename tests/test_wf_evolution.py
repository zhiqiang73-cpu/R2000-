"""
Quick validation test for core/wf_evolution.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.wf_evolution import (
    VectorizedSimilarityEngine, simulate_trades_from_scores,
    expand_group_weights_to_32d, DEFAULT_GROUP_WEIGHTS,
    GROUP_NAMES, GROUP_TO_INDICES, ALL_FEATURE_NAMES,
    EvolutionResult,
)


def test_feature_groups():
    """Validate all 32 features are covered by exactly one group"""
    all_covered = set()
    for group, indices in GROUP_TO_INDICES.items():
        for idx in indices:
            assert idx not in all_covered, f"Index {idx} duplicated in group {group}"
            all_covered.add(idx)
    
    missing = set(range(32)) - all_covered
    assert not missing, f"Indices not covered: {missing}"
    print("[PASS] Feature group coverage: all 32 features covered")


def test_weight_expansion():
    """Test 8D -> 32D weight expansion"""
    # All ones
    w8 = np.ones(8)
    w32 = expand_group_weights_to_32d(w8)
    assert w32.shape == (32,)
    assert np.all(w32 == 1.0)
    
    # Different per-group
    w8 = np.array([2.0, 1.0, 1.5, 3.0, 0.5, 1.2, 0.8, 2.5])
    w32 = expand_group_weights_to_32d(w8)
    assert w32.shape == (32,)
    
    # Check RSI group features all get weight 2.0
    for idx in GROUP_TO_INDICES["RSI"]:
        assert w32[idx] == 2.0, f"RSI feature at idx {idx} should be 2.0, got {w32[idx]}"
    
    # Check Structure group features all get weight 0.8
    for idx in GROUP_TO_INDICES["Structure"]:
        assert w32[idx] == 0.8, f"Structure feature at idx {idx} should be 0.8, got {w32[idx]}"
    
    print("[PASS] Weight expansion: 8D -> 32D correct")


def test_vectorized_similarity():
    """Test VectorizedSimilarityEngine score computation"""
    np.random.seed(42)
    n_eval = 50
    n_proto = 10
    
    engine = VectorizedSimilarityEngine()
    engine._bar_means = np.random.randn(n_eval, 32)
    engine._eval_indices = list(range(60, 60 + n_eval * 10, 10))
    engine._proto_means = np.random.randn(n_proto, 32)
    engine._proto_confidences = np.random.uniform(0.3, 0.9, n_proto)
    engine._proto_directions = ["LONG"] * 5 + ["SHORT"] * 5
    engine._proto_objects = [None] * n_proto
    engine._ready = True
    
    weights = expand_group_weights_to_32d(DEFAULT_GROUP_WEIGHTS)
    
    final_scores, combined = engine.compute_score_matrix(weights)
    
    assert final_scores.shape == (n_eval, n_proto)
    assert combined.shape == (n_eval, n_proto)
    assert np.all(np.isfinite(final_scores))
    assert np.all(np.isfinite(combined))
    # Final scores should be <= combined (because confidence <= 1)
    assert np.all(final_scores <= combined + 1e-6)
    
    print(f"[PASS] Vectorized similarity: shape={final_scores.shape}, "
          f"range=[{final_scores.min():.4f}, {final_scores.max():.4f}]")


def test_trade_simulation():
    """Test simulate_trades_from_scores"""
    np.random.seed(42)
    n_eval = 100
    n_proto = 8
    n_bars = 2000
    
    # Create realistic-ish data
    close = np.cumsum(np.random.randn(n_bars) * 0.1) + 100.0
    atr = np.ones(n_bars) * 0.5
    
    # Create score matrix with some high-scoring entries
    scores = np.random.uniform(0.3, 0.8, (n_eval, n_proto))
    eval_indices = list(range(60, 60 + n_eval * 10, 10))
    directions = ["LONG"] * 4 + ["SHORT"] * 4
    
    result = simulate_trades_from_scores(
        final_scores=scores,
        eval_indices=eval_indices,
        proto_directions=directions,
        proto_objects=[None] * n_proto,
        close_prices=close,
        atr_values=atr,
        fusion_threshold=0.5,
        cosine_min_threshold=0.0,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        max_hold_bars=240,
    )
    
    assert isinstance(result, dict)
    assert "n_trades" in result
    assert "sharpe_ratio" in result
    assert "win_rate" in result
    assert "max_drawdown" in result
    assert result["n_trades"] >= 0
    
    if result["n_trades"] > 0:
        assert 0 <= result["win_rate"] <= 1.0
    
    print(f"[PASS] Trade simulation: trades={result['n_trades']}, "
          f"sharpe={result['sharpe_ratio']:.3f}, winrate={result['win_rate']:.1%}")


def test_evolution_result():
    """Test EvolutionResult dataclass"""
    r = EvolutionResult()
    r.holdout_sharpe = 0.8
    r.holdout_n_trades = 50
    r.holdout_win_rate = 0.55
    r.holdout_max_drawdown = 8.0
    
    assert r.check_holdout_pass() == True
    
    r.holdout_sharpe = 0.1  # Below threshold
    assert r.check_holdout_pass() == False
    
    r.best_group_weights = DEFAULT_GROUP_WEIGHTS.copy()
    r.compute_group_importance()
    assert len(r.group_importance) == len(GROUP_NAMES)
    
    print("[PASS] EvolutionResult: pass criteria and importance work")


if __name__ == "__main__":
    test_feature_groups()
    test_weight_expansion()
    test_vectorized_similarity()
    test_trade_simulation()
    test_evolution_result()
    print("\n=== ALL TESTS PASSED ===")
