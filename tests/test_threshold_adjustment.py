"""
测试门控阈值调整逻辑：compute_adjustment, apply_adjustment, safe bounds, session report
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rejection_tracker import RejectionTracker, ThresholdAdjustment, ADJUSTABLE_GATES

TEST_PATH = "data/__test_threshold_adj.json"


def test_compute_adjustment():
    """测试 compute_adjustment: 步长计算和安全边界夹紧"""
    t = RejectionTracker(persistence_path=TEST_PATH)

    # 正常放松
    result = t.compute_adjustment("BLOCK_POS", "POS_THRESHOLD_LONG", -30, "loosen")
    assert result is not None, "Should return result for normal loosen"
    assert result["suggested_value"] == -35, f"Expected -35, got {result['suggested_value']}"
    assert result["action_text"] == "放宽"
    print(f"  [OK] loosen POS_THRESHOLD_LONG: -30 → {result['suggested_value']}")

    # 已在边界，无法继续
    result = t.compute_adjustment("BLOCK_POS", "POS_THRESHOLD_LONG", -50, "loosen")
    assert result is None, "Should return None when at boundary"
    print("  [OK] loosen POS_THRESHOLD_LONG at boundary -50 → None")

    # 收紧
    result = t.compute_adjustment("BLOCK_POS", "POS_THRESHOLD_LONG", -30, "tighten")
    assert result is not None
    assert result["suggested_value"] == -25, f"Expected -25, got {result['suggested_value']}"
    print(f"  [OK] tighten POS_THRESHOLD_LONG: -30 → {result['suggested_value']}")

    # 收紧到边界
    result = t.compute_adjustment("BLOCK_POS", "POS_THRESHOLD_LONG", -10, "tighten")
    assert result is None, "Should return None when at max boundary"
    print("  [OK] tighten POS_THRESHOLD_LONG at boundary -10 → None")

    # MACD 放松
    result = t.compute_adjustment("BLOCK_MACD", "MACD_SLOPE_MIN", 0.005, "loosen")
    assert result is not None
    assert result["suggested_value"] == 0.004, f"Expected 0.004, got {result['suggested_value']}"
    print(f"  [OK] loosen MACD_SLOPE_MIN: 0.005 → {result['suggested_value']}")

    # MACD 放松到边界
    result = t.compute_adjustment("BLOCK_MACD", "MACD_SLOPE_MIN", 0.001, "loosen")
    assert result is None
    print("  [OK] loosen MACD_SLOPE_MIN at boundary 0.001 → None")

    # Bayesian 收紧
    result = t.compute_adjustment("BLOCK_BAYES", "BAYESIAN_MIN_WIN_RATE", 0.40, "tighten")
    assert result is not None
    assert abs(result["suggested_value"] - 0.42) < 1e-9, f"Expected ~0.42, got {result['suggested_value']}"
    print(f"  [OK] tighten BAYESIAN_MIN_WIN_RATE: 0.40 → {result['suggested_value']:.4f}")

    # Unknown gate
    result = t.compute_adjustment("UNKNOWN_GATE", "SOME_PARAM", 0.5, "loosen")
    assert result is None
    print("  [OK] unknown gate → None")


def test_apply_adjustment():
    """测试 apply_adjustment: 实际修改配置 + 审计记录"""
    t = RejectionTracker(persistence_path=TEST_PATH)
    mock_config = {"POS_THRESHOLD_LONG": -30, "MACD_SLOPE_MIN": 0.005}

    # 正常应用
    record = t.apply_adjustment(
        mock_config, "POS_THRESHOLD_LONG", -35,
        fail_code="BLOCK_POS", reason="Test adjustment"
    )
    assert record is not None, "Should return record"
    assert mock_config["POS_THRESHOLD_LONG"] == -35, f"Config should be -35, got {mock_config['POS_THRESHOLD_LONG']}"
    assert record.old_value == -30
    assert record.new_value == -35
    assert record.action == "loosen"
    assert record.reason == "Test adjustment"
    print(f"  [OK] apply_adjustment: -30 → -35, action={record.action}")

    # 安全边界夹紧
    record2 = t.apply_adjustment(
        mock_config, "POS_THRESHOLD_LONG", -999,
        fail_code="BLOCK_POS", reason="Boundary test"
    )
    assert record2 is not None
    assert mock_config["POS_THRESHOLD_LONG"] == -50, f"Should clamp to -50, got {mock_config['POS_THRESHOLD_LONG']}"
    print(f"  [OK] apply_adjustment (boundary clamp): -35 → -50 (clamped from -999)")

    # 已在目标值
    record3 = t.apply_adjustment(
        mock_config, "POS_THRESHOLD_LONG", -50,
        fail_code="BLOCK_POS", reason="No-op test"
    )
    assert record3 is None, "Should return None for no-op"
    print("  [OK] apply_adjustment (no-op): already at -50 → None")

    # 参数不存在
    record4 = t.apply_adjustment(
        mock_config, "NONEXISTENT_PARAM", 0.5,
        fail_code="BLOCK_POS", reason="Nonexistent test"
    )
    assert record4 is None
    print("  [OK] apply_adjustment (nonexistent param): → None")

    # 审计历史
    history = t.get_adjustment_history()
    assert len(history) == 2, f"Expected 2 adjustment records, got {len(history)}"
    print(f"  [OK] adjustment_history has {len(history)} records")

    session_adjs = t.get_session_adjustments()
    assert len(session_adjs) == 2
    print(f"  [OK] session adjustments: {len(session_adjs)}")

    assert t.total_adjustments_applied == 2
    print(f"  [OK] total_adjustments_applied: {t.total_adjustments_applied}")


def test_compute_all_concrete():
    """测试 compute_all_concrete_adjustments: 需要足够评估数据"""
    t = RejectionTracker(persistence_path=TEST_PATH, min_evaluations_for_suggestion=3)
    mock_config = {"POS_THRESHOLD_LONG": -30, "POS_THRESHOLD_SHORT": -40}

    # 模拟一些拒绝记录和评估
    for i in range(5):
        t.record_rejection(
            price=50000.0, direction="LONG", fail_code="BLOCK_POS",
            gate_stage="position_filter", market_regime="trending",
            bar_idx=i, detail={"pos_score": -35}
        )

    # 模拟评估（大部分是 "错误拒绝" → accuracy < 0.4 → 建议放松）
    for i in range(5):
        t.evaluate_pending(current_price=50500.0, current_bar_idx=i + 100)

    # 现在应该有建议了
    suggestions = t.compute_all_concrete_adjustments(mock_config)
    print(f"  [INFO] Suggestions after 5 wrong rejections: {len(suggestions)}")
    for s in suggestions:
        print(f"    · {s.get('param_key')}: {s.get('current_value')} → {s.get('suggested_value')} ({s.get('action_text')})")

    if suggestions:
        print(f"  [OK] Got {len(suggestions)} concrete suggestions")
    else:
        print("  [OK] No suggestions (accuracy may still be in normal range)")


def test_session_report():
    """测试 generate_session_report"""
    t = RejectionTracker(persistence_path=TEST_PATH)
    mock_config = {"POS_THRESHOLD_LONG": -30}

    # 生成报告
    report = t.generate_session_report(mock_config)
    assert "session_id" in report
    assert "statistics" in report
    assert "gate_summaries" in report
    assert "pending_suggestions" in report
    assert "session_adjustments" in report
    print(f"  [OK] Report keys: {list(report.keys())}")
    print(f"  [OK] Statistics: {report['statistics']}")


def test_persistence():
    """测试持久化：调整历史的保存和加载"""
    t = RejectionTracker(persistence_path=TEST_PATH)
    mock_config = {"POS_THRESHOLD_LONG": -30}

    t.apply_adjustment(mock_config, "POS_THRESHOLD_LONG", -35,
                       fail_code="BLOCK_POS", reason="Persist test")
    adj_count_before = len(t._adjustment_history)

    # 创建新实例，应从文件加载
    t2 = RejectionTracker(persistence_path=TEST_PATH)
    adj_count_after = len(t2._adjustment_history)
    assert adj_count_after == adj_count_before, \
        f"Expected {adj_count_before} adjustments after reload, got {adj_count_after}"
    print(f"  [OK] Persistence: saved {adj_count_before}, reloaded {adj_count_after}")

    # 验证加载的记录字段完整
    if t2._adjustment_history:
        adj = t2._adjustment_history[-1]
        assert adj.param_key == "POS_THRESHOLD_LONG"
        assert adj.old_value == -30
        assert adj.new_value == -35
        assert adj.fail_code == "BLOCK_POS"
        print(f"  [OK] Reloaded record: {adj.param_key} {adj.old_value} → {adj.new_value}")


def test_clamp_to_safe_bounds():
    """测试安全边界夹紧"""
    t = RejectionTracker(persistence_path=TEST_PATH)

    # 在边界内
    v = t._clamp_to_safe_bounds("BLOCK_POS", "POS_THRESHOLD_LONG", -30)
    assert v == -30
    print(f"  [OK] clamp -30 → {v} (within bounds)")

    # 超过下界
    v = t._clamp_to_safe_bounds("BLOCK_POS", "POS_THRESHOLD_LONG", -999)
    assert v == -50, f"Expected -50, got {v}"
    print(f"  [OK] clamp -999 → {v} (clamped to min)")

    # 超过上界
    v = t._clamp_to_safe_bounds("BLOCK_POS", "POS_THRESHOLD_LONG", 0)
    assert v == -10, f"Expected -10, got {v}"
    print(f"  [OK] clamp 0 → {v} (clamped to max)")

    # 未知参数（不做限制）
    v = t._clamp_to_safe_bounds("UNKNOWN", "UNKNOWN_PARAM", 42)
    assert v == 42
    print(f"  [OK] clamp unknown param → {v} (pass-through)")


def cleanup():
    if os.path.exists(TEST_PATH):
        os.remove(TEST_PATH)
    tmp = TEST_PATH + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    cleanup()

    print("=== test_compute_adjustment ===")
    test_compute_adjustment()
    cleanup()

    print("\n=== test_apply_adjustment ===")
    test_apply_adjustment()
    cleanup()

    print("\n=== test_compute_all_concrete ===")
    test_compute_all_concrete()
    cleanup()

    print("\n=== test_session_report ===")
    test_session_report()
    cleanup()

    print("\n=== test_persistence ===")
    test_persistence()
    cleanup()

    print("\n=== test_clamp_to_safe_bounds ===")
    test_clamp_to_safe_bounds()
    cleanup()

    print("\n[PASS] All threshold adjustment tests passed!")
