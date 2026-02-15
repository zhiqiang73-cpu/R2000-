"""
R3000 PaperOrder → TradeRecord 转换模块单元测试

测试增量训练相关功能：
- PaperOrder 到 TradeRecord 的转换
- 批量转换
- 模板创建
- IncrementalTrainer 工作流
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_paper_order(
    order_id: str = "SIM_000001",
    side: str = "LONG",
    profit_pct: float = 2.5,
    has_trajectory: bool = True,
    status: str = "CLOSED"
):
    """创建模拟 PaperOrder"""
    from core.paper_trader import PaperOrder, OrderSide, OrderStatus, CloseReason
    
    # 创建模拟轨迹矩阵 (60, 32)
    trajectory = None
    if has_trajectory:
        np.random.seed(hash(order_id) % 2**32)
        trajectory = np.random.randn(60, 32).astype(np.float32)
    
    order = PaperOrder(
        order_id=order_id,
        symbol="BTCUSDT",
        side=OrderSide.LONG if side == "LONG" else OrderSide.SHORT,
        quantity=0.01,
        margin_used=100.0,
        entry_price=50000.0,
        entry_time=datetime.now() - timedelta(hours=1),
        entry_bar_idx=100,
        take_profit=51000.0,
        stop_loss=49500.0,
        status=OrderStatus.CLOSED if status == "CLOSED" else OrderStatus.FILLED,
        exit_price=50000.0 * (1 + profit_pct / 100 / 10),  # 考虑杠杆
        exit_time=datetime.now(),
        exit_bar_idx=110,
        close_reason=CloseReason.TAKE_PROFIT if profit_pct > 0 else CloseReason.STOP_LOSS,
        realized_pnl=profit_pct * 10,  # 简化计算
        profit_pct=profit_pct,
        template_fingerprint=f"fp_{order_id}",
        entry_similarity=0.85,
        hold_bars=10,
        entry_trajectory=trajectory,
    )
    return order


def test_paper_order_to_trade_record():
    """测试单个 PaperOrder 转换"""
    print("\n[Test 1] paper_order_to_trade_record...")
    
    from core.trajectory_engine import paper_order_to_trade_record
    from core.paper_trader import OrderSide
    
    # 测试盈利多单
    order = create_mock_paper_order(
        order_id="SIM_001",
        side="LONG",
        profit_pct=5.0,
        has_trajectory=True
    )
    
    record = paper_order_to_trade_record(order)
    
    assert record is not None, "Conversion should succeed"
    assert record.side == 1, f"Expected side=1 (LONG), got {record.side}"
    assert record.profit_pct == 5.0, f"Expected profit_pct=5.0, got {record.profit_pct}"
    assert record.exit_reason == "tp", f"Expected exit_reason='tp', got {record.exit_reason}"
    assert record.entry_trajectory is not None, "Should have trajectory"
    assert record.entry_trajectory.shape == (60, 32), f"Expected (60,32), got {record.entry_trajectory.shape}"
    
    print(f"    [OK] Profitable LONG: side={record.side}, profit={record.profit_pct}%, exit={record.exit_reason}")
    
    # 测试亏损空单
    order_short = create_mock_paper_order(
        order_id="SIM_002",
        side="SHORT",
        profit_pct=-3.0,
        has_trajectory=True
    )
    
    record_short = paper_order_to_trade_record(order_short)
    
    assert record_short is not None, "Conversion should succeed"
    assert record_short.side == -1, f"Expected side=-1 (SHORT), got {record_short.side}"
    assert record_short.profit_pct == -3.0, f"Expected profit_pct=-3.0, got {record_short.profit_pct}"
    assert record_short.exit_reason == "sl", f"Expected exit_reason='sl', got {record_short.exit_reason}"
    
    print(f"    [OK] Losing SHORT: side={record_short.side}, profit={record_short.profit_pct}%, exit={record_short.exit_reason}")
    
    # 测试未平仓订单（应返回 None）
    order_open = create_mock_paper_order(
        order_id="SIM_003",
        side="LONG",
        profit_pct=0.0,
        has_trajectory=True,
        status="FILLED"  # 未平仓
    )
    
    record_open = paper_order_to_trade_record(order_open)
    assert record_open is None, "Open order should return None"
    print(f"    [OK] Open order correctly returns None")
    
    print("    [OK] Test 1 PASSED")


def test_paper_orders_to_trade_records():
    """测试批量转换"""
    print("\n[Test 2] paper_orders_to_trade_records...")
    
    from core.trajectory_engine import paper_orders_to_trade_records
    
    # 创建混合订单列表
    orders = [
        create_mock_paper_order(f"SIM_{i:03d}", "LONG" if i % 2 == 0 else "SHORT",
                                profit_pct=float(i - 5), has_trajectory=True)
        for i in range(10)
    ]
    
    # 添加一个未平仓订单
    orders.append(create_mock_paper_order("SIM_OPEN", "LONG", 0.0, True, status="FILLED"))
    
    # 添加一个无轨迹订单
    orders.append(create_mock_paper_order("SIM_NO_TRAJ", "SHORT", 2.0, has_trajectory=False))
    
    records = paper_orders_to_trade_records(orders, verbose=True)
    
    # 应该转换 10 + 1(无轨迹但已平仓) = 11 笔，跳过 1 笔未平仓
    assert len(records) == 11, f"Expected 11 records, got {len(records)}"
    
    # 检查方向分布
    long_count = sum(1 for r in records if r.side == 1)
    short_count = sum(1 for r in records if r.side == -1)
    print(f"    转换结果: {len(records)} 笔 (多单 {long_count}, 空单 {short_count})")
    
    print("    [OK] Test 2 PASSED")


def test_create_templates_from_paper_orders():
    """测试从 PaperOrder 创建模板"""
    print("\n[Test 3] create_templates_from_paper_orders...")
    
    from core.trajectory_engine import create_templates_from_paper_orders
    
    # 创建订单列表（包含盈亏混合）
    orders = []
    for i in range(20):
        profit = float((i % 10) - 3)  # -3 到 6
        orders.append(create_mock_paper_order(
            f"TPL_{i:03d}",
            "LONG" if i % 2 == 0 else "SHORT",
            profit_pct=profit,
            has_trajectory=True
        ))
    
    # 添加无轨迹订单
    orders.append(create_mock_paper_order("TPL_NO_TRAJ", "LONG", 5.0, has_trajectory=False))
    
    # 测试不同阈值
    templates_all = create_templates_from_paper_orders(orders, min_profit_pct=-10.0, verbose=True)
    templates_profit = create_templates_from_paper_orders(orders, min_profit_pct=0.0, verbose=True)
    templates_high = create_templates_from_paper_orders(orders, min_profit_pct=3.0, verbose=True)
    
    print(f"    阈值测试: all={len(templates_all)}, profit>0={len(templates_profit)}, profit>3={len(templates_high)}")
    
    assert len(templates_all) == 20, f"Expected 20 templates (no filter), got {len(templates_all)}"
    assert len(templates_profit) < len(templates_all), "Higher threshold should filter more"
    assert len(templates_high) < len(templates_profit), "Even higher threshold should filter more"
    
    # 验证模板结构
    if templates_profit:
        t = templates_profit[0]
        assert t.pre_entry.shape == (60, 32), f"Expected (60,32), got {t.pre_entry.shape}"
        assert t.direction in ("LONG", "SHORT"), f"Invalid direction: {t.direction}"
        print(f"    模板结构验证: direction={t.direction}, profit={t.profit_pct:.2f}%, pre_entry={t.pre_entry.shape}")
    
    print("    [OK] Test 3 PASSED")


def test_incremental_trainer():
    """测试 IncrementalTrainer 工作流"""
    print("\n[Test 4] IncrementalTrainer...")
    
    from core.trajectory_engine import IncrementalTrainer, create_templates_from_paper_orders
    
    # 创建训练器（无原型库）
    trainer = IncrementalTrainer()
    
    # 添加订单
    orders = [
        create_mock_paper_order(f"INC_{i:03d}", "LONG" if i % 2 == 0 else "SHORT",
                                profit_pct=float(i), has_trajectory=True)
        for i in range(5)
    ]
    trainer.add_paper_orders(orders)
    
    assert len(trainer.pending_orders) == 5, f"Expected 5 pending orders, got {len(trainer.pending_orders)}"
    print(f"    添加订单: {len(trainer.pending_orders)} 笔")
    
    # 执行训练
    result = trainer.train(min_profit_pct=0.0, verbose=True)
    
    assert result["orders_processed"] == 5, f"Expected 5 orders processed, got {result['orders_processed']}"
    assert result["templates_added"] > 0, "Should have added templates"
    assert len(trainer.pending_orders) == 0, "Pending orders should be cleared"
    
    print(f"    训练结果: 处理 {result['orders_processed']} 笔, 新增 {result['templates_added']} 个模板")
    
    # 验证有记忆库返回
    assert "memory" in result, "Should return memory when no library"
    memory = result["memory"]
    assert memory.total_count > 0, "Memory should have templates"
    print(f"    记忆库: {memory.total_count} 个模板")
    
    print("    [OK] Test 4 PASSED")


def test_trade_record_serialization():
    """测试 PaperOrderTradeRecord 序列化"""
    print("\n[Test 5] PaperOrderTradeRecord serialization...")
    
    from core.trajectory_engine import paper_order_to_trade_record, PaperOrderTradeRecord
    
    order = create_mock_paper_order("SER_001", "LONG", 3.5, has_trajectory=True)
    record = paper_order_to_trade_record(order)
    
    # 序列化
    data = record.to_dict()
    assert isinstance(data, dict), "to_dict should return dict"
    assert data["profit_pct"] == 3.5, f"Expected profit_pct=3.5, got {data['profit_pct']}"
    assert data["entry_trajectory"] is not None, "Should have trajectory data"
    
    print(f"    序列化: {len(data)} 个字段")
    
    # 反序列化
    restored = PaperOrderTradeRecord.from_dict(data)
    assert restored.profit_pct == record.profit_pct, "Profit should match"
    assert restored.side == record.side, "Side should match"
    assert restored.entry_trajectory is not None, "Should have trajectory"
    assert np.allclose(restored.entry_trajectory, record.entry_trajectory), "Trajectory should match"
    
    print(f"    反序列化: profit={restored.profit_pct}%, side={restored.side}")
    
    print("    [OK] Test 5 PASSED")


def test_get_incremental_training_candidates():
    """测试增量训练候选筛选"""
    print("\n[Test 6] get_incremental_training_candidates...")
    
    from core.trajectory_engine import get_incremental_training_candidates
    
    # 创建混合订单
    orders = []
    for i in range(15):
        profit = float(i - 7)  # -7 到 7
        has_traj = i % 3 != 0  # 每3个有1个无轨迹
        status = "CLOSED" if i % 5 != 0 else "FILLED"  # 每5个有1个未平仓
        orders.append(create_mock_paper_order(
            f"CAND_{i:03d}",
            "LONG" if i % 2 == 0 else "SHORT",
            profit_pct=profit,
            has_trajectory=has_traj,
            status=status
        ))
    
    # 测试不同条件
    candidates1, stats1 = get_incremental_training_candidates(
        orders, min_profit_pct=0.0, require_trajectory=True, verbose=True
    )
    
    candidates2, stats2 = get_incremental_training_candidates(
        orders, min_profit_pct=0.0, require_trajectory=False, verbose=True
    )
    
    candidates3, stats3 = get_incremental_training_candidates(
        orders, min_profit_pct=3.0, require_trajectory=True, verbose=True
    )
    
    print(f"    筛选测试:")
    print(f"      require_traj=True, min=0: {len(candidates1)} 笔")
    print(f"      require_traj=False, min=0: {len(candidates2)} 笔")
    print(f"      require_traj=True, min=3: {len(candidates3)} 笔")
    
    assert len(candidates2) >= len(candidates1), "Without trajectory requirement should have more"
    assert len(candidates1) >= len(candidates3), "Higher profit threshold should have less"
    
    # 验证统计信息
    assert "total" in stats1, "Stats should have total"
    assert "qualified" in stats1, "Stats should have qualified"
    assert stats1["qualified"] == len(candidates1), "Qualified should match candidates"
    
    print("    [OK] Test 6 PASSED")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("PaperOrder → TradeRecord 转换模块测试")
    print("=" * 60)
    
    try:
        test_paper_order_to_trade_record()
        test_paper_orders_to_trade_records()
        test_create_templates_from_paper_orders()
        test_incremental_trainer()
        test_trade_record_serialization()
        test_get_incremental_training_candidates()
        
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
