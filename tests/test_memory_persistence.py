"""
R3000 记忆持久化功能测试
"""

import sys
import os
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_trajectory_template_serialization():
    """测试模板序列化"""
    print("\n[Test 1] TrajectoryTemplate 序列化...")

    from core.trajectory_engine import TrajectoryTemplate

    # 创建模板
    template = TrajectoryTemplate(
        trade_idx=5,
        regime="强多头",
        direction="LONG",
        profit_pct=2.5,
        pre_entry=np.random.randn(60, 32),
        holding=np.random.randn(100, 32),
        pre_exit=np.random.randn(30, 32),
        entry_idx=200,
        exit_idx=300,
    )

    # 序列化
    d = template.to_dict()
    assert isinstance(d, dict)
    assert d["regime"] == "强多头"
    assert d["direction"] == "LONG"
    assert len(d["pre_entry"]) == 60
    print(f"    序列化: {len(d)} 个字段")

    # 反序列化
    restored = TrajectoryTemplate.from_dict(d)
    assert restored.regime == template.regime
    assert restored.direction == template.direction
    assert restored.profit_pct == template.profit_pct
    assert restored.pre_entry.shape == template.pre_entry.shape
    assert np.allclose(restored.pre_entry, template.pre_entry)
    print(f"    反序列化: pre_entry shape = {restored.pre_entry.shape}")

    # 指纹
    fp1 = template.fingerprint()
    fp2 = restored.fingerprint()
    print(f"    指纹: {fp1[:20]}...")

    print("    [PASS] TrajectoryTemplate 序列化")
    return True


def test_memory_save_load():
    """测试记忆体保存加载"""
    print("\n[Test 2] TrajectoryMemory 保存/加载...")

    from core.trajectory_engine import TrajectoryMemory, TrajectoryTemplate

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 创建记忆体并添加模板
        memory = TrajectoryMemory()

        for i in range(10):
            regime = ["强多头", "弱多头", "震荡"][i % 3]
            direction = "LONG" if i % 2 == 0 else "SHORT"
            template = TrajectoryTemplate(
                trade_idx=i,
                regime=regime,
                direction=direction,
                profit_pct=1.0 + i * 0.5,
                pre_entry=np.random.randn(60, 32),
                holding=np.random.randn(50 + i * 10, 32),
                pre_exit=np.random.randn(30, 32),
            )
            memory._add_template(regime, direction, template)

        assert memory.total_count == 10
        print(f"    创建记忆体: {memory.total_count} 个模板")

        # 保存
        filepath = os.path.join(temp_dir, "test_memory.pkl")
        saved_path = memory.save(filepath, verbose=False)
        assert os.path.exists(saved_path)
        file_size = os.path.getsize(saved_path)
        print(f"    保存: {saved_path} ({file_size / 1024:.1f} KB)")

        # 加载
        loaded = TrajectoryMemory.load(filepath, verbose=False)
        assert loaded.total_count == 10
        print(f"    加载: {loaded.total_count} 个模板")

        # 验证内容
        orig_templates = memory.get_all_templates()
        loaded_templates = loaded.get_all_templates()
        assert len(orig_templates) == len(loaded_templates)

        # 检查第一个模板
        orig = orig_templates[0]
        load = loaded_templates[0]
        assert orig.regime == load.regime
        assert orig.direction == load.direction
        assert orig.profit_pct == load.profit_pct
        print(f"    验证: 模板内容一致")

        print("    [PASS] TrajectoryMemory 保存/加载")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_memory_merge():
    """测试记忆体合并"""
    print("\n[Test 3] TrajectoryMemory 合并...")

    from core.trajectory_engine import TrajectoryMemory, TrajectoryTemplate

    # 创建两个记忆体
    memory1 = TrajectoryMemory()
    memory2 = TrajectoryMemory()

    # 添加模板到第一个
    for i in range(5):
        template = TrajectoryTemplate(
            trade_idx=i,
            regime="测试",
            direction="LONG",
            profit_pct=1.0 + i,
            pre_entry=np.random.randn(60, 32),
            holding=np.random.randn(50, 32),
            pre_exit=np.random.randn(30, 32),
        )
        memory1._add_template("测试", "LONG", template)

    # 添加模板到第二个（3个新的，2个重复的）
    for i in range(3, 8):
        template = TrajectoryTemplate(
            trade_idx=i,
            regime="测试",
            direction="LONG",
            profit_pct=1.0 + i,
            pre_entry=np.random.randn(60, 32),
            holding=np.random.randn(50, 32),
            pre_exit=np.random.randn(30, 32),
        )
        memory2._add_template("测试", "LONG", template)

    print(f"    Memory1: {memory1.total_count} 个模板")
    print(f"    Memory2: {memory2.total_count} 个模板")

    # 合并（带去重）
    added = memory1.merge(memory2, deduplicate=True, verbose=False)
    print(f"    合并后: {memory1.total_count} 个模板 (新增 {added})")

    # 所有模板都是新的（因为随机数据导致指纹不同）
    assert memory1.total_count == 10
    assert added == 5

    print("    [PASS] TrajectoryMemory 合并")
    return True


def test_list_memories():
    """测试列出记忆文件"""
    print("\n[Test 4] 列出记忆文件...")

    from core.trajectory_engine import TrajectoryMemory, TrajectoryTemplate

    temp_dir = tempfile.mkdtemp()

    try:
        # 创建几个记忆文件
        for i in range(3):
            memory = TrajectoryMemory()
            template = TrajectoryTemplate(
                trade_idx=i,
                regime="测试",
                direction="LONG",
                profit_pct=1.0,
                pre_entry=np.random.randn(60, 32),
                holding=np.random.randn(50, 32),
                pre_exit=np.random.randn(30, 32),
            )
            memory._add_template("测试", "LONG", template)
            memory.save(os.path.join(temp_dir, f"memory_{i}.pkl"), verbose=False)

        # 列出文件
        files = TrajectoryMemory.list_saved_memories(temp_dir)
        assert len(files) == 3
        print(f"    找到 {len(files)} 个记忆文件")

        for f in files:
            print(f"      - {f['filename']}: {f['size_mb']:.3f} MB")

        # 加载最新
        latest = TrajectoryMemory.load_latest(temp_dir, verbose=False)
        assert latest is not None
        assert latest.total_count == 1
        print(f"    加载最新: {latest.total_count} 个模板")

        # 加载并合并全部
        merged = TrajectoryMemory.load_and_merge_all(temp_dir, verbose=False)
        assert merged.total_count == 3
        print(f"    合并全部: {merged.total_count} 个模板")

        print("    [PASS] 列出记忆文件")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_json_export():
    """测试 JSON 统计导出"""
    print("\n[Test 5] JSON 统计导出...")

    from core.trajectory_engine import TrajectoryMemory, TrajectoryTemplate
    import json

    temp_dir = tempfile.mkdtemp()

    try:
        memory = TrajectoryMemory()

        for i in range(10):
            regime = ["强多头", "弱空头"][i % 2]
            direction = ["LONG", "SHORT"][i % 2]
            template = TrajectoryTemplate(
                trade_idx=i,
                regime=regime,
                direction=direction,
                profit_pct=1.0 + i * 0.3,
                pre_entry=np.random.randn(60, 32),
                holding=np.random.randn(50, 32),
                pre_exit=np.random.randn(30, 32),
            )
            memory._add_template(regime, direction, template)

        # 导出 JSON
        json_path = os.path.join(temp_dir, "stats.json")
        memory.export_stats_json(json_path)

        assert os.path.exists(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        assert stats["total_templates"] == 10
        assert len(stats["regimes"]) == 2
        print(f"    导出统计: {stats['total_templates']} 模板, {len(stats['regimes'])} 个状态")

        for regime, directions in stats["regimes"].items():
            for direction, info in directions.items():
                print(f"      {regime}/{direction}: {info['count']} 个, "
                      f"avg={info['avg_profit_pct']:.2f}%")

        print("    [PASS] JSON 统计导出")
        return True

    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("R3000 记忆持久化 - 单元测试")
    print("=" * 60)

    tests = [
        test_trajectory_template_serialization,
        test_memory_save_load,
        test_memory_merge,
        test_list_memories,
        test_json_export,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
