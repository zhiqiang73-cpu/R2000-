"""
R3000 轨迹匹配核心模块单元测试
不依赖真实数据，使用模拟数据测试核心功能
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_df(n: int = 1000):
    """创建模拟K线数据"""
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.random.randint(100, 10000, n).astype(float)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })

    # 添加必要的指标
    df['atr'] = (df['high'] - df['low']).rolling(14).mean().fillna(100)
    df['rsi'] = 50 + np.random.randn(n) * 15
    df['macd'] = np.random.randn(n) * 10
    df['macd_signal'] = df['macd'].rolling(9).mean().fillna(0)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['volume_ma5'] = df['volume'].rolling(5).mean().fillna(df['volume'])
    df['volume_ma20'] = df['volume'].rolling(20).mean().fillna(df['volume'])
    df['obv'] = df['volume'].cumsum()
    df['adx'] = 25 + np.random.randn(n) * 10
    df['boll_mid'] = df['close'].rolling(20).mean().fillna(df['close'])
    df['boll_std'] = df['close'].rolling(20).std().fillna(100)
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    df['kdj_k'] = 50 + np.random.randn(n) * 20
    df['kdj_d'] = df['kdj_k'].rolling(3).mean().fillna(50)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    return df


def test_feature_vector_engine():
    """测试特征向量引擎"""
    print("\n[Test 1] FeatureVectorEngine...")

    from core.feature_vector import FeatureVectorEngine, N_A, N_B, N_C

    df = create_mock_df(500)
    engine = FeatureVectorEngine()
    engine.precompute(df)

    assert engine.is_ready, "Engine should be ready after precompute"
    assert engine._n == 500, f"Expected 500 rows, got {engine._n}"

    # 测试 get_abc
    a, b, c = engine.get_abc(100)
    print(f"    get_abc(100): A={a:.4f}, B={b:.4f}, C={c:.4f}")

    # 测试 get_raw
    raw = engine.get_raw(100)
    assert raw['a'].shape == (N_A,), f"Expected ({N_A},), got {raw['a'].shape}"
    assert raw['b'].shape == (N_B,), f"Expected ({N_B},), got {raw['b'].shape}"
    assert raw['c'].shape == (N_C,), f"Expected ({N_C},), got {raw['c'].shape}"
    print(f"    get_raw(100): a={raw['a'].shape}, b={raw['b'].shape}, c={raw['c'].shape}")

    # 测试 get_raw_matrix
    matrix = engine.get_raw_matrix(100, 160)
    assert matrix.shape == (60, 32), f"Expected (60, 32), got {matrix.shape}"
    print(f"    get_raw_matrix(100, 160): shape={matrix.shape}")

    # 边界测试
    matrix_boundary = engine.get_raw_matrix(0, 10)
    assert matrix_boundary.shape == (10, 32)
    print(f"    get_raw_matrix(0, 10): shape={matrix_boundary.shape}")

    print("    [PASS] FeatureVectorEngine")
    return True


def test_trajectory_template():
    """测试轨迹模板"""
    print("\n[Test 2] TrajectoryTemplate...")

    from core.trajectory_engine import TrajectoryTemplate

    pre_entry = np.random.randn(60, 32)
    holding = np.random.randn(120, 32)
    pre_exit = np.random.randn(30, 32)

    template = TrajectoryTemplate(
        trade_idx=0,
        regime="强多头",
        direction="LONG",
        profit_pct=2.5,
        pre_entry=pre_entry,
        holding=holding,
        pre_exit=pre_exit,
        entry_idx=100,
        exit_idx=220,
    )

    assert template.pre_entry.shape == (60, 32)
    assert template.holding.shape == (120, 32)
    assert template.pre_exit.shape == (30, 32)
    assert template.pre_entry_flat.shape == (60 * 32,)
    print(f"    模板创建: pre_entry_flat.shape={template.pre_entry_flat.shape}")

    print("    [PASS] TrajectoryTemplate")
    return True


def test_trajectory_matcher():
    """测试轨迹匹配器"""
    print("\n[Test 3] TrajectoryMatcher...")

    from core.trajectory_matcher import TrajectoryMatcher
    from core.trajectory_engine import TrajectoryTemplate

    matcher = TrajectoryMatcher(cosine_top_k=5, dtw_radius=5)

    # 创建模拟模板
    np.random.seed(42)
    templates = []
    for i in range(10):
        pre_entry = np.random.randn(60, 32)
        holding = np.random.randn(50, 32)
        pre_exit = np.random.randn(30, 32)
        templates.append(TrajectoryTemplate(
            trade_idx=i,
            regime="测试",
            direction="LONG",
            profit_pct=1.0 + i * 0.5,
            pre_entry=pre_entry,
            holding=holding,
            pre_exit=pre_exit,
        ))

    # 创建查询（与第一个模板相似）
    query = templates[0].pre_entry + np.random.randn(60, 32) * 0.1

    result = matcher.match_entry(
        query, templates,
        cosine_threshold=0.3,
        dtw_threshold=0.8
    )

    print(f"    匹配结果: matched={result.matched}, "
          f"cosine={result.cosine_sim:.3f}, "
          f"dtw_sim={result.dtw_similarity:.3f}")
    print(f"    top_k_indices: {result.top_k_indices[:3]}...")

    # 持仓监控测试
    holding_so_far = np.random.randn(30, 32)
    divergence, should_exit = matcher.monitor_holding(
        holding_so_far, templates[0], 0.7
    )
    print(f"    持仓监控: divergence={divergence:.3f}, should_exit={should_exit}")

    print("    [PASS] TrajectoryMatcher")
    return True


def test_trading_params():
    """测试交易参数"""
    print("\n[Test 4] TradingParams...")

    from core.ga_trading_optimizer import TradingParams

    # 随机生成
    params = TradingParams.random()
    print(f"    随机参数: cosine_th={params.cosine_threshold:.3f}, "
          f"dtw_th={params.dtw_threshold:.3f}")

    # 转换为数组
    arr = params.to_array()
    assert arr.shape == (8,), f"Expected (8,), got {arr.shape}"

    # 从数组还原
    restored = TradingParams.from_array(arr)
    assert restored.cosine_threshold == params.cosine_threshold

    # 裁剪测试
    out_of_range = TradingParams(cosine_threshold=1.5, dtw_threshold=-0.5)
    clipped = out_of_range.clip()
    assert 0.3 <= clipped.cosine_threshold <= 0.9
    assert 0.1 <= clipped.dtw_threshold <= 0.8
    print(f"    裁剪测试: {out_of_range.cosine_threshold} -> {clipped.cosine_threshold}")

    print("    [PASS] TradingParams")
    return True


def test_walk_forward_split():
    """测试 Walk-Forward 数据分割"""
    print("\n[Test 5] WalkForward 数据分割...")

    from core.walk_forward import WalkForwardValidator

    validator = WalkForwardValidator(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        step_size=1000,
    )

    # 计算 10000 行数据的分割
    ranges = validator._compute_fold_ranges(10000, n_folds=3)

    print(f"    折数: {len(ranges)}")
    for i, (train, val, test) in enumerate(ranges):
        print(f"    Fold {i}: 训练{train}, 验证{val}, 测试{test}")

    # 验证范围不重叠
    for train, val, test in ranges:
        assert train[1] == val[0], "Train end should equal Val start"
        assert val[1] == test[0], "Val end should equal Test start"

    print("    [PASS] WalkForward 数据分割")
    return True


def test_cosine_similarity():
    """测试余弦相似度"""
    print("\n[Test 6] 余弦相似度...")

    from core.trajectory_matcher import TrajectoryMatcher

    # 相同向量应该有相似度 1.0
    v1 = np.array([1, 2, 3, 4, 5])
    sim = TrajectoryMatcher._cosine_similarity(v1, v1)
    assert abs(sim - 1.0) < 1e-6, f"Same vector should have sim=1.0, got {sim}"
    print(f"    相同向量: sim={sim:.6f}")

    # 正交向量应该有相似度 0
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    sim = TrajectoryMatcher._cosine_similarity(v2, v3)
    assert abs(sim) < 1e-6, f"Orthogonal vectors should have sim=0, got {sim}"
    print(f"    正交向量: sim={sim:.6f}")

    # 相反向量应该有相似度 -1.0
    v4 = np.array([1, 2, 3])
    v5 = np.array([-1, -2, -3])
    sim = TrajectoryMatcher._cosine_similarity(v4, v5)
    assert abs(sim + 1.0) < 1e-6, f"Opposite vectors should have sim=-1.0, got {sim}"
    print(f"    相反向量: sim={sim:.6f}")

    print("    [PASS] 余弦相似度")
    return True


def test_dtw():
    """测试 DTW 距离"""
    print("\n[Test 7] DTW 距离...")

    from core.trajectory_matcher import TrajectoryMatcher

    matcher = TrajectoryMatcher(dtw_radius=5)

    # 相同序列应该有距离 0
    seq1 = np.random.randn(20, 5)
    dist = matcher._compute_dtw_distance(seq1, seq1)
    assert dist < 0.01, f"Same sequence should have dist~0, got {dist}"
    print(f"    相同序列: dist={dist:.6f}")

    # 不同序列应该有非零距离
    seq2 = np.random.randn(20, 5)
    dist = matcher._compute_dtw_distance(seq1, seq2)
    print(f"    随机序列: dist={dist:.6f}")

    # 不同长度
    seq3 = np.random.randn(15, 5)
    dist = matcher._compute_dtw_distance(seq1, seq3)
    print(f"    不同长度: dist={dist:.6f}")

    print("    [PASS] DTW 距离")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("R3000 轨迹匹配核心模块 - 单元测试")
    print("=" * 60)

    tests = [
        test_feature_vector_engine,
        test_trajectory_template,
        test_trajectory_matcher,
        test_trading_params,
        test_walk_forward_split,
        test_cosine_similarity,
        test_dtw,
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
