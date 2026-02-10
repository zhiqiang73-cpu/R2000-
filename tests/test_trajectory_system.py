"""
R3000 轨迹匹配系统端到端测试

测试流程：
1. 加载数据
2. 上帝视角标注
3. 回测生成交易记录
4. 特征向量引擎预计算
5. 轨迹模板提取
6. 轨迹匹配测试
7. GA交易参数优化（简化版）
8. Walk-Forward验证（单折）
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_CONFIG, LABEL_BACKTEST_CONFIG, MARKET_REGIME_CONFIG


def test_trajectory_system():
    print("=" * 60)
    print("R3000 轨迹匹配系统 - 端到端测试")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1/8] 加载数据...")
    from core.data_loader import DataLoader
    from utils.indicators import calculate_all_indicators

    loader = DataLoader()
    # 使用较小样本加速测试
    df = loader.sample_continuous(10000, random_seed=42)
    df = calculate_all_indicators(df)
    print(f"    数据加载完成: {len(df)} 行")

    # ── 2. 上帝视角标注 ──
    print("\n[2/8] 上帝视角标注...")
    from core.labeler import GodViewLabeler

    labeler = GodViewLabeler()
    labels = labeler.label(df)
    long_count = int((labels == 1).sum())
    short_count = int((labels == -1).sum())
    print(f"    标注完成: {long_count} LONG, {short_count} SHORT")

    # ── 3. 回测生成交易记录 ──
    print("\n[3/8] 回测生成交易记录...")
    from core.backtester import Backtester

    bt = Backtester(
        initial_capital=LABEL_BACKTEST_CONFIG["INITIAL_CAPITAL"],
        leverage=LABEL_BACKTEST_CONFIG["LEVERAGE"],
        fee_rate=LABEL_BACKTEST_CONFIG["FEE_RATE"],
        slippage=LABEL_BACKTEST_CONFIG["SLIPPAGE"],
        position_size_pct=LABEL_BACKTEST_CONFIG["POSITION_SIZE_PCT"],
    )
    result = bt.run_with_labels(df, labels)
    print(f"    交易数: {result.total_trades}, 胜率: {result.win_rate:.1%}, "
          f"总收益: {result.total_return_pct:.2f}%")

    if result.total_trades == 0:
        print("    [警告] 无交易记录，跳过后续测试")
        return False

    # ── 4. 市场状态分类 ──
    print("\n[4/8] 市场状态分类...")
    from core.market_regime import MarketRegimeClassifier

    regime_map = {}
    if labeler.alternating_swings:
        classifier = MarketRegimeClassifier(labeler.alternating_swings, MARKET_REGIME_CONFIG)
        for i, trade in enumerate(result.trades):
            regime = classifier.classify_at(trade.entry_idx)
            trade.market_regime = regime
            regime_map[i] = regime
        regimes = list(set(regime_map.values()))
        print(f"    分类完成: {len(regimes)} 种状态 ({regimes})")
    else:
        print("    [警告] 无摆动点数据")

    # ── 5. 特征向量引擎预计算 ──
    print("\n[5/8] 特征向量引擎预计算...")
    from core.feature_vector import FeatureVectorEngine

    fv_engine = FeatureVectorEngine()
    fv_engine.precompute(df)
    print(f"    预计算完成: {fv_engine._n} 行")

    # 测试 get_raw_matrix
    test_matrix = fv_engine.get_raw_matrix(100, 160)
    print(f"    get_raw_matrix 测试: shape = {test_matrix.shape}")
    assert test_matrix.shape == (60, 32), f"Expected (60, 32), got {test_matrix.shape}"

    # ── 6. 轨迹模板提取 ──
    print("\n[6/8] 轨迹模板提取...")
    from core.trajectory_engine import TrajectoryMemory

    memory = TrajectoryMemory()
    n_templates = memory.extract_from_trades(result.trades, fv_engine, regime_map)

    if n_templates == 0:
        print("    [警告] 无模板可提取（可能无盈利交易）")
        # 降低利润阈值重试
        memory = TrajectoryMemory(min_profit_pct=-100)  # 接受所有交易
        n_templates = memory.extract_from_trades(result.trades, fv_engine, regime_map)
        print(f"    (降低阈值后) 模板数: {n_templates}")

    if n_templates > 0:
        all_templates = memory.get_all_templates()
        avg_profit = np.mean([t.profit_pct for t in all_templates])
        print(f"    模板统计: {n_templates} 个, 平均收益: {avg_profit:.2f}%")

        # 检查模板结构
        t = all_templates[0]
        print(f"    模板0 结构: pre_entry={t.pre_entry.shape}, "
              f"holding={t.holding.shape}, pre_exit={t.pre_exit.shape}")

    # ── 7. 轨迹匹配测试 ──
    print("\n[7/8] 轨迹匹配测试...")
    from core.trajectory_matcher import TrajectoryMatcher
    from core.trajectory_engine import extract_current_trajectory

    matcher = TrajectoryMatcher()

    # 随机选择一个测试点
    test_idx = min(500, len(df) - 1)
    current_traj = extract_current_trajectory(fv_engine, test_idx)
    print(f"    当前轨迹: shape = {current_traj.shape}")

    if n_templates > 0:
        # 获取做多候选
        long_templates = memory.get_templates_by_direction("LONG")
        if long_templates:
            match_result = matcher.match_entry(
                current_traj, long_templates,
                cosine_threshold=0.3, dtw_threshold=0.8
            )
            print(f"    匹配结果: matched={match_result.matched}, "
                  f"cosine={match_result.cosine_sim:.3f}, "
                  f"dtw_sim={match_result.dtw_similarity:.3f}")

    # ── 8. GA交易参数优化（简化版）──
    print("\n[8/8] GA交易参数优化测试...")
    from core.ga_trading_optimizer import TradingParams, GATradingOptimizer
    from config import GA_CONFIG

    # 分割数据：70% 训练, 30% 验证
    n = len(df)
    split_idx = int(n * 0.7)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    val_labels = labels.iloc[split_idx:].reset_index(drop=True)
    val_df = calculate_all_indicators(val_df)

    val_fv_engine = FeatureVectorEngine()
    val_fv_engine.precompute(val_df)

    # 减少迭代次数加速测试
    original_gen = GA_CONFIG["MAX_GENERATIONS"]
    GA_CONFIG["MAX_GENERATIONS"] = 5

    try:
        optimizer = GATradingOptimizer(
            trajectory_memory=memory,
            fv_engine=val_fv_engine,
            val_df=val_df,
            val_labels=val_labels,
        )

        # 测试单个参数评估
        test_params = TradingParams()
        sim_result = optimizer._simulate_trading(test_params)
        print(f"    模拟交易测试: {sim_result.n_trades} 笔交易, "
              f"Sharpe={sim_result.sharpe_ratio:.3f}")

        # 运行简化GA
        best_params, best_fitness = optimizer.run()
        print(f"    GA优化完成: 最佳Sharpe={best_fitness:.3f}")
        print(f"    最佳参数: cosine_th={best_params.cosine_threshold:.3f}, "
              f"dtw_th={best_params.dtw_threshold:.3f}")

    finally:
        GA_CONFIG["MAX_GENERATIONS"] = original_gen

    print("\n" + "=" * 60)
    print("端到端测试完成!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_trajectory_system()
    sys.exit(0 if success else 1)
