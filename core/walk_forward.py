"""
R3000 Walk-Forward 时序验证框架
核心原则：时序数据绝对不能随机打乱

验证流程：
  1. 按时间顺序分割：训练60% → 验证20% → 测试20%
  2. 训练集：提取轨迹模板
  3. 验证集：GA优化交易参数
  4. 测试集：评估最终性能（未参与任何拟合）
  5. 滑动窗口重复多次，取平均
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WALK_FORWARD_CONFIG, TRAJECTORY_CONFIG, LABEL_BACKTEST_CONFIG, GA_CONFIG


@dataclass
class FoldResult:
    """单折验证结果"""
    fold_idx: int
    train_range: Tuple[int, int]    # (start_idx, end_idx)
    val_range: Tuple[int, int]
    test_range: Tuple[int, int]

    # 训练集统计
    n_train_templates: int = 0

    # 验证集GA结果
    val_sharpe: float = 0.0
    val_n_trades: int = 0
    val_win_rate: float = 0.0
    best_params: Optional[object] = None

    # 测试集最终评估（最重要）
    test_sharpe: float = 0.0
    test_n_trades: int = 0
    test_win_rate: float = 0.0
    test_total_profit: float = 0.0
    test_max_drawdown: float = 0.0


@dataclass
class WalkForwardResult:
    """Walk-Forward 完整结果"""
    folds: List[FoldResult] = field(default_factory=list)

    # 汇总统计
    avg_test_sharpe: float = 0.0
    avg_test_win_rate: float = 0.0
    avg_test_profit: float = 0.0
    std_test_sharpe: float = 0.0

    # 稳定性指标
    n_profitable_folds: int = 0     # 多少折测试集盈利
    consistency_ratio: float = 0.0  # 盈利折数 / 总折数

    def summarize(self):
        """计算汇总统计"""
        if not self.folds:
            return

        test_sharpes = [f.test_sharpe for f in self.folds]
        test_profits = [f.test_total_profit for f in self.folds]
        test_win_rates = [f.test_win_rate for f in self.folds]

        self.avg_test_sharpe = float(np.mean(test_sharpes))
        self.std_test_sharpe = float(np.std(test_sharpes))
        self.avg_test_win_rate = float(np.mean(test_win_rates))
        self.avg_test_profit = float(np.mean(test_profits))

        self.n_profitable_folds = sum(1 for p in test_profits if p > 0)
        self.consistency_ratio = self.n_profitable_folds / len(self.folds)

    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("Walk-Forward 验证结果")
        print("=" * 60)

        for fold in self.folds:
            print(f"\nFold {fold.fold_idx + 1}:")
            print(f"  训练集: {fold.train_range[0]} ~ {fold.train_range[1]} "
                  f"({fold.n_train_templates} 模板)")
            print(f"  验证集: Sharpe={fold.val_sharpe:.3f}, "
                  f"交易={fold.val_n_trades}, 胜率={fold.val_win_rate:.1%}")
            print(f"  测试集: Sharpe={fold.test_sharpe:.3f}, "
                  f"交易={fold.test_n_trades}, 胜率={fold.test_win_rate:.1%}, "
                  f"利润={fold.test_total_profit:.2f}%")

        print(f"\n{'=' * 60}")
        print(f"汇总 ({len(self.folds)} 折):")
        print(f"  平均测试Sharpe: {self.avg_test_sharpe:.3f} ± {self.std_test_sharpe:.3f}")
        print(f"  平均测试胜率: {self.avg_test_win_rate:.1%}")
        print(f"  平均测试利润: {self.avg_test_profit:.2f}%")
        print(f"  盈利折数: {self.n_profitable_folds}/{len(self.folds)} "
              f"(一致性: {self.consistency_ratio:.1%})")
        print("=" * 60)


class WalkForwardValidator:
    """
    Walk-Forward 验证器

    使用方式：
        validator = WalkForwardValidator()
        result = validator.run(df, labels, n_folds=3)
        result.print_summary()
    """

    def __init__(self,
                 train_ratio: float = None,
                 val_ratio: float = None,
                 test_ratio: float = None,
                 step_size: int = None,
                 min_train_trades: int = None,
                 min_val_trades: int = None):
        self.train_ratio = train_ratio or WALK_FORWARD_CONFIG["TRAIN_RATIO"]
        self.val_ratio = val_ratio or WALK_FORWARD_CONFIG["VAL_RATIO"]
        self.test_ratio = test_ratio or WALK_FORWARD_CONFIG["TEST_RATIO"]
        self.step_size = step_size or WALK_FORWARD_CONFIG["STEP_SIZE"]
        self.min_train_trades = min_train_trades or WALK_FORWARD_CONFIG["MIN_TRAIN_TRADES"]
        self.min_val_trades = min_val_trades or WALK_FORWARD_CONFIG["MIN_VAL_TRADES"]

    def run(self, df: pd.DataFrame, labels: pd.Series,
            n_folds: int = None, callback=None) -> WalkForwardResult:
        """
        运行 Walk-Forward 验证

        Args:
            df: 完整数据集 DataFrame
            labels: 标注序列
            n_folds: 折数
            callback: 进度回调 fn(fold_idx, stage, message)

        Returns:
            WalkForwardResult 完整结果
        """
        n_folds = n_folds or WALK_FORWARD_CONFIG["N_FOLDS"]
        n = len(df)
        result = WalkForwardResult()

        # 计算每折的数据范围
        fold_ranges = self._compute_fold_ranges(n, n_folds)

        for fold_idx, (train_range, val_range, test_range) in enumerate(fold_ranges):
            if callback:
                callback(fold_idx, "start", f"Fold {fold_idx + 1}/{n_folds}")

            fold_result = self._run_single_fold(
                df, labels, fold_idx, train_range, val_range, test_range, callback
            )
            result.folds.append(fold_result)

        result.summarize()
        return result

    def _compute_fold_ranges(self, n: int, n_folds: int) -> List[Tuple]:
        """
        计算每折的数据范围

        滑动窗口方式：
        Fold 1: [0 ~ train_end | val_start ~ val_end | test_start ~ test_end]
        Fold 2: [step ~ train_end+step | ...]
        """
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        train_size = int(n * self.train_ratio / total_ratio)
        val_size = int(n * self.val_ratio / total_ratio)
        test_size = int(n * self.test_ratio / total_ratio)

        fold_length = train_size + val_size + test_size
        ranges = []

        for i in range(n_folds):
            start = i * self.step_size
            if start + fold_length > n:
                break

            train_start = start
            train_end = start + train_size
            val_start = train_end
            val_end = val_start + val_size
            test_start = val_end
            test_end = min(test_start + test_size, n)

            ranges.append((
                (train_start, train_end),
                (val_start, val_end),
                (test_start, test_end),
            ))

        return ranges

    def _run_single_fold(self, df: pd.DataFrame, labels: pd.Series,
                          fold_idx: int,
                          train_range: Tuple[int, int],
                          val_range: Tuple[int, int],
                          test_range: Tuple[int, int],
                          callback=None) -> FoldResult:
        """运行单折验证"""
        from core.labeler import GodViewLabeler
        from core.backtester import Backtester
        from core.market_regime import MarketRegimeClassifier
        from core.feature_vector import FeatureVectorEngine
        from core.trajectory_engine import TrajectoryMemory
        from core.ga_trading_optimizer import GATradingOptimizer, TradingParams
        from utils.indicators import calculate_all_indicators
        from config import MARKET_REGIME_CONFIG

        fold_result = FoldResult(
            fold_idx=fold_idx,
            train_range=train_range,
            val_range=val_range,
            test_range=test_range,
        )

        # ── 1. 准备训练集 ──
        if callback:
            callback(fold_idx, "train", "提取训练集模板...")

        train_df = df.iloc[train_range[0]:train_range[1]].copy().reset_index(drop=True)
        train_labels = labels.iloc[train_range[0]:train_range[1]].reset_index(drop=True)

        # 计算指标
        train_df = calculate_all_indicators(train_df)

        # 上帝视角回测
        labeler = GodViewLabeler()
        labeler.label(train_df)  # 需要重新标注以获取 alternating_swings

        bt = Backtester(
            initial_capital=LABEL_BACKTEST_CONFIG["INITIAL_CAPITAL"],
            leverage=LABEL_BACKTEST_CONFIG["LEVERAGE"],
            fee_rate=LABEL_BACKTEST_CONFIG["FEE_RATE"],
            slippage=LABEL_BACKTEST_CONFIG["SLIPPAGE"],
            position_size_pct=LABEL_BACKTEST_CONFIG["POSITION_SIZE_PCT"],
        )
        train_bt_result = bt.run_with_labels(train_df, train_labels)

        # 市场状态分类
        regime_classifier = None
        regime_map = {}
        if labeler.alternating_swings:
            regime_classifier = MarketRegimeClassifier(
                labeler.alternating_swings, MARKET_REGIME_CONFIG
            )
            for i, trade in enumerate(train_bt_result.trades):
                regime_map[i] = regime_classifier.classify_at(trade.entry_idx)
                trade.market_regime = regime_map[i]

        # 特征向量引擎
        fv_engine = FeatureVectorEngine()
        fv_engine.precompute(train_df)

        # 提取轨迹模板
        memory = TrajectoryMemory()
        n_templates = memory.extract_from_trades(
            train_bt_result.trades, fv_engine, regime_map, verbose=False
        )
        fold_result.n_train_templates = n_templates

        if n_templates < self.min_train_trades:
            print(f"[Fold {fold_idx + 1}] 训练集模板不足 ({n_templates} < {self.min_train_trades})")
            return fold_result

        # ── 2. 验证集 GA 优化 ──
        if callback:
            callback(fold_idx, "val", "GA优化交易参数...")

        val_df = df.iloc[val_range[0]:val_range[1]].copy().reset_index(drop=True)
        val_labels = labels.iloc[val_range[0]:val_range[1]].reset_index(drop=True)
        val_df = calculate_all_indicators(val_df)

        val_fv_engine = FeatureVectorEngine()
        val_fv_engine.precompute(val_df)

        # 减少GA代数以加快速度
        original_max_gen = GA_CONFIG["MAX_GENERATIONS"]
        GA_CONFIG["MAX_GENERATIONS"] = min(30, original_max_gen)

        try:
            optimizer = GATradingOptimizer(
                trajectory_memory=memory,
                fv_engine=val_fv_engine,
                val_df=val_df,
                val_labels=val_labels,
                regime_classifier=None,  # 验证集无法用训练集的分类器
            )
            best_params, best_sharpe = optimizer.run()
        finally:
            GA_CONFIG["MAX_GENERATIONS"] = original_max_gen

        fold_result.best_params = best_params
        fold_result.val_sharpe = best_sharpe

        # 获取验证集详细统计
        val_result = optimizer._simulate_trading(best_params)
        fold_result.val_n_trades = val_result.n_trades
        fold_result.val_win_rate = val_result.win_rate

        # ── 3. 测试集评估 ──
        if callback:
            callback(fold_idx, "test", "测试集最终评估...")

        test_df = df.iloc[test_range[0]:test_range[1]].copy().reset_index(drop=True)
        test_labels = labels.iloc[test_range[0]:test_range[1]].reset_index(drop=True)
        test_df = calculate_all_indicators(test_df)

        test_fv_engine = FeatureVectorEngine()
        test_fv_engine.precompute(test_df)

        # 用验证集优化的参数在测试集上评估
        test_optimizer = GATradingOptimizer(
            trajectory_memory=memory,
            fv_engine=test_fv_engine,
            val_df=test_df,
            val_labels=test_labels,
            regime_classifier=None,
        )
        test_result = test_optimizer._simulate_trading(best_params)

        fold_result.test_sharpe = test_result.sharpe_ratio
        fold_result.test_n_trades = test_result.n_trades
        fold_result.test_win_rate = test_result.win_rate
        fold_result.test_total_profit = test_result.total_profit
        fold_result.test_max_drawdown = test_result.max_drawdown

        return fold_result


def quick_train_test_split(df: pd.DataFrame, labels: pd.Series,
                            train_ratio: float = 0.7) -> Tuple:
    """
    快速训练/测试分割（无验证集，仅用于快速原型）

    Returns:
        (train_df, train_labels, test_df, test_labels)
    """
    n = len(df)
    split_idx = int(n * train_ratio)

    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    train_labels = labels.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    test_labels = labels.iloc[split_idx:].reset_index(drop=True)

    return train_df, train_labels, test_df, test_labels
