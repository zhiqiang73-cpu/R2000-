"""
R3000 GA交易参数优化器
直接优化交易参数（而非权重），适应度 = 验证集模拟交易的Sharpe Ratio

GA搜索空间：
  - cosine_threshold: 余弦粗筛阈值 [0.3, 0.9]
  - dtw_threshold: DTW入场阈值 [0.1, 0.8]
  - hold_divergence_limit: 持仓偏离上限 [0.3, 0.9]
  - exit_match_threshold: 离场匹配阈值 [0.2, 0.8]
  - stop_loss_atr: 止损ATR倍数 [1.0, 5.0]
  - take_profit_atr: 止盈ATR倍数 [1.5, 8.0]
  - max_hold_bars: 最大持仓K线数 [60, 480]
  - min_templates_agree: 最少几个模板一致 [1, 5]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GA_CONFIG, TRAJECTORY_CONFIG


@dataclass
class TradingParams:
    """交易参数（GA搜索空间）"""
    cosine_threshold: float = 0.6       # 余弦粗筛阈值 [0.3, 0.9]
    dtw_threshold: float = 0.5          # DTW入场阈值 [0.1, 0.8]
    hold_divergence_limit: float = 0.7  # 持仓偏离上限 [0.3, 0.9]
    exit_match_threshold: float = 0.5   # 离场匹配阈值 [0.2, 0.8]
    stop_loss_atr: float = 2.0          # 止损ATR倍数 [1.0, 5.0]
    take_profit_atr: float = 3.0        # 止盈ATR倍数 [1.5, 8.0]
    max_hold_bars: int = 240            # 最大持仓K线数 [60, 480]
    min_templates_agree: int = 1        # 最少几个模板一致 [1, 5]

    # 参数范围
    RANGES = {
        "cosine_threshold": (0.3, 0.9),
        "dtw_threshold": (0.1, 0.8),
        "hold_divergence_limit": (0.3, 0.9),
        "exit_match_threshold": (0.2, 0.8),
        "stop_loss_atr": (1.0, 5.0),
        "take_profit_atr": (1.5, 8.0),
        "max_hold_bars": (60, 480),
        "min_templates_agree": (1, 5),
    }

    def to_array(self) -> np.ndarray:
        """转为数组（用于GA）"""
        return np.array([
            self.cosine_threshold,
            self.dtw_threshold,
            self.hold_divergence_limit,
            self.exit_match_threshold,
            self.stop_loss_atr,
            self.take_profit_atr,
            float(self.max_hold_bars),
            float(self.min_templates_agree),
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TradingParams':
        """从数组还原"""
        return cls(
            cosine_threshold=float(arr[0]),
            dtw_threshold=float(arr[1]),
            hold_divergence_limit=float(arr[2]),
            exit_match_threshold=float(arr[3]),
            stop_loss_atr=float(arr[4]),
            take_profit_atr=float(arr[5]),
            max_hold_bars=int(round(arr[6])),
            min_templates_agree=max(1, int(round(arr[7]))),
        )

    @classmethod
    def random(cls) -> 'TradingParams':
        """随机生成参数"""
        return cls(
            cosine_threshold=np.random.uniform(*cls.RANGES["cosine_threshold"]),
            dtw_threshold=np.random.uniform(*cls.RANGES["dtw_threshold"]),
            hold_divergence_limit=np.random.uniform(*cls.RANGES["hold_divergence_limit"]),
            exit_match_threshold=np.random.uniform(*cls.RANGES["exit_match_threshold"]),
            stop_loss_atr=np.random.uniform(*cls.RANGES["stop_loss_atr"]),
            take_profit_atr=np.random.uniform(*cls.RANGES["take_profit_atr"]),
            max_hold_bars=int(np.random.uniform(*cls.RANGES["max_hold_bars"])),
            min_templates_agree=int(np.random.uniform(*cls.RANGES["min_templates_agree"])),
        )

    def clip(self) -> 'TradingParams':
        """裁剪到有效范围"""
        return TradingParams(
            cosine_threshold=np.clip(self.cosine_threshold, *self.RANGES["cosine_threshold"]),
            dtw_threshold=np.clip(self.dtw_threshold, *self.RANGES["dtw_threshold"]),
            hold_divergence_limit=np.clip(self.hold_divergence_limit, *self.RANGES["hold_divergence_limit"]),
            exit_match_threshold=np.clip(self.exit_match_threshold, *self.RANGES["exit_match_threshold"]),
            stop_loss_atr=np.clip(self.stop_loss_atr, *self.RANGES["stop_loss_atr"]),
            take_profit_atr=np.clip(self.take_profit_atr, *self.RANGES["take_profit_atr"]),
            max_hold_bars=int(np.clip(self.max_hold_bars, *self.RANGES["max_hold_bars"])),
            min_templates_agree=int(np.clip(self.min_templates_agree, *self.RANGES["min_templates_agree"])),
        )


@dataclass
class GAIndividual:
    """GA个体"""
    params: TradingParams
    fitness: float = -np.inf

    @classmethod
    def random(cls) -> 'GAIndividual':
        return cls(params=TradingParams.random())


@dataclass
class SimulatedTradeResult:
    """模拟交易结果"""
    n_trades: int = 0
    n_wins: int = 0
    total_profit: float = 0.0
    profits: List[float] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


class GATradingOptimizer:
    """
    GA交易参数优化器

    使用方式：
        optimizer = GATradingOptimizer(
            trajectory_memory=memory,
            fv_engine=engine,
            val_df=val_df,
            val_labels=val_labels,
        )
        best_params, best_fitness = optimizer.run()
    """

    def __init__(self,
                 trajectory_memory,
                 fv_engine,
                 val_df,
                 val_labels,
                 regime_classifier=None,
                 callback: Callable = None):
        """
        Args:
            trajectory_memory: TrajectoryMemory 实例（训练集模板）
            fv_engine: FeatureVectorEngine 实例（需要在验证集上重新预计算）
            val_df: 验证集 DataFrame
            val_labels: 验证集标签
            regime_classifier: 市场状态分类器（可选）
            callback: 回调函数 fn(generation, best_fitness, best_params)
        """
        self.memory = trajectory_memory
        self.fv_engine = fv_engine
        self.val_df = val_df
        self.val_labels = val_labels
        self.regime_classifier = regime_classifier
        self.callback = callback

        # GA参数
        self.pop_size = GA_CONFIG["POPULATION_SIZE"]
        self.elite_ratio = GA_CONFIG["ELITE_RATIO"]
        self.mutation_rate = GA_CONFIG["MUTATION_RATE"]
        self.crossover_rate = GA_CONFIG["CROSSOVER_RATE"]
        self.max_gen = GA_CONFIG["MAX_GENERATIONS"]
        self.early_stop = GA_CONFIG["EARLY_STOP_GENERATIONS"]
        self.n_elite = max(1, int(self.pop_size * self.elite_ratio))

        # 预计算验证集特征
        self._prepare_validation_data()

    def _prepare_validation_data(self):
        """预计算验证集数据"""
        # 确保验证集特征已计算
        if not self.fv_engine.is_ready:
            from utils.indicators import calculate_all_indicators
            self.val_df = calculate_all_indicators(self.val_df)
            self.fv_engine.precompute(self.val_df)

        # ATR用于止盈止损计算
        self.val_atr = self.val_df['atr'].values if 'atr' in self.val_df.columns else np.ones(len(self.val_df))
        self.val_close = self.val_df['close'].values

    def run(self) -> Tuple[TradingParams, float]:
        """
        运行GA优化

        Returns:
            (best_params, best_fitness)
        """
        # 初始化种群
        population = [GAIndividual.random() for _ in range(self.pop_size)]

        # 加入默认参数作为种子
        population[0] = GAIndividual(params=TradingParams())

        best_ever = GAIndividual(params=TradingParams(), fitness=-np.inf)
        no_improve = 0

        for gen in range(self.max_gen):
            # 评估适应度
            for ind in population:
                ind.fitness = self._evaluate(ind.params)

            # 排序（降序）
            population.sort(key=lambda x: x.fitness, reverse=True)

            if population[0].fitness > best_ever.fitness:
                best_ever = GAIndividual(
                    params=population[0].params,
                    fitness=population[0].fitness,
                )
                no_improve = 0
            else:
                no_improve += 1

            if self.callback:
                self.callback(gen, best_ever.fitness, best_ever.params)

            if no_improve >= self.early_stop:
                print(f"[GA-Trading] 早停: 代数={gen}, Sharpe={best_ever.fitness:.4f}")
                break

            if gen % 10 == 0:
                avg_fit = np.mean([ind.fitness for ind in population])
                print(f"[GA-Trading] Gen {gen}: best_sharpe={best_ever.fitness:.4f}, "
                      f"avg={avg_fit:.4f}")

            # 精英保留
            new_pop = [GAIndividual(params=population[i].params) for i in range(self.n_elite)]

            # 生成剩余个体
            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament_select(population)

                if np.random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = GAIndividual(params=p1.params)

                self._mutate(child)
                new_pop.append(child)

            population = new_pop

        return best_ever.params, best_ever.fitness

    def _evaluate(self, params: TradingParams) -> float:
        """
        评估适应度 = 在验证集上模拟交易的Sharpe Ratio
        """
        result = self._simulate_trading(params)

        # Sharpe Ratio 作为适应度
        # 如果交易数太少，惩罚
        if result.n_trades < 5:
            return -10.0

        return result.sharpe_ratio

    def _simulate_trading(self, params: TradingParams) -> SimulatedTradeResult:
        """
        在验证集上模拟交易

        简化逻辑：
          1. 遍历每根K线
          2. 如果未持仓，检查是否有入场信号（轨迹匹配）
          3. 如果持仓，检查止盈止损或离场信号
        """
        from core.trajectory_matcher import TrajectoryMatcher
        from core.trajectory_engine import extract_current_trajectory

        matcher = TrajectoryMatcher()
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]

        n = len(self.val_df)
        profits = []
        position = None  # (entry_idx, entry_price, side, matched_template)

        for i in range(pre_window, n - 1):
            if position is None:
                # 检查入场信号
                current_traj = self.fv_engine.get_raw_matrix(i - pre_window, i)
                if current_traj.shape[0] < pre_window:
                    continue

                # 获取当前市场状态
                regime = "未知"
                if self.regime_classifier:
                    regime = self.regime_classifier.classify_at(i)

                # 尝试匹配做多
                long_candidates = self.memory.get_candidates(regime, "LONG")
                if not long_candidates:
                    long_candidates = self.memory.get_templates_by_direction("LONG")

                long_result = matcher.match_entry(
                    current_traj, long_candidates,
                    params.cosine_threshold, params.dtw_threshold
                )

                # 尝试匹配做空
                short_candidates = self.memory.get_candidates(regime, "SHORT")
                if not short_candidates:
                    short_candidates = self.memory.get_templates_by_direction("SHORT")

                short_result = matcher.match_entry(
                    current_traj, short_candidates,
                    params.cosine_threshold, params.dtw_threshold
                )

                # 选择更好的匹配
                if long_result.matched and short_result.matched:
                    if long_result.dtw_similarity > short_result.dtw_similarity:
                        side = 1
                        matched = long_result.best_template
                    else:
                        side = -1
                        matched = short_result.best_template
                elif long_result.matched:
                    side = 1
                    matched = long_result.best_template
                elif short_result.matched:
                    side = -1
                    matched = short_result.best_template
                else:
                    continue

                # 开仓
                entry_price = self.val_close[i]
                atr = self.val_atr[i]
                position = {
                    "entry_idx": i,
                    "entry_price": entry_price,
                    "side": side,
                    "template": matched,
                    "stop_loss": entry_price - side * params.stop_loss_atr * atr,
                    "take_profit": entry_price + side * params.take_profit_atr * atr,
                }

            else:
                # 持仓中，检查离场
                current_price = self.val_close[i]
                side = position["side"]
                hold_bars = i - position["entry_idx"]

                # 止盈止损检查
                if side == 1:  # 多头
                    if current_price >= position["take_profit"]:
                        profit_pct = (current_price / position["entry_price"] - 1) * 100
                        profits.append(profit_pct)
                        position = None
                        continue
                    elif current_price <= position["stop_loss"]:
                        profit_pct = (current_price / position["entry_price"] - 1) * 100
                        profits.append(profit_pct)
                        position = None
                        continue
                else:  # 空头
                    if current_price <= position["take_profit"]:
                        profit_pct = (1 - current_price / position["entry_price"]) * 100
                        profits.append(profit_pct)
                        position = None
                        continue
                    elif current_price >= position["stop_loss"]:
                        profit_pct = (1 - current_price / position["entry_price"]) * 100
                        profits.append(profit_pct)
                        position = None
                        continue

                # 最大持仓时间
                if hold_bars >= params.max_hold_bars:
                    if side == 1:
                        profit_pct = (current_price / position["entry_price"] - 1) * 100
                    else:
                        profit_pct = (1 - current_price / position["entry_price"]) * 100
                    profits.append(profit_pct)
                    position = None
                    continue

                # 持仓偏离检查
                if position["template"] is not None:
                    holding_traj = self.fv_engine.get_raw_matrix(
                        position["entry_idx"], i + 1
                    )
                    divergence, should_exit = matcher.monitor_holding(
                        holding_traj, position["template"],
                        params.hold_divergence_limit
                    )
                    if should_exit:
                        if side == 1:
                            profit_pct = (current_price / position["entry_price"] - 1) * 100
                        else:
                            profit_pct = (1 - current_price / position["entry_price"]) * 100
                        profits.append(profit_pct)
                        position = None
                        continue

        # 计算统计
        result = SimulatedTradeResult()
        result.n_trades = len(profits)
        if result.n_trades == 0:
            return result

        profits_arr = np.array(profits)
        result.profits = profits
        result.n_wins = int((profits_arr > 0).sum())
        result.total_profit = float(profits_arr.sum())
        result.win_rate = result.n_wins / result.n_trades

        # Sharpe Ratio（假设无风险收益为0）
        mean_return = profits_arr.mean()
        std_return = profits_arr.std()
        if std_return > 1e-9:
            result.sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 年化
        else:
            result.sharpe_ratio = mean_return * 10 if mean_return > 0 else -10

        # 最大回撤
        cumsum = profits_arr.cumsum()
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        result.max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0

        return result

    def _tournament_select(self, population: List[GAIndividual],
                           k: int = 3) -> Tuple[GAIndividual, GAIndividual]:
        """锦标赛选择"""
        def pick():
            candidates = np.random.choice(len(population), size=k, replace=False)
            best = max(candidates, key=lambda i: population[i].fitness)
            return population[best]
        return pick(), pick()

    def _crossover(self, p1: GAIndividual, p2: GAIndividual) -> GAIndividual:
        """均匀交叉"""
        arr1 = p1.params.to_array()
        arr2 = p2.params.to_array()
        mask = np.random.random(len(arr1)) < 0.5
        child_arr = np.where(mask, arr1, arr2)
        child_params = TradingParams.from_array(child_arr).clip()
        return GAIndividual(params=child_params)

    def _mutate(self, ind: GAIndividual):
        """高斯变异"""
        arr = ind.params.to_array()
        for i in range(len(arr)):
            if np.random.random() < self.mutation_rate:
                # 根据参数范围确定变异强度
                param_name = list(TradingParams.RANGES.keys())[i]
                low, high = TradingParams.RANGES[param_name]
                std = (high - low) * 0.1
                arr[i] += np.random.randn() * std
        ind.params = TradingParams.from_array(arr).clip()
