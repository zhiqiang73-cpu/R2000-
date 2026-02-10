"""
R3000 遗传算法权重优化器
正向递归：用 GA 优化 ABC 轴的子特征权重
使目标函数最大化 —— 在向量空间中，盈利交易的入场点形成紧凑集群，
不同方向/阶段的集群之间有最大分离度

适应度函数：
  fitness = α * compactness + β * separation + γ * similarity_profit_correlation

  compactness: 同类点云的紧凑度（spread 越小越好）
  separation:  不同方向（LONG vs SHORT）入场集群的距离
  sim_corr:    相似度和交易利润的相关性（高相似度→高利润）
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GA_CONFIG, VECTOR_SPACE_CONFIG
from core.feature_vector import FeatureVectorEngine, AxisWeights, N_A, N_B, N_C
from core.vector_memory import VectorMemory


@dataclass
class GAIndividual:
    """GA个体"""
    weights: np.ndarray  # 展平的权重向量
    fitness: float = -np.inf

    @classmethod
    def random(cls, dim: int) -> 'GAIndividual':
        w_min = VECTOR_SPACE_CONFIG.get("WEIGHT_MIN", -10.0)
        w_max = VECTOR_SPACE_CONFIG.get("WEIGHT_MAX", 10.0)
        w = np.random.uniform(w_min, w_max, dim)
        return cls(weights=w)


class GAWeightOptimizer:
    """
    遗传算法权重优化器

    核心流程（正向递归）：
      1. 初始化种群（随机权重）
      2. 对每个个体：用其权重计算所有交易的 ABC 坐标 → 评估适应度
      3. 选择 → 交叉 → 变异 → 下一代
      4. 重复直至收敛或达到最大代数
      5. 返回最优权重

    正向递归意义：每一代的最优权重都比上一代更能准确区分市场中的
    不同交易模式（通过向量空间的几何结构来实现）。
    """

    def __init__(self, engine: FeatureVectorEngine,
                 trades: list, regime_map: Dict[int, str],
                 callback=None):
        """
        Args:
            engine: 已预计算的特征向量引擎
            trades: 所有 TradeRecord 列表
            regime_map: {trade_idx: regime_string}
            callback: 可选回调 fn(generation, best_fitness, best_weights)
        """
        self.engine = engine
        self.trades = trades
        self.regime_map = regime_map
        self.callback = callback

        # GA parameters
        self.pop_size = GA_CONFIG["POPULATION_SIZE"]
        self.elite_ratio = GA_CONFIG["ELITE_RATIO"]
        self.mutation_rate = GA_CONFIG["MUTATION_RATE"]
        self.mutation_strength = GA_CONFIG["MUTATION_STRENGTH"]
        self.crossover_rate = GA_CONFIG["CROSSOVER_RATE"]
        self.max_gen = GA_CONFIG["MAX_GENERATIONS"]
        self.early_stop = GA_CONFIG["EARLY_STOP_GENERATIONS"]

        # 维度
        self.dim = N_A + N_B + N_C
        self.n_elite = max(1, int(self.pop_size * self.elite_ratio))

        # 预计算原始子特征（不随权重变化）
        self._raw_a = engine._raw_a
        self._raw_b = engine._raw_b
        self._raw_c = engine._raw_c

        # 提取交易的入场/离场索引和侧面
        self._entry_idx = np.array([t.entry_idx for t in trades], dtype=int)
        self._exit_idx = np.array([t.exit_idx for t in trades], dtype=int)
        self._sides = np.array([t.side for t in trades])
        self._profits = np.array([t.profit_pct for t in trades])
        self._regimes = [regime_map.get(i, '未知') for i in range(len(trades))]

    def run(self) -> Tuple[AxisWeights, float]:
        """
        运行GA优化

        Returns:
            (best_weights, best_fitness)
        """
        # 初始化种群
        population = [GAIndividual.random(self.dim) for _ in range(self.pop_size)]

        # 加入默认等权作为种子
        default_w = AxisWeights().to_flat()
        population[0] = GAIndividual(weights=default_w.copy())

        best_ever = GAIndividual(weights=default_w.copy(), fitness=-np.inf)
        no_improve = 0

        for gen in range(self.max_gen):
            # 评估适应度
            for ind in population:
                ind.fitness = self._evaluate(ind.weights)

            # 排序（降序）
            population.sort(key=lambda x: x.fitness, reverse=True)

            if population[0].fitness > best_ever.fitness:
                best_ever = GAIndividual(
                    weights=population[0].weights.copy(),
                    fitness=population[0].fitness,
                )
                no_improve = 0
            else:
                no_improve += 1

            if self.callback:
                self.callback(gen, best_ever.fitness, best_ever.weights)

            if no_improve >= self.early_stop:
                print(f"[GA] 早停: 代数={gen}, 最优适应度={best_ever.fitness:.4f}")
                break

            if gen % 10 == 0:
                avg_fit = np.mean([ind.fitness for ind in population])
                print(f"[GA] Gen {gen}: best={best_ever.fitness:.4f}, "
                      f"avg={avg_fit:.4f}")

            # 精英保留
            new_pop = [GAIndividual(weights=population[i].weights.copy())
                       for i in range(self.n_elite)]

            # 生成剩余个体
            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament_select(population)

                if np.random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = GAIndividual(weights=p1.weights.copy())

                self._mutate(child)
                new_pop.append(child)

            population = new_pop

        return AxisWeights.from_flat(best_ever.weights), best_ever.fitness

    def _evaluate(self, weights: np.ndarray) -> float:
        """
        评估适应度 —— 严格按市场状态分别评估，再加权汇总

        对每个市场状态独立计算：
          1. 紧凑度（同方向入场点集群内散度的倒数）
          2. 分离度（LONG_ENTRY vs SHORT_ENTRY 质心距离）
          3. 利润-距离相关性（距集群中心近的交易是否利润更高）
        最终按各市场状态的交易数量加权平均
        """
        w_a = weights[:N_A]
        w_b = weights[N_A:N_A + N_B]
        w_c = weights[N_A + N_B:]

        # 计算所有交易入场的 ABC 坐标（批量向量化）
        entry_a = self._raw_a[self._entry_idx] @ w_a
        entry_b = self._raw_b[self._entry_idx] @ w_b
        entry_c = self._raw_c[self._entry_idx] @ w_c

        # 按市场状态分组
        unique_regimes = set(self._regimes)
        regime_arr = np.array(self._regimes)

        total_fitness = 0.0
        total_weight = 0.0

        for regime in unique_regimes:
            regime_mask = regime_arr == regime
            n_regime = int(regime_mask.sum())
            if n_regime < 4:  # 样本太少，跳过
                continue

            r_entry_a = entry_a[regime_mask]
            r_entry_b = entry_b[regime_mask]
            r_entry_c = entry_c[regime_mask]
            r_sides = self._sides[regime_mask]
            r_profits = self._profits[regime_mask]

            long_mask = r_sides == 1
            short_mask = r_sides == -1

            # --- 紧凑度（在此市场状态内） ---
            compactness = 0.0
            n_clusters = 0
            for mask in [long_mask, short_mask]:
                if mask.sum() < 2:
                    continue
                pts = np.column_stack([r_entry_a[mask], r_entry_b[mask], r_entry_c[mask]])
                centroid = pts.mean(axis=0)
                dists = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
                spread = dists.mean()
                compactness += 1.0 / (spread + 0.01)
                n_clusters += 1
            if n_clusters > 0:
                compactness /= n_clusters

            # --- 分离度（在此市场状态内，LONG vs SHORT） ---
            separation = 0.0
            if long_mask.sum() >= 2 and short_mask.sum() >= 2:
                long_centroid = np.array([
                    r_entry_a[long_mask].mean(),
                    r_entry_b[long_mask].mean(),
                    r_entry_c[long_mask].mean(),
                ])
                short_centroid = np.array([
                    r_entry_a[short_mask].mean(),
                    r_entry_b[short_mask].mean(),
                    r_entry_c[short_mask].mean(),
                ])
                separation = np.sqrt(np.sum((long_centroid - short_centroid) ** 2))

            # --- 利润-距离相关性（在此市场状态内） ---
            sim_corr = 0.0
            if n_regime >= 5:
                dists_to_centroid = np.zeros(n_regime)
                for mask in [long_mask, short_mask]:
                    if mask.sum() < 2:
                        continue
                    pts = np.column_stack([r_entry_a[mask], r_entry_b[mask], r_entry_c[mask]])
                    centroid = pts.mean(axis=0)
                    d = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
                    dists_to_centroid[mask] = d

                if np.std(dists_to_centroid) > 1e-9 and np.std(r_profits) > 1e-9:
                    corr = np.corrcoef(dists_to_centroid, r_profits)[0, 1]
                    if not np.isnan(corr):
                        sim_corr = -corr  # 负相关 = 近中心→高利润

            # 此市场状态的局部适应度
            regime_fitness = (0.3 * compactness +
                              0.4 * separation +
                              0.3 * max(0, sim_corr))

            # 以交易数量为权重汇总
            total_fitness += regime_fitness * n_regime
            total_weight += n_regime

        return total_fitness / (total_weight + 1e-9) if total_weight > 0 else 0.0

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
        mask = np.random.random(self.dim) < 0.5
        child_w = np.where(mask, p1.weights, p2.weights)
        return GAIndividual(weights=child_w)

    def _mutate(self, ind: GAIndividual):
        """高斯变异"""
        mask = np.random.random(self.dim) < self.mutation_rate
        if mask.any():
            noise = np.random.randn(mask.sum()) * self.mutation_strength
            ind.weights[mask] += noise
            # 裁剪到合理范围
            w_min = VECTOR_SPACE_CONFIG.get("WEIGHT_MIN", -10.0)
            w_max = VECTOR_SPACE_CONFIG.get("WEIGHT_MAX", 10.0)
            ind.weights = np.clip(ind.weights, w_min, w_max)
