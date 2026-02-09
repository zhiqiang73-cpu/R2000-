"""
R3000 遗传算法优化器
优化入场/出场参数、仓位管理参数，适应度函数基于回测结果
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from copy import deepcopy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GA_CONFIG, FEATURE_CONFIG, BACKTEST_CONFIG
from core.backtester import Backtester, BacktestResult


@dataclass
class Individual:
    """个体（策略参数集）"""
    # 特征权重
    long_weights: np.ndarray       # 做多权重 (n_features,)
    short_weights: np.ndarray      # 做空权重 (n_features,)
    
    # 阈值
    long_threshold: float          # 做多阈值
    short_threshold: float         # 做空阈值
    
    # 出场参数
    take_profit: float             # 止盈比例
    stop_loss: float               # 止损比例
    max_hold: int                  # 最大持仓周期
    
    # 适应度
    fitness: float = 0.0
    metrics: Dict = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """转换为一维数组"""
        return np.concatenate([
            self.long_weights,
            self.short_weights,
            [self.long_threshold, self.short_threshold,
             self.take_profit, self.stop_loss, self.max_hold]
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, n_features: int) -> 'Individual':
        """从一维数组创建"""
        long_weights = arr[:n_features]
        short_weights = arr[n_features:2*n_features]
        params = arr[2*n_features:]
        
        return cls(
            long_weights=long_weights,
            short_weights=short_weights,
            long_threshold=params[0],
            short_threshold=params[1],
            take_profit=params[2],
            stop_loss=params[3],
            max_hold=int(params[4])
        )
    
    @classmethod
    def random(cls, n_features: int) -> 'Individual':
        """随机生成个体"""
        return cls(
            long_weights=np.random.uniform(-0.5, 0.5, n_features),
            short_weights=np.random.uniform(-0.5, 0.5, n_features),
            long_threshold=np.random.uniform(0.1, 0.5),
            short_threshold=np.random.uniform(0.1, 0.5),
            take_profit=np.random.uniform(0.01, 0.05),
            stop_loss=np.random.uniform(0.005, 0.03),
            max_hold=np.random.randint(30, 240)
        )


@dataclass
class EvolutionResult:
    """进化结果"""
    best_individual: Individual
    best_fitness: float
    fitness_history: List[float]
    generation: int
    population_stats: Dict


class GeneticOptimizer:
    """
    遗传算法优化器
    
    功能：
    1. 优化特征权重和阈值
    2. 优化止盈止损参数
    3. 使用回测结果作为适应度评估
    """
    
    def __init__(self,
                 population_size: int = None,
                 elite_ratio: float = None,
                 mutation_rate: float = None,
                 mutation_strength: float = None,
                 crossover_rate: float = None,
                 max_generations: int = None,
                 early_stop_generations: int = None):
        """初始化优化器"""
        self.population_size = population_size or GA_CONFIG["POPULATION_SIZE"]
        self.elite_ratio = elite_ratio or GA_CONFIG["ELITE_RATIO"]
        self.mutation_rate = mutation_rate or GA_CONFIG["MUTATION_RATE"]
        self.mutation_strength = mutation_strength or GA_CONFIG["MUTATION_STRENGTH"]
        self.crossover_rate = crossover_rate or GA_CONFIG["CROSSOVER_RATE"]
        self.max_generations = max_generations or GA_CONFIG["MAX_GENERATIONS"]
        self.early_stop_generations = early_stop_generations or GA_CONFIG["EARLY_STOP_GENERATIONS"]
        
        self.n_features = FEATURE_CONFIG["NUM_FEATURES"]
        self.fitness_weights = GA_CONFIG["FITNESS_WEIGHTS"]
        
        # 状态
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.generation = 0
        
        # 回测器
        self.backtester = Backtester()
        
        # 回调
        self.on_generation_complete: Optional[Callable] = None
        
    def evolve(self, df: pd.DataFrame, features: np.ndarray,
               initial_population: List[Individual] = None,
               verbose: bool = True) -> EvolutionResult:
        """
        执行进化优化
        
        Args:
            df: K线数据
            features: 特征矩阵
            initial_population: 初始种群（可选）
            verbose: 是否打印进度
            
        Returns:
            进化结果
        """
        # 初始化种群
        if initial_population:
            self.population = initial_population
        else:
            self._initialize_population()
        
        if verbose:
            print(f"[GA] 开始进化优化，种群大小: {self.population_size}")
        
        # 评估初始种群
        self._evaluate_population(df, features)
        self._sort_population()
        
        self.best_individual = deepcopy(self.population[0])
        self.fitness_history = [self.best_individual.fitness]
        
        no_improve_count = 0
        
        for gen in range(self.max_generations):
            self.generation = gen + 1
            
            # 选择
            parents = self._selection()
            
            # 交叉
            offspring = self._crossover(parents)
            
            # 变异
            offspring = self._mutation(offspring)
            
            # 精英保留
            n_elite = int(self.population_size * self.elite_ratio)
            elite = [deepcopy(ind) for ind in self.population[:n_elite]]
            
            # 新种群
            self.population = elite + offspring[:self.population_size - n_elite]
            
            # 评估
            self._evaluate_population(df, features)
            self._sort_population()
            
            # 更新最优
            if self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = deepcopy(self.population[0])
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            self.fitness_history.append(self.best_individual.fitness)
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"[GA] 第 {gen+1} 代: 最优适应度 = {self.best_individual.fitness:.4f}")
            
            # 回调
            if self.on_generation_complete:
                self.on_generation_complete(self.generation, self.best_individual)
            
            # 早停
            if no_improve_count >= self.early_stop_generations:
                if verbose:
                    print(f"[GA] 早停于第 {gen+1} 代（{no_improve_count} 代无改进）")
                break
        
        if verbose:
            print(f"[GA] 进化完成，最优适应度: {self.best_individual.fitness:.4f}")
            print(f"[GA] 最优参数: TP={self.best_individual.take_profit:.2%}, "
                  f"SL={self.best_individual.stop_loss:.2%}, "
                  f"MaxHold={self.best_individual.max_hold}")
        
        return EvolutionResult(
            best_individual=self.best_individual,
            best_fitness=self.best_individual.fitness,
            fitness_history=self.fitness_history,
            generation=self.generation,
            population_stats=self._get_population_stats()
        )
    
    def _initialize_population(self):
        """初始化随机种群"""
        self.population = [
            Individual.random(self.n_features) 
            for _ in range(self.population_size)
        ]
    
    def _evaluate_population(self, df: pd.DataFrame, features: np.ndarray):
        """评估种群适应度"""
        for ind in self.population:
            if ind.fitness == 0:  # 未评估过
                ind.fitness, ind.metrics = self._evaluate_individual(ind, df, features)
    
    def _evaluate_individual(self, ind: Individual, 
                             df: pd.DataFrame, 
                             features: np.ndarray) -> Tuple[float, Dict]:
        """评估单个个体"""
        # 设置回测参数
        self.backtester.take_profit_pct = ind.take_profit
        self.backtester.stop_loss_pct = ind.stop_loss
        
        # 合并权重
        strategy_weights = np.concatenate([ind.long_weights, ind.short_weights])
        
        # 运行回测
        result = self.backtester.run_with_strategy(
            df, features, strategy_weights,
            ind.long_threshold, ind.short_threshold,
            ind.max_hold
        )
        
        # 计算适应度
        fitness = self._calculate_fitness(result)
        
        metrics = {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_return": result.total_return_pct,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor
        }
        
        return fitness, metrics
    
    def _calculate_fitness(self, result: BacktestResult) -> float:
        """计算适应度分数"""
        # 基础检查
        if result.total_trades < 5:
            return -1.0  # 交易次数过少
        
        # 各指标得分
        scores = {}
        
        # 夏普比率得分 (归一化到 0-1)
        sharpe = result.sharpe_ratio
        scores['sharpe_ratio'] = np.tanh(sharpe / 3)  # 3 是一个不错的夏普比率
        
        # 最大回撤得分 (越小越好)
        mdd = result.max_drawdown
        scores['max_drawdown'] = 1 - min(mdd, 1)
        
        # 胜率得分
        scores['win_rate'] = result.win_rate
        
        # 盈亏比得分 (归一化)
        pf = result.profit_factor
        if pf == float('inf'):
            pf = 10
        scores['profit_factor'] = np.tanh(pf / 3)
        
        # 加权求和
        fitness = sum(
            scores.get(key, 0) * weight 
            for key, weight in self.fitness_weights.items()
        )
        
        return fitness
    
    def _sort_population(self):
        """按适应度排序（降序）"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
    
    def _selection(self) -> List[Individual]:
        """锦标赛选择"""
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # 随机选择参赛者
            contestants = np.random.choice(
                len(self.population), 
                size=tournament_size, 
                replace=False
            )
            # 选择适应度最高的
            winner_idx = max(contestants, key=lambda i: self.population[i].fitness)
            parents.append(deepcopy(self.population[winner_idx]))
        
        return parents
    
    def _crossover(self, parents: List[Individual]) -> List[Individual]:
        """均匀交叉"""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if np.random.random() < self.crossover_rate:
                # 转换为数组
                arr1 = parent1.to_array()
                arr2 = parent2.to_array()
                
                # 均匀交叉
                mask = np.random.random(len(arr1)) < 0.5
                child1_arr = np.where(mask, arr1, arr2)
                child2_arr = np.where(mask, arr2, arr1)
                
                child1 = Individual.from_array(child1_arr, self.n_features)
                child2 = Individual.from_array(child2_arr, self.n_features)
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([deepcopy(parent1), deepcopy(parent2)])
        
        return offspring
    
    def _mutation(self, offspring: List[Individual]) -> List[Individual]:
        """高斯变异"""
        for ind in offspring:
            if np.random.random() < self.mutation_rate:
                # 权重变异
                mask = np.random.random(self.n_features) < 0.2
                ind.long_weights[mask] += np.random.randn(mask.sum()) * self.mutation_strength
                ind.long_weights = np.clip(ind.long_weights, -1, 1)
                
                mask = np.random.random(self.n_features) < 0.2
                ind.short_weights[mask] += np.random.randn(mask.sum()) * self.mutation_strength
                ind.short_weights = np.clip(ind.short_weights, -1, 1)
                
                # 阈值变异
                if np.random.random() < 0.3:
                    ind.long_threshold += np.random.randn() * 0.05
                    ind.long_threshold = np.clip(ind.long_threshold, 0.05, 0.8)
                
                if np.random.random() < 0.3:
                    ind.short_threshold += np.random.randn() * 0.05
                    ind.short_threshold = np.clip(ind.short_threshold, 0.05, 0.8)
                
                # 出场参数变异
                if np.random.random() < 0.3:
                    ind.take_profit += np.random.randn() * 0.005
                    ind.take_profit = np.clip(ind.take_profit, 0.005, 0.1)
                
                if np.random.random() < 0.3:
                    ind.stop_loss += np.random.randn() * 0.003
                    ind.stop_loss = np.clip(ind.stop_loss, 0.003, 0.05)
                
                if np.random.random() < 0.3:
                    ind.max_hold += int(np.random.randn() * 20)
                    ind.max_hold = np.clip(ind.max_hold, 15, 480)
                
                # 重置适应度
                ind.fitness = 0
        
        return offspring
    
    def _get_population_stats(self) -> Dict:
        """获取种群统计信息"""
        fitnesses = [ind.fitness for ind in self.population]
        return {
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
        }
    
    def get_best_strategy(self) -> Dict:
        """获取最优策略参数"""
        if self.best_individual is None:
            return {}
        
        return {
            "long_weights": self.best_individual.long_weights.tolist(),
            "short_weights": self.best_individual.short_weights.tolist(),
            "long_threshold": self.best_individual.long_threshold,
            "short_threshold": self.best_individual.short_threshold,
            "take_profit": self.best_individual.take_profit,
            "stop_loss": self.best_individual.stop_loss,
            "max_hold": self.best_individual.max_hold,
            "fitness": self.best_individual.fitness,
            "metrics": self.best_individual.metrics
        }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    n = 5000
    n_features = 52
    
    # 模拟 K 线数据
    test_df = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(n) * 50),
    })
    test_df['close'] = test_df['open'] + np.random.randn(n) * 30
    test_df['high'] = test_df[['open', 'close']].max(axis=1) + abs(np.random.randn(n) * 20)
    test_df['low'] = test_df[['open', 'close']].min(axis=1) - abs(np.random.randn(n) * 20)
    
    # 模拟特征
    features = np.random.randn(n, n_features) * 0.5
    
    # 执行优化
    optimizer = GeneticOptimizer(
        population_size=20,
        max_generations=30,
        early_stop_generations=10
    )
    
    result = optimizer.evolve(test_df, features, verbose=True)
    
    print("\n最优策略:")
    best = optimizer.get_best_strategy()
    print(f"  止盈: {best['take_profit']:.2%}")
    print(f"  止损: {best['stop_loss']:.2%}")
    print(f"  最大持仓: {best['max_hold']} 周期")
    print(f"  适应度: {best['fitness']:.4f}")
    print(f"  指标: {best['metrics']}")
