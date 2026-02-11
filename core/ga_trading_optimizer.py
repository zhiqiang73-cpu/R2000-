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
    """交易参数（贝叶斯优化搜索空间）"""
    # ── 入场参数（优化） ──
    cosine_threshold: float = 0.6       # 余弦粗筛阈值 [0.3, 0.9]
    dtw_threshold: float = 0.5          # DTW入场阈值 [0.1, 0.8]
    min_templates_agree: int = 1        # 最少几个模板一致 [1, 5]

    # ── 离场参数（优化） ──
    stop_loss_atr: float = 2.0          # 止损ATR倍数 [1.0, 5.0]
    take_profit_atr: float = 3.0        # 止盈ATR倍数 [1.5, 8.0]
    max_hold_bars: int = 240            # 最大持仓K线数 [60, 480]
    hold_divergence_limit: float = 0.7  # 持仓偏离上限 [0.3, 0.9]（回测用）

    # ── 动态追踪参数（固定值，仅模拟盘用，不优化） ──
    hold_safe_threshold: float = 0.7    # 安全线：相似度高于此值继续持仓
    hold_alert_threshold: float = 0.5   # 警戒线：收紧止损
    hold_derail_threshold: float = 0.3  # 脱轨线：强制平仓
    hold_check_interval: int = 3        # 动态追踪检查间隔K线数
    use_dynamic_tracking: bool = False  # 是否启用动态追踪（Walk-Forward=False, 模拟盘=True）

    # 参数范围（仅包含需要优化的参数）
    RANGES = {
        "cosine_threshold": (0.3, 0.9),
        "dtw_threshold": (0.1, 0.8),
        "min_templates_agree": (1, 5),
        "stop_loss_atr": (1.0, 5.0),
        "take_profit_atr": (1.5, 8.0),
        "max_hold_bars": (60, 480),
        "hold_divergence_limit": (0.3, 0.9),
    }

    def to_array(self) -> np.ndarray:
        """转为数组（用于GA兼容）"""
        return np.array([
            self.cosine_threshold,
            self.dtw_threshold,
            float(self.min_templates_agree),
            self.stop_loss_atr,
            self.take_profit_atr,
            float(self.max_hold_bars),
            self.hold_divergence_limit,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TradingParams':
        """从数组还原"""
        return cls(
            cosine_threshold=float(arr[0]),
            dtw_threshold=float(arr[1]),
            min_templates_agree=max(1, int(round(arr[2]))),
            stop_loss_atr=float(arr[3]),
            take_profit_atr=float(arr[4]),
            max_hold_bars=int(round(arr[5])),
            hold_divergence_limit=float(arr[6]),
        )

    @classmethod
    def random(cls) -> 'TradingParams':
        """随机生成参数"""
        return cls(
            cosine_threshold=np.random.uniform(*cls.RANGES["cosine_threshold"]),
            dtw_threshold=np.random.uniform(*cls.RANGES["dtw_threshold"]),
            min_templates_agree=int(np.random.uniform(*cls.RANGES["min_templates_agree"])),
            stop_loss_atr=np.random.uniform(*cls.RANGES["stop_loss_atr"]),
            take_profit_atr=np.random.uniform(*cls.RANGES["take_profit_atr"]),
            max_hold_bars=int(np.random.uniform(*cls.RANGES["max_hold_bars"])),
            hold_divergence_limit=np.random.uniform(*cls.RANGES["hold_divergence_limit"]),
        )

    def clip(self) -> 'TradingParams':
        """裁剪到有效范围"""
        return TradingParams(
            cosine_threshold=np.clip(self.cosine_threshold, *self.RANGES["cosine_threshold"]),
            dtw_threshold=np.clip(self.dtw_threshold, *self.RANGES["dtw_threshold"]),
            min_templates_agree=int(np.clip(self.min_templates_agree, *self.RANGES["min_templates_agree"])),
            stop_loss_atr=np.clip(self.stop_loss_atr, *self.RANGES["stop_loss_atr"]),
            take_profit_atr=np.clip(self.take_profit_atr, *self.RANGES["take_profit_atr"]),
            max_hold_bars=int(np.clip(self.max_hold_bars, *self.RANGES["max_hold_bars"])),
            hold_divergence_limit=np.clip(self.hold_divergence_limit, *self.RANGES["hold_divergence_limit"]),
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
class SimulatedTradeRecord:
    """单笔模拟交易记录"""
    entry_idx: int
    exit_idx: int
    side: int  # 1=LONG, -1=SHORT
    profit_pct: float
    matched_template: object = None  # TrajectoryTemplate
    template_fingerprint: str = ""


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
    trades: List[SimulatedTradeRecord] = field(default_factory=list)  # 详细交易记录


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

    def run(self, sub_callback: Callable = None) -> Tuple[TradingParams, float]:
        """
        运行GA优化

        Args:
            sub_callback: 细粒度回调 fn(gen, ind_idx, pop_size) 用于更频繁更新

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
            # 评估适应度（每个个体评估后更新进度）
            for ind_idx, ind in enumerate(population):
                ind.fitness = self._evaluate(ind.params)
                # 细粒度进度回调
                if sub_callback and ind_idx % 5 == 0:  # 每5个个体更新一次
                    sub_callback(gen, ind_idx, self.pop_size)

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

    def _simulate_trading(self, params: TradingParams,
                          record_templates: bool = False,
                          fast_mode: bool = True) -> SimulatedTradeResult:
        """
        在验证集上模拟交易

        简化逻辑：
          1. 遍历每根K线
          2. 如果未持仓，检查是否有入场信号（轨迹匹配）
          3. 如果持仓，检查止盈止损或离场信号

        Args:
            params: 交易参数
            record_templates: 是否记录详细的模板匹配信息（用于模板评估）
            fast_mode: 快速模式（跳跃采样）
        """
        from core.trajectory_matcher import TrajectoryMatcher
        from core.trajectory_engine import extract_current_trajectory

        matcher = TrajectoryMatcher()
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
        skip_bars = TRAJECTORY_CONFIG.get("EVAL_SKIP_BARS", 5) if fast_mode else 1

        n = len(self.val_df)
        profits = []
        trades = []  # 详细交易记录
        position = None  # (entry_idx, entry_price, side, matched_template)

        def close_position(exit_idx: int, profit_pct: float):
            """关闭仓位并记录交易"""
            nonlocal position
            profits.append(profit_pct)
            if record_templates and position is not None:
                template = position.get("template")
                fp = template.fingerprint() if template else ""
                trades.append(SimulatedTradeRecord(
                    entry_idx=position["entry_idx"],
                    exit_idx=exit_idx,
                    side=position["side"],
                    profit_pct=profit_pct,
                    matched_template=template,
                    template_fingerprint=fp,
                ))
            position = None

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
                        close_position(i, profit_pct)
                        continue
                    elif current_price <= position["stop_loss"]:
                        profit_pct = (current_price / position["entry_price"] - 1) * 100
                        close_position(i, profit_pct)
                        continue
                else:  # 空头
                    if current_price <= position["take_profit"]:
                        profit_pct = (1 - current_price / position["entry_price"]) * 100
                        close_position(i, profit_pct)
                        continue
                    elif current_price >= position["stop_loss"]:
                        profit_pct = (1 - current_price / position["entry_price"]) * 100
                        close_position(i, profit_pct)
                        continue

                # 最大持仓时间
                if hold_bars >= params.max_hold_bars:
                    if side == 1:
                        profit_pct = (current_price / position["entry_price"] - 1) * 100
                    else:
                        profit_pct = (1 - current_price / position["entry_price"]) * 100
                    close_position(i, profit_pct)
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
                        close_position(i, profit_pct)
                        continue

        # 计算统计
        result = SimulatedTradeResult()
        result.n_trades = len(profits)
        result.trades = trades
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


# ============================================================================
# 贝叶斯优化器（使用 Optuna，比 GA 快 5-10 倍）
# ============================================================================

class BayesianTradingOptimizer:
    """
    贝叶斯交易参数优化器（基于Optuna的TPE采样器）
    
    相比GA优化器：
      - 评估次数从 900次（30代×30个体）降到 50-100次
      - 速度快 5-10 倍
      - 最终效果相当
    
    使用方式与 GATradingOptimizer 完全相同：
        optimizer = BayesianTradingOptimizer(
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
                 callback: Callable = None,
                 n_trials: int = 60,
                 prototype_library=None):  # 新增：原型库（使用时替代模板）
        """
        Args:
            trajectory_memory: TrajectoryMemory 实例（训练集模板）
            fv_engine: FeatureVectorEngine 实例
            val_df: 验证集 DataFrame
            val_labels: 验证集标签
            regime_classifier: 市场状态分类器（可选）
            callback: 回调函数 fn(trial_idx, best_fitness, best_params)
            n_trials: 优化评估次数（默认60，相当于GA的 2代×30个体）
            prototype_library: PrototypeLibrary 实例（可选，提供时使用原型匹配）
        """
        self.memory = trajectory_memory
        self.fv_engine = fv_engine
        self.val_df = val_df
        self.val_labels = val_labels
        self.regime_classifier = regime_classifier
        self.callback = callback
        self.n_trials = n_trials
        
        # 原型库支持
        self.prototype_library = prototype_library
        self.use_prototypes = prototype_library is not None
        
        # 最佳结果追踪
        self._best_params = TradingParams()
        self._best_fitness = -np.inf
        self._trial_count = 0
        
        # 入场匹配缓存（同一数据集上，多trial复用）
        self._entry_match_cache = None
        self._cache_pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
        self._cache_min_candidates = 50
        
        # 预计算验证集数据
        self._prepare_validation_data()
    
    def _prepare_validation_data(self):
        """预计算验证集数据"""
        if not self.fv_engine.is_ready:
            from utils.indicators import calculate_all_indicators
            self.val_df = calculate_all_indicators(self.val_df)
            self.fv_engine.precompute(self.val_df)
        
        self.val_atr = self.val_df['atr'].values if 'atr' in self.val_df.columns else np.ones(len(self.val_df))
        self.val_close = self.val_df['close'].values
    
    def _get_direction_candidates(self, direction: str):
        """获取方向候选模板（候选不足时回退全量）"""
        if self.memory is None:
            return []
        regime = "未知"
        candidates = self.memory.get_candidates(regime, direction)
        if len(candidates) < self._cache_min_candidates:
            candidates = self.memory.get_templates_by_direction(direction)
        return candidates
    
    def _build_entry_match_cache(self):
        """
        预计算每个bar的入场匹配Top-K（与参数无关）：
        后续trial只做阈值判断，避免重复DTW。
        """
        from core.trajectory_matcher import TrajectoryMatcher
        
        # 仅在无regime分类时启用缓存（Batch-WF路径）
        if self.regime_classifier is not None:
            self._entry_match_cache = None
            return
        
        n = len(self.val_df)
        pre_window = self._cache_pre_window
        matcher = TrajectoryMatcher()
        top_k = matcher.cosine_top_k
        
        long_candidates = self._get_direction_candidates("LONG")
        short_candidates = self._get_direction_candidates("SHORT")
        
        def build_vector_store(candidates):
            """
            预构建可向量化的模板矩阵：
              key = flatten长度, value = (matrix, row_norms, templates)
            """
            stores = {}
            for tmpl in candidates:
                flat = tmpl.pre_entry_flat
                if flat.size == 0:
                    continue
                l = len(flat)
                stores.setdefault(l, []).append((tmpl, flat))
            
            out = {}
            for l, items in stores.items():
                mats = np.stack([x[1] for x in items], axis=0).astype(np.float64, copy=False)
                norms = np.linalg.norm(mats, axis=1)
                tmpls = [x[0] for x in items]
                out[l] = (mats, norms, tmpls)
            return out
        
        long_store = build_vector_store(long_candidates)
        short_store = build_vector_store(short_candidates)
        
        cache = {
            "long": [None] * n,
            "short": [None] * n,
        }
        
        def compute_topk(current_traj, candidates, store):
            if len(candidates) == 0 or current_traj.size == 0:
                return []
            
            current_flat = matcher._zscore_flatten(current_traj)
            q_len = len(current_flat)
            q_norm = np.linalg.norm(current_flat)
            if q_norm < 1e-12:
                return []
            
            cosine_scores = []
            
            # 向量化主路径：相同长度模板一批算余弦
            if q_len in store:
                mats, row_norms, tmpls = store[q_len]
                denom = row_norms * q_norm
                valid = denom > 1e-12
                sims = np.full(len(tmpls), -1.0, dtype=np.float64)
                if np.any(valid):
                    sims[valid] = (mats[valid] @ current_flat) / denom[valid]
                for tmpl, sim in zip(tmpls, sims):
                    if sim > -0.5:
                        cosine_scores.append((tmpl, float(sim)))
            
            # 兼容少量“长度不一致模板”（极少见）
            if len(cosine_scores) < min(top_k * 3, len(candidates)):
                for template in candidates:
                    if template.pre_entry_flat.size == 0:
                        continue
                    if len(template.pre_entry_flat) == q_len:
                        continue
                    t_flat = matcher._align_vectors(current_flat, template.pre_entry_flat)
                    sim = matcher._cosine_similarity(current_flat, t_flat)
                    cosine_scores.append((template, float(sim)))
            
            if not cosine_scores:
                return []
            
            cosine_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = cosine_scores[:top_k]
            
            rows = []
            for tmpl, cos_sim in top_candidates:
                dtw_dist = float(matcher._compute_dtw_distance(current_traj, tmpl.pre_entry))
                rows.append((
                    tmpl,                         # template
                    float(cos_sim),              # cosine
                    dtw_dist,                    # dtw_distance
                    float(max(0.0, 1.0 - dtw_dist)),  # dtw_similarity
                ))
            return rows
        
        for i in range(pre_window, n - 1):
            current_traj = self.fv_engine.get_raw_matrix(i - pre_window, i)
            if current_traj.shape[0] < pre_window:
                continue
            cache["long"][i] = compute_topk(current_traj, long_candidates, long_store)
            cache["short"][i] = compute_topk(current_traj, short_candidates, short_store)
        
        self._entry_match_cache = cache
    
    @staticmethod
    def _select_from_cached_rows(rows, cosine_threshold: float, dtw_threshold: float):
        """
        从缓存行中按当前阈值选最佳匹配
        Returns:
            (matched, best_template, dtw_similarity)
        """
        if not rows:
            return False, None, 0.0
        
        best = None
        for tmpl, cos_sim, dtw_dist, dtw_sim in rows:
            if cos_sim < cosine_threshold:
                continue
            if best is None or dtw_dist < best[2]:
                best = (tmpl, cos_sim, dtw_dist, dtw_sim)
        
        if best is None:
            return False, None, 0.0
        
        matched = best[2] <= dtw_threshold
        return matched, best[0], best[3]
    
    def run(self, sub_callback: Callable = None) -> Tuple[TradingParams, float]:
        """
        运行贝叶斯优化
        
        Args:
            sub_callback: 细粒度回调 fn(trial_idx, n_trials, pop_size) 兼容GA接口
        
        Returns:
            (best_params, best_fitness)
        """
        import optuna
        
        # 禁用 Optuna 日志输出
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # 预构建入场匹配缓存（多trial复用）
        # 原型模式不使用模板DTW缓存，避免访问 self.memory(None)
        if (not self.use_prototypes) and self._entry_match_cache is None:
            self._build_entry_match_cache()
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna 目标函数 — 搜索 7 个交易参数（回测优化）"""
            # 入场参数
            cosine_threshold = trial.suggest_float(
                "cosine_threshold", *TradingParams.RANGES["cosine_threshold"])
            dtw_threshold = trial.suggest_float(
                "dtw_threshold", *TradingParams.RANGES["dtw_threshold"])
            min_templates_agree = trial.suggest_int(
                "min_templates_agree", *TradingParams.RANGES["min_templates_agree"])

            # 离场参数
            stop_loss_atr = trial.suggest_float(
                "stop_loss_atr", *TradingParams.RANGES["stop_loss_atr"])
            take_profit_atr = trial.suggest_float(
                "take_profit_atr", *TradingParams.RANGES["take_profit_atr"])
            max_hold_bars = trial.suggest_int(
                "max_hold_bars", *TradingParams.RANGES["max_hold_bars"])
            hold_divergence_limit = trial.suggest_float(
                "hold_divergence_limit", *TradingParams.RANGES["hold_divergence_limit"])

            params = TradingParams(
                cosine_threshold=cosine_threshold,
                dtw_threshold=dtw_threshold,
                min_templates_agree=min_templates_agree,
                stop_loss_atr=stop_loss_atr,
                take_profit_atr=take_profit_atr,
                max_hold_bars=max_hold_bars,
                hold_divergence_limit=hold_divergence_limit,
                use_dynamic_tracking=False,  # Walk-Forward 不用动态追踪
            )
            
            # 评估
            fitness = self._evaluate(params)
            
            # 更新最佳结果
            if fitness > self._best_fitness:
                self._best_fitness = fitness
                self._best_params = params
            
            self._trial_count += 1
            
            # 回调进度
            if self.callback:
                self.callback(self._trial_count, self._best_fitness, self._best_params)
            
            # 细粒度回调（兼容GA接口）
            if sub_callback:
                # 回调: (trial_idx, total_trials, total_trials)
                sub_callback(self._trial_count, self.n_trials, self.n_trials)
            
            return fitness
        
        # 创建 Optuna 研究（使用TPE采样器，即贝叶斯优化）
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
        study = optuna.create_study(
            direction="maximize",  # 最大化 Sharpe Ratio
            sampler=sampler,
        )
        
        # 添加默认参数作为初始点
        default_params = TradingParams()
        study.enqueue_trial({
            "cosine_threshold": default_params.cosine_threshold,
            "dtw_threshold": default_params.dtw_threshold,
            "min_templates_agree": default_params.min_templates_agree,
            "stop_loss_atr": default_params.stop_loss_atr,
            "take_profit_atr": default_params.take_profit_atr,
            "max_hold_bars": default_params.max_hold_bars,
            "hold_divergence_limit": default_params.hold_divergence_limit,
        })
        
        # 运行优化
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        # 返回最佳结果
        best_trial = study.best_trial
        best_params = TradingParams(
            cosine_threshold=best_trial.params["cosine_threshold"],
            dtw_threshold=best_trial.params["dtw_threshold"],
            min_templates_agree=best_trial.params["min_templates_agree"],
            stop_loss_atr=best_trial.params["stop_loss_atr"],
            take_profit_atr=best_trial.params["take_profit_atr"],
            max_hold_bars=best_trial.params["max_hold_bars"],
            hold_divergence_limit=best_trial.params["hold_divergence_limit"],
            use_dynamic_tracking=False,  # 回测不用动态追踪
        )
        
        print(f"[Bayesian] 优化完成: trials={self.n_trials}, best_sharpe={study.best_value:.4f}")
        
        return best_params, study.best_value
    
    def _evaluate(self, params: TradingParams) -> float:
        """评估适应度"""
        result = self._simulate_trading(params)
        if result.n_trades < 5:
            return -10.0
        return result.sharpe_ratio
    
    def _simulate_trading(self, params: TradingParams,
                          record_templates: bool = False,
                          fast_mode: bool = True) -> SimulatedTradeResult:
        """
        在验证集上模拟交易
        
        三种模式：
          A. use_prototypes=True（原型匹配模式）
             → 使用 PrototypeMatcher 匹配 52 个原型
             → 速度快 200 倍，统计更稳健
          
          B. use_dynamic_tracking=False（模板 Walk-Forward回测）
             → 简化逻辑：TP/SL + 单一偏离度 hold_divergence_limit
             → 交易数多，统计可靠，适合参数优化
          
          C. use_dynamic_tracking=True（模拟盘/实盘）
             → 完整逻辑：动态追踪 + 3阈值 + 模板切换
             → 保守管理，风险控制，适合真实交易
        
        Args:
            params: 交易参数
            record_templates: 是否记录模板/原型信息
            fast_mode: 快速模式（入场跳跃采样）
        """
        # ── 原型模式：使用 PrototypeMatcher ──
        if self.use_prototypes:
            return self._simulate_trading_prototypes(params, record_templates, fast_mode)
        
        from core.trajectory_matcher import TrajectoryMatcher

        matcher = TrajectoryMatcher()
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
        skip_bars = TRAJECTORY_CONFIG.get("EVAL_SKIP_BARS", 5) if fast_mode else 1
        check_interval = max(2, params.hold_check_interval)

        n = len(self.val_df)
        profits = []
        trades = []
        position = None

        def calc_profit(entry_price, current_price, side):
            if side == 1:
                return (current_price / entry_price - 1) * 100
            else:
                return (1 - current_price / entry_price) * 100

        def close_position(exit_idx: int, profit_pct: float):
            nonlocal position
            profits.append(profit_pct)
            if record_templates and position is not None:
                template = position.get("template")
                fp = template.fingerprint() if template else ""
                trades.append(SimulatedTradeRecord(
                    entry_idx=position["entry_idx"],
                    exit_idx=exit_idx,
                    side=position["side"],
                    profit_pct=profit_pct,
                    matched_template=template,
                    template_fingerprint=fp,
                ))
            position = None

        def find_best_holding_match(holding_traj, direction, top_k=3):
            """
            快速动态模板追踪（两阶段：余弦粗筛 + DTW精筛）
            
            速度：
              - 余弦粗筛 2800 个模板：< 1ms（向量点积）
              - DTW精筛 Top-3：< 10ms
              - 总计：< 11ms（远快于1分钟K线间隔）
            
            Returns:
                (best_template, best_similarity)
            """
            candidates = self.memory.get_templates_by_direction(
                "LONG" if direction == 1 else "SHORT"
            )
            if not candidates:
                return None, 0.0

            # ── 第1阶段：余弦相似度快速粗筛（微秒级）──
            current_len = len(holding_traj)
            # 展平+标准化当前持仓轨迹
            current_flat = holding_traj.flatten()
            c_std = current_flat.std()
            if c_std > 1e-9:
                current_flat = (current_flat - current_flat.mean()) / c_std
            else:
                current_flat = current_flat - current_flat.mean()

            cosine_scores = []
            for idx, tmpl in enumerate(candidates):
                if tmpl.holding.size == 0:
                    continue
                # 取对应长度的模板holding段
                tmpl_len = len(tmpl.holding)
                if current_len <= tmpl_len:
                    tmpl_seg = tmpl.holding[:current_len]
                else:
                    tmpl_seg = tmpl.holding

                # 展平+标准化模板段
                tmpl_flat = tmpl_seg.flatten()
                t_std = tmpl_flat.std()
                if t_std > 1e-9:
                    tmpl_flat = (tmpl_flat - tmpl_flat.mean()) / t_std
                else:
                    tmpl_flat = tmpl_flat - tmpl_flat.mean()

                # 对齐长度
                min_len = min(len(current_flat), len(tmpl_flat))
                if min_len == 0:
                    continue
                c = current_flat[:min_len]
                t = tmpl_flat[:min_len]

                # 余弦相似度（向量点积，微秒级）
                norm_c = np.linalg.norm(c)
                norm_t = np.linalg.norm(t)
                if norm_c < 1e-9 or norm_t < 1e-9:
                    continue
                cos_sim = float(np.dot(c, t) / (norm_c * norm_t))
                cosine_scores.append((idx, cos_sim))

            if not cosine_scores:
                return None, 0.0

            # 取余弦 Top-K
            cosine_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = cosine_scores[:top_k]

            # ── 第2阶段：DTW精筛 Top-K（毫秒级）──
            best_sim = -1.0
            best_tmpl = None
            for idx, cos_sim in top_candidates:
                tmpl = candidates[idx]
                tmpl_len = len(tmpl.holding)
                if current_len <= tmpl_len:
                    tmpl_seg = tmpl.holding[:current_len]
                    h_traj = holding_traj
                else:
                    tmpl_seg = tmpl.holding
                    h_traj = holding_traj[:tmpl_len]

                dtw_dist = matcher._compute_dtw_distance(h_traj, tmpl_seg)
                sim = max(0.0, 1.0 - dtw_dist)
                if sim > best_sim:
                    best_sim = sim
                    best_tmpl = tmpl

            return best_tmpl, best_sim

        # ── 主循环 ──
        i = pre_window
        while i < n - 1:
            if position is None:
                # ── 入场逻辑 ──
                if self._entry_match_cache is not None:
                    long_rows = self._entry_match_cache["long"][i]
                    short_rows = self._entry_match_cache["short"][i]
                    
                    long_matched, long_tmpl, long_sim = self._select_from_cached_rows(
                        long_rows, params.cosine_threshold, params.dtw_threshold
                    )
                    short_matched, short_tmpl, short_sim = self._select_from_cached_rows(
                        short_rows, params.cosine_threshold, params.dtw_threshold
                    )
                    
                    if long_matched and short_matched:
                        if long_sim > short_sim:
                            side = 1
                            matched = long_tmpl
                        else:
                            side = -1
                            matched = short_tmpl
                    elif long_matched:
                        side = 1
                        matched = long_tmpl
                    elif short_matched:
                        side = -1
                        matched = short_tmpl
                    else:
                        i += skip_bars
                        continue
                else:
                    current_traj = self.fv_engine.get_raw_matrix(i - pre_window, i)
                    if current_traj.shape[0] < pre_window:
                        i += skip_bars
                        continue
    
                    regime = "未知"
                    if self.regime_classifier:
                        regime = self.regime_classifier.classify_at(i)
    
                    # 获取候选模板（按市场状态过滤，若数量不足则使用全量）
                    MIN_CANDIDATES = 50  # 至少需要50个候选模板才有意义
                    
                    long_candidates = self.memory.get_candidates(regime, "LONG")
                    if len(long_candidates) < MIN_CANDIDATES:
                        long_candidates = self.memory.get_templates_by_direction("LONG")
    
                    long_result = matcher.match_entry(
                        current_traj, long_candidates,
                        params.cosine_threshold, params.dtw_threshold
                    )
    
                    short_candidates = self.memory.get_candidates(regime, "SHORT")
                    if len(short_candidates) < MIN_CANDIDATES:
                        short_candidates = self.memory.get_templates_by_direction("SHORT")
    
                    short_result = matcher.match_entry(
                        current_traj, short_candidates,
                        params.cosine_threshold, params.dtw_threshold
                    )
    
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
                        i += skip_bars
                        continue

                entry_price = self.val_close[i]
                atr = self.val_atr[i]
                position = {
                    "entry_idx": i,
                    "entry_price": entry_price,
                    "side": side,
                    "template": matched,  # 当前管理模板（可动态切换）
                    "stop_loss": entry_price - side * params.stop_loss_atr * atr,
                    "take_profit": entry_price + side * params.take_profit_atr * atr,
                    "original_stop_loss": entry_price - side * params.stop_loss_atr * atr,
                    "alert_mode": False,  # 是否进入警戒模式
                }
                i += 1
            else:
                # ── 持仓逻辑 ──
                current_price = self.val_close[i]
                side = position["side"]
                hold_bars = i - position["entry_idx"]

                # 1. 硬止盈止损（每根K线检查）
                pnl = calc_profit(position["entry_price"], current_price, side)
                if side == 1:
                    if current_price >= position["take_profit"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                    elif current_price <= position["stop_loss"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                else:
                    if current_price <= position["take_profit"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                    elif current_price >= position["stop_loss"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue

                # 2. 最大持仓时间
                if hold_bars >= params.max_hold_bars:
                    close_position(i, pnl)
                    i += skip_bars
                    continue

                # 3. 持仓偏离检查（根据模式选择）
                if params.use_dynamic_tracking:
                    # ── 模式A：完整动态追踪（模拟盘/实盘）──
                    if hold_bars > 0 and hold_bars % check_interval == 0:
                        holding_traj = self.fv_engine.get_raw_matrix(
                            position["entry_idx"], i + 1
                        )
                        if holding_traj.size > 0:
                            best_tmpl, best_sim = find_best_holding_match(
                                holding_traj, side
                            )

                            if best_sim >= params.hold_safe_threshold:
                                # 安全区：切换到最佳模板（如果变了）
                                if best_tmpl is not None:
                                    position["template"] = best_tmpl
                                # 如果之前在警戒模式，恢复原止损
                                if position["alert_mode"]:
                                    position["stop_loss"] = position["original_stop_loss"]
                                    position["alert_mode"] = False

                            elif best_sim >= params.hold_alert_threshold:
                                # 警戒区：收紧止损到成本价（保本）
                                if not position["alert_mode"]:
                                    position["alert_mode"] = True
                                    # 止损移到入场价（保本止损）
                                    position["stop_loss"] = position["entry_price"]
                                if best_tmpl is not None:
                                    position["template"] = best_tmpl

                            else:
                                # 脱轨区：所有已知模式都不像了 → 强制平仓
                                close_position(i, pnl)
                                i += skip_bars
                                continue
                else:
                    # ── 模式B：简化偏离检查（Walk-Forward回测）──
                    # 注意：至少持仓10个bar后才开始检查偏离度
                    # 因为太短的轨迹与模板比较没有意义（DTW距离会很大）
                    MIN_HOLD_BARS_FOR_DIVERGENCE = 10
                    if position["template"] is not None and hold_bars >= MIN_HOLD_BARS_FOR_DIVERGENCE:
                        holding_traj = self.fv_engine.get_raw_matrix(
                            position["entry_idx"], i + 1
                        )
                        divergence, should_exit = matcher.monitor_holding(
                            holding_traj, position["template"],
                            params.hold_divergence_limit
                        )
                        if should_exit:
                            close_position(i, pnl)
                            i += skip_bars
                            continue

                i += 1

        # ── 统计结果 ──
        result = SimulatedTradeResult()
        result.n_trades = len(profits)
        result.trades = trades
        if result.n_trades == 0:
            return result

        profits_arr = np.array(profits)
        result.profits = profits
        result.n_wins = int((profits_arr > 0).sum())
        result.total_profit = float(profits_arr.sum())
        result.win_rate = result.n_wins / result.n_trades

        mean_return = profits_arr.mean()
        std_return = profits_arr.std()
        if std_return > 1e-9:
            result.sharpe_ratio = mean_return / std_return * np.sqrt(252)
        else:
            result.sharpe_ratio = mean_return * 10 if mean_return > 0 else -10

        cumsum = profits_arr.cumsum()
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        result.max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0

        return result

    def _simulate_trading_prototypes(self, params: TradingParams,
                                      record_templates: bool = False,
                                      fast_mode: bool = True) -> SimulatedTradeResult:
        """
        使用原型进行模拟交易（速度快 200 倍）
        
        匹配逻辑：
          - 使用 PrototypeMatcher 对当前 K 线窗口进行原型匹配
          - 余弦相似度 + 投票机制决定入场
          - 简化的持仓健康度检查
        
        Args:
            params: 交易参数
            record_templates: 是否记录原型信息
            fast_mode: 快速模式（入场跳跃采样）
        
        Returns:
            SimulatedTradeResult
        """
        from core.template_clusterer import PrototypeMatcher
        
        # 初始化原型匹配器
        proto_matcher = PrototypeMatcher(
            library=self.prototype_library,
            cosine_threshold=params.cosine_threshold,
            min_prototypes_agree=params.min_templates_agree,
        )
        
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
        skip_bars = TRAJECTORY_CONFIG.get("EVAL_SKIP_BARS", 5) if fast_mode else 1
        
        n = len(self.val_df)
        profits = []
        trades = []
        position = None
        
        def calc_profit(entry_price, current_price, side):
            if side == 1:
                return (current_price / entry_price - 1) * 100
            else:
                return (1 - current_price / entry_price) * 100
        
        def close_position(exit_idx: int, profit_pct: float):
            nonlocal position
            profits.append(profit_pct)
            if record_templates and position is not None:
                proto = position.get("prototype")
                # 原型没有 fingerprint，用 prototype_id 代替
                fp = f"proto_{proto.direction}_{proto.prototype_id}" if proto else ""
                trades.append(SimulatedTradeRecord(
                    entry_idx=position["entry_idx"],
                    exit_idx=exit_idx,
                    side=position["side"],
                    profit_pct=profit_pct,
                    matched_template=proto,
                    template_fingerprint=fp,
                ))
            position = None
        
        i = pre_window
        while i < n - 1:
            if position is None:
                # ── 入场检查 ──
                current_traj = self.fv_engine.get_raw_matrix(i - pre_window, i)
                if current_traj.shape[0] < pre_window:
                    i += skip_bars
                    continue
                
                # 原型匹配（余弦相似度 + 投票）
                match_result = proto_matcher.match_entry(current_traj, direction=None)
                
                if match_result["matched"]:
                    side = 1 if match_result["direction"] == "LONG" else -1
                    best_proto = match_result["best_prototype"]
                    
                    # 开仓
                    entry_price = self.val_close[i]
                    atr = self.val_atr[i]
                    position = {
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "side": side,
                        "prototype": best_proto,
                        "stop_loss": entry_price * (1 - params.stop_loss_atr * atr / entry_price) if side == 1 else entry_price * (1 + params.stop_loss_atr * atr / entry_price),
                        "take_profit": entry_price * (1 + params.take_profit_atr * atr / entry_price) if side == 1 else entry_price * (1 - params.take_profit_atr * atr / entry_price),
                        "expected_hold": int(best_proto.avg_hold_bars) if best_proto else 60,
                    }
                
                i += skip_bars
            else:
                # ── 持仓管理 ──
                current_price = self.val_close[i]
                hold_bars = i - position["entry_idx"]
                pnl = calc_profit(position["entry_price"], current_price, position["side"])
                
                # 止盈止损检查
                if position["side"] == 1:
                    if current_price <= position["stop_loss"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                    if current_price >= position["take_profit"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                else:
                    if current_price >= position["stop_loss"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                    if current_price <= position["take_profit"]:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                
                # 最大持仓时间
                if hold_bars >= params.max_hold_bars:
                    close_position(i, pnl)
                    i += skip_bars
                    continue
                
                # 持仓健康度检查（使用原型的 holding_centroid）
                if hold_bars >= 10 and position["prototype"] is not None:
                    holding_traj = self.fv_engine.get_raw_matrix(
                        position["entry_idx"], i + 1
                    )
                    health, status = proto_matcher.check_holding_health(
                        holding_traj, position["prototype"]
                    )
                    # 健康度低于偏离限制时平仓
                    if health < params.hold_divergence_limit:
                        close_position(i, pnl)
                        i += skip_bars
                        continue
                
                i += 1
        
        # ── 统计结果 ──
        result = SimulatedTradeResult()
        result.n_trades = len(profits)
        result.trades = trades
        if result.n_trades == 0:
            return result
        
        profits_arr = np.array(profits)
        result.profits = profits
        result.n_wins = int((profits_arr > 0).sum())
        result.total_profit = float(profits_arr.sum())
        result.win_rate = result.n_wins / result.n_trades
        
        mean_return = profits_arr.mean()
        std_return = profits_arr.std()
        if std_return > 1e-9:
            result.sharpe_ratio = mean_return / std_return * np.sqrt(252)
        else:
            result.sharpe_ratio = mean_return * 10 if mean_return > 0 else -10
        
        cumsum = profits_arr.cumsum()
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum
        result.max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0
        
        return result
