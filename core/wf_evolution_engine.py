"""
R3000 WF Evolution Engine
特征权重进化引擎 — 使用 CMA-ES 在多折 Walk-Forward 框架下搜索最优特征权重

搜索空间 (12D):
  - 8 维特征组权重: [rsi, macd, volatility, momentum, volume, trend, structure, adx]
  - fusion_threshold: 多维融合分数阈值
  - cosine_min_threshold: 余弦相似度最低阈值
  - euclidean_min_threshold: 欧氏距离最低阈值
  - dtw_min_threshold: DTW形态最低阈值

适应度 = 内部多折 WF 的平均 Sharpe Ratio − L2 正则化

流程:
  1. 从 parquet 采样 SAMPLE_SIZE 根K线
  2. 切分 holdout (最后 HOLDOUT_RATIO%)
  3. 剩余数据按 INNER_FOLDS 折做 Walk-Forward
  4. 每折: 用贝叶斯优化器搜索交易参数，注入进化权重
  5. 适应度 = mean(fold_sharpe) − L2 * ||w − w_default||²
  6. CMA-ES / TPE 搜索最优权重
  7. 用最优权重在 holdout 上做最终验证

输出:
  - evolved_weights.json: 最优 8 维组权重 + 阈值
  - holdout 验证结果
"""

import numpy as np
import json
import os
import time
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WF_EVOLUTION_CONFIG, TRAJECTORY_CONFIG


@dataclass
class EvolutionProgress:
    """进化进度信息（传递给 UI 回调）"""
    trial_idx: int = 0
    n_trials: int = 60
    best_fitness: float = -np.inf
    current_fitness: float = -np.inf
    fold_detail: str = ""
    phase: str = "optimizing"      # "optimizing" | "holdout" | "done"
    elapsed_sec: float = 0.0
    eta_sec: float = 0.0
    group_weights: Optional[np.ndarray] = None  # 当前最优 8 维组权重
    fusion_threshold: float = 0.65
    cosine_min_threshold: float = 0.70
    euclidean_min_threshold: float = 0.50
    dtw_min_threshold: float = 0.40


@dataclass
class EvolutionResult:
    """进化最终结果"""
    success: bool = False
    group_weights: Optional[np.ndarray] = None   # 8 维组权重
    full_weights: Optional[np.ndarray] = None     # 32 维展开后的特征权重
    fusion_threshold: float = 0.65
    cosine_min_threshold: float = 0.70
    euclidean_min_threshold: float = 0.50
    dtw_min_threshold: float = 0.40
    best_fitness: float = -np.inf
    n_trials: int = 0
    elapsed_sec: float = 0.0
    direction: Optional[str] = None   # "LONG" | "SHORT" | None(旧版单组)

    # Holdout 验证结果
    holdout_sharpe: float = 0.0
    holdout_win_rate: float = 0.0
    holdout_profit: float = 0.0
    holdout_drawdown: float = 0.0
    holdout_n_trades: int = 0
    holdout_profit_factor: float = 0.0
    holdout_passed: bool = False
    error_message: str = ""


class WFEvolutionEngine:
    """
    WF 特征权重进化引擎

    使用方式 (后台线程):
        engine = WFEvolutionEngine(
            prototype_library=lib,
            on_progress=progress_cb,
        )
        result = engine.run()
    """

    def __init__(self,
                 prototype_library,
                 on_progress: Optional[Callable[[EvolutionProgress], None]] = None,
                 sample_size: int = None,
                 n_trials: int = None,
                 inner_folds: int = None,
                 holdout_ratio: float = None,
                 direction: Optional[str] = None):
        """
        Args:
            prototype_library: PrototypeLibrary 实例
            on_progress: 进度回调 fn(EvolutionProgress)
            sample_size: 采样K线数 (覆盖 config)
            n_trials: CMA-ES 试验次数 (覆盖 config)
            inner_folds: 内部WF折数 (覆盖 config)
            holdout_ratio: 留出比例 (覆盖 config)
            direction: 多空分开进化时使用 "LONG" | "SHORT" | None(双向，兼容旧行为)
        """
        self.prototype_library = prototype_library
        self.on_progress = on_progress
        self.direction = direction  # 本次进化仅评估的方向

        cfg = WF_EVOLUTION_CONFIG
        self.sample_size = sample_size or cfg["SAMPLE_SIZE"]
        self.n_trials = n_trials or cfg["N_TRIALS"]
        self.inner_folds = inner_folds or cfg["INNER_FOLDS"]
        self.holdout_ratio = holdout_ratio or cfg["HOLDOUT_RATIO"]

        self.weight_min = cfg["WEIGHT_MIN"]
        self.weight_max = cfg["WEIGHT_MAX"]
        self.fusion_range = cfg["FUSION_THRESHOLD_RANGE"]
        self.cosine_min_range = cfg["COSINE_MIN_THRESHOLD_RANGE"]
        self.euclidean_min_range = cfg["EUCLIDEAN_MIN_THRESHOLD_RANGE"]
        self.dtw_min_range = cfg["DTW_MIN_THRESHOLD_RANGE"]
        self.l2_lambda = cfg["L2_LAMBDA"]
        self.min_trades_per_fold = cfg["MIN_TRADES_PER_FOLD"]
        self.min_trades_penalty = cfg["MIN_TRADES_PENALTY"]
        self.eval_skip_bars = cfg.get("EVAL_SKIP_BARS", 10)
        self.holdout_pass_criteria = cfg["HOLDOUT_PASS_CRITERIA"]
        self.results_dir = cfg["RESULTS_DIR"]
        self.evolved_weights_file = cfg["EVOLVED_WEIGHTS_FILE"]
        self.evolved_weights_file_long = cfg.get("EVOLVED_WEIGHTS_FILE_LONG") or (
            cfg["EVOLVED_WEIGHTS_FILE"].replace(".json", "_long.json"))
        self.evolved_weights_file_short = cfg.get("EVOLVED_WEIGHTS_FILE_SHORT") or (
            cfg["EVOLVED_WEIGHTS_FILE"].replace(".json", "_short.json"))
        self.optimizer_type = cfg.get("OPTIMIZER", "cma-es")
        self.cma_sigma0 = cfg.get("CMA_SIGMA0", 0.5)
        self.tpe_fallback = cfg.get("TPE_FALLBACK", True)
        self.feature_groups = cfg["FEATURE_GROUPS"]

        # 停止标志
        self._stop_requested = False
        self._start_time = 0.0
        self._last_error_message = ""

        # 缓存：默认权重（用于 L2 正则化基准）
        self._default_group_weights = self._get_default_group_weights()

    def stop(self):
        """请求停止进化"""
        self._stop_requested = True

    def _get_default_group_weights(self) -> np.ndarray:
        """获取配置中的默认组权重"""
        group_order = ["rsi", "macd", "volatility", "momentum",
                       "volume", "trend", "structure", "adx"]
        defaults = []
        for name in group_order:
            cfg_group = self.feature_groups.get(name, {})
            defaults.append(cfg_group.get("default_weight", 1.0))
        return np.array(defaults, dtype=np.float64)

    def run(self, direction: Optional[str] = None) -> EvolutionResult:
        """
        运行 WF Evolution (主入口, 阻塞调用, 应在后台线程执行)

        Args:
            direction: 本次进化仅评估的方向 "LONG" | "SHORT" | None(双向)。
                       若为 None 则使用 __init__ 的 direction。

        Returns:
            EvolutionResult
        """
        self._stop_requested = False
        self._start_time = time.time()
        self._current_direction = direction if direction is not None else self.direction

        result = EvolutionResult()

        try:
            # ── 1. 加载数据 ──
            self._emit_progress(EvolutionProgress(
                phase="loading", trial_idx=0, n_trials=self.n_trials))

            from core.data_loader import DataLoader
            loader = DataLoader()
            full_df = loader.load_full_data()

            if full_df is None or len(full_df) == 0:
                print("[WF-Evo] 数据加载失败")
                result.error_message = "数据加载失败：未读取到可用K线数据。"
                return result

            # 采样
            total_rows = len(full_df)
            if self.sample_size < total_rows:
                offset = total_rows - self.sample_size
                df = full_df.iloc[offset:].copy().reset_index(drop=True)
            else:
                df = full_df.copy().reset_index(drop=True)

            n = len(df)
            print(f"[WF-Evo] 数据采样: {n:,} 根K线 (总 {total_rows:,})")

            # ── 2. 分割 holdout ──
            holdout_start = int(n * (1 - self.holdout_ratio))
            train_df = df.iloc[:holdout_start].copy().reset_index(drop=True)
            holdout_df = df.iloc[holdout_start:].copy().reset_index(drop=True)
            print(f"[WF-Evo] 训练集: {len(train_df):,}, Holdout: {len(holdout_df):,}")

            # ── 3. 预计算指标 ──
            from utils.indicators import calculate_all_indicators
            train_df = calculate_all_indicators(train_df)
            holdout_df = calculate_all_indicators(holdout_df)

            # ── 4. 分折 ──
            fold_size = len(train_df) // self.inner_folds
            folds = []
            for i in range(self.inner_folds):
                start = i * fold_size
                end = start + fold_size if i < self.inner_folds - 1 else len(train_df)
                folds.append(train_df.iloc[start:end].copy().reset_index(drop=True))
            print(f"[WF-Evo] 内部 {self.inner_folds} 折, 每折 ~{fold_size:,} bars")

            if self._stop_requested:
                return result

            # ── 5. CMA-ES / TPE 优化 ──
            best_weights, best_fitness = self._run_optimization(folds)

            if self._stop_requested:
                result.success = False
                result.n_trials = self.n_trials
                result.elapsed_sec = time.time() - self._start_time
                result.error_message = "用户手动停止进化。"
                return result

            if best_weights is None:
                print("[WF-Evo] 优化无有效结果")
                result.error_message = (
                    self._last_error_message
                    or "优化阶段未得到有效权重。可尝试降低试验次数、减小样本量或检查数据完整性。"
                )
                return result

            # 解析结果
            group_w = best_weights[:8]
            fusion_th = best_weights[8]
            cosine_min_th = best_weights[9]
            euclidean_min_th = best_weights[10]
            dtw_min_th = best_weights[11]

            result.group_weights = group_w
            result.fusion_threshold = fusion_th
            result.cosine_min_threshold = cosine_min_th
            result.euclidean_min_threshold = euclidean_min_th
            result.dtw_min_threshold = dtw_min_th
            result.best_fitness = best_fitness
            result.n_trials = self.n_trials

            # 展开为 32 维
            from core.template_clusterer import MultiSimilarityCalculator
            result.full_weights = MultiSimilarityCalculator.expand_group_weights(group_w)

            # ── 6. Holdout 验证 ──
            self._emit_progress(EvolutionProgress(
                phase="holdout",
                trial_idx=self.n_trials,
                n_trials=self.n_trials,
                best_fitness=best_fitness,
                group_weights=group_w,
                fusion_threshold=fusion_th,
                cosine_min_threshold=cosine_min_th,
                euclidean_min_threshold=euclidean_min_th,
                dtw_min_threshold=dtw_min_th,
            ))

            holdout_result = self._evaluate_on_data(
                holdout_df, group_w, fusion_th, cosine_min_th, euclidean_min_th, dtw_min_th)

            result.holdout_sharpe = holdout_result.get("sharpe", 0.0)
            result.holdout_win_rate = holdout_result.get("win_rate", 0.0)
            result.holdout_profit = holdout_result.get("profit", 0.0)
            result.holdout_drawdown = holdout_result.get("drawdown", 0.0)
            result.holdout_n_trades = holdout_result.get("n_trades", 0)
            result.holdout_profit_factor = holdout_result.get("profit_factor", 0.0)

            # 检查是否通过
            criteria = self.holdout_pass_criteria
            result.holdout_passed = (
                result.holdout_sharpe >= criteria["min_sharpe"]
                and result.holdout_win_rate >= criteria["min_win_rate"]
                and result.holdout_drawdown <= criteria["max_drawdown"]
                and result.holdout_n_trades >= criteria["min_trades"]
                and result.holdout_profit_factor >= criteria["min_profit_factor"]
            )

            result.success = True
            result.elapsed_sec = time.time() - self._start_time
            result.direction = self._current_direction

            print(f"[WF-Evo] 完成! fitness={best_fitness:.4f}, "
                  f"holdout_sharpe={result.holdout_sharpe:.4f}, "
                  f"passed={result.holdout_passed}, "
                  f"elapsed={result.elapsed_sec:.0f}s")

            # ── 7. 通知完成 ──
            self._emit_progress(EvolutionProgress(
                phase="done",
                trial_idx=self.n_trials,
                n_trials=self.n_trials,
                best_fitness=best_fitness,
                group_weights=group_w,
                fusion_threshold=fusion_th,
                cosine_min_threshold=cosine_min_th,
                elapsed_sec=result.elapsed_sec,
            ))

            return result

        except Exception as e:
            import traceback
            print(f"[WF-Evo] 异常: {e}")
            traceback.print_exc()
            result.elapsed_sec = time.time() - self._start_time
            result.error_message = f"运行异常: {e}"
            return result

    def _run_optimization(self, folds) -> Tuple[Optional[np.ndarray], float]:
        """
        运行 CMA-ES (或 TPE) 优化

        Args:
            folds: 内部 WF 折列表

        Returns:
            (best_10d_vector, best_fitness) 或 (None, -inf)
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_result = {"weights": None, "fitness": -np.inf}
        trial_start_time = time.time()

        def objective(trial: optuna.Trial) -> float:
            if self._stop_requested:
                raise optuna.exceptions.OptunaError("用户停止")

            # ── 采样 8 维组权重 + 4 维阈值 ──
            group_w = np.array([
                trial.suggest_float("w_rsi", self.weight_min, self.weight_max),
                trial.suggest_float("w_macd", self.weight_min, self.weight_max),
                trial.suggest_float("w_volatility", self.weight_min, self.weight_max),
                trial.suggest_float("w_momentum", self.weight_min, self.weight_max),
                trial.suggest_float("w_volume", self.weight_min, self.weight_max),
                trial.suggest_float("w_trend", self.weight_min, self.weight_max),
                trial.suggest_float("w_structure", self.weight_min, self.weight_max),
                trial.suggest_float("w_adx", self.weight_min, self.weight_max),
            ])

            fusion_th = trial.suggest_float(
                "fusion_threshold", *self.fusion_range)
            cosine_min_th = trial.suggest_float(
                "cosine_min_threshold", *self.cosine_min_range)
            euclidean_min_th = trial.suggest_float(
                "euclidean_min_threshold", *self.euclidean_min_range)
            dtw_min_th = trial.suggest_float(
                "dtw_min_threshold", *self.dtw_min_range)

            # ── 多折评估 ──
            fold_sharpes = []
            for fold_idx, fold_df in enumerate(folds):
                if self._stop_requested:
                    raise optuna.exceptions.OptunaError("用户停止")
                try:
                    fold_result = self._evaluate_on_data(
                        fold_df, group_w, fusion_th, cosine_min_th, euclidean_min_th, dtw_min_th)

                    n_trades = fold_result.get("n_trades", 0)
                    if n_trades < self.min_trades_per_fold:
                        fold_sharpes.append(self.min_trades_penalty)
                    else:
                        fold_sharpes.append(fold_result.get("sharpe", self.min_trades_penalty))
                except Exception as e:
                    # 单 trial 或单 fold 失败不应终止整个优化过程，直接惩罚该 fold
                    self._last_error_message = f"Fold{fold_idx + 1} 评估异常: {e}"
                    fold_sharpes.append(self.min_trades_penalty)

            # ── 适应度 = 平均 Sharpe − L2 正则化 ──
            mean_sharpe = np.mean(fold_sharpes)
            l2_penalty = self.l2_lambda * np.sum(
                (group_w - self._default_group_weights) ** 2)
            fitness = mean_sharpe - l2_penalty

            # 更新最优
            if fitness > best_result["fitness"]:
                best_result["fitness"] = fitness
                best_result["weights"] = np.concatenate([
                    group_w, [fusion_th, cosine_min_th, euclidean_min_th, dtw_min_th]])

            # ── 进度回调 ──
            trial_idx = trial.number + 1
            elapsed = time.time() - trial_start_time
            if trial_idx > 0:
                eta = elapsed / trial_idx * (self.n_trials - trial_idx)
            else:
                eta = 0.0

            fold_detail = " | ".join(
                [f"F{i}={s:.2f}" for i, s in enumerate(fold_sharpes)])

            self._emit_progress(EvolutionProgress(
                trial_idx=trial_idx,
                n_trials=self.n_trials,
                best_fitness=best_result["fitness"],
                current_fitness=fitness,
                fold_detail=fold_detail,
                phase="optimizing",
                elapsed_sec=elapsed,
                eta_sec=eta,
                group_weights=best_result["weights"][:8] if best_result["weights"] is not None else None,
                fusion_threshold=fusion_th,
                cosine_min_threshold=cosine_min_th,
                euclidean_min_threshold=euclidean_min_th,
                dtw_min_threshold=dtw_min_th,
            ))

            return fitness

        # ── 创建 Optuna study ──
        use_cmaes = (self.optimizer_type == "cma-es")
        if use_cmaes:
            try:
                import cmaes  # noqa: F401
            except ModuleNotFoundError:
                print("[WF-Evo] 未安装 cmaes 包 (pip install cmaes)，回退 TPE")
                use_cmaes = False
        try:
            if use_cmaes:
                restart_strat = WF_EVOLUTION_CONFIG.get("CMA_RESTART_STRATEGY", "ipop")
                sampler = optuna.samplers.CmaEsSampler(
                    sigma0=self.cma_sigma0,
                    seed=42,
                    restart_strategy=restart_strat if restart_strat else None,
                )
            else:
                sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
        except Exception as e:
            print(f"[WF-Evo] CMA-ES 采样器创建失败 ({e}), 回退 TPE")
            sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # 添加默认权重作为初始试验
        default_trial = {
            "w_rsi": float(self._default_group_weights[0]),
            "w_macd": float(self._default_group_weights[1]),
            "w_volatility": float(self._default_group_weights[2]),
            "w_momentum": float(self._default_group_weights[3]),
            "w_volume": float(self._default_group_weights[4]),
            "w_trend": float(self._default_group_weights[5]),
            "w_structure": float(self._default_group_weights[6]),
            "w_adx": float(self._default_group_weights[7]),
            "fusion_threshold": 0.65,
            "cosine_min_threshold": 0.70,
            "euclidean_min_threshold": 0.50,
            "dtw_min_threshold": 0.40,
        }
        study.enqueue_trial(default_trial)

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=False,
            )
        except optuna.exceptions.OptunaError as e:
            if "用户停止" in str(e):
                print("[WF-Evo] 用户请求停止")
            else:
                raise
        except Exception as e:
            print(f"[WF-Evo] 优化异常: {e}")
            self._last_error_message = f"优化器异常: {e}"
            # 如果 CMA-ES 失败且允许 TPE 回退
            if self.optimizer_type == "cma-es" and self.tpe_fallback:
                print("[WF-Evo] 回退到 TPE...")
                sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
                study = optuna.create_study(
                    direction="maximize", sampler=sampler)
                study.enqueue_trial(default_trial)
                try:
                    study.optimize(
                        objective,
                        n_trials=self.n_trials,
                        show_progress_bar=False,
                    )
                except optuna.exceptions.OptunaError:
                    pass
                except Exception as e2:
                    self._last_error_message = f"TPE 回退异常: {e2}"

        return best_result["weights"], best_result["fitness"]

    def _evaluate_on_data(self, df, group_weights: np.ndarray,
                          fusion_threshold: float,
                          cosine_min_threshold: float,
                          euclidean_min_threshold: float,
                          dtw_min_threshold: float) -> Dict[str, float]:
        """
        在给定数据集上评估一组权重

        Args:
            df: 带指标的 DataFrame
            group_weights: 8 维组权重
            fusion_threshold: 融合阈值
            cosine_min_threshold: 余弦最低阈值
            euclidean_min_threshold: 欧氏距离最低阈值
            dtw_min_threshold: DTW形态最低阈值

        Returns:
            {"sharpe", "win_rate", "profit", "drawdown", "n_trades", "profit_factor"}
        """
        from core.feature_vector import FeatureVectorEngine
        from core.ga_trading_optimizer import BayesianTradingOptimizer, TradingParams
        from core.template_clusterer import MultiSimilarityCalculator

        # 展开为 32 维权重
        full_weights = MultiSimilarityCalculator.expand_group_weights(group_weights)

        # 创建特征引擎
        fv_engine = FeatureVectorEngine()
        fv_engine.precompute(df)

        # 创建贝叶斯优化器（使用原型）
        optimizer = BayesianTradingOptimizer(
            trajectory_memory=None,
            fv_engine=fv_engine,
            val_df=df,
            val_labels=None,
            prototype_library=self.prototype_library,
            n_trials=15,  # 内部参数优化用较少 trials
        )

        # 注入进化权重到默认参数
        default_params = TradingParams(
            fusion_threshold=fusion_threshold,
            cosine_threshold=cosine_min_threshold,
            feature_weights=full_weights,
        )

        # 直接用默认参数评估（不再做内部贝叶斯搜索，加速）
        # 多空分开进化：仅评估当前方向时只模拟该方向交易
        direction_filter = getattr(self, "_current_direction", None)
        result = optimizer._simulate_trading(
            default_params,
            record_templates=False,
            fast_mode=True,
            direction_filter=direction_filter,
        )

        # 计算 profit factor
        profits_arr = np.array(result.profits) if result.profits else np.array([])
        wins_sum = float(profits_arr[profits_arr > 0].sum()) if len(profits_arr) > 0 else 0.0
        losses_sum = float(abs(profits_arr[profits_arr < 0].sum())) if len(profits_arr) > 0 else 0.0
        profit_factor = wins_sum / losses_sum if losses_sum > 1e-9 else (
            10.0 if wins_sum > 0 else 0.0)

        return {
            "sharpe": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "profit": result.total_profit,
            "drawdown": result.max_drawdown,
            "n_trades": result.n_trades,
            "profit_factor": profit_factor,
        }

    def _emit_progress(self, progress: EvolutionProgress):
        """发射进度回调"""
        if self.on_progress:
            try:
                self.on_progress(progress)
            except Exception as e:
                print(f"[WF-Evo] 进度回调异常: {e}")

    # ── 持久化 ──

    def save_result(self, result: EvolutionResult) -> str:
        """
        保存进化结果到 JSON

        Args:
            result: EvolutionResult

        Returns:
            保存的文件路径
        """
        os.makedirs(self.results_dir, exist_ok=True)

        # 按方向选择保存路径（多空分开保存）
        if getattr(result, "direction", None) == "LONG":
            filepath = self.evolved_weights_file_long
        elif getattr(result, "direction", None) == "SHORT":
            filepath = self.evolved_weights_file_short
        else:
            filepath = self.evolved_weights_file

        # 转为原生 Python 类型，避免 numpy.bool_/float64 等导致 JSON 序列化失败
        data = {
            "version": "1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": result.direction,
            "group_weights": result.group_weights.tolist() if result.group_weights is not None else None,
            "full_weights": result.full_weights.tolist() if result.full_weights is not None else None,
            "fusion_threshold": float(result.fusion_threshold),
            "cosine_min_threshold": float(result.cosine_min_threshold),
            "euclidean_min_threshold": float(result.euclidean_min_threshold),
            "dtw_min_threshold": float(result.dtw_min_threshold),
            "best_fitness": float(result.best_fitness),
            "n_trials": int(result.n_trials),
            "elapsed_sec": float(result.elapsed_sec),
            "holdout": {
                "sharpe": float(result.holdout_sharpe),
                "win_rate": float(result.holdout_win_rate),
                "profit": float(result.holdout_profit),
                "drawdown": float(result.holdout_drawdown),
                "n_trades": int(result.holdout_n_trades),
                "profit_factor": float(result.holdout_profit_factor),
                "passed": bool(result.holdout_passed),
            },
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[WF-Evo] 结果已保存: {filepath}")
        return filepath

    @staticmethod
    def load_result(filepath: str) -> Optional[EvolutionResult]:
        """
        从 JSON 加载进化结果

        Args:
            filepath: JSON 文件路径

        Returns:
            EvolutionResult 或 None
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = EvolutionResult()
            result.success = True
            result.direction = data.get("direction")
            result.group_weights = np.array(data["group_weights"]) if data.get("group_weights") else None
            result.full_weights = np.array(data["full_weights"]) if data.get("full_weights") else None
            result.fusion_threshold = data.get("fusion_threshold", 0.65)
            result.cosine_min_threshold = data.get("cosine_min_threshold", 0.70)
            result.euclidean_min_threshold = data.get("euclidean_min_threshold", 0.50)
            result.dtw_min_threshold = data.get("dtw_min_threshold", 0.40)
            result.best_fitness = data.get("best_fitness", -np.inf)
            result.n_trials = data.get("n_trials", 0)
            result.elapsed_sec = data.get("elapsed_sec", 0.0)

            holdout = data.get("holdout", {})
            result.holdout_sharpe = holdout.get("sharpe", 0.0)
            result.holdout_win_rate = holdout.get("win_rate", 0.0)
            result.holdout_profit = holdout.get("profit", 0.0)
            result.holdout_drawdown = holdout.get("drawdown", 0.0)
            result.holdout_n_trades = holdout.get("n_trades", 0)
            result.holdout_profit_factor = holdout.get("profit_factor", 0.0)
            result.holdout_passed = holdout.get("passed", False)

            return result
        except Exception as e:
            print(f"[WF-Evo] 加载结果失败: {e}")
            return None
