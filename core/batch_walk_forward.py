"""
R3000 批量 Walk-Forward 验证引擎

核心思路：
  直接用全局指纹模板库（而非每折重新提取），在多轮随机数据段上验证，
  快速、自动地找出"经得起考验"的有价值模板。

流程（每轮）：
  1. 从完整数据集随机采样一段（如 50000 根K线）
  2. 前半段做验证集（贝叶斯优化匹配参数），后半段做测试集
  3. 用全局 7000+ 模板去匹配测试集
  4. 记录哪些模板被匹配到，以及该交易是否盈利
  5. 跨轮累积评估结果

优势：
  - 每轮不需要重新标注、重新提取模板（省掉最耗时的步骤）
  - 一键执行 N 轮，自动化
  - 实时累积计数器，看得到进展
"""

import numpy as np
import time
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass, field
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAJECTORY_CONFIG, WALK_FORWARD_CONFIG


@dataclass
class BatchRoundResult:
    """单轮验证结果"""
    round_idx: int = 0
    data_start: int = 0
    data_end: int = 0

    # 贝叶斯优化结果
    best_sharpe: float = 0.0
    best_params: object = None

    # 测试集统计
    test_n_trades: int = 0
    test_n_wins: int = 0
    test_total_profit: float = 0.0
    test_win_rate: float = 0.0
    test_sharpe: float = 0.0

    # 本轮匹配到的模板指纹 → 收益
    template_matches: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class BatchWalkForwardResult:
    """批量Walk-Forward总结果"""
    n_rounds: int = 0
    completed_rounds: int = 0
    total_elapsed: float = 0.0

    # 各轮结果
    round_results: List[BatchRoundResult] = field(default_factory=list)

    # 累计统计
    total_match_events: int = 0  # 总匹配次数
    unique_templates_matched: int = 0  # 被匹配过的唯一模板数

    # 验证结果（最终评估）
    verified_long: int = 0
    verified_short: int = 0
    excellent_count: int = 0
    qualified_count: int = 0
    pending_count: int = 0
    eliminated_count: int = 0


class BatchWalkForwardEngine:
    """
    批量 Walk-Forward 验证引擎

    用法（模板模式）：
        engine = BatchWalkForwardEngine(
            data_loader=loader,
            global_memory=memory,
            n_rounds=30,
            sample_size=50000,
        )
        result = engine.run(callback=my_progress_fn)
    
    用法（原型模式，推荐）：
        engine = BatchWalkForwardEngine(
            data_loader=loader,
            global_memory=None,  # 不需要模板库
            prototype_library=proto_lib,  # 使用原型库
            n_rounds=30,
            sample_size=50000,
        )
        result = engine.run(callback=my_progress_fn)
    """

    def __init__(self,
                 data_loader,
                 global_memory,
                 n_rounds: int = 30,
                 sample_size: int = 50000,
                 n_trials: int = 20,
                 val_ratio: float = 0.5,
                 round_workers: int = 1,
                 prototype_library=None):
        """
        Args:
            data_loader: DataLoader 实例（已加载完整数据）
            global_memory: TrajectoryMemory 全局指纹模板库（原型模式下可为None）
            n_rounds: 验证轮数
            sample_size: 每轮采样K线数
            n_trials: 每轮贝叶斯优化评估次数（越少越快）
            val_ratio: val/test 分割比例（0.5 = 各占一半）
            round_workers: 并行轮次工作线程数（1=串行）
            prototype_library: PrototypeLibrary 实例（可选，提供时使用原型匹配）
        """
        self.data_loader = data_loader
        self.global_memory = global_memory
        self.n_rounds = n_rounds
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.val_ratio = val_ratio
        self.round_workers = max(1, int(round_workers))
        
        # 原型库支持
        self.prototype_library = prototype_library
        self.use_prototypes = prototype_library is not None

        # 累积评估器
        self._evaluator = None
        self._prototype_stats = {}
        self._stopped = False
        
        # 【新增】缓存完整数据集，避免每轮重复加载
        self._cached_full_df = None

    def _sample_continuous_local(self, total_rows: int, round_idx: int):
        """
        线程安全的本地采样（不修改DataLoader内部状态）
        """
        # 【优化】使用缓存数据，避免重复加载
        if self._cached_full_df is None:
            print(f"[BatchWF] 首次加载完整数据集...")
            self._cached_full_df = self.data_loader.load_full_data()
            print(f"[BatchWF] 数据加载完成: {len(self._cached_full_df)} 根K线")
        
        full_df = self._cached_full_df
        warmup = getattr(self.data_loader, "warmup_bars", 200)
        sample_size = self.sample_size
        max_sample = total_rows - warmup
        if sample_size > max_sample:
            sample_size = max_sample
        max_start = total_rows - sample_size
        rng = np.random.default_rng(round_idx * 7919 + 42)
        start_idx = int(rng.integers(warmup, max_start))
        end_idx = start_idx + sample_size
        sample_df = full_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        return sample_df, start_idx, end_idx

    def stop(self):
        """外部调用以停止批量验证"""
        self._stopped = True

    def run(self, callback: Callable = None) -> BatchWalkForwardResult:
        """
        运行批量 Walk-Forward 验证

        Args:
            callback: 进度回调
                fn(round_idx, n_rounds, round_result, cumulative_stats)
                cumulative_stats = {
                    "total_match_events": int,
                    "unique_matched": int,
                    "verified_long": int,
                    "verified_short": int,
                    "excellent": int,
                    "qualified": int,
                    "pending": int,
                    "eliminated": int,
                    "elapsed": float,
                    "eta_seconds": float,
                }

        Returns:
            BatchWalkForwardResult
        """
        from core.template_evaluator import TemplateEvaluator

        # 初始化评估器（模板模式）或原型统计（原型模式）
        if self.use_prototypes:
            self._evaluator = None
            self._prototype_stats = {}
        else:
            self._evaluator = TemplateEvaluator(self.global_memory)
        self._stopped = False

        result = BatchWalkForwardResult(n_rounds=self.n_rounds)
        start_time = time.time()
        round_progress = {i: 0.0 for i in range(self.n_rounds)}  # 每轮 0~1 的执行进度
        from threading import Lock
        progress_lock = Lock()

        # 确保数据已加载
        self.data_loader.load_full_data()
        total_rows = self.data_loader.total_rows

        if self.use_prototypes:
            proto_count = self.prototype_library.total_count if self.prototype_library is not None else 0
            source_info = f"原型库={proto_count}个"
        else:
            source_info = f"模板库={self.global_memory.total_count}个"
        print(f"[BatchWF] 启动批量Walk-Forward: {self.n_rounds}轮, "
              f"每轮{self.sample_size}根K线, "
              f"{source_info}")
        
        def current_global_progress_pct() -> float:
            if self.n_rounds <= 0:
                return 0.0
            with progress_lock:
                return (sum(round_progress.values()) / self.n_rounds) * 100.0

        def apply_round_result(round_idx: int, round_result: BatchRoundResult, round_elapsed: float):
            """统一处理单轮结果（支持串行/并行）"""
            with progress_lock:
                round_progress[round_idx] = 1.0
            
            result.round_results.append(round_result)
            result.completed_rounds += 1
            result.total_match_events += len(round_result.template_matches)
            
            # 中间评估：模板模式走 TemplateEvaluator；原型模式走本地统计
            min_matches = WALK_FORWARD_CONFIG.get("EVAL_MIN_MATCHES", 3)
            min_win_rate = WALK_FORWARD_CONFIG.get("EVAL_MIN_WIN_RATE", 0.6)
            if self.use_prototypes:
                for fp, profit in round_result.template_matches:
                    stat = self._prototype_stats.setdefault(fp, {
                        "match_count": 0,
                        "win_count": 0,
                        "total_profit": 0.0,
                        "direction": self._parse_direction_from_fp(fp),
                    })
                    stat["match_count"] += 1
                    stat["total_profit"] += float(profit)
                    if profit > 0:
                        stat["win_count"] += 1

                excellent = 0
                qualified = 0
                pending = 0
                eliminated = 0
                verified_long = 0
                verified_short = 0

                for stat in self._prototype_stats.values():
                    match_count = stat["match_count"]
                    win_rate = stat["win_count"] / match_count if match_count > 0 else 0.0
                    avg_profit = stat["total_profit"] / match_count if match_count > 0 else 0.0

                    if match_count < min_matches:
                        grade = "待观察"
                        pending += 1
                    elif avg_profit > 0 and WALK_FORWARD_CONFIG.get("EVAL_USE_EXPECTED_PROFIT", True):
                        # 【核心改进】期望收益为正 → 正期望系统 → 合格
                        # 即使胜率仅45%，只要盈亏比足够（赚多亏少），仍是好策略
                        grade = "合格"
                        qualified += 1
                    elif win_rate >= min_win_rate and avg_profit >= 0.0:
                        # 备用：纯胜率模式（当 EVAL_USE_EXPECTED_PROFIT=False 时）
                        grade = "合格"
                        qualified += 1
                    else:
                        grade = "淘汰"
                        eliminated += 1

                    if grade in ("优质", "合格", "待观察"):
                        if stat["direction"] == "SHORT":
                            verified_short += 1
                        else:
                            verified_long += 1

                result.verified_long = verified_long
                result.verified_short = verified_short
                result.excellent_count = excellent
                result.qualified_count = qualified
                result.pending_count = pending
                result.eliminated_count = eliminated
                result.unique_templates_matched = len(self._prototype_stats)
            else:
                # 先把本轮模板匹配写入评估器（线程安全：仅主线程调用）
                for fp, profit in round_result.template_matches:
                    self._evaluator.record_match(fp, profit)

                interim_eval = self._evaluator.evaluate(
                    min_matches=min_matches,
                    min_win_rate=min_win_rate,
                )
                
                verified_long = 0
                verified_short = 0
                for perf in interim_eval.performances:
                    if perf.grade in ("优质", "合格", "待观察"):
                        if perf.template.direction == "LONG":
                            verified_long += 1
                        else:
                            verified_short += 1
                
                result.verified_long = verified_long
                result.verified_short = verified_short
                result.excellent_count = interim_eval.excellent_count
                result.qualified_count = interim_eval.qualified_count
                result.pending_count = interim_eval.pending_count
                result.eliminated_count = interim_eval.eliminated_count
                result.unique_templates_matched = interim_eval.evaluated_templates
            
            elapsed = time.time() - start_time
            avg_per_round = elapsed / max(1, result.completed_rounds)
            eta_seconds = avg_per_round * (self.n_rounds - result.completed_rounds)
            
            print(f"[BatchWF] Round {round_idx + 1}/{self.n_rounds} "
                  f"({round_elapsed:.1f}s) | "
                  f"本轮匹配={len(round_result.template_matches)}, "
                  f"累计={result.total_match_events} | "
                  f"已验证: L={verified_long} S={verified_short} | "
                  f"优质={result.excellent_count} "
                  f"合格={result.qualified_count}")
            
            if callback:
                cumulative_stats = {
                    "total_match_events": result.total_match_events,
                    "unique_matched": result.unique_templates_matched,
                    "verified_long": verified_long,
                    "verified_short": verified_short,
                    "excellent": result.excellent_count,
                    "qualified": result.qualified_count,
                    "pending": result.pending_count,
                    "eliminated": result.eliminated_count,
                    "elapsed": elapsed,
                    "eta_seconds": eta_seconds,
                    "round_trades": round_result.test_n_trades,
                    "round_sharpe": round_result.test_sharpe,
                    "global_progress_pct": current_global_progress_pct(),
                }
                # 用“已完成轮次”驱动UI百分比，避免并行乱序导致进度跳动
                callback(max(0, result.completed_rounds - 1), self.n_rounds, round_result, cumulative_stats)
        
        # 串行模式：逐轮执行，支持 trial 级细粒度进度
        # 注：CPU密集计算在Python GIL下无法通过多线程加速，
        #     并行模式反而引起UI进度显示混乱和锁竞争，故统一使用串行。
        if True:
            for round_idx in range(self.n_rounds):
                if self._stopped:
                    print(f"[BatchWF] 已停止 (完成 {round_idx} 轮)")
                    break
                
                round_start = time.time()
                if self._evaluator is not None:
                    self._evaluator.refresh_from_memory()
                
                if callback:
                    # 标记该轮已进入执行（避免长时间停在0%）
                    round_progress[round_idx] = max(round_progress[round_idx], 0.01)
                    cumulative_stats = {
                        "total_match_events": result.total_match_events,
                        "unique_matched": result.unique_templates_matched,
                        "verified_long": result.verified_long,
                        "verified_short": result.verified_short,
                        "excellent": result.excellent_count,
                        "qualified": result.qualified_count,
                        "pending": result.pending_count,
                        "eliminated": result.eliminated_count,
                        "eta_seconds": 0,
                        "running": True,
                        "global_progress_pct": current_global_progress_pct(),
                    }
                    callback(round_idx, self.n_rounds, None, cumulative_stats)
                
                def trial_progress_cb(trial_idx, trial_total, _):
                    if not callback:
                        return
                    # 【优化】降低信号发射频率，避免UI阻塞（每5个trial或最后一个trial才发送）
                    if trial_idx % 5 != 0 and trial_idx != trial_total:
                        return
                    
                    frac = min(1.0, max(0.0, trial_idx / max(1, trial_total)))
                    # 5% 启动 + 90% 贝叶斯优化 + 5% 收尾（在apply_round_result里补齐）
                    round_progress[round_idx] = max(round_progress[round_idx], 0.05 + 0.90 * frac)
                    elapsed = time.time() - start_time
                    cumulative_stats = {
                        "total_match_events": result.total_match_events,
                        "unique_matched": result.unique_templates_matched,
                        "verified_long": result.verified_long,
                        "verified_short": result.verified_short,
                        "excellent": result.excellent_count,
                        "qualified": result.qualified_count,
                        "pending": result.pending_count,
                        "eliminated": result.eliminated_count,
                        "eta_seconds": 0,
                        "running": True,
                        "phase": "bayes_opt",
                        "trial_idx": trial_idx,
                        "trial_total": trial_total,
                        "elapsed": elapsed,
                        "global_progress_pct": current_global_progress_pct(),
                    }
                    callback(round_idx, self.n_rounds, None, cumulative_stats)
                
                round_result = self._run_single_round(
                    round_idx, total_rows, round_progress_callback=trial_progress_cb
                )
                apply_round_result(round_idx, round_result, time.time() - round_start)
        else:
            # 并行模式：并行round（启用线程安全trial级进度）
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from threading import Lock
            workers = min(self.round_workers, self.n_rounds)
            print(f"[BatchWF] 并行模式: workers={workers}")
            progress_lock = Lock()
            
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_meta = {}
                
                def make_parallel_trial_cb(ridx: int):
                    def _cb(trial_idx, trial_total, _):
                        if not callback:
                            return
                        with progress_lock:
                            frac = min(1.0, max(0.0, trial_idx / max(1, trial_total)))
                            round_progress[ridx] = max(round_progress[ridx], 0.05 + 0.90 * frac)
                            elapsed = time.time() - start_time
                            cumulative_stats = {
                                "total_match_events": result.total_match_events,
                                "unique_matched": result.unique_templates_matched,
                                "verified_long": result.verified_long,
                                "verified_short": result.verified_short,
                                "excellent": result.excellent_count,
                                "qualified": result.qualified_count,
                                "pending": result.pending_count,
                                "eliminated": result.eliminated_count,
                                "eta_seconds": 0,
                                "running": True,
                                "phase": "bayes_opt",
                                "trial_idx": trial_idx,
                                "trial_total": trial_total,
                                "elapsed": elapsed,
                                "global_progress_pct": current_global_progress_pct(),
                            }
                            callback(ridx, self.n_rounds, None, cumulative_stats)
                    return _cb
                
                for round_idx in range(self.n_rounds):
                    if self._stopped:
                        break
                    if self._evaluator is not None:
                        self._evaluator.refresh_from_memory()
                    start_ts = time.time()
                    fut = ex.submit(
                        self._run_single_round,
                        round_idx,
                        total_rows,
                        make_parallel_trial_cb(round_idx),
                    )
                    future_to_meta[fut] = (round_idx, start_ts)
                    if callback:
                        with progress_lock:
                            round_progress[round_idx] = max(round_progress[round_idx], 0.01)
                        cumulative_stats = {
                            "total_match_events": result.total_match_events,
                            "unique_matched": result.unique_templates_matched,
                            "verified_long": result.verified_long,
                            "verified_short": result.verified_short,
                            "excellent": result.excellent_count,
                            "qualified": result.qualified_count,
                            "pending": result.pending_count,
                            "eliminated": result.eliminated_count,
                            "eta_seconds": 0,
                            "running": True,
                            "global_progress_pct": current_global_progress_pct(),
                        }
                        callback(round_idx, self.n_rounds, None, cumulative_stats)
                
                for fut in as_completed(future_to_meta):
                    if self._stopped:
                        break
                    round_idx, start_ts = future_to_meta[fut]
                    round_result = fut.result()
                    apply_round_result(round_idx, round_result, time.time() - start_ts)

        # ── 最终评估 ──
        result.total_elapsed = time.time() - start_time

        print(f"\n[BatchWF] 批量验证完成: "
              f"{result.completed_rounds}轮, "
              f"耗时{result.total_elapsed:.1f}s")
        print(f"[BatchWF] 累计匹配事件: {result.total_match_events}")
        print(f"[BatchWF] 已验证模板: LONG={result.verified_long}, "
              f"SHORT={result.verified_short}")
        print(f"[BatchWF] 评级: 优质={result.excellent_count}, "
              f"合格={result.qualified_count}, "
              f"待观察={result.pending_count}, "
              f"淘汰={result.eliminated_count}")

        return result

    def get_evaluation_result(self):
        """获取当前累积的评估结果（供外部使用）"""
        # 原型模式暂不返回模板评估对象（UI避免误当模板筛选）
        if self._evaluator is None:
            return None

        min_matches = WALK_FORWARD_CONFIG.get("EVAL_MIN_MATCHES", 3)
        min_win_rate = WALK_FORWARD_CONFIG.get("EVAL_MIN_WIN_RATE", 0.6)
        return self._evaluator.evaluate(
            min_matches=min_matches,
            min_win_rate=min_win_rate,
        )

    def get_verified_prototype_fingerprints(self) -> set:
        """
        获取原型模式下"可保留"的原型指纹集合（去重）。
        口径与模板评估一致：优质/合格/待观察都保留，仅淘汰移除。
        """
        if not self.use_prototypes:
            return set()

        min_matches = WALK_FORWARD_CONFIG.get("EVAL_MIN_MATCHES", 3)
        min_win_rate = WALK_FORWARD_CONFIG.get("EVAL_MIN_WIN_RATE", 0.6)
        keep = set()
        for fp, stat in self._prototype_stats.items():
            match_count = stat.get("match_count", 0)
            win_count = stat.get("win_count", 0)
            total_profit = stat.get("total_profit", 0.0)
            if match_count <= 0:
                continue
            win_rate = win_count / match_count
            avg_profit = total_profit / match_count
            use_expected = WALK_FORWARD_CONFIG.get("EVAL_USE_EXPECTED_PROFIT", True)
            if match_count < min_matches:
                grade = "待观察"
            elif use_expected and avg_profit > 0:
                grade = "合格"
            elif (not use_expected) and win_rate >= min_win_rate and avg_profit >= 0.0:
                grade = "合格"
            else:
                grade = "淘汰"
            if grade in ("优质", "合格", "待观察"):
                keep.add(fp)
        return keep

    def get_prototype_stats(self) -> dict:
        """获取原型模式下的原型统计数据（供 PrototypeLibrary.apply_wf_verification 使用）"""
        return dict(self._prototype_stats)


    @staticmethod
    def _parse_direction_from_fp(fp: str) -> str:
        """
        从原型指纹字符串解析方向。
        支持:
          - proto_LONG_12
          - proto_SHORT_3
        """
        if not fp:
            return "LONG"
        m = re.match(r"^proto_(LONG|SHORT)_\d+$", fp)
        if m:
            return m.group(1)
        return "LONG"

    def _run_single_round(self, round_idx: int, total_rows: int,
                          round_progress_callback: Optional[Callable] = None) -> BatchRoundResult:
        """
        运行单轮验证

        Args:
            round_idx: 轮次索引
            total_rows: 完整数据集行数

        Returns:
            BatchRoundResult
        """
        from core.feature_vector import FeatureVectorEngine
        from core.ga_trading_optimizer import BayesianTradingOptimizer, TradingParams
        from utils.indicators import calculate_all_indicators

        # 【诊断日志】开始轮次
        print(f"\n[BatchWF] ========== Round {round_idx + 1} 开始 ==========")
        
        round_result = BatchRoundResult(round_idx=round_idx)

        # ── 1. 线程安全采样一段数据（每轮确定性但不同） ──
        sample_df, s_idx, e_idx = self._sample_continuous_local(total_rows, round_idx)
        round_result.data_start = s_idx
        round_result.data_end = e_idx

        # ── 2. 指标计算项（已移除冗余计算） ──
        # 【性能优化】基准数据集 self.full_df 已经预计算过指标，此处只需 copy

        # ── 3. 分割 val / test ──
        n = len(sample_df)
        split_idx = int(n * self.val_ratio)

        val_df = sample_df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = sample_df.iloc[split_idx:].copy().reset_index(drop=True)

        # ── 4. 构建特征引擎 ──
        val_fv = FeatureVectorEngine()
        val_fv.precompute(val_df)
        
        test_fv = FeatureVectorEngine()
        test_fv.precompute(test_df)
        print(f"[BatchWF] Round {round_idx + 1}: 特征引擎构建完成")

        # ── 5. 贝叶斯优化匹配参数（在 val 上） ──
        # 原型模式：使用 PrototypeLibrary（速度快 200 倍）
        # 模板模式：使用 global_memory（兼容旧逻辑）
        print(f"[BatchWF] Round {round_idx + 1}: 开始贝叶斯优化 ({self.n_trials} trials)...")
        val_optimizer = BayesianTradingOptimizer(
            trajectory_memory=self.global_memory,
            fv_engine=val_fv,
            val_df=val_df,
            val_labels=None,  # 不需要标签
            regime_classifier=None,  # 无市场状态分类
            callback=None,  # 单轮内不需要细粒度回调
            n_trials=self.n_trials,
            prototype_library=self.prototype_library,  # 原型库（如有）
        )

        best_params, best_sharpe = val_optimizer.run(
            sub_callback=round_progress_callback
        )
        round_result.best_sharpe = best_sharpe
        round_result.best_params = best_params
        print(f"[BatchWF] Round {round_idx + 1}: 贝叶斯优化完成 (Sharpe={best_sharpe:.3f})")

        # ── 6. 在 test 上用最优参数模拟交易 ──
        print(f"[BatchWF] Round {round_idx + 1}: 测试集交易模拟...")
        test_optimizer = BayesianTradingOptimizer(
            trajectory_memory=self.global_memory,
            fv_engine=test_fv,
            val_df=test_df,
            val_labels=None,
            regime_classifier=None,
            prototype_library=self.prototype_library,  # 原型库（如有）
        )

        test_result = test_optimizer._simulate_trading(
            best_params,
            record_templates=True,  # 记录模板/原型匹配信息
            fast_mode=False,  # 精确模式
        )

        # ── 7. 收集结果 ──
        round_result.test_n_trades = test_result.n_trades
        round_result.test_n_wins = test_result.n_wins
        round_result.test_total_profit = test_result.total_profit
        round_result.test_win_rate = test_result.win_rate
        round_result.test_sharpe = test_result.sharpe_ratio

        # ── 8. 收集模板匹配（由run统一写入评估器，支持并行）──
        for trade in test_result.trades:
            if trade.template_fingerprint:
                round_result.template_matches.append(
                    (trade.template_fingerprint, trade.profit_pct)
                )

        print(f"[BatchWF] Round {round_idx + 1}: 完成 | "
              f"交易={test_result.n_trades}, "
              f"胜率={test_result.win_rate:.1%}, "
              f"匹配模板={len(round_result.template_matches)}")
        
        return round_result
