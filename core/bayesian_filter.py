"""
贝叶斯交易过滤器 - 使用 Thompson Sampling 和凯利公式
通过实战交易持续学习，动态调整每个原型的可信度和仓位
"""

import json
import os
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BetaDistribution:
    """
    Beta 分布参数（描述胜率的概率分布）+ 盈亏比跟踪
    
    α = 累计赢的次数 + 先验
    β = 累计亏的次数 + 先验
    
    预测胜率 = α / (α + β)
    方差 = αβ / [(α+β)²(α+β+1)] → 交易越多越确信
    盈亏比 = 平均赢利 / 平均亏损
    """
    alpha: float = 1.0  # 赢的次数（含先验）
    beta: float = 1.0   # 亏的次数（含先验）
    last_update_time: float = 0.0
    trade_count: int = 0  # 实盘交易次数（不含先验）
    
    # 盈亏比跟踪（用于凯利公式）
    total_win_profit: float = 0.0   # 累计赢利金额（保证金百分比）
    total_loss_amount: float = 0.0  # 累计亏损金额（保证金百分比，绝对值）
    win_count_real: int = 0         # 实际赢的次数（不含先验）
    loss_count_real: int = 0        # 实际亏的次数（不含先验）
    
    @property
    def expected_win_rate(self) -> float:
        """预测胜率（后验均值）"""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """方差（不确定性度量）"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    @property
    def std_dev(self) -> float:
        """标准差"""
        return np.sqrt(self.variance)
    
    @property
    def avg_win(self) -> float:
        """平均赢利（保证金百分比）"""
        if self.win_count_real == 0:
            return 0.0
        return self.total_win_profit / self.win_count_real
    
    @property
    def avg_loss(self) -> float:
        """平均亏损（保证金百分比，正数）"""
        if self.loss_count_real == 0:
            return 0.0
        return self.total_loss_amount / self.loss_count_real
    
    @property
    def profit_loss_ratio(self) -> float:
        """
        盈亏比（b in Kelly formula）
        = 平均赢利 / 平均亏损
        
        例：平均赢 2%，平均亏 1% → 盈亏比 = 2.0
        """
        if self.avg_loss < 1e-9:
            return 1.0  # 防止除零，默认 1:1
        return self.avg_win / self.avg_loss
    
    def thompson_sample(self) -> float:
        """
        Thompson Sampling：从后验分布采样
        用于多臂老虎机问题，自动平衡探索与利用
        """
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, is_win: bool, profit_pct: float = 0.0, weight: float = 1.0):
        """
        更新后验（新增一笔交易）
        
        Args:
            is_win: 是否盈利
            profit_pct: 盈亏百分比（基于保证金），正数=赢，负数=亏
            weight: 学习权重（翻转单成功=2.0，普通=1.0）
                    权重>1表示这笔交易对贝叶斯分布的影响更大
                    让系统从"聪明的决策"中学得更快
        """
        if is_win:
            self.alpha += weight
            self.win_count_real += 1
            self.total_win_profit += abs(profit_pct)
        else:
            self.beta += weight
            self.loss_count_real += 1
            self.total_loss_amount += abs(profit_pct)
        
        self.trade_count += 1
        self.last_update_time = time.time()
    
    def decay(self, factor: float = 0.95):
        """
        时间衰减：远期数据权重降低
        让系统能适应市场变化
        
        同时衰减盈亏跟踪数据，避免凯利公式被过时的盈亏比永久锁定
        """
        self.alpha = max(1.0, self.alpha * factor)
        self.beta = max(1.0, self.beta * factor)
        # 同步衰减盈亏跟踪（保持凯利公式的盈亏比也能适应市场变化）
        self.total_win_profit *= factor
        self.total_loss_amount *= factor
    
    def shrink_towards_prior(self, shrink_factor: float = 0.3):
        """
        将分布收缩向中性先验 Beta(1,1)
        
        用于加速稀释旧的"脏数据"（在bug修复前收集的交易结果）
        shrink_factor=0 → 完全重置为 Beta(1,1)
        shrink_factor=1 → 保持不变
        """
        self.alpha = 1.0 + (self.alpha - 1.0) * shrink_factor
        self.beta = 1.0 + (self.beta - 1.0) * shrink_factor
        # 同步收缩盈亏跟踪
        self.total_win_profit *= shrink_factor
        self.total_loss_amount *= shrink_factor
    
    def reset(self):
        """完全重置为中性先验 Beta(1,1)，清除所有实盘数据"""
        self.alpha = 1.0
        self.beta = 1.0
        self.last_update_time = time.time()
        self.trade_count = 0
        self.total_win_profit = 0.0
        self.total_loss_amount = 0.0
        self.win_count_real = 0
        self.loss_count_real = 0
    
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "last_update_time": self.last_update_time,
            "trade_count": self.trade_count,
            "total_win_profit": self.total_win_profit,
            "total_loss_amount": self.total_loss_amount,
            "win_count_real": self.win_count_real,
            "loss_count_real": self.loss_count_real,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'BetaDistribution':
        return cls(
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
            last_update_time=d.get("last_update_time", 0.0),
            trade_count=d.get("trade_count", 0),
            total_win_profit=d.get("total_win_profit", 0.0),
            total_loss_amount=d.get("total_loss_amount", 0.0),
            win_count_real=d.get("win_count_real", 0),
            loss_count_real=d.get("loss_count_real", 0),
        )


class BayesianTradeFilter:
    """
    贝叶斯交易过滤器
    
    为每个 (原型指纹 × 市场状态) 组合维护一个 Beta 分布
    通过实战交易持续学习，动态调整可信度
    """
    
    # 数据版本：用于检测旧数据并触发一次性迁移
    # v1 (无版本字段)：bug修复前收集的数据，可能包含脏数据
    # v2：修复了前3根保护期、偏离预警、市场反转检测等问题后的数据
    DATA_VERSION = 2
    
    # 迁移时对有实盘交易数据的分布收缩强度（0=完全重置，1=不变）
    MIGRATION_SHRINK_FACTOR = 0.3
    
    def __init__(self, 
                 prior_strength: float = 10.0,
                 min_win_rate_threshold: float = 0.50,
                 thompson_sampling: bool = True,
                 decay_enabled: bool = True,
                 decay_interval_hours: float = 24.0,
                 decay_factor: float = 0.95,
                 persistence_path: str = "data/bayesian_state.json"):
        """
        Args:
            prior_strength: 先验强度（历史数据权重，相当于多少笔实盘交易）
            min_win_rate_threshold: 最低胜率阈值（低于此值直接拒绝）
            thompson_sampling: 是否使用 Thompson Sampling（否则用后验均值）
            decay_enabled: 是否启用时间衰减
            decay_interval_hours: 衰减间隔（小时）
            decay_factor: 衰减因子（0-1，越小遗忘越快）
            persistence_path: 持久化文件路径
        """
        self.prior_strength = prior_strength
        self.min_win_rate_threshold = min_win_rate_threshold
        self.thompson_sampling = thompson_sampling
        self.decay_enabled = decay_enabled
        self.decay_interval_hours = decay_interval_hours
        self.decay_factor = decay_factor
        self.persistence_path = persistence_path
        
        # 核心状态：key = "原型指纹_市场状态"，value = BetaDistribution
        self.distributions: Dict[str, BetaDistribution] = {}
        
        # 衰减时间戳
        self.last_decay_time = time.time()
        # 持久化时间（用于 UI 记忆时间范围）
        self._created_at: float = 0.0
        self._last_save_time: float = 0.0
        
        # 统计
        self.total_signals_received = 0
        self.total_signals_accepted = 0
        self.total_signals_rejected = 0
        
        # 加载已有状态
        self.load()
    
    def _get_key(self, prototype_fingerprint: str, market_regime: str) -> str:
        """生成唯一键"""
        return f"{prototype_fingerprint}_{market_regime}"
    
    def initialize_from_prototype(self, 
                                   prototype_fingerprint: str,
                                   market_regime: str,
                                   historical_win_rate: float,
                                   historical_sample_count: int,
                                   historical_avg_profit_pct: float = 0.0):
        """
        用原型库的历史回测数据初始化先验
        
        Args:
            prototype_fingerprint: 原型指纹（如 "proto_LONG_3"）
            market_regime: 市场状态（如 "强空头"）
            historical_win_rate: 历史胜率（0-1）
            historical_sample_count: 历史样本数
            historical_avg_profit_pct: 历史平均收益率（用于估算盈亏比）
        """
        key = self._get_key(prototype_fingerprint, market_regime)
        
        if key in self.distributions:
            # 已有实盘数据，不覆盖
            return
        
        # 用先验强度缩放历史样本
        effective_count = min(historical_sample_count, self.prior_strength)
        
        alpha_prior = max(1.0, effective_count * historical_win_rate)
        beta_prior = max(1.0, effective_count * (1 - historical_win_rate))
        
        # 估算盈亏比先验（从 avg_profit_pct 推算）
        # 假设亏损交易平均亏 -1%，根据总平均收益反推赢利交易平均赢多少
        # avg_profit = win_rate * avg_win + (1-win_rate) * avg_loss
        # 简化估算：avg_win ≈ |avg_profit| * 2, avg_loss ≈ 1%
        avg_win_estimate = max(1.0, abs(historical_avg_profit_pct) * 2)
        avg_loss_estimate = 1.0
        
        self.distributions[key] = BetaDistribution(
            alpha=alpha_prior,
            beta=beta_prior,
            last_update_time=time.time(),
            trade_count=0,  # 实盘交易计数从 0 开始
            total_win_profit=avg_win_estimate * alpha_prior,  # 先验盈利
            total_loss_amount=avg_loss_estimate * beta_prior,  # 先验亏损
            win_count_real=0,
            loss_count_real=0,
        )
        
        print(f"[BayesianFilter] 初始化先验: {key} | "
              f"胜率={historical_win_rate:.1%} | 样本={historical_sample_count} | "
              f"Beta(α={alpha_prior:.1f}, β={beta_prior:.1f}) | "
              f"盈亏比≈{avg_win_estimate/avg_loss_estimate:.2f}")
    
    def should_trade(self, 
                     prototype_fingerprint: str,
                     market_regime: str) -> Tuple[bool, float, str]:
        """
        判断是否应该执行这个交易信号
        
        改进逻辑：
        - 实盘交易不足3笔的原型：直接放行（数据不足，Thompson采样无意义）
        - 实盘交易≥3笔的原型：正常走Thompson采样/后验均值判断
        
        Returns:
            (是否通过, 预测胜率, 原因说明)
        """
        self.total_signals_received += 1
        key = self._get_key(prototype_fingerprint, market_regime)
        
        # 如果没有先验（新原型），使用中性先验
        if key not in self.distributions:
            self.distributions[key] = BetaDistribution(
                alpha=1.0,  # 中性先验：Beta(1,1) = 均匀分布
                beta=1.0,
                last_update_time=time.time(),
                trade_count=0,
            )
            print(f"[BayesianFilter] 新原型 {key}，使用中性先验 Beta(1,1)")
        
        dist = self.distributions[key]
        
        # 【关键改进】实盘数据不足时直接放行
        # 原因：0-2笔交易的Thompson采样本质上是掷骰子，没有统计意义
        # 让系统先积累数据，有了足够样本后再让贝叶斯发挥过滤作用
        MIN_TRADES_FOR_FILTER = 3
        if dist.trade_count < MIN_TRADES_FOR_FILTER:
            self.total_signals_accepted += 1
            reason = (f"实盘数据不足({dist.trade_count}/{MIN_TRADES_FOR_FILTER}笔)，"
                      f"先验期望={dist.expected_win_rate:.1%}，直接放行积累数据")
            print(f"[BayesianFilter] ✅ {key} 数据不足({dist.trade_count}笔)，放行")
            return True, dist.expected_win_rate, reason
        
        # 实盘数据充足，正常走贝叶斯判断
        # Thompson Sampling：从后验分布采样
        if self.thompson_sampling:
            sampled_win_rate = dist.thompson_sample()
            decision = sampled_win_rate >= self.min_win_rate_threshold
            reason = f"Thompson采样={sampled_win_rate:.1%}(基于{dist.trade_count}笔实盘)"
        else:
            # 直接用后验均值
            expected_wr = dist.expected_win_rate
            decision = expected_wr >= self.min_win_rate_threshold
            reason = f"预测胜率={expected_wr:.1%}(基于{dist.trade_count}笔实盘)"
        
        if decision:
            self.total_signals_accepted += 1
            return True, dist.expected_win_rate, f"{reason}, 通过"
        else:
            self.total_signals_rejected += 1
            return False, dist.expected_win_rate, f"{reason}, 拒绝(< {self.min_win_rate_threshold:.0%})"

    def get_expected_win_rate(self, prototype_fingerprint: str, market_regime: str) -> float:
        """
        获取当前后验胜率（不触发采样/不计入门控统计）
        用于UI展示“匹配原型对应的贝叶斯概率”。
        """
        key = self._get_key(prototype_fingerprint, market_regime)
        dist = self.distributions.get(key)
        if dist is None:
            return 0.0
        return float(dist.expected_win_rate)
    
    def update_trade_result(self,
                            prototype_fingerprint: str,
                            market_regime: str,
                            is_win: bool,
                            profit_pct: float = 0.0,
                            is_flip_trade: bool = False):
        """
        更新交易结果（平仓后调用）
        
        Args:
            prototype_fingerprint: 原型指纹
            market_regime: 市场状态
            is_win: 是否盈利
            profit_pct: 盈亏百分比（用于未来凯利公式）
            is_flip_trade: 是否为翻转单（翻转单学习权重加倍）
        """
        key = self._get_key(prototype_fingerprint, market_regime)
        
        if key not in self.distributions:
            # 不应该发生，但防御性处理
            self.distributions[key] = BetaDistribution(
                alpha=1.0,
                beta=1.0,
                last_update_time=time.time(),
                trade_count=0,
            )
        
        # 翻转单学习权重：成功翻转=2.0x，失败翻转=1.5x，普通=1.0x
        # 原理：翻转单是"系统主动识别极端位置后做出的聪明决策"
        # 成功了说明系统判断对了，应该加速学习这个经验
        # 失败了也比普通失败信息量大，但权重比成功低
        if is_flip_trade:
            weight = 2.0 if is_win else 1.5
            flip_label = f"翻转单(权重={weight}x)"
        else:
            weight = 1.0
            flip_label = "普通单"
        
        dist = self.distributions[key]
        old_wr = dist.expected_win_rate
        old_plr = dist.profit_loss_ratio
        dist.update(is_win, profit_pct, weight=weight)
        new_wr = dist.expected_win_rate
        new_plr = dist.profit_loss_ratio
        
        print(f"[BayesianFilter] 更新: {key} | "
              f"{'赢' if is_win else '亏'} {profit_pct:+.2f}% [{flip_label}] | "
              f"胜率 {old_wr:.1%} → {new_wr:.1%} | "
              f"盈亏比 {old_plr:.2f} → {new_plr:.2f} | "
              f"Beta(α={dist.alpha:.1f}, β={dist.beta:.1f}) | "
              f"实盘{dist.trade_count}笔")
        
        # 自动衰减检查
        if self.decay_enabled:
            self._check_decay()
        
        # 自动持久化
        self.save()
    
    def _check_decay(self):
        """检查是否需要时间衰减"""
        now = time.time()
        hours_since_decay = (now - self.last_decay_time) / 3600.0
        
        if hours_since_decay >= self.decay_interval_hours:
            print(f"[BayesianFilter] 执行时间衰减 (间隔={hours_since_decay:.1f}小时, 因子={self.decay_factor})")
            for key, dist in self.distributions.items():
                old_wr = dist.expected_win_rate
                dist.decay(self.decay_factor)
                new_wr = dist.expected_win_rate
                if abs(new_wr - old_wr) > 0.01:
                    print(f"  {key}: {old_wr:.1%} → {new_wr:.1%}")
            self.last_decay_time = now
            self.save()
    
    def reset_distribution(self, prototype_fingerprint: str, market_regime: str) -> bool:
        """
        重置指定原型×市场状态的分布为中性先验
        
        Args:
            prototype_fingerprint: 原型指纹（如 "proto_SHORT_0"）
            market_regime: 市场状态（如 "震荡偏空"）
        
        Returns:
            是否成功重置（False 表示找不到该 key）
        """
        key = self._get_key(prototype_fingerprint, market_regime)
        if key not in self.distributions:
            print(f"[BayesianFilter] 重置失败: 找不到 {key}")
            return False
        
        dist = self.distributions[key]
        old_wr = dist.expected_win_rate
        old_trades = dist.trade_count
        dist.reset()
        
        print(f"[BayesianFilter] 已重置: {key} | "
              f"原胜率={old_wr:.1%}, 原交易数={old_trades} -> Beta(1,1)")
        self.save()
        return True
    
    def reset_all_distributions(self) -> int:
        """
        重置所有分布为中性先验（完全清空历史学习数据）
        
        Returns:
            重置的分布数量
        """
        count = len(self.distributions)
        if count == 0:
            print("[BayesianFilter] 没有分布需要重置")
            return 0
        
        for key, dist in self.distributions.items():
            dist.reset()
        
        # 重置统计
        self.total_signals_received = 0
        self.total_signals_accepted = 0
        self.total_signals_rejected = 0
        
        print(f"[BayesianFilter] 已重置全部 {count} 个分布为 Beta(1,1)")
        self.save()
        return count
    
    def _migrate_stale_data(self, loaded_version: int):
        """
        一次性迁移旧版本数据：对在 bug 修复前收集的实盘数据进行收缩
        
        策略：
        - 没有实盘交易数据（trade_count=0）的分布：仅来自回测先验，保留
        - 有实盘交易数据的分布：收缩向中性先验，稀释脏数据影响
        """
        if loaded_version >= self.DATA_VERSION:
            return  # 已经是最新版本
        
        migrated_count = 0
        print(f"[BayesianFilter] [WARNING] 检测到旧版数据 (v{loaded_version} -> v{self.DATA_VERSION})，开始迁移...")
        
        for key, dist in self.distributions.items():
            if dist.trade_count > 0:
                old_wr = dist.expected_win_rate
                old_plr = dist.profit_loss_ratio
                old_trades = dist.trade_count
                
                # 收缩分布向中性先验
                dist.shrink_towards_prior(self.MIGRATION_SHRINK_FACTOR)
                
                new_wr = dist.expected_win_rate
                print(f"  [{key}] 实盘{old_trades}笔 | "
                      f"胜率 {old_wr:.1%} -> {new_wr:.1%} | "
                      f"盈亏比 {old_plr:.2f} -> {dist.profit_loss_ratio:.2f} | "
                      f"(收缩因子={self.MIGRATION_SHRINK_FACTOR})")
                migrated_count += 1
        
        if migrated_count > 0:
            print(f"[BayesianFilter] 迁移完成: {migrated_count} 个分布已收缩，"
                  f"脏数据影响已稀释 {(1-self.MIGRATION_SHRINK_FACTOR):.0%}")
            self.save()
        else:
            print(f"[BayesianFilter] 迁移完成: 无需收缩（无旧实盘数据）")
    
    def calculate_kelly_fraction(self,
                                  prototype_fingerprint: str,
                                  market_regime: str,
                                  kelly_fraction: float = 0.25,
                                  max_position_pct: float = 0.5,
                                  min_position_pct: float = 0.05,
                                  min_sample_count: int = 3) -> Tuple[float, str]:
        """
        计算凯利仓位比例
        
        凯利公式：f* = p - (1-p) / b
        p = 胜率
        b = 盈亏比（平均赢利 / 平均亏损）
        
        实战使用分数凯利（0.25-0.5）避免波动过大
        
        Args:
            prototype_fingerprint: 原型指纹
            market_regime: 市场状态
            kelly_fraction: 凯利分数（0-1，建议 0.25-0.5）
            max_position_pct: 最大仓位比例（上限保护）
            min_position_pct: 最小仓位比例（样本不足时保守试探）
            min_sample_count: 最少样本数（少于此值用最小仓位）
        
        Returns:
            (仓位比例, 计算说明)
        """
        key = self._get_key(prototype_fingerprint, market_regime)
        
        # 如果没有数据，用最小仓位试探
        if key not in self.distributions:
            return min_position_pct, f"新原型，试探仓位 {min_position_pct:.0%}"
        
        dist = self.distributions[key]
        
        # 样本不足，用最小仓位
        if dist.trade_count < min_sample_count:
            return min_position_pct, f"样本不足({dist.trade_count}<{min_sample_count})，试探仓位 {min_position_pct:.0%}"
        
        p = dist.expected_win_rate  # 胜率
        b = dist.profit_loss_ratio  # 盈亏比
        
        # 凯利公式
        kelly_full = p - (1 - p) / b
        
        # 凯利值 <= 0 时，使用最小仓位继续学习（不拒绝交易）
        if kelly_full <= 0:
            return min_position_pct, f"凯利={kelly_full:.2%}≤0，使用最小仓位{min_position_pct:.0%}继续学习"
        
        # 分数凯利（降低波动）
        kelly_scaled = kelly_full * kelly_fraction
        
        # 上下限保护
        position_pct = np.clip(kelly_scaled, min_position_pct, max_position_pct)
        
        reason = (
            f"凯利={kelly_full:.2%}(胜率={p:.1%}, 盈亏比={b:.2f}) | "
            f"最终仓位={position_pct:.1%}"
        )
        
        return position_pct, reason
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_signals_received": self.total_signals_received,
            "total_signals_accepted": self.total_signals_accepted,
            "total_signals_rejected": self.total_signals_rejected,
            "accept_rate": self.total_signals_accepted / max(1, self.total_signals_received),
            "unique_keys": len(self.distributions),
            "total_live_trades": sum(d.trade_count for d in self.distributions.values()),
        }
    
    def get_top_performers(self, top_n: int = 10) -> list:
        """获取表现最好的原型×市场状态组合"""
        items = []
        for key, dist in self.distributions.items():
            if dist.trade_count >= 3:  # 至少 3 笔实盘交易
                items.append({
                    "key": key,
                    "win_rate": dist.expected_win_rate,
                    "trade_count": dist.trade_count,
                    "alpha": dist.alpha,
                    "beta": dist.beta,
                    "std_dev": dist.std_dev,
                })
        items.sort(key=lambda x: x["win_rate"], reverse=True)
        return items[:top_n]
    
    def save(self):
        """持久化到 JSON"""
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            now = time.time()
            if self._created_at <= 0:
                self._created_at = now
            self._last_save_time = now
            data = {
                "config": {
                    "prior_strength": self.prior_strength,
                    "min_win_rate_threshold": self.min_win_rate_threshold,
                    "thompson_sampling": self.thompson_sampling,
                    "decay_enabled": self.decay_enabled,
                    "decay_interval_hours": self.decay_interval_hours,
                    "decay_factor": self.decay_factor,
                },
                "state": {
                    "data_version": self.DATA_VERSION,
                    "created_at": self._created_at,
                    "last_save_time": self._last_save_time,
                    "last_decay_time": self.last_decay_time,
                    "total_signals_received": self.total_signals_received,
                    "total_signals_accepted": self.total_signals_accepted,
                    "total_signals_rejected": self.total_signals_rejected,
                },
                "distributions": {
                    key: dist.to_dict() for key, dist in self.distributions.items()
                },
            }
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[BayesianFilter] 保存失败: {e}")
    
    def load(self):
        """从 JSON 加载，并自动检测/迁移旧版数据"""
        if not os.path.exists(self.persistence_path):
            print(f"[BayesianFilter] 持久化文件不存在，使用初始状态")
            return
        
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载配置（可选，保持初始化参数为准）
            # config = data.get("config", {})
            
            # 加载状态
            state = data.get("state", {})
            loaded_version = state.get("data_version", 1)  # 无版本字段 = v1（旧数据）
            self._created_at = state.get("created_at", 0.0)
            self._last_save_time = state.get("last_save_time", 0.0)
            if self._created_at <= 0 and self._last_save_time > 0:
                self._created_at = self._last_save_time
            self.last_decay_time = state.get("last_decay_time", time.time())
            self.total_signals_received = state.get("total_signals_received", 0)
            self.total_signals_accepted = state.get("total_signals_accepted", 0)
            self.total_signals_rejected = state.get("total_signals_rejected", 0)
            
            # 加载分布
            dists_data = data.get("distributions", {})
            self.distributions = {
                key: BetaDistribution.from_dict(d) for key, d in dists_data.items()
            }
            
            print(f"[BayesianFilter] 加载成功: {len(self.distributions)} 个分布, "
                  f"累计 {self.total_signals_received} 个信号, "
                  f"数据版本 v{loaded_version}")
            
            # 自动迁移旧版数据（稀释 bug 修复前的脏数据）
            if loaded_version < self.DATA_VERSION:
                self._migrate_stale_data(loaded_version)
                
        except Exception as e:
            print(f"[BayesianFilter] 加载失败: {e}，使用初始状态")
