"""
离场信号学习器 - 自适应优化每笔交易的离场决策

核心功能：
1. 学习每个原型的最优止盈距离（基于历史峰值利润）
2. 学习每种离场信号的可信度（信号触发 → 实际是好的离场吗？）
3. 提供分级响应策略（可信度越高，响应越激进）

设计原则（避免过拟合）：
- 学习 peak_profit_pct（峰值浮盈），而非实际平仓价
- 用 ATR 倍数，而非绝对百分比（适应不同波动率）
- 样本 < 10 笔时，使用先验（回测数据）
- 时间衰减，适应市场变化
- 赢和亏的交易都记录，避免幸存者偏差
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SignalPerformance:
    """
    单个离场信号的历史表现
    
    用于回答：这个信号可信吗？触发后平仓是好决策吗？
    """
    signal_name: str                    # 信号名称 (如 "momentum_decay")
    
    # 触发历史
    triggered_count: int = 0            # 触发过多少次
    success_count: int = 0              # 触发后平仓确实更好的次数
    false_alarm_count: int = 0          # 触发但不应该平（价格继续往有利方向走）
    
    # 盈亏跟踪
    total_profit_if_exit: float = 0.0   # 如果在信号触发时立即平，累计收益
    total_profit_actual: float = 0.0    # 实际平仓的累计收益
    
    last_update_time: float = 0.0       # 最后更新时间
    
    @property
    def confidence(self) -> float:
        """
        可信度 (0-1)
        
        计算方法：
        - 如果触发后平仓平均收益 > 实际平仓平均收益，则可信
        - confidence = (成功次数 + 1) / (触发次数 + 2)  # 贝叶斯平滑
        """
        if self.triggered_count == 0:
            return 0.5  # 无历史数据，中性先验
        
        # 贝叶斯平滑（+1 +2 是 Beta(1,1) 先验）
        return (self.success_count + 1) / (self.triggered_count + 2)
    
    @property
    def avg_profit_if_exit(self) -> float:
        """如果立即平，平均收益"""
        if self.triggered_count == 0:
            return 0.0
        return self.total_profit_if_exit / self.triggered_count
    
    @property
    def avg_profit_actual(self) -> float:
        """实际平仓，平均收益"""
        if self.triggered_count == 0:
            return 0.0
        return self.total_profit_actual / self.triggered_count
    
    def update(self, profit_if_exit: float, profit_actual: float):
        """
        更新信号表现
        
        Args:
            profit_if_exit: 如果在信号触发时立即平仓，能获得的收益（%）
            profit_actual: 实际平仓时获得的收益（%）
        """
        self.triggered_count += 1
        self.total_profit_if_exit += profit_if_exit
        self.total_profit_actual += profit_actual
        
        # 判断是否成功（立即平比实际平更好，或至少差不多）
        if profit_if_exit >= profit_actual * 0.95:  # 容忍 5% 差异
            self.success_count += 1
        else:
            self.false_alarm_count += 1
        
        self.last_update_time = time.time()
    
    def decay(self, factor: float = 0.95):
        """时间衰减（远期数据权重降低）"""
        self.triggered_count = max(1, int(self.triggered_count * factor))
        self.success_count = max(0, int(self.success_count * factor))
        self.false_alarm_count = max(0, int(self.false_alarm_count * factor))
        self.total_profit_if_exit *= factor
        self.total_profit_actual *= factor
    
    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "triggered_count": self.triggered_count,
            "success_count": self.success_count,
            "false_alarm_count": self.false_alarm_count,
            "total_profit_if_exit": self.total_profit_if_exit,
            "total_profit_actual": self.total_profit_actual,
            "last_update_time": self.last_update_time,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SignalPerformance':
        return cls(
            signal_name=d["signal_name"],
            triggered_count=d.get("triggered_count", 0),
            success_count=d.get("success_count", 0),
            false_alarm_count=d.get("false_alarm_count", 0),
            total_profit_if_exit=d.get("total_profit_if_exit", 0.0),
            total_profit_actual=d.get("total_profit_actual", 0.0),
            last_update_time=d.get("last_update_time", 0.0),
        )


@dataclass
class PrototypeExitLearning:
    """
    单个原型的离场学习数据
    """
    prototype_fingerprint: str          # 原型指纹
    
    # 最优止盈距离学习（基于历史峰值利润）
    peak_profit_history: List[float] = field(default_factory=list)  # 历史峰值利润 (%)
    optimal_tp_pct: float = 0.0         # 最优 TP（百分比，用于显示）
    optimal_tp_atr_multiplier: float = 0.0  # 最优 TP（ATR 倍数，用于实际计算）
    
    # 离场信号可信度学习
    signal_performances: Dict[str, SignalPerformance] = field(default_factory=dict)
    
    # 统计
    total_trades: int = 0
    last_update_time: float = 0.0
    
    def update_peak_profit(self, peak_profit_pct: float, atr_at_entry: float, entry_price: float):
        """
        记录一笔交易的峰值利润
        
        Args:
            peak_profit_pct: 峰值利润（杠杆化收益率 %）
            atr_at_entry: 入场时的 ATR
            entry_price: 入场价格
        """
        self.peak_profit_history.append(peak_profit_pct)
        self.total_trades += 1
        
        # 保留最近 100 笔
        if len(self.peak_profit_history) > 100:
            self.peak_profit_history.pop(0)
        
        # 更新最优 TP：使用 75 分位数（既有追求，又不太激进）
        if len(self.peak_profit_history) >= 10:
            self.optimal_tp_pct = float(np.percentile(self.peak_profit_history, 75))
            
            # 转换为 ATR 倍数（price_pct / leverage = atr_multiplier）
            # peak_profit_pct 是杠杆化的，需要除以 10 得到价格百分比
            price_pct = self.optimal_tp_pct / 10.0 / 100.0  # 例如 15% → 1.5% 价格移动
            atr_ratio = atr_at_entry / entry_price  # ATR 占价格的比例
            if atr_ratio > 1e-9:
                self.optimal_tp_atr_multiplier = price_pct / atr_ratio
            else:
                self.optimal_tp_atr_multiplier = 3.0  # 默认值
        
        self.last_update_time = time.time()
    
    def get_signal_performance(self, signal_name: str) -> SignalPerformance:
        """获取某个信号的表现（如果不存在则创建）"""
        if signal_name not in self.signal_performances:
            self.signal_performances[signal_name] = SignalPerformance(signal_name=signal_name)
        return self.signal_performances[signal_name]
    
    def update_signal(self, signal_name: str, profit_if_exit: float, profit_actual: float):
        """
        更新信号表现
        
        Args:
            signal_name: 信号名称
            profit_if_exit: 信号触发时的利润（%）
            profit_actual: 实际平仓时的利润（%）
        """
        perf = self.get_signal_performance(signal_name)
        perf.update(profit_if_exit, profit_actual)
        self.last_update_time = time.time()
    
    def decay(self, factor: float = 0.95):
        """时间衰减"""
        for perf in self.signal_performances.values():
            perf.decay(factor)
    
    def to_dict(self) -> dict:
        return {
            "prototype_fingerprint": self.prototype_fingerprint,
            "peak_profit_history": self.peak_profit_history,
            "optimal_tp_pct": self.optimal_tp_pct,
            "optimal_tp_atr_multiplier": self.optimal_tp_atr_multiplier,
            "signal_performances": {
                name: perf.to_dict() for name, perf in self.signal_performances.items()
            },
            "total_trades": self.total_trades,
            "last_update_time": self.last_update_time,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PrototypeExitLearning':
        return cls(
            prototype_fingerprint=d["prototype_fingerprint"],
            peak_profit_history=d.get("peak_profit_history", []),
            optimal_tp_pct=d.get("optimal_tp_pct", 0.0),
            optimal_tp_atr_multiplier=d.get("optimal_tp_atr_multiplier", 0.0),
            signal_performances={
                name: SignalPerformance.from_dict(perf_dict)
                for name, perf_dict in d.get("signal_performances", {}).items()
            },
            total_trades=d.get("total_trades", 0),
            last_update_time=d.get("last_update_time", 0.0),
        )


class ExitSignalLearner:
    """
    离场信号学习器
    
    为每个原型维护：
    1. 最优止盈距离（基于历史峰值利润统计）
    2. 各种离场信号的可信度（基于历史触发效果）
    """
    
    def __init__(self,
                 persistence_path: str = "data/exit_learning_state.json",
                 decay_enabled: bool = True,
                 decay_interval_hours: float = 48.0,
                 decay_factor: float = 0.95):
        """
        Args:
            persistence_path: 持久化文件路径
            decay_enabled: 是否启用时间衰减
            decay_interval_hours: 衰减间隔（小时）
            decay_factor: 衰减因子（0-1）
        """
        self.persistence_path = persistence_path
        self.decay_enabled = decay_enabled
        self.decay_interval_hours = decay_interval_hours
        self.decay_factor = decay_factor
        
        # 核心数据结构：key = prototype_fingerprint, value = PrototypeExitLearning
        self.prototypes: Dict[str, PrototypeExitLearning] = {}
        
        self.last_decay_time = time.time()
        
        # 加载已有数据
        self.load()
    
    def get_or_create(self, prototype_fingerprint: str) -> PrototypeExitLearning:
        """获取或创建原型的学习数据"""
        if prototype_fingerprint not in self.prototypes:
            self.prototypes[prototype_fingerprint] = PrototypeExitLearning(
                prototype_fingerprint=prototype_fingerprint
            )
        return self.prototypes[prototype_fingerprint]
    
    def record_trade_exit(self,
                          prototype_fingerprint: str,
                          peak_profit_pct: float,
                          actual_profit_pct: float,
                          atr_at_entry: float,
                          entry_price: float,
                          signals_triggered: List[Tuple[str, float]]):
        """
        记录一笔交易的离场结果
        
        Args:
            prototype_fingerprint: 原型指纹
            peak_profit_pct: 持仓期间的峰值利润（%）
            actual_profit_pct: 实际平仓利润（%）
            atr_at_entry: 入场时的 ATR
            entry_price: 入场价格
            signals_triggered: 持仓期间触发过的信号 [(signal_name, profit_at_trigger), ...]
        """
        proto_learning = self.get_or_create(prototype_fingerprint)
        
        # 1. 更新峰值利润历史
        proto_learning.update_peak_profit(peak_profit_pct, atr_at_entry, entry_price)
        
        # 2. 更新信号表现
        for signal_name, profit_if_exit in signals_triggered:
            proto_learning.update_signal(signal_name, profit_if_exit, actual_profit_pct)
        
        # 3. 自动衰减
        if self.decay_enabled:
            self._check_decay()
        
        # 4. 持久化
        self.save()
        
        print(f"[ExitLearner] 记录交易: {prototype_fingerprint} | "
              f"峰值={peak_profit_pct:.1f}% 实际={actual_profit_pct:.1f}% | "
              f"最优TP={proto_learning.optimal_tp_pct:.1f}%({proto_learning.optimal_tp_atr_multiplier:.1f}×ATR) | "
              f"信号触发={len(signals_triggered)}个")
    
    def get_optimal_tp(self, prototype_fingerprint: str, atr: float,
                       entry_price: float, leverage: float = 10) -> Tuple[float, float, str]:
        """
        获取最优止盈价格
        
        Args:
            prototype_fingerprint: 原型指纹
            atr: 当前 ATR
            entry_price: 入场价格
            leverage: 杠杆倍数
        
        Returns:
            (TP价格, TP百分比, 说明)
        """
        proto_learning = self.prototypes.get(prototype_fingerprint)
        
        # 样本不足，使用默认值
        if not proto_learning or len(proto_learning.peak_profit_history) < 10:
            # 默认：3 × ATR
            default_atr_mult = 3.0
            tp_pct = (atr * default_atr_mult / entry_price) * leverage * 100
            reason = f"样本不足(< 10笔)，使用默认 {default_atr_mult:.1f}×ATR"
            return tp_pct, default_atr_mult, reason
        
        # 使用学到的最优距离
        atr_mult = proto_learning.optimal_tp_atr_multiplier
        tp_pct = (atr * atr_mult / entry_price) * leverage * 100
        sample_count = len(proto_learning.peak_profit_history)
        reason = f"学习样本={sample_count}笔，75分位={proto_learning.optimal_tp_pct:.1f}% → {atr_mult:.1f}×ATR"
        
        return tp_pct, atr_mult, reason
    
    def get_signal_confidence(self, prototype_fingerprint: str,
                             signal_name: str) -> Tuple[float, str]:
        """
        获取某个信号对某个原型的可信度
        
        Args:
            prototype_fingerprint: 原型指纹
            signal_name: 信号名称
        
        Returns:
            (可信度 0-1, 说明)
        """
        proto_learning = self.prototypes.get(prototype_fingerprint)
        
        if not proto_learning:
            return 0.5, "无历史数据，中性先验"
        
        perf = proto_learning.signal_performances.get(signal_name)
        if not perf or perf.triggered_count == 0:
            return 0.5, "该信号未触发过，中性先验"
        
        confidence = perf.confidence
        reason = (
            f"历史触发{perf.triggered_count}次，"
            f"成功{perf.success_count}次，"
            f"虚警{perf.false_alarm_count}次 → 可信度={confidence:.1%}"
        )
        
        return confidence, reason
    
    def get_response_strategy(self, confidence: float) -> Tuple[str, str]:
        """
        根据可信度返回响应策略
        
        Args:
            confidence: 可信度 (0-1)
        
        Returns:
            (策略代码, 说明)
            策略代码:
            - "immediate_exit": 立即平仓
            - "tighten_stop": 收紧追踪止损
            - "monitor": 提示监控，但保持原策略
            - "ignore": 忽略信号
        """
        if confidence > 0.8:
            return "immediate_exit", f"高可信度({confidence:.0%})，建议立即平仓"
        elif confidence > 0.6:
            return "tighten_stop", f"中高可信度({confidence:.0%})，收紧追踪止损到成本价以上"
        elif confidence > 0.4:
            return "monitor", f"中性可信度({confidence:.0%})，提示监控但保持原策略"
        else:
            return "ignore", f"低可信度({confidence:.0%})，历史表现差，忽略此信号"
    
    def _check_decay(self):
        """检查是否需要时间衰减"""
        now = time.time()
        hours_since_decay = (now - self.last_decay_time) / 3600.0
        
        if hours_since_decay >= self.decay_interval_hours:
            print(f"[ExitLearner] 执行时间衰减 (间隔={hours_since_decay:.1f}小时, 因子={self.decay_factor})")
            for proto_fp, proto_learning in self.prototypes.items():
                old_trades = proto_learning.total_trades
                proto_learning.decay(self.decay_factor)
                if old_trades > 0:
                    print(f"  {proto_fp}: 交易{old_trades}笔 → 衰减后等效~{int(old_trades * self.decay_factor)}笔")
            self.last_decay_time = now
            self.save()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_prototypes = len(self.prototypes)
        total_trades = sum(p.total_trades for p in self.prototypes.values())
        learned_prototypes = sum(1 for p in self.prototypes.values() if len(p.peak_profit_history) >= 10)
        
        return {
            "total_prototypes": total_prototypes,
            "learned_prototypes": learned_prototypes,
            "total_trades": total_trades,
            "avg_trades_per_prototype": total_trades / max(1, total_prototypes),
        }
    
    def save(self):
        """持久化到 JSON"""
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            data = {
                "config": {
                    "decay_enabled": self.decay_enabled,
                    "decay_interval_hours": self.decay_interval_hours,
                    "decay_factor": self.decay_factor,
                },
                "state": {
                    "last_decay_time": self.last_decay_time,
                },
                "prototypes": {
                    fp: proto.to_dict() for fp, proto in self.prototypes.items()
                },
            }
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ExitLearner] 保存失败: {e}")
    
    def load(self):
        """从 JSON 加载"""
        if not os.path.exists(self.persistence_path):
            print(f"[ExitLearner] 持久化文件不存在，使用初始状态")
            return
        
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载状态
            state = data.get("state", {})
            self.last_decay_time = state.get("last_decay_time", time.time())
            
            # 加载原型数据
            protos_data = data.get("prototypes", {})
            self.prototypes = {
                fp: PrototypeExitLearning.from_dict(proto_dict)
                for fp, proto_dict in protos_data.items()
            }
            
            print(f"[ExitLearner] 加载成功: {len(self.prototypes)} 个原型, "
                  f"累计 {sum(p.total_trades for p in self.prototypes.values())} 笔交易")
        except Exception as e:
            print(f"[ExitLearner] 加载失败: {e}，使用初始状态")
