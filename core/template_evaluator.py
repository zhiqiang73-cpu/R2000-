"""
R3000 模板评估器
统计每个模板在Walk-Forward测试中的表现，评级并筛选优质模板

评估指标：
  - 匹配次数：模板被匹配到的次数
  - 胜率：匹配后盈利交易的比例
  - 平均收益：匹配后交易的平均收益率
  - 累计收益：所有匹配交易的总收益率

筛选标准：
  - 最小匹配次数（默认 >= 3）
  - 最小胜率（默认 >= 60%）
  - 平均收益为正
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.trajectory_engine import TrajectoryTemplate, TrajectoryMemory


@dataclass
class TemplatePerformance:
    """单个模板的表现统计"""
    fingerprint: str
    template: TrajectoryTemplate
    global_idx: int  # 在记忆库中的全局索引
    
    # 匹配统计
    match_count: int = 0          # 被匹配次数
    win_count: int = 0            # 盈利次数
    loss_count: int = 0           # 亏损次数
    
    # 收益统计
    total_profit: float = 0.0     # 累计收益率
    profits: List[float] = field(default_factory=list)  # 每次匹配的收益率
    
    # 评级
    grade: str = "未评估"         # "优质" / "合格" / "待观察" / "淘汰"
    score: float = 0.0            # 综合评分 (0~100)
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.match_count == 0:
            return 0.0
        return self.win_count / self.match_count
    
    @property
    def avg_profit(self) -> float:
        """平均收益率"""
        if not self.profits:
            return 0.0
        return float(np.mean(self.profits))
    
    @property
    def max_profit(self) -> float:
        """最大单笔收益"""
        return max(self.profits) if self.profits else 0.0
    
    @property
    def max_loss(self) -> float:
        """最大单笔亏损"""
        return min(self.profits) if self.profits else 0.0
    
    def add_match(self, profit_pct: float):
        """记录一次匹配结果"""
        self.match_count += 1
        self.profits.append(profit_pct)
        self.total_profit += profit_pct
        if profit_pct > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
    
    def calculate_grade(self, min_matches: int = 3, min_win_rate: float = 0.6,
                        min_avg_profit: float = 0.0) -> str:
        """
        计算模板评级
        
        Args:
            min_matches: 最小匹配次数要求
            min_win_rate: 最小胜率要求 (0~1)
            min_avg_profit: 最小平均收益要求
        
        Returns:
            评级: "优质" / "合格" / "待观察" / "淘汰"
        """
        # 匹配次数不足 → 待观察
        if self.match_count < min_matches:
            self.grade = "待观察"
            self.score = 20.0 + self.match_count * 5  # 有一些匹配但不够
            return self.grade
        
        # 胜率和收益都达标 → 优质
        if self.win_rate >= min_win_rate and self.avg_profit >= min_avg_profit:
            # 计算评分：基础60分 + 胜率加成 + 收益加成
            base_score = 60.0
            win_rate_bonus = (self.win_rate - min_win_rate) * 100  # 胜率超标部分
            profit_bonus = min(20.0, self.avg_profit * 10)  # 收益加成，上限20分
            match_bonus = min(10.0, (self.match_count - min_matches) * 2)  # 匹配次数加成
            
            self.score = min(100.0, base_score + win_rate_bonus + profit_bonus + match_bonus)
            
            if self.score >= 80:
                self.grade = "优质"
            else:
                self.grade = "合格"
            return self.grade
        
        # 胜率或收益不达标 → 淘汰
        self.grade = "淘汰"
        self.score = max(0.0, self.win_rate * 50 + self.avg_profit * 5)
        return self.grade
    
    def to_dict(self) -> dict:
        """转换为字典（用于显示）"""
        return {
            "fingerprint": self.fingerprint[:8] + "...",  # 截断显示
            "direction": self.template.direction,
            "regime": self.template.regime,
            "match_count": self.match_count,
            "win_rate": f"{self.win_rate:.1%}",
            "avg_profit": f"{self.avg_profit:.2f}%",
            "total_profit": f"{self.total_profit:.2f}%",
            "grade": self.grade,
            "score": f"{self.score:.1f}",
        }


@dataclass
class EvaluationResult:
    """评估结果汇总"""
    performances: List[TemplatePerformance] = field(default_factory=list)
    
    # 统计
    total_templates: int = 0
    evaluated_templates: int = 0  # 至少被匹配1次的模板
    excellent_count: int = 0      # 优质
    qualified_count: int = 0      # 合格
    pending_count: int = 0        # 待观察
    eliminated_count: int = 0     # 淘汰
    
    # 筛选后的指纹集合
    keep_fingerprints: Set[str] = field(default_factory=set)
    remove_fingerprints: Set[str] = field(default_factory=set)
    
    def summarize(self, min_matches: int = 3, min_win_rate: float = 0.6):
        """计算汇总统计和筛选结果"""
        self.total_templates = len(self.performances)
        self.evaluated_templates = sum(1 for p in self.performances if p.match_count > 0)
        
        self.excellent_count = 0
        self.qualified_count = 0
        self.pending_count = 0
        self.eliminated_count = 0
        
        self.keep_fingerprints = set()
        self.remove_fingerprints = set()
        
        for p in self.performances:
            p.calculate_grade(min_matches=min_matches, min_win_rate=min_win_rate)
            
            if p.grade == "优质":
                self.excellent_count += 1
                self.keep_fingerprints.add(p.fingerprint)
            elif p.grade == "合格":
                self.qualified_count += 1
                self.keep_fingerprints.add(p.fingerprint)
            elif p.grade == "待观察":
                self.pending_count += 1
                # 待观察的保留，继续收集数据
                self.keep_fingerprints.add(p.fingerprint)
            else:  # 淘汰
                self.eliminated_count += 1
                self.remove_fingerprints.add(p.fingerprint)
        
        # 按评分排序
        self.performances.sort(key=lambda p: p.score, reverse=True)
    
    def get_summary_text(self) -> str:
        """生成摘要文本"""
        lines = [
            f"模板评估结果:",
            f"  总模板数: {self.total_templates}",
            f"  被匹配过: {self.evaluated_templates}",
            f"  ---------------------",
            f"  优质: {self.excellent_count}",
            f"  合格: {self.qualified_count}",
            f"  待观察: {self.pending_count}",
            f"  淘汰: {self.eliminated_count}",
            f"  ---------------------",
            f"  建议保留: {len(self.keep_fingerprints)}",
            f"  建议删除: {len(self.remove_fingerprints)}",
        ]
        return "\n".join(lines)
    
    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 50)
        print(self.get_summary_text())
        print("=" * 50)
        
        # 打印Top 10优质模板
        print("\nTop 10 优质模板:")
        for i, p in enumerate(self.performances[:10]):
            print(f"  {i+1}. {p.template.direction}/{p.template.regime}: "
                  f"匹配{p.match_count}次, 胜率{p.win_rate:.1%}, "
                  f"均利{p.avg_profit:.2f}%, 评分{p.score:.1f}")


class TemplateEvaluator:
    """
    模板评估器
    
    在Walk-Forward过程中收集模板匹配数据，
    评估每个模板的表现并筛选优质模板。
    
    使用方式：
        evaluator = TemplateEvaluator(memory)
        # 在每次匹配成功时记录
        evaluator.record_match(template_fingerprint, trade_profit)
        # 完成后评估
        result = evaluator.evaluate()
        # 应用筛选
        memory.filter_by_fingerprints(result.keep_fingerprints)
    """
    
    def __init__(self, memory: TrajectoryMemory):
        """
        Args:
            memory: 要评估的模板记忆库
        """
        self.memory = memory
        
        # 初始化所有模板的性能追踪
        self._performances: Dict[str, TemplatePerformance] = {}
        self._init_performances()
    
    def _init_performances(self):
        """初始化所有模板的性能记录"""
        all_templates = self.memory.get_all_templates()
        
        for i, template in enumerate(all_templates):
            fp = template.fingerprint()
            if fp not in self._performances:
                self._performances[fp] = TemplatePerformance(
                    fingerprint=fp,
                    template=template,
                    global_idx=i,
                )
    
    def refresh_from_memory(self):
        """
        从记忆库刷新模板列表（支持动态增长的记忆库）
        
        仅添加新模板的性能记录，不影响已有记录的匹配数据。
        """
        all_templates = self.memory.get_all_templates()
        new_count = 0
        for i, template in enumerate(all_templates):
            fp = template.fingerprint()
            if fp not in self._performances:
                self._performances[fp] = TemplatePerformance(
                    fingerprint=fp,
                    template=template,
                    global_idx=i,
                )
                new_count += 1
        if new_count > 0:
            print(f"[TemplateEvaluator] 刷新: 新增 {new_count} 个模板追踪 "
                  f"(总计 {len(self._performances)} 个)")

    def record_match(self, template_fingerprint: str, profit_pct: float):
        """
        记录一次模板匹配结果
        
        Args:
            template_fingerprint: 模板指纹
            profit_pct: 这笔交易的收益率
        """
        if template_fingerprint in self._performances:
            self._performances[template_fingerprint].add_match(profit_pct)
    
    def record_match_by_template(self, template: TrajectoryTemplate, profit_pct: float):
        """
        记录一次模板匹配结果（通过模板对象）
        
        Args:
            template: 模板对象
            profit_pct: 这笔交易的收益率
        """
        fp = template.fingerprint()
        self.record_match(fp, profit_pct)
    
    def evaluate(self, min_matches: int = 3, min_win_rate: float = 0.6,
                 min_avg_profit: float = 0.0) -> EvaluationResult:
        """
        评估所有模板并生成结果
        
        Args:
            min_matches: 最小匹配次数要求（低于此为"待观察"）
            min_win_rate: 最小胜率要求（低于此为"淘汰"）
            min_avg_profit: 最小平均收益要求
        
        Returns:
            EvaluationResult 评估结果
        """
        result = EvaluationResult()
        result.performances = list(self._performances.values())
        result.summarize(min_matches=min_matches, min_win_rate=min_win_rate)
        return result
    
    def get_performance(self, template_fingerprint: str) -> Optional[TemplatePerformance]:
        """获取指定模板的性能统计"""
        return self._performances.get(template_fingerprint)
    
    def clear(self):
        """清空所有统计（重新开始评估）"""
        for p in self._performances.values():
            p.match_count = 0
            p.win_count = 0
            p.loss_count = 0
            p.total_profit = 0.0
            p.profits = []
            p.grade = "未评估"
            p.score = 0.0


def integrate_with_walk_forward(evaluator: TemplateEvaluator,
                                 matched_template: TrajectoryTemplate,
                                 trade_profit: float):
    """
    Walk-Forward集成函数：在每次匹配交易时调用
    
    Args:
        evaluator: TemplateEvaluator 实例
        matched_template: 匹配到的模板
        trade_profit: 交易收益率
    """
    if matched_template is not None:
        evaluator.record_match_by_template(matched_template, trade_profit)
