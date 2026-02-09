"""
R3000 模式挖掘引擎
因果关系分析、多空转换逻辑挖掘、生存分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATTERN_CONFIG, LABELING_CONFIG


@dataclass
class FeatureImportance:
    """特征重要性结果"""
    feature_name: str
    importance: float
    rank: int


@dataclass
class LongShortLogic:
    """多空转换逻辑"""
    condition_type: str          # "long" or "short"
    feature_name: str
    threshold: float
    direction: str               # "above" or "below"
    probability: float           # 条件满足时的信号概率


@dataclass
class SurvivalStats:
    """生存分析统计"""
    hold_periods_mean: float
    hold_periods_median: float
    hold_periods_std: float
    hold_periods_quantiles: Dict[float, float]
    profit_mean: float
    profit_median: float
    profit_std: float
    profit_quantiles: Dict[float, float]
    win_rate: float
    risk_reward_mean: float


class PatternMiner:
    """
    模式挖掘器
    
    功能：
    1. 因果关系分析：哪些指标组合导致高/低点出现
    2. 多空转换逻辑：做多与做空的触发逻辑差异
    3. 生存分析：持仓时间与止盈止损空间的统计分布
    """
    
    def __init__(self):
        self.feature_importances: List[FeatureImportance] = []
        self.long_logic: List[LongShortLogic] = []
        self.short_logic: List[LongShortLogic] = []
        self.survival_stats: Optional[SurvivalStats] = None
        
        # 模型
        self.rf_classifier: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        
    def analyze_all(self, features: np.ndarray, labels: np.ndarray,
                    feature_names: List[str], trades: List = None,
                    prices: pd.Series = None) -> Dict:
        """
        执行完整的模式挖掘分析
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标注数组 (n_samples,)
            feature_names: 特征名称列表
            trades: 交易列表（用于生存分析）
            prices: 价格序列（用于生存分析）
            
        Returns:
            分析结果字典
        """
        results = {}
        
        # 1. 因果关系分析
        print("[PatternMiner] 执行因果关系分析...")
        results['feature_importance'] = self.analyze_causality(
            features, labels, feature_names
        )
        
        # 2. 多空转换逻辑挖掘
        print("[PatternMiner] 挖掘多空转换逻辑...")
        results['long_short_logic'] = self.mine_long_short_logic(
            features, labels, feature_names
        )
        
        # 3. 生存分析
        if trades:
            print("[PatternMiner] 执行生存分析...")
            results['survival'] = self.survival_analysis(trades, prices)
        
        return results
    
    def analyze_causality(self, features: np.ndarray, labels: np.ndarray,
                          feature_names: List[str]) -> Dict:
        """
        因果关系分析
        
        使用随机森林分析特征重要性，识别导致买入/卖出信号的关键特征
        """
        # 过滤掉 HOLD 标签
        mask = labels != 0
        X = features[mask]
        y = labels[mask]
        
        if len(X) < 10:
            print("[PatternMiner] 样本量不足，跳过因果分析")
            return {}
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 转换为二分类：BUY=1, SELL=0
        y_binary = (y > 0).astype(int)
        
        # 训练随机森林
        self.rf_classifier = RandomForestClassifier(
            n_estimators=PATTERN_CONFIG["RF_N_ESTIMATORS"],
            max_depth=PATTERN_CONFIG["RF_MAX_DEPTH"],
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier.fit(X_scaled, y_binary)
        
        # 获取特征重要性
        importances = self.rf_classifier.feature_importances_
        
        # 排序
        sorted_idx = np.argsort(importances)[::-1]
        
        self.feature_importances = []
        for rank, idx in enumerate(sorted_idx):
            self.feature_importances.append(FeatureImportance(
                feature_name=feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                importance=float(importances[idx]),
                rank=rank + 1
            ))
        
        # 构建结果
        result = {
            "top_features": [
                {
                    "name": fi.feature_name,
                    "importance": fi.importance,
                    "rank": fi.rank
                }
                for fi in self.feature_importances[:10]
            ],
            "model_accuracy": float(self.rf_classifier.score(X_scaled, y_binary)),
            "total_samples": len(X),
            "buy_samples": int((y > 0).sum()),
            "sell_samples": int((y < 0).sum()),
        }
        
        print(f"[PatternMiner] 模型准确率: {result['model_accuracy']:.2%}")
        print(f"[PatternMiner] Top 5 特征: {[fi.feature_name for fi in self.feature_importances[:5]]}")
        
        return result
    
    def mine_long_short_logic(self, features: np.ndarray, labels: np.ndarray,
                               feature_names: List[str]) -> Dict:
        """
        多空转换逻辑挖掘
        
        统计分析做多和做空信号的特征分布差异
        """
        # 分离买入和卖出样本
        buy_mask = labels > 0
        sell_mask = labels < 0
        
        X_buy = features[buy_mask]
        X_sell = features[sell_mask]
        
        if len(X_buy) < 5 or len(X_sell) < 5:
            print("[PatternMiner] 样本量不足，跳过多空逻辑挖掘")
            return {}
        
        self.long_logic = []
        self.short_logic = []
        
        n_features = features.shape[1]
        
        for i in range(n_features):
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            
            # 计算买入和卖出样本的特征分布
            buy_mean = np.mean(X_buy[:, i])
            buy_std = np.std(X_buy[:, i])
            sell_mean = np.mean(X_sell[:, i])
            sell_std = np.std(X_sell[:, i])
            
            # 计算差异显著性（简化版）
            diff = buy_mean - sell_mean
            pooled_std = np.sqrt((buy_std**2 + sell_std**2) / 2) + 1e-9
            effect_size = abs(diff) / pooled_std
            
            if effect_size > 0.3:  # 中等效应量
                # 做多逻辑
                if diff > 0:
                    threshold = (buy_mean + sell_mean) / 2
                    self.long_logic.append(LongShortLogic(
                        condition_type="long",
                        feature_name=fname,
                        threshold=float(threshold),
                        direction="above",
                        probability=float(effect_size / (effect_size + 1))
                    ))
                    self.short_logic.append(LongShortLogic(
                        condition_type="short",
                        feature_name=fname,
                        threshold=float(threshold),
                        direction="below",
                        probability=float(effect_size / (effect_size + 1))
                    ))
                else:
                    threshold = (buy_mean + sell_mean) / 2
                    self.long_logic.append(LongShortLogic(
                        condition_type="long",
                        feature_name=fname,
                        threshold=float(threshold),
                        direction="below",
                        probability=float(effect_size / (effect_size + 1))
                    ))
                    self.short_logic.append(LongShortLogic(
                        condition_type="short",
                        feature_name=fname,
                        threshold=float(threshold),
                        direction="above",
                        probability=float(effect_size / (effect_size + 1))
                    ))
        
        # 按概率排序
        self.long_logic.sort(key=lambda x: x.probability, reverse=True)
        self.short_logic.sort(key=lambda x: x.probability, reverse=True)
        
        result = {
            "long_conditions": [
                {
                    "feature": ll.feature_name,
                    "threshold": ll.threshold,
                    "direction": ll.direction,
                    "probability": ll.probability
                }
                for ll in self.long_logic[:10]
            ],
            "short_conditions": [
                {
                    "feature": sl.feature_name,
                    "threshold": sl.threshold,
                    "direction": sl.direction,
                    "probability": sl.probability
                }
                for sl in self.short_logic[:10]
            ],
            "feature_distributions": {
                "buy": {
                    "mean": float(np.mean(X_buy, axis=0).mean()),
                    "std": float(np.std(X_buy, axis=0).mean())
                },
                "sell": {
                    "mean": float(np.mean(X_sell, axis=0).mean()),
                    "std": float(np.std(X_sell, axis=0).mean())
                }
            }
        }
        
        print(f"[PatternMiner] 发现 {len(self.long_logic)} 个做多条件")
        print(f"[PatternMiner] 发现 {len(self.short_logic)} 个做空条件")
        
        return result
    
    def survival_analysis(self, trades: List, prices: pd.Series = None) -> Dict:
        """
        生存分析
        
        统计持仓时间、止盈止损空间的分布
        """
        if not trades:
            return {}
        
        # 提取交易统计
        hold_periods = [t.hold_periods for t in trades]
        profits = [t.profit for t in trades]
        profit_pcts = [t.profit_pct for t in trades]
        
        # 计算风险收益比
        risk_rewards = []
        for t in trades:
            risk = abs(t.buy_price - min(t.buy_price, t.sell_price))
            reward = max(0, t.profit)
            if risk > 0:
                risk_rewards.append(reward / risk)
        
        # 计算分位数
        quantiles = PATTERN_CONFIG["PROFIT_QUANTILES"]
        hold_quantiles = {q: float(np.quantile(hold_periods, q)) for q in quantiles}
        profit_quantiles = {q: float(np.quantile(profit_pcts, q)) for q in quantiles}
        
        # 胜率
        win_rate = sum(1 for p in profits if p > 0) / len(profits)
        
        self.survival_stats = SurvivalStats(
            hold_periods_mean=float(np.mean(hold_periods)),
            hold_periods_median=float(np.median(hold_periods)),
            hold_periods_std=float(np.std(hold_periods)),
            hold_periods_quantiles=hold_quantiles,
            profit_mean=float(np.mean(profit_pcts)),
            profit_median=float(np.median(profit_pcts)),
            profit_std=float(np.std(profit_pcts)),
            profit_quantiles=profit_quantiles,
            win_rate=win_rate,
            risk_reward_mean=float(np.mean(risk_rewards)) if risk_rewards else 0
        )
        
        # 构建直方图数据
        n_bins = PATTERN_CONFIG["SURVIVAL_BINS"]
        hold_hist, hold_bins = np.histogram(hold_periods, bins=n_bins)
        profit_hist, profit_bins = np.histogram(profit_pcts, bins=n_bins)
        
        result = {
            "hold_periods": {
                "mean": self.survival_stats.hold_periods_mean,
                "median": self.survival_stats.hold_periods_median,
                "std": self.survival_stats.hold_periods_std,
                "quantiles": self.survival_stats.hold_periods_quantiles,
                "histogram": {
                    "counts": hold_hist.tolist(),
                    "bins": hold_bins.tolist()
                }
            },
            "profit": {
                "mean": self.survival_stats.profit_mean,
                "median": self.survival_stats.profit_median,
                "std": self.survival_stats.profit_std,
                "quantiles": self.survival_stats.profit_quantiles,
                "histogram": {
                    "counts": profit_hist.tolist(),
                    "bins": profit_bins.tolist()
                }
            },
            "win_rate": self.survival_stats.win_rate,
            "risk_reward_mean": self.survival_stats.risk_reward_mean,
            "total_trades": len(trades)
        }
        
        print(f"[PatternMiner] 生存分析完成:")
        print(f"  平均持仓: {self.survival_stats.hold_periods_mean:.1f} 周期")
        print(f"  平均收益: {self.survival_stats.profit_mean:.2f}%")
        print(f"  胜率: {self.survival_stats.win_rate:.1%}")
        
        return result
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """获取特征重要性排名"""
        return [(fi.feature_name, fi.importance) for fi in self.feature_importances]
    
    def predict_signal(self, features: np.ndarray) -> Tuple[int, float]:
        """
        使用训练好的模型预测信号
        
        Args:
            features: 特征向量 (n_features,)
            
        Returns:
            (signal, probability) - 信号方向和概率
        """
        if self.rf_classifier is None:
            return 0, 0.0
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.rf_classifier.predict_proba(features_scaled)[0]
        
        # proba[0] = SELL 概率, proba[1] = BUY 概率
        if proba[1] > 0.6:
            return 1, proba[1]
        elif proba[0] > 0.6:
            return -1, proba[0]
        else:
            return 0, max(proba)
    
    def get_analysis_summary(self) -> Dict:
        """获取分析摘要"""
        summary = {
            "top_features": self.get_feature_importance_ranking()[:5],
            "long_conditions_count": len(self.long_logic),
            "short_conditions_count": len(self.short_logic),
        }
        
        if self.survival_stats:
            summary["survival"] = {
                "avg_hold": self.survival_stats.hold_periods_mean,
                "avg_profit": self.survival_stats.profit_mean,
                "win_rate": self.survival_stats.win_rate
            }
        
        return summary


# 测试代码
if __name__ == "__main__":
    from labeler import Trade
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 52
    
    # 生成特征
    features = np.random.randn(n_samples, n_features)
    
    # 生成标签（模拟）
    labels = np.zeros(n_samples)
    buy_idx = np.random.choice(n_samples, 50, replace=False)
    sell_idx = np.random.choice([i for i in range(n_samples) if i not in buy_idx], 50, replace=False)
    labels[buy_idx] = 1
    labels[sell_idx] = -1
    
    # 特征名称
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # 模拟交易
    trades = []
    for i in range(30):
        trades.append(Trade(
            buy_idx=i*10,
            sell_idx=i*10 + np.random.randint(30, 120),
            buy_price=50000 + np.random.randn() * 500,
            sell_price=50000 + np.random.randn() * 500 + np.random.randn() * 200,
            profit=np.random.randn() * 100,
            hold_periods=np.random.randint(30, 120)
        ))
    
    # 执行分析
    miner = PatternMiner()
    results = miner.analyze_all(features, labels, feature_names, trades)
    
    print("\n分析结果:")
    for key, value in results.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for k, v in list(value.items())[:3]:
                print(f"  {k}: {v}")
