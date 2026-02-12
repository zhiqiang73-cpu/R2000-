"""
R3000 模板聚类引擎
将大量碎片化模板聚类成少量"原型"，提高匹配效率和统计可靠性

核心思路：
  - 每个模板的完整交易 (pre_entry + holding + pre_exit) 压缩成一个统计向量
  - 使用 K-Means 聚类得到若干原型
  - 每个原型代表一类反复出现的交易模式
  - 原型自带历史胜率、平均收益等统计信息

存储：
  - 原型保存在 data/prototypes/ 目录
  - 支持增量更新（重新聚类）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os
import pickle
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAJECTORY_CONFIG

# 默认原型存储路径
DEFAULT_PROTOTYPE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "prototypes"
)


@dataclass
class Prototype:
    """
    交易原型 - 代表一类反复出现的交易模式
    
    每个原型属于特定的 (方向, 市场状态) 组合
    """
    prototype_id: int                   # 原型ID
    direction: str                      # "LONG" / "SHORT"
    regime: str = ""                    # 市场状态 (强多头/弱多头/震荡偏多/震荡偏空/弱空头/强空头)
    
    # 聚类中心向量（用于匹配）
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))  # 统计向量
    
    # 分段统计向量（用于三阶段匹配）
    pre_entry_centroid: np.ndarray = field(default_factory=lambda: np.array([]))  # 入场前特征中心 (32,)
    holding_centroid: np.ndarray = field(default_factory=lambda: np.array([]))    # 持仓特征中心 (64,) = mean + std
    pre_exit_centroid: np.ndarray = field(default_factory=lambda: np.array([]))   # 出场前特征中心 (32,)
    
    # 统计信息（来自聚类成员）
    member_count: int = 0               # 成员模板数量
    win_count: int = 0                  # 盈利交易数
    total_profit_pct: float = 0.0       # 总收益率
    avg_profit_pct: float = 0.0         # 平均收益率
    win_rate: float = 0.0               # 胜率
    avg_hold_bars: float = 0.0          # 平均持仓K线数
    
    # 成员指纹（用于追溯）
    member_fingerprints: List[str] = field(default_factory=list)
    
    # Walk-Forward 验证状态
    verified: bool = False              # 是否通过 WF 验证
    wf_grade: str = ""                  # WF 评级: "合格"/"待观察"/"淘汰"/""
    wf_match_count: int = 0             # WF 中累计匹配次数
    wf_win_rate: float = 0.0            # WF 中累计胜率
    
    # 用于快速余弦匹配的展平向量
    pre_entry_flat: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """初始化时预计算展平向量"""
        if self.pre_entry_centroid.size > 0 and self.pre_entry_flat.size == 0:
            self.pre_entry_flat = self._zscore_normalize(self.pre_entry_centroid)
    
    @staticmethod
    def _zscore_normalize(vec: np.ndarray) -> np.ndarray:
        """z-score 标准化"""
        if vec.size == 0:
            return vec
        std = vec.std()
        if std < 1e-9:
            return vec - vec.mean()
        return (vec - vec.mean()) / std
    
    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "prototype_id": self.prototype_id,
            "direction": self.direction,
            "regime": self.regime,
            "centroid": self.centroid.tolist() if self.centroid.size > 0 else [],
            "pre_entry_centroid": self.pre_entry_centroid.tolist() if self.pre_entry_centroid.size > 0 else [],
            "holding_centroid": self.holding_centroid.tolist() if self.holding_centroid.size > 0 else [],
            "pre_exit_centroid": self.pre_exit_centroid.tolist() if self.pre_exit_centroid.size > 0 else [],
            "member_count": self.member_count,
            "win_count": self.win_count,
            "total_profit_pct": self.total_profit_pct,
            "avg_profit_pct": self.avg_profit_pct,
            "win_rate": self.win_rate,
            "avg_hold_bars": self.avg_hold_bars,
            "member_fingerprints": self.member_fingerprints,
            "verified": self.verified,
            "wf_grade": self.wf_grade,
            "wf_match_count": self.wf_match_count,
            "wf_win_rate": self.wf_win_rate,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Prototype':
        """从字典还原"""
        proto = cls(
            prototype_id=d["prototype_id"],
            direction=d["direction"],
            regime=d.get("regime", ""),  # 兼容旧版本
            centroid=np.array(d.get("centroid", [])),
            pre_entry_centroid=np.array(d.get("pre_entry_centroid", [])),
            holding_centroid=np.array(d.get("holding_centroid", [])),
            pre_exit_centroid=np.array(d.get("pre_exit_centroid", [])),
            member_count=d.get("member_count", 0),
            win_count=d.get("win_count", 0),
            total_profit_pct=d.get("total_profit_pct", 0.0),
            avg_profit_pct=d.get("avg_profit_pct", 0.0),
            win_rate=d.get("win_rate", 0.0),
            avg_hold_bars=d.get("avg_hold_bars", 0.0),
            member_fingerprints=d.get("member_fingerprints", []),
            verified=d.get("verified", False),
            wf_grade=d.get("wf_grade", ""),
            wf_match_count=d.get("wf_match_count", 0),
            wf_win_rate=d.get("wf_win_rate", 0.0),
        )
        return proto


@dataclass
class PrototypeLibrary:
    """原型库 - 存储所有原型"""
    long_prototypes: List[Prototype] = field(default_factory=list)
    short_prototypes: List[Prototype] = field(default_factory=list)
    
    # 元信息
    created_at: str = ""
    source_template_count: int = 0
    clustering_params: Dict = field(default_factory=dict)
    source_symbol: str = ""
    source_interval: str = ""
    
    @property
    def total_count(self) -> int:
        return len(self.long_prototypes) + len(self.short_prototypes)
    
    def get_prototypes_by_direction(self, direction: str) -> List[Prototype]:
        """获取指定方向的原型"""
        if direction == "LONG":
            return self.long_prototypes
        elif direction == "SHORT":
            return self.short_prototypes
        else:
            return []
    
    def get_prototypes_by_direction_and_regime(self, direction: str, regime: str) -> List[Prototype]:
        """
        获取指定方向和市场状态的原型
        
        Args:
            direction: "LONG" / "SHORT"
            regime: 市场状态（强多头/弱多头/震荡偏多/震荡偏空/弱空头/强空头）
        
        Returns:
            匹配的原型列表
        """
        prototypes = self.get_prototypes_by_direction(direction)
        return [p for p in prototypes if p.regime == regime]
    
    def get_available_regimes(self) -> List[str]:
        """获取原型库中存在的所有市场状态"""
        regimes = set()
        for p in self.get_all_prototypes():
            if p.regime:
                regimes.add(p.regime)
        return sorted(regimes)
    
    def apply_wf_verification(self, prototype_stats: dict,
                               min_matches: int = 3,
                               min_win_rate: float = 0.6) -> dict:
        """
        将 Walk-Forward 验证结果回写到原型库中
        
        Args:
            prototype_stats: BatchWalkForwardEngine._prototype_stats
                格式: {fp_string: {"match_count": int, "win_count": int, "total_profit": float, ...}}
            min_matches: 最小匹配次数阈值
            min_win_rate: 最小胜率阈值
        
        Returns:
            {"qualified": int, "pending": int, "eliminated": int, "total_verified": int}
        """
        counts = {"qualified": 0, "pending": 0, "eliminated": 0, "total_verified": 0}
        
        # 构建 fp -> stats 的映射
        for proto in self.get_all_prototypes():
            fp = f"proto_{proto.direction}_{proto.prototype_id}"
            stat = prototype_stats.get(fp)
            
            if stat is None:
                # 该原型在 WF 中从未被匹配
                proto.verified = False
                proto.wf_grade = "待观察"
                proto.wf_match_count = 0
                proto.wf_win_rate = 0.0
                counts["pending"] += 1
                counts["total_verified"] += 1  # 待观察也保留
                continue
            
            match_count = stat.get("match_count", 0)
            win_count = stat.get("win_count", 0)
            win_rate = win_count / match_count if match_count > 0 else 0.0
            
            avg_profit = stat.get("total_profit", 0.0) / match_count if match_count > 0 else 0.0
            proto.wf_match_count = match_count
            proto.wf_win_rate = win_rate
            
            if match_count < min_matches:
                proto.verified = False
                proto.wf_grade = "待观察"
                counts["pending"] += 1
                counts["total_verified"] += 1
            elif self._is_qualified(win_rate, avg_profit, min_win_rate):
                proto.verified = True
                proto.wf_grade = "合格"
                counts["qualified"] += 1
                counts["total_verified"] += 1
            else:
                proto.verified = False
                proto.wf_grade = "淘汰"
                counts["eliminated"] += 1
        
        print(f"[PrototypeLibrary] WF验证结果已回写(期望收益模式): "
              f"合格={counts['qualified']}, "
              f"待观察={counts['pending']}, "
              f"淘汰={counts['eliminated']}")
        
        return counts
    
    @staticmethod
    def _is_qualified(win_rate: float, avg_profit: float, min_win_rate: float) -> bool:
        """判断原型是否合格：优先用期望收益，备用纯胜率"""
        from config import WALK_FORWARD_CONFIG
        use_expected = WALK_FORWARD_CONFIG.get("EVAL_USE_EXPECTED_PROFIT", True)
        if use_expected:
            return avg_profit > 0  # 期望收益为正即合格
        else:
            return win_rate >= min_win_rate and avg_profit >= 0.0  # 旧逻辑

    def get_all_prototypes(self) -> List[Prototype]:
        """获取所有原型"""
        return self.long_prototypes + self.short_prototypes

    
    def save(self, filepath: str = None, verbose: bool = True) -> str:
        """保存原型库到文件"""
        if filepath is None:
            os.makedirs(DEFAULT_PROTOTYPE_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(DEFAULT_PROTOTYPE_DIR, f"prototypes_{timestamp}.pkl")
        
        data = {
            "version": "1.0",
            "created_at": self.created_at or datetime.now().isoformat(),
            "source_template_count": self.source_template_count,
            "clustering_params": self.clustering_params,
            "source_symbol": self.source_symbol,
            "source_interval": self.source_interval,
            "long_prototypes": [p.to_dict() for p in self.long_prototypes],
            "short_prototypes": [p.to_dict() for p in self.short_prototypes],
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if verbose:
            size_kb = os.path.getsize(filepath) / 1024
            print(f"[PrototypeLibrary] 已保存: {filepath}")
            print(f"    LONG原型: {len(self.long_prototypes)}, "
                  f"SHORT原型: {len(self.short_prototypes)}, "
                  f"文件大小: {size_kb:.1f} KB")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str, verbose: bool = True) -> 'PrototypeLibrary':
        """从文件加载原型库"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        library = cls()
        library.created_at = data.get("created_at", "")
        library.source_template_count = data.get("source_template_count", 0)
        library.clustering_params = data.get("clustering_params", {})
        library.source_symbol = (data.get("source_symbol", "") or "").upper()
        library.source_interval = (data.get("source_interval", "") or "").strip()
        library.long_prototypes = [
            Prototype.from_dict(d) for d in data.get("long_prototypes", [])
        ]
        library.short_prototypes = [
            Prototype.from_dict(d) for d in data.get("short_prototypes", [])
        ]
        
        if verbose:
            print(f"[PrototypeLibrary] 已加载: {filepath}")
            print(f"    LONG原型: {len(library.long_prototypes)}, "
                  f"SHORT原型: {len(library.short_prototypes)}")
        
        return library
    
    @classmethod
    def load_latest(cls, prototype_dir: str = None, 
                    verbose: bool = True) -> Optional['PrototypeLibrary']:
        """加载最新的原型库"""
        if prototype_dir is None:
            prototype_dir = DEFAULT_PROTOTYPE_DIR
        
        if not os.path.exists(prototype_dir):
            if verbose:
                print("[PrototypeLibrary] 没有找到原型库目录")
            return None
        
        files = []
        for filename in os.listdir(prototype_dir):
            if filename.endswith('.pkl') and filename.startswith('prototypes_'):
                filepath = os.path.join(prototype_dir, filename)
                stat = os.stat(filepath)
                files.append((filepath, stat.st_mtime))
        
        if not files:
            if verbose:
                print("[PrototypeLibrary] 没有找到已保存的原型库")
            return None
        
        # 按修改时间排序，取最新的
        files.sort(key=lambda x: x[1], reverse=True)
        return cls.load(files[0][0], verbose=verbose)


class TemplateClusterer:
    """
    模板聚类引擎
    
    使用方式：
        clusterer = TemplateClusterer(n_clusters=30)
        library = clusterer.fit(trajectory_memory)
        library.save()
    """
    
    def __init__(self, 
                 n_clusters_long: int = 30,
                 n_clusters_short: int = 30,
                 min_cluster_size: int = 3,
                 random_state: int = 42):
        """
        Args:
            n_clusters_long: LONG 方向的聚类数
            n_clusters_short: SHORT 方向的聚类数
            min_cluster_size: 最小簇大小（小于此的簇会被丢弃）
            random_state: 随机种子
        """
        self.n_clusters_long = n_clusters_long
        self.n_clusters_short = n_clusters_short
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
    
    def fit(self, trajectory_memory, verbose: bool = True) -> PrototypeLibrary:
        """
        对模板库进行聚类，生成原型库
        
        聚类策略：按 (方向, 市场状态) 分组，每组独立聚类
        这确保了同一原型内的模板都来自相同的市场环境
        
        Args:
            trajectory_memory: TrajectoryMemory 实例
            verbose: 是否打印过程信息
        
        Returns:
            PrototypeLibrary 原型库
        """
        from sklearn.cluster import KMeans
        from core.market_regime import MarketRegime
        
        if verbose:
            print(f"[TemplateClusterer] 开始聚类（按方向+市场状态分组）...")
            print(f"    模板总数: {trajectory_memory.total_count}")
            print(f"    每组LONG聚类数: {self.n_clusters_long}, SHORT聚类数: {self.n_clusters_short}")
        
        library = PrototypeLibrary()
        library.created_at = datetime.now().isoformat()
        library.source_template_count = trajectory_memory.total_count
        library.clustering_params = {
            "n_clusters_long": self.n_clusters_long,
            "n_clusters_short": self.n_clusters_short,
            "min_cluster_size": self.min_cluster_size,
            "cluster_by_regime": True,  # 标记新版本聚类方式
        }
        library.source_symbol = getattr(trajectory_memory, "source_symbol", "").upper()
        library.source_interval = getattr(trajectory_memory, "source_interval", "").strip()
        
        # 6个市场状态
        all_regimes = MarketRegime.ALL_REGIMES + [MarketRegime.UNKNOWN]
        
        long_prototypes = []
        short_prototypes = []
        
        # 按 (direction, regime) 分组聚类
        for direction in ["LONG", "SHORT"]:
            all_templates = trajectory_memory.get_templates_by_direction(direction)
            base_n_clusters = self.n_clusters_long if direction == "LONG" else self.n_clusters_short
            
            if len(all_templates) == 0:
                if verbose:
                    print(f"    {direction}: 无模板，跳过")
                continue
            
            # 按市场状态分组
            # 【消除 regime="未知"】将"未知"状态的模板归入方向对应的默认 regime
            regime_groups = {}
            for t in all_templates:
                regime = t.regime if t.regime else ""
                # 消除"未知"：LONG 归入震荡偏多，SHORT 归入震荡偏空
                if not regime or regime == MarketRegime.UNKNOWN:
                    regime = MarketRegime.RANGE_BULL if direction == "LONG" else MarketRegime.RANGE_BEAR
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(t)
            
            if verbose:
                print(f"    {direction}: 共{len(all_templates)}个模板，分布在{len(regime_groups)}个市场状态")
            
            direction_prototypes = []
            
            for regime, templates in regime_groups.items():
                if len(templates) == 0:
                    continue
                
                # 根据该组的模板数量调整聚类数
                # 每个状态至少1个原型，最多base_n_clusters个
                n_clusters = min(base_n_clusters, max(1, len(templates)))
                
                # 1. 提取统计向量
                vectors, template_refs = self._extract_feature_vectors(templates)
                
                if len(vectors) == 0:
                    continue
                
                if len(vectors) < n_clusters:
                    n_clusters = max(1, len(vectors))
                
                # 2. K-Means 聚类
                if len(vectors) <= n_clusters:
                    labels = np.arange(len(vectors))
                    n_clusters = len(vectors)
                else:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        n_init=10,
                        max_iter=300,
                    )
                    labels = kmeans.fit_predict(vectors)
                
                # 3. 为每个簇生成原型（传入 regime）
                prototypes = self._build_prototypes_from_clusters(
                    direction, regime, vectors, labels, template_refs, n_clusters,
                    id_offset=len(direction_prototypes)
                )
                
                direction_prototypes.extend(prototypes)
                
                if verbose and len(prototypes) > 0:
                    print(f"        {direction}/{regime}: {len(templates)}模板 → {len(prototypes)}原型")
            
            if direction == "LONG":
                long_prototypes = direction_prototypes
            else:
                short_prototypes = direction_prototypes
        
        library.long_prototypes = long_prototypes
        library.short_prototypes = short_prototypes
        
        if verbose:
            print(f"[TemplateClusterer] 聚类完成: "
                  f"LONG={len(library.long_prototypes)}, "
                  f"SHORT={len(library.short_prototypes)}")
            self._print_prototype_stats(library)
        
        return library
    
    def _extract_feature_vectors(self, templates: list) -> Tuple[np.ndarray, list]:
        """
        从模板列表提取统计向量
        
        每个模板的完整交易压缩为：
          - pre_entry: mean(32) 
          - holding: mean(32) + std(32) = 64
          - pre_exit: mean(32)
          - meta: [hold_bars_norm, profit_pct_norm] = 2
          总计: 32 + 64 + 32 + 2 = 130 维
        
        Returns:
            (vectors, template_refs): 向量数组和对应的模板引用列表
        """
        vectors = []
        template_refs = []
        
        for template in templates:
            try:
                vec = self._template_to_vector(template)
                if vec is not None:
                    vectors.append(vec)
                    template_refs.append(template)
            except Exception as e:
                # 跳过无效模板
                continue
        
        if len(vectors) == 0:
            return np.array([]), []
        
        return np.array(vectors), template_refs
    
    def _template_to_vector(self, template) -> Optional[np.ndarray]:
        """
        将单个模板转换为统计向量
        
        统计摘要法：
          - pre_entry: 时间维度均值 (32,)
          - holding: 时间维度均值 + 标准差 (64,)
          - pre_exit: 时间维度均值 (32,)
          - meta: 持仓时长（归一化）+ 收益率（归一化）(2,)
        """
        # pre_entry 统计
        if template.pre_entry.size == 0:
            return None
        pre_entry_mean = template.pre_entry.mean(axis=0)  # (32,)
        
        # holding 统计
        if template.holding.size == 0:
            holding_mean = np.zeros(32)
            holding_std = np.zeros(32)
        else:
            holding_mean = template.holding.mean(axis=0)  # (32,)
            holding_std = template.holding.std(axis=0)    # (32,)
        
        # pre_exit 统计
        if template.pre_exit.size == 0:
            pre_exit_mean = np.zeros(32)
        else:
            pre_exit_mean = template.pre_exit.mean(axis=0)  # (32,)
        
        # 元信息（归一化）
        hold_bars = template.holding.shape[0] if template.holding.size > 0 else 0
        hold_bars_norm = hold_bars / 200.0  # 假设最大200根K线
        profit_norm = template.profit_pct / 10.0  # 假设最大10%收益
        meta = np.array([hold_bars_norm, profit_norm])
        
        # 拼接
        vec = np.concatenate([
            pre_entry_mean,      # 32
            holding_mean,        # 32
            holding_std,         # 32
            pre_exit_mean,       # 32
            meta,                # 2
        ])  # 总计 130 维
        
        return vec
    
    def _build_prototypes_from_clusters(self, direction: str, 
                                         regime: str,
                                         vectors: np.ndarray,
                                         labels: np.ndarray,
                                         template_refs: list,
                                         n_clusters: int,
                                         id_offset: int = 0) -> List[Prototype]:
        """
        从聚类结果构建原型列表
        
        Args:
            direction: 方向 (LONG/SHORT)
            regime: 市场状态 (6态之一)
            vectors: 特征向量
            labels: 聚类标签
            template_refs: 模板引用
            n_clusters: 簇数量
            id_offset: ID偏移量（用于保证全局唯一）
        """
        prototypes = []
        
        for cluster_id in range(n_clusters):
            # 找到属于该簇的所有模板
            mask = (labels == cluster_id)
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) < self.min_cluster_size:
                # 簇太小，丢弃
                continue
            
            # 获取该簇的向量和模板
            cluster_vectors = vectors[mask]
            cluster_templates = [template_refs[i] for i in cluster_indices]
            
            # 计算聚类中心
            centroid = cluster_vectors.mean(axis=0)
            
            # 拆分中心向量为各段
            pre_entry_centroid = centroid[:32]
            holding_centroid = centroid[32:96]  # mean(32) + std(32)
            pre_exit_centroid = centroid[96:128]
            
            # 统计成员信息
            member_count = len(cluster_templates)
            profits = [t.profit_pct for t in cluster_templates]
            hold_bars = [t.holding.shape[0] if t.holding.size > 0 else 0 
                        for t in cluster_templates]
            
            total_profit = sum(profits)
            avg_profit = np.mean(profits) if profits else 0.0
            win_count = sum(1 for p in profits if p > 0)
            win_rate = win_count / member_count if member_count > 0 else 0.0
            avg_hold = np.mean(hold_bars) if hold_bars else 0.0
            
            # 收集成员指纹
            member_fps = [t.fingerprint() for t in cluster_templates]
            
            prototype = Prototype(
                prototype_id=id_offset + len(prototypes),
                direction=direction,
                regime=regime,  # 新增：市场状态
                centroid=centroid,
                pre_entry_centroid=pre_entry_centroid,
                holding_centroid=holding_centroid,
                pre_exit_centroid=pre_exit_centroid,
                member_count=member_count,
                win_count=win_count,
                total_profit_pct=total_profit,
                avg_profit_pct=avg_profit,
                win_rate=win_rate,
                avg_hold_bars=avg_hold,
                member_fingerprints=member_fps,
            )
            
            prototypes.append(prototype)
        
        return prototypes
    
    def _print_prototype_stats(self, library: PrototypeLibrary):
        """打印原型统计信息"""
        for direction, protos in [("LONG", library.long_prototypes), 
                                   ("SHORT", library.short_prototypes)]:
            if not protos:
                continue
            
            total_members = sum(p.member_count for p in protos)
            avg_win_rate = np.mean([p.win_rate for p in protos])
            avg_profit = np.mean([p.avg_profit_pct for p in protos])
            
            # 统计各市场状态的原型数量
            regime_counts = {}
            for p in protos:
                r = p.regime or "未知"
                regime_counts[r] = regime_counts.get(r, 0) + 1
            regime_str = ", ".join([f"{r}:{c}" for r, c in sorted(regime_counts.items())])
            
            print(f"    [{direction}] {len(protos)}个原型, "
                  f"覆盖{total_members}个模板, "
                  f"平均胜率{avg_win_rate:.1%}, "
                  f"平均收益{avg_profit:.2f}%")
            print(f"        市场状态分布: {regime_str}")


class PrototypeMatcher:
    """
    原型匹配器 - 用于实时匹配
    
    使用方式：
        matcher = PrototypeMatcher(prototype_library)
        result = matcher.match_entry(current_window, direction="LONG")
    """
    
    def __init__(self, library: PrototypeLibrary, 
                 cosine_threshold: float = 0.5,
                 min_prototypes_agree: int = 1):
        """
        Args:
            library: 原型库
            cosine_threshold: 余弦相似度阈值
            min_prototypes_agree: 最少需要多少个原型同意才开仓
        """
        self.library = library
        self.cosine_threshold = cosine_threshold
        self.min_prototypes_agree = min_prototypes_agree
        
        # 【性能优化】预先堆叠原型中心向量，实现向量化匹配
        self._long_centroids = None
        self._short_centroids = None
        self._long_protos = []
        self._short_protos = []
        
        if library:
            self._prepare_centroids()
            
    def _prepare_centroids(self):
        """预先将原型中心向量堆叠成矩阵"""
        # LONG
        long_list = self.library.get_prototypes_by_direction("LONG")
        if long_list:
            # 过滤掉无效原型
            long_list = [p for p in long_list if p.pre_entry_centroid.size == 32]
            if long_list:
                self._long_centroids = np.stack([p.pre_entry_centroid for p in long_list])
                # 预归一化
                norms = np.linalg.norm(self._long_centroids, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._long_centroids /= norms
                self._long_protos = long_list
        
        # SHORT
        short_list = self.library.get_prototypes_by_direction("SHORT")
        if short_list:
            short_list = [p for p in short_list if p.pre_entry_centroid.size == 32]
            if short_list:
                self._short_centroids = np.stack([p.pre_entry_centroid for p in short_list])
                norms = np.linalg.norm(self._short_centroids, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._short_centroids /= norms
                self._short_protos = short_list

    def match_entry(self, current_window: np.ndarray, 
                    direction: str = None,
                    regime: str = None) -> Dict:
        """
        入场匹配：当前轨迹 vs 原型
        
        匹配逻辑：
        1. 如果指定了 regime，只在该市场状态的原型中匹配
        2. 如果指定了 direction，只匹配该方向
        3. 返回相似度最高的匹配结果
        
        Args:
            current_window: (window_size, 32) 当前K线窗口
            direction: 指定方向，None则双向匹配
            regime: 当前市场状态，None则匹配所有状态
        
        Returns:
            {
                "matched": bool,
                "direction": str,
                "best_prototype": Prototype,
                "similarity": float,
                "vote_long": int,
                "vote_short": int,
                "vote_ratio": float,
                "top_matches": [(prototype, similarity), ...]
            }
        """
        if current_window.size == 0 or self.library is None:
            return self._empty_result()
        
        # 提取当前窗口的统计向量（只用 pre_entry 部分）
        current_vec = self._window_to_vector(current_window)
        
        # 归一化当前向量，供余弦相似度（点积）使用
        c_norm = np.linalg.norm(current_vec)
        if c_norm < 1e-9:
            return self._empty_result()
        current_unit = current_vec / c_norm
        
        # 分别匹配 LONG 和 SHORT（传入 regime 过滤）
        # 如果 regime 为 None，直接走向量化矩阵运算
        if regime is None:
            long_matches = self._match_direction_vectorized(current_unit, "LONG")
            short_matches = self._match_direction_vectorized(current_unit, "SHORT")
        else:
            # 维持原有 regime 库过滤逻辑（因为 regime 过滤后的候选集通常很小，向量化优势不明显）
            long_matches = self._match_direction(current_vec, "LONG", regime)
            short_matches = self._match_direction(current_vec, "SHORT", regime)
        
        # 投票统计
        vote_long = len([m for m in long_matches if m[1] >= self.cosine_threshold])
        vote_short = len([m for m in short_matches if m[1] >= self.cosine_threshold])
        total_votes = vote_long + vote_short
        
        # 确定方向
        if direction is not None:
            # 指定方向
            matches = long_matches if direction == "LONG" else short_matches
            votes = vote_long if direction == "LONG" else vote_short
        else:
            # 双向选择投票多的
            if vote_long > vote_short:
                direction = "LONG"
                matches = long_matches
                votes = vote_long
            elif vote_short > vote_long:
                direction = "SHORT"
                matches = short_matches
                votes = vote_short
            else:
                # 平局，看最高相似度
                if long_matches and short_matches:
                    if long_matches[0][1] >= short_matches[0][1]:
                        direction = "LONG"
                        matches = long_matches
                        votes = vote_long
                    else:
                        direction = "SHORT"
                        matches = short_matches
                        votes = vote_short
                elif long_matches:
                    direction = "LONG"
                    matches = long_matches
                    votes = vote_long
                elif short_matches:
                    direction = "SHORT"
                    matches = short_matches
                    votes = vote_short
                else:
                    return self._empty_result()
        
        if not matches:
            return self._empty_result()
        
        best_proto, best_sim = matches[0]
        
        # 检查是否满足匹配条件
        matched = (best_sim >= self.cosine_threshold and 
                   votes >= self.min_prototypes_agree)
        
        vote_ratio = vote_long / total_votes if total_votes > 0 else 0.5
        
        return {
            "matched": matched,
            "direction": direction,
            "best_prototype": best_proto,
            "similarity": best_sim,
            "vote_long": vote_long,
            "vote_short": vote_short,
            "vote_ratio": vote_ratio,
            "top_matches": matches[:5],
        }

    def _match_direction_vectorized(self, current_unit: np.ndarray, 
                                     direction: str) -> List[Tuple[Prototype, float]]:
        """【性能核心】矩阵化匹配指定方向的所有原型"""
        centroids = self._long_centroids if direction == "LONG" else self._short_centroids
        protos = self._long_protos if direction == "LONG" else self._short_protos
        
        if centroids is None:
            return []
        
        # 计算余弦相似度：矩阵 × 向量
        sims = centroids @ current_unit
        
        results = []
        for i, sim in enumerate(sims):
            results.append((protos[i], float(sim)))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def check_holding_health(self, holding_so_far: np.ndarray,
                             matched_prototype: Prototype) -> Tuple[float, str]:
        """
        检查持仓健康度
        
        Args:
            holding_so_far: (n, 32) 从入场到当前的持仓轨迹
            matched_prototype: 匹配的原型
        
        Returns:
            (health_score, status): 健康分数(0~1) 和 状态描述
        """
        if holding_so_far.size == 0:
            return 1.0, "正常"
        
        # 计算当前持仓的统计向量
        current_mean = holding_so_far.mean(axis=0)
        current_std = holding_so_far.std(axis=0)
        current_vec = np.concatenate([current_mean, current_std])
        
        # 与原型的 holding_centroid 比较
        proto_vec = matched_prototype.holding_centroid
        
        # 余弦相似度作为健康度
        similarity = self._cosine_similarity(current_vec, proto_vec)
        
        if similarity >= 0.7:
            status = "健康"
        elif similarity >= 0.5:
            status = "正常"
        elif similarity >= 0.3:
            status = "偏离"
        else:
            status = "危险"
        
        return similarity, status
    
    def rematch_by_holding(self, holding_so_far: np.ndarray, 
                           current_prototype: Prototype,
                           direction: str,
                           regime: str = None,
                           switch_threshold: float = 0.15) -> Tuple[Optional[Prototype], float, bool]:
        """
        根据持仓轨迹重新匹配更合适的原型
        
        当持仓过程中发现有其他原型的持仓段更匹配当前轨迹时，切换到该原型。
        这允许系统在市场动态变化时调整预期。
        
        Args:
            holding_so_far: (n, 32) 从入场到当前的持仓轨迹
            current_prototype: 当前匹配的原型
            direction: 交易方向 (LONG/SHORT)
            regime: 市场状态过滤，None则匹配所有状态
            switch_threshold: 切换阈值，新原型相似度需超出当前原型此值才切换
        
        Returns:
            (best_prototype, similarity, switched): 
              - best_prototype: 最匹配的原型（可能是当前原型或新原型）
              - similarity: 与该原型的相似度
              - switched: 是否发生了切换
        """
        if holding_so_far.size == 0 or self.library is None:
            return current_prototype, 1.0, False
        
        # 计算当前持仓的统计向量 (64D = mean + std)
        current_mean = holding_so_far.mean(axis=0)
        current_std = holding_so_far.std(axis=0)
        current_vec = np.concatenate([current_mean, current_std])
        
        # 当前原型的相似度
        current_sim = self._cosine_similarity(current_vec, current_prototype.holding_centroid)
        
        # 获取同方向的原型候选
        if regime:
            candidates = self.library.get_prototypes_by_direction_and_regime(direction, regime)
        else:
            candidates = self.library.get_prototypes_by_direction(direction)
        
        # 找到持仓段最匹配的原型
        best_proto = current_prototype
        best_sim = current_sim
        
        for proto in candidates:
            if proto.prototype_id == current_prototype.prototype_id:
                continue  # 跳过当前原型
            if proto.holding_centroid.size == 0:
                continue
            
            sim = self._cosine_similarity(current_vec, proto.holding_centroid)
            if sim > best_sim + switch_threshold:  # 必须显著优于当前
                best_sim = sim
                best_proto = proto
        
        switched = (best_proto.prototype_id != current_prototype.prototype_id)
        return best_proto, best_sim, switched
    
    def check_exit_pattern(self, recent_trajectory: np.ndarray,
                           current_prototype: Prototype,
                           direction: str,
                           entry_price: float,
                           current_price: float,
                           stop_loss: float,
                           take_profit: float,
                           current_regime: str = None) -> Dict:
        """
        检查离场模式：综合 pre_exit 匹配 + 止损保护 + 市场状态 + K线空间位置
        
        这是三阶段匹配系统的最后一环：当持仓轨迹开始像原型的出场段时，
        结合风控因素决定是否触发离场。
        
        Args:
            recent_trajectory: (n, 32) 最近的轨迹（通常取持仓末尾或 pre_exit_window 根）
            current_prototype: 当前匹配的原型
            direction: 交易方向
            entry_price: 入场价格
            current_price: 当前价格
            stop_loss: 止损价格
            take_profit: 止盈价格
            current_regime: 当前市场状态
        
        Returns:
            {
                "should_exit": bool,           # 是否建议离场
                "exit_signal_strength": float, # 离场信号强度 (0~1)
                "exit_reason": str,            # 离场原因
                "pattern_similarity": float,   # 与出场模式的相似度
                "price_position": float,       # 价格位置 (0=入场, 1=止盈, -1=止损)
                "regime_warning": bool,        # 市场状态是否变化
                "details": dict,               # 详细信息
            }
        """
        result = {
            "should_exit": False,
            "exit_signal_strength": 0.0,
            "exit_reason": "",
            "pattern_similarity": 0.0,
            "price_position": 0.0,
            "regime_warning": False,
            "details": {},
        }
        
        if recent_trajectory.size == 0 or current_prototype is None:
            return result
        
        # ══════════════════════════════════════════════════════════
        # 1. Pre-exit 模式匹配：当前轨迹是否像原型的出场段？
        # ══════════════════════════════════════════════════════════
        current_vec = recent_trajectory.mean(axis=0)  # 32D
        proto_exit_vec = current_prototype.pre_exit_centroid
        
        if proto_exit_vec.size > 0:
            pattern_sim = self._cosine_similarity(current_vec, proto_exit_vec)
        else:
            pattern_sim = 0.0
        
        result["pattern_similarity"] = pattern_sim
        result["details"]["pre_exit_similarity"] = pattern_sim
        
        # ══════════════════════════════════════════════════════════
        # 2. K线空间位置：当前价格在入场价、止盈、止损之间的位置
        # ══════════════════════════════════════════════════════════
        # 归一化价格位置：0=入场价，1=止盈，-1=止损
        if direction == "LONG":
            tp_distance = take_profit - entry_price
            sl_distance = entry_price - stop_loss
            price_move = current_price - entry_price
            
            if price_move >= 0 and tp_distance > 0:
                price_position = min(1.0, price_move / tp_distance)
            elif price_move < 0 and sl_distance > 0:
                price_position = max(-1.0, price_move / sl_distance)
            else:
                price_position = 0.0
        else:  # SHORT
            tp_distance = entry_price - take_profit
            sl_distance = stop_loss - entry_price
            price_move = entry_price - current_price
            
            if price_move >= 0 and tp_distance > 0:
                price_position = min(1.0, price_move / tp_distance)
            elif price_move < 0 and sl_distance > 0:
                price_position = max(-1.0, price_move / sl_distance)
            else:
                price_position = 0.0
        
        result["price_position"] = price_position
        result["details"]["price_pct_to_tp"] = price_position if price_position > 0 else 0
        result["details"]["price_pct_to_sl"] = -price_position if price_position < 0 else 0
        
        # ══════════════════════════════════════════════════════════
        # 3. 市场状态变化检测
        # ══════════════════════════════════════════════════════════
        proto_regime = current_prototype.regime
        regime_warning = False
        
        if current_regime and proto_regime:
            # 检测逆向市场状态变化
            from core.market_regime import MarketRegime
            bull_regimes = {MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL, MarketRegime.RANGE_BULL}
            bear_regimes = {MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR, MarketRegime.RANGE_BEAR}
            
            proto_is_bull = proto_regime in bull_regimes
            current_is_bear = current_regime in bear_regimes
            proto_is_bear = proto_regime in bear_regimes
            current_is_bull = current_regime in bull_regimes
            
            # LONG 仓位遇到熊市信号，或 SHORT 仓位遇到牛市信号
            if direction == "LONG" and proto_is_bull and current_is_bear:
                regime_warning = True
            elif direction == "SHORT" and proto_is_bear and current_is_bull:
                regime_warning = True
        
        result["regime_warning"] = regime_warning
        result["details"]["proto_regime"] = proto_regime
        result["details"]["current_regime"] = current_regime
        
        # ══════════════════════════════════════════════════════════
        # 4. 综合离场决策
        # ══════════════════════════════════════════════════════════
        # 计算离场信号强度（加权综合）
        exit_signal = 0.0
        exit_reasons = []
        
        # 4.1 出场模式匹配 (权重 40%)
        if pattern_sim >= 0.7:
            exit_signal += 0.4 * pattern_sim
            exit_reasons.append(f"出场模式匹配 {pattern_sim:.0%}")
        elif pattern_sim >= 0.5:
            exit_signal += 0.2 * pattern_sim
        
        # 4.2 价格接近止盈 (权重 30%)
        if price_position >= 0.8:
            exit_signal += 0.3
            exit_reasons.append(f"接近止盈 {price_position:.0%}")
        elif price_position >= 0.5:
            exit_signal += 0.15
        
        # 4.3 价格接近止损 (权重 20%) - 止损优先保护
        if price_position <= -0.7:
            exit_signal += 0.3  # 接近止损时提高权重
            exit_reasons.append(f"接近止损 {-price_position:.0%}")
        elif price_position <= -0.5:
            exit_signal += 0.15
        
        # 4.4 市场状态逆转 (权重 10%)
        if regime_warning:
            exit_signal += 0.2
            exit_reasons.append("市场状态逆转")
        
        result["exit_signal_strength"] = min(1.0, exit_signal)
        
        # 离场决策阈值
        EXIT_THRESHOLD = 0.6
        
        if exit_signal >= EXIT_THRESHOLD:
            result["should_exit"] = True
            result["exit_reason"] = " + ".join(exit_reasons) if exit_reasons else "综合信号触发"
        elif price_position <= -0.9:
            # 接近止损时强制建议离场
            result["should_exit"] = True
            result["exit_reason"] = "止损保护"
        
        return result
    
    def _match_direction(self, current_vec: np.ndarray, 
                         direction: str,
                         regime: str = None) -> List[Tuple[Prototype, float]]:
        """
        匹配指定方向的原型
        
        Args:
            current_vec: 当前特征向量
            direction: 方向 (LONG/SHORT)
            regime: 市场状态过滤，None则匹配所有状态
        
        Returns:
            [(原型, 相似度), ...] 按相似度降序
        """
        if regime:
            # 只匹配指定市场状态的原型
            prototypes = self.library.get_prototypes_by_direction_and_regime(direction, regime)
        else:
            # 匹配所有原型
            prototypes = self.library.get_prototypes_by_direction(direction)
        
        results = []
        for proto in prototypes:
            if proto.pre_entry_centroid.size == 0:
                continue
            sim = self._cosine_similarity(current_vec, proto.pre_entry_centroid)
            results.append((proto, sim))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _window_to_vector(self, window: np.ndarray) -> np.ndarray:
        """将K线窗口转换为统计向量（只取均值，用于与 pre_entry_centroid 比较）"""
        return window.mean(axis=0)  # (32,)
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        if v1.size == 0 or v2.size == 0:
            return 0.0
        
        # 对齐长度
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _empty_result(self) -> Dict:
        """返回空匹配结果"""
        return {
            "matched": False,
            "direction": None,
            "best_prototype": None,
            "similarity": 0.0,
            "vote_long": 0,
            "vote_short": 0,
            "vote_ratio": 0.5,
            "top_matches": [],
        }


# ══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════════

def cluster_templates(trajectory_memory, 
                      n_clusters_long: int = 30,
                      n_clusters_short: int = 30,
                      save: bool = True,
                      verbose: bool = True) -> PrototypeLibrary:
    """
    便捷函数：对模板库进行聚类并保存
    
    Args:
        trajectory_memory: TrajectoryMemory 实例
        n_clusters_long: LONG 聚类数
        n_clusters_short: SHORT 聚类数
        save: 是否保存到文件
        verbose: 是否打印信息
    
    Returns:
        PrototypeLibrary
    """
    clusterer = TemplateClusterer(
        n_clusters_long=n_clusters_long,
        n_clusters_short=n_clusters_short,
    )
    
    library = clusterer.fit(trajectory_memory, verbose=verbose)
    
    if save:
        library.save(verbose=verbose)
    
    return library
