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
    """
    prototype_id: int                   # 原型ID
    direction: str                      # "LONG" / "SHORT"
    
    # 聚类中心向量（用于匹配）
    centroid: np.ndarray                # 统计向量
    
    # 分段统计向量（用于三阶段匹配）
    pre_entry_centroid: np.ndarray      # 入场前特征中心 (32,)
    holding_centroid: np.ndarray        # 持仓特征中心 (64,) = mean + std
    pre_exit_centroid: np.ndarray       # 出场前特征中心 (32,)
    
    # 统计信息（来自聚类成员）
    member_count: int = 0               # 成员模板数量
    win_count: int = 0                  # 盈利交易数
    total_profit_pct: float = 0.0       # 总收益率
    avg_profit_pct: float = 0.0         # 平均收益率
    win_rate: float = 0.0               # 胜率
    avg_hold_bars: float = 0.0          # 平均持仓K线数
    
    # 成员指纹（用于追溯）
    member_fingerprints: List[str] = field(default_factory=list)
    
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
            "centroid": self.centroid.tolist(),
            "pre_entry_centroid": self.pre_entry_centroid.tolist(),
            "holding_centroid": self.holding_centroid.tolist(),
            "pre_exit_centroid": self.pre_exit_centroid.tolist(),
            "member_count": self.member_count,
            "win_count": self.win_count,
            "total_profit_pct": self.total_profit_pct,
            "avg_profit_pct": self.avg_profit_pct,
            "win_rate": self.win_rate,
            "avg_hold_bars": self.avg_hold_bars,
            "member_fingerprints": self.member_fingerprints,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Prototype':
        """从字典还原"""
        proto = cls(
            prototype_id=d["prototype_id"],
            direction=d["direction"],
            centroid=np.array(d["centroid"]),
            pre_entry_centroid=np.array(d["pre_entry_centroid"]),
            holding_centroid=np.array(d["holding_centroid"]),
            pre_exit_centroid=np.array(d["pre_exit_centroid"]),
            member_count=d.get("member_count", 0),
            win_count=d.get("win_count", 0),
            total_profit_pct=d.get("total_profit_pct", 0.0),
            avg_profit_pct=d.get("avg_profit_pct", 0.0),
            win_rate=d.get("win_rate", 0.0),
            avg_hold_bars=d.get("avg_hold_bars", 0.0),
            member_fingerprints=d.get("member_fingerprints", []),
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
        
        Args:
            trajectory_memory: TrajectoryMemory 实例
            verbose: 是否打印过程信息
        
        Returns:
            PrototypeLibrary 原型库
        """
        from sklearn.cluster import KMeans
        
        if verbose:
            print(f"[TemplateClusterer] 开始聚类...")
            print(f"    模板总数: {trajectory_memory.total_count}")
            print(f"    LONG聚类数: {self.n_clusters_long}, SHORT聚类数: {self.n_clusters_short}")
        
        library = PrototypeLibrary()
        library.created_at = datetime.now().isoformat()
        library.source_template_count = trajectory_memory.total_count
        library.clustering_params = {
            "n_clusters_long": self.n_clusters_long,
            "n_clusters_short": self.n_clusters_short,
            "min_cluster_size": self.min_cluster_size,
        }
        
        # 分别处理 LONG 和 SHORT
        for direction in ["LONG", "SHORT"]:
            templates = trajectory_memory.get_templates_by_direction(direction)
            n_clusters = self.n_clusters_long if direction == "LONG" else self.n_clusters_short
            
            if len(templates) < n_clusters:
                # 模板数不足，每个模板一个原型
                n_clusters = max(1, len(templates))
            
            if len(templates) == 0:
                if verbose:
                    print(f"    {direction}: 无模板，跳过")
                continue
            
            # 1. 提取统计向量
            vectors, template_refs = self._extract_feature_vectors(templates)
            
            if len(vectors) < n_clusters:
                n_clusters = max(1, len(vectors))
            
            if verbose:
                print(f"    {direction}: {len(templates)}个模板 → 提取{len(vectors)}个向量")
            
            # 2. K-Means 聚类
            if len(vectors) <= n_clusters:
                # 太少，每个向量一个簇
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
            
            # 3. 为每个簇生成原型
            prototypes = self._build_prototypes_from_clusters(
                direction, vectors, labels, template_refs, n_clusters
            )
            
            if direction == "LONG":
                library.long_prototypes = prototypes
            else:
                library.short_prototypes = prototypes
            
            if verbose:
                print(f"    {direction}: 生成 {len(prototypes)} 个原型")
        
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
                                         vectors: np.ndarray,
                                         labels: np.ndarray,
                                         template_refs: list,
                                         n_clusters: int) -> List[Prototype]:
        """从聚类结果构建原型列表"""
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
                prototype_id=len(prototypes),
                direction=direction,
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
            
            print(f"    [{direction}] {len(protos)}个原型, "
                  f"覆盖{total_members}个模板, "
                  f"平均胜率{avg_win_rate:.1%}, "
                  f"平均收益{avg_profit:.2f}%")


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
    
    def match_entry(self, current_window: np.ndarray, 
                    direction: str = None) -> Dict:
        """
        入场匹配：当前轨迹 vs 原型
        
        Args:
            current_window: (window_size, 32) 当前K线窗口
            direction: 指定方向，None则双向匹配
        
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
        if current_window.size == 0:
            return self._empty_result()
        
        # 提取当前窗口的统计向量（只用 pre_entry 部分）
        current_vec = self._window_to_vector(current_window)
        
        # 分别匹配 LONG 和 SHORT
        long_matches = self._match_direction(current_vec, "LONG")
        short_matches = self._match_direction(current_vec, "SHORT")
        
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
    
    def _match_direction(self, current_vec: np.ndarray, 
                         direction: str) -> List[Tuple[Prototype, float]]:
        """匹配指定方向的所有原型"""
        prototypes = self.library.get_prototypes_by_direction(direction)
        
        results = []
        for proto in prototypes:
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
