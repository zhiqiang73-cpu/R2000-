"""
原型库重新训练和验证脚本

功能：
1. 加载最新的 TrajectoryMemory（轨迹模板库）
2. 使用 TemplateClusterer 重新聚类生成原型库
3. 保存新的原型库
4. 验证多维相似度系统的效果

验证指标：
- 相似度分布（应该从 97%-99.8% 分散到 60%-95%）
- 多维相似度分解（余弦 + 欧氏 + DTW）
- 置信度分布

使用方法：
    python scripts/retrain_prototypes.py
    python scripts/retrain_prototypes.py --verify-only  # 仅验证，不重新训练
    python scripts/retrain_prototypes.py --n-long 30 --n-short 30  # 指定聚类数
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trajectory_engine import TrajectoryMemory
from core.template_clusterer import (
    TemplateClusterer, PrototypeLibrary, PrototypeMatcher, 
    MultiSimilarityCalculator
)
from config import (
    PROTOTYPE_CONFIG, SIMILARITY_CONFIG, CONFIDENCE_CONFIG,
    TRAJECTORY_CONFIG, FEATURE_WEIGHTS_CONFIG
)


class PrototypeRetrainer:
    """原型库重训练器"""
    
    def __init__(self, 
                 n_clusters_long: int = None,
                 n_clusters_short: int = None,
                 verbose: bool = True):
        """
        Args:
            n_clusters_long: LONG 方向聚类数，默认从配置读取
            n_clusters_short: SHORT 方向聚类数，默认从配置读取
            verbose: 是否打印详细信息
        """
        self.n_clusters_long = n_clusters_long or PROTOTYPE_CONFIG.get("N_CLUSTERS_LONG", 30)
        self.n_clusters_short = n_clusters_short or PROTOTYPE_CONFIG.get("N_CLUSTERS_SHORT", 30)
        self.verbose = verbose
        
        self.memory: TrajectoryMemory = None
        self.library: PrototypeLibrary = None
        self.old_library: PrototypeLibrary = None
    
    def load_memory(self) -> bool:
        """加载最新的轨迹模板库"""
        print("\n" + "=" * 60)
        print("步骤 1: 加载轨迹模板库 (TrajectoryMemory)")
        print("=" * 60)
        
        self.memory = TrajectoryMemory.load_latest(verbose=self.verbose)
        
        if self.memory is None:
            print("[ERROR] 没有找到轨迹模板库，请先运行回测生成模板")
            return False
        
        print(f"\n[INFO] 已加载轨迹模板库:")
        print(f"    - 模板总数: {self.memory.total_count}")
        print(f"    - 来源品种: {self.memory.source_symbol or '未知'}")
        print(f"    - 来源周期: {self.memory.source_interval or '未知'}")
        
        # 打印各状态分布
        stats = self.memory.get_stats()
        print(f"\n    模板分布:")
        for regime in sorted(stats.keys()):
            for direction in ["LONG", "SHORT"]:
                dir_stats = stats[regime].get(direction, {})
                count = dir_stats.get("count", 0) if isinstance(dir_stats, dict) else 0
                if count > 0:
                    print(f"        {regime} {direction}: {count}")
        
        return True
    
    def load_old_library(self) -> bool:
        """加载旧的原型库（用于对比）"""
        print("\n" + "=" * 60)
        print("步骤 2: 加载现有原型库（用于对比）")
        print("=" * 60)
        
        self.old_library = PrototypeLibrary.load_latest(verbose=self.verbose)
        
        if self.old_library is None:
            print("[INFO] 没有找到现有原型库，这是首次训练")
            return False
        
        print(f"\n[INFO] 已加载现有原型库:")
        print(f"    - 原型总数: {self.old_library.total_count}")
        print(f"    - LONG 原型: {len(self.old_library.long_prototypes)}")
        print(f"    - SHORT 原型: {len(self.old_library.short_prototypes)}")
        print(f"    - 创建时间: {self.old_library.created_at}")
        
        return True
    
    def retrain(self) -> bool:
        """重新训练原型库"""
        if self.memory is None:
            print("[ERROR] 请先加载轨迹模板库")
            return False
        
        print("\n" + "=" * 60)
        print("步骤 3: 重新训练原型库 (TemplateClusterer)")
        print("=" * 60)
        
        print(f"\n[INFO] 聚类参数:")
        print(f"    - LONG 聚类数: {self.n_clusters_long}")
        print(f"    - SHORT 聚类数: {self.n_clusters_short}")
        print(f"    - 最小簇大小: {PROTOTYPE_CONFIG.get('MIN_CLUSTER_SIZE', 3)}")
        
        print(f"\n[INFO] 特征权重配置:")
        print(f"    - Layer A (即时信号, 16维): {FEATURE_WEIGHTS_CONFIG.get('LAYER_A_WEIGHT', 1.5)}x")
        print(f"    - Layer B (动量变化, 10维): {FEATURE_WEIGHTS_CONFIG.get('LAYER_B_WEIGHT', 1.2)}x")
        print(f"    - Layer C (结构位置, 6维): {FEATURE_WEIGHTS_CONFIG.get('LAYER_C_WEIGHT', 1.0)}x")
        
        print(f"\n[INFO] 多维相似度配置:")
        print(f"    - 余弦相似度权重: {SIMILARITY_CONFIG.get('COSINE_WEIGHT', 0.30)}")
        print(f"    - 欧氏距离权重: {SIMILARITY_CONFIG.get('EUCLIDEAN_WEIGHT', 0.40)}")
        print(f"    - DTW权重: {SIMILARITY_CONFIG.get('DTW_WEIGHT', 0.30)}")
        
        # 创建聚类器并训练
        clusterer = TemplateClusterer(
            n_clusters_long=self.n_clusters_long,
            n_clusters_short=self.n_clusters_short,
            min_cluster_size=PROTOTYPE_CONFIG.get("MIN_CLUSTER_SIZE", 3),
        )
        
        print("\n[INFO] 开始聚类...")
        self.library = clusterer.fit(self.memory, verbose=self.verbose)
        
        print(f"\n[SUCCESS] 训练完成!")
        print(f"    - 生成原型总数: {self.library.total_count}")
        print(f"    - LONG 原型: {len(self.library.long_prototypes)}")
        print(f"    - SHORT 原型: {len(self.library.short_prototypes)}")
        
        return True
    
    def save_library(self) -> str:
        """保存新的原型库"""
        if self.library is None:
            print("[ERROR] 请先训练原型库")
            return None
        
        print("\n" + "=" * 60)
        print("步骤 4: 保存原型库")
        print("=" * 60)
        
        filepath = self.library.save(verbose=self.verbose)
        print(f"\n[SUCCESS] 已保存原型库: {filepath}")
        
        return filepath
    
    def verify(self) -> Dict:
        """验证新原型库的效果"""
        library = self.library or self.old_library
        
        if library is None:
            print("[ERROR] 没有可验证的原型库")
            return {}
        
        print("\n" + "=" * 60)
        print("步骤 5: 验证原型库效果")
        print("=" * 60)
        
        results = {}
        
        # 5.1 置信度分布
        print("\n[验证 1] 置信度分布")
        print("-" * 40)
        results["confidence"] = self._analyze_confidence(library)
        
        # 5.2 原型质量分析
        print("\n[验证 2] 原型质量分析")
        print("-" * 40)
        results["quality"] = self._analyze_quality(library)
        
        # 5.3 相似度分布测试
        if self.memory and self.memory.total_count > 0:
            print("\n[验证 3] 相似度分布测试（使用模板样本）")
            print("-" * 40)
            results["similarity"] = self._test_similarity_distribution(library)
        
        # 5.4 新旧对比（如果有旧库）
        if self.old_library and self.library:
            print("\n[验证 4] 新旧原型库对比")
            print("-" * 40)
            results["comparison"] = self._compare_libraries()
        
        return results
    
    def _analyze_confidence(self, library: PrototypeLibrary) -> Dict:
        """分析置信度分布"""
        confidences = []
        for proto in library.get_all_prototypes():
            confidences.append(proto.confidence)
        
        if not confidences:
            print("    没有原型可分析")
            return {}
        
        confidences = np.array(confidences)
        
        stats = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
        }
        
        # 置信度分段统计
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels = ["[0-30%)", "[30-50%)", "[50-70%)", "[70-90%)", "[90-100%]"]
        hist, _ = np.histogram(confidences, bins=bins)
        
        print(f"    置信度统计:")
        print(f"        均值: {stats['mean']:.3f}")
        print(f"        标准差: {stats['std']:.3f}")
        print(f"        范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"        中位数: {stats['median']:.3f}")
        print(f"\n    置信度分布:")
        for i, (label, count) in enumerate(zip(labels, hist)):
            pct = count / len(confidences) * 100
            bar = "█" * int(pct / 5)
            print(f"        {label}: {count:3d} ({pct:5.1f}%) {bar}")
        
        stats["distribution"] = dict(zip(labels, hist.tolist()))
        return stats
    
    def _analyze_quality(self, library: PrototypeLibrary) -> Dict:
        """分析原型质量"""
        win_rates = []
        member_counts = []
        avg_profits = []
        
        for proto in library.get_all_prototypes():
            win_rates.append(proto.win_rate)
            member_counts.append(proto.member_count)
            avg_profits.append(proto.avg_profit_pct)
        
        if not win_rates:
            return {}
        
        stats = {
            "avg_win_rate": float(np.mean(win_rates)),
            "avg_member_count": float(np.mean(member_counts)),
            "total_members": int(np.sum(member_counts)),
            "avg_profit_pct": float(np.mean(avg_profits)),
        }
        
        print(f"    原型质量统计:")
        print(f"        平均胜率: {stats['avg_win_rate']*100:.1f}%")
        print(f"        平均成员数: {stats['avg_member_count']:.1f}")
        print(f"        总成员数: {stats['total_members']}")
        print(f"        平均收益率: {stats['avg_profit_pct']:.2f}%")
        
        # 高质量原型统计
        high_conf = sum(1 for p in library.get_all_prototypes() 
                       if p.confidence >= CONFIDENCE_CONFIG.get("HIGH_CONFIDENCE", 0.70))
        print(f"\n    高置信度原型: {high_conf}/{library.total_count} "
              f"({high_conf/library.total_count*100:.1f}%)")
        
        # 数据完整性检查
        print(f"\n    数据完整性检查:")
        n_has_centroid = sum(1 for p in library.get_all_prototypes() 
                            if p.pre_entry_centroid is not None and p.pre_entry_centroid.size == 32)
        n_has_repr = sum(1 for p in library.get_all_prototypes() 
                        if p.representative_pre_entry is not None and p.representative_pre_entry.size > 0)
        n_has_weighted = sum(1 for p in library.get_all_prototypes() 
                            if p.weighted_mean is not None and p.weighted_mean.size > 0)
        
        print(f"        有效中心向量: {n_has_centroid}/{library.total_count}")
        print(f"        有效代表序列 (DTW): {n_has_repr}/{library.total_count}")
        print(f"        有效加权均值 (v3): {n_has_weighted}/{library.total_count}")
        
        # 检查第一个原型的代表序列形状
        if library.long_prototypes:
            proto = library.long_prototypes[0]
            print(f"\n    样例原型数据 (第一个LONG):")
            print(f"        centroid shape: {proto.pre_entry_centroid.shape if proto.pre_entry_centroid is not None else 'None'}")
            print(f"        repr_pre_entry shape: {proto.representative_pre_entry.shape if proto.representative_pre_entry is not None and proto.representative_pre_entry.size > 0 else 'Empty'}")
            print(f"        weighted_mean shape: {proto.weighted_mean.shape if proto.weighted_mean is not None and proto.weighted_mean.size > 0 else 'Empty'}")
        
        stats["high_confidence_count"] = high_conf
        return stats
    
    def _test_similarity_distribution(self, library: PrototypeLibrary) -> Dict:
        """测试相似度分布"""
        # 从模板中随机抽取样本进行测试
        all_templates = []
        for regime in self.memory._templates:
            for direction in self.memory._templates[regime]:
                all_templates.extend(self.memory._templates[regime][direction])
        
        if not all_templates:
            print("    没有模板可测试")
            return {}
        
        # 随机抽取最多100个样本
        np.random.seed(42)
        n_samples = min(100, len(all_templates))
        sample_indices = np.random.choice(len(all_templates), n_samples, replace=False)
        
        matcher = PrototypeMatcher(library, enable_multi_similarity=True)
        calculator = MultiSimilarityCalculator()
        
        # 收集相似度分数
        cosine_scores = []
        euclidean_scores = []
        dtw_scores = []
        fusion_scores = []
        
        print(f"    测试样本数: {n_samples}")
        print(f"    DTW Radius: {calculator.dtw_radius}")
        print(f"    Euclidean Max Distance: {calculator.euclidean_max_dist}")
        print(f"    计算中...")
        
        # 调试：检查实际距离值
        debug_euclidean_distances = []
        debug_dtw_distances = []
        
        for idx in sample_indices:
            template = all_templates[idx]
            window = template.pre_entry
            
            if window is None or window.shape[0] < 10:
                continue
            
            direction = template.direction
            prototypes = library.get_prototypes_by_direction(direction)
            
            if not prototypes:
                continue
            
            # 计算与最佳匹配原型的相似度
            best_result = None
            best_fusion = -1
            
            for proto in prototypes[:10]:  # 限制为前10个以加速
                if proto.pre_entry_centroid is None or proto.pre_entry_centroid.size != 32:
                    continue
                
                # 使用多维相似度计算
                result = calculator.compute_similarity(
                    window, proto,
                    include_dtw=True
                )
                
                if result and result.get("combined_score", 0) > best_fusion:
                    best_fusion = result["combined_score"]
                    best_result = result
            
            if best_result:
                cosine_scores.append(best_result.get("cosine_similarity", 0))
                euclidean_scores.append(best_result.get("euclidean_similarity", 0))
                dtw_scores.append(best_result.get("dtw_similarity", 0))
                fusion_scores.append(best_result.get("combined_score", 0))
                
                # 调试：计算原始距离值
                if len(debug_euclidean_distances) < 5:  # 只记录前5个
                    try:
                        best_proto = prototypes[0]
                        window_mean = window.mean(axis=0) if window.ndim == 2 else window
                        proto_centroid = best_proto.pre_entry_centroid
                        raw_dist = np.linalg.norm(window_mean - proto_centroid)
                        debug_euclidean_distances.append(raw_dist)
                        
                        # 调试DTW：计算原始DTW距离
                        if best_proto.representative_pre_entry.size > 0:
                            seq1 = window
                            seq2 = best_proto.representative_pre_entry
                            if seq1.ndim == 2 and seq2.ndim == 2:
                                n, m = len(seq1), len(seq2)
                                # 简单累积距离估算
                                total_dist = 0
                                for i in range(min(n, m)):
                                    total_dist += np.linalg.norm(seq1[i] - seq2[i])
                                debug_dtw_distances.append(total_dist)
                    except:
                        pass
        
        if not fusion_scores:
            print("    无法计算相似度")
            return {}
        
        # 打印调试信息
        if debug_euclidean_distances:
            print(f"\n    [调试] 原始欧氏距离样本: {debug_euclidean_distances[:5]}")
            print(f"    [调试] 当前 euclidean_max_dist 配置: {calculator.euclidean_max_dist}")
            avg_dist = np.mean(debug_euclidean_distances)
            print(f"    [调试] 建议调整 euclidean_max_dist 为: {avg_dist * 2:.1f} 以获得更好的分布")
        
        if debug_dtw_distances:
            print(f"\n    [调试] 原始DTW累积距离样本: {debug_dtw_distances[:5]}")
            avg_dtw = np.mean(debug_dtw_distances)
            print(f"    [调试] DTW平均累积距离: {avg_dtw:.1f}")
            print(f"    [调试] DTW归一化参数 (max_len * sqrt(32) * 2): {60 * np.sqrt(32) * 2:.1f}")
            print(f"    [调试] 建议调整 DTW_MAX_DISTANCE 为: {avg_dtw * 1.5:.1f}")
        
        # 打印分布统计
        print(f"\n    多维相似度分布 (n={len(fusion_scores)}):")
        
        for name, scores in [
            ("余弦相似度", cosine_scores),
            ("欧氏距离相似度", euclidean_scores),
            ("DTW相似度", dtw_scores),
            ("融合分数", fusion_scores),
        ]:
            scores_arr = np.array(scores)
            print(f"\n    {name}:")
            print(f"        均值: {np.mean(scores_arr):.3f}")
            print(f"        标准差: {np.std(scores_arr):.3f}")
            print(f"        范围: [{np.min(scores_arr):.3f}, {np.max(scores_arr):.3f}]")
            
            # 分布直方图
            bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, _ = np.histogram(scores_arr, bins=bins)
            print(f"        分布:")
            for i in range(len(hist)):
                pct = hist[i] / len(scores_arr) * 100
                bar = "█" * int(pct / 3)
                print(f"            [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:3d} ({pct:5.1f}%) {bar}")
        
        return {
            "cosine": {"mean": float(np.mean(cosine_scores)), "std": float(np.std(cosine_scores))},
            "euclidean": {"mean": float(np.mean(euclidean_scores)), "std": float(np.std(euclidean_scores))},
            "dtw": {"mean": float(np.mean(dtw_scores)), "std": float(np.std(dtw_scores))},
            "fusion": {"mean": float(np.mean(fusion_scores)), "std": float(np.std(fusion_scores))},
            "sample_count": len(fusion_scores),
        }
    
    def _compare_libraries(self) -> Dict:
        """对比新旧原型库"""
        old_total = self.old_library.total_count
        new_total = self.library.total_count
        
        old_long = len(self.old_library.long_prototypes)
        new_long = len(self.library.long_prototypes)
        
        old_short = len(self.old_library.short_prototypes)
        new_short = len(self.library.short_prototypes)
        
        print(f"    原型数量变化:")
        print(f"        总数: {old_total} → {new_total} ({new_total - old_total:+d})")
        print(f"        LONG: {old_long} → {new_long} ({new_long - old_long:+d})")
        print(f"        SHORT: {old_short} → {new_short} ({new_short - old_short:+d})")
        
        # 置信度对比
        old_conf = [p.confidence for p in self.old_library.get_all_prototypes()]
        new_conf = [p.confidence for p in self.library.get_all_prototypes()]
        
        if old_conf and new_conf:
            print(f"\n    平均置信度变化:")
            print(f"        旧库: {np.mean(old_conf):.3f}")
            print(f"        新库: {np.mean(new_conf):.3f}")
            print(f"        变化: {np.mean(new_conf) - np.mean(old_conf):+.3f}")
        
        return {
            "old_total": old_total,
            "new_total": new_total,
            "old_long": old_long,
            "new_long": new_long,
            "old_short": old_short,
            "new_short": new_short,
        }


def main():
    parser = argparse.ArgumentParser(description="原型库重新训练和验证")
    parser.add_argument("--verify-only", action="store_true", 
                       help="仅验证现有原型库，不重新训练")
    parser.add_argument("--n-long", type=int, default=None,
                       help="LONG 方向聚类数")
    parser.add_argument("--n-short", type=int, default=None,
                       help="SHORT 方向聚类数")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存新原型库（仅用于测试）")
    
    args = parser.parse_args()
    
    print("\n" + "═" * 60)
    print("    R3000 原型库重新训练与验证工具")
    print("    " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═" * 60)
    
    retrainer = PrototypeRetrainer(
        n_clusters_long=args.n_long,
        n_clusters_short=args.n_short,
    )
    
    if args.verify_only:
        # 仅验证模式
        print("\n[MODE] 仅验证模式")
        
        # 加载现有原型库
        if not retrainer.load_old_library():
            print("\n[ERROR] 没有找到现有原型库，无法验证")
            return
        
        # 加载模板（用于相似度测试）
        retrainer.load_memory()
        
        # 验证
        results = retrainer.verify()
    else:
        # 完整重训练模式
        print("\n[MODE] 完整重训练模式")
        
        # 加载模板库
        if not retrainer.load_memory():
            return
        
        # 加载旧库（用于对比）
        retrainer.load_old_library()
        
        # 重新训练
        if not retrainer.retrain():
            return
        
        # 保存
        if not args.no_save:
            retrainer.save_library()
        
        # 验证
        results = retrainer.verify()
    
    # 总结
    print("\n" + "═" * 60)
    print("    验证完成")
    print("═" * 60)
    
    if "similarity" in results:
        fusion = results["similarity"].get("fusion", {})
        print(f"\n[关键指标] 融合分数分布:")
        print(f"    均值: {fusion.get('mean', 0):.3f}")
        print(f"    标准差: {fusion.get('std', 0):.3f}")
        
        # 检查是否达到预期效果
        if fusion.get('std', 0) > 0.1:
            print("\n[SUCCESS] 相似度分布已分散（标准差 > 0.1）")
        else:
            print("\n[WARNING] 相似度分布仍然集中（标准差 <= 0.1）")
    
    print("\n" + "═" * 60)
    print("    脚本执行完成")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
