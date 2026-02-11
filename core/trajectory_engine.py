"""
R3000 轨迹模板引擎
从上帝视角盈利交易中提取三段轨迹模板：入场前、持仓中、离场前
按市场状态和方向分组存储，用于后续DTW匹配

核心数据结构：
  TrajectoryTemplate - 单笔交易的三段轨迹
  TrajectoryMemory   - 按 regime x direction 分组的模板库

持久化存储：
  - 支持 save()/load() 将记忆体保存到本地
  - 支持 merge() 增量合并多个记忆体
  - 跨会话积累记忆
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os
import pickle
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAJECTORY_CONFIG

# 默认记忆存储路径
DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "memory"
)


@dataclass
class TrajectoryTemplate:
    """单笔盈利交易的轨迹模板"""
    trade_idx: int              # 原始交易索引
    regime: str                 # 市场状态
    direction: str              # "LONG" / "SHORT"
    profit_pct: float           # 收益率百分比

    # 三段轨迹矩阵（原始值，不标准化）
    pre_entry: np.ndarray       # (pre_entry_window, 32) 入场前
    holding: np.ndarray         # (hold_periods, 32) 持仓中
    pre_exit: np.ndarray        # (pre_exit_window, 32) 离场前

    # 用于余弦粗筛的预计算展平向量（z-score标准化后展平）
    pre_entry_flat: np.ndarray = field(default_factory=lambda: np.array([]))

    # 交易的原始索引信息
    entry_idx: int = 0
    exit_idx: int = 0

    def __post_init__(self):
        """初始化时预计算展平向量"""
        if self.pre_entry.size > 0 and self.pre_entry_flat.size == 0:
            self.pre_entry_flat = self._zscore_flatten(self.pre_entry)

    @staticmethod
    def _zscore_flatten(matrix: np.ndarray) -> np.ndarray:
        """z-score标准化后展平为一维向量"""
        if matrix.size == 0:
            return np.array([])
        flat = matrix.flatten()
        std = flat.std()
        if std < 1e-9:
            return flat - flat.mean()
        return (flat - flat.mean()) / std

    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "trade_idx": self.trade_idx,
            "regime": self.regime,
            "direction": self.direction,
            "profit_pct": self.profit_pct,
            "pre_entry": self.pre_entry.tolist(),
            "holding": self.holding.tolist(),
            "pre_exit": self.pre_exit.tolist(),
            "entry_idx": self.entry_idx,
            "exit_idx": self.exit_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrajectoryTemplate':
        """从字典还原"""
        return cls(
            trade_idx=d["trade_idx"],
            regime=d["regime"],
            direction=d["direction"],
            profit_pct=d["profit_pct"],
            pre_entry=np.array(d["pre_entry"]),
            holding=np.array(d["holding"]),
            pre_exit=np.array(d["pre_exit"]),
            entry_idx=d.get("entry_idx", 0),
            exit_idx=d.get("exit_idx", 0),
        )

    def fingerprint(self) -> str:
        """生成模板的唯一指纹（用于去重）"""
        # 使用入场轨迹的哈希作为指纹
        data = (self.regime, self.direction, self.profit_pct,
                self.pre_entry.tobytes()[:1000])  # 取前1000字节避免太长
        return str(hash(data))


class TrajectoryMemory:
    """
    轨迹模板库

    结构：templates[regime][direction] = List[TrajectoryTemplate]

    使用方式：
        memory = TrajectoryMemory()
        memory.extract_from_trades(trades, fv_engine, regime_map)
        candidates = memory.get_candidates("强多头", "LONG")
    """

    def __init__(self, pre_entry_window: int = None, pre_exit_window: int = None,
                 min_profit_pct: float = None):
        self.pre_entry_window = pre_entry_window or TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
        self.pre_exit_window = pre_exit_window or TRAJECTORY_CONFIG["PRE_EXIT_WINDOW"]
        self.min_profit_pct = min_profit_pct if min_profit_pct is not None else TRAJECTORY_CONFIG["MIN_PROFIT_PCT"]

        # 模板存储：templates[regime][direction] = [TrajectoryTemplate, ...]
        self._templates: Dict[str, Dict[str, List[TrajectoryTemplate]]] = {}
        self._total_count = 0

    def extract_from_trades(self, trades: list, fv_engine, regime_map: Dict[int, str],
                            verbose: bool = True) -> int:
        """
        从交易列表中提取盈利交易的轨迹模板

        Args:
            trades: TradeRecord 列表
            fv_engine: FeatureVectorEngine 实例（已预计算）
            regime_map: {trade_idx: regime_string}
            verbose: 是否打印统计

        Returns:
            提取的模板数量
        """
        self._templates.clear()
        self._total_count = 0
        skipped_loss = 0
        skipped_short = 0

        for i, trade in enumerate(trades):
            # 只提取盈利交易
            if trade.profit_pct < self.min_profit_pct:
                skipped_loss += 1
                continue

            # 检查入场前是否有足够K线
            if trade.entry_idx < self.pre_entry_window:
                skipped_short += 1
                continue

            regime = regime_map.get(i, "未知")
            direction = "LONG" if trade.side == 1 else "SHORT"

            # 提取三段轨迹
            pre_entry = fv_engine.get_raw_matrix(
                trade.entry_idx - self.pre_entry_window,
                trade.entry_idx
            )

            holding = fv_engine.get_raw_matrix(
                trade.entry_idx,
                trade.exit_idx + 1
            )

            # 离场前轨迹：取持仓末尾或完整pre_exit_window
            exit_start = max(trade.entry_idx, trade.exit_idx - self.pre_exit_window + 1)
            pre_exit = fv_engine.get_raw_matrix(exit_start, trade.exit_idx + 1)

            template = TrajectoryTemplate(
                trade_idx=i,
                regime=regime,
                direction=direction,
                profit_pct=trade.profit_pct,
                pre_entry=pre_entry,
                holding=holding,
                pre_exit=pre_exit,
                entry_idx=trade.entry_idx,
                exit_idx=trade.exit_idx,
            )

            self._add_template(regime, direction, template)

        if verbose:
            print(f"[TrajectoryMemory] 提取模板: {self._total_count} 个 "
                  f"(跳过亏损: {skipped_loss}, 跳过K线不足: {skipped_short})")
            self._print_stats()

        return self._total_count

    def _add_template(self, regime: str, direction: str, template: TrajectoryTemplate):
        """添加模板到库中"""
        if regime not in self._templates:
            self._templates[regime] = {}
        if direction not in self._templates[regime]:
            self._templates[regime][direction] = []
        self._templates[regime][direction].append(template)
        self._total_count += 1

    def get_candidates(self, regime: str, direction: str) -> List[TrajectoryTemplate]:
        """获取指定市场状态和方向的所有模板"""
        if regime not in self._templates:
            return []
        if direction not in self._templates[regime]:
            return []
        return self._templates[regime][direction]

    def get_all_templates(self) -> List[TrajectoryTemplate]:
        """获取所有模板（用于全局匹配）"""
        all_templates = []
        for regime_dict in self._templates.values():
            for direction_list in regime_dict.values():
                all_templates.extend(direction_list)
        return all_templates

    def get_templates_by_direction(self, direction: str) -> List[TrajectoryTemplate]:
        """获取指定方向的所有模板（跨市场状态）"""
        templates = []
        for regime_dict in self._templates.values():
            if direction in regime_dict:
                templates.extend(regime_dict[direction])
        return templates

    def get_regime_list(self) -> List[str]:
        """获取所有市场状态"""
        return list(self._templates.keys())

    @property
    def total_count(self) -> int:
        return self._total_count

    def _print_stats(self):
        """打印分布统计"""
        for regime in sorted(self._templates.keys()):
            for direction in ["LONG", "SHORT"]:
                templates = self.get_candidates(regime, direction)
                if templates:
                    avg_profit = np.mean([t.profit_pct for t in templates])
                    print(f"    {regime}/{direction}: {len(templates)} 个模板, "
                          f"平均收益 {avg_profit:.2f}%")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {}
        for regime in self._templates:
            stats[regime] = {}
            for direction in self._templates[regime]:
                templates = self._templates[regime][direction]
                stats[regime][direction] = {
                    "count": len(templates),
                    "avg_profit": np.mean([t.profit_pct for t in templates]) if templates else 0,
                    "avg_hold": np.mean([t.holding.shape[0] for t in templates]) if templates else 0,
                }
        return stats

    def clear(self):
        """清空模板库"""
        self._templates.clear()
        self._total_count = 0

    # ══════════════════════════════════════════════════════════════════════════
    # 持久化存储方法
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, filepath: str = None, verbose: bool = True) -> str:
        """
        将记忆体保存到本地文件

        Args:
            filepath: 保存路径，None则使用默认路径（带时间戳）
            verbose: 是否打印信息

        Returns:
            实际保存的文件路径
        """
        if filepath is None:
            # 确保目录存在
            os.makedirs(DEFAULT_MEMORY_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(DEFAULT_MEMORY_DIR, f"memory_{timestamp}.pkl")

        # 构建可序列化的数据结构
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "config": {
                "pre_entry_window": self.pre_entry_window,
                "pre_exit_window": self.pre_exit_window,
                "min_profit_pct": self.min_profit_pct,
            },
            "templates": {},
            "stats": {
                "total_count": self._total_count,
                "regime_count": len(self._templates),
            }
        }

        # 序列化所有模板
        for regime in self._templates:
            data["templates"][regime] = {}
            for direction in self._templates[regime]:
                data["templates"][regime][direction] = [
                    t.to_dict() for t in self._templates[regime][direction]
                ]

        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[TrajectoryMemory] 已保存: {filepath}")
            print(f"    模板数: {self._total_count}, 文件大小: {size_mb:.2f} MB")

        return filepath

    @classmethod
    def load(cls, filepath: str, verbose: bool = True) -> 'TrajectoryMemory':
        """
        从本地文件加载记忆体

        Args:
            filepath: 文件路径
            verbose: 是否打印信息

        Returns:
            加载的 TrajectoryMemory 实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # 验证版本
        version = data.get("version", "unknown")
        if version != "1.0":
            print(f"[TrajectoryMemory] 警告: 文件版本 {version}, 当前支持 1.0")

        # 还原配置
        config = data.get("config", {})
        memory = cls(
            pre_entry_window=config.get("pre_entry_window"),
            pre_exit_window=config.get("pre_exit_window"),
            min_profit_pct=config.get("min_profit_pct"),
        )

        # 还原所有模板
        for regime in data.get("templates", {}):
            for direction in data["templates"][regime]:
                for t_dict in data["templates"][regime][direction]:
                    template = TrajectoryTemplate.from_dict(t_dict)
                    memory._add_template(regime, direction, template)

        if verbose:
            created_at = data.get("created_at", "未知")
            print(f"[TrajectoryMemory] 已加载: {filepath}")
            print(f"    创建时间: {created_at}, 模板数: {memory._total_count}")

        return memory

    def merge(self, other: 'TrajectoryMemory', deduplicate: bool = True,
              verbose: bool = True) -> int:
        """
        合并另一个记忆体（增量添加，不覆盖）

        Args:
            other: 要合并的另一个 TrajectoryMemory
            deduplicate: 是否去重（基于指纹）
            verbose: 是否打印信息

        Returns:
            新增的模板数量
        """
        added = 0
        skipped = 0

        # 如果需要去重，先收集已有模板的指纹
        existing_fingerprints = set()
        if deduplicate:
            for t in self.get_all_templates():
                existing_fingerprints.add(t.fingerprint())

        # 遍历另一个记忆体的所有模板
        for regime in other._templates:
            for direction in other._templates[regime]:
                for template in other._templates[regime][direction]:
                    # 去重检查
                    if deduplicate:
                        fp = template.fingerprint()
                        if fp in existing_fingerprints:
                            skipped += 1
                            continue
                        existing_fingerprints.add(fp)

                    self._add_template(regime, direction, template)
                    added += 1

        if verbose:
            print(f"[TrajectoryMemory] 合并完成: 新增 {added} 个模板"
                  + (f", 跳过重复 {skipped} 个" if skipped > 0 else ""))

        return added

    def merge_from_file(self, filepath: str, deduplicate: bool = True,
                        verbose: bool = True) -> int:
        """
        从文件加载并合并到当前记忆体

        Args:
            filepath: 文件路径
            deduplicate: 是否去重
            verbose: 是否打印信息

        Returns:
            新增的模板数量
        """
        other = TrajectoryMemory.load(filepath, verbose=False)
        return self.merge(other, deduplicate, verbose)

    @staticmethod
    def list_saved_memories(memory_dir: str = None) -> List[Dict]:
        """
        列出所有已保存的记忆体文件

        Args:
            memory_dir: 记忆目录，None则使用默认目录

        Returns:
            [{"path": str, "size_mb": float, "modified": str}, ...]
        """
        if memory_dir is None:
            memory_dir = DEFAULT_MEMORY_DIR

        if not os.path.exists(memory_dir):
            return []

        files = []
        for filename in os.listdir(memory_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(memory_dir, filename)
                stat = os.stat(filepath)
                files.append({
                    "path": filepath,
                    "filename": filename,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        # 按修改时间排序（最新的在前）
        files.sort(key=lambda x: x["modified"], reverse=True)
        return files

    @classmethod
    def load_latest(cls, memory_dir: str = None, verbose: bool = True) -> Optional['TrajectoryMemory']:
        """
        加载最新的记忆体文件

        Args:
            memory_dir: 记忆目录
            verbose: 是否打印信息

        Returns:
            TrajectoryMemory 实例，如果没有文件则返回 None
        """
        files = cls.list_saved_memories(memory_dir)
        if not files:
            if verbose:
                print("[TrajectoryMemory] 没有找到已保存的记忆体")
            return None

        return cls.load(files[0]["path"], verbose=verbose)

    @classmethod
    def load_and_merge_all(cls, memory_dir: str = None,
                           verbose: bool = True) -> 'TrajectoryMemory':
        """
        加载并合并目录下所有记忆体文件

        Args:
            memory_dir: 记忆目录
            verbose: 是否打印信息

        Returns:
            合并后的 TrajectoryMemory 实例
        """
        files = cls.list_saved_memories(memory_dir)
        if not files:
            if verbose:
                print("[TrajectoryMemory] 没有找到已保存的记忆体，返回空实例")
            return cls()

        # 加载第一个作为基础
        memory = cls.load(files[0]["path"], verbose=False)
        total_added = memory._total_count

        # 合并其余文件
        for f in files[1:]:
            added = memory.merge_from_file(f["path"], deduplicate=True, verbose=False)
            total_added += added

        if verbose:
            print(f"[TrajectoryMemory] 已加载并合并 {len(files)} 个文件")
            print(f"    总模板数: {memory._total_count}")

        return memory

    def export_stats_json(self, filepath: str = None) -> str:
        """
        导出统计信息为 JSON（便于查看）

        Args:
            filepath: 保存路径，None则使用默认路径

        Returns:
            保存的文件路径
        """
        if filepath is None:
            os.makedirs(DEFAULT_MEMORY_DIR, exist_ok=True)
            filepath = os.path.join(DEFAULT_MEMORY_DIR, "memory_stats.json")

        stats = {
            "generated_at": datetime.now().isoformat(),
            "total_templates": self._total_count,
            "regimes": {},
        }

        for regime in self._templates:
            stats["regimes"][regime] = {}
            for direction in self._templates[regime]:
                templates = self._templates[regime][direction]
                profits = [t.profit_pct for t in templates]
                stats["regimes"][regime][direction] = {
                    "count": len(templates),
                    "avg_profit_pct": float(np.mean(profits)) if profits else 0,
                    "min_profit_pct": float(np.min(profits)) if profits else 0,
                    "max_profit_pct": float(np.max(profits)) if profits else 0,
                }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return filepath

    def get_template_with_global_index(self, global_idx: int) -> Optional[TrajectoryTemplate]:
        """
        根据全局索引获取模板

        Args:
            global_idx: 全局模板索引（跨所有regime/direction）

        Returns:
            模板或None
        """
        current = 0
        for regime in sorted(self._templates.keys()):
            for direction in ["LONG", "SHORT"]:
                templates = self.get_candidates(regime, direction)
                if current + len(templates) > global_idx:
                    return templates[global_idx - current]
                current += len(templates)
        return None

    def filter_by_fingerprints(self, keep_fingerprints: set, verbose: bool = True) -> int:
        """
        根据指纹集合过滤模板，只保留指纹在集合中的模板

        Args:
            keep_fingerprints: 要保留的模板指纹集合
            verbose: 是否打印信息

        Returns:
            删除的模板数量
        """
        removed = 0
        new_templates = {}

        for regime in self._templates:
            new_templates[regime] = {}
            for direction in self._templates[regime]:
                kept = []
                for template in self._templates[regime][direction]:
                    if template.fingerprint() in keep_fingerprints:
                        kept.append(template)
                    else:
                        removed += 1
                if kept:
                    new_templates[regime][direction] = kept

        # 清理空的regime
        self._templates = {r: d for r, d in new_templates.items() if d}
        self._total_count = sum(
            len(templates)
            for regime_dict in self._templates.values()
            for templates in regime_dict.values()
        )

        if verbose:
            print(f"[TrajectoryMemory] 过滤完成: 删除 {removed} 个模板, 保留 {self._total_count} 个")

        return removed

    def remove_by_fingerprints(self, remove_fingerprints: set, verbose: bool = True) -> int:
        """
        根据指纹集合删除模板

        Args:
            remove_fingerprints: 要删除的模板指纹集合
            verbose: 是否打印信息

        Returns:
            删除的模板数量
        """
        all_templates = self.get_all_templates()
        keep_fingerprints = set()
        for t in all_templates:
            fp = t.fingerprint()
            if fp not in remove_fingerprints:
                keep_fingerprints.add(fp)

        return self.filter_by_fingerprints(keep_fingerprints, verbose)


def extract_current_trajectory(fv_engine, current_idx: int,
                                pre_window: int = None) -> np.ndarray:
    """
    提取当前K线的入场前轨迹（用于实时匹配）

    Args:
        fv_engine: FeatureVectorEngine 实例
        current_idx: 当前K线索引
        pre_window: 回看窗口大小

    Returns:
        (pre_window, 32) 的特征矩阵
    """
    if pre_window is None:
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]

    start_idx = max(0, current_idx - pre_window + 1)
    return fv_engine.get_raw_matrix(start_idx, current_idx + 1)
