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
                 min_profit_pct: float = None,
                 source_symbol: str = "",
                 source_interval: str = ""):
        self.pre_entry_window = pre_entry_window or TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
        self.pre_exit_window = pre_exit_window or TRAJECTORY_CONFIG["PRE_EXIT_WINDOW"]
        self.min_profit_pct = min_profit_pct if min_profit_pct is not None else TRAJECTORY_CONFIG["MIN_PROFIT_PCT"]
        # 记忆库来源信息（用于和实时交易配置做一致性校验）
        self.source_symbol = (source_symbol or "").upper()
        self.source_interval = (source_interval or "").strip()

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

            regime = regime_map.get(i, "")
            direction = "LONG" if trade.side == 1 else "SHORT"
            
            # 【消除 regime="未知"】如果 regime 为空或"未知"，根据盈利方向推断默认 regime
            # 逻辑：LONG 盈利交易至少在偏多环境，SHORT 盈利交易至少在偏空环境
            from core.market_regime import MarketRegime
            if not regime or regime == MarketRegime.UNKNOWN:
                regime = MarketRegime.RANGE_BULL if direction == "LONG" else MarketRegime.RANGE_BEAR

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
    
    def get_template_by_fingerprint(self, fingerprint: str) -> Optional[TrajectoryTemplate]:
        """按指纹查找模板"""
        if not fingerprint:
            return None
        for t in self.get_all_templates():
            if t.fingerprint() == fingerprint:
                return t
        return None

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
                "source_symbol": self.source_symbol,
                "source_interval": self.source_interval,
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
            source_symbol=config.get("source_symbol", ""),
            source_interval=config.get("source_interval", ""),
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


# ══════════════════════════════════════════════════════════════════════════════
# 轨迹重建工具函数（从历史K线数据重建轨迹）
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_trajectory_from_klines(
    klines: 'pd.DataFrame',
    entry_time: datetime,
    pre_window: int = None,
    warmup_bars: int = 200
) -> Optional[np.ndarray]:
    """
    从历史K线数据重建入场前轨迹矩阵
    
    用于为历史交易记录（未保存轨迹数据的订单）补充轨迹矩阵，
    以便进行增量训练或回测分析。
    
    Args:
        klines: 包含 OHLCV 数据的 DataFrame，必须包含 timestamp/open_time 列
                （或索引为时间戳）以及 open/high/low/close/volume 列
        entry_time: 入场时间 (datetime 对象)
        pre_window: 入场前回看窗口大小（默认使用 TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]）
        warmup_bars: 预热期K线数量（用于计算技术指标）
    
    Returns:
        (pre_window, 32) 的轨迹矩阵，如果数据不足或出错则返回 None
    
    Example:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> from core.trajectory_engine import rebuild_trajectory_from_klines
        >>> 
        >>> # 加载历史K线数据
        >>> klines = pd.read_parquet("btcusdt_1m.parquet")
        >>> 
        >>> # 为某笔历史交易重建轨迹
        >>> entry_time = datetime(2024, 1, 15, 10, 30, 0)
        >>> trajectory = rebuild_trajectory_from_klines(klines, entry_time)
        >>> if trajectory is not None:
        ...     print(f"轨迹形状: {trajectory.shape}")  # (60, 32)
    """
    import pandas as pd
    from utils.indicators import calculate_all_indicators
    from core.feature_vector import FeatureVectorEngine
    
    if pre_window is None:
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
    
    # 确保 klines 是 DataFrame
    if not isinstance(klines, pd.DataFrame):
        print("[TrajectoryRebuild] 错误: klines 必须是 pandas DataFrame")
        return None
    
    # 查找时间列
    time_col = None
    for col in ['timestamp', 'open_time', 'time', 'datetime']:
        if col in klines.columns:
            time_col = col
            break
    
    # 如果没有时间列，尝试使用索引
    if time_col is None:
        if isinstance(klines.index, pd.DatetimeIndex):
            klines = klines.reset_index()
            time_col = klines.columns[0]
        else:
            print("[TrajectoryRebuild] 错误: 未找到时间列，请确保 DataFrame 包含 timestamp/open_time/time/datetime 列")
            return None
    
    # 转换时间列为 datetime
    try:
        if klines[time_col].dtype in ['int64', 'float64']:
            # 毫秒时间戳
            klines[time_col] = pd.to_datetime(klines[time_col], unit='ms')
        else:
            klines[time_col] = pd.to_datetime(klines[time_col])
    except Exception as e:
        print(f"[TrajectoryRebuild] 时间列转换失败: {e}")
        return None
    
    # 确保 entry_time 是 datetime 对象
    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time)
    
    # 移除时区信息以便比较（如果有）
    if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is not None:
        entry_time = entry_time.replace(tzinfo=None)
    
    # 查找入场时间对应的K线索引
    klines_sorted = klines.sort_values(time_col).reset_index(drop=True)
    
    # 确保 K 线时间戳也没有时区信息
    kline_times = klines_sorted[time_col]
    if hasattr(kline_times.iloc[0], 'tzinfo') and kline_times.iloc[0].tzinfo is not None:
        klines_sorted[time_col] = kline_times.dt.tz_localize(None)
    
    # 查找最接近 entry_time 的 K 线
    time_diffs = abs(klines_sorted[time_col] - entry_time)
    entry_idx = time_diffs.idxmin()
    
    # 检查时间差是否在合理范围内（1分钟K线允许1分钟偏差）
    min_diff = time_diffs.min()
    if min_diff > pd.Timedelta(minutes=2):
        print(f"[TrajectoryRebuild] 警告: 最接近的K线时间差为 {min_diff}，可能数据不匹配")
    
    # 检查是否有足够的数据（预热期 + 回看窗口）
    required_bars = warmup_bars + pre_window
    if entry_idx < required_bars - 1:
        print(f"[TrajectoryRebuild] 数据不足: 需要 {required_bars} 根K线，入场索引 {entry_idx}")
        return None
    
    # 提取计算所需的K线范围（包含预热期）
    start_idx = max(0, entry_idx - required_bars + 1)
    end_idx = entry_idx + 1
    df_slice = klines_sorted.iloc[start_idx:end_idx].copy()
    
    # 计算技术指标
    try:
        df_with_indicators = calculate_all_indicators(df_slice)
    except Exception as e:
        print(f"[TrajectoryRebuild] 计算指标失败: {e}")
        return None
    
    # 创建特征向量引擎并预计算
    try:
        fv_engine = FeatureVectorEngine()
        fv_engine.precompute(df_with_indicators)
    except Exception as e:
        print(f"[TrajectoryRebuild] 特征向量计算失败: {e}")
        return None
    
    # 提取轨迹矩阵（相对于切片后的索引）
    local_entry_idx = len(df_with_indicators) - 1
    trajectory = fv_engine.get_raw_matrix(
        local_entry_idx - pre_window + 1,
        local_entry_idx + 1
    )
    
    # 验证轨迹形状
    if trajectory.shape != (pre_window, 32):
        print(f"[TrajectoryRebuild] 轨迹形状异常: {trajectory.shape}，期望 ({pre_window}, 32)")
        return None
    
    return trajectory


def rebuild_trajectory_batch(
    klines: 'pd.DataFrame',
    entry_times: List[datetime],
    pre_window: int = None,
    warmup_bars: int = 200,
    verbose: bool = True
) -> List[Optional[np.ndarray]]:
    """
    批量重建多笔交易的轨迹矩阵
    
    优化版本：共享技术指标计算，避免重复计算
    
    Args:
        klines: 包含 OHLCV 数据的 DataFrame
        entry_times: 入场时间列表
        pre_window: 入场前回看窗口大小
        warmup_bars: 预热期K线数量
        verbose: 是否打印进度信息
    
    Returns:
        与 entry_times 等长的轨迹矩阵列表，无法重建的位置为 None
    
    Example:
        >>> from core.trajectory_engine import rebuild_trajectory_batch
        >>> 
        >>> entry_times = [datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 16, 14, 0)]
        >>> trajectories = rebuild_trajectory_batch(klines, entry_times)
        >>> success_count = sum(1 for t in trajectories if t is not None)
        >>> print(f"成功重建: {success_count}/{len(entry_times)}")
    """
    import pandas as pd
    from utils.indicators import calculate_all_indicators
    from core.feature_vector import FeatureVectorEngine
    
    if pre_window is None:
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
    
    results = [None] * len(entry_times)
    
    if not entry_times:
        return results
    
    # 查找时间列
    time_col = None
    for col in ['timestamp', 'open_time', 'time', 'datetime']:
        if col in klines.columns:
            time_col = col
            break
    
    if time_col is None:
        if isinstance(klines.index, pd.DatetimeIndex):
            klines = klines.reset_index()
            time_col = klines.columns[0]
        else:
            print("[TrajectoryRebuild] 错误: 未找到时间列")
            return results
    
    # 转换时间列
    klines = klines.copy()
    try:
        if klines[time_col].dtype in ['int64', 'float64']:
            klines[time_col] = pd.to_datetime(klines[time_col], unit='ms')
        else:
            klines[time_col] = pd.to_datetime(klines[time_col])
    except Exception as e:
        print(f"[TrajectoryRebuild] 时间列转换失败: {e}")
        return results
    
    # 排序并重置索引
    klines_sorted = klines.sort_values(time_col).reset_index(drop=True)
    
    # 移除时区信息
    if hasattr(klines_sorted[time_col].iloc[0], 'tzinfo') and klines_sorted[time_col].iloc[0].tzinfo is not None:
        klines_sorted[time_col] = klines_sorted[time_col].dt.tz_localize(None)
    
    # 一次性计算所有技术指标
    if verbose:
        print(f"[TrajectoryRebuild] 计算技术指标 ({len(klines_sorted)} 根K线)...")
    
    try:
        df_with_indicators = calculate_all_indicators(klines_sorted)
    except Exception as e:
        print(f"[TrajectoryRebuild] 计算指标失败: {e}")
        return results
    
    # 创建特征向量引擎
    try:
        fv_engine = FeatureVectorEngine()
        fv_engine.precompute(df_with_indicators)
    except Exception as e:
        print(f"[TrajectoryRebuild] 特征向量计算失败: {e}")
        return results
    
    # 批量处理每个入场时间
    success_count = 0
    for i, entry_time in enumerate(entry_times):
        # 处理时间格式
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time)
            except ValueError:
                continue
        
        if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is not None:
            entry_time = entry_time.replace(tzinfo=None)
        
        # 查找最接近的 K 线索引
        time_diffs = abs(df_with_indicators[time_col] - entry_time)
        entry_idx = time_diffs.idxmin()
        
        # 检查数据是否充足
        required_bars = warmup_bars + pre_window
        if entry_idx < required_bars - 1:
            continue
        
        # 提取轨迹
        trajectory = fv_engine.get_raw_matrix(
            entry_idx - pre_window + 1,
            entry_idx + 1
        )
        
        if trajectory.shape == (pre_window, 32):
            results[i] = trajectory
            success_count += 1
    
    if verbose:
        print(f"[TrajectoryRebuild] 完成: {success_count}/{len(entry_times)} 笔交易轨迹重建成功")
    
    return results


def rebuild_orders_trajectories(
    orders: List['PaperOrder'],
    klines: 'pd.DataFrame',
    pre_window: int = None,
    warmup_bars: int = 200,
    skip_existing: bool = True,
    verbose: bool = True
) -> int:
    """
    为 PaperOrder 列表批量重建缺失的轨迹数据
    
    直接修改 orders 列表中的 entry_trajectory 属性
    
    Args:
        orders: PaperOrder 对象列表
        klines: 历史K线数据 DataFrame
        pre_window: 入场前回看窗口大小
        warmup_bars: 预热期K线数量
        skip_existing: 是否跳过已有轨迹数据的订单
        verbose: 是否打印进度信息
    
    Returns:
        成功重建轨迹的订单数量
    
    Example:
        >>> from core.paper_trader import load_trade_history_from_file
        >>> from core.trajectory_engine import rebuild_orders_trajectories
        >>> 
        >>> # 加载历史订单
        >>> orders = load_trade_history_from_file("data/paper_trading/history.json")
        >>> 
        >>> # 加载K线数据
        >>> klines = pd.read_parquet("btcusdt_1m.parquet")
        >>> 
        >>> # 重建轨迹
        >>> rebuilt_count = rebuild_orders_trajectories(orders, klines)
        >>> print(f"重建了 {rebuilt_count} 笔订单的轨迹")
    """
    if pre_window is None:
        pre_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
    
    # 筛选需要重建轨迹的订单
    orders_to_rebuild = []
    indices_to_rebuild = []
    
    for i, order in enumerate(orders):
        needs_rebuild = (
            order.entry_trajectory is None or 
            (isinstance(order.entry_trajectory, np.ndarray) and order.entry_trajectory.size == 0)
        )
        
        if skip_existing and not needs_rebuild:
            continue
        
        if order.entry_time is not None:
            orders_to_rebuild.append(order)
            indices_to_rebuild.append(i)
    
    if not orders_to_rebuild:
        if verbose:
            print("[TrajectoryRebuild] 没有需要重建轨迹的订单")
        return 0
    
    if verbose:
        print(f"[TrajectoryRebuild] 准备重建 {len(orders_to_rebuild)} 笔订单的轨迹...")
    
    # 提取入场时间列表
    entry_times = [order.entry_time for order in orders_to_rebuild]
    
    # 批量重建
    trajectories = rebuild_trajectory_batch(
        klines, entry_times, pre_window, warmup_bars, verbose=verbose
    )
    
    # 更新订单
    rebuilt_count = 0
    for idx, trajectory in zip(indices_to_rebuild, trajectories):
        if trajectory is not None:
            orders[idx].entry_trajectory = trajectory
            rebuilt_count += 1
    
    if verbose:
        print(f"[TrajectoryRebuild] 成功为 {rebuilt_count}/{len(orders_to_rebuild)} 笔订单重建轨迹")
    
    return rebuilt_count


# ══════════════════════════════════════════════════════════════════════════════
# 轨迹格式转换工具
# ══════════════════════════════════════════════════════════════════════════════

def convert_trajectory_to_weighted(
    trajectory: np.ndarray,
    layer_weights: Tuple[float, float, float] = None
) -> np.ndarray:
    """
    将原始轨迹矩阵转换为加权版本
    
    用于指纹3D图匹配系统：根据特征重要性对轨迹应用层级权重
    
    Args:
        trajectory: (time, 32) 原始轨迹矩阵
        layer_weights: (w_a, w_b, w_c) 三层权重，None使用配置默认值
    
    Returns:
        (time, 32) 加权后的轨迹矩阵
    """
    from core.feature_vector import FeatureVectorEngine
    
    if trajectory is None or trajectory.size == 0:
        return np.array([])
    
    return FeatureVectorEngine.apply_weights(
        trajectory,
        layer_weights=layer_weights,
        normalize=False
    )


def extract_trajectory_features(
    trajectory: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    从轨迹矩阵提取多维特征（用于指纹3D图匹配）
    
    返回结构化的特征表示，包含：
    - raw_sequence: 原始轨迹序列（用于DTW）
    - weighted_mean: 加权均值向量（用于快速匹配）
    - trend_vector: 趋势特征向量
    - temporal_stats: 时间维度统计特征
    
    Args:
        trajectory: (time, 32) 轨迹矩阵
    
    Returns:
        特征字典
    
    Example:
        >>> features = extract_trajectory_features(trajectory)
        >>> print(features.keys())
        dict_keys(['raw_sequence', 'weighted_mean', 'trend_vector', 'temporal_stats'])
    """
    from core.feature_vector import FeatureVectorEngine, apply_weights, extract_trends
    
    if trajectory is None or trajectory.size == 0 or trajectory.ndim != 2:
        return {
            'raw_sequence': np.array([]),
            'weighted_mean': np.zeros(32),
            'trend_vector': np.zeros(32),
            'temporal_stats': {},
        }
    
    # 原始序列
    raw_sequence = trajectory.copy()
    
    # 加权均值
    weighted_trajectory = apply_weights(trajectory)
    weighted_mean = weighted_trajectory.mean(axis=0)
    
    # 趋势特征
    trend_vector = extract_trends(trajectory)
    
    # 时间维度统计特征
    temporal_stats = {
        'mean': trajectory.mean(axis=0),            # 均值向量
        'std': trajectory.std(axis=0),              # 标准差向量
        'min': trajectory.min(axis=0),              # 最小值向量
        'max': trajectory.max(axis=0),              # 最大值向量
        'first': trajectory[0],                     # 首行特征
        'last': trajectory[-1],                     # 末行特征
        'range': trajectory.max(axis=0) - trajectory.min(axis=0),  # 范围向量
    }
    
    return {
        'raw_sequence': raw_sequence,
        'weighted_mean': weighted_mean,
        'trend_vector': trend_vector,
        'temporal_stats': temporal_stats,
    }


def validate_trajectory(
    trajectory: np.ndarray,
    expected_window: int = None,
    check_nan: bool = True,
    check_zero: bool = True
) -> Tuple[bool, str]:
    """
    验证轨迹矩阵的有效性
    
    Args:
        trajectory: 待验证的轨迹矩阵
        expected_window: 期望的时间窗口大小（默认使用 PRE_ENTRY_WINDOW）
        check_nan: 是否检查 NaN 值
        check_zero: 是否检查全零行
    
    Returns:
        (is_valid, error_message) 元组
    """
    if expected_window is None:
        expected_window = TRAJECTORY_CONFIG["PRE_ENTRY_WINDOW"]
    
    # 基本检查
    if trajectory is None:
        return False, "轨迹为 None"
    
    if not isinstance(trajectory, np.ndarray):
        return False, f"轨迹类型错误: {type(trajectory)}"
    
    if trajectory.size == 0:
        return False, "轨迹为空"
    
    if trajectory.ndim != 2:
        return False, f"轨迹维度错误: {trajectory.ndim}，期望 2"
    
    # 形状检查
    if trajectory.shape[1] != 32:
        return False, f"特征维度错误: {trajectory.shape[1]}，期望 32"
    
    if trajectory.shape[0] != expected_window:
        return False, f"时间窗口错误: {trajectory.shape[0]}，期望 {expected_window}"
    
    # NaN 检查
    if check_nan:
        nan_count = np.isnan(trajectory).sum()
        if nan_count > 0:
            return False, f"包含 {nan_count} 个 NaN 值"
    
    # 全零行检查
    if check_zero:
        zero_rows = (trajectory == 0).all(axis=1).sum()
        if zero_rows > trajectory.shape[0] * 0.5:  # 超过50%全零行
            return False, f"过多全零行: {zero_rows}/{trajectory.shape[0]}"
    
    return True, "OK"


def trajectory_to_serializable(trajectory: np.ndarray) -> Optional[List[List[float]]]:
    """
    将轨迹矩阵转换为 JSON 可序列化格式
    
    Args:
        trajectory: numpy 数组
    
    Returns:
        嵌套列表，或 None（如果输入无效）
    """
    if trajectory is None or not isinstance(trajectory, np.ndarray):
        return None
    
    if trajectory.size == 0:
        return None
    
    # 处理 NaN 和 Inf
    clean = np.nan_to_num(trajectory, nan=0.0, posinf=0.0, neginf=0.0)
    
    return clean.tolist()


def trajectory_from_serializable(data: List[List[float]]) -> Optional[np.ndarray]:
    """
    从 JSON 序列化格式还原轨迹矩阵
    
    Args:
        data: 嵌套列表
    
    Returns:
        numpy 数组，或 None（如果输入无效）
    """
    if data is None or not isinstance(data, list):
        return None
    
    try:
        trajectory = np.array(data, dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] != 32:
            return None
        return trajectory
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 原型格式迁移工具
# ══════════════════════════════════════════════════════════════════════════════

def migrate_prototype_centroid_to_weighted_mean(
    centroid: np.ndarray
) -> np.ndarray:
    """
    将旧版原型的 pre_entry_centroid 迁移为新版 weighted_mean
    
    用于原型库版本升级：旧格式只存储简单均值，新格式使用加权均值
    
    Args:
        centroid: 旧版 (32,) 中心向量
    
    Returns:
        加权后的 (32,) 向量
    """
    from core.feature_vector import apply_weights
    
    if centroid is None or centroid.size != 32:
        return centroid if centroid is not None else np.array([])
    
    return apply_weights(centroid)


# ══════════════════════════════════════════════════════════════════════════════
# PaperOrder → TradeRecord 转换工具（增量训练支持）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PaperOrderTradeRecord:
    """
    PaperOrder 到 TradeRecord 的适配器类
    
    将 PaperOrder 格式转换为与 TrajectoryMemory.extract_from_trades() 兼容的格式，
    用于将模拟交易结果纳入增量训练流程。
    
    与 backtester.TradeRecord 的主要区别：
    - 直接携带 entry_trajectory（PaperOrder 已保存）
    - 使用 datetime 而非 K线索引（更适合实时交易）
    - 包含原型匹配信息（template_fingerprint, entry_similarity）
    """
    # 基本交易信息
    entry_idx: int                  # 入场 K 线索引（相对或绝对）
    exit_idx: int                   # 离场 K 线索引
    side: int                       # 1=多, -1=空
    profit_pct: float               # 收益率百分比
    
    # 价格信息
    entry_price: float = 0.0
    exit_price: float = 0.0
    
    # 时间信息
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # 持仓信息
    hold_periods: int = 0           # 持仓K线数
    
    # 平仓原因
    exit_reason: str = ""           # "tp", "sl", "trailing", "derail", "manual" 等
    
    # 原型匹配信息
    template_fingerprint: Optional[str] = None
    entry_similarity: float = 0.0
    
    # 【指纹3D图】轨迹数据（PaperOrder 已保存，无需重新计算）
    entry_trajectory: Optional[np.ndarray] = None  # (60, 32) 入场前轨迹
    
    # 市场状态
    market_regime: str = ""
    
    # 订单来源信息
    source_order_id: str = ""       # 原始 PaperOrder ID
    
    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        trajectory_data = None
        if self.entry_trajectory is not None and isinstance(self.entry_trajectory, np.ndarray):
            trajectory_data = self.entry_trajectory.tolist()
        
        return {
            "entry_idx": self.entry_idx,
            "exit_idx": self.exit_idx,
            "side": self.side,
            "profit_pct": self.profit_pct,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "hold_periods": self.hold_periods,
            "exit_reason": self.exit_reason,
            "template_fingerprint": self.template_fingerprint,
            "entry_similarity": self.entry_similarity,
            "entry_trajectory": trajectory_data,
            "market_regime": self.market_regime,
            "source_order_id": self.source_order_id,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PaperOrderTradeRecord':
        """从字典还原"""
        entry_time = None
        if d.get("entry_time"):
            try:
                entry_time = datetime.fromisoformat(d["entry_time"])
            except (ValueError, TypeError):
                pass
        
        exit_time = None
        if d.get("exit_time"):
            try:
                exit_time = datetime.fromisoformat(d["exit_time"])
            except (ValueError, TypeError):
                pass
        
        entry_trajectory = None
        if d.get("entry_trajectory"):
            try:
                entry_trajectory = np.array(d["entry_trajectory"], dtype=np.float32)
            except (ValueError, TypeError):
                pass
        
        return cls(
            entry_idx=d.get("entry_idx", 0),
            exit_idx=d.get("exit_idx", 0),
            side=d.get("side", 1),
            profit_pct=d.get("profit_pct", 0.0),
            entry_price=d.get("entry_price", 0.0),
            exit_price=d.get("exit_price", 0.0),
            entry_time=entry_time,
            exit_time=exit_time,
            hold_periods=d.get("hold_periods", 0),
            exit_reason=d.get("exit_reason", ""),
            template_fingerprint=d.get("template_fingerprint"),
            entry_similarity=d.get("entry_similarity", 0.0),
            entry_trajectory=entry_trajectory,
            market_regime=d.get("market_regime", ""),
            source_order_id=d.get("source_order_id", ""),
        )


def paper_order_to_trade_record(
    order: 'PaperOrder',
    base_bar_idx: int = 0,
    market_regime: str = ""
) -> Optional[PaperOrderTradeRecord]:
    """
    将 PaperOrder 转换为增量训练兼容的 TradeRecord 格式
    
    Args:
        order: PaperOrder 对象（必须已平仓）
        base_bar_idx: 基准 K 线索引（用于转换相对索引）
        market_regime: 市场状态标签
    
    Returns:
        PaperOrderTradeRecord 对象，或 None（如果订单未平仓或数据无效）
    
    Example:
        >>> from core.paper_trader import PaperOrder, OrderStatus, OrderSide
        >>> from core.trajectory_engine import paper_order_to_trade_record
        >>> 
        >>> # 假设有一个已平仓的 PaperOrder
        >>> order = PaperOrder(...)  # 已平仓
        >>> trade_record = paper_order_to_trade_record(order)
        >>> if trade_record:
        ...     print(f"转换成功: {trade_record.profit_pct:.2f}%")
    """
    # 导入 PaperOrder 相关枚举（避免循环导入）
    from core.paper_trader import OrderStatus, OrderSide, CloseReason
    
    # 检查订单是否已平仓
    if order.status != OrderStatus.CLOSED:
        return None
    
    # 转换方向: OrderSide.LONG -> 1, OrderSide.SHORT -> -1
    side = 1 if order.side == OrderSide.LONG else -1
    
    # 转换平仓原因
    exit_reason_map = {
        CloseReason.TAKE_PROFIT: "tp",
        CloseReason.STOP_LOSS: "sl",
        CloseReason.TRAILING_STOP: "trailing",
        CloseReason.DERAIL: "derail",
        CloseReason.MAX_HOLD: "timeout",
        CloseReason.MANUAL: "manual",
        CloseReason.SIGNAL: "signal",
        CloseReason.EXCHANGE_CLOSE: "exchange",
        CloseReason.POSITION_FLIP: "flip",
    }
    exit_reason = exit_reason_map.get(order.close_reason, "unknown")
    
    # 计算 K 线索引
    entry_idx = base_bar_idx + order.entry_bar_idx
    exit_idx = entry_idx + order.hold_bars if order.exit_bar_idx is None else base_bar_idx + order.exit_bar_idx
    
    return PaperOrderTradeRecord(
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        side=side,
        profit_pct=order.profit_pct,
        entry_price=order.entry_price,
        exit_price=order.exit_price or 0.0,
        entry_time=order.entry_time,
        exit_time=order.exit_time,
        hold_periods=order.hold_bars,
        exit_reason=exit_reason,
        template_fingerprint=order.template_fingerprint,
        entry_similarity=order.entry_similarity,
        entry_trajectory=order.entry_trajectory,
        market_regime=market_regime,
        source_order_id=order.order_id,
    )


def paper_orders_to_trade_records(
    orders: List['PaperOrder'],
    base_bar_idx: int = 0,
    market_regime: str = "",
    filter_closed: bool = True,
    verbose: bool = True
) -> List[PaperOrderTradeRecord]:
    """
    批量将 PaperOrder 列表转换为 TradeRecord 格式
    
    Args:
        orders: PaperOrder 对象列表
        base_bar_idx: 基准 K 线索引
        market_regime: 默认市场状态标签（订单未指定时使用）
        filter_closed: 是否只转换已平仓订单
        verbose: 是否打印转换统计
    
    Returns:
        PaperOrderTradeRecord 列表
    
    Example:
        >>> from core.paper_trader import load_trade_history_from_file
        >>> from core.trajectory_engine import paper_orders_to_trade_records
        >>> 
        >>> orders = load_trade_history_from_file("data/paper_trading/history.json")
        >>> trade_records = paper_orders_to_trade_records(orders)
        >>> print(f"转换了 {len(trade_records)} 笔交易")
    """
    from core.paper_trader import OrderStatus
    
    records = []
    skipped_open = 0
    skipped_invalid = 0
    
    for order in orders:
        # 过滤未平仓订单
        if filter_closed and order.status != OrderStatus.CLOSED:
            skipped_open += 1
            continue
        
        record = paper_order_to_trade_record(order, base_bar_idx, market_regime)
        if record is not None:
            records.append(record)
        else:
            skipped_invalid += 1
    
    if verbose:
        print(f"[PaperOrder→TradeRecord] 转换完成: {len(records)} 笔成功")
        if skipped_open > 0:
            print(f"    跳过未平仓: {skipped_open}")
        if skipped_invalid > 0:
            print(f"    跳过无效: {skipped_invalid}")
    
    return records


def create_templates_from_paper_orders(
    orders: List['PaperOrder'],
    min_profit_pct: float = None,
    pre_entry_window: int = None,
    verbose: bool = True
) -> List[TrajectoryTemplate]:
    """
    从 PaperOrder 列表直接创建 TrajectoryTemplate
    
    与 TrajectoryMemory.extract_from_trades() 不同，此函数直接使用
    PaperOrder 中已保存的 entry_trajectory，无需 FeatureVectorEngine。
    
    适用场景：
    - 实时模拟交易的增量训练
    - 轨迹数据已保存的历史订单
    
    Args:
        orders: PaperOrder 对象列表（必须包含 entry_trajectory）
        min_profit_pct: 最小收益率阈值（低于此收益的订单不生成模板）
        pre_entry_window: 预期的轨迹窗口大小（用于验证）
        verbose: 是否打印统计
    
    Returns:
        TrajectoryTemplate 列表
    
    Example:
        >>> from core.paper_trader import load_trade_history_from_file
        >>> from core.trajectory_engine import create_templates_from_paper_orders
        >>> 
        >>> orders = load_trade_history_from_file("data/paper_trading/history.json")
        >>> templates = create_templates_from_paper_orders(orders, min_profit_pct=0.5)
        >>> print(f"创建了 {len(templates)} 个模板")
    """
    from core.paper_trader import OrderStatus, OrderSide
    from core.market_regime import MarketRegime
    
    if min_profit_pct is None:
        min_profit_pct = TRAJECTORY_CONFIG.get("MIN_PROFIT_PCT", 0.0)
    
    if pre_entry_window is None:
        pre_entry_window = TRAJECTORY_CONFIG.get("PRE_ENTRY_WINDOW", 60)
    
    templates = []
    skipped_open = 0
    skipped_loss = 0
    skipped_no_traj = 0
    skipped_invalid_traj = 0
    
    for i, order in enumerate(orders):
        # 跳过未平仓订单
        if order.status != OrderStatus.CLOSED:
            skipped_open += 1
            continue
        
        # 跳过亏损交易
        if order.profit_pct < min_profit_pct:
            skipped_loss += 1
            continue
        
        # 检查轨迹数据
        if order.entry_trajectory is None:
            skipped_no_traj += 1
            continue
        
        # 验证轨迹形状
        traj = order.entry_trajectory
        if not isinstance(traj, np.ndarray) or traj.ndim != 2:
            skipped_invalid_traj += 1
            continue
        
        if traj.shape[1] != 32:
            skipped_invalid_traj += 1
            continue
        
        # 确定方向
        direction = "LONG" if order.side == OrderSide.LONG else "SHORT"
        
        # 推断市场状态（如果未指定，根据方向推断默认值）
        regime = MarketRegime.RANGE_BULL if direction == "LONG" else MarketRegime.RANGE_BEAR
        
        # 创建模板
        # 注意：由于 PaperOrder 只保存入场前轨迹，holding 和 pre_exit 使用空数组
        template = TrajectoryTemplate(
            trade_idx=i,
            regime=regime,
            direction=direction,
            profit_pct=order.profit_pct,
            pre_entry=traj,
            holding=np.array([]),  # PaperOrder 未保存持仓轨迹
            pre_exit=np.array([]),  # PaperOrder 未保存离场轨迹
            entry_idx=order.entry_bar_idx,
            exit_idx=order.exit_bar_idx or (order.entry_bar_idx + order.hold_bars),
        )
        
        templates.append(template)
    
    if verbose:
        print(f"[PaperOrder→Template] 创建模板: {len(templates)} 个")
        if skipped_open > 0:
            print(f"    跳过未平仓: {skipped_open}")
        if skipped_loss > 0:
            print(f"    跳过亏损: {skipped_loss}")
        if skipped_no_traj > 0:
            print(f"    跳过无轨迹: {skipped_no_traj}")
        if skipped_invalid_traj > 0:
            print(f"    跳过无效轨迹: {skipped_invalid_traj}")
    
    return templates


class IncrementalTrainer:
    """
    增量训练器 - 从模拟交易中学习并更新原型库
    
    工作流程：
    1. 收集已平仓的 PaperOrder
    2. 转换为 TrajectoryTemplate
    3. 合并到现有原型库
    4. （可选）触发重新聚类
    
    使用方式：
        trainer = IncrementalTrainer(prototype_library)
        
        # 添加模拟交易
        trainer.add_paper_orders(orders)
        
        # 执行增量训练
        result = trainer.train()
        
        # 保存更新后的原型库
        trainer.save_library("data/prototypes/updated.pkl")
    
    示例：
        >>> from core.template_clusterer import PrototypeLibrary
        >>> from core.trajectory_engine import IncrementalTrainer
        >>> from core.paper_trader import load_trade_history_from_file
        >>> 
        >>> # 加载现有原型库
        >>> library = PrototypeLibrary.load("data/prototypes/btc_1m.pkl")
        >>> 
        >>> # 创建增量训练器
        >>> trainer = IncrementalTrainer(library)
        >>> 
        >>> # 添加模拟交易记录
        >>> orders = load_trade_history_from_file("data/paper_trading/history.json")
        >>> trainer.add_paper_orders(orders)
        >>> 
        >>> # 训练并获取结果
        >>> result = trainer.train(min_profit_pct=0.5)
        >>> print(f"新增模板: {result['templates_added']}")
    """
    
    def __init__(self, library: 'PrototypeLibrary' = None):
        """
        初始化增量训练器
        
        Args:
            library: 现有的原型库，None 则创建新库
        """
        self.library = library
        self.pending_orders: List['PaperOrder'] = []
        self.pending_templates: List[TrajectoryTemplate] = []
        self._training_history: List[Dict] = []
    
    def add_paper_orders(self, orders: List['PaperOrder']):
        """
        添加待训练的 PaperOrder 列表
        
        Args:
            orders: PaperOrder 对象列表
        """
        self.pending_orders.extend(orders)
        print(f"[IncrementalTrainer] 添加 {len(orders)} 笔订单 (累计 {len(self.pending_orders)} 笔)")
    
    def add_paper_order(self, order: 'PaperOrder'):
        """
        添加单个 PaperOrder
        
        Args:
            order: PaperOrder 对象
        """
        self.pending_orders.append(order)
    
    def add_templates(self, templates: List[TrajectoryTemplate]):
        """
        直接添加 TrajectoryTemplate 列表
        
        Args:
            templates: TrajectoryTemplate 列表
        """
        self.pending_templates.extend(templates)
        print(f"[IncrementalTrainer] 添加 {len(templates)} 个模板 (累计 {len(self.pending_templates)} 个)")
    
    def train(
        self,
        min_profit_pct: float = None,
        recluster: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        执行增量训练
        
        Args:
            min_profit_pct: 最小收益率阈值（低于此收益的订单不训练）
            recluster: 是否触发重新聚类（否则只添加新模板）
            verbose: 是否打印训练过程
        
        Returns:
            训练结果字典:
            {
                "orders_processed": int,       # 处理的订单数
                "templates_added": int,        # 新增的模板数
                "templates_before": int,       # 训练前模板总数
                "templates_after": int,        # 训练后模板总数
                "recluster_performed": bool,   # 是否执行了重新聚类
                "timestamp": str,              # 训练时间
            }
        """
        result = {
            "orders_processed": 0,
            "templates_added": 0,
            "templates_before": 0,
            "templates_after": 0,
            "recluster_performed": False,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 1. 从 PaperOrder 创建模板
        if self.pending_orders:
            if verbose:
                print(f"[IncrementalTrainer] 从 {len(self.pending_orders)} 笔订单创建模板...")
            
            new_templates = create_templates_from_paper_orders(
                self.pending_orders,
                min_profit_pct=min_profit_pct,
                verbose=verbose
            )
            self.pending_templates.extend(new_templates)
            result["orders_processed"] = len(self.pending_orders)
            self.pending_orders.clear()
        
        # 2. 检查是否有待添加的模板
        if not self.pending_templates:
            if verbose:
                print("[IncrementalTrainer] 没有待添加的模板")
            return result
        
        # 3. 如果没有原型库，创建临时记忆库
        if self.library is None:
            if verbose:
                print("[IncrementalTrainer] 创建新的轨迹记忆库...")
            memory = TrajectoryMemory()
            for template in self.pending_templates:
                memory._add_template(template.regime, template.direction, template)
            
            result["templates_added"] = len(self.pending_templates)
            result["templates_after"] = memory.total_count
            
            self._training_history.append(result)
            self.pending_templates.clear()
            
            if verbose:
                print(f"[IncrementalTrainer] 训练完成: 新增 {result['templates_added']} 个模板")
            
            # 返回记忆库供调用方使用
            result["memory"] = memory
            return result
        
        # 4. 有原型库时，合并模板到记忆库或直接更新原型
        # TODO: 实现与 PrototypeLibrary 的集成
        # 目前返回待添加的模板列表
        result["templates_added"] = len(self.pending_templates)
        result["pending_templates"] = self.pending_templates.copy()
        
        self._training_history.append(result)
        self.pending_templates.clear()
        
        if verbose:
            print(f"[IncrementalTrainer] 训练完成: 准备 {result['templates_added']} 个模板待合并")
        
        return result
    
    def get_training_history(self) -> List[Dict]:
        """获取训练历史"""
        return self._training_history.copy()
    
    def save_pending_to_file(self, filepath: str):
        """
        将待训练的订单保存到文件（断点续训）
        
        Args:
            filepath: 保存路径
        """
        data = {
            "pending_orders": [
                order.to_dict() if hasattr(order, 'to_dict') else order
                for order in self.pending_orders
            ],
            "pending_templates": [
                t.to_dict() for t in self.pending_templates
            ],
            "saved_at": datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[IncrementalTrainer] 已保存待训练数据: {filepath}")
    
    @classmethod
    def load_pending_from_file(cls, filepath: str, library: 'PrototypeLibrary' = None) -> 'IncrementalTrainer':
        """
        从文件加载待训练的数据（断点续训）
        
        Args:
            filepath: 文件路径
            library: 原型库
        
        Returns:
            IncrementalTrainer 实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls(library)
        
        # 还原模板
        for t_dict in data.get("pending_templates", []):
            template = TrajectoryTemplate.from_dict(t_dict)
            trainer.pending_templates.append(template)
        
        # 注意：PaperOrder 需要从外部加载
        print(f"[IncrementalTrainer] 已加载待训练数据: {len(trainer.pending_templates)} 个模板")
        
        return trainer


def get_incremental_training_candidates(
    orders: List['PaperOrder'],
    min_profit_pct: float = 0.0,
    require_trajectory: bool = True,
    verbose: bool = True
) -> Tuple[List['PaperOrder'], Dict]:
    """
    筛选适合增量训练的订单
    
    Args:
        orders: PaperOrder 列表
        min_profit_pct: 最小收益率阈值
        require_trajectory: 是否要求有轨迹数据
        verbose: 是否打印统计
    
    Returns:
        (筛选后的订单列表, 统计信息字典)
    
    Example:
        >>> candidates, stats = get_incremental_training_candidates(orders, min_profit_pct=0.5)
        >>> print(f"候选订单: {len(candidates)}, 盈利率: {stats['profit_rate']:.1%}")
    """
    from core.paper_trader import OrderStatus
    
    candidates = []
    stats = {
        "total": len(orders),
        "closed": 0,
        "profitable": 0,
        "has_trajectory": 0,
        "qualified": 0,
        "profit_rate": 0.0,
        "avg_profit_pct": 0.0,
    }
    
    profits = []
    
    for order in orders:
        # 只考虑已平仓订单
        if order.status != OrderStatus.CLOSED:
            continue
        stats["closed"] += 1
        
        # 收益统计
        if order.profit_pct > 0:
            stats["profitable"] += 1
        
        # 轨迹数据检查
        has_traj = (
            order.entry_trajectory is not None and
            isinstance(order.entry_trajectory, np.ndarray) and
            order.entry_trajectory.size > 0
        )
        if has_traj:
            stats["has_trajectory"] += 1
        
        # 筛选条件
        if order.profit_pct < min_profit_pct:
            continue
        
        if require_trajectory and not has_traj:
            continue
        
        candidates.append(order)
        profits.append(order.profit_pct)
    
    stats["qualified"] = len(candidates)
    stats["profit_rate"] = stats["profitable"] / stats["closed"] if stats["closed"] > 0 else 0.0
    stats["avg_profit_pct"] = np.mean(profits) if profits else 0.0
    
    if verbose:
        print(f"[增量训练筛选] 候选: {stats['qualified']}/{stats['total']}")
        print(f"    已平仓: {stats['closed']}, 盈利: {stats['profitable']} ({stats['profit_rate']:.1%})")
        print(f"    有轨迹: {stats['has_trajectory']}, 平均收益: {stats['avg_profit_pct']:+.2f}%")
    
    return candidates, stats
