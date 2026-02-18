"""
冷启动管理器 - 管理交易系统冷启动阶段的门槛放宽与频率监控

冷启动模式用于：
- 系统刚部署时，通过降低门槛快速收集交易数据
- 长时间无交易时，自动检测并适当放宽门槛

工作流：
1. 用户启用冷启动 → enabled=True → 应用宽松门槛
2. 每根K线 → check_frequency() → 检查是否长时间无交易
3. 如频率过低 → auto_relax_thresholds() → 进一步放宽5%
4. 用户关闭冷启动 → enabled=False → 恢复正常门槛
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class ColdStartState:
    """冷启动状态数据"""
    enabled: bool = False
    # 当前生效的门槛（冷启动时为宽松值，否则为正常值）
    fusion_threshold: float = 0.65
    cosine_threshold: float = 0.70
    euclidean_threshold: float = 0.35
    dtw_threshold: float = 0.30
    # 自动放宽次数（用于追踪放宽历史）
    auto_relax_count: int = 0
    # 上次交易时间戳
    last_trade_time: Optional[float] = None
    # 今日交易笔数
    trades_today: int = 0
    trades_today_date: Optional[str] = None  # 用于检测日期切换
    # 自动毕业相关
    graduated: bool = False
    graduation_time: Optional[str] = None
    trade_results: list = field(default_factory=list)  # [(profit_pct, timestamp), ...]
    # 持久化时间（用于 UI 记忆时间范围）
    created_at: Optional[float] = None   # 首次创建时间戳
    last_save_time: Optional[float] = None  # 最后保存时间戳


class ColdStartManager:
    """
    冷启动管理器
    
    管理交易系统的冷启动模式，包括：
    - 门槛放宽控制
    - 交易频率监控
    - 自动放宽机制
    - 状态持久化
    
    用法：
        manager = ColdStartManager(
            on_threshold_changed=my_callback,
            on_auto_relax=my_relax_callback,
        )
        
        # 启用冷启动
        manager.set_enabled(True)
        
        # 获取当前门槛
        thresholds = manager.get_thresholds()
        
        # 每根K线检查频率
        if manager.check_frequency():
            print("交易频率过低，已自动放宽门槛")
    """
    
    def __init__(
        self,
        state_file: str = "data/cold_start_state.json",
        on_threshold_changed: Optional[Callable[[Dict[str, float]], None]] = None,
        on_auto_relax: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            state_file: 状态持久化文件路径
            on_threshold_changed: 门槛变化回调
            on_auto_relax: 自动放宽时的通知回调
        """
        self._state_file = state_file
        self._on_threshold_changed = on_threshold_changed
        self._on_auto_relax = on_auto_relax
        
        # 从配置加载默认值
        from config import COLD_START_CONFIG, SIMILARITY_CONFIG
        self._cold_config = COLD_START_CONFIG
        self._sim_config = SIMILARITY_CONFIG
        
        # 正常门槛（从 SIMILARITY_CONFIG 读取）
        self._normal_thresholds = {
            "fusion": self._sim_config.get("FUSION_THRESHOLD", 0.65),
            "cosine": self._sim_config.get("COSINE_MIN_THRESHOLD", 0.70),
            "euclidean": 0.35,  # 默认值
            "dtw": 0.30,        # 默认值
        }
        
        # 冷启动宽松门槛（从 COLD_START_CONFIG 读取）
        self._cold_thresholds = self._cold_config.get("THRESHOLDS", {
            "fusion": 0.30,
            "cosine": 0.50,
            "euclidean": 0.25,
            "dtw": 0.20,
        })
        
        # 频率监控配置
        freq_cfg = self._cold_config.get("FREQUENCY_MONITOR", {})
        self._target_trades_per_hour = freq_cfg.get("TARGET_TRADES_PER_HOUR", 1.5)
        self._low_freq_threshold_hours = freq_cfg.get("LOW_FREQUENCY_THRESHOLD_HOURS", 1.0)
        self._auto_relax_percent = freq_cfg.get("AUTO_RELAX_PERCENT", 0.05)
        
        # 内部状态
        self._state = ColdStartState()
        self._state.enabled = self._cold_config.get("ENABLED_BY_DEFAULT", False)
        
        # 初始化门槛
        self._apply_thresholds()
        
        # 加载持久化状态
        self._load_state()
        
        # 脏标志（用于延迟保存）
        self._dirty = False
        self._last_save_time = 0.0
    
    @property
    def enabled(self) -> bool:
        """冷启动模式是否启用"""
        return self._state.enabled
    
    @property
    def current_thresholds(self) -> Dict[str, float]:
        """当前生效的门槛值"""
        return {
            "fusion": self._state.fusion_threshold,
            "cosine": self._state.cosine_threshold,
            "euclidean": self._state.euclidean_threshold,
            "dtw": self._state.dtw_threshold,
        }
    
    @property
    def last_trade_time(self) -> Optional[float]:
        """上次交易时间戳"""
        return self._state.last_trade_time
    
    @property
    def trades_today(self) -> int:
        """今日交易笔数"""
        return self._state.trades_today
    
    @property
    def auto_relax_count(self) -> int:
        """自动放宽次数"""
        return self._state.auto_relax_count
    
    def set_enabled(self, enabled: bool) -> None:
        """
        设置冷启动模式开关
        
        Args:
            enabled: 是否启用冷启动模式
        """
        if self._state.enabled == enabled:
            return
        
        self._state.enabled = enabled
        self._apply_thresholds()
        self._dirty = True
        
        print(f"[ColdStart] 冷启动模式{'启用' if enabled else '关闭'}")
        
        if self._on_threshold_changed:
            self._on_threshold_changed(self.current_thresholds)
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        获取当前生效的门槛值
        
        Returns:
            包含 fusion, cosine, euclidean, dtw 的门槛字典
        """
        return self.current_thresholds
    
    def record_trade(self, timestamp: Optional[float] = None, profit_pct: Optional[float] = None) -> bool:
        """
        记录一笔交易（用于频率统计和成功率跟踪）
        
        Args:
            timestamp: 交易时间戳，默认为当前时间
            profit_pct: 交易盈亏百分比（用于自动毕业判断）
        
        Returns:
            是否触发了自动毕业
        """
        if timestamp is None:
            timestamp = time.time()
        
        self._state.last_trade_time = timestamp
        
        # 检查日期切换
        today = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        if self._state.trades_today_date != today:
            self._state.trades_today_date = today
            self._state.trades_today = 0
        
        self._state.trades_today += 1
        
        # 记录交易结果（用于自动毕业）
        if profit_pct is not None and self._state.enabled and not self._state.graduated:
            self._state.trade_results.append((profit_pct, timestamp))
            
            # 只保留最近100笔交易结果
            if len(self._state.trade_results) > 100:
                self._state.trade_results = self._state.trade_results[-100:]
            
            # 检查是否达到毕业条件
            graduated = self._check_graduation()
            if graduated:
                self._dirty = True
                return True
        
        self._dirty = True
        return False
    
    def check_frequency(self, current_time: Optional[float] = None) -> bool:
        """
        检查交易频率是否过低
        
        如果超过阈值时间无交易，且冷启动已启用，则自动放宽门槛。
        
        Args:
            current_time: 当前时间戳，默认为当前时间
            
        Returns:
            是否触发了自动放宽
        """
        if not self._state.enabled:
            return False
        
        if current_time is None:
            current_time = time.time()
        
        # 首次运行或无交易记录
        if self._state.last_trade_time is None:
            return False
        
        # 计算距上次交易的时间（小时）
        hours_since_last = (current_time - self._state.last_trade_time) / 3600.0
        
        # 检查是否超过频率阈值
        if hours_since_last >= self._low_freq_threshold_hours:
            # 自动放宽门槛
            self.auto_relax_thresholds()
            # 重置上次交易时间，避免连续触发
            self._state.last_trade_time = current_time
            return True
        
        return False
    
    def auto_relax_thresholds(self) -> None:
        """
        自动放宽门槛（在当前基础上降低配置的百分比）
        """
        if not self._state.enabled:
            return
        
        relax_pct = self._auto_relax_percent
        
        # 在当前门槛基础上放宽
        old_fusion = self._state.fusion_threshold
        old_cosine = self._state.cosine_threshold
        old_euclidean = self._state.euclidean_threshold
        old_dtw = self._state.dtw_threshold
        
        self._state.fusion_threshold = max(0.10, old_fusion * (1 - relax_pct))
        self._state.cosine_threshold = max(0.30, old_cosine * (1 - relax_pct))
        self._state.euclidean_threshold = max(0.10, old_euclidean * (1 - relax_pct))
        self._state.dtw_threshold = max(0.10, old_dtw * (1 - relax_pct))
        
        self._state.auto_relax_count += 1
        self._dirty = True
        
        print(f"[ColdStart] 自动放宽门槛 (第{self._state.auto_relax_count}次): "
              f"融合 {old_fusion:.2f}→{self._state.fusion_threshold:.2f}, "
              f"余弦 {old_cosine:.2f}→{self._state.cosine_threshold:.2f}")
        
        if self._on_auto_relax:
            self._on_auto_relax(
                f"交易频率过低，门槛已自动放宽5% (第{self._state.auto_relax_count}次)"
            )
        
        if self._on_threshold_changed:
            self._on_threshold_changed(self.current_thresholds)
    
    def get_frequency_status(self) -> Dict[str, Any]:
        """
        获取频率监控状态
        
        Returns:
            包含频率监控状态信息的字典
        """
        now = time.time()
        
        # 距上次交易时间
        if self._state.last_trade_time:
            seconds_since = now - self._state.last_trade_time
            minutes_since = seconds_since / 60.0
            hours_since = seconds_since / 3600.0
        else:
            minutes_since = None
            hours_since = None
        
        # 计算今日交易频率
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        hours_today = (now - today_start) / 3600.0
        trades_per_hour = self._state.trades_today / max(hours_today, 0.1)
        
        # 状态判断
        if not self._state.enabled:
            status = "关闭"
        elif hours_since is not None and hours_since >= self._low_freq_threshold_hours:
            status = "频率过低"
        elif trades_per_hour < self._target_trades_per_hour * 0.5:
            status = "偏低"
        else:
            status = "正常"
        
        return {
            "enabled": self._state.enabled,
            "minutes_since_last_trade": minutes_since,
            "hours_since_last_trade": hours_since,
            "trades_today": self._state.trades_today,
            "trades_per_hour": trades_per_hour,
            "target_per_hour": self._target_trades_per_hour,
            "status": status,
            "auto_relax_count": self._state.auto_relax_count,
        }
    
    def get_graduation_status(self) -> Dict[str, Any]:
        """
        获取毕业状态信息
        
        Returns:
            包含毕业相关的状态信息
        """
        if not self._state.enabled and not self._state.graduated:
            return {
                "graduated": False,
                "enabled": False,
                "status": "未启用",
            }
        
        if self._state.graduated:
            return {
                "graduated": True,
                "graduation_time": self._state.graduation_time,
                "status": "已毕业",
            }
        
        # 冷启动中，计算当前进度
        if len(self._state.trade_results) > 0:
            profitable = sum(1 for p, _ in self._state.trade_results if p > 0)
            total = len(self._state.trade_results)
            success_rate = profitable / total
            
            min_trades = self._cold_config.get("COLD_START_MIN_TRADES_FOR_GRADUATE", 20)
            target_rate = self._cold_config.get("COLD_START_SUCCESS_RATE_THRESHOLD", 0.80)
            
            return {
                "graduated": False,
                "enabled": True,
                "current_success_rate": success_rate,
                "target_success_rate": target_rate,
                "current_trades": total,
                "min_trades": min_trades,
                "status": f"进行中 ({total}/{min_trades}笔, {success_rate:.1%})",
            }
        else:
            return {
                "graduated": False,
                "enabled": True,
                "current_trades": 0,
                "min_trades": self._cold_config.get("COLD_START_MIN_TRADES_FOR_GRADUATE", 20),
                "status": "等待首笔交易",
            }
    
    def get_state_for_ui(self) -> Dict[str, Any]:
        """
        获取用于UI显示的状态数据
        
        Returns:
            包含UI展示所需的完整状态信息
        """
        freq_status = self.get_frequency_status()
        grad_status = self.get_graduation_status()
        
        return {
            "enabled": self._state.enabled,
            "thresholds": self.current_thresholds,
            "normal_thresholds": self._normal_thresholds,
            "cold_thresholds": self._cold_thresholds,
            "frequency": freq_status,
            "graduation": grad_status,
        }
    
    def _check_graduation(self) -> bool:
        """
        检查是否达到毕业条件
        
        Returns:
            是否触发了自动毕业
        """
        if not self._state.enabled or self._state.graduated:
            return False
        
        # 从配置读取毕业条件
        min_trades = self._cold_config.get("COLD_START_MIN_TRADES_FOR_GRADUATE", 20)
        success_rate_threshold = self._cold_config.get("COLD_START_SUCCESS_RATE_THRESHOLD", 0.80)
        
        # 检查交易笔数
        if len(self._state.trade_results) < min_trades:
            return False
        
        # 计算成功率（盈利笔数 / 总笔数）
        profitable_trades = sum(1 for profit, _ in self._state.trade_results if profit > 0)
        total_trades = len(self._state.trade_results)
        success_rate = profitable_trades / total_trades
        
        # 检查是否达到毕业条件
        if success_rate >= success_rate_threshold:
            # 自动毕业！
            self._state.graduated = True
            self._state.graduation_time = datetime.now().isoformat()
            self._state.enabled = False  # 关闭冷启动模式
            
            # 切换回正常阈值
            self._apply_thresholds()
            
            print(f"[ColdStart] 🎓 自动毕业！成功率: {success_rate:.1%} ({profitable_trades}/{total_trades}笔) "
                  f"| 毕业时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self._on_threshold_changed:
                self._on_threshold_changed(self.current_thresholds)
            
            return True
        
        return False
    
    def _apply_thresholds(self) -> None:
        """根据当前模式应用对应的门槛值"""
        if self._state.enabled and not self._state.graduated:
            # 冷启动模式：使用宽松门槛
            self._state.fusion_threshold = self._cold_thresholds.get("fusion", 0.30)
            self._state.cosine_threshold = self._cold_thresholds.get("cosine", 0.50)
            self._state.euclidean_threshold = self._cold_thresholds.get("euclidean", 0.25)
            self._state.dtw_threshold = self._cold_thresholds.get("dtw", 0.10)
        else:
            # 正常模式：使用标准门槛
            self._state.fusion_threshold = self._normal_thresholds.get("fusion", 0.65)
            self._state.cosine_threshold = self._normal_thresholds.get("cosine", 0.70)
            self._state.euclidean_threshold = self._normal_thresholds.get("euclidean", 0.35)
            self._state.dtw_threshold = self._normal_thresholds.get("dtw", 0.30)
        
        # 重置自动放宽计数
        if not self._state.graduated:
            self._state.auto_relax_count = 0
    
    def _load_state(self) -> None:
        """从文件加载状态"""
        if not os.path.exists(self._state_file):
            return
        
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 启动时始终遵从配置的 ENABLED_BY_DEFAULT，不恢复上次的 enabled 状态。
            # 这样程序重启后冷启动总是"关闭"（除非用户在 UI 中手动开启）。
            self._state.enabled = self._cold_config.get("ENABLED_BY_DEFAULT", False)
            self._state.last_trade_time = data.get("last_trade_time")
            self._state.trades_today = data.get("trades_today", 0)
            self._state.trades_today_date = data.get("trades_today_date")
            self._state.auto_relax_count = data.get("auto_relax_count", 0)
            self._state.graduated = data.get("graduated", False)
            self._state.graduation_time = data.get("graduation_time")
            self._state.trade_results = data.get("trade_results", [])
            self._state.created_at = data.get("created_at")
            self._state.last_save_time = data.get("last_save_time")
            
            # 如果启用状态已保存，恢复门槛值
            if self._state.enabled:
                # 冷启动阈值取“更宽松”的一侧，避免旧状态比新配置更严
                self._state.fusion_threshold = min(
                    data.get("fusion_threshold", self._cold_thresholds["fusion"]),
                    self._cold_thresholds["fusion"],
                )
                self._state.cosine_threshold = min(
                    data.get("cosine_threshold", self._cold_thresholds["cosine"]),
                    self._cold_thresholds["cosine"],
                )
                self._state.euclidean_threshold = min(
                    data.get("euclidean_threshold", self._cold_thresholds["euclidean"]),
                    self._cold_thresholds["euclidean"],
                )
                self._state.dtw_threshold = min(
                    data.get("dtw_threshold", self._cold_thresholds["dtw"]),
                    self._cold_thresholds["dtw"],
                )
            
            print(f"[ColdStart] 已加载状态: enabled={self._state.enabled}, "
                  f"trades_today={self._state.trades_today}")
            
        except Exception as e:
            print(f"[ColdStart] 加载状态失败: {e}")
    
    def save_state(self) -> None:
        """保存状态到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            now = time.time()
            if self._state.created_at is None:
                self._state.created_at = now
            self._state.last_save_time = now
            
            data = {
                "enabled": self._state.enabled,
                "fusion_threshold": self._state.fusion_threshold,
                "cosine_threshold": self._state.cosine_threshold,
                "euclidean_threshold": self._state.euclidean_threshold,
                "dtw_threshold": self._state.dtw_threshold,
                "auto_relax_count": self._state.auto_relax_count,
                "last_trade_time": self._state.last_trade_time,
                "trades_today": self._state.trades_today,
                "trades_today_date": self._state.trades_today_date,
                "graduated": self._state.graduated,
                "graduation_time": self._state.graduation_time,
                "trade_results": self._state.trade_results[-50:],  # 只保存最近50笔
                "created_at": self._state.created_at,
                "last_save_time": self._state.last_save_time,
                "saved_at": datetime.now().isoformat(),
            }
            
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._dirty = False
            self._last_save_time = now
            
        except Exception as e:
            print(f"[ColdStart] 保存状态失败: {e}")
    
    @staticmethod
    def persist_enabled_state(enabled: bool, state_file: str = "data/cold_start_state.json") -> None:
        """
        在引擎未运行时持久化冷启动开关到文件，确保下次启动时能正确加载。

        Args:
            enabled: 冷启动是否启用
            state_file: 状态文件路径
        """
        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            data: Dict[str, Any] = {}
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["enabled"] = enabled
            if enabled:
                cfg = __import__("config", fromlist=["COLD_START_CONFIG"]).COLD_START_CONFIG
                th = cfg.get("THRESHOLDS", {})
                data["fusion_threshold"] = th.get("fusion", 0.30)
                data["cosine_threshold"] = th.get("cosine", 0.50)
                data["euclidean_threshold"] = th.get("euclidean", 0.25)
                data["dtw_threshold"] = th.get("dtw", 0.20)
            data["saved_at"] = datetime.now().isoformat()
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[ColdStart] 已持久化冷启动开关: enabled={enabled}（引擎未运行，下次启动时生效）")
        except Exception as e:
            print(f"[ColdStart] 持久化冷启动状态失败: {e}")

    def save_if_dirty(self, min_interval_sec: float = 60.0) -> None:
        """
        如果有变更且距上次保存超过指定间隔，则保存
        
        Args:
            min_interval_sec: 最小保存间隔（秒）
        """
        if not self._dirty:
            return
        
        now = time.time()
        if now - self._last_save_time < min_interval_sec:
            return
        
        self.save_state()
