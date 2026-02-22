"""
信号分析标签页
功能：多条件组合回测分析，发现高胜率信号组合
后台线程分析 + 1% 粒度进度条 + 本轮/累计双结果表 + 多轮历史区
新增：累计表市场状态列/估算P&L列 + 实盘监控面板 + 风控开关面板
"""
from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_RISK_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'risk_control_state.json'
)
_SIGNAL_SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'signal_analysis_settings.json'
)

# ─────────────────────────────────────────────────────────────────────────────
# 颜色常量
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK      = "#1A1E24"
BG_PANEL     = "#1F252D"
BG_CARD      = "#252C35"
BORDER_COLOR = "#2D3640"
TEXT_PRIMARY = "#E7EDF4"
TEXT_DIM     = "#7A8694"
ACCENT_CYAN  = "#00C8D4"
ACCENT_GOLD  = "#D9B36A"
TIER_ELITE   = "#FF6B35"   # 精品 - 橙红
TIER_GOOD    = "#00C8D4"   # 优质 - 青
TIER_CAND    = "#4DA3FF"   # 候选 - 蓝
TIER_HIGH_FREQ = "#00CED1" # 高频 - 青色
WARN_COLOR   = "#F5A623"   # 警告 - 橙
LONG_COLOR   = "#26A69A"   # 做多 - 绿
SHORT_COLOR  = "#EF5350"   # 做空 - 红
GOOD_COLOR   = "#4CAF50"   # 正常 - 绿
DECAY_MILD   = "#F5A623"   # 轻微衰减 - 橙
DECAY_SEVERE = "#EF5350"   # 严重衰减 - 红
TIER_HIGH_FREQ = "#00CED1" # 高频层 - 青色


# ─────────────────────────────────────────────────────────────────────────────
# 后台工作者
# ─────────────────────────────────────────────────────────────────────────────

class SignalAnalysisWorker(QtCore.QObject):
    """
    后台线程：调用 signal_analyzer.analyze() 分析两个方向，
    合并结果，合并到 signal_store，发出进度和完成信号。
    """
    progress = QtCore.pyqtSignal(int, str)       # (percent 0-100, status_text)
    finished = QtCore.pyqtSignal(list)            # List[dict] 本轮结果
    error    = QtCore.pyqtSignal(str)

    def __init__(self, df, excluded_families=None, validation_split=0.0, max_hold: int = 60,
                 min_atr_ratio: float = 0.0004, max_atr_sl_mult: float = 0.8, parent=None):
        super().__init__(parent)
        self._df = df
        self._excluded_families = excluded_families or []
        self._validation_split = validation_split
        self._max_hold = max_hold
        self._min_atr_ratio = min_atr_ratio
        self._max_atr_sl_mult = max_atr_sl_mult
        self._stop = False

    def stop(self):
        self._stop = True

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from core.signal_analyzer import analyze
            from core import signal_store

            all_results: List[dict] = []

            def cb_long(pct: int, text: str):
                if not self._stop:
                    self.progress.emit(max(1, min(pct // 2, 49)), f"[做多] {text}")

            def cb_short(pct: int, text: str):
                if not self._stop:
                    self.progress.emit(50 + max(0, min(pct // 2, 49)), f"[做空] {text}")

            _atr_kw = dict(min_atr_ratio=self._min_atr_ratio, max_atr_sl_mult=self._max_atr_sl_mult)

            if not self._stop:
                long_p1 = analyze(
                    self._df, 'long', pool_id='pool1', progress_cb=cb_long,
                    excluded_families=self._excluded_families,
                    validation_split=self._validation_split,
                    max_hold=self._max_hold, **_atr_kw,
                )
                all_results.extend(long_p1)

            if not self._stop:
                short_p1 = analyze(
                    self._df, 'short', pool_id='pool1', progress_cb=cb_short,
                    excluded_families=self._excluded_families,
                    validation_split=self._validation_split,
                    max_hold=self._max_hold, **_atr_kw,
                )
                all_results.extend(short_p1)

            if not self._stop:
                long_p2 = analyze(
                    self._df, 'long', pool_id='pool2', progress_cb=cb_long,
                    excluded_families=self._excluded_families,
                    validation_split=self._validation_split,
                    max_hold=self._max_hold, **_atr_kw,
                )
                all_results.extend(long_p2)

            if not self._stop:
                short_p2 = analyze(
                    self._df, 'short', pool_id='pool2', progress_cb=cb_short,
                    excluded_families=self._excluded_families,
                    validation_split=self._validation_split,
                    max_hold=self._max_hold, **_atr_kw,
                )
                all_results.extend(short_p2)

            if not self._stop:
                self.progress.emit(99, "写入持久化状态...")
                signal_store.merge_rounds(long_p1, short_p1, bar_count=len(self._df), pool_id='pool1')
                signal_store.merge_rounds(long_p2, short_p2, bar_count=len(self._df), pool_id='pool2')
                # 两次 merge 都完成后，执行一次去重并更新缓存（O(n²)，只跑一次）
                signal_store.rebuild_pruned_cache(pool_id='pool1')
                signal_store.rebuild_pruned_cache(pool_id='pool2')

            if not self._stop:
                self.progress.emit(100, f"分析完成，共 {len(all_results)} 个有效组合")
                self.finished.emit(all_results)

        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class _InitialLoadWorker(QtCore.QObject):
    """
    后台线程：首次激活 Tab 时一次性加载所有初始数据，预热 signal_store 内存缓存。
    完成后主线程直接调用各 _refresh_* 方法（命中缓存，几乎无 IO）。
    """
    finished = QtCore.pyqtSignal(dict)
    error    = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from core import signal_store
            # 预热两池缓存
            signal_store.get_cumulative_results(top_n=200, pool_id='pool1')
            signal_store.get_cumulative_results(top_n=200, pool_id='pool2')
            rounds = signal_store.get_rounds()
            self.finished.emit({
                'rounds':     rounds,
            })
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _tier_color(tier: str) -> str:
    return {
        "精品": TIER_ELITE,
        "优质": TIER_GOOD,
        "候选": TIER_CAND,
    }.get(tier, TEXT_DIM)


def _tier_from_rate(rate: float, direction: str, pool_id: str = 'pool1') -> str:
    """根据综合命中率、方向和策略池返回层级（与 signal_analyzer 门槛一致）"""
    if pool_id == 'pool2':
        # pool2 双向对称门槛
        if rate >= 0.59: return '精品'
        if rate >= 0.55: return '优质'
        if rate >= 0.52: return '候选'
    elif direction == 'long':
        if rate >= 0.71: return '精品'
        if rate >= 0.67: return '优质'
        if rate >= 0.64: return '候选'
    else:
        if rate >= 0.59: return '精品'
        if rate >= 0.55: return '优质'
        if rate >= 0.52: return '候选'
    return ''


def _rate_color(rate: float, direction: str = 'long') -> str:
    if direction == 'short':
        if rate >= 0.59: return TIER_ELITE
        if rate >= 0.55: return TIER_GOOD
        if rate >= 0.52: return TIER_CAND
        return TEXT_DIM
    else:
        if rate >= 0.71: return TIER_ELITE
        if rate >= 0.67: return TIER_GOOD
        if rate >= 0.64: return TIER_CAND
        return TEXT_DIM


def _pnl_color(pnl: float) -> str:
    if pnl > 0:
        return LONG_COLOR
    if pnl < 0:
        return SHORT_COLOR
    return TEXT_DIM


def _ev_per_trigger_pct(overall_rate: float, direction: str, pool_id: str = 'pool1') -> float:
    """单次触发期望盈亏（百分比，未考虑杠杆）。"""
    # 含费后净值（与 signal_store.py 保持一致）
    if pool_id == 'pool2':
        tp_pct, sl_pct = 0.0094, 0.0086  # Pool2 双向对称
    else:
        if direction == "short":
            tp_pct, sl_pct = 0.0074, 0.0066
        else:
            tp_pct, sl_pct = 0.0054, 0.0086
    per_trade = overall_rate * tp_pct - (1.0 - overall_rate) * sl_pct
    return round(per_trade * 100, 4)


def _make_table(headers: List[str]) -> QtWidgets.QTableWidget:
    tbl = QtWidgets.QTableWidget(0, len(headers))
    tbl.setHorizontalHeaderLabels(headers)
    tbl.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
    tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    tbl.setAlternatingRowColors(True)
    tbl.verticalHeader().setVisible(False)
    tbl.setShowGrid(False)
    tbl.setStyleSheet(f"""
        QTableWidget {{
            background-color: {BG_CARD};
            alternate-background-color: {BG_PANEL};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_COLOR};
            gridline-color: {BORDER_COLOR};
            font-size: 12px;
        }}
        QHeaderView::section {{
            background-color: {BG_PANEL};
            color: {TEXT_DIM};
            padding: 4px 8px;
            border: none;
            border-bottom: 1px solid {BORDER_COLOR};
            font-size: 11px;
            font-weight: bold;
        }}
        QTableWidget::item {{
            padding: 4px 8px;
        }}
        QTableWidget::item:selected {{
            background-color: #2A3A4A;
            color: {TEXT_PRIMARY};
        }}
    """)
    hdr = tbl.horizontalHeader()
    hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    hdr.setStretchLastSection(True)   # 最后一列自动填满剩余宽度，避免横向滚动
    tbl.setSortingEnabled(True)
    return tbl


def _set_cumul_col_widths(tbl: QtWidgets.QTableWidget) -> None:
    """为累计结果表设置紧凑固定宽度的列，避免窄列浪费空间。"""
    hdr = tbl.horizontalHeader()
    RM = QtWidgets.QHeaderView.ResizeMode
    # 极短列固定宽度
    _fixed: dict[int, int] = {
        0:  30,   # #
        1:  44,   # 方向
        2:  44,   # 层级
        3:  60,   # 出现轮次
        4:  60,   # 累计触发
        5:  60,   # 累计命中
        6:  72,   # 综合命中率
        7:  72,   # 平均命中率
        8:  48,   # 波动
        9:  60,   # 综合评分
        10: 110,  # 随机基准（含超越幅度）
        11: 90,   # 各状态命中率
        12: 72,   # 估算总盈亏
        13: 58,   # 单次EV
        14: 56,   # 平均持仓
        # 15 条件组合 → Stretch（由 setStretchLastSection 控制，不设固定宽度）
    }
    for col, width in _fixed.items():
        hdr.setSectionResizeMode(col, RM.Fixed)
        tbl.setColumnWidth(col, width)


class _SortableItem(QtWidgets.QTableWidgetItem):
    def __init__(self, text: str, sort_value):
        super().__init__(text)
        self.setData(QtCore.Qt.ItemDataRole.UserRole, sort_value)

    def __lt__(self, other):
        try:
            v1 = self.data(QtCore.Qt.ItemDataRole.UserRole)
            v2 = other.data(QtCore.Qt.ItemDataRole.UserRole)
            if v1 is not None and v2 is not None:
                return v1 < v2
        except Exception:
            pass
        return super().__lt__(other)


def _set_item(tbl: QtWidgets.QTableWidget, row: int, col: int,
              text: str, color: Optional[str] = None, bold: bool = False,
              sort_value: Optional[float] = None, tooltip: Optional[str] = None):
    if sort_value is None:
        item = QtWidgets.QTableWidgetItem(text)
    else:
        item = _SortableItem(text, sort_value)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    if color:
        item.setForeground(QtGui.QColor(color))
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    if tooltip:
        item.setToolTip(tooltip)
    tbl.setItem(row, col, item)


def _dominant_market_state(breakdown: Optional[dict]) -> str:
    """从 market_state_breakdown 中找出触发次数最多的状态（保留用于兼容）。"""
    if not breakdown:
        return "-"
    best_state = "-"
    best_count = 0
    total = sum(v.get("total_triggers", 0) for v in breakdown.values())
    for state, info in breakdown.items():
        cnt = info.get("total_triggers", 0)
        if cnt > best_count:
            best_count = cnt
            best_state = state
    if total > 0 and best_count > 0:
        pct = best_count / total
        return f"{best_state}({pct:.0%})"
    return best_state


def _get_state_rate(breakdown: Optional[dict], state: str) -> Tuple[float, int]:
    """返回指定市场状态的命中率和触发次数，无数据则返回 (0.0, 0)。"""
    if not breakdown:
        return 0.0, 0
    info = breakdown.get(state) or {}
    return info.get("avg_rate", 0.0), info.get("total_triggers", 0)


def _format_state_detail(breakdown: Optional[dict], direction: str) -> str:
    """
    格式化市场状态明细，对命中率未达候选门槛的状态加 ⚠ 警告。
    门槛：做多 0.64，做空 0.52（候选门槛）。
    触发次数 < 5 的状态不展示（样本不足）。
    示例输出："多头80%  震荡67%  ⚠空头42%"
    """
    if not breakdown:
        return "-"
    threshold = 0.64 if direction == 'long' else 0.52
    state_map = {"多头趋势": "多头", "空头趋势": "空头", "震荡市": "震荡"}
    parts = []
    for state, label in state_map.items():
        r, t = _get_state_rate(breakdown, state)
        if t < 5:
            continue
        prefix = "⚠" if r < threshold else ""
        parts.append(f"{prefix}{label}{r:.0%}")
    return "  ".join(parts) if parts else "-"


def _format_timestamp(ts: str) -> str:
    if not ts:
        return ""
    if isinstance(ts, str) and ts.isdigit():
        try:
            sec = int(ts) / (1000.0 if len(ts) >= 13 else 1.0)
            return datetime.fromtimestamp(sec).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ts
    try:
        fixed = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(fixed)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


from core.signal_utils import _COND_LABELS, _cond_label, _format_conditions


def _family_label(name: str, direction: str) -> str:
    base = name.replace("_strict", "").replace("_loose", "")
    info = _COND_LABELS.get(base)
    if not info:
        return name
    label = info.get("label", base)
    if base == "consec_bear":
        return "连续阴线"
    if base == "consec_bull":
        return "连续阳线"
    if base == "lower_shd":
        return "下影线"
    if base == "upper_shd":
        return "上影线"
    return label


def _family_key(conditions: List[str], direction: str) -> Tuple[str, Tuple[str, ...]]:
    families = sorted({ _family_label(c, direction) for c in conditions })
    return (direction, tuple(families))


# ─────────────────────────────────────────────────────────────────────────────
# 风控状态持久化
# ─────────────────────────────────────────────────────────────────────────────

def _load_risk_state() -> dict:
    try:
        if os.path.exists(_RISK_STATE_FILE):
            with open(_RISK_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {"daily_loss_limit": True, "streak_loss_pause": True}


def _save_risk_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_RISK_STATE_FILE), exist_ok=True)
        with open(_RISK_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_signal_settings() -> dict:
    try:
        if os.path.exists(_SIGNAL_SETTINGS_FILE):
            with open(_SIGNAL_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_signal_settings(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_SIGNAL_SETTINGS_FILE), exist_ok=True)
        with open(_SIGNAL_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 主 Tab 组件
# ─────────────────────────────────────────────────────────────────────────────

class SignalAnalysisTab(QtWidgets.QWidget):
    """
    信号分析页签
    布局（从上到下）：
      ① 操作按钮区
      ② 风控开关栏
      ③ 进度条 + 状态文字
      ④ 左右分栏：本轮结果表 | 累计结果表（含市场状态列+估算P&L列）
      ⑤ 多轮历史文本区
      ⑥ 实盘监控折叠区
    """

    # 发出此信号，通知 MainWindow 加载新一批历史数据（不同时间段的 50000 根）
    request_new_data = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None
        self._worker: Optional[SignalAnalysisWorker] = None
        self._thread: Optional[QtCore.QThread] = None
        self._init_worker: Optional[_InitialLoadWorker] = None
        self._init_thread: Optional[QtCore.QThread] = None
        self._running = False
        self._main_window = None
        self._risk_state = _load_risk_state()
        self._auto_run_on_next_data = False   # 新数据到达后自动开始分析

        self._setup_ui()
        self._apply_style()

    # ── 构建 UI ───────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ① 操作按钮区
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_start = QtWidgets.QPushButton("▶  开始分析")
        self._btn_start.setFixedHeight(34)
        self._btn_start.clicked.connect(self._on_start)

        self._btn_new_data = QtWidgets.QPushButton("🔄  换新数据再验证")
        self._btn_new_data.setFixedHeight(34)
        self._btn_new_data.setEnabled(False)
        self._btn_new_data.setToolTip("加载新一批历史数据后再次运行分析，结果将合并到累计记录")
        self._btn_new_data.clicked.connect(self._on_new_data)

        self._btn_auto_50 = QtWidgets.QPushButton("🔁  50次自动换新验证")
        self._btn_auto_50.setFixedHeight(34)
        self._btn_auto_50.setEnabled(False)
        self._btn_auto_50.setToolTip("自动执行50轮：换新数据 -> 分析 -> 合并结果")
        self._btn_auto_50.clicked.connect(self._on_auto_50)
        self._auto_count = 0

        self._btn_clear = QtWidgets.QPushButton("🗑  清空记录")
        self._btn_clear.setFixedHeight(34)
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_stop = QtWidgets.QPushButton("■  停止")
        self._btn_stop.setFixedHeight(34)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        info_lbl = QtWidgets.QLabel(
            "分析策略：9指标×2阈值=18条件/方向 | 2-5条件组合 | 含手续费门槛 64/67/71%"
        )
        info_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_new_data)
        btn_row.addWidget(self._btn_auto_50)
        btn_row.addWidget(self._btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(info_lbl)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_clear)
        root.addLayout(btn_row)

        # ①b 排除条件族（可折叠，下次分析生效）
        _sig_settings = _load_signal_settings()
        exclude_row = QtWidgets.QHBoxLayout()
        exclude_row.setSpacing(20)
        exclude_row.setContentsMargins(10, 6, 10, 6)

        self._chk_exclude_ma5 = QtWidgets.QCheckBox("排除偏离MA5类条件（下次分析生效）")
        self._chk_exclude_ma5.setChecked(bool(_sig_settings.get("exclude_ma5", False)))
        self._chk_exclude_ma5.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        self._chk_exclude_ma5.stateChanged.connect(self._on_exclude_changed)

        self._chk_exclude_ma5_slope = QtWidgets.QCheckBox("排除均线斜率类条件")
        self._chk_exclude_ma5_slope.setChecked(bool(_sig_settings.get("exclude_ma5_slope", False)))
        self._chk_exclude_ma5_slope.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        self._chk_exclude_ma5_slope.stateChanged.connect(self._on_exclude_changed)

        self._chk_validation_split = QtWidgets.QCheckBox("启用70/30训练验证分割（推荐）")
        self._chk_validation_split.setChecked(
            bool(_sig_settings.get("validation_split_enabled", True))
        )
        self._chk_validation_split.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 12px;")
        self._chk_validation_split.stateChanged.connect(self._on_exclude_changed)

        exclude_row.addWidget(self._chk_exclude_ma5)
        exclude_row.addWidget(self._chk_exclude_ma5_slope)
        exclude_row.addWidget(self._chk_validation_split)

        # 最大持仓 SpinBox
        lbl_max_hold = QtWidgets.QLabel("最大持仓K线:")
        self._spn_max_hold = QtWidgets.QSpinBox()
        self._spn_max_hold.setRange(20, 240)
        self._spn_max_hold.setValue(int(_sig_settings.get("max_hold", 60)))
        self._spn_max_hold.setFixedWidth(60)
        self._spn_max_hold.setFixedHeight(24)
        self._spn_max_hold.valueChanged.connect(self._on_exclude_changed)
        exclude_row.addWidget(lbl_max_hold)
        exclude_row.addWidget(self._spn_max_hold)

        exclude_row.addStretch()

        exclude_frame = QtWidgets.QFrame()
        exclude_frame.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 3px; padding: 6px 12px;"
        )
        exclude_frame.setLayout(exclude_row)
        root.addWidget(exclude_frame)

        # ①c 数据模式（专项市场状态分析）
        data_mode_row = QtWidgets.QHBoxLayout()
        data_mode_row.setSpacing(10)
        data_mode_row.setContentsMargins(10, 6, 10, 6)

        lbl_mode = QtWidgets.QLabel("数据模式:")
        lbl_mode.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        data_mode_row.addWidget(lbl_mode)

        self._cmb_regime_filter = QtWidgets.QComboBox()
        self._cmb_regime_filter.addItems(["全量", "仅空头趋势", "仅多头趋势", "仅震荡市"])
        self._cmb_regime_filter.setFixedWidth(120)
        self._cmb_regime_filter.setFixedHeight(24)
        self._cmb_regime_filter.setStyleSheet(f"""
            QComboBox {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QComboBox:hover {{
                border-color: {ACCENT_CYAN};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                selection-background-color: {ACCENT_CYAN};
                selection-color: {BG_DARK};
                border: 1px solid {BORDER_COLOR};
            }}
        """)
        _mode_val = _sig_settings.get("regime_filter", "全量")
        if _mode_val in ["全量", "仅空头趋势", "仅多头趋势", "仅震荡市"]:
            self._cmb_regime_filter.setCurrentText(_mode_val)
        data_mode_row.addWidget(self._cmb_regime_filter)

        mode_tip = QtWidgets.QLabel("（下次分析生效）")
        mode_tip.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        data_mode_row.addWidget(mode_tip)
        data_mode_row.addStretch()

        data_mode_frame = QtWidgets.QFrame()
        data_mode_frame.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 3px; padding: 6px 12px;"
        )
        data_mode_frame.setLayout(data_mode_row)
        root.addWidget(data_mode_frame)

        # ② 风控开关栏（紧凑横排）
        risk_row = QtWidgets.QHBoxLayout()
        risk_row.setSpacing(28)
        risk_row.setContentsMargins(10, 8, 10, 8)

        self._chk_daily_loss = QtWidgets.QCheckBox("  日亏损限制 5%")
        self._chk_daily_loss.setChecked(self._risk_state.get("daily_loss_limit", True))
        self._chk_daily_loss.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; spacing: 10px;")
        self._chk_daily_loss.stateChanged.connect(self._on_risk_changed)

        self._chk_streak = QtWidgets.QCheckBox("  连续止损 10次暂停")
        self._chk_streak.setChecked(self._risk_state.get("streak_loss_pause", True))
        self._chk_streak.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; spacing: 10px;")
        self._chk_streak.stateChanged.connect(self._on_risk_changed)

        self._lbl_streak = QtWidgets.QLabel("当前连亏: 0次")
        self._lbl_streak.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")

        self._lbl_daily_pnl = QtWidgets.QLabel("今日盈亏: +0.00%")
        self._lbl_daily_pnl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px;")

        risk_sep = QtWidgets.QFrame()
        risk_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        risk_sep.setStyleSheet(f"color: {BORDER_COLOR};")

        risk_lbl = QtWidgets.QLabel("风控:")
        risk_lbl.setStyleSheet(f"color: {ACCENT_GOLD}; font-size: 15px; font-weight: bold;")

        risk_row.addWidget(risk_lbl)
        risk_row.addWidget(self._chk_daily_loss)
        risk_row.addWidget(self._chk_streak)
        risk_row.addSpacing(20)
        risk_row.addWidget(self._lbl_streak)
        risk_row.addWidget(self._lbl_daily_pnl)
        risk_row.addStretch()

        risk_frame = QtWidgets.QFrame()
        risk_frame.setStyleSheet(
            f"background-color: {BG_PANEL}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 3px; padding: 8px 12px;"
        )
        risk_frame.setLayout(risk_row)
        risk_frame.setMinimumHeight(48)
        root.addWidget(risk_frame)

        # ③ 进度条 + 状态文字
        prog_row = QtWidgets.QHBoxLayout()
        prog_row.setSpacing(8)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(18)
        self._progress.setTextVisible(True)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_CYAN};
                border-radius: 2px;
            }}
        """)

        self._status_lbl = QtWidgets.QLabel("就绪  —  请先在「上帝视角训练」页签加载历史K线数据")
        self._status_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        prog_row.addWidget(self._progress, stretch=1)
        prog_row.addWidget(self._status_lbl, stretch=2)
        root.addLayout(prog_row)

        # ④ 分栏：本轮结果表 + 累计结果表
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER_COLOR}; }}")

        # 本轮结果
        left_box = QtWidgets.QGroupBox("本轮结果")
        left_box.setStyleSheet(self._group_box_style())
        left_layout = QtWidgets.QVBoxLayout(left_box)
        left_layout.setContentsMargins(6, 6, 6, 6)

        self._round_table = _make_table([
            "#", "方向", "净命中率", "触发次数", "命中次数", "层级", "条件组合"
        ])
        left_layout.addWidget(self._round_table)
        splitter.addWidget(left_box)

        # 累计结果（含"各状态命中率"和"估算总盈亏"列）
        right_box = QtWidgets.QGroupBox("累计结果（多轮合并）")
        right_box.setStyleSheet(self._group_box_style())
        right_layout = QtWidgets.QVBoxLayout(right_box)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # 筛选控件 + 总数统计 + 导出按钮
        cumul_count_row = QtWidgets.QHBoxLayout()
        cumul_count_row.setSpacing(8)
        
        self._btn_export_cumul = QtWidgets.QPushButton("📄 导出TXT")
        self._btn_export_cumul.setFixedHeight(26)
        self._btn_export_cumul.setFixedWidth(100)
        self._btn_export_cumul.setToolTip("导出累计结果为TXT文件")
        self._btn_export_cumul.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                font-size: 11px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_CYAN};
                color: {BG_DARK};
                border-color: {ACCENT_CYAN};
            }}
        """)
        self._btn_export_cumul.clicked.connect(self._export_cumulative_txt)
        cumul_count_row.addWidget(self._btn_export_cumul)

        self._btn_backup_github = QtWidgets.QPushButton("☁ 备份GitHub")
        self._btn_backup_github.setFixedHeight(26)
        self._btn_backup_github.setFixedWidth(110)
        self._btn_backup_github.setToolTip("备份信号池数据到可提交目录（含 pool1/pool2）")
        self._btn_backup_github.setStyleSheet(self._btn_export_cumul.styleSheet())
        self._btn_backup_github.clicked.connect(self._backup_to_github)
        cumul_count_row.addWidget(self._btn_backup_github)

        self._btn_import_pool = QtWidgets.QPushButton("📥 导入数据")
        self._btn_import_pool.setFixedHeight(26)
        self._btn_import_pool.setFixedWidth(100)
        self._btn_import_pool.setToolTip("从 TXT/JSON 文件导入 Pool1/Pool2 数据（TXT自动识别池子）")
        self._btn_import_pool.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_CARD};
                color: #7ecfad;
                border: 1px solid #3a8a68;
                border-radius: 3px;
                font-size: 11px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #3a8a68;
                color: {BG_DARK};
                border-color: #7ecfad;
            }}
        """)
        self._btn_import_pool.clicked.connect(self._import_pool_data)
        cumul_count_row.addWidget(self._btn_import_pool)
        
        # 方向筛选
        lbl_dir = QtWidgets.QLabel("方向:")
        lbl_dir.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        cumul_count_row.addWidget(lbl_dir)
        
        self._cmb_direction = QtWidgets.QComboBox()
        self._cmb_direction.addItems(["全部", "做多", "做空"])
        self._cmb_direction.setFixedWidth(80)
        self._cmb_direction.setFixedHeight(24)
        self._cmb_direction.setStyleSheet(f"""
            QComboBox {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QComboBox:hover {{
                border-color: {ACCENT_CYAN};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                selection-background-color: {ACCENT_CYAN};
                selection-color: {BG_DARK};
                border: 1px solid {BORDER_COLOR};
            }}
        """)
        self._cmb_direction.currentIndexChanged.connect(self._on_filter_changed)
        cumul_count_row.addWidget(self._cmb_direction)
        
        cumul_count_row.addSpacing(12)
        
        # 最少轮次筛选
        lbl_rounds = QtWidgets.QLabel("最少轮次:")
        lbl_rounds.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        cumul_count_row.addWidget(lbl_rounds)
        
        self._spn_min_rounds = QtWidgets.QSpinBox()
        self._spn_min_rounds.setRange(1, 50)
        _settings = _load_signal_settings()
        self._spn_min_rounds.setValue(int(_settings.get('min_rounds', 5)))
        self._spn_min_rounds.setFixedWidth(60)
        self._spn_min_rounds.setFixedHeight(24)
        self._spn_min_rounds.setStyleSheet(f"""
            QSpinBox {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }}
            QSpinBox:hover {{
                border-color: {ACCENT_CYAN};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {BG_PANEL};
                border: none;
                width: 16px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: {ACCENT_CYAN};
            }}
        """)
        self._spn_min_rounds.valueChanged.connect(self._on_filter_changed)
        cumul_count_row.addWidget(self._spn_min_rounds)

        self._btn_save_settings = QtWidgets.QPushButton("保存")
        self._btn_save_settings.setFixedWidth(50)
        self._btn_save_settings.setFixedHeight(24)
        self._btn_save_settings.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_PANEL};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                border-color: {ACCENT_CYAN};
                background-color: {BG_CARD};
            }}
        """)
        self._btn_save_settings.clicked.connect(self._on_save_settings)
        cumul_count_row.addWidget(self._btn_save_settings)
        
        cumul_count_row.addStretch(1)
        self._cumul_count_lbl = QtWidgets.QLabel("共 0 个 | 做多 0 | 做空 0 | 精品 0 | 优质 0")
        self._cumul_count_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        cumul_count_row.addWidget(self._cumul_count_lbl)
        right_layout.addLayout(cumul_count_row)

        cumul_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        cumul_splitter.setHandleWidth(4)
        cumul_splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER_COLOR}; }}")

        p1_box = QtWidgets.QWidget()
        p1_layout = QtWidgets.QVBoxLayout(p1_box)
        p1_layout.setContentsMargins(0, 0, 0, 0)
        p1_lbl = QtWidgets.QLabel("策略池 1（TP 0.6% / SL 0.8%，做多≥64% / 做空≥52%）")
        p1_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-weight: bold; font-size: 11px; padding: 2px;")
        p1_layout.addWidget(p1_lbl)
        self._cumul_table_p1 = _make_table([
            "#", "方向", "层级", "出现轮次", "累计触发", "累计命中",
            "综合命中率", "平均命中率", "波动", "综合评分",
            "随机基准", "各状态命中率", "估算总盈亏", "单次EV", "平均持仓", "条件组合"
        ])
        _set_cumul_col_widths(self._cumul_table_p1)
        p1_layout.addWidget(self._cumul_table_p1)
        cumul_splitter.addWidget(p1_box)

        p2_box = QtWidgets.QWidget()
        p2_layout = QtWidgets.QVBoxLayout(p2_box)
        p2_layout.setContentsMargins(0, 4, 0, 0)
        p2_lbl = QtWidgets.QLabel("策略池 2（TP 1.0% / SL 0.8%，做多≥52% / 做空≥52%）")
        p2_lbl.setStyleSheet(f"color: {ACCENT_GOLD}; font-weight: bold; font-size: 11px; padding: 2px;")
        p2_layout.addWidget(p2_lbl)
        self._cumul_table_p2 = _make_table([
            "#", "方向", "层级", "出现轮次", "累计触发", "累计命中",
            "综合命中率", "平均命中率", "波动", "综合评分",
            "随机基准", "各状态命中率", "估算总盈亏", "单次EV", "平均持仓", "条件组合"
        ])
        _set_cumul_col_widths(self._cumul_table_p2)
        p2_layout.addWidget(self._cumul_table_p2)
        cumul_splitter.addWidget(p2_box)

        right_layout.addWidget(cumul_splitter)
        splitter.addWidget(right_box)

        splitter.setSizes([450, 550])

        # ⑤ 多轮历史文本区
        hist_box = QtWidgets.QGroupBox("多轮历史记录")
        hist_box.setStyleSheet(self._group_box_style())
        hist_layout = QtWidgets.QVBoxLayout(hist_box)
        hist_layout.setContentsMargins(6, 6, 6, 6)

        self._history_text = QtWidgets.QTextEdit()
        self._history_text.setReadOnly(True)
        self._history_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_CARD};
                color: {TEXT_DIM};
                border: none;
                font-family: "Consolas", monospace;
                font-size: 11px;
            }}
        """)
        hist_layout.addWidget(self._history_text)

        # ⑥ 实盘监控折叠区
        live_box = QtWidgets.QGroupBox("实盘监控")
        live_box.setStyleSheet(self._group_box_style())
        live_box.setCheckable(True)
        live_box.setChecked(True)
        live_layout = QtWidgets.QVBoxLayout(live_box)
        live_layout.setContentsMargins(6, 6, 6, 6)

        self._live_table = _make_table([
            "组合", "回测命中率", "实盘命中率", "实盘次数", "连亏次数", "状态"
        ])
        live_layout.addWidget(self._live_table)

        # ⑤/⑥ 使用垂直分割器，允许手动调整高度
        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        bottom_splitter.setHandleWidth(4)
        bottom_splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER_COLOR}; }}")
        bottom_splitter.addWidget(hist_box)
        bottom_splitter.addWidget(live_box)
        bottom_splitter.setSizes([280, 320])

        # 总垂直分割器：上（本轮+累计表）↕ 下（历史+监控）
        main_vsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_vsplit.setHandleWidth(5)
        main_vsplit.setStyleSheet(
            f"QSplitter::handle {{ background: {ACCENT_CYAN}; border-radius: 2px; }}"
        )
        main_vsplit.addWidget(splitter)
        main_vsplit.addWidget(bottom_splitter)
        main_vsplit.setSizes([400, 400])

        root.addWidget(main_vsplit, stretch=5)

        self._initial_load_done = False

    def ensure_initial_load(self):
        """首次激活标签页时在后台线程加载数据，避免阻塞主线程导致 UI 卡死。"""
        if not self._initial_load_done:
            self._initial_load_done = True
            self._status_lbl.setText("加载中...")
            self._init_thread = QtCore.QThread(self)
            self._init_worker = _InitialLoadWorker()
            self._init_worker.moveToThread(self._init_thread)
            self._init_thread.started.connect(self._init_worker.run)
            self._init_worker.finished.connect(self._on_initial_load_done)
            self._init_worker.finished.connect(self._init_thread.quit)
            self._init_worker.error.connect(self._on_initial_load_error)
            self._init_worker.error.connect(self._init_thread.quit)
            self._init_thread.finished.connect(self._init_thread.deleteLater)
            self._init_thread.start()

    @QtCore.pyqtSlot(dict)
    def _on_initial_load_done(self, _data: dict):
        """初始数据加载完成回调（主线程）。signal_store 内存缓存已热，各 refresh 方法几乎无 IO。"""
        self._refresh_cumulative_table()
        self._refresh_history_text()
        self._refresh_backtest_feedback_table()
        self._refresh_risk_display()
        self._status_lbl.setText("就绪  —  请先在「上帝视角训练」页签加载历史K线数据")

    @QtCore.pyqtSlot(str)
    def _on_initial_load_error(self, msg: str):
        self._status_lbl.setText(f"初始加载失败：{msg[:80]}")

    def _group_box_style(self) -> str:
        return f"""
            QGroupBox {{
                background-color: {BG_PANEL};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                margin-top: 14px;
                color: {TEXT_DIM};
                font-size: 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {ACCENT_GOLD};
            }}
            QGroupBox::indicator {{
                width: 14px;
                height: 14px;
            }}
        """

    def _apply_style(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_DARK};
                color: {TEXT_PRIMARY};
            }}
            QPushButton {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 4px 14px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #2E3640;
                border-color: {ACCENT_CYAN};
                color: {ACCENT_CYAN};
            }}
            QPushButton:disabled {{
                color: {TEXT_DIM};
                border-color: {BORDER_COLOR};
            }}
            QCheckBox {{
                color: {TEXT_PRIMARY};
                font-size: 13px;
                spacing: 10px;
                padding: 2px 6px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 2px;
                background-color: {BG_CARD};
            }}
            QCheckBox::indicator:checked {{
                background-color: {ACCENT_CYAN};
                border-color: {ACCENT_CYAN};
            }}
        """)

    # ── 槽函数 ────────────────────────────────────────────────────────────

    def _on_start(self):
        if self._df is None:
            QtWidgets.QMessageBox.warning(
                self, "无数据",
                "请先在「上帝视角训练」页签加载历史 K 线数据，然后返回此页签开始分析。"
            )
            return
        self._run_analysis()

    def _on_new_data(self):
        """换新数据：通知 MainWindow 重新从本地数据库加载不同时间段的 50000 根 K 线。"""
        if self._running:
            return
        self._auto_run_on_next_data = True
        self._round_table.setRowCount(0)
        self._set_running(False)
        self._btn_new_data.setEnabled(False)
        self._btn_auto_50.setEnabled(False)
        self._status_lbl.setText("正在请求加载新一批历史数据，请稍候...")
        self.request_new_data.emit()

    def _on_auto_50(self):
        """自动执行 50 轮验证"""
        if self._is_backtest_running():
            QtWidgets.QMessageBox.warning(
                self, "提示", "正在回测，请先停止回测再开始自动验证。"
            )
            return
        if self._running:
            return
        self._auto_count = 50
        self._status_lbl.setText(f"已开启 50 轮自动验证 (剩余 {self._auto_count} 轮)...")
        self._on_new_data()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self._auto_count = 0
        self._set_running(False)
        self._status_lbl.setText("已停止")

    def _on_clear(self):
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空",
            "确定要清空所有信号分析记录吗？此操作不可恢复。",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                from core import signal_store
                signal_store.clear("pool1")
                signal_store.clear("pool2")
            except Exception:
                pass
            self._round_table.setRowCount(0)
            if hasattr(self, '_cumul_table_p1'):
                self._cumul_table_p1.setRowCount(0)
                self._cumul_table_p2.setRowCount(0)
            self._live_table.setRowCount(0)
            self._history_text.clear()
            # 强制从已清空的 signal_store 刷新累计表，确保 UI 与磁盘状态一致
            self._refresh_cumulative_table()
            self._status_lbl.setText("记录已清空（池子1: 0条，池子2: 0条）— 磁盘缓存已同步删除")

    def _on_risk_changed(self):
        self._risk_state["daily_loss_limit"]  = self._chk_daily_loss.isChecked()
        self._risk_state["streak_loss_pause"] = self._chk_streak.isChecked()
        _save_risk_state(self._risk_state)

    def _on_exclude_changed(self):
        """排除条件族勾选变化时保存到 signal_analysis_settings.json"""
        _settings = _load_signal_settings()
        _settings["exclude_ma5"] = self._chk_exclude_ma5.isChecked()
        _settings["exclude_ma5_slope"] = self._chk_exclude_ma5_slope.isChecked()
        _settings["validation_split_enabled"] = self._chk_validation_split.isChecked()
        _settings["max_hold"] = self._spn_max_hold.value()
        _save_signal_settings(_settings)

    def _on_filter_changed(self):
        """筛选条件变化时刷新累计结果表格"""
        self._refresh_cumulative_table()

    def _on_save_settings(self):
        """保存 min_rounds、排除条件族等到 signal_analysis_settings.json"""
        state = _load_signal_settings()
        state['min_rounds'] = self._spn_min_rounds.value()
        state['exclude_ma5'] = self._chk_exclude_ma5.isChecked()
        state['exclude_ma5_slope'] = self._chk_exclude_ma5_slope.isChecked()
        state['validation_split_enabled'] = self._chk_validation_split.isChecked()
        state['max_hold'] = self._spn_max_hold.value()
        if hasattr(self, "_cmb_regime_filter"):
            state['regime_filter'] = self._cmb_regime_filter.currentText()
        _save_signal_settings(state)

    @QtCore.pyqtSlot(int, str)
    def _on_progress(self, pct: int, text: str):
        self._progress.setValue(pct)
        self._status_lbl.setText(text)

    @QtCore.pyqtSlot(list)
    def _on_finished(self, results: list):
        self._set_running(False)
        self._populate_round_table(results)
        self._refresh_cumulative_table()
        self._refresh_history_text()
        self._refresh_backtest_feedback_table()
        n = len(results)
        
        if self._auto_count > 1:
            self._auto_count -= 1
            self._status_lbl.setText(
                f"第 {50 - self._auto_count} 轮完成，正在开启下一轮 (剩余 {self._auto_count} 轮)..."
            )
            # 延迟 1 秒后自动加载新数据
            QtCore.QTimer.singleShot(1000, self._on_new_data)
        else:
            self._auto_count = 0
            self._status_lbl.setText(
                f"分析完成 ✓  本轮发现 {n} 个有效组合 | 门槛：含手续费净命中率 ≥ 64%"
            )
            self._btn_new_data.setEnabled(True)
            self._btn_auto_50.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        self._set_running(False)
        self._status_lbl.setText(f"❌ 错误：{msg[:80]}")
        QtWidgets.QMessageBox.critical(self, "分析错误", msg[:600])

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _run_analysis(self):
        if self._running:
            return
        self._set_running(True)
        self._progress.setValue(0)
        self._status_lbl.setText("正在初始化...")

        df_to_use = self._df
        # 专项市场状态过滤（下次分析生效）
        if hasattr(self, "_cmb_regime_filter"):
            regime_filter = self._cmb_regime_filter.currentText()
            if regime_filter and regime_filter != "全量":
                try:
                    import numpy as np
                    adx_vals = df_to_use['adx'].values.astype(float)
                    slope_vals = df_to_use['ma5_slope'].values.astype(float)
                    state_arr = np.where(
                        adx_vals > 25,
                        np.where(slope_vals > 0, "仅多头趋势", "仅空头趋势"),
                        "仅震荡市",
                    )
                    df_to_use = df_to_use[state_arr == regime_filter].reset_index(drop=True)
                    if len(df_to_use) < 200:
                        self._set_running(False)
                        QtWidgets.QMessageBox.warning(
                            self,
                            "提示",
                            f"当前数据段中「{regime_filter}」K线不足200根，请换新数据。"
                        )
                        return
                except Exception:
                    # 如果缺少列或过滤失败，回退为全量
                    df_to_use = self._df

        # 构建排除条件族列表（排除偏离MA5类 → close_vs_ma5，排除均线斜率类 → ma5_slope）
        excluded_families: List[str] = []
        if self._chk_exclude_ma5.isChecked():
            excluded_families.append("close_vs_ma5")
        if self._chk_exclude_ma5_slope.isChecked():
            excluded_families.append("ma5_slope")

        validation_split = (
            0.3 if self._chk_validation_split.isChecked() else 0.0
        )

        self._thread = QtCore.QThread(self)
        self._worker = SignalAnalysisWorker(
            df_to_use,
            excluded_families=excluded_families,
            validation_split=validation_split,
            max_hold=self._spn_max_hold.value(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _set_running(self, running: bool):
        self._running = running
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_new_data.setEnabled(not running)
        self._btn_auto_50.setEnabled(not running)
        self._btn_clear.setEnabled(not running)

    def set_main_window(self, main_window):
        self._main_window = main_window

    def is_busy(self) -> bool:
        return self._running or self._auto_count > 0 or self._auto_run_on_next_data

    def _is_backtest_running(self) -> bool:
        return bool(self._main_window and getattr(self._main_window, "is_playing", False))

    def _populate_round_table(self, results: List[dict]):
        tbl = self._round_table
        tbl.setSortingEnabled(False)
        tbl.setRowCount(0)
        results_sorted = sorted(results, key=lambda r: r.get('hit_rate', r.get('hit_rate_net', 0)), reverse=True)
        for seq, r in enumerate(results_sorted, start=1):
            row = tbl.rowCount()
            tbl.insertRow(row)

            direction_str = "做多" if r["direction"] == "long" else "做空"
            dir_color  = LONG_COLOR if r["direction"] == "long" else SHORT_COLOR
            tier_color = _tier_color(r["tier"])
            hit_rate   = r.get('hit_rate', r.get('hit_rate_net', 0))
            rate_color = _rate_color(hit_rate, r["direction"])
            warning_prefix = "⚠ " if r.get("low_sample_warn", r.get("warning", False)) else ""

            _set_item(tbl, row, 0, str(seq), TEXT_DIM)
            _set_item(tbl, row, 1, direction_str, dir_color, bold=True)
            _set_item(tbl, row, 2,
                      f"{warning_prefix}{hit_rate:.1%}",
                      rate_color, bold=True, sort_value=hit_rate)
            _set_item(tbl, row, 3, str(r["trigger_count"]), sort_value=r["trigger_count"])
            _set_item(tbl, row, 4, str(r.get("hit_count", "")),
                      sort_value=r.get("hit_count", 0))
            _set_item(tbl, row, 5, r["tier"], tier_color, bold=True)
            _set_item(tbl, row, 6,
                      _format_conditions(r["conditions"], r["direction"]),
                      TEXT_DIM)
        tbl.setSortingEnabled(True)

    def _refresh_cumulative_table(self):
        try:
            from core import signal_store
            
            # 读取筛选条件
            dir_map = {"全部": None, "做多": "long", "做空": "short"}
            direction = dir_map.get(
                self._cmb_direction.currentText() if hasattr(self, '_cmb_direction') else "全部"
            )
            min_rounds = self._spn_min_rounds.value() if hasattr(self, '_spn_min_rounds') else 1
            
            def _filter_combos(combos):
                # 应用最少轮次筛选
                combos = [c for c in combos if c.get("appear_rounds", 0) >= min_rounds]

                # 方案B：条件族限额，防止"偏离MA5"类条件垄断策略池
                MAX_PER_FAMILY = 20
                family_count: dict = {}
                capped_combos = []
                for c in combos:
                    conditions = c.get("conditions", [])
                    families = set(
                        re.sub(r"_(loose|strict|mod_loose|mod_strict)$", "", cond)
                        for cond in conditions
                    )
                    if all(family_count.get(f, 0) < MAX_PER_FAMILY for f in families):
                        capped_combos.append(c)
                        for f in families:
                            family_count[f] = family_count.get(f, 0) + 1
                combos = capped_combos

                # 做空硬门槛：空头趋势下胜率必须 ≥ 52%（有足够样本时才检查）
                hard_filtered = []
                for c in combos:
                    if c.get("direction") == "short":
                        breakdown = c.get("market_state_breakdown", {}) or {}
                        bear_info = breakdown.get("空头趋势", {}) or {}
                        bear_rate = float(bear_info.get("avg_rate", 0.0))
                        bear_triggers = int(bear_info.get("total_triggers", 0))
                        if bear_triggers >= 5 and bear_rate < 0.52:
                            continue  # 剔除：有样本但胜率不达标
                    hard_filtered.append(c)
                return hard_filtered

            # 获取数据（已在后台去重+层级过滤，这里只做方向和轮次过滤）
            c1, cumul1 = signal_store.get_cumulative_results(top_n=500, direction=direction, pool_id='pool1')
            c2, cumul2 = signal_store.get_cumulative_results(top_n=500, direction=direction, pool_id='pool2')
            
            combos_p1 = _filter_combos(c1)
            combos_p2 = _filter_combos(c2)

        except Exception:
            return
        
        # 缓存当前筛选后的列表，供导出使用（保证导出与表格一致）
        self._latest_cumulative_combos_p1 = list(combos_p1)
        self._latest_cumulative_combos_p2 = list(combos_p2)

        # 更新总数统计（含层级数量）- 基于过滤后的数据
        total = len(combos_p1) + len(combos_p2)
        long_count = sum(1 for c in combos_p1 + combos_p2 if c.get("direction") == "long")
        short_count = total - long_count
        elite_count = good_count = 0
        for c in combos_p1:
            tier = _tier_from_rate(c.get("overall_rate", 0.0), c.get("direction", "long"), pool_id='pool1')
            if tier == "精品": elite_count += 1
            elif tier == "优质": good_count += 1
        for c in combos_p2:
            tier = _tier_from_rate(c.get("overall_rate", 0.0), c.get("direction", "long"), pool_id='pool2')
            if tier == "精品": elite_count += 1
            elif tier == "优质": good_count += 1
        cumul_lbl = getattr(self, "_cumul_count_lbl", None)
        if cumul_lbl:
            cumul_lbl.setText(
                f"共 {total} 个 | 做多 {long_count} | 做空 {short_count} | "
                f"精品 {elite_count} | 优质 {good_count}"
            )

        def _populate_table(tbl, combos, pool_id):
            tbl.setSortingEnabled(False)
            tbl.setRowCount(0)
            for seq, c in enumerate(combos, start=1):
                row = tbl.rowCount()
                tbl.insertRow(row)

                direction_val = c.get("direction", "long")
                dir_color     = LONG_COLOR if direction_val == "long" else SHORT_COLOR
                dir_str       = "做多" if direction_val == "long" else "做空"
                overall_rate  = c.get("overall_rate", 0.0)
                avg_rate      = c.get("avg_rate", 0.0)
                overall_color = _rate_color(overall_rate, direction_val)
                avg_color     = _rate_color(avg_rate, direction_val)
                rate_std      = c.get("rate_std", 0.0)
                score         = c.get("综合评分", 0.0)
                baseline      = 0.61 if direction_val == "long" else 0.47
                score_color   = _rate_color(
                    score / 100.0 * 0.75 + baseline * (1 - score / 100.0),
                    direction_val
                )

                # 各状态命中率明细
                breakdown  = c.get("market_state_breakdown") or {}
                dom_state  = _format_state_detail(breakdown, direction_val)

                # 单次EV（实时计算，避免旧数据缺字段）
                ev_pct   = _ev_per_trigger_pct(overall_rate, direction_val, pool_id=pool_id)

                # 估算总盈亏（实时计算，避免旧数据用错策略池参数）
                total_triggers_val = c.get("total_triggers", 0) or 0
                pnl_pct  = round(ev_pct * total_triggers_val, 4)
                pnl_str    = f"{pnl_pct:+.2f}%" if pnl_pct != 0 else "0.00%"
                pnl_color  = _pnl_color(pnl_pct)
                ev_str   = f"{ev_pct:+.2f}%" if ev_pct != 0 else "0.00%"
                ev_color = _pnl_color(ev_pct)

                # 层级（根据综合命中率、方向与策略池）
                tier_str   = _tier_from_rate(overall_rate, direction_val, pool_id=pool_id)
                tier_color = _tier_color(tier_str) if tier_str else TEXT_DIM

                # 随机基准 & 超越随机幅度（edge_over_random）
                avg_rb = c.get("avg_random_baseline", 0.0) or 0.0
                if avg_rb > 0.0:
                    edge = overall_rate - avg_rb
                    edge_sign = "+" if edge >= 0 else ""
                    rb_tip = f"随机基准 {avg_rb:.1%}，策略超越随机 {edge_sign}{edge:.1%}"
                    # 颜色：超越10%+绿，5-10%黄，<5%橙红
                    if edge >= 0.10:
                        rb_color = "#4caf50"
                    elif edge >= 0.05:
                        rb_color = ACCENT_GOLD
                    else:
                        rb_color = "#ff7043"
                    rb_str = f"{avg_rb:.1%}（{edge_sign}{edge:.0%}）"
                else:
                    rb_str, rb_color, rb_tip = "-", TEXT_DIM, "暂无随机基准（需重新分析以生成）"

                _set_item(tbl, row,  0, str(seq), TEXT_DIM)
                _set_item(tbl, row,  1, dir_str, dir_color, bold=True)
                _set_item(tbl, row,  2, tier_str or "--", tier_color, bold=bool(tier_str))
                _set_item(tbl, row,  3, str(c.get("appear_rounds", 0)), ACCENT_GOLD,
                          sort_value=c.get("appear_rounds", 0))
                _set_item(tbl, row,  4, str(c.get("total_triggers", "")),
                          sort_value=c.get("total_triggers", 0))
                _set_item(tbl, row,  5, str(c.get("total_hits", "")),
                          sort_value=c.get("total_hits", 0))
                _set_item(tbl, row,  6, f"{overall_rate:.1%}", overall_color, bold=True,
                          sort_value=overall_rate)
                _set_item(tbl, row,  7, f"{avg_rate:.1%}", avg_color, sort_value=avg_rate)
                _set_item(tbl, row,  8, f"{rate_std:.3f}", TEXT_DIM, sort_value=rate_std)
                _set_item(tbl, row,  9, f"{score:.1f}", score_color, bold=True,
                          sort_value=score)
                _set_item(tbl, row, 10, rb_str, rb_color,
                          sort_value=avg_rb, tooltip=rb_tip)
                _set_item(tbl, row, 11, dom_state, TEXT_DIM)
                _set_item(tbl, row, 12, pnl_str, pnl_color, bold=(pnl_pct != 0),
                          sort_value=pnl_pct)
                _set_item(tbl, row, 13, ev_str, ev_color, bold=(ev_pct != 0),
                          sort_value=ev_pct)
                # 平均持仓（0 显示 "-"）
                avg_hold = c.get("avg_hold_bars", 0) or 0
                avg_hold_str = str(avg_hold) if avg_hold else "-"
                decay_tip = ""
                if avg_hold >= 10:
                    d1, d2 = int(avg_hold * 0.50), int(avg_hold * 0.70)
                    decay_tip = f"衰减计划: {d1}根/-6% | {d2}根/-3%"
                _set_item(tbl, row, 14, avg_hold_str, TEXT_DIM, sort_value=avg_hold,
                          tooltip=decay_tip or None)
                _set_item(tbl, row, 15,
                          _format_conditions(c.get("conditions", []), c.get("direction", "")),
                          TEXT_DIM)
            tbl.setSortingEnabled(True)

        _populate_table(self._cumul_table_p1, combos_p1, 'pool1')
        _populate_table(self._cumul_table_p2, combos_p2, 'pool2')

    def _export_cumulative_txt(self):
        """导出累计结果为TXT文件"""
        combos_p1 = getattr(self, "_latest_cumulative_combos_p1", None)
        combos_p2 = getattr(self, "_latest_cumulative_combos_p2", None)
        
        if combos_p1 is None or combos_p2 is None:
            try:
                from core import signal_store
                dir_map = {"全部": None, "做多": "long", "做空": "short"}
                direction = dir_map.get(
                    self._cmb_direction.currentText() if hasattr(self, '_cmb_direction') else "全部"
                )
                min_rounds = self._spn_min_rounds.value() if hasattr(self, '_spn_min_rounds') else 1
                
                c1, _ = signal_store.get_cumulative_results(top_n=1000, direction=direction, pool_id='pool1')
                c2, _ = signal_store.get_cumulative_results(top_n=1000, direction=direction, pool_id='pool2')
                combos_p1 = [c for c in c1 if c.get("appear_rounds", 0) >= min_rounds]
                combos_p2 = [c for c in c2 if c.get("appear_rounds", 0) >= min_rounds]
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "导出失败", f"读取数据失败:\n{e}")
                return
        
        if not combos_p1 and not combos_p2:
            QtWidgets.QMessageBox.information(self, "无数据", "当前没有可导出的累计结果。")
            return

        default_name = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出累计结果",
            f"cumulative_results_{default_name}.txt",
            "Text Files (*.txt)"
        )
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        
        headers = [
            "序号", "方向", "层级", "出现轮次", "累计触发", "累计命中",
            "综合命中率", "平均命中率", "波动", "综合评分",
            "各状态命中率", "估算总盈亏", "单次EV", "平均持仓", "条件组合"
        ]
        
        def _write_pool(f, combos, pool_id, title):
            f.write(f"=== {title} ===\n")
            f.write("\t".join(headers) + "\n")
            seen_keys = set()
            seq = 0
            for c in combos:
                combo_key = c.get("combo_key")
                if combo_key:
                    if combo_key in seen_keys:
                        continue
                    seen_keys.add(combo_key)
                seq += 1
                direction_val = c.get("direction", "long")
                dir_str = "做多" if direction_val == "long" else "做空"
                overall_rate = c.get("overall_rate", 0.0)
                avg_rate = c.get("avg_rate", 0.0)
                rate_std = c.get("rate_std", 0.0)
                score = c.get("综合评分", 0.0)
                ev_pct = _ev_per_trigger_pct(overall_rate, direction_val, pool_id=pool_id)
                # 实时计算，避免旧数据用错策略池参数
                pnl_pct = round(ev_pct * (c.get("total_triggers", 0) or 0), 4)

                tier_str = _tier_from_rate(overall_rate, direction_val, pool_id=pool_id) or "--"
                
                breakdown = c.get("market_state_breakdown") or {}
                state_detail = _format_state_detail(breakdown, direction_val)
                
                conditions_str = _format_conditions(
                    c.get("conditions", []),
                    c.get("direction", "")
                )
                
                avg_hold = c.get("avg_hold_bars", 0) or 0
                row = [
                    str(seq),
                    dir_str,
                    tier_str,
                    str(c.get("appear_rounds", 0)),
                    str(c.get("total_triggers", 0)),
                    str(c.get("total_hits", 0)),
                    f"{overall_rate:.1%}",
                    f"{avg_rate:.1%}",
                    f"{rate_std:.3f}",
                    f"{score:.1f}",
                    state_detail,
                    f"{pnl_pct:+.2f}%" if pnl_pct != 0 else "0.00%",
                    f"{ev_pct:+.2f}%" if ev_pct != 0 else "0.00%",
                    str(avg_hold) if avg_hold else "-",
                    conditions_str,
                ]
                f.write("\t".join(row) + "\n")
            f.write("\n")
            return seq

        try:
            with open(path, "w", encoding="utf-8") as f:
                cnt1 = _write_pool(f, combos_p1, 'pool1', "策略池 1（TP 0.6% / SL 0.8%，做多≥64% / 做空≥52%）")
                cnt2 = _write_pool(f, combos_p2, 'pool2', "策略池 2（TP 1.0% / SL 0.8%，做多≥52% / 做空≥52%）")
            
            QtWidgets.QMessageBox.information(self, "导出完成", f"已导出 {cnt1 + cnt2} 条记录到:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", f"写入文件失败:\n{e}")

    def _backup_to_github(self):
        """备份 signal_store 数据到 GitHub 目录（双 pool）"""
        try:
            from core import signal_store
            result = signal_store.backup_to_github()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "备份失败", f"写入失败:\n{e}")
            return

        target_dir = result.get("target_dir", "")
        files = result.get("files", []) or []
        if not files:
            QtWidgets.QMessageBox.information(self, "无数据", "当前没有可备份的数据。")
            return

        file_list = "\n".join(os.path.basename(p) for p in files)
        QtWidgets.QMessageBox.information(
            self,
            "备份完成",
            f"已备份到:\n{target_dir}\n\n文件:\n{file_list}"
        )

    def _import_pool_data(self):
        """从外部 TXT/JSON 文件导入 Pool1/Pool2 数据，追加合并到现有数据中。"""
        from PyQt6 import QtWidgets as _QW

        # 选择要导入的文件（优先TXT，也支持JSON）
        path, _ = _QW.QFileDialog.getOpenFileName(
            self,
            "选择要导入的文件（TXT 或 JSON）",
            "",
            "TXT 文件 (*.txt);;JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        try:
            from core import signal_store

            if path.lower().endswith(".txt"):
                # TXT 格式：自动识别 Pool1/Pool2
                result = signal_store.import_from_txt(path)
                p1_count = result.get("pool1_imported", 0)
                p2_count = result.get("pool2_imported", 0)
                errors = result.get("errors", [])

                msg = f"导入完成！\n\n"
                msg += f"Pool 1 导入：{p1_count} 条\n"
                msg += f"Pool 2 导入：{p2_count} 条\n"
                if errors:
                    msg += f"\n警告（{len(errors)} 条）：\n"
                    msg += "\n".join(errors[:5])
                    if len(errors) > 5:
                        msg += f"\n... 还有 {len(errors) - 5} 条警告"

                _QW.QMessageBox.information(self, "导入完成", msg)

                # 刷新两个池的缓存
                signal_store.invalidate_cache()
                if p1_count > 0:
                    signal_store.rebuild_pruned_cache(pool_id='pool1')
                if p2_count > 0:
                    signal_store.rebuild_pruned_cache(pool_id='pool2')

            else:
                # JSON 格式：需要选择目标池
                pool_choice, ok = _QW.QInputDialog.getItem(
                    self, "选择导入目标",
                    "将 JSON 数据导入到哪个策略池？",
                    ["Pool 1（TP 0.6% / SL 0.8%）", "Pool 2（TP 1.0% / SL 0.8%）"],
                    0, False
                )
                if not ok:
                    return
                pool_id = "pool1" if "Pool 1" in pool_choice else "pool2"

                result = signal_store.import_from_file(path, pool_id)
                merged_rounds  = result.get("merged_rounds", 0)
                merged_combos  = result.get("merged_combos", 0)
                skipped_rounds = result.get("skipped_rounds", 0)

                _QW.QMessageBox.information(
                    self,
                    "导入完成",
                    f"已合并到 {pool_choice}\n\n"
                    f"新增轮次：{merged_rounds}\n"
                    f"累计组合更新：{merged_combos}\n"
                    f"跳过重复轮次：{skipped_rounds}\n\n"
                    f"数据已写入，请刷新查看。"
                )

                signal_store.invalidate_cache()
                signal_store.rebuild_pruned_cache(pool_id=pool_id)

        except Exception as e:
            import traceback
            _QW.QMessageBox.critical(self, "导入失败", f"导入出错：\n{e}\n\n{traceback.format_exc()}")
            return

        self._refresh_cumul_tables()
        self._refresh_history_text()

    def _refresh_backtest_feedback_table(self):
        """刷新回测信号反馈面板：从纸交易记录统计各组合表现。"""
        tbl = self._live_table
        tbl.setSortingEnabled(False)
        tbl.setRowCount(0)

        try:
            from core import signal_store
            cumul1 = signal_store.get_cumulative(pool_id='pool1')
            cumul2 = signal_store.get_cumulative(pool_id='pool2')
            cumulative = {**cumul1, **cumul2}
        except Exception:
            cumulative = {}

        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "live_trade_history.json",
        )
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        trades = data.get("trades", []) if isinstance(data, dict) else []
        stats: Dict[str, dict] = {}

        def _normalize_regime_label(regime: str) -> str:
            if not regime:
                return "-"
            mapping = {
                "多头趋势": "多头",
                "空头趋势": "空头",
                "震荡市": "震荡",
                "震荡偏多": "震荡偏多",
                "震荡偏空": "震荡偏空",
            }
            return mapping.get(regime, regime)

        def _is_market_match(direction: str, regime: str) -> bool:
            if not regime:
                return False
            if direction == "long":
                return ("多头" in regime) or ("偏多" in regime)
            return ("空头" in regime) or ("偏空" in regime)

        def _is_win(trade: dict) -> bool:
            pnl = trade.get("realized_pnl", None)
            if isinstance(pnl, (int, float)):
                return pnl > 0
            pct = trade.get("profit_pct", None)
            if isinstance(pct, (int, float)):
                return pct > 0
            reason = str(trade.get("close_reason", ""))
            return "止盈" in reason

        def _is_stop_loss(trade: dict) -> bool:
            reason = str(trade.get("close_reason", "")) + str(trade.get("close_reason_detail", ""))
            return "止损" in reason

        for trade in trades:
            if trade.get("status") != "CLOSED":
                continue
            combo_keys = trade.get("signal_combo_keys") or []
            template = trade.get("template_fingerprint")
            if not combo_keys and isinstance(template, str):
                tpl = template.lower()
                if tpl.startswith("long|") or tpl.startswith("short|"):
                    combo_keys = [template]
            if not combo_keys:
                continue

            for key in combo_keys:
                if not isinstance(key, str) or "|" not in key:
                    continue
                direction = key.split("|", 1)[0].lower()
                if direction not in ("long", "short"):
                    direction = "long" if str(trade.get("side", "")).upper() == "LONG" else "short"

                entry = stats.setdefault(
                    key,
                    {
                        "direction": direction,
                        "total": 0,
                        "wins": 0,
                        "stop_loss": 0,
                        "market_match": 0,
                        "regimes": {},
                    },
                )
                entry["total"] += 1
                win = _is_win(trade)
                entry["wins"] += 1 if win else 0
                entry["stop_loss"] += 1 if _is_stop_loss(trade) else 0

                regime = trade.get("regime_at_entry", "")
                label = _normalize_regime_label(regime)
                reg_stats = entry["regimes"].setdefault(label, {"total": 0, "wins": 0})
                reg_stats["total"] += 1
                reg_stats["wins"] += 1 if win else 0
                entry["market_match"] += 1 if _is_market_match(direction, regime) else 0

        if not stats:
            row = tbl.rowCount()
            tbl.insertRow(row)
            message = "暂无组合信号记录，请在「精品信号模式」运行模拟盘/纸交易生成回测数据。"
            _set_item(tbl, row, 0, message, TEXT_DIM)
            if tbl.columnCount() > 1:
                tbl.setSpan(row, 0, 1, tbl.columnCount())
            tbl.setSortingEnabled(True)
            return

        rows: List[dict] = []
        for key, entry in stats.items():
            total = entry["total"]
            wins = entry["wins"]
            hit_rate = wins / total if total else 0.0
            market_match = entry["market_match"] / total if total else 0.0
            stop_loss_rate = entry["stop_loss"] / total if total else 0.0

            cumulative_entry = cumulative.get(key) or {}
            pool_rate = cumulative_entry.get("avg_rate", 0.0)
            tier_rate = cumulative_entry.get("overall_rate", pool_rate)
            _pool_id_for_tier = cumulative_entry.get('pool_id', 'pool1')
            tier_str = _tier_from_rate(tier_rate, entry["direction"], pool_id=_pool_id_for_tier) or "--"

            conditions: List[str] = []
            if "|" in key:
                conditions = [c for c in key.split("|", 1)[1].split("+") if c]
            conditions_label = _format_conditions(conditions, entry["direction"]) if conditions else key

            regime_parts = []
            for label, reg in entry["regimes"].items():
                if reg["total"] == 0:
                    continue
                reg_rate = reg["wins"] / reg["total"]
                regime_parts.append((reg["total"], f"{label}:{reg_rate:.0%}"))
            regime_parts.sort(key=lambda x: x[0], reverse=True)
            regime_text = " ".join([p[1] for p in regime_parts]) if regime_parts else "-"

            issues = []
            if total < 5:
                issues.append("样本不足")
            if pool_rate > 0 and hit_rate < pool_rate - 0.10:
                issues.append("命中率低于预期")
            if market_match < 0.5:
                issues.append("市场状态不匹配")
            if stop_loss_rate > 0.6:
                issues.append("止损频发")
            issues_text = "、".join(issues) if issues else "-"

            rows.append({
                "key": key,
                "direction": entry["direction"],
                "tier": tier_str,
                "conditions_label": conditions_label,
                "total": total,
                "hit_rate": hit_rate,
                "pool_rate": pool_rate,
                "regime_text": regime_text,
                "market_match": market_match,
                "issues": issues_text,
                "pool_gap": pool_rate - hit_rate,
            })

        def _sort_key(r: dict):
            pool_missing = 1 if r["pool_rate"] <= 0 else 0
            return (pool_missing, r["market_match"], -r["pool_gap"])

        rows.sort(key=_sort_key)

        for seq, r in enumerate(rows, start=1):
            row = tbl.rowCount()
            tbl.insertRow(row)
            dir_str = "做多" if r["direction"] == "long" else "做空"
            dir_color = LONG_COLOR if r["direction"] == "long" else SHORT_COLOR
            tier_color = _tier_color(r["tier"]) if r["tier"] not in ("", "--") else TEXT_DIM
            hit_color = _rate_color(r["hit_rate"], r["direction"])
            pool_color = _rate_color(r["pool_rate"], r["direction"])

            _set_item(tbl, row, 0, str(seq), TEXT_DIM)
            _set_item(tbl, row, 1, dir_str, dir_color, bold=True)
            _set_item(tbl, row, 2, r["tier"], tier_color, bold=r["tier"] not in ("", "--"))
            _set_item(tbl, row, 3, r["conditions_label"], TEXT_DIM)
            _set_item(tbl, row, 4, str(r["total"]), TEXT_DIM, sort_value=r["total"])
            _set_item(tbl, row, 5, f"{r['hit_rate']:.1%}", hit_color, bold=True,
                      sort_value=r["hit_rate"])
            _set_item(tbl, row, 6, f"{r['pool_rate']:.1%}", pool_color,
                      sort_value=r["pool_rate"])
            _set_item(tbl, row, 7, r["regime_text"], TEXT_DIM)
            _set_item(tbl, row, 8, f"{r['market_match']:.0%}", TEXT_DIM,
                      sort_value=r["market_match"])
            _set_item(tbl, row, 9, r["issues"], WARN_COLOR if r["issues"] != "-" else TEXT_DIM)
        tbl.setSortingEnabled(True)

    def _refresh_risk_display(self):
        """从 signal_store 读取最大连亏次数，更新风控显示标签。"""
        try:
            from core import signal_store
            cumul1 = signal_store.get_cumulative(pool_id='pool1')
            cumul2 = signal_store.get_cumulative(pool_id='pool2')
            cumulative = {**cumul1, **cumul2}
            max_streak = max(
                (e.get('live_tracking', {}).get('streak_loss', 0) for e in cumulative.values()),
                default=0
            )
            color = DECAY_SEVERE if max_streak >= 5 else TEXT_DIM
            self._lbl_streak.setStyleSheet(f"color: {color}; font-size: 14px;")
            self._lbl_streak.setText(f"当前连亏: {max_streak}次")
        except Exception:
            pass

    def _refresh_history_text(self):
        try:
            from core import signal_store
            rounds_p1 = signal_store.get_rounds(pool_id='pool1')
            rounds_p2 = signal_store.get_rounds(pool_id='pool2')
        except Exception:
            return

        def _render_rounds(rounds, pool_label):
            lines = []
            if not rounds:
                return lines
            lines.append(f"═══ {pool_label} ═══")
            for i, rnd in enumerate(reversed(rounds[-20:])):
                rnd_id  = rnd.get('round_id', '?')
                ts      = _format_timestamp(rnd.get('timestamp', ''))
                results = rnd.get('results', [])
                cnt     = len(results)
                top3    = results[:3]
                lines.append(f"【第 {rnd_id} 轮】{ts}  |  共 {cnt} 个有效组合")
                for j, item in enumerate(top3):
                    dir_str = "做多" if item.get("direction") == "long" else "做空"
                    conds   = _format_conditions(item.get("conditions", []), item.get("direction", ""))
                    tier    = item.get("tier", "")
                    hr      = item.get("hit_rate", 0.0)
                    tc      = item.get("trigger_count", 0)
                    lines.append(
                        f"  {j+1}. {dir_str} {tier}  "
                        f"命中率 {hr:.1%}  "
                        f"触发 {tc} 次  [{conds}]"
                    )
                lines.append("")
            return lines

        lines = _render_rounds(rounds_p1, "策略池 1（TP 0.6% / SL 0.8%）")
        lines += _render_rounds(rounds_p2, "策略池 2（TP 1.0% / SL 0.8%）")
        self._history_text.setPlainText("\n".join(lines))

    # ── 外部接口 ──────────────────────────────────────────────────────────

    def set_data(self, df):
        """
        由 MainWindow 在数据加载完成后调用，传入历史 K 线 DataFrame。
        若 _auto_run_on_next_data 为 True（由"换新数据再验证"触发），
        则新数据到达后自动开始分析。
        """
        self._df = df
        bar_count = len(df) if df is not None else 0
        if bar_count > 0:
            self._status_lbl.setText(
                f"已加载 {bar_count:,} 根K线  —  点击「开始分析」运行"
            )
            if self._auto_run_on_next_data:
                self._auto_run_on_next_data = False
                # 延迟 200ms 让 UI 渲染完毕再启动分析
                QtCore.QTimer.singleShot(200, self._run_analysis)
        else:
            self._status_lbl.setText("数据为空，请重新加载")
            self._auto_run_on_next_data = False

    def refresh_live_data(self):
        """外部调用：刷新实盘监控面板和风控显示（供 MainWindow 定时调用）。"""
        self._refresh_backtest_feedback_table()
        self._refresh_risk_display()
