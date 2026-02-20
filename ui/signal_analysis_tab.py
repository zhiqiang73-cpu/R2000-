"""
信号分析标签页
功能：多条件组合回测分析，发现高胜率信号组合
后台线程分析 + 1% 粒度进度条 + 本轮/累计双结果表 + 多轮历史区
新增：累计表市场状态列/估算P&L列 + 实盘监控面板 + 风控开关面板
"""
from __future__ import annotations

import json
import os
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

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df
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

            if not self._stop:
                long_results = analyze(self._df, 'long', progress_cb=cb_long)
                all_results.extend(long_results)

            if not self._stop:
                short_results = analyze(self._df, 'short', progress_cb=cb_short)
                all_results.extend(short_results)

            if not self._stop:
                self.progress.emit(99, "写入持久化状态...")
                long_res  = [r for r in all_results if r['direction'] == 'long']
                short_res = [r for r in all_results if r['direction'] == 'short']
                if long_res:
                    signal_store.merge_round(long_res,  direction='long',  bar_count=len(self._df))
                if short_res:
                    signal_store.merge_round(short_res, direction='short', bar_count=len(self._df))
                # 两次 merge 都完成后，执行一次去重并更新缓存（O(n²)，只跑一次）
                signal_store.rebuild_pruned_cache()

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
            # 一次调用同时拿到 items 和 cumulative，消除重复 get_cumulative() 调用
            combos, cumulative = signal_store.get_cumulative_results(top_n=200)
            rounds = signal_store.get_rounds()
            self.finished.emit({
                'combos':     combos,
                'cumulative': cumulative,
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


def _tier_from_rate(rate: float, direction: str) -> str:
    """根据综合命中率和方向返回层级（与 signal_analyzer 门槛一致）"""
    if direction == 'long':
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


def _make_table(headers: List[str]) -> QtWidgets.QTableWidget:
    tbl = QtWidgets.QTableWidget(0, len(headers))
    tbl.setHorizontalHeaderLabels(headers)
    tbl.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
    tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    tbl.setAlternatingRowColors(True)
    tbl.verticalHeader().setVisible(False)
    tbl.horizontalHeader().setStretchLastSection(True)
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
    tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    tbl.setSortingEnabled(True)
    return tbl


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
              sort_value: Optional[float] = None):
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
        self._spn_min_rounds.setValue(3)
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
        
        cumul_count_row.addStretch(1)
        self._cumul_count_lbl = QtWidgets.QLabel("共 0 个 | 做多 0 | 做空 0 | 精品 0 | 优质 0")
        self._cumul_count_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        cumul_count_row.addWidget(self._cumul_count_lbl)
        right_layout.addLayout(cumul_count_row)

        self._cumul_table = _make_table([
            "#", "方向", "层级", "出现轮次", "累计触发", "累计命中",
            "综合命中率", "平均命中率", "波动", "综合评分",
            "各状态命中率", "估算总盈亏", "条件组合"
        ])
        right_layout.addWidget(self._cumul_table)
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
        self._refresh_live_monitor_table()
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
                signal_store.clear()
            except Exception:
                pass
            self._round_table.setRowCount(0)
            self._cumul_table.setRowCount(0)
            self._live_table.setRowCount(0)
            self._history_text.clear()
            self._status_lbl.setText("记录已清空")

    def _on_risk_changed(self):
        self._risk_state["daily_loss_limit"]  = self._chk_daily_loss.isChecked()
        self._risk_state["streak_loss_pause"] = self._chk_streak.isChecked()
        _save_risk_state(self._risk_state)

    def _on_filter_changed(self):
        """筛选条件变化时刷新累计结果表格"""
        self._refresh_cumulative_table()

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
        self._refresh_live_monitor_table()
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

        self._thread = QtCore.QThread(self)
        self._worker = SignalAnalysisWorker(self._df)
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
            
            # 获取数据（已在后台去重+层级过滤，这里只做方向和轮次过滤）
            combos, cumulative = signal_store.get_cumulative_results(top_n=500, direction=direction)
            
            # 应用最少轮次筛选
            combos = [c for c in combos if c.get("appear_rounds", 0) >= min_rounds]
            
        except Exception:
            return

        # 更新总数统计（含层级数量）- 基于过滤后的数据
        filtered_cumulative = {c.get('combo_key', ''): c for c in combos}
        total = len(filtered_cumulative)
        long_count = sum(1 for c in combos if c.get("direction") == "long")
        short_count = total - long_count
        elite_count = good_count = 0
        for c in combos:
            tier = _tier_from_rate(c.get("overall_rate", 0.0), c.get("direction", "long"))
            if tier == "精品": elite_count += 1
            elif tier == "优质": good_count += 1
        cumul_lbl = getattr(self, "_cumul_count_lbl", None)
        if cumul_lbl:
            cumul_lbl.setText(
                f"共 {total} 个 | 做多 {long_count} | 做空 {short_count} | "
                f"精品 {elite_count} | 优质 {good_count}"
            )

        tbl = self._cumul_table
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

            # 估算总盈亏
            pnl_pct    = c.get("estimated_pnl_pct", 0.0)
            pnl_str    = f"{pnl_pct:+.2f}%" if pnl_pct != 0 else "0.00%"
            pnl_color  = _pnl_color(pnl_pct)

            # 层级（根据综合命中率与方向）
            tier_str   = _tier_from_rate(overall_rate, direction_val)
            tier_color = _tier_color(tier_str) if tier_str else TEXT_DIM

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
            _set_item(tbl, row, 10, dom_state, TEXT_DIM)
            _set_item(tbl, row, 11, pnl_str, pnl_color, bold=(pnl_pct != 0),
                      sort_value=pnl_pct)
            _set_item(tbl, row, 12,
                      _format_conditions(c.get("conditions", []), c.get("direction", "")),
                      TEXT_DIM)
        tbl.setSortingEnabled(True)

    def _export_cumulative_txt(self):
        """导出累计结果为TXT文件"""
        try:
            from core import signal_store
            combos, _ = signal_store.get_cumulative_results(top_n=1000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", f"读取数据失败:\n{e}")
            return
        
        if not combos:
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
            "各状态命中率", "估算总盈亏", "条件组合"
        ]
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\t".join(headers) + "\n")
                
                for seq, c in enumerate(combos, start=1):
                    direction_val = c.get("direction", "long")
                    dir_str = "做多" if direction_val == "long" else "做空"
                    overall_rate = c.get("overall_rate", 0.0)
                    avg_rate = c.get("avg_rate", 0.0)
                    rate_std = c.get("rate_std", 0.0)
                    score = c.get("综合评分", 0.0)
                    pnl_pct = c.get("estimated_pnl_pct", 0.0)
                    
                    # 层级
                    tier_str = _tier_from_rate(overall_rate, direction_val) or "--"
                    
                    # 各状态命中率明细
                    breakdown = c.get("market_state_breakdown") or {}
                    state_detail = _format_state_detail(breakdown, direction_val)
                    
                    # 条件组合
                    conditions_str = _format_conditions(
                        c.get("conditions", []),
                        c.get("direction", "")
                    )
                    
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
                        conditions_str,
                    ]
                    f.write("\t".join(row) + "\n")
            
            QtWidgets.QMessageBox.information(self, "导出完成", f"已导出 {len(combos)} 条记录到:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", f"写入文件失败:\n{e}")

    def _refresh_live_monitor_table(self):
        """刷新实盘监控面板：显示有实盘记录的组合及其命中率衰减情况。"""
        try:
            from core import signal_store
            cumulative = signal_store.get_cumulative()
        except Exception:
            return

        tbl = self._live_table
        tbl.setRowCount(0)

        rows = []
        for key, entry in cumulative.items():
            lt = entry.get('live_tracking') or {}
            if lt.get('total', 0) == 0:
                continue
            rows.append((key, entry, lt))

        # 衰减最严重的排前面
        def _decay_key(item):
            _, entry, lt = item
            return entry.get('avg_rate', 0.0) - lt.get('live_rate', 0.0)
        rows.sort(key=_decay_key, reverse=True)

        for key, entry, lt in rows:
            row = tbl.rowCount()
            tbl.insertRow(row)

            conditions   = entry.get('conditions', [])
            combo_label  = _format_conditions(conditions[:3], entry.get("direction", ""))
            if len(conditions) > 3:
                combo_label += f" +{len(conditions) - 3}"

            avg_rate  = entry.get('avg_rate', 0.0)
            live_rate = lt.get('live_rate', 0.0)
            total     = lt.get('total', 0)
            streak    = lt.get('streak_loss', 0)
            decay     = avg_rate - live_rate

            # 状态判定
            if total < 10 or decay < 0.05:
                status_text  = "正常"
                status_color = GOOD_COLOR
            elif decay < 0.10:
                status_text  = "⚠ 轻微衰减"
                status_color = DECAY_MILD
            else:
                status_text  = "⛔ 严重衰减"
                status_color = DECAY_SEVERE

            _set_item(tbl, row, 0, combo_label, TEXT_DIM)
            _set_item(tbl, row, 1, f"{avg_rate:.1%}",
                      _rate_color(avg_rate, entry.get("direction", "long")))
            _set_item(tbl, row, 2, f"{live_rate:.1%}",
                      _rate_color(live_rate, entry.get("direction", "long")), bold=True)
            _set_item(tbl, row, 3, str(total), TEXT_DIM)
            _set_item(tbl, row, 4, str(streak),
                      DECAY_SEVERE if streak >= 5 else TEXT_DIM)
            _set_item(tbl, row, 5, status_text, status_color, bold=True)

    def _refresh_risk_display(self):
        """从 signal_store 读取最大连亏次数，更新风控显示标签。"""
        try:
            from core import signal_store
            cumulative = signal_store.get_cumulative()
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
            rounds = signal_store.get_rounds()
        except Exception:
            return

        lines = []
        for i, rnd in enumerate(reversed(rounds[-20:])):
            rnd_id    = rnd.get('round_id', '?')
            ts        = _format_timestamp(rnd.get('timestamp', ''))
            results   = rnd.get('results', [])
            cnt       = len(results)
            top3      = results[:3]
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
        self._refresh_live_monitor_table()
        self._refresh_risk_display()
