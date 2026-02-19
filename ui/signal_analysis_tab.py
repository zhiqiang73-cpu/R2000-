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

            if not self._stop:
                self.progress.emit(100, f"分析完成，共 {len(all_results)} 个有效组合")
                self.finished.emit(all_results)

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


def _make_tree_widget(headers: List[str]) -> QtWidgets.QTreeWidget:
    """创建带样式的 QTreeWidget（用于折叠式精品推荐表）"""
    tree = QtWidgets.QTreeWidget()
    tree.setHeaderLabels(headers)
    tree.setAlternatingRowColors(True)
    tree.setRootIsDecorated(True)
    tree.setIndentation(20)
    tree.setAnimated(True)
    tree.setExpandsOnDoubleClick(True)
    tree.setStyleSheet(f"""
        QTreeWidget {{
            background-color: {BG_CARD};
            alternate-background-color: {BG_PANEL};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_COLOR};
            font-size: 12px;
            outline: none;
        }}
        QTreeWidget::item {{
            padding: 4px 6px;
            border: none;
        }}
        QTreeWidget::item:selected {{
            background-color: #2A3A4A;
            color: {TEXT_PRIMARY};
        }}
        QTreeWidget::item:hover {{
            background-color: #2E3640;
        }}
        QTreeWidget::branch {{
            background-color: transparent;
        }}
        QTreeWidget::branch:has-siblings:!adjoins-item {{
            border-image: none;
        }}
        QTreeWidget::branch:has-siblings:adjoins-item {{
            border-image: none;
        }}
        QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {{
            border-image: none;
        }}
        QTreeWidget::branch:has-children:!has-siblings:closed,
        QTreeWidget::branch:closed:has-children:has-siblings {{
            image: none;
            border-image: none;
        }}
        QTreeWidget::branch:open:has-children:!has-siblings,
        QTreeWidget::branch:open:has-children:has-siblings {{
            image: none;
            border-image: none;
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
    """)
    header = tree.header()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
    return tree


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
        self._running = False
        self._risk_state = _load_risk_state()
        self._auto_run_on_next_data = False   # 新数据到达后自动开始分析
        self._family_sort_col = None
        self._family_sort_desc = True

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

        self._btn_family = QtWidgets.QPushButton("🧩  族群汇总")
        self._btn_family.setFixedHeight(34)
        self._btn_family.setToolTip("按指标族群聚合（忽略阈值差异）")
        self._btn_family.clicked.connect(self._on_family_summary)

        info_lbl = QtWidgets.QLabel(
            "分析策略：9指标×2阈值=18条件/方向 | 2-5条件组合 | 含手续费门槛 64/67/71%"
        )
        info_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_new_data)
        btn_row.addWidget(self._btn_auto_50)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(self._btn_family)
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

        self._cumul_table = _make_table([
            "#", "方向", "出现轮次", "累计触发", "累计命中",
            "综合命中率", "平均命中率", "波动", "综合评分",
            "各状态命中率", "估算总盈亏", "条件组合"
        ])
        right_layout.addWidget(self._cumul_table)
        splitter.addWidget(right_box)

        splitter.setSizes([450, 550])

        # ④-2 精品推荐表（双层策略池：精品层 + 高频层）
        family_box = QtWidgets.QGroupBox(
            "精品推荐 — 精品层（高质量）+ 高频层（高触发）"
        )
        family_box.setStyleSheet(self._group_box_style())
        family_layout = QtWidgets.QVBoxLayout(family_box)
        family_layout.setContentsMargins(6, 6, 6, 6)

        self._family_tree = self._make_family_tree()
        family_layout.addWidget(self._family_tree)
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

        # ④-2/⑤/⑥ 使用垂直分割器，允许手动调整高度
        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        bottom_splitter.setHandleWidth(4)
        bottom_splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER_COLOR}; }}")
        bottom_splitter.addWidget(family_box)
        bottom_splitter.addWidget(hist_box)
        bottom_splitter.addWidget(live_box)
        bottom_splitter.setSizes([280, 160, 180])

        # 总垂直分割器：上（本轮+累计表）↕ 下（族群+历史+监控）
        main_vsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_vsplit.setHandleWidth(5)
        main_vsplit.setStyleSheet(
            f"QSplitter::handle {{ background: {ACCENT_CYAN}; border-radius: 2px; }}"
        )
        main_vsplit.addWidget(splitter)
        main_vsplit.addWidget(bottom_splitter)
        main_vsplit.setSizes([400, 400])

        root.addWidget(main_vsplit, stretch=5)

        # 初始化：加载已有累计记录
        self._refresh_cumulative_table()
        self._refresh_family_table()
        self._refresh_history_text()
        self._refresh_live_monitor_table()
        self._refresh_risk_display()

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

    def _make_family_tree(self) -> QtWidgets.QTreeWidget:
        """创建精品推荐的折叠树形表格"""
        tree = QtWidgets.QTreeWidget()
        headers = [
            "层级", "市场状态", "序号", "方向", "开仓条件组合", "出现轮次",
            "状态触发", "状态命中", "状态命中率",
            "综合评分", "多头命中率", "空头命中率", "震荡命中率"
        ]
        tree.setHeaderLabels(headers)
        tree.setColumnCount(len(headers))
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(20)
        tree.setAnimated(True)
        tree.setExpandsOnDoubleClick(True)
        tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {BG_CARD};
                alternate-background-color: {BG_PANEL};
                color: {TEXT_PRIMARY};
                border: 1px solid {BORDER_COLOR};
                gridline-color: {BORDER_COLOR};
                font-size: 12px;
            }}
            QTreeWidget::item {{
                padding: 4px 8px;
                border: none;
            }}
            QTreeWidget::item:selected {{
                background-color: #2A3A4A;
                color: {TEXT_PRIMARY};
            }}
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                image: none;
                border-image: none;
            }}
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {{
                image: none;
                border-image: none;
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
        """)
        header = tree.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        return tree

    def _select_high_freq_top(self, combos: List[dict], top_n: int = 6,
                              min_triggers: int = 20,
                              min_score: float = 70.0,
                              max_conditions: int = 3,
                              max_overlap: float = 0.5,
                              min_state_rate: Optional[float] = None) -> List[dict]:
        """
        高频策略筛选：
        1. 条件数量 2-3 个
        2. 触发次数：
           - 普通模式：total_triggers >= min_triggers（全局）
           - 震荡市专项（min_state_rate 已设置）：state_triggers >= min_triggers
        3. 质量门槛：
           - 普通模式：综合评分 >= min_score
           - 震荡市专项：state_rate >= min_state_rate（状态专项命中率）
        4. 排序：命中率 > 触发次数（强调胜率优先）
        5. 多样性约束放宽到 50%
        """
        qualified = []
        for c in combos:
            conditions = c.get("conditions", [])
            cond_count = len(conditions)
            if min_state_rate is not None:
                # 震荡市专项模式：用状态专项命中率替代全局综合评分
                ok = (2 <= cond_count <= max_conditions
                      and c.get("state_triggers", 0) >= min_triggers
                      and c.get("state_rate", 0.0) >= min_state_rate)
            else:
                total_triggers = c.get("total_triggers", 0)
                score = c.get("综合评分", 0.0)
                ok = (2 <= cond_count <= max_conditions
                      and total_triggers >= min_triggers
                      and score >= min_score)
            if ok:
                qualified.append(c)

        sorted_combos = sorted(
            qualified,
            key=lambda c: (c.get("overall_rate", 0.0), c.get("total_triggers", 0)),
            reverse=True
        )

        selected: List[dict] = []
        selected_families: List[frozenset] = []

        for c in sorted_combos:
            if len(selected) >= top_n:
                break
            families = self._get_family_set(c.get("conditions", []))
            max_current_overlap = 0.0
            for sf in selected_families:
                overlap = self._family_overlap_ratio(families, sf)
                max_current_overlap = max(max_current_overlap, overlap)

            if max_current_overlap < max_overlap:
                selected.append(c)
                selected_families.append(families)

        return selected

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
            self._family_tree.clear()
            self._live_table.setRowCount(0)
            self._history_text.clear()
            self._status_lbl.setText("记录已清空")

    def _on_risk_changed(self):
        self._risk_state["daily_loss_limit"]  = self._chk_daily_loss.isChecked()
        self._risk_state["streak_loss_pause"] = self._chk_streak.isChecked()
        _save_risk_state(self._risk_state)

    @QtCore.pyqtSlot(int, str)
    def _on_progress(self, pct: int, text: str):
        self._progress.setValue(pct)
        self._status_lbl.setText(text)

    @QtCore.pyqtSlot(list)
    def _on_finished(self, results: list):
        self._set_running(False)
        self._populate_round_table(results)
        self._refresh_cumulative_table()
        self._refresh_family_table()
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

    def _on_family_summary(self):
        """手动刷新族群汇总视图。"""
        self._refresh_family_table()

    def _on_family_sort(self, section: int):
        """
        族群汇总表自定义排序：
        0/1 列不排序；2/3/4/5/6 列用于排序父行并重绘。
        """
        if section < 2:
            return
        if self._family_sort_col == section:
            self._family_sort_desc = not self._family_sort_desc
        else:
            self._family_sort_col = section
            self._family_sort_desc = True
        self._refresh_family_table()

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
            combos = signal_store.get_cumulative_results(top_n=200)
        except Exception:
            return

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

            _set_item(tbl, row,  0, str(seq), TEXT_DIM)
            _set_item(tbl, row,  1, dir_str, dir_color, bold=True)
            _set_item(tbl, row,  2, str(c.get("appear_rounds", 0)), ACCENT_GOLD,
                      sort_value=c.get("appear_rounds", 0))
            _set_item(tbl, row,  3, str(c.get("total_triggers", "")),
                      sort_value=c.get("total_triggers", 0))
            _set_item(tbl, row,  4, str(c.get("total_hits", "")),
                      sort_value=c.get("total_hits", 0))
            _set_item(tbl, row,  5, f"{overall_rate:.1%}", overall_color, bold=True,
                      sort_value=overall_rate)
            _set_item(tbl, row,  6, f"{avg_rate:.1%}", avg_color, sort_value=avg_rate)
            _set_item(tbl, row,  7, f"{rate_std:.3f}", TEXT_DIM, sort_value=rate_std)
            _set_item(tbl, row,  8, f"{score:.1f}", score_color, bold=True,
                      sort_value=score)
            _set_item(tbl, row,  9, dom_state, TEXT_DIM)
            _set_item(tbl, row, 10, pnl_str, pnl_color, bold=(pnl_pct != 0),
                      sort_value=pnl_pct)
            _set_item(tbl, row, 11,
                      _format_conditions(c.get("conditions", []), c.get("direction", "")),
                      TEXT_DIM)
        tbl.setSortingEnabled(True)

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

    # ── 智能合并辅助函数 ───────────────────────────────────────────────────

    def _get_condition_family(self, cond: str) -> str:
        """提取条件的指标族，如 'boll_pos_loose' → 'boll_pos'"""
        for suffix in ('_loose', '_strict'):
            if cond.endswith(suffix):
                return cond[:-len(suffix)]
        return cond

    def _get_family_set(self, conditions: List[str]) -> frozenset:
        """获取条件组合涉及的指标族集合"""
        return frozenset(self._get_condition_family(c) for c in conditions)

    def _family_overlap_ratio(self, families_a: frozenset, families_b: frozenset) -> float:
        """计算两个指标族集合的重叠度（Jaccard）"""
        if not families_a or not families_b:
            return 0.0
        intersection = len(families_a & families_b)
        union = len(families_a | families_b)
        return intersection / union if union > 0 else 0.0

    def _is_loose_version(self, cond: str) -> bool:
        """判断是否为宽松版条件"""
        return cond.endswith('_loose')

    def _merge_similar_combos(self, combos: List[dict]) -> List[dict]:
        """
        合并相似组合：
        1. 按指标族集合分组
        2. 同一组内，选取每个指标族的宽松版（覆盖范围更大）
        3. 返回合并后的代表组合列表
        """
        from collections import defaultdict

        # 按指标族集合分组
        family_groups: Dict[frozenset, List[dict]] = defaultdict(list)
        for c in combos:
            families = self._get_family_set(c.get("conditions", []))
            family_groups[families].append(c)

        merged: List[dict] = []
        for families, group in family_groups.items():
            if len(group) == 1:
                # 只有一个组合，直接保留
                merged.append(group[0])
            else:
                # 多个组合共用相同指标族，选择最优代表
                # 优先选：综合评分最高 + 使用宽松版条件多的
                def score_combo(c):
                    conds = c.get("conditions", [])
                    loose_count = sum(1 for cd in conds if self._is_loose_version(cd))
                    # 综合评分权重 + 宽松版数量加成
                    return c.get("综合评分", 0.0) + loose_count * 0.5
                best = max(group, key=score_combo)
                # 合并统计信息：累加触发/命中次数，取最高出现轮次
                merged_entry = dict(best)
                merged_entry["appear_rounds"] = max(g.get("appear_rounds", 0) for g in group)
                merged_entry["total_triggers"] = sum(int(g.get("total_triggers", 0) or 0) for g in group)
                merged_entry["total_hits"] = sum(int(g.get("total_hits", 0) or 0) for g in group)
                if merged_entry["total_triggers"] > 0:
                    merged_entry["overall_rate"] = merged_entry["total_hits"] / merged_entry["total_triggers"]
                merged_entry["_merged_count"] = len(group)  # 标记合并了多少个
                merged.append(merged_entry)

        return merged

    def _select_diverse_top(self, combos: List[dict], top_n: int = 6,
                            max_overlap: float = 0.3,
                            min_score: float = 80.0) -> List[dict]:
        """
        多样性选择（质量优先 + 多样性保障）：
        1. 只考虑综合评分 >= min_score 的策略
        2. 按综合评分降序
        3. 逐个加入，如果与已选组合重叠度 < max_overlap 才加入
        4. 宁缺毋滥：不足 top_n 时不补充低质量策略
        """
        # 质量门槛：只考虑高分策略
        qualified = [c for c in combos if c.get("综合评分", 0.0) >= min_score]
        sorted_combos = sorted(qualified, key=lambda c: c.get("综合评分", 0.0), reverse=True)

        selected: List[dict] = []
        selected_families: List[frozenset] = []

        for c in sorted_combos:
            if len(selected) >= top_n:
                break
            families = self._get_family_set(c.get("conditions", []))
            max_current_overlap = 0.0
            for sf in selected_families:
                overlap = self._family_overlap_ratio(families, sf)
                max_current_overlap = max(max_current_overlap, overlap)

            # 严格多样性：重叠度必须 < 阈值才能入选
            if max_current_overlap < max_overlap:
                selected.append(c)
                selected_families.append(families)

        # 宁缺毋滥：不再补充低质量/高重叠策略
        return selected

    def _refresh_family_table(self):
        """
        刷新精选组合表（双层策略池 + 折叠式分组）：
        层级结构：
          精品层（高质量）
            ├─ 多头趋势
            │   ├─ 做多 (6条)
            │   └─ 做空 (6条)
            ├─ 空头趋势
            └─ 震荡市
          高频层（高触发）
            ├─ 多头趋势
            ...
        
        精品层筛选：评分>=80, 多样性<30%
        高频层筛选：2-3条件, 触发>=10, 评分>=70, 多样性<50%
        """
        try:
            from core import signal_store
            combos = signal_store.get_cumulative_results(top_n=500)
        except Exception:
            return

        tree = self._family_tree
        tree.clear()

        _BG_LONG      = QtGui.QColor("#1E2D2A")
        _BG_SHORT     = QtGui.QColor("#2D1E1E")
        _BG_TIER_ELITE = QtGui.QColor("#2A2520")  # 精品层背景
        _BG_TIER_FREQ  = QtGui.QColor("#1E2A2A")  # 高频层背景

        STATES = ["多头趋势", "空头趋势", "震荡市"]
        STATE_COLORS = {
            "多头趋势": LONG_COLOR,
            "空头趋势": SHORT_COLOR,
            "震荡市":   ACCENT_GOLD,
        }
        TIER_COLORS = {
            "精品": ACCENT_GOLD,
            "高频": TIER_HIGH_FREQ,
        }
        TIER_BGS = {
            "精品": _BG_TIER_ELITE,
            "高频": _BG_TIER_FREQ,
        }
        MIN_STATE_TRIGGERS = 5   # 该状态触发次数不足时跳过

        def _create_tier_node(tier_name: str, tier_color: str, bg_color: QtGui.QColor) -> QtWidgets.QTreeWidgetItem:
            """创建层级根节点"""
            node = QtWidgets.QTreeWidgetItem([tier_name])
            node.setForeground(0, QtGui.QColor(tier_color))
            font = node.font(0)
            font.setBold(True)
            font.setPointSize(11)
            node.setFont(0, font)
            for col in range(13):
                node.setBackground(col, bg_color)
            return node

        def _create_state_node(parent: QtWidgets.QTreeWidgetItem, 
                               market_state: str, count: int) -> QtWidgets.QTreeWidgetItem:
            """创建市场状态分组节点"""
            state_color = STATE_COLORS.get(market_state, ACCENT_GOLD)
            node = QtWidgets.QTreeWidgetItem(parent, ["", market_state, "", "", f"共 {count} 条策略"])
            node.setForeground(1, QtGui.QColor(state_color))
            font = node.font(1)
            font.setBold(True)
            node.setFont(1, font)
            node.setForeground(4, QtGui.QColor(TEXT_DIM))
            return node

        def _create_direction_node(parent: QtWidgets.QTreeWidgetItem,
                                   direction: str, dir_label: str, 
                                   count: int) -> QtWidgets.QTreeWidgetItem:
            """创建方向分组节点"""
            dir_color = LONG_COLOR if direction == "long" else SHORT_COLOR
            node = QtWidgets.QTreeWidgetItem(parent, ["", "", "", dir_label, f"({count}条)"])
            node.setForeground(3, QtGui.QColor(dir_color))
            font = node.font(3)
            font.setBold(True)
            node.setFont(3, font)
            node.setForeground(4, QtGui.QColor(TEXT_DIM))
            return node

        def _add_combo_item(parent: QtWidgets.QTreeWidgetItem, tier_label: str,
                           tier_color: str, seq: int, c: dict,
                           market_state: str, direction: str):
            """添加单个策略组合到树节点"""
            conditions   = c.get("conditions", [])
            merged_count = c.get("_merged_count", 1)
            appear       = c.get("appear_rounds", 0)
            score        = c.get("综合评分", c.get("score", 0.0))
            dir_label    = "做多" if direction == "long" else "做空"
            dir_color    = LONG_COLOR if direction == "long" else SHORT_COLOR

            # 该市场状态下的命中率和触发次数
            state_rate, state_triggers = _get_state_rate(
                c.get("market_state_breakdown"), market_state)
            state_hits = round(state_triggers * state_rate)

            # 三状态命中率明细
            breakdown = c.get("market_state_breakdown") or {}
            threshold = 0.64 if direction == "long" else 0.52

            def _state_cell(st: str):
                r, t = _get_state_rate(breakdown, st)
                if t < 5:
                    return "-", TEXT_DIM
                warn = r < threshold
                text = f"{'⚠' if warn else ''}{r:.0%}({t})"
                color = WARN_COLOR if warn else _rate_color(r, direction)
                return text, color

            bull_text, bull_color = _state_cell("多头趋势")
            bear_text, bear_color = _state_cell("空头趋势")
            side_text, side_color = _state_cell("震荡市")

            cond_text = _format_conditions(conditions, direction)
            if merged_count > 1:
                cond_text = f"[合并{merged_count}] {cond_text}"

            appear_color = (ACCENT_GOLD if appear >= 5
                           else TEXT_PRIMARY if appear >= 3
                           else TEXT_DIM)
            state_color_val = STATE_COLORS.get(market_state, ACCENT_GOLD)

            item = QtWidgets.QTreeWidgetItem(parent, [
                tier_label,                           # 0: 层级
                market_state,                         # 1: 市场状态
                str(seq),                             # 2: 序号
                dir_label,                            # 3: 方向
                cond_text,                            # 4: 开仓条件组合
                str(appear),                          # 5: 出现轮次
                str(state_triggers),                  # 6: 状态触发
                str(state_hits),                      # 7: 状态命中
                f"{state_rate:.1%}",                  # 8: 状态命中率
                f"{score:.1f}",                       # 9: 综合评分
                bull_text,                            # 10: 多头命中率
                bear_text,                            # 11: 空头命中率
                side_text,                            # 12: 震荡命中率
            ])

            # 设置颜色
            item.setForeground(0, QtGui.QColor(tier_color))
            item.setForeground(1, QtGui.QColor(state_color_val))
            item.setForeground(2, QtGui.QColor(TEXT_DIM))
            item.setForeground(3, QtGui.QColor(dir_color))
            item.setForeground(4, QtGui.QColor(TEXT_PRIMARY))
            item.setForeground(5, QtGui.QColor(appear_color))
            item.setForeground(6, QtGui.QColor(TEXT_PRIMARY))
            item.setForeground(7, QtGui.QColor(TEXT_PRIMARY))
            item.setForeground(8, QtGui.QColor(_rate_color(state_rate, direction)))
            item.setForeground(9, QtGui.QColor(TIER_ELITE if score >= 80 else ACCENT_GOLD))
            item.setForeground(10, QtGui.QColor(bull_color))
            item.setForeground(11, QtGui.QColor(bear_color))
            item.setForeground(12, QtGui.QColor(side_color))

            # 设置加粗
            for col in [0, 3, 8, 9]:
                font = item.font(col)
                font.setBold(True)
                item.setFont(col, font)
            if appear >= 5:
                font = item.font(5)
                font.setBold(True)
                item.setFont(5, font)

            # 设置行背景色
            row_bg = _BG_LONG if direction == "long" else _BG_SHORT
            for col in range(13):
                item.setBackground(col, row_bg)

        def _get_tier_combos(all_combos: List[dict], market_state: str, direction: str,
                            tier: str, elite_keys: set) -> List[dict]:
            """获取指定层级的策略组合"""
            # 过滤：该状态触发次数 >= MIN_STATE_TRIGGERS
            candidates = [
                c for c in all_combos
                if c.get("direction") == direction
                and _get_state_rate(
                    c.get("market_state_breakdown"), market_state)[1] >= MIN_STATE_TRIGGERS
            ]

            if not candidates:
                return []

            # 智能合并（宽松取值）
            merged = self._merge_similar_combos(candidates)

            # 按该状态命中率降序（主排序键），综合评分为次排序键
            merged.sort(
                key=lambda c: (
                    _get_state_rate(c.get("market_state_breakdown"), market_state)[0],
                    c.get("综合评分", 0.0)
                ),
                reverse=True
            )

            if tier == "精品":
                # 精品层：评分>=80, 多样性<30%
                return self._select_diverse_top(merged, top_n=6, min_score=80.0, max_overlap=0.3)
            elif tier == "高频":
                high_freq_candidates = [
                    c for c in merged
                    if frozenset(c.get("conditions", [])) not in elite_keys
                ]
                if market_state == "震荡市":
                    # 震荡市：从完整 merged 中找（包括已在精品层的 2-3 条件组合）
                    sideways_pool = [dict(c) for c in merged]
                    for c in sideways_pool:
                        r, t = _get_state_rate(c.get("market_state_breakdown"), market_state)
                        c["state_rate"] = r
                        c["state_triggers"] = t
                    _min_state_rate = 0.64 if direction == "long" else 0.52
                    return self._select_high_freq_top(sideways_pool, top_n=10,
                                                      min_triggers=5, min_score=65.0,
                                                      max_conditions=3, max_overlap=0.5,
                                                      min_state_rate=_min_state_rate)
                else:
                    return self._select_high_freq_top(high_freq_candidates, top_n=6,
                                                      min_triggers=10, min_score=70.0,
                                                      max_conditions=3, max_overlap=0.5)

        TIERS = [
            ("精品", "精品层（高质量：评分≥80）",               ACCENT_GOLD,    _BG_TIER_ELITE),
            ("高频", "高频层（触发优先；震荡市扩容至10个）",      TIER_HIGH_FREQ, _BG_TIER_FREQ),
        ]

        # 先收集精品层的组合key，用于高频层去重
        all_elite_keys: set = set()
        for market_state in STATES:
            for direction, _ in [("long", "做多"), ("short", "做空")]:
                elite_combos = _get_tier_combos(combos, market_state, direction, "精品", set())
                for ec in elite_combos:
                    all_elite_keys.add(frozenset(ec.get("conditions", [])))

        for tier_key, tier_name, tier_color, tier_bg in TIERS:
            tier_node = _create_tier_node(tier_name, tier_color, tier_bg)
            tier_total = 0

            for market_state in STATES:
                state_combos_by_dir = {}
                for direction, dir_label in [("long", "做多"), ("short", "做空")]:
                    elite_keys_for_tier = all_elite_keys if tier_key == "高频" else set()
                    top_combos = _get_tier_combos(combos, market_state, direction, tier_key, elite_keys_for_tier)
                    if top_combos:
                        state_combos_by_dir[direction] = (dir_label, top_combos)
                        tier_total += len(top_combos)

                if not state_combos_by_dir:
                    continue

                # 计算该状态总策略数
                state_count = sum(len(v[1]) for v in state_combos_by_dir.values())
                state_node = _create_state_node(tier_node, market_state, state_count)

                for direction, (dir_label, top_combos) in state_combos_by_dir.items():
                    dir_node = _create_direction_node(state_node, direction, dir_label, len(top_combos))
                    
                    for seq, c in enumerate(top_combos, start=1):
                        _add_combo_item(dir_node, tier_key, tier_color, seq, c,
                                       market_state, direction)

            # 更新层级节点显示总数
            tier_node.setText(0, f"{tier_name} - 共 {tier_total} 条")
            tree.addTopLevelItem(tier_node)

        # 默认展开两层
        for i in range(tree.topLevelItemCount()):
            tier_node = tree.topLevelItem(i)
            tier_node.setExpanded(True)
            # 展开所有状态节点
            for j in range(tier_node.childCount()):
                tier_node.child(j).setExpanded(True)

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
