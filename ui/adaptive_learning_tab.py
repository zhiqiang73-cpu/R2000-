"""
自适应学习标签页 - 盈亏驱动版
核心逻辑：每笔交易盈亏 → 分析原因 → 自动调整参数 → 提升盈利
重构：统一卡片风格 + 记忆时间进度 + 从状态文件刷新
"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from PyQt6 import QtWidgets, QtCore, QtGui

from config import UI_CONFIG, PAPER_TRADING_CONFIG, SIMILARITY_CONFIG


# ═══════════════════════════════════════════════════════════
# 通用卡片组件 AdaptiveLearningCard（参考 Entry Overview 风格）
# ═══════════════════════════════════════════════════════════

class AdaptiveLearningCard(QtWidgets.QFrame):
    """
    自适应学习卡片 - 统一风格
    标题栏（渐变 + 左侧色条）+ 参数表格 + 底部最近调整
    """
    def __init__(
        self,
        title: str,
        icon: str,
        accent_color: str,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._icon = icon
        self._accent_color = accent_color
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("adaptiveLearningCard")
        self.setStyleSheet("""
            QFrame#adaptiveLearningCard {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 8px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(0)

        # 标题栏：渐变 + 左侧 3px 色条
        header = QtWidgets.QWidget()
        header.setObjectName("cardHeader")
        # 将 #rrggbb 转为 rgba 用于渐变
        r, g, b = int(self._accent_color[1:3], 16), int(self._accent_color[3:5], 16), int(self._accent_color[5:7], 16)
        header.setStyleSheet(f"""
            QWidget#cardHeader {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba({r},{g},{b}, 0.25), stop:1 #2d2d2d);
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-left: 3px solid {self._accent_color};
            }}
        """)
        header_h = QtWidgets.QHBoxLayout(header)
        header_h.setContentsMargins(10, 6, 10, 6)
        header_h.setSpacing(8)
        title_lbl = QtWidgets.QLabel(f"{self._icon} {self._title}")
        title_lbl.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 12px; background: transparent;")
        header_h.addWidget(title_lbl)
        header_h.addStretch()
        self._sample_label = QtWidgets.QLabel("")
        self._sample_label.setStyleSheet("color: #888; font-size: 10px; background: transparent;")
        header_h.addWidget(self._sample_label)
        layout.addWidget(header)

        # 分隔线
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #555; border: none;")
        layout.addWidget(sep)

        # 表格区域：参数 | 当前值 | 调整范围 | 状态
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["参数", "当前值", "调整范围", "状态"])
        self._table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #999;
                padding: 4px 6px;
                border: none;
                border-bottom: 1px solid #555;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                border: none;
                color: #ddd;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QTableWidget::item:alternate {
                background-color: rgba(58, 58, 58, 0.5);
            }
        """)
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.setMinimumHeight(60)
        layout.addWidget(self._table)

        # 底部：最近调整
        footer = QtWidgets.QWidget()
        footer.setStyleSheet("background: transparent;")
        footer_h = QtWidgets.QHBoxLayout(footer)
        footer_h.setContentsMargins(10, 4, 10, 6)
        self._last_adjust_label = QtWidgets.QLabel("最近调整: -")
        self._last_adjust_label.setStyleSheet("color: #888; font-size: 10px;")
        footer_h.addWidget(self._last_adjust_label)
        footer_h.addStretch()
        layout.addWidget(footer)

    def set_sample_count(self, text: str):
        """设置样本数显示，如 '样本: 25笔'"""
        self._sample_label.setText(text)

    def set_content(
        self,
        rows: List[Tuple[str, str, str, str]],
        last_adjustment: str = "",
    ):
        """
        设置表格行：(参数名, 当前值, 调整范围, 状态徽章)
        状态建议: ✓ 已学习 / ≈ 学习中 / -- 未学习
        """
        self._table.setRowCount(0)
        for i, (param, current, range_txt, status) in enumerate(rows):
            row = self._table.rowCount()
            self._table.insertRow(row)
            for col, text in enumerate([param, current, range_txt, status]):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if col == 3:
                    if "✓" in status or "已学习" in status:
                        item.setForeground(QtGui.QColor("#4CAF50"))
                    elif "≈" in status or "学习中" in status:
                        item.setForeground(QtGui.QColor("#FFA726"))
                    else:
                        item.setForeground(QtGui.QColor("#888"))
                self._table.setItem(row, col, item)
        self._last_adjust_label.setText(f"最近调整: {last_adjustment}" if last_adjustment else "最近调整: -")


# ═══════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """交易记录（简化版，用于盈亏分析）"""
    order_id: str
    profit_usdt: float       # 盈亏金额 (USDT)
    profit_pct: float        # 盈亏百分比 (%)
    close_reason: str        # 平仓原因
    direction: str           # 方向 (LONG/SHORT)
    hold_bars: int           # 持仓K线数
    peak_profit_pct: float   # 峰值利润 (%)
    entry_time: datetime
    exit_time: datetime


@dataclass
class CloseReasonStats:
    """平仓原因统计"""
    reason: str              # 原因名称
    count: int = 0           # 笔数
    win_count: int = 0       # 盈利笔数
    total_pnl: float = 0.0   # 总盈亏 (USDT)
    total_pnl_pct: float = 0.0  # 总盈亏 (%)
    avg_hold_bars: float = 0.0  # 平均持仓K线数
    avg_peak_loss: float = 0.0  # 平均峰值流失 (%)
    suggestion: str = ""     # 调整建议


@dataclass
class AdaptationResult:
    """自适应调整结果"""
    parameter: str           # 参数名称
    old_value: float         # 原值
    new_value: float         # 新值
    reason: str              # 调整原因
    timestamp: datetime = None


# ═══════════════════════════════════════════════════════════
# 兼容层：虚拟对象（保持与旧代码的兼容性）
# ═══════════════════════════════════════════════════════════

class _DummySignal(QtCore.QObject):
    """虚拟信号对象"""
    adjustment_confirmed = QtCore.pyqtSignal(str, float)


class _DummyTrackerCard:
    """虚拟追踪器卡片（兼容旧接口）"""
    def __init__(self):
        self._signal_obj = _DummySignal()
        self.adjustment_confirmed = self._signal_obj.adjustment_confirmed

    def update_records(self, records): pass
    def update_scores(self, scores): pass
    def set_suggestions(self, suggestions): pass
    def update_rejections(self, rejections): pass
    def update_gate_scores(self, scores): pass
    def _on_suggest_clicked(self): pass


class _DummyTradeTimeline:
    """虚拟交易时间线（兼容旧接口）"""
    def add_trade(self, order, deepseek_review=None): pass
    def update_deepseek_review(self, order_id, review): pass
    def clear_trades(self): pass


# ═══════════════════════════════════════════════════════════
# UI组件：盈亏总览卡片
# ═══════════════════════════════════════════════════════════

class ProfitSummaryCard(QtWidgets.QFrame):
    """
    盈亏总览卡片
    展示：累计盈亏、今日盈亏、胜率、盈亏比
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                border: 1px solid #3a3a5a;
                border-radius: 10px;
            }
        """)
        self.setFixedHeight(120)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        
        # 标题行
        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("💰 盈亏总览")
        title.setStyleSheet("color: #fff; font-size: 14px; font-weight: bold;")
        title_row.addWidget(title)
        title_row.addStretch()
        
        # 交易数统计
        self._trade_count_label = QtWidgets.QLabel("总交易: 0 笔")
        self._trade_count_label.setStyleSheet("color: #888; font-size: 11px;")
        title_row.addWidget(self._trade_count_label)
        
        layout.addLayout(title_row)
        
        # 指标行
        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.setSpacing(20)
        
        # 累计盈亏
        self._total_pnl_widget = self._create_metric_widget("累计盈亏", "$0.00", "#4FC3F7")
        metrics_row.addWidget(self._total_pnl_widget)
        
        # 今日盈亏
        self._today_pnl_widget = self._create_metric_widget("今日盈亏", "$0.00", "#81C784")
        metrics_row.addWidget(self._today_pnl_widget)
        
        # 胜率
        self._winrate_widget = self._create_metric_widget("胜率", "0%", "#FFA726")
        metrics_row.addWidget(self._winrate_widget)
        
        # 盈亏比
        self._profit_factor_widget = self._create_metric_widget("盈亏比", "0:1", "#BA68C8")
        metrics_row.addWidget(self._profit_factor_widget)
        
        # 连胜/连亏
        self._streak_widget = self._create_metric_widget("连续", "-", "#90A4AE")
        metrics_row.addWidget(self._streak_widget)
        
        metrics_row.addStretch()
        layout.addLayout(metrics_row)
    
    def _create_metric_widget(self, label: str, value: str, color: str) -> QtWidgets.QWidget:
        """创建单个指标小部件"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # 数值
        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        value_label.setObjectName("value")
        layout.addWidget(value_label)
        
        # 标签
        name_label = QtWidgets.QLabel(label)
        name_label.setStyleSheet("color: #888; font-size: 10px;")
        name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)
        
        widget.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
            }
        """)
        widget.setMinimumWidth(90)
        
        return widget
    
    def _update_metric(self, widget: QtWidgets.QWidget, value: str, color: str = None):
        """更新指标值"""
        value_label = widget.findChild(QtWidgets.QLabel, "value")
        if value_label:
            value_label.setText(value)
            if color:
                value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
    
    def update_summary(self, total_pnl: float, today_pnl: float, 
                       win_rate: float, profit_factor: float,
                       total_trades: int, consecutive_wins: int, consecutive_losses: int):
        """更新盈亏总览"""
        # 累计盈亏
        pnl_color = "#4CAF50" if total_pnl >= 0 else "#F44336"
        self._update_metric(self._total_pnl_widget, f"${total_pnl:+,.2f}", pnl_color)
        
        # 今日盈亏
        today_color = "#4CAF50" if today_pnl >= 0 else "#F44336"
        self._update_metric(self._today_pnl_widget, f"${today_pnl:+,.2f}", today_color)
        
        # 胜率
        wr_color = "#4CAF50" if win_rate >= 50 else "#FFA726" if win_rate >= 40 else "#F44336"
        self._update_metric(self._winrate_widget, f"{win_rate:.0f}%", wr_color)
        
        # 盈亏比
        pf_color = "#4CAF50" if profit_factor >= 1.5 else "#FFA726" if profit_factor >= 1.0 else "#F44336"
        self._update_metric(self._profit_factor_widget, f"{profit_factor:.1f}:1", pf_color)
        
        # 连胜/连亏
        if consecutive_wins > 0:
            self._update_metric(self._streak_widget, f"🔥 {consecutive_wins}连胜", "#4CAF50")
        elif consecutive_losses > 0:
            self._update_metric(self._streak_widget, f"❄️ {consecutive_losses}连亏", "#F44336")
        else:
            self._update_metric(self._streak_widget, "-", "#888")
        
        # 总交易数
        self._trade_count_label.setText(f"总交易: {total_trades} 笔")


# ═══════════════════════════════════════════════════════════
# UI组件：盈亏分析表格
# ═══════════════════════════════════════════════════════════

class TradeAnalysisTable(QtWidgets.QFrame):
    """
    盈亏分析表格
    按平仓原因分组统计：止损/止盈/追踪/超时
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stats: Dict[str, CloseReasonStats] = {}
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 10px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        
        # 标题行
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("📊 盈亏分析")
        title.setStyleSheet("color: #4FC3F7; font-size: 14px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        
        self._period_label = QtWidgets.QLabel("最近 20 笔")
        self._period_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self._period_label)
        
        layout.addLayout(header)
        
        # 表格
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "平仓原因", "笔数", "盈利笔数", "总盈亏", "平均流失", "建议"
        ])
        self._table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #252526;
                color: #aaa;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #3a3a3a;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                border: none;
                color: #ddd;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 10px 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QTableWidget::item:alternate {
                background-color: #232323;
            }
            QTableWidget::item:selected {
                background-color: #333;
            }
        """)
        
        # 列宽设置
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 100)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 60)
        self._table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 80)
        self._table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, 100)
        self._table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(4, 80)
        self._table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setMinimumHeight(180)
        
        layout.addWidget(self._table)
        
        # 空状态提示
        self._empty_hint = QtWidgets.QLabel("暂无交易数据\n开始模拟交易后将自动分析盈亏原因")
        self._empty_hint.setStyleSheet("color: #666; font-size: 11px;")
        self._empty_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_hint)
        
        self._refresh_empty_state()
    
    def _refresh_empty_state(self):
        """刷新空状态显示"""
        has_data = self._table.rowCount() > 0
        self._table.setVisible(has_data)
        self._empty_hint.setVisible(not has_data)
    
    def update_analysis(self, stats_list: List[CloseReasonStats], trade_count: int = 20):
        """更新盈亏分析"""
        self._period_label.setText(f"最近 {trade_count} 笔")
        self._table.setRowCount(0)
        
        # 原因图标映射
        reason_icons = {
            "止损": "🔻",
            "止盈": "🎯",
            "追踪止损": "📈",
            "保本止损": "🛡️",
            "超时离场": "⏰",
            "脱轨": "⚠️",
            "手动平仓": "✋",
            "位置翻转": "🔄",
        }
        
        for stats in stats_list:
            row = self._table.rowCount()
            self._table.insertRow(row)
            
            # 原因（带图标）
            icon = reason_icons.get(stats.reason, "📋")
            reason_item = QtWidgets.QTableWidgetItem(f"{icon} {stats.reason}")
            reason_item.setFlags(reason_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, reason_item)
            
            # 笔数
            count_item = QtWidgets.QTableWidgetItem(str(stats.count))
            count_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            count_item.setFlags(count_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, count_item)
            
            # 盈利笔数
            win_item = QtWidgets.QTableWidgetItem(str(stats.win_count))
            win_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            win_item.setFlags(win_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            win_rate = (stats.win_count / stats.count * 100) if stats.count > 0 else 0
            if win_rate >= 60:
                win_item.setForeground(QtGui.QColor("#4CAF50"))
            elif win_rate < 30:
                win_item.setForeground(QtGui.QColor("#F44336"))
            self._table.setItem(row, 2, win_item)
            
            # 总盈亏
            pnl_item = QtWidgets.QTableWidgetItem(f"${stats.total_pnl:+,.2f}")
            pnl_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            pnl_item.setFlags(pnl_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            if stats.total_pnl >= 0:
                pnl_item.setForeground(QtGui.QColor("#4CAF50"))
            else:
                pnl_item.setForeground(QtGui.QColor("#F44336"))
            self._table.setItem(row, 3, pnl_item)
            
            # 平均流失（峰值利润 vs 实际利润）
            if stats.avg_peak_loss > 0:
                loss_item = QtWidgets.QTableWidgetItem(f"-{stats.avg_peak_loss:.1f}%")
                loss_item.setForeground(QtGui.QColor("#FFA726"))
            else:
                loss_item = QtWidgets.QTableWidgetItem("-")
                loss_item.setForeground(QtGui.QColor("#666"))
            loss_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            loss_item.setFlags(loss_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 4, loss_item)
            
            # 建议
            suggestion_item = QtWidgets.QTableWidgetItem(stats.suggestion)
            suggestion_item.setFlags(suggestion_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            if "保持" in stats.suggestion:
                suggestion_item.setForeground(QtGui.QColor("#888"))
            elif "放宽" in stats.suggestion or "提前" in stats.suggestion:
                suggestion_item.setForeground(QtGui.QColor("#4FC3F7"))
            elif "收紧" in stats.suggestion or "缩短" in stats.suggestion:
                suggestion_item.setForeground(QtGui.QColor("#FFA726"))
            self._table.setItem(row, 5, suggestion_item)
        
        self._refresh_empty_state()
    
    def clear(self):
        """清空数据"""
        self._table.setRowCount(0)
        self._refresh_empty_state()


# ═══════════════════════════════════════════════════════════
# UI组件：自适应调整结果表格
# ═══════════════════════════════════════════════════════════

class AdaptationResultTable(QtWidgets.QFrame):
    """
    自适应调整结果表格
    展示：参数、原值、新值、调整原因
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[AdaptationResult] = []
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 10px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        
        # 标题行
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("🔧 自适应调整结果")
        title.setStyleSheet("color: #81C784; font-size: 14px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        
        self._adjustment_count_label = QtWidgets.QLabel("0 项调整")
        self._adjustment_count_label.setStyleSheet("color: #888; font-size: 11px;")
        header.addWidget(self._adjustment_count_label)
        
        layout.addLayout(header)
        
        # 表格
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["参数", "原值", "新值", "调整原因"])
        self._table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #252526;
                color: #aaa;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #3a3a3a;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                border: none;
                color: #ddd;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 10px 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QTableWidget::item:alternate {
                background-color: #232323;
            }
            QTableWidget::item:selected {
                background-color: #333;
            }
        """)
        
        # 列宽设置
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 100)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(1, 90)
        self._table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 90)
        self._table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setMinimumHeight(140)
        self._table.setMaximumHeight(180)
        
        layout.addWidget(self._table)
        
        # 空状态提示
        self._empty_hint = QtWidgets.QLabel("暂无参数调整\n系统将根据盈亏数据自动优化参数")
        self._empty_hint.setStyleSheet("color: #666; font-size: 11px;")
        self._empty_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_hint)
        
        self._refresh_empty_state()
    
    def _refresh_empty_state(self):
        """刷新空状态显示"""
        has_data = self._table.rowCount() > 0
        self._table.setVisible(has_data)
        self._empty_hint.setVisible(not has_data)
    
    def update_results(self, results: List[AdaptationResult]):
        """更新调整结果"""
        self._results = results
        self._table.setRowCount(0)
        
        # 参数显示名称映射
        param_display = {
            "STOP_LOSS_ATR": "止损距离",
            "TAKE_PROFIT_ATR": "止盈距离",
            "TRAILING_STAGE1_PCT": "追踪启动",
            "FUSION_THRESHOLD": "匹配阈值",
            "ENTRY_COOLDOWN_SEC": "开仓冷却",
            "MIN_RR_RATIO": "盈亏比",
            "MAX_HOLD_BARS": "最大持仓",
            "KELLY_FRACTION": "凯利系数",
        }
        
        # 参数单位映射
        param_units = {
            "STOP_LOSS_ATR": "×ATR",
            "TAKE_PROFIT_ATR": "×ATR",
            "TRAILING_STAGE1_PCT": "%",
            "FUSION_THRESHOLD": "",
            "ENTRY_COOLDOWN_SEC": "秒",
            "MIN_RR_RATIO": "",
            "MAX_HOLD_BARS": "根",
            "KELLY_FRACTION": "",
        }
        
        for result in results:
            row = self._table.rowCount()
            self._table.insertRow(row)
            
            # 参数名
            display_name = param_display.get(result.parameter, result.parameter)
            param_item = QtWidgets.QTableWidgetItem(display_name)
            param_item.setFlags(param_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            param_item.setForeground(QtGui.QColor("#4FC3F7"))
            self._table.setItem(row, 0, param_item)
            
            # 原值
            unit = param_units.get(result.parameter, "")
            old_item = QtWidgets.QTableWidgetItem(f"{result.old_value:.2f}{unit}")
            old_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            old_item.setFlags(old_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            old_item.setForeground(QtGui.QColor("#888"))
            self._table.setItem(row, 1, old_item)
            
            # 新值（带变化指示）
            change = result.new_value - result.old_value
            arrow = "↑" if change > 0 else "↓" if change < 0 else "―"
            new_text = f"{result.new_value:.2f}{unit} {arrow}"
            new_item = QtWidgets.QTableWidgetItem(new_text)
            new_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            new_item.setFlags(new_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            if change > 0:
                new_item.setForeground(QtGui.QColor("#4CAF50"))
            elif change < 0:
                new_item.setForeground(QtGui.QColor("#F44336"))
            else:
                new_item.setForeground(QtGui.QColor("#888"))
            self._table.setItem(row, 2, new_item)
            
            # 原因
            reason_item = QtWidgets.QTableWidgetItem(result.reason)
            reason_item.setFlags(reason_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 3, reason_item)
        
        self._adjustment_count_label.setText(f"{len(results)} 项调整")
        self._refresh_empty_state()
    
    def clear(self):
        """清空数据"""
        self._results = []
        self._table.setRowCount(0)
        self._adjustment_count_label.setText("0 项调整")
        self._refresh_empty_state()


# ═══════════════════════════════════════════════════════════
# UI组件：调整效果追踪
# ═══════════════════════════════════════════════════════════

class EffectTrackingPanel(QtWidgets.QFrame):
    """
    调整效果追踪面板
    展示：调整前后的表现对比
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
            }
        """)
        self.setFixedHeight(70)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(20)
        
        # 标题
        title = QtWidgets.QLabel("📈 调整效果追踪")
        title.setStyleSheet("color: #FFA726; font-size: 12px; font-weight: bold;")
        layout.addWidget(title)
        
        # 分隔线
        sep = QtWidgets.QFrame()
        sep.setStyleSheet("background-color: #444;")
        sep.setFixedWidth(1)
        sep.setFixedHeight(30)
        layout.addWidget(sep)
        
        # 胜率变化
        self._winrate_change = QtWidgets.QLabel("胜率: -- → --")
        self._winrate_change.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._winrate_change)
        
        # 平均盈利变化
        self._avg_profit_change = QtWidgets.QLabel("平均盈利: -- → --")
        self._avg_profit_change.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._avg_profit_change)
        
        # 盈亏比变化
        self._pf_change = QtWidgets.QLabel("盈亏比: -- → --")
        self._pf_change.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._pf_change)
        
        layout.addStretch()
        
        # 上次调整时间
        self._last_adjustment = QtWidgets.QLabel("上次调整: -")
        self._last_adjustment.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._last_adjustment)
    
    def update_effect(self, 
                      old_winrate: float, new_winrate: float,
                      old_avg_profit: float, new_avg_profit: float,
                      old_pf: float, new_pf: float,
                      last_adjustment_time: datetime = None):
        """更新调整效果"""
        # 胜率
        wr_arrow = "↑" if new_winrate > old_winrate else "↓" if new_winrate < old_winrate else "―"
        wr_color = "#4CAF50" if new_winrate > old_winrate else "#F44336" if new_winrate < old_winrate else "#888"
        self._winrate_change.setText(f"胜率: {old_winrate:.0f}% → {new_winrate:.0f}% {wr_arrow}")
        self._winrate_change.setStyleSheet(f"color: {wr_color}; font-size: 11px;")
        
        # 平均盈利
        ap_arrow = "↑" if new_avg_profit > old_avg_profit else "↓" if new_avg_profit < old_avg_profit else "―"
        ap_color = "#4CAF50" if new_avg_profit > old_avg_profit else "#F44336" if new_avg_profit < old_avg_profit else "#888"
        self._avg_profit_change.setText(f"平均盈利: ${old_avg_profit:.1f} → ${new_avg_profit:.1f} {ap_arrow}")
        self._avg_profit_change.setStyleSheet(f"color: {ap_color}; font-size: 11px;")
        
        # 盈亏比
        pf_arrow = "↑" if new_pf > old_pf else "↓" if new_pf < old_pf else "―"
        pf_color = "#4CAF50" if new_pf > old_pf else "#F44336" if new_pf < old_pf else "#888"
        self._pf_change.setText(f"盈亏比: {old_pf:.1f}:1 → {new_pf:.1f}:1 {pf_arrow}")
        self._pf_change.setStyleSheet(f"color: {pf_color}; font-size: 11px;")
        
        # 上次调整时间
        if last_adjustment_time:
            time_str = last_adjustment_time.strftime("%m-%d %H:%M")
            self._last_adjustment.setText(f"上次调整: {time_str}")
        else:
            self._last_adjustment.setText("上次调整: -")


# ═══════════════════════════════════════════════════════════
# UI组件：冷启动面板
# ═══════════════════════════════════════════════════════════

class ColdStartPanel(QtWidgets.QFrame):
    """冷启动系统面板"""
    cold_start_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_enabled = False
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)
        self.setFixedHeight(45)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # 标题
        title = QtWidgets.QLabel("🚀 冷启动模式")
        title.setStyleSheet("color: #fff; font-size: 12px; font-weight: bold;")
        layout.addWidget(title)

        # 开关
        self._toggle = QtWidgets.QCheckBox("启用")
        self._toggle.setStyleSheet("""
            QCheckBox { color: #aaa; font-size: 11px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:unchecked { border: 1px solid #666; background-color: #333; border-radius: 3px; }
            QCheckBox::indicator:checked { border: 1px solid #007acc; background-color: #007acc; border-radius: 3px; }
        """)
        self._toggle.stateChanged.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        # 状态
        self._status = QtWidgets.QLabel("已关闭")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status)

        layout.addStretch()

        # 门槛显示
        threshold_label = QtWidgets.QLabel("当前门槛:")
        threshold_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(threshold_label)

        self._thresholds_display = QtWidgets.QLabel("融合0.65 | 余弦0.70")
        self._thresholds_display.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._thresholds_display)

    def _on_toggle(self, state):
        self._is_enabled = state == QtCore.Qt.CheckState.Checked.value
        self._status.setText("已启用" if self._is_enabled else "已关闭")
        self._status.setStyleSheet(f"color: {'#4CAF50' if self._is_enabled else '#888'}; font-size: 11px;")
        self.cold_start_toggled.emit(self._is_enabled)

    def set_enabled(self, enabled: bool):
        self._toggle.blockSignals(True)
        self._toggle.setChecked(enabled)
        self._is_enabled = enabled
        self._status.setText("已启用" if enabled else "已关闭")
        self._status.setStyleSheet(f"color: {'#4CAF50' if enabled else '#888'}; font-size: 11px;")
        self._toggle.blockSignals(False)

    def is_enabled(self) -> bool:
        return self._is_enabled

    def update_thresholds(self, fusion: float, cosine: float, euclidean: float = 0, dtw: float = 0, **kwargs):
        self._thresholds_display.setText(f"融合{fusion:.2f} | 余弦{cosine:.2f}")


# ═══════════════════════════════════════════════════════════
# 主标签页
# ═══════════════════════════════════════════════════════════

# ── 状态文件路径（与 live_trading_engine 一致）──
ADAPTIVE_STATE_FILES = {
    "kelly": "data/adaptive_controller_state.json",
    "bayesian": "data/bayesian_state.json",
    "tpsl": "data/tpsl_tracker_state.json",
    "rejection": "data/rejection_tracker_state.json",
    "exit_timing": "data/exit_timing_state.json",
    "near_miss": "data/near_miss_tracker_state.json",
}


class AdaptiveLearningTab(QtWidgets.QWidget):
    """
    自适应学习标签页 - 统一卡片版
    
    布局：
    - 顶部：记忆时间进度条 + [刷新] [清除]
    - 中部：6 个统一风格卡片（2x3 网格）
    - 底部：冷启动面板
    """
    clear_memory_requested = QtCore.pyqtSignal()  # 请求清除记忆（由主窗口处理）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._trade_records: List[TradeRecord] = []
        self._adaptation_results: List[AdaptationResult] = []
        self._data_dir: str = "data"  # 状态文件所在目录（可配置）
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # ═══ 顶部：记忆时间进度 + 操作 ═══
        top_bar = QtWidgets.QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 8px;
            }
        """)
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(12)

        title_lbl = QtWidgets.QLabel("📊 自适应学习状态")
        title_lbl.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 13px;")
        top_layout.addWidget(title_lbl)

        self._time_range_label = QtWidgets.QLabel("记忆时间: -")
        self._time_range_label.setStyleSheet("color: #aaa; font-size: 11px;")
        top_layout.addWidget(self._time_range_label)

        top_layout.addStretch()

        self._refresh_btn = QtWidgets.QPushButton("刷新")
        self._refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #ddd;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #555; }
        """)
        self._refresh_btn.clicked.connect(self.refresh_from_state_files)
        top_layout.addWidget(self._refresh_btn)

        self._clear_btn = QtWidgets.QPushButton("清除")
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a3a3a;
                color: #ffaaaa;
                border: 1px solid #664444;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #6a4a4a; }
        """)
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        top_layout.addWidget(self._clear_btn)

        root_layout.addWidget(top_bar)

        # ═══ 6 个统一风格卡片（2x3 网格）═══
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)

        self._card_kelly = AdaptiveLearningCard("凯利仓位学习", "💹", "#4CAF50")
        self._card_bayesian = AdaptiveLearningCard("贝叶斯胜率学习", "🎯", "#2196F3")
        self._card_tpsl = AdaptiveLearningCard("TP/SL距离学习", "📉", "#FF9800")
        self._card_rejection = AdaptiveLearningCard("门控拦截追踪", "🚫", "#F44336")
        self._card_exit_timing = AdaptiveLearningCard("出场时机学习", "⏱", "#9C27B0")
        self._card_near_miss = AdaptiveLearningCard("近似信号追踪", "🔍", "#607D8B")

        grid.addWidget(self._card_kelly, 0, 0)
        grid.addWidget(self._card_bayesian, 0, 1)
        grid.addWidget(self._card_tpsl, 0, 2)
        grid.addWidget(self._card_rejection, 1, 0)
        grid.addWidget(self._card_exit_timing, 1, 1)
        grid.addWidget(self._card_near_miss, 1, 2)

        root_layout.addLayout(grid, 1)

        # ═══ 底部：冷启动面板 ═══
        self.cold_start_panel = ColdStartPanel()
        root_layout.addWidget(self.cold_start_panel)

        # 首次从状态文件刷新
        QtCore.QTimer.singleShot(100, self.refresh_from_state_files)

    def _on_clear_clicked(self):
        """清除记忆：弹窗确认后发送信号，由主窗口执行实际清除"""
        reply = QtWidgets.QMessageBox.question(
            self,
            "清除学习记忆",
            "确定要清除所有自适应学习状态吗？将删除各 tracker 的状态文件，学习数据需重新积累。",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.clear_memory_requested.emit()
            self.refresh_from_state_files()

    def _load_state_file(self, key: str) -> Optional[Dict]:
        """读取单个状态文件 JSON，key 为 ADAPTIVE_STATE_FILES 的键。"""
        path = ADAPTIVE_STATE_FILES.get(key, "")
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def refresh_from_state_files(self):
        """
        从各状态文件读取 created_at / last_save_time 及内容，更新顶部时间进度与 6 张卡片。
        """
        all_created: List[float] = []
        all_last_save: List[float] = []

        # ── 凯利仓位（adaptive_controller_state.json）──
        data = self._load_state_file("kelly")
        if data:
            created = data.get("created_at", 0)
            last_save = data.get("last_save_time", 0)
            if isinstance(created, (int, float)) and created > 0:
                all_created.append(float(created))
            if isinstance(last_save, (int, float)) and last_save > 0:
                all_last_save.append(float(last_save))
            kelly_adapter = data.get("kelly_adapter") or {}
            kelly_fraction = kelly_adapter.get("kelly_fraction") or PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
            kelly_max = kelly_adapter.get("kelly_max")
            kelly_min = kelly_adapter.get("kelly_min")
            if kelly_max is None:
                kelly_max = PAPER_TRADING_CONFIG.get("KELLY_MAX_POSITION", 0.8)
            if kelly_min is None:
                kelly_min = PAPER_TRADING_CONFIG.get("KELLY_MIN_POSITION", 0.1)
            # 显示为百分比（若存的是 0~1）
            kelly_max_pct = int(kelly_max * 100) if isinstance(kelly_max, (int, float)) and 0 < kelly_max <= 1 else int(kelly_max)
            kelly_min_pct = int(kelly_min * 100) if isinstance(kelly_min, (int, float)) and 0 < kelly_min <= 1 else int(kelly_min)
            hist = kelly_adapter.get("adjustment_history") or data.get("parameter_history") or []
            sample_count = len(hist) if isinstance(hist, list) else 0
            last_adj = ""
            for h in reversed((hist[:10] if isinstance(hist, list) else [])):
                if isinstance(h, dict) and ("KELLY" in str(h.get("parameter", "")) or "kelly" in str(h.get("parameter", "")).lower()):
                    last_adj = f"{h.get('parameter', '')} {h.get('old_value', '')}→{h.get('new_value', '')}"
                    break
            self._card_kelly.set_sample_count(f"样本: {sample_count}笔" if sample_count else "")
            self._card_kelly.set_content([
                ("KELLY_FRACTION", f"{kelly_fraction:.2f}", "0.25~1.0", "✓ 已学习" if sample_count else "-- 未学习"),
                ("KELLY_MAX", f"{kelly_max_pct}%", "50%~90%", "✓ 已学习" if sample_count else "-- 未学习"),
                ("KELLY_MIN", f"{kelly_min_pct}%", "3%~10%", "✓ 已学习" if sample_count else "-- 未学习"),
            ], last_adj)
        else:
            self._card_kelly.set_sample_count("")
            self._card_kelly.set_content([
                ("KELLY_FRACTION", "-", "0.8~1.0", "-- 未学习"),
                ("KELLY_MAX", "-", "50%~90%", "-- 未学习"),
                ("KELLY_MIN", "-", "3%~10%", "-- 未学习"),
            ], "")

        # ── 贝叶斯（bayesian_state.json）──
        data = self._load_state_file("bayesian")
        if data:
            state = data.get("state", {})
            created = state.get("created_at", 0)
            last_save = state.get("last_save_time", 0)
            if created > 0:
                all_created.append(float(created))
            if last_save > 0:
                all_last_save.append(float(last_save))
            dists = data.get("distributions", {})
            total_recv = state.get("total_signals_received", 0)
            total_acc = state.get("total_signals_accepted", 0)
            sample = total_recv
            status = "✓ 已学习" if len(dists) > 0 or total_recv > 0 else "-- 未学习"
            self._card_bayesian.set_sample_count(f"样本: {total_recv} 信号" if sample else "")
            self._card_bayesian.set_content([
                ("分布数", str(len(dists)), "-", status),
                ("总信号/通过", f"{total_recv} / {total_acc}", "-", status),
            ], "")
        else:
            self._card_bayesian.set_sample_count("")
            self._card_bayesian.set_content([
                ("分布数", "-", "-", "-- 未学习"),
                ("总信号/通过", "-", "-", "-- 未学习"),
            ], "")

        # ── TP/SL（tpsl_tracker_state.json）──
        data = self._load_state_file("tpsl")
        if data:
            state = data.get("state", {})
            created = state.get("created_at", 0)
            last_save = state.get("last_save_time", 0)
            if created > 0:
                all_created.append(float(created))
            if last_save > 0:
                all_last_save.append(float(last_save))
            total_rec = state.get("total_records", 0)
            total_ev = state.get("total_evaluations", 0)
            self._card_tpsl.set_sample_count(f"样本: {total_rec}笔" if total_rec else "")
            self._card_tpsl.set_content([
                ("记录数", str(total_rec), "-", "✓ 已学习" if total_rec else "-- 未学习"),
                ("已评估", str(total_ev), "-", "✓ 已学习" if total_ev else "-- 未学习"),
            ], "")
        else:
            self._card_tpsl.set_sample_count("")
            self._card_tpsl.set_content([
                ("记录数", "-", "-", "-- 未学习"),
                ("已评估", "-", "-", "-- 未学习"),
            ], "")

        # ── 门控拦截（rejection_tracker_state.json）──
        data = self._load_state_file("rejection")
        if data:
            state = data.get("state", {})
            created = state.get("created_at", 0)
            last_save = state.get("last_save_time", 0)
            if created > 0:
                all_created.append(float(created))
            if last_save > 0:
                all_last_save.append(float(last_save))
            total_rej = state.get("total_rejections_recorded", 0)
            total_ev = state.get("total_evaluations_done", 0)
            self._card_rejection.set_sample_count(f"样本: {total_rej}笔" if total_rej else "")
            self._card_rejection.set_content([
                ("拒绝记录", str(total_rej), "-", "✓ 已学习" if total_rej else "-- 未学习"),
                ("已评估", str(total_ev), "-", "✓ 已学习" if total_ev else "-- 未学习"),
            ], "")
        else:
            self._card_rejection.set_sample_count("")
            self._card_rejection.set_content([
                ("拒绝记录", "-", "-", "-- 未学习"),
                ("已评估", "-", "-", "-- 未学习"),
            ], "")

        # ── 出场时机（exit_timing_state.json）──
        data = self._load_state_file("exit_timing")
        if data:
            state = data.get("state", {})
            created = state.get("created_at", 0)
            last_save = state.get("last_save_time", 0)
            if created > 0:
                all_created.append(float(created))
            if last_save > 0:
                all_last_save.append(float(last_save))
            total_ex = state.get("total_exits_recorded", 0)
            total_ev = state.get("total_evaluations_done", 0)
            self._card_exit_timing.set_sample_count(f"样本: {total_ex}笔" if total_ex else "")
            self._card_exit_timing.set_content([
                ("出场记录", str(total_ex), "-", "✓ 已学习" if total_ex else "-- 未学习"),
                ("已评估", str(total_ev), "-", "✓ 已学习" if total_ev else "-- 未学习"),
            ], "")
        else:
            self._card_exit_timing.set_sample_count("")
            self._card_exit_timing.set_content([
                ("出场记录", "-", "-", "-- 未学习"),
                ("已评估", "-", "-", "-- 未学习"),
            ], "")

        # ── 近似信号（near_miss_tracker_state.json）──
        data = self._load_state_file("near_miss")
        if data:
            state = data.get("state", {})
            created = state.get("created_at", 0)
            last_save = state.get("last_save_time", 0)
            if created > 0:
                all_created.append(float(created))
            if last_save > 0:
                all_last_save.append(float(last_save))
            total_nm = state.get("total_near_misses_recorded", 0)
            total_ev = state.get("total_evaluations_done", 0)
            self._card_near_miss.set_sample_count(f"样本: {total_nm}笔" if total_nm else "")
            self._card_near_miss.set_content([
                ("近似信号记录", str(total_nm), "-", "✓ 已学习" if total_nm else "-- 未学习"),
                ("已评估", str(total_ev), "-", "✓ 已学习" if total_ev else "-- 未学习"),
            ], "")
        else:
            self._card_near_miss.set_sample_count("")
            self._card_near_miss.set_content([
                ("近似信号记录", "-", "-", "-- 未学习"),
                ("已评估", "-", "-", "-- 未学习"),
            ], "")

        # ── 顶部时间进度 ──
        if all_created and all_last_save:
            t0 = min(all_created)
            t1 = max(all_last_save)
            dt_start = datetime.fromtimestamp(t0)
            dt_end = datetime.fromtimestamp(t1)
            duration_sec = max(0, t1 - t0)
            hours = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            duration_str = f"{hours}小时{minutes}分钟" if hours else f"{minutes}分钟"
            self._time_range_label.setText(
                f"记忆时间: {dt_start.strftime('%Y-%m-%d %H:%M')} → {dt_end.strftime('%Y-%m-%d %H:%M')}  (持续学习 {duration_str})"
            )
        else:
            self._time_range_label.setText("记忆时间: 暂无持久化数据，开始交易后将自动积累")

    # ═══════════════════════════════════════════════════════════
    # 公共更新接口
    # ═══════════════════════════════════════════════════════════

    def update_from_trades(self, closed_orders: List[Any]):
        """
        从已平仓订单列表更新所有组件
        
        Args:
            closed_orders: PaperOrder 列表
        """
        if not closed_orders:
            return
        
        # 转换为 TradeRecord
        self._trade_records = []
        for order in closed_orders:
            try:
                record = TradeRecord(
                    order_id=order.order_id,
                    profit_usdt=getattr(order, 'realized_pnl', 0.0),
                    profit_pct=getattr(order, 'profit_pct', 0.0),
                    close_reason=order.close_reason.value if order.close_reason else "未知",
                    direction=order.side.value if order.side else "LONG",
                    hold_bars=getattr(order, 'hold_bars', 0),
                    peak_profit_pct=getattr(order, 'peak_profit_pct', 0.0),
                    entry_time=order.entry_time if order.entry_time else datetime.now(),
                    exit_time=order.exit_time if order.exit_time else datetime.now(),
                )
                self._trade_records.append(record)
            except Exception as e:
                print(f"[AdaptiveLearningTab] 解析订单失败: {e}")
                continue
        
        # 数据已保留供兼容；展示改为从状态文件刷新
        self.refresh_from_state_files()

    def _update_profit_summary(self):
        """已由 6 卡片 + 状态文件刷新替代，保留为空以兼容调用"""
        pass

    def _update_trade_analysis(self):
        """已由状态文件刷新替代，保留为空以兼容调用"""
        pass

    def _normalize_close_reason(self, reason: str) -> str:
        """标准化平仓原因"""
        if "止损" in reason and "追踪" not in reason and "保本" not in reason:
            return "止损"
        elif "止盈" in reason:
            return "止盈"
        elif "追踪止损" in reason or "保本止损" in reason or "保本" in reason:
            return "追踪止损"
        elif "超时" in reason or "MAX_HOLD" in reason:
            return "超时离场"
        elif "脱轨" in reason or "DERAIL" in reason:
            return "脱轨"
        elif "手动" in reason or "MANUAL" in reason:
            return "手动平仓"
        elif "翻转" in reason:
            return "位置翻转"
        else:
            return reason
    
    def _generate_suggestion(self, reason: str, stats: CloseReasonStats) -> str:
        """根据盈亏统计生成调整建议"""
        win_rate = (stats.win_count / stats.count * 100) if stats.count > 0 else 0
        
        if reason == "止损":
            # 止损触发的亏损分析
            if stats.count >= 3 and stats.total_pnl < 0:
                # 检查是否止损后反转（通过平均峰值流失判断）
                if stats.avg_peak_loss > 20:  # 峰值流失大说明止损过紧
                    return "放宽止损距离"
                elif stats.avg_hold_bars < 30:  # 持仓时间短说明入场时机不好
                    return "提高入场阈值"
                else:
                    return "观察中..."
            elif stats.total_pnl >= 0:
                return "保持当前设置"
            else:
                return "观察中..."
        
        elif reason == "止盈":
            # 止盈触发的盈利分析
            if win_rate >= 80 and stats.avg_peak_loss > 30:
                # 止盈触发但峰值流失大，说明可以更早止盈或用追踪
                return "考虑追踪止损"
            elif win_rate >= 80:
                return "✓ 保持当前设置"
            else:
                return "检查止盈距离"
        
        elif reason == "追踪止损":
            # 追踪止损分析
            if win_rate >= 60:
                if stats.avg_peak_loss > 35:
                    return "提前启动追踪"
                else:
                    return "✓ 追踪有效"
            else:
                return "调整追踪阈值"
        
        elif reason == "超时离场":
            # 超时平仓分析
            if win_rate < 40 and stats.count >= 2:
                return "缩短最大持仓"
            elif win_rate >= 50:
                return "保持当前设置"
            else:
                return "观察中..."
        
        elif reason == "脱轨":
            # 脱轨离场分析
            if win_rate < 50 and stats.count >= 2:
                return "收紧脱轨阈值"
            else:
                return "保持当前设置"
        
        else:
            return "保持当前设置"
    
    def _update_adaptation_results(self):
        """更新自适应调整结果"""
        # 从盈亏分析中提取调整建议并转化为实际调整
        if not self._trade_records:
            return
        
        results: List[AdaptationResult] = []
        
        # 分析止损情况
        stop_loss_records = [r for r in self._trade_records[-20:] 
                           if "止损" in r.close_reason and "追踪" not in r.close_reason and "保本" not in r.close_reason]
        if len(stop_loss_records) >= 3:
            loss_count = sum(1 for r in stop_loss_records if r.profit_usdt < 0)
            # 检查止损后是否反转（通过峰值判断）
            reversal_count = sum(1 for r in stop_loss_records if r.peak_profit_pct > abs(r.profit_pct) * 0.3)
            
            if loss_count >= len(stop_loss_records) * 0.7 and reversal_count >= 2:
                current_sl = PAPER_TRADING_CONFIG.get("STOP_LOSS_ATR", 2.0)
                new_sl = min(current_sl + 0.2, 3.0)
                if new_sl != current_sl:
                    results.append(AdaptationResult(
                        parameter="STOP_LOSS_ATR",
                        old_value=current_sl,
                        new_value=new_sl,
                        reason=f"{loss_count}笔止损后反转",
                        timestamp=datetime.now()
                    ))
        
        # 分析追踪止损情况
        trailing_records = [r for r in self._trade_records[-20:] 
                          if "追踪" in r.close_reason or "保本" in r.close_reason]
        if trailing_records:
            avg_peak_loss = sum(r.peak_profit_pct - r.profit_pct for r in trailing_records) / len(trailing_records)
            if avg_peak_loss > 30:  # 峰值流失超过30%
                current_ts = PAPER_TRADING_CONFIG.get("TRAILING_STAGE1_PCT", 1.0)
                new_ts = max(current_ts - 0.2, 0.5)
                if new_ts != current_ts:
                    results.append(AdaptationResult(
                        parameter="TRAILING_STAGE1_PCT",
                        old_value=current_ts,
                        new_value=new_ts,
                        reason=f"峰值利润流失{avg_peak_loss:.0f}%",
                        timestamp=datetime.now()
                    ))
        
        # 分析连续亏损情况
        sorted_records = sorted(self._trade_records, key=lambda x: x.exit_time, reverse=True)
        consecutive_losses = 0
        for r in sorted_records[:10]:
            if r.profit_usdt < 0:
                consecutive_losses += 1
            else:
                break
        
        if consecutive_losses >= 3:
            current_threshold = SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.65)
            new_threshold = min(current_threshold + 0.02 * (consecutive_losses - 2), 0.75)
            if new_threshold != current_threshold:
                results.append(AdaptationResult(
                    parameter="FUSION_THRESHOLD",
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"连亏{consecutive_losses}笔，收紧",
                    timestamp=datetime.now()
                ))
        
        self._adaptation_results = results

    def update_effect_tracking(self,
                              old_stats: Dict[str, float],
                              new_stats: Dict[str, float],
                              last_adjustment_time: datetime = None):
        """已由状态文件刷新替代，保留为空以兼容调用"""
        pass
    
    # ═══════════════════════════════════════════════════════════
    # 冷启动面板接口
    # ═══════════════════════════════════════════════════════════

    def set_cold_start_enabled(self, enabled: bool):
        self.cold_start_panel.set_enabled(enabled)

    def is_cold_start_enabled(self) -> bool:
        return self.cold_start_panel.is_enabled()

    def update_cold_start_thresholds(self, fusion: float, cosine: float, 
                                     euclidean: float = 0, dtw: float = 0, **kwargs):
        self.cold_start_panel.update_thresholds(fusion, cosine, euclidean, dtw)

    # ═══════════════════════════════════════════════════════════
    # 兼容旧接口
    # ═══════════════════════════════════════════════════════════

    @property
    def rejection_log_card(self):
        if not hasattr(self, '_dummy_rejection_card'):
            self._dummy_rejection_card = _DummyTrackerCard()
        return self._dummy_rejection_card

    @property
    def exit_timing_card(self):
        if not hasattr(self, '_dummy_exit_timing_card'):
            self._dummy_exit_timing_card = _DummyTrackerCard()
        return self._dummy_exit_timing_card

    @property
    def tpsl_card(self):
        if not hasattr(self, '_dummy_tpsl_card'):
            self._dummy_tpsl_card = _DummyTrackerCard()
        return self._dummy_tpsl_card

    @property
    def near_miss_card(self):
        if not hasattr(self, '_dummy_near_miss_card'):
            self._dummy_near_miss_card = _DummyTrackerCard()
        return self._dummy_near_miss_card

    @property
    def regime_card(self):
        if not hasattr(self, '_dummy_regime_card'):
            self._dummy_regime_card = _DummyTrackerCard()
        return self._dummy_regime_card

    @property
    def early_exit_card(self):
        if not hasattr(self, '_dummy_early_exit_card'):
            self._dummy_early_exit_card = _DummyTrackerCard()
        return self._dummy_early_exit_card

    @property
    def trade_timeline(self):
        if not hasattr(self, '_dummy_trade_timeline'):
            self._dummy_trade_timeline = _DummyTradeTimeline()
        return self._dummy_trade_timeline

    def update_summary(self, total: int, accuracy: float, adjustments: int):
        """兼容旧接口"""
        pass

    def update_entry_gate(self, rejections, scores, suggestions):
        pass

    def update_exit_timing(self, records, scores, suggestions):
        pass

    def update_tpsl(self, records, scores, suggestions):
        pass

    def update_near_miss(self, records, scores, suggestions):
        pass

    def update_regime(self, records, scores, suggestions):
        pass

    def update_early_exit(self, records, scores, suggestions):
        pass

    def update_overview(self, total_decisions: int = 0, learning_count: int = 0, 
                        completed_count: int = 0, improvement_pct: float = 0.0):
        """兼容旧接口 - 用新数据驱动"""
        pass
    
    def update_adaptation_card(self, param_key: str, **kwargs):
        """兼容旧接口"""
        pass

    def add_adjustment_record(self, timestamp: str, param_name: str,
                              old_value: float, new_value: float, reason: str):
        """兼容旧接口 - 添加调整记录"""
        try:
            ts = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else datetime.now()
        except:
            ts = datetime.now()
        
        result = AdaptationResult(
            parameter=param_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            timestamp=ts
        )
        self._adaptation_results.append(result)
        self.refresh_from_state_files()

    def update_adaptive_dashboard(self, dashboard_data: Dict[str, Any]):
        """兼容旧接口 - 从自适应控制器更新仪表板数据"""
        if not dashboard_data:
            return
        
        # 从dashboard数据更新
        recent_adjustments = dashboard_data.get('recent_adjustments', [])
        
        # 转换为AdaptationResult
        results = []
        for adj in recent_adjustments[-10:]:
            try:
                ts = datetime.fromisoformat(adj.get('timestamp', '')) if adj.get('timestamp') else datetime.now()
            except:
                ts = datetime.now()
            
            results.append(AdaptationResult(
                parameter=adj.get('parameter', ''),
                old_value=adj.get('old_value', 0),
                new_value=adj.get('new_value', 0),
                reason=adj.get('reason', ''),
                timestamp=ts
            ))
        
        if results:
            self.refresh_from_state_files()

    def update_cold_start_frequency(self, last_trade_time, today_trades, trades_per_hour, status="normal"):
        pass

    def show_cold_start_auto_relax(self, message: str):
        pass

    def hide_cold_start_auto_relax(self):
        pass

    def add_trade_to_timeline(self, order, deepseek_review=None):
        pass

    def add_deepseek_review(self, order_id: str, review_data: Dict[str, Any]):
        pass

    def get_deepseek_review(self, order_id: str):
        return None
