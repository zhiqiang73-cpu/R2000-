"""
自适应学习标签页 - Cyberpunk/Sci-Fi HUD 风格
核心逻辑：每笔交易盈亏 → 分析原因 → 自动调整参数 → 提升盈利
特性：高密度数据可视化 + 赛博朋克风格 + 自定义绘图组件
"""
import json
import os
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from PyQt6 import QtWidgets, QtCore, QtGui

from config import UI_CONFIG, PAPER_TRADING_CONFIG, SIMILARITY_CONFIG


# ═══════════════════════════════════════════════════════════
# 自定义绘图组件：赛博朋克风格仪表盘
# ═══════════════════════════════════════════════════════════

class CircularGaugeWidget(QtWidgets.QWidget):
    """
    环形仪表盘组件 (Cyberpunk Style)
    用于展示百分比数据 (如胜率、凯利仓位)
    特性：发光圆环、中心大数字、动态颜色
    """
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._value = 0.0
        self._max_value = 100.0
        self._color = QtGui.QColor("#D9B36A")  # Warm accent
        self.setMinimumSize(120, 120)

    def set_value(self, value: float, max_value: float = 100.0, color: str = "#00F0FF"):
        self._value = value
        self._max_value = max_value
        self._color = QtGui.QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        center = rect.center()
        center_f = QtCore.QPointF(center.x(), center.y())
        center_f = QtCore.QPointF(center.x(), center.y())
        radius = min(rect.width(), rect.height()) / 2 - 10
        
        # 1. 绘制背景轨道 (暗淡)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2B3138"), 6))
        painter.drawEllipse(QtCore.QRectF(
            center_f.x() - radius,
            center_f.y() - radius,
            radius * 2,
            radius * 2
        ))

        # 2. 绘制进度弧 (发光)
        if self._max_value > 0:
            angle = int((self._value / self._max_value) * 360 * 16)
        else:
            angle = 0
            
        pen = QtGui.QPen(self._color, 6)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        # startAngle: 90 degrees (12 o'clock) = 90 * 16
        # spanAngle: Negative for clockwise
        painter.drawArc(
            int(center.x() - radius), int(center.y() - radius), 
            int(radius * 2), int(radius * 2), 
            90 * 16, -angle
        )

        # 3. 绘制中心文字
        painter.setPen(QtGui.QColor("#E7EDF4"))
        font = QtGui.QFont("Bahnschrift", 16, QtGui.QFont.Weight.DemiBold)
        painter.setFont(font)
        text = f"{self._value:.1f}%" if "%" in self._title else f"{self._value}"
        if "Leverage" in self._title or "Leverage" in self._title:
             text = f"{self._value:.0f}x"

        metrics = QtGui.QFontMetrics(font)
        text_rect = metrics.boundingRect(text)
        painter.drawText(
            int(center.x() - text_rect.width() / 2),
            int(center.y() + text_rect.height() / 4),
            text
        )

        # 4. 绘制标题
        painter.setPen(QtGui.QColor("#8F9AA6"))
        font_title = QtGui.QFont("Segoe UI", 9)
        painter.setFont(font_title)
        metrics_title = QtGui.QFontMetrics(font_title)
        title_rect = metrics_title.boundingRect(self._title)
        painter.drawText(
            int(center.x() - title_rect.width() / 2),
            int(center.y() + radius + 15),
            self._title
        )
        
        painter.end()


class RadarChartWidget(QtWidgets.QWidget):
    """
    雷达图组件 (Cyberpunk Style)
    用于展示多维市场状态准确率
    """
    def __init__(self, labels: List[str], parent=None):
        super().__init__(parent)
        self._labels = labels
        self._values = [0.0] * len(labels)  # 0.0 - 1.0
        self._color = QtGui.QColor("#4DA3FF")  # Cool blue
        self.setMinimumSize(180, 180)

    def set_values(self, values: List[float]):
        """Values should be between 0.0 and 1.0"""
        if len(values) != len(self._labels):
            return
        self._values = values
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 - 25
        n = len(self._labels)
        angle_step = 2 * math.pi / n

        # 1. 绘制网格 (同心多边形)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2B3138"), 1))
        for r_factor in [0.25, 0.5, 0.75, 1.0]:
            r = radius * r_factor
            points = []
            for i in range(n):
                angle = i * angle_step - math.pi / 2
                x = center.x() + r * math.cos(angle)
                y = center.y() + r * math.sin(angle)
                points.append(QtCore.QPointF(x, y))
            painter.drawPolygon(points)

        # 2. 绘制轴线
        center_f = QtCore.QPointF(center.x(), center.y())  # 创建 QPointF 用于 drawLine
        painter.setPen(QtGui.QPen(QtGui.QColor("#3A414A"), 1, QtCore.Qt.PenStyle.DashLine))
        for i in range(n):
            angle = i * angle_step - math.pi / 2
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            painter.drawLine(QtCore.QLineF(center_f, QtCore.QPointF(x, y)))
            
            # 绘制标签
            lbl_r = radius + 15
            lbl_x = center.x() + lbl_r * math.cos(angle)
            lbl_y = center.y() + lbl_r * math.sin(angle)
            
            painter.setPen(QtGui.QColor("#9AA5B1"))
            font = QtGui.QFont("Segoe UI", 9)
            painter.setFont(font)
            fm = QtGui.QFontMetrics(font)
            txt = self._labels[i]
            tw = fm.horizontalAdvance(txt)
            th = fm.height()
            painter.drawText(int(lbl_x - tw/2), int(lbl_y + th/4), txt)

        # 3. 绘制数据区域
        data_points = []
        for i, val in enumerate(self._values):
            angle = i * angle_step - math.pi / 2
            r = radius * val
            x = center.x() + r * math.cos(angle)
            y = center.y() + r * math.sin(angle)
            data_points.append(QtCore.QPointF(x, y))

        # 填充区域
        fill_color = QtGui.QColor(self._color)
        fill_color.setAlpha(60)
        painter.setBrush(QtGui.QBrush(fill_color))
        painter.setPen(QtGui.QPen(self._color, 2))
        painter.drawPolygon(data_points)

        # 绘制顶点
        painter.setBrush(QtGui.QBrush(self._color))
        for p in data_points:
            painter.drawEllipse(QtCore.QRectF(p.x() - 3, p.y() - 3, 6, 6))
        
        painter.end()


class SparklineWidget(QtWidgets.QWidget):
    """
    迷你趋势图组件
    用于在卡片中展示参数历史变化
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []
        self._color = QtGui.QColor("#F0F")
        self.setFixedHeight(30)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_data(self, data: List[float], color: str = "#FF0055"):
        self._data = data
        self._color = QtGui.QColor(color)
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return
            
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        w = rect.width()
        h = rect.height()
        
        min_val = min(self._data)
        max_val = max(self._data)
        rng = max_val - min_val if max_val != min_val else 1.0
        
        points = []
        step = w / (len(self._data) - 1) if len(self._data) > 1 else 0
        
        for i, val in enumerate(self._data):
            x = i * step
            # Normalize to 0-1, then flip Y (0 is top)
            normalized = (val - min_val) / rng
            y = h - (normalized * h)
            # Add padding
            y = 2 + y * (h - 4) / h
            points.append(QtCore.QPointF(x, y))
            
        pen = QtGui.QPen(self._color, 1.5)
        painter.setPen(pen)
        painter.drawPolyline(points)
        
        painter.end()


# ═══════════════════════════════════════════════════════════
# HUD 风格卡片容器
# ═══════════════════════════════════════════════════════════

class HudCard(QtWidgets.QFrame):
    """HUD 风格卡片容器；带 adjustment_confirmed 供 main_window 连接"""
    adjustment_confirmed = QtCore.pyqtSignal(str, float)

    """
    HUD 风格卡片容器
    特性：半透明背景、发光边角、非矩形感
    """
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(18, 22, 26, 0.95);
                border: 1px solid #2A3139;
                border-radius: 8px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QtWidgets.QWidget()
        header.setStyleSheet("background: #131920; border-bottom: 1px solid #2A3139;")
        header.setFixedHeight(32)
        
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        # Glowing accent
        accent = QtWidgets.QFrame()
        accent.setFixedSize(4, 16)
        accent.setStyleSheet("background-color: #D9B36A;")
        header_layout.addWidget(accent)
        
        title_lbl = QtWidgets.QLabel(self._title)
        title_lbl.setStyleSheet("color: #D9B36A; font-family: 'Bahnschrift', 'Segoe UI'; font-weight: 600; font-size: 12px; letter-spacing: 0.5px;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # Content Container
        self.content_area = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)
        layout.addWidget(self.content_area)

    def add_widget(self, widget: QtWidgets.QWidget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout: QtWidgets.QLayout):
        self.content_layout.addLayout(layout)


# ═══════════════════════════════════════════════════════════
# 数据模型 (简略版，与之前保持一致)
# ═══════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    order_id: str
    profit_usdt: float
    profit_pct: float
    close_reason: str
    direction: str
    hold_bars: int
    peak_profit_pct: float
    entry_time: datetime
    exit_time: datetime

@dataclass
class CloseReasonStats:
    reason: str
    count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_hold_bars: float = 0.0
    avg_peak_loss: float = 0.0
    suggestion: str = ""

@dataclass
class AdaptationResult:
    parameter: str
    old_value: float
    new_value: float
    reason: str
    timestamp: datetime = None

# Dummy Classes for compatibility
class _DummySignal(QtCore.QObject):
    adjustment_confirmed = QtCore.pyqtSignal(str, float)

class _DummyTrackerCard:
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
    def add_trade(self, order, deepseek_review=None): pass
    def update_deepseek_review(self, order_id, review): pass
    def clear_trades(self): pass


class _EntryConditionsAdaptiveCard(QtWidgets.QGroupBox):
    """开仓条件自适应列表卡片 - 显示融合/余弦/欧氏/DTW 当前值与阈值"""
    def __init__(self, parent=None):
        super().__init__("开仓条件自适应", parent)
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #333;
                margin-top: 2px;
                padding: 2px 4px 2px 4px;
                color: #555;
                font-weight: bold;
                font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
                margin-top: 2px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 16, 6, 6)
        layout.setSpacing(2)
        self._rows: Dict[str, Dict[str, QtWidgets.QLabel]] = {}
        for key, label in [
            ("fusion", "融合评分"),
            ("cosine", "余弦"),
            ("euclidean", "欧氏"),
            ("dtw", "DTW"),
        ]:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label + ":")
            lbl.setStyleSheet("color: #888; font-size: 10px; min-width: 48px;")
            val_lbl = QtWidgets.QLabel("--")
            val_lbl.setStyleSheet("color: #D9B36A; font-weight: bold; font-size: 11px;")
            thresh_lbl = QtWidgets.QLabel("/ --")
            thresh_lbl.setStyleSheet("color: #66707C; font-size: 9px;")
            status_lbl = QtWidgets.QLabel("--")
            status_lbl.setStyleSheet("color: #888; font-size: 9px;")
            row.addWidget(lbl)
            row.addWidget(val_lbl, 1)
            row.addWidget(thresh_lbl)
            row.addWidget(status_lbl)
            layout.addLayout(row)
            self._rows[key] = {"value": val_lbl, "threshold": thresh_lbl, "status": status_lbl}
    
    def update_values(self, fusion: float = 0.0, cosine: float = 0.0, euclidean: float = 0.0, dtw: float = 0.0,
                      fusion_th: float = 0.4, cos_th: float = 0.7, euc_th: float = 0.35, dtw_th: float = 0.3):
        """更新当前值与阈值，并显示状态"""
        data = [
            ("fusion", fusion, fusion_th),
            ("cosine", cosine, cos_th),
            ("euclidean", euclidean, euc_th),
            ("dtw", dtw, dtw_th),
        ]
        for key, val, th in data:
            row = self._rows.get(key)
            if not row:
                continue
            if val < 0.001:
                row["value"].setText("--")
                row["threshold"].setText(f"/ {th:.0%}")
                row["status"].setText("--")
                row["value"].setStyleSheet("color: #66707C; font-weight: bold; font-size: 11px;")
            else:
                row["value"].setText(f"{val:.0%}")
                row["threshold"].setText(f"/ {th:.0%}")
                passed = val >= th
                near = val >= th - 0.10
                if passed:
                    row["status"].setText("✓")
                    row["value"].setStyleSheet("color: #089981; font-weight: bold; font-size: 11px;")
                elif near:
                    row["status"].setText("≈")
                    row["value"].setStyleSheet("color: #FFD54F; font-weight: bold; font-size: 11px;")
                else:
                    row["status"].setText("✗")
                    row["value"].setStyleSheet("color: #f23645; font-weight: bold; font-size: 11px;")


class ColdStartPanel(QtWidgets.QWidget):
    """冷启动模式开关面板"""
    cold_start_toggled = QtCore.pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        
        self._chk = QtWidgets.QCheckBox("冷启动")
        self._chk.setToolTip("放宽匹配门槛以增加初始交易频率")
        self._chk.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._chk.setStyleSheet("""
            QCheckBox {
                color: #D9B36A;
                font-weight: 600;
                padding: 4px 8px;
            }
            QCheckBox:checked { color: #6BCB77; }
        """)
        self._chk.stateChanged.connect(self._on_state_changed)
        layout.addWidget(self._chk)
        
    def _on_state_changed(self, state):
        enabled = self._chk.isChecked()
        self.cold_start_toggled.emit(enabled)
    
    def set_checked(self, checked: bool):
        """程序化设置勾选状态（不触发信号）"""
        self._chk.blockSignals(True)
        self._chk.setChecked(checked)
        self._chk.blockSignals(False)


# ═══════════════════════════════════════════════════════════
# 主界面：AdaptiveLearningTab
# ═══════════════════════════════════════════════════════════

class AdaptiveLearningTab(QtWidgets.QWidget):
    """
    Main Tab - Cyberpunk Edition
    """
    clear_memory_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cold_start_enabled = False
        self._deepseek_reviews = {}
        # 先创建 cold_start_panel，供 _init_ui 中的 top_layout 使用
        self.cold_start_panel = ColdStartPanel(self)
        self.rejection_log_card = _DummyTrackerCard()
        self.exit_timing_card = _DummyTrackerCard()
        self.near_miss_card = _DummyTrackerCard()
        self.early_exit_card = _DummyTrackerCard()
        self.trade_timeline = _DummyTradeTimeline()
        self._init_ui()
        self._init_data_structures()
        # Load state after UI is ready
        QtCore.QTimer.singleShot(500, self.refresh_from_state_files)

    def _init_ui(self):
        # Global Style
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #0E1114, stop:1 #141A20);
                color: #D7DEE6;
                font-family: 'Bahnschrift', 'Segoe UI', sans-serif;
            }
            QScrollBar:vertical {
                border: none;
                background: #0E1114;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #2A3139;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 12)
        main_layout.setSpacing(12)

        # ═══ 1. Top Bar (Mission Control) ═══
        top_bar = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title with digital clock style
        title_box = QtWidgets.QWidget()
        tb_layout = QtWidgets.QVBoxLayout(title_box)
        title_lbl = QtWidgets.QLabel("自适应系统 // R3000")
        title_lbl.setStyleSheet("color: #D9B36A; font-family: 'Bahnschrift', 'Segoe UI'; font-size: 22px; letter-spacing: 1px; font-weight: 600;")
        sub_lbl = QtWidgets.QLabel("自主优化模块")
        sub_lbl.setStyleSheet("color: #66707C; font-size: 10px; letter-spacing: 2px; font-weight: 600;")
        tb_layout.addWidget(title_lbl)
        tb_layout.addWidget(sub_lbl)
        top_layout.addWidget(title_box)
        
        top_layout.addStretch()
        
        # Status Badge
        self._status_badge = QtWidgets.QLabel("● 系统运行中")
        self._status_badge.setStyleSheet("color: #D9B36A; border: 1px solid #3A3F46; border-radius: 6px; padding: 4px 10px; font-weight: 600; background: #10151A;")
        top_layout.addWidget(self._status_badge)
        
        # Cold Start Toggle
        top_layout.addWidget(self.cold_start_panel)
        
        # Action Buttons
        refresh_btn = QtWidgets.QPushButton("刷新数据")
        refresh_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #121820;
                color: #D9B36A;
                border: 1px solid #3A3F46;
                padding: 6px 16px;
                font-family: 'Bahnschrift', 'Segoe UI';
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QPushButton:hover { background-color: #151C25; border-color: #5A4B2E; }
        """)
        refresh_btn.clicked.connect(self.refresh_from_state_files)
        top_layout.addWidget(refresh_btn)
        
        main_layout.addWidget(top_bar)

        # ═══ 2. Dashboard Grid (Asymmetric) ═══
        dash_layout = QtWidgets.QHBoxLayout()
        dash_layout.setSpacing(20)
        
        # ── Left Column (Core Stats - 35%) ──
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(20)
        
        # Kelly & Leverage Card
        self.kelly_card = HudCard("凯利仓位 & 杠杆")
        gauges_layout = QtWidgets.QHBoxLayout()
        
        # Kelly Gauge
        self.kelly_gauge = CircularGaugeWidget("仓位 %")
        gauges_layout.addWidget(self.kelly_gauge)
        
        # Leverage Gauge (NEW)
        self.lev_gauge = CircularGaugeWidget("杠杆 x")
        self.lev_gauge.set_value(0, 100, "#4DA3FF") # Cool blue
        gauges_layout.addWidget(self.lev_gauge)
        
        self.kelly_card.add_layout(gauges_layout)
        
        # Sparklines for kelly parameters
        self.kelly_card.add_widget(QtWidgets.QLabel("凯利分数历史："))
        self.spark_kelly = SparklineWidget()
        self.kelly_card.add_widget(self.spark_kelly)
        left_col.addWidget(self.kelly_card)
        
        # Win Rate Gauge (Bayesian)
        self.bayesian_card = HudCard("贝叶斯概率")
        self.winrate_gauge = CircularGaugeWidget("预估胜率")
        w_layout = QtWidgets.QHBoxLayout()
        w_layout.addStretch()
        w_layout.addWidget(self.winrate_gauge)
        w_layout.addStretch()
        self.bayesian_card.add_layout(w_layout)
        left_col.addWidget(self.bayesian_card)
        
        dash_layout.addLayout(left_col, 35)
        
        # ── Right Column (Analysis Modules - 65%) ──
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(20)
        
        # Row 1: Market Regime Radar + TPSL
        row1 = QtWidgets.QHBoxLayout()
        
        # Regime Radar
        self.regime_card = HudCard("市场状态准确率")
        self.radar_chart = RadarChartWidget(["趋势", "震荡", "波动", "弱势"])
        r_layout = QtWidgets.QHBoxLayout()
        r_layout.addStretch()
        r_layout.addWidget(self.radar_chart)
        r_layout.addStretch()
        self.regime_card.add_layout(r_layout)
        row1.addWidget(self.regime_card, 1)
        
        # TPSL Stats
        self.tpsl_card = HudCard("止盈止损优化")
        tpsl_grid = QtWidgets.QGridLayout()
        tpsl_grid.addWidget(QtWidgets.QLabel("动态止盈 (ATR x)："), 0, 0)
        self.lbl_tp_mult = QtWidgets.QLabel("3.5")
        self.lbl_tp_mult.setStyleSheet("color: #D9B36A; font-family: 'Bahnschrift', 'Segoe UI'; font-size: 14px;")
        tpsl_grid.addWidget(self.lbl_tp_mult, 0, 1)
        
        tpsl_grid.addWidget(QtWidgets.QLabel("动态止损 (ATR x)："), 1, 0)
        self.lbl_sl_mult = QtWidgets.QLabel("2.0")
        self.lbl_sl_mult.setStyleSheet("color: #D9B36A; font-family: 'Bahnschrift', 'Segoe UI'; font-size: 14px;")
        tpsl_grid.addWidget(self.lbl_sl_mult, 1, 1)
        
        tpsl_grid.addWidget(QtWidgets.QLabel("优化历史："), 2, 0, 1, 2)
        self.spark_tpsl = SparklineWidget()
        tpsl_grid.addWidget(self.spark_tpsl, 3, 0, 1, 2)
        
        self.tpsl_card.add_layout(tpsl_grid)
        row1.addWidget(self.tpsl_card, 1)
        
        right_col.addLayout(row1)
        
        # Row 2: Secondary Trackers (Exit Timing, Near Miss)
        row2 = QtWidgets.QHBoxLayout()
        
        # Exit Timing
        self.exit_card = HudCard("出场时机追踪")
        self.lbl_exit_status = QtWidgets.QLabel("状态：校准中...")
        self.lbl_exit_status.setStyleSheet("color: #D9B36A;")
        self.exit_card.add_widget(self.lbl_exit_status)
        self.exit_card.add_widget(QtWidgets.QLabel("过早出场：0"))
        
        # Near Miss
        self.miss_card = HudCard("近似信号分析")
        self.lbl_miss_status = QtWidgets.QLabel("错过机会：0")
        self.miss_card.add_widget(self.lbl_miss_status)
        
        row2.addWidget(self.exit_card)
        row2.addWidget(self.miss_card)
        right_col.addLayout(row2)

        dash_layout.addLayout(right_col, 65)
        main_layout.addLayout(dash_layout)

        # ═══ 3. Bottom: 左侧系统日志 + 右侧开仓条件自适应（列表）═══
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(12)
        # 左侧：系统日志
        console_group = QtWidgets.QGroupBox("系统日志")
        console_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #333;
                margin-top: 2px;
                padding: 2px 4px 2px 4px;
                color: #555;
                font-weight: bold;
                font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
                margin-top: 2px;
            }
        """)
        con_layout = QtWidgets.QVBoxLayout(console_group)
        con_layout.setContentsMargins(4, 8, 4, 4)
        con_layout.setSpacing(2)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0F1419;
                color: #B9C2CC;
                font-family: 'Consolas', 'Courier New';
                font-size: 10px;
                border: 1px solid #1F262D;
                padding: 3px;
            }
        """)
        self.log_text.setMinimumHeight(140)
        self.log_text.setPlaceholderText("> 系统就绪。等待数据流...")
        con_layout.addWidget(self.log_text)
        bottom_row.addWidget(console_group, 3)  # 左侧日志占 3 份
        
        # 右侧：开仓条件自适应（列表）
        self.entry_conditions_card = _EntryConditionsAdaptiveCard()
        bottom_row.addWidget(self.entry_conditions_card, 1)  # 右侧占 1 份
        
        main_layout.addLayout(bottom_row, 1)  # stretch=1：底部占据剩余空间

    def _init_data_structures(self):
        """Initialize stats containers"""
        self.trade_records: List[TradeRecord] = []
        self.adaptation_results: List[AdaptationResult] = []

    def append_adaptive_journal(self, text: str):
        """Add log entry with timestamp"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendHtml(f"<span style='color: #555;'>[{ts}]</span> {text}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def refresh_from_state_files(self):
        """Load data from JSON files and update UI"""
        try:
            self.append_adaptive_journal("初始化：加载模块...")
            
            # 1. Update Kelly & Leverage Gauges from adaptive_controller_state.json
            try:
                adaptive_data = self._load_state_file("adaptive")
                if adaptive_data and "kelly_adapter" in adaptive_data:
                    adapter = adaptive_data["kelly_adapter"]
                    kf = adapter.get("kelly_fraction") or PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
                    self.kelly_gauge.set_value(round(kf * 100, 1), 100.0, "#D9B36A")
                    
                    # --- LEVERAGE Logic (User requested) ---
                    lev = adapter.get("leverage") or PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20)
                    self.lev_gauge.set_value(float(lev), 100.0, "#4DA3FF")
                    
                    # Update sparkline if history exists
                    hist = adapter.get("recent_performance", [])
                    if hist:
                        self.spark_kelly.set_data(hist[-20:], "#D9B36A")
                    self.append_adaptive_journal(f"模块 [凯利] 已加载。分数：{kf:.2f}, 杠杆：{lev}x")
                    
                    # Update regime radar from regime_adapter
                    if "regime_adapter" in adaptive_data:
                        regime_data = adaptive_data["regime_adapter"]
                        regime_accuracy = regime_data.get("regime_accuracy", {})
                        # Extract accuracy for known regimes
                        trending_acc = self._get_regime_accuracy(regime_accuracy, ["强多头", "强空头", "多头", "空头"])
                        ranging_acc = self._get_regime_accuracy(regime_accuracy, ["震荡", "弱"])
                        volatile_acc = self._get_regime_accuracy(regime_accuracy, ["强", "爆发"])
                        weak_acc = self._get_regime_accuracy(regime_accuracy, ["弱多头", "弱空头"])
                        self.radar_chart.set_values([trending_acc, ranging_acc, volatile_acc, weak_acc])
                else:
                    # Use defaults
                    self._set_default_values()
            except Exception as e:
                self.append_adaptive_journal(f"错误：凯利数据加载失败 - {e}")
                self._set_default_values()

            # 2. Update Bayesian Gauge
            try:
                bayes = self._load_state_file("bayesian")
                if bayes:
                    # Try different possible structures
                    total = bayes.get("total_signals_received", 0) or bayes.get("state", {}).get("total_signals_received", 0)
                    wins = bayes.get("total_signals_accepted", 0) or bayes.get("state", {}).get("total_signals_accepted", 0)
                    rate = (wins / total * 100) if total > 0 else 50.0
                    self.winrate_gauge.set_value(round(rate, 1), 100.0, "#6BCB77")
                    self.append_adaptive_journal(f"模块 [贝叶斯] 已加载。胜率：{rate:.1f}%")
            except Exception as e:
                self.append_adaptive_journal(f"错误：贝叶斯数据加载失败 - {e}")
                
            # 3. TPSL
            try:
                tpsl = self._load_state_file("tpsl")
                if tpsl:
                    self.append_adaptive_journal("模块 [止盈止损] 已加载。")
                    # Extract TP/SL multipliers if available
                    tp_mult = PAPER_TRADING_CONFIG.get("TAKE_PROFIT_ATR", 3.5)
                    sl_mult = PAPER_TRADING_CONFIG.get("STOP_LOSS_ATR", 2.0)
                    self.lbl_tp_mult.setText(f"{tp_mult:.1f}")
                    self.lbl_sl_mult.setText(f"{sl_mult:.1f}")
                    # Set sparkline from history if available
                    if "history" in tpsl:
                        hist = [h.get("profit_pct", 0) for h in tpsl["history"][-10:]]
                        if hist:
                            self.spark_tpsl.set_data(hist, "#D9B36A")
            except Exception as e:
                self.append_adaptive_journal(f"错误：止盈止损数据加载失败 - {e}")

            self.append_adaptive_journal("系统刷新完成。")
        except Exception as e:
            self.append_adaptive_journal(f"严重错误：刷新失败 - {e}")
            import traceback
            traceback.print_exc()

    def _set_default_values(self):
        """设置默认值（当状态文件不存在时）"""
        lev = PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20)
        self.lev_gauge.set_value(float(lev), 100.0, "#4DA3FF")
        kf = PAPER_TRADING_CONFIG.get("KELLY_FRACTION", 0.25)
        self.kelly_gauge.set_value(round(kf * 100, 1), 100.0, "#D9B36A")
        self.append_adaptive_journal("使用默认配置值")

    def _get_regime_accuracy(self, regime_accuracy: Dict, keywords: List[str]) -> float:
        """Helper to get average accuracy for regimes matching keywords"""
        total_correct = 0
        total_wrong = 0
        for regime, stats in regime_accuracy.items():
            if any(kw in regime for kw in keywords):
                total_correct += stats.get("correct", 0)
                total_wrong += stats.get("wrong", 0)
        total = total_correct + total_wrong
        return (total_correct / total) if total > 0 else 0.5

    def update_kelly_leverage_realtime(self, kelly_fraction: float, leverage: int, recent_performance: List[float] = None):
        """
        Real-time update from main_window when adaptive controller updates
        
        Args:
            kelly_fraction: Current Kelly fraction (0.0-1.0)
            leverage: Current leverage (5-100)
            recent_performance: Recent trade profits (%)
        """
        self.kelly_gauge.set_value(round(kelly_fraction * 100, 1), 100.0, "#D9B36A")
        self.lev_gauge.set_value(float(leverage), 100.0, "#4DA3FF")
        
        if recent_performance:
            self.spark_kelly.set_data(recent_performance[-20:], "#D9B36A")
        
        self.append_adaptive_journal(f"实时更新：凯利={kelly_fraction:.2f}, 杠杆={leverage}x")

    def _load_state_file(self, key: str) -> Optional[Dict]:
        """Helper to load JSON state"""
        # Mapping key to filename from CONFIG
        files = {
            "kelly": "data/adaptive_controller_state.json",  # KellyAdapter data is in adaptive_controller_state.json
            "adaptive": "data/adaptive_controller_state.json",
            "bayesian": PAPER_TRADING_CONFIG.get("BAYESIAN_STATE_FILE", "data/bayesian_state.json"),
            "tpsl": PAPER_TRADING_CONFIG.get("TPSL_STATE_FILE", "data/tpsl_tracker_state.json"),
            "exit_timing": PAPER_TRADING_CONFIG.get("EXIT_TIMING_STATE_FILE", "data/exit_timing_state.json"),
            "near_miss": PAPER_TRADING_CONFIG.get("NEAR_MISS_STATE_FILE", "data/near_miss_tracker_state.json"),
        }
        # In real app, use the actual paths. For now trying common ones.
        fname = files.get(key, f"data/{key}_state.json")
        
        # For verification script compatibility, check if method is overridden (Mock)
        # The verification script overrides this method, but if we call it internally,
        # we need to be careful. In this file, we just return empty or dummy if not found.
        if not os.path.exists(fname):
            return None
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _on_clear_clicked(self):
        self.append_adaptive_journal("HUD模式下忽略清空命令（安全锁）。")

    def set_cold_start_enabled(self, enabled: bool):
        self._cold_start_enabled = bool(enabled)
        if hasattr(self, 'cold_start_panel') and hasattr(self.cold_start_panel, 'set_checked'):
            self.cold_start_panel.set_checked(enabled)

    def is_cold_start_enabled(self) -> bool:
        return getattr(self, "_cold_start_enabled", False)

    def update_cold_start_thresholds(self, **kwargs):
        pass

    def update_cold_start_frequency(self, last_trade_time=None, today_trades=0, trades_per_hour=0.0, status="normal"):
        pass

    def show_cold_start_auto_relax(self, message: str):
        pass

    def hide_cold_start_auto_relax(self):
        pass

    # --- Compatibility stubs for main_window integrations ---
    def update_entry_gate(self, *args, **kwargs):
        pass

    def update_entry_conditions_adaptive(self, *,
                                        fusion: float = 0.0, cosine: float = 0.0,
                                        euclidean: float = 0.0, dtw: float = 0.0,
                                        fusion_th: float = 0.4, cos_th: float = 0.7,
                                        euc_th: float = 0.35, dtw_th: float = 0.3):
        """更新开仓条件自适应列表（融合/余弦/欧氏/DTW 当前值与阈值）"""
        if hasattr(self, "entry_conditions_card") and self.entry_conditions_card:
            self.entry_conditions_card.update_values(
                fusion=fusion, cosine=cosine, euclidean=euclidean, dtw=dtw,
                fusion_th=fusion_th, cos_th=cos_th, euc_th=euc_th, dtw_th=dtw_th,
            )

    def update_entry_gate(self, *args, **kwargs):
        pass

    def update_exit_timing(self, records=None, scores=None, suggestions=None):
        pass

    def update_tpsl(self, records=None, scores=None, suggestions=None):
        pass

    def update_near_miss(self, records=None, scores=None, suggestions=None):
        pass

    def update_regime(self, records=None, scores=None, suggestions=None):
        pass

    def update_early_exit(self, records=None, scores=None, suggestions=None):
        pass

    def update_summary(self, total=0, accuracy=0.0, adjustments_applied=0):
        pass

    def update_adaptive_dashboard(self, dashboard_data=None):
        pass

    def add_trade_to_timeline(self, order, deepseek_review=None):
        if self.trade_timeline:
            self.trade_timeline.add_trade(order, deepseek_review=deepseek_review)

    def add_deepseek_review(self, order_id, review):
        self._deepseek_reviews[order_id] = review

    def get_deepseek_review(self, order_id):
        return self._deepseek_reviews.get(order_id)
