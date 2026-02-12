"""
R3000 分析面板
右侧分析面板：交易明细、市场状态、指纹图、轨迹匹配
深色主题
"""
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UI_CONFIG

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    # 设置深色主题
    plt.style.use('dark_background')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FeatureImportanceWidget(QtWidgets.QWidget):
    """特征重要性显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(5, 4), dpi=100, facecolor=UI_CONFIG['THEME_BACKGROUND'])
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            # 回退到表格显示
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["特征", "重要性"])
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setStyleSheet(f"""
                QTableWidget {{
                    background-color: {UI_CONFIG['THEME_SURFACE']};
                    color: {UI_CONFIG['THEME_TEXT']};
                    gridline-color: #444;
                }}
                QHeaderView::section {{
                    background-color: #333;
                    color: {UI_CONFIG['THEME_TEXT']};
                    border: 1px solid #444;
                }}
            """)
            layout.addWidget(self.table)
    
    def update_data(self, feature_names: List[str], importances: List[float]):
        """更新特征重要性数据"""
        if HAS_MATPLOTLIB:
            self.figure.clear()
            ax = self.figure.add_subplot(111, facecolor=UI_CONFIG['THEME_BACKGROUND'])
            
            # 取前20个最重要的特征
            n_show = min(20, len(feature_names))
            indices = np.argsort(importances)[::-1][:n_show]
            
            names = [feature_names[i][:15] for i in indices]
            values = [importances[i] for i in indices]
            
            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, color=UI_CONFIG['THEME_ACCENT'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=8, color=UI_CONFIG['THEME_TEXT'])
            ax.invert_yaxis()
            ax.set_xlabel('重要性', color=UI_CONFIG['THEME_TEXT'])
            ax.set_title('特征重要性排名', color=UI_CONFIG['THEME_TEXT'])
            ax.tick_params(colors=UI_CONFIG['THEME_TEXT'])
            
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            self.table.setRowCount(len(feature_names))
            for i, (name, imp) in enumerate(zip(feature_names, importances)):
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{imp:.4f}"))


class SurvivalAnalysisWidget(QtWidgets.QWidget):
    """生存分析显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 统计信息
        self.stats_group = QtWidgets.QGroupBox("统计摘要")
        self.stats_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """)
        stats_layout = QtWidgets.QFormLayout(self.stats_group)
        
        self.hold_mean_label = QtWidgets.QLabel("-")
        self.hold_median_label = QtWidgets.QLabel("-")
        self.profit_mean_label = QtWidgets.QLabel("-")
        self.profit_median_label = QtWidgets.QLabel("-")
        self.win_rate_label = QtWidgets.QLabel("-")
        self.rr_label = QtWidgets.QLabel("-")
        
        stats_layout.addRow("平均持仓周期:", self.hold_mean_label)
        stats_layout.addRow("中位持仓周期:", self.hold_median_label)
        stats_layout.addRow("平均收益率:", self.profit_mean_label)
        stats_layout.addRow("中位收益率:", self.profit_median_label)
        stats_layout.addRow("胜率:", self.win_rate_label)
        stats_layout.addRow("平均风险收益比:", self.rr_label)
        
        layout.addWidget(self.stats_group)
        
        # 直方图
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(5, 3), dpi=100, facecolor=UI_CONFIG['THEME_BACKGROUND'])
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
    
    def update_data(self, survival_data: Dict):
        """更新生存分析数据"""
        if not survival_data:
            return
        
        # 更新统计标签
        if 'hold_periods' in survival_data:
            hp = survival_data['hold_periods']
            self.hold_mean_label.setText(f"{hp.get('mean', 0):.1f}")
            self.hold_median_label.setText(f"{hp.get('median', 0):.1f}")
        
        if 'profit' in survival_data:
            pf = survival_data['profit']
            self.profit_mean_label.setText(f"{pf.get('mean', 0):.2f}%")
            self.profit_median_label.setText(f"{pf.get('median', 0):.2f}%")
        
        self.win_rate_label.setText(f"{survival_data.get('win_rate', 0):.1%}")
        self.rr_label.setText(f"{survival_data.get('risk_reward_mean', 0):.2f}")
        
        # 绘制直方图
        if HAS_MATPLOTLIB:
            self.figure.clear()
            
            if 'hold_periods' in survival_data and 'histogram' in survival_data['hold_periods']:
                ax1 = self.figure.add_subplot(121, facecolor=UI_CONFIG['THEME_BACKGROUND'])
                hist = survival_data['hold_periods']['histogram']
                bins = hist['bins']
                counts = hist['counts']
                ax1.bar(bins[:-1], counts, width=np.diff(bins), align='edge', color=UI_CONFIG['THEME_ACCENT'], alpha=0.7)
                ax1.set_xlabel('持仓周期', color=UI_CONFIG['THEME_TEXT'])
                ax1.set_ylabel('频次', color=UI_CONFIG['THEME_TEXT'])
                ax1.set_title('持仓时间分布', color=UI_CONFIG['THEME_TEXT'])
                ax1.tick_params(colors=UI_CONFIG['THEME_TEXT'])
            
            if 'profit' in survival_data and 'histogram' in survival_data['profit']:
                ax2 = self.figure.add_subplot(122, facecolor=UI_CONFIG['THEME_BACKGROUND'])
                hist = survival_data['profit']['histogram']
                bins = hist['bins']
                counts = hist['counts']
                colors = [UI_CONFIG['CHART_DOWN_COLOR'] if b < 0 else UI_CONFIG['CHART_UP_COLOR'] for b in bins[:-1]]
                ax2.bar(bins[:-1], counts, width=np.diff(bins), align='edge', color=colors, alpha=0.7)
                ax2.set_xlabel('收益率 (%)', color=UI_CONFIG['THEME_TEXT'])
                ax2.set_ylabel('频次', color=UI_CONFIG['THEME_TEXT'])
                ax2.set_title('收益分布', color=UI_CONFIG['THEME_TEXT'])
                ax2.tick_params(colors=UI_CONFIG['THEME_TEXT'])
            
            self.figure.tight_layout()
            self.canvas.draw()


class PatternLogicWidget(QtWidgets.QWidget):
    """多空转换逻辑显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        table_style = f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #444;
                border: 1px solid #444;
            }}
            QHeaderView::section {{
                background-color: #333;
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #444;
                padding: 5px;
            }}
        """
        
        group_style = f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """
        
        # 做多条件表格
        self.long_group = QtWidgets.QGroupBox("做多条件")
        self.long_group.setStyleSheet(group_style)
        long_layout = QtWidgets.QVBoxLayout(self.long_group)
        self.long_table = QtWidgets.QTableWidget()
        self.long_table.setColumnCount(3)
        self.long_table.setHorizontalHeaderLabels(["特征", "条件", "概率"])
        self.long_table.horizontalHeader().setStretchLastSection(True)
        self.long_table.setStyleSheet(table_style)
        long_layout.addWidget(self.long_table)
        layout.addWidget(self.long_group)
        
        # 做空条件表格
        self.short_group = QtWidgets.QGroupBox("做空条件")
        self.short_group.setStyleSheet(group_style)
        short_layout = QtWidgets.QVBoxLayout(self.short_group)
        self.short_table = QtWidgets.QTableWidget()
        self.short_table.setColumnCount(3)
        self.short_table.setHorizontalHeaderLabels(["特征", "条件", "概率"])
        self.short_table.horizontalHeader().setStretchLastSection(True)
        self.short_table.setStyleSheet(table_style)
        short_layout.addWidget(self.short_table)
        layout.addWidget(self.short_group)
    
    def update_data(self, logic_data: Dict):
        """更新多空逻辑数据"""
        if not logic_data:
            return
        
        # 更新做多条件
        long_conditions = logic_data.get('long_conditions', [])
        self.long_table.setRowCount(len(long_conditions))
        for i, cond in enumerate(long_conditions):
            self.long_table.setItem(i, 0, QtWidgets.QTableWidgetItem(cond['feature'][:20]))
            condition_str = f"{cond['direction']} {cond['threshold']:.3f}"
            self.long_table.setItem(i, 1, QtWidgets.QTableWidgetItem(condition_str))
            self.long_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{cond['probability']:.2%}"))
        
        # 更新做空条件
        short_conditions = logic_data.get('short_conditions', [])
        self.short_table.setRowCount(len(short_conditions))
        for i, cond in enumerate(short_conditions):
            self.short_table.setItem(i, 0, QtWidgets.QTableWidgetItem(cond['feature'][:20]))
            condition_str = f"{cond['direction']} {cond['threshold']:.3f}"
            self.short_table.setItem(i, 1, QtWidgets.QTableWidgetItem(condition_str))
            self.short_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{cond['probability']:.2%}"))


class TradeLogWidget(QtWidgets.QWidget):
    """交易明细显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "方向", "入场时间", "入场价", "出场时间", "出场价", "盈利(USDT)", "收益率%", "持仓", "市场状态", "指纹摘要"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #444;
                border: 1px solid #444;
            }}
            QHeaderView::section {{
                background-color: #333;
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #444;
                padding: 4px;
            }}
        """)
        layout.addWidget(self.table)
    
    def update_trades(self, trades: List[Dict]):
        """更新交易明细"""
        self.table.setRowCount(len(trades))
        up_color = QtGui.QColor(UI_CONFIG['CHART_UP_COLOR'])
        down_color = QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR'])
        
        for i, t in enumerate(trades):
            # 方向
            side_item = QtWidgets.QTableWidgetItem(t.get("side", ""))
            if "LONG" in t.get("side", ""):
                side_item.setForeground(QtGui.QBrush(up_color))
            elif "SHORT" in t.get("side", ""):
                side_item.setForeground(QtGui.QBrush(down_color))
            self.table.setItem(i, 0, side_item)
            
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(t.get("entry_time", "")))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(t.get("entry_price", "")))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(t.get("exit_time", "")))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(t.get("exit_price", "")))
            
            # 盈利和收益率着色
            profit_str = t.get("profit", "0")
            profit_pct_str = t.get("profit_pct", "0")
            try:
                profit_val = float(profit_str)
                color = up_color if profit_val >= 0 else down_color
            except ValueError:
                color = QtGui.QColor(UI_CONFIG['THEME_TEXT'])
                
            profit_item = QtWidgets.QTableWidgetItem(profit_str)
            profit_item.setForeground(QtGui.QBrush(color))
            self.table.setItem(i, 5, profit_item)
            
            pct_item = QtWidgets.QTableWidgetItem(profit_pct_str)
            pct_item.setForeground(QtGui.QBrush(color))
            self.table.setItem(i, 6, pct_item)
            
            self.table.setItem(i, 7, QtWidgets.QTableWidgetItem(t.get("hold", "")))
            
            # 市场状态
            regime_str = t.get("regime", "")
            regime_item = QtWidgets.QTableWidgetItem(regime_str)
            regime_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            # 尝试为市场状态着色
            try:
                from core.market_regime import MarketRegime
                regime_color = MarketRegime.COLORS.get(regime_str, UI_CONFIG['THEME_TEXT'])
                regime_item.setForeground(QtGui.QBrush(QtGui.QColor(regime_color)))
            except Exception:
                pass
            self.table.setItem(i, 8, regime_item)

            # 指纹摘要（模板ID + 相似度）
            fp_str = t.get("fingerprint", "--")
            fp_item = QtWidgets.QTableWidgetItem(fp_str)
            # 有匹配时用主题高亮色，无匹配时灰色
            if fp_str and fp_str != "--":
                fp_item.setForeground(QtGui.QBrush(QtGui.QColor(UI_CONFIG['THEME_ACCENT'])))
            else:
                fp_item.setForeground(QtGui.QBrush(QtGui.QColor("#888")))
            fp_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 9, fp_item)


class MarketRegimeWidget(QtWidgets.QWidget):
    """市场状态统计组件 - 显示6种状态的出现次数和正确率"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 当前K线所处的市场状态
        cur_group = QtWidgets.QGroupBox("当前市场状态")
        cur_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
            }}
        """)
        cur_layout = QtWidgets.QHBoxLayout(cur_group)
        self.current_regime_label = QtWidgets.QLabel("--")
        self.current_regime_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.current_regime_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        cur_layout.addWidget(self.current_regime_label)
        layout.addWidget(cur_group)

        # 统计表格
        stats_group = QtWidgets.QGroupBox("各状态交易统计")
        stats_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
            }}
        """)
        stats_layout = QtWidgets.QVBoxLayout(stats_group)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "市场状态", "交易数", "做多", "做空", "盈利数", "正确率%", "平均收益%"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #444;
                border: 1px solid #444;
                font-size: 12px;
            }}
            QHeaderView::section {{
                background-color: #333;
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #444;
                padding: 4px;
                font-weight: bold;
            }}
            QTableWidget::item {{
                padding: 3px;
            }}
        """)
        stats_layout.addWidget(self.table)
        layout.addWidget(stats_group, stretch=1)

        # 初始化表格行
        self._init_table()

    def _init_table(self):
        """初始化7行（6种状态 + 1未知）"""
        from core.market_regime import MarketRegime
        regimes = MarketRegime.ALL_REGIMES + [MarketRegime.UNKNOWN]
        self.table.setRowCount(len(regimes))

        for row, regime in enumerate(regimes):
            # 市场状态名（带颜色）
            item = QtWidgets.QTableWidgetItem(regime)
            color = MarketRegime.COLORS.get(regime, "#888")
            item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.table.setItem(row, 0, item)

            # 其余列初始化为 0 或 --
            for col in range(1, 7):
                cell = QtWidgets.QTableWidgetItem("0" if col < 6 else "0.0")
                cell.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, cell)

    def update_current_regime(self, regime: str):
        """更新当前市场状态显示"""
        from core.market_regime import MarketRegime
        color = MarketRegime.COLORS.get(regime, "#888")
        self.current_regime_label.setText(regime)
        self.current_regime_label.setStyleSheet(
            f"font-size: 18px; font-weight: bold; padding: 8px; color: {color};"
        )

    def update_stats(self, regime_stats: Dict):
        """
        更新统计表格

        Args:
            regime_stats: {regime_str: {count, long, short, wins, losses, win_rate, avg_profit_pct, total_profit}}
        """
        from core.market_regime import MarketRegime
        regimes = MarketRegime.ALL_REGIMES + [MarketRegime.UNKNOWN]

        for row, regime in enumerate(regimes):
            s = regime_stats.get(regime, {})
            count = s.get("count", 0)
            long_n = s.get("long", 0)
            short_n = s.get("short", 0)
            wins = s.get("wins", 0)
            win_rate = s.get("win_rate", 0.0)
            avg_pct = s.get("avg_profit_pct", 0.0)

            # 交易数
            count_item = QtWidgets.QTableWidgetItem(str(count))
            count_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, count_item)

            # 做多
            long_item = QtWidgets.QTableWidgetItem(str(long_n))
            long_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if long_n > 0:
                long_item.setForeground(QtGui.QBrush(QtGui.QColor(UI_CONFIG['CHART_UP_COLOR'])))
            self.table.setItem(row, 2, long_item)

            # 做空
            short_item = QtWidgets.QTableWidgetItem(str(short_n))
            short_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if short_n > 0:
                short_item.setForeground(QtGui.QBrush(QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR'])))
            self.table.setItem(row, 3, short_item)

            # 盈利数
            win_item = QtWidgets.QTableWidgetItem(str(wins))
            win_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, win_item)

            # 正确率%
            rate_str = f"{win_rate * 100:.1f}" if count > 0 else "--"
            rate_item = QtWidgets.QTableWidgetItem(rate_str)
            rate_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if count > 0:
                rate_color = UI_CONFIG['CHART_UP_COLOR'] if win_rate >= 0.5 else UI_CONFIG['CHART_DOWN_COLOR']
                rate_item.setForeground(QtGui.QBrush(QtGui.QColor(rate_color)))
                font = rate_item.font()
                font.setBold(True)
                rate_item.setFont(font)
            self.table.setItem(row, 5, rate_item)

            # 平均收益%
            avg_str = f"{avg_pct:.2f}" if count > 0 else "--"
            avg_item = QtWidgets.QTableWidgetItem(avg_str)
            avg_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if count > 0:
                avg_color = UI_CONFIG['CHART_UP_COLOR'] if avg_pct >= 0 else UI_CONFIG['CHART_DOWN_COLOR']
                avg_item.setForeground(QtGui.QBrush(QtGui.QColor(avg_color)))
            self.table.setItem(row, 6, avg_item)


class TrajectoryMatchWidget(QtWidgets.QWidget):
    """轨迹匹配结果显示组件"""

    # 信号
    batch_wf_requested = QtCore.pyqtSignal()       # 批量Walk-Forward
    batch_wf_stop_requested = QtCore.pyqtSignal()  # 停止批量验证
    save_memory_requested = QtCore.pyqtSignal()
    load_memory_requested = QtCore.pyqtSignal()
    clear_memory_requested = QtCore.pyqtSignal()
    merge_all_requested = QtCore.pyqtSignal()
    apply_template_filter_requested = QtCore.pyqtSignal()  # 应用模板筛选
    # 原型相关信号
    generate_prototypes_requested = QtCore.pyqtSignal(int, int)  # (n_long, n_short)
    load_prototypes_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._batch_last_progress = 0
        self._init_ui()

    def _init_ui(self):
        # 外层布局（仅包含滚动区域）
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # 滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
            }}
            QScrollBar:vertical {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #555;
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #777;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        # 内容容器
        content_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        self._trajectory_content_layout = layout

        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

        group_style = f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
            }}
        """

        # ── 实时匹配状态（已移除UI，保留变量供兼容）──
        self.current_regime_label = QtWidgets.QLabel("--")
        self.best_match_label = QtWidgets.QLabel("--")
        self.cosine_sim_label = QtWidgets.QLabel("--")
        self.dtw_sim_label = QtWidgets.QLabel("--")
        self.match_direction_label = QtWidgets.QLabel("--")

        # ══════════════════════════════════════════════════════════
        # ── 指纹模板库（合并：模板库统计 + 模板评估） ──
        # ══════════════════════════════════════════════════════════
        template_group = QtWidgets.QGroupBox("指纹模板库")
        template_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {UI_CONFIG['THEME_ACCENT']};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 18px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
                font-size: {UI_CONFIG.get('FONT_SIZE_LARGE', 14)}px;
            }}
        """)
        template_layout = QtWidgets.QVBoxLayout(template_group)

        # ── 已验证模板 (核心醒目显示) ──
        verified_frame = QtWidgets.QFrame()
        verified_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #1a2a1a;
                border: 1px solid #2a5a2a;
                border-radius: 5px;
                padding: 8px;
            }}
        """)
        verified_inner = QtWidgets.QVBoxLayout(verified_frame)
        verified_inner.setContentsMargins(8, 6, 8, 6)

        # 标题行
        verified_title = QtWidgets.QLabel("Walk-Forward 已验证模板")
        verified_title.setStyleSheet(f"""
            color: {UI_CONFIG['CHART_UP_COLOR']};
            font-size: {UI_CONFIG.get('FONT_SIZE_LARGE', 14)}px;
            font-weight: bold;
        """)
        verified_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        verified_inner.addWidget(verified_title)

        # LONG / SHORT 数字 — 大字醒目
        verified_counts_layout = QtWidgets.QHBoxLayout()

        # LONG 已验证
        long_box = QtWidgets.QVBoxLayout()
        self.verified_long_count = QtWidgets.QLabel("0")
        self.verified_long_count.setStyleSheet(f"""
            color: {UI_CONFIG['CHART_UP_COLOR']};
            font-size: 28px;
            font-weight: bold;
        """)
        self.verified_long_count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        long_box_title = QtWidgets.QLabel("LONG")
        long_box_title.setStyleSheet(f"color: {UI_CONFIG['CHART_UP_COLOR']}; font-size: 12px;")
        long_box_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        long_box.addWidget(self.verified_long_count)
        long_box.addWidget(long_box_title)
        verified_counts_layout.addLayout(long_box)

        # 分隔
        sep_label = QtWidgets.QLabel("|")
        sep_label.setStyleSheet("color: #555; font-size: 28px;")
        sep_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sep_label.setFixedWidth(20)
        verified_counts_layout.addWidget(sep_label)

        # SHORT 已验证
        short_box = QtWidgets.QVBoxLayout()
        self.verified_short_count = QtWidgets.QLabel("0")
        self.verified_short_count.setStyleSheet(f"""
            color: {UI_CONFIG['CHART_DOWN_COLOR']};
            font-size: 28px;
            font-weight: bold;
        """)
        self.verified_short_count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        short_box_title = QtWidgets.QLabel("SHORT")
        short_box_title.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']}; font-size: 12px;")
        short_box_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        short_box.addWidget(self.verified_short_count)
        short_box.addWidget(short_box_title)
        verified_counts_layout.addLayout(short_box)

        verified_inner.addLayout(verified_counts_layout)

        # 验证状态行（评级详情）
        verified_detail_layout = QtWidgets.QHBoxLayout()
        self.eval_excellent_label = QtWidgets.QLabel("0")
        self.eval_excellent_label.setStyleSheet(f"color: {UI_CONFIG['CHART_UP_COLOR']}; font-weight: bold;")
        self.eval_qualified_label = QtWidgets.QLabel("0")
        self.eval_qualified_label.setStyleSheet(f"color: {UI_CONFIG['THEME_ACCENT']}; font-weight: bold;")
        self.eval_pending_label = QtWidgets.QLabel("0")
        self.eval_pending_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.eval_eliminated_label = QtWidgets.QLabel("0")
        self.eval_eliminated_label.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']}; font-weight: bold;")

        for lbl_text, lbl_widget in [("优质:", self.eval_excellent_label),
                                      ("合格:", self.eval_qualified_label),
                                      ("待观:", self.eval_pending_label),
                                      ("淘汰:", self.eval_eliminated_label)]:
            mini = QtWidgets.QHBoxLayout()
            tag = QtWidgets.QLabel(lbl_text)
            tag.setStyleSheet("color: #aaa; font-size: 11px;")
            mini.addWidget(tag)
            mini.addWidget(lbl_widget)
            verified_detail_layout.addLayout(mini)

        verified_inner.addLayout(verified_detail_layout)

        template_layout.addWidget(verified_frame)

        # ── 记忆库概况（次要信息） ──
        stats_layout = QtWidgets.QHBoxLayout()
        stats_left = QtWidgets.QFormLayout()
        stats_right = QtWidgets.QFormLayout()

        self.total_templates_label = QtWidgets.QLabel("0")
        self.long_templates_label = QtWidgets.QLabel("0")
        self.short_templates_label = QtWidgets.QLabel("0")
        self.avg_profit_label = QtWidgets.QLabel("--")
        self.eval_total_label = QtWidgets.QLabel("--")
        self.eval_evaluated_label = QtWidgets.QLabel("--")

        stats_left.addRow("记忆库总数:", self.total_templates_label)
        stats_left.addRow("做多:", self.long_templates_label)
        stats_right.addRow("做空:", self.short_templates_label)
        stats_right.addRow("已匹配:", self.eval_evaluated_label)

        stats_layout.addLayout(stats_left)
        stats_layout.addLayout(stats_right)
        template_layout.addLayout(stats_layout)

        # Top模板表格
        self.eval_table = QtWidgets.QTableWidget()
        self.eval_table.setColumnCount(5)
        self.eval_table.setHorizontalHeaderLabels(["方向", "状态", "匹配", "胜率", "评级"])
        self.eval_table.horizontalHeader().setStretchLastSection(True)
        self.eval_table.setMinimumHeight(80)
        self.eval_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.eval_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.eval_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #444;
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: #333;
                color: {UI_CONFIG['THEME_TEXT']};
                padding: 3px;
                border: 1px solid #444;
            }}
        """)
        self.eval_table.setVisible(False)
        template_layout.addWidget(self.eval_table)

        # 应用筛选按钮 + 保存按钮（同一行）
        filter_btn_layout = QtWidgets.QHBoxLayout()
        self.apply_filter_btn = QtWidgets.QPushButton("应用筛选 (淘汰差模板)")
        self.apply_filter_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #aa6600;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: #cc8800;
            }}
            QPushButton:disabled {{
                background-color: #555;
                color: #888;
            }}
        """)
        self.apply_filter_btn.setEnabled(False)
        self.apply_filter_btn.setToolTip("删除所有评级为'淘汰'的模板，保留优质/合格/待观察，并自动保存")
        self.apply_filter_btn.clicked.connect(self.apply_template_filter_requested.emit)
        filter_btn_layout.addWidget(self.apply_filter_btn)
        template_layout.addLayout(filter_btn_layout)

        # 评级标准说明
        eval_note = QtWidgets.QLabel("评级: 匹配>=3次 & 胜率>=60% => 优质/合格 | <3次 => 待观察 | 胜率<60% => 淘汰")
        eval_note.setStyleSheet("color: #666; font-size: 10px;")
        eval_note.setWordWrap(True)
        template_layout.addWidget(eval_note)

        layout.addWidget(template_group)

        # ══════════════════════════════════════════════════════════
        # ── 原型库（聚类后的交易模式） ──
        # ══════════════════════════════════════════════════════════
        prototype_group = QtWidgets.QGroupBox("原型库（聚类模式）")
        prototype_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid #9933cc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 18px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
                font-size: {UI_CONFIG.get('FONT_SIZE_LARGE', 14)}px;
            }}
        """)
        proto_layout = QtWidgets.QVBoxLayout(prototype_group)

        # 原型统计
        proto_stats_frame = QtWidgets.QFrame()
        proto_stats_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2a1a3a;
                border: 1px solid #4a2a6a;
                border-radius: 5px;
                padding: 6px;
            }}
        """)
        proto_stats_inner = QtWidgets.QVBoxLayout(proto_stats_frame)
        proto_stats_inner.setContentsMargins(8, 6, 8, 6)

        # 原型数量显示
        proto_counts_layout = QtWidgets.QHBoxLayout()

        # LONG 原型
        proto_long_box = QtWidgets.QVBoxLayout()
        self.proto_long_count = QtWidgets.QLabel("0")
        self.proto_long_count.setStyleSheet(f"""
            color: {UI_CONFIG['CHART_UP_COLOR']};
            font-size: 24px;
            font-weight: bold;
        """)
        self.proto_long_count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        proto_long_title = QtWidgets.QLabel("LONG原型")
        proto_long_title.setStyleSheet(f"color: {UI_CONFIG['CHART_UP_COLOR']}; font-size: 11px;")
        proto_long_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        proto_long_box.addWidget(self.proto_long_count)
        proto_long_box.addWidget(proto_long_title)
        proto_counts_layout.addLayout(proto_long_box)

        proto_sep = QtWidgets.QLabel("|")
        proto_sep.setStyleSheet("color: #555; font-size: 24px;")
        proto_sep.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        proto_sep.setFixedWidth(20)
        proto_counts_layout.addWidget(proto_sep)

        # SHORT 原型
        proto_short_box = QtWidgets.QVBoxLayout()
        self.proto_short_count = QtWidgets.QLabel("0")
        self.proto_short_count.setStyleSheet(f"""
            color: {UI_CONFIG['CHART_DOWN_COLOR']};
            font-size: 24px;
            font-weight: bold;
        """)
        self.proto_short_count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        proto_short_title = QtWidgets.QLabel("SHORT原型")
        proto_short_title.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']}; font-size: 11px;")
        proto_short_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        proto_short_box.addWidget(self.proto_short_count)
        proto_short_box.addWidget(proto_short_title)
        proto_counts_layout.addLayout(proto_short_box)

        proto_stats_inner.addLayout(proto_counts_layout)

        # 原型详情
        proto_detail_layout = QtWidgets.QHBoxLayout()
        self.proto_source_label = QtWidgets.QLabel("来源: 0 模板")
        self.proto_source_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.proto_avg_winrate_label = QtWidgets.QLabel("平均胜率: --")
        self.proto_avg_winrate_label.setStyleSheet("color: #aaa; font-size: 11px;")
        proto_detail_layout.addWidget(self.proto_source_label)
        proto_detail_layout.addStretch()
        proto_detail_layout.addWidget(self.proto_avg_winrate_label)
        proto_stats_inner.addLayout(proto_detail_layout)

        # WF评级分布行
        wf_dist_layout = QtWidgets.QHBoxLayout()
        wf_dist_layout.setContentsMargins(0, 4, 0, 0)
        
        # 合格
        self.proto_qualified_label = QtWidgets.QLabel("合格: 0")
        self.proto_qualified_label.setStyleSheet("color: #089981; font-weight: bold; font-size: 11px;")
        wf_dist_layout.addWidget(self.proto_qualified_label)
        wf_dist_layout.addSpacing(10)
        
        # 待观察
        self.proto_pending_label = QtWidgets.QLabel("待观察: 0")
        self.proto_pending_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 11px;")
        wf_dist_layout.addWidget(self.proto_pending_label)
        wf_dist_layout.addSpacing(10)
        
        # 淘汰
        self.proto_eliminated_label = QtWidgets.QLabel("淘汰: 0")
        self.proto_eliminated_label.setStyleSheet("color: #f23645; font-weight: bold; font-size: 11px;")
        wf_dist_layout.addWidget(self.proto_eliminated_label)
        
        wf_dist_layout.addStretch()
        proto_stats_inner.addLayout(wf_dist_layout)

        proto_layout.addWidget(proto_stats_frame)

        # 聚类参数行
        proto_params_layout = QtWidgets.QHBoxLayout()
        proto_params_layout.addWidget(QtWidgets.QLabel("LONG:"))
        self.proto_n_long_spin = QtWidgets.QSpinBox()
        self.proto_n_long_spin.setRange(5, 100)
        self.proto_n_long_spin.setValue(30)
        self.proto_n_long_spin.setFixedWidth(55)
        self.proto_n_long_spin.setToolTip("LONG 方向聚类数")
        proto_params_layout.addWidget(self.proto_n_long_spin)

        proto_params_layout.addWidget(QtWidgets.QLabel("SHORT:"))
        self.proto_n_short_spin = QtWidgets.QSpinBox()
        self.proto_n_short_spin.setRange(5, 100)
        self.proto_n_short_spin.setValue(30)
        self.proto_n_short_spin.setFixedWidth(55)
        self.proto_n_short_spin.setToolTip("SHORT 方向聚类数")
        proto_params_layout.addWidget(self.proto_n_short_spin)
        proto_params_layout.addStretch()
        proto_layout.addLayout(proto_params_layout)

        # 按钮行
        proto_btn_layout = QtWidgets.QHBoxLayout()

        self.generate_proto_btn = QtWidgets.QPushButton("生成原型")
        self.generate_proto_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #9933cc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #aa44dd;
            }}
            QPushButton:disabled {{
                background-color: #555;
                color: #888;
            }}
        """)
        self.generate_proto_btn.setEnabled(False)
        self.generate_proto_btn.setToolTip("对模板库进行 K-Means 聚类，生成交易原型")
        self.generate_proto_btn.clicked.connect(self._on_generate_prototypes_clicked)
        proto_btn_layout.addWidget(self.generate_proto_btn)

        self.load_proto_btn = QtWidgets.QPushButton("加载原型")
        self.load_proto_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #444;
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #555;
                padding: 8px 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)
        self.load_proto_btn.clicked.connect(self.load_prototypes_requested.emit)
        proto_btn_layout.addWidget(self.load_proto_btn)

        proto_layout.addLayout(proto_btn_layout)

        # 原型列表表格
        self.proto_table = QtWidgets.QTableWidget()
        self.proto_table.setColumnCount(7)
        self.proto_table.setHorizontalHeaderLabels(["方向", "成员", "胜率", "平均收益", "持仓", "WF验证", "ID"])
        self.proto_table.horizontalHeader().setStretchLastSection(True)
        self.proto_table.setMinimumHeight(80)
        self.proto_table.setMaximumHeight(150)
        self.proto_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.proto_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #444;
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: #333;
                color: {UI_CONFIG['THEME_TEXT']};
                padding: 3px;
                border: 1px solid #444;
            }}
        """)
        self.proto_table.setVisible(False)
        proto_layout.addWidget(self.proto_table)

        # 说明
        proto_note = QtWidgets.QLabel("聚类将模板压缩为少量原型，提升匹配效率和统计可靠性")
        proto_note.setStyleSheet("color: #666; font-size: 10px;")
        proto_note.setWordWrap(True)
        proto_layout.addWidget(proto_note)

        layout.addWidget(prototype_group)

        # ══════════════════════════════════════════════════════════
        # ── 批量 Walk-Forward 验证 ──
        # ══════════════════════════════════════════════════════════
        batch_group = QtWidgets.QGroupBox("批量 Walk-Forward 验证")
        batch_group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid #cc8800;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 18px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
                font-size: {UI_CONFIG.get('FONT_SIZE_LARGE', 14)}px;
            }}
        """)
        batch_layout = QtWidgets.QVBoxLayout(batch_group)

        # 参数设置行
        batch_params_layout = QtWidgets.QHBoxLayout()

        batch_params_layout.addWidget(QtWidgets.QLabel("轮数:"))
        self.batch_rounds_spin = QtWidgets.QSpinBox()
        self.batch_rounds_spin.setRange(1, 100)
        from config import WALK_FORWARD_CONFIG
        default_rounds = WALK_FORWARD_CONFIG.get("BATCH_DEFAULT_ROUNDS", 20)
        self.batch_rounds_spin.setValue(default_rounds)
        self.batch_rounds_spin.setFixedWidth(60)
        self.batch_rounds_spin.setToolTip("验证轮数（每轮采样不同数据段，建议≥20轮）")
        batch_params_layout.addWidget(self.batch_rounds_spin)

        batch_params_layout.addWidget(QtWidgets.QLabel("采样:"))
        self.batch_sample_spin = QtWidgets.QSpinBox()
        self.batch_sample_spin.setRange(10000, 200000)
        self.batch_sample_spin.setValue(50000)
        self.batch_sample_spin.setSingleStep(10000)
        self.batch_sample_spin.setFixedWidth(80)
        self.batch_sample_spin.setToolTip("每轮采样K线数")
        self.batch_sample_spin.setSuffix("K线")
        batch_params_layout.addWidget(self.batch_sample_spin)

        batch_params_layout.addStretch()
        batch_layout.addLayout(batch_params_layout)

        # 启动/停止按钮行
        batch_btn_layout = QtWidgets.QHBoxLayout()

        self.batch_wf_btn = QtWidgets.QPushButton("一键批量验证")
        self.batch_wf_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #cc8800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: {UI_CONFIG.get('FONT_SIZE_LARGE', 14)}px;
            }}
            QPushButton:hover {{
                background-color: #ee9900;
            }}
            QPushButton:disabled {{
                background-color: #555;
                color: #888;
            }}
        """)
        self.batch_wf_btn.setEnabled(False)
        self.batch_wf_btn.clicked.connect(self.batch_wf_requested.emit)
        batch_btn_layout.addWidget(self.batch_wf_btn)

        self.batch_stop_btn = QtWidgets.QPushButton("停止")
        self.batch_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #aa3333;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #cc4444;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.batch_stop_btn.setEnabled(False)
        self.batch_stop_btn.clicked.connect(self.batch_wf_stop_requested.emit)
        batch_btn_layout.addWidget(self.batch_stop_btn)

        batch_layout.addLayout(batch_btn_layout)

        # 进度条
        self.batch_progress_bar = QtWidgets.QProgressBar()
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QProgressBar::chunk {{
                background-color: #cc8800;
            }}
        """)
        batch_layout.addWidget(self.batch_progress_bar)

        # 实时计数器 — 醒目大字
        batch_counter_frame = QtWidgets.QFrame()
        batch_counter_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2a2a1a;
                border: 1px solid #5a5a2a;
                border-radius: 5px;
                padding: 6px;
            }}
        """)
        batch_counter_inner = QtWidgets.QVBoxLayout(batch_counter_frame)
        batch_counter_inner.setContentsMargins(8, 4, 8, 4)

        # 轮次 + ETA
        batch_status_layout = QtWidgets.QHBoxLayout()
        self.batch_round_label = QtWidgets.QLabel("待启动")
        self.batch_round_label.setStyleSheet(f"color: #cc8800; font-size: 12px;")
        self.batch_eta_label = QtWidgets.QLabel("")
        self.batch_eta_label.setStyleSheet("color: #888; font-size: 11px;")
        batch_status_layout.addWidget(self.batch_round_label)
        batch_status_layout.addStretch()
        batch_status_layout.addWidget(self.batch_eta_label)
        batch_counter_inner.addLayout(batch_status_layout)

        # 累计匹配 + 涉及原型
        batch_match_layout = QtWidgets.QHBoxLayout()
        match_tag = QtWidgets.QLabel("累计匹配:")
        match_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_match_count_label = QtWidgets.QLabel("0")
        self.batch_match_count_label.setStyleSheet("color: #cc8800; font-weight: bold; font-size: 13px;")
        unique_tag = QtWidgets.QLabel("涉及原型:")
        unique_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_unique_label = QtWidgets.QLabel("0")
        self.batch_unique_label.setStyleSheet("color: #cc8800; font-weight: bold; font-size: 13px;")
        batch_match_layout.addWidget(match_tag)
        batch_match_layout.addWidget(self.batch_match_count_label)
        batch_match_layout.addSpacing(10)
        batch_match_layout.addWidget(unique_tag)
        batch_match_layout.addWidget(self.batch_unique_label)
        batch_match_layout.addStretch()
        batch_counter_inner.addLayout(batch_match_layout)

        # 4个评级分类计数
        batch_rating_layout = QtWidgets.QHBoxLayout()
        
        qualified_tag = QtWidgets.QLabel("合格:")
        qualified_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_qualified_label = QtWidgets.QLabel("0")
        self.batch_qualified_label.setStyleSheet("color: #089981; font-weight: bold; font-size: 14px;")
        
        pending_tag = QtWidgets.QLabel("待观察:")
        pending_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_pending_label = QtWidgets.QLabel("0")
        self.batch_pending_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        
        eliminated_tag = QtWidgets.QLabel("淘汰:")
        eliminated_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_eliminated_label = QtWidgets.QLabel("0")
        self.batch_eliminated_label.setStyleSheet("color: #f23645; font-weight: bold; font-size: 14px;")

        batch_rating_layout.addWidget(qualified_tag)
        batch_rating_layout.addWidget(self.batch_qualified_label)
        batch_rating_layout.addSpacing(10)
        batch_rating_layout.addWidget(pending_tag)
        batch_rating_layout.addWidget(self.batch_pending_label)
        batch_rating_layout.addSpacing(10)
        batch_rating_layout.addWidget(eliminated_tag)
        batch_rating_layout.addWidget(self.batch_eliminated_label)
        batch_rating_layout.addStretch()
        batch_counter_inner.addLayout(batch_rating_layout)

        # 本轮信息（Sharpe）
        batch_round_info = QtWidgets.QHBoxLayout()
        round_trades_tag = QtWidgets.QLabel("本轮交易:")
        round_trades_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_round_trades_label = QtWidgets.QLabel("0")
        self.batch_round_trades_label.setStyleSheet("color: #ccc; font-size: 12px;")
        round_sharpe_tag = QtWidgets.QLabel("Sharpe:")
        round_sharpe_tag.setStyleSheet("color: #aaa; font-size: 11px;")
        self.batch_round_sharpe_label = QtWidgets.QLabel("0.000")
        self.batch_round_sharpe_label.setStyleSheet("color: #ccc; font-size: 12px;")
        batch_round_info.addWidget(round_trades_tag)
        batch_round_info.addWidget(self.batch_round_trades_label)
        batch_round_info.addSpacing(15)
        batch_round_info.addWidget(round_sharpe_tag)
        batch_round_info.addWidget(self.batch_round_sharpe_label)
        batch_round_info.addStretch()
        batch_counter_inner.addLayout(batch_round_info)

        batch_layout.addWidget(batch_counter_frame)

        batch_note = QtWidgets.QLabel("直接用全局模板库在多段随机数据上验证，无需重新标注")
        batch_note.setStyleSheet("color: #666; font-size: 10px;")
        batch_note.setWordWrap(True)
        batch_layout.addWidget(batch_note)

        layout.addWidget(batch_group)

        # ── 最优交易参数 ──
        self.params_group = QtWidgets.QGroupBox("优化参数")
        self.params_group.setStyleSheet(group_style)
        params_layout = QtWidgets.QFormLayout(self.params_group)

        self.param_labels = {}
        param_names = [
            ("cosine_threshold", "余弦阈值"),
            ("dtw_threshold", "DTW阈值"),
            ("min_templates_agree", "最少模板"),
            ("stop_loss_atr", "止损ATR"),
            ("take_profit_atr", "止盈ATR"),
            ("max_hold_bars", "最大持仓"),
            ("hold_divergence_limit", "偏离上限"),
        ]
        for key, name in param_names:
            label = QtWidgets.QLabel("--")
            self.param_labels[key] = label
            params_layout.addRow(f"{name}:", label)

        layout.addWidget(self.params_group)

        # ── 记忆管理 ──
        self.memory_group = QtWidgets.QGroupBox("记忆管理")
        self.memory_group.setStyleSheet(group_style)
        memory_layout = QtWidgets.QVBoxLayout(self.memory_group)

        # 记忆统计
        memory_stats_layout = QtWidgets.QFormLayout()
        self.memory_count_label = QtWidgets.QLabel("0")
        self.memory_files_label = QtWidgets.QLabel("0 个文件")
        memory_stats_layout.addRow("已加载模板:", self.memory_count_label)
        memory_stats_layout.addRow("本地存档:", self.memory_files_label)
        memory_layout.addLayout(memory_stats_layout)

        # 按钮样式
        btn_style = f"""
            QPushButton {{
                background-color: #444;
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:disabled {{
                background-color: #333;
                color: #666;
            }}
        """
        btn_style_accent = f"""
            QPushButton {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: #0088dd;
            }}
            QPushButton:disabled {{
                background-color: #555;
                color: #888;
            }}
        """
        btn_style_danger = """
            QPushButton {
                background-color: #aa3333;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #cc4444;
            }
        """

        # 第一行：保存 + 加载
        btn_row1 = QtWidgets.QHBoxLayout()
        self.save_memory_btn = QtWidgets.QPushButton("保存记忆")
        self.save_memory_btn.setStyleSheet(btn_style_accent)
        self.save_memory_btn.setEnabled(False)
        self.save_memory_btn.clicked.connect(self.save_memory_requested.emit)
        btn_row1.addWidget(self.save_memory_btn)

        self.load_memory_btn = QtWidgets.QPushButton("加载最新")
        self.load_memory_btn.setStyleSheet(btn_style)
        self.load_memory_btn.clicked.connect(self.load_memory_requested.emit)
        btn_row1.addWidget(self.load_memory_btn)
        memory_layout.addLayout(btn_row1)

        # 第二行：合并全部 + 清空
        btn_row2 = QtWidgets.QHBoxLayout()
        self.merge_all_btn = QtWidgets.QPushButton("合并全部")
        self.merge_all_btn.setStyleSheet(btn_style)
        self.merge_all_btn.setToolTip("加载并合并所有历史记忆文件")
        self.merge_all_btn.clicked.connect(self.merge_all_requested.emit)
        btn_row2.addWidget(self.merge_all_btn)

        self.clear_memory_btn = QtWidgets.QPushButton("清空记忆")
        self.clear_memory_btn.setStyleSheet(btn_style_danger)
        self.clear_memory_btn.clicked.connect(self.clear_memory_requested.emit)
        btn_row2.addWidget(self.clear_memory_btn)
        memory_layout.addLayout(btn_row2)

        layout.addWidget(self.memory_group)

        layout.addStretch()

    def extract_bottom_tools_widget(self) -> QtWidgets.QWidget:
        """
        提取“优化参数 + 记忆管理”区域，用于移动到左下角控制区。
        """
        container = QtWidgets.QGroupBox("优化与记忆")
        container.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: {UI_CONFIG['THEME_TEXT']};
                font-weight: bold;
            }}
        """)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        for name in ("params_group", "memory_group"):
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                self._trajectory_content_layout.removeWidget(w)
            except Exception:
                pass
            w.setParent(container)
            v.addWidget(w)

        return container

    def update_match_status(self, regime: str, best_match: str,
                            cosine_sim: float, dtw_sim: float, direction: str):
        """更新当前匹配状态"""
        self.current_regime_label.setText(regime)
        self.best_match_label.setText(best_match)
        self.cosine_sim_label.setText(f"{cosine_sim:.1%}")
        self.dtw_sim_label.setText(f"{dtw_sim:.1%}")

        # 方向着色
        self.match_direction_label.setText(direction)
        if direction == "LONG":
            self.match_direction_label.setStyleSheet(f"color: {UI_CONFIG['CHART_UP_COLOR']};")
        elif direction == "SHORT":
            self.match_direction_label.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']};")
        else:
            self.match_direction_label.setStyleSheet("color: #888;")

    def update_template_stats(self, total: int, long_count: int, short_count: int,
                               avg_profit: float):
        """更新模板库统计"""
        self.total_templates_label.setText(str(total))
        self.long_templates_label.setText(str(long_count))
        self.short_templates_label.setText(str(short_count))
        self.avg_profit_label.setText(f"{avg_profit:.2f}%")

    def update_trading_params(self, params):
        """更新GA优化后的交易参数"""
        if params is None:
            return
        for key, label in self.param_labels.items():
            if hasattr(params, key):
                value = getattr(params, key)
                if isinstance(value, float):
                    label.setText(f"{value:.3f}")
                else:
                    label.setText(str(value))

    def update_memory_stats(self, template_count: int, file_count: int):
        """更新记忆统计"""
        self.memory_count_label.setText(str(template_count))
        self.memory_files_label.setText(f"{file_count} 个文件")
        # 有模板时启用保存按钮
        self.save_memory_btn.setEnabled(template_count > 0)

    def enable_save_memory(self, enabled: bool):
        """启用/禁用保存按钮"""
        self.save_memory_btn.setEnabled(enabled)

    def update_template_evaluation(self, eval_result):
        """
        更新模板评估结果（合并到指纹模板库区块）

        Args:
            eval_result: EvaluationResult 实例
        """
        if eval_result is None:
            return

        # ── 计算已验证模板的 LONG / SHORT 数量 ──
        # "已验证" = 优质 + 合格 + 待观察（即保留的模板）
        verified_long = 0
        verified_short = 0
        for perf in eval_result.performances:
            if perf.grade in ("优质", "合格", "待观察"):
                if perf.template.direction == "LONG":
                    verified_long += 1
                else:
                    verified_short += 1

        # 更新醒目的 LONG/SHORT 已验证数量
        self.verified_long_count.setText(str(verified_long))
        self.verified_short_count.setText(str(verified_short))

        # 更新评估统计标签
        self.eval_total_label.setText(str(eval_result.total_templates))
        self.eval_evaluated_label.setText(str(eval_result.evaluated_templates))

        # 优质 - 绿色
        self.eval_excellent_label.setText(str(eval_result.excellent_count))
        self.eval_excellent_label.setStyleSheet(f"color: {UI_CONFIG['CHART_UP_COLOR']}; font-weight: bold;")

        # 合格 - 蓝色
        self.eval_qualified_label.setText(str(eval_result.qualified_count))
        self.eval_qualified_label.setStyleSheet(f"color: {UI_CONFIG['THEME_ACCENT']}; font-weight: bold;")

        # 待观察 - 黄色
        self.eval_pending_label.setText(str(eval_result.pending_count))
        self.eval_pending_label.setStyleSheet("color: #ffaa00; font-weight: bold;")

        # 淘汰 - 红色
        self.eval_eliminated_label.setText(str(eval_result.eliminated_count))
        self.eval_eliminated_label.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']}; font-weight: bold;")

        # 更新表格 - 只显示优质+合格+待观察模板（有价值的）
        self.eval_table.setVisible(True)
        self.eval_table.setRowCount(0)

        # 筛选有价值的模板（优质、合格、待观察），排在前面
        valuable = [p for p in eval_result.performances if p.grade in ("优质", "合格", "待观察")]
        # 再补充几条淘汰的，供参考
        eliminated = [p for p in eval_result.performances if p.grade == "淘汰"]
        display_list = valuable[:20] + eliminated[:5]

        for i, perf in enumerate(display_list):
            self.eval_table.insertRow(i)

            # 方向
            direction = perf.template.direction
            dir_item = QtWidgets.QTableWidgetItem(direction)
            if direction == "LONG":
                dir_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_UP_COLOR']))
            else:
                dir_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR']))
            self.eval_table.setItem(i, 0, dir_item)

            # 市场状态
            regime_item = QtWidgets.QTableWidgetItem(perf.template.regime[:6])
            self.eval_table.setItem(i, 1, regime_item)

            # 匹配次数
            match_item = QtWidgets.QTableWidgetItem(str(perf.match_count))
            match_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.eval_table.setItem(i, 2, match_item)

            # 胜率
            winrate_item = QtWidgets.QTableWidgetItem(f"{perf.win_rate:.0%}")
            winrate_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if perf.win_rate >= 0.6:
                winrate_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_UP_COLOR']))
            elif perf.win_rate < 0.4:
                winrate_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR']))
            self.eval_table.setItem(i, 3, winrate_item)

            # 评级
            grade_item = QtWidgets.QTableWidgetItem(perf.grade)
            grade_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if perf.grade == "优质":
                grade_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_UP_COLOR']))
                grade_item.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            elif perf.grade == "合格":
                grade_item.setForeground(QtGui.QColor(UI_CONFIG['THEME_ACCENT']))
            elif perf.grade == "待观察":
                grade_item.setForeground(QtGui.QColor("#ffaa00"))
            else:
                grade_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR']))
            self.eval_table.setItem(i, 4, grade_item)

        # 如果有淘汰的模板，启用筛选按钮
        self.apply_filter_btn.setEnabled(eval_result.eliminated_count > 0)

    def reset_template_evaluation(self):
        """重置模板评估显示"""
        self.verified_long_count.setText("0")
        self.verified_short_count.setText("0")
        self.eval_total_label.setText("--")
        self.eval_evaluated_label.setText("--")
        self.eval_excellent_label.setText("0")
        self.eval_qualified_label.setText("0")
        self.eval_pending_label.setText("0")
        self.eval_eliminated_label.setText("0")
        self.eval_table.setRowCount(0)
        self.eval_table.setVisible(False)
        self.apply_filter_btn.setEnabled(False)

    # ── 原型库方法 ──

    def _on_generate_prototypes_clicked(self):
        """生成原型按钮点击"""
        n_long = self.proto_n_long_spin.value()
        n_short = self.proto_n_short_spin.value()
        self.generate_prototypes_requested.emit(n_long, n_short)

    def update_prototype_stats(self, library):
        """
        更新原型库统计显示
        
        Args:
            library: PrototypeLibrary 实例
        """
        if library is None:
            self.proto_long_count.setText("0")
            self.proto_short_count.setText("0")
            self.proto_source_label.setText("来源: 0 模板")
            self.proto_avg_winrate_label.setText("平均胜率: --")
            self.proto_table.setRowCount(0)
            self.proto_table.setVisible(False)
            return

        # 更新统计数字
        n_long = len(library.long_prototypes)
        n_short = len(library.short_prototypes)
        self.proto_long_count.setText(str(n_long))
        self.proto_short_count.setText(str(n_short))
        self.proto_source_label.setText(f"来源: {library.source_template_count} 模板")

        # 计算平均胜率和评级分布
        all_protos = library.get_all_prototypes()
        n_excellent = 0
        n_qualified = 0
        n_pending = 0
        n_eliminated = 0
        
        if all_protos:
            avg_win = sum(p.win_rate for p in all_protos) / len(all_protos)
            self.proto_avg_winrate_label.setText(f"平均胜率: {avg_win:.1%}")
            
            # 统计 WF 评级
            for p in all_protos:
                g = getattr(p, "wf_grade", "")
                if g == "优质": n_excellent += 1
                elif g == "合格": n_qualified += 1
                elif g == "待观察": n_pending += 1
                elif g == "淘汰": n_eliminated += 1
                # 如果没有wf_grade，默认为待观察（如果是新生成的话）或者不统计
                # 这里主要统计已经标记的
            
            # 优质合并入合格显示，或者单独显示？UI只加了3个label，优质算合格的一种
            n_qualified += n_excellent 
            
            self.proto_qualified_label.setText(f"合格: {n_qualified}")
            self.proto_pending_label.setText(f"待观察: {n_pending}")
            self.proto_eliminated_label.setText(f"淘汰: {n_eliminated}")
            
        else:
            self.proto_avg_winrate_label.setText("平均胜率: --")
            self.proto_qualified_label.setText("合格: 0")
            self.proto_pending_label.setText("待观察: 0")
            self.proto_eliminated_label.setText("淘汰: 0")

        # 更新表格（显示前10个）
        display_protos = sorted(all_protos, key=lambda p: p.member_count, reverse=True)[:10]
        self.proto_table.setRowCount(len(display_protos))

        for i, proto in enumerate(display_protos):
            # 方向
            dir_item = QtWidgets.QTableWidgetItem(proto.direction)
            if proto.direction == "LONG":
                dir_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_UP_COLOR']))
            else:
                dir_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR']))
            self.proto_table.setItem(i, 0, dir_item)

            # 成员数
            member_item = QtWidgets.QTableWidgetItem(str(proto.member_count))
            member_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.proto_table.setItem(i, 1, member_item)

            # 胜率
            winrate_item = QtWidgets.QTableWidgetItem(f"{proto.win_rate:.0%}")
            winrate_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if proto.win_rate >= 0.6:
                winrate_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_UP_COLOR']))
            elif proto.win_rate < 0.4:
                winrate_item.setForeground(QtGui.QColor(UI_CONFIG['CHART_DOWN_COLOR']))
            self.proto_table.setItem(i, 2, winrate_item)

            # 平均收益
            profit_item = QtWidgets.QTableWidgetItem(f"{proto.avg_profit_pct:.2f}%")
            profit_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.proto_table.setItem(i, 3, profit_item)

            # 平均持仓
            hold_item = QtWidgets.QTableWidgetItem(f"{proto.avg_hold_bars:.0f}")
            hold_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.proto_table.setItem(i, 4, hold_item)

            # WF验证状态
            grade = getattr(proto, 'wf_grade', '')
            if grade == "合格":
                grade_item = QtWidgets.QTableWidgetItem("合格")
                grade_item.setForeground(QtGui.QColor("#089981"))
            elif grade == "待观察":
                grade_item = QtWidgets.QTableWidgetItem("待观察")
                grade_item.setForeground(QtGui.QColor("#ffaa00"))
            elif grade == "淘汰":
                grade_item = QtWidgets.QTableWidgetItem("淘汰")
                grade_item.setForeground(QtGui.QColor("#f23645"))
            else:
                grade_item = QtWidgets.QTableWidgetItem("-")
                grade_item.setForeground(QtGui.QColor("#666"))
            grade_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.proto_table.setItem(i, 5, grade_item)

            # ID
            id_item = QtWidgets.QTableWidgetItem(str(proto.prototype_id))
            id_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.proto_table.setItem(i, 6, id_item)


        self.proto_table.setVisible(len(display_protos) > 0)

    def enable_generate_prototypes(self, enabled: bool):
        """启用/禁用生成原型按钮"""
        self.generate_proto_btn.setEnabled(enabled)

    def reset_prototype_stats(self):
        """重置原型库显示"""
        self.proto_long_count.setText("0")
        self.proto_short_count.setText("0")
        self.proto_source_label.setText("来源: 0 模板")
        self.proto_avg_winrate_label.setText("平均胜率: --")
        self.proto_qualified_label.setText("合格: 0")
        self.proto_pending_label.setText("待观察: 0")
        self.proto_eliminated_label.setText("淘汰: 0")
        self.proto_table.setRowCount(0)
        self.proto_table.setVisible(False)

    # ── 批量 Walk-Forward 方法 ──

    def update_batch_wf_progress(self, round_idx: int, n_rounds: int,
                                  cumulative_stats: dict):
        """更新批量WF进度（每轮完成后调用）"""
        # 检查是否是"正在运行"状态（轮次开始时发送）
        is_running = cumulative_stats.get("running", False)
        global_pct = cumulative_stats.get("global_progress_pct", None)
        
        # 进度条
        self.batch_progress_bar.setVisible(True)
        
        if is_running:
            # 轮次运行中：支持trial级细粒度进度
            phase = cumulative_stats.get("phase", "")
            trial_idx = int(cumulative_stats.get("trial_idx", 0))
            trial_total = max(1, int(cumulative_stats.get("trial_total", 1)))
            
            if phase == "bayes_opt":
                if global_pct is not None:
                    pct = int(max(self._batch_last_progress, min(100, global_pct)))
                else:
                    frac = min(1.0, max(0.0, trial_idx / trial_total))
                    pct = int(((round_idx + frac) / n_rounds) * 100)
                self.batch_progress_bar.setFormat(
                    f"第 {round_idx + 1}/{n_rounds} 轮优化中: trial {trial_idx}/{trial_total} ({pct}%)"
                )
                self.batch_round_label.setText(
                    f"第 {round_idx + 1} 轮运行中（贝叶斯优化 {trial_idx}/{trial_total}）"
                )
            else:
                if global_pct is not None:
                    pct = int(max(self._batch_last_progress, min(100, global_pct)))
                else:
                    pct = int(round_idx / n_rounds * 100)  # 还没完成这一轮
                self.batch_progress_bar.setFormat(f"第 {round_idx + 1}/{n_rounds} 轮运行中... ({pct}%)")
                self.batch_round_label.setText(f"第 {round_idx + 1} 轮运行中...")
            
            self._batch_last_progress = max(self._batch_last_progress, pct)
            self.batch_progress_bar.setValue(pct)
            self.batch_eta_label.setText("计算中...")
        else:
            # 轮次完成
            if global_pct is not None:
                pct = int(max(self._batch_last_progress, min(100, global_pct)))
            else:
                pct = int((round_idx + 1) / n_rounds * 100)
            self._batch_last_progress = max(self._batch_last_progress, pct)
            self.batch_progress_bar.setValue(pct)

            # ETA
            eta = cumulative_stats.get("eta_seconds", 0)
            if eta > 60:
                eta_str = f"剩余 {int(eta // 60)}分{int(eta % 60)}秒"
            else:
                eta_str = f"剩余 {int(eta)}秒"
            self.batch_progress_bar.setFormat(f"Round {round_idx + 1}/{n_rounds} ({pct}%) | {eta_str}")

            # 轮次
            self.batch_round_label.setText(f"Round {round_idx + 1} / {n_rounds}")
            self.batch_eta_label.setText(eta_str)

        # 累计匹配
        self.batch_match_count_label.setText(str(cumulative_stats.get("total_match_events", 0)))
        self.batch_unique_label.setText(str(cumulative_stats.get("unique_matched", 0)))

        # 4个评级分类计数
        self.batch_qualified_label.setText(str(cumulative_stats.get("qualified", 0)))
        self.batch_pending_label.setText(str(cumulative_stats.get("pending", 0)))
        self.batch_eliminated_label.setText(str(cumulative_stats.get("eliminated", 0)))

        # 本轮信息
        round_trades = cumulative_stats.get("round_trades", 0)
        round_sharpe = cumulative_stats.get("round_sharpe", 0.0)
        self.batch_round_trades_label.setText(str(round_trades))

        if round_sharpe >= 0:
            self.batch_round_sharpe_label.setText(f"{round_sharpe:.3f}")
            self.batch_round_sharpe_label.setStyleSheet(
                f"color: {UI_CONFIG['CHART_UP_COLOR']}; font-size: 12px;")
        else:
            self.batch_round_sharpe_label.setText(f"{round_sharpe:.3f}")
            self.batch_round_sharpe_label.setStyleSheet(
                f"color: {UI_CONFIG['CHART_DOWN_COLOR']}; font-size: 12px;")


    def on_batch_wf_started(self):
        """批量WF开始时调用"""
        self.batch_wf_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.batch_rounds_spin.setEnabled(False)
        self.batch_sample_spin.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self._batch_last_progress = 0
        self.batch_round_label.setText("启动中...")
        self.batch_eta_label.setText("")
        self.batch_match_count_label.setText("0")
        self.batch_unique_label.setText("0")
        self.batch_round_trades_label.setText("--")
        self.batch_round_sharpe_label.setText("--")

    def on_batch_wf_finished(self):
        """批量WF完成时调用"""
        self.batch_wf_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.batch_rounds_spin.setEnabled(True)
        self.batch_sample_spin.setEnabled(True)
        self._batch_last_progress = 0
        self.batch_progress_bar.setVisible(False)
        self.batch_round_label.setText("完成")

    def enable_batch_wf(self, enabled: bool):
        """启用/禁用批量验证按钮"""
        self.batch_wf_btn.setEnabled(enabled)


class AnalysisPanel(QtWidgets.QWidget):
    """
    分析面板 - 深色主题
    
    包含标签页：
    1. 交易明细
    2. 市场状态
    3. 指纹图（3D地形图显示轨迹指纹）
    4. 轨迹匹配
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        # 获取字体大小配置
        font_normal = UI_CONFIG.get('FONT_SIZE_NORMAL', 12)
        font_small = UI_CONFIG.get('FONT_SIZE_SMALL', 11)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
                font-size: {font_normal}px;
            }}
            QLabel {{
                font-size: {font_normal}px;
            }}
            QGroupBox {{
                font-size: {font_normal}px;
            }}
            QTabWidget::pane {{
                border: 1px solid #444;
                background-color: {UI_CONFIG['THEME_SURFACE']};
            }}
            QTabBar::tab {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                padding: 8px 16px;
                border: 1px solid #444;
                border-bottom: none;
                font-size: {font_normal}px;
            }}
            QTabBar::tab:selected {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                color: white;
            }}
            QTableWidget {{
                font-size: {font_small}px;
            }}
            QPushButton {{
                font-size: {font_normal}px;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标签页
        self.tabs = QtWidgets.QTabWidget()

        # 交易明细标签页
        self.trade_log_widget = TradeLogWidget()
        self.tabs.addTab(self.trade_log_widget, "交易明细")

        # 市场状态标签页
        self.market_regime_widget = MarketRegimeWidget()
        self.tabs.addTab(self.market_regime_widget, "市场状态")

        # 指纹地形图标签页
        from ui.vector_space_widget import FingerprintWidget
        self.fingerprint_widget = FingerprintWidget()
        self.tabs.addTab(self.fingerprint_widget, "指纹图")

        # 轨迹匹配标签页
        self.trajectory_widget = TrajectoryMatchWidget()
        self.tabs.addTab(self.trajectory_widget, "轨迹匹配")
        
        layout.addWidget(self.tabs)
        
        # 设置最小宽度（增大以适应高分辨率）
        self.setMinimumWidth(380)
    
    def update_feature_importance(self, feature_names: List[str], importances: List[float]):
        """更新特征重要性（已弃用，保持接口兼容）"""
        pass  # UI已移除，保留方法签名
    
    def update_pattern_logic(self, logic_data: Dict):
        """更新多空逻辑（已弃用，保持接口兼容）"""
        pass  # UI已移除，保留方法签名
    
    def update_survival_analysis(self, survival_data: Dict):
        """更新生存分析（已弃用，保持接口兼容）"""
        pass  # UI已移除，保留方法签名
    
    def update_all(self, analysis_results: Dict):
        """更新所有分析结果（已弃用，保持接口兼容）"""
        pass  # UI已移除，保留方法签名

    def update_trade_log(self, trades: List[Dict]):
        """更新交易明细"""
        self.trade_log_widget.update_trades(trades)

    def update_market_regime(self, current_regime: str, regime_stats: Dict):
        """更新市场状态统计"""
        self.market_regime_widget.update_current_regime(current_regime)
        self.market_regime_widget.update_stats(regime_stats)

    def update_fingerprint_templates(self, templates: list):
        """更新指纹图模板列表"""
        self.fingerprint_widget.set_templates(templates)

    def update_fingerprint_current(self, fingerprint, best_match_idx: int = -1,
                                    cosine_sim: float = 0.0, dtw_sim: float = 0.0):
        """更新当前K线的指纹及最佳匹配"""
        self.fingerprint_widget.set_current_fingerprint(
            fingerprint, best_match_idx, cosine_sim, dtw_sim
        )

    # ── 轨迹匹配相关方法 ──
    def update_trajectory_match_status(self, regime: str, best_match: str,
                                        cosine_sim: float, dtw_sim: float,
                                        direction: str):
        """更新轨迹匹配状态"""
        self.trajectory_widget.update_match_status(
            regime, best_match, cosine_sim, dtw_sim, direction
        )

    def update_trajectory_template_stats(self, total: int, long_count: int,
                                          short_count: int, avg_profit: float):
        """更新模板库统计"""
        self.trajectory_widget.update_template_stats(
            total, long_count, short_count, avg_profit
        )

    def update_trading_params(self, params):
        """更新 GA 优化后的交易参数"""
        self.trajectory_widget.update_trading_params(params)

    def update_memory_stats(self, template_count: int, file_count: int):
        """更新记忆统计"""
        self.trajectory_widget.update_memory_stats(template_count, file_count)

    def enable_save_memory(self, enabled: bool):
        """启用/禁用保存记忆按钮"""
        self.trajectory_widget.enable_save_memory(enabled)

    def update_template_evaluation(self, eval_result):
        """更新模板评估结果"""
        self.trajectory_widget.update_template_evaluation(eval_result)

    def reset_template_evaluation(self):
        """重置模板评估显示"""
        self.trajectory_widget.reset_template_evaluation()

    # ── 批量 Walk-Forward 方法转发 ──

    def update_batch_wf_progress(self, round_idx: int, n_rounds: int,
                                  cumulative_stats: dict):
        """更新批量WF进度"""
        self.trajectory_widget.update_batch_wf_progress(
            round_idx, n_rounds, cumulative_stats)

    def on_batch_wf_started(self):
        """批量WF开始"""
        self.trajectory_widget.on_batch_wf_started()

    def on_batch_wf_finished(self):
        """批量WF完成"""
        self.trajectory_widget.on_batch_wf_finished()

    def enable_batch_wf(self, enabled: bool):
        """启用/禁用批量验证按钮"""
        self.trajectory_widget.enable_batch_wf(enabled)


# 测试代码
if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    
    panel = AnalysisPanel()
    panel.setWindowTitle("分析面板测试")
    panel.resize(400, 600)
    
    # 测试交易数据
    test_trades = [
        {
            "side": "LONG", "entry_time": "2024-01-01 10:00",
            "entry_price": "42000.0", "exit_time": "2024-01-01 12:00",
            "exit_price": "42500.0", "profit": "50.0", "profit_pct": "1.19",
            "hold": "120", "regime": "TREND_UP", "fingerprint": "T#3 | Sim=0.85"
        },
        {
            "side": "SHORT", "entry_time": "2024-01-02 14:00",
            "entry_price": "43000.0", "exit_time": "2024-01-02 16:00",
            "exit_price": "42500.0", "profit": "50.0", "profit_pct": "1.16",
            "hold": "120", "regime": "TREND_DOWN", "fingerprint": "--"
        },
    ]
    panel.update_trade_log(test_trades)
    
    panel.show()
    sys.exit(app.exec())
