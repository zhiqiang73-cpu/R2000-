"""
R3000 分析面板
右侧分析面板：特征重要性、模式统计、生存分析
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
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "方向", "入场时间", "入场价", "出场时间", "出场价", "盈利(USDT)", "收益率%", "持仓", "市场状态"
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


class AnalysisPanel(QtWidgets.QWidget):
    """
    分析面板 - 深色主题
    
    包含三个标签页：
    1. 特征重要性
    2. 多空逻辑
    3. 生存分析
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
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
            }}
            QTabBar::tab:selected {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                color: white;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标签页
        self.tabs = QtWidgets.QTabWidget()
        
        # 特征重要性标签页
        self.feature_widget = FeatureImportanceWidget()
        self.tabs.addTab(self.feature_widget, "特征重要性")
        
        # 多空逻辑标签页
        self.pattern_widget = PatternLogicWidget()
        self.tabs.addTab(self.pattern_widget, "多空逻辑")
        
        # 生存分析标签页
        self.survival_widget = SurvivalAnalysisWidget()
        self.tabs.addTab(self.survival_widget, "生存分析")

        # 交易明细标签页
        self.trade_log_widget = TradeLogWidget()
        self.tabs.addTab(self.trade_log_widget, "交易明细")

        # 市场状态标签页
        self.market_regime_widget = MarketRegimeWidget()
        self.tabs.addTab(self.market_regime_widget, "市场状态")
        
        layout.addWidget(self.tabs)
        
        # 设置最小宽度
        self.setMinimumWidth(300)
    
    def update_feature_importance(self, feature_names: List[str], importances: List[float]):
        """更新特征重要性"""
        self.feature_widget.update_data(feature_names, importances)
    
    def update_pattern_logic(self, logic_data: Dict):
        """更新多空逻辑"""
        self.pattern_widget.update_data(logic_data)
    
    def update_survival_analysis(self, survival_data: Dict):
        """更新生存分析"""
        self.survival_widget.update_data(survival_data)
    
    def update_all(self, analysis_results: Dict):
        """更新所有分析结果"""
        if 'feature_importance' in analysis_results:
            fi = analysis_results['feature_importance']
            if 'top_features' in fi:
                names = [f['name'] for f in fi['top_features']]
                imps = [f['importance'] for f in fi['top_features']]
                self.update_feature_importance(names, imps)
        
        if 'long_short_logic' in analysis_results:
            self.update_pattern_logic(analysis_results['long_short_logic'])
        
        if 'survival' in analysis_results:
            self.update_survival_analysis(analysis_results['survival'])

    def update_trade_log(self, trades: List[Dict]):
        """更新交易明细"""
        self.trade_log_widget.update_trades(trades)

    def update_market_regime(self, current_regime: str, regime_stats: Dict):
        """更新市场状态统计"""
        self.market_regime_widget.update_current_regime(current_regime)
        self.market_regime_widget.update_stats(regime_stats)


# 测试代码
if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    
    panel = AnalysisPanel()
    panel.setWindowTitle("分析面板测试")
    panel.resize(400, 600)
    
    # 测试数据
    feature_names = [f"feature_{i}" for i in range(20)]
    importances = np.random.random(20).tolist()
    panel.update_feature_importance(feature_names, importances)
    
    logic_data = {
        'long_conditions': [
            {'feature': 'rsi', 'direction': 'below', 'threshold': 0.3, 'probability': 0.65},
            {'feature': 'macd', 'direction': 'above', 'threshold': 0.1, 'probability': 0.58},
        ],
        'short_conditions': [
            {'feature': 'rsi', 'direction': 'above', 'threshold': 0.7, 'probability': 0.62},
        ]
    }
    panel.update_pattern_logic(logic_data)
    
    survival_data = {
        'hold_periods': {
            'mean': 45.5,
            'median': 38.0,
            'histogram': {
                'bins': [0, 10, 20, 30, 40, 50, 60],
                'counts': [5, 12, 18, 15, 8, 4]
            }
        },
        'profit': {
            'mean': 0.85,
            'median': 0.62,
            'histogram': {
                'bins': [-2, -1, 0, 1, 2, 3],
                'counts': [3, 8, 12, 20, 10]
            }
        },
        'win_rate': 0.58,
        'risk_reward_mean': 1.8
    }
    panel.update_survival_analysis(survival_data)
    
    panel.show()
    sys.exit(app.exec())
