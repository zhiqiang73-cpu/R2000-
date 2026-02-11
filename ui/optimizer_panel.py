"""
R3000 优化器面板
遗传算法优化结果显示
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
    plt.style.use('dark_background')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class OptimizerPanel(QtWidgets.QWidget):
    """
    优化器面板 - 深色主题
    
    显示 GA 优化过程和结果：
    - 进度显示
    - 适应度曲线
    - 最优策略参数
    - 回测指标
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fitness_history = []
        self._init_ui()
    
    def _init_ui(self):
        """初始化 UI"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: {UI_CONFIG['THEME_SURFACE']};
            }}
            QProgressBar::chunk {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
            }}
        """)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # === 左侧：进度和图表 ===
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 进度组
        progress_group = QtWidgets.QGroupBox("优化进度")
        progress_layout = QtWidgets.QVBoxLayout(progress_group)
        
        self.progress_label = QtWidgets.QLabel("代数: 0 / 0")
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(progress_group)
        
        # 适应度曲线
        curve_group = QtWidgets.QGroupBox("适应度曲线")
        curve_layout = QtWidgets.QVBoxLayout(curve_group)
        
        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(4, 2), dpi=100, facecolor=UI_CONFIG['THEME_BACKGROUND'])
            self.canvas = FigureCanvas(self.figure)
            curve_layout.addWidget(self.canvas)
        else:
            self.curve_label = QtWidgets.QLabel("需要 matplotlib 显示曲线")
            self.curve_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            curve_layout.addWidget(self.curve_label)
        
        left_layout.addWidget(curve_group)
        # 用户要求精简底部面板：隐藏“优化进度/适应度曲线”区域
        left_widget.setVisible(False)
        layout.addWidget(left_widget, stretch=0)
        
        # === 中间：最优策略参数 ===
        params_group = QtWidgets.QGroupBox("最优策略参数")
        params_layout = QtWidgets.QFormLayout(params_group)
        
        self.long_threshold_label = QtWidgets.QLabel("--")
        params_layout.addRow("做多阈值:", self.long_threshold_label)
        
        self.short_threshold_label = QtWidgets.QLabel("--")
        params_layout.addRow("做空阈值:", self.short_threshold_label)
        
        self.tp_label = QtWidgets.QLabel("--")
        params_layout.addRow("止盈 (%):", self.tp_label)
        
        self.sl_label = QtWidgets.QLabel("--")
        params_layout.addRow("止损 (%):", self.sl_label)
        
        self.max_hold_label = QtWidgets.QLabel("--")
        params_layout.addRow("最大持仓:", self.max_hold_label)
        
        self.fitness_label = QtWidgets.QLabel("--")
        self.fitness_label.setStyleSheet(f"color: {UI_CONFIG['THEME_ACCENT']}; font-weight: bold; font-size: 14px;")
        params_layout.addRow("适应度:", self.fitness_label)
        
        # 用户要求精简底部面板：隐藏“最优策略参数”区域
        params_group.setVisible(False)
        layout.addWidget(params_group, stretch=0)
        
        # === 右侧：回测指标 ===
        metrics_group = QtWidgets.QGroupBox("回测指标")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)
        
        # 第一列：基础指标
        self.initial_capital_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("初始本金:"), 0, 0)
        metrics_layout.addWidget(self.initial_capital_label, 0, 1)

        self.total_profit_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("累计盈亏:"), 1, 0)
        metrics_layout.addWidget(self.total_profit_label, 1, 1)

        self.total_return_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("总收益率:"), 2, 0)
        metrics_layout.addWidget(self.total_return_label, 2, 1)

        self.win_rate_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("总胜率:"), 3, 0)
        metrics_layout.addWidget(self.win_rate_label, 3, 1)

        self.max_dd_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("最大回撤:"), 4, 0)
        metrics_layout.addWidget(self.max_dd_label, 4, 1)

        # 第二列：当前/最近开仓信息
        metrics_layout.addWidget(QtWidgets.QLabel("<b>当前/最近开仓:</b>"), 0, 2)
        
        self.cur_margin_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("保证金:"), 1, 2)
        metrics_layout.addWidget(self.cur_margin_label, 1, 3)

        self.cur_tp_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("止盈价:"), 2, 2)
        metrics_layout.addWidget(self.cur_tp_label, 2, 3)

        self.cur_sl_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("止损价:"), 3, 2)
        metrics_layout.addWidget(self.cur_sl_label, 3, 3)

        self.cur_liq_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("强平价:"), 4, 2)
        metrics_layout.addWidget(self.cur_liq_label, 4, 3)

        # 第三列：多空细分
        self.long_win_rate_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("做多胜率:"), 1, 4)
        metrics_layout.addWidget(self.long_win_rate_label, 1, 5)

        self.long_profit_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("做多盈亏:"), 2, 4)
        metrics_layout.addWidget(self.long_profit_label, 2, 5)

        self.short_win_rate_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("做空胜率:"), 3, 4)
        metrics_layout.addWidget(self.short_win_rate_label, 3, 5)

        self.short_profit_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(QtWidgets.QLabel("做空盈亏:"), 4, 4)
        metrics_layout.addWidget(self.short_profit_label, 4, 5)
        
        layout.addWidget(metrics_group, stretch=2)
    
    @staticmethod
    def _fmt_amount(v: float, signed: bool = False) -> str:
        """
        统一金额显示：
        - 大数用科学计数法 a × 10^b
        - 普通数保持千分位
        """
        try:
            x = float(v)
        except Exception:
            return "--"
        if not np.isfinite(x):
            return "--"
        if x == 0:
            return "0"
        ax = abs(x)
        sign = "+" if (signed and x > 0) else ("-" if x < 0 else "")
        if ax >= 1e6:
            exp = int(np.floor(np.log10(ax)))
            coeff = ax / (10 ** exp)
            return f"{sign}{coeff:.3f} × 10^{exp}"
        if signed:
            return f"{x:+,.2f}"
        return f"{x:,.2f}"
    
    def reset(self):
        """重置面板"""
        self.fitness_history = []
        self.progress_bar.setValue(0)
        self.progress_label.setText("代数: 0 / 0")
        
        # 清空曲线
        if HAS_MATPLOTLIB:
            self.figure.clear()
            self.canvas.draw()
        
        # 重置标签
        labels_to_reset = [
            self.long_threshold_label, self.short_threshold_label,
            self.tp_label, self.sl_label, self.max_hold_label,
            self.fitness_label, self.win_rate_label,
            self.total_return_label, self.total_profit_label, self.max_dd_label,
            self.long_win_rate_label, self.long_profit_label, 
            self.short_win_rate_label, self.short_profit_label,
            self.initial_capital_label, self.cur_margin_label,
            self.cur_tp_label, self.cur_sl_label, self.cur_liq_label
        ]
        for label in labels_to_reset:
            label.setText("--")
            label.setStyleSheet("")

    
    def update_progress(self, current_gen: int, max_gen: int):
        """更新进度"""
        self.progress_label.setText(f"代数: {current_gen} / {max_gen}")
        self.progress_bar.setMaximum(max_gen)
        self.progress_bar.setValue(current_gen)
    
    def add_fitness_point(self, fitness: float):
        """添加适应度点"""
        self.fitness_history.append(fitness)
        self._update_fitness_curve()
    
    def update_fitness_curve(self, fitness_history: List[float]):
        """更新适应度曲线"""
        self.fitness_history = list(fitness_history)
        self._update_fitness_curve()
    
    def _update_fitness_curve(self):
        """内部更新曲线"""
        if not HAS_MATPLOTLIB:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111, facecolor=UI_CONFIG['THEME_BACKGROUND'])
        
        if self.fitness_history:
            generations = list(range(1, len(self.fitness_history) + 1))
            ax.plot(generations, self.fitness_history, color=UI_CONFIG['THEME_ACCENT'], 
                    linewidth=2, marker='o', markersize=3)
            ax.fill_between(generations, self.fitness_history, alpha=0.3, color=UI_CONFIG['THEME_ACCENT'])
        
        ax.set_xlabel('代数', color=UI_CONFIG['THEME_TEXT'], fontsize=9)
        ax.set_ylabel('适应度', color=UI_CONFIG['THEME_TEXT'], fontsize=9)
        ax.tick_params(colors=UI_CONFIG['THEME_TEXT'], labelsize=8)
        ax.grid(True, alpha=0.3, color='#444')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_strategy_params(self, strategy_data: Dict):
        """更新策略参数"""
        if 'long_threshold' in strategy_data:
            self.long_threshold_label.setText(f"{strategy_data['long_threshold']:.4f}")
        if 'short_threshold' in strategy_data:
            self.short_threshold_label.setText(f"{strategy_data['short_threshold']:.4f}")
        if 'take_profit_pct' in strategy_data:
            self.tp_label.setText(f"{strategy_data['take_profit_pct']:.2f}%")
        if 'stop_loss_pct' in strategy_data:
            self.sl_label.setText(f"{strategy_data['stop_loss_pct']:.2f}%")
        if 'max_hold_periods' in strategy_data:
            self.max_hold_label.setText(f"{strategy_data['max_hold_periods']}")
        if 'fitness' in strategy_data:
            self.fitness_label.setText(f"{strategy_data['fitness']:.4f}")
    
    def update_backtest_metrics(self, metrics: Dict):
        """更新回测指标"""
        if 'initial_capital' in metrics:
            self.initial_capital_label.setText(self._fmt_amount(metrics['initial_capital']))
            
        if 'win_rate' in metrics:
            rate = metrics['win_rate']
            color = UI_CONFIG['CHART_UP_COLOR'] if rate >= 0.5 else UI_CONFIG['CHART_DOWN_COLOR']
            self.win_rate_label.setText(f"{rate*100:.1f}%")
            self.win_rate_label.setStyleSheet(f"color: {color};")
            
        if 'total_return' in metrics:
            ret = metrics['total_return']
            color = UI_CONFIG['CHART_UP_COLOR'] if ret >= 0 else UI_CONFIG['CHART_DOWN_COLOR']
            self.total_return_label.setText(f"{ret*100:.2f}%")
            self.total_return_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
        if 'total_profit' in metrics:
            profit = metrics['total_profit']
            color = UI_CONFIG['CHART_UP_COLOR'] if profit >= 0 else UI_CONFIG['CHART_DOWN_COLOR']
            self.total_profit_label.setText(self._fmt_amount(profit, signed=True))
            self.total_profit_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
        if 'max_drawdown' in metrics:
            self.max_dd_label.setText(f"{metrics['max_drawdown']*100:.2f}%")
            self.max_dd_label.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']};")

        # 当前/最近开仓信息
        if 'current_pos' in metrics and metrics['current_pos']:
            pos = metrics['current_pos']
            self.cur_margin_label.setText(self._fmt_amount(pos.margin))
            self.cur_tp_label.setText(self._fmt_amount(pos.take_profit))
            self.cur_sl_label.setText(self._fmt_amount(pos.stop_loss))
            self.cur_liq_label.setText(self._fmt_amount(pos.liquidation_price))
            self.cur_liq_label.setStyleSheet(f"color: {UI_CONFIG['CHART_DOWN_COLOR']};")
        elif 'last_trade' in metrics and metrics['last_trade']:
            t = metrics['last_trade']
            self.cur_margin_label.setText(self._fmt_amount(t.margin))
            self.cur_tp_label.setText(self._fmt_amount(t.take_profit))
            self.cur_sl_label.setText(self._fmt_amount(t.stop_loss))
            self.cur_liq_label.setText(self._fmt_amount(t.liquidation_price))
            self.cur_liq_label.setStyleSheet("")

        if 'long_win_rate' in metrics:
            rate = metrics['long_win_rate']
            color = UI_CONFIG['CHART_UP_COLOR'] if rate >= 0.5 else UI_CONFIG['CHART_DOWN_COLOR']
            self.long_win_rate_label.setText(f"{rate*100:.1f}%")
            self.long_win_rate_label.setStyleSheet(f"color: {color};")
            
        if 'long_profit' in metrics:
            profit = metrics['long_profit']
            color = UI_CONFIG['CHART_UP_COLOR'] if profit >= 0 else UI_CONFIG['CHART_DOWN_COLOR']
            self.long_profit_label.setText(self._fmt_amount(profit, signed=True))
            self.long_profit_label.setStyleSheet(f"color: {color};")
            
        if 'short_win_rate' in metrics:
            rate = metrics['short_win_rate']
            color = UI_CONFIG['CHART_UP_COLOR'] if rate >= 0.5 else UI_CONFIG['CHART_DOWN_COLOR']
            self.short_win_rate_label.setText(f"{rate*100:.1f}%")
            self.short_win_rate_label.setStyleSheet(f"color: {color};")
            
        if 'short_profit' in metrics:
            profit = metrics['short_profit']
            color = UI_CONFIG['CHART_UP_COLOR'] if profit >= 0 else UI_CONFIG['CHART_DOWN_COLOR']
            self.short_profit_label.setText(self._fmt_amount(profit, signed=True))
            self.short_profit_label.setStyleSheet(f"color: {color};")

    
    def update_all(self, result):
        """更新所有显示"""
        # 更新适应度曲线
        if hasattr(result, 'fitness_history'):
            self.update_fitness_curve(result.fitness_history)
        
        # 更新最优策略
        if hasattr(result, 'best_individual') and result.best_individual:
            best = result.best_individual
            self.update_strategy_params({
                'long_threshold': best.long_threshold,
                'short_threshold': best.short_threshold,
                'take_profit_pct': best.take_profit_pct,
                'stop_loss_pct': best.stop_loss_pct,
                'max_hold_periods': best.max_hold_periods,
                'fitness': best.fitness,
            })
        
        # 更新回测指标
        if hasattr(result, 'best_metrics') and result.best_metrics:
            self.update_backtest_metrics(result.best_metrics)
