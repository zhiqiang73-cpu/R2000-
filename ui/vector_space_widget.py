"""
R3000 3D指纹地形图可视化
使用 matplotlib 的 surface_plot 显示轨迹指纹矩阵

显示内容：
  X轴 = 时间（0~59，入场前60根K线）
  Y轴 = 特征维度（0~31，32个特征）
  Z轴 = 特征值（高低起伏形成地形）

三种模式：
  1. 单模板浏览：从下拉列表选择一个模板，显示其3D地形
  2. 对比模式：蓝色=当前K线窗口指纹，金色=选中模板，叠加显示
  3. 最佳匹配：自动选取当前K线的最佳匹配模板进行对比
"""

from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UI_CONFIG, TRAJECTORY_CONFIG

# matplotlib 3D
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    HAS_MPL3D = True
except ImportError:
    HAS_MPL3D = False


class FingerprintWidget(QtWidgets.QWidget):
    """3D指纹地形图可视化组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._templates = []  # List[TrajectoryTemplate]
        self._current_fingerprint = None  # np.ndarray (60, 32) 当前K线指纹
        self._selected_template_idx = -1
        self._best_match_idx = -1
        self._best_cosine = 0.0
        self._best_dtw = 0.0
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        # ── 顶部控制栏 ──
        ctrl = QtWidgets.QHBoxLayout()

        # 模板选择下拉框
        ctrl.addWidget(QtWidgets.QLabel("模板:"))
        self.template_combo = QtWidgets.QComboBox()
        self.template_combo.setMinimumWidth(180)
        self.template_combo.addItem("-- 选择模板 --")
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        ctrl.addWidget(self.template_combo)

        ctrl.addSpacing(8)

        # 方向筛选
        ctrl.addWidget(QtWidgets.QLabel("方向:"))
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItems(["全部", "LONG", "SHORT"])
        self.direction_combo.currentTextChanged.connect(self._rebuild_template_list)
        ctrl.addWidget(self.direction_combo)

        ctrl.addSpacing(8)

        # 市场状态筛选
        ctrl.addWidget(QtWidgets.QLabel("状态:"))
        self.regime_combo = QtWidgets.QComboBox()
        self.regime_combo.setMinimumWidth(80)
        self.regime_combo.addItem("全部")
        self.regime_combo.currentTextChanged.connect(self._rebuild_template_list)
        ctrl.addWidget(self.regime_combo)

        ctrl.addStretch()

        # 统计信息
        self.stats_label = QtWidgets.QLabel("模板数: 0")
        self.stats_label.setStyleSheet(f"color: {UI_CONFIG['THEME_TEXT']}; font-size: 11px;")
        ctrl.addWidget(self.stats_label)

        layout.addLayout(ctrl)

        # ── 显示模式控制 ──
        mode_layout = QtWidgets.QHBoxLayout()

        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_single = QtWidgets.QRadioButton("单模板")
        self.mode_compare = QtWidgets.QRadioButton("对比模式")
        self.mode_best = QtWidgets.QRadioButton("最佳匹配")
        self.mode_single.setChecked(True)

        self.mode_group.addButton(self.mode_single, 0)
        self.mode_group.addButton(self.mode_compare, 1)
        self.mode_group.addButton(self.mode_best, 2)

        self.mode_single.toggled.connect(self._refresh_plot)
        self.mode_compare.toggled.connect(self._refresh_plot)
        self.mode_best.toggled.connect(self._refresh_plot)

        mode_layout.addWidget(self.mode_single)
        mode_layout.addWidget(self.mode_compare)
        mode_layout.addWidget(self.mode_best)
        mode_layout.addStretch()

        layout.addLayout(mode_layout)

        # ── 相似度显示栏 ──
        sim_layout = QtWidgets.QHBoxLayout()

        self.cosine_label = QtWidgets.QLabel("余弦: --")
        self.cosine_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        sim_layout.addWidget(self.cosine_label)

        sim_layout.addSpacing(15)

        self.dtw_label = QtWidgets.QLabel("DTW: --")
        self.dtw_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        sim_layout.addWidget(self.dtw_label)

        sim_layout.addSpacing(15)

        self.profit_label = QtWidgets.QLabel("收益: --")
        self.profit_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        sim_layout.addWidget(self.profit_label)

        sim_layout.addStretch()
        layout.addLayout(sim_layout)

        # ── 3D 图表 ──
        if HAS_MPL3D:
            self.figure = Figure(figsize=(6, 5), dpi=90,
                                 facecolor=UI_CONFIG['THEME_BACKGROUND'])
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setStyleSheet(f"background-color: {UI_CONFIG['THEME_BACKGROUND']};")
            layout.addWidget(self.canvas, stretch=1)
            self._init_3d_axes()
        else:
            label = QtWidgets.QLabel("需要 matplotlib 才能显示3D指纹地形图")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {UI_CONFIG['THEME_TEXT']}; font-size: 14px;")
            layout.addWidget(label, stretch=1)

    def _init_3d_axes(self):
        """初始化3D坐标轴"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection='3d',
                                           facecolor=UI_CONFIG['CHART_BACKGROUND'])
        self.ax.set_xlabel('时间 (K线)', color=UI_CONFIG['THEME_TEXT'],
                          fontsize=9, labelpad=8)
        self.ax.set_ylabel('特征维度', color=UI_CONFIG['THEME_TEXT'],
                          fontsize=9, labelpad=8)
        self.ax.set_zlabel('特征值', color=UI_CONFIG['THEME_TEXT'],
                          fontsize=9, labelpad=8)
        self.ax.tick_params(colors=UI_CONFIG['THEME_TEXT'], labelsize=7)

        # 设置网格
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#444')
        self.ax.yaxis.pane.set_edgecolor('#444')
        self.ax.zaxis.pane.set_edgecolor('#444')
        self.ax.grid(True, alpha=0.2, color='#555')

        self.figure.tight_layout(pad=1.5)

    # ══════════════════════════════════════════════════════════════════════════
    # 数据更新接口
    # ══════════════════════════════════════════════════════════════════════════

    def set_templates(self, templates: List):
        """
        设置模板列表

        Args:
            templates: List[TrajectoryTemplate]
        """
        self._templates = templates if templates else []

        # 更新市场状态下拉框
        regimes = set()
        for t in self._templates:
            regimes.add(t.regime)

        current_regime = self.regime_combo.currentText()
        self.regime_combo.blockSignals(True)
        self.regime_combo.clear()
        self.regime_combo.addItem("全部")
        for r in sorted(regimes):
            self.regime_combo.addItem(r)
        idx = self.regime_combo.findText(current_regime)
        if idx >= 0:
            self.regime_combo.setCurrentIndex(idx)
        self.regime_combo.blockSignals(False)

        self._rebuild_template_list()
        self.stats_label.setText(f"模板数: {len(self._templates)}")

    def set_current_fingerprint(self, fingerprint: np.ndarray,
                                 best_match_idx: int = -1,
                                 cosine_sim: float = 0.0,
                                 dtw_sim: float = 0.0):
        """
        设置当前K线的指纹矩阵

        Args:
            fingerprint: (window, 32) 当前K线窗口的原始特征矩阵
            best_match_idx: 最佳匹配的模板索引（在筛选后的列表中）
            cosine_sim: 余弦相似度
            dtw_sim: DTW相似度
        """
        self._current_fingerprint = fingerprint
        self._best_match_idx = best_match_idx
        self._best_cosine = cosine_sim
        self._best_dtw = dtw_sim

        # 更新相似度标签
        if cosine_sim > 0:
            color = UI_CONFIG['CHART_UP_COLOR'] if cosine_sim > 0.6 else '#FFA500'
            self.cosine_label.setText(f"余弦: {cosine_sim:.1%}")
            self.cosine_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {color};")
        else:
            self.cosine_label.setText("余弦: --")
            self.cosine_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {UI_CONFIG['THEME_TEXT']};")

        if dtw_sim > 0:
            color = UI_CONFIG['CHART_UP_COLOR'] if dtw_sim > 0.6 else '#FFA500'
            self.dtw_label.setText(f"DTW: {dtw_sim:.1%}")
            self.dtw_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {color};")
        else:
            self.dtw_label.setText("DTW: --")
            self.dtw_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {UI_CONFIG['THEME_TEXT']};")

        self._refresh_plot()

    def _rebuild_template_list(self, *args):
        """根据筛选条件重建模板下拉列表"""
        direction_filter = self.direction_combo.currentText()
        regime_filter = self.regime_combo.currentText()

        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem("-- 选择模板 --")

        self._filtered_templates = []
        for i, t in enumerate(self._templates):
            if direction_filter != "全部" and t.direction != direction_filter:
                continue
            if regime_filter != "全部" and t.regime != regime_filter:
                continue
            self._filtered_templates.append((i, t))
            label = f"#{i} {t.direction} {t.regime} +{t.profit_pct:.1f}%"
            self.template_combo.addItem(label)

        self.template_combo.blockSignals(False)
        self._refresh_plot()

    def _on_template_changed(self, index: int):
        """模板选择变化"""
        if index > 0 and index - 1 < len(self._filtered_templates):
            self._selected_template_idx = index - 1
            orig_idx, t = self._filtered_templates[self._selected_template_idx]
            # 更新收益标签
            color = UI_CONFIG['CHART_UP_COLOR'] if t.profit_pct > 0 else UI_CONFIG['CHART_DOWN_COLOR']
            self.profit_label.setText(f"收益: {t.profit_pct:.2f}%")
            self.profit_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {color};")
        else:
            self._selected_template_idx = -1
            self.profit_label.setText("收益: --")
            self.profit_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {UI_CONFIG['THEME_TEXT']};")

        self._refresh_plot()

    # ══════════════════════════════════════════════════════════════════════════
    # 3D绘图
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_plot(self, *args):
        """刷新3D地形图"""
        if not HAS_MPL3D:
            return

        self._init_3d_axes()

        mode = self.mode_group.checkedId()  # 0=单模板, 1=对比, 2=最佳匹配

        if mode == 0:
            # 单模板浏览
            self._plot_single_template()
        elif mode == 1:
            # 对比模式
            self._plot_compare()
        else:
            # 最佳匹配
            self._plot_best_match()

        self.canvas.draw_idle()

    def _plot_single_template(self):
        """绘制单个模板的3D地形"""
        if self._selected_template_idx < 0 or self._selected_template_idx >= len(self._filtered_templates):
            self.ax.set_title("请选择一个模板", color=UI_CONFIG['THEME_TEXT'], fontsize=10)
            return

        orig_idx, template = self._filtered_templates[self._selected_template_idx]
        matrix = template.pre_entry  # (60, 32)

        self._draw_surface(matrix, color='gold', alpha=0.8, label='模板指纹')
        self.ax.set_title(f"模板 #{orig_idx}: {template.direction} {template.regime}",
                         color=UI_CONFIG['THEME_TEXT'], fontsize=10)

    def _plot_compare(self):
        """对比当前指纹与选中模板"""
        has_current = self._current_fingerprint is not None and self._current_fingerprint.size > 0
        has_template = (self._selected_template_idx >= 0 and
                        self._selected_template_idx < len(self._filtered_templates))

        if has_template:
            orig_idx, template = self._filtered_templates[self._selected_template_idx]
            self._draw_surface(template.pre_entry, color='gold', alpha=0.6, label='模板')

        if has_current:
            self._draw_surface(self._current_fingerprint, color='deepskyblue', alpha=0.5, label='当前')

        if has_current or has_template:
            self.ax.legend(loc='upper left', fontsize=8, framealpha=0.6,
                          facecolor='#333', edgecolor='#555',
                          labelcolor=UI_CONFIG['THEME_TEXT'])

        title = "对比模式"
        if has_current and has_template:
            title += f" | 余弦: {self._best_cosine:.1%}"
        self.ax.set_title(title, color=UI_CONFIG['THEME_TEXT'], fontsize=10)

    def _plot_best_match(self):
        """绘制当前指纹与最佳匹配模板"""
        has_current = self._current_fingerprint is not None and self._current_fingerprint.size > 0

        if has_current:
            self._draw_surface(self._current_fingerprint, color='deepskyblue', alpha=0.5, label='当前')

        if self._best_match_idx >= 0 and self._best_match_idx < len(self._filtered_templates):
            orig_idx, template = self._filtered_templates[self._best_match_idx]
            self._draw_surface(template.pre_entry, color='gold', alpha=0.6, label=f'最佳匹配 #{orig_idx}')

            # 更新收益标签
            color = UI_CONFIG['CHART_UP_COLOR'] if template.profit_pct > 0 else UI_CONFIG['CHART_DOWN_COLOR']
            self.profit_label.setText(f"收益: {template.profit_pct:.2f}%")
            self.profit_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {color};")

        if has_current or self._best_match_idx >= 0:
            self.ax.legend(loc='upper left', fontsize=8, framealpha=0.6,
                          facecolor='#333', edgecolor='#555',
                          labelcolor=UI_CONFIG['THEME_TEXT'])

        title = "最佳匹配"
        if self._best_cosine > 0:
            title += f" | 余弦: {self._best_cosine:.1%} | DTW: {self._best_dtw:.1%}"
        self.ax.set_title(title, color=UI_CONFIG['THEME_TEXT'], fontsize=10)

    def _draw_surface(self, matrix: np.ndarray, color: str = 'viridis',
                      alpha: float = 0.8, label: str = None):
        """
        绘制3D曲面

        Args:
            matrix: (time, features) 指纹矩阵
            color: 颜色
            alpha: 透明度
            label: 图例标签
        """
        if matrix is None or matrix.size == 0:
            return

        # 确保矩阵是2D的
        if matrix.ndim != 2:
            return

        n_time, n_features = matrix.shape

        # 创建网格
        X = np.arange(n_time)
        Y = np.arange(n_features)
        X, Y = np.meshgrid(X, Y)

        # Z值（转置使特征在Y轴）
        Z = matrix.T  # (32, 60)

        # 绘制曲面
        if isinstance(color, str) and color in plt.colormaps():
            surf = self.ax.plot_surface(X, Y, Z, cmap=color, alpha=alpha,
                                        linewidth=0, antialiased=True)
        else:
            surf = self.ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                                        linewidth=0, antialiased=True, label=label)

        # 设置轴标签
        self.ax.set_xlim(0, n_time - 1)
        self.ax.set_ylim(0, n_features - 1)

        # 设置刻度
        self.ax.set_xticks([0, n_time // 2, n_time - 1])
        self.ax.set_yticks([0, n_features // 2, n_features - 1])

    def clear_plot(self):
        """清空图表"""
        self._templates = []
        self._current_fingerprint = None
        self._selected_template_idx = -1
        self._best_match_idx = -1
        self.template_combo.clear()
        self.template_combo.addItem("-- 选择模板 --")
        self.cosine_label.setText("余弦: --")
        self.dtw_label.setText("DTW: --")
        self.profit_label.setText("收益: --")
        self.stats_label.setText("模板数: 0")
        if HAS_MPL3D:
            self._init_3d_axes()
            self.canvas.draw_idle()


# 保持向后兼容的别名
VectorSpaceWidget = FingerprintWidget
