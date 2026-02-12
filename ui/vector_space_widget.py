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
        self.mode_overlay = QtWidgets.QRadioButton("Overlay")
        self.mode_single = QtWidgets.QRadioButton("Single")
        self.mode_compare = QtWidgets.QRadioButton("Compare")
        self.mode_overlay.setChecked(True)  # 默认叠加模式

        self.mode_group.addButton(self.mode_overlay, 0)
        self.mode_group.addButton(self.mode_single, 1)
        self.mode_group.addButton(self.mode_compare, 2)

        self.mode_overlay.toggled.connect(self._refresh_plot)
        self.mode_single.toggled.connect(self._refresh_plot)
        self.mode_compare.toggled.connect(self._refresh_plot)

        mode_layout.addWidget(self.mode_overlay)
        mode_layout.addWidget(self.mode_single)
        mode_layout.addWidget(self.mode_compare)

        # 显示数量控制
        mode_layout.addSpacing(15)
        mode_layout.addWidget(QtWidgets.QLabel("N:"))
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 20)
        self.count_spin.setValue(5)
        self.count_spin.setMaximumWidth(50)
        self.count_spin.valueChanged.connect(self._refresh_plot)
        mode_layout.addWidget(self.count_spin)
        
        # 标准化显示选项（让变化更明显）
        mode_layout.addSpacing(10)
        self.normalize_checkbox = QtWidgets.QCheckBox("归一化")
        self.normalize_checkbox.setToolTip("对每个模板单独做 z-score 标准化，使特征变化更明显")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.stateChanged.connect(self._refresh_plot)
        mode_layout.addWidget(self.normalize_checkbox)

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
        self.ax.set_xlabel('Time (Bars)', color=UI_CONFIG['THEME_TEXT'],
                          fontsize=9, labelpad=8)
        self.ax.set_ylabel('Features', color=UI_CONFIG['THEME_TEXT'],
                          fontsize=9, labelpad=8)
        self.ax.set_zlabel('Value', color=UI_CONFIG['THEME_TEXT'],
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

        # 数据质量检查
        if self._templates:
            valid_count = 0
            empty_count = 0
            for t in self._templates[:10]:  # 只检查前10个
                if hasattr(t, 'pre_entry') and t.pre_entry is not None:
                    if t.pre_entry.size > 0 and t.pre_entry.ndim == 2:
                        valid_count += 1
                        print(f"[FingerprintWidget] 模板检查: pre_entry.shape={t.pre_entry.shape}, "
                              f"range=[{t.pre_entry.min():.2f}, {t.pre_entry.max():.2f}]")
                    else:
                        empty_count += 1
                        print(f"[FingerprintWidget] ⚠ 模板异常: pre_entry.size={t.pre_entry.size}, ndim={t.pre_entry.ndim}")
                else:
                    empty_count += 1
                    print(f"[FingerprintWidget] ⚠ 模板缺少 pre_entry 属性")
            if empty_count > 0:
                print(f"[FingerprintWidget] ⚠ 警告: {empty_count} 个模板数据异常!")

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

        mode = self.mode_group.checkedId()  # 0=叠加, 1=单模板, 2=对比

        if mode == 0:
            # 叠加浏览（多个指纹曲面叠加）
            self._plot_overlay()
        elif mode == 1:
            # 单模板浏览
            self._plot_single_template()
        else:
            # 对比模式
            self._plot_compare()

        self.canvas.draw_idle()

    def _plot_overlay(self):
        """叠加绘制多个指纹曲面"""
        max_count = self.count_spin.value()
        templates_to_show = self._filtered_templates[:max_count]

        if not templates_to_show:
            self.ax.set_title("No templates to display", color=UI_CONFIG['THEME_TEXT'], fontsize=10)
            return

        # 颜色调色板：LONG用蓝色系，SHORT用红色系
        colors_long = ['#00BFFF', '#1E90FF', '#4169E1', '#0000CD', '#00008B']
        colors_short = ['#FF6347', '#FF4500', '#DC143C', '#B22222', '#8B0000']

        long_idx = 0
        short_idx = 0
        
        # 调试：收集所有模板的数据统计
        all_z_values = []

        for orig_idx, t in templates_to_show:
            if t.direction == "LONG":
                color = colors_long[long_idx % len(colors_long)]
                long_idx += 1
            else:
                color = colors_short[short_idx % len(colors_short)]
                short_idx += 1

            # 使用pre_entry作为指纹（半透明+网格线）
            matrix = t.pre_entry
            if matrix is not None and matrix.size > 0:
                # 调试输出
                print(f"[Fingerprint] Template #{orig_idx}: shape={matrix.shape}, "
                      f"min={matrix.min():.4f}, max={matrix.max():.4f}, "
                      f"mean={matrix.mean():.4f}, std={matrix.std():.4f}")
                
                # 如果启用归一化，对每个模板单独做 z-score
                if self.normalize_checkbox.isChecked():
                    matrix = matrix.copy()
                    std = matrix.std()
                    if std > 1e-9:
                        matrix = (matrix - matrix.mean()) / std
                    else:
                        matrix = matrix - matrix.mean()
                
                all_z_values.append(matrix)
            self._draw_wireframe_surface(matrix, color=color, alpha=0.35)

        # 如果有当前K线指纹，用金色高亮显示
        if self._current_fingerprint is not None and self._current_fingerprint.size > 0:
            self._draw_wireframe_surface(self._current_fingerprint, color='#FFD700', alpha=0.7)
        
        # 调试：打印整体数据范围并设置合理的 z 轴范围
        if all_z_values:
            combined = np.concatenate([m.flatten() for m in all_z_values])
            z_min, z_max = combined.min(), combined.max()
            z_std = combined.std()
            print(f"[Fingerprint] Combined stats: min={z_min:.4f}, "
                  f"max={z_max:.4f}, std={z_std:.4f}")
            
            # 如果数据范围太小，尝试使用 percentile 排除异常值
            if z_std > 0:
                p5, p95 = np.percentile(combined, [5, 95])
                # 仅当范围合理时设置 z 轴限制
                z_range = p95 - p5
                if z_range > 0.01:  # 避免除零
                    margin = z_range * 0.1
                    self.ax.set_zlim(p5 - margin, p95 + margin)
                    print(f"[Fingerprint] Z轴范围: [{p5 - margin:.2f}, {p95 + margin:.2f}]")

        self.ax.set_title(f"Overlay: {len(templates_to_show)} templates",
                         color=UI_CONFIG['THEME_TEXT'], fontsize=10)

    def _plot_single_template(self):
        """绘制单个模板的3D地形（线框风格）"""
        if self._selected_template_idx < 0 or self._selected_template_idx >= len(self._filtered_templates):
            self.ax.set_title("Select a template", color=UI_CONFIG['THEME_TEXT'], fontsize=10)
            return

        orig_idx, template = self._filtered_templates[self._selected_template_idx]
        matrix = template.pre_entry  # (60, 32)
        
        # 归一化选项
        if matrix is not None and matrix.size > 0 and self.normalize_checkbox.isChecked():
            matrix = matrix.copy()
            std = matrix.std()
            if std > 1e-9:
                matrix = (matrix - matrix.mean()) / std
            else:
                matrix = matrix - matrix.mean()

        self._draw_wireframe_surface(matrix, color='#FFD700', alpha=0.6)
        self.ax.set_title(f"Template #{orig_idx}: {template.direction} {template.regime}",
                         color=UI_CONFIG['THEME_TEXT'], fontsize=10)

    def _plot_compare(self):
        """对比当前指纹与选中模板（线框风格）"""
        has_current = self._current_fingerprint is not None and self._current_fingerprint.size > 0
        has_template = (self._selected_template_idx >= 0 and
                        self._selected_template_idx < len(self._filtered_templates))

        if has_template:
            orig_idx, template = self._filtered_templates[self._selected_template_idx]
            self._draw_wireframe_surface(template.pre_entry, color='#FFD700', alpha=0.5)

        if has_current:
            self._draw_wireframe_surface(self._current_fingerprint, color='#00BFFF', alpha=0.5)

        title = "Compare"
        if has_current and has_template:
            title += f" | Cosine: {self._best_cosine:.1%}"
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

        # 绘制曲面（带网格线）
        if isinstance(color, str) and color in plt.colormaps():
            surf = self.ax.plot_surface(X, Y, Z, cmap=color, alpha=alpha,
                                        linewidth=0.3, edgecolor='#333',
                                        antialiased=True)
        else:
            surf = self.ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                                        linewidth=0.3, edgecolor='#333',
                                        antialiased=True, label=label)

        # 设置轴标签
        self.ax.set_xlim(0, n_time - 1)
        self.ax.set_ylim(0, n_features - 1)

        # 设置刻度
        self.ax.set_xticks([0, n_time // 2, n_time - 1])
        self.ax.set_yticks([0, n_features // 2, n_features - 1])

    def _draw_wireframe_surface(self, matrix: np.ndarray, color: str = '#00BFFF',
                                 alpha: float = 0.4):
        """
        绘制半透明线框曲面（如用户参考图片的风格）

        Args:
            matrix: (time, features) 指纹矩阵
            color: 曲面和网格线颜色
            alpha: 透明度 (0.3~0.5 效果最好)
        """
        if matrix is None or matrix.size == 0:
            return

        if matrix.ndim != 2:
            return

        n_time, n_features = matrix.shape

        # 创建网格
        X = np.arange(n_time)
        Y = np.arange(n_features)
        X, Y = np.meshgrid(X, Y)

        # Z值（转置使特征在Y轴）
        Z = matrix.T  # (features, time)

        # 半透明填充 + 细网格线
        self.ax.plot_surface(
            X, Y, Z,
            color=color,
            alpha=alpha,
            rstride=2,            # 行步长（减少密度提升性能）
            cstride=2,            # 列步长
            linewidth=0.4,        # 细网格线
            edgecolor=color,      # 网格线同色
            antialiased=True,
            shade=False           # 禁用光照，保持颜色一致
        )

        # 设置轴范围
        self.ax.set_xlim(0, n_time - 1)
        self.ax.set_ylim(0, n_features - 1)

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
