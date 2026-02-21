"""
R3000 控制面板
左侧控制面板：参数配置、数据采样设置、标注参数、速度控制
深色主题
"""
from PyQt6 import QtWidgets, QtCore, QtGui
from typing import Dict, Optional, Callable
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, LABELING_CONFIG, GA_CONFIG, UI_CONFIG


class ControlPanel(QtWidgets.QWidget):
    """
    控制面板
    
    功能：
    - 数据采样配置
    - 标注参数配置
    - 播放速度控制
    - 状态显示
    """
    
    # 信号
    sample_requested = QtCore.pyqtSignal(int, object)  # (sample_size, seed)
    label_requested = QtCore.pyqtSignal(dict)           # labeling_params
    quick_label_requested = QtCore.pyqtSignal(dict)     # 仅标注（不播放动画）
    analyze_requested = QtCore.pyqtSignal()
    optimize_requested = QtCore.pyqtSignal(dict)       # ga_params
    play_requested = QtCore.pyqtSignal()               # 开始播放
    pause_requested = QtCore.pyqtSignal()              # 暂停播放
    stop_requested = QtCore.pyqtSignal()               # 停止播放
    speed_changed = QtCore.pyqtSignal(int)             # 速度变化
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化 UI"""
        # 深色主题样式
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
            QPushButton {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #0098ff;
            }}
            QPushButton:disabled {{
                background-color: #444;
                color: #888;
            }}
            QPushButton#playBtn {{
                background-color: #089981;
            }}
            QPushButton#playBtn:hover {{
                background-color: #0ab090;
            }}
            QPushButton#stopBtn {{
                background-color: #f23645;
            }}
            QPushButton#stopBtn:hover {{
                background-color: #ff4555;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                border: 1px solid #444;
                border-radius: 3px;
                padding: 3px;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #444;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {UI_CONFIG['THEME_ACCENT']};
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {UI_CONFIG['THEME_ACCENT']};
                border-radius: 3px;
            }}
            QLabel {{
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        self.main_layout = layout
        
        # === 数据采样组 ===
        sample_group = QtWidgets.QGroupBox("数据采样")
        sample_layout = QtWidgets.QFormLayout(sample_group)
        
        self.sample_size_spin = QtWidgets.QSpinBox()
        self.sample_size_spin.setRange(10000, 200000)
        self.sample_size_spin.setSingleStep(10000)
        self.sample_size_spin.setValue(DATA_CONFIG["SAMPLE_SIZE"])
        self.sample_size_spin.setToolTip("从历史数据中采样的连续K线数量\n(50000根 ≈ 35天的1分钟数据)")
        sample_layout.addRow("采样数量:", self.sample_size_spin)
        
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(-1, 999999)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("随机")
        self.seed_spin.setToolTip(
            "随机种子用于控制采样的随机性：\n"
            "• 设为「随机」(-1)：每次采样不同的数据段\n"
            "• 设为具体数字(如42)：每次采样相同的数据段\n"
            "  （用于复现结果或对比测试）"
        )
        
        # 添加说明标签
        seed_label = QtWidgets.QLabel("随机种子:")
        seed_label.setToolTip(self.seed_spin.toolTip())
        sample_layout.addRow(seed_label, self.seed_spin)
        
        self.sample_btn = QtWidgets.QPushButton("采样数据")
        self.sample_btn.clicked.connect(self._on_sample_clicked)
        sample_layout.addRow(self.sample_btn)
        
        # 数据时间范围显示
        self.time_range_label = QtWidgets.QLabel("时间范围: --")
        self.time_range_label.setWordWrap(True)
        self.time_range_label.setStyleSheet("color: #888; font-size: 11px;")
        sample_layout.addRow(self.time_range_label)
        
        layout.addWidget(sample_group)
        
        # === 高低点标注组 ===
        label_group = QtWidgets.QGroupBox("高低点标注")
        label_layout = QtWidgets.QFormLayout(label_group)

        self.swing_window_spin = QtWidgets.QSpinBox()
        self.swing_window_spin.setRange(2, 20)
        self.swing_window_spin.setValue(LABELING_CONFIG["SWING_WINDOW"])
        self.swing_window_spin.setToolTip("高低点窗口（越小越敏感，信号越多）")
        label_layout.addRow("高低点窗口:", self.swing_window_spin)
        
        layout.addWidget(label_group)
        
        # === 播放控制组 ===
        play_group = QtWidgets.QGroupBox("标注播放控制")
        play_layout = QtWidgets.QVBoxLayout(play_group)
        
        # 速度滑块
        speed_layout = QtWidgets.QHBoxLayout()
        speed_layout.addWidget(QtWidgets.QLabel("速度:"))
        
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(UI_CONFIG["MIN_SPEED"], UI_CONFIG["MAX_SPEED"])
        self.speed_slider.setValue(UI_CONFIG["DEFAULT_SPEED"])
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QtWidgets.QLabel(f"{UI_CONFIG['DEFAULT_SPEED']}x")
        self.speed_label.setMinimumWidth(40)
        speed_layout.addWidget(self.speed_label)
        
        play_layout.addLayout(speed_layout)
        
        # 播放按钮
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.play_btn = QtWidgets.QPushButton("▶ 开始回测")
        self.play_btn.setObjectName("playBtn")
        self.play_btn.clicked.connect(self._on_play_clicked)
        btn_layout.addWidget(self.play_btn)
        
        self.stop_btn = QtWidgets.QPushButton("■ 停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        play_layout.addLayout(btn_layout)
        
        # 仅标注按钮（快速模式，不播放动画）
        self.quick_label_btn = QtWidgets.QPushButton("⚡ 仅标注")
        self.quick_label_btn.setToolTip("只计算标注，不播放动画，完成后可直接运行Walk-Forward")
        self.quick_label_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2d5a2d;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #3d7a3d;
            }}
            QPushButton:disabled {{
                background-color: #444;
                color: #888;
            }}
        """)
        self.quick_label_btn.clicked.connect(self._on_quick_label_clicked)
        play_layout.addWidget(self.quick_label_btn)
        
        # 严格市场状态过滤开关
        self.strict_state_filter_chk = QtWidgets.QCheckBox("⚠ 严格状态过滤")
        self.strict_state_filter_chk.setToolTip(
            "开启后：回测时若某策略在当前市场状态下命中率不达标（被标 ⚠），\n"
            "则该市场状态下禁止开仓（多空均不允许）。\n"
            "关闭：与原来一致，不限制。"
        )
        self.strict_state_filter_chk.setStyleSheet("color: #e0a020; font-size: 12px;")
        play_layout.addWidget(self.strict_state_filter_chk)

        # 回测试验 TP/SL 开关
        self.alt_tpsl_chk = QtWidgets.QCheckBox("试验新TP/SL：+0.4% / -0.5%（仅回测）")
        self.alt_tpsl_chk.setToolTip(
            "开启后：回测使用 TP=+0.4% / SL=-0.5%（价格）\n"
            "仅用于回测对比，不影响策略池或实盘参数。"
        )
        self.alt_tpsl_chk.setStyleSheet("color: #7fb0ff; font-size: 12px;")
        play_layout.addWidget(self.alt_tpsl_chk)

        # 进度显示
        self.play_progress_label = QtWidgets.QLabel("进度: 0 / 0")
        self.play_progress_label.setStyleSheet("color: #888;")
        play_layout.addWidget(self.play_progress_label)
        
        layout.addWidget(play_group)
        
        # === 模式挖掘组（已隐藏，保留后端）===
        # analyze_group 和 ga_group 已移除UI入口，相关方法保留以便未来使用
        
        # === 状态显示 ===
        status_group = QtWidgets.QGroupBox("状态")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
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
        status_layout.addWidget(self.progress_bar)
        
        layout.addWidget(status_group)
        
        # 弹性空间
        self._bottom_spacer = QtWidgets.QSpacerItem(
            20, 40,
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        layout.addSpacerItem(self._bottom_spacer)
        
        # 设置最小宽度
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)

    def add_bottom_widget(self, widget: QtWidgets.QWidget):
        """把外部组件插入到控制面板左下区域（在弹性空间之前）"""
        if widget is None:
            return
        # 避免重复插入
        if widget.parent() is self:
            return
        widget.setParent(self)
        insert_idx = max(0, self.main_layout.count() - 1)
        self.main_layout.insertWidget(insert_idx, widget)
    
    def _on_sample_clicked(self):
        """采样按钮点击"""
        sample_size = self.sample_size_spin.value()
        seed = self.seed_spin.value()
        if seed == -1:
            seed = None
        self.sample_requested.emit(sample_size, seed)
    
    def _on_play_clicked(self):
        """播放按钮点击"""
        if self.play_btn.text().startswith("▶"):
            # 开始播放
            params = {
                "swing_window": self.swing_window_spin.value(),
            }
            self.label_requested.emit(params)
        else:
            # 暂停
            self.pause_requested.emit()
    
    def _on_stop_clicked(self):
        """停止按钮点击"""
        self.stop_requested.emit()
    
    def _on_quick_label_clicked(self):
        """仅标注按钮点击 - 快速计算标注，不播放动画"""
        params = {
            "swing_window": self.swing_window_spin.value(),
        }
        self.quick_label_requested.emit(params)
    
    def _on_speed_changed(self, value):
        """速度滑块变化"""
        self.speed_label.setText(f"{value}x")
        self.speed_changed.emit(value)
    
    def _on_analyze_clicked(self):
        """分析按钮点击（UI已移除，保留方法以便未来使用）"""
        self.analyze_requested.emit()
    
    def _on_optimize_clicked(self):
        """优化按钮点击（UI已移除，保留方法以便未来使用）"""
        # UI控件已移除，使用默认配置
        from config import GA_CONFIG
        params = {
            "population_size": GA_CONFIG.get("POPULATION_SIZE", 50),
            "max_generations": GA_CONFIG.get("MAX_GENERATIONS", 100),
            "mutation_rate": GA_CONFIG.get("MUTATION_RATE", 0.1),
        }
        self.optimize_requested.emit(params)
    
    def set_time_range(self, start_time: str, end_time: str):
        """设置时间范围显示"""
        if start_time and end_time:
            self.time_range_label.setText(f"时间范围:\n{start_time}\n至 {end_time}")
        else:
            self.time_range_label.setText("时间范围: --")
    
    def set_playing_state(self, is_playing: bool):
        """设置播放状态"""
        if is_playing:
            self.play_btn.setText("⏸ 暂停")
            self.stop_btn.setEnabled(True)
            self.sample_btn.setEnabled(False)
        else:
            self.play_btn.setText("▶ 开始回测")
            self.stop_btn.setEnabled(False)
            self.sample_btn.setEnabled(True)
    
    def update_play_progress(self, current: int, total: int):
        """更新播放进度"""
        self.play_progress_label.setText(f"进度: {current:,} / {total:,}")
    
    def get_speed(self) -> int:
        """获取当前速度"""
        return self.speed_slider.value()

    def get_strict_state_filter(self) -> bool:
        """获取严格市场状态过滤开关状态"""
        return self.strict_state_filter_chk.isChecked()

    def get_alt_tpsl(self) -> bool:
        """获取回测试验TP/SL开关状态"""
        return self.alt_tpsl_chk.isChecked()
    
    def set_status(self, text: str):
        """设置状态文本"""
        self.status_label.setText(text)
    
    def set_progress(self, value: int, maximum: int = 100):
        """设置进度"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """隐藏进度条"""
        self.progress_bar.setVisible(False)
    
    def set_buttons_enabled(self, enabled: bool):
        """设置按钮可用性"""
        self.sample_btn.setEnabled(enabled)
        self.quick_label_btn.setEnabled(enabled)
        # analyze_btn 和 optimize_btn 已从UI移除，保留方法以便未来扩展
    
    def get_labeling_params(self) -> Dict:
        """获取当前标注参数"""
        return {
            "swing_window": self.swing_window_spin.value(),
        }
    
    def get_ga_params(self) -> Dict:
        """获取当前 GA 参数（UI已移除，返回默认配置）"""
        from config import GA_CONFIG
        return {
            "population_size": GA_CONFIG.get("POPULATION_SIZE", 50),
            "max_generations": GA_CONFIG.get("MAX_GENERATIONS", 100),
            "mutation_rate": GA_CONFIG.get("MUTATION_RATE", 0.1),
        }
