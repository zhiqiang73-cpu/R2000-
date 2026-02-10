"""
R3000 主窗口
PyQt6 主窗口：深色主题、动态 K 线播放、标注可视化
"""
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
import pandas as pd
from typing import Optional
import sys
import os
import time
import traceback
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (UI_CONFIG, DATA_CONFIG, LABEL_BACKTEST_CONFIG,
                    MARKET_REGIME_CONFIG, VECTOR_SPACE_CONFIG,
                    TRAJECTORY_CONFIG, WALK_FORWARD_CONFIG, MEMORY_CONFIG)
from ui.chart_widget import ChartWidget
from ui.control_panel import ControlPanel
from ui.analysis_panel import AnalysisPanel
from ui.optimizer_panel import OptimizerPanel


class LabelingWorker(QtCore.QObject):
    """标注工作者 - 先显示K线动画，同时在后台计算标注"""
    step_completed = QtCore.pyqtSignal(int)         # 当前索引
    label_found = QtCore.pyqtSignal(int, int)       # (索引, 标注类型)
    labeling_started = QtCore.pyqtSignal()          # 标注计算开始
    labeling_progress = QtCore.pyqtSignal(str)      # 标注计算进度
    labels_ready = QtCore.pyqtSignal(object)        # 标注序列就绪
    finished = QtCore.pyqtSignal(object)            # 标注结果
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, df, params):
        super().__init__()
        self.df = df
        self.params = params
        self.is_running = False
        self._stop_requested = False
        self._pause_requested = False
        self.speed = UI_CONFIG["DEFAULT_SPEED"]
        self.current_idx = 0
        
        # 标注结果
        self.labels = None
        self.labeler = None
        self._labels_ready = False
    
    @QtCore.pyqtSlot()
    def run_labeling(self):
        """执行标注并逐步播放 - 分离计算和播放"""
        try:
            import threading
            from core.labeler import GodViewLabeler
            
            n = len(self.df)
            self.is_running = True
            self._stop_requested = False
            self._pause_requested = False
            self._labels_ready = False
            self.current_idx = 0
            
            # 在后台线程计算标注
            def compute_labels():
                try:
                    self.labeling_started.emit()
                    self.labeling_progress.emit("正在计算上帝视角标注...")
                    
                    self.labeler = GodViewLabeler(
                        swing_window=self.params.get('swing_window')
                    )
                    
                    self.labels = self.labeler.label(self.df)
                    self._labels_ready = True
                    self.labels_ready.emit(self.labels)
                    self.labeling_progress.emit("标注计算完成，正在播放...")
                except Exception as e:
                    self.error.emit(str(e) + "\n" + traceback.format_exc())
            
            # 启动标注计算线程
            label_thread = threading.Thread(target=compute_labels, daemon=True)
            label_thread.start()
            
            # 同时开始 K 线动画播放
            last_emit_time = 0
            min_emit_interval = 0.04  # 25 FPS
            
            while self.is_running and not self._stop_requested and self.current_idx < n:
                # 检查暂停
                while self._pause_requested and not self._stop_requested:
                    time.sleep(0.1)
                
                if self._stop_requested:
                    break
                
                # 发送步骤完成信号 - K线前进
                now = time.time()
                if self.speed <= 10 or (now - last_emit_time) >= min_emit_interval:
                    self.step_completed.emit(self.current_idx)
                    
                    # 如果标注已计算完成，检查是否有标注
                    if self._labels_ready and self.labels is not None:
                        if self.current_idx < len(self.labels):
                            label_val = self.labels.iloc[self.current_idx]
                            if label_val != 0:
                                self.label_found.emit(self.current_idx, int(label_val))
                    
                    last_emit_time = now
                
                self.current_idx += 1
                
                # 速度控制: 10x = 每秒1根K线
                sleep_time = 10.0 / max(1, self.speed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 等待标注计算完成
            label_thread.join(timeout=30)
            
            # 完成
            self.finished.emit({
                'labels': self.labels,
                'labeler': self.labeler,
                'stats': self.labeler.get_statistics() if self.labeler else {}
            })
            
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())
        
        self.is_running = False
    
    def pause(self):
        """暂停"""
        self._pause_requested = True
    
    def resume(self):
        """恢复"""
        self._pause_requested = False
    
    def stop(self):
        """停止"""
        self._stop_requested = True
        self._pause_requested = False
        self.is_running = False
    
    def set_speed(self, speed: int):
        """设置速度"""
        self.speed = speed


class DataLoaderWorker(QtCore.QObject):
    """数据加载工作者"""
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, sample_size, seed):
        super().__init__()
        self.sample_size = sample_size
        self.seed = seed
    
    @QtCore.pyqtSlot()
    def process(self):
        try:
            from core.data_loader import DataLoader
            from utils.indicators import calculate_all_indicators
            
            loader = DataLoader()
            df = loader.sample_continuous(self.sample_size, self.seed)
            df = calculate_all_indicators(df)
            mtf_data = loader.get_mtf_data()
            
            self.finished.emit({
                'df': df,
                'mtf_data': mtf_data,
                'loader': loader
            })
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class AnalyzeWorker(QtCore.QObject):
    """分析工作者"""
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, df, labels, mtf_data, labeler):
        super().__init__()
        self.df = df
        self.labels = labels
        self.mtf_data = mtf_data
        self.labeler = labeler
    
    @QtCore.pyqtSlot()
    def process(self):
        try:
            from core.features import FeatureExtractor
            from core.pattern_miner import PatternMiner
            
            extractor = FeatureExtractor()
            features = extractor.extract_all_features(self.df, self.mtf_data)
            feature_names = extractor.get_feature_names()
            
            labeled_features, label_values = extractor.extract_at_labels(
                self.df, self.labels, self.mtf_data
            )
            
            miner = PatternMiner()
            trades = self.labeler.optimal_trades if self.labeler else []
            
            analysis_results = miner.analyze_all(
                labeled_features, label_values, feature_names, trades
            )
            
            self.finished.emit({
                'features': features,
                'feature_names': feature_names,
                'analysis_results': analysis_results,
                'extractor': extractor,
                'miner': miner
            })
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class BacktestCatchupWorker(QtCore.QObject):
    """标注回测追赶工作者（避免主线程卡顿）"""
    finished = QtCore.pyqtSignal(object, object, int)  # backtester, result, last_idx
    error = QtCore.pyqtSignal(str)

    def __init__(self, df, labels, end_idx, cfg):
        super().__init__()
        self.df = df
        self.labels = labels
        self.end_idx = end_idx
        self.cfg = cfg

    @QtCore.pyqtSlot()
    def process(self):
        try:
            from core.backtester import Backtester

            backtester = Backtester(
                initial_capital=self.cfg["INITIAL_CAPITAL"],
                leverage=self.cfg["LEVERAGE"],
                fee_rate=self.cfg["FEE_RATE"],
                slippage=self.cfg["SLIPPAGE"],
                position_size_pct=self.cfg["POSITION_SIZE_PCT"],
            )

            for i in range(0, self.end_idx + 1):
                label = int(self.labels.iloc[i]) if self.labels is not None else 0
                close = float(self.df['close'].iloc[i])
                high = float(self.df['high'].iloc[i])
                low = float(self.df['low'].iloc[i])
                backtester.step_with_label(i, close, high, low, label)

            result = backtester.get_realtime_result()
            self.finished.emit(backtester, result, self.end_idx)
        except Exception as e:
            self.error.emit(str(e) + "\n" + traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    """
    R3000 主窗口 - 深色主题
    
    布局：
    - 左侧：控制面板
    - 中央：K线图表（动态播放）
    - 右侧：分析面板
    - 底部：优化器面板
    """

    # GA 完成信号
    _ga_done_signal = QtCore.pyqtSignal(float)
    # Walk-Forward 完成信号
    _wf_done_signal = QtCore.pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        
        # 数据存储
        self.df: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.features: Optional[np.ndarray] = None
        self.mtf_data = {}
        
        # 核心模块
        self.data_loader = None
        self.labeler = None
        self.feature_extractor = None
        self.pattern_miner = None
        self.optimizer = None
        
        # 工作线程
        self.worker_thread: Optional[QtCore.QThread] = None
        self.labeling_worker: Optional[LabelingWorker] = None
        self.is_playing = False
        self.rt_backtester = None
        self.rt_last_idx = -1
        self.rt_last_trade_count = 0
        self.rt_catchup_thread: Optional[QtCore.QThread] = None
        self.rt_catchup_worker: Optional[BacktestCatchupWorker] = None
        self._labels_ready = False
        
        # 市场状态分类器
        self.regime_classifier = None
        self.regime_map: dict = {}  # {trade_index: regime_string}
        
        # 向量空间引擎和记忆体
        self.fv_engine = None       # FeatureVectorEngine
        self.vector_memory = None   # VectorMemory
        self._fv_ready = False
        self._ga_running = False

        # 轨迹匹配相关
        self.trajectory_memory = None

        # GA 完成信号（analysis_panel 在后续 _init_ui 中创建后再连接按钮）
        self._ga_done_signal.connect(self._on_ga_finished)
        # Walk-Forward 完成信号
        self._wf_done_signal.connect(self._on_walk_forward_finished)
        
        self._init_ui()
        self._connect_signals()

        # 自动加载已有记忆（如果配置了）
        self._auto_load_memory()
    
    def _init_ui(self):
        """初始化 UI - 深色主题"""
        self.setWindowTitle(UI_CONFIG["WINDOW_TITLE"])
        self.resize(UI_CONFIG["WINDOW_WIDTH"], UI_CONFIG["WINDOW_HEIGHT"])
        
        # 深色主题样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QMenuBar {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QMenuBar::item:selected {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
            }}
            QMenu {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #444;
            }}
            QMenu::item:selected {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
            }}
            QStatusBar {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QSplitter::handle {{
                background-color: #444;
            }}
        """)
        
        # 中央组件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 左侧控制面板
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel)
        
        # 中央区域（图表 + 优化器）
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)
        
        # K线图表
        self.chart_widget = ChartWidget()
        center_layout.addWidget(self.chart_widget, stretch=3)
        
        # 优化器面板
        self.optimizer_panel = OptimizerPanel()
        self.optimizer_panel.setMaximumHeight(280)
        center_layout.addWidget(self.optimizer_panel, stretch=1)
        
        main_layout.addWidget(center_widget, stretch=1)
        
        # 右侧分析面板
        self.analysis_panel = AnalysisPanel()
        main_layout.addWidget(self.analysis_panel)

        # 注：旧的向量空间GA按钮已移除，指纹图使用不同的交互方式
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 菜单栏
        self._create_menus()
    
    def _create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        load_action = QtGui.QAction("加载数据(&L)", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._on_load_data)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QtGui.QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        self.show_optimizer_action = QtGui.QAction("显示优化器面板", self)
        self.show_optimizer_action.setCheckable(True)
        self.show_optimizer_action.setChecked(True)
        self.show_optimizer_action.triggered.connect(self._toggle_optimizer_panel)
        view_menu.addAction(self.show_optimizer_action)
        
        self.show_analysis_action = QtGui.QAction("显示分析面板", self)
        self.show_analysis_action.setCheckable(True)
        self.show_analysis_action.setChecked(True)
        self.show_analysis_action.triggered.connect(self._toggle_analysis_panel)
        view_menu.addAction(self.show_analysis_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QtGui.QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self):
        """连接信号"""
        self.control_panel.sample_requested.connect(self._on_sample_requested)
        self.control_panel.label_requested.connect(self._on_label_requested)
        self.control_panel.analyze_requested.connect(self._on_analyze_requested)
        self.control_panel.optimize_requested.connect(self._on_optimize_requested)
        self.control_panel.pause_requested.connect(self._on_pause_requested)
        self.control_panel.stop_requested.connect(self._on_stop_requested)
        self.control_panel.speed_changed.connect(self._on_speed_changed)

        # 轨迹匹配相关
        self.analysis_panel.trajectory_widget.walk_forward_requested.connect(
            self._on_walk_forward_requested
        )

        # 记忆管理
        self.analysis_panel.trajectory_widget.save_memory_requested.connect(
            self._on_save_memory
        )
        self.analysis_panel.trajectory_widget.load_memory_requested.connect(
            self._on_load_memory
        )
        self.analysis_panel.trajectory_widget.clear_memory_requested.connect(
            self._on_clear_memory
        )
        self.analysis_panel.trajectory_widget.merge_all_requested.connect(
            self._on_merge_all_memory
        )
    
    def _on_load_data(self):
        """加载数据"""
        self._on_sample_requested(DATA_CONFIG["SAMPLE_SIZE"], None)
    
    def _on_sample_requested(self, sample_size: int, seed):
        """处理采样请求"""
        self.control_panel.set_status("正在加载数据...")
        self.control_panel.set_buttons_enabled(False)
        self.statusBar().showMessage("正在加载数据...")
        
        # 创建工作线程
        self.worker_thread = QtCore.QThread()
        self.data_worker = DataLoaderWorker(sample_size, seed)
        self.data_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.data_worker.process)
        self.data_worker.finished.connect(self._on_sample_finished)
        self.data_worker.error.connect(self._on_worker_error)
        self.data_worker.finished.connect(self.worker_thread.quit)
        self.data_worker.error.connect(self.worker_thread.quit)
        
        self.worker_thread.start()
    
    def _on_sample_finished(self, result):
        """采样完成"""
        self.df = result['df']
        self.mtf_data = result['mtf_data']
        self.data_loader = result['loader']
        self.labels = None
        self.features = None
        
        # 更新图表
        self.chart_widget.set_data(self.df, show_all=True)
        
        # 显示时间范围
        start_time, end_time = self.chart_widget.get_data_time_range()
        self.control_panel.set_time_range(start_time, end_time)
        
        self.control_panel.set_status(f"已加载 {len(self.df):,} 根 K 线")
        self.control_panel.set_buttons_enabled(True)
        self.statusBar().showMessage(f"数据加载完成: {len(self.df):,} 根 K 线 | {start_time} 至 {end_time}")
    
    def _on_label_requested(self, params: dict):
        """处理标注请求 - 开始动画播放"""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载数据")
            return
        
        if self.is_playing:
            # 如果正在播放，则暂停/恢复
            if self.labeling_worker:
                if self.control_panel.play_btn.text().startswith("⏸"):
                    self.labeling_worker.pause()
                    self.control_panel.play_btn.setText("▶ 继续")
                else:
                    self.labeling_worker.resume()
                    self.control_panel.play_btn.setText("⏸ 暂停")
            return
        
        # 开始新的标注播放
        self.is_playing = True
        self._labels_ready = False
        self.rt_last_idx = -1
        self.rt_backtester = None
        self.rt_last_trade_count = 0
        self.regime_classifier = None
        self.regime_map = {}
        self.fv_engine = None
        self.vector_memory = None
        self._fv_ready = False
        self.analysis_panel.update_trade_log([])
        self.analysis_panel.fingerprint_widget.clear_plot()
        self.control_panel.set_playing_state(True)
        self.control_panel.set_status("正在执行上帝视角标注...")
        self.statusBar().showMessage("正在标注...")
        
        # 重置图表
        self.chart_widget.set_data(self.df, show_all=False)
        
        # 创建标注工作线程
        self.worker_thread = QtCore.QThread()
        self.labeling_worker = LabelingWorker(self.df, params)
        self.labeling_worker.speed = self.control_panel.get_speed()
        self.labeling_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.labeling_worker.run_labeling)
        self.labeling_worker.step_completed.connect(self._on_labeling_step, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.label_found.connect(self._on_label_found, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.labeling_progress.connect(self._on_labeling_progress, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.labels_ready.connect(self._on_labels_ready, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.finished.connect(self._on_labeling_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.error.connect(self._on_worker_error, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.finished.connect(self.worker_thread.quit)
        self.labeling_worker.error.connect(self.worker_thread.quit)
        
        self.worker_thread.start()
    
    def _on_labeling_step(self, idx: int):
        """标注步骤完成"""
        # 前进一根 K 线
        self.chart_widget.advance_one_candle()
        
        # 更新进度
        total = len(self.df) if self.df is not None else 0
        self.control_panel.update_play_progress(idx + 1, total)

        # 实时回测统计
        if self.df is not None and self.labels is not None and self._labels_ready and self.rt_backtester is not None:
            if idx > self.rt_last_idx:
                label_val = int(self.labels.iloc[idx]) if idx < len(self.labels) else 0
                close = float(self.df['close'].iloc[idx])
                high = float(self.df['high'].iloc[idx])
                low = float(self.df['low'].iloc[idx])
                bt_result = self.rt_backtester.step_with_label(idx, close, high, low, label_val)
                self.rt_last_idx = idx

                metrics = {
                    "initial_capital": bt_result.initial_capital,
                    "total_trades": bt_result.total_trades,
                    "win_rate": bt_result.win_rate,
                    "total_return": bt_result.total_return_pct / 100.0,
                    "total_profit": bt_result.total_profit,
                    "max_drawdown": bt_result.max_drawdown,
                    "sharpe_ratio": bt_result.sharpe_ratio,
                    "profit_factor": bt_result.profit_factor,
                    "long_win_rate": bt_result.long_win_rate,
                    "long_profit": bt_result.long_profit,
                    "short_win_rate": bt_result.short_win_rate,
                    "short_profit": bt_result.short_profit,
                    "current_pos": bt_result.current_pos,
                    "last_trade": bt_result.trades[-1] if bt_result.trades else None
                }
                self.optimizer_panel.update_backtest_metrics(metrics)

                # 仅在交易数量变化时刷新明细 + 市场状态 + 向量
                if self.rt_backtester is not None and len(self.rt_backtester.trades) != self.rt_last_trade_count:
                    new_count = len(self.rt_backtester.trades)
                    for ti in range(self.rt_last_trade_count, new_count):
                        trade = self.rt_backtester.trades[ti]
                        # 市场状态分类
                        if self.regime_classifier is not None:
                            regime = self.regime_classifier.classify_at(trade.entry_idx)
                            trade.market_regime = regime
                            self.regime_map[ti] = regime
                        # 向量坐标记录
                        if self._fv_ready and self.fv_engine:
                            self._record_trade_vectors(trade)
                    self.rt_last_trade_count = new_count
                    self.analysis_panel.update_trade_log(self._format_trades(self.rt_backtester.trades))
                    self._update_regime_stats()
                    # 每10笔交易刷新一次3D图（节省性能）
                    if new_count % 10 == 0 or new_count < 20:
                        self._update_vector_space_plot()

                # 实时更新当前K线的市场状态
                if self.regime_classifier is not None:
                    current_regime = self.regime_classifier.classify_at(idx)
                    self.analysis_panel.market_regime_widget.update_current_regime(current_regime)
    
    def _on_label_found(self, idx: int, label_type: int):
        """发现标注点"""
        label_map = {
            1: "LONG 入场",
            2: "LONG 出场",
            -1: "SHORT 入场",
            -2: "SHORT 出场"
        }
        label_str = label_map.get(label_type, "未知")
        self.statusBar().showMessage(f"发现 {label_str} 信号 @ 索引 {idx}")
        
        # 更新图表上的标记
        if self.df is not None and self.labeling_worker and self.labeling_worker.labels is not None:
            self.chart_widget.add_signal_at(idx, label_type, self.df)
    
    def _on_labeling_progress(self, msg: str):
        """标注进度更新"""
        self.control_panel.set_status(msg)
        self.statusBar().showMessage(msg)

    def _on_labels_ready(self, labels: pd.Series):
        """标注结果就绪（播放过程中展示标记）"""
        self.labels = labels
        self._labels_ready = True
        self.chart_widget.set_labels(labels)

        # 创建市场状态分类器
        if self.labeling_worker and self.labeling_worker.labeler:
            try:
                from core.market_regime import MarketRegimeClassifier
                alt_swings = self.labeling_worker.labeler.alternating_swings
                if alt_swings:
                    self.regime_classifier = MarketRegimeClassifier(
                        alt_swings, MARKET_REGIME_CONFIG
                    )
                    print(f"[MarketRegime] 分类器就绪, 交替摆动点: {len(alt_swings)}")
            except Exception as e:
                print(f"[MarketRegime] 初始化失败: {e}")

        # 初始化特征向量引擎 + 记忆体
        if self.df is not None:
            try:
                from core.feature_vector import FeatureVectorEngine
                from core.vector_memory import VectorMemory
                self.fv_engine = FeatureVectorEngine()
                self.fv_engine.precompute(self.df)
                self.vector_memory = VectorMemory(
                    k_neighbors=VECTOR_SPACE_CONFIG["K_NEIGHBORS"],
                    min_points=VECTOR_SPACE_CONFIG["MIN_CLOUD_POINTS"],
                )
                self._fv_ready = True
                print("[FeatureVector] 引擎和记忆体就绪")
            except Exception as e:
                print(f"[FeatureVector] 初始化失败: {e}")
                traceback.print_exc()

        # 启动回测追赶（避免主线程卡顿）
        if self.df is not None:
            end_idx = max(0, self.chart_widget.current_display_index - 1)
            self.rt_catchup_thread = QtCore.QThread()
            self.rt_catchup_worker = BacktestCatchupWorker(self.df, self.labels, end_idx, LABEL_BACKTEST_CONFIG)
            self.rt_catchup_worker.moveToThread(self.rt_catchup_thread)

            self.rt_catchup_thread.started.connect(self.rt_catchup_worker.process)
            self.rt_catchup_worker.finished.connect(self._on_rt_catchup_finished)
            self.rt_catchup_worker.error.connect(self._on_worker_error)
            self.rt_catchup_worker.finished.connect(self.rt_catchup_thread.quit)
            self.rt_catchup_worker.error.connect(self.rt_catchup_thread.quit)

            self.rt_catchup_thread.start()

    def _on_rt_catchup_finished(self, backtester, result, last_idx: int):
        """回测追赶完成"""
        self.rt_backtester = backtester
        self.rt_last_idx = last_idx

        metrics = {
            "initial_capital": result.initial_capital,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_return": result.total_return_pct / 100.0,
            "total_profit": result.total_profit,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "profit_factor": result.profit_factor,
            "long_win_rate": result.long_win_rate,
            "long_profit": result.long_profit,
            "short_win_rate": result.short_win_rate,
            "short_profit": result.short_profit,
            "current_pos": result.current_pos,
            "last_trade": result.trades[-1] if result.trades else None
        }
        self.optimizer_panel.update_backtest_metrics(metrics)
        self.rt_last_trade_count = len(self.rt_backtester.trades) if self.rt_backtester else 0

        # 为追赶期间产生的所有交易分类市场状态 + 填充向量记忆体
        if self.rt_backtester:
            for ti, trade in enumerate(self.rt_backtester.trades):
                if self.regime_classifier is not None:
                    regime = self.regime_classifier.classify_at(trade.entry_idx)
                    trade.market_regime = regime
                    self.regime_map[ti] = regime
                # 填充向量坐标和记忆体
                if self._fv_ready and self.fv_engine:
                    self._record_trade_vectors(trade)

        if self.rt_backtester:
            self.analysis_panel.update_trade_log(self._format_trades(self.rt_backtester.trades))
        self._update_regime_stats()
        self._update_vector_space_plot()

    def _format_trades(self, trades):
        """格式化交易明细（仅展示最近200条）"""
        if self.df is None:
            return []

        time_col = None
        for col in ['timestamp', 'open_time', 'time']:
            if col in self.df.columns:
                time_col = col
                break

        def fmt_time(idx):
            if time_col is None:
                return str(idx)
            ts = self.df[time_col].iloc[idx]
            try:
                if isinstance(ts, (int, float)):
                    return QtCore.QDateTime.fromSecsSinceEpoch(int(ts / 1000)).toString("MM-dd HH:mm")
                return pd.to_datetime(ts).strftime('%m-%d %H:%M')
            except Exception:
                return str(idx)

        rows = []
        for t in trades[-200:]:
            side = "LONG" if t.side == 1 else "SHORT"
            entry_abc = getattr(t, 'entry_abc', (0, 0, 0))
            rows.append({
                "side": side,
                "entry_time": fmt_time(t.entry_idx),
                "entry_price": f"{t.entry_price:.2f}",
                "exit_time": fmt_time(t.exit_idx),
                "exit_price": f"{t.exit_price:.2f}",
                "profit": f"{t.profit:.2f}",
                "profit_pct": f"{t.profit_pct:.2f}",
                "hold": str(t.hold_periods),
                "regime": getattr(t, 'market_regime', ''),
                "abc": f"({entry_abc[0]:.1f},{entry_abc[1]:.1f},{entry_abc[2]:.1f})",
            })
        return rows
    
    def _record_trade_vectors(self, trade):
        """为一笔交易记录入场和离场的 ABC 向量坐标到记忆体"""
        if not self._fv_ready or self.fv_engine is None or self.vector_memory is None:
            return
        regime = getattr(trade, 'market_regime', '') or '未知'
        direction = "LONG" if trade.side == 1 else "SHORT"

        # 入场坐标
        entry_abc = self.fv_engine.get_abc(trade.entry_idx)
        trade.entry_abc = entry_abc
        self.vector_memory.add_point(regime, direction, "ENTRY", *entry_abc)

        # 离场坐标
        exit_abc = self.fv_engine.get_abc(trade.exit_idx)
        trade.exit_abc = exit_abc
        self.vector_memory.add_point(regime, direction, "EXIT", *exit_abc)

    def _update_vector_space_plot(self):
        """更新向量空间/指纹图（兼容旧调用）"""
        # 向量空间3D散点图已替换为指纹地形图
        # 指纹图的更新通过 _update_fingerprint_view 方法
        pass

    def _update_fingerprint_view(self):
        """更新指纹图3D地形视图"""
        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            return

        try:
            templates = self.trajectory_memory.get_all_templates()
            self.analysis_panel.update_fingerprint_templates(templates)
        except Exception as e:
            print(f"[Fingerprint] 3D图更新失败: {e}")

    def _on_ga_optimize(self):
        """GA 优化权重按钮点击（向量空间旧功能，已废弃）"""
        # 旧的ABC向量空间GA优化已移除
        # 新的轨迹匹配使用 GATradingOptimizer 通过 Walk-Forward 验证
        pass

    def _on_ga_finished(self, fitness: float):
        """GA 优化完成（旧功能，保留信号处理）"""
        self._ga_running = False
        if fitness >= 0:
            self.statusBar().showMessage(f"GA 优化完成! 适应度: {fitness:.4f}")
        else:
            self.statusBar().showMessage("GA 优化失败")

    # ══════════════════════════════════════════════════════════════════════════
    # 轨迹匹配相关方法
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_trajectory_templates(self, trades):
        """提取轨迹模板"""
        if not self._fv_ready or self.fv_engine is None:
            return

        try:
            from core.trajectory_engine import TrajectoryMemory

            # 检查是否已有记忆体，如果有则合并，否则新建
            if hasattr(self, 'trajectory_memory') and self.trajectory_memory is not None:
                # 提取新模板到临时记忆体
                new_memory = TrajectoryMemory()
                n_new = new_memory.extract_from_trades(
                    trades, self.fv_engine, self.regime_map, verbose=False
                )
                # 合并到现有记忆体
                if n_new > 0:
                    added = self.trajectory_memory.merge(
                        new_memory,
                        deduplicate=MEMORY_CONFIG.get("DEDUPLICATE", True),
                        verbose=True
                    )
                    n_templates = self.trajectory_memory.total_count
                    print(f"[TrajectoryMemory] 增量合并: 新增 {added} 个模板, 总计 {n_templates}")
                else:
                    n_templates = self.trajectory_memory.total_count
            else:
                # 新建记忆体
                self.trajectory_memory = TrajectoryMemory()
                n_templates = self.trajectory_memory.extract_from_trades(
                    trades, self.fv_engine, self.regime_map
                )

            if n_templates > 0:
                # 更新 UI 统计
                self._update_trajectory_ui()
                self._update_memory_stats()

                # 启用 Walk-Forward 验证按钮
                self.analysis_panel.enable_walk_forward(True)

                # 自动保存（如果配置了）
                if MEMORY_CONFIG.get("AUTO_SAVE", True):
                    try:
                        filepath = self.trajectory_memory.save(verbose=False)
                        print(f"[TrajectoryMemory] 自动保存: {filepath}")
                        self._update_memory_stats()
                    except Exception as save_err:
                        print(f"[TrajectoryMemory] 自动保存失败: {save_err}")

            else:
                print("[TrajectoryMemory] 无盈利交易可提取模板")

        except Exception as e:
            print(f"[TrajectoryMemory] 模板提取失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_walk_forward_requested(self):
        """Walk-Forward 验证请求"""
        if self.df is None or self.labels is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先完成标注")
            return

        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先提取轨迹模板")
            return

        # 禁用按钮，显示运行中状态
        self.analysis_panel.enable_walk_forward(False)
        self.analysis_panel.update_walk_forward_result(0, 0, 0, "运行中...")
        self.statusBar().showMessage("Walk-Forward 验证运行中...")

        # 在后台线程运行
        self._wf_thread = threading.Thread(target=self._run_walk_forward)
        self._wf_thread.start()

    def _run_walk_forward(self):
        """在后台运行 Walk-Forward 验证"""
        try:
            from core.walk_forward import WalkForwardValidator

            validator = WalkForwardValidator()
            result = validator.run(
                self.df, self.labels,
                n_folds=WALK_FORWARD_CONFIG["N_FOLDS"],
                callback=self._wf_progress_callback
            )

            # 通过信号更新 UI
            self._wf_done_signal.emit(result)

        except Exception as e:
            print(f"[WalkForward] 验证失败: {e}")
            import traceback
            traceback.print_exc()
            # 发送空结果
            from core.walk_forward import WalkForwardResult
            self._wf_done_signal.emit(WalkForwardResult())

    def _wf_progress_callback(self, fold_idx: int, stage: str, message: str):
        """Walk-Forward 进度回调"""
        status = f"Walk-Forward: Fold {fold_idx + 1} - {stage}"
        QtCore.QMetaObject.invokeMethod(
            self.statusBar(), "showMessage",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, status)
        )

    def _on_walk_forward_finished(self, result):
        """Walk-Forward 验证完成"""
        self.analysis_panel.enable_walk_forward(True)

        if not result.folds:
            self.analysis_panel.update_walk_forward_result(0, 0, 0, "失败")
            self.statusBar().showMessage("Walk-Forward 验证失败")
            return

        result.summarize()

        self.analysis_panel.update_walk_forward_result(
            result.avg_test_sharpe,
            result.consistency_ratio,
            result.avg_test_profit,
            "完成"
        )

        # 如果有最优参数，更新显示
        if result.folds and result.folds[0].best_params:
            self.analysis_panel.update_trading_params(result.folds[0].best_params)

        result.print_summary()
        self.statusBar().showMessage(
            f"Walk-Forward 完成: 平均Sharpe={result.avg_test_sharpe:.3f}, "
            f"一致性={result.consistency_ratio:.0%}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 记忆持久化管理
    # ══════════════════════════════════════════════════════════════════════════

    def _auto_load_memory(self):
        """启动时自动加载已有记忆"""
        if not MEMORY_CONFIG.get("AUTO_LOAD", True):
            self._update_memory_stats()
            return

        try:
            from core.trajectory_engine import TrajectoryMemory

            files = TrajectoryMemory.list_saved_memories()
            if not files:
                print("[TrajectoryMemory] 启动: 无历史记忆文件")
                self._update_memory_stats()
                return

            # 加载最新的记忆文件
            memory = TrajectoryMemory.load(files[0]["path"], verbose=True)
            if memory and memory.total_count > 0:
                self.trajectory_memory = memory
                self._update_memory_stats()
                self._update_trajectory_ui()
                print(f"[TrajectoryMemory] 自动加载: {memory.total_count} 个模板")
            else:
                self._update_memory_stats()

        except Exception as e:
            print(f"[TrajectoryMemory] 自动加载失败: {e}")
            self._update_memory_stats()

    def _update_memory_stats(self):
        """更新记忆统计显示"""
        template_count = 0
        if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
            template_count = self.trajectory_memory.total_count

        from core.trajectory_engine import TrajectoryMemory
        files = TrajectoryMemory.list_saved_memories()
        file_count = len(files)

        self.analysis_panel.update_memory_stats(template_count, file_count)

    def _on_save_memory(self):
        """保存当前记忆体到本地"""
        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            QtWidgets.QMessageBox.warning(self, "警告", "没有可保存的记忆体")
            return

        if self.trajectory_memory.total_count == 0:
            QtWidgets.QMessageBox.warning(self, "警告", "记忆体为空")
            return

        try:
            filepath = self.trajectory_memory.save()
            self._update_memory_stats()
            QtWidgets.QMessageBox.information(
                self, "保存成功",
                f"已保存 {self.trajectory_memory.total_count} 个模板\n"
                f"文件: {filepath}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "保存失败", str(e))

    def _on_load_memory(self):
        """加载最新的记忆体"""
        try:
            from core.trajectory_engine import TrajectoryMemory

            # 如果配置为合并模式
            if MEMORY_CONFIG.get("MERGE_ON_LOAD", True):
                if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
                    # 从最新文件合并
                    files = TrajectoryMemory.list_saved_memories()
                    if files:
                        added = self.trajectory_memory.merge_from_file(
                            files[0]["path"],
                            deduplicate=MEMORY_CONFIG.get("DEDUPLICATE", True)
                        )
                        self._update_memory_stats()
                        self._update_trajectory_ui()
                        self.statusBar().showMessage(f"已合并 {added} 个模板")
                        return
                    else:
                        QtWidgets.QMessageBox.information(self, "提示", "没有找到已保存的记忆文件")
                        return

            # 覆盖加载模式
            memory = TrajectoryMemory.load_latest()
            if memory is None:
                QtWidgets.QMessageBox.information(self, "提示", "没有找到已保存的记忆文件")
                return

            self.trajectory_memory = memory
            self._update_memory_stats()
            self._update_trajectory_ui()
            self.analysis_panel.enable_walk_forward(True)
            self.statusBar().showMessage(f"已加载 {memory.total_count} 个模板")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(e))
            import traceback
            traceback.print_exc()

    def _on_merge_all_memory(self):
        """加载并合并所有历史记忆"""
        try:
            from core.trajectory_engine import TrajectoryMemory

            if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
                # 合并所有文件到当前记忆体
                files = TrajectoryMemory.list_saved_memories()
                if not files:
                    QtWidgets.QMessageBox.information(self, "提示", "没有找到已保存的记忆文件")
                    return

                total_added = 0
                for f in files:
                    added = self.trajectory_memory.merge_from_file(
                        f["path"],
                        deduplicate=True,
                        verbose=False
                    )
                    total_added += added

                self._update_memory_stats()
                self._update_trajectory_ui()
                QtWidgets.QMessageBox.information(
                    self, "合并完成",
                    f"从 {len(files)} 个文件中合并了 {total_added} 个新模板\n"
                    f"当前总模板数: {self.trajectory_memory.total_count}"
                )
            else:
                # 没有当前记忆体，创建并合并全部
                memory = TrajectoryMemory.load_and_merge_all()
                self.trajectory_memory = memory
                self._update_memory_stats()
                self._update_trajectory_ui()
                if memory.total_count > 0:
                    self.analysis_panel.enable_walk_forward(True)
                    QtWidgets.QMessageBox.information(
                        self, "加载完成",
                        f"已加载并合并全部历史记忆\n"
                        f"总模板数: {memory.total_count}"
                    )
                else:
                    QtWidgets.QMessageBox.information(self, "提示", "没有找到历史记忆文件")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "合并失败", str(e))
            import traceback
            traceback.print_exc()

    def _on_clear_memory(self):
        """清空当前记忆体"""
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空",
            "确定要清空当前加载的记忆吗？\n（本地保存的文件不会被删除）",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if hasattr(self, 'trajectory_memory') and self.trajectory_memory:
                self.trajectory_memory.clear()
            self._update_memory_stats()
            self._update_trajectory_ui()
            self.statusBar().showMessage("记忆已清空")

    def _update_trajectory_ui(self):
        """更新轨迹匹配相关的UI"""
        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            self.analysis_panel.update_trajectory_template_stats(0, 0, 0, 0)
            self.analysis_panel.update_fingerprint_templates([])
            return

        memory = self.trajectory_memory
        total = memory.total_count
        long_count = len(memory.get_templates_by_direction("LONG"))
        short_count = len(memory.get_templates_by_direction("SHORT"))
        all_templates = memory.get_all_templates()
        avg_profit = np.mean([t.profit_pct for t in all_templates]) if all_templates else 0

        # 更新轨迹匹配面板统计
        self.analysis_panel.update_trajectory_template_stats(
            total, long_count, short_count, avg_profit
        )

        # 更新指纹图3D地形视图
        self.analysis_panel.update_fingerprint_templates(all_templates)

    def _update_regime_stats(self):
        """更新市场状态统计到 UI"""
        if self.rt_backtester is None or not self.regime_map:
            return
        try:
            from core.market_regime import MarketRegimeClassifier, MarketRegime
            stats = MarketRegimeClassifier.compute_regime_stats(
                self.rt_backtester.trades, self.regime_map
            )
            # 当前市场状态
            current_regime = MarketRegime.UNKNOWN
            if self.regime_classifier is not None and self.chart_widget.current_display_index > 0:
                current_regime = self.regime_classifier.classify_at(
                    self.chart_widget.current_display_index
                )
            self.analysis_panel.update_market_regime(current_regime, stats)
        except Exception as e:
            print(f"[MarketRegime] 统计更新失败: {e}")

    def _on_labeling_finished(self, result):
        """标注完成"""
        self.labels = result['labels']
        self.labeler = result['labeler']
        
        # 显示全部数据和标注
        self.chart_widget.set_data(self.df, self.labels, show_all=True)
        
        # 更新状态 - LONG/SHORT 统计
        long_count = int((self.labels == 1).sum())   # LONG_ENTRY
        short_count = int((self.labels == -1).sum()) # SHORT_ENTRY
        stats = result.get('stats', {})
        
        status_text = f"标注完成: {long_count} LONG + {short_count} SHORT"
        if stats:
            status_text += f" | 平均收益: {stats.get('avg_profit_pct', 0):.2f}%"
        
        self.control_panel.set_status(status_text)
        self.statusBar().showMessage(status_text)

        # 标注回测（基于标记点）
        if self.df is not None and self.labels is not None:
            try:
                from core.backtester import Backtester
                from core.market_regime import MarketRegimeClassifier

                bt_cfg = LABEL_BACKTEST_CONFIG
                backtester = Backtester(
                    initial_capital=bt_cfg["INITIAL_CAPITAL"],
                    leverage=bt_cfg["LEVERAGE"],
                    fee_rate=bt_cfg["FEE_RATE"],
                    slippage=bt_cfg["SLIPPAGE"],
                    position_size_pct=bt_cfg["POSITION_SIZE_PCT"],
                )
                bt_result = backtester.run_with_labels(self.df, self.labels)

                metrics = {
                    "initial_capital": bt_result.initial_capital,
                    "total_trades": bt_result.total_trades,
                    "win_rate": bt_result.win_rate,
                    "total_return": bt_result.total_return_pct / 100.0,
                    "total_profit": bt_result.total_profit,
                    "max_drawdown": bt_result.max_drawdown,
                    "sharpe_ratio": bt_result.sharpe_ratio,
                    "profit_factor": bt_result.profit_factor,
                    "long_win_rate": bt_result.long_win_rate,
                    "long_profit": bt_result.long_profit,
                    "short_win_rate": bt_result.short_win_rate,
                    "short_profit": bt_result.short_profit,
                    "current_pos": bt_result.current_pos,
                    "last_trade": bt_result.trades[-1] if bt_result.trades else None
                }
                self.optimizer_panel.update_backtest_metrics(metrics)

                # 最终市场状态分类 + 向量记忆体构建
                if self.labeler and self.labeler.alternating_swings:
                    classifier = MarketRegimeClassifier(
                        self.labeler.alternating_swings, MARKET_REGIME_CONFIG
                    )
                    self.regime_classifier = classifier
                    self.regime_map = {}

                    # 初始化向量引擎（如果还没有）
                    if not self._fv_ready:
                        try:
                            from core.feature_vector import FeatureVectorEngine
                            from core.vector_memory import VectorMemory
                            self.fv_engine = FeatureVectorEngine()
                            self.fv_engine.precompute(self.df)
                            self.vector_memory = VectorMemory(
                                k_neighbors=VECTOR_SPACE_CONFIG["K_NEIGHBORS"],
                                min_points=VECTOR_SPACE_CONFIG["MIN_CLOUD_POINTS"],
                            )
                            self._fv_ready = True
                        except Exception as fv_err:
                            print(f"[FeatureVector] 最终初始化失败: {fv_err}")
                    else:
                        # 清空旧记忆体重新构建
                        if self.vector_memory:
                            self.vector_memory.clear()

                    for ti, trade in enumerate(bt_result.trades):
                        regime = classifier.classify_at(trade.entry_idx)
                        trade.market_regime = regime
                        self.regime_map[ti] = regime
                        # 记录向量坐标
                        if self._fv_ready and self.fv_engine:
                            self._record_trade_vectors(trade)

                    # 保存回测器引用以便统计
                    self.rt_backtester = backtester
                    self._update_regime_stats()
                    self._update_vector_space_plot()
                    self.analysis_panel.update_trade_log(self._format_trades(bt_result.trades))

                    # 打印记忆体统计
                    if self.vector_memory:
                        stats = self.vector_memory.get_stats()
                        total = self.vector_memory.total_points()
                        print(f"[VectorMemory] 记忆体构建完成: {total} 个点, "
                              f"{len(stats)} 个市场状态")

                    # ── 轨迹模板提取 ──
                    self._extract_trajectory_templates(bt_result.trades)

            except Exception as e:
                self.statusBar().showMessage(f"标注回测失败: {str(e)}")
                traceback.print_exc()
        
        self.is_playing = False
        self.control_panel.set_playing_state(False)
        self.labeling_worker = None
    
    def _on_pause_requested(self):
        """暂停请求"""
        if self.labeling_worker:
            self.labeling_worker.pause()
            self.control_panel.play_btn.setText("▶ 继续")
    
    def _on_stop_requested(self):
        """停止请求"""
        if self.labeling_worker:
            self.labeling_worker.stop()
        
        self.is_playing = False
        self.control_panel.set_playing_state(False)
        
        # 显示已有的标注
        if self.labels is not None:
            self.chart_widget.set_data(self.df, self.labels, show_all=True)
        
        self.statusBar().showMessage("已停止")
    
    def _on_speed_changed(self, speed: int):
        """速度变化"""
        if self.labeling_worker:
            self.labeling_worker.set_speed(speed)
        if self.chart_widget:
            self.chart_widget.set_render_stride(speed)
    
    def _on_analyze_requested(self):
        """处理分析请求"""
        if self.df is None or self.labels is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载数据并执行标注")
            return
        
        self.control_panel.set_status("正在提取特征和分析模式...")
        self.control_panel.set_buttons_enabled(False)
        self.statusBar().showMessage("正在分析...")
        
        # 创建工作线程
        self.worker_thread = QtCore.QThread()
        self.analyze_worker = AnalyzeWorker(self.df, self.labels, self.mtf_data, self.labeler)
        self.analyze_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.analyze_worker.process)
        self.analyze_worker.finished.connect(self._on_analyze_finished)
        self.analyze_worker.error.connect(self._on_worker_error)
        self.analyze_worker.finished.connect(self.worker_thread.quit)
        self.analyze_worker.error.connect(self.worker_thread.quit)
        
        self.worker_thread.start()
    
    def _on_analyze_finished(self, result):
        """分析完成"""
        self.features = result['features']
        self.feature_extractor = result['extractor']
        self.pattern_miner = result['miner']
        
        # 更新分析面板
        self.analysis_panel.update_all(result['analysis_results'])
        
        self.control_panel.set_status("分析完成")
        self.control_panel.set_buttons_enabled(True)
        self.statusBar().showMessage("模式分析完成")
    
    def _on_optimize_requested(self, params: dict):
        """处理优化请求"""
        if self.df is None or self.features is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载数据并执行分析")
            return
        
        self.control_panel.set_status("正在执行遗传算法优化...")
        self.control_panel.set_buttons_enabled(False)
        self.optimizer_panel.reset()
        self.statusBar().showMessage("正在优化...")
        
        # 在主线程中运行（简化处理）
        QtCore.QTimer.singleShot(100, lambda: self._run_optimization(params))
    
    def _run_optimization(self, params):
        """运行优化"""
        try:
            from core.genetic_optimizer import GeneticOptimizer
            
            self.optimizer = GeneticOptimizer(
                population_size=params['population_size'],
                max_generations=params['max_generations'],
                mutation_rate=params['mutation_rate']
            )
            
            # 设置回调
            def on_generation(gen, best):
                self.optimizer_panel.update_progress(gen, params['max_generations'])
                self.optimizer_panel.add_fitness_point(best.fitness)
                QtWidgets.QApplication.processEvents()
            
            self.optimizer.on_generation_complete = on_generation
            
            result = self.optimizer.evolve(self.df, self.features, verbose=True)
            
            # 更新优化器面板
            self.optimizer_panel.update_all(result)
            
            best_fitness = result.best_fitness
            self.control_panel.set_status(f"优化完成: 最优适应度 = {best_fitness:.4f}")
            self.statusBar().showMessage(f"优化完成: 最优适应度 = {best_fitness:.4f}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"优化失败:\n{str(e)}")
        
        self.control_panel.set_buttons_enabled(True)
    
    def _on_worker_error(self, error_msg: str):
        """工作线程错误"""
        self.control_panel.set_status("错误")
        self.control_panel.set_buttons_enabled(True)
        self.control_panel.set_playing_state(False)
        self.is_playing = False
        self.statusBar().showMessage("发生错误")
        
        QtWidgets.QMessageBox.critical(self, "错误", f"操作失败:\n{error_msg}")
    
    def _toggle_optimizer_panel(self, checked: bool):
        """切换优化器面板可见性"""
        self.optimizer_panel.setVisible(checked)
    
    def _toggle_analysis_panel(self, checked: bool):
        """切换分析面板可见性"""
        self.analysis_panel.setVisible(checked)
    
    def _show_about(self):
        """显示关于对话框"""
        QtWidgets.QMessageBox.about(
            self,
            "关于 R3000",
            "R3000 量化 MVP 系统\n\n"
            "功能：\n"
            "• 上帝视角标注：自动识别理想买卖点\n"
            "• 动态 K 线播放：可视化标注过程\n"
            "• 特征提取：52维技术指标特征\n"
            "• 模式挖掘：因果分析、多空逻辑、生存分析\n"
            "• 遗传算法优化：策略参数自动优化\n\n"
            "版本：1.0.0"
        )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.is_playing:
            reply = QtWidgets.QMessageBox.question(
                self,
                "确认退出",
                "标注正在进行中，确定要退出吗？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                if self.labeling_worker:
                    self.labeling_worker.stop()
                if self.worker_thread:
                    self.worker_thread.quit()
                    self.worker_thread.wait(1000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QtWidgets.QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建深色调色板
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(UI_CONFIG['THEME_BACKGROUND']))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(UI_CONFIG['THEME_SURFACE']))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(UI_CONFIG['THEME_BACKGROUND']))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(UI_CONFIG['THEME_SURFACE']))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(UI_CONFIG['THEME_TEXT']))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(UI_CONFIG['THEME_ACCENT']))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor('#ffffff'))
    app.setPalette(palette)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
