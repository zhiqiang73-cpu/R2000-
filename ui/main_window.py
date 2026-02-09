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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UI_CONFIG, DATA_CONFIG, LABEL_BACKTEST_CONFIG, MARKET_REGIME_CONFIG
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
        
        self._init_ui()
        self._connect_signals()
    
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
        self.analysis_panel.update_trade_log([])
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

                # 仅在交易数量变化时刷新明细 + 市场状态
                if self.rt_backtester is not None and len(self.rt_backtester.trades) != self.rt_last_trade_count:
                    new_count = len(self.rt_backtester.trades)
                    # 为新产生的交易分类市场状态
                    if self.regime_classifier is not None:
                        for ti in range(self.rt_last_trade_count, new_count):
                            trade = self.rt_backtester.trades[ti]
                            regime = self.regime_classifier.classify_at(trade.entry_idx)
                            trade.market_regime = regime
                            self.regime_map[ti] = regime
                    self.rt_last_trade_count = new_count
                    self.analysis_panel.update_trade_log(self._format_trades(self.rt_backtester.trades))
                    # 更新市场状态统计
                    self._update_regime_stats()

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

        # 为追赶期间产生的所有交易分类市场状态
        if self.regime_classifier is not None and self.rt_backtester:
            for ti, trade in enumerate(self.rt_backtester.trades):
                regime = self.regime_classifier.classify_at(trade.entry_idx)
                trade.market_regime = regime
                self.regime_map[ti] = regime

        if self.rt_backtester:
            self.analysis_panel.update_trade_log(self._format_trades(self.rt_backtester.trades))
        self._update_regime_stats()

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
            })
        return rows
    
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

                # 最终市场状态分类
                if self.labeler and self.labeler.alternating_swings:
                    classifier = MarketRegimeClassifier(
                        self.labeler.alternating_swings, MARKET_REGIME_CONFIG
                    )
                    self.regime_classifier = classifier
                    self.regime_map = {}
                    for ti, trade in enumerate(bt_result.trades):
                        regime = classifier.classify_at(trade.entry_idx)
                        trade.market_regime = regime
                        self.regime_map[ti] = regime
                    # 保存回测器引用以便统计
                    self.rt_backtester = backtester
                    self._update_regime_stats()
                    self.analysis_panel.update_trade_log(self._format_trades(bt_result.trades))
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
