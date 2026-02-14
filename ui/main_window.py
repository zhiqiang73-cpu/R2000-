"""
R3000 主窗口
PyQt6 主窗口：深色主题、动态 K 线播放、标注可视化
"""
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
import pandas as pd
import json
import re
from typing import Optional
import sys
import os
import time
import traceback
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.paper_trader import load_trade_history_from_file, save_trade_history_to_file
from config import (UI_CONFIG, DATA_CONFIG, LABEL_BACKTEST_CONFIG,
                    MARKET_REGIME_CONFIG, VECTOR_SPACE_CONFIG,
                    TRAJECTORY_CONFIG, WALK_FORWARD_CONFIG, MEMORY_CONFIG,
                    PAPER_TRADING_CONFIG)
from ui.chart_widget import ChartWidget
from ui.control_panel import ControlPanel
from ui.analysis_panel import AnalysisPanel
from ui.optimizer_panel import OptimizerPanel
from ui.paper_trading_tab import PaperTradingTab


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


class QuickLabelWorker(QtCore.QObject):
    """仅标注工作者 - 在后台计算标注与回测，避免UI卡死"""
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, df, params):
        super().__init__()
        self.df = df
        self.params = params

    @QtCore.pyqtSlot()
    def process(self):
        try:
            from core.labeler import GodViewLabeler
            from core.backtester import Backtester
            from core.market_regime import MarketRegimeClassifier
            from core.feature_vector import FeatureVectorEngine
            from core.vector_memory import VectorMemory
            from utils.indicators import calculate_all_indicators

            self.progress.emit("正在计算指标...")
            df = calculate_all_indicators(self.df.copy())

            self.progress.emit("正在执行上帝视角标注...")
            labeler = GodViewLabeler(
                swing_window=self.params.get('swing_window')
            )
            labels = labeler.label(df, use_dp_optimization=False)

            self.progress.emit("正在进行回测统计...")
            bt_cfg = LABEL_BACKTEST_CONFIG
            backtester = Backtester(
                initial_capital=bt_cfg["INITIAL_CAPITAL"],
                leverage=bt_cfg["LEVERAGE"],
                fee_rate=bt_cfg["FEE_RATE"],
                slippage=bt_cfg["SLIPPAGE"],
                position_size_pct=bt_cfg["POSITION_SIZE_PCT"],
            )
            bt_result = backtester.run_with_labels(df, labels)

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

            regime_classifier = None
            regime_map = {}
            fv_engine = None
            vector_memory = None

            if labeler.alternating_swings:
                self.progress.emit("正在生成市场状态与向量空间...")
                classifier = MarketRegimeClassifier(
                    labeler.alternating_swings, MARKET_REGIME_CONFIG
                )
                regime_classifier = classifier

                fv_engine = FeatureVectorEngine()
                fv_engine.precompute(df)
                vector_memory = VectorMemory(
                    k_neighbors=VECTOR_SPACE_CONFIG["K_NEIGHBORS"],
                    min_points=VECTOR_SPACE_CONFIG["MIN_CLOUD_POINTS"],
                )

                for ti, trade in enumerate(bt_result.trades):
                    regime = classifier.classify_at(trade.entry_idx)
                    trade.market_regime = regime
                    regime_map[ti] = regime

                    regime_name = regime or '未知'
                    direction = "LONG" if trade.side == 1 else "SHORT"

                    entry_abc = fv_engine.get_abc(trade.entry_idx)
                    trade.entry_abc = entry_abc
                    vector_memory.add_point(regime_name, direction, "ENTRY", *entry_abc)

                    exit_abc = fv_engine.get_abc(trade.exit_idx)
                    trade.exit_abc = exit_abc
                    vector_memory.add_point(regime_name, direction, "EXIT", *exit_abc)

            self.finished.emit({
                "df": df,
                "labels": labels,
                "labeler": labeler,
                "backtester": backtester,
                "bt_result": bt_result,
                "metrics": metrics,
                "regime_classifier": regime_classifier,
                "regime_map": regime_map,
                "fv_engine": fv_engine,
                "vector_memory": vector_memory
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
    # Walk-Forward 信号
    # 批量 Walk-Forward 信号
    _batch_wf_progress_signal = QtCore.pyqtSignal(int, int, dict)  # round_idx, n_rounds, cumulative_stats
    _batch_wf_done_signal = QtCore.pyqtSignal(object)  # BatchWalkForwardResult
    
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
        
        # 原型库（聚类后的交易模式）
        self._prototype_library = None

        # Walk-Forward 结果（用于模板评估）
        self._last_wf_result = None
        self._last_eval_result = None
        
        # 批量 Walk-Forward
        self._batch_wf_engine = None
        self._batch_wf_running = False
        self._last_verified_prototype_fps = set()  # 批量WF后可用原型集合
        
        # 模拟交易相关
        self._live_engine = None
        self._live_running = False
        self._live_chart_timer = QtCore.QTimer(self)
        refresh_ms = int(PAPER_TRADING_CONFIG.get("REALTIME_UI_REFRESH_MS", 1000))
        self._live_chart_timer.setInterval(max(50, refresh_ms))  # UI刷新频率
        self._live_chart_timer.timeout.connect(self._on_live_chart_tick)

        # GA 完成信号（analysis_panel 在后续 _init_ui 中创建后再连接按钮）
        self._ga_done_signal.connect(self._on_ga_finished)
        # Walk-Forward 信号
        # 批量WF信号
        self._batch_wf_progress_signal.connect(self._on_batch_wf_progress)
        self._batch_wf_done_signal.connect(self._on_batch_wf_finished)
        
        self._init_ui()
        self._connect_signals()
        self._load_saved_paper_api_config()

        # 自动加载已有记忆（如果配置了）
        self._auto_load_memory()
        
        # 自动加载已有原型库（如果配置了）
        self._auto_load_prototypes()
        
        # 自动加载历史交易记录（程序启动即显示）
        self._load_paper_trade_history_on_start()
    
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
        
        # 中央组件 - 顶层Tab切换
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 顶层布局
        top_layout = QtWidgets.QVBoxLayout(central_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        
        # 创建顶层Tab
        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
            }}
            QTabBar::tab {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                padding: 12px 30px;
                margin-right: 2px;
                font-size: 14px;
                font-weight: bold;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }}
            QTabBar::tab:selected {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #3a3a3a;
            }}
        """)
        top_layout.addWidget(self.main_tabs)
        
        # ============ Tab 1: 上帝视角训练 ============
        training_tab = QtWidgets.QWidget()
        training_layout = QtWidgets.QHBoxLayout(training_tab)
        training_layout.setContentsMargins(5, 5, 5, 5)
        training_layout.setSpacing(5)
        
        # 左侧控制面板
        self.control_panel = ControlPanel()
        training_layout.addWidget(self.control_panel)
        
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
        
        training_layout.addWidget(center_widget, stretch=1)
        
        # 右侧分析面板
        self.analysis_panel = AnalysisPanel()
        training_layout.addWidget(self.analysis_panel)

        # 把“优化参数 + 记忆管理”移动到左下角（用户指定）
        try:
            bottom_tools = self.analysis_panel.trajectory_widget.extract_bottom_tools_widget()
            self.control_panel.add_bottom_widget(bottom_tools)
        except Exception as e:
            print(f"[UI] 移动优化/记忆区域失败: {e}")
        
        self.main_tabs.addTab(training_tab, "📊 上帝视角训练")
        
        # ============ Tab 2: 模拟交易 ============
        self.paper_trading_tab = PaperTradingTab()
        self.main_tabs.addTab(self.paper_trading_tab, "💹 模拟交易")
        
        # 连接删除交易记录信号
        self.paper_trading_tab.trade_log.delete_trade_signal.connect(self._on_trade_delete_requested)
        
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
        self.control_panel.quick_label_requested.connect(self._on_quick_label_requested)
        # analyze_requested 和 optimize_requested 信号已从UI移除，保留信号定义以供后端使用
        # 不再连接到前端按钮
        self.control_panel.pause_requested.connect(self._on_pause_requested)
        self.control_panel.stop_requested.connect(self._on_stop_requested)
        self.control_panel.speed_changed.connect(self._on_speed_changed)

        # 轨迹匹配相关
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
        self.analysis_panel.trajectory_widget.apply_template_filter_requested.connect(
            self._on_apply_template_filter
        )
        # 批量 Walk-Forward
        self.analysis_panel.trajectory_widget.batch_wf_requested.connect(
            self._on_batch_wf_requested
        )
        self.analysis_panel.trajectory_widget.batch_wf_stop_requested.connect(
            self._on_batch_wf_stop
        )
        
        # 原型库信号
        self.analysis_panel.trajectory_widget.generate_prototypes_requested.connect(
            self._on_generate_prototypes
        )
        self.analysis_panel.trajectory_widget.load_prototypes_requested.connect(
            self._on_load_prototypes
        )
        
        # 模拟交易信号
        self.paper_trading_tab.control_panel.start_requested.connect(
            self._on_paper_trading_start
        )
        self.paper_trading_tab.control_panel.stop_requested.connect(
            self._on_paper_trading_stop
        )
        self.paper_trading_tab.control_panel.test_connection_requested.connect(
            self._on_paper_trading_test_connection
        )
        self.paper_trading_tab.control_panel.save_api_requested.connect(
            self._on_paper_api_save_requested
        )
        self.paper_trading_tab.status_panel.save_profitable_requested.connect(
            self._on_save_profitable_templates
        )
        self.paper_trading_tab.status_panel.delete_losing_requested.connect(
            self._on_delete_losing_templates
        )

    def _infer_source_meta(self) -> tuple:
        """从数据文件名推断来源交易对与时间框架（如 btcusdt_1m.parquet）"""
        data_file = ""
        if hasattr(self, "data_loader") and self.data_loader is not None:
            data_file = getattr(self.data_loader, "data_file", "") or ""
        if not data_file:
            data_file = DATA_CONFIG.get("DATA_FILE", "")
        base = os.path.basename(str(data_file)).lower()
        m = re.search(r"([a-z0-9]+)_(\d+[mhd])", base)
        if not m:
            return "", ""
        symbol = m.group(1).upper()
        interval = m.group(2)
        return symbol, interval
    
    def _on_load_data(self):
        """加载数据"""
        self._on_sample_requested(DATA_CONFIG["SAMPLE_SIZE"], None)
    
    def _on_sample_requested(self, sample_size: int, seed):
        """处理采样请求"""
        self._sampling_in_progress = True
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
        self.worker_thread.finished.connect(self._on_sample_thread_finished)
        
        self.worker_thread.start()
    
    def _on_sample_finished(self, result):
        """采样完成"""
        try:
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
            self._sampling_in_progress = False
        except Exception as e:
            self._on_worker_error(str(e) + "\n" + traceback.format_exc())
    
    def _on_sample_thread_finished(self):
        """采样线程结束兜底处理，避免 UI 卡在加载态"""
        if getattr(self, "_sampling_in_progress", False):
            self._sampling_in_progress = False
            self.control_panel.set_buttons_enabled(True)
            self.control_panel.set_status("数据加载中断，请重试")
            self.statusBar().showMessage("数据加载中断：未收到完成回调")

    def _on_worker_error(self, error_msg: str):
        """通用后台任务错误处理"""
        self._sampling_in_progress = False
        self.control_panel.set_buttons_enabled(True)
        self.control_panel.set_status(f"错误: {error_msg}")
        self.statusBar().showMessage(f"任务出错: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "错误", f"后台任务出错:\n{error_msg}")
    

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

    def _on_quick_label_requested(self, params: dict):
        """仅标注模式 - 快速计算标注，不播放动画，完成后可直接运行Walk-Forward"""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载数据")
            return

        if self.is_playing:
            QtWidgets.QMessageBox.warning(self, "警告", "正在播放中，请先停止")
            return

        # 禁用按钮，显示进度
        self.control_panel.set_buttons_enabled(False)
        self.control_panel.set_status("正在快速标注...")
        self.statusBar().showMessage("正在计算标注（仅标注模式）...")

        # 重置状态
        self._labels_ready = False
        self.rt_last_idx = -1
        self.rt_backtester = None
        self.rt_last_trade_count = 0
        self.regime_classifier = None
        self.regime_map = {}
        self.fv_engine = None
        self.vector_memory = None
        self._fv_ready = False

        self.quick_label_thread = QtCore.QThread()
        self.quick_label_worker = QuickLabelWorker(self.df, params)
        self.quick_label_worker.moveToThread(self.quick_label_thread)

        self.quick_label_thread.started.connect(self.quick_label_worker.process)
        self.quick_label_worker.progress.connect(self._on_quick_label_progress, QtCore.Qt.ConnectionType.QueuedConnection)
        self.quick_label_worker.finished.connect(self._on_quick_label_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        self.quick_label_worker.error.connect(self._on_quick_label_error, QtCore.Qt.ConnectionType.QueuedConnection)
        self.quick_label_worker.finished.connect(self.quick_label_thread.quit)
        self.quick_label_worker.error.connect(self.quick_label_thread.quit)

        self.quick_label_thread.start()

    def _on_quick_label_progress(self, msg: str):
        """快速标注进度更新"""
        self.control_panel.set_status(msg)
        self.statusBar().showMessage(msg)

    def _on_quick_label_error(self, msg: str):
        """快速标注失败"""
        QtWidgets.QMessageBox.critical(self, "标注失败", msg)
        self.control_panel.set_buttons_enabled(True)

    def _on_quick_label_finished(self, result: dict):
        """快速标注完成"""
        self.df = result["df"]
        self.labels = result["labels"]
        self.labeler = result["labeler"]
        self._labels_ready = True

        # 显示全部数据和标注
        self.chart_widget.set_data(self.df, self.labels, show_all=True)

        # 统计
        long_count = int((self.labels == 1).sum())
        short_count = int((self.labels == -1).sum())
        stats = self.labeler.get_statistics() if self.labeler else {}

        status_text = f"快速标注完成: {long_count} LONG + {short_count} SHORT"
        if stats:
            status_text += f" | 平均收益: {stats.get('avg_profit_pct', 0):.2f}%"

        self.control_panel.set_status(status_text)
        self.statusBar().showMessage(status_text)

        # 回测指标
        bt_result = result.get("bt_result")
        metrics = result.get("metrics", {})
        self.optimizer_panel.update_backtest_metrics(metrics)

        # 市场状态分类 / 向量空间
        self.regime_classifier = result.get("regime_classifier")
        self.regime_map = result.get("regime_map", {})
        self.fv_engine = result.get("fv_engine")
        self.vector_memory = result.get("vector_memory")
        self._fv_ready = self.fv_engine is not None

        if bt_result is not None:
            self.rt_backtester = result.get("backtester")
            self._update_regime_stats()
            self._update_vector_space_plot()
            self.analysis_panel.update_trade_log(self._format_trades(bt_result.trades))

            # 轨迹模板提取
            self._extract_trajectory_templates(bt_result.trades)

        # 启用批量验证
        self.analysis_panel.enable_batch_wf(True)

        if bt_result:
            msg = (
                f"标注完成！共 {bt_result.total_trades} 笔交易\n"
                f"胜率: {bt_result.win_rate:.1%}\n"
                f"总收益: {bt_result.total_return_pct:.2f}%\n\n"
                f"现在可以运行 Walk-Forward 验证了"
            )
        else:
            msg = "标注完成！\n\n现在可以运行 Walk-Forward 验证了"
        QtWidgets.QMessageBox.information(self, "快速标注完成", msg)
        self.control_panel.set_buttons_enabled(True)
    
    def _on_labeling_step(self, idx: int):
        """标注步骤完成"""
        try:
            # 前进一根 K 线
            self.chart_widget.advance_one_candle()
            
            # 更新进度
            total = len(self.df) if self.df is not None else 0
            self.control_panel.update_play_progress(idx + 1, total)
        except Exception as e:
            self._on_worker_error(str(e) + "\n" + traceback.format_exc())
            if self.labeling_worker:
                self.labeling_worker.stop()
            self.is_playing = False
            self.control_panel.set_playing_state(False)
            return

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

                # 仅在交易数量变化时刷新明细 + 市场状态 + 向量 + 实时指纹
                if self.rt_backtester is not None and len(self.rt_backtester.trades) != self.rt_last_trade_count:
                    new_count = len(self.rt_backtester.trades)
                    templates_added = 0
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
                        # 实时提取轨迹模板（盈利交易）
                        if self._extract_single_trade_template(trade, ti):
                            templates_added += 1
                    self.rt_last_trade_count = new_count
                    self.analysis_panel.update_trade_log(self._format_trades(self.rt_backtester.trades))
                    self._update_regime_stats()
                    # 每10笔交易刷新一次3D图（节省性能）
                    if new_count % 10 == 0 or new_count < 20:
                        self._update_vector_space_plot()
                    # 实时更新指纹图（有新模板时或每10笔检查一次）
                    if templates_added > 0 or (new_count % 10 == 0):
                        self._update_fingerprint_view()
                        self._update_memory_stats()

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

        # 仅做轻量初始化：重计算（FV precompute）延后到标注完成阶段，避免“开始标记”卡UI
        if self.df is not None:
            try:
                from core.trajectory_engine import TrajectoryMemory
                if self.trajectory_memory is None:
                    src_symbol, src_interval = self._infer_source_meta()
                    self.trajectory_memory = TrajectoryMemory(
                        source_symbol=src_symbol,
                        source_interval=src_interval,
                    )
                    print("[TrajectoryMemory] 轨迹记忆体就绪（实时积累模式）")
            except Exception as e:
                print(f"[TrajectoryMemory] 初始化失败: {e}")

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

        # 为追赶期间产生的所有交易分类市场状态 + 填充向量记忆体 + 提取轨迹模板
        templates_added = 0
        if self.rt_backtester:
            for ti, trade in enumerate(self.rt_backtester.trades):
                if self.regime_classifier is not None:
                    regime = self.regime_classifier.classify_at(trade.entry_idx)
                    trade.market_regime = regime
                    self.regime_map[ti] = regime
                # 填充向量坐标和记忆体
                if self._fv_ready and self.fv_engine:
                    self._record_trade_vectors(trade)
                # 实时提取轨迹模板（盈利交易）
                if self._extract_single_trade_template(trade, ti):
                    templates_added += 1

        if self.rt_backtester:
            self.analysis_panel.update_trade_log(self._format_trades(self.rt_backtester.trades))
        self._update_regime_stats()
        self._update_vector_space_plot()
        
        # 更新指纹图（追赶期间提取的模板）
        if templates_added > 0:
            self._update_fingerprint_view()
            self._update_memory_stats()
            print(f"[TrajectoryMemory] 追赶阶段提取: {templates_added} 个模板")

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
            # 指纹摘要：模板ID + 相似度
            template_idx = getattr(t, 'matched_template_idx', None)
            entry_sim = getattr(t, 'entry_similarity', None)
            if template_idx is not None and entry_sim is not None:
                fingerprint = f"T#{template_idx} | Sim={entry_sim:.2f}"
            else:
                fingerprint = "--"
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
                "fingerprint": fingerprint,
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

    def _extract_single_trade_template(self, trade, trade_idx: int) -> bool:
        """
        实时提取单笔交易的轨迹模板
        
        Args:
            trade: TradeRecord 交易记录
            trade_idx: 交易在列表中的索引
            
        Returns:
            True 如果成功提取并添加模板，False 否则
        """
        if not self._fv_ready or self.fv_engine is None:
            return False
        
        if self.trajectory_memory is None:
            return False
        
        # 检查是否盈利交易
        min_profit = TRAJECTORY_CONFIG.get("MIN_PROFIT_PCT", 0.5)
        if trade.profit_pct < min_profit:
            return False
        
        # 检查入场前是否有足够K线
        pre_entry_window = TRAJECTORY_CONFIG.get("PRE_ENTRY_WINDOW", 60)
        if trade.entry_idx < pre_entry_window:
            return False
        
        try:
            from core.trajectory_engine import TrajectoryTemplate
            
            regime = self.regime_map.get(trade_idx, getattr(trade, 'market_regime', '未知'))
            direction = "LONG" if trade.side == 1 else "SHORT"
            
            # 提取三段轨迹
            pre_entry = self.fv_engine.get_raw_matrix(
                trade.entry_idx - pre_entry_window,
                trade.entry_idx
            )
            
            holding = self.fv_engine.get_raw_matrix(
                trade.entry_idx,
                trade.exit_idx + 1
            )
            
            # 离场前轨迹
            pre_exit_window = TRAJECTORY_CONFIG.get("PRE_EXIT_WINDOW", 30)
            exit_start = max(trade.entry_idx, trade.exit_idx - pre_exit_window + 1)
            pre_exit = self.fv_engine.get_raw_matrix(exit_start, trade.exit_idx + 1)
            
            template = TrajectoryTemplate(
                trade_idx=trade_idx,
                regime=regime,
                direction=direction,
                profit_pct=trade.profit_pct,
                pre_entry=pre_entry,
                holding=holding,
                pre_exit=pre_exit,
                entry_idx=trade.entry_idx,
                exit_idx=trade.exit_idx,
            )
            
            # 添加到记忆体
            self.trajectory_memory._add_template(regime, direction, template)
            return True
            
        except Exception as e:
            print(f"[TrajectoryMemory] 单笔模板提取失败: {e}")
            return False

    def _extract_trajectory_templates(self, trades):
        """提取轨迹模板"""
        if not self._fv_ready or self.fv_engine is None:
            return

        try:
            from core.trajectory_engine import TrajectoryMemory

            # 检查是否已有记忆体，如果有则合并，否则新建
            if hasattr(self, 'trajectory_memory') and self.trajectory_memory is not None:
                # 提取新模板到临时记忆体
                src_symbol, src_interval = self._infer_source_meta()
                new_memory = TrajectoryMemory(
                    source_symbol=src_symbol,
                    source_interval=src_interval,
                )
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
                src_symbol, src_interval = self._infer_source_meta()
                self.trajectory_memory = TrajectoryMemory(
                    source_symbol=src_symbol,
                    source_interval=src_interval,
                )
                n_templates = self.trajectory_memory.extract_from_trades(
                    trades, self.fv_engine, self.regime_map
                )

            if n_templates > 0:
                # 更新 UI 统计
                self._update_trajectory_ui()
                self._update_memory_stats()

                # 启用批量验证按钮
                self.analysis_panel.enable_batch_wf(True)

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

    # ══════════════════════════════════════════════════════════════════════════
    # 模板评估与筛选
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate_templates_from_wf(self):
        """从 Walk-Forward 结果评估模板"""
        if self._last_wf_result is None:
            return

        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            print("[TemplateEvaluator] 无记忆体可评估")
            return

        try:
            from core.walk_forward import evaluate_templates_from_wf_result
            from config import WALK_FORWARD_CONFIG

            # 获取评估参数
            min_matches = WALK_FORWARD_CONFIG.get("EVAL_MIN_MATCHES", 3)
            min_win_rate = WALK_FORWARD_CONFIG.get("EVAL_MIN_WIN_RATE", 0.6)

            # 评估模板
            eval_result = evaluate_templates_from_wf_result(
                self._last_wf_result,
                self.trajectory_memory,
                min_matches=min_matches,
                min_win_rate=min_win_rate
            )

            # 保存评估结果（内存）
            self._last_eval_result = eval_result

            # 更新UI
            self.analysis_panel.update_template_evaluation(eval_result)

            # 打印摘要
            eval_result.print_summary()

            print(f"[TemplateEvaluator] 评估完成: "
                  f"优质{eval_result.excellent_count}, "
                  f"合格{eval_result.qualified_count}, "
                  f"待观察{eval_result.pending_count}, "
                  f"淘汰{eval_result.eliminated_count}")
            
            # 自动保存评估结果到磁盘（新增）
            self._save_evaluation_result(eval_result)

        except Exception as e:
            import traceback
            print(f"[TemplateEvaluator] 评估失败: {e}")
            traceback.print_exc()

    def _save_evaluation_result(self, eval_result):
        """
        保存评估结果到磁盘
        
        Args:
            eval_result: EvaluationResult 实例
        """
        try:
            import pickle
            from datetime import datetime
            import os
            
            # 确保目录存在
            eval_dir = "data/evaluation"
            os.makedirs(eval_dir, exist_ok=True)
            
            # 生成文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(eval_dir, f"eval_{timestamp}.pkl")
            
            # 保存对象（包含完整的评估结果）
            with open(filepath, 'wb') as f:
                pickle.dump(eval_result, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 同时保存一个"最新"的副本（方便程序启动时加载）
            latest_filepath = os.path.join(eval_dir, "eval_latest.pkl")
            with open(latest_filepath, 'wb') as f:
                pickle.dump(eval_result, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"[TemplateEvaluator] 评估结果已保存: {filepath} ({file_size:.2f} KB)")
            
        except Exception as e:
            print(f"[TemplateEvaluator] 保存评估结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_latest_evaluation_result(self):
        """
        尝试加载最新的评估结果
        
        Returns:
            EvaluationResult 或 None
        """
        try:
            import pickle
            filepath = "data/evaluation/eval_latest.pkl"
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'rb') as f:
                eval_result = pickle.load(f)
            
            print(f"[TemplateEvaluator] 已加载上次评估结果: "
                  f"优质{eval_result.excellent_count}, "
                  f"合格{eval_result.qualified_count}, "
                  f"待观察{eval_result.pending_count}, "
                  f"淘汰{eval_result.eliminated_count}")
            
            return eval_result
            
        except Exception as e:
            print(f"[TemplateEvaluator] 加载评估结果失败: {e}")
            return None

    def _on_apply_template_filter(self):
        """应用模板筛选（删除淘汰的模板）"""
        if self._last_eval_result is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先运行批量 Walk-Forward 验证")
            return

        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            QtWidgets.QMessageBox.warning(self, "警告", "无记忆体可筛选")
            return

        n_eliminated = self._last_eval_result.eliminated_count
        n_remove_fps = len(self._last_eval_result.remove_fingerprints)
        if n_eliminated == 0 or n_remove_fps == 0:
            QtWidgets.QMessageBox.information(self, "提示", "没有需要淘汰的模板")
            return

        # 计算当前记忆库中有多少新增模板（未被评估过的）
        current_total = self.trajectory_memory.total_count
        evaluated_total = self._last_eval_result.total_templates
        new_since_eval = max(0, current_total - evaluated_total)

        # 确认对话框
        msg = (
            f"将删除 {n_remove_fps} 个评级为'淘汰'的模板。\n"
            f"保留 {len(self._last_eval_result.keep_fingerprints)} 个已验证模板（优质/合格/待观察）。\n"
        )
        if new_since_eval > 0:
            msg += f"另有 {new_since_eval} 个新增模板（未被评估）将保留不动。\n"
        msg += "\n确定执行筛选吗？"

        reply = QtWidgets.QMessageBox.question(
            self, "确认筛选", msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            # 执行筛选 — 用 remove 而非 filter，保护新增模板
            old_count = self.trajectory_memory.total_count
            removed = self.trajectory_memory.remove_by_fingerprints(
                self._last_eval_result.remove_fingerprints,
                verbose=True
            )
            new_count = self.trajectory_memory.total_count

            # ── 自动保存筛选后的记忆库 ──
            save_path = self.trajectory_memory.save(verbose=True)
            print(f"[筛选] 已自动保存筛选后记忆库: {save_path}")

            # 更新UI
            self._update_memory_stats()
            self._update_trajectory_ui()

            # 更新评估结果以反映筛选后状态（不清空，而是更新）
            # 保留评估结果，只更新已验证数量
            self.analysis_panel.update_template_evaluation(self._last_eval_result)

            # 更新指纹图
            self._update_fingerprint_view()

            QtWidgets.QMessageBox.information(
                self, "筛选完成",
                f"已删除 {old_count - new_count} 个淘汰模板\n"
                f"保留 {new_count} 个模板（已验证 + 新增未评估）\n"
                f"已自动保存到: {save_path}\n\n"
                "提示: 新增未评估的模板不受影响，可继续批量验证。"
            )

            self.statusBar().showMessage(
                f"模板筛选完成: 删除{old_count - new_count}个, 保留{new_count}个, 已自动保存"
            )

        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "筛选失败", str(e))
            traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    # 批量 Walk-Forward 验证
    # ══════════════════════════════════════════════════════════════════════════

    def _on_batch_wf_requested(self):
        """批量 Walk-Forward 验证请求"""
        # 检查是否有原型库（优先使用）或模板库
        has_prototypes = hasattr(self, '_prototype_library') and self._prototype_library is not None
        has_templates = hasattr(self, 'trajectory_memory') and self.trajectory_memory is not None
        
        if has_prototypes:
            proto_count = self._prototype_library.total_count
            use_prototypes = True
            source_desc = f"原型库: {proto_count} 个原型（LONG={len(self._prototype_library.long_prototypes)}, SHORT={len(self._prototype_library.short_prototypes)}）"
            speed_desc = "每轮预计 5-15 秒"
        elif has_templates and self.trajectory_memory.total_count > 0:
            use_prototypes = False
            source_desc = f"模板库: {self.trajectory_memory.total_count} 个模板"
            speed_desc = "每轮预计 30-60 秒"
        else:
            QtWidgets.QMessageBox.warning(
                self, "警告",
                "请先生成原型库（推荐）或加载模板库"
            )
            return

        if self._batch_wf_running:
            QtWidgets.QMessageBox.information(self, "提示", "批量验证已在运行中")
            return

        # 获取参数
        n_rounds = self.analysis_panel.trajectory_widget.batch_rounds_spin.value()
        sample_size = self.analysis_panel.trajectory_widget.batch_sample_spin.value()

        # 确认对话框
        mode_str = "【原型模式 - 快速】" if use_prototypes else "【模板模式】"
        reply = QtWidgets.QMessageBox.question(
            self, f"确认批量验证 {mode_str}",
            f"将启动批量 Walk-Forward 验证:\n\n"
            f"  {source_desc}\n"
            f"  验证轮数: {n_rounds} 轮\n"
            f"  每轮采样: {sample_size:,} 根K线\n"
            f"  贝叶斯优化: 20 trials/轮\n\n"
            f"{speed_desc}。\n"
            f"继续吗？",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # UI 更新
        self._batch_wf_running = True
        self.analysis_panel.on_batch_wf_started()
        mode_label = "原型" if use_prototypes else "模板"
        self.statusBar().showMessage(f"批量Walk-Forward启动（{mode_label}模式）: {n_rounds}轮...")

        # 在后台线程运行
        import threading
        
        # 保存当前使用的库引用
        prototype_lib = self._prototype_library if use_prototypes else None
        memory_lib = self.trajectory_memory if not use_prototypes else None

        def _run_batch_wf():
            try:
                from core.batch_walk_forward import BatchWalkForwardEngine
                from core.data_loader import DataLoader

                # 创建数据加载器
                data_loader = DataLoader()
                data_loader.load_full_data()

                # 创建引擎（原型模式或模板模式）
                self._batch_wf_engine = BatchWalkForwardEngine(
                    data_loader=data_loader,
                    global_memory=memory_lib,
                    n_rounds=n_rounds,
                    sample_size=sample_size,
                    n_trials=20,  # 每轮20次贝叶斯优化（快速）
                    round_workers=WALK_FORWARD_CONFIG.get("BATCH_ROUND_WORKERS", 1),
                    prototype_library=prototype_lib,  # 原型库（如有）
                )

                # 进度回调（通过信号传到主线程）
                def progress_callback(round_idx, n_rounds, round_result, cumulative_stats):
                    self._batch_wf_progress_signal.emit(
                        round_idx, n_rounds, cumulative_stats
                    )

                # 运行
                result = self._batch_wf_engine.run(callback=progress_callback)

                # 完成
                self._batch_wf_done_signal.emit(result)

            except Exception as e:
                import traceback
                print(f"[BatchWF] 批量验证失败: {e}")
                traceback.print_exc()
                self._batch_wf_done_signal.emit(None)

        thread = threading.Thread(target=_run_batch_wf, daemon=True)
        thread.start()

    def _on_batch_wf_stop(self):
        """停止批量WF"""
        if self._batch_wf_engine is not None:
            self._batch_wf_engine.stop()
            self.statusBar().showMessage("正在停止批量验证...")

    def _on_batch_wf_progress(self, round_idx: int, n_rounds: int, cumulative_stats: dict):
        """批量WF进度更新（主线程槽函数）"""
        # 更新UI进度
        self.analysis_panel.update_batch_wf_progress(
            round_idx, n_rounds, cumulative_stats
        )

        # 同步更新顶部指纹模板库的已验证数量
        verified_long = cumulative_stats.get("verified_long", 0)
        verified_short = cumulative_stats.get("verified_short", 0)
        self.analysis_panel.trajectory_widget.verified_long_count.setText(str(verified_long))
        self.analysis_panel.trajectory_widget.verified_short_count.setText(str(verified_short))

        # 更新评级数字
        excellent = cumulative_stats.get("excellent", 0)
        qualified = cumulative_stats.get("qualified", 0)
        pending = cumulative_stats.get("pending", 0)
        eliminated = cumulative_stats.get("eliminated", 0)
        self.analysis_panel.trajectory_widget.eval_excellent_label.setText(str(excellent))
        self.analysis_panel.trajectory_widget.eval_qualified_label.setText(str(qualified))
        self.analysis_panel.trajectory_widget.eval_pending_label.setText(str(pending))
        self.analysis_panel.trajectory_widget.eval_eliminated_label.setText(str(eliminated))

        # 区分运行中和完成状态
        is_running = cumulative_stats.get("running", False)
        progress_pct = cumulative_stats.get("global_progress_pct", None)
        if is_running:
            phase = cumulative_stats.get("phase", "")
            pct_text = f" | {int(progress_pct)}%" if progress_pct is not None else ""
            if phase == "build_cache":
                i_idx = cumulative_stats.get("trial_idx", 0)
                n_total = cumulative_stats.get("trial_total", 1)
                self.statusBar().showMessage(
                    f"批量WF: 第 {round_idx + 1}/{n_rounds} 轮 | 预构建匹配缓存 ({i_idx}/{n_total}){pct_text} ..."
                )
            elif phase == "bayes_opt":
                trial_idx = cumulative_stats.get("trial_idx", 0)
                trial_total = cumulative_stats.get("trial_total", 20)
                self.statusBar().showMessage(
                    f"批量WF: 第 {round_idx + 1}/{n_rounds} 轮 | 贝叶斯优化 ({trial_idx}/{trial_total}){pct_text} ..."
                )
            else:
                self.statusBar().showMessage(
                    f"批量WF: 第 {round_idx + 1}/{n_rounds} 轮运行中... {pct_text}"
                )
        else:
            self.statusBar().showMessage(
                f"批量WF: Round {round_idx + 1}/{n_rounds} 完成 | "
                f"匹配={cumulative_stats.get('total_match_events', 0)} | "
                f"已验证: L={verified_long} S={verified_short}"
            )

    def _on_batch_wf_finished(self, result):
        """批量WF完成（主线程槽函数）"""
        self._batch_wf_running = False
        self.analysis_panel.on_batch_wf_finished()

        if result is None:
            self.statusBar().showMessage("批量Walk-Forward 失败")
            QtWidgets.QMessageBox.critical(self, "错误", "批量验证运行失败，请查看控制台日志")
            return

        # 获取最终评估结果
        from config import WALK_FORWARD_CONFIG
        wf_counts = None
        if self._batch_wf_engine is not None:
            # 原型模式：将验证结果回写到原型库
            if getattr(self._batch_wf_engine, "use_prototypes", False):
                self._last_verified_prototype_fps = self._batch_wf_engine.get_verified_prototype_fingerprints()
                
                # 回写验证状态到原型库
                if self._prototype_library is not None:
                    proto_stats = self._batch_wf_engine.get_prototype_stats()
                    min_matches = WALK_FORWARD_CONFIG.get("EVAL_MIN_MATCHES", 3)
                    min_win_rate = WALK_FORWARD_CONFIG.get("EVAL_MIN_WIN_RATE", 0.6)
                    wf_counts = self._prototype_library.apply_wf_verification(
                        proto_stats, min_matches, min_win_rate
                    )
                    
                    # 刷新原型表格（会显示验证标记）
                    self.analysis_panel.trajectory_widget.update_prototype_stats(
                        self._prototype_library
                    )
                    
                    # 自动保存带验证状态的原型库
                    try:
                        save_path = self._prototype_library.save(verbose=True)
                        print(f"[BatchWF] 已保存带验证标记的原型库: {save_path}")
                    except Exception as e:
                        print(f"[BatchWF] 原型库保存失败: {e}")

            eval_result = self._batch_wf_engine.get_evaluation_result()
            if eval_result is not None:
                self._last_eval_result = eval_result
                self.analysis_panel.update_template_evaluation(eval_result)
                # 自动保存评估结果
                self._save_evaluation_result(eval_result)

        # 显示完成信息
        elapsed_min = int(result.total_elapsed // 60)
        elapsed_sec = int(result.total_elapsed % 60)
        time_str = f"{elapsed_min}分{elapsed_sec}秒" if elapsed_min > 0 else f"{elapsed_sec}秒"

        # 构建验证摘要
        if wf_counts:
            verify_summary = (
                f"\n验证结果回写:\n"
                f"  合格: {wf_counts['qualified']}\n"
                f"  待观察: {wf_counts['pending']}\n"
                f"  淘汰: {wf_counts['eliminated']}\n"
                f"  保留: {wf_counts['total_verified']} / {result.unique_templates_matched}\n"
            )
        else:
            verify_summary = ""

        msg = (
            f"批量 Walk-Forward 验证完成!\n\n"
            f"完成轮数: {result.completed_rounds} / {result.n_rounds}\n"
            f"总耗时: {time_str}\n"
            f"累计匹配事件: {result.total_match_events}\n"
            f"涉及原型: {result.unique_templates_matched}\n"
            f"{verify_summary}\n"
            f"合格+待观察的原型已标记为\"已验证\"。"
        )

        self.statusBar().showMessage(
            f"批量WF完成: {result.completed_rounds}轮, "
            f"已验证 L={result.verified_long} S={result.verified_short}, "
            f"耗时{time_str}"
        )
        QtWidgets.QMessageBox.information(self, "批量验证完成", msg)


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

    def _auto_load_prototypes(self):
        """启动时自动加载已有原型库"""
        from config import PROTOTYPE_CONFIG
        
        if not PROTOTYPE_CONFIG.get("AUTO_LOAD_PROTOTYPE", True):
            return
        
        try:
            from core.template_clusterer import PrototypeLibrary
            
            library = PrototypeLibrary.load_latest(verbose=True)
            if library and library.total_count > 0:
                self._prototype_library = library
                self._last_verified_prototype_fps = set()
                self.analysis_panel.trajectory_widget.update_prototype_stats(library)
                self._update_trajectory_ui()
                print(f"[PrototypeLibrary] 自动加载: LONG={len(library.long_prototypes)}, "
                      f"SHORT={len(library.short_prototypes)}")
            else:
                print("[PrototypeLibrary] 启动: 无历史原型库文件")
        except Exception as e:
            print(f"[PrototypeLibrary] 自动加载失败: {e}")

    def _on_generate_prototypes(self, n_long: int, n_short: int):
        """生成原型库"""
        if not hasattr(self, 'trajectory_memory') or self.trajectory_memory is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载模板库")
            return
        
        if self.trajectory_memory.total_count == 0:
            QtWidgets.QMessageBox.warning(self, "警告", "模板库为空")
            return
        
        try:
            from core.template_clusterer import TemplateClusterer
            
            self.statusBar().showMessage(f"正在聚类... LONG={n_long}, SHORT={n_short}")
            QtWidgets.QApplication.processEvents()
            
            clusterer = TemplateClusterer(
                n_clusters_long=n_long,
                n_clusters_short=n_short,
            )
            
            library = clusterer.fit(self.trajectory_memory, verbose=True)

            # 绑定来源信息（交易对 + 时间框架）
            src_symbol = getattr(self.trajectory_memory, "source_symbol", "")
            src_interval = getattr(self.trajectory_memory, "source_interval", "")
            if not src_symbol or not src_interval:
                infer_symbol, infer_interval = self._infer_source_meta()
                src_symbol = src_symbol or infer_symbol
                src_interval = src_interval or infer_interval
            library.source_symbol = (src_symbol or "").upper()
            library.source_interval = (src_interval or "").strip()
            
            # 保存原型库
            save_path = library.save(verbose=True)
            
            self._prototype_library = library
            self._last_verified_prototype_fps = set()
            self.analysis_panel.trajectory_widget.update_prototype_stats(library)
            self._update_trajectory_ui()
            
            self.statusBar().showMessage(
                f"原型生成完成: LONG={len(library.long_prototypes)}, "
                f"SHORT={len(library.short_prototypes)}", 5000
            )
            
            QtWidgets.QMessageBox.information(
                self, "原型生成完成",
                f"已生成原型库:\n\n"
                f"  LONG 原型: {len(library.long_prototypes)}\n"
                f"  SHORT 原型: {len(library.short_prototypes)}\n"
                f"  来源模板: {library.source_template_count}\n\n"
                f"文件: {save_path}"
            )
            
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(
                self, "原型生成失败",
                f"错误: {e}\n\n{traceback.format_exc()}"
            )
            self.statusBar().showMessage("原型生成失败", 3000)

    def _on_load_prototypes(self):
        """加载最新原型库"""
        try:
            from core.template_clusterer import PrototypeLibrary
            
            library = PrototypeLibrary.load_latest(verbose=True)
            if library is None or library.total_count == 0:
                QtWidgets.QMessageBox.warning(self, "警告", "没有找到已保存的原型库")
                return
            
            self._prototype_library = library
            self._last_verified_prototype_fps = set()
            self.analysis_panel.trajectory_widget.update_prototype_stats(library)
            self._update_trajectory_ui()
            
            QtWidgets.QMessageBox.information(
                self, "加载成功",
                f"已加载原型库:\n\n"
                f"  LONG 原型: {len(library.long_prototypes)}\n"
                f"  SHORT 原型: {len(library.short_prototypes)}\n"
                f"  来源模板: {library.source_template_count}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(e))

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
            self.analysis_panel.enable_batch_wf(True)
            self.statusBar().showMessage(f"已加载 {memory.total_count} 个模板")
            
            # 尝试加载最新的评估结果（新增）
            self._last_eval_result = self._load_latest_evaluation_result()
            if self._last_eval_result:
                self.analysis_panel.update_template_evaluation(self._last_eval_result)

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
                    self.analysis_panel.enable_batch_wf(True)
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
        has_templates = (hasattr(self, 'trajectory_memory') and 
                         self.trajectory_memory is not None and
                         self.trajectory_memory.total_count > 0)
        has_prototypes = (hasattr(self, '_prototype_library') and 
                          self._prototype_library is not None and
                          self._prototype_library.total_count > 0)
        
        if not has_templates:
            self.analysis_panel.update_trajectory_template_stats(0, 0, 0, 0)
            self.analysis_panel.update_fingerprint_templates([])
            self.analysis_panel.trajectory_widget.enable_generate_prototypes(False)
        else:
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
            
            # 启用原型生成按钮（有模板时）
            self.analysis_panel.trajectory_widget.enable_generate_prototypes(True)
        
        # 启用批量验证按钮（有原型库 或 有模板库）
        self.analysis_panel.enable_batch_wf(has_prototypes or has_templates)

        # 同步模拟交易页可用聚合指纹图数量（避免显示0）
        try:
            if has_prototypes:
                verified = len(getattr(self, "_last_verified_prototype_fps", set()))
                active_count = verified if verified > 0 else self._prototype_library.total_count
                long_n = len(self._prototype_library.long_prototypes)
                short_n = len(self._prototype_library.short_prototypes)
                detail = f"LONG={long_n}, SHORT={short_n}" if verified == 0 else f"已验证={verified}"
                self.paper_trading_tab.control_panel.update_template_count(
                    active_count, mode="prototype", detail=detail
                )
            elif has_templates:
                self.paper_trading_tab.control_panel.update_template_count(
                    self.trajectory_memory.total_count, mode="template"
                )
            else:
                self.paper_trading_tab.control_panel.update_template_count(0, mode="prototype")
        except Exception as e:
            print(f"[UI] 同步可用聚合指纹图数量失败: {e}")

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
            "• 遗传算法优化：策略参数自动优化\n"
            "• 模拟交易：实时K线匹配与虚拟下单\n\n"
            "版本：1.1.0"
        )
    
    # ============ 模拟交易相关方法 ============
    
    def _paper_api_config_path(self) -> str:
        save_dir = os.path.join("data", "paper_trading")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, "api_config.json")
    
    def _load_saved_paper_api_config(self):
        """启动时加载已保存的模拟交易API配置"""
        try:
            path = self._paper_api_config_path()
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.paper_trading_tab.control_panel.set_api_config(cfg)
            self.statusBar().showMessage("已加载模拟交易API配置", 3000)
        except Exception as e:
            print(f"[MainWindow] 加载API配置失败: {e}")
    
    def _load_paper_trade_history_on_start(self):
        """程序启动时从本地文件加载历史交易记录并显示"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            history_file = os.path.join(project_root, "data", "live_trade_history.json")
            history = load_trade_history_from_file(history_file)
            if history:
                self.paper_trading_tab.load_historical_trades(history)
                self.statusBar().showMessage(f"已加载 {len(history)} 条历史交易记录", 3000)
        except Exception as e:
            print(f"[MainWindow] 加载历史交易记录失败: {e}")
    
    def _on_paper_api_save_requested(self, cfg: dict):
        """保存模拟交易API配置"""
        try:
            path = self._paper_api_config_path()
            payload = {
                "symbol": cfg.get("symbol", "BTCUSDT"),
                "interval": cfg.get("interval", "1m"),
                "api_key": cfg.get("api_key", ""),
                "api_secret": cfg.get("api_secret", ""),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.paper_trading_tab.control_panel.update_connection_status(
                True, "API配置已保存（下次启动自动加载）"
            )
            self.statusBar().showMessage("模拟交易API配置已保存", 3000)
        except Exception as e:
            msg = f"保存API配置失败: {e}"
            self.paper_trading_tab.control_panel.update_connection_status(False, msg)
            self.statusBar().showMessage(msg, 5000)
    
    def _on_paper_trading_test_connection(self):
        """测试API连接"""
        from core.live_data_feed import LiveDataFeed
        
        config = {
            "symbol": self.paper_trading_tab.control_panel.symbol_combo.currentText(),
            "api_key": self.paper_trading_tab.control_panel.api_key_edit.text().strip() or None,
            "api_secret": self.paper_trading_tab.control_panel.api_secret_edit.text().strip() or None,
            "use_testnet": PAPER_TRADING_CONFIG.get("USE_TESTNET", True),
            "market_type": PAPER_TRADING_CONFIG.get("MARKET_TYPE", "futures"),
        }
        
        # 获取代理设置
        http_proxy, socks_proxy = self._get_proxy_settings()
        
        feed = LiveDataFeed(
            symbol=config["symbol"],
            api_key=config["api_key"],
            api_secret=config["api_secret"],
            use_testnet=config["use_testnet"],
            market_type=config["market_type"],
            http_proxy=http_proxy,
            socks_proxy=socks_proxy,
        )
        
        success, message = feed.test_connection()
        self.paper_trading_tab.control_panel.update_connection_status(success, message)
    
    def _on_paper_trading_start(self, config: dict):
        """启动模拟交易"""
        if self._live_running:
            return

        # 真实测试网执行模式：必须提供API凭证
        if not config.get("api_key") or not config.get("api_secret"):
            QtWidgets.QMessageBox.warning(
                self, "缺少API",
                "当前为 Binance 测试网真实执行模式，必须填写 API Key 和 API Secret。"
            )
            return
        
        # 优先使用聚合指纹图（原型库）
        has_prototypes = (
            self._prototype_library is not None and
            self._prototype_library.total_count > 0
        )
        has_templates = (
            self.trajectory_memory is not None and
            self.trajectory_memory.total_count > 0
        )
        if (not has_prototypes) and (not has_templates):
            QtWidgets.QMessageBox.warning(
                self, "警告",
                "没有可用的原型库或模板库，请先训练并生成原型。"
            )
            return

        # 时间框架/交易对一致性校验（不允许错配）
        selected_symbol = (config.get("symbol") or "").upper()
        selected_interval = (config.get("interval") or "").strip()
        if has_prototypes:
            lib = self._prototype_library
            lib_symbol = (getattr(lib, "source_symbol", "") or "").upper()
            lib_interval = (getattr(lib, "source_interval", "") or "").strip()
            if not lib_symbol or not lib_interval:
                QtWidgets.QMessageBox.warning(
                    self, "原型库缺少来源信息",
                    "当前原型库没有记录来源的交易对/时间框架，\n"
                    "为了避免错误匹配，系统已阻止启动。\n\n"
                    "请使用最新版本重新生成原型库，或在正确的K线周期下重建记忆库再聚类。"
                )
                return
            if lib_symbol != selected_symbol or lib_interval != selected_interval:
                QtWidgets.QMessageBox.warning(
                    self, "时间框架/交易对不匹配",
                    f"原型库来源: {lib_symbol} {lib_interval}\n"
                    f"当前选择: {selected_symbol} {selected_interval}\n\n"
                    "原型与时间框架不一致会导致错误匹配，系统已阻止启动。"
                )
                return
        else:
            mem = self.trajectory_memory
            mem_symbol = (getattr(mem, "source_symbol", "") or "").upper()
            mem_interval = (getattr(mem, "source_interval", "") or "").strip()
            if not mem_symbol or not mem_interval:
                QtWidgets.QMessageBox.warning(
                    self, "记忆库缺少来源信息",
                    "当前模板记忆库没有记录来源的交易对/时间框架，\n"
                    "为了避免错误匹配，系统已阻止启动。\n\n"
                    "请在正确的K线周期下重新生成记忆库。"
                )
                return
            if mem_symbol != selected_symbol or mem_interval != selected_interval:
                QtWidgets.QMessageBox.warning(
                    self, "时间框架/交易对不匹配",
                    f"记忆库来源: {mem_symbol} {mem_interval}\n"
                    f"当前选择: {selected_symbol} {selected_interval}\n\n"
                    "记忆库与时间框架不一致会导致错误匹配，系统已阻止启动。"
                )
                return

        # 模板模式下的合格模板指纹
        qualified_fingerprints = set()
        if (not has_prototypes) and config.get("use_qualified_only", True) and self._last_eval_result:
            qualified_fingerprints = self._last_eval_result.keep_fingerprints
        
        # 模板模式下：如果没有合格模板且选择了只用合格模板，给出警告
        if (not has_prototypes) and config.get("use_qualified_only", True) and not qualified_fingerprints:
            reply = QtWidgets.QMessageBox.question(
                self, "提示",
                "没有经过验证的合格模板。\n\n"
                "是否使用全部模板进行模拟交易？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                return
            config["use_qualified_only"] = False
        
        # 选择模拟交易数据源
        if has_prototypes:
            # 有批量WF结果则用已验证原型；否则直接用全原型（聚合指纹图）
            verified_proto_fps = set(self._last_verified_prototype_fps)
            use_verified_protos = len(verified_proto_fps) > 0
            active_count = len(verified_proto_fps) if use_verified_protos else self._prototype_library.total_count
            long_n = len(self._prototype_library.long_prototypes)
            short_n = len(self._prototype_library.short_prototypes)
            detail = f"LONG={long_n}, SHORT={short_n}" if (not use_verified_protos) else f"已验证={len(verified_proto_fps)}"
            self.paper_trading_tab.control_panel.update_template_count(
                active_count, mode="prototype", detail=detail
            )
        else:
            verified_proto_fps = set()
            use_verified_protos = False
            template_count = len(qualified_fingerprints) if config.get("use_qualified_only") else self.trajectory_memory.total_count
            self.paper_trading_tab.control_panel.update_template_count(
                template_count, mode="template"
            )
        
        # 创建交易引擎
        from core.live_trading_engine import LiveTradingEngine
        
        try:
            # 获取代理设置
            http_proxy, socks_proxy = self._get_proxy_settings()
            
            self._live_engine = LiveTradingEngine(
                trajectory_memory=self.trajectory_memory,
                prototype_library=self._prototype_library if has_prototypes else None,
                symbol=config["symbol"],
                interval=config["interval"],
                initial_balance=config["initial_balance"],
                leverage=config["leverage"],
                use_qualified_only=(config.get("use_qualified_only", True) and (not has_prototypes)),
                qualified_fingerprints=qualified_fingerprints,
                qualified_prototype_fingerprints=(verified_proto_fps if use_verified_protos else set()),
                api_key=config.get("api_key"),
                api_secret=config.get("api_secret"),
                use_testnet=PAPER_TRADING_CONFIG.get("USE_TESTNET", True),
                market_type=PAPER_TRADING_CONFIG.get("MARKET_TYPE", "futures"),
                http_proxy=http_proxy,
                socks_proxy=socks_proxy,
                on_state_update=self._on_live_state_update,
                on_kline=self._on_live_kline,
                on_price_tick=self._on_live_price_tick,
                on_trade_opened=self._on_live_trade_opened,
                on_trade_closed=self._on_live_trade_closed,
                on_error=self._handle_live_error,
            )
            
            success = self._live_engine.start()
            if success:
                self._live_running = True
                self.paper_trading_tab.control_panel.set_running(True)
                # 先获取历史记录（避免 reset 清空后无数据恢复）
                history = self._live_engine.paper_trader.order_history
                if not history:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    history_file = os.path.join(project_root, "data", "live_trade_history.json")
                    history = load_trade_history_from_file(history_file)
                self.paper_trading_tab.reset()
                if history:
                    self.paper_trading_tab.load_historical_trades(history)
                    self.paper_trading_tab.status_panel.append_event(f"成功恢复 {len(history)} 条历史交易记录")
                
                self._live_chart_timer.start()
                if has_prototypes:
                    mode_msg = f"聚合指纹图模式({ '已验证原型' if use_verified_protos else '全原型' })"
                    self.statusBar().showMessage(f"模拟交易已启动: {config['symbol']} | {mode_msg}")
                else:
                    self.statusBar().showMessage(f"模拟交易已启动: {config['symbol']} | 模板模式")
            else:
                QtWidgets.QMessageBox.warning(self, "启动失败", "无法启动模拟交易，请检查网络连接。")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"启动模拟交易失败:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_paper_trading_stop(self):
        """停止模拟交易"""
        if self._live_engine:
            self._live_engine.stop()
        
        self._live_running = False
        self._live_chart_timer.stop()
        self.paper_trading_tab.control_panel.set_running(False)
        self.statusBar().showMessage("模拟交易已停止")
    
    def _on_trade_delete_requested(self, order):
        """删除交易记录"""
        try:
            # 从live_engine的历史记录中删除
            if self._live_engine and hasattr(self._live_engine, 'paper_trader'):
                trader = self._live_engine.paper_trader
                if hasattr(trader, 'order_history'):
                    # 根据订单特征删除（比较order_id或entry_time+entry_price）
                    trader.order_history = [
                        o for o in trader.order_history
                        if not self._is_same_order(o, order)
                    ]
            
            # 更新持久化文件
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            history_file = os.path.join(project_root, "data", "live_trade_history.json")
            
            # 读取现有历史
            existing_history = load_trade_history_from_file(history_file)
            
            # 过滤掉要删除的记录
            filtered_history = [
                o for o in existing_history
                if not self._is_same_order(o, order)
            ]
            
            # 保存回文件
            save_trade_history_to_file(filtered_history, history_file)
            
            self.statusBar().showMessage("交易记录已删除", 3000)
            
        except Exception as e:
            import traceback
            print(f"[MainWindow] 删除交易记录失败: {e}")
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(
                self,
                "删除失败",
                f"删除交易记录时发生错误:\n{str(e)}"
            )
    
    def _is_same_order(self, order1, order2) -> bool:
        """判断两个订单是否相同"""
        # 优先通过order_id判断
        id1 = getattr(order1, "order_id", None)
        id2 = getattr(order2, "order_id", None)
        if id1 and id2 and id1 == id2:
            return True
        
        # 否则通过入场时间+入场价+方向判断
        time1 = getattr(order1, "entry_time", None)
        time2 = getattr(order2, "entry_time", None)
        price1 = getattr(order1, "entry_price", 0.0)
        price2 = getattr(order2, "entry_price", 0.0)
        side1 = getattr(order1, "side", None)
        side2 = getattr(order2, "side", None)
        
        if time1 and time2 and time1 == time2:
            if abs(price1 - price2) < 0.01:
                if side1 and side2 and side1 == side2:
                    return True
        
        return False
    
    def _on_live_state_update(self, state):
        """实时状态更新"""
        # 在主线程中更新UI
        QtCore.QMetaObject.invokeMethod(
            self, "_update_live_state",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, state)
        )
    
    @QtCore.pyqtSlot(object)
    def _update_live_state(self, state):
        """更新实时状态（主线程）"""
        # 更新控制面板
        self.paper_trading_tab.control_panel.update_ws_status(state.is_connected)
        self.paper_trading_tab.control_panel.update_price(state.current_price)
        self.paper_trading_tab.control_panel.update_bar_count(state.total_bars)
        self.paper_trading_tab.control_panel.update_position_direction(state.position_side)
        
        # 更新持仓
        if self._live_engine:
            order = self._live_engine.paper_trader.current_position
            self.paper_trading_tab.status_panel.update_position(order)
            self.paper_trading_tab.status_panel.update_current_price(state.current_price)
            # 更新持仓标记（显示当前持仓在K线上的位置）
            current_idx = getattr(self._live_engine, "_current_bar_idx", None)
            self.paper_trading_tab.update_position_marker(order, current_idx, state.current_price)
            
            # 更新统计
            stats = self._live_engine.get_stats()
            self.paper_trading_tab.status_panel.update_stats(stats)
            self.paper_trading_tab.control_panel.update_account_stats(stats)
            
            # 更新模板统计
            profitable = len(self._live_engine.get_profitable_templates())
            losing = len(self._live_engine.get_losing_templates())
            matched = profitable + losing
            self.paper_trading_tab.status_panel.update_template_stats(matched, profitable, losing)
            
            # 更新匹配状态与因果
            matched_fp = ""
            matched_sim = None
            if order is not None and getattr(order, "template_fingerprint", ""):
                matched_fp = order.template_fingerprint
                matched_sim = getattr(order, "entry_similarity", None)
            elif getattr(state, "best_match_template", None):
                matched_fp = state.best_match_template
                matched_sim = getattr(state, "best_match_similarity", None)

            # 【UI层防护】regime-direction 不一致时清除显示，防止误导
            if matched_fp and not (order is not None and getattr(order, "template_fingerprint", "")):
                regime = state.market_regime
                bull_set = {"强多头", "弱多头", "震荡偏多"}
                bear_set = {"强空头", "弱空头", "震荡偏空"}
                if regime in bull_set and "SHORT" in matched_fp.upper():
                    matched_fp = ""
                    matched_sim = 0.0
                elif regime in bear_set and "LONG" in matched_fp.upper():
                    matched_fp = ""
                    matched_sim = 0.0

            # UI展示用：如果state里暂时没有贝叶斯胜率，按当前匹配原型即时读取后验均值
            bayesian_wr = getattr(state, "bayesian_win_rate", 0.0)
            if bayesian_wr <= 0 and matched_fp and self._live_engine:
                bf = getattr(self._live_engine, "_bayesian_filter", None)
                if bf is not None:
                    try:
                        bayesian_wr = bf.get_expected_win_rate(matched_fp, state.market_regime)
                    except Exception:
                        bayesian_wr = 0.0

            self.paper_trading_tab.status_panel.update_matching_context(
                state.market_regime,
                state.fingerprint_status,
                state.decision_reason,
                matched_fp,
                matched_sim,
                swing_points_count=getattr(state, "swing_points_count", 0),
                entry_threshold=getattr(state, "entry_threshold", None),
                macd_ready=getattr(state, "macd_ready", False),
                kdj_ready=getattr(state, "kdj_ready", False),
                bayesian_win_rate=bayesian_wr,
                kelly_position_pct=getattr(state, "kelly_position_pct", 0.0),
            )
            self.paper_trading_tab.control_panel.update_kelly_position_display(
                getattr(state, "kelly_position_pct", 0.0)
            )
            
            # 【决策说明日志】decision_reason 变化时追加到事件日志
            reason = state.decision_reason or ""
            if reason and reason != "-":
                last_reason = getattr(self, "_last_logged_decision_reason", "")
                if reason != last_reason:
                    self._last_logged_decision_reason = reason
                    self.paper_trading_tab.status_panel.append_event(f"[决策] {reason}")
            # 更新持仓监控 (NEW)
            self.paper_trading_tab.status_panel.update_monitoring(
                state.hold_reason,
                state.danger_level,
                state.exit_reason
            )
            pending_orders = []
            try:
                current_bar_idx = getattr(self._live_engine, "_current_bar_idx", None)
                pending_orders = self._live_engine.paper_trader.get_pending_entry_orders_snapshot(current_bar_idx)
            except Exception:
                pending_orders = []
            self.paper_trading_tab.status_panel.update_pending_orders(pending_orders)
            self.paper_trading_tab.control_panel.update_match_preview(
                matched_fp,
                matched_sim,
                state.fingerprint_status,
            )

            # 若开仓回调未触发，兜底补记开仓记录
            if order is not None:
                entry_key = (
                    getattr(order, "order_id", ""),
                    getattr(order, "entry_time", None),
                    getattr(order, "entry_bar_idx", None),
                    getattr(order, "entry_price", None),
                )
                if getattr(self, "_last_logged_open_key", None) != entry_key:
                    self.paper_trading_tab.trade_log.add_trade(order)
                    self._last_logged_open_key = entry_key
            
            # 检查并显示最新事件到日志
            last_event = getattr(state, "last_event", "")
            if last_event and last_event != getattr(self, "_last_logged_event", ""):
                self._last_logged_event = last_event
                self.paper_trading_tab.status_panel.append_event(last_event)
            
            # 更新指纹轨迹叠加显示
            self._update_fingerprint_trajectory_overlay(state)
    
    def _on_live_price_tick(self, price: float, ts_ms: int):
        """低延迟逐笔价格更新（避免重UI流程）"""
        QtCore.QMetaObject.invokeMethod(
            self, "_update_live_price_tick",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(float, float(price)),
        )

    @QtCore.pyqtSlot(float)
    def _update_live_price_tick(self, price: float):
        """主线程更新价格标签（轻量）"""
        if not self._live_running:
            return
        try:
            self.paper_trading_tab.control_panel.update_price(price)
            self.paper_trading_tab.status_panel.update_current_price(price)
        except Exception:
            pass

    def _on_live_kline(self, kline):
        """实时K线更新"""
        # 在主线程中更新图表
        QtCore.QMetaObject.invokeMethod(
            self, "_update_live_chart",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, kline)
        )
    
    @QtCore.pyqtSlot(object)
    def _update_live_chart(self, kline):
        """更新实时K线图表（主线程）"""
        self._refresh_live_chart()

    @QtCore.pyqtSlot()
    def _on_live_chart_tick(self):
        """1秒定时刷新K线图，保证时间流动感"""
        if not self._live_running:
            return
        self._refresh_live_chart()

    def _refresh_live_chart(self):
        """统一刷新实时图表"""
        if not self._live_engine:
            return
        
        try:
            # 获取历史K线数据
            df = self._live_engine.get_history_df()
            if df.empty:
                return
            
            # 更新模拟交易Tab的图表 (使用增量更新，避免重置信号标记)
            self.paper_trading_tab.chart_widget.update_kline(df)
            
            # 视图随K线滚动更新（仅在 K 线增加时滚动，避免每秒抖动）
            n = len(df)
            if not hasattr(self, "_last_live_n") or n > self._last_live_n:
                self._last_live_n = n
                visible = 50
                future_pad = 0
                if hasattr(self.paper_trading_tab.chart_widget, "get_overlay_padding"):
                    future_pad = self.paper_trading_tab.chart_widget.get_overlay_padding()
                self.paper_trading_tab.chart_widget.candle_plot.setXRange(
                    n - visible, n + max(5, max(0, future_pad)), padding=0
                )
            
            # 【关键】实时更新 TP/SL 虚线位置（追踪止损更新后自动跟随）
            order = self._live_engine._paper_trader.current_position
            if order is not None:
                tp = getattr(order, "take_profit", None)
                sl = getattr(order, "stop_loss", None)
                self.paper_trading_tab.chart_widget.set_tp_sl_lines(tp, sl)
                
                # 【实时偏离检测】持仓中检查价格是否偏离概率扇形置信带
                self._check_deviation_warning(df)
            else:
                # 无持仓时清除虚线
                self.paper_trading_tab.chart_widget.set_tp_sl_lines(None, None)
                
        except Exception as e:
            print(f"[MainWindow] 更新实时图表失败: {e}")
    
    def _check_deviation_warning(self, df):
        """
        持仓中实时偏离检测：检查当前价格是否偏离原型的概率扇形置信带
        
        - inside: 正常 — 价格在25%-75%区间内
        - edge: 边缘预警 — 偏离置信区但未超出极端范围
        - outside: 严重偏离 — 价格超出扩展范围
        """
        chart = self.paper_trading_tab.chart_widget
        if not hasattr(chart, 'check_price_deviation'):
            return
        
        current_price = float(df['close'].iloc[-1])
        current_idx = len(df) - 1
        
        deviation = chart.check_price_deviation(current_price, current_idx)
        # outside 连续确认，降低偶发误报
        if not hasattr(self, "_deviation_outside_count"):
            self._deviation_outside_count = 0
        if deviation == "outside":
            self._deviation_outside_count += 1
            if self._deviation_outside_count < 2:
                deviation = "edge"
        else:
            self._deviation_outside_count = 0
        
        # 节流：同状态不重复报告
        last_deviation = getattr(self, '_last_deviation_state', 'unknown')
        if deviation == last_deviation:
            return
        self._last_deviation_state = deviation
        
        status_panel = getattr(self.paper_trading_tab, 'status_panel', None)
        if status_panel is None:
            return
        
        if deviation == "edge":
            msg = f"[偏离预警] 当前价 {current_price:.2f} 偏离概率置信区间边缘，注意风险"
            status_panel.append_event(msg)
            self.statusBar().showMessage(f"⚠ 偏离预警: 价格偏离置信带边缘", 5000)
            # 与持仓监控联动：提高风险感知，避免UI仍显示低警觉
            try:
                st = self._live_engine.state
                st.danger_level = max(float(getattr(st, "danger_level", 0.0) or 0.0), 60.0)
                st.hold_reason = "价格接近扇形边缘，进入偏离预警。"
                st.exit_reason = "边缘偏离：关注回归失败风险。"
            except Exception:
                pass
        elif deviation == "outside":
            msg = f"[严重偏离] 当前价 {current_price:.2f} 已完全偏离概率扇形，考虑提前离场！"
            status_panel.append_event(msg)
            self.statusBar().showMessage(f"🚨 严重偏离: 价格超出概率扇形范围！", 8000)
            # 与持仓监控联动：显式拉高警觉度
            try:
                st = self._live_engine.state
                st.danger_level = max(float(getattr(st, "danger_level", 0.0) or 0.0), 90.0)
                st.hold_reason = "价格已严重偏离扇形置信带。"
                st.exit_reason = "严重偏离：建议收紧止损或主动减仓。"
            except Exception:
                pass

    def _reconstruct_future_prices_from_features(self, feature_rows: np.ndarray, df, steps: int = 5) -> np.ndarray:
        """
        用32维特征（重点使用C层空间特征）逆向还原未来价格轨迹。
        返回长度=steps 的未来价格（不含当前点）。
        """
        if feature_rows is None or feature_rows.size == 0:
            return np.array([])
        if feature_rows.ndim != 2 or feature_rows.shape[1] < 32:
            return np.array([])

        steps = max(1, min(int(steps), len(feature_rows)))
        f = feature_rows[:steps]

        close_hist = list(df['close'].iloc[-20:].astype(float).values)
        high_hist = list(df['high'].iloc[-20:].astype(float).values)
        low_hist = list(df['low'].iloc[-20:].astype(float).values)
        atr_series = df['atr'] if 'atr' in df.columns else None
        if atr_series is not None and len(atr_series) > 0:
            atr_vals = atr_series.iloc[-20:].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            atr_ref = float(np.median(atr_vals)) if len(atr_vals) > 0 else 0.0
        else:
            atr_ref = 0.0
        if atr_ref <= 0:
            atr_ref = max((max(high_hist) - min(low_hist)) / max(len(close_hist), 1), close_hist[-1] * 0.001)

        out = []
        prev = float(close_hist[-1])
        for i in range(steps):
            row = f[i]
            c0 = float(np.clip(row[26], 0.0, 1.0))   # price_in_range
            c1 = max(0.0, float(row[27]))            # dist_to_high_atr
            c2 = max(0.0, float(row[28]))            # dist_to_low_atr
            c4 = float(np.clip(row[30], 0.0, 1.0))   # price_vs_20high
            c5 = float(np.clip(row[31], 0.0, 1.0))   # price_vs_20low

            high_ref = max(high_hist)
            low_ref = min(low_hist)
            range_ref = max(high_ref - low_ref, max(prev * 0.0005, 1e-6))

            # 多方程逆推候选（来源于Layer-C定义）
            cand = []
            cand.append(low_ref + c0 * range_ref)                       # from price_in_range
            cand.append(high_ref - c1 * atr_ref)                        # from dist_to_high_atr
            cand.append(low_ref + c2 * atr_ref)                         # from dist_to_low_atr
            cand.append(high_ref - (1.0 - c4) * range_ref)              # from price_vs_20high
            cand.append(low_ref + (1.0 - c5) * range_ref)               # from price_vs_20low

            w = np.array([0.42, 0.22, 0.22, 0.07, 0.07], dtype=float)
            price = float(np.dot(w, np.array(cand, dtype=float)))

            # 平滑与限幅，防止跳点
            max_step = max(prev * 0.01, 2.5 * atr_ref)
            delta = np.clip(price - prev, -max_step, max_step)
            price = prev + 0.65 * delta

            out.append(price)
            prev = price
            close_hist.append(price)
            high_hist.append(price)
            low_hist.append(price)
            if len(close_hist) > 20:
                close_hist.pop(0)
                high_hist.pop(0)
                low_hist.pop(0)

        return np.array(out, dtype=float)

    def _update_fingerprint_trajectory_overlay(self, state):
        """
        将匹配原型的概率扇形图叠加到K线图上
        
        使用原型成员的真实历史交易数据（收益率+持仓时长）构建概率分布，
        而非从特征向量反推价格，确保方向一致性和真实性。
        """
        if not self._live_engine:
            return
        chart = getattr(self.paper_trading_tab, "chart_widget", None)
        if chart is None:
            return
        
        df = chart.df
        if df is None or df.empty:
            return
        
        # 获取匹配信息
        matched_sim = None
        if self._live_engine.paper_trader and self._live_engine.paper_trader.current_position:
            matched_sim = getattr(self._live_engine.paper_trader.current_position, "entry_similarity", None)
        if matched_sim is None:
            matched_sim = getattr(state, "best_match_similarity", 0.0)
        
        matched_fp = getattr(state, "best_match_template", "") or ""
        
        # 获取当前匹配的原型（优先引擎状态，其次从原型库解析）
        proto = getattr(self._live_engine, "_current_prototype", None)
        if proto is None and matched_fp:
            proto = self._find_prototype_from_match(matched_fp)
        if proto is None and not matched_fp:
            return
        
        # 节流：同一bar+同一原型不重复重算（但首次绘制不跳过）
        current_bar_idx = int(getattr(self._live_engine, "_current_bar_idx", len(df) - 1))
        overlay_sig = (getattr(proto, "prototype_id", matched_fp), current_bar_idx)
        if getattr(self, "_last_overlay_signature", None) == overlay_sig:
            return
        self._last_overlay_signature = overlay_sig
        
        # 优先绘制概率扇形图（原型模式）
        if proto is not None:
            member_stats = getattr(proto, "member_trade_stats", [])
            if not member_stats or len(member_stats) < 3:
                member_stats = self._synthesize_member_stats(proto)
            
            if member_stats and len(member_stats) >= 3:
                direction = proto.direction
                regime_short = proto.regime[:2] if proto.regime else ""
                label = f"{direction} {regime_short}_{proto.prototype_id}"
                
                current_price = float(df["close"].iloc[-1])
                leverage = getattr(self._live_engine, "fixed_leverage", 10.0)
                start_idx = len(df) - 1
                chart.set_probability_fan(
                    entry_price=current_price,
                    start_idx=start_idx,
                    member_trade_stats=member_stats,
                    direction=direction,
                    similarity=matched_sim or 0.0,
                    label=label,
                    leverage=leverage,
                    max_bars=5,
                )
                return
        
        # 回退：没有可用原型数据时，显示旧的“未来5根K线”预测轨迹
        template = None
        if matched_fp and not matched_fp.startswith("proto_") and self.trajectory_memory:
            template = self.trajectory_memory.get_template_by_fingerprint(matched_fp)
        if template is None:
            template = getattr(self._live_engine, "_current_template", None)
        if template is None or template.holding.size == 0:
            return
        traj_future = template.holding
        if traj_future.ndim != 2 or traj_future.shape[1] < 32:
            return
        projected_future = self._reconstruct_future_prices_from_features(traj_future, df, steps=5)
        if projected_future.size == 0:
            return
        current_price = float(df["close"].iloc[-1])
        recent_n = min(80, len(df))
        recent_range = float(df["high"].iloc[-recent_n:].max() - df["low"].iloc[-recent_n:].min())
        band_base = max(current_price * 0.0008, recent_range * 0.02)
        band_steps = np.linspace(0.35, 1.0, len(projected_future))
        band_future = band_base * band_steps
        prices = np.concatenate([[current_price], projected_future], axis=0)
        lower = np.concatenate([[current_price], projected_future - band_future], axis=0)
        upper = np.concatenate([[current_price], projected_future + band_future], axis=0)
        start_idx = len(df) - 1
        label = f"{template.direction} {template.fingerprint()[:8]}"
        chart.set_fingerprint_trajectory(
            prices, start_idx, matched_sim or 0.0, label,
            lower=lower, upper=upper
        )
    
    @staticmethod
    def _synthesize_member_stats(proto) -> list:
        """
        从原型的汇总统计（avg_profit_pct, avg_hold_bars, member_count, win_rate）
        合成近似的 member_trade_stats，用于兼容旧原型库绘制概率扇形图。
        
        生成方式：以均值为中心，模拟合理的散布分布
        """
        avg_profit = getattr(proto, "avg_profit_pct", 0.0)
        avg_hold = getattr(proto, "avg_hold_bars", 0.0)
        member_count = getattr(proto, "member_count", 0)
        win_rate = getattr(proto, "win_rate", 0.0)
        
        if member_count < 3 or avg_hold <= 0:
            return []
        
        n = max(member_count, 5)  # 至少生成5条路径
        n = min(n, 30)  # 上限30条，避免计算过多
        
        import numpy as np
        rng = np.random.RandomState(int(abs(avg_profit * 1000) + avg_hold))  # 固定种子，同原型结果一致
        
        stats = []
        for i in range(n):
            # 根据胜率决定是盈利还是亏损
            is_win = rng.random() < win_rate
            
            if is_win:
                # 盈利交易：在平均收益附近波动 (±50%)
                profit = avg_profit * (0.5 + rng.random())
            else:
                # 亏损交易：小幅亏损（平均收益的负面）
                profit = -abs(avg_profit) * (0.2 + rng.random() * 0.5)
            
            # 持仓时长：在平均值附近波动 (±60%)
            hold = int(avg_hold * (0.4 + rng.random() * 1.2))
            hold = max(2, hold)
            
            stats.append((float(profit), hold))
        
        return stats

    def _find_prototype_from_match(self, matched_fp: str):
        """
        从匹配指纹中解析原型ID并在已加载的原型库中查找。
        期望格式: proto_LONG_28_震荡 / proto_SHORT_12_强空
        """
        if not matched_fp:
            return None
        library = getattr(self, "_prototype_library", None)
        if library is None:
            return None
        import re
        m = re.match(r"proto_(LONG|SHORT)_(\d+)", matched_fp)
        if not m:
            return None
        direction = m.group(1)
        proto_id = int(m.group(2))
        candidates = library.long_prototypes if direction == "LONG" else library.short_prototypes
        for p in candidates:
            if getattr(p, "prototype_id", None) == proto_id:
                return p
        return None
    
    def _on_live_trade_opened(self, order):
        """实时交易开仓回调"""
        # 在主线程中处理
        QtCore.QMetaObject.invokeMethod(
            self, "_handle_live_trade_opened",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, order)
        )
    
    @QtCore.pyqtSlot(object)
    def _handle_live_trade_opened(self, order):
        """处理实时交易开仓（主线程）"""
        try:
            # 添加图表标记
            side = order.side.value
            self.paper_trading_tab.add_trade_marker(
                bar_idx=getattr(order, "entry_bar_idx", None),
                price=order.entry_price,
                side=side,
                is_entry=True
            )
            
            # 绘制止盈止损线（sync 来的仓位可能无 TP/SL）
            tp = getattr(order, "take_profit", None)
            sl = getattr(order, "stop_loss", None)
            self.paper_trading_tab.update_tp_sl_lines(tp_price=tp, sl_price=sl)
            
            # 记录事件
            fp_short = order.template_fingerprint[:12] if order.template_fingerprint else "-"
            tp_text = f"{order.take_profit:.2f}" if getattr(order, "take_profit", None) is not None else "未设置"
            sl_text = f"{order.stop_loss:.2f}" if getattr(order, "stop_loss", None) is not None else "未设置"
            event_msg = (
                f"[开仓] {side} @ {order.entry_price:.2f} | "
                f"TP={tp_text} SL={sl_text} | "
                f"原型={fp_short} (相似度={order.entry_similarity:.2%})"
            )
            self.paper_trading_tab.status_panel.append_event(event_msg)
            
            # 添加到交易记录表格（开仓时即显示，状态为持仓中）
            self.paper_trading_tab.trade_log.add_trade(order)
            self._last_logged_open_key = (
                getattr(order, "order_id", ""),
                getattr(order, "entry_time", None),
                getattr(order, "entry_bar_idx", None),
                getattr(order, "entry_price", None),
            )
            
            print(f"[MainWindow] 实时交易开仓: {event_msg}")
        except Exception as e:
            print(f"[MainWindow] 处理开仓失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_live_trade_closed(self, order):
        """实时交易平仓回调"""
        # 在主线程中处理
        QtCore.QMetaObject.invokeMethod(
            self, "_handle_live_trade_closed",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, order)
        )
    
    @QtCore.pyqtSlot(object)
    def _handle_live_trade_closed(self, order):
        """处理实时交易平仓（主线程）"""
        try:
            # 添加平仓标记（区分保本/止盈/脱轨/信号/超时）
            side = order.side.value
            exit_bar = getattr(order, "exit_bar_idx", None)
            exit_px = getattr(order, "exit_price", None)
            
            # 根据真实平仓原因 + 追踪阶段 确定标记类型
            close_reason_str = None
            if order.close_reason:
                reason_val = order.close_reason.value  # "止盈"/"止损"/"脱轨"/"超时"/"信号"/"手动"
                trailing = getattr(order, "trailing_stage", 0)
                if reason_val == "止盈" and trailing >= 1 and order.profit_pct < 0.3:
                    # 追踪止损触发在保本区 (利润<0.3%) → 保本平仓
                    close_reason_str = "保本"
                elif reason_val == "止盈":
                    # 真正的止盈（利润较大）
                    close_reason_str = "止盈"
                elif reason_val == "止损" and trailing >= 1:
                    # 追踪阶段的止损 → 实际是保本触发
                    close_reason_str = "保本"
                elif reason_val == "止损":
                    # 原始止损触发（无追踪保护）
                    close_reason_str = "止损"
                else:
                    close_reason_str = reason_val  # 脱轨/超时/信号/手动
            
            self.paper_trading_tab.add_trade_marker(
                bar_idx=exit_bar,
                price=exit_px,
                side=side,
                is_entry=False,
                close_reason=close_reason_str
            )
            
            # 清除止盈止损线
            self.paper_trading_tab.update_tp_sl_lines(None, None)
            
            # 添加到交易记录表格
            self.paper_trading_tab.trade_log.add_trade(order)
            
            # 记录事件（使用细化后的平仓原因）
            reason_display = close_reason_str or (order.close_reason.value if order.close_reason else "未知")
            profit_color = "盈利" if order.profit_pct >= 0 else "亏损"
            pnl_usdt = getattr(order, "realized_pnl", 0.0)
            event_msg = (
                f"[平仓] {side} @ {order.exit_price:.2f} | "
                f"{profit_color} {order.profit_pct:+.2f}% ({pnl_usdt:+.2f} USDT) | "
                f"原因={reason_display} | 持仓={order.hold_bars}根K线"
            )
            self.paper_trading_tab.status_panel.append_event(event_msg)
            
            print(f"[MainWindow] 实时交易平仓: {event_msg}")
        except Exception as e:
            print(f"[MainWindow] 处理平仓失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_proxy_settings(self):
        """获取代理设置"""
        http_proxy = None
        socks_proxy = None
        
        if hasattr(self.paper_trading_tab.control_panel, 'proxy_edit'):
            proxy_text = self.paper_trading_tab.control_panel.proxy_edit.text().strip()
            if proxy_text:
                if proxy_text.startswith('socks'):
                    socks_proxy = proxy_text
                else:
                    http_proxy = proxy_text
        
        return http_proxy, socks_proxy
    
    def _on_live_error(self, error_msg: str):
        """实时交易错误"""
        QtCore.QMetaObject.invokeMethod(
            self, "_handle_live_error",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, error_msg)
        )
    
    @QtCore.pyqtSlot(str)
    def _handle_live_error(self, error_msg: str):
        """处理错误（主线程）"""
        self.statusBar().showMessage(f"错误: {error_msg}")
        self.paper_trading_tab.status_panel.append_event(f"错误: {error_msg}")
    
    def _on_save_profitable_templates(self):
        """保存盈利模板"""
        if not self._live_engine:
            self.paper_trading_tab.status_panel.set_action_status("模拟交易未运行")
            return
        
        profitable_fps = self._live_engine.get_profitable_templates()
        if not profitable_fps:
            self.paper_trading_tab.status_panel.set_action_status("没有盈利的模板")
            return
        
        # 将这些模板标记为"实战验证"
        # 实际上模板已经在记忆库中，这里可以更新评估结果
        count = len(profitable_fps)
        
        # 保存到文件
        import json
        import os
        from datetime import datetime
        
        save_dir = "data/sim_verified"
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"profitable_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump({
                "fingerprints": profitable_fps,
                "count": count,
                "timestamp": timestamp,
            }, f, indent=2)
        
        self.paper_trading_tab.status_panel.set_action_status(
            f"已保存 {count} 个盈利模板到:\n{filepath}"
        )
        
        QtWidgets.QMessageBox.information(
            self, "保存成功",
            f"已保存 {count} 个盈利模板指纹。\n\n"
            f"文件: {filepath}"
        )
    
    def _on_delete_losing_templates(self):
        """删除亏损模板"""
        if not self._live_engine:
            self.paper_trading_tab.status_panel.set_action_status("模拟交易未运行")
            return
        
        if getattr(self._live_engine, "use_prototypes", False):
            self.paper_trading_tab.status_panel.set_action_status("原型模式下不支持删除亏损模板")
            return
        
        losing_fps = self._live_engine.get_losing_templates()
        if not losing_fps:
            self.paper_trading_tab.status_panel.set_action_status("没有亏损的模板")
            return
        
        count = len(losing_fps)
        
        reply = QtWidgets.QMessageBox.question(
            self, "确认删除",
            f"确定要从记忆库中删除 {count} 个亏损模板吗？\n\n"
            "此操作不可撤销！",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        
        # 从记忆库删除
        if self.trajectory_memory:
            removed = self.trajectory_memory.remove_by_fingerprints(set(losing_fps))
            self.trajectory_memory.save()
            
            # 更新UI
            self.analysis_panel.trajectory_widget.update_memory_stats(
                self.trajectory_memory.total_count,
                self.trajectory_memory.count_by_direction("LONG"),
                self.trajectory_memory.count_by_direction("SHORT"),
            )
            
            self.paper_trading_tab.status_panel.set_action_status(
                f"已删除 {removed} 个亏损模板"
            )
            
            QtWidgets.QMessageBox.information(
                self, "删除成功",
                f"已从记忆库中删除 {removed} 个亏损模板。"
            )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 检查是否有正在进行的操作
        running_tasks = []
        if self.is_playing:
            running_tasks.append("标注")
        if self._live_running:
            running_tasks.append("模拟交易")
        
        if running_tasks:
            reply = QtWidgets.QMessageBox.question(
                self,
                "确认退出",
                f"{', '.join(running_tasks)}正在进行中，确定要退出吗？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                # 停止标注
                if self.labeling_worker:
                    self.labeling_worker.stop()
                if self.worker_thread:
                    self.worker_thread.quit()
                    self.worker_thread.wait(1000)
                
                # 停止模拟交易
                if self._live_engine:
                    self._live_engine.stop()
                
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
