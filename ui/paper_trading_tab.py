"""
R3000 模拟交易Tab
实时模拟交易界面：连接行情、模板匹配、虚拟下单

布局：
  - 左侧：控制面板（API配置、启动/停止、参数设置）
  - 中间：K线图 + 交易记录
  - 右侧：实时状态 + 账户统计 + 模板操作
"""

from PyQt6 import QtWidgets, QtCore, QtGui
from typing import Optional, Dict, List, Set
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UI_CONFIG, VECTOR_SPACE_CONFIG, MARKET_REGIME_CONFIG
from core.paper_trader import OrderStatus


class PaperTradingControlPanel(QtWidgets.QWidget):
    """模拟交易控制面板（左侧）"""
    
    # 信号
    start_requested = QtCore.pyqtSignal(dict)  # 启动请求，携带配置
    stop_requested = QtCore.pyqtSignal()       # 停止请求
    test_connection_requested = QtCore.pyqtSignal()  # 测试连接
    save_api_requested = QtCore.pyqtSignal(dict)      # 保存API配置
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        self.setFixedWidth(280)
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
            QPushButton#startBtn {{
                background-color: #089981;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#startBtn:hover {{
                background-color: #0ab090;
            }}
            QPushButton#stopBtn {{
                background-color: #f23645;
            }}
            QPushButton#stopBtn:hover {{
                background-color: #ff4555;
            }}
            QLineEdit {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                border: 1px solid #444;
                border-radius: 3px;
                padding: 5px;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                border: 1px solid #444;
                border-radius: 3px;
                padding: 3px;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # === 交易对设置 ===
        symbol_group = QtWidgets.QGroupBox("交易设置")
        symbol_layout = QtWidgets.QFormLayout(symbol_group)
        
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
        self.symbol_combo.setCurrentText("BTCUSDT")
        symbol_layout.addRow("交易对:", self.symbol_combo)
        
        self.interval_combo = QtWidgets.QComboBox()
        self.interval_combo.addItems(["1m", "3m", "5m", "15m", "30m", "1h", "4h"])
        self.interval_combo.setCurrentText("1m")
        symbol_layout.addRow("时间框架:", self.interval_combo)
        
        layout.addWidget(symbol_group)
        
        # === API配置 ===
        api_group = QtWidgets.QGroupBox("API配置（必填，测试网真实执行）")
        api_layout = QtWidgets.QFormLayout(api_group)
        
        self.api_key_edit = QtWidgets.QLineEdit()
        self.api_key_edit.setPlaceholderText("必须填写")
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        api_layout.addRow("API Key:", self.api_key_edit)
        
        self.api_secret_edit = QtWidgets.QLineEdit()
        self.api_secret_edit.setPlaceholderText("必须填写")
        self.api_secret_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        api_layout.addRow("API Secret:", self.api_secret_edit)
        
        self.test_conn_btn = QtWidgets.QPushButton("测试连接")
        self.test_conn_btn.clicked.connect(self.test_connection_requested.emit)
        api_layout.addRow(self.test_conn_btn)
        
        self.save_api_btn = QtWidgets.QPushButton("保存API配置")
        self.save_api_btn.clicked.connect(self._on_save_api_clicked)
        api_layout.addRow(self.save_api_btn)
        
        self.conn_status_label = QtWidgets.QLabel("未测试")
        self.conn_status_label.setStyleSheet("color: #888;")
        api_layout.addRow("状态:", self.conn_status_label)
        
        layout.addWidget(api_group)
        
        # === 账户设置与统计（合并） ===
        account_group = QtWidgets.QGroupBox("账户设置与统计")
        account_layout = QtWidgets.QFormLayout(account_group)
        
        self.balance_spin = QtWidgets.QDoubleSpinBox()
        self.balance_spin.setRange(100, 1000000)
        self.balance_spin.setValue(5000)
        self.balance_spin.setSuffix(" USDT")
        account_layout.addRow("初始资金:", self.balance_spin)
        
        self.leverage_spin = QtWidgets.QSpinBox()
        self.leverage_spin.setRange(10, 10)
        self.leverage_spin.setValue(10)
        self.leverage_spin.setSuffix("x")
        self.leverage_spin.setToolTip("实时执行固定为 10x")
        account_layout.addRow("杠杆:", self.leverage_spin)

        self.position_size_hint_label = QtWidgets.QLabel("50%")
        self.position_size_hint_label.setStyleSheet("color: #9ad1ff;")
        account_layout.addRow("单次仓位:", self.position_size_hint_label)
        
        # 实时统计（合并展示）
        self.snapshot_balance_label = QtWidgets.QLabel("-")
        self.snapshot_balance_label.setStyleSheet("color: #ccc; font-weight: bold;")
        account_layout.addRow("当前权益:", self.snapshot_balance_label)

        self.snapshot_available_margin_label = QtWidgets.QLabel("-")
        self.snapshot_available_margin_label.setStyleSheet("color: #9ad1ff;")
        account_layout.addRow("可用保证金:", self.snapshot_available_margin_label)
        
        self.snapshot_pnl_label = QtWidgets.QLabel("-")
        account_layout.addRow("累计盈亏:", self.snapshot_pnl_label)
        
        self.snapshot_winrate_label = QtWidgets.QLabel("-")
        account_layout.addRow("胜率:", self.snapshot_winrate_label)
        
        layout.addWidget(account_group)
        
        # === 聚合指纹图筛选 ===
        template_group = QtWidgets.QGroupBox("聚合指纹图筛选")
        template_layout = QtWidgets.QVBoxLayout(template_group)
        
        self.use_all_radio = QtWidgets.QRadioButton("使用全部聚合指纹图")
        self.use_qualified_radio = QtWidgets.QRadioButton("仅用已验证聚合指纹图")
        self.use_qualified_radio.setChecked(True)
        
        template_layout.addWidget(self.use_all_radio)
        template_layout.addWidget(self.use_qualified_radio)
        
        self.template_count_label = QtWidgets.QLabel("可用聚合指纹图: 0 个")
        self.template_count_label.setStyleSheet("color: #888; font-size: 11px;")
        template_layout.addWidget(self.template_count_label)

        self.last_matched_proto_label = QtWidgets.QLabel("-")
        self.last_matched_proto_label.setWordWrap(True)
        self.last_matched_proto_label.setStyleSheet("color: #9fd6ff; font-size: 11px;")
        template_layout.addWidget(QtWidgets.QLabel("当前匹配:"))
        template_layout.addWidget(self.last_matched_proto_label)

        self.last_match_sim_label = QtWidgets.QLabel("-")
        self.last_match_sim_label.setStyleSheet("color: #888; font-size: 11px;")
        template_layout.addWidget(self.last_match_sim_label)
        
        layout.addWidget(template_group)
        
        # === 控制按钮 ===
        control_group = QtWidgets.QGroupBox("控制")
        control_layout = QtWidgets.QVBoxLayout(control_group)
        
        self.start_btn = QtWidgets.QPushButton("▶ 启动模拟盘")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._on_start_clicked)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QtWidgets.QPushButton("■ 停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        control_layout.addWidget(self.stop_btn)
        
        layout.addWidget(control_group)
        
        # === 运行状态 ===
        status_group = QtWidgets.QGroupBox("运行状态")
        status_layout = QtWidgets.QFormLayout(status_group)
        
        self.run_status_label = QtWidgets.QLabel("未运行")
        self.run_status_label.setStyleSheet("color: #888;")
        status_layout.addRow("状态:", self.run_status_label)
        
        self.ws_status_label = QtWidgets.QLabel("未连接")
        self.ws_status_label.setStyleSheet("color: #888;")
        status_layout.addRow("WebSocket:", self.ws_status_label)
        
        self.current_price_label = QtWidgets.QLabel("-")
        status_layout.addRow("当前价格:", self.current_price_label)
        
        self.bar_count_label = QtWidgets.QLabel("0")
        status_layout.addRow("已处理K线:", self.bar_count_label)
        
        self.pos_dir_label = QtWidgets.QLabel("-")
        status_layout.addRow("持仓方向:", self.pos_dir_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
    
    def _on_start_clicked(self):
        config = {
            "symbol": self.symbol_combo.currentText(),
            "interval": self.interval_combo.currentText(),
            "api_key": self.api_key_edit.text().strip() or None,
            "api_secret": self.api_secret_edit.text().strip() or None,
            "initial_balance": self.balance_spin.value(),
            "leverage": self.leverage_spin.value(),
            "use_qualified_only": self.use_qualified_radio.isChecked(),
        }
        self.start_requested.emit(config)
    
    def _on_save_api_clicked(self):
        config = {
            "symbol": self.symbol_combo.currentText(),
            "interval": self.interval_combo.currentText(),
            "api_key": self.api_key_edit.text().strip() or "",
            "api_secret": self.api_secret_edit.text().strip() or "",
        }
        self.save_api_requested.emit(config)
    
    def set_running(self, running: bool):
        """设置运行状态"""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.symbol_combo.setEnabled(not running)
        self.interval_combo.setEnabled(not running)
        self.api_key_edit.setEnabled(not running)
        self.api_secret_edit.setEnabled(not running)
        self.save_api_btn.setEnabled(not running)
        self.balance_spin.setEnabled(not running)
        self.leverage_spin.setEnabled(not running)
        
        if running:
            self.run_status_label.setText("运行中")
            self.run_status_label.setStyleSheet("color: #089981; font-weight: bold;")
        else:
            self.run_status_label.setText("已停止")
            self.run_status_label.setStyleSheet("color: #888;")
    
    def update_connection_status(self, success: bool, message: str):
        """更新连接状态"""
        if success:
            self.conn_status_label.setText(message)
            self.conn_status_label.setStyleSheet("color: #089981;")
        else:
            self.conn_status_label.setText(message)
            self.conn_status_label.setStyleSheet("color: #f23645;")
    
    def update_ws_status(self, connected: bool):
        """更新WebSocket状态"""
        if connected:
            self.ws_status_label.setText("已连接")
            self.ws_status_label.setStyleSheet("color: #089981;")
        else:
            self.ws_status_label.setText("断开")
            self.ws_status_label.setStyleSheet("color: #f23645;")
    
    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price_label.setText(f"${price:,.2f}")
    
    def update_bar_count(self, count: int):
        """更新K线计数"""
        self.bar_count_label.setText(str(count))
    
    def update_template_count(self, count: int, mode: str = "prototype", detail: str = ""):
        """更新可用匹配池数量（区分原型/模板，避免误解）"""
        if mode == "template":
            text = f"可用模板: {count} 个"
        else:
            text = f"可用聚合指纹图: {count} 个"
        if detail:
            text = f"{text} ({detail})"
        self.template_count_label.setText(text)

    def update_match_preview(self, fp: str, similarity: float, fp_status: str = ""):
        """更新左侧筛选区中的匹配预览（聚合指纹图）"""
        if fp:
            self.last_matched_proto_label.setText(fp if len(fp) <= 28 else (fp[:28] + "..."))
        else:
            self.last_matched_proto_label.setText("-")

        if similarity is None:
            sim_text = "-"
            color = "#888"
        else:
            sim_text = f"配合度: {float(similarity):.2%}"
            if similarity >= 0.75:
                color = "#089981"
            elif similarity >= 0.60:
                color = "#FFD54F"
            else:
                color = "#f23645"
        if fp_status:
            sim_text = f"{sim_text} | 状态: {fp_status}"
        self.last_match_sim_label.setText(sim_text)
        self.last_match_sim_label.setStyleSheet(f"color: {color}; font-size: 11px;")
    
    def set_api_config(self, cfg: dict):
        """回填API配置"""
        if not cfg:
            return
        symbol = cfg.get("symbol")
        if symbol:
            idx = self.symbol_combo.findText(symbol)
            if idx >= 0:
                self.symbol_combo.setCurrentIndex(idx)
        interval = cfg.get("interval")
        if interval:
            idx = self.interval_combo.findText(interval)
            if idx >= 0:
                self.interval_combo.setCurrentIndex(idx)
        self.api_key_edit.setText(cfg.get("api_key", ""))
        self.api_secret_edit.setText(cfg.get("api_secret", ""))
    
    def update_account_stats(self, stats: dict):
        """更新合并的账户统计快照"""
        bal = stats.get("current_balance", 0.0)
        available = stats.get("available_margin", 0.0)
        pnl = stats.get("total_pnl", 0.0)
        win_rate = stats.get("win_rate", 0.0)
        self.snapshot_balance_label.setText(f"{bal:,.2f} USDT")
        self.snapshot_available_margin_label.setText(f"{available:,.2f} USDT")
        pnl_color = "#089981" if pnl >= 0 else "#f23645"
        self.snapshot_pnl_label.setText(f"{pnl:+,.2f} USDT")
        self.snapshot_pnl_label.setStyleSheet(f"color: {pnl_color};")
        wr_color = "#089981" if win_rate >= 0.5 else "#f23645"
        self.snapshot_winrate_label.setText(f"{win_rate:.1%}")
        self.snapshot_winrate_label.setStyleSheet(f"color: {wr_color};")
    
    def update_position_direction(self, side: str):
        """更新运行状态中的持仓方向"""
        self.pos_dir_label.setText(side or "-")
        if side == "LONG":
            self.pos_dir_label.setStyleSheet("color: #089981; font-weight: bold;")
        elif side == "SHORT":
            self.pos_dir_label.setStyleSheet("color: #f23645; font-weight: bold;")
        else:
            self.pos_dir_label.setStyleSheet("color: #888;")


class PaperTradingStatusPanel(QtWidgets.QWidget):
    """模拟交易状态面板（右侧）"""
    
    # 信号
    save_profitable_requested = QtCore.pyqtSignal()  # 保存盈利模板
    delete_losing_requested = QtCore.pyqtSignal()    # 删除亏损模板
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        self.setFixedWidth(320)
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
                padding: 8px 15px;
                border-radius: 4px;
                font-size: 12px;
            }}
            QPushButton#saveBtn {{
                background-color: #089981;
                color: white;
                border: none;
            }}
            QPushButton#saveBtn:hover {{
                background-color: #0ab090;
            }}
            QPushButton#deleteBtn {{
                background-color: #f23645;
                color: white;
                border: none;
            }}
            QPushButton#deleteBtn:hover {{
                background-color: #ff4555;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # === 实时持仓状态 ===
        position_group = QtWidgets.QGroupBox("当前持仓")
        position_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
            }
        """)
        position_layout = QtWidgets.QFormLayout(position_group)
        
        self.position_side_label = QtWidgets.QLabel("-")
        self.position_side_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        position_layout.addRow("方向:", self.position_side_label)
        
        self.position_qty_label = QtWidgets.QLabel("-")
        position_layout.addRow("数量:", self.position_qty_label)
        
        self.position_margin_label = QtWidgets.QLabel("-")
        position_layout.addRow("保证金占用:", self.position_margin_label)
        
        self.position_entry_label = QtWidgets.QLabel("-")
        position_layout.addRow("入场价:", self.position_entry_label)
        
        self.position_current_label = QtWidgets.QLabel("-")
        position_layout.addRow("当前价:", self.position_current_label)
        
        self.position_pnl_label = QtWidgets.QLabel("-")
        self.position_pnl_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        position_layout.addRow("浮动盈亏:", self.position_pnl_label)
        
        self.position_pnl_pct_label = QtWidgets.QLabel("-")
        position_layout.addRow("收益率:", self.position_pnl_pct_label)
        
        # 追踪状态
        self.tracking_status_label = QtWidgets.QLabel("-")
        self.tracking_status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        position_layout.addRow("追踪状态:", self.tracking_status_label)
        
        layout.addWidget(position_group)
        
        # === 匹配与市场状态 ===
        market_group = QtWidgets.QGroupBox("匹配与市场状态")
        market_layout = QtWidgets.QFormLayout(market_group)
        
        self.market_regime_label = QtWidgets.QLabel("未知")
        market_layout.addRow("市场状态:", self.market_regime_label)
        
        self.swing_points_label = QtWidgets.QLabel(f"0 / {MARKET_REGIME_CONFIG.get('LOOKBACK_SWINGS', 4)}")
        self.swing_points_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.swing_points_label.setToolTip(f"已检测到的摆动点数量 / 激活分类所需的最少点数({MARKET_REGIME_CONFIG.get('LOOKBACK_SWINGS', 4)}: 3高+3低)")
        market_layout.addRow("摆动点检测:", self.swing_points_label)
        
        self.fingerprint_status_label = QtWidgets.QLabel("待匹配")
        market_layout.addRow("指纹匹配:", self.fingerprint_status_label)

        self.matched_fingerprint_label = QtWidgets.QLabel("-")
        self.matched_fingerprint_label.setWordWrap(True)
        self.matched_fingerprint_label.setMinimumWidth(120)
        self.matched_fingerprint_label.setStyleSheet("color: #9fd6ff; font-weight: bold; font-size: 12px;")
        market_layout.addRow("匹配原型:", self.matched_fingerprint_label)

        # 实时配合度 + 开仓阈值 + 距离
        self.matched_similarity_label = QtWidgets.QLabel("-")
        self.matched_similarity_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        market_layout.addRow("实时配合度:", self.matched_similarity_label)
        
        self.entry_threshold_label = QtWidgets.QLabel("-")
        self.entry_threshold_label.setStyleSheet("color: #888;")
        market_layout.addRow("开仓阈值:", self.entry_threshold_label)
        
        self.distance_to_entry_label = QtWidgets.QLabel("-")
        self.distance_to_entry_label.setStyleSheet("font-weight: bold;")
        market_layout.addRow("距离开仓:", self.distance_to_entry_label)
        
        self.reason_label = QtWidgets.QLabel("-")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("color: #bbb;")
        market_layout.addRow("决策说明:", self.reason_label)
        
        # 动能门控 (Aim/Exit)
        self.indicators_container = QtWidgets.QWidget()
        indicators_h_layout = QtWidgets.QHBoxLayout(self.indicators_container)
        indicators_h_layout.setContentsMargins(0, 5, 0, 5)
        indicators_h_layout.setSpacing(8)
        
        self.macd_status_badge = QtWidgets.QLabel(" MACD ")
        self.macd_status_badge.setStyleSheet(self._badge_style(False))
        self.macd_status_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        indicators_h_layout.addWidget(self.macd_status_badge)
        
        self.kdj_status_badge = QtWidgets.QLabel(" KDJ ")
        self.kdj_status_badge.setStyleSheet(self._badge_style(False))
        self.kdj_status_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        indicators_h_layout.addWidget(self.kdj_status_badge)
        indicators_h_layout.addStretch()
        
        market_layout.addRow("动能门控:", self.indicators_container)
        
        layout.addWidget(market_group)

        # === 持仓监控与说明 (NEW) ===
        monitor_group = QtWidgets.QGroupBox("持仓监控与说明")
        monitor_layout = QtWidgets.QVBoxLayout(monitor_group)

        # 1. 为何继续持仓
        monitor_layout.addWidget(QtWidgets.QLabel("【持仓理由】"))
        self.hold_reason_label = QtWidgets.QLabel("未持仓")
        self.hold_reason_label.setWordWrap(True)
        self.hold_reason_label.setStyleSheet("color: #ccc; padding: 2px;")
        monitor_layout.addWidget(self.hold_reason_label)

        # 2. 持仓警觉度 (Danger Bar)
        monitor_layout.addWidget(QtWidgets.QLabel("【持仓警觉度】(100%触碰平仓线)"))
        self.danger_bar = QtWidgets.QProgressBar()
        self.danger_bar.setRange(0, 100)
        self.danger_bar.setValue(0)
        self.danger_bar.setTextVisible(True)
        self.danger_bar.setFormat("%p%")
        self.danger_bar.setFixedHeight(12)
        self.danger_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: #333;
                color: white;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                stop:0 #089981, stop:0.5 #FFD54F, stop:1 #f23645);
            }
        """)
        monitor_layout.addWidget(self.danger_bar)

        # 3. 平仓状态监控
        monitor_layout.addWidget(QtWidgets.QLabel("【平仓预判】"))
        self.exit_monitor_label = QtWidgets.QLabel("-")
        self.exit_monitor_label.setWordWrap(True)
        self.exit_monitor_label.setStyleSheet("color: #ef9a9a; padding: 2px;")
        monitor_layout.addWidget(self.exit_monitor_label)

        layout.addWidget(monitor_group)
        
        # === 右下事件日志 ===
        event_group = QtWidgets.QGroupBox("实时日志")
        event_layout = QtWidgets.QVBoxLayout(event_group)
        self.event_log = QtWidgets.QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumHeight(160)
        self.event_log.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                border: 1px solid #444;
                color: {UI_CONFIG['THEME_TEXT']};
                font-size: 11px;
            }}
        """)
        event_layout.addWidget(self.event_log)
        layout.addWidget(event_group)
        
        layout.addStretch()
    
    def update_position(self, order):
        """更新持仓显示"""
        if order is None:
            self.position_side_label.setText("-")
            self.position_side_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")
            self.position_qty_label.setText("-")
            self.position_margin_label.setText("-")
            self.position_entry_label.setText("-")
            self.position_current_label.setText("-")
            self.position_pnl_label.setText("-")
            self.position_pnl_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #888;")
            self.position_pnl_pct_label.setText("-")
            self.tracking_status_label.setText("-")
            self.tracking_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #888;")
        else:
            # 方向
            side_text = order.side.value
            side_color = "#089981" if order.side.value == "LONG" else "#f23645"
            self.position_side_label.setText(side_text)
            self.position_side_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {side_color};")
            
            # 数量
            self.position_qty_label.setText(f"{order.quantity:.6f}")
            self.position_margin_label.setText(f"{order.margin_used:,.2f} USDT")
            
            # 入场价
            self.position_entry_label.setText(f"${order.entry_price:,.2f}")
            
            # 盈亏
            pnl_color = "#089981" if order.unrealized_pnl >= 0 else "#f23645"
            self.position_pnl_label.setText(f"{order.unrealized_pnl:+,.2f} USDT")
            self.position_pnl_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {pnl_color};")
            
            self.position_pnl_pct_label.setText(f"{order.profit_pct:+.2f}%")
            self.position_pnl_pct_label.setStyleSheet(f"color: {pnl_color};")
            
            # 追踪状态
            tracking = order.tracking_status
            if tracking == "安全":
                tracking_color = "#089981"
                tracking_icon = "🟢"
            elif tracking == "警戒":
                tracking_color = "#FFD54F"
                tracking_icon = "🟡"
            elif tracking == "危险":
                tracking_color = "#FF8C00"
                tracking_icon = "🟠"
            else:
                tracking_color = "#f23645"
                tracking_icon = "🔴"
            
            self.tracking_status_label.setText(f"{tracking_icon} {tracking}")
            self.tracking_status_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {tracking_color};")
            
    def update_monitoring(self, hold_reason: str, danger_level: float, exit_reason: str):
        """更新持仓监控说明 (NEW)"""
        self.hold_reason_label.setText(hold_reason or "未持仓")
        self.danger_bar.setValue(int(danger_level))
        self.exit_monitor_label.setText(exit_reason or "-")
        
    def update_matching_context(self, market_regime: str, fp_status: str, reason: str,
                                matched_fp: str = "", matched_similarity: float = None,
                                swing_points_count: int = 0,
                                entry_threshold: float = None,
                                macd_ready: bool = False,
                                kdj_ready: bool = False):
        """更新匹配状态和因果说明"""
        self.macd_status_badge.setStyleSheet(self._badge_style(macd_ready))
        self.kdj_status_badge.setStyleSheet(self._badge_style(kdj_ready))
        regime = market_regime or "未知"
        self.market_regime_label.setText(regime)
        
        # 更新摆动点计数显示
        lookback = MARKET_REGIME_CONFIG.get("LOOKBACK_SWINGS", 4)
        sp_text = f"{swing_points_count} / {lookback}"
        if swing_points_count >= lookback:
            sp_color = "#089981"  # 绿色 - 已激活分类
            sp_text += "  [已激活]"
        elif swing_points_count >= 1:
            sp_color = "#ffaa00"  # 黄色 - 检测中
            sp_text += "  [检测中...]"
        else:
            sp_color = "#f23645"  # 红色 - 等待
            sp_text += "  [等待数据]"
        self.swing_points_label.setText(sp_text)
        self.swing_points_label.setStyleSheet(f"color: {sp_color}; font-weight: bold;")
        
        # 根据6态市场状态着色（与上帝视角训练一致）
        regime_colors = {
            "强多头": "#00E676",   # 亮绿
            "弱多头": "#66BB6A",   # 绿
            "震荡偏多": "#A5D6A7", # 浅绿
            "震荡偏空": "#EF9A9A", # 浅红
            "弱空头": "#EF5350",   # 红
            "强空头": "#FF1744",   # 亮红
            "未知": "#888888",     # 灰
        }
        color = regime_colors.get(regime, "#888888")
        self.market_regime_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.fingerprint_status_label.setText(fp_status or "待匹配")
        if matched_fp:
            # 完整显示原型名称，并设置 tooltip
            self.matched_fingerprint_label.setText(matched_fp)
            self.matched_fingerprint_label.setToolTip(matched_fp)
            # 根据方向着色
            if "LONG" in matched_fp:
                self.matched_fingerprint_label.setStyleSheet(
                    "color: #089981; font-weight: bold; font-size: 12px;")
            elif "SHORT" in matched_fp:
                self.matched_fingerprint_label.setStyleSheet(
                    "color: #f23645; font-weight: bold; font-size: 12px;")
            else:
                self.matched_fingerprint_label.setStyleSheet(
                    "color: #9fd6ff; font-weight: bold; font-size: 12px;")
        else:
            self.matched_fingerprint_label.setText("-")
            self.matched_fingerprint_label.setToolTip("")
            self.matched_fingerprint_label.setStyleSheet(
                "color: #9fd6ff; font-weight: bold; font-size: 12px;")

        # 开仓阈值优先使用引擎运行时值，避免UI与执行逻辑不一致
        if entry_threshold is None or entry_threshold <= 0:
            entry_threshold = VECTOR_SPACE_CONFIG.get("ENTRY_SIM_THRESHOLD", 70.0) / 100.0
        else:
            entry_threshold = float(entry_threshold)
        
        if matched_similarity is None or matched_similarity <= 0:
            self.matched_similarity_label.setText("-")
            self.matched_similarity_label.setStyleSheet("color: #888; font-weight: bold; font-size: 13px;")
            self.entry_threshold_label.setText(f"{entry_threshold:.0%}")
            self.distance_to_entry_label.setText("-")
            self.distance_to_entry_label.setStyleSheet("color: #888; font-weight: bold;")
        else:
            sim = float(matched_similarity)
            self.matched_similarity_label.setText(f"{sim:.2%}")
            
            # 根据相似度着色
            if sim >= entry_threshold:
                color = "#089981"  # 绿色 - 达到开仓条件
                self.distance_to_entry_label.setText("✓ 已达标")
                self.distance_to_entry_label.setStyleSheet("color: #089981; font-weight: bold;")
            elif sim >= entry_threshold - 0.1:
                color = "#FFD54F"  # 黄色 - 接近
                distance = entry_threshold - sim
                self.distance_to_entry_label.setText(f"差 {distance:.1%}")
                self.distance_to_entry_label.setStyleSheet("color: #FFD54F; font-weight: bold;")
            else:
                color = "#f23645"  # 红色 - 差距较大
                distance = entry_threshold - sim
                self.distance_to_entry_label.setText(f"差 {distance:.1%}")
                self.distance_to_entry_label.setStyleSheet("color: #f23645; font-weight: bold;")
            
            self.matched_similarity_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")
            self.entry_threshold_label.setText(f"{entry_threshold:.0%}")
            self.entry_threshold_label.setStyleSheet("color: #888;")
        
        self.reason_label.setText(reason or "-")
    
    def append_event(self, text: str):
        """追加右下事件日志"""
        t = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
        self.event_log.appendPlainText(f"[{t}] {text}")
        sb = self.event_log.verticalScrollBar()
        sb.setValue(sb.maximum())
    
    def update_current_price(self, price: float):
        """更新当前价格"""
        self.position_current_label.setText(f"${price:,.2f}")

    def _badge_style(self, active: bool) -> str:
        """生成指标徽章样式"""
        bg_color = "#089981" if active else "#333"
        text_color = "#fff" if active else "#777"
        border_color = "#0ab090" if active else "#555"
        return f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
                padding: 2px 6px;
                min-width: 45px;
            }}
        """

    @staticmethod
    def _fmt_percent(value: float) -> str:
        """百分比格式化：极大值使用科学计数法 a × 10^b%"""
        try:
            v = float(value)
        except Exception:
            return "-"
        if not np.isfinite(v):
            return "-"
        av = abs(v)
        if av >= 1e6:
            sign = "+" if v >= 0 else "-"
            b = int(np.floor(np.log10(av)))
            a = av / (10 ** b)
            return f"{sign}{a:.3f} × 10^{b}%"
        return f"{v:+.2f}%"
    
    def update_stats(self, stats: dict):
        """更新账户统计（面板简化后，账户统计主要显示在配置区快照中，此处仅作为接口保留或处理状态）"""
        # 面板已简化，不再显示冗余的账户详情
        pass
    
    def update_template_stats(self, matched: int, profitable: int, losing: int):
        """更新模板统计（面板简化后，统计逻辑已移除，接口保留以兼容主流程）"""
        pass
    
    def set_action_status(self, message: str):
        """设置操作状态"""
        self.action_status_label.setText(message)


class PaperTradingTradeLog(QtWidgets.QWidget):
    """模拟交易记录表格"""
    
    # 定义信号
    delete_trade_signal = QtCore.pyqtSignal(object)  # 删除交易记录信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._rows_by_key = {}
    
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.stacked = QtWidgets.QStackedWidget()
        # 空状态页
        empty_page = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty_page)
        self.empty_label = QtWidgets.QLabel("暂无交易记录\n\n启动模拟交易后，此处将显示交易明细")
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(f"color: #666; font-size: 12px; padding: 30px; background-color: {UI_CONFIG['THEME_SURFACE']};")
        empty_layout.addWidget(self.empty_label)
        self.stacked.addWidget(empty_page)
        
        # 表格页
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "时间", "方向", "入场价", "出场价", "止盈", "止损", "盈亏%", "盈亏(USDT)", "手续费", "原因", "相似度", "持仓", "操作"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #444;
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QHeaderView::section {{
                background-color: #333;
                color: {UI_CONFIG['THEME_TEXT']};
                border: 1px solid #444;
                padding: 4px;
            }}
            QTableWidget::item:alternate {{
                background-color: #2a2a2a;
            }}
        """)
        self.table.setMinimumHeight(80)
        table_page = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.table)
        self.stacked.addWidget(table_page)
        layout.addWidget(self.stacked)
    
    def _update_empty_state(self):
        """根据是否有数据显示空状态或表格"""
        has_data = self.table.rowCount() > 0
        self.stacked.setCurrentIndex(1 if has_data else 0)
    
    def add_trade(self, order):
        """添加单个交易记录"""
        key = self._trade_key(order)
        if key in self._rows_by_key:
            # 已存在则更新（例如平仓、或同步更新）
            self._update_trade_row(self._rows_by_key[key], order)
        else:
            row = self._insert_trade_row(order)
            self._rows_by_key[key] = row
        self._update_empty_state()
    
    def set_history(self, trades: List):
        """批量设置历史记录"""
        self.table.setRowCount(0)
        self._rows_by_key.clear()
        for order in trades:
            row = self._insert_trade_row(order)
            self._rows_by_key[self._trade_key(order)] = row
        self._update_empty_state()
            
    def _trade_key(self, order) -> str:
        """生成稳定的交易标识"""
        order_id = str(getattr(order, "order_id", "") or "")
        if order_id and not order_id.startswith("EXCHANGE_SYNC"):
            return order_id
        side = getattr(order, "side", None)
        side_val = side.value if side else "-"
        entry_price = getattr(order, "entry_price", 0.0)
        quantity = getattr(order, "quantity", 0.0)
        return f"SYNC-{side_val}-{entry_price:.2f}-{quantity:.6f}"
    
    def _insert_trade_row(self, order):
        """内部通用插入行逻辑"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # 为了美观，新纪录放前面？或者按时间排序。这里维持原有顺序，但在 TableWidget 中 insertRow(0) 可以置顶
        # 目前按时间顺序追加
        
        self._update_trade_row(row, order)
        # 滚动到最新
        self.table.scrollToBottom()
        return row
    
    def _update_trade_row(self, row: int, order):
        """更新表格行数据"""
        # 时间
        time_str = "-"
        if order.exit_time:
            time_str = order.exit_time.strftime("%m-%d %H:%M")
        elif order.entry_time:
            time_str = order.entry_time.strftime("%m-%d %H:%M") + "(持)"
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
        
        # 方向
        side_val = order.side.value
        side_item = QtWidgets.QTableWidgetItem(side_val)
        side_color = QtGui.QColor("#089981") if side_val == "LONG" else QtGui.QColor("#f23645")
        side_item.setForeground(side_color)
        self.table.setItem(row, 1, side_item)
        
        # 入场价
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{order.entry_price:.2f}"))
        
        # 出场价
        exit_price = order.exit_price if order.exit_price else "-"
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{exit_price:.2f}" if isinstance(exit_price, float) else exit_price))

        # 止盈 / 止损
        tp_val = getattr(order, "take_profit", None)
        sl_val = getattr(order, "stop_loss", None)
        tp_text = f"{tp_val:.2f}" if isinstance(tp_val, float) else "-"
        sl_text = f"{sl_val:.2f}" if isinstance(sl_val, float) else "-"
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(tp_text))
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(sl_text))

        # 盈亏%
        pnl_pct_item = QtWidgets.QTableWidgetItem(f"{order.profit_pct:+.2f}%")
        pnl_color = QtGui.QColor("#089981") if order.profit_pct >= 0 else QtGui.QColor("#f23645")
        pnl_pct_item.setForeground(pnl_color)
        self.table.setItem(row, 6, pnl_pct_item)
        
        # 盈亏(USDT) - 开仓显示未实现，平仓显示已实现
        is_closed = getattr(order, "status", None) == OrderStatus.CLOSED or order.exit_time is not None
        if is_closed:
            pnl_val = getattr(order, "realized_pnl", 0.0)
        else:
            pnl_val = getattr(order, "unrealized_pnl", 0.0)
        pnl_usdt_item = QtWidgets.QTableWidgetItem(f"{pnl_val:+,.2f}")
        pnl_usdt_item.setForeground(pnl_color)
        self.table.setItem(row, 7, pnl_usdt_item)
        
        # 手续费
        fee_val = getattr(order, "total_fee", 0.0)
        fee_item = QtWidgets.QTableWidgetItem(f"{fee_val:.4f}")
        fee_item.setForeground(QtGui.QColor("#f9a825"))  # 黄色
        self.table.setItem(row, 8, fee_item)
        
        # 原因
        reason = order.close_reason.value if order.close_reason else "-"
        self.table.setItem(row, 9, QtWidgets.QTableWidgetItem(reason))
        
        # 相似度（从第11列移到第10列）
        self.table.setItem(row, 10, QtWidgets.QTableWidgetItem(f"{order.entry_similarity:.2%}"))
        
        # 持仓时长（从第12列移到第11列）
        self.table.setItem(row, 11, QtWidgets.QTableWidgetItem(str(order.hold_bars)))
        
        # 操作按钮（第12列）
        delete_btn = QtWidgets.QPushButton("删除")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:pressed {
                background-color: #8b0000;
            }
        """)
        delete_btn.clicked.connect(lambda checked=False, o=order: self._on_delete_clicked(o))
        self.table.setCellWidget(row, 12, delete_btn)
    
    def _on_delete_clicked(self, order):
        """删除按钮点击事件"""
        # 确认对话框
        reply = QtWidgets.QMessageBox.question(
            self.table,
            "确认删除",
            f"确定要删除此交易记录吗？\n\n"
            f"时间: {order.entry_time.strftime('%m-%d %H:%M') if order.entry_time else '-'}\n"
            f"方向: {order.side.value}\n"
            f"入场价: {order.entry_price:.2f}",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        
        # 从表格中删除
        key = self._trade_key(order)
        if key in self._rows_by_key:
            row = self._rows_by_key[key]
            self.table.removeRow(row)
            del self._rows_by_key[key]
            
            # 更新后续行的索引映射
            for k, v in list(self._rows_by_key.items()):
                if v > row:
                    self._rows_by_key[k] = v - 1
        
        self._update_empty_state()
        
        # 触发删除信号，让主窗口处理数据持久化
        self.delete_trade_signal.emit(order)
    
    def clear(self):
        """清空表格"""
        self.table.setRowCount(0)
        self._update_empty_state()


class PaperTradingTab(QtWidgets.QWidget):
    """模拟交易标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
            }}
        """)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 左侧：控制面板
        self.control_panel = PaperTradingControlPanel()
        layout.addWidget(self.control_panel)
        
        # 中间区域
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)
        
        # K线图（使用现有的ChartWidget）—— 占主要空间
        from ui.chart_widget import ChartWidget
        self.chart_widget = ChartWidget()
        self.chart_widget.setMinimumHeight(350)
        center_layout.addWidget(self.chart_widget, stretch=4)
        
        # 交易记录
        trade_group = QtWidgets.QGroupBox("交易记录")
        trade_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """)
        trade_layout = QtWidgets.QVBoxLayout(trade_group)
        self.trade_log = PaperTradingTradeLog()
        trade_layout.addWidget(self.trade_log)
        trade_group.setMinimumHeight(120)
        
        center_layout.addWidget(trade_group, stretch=1)
        
        layout.addWidget(center_widget, stretch=1)
        
        # 右侧：状态面板
        self.status_panel = PaperTradingStatusPanel()
        layout.addWidget(self.status_panel)
    
    def load_historical_trades(self, trades: List):
        """加载历史交易记录到界面"""
        self.trade_log.set_history(trades)
        
    def reset(self):
        """重置界面（不清空交易记录，历史数据应保留）"""
        self.status_panel.update_position(None)
        self.status_panel.update_matching_context("未知", "待匹配", "-")
        self.control_panel.update_match_preview("", None, "待匹配")
        self.status_panel.event_log.clear()
        self.status_panel.update_stats({
            "initial_balance": 0,
            "current_balance": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "total_trades": 0,
            "win_rate": 0,
            "max_drawdown_pct": 0,
        })
        self.status_panel.update_template_stats(0, 0, 0)
        self.control_panel.update_account_stats({
            "current_balance": 0,
            "total_pnl": 0,
            "win_rate": 0,
        })
        self.control_panel.update_position_direction("-")
    
    def add_trade_marker(self, bar_idx: int, price: float, side: str,
                         is_entry: bool = True, close_reason: str = None):
        """
        在图表上添加交易标记
        
        Args:
            bar_idx: K线索引
            price: 价格
            side: 方向（LONG/SHORT）
            is_entry: True=入场，False=离场
            close_reason: 平仓原因字符串（止盈/止损/脱轨/超时/信号/手动）
        """
        if bar_idx is None or price is None:
            return
        
        if is_entry:
            signal_type = 1 if side == "LONG" else -1
        else:
            # 根据 close_reason 映射到不同标记类型
            # 5=保本, 6=部分止盈, 7=脱轨, 8=信号离场, 9=超时, 2/-2=普通EXIT
            reason_map = {
                "保本": 5,      # 追踪止损保本触发
                "止盈": 6,      # 止盈
                "脱轨": 7,      # 相似度脱轨
                "信号": 8,      # 信号离场
                "超时": 9,      # 超过最大持仓
                "止损": 10,     # 止损
            }
            signal_type = reason_map.get(close_reason, 2 if side == "LONG" else -2)
        
        self.chart_widget.signal_marker.add_signal(bar_idx, price, signal_type)
    
    def update_tp_sl_lines(self, tp_price: float = None, sl_price: float = None):
        """
        更新图表上的止盈止损虚线（InfiniteLine）
        
        Args:
            tp_price: 止盈价格，None则隐藏TP线
            sl_price: 止损价格，None则隐藏SL线
        """
        # 直接调用新的 InfiniteLine 接口
        self.chart_widget.set_tp_sl_lines(tp_price, sl_price)

    def update_position_marker(self, order, bar_idx: int = None, price: float = None):
        """
        更新当前持仓标记（单点，随K线移动）
        """
        if order is None:
            self.chart_widget.set_current_position_marker()
            return
        side = order.side.value
        self.chart_widget.set_current_position_marker(bar_idx, price, side)