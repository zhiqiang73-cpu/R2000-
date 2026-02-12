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
from config import UI_CONFIG, VECTOR_SPACE_CONFIG


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
        
        self.swing_points_label = QtWidgets.QLabel("0 / 4")
        self.swing_points_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.swing_points_label.setToolTip("已检测到的摆动点数量 / 激活分类所需的最少点数(4: 2高+2低)")
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
                                entry_threshold: float = None):
        """更新匹配状态和因果说明"""
        regime = market_regime or "未知"
        self.market_regime_label.setText(regime)
        
        # 更新摆动点计数显示
        sp_text = f"{swing_points_count} / 4"
        if swing_points_count >= 4:
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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "时间", "方向", "入场价", "出场价", "盈亏%", "原因", "模板", "相似度", "持仓"
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
        
        layout.addWidget(self.table)
    
    def add_trade(self, order):
        """添加单个交易记录"""
        self._insert_trade_row(order)
    
    def set_history(self, trades: List):
        """批量设置历史记录"""
        self.table.setRowCount(0)
        for order in trades:
            self._insert_trade_row(order)
            
    def _insert_trade_row(self, order):
        """内部通用插入行逻辑"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # 为了美观，新纪录放前面？或者按时间排序。这里维持原有顺序，但在 TableWidget 中 insertRow(0) 可以置顶
        # 目前按时间顺序追加
        
        # 时间
        time_str = "-"
        if order.exit_time:
            time_str = order.exit_time.strftime("%m-%d %H:%M")
        elif order.entry_time:
            time_str = order.entry_time.strftime("%m-%d %H:%M") + "(持)"
            
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
        
        # 方向
        side_item = QtWidgets.QTableWidgetItem(order.side.value)
        side_color = QtGui.QColor("#089981") if order.side.value == "LONG" else QtGui.QColor("#f23645")
        side_item.setForeground(side_color)
        self.table.setItem(row, 1, side_item)
        
        # 入场价
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{order.entry_price:.2f}"))
        
        # 出场价
        exit_price = order.exit_price if order.exit_price else "-"
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{exit_price:.2f}" if isinstance(exit_price, float) else exit_price))
        
        # 盈亏%
        pnl_item = QtWidgets.QTableWidgetItem(f"{order.profit_pct:+.2f}%")
        pnl_color = QtGui.QColor("#089981") if order.profit_pct >= 0 else QtGui.QColor("#f23645")
        pnl_item.setForeground(pnl_color)
        self.table.setItem(row, 4, pnl_item)
        
        # 原因
        reason = order.close_reason.value if order.close_reason else "-"
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(reason))
        
        # 模板
        template = order.template_fingerprint[:8] if order.template_fingerprint else "-"
        self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(template))
        
        # 相似度
        self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(f"{order.entry_similarity:.2%}"))
        
        # 持仓时长
        self.table.setItem(row, 8, QtWidgets.QTableWidgetItem(str(order.hold_bars)))
        
        # 滚动到最新
        self.table.scrollToBottom()
    
    def clear(self):
        """清空表格"""
        self.table.setRowCount(0)


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
        
        center_layout.addWidget(trade_group, stretch=1)
        
        layout.addWidget(center_widget, stretch=1)
        
        # 右侧：状态面板
        self.status_panel = PaperTradingStatusPanel()
        layout.addWidget(self.status_panel)
    
    def load_historical_trades(self, trades: List):
        """加载历史交易记录到界面"""
        self.trade_log.set_history(trades)
        
    def reset(self):
        """重置界面"""
        self.trade_log.clear()
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
    
    def add_trade_marker(self, bar_idx: int, price: float, side: str, is_entry: bool = True):
        """
        在图表上添加交易标记
        
        Args:
            bar_idx: K线索引
            price: 价格
            side: 方向（LONG/SHORT）
            is_entry: True=入场，False=离场
        """
        if side == "LONG":
            signal_type = 1 if is_entry else 2
        else:  # SHORT
            signal_type = -1 if is_entry else -2
        
        self.chart_widget.signal_marker.add_signal(bar_idx, price, signal_type)
    
    def update_tp_sl_lines(self, tp_price: float = None, sl_price: float = None):
        """
        更新图表上的止盈止损线
        
        Args:
            tp_price: 止盈价
            sl_price: 止损价
        """
        if tp_price is not None and sl_price is not None:
            self.chart_widget._set_tp_sp(tp_price, sl_price)
        else:
            # 清除TP/SL线
            self.chart_widget._last_tp = None
            self.chart_widget._last_sp = None
            self.chart_widget._update_tp_sp_segment()