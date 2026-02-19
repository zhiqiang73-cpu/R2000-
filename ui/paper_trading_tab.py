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
from config import (UI_CONFIG, VECTOR_SPACE_CONFIG, MARKET_REGIME_CONFIG,
                     SIMILARITY_CONFIG, PAPER_TRADING_CONFIG, COLD_START_CONFIG)
from core.paper_trader import OrderStatus


class PaperTradingControlPanel(QtWidgets.QWidget):
    """模拟交易控制面板（左侧）"""
    
    # 信号
    start_requested = QtCore.pyqtSignal(dict)  # 启动请求，携带配置
    stop_requested = QtCore.pyqtSignal()       # 停止请求
    test_connection_requested = QtCore.pyqtSignal()  # 测试连接
    save_api_requested = QtCore.pyqtSignal(dict)      # 保存API配置
    clear_memory_requested = QtCore.pyqtSignal()      # 清除学习记忆
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._kelly_last_update_ts = 0.0
        self._kelly_blink_state = False
        self._kelly_timer = QtCore.QTimer(self)
        self._kelly_timer.timeout.connect(self._update_kelly_heartbeat)
        self._kelly_timer.start(500)
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
        
        # === 账户设置与统计（合并，移动到持仓页） ===
        account_group = QtWidgets.QGroupBox("账户设置与统计")
        account_layout = QtWidgets.QFormLayout(account_group)
        
        self.balance_spin = QtWidgets.QDoubleSpinBox()
        self.balance_spin.setRange(100, 1000000)
        self.balance_spin.setValue(5000)
        self.balance_spin.setSuffix(" USDT")
        account_layout.addRow("初始资金:", self.balance_spin)
        
        self.leverage_spin = QtWidgets.QSpinBox()
        self.leverage_spin.setRange(5, 100)
        self.leverage_spin.setValue(20)
        self.leverage_spin.setSuffix("x")
        self.leverage_spin.setToolTip("默认 20x；自适应每笔开平仓按盈亏在 5x~100x 间自动调整")
        account_layout.addRow("杠杆:", self.leverage_spin)

        # 单次仓位 + 凯利公式标识 + 心跳灯
        position_size_container = QtWidgets.QWidget()
        position_size_h_layout = QtWidgets.QHBoxLayout(position_size_container)
        position_size_h_layout.setContentsMargins(0, 0, 0, 0)
        position_size_h_layout.setSpacing(5)
        
        self.position_size_hint_label = QtWidgets.QLabel("50%")
        self.position_size_hint_label.setStyleSheet("color: #9ad1ff; font-weight: bold; font-size: 13px;")
        position_size_h_layout.addWidget(self.position_size_hint_label)
        
        self.kelly_formula_badge = QtWidgets.QLabel("[凯利]")
        self.kelly_formula_badge.setStyleSheet("""
            QLabel {
                color: #FFD700;
                background-color: rgba(255, 215, 0, 0.15);
                border: 1px solid #FFD700;
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self.kelly_formula_badge.setToolTip("凯利公式动态仓位（根据贝叶斯胜率和盈亏比计算）")
        self.kelly_formula_badge.hide()  # 默认隐藏，有凯利仓位时显示
        position_size_h_layout.addWidget(self.kelly_formula_badge)
        
        self.kelly_heartbeat_label = QtWidgets.QLabel("●")
        self.kelly_heartbeat_label.setStyleSheet("color: #666; font-size: 12px;")
        self.kelly_heartbeat_label.setToolTip("凯利仓位心跳\n绿色闪烁=动态仓位更新中\n灰色=固定仓位")
        position_size_h_layout.addWidget(self.kelly_heartbeat_label)
        position_size_h_layout.addStretch()
        
        account_layout.addRow("单次仓位:", position_size_container)
        
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
        
        # 注意：账户设置区移动到右侧“持仓”页显示
        self.account_group = account_group
        
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
        
        # 清除学习记忆按钮
        self.clear_memory_btn = QtWidgets.QPushButton("🗑 清除学习记忆")
        self.clear_memory_btn.setToolTip(
            "清除所有自适应学习数据：\n"
            "• 贝叶斯过滤器统计\n"
            "• 凯利仓位学习数据\n"
            "• TP/SL评估记录\n"
            "• 拒绝追踪记录\n"
            "• 冷启动状态\n\n"
            "⚠ 交易历史记录将保留"
        )
        self.clear_memory_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #FF9800;
                border: 1px solid #FF9800;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #FF9800;
                color: #000;
            }
        """)
        self.clear_memory_btn.clicked.connect(self._on_clear_memory_clicked)
        control_layout.addWidget(self.clear_memory_btn)
        
        # 反向下单模式开关
        self.reverse_signal_checkbox = QtWidgets.QCheckBox("🔄 反向下单模式")
        self.reverse_signal_checkbox.setStyleSheet("""
            QCheckBox {
                color: #FF5252;
                font-size: 12px;
                font-weight: bold;
                padding: 5px 0;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:checked {
                background-color: #FF5252;
                border: 1px solid #FF5252;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
            }
        """)
        self.reverse_signal_checkbox.setToolTip(
            "测试功能：将所有LONG信号变为SHORT，SHORT变为LONG\n"
            "用于测试信号方向是否反了\n"
            "⚠ 仅用于诊断，不要依赖此模式长期交易"
        )
        self.reverse_signal_checkbox.stateChanged.connect(self._on_reverse_mode_changed)
        control_layout.addWidget(self.reverse_signal_checkbox)

        # 精品信号模式开关
        self.signal_mode_checkbox = QtWidgets.QCheckBox("💎 精品信号开仓")
        self.signal_mode_checkbox.setChecked(True)
        self.signal_mode_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {UI_CONFIG['THEME_ACCENT']};
                font-size: 12px;
                font-weight: bold;
                padding: 5px 0;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                border: 1px solid {UI_CONFIG['THEME_ACCENT']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:unchecked {{
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
            }}
        """)
        self.signal_mode_checkbox.setToolTip(
            "勾选：按精品信号组合开仓（固定5%仓位 + 固定TP/SL）\n"
            "取消：使用原指纹/原型匹配策略"
        )
        self.signal_mode_checkbox.stateChanged.connect(self._on_signal_mode_changed)
        control_layout.addWidget(self.signal_mode_checkbox)
        
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
        
        self.weight_mode_label = QtWidgets.QLabel("-")
        self.weight_mode_label.setStyleSheet("color: #888;")
        status_layout.addRow("匹配权重:", self.weight_mode_label)
        
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
    
    def _on_reverse_mode_changed(self, state):
        """反向下单模式开关变更"""
        enabled = (state == QtCore.Qt.CheckState.Checked.value)
        
        # 更新配置
        from config import PAPER_TRADING_CONFIG
        PAPER_TRADING_CONFIG["REVERSE_SIGNAL_MODE"] = enabled
        
        # 更新引擎（如果已经运行）
        if hasattr(self, '_engine') and self._engine:
            self._engine._reverse_signal_mode = enabled
        
        # UI提示
        if enabled:
            print(f"[UI] ⚠️ 反向模式已启用！所有信号将反向操作")
            self.reverse_signal_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #FF5252;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 5px 0;
                    background-color: rgba(255, 82, 82, 0.15);
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:checked {
                    background-color: #FF5252;
                    border: 1px solid #FF5252;
                    border-radius: 3px;
                }
            """)
        else:
            print(f"[UI] 反向模式已关闭")
            self.reverse_signal_checkbox.setStyleSheet("""
                QCheckBox {
                    color: #FF5252;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 5px 0;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #333;
                    border: 1px solid #555;
                    border-radius: 3px;
                }
            """)

    def _on_signal_mode_changed(self, state):
        """精品信号模式开关变更"""
        enabled = (state == QtCore.Qt.CheckState.Checked.value)
        # 更新引擎（如果已经运行）
        if hasattr(self, '_engine') and self._engine:
            self._engine.use_signal_mode = enabled
        print(f"[UI] 精品信号模式: {'开启' if enabled else '关闭'}")

    def update_signal_mode_info(self, info: dict):
        """转发到 status_panel（标签在那边）"""
        pass
    
    def _on_clear_memory_clicked(self):
        """清除学习记忆按钮点击"""
        # 弹出确认对话框
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认清除学习记忆",
            "确定要清除所有自适应学习数据吗？\n\n"
            "将清除：\n"
            "• 贝叶斯过滤器统计\n"
            "• 凯利仓位学习数据\n"
            "• TP/SL评估记录\n"
            "• 拒绝追踪记录\n"
            "• 冷启动状态\n\n"
            "交易历史记录将保留。\n\n"
            "此操作不可撤销！",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.clear_memory_requested.emit()
    
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

    def update_weight_mode(self, using_evolved: Optional[bool] = None):
        """更新匹配权重显示：进化后 / 进化前（模拟交易运行时在 UI 端区分）。None=未运行显示 -"""
        if using_evolved is None:
            self.weight_mode_label.setText("-")
            self.weight_mode_label.setStyleSheet("color: #888;")
        elif using_evolved:
            self.weight_mode_label.setText("进化后")
            self.weight_mode_label.setStyleSheet("color: #089981; font-weight: bold;")
        else:
            self.weight_mode_label.setText("进化前")
            self.weight_mode_label.setStyleSheet("color: #888;")

    def update_match_preview(self, fp: str, similarity: float, fp_status: str = "", 
                             prototype_confidence: float = 0.0):
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
            # 显示置信度（如果有）
            if prototype_confidence > 0:
                conf_level = ""
                if prototype_confidence >= 0.70:
                    conf_level = "高"
                elif prototype_confidence >= 0.50:
                    conf_level = "中"
                elif prototype_confidence >= 0.30:
                    conf_level = "低"
                else:
                    conf_level = "极低"
                sim_text = f"{sim_text} | 置信: {prototype_confidence:.0%}({conf_level})"
        if fp_status:
            sim_text = f"{sim_text} | {fp_status}"
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

    def update_kelly_position_display(self, kelly_position_pct: float):
        """更新左侧单次仓位显示（凯利动态仓位）"""
        import time
        if kelly_position_pct and kelly_position_pct > 0:
            self.position_size_hint_label.setText(f"{kelly_position_pct:.1%}")
            self.kelly_formula_badge.show()
            if kelly_position_pct >= 0.30:
                color = "#00E676"
            elif kelly_position_pct >= 0.15:
                color = "#FFD700"
            else:
                color = "#9ad1ff"
            self.position_size_hint_label.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 13px;"
            )
            self._kelly_last_update_ts = time.time()
        else:
            self.position_size_hint_label.setText("50%")
            self.position_size_hint_label.setStyleSheet("color: #9ad1ff; font-weight: bold; font-size: 13px;")
            self.kelly_formula_badge.hide()

    def _update_kelly_heartbeat(self):
        """更新凯利心跳灯"""
        import time
        elapsed = time.time() - self._kelly_last_update_ts
        self._kelly_blink_state = not self._kelly_blink_state
        if elapsed > 3.0:
            self.kelly_heartbeat_label.setStyleSheet("color: #666; font-size: 12px;")
        else:
            color = "#00E676" if self._kelly_blink_state else "#0a5c33"
            self.kelly_heartbeat_label.setStyleSheet(f"color: {color}; font-size: 12px;")


class PaperTradingStatusPanel(QtWidgets.QWidget):
    """模拟交易状态面板（右侧）"""
    
    # 信号
    save_profitable_requested = QtCore.pyqtSignal()  # 保存盈利模板
    delete_losing_requested = QtCore.pyqtSignal()    # 删除亏损模板
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 心跳监控
        self._heartbeats = {}  # {模块名: 最后更新时间}
        self._heartbeat_indicators = {}  # {模块名: QLabel}
        self._heartbeat_timer = QtCore.QTimer()
        self._heartbeat_timer.timeout.connect(self._update_heartbeats)
        self._heartbeat_timer.start(500)  # 每500ms检查一次
        self._heartbeat_blink_state = False
        
        self._init_ui()
    
    def _init_ui(self):
        self.setMinimumWidth(280)  # 最小宽度，可与左侧分隔条拖拽拉宽
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)
        
        # ══════ 创建标签页容器 ══════
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                background-color: {UI_CONFIG['THEME_SURFACE']};
                margin-top: -1px;
            }}
            QTabBar::tab {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #252525);
                color: #aaa;
                padding: 10px 16px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 65px;
                margin-right: 2px;
                font-weight: 500;
                font-size: 12px;
            }}
            QTabBar::tab:selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {UI_CONFIG['THEME_ACCENT']}, stop:1 #006699);
                color: white;
                border-color: {UI_CONFIG['THEME_ACCENT']};
                font-weight: bold;
                padding-bottom: 12px;
            }}
            QTabBar::tab:hover:!selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #2d2d2d);
                color: #ddd;
            }}
            QTabBar::tab:first {{
                margin-left: 0px;
            }}
        """)
        
        # ══════ Tab 1: 持仓（含委托单、账户设置与统计） ══════
        self._create_position_tab()
        
        # ══════ Tab 2: 精品 ══════
        self._create_signal_mode_tab()

        # ══════ Tab 3: 匹配 ══════
        self._create_matching_tab()

        # ══════ Tab 4: 委托单 ══════
        self._create_pending_tab()

        # ══════ Tab 5: 推理 ══════
        self._create_monitoring_tab()
        
        # ══════ Tab 6: 日志 ══════
        self._create_log_tab()
        
        layout.addWidget(self.tabs)
    
    def _create_signal_mode_tab(self):
        """创建精品信号模式监控标签页"""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(10)
        
        # 标题
        title_label = QtWidgets.QLabel("💎 精品信号匹配")
        title_label.setStyleSheet(f"color: {UI_CONFIG['THEME_ACCENT']}; font-weight: bold; font-size: 14px;")
        tab_layout.addWidget(title_label)
        
        # 市场状态卡片
        state_card = QtWidgets.QWidget()
        state_card.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)
        state_layout = QtWidgets.QFormLayout(state_card)
        state_layout.setContentsMargins(15, 12, 15, 12)
        state_layout.setSpacing(10)
        
        self.sm_market_state_label = QtWidgets.QLabel("-")
        self.sm_market_state_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #aaa;")
        state_layout.addRow("当前市场状态:", self.sm_market_state_label)
        
        self.sm_today_count_label = QtWidgets.QLabel("0 次")
        self.sm_today_count_label.setStyleSheet("font-size: 14px; color: #ccc;")
        state_layout.addRow("今日触发次数:", self.sm_today_count_label)
        
        self.sm_pool_status_label = QtWidgets.QLabel("正在检查...")
        self.sm_pool_status_label.setStyleSheet("font-size: 12px; color: #888;")
        state_layout.addRow("策略池状态:", self.sm_pool_status_label)
        
        tab_layout.addWidget(state_card)
        
        # 最新触发组合卡片
        trigger_card = QtWidgets.QGroupBox("最新触发组合")
        trigger_card.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        trigger_layout = QtWidgets.QVBoxLayout(trigger_card)
        
        self.sm_trigger_info_label = QtWidgets.QLabel("等待信号触发...")
        self.sm_trigger_info_label.setWordWrap(True)
        self.sm_trigger_info_label.setStyleSheet("font-size: 13px; color: #bbb; line-height: 1.5; padding: 5px;")
        trigger_layout.addWidget(self.sm_trigger_info_label)
        
        tab_layout.addWidget(trigger_card)

        # 精品池明细卡片（三状态分组，含精品+高频双层）
        pool_card = QtWidgets.QGroupBox("精品策略池（按市场状态）— 精品(金色) + 高频(青色)")
        pool_card.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        pool_vbox = QtWidgets.QVBoxLayout(pool_card)
        pool_vbox.setSpacing(0)
        pool_vbox.setContentsMargins(4, 4, 4, 4)

        self.sm_all_states_text = QtWidgets.QTextEdit()
        self.sm_all_states_text.setReadOnly(True)
        self.sm_all_states_text.setStyleSheet(
            "background-color:#1a1a1a; color:#ccc; border:none;"
        )
        pool_vbox.addWidget(self.sm_all_states_text)

        tab_layout.addWidget(pool_card, 1)   # stretch=1 让明细区域占满剩余空间
        
        self.tabs.addTab(tab, "精品")

    def update_signal_mode_info(self, info: dict):
        """更新精品信号模式状态面板（由 main_window 状态回调调用）"""
        market_state = info.get("state", "-") if info else "-"
        today_count  = info.get("today_count", 0) if info else 0
        triggered_keys = set(info.get("triggered_keys", [])) if info else set()

        # 引擎提供的当前状态已注解精品池（含 matched/unmatched 条件信息）
        engine_long_pool  = info.get("long_pool",  []) if info else []
        engine_short_pool = info.get("short_pool", []) if info else []

        # === 始终从 signal_store 读取三个状态的完整精品池 ===
        _ALL_STATES = ["多头趋势", "空头趋势", "震荡市"]
        all_state_pools: dict = {}   # state -> {"long": [...], "short": [...]}
        pool_total = 0
        try:
            from core import signal_store
            for st in _ALL_STATES:
                lp = signal_store.get_premium_pool(state=st, direction="long")
                sp = signal_store.get_premium_pool(state=st, direction="short")
                # 若引擎正在运行且当前状态匹配，用已注解版本替换（保留绿/红条件信息）
                if st == market_state and engine_long_pool:
                    lp = engine_long_pool
                if st == market_state and engine_short_pool:
                    sp = engine_short_pool
                all_state_pools[st] = {"long": lp, "short": sp}
                pool_total += len(lp) + len(sp)
        except Exception:
            pass

        _engine_stopped = (not info) or market_state == "-"

        # ── 市场状态标签 ──
        state_color = "#888"
        if "多头" in market_state:
            state_color = "#089981"
        elif "空头" in market_state:
            state_color = "#f23645"
        elif "震荡" in market_state:
            state_color = "#FFB74D"

        self.sm_market_state_label.setText(market_state if market_state != "-" else "等待引擎启动")
        self.sm_market_state_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {state_color};"
        )
        self.sm_today_count_label.setText(f"{today_count} 次")

        # ── 触发组合卡片 ──
        if info and info.get("combo_key"):
            conditions   = info.get("conditions", [])
            direction    = info.get("direction", "long")
            score        = info.get("score", 0.0)
            trigger_time = info.get("time", "-")
            try:
                from core.signal_utils import _format_conditions
                cond_desc = _format_conditions(conditions, direction)
            except Exception:
                cond_desc = " & ".join(conditions[:3])
            dir_color = '#089981' if direction == 'long' else '#f23645'
            self.sm_trigger_info_label.setText(
                f"<b>方向:</b> <span style='color:{dir_color}'>"
                f"{'做多' if direction == 'long' else '做空'}</span><br>"
                f"<b>条件:</b> {cond_desc}<br>"
                f"<b>评分:</b> {score:.1f}  <b>时间:</b> {trigger_time}"
            )
        elif info and info.get("warning"):
            self.sm_trigger_info_label.setText(
                f"<span style='color:#f23645;'>{info['warning']}</span>"
            )
        else:
            self.sm_trigger_info_label.setText(
                "<span style='color:#666;'>等待信号触发...</span>"
            )

        # ── 策略池状态标签 ──
        if pool_total == 0:
            self.sm_pool_status_label.setText("⚠ 策略池为空，请先完成信号分析")
            self.sm_pool_status_label.setStyleSheet(
                "font-size: 12px; color: #f23645; font-weight: bold;"
            )
        elif _engine_stopped:
            self.sm_pool_status_label.setText(
                f"✅ 已加载策略池: 共 {pool_total} 个策略（3状态×多空 Top6）— 引擎待启动"
            )
            self.sm_pool_status_label.setStyleSheet("font-size: 12px; color: #FFB74D;")
        else:
            cur_l = len(engine_long_pool)
            cur_s = len(engine_short_pool)
            self.sm_pool_status_label.setText(
                f"当前[{market_state}]: 做多{cur_l} / 做空{cur_s}  (总{pool_total}/36)"
            )
            self.sm_pool_status_label.setStyleSheet("font-size: 12px; color: #089981;")

        # ── 三状态精品池明细 ──
        self.sm_all_states_text.setHtml(
            self._format_all_states_html(all_state_pools, market_state, triggered_keys)
        )

    def _format_all_states_html(
        self,
        all_state_pools: dict,   # state -> {"long": [...], "short": [...]}
        current_state: str,
        triggered_keys: set,
    ) -> str:
        """
        指标×状态 表格视图（精品+高频双层颜色区分）。
        - 行 = 指标类别（布林位置、偏离MA5、ATR波动率…）
        - 列 = 3状态 × 做多/做空 = 6列
        - 单元格 = (1/2/3) + 亮灯/灭灯
          精品策略编号用金色，高频策略编号用青色
        - 当前状态列高亮边框
        - 表格下方：全亮策略摘要（含层级标签）
        """
        # ── 层级颜色 ──────────────────────────────────────────────
        TIER_COLOR_ELITE = "#D9B36A"  # 精品 - 金色
        TIER_COLOR_FREQ  = "#00CED1"  # 高频 - 青色
        TIER_BG_ELITE    = "#2A2520"
        TIER_BG_FREQ     = "#1E2A2A"

        def _tier_color(tier: str) -> str:
            return TIER_COLOR_FREQ if tier == "高频" else TIER_COLOR_ELITE

        # ── 指标顺序与标签 ──────────────────────────────────────────
        INDICATOR_ORDER = [
            "boll_pos", "close_vs_ma5", "atr_ratio", "vol_ratio",
            "rsi", "k", "j", "lower_shd", "upper_shd",
            "consec_bear", "consec_bull",
        ]
        INDICATOR_LABELS = {
            "boll_pos":     "布林位置",
            "close_vs_ma5": "偏离MA5",
            "atr_ratio":    "ATR波动率",
            "vol_ratio":    "量比",
            "rsi":          "RSI",
            "k":            "KDJ-K",
            "j":            "KDJ-J",
            "lower_shd":    "下影线/实体",
            "upper_shd":    "上影线/实体",
            "consec_bear":  "连续阴线",
            "consec_bull":  "连续阳线",
        }
        STATES = ["多头趋势", "空头趋势", "震荡市"]
        STATE_LABELS = {"多头趋势": "📈多头趋势", "空头趋势": "📉空头趋势", "震荡市": "↔震荡市"}
        STATE_COLORS = {"多头趋势": "#089981", "空头趋势": "#f23645", "震荡市": "#FFB74D"}
        COLS = [
            ("多头趋势", "long"), ("多头趋势", "short"),
            ("空头趋势", "long"), ("空头趋势", "short"),
            ("震荡市",   "long"), ("震荡市",   "short"),
        ]

        def _get_base(cond: str) -> str:
            for s in ("_loose", "_strict"):
                if cond.endswith(s):
                    return cond[:-len(s)]
            return cond

        # ── 构建每列的指标倒排索引 ──────────────────────────────────
        # col_map[(state, dir)] = {base: [(strategy_idx, is_matched_or_None, tier), ...]}
        col_map: dict = {}
        for state, direction in COLS:
            pools = all_state_pools.get(state, {})
            pool  = pools.get(direction, [])
            is_cur = (state == current_state)
            idx_map: dict = {}
            for idx, item in enumerate(pool, 1):
                conditions = item.get("conditions", []) or []
                matched    = set(item.get("matched_conditions",   []) or [])
                unmatched  = set(item.get("unmatched_conditions", []) or [])
                has_ann    = bool(matched or unmatched)
                tier       = item.get("tier", "精品")
                for cond in conditions:
                    base = _get_base(cond)
                    if base not in idx_map:
                        idx_map[base] = []
                    if is_cur and has_ann:
                        is_matched = cond in matched
                    else:
                        is_matched = None
                    idx_map[base].append((idx, is_matched, tier))
            col_map[(state, direction)] = idx_map

        # ── 构建每列的策略 tier 映射（用于表头图例） ──────────────
        col_tier_map: dict = {}   # (state, direction) -> {idx: tier}
        for state, direction in COLS:
            pools = all_state_pools.get(state, {})
            pool  = pools.get(direction, [])
            col_tier_map[(state, direction)] = {
                idx: item.get("tier", "精品") for idx, item in enumerate(pool, 1)
            }

        # ── 收集所有出现过的指标 ──────────────────────────────────
        all_used: set = set()
        for v in col_map.values():
            all_used.update(v.keys())
        if not all_used:
            return ("<div style='color:#555;padding:20px;text-align:center;'>"
                    "无精品策略，请先完成信号分析</div>")

        ordered = [b for b in INDICATOR_ORDER if b in all_used]
        ordered += sorted(all_used - set(INDICATOR_ORDER))

        # ── 表格样式常量 ────────────────────────────────────────────
        TH  = ("padding:4px 6px;text-align:center;font-weight:bold;"
               "border:1px solid #2a2a2a;font-size:11px;")
        TD  = ("padding:3px 5px;text-align:left;"
               "border:1px solid #2a2a2a;font-size:11px;vertical-align:middle;")
        TDL = ("padding:3px 6px;text-align:left;"
               "border:1px solid #2a2a2a;font-size:11px;font-weight:bold;"
               "white-space:nowrap;background:#1c1c1c;color:#aaa;")

        # ── 图例 ────────────────────────────────────────────────────
        h = [
            f"<div style='margin-bottom:6px;font-size:11px;'>"
            f"<span style='color:{TIER_COLOR_ELITE};font-weight:bold;'>■ 精品策略</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{TIER_COLOR_FREQ};font-weight:bold;'>■ 高频策略</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#666;'>绿=当前K线满足 红=不满足</span>"
            f"</div>"
        ]
        h.append("<table style='width:100%;border-collapse:collapse;'>")

        # ── 表头行1：状态（每2列合并） ────────────────────────────
        h.append("<tr>")
        h.append(f"<th style='{TH}background:#111;color:#444;'>指标</th>")
        for state in STATES:
            color   = STATE_COLORS[state]
            is_cur  = (state == current_state)
            bdr     = f"border-bottom:2px solid {color};" if is_cur else ""
            bg      = "#1d2424" if is_cur else "#181818"
            active  = "▶ " if is_cur else ""
            h.append(f"<th colspan='2' style='{TH}{bdr}background:{bg};"
                     f"color:{color};'>{active}{STATE_LABELS[state]}</th>")
        h.append("</tr>")

        # ── 表头行2：做多/做空 ────────────────────────────────────
        h.append("<tr>")
        h.append(f"<th style='{TH}background:#111;color:#444;'></th>")
        for state in STATES:
            is_cur = (state == current_state)
            for direction, dir_label in [("long", "做多"), ("short", "做空")]:
                dir_color = "#089981" if direction == "long" else "#f23645"
                bg = "#1a221a" if (is_cur and direction == "long") else \
                     "#221a1a" if (is_cur and direction == "short") else "#181818"
                h.append(f"<th style='{TH}background:{bg};color:{dir_color};'>"
                         f"{dir_label}</th>")
        h.append("</tr>")

        # ── 数据行：每个指标一行 ──────────────────────────────────
        for base in ordered:
            ind_label = INDICATOR_LABELS.get(base, base)
            h.append("<tr>")
            h.append(f"<td style='{TDL}'>{ind_label}</td>")

            for state, direction in COLS:
                entries = col_map.get((state, direction), {}).get(base, [])
                is_cur  = (state == current_state)

                if not entries:
                    h.append(f"<td style='{TD}background:#151515;'></td>")
                    continue

                ms = [m for _, m, _ in entries if m is not None]

                if not ms:
                    bg = "#181818"
                elif all(ms):
                    bg = "#1a2a1a"
                elif any(ms):
                    bg = "#252015"
                else:
                    bg = "#261717"

                # 每个策略编号按 tier 着色
                num_parts = []
                for idx, is_matched, tier in entries:
                    tc = _tier_color(tier)
                    if ms:
                        if is_matched:
                            c = "#4CAF50"
                        elif is_matched is False:
                            c = "#f23645"
                        else:
                            c = tc
                    else:
                        c = tc
                    num_parts.append(f"<span style='color:{c};'>{idx}</span>")

                nums_html = f"<span>({'/'.join(num_parts)})</span>"

                # 状态后缀
                if not ms:
                    suffix = ""
                elif all(ms):
                    suffix = " <b style='color:#4CAF50;'>亮灯</b>"
                elif any(ms):
                    suffix = " <b style='color:#FFB74D;'>部分</b>"
                else:
                    suffix = " <b style='color:#f23645;'>未满足</b>"

                h.append(f"<td style='{TD}background:{bg};'>"
                         f"{nums_html}{suffix}</td>")
            h.append("</tr>")

        h.append("</table>")

        # ── 全亮策略摘要 ──────────────────────────────────────────
        summary_parts = []
        if current_state in STATES:
            for direction, dir_label, dir_color in [
                ("long",  "做多", "#089981"),
                ("short", "做空", "#f23645"),
            ]:
                pool = all_state_pools.get(current_state, {}).get(direction, [])
                for idx, item in enumerate(pool, 1):
                    conditions = item.get("conditions", []) or []
                    matched    = set(item.get("matched_conditions", []) or [])
                    unmatched  = set(item.get("unmatched_conditions", []) or [])
                    tier       = item.get("tier", "精品")
                    tier_color = _tier_color(tier)
                    tier_badge = (
                        f"<span style='background:{tier_color};color:#000;font-size:9px;"
                        f"padding:1px 4px;border-radius:2px;font-weight:bold;'>{tier}</span>"
                    )
                    if not conditions:
                        continue
                    is_triggered = item.get("combo_key") in triggered_keys
                    all_lit = bool(matched) and len(matched) == len(conditions)
                    if is_triggered:
                        badge = ("<span style='background:#00C8D4;color:#000;font-size:10px;"
                                 "padding:1px 5px;border-radius:3px;font-weight:bold;'>●开仓</span>")
                        summary_parts.append(
                            f"<div style='margin:2px 0;padding:3px 8px;"
                            f"background:#0d2a2a;border-left:3px solid #00C8D4;"
                            f"border-radius:2px;font-size:11px;'>"
                            f"{tier_badge}&nbsp;"
                            f"<span style='color:{dir_color};font-weight:bold;'>"
                            f"[{dir_label}策略{idx}]</span>&nbsp;{badge}&nbsp;"
                            f"<span style='color:#00C8D4;'>已触发开仓</span>"
                            f"</div>"
                        )
                    elif all_lit:
                        match_cnt = len(matched)
                        summary_parts.append(
                            f"<div style='margin:2px 0;padding:3px 8px;"
                            f"background:#1a2a1a;border-left:3px solid #4CAF50;"
                            f"border-radius:2px;font-size:11px;'>"
                            f"{tier_badge}&nbsp;"
                            f"<span style='color:{dir_color};font-weight:bold;'>"
                            f"[{dir_label}策略{idx}]</span>&nbsp;"
                            f"<span style='color:#4CAF50;font-weight:bold;'>全亮 {match_cnt}/{len(conditions)}</span>"
                            f"&nbsp;<span style='color:#666;'>胜率{item.get('state_rate',0):.0%}"
                            f" 评分{item.get('score',0):.1f}</span>"
                            f"</div>"
                        )
                    else:
                        match_cnt = len(matched)
                        if match_cnt > 0:
                            summary_parts.append(
                                f"<div style='margin:2px 0;padding:3px 8px;"
                                f"background:#1a1a1a;border-left:3px solid #333;"
                                f"border-radius:2px;font-size:11px;'>"
                                f"{tier_badge}&nbsp;"
                                f"<span style='color:#555;'>[{dir_label}策略{idx}]</span>&nbsp;"
                                f"<span style='color:#FFB74D;'>{match_cnt}/{len(conditions)} 条件满足</span>"
                                f"</div>"
                            )
                        else:
                            summary_parts.append(
                                f"<div style='margin:2px 0;padding:3px 8px;"
                                f"background:#1f1515;border-left:3px solid #f23645;"
                                f"border-radius:2px;font-size:11px;'>"
                                f"{tier_badge}&nbsp;"
                                f"<span style='color:#777;'>[{dir_label}策略{idx}]</span>&nbsp;"
                                f"<span style='color:#f23645;'>0/{len(conditions)} 条件满足</span>"
                                f"</div>"
                            )

        if summary_parts:
            cur_color = STATE_COLORS.get(current_state, "#888")
            h.append(
                f"<div style='margin-top:10px;padding:6px;border:1px solid #2a2a2a;"
                f"border-radius:4px;background:#181818;'>"
                f"<div style='color:{cur_color};font-weight:bold;font-size:11px;"
                f"margin-bottom:4px;'>▶ {current_state} 当前触发情况</div>"
            )
            h.extend(summary_parts)
            h.append("</div>")
        else:
            h.append(
                "<div style='margin-top:10px;padding:8px;border:1px solid #2a2a2a;"
                "border-radius:4px;background:#181818;color:#666;font-size:11px;'>"
                "当前状态暂无可统计的触发明细</div>"
            )

        return "".join(h)

    def _create_position_tab(self):
        """创建持仓标签页"""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)
        
        # 使用滚动区域，容纳更多内容
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #2a2a2a;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        content = QtWidgets.QWidget()
        tab_layout_inner = QtWidgets.QVBoxLayout(content)
        tab_layout_inner.setContentsMargins(14, 14, 14, 14)
        tab_layout_inner.setSpacing(12)
        
        # 持仓信息表单
        position_form = QtWidgets.QFormLayout()
        position_form.setSpacing(10)
        position_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        
        # 方向标签 - 突出显示
        self.position_side_label = QtWidgets.QLabel("-")
        self.position_side_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        position_form.addRow("方向:", self.position_side_label)
        
        # 数量
        self.position_qty_label = QtWidgets.QLabel("-")
        self.position_qty_label.setStyleSheet("color: #ccc;")
        position_form.addRow("数量:", self.position_qty_label)
        
        # 保证金占用
        self.position_margin_label = QtWidgets.QLabel("-")
        self.position_margin_label.setStyleSheet("color: #9ad1ff;")
        position_form.addRow("保证金:", self.position_margin_label)
        
        # 杠杆（含自适应亮灯：绿=在用，灰=未用）
        self.position_leverage_label = QtWidgets.QLabel("-")
        self.position_leverage_label.setStyleSheet("color: #FFB74D; font-weight: bold;")
        self.adaptive_leverage_lamp = QtWidgets.QLabel("●")
        self.adaptive_leverage_lamp.setStyleSheet("color: #666; font-size: 12px;")
        self.adaptive_leverage_lamp.setToolTip(
            "亮灯=杠杆参与「凯利仓位学习」，会随表现与回撤自动调整；\n"
            "灰=未启用自适应（无凯利适配器）。"
        )
        leverage_row = QtWidgets.QHBoxLayout()
        leverage_row.setContentsMargins(0, 0, 0, 0)
        leverage_row.setSpacing(6)
        leverage_row.addWidget(self.position_leverage_label)
        leverage_row.addWidget(self.adaptive_leverage_lamp)
        leverage_row.addStretch()
        position_form.addRow("杠杆:", leverage_row)
        
        # 入场价
        self.position_entry_label = QtWidgets.QLabel("-")
        self.position_entry_label.setStyleSheet("color: #ccc;")
        position_form.addRow("入场价:", self.position_entry_label)
        
        # 当前价
        self.position_current_label = QtWidgets.QLabel("-")
        self.position_current_label.setStyleSheet("color: #FFD54F;")
        position_form.addRow("当前价:", self.position_current_label)
        
        # 分隔线
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator1.setStyleSheet("background-color: #3a3a3a;")
        tab_layout_inner.addLayout(position_form)
        tab_layout_inner.addWidget(separator1)
        
        # 盈亏信息（强调显示）
        pnl_form = QtWidgets.QFormLayout()
        pnl_form.setSpacing(8)
        pnl_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.position_pnl_label = QtWidgets.QLabel("-")
        self.position_pnl_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        pnl_form.addRow("浮动盈亏:", self.position_pnl_label)
        
        self.position_pnl_pct_label = QtWidgets.QLabel("-")
        self.position_pnl_pct_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        pnl_form.addRow("收益率:", self.position_pnl_pct_label)
        
        tab_layout_inner.addLayout(pnl_form)
        
        # 分隔线
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: #3a3a3a;")
        tab_layout_inner.addWidget(separator2)
        
        # 追踪状态（醒目显示）
        tracking_form = QtWidgets.QFormLayout()
        tracking_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.tracking_status_label = QtWidgets.QLabel("-")
        self.tracking_status_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        tracking_form.addRow("追踪状态:", self.tracking_status_label)
        
        tab_layout_inner.addLayout(tracking_form)
        
        # 分隔线
        separator3 = QtWidgets.QFrame()
        separator3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator3.setStyleSheet("background-color: #3a3a3a;")
        tab_layout_inner.addWidget(separator3)
        
        # 账户设置与统计（从左侧控制面板移入）
        self._account_group_container = QtWidgets.QWidget()
        self._account_group_layout = QtWidgets.QVBoxLayout(self._account_group_container)
        self._account_group_layout.setContentsMargins(0, 0, 0, 0)
        self._account_group_layout.setSpacing(6)
        tab_layout_inner.addWidget(self._account_group_container)
        
        # 分隔线
        separator4 = QtWidgets.QFrame()
        separator4.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator4.setStyleSheet("background-color: #3a3a3a;")
        tab_layout_inner.addWidget(separator4)
        
        # 委托单监控区（合并到持仓页）
        pending_section = self._build_pending_orders_section()
        tab_layout_inner.addWidget(pending_section)
        
        tab_layout_inner.addStretch()
        
        scroll_area.setWidget(content)
        tab_layout.addWidget(scroll_area)
        
        self.tabs.addTab(tab, "💼 持仓")

    def attach_account_group(self, account_group: QtWidgets.QGroupBox):
        """把账户设置/统计区放入持仓页"""
        if not hasattr(self, "_account_group_layout"):
            return
        # 解除旧父级，重新挂载
        account_group.setParent(self._account_group_container)
        self._account_group_layout.addWidget(account_group)

    def _build_pending_orders_section(self) -> QtWidgets.QWidget:
        """构建委托单监控区块（复用）"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        title = QtWidgets.QLabel("📋 委托单")
        title.setStyleSheet("color: #e0e0e0; font-size: 12px; font-weight: bold;")
        layout.addWidget(title)
        
        # 提示标签（美化）
        self.pending_orders_hint_label = QtWidgets.QLabel("当前无挂单")
        self.pending_orders_hint_label.setStyleSheet("""
            color: #888;
            font-size: 11px;
            padding: 4px 6px;
            background-color: #252526;
            border: 1px solid #3a3a3a;
            border-radius: 4px;
        """)
        layout.addWidget(self.pending_orders_hint_label)
        
        # 委托单表格（美化）
        self.pending_orders_table = QtWidgets.QTableWidget()
        self.pending_orders_table.setColumnCount(6)
        self.pending_orders_table.setHorizontalHeaderLabels(["方向", "挂单价", "数量", "状态", "原型", "TP/SL%"])
        self.pending_orders_table.verticalHeader().setVisible(False)
        self.pending_orders_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pending_orders_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.pending_orders_table.setAlternatingRowColors(True)
        self.pending_orders_table.setMinimumHeight(140)
        self.pending_orders_table.horizontalHeader().setStretchLastSection(True)
        self.pending_orders_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                gridline-color: #2a2a2a;
                color: #d0d0d0;
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: #252526;
                color: #bdbdbd;
                border: none;
                padding: 6px 8px;
                font-weight: bold;
                font-size: 10px;
            }}
            QTableWidget::item {{
                padding: 4px 6px;
                border-bottom: 1px solid #2a2a2a;
            }}
            QTableWidget::item:selected {{
                background-color: #2d2d2d;
            }}
        """)
        layout.addWidget(self.pending_orders_table)
        
        return container
    
    def _create_matching_tab(self):
        """创建匹配状态标签页"""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        
        # 添加滚动区域（匹配标签页内容较多）
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #2a2a2a;
                width: 10px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #555;
                border-radius: 5px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #666;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        scroll_content = QtWidgets.QWidget()
        market_layout = QtWidgets.QFormLayout(scroll_content)
        
        # 复制原 market_group 的内容
        # 市场状态 + 心跳指示器
        market_regime_container = QtWidgets.QWidget()
        market_regime_h_layout = QtWidgets.QHBoxLayout(market_regime_container)
        market_regime_h_layout.setContentsMargins(0, 0, 0, 0)
        market_regime_h_layout.setSpacing(5)
        
        self.market_regime_label = QtWidgets.QLabel("未知")
        market_regime_h_layout.addWidget(self.market_regime_label)
        
        self._heartbeat_indicators["market"] = self._create_heartbeat_indicator()
        market_regime_h_layout.addWidget(self._heartbeat_indicators["market"])
        market_regime_h_layout.addStretch()
        
        market_layout.addRow("市场状态:", market_regime_container)
        
        self.swing_points_label = QtWidgets.QLabel(f"0 / {MARKET_REGIME_CONFIG.get('LOOKBACK_SWINGS', 4)}")
        self.swing_points_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        self.swing_points_label.setToolTip(f"已检测到的摆动点数量 / 激活分类所需的最少点数({MARKET_REGIME_CONFIG.get('LOOKBACK_SWINGS', 4)}: 3高+3低)")
        market_layout.addRow("摆动点检测:", self.swing_points_label)
        
        # 指纹匹配 + 心跳指示器
        fingerprint_container = QtWidgets.QWidget()
        fingerprint_h_layout = QtWidgets.QHBoxLayout(fingerprint_container)
        fingerprint_h_layout.setContentsMargins(0, 0, 0, 0)
        fingerprint_h_layout.setSpacing(5)
        
        self.fingerprint_status_label = QtWidgets.QLabel("待匹配")
        fingerprint_h_layout.addWidget(self.fingerprint_status_label)
        
        self._heartbeat_indicators["fingerprint"] = self._create_heartbeat_indicator()
        fingerprint_h_layout.addWidget(self._heartbeat_indicators["fingerprint"])
        fingerprint_h_layout.addStretch()
        
        market_layout.addRow("指纹匹配:", fingerprint_container)

        self.matched_fingerprint_label = QtWidgets.QLabel("-")
        self.matched_fingerprint_label.setWordWrap(True)
        self.matched_fingerprint_label.setMinimumWidth(120)
        self.matched_fingerprint_label.setStyleSheet("color: #9fd6ff; font-weight: bold; font-size: 12px;")
        market_layout.addRow("匹配原型:", self.matched_fingerprint_label)
        
        # 贝叶斯胜率（原型旁边显示）
        self.bayesian_win_rate_label = QtWidgets.QLabel("-")
        self.bayesian_win_rate_label.setStyleSheet("color: #FFD700; font-weight: bold; font-size: 13px;")
        self.bayesian_win_rate_label.setToolTip("贝叶斯预测的胜率（Thompson Sampling采样值）")
        market_layout.addRow("贝叶斯胜率:", self.bayesian_win_rate_label)

        # 实时配合度 + 开仓阈值 + 距离
        self.matched_similarity_label = QtWidgets.QLabel("-")
        self.matched_similarity_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        market_layout.addRow("实时配合度:", self.matched_similarity_label)
        
        # 【指纹3D图】多维相似度分解显示
        self.multi_sim_container = QtWidgets.QWidget()
        multi_sim_layout = QtWidgets.QHBoxLayout(self.multi_sim_container)
        multi_sim_layout.setContentsMargins(0, 2, 0, 2)
        multi_sim_layout.setSpacing(4)
        
        # 方向相似度（余弦）
        self.cos_sim_badge = QtWidgets.QLabel("方向: -")
        self.cos_sim_badge.setStyleSheet(self._similarity_badge_style("#4A90D9"))
        self.cos_sim_badge.setToolTip("方向相似度（余弦）\n衡量特征变化方向是否一致")
        multi_sim_layout.addWidget(self.cos_sim_badge)
        
        # 距离相似度（欧氏）
        self.euc_sim_badge = QtWidgets.QLabel("距离: -")
        self.euc_sim_badge.setStyleSheet(self._similarity_badge_style("#7B68EE"))
        self.euc_sim_badge.setToolTip("距离相似度（欧氏）\n衡量特征数值是否接近")
        multi_sim_layout.addWidget(self.euc_sim_badge)
        
        # 形态相似度（DTW）
        self.dtw_sim_badge = QtWidgets.QLabel("形态: -")
        self.dtw_sim_badge.setStyleSheet(self._similarity_badge_style("#20B2AA"))
        self.dtw_sim_badge.setToolTip("形态相似度（DTW）\n衡量时间序列形态是否匹配")
        multi_sim_layout.addWidget(self.dtw_sim_badge)
        
        multi_sim_layout.addStretch()
        market_layout.addRow("相似度分解:", self.multi_sim_container)
        
        # 原型置信度
        self.confidence_container = QtWidgets.QWidget()
        confidence_h_layout = QtWidgets.QHBoxLayout(self.confidence_container)
        confidence_h_layout.setContentsMargins(0, 0, 0, 0)
        confidence_h_layout.setSpacing(5)
        
        self.confidence_label = QtWidgets.QLabel("-")
        self.confidence_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        confidence_h_layout.addWidget(self.confidence_label)
        
        self.confidence_level_badge = QtWidgets.QLabel("")
        self.confidence_level_badge.setStyleSheet("""
            QLabel {
                color: #888;
                background-color: rgba(136, 136, 136, 0.15);
                border: 1px solid #888;
                border-radius: 3px;
                padding: 0px 4px;
                font-size: 9px;
            }
        """)
        self.confidence_level_badge.hide()
        confidence_h_layout.addWidget(self.confidence_level_badge)
        confidence_h_layout.addStretch()
        
        market_layout.addRow("原型置信度:", self.confidence_container)
        
        self.entry_threshold_label = QtWidgets.QLabel("-")
        self.entry_threshold_label.setStyleSheet("color: #888;")
        market_layout.addRow("开仓阈值:", self.entry_threshold_label)
        
        self.distance_to_entry_label = QtWidgets.QLabel("-")
        self.distance_to_entry_label.setStyleSheet("font-weight: bold;")
        market_layout.addRow("距离开仓:", self.distance_to_entry_label)
        
        self.position_score_label = QtWidgets.QLabel("-")
        self.position_score_label.setStyleSheet("font-weight: bold;")
        self.position_score_label.setToolTip("空间位置评分(-100~+100)，越高表示当前方向越有利")
        market_layout.addRow("空间位置评分:", self.position_score_label)
        
        self.reason_label = QtWidgets.QLabel("-")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("color: #bbb;")
        market_layout.addRow("决策说明:", self.reason_label)
        
        # 动能门控 (Aim/Exit) + 心跳指示器
        indicators_main_container = QtWidgets.QWidget()
        indicators_main_layout = QtWidgets.QHBoxLayout(indicators_main_container)
        indicators_main_layout.setContentsMargins(0, 0, 0, 0)
        indicators_main_layout.setSpacing(5)
        
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
        
        indicators_main_layout.addWidget(self.indicators_container)
        self._heartbeat_indicators["gate"] = self._create_heartbeat_indicator()
        indicators_main_layout.addWidget(self._heartbeat_indicators["gate"])
        indicators_main_layout.addStretch()
        
        market_layout.addRow("动能门控:", indicators_main_container)

        # ══════════════════════════════════════════════════════════════
        # 开仓条件总览 / 平仓条件总览  (overview cards)
        # ══════════════════════════════════════════════════════════════
        self._build_entry_overview_card(market_layout)
        self._build_exit_overview_card(market_layout)

        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)
        
        self.tabs.addTab(tab, "🎯 匹配")
    
    # ─────────────────────────────────────────────────────────
    #  Entry / Exit overview card builders
    # ─────────────────────────────────────────────────────────
    def _build_entry_overview_card(self, parent_layout: QtWidgets.QFormLayout):
        """开仓条件总览卡片 - 8行 x 5列 (条件 / 正常阈值 / 冷启动阈值 / 当前值 / 状态)"""

        # -- outer frame (dark card with rounded corners and subtle border) --
        card = QtWidgets.QFrame()
        card.setObjectName("entryCard")
        card.setStyleSheet("""
            QFrame#entryCard {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 8px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 6)
        card_layout.setSpacing(0)

        # -- header bar: green accent band + title + cold-start badge --
        header_widget = QtWidgets.QWidget()
        header_widget.setObjectName("entryHeader")
        header_widget.setStyleSheet("""
            QWidget#entryHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(8, 153, 129, 0.25), stop:1 #2d2d2d);
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-left: 3px solid #089981;
            }
        """)
        header_h = QtWidgets.QHBoxLayout(header_widget)
        header_h.setContentsMargins(10, 6, 10, 6)
        header_h.setSpacing(8)

        title = QtWidgets.QLabel("开仓条件总览")
        title.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 12px; background: transparent;")
        header_h.addWidget(title)
        header_h.addStretch()

        # Cold start mode indicator badge (prominent, with tooltip)
        self._cold_start_badge = QtWidgets.QLabel("正常模式")
        self._cold_start_badge.setToolTip(
            "当前匹配模式\n"
            "正常模式: 使用标准阈值进行匹配\n"
            "冷启动模式: 放宽阈值以增加初始交易频率"
        )
        self._cold_start_badge.setStyleSheet(self._cold_start_badge_style(False))
        header_h.addWidget(self._cold_start_badge)
        card_layout.addWidget(header_widget)

        # -- separator --
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #555; border: none;")
        card_layout.addWidget(sep)

        # -- grid container --
        grid_widget = QtWidgets.QWidget()
        grid_widget.setStyleSheet("background: transparent;")
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(0)
        grid.setContentsMargins(6, 0, 6, 2)

        # column headers with dark background
        col_headers = ["条件", "正常阈值", "冷启动阈值", "当前值", "状态"]
        self._entry_col_header_lbls: Dict[int, QtWidgets.QLabel] = {}
        for ci, text in enumerate(col_headers):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(
                "color: #999; font-size: 9px; font-weight: bold; "
                "background-color: #3a3a3a; padding: 4px 3px; "
                "border-bottom: 1px solid #555;"
            )
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, ci)
            self._entry_col_header_lbls[ci] = lbl

        # -- read thresholds from config --
        cos_normal = SIMILARITY_CONFIG.get("COSINE_MIN_THRESHOLD", 0.70)
        fusion_normal = SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.65)
        euc_normal = 0.35   # approximate normal threshold
        dtw_normal = 0.30   # approximate normal threshold

        cos_cold = COLD_START_CONFIG.get("THRESHOLDS", {}).get("cosine", 0.50)
        fusion_cold = COLD_START_CONFIG.get("THRESHOLDS", {}).get("fusion", 0.30)
        euc_cold = COLD_START_CONFIG.get("THRESHOLDS", {}).get("euclidean", 0.25)
        dtw_cold = COLD_START_CONFIG.get("THRESHOLDS", {}).get("dtw", 0.10)

        macd_slope = PAPER_TRADING_CONFIG.get("MACD_SLOPE_MIN", 0.003)
        bayes_min = PAPER_TRADING_CONFIG.get("BAYESIAN_MIN_WIN_RATE", 0.40)
        pos_long = PAPER_TRADING_CONFIG.get("POS_THRESHOLD_LONG", -30)

        # Row definitions: (name, normal_text, cold_text, key)
        entry_rows = [
            ("余弦相似度",  f"{cos_normal:.0%}",     f"{cos_cold:.0%}",     "cosine"),
            ("融合评分",    f"{fusion_normal:.0%}",   f"{fusion_cold:.0%}",  "fusion"),
            ("欧氏距离",    f"{euc_normal:.0%}",      f"{euc_cold:.0%}",     "euclidean"),
            ("DTW形态",     f"{dtw_normal:.0%}",      f"{dtw_cold:.0%}",     "dtw"),
            ("MACD趋势",   f"斜率≥{macd_slope}",     "跳过",                "macd"),
            ("KDJ指标",     "J≥D 或 K≥D",            "同上",                "kdj"),
            ("贝叶斯胜率",  f"≥{bayes_min:.0%}",      "同上",                "bayesian"),
            ("位置评分(多)", f"≥{pos_long}",           "同上",                "position"),
        ]

        self._entry_overview_labels: Dict[str, Dict[str, QtWidgets.QLabel]] = {}

        for ri, (name, normal_txt, cold_txt, key) in enumerate(entry_rows, start=1):
            row_labels: Dict[str, QtWidgets.QLabel] = {}
            # alternating row background
            row_bg = "rgba(58, 58, 58, 0.5)" if ri % 2 == 0 else "transparent"

            # col 0 - condition name
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(
                f"color: #ddd; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(name_lbl, ri, 0)

            # col 1 - normal threshold
            normal_lbl = QtWidgets.QLabel(normal_txt)
            normal_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            normal_lbl.setStyleSheet(
                f"color: #aaa; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(normal_lbl, ri, 1)
            row_labels["normal"] = normal_lbl

            # col 2 - cold start threshold
            cold_lbl = QtWidgets.QLabel(cold_txt)
            cold_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cold_lbl.setStyleSheet(
                f"color: #aaa; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(cold_lbl, ri, 2)
            row_labels["cold"] = cold_lbl

            # col 3 - realtime value (dynamic)
            rt_lbl = QtWidgets.QLabel("--")
            rt_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            rt_lbl.setStyleSheet(
                f"color: #666; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(rt_lbl, ri, 3)
            row_labels["realtime"] = rt_lbl

            # col 4 - status badge (dynamic)
            status_lbl = QtWidgets.QLabel("--")
            status_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            status_lbl.setStyleSheet(
                f"color: #666; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(status_lbl, ri, 4)
            row_labels["status"] = status_lbl

            # store row background for realtime updates
            row_labels["_row_bg"] = row_bg  # type: ignore[assignment]
            self._entry_overview_labels[key] = row_labels

        # column stretch
        grid.setColumnStretch(0, 3)  # name
        grid.setColumnStretch(1, 2)  # normal
        grid.setColumnStretch(2, 2)  # cold
        grid.setColumnStretch(3, 2)  # realtime
        grid.setColumnStretch(4, 1)  # status

        card_layout.addWidget(grid_widget)
        parent_layout.addRow(card)

    # ─── Status badge helper methods ────────────────────────
    @staticmethod
    def _cold_start_badge_style(active: bool) -> str:
        """Generate stylesheet for the cold start mode indicator badge."""
        if active:
            return (
                "QLabel {"
                "  color: #FF9800;"
                "  background-color: rgba(255, 152, 0, 0.18);"
                "  border: 1px solid #FF9800;"
                "  border-radius: 4px;"
                "  padding: 2px 10px;"
                "  font-size: 10px;"
                "  font-weight: bold;"
                "}"
            )
        return (
            "QLabel {"
            "  color: #4FC3F7;"
            "  background-color: rgba(79, 195, 247, 0.15);"
            "  border: 1px solid #4FC3F7;"
            "  border-radius: 4px;"
            "  padding: 2px 10px;"
            "  font-size: 10px;"
            "  font-weight: bold;"
            "}"
        )

    @staticmethod
    def _status_badge_pass() -> str:
        """Status badge stylesheet: PASS (green)."""
        return (
            "color: #089981; font-size: 11px; font-weight: bold; "
            "padding: 2px 4px; background: rgba(8,153,129,0.12); "
            "border-radius: 3px;"
        )

    @staticmethod
    def _status_badge_fail() -> str:
        """Status badge stylesheet: FAIL (red)."""
        return (
            "color: #f23645; font-size: 11px; font-weight: bold; "
            "padding: 2px 4px; background: rgba(242,54,69,0.12); "
            "border-radius: 3px;"
        )

    @staticmethod
    def _status_badge_near() -> str:
        """Status badge stylesheet: NEAR threshold (yellow)."""
        return (
            "color: #FFD54F; font-size: 11px; font-weight: bold; "
            "padding: 2px 4px; background: rgba(255,213,79,0.12); "
            "border-radius: 3px;"
        )

    @staticmethod
    def _status_badge_none() -> str:
        """Status badge stylesheet: no data (gray)."""
        return (
            "color: #666; font-size: 10px; "
            "padding: 2px 4px; background: transparent;"
        )

    def _build_exit_overview_card(self, parent_layout: QtWidgets.QFormLayout):
        """平仓条件总览卡片 - 条件 + 阈值描述"""

        card = QtWidgets.QFrame()
        card.setObjectName("exitCard")
        card.setStyleSheet("""
            QFrame#exitCard {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 8px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 6)
        card_layout.setSpacing(0)

        # -- header bar: red accent band + title --
        exit_header = QtWidgets.QWidget()
        exit_header.setObjectName("exitHeader")
        exit_header.setStyleSheet("""
            QWidget#exitHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(242, 54, 69, 0.25), stop:1 #2d2d2d);
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-left: 3px solid #f23645;
            }
        """)
        exit_header_h = QtWidgets.QHBoxLayout(exit_header)
        exit_header_h.setContentsMargins(10, 6, 10, 6)
        title = QtWidgets.QLabel("平仓条件总览")
        title.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 12px; background: transparent;")
        exit_header_h.addWidget(title)
        exit_header_h.addStretch()
        card_layout.addWidget(exit_header)

        # -- separator --
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #555; border: none;")
        card_layout.addWidget(sep)

        # -- grid container --
        grid_widget = QtWidgets.QWidget()
        grid_widget.setStyleSheet("background: transparent;")
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(0)
        grid.setContentsMargins(6, 0, 6, 2)

        # column headers
        for ci, text in enumerate(["条件", "阈值"]):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(
                "color: #999; font-size: 9px; font-weight: bold; "
                "background-color: #3a3a3a; padding: 4px 4px; "
                "border-bottom: 1px solid #555;"
            )
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter if ci else QtCore.Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(lbl, 0, ci)

        # -- read thresholds from config（平仓条件以分段止盈/分段止损为主，不再单独展示硬止损）--
        safe_th = PAPER_TRADING_CONFIG.get("HOLD_SAFE_THRESHOLD", 0.7)
        alert_th = PAPER_TRADING_CONFIG.get("HOLD_ALERT_THRESHOLD", 0.5)
        derail_th = PAPER_TRADING_CONFIG.get("HOLD_DERAIL_THRESHOLD", 0.3)
        tp1 = PAPER_TRADING_CONFIG.get("STAGED_TP_1_PCT", 5.0)
        tp2 = PAPER_TRADING_CONFIG.get("STAGED_TP_2_PCT", 10.0)
        sl1 = PAPER_TRADING_CONFIG.get("STAGED_SL_1_PCT", 5.0)
        sl2 = PAPER_TRADING_CONFIG.get("STAGED_SL_2_PCT", 10.0)
        r1 = PAPER_TRADING_CONFIG.get("STAGED_TP_RATIO_1", 0.30)
        mom_min = PAPER_TRADING_CONFIG.get("MOMENTUM_MIN_PROFIT_PCT", 1.5)
        mom_decay = PAPER_TRADING_CONFIG.get("MOMENTUM_DECAY_THRESHOLD", 0.5)
        max_hold = PAPER_TRADING_CONFIG.get("MAX_HOLD_BARS", 240)

        exit_rows = [
            ("分段止盈",      f"峰值 ≥ {tp1:.0f}% 减仓{r1:.0%}，≥ {tp2:.0f}% 再减{r1:.0%}"),
            ("分段止损",      f"亏损 ≥ {sl1:.0f}% 减仓{r1:.0%}，≥ {sl2:.0f}% 再减{r1:.0%}（与开仓硬止损并存）"),
            ("安全持仓",      f"相似度 ≥ {safe_th:.0%}"),
            ("警戒",          f"相似度 {alert_th:.0%}~{safe_th:.0%}（收紧止损，不平仓）"),
            ("动能衰竭",      f"盈利 ≥ {mom_min:.1f}% 且 K线缩量{mom_decay:.0%}"),
            ("最大持仓",      f"{max_hold}根K线"),
        ]

        for ri, (name, desc) in enumerate(exit_rows, start=1):
            row_bg = "rgba(58, 58, 58, 0.5)" if ri % 2 == 0 else "transparent"
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(
                f"color: #ddd; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(name_lbl, ri, 0)

            desc_lbl = QtWidgets.QLabel(desc)
            desc_lbl.setStyleSheet(
                f"color: #aaa; font-size: 10px; padding: 3px 4px; background: {row_bg};"
            )
            grid.addWidget(desc_lbl, ri, 1)

        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 5)

        card_layout.addWidget(grid_widget)
        parent_layout.addRow(card)

    # ─────────────────────────────────────────────────────────
    #  Entry overview real-time updater
    # ─────────────────────────────────────────────────────────
    def _update_entry_overview(self, *,
                               cosine: float = 0.0,
                               fusion: float = 0.0,
                               euclidean: float = 0.0,
                               dtw: float = 0.0,
                               macd_ready: bool = False,
                               kdj_ready: bool = False,
                               bayesian_win_rate: float = 0.0,
                               position_score: float = 0.0,
                               cold_start_active: bool = False):
        """Refresh entry overview card with live values and pass/fail badges on each tick."""

        if not hasattr(self, "_entry_overview_labels"):
            return

        # ── Cold start mode indicator badge ──
        self._cold_start_badge.setText("冷启动模式" if cold_start_active else "正常模式")
        self._cold_start_badge.setStyleSheet(self._cold_start_badge_style(cold_start_active))

        # ── Highlight the active threshold column ──
        # Active column: glow background; inactive column: dimmed
        for key, row_lbls in self._entry_overview_labels.items():
            normal_lbl = row_lbls["normal"]
            cold_lbl = row_lbls["cold"]
            if cold_start_active:
                normal_lbl.setStyleSheet(
                    "color: #555; font-size: 10px; padding: 3px 4px; background: transparent;"
                )
                cold_lbl.setStyleSheet(
                    "color: #FFA726; font-size: 10px; padding: 3px 4px; font-weight: bold; "
                    "background: rgba(255, 152, 0, 0.10); border-radius: 2px;"
                )
            else:
                normal_lbl.setStyleSheet(
                    "color: #4FC3F7; font-size: 10px; padding: 3px 4px; font-weight: bold; "
                    "background: rgba(79, 195, 247, 0.10); border-radius: 2px;"
                )
                cold_lbl.setStyleSheet(
                    "color: #555; font-size: 10px; padding: 3px 4px; background: transparent;"
                )

        # Highlight column headers (col 1 = normal, col 2 = cold)
        if hasattr(self, "_entry_col_header_lbls"):
            base_hdr = "font-size: 9px; font-weight: bold; padding: 4px 3px; border-bottom: 1px solid #555;"
            if cold_start_active:
                self._entry_col_header_lbls.get(1, QtWidgets.QLabel()).setStyleSheet(
                    f"color: #777; {base_hdr} background-color: #3a3a3a;"
                )
                self._entry_col_header_lbls.get(2, QtWidgets.QLabel()).setStyleSheet(
                    f"color: #FFA726; {base_hdr} background-color: rgba(255,152,0,0.12);"
                )
            else:
                self._entry_col_header_lbls.get(1, QtWidgets.QLabel()).setStyleSheet(
                    f"color: #4FC3F7; {base_hdr} background-color: rgba(79,195,247,0.12);"
                )
                self._entry_col_header_lbls.get(2, QtWidgets.QLabel()).setStyleSheet(
                    f"color: #777; {base_hdr} background-color: #3a3a3a;"
                )

        # ── Determine active thresholds for pass/fail evaluation ──
        cos_normal = SIMILARITY_CONFIG.get("COSINE_MIN_THRESHOLD", 0.70)
        fusion_normal = SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.65)
        euc_normal, dtw_normal = 0.35, 0.30
        cold_th = COLD_START_CONFIG.get("THRESHOLDS", {})
        cos_cold = cold_th.get("cosine", 0.50)
        fusion_cold = cold_th.get("fusion", 0.30)
        euc_cold = cold_th.get("euclidean", 0.25)
        dtw_cold = cold_th.get("dtw", 0.10)

        if cold_start_active:
            th_cos, th_fus, th_euc, th_dtw = cos_cold, fusion_cold, euc_cold, dtw_cold
        else:
            th_cos, th_fus, th_euc, th_dtw = cos_normal, fusion_normal, euc_normal, dtw_normal

        bayes_min = PAPER_TRADING_CONFIG.get("BAYESIAN_MIN_WIN_RATE", 0.40)
        pos_long = PAPER_TRADING_CONFIG.get("POS_THRESHOLD_LONG", -30)
        macd_bypass = COLD_START_CONFIG.get("MACD_BYPASS", True) and cold_start_active

        # ── Helper: update a single row's realtime value + status badge ──
        def _set_row(key: str, value_text: str, passed: bool, near: bool = False,
                     no_data: bool = False):
            row = self._entry_overview_labels.get(key)
            if not row:
                return
            rt_lbl = row["realtime"]
            st_lbl = row["status"]

            if no_data:
                rt_lbl.setText("--")
                rt_lbl.setStyleSheet(self._status_badge_none())
                st_lbl.setText("--")
                st_lbl.setStyleSheet(self._status_badge_none())
            elif passed:
                rt_lbl.setText(value_text)
                rt_lbl.setStyleSheet(
                    "color: #089981; font-size: 10px; padding: 3px 4px; "
                    "font-weight: bold; background: transparent;"
                )
                st_lbl.setText("✓")
                st_lbl.setStyleSheet(self._status_badge_pass())
            elif near:
                rt_lbl.setText(value_text)
                rt_lbl.setStyleSheet(
                    "color: #FFD54F; font-size: 10px; padding: 3px 4px; "
                    "font-weight: bold; background: transparent;"
                )
                st_lbl.setText("≈")
                st_lbl.setStyleSheet(self._status_badge_near())
            else:
                rt_lbl.setText(value_text)
                rt_lbl.setStyleSheet(
                    "color: #f23645; font-size: 10px; padding: 3px 4px; "
                    "font-weight: bold; background: transparent;"
                )
                st_lbl.setText("✗")
                st_lbl.setStyleSheet(self._status_badge_fail())

        # ── Update each entry condition row ──
        # Similarity metrics (show percentage, check vs active threshold)
        for metric_key, val, threshold in [
            ("cosine", cosine, th_cos),
            ("fusion", fusion, th_fus),
            ("euclidean", euclidean, th_euc),
            ("dtw", dtw, th_dtw),
        ]:
            has_data = val > 0.001
            passed = val >= threshold
            near_th = not passed and val >= threshold - 0.10
            _set_row(metric_key, f"{val:.0%}", passed, near_th, no_data=not has_data)

        # MACD trend gate
        if macd_bypass:
            _set_row("macd", "跳过", True)
        else:
            _set_row("macd", "就绪" if macd_ready else "未就绪", macd_ready)

        # KDJ gate
        _set_row("kdj", "就绪" if kdj_ready else "未就绪", kdj_ready)

        # Bayesian win rate
        has_bayes = bayesian_win_rate > 0.001
        bayes_pass = bayesian_win_rate >= bayes_min
        bayes_near = not bayes_pass and bayesian_win_rate >= bayes_min - 0.05
        _set_row("bayesian", f"{bayesian_win_rate:.0%}", bayes_pass, bayes_near,
                 no_data=not has_bayes)

        # Position score
        has_pos = position_score != 0.0
        pos_pass = position_score >= pos_long
        pos_near = not pos_pass and position_score >= pos_long - 10
        _set_row("position", f"{position_score:+.0f}", pos_pass, pos_near,
                 no_data=not has_pos)

    def _create_pending_tab(self):
        """创建委托单监控标签页"""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(10)
        
        # 提示标签（美化）
        self.pending_orders_hint_label = QtWidgets.QLabel("当前无挂单")
        self.pending_orders_hint_label.setStyleSheet("""
            color: #888;
            font-size: 11px;
            padding: 4px;
        """)
        tab_layout.addWidget(self.pending_orders_hint_label)

        # 委托单表格（美化）
        self.pending_orders_table = QtWidgets.QTableWidget()
        self.pending_orders_table.setColumnCount(6)
        self.pending_orders_table.setHorizontalHeaderLabels(["方向", "挂单价", "数量", "状态", "原型", "TP/SL%"])
        self.pending_orders_table.verticalHeader().setVisible(False)
        self.pending_orders_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pending_orders_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.pending_orders_table.setAlternatingRowColors(True)
        self.pending_orders_table.setMinimumHeight(120)
        self.pending_orders_table.horizontalHeader().setStretchLastSection(True)
        self.pending_orders_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                color: {UI_CONFIG['THEME_TEXT']};
                gridline-color: #3a3a3a;
                font-size: 11px;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }}
            QHeaderView::section {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d2d2d, stop:1 #252525);
                color: {UI_CONFIG['THEME_TEXT']};
                border: none;
                border-right: 1px solid #3a3a3a;
                border-bottom: 1px solid #3a3a3a;
                padding: 6px 4px;
                font-weight: bold;
                font-size: 10px;
            }}
            QTableWidget::item {{
                padding: 5px;
                border-bottom: 1px solid #3a3a3a;
            }}
            QTableWidget::item:alternate {{
                background-color: #272727;
            }}
        """)
        tab_layout.addWidget(self.pending_orders_table)
        tab_layout.addStretch()
        
        self.tabs.addTab(tab, "📋 委托单")
    
    def _create_monitoring_tab(self):
        """创建持仓监控标签页 - 5层逻辑链展示"""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(12, 12, 12, 12)
        tab_layout.setSpacing(8)
        
        # 标题
        title_label = QtWidgets.QLabel("📊 智能推理引擎")
        title_label.setStyleSheet("color: #007acc; font-weight: bold; font-size: 14px;")
        tab_layout.addWidget(title_label)

        # 持仓监控概览
        monitoring_card = QtWidgets.QWidget()
        monitoring_card.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)
        monitoring_layout = QtWidgets.QFormLayout(monitoring_card)
        monitoring_layout.setContentsMargins(10, 8, 10, 8)
        monitoring_layout.setSpacing(6)
        monitoring_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.hold_reason_label = QtWidgets.QLabel("-")
        self.hold_reason_label.setWordWrap(True)
        self.hold_reason_label.setStyleSheet("color: #ccc;")
        monitoring_layout.addRow("持仓说明:", self.hold_reason_label)

        self.danger_bar = QtWidgets.QProgressBar()
        self.danger_bar.setRange(0, 100)
        self.danger_bar.setValue(0)
        self.danger_bar.setTextVisible(True)
        self.danger_bar.setFormat("%p%")
        self.danger_bar.setFixedHeight(8)
        self.danger_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #f39c12;
                border-radius: 3px;
            }
        """)
        monitoring_layout.addRow("风险等级:", self.danger_bar)

        self.exit_monitor_label = QtWidgets.QLabel("-")
        self.exit_monitor_label.setWordWrap(True)
        self.exit_monitor_label.setStyleSheet("color: #bbb;")
        monitoring_layout.addRow("离场监控:", self.exit_monitor_label)

        tab_layout.addWidget(monitoring_card)
        
        # 滚动区域（容纳5层卡片）
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)
        
        # 5层推理卡片
        self._reasoning_layer_widgets = {}
        layer_configs = [
            ("market_stance", "🌍", "市场态势", "#4CAF50"),
            ("pattern_tracking", "🎯", "模式追踪", "#2196F3"),
            ("momentum_analysis", "📈", "动量分析", "#FF9800"),
            ("pnl_assessment", "💰", "盈亏评估", "#9C27B0"),
            ("safety_check", "🛡️", "安全检查", "#F44336"),
        ]
        
        for layer_id, icon, name, color in layer_configs:
            layer_card = self._create_reasoning_layer_card(layer_id, icon, name, color)
            self._reasoning_layer_widgets[layer_id] = layer_card
            scroll_layout.addWidget(layer_card['container'])
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)
        
        # 综合决策卡片
        verdict_card = self._create_verdict_card()
        self._verdict_widgets = verdict_card
        tab_layout.addWidget(verdict_card['container'])
        
        # 自适应参考区域（可折叠）
        adaptive_section = self._create_adaptive_reference_section()
        self._adaptive_ref_widgets = adaptive_section
        tab_layout.addWidget(adaptive_section['container'])
        
        self.tabs.addTab(tab, "🧠 推理")
    
    def _create_reasoning_layer_card(self, layer_id, icon, name, theme_color):
        """创建单个推理层卡片"""
        container = QtWidgets.QWidget()
        container.setStyleSheet(f"""
            QWidget {{
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }}
        """)
        
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # 顶部：图标 + 层名
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(8)
        
        icon_label = QtWidgets.QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")
        header_layout.addWidget(icon_label)
        
        name_label = QtWidgets.QLabel(name)
        name_label.setStyleSheet(f"color: {theme_color}; font-weight: bold; font-size: 12px;")
        header_layout.addWidget(name_label)
        
        status_badge = QtWidgets.QLabel("待评估")
        status_badge.setStyleSheet("""
            background-color: #555;
            color: #ccc;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
        """)
        header_layout.addWidget(status_badge)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # 中部：进度条 + 摘要
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(50)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(6)
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: #1e1e1e;
            }}
            QProgressBar::chunk {{
                background-color: {theme_color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress_bar)
        
        summary_label = QtWidgets.QLabel("-")
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(summary_label)
        
        # 底部：可展开的详情
        detail_label = QtWidgets.QLabel("")
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet("color: #888; font-size: 10px; margin-top: 4px;")
        detail_label.setVisible(False)
        layout.addWidget(detail_label)
        
        return {
            'container': container,
            'status_badge': status_badge,
            'progress_bar': progress_bar,
            'summary_label': summary_label,
            'detail_label': detail_label,
            'theme_color': theme_color,
        }
    
    def _create_verdict_card(self):
        """创建综合决策卡片（简朴、新闻媒体风格）"""
        container = QtWidgets.QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        
        # 新闻风格字体（宋体/报宋/Georgia）
        news_font = "Georgia, SimSun, 宋体, serif"
        
        # 决策标题
        title_label = QtWidgets.QLabel("综合决策")
        title_label.setStyleSheet(f"color: #b0b0b0; font-family: {news_font}; font-size: 12px; font-weight: normal;")
        layout.addWidget(title_label)
        
        # 决策建议
        verdict_label = QtWidgets.QLabel("等待持仓信号...")
        verdict_label.setWordWrap(True)
        verdict_label.setStyleSheet(f"""
            color: #e0e0e0;
            font-family: {news_font};
            font-size: 12px;
            font-weight: normal;
            line-height: 1.5;
            padding: 6px 0;
        """)
        layout.addWidget(verdict_label)
        
        # 推荐操作
        action_label = QtWidgets.QLabel("")
        action_label.setWordWrap(True)
        action_label.setStyleSheet(f"color: #a0a0a0; font-family: {news_font}; font-size: 11px; font-weight: normal;")
        layout.addWidget(action_label)
        
        # DeepSeek 持仓建议（含心跳灯）
        ds_row = QtWidgets.QHBoxLayout()
        ds_heartbeat = QtWidgets.QLabel("○")
        ds_heartbeat.setStyleSheet("color: #666; font-size: 10px;")
        ds_heartbeat.setToolTip("DeepSeek 心跳\n绿=已发送/请求中\n灰=未持仓或未到间隔")
        ds_row.addWidget(ds_heartbeat)
        ds_label = QtWidgets.QLabel("DeepSeek")
        ds_label.setStyleSheet(f"color: #808080; font-family: {news_font}; font-size: 11px;")
        ds_row.addWidget(ds_label)
        ds_row.addStretch()
        layout.addLayout(ds_row)
        deepseek_advice_label = QtWidgets.QLabel("")
        deepseek_advice_label.setWordWrap(True)
        deepseek_advice_label.setStyleSheet(f"color: #a0a0a0; font-family: {news_font}; font-size: 11px; line-height: 1.4;")
        layout.addWidget(deepseek_advice_label)
        
        return {
            'container': container,
            'verdict_label': verdict_label,
            'action_label': action_label,
            'ds_heartbeat': ds_heartbeat,
            'deepseek_advice_label': deepseek_advice_label,
        }
    
    def _create_adaptive_reference_section(self):
        """创建自适应参考区域（简朴、新闻风格）"""
        news_font = "Georgia, SimSun, 宋体, serif"
        container = QtWidgets.QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        
        # 可折叠标题
        header_layout = QtWidgets.QHBoxLayout()
        
        expand_btn = QtWidgets.QPushButton("▶")
        expand_btn.setFixedSize(20, 20)
        expand_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #707070;
                font-size: 11px;
            }
            QPushButton:hover {
                color: #909090;
            }
        """)
        expand_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        header_layout.addWidget(expand_btn)
        
        title_label = QtWidgets.QLabel("自适应学习参考")
        title_label.setStyleSheet(f"color: #b0b0b0; font-family: {news_font}; font-size: 12px; font-weight: normal;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # 内容区域（默认隐藏）
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 0, 0, 0)
        content_layout.setSpacing(6)
        
        # 原型历史表现
        proto_stats_label = QtWidgets.QLabel("原型历史: -")
        proto_stats_label.setStyleSheet(f"color: #a0a0a0; font-family: {news_font}; font-size: 11px;")
        content_layout.addWidget(proto_stats_label)
        
        # 最近调整记录
        adjustments_label = QtWidgets.QLabel("最近调整: 无")
        adjustments_label.setWordWrap(True)
        adjustments_label.setStyleSheet(f"color: #a0a0a0; font-family: {news_font}; font-size: 11px;")
        content_layout.addWidget(adjustments_label)
        
        content_widget.setVisible(False)
        layout.addWidget(content_widget)
        
        # 折叠/展开逻辑
        def toggle_expand():
            is_visible = content_widget.isVisible()
            content_widget.setVisible(not is_visible)
            expand_btn.setText("▼" if not is_visible else "▶")
        
        expand_btn.clicked.connect(toggle_expand)
        
        return {
            'container': container,
            'content_widget': content_widget,
            'proto_stats_label': proto_stats_label,
            'adjustments_label': adjustments_label,
        }
    
    def update_reasoning_layers(self, reasoning_result=None, state=None, order=None):
        """更新5层推理显示（根据TradeReasoning结果）"""
        if not hasattr(self, '_reasoning_layer_widgets'):
            return
        
        # 映射层ID与索引
        layer_ids = ['market_stance', 'pattern_tracking', 'momentum_analysis', 'pnl_assessment', 'safety_check']
        
        # 从 reasoning_result 读取真实数据
        if reasoning_result is not None and hasattr(reasoning_result, 'layers') and len(reasoning_result.layers) >= 5:
            layers = reasoning_result.layers
            status_map = {'favorable': '有利', 'neutral': '中性', 'adverse': '不利'}
            progress_map = {'favorable': 80, 'neutral': 55, 'adverse': 30}
            layers_data = {}
            for i, layer_id in enumerate(layer_ids):
                if i < len(layers):
                    layer = layers[i]
                    layers_data[layer_id] = {
                        'status': status_map.get(layer.status, layer.status),
                        'progress': progress_map.get(layer.status, 50),
                        'summary': layer.summary,
                        'detail': layer.detail,
                    }
                else:
                    layers_data[layer_id] = {'status': '-', 'progress': 50, 'summary': '-', 'detail': ''}
        else:
            # 无持仓时显示占位
            layers_data = {lid: {'status': '待评估', 'progress': 50, 'summary': '-', 'detail': ''} for lid in layer_ids}
        
        for layer_id, data in layers_data.items():
            if layer_id in self._reasoning_layer_widgets:
                widgets = self._reasoning_layer_widgets[layer_id]
                
                # 更新状态徽章
                status = data['status']
                if status in ['有利', '对齐', '加强中', '良好', '安全']:
                    badge_color = '#4CAF50'
                elif status in ['中性', '漂移', '维持', '可接受', '注意']:
                    badge_color = '#FF9800'
                else:
                    badge_color = '#F44336'
                
                widgets['status_badge'].setText(status)
                widgets['status_badge'].setStyleSheet(f"""
                    background-color: {badge_color};
                    color: white;
                    padding: 2px 8px;
                    border-radius: 10px;
                    font-size: 10px;
                    font-weight: bold;
                """)
                
                # 更新进度条
                widgets['progress_bar'].setValue(data['progress'])
                
                # 更新摘要
                widgets['summary_label'].setText(data['summary'])
                
                # 更新详情
                widgets['detail_label'].setText(data['detail'])
        
        # 更新综合决策（使用 reasoning_result + state 持仓建议）
        if hasattr(self, '_verdict_widgets'):
            action_text = ""
            # 首先根据是否有持仓设置基础文本
            if order is not None:
                verdict_text = "持仓中"
                # 如果有推理结果，使用详细判断
                if reasoning_result is not None and hasattr(reasoning_result, 'verdict'):
                    verdict_map = {
                        'hold_firm': '坚定持仓',
                        'tighten_watch': '收紧观察',
                        'prepare_exit': '准备平仓',
                        'exit_now': '立即平仓',
                    }
                    verdict_text = verdict_map.get(reasoning_result.verdict, reasoning_result.verdict)
                    if hasattr(reasoning_result, 'narrative') and reasoning_result.narrative:
                        verdict_text = f"{verdict_text} | {reasoning_result.narrative}"
            else:
                # 无持仓时
                verdict_text = "等待入场信号"
            # 叠加持仓止盈建议
            if state is not None and getattr(state, 'holding_exit_suggestion', ''):
                exit_sug = state.holding_exit_suggestion
                action_text = f"止盈建议: {exit_sug}"
                if getattr(state, 'position_suggestion', ''):
                    action_text += f" | 仓位: {state.position_suggestion}"
                if getattr(state, 'tpsl_action', ''):
                    tpsl_map = {'hold': '保持TP/SL', 'recalc': '重算TP/SL', 'tighten_sl_only': '仅收紧SL'}
                    action_text += f" | TP/SL: {tpsl_map.get(state.tpsl_action, state.tpsl_action)}"
            if action_text:
                self._verdict_widgets['action_label'].setText(action_text)
            self._verdict_widgets['verdict_label'].setText(verdict_text)
        
        # 更新 DeepSeek 持仓建议与心跳灯
        if hasattr(self, '_verdict_widgets') and 'ds_heartbeat' in self._verdict_widgets:
            hb = bool(getattr(state, 'deepseek_heartbeat', False)) if state else False
            self._verdict_widgets['ds_heartbeat'].setText("●" if hb else "○")
            self._verdict_widgets['ds_heartbeat'].setStyleSheet(
                "color: #00E676; font-size: 10px;" if hb else "color: #666; font-size: 10px;"
            )
        if hasattr(self, '_verdict_widgets') and 'deepseek_advice_label' in self._verdict_widgets:
            adv = (getattr(state, 'deepseek_holding_advice', '') or '') if state else ''
            jdg = (getattr(state, 'deepseek_judgement', '') or '') if state else ''
            parts = []
            if adv:
                parts.append(adv[:300] + "..." if len(adv) > 300 else adv)
            if jdg:
                parts.append(f"[评判] {jdg[:150]}..." if len(jdg) > 150 else f"[评判] {jdg}")
            self._verdict_widgets['deepseek_advice_label'].setText("\n".join(parts) if parts else "")
        
        # 更新自适应参考（从 state 拉取持仓相关数据）
        self._update_adaptive_reference(state, order)
    
    def _update_adaptive_reference(self, state=None, order=None):
        """更新自适应学习参考（出场时机/TP-SL/原型/仓位建议）"""
        if not hasattr(self, '_adaptive_ref_widgets'):
            return
        proto_text = "原型历史: -"
        adjustments_text = "最近调整: 无"
        if order is not None and getattr(order, 'template_fingerprint', ''):
            fp = order.template_fingerprint
            sim = getattr(order, 'entry_similarity', 0) or getattr(order, 'current_similarity', 0)
            proto_text = f"原型: {fp} | 相似度: {sim:.1%}"
        if state is not None:
            parts = []
            if getattr(state, 'exit_timing_scores', {}):
                for k, v in list(state.exit_timing_scores.items())[:2]:
                    if isinstance(v, dict) and v.get('suggestion'):
                        parts.append(f"出场时机({k}): {v.get('suggestion', '')}")
            if getattr(state, 'tpsl_scores', {}):
                for k, v in list(state.tpsl_scores.items())[:2]:
                    if isinstance(v, dict) and v.get('suggestion'):
                        parts.append(f"TP-SL({k}): {v.get('suggestion', '')}")
            if parts:
                adjustments_text = " | ".join(parts)
            if getattr(state, 'position_suggestion', ''):
                adjustments_text += f" | 仓位建议: {state.position_suggestion}"
            if getattr(state, 'holding_regime_change', ''):
                adjustments_text += f" | 市场状态: {state.holding_regime_change}"
        self._adaptive_ref_widgets['proto_stats_label'].setText(proto_text)
        self._adaptive_ref_widgets['adjustments_label'].setText(adjustments_text)
    
    def _create_log_tab(self):
        """创建实时日志标签页"""
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout(tab)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        
        # 实时日志（使用等宽字体，美化边框）
        self.event_log = QtWidgets.QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(800)
        self.event_log.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                color: #e0e0e0;
                font-family: 'Consolas', 'Courier New', 'Monospace';
                font-size: 11px;
                padding: 6px;
                selection-background-color: #4a4a4a;
            }}
        """)
        tab_layout.addWidget(self.event_log)
        
        self.tabs.addTab(tab, "📝 日志")
    
    
    def update_position(self, order):
        """更新持仓显示"""
        if order is None:
            self.position_side_label.setText("-")
            self.position_side_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")
            self.position_qty_label.setText("-")
            self.position_margin_label.setText("-")
            self.position_leverage_label.setText("-")
            self.position_leverage_label.setStyleSheet("color: #888;")
            self.update_adaptive_leverage_lamp(False)
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
            
            # 杠杆（从订单或交易器获取）
            current_leverage = getattr(order, 'leverage', None) or (self._paper_trader.leverage if hasattr(self, '_paper_trader') and self._paper_trader else 20)
            self.position_leverage_label.setText(f"{current_leverage}x")
            
            # 根据杠杆高低设置颜色提示
            if current_leverage >= 30:
                leverage_color = "#FF5252"  # 红色：高风险
            elif current_leverage >= 20:
                leverage_color = "#FFB74D"  # 橙色：中等风险
            else:
                leverage_color = "#81C784"  # 绿色：低风险
            self.position_leverage_label.setStyleSheet(f"color: {leverage_color}; font-weight: bold;")
            
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
            
    def update_adaptive_leverage_lamp(self, is_active: bool):
        """亮灯=自适应杠杆在用（绿），灰=未用"""
        if hasattr(self, "adaptive_leverage_lamp"):
            self.adaptive_leverage_lamp.setStyleSheet(
                "color: #4CAF50; font-size: 12px;" if is_active else "color: #666; font-size: 12px;"
            )
            
    def update_monitoring(self, hold_reason: str, danger_level: float, exit_reason: str):
        """更新持仓监控说明 (NEW)"""
        # 触发心跳
        self._trigger_heartbeat("holding")
        
        self.hold_reason_label.setText(hold_reason or "未持仓")
        self.danger_bar.setValue(int(danger_level))
        self.exit_monitor_label.setText(exit_reason or "-")
        
    def update_matching_context(self, market_regime: str, fp_status: str, reason: str,
                                matched_fp: str = "", matched_similarity: float = None,
                                swing_points_count: int = 0,
                                entry_threshold: float = None,
                                macd_ready: bool = False,
                                kdj_ready: bool = False,
                                bayesian_win_rate: float = 0.0,
                                kelly_position_pct: float = 0.0,
                                position_score: float = 0.0,
                                # 【指纹3D图】多维相似度分解
                                cosine_similarity: float = 0.0,
                                euclidean_similarity: float = 0.0,
                                dtw_similarity: float = 0.0,
                                prototype_confidence: float = 0.0,
                                final_match_score: float = 0.0,
                                cold_start_active: bool = False):
        """更新匹配状态和因果说明"""
        # 触发心跳
        self._trigger_heartbeat("market")
        self._trigger_heartbeat("fingerprint")
        self._trigger_heartbeat("gate")
        
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
            # 完整显示原型名称；若有贝叶斯概率，直接拼在原型旁边
            fp_display = matched_fp
            if bayesian_win_rate > 0:
                fp_display = f"{matched_fp}  |  贝叶斯 {bayesian_win_rate:.1%}"
            self.matched_fingerprint_label.setText(fp_display)
            self.matched_fingerprint_label.setToolTip(fp_display)
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
        
        # 【指纹3D图】更新多维相似度分解显示（未达标时也显示当前值，便于观察）
        cos_v = max(0.0, float(cosine_similarity or 0.0))
        euc_v = max(0.0, float(euclidean_similarity or 0.0))
        dtw_v = max(0.0, float(dtw_similarity or 0.0))
        conf_v = max(0.0, float(prototype_confidence or 0.0))

        self.cos_sim_badge.setText(f"方向: {cos_v:.0%}")
        self.cos_sim_badge.setStyleSheet(self._similarity_badge_style(self._get_similarity_color(cos_v)))
        self.euc_sim_badge.setText(f"距离: {euc_v:.0%}")
        self.euc_sim_badge.setStyleSheet(self._similarity_badge_style(self._get_similarity_color(euc_v)))
        self.dtw_sim_badge.setText(f"形态: {dtw_v:.0%}")
        self.dtw_sim_badge.setStyleSheet(self._similarity_badge_style(self._get_similarity_color(dtw_v)))

        # 原型置信度
        if conf_v > 0:
            conf_color = self._get_confidence_color(conf_v)
            conf_level = self._get_confidence_level(conf_v)
            self.confidence_label.setText(f"{conf_v:.1%}")
            self.confidence_label.setStyleSheet(f"color: {conf_color}; font-weight: bold; font-size: 12px;")
            self.confidence_level_badge.setText(conf_level)

            # 置信度等级徽章着色
            if conf_level == "高":
                badge_style = """
                    QLabel {
                        color: #00E676;
                        background-color: rgba(0, 230, 118, 0.15);
                        border: 1px solid #00E676;
                        border-radius: 3px;
                        padding: 0px 4px;
                        font-size: 9px;
                        font-weight: bold;
                    }
                """
            elif conf_level == "中":
                badge_style = """
                    QLabel {
                        color: #FFD700;
                        background-color: rgba(255, 215, 0, 0.15);
                        border: 1px solid #FFD700;
                        border-radius: 3px;
                        padding: 0px 4px;
                        font-size: 9px;
                        font-weight: bold;
                    }
                """
            else:
                badge_style = """
                    QLabel {
                        color: #f23645;
                        background-color: rgba(242, 54, 69, 0.15);
                        border: 1px solid #f23645;
                        border-radius: 3px;
                        padding: 0px 4px;
                        font-size: 9px;
                        font-weight: bold;
                    }
                """
            self.confidence_level_badge.setStyleSheet(badge_style)
            self.confidence_level_badge.show()
        else:
            self.confidence_label.setText("0.0%")
            self.confidence_label.setStyleSheet("color: #888; font-weight: bold; font-size: 12px;")
            self.confidence_level_badge.hide()
        
        # 更新贝叶斯胜率
        if bayesian_win_rate > 0:
            self.bayesian_win_rate_label.setText(f"{bayesian_win_rate:.1%}")
            # 根据胜率着色
            if bayesian_win_rate >= 0.60:
                wr_color = "#00E676"  # 绿色 - 高胜率
            elif bayesian_win_rate >= 0.50:
                wr_color = "#FFD700"  # 金色 - 及格
            else:
                wr_color = "#f23645"  # 红色 - 低胜率
            self.bayesian_win_rate_label.setStyleSheet(f"color: {wr_color}; font-weight: bold; font-size: 13px;")
        else:
            self.bayesian_win_rate_label.setText("-")
            self.bayesian_win_rate_label.setStyleSheet("color: #888; font-weight: bold; font-size: 13px;")
        
        # 空间位置评分显示
        self.position_score_label.setText(f"{position_score:+.0f}" if position_score != 0 else "-")
        if position_score > 40:
            self.position_score_label.setStyleSheet("color: #00E676; font-weight: bold;")
        elif position_score > 0:
            self.position_score_label.setStyleSheet("color: #66BB6A; font-weight: bold;")
        elif position_score < -20:
            self.position_score_label.setStyleSheet("color: #f23645; font-weight: bold;")
        elif position_score < 0:
            self.position_score_label.setStyleSheet("color: #EF9A9A; font-weight: bold;")
        else:
            self.position_score_label.setStyleSheet("color: #888; font-weight: bold;")
        
        self.reason_label.setText(reason or "-")

        # ── 开仓条件总览卡片实时刷新 ──
        self._update_entry_overview(
            cosine=float(cosine_similarity or 0.0),
            fusion=float(matched_similarity or 0.0),
            euclidean=float(euclidean_similarity or 0.0),
            dtw=float(dtw_similarity or 0.0),
            macd_ready=macd_ready,
            kdj_ready=kdj_ready,
            bayesian_win_rate=float(bayesian_win_rate or 0.0),
            position_score=float(position_score or 0.0),
            cold_start_active=cold_start_active,
        )

    def update_pending_orders(self, pending_orders: List[dict]):
        """更新委托单监控表（挂单中）"""
        # 高频状态回调下避免重复全量重绘表格
        signature = tuple(
            (
                str(o.get("order_type", "")),
                str(o.get("side", "")),
                round(float(o.get("trigger_price", 0.0) or 0.0), 4),
                round(float(o.get("quantity", 0.0) or 0.0), 6),
                str(o.get("status", "")),
                str(o.get("template_fingerprint", "")),
                round(float(o.get("take_profit", o.get("tp", 0.0)) or 0.0), 4),
                round(float(o.get("stop_loss", o.get("sl", 0.0)) or 0.0), 4),
            )
            for o in (pending_orders or [])
        )
        if signature == getattr(self, "_last_pending_orders_signature", None):
            return
        self._last_pending_orders_signature = signature
        rows = len(pending_orders or [])
        self.pending_orders_table.setRowCount(rows)
        if rows == 0:
            self.pending_orders_hint_label.setText("当前无挂单")
            self.pending_orders_hint_label.setStyleSheet("color: #888; font-size: 11px;")
            return

        self.pending_orders_hint_label.setText(f"挂单中: {rows} 笔")
        self.pending_orders_hint_label.setStyleSheet("color: #FFD54F; font-size: 11px; font-weight: bold;")

        for row, o in enumerate(pending_orders):
            side = (o.get("side") or "-").upper()
            side_item = QtWidgets.QTableWidgetItem(side)
            # 颜色：LONG/BUY=绿色（做多/平空），SHORT/SELL=红色（做空/平多）
            is_bullish = side in ("LONG", "BUY")
            side_item.setForeground(QtGui.QColor("#089981") if is_bullish else QtGui.QColor("#f23645"))
            self.pending_orders_table.setItem(row, 0, side_item)

            self.pending_orders_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(o.get('trigger_price', 0.0)):.2f}"))
            self.pending_orders_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(o.get('quantity', 0.0)):.3f}"))
            self.pending_orders_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(o.get("status", "等待成交"))))

            fp = str(o.get("template_fingerprint", "-"))
            fp_item = QtWidgets.QTableWidgetItem(fp)
            fp_item.setToolTip(fp)
            self.pending_orders_table.setItem(row, 4, fp_item)

            # TP/SL%：预计盈亏（金额 + 百分比）。保护单单行显示该笔预计亏/赚；入场单显示 TP/SL 两档预计。
            trigger = float(o.get("trigger_price", 0) or 0)
            entry_price = o.get("entry_price")
            if entry_price is not None:
                try:
                    entry_price = float(entry_price)
                except (TypeError, ValueError):
                    entry_price = None
            qty = float(o.get("quantity", 0) or 0)
            order_type = (o.get("order_type") or "").lower()
            status_str = str(o.get("status", ""))
            tpsl_text = "-"
            if trigger > 0 and entry_price and entry_price > 0 and qty > 0 and order_type in ("sl", "tp"):
                # 保护单：该行是止损或止盈，trigger 即为 SL 价或 TP 价
                is_short = "BUY" in (o.get("side") or "").upper()  # 平空 = 原仓位 SHORT
                if order_type == "sl":
                    if is_short:
                        loss_usdt = (trigger - entry_price) * qty
                        loss_pct = (trigger - entry_price) / entry_price * 100
                    else:
                        loss_usdt = (entry_price - trigger) * qty
                        loss_pct = (entry_price - trigger) / entry_price * 100
                    tpsl_text = f"预计亏 {loss_usdt:+.2f} USDT ({loss_pct:+.2f}%)"
                else:
                    if is_short:
                        profit_usdt = (entry_price - trigger) * qty
                        profit_pct = (entry_price - trigger) / entry_price * 100
                    else:
                        profit_usdt = (trigger - entry_price) * qty
                        profit_pct = (trigger - entry_price) / entry_price * 100
                    tpsl_text = f"预计赚 {profit_usdt:+.2f} USDT ({profit_pct:+.2f}%)"
            elif trigger > 0 and qty > 0 and (tp_price := (o.get("take_profit") or o.get("tp"))) is not None and (sl_price := (o.get("stop_loss") or o.get("sl"))) is not None:
                # 入场挂单：用 trigger 作入场价，计算 TP/SL 两档预计盈亏
                try:
                    tp_val, sl_val = float(tp_price), float(sl_price)
                    entry = trigger
                    side_upper = (o.get("side") or "-").upper()
                    if "LONG" in side_upper or side_upper == "BUY":
                        tp_usdt = (tp_val - entry) * qty
                        tp_pct = (tp_val - entry) / entry * 100
                        sl_usdt = (entry - sl_val) * qty
                        sl_pct = (entry - sl_val) / entry * 100
                    else:
                        tp_usdt = (entry - tp_val) * qty
                        tp_pct = (entry - tp_val) / entry * 100
                        sl_usdt = (sl_val - entry) * qty
                        sl_pct = (sl_val - entry) / entry * 100
                    tpsl_text = f"TP 预计赚 {tp_usdt:+.2f} USDT ({tp_pct:+.2f}%) | SL 预计亏 {sl_usdt:+.2f} USDT ({sl_pct:+.2f}%)"
                except (TypeError, ValueError):
                    pass
            tpsl_item = QtWidgets.QTableWidgetItem(tpsl_text)
            tpsl_item.setForeground(QtGui.QColor("#AB47BC"))
            self.pending_orders_table.setItem(row, 5, tpsl_item)
    
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
    
    def _similarity_badge_style(self, color: str) -> str:
        """生成相似度徽章样式"""
        rgb_str = self._hex_to_rgb_str(color)
        return f"""
            QLabel {{
                background-color: rgba({rgb_str}, 0.15);
                color: {color};
                border: 1px solid {color};
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 10px;
                font-weight: bold;
            }}
        """
    
    @staticmethod
    def _hex_to_rgb_str(hex_color: str) -> str:
        """将 #RRGGBB 转换为 'R, G, B' 字符串（用于 rgba()）"""
        h = hex_color.lstrip('#')
        if len(h) != 6:
            return "136, 136, 136"
        return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"
    
    def _get_similarity_color(self, value: float) -> str:
        """根据相似度值获取颜色"""
        if value >= 0.80:
            return "#00E676"  # 亮绿
        elif value >= 0.70:
            return "#089981"  # 绿
        elif value >= 0.60:
            return "#FFD54F"  # 黄
        elif value >= 0.50:
            return "#FF9800"  # 橙
        else:
            return "#f23645"  # 红
    
    def _get_confidence_color(self, confidence: float) -> str:
        """根据置信度获取颜色"""
        if confidence >= 0.70:
            return "#00E676"  # 亮绿 - 高置信度
        elif confidence >= 0.50:
            return "#FFD700"  # 金色 - 中置信度
        elif confidence >= 0.30:
            return "#FF9800"  # 橙色 - 低置信度
        else:
            return "#f23645"  # 红色 - 极低置信度
    
    def _get_confidence_level(self, confidence: float) -> str:
        """获取置信度等级描述"""
        if confidence >= 0.70:
            return "高"
        elif confidence >= 0.50:
            return "中"
        elif confidence >= 0.30:
            return "低"
        else:
            return "极低"
    
    def _create_heartbeat_indicator(self) -> QtWidgets.QLabel:
        """创建心跳指示器（圆点）"""
        indicator = QtWidgets.QLabel("●")
        indicator.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 0px;
                margin: 0px;
            }
        """)
        indicator.setFixedWidth(15)
        indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        indicator.setToolTip("系统心跳指示器\n绿色闪烁=正常运行\n灰色=停止/异常")
        return indicator
    
    def _update_heartbeats(self):
        """更新心跳显示（每500ms调用）"""
        import time
        current_time = time.time()
        self._heartbeat_blink_state = not self._heartbeat_blink_state
        
        for module, indicator in self._heartbeat_indicators.items():
            last_update = self._heartbeats.get(module, 0)
            time_since_update = current_time - last_update
            
            # 超过3秒未更新 = 掉线/停止
            if time_since_update > 3.0:
                indicator.setStyleSheet("""
                    QLabel {
                        color: #666;
                        font-size: 12px;
                    }
                """)
            else:
                # 心跳闪烁：绿色 <-> 深绿
                color = "#00E676" if self._heartbeat_blink_state else "#089981"
                indicator.setStyleSheet(f"""
                    QLabel {{
                        color: {color};
                        font-size: 12px;
                    }}
                """)
    
    def _trigger_heartbeat(self, module: str):
        """触发心跳（在数据更新时调用）"""
        import time
        self._heartbeats[module] = time.time()

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
        self.table.setColumnCount(16)  # 原13列 + 新增3列 = 16列
        self.table.setHorizontalHeaderLabels([
            "时间", "方向", "数量", "入场价", "出场价", "止盈", "止损", 
            "盈亏%", "峰值%", "精准度", "信号",  # 新增3列：峰值利润、止盈精准度、信号触发
            "盈亏(USDT)", "手续费", "原因", "持仓", "操作"
        ])
        
        # 【自动调整列宽】确保所有内容都能完整显示
        header = self.table.horizontalHeader()
        
        # 统一策略：所有列自适应内容
        for col in range(self.table.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        
        # 精细控制关键列的最小宽度（确保美观和可读性）
        min_widths = {
            0: 90,   # 时间：mm-dd HH:MM
            1: 50,   # 方向：LONG/SHORT
            2: 70,   # 数量：0.0340
            3: 80,   # 入场价：67076.00
            4: 80,   # 出场价：66983.26
            5: 80,   # 止盈：67579.07
            6: 80,   # 止损：66740.62
            7: 70,   # 盈亏%：+7.5%
            8: 70,   # 峰值%：+8.2%
            9: 80,   # 精准度：91.5%✓
            10: 50,  # 信号：2个
            11: 90,  # 盈亏(USDT)：+45.00
            12: 70,  # 手续费：0.0000
            13: 80,  # 原因：追踪止盈
            14: 50,  # 持仓：12
            15: 60,  # 操作：删除按钮
        }
        
        for col, min_width in min_widths.items():
            current_width = self.table.columnWidth(col)
            if current_width < min_width:
                self.table.setColumnWidth(col, min_width)
        
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
        """添加单个交易记录。分段止盈/止损每段一条；同一笔的「持」行在收到平仓时复用为平仓行。"""
        key = self._trade_key(order)
        if key in self._rows_by_key:
            # 已存在则更新（例如平仓、或同步更新）
            row_idx = self._rows_by_key[key]
            self._update_trade_row(row_idx, order)
            self.table.resizeRowToContents(row_idx)
        else:
            order_id = str(getattr(order, "order_id", "") or "")
            exit_time = getattr(order, "exit_time", None)
            # 已平仓且存在同笔持仓行时，复用该行显示平仓，避免留下多余「持」
            if exit_time is not None and order_id and order_id in self._rows_by_key:
                row_idx = self._rows_by_key.pop(order_id)
                self._update_trade_row(row_idx, order)
                self._rows_by_key[key] = row_idx
                self.table.resizeRowToContents(row_idx)
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
        
        # 批量加载后，再次调整所有行高（性能优化）
        self.table.resizeRowsToContents()
            
    def _trade_key(self, order) -> str:
        """生成稳定的交易标识。已平仓（含分段）用 order_id+exit_ts 区分，保证每段一行。"""
        order_id = str(getattr(order, "order_id", "") or "")
        exit_time = getattr(order, "exit_time", None)
        exit_ts = exit_time.timestamp() if exit_time else 0.0
        if order_id and not order_id.startswith("EXCHANGE_SYNC"):
            # 已平仓则按平仓时间区分，便于分段止盈/分段止损每段一条记录
            if exit_time is not None:
                return f"{order_id}-{exit_ts:.3f}"
            return order_id
        side = getattr(order, "side", None)
        side_val = side.value if side else "-"
        entry_price = getattr(order, "entry_price", 0.0)
        quantity = getattr(order, "quantity", 0.0)
        entry_time = getattr(order, "entry_time", None)
        entry_ts = entry_time.timestamp() if entry_time else 0.0
        return f"SYNC-{side_val}-{entry_price:.2f}-{quantity:.6f}-{entry_ts:.0f}-{exit_ts:.0f}"
    
    def _insert_trade_row(self, order):
        """内部通用插入行逻辑"""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        # 为了美观，新纪录放前面？或者按时间排序。这里维持原有顺序，但在 TableWidget 中 insertRow(0) 可以置顶
        # 目前按时间顺序追加
        
        self._update_trade_row(row, order)
        
        # 自动调整行高
        self.table.resizeRowToContents(row)
        
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
        
        # 方向（翻转单加标记）
        side_val = order.side.value
        is_flip = getattr(order, 'is_flip_trade', False)
        side_display = f"🔄{side_val}" if is_flip else side_val
        side_item = QtWidgets.QTableWidgetItem(side_display)
        side_color = QtGui.QColor("#089981") if side_val == "LONG" else QtGui.QColor("#f23645")
        side_item.setForeground(side_color)
        if is_flip:
            flip_reason = getattr(order, 'flip_reason', '位置翻转')
            side_item.setToolTip(f"翻转单: {flip_reason}")
        self.table.setItem(row, 1, side_item)
        
        # 数量
        quantity = getattr(order, "quantity", 0.0)
        qty_item = QtWidgets.QTableWidgetItem(f"{quantity:.4f}")
        qty_item.setForeground(QtGui.QColor("#9e9e9e"))  # 灰色
        self.table.setItem(row, 2, qty_item)
        
        # 入场价
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{order.entry_price:.2f}"))
        
        # 出场价
        exit_price = order.exit_price if order.exit_price else "-"
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{exit_price:.2f}" if isinstance(exit_price, float) else exit_price))

        # 止盈 / 止损
        tp_val = getattr(order, "take_profit", None)
        sl_val = getattr(order, "stop_loss", None)
        tp_text = f"{tp_val:.2f}" if isinstance(tp_val, float) else "-"
        sl_text = f"{sl_val:.2f}" if isinstance(sl_val, float) else "-"
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(tp_text))
        self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(sl_text))

        # 盈亏%
        pnl_pct_item = QtWidgets.QTableWidgetItem(f"{order.profit_pct:+.2f}%")
        pnl_color = QtGui.QColor("#089981") if order.profit_pct >= 0 else QtGui.QColor("#f23645")
        pnl_pct_item.setForeground(pnl_color)
        self.table.setItem(row, 7, pnl_pct_item)
        
        # ========== 新增列：峰值利润% ==========
        peak_pct = getattr(order, "peak_profit_pct", 0.0)
        is_closed = getattr(order, "status", None) == OrderStatus.CLOSED or order.exit_time is not None
        
        if is_closed and peak_pct != 0.0:
            peak_item = QtWidgets.QTableWidgetItem(f"{peak_pct:+.2f}%")
            # 峰值利润用紫色显示（区别于实际盈亏）
            peak_item.setForeground(QtGui.QColor("#AB47BC"))
            peak_item.setToolTip(f"持仓期间的最高利润：{peak_pct:+.2f}%")
        else:
            peak_item = QtWidgets.QTableWidgetItem("-")
            peak_item.setForeground(QtGui.QColor("#666"))
        self.table.setItem(row, 8, peak_item)
        
        # ========== 新增列：止盈精准度 ==========
        if is_closed and peak_pct > 0.01:  # 峰值利润 > 0.01% 才计算精准度
            accuracy = (order.profit_pct / peak_pct) * 100
            accuracy_item = QtWidgets.QTableWidgetItem(f"{accuracy:.1f}%")
            
            # 根据精准度设置颜色
            if accuracy >= 90:
                accuracy_item.setForeground(QtGui.QColor("#089981"))  # 绿色：优秀
                grade = "✓ 优秀"
            elif accuracy >= 70:
                accuracy_item.setForeground(QtGui.QColor("#FFD54F"))  # 黄色：良好
                grade = "○ 良好"
            elif accuracy >= 50:
                accuracy_item.setForeground(QtGui.QColor("#FF9800"))  # 橙色：一般
                grade = "△ 一般"
            else:
                accuracy_item.setForeground(QtGui.QColor("#f23645"))  # 红色：差
                grade = "✗ 差"
            
            # 工具提示：详细说明
            tooltip = (
                f"止盈精准度：{accuracy:.1f}% ({grade})\n"
                f"实际平仓：{order.profit_pct:+.2f}% / 峰值利润：{peak_pct:+.2f}%\n\n"
                f"评级标准：\n"
                f"  ≥90%  ✓ 优秀（几乎在最佳点位）\n"
                f"  70-90% ○ 良好（可接受的回撤）\n"
                f"  50-70% △ 一般（错过较多利润）\n"
                f"  <50%   ✗ 差（严重卖飞）"
            )
            accuracy_item.setToolTip(tooltip)
        elif is_closed and peak_pct < 0:
            # 峰值为负（全程亏损），精准度无意义
            accuracy_item = QtWidgets.QTableWidgetItem("N/A")
            accuracy_item.setForeground(QtGui.QColor("#666"))
            accuracy_item.setToolTip("全程亏损，无精准度数据")
        else:
            # 持仓中或峰值为0
            accuracy_item = QtWidgets.QTableWidgetItem("-")
            accuracy_item.setForeground(QtGui.QColor("#666"))
        self.table.setItem(row, 9, accuracy_item)
        
        # ========== 新增列：离场信号触发 + 基于峰值的精简建议 ==========
        signals = getattr(order, "exit_signals_triggered", [])
        signal_count = len(signals)
        main_text = self._format_signal_description(order)
        suggestion = self._peak_suggestion(order, peak_pct, order.profit_pct) if is_closed else ""
        if suggestion:
            main_text = main_text + "\n" + suggestion
        signal_item = QtWidgets.QTableWidgetItem(main_text)
        if signal_count > 0:
            signal_item.setForeground(QtGui.QColor("#00BCD4"))
            signal_details = []
            for i, (signal_name, profit_at_trigger) in enumerate(signals, 1):
                signal_name_cn = {
                    "momentum_decay": "动量衰减",
                    "market_reversal": "市场反转",
                    "pattern_exit": "形态离场",
                    "derail": "脱轨",
                }.get(signal_name, signal_name)
                signal_details.append(f"{i}. {signal_name_cn} (触发时利润: {profit_at_trigger:+.2f}%)")
            tooltip = "持仓期间触发的离场信号：\n" + "\n".join(signal_details)
            if suggestion:
                tooltip += f"\n建议：{suggestion}"
            signal_item.setToolTip(tooltip)
        else:
            signal_item.setForeground(QtGui.QColor("#666"))
            if suggestion:
                signal_item.setToolTip(f"建议：{suggestion}")
        self.table.setItem(row, 10, signal_item)
        
        # 盈亏(USDT) - 开仓显示未实现，平仓显示已实现（索引 +3）
        if is_closed:
            pnl_val = getattr(order, "realized_pnl", 0.0)
        else:
            pnl_val = getattr(order, "unrealized_pnl", 0.0)
        pnl_usdt_item = QtWidgets.QTableWidgetItem(f"{pnl_val:+,.2f}")
        pnl_usdt_item.setForeground(pnl_color)
        self.table.setItem(row, 11, pnl_usdt_item)
        
        # 手续费（索引 +3）
        fee_val = getattr(order, "total_fee", 0.0)
        fee_item = QtWidgets.QTableWidgetItem(f"{fee_val:.4f}")
        fee_item.setForeground(QtGui.QColor("#f9a825"))  # 黄色
        self.table.setItem(row, 12, fee_item)
        
        # 原因（具体分类）（索引 +3）
        reason_display = self._classify_exit_reason(order)
        reason_item = QtWidgets.QTableWidgetItem(reason_display)
        if hasattr(order, 'decision_reason') and order.decision_reason:
            reason_item.setToolTip(order.decision_reason)  # 悬停显示完整原因
        self.table.setItem(row, 13, reason_item)
        
        # 持仓时长（索引 +3）
        self.table.setItem(row, 14, QtWidgets.QTableWidgetItem(str(order.hold_bars)))
        
        # 操作按钮（索引 +3）
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
        self.table.setCellWidget(row, 15, delete_btn)

    def _format_signal_description(self, order) -> str:
        """格式化信号列描述，支持精品信号解析"""
        fp = getattr(order, "template_fingerprint", "") or ""
        if not fp:
            return "-"
            
        # 精品信号格式: "long|cond1+cond2+cond3" 或 "short|cond1+cond2"
        if "|" in fp and (fp.startswith("long|") or fp.startswith("short|")):
            try:
                direction, cond_str = fp.split("|", 1)
                conditions = cond_str.split("+")
                
                from core.signal_utils import _format_conditions
                desc = _format_conditions(conditions, direction)
                
                # 截断长描述
                if len(desc) > 24:
                    desc = desc[:21] + "..."
                return desc
            except Exception:
                return fp
        
        # 原有逻辑：返回简短指纹或ID
        if len(fp) > 12:
            return fp[:10] + ".."
        return fp
    
    def _peak_suggestion(self, order, peak_pct: float, profit_pct: float) -> str:
        """基于峰值与实际盈亏给出精简建议（仅已平仓且峰值有效时）。"""
        if peak_pct <= 0:
            return ""
        detail = getattr(order, "decision_reason", "") or ""
        reason_val = getattr(order.close_reason, "value", "") if order.close_reason else ""
        # 峰值高但未兑现 → 应及时止盈
        if peak_pct >= 1.2 and profit_pct < peak_pct * 0.5:
            if "止盈" not in detail and "触及止盈" not in detail:
                return "应及时止盈"
        # 止损但曾有利 → 可放宽或追踪
        if reason_val == "止损" and peak_pct > abs(profit_pct) * 0.3:
            return "可放宽止损"
        # 追踪止损但回撤大 → 可提前追踪
        if "追踪" in detail or "追踪止损" in str(reason_val):
            if profit_pct < peak_pct * 0.6 and peak_pct >= 1.0:
                return "可提前追踪"
        return ""

    def _classify_exit_reason(self, order) -> str:
        """
        从订单信息中提取具体的离场分类

        Returns:
            具体的离场原因分类字符串
        """
        if not order.close_reason:
            return "-"
        
        # 获取详细原因
        detail = getattr(order, 'decision_reason', '')
        
        # 基于decision_reason和close_reason综合判断
        if "触及止盈价" in detail:
            return "固定止盈"
        elif "追踪止损" in detail or order.close_reason.value == "追踪止损":
            # 追踪止损/保本止损：SL移至盈利区后触发
            if "保本" in detail:
                return "保本止损"
            elif "锁利" in detail:
                return "锁利止损"
            elif "紧追" in detail:
                return "紧追止损"
            else:
                return "追踪止损"
        elif "追踪止盈" in detail:
            # 兼容旧数据（修复前的记录可能用"追踪止盈"）
            if "保本" in detail:
                return "保本止损"
            elif "锁利" in detail:
                return "锁利止损"
            elif "紧追" in detail:
                return "紧追止损"
            else:
                return "追踪止损"
        elif order.close_reason.value == "分段止盈":
            return "分段止盈"
        elif order.close_reason.value == "分段止损":
            return "分段止损"
        elif "阶梯止盈" in detail or "partial" in detail.lower():
            return "分段减仓"
        elif "触及止损价" in detail or order.close_reason.value == "止损":
            return "止损"
        elif "市场反转" in detail:
            # 提取具体的市场反转原因
            if "MACD" in detail and "KDJ" in detail:
                return "市场反转"
            else:
                return "市场反转"
        elif "信号" in detail or "离场模式" in detail:
            # 提取具体的信号类型
            if "反转形态" in detail:
                return "反转信号"
            elif "加速" in detail:
                return "加速信号"
            elif "脱轨" in detail:
                return "脱轨信号"
            else:
                return "形态信号"
        elif order.close_reason.value == "DERAIL":
            return "相似度脱轨"
        elif order.close_reason.value == "MAX_HOLD":
            return "超时离场"
        elif order.close_reason.value == "MANUAL":
            return "手动平仓"
        elif order.close_reason.value == "交易所平仓":
            return "交易所平仓"
        elif order.close_reason.value == "位置翻转":
            flip_reason = getattr(order, 'flip_reason', '')
            if "底部" in detail or "底部" in flip_reason:
                return "🔄底部翻转"
            elif "顶部" in detail or "顶部" in flip_reason:
                return "🔄顶部翻转"
            else:
                return "🔄位置翻转"
        else:
            # 回退到原始CloseReason
            return order.close_reason.value
    
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


class RejectionLogCard(QtWidgets.QWidget):
    """
    拒绝记录卡片面板 —— 显示被门控拦截的交易信号及其事后评估。

    展示内容：
    - 最近 20 条拒绝记录（滚动列表），每条包含时间、方向、门控代码（色标徽章）、价格、详情
    - 评估完成后显示绿色✓（正确拒绝）或红色✗（错误拒绝）+ 价格结果
    - 底部汇总：每个门控的准确率柱状图
    - "建议调整" 按钮（展示门控调参建议，需手动确认）
    """

    # 门控代码 → 色标
    FAIL_CODE_COLORS = {
        "BLOCK_POS":       "#FF9800",   # orange
        "BLOCK_MACD":      "#F44336",   # red
        "BLOCK_BAYES":     "#9C27B0",   # purple
        "BLOCK_KELLY_NEG": "#673AB7",   # deep purple
        "FLIP_NO_MATCH":   "#FFC107",   # yellow
        "BLOCK_REGIME_UNKNOWN": "#607D8B",  # blue gray
        "BLOCK_REGIME_CONFLICT": "#795548", # brown
    }

    # 门控代码 → 中文标签
    FAIL_CODE_LABELS = {
        "BLOCK_POS":       "位置过滤",
        "BLOCK_MACD":      "MACD门控",
        "BLOCK_BAYES":     "贝叶斯过滤",
        "BLOCK_KELLY_NEG": "凯利否决",
        "FLIP_NO_MATCH":   "无匹配",
        "BLOCK_REGIME_UNKNOWN": "市场未知",
        "BLOCK_REGIME_CONFLICT": "方向冲突",
    }

    # 信号：请求显示门控调参建议（detail dict list）
    suggest_adjustments_requested = QtCore.pyqtSignal()
    # 信号：用户确认应用某个阈值调整 (param_key, new_value)
    adjustment_confirmed = QtCore.pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rejections: List[Dict] = []      # 最近拒绝记录
        self._gate_scores: Dict[str, Dict] = {}  # fail_code → {correct, wrong, accuracy, ...}
        self._suggestions: List[Dict] = []      # 当前调整建议
        self._init_ui()

    # ------------------------------------------------------------------
    # UI 初始化
    # ------------------------------------------------------------------
    def _init_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
        """)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 主容器 GroupBox ──
        group = QtWidgets.QGroupBox("拒绝记录")
        group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(8, 14, 8, 8)
        group_layout.setSpacing(6)

        # ── 顶部：统计摘要行 ──
        summary_row = QtWidgets.QHBoxLayout()
        summary_row.setSpacing(8)

        self._total_label = QtWidgets.QLabel("总拒绝: 0")
        self._total_label.setStyleSheet("color: #888; font-size: 11px;")
        summary_row.addWidget(self._total_label)

        self._evaluated_label = QtWidgets.QLabel("已评估: 0")
        self._evaluated_label.setStyleSheet("color: #888; font-size: 11px;")
        summary_row.addWidget(self._evaluated_label)

        summary_row.addStretch()

        self._suggest_btn = QtWidgets.QPushButton("建议调整")
        self._suggest_btn.setFixedHeight(22)
        self._suggest_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {UI_CONFIG['THEME_ACCENT']};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: #0098ff;
            }}
            QPushButton:disabled {{
                background-color: #444;
                color: #888;
            }}
        """)
        self._suggest_btn.setToolTip(
            "根据拒绝评估结果，给出门控阈值调整建议（需手动确认）\n"
            "需至少 20 次评估才会产生建议"
        )
        self._suggest_btn.setEnabled(False)
        self._suggest_btn.clicked.connect(self._on_suggest_clicked)
        summary_row.addWidget(self._suggest_btn)

        self._suggest_status_label = QtWidgets.QLabel("")
        self._suggest_status_label.setStyleSheet("color: #888; font-size: 10px;")
        summary_row.addWidget(self._suggest_status_label)

        group_layout.addLayout(summary_row)

        # ── 中部：拒绝记录滚动列表 ──
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {UI_CONFIG['THEME_SURFACE']};
            }}
        """)

        self._list_container = QtWidgets.QWidget()
        self._list_layout = QtWidgets.QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(4, 4, 4, 4)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch()  # 底部弹性空间

        # 空状态提示
        self._empty_label = QtWidgets.QLabel("暂无拒绝记录\n系统运行后，被门控拦截的信号将显示在此处")
        self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #666; font-size: 11px; padding: 20px;")
        self._list_layout.insertWidget(0, self._empty_label)

        self._scroll.setWidget(self._list_container)
        group_layout.addWidget(self._scroll, stretch=1)

        # ── 底部：门控准确率柱状条 ──
        self._accuracy_container = QtWidgets.QWidget()
        accuracy_layout = QtWidgets.QVBoxLayout(self._accuracy_container)
        accuracy_layout.setContentsMargins(0, 4, 0, 0)
        accuracy_layout.setSpacing(3)

        accuracy_title = QtWidgets.QLabel("门控准确率")
        accuracy_title.setStyleSheet("color: #aaa; font-size: 10px; font-weight: bold;")
        accuracy_layout.addWidget(accuracy_title)

        self._accuracy_bars: Dict[str, QtWidgets.QProgressBar] = {}
        self._accuracy_labels: Dict[str, QtWidgets.QLabel] = {}
        for code in self.FAIL_CODE_COLORS:
            bar_row = QtWidgets.QHBoxLayout()
            bar_row.setSpacing(4)

            label = QtWidgets.QLabel(self.FAIL_CODE_LABELS.get(code, code))
            label.setFixedWidth(68)
            label.setStyleSheet(f"color: {self.FAIL_CODE_COLORS[code]}; font-size: 10px;")
            bar_row.addWidget(label)

            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setFixedHeight(12)
            color = self.FAIL_CODE_COLORS[code]
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #444;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #333;
                    color: white;
                    font-size: 9px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 2px;
                }}
            """)
            bar_row.addWidget(bar, stretch=1)

            count_label = QtWidgets.QLabel("0/0")
            count_label.setFixedWidth(36)
            count_label.setStyleSheet("color: #888; font-size: 9px;")
            count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            bar_row.addWidget(count_label)

            accuracy_layout.addLayout(bar_row)
            self._accuracy_bars[code] = bar
            self._accuracy_labels[code] = count_label

        self._accuracy_container.setVisible(False)  # 有数据时才显示
        group_layout.addWidget(self._accuracy_container)

        root_layout.addWidget(group)

    # ------------------------------------------------------------------
    # 公共方法 —— 供引擎/主窗口调用
    # ------------------------------------------------------------------
    def update_rejections(self, rejections: List[Dict]):
        """
        用新的拒绝记录列表刷新卡片。

        每个 dict 预期字段：
            timestamp (str)          – 可读时间 "HH:MM:SS"
            direction (str)          – "LONG" / "SHORT"
            fail_code (str)          – "BLOCK_POS" 等
            price (float)            – 拒绝时价格
            detail_summary (str)     – 关键参数摘要（如 "slope=-0.003"）
            evaluated (bool)         – 是否已评估
            was_correct (bool|None)  – 评估结论
            price_move_pct (float|None) – 价格变动 %
        """
        self._rejections = list(rejections or [])
        self._rebuild_rejection_list()

    def update_gate_scores(self, gate_scores: Dict[str, Dict]):
        """
        刷新门控准确率汇总。

        gate_scores: {fail_code: {correct_count, wrong_count, accuracy, ...}}
        """
        self._gate_scores = dict(gate_scores or {})
        self._refresh_accuracy_bars()

    def add_rejection(self, rec: Dict):
        """追加单条拒绝记录（最多保留 20 条，FIFO）"""
        self._rejections.append(rec)
        if len(self._rejections) > 20:
            self._rejections = self._rejections[-20:]
        self._rebuild_rejection_list()

    def set_suggestions(self, suggestions: List[Dict]):
        """
        设置当前调整建议。

        由外部（引擎/主窗口）调用，传入 RejectionTracker.suggest_threshold_adjustments() 的结果。
        每个 dict 包含: fail_code, param_key, action, action_text, label,
                        current_value, suggested_value, accuracy, reason
        """
        self._suggestions = list(suggestions or [])
        # 有建议时启用按钮（即使评估次数不到20，只要有建议就可点击）
        if self._suggestions:
            self._suggest_btn.setEnabled(True)
            self._suggest_btn.setToolTip(
                f"有 {len(self._suggestions)} 项调整建议可审核\n点击查看详情并手动确认"
            )

    def clear(self):
        """清空所有拒绝记录和准确率"""
        self._rejections.clear()
        self._gate_scores.clear()
        self._suggestions.clear()
        self._rebuild_rejection_list()
        self._refresh_accuracy_bars()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _on_suggest_clicked(self):
        """建议调整按钮点击 → 弹出手动确认对话框"""
        # 也触发外部信号（兼容旧流程：如果外部需要先刷新建议再打开对话框）
        self.suggest_adjustments_requested.emit()

        if not self._suggestions:
            self._suggest_status_label.setText("暂无建议")
            self._suggest_status_label.setStyleSheet("color: #888; font-size: 10px;")
            QtWidgets.QMessageBox.information(
                self,
                "暂无调整建议",
                "当前没有门控阈值调整建议。\n\n"
                "可能的原因：\n"
                "  · 评估次数不足（需至少 20 次）\n"
                "  · 门控准确率处于正常范围（40%~80%）\n"
                "  · 参数已在边界值",
            )
            return

        # 弹出手动确认对话框
        dialog = _AdjustmentConfirmDialog(self._suggestions, self)
        result = dialog.exec()

        if result == QtWidgets.QDialog.DialogCode.Accepted:
            accepted = dialog.get_accepted_adjustments()
            applied_count = 0
            for adj in accepted:
                self.adjustment_confirmed.emit(adj["param_key"], adj["suggested_value"])
                applied_count += 1

            if applied_count > 0:
                self._suggest_status_label.setText(f"已应用 {applied_count} 项")
                self._suggest_status_label.setStyleSheet("color: #089981; font-size: 10px;")
            else:
                self._suggest_status_label.setText("未选择")
                self._suggest_status_label.setStyleSheet("color: #888; font-size: 10px;")
        else:
            self._suggest_status_label.setText("已取消")
            self._suggest_status_label.setStyleSheet("color: #888; font-size: 10px;")

    def _rebuild_rejection_list(self):
        """重建拒绝记录卡片列表"""
        # 移除旧条目（保留最后的 stretch）
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        total = len(self._rejections)
        evaluated = sum(1 for r in self._rejections if r.get("evaluated"))

        self._total_label.setText(f"总拒绝: {total}")
        self._evaluated_label.setText(f"已评估: {evaluated}")

        if total == 0:
            # 显示空状态
            empty = QtWidgets.QLabel("暂无拒绝记录\n系统运行后，被门控拦截的信号将显示在此处")
            empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color: #666; font-size: 11px; padding: 20px;")
            self._list_layout.insertWidget(0, empty)
            self._suggest_btn.setEnabled(False)
            return

        # 逆序展示（最新在上方）
        for rec in reversed(self._rejections):
            card = self._create_rejection_card(rec)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

        self._suggest_btn.setEnabled(evaluated >= 20 or bool(self._suggestions))

    def _create_rejection_card(self, rec: Dict) -> QtWidgets.QFrame:
        """为单条拒绝记录创建一个迷你卡片 widget"""
        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {UI_CONFIG['THEME_SURFACE']};
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px;
            }}
        """)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(6, 4, 6, 4)
        card_layout.setSpacing(2)

        # ── 第一行：时间 | 方向 | 门控徽章 | 评估结果 ──
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)

        # 时间（优先使用可读字符串）
        ts = rec.get("timestamp_str") or rec.get("timestamp", "-")
        # 如果是完整日期时间字符串，只显示时分秒
        ts_display = str(ts)
        if len(ts_display) > 10 and " " in ts_display:
            ts_display = ts_display.split(" ")[-1]  # "HH:MM:SS"
        time_lbl = QtWidgets.QLabel(ts_display)
        time_lbl.setStyleSheet("color: #888; font-size: 10px;")
        row1.addWidget(time_lbl)

        # 方向
        direction = rec.get("direction", "-")
        dir_lbl = QtWidgets.QLabel(direction)
        dir_color = "#089981" if direction == "LONG" else "#f23645"
        dir_lbl.setStyleSheet(f"color: {dir_color}; font-weight: bold; font-size: 10px;")
        row1.addWidget(dir_lbl)

        # 门控徽章
        fail_code = rec.get("fail_code", "")
        badge_color = self.FAIL_CODE_COLORS.get(fail_code, "#888")
        badge_text = self.FAIL_CODE_LABELS.get(fail_code, fail_code)
        badge = QtWidgets.QLabel(badge_text)
        badge.setStyleSheet(f"""
            QLabel {{
                background-color: rgba({self._hex_to_rgb_str(badge_color)}, 0.2);
                color: {badge_color};
                border: 1px solid {badge_color};
                border-radius: 3px;
                padding: 0px 4px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        badge.setToolTip(f"门控代码: {fail_code}")
        row1.addWidget(badge)

        row1.addStretch()

        # 评估结果
        if rec.get("evaluated"):
            was_correct = rec.get("was_correct")
            move_pct = rec.get("price_move_pct", 0.0)
            if was_correct:
                eval_lbl = QtWidgets.QLabel(f"✓ {move_pct:+.2f}%")
                eval_lbl.setStyleSheet("color: #089981; font-size: 10px; font-weight: bold;")
                eval_lbl.setToolTip("正确拒绝（避免了亏损）")
            else:
                eval_lbl = QtWidgets.QLabel(f"✗ {move_pct:+.2f}%")
                eval_lbl.setStyleSheet("color: #f23645; font-size: 10px; font-weight: bold;")
                eval_lbl.setToolTip("错误拒绝（错过了盈利机会）")
            row1.addWidget(eval_lbl)

        card_layout.addLayout(row1)

        # ── 第二行：价格 + 详情摘要 ──
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(6)

        price = rec.get("price_at_rejection") or rec.get("price", 0.0)
        price_lbl = QtWidgets.QLabel(f"${price:,.2f}" if price else "-")
        price_lbl.setStyleSheet("color: #ccc; font-size: 10px;")
        row2.addWidget(price_lbl)

        # 详情摘要：从 detail dict 提取关键信息
        detail_dict = rec.get("detail", {})
        detail_summary = rec.get("detail_summary", "")
        if not detail_summary and isinstance(detail_dict, dict) and detail_dict:
            # 自动生成摘要
            parts = []
            if "pos_score" in detail_dict:
                parts.append(f"评分={detail_dict['pos_score']:.0f}")
            if "slope" in detail_dict:
                parts.append(f"斜率={detail_dict['slope']:+.4f}")
            if "similarity" in detail_dict:
                parts.append(f"匹配={detail_dict['similarity']:.1%}")
            detail_summary = " | ".join(parts) if parts else ""
        if detail_summary:
            detail_lbl = QtWidgets.QLabel(str(detail_summary))
            detail_lbl.setStyleSheet("color: #777; font-size: 9px;")
            detail_lbl.setToolTip(str(detail_dict or detail_summary))
            # 截断过长文本
            if len(str(detail_summary)) > 40:
                detail_lbl.setText(str(detail_summary)[:40] + "...")
            row2.addWidget(detail_lbl)

        row2.addStretch()
        card_layout.addLayout(row2)

        return card

    def _refresh_accuracy_bars(self):
        """刷新底部门控准确率柱状条"""
        has_data = False
        for code, bar in self._accuracy_bars.items():
            score = self._gate_scores.get(code, {})
            correct = score.get("correct_count", 0)
            wrong = score.get("wrong_count", 0)
            total = correct + wrong
            if total > 0:
                has_data = True
                accuracy = int(score.get("accuracy", correct / total) * 100)
                bar.setValue(accuracy)
                self._accuracy_labels[code].setText(f"{correct}/{total}")
            else:
                bar.setValue(0)
                self._accuracy_labels[code].setText("0/0")

        self._accuracy_container.setVisible(has_data)

    @staticmethod
    def _hex_to_rgb_str(hex_color: str) -> str:
        """将 #RRGGBB 转换为 'R, G, B' 字符串（用于 rgba()）"""
        h = hex_color.lstrip('#')
        if len(h) != 6:
            return "136, 136, 136"
        return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


class _AdjustmentConfirmDialog(QtWidgets.QDialog):
    """
    阈值调整确认对话框

    显示所有建议项，用户逐项勾选确认后应用。
    调整仅影响运行时配置，不写入文件。
    """

    def __init__(self, suggestions: List[dict], parent=None):
        super().__init__(parent)
        self._suggestions = suggestions
        self._checkboxes: List[QtWidgets.QCheckBox] = []
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("门控阈值调整建议")
        self.setMinimumWidth(480)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {UI_CONFIG['THEME_BACKGROUND']};
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QLabel {{
                color: {UI_CONFIG['THEME_TEXT']};
            }}
            QCheckBox {{
                color: {UI_CONFIG['THEME_TEXT']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # 警告标题
        warning = QtWidgets.QLabel(
            "⚠️ 以下调整基于门控拒绝后的价格走势统计。\n"
            "请仔细审查后勾选要应用的调整项。调整仅影响当前运行时配置。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "color: #FFD54F; font-size: 11px; padding: 8px; "
            "background-color: rgba(255,213,79,0.1); "
            "border: 1px solid #FFD54F; border-radius: 4px;"
        )
        layout.addWidget(warning)

        # 建议列表
        for i, sug in enumerate(self._suggestions):
            frame = QtWidgets.QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 4px;
                    padding: 4px;
                }
            """)
            f_layout = QtWidgets.QVBoxLayout(frame)
            f_layout.setContentsMargins(8, 6, 8, 6)
            f_layout.setSpacing(4)

            # 勾选框 + 参数名
            action_color = "#FF9800" if sug.get("action") == "loosen" else "#4CAF50"
            action_text = sug.get("action_text", sug.get("action", ""))
            cb = QtWidgets.QCheckBox(f"{sug.get('label', sug.get('param_key', ''))}  ({action_text})")
            cb.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {action_color};")
            f_layout.addWidget(cb)
            self._checkboxes.append(cb)

            # 详情
            accuracy = sug.get("accuracy", 0)
            detail = QtWidgets.QLabel(
                f"参数: {sug.get('param_key', '')}\n"
                f"当前值: {sug.get('current_value', '')}  →  建议值: {sug.get('suggested_value', '')}\n"
                f"门控准确率: {accuracy:.0%}\n"
                f"原因: {sug.get('reason', '')}"
            )
            detail.setStyleSheet("color: #bbb; font-size: 10px; padding-left: 24px;")
            detail.setWordWrap(True)
            f_layout.addWidget(detail)

            layout.addWidget(frame)

        # 按钮
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("取消")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #ccc;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QtWidgets.QPushButton("应用选中的调整")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #089981;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0ab090;
            }
        """)
        apply_btn.clicked.connect(self.accept)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

    def get_accepted_adjustments(self) -> List[dict]:
        """获取用户勾选的调整项"""
        result = []
        for i, cb in enumerate(self._checkboxes):
            if cb.isChecked():
                result.append(self._suggestions[i])
        return result


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
        
        # 使用 QSplitter 让左/中/右三栏可拖拽调整宽度
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #444;
                width: 6px;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #089981;
            }
        """)
        
        # 左侧：控制面板
        self.control_panel = PaperTradingControlPanel()
        self.control_panel.setMinimumWidth(260)
        splitter.addWidget(self.control_panel)
        
        # 中间区域：使用垂直 QSplitter 让用户可调整图表和交易记录大小
        center_vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        center_vertical_splitter.setChildrenCollapsible(False)
        center_vertical_splitter.setHandleWidth(6)
        center_vertical_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #444;
                height: 6px;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #089981;
            }
        """)
        
        # K线图（使用现有的ChartWidget）—— 占主要空间
        from ui.chart_widget import ChartWidget
        self.chart_widget = ChartWidget()
        self.chart_widget.setMinimumHeight(350)
        center_vertical_splitter.addWidget(self.chart_widget)
        
        # 底部区域：交易记录（全宽）
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
        center_vertical_splitter.addWidget(trade_group)
        
        # 设置垂直分割器初始比例：图表 600px，交易记录 200px
        center_vertical_splitter.setSizes([600, 200])
        center_vertical_splitter.setMinimumWidth(300)
        splitter.addWidget(center_vertical_splitter)
        
        # 右侧：状态面板（可拖拽拉宽）
        self.status_panel = PaperTradingStatusPanel()
        splitter.addWidget(self.status_panel)
        
        # 将账户设置与统计移动到持仓页
        if hasattr(self.control_panel, "account_group"):
            self.status_panel.attach_account_group(self.control_panel.account_group)
        
        # 初始比例约 左:中:右 = 1:4:1.2，右侧给够宽避免被挤扁
        splitter.setSizes([280, 600, 380])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)
        layout.addWidget(splitter)
    
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
                "保本": 5,          # 追踪止损保本触发
                "止盈": 6,          # 止盈
                "分段止盈": 6,      # 阶梯止盈部分平仓
                "分段止损": 10,     # 阶梯止损部分平仓
                "脱轨": 7,          # 相似度脱轨
                "信号": 8,          # 信号离场
                "超时": 9,          # 超过最大持仓
                "止损": 10,         # 止损
                "交易所平仓": 8,    # 交易所侧被动平仓（用信号标记颜色）
                "位置翻转": 8,      # 价格位置翻转平仓（用信号标记颜色）
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