"""
R3000 K线图表组件
基于 PyQtGraph 实现高性能 K 线图表，支持标注点可视化
深色主题 + 动态渲染
"""
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

# UTC+8 上海时区
_TZ_SHANGHAI = timezone(timedelta(hours=8))
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UI_CONFIG, BACKTEST_CONFIG


class DateAxisItem(pg.AxisItem):
    """自定义时间轴，将索引转换为日期时间显示"""
    
    def __init__(self, timestamps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps = timestamps or []
        # 深色主题
        self.setTextPen(pg.mkPen(UI_CONFIG["THEME_TEXT"]))
    
    def set_timestamps(self, timestamps):
        """设置时间戳列表"""
        self.timestamps = list(timestamps) if timestamps is not None else []
    
    def tickStrings(self, values, scale, spacing):
        """将索引值转换为时间字符串"""
        strings = []
        for v in values:
            idx = int(v)
            if 0 <= idx < len(self.timestamps):
                ts = self.timestamps[idx]
                try:
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts / 1000, tz=_TZ_SHANGHAI)
                    else:
                        dt = pd.to_datetime(ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=_TZ_SHANGHAI)
                    if spacing > 60:
                        strings.append(dt.strftime('%m-%d %H:%M'))
                    else:
                        strings.append(dt.strftime('%H:%M'))
                except:
                    strings.append('')
            else:
                strings.append('')
        return strings


class CandlestickItem(pg.GraphicsObject):
    """自定义K线绘制项 - 动态窗口化渲染"""
    
    def __init__(self, data=None):
        pg.GraphicsObject.__init__(self)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption)
        
        self._last_range = (None, None)
        self.picture = QtGui.QPicture()
        self._rect = QtCore.QRectF()
        self.data = np.array([])
        
        # 颜色配置 - 币安风格
        self.up_color = UI_CONFIG["CHART_UP_COLOR"]
        self.down_color = UI_CONFIG["CHART_DOWN_COLOR"]
        
        if data is not None and len(data) > 0:
            self.update_data(data)
    
    def update_data(self, data):
        """更新K线数据: [index, open, close, low, high]"""
        self.data = np.array(data)
        self._last_range = (None, None)  # 强制重绘
        
        if len(self.data) > 0:
            lows = self.data[:, 3]
            highs = self.data[:, 4]
            valid_lows = lows[lows > 1e-5]
            valid_highs = highs[highs > 1e-5]
            
            y_min = np.min(valid_lows) if len(valid_lows) > 0 else np.min(lows)
            y_max = np.max(valid_highs) if len(valid_highs) > 0 else np.max(highs)
            
            x_min = np.min(self.data[:, 0])
            x_max = np.max(self.data[:, 0])
            self._rect = QtCore.QRectF(x_min - 1, y_min, x_max - x_min + 2, y_max - y_min)
        else:
            self._rect = QtCore.QRectF()
        self.update()
    
    def _generate_picture(self, x_min, x_max):
        """仅生成当前视口范围内的K线图"""
        self.picture = QtGui.QPicture()
        if len(self.data) == 0:
            return
        
        p = QtGui.QPainter(self.picture)
        
        # 确定绘制范围索引
        indices = np.where((self.data[:, 0] >= x_min - 5) & (self.data[:, 0] <= x_max + 5))[0]
        if len(indices) == 0:
            p.end()
            return
        
        start_idx = indices[0]
        end_idx = indices[-1]
        
        w = 0.8
        if len(self.data) > 1:
            w = (self.data[1][0] - self.data[0][0]) * 0.8  # 增加 K 线之间间隙
        
        visible_data = self.data[start_idx:end_idx + 1]
        
        for row in visible_data:
            t, open_, close, low, high = row[:5]
            color = self.up_color if close >= open_ else self.down_color
            
            # 使用 antialiasing 绘制高质量线条
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            
            # 绘制影线（极细，1.2px，更专业）
            p.setPen(pg.mkPen(color, width=1.2))
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            
            # 绘制实体（币安风格：无边框实心）
            p.setPen(pg.mkPen(color, width=0))
            p.setBrush(pg.mkBrush(color))
            
            body_height = close - open_
            # 即使平价也给一个可见高度
            if abs(body_height) < 1e-8:
                body_height = (high - low) * 0.05 + 1e-9
            
            p.drawRect(QtCore.QRectF(t - w/2, open_, w, body_height))
        
        p.end()
    
    def paint(self, p, *args):
        view_box = self.getViewBox()
        if not view_box:
            return
        
        v_range = view_box.viewRange()[0]
        x_min, x_max = v_range[0], v_range[1]
        
        # 增加缓冲区，减少滚动时的重绘频率
        buffer = 10
        if (self._last_range[0] is None or
            x_min < self._last_range[0] or 
            x_max > self._last_range[1]):
            # 生成一个比可见范围稍大的图像
            self._generate_picture(x_min - buffer, x_max + buffer)
            self._last_range = (x_min - buffer, x_max + buffer)
        
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        return self._rect


class SignalMarker(pg.ScatterPlotItem):
    """LONG/SHORT/EXIT 信号标记 - 使用 ScatterPlotItem 实现固定像素大小"""
    
    def __init__(self):
        super().__init__(pxMode=True)
        self.long_entry_color = UI_CONFIG["CHART_LONG_ENTRY_COLOR"]
        self.long_exit_color = UI_CONFIG["CHART_LONG_EXIT_COLOR"]
        self.short_entry_color = UI_CONFIG["CHART_SHORT_ENTRY_COLOR"]
        self.short_exit_color = UI_CONFIG["CHART_SHORT_EXIT_COLOR"]
        
        self.historical_data = [] # 历史标签（如回测信号）
        self.live_data = []       # 实时信号（如实盘成交）

    def update_signals(self, long_entry=None, long_exit=None, short_entry=None, short_exit=None,
                       take_profit=None, stop_loss=None):
        """批量更新历史信号"""
        data = []
        if long_entry:
            data.extend(self._build_items(long_entry, 1))
        if long_exit:
            data.extend(self._build_items(long_exit, 2))
        if short_entry:
            data.extend(self._build_items(short_entry, -1))
        if short_exit:
            data.extend(self._build_items(short_exit, -2))
        if take_profit:
            data.extend(self._build_items(take_profit, 3))
        if stop_loss:
            data.extend(self._build_items(stop_loss, 4))

        self.historical_data = data
        self._refresh()

    def clear_signals(self):
        """清空所有信号"""
        self.historical_data = []
        self.live_data = []
        self.setData([])

    def add_signal(self, x, y, signal_type):
        """
        添加实时成交信号
        """
        self.live_data.append(self._build_item(x, y, signal_type))
        self._refresh()

    def _refresh(self):
        """合并显示所有数据"""
        self.setData(self.historical_data + self.live_data)

    def _build_items(self, points, signal_type):
        return [self._build_item(x, y, signal_type) for x, y in points]

    def _build_item(self, x, y, signal_type):
        symbol = 't1' if signal_type in (1, 2) else 't'
        if signal_type == 3:
            symbol = 'o'
        if signal_type == 4:
            symbol = 's'

        color = ""
        label = ""

        if signal_type == 1:
            color = self.long_entry_color
            label = "LONG"
        elif signal_type == 2:
            color = self.long_exit_color
            label = "EXIT"
        elif signal_type == -1:
            color = self.short_entry_color
            label = "SHORT"
        elif signal_type == -2:
            color = self.short_exit_color
            label = "EXIT"
        elif signal_type == 3:
            color = UI_CONFIG["CHART_TAKE_PROFIT_COLOR"]
            label = "TP"
        elif signal_type == 4:
            color = UI_CONFIG["CHART_STOP_LOSS_COLOR"]
            label = "SL"

        return {
            'pos': (x, y),
            'size': 10,
            'symbol': symbol,
            'brush': pg.mkBrush(color),
            'pen': pg.mkPen(None),
            'data': label
        }

    def paint(self, painter, option, widget):
        # 先让父类画出三角形
        super().paint(painter, option, widget)

        # 手动画出文字标签（屏幕坐标，避免缩放变形）
        view_box = self.getViewBox()
        
        # 使用合并后的数据绘制
        all_markers = self.historical_data + self.live_data
        if not view_box or not all_markers:
            return

        views = view_box.scene().views() if view_box.scene() else []
        if not views:
            return
        view = views[0]

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setFont(QtGui.QFont('Arial', 8, QtGui.QFont.Weight.Bold))

        for item in all_markers:
            x, y = item['pos']
            label = item['data']
            sp = view_box.mapViewToScene(QtCore.QPointF(x, y))
            wp = view.mapFromScene(sp)

            # 样式：专业小方块/背景
            color = QtGui.QColor(item['brush'].color())
            painter.setPen(QtGui.QPen(color, 1))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(15, 15, 15, 220))) 

            # 根据信号类型决定标签是在上方还是下方
            # 标记点 (x, y) 已经是精确的 High (SHORT) 或 Low (LONG)
            is_above = "SHORT" in label or (label == "EXIT" and item['symbol'] == 't')
            offset_y = 10 if is_above else -24
            
            painter.save()
            painter.resetTransform()
            
            # 绘制圆角标签背景
            rect = QtCore.QRectF(float(wp.x() - 18), float(wp.y() + offset_y), 36, 14)
            painter.drawRoundedRect(rect, 2, 2)
            
            # 绘制文字
            painter.setPen(QtGui.QPen(color))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, label)
            painter.restore()

    def add_tp_sp(self, tp_point=None, sp_point=None):
        """添加止盈止损点 (ScatterPlotItem 不需要这个，由 InfiniteLine 处理)"""
        pass

    def boundingRect(self):
        # 使用父类的边界计算，避免空范围导致不渲染
        return pg.ScatterPlotItem.boundingRect(self)


class ChartWidget(QtWidgets.QWidget):
    """
    K线图表组件 - 深色主题
    
    功能：
    - K 线图显示（深色背景）
    - 成交量子图
    - 买卖信号标记
    - 动态 K 线播放
    - 缩放和平移
    - 十字线
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.df: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.current_display_index = 0
        self._y_range_tick = 0
        self._incremental_signals = False
        self._signal_range_tick = 0
        self._render_stride = 1
        self._last_tp = None
        self._last_sp = None
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化 UI"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 设置深色背景
        self.setStyleSheet(f"background-color: {UI_CONFIG['CHART_BACKGROUND']};")
        
        # 创建 PyQtGraph 图形布局
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground(UI_CONFIG['CHART_BACKGROUND'])
        layout.addWidget(self.graphics_layout)
        
        # 创建时间轴
        self.date_axis = DateAxisItem(orientation='bottom')
        
        # K线图区域
        self.candle_plot = self.graphics_layout.addPlot(
            row=0, col=0,
            axisItems={'bottom': self.date_axis}
        )
        self.candle_plot.setLabel('right', '价格', color=UI_CONFIG['THEME_TEXT'])
        self.candle_plot.showAxis('left', False)
        self.candle_plot.showAxis('right', True)
        self.candle_plot.showGrid(x=True, y=True, alpha=0.3)
        self.candle_plot.setMouseEnabled(x=True, y=True)
        
        # 设置轴颜色
        for axis in ['right', 'bottom']:
            self.candle_plot.getAxis(axis).setTextPen(pg.mkPen(UI_CONFIG['THEME_TEXT']))
            self.candle_plot.getAxis(axis).setPen(pg.mkPen(UI_CONFIG['CHART_GRID_COLOR']))
        
        # K线项
        self.candle_item = CandlestickItem()
        self.candle_plot.addItem(self.candle_item)
        
        # 信号标记
        self.signal_marker = SignalMarker()
        self.candle_plot.addItem(self.signal_marker)
        self.signal_marker.setZValue(20)
        
        # 成交量图区域
        self.volume_plot = self.graphics_layout.addPlot(row=1, col=0)
        self.volume_plot.setLabel('right', '成交量', color=UI_CONFIG['THEME_TEXT'])
        self.volume_plot.showAxis('left', False)
        self.volume_plot.showAxis('right', True)
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setMaximumHeight(100)
        
        # 设置成交量轴颜色
        for axis in ['right', 'bottom']:
            self.volume_plot.getAxis(axis).setTextPen(pg.mkPen(UI_CONFIG['THEME_TEXT']))
            self.volume_plot.getAxis(axis).setPen(pg.mkPen(UI_CONFIG['CHART_GRID_COLOR']))
        
        # 链接 X 轴
        self.volume_plot.setXLink(self.candle_plot)
        
        # 成交量柱状图（红绿色区分）
        self.volume_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush='gray')
        self.volume_plot.addItem(self.volume_bars)

        # === 独立 KDJ 面板（币安风格）===
        self.kdj_plot = self.graphics_layout.addPlot(row=2, col=0)
        self.kdj_plot.setLabel('right', 'KDJ', color=UI_CONFIG['THEME_TEXT'])
        self.kdj_plot.showAxis('left', False)
        self.kdj_plot.showAxis('right', True)
        self.kdj_plot.showAxis('bottom', False)
        self.kdj_plot.showGrid(x=True, y=True, alpha=0.2)
        self.kdj_plot.setMaximumHeight(80)
        self.kdj_plot.setXLink(self.candle_plot)
        self.kdj_plot.setYRange(0, 100, padding=0.05)
        for axis in ['right']:
            self.kdj_plot.getAxis(axis).setTextPen(pg.mkPen(UI_CONFIG['THEME_TEXT']))
            self.kdj_plot.getAxis(axis).setPen(pg.mkPen(UI_CONFIG['CHART_GRID_COLOR']))
        
        # KDJ 曲线：K(白/浅蓝) D(黄) J(紫)
        self.k_curve = pg.PlotDataItem(pen=pg.mkPen('#87CEEB', width=1.2))  # 浅蓝
        self.d_curve = pg.PlotDataItem(pen=pg.mkPen('#FFD700', width=1.2))  # 金黄
        self.j_curve = pg.PlotDataItem(pen=pg.mkPen('#FF00FF', width=1.2))  # 紫色
        self.kdj_plot.addItem(self.k_curve)
        self.kdj_plot.addItem(self.d_curve)
        self.kdj_plot.addItem(self.j_curve)
        
        # 超买超卖参考线
        self.kdj_plot.addItem(pg.InfiniteLine(pos=80, angle=0, pen=pg.mkPen('#555', width=0.5, style=QtCore.Qt.PenStyle.DashLine)))
        self.kdj_plot.addItem(pg.InfiniteLine(pos=20, angle=0, pen=pg.mkPen('#555', width=0.5, style=QtCore.Qt.PenStyle.DashLine)))

        # === 独立 MACD 面板（币安风格）===
        self.macd_plot = self.graphics_layout.addPlot(row=3, col=0)
        self.macd_plot.setLabel('right', 'MACD', color=UI_CONFIG['THEME_TEXT'])
        self.macd_plot.showAxis('left', False)
        self.macd_plot.showAxis('right', True)
        self.macd_plot.showAxis('bottom', False)
        self.macd_plot.showGrid(x=True, y=True, alpha=0.2)
        self.macd_plot.setMaximumHeight(80)
        self.macd_plot.setXLink(self.candle_plot)
        for axis in ['right']:
            self.macd_plot.getAxis(axis).setTextPen(pg.mkPen(UI_CONFIG['THEME_TEXT']))
            self.macd_plot.getAxis(axis).setPen(pg.mkPen(UI_CONFIG['CHART_GRID_COLOR']))
        
        # MACD 柱状图（红绿色）- 使用两个 BarGraphItem
        self.macd_hist_pos = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#089981')  # 绿色
        self.macd_hist_neg = pg.BarGraphItem(x=[], height=[], width=0.6, brush='#f23645')  # 红色
        self.macd_plot.addItem(self.macd_hist_pos)
        self.macd_plot.addItem(self.macd_hist_neg)
        
        # MACD 曲线：DIF(白) DEA(黄)
        self.macd_curve = pg.PlotDataItem(pen=pg.mkPen('#FFFFFF', width=1.2))     # DIF 白色
        self.macd_signal_curve = pg.PlotDataItem(pen=pg.mkPen('#FFD700', width=1.2))  # DEA 金黄
        self.macd_plot.addItem(self.macd_curve)
        self.macd_plot.addItem(self.macd_signal_curve)
        
        # 零轴线
        self.macd_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#555', width=0.5)))
        
        # === 指纹轨迹叠加（虚线）===
        self.fingerprint_traj_curve = pg.PlotDataItem(
            pen=pg.mkPen('#FFD700', width=0.9, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.fingerprint_traj_curve.setZValue(6)
        self.candle_plot.addItem(self.fingerprint_traj_curve)
        # 置信区间（上下界 + 填充）
        self.fingerprint_upper_curve = pg.PlotDataItem(pen=pg.mkPen('#FFD700', width=0.6))
        self.fingerprint_lower_curve = pg.PlotDataItem(pen=pg.mkPen('#FFD700', width=0.6))
        self.fingerprint_band = pg.FillBetweenItem(
            self.fingerprint_upper_curve,
            self.fingerprint_lower_curve,
            brush=pg.mkBrush(255, 215, 0, 40),
        )
        self.fingerprint_upper_curve.setZValue(5)
        self.fingerprint_lower_curve.setZValue(5)
        self.fingerprint_band.setZValue(4)
        self.candle_plot.addItem(self.fingerprint_upper_curve)
        self.candle_plot.addItem(self.fingerprint_lower_curve)
        self.candle_plot.addItem(self.fingerprint_band)
        
        self.fingerprint_label = pg.TextItem(anchor=(0, 0), color="#FFD700")
        self.fingerprint_label.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Weight.Bold))
        self.candle_plot.addItem(self.fingerprint_label, ignoreBounds=True)
        
        self._overlay_future_padding = 0
        
        # 【重要改动】TP / SL 水平虚线（InfiniteLine）- 延伸整个可视区域
        # 使用虚线样式 + 标签，清晰显示止盈止损位置
        tp_color = UI_CONFIG["CHART_TAKE_PROFIT_COLOR"]
        sl_color = UI_CONFIG["CHART_STOP_LOSS_COLOR"]
        
        self.tp_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(tp_color, width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            label='TP: {value:.2f}',
            labelOpts={'position': 0.95, 'color': tp_color, 'fill': (30, 30, 30, 200), 'movable': False}
        )
        self.sl_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(sl_color, width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            label='SL: {value:.2f}',
            labelOpts={'position': 0.95, 'color': sl_color, 'fill': (30, 30, 30, 200), 'movable': False}
        )
        
        self.tp_line.setVisible(False)
        self.sl_line.setVisible(False)
        self.tp_line.setZValue(10)  # 高于K线但低于十字线
        self.sl_line.setZValue(10)
        
        self.candle_plot.addItem(self.tp_line, ignoreBounds=True)
        self.candle_plot.addItem(self.sl_line, ignoreBounds=True)
        
        # 十字线
        self.vline = pg.InfiniteLine(angle=90, movable=False, 
                                      pen=pg.mkPen(UI_CONFIG['THEME_TEXT'], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False, 
                                      pen=pg.mkPen(UI_CONFIG['THEME_TEXT'], width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.candle_plot.addItem(self.vline, ignoreBounds=True)
        self.candle_plot.addItem(self.hline, ignoreBounds=True)
        self.vline.setVisible(False)
        self.hline.setVisible(False)
        
        # 鼠标事件
        self.candle_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        # === 币安风格叠加层 ===
        # 1. 实时价格线 (带右侧标)
        self.price_line = pg.InfiniteLine(angle=0, movable=False, 
                                        pen=pg.mkPen("#888", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.candle_plot.addItem(self.price_line, ignoreBounds=True)
        self.price_label = pg.TextItem(anchor=(0, 0.5), color="#eee")
        self.price_label.setFont(QtGui.QFont('Arial', 9, QtGui.QFont.Weight.Bold))
        self.candle_plot.addItem(self.price_label, ignoreBounds=True)

        # 2. 信息看板 (左上角 OHLCV)
        self.overlay_layout = QtWidgets.QGraphicsGridLayout()
        self.info_item = pg.GraphicsWidget()
        self.info_item.setLayout(self.overlay_layout)
        self.candle_plot.scene().addItem(self.info_item)
        
        # 使用 TextItem 模拟顶栏信息
        self.ohlc_text = pg.TextItem(anchor=(0, 0), color="#aaa")
        self.ohlc_text.setFont(QtGui.QFont('Consolas', 10, QtGui.QFont.Weight.Bold))
        self.candle_plot.addItem(self.ohlc_text, ignoreBounds=True)
        
        # 十字线价格标签
        self.crosshair_price_label = pg.TextItem(anchor=(1, 0.5), fill=(40, 40, 40, 230))
        self.candle_plot.addItem(self.crosshair_price_label, ignoreBounds=True)
        self.crosshair_price_label.setVisible(False)
    
    def set_data(self, df: pd.DataFrame, labels: pd.Series = None, show_all: bool = True):
        """
        全量设置数据（会清空当前状态）
        """
        self.df = df
        self.labels = labels
        
        # 仅在真正切换品种或手动重置时清空信号
        if show_all:
            self.signal_marker.clear_signals()
            self._last_tp = None
            self._last_sp = None
            # 清空 TP/SL 虚线
            self.set_tp_sl_lines(None, None)
        
        if df is None or len(df) == 0:
            return

        # 更新时间轴
        if 'timestamp' in df.columns:
            self.date_axis.set_timestamps(df['timestamp'].tolist())
        elif 'open_time' in df.columns:
            self.date_axis.set_timestamps(df['open_time'].tolist())
        
        if show_all:
            self._incremental_signals = False
            self._display_range(0, len(df))
            self.candle_plot.autoRange()
            self._update_tp_sp_segment()
        else:
            self._incremental_signals = True
            self.current_display_index = 0
            self._y_range_tick = 0
            self._signal_range_tick = 0
            # 从首根K线开始播放
            self._display_range(0, 1)
            self._update_tp_sp_segment()

    def update_kline(self, df: pd.DataFrame):
        """
        增量更新K线（流畅更新，不停轴，不跳变）
        """
        if df is None or df.empty:
            return
            
        old_len = len(self.df) if self.df is not None else 0
        self.df = df
        
        # 更新时间轴（支持 timestamp 或 open_time）
        if 'timestamp' in df.columns:
            self.date_axis.set_timestamps(df['timestamp'].tolist())
        elif 'open_time' in df.columns:
            self.date_axis.set_timestamps(df['open_time'].tolist())
        
        # 仅渲染末尾变化的区域
        self._display_range(0, len(df))
        
        # 智能缩放 (Hysteresis Scaling)
        self._smart_auto_scale()

    def _smart_auto_scale(self):
        """
        稳健的自动缩放：避免频繁跳动，增加视觉稳定性
        """
        if self.df is None or self.df.empty:
            return
            
        # 获取当前视图范围
        view_box = self.candle_plot.getViewBox()
        x_range = view_box.viewRange()[0]
        y_range = view_box.viewRange()[1]
        
        display_start = max(0, int(x_range[0]))
        display_end = min(len(self.df), int(x_range[1]) + 1)
        
        if display_end <= display_start:
            return
            
        df_visible = self.df.iloc[display_start:display_end]
        actual_min = df_visible['low'].min()
        actual_max = df_visible['high'].max()
        
        # 基础边距 (2%)
        padding = (actual_max - actual_min) * 0.1
        target_min = actual_min - padding
        target_max = actual_max + padding
        
        # 稳定性检查：回滞机制 (Hysteresis)
        curr_min, curr_max = y_range
        
        # 如果当前价格已经超出当前视图，或者余量严重不足 (<5%)，则触发缩放
        # 或者如果余量过于宽敞 (>60%)，为了美观也适当收紧
        margin_top = (curr_max - actual_max) / (actual_max - actual_min + 1e-9)
        margin_bottom = (actual_min - curr_min) / (actual_max - actual_min + 1e-9)
        
        needs_upscale = actual_max > curr_max * 0.98 or actual_min < curr_min * 1.02
        needs_downscale = margin_top > 0.6 or margin_bottom > 0.6
        
        if needs_upscale or needs_downscale:
            # 引入平滑因子或直接设置目标范围并带上 15% 缓冲区
            self.candle_plot.setYRange(target_min, target_max, padding=0)
    
    def _display_range(self, start_idx: int, end_idx: int):
        """显示指定范围的 K 线 (已优化：移除激进截断，支持历史滚动)"""
        if self.df is None or end_idx <= start_idx:
            return
        
        # 渲染最近 1000 根 K 线，保证用户可以回滚查看，同时兼顾性能
        history_limit = 1000
        actual_start = max(0, end_idx - history_limit)
        df_slice = self.df.iloc[actual_start:end_idx]
        n = len(df_slice)
        
        # 准备K线数据: [index, open, close, low, high]
        candle_data = np.zeros((n, 5))
        # 使用真实的整数索引
        candle_data[:, 0] = np.arange(actual_start, actual_start + n)
        candle_data[:, 1] = df_slice['open'].values
        candle_data[:, 2] = df_slice['close'].values
        candle_data[:, 3] = df_slice['low'].values
        candle_data[:, 4] = df_slice['high'].values
        
        # 更新K线
        self.candle_item.update_data(candle_data)
        
        # 更新成交量
        if 'volume' in df_slice.columns:
            x = np.arange(actual_start, actual_start + n)
            height = df_slice['volume'].values
            colors = []
            for i in range(len(df_slice)):
                c = df_slice['close'].values[i]
                o = df_slice['open'].values[i]
                color_hex = UI_CONFIG["CHART_UP_COLOR"] if c >= o else UI_CONFIG["CHART_DOWN_COLOR"]
                qcolor = QtGui.QColor(color_hex)
                qcolor.setAlpha(180) # 略微透明，更专业
                colors.append(pg.mkBrush(qcolor))
            
            self.volume_bars.setOpts(x=x, height=height, brushes=colors, width=0.8)
        
        # 更新实时价格线
        last_price = self.df['close'].iloc[-1]
        self.price_line.setPos(last_price)
        self._update_ohlc_text(len(self.df)-1)

        # === 实时渲染 KDJ 指标（独立面板）===
        if 'k' in df_slice.columns:
            indices = np.arange(actual_start, actual_start + n)
            self.k_curve.setData(x=indices, y=df_slice['k'].values)
            self.d_curve.setData(x=indices, y=df_slice['d'].values)
            self.j_curve.setData(x=indices, y=df_slice['j'].values)
        
        # === 实时渲染 MACD 指标（独立面板）===
        if 'macd' in df_slice.columns and 'macd_hist' in df_slice.columns:
            indices = np.arange(actual_start, actual_start + n)
            macd_vals = df_slice['macd'].values
            signal_vals = df_slice['macd_signal'].values
            hist_vals = df_slice['macd_hist'].values
            
            # MACD DIF/DEA 曲线
            self.macd_curve.setData(x=indices, y=macd_vals)
            self.macd_signal_curve.setData(x=indices, y=signal_vals)
            
            # MACD 柱状图（红绿分离）
            pos_mask = hist_vals >= 0
            neg_mask = hist_vals < 0
            
            pos_x = indices[pos_mask]
            pos_h = hist_vals[pos_mask]
            neg_x = indices[neg_mask]
            neg_h = hist_vals[neg_mask]
            
            self.macd_hist_pos.setOpts(x=pos_x, height=pos_h, width=0.6)
            self.macd_hist_neg.setOpts(x=neg_x, height=neg_h, width=0.6)
            
            # 自动调整 MACD 面板 Y 轴范围
            all_vals = np.concatenate([macd_vals, signal_vals, hist_vals])
            y_min, y_max = all_vals.min(), all_vals.max()
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
            self.macd_plot.setYRange(y_min - margin, y_max + margin, padding=0)
        
        # 更新信号标记
        if self.labels is not None:
            if not self._incremental_signals:
                self._update_signal_markers_range(actual_start, end_idx)
        self._update_tp_sp_segment()
        
        self.current_display_index = end_idx
    
    def advance_one_candle(self) -> bool:
        """
        前进一根 K 线
        
        Returns:
            True 如果还有更多数据，False 如果已到达末尾
        """
        if self.df is None:
            return False
        
        if self.current_display_index >= len(self.df):
            return False
        
        self.current_display_index += 1
        self._signal_range_tick += 1

        # 渲染降频（高倍速时减少重绘）
        if self._render_stride > 1 and (self.current_display_index % self._render_stride != 0):
            return self.current_display_index < len(self.df)

        self._display_range(0, self.current_display_index)
        
        # 自动滚动视图 - 保持右侧留白
        visible_range = 40
        # 始终让 X 轴流动：右侧对齐当前K线
        self.candle_plot.setXRange(
            self.current_display_index - visible_range + 1,
            self.current_display_index + 1,
            padding=0
        )
        self._update_tp_sp_segment()
        
        # 自动调整Y轴范围：使用 smart 缩放替代简单的 setYRange
        self._y_range_tick += 1
        if self.current_display_index > 0 and (self._y_range_tick % 10 == 0 or self.current_display_index <= visible_range):
            self._smart_auto_scale()
        
        return self.current_display_index < len(self.df)

    def set_render_stride(self, speed: int):
        """根据速度降低渲染频率以减少卡顿"""
        self._render_stride = 1
    
    def add_signal_at(self, idx: int, label_type: int, df: pd.DataFrame = None):
        """
        在指定位置添加信号标记
        
        Args:
            idx: K线索引
            label_type: 1=LONG_ENTRY, 2=LONG_EXIT, -1=SHORT_ENTRY, -2=SHORT_EXIT
            df: 数据源（可选，默认使用self.df）
        """
        if df is None:
            df = self.df
        if df is None or idx < 0 or idx >= len(df):
            return
        
        # 根据信号类型确定 Y 位置
        if label_type == 1:  # LONG_ENTRY - 在低点下方
            y = df['low'].iloc[idx]
            # 翻转策略：低点入场做多，同时也意味着之前的空头离场
            # 我们在同一位置补一个 EXIT 标记
            self.signal_marker.add_signal(idx, y, -2) # SHORT_EXIT
        elif label_type == -1:  # SHORT_ENTRY - 在高点上方
            y = df['high'].iloc[idx]
            # 翻转策略：高点入场做空，同时也意味着之前的多头离场
            self.signal_marker.add_signal(idx, y, 2) # LONG_EXIT
        elif label_type == 2:  # LONG_EXIT - 在高点上方
            y = df['high'].iloc[idx]
        elif label_type == -2:  # SHORT_EXIT - 在低点下方
            y = df['low'].iloc[idx]
        else:
            return
        
        self.signal_marker.add_signal(idx, y, label_type)
        
        # 入场时同步更新 TP/SP（基于近端高低点）
        if label_type in (1, -1) and 'close' in df.columns:
            tp_price, sp_price = self._calc_tp_sp(idx, label_type, df)
            if tp_price is not None and sp_price is not None:
                self._set_tp_sp(tp_price, sp_price)

    def set_tp_sl_lines(self, tp_price: float = None, sl_price: float = None):
        """
        设置/更新止盈止损虚线位置
        
        Args:
            tp_price: 止盈价格，None则隐藏
            sl_price: 止损价格，None则隐藏
        """
        if tp_price is not None:
            self.tp_line.setValue(tp_price)
            self.tp_line.setVisible(True)
        else:
            self.tp_line.setVisible(False)
        
        if sl_price is not None:
            self.sl_line.setValue(sl_price)
            self.sl_line.setVisible(True)
        else:
            self.sl_line.setVisible(False)
    
    def _set_tp_sp(self, tp_price: float, sp_price: float):
        """兼容旧接口，内部转发到新方法"""
        self.set_tp_sl_lines(tp_price, sp_price)

    def _calc_tp_sp(self, idx: int, label_type: int, df: pd.DataFrame):
        """基于近端高低点计算 TP/SP（短线逻辑）"""
        lookback = int(BACKTEST_CONFIG.get("TP_SP_LOOKBACK", 20))
        start = max(0, idx - lookback + 1)
        if start >= idx + 1:
            return None, None

        highs = df['high'].iloc[start:idx + 1]
        lows = df['low'].iloc[start:idx + 1]
        if len(highs) == 0 or len(lows) == 0:
            return None, None

        recent_high = float(highs.max())
        recent_low = float(lows.min())
        entry_price = float(df['close'].iloc[idx])

        if label_type == 1:  # LONG_ENTRY
            tp_price = max(entry_price, recent_high)
            sp_price = min(entry_price, recent_low)
        else:  # SHORT_ENTRY
            tp_price = min(entry_price, recent_low)
            sp_price = max(entry_price, recent_high)

        return tp_price, sp_price

    def _update_tp_sp_segment(self):
        """
        【已废弃】旧的短线段更新方法，现在改用 InfiniteLine 虚线
        保留此方法以兼容旧代码调用，实际上不做任何操作
        新代码请使用 set_tp_sl_lines() 方法
        """
        pass  # InfiniteLine 不需要每次viewRange变化时更新
    
    def _update_signal_markers_range(self, start_idx: int, end_idx: int):
        """更新指定范围内的信号标记 - LONG/SHORT/EXIT 系统"""
        long_entry = []
        long_exit = []
        short_entry = []
        short_exit = []
        last_tp = None
        last_sp = None
        
        for i in range(start_idx, min(end_idx, len(self.labels))):
            label = self.labels.iloc[i]
            if label == 1:  # LONG_ENTRY
                long_entry.append((i, self.df['low'].iloc[i]))
                # 翻转策略补离场标记
                short_exit.append((i, self.df['low'].iloc[i]))
                last_tp, last_sp = self._calc_tp_sp(i, 1, self.df)
            elif label == 2:  # LONG_EXIT
                long_exit.append((i, self.df['high'].iloc[i]))
            elif label == -1:  # SHORT_ENTRY
                short_entry.append((i, self.df['high'].iloc[i]))
                # 翻转策略补离场标记
                long_exit.append((i, self.df['high'].iloc[i]))
                last_tp, last_sp = self._calc_tp_sp(i, -1, self.df)
            elif label == -2:  # SHORT_EXIT
                short_exit.append((i, self.df['low'].iloc[i]))
        
        self.signal_marker.update_signals(long_entry, long_exit, short_entry, short_exit)
        if last_tp is not None and last_sp is not None:
            self._set_tp_sp(last_tp, last_sp)
    
    def _update_signal_markers(self, df: pd.DataFrame, labels: pd.Series):
        """更新信号标记 - LONG/SHORT/EXIT 系统"""
        long_entry = []
        long_exit = []
        short_entry = []
        short_exit = []
        last_tp = None
        last_sp = None
        
        for i, label in enumerate(labels):
            if label == 1:  # LONG_ENTRY
                long_entry.append((i, df['low'].iloc[i]))
                # 翻转策略补离场标记
                short_exit.append((i, df['low'].iloc[i]))
                last_tp, last_sp = self._calc_tp_sp(i, 1, df)
            elif label == 2:  # LONG_EXIT
                long_exit.append((i, df['high'].iloc[i]))
            elif label == -1:  # SHORT_ENTRY
                short_entry.append((i, df['high'].iloc[i]))
                # 翻转策略补离场标记
                long_exit.append((i, df['high'].iloc[i]))
                last_tp, last_sp = self._calc_tp_sp(i, -1, df)
            elif label == -2:  # SHORT_EXIT
                short_exit.append((i, df['low'].iloc[i]))
        
        self.signal_marker.update_signals(long_entry, long_exit, short_entry, short_exit)
        if last_tp is not None and last_sp is not None:
            self._set_tp_sp(last_tp, last_sp)
    
    def set_labels(self, labels: pd.Series):
        """设置标注"""
        self.labels = labels
        if self.df is not None:
            self._update_signal_markers(self.df, labels)
            self._update_tp_sp_segment()
    
    def reset_playback(self):
        """重置播放位置"""
        self.current_display_index = 0
        if self.df is not None:
            self._display_range(0, 1)
    
    def _on_mouse_moved(self, pos):
        """鼠标移动：更新十字线与信息看板"""
        if self.df is None or len(self.df) == 0:
            return
            
        vb = self.candle_plot.getViewBox()
        if self.candle_plot.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            index = int(mouse_point.x())
            
            if 0 <= index < len(self.df):
                # 更新十字线
                self.vline.setPos(mouse_point.x())
                self.hline.setPos(mouse_point.y())
                self.vline.setVisible(True)
                self.hline.setVisible(True)
                
                # 更新十字线旁边的价格标签
                view_range = vb.viewRange()
                self.crosshair_price_label.setPos(view_range[0][1], mouse_point.y())
                self.crosshair_price_label.setHtml(f'<div style="background-color: #333; color: white; padding: 2px;">{mouse_point.y():.2f}</div>')
                self.crosshair_price_label.setVisible(True)
                
                # 更新顶部信息
                self._update_ohlc_text(index)
            else:
                self.vline.setVisible(False)
                self.hline.setVisible(False)
                self.crosshair_price_label.setVisible(False)
        else:
            self.vline.setVisible(False)
            self.hline.setVisible(False)
            self.crosshair_price_label.setVisible(False)

    def _update_ohlc_text(self, index):
        """更新左上角 OHLCV 文本"""
        if self.df is None or index < 0 or index >= len(self.df):
            return
            
        row = self.df.iloc[index]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        v = row.get('volume', 0)
        
        # 计算涨跌幅
        if index > 0:
            pc = self.df.iloc[index-1]['close']
            chg = (c - pc) / (pc + 1e-9)
        else:
            chg = 0
            
        color = UI_CONFIG["CHART_UP_COLOR"] if c >= o else UI_CONFIG["CHART_DOWN_COLOR"]
        
        ts = ""
        if 'timestamp' in self.df.columns:
            ts_val = self.df.iloc[index]['timestamp']
            try:
                dt = datetime.fromtimestamp(float(ts_val) / 1000, tz=_TZ_SHANGHAI)
                ts = dt.strftime('%Y/%m/%d %H:%M')
            except:
                ts = str(ts_val)
            
        html = f'<span style="color: #888;">{ts}</span> '
        html += f'<span style="color: #aaa;">O:</span><span style="color: {color};">{o:,.2f}</span> '
        html += f'<span style="color: #aaa;">H:</span><span style="color: {color};">{h:,.2f}</span> '
        html += f'<span style="color: #aaa;">L:</span><span style="color: {color};">{l:,.2f}</span> '
        html += f'<span style="color: #aaa;">C:</span><span style="color: {color};">{c:,.2f}</span> '
        html += f'<span style="color: #aaa;">Chg:</span><span style="color: {color};">{chg:+.2%}</span> '
        html += f'<span style="color: #aaa;">V:</span><span style="color: #888;">{v:,.0f}</span>'
        
        # 固定在当前视图左上角
        rect = self.candle_plot.viewRect()
        self.ohlc_text.setPos(rect.left(), rect.top())
        self.ohlc_text.setHtml(html)
        
        # 实时价格线标签放在右侧轴上
        last_c = self.df['close'].iloc[-1]
        self.price_label.setPos(rect.right(), last_c)
        self.price_label.setHtml(f'<div style="background-color: {color}; color: white; padding: 1px 4px; border-radius: 2px;">{last_c:,.2f}</div>')
    
    def get_data_time_range(self) -> Tuple[str, str]:
        """获取数据的时间范围"""
        if self.df is None or len(self.df) == 0:
            return "", ""
        
        start_time = ""
        end_time = ""
        
        time_col = None
        for col in ['timestamp', 'open_time', 'time']:
            if col in self.df.columns:
                time_col = col
                break
        
        if time_col:
            try:
                start_ts = self.df[time_col].iloc[0]
                end_ts = self.df[time_col].iloc[-1]
                
                if isinstance(start_ts, (int, float, np.integer, np.floating)):
                    start_time = datetime.fromtimestamp(float(start_ts) / 1000, tz=_TZ_SHANGHAI).strftime('%Y-%m-%d %H:%M')
                    end_time = datetime.fromtimestamp(float(end_ts) / 1000, tz=_TZ_SHANGHAI).strftime('%Y-%m-%d %H:%M')
                else:
                    start_dt = pd.to_datetime(start_ts)
                    end_dt = pd.to_datetime(end_ts)
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.tz_localize(_TZ_SHANGHAI)
                    else:
                        start_dt = start_dt.tz_convert(_TZ_SHANGHAI)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.tz_localize(_TZ_SHANGHAI)
                    else:
                        end_dt = end_dt.tz_convert(_TZ_SHANGHAI)
                    start_time = start_dt.strftime('%Y-%m-%d %H:%M')
                    end_time = end_dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        return start_time, end_time
    
    def zoom_to_range(self, start: int, end: int):
        """缩放到指定范围"""
        self.candle_plot.setXRange(start, end, padding=0.02)
    
    def auto_range(self):
        """自动范围"""
        self.candle_plot.autoRange()
        self.volume_plot.autoRange()
    
    def set_fingerprint_trajectory(self, prices: np.ndarray, start_idx: int,
                                   similarity: float, label: str,
                                   lower: np.ndarray = None, upper: np.ndarray = None):
        """在K线图上叠加指纹轨迹虚线"""
        if prices is None or len(prices) == 0:
            self.clear_fingerprint_trajectory()
            return
        
        x = np.arange(start_idx, start_idx + len(prices))
        self.fingerprint_traj_curve.setData(x=x, y=prices)
        if lower is not None and upper is not None and len(lower) == len(prices) and len(upper) == len(prices):
            self.fingerprint_lower_curve.setData(x=x, y=lower)
            self.fingerprint_upper_curve.setData(x=x, y=upper)
        else:
            self.fingerprint_lower_curve.setData([], [])
            self.fingerprint_upper_curve.setData([], [])
        
        if self.df is not None and len(self.df) > 0:
            max_x = int(x[-1])
            self._overlay_future_padding = max(0, max_x - (len(self.df) - 1))
        else:
            self._overlay_future_padding = 0
        
        sim_pct = similarity * 100
        color = "#00FF00" if sim_pct >= 70 else ("#FFD700" if sim_pct >= 50 else "#FF6347")
        self.fingerprint_label.setText(f"{label} {sim_pct:.1f}%")
        self.fingerprint_label.setColor(color)
        
        rect = self.candle_plot.viewRect()
        self.fingerprint_label.setPos(rect.left() + 2, rect.top() + 20)
    
    def clear_fingerprint_trajectory(self):
        """清除指纹轨迹叠加"""
        self.fingerprint_traj_curve.setData([], [])
        self.fingerprint_upper_curve.setData([], [])
        self.fingerprint_lower_curve.setData([], [])
        self.fingerprint_label.setText("")
        self._overlay_future_padding = 0
    
    def get_overlay_padding(self) -> int:
        """获取轨迹对未来的X轴扩展长度"""
        return int(self._overlay_future_padding or 0)


# 测试代码
if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 深色主题
    app.setStyle('Fusion')
    
    # 创建测试数据
    np.random.seed(42)
    n = 500
    
    test_df = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(n) * 50),
    })
    test_df['close'] = test_df['open'] + np.random.randn(n) * 30
    test_df['high'] = test_df[['open', 'close']].max(axis=1) + abs(np.random.randn(n) * 20)
    test_df['low'] = test_df[['open', 'close']].min(axis=1) - abs(np.random.randn(n) * 20)
    test_df['volume'] = np.random.randint(100, 10000, n)
    
    labels = pd.Series(np.zeros(n))
    labels.iloc[50] = 1
    labels.iloc[100] = -1
    labels.iloc[200] = 1
    labels.iloc[300] = -1
    
    chart = ChartWidget()
    chart.set_data(test_df, labels)
    chart.setWindowTitle("R3000 K线图表测试")
    chart.resize(1200, 600)
    chart.show()
    
    sys.exit(app.exec())
