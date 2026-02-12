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
        
        # K线宽度 —— 更粗更清晰
        w = 0.8
        if len(self.data) > 1:
            w = min((self.data[1][0] - self.data[0][0]) * 0.85, 6.0)
        
        visible_data = self.data[start_idx:end_idx + 1]
        
        for row in visible_data:
            t, open_, close, low, high = row[:5]
            color = self.up_color if close >= open_ else self.down_color
            
            # 绘制影线（极细，更专业）
            p.setPen(pg.mkPen(color, width=1.0))
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            
            # 绘制实体（币安风格：无边框实心）
            p.setPen(pg.mkPen(color, width=0))
            p.setBrush(pg.mkBrush(color))
            
            body_height = close - open_
            # 即使平价也给一个极小的厚度，保证可见
            if abs(body_height) < 1e-6:
                body_height = 0.01 * (high - low + 1e-9)
            
            rect_w = w * 0.9
            p.drawRect(QtCore.QRectF(t - rect_w/2, open_, rect_w, body_height))
        
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

            # 样式优化：不仅显示文字，还增加背景框增加专业感
            color = QtGui.QColor(item['brush'].color())
            painter.setPen(QtGui.QPen(color, 1))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 20, 20, 200))) # 半透明深色背景

            offset_y = -22 if ("LONG" in label or (label == "EXIT" and item['symbol'] == 't1')) else 12
            
            painter.save()
            painter.resetTransform()
            
            # 计算背景框
            rect = QtCore.QRectF(float(wp.x() - 20), float(wp.y() + offset_y), 40, 14)
            painter.drawRoundedRect(rect, 3, 3)
            
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
        self.candle_plot.setLabel('left', '价格', color=UI_CONFIG['THEME_TEXT'])
        self.candle_plot.showGrid(x=True, y=True, alpha=0.3)
        self.candle_plot.setMouseEnabled(x=True, y=True)
        
        # 设置轴颜色
        for axis in ['left', 'bottom']:
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
        self.volume_plot.setLabel('left', '成交量', color=UI_CONFIG['THEME_TEXT'])
        self.volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self.volume_plot.setMaximumHeight(100)
        
        # 设置成交量轴颜色
        for axis in ['left', 'bottom']:
            self.volume_plot.getAxis(axis).setTextPen(pg.mkPen(UI_CONFIG['THEME_TEXT']))
            self.volume_plot.getAxis(axis).setPen(pg.mkPen(UI_CONFIG['CHART_GRID_COLOR']))
        
        # 链接 X 轴
        self.volume_plot.setXLink(self.candle_plot)
        
        # 成交量柱状图
        self.volume_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush='gray')
        self.volume_plot.addItem(self.volume_bars)
        
        # TP / SP 短线段 + 文字（清晰但不遮挡K线）
        tp_color = QtGui.QColor(UI_CONFIG["CHART_TAKE_PROFIT_COLOR"])
        sp_color = QtGui.QColor(UI_CONFIG["CHART_STOP_LOSS_COLOR"])
        self.tp_segment = pg.PlotDataItem([], [], pen=pg.mkPen(tp_color, width=1.6))
        self.sp_segment = pg.PlotDataItem([], [], pen=pg.mkPen(sp_color, width=1.6))
        self.candle_plot.addItem(self.tp_segment)
        self.candle_plot.addItem(self.sp_segment)
        self.tp_segment.setZValue(5)
        self.sp_segment.setZValue(5)

        self.tp_label = pg.TextItem(text="TP", color=tp_color, anchor=(1, 0.5))
        self.sp_label = pg.TextItem(text="SP", color=sp_color, anchor=(1, 0.5))
        self.candle_plot.addItem(self.tp_label)
        self.candle_plot.addItem(self.sp_label)
        self.tp_label.setVisible(False)
        self.sp_label.setVisible(False)
        
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
            self._update_tp_sp_segment()
        
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
        
        # 更新时间轴
        if 'timestamp' in df.columns:
            self.date_axis.set_timestamps(df['timestamp'].tolist())
        
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
        
        # 稳定性检查：如果当前范围已经包含了目标范围，且余量充足(>30%)，则不更新
        curr_min, curr_max = y_range
        margin_top = (curr_max - actual_max) / (actual_max - actual_min + 1e-5)
        margin_bottom = (actual_min - curr_min) / (actual_max - actual_min + 1e-5)
        
        # 如果价格超出了当前范围，或者余量太小(<5%)，则需要平滑调整
        if actual_max > curr_max or actual_min < curr_min or margin_top < 0.05 or margin_bottom < 0.05:
            # 立即调整到目标范围，带上充足余量
            self.candle_plot.setYRange(target_min, target_max, padding=0)
        elif margin_top > 0.5 or margin_bottom > 0.5:
            # 如果余量实在太大，也收紧一下
            self.candle_plot.setYRange(target_min, target_max, padding=0)
    
    def _display_range(self, start_idx: int, end_idx: int):
        """显示指定范围的 K 线"""
        if self.df is None or end_idx <= start_idx:
            return
        
        visible_range = 40
        display_start = max(start_idx, end_idx - visible_range + 1)
        df_slice = self.df.iloc[display_start:end_idx]
        n = len(df_slice)
        
        # 准备K线数据: [index, open, close, low, high]
        candle_data = np.zeros((n, 5))
        candle_data[:, 0] = np.arange(display_start, display_start + n)
        candle_data[:, 1] = df_slice['open'].values
        candle_data[:, 2] = df_slice['close'].values
        candle_data[:, 3] = df_slice['low'].values
        candle_data[:, 4] = df_slice['high'].values
        
        # 更新K线
        self.candle_item.update_data(candle_data)
        
        # 更新成交量
        if 'volume' in df_slice.columns:
            x = np.arange(display_start, display_start + n)
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
        
        # 更新信号标记
        if self.labels is not None:
            if not self._incremental_signals:
                self._update_signal_markers_range(display_start, end_idx)
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
        
        # 自动调整Y轴范围（降低频率减少卡顿）
        self._y_range_tick += 1
        if self.current_display_index > 0 and (self._y_range_tick % 5 == 0 or self.current_display_index <= visible_range):
            start = max(0, self.current_display_index - visible_range)
            end = self.current_display_index
            df_visible = self.df.iloc[start:end]
            if len(df_visible) > 0:
                y_min = df_visible['low'].min() * 0.999
                y_max = df_visible['high'].max() * 1.001
                self.candle_plot.setYRange(y_min, y_max, padding=0.02)
        
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

    def _set_tp_sp(self, tp_price: float, sp_price: float):
        """设置 TP/SP 数值并刷新短线段"""
        self._last_tp = tp_price
        self._last_sp = sp_price
        self._update_tp_sp_segment()

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
        """用短线段显示 TP/SP，避免遮挡K线"""
        if self.df is None or self._last_tp is None or self._last_sp is None:
            self.tp_segment.setData([], [])
            self.sp_segment.setData([], [])
            self.tp_label.setVisible(False)
            self.sp_label.setVisible(False)
            return

        view_range = self.candle_plot.viewRange()[0]
        x_end = view_range[1]
        x_start = max(view_range[0], x_end - 12)

        self.tp_segment.setData([x_start, x_end], [self._last_tp, self._last_tp])
        self.sp_segment.setData([x_start, x_end], [self._last_sp, self._last_sp])

        self.tp_label.setPos(x_end - 0.5, self._last_tp)
        self.sp_label.setPos(x_end - 0.5, self._last_sp)
        self.tp_label.setVisible(True)
        self.sp_label.setVisible(True)
    
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
            chg = (c - pc) / pc
        else:
            chg = 0
            
        # 币安风格色码
        color = UI_CONFIG["CHART_UP_COLOR"] if c >= o else UI_CONFIG["CHART_DOWN_COLOR"]
        
        # 格式化文本
        ts = ""
        if 'timestamp' in self.df.columns:
            ts_val = self.df.iloc[index]['timestamp']
            try:
                if isinstance(ts_val, (int, float, np.integer, np.floating)):
                    dt = datetime.fromtimestamp(float(ts_val) / 1000, tz=_TZ_SHANGHAI)
                else:
                    dt = pd.to_datetime(ts_val)
                    if dt.tzinfo is None:
                        dt = dt.tz_localize(_TZ_SHANGHAI)
                    else:
                        dt = dt.tz_convert(_TZ_SHANGHAI)
                ts = dt.strftime('%Y/%m/%d %H:%M')
            except Exception:
                ts = str(ts_val)
            
        html = f'<span style="color: #aaa;">{ts}</span> '
        html += f'<span style="color: #eee;">开:</span> <span style="color: {color};">{o:,.2f}</span> '
        html += f'<span style="color: #eee;">高:</span> <span style="color: {color};">{h:,.2f}</span> '
        html += f'<span style="color: #eee;">低:</span> <span style="color: {color};">{l:,.2f}</span> '
        html += f'<span style="color: #eee;">收:</span> <span style="color: {color};">{c:,.2f}</span> '
        html += f'<span style="color: #eee;">幅:</span> <span style="color: {color};">{chg:+.2%}</span> '
        html += f'<span style="color: #eee;">量:</span> <span style="color: #FFB300;">{v:,.2f}</span>'
        
        # 将文本放置在左上角适当位置
        view_range = self.candle_plot.getViewBox().viewRange()
        self.ohlc_text.setPos(view_range[0][0], view_range[1][1])
        self.ohlc_text.setHtml(html)
        
        # 更新实时价格标签位置
        last_c = self.df['close'].iloc[-1]
        self.price_label.setPos(view_range[0][1], last_c)
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
