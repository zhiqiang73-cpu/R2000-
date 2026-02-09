"""
R3000 K线图表组件
基于 PyQtGraph 实现高性能 K 线图表，支持标注点可视化
深色主题 + 动态渲染
"""
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple
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
                        dt = datetime.fromtimestamp(ts / 1000)
                    else:
                        dt = pd.to_datetime(ts)
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
        
        # K线宽度
        w = 0.8
        if len(self.data) > 1:
            w = min((self.data[1][0] - self.data[0][0]) * 0.8, 4.0)
        
        visible_data = self.data[start_idx:end_idx + 1]
        
        for row in visible_data:
            t, open_, close, low, high = row[:5]
            color = self.up_color if close >= open_ else self.down_color
            
            # 绘制影线
            p.setPen(pg.mkPen(color, width=1))
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            
            # 绘制实体
            p.setPen(pg.mkPen(color, width=1))
            p.setBrush(pg.mkBrush(color))
            
            body_height = close - open_
            if abs(body_height) < 1e-5:
                body_height = (high - low) * 0.01 if high != low else 0.1
            
            rect_w = w * 0.6
            p.drawRect(QtCore.QRectF(t - rect_w/2, open_, rect_w, body_height))
        
        p.end()
    
    def paint(self, p, *args):
        view_box = self.getViewBox()
        if not view_box:
            return
        
        v_range = view_box.viewRange()[0]
        x_min, x_max = v_range[0], v_range[1]
        
        if (self._last_range[0] is None or
            abs(x_min - self._last_range[0]) > 2 or
            abs(x_max - self._last_range[1]) > 2):
            self._generate_picture(x_min, x_max)
            self._last_range = (x_min, x_max)
        
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        return self._rect


class SignalMarker(pg.ScatterPlotItem):
    """LONG/SHORT/EXIT 信号标记 - 使用 ScatterPlotItem 实现固定像素大小"""
    
    def __init__(self):
        super().__init__(pxMode=True)  # 关键：pxMode=True 保证标记大小不随缩放改变
        self.long_entry_color = UI_CONFIG["CHART_LONG_ENTRY_COLOR"]
        self.long_exit_color = UI_CONFIG["CHART_LONG_EXIT_COLOR"]
        self.short_entry_color = UI_CONFIG["CHART_SHORT_ENTRY_COLOR"]
        self.short_exit_color = UI_CONFIG["CHART_SHORT_EXIT_COLOR"]
        
        self.all_data = []

    def update_signals(self, long_entry=None, long_exit=None, short_entry=None, short_exit=None,
                       take_profit=None, stop_loss=None):
        """更新信号点（批量更新，避免逐点 setData 卡顿）"""
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

        self.all_data = data
        self.setData(self.all_data)

    def clear_signals(self):
        """清空所有信号"""
        self.all_data = []
        self.setData([])

    def add_signal(self, x, y, signal_type):
        """
        添加单个信号
        signal_type: 1=LONG_ENTRY, 2=LONG_EXIT, -1=SHORT_ENTRY, -2=SHORT_EXIT, 3=TP, 4=SL
        """
        self.all_data.append(self._build_item(x, y, signal_type))
        self.setData(self.all_data)

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
        if not view_box or not self.all_data:
            return

        views = view_box.scene().views() if view_box.scene() else []
        if not views:
            return
        view = views[0]

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setFont(QtGui.QFont('Arial', 8, QtGui.QFont.Weight.Bold))

        for item in self.all_data:
            x, y = item['pos']
            label = item['data']
            sp = view_box.mapViewToScene(QtCore.QPointF(x, y))
            wp = view.mapFromScene(sp)

            painter.setPen(QtGui.QPen(QtGui.QColor(item['brush'].color())))
            offset_y = -16 if ("LONG" in label or (label == "EXIT" and item['symbol'] == 't1')) else 16

            painter.save()
            painter.resetTransform()
            wp_f = QtCore.QPointF(wp)
            painter.drawText(wp_f + QtCore.QPointF(8, float(offset_y)), label)
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
        
        # 信息标签
        self.info_label = pg.TextItem(anchor=(0, 0), color=UI_CONFIG['THEME_TEXT'])
        self.info_label.setFont(QtGui.QFont('Consolas', 10))
        self.candle_plot.addItem(self.info_label)
    
    def set_data(self, df: pd.DataFrame, labels: pd.Series = None, show_all: bool = True):
        """
        设置数据
        
        Args:
            df: K线数据 DataFrame
            labels: 标注序列（可选）
            show_all: True=显示全部数据, False=准备动态播放
        """
        self.df = df
        self.labels = labels
        
        # 清空信号标记
        self.signal_marker.clear_signals()
        self._last_tp = None
        self._last_sp = None
        self._update_tp_sp_segment()
        
        if df is None or len(df) == 0:
            return

        # 仅在设置数据时更新时间轴，避免每帧更新带来的卡顿
        if 'timestamp' in df.columns:
            self.date_axis.set_timestamps(df['timestamp'].tolist())
        elif 'open_time' in df.columns:
            self.date_axis.set_timestamps(df['open_time'].tolist())
        
        if show_all:
            self._incremental_signals = False
            self._display_range(0, len(df))
            # 自动缩放
            self.candle_plot.autoRange()
            self._update_tp_sp_segment()
        else:
            self._incremental_signals = True
            # 动画播放模式 - 从第一根K线开始
            self.current_display_index = 1
            self._display_range(0, 1)
            
            # 设置初始视图范围
            visible_range = 60
            self.candle_plot.setXRange(0, visible_range + 10, padding=0)
            
            # 设置Y轴范围 - 使用前1000根数据的范围
            sample_size = min(1000, len(df))
            y_min = df['low'].iloc[:sample_size].min() * 0.998
            y_max = df['high'].iloc[:sample_size].max() * 1.002
            self.candle_plot.setYRange(y_min, y_max, padding=0.02)
            self._update_tp_sp_segment()
    
    def _display_range(self, start_idx: int, end_idx: int):
        """显示指定范围的 K 线"""
        if self.df is None or end_idx <= start_idx:
            return
        
        visible_range = 60
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
            colors = np.where(
                df_slice['close'].values >= df_slice['open'].values,
                UI_CONFIG["CHART_UP_COLOR"],
                UI_CONFIG["CHART_DOWN_COLOR"]
            )
            self.volume_bars.setOpts(x=x, height=height, brushes=colors)
        
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
        visible_range = 60
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
        """鼠标移动事件"""
        if self.df is None:
            return
        
        mouse_point = self.candle_plot.vb.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = mouse_point.y()
        
        if 0 <= x < len(self.df):
            self.vline.setPos(x)
            self.hline.setPos(y)
            self.vline.setVisible(True)
            self.hline.setVisible(True)
            
            row = self.df.iloc[x]
            info_text = f"O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f}"
            if 'volume' in self.df.columns:
                info_text += f" V:{row['volume']:.0f}"
            
            self.info_label.setText(info_text)
            self.info_label.setPos(x + 5, self.df['high'].max())
        else:
            self.vline.setVisible(False)
            self.hline.setVisible(False)
    
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
                
                if isinstance(start_ts, (int, float)):
                    start_time = datetime.fromtimestamp(start_ts / 1000).strftime('%Y-%m-%d %H:%M')
                    end_time = datetime.fromtimestamp(end_ts / 1000).strftime('%Y-%m-%d %H:%M')
                else:
                    start_time = pd.to_datetime(start_ts).strftime('%Y-%m-%d %H:%M')
                    end_time = pd.to_datetime(end_ts).strftime('%Y-%m-%d %H:%M')
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
