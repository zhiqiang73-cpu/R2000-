"""
R3000 主窗口
PyQt6 主窗口：深色主题、动态 K 线播放、标注可视化
"""
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from typing import Optional, List
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
                    PAPER_TRADING_CONFIG, WF_EVOLUTION_CONFIG, DEEPSEEK_CONFIG)
from ui.chart_widget import ChartWidget
from ui.control_panel import ControlPanel
from ui.analysis_panel import AnalysisPanel
from ui.optimizer_panel import OptimizerPanel
from ui.paper_trading_tab import PaperTradingTab, PaperTradingTradeLog
from ui.adaptive_learning_tab import AdaptiveLearningTab
from ui.signal_analysis_tab import SignalAnalysisTab
from core.adaptive_controller import AdaptiveController, TradeContext as AdaptiveTradeContext
from core.deepseek_reviewer import DeepSeekReviewer, TradeContext as DeepSeekTradeContext


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


class PaperTradingStartWorker(QtCore.QObject):
    """后台线程：创建 LiveTradingEngine + start()，避免阻塞 UI"""
    succeeded = QtCore.pyqtSignal(object)   # engine
    failed    = QtCore.pyqtSignal(str)      # error message
    progress  = QtCore.pyqtSignal(str)      # status text

    def __init__(self, build_fn):
        super().__init__()
        self._build_fn = build_fn

    def run(self):
        try:
            engine, ok = self._build_fn(self.progress.emit)
            if ok:
                self.succeeded.emit(engine)
            else:
                self.failed.emit("无法启动模拟交易，请检查网络连接。")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class TradeHistoryIOWorker(QtCore.QObject):
    """后台线程：历史交易文件读/删/写"""
    loaded = QtCore.pyqtSignal(list)            # history list
    deleted = QtCore.pyqtSignal(int, int)       # (removed_count, remaining_count)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, history_file: str, action: str, order=None):
        super().__init__()
        self._history_file = history_file
        self._action = action
        self._order = order

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if self._action == "load":
                history = load_trade_history_from_file(self._history_file) or []
                self.loaded.emit(history)
                return

            if self._action == "delete":
                existing_history = load_trade_history_from_file(self._history_file) or []
                filtered_history = [
                    o for o in existing_history
                    if not self._is_same_order(o, self._order)
                ]
                save_trade_history_to_file(filtered_history, self._history_file)
                removed = len(existing_history) - len(filtered_history)
                self.deleted.emit(removed, len(filtered_history))
                return

            raise ValueError(f"Unknown action: {self._action}")
        except Exception as e:
            self.failed.emit(str(e))

    @staticmethod
    def _is_same_order(order1, order2) -> bool:
        """判断两个订单是否相同（供后台线程使用）"""
        if order1 is None or order2 is None:
            return False
        id1 = getattr(order1, "order_id", None)
        id2 = getattr(order2, "order_id", None)
        if id1 and id2 and id1 == id2:
            return True
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


class SignalBacktestWorker(QtCore.QObject):
    """信号回测工作者 - 使用策略池信号驱动回测，动画接口与 LabelingWorker 完全一致"""
    step_completed    = QtCore.pyqtSignal(int)
    label_found       = QtCore.pyqtSignal(int, int)
    labeling_progress = QtCore.pyqtSignal(str)
    labels_ready      = QtCore.pyqtSignal(object)
    finished          = QtCore.pyqtSignal(object)   # dict: {bt_result, metrics}
    rt_update         = QtCore.pyqtSignal(dict, list)   # (running_metrics, completed_trades)
    error             = QtCore.pyqtSignal(str)

    def __init__(self, df):
        super().__init__()
        self.df = df
        self.labels = None
        self.labeler = None   # 保持与 LabelingWorker 接口一致，避免 _on_labels_ready 中 AttributeError
        self.speed  = UI_CONFIG["DEFAULT_SPEED"]
        self._stop_requested  = False
        self._pause_requested = False
        self.is_running = False
        self._labels_ready = False
        self.strict_state_filter = False  # 严格市场状态过滤：⚠状态禁止开仓
        self.use_alt_tpsl = False  # 回测试验 TP/SL 开关

    @QtCore.pyqtSlot()
    def run_backtest(self):
        """后台预计算 + 前台动画循环"""
        try:
            import threading as _threading
            import pandas as _pd
            from utils.indicators import calculate_all_indicators
            from core.signal_store import get_cumulative
            from core.signal_analyzer import (
                _build_condition_arrays,
                LONG_TP1_PCT, LONG_SL_PCT,
                SHORT_TP1_PCT, SHORT_SL_PCT,
                MAX_HOLD,
            )
            from core.market_state_detector import detect_state
            from core.backtester import TradeRecord, BacktestResult, Position, PositionSide
            from config import PAPER_TRADING_CONFIG, BACKTEST_CONFIG

            # 回测试验 TP/SL（仅影响回测，不改配置/策略池）
            long_tp_pct = LONG_TP1_PCT
            long_sl_pct = LONG_SL_PCT
            short_tp_pct = SHORT_TP1_PCT
            short_sl_pct = SHORT_SL_PCT
            if self.use_alt_tpsl:
                long_tp_pct = 0.004
                long_sl_pct = 0.005
                short_tp_pct = 0.004
                short_sl_pct = 0.005

            n = len(self.df)
            INITIAL_CAP = float(PAPER_TRADING_CONFIG.get("DEFAULT_BALANCE", 5000.0))
            self.is_running = True
            self._stop_requested = False
            self._pause_requested = False
            self._labels_ready = False
            self._precompute_ready = False
            self._precompute_failed = False
            self._df_work = None
            self._cond = None
            self._valid = None

            def _make_running_metrics(records, init_cap, current_pos=None):
                total_trades = len(records)
                wins   = [r for r in records if r.profit > 0]
                losses = [r for r in records if r.profit <= 0]
                total_profit = sum(r.profit for r in records)
                final_cap = init_cap + total_profit
                total_return_pct = ((final_cap - init_cap) / init_cap * 100.0) if init_cap else 0.0
                gross_profit = sum(r.profit for r in wins)
                gross_loss = abs(sum(r.profit for r in losses))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
                avg_profit_pct = (
                    sum(r.profit_pct for r in records) / total_trades if total_trades else 0.0
                )
                avg_hold_periods = (
                    sum(r.hold_periods for r in records) / total_trades if total_trades else 0.0
                )
                long_recs  = [r for r in records if r.side ==  1]
                short_recs = [r for r in records if r.side == -1]
                long_wins  = [r for r in long_recs  if r.profit > 0]
                short_wins = [r for r in short_recs if r.profit > 0]
                long_profit = sum(r.profit for r in long_recs)
                short_profit = sum(r.profit for r in short_recs)
                cap2, peak, max_dd = init_cap, init_cap, 0.0
                for r in records:
                    cap2 += r.profit
                    peak = max(peak, cap2)
                    dd = (peak - cap2) / peak * 100.0 if peak > 0 else 0.0
                    if dd > max_dd:
                        max_dd = dd

                out = {
                    "initial_capital": init_cap,
                    "total_trades": total_trades,
                    "win_rate": len(wins) / total_trades if total_trades else 0.0,
                    "total_return": total_return_pct / 100.0,
                    "total_profit": total_profit,
                    "max_drawdown": max_dd,
                    "sharpe_ratio": 0.0,
                    "profit_factor": profit_factor,
                    "long_win_rate": len(long_wins) / len(long_recs) if long_recs else 0.0,
                    "long_profit": long_profit,
                    "short_win_rate": len(short_wins) / len(short_recs) if short_recs else 0.0,
                    "short_profit": short_profit,
                    "current_pos": current_pos,
                    "last_trade": records[-1] if records else None,
                    "avg_profit_pct": avg_profit_pct,
                    "avg_hold_periods": avg_hold_periods,
                }
                return out

            def precompute():
                try:
                    self.labeling_progress.emit("正在计算所有指标...")
                    df_work = calculate_all_indicators(self.df.copy())
                    cumulative = get_cumulative()
                    valid = {k: v for k, v in cumulative.items()
                             if v.get('appear_rounds', 0) >= 3}
                    if not valid:
                        self._precompute_failed = True
                        self.error.emit("策略池为空，请先完成信号分析")
                        return

                    cond = {
                        'long':  _build_condition_arrays(df_work, 'long'),
                        'short': _build_condition_arrays(df_work, 'short'),
                    }

                    self._df_work = df_work
                    self._cond = cond
                    self._valid = valid
                    self._precompute_ready = True
                    self.labeling_progress.emit("预计算完成，正在播放...")
                except Exception as exc:
                    self._precompute_failed = True
                    self.error.emit(str(exc) + "\n" + traceback.format_exc())

            precompute_thread = _threading.Thread(target=precompute, daemon=True)
            precompute_thread.start()

            labels_arr = None
            df_work = None
            cond = None
            valid = None
            prep_initialized = False

            trade_records = []
            trade_logs = []
            capital = INITIAL_CAP
            LEVERAGE = int(PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20))
            PCT = float(PAPER_TRADING_CONFIG.get("POSITION_SIZE_PCT", 0.05))
            FEE = float(BACKTEST_CONFIG.get("FEE_RATE", 0.0004))
            in_pos = False
            e_price = e_idx = e_dir = tp = sl = None
            e_margin = None
            e_key = None
            e_info = None
            # 动态止损追踪状态
            peak_pnl_pct = 0.0           # 持仓期间杠杆后峰值盈亏%
            ever_reached_threshold = False  # 是否曾触达保本阈值
            pos_stage = 1                # 1=阶段1(固定SL+时间衰减), 2=阶段2(追踪止损)
            peak_price = 0.0             # 阶段2追踪用峰值价格
            stage2_margin = 0.0          # 阶段2剩余保证金
            
            def _append_trade_log(event, idx, side, price, info=None):
                trade_logs.append({
                    "idx": idx,
                    "event": event,
                    "side": "LONG" if side == 'long' else ("SHORT" if side == 'short' else ""),
                    "price": price,
                    "info": info or {},
                })
                if len(trade_logs) > 800:
                    trade_logs[:] = trade_logs[-800:]

            cur = 0
            while self.is_running and not self._stop_requested and cur < n:
                while self._pause_requested and not self._stop_requested:
                    time.sleep(0.05)
                if self._stop_requested:
                    break
                if self._precompute_failed:
                    break

                if self._precompute_ready and not prep_initialized:
                    df_work = self._df_work
                    cond = self._cond
                    valid = self._valid
                    labels_arr = np.zeros(len(df_work), dtype=int)
                    self.labels = _pd.Series(labels_arr, index=df_work.index, copy=False)
                    self._labels_ready = True
                    self.labels_ready.emit(self.labels)
                    prep_initialized = True
                    self.labeling_progress.emit("正在逐 bar 扫描信号...")

                if prep_initialized and cur >= 50 and cur < len(df_work):
                    row = df_work.iloc[cur]
                    hi, lo, cl = float(row['high']), float(row['low']), float(row['close'])
                    adx   = row.get('adx')
                    slope = row.get('ma5_slope')

                    if in_pos:
                        _cfg   = PAPER_TRADING_CONFIG
                        is_long = (e_dir == 'long')
                        hold_bars = cur - e_idx
                        _lev  = LEVERAGE

                        # ── 更新峰值浮盈（用于保本/时间衰减判断）──────────────
                        best_bar = hi if is_long else lo
                        bar_pnl = (best_bar - e_price) / e_price * (_lev if is_long else -_lev) * 100
                        if bar_pnl > peak_pnl_pct:
                            peak_pnl_pct = bar_pnl

                        # ── 动态止损更新 ──────────────────────────────────────
                        if pos_stage == 1:
                            bp_thr  = float(_cfg.get("BREAKEVEN_THRESHOLD_PCT", 6.0))
                            bp_sl   = float(_cfg.get("BREAKEVEN_SL_PCT", -3.0))
                            td_en   = bool(_cfg.get("TIME_DECAY_ENABLED", True))
                            bar1    = int(_cfg.get("TIME_DECAY_BAR_1", 120))
                            bar2    = int(_cfg.get("TIME_DECAY_BAR_2", 180))
                            sl1_pnl = float(_cfg.get("TIME_DECAY_SL_1", -10.0))
                            sl2_pnl = float(_cfg.get("TIME_DECAY_SL_2", -5.0))

                            if not ever_reached_threshold and peak_pnl_pct >= bp_thr:
                                # 保本触发：浮盈达到阈值，止损上移
                                ever_reached_threshold = True
                                bp_sl_price = (e_price * (1.0 + bp_sl / 100.0 / _lev) if is_long
                                               else e_price * (1.0 - bp_sl / 100.0 / _lev))
                                if (is_long and bp_sl_price > sl) or (not is_long and bp_sl_price < sl):
                                    sl = bp_sl_price
                                    _append_trade_log("breakeven_sl", cur, e_dir, sl, {
                                        "reason": f"浮盈峰值{peak_pnl_pct:.1f}%≥{bp_thr:.0f}%→止损上移至{bp_sl:.0f}%"
                                    })
                            elif not ever_reached_threshold and td_en:
                                # 时间衰减：持仓过久自动收窄止损
                                target_pnl, td_label = None, ""
                                if hold_bars >= bar2:
                                    target_pnl = sl2_pnl
                                    td_label = f"持仓{hold_bars}根≥{bar2}根→收窄至{sl2_pnl:.0f}%"
                                elif hold_bars >= bar1:
                                    target_pnl = sl1_pnl
                                    td_label = f"持仓{hold_bars}根≥{bar1}根→收窄至{sl1_pnl:.0f}%"
                                if target_pnl is not None:
                                    new_sl = (e_price * (1.0 + target_pnl / 100.0 / _lev) if is_long
                                              else e_price * (1.0 - target_pnl / 100.0 / _lev))
                                    if (is_long and new_sl > sl) or (not is_long and new_sl < sl):
                                        if hold_bars in (bar1, bar2):  # 首次跨过阈值时记录日志
                                            _append_trade_log("time_decay_sl", cur, e_dir, new_sl, {
                                                "reason": f"⏱ {td_label}"
                                            })
                                        sl = new_sl

                        elif pos_stage == 2:
                            # 阶段2：追踪止损（跟随最高/最低价）
                            trail_pct = float(_cfg.get("TRAILING_STOP_PCT", 0.08))
                            if is_long:
                                if hi > peak_price:
                                    peak_price = hi
                                sl = max(sl, peak_price * (1.0 - trail_pct))
                            else:
                                if lo < peak_price:
                                    peak_price = lo
                                sl = min(sl, peak_price * (1.0 + trail_pct))

                        # ── 判断出场条件（用更新后的 sl 计算）───────────────────
                        hit_tp = (pos_stage == 1) and (
                            (is_long and hi >= tp) or (not is_long and lo <= tp)
                        )
                        hit_sl = (is_long and lo <= sl) or (not is_long and hi >= sl)
                        timed  = hold_bars >= MAX_HOLD

                        if hit_tp and not hit_sl:
                            # ── 阶段1 TP1 命中：分批止盈 70%，剩余 30% 追踪 ──
                            tp1_ratio    = float(_cfg.get("STAGED_TP_RATIO_1", 0.70))
                            remain_ratio = 1.0 - tp1_ratio
                            active_margin  = e_margin or (capital * PCT)
                            partial_margin = active_margin * tp1_ratio
                            pnl_pct = (
                                ((tp - e_price) / e_price if is_long else (e_price - tp) / e_price) - FEE * 2
                            )
                            pnl = pnl_pct * partial_margin * LEVERAGE
                            capital += pnl
                            _append_trade_log("tp1_partial", cur, e_dir, tp, {
                                "profit_pct": round(pnl_pct * 100.0, 2),
                                "hold_bars": hold_bars,
                                "entry_price": round(e_price, 2),
                                "stop_loss": sl,
                                "take_profit": tp,
                                "note": f"TP1分批平仓{tp1_ratio:.0%}→剩余{remain_ratio:.0%}进追踪阶段",
                            })
                            trade_records.append(TradeRecord(
                                entry_idx=e_idx, exit_idx=cur,
                                side=1 if is_long else -1,
                                entry_price=e_price, exit_price=tp,
                                size=partial_margin * LEVERAGE / e_price,
                                profit=pnl, profit_pct=pnl_pct * 100,
                                hold_periods=hold_bars, exit_reason="tp1_partial",
                                stop_loss=sl or 0.0,
                                take_profit=tp or 0.0,
                                liquidation_price=0.0,
                                margin=partial_margin,
                                signal_key=e_key or "",
                                signal_rate=(e_info or {}).get('overall_rate', 0.0),
                                signal_score=(e_info or {}).get('综合评分', 0.0),
                            ))
                            # 进入阶段2：SL 上移至开仓价保本，追踪止损从 TP1 开始
                            pos_stage  = 2
                            e_margin   = active_margin * remain_ratio
                            peak_price = tp
                            trail_pct  = float(_cfg.get("TRAILING_STOP_PCT", 0.08))
                            sl = max(e_price, peak_price * (1.0 - trail_pct)) if is_long \
                                 else min(e_price, peak_price * (1.0 + trail_pct))
                            # 发射剩余仓位持仓更新
                            cur_pos = Position(
                                side=PositionSide.LONG if is_long else PositionSide.SHORT,
                                entry_price=e_price, entry_idx=e_idx,
                                size=e_margin * LEVERAGE / e_price,
                                stop_loss=sl, take_profit=tp, liquidation_price=0.0,
                                margin=e_margin,
                            )
                            try:
                                cur_pos.signal_key   = e_key or ""
                                cur_pos.signal_rate  = (e_info or {}).get('overall_rate', 0.0)
                                cur_pos.signal_score = (e_info or {}).get('综合评分', 0.0)
                            except Exception:
                                pass
                            m = _make_running_metrics(trade_records, INITIAL_CAP, cur_pos)
                            m["current_bar"] = cur
                            m["trade_logs"]  = list(trade_logs)
                            self.rt_update.emit(m, list(trade_records))

                        elif hit_sl or timed:
                            # ── SL 触发 / 超时平仓（含阶段2追踪止损触发）────────
                            active_margin = e_margin or (capital * PCT)
                            x_price = sl if hit_sl else cl
                            reason  = ('trailing_sl' if pos_stage == 2 else 'sl') if hit_sl else 'timeout'
                            pnl_pct = (
                                ((x_price - e_price) / e_price if is_long else (e_price - x_price) / e_price) - FEE * 2
                            )
                            pnl = pnl_pct * active_margin * LEVERAGE
                            capital += pnl
                            labels_arr[cur] = 2 if is_long else -2
                            self.label_found.emit(cur, int(labels_arr[cur]))
                            _append_trade_log(reason, cur, e_dir, x_price, {
                                "profit_pct": round(pnl_pct * 100.0, 2),
                                "hold_bars": hold_bars,
                                "entry_price": round(e_price, 2),
                                "stop_loss": sl,
                                "take_profit": tp,
                            })
                            trade_records.append(TradeRecord(
                                entry_idx=e_idx, exit_idx=cur,
                                side=1 if is_long else -1,
                                entry_price=e_price, exit_price=x_price,
                                size=active_margin * LEVERAGE / e_price,
                                profit=pnl, profit_pct=pnl_pct * 100,
                                hold_periods=hold_bars, exit_reason=reason,
                                stop_loss=sl or 0.0,
                                take_profit=tp or 0.0,
                                liquidation_price=0.0,
                                margin=active_margin,
                                signal_key=e_key or "",
                                signal_rate=(e_info or {}).get('overall_rate', 0.0),
                                signal_score=(e_info or {}).get('综合评分', 0.0),
                            ))
                            in_pos = False
                            e_margin = None
                            self.rt_update.emit(
                                {**_make_running_metrics(trade_records, INITIAL_CAP), "trade_logs": list(trade_logs)},
                                list(trade_records),
                            )

                        else:
                            # ── 持仓中：每步刷新持仓和交易明细 ─────────────────
                            cur_pos = Position(
                                side=PositionSide.LONG if is_long else PositionSide.SHORT,
                                entry_price=e_price, entry_idx=e_idx,
                                size=(e_margin or (capital * PCT)) * LEVERAGE / e_price,
                                stop_loss=sl, take_profit=tp, liquidation_price=0.0,
                                margin=e_margin or (capital * PCT),
                            )
                            try:
                                cur_pos.signal_key   = e_key or ""
                                cur_pos.signal_rate  = (e_info or {}).get('overall_rate', 0.0)
                                cur_pos.signal_score = (e_info or {}).get('综合评分', 0.0)
                            except Exception:
                                pass
                            m = _make_running_metrics(trade_records, INITIAL_CAP, cur_pos)
                            m["current_bar"] = cur
                            m["trade_logs"]  = list(trade_logs)
                            self.rt_update.emit(m, list(trade_records))

                        self.step_completed.emit(cur)
                        cur += 1
                        sleep_time = max(0.001, 10.0 / max(1, self.speed))
                        time.sleep(sleep_time)
                        continue

                    if _pd.isna(adx) or _pd.isna(slope):
                        self.step_completed.emit(cur)
                        cur += 1
                        sleep_time = max(0.001, 10.0 / max(1, self.speed))
                        time.sleep(sleep_time)
                        continue

                    state = detect_state(float(adx), float(slope))
                    # 方向预过滤：多头趋势只做多，空头趋势只做空，震荡市两边均可
                    if state == '多头趋势':
                        _allowed = {'long'}
                    elif state == '空头趋势':
                        _allowed = {'short'}
                    else:
                        _allowed = {'long', 'short'}
                    _triggered = []
                    for key, entry in valid.items():
                        d = entry.get('direction', 'long')
                        if d not in _allowed:
                            continue
                        conds = entry.get('conditions', [])
                        bkd   = (entry.get('market_state_breakdown') or {}).get(state, {})
                        if bkd.get('total_triggers', 0) == 0:
                            continue
                        arr = cond.get(d, {})
                        if all(bool(arr.get(c, [False])[cur]) for c in conds if c in arr):
                            state_wr = bkd.get('avg_rate', 0.0)
                            # 严格市场状态过滤：若该状态下触发次数≥5且命中率低于候选门槛（⚠），则跳过
                            if self.strict_state_filter:
                                _thresh = 0.64 if d == 'long' else 0.52
                                _trig   = bkd.get('total_triggers', 0)
                                if _trig >= 5 and state_wr < _thresh:
                                    continue
                            _triggered.append((key, entry, d, state_wr, len(conds)))
                    # 按状态专项命中率降序，同分时少条件优先
                    best_entry = None
                    if _triggered:
                        _triggered.sort(key=lambda x: (-x[3], x[4]))
                        _best = _triggered[0]
                        best_entry = (_best[0], _best[1], _best[2])
                    if best_entry:
                        key, info, d = best_entry
                        in_pos, e_price, e_idx, e_dir = True, cl, cur, d
                        e_key, e_info = key, info
                        labels_arr[cur] = 1 if d == 'long' else -1
                        self.label_found.emit(cur, int(labels_arr[cur]))
                        tp = cl * (1 + long_tp_pct)  if d == 'long' else cl * (1 - short_tp_pct)
                        sl = cl * (1 - long_sl_pct)  if d == 'long' else cl * (1 + short_sl_pct)
                        e_margin = capital * PCT
                        # 重置动态止损状态
                        peak_pnl_pct = 0.0
                        ever_reached_threshold = False
                        pos_stage  = 1
                        peak_price = cl
                        stage2_margin = 0.0
                        _append_trade_log("entry", cur, d, cl, {
                            "reason": "signal",
                            "meta": {"signal_key": key},
                            "stop_loss": sl,
                            "take_profit": tp,
                            "leverage": LEVERAGE,
                            "margin": e_margin,
                        })
                        # 开仓时立即刷新持仓和交易明细
                        cur_pos = Position(
                            side=PositionSide.LONG if d == 'long' else PositionSide.SHORT,
                            entry_price=e_price, entry_idx=e_idx,
                            size=e_margin * LEVERAGE / e_price,
                            stop_loss=sl, take_profit=tp, liquidation_price=0.0,
                            margin=e_margin,
                        )
                        m = _make_running_metrics(trade_records, INITIAL_CAP, cur_pos)
                        m["current_bar"] = cur
                        m["trade_logs"] = list(trade_logs)
                        self.rt_update.emit(m, list(trade_records))

                self.step_completed.emit(cur)
                cur += 1

                sleep_time = max(0.001, 10.0 / max(1, self.speed))
                time.sleep(sleep_time)

            precompute_thread.join(timeout=10)

            if self._stop_requested or self._precompute_failed or not prep_initialized:
                self.is_running = False
                return

            final_labels = _pd.Series(labels_arr, index=df_work.index)
            self.labels = final_labels

            # 构建 BacktestResult
            def _make_bt_result(records, init_cap, final_cap):
                res = BacktestResult()
                res.initial_capital  = init_cap
                res.current_capital  = final_cap
                res.total_return_pct = (final_cap - init_cap) / init_cap * 100.0
                res.trades           = records
                res.total_trades     = len(records)
                wins   = [r for r in records if r.profit > 0]
                losses = [r for r in records if r.profit <= 0]
                res.win_trades  = len(wins)
                res.loss_trades = len(losses)
                res.win_rate    = len(wins) / len(records) if records else 0.0
                res.total_profit    = sum(r.profit for r in records)
                res.gross_profit    = sum(r.profit for r in wins)
                res.gross_loss      = abs(sum(r.profit for r in losses))
                res.avg_win         = res.gross_profit / len(wins)   if wins   else 0.0
                res.avg_loss        = res.gross_loss   / len(losses) if losses else 0.0
                res.profit_factor   = res.gross_profit / res.gross_loss if res.gross_loss > 0 else 0.0
                res.avg_profit_pct  = (sum(r.profit_pct for r in records) / len(records)
                                       if records else 0.0)
                res.avg_hold_periods = (sum(r.hold_periods for r in records) / len(records)
                                        if records else 0.0)
                long_recs  = [r for r in records if r.side ==  1]
                short_recs = [r for r in records if r.side == -1]
                res.long_trades   = len(long_recs)
                res.short_trades  = len(short_recs)
                lw = [r for r in long_recs  if r.profit > 0]
                sw = [r for r in short_recs if r.profit > 0]
                res.long_win_rate  = len(lw) / len(long_recs)  if long_recs  else 0.0
                res.short_win_rate = len(sw) / len(short_recs) if short_recs else 0.0
                res.long_profit    = sum(r.profit for r in long_recs)
                res.short_profit   = sum(r.profit for r in short_recs)
                cap2, peak, max_dd = init_cap, init_cap, 0.0
                for r in records:
                    cap2 += r.profit
                    peak = max(peak, cap2)
                    dd   = (peak - cap2) / peak * 100.0 if peak > 0 else 0.0
                    if dd > max_dd:
                        max_dd = dd
                res.max_drawdown = max_dd
                return res

            bt_result = _make_bt_result(trade_records, INITIAL_CAP, capital)
            bt_result.trade_logs = list(trade_logs)
            metrics = {
                "initial_capital": bt_result.initial_capital,
                "total_trades":    bt_result.total_trades,
                "win_rate":        bt_result.win_rate,
                "total_return":    bt_result.total_return_pct / 100.0,
                "total_profit":    bt_result.total_profit,
                "max_drawdown":    bt_result.max_drawdown,
                "sharpe_ratio":    bt_result.sharpe_ratio,
                "profit_factor":   bt_result.profit_factor,
                "long_win_rate":   bt_result.long_win_rate,
                "long_profit":     bt_result.long_profit,
                "short_win_rate":  bt_result.short_win_rate,
                "short_profit":    bt_result.short_profit,
                "current_pos":     None,
                "last_trade":      bt_result.trades[-1] if bt_result.trades else None,
                "trade_logs":      list(trade_logs),
            }
            self.finished.emit({
                'bt_result': bt_result,
                'metrics': metrics,
                'final_labels': final_labels,
            })

        except Exception as exc:
            self.error.emit(str(exc) + "\n" + traceback.format_exc())

        self.is_running = False

    def pause(self):
        self._pause_requested = True

    def resume(self):
        self._pause_requested = False

    def stop(self):
        self._stop_requested  = True
        self._pause_requested = False
        self.is_running = False

    def set_speed(self, speed: int):
        self.speed = speed


class DataLoaderWorker(QtCore.QObject):
    """数据加载工作者"""
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, sample_size, seed, data_file=None):
        super().__init__()
        self.sample_size = sample_size
        self.seed = seed
        self.data_file = data_file
    
    @QtCore.pyqtSlot()
    def process(self):
        try:
            from core.data_loader import DataLoader
            from utils.indicators import calculate_all_indicators
            
            loader = DataLoader(data_file=self.data_file)
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
    # WF Evolution 信号
    _evo_progress_signal = QtCore.pyqtSignal(object)   # EvolutionProgress
    _evo_done_signal = QtCore.pyqtSignal(object)       # EvolutionResult
    # DeepSeek 每 2 分钟意见结果（从工作线程发射到主线程）
    _deepseek_interval_result = QtCore.pyqtSignal(dict)
    
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
        
        # WF Evolution (特征权重进化)
        self._evo_engine = None
        self._evo_running = False
        self._evo_result = None  # 最新 EvolutionResult（或做多结果，用于兼容）
        self._evo_result_long = None   # 多空分开进化：做多结果
        self._evo_result_short = None  # 多空分开进化：做空结果
        
        # 模拟交易相关
        self._live_engine = None
        self._live_running = False
        self._paper_start_thread = None
        self._paper_start_worker = None
        self._live_chart_timer = QtCore.QTimer(self)
        refresh_ms = int(PAPER_TRADING_CONFIG.get("REALTIME_UI_REFRESH_MS", 1000))
        self._live_chart_timer.setInterval(max(50, refresh_ms))  # UI刷新频率
        self._live_chart_timer.timeout.connect(self._on_live_chart_tick)
        # 高频数据流下的UI保护：把图表和重型状态更新限制在可控频率
        self._live_chart_min_interval_sec = max(0.15, refresh_ms / 1000.0)
        self._last_live_chart_refresh_ts = 0.0
        self._live_chart_dirty = False
        self._live_state_ui_interval_sec = max(0.20, refresh_ms / 1000.0)
        self._last_live_state_ui_ts = 0.0
        self._last_live_state_bar_count = -1
        self._last_live_state_order_id = None
        self._last_ui_state_event = ""

        # 自适应控制器和DeepSeek复盘
        self._adaptive_controller = AdaptiveController()
        # DeepSeek 功能已禁用（按需求移除轮询/后台任务）
        self._deepseek_enabled = False
        self._deepseek_reviewer = None
        
        # 自适应仪表板刷新定时器（每10秒）
        self._adaptive_dashboard_timer = QtCore.QTimer(self)
        self._adaptive_dashboard_timer.setInterval(10000)
        self._adaptive_dashboard_timer.timeout.connect(self._refresh_adaptive_dashboard)
        
        # DeepSeek 复盘结果轮询定时器
        self._deepseek_poll_timer = QtCore.QTimer(self)
        self._deepseek_poll_timer.setInterval(5000)  # 每5秒检查一次
        self._deepseek_poll_timer.timeout.connect(self._poll_deepseek_reviews)
        # DeepSeek 每 2 分钟一次意见（持仓=持仓建议，无仓=市场/等待建议）
        self._deepseek_interval_timer = QtCore.QTimer(self)
        self._deepseek_interval_timer.setInterval(120000)  # 2 分钟
        self._deepseek_interval_timer.timeout.connect(self._on_deepseek_interval_tick)
        self._deepseek_interval_result.connect(self._on_deepseek_interval_result)

        # GA 完成信号（analysis_panel 在后续 _init_ui 中创建后再连接按钮）
        self._ga_done_signal.connect(self._on_ga_finished)
        # Walk-Forward 信号
        # 批量WF信号
        self._batch_wf_progress_signal.connect(self._on_batch_wf_progress)
        self._batch_wf_done_signal.connect(self._on_batch_wf_finished)
        # WF Evolution 信号
        self._evo_progress_signal.connect(self._on_evo_progress)
        self._evo_done_signal.connect(self._on_evo_finished)
        
        self._init_ui()
        self._connect_signals()
        self._load_saved_paper_api_config()

        # 延迟加载：UI 先显示，再异步加载数据（消除启动白屏）
        QtCore.QTimer.singleShot(0, self._deferred_startup_load)
    
    def _deferred_startup_load(self):
        """窗口显示后再加载数据，消除启动白屏"""
        self.statusBar().showMessage("正在加载数据...")
        QtWidgets.QApplication.processEvents()

        self._auto_load_memory()
        self._auto_load_prototypes()
        self._load_paper_trade_history_on_start()

        self.statusBar().showMessage("就绪", 3000)

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
        # 启动时加载历史交易记录（UI 端永久记忆，仅显示最近10笔避免卡顿）
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            history_file = os.path.join(project_root, "data", "live_trade_history.json")
            history = load_trade_history_from_file(history_file)
            if history:
                # 仅显示最近10笔，减少UI渲染负担（文件和统计仍保留全量）
                self.paper_trading_tab.load_historical_trades(history[-10:])
                print(f"[MainWindow] 已加载历史交易记录: 显示最近{min(10, len(history))}笔 / 共{len(history)}笔")
        except Exception as e:
            print(f"[MainWindow] 启动时加载交易记录失败: {e}")

        # ============ Tab 3: 自适应学习 ============
        self.adaptive_learning_tab = AdaptiveLearningTab()
        self.main_tabs.addTab(self.adaptive_learning_tab, "🧠 自适应学习")
        # 引擎未启动时，从文件初始化冷启动面板显示
        try:
            from config import COLD_START_CONFIG, SIMILARITY_CONFIG
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cs_state_path = os.path.join(project_root, "data", "cold_start_state.json")
            cs_enabled = False
            thresholds = COLD_START_CONFIG.get("THRESHOLDS", {})
            normal_thresholds = {
                "fusion": SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.65),
                "cosine": SIMILARITY_CONFIG.get("COSINE_MIN_THRESHOLD", 0.70),
                "euclidean": 0.35,
                "dtw": 0.30,
            }
            if os.path.exists(cs_state_path):
                with open(cs_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cs_enabled = bool(data.get("enabled", False))
                thresholds = {
                    "fusion": data.get("fusion_threshold", thresholds.get("fusion", 0.30)),
                    "cosine": data.get("cosine_threshold", thresholds.get("cosine", 0.50)),
                    "euclidean": data.get("euclidean_threshold", thresholds.get("euclidean", 0.25)),
                    "dtw": data.get("dtw_threshold", thresholds.get("dtw", 0.10)),
                }
            self.adaptive_learning_tab.set_cold_start_enabled(cs_enabled)
            self.adaptive_learning_tab.update_cold_start_thresholds(
                fusion=thresholds.get("fusion", 0.30),
                cosine=thresholds.get("cosine", 0.50),
                euclidean=thresholds.get("euclidean", 0.25),
                dtw=thresholds.get("dtw", 0.10),
                normal_thresholds=normal_thresholds,
            )
        except Exception as e:
            print(f"[UI] 初始化冷启动面板失败: {e}")
        
        # ============ Tab 4: 信号分析 ============
        self.signal_analysis_tab = SignalAnalysisTab()
        self.signal_analysis_tab.set_main_window(self)
        self.main_tabs.addTab(self.signal_analysis_tab, "🔍 信号分析")
        # "换新数据再验证"按钮 → 触发重新加载不同时间段的数据
        self.signal_analysis_tab.request_new_data.connect(self._on_signal_request_new_data)
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)

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
        
        # WF Evolution 信号
        self.analysis_panel.trajectory_widget.wf_evolution_requested.connect(
            self._on_evolution_requested
        )
        self.analysis_panel.trajectory_widget.wf_evolution_stop_requested.connect(
            self._on_evolution_stop
        )
        self.analysis_panel.trajectory_widget.wf_evolution_save_requested.connect(
            self._on_evolution_save
        )
        self.analysis_panel.trajectory_widget.wf_evolution_apply_requested.connect(
            self._on_evolution_apply
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
        # 清除学习记忆信号
        self.paper_trading_tab.control_panel.clear_memory_requested.connect(
            self._on_clear_learning_memory
        )
        self.paper_trading_tab.status_panel.save_profitable_requested.connect(
            self._on_save_profitable_templates
        )
        self.paper_trading_tab.status_panel.delete_losing_requested.connect(
            self._on_delete_losing_templates
        )
        # 门控拒绝追踪 - 阈值调整确认信号（迁移到自适应学习Tab）
        self.adaptive_learning_tab.rejection_log_card.adjustment_confirmed.connect(
            self._on_rejection_adjustment_confirmed
        )
        # 自适应学习其它面板的调整确认
        self.adaptive_learning_tab.exit_timing_card.adjustment_confirmed.connect(
            self._on_exit_timing_adjustment_confirmed
        )
        self.adaptive_learning_tab.tpsl_card.adjustment_confirmed.connect(
            self._on_tpsl_adjustment_confirmed
        )
        self.adaptive_learning_tab.near_miss_card.adjustment_confirmed.connect(
            self._on_near_miss_adjustment_confirmed
        )
        self.adaptive_learning_tab.regime_card.adjustment_confirmed.connect(
            self._on_regime_adjustment_confirmed
        )
        self.adaptive_learning_tab.early_exit_card.adjustment_confirmed.connect(
            self._on_early_exit_adjustment_confirmed
        )
        
        # 冷启动系统信号
        self.adaptive_learning_tab.cold_start_panel.cold_start_toggled.connect(
            self._on_cold_start_toggled
        )
        # 自适应学习 Tab 内「清除」按钮（Tab 内已确认，直接执行清除并刷新）
        self.adaptive_learning_tab.clear_memory_requested.connect(
            self._on_clear_adaptive_learning_requested
        )

    def _on_main_tab_changed(self, index: int):
        """主标签切换：延迟加载重资源标签，避免启动卡顿。"""
        widget = self.main_tabs.widget(index) if self.main_tabs else None
        if widget is self.signal_analysis_tab:
            self.signal_analysis_tab.ensure_initial_load()

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
        """加载数据：弹出文件选择对话框，用户选择 parquet 文件后加载"""
        self._show_file_dialog_and_load(DATA_CONFIG["SAMPLE_SIZE"], None)
    
    def _show_file_dialog_and_load(self, sample_size: int, seed, start_dir: str = None):
        """弹出文件选择对话框，选择后加载。start_dir 为 None 时使用 data/ 或项目根目录"""
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        init_dir = start_dir or (str(data_dir) if data_dir.exists() else str(project_root))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择数据文件",
            init_dir,
            "Parquet 文件 (*.parquet);;所有文件 (*.*)"
        )
        if not path:
            return
        self._on_sample_requested(sample_size, seed, data_file=path)
    
    def _on_sample_requested(self, sample_size: int, seed, data_file: str = None):
        """处理采样请求。data_file 为空则使用最近加载的文件"""
        if not data_file:
            data_file = getattr(self, "_last_data_file", None)
        if not data_file:
            QtWidgets.QMessageBox.warning(self, "提示", "请先通过“文件 → 加载数据”选择数据文件")
            return
        self._sampling_in_progress = True
        self.control_panel.set_status("正在加载数据...")
        self.control_panel.set_buttons_enabled(False)
        self.statusBar().showMessage("正在加载数据...")
        
        # 创建工作线程
        self.worker_thread = QtCore.QThread()
        self.data_worker = DataLoaderWorker(sample_size, seed, data_file=data_file)
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
            self._last_data_file = getattr(self.data_loader, 'data_file', None) or DATA_CONFIG.get("DATA_FILE")
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
            # 将历史K线数据传入信号分析页签
            try:
                self.signal_analysis_tab.set_data(self.df)
            except Exception as e:
                print(f"[MainWindow] 传递数据到信号分析页签失败: {e}")
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

    def _on_signal_request_new_data(self):
        """
        信号分析页签请求加载新一批历史数据（不同时间段的 50000 根 K 线）。
        若已有加载过的文件则用该文件重新采样；否则弹出文件选择。
        """
        import random as _random
        sample_size = DATA_CONFIG.get("SAMPLE_SIZE", 50000)
        random_seed = _random.randint(0, 999999)
        last_file = getattr(self, "_last_data_file", None)
        if last_file and Path(str(last_file)).exists():
            self.statusBar().showMessage(f"正在加载新一批历史数据（种子={random_seed}）...")
            self._on_sample_requested(sample_size, random_seed, data_file=last_file)
        else:
            self.statusBar().showMessage(f"正在加载新一批历史数据（种子={random_seed}）...")
            self._show_file_dialog_and_load(sample_size, random_seed)

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

        if (
            self.signal_analysis_tab
            and hasattr(self.signal_analysis_tab, "is_busy")
            and self.signal_analysis_tab.is_busy()
        ):
            QtWidgets.QMessageBox.warning(
                self, "提示", "自动验证运行中，请先停止再开始回测。"
            )
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
        self.analysis_panel.update_backtest_log([])
        self.analysis_panel.fingerprint_widget.clear_plot()
        self.control_panel.set_playing_state(True)
        self.control_panel.set_status("正在启动实时回测...")
        self.statusBar().showMessage("正在启动实时回测...")

        # 重置图表
        self.chart_widget.set_data(self.df, show_all=False)
        speed = self.control_panel.get_speed()
        self.chart_widget.set_render_stride(speed)
        self.chart_widget.set_fast_playback(True)
        self._progress_stride = 1 if speed <= 10 else (2 if speed <= 20 else 3)

        # 创建信号回测工作线程
        self.worker_thread = QtCore.QThread()
        self.labeling_worker = SignalBacktestWorker(self.df)
        self.labeling_worker.speed = speed
        self.labeling_worker.strict_state_filter = self.control_panel.get_strict_state_filter()
        self.labeling_worker.use_alt_tpsl = self.control_panel.get_alt_tpsl()
        self.labeling_worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.labeling_worker.run_backtest)
        self.labeling_worker.step_completed.connect(self._on_signal_bt_step,     QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.label_found.connect(self._on_label_found,           QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.labeling_progress.connect(self._on_labeling_progress, QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.labels_ready.connect(self._on_labels_ready,         QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.rt_update.connect(self._on_signal_bt_rt_update,     QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.finished.connect(self._on_signal_bt_finished,       QtCore.Qt.ConnectionType.QueuedConnection)
        self.labeling_worker.error.connect(self._on_signal_bt_error,             QtCore.Qt.ConnectionType.QueuedConnection)
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
            self.analysis_panel.update_backtest_log(self._format_backtest_logs(getattr(bt_result, "trade_logs", [])))

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
                    self.analysis_panel.update_backtest_log(self._format_backtest_logs(self.rt_backtester.trade_logs))
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
        if self.df is not None and self.labels is not None:
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

        if isinstance(self.labeling_worker, SignalBacktestWorker):
            cur_idx = getattr(self.chart_widget, 'current_display_index', 0)
            for i in range(min(cur_idx, len(labels))):
                v = int(labels.iloc[i])
                if v != 0:
                    self.chart_widget.add_signal_at(i, v, self.df)
            return

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

        # 补发：动画已经过了但 labels 那时还没 ready 的标记
        if self.df is not None and isinstance(self.labeling_worker, SignalBacktestWorker):
            cur_idx = getattr(self.chart_widget, 'current_display_index', 0)
            for i in range(min(cur_idx, len(labels))):
                v = int(labels.iloc[i])
                if v != 0:
                    self.chart_widget.add_signal_at(i, v, self.df)
            return

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
            self.analysis_panel.update_backtest_log(self._format_backtest_logs(self.rt_backtester.trade_logs))
        self._update_regime_stats()
        self._update_vector_space_plot()
        
        # 更新指纹图（追赶期间提取的模板）
        if templates_added > 0:
            self._update_fingerprint_view()
            self._update_memory_stats()
            print(f"[TrajectoryMemory] 追赶阶段提取: {templates_added} 个模板")

    def _format_trades(self, trades, current_pos=None, current_bar=0):
        """格式化交易明细（仅展示最近200条），可选附加当前持仓行"""
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
        # 当前持仓行（若有）
        if current_pos is not None:
            side_val = getattr(current_pos, 'side', 0)
            side = "LONG" if side_val == 1 else "SHORT"
            entry_idx = getattr(current_pos, 'entry_idx', 0)
            entry_price = getattr(current_pos, 'entry_price', 0.0)
            hold_bars = max(0, current_bar - entry_idx)
            # 策略编号（优先 signal_key，其次模板ID）
            signal_key = getattr(current_pos, 'signal_key', '') or ''
            signal_rate = getattr(current_pos, 'signal_rate', 0.0)
            signal_score = getattr(current_pos, 'signal_score', 0.0)
            if signal_key:
                fingerprint = f"#{signal_key[:8]} 率{signal_rate:.0%} 分{signal_score:.0f}"
            else:
                template_idx = getattr(current_pos, 'matched_template_idx', None)
                entry_sim = getattr(current_pos, 'entry_similarity', None)
                if template_idx is not None and entry_sim is not None:
                    fingerprint = f"T#{template_idx} | Sim={entry_sim:.2f}"
                else:
                    fingerprint = "--"
            avg_hold = getattr(current_pos, 'avg_hold_bars', 0) or 0
            rows.append({
                "side": side,
                "entry_time": fmt_time(entry_idx),
                "entry_price": f"{entry_price:.2f}",
                "exit_time": "持仓中",
                "exit_price": "--",
                "profit": "--",
                "profit_pct": "--",
                "hold": f"已持{hold_bars}根",
                "avg_hold": str(avg_hold) if avg_hold else "-",
                "regime": "",
                "fingerprint": fingerprint,
            })
        for t in trades[-200:]:
            side = "LONG" if t.side == 1 else "SHORT"
            signal_key = getattr(t, 'signal_key', '') or ''
            signal_rate = getattr(t, 'signal_rate', 0.0)
            signal_score = getattr(t, 'signal_score', 0.0)
            if signal_key:
                fingerprint = f"#{signal_key[:8]} 率{signal_rate:.0%} 分{signal_score:.0f}"
            else:
                # 指纹摘要：模板ID + 相似度
                template_idx = getattr(t, 'matched_template_idx', None)
                entry_sim = getattr(t, 'entry_similarity', None)
                if template_idx is not None and entry_sim is not None:
                    fingerprint = f"T#{template_idx} | Sim={entry_sim:.2f}"
                else:
                    fingerprint = "--"
            avg_hold = getattr(t, 'avg_hold_bars', 0) or 0
            rows.append({
                "side": side,
                "entry_time": fmt_time(t.entry_idx),
                "entry_price": f"{t.entry_price:.2f}",
                "exit_time": fmt_time(t.exit_idx),
                "exit_price": f"{t.exit_price:.2f}",
                "profit": f"{t.profit:.2f}",
                "profit_pct": f"{t.profit_pct:.2f}",
                "hold": str(t.hold_periods),
                "avg_hold": str(avg_hold) if avg_hold else "-",
                "regime": getattr(t, 'market_regime', ''),
                "fingerprint": fingerprint,
            })
        return rows

    def _format_backtest_logs(self, logs: List[dict]):
        """格式化回测日志（仅展示最近500条）"""
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

        event_map = {
            "entry":        "开仓",
            "tp1":          "TP1 部分止盈",
            "tp":           "止盈",
            "sl":           "止损",
            "trailing_sl":  "追踪止损",
            "signal":       "信号平仓",
            "end":          "强制平仓",
            "timeout":      "超时平仓",
            # legacy keys
            "exit":              "平仓",
            "partial_exit":      "部分平仓",
            "sl_move_breakeven": "止损上移",
            "sl_time_decay":     "止损收窄",
            "sl_trailing_update":"追踪更新",
            "stage2_enter":      "进入阶段2",
        }

        rows = []
        for e in (logs or [])[-500:]:
            idx   = e.get("idx", 0)
            info  = e.get("info", {}) or {}
            ev    = e.get("event", "")
            event = event_map.get(ev, ev)
            side  = e.get("side", "")
            price = e.get("price", "")
            stop_loss   = info.get("stop_loss", "")
            take_profit = info.get("take_profit", "")

            # 精炼 detail 列
            detail_parts = []
            profit_pct = info.get("profit_pct")
            hold_bars  = info.get("hold_bars")
            entry_price = info.get("entry_price")

            if ev == "entry":
                reason = info.get("reason", "")
                if reason:
                    detail_parts.append(f"原因={reason}")
                lev = info.get("leverage")
                if lev:
                    detail_parts.append(f"{lev}x杠杆")
                meta = info.get("meta", {})
                if isinstance(meta, dict):
                    ls = meta.get("long_score")
                    ss = meta.get("short_score")
                    if ls is not None:
                        detail_parts.append(f"多分={ls:.2f} 空分={ss:.2f}")
            elif ev == "tp1":
                ratio = info.get("ratio", 70)
                if profit_pct is not None:
                    sign = "+" if profit_pct >= 0 else ""
                    detail_parts.append(f"平仓{ratio:.0f}% {sign}{profit_pct:.2f}%  → 进入追踪止损")
            else:
                # sl / trailing_sl / signal / end
                if profit_pct is not None:
                    sign = "+" if profit_pct >= 0 else ""
                    detail_parts.append(f"盈亏 {sign}{profit_pct:.2f}%")
                if hold_bars is not None:
                    detail_parts.append(f"持仓 {hold_bars} 根")
                if entry_price:
                    detail_parts.append(f"入场 {entry_price}")

            rows.append({
                "event":      event,
                "time":       fmt_time(idx),
                "side":       side,
                "price":      f"{price:.2f}" if isinstance(price, (int, float)) else str(price),
                "stop_loss":  f"{stop_loss:.2f}" if isinstance(stop_loss, (int, float)) else str(stop_loss),
                "take_profit":f"{take_profit:.2f}" if isinstance(take_profit, (int, float)) else str(take_profit),
                "detail":     " | ".join(detail_parts),
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
    # WF Evolution (特征权重进化)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_evolution_requested(self, config: dict):
        """
        开始 WF Evolution（UI 信号槽）

        Args:
            config: {"sample_size", "n_trials", "inner_folds", "holdout_ratio"}
        """
        if self._evo_running:
            QtWidgets.QMessageBox.warning(self, "进化运行中", "请先停止当前进化再启动新的。")
            return

        if self._prototype_library is None:
            QtWidgets.QMessageBox.warning(
                self, "缺少原型库",
                "需要先加载原型库才能运行特征权重进化。\n"
                "请在「指纹模板库 → 原型」区域加载或生成原型。"
            )
            return

        # 确认对话框
        n_trials = config.get("n_trials", 60)
        sample_size = config.get("sample_size", 300000)
        inner_folds = config.get("inner_folds", 3)
        holdout_pct = int(config.get("holdout_ratio", 0.30) * 100)

        reply = QtWidgets.QMessageBox.question(
            self, "确认 WF 权重进化",
            f"将启动 CMA-ES 特征权重进化:\n\n"
            f"  数据量: {sample_size:,} 根K线\n"
            f"  试验次数: {n_trials}\n"
            f"  内部折数: {inner_folds} 折\n"
            f"  Holdout: {holdout_pct}%\n"
            f"  搜索维度: 8 组权重 + 2 阈值\n\n"
            f"预计耗时较长（取决于数据量和试验次数）。\n"
            f"继续吗？",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # ── UI 更新 ──
        self._evo_running = True
        self._evo_result = None
        self._evo_result_long = None
        self._evo_result_short = None
        self.analysis_panel.on_evolution_started()
        self.statusBar().showMessage(f"WF Evolution 启动: {n_trials} trials, {inner_folds} folds...")

        # ── 后台线程运行 ──
        prototype_lib = self._prototype_library

        def _run_evolution():
            try:
                from core.wf_evolution_engine import WFEvolutionEngine

                def progress_callback(progress):
                    self._evo_progress_signal.emit(progress)

                self._evo_engine = WFEvolutionEngine(
                    prototype_library=prototype_lib,
                    on_progress=progress_callback,
                    sample_size=config.get("sample_size"),
                    n_trials=config.get("n_trials"),
                    inner_folds=config.get("inner_folds"),
                    holdout_ratio=config.get("holdout_ratio"),
                )

                # 多空分开进化：先做多，后做空，各保存并汇总
                results = {"long": None, "short": None}
                for direction in ("LONG", "SHORT"):
                    if getattr(self._evo_engine, "_stop_requested", False):
                        break
                    result = self._evo_engine.run(direction=direction)
                    if result.success:
                        self._evo_engine.save_result(result)
                    results[direction.lower()] = result
                self._evo_done_signal.emit(results)

            except Exception as e:
                import traceback
                print(f"[WF-Evo] 进化异常: {e}")
                traceback.print_exc()
                self._evo_done_signal.emit({"long": None, "short": None})

        thread = threading.Thread(target=_run_evolution, daemon=True)
        thread.start()

    def _on_evolution_stop(self):
        """停止 WF Evolution"""
        if self._evo_engine is not None:
            self._evo_engine.stop()
            self.statusBar().showMessage("正在停止 WF Evolution...")

    def _on_evo_progress(self, progress):
        """
        WF Evolution 进度更新（主线程槽函数）

        Args:
            progress: EvolutionProgress 实例
        """
        phase = getattr(progress, "phase", "optimizing")
        trial_idx = getattr(progress, "trial_idx", 0)
        n_trials = getattr(progress, "n_trials", 60)
        best_fitness = getattr(progress, "best_fitness", -999)
        eta_sec = getattr(progress, "eta_sec", 0.0)
        fold_detail = getattr(progress, "fold_detail", "")
        group_weights = getattr(progress, "group_weights", None)
        fusion_th = getattr(progress, "fusion_threshold", 0.0)
        cosine_min_th = getattr(progress, "cosine_min_threshold", 0.0)

        if phase == "loading":
            self.statusBar().showMessage("WF Evolution: 加载数据中...")
            return

        if phase == "holdout":
            self.statusBar().showMessage("WF Evolution: Holdout 验证中...")
            return

        if phase == "done":
            return  # 由 _on_evo_finished 处理

        # ── 更新 UI ──
        # 进度条和试验进度
        self.analysis_panel.update_evolution_progress(
            trial_idx=trial_idx,
            n_trials=n_trials,
            fold_idx=0,
            n_folds=3,
            best_fitness=best_fitness if best_fitness > -100 else 0.0,
            eta_seconds=eta_sec,
            phase=phase,
        )

        # 更新权重柱状图
        if group_weights is not None:
            weights_list = group_weights.tolist() if hasattr(group_weights, 'tolist') else list(group_weights)
            self.analysis_panel.update_evolution_weights(
                weights_list, fusion_th, cosine_min_th)

        # 状态栏
        fitness_str = f"{best_fitness:.4f}" if best_fitness > -100 else "--"
        eta_min = int(eta_sec // 60) if eta_sec > 0 else 0
        eta_s = int(eta_sec % 60) if eta_sec > 0 else 0
        eta_str = f", ETA: {eta_min}m{eta_s}s" if eta_sec > 0 else ""
        self.statusBar().showMessage(
            f"WF Evolution: Trial {trial_idx}/{n_trials} | "
            f"Best Sharpe: {fitness_str}{eta_str}"
        )

    def _on_evo_finished(self, result):
        """
        WF Evolution 完成（主线程槽函数）

        Args:
            result: EvolutionResult 实例、或 dict {"long": res, "short": res}（多空分开进化）
        """
        self._evo_running = False
        # 多空分开进化：解析两套结果
        if isinstance(result, dict):
            self._evo_result_long = result.get("long")
            self._evo_result_short = result.get("short")
            self._evo_result = self._evo_result_long or self._evo_result_short
        else:
            self._evo_result = result
            self._evo_result_long = None
            self._evo_result_short = None
        self.analysis_panel.on_evolution_finished()

        # 至少有一个成功才算完成
        any_ok = (self._evo_result_long and self._evo_result_long.success) or (
            self._evo_result_short and self._evo_result_short.success)
        if not any_ok:
            self.statusBar().showMessage("WF Evolution 失败或已停止")
            if self._evo_result is not None and not self._evo_result.success:
                error_detail = getattr(self._evo_result, "error_message", "") or "未返回具体错误信息。"
                QtWidgets.QMessageBox.warning(
                    self, "进化未成功",
                    "WF Evolution 未产生有效结果。\n"
                    "可能原因：数据不足、试验次数太少、或用户中止。\n\n"
                    f"详细原因：{error_detail}"
                )
            return

        # ── 更新 holdout / 权重显示：做多 + 做空两套（多空分开时） ──
        if self._evo_result_long and self._evo_result_long.success and self._evo_result_short and self._evo_result_short.success:
            holdout_long = {
                "sharpe": self._evo_result_long.holdout_sharpe,
                "win_rate": self._evo_result_long.holdout_win_rate,
                "profit": self._evo_result_long.holdout_profit,
                "drawdown": self._evo_result_long.holdout_drawdown,
                "n_trades": self._evo_result_long.holdout_n_trades,
                "profit_factor": self._evo_result_long.holdout_profit_factor,
                "passed": self._evo_result_long.holdout_passed,
            }
            holdout_short = {
                "sharpe": self._evo_result_short.holdout_sharpe,
                "win_rate": self._evo_result_short.holdout_win_rate,
                "profit": self._evo_result_short.holdout_profit,
                "drawdown": self._evo_result_short.holdout_drawdown,
                "n_trades": self._evo_result_short.holdout_n_trades,
                "profit_factor": self._evo_result_short.holdout_profit_factor,
                "passed": self._evo_result_short.holdout_passed,
            }
            self.analysis_panel.update_evolution_holdout({"long": holdout_long, "short": holdout_short})
            w_long = self._evo_result_long.group_weights.tolist() if self._evo_result_long.group_weights is not None else []
            w_short = self._evo_result_short.group_weights.tolist() if self._evo_result_short.group_weights is not None else []
            self.analysis_panel.update_evolution_weights(
                w_long,
                self._evo_result_long.fusion_threshold,
                self._evo_result_long.cosine_min_threshold,
                short_weights=w_short if w_short else None,
                short_fusion=self._evo_result_short.fusion_threshold,
                short_cosine=self._evo_result_short.cosine_min_threshold,
            )
        else:
            primary = self._evo_result_long if (self._evo_result_long and self._evo_result_long.success) else self._evo_result_short
            if primary is None and self._evo_result and self._evo_result.success:
                primary = self._evo_result
            if primary is not None:
                holdout_data = {
                    "sharpe": primary.holdout_sharpe,
                    "win_rate": primary.holdout_win_rate,
                    "profit": primary.holdout_profit,
                    "drawdown": primary.holdout_drawdown,
                    "n_trades": primary.holdout_n_trades,
                    "profit_factor": primary.holdout_profit_factor,
                    "passed": primary.holdout_passed,
                }
                self.analysis_panel.update_evolution_holdout(holdout_data)
                if primary.group_weights is not None:
                    weights_list = primary.group_weights.tolist()
                    self.analysis_panel.update_evolution_weights(
                        weights_list, primary.fusion_threshold, primary.cosine_min_threshold)

        # ── 完成提示（多空分开时汇总做多/做空） ──
        parts = []
        if self._evo_result_long and self._evo_result_long.success:
            r = self._evo_result_long
            parts.append(f"做多: Sharpe={r.holdout_sharpe:.4f}, 通过={'✓' if r.holdout_passed else '✗'}")
        if self._evo_result_short and self._evo_result_short.success:
            r = self._evo_result_short
            parts.append(f"做空: Sharpe={r.holdout_sharpe:.4f}, 通过={'✓' if r.holdout_passed else '✗'}")
        time_str = ""
        if primary is not None:
            elapsed_min = int(primary.elapsed_sec // 60)
            elapsed_sec = int(primary.elapsed_sec % 60)
            time_str = f"{elapsed_min}分{elapsed_sec}秒" if elapsed_min > 0 else f"{elapsed_sec}秒"
        self.statusBar().showMessage("WF Evolution 完成! " + ", ".join(parts) + (f", 耗时{time_str}" if time_str else ""))

        msg_lines = ["WF 特征权重进化完成（多空分开进化）!\n"]
        if self._evo_result_long and self._evo_result_long.success:
            r = self._evo_result_long
            msg_lines.append(f"── 做多 Holdout {'✓ 通过' if r.holdout_passed else '✗ 未通过'} ──\n  Sharpe: {r.holdout_sharpe:.4f}, 胜率: {r.holdout_win_rate:.1%}, 收益: {r.holdout_profit:.2f}%, 笔数: {r.holdout_n_trades}\n")
        if self._evo_result_short and self._evo_result_short.success:
            r = self._evo_result_short
            msg_lines.append(f"── 做空 Holdout {'✓ 通过' if r.holdout_passed else '✗ 未通过'} ──\n  Sharpe: {r.holdout_sharpe:.4f}, 胜率: {r.holdout_win_rate:.1%}, 收益: {r.holdout_profit:.2f}%, 笔数: {r.holdout_n_trades}\n")
        msg_lines.append("\n可以点击「保存权重」持久化，或「应用到实盘」立即生效。")
        QtWidgets.QMessageBox.information(self, "WF Evolution 完成", "".join(msg_lines))

    def _on_evolution_save(self):
        """保存 WF Evolution 结果（多空分开时保存做多/做空两套文件）"""
        to_save = []
        if self._evo_result_long and self._evo_result_long.success:
            to_save.append(("做多", self._evo_result_long))
        if self._evo_result_short and self._evo_result_short.success:
            to_save.append(("做空", self._evo_result_short))
        if not to_save and self._evo_result and self._evo_result.success:
            to_save.append(("进化", self._evo_result))
        if not to_save:
            QtWidgets.QMessageBox.warning(self, "无结果", "没有可保存的进化结果。")
            return

        try:
            from core.wf_evolution_engine import WFEvolutionEngine
            engine = WFEvolutionEngine(prototype_library=self._prototype_library)
            paths = []
            for label, res in to_save:
                fp = engine.save_result(res)
                paths.append(f"{label}: {fp}")
            self.statusBar().showMessage("进化权重已保存: " + ", ".join(p for p in paths))
            QtWidgets.QMessageBox.information(
                self, "保存成功",
                "进化后的特征权重已保存到:\n" + "\n".join(paths))
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "保存失败", f"保存进化结果失败:\n{e}")

    def _on_evolution_apply(self):
        """
        将进化后的特征权重应用到实盘/模拟盘（多空分开时做多/做空各一套）
        """
        has_long = self._evo_result_long and self._evo_result_long.success and self._evo_result_long.full_weights is not None
        has_short = self._evo_result_short and self._evo_result_short.success and self._evo_result_short.full_weights is not None
        if not has_long and not has_short:
            if self._evo_result and self._evo_result.success and self._evo_result.full_weights is not None:
                has_long = True  # 单组结果当作做多，共用一套
            else:
                QtWidgets.QMessageBox.warning(self, "无结果", "没有可应用的进化结果。")
                return

        # 多空两套或单组
        if has_long and self._evo_result_long:
            long_w = self._evo_result_long.full_weights
            long_f = self._evo_result_long.fusion_threshold
            long_c = self._evo_result_long.cosine_min_threshold
            long_e = getattr(self._evo_result_long, "euclidean_min_threshold", 0.50)
            long_d = getattr(self._evo_result_long, "dtw_min_threshold", 0.40)
        elif self._evo_result and self._evo_result.full_weights is not None:
            long_w = self._evo_result.full_weights
            long_f = self._evo_result.fusion_threshold
            long_c = self._evo_result.cosine_min_threshold
            long_e = getattr(self._evo_result, "euclidean_min_threshold", 0.50)
            long_d = getattr(self._evo_result, "dtw_min_threshold", 0.40)
        else:
            long_w = long_f = long_c = long_e = long_d = None
        if has_short and self._evo_result_short:
            short_w = self._evo_result_short.full_weights
            short_f = self._evo_result_short.fusion_threshold
            short_c = self._evo_result_short.cosine_min_threshold
            short_e = getattr(self._evo_result_short, "euclidean_min_threshold", 0.50)
            short_d = getattr(self._evo_result_short, "dtw_min_threshold", 0.40)
        else:
            short_w = short_f = short_c = short_e = short_d = None
        if long_w is not None and short_w is None:
            short_w, short_f, short_c, short_e, short_d = long_w, long_f, long_c, long_e, long_d
        fusion_th = long_f or 0.65
        cosine_min_th = long_c or 0.70
        euclidean_min_th = long_e or 0.50
        dtw_min_th = long_d or 0.40

        # ── 绑定到当前聚合指纹图 ──
        if self._prototype_library is not None:
            try:
                if has_long and has_short:
                    self._prototype_library.set_evolved_params(
                        long_weights=long_w, long_fusion=long_f, long_cosine=long_c, long_euclidean=long_e, long_dtw=long_d,
                        short_weights=short_w, short_fusion=short_f, short_cosine=short_c, short_euclidean=short_e, short_dtw=short_d)
                else:
                    self._prototype_library.set_evolved_params(
                        full_weights=long_w, fusion_threshold=fusion_th, cosine_min_threshold=cosine_min_th,
                        euclidean_min_threshold=euclidean_min_th, dtw_min_threshold=dtw_min_th)
            except Exception as e:
                print(f"[WF-Evo] 绑定进化结果到原型库失败: {e}")

        applied = False
        if self._live_engine is not None and self._live_running:
            try:
                proto_matcher = getattr(self._live_engine, "_proto_matcher", None)
                if proto_matcher is not None:
                    proto_matcher.set_feature_weights(long_w, short_weights=short_w)
                    proto_matcher.fusion_threshold = fusion_th
                    proto_matcher.cosine_threshold = cosine_min_th
                    proto_matcher.set_single_dimension_thresholds(
                        long_euclidean=long_e, long_dtw=long_d,
                        short_euclidean=short_e, short_dtw=short_d)
                    self._live_engine.cosine_threshold = cosine_min_th
                    if getattr(self._live_engine, "state", None) is not None:
                        self._live_engine.state.entry_threshold = cosine_min_th
                    print(f"[WF-Evo] 已注入进化权重到实盘引擎 (多空分开={bool(short_w)})")
                    applied = True
            except Exception as e:
                print(f"[WF-Evo] 注入实盘引擎失败: {e}")

        if applied:
            self.statusBar().showMessage("进化权重已应用到实盘引擎并绑定到聚合指纹图")
            QtWidgets.QMessageBox.information(
                self, "应用成功",
                "进化结果已给到当前聚合指纹图，并已注入运行中的交易引擎。\n"
                "保存原型库时，进化权重会一并保存。"
            )
        else:
            self.statusBar().showMessage("进化结果已绑定到当前聚合指纹图；保存原型库时会一并保存权重")
            try:
                from core.wf_evolution_engine import WFEvolutionEngine
                engine = WFEvolutionEngine(prototype_library=self._prototype_library)
                if has_long:
                    engine.save_result(self._evo_result_long)
                if has_short:
                    engine.save_result(self._evo_result_short)
                if not has_long and not has_short and self._evo_result:
                    engine.save_result(self._evo_result)
                QtWidgets.QMessageBox.information(
                    self, "已给到聚合指纹图",
                    "进化结果已绑定到当前聚合指纹图。\n"
                    "保存原型库或启动模拟交易时将自动使用该权重。"
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "保存失败", f"保存权重失败:\n{e}")

    def _gather_evolved_weights(self) -> dict:
        """
        收集进化权重（不依赖 engine 实例）。
        优先级: 原型库自带 → 内存 _evo_result_long/_short → 磁盘 long/short 文件。
        返回 dict（key = engine 属性名）；若无可用权重返回空 dict。
        """
        long_w, long_f, long_c, long_e, long_d = None, None, None, None, None
        short_w, short_f, short_c, short_e, short_d = None, None, None, None, None
        source = ""

        if self._prototype_library is not None:
            (lw, lf, lc, le, ld), (sw, sf, sc, se, sd) = self._prototype_library.get_evolved_params()
            if lw is not None:
                long_w, long_f, long_c, long_e, long_d = lw, lf, lc, le, ld
                short_w, short_f, short_c, short_e, short_d = sw, sf, sc, se, sd
                source = "当前聚合指纹图（原型库）"
        if long_w is None and self._evo_result_long and self._evo_result_long.success and self._evo_result_long.full_weights is not None:
            long_w = self._evo_result_long.full_weights
            long_f = self._evo_result_long.fusion_threshold
            long_c = self._evo_result_long.cosine_min_threshold
            long_e = getattr(self._evo_result_long, "euclidean_min_threshold", 0.50)
            long_d = getattr(self._evo_result_long, "dtw_min_threshold", 0.40)
            if self._evo_result_short and self._evo_result_short.success and self._evo_result_short.full_weights is not None:
                short_w = self._evo_result_short.full_weights
                short_f = self._evo_result_short.fusion_threshold
                short_c = self._evo_result_short.cosine_min_threshold
                short_e = getattr(self._evo_result_short, "euclidean_min_threshold", 0.50)
                short_d = getattr(self._evo_result_short, "dtw_min_threshold", 0.40)
            else:
                short_w, short_f, short_c, short_e, short_d = long_w, long_f, long_c, long_e, long_d
            source = "本次进化结果"
        if long_w is None and self._evo_result and self._evo_result.success and self._evo_result.full_weights is not None:
            long_w = self._evo_result.full_weights
            long_f = self._evo_result.fusion_threshold
            long_c = self._evo_result.cosine_min_threshold
            long_e = getattr(self._evo_result, "euclidean_min_threshold", 0.50)
            long_d = getattr(self._evo_result, "dtw_min_threshold", 0.40)
            short_w, short_f, short_c, short_e, short_d = long_w, long_f, long_c, long_e, long_d
            source = "本次进化结果"
        if long_w is None:
            try:
                from config import WF_EVOLUTION_CONFIG
                from core.wf_evolution_engine import WFEvolutionEngine
                cfg = WF_EVOLUTION_CONFIG
                long_path = cfg.get("EVOLVED_WEIGHTS_FILE_LONG") or (cfg.get("EVOLVED_WEIGHTS_FILE", "").replace(".json", "_long.json"))
                short_path = cfg.get("EVOLVED_WEIGHTS_FILE_SHORT") or (cfg.get("EVOLVED_WEIGHTS_FILE", "").replace(".json", "_short.json"))
                loaded_long = WFEvolutionEngine.load_result(long_path)
                loaded_short = WFEvolutionEngine.load_result(short_path)
                if loaded_long and loaded_long.success and loaded_long.full_weights is not None:
                    long_w, long_f, long_c = loaded_long.full_weights, loaded_long.fusion_threshold, loaded_long.cosine_min_threshold
                    long_e = getattr(loaded_long, "euclidean_min_threshold", 0.50)
                    long_d = getattr(loaded_long, "dtw_min_threshold", 0.40)
                    if loaded_short and loaded_short.success and loaded_short.full_weights is not None:
                        short_w, short_f, short_c = loaded_short.full_weights, loaded_short.fusion_threshold, loaded_short.cosine_min_threshold
                        short_e = getattr(loaded_short, "euclidean_min_threshold", 0.50)
                        short_d = getattr(loaded_short, "dtw_min_threshold", 0.40)
                    else:
                        short_w, short_f, short_c, short_e, short_d = long_w, long_f, long_c, long_e, long_d
                    source = f"文件 {long_path}"
                elif loaded_short and loaded_short.success and loaded_short.full_weights is not None:
                    short_w, short_f, short_c = loaded_short.full_weights, loaded_short.fusion_threshold, loaded_short.cosine_min_threshold
                    short_e = getattr(loaded_short, "euclidean_min_threshold", 0.50)
                    short_d = getattr(loaded_short, "dtw_min_threshold", 0.40)
                    long_w, long_f, long_c, long_e, long_d = short_w, short_f, short_c, short_e, short_d
                    source = f"文件 {short_path}"
                else:
                    single_path = cfg.get("EVOLVED_WEIGHTS_FILE", "data/wf_evolution/evolved_weights.json")
                    loaded = WFEvolutionEngine.load_result(single_path)
                    if loaded and loaded.success and loaded.full_weights is not None:
                        long_w = loaded.full_weights
                        long_f = loaded.fusion_threshold
                        long_c = getattr(loaded, "cosine_min_threshold", None)
                        long_e = getattr(loaded, "euclidean_min_threshold", 0.50)
                        long_d = getattr(loaded, "dtw_min_threshold", 0.40)
                        short_w, short_f, short_c, short_e, short_d = long_w, long_f, long_c, long_e, long_d
                        source = f"文件 {single_path}"
            except Exception as e:
                print(f"[WF-Evo] 加载已保存权重失败: {e}")

        if long_w is None:
            return {}

        effective_short_w = short_w if (short_w is not None and short_w is not long_w) else None
        result = {
            "_pending_evolved_weights_long": long_w,
            "_pending_evolved_weights_short": effective_short_w,
            "_pending_evolved_fusion_th": long_f,
            "_pending_evolved_cosine_th": long_c,
            "_pending_evolved_euclidean_th_long": long_e,
            "_pending_evolved_euclidean_th_short": short_e,
            "_pending_evolved_dtw_th_long": long_d,
            "_pending_evolved_dtw_th_short": short_d,
            "_pending_evolved_weights": long_w,
        }
        print(f"[WF-Evo] 进化权重已收集 (来源: {source}, 多空分开={effective_short_w is not None})")
        return result

    def _inject_evolved_weights_to_engine(self):
        """将进化权重挂载到已创建的 LiveTradingEngine。"""
        if self._live_engine is None:
            return
        weights = self._gather_evolved_weights()
        for attr_name, val in weights.items():
            setattr(self._live_engine, attr_name, val)

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
                # 原型库就绪 → 启用进化按钮
                self.analysis_panel.enable_evolution(True)
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
            # 原型库就绪 → 启用进化按钮
            self.analysis_panel.enable_evolution(True)
            
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
            # 原型库就绪 → 启用进化按钮
            self.analysis_panel.enable_evolution(True)
            
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
            stats = MarketRegimeClassifier.compute_direction_regime_stats(
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
                    self.analysis_panel.update_backtest_log(self._format_backtest_logs(getattr(bt_result, "trade_logs", [])))

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
    
    def _on_signal_bt_step(self, target_idx: int):
        """信号回测专用步进：一次推进到 target_idx，只触发1次图表渲染"""
        try:
            self.chart_widget.advance_to(target_idx)
            total = len(self.df) if self.df is not None else 0
            progress_stride = getattr(self, "_progress_stride", 1)
            if target_idx % progress_stride == 0 or target_idx >= total - 1:
                self.control_panel.update_play_progress(target_idx + 1, total)
        except Exception as e:
            if self.labeling_worker:
                self.labeling_worker.stop()
            self.is_playing = False
            self.control_panel.set_playing_state(False)

    def _on_signal_bt_rt_update(self, metrics: dict, trades: list):
        """信号回测实时指标刷新"""
        self.optimizer_panel.update_backtest_metrics(metrics)
        current_pos = metrics.get("current_pos")
        current_bar = metrics.get("current_bar", 0)
        self.analysis_panel.update_trade_log(self._format_trades(trades, current_pos=current_pos, current_bar=current_bar))
        logs = metrics.get("trade_logs")
        if logs is not None:
            self.analysis_panel.update_backtest_log(self._format_backtest_logs(logs))
        elif self.rt_backtester:
            self.analysis_panel.update_backtest_log(self._format_backtest_logs(self.rt_backtester.trade_logs))
        # 同步更新图表 SL/TP 线（反映时间衰减和追踪止损的实时变化）
        if hasattr(self, 'chart_widget') and self.chart_widget is not None:
            if current_pos is not None:
                _bt_sl = getattr(current_pos, 'stop_loss', None)
                _bt_tp = getattr(current_pos, 'take_profit', None)
                self.chart_widget._tp_sl_locked = True
                if hasattr(self.chart_widget, 'set_tp_sl_lines'):
                    self.chart_widget.set_tp_sl_lines(_bt_tp, _bt_sl)
            else:
                self.chart_widget._tp_sl_locked = False
                if hasattr(self.chart_widget, 'set_tp_sl_lines'):
                    self.chart_widget.set_tp_sl_lines(None, None)

    def _on_signal_bt_finished(self, result: dict):
        """信号回测完成"""
        bt_result = result["bt_result"]
        metrics   = result["metrics"]
        final_labels = result.get("final_labels") or self.labels

        # 显示全部数据和标注
        self.labels = final_labels
        self.chart_widget.set_data(self.df, final_labels, show_all=True)

        # 更新指标面板
        self.optimizer_panel.update_backtest_metrics(metrics)

        # 更新交易明细
        self.analysis_panel.update_trade_log(self._format_trades(bt_result.trades))
        self.analysis_panel.update_backtest_log(self._format_backtest_logs(getattr(bt_result, "trade_logs", [])))

        self.is_playing = False
        self.control_panel.set_playing_state(False)
        self.labeling_worker = None
        if self.chart_widget:
            self.chart_widget.set_fast_playback(False)

        long_n  = bt_result.long_trades
        short_n = bt_result.short_trades
        status = (
            f"信号回测完成: {bt_result.total_trades}笔"
            f"(多{long_n}/空{short_n}) | "
            f"胜率{bt_result.win_rate:.1%} | "
            f"收益{bt_result.total_return_pct:.2f}%"
        )
        self.control_panel.set_status(status)
        self.statusBar().showMessage(status)

        QtWidgets.QMessageBox.information(
            self, "信号回测结果",
            f"总交易: {bt_result.total_trades} 笔\n"
            f"胜率: {bt_result.win_rate:.1%}\n"
            f"总收益: {bt_result.total_return_pct:.2f}%\n"
            f"最大回撤: {bt_result.max_drawdown:.2f}%\n"
            f"做多: {long_n}笔  做空: {short_n}笔",
        )

    def _on_signal_bt_error(self, error_msg: str):
        """信号回测出错"""
        self.is_playing = False
        self.control_panel.set_playing_state(False)
        self.labeling_worker = None
        self._on_worker_error(error_msg)

    def _on_pause_requested(self):
        """暂停请求"""
        if self.labeling_worker:
            self.labeling_worker.pause()
            self.control_panel.play_btn.setText("▶ 继续")
    
    def _on_stop_requested(self):
        """停止请求"""
        try:
            if self.labeling_worker:
                self.labeling_worker.stop()
            # 通知工作线程退出事件循环，避免 QThread 残留导致崩溃
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait(15000)  # 预计算可能需10s，留足余量
        except Exception as e:
            print(f"[MainWindow] 停止回测时异常: {e}")
            import traceback
            traceback.print_exc()

        self.is_playing = False
        self.control_panel.set_playing_state(False)
        if self.chart_widget:
            self.chart_widget.set_fast_playback(False)

        try:
            # 显示已有的标注
            if self.df is not None and self.labels is not None:
                self.chart_widget.set_data(self.df, self.labels, show_all=True)
        except Exception as e:
            print(f"[MainWindow] 停止后刷新图表异常: {e}")

        self.labeling_worker = None
        self.statusBar().showMessage("已停止")
    
    def _on_speed_changed(self, speed: int):
        """速度变化"""
        if self.labeling_worker:
            self.labeling_worker.set_speed(speed)
        if self.chart_widget:
            self.chart_widget.set_render_stride(speed)
        self._progress_stride = 1 if speed <= 10 else (2 if speed <= 20 else 3)
    
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
    
    def _start_trade_history_io_worker(self, action: str, order=None,
                                       on_loaded=None, on_deleted=None, on_failed=None):
        """启动历史交易文件 IO 的后台线程"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        history_file = os.path.join(project_root, "data", "live_trade_history.json")

        thread = QtCore.QThread(self)
        worker = TradeHistoryIOWorker(history_file, action, order)
        worker.moveToThread(thread)

        if on_loaded:
            worker.loaded.connect(on_loaded)
        if on_deleted:
            worker.deleted.connect(on_deleted)
        if on_failed:
            worker.failed.connect(on_failed)
        else:
            worker.failed.connect(lambda msg: self._on_trade_history_io_failed(action, msg))

        worker.loaded.connect(thread.quit)
        worker.deleted.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        if not hasattr(self, "_history_io_threads"):
            self._history_io_threads = []
        self._history_io_threads.append(thread)

        def _cleanup():
            try:
                self._history_io_threads.remove(thread)
            except ValueError:
                pass
        thread.finished.connect(_cleanup)

        thread.started.connect(worker.run)
        thread.start()

    def _on_trade_history_io_failed(self, action: str, error_msg: str):
        """历史交易文件 IO 失败提示"""
        self.statusBar().showMessage("历史交易处理失败", 5000)
        QtWidgets.QMessageBox.warning(
            self,
            "历史交易处理失败",
            f"{action} 操作失败:\n{error_msg}"
        )
    
    def _load_paper_trade_history_on_start(self):
        """程序启动时从本地文件加载历史交易记录并显示（后台 IO）"""
        display_limit = PaperTradingTradeLog.MAX_DISPLAY_TRADES

        def _apply_history(history):
            if not history:
                return
            display_count = min(display_limit, len(history))
            self.paper_trading_tab.load_historical_trades(history[-display_count:])
            self.statusBar().showMessage(
                f"已加载历史交易记录: 显示{display_count}笔 / 共{len(history)}笔", 3000
            )

        self._start_trade_history_io_worker("load", on_loaded=_apply_history)
    
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
    
    def _clear_adaptive_learning_files_impl(self):
        """实际执行：删除学习状态文件并重置引擎状态。返回删除的文件数。"""
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        learning_files = [
            "data/bayesian_state.json",
            "data/rejection_tracker_state.json",
            "data/tpsl_tracker_state.json",
            "data/exit_timing_state.json",
            "data/exit_learning_state.json",
            "data/near_miss_tracker_state.json",
            "data/early_exit_state.json",
            "data/cold_start_state.json",
            "data/adaptive_controller_state.json",
        ]
        deleted_count = 0
        for rel_path in learning_files:
            full_path = os.path.join(project_root, rel_path)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    deleted_count += 1
                    print(f"[MainWindow] 已删除学习记忆: {rel_path}")
                except Exception as e:
                    print(f"[MainWindow] 删除失败 {rel_path}: {e}")
        if self._live_engine:
            if hasattr(self._live_engine, '_bayesian_filter') and self._live_engine._bayesian_filter:
                self._live_engine._bayesian_filter.distributions.clear()
                self._live_engine._bayesian_filter.total_signals_received = 0
                self._live_engine._bayesian_filter.total_signals_accepted = 0
                self._live_engine._bayesian_filter.total_signals_rejected = 0
            if hasattr(self._live_engine, '_adaptive_controller') and self._live_engine._adaptive_controller:
                self._live_engine._adaptive_controller.reset_all()
        return deleted_count

    def _on_clear_adaptive_learning_requested(self):
        """自适应学习 Tab 内「清除」已确认后调用：删除文件、重置引擎、刷新 Tab。"""
        deleted_count = self._clear_adaptive_learning_files_impl()
        self.paper_trading_tab.status_panel.append_event(
            f"[系统] 已清除 {deleted_count} 个学习记忆文件"
        )
        self.statusBar().showMessage(f"已清除 {deleted_count} 个自适应学习记忆文件", 5000)
        if hasattr(self, "adaptive_learning_tab") and hasattr(self.adaptive_learning_tab, "refresh_from_state_files"):
            self.adaptive_learning_tab.refresh_from_state_files()

    def _on_clear_learning_memory(self):
        """清除所有自适应学习记忆（先确认再执行）"""
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空",
            "确定要清空所有自适应学习记忆吗？\n（将删除各 tracker 的状态文件，学习数据需重新积累）",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        deleted_count = self._clear_adaptive_learning_files_impl()
        self.paper_trading_tab.status_panel.append_event(
            f"[系统] 已清除 {deleted_count} 个学习记忆文件"
        )
        self.statusBar().showMessage(f"已清除 {deleted_count} 个自适应学习记忆文件", 5000)
        if hasattr(self, "adaptive_learning_tab") and hasattr(self.adaptive_learning_tab, "refresh_from_state_files"):
            self.adaptive_learning_tab.refresh_from_state_files()
        QtWidgets.QMessageBox.information(
            self,
            "清除完成",
            f"已清除 {deleted_count} 个学习记忆文件。\n\n建议重启程序以确保所有组件使用新的初始状态。"
        )

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
        if self._paper_start_thread and self._paper_start_thread.isRunning():
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
        
        # ── 禁用按钮，显示加载状态 ──
        self.paper_trading_tab.control_panel.start_btn.setEnabled(False)
        self.paper_trading_tab.control_panel.run_status_label.setText("连接中...")
        self.paper_trading_tab.control_panel.run_status_label.setStyleSheet(
            "color: #FFA500; font-weight: bold;")
        self.statusBar().showMessage("正在连接交易所，请稍候...")

        # ── 准备在后台线程中需要的参数（全部在主线程取好） ──
        http_proxy, socks_proxy = self._get_proxy_settings()
        effective_leverage = int(PAPER_TRADING_CONFIG.get("LEVERAGE_DEFAULT", 20))

        try:
            desired_cold_start = self.adaptive_learning_tab.is_cold_start_enabled()
        except Exception:
            desired_cold_start = False

        evolved_weights = self._gather_evolved_weights()

        trajectory_memory = self.trajectory_memory
        prototype_library = self._prototype_library if has_prototypes else None
        adaptive_controller = self._adaptive_controller

        def _build_engine(emit_progress):
            """在后台线程运行：创建引擎 + start()"""
            from core.live_trading_engine import LiveTradingEngine

            emit_progress("正在初始化交易引擎...")
            engine = LiveTradingEngine(
                trajectory_memory=trajectory_memory,
                prototype_library=prototype_library,
                symbol=config["symbol"],
                interval=config["interval"],
                initial_balance=config["initial_balance"],
                adaptive_controller=adaptive_controller,
                leverage=effective_leverage,
                use_qualified_only=(config.get("use_qualified_only", True) and (not has_prototypes)),
                qualified_fingerprints=qualified_fingerprints,
                qualified_prototype_fingerprints=(verified_proto_fps if use_verified_protos else set()),
                oscillation_filter_enabled=config.get("oscillation_filter_enabled", True),
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

            engine.set_cold_start_enabled(desired_cold_start)
            engine._deepseek_reviewer = None

            for attr_name, val in evolved_weights.items():
                setattr(engine, attr_name, val)

            emit_progress("正在连接交易所...")
            ok = engine.start()
            return engine, ok

        # ── 后台线程启动 ──
        self._paper_start_thread = QtCore.QThread(self)
        worker = PaperTradingStartWorker(_build_engine)
        worker.moveToThread(self._paper_start_thread)
        self._paper_start_worker = worker

        worker.progress.connect(
            lambda msg: self.statusBar().showMessage(msg))
        worker.succeeded.connect(
            lambda engine: self._on_paper_engine_ready(engine, config, has_prototypes, use_verified_protos))
        worker.failed.connect(self._on_paper_engine_failed)
        self._paper_start_thread.started.connect(worker.run)
        worker.succeeded.connect(self._paper_start_thread.quit)
        worker.failed.connect(self._paper_start_thread.quit)
        self._paper_start_thread.start()

    # ── 后台线程回调 ──────────────────────────────────────────────
    def _on_paper_engine_ready(self, engine, config, has_prototypes, use_verified_protos):
        """LiveTradingEngine 在后台创建+start 成功，回到主线程更新 UI"""
        self._live_engine = engine
        self._live_running = True
        self.paper_trading_tab.control_panel.set_running(True)
        using_evolved = getattr(engine, "_using_evolved_weights", False)
        self.paper_trading_tab.control_panel.update_weight_mode(using_evolved)

        self._deepseek_reviewer = None
        if hasattr(engine, "_deepseek_reviewer"):
            engine._deepseek_reviewer = None

        self.paper_trading_tab.reset()
        display_limit = PaperTradingTradeLog.MAX_DISPLAY_TRADES

        def _apply_history(history):
            final_history = history or []
            if not final_history:
                final_history = getattr(engine.paper_trader, "order_history", None) or []
            if final_history:
                engine.paper_trader.order_history = list(final_history)
                display_count = min(display_limit, len(final_history))
                self.paper_trading_tab.load_historical_trades(final_history[-display_count:])
                self.paper_trading_tab.status_panel.append_event(
                    f"成功恢复历史交易记录: 显示{display_count}笔 / 共{len(final_history)}笔"
                )

        self._start_trade_history_io_worker("load", on_loaded=_apply_history)

        self._live_chart_dirty = True
        self._last_live_chart_refresh_ts = 0.0
        self._last_live_state_ui_ts = 0.0
        self._last_live_state_bar_count = -1
        self._last_live_state_order_id = None
        self._last_ui_state_event = ""
        self._live_chart_timer.start()
        self._adaptive_dashboard_timer.start()

        self._init_cold_start_panel_from_engine()

        if getattr(engine, "use_signal_mode", False):
            self.statusBar().showMessage(
                f"模拟交易已启动: {config['symbol']} | 精品+高频信号模式（精品优先）")
        elif has_prototypes:
            mode_msg = f"聚合指纹图模式({'已验证原型' if use_verified_protos else '全原型'})"
            self.statusBar().showMessage(
                f"模拟交易已启动: {config['symbol']} | {mode_msg}")
        else:
            self.statusBar().showMessage(
                f"模拟交易已启动: {config['symbol']} | 模板模式")

    def _on_paper_engine_failed(self, error_msg: str):
        """后台启动失败"""
        self.paper_trading_tab.control_panel.start_btn.setEnabled(True)
        self.paper_trading_tab.control_panel.run_status_label.setText("已停止")
        self.paper_trading_tab.control_panel.run_status_label.setStyleSheet("color: #888;")
        self.statusBar().showMessage("模拟交易启动失败")
        QtWidgets.QMessageBox.critical(
            self, "启动失败", f"模拟交易启动失败:\n{error_msg}")
    
    def _on_paper_trading_stop(self):
        """停止模拟交易"""
        if self._paper_start_thread and self._paper_start_thread.isRunning():
            self._paper_start_thread.quit()
            self._paper_start_thread.wait(3000)
        if self._live_engine:
            # 停止前保存当前交易记录到文件，避免重启后丢失
            trader = getattr(self._live_engine, "paper_trader", None)
            if trader is not None and getattr(trader, "order_history", None) and getattr(trader, "save_history", None):
                path = getattr(trader, "history_file", None)
                if path:
                    try:
                        trader.save_history(path)
                    except Exception as e:
                        print(f"[MainWindow] 停止时保存交易记录失败: {e}")
            self._live_engine.stop()
            # 检查会话结束报告中的调整建议
            self._show_session_end_suggestions()
        
        self._live_running = False
        self._live_chart_timer.stop()
        self._live_chart_dirty = False
        
        # NEW: 停止自适应仪表板刷新
        self._adaptive_dashboard_timer.stop()
        self._deepseek_poll_timer.stop()
        self._deepseek_interval_timer.stop()
        
        # NEW: 停止DeepSeek后台工作线程
        if hasattr(self, '_deepseek_reviewer'):
            self._deepseek_reviewer.stop_background_worker()
        
        # NEW: 保存自适应控制器状态
        if hasattr(self, '_adaptive_controller'):
            self._adaptive_controller.save_state()
        self._adaptive_dashboard_timer.stop()
        self._deepseek_poll_timer.stop()
        self._deepseek_interval_timer.stop()
        
        # 停止DeepSeek后台工作线程
        if self._deepseek_reviewer:
            self._deepseek_reviewer.stop_background_worker()
        self.paper_trading_tab.control_panel.set_running(False)
        self.paper_trading_tab.control_panel.update_weight_mode(None)
        self.statusBar().showMessage("模拟交易已停止")

    def closeEvent(self, event: QtGui.QCloseEvent):
        """窗口关闭时兜底保存交易记录，避免重启丢失"""
        if self._paper_start_thread and self._paper_start_thread.isRunning():
            self._paper_start_thread.quit()
            self._paper_start_thread.wait(3000)
        # 停止回测工作线程，避免 QThread 残留导致崩溃
        if getattr(self, "labeling_worker", None):
            self.labeling_worker.stop()
        if getattr(self, "worker_thread", None) and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        try:
            if self._live_engine:
                trader = getattr(self._live_engine, "paper_trader", None)
                if trader is not None and getattr(trader, "save_history", None):
                    path = getattr(trader, "history_file", None)
                    if path:
                        trader.save_history(path)
                # 关闭前停止引擎，避免后台线程残留
                if getattr(self, "_live_running", False):
                    try:
                        self._live_engine.stop()
                    except Exception as e:
                        print(f"[MainWindow] 关闭时停止引擎失败: {e}")
        except Exception as e:
            print(f"[MainWindow] 关闭时保存交易记录失败: {e}")
        super().closeEvent(event)
    
    def _show_session_end_suggestions(self):
        """
        会话结束时展示调整建议。
        
        如果有待处理的门控阈值调整建议，弹出确认对话框供用户审核。
        按照计划：调整在会话结束时提出，需手动确认。
        """
        if not self._live_engine:
            return
        
        report = self._live_engine.get_session_end_report()
        if not report:
            return
        
        suggestions = report.get("pending_suggestions", [])
        session_adjustments = report.get("session_adjustments", [])
        stats = report.get("statistics", {})
        
        # 如果本次会话有已应用的调整，显示摘要
        if session_adjustments:
            adj_summary = "\n".join(
                f"  · {adj.get('param_key')}: {adj.get('old_value')} → {adj.get('new_value')} "
                f"({adj.get('timestamp_str', '')})"
                for adj in session_adjustments
            )
            self.paper_trading_tab.status_panel.append_event(
                f"会话调整记录 ({len(session_adjustments)}项):\n{adj_summary}"
            )
        
        # 如果有新的待确认建议，弹出对话框
        if not suggestions:
            return
        
        # 日志记录
        self.paper_trading_tab.status_panel.append_event(
            f"📊 会话结束: 总拒绝={stats.get('total_rejections', 0)}, "
            f"总评估={stats.get('total_evaluations', 0)}, "
            f"有 {len(suggestions)} 项阈值调整建议"
        )
        
        # 传递给 RejectionLogCard 并自动弹出确认对话框
        self.adaptive_learning_tab.rejection_log_card.set_suggestions(suggestions)
        self.adaptive_learning_tab.rejection_log_card._on_suggest_clicked()
    
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

            def _on_deleted(removed_count: int, remaining_count: int):
                if removed_count > 0:
                    self.statusBar().showMessage("交易记录已删除", 3000)
                else:
                    self.statusBar().showMessage("未找到匹配交易记录", 3000)

            def _on_failed(msg: str):
                self.statusBar().showMessage("删除交易记录失败", 5000)
                QtWidgets.QMessageBox.warning(
                    self,
                    "删除失败",
                    f"删除交易记录时发生错误:\n{msg}"
                )

            self._start_trade_history_io_worker(
                "delete",
                order=order,
                on_deleted=_on_deleted,
                on_failed=_on_failed
            )
            
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
        """更新实时状态（主线程）— 拆分轻量/重量更新，避免主线程阻塞"""
        now = time.time()

        # ── 轻量更新：价格/连接状态/持仓方向（每次都执行，开销极低） ──
        self.paper_trading_tab.control_panel.update_ws_status(state.is_connected)
        self.paper_trading_tab.control_panel.update_price(state.current_price)
        self.paper_trading_tab.control_panel.update_bar_count(state.total_bars)
        self.paper_trading_tab.control_panel.update_position_direction(state.position_side)

        # ── 节流门控：判断是否需要执行重量更新 ──
        current_order_id = None
        if self._live_engine:
            order_for_gate = self._live_engine.paper_trader.current_position
            if order_for_gate is not None:
                current_order_id = getattr(order_for_gate, "order_id", None)
        bar_changed = (state.total_bars != self._last_live_state_bar_count)
        order_changed = (current_order_id != self._last_live_state_order_id)
        event_changed = ((getattr(state, "last_event", "") or "") != self._last_ui_state_event)

        # 重量更新至少间隔 2 秒；仅在 bar/order/event 变化时触发
        min_heavy_interval = 2.0
        if (not bar_changed and not order_changed and not event_changed and
                now - self._last_live_state_ui_ts < min_heavy_interval):
            return
        self._last_live_state_ui_ts = now
        self._last_live_state_bar_count = state.total_bars
        self._last_live_state_order_id = current_order_id
        self._last_ui_state_event = (getattr(state, "last_event", "") or "")
        
        # 更新持仓
        if self._live_engine:
            order = self._live_engine.paper_trader.current_position
            self.paper_trading_tab.status_panel.update_position(order)
            # 自适应杠杆亮灯：有凯利仓位学习即亮绿（会随学习调整，未必已调过）
            ac = getattr(self._live_engine, "_adaptive_controller", None)
            kelly = getattr(ac, "kelly_adapter", None) if ac else None
            leverage_active = kelly is not None
            self.paper_trading_tab.status_panel.update_adaptive_leverage_lamp(leverage_active)
            # 开仓纪要：首次出现该持仓时记一笔
            if order is not None:
                last_id = getattr(self, "_last_adaptive_position_id", None)
                if order.order_id != last_id:
                    self._last_adaptive_position_id = order.order_id
                    lev = getattr(order, "leverage", None) or (self._live_engine.paper_trader.leverage if self._live_engine.paper_trader else 20)
                    kp = getattr(order, "kelly_position_pct", None)
                    kp_str = f" 凯利{kp:.1%}" if kp and kp > 0 else ""
                    self.adaptive_learning_tab.append_adaptive_journal(
                        f"开仓 {order.side.value} 杠杆{lev}x{kp_str} @ {order.entry_price:,.2f} 数量{order.quantity:.4f}"
                    )
            else:
                self._last_adaptive_position_id = None
            self.paper_trading_tab.status_panel.update_current_price(state.current_price)
            # 更新持仓标记（显示当前持仓在K线上的位置）
            current_idx = getattr(self._live_engine, "_current_bar_idx", None)
            self.paper_trading_tab.update_position_marker(order, current_idx, state.current_price)
            
            # 更新统计
            stats = self._live_engine.get_stats()
            self.paper_trading_tab.status_panel.update_stats(stats)
            # 同步回测指标面板（用于展示当前仓位TP/SL与实时盈亏）
            try:
                last_trade = None
                if self._live_engine and getattr(self._live_engine, "paper_trader", None):
                    hist = getattr(self._live_engine.paper_trader, "order_history", None)
                    if hist:
                        last_trade = hist[-1]
                metrics = {
                    "initial_capital": stats.get("initial_balance", 0.0),
                    "total_profit": stats.get("total_pnl", 0.0),
                    "total_return": (stats.get("total_pnl_pct", 0.0) / 100.0),
                    "win_rate": stats.get("win_rate", 0.0),
                    "max_drawdown": (stats.get("max_drawdown_pct", 0.0) / 100.0),
                    "long_win_rate": stats.get("long_win_rate", 0.0),
                    "short_win_rate": stats.get("short_win_rate", 0.0),
                    "long_profit": 0.0,
                    "short_profit": 0.0,
                    "current_pos": order,
                    "last_trade": last_trade,
                }
                self.optimizer_panel.update_backtest_metrics(metrics)
            except Exception:
                pass
            
            # 更新推理引擎显示（综合决策在 status_panel 的「推理」子 tab）
            status_panel = getattr(self.paper_trading_tab, 'status_panel', None)
            if status_panel and hasattr(status_panel, 'update_reasoning_layers'):
                status_panel.update_reasoning_layers(
                    getattr(state, 'reasoning_result', None),
                    state,
                    order
                )
            self.paper_trading_tab.control_panel.update_account_stats(stats)

            # 更新精品信号模式状态面板
            if getattr(state, 'signal_mode_active', False) or getattr(self._live_engine, 'use_signal_mode', False):
                sm_info = getattr(state, 'signal_mode_info', {}) or {}
                self.paper_trading_tab.status_panel.update_signal_mode_info(sm_info)
                self.paper_trading_tab.control_panel.update_signal_mode_info(sm_info)

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
                position_score=getattr(state, "position_score", 0.0),
                # 【指纹3D图】多维相似度分解
                cosine_similarity=getattr(state, "cosine_similarity", 0.0),
                euclidean_similarity=getattr(state, "euclidean_similarity", 0.0),
                dtw_similarity=getattr(state, "dtw_similarity", 0.0),
                prototype_confidence=getattr(state, "prototype_confidence", 0.0),
                final_match_score=getattr(state, "final_match_score", 0.0),
                cold_start_active=getattr(state, "cold_start_enabled", False),
            )
            # 凯利仓位显示：优先从当前持仓读取，其次从 state 读取
            kelly_pct = 0.0
            if self._live_engine:
                order = self._live_engine.paper_trader.current_position
                if order and hasattr(order, 'kelly_position_pct') and order.kelly_position_pct > 0:
                    kelly_pct = order.kelly_position_pct
                elif getattr(state, "kelly_position_pct", 0.0) > 0:
                    kelly_pct = state.kelly_position_pct
            self.paper_trading_tab.control_panel.update_kelly_position_display(kelly_pct)
            
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
                prototype_confidence=getattr(state, "prototype_confidence", 0.0),
            )
            
            # ── 自适应学习面板仅在 bar 变化时更新（1 分钟级别），避免 tick 级重复刷新 ──
            if not bar_changed:
                # 非 bar 变化：跳过所有重量级自适应学习面板更新，仅更新事件日志和指纹轨迹
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
                last_event = getattr(state, "last_event", "")
                if last_event and last_event != getattr(self, "_last_logged_event", ""):
                    self._last_logged_event = last_event
                    self.paper_trading_tab.status_panel.append_event(last_event)
                self._update_fingerprint_trajectory_overlay(state)
                return

            # ── 以下为 bar 变化时的完整重量更新 ──
            cos_sim = getattr(state, "cosine_similarity", 0.0)
            euc_sim = getattr(state, "euclidean_similarity", 0.0)
            dtw_sim = getattr(state, "dtw_similarity", 0.0)
            fus_sim = getattr(state, "final_match_score", 0.0)
            cold_on = getattr(state, "cold_start_enabled", False)
            if cold_on and self._live_engine:
                try:
                    cs = self._live_engine.get_cold_start_state()
                    th = cs.get("thresholds", {})
                    th_n = cs.get("normal_thresholds", {})
                    fus_th = th.get("fusion", th_n.get("fusion", 0.40))
                    cos_th = th.get("cosine", th_n.get("cosine", 0.70))
                    euc_th = th.get("euclidean", th_n.get("euclidean", 0.35))
                    dtw_th = th.get("dtw", th_n.get("dtw", 0.30))
                except Exception:
                    from config import SIMILARITY_CONFIG, COLD_START_CONFIG
                    th_c = COLD_START_CONFIG.get("THRESHOLDS", {})
                    fus_th = th_c.get("fusion", 0.30)
                    cos_th = th_c.get("cosine", 0.50)
                    euc_th = th_c.get("euclidean", 0.25)
                    dtw_th = th_c.get("dtw", 0.10)
            else:
                from config import SIMILARITY_CONFIG, COLD_START_CONFIG
                fus_th = SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.40)
                cos_th = SIMILARITY_CONFIG.get("COSINE_MIN_THRESHOLD", 0.70)
                euc_th = 0.35
                dtw_th = 0.30
            self.adaptive_learning_tab.update_entry_conditions_adaptive(
                fusion=fus_sim, cosine=cos_sim, euclidean=euc_sim, dtw=dtw_sim,
                fusion_th=fus_th, cos_th=cos_th, euc_th=euc_th, dtw_th=dtw_th,
            )
            # ── 更新自适应学习面板 ──
            rejection_history = getattr(state, "rejection_history", [])
            gate_scores = getattr(state, "gate_scores", {})
            if rejection_history or gate_scores:
                tracker = getattr(self._live_engine, "_rejection_tracker", None)
                expanded = []
                if tracker:
                    from config import PAPER_TRADING_CONFIG as _ptc_rej
                    expanded = tracker.compute_all_concrete_adjustments(_ptc_rej)
                self.adaptive_learning_tab.update_entry_gate(
                    rejection_history, gate_scores, expanded
                )

            # 出场时机
            exit_records = getattr(state, "exit_timing_history", [])
            exit_scores = getattr(state, "exit_timing_scores", {})
            exit_suggestions = []
            tracker = getattr(self._live_engine, "_exit_timing_tracker", None)
            if tracker:
                from config import PAPER_TRADING_CONFIG as _ptc_exit
                state = getattr(self._live_engine, "state", None)
                current_regime = getattr(state, "market_regime", "") or ""
                exit_suggestions = tracker.get_suggestions(_ptc_exit, current_regime=current_regime)
            self.adaptive_learning_tab.update_exit_timing(exit_records, exit_scores, exit_suggestions)

            # 止盈止损
            tpsl_records = getattr(state, "tpsl_history", [])
            tpsl_scores = getattr(state, "tpsl_scores", {})
            tpsl_suggestions = []
            tracker = getattr(self._live_engine, "_tpsl_tracker", None)
            if tracker:
                from config import PAPER_TRADING_CONFIG as _ptc_tpsl
                tpsl_suggestions = tracker.get_suggestions(_ptc_tpsl)
            self.adaptive_learning_tab.update_tpsl(tpsl_records, tpsl_scores, tpsl_suggestions)

            # 近似信号
            nm_records = getattr(state, "near_miss_history", [])
            nm_scores = getattr(state, "near_miss_scores", {})
            nm_suggestions = []
            tracker = getattr(self._live_engine, "_near_miss_tracker", None)
            if tracker:
                from config import PAPER_TRADING_CONFIG as _ptc_nm, SIMILARITY_CONFIG
                fusion_th = None
                pm = getattr(self._live_engine, "_proto_matcher", None)
                if pm:
                    fusion_th = getattr(pm, "fusion_threshold", SIMILARITY_CONFIG.get("FUSION_THRESHOLD", 0.40))
                nm_suggestions = tracker.get_suggestions(_ptc_nm, fusion_threshold=fusion_th)
            self.adaptive_learning_tab.update_near_miss(nm_records, nm_scores, nm_suggestions)

            # 市场状态拦截（来自 RejectionTracker 的 fail_code 过滤）
            regime_records = getattr(state, "regime_history", [])
            regime_scores = getattr(state, "regime_scores", {})
            self.adaptive_learning_tab.update_regime(regime_records, regime_scores, [])

            # 早期出场
            ee_records = getattr(state, "early_exit_history", [])
            ee_scores = getattr(state, "early_exit_scores", {})
            ee_suggestions = []
            tracker = getattr(self._live_engine, "_early_exit_tracker", None)
            if tracker:
                from config import PAPER_TRADING_CONFIG as _ptc_ee
                ee_suggestions = tracker.get_suggestions(_ptc_ee)
            self.adaptive_learning_tab.update_early_exit(ee_records, ee_scores, ee_suggestions)

            # 汇总统计
            summary_total = (
                len(rejection_history) + len(exit_records) + len(tpsl_records)
                + len(nm_records) + len(regime_records) + len(ee_records)
            )
            def _score_acc(scores_dict):
                total = 0
                correct = 0
                for s in (scores_dict or {}).values():
                    total_val = s.get("total", None)
                    if total_val is None:
                        total_val = int(s.get("correct_count", 0)) + int(s.get("wrong_count", 0))
                    total += int(total_val)
                    if "correct" in s:
                        correct += int(s.get("correct", 0))
                    else:
                        correct += int(s.get("correct_count", 0))
                return correct, total
            corr = 0
            tot = 0
            for sd in (gate_scores, exit_scores, tpsl_scores, nm_scores, regime_scores, ee_scores):
                c, t = _score_acc(sd)
                corr += c
                tot += t
            summary_acc = (corr / tot) if tot > 0 else 0.0
            adjustments_applied = getattr(state, "adaptive_adjustments_applied", 0)
            self.adaptive_learning_tab.update_summary(summary_total, summary_acc, adjustments_applied)

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
        self._live_chart_dirty = True
        self._refresh_live_chart(force=False)

    @QtCore.pyqtSlot()
    def _on_live_chart_tick(self):
        """1秒定时刷新K线图，保证时间流动感"""
        if not self._live_running:
            return
        self._refresh_live_chart(force=True)
        
        # 冷启动面板定时刷新（节流：每5秒刷新一次）
        now = time.time()
        if not hasattr(self, "_last_cold_start_refresh"):
            self._last_cold_start_refresh = 0
        if now - self._last_cold_start_refresh >= 5.0:
            self._last_cold_start_refresh = now
            self._refresh_cold_start_panel()

    def _refresh_adaptive_dashboard(self):
        """刷新自适应控制器仪表板"""
        if not self._adaptive_controller:
            return
        
        try:
            # 获取仪表板数据
            dashboard_data = self._adaptive_controller.get_dashboard_data()
            
            # 更新UI
            self.adaptive_learning_tab.update_adaptive_dashboard(dashboard_data)
            
            # 【实时更新】Kelly 和杠杆仪表盘
            if hasattr(self._adaptive_controller, 'kelly_adapter'):
                kelly_params = self._adaptive_controller.kelly_adapter.get_current_parameters()
                kelly_fraction = kelly_params.get("KELLY_FRACTION", 0.25)
                leverage = kelly_params.get("LEVERAGE", 20)
                recent_perf = list(self._adaptive_controller.kelly_adapter.recent_performance) if hasattr(self._adaptive_controller.kelly_adapter, 'recent_performance') else []
                
                # 调用实时更新方法
                if hasattr(self.adaptive_learning_tab, 'update_kelly_leverage_realtime'):
                    self.adaptive_learning_tab.update_kelly_leverage_realtime(
                        kelly_fraction=kelly_fraction,
                        leverage=leverage,
                        recent_performance=recent_perf
                    )
        except Exception as e:
            print(f"[MainWindow] 刷新自适应仪表板失败: {e}")
    
    def _poll_deepseek_reviews(self):
        """轮询DeepSeek复盘结果并更新时间线UI"""
        if not self._live_engine or not hasattr(self._live_engine, '_deepseek_reviewer'):
            return
        
        reviewer = self._live_engine._deepseek_reviewer
        if not reviewer or not reviewer.enabled:
            return
        
        try:
            recent_reviews = reviewer.get_all_reviews(limit=20)
            
            for review in recent_reviews:
                trade_ctx = review.get('trade_context', {})
                order_id = trade_ctx.get('order_id')
                
                if order_id:
                    if not self.adaptive_learning_tab.get_deepseek_review(order_id):
                        self.adaptive_learning_tab.add_deepseek_review(order_id, review)
                        
                        if hasattr(self.adaptive_learning_tab, 'trade_timeline'):
                            self.adaptive_learning_tab.trade_timeline.update_deepseek_review(order_id, review)
        except Exception as e:
            print(f"[MainWindow] 轮询DeepSeek复盘结果失败: {e}")
    
    def _on_deepseek_interval_tick(self):
        """每 2 分钟触发：有持仓请求持仓建议，无持仓请求市场/等待建议。"""
        if not self._live_engine or not getattr(self._live_engine, '_deepseek_reviewer', None) or not self._live_engine._deepseek_reviewer.enabled:
            return
        reviewer = self._live_engine._deepseek_reviewer
        state = self._live_engine.state
        df = getattr(self._live_engine, 'get_df_for_chart', lambda: None)() if self._live_engine else None
        if df is None:
            df = getattr(self._live_engine, '_df_buffer', None)
        order = self._live_engine.paper_trader.current_position if self._live_engine else None
        reasoning = getattr(state, 'reasoning_result', None)
        def on_result(result):
            self._deepseek_interval_result.emit(result)
        if order is not None:
            reviewer.request_holding_advice_async(order, df, state, reasoning, on_result)
        else:
            reviewer.request_idle_advice_async(state, df, on_result)
    
    @QtCore.pyqtSlot(dict)
    def _on_deepseek_interval_result(self, result):
        """主线程：把 2 分钟意见写入 state，供推理/DeepSeek 展示。"""
        if not self._live_engine:
            return
        advice = (result.get("advice") or "").strip()
        judgement = (result.get("judgement") or "").strip()
        self._live_engine.state.deepseek_holding_advice = advice
        self._live_engine.state.deepseek_judgement = judgement
        self._live_engine.state.deepseek_heartbeat = True
    
    def _refresh_adaptive_timeline(self):
        """刷新自适应学习时间线（检查新的复盘结果）- 别名方法"""
        self._poll_deepseek_reviews()
    
    def _trigger_deepseek_review(self, order):
        """触发异步DeepSeek AI复盘分析"""
        if not self._deepseek_reviewer or not self._deepseek_reviewer.enabled:
            return
        
        try:
            # 构建交易上下文
            from core.deepseek_reviewer import TradeContext as DeepSeekTradeContext
            
            # 获取反事实分析结果（如果已完成）
            counterfactual_result = None
            if hasattr(self._live_engine, '_adaptive_controller') and self._live_engine._adaptive_controller:
                # 尝试从最近的诊断历史中获取反事实分析
                from core.adaptive_controller import TradeContext as AdaptiveTradeContext
                
                # 构建完整上下文用于反事实分析
                if hasattr(order, 'entry_snapshot') and hasattr(order, 'exit_snapshot'):
                    # 构建价格历史（简化版）
                    price_history = []
                    if hasattr(order, 'indicator_snapshots_during_hold'):
                        for i, snapshot in enumerate(order.indicator_snapshots_during_hold):
                            if hasattr(snapshot, 'bar_idx') and hasattr(snapshot, 'price'):
                                price_history.append((snapshot.bar_idx, snapshot.price))
                    
                    # 构建AdaptiveTradeContext
                    adaptive_context = AdaptiveTradeContext(
                        order=order,
                        entry_snapshot=order.entry_snapshot,
                        exit_snapshot=order.exit_snapshot,
                        price_history=price_history,
                        indicator_snapshots=getattr(order, 'indicator_snapshots_during_hold', []),
                        profit_pct=order.profit_pct,
                        hold_bars=order.hold_bars,
                        close_reason=order.close_reason
                    )
                    
                    # 执行反事实分析
                    counterfactual = self._adaptive_controller.counterfactual_analysis(adaptive_context)
                    counterfactual_result = {
                        'scenario': counterfactual.scenario,
                        'entry_improvement_pct': counterfactual.entry_improvement_pct,
                        'exit_improvement_pct': counterfactual.exit_improvement_pct,
                        'sl_tp_improvement_pct': counterfactual.sl_tp_improvement_pct,
                        'total_improvement_pct': counterfactual.total_improvement_pct,
                        'better_entry_bar': counterfactual.better_entry_bar,
                        'optimal_sl_atr': counterfactual.optimal_sl_atr,
                    }
            
            # 获取原型统计（如果有）
            prototype_stats = None
            if hasattr(order, 'template_fingerprint') and order.template_fingerprint:
                # 尝试从引擎获取原型统计
                if self._live_engine and hasattr(self._live_engine, 'get_template_stats'):
                    prototype_stats = self._live_engine.get_template_stats(order.template_fingerprint)
            
            # 构建DeepSeek交易上下文
            trade_context = DeepSeekTradeContext.from_order(
                order,
                counterfactual_result=counterfactual_result,
                prototype_stats=prototype_stats,
                feature_patterns=None  # TODO: 从FeaturePatternDB获取
            )
            
            # 添加到异步复盘队列
            self._deepseek_reviewer.add_trade_for_review(trade_context)
            
            print(f"[MainWindow] 已添加交易 {order.order_id} 到DeepSeek复盘队列")
        except Exception as e:
            print(f"[MainWindow] 触发DeepSeek复盘失败: {e}")
            import traceback
            traceback.print_exc()

    def _refresh_live_chart(self, force: bool = False):
        """统一刷新实时图表"""
        if not self._live_engine:
            return
        now = time.time()
        if not force:
            if now - self._last_live_chart_refresh_ts < self._live_chart_min_interval_sec:
                return
        else:
            # 定时器只在有新数据时刷新，避免空转重绘
            if not self._live_chart_dirty:
                return
            if now - self._last_live_chart_refresh_ts < self._live_chart_min_interval_sec:
                return
        
        try:
            # 获取历史K线数据
            df = self._live_engine.get_history_df()
            if df.empty:
                return
            
            # 更新模拟交易Tab的图表 (使用增量更新，避免重置信号标记)
            self.paper_trading_tab.chart_widget.update_kline(df)
            
            # 视图随K线滚动更新（仅在 K 线增加时滚动，避免交易标记被挤出视野）
            n = len(df)
            if not hasattr(self, "_last_live_n") or n > self._last_live_n:
                self._last_live_n = n
                visible = 50
                future_pad = 0
                if hasattr(self.paper_trading_tab.chart_widget, "get_overlay_padding"):
                    future_pad = self.paper_trading_tab.chart_widget.get_overlay_padding()
                x_left = n - visible
                if hasattr(self.paper_trading_tab.chart_widget, "get_rightmost_signal_index"):
                    rightmost = self.paper_trading_tab.chart_widget.get_rightmost_signal_index()
                    if rightmost >= 0 and rightmost < x_left:
                        x_left = max(0, rightmost - 10)
                self.paper_trading_tab.chart_widget.candle_plot.setXRange(
                    x_left, n + max(5, max(0, future_pad)), padding=0
                )
            
            # 实时更新 TP/SL 虚线位置（仅使用本地缓存，不触发 HTTP 请求）
            order = self._live_engine._paper_trader.current_position
            if order is not None:
                tp = getattr(order, "take_profit", None)
                sl = getattr(order, "stop_loss", None)
                # 锁定为真实委托价，避免图表被历史高低点覆盖
                self.paper_trading_tab.chart_widget._tp_sl_locked = True
                self.paper_trading_tab.chart_widget.set_tp_sl_lines(tp, sl)
                
                # 【实时偏离检测】持仓中检查价格是否偏离概率扇形置信带
                self._check_deviation_warning(df)
            else:
                # 无持仓时清除虚线
                self.paper_trading_tab.chart_widget._tp_sl_locked = False
                self.paper_trading_tab.chart_widget.set_tp_sl_lines(None, None)
            self._last_live_chart_refresh_ts = now
            self._live_chart_dirty = False
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

    def _append_kdj_macd_prediction_suffix(self, label: str, df) -> str:
        """
        为未来走势预测标签追加 KDJ/MACD 情境与秒级分析说明，使预测更严谨可解释。
        """
        if df is None or len(df) == 0:
            return label + " | 按最新K线实时更新"
        row = df.iloc[-1]
        suf = []
        j_val = row.get("j")
        if j_val is not None and not (isinstance(j_val, (int, float)) and getattr(np, "isnan", lambda x: False)(float(j_val))):
            try:
                j_f = float(j_val)
                if not np.isnan(j_f):
                    kdj_dir = "多" if j_f > 50 else "空"
                    suf.append(f"KDJ J={j_f:.0f}({kdj_dir})")
            except (TypeError, ValueError):
                pass
        hist = row.get("macd_hist")
        if hist is not None:
            try:
                h_f = float(hist)
                if not np.isnan(h_f):
                    macd_dir = "柱正" if h_f > 0 else "柱负"
                    suf.append(f"MACD {macd_dir}")
            except (TypeError, ValueError):
                pass
        if suf:
            label = f"{label} | {' '.join(suf)}"
        return label + " | 实时更新(秒级分析)"

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
        
        # 构建多维相似度分解字典（从引擎状态提取）
        similarity_breakdown = None
        if state:
            cos_sim = getattr(state, "cosine_similarity", 0.0)
            euc_sim = getattr(state, "euclidean_similarity", 0.0)
            dtw_sim = getattr(state, "dtw_similarity", 0.0)
            confidence = getattr(state, "prototype_confidence", 0.0)
            final_score = getattr(state, "final_match_score", 0.0)
            
            # 只有当有多维数据时才构建 breakdown
            if cos_sim > 0 or euc_sim > 0 or dtw_sim > 0:
                similarity_breakdown = {
                    "combined_score": matched_sim or 0.0,
                    "cosine_similarity": cos_sim,
                    "euclidean_similarity": euc_sim,
                    "dtw_similarity": dtw_sim,
                    "confidence": confidence,
                    "final_score": final_score if final_score > 0 else (matched_sim or 0.0),
                }
        
        # 优先绘制概率扇形图（原型模式）
        if proto is not None:
            member_stats = getattr(proto, "member_trade_stats", [])
            if not member_stats or len(member_stats) < 3:
                member_stats = self._synthesize_member_stats(proto)
            
            if member_stats and len(member_stats) >= 3:
                direction = proto.direction
                regime_short = proto.regime[:2] if proto.regime else ""
                label = f"{direction} {regime_short}_{proto.prototype_id}"
                label = self._append_kdj_macd_prediction_suffix(label, df)
                current_price = float(df["close"].iloc[-1])
                leverage = getattr(self._live_engine, "fixed_leverage", 20.0)
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
                    similarity_breakdown=similarity_breakdown,
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
        label = self._append_kdj_macd_prediction_suffix(label, df)
        chart.set_fingerprint_trajectory(
            prices, start_idx, matched_sim or 0.0, label,
            lower=lower, upper=upper,
            similarity_breakdown=similarity_breakdown
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
            # 添加图表标记（缺省时用当前 K 线索引，避免标记不显示）
            side = order.side.value
            bar_idx = getattr(order, "entry_bar_idx", None)
            if bar_idx is None and self._live_engine:
                df = getattr(self._live_engine, "get_history_df", lambda: None)()
                if df is not None and not df.empty:
                    bar_idx = len(df) - 1
            if bar_idx is None:
                bar_idx = getattr(self._live_engine, "_current_bar_idx", 0)
            self.paper_trading_tab.add_trade_marker(
                bar_idx=bar_idx,
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
            if (exit_bar is None or exit_px is None) and self._live_engine:
                df = getattr(self._live_engine, "get_history_df", lambda: None)()
                if df is not None and not df.empty:
                    if exit_bar is None:
                        exit_bar = len(df) - 1
                    if exit_px is None:
                        exit_px = float(df["close"].iloc[-1])

            # 根据真实平仓原因 + 追踪阶段 确定标记类型
            close_reason_str = None
            if order.close_reason:
                reason_val = order.close_reason.value  # "止盈"/"止损"/"分段止盈"/"分段止损"/"脱轨"/"超时"/"信号"/"手动"
                trailing = getattr(order, "trailing_stage", 0)
                if reason_val in ("分段止盈", "分段止损"):
                    close_reason_str = reason_val
                elif reason_val == "止盈" and trailing >= 1 and order.profit_pct < 0.3:
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
            
            # ── 添加到时间线UI ──
            if hasattr(self, 'adaptive_learning_tab'):
                try:
                    # DeepSeek复盘结果稍后会通过定时器轮询获取并更新
                    # 此处先添加交易到时间线，复盘结果会异步更新
                    self.adaptive_learning_tab.add_trade_to_timeline(order, deepseek_review=None)
                except Exception as e:
                    print(f"[MainWindow] 添加交易到时间线失败: {e}")
            
            # 记录事件（使用细化后的平仓原因；若有 close_reason_detail 则一并展示）
            reason_display = close_reason_str or (order.close_reason.value if order.close_reason else "未知")
            detail = getattr(order, "close_reason_detail", "") or ""
            if detail:
                reason_display = f"{reason_display}({detail})"
            profit_color = "盈利" if order.profit_pct >= 0 else "亏损"
            pnl_usdt = getattr(order, "realized_pnl", 0.0)
            event_msg = (
                f"[平仓] {side} @ {order.exit_price:.2f} | "
                f"{profit_color} {order.profit_pct:+.2f}% ({pnl_usdt:+.2f} USDT) | "
                f"原因={reason_display} | 持仓={order.hold_bars}根K线"
            )
            self.paper_trading_tab.status_panel.append_event(event_msg)
            
            # 自适应页·开平仓纪要：平仓一条
            reason_display = close_reason_str or (order.close_reason.value if order.close_reason else "未知")
            self.adaptive_learning_tab.append_adaptive_journal(
                f"平仓 {side} 盈亏{order.profit_pct:+.2f}% ({getattr(order, 'realized_pnl', 0):+.2f} USDT) 原因={reason_display} 持仓{order.hold_bars}根K"
            )
            
            # NEW: 添加交易到时间线
            self.adaptive_learning_tab.add_trade_to_timeline(order, deepseek_review=None)
            
            # NEW: 触发异步DeepSeek AI复盘
            self._trigger_deepseek_review(order)
            
            # NEW: 立即刷新自适应仪表板
            self._refresh_adaptive_dashboard()
            
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
    
    def _build_adaptive_trade_context(self, order) -> Optional[AdaptiveTradeContext]:
        """
        为自适应控制器构建交易上下文
        
        Args:
            order: PaperOrder 对象
        
        Returns:
            AdaptiveTradeContext 或 None
        """
        try:
            # 获取决策快照
            entry_snapshot = getattr(order, 'entry_snapshot', None)
            exit_snapshot = getattr(order, 'exit_snapshot', None)
            
            if not entry_snapshot:
                print(f"[MainWindow] 订单 {order.order_id} 缺少入场快照，跳过自适应分析")
                return None
            
            # 获取持仓期间的指标快照
            indicator_snapshots = getattr(order, 'indicator_snapshots', [])
            
            # 获取价格历史（从引擎的价格缓存中提取）
            price_history = []
            if self._live_engine and hasattr(self._live_engine, '_price_history'):
                # 提取该订单持仓期间的价格
                entry_bar = order.entry_bar_idx
                exit_bar = order.exit_bar_idx or order.entry_bar_idx
                for bar_idx in range(entry_bar, exit_bar + 1):
                    if bar_idx in self._live_engine._price_history:
                        price = self._live_engine._price_history[bar_idx]
                        price_history.append((bar_idx, price))
            
            # 构建上下文
            context = AdaptiveTradeContext(
                order=order,
                entry_snapshot=entry_snapshot,
                exit_snapshot=exit_snapshot,
                price_history=price_history,
                indicator_snapshots=indicator_snapshots,
                profit_pct=order.profit_pct,
                hold_bars=order.hold_bars,
                close_reason=order.close_reason,
            )
            
            return context
        
        except Exception as e:
            print(f"[MainWindow] 构建自适应交易上下文失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_deepseek_trade_context(self, order) -> Optional[DeepSeekTradeContext]:
        """
        为DeepSeek构建交易上下文
        
        Args:
            order: PaperOrder 对象
        
        Returns:
            DeepSeekTradeContext 或 None
        """
        try:
            # 获取快照
            entry_snapshot = getattr(order, 'entry_snapshot', None)
            exit_snapshot = getattr(order, 'exit_snapshot', None)
            
            if not entry_snapshot:
                return None
            
            # 转换为字典
            entry_snapshot_dict = {}
            if hasattr(entry_snapshot, 'to_dict'):
                entry_snapshot_dict = entry_snapshot.to_dict()
            elif isinstance(entry_snapshot, dict):
                entry_snapshot_dict = entry_snapshot
            
            exit_snapshot_dict = {}
            if exit_snapshot:
                if hasattr(exit_snapshot, 'to_dict'):
                    exit_snapshot_dict = exit_snapshot.to_dict()
                elif isinstance(exit_snapshot, dict):
                    exit_snapshot_dict = exit_snapshot
            
            # 获取原型统计（如果有）
            prototype_stats = None
            if hasattr(order, 'template_fingerprint') and order.template_fingerprint:
                # 从引擎获取该原型的历史表现
                if self._live_engine and hasattr(self._live_engine, 'get_template_stats'):
                    try:
                        prototype_stats = self._live_engine.get_template_stats(order.template_fingerprint)
                    except:
                        pass
            
            # 获取反事实分析结果（从adaptive_controller）
            counterfactual_result = None
            if self._adaptive_controller and hasattr(self._adaptive_controller, 'diagnosis_history'):
                # 查找最近的诊断记录
                for diag_entry in reversed(list(self._adaptive_controller.diagnosis_history)):
                    if diag_entry.get('order_id') == order.order_id:
                        # 找到了，提取反事实结果（如果有保存）
                        break
            
            # 构建上下文
            context = DeepSeekTradeContext.from_order(
                order,
                counterfactual_result=counterfactual_result,
                prototype_stats=prototype_stats,
                feature_patterns=None,  # 可以从 feature_db 提取
            )
            
            return context
        
        except Exception as e:
            print(f"[MainWindow] 构建DeepSeek交易上下文失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _expand_adjustment_suggestions(self, raw_suggestions: list) -> list:
        """
        将 RejectionTracker.suggest_all_adjustments() 的输出
        展开为 UI 对话框所需的逐参数条目格式。
        
        已弃用：推荐直接使用 RejectionTracker.compute_all_concrete_adjustments()，
        该方法集中了步长计算、安全边界夹紧等逻辑。
        此方法保留作为回退兼容。
        """
        from config import PAPER_TRADING_CONFIG
        # 委托给 tracker 的集中化方法
        tracker = getattr(self._live_engine, "_rejection_tracker", None) if self._live_engine else None
        if tracker:
            return tracker.compute_all_concrete_adjustments(PAPER_TRADING_CONFIG)
        
        # 回退：引擎不存在时使用原始逻辑
        expanded = []
        for sug in (raw_suggestions or []):
            action = sug.get("action", "")
            action_text = "放宽" if action == "loosen" else "收紧"
            params = sug.get("adjustable_params", {})
            for param_key, param_def in params.items():
                current_val = PAPER_TRADING_CONFIG.get(param_key)
                if current_val is None:
                    continue
                step = param_def.get("loosen_step" if action == "loosen" else "tighten_step", 0)
                new_val = current_val + step
                new_val = max(param_def.get("min", new_val), min(param_def.get("max", new_val), new_val))
                if new_val == current_val:
                    continue
                expanded.append({
                    "fail_code": sug.get("fail_code", ""),
                    "param_key": param_key,
                    "action": action,
                    "action_text": action_text,
                    "label": param_key.replace("_", " ").title(),
                    "current_value": current_val,
                    "suggested_value": new_val,
                    "accuracy": sug.get("accuracy", 0),
                    "reason": sug.get("reason", ""),
                })
        return expanded
    
    def _on_rejection_adjustment_confirmed(self, param_key: str, new_value: float):
        """
        用户在 RejectionLogCard 确认对话框中确认应用某个阈值调整。
        通过引擎的拒绝跟踪器应用（带审计日志和安全边界夹紧）。
        """
        if self._live_engine:
            # 通过引擎接口应用（带审计日志）
            result = self._live_engine.apply_threshold_adjustment(
                param_key=param_key,
                new_value=new_value,
                reason="用户手动确认应用",
            )
            if result:
                old_val = result.get("old_value", "?")
                new_val = result.get("new_value", new_value)
                msg = f"门控阈值调整: {param_key} {old_val} → {new_val} (已审计)"
            else:
                msg = f"门控阈值调整跳过: {param_key} (已在目标值或参数不存在)"
        else:
            # 引擎不存在时直接修改配置（无审计）
            from config import PAPER_TRADING_CONFIG
            old_value = PAPER_TRADING_CONFIG.get(param_key)
            if old_value is None:
                return
            PAPER_TRADING_CONFIG[param_key] = new_value
            msg = f"门控阈值调整: {param_key} {old_value} → {new_value} (无审计)"
        
        print(f"[MainWindow] {msg}")
        self.paper_trading_tab.status_panel.append_event(msg)
        self.statusBar().showMessage(msg, 5000)

    def _apply_adaptive_adjustment(self, source: str, param_key: str, new_value: float, label: str):
        if self._live_engine and hasattr(self._live_engine, "apply_adaptive_adjustment"):
            result = self._live_engine.apply_adaptive_adjustment(
                source=source,
                param_key=param_key,
                new_value=new_value,
                reason="用户手动确认应用",
            )
            if result:
                old_val = result.get("old_value", "?")
                new_val = result.get("new_value", new_value)
                msg = f"{label} 调整: {param_key} {old_val} → {new_val} (已审计)"
            else:
                msg = f"{label} 调整跳过: {param_key} (已在目标值或参数不存在)"
        else:
            msg = f"{label} 调整未应用: 引擎未运行"
        print(f"[MainWindow] {msg}")
        self.paper_trading_tab.status_panel.append_event(msg)
        self.statusBar().showMessage(msg, 5000)

    def _on_exit_timing_adjustment_confirmed(self, param_key: str, new_value: float):
        self._apply_adaptive_adjustment("exit_timing", param_key, new_value, "出场时机")

    def _on_tpsl_adjustment_confirmed(self, param_key: str, new_value: float):
        self._apply_adaptive_adjustment("tpsl", param_key, new_value, "止盈止损")

    def _on_near_miss_adjustment_confirmed(self, param_key: str, new_value: float):
        self._apply_adaptive_adjustment("near_miss", param_key, new_value, "近似信号")

    def _on_regime_adjustment_confirmed(self, param_key: str, new_value: float):
        self._apply_adaptive_adjustment("regime", param_key, new_value, "市场状态")

    def _on_early_exit_adjustment_confirmed(self, param_key: str, new_value: float):
        self._apply_adaptive_adjustment("early_exit", param_key, new_value, "早期出场")
    
    # ─── 冷启动系统 ─────────────────────────────────
    
    def _on_cold_start_toggled(self, enabled: bool):
        """处理冷启动模式开关切换"""
        if not self._live_engine:
            # 引擎未运行时，持久化到文件，确保下次启动时正确加载 50% 阈值
            from core.cold_start_manager import ColdStartManager
            ColdStartManager.persist_enabled_state(enabled)
            self.statusBar().showMessage(
                f"冷启动模式{'已启用' if enabled else '已关闭'}（已保存，下次启动时生效）",
                3000
            )
            return
        
        # 调用引擎设置冷启动状态
        self._live_engine.set_cold_start_enabled(enabled)
        
        # 刷新UI显示
        self._refresh_cold_start_panel()
        
        self.statusBar().showMessage(
            f"冷启动模式{'已启用' if enabled else '已关闭'}",
            3000
        )
    
    def _refresh_cold_start_panel(self):
        """刷新冷启动面板状态"""
        if not self._live_engine:
            return
        
        try:
            # 获取冷启动状态
            cold_state = self._live_engine.get_cold_start_state()
            
            # 更新开关状态（不触发信号）
            self.adaptive_learning_tab.set_cold_start_enabled(
                cold_state.get("enabled", False)
            )
            
            # 更新门槛值
            thresholds = cold_state.get("thresholds", {})
            normal_thresholds = cold_state.get("normal_thresholds", {})
            self.adaptive_learning_tab.update_cold_start_thresholds(
                fusion=thresholds.get("fusion", 0.65),
                cosine=thresholds.get("cosine", 0.70),
                euclidean=thresholds.get("euclidean", 0.35),
                dtw=thresholds.get("dtw", 0.30),
                normal_thresholds=normal_thresholds,
            )
            
            # 更新频率监控
            freq = cold_state.get("frequency", {})
            last_trade_ts = freq.get("last_trade_time")
            last_trade_time = None
            if last_trade_ts is not None:
                # 从 minutes_since_last_trade 推算 datetime
                minutes_since = freq.get("minutes_since_last_trade")
                if minutes_since is not None:
                    from datetime import datetime, timedelta
                    last_trade_time = datetime.now() - timedelta(minutes=minutes_since)
            
            trades_today = freq.get("trades_today", 0)
            trades_per_hour = freq.get("trades_per_hour", 0.0)
            
            # 映射状态
            status_map = {
                "正常": "normal",
                "偏低": "low",
                "频率过低": "warning",
                "关闭": "normal",
            }
            raw_status = freq.get("status", "正常")
            mapped_status = status_map.get(raw_status, "normal")
            
            self.adaptive_learning_tab.update_cold_start_frequency(
                last_trade_time=last_trade_time,
                today_trades=trades_today,
                trades_per_hour=trades_per_hour,
                status=mapped_status,
            )
            
            # 检测自动放宽通知（比较 auto_relax_count 变化）
            auto_relax_count = freq.get("auto_relax_count", 0)
            if not hasattr(self, "_last_auto_relax_count"):
                self._last_auto_relax_count = 0
            
            if auto_relax_count > self._last_auto_relax_count:
                # 发生了新的自动放宽，显示通知
                self._last_auto_relax_count = auto_relax_count
                self.adaptive_learning_tab.show_cold_start_auto_relax(
                    f"交易频率过低，门槛已自动放宽5% (第{auto_relax_count}次)"
                )
            
        except Exception as e:
            print(f"[MainWindow] 刷新冷启动面板失败: {e}")
    
    def _init_cold_start_panel_from_engine(self):
        """从引擎状态初始化冷启动面板"""
        if not self._live_engine:
            return
        
        try:
            cold_state = self._live_engine.get_cold_start_state()
            
            # 设置开关状态（不触发信号）
            self.adaptive_learning_tab.set_cold_start_enabled(
                cold_state.get("enabled", False)
            )
            
            # 初始化 auto_relax_count 跟踪器（避免启动时误触发通知）
            freq = cold_state.get("frequency", {})
            self._last_auto_relax_count = freq.get("auto_relax_count", 0)
            
            # 隐藏之前可能遗留的通知
            self.adaptive_learning_tab.hide_cold_start_auto_relax()
            
            # 初始刷新一次
            self._refresh_cold_start_panel()
            
        except Exception as e:
            print(f"[MainWindow] 初始化冷启动面板失败: {e}")
    
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
