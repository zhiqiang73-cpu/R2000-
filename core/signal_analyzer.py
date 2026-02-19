"""
core/signal_analyzer.py
历史数据信号组合分析器（三层提速版本 v3.1）

三层提速策略：
  1. 预计算 outcome 数组 — 一次性扫描 O(N×240)，组合评估直接查表
  2. NumPy 向量化 AND — 条件组合触发检测纯向量运算，单次 < 100 μs
  3. 单条件预剪枝 — 仅按触发次数过滤（MIN_TRIGGERS），不做命中率筛选
     原因：数据量较大时（50000根）大数定律使单条件命中率收敛至随机基准，
     命中率剪枝会误杀所有条件；改为只保留有足够触发次数的条件，
     让最终 64/67/71%（做多）/ 52/55/59%（做空）组合门槛完成筛选。

回测精度规范（v3.1）：
  - 入场：信号K线下一根的 open（市价单）
  - 触发判断：用影线 bar.high / bar.low（限价单到价即成交）
  - 同K线双触发：悲观假设，一律按止损处理（outcomes[i] = False）
  - 超时：240 根内未触发 → 未命中

费率假设（Binance 合约标准）：
  - 单次往返合计 ~0.06% 名义仓位
  - 含费后零期望基准：做多 61%，做空 47%
  - 做多候选 / 优质 / 精品门槛：64% / 67% / 71%
  - 做空候选 / 优质 / 精品门槛：52% / 55% / 59%
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.market_state_detector import detect_state

# ── 回测参数 ────────────────────────────────────────────────────────────────
MAX_HOLD  = 240     # 最大持仓根数（超时则算未命中）

# 做多参数（原有，TP+0.6% / SL−0.8%）
LONG_TP1_PCT = 0.006   # 价格+0.6%
LONG_SL_PCT  = 0.008   # 价格−0.8%

# 做空参数（新增，TP−0.8% / SL+0.6%）
SHORT_TP1_PCT = 0.008   # 价格−0.8%（做空方向的止盈）
SHORT_SL_PCT  = 0.006   # 价格+0.6%（做空方向的止损）

# 保留旧名称以兼容外部引用
TP1_PCT = LONG_TP1_PCT
SL_PCT  = LONG_SL_PCT

# ── 筛选门槛（含手续费）─────────────────────────────────────────────────────
# 做多：含费后随机基准 61%
TIER_RANDOM_BASELINE = 0.61   # 含费后随机基准（做多）
TIER_CANDIDATE       = 0.64   # 做多候选组合
TIER_QUALITY         = 0.67   # 做多优质组合
TIER_PREMIUM         = 0.71   # 做多精品组合

# 做空：含费后随机基准 47%（止盈 0.8% - 0.06% = 0.74%，止损 0.6% + 0.06% = 0.66%）
SHORT_TIER_CANDIDATE = 0.52   # 做空候选组合
SHORT_TIER_QUALITY   = 0.55   # 做空优质组合
SHORT_TIER_PREMIUM   = 0.59   # 做空精品组合

MIN_TRIGGERS         = 8      # 最少触发次数（低于此不算出现）
WARN_TRIGGERS_MAX    = 14     # 样本偏少警告上限

# ── 单条件剪枝：仅保留触发次数 >= MIN_TRIGGERS 的条件，不做命中率筛选 ────────
# 数据量大（如50000根）时命中率收敛至随机基准，命中率剪枝会误杀所有条件
PRUNE_MIN_RATE = None   # 已弃用，保留仅为兼容性，实际不使用


# ═══════════════════════════════════════════════════════════════════════════
# 策略 1：预计算 outcome 数组
# ═══════════════════════════════════════════════════════════════════════════

def _precompute_outcomes(
    high: np.ndarray,
    low: np.ndarray,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    direction: str,
    tp_pct: float = TP1_PCT,
    sl_pct: float = SL_PCT,
    max_hold: int = MAX_HOLD,
) -> np.ndarray:
    """
    对每个 bar i 预计算：若在 bar i 收盘后发出信号、于 bar i+1 开盘入场，
    最终止盈先触发(True)还是止损先触发/超时(False)。

    同K线双触发规则：
      - 悲观假设：同K线同时触及 TP 和 SL → 一律视为止损（False）

    Returns:
        shape=(n,) 的 bool 数组；outcomes[n-1] 恒为 False（无法入场）
    """
    n = len(high)
    outcomes = np.zeros(n, dtype=bool)

    for i in range(n - 1):
        entry_idx = i + 1
        entry_price = open_arr[entry_idx]
        if entry_price <= 0.0:
            continue

        if direction == 'long':
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)
        else:
            tp_price = entry_price * (1.0 - tp_pct)
            sl_price = entry_price * (1.0 + sl_pct)

        end_idx  = min(entry_idx + max_hold, n)
        w_high   = high[entry_idx:end_idx]
        w_low    = low[entry_idx:end_idx]
        w_open   = open_arr[entry_idx:end_idx]
        w_close  = close_arr[entry_idx:end_idx]

        if direction == 'long':
            tp_hits = w_high >= tp_price
            sl_hits = w_low  <= sl_price
        else:
            tp_hits = w_low  <= tp_price
            sl_hits = w_high >= sl_price

        either = tp_hits | sl_hits
        if not either.any():
            # 超时：算未命中
            outcomes[i] = False
            continue

        first = int(np.argmax(either))

        if tp_hits[first] and sl_hits[first]:
            outcomes[i] = False   # 双触发，悲观假设，按止损处理
        else:
            outcomes[i] = bool(tp_hits[first])

    return outcomes


# ═══════════════════════════════════════════════════════════════════════════
# 策略 2：预计算 30 个条件 bool 数组
# ═══════════════════════════════════════════════════════════════════════════

def _build_condition_arrays(df: pd.DataFrame, direction: str) -> Dict[str, np.ndarray]:
    """
    根据 df 中已计算好的技术指标，构建该方向的 30 个条件 bool 数组。

    极端条件：每方向 9 指标 × 2 阈值（宽松/严格）= 18 个条件
    温和条件：12 个（见下方温和型指标块），shape=(n,)。
    NaN 值（滚动窗口初始段）自动视为条件不满足（False）。

    依赖列：rsi, k, j, boll_position, volume_ratio, close_vs_ma5, atr,
            high, low, open, close,
            macd_hist, ma5_slope, obv, obv_ma
    """
    n = len(df)

    # ── 派生指标（indicators.py 中未直接提供的）─────────────────────────────
    body_abs     = (df['close'] - df['open']).abs().values
    lower_shadow = (df[['open', 'close']].min(axis=1) - df['low']).clip(lower=0).values
    upper_shadow = (df['high'] - df[['open', 'close']].max(axis=1)).clip(lower=0).values
    lower_shd_r  = lower_shadow / (body_abs + 1e-9)
    upper_shd_r  = upper_shadow / (body_abs + 1e-9)

    bar_range = (df['high'] - df['low']).values.astype(float)
    atr_vals  = df['atr'].values.astype(float) if 'atr' in df.columns else np.ones(n)
    atr_ratio = bar_range / (atr_vals + 1e-9)

    is_bearish_s = (df['close'] < df['open']).astype(int)
    is_bullish_s = (df['close'] >= df['open']).astype(int)
    bear_roll2 = is_bearish_s.rolling(2).sum().fillna(0).values >= 2
    bear_roll3 = is_bearish_s.rolling(3).sum().fillna(0).values >= 3
    bull_roll2 = is_bullish_s.rolling(2).sum().fillna(0).values >= 2
    bull_roll3 = is_bullish_s.rolling(3).sum().fillna(0).values >= 3

    # ── 主要指标 ────────────────────────────────────────────────────────────
    def _col(name: str, default: float) -> np.ndarray:
        return df[name].values.astype(float) if name in df.columns else np.full(n, default)

    rsi    = _col('rsi',          50.0)
    k_val  = _col('k',            50.0)
    j_val  = _col('j',            50.0)
    bpos   = _col('boll_position', 0.5)
    vratio = _col('volume_ratio',  1.0)
    cvma5  = _col('close_vs_ma5',  0.0)

    # 温和型指标
    macd_hist      = _col('macd_hist',  0.0)
    macd_hist_prev = np.roll(macd_hist, 1); macd_hist_prev[0] = 0.0
    ma5_slope      = _col('ma5_slope',  0.0)
    obv_vals       = _col('obv',        0.0)
    obv_ma_vals    = _col('obv_ma',     0.0)

    conds: Dict[str, np.ndarray] = {}

    if direction == 'long':
        conds['rsi_loose']           = rsi    < 40
        conds['rsi_strict']          = rsi    < 30
        conds['k_loose']             = k_val  < 35
        conds['k_strict']            = k_val  < 25
        conds['j_loose']             = j_val  < 25
        conds['j_strict']            = j_val  < 15
        conds['boll_pos_loose']      = bpos   < 0.25
        conds['boll_pos_strict']     = bpos   < 0.15
        conds['vol_ratio_loose']     = vratio > 1.3
        conds['vol_ratio_strict']    = vratio > 1.8
        conds['close_vs_ma5_loose']  = cvma5  < -1.0
        conds['close_vs_ma5_strict'] = cvma5  < -1.5
        conds['lower_shd_loose']     = lower_shd_r > 1.5
        conds['lower_shd_strict']    = lower_shd_r > 2.5
        conds['consec_bear_loose']   = bear_roll2
        conds['consec_bear_strict']  = bear_roll3
        conds['atr_ratio_loose']     = atr_ratio   > 1.2
        conds['atr_ratio_strict']    = atr_ratio   > 1.8
        # 温和型条件
        conds['rsi_mod_loose']           = rsi       < 50
        conds['rsi_mod_strict']          = rsi       < 45
        conds['boll_pos_mod_loose']      = bpos      < 0.40
        conds['boll_pos_mod_strict']     = bpos      < 0.35
        conds['macd_hist_turn_loose']    = macd_hist > 0
        conds['macd_hist_turn_strict']   = (macd_hist > 0) & (macd_hist_prev <= 0)
        conds['close_vs_ma5_mod_loose']  = cvma5     < -0.3
        conds['close_vs_ma5_mod_strict'] = cvma5     < -0.5
        conds['ma5_slope_dir_loose']     = ma5_slope < 0
        conds['ma5_slope_dir_strict']    = ma5_slope < -0.05
        conds['obv_trend_loose']         = obv_vals  > obv_ma_vals
        conds['vol_ratio_mod_loose']     = vratio    > 1.1
    else:
        conds['rsi_loose']           = rsi    > 60
        conds['rsi_strict']          = rsi    > 70
        conds['k_loose']             = k_val  > 65
        conds['k_strict']            = k_val  > 75
        conds['j_loose']             = j_val  > 75
        conds['j_strict']            = j_val  > 85
        conds['boll_pos_loose']      = bpos   > 0.75
        conds['boll_pos_strict']     = bpos   > 0.85
        conds['vol_ratio_loose']     = vratio > 1.3
        conds['vol_ratio_strict']    = vratio > 1.8
        conds['close_vs_ma5_loose']  = cvma5  > 1.0
        conds['close_vs_ma5_strict'] = cvma5  > 1.5
        conds['upper_shd_loose']     = upper_shd_r > 1.5
        conds['upper_shd_strict']    = upper_shd_r > 2.5
        conds['consec_bull_loose']   = bull_roll2
        conds['consec_bull_strict']  = bull_roll3
        conds['atr_ratio_loose']     = atr_ratio   > 1.2
        conds['atr_ratio_strict']    = atr_ratio   > 1.8
        # 温和型条件
        conds['rsi_mod_loose']           = rsi       > 50
        conds['rsi_mod_strict']          = rsi       > 55
        conds['boll_pos_mod_loose']      = bpos      > 0.60
        conds['boll_pos_mod_strict']     = bpos      > 0.65
        conds['macd_hist_turn_loose']    = macd_hist < 0
        conds['macd_hist_turn_strict']   = (macd_hist < 0) & (macd_hist_prev >= 0)
        conds['close_vs_ma5_mod_loose']  = cvma5     > 0.3
        conds['close_vs_ma5_mod_strict'] = cvma5     > 0.5
        conds['ma5_slope_dir_loose']     = ma5_slope > 0
        conds['ma5_slope_dir_strict']    = ma5_slope > 0.05
        conds['obv_trend_loose']         = obv_vals  < obv_ma_vals
        conds['vol_ratio_mod_loose']     = vratio    > 1.1

    return {name: np.asarray(arr, dtype=bool) for name, arr in conds.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 策略 3：单条件预剪枝
# ═══════════════════════════════════════════════════════════════════════════

def _prune_conditions(
    cond_arrays: Dict[str, np.ndarray],
    outcome: np.ndarray,
    min_rate: float = None,   # 不再使用，保留参数签名兼容性
) -> Dict[str, np.ndarray]:
    """
    单条件预剪枝：仅保留触发次数 >= MIN_TRIGGERS 的条件。

    不做命中率筛选，原因：
      数据量大（如 50000 根）时，大数定律使单条件命中率收敛至随机基准
      (~57%)，若设命中率门槛会误杀所有条件。最终筛选由组合的 64/67/71%
      门槛完成。

    Returns:
        保留条件的子集 dict。
    """
    valid: Dict[str, np.ndarray] = {}
    for name, arr in cond_arrays.items():
        indices = np.nonzero(arr)[0]
        if len(indices) >= MIN_TRIGGERS:
            valid[name] = arr
    return valid


# ═══════════════════════════════════════════════════════════════════════════
# 市场状态分类汇总
# ═══════════════════════════════════════════════════════════════════════════

_ALL_STATES = ("多头趋势", "空头趋势", "震荡市")


def _build_market_state_breakdown(
    trigger_indices: np.ndarray,
    outcome: np.ndarray,
    market_state_arr: np.ndarray,
) -> dict:
    """
    对给定触发 bar 集合，按市场状态分组统计命中率。

    Args:
        trigger_indices:  触发 bar 的索引数组
        outcome:          全局 outcome bool 数组
        market_state_arr: 每个 bar 对应的市场状态字符串 ndarray

    Returns:
        {
          "多头趋势": {"triggers": int, "hits": int, "rate": float},
          "空头趋势": {"triggers": int, "hits": int, "rate": float},
          "震荡市":   {"triggers": int, "hits": int, "rate": float},
        }
    """
    breakdown: dict = {}
    states_at_trigger = market_state_arr[trigger_indices]
    outcomes_at_trigger = outcome[trigger_indices]

    for state in _ALL_STATES:
        mask = states_at_trigger == state
        t = int(mask.sum())
        h = int(outcomes_at_trigger[mask].sum()) if t > 0 else 0
        breakdown[state] = {
            "triggers": t,
            "hits":     h,
            "rate":     round(h / t, 6) if t > 0 else 0.0,
        }
    return breakdown


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def analyze(
    df: pd.DataFrame,
    direction: str,
    fee_pct: float = 0.0006,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> List[dict]:
    """
    对给定 K 线数据执行信号组合分析。

    Args:
        df:          已计算好所有技术指标的 DataFrame（含 open/high/low/close 及
                     rsi/k/j/boll_position/volume_ratio/close_vs_ma5/atr 等列）
        direction:   'long' 或 'short'
        fee_pct:     单次往返手续费占名义仓位比例（默认 0.0006 = 0.06%）
                     当前版本用于说明含费门槛来源，门槛已内置为 64/67/71%。
        progress_cb: 进度回调 progress_cb(percent: int, status_text: str)
                     percent 为 0-100 整数

    Returns:
        List[dict]，仅包含命中率 ≥ 候选门槛（做多 64%，做空 52%）的结果，按 hit_rate 降序。
        每个 dict 字段：
          direction              : 'long' / 'short'
          conditions             : List[str]  条件名称列表
          trigger_count          : int        总触发次数
          hit_count              : int        命中次数
          hit_rate               : float      原始命中率（0-1）
          tier                   : str        '候选' / '优质' / '精品'
          low_sample_warn        : bool       True 表示样本偏少（触发次数 8-14）
          market_state_breakdown : dict       按市场状态分组的触发/命中统计
                                              {状态: {triggers, hits, rate}}

    Raises:
        ValueError: direction 不是 'long' / 'short'
    """
    if direction not in ('long', 'short'):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    # ── 1. 提取 NumPy 数组 ──────────────────────────────────────────────────
    high_arr  = df['high'].values.astype(float)
    low_arr   = df['low'].values.astype(float)
    open_arr  = df['open'].values.astype(float)
    close_arr = df['close'].values.astype(float)

    n = len(df)
    adx_arr       = df['adx'].values.astype(float)       if 'adx'       in df.columns else np.full(n, 20.0)
    ma5_slope_arr = df['ma5_slope'].values.astype(float) if 'ma5_slope' in df.columns else np.zeros(n)
    market_state_arr = np.array(
        [detect_state(float(adx_arr[i]), float(ma5_slope_arr[i])) for i in range(n)],
        dtype=object,
    )

    if progress_cb:
        progress_cb(0, "正在预计算止盈止损结果数组...")

    # ── 2. 策略1：预计算 outcome 数组（O(N×240)，执行一次）───────────────────
    tp_pct = LONG_TP1_PCT if direction == 'long' else SHORT_TP1_PCT
    sl_pct = LONG_SL_PCT  if direction == 'long' else SHORT_SL_PCT
    outcome = _precompute_outcomes(high_arr, low_arr, open_arr, close_arr, direction,
                                   tp_pct=tp_pct, sl_pct=sl_pct)

    if progress_cb:
        progress_cb(5, "正在构建条件数组...")

    # ── 3. 策略2：构建 30 个条件 bool 数组 ──────────────────────────────────
    all_conds = _build_condition_arrays(df, direction)

    if progress_cb:
        progress_cb(8, "正在进行单条件预剪枝...")

    # ── 4. 策略3：单条件预剪枝 ──────────────────────────────────────────────
    valid_conds  = _prune_conditions(all_conds, outcome, PRUNE_MIN_RATE)
    cond_names   = list(valid_conds.keys())
    cond_arr_lst = [valid_conds[name] for name in cond_names]
    nc = len(cond_names)

    # ── 5. 统计总组合数（用于进度分母）──────────────────────────────────────
    max_r = min(6, nc + 1)
    total_combos = sum(math.comb(nc, r) for r in range(2, max_r))

    if progress_cb:
        dir_label = "做多" if direction == 'long' else "做空"
        progress_cb(
            10,
            f"剪枝后保留 {nc} 个条件，{dir_label}方向共 {total_combos} 种组合..."
        )

    # ── 6. 枚举组合 + 向量化评估 ─────────────────────────────────────────────
    results: List[dict] = []
    processed  = 0
    last_pct   = 10
    report_gap = max(1, total_combos // 90)   # 保证约 90 次进度更新

    for r in range(2, max_r):
        for combo_indices in itertools.combinations(range(nc), r):
            # 向量化 AND（策略2）
            trigger_mask = cond_arr_lst[combo_indices[0]].copy()
            for idx in combo_indices[1:]:
                trigger_mask &= cond_arr_lst[idx]

            trigger_count = int(trigger_mask.sum())
            processed += 1

            if trigger_count < MIN_TRIGGERS:
                if processed % report_gap == 0 and progress_cb:
                    pct = 10 + int(processed / total_combos * 88)
                    if pct > last_pct:
                        last_pct = pct
                        progress_cb(pct, f"正在计算第 {processed} / {total_combos} 个组合...")
                continue

            indices   = np.nonzero(trigger_mask)[0]
            hit_count = int(outcome[indices].sum())
            hit_rate  = hit_count / trigger_count

            t_prem = TIER_PREMIUM   if direction == 'long' else SHORT_TIER_PREMIUM
            t_qual = TIER_QUALITY   if direction == 'long' else SHORT_TIER_QUALITY
            t_cand = TIER_CANDIDATE if direction == 'long' else SHORT_TIER_CANDIDATE

            if hit_rate >= t_prem:
                tier = '精品'
            elif hit_rate >= t_qual:
                tier = '优质'
            elif hit_rate >= t_cand:
                tier = '候选'
            else:
                tier = ''

            if tier:
                results.append({
                    'direction':              direction,
                    'conditions':             [cond_names[i] for i in combo_indices],
                    'trigger_count':          trigger_count,
                    'hit_count':              hit_count,
                    'hit_rate':               hit_rate,
                    'tier':                   tier,
                    'low_sample_warn':        trigger_count <= WARN_TRIGGERS_MAX,
                    'market_state_breakdown': _build_market_state_breakdown(
                        indices, outcome, market_state_arr
                    ),
                })

            if processed % report_gap == 0 and progress_cb:
                pct = 10 + int(processed / total_combos * 88)
                if pct > last_pct:
                    last_pct = pct
                    progress_cb(pct, f"正在计算第 {processed} / {total_combos} 个组合...")

    results.sort(key=lambda x: x['hit_rate'], reverse=True)

    if progress_cb:
        tc = {t: sum(1 for r in results if r['tier'] == t)
              for t in ('候选', '优质', '精品')}
        progress_cb(
            100,
            f"分析完成：候选 {tc['候选']} / 优质 {tc['优质']} / 精品 {tc['精品']}"
        )

    return results
