"""
core/signal_store.py
信号分析结果持久化模块

负责读写 data/signal_analysis_state.json，合并多轮回测结果，
并维护每个组合跨轮次的稳定性评分。

数据结构（JSON 顶层）：
  {
    "version": "3.0",
    "last_updated": "2026-02-18T...",
    "rounds": [
        {
            "round_id": 1,
            "timestamp": "...",
            "direction": "long",
            "bar_count": 5000,
            "results": [...]   # 本轮候选列表（来自 signal_analyzer.analyze）
        },
        ...
    ],
    "cumulative": {
        "<combo_key>": {
            "direction":        "long",
            "conditions":       [...],
            "appear_rounds":    3,
            "avg_rate":         0.682,
            "rate_std":         0.031,
            "stability_score":  0.712,
            "综合评分":          85.4,
            "history":          [{"round_id": 1, "hit_rate": 0.70, "trigger_count": 12}, ...]
        },
        ...
    }
  }

combo_key 格式："{direction}|{条件1}+{条件2}+..."（条件名按字典序排序，保证唯一性）
"""
from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 高速 JSON 序列化（提速 5-10 倍）
try:
    import orjson
    _USE_ORJSON = True
except ImportError:
    _USE_ORJSON = False

# ── 文件路径 ─────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(os.path.dirname(_THIS_DIR), 'data')
STATE_FILE = os.path.join(_DATA_DIR, 'signal_analysis_state.json')
CUMULATIVE_CACHE_FILE = os.path.join(_DATA_DIR, 'signal_cumulative_cache.json')

_SCHEMA_VERSION = "3.0"

# ── 内存缓存（避免每次调用重新解析大文件） ──────────────────────────────────
_cache: Dict[str, dict] = {}
_cache_mtime: Dict[str, float] = {}

# ── 轻量级 cumulative 缓存（仅缓存 cumulative 部分，~2-5MB） ──────────────
_cumul_cache: Dict[str, dict] = {}
_cumul_cache_mtime: Dict[str, float] = {}

# ── rounds 内存缓存（避免每次重新解析大文件的 rounds 字段） ─────────────────
_rounds_cache: Dict[str, list] = {}
_rounds_cache_mtime: Dict[str, float] = {}

def _get_files(pool_id: str) -> Tuple[str, str]:
    if pool_id == "pool2":
        return (os.path.join(_DATA_DIR, "signal_analysis_state_pool2.json"),
                os.path.join(_DATA_DIR, "signal_cumulative_cache_pool2.json"))
    return (STATE_FILE, CUMULATIVE_CACHE_FILE)


# ── 写入锁（防止同进程并发写） ────────────────────────────────────────────
import threading as _threading
_write_lock = _threading.Lock()

def _win_long_path(path: str) -> str:
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path)
    abs_path = os.path.normpath(abs_path)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        # UNC path
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def _safe_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except OSError as e:
        if os.name == "nt" and e.errno == 22:
            return os.path.exists(_win_long_path(path))
        raise


def _safe_makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if os.name == "nt" and e.errno == 22:
            os.makedirs(_win_long_path(path), exist_ok=True)
        else:
            raise


def _open_file(path: str, mode: str, encoding: Optional[str] = None):
    try:
        if "b" in mode:
            return open(path, mode)
        return open(path, mode, encoding=encoding)
    except OSError as e:
        if os.name == "nt" and e.errno == 22:
            long_path = _win_long_path(path)
            if "b" in mode:
                return open(long_path, mode)
            return open(long_path, mode, encoding=encoding)
        raise


def _safe_getmtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError as e:
        if os.name == "nt" and e.errno == 22:
            return os.path.getmtime(_win_long_path(path))
        raise


# ═══════════════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════════════

def _combo_key(direction: str, conditions: List[str]) -> str:
    """生成组合的唯一键（条件排序后拼接）。"""
    return f"{direction}|{'+'.join(sorted(conditions))}"


def _load_raw(pool_id: str = "pool1") -> dict:
    """加载 JSON 文件，不存在则返回空骨架。文件未变化时直接返回内存缓存。"""
    global _cache, _cache_mtime
    state_file, _ = _get_files(pool_id)
    if _safe_exists(state_file):
        try:
            mtime = _safe_getmtime(state_file)
            if pool_id in _cache and mtime == _cache_mtime.get(pool_id):
                return _cache[pool_id]
            with _open_file(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                _cache[pool_id] = data
                _cache_mtime[pool_id] = mtime
                return _cache[pool_id]
        except (json.JSONDecodeError, OSError):
            pass
    return _empty_state()


def _empty_state() -> dict:
    return {
        "version":      _SCHEMA_VERSION,
        "last_updated": "",
        "rounds":       [],
        "cumulative":   {},
    }


def _save_raw(data: dict, pool_id: str = "pool1") -> None:
    """直接写入目标文件（使用 orjson 高速序列化，移除 fsync 提速）。"""
    global _cache, _cache_mtime, _rounds_cache, _rounds_cache_mtime
    _safe_makedirs(_DATA_DIR)
    state_file, _ = _get_files(pool_id)
    with _write_lock:
        if _USE_ORJSON:
            with _open_file(state_file, 'wb') as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        else:
            with _open_file(state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        _cache[pool_id] = data
        _cache_mtime[pool_id] = _safe_getmtime(state_file)
        _rounds_cache[pool_id] = data.get('rounds', [])
        _rounds_cache_mtime[pool_id] = _cache_mtime[pool_id]
    _save_cumulative_cache(data, pool_id)


def _save_cumulative_cache(data: dict, pool_id: str = "pool1") -> None:
    """将 cumulative 部分单独写入轻量级缓存文件（使用 orjson 高速序列化）。"""
    global _cumul_cache, _cumul_cache_mtime
    _safe_makedirs(_DATA_DIR)
    _, cumul_file = _get_files(pool_id)
    cache_data = {
        "version":      data.get("version", _SCHEMA_VERSION),
        "last_updated": data.get("last_updated", ""),
        "cumulative":   data.get("cumulative", {}),
    }
    with _write_lock:
        if _USE_ORJSON:
            with _open_file(cumul_file, 'wb') as f:
                f.write(orjson.dumps(cache_data, option=orjson.OPT_INDENT_2))
        else:
            with _open_file(cumul_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
        _cumul_cache[pool_id] = cache_data["cumulative"]
        _cumul_cache_mtime[pool_id] = _safe_getmtime(cumul_file)


def _load_cumulative_fast(pool_id: str = "pool1") -> dict:
    global _cumul_cache, _cumul_cache_mtime

    _, cumul_file = _get_files(pool_id)

    # 1. 内存命中
    if pool_id in _cumul_cache and _safe_exists(cumul_file):
        try:
            mtime = _safe_getmtime(cumul_file)
            if mtime == _cumul_cache_mtime.get(pool_id):
                return _cumul_cache[pool_id]
        except OSError:
            pass

    # 2. 缓存文件存在，读取它
    if _safe_exists(cumul_file):
        try:
            mtime = _safe_getmtime(cumul_file)
            with _open_file(cumul_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            if isinstance(cache_data, dict):
                _cumul_cache[pool_id] = cache_data.get('cumulative', {})
                _cumul_cache_mtime[pool_id] = mtime
                return _cumul_cache[pool_id]
        except (json.JSONDecodeError, OSError):
            pass

    # 3. 缓存文件不存在，回退到完整文件并生成缓存
    state_file, _ = _get_files(pool_id)
    if _safe_exists(state_file):
        data = _load_raw(pool_id)
        try:
            _save_cumulative_cache(data, pool_id)
        except Exception:
            pass
        return data.get('cumulative', {})

    # 4. 无数据
    return {}


def _load_rounds_fast(pool_id: str = "pool1") -> list:
    global _rounds_cache, _rounds_cache_mtime

    state_file, _ = _get_files(pool_id)
    if pool_id in _rounds_cache and _safe_exists(state_file):
        try:
            mtime = _safe_getmtime(state_file)
            if mtime == _rounds_cache_mtime.get(pool_id):
                return _rounds_cache[pool_id]
        except OSError:
            pass

    data = _load_raw(pool_id)
    _rounds_cache[pool_id] = data.get('rounds', [])
    _rounds_cache_mtime[pool_id] = _cache_mtime.get(pool_id, 0.0)
    return _rounds_cache[pool_id]


def invalidate_cache() -> None:
    """强制清除所有内存缓存，下次加载将重新读取磁盘文件。"""
    global _cache, _cache_mtime, _cumul_cache, _cumul_cache_mtime, _rounds_cache, _rounds_cache_mtime
    _cache.clear()
    _cache_mtime.clear()
    _cumul_cache.clear()
    _cumul_cache_mtime.clear()
    _rounds_cache.clear()
    _rounds_cache_mtime.clear()


def _next_round_id(data: dict) -> int:
    rounds = data.get('rounds', [])
    if not rounds:
        return 1
    return max(r.get('round_id', 0) for r in rounds) + 1


# ═══════════════════════════════════════════════════════════════════════════
# 累计指标计算
# ═══════════════════════════════════════════════════════════════════════════

# Pool1 做多：含费后 TP净0.54%，SL净0.86%
_LONG_TP_PCT  = 0.0054
_LONG_SL_PCT  = 0.0086

# Pool1 做空：含费后 TP净0.74%，SL净0.66%
_SHORT_TP_PCT = 0.0074
_SHORT_SL_PCT = 0.0066

# Pool2 双向对称：含费后 TP净0.94%，SL净0.86%
_POOL2_TP_PCT = 0.0094
_POOL2_SL_PCT = 0.0086

_ALL_STATES = ("多头趋势", "空头趋势", "震荡市")


def _get_tp_sl(direction: str, pool_id: str = 'pool1'):
    """根据方向和策略池返回含费后净 TP/SL。"""
    if pool_id == 'pool2':
        return _POOL2_TP_PCT, _POOL2_SL_PCT
    if direction == 'long':
        return _LONG_TP_PCT, _LONG_SL_PCT
    return _SHORT_TP_PCT, _SHORT_SL_PCT


def _estimated_pnl_pct(overall_rate: float, total_triggers: int,
                        direction: str = 'long', pool_id: str = 'pool1') -> float:
    """
    估算累计盈亏百分比。

    Pool1 做多：每笔期望 = overall_rate × 0.54% - (1 - overall_rate) × 0.86%
    Pool1 做空：每笔期望 = overall_rate × 0.74% - (1 - overall_rate) × 0.66%
    Pool2 双向：每笔期望 = overall_rate × 0.94% - (1 - overall_rate) × 0.86%
    估算总盈亏 = 每笔期望 × total_triggers
    """
    tp_pct, sl_pct = _get_tp_sl(direction, pool_id)
    per_trade = overall_rate * tp_pct - (1.0 - overall_rate) * sl_pct
    return round(per_trade * total_triggers * 100, 4)   # 单位：%


def _ev_per_trigger_pct(overall_rate: float, direction: str = 'long',
                         pool_id: str = 'pool1') -> float:
    """单次触发期望盈亏（百分比，未考虑杠杆）。"""
    tp_pct, sl_pct = _get_tp_sl(direction, pool_id)
    per_trade = overall_rate * tp_pct - (1.0 - overall_rate) * sl_pct
    return round(per_trade * 100, 4)   # 单位：%


def _merge_market_state_breakdown(history: List[dict]) -> dict:
    """
    聚合多轮历史记录中的 market_state_breakdown，得到累计汇总。

    history 元素可含 'market_state_breakdown' 字段（来自 signal_analyzer）。
    """
    totals: dict = {s: {"total_triggers": 0, "total_hits": 0} for s in _ALL_STATES}

    for h in history:
        bkd = h.get('market_state_breakdown') or {}
        for state in _ALL_STATES:
            sb = bkd.get(state) or {}
            totals[state]["total_triggers"] += sb.get("triggers", 0)
            totals[state]["total_hits"]     += sb.get("hits", 0)

    result: dict = {}
    for state in _ALL_STATES:
        t = totals[state]["total_triggers"]
        h = totals[state]["total_hits"]
        result[state] = {
            "total_triggers": t,
            "total_hits":     h,
            "avg_rate":       round(h / t, 6) if t > 0 else 0.0,
        }
    return result


def _get_state_rate(breakdown: Optional[dict], state: str) -> Tuple[float, int]:
    """返回指定市场状态的命中率和触发次数，无数据则返回 (0.0, 0)。"""
    if not breakdown:
        return 0.0, 0
    info = breakdown.get(state) or {}
    # 字段名为 avg_rate，不是 rate
    return float(info.get("avg_rate", 0.0)), int(info.get("total_triggers", 0))


def _default_live_tracking() -> dict:
    return {
        "total":        0,
        "hits":         0,
        "miss":         0,
        "live_rate":    0.0,
        "last_updated": "",
        "streak_loss":  0,
        "alert":        False,
    }


def _compute_cumulative_metrics(history: List[dict], direction: str = 'long',
                                 pool_id: str = 'pool1') -> dict:
    """
    根据历次命中率记录计算累计指标。

    history 元素字段：round_id, hit_rate, trigger_count, hit_count,
                      market_state_breakdown (可选), hold_bars (可选)
    direction: 'long' 或 'short'，影响评分基准线和 P&L 常量

    返回字段：
      appear_rounds         : int    出现轮次数
      avg_rate              : float  平均命中率
      rate_std              : float  命中率标准差
      stability_score       : float  = avg_rate × (1 - rate_std) × ln(appear_rounds + 1)
      综合评分               : float  加权综合评分（0-100）
      total_triggers        : int    累计触发次数
      total_hits            : int    累计命中次数
      overall_rate          : float  总体命中率（total_hits / total_triggers）
      estimated_pnl_pct     : float  估算累计盈亏百分比
      market_state_breakdown: dict   按市场状态聚合的触发/命中统计
      avg_hold_bars         : int    自然结束交易的平均持仓根数（排除超时）
      hold_bars_sample_count: int    参与统计的样本数
    """
    rates = [h['hit_rate'] for h in history]
    n = len(rates)

    total_triggers = sum(h['trigger_count'] for h in history)
    total_hits     = sum(h.get('hit_count', 0) for h in history)
    overall_rate   = total_hits / total_triggers if total_triggers > 0 else 0.0

    # 聚合 hold_bars：收集所有轮次中自然结束的持仓记录（排除 hold=-1 超时）
    all_hold_bars: List[int] = []
    for h in history:
        all_hold_bars.extend(h.get('hold_bars', []))
    valid_holds = [hb for hb in all_hold_bars if hb >= 0]
    avg_hold_bars = int(round(statistics.median(valid_holds))) if valid_holds else 0
    hold_bars_sample_count = len(valid_holds)

    if n == 0:
        return {
            'appear_rounds':          0,
            'avg_rate':               0.0,
            'rate_std':               0.0,
            'stability_score':        0.0,
            '综合评分':                0.0,
            'total_triggers':         0,
            'total_hits':             0,
            'overall_rate':           0.0,
            'estimated_pnl_pct':      0.0,
            'market_state_breakdown': _merge_market_state_breakdown([]),
            'avg_hold_bars':          avg_hold_bars,
            'hold_bars_sample_count': hold_bars_sample_count,
        }

    avg_rate = sum(rates) / n
    variance = sum((r - avg_rate) ** 2 for r in rates) / n
    rate_std = math.sqrt(variance)

    stability_score = avg_rate * (1.0 - rate_std) * math.log(n + 1)

    # 综合评分（0-100）：稳定性占 40%，命中率占 30%，轮次占 30%
    # 稳定性权重最高，防止"高命中但仅 1 轮"的过拟合组合排到前面
    # 做多基准 0.61 / 满分 0.75；做空基准 0.47 / 满分 0.65
    if direction == 'short':
        rate_baseline, rate_max = 0.47, 0.65
    else:
        rate_baseline, rate_max = 0.61, 0.75
    rate_score  = min(100.0, max(0.0, (avg_rate - rate_baseline) / (rate_max - rate_baseline) * 100.0))
    stab_score  = min(100.0, max(0.0, (1.0 - rate_std / 0.08) * 100.0))
    # 轮次衰减折扣：n<5 时降低稳定性分，避免小样本虚高
    round_factor = min(1.0, n / 5.0)
    stab_score *= round_factor
    round_score = min(100.0, math.log(n + 1) / math.log(21) * 100.0)  # n=20 时满分
    composite   = rate_score * 0.30 + stab_score * 0.40 + round_score * 0.30

    return {
        'appear_rounds':          n,
        'avg_rate':               round(avg_rate, 6),
        'rate_std':               round(rate_std, 6),
        'stability_score':        round(stability_score, 6),
        '综合评分':                round(composite, 2),
        'total_triggers':         total_triggers,
        'total_hits':             total_hits,
        'overall_rate':           round(overall_rate, 6),
        'estimated_pnl_pct':      _estimated_pnl_pct(overall_rate, total_triggers, direction, pool_id),
        'ev_per_trigger_pct':     _ev_per_trigger_pct(overall_rate, direction, pool_id),
        'market_state_breakdown': _merge_market_state_breakdown(history),
        'avg_hold_bars':          avg_hold_bars,
        'hold_bars_sample_count': hold_bars_sample_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════════════════

def _merge_round_into_data(
    data: dict,
    new_results: List[dict],
    direction: str,
    bar_count: int,
    timestamp: str,
    round_id: int,
) -> dict:
    """将一轮结果合并进 data（不落盘），返回更新的 cumulative 条目。"""
    # 记录本轮原始结果
    data['rounds'].append({
        'round_id':  round_id,
        'timestamp': timestamp,
        'direction': direction,
        'bar_count': bar_count,
        'results':   new_results,
    })

    cumulative: Dict[str, dict] = data.setdefault('cumulative', {})
    updated_keys: dict = {}

    for item in new_results:
        key = _combo_key(item['direction'], item['conditions'])
        item_pool_id = item.get('pool_id', 'pool1')

        if key not in cumulative:
            cumulative[key] = {
                'direction':    item['direction'],
                'pool_id':      item_pool_id,
                'conditions':   sorted(item['conditions']),
                'history':      [],
                'live_tracking': _default_live_tracking(),
            }
        else:
            if 'live_tracking' not in cumulative[key]:
                cumulative[key]['live_tracking'] = _default_live_tracking()
            # 确保 pool_id 始终更新（兼容旧数据）
            cumulative[key]['pool_id'] = item_pool_id

        entry = cumulative[key]
        entry['history'].append({
            'round_id':               round_id,
            'hit_rate':               item['hit_rate'],
            'trigger_count':          item['trigger_count'],
            'hit_count':              item['hit_count'],
            'tier':                   item['tier'],
            'market_state_breakdown': item.get('market_state_breakdown'),
            'hold_bars':              item.get('hold_bars', []),
        })

        metrics = _compute_cumulative_metrics(entry['history'], direction=item['direction'],
                                               pool_id=item_pool_id)
        entry.update(metrics)
        # 条件数量惩罚：超过 3 条每多 1 条扣 3 分（防过拟合）
        n_conds = len(entry.get('conditions', []))
        if n_conds > 3:
            penalty = (n_conds - 3) * 3.0
            entry['综合评分'] = round(max(0.0, entry.get('综合评分', 0.0) - penalty), 2)
        updated_keys[key] = entry

    return updated_keys


def merge_round(
    new_results: List[dict],
    direction: str,
    bar_count: int = 5000,
    timestamp: Optional[str] = None,
    pool_id: str = "pool1",
) -> dict:
    data = _load_raw(pool_id)

    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec='seconds')

    round_id = _next_round_id(data)
    updated_keys = _merge_round_into_data(
        data, new_results, direction, bar_count, timestamp, round_id
    )

    data['version']      = _SCHEMA_VERSION
    data['last_updated'] = timestamp
    _save_raw(data, pool_id)

    return updated_keys


def merge_results(
    results: List[dict],
    pool_id: str = "pool1",
    bar_count: int = 5000,
    timestamp: Optional[str] = None,
) -> dict:
    """合并 signal_analyzer.analyze 返回的结果，支持 pool_id。"""
    data = _load_raw(pool_id)
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec='seconds')

    updated_keys: dict = {}
    round_id = _next_round_id(data)
    
    long_results = [r for r in results if r.get('direction') == 'long']
    short_results = [r for r in results if r.get('direction') == 'short']

    if long_results:
        updated_keys.update(
            _merge_round_into_data(data, long_results, 'long', bar_count, timestamp, round_id)
        )
        round_id += 1
    if short_results:
        updated_keys.update(
            _merge_round_into_data(data, short_results, 'short', bar_count, timestamp, round_id)
        )

    data['version']      = _SCHEMA_VERSION
    data['last_updated'] = timestamp
    _save_raw(data, pool_id)

    return updated_keys

def merge_rounds(
    long_results: Optional[List[dict]],
    short_results: Optional[List[dict]],
    bar_count: int = 5000,
    timestamp: Optional[str] = None,
    pool_id: str = "pool1",
) -> dict:
    """旧的接口，通过 pool_id 支持多池"""
    data = _load_raw(pool_id)
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec='seconds')

    updated_keys: dict = {}
    round_id = _next_round_id(data)

    if long_results:
        updated_keys.update(
            _merge_round_into_data(data, long_results, 'long', bar_count, timestamp, round_id)
        )
        round_id += 1
    if short_results:
        updated_keys.update(
            _merge_round_into_data(data, short_results, 'short', bar_count, timestamp, round_id)
        )

    data['version']      = _SCHEMA_VERSION
    data['last_updated'] = timestamp
    _save_raw(data, pool_id)

    return updated_keys


def _tier_from_rate(overall_rate: float, direction: str) -> str:
    """根据综合命中率和方向返回层级，与 signal_analyzer 门槛一致。"""
    if direction == 'long':
        if overall_rate >= 0.71: return '精品'
        if overall_rate >= 0.67: return '优质'
        if overall_rate >= 0.64: return '候选'
    else:
        if overall_rate >= 0.59: return '精品'
        if overall_rate >= 0.55: return '优质'
        if overall_rate >= 0.52: return '候选'
    return ''


def _cond_to_family(cond: str) -> str:
    """提取条件的指标族名称，去掉 _loose/_strict 后缀。"""
    for suffix in ('_loose', '_strict'):
        if cond.endswith(suffix):
            return cond[:-len(suffix)]
    return cond


def _cond_is_loose(cond: str) -> bool:
    """判断条件是否为宽松版。"""
    return cond.endswith('_loose')


def _cond_covers(cond_a: str, cond_b: str) -> bool:
    """
    判断条件 A 是否覆盖条件 B（即 A 触发范围 >= B）。
    覆盖关系：
      - 完全相同：A == B
      - 同指标族，A 为宽松版，B 为严格版：宽松覆盖严格
    """
    if cond_a == cond_b:
        return True
    family_a = _cond_to_family(cond_a)
    family_b = _cond_to_family(cond_b)
    if family_a == family_b and _cond_is_loose(cond_a) and not _cond_is_loose(cond_b):
        return True
    return False


def _combo_a_covers_b(conds_a: List[str], conds_b: List[str]) -> bool:
    """
    判断策略 A（条件列表）是否覆盖策略 B。
    覆盖定义：A 中每一个条件，在 B 中都能找到一个被它覆盖的条件。
    即：B 触发 → A 必然触发（A 的条件是 B 条件的宽松子集）。
    注意：A 的条件数量必须 <= B，否则 A 比 B 更严格，不构成覆盖。
    """
    if len(conds_a) > len(conds_b):
        return False
    for ca in conds_a:
        if not any(_cond_covers(ca, cb) for cb in conds_b):
            return False
    return True


def _simplify_conditions(conditions: List[str]) -> List[str]:
    """
    规范化条件列表：若同族同时含宽松+严格，仅保留严格版本。
    ['ma5_slope_dir_loose', 'ma5_slope_dir_strict'] → ['ma5_slope_dir_strict']
    """
    family_to_versions: Dict[str, List[str]] = {}
    no_family: List[str] = []
    for c in conditions:
        fam = _cond_to_family(c)
        if fam != c:
            family_to_versions.setdefault(fam, []).append(c)
        else:
            no_family.append(c)
    result: List[str] = []
    for fam, versions in family_to_versions.items():
        if len(versions) == 1:
            result.append(versions[0])
        else:
            # 同族多版本：只保留严格版
            strict = [v for v in versions if v.endswith('_strict')]
            result.extend(strict if strict else versions)
    return result + no_family


def _normalize_all_entries(entries: Dict[str, dict]) -> Dict[str, dict]:
    """对全部条目做条件规范化，重复 key 时合并（保留触发数更多的）。"""
    normalized: Dict[str, dict] = {}
    for key, entry in entries.items():
        conds = entry.get('conditions', [])
        new_conds = _simplify_conditions(conds)
        new_key = entry.get('direction', 'long') + ':' + ','.join(sorted(new_conds))
        if new_key in normalized:
            # key 碰撞：保留触发数更多的
            if entry.get('total_triggers', 0) >= normalized[new_key].get('total_triggers', 0):
                normalized[new_key] = {**entry, 'conditions': new_conds}
        else:
            normalized[new_key] = {**entry, 'conditions': new_conds}
    return normalized


def _prune_redundant_subsets(entries: Dict[str, dict]) -> Dict[str, dict]:
    """
    去重：若 A 能覆盖 B，且 A 命中率 >= B，则删除 B。

    分组仅按 direction，允许跨命中率比较（解决“新简策略替换旧复杂策略”）。
    """
    from collections import defaultdict

    # 按方向分组，同方向内比较冗余
    groups: dict = defaultdict(list)
    for key, entry in entries.items():
        dir_ = entry.get('direction', 'long')
        groups[dir_].append(key)

    to_remove: set = set()

    for group_keys in groups.values():
        if len(group_keys) < 2:
            continue  # 单条目组，无需比较

        # 按条件数量升序 + 命中率降序（条件少&表现好优先作为 A）
        group_keys.sort(
            key=lambda k: (
                len(entries[k].get('conditions', [])),
                -(entries[k].get('overall_rate', 0.0) or 0.0),
            )
        )

        for i, key_b in enumerate(group_keys):
            if key_b in to_remove:
                continue
            entry_b    = entries[key_b]
            conds_b    = entry_b.get('conditions', [])
            triggers_b = entry_b.get('total_triggers', 0) or 0
            rate_b = entry_b.get('overall_rate', 0.0) or 0.0
            rounds_b = max(entry_b.get('appear_rounds', 1) or 1, 1)
            avg_trig_b = triggers_b / rounds_b

            for key_a in group_keys[:i]:  # 只与条件更少的比
                if key_a in to_remove:
                    continue
                entry_a    = entries[key_a]
                conds_a    = entry_a.get('conditions', [])
                triggers_a = entry_a.get('total_triggers', 0) or 0
                rate_a = entry_a.get('overall_rate', 0.0) or 0.0
                rounds_a = max(entry_a.get('appear_rounds', 1) or 1, 1)
                avg_trig_a = triggers_a / rounds_a

                # A 命中率更低则不可能替代 B
                if rate_a + 1e-12 < rate_b:
                    continue

                # 若 A 每轮平均触发显著低于 B，跳过（避免新策略因轮次少被误杀）
                if avg_trig_a < avg_trig_b * 0.8:
                    continue

                # A 严格覆盖 B（条件数更少且逻辑覆盖）
                if len(conds_a) < len(conds_b) and _combo_a_covers_b(conds_a, conds_b):
                    to_remove.add(key_b)
                    break

                # A 与 B 条件数相同但 A 全为宽松版（A 覆盖 B）
                if len(conds_a) == len(conds_b) and conds_a != conds_b:
                    if _combo_a_covers_b(conds_a, conds_b):
                        to_remove.add(key_b)
                        break

    return {k: v for k, v in entries.items() if k not in to_remove}


def _ensure_pruned_cache_fresh(pool_id: str = "pool1") -> None:
    state_file, cumul_file = _get_files(pool_id)
    try:
        cache_mtime = _safe_getmtime(cumul_file) if _safe_exists(cumul_file) else 0.0
    except OSError:
        cache_mtime = 0.0
    try:
        state_mtime = _safe_getmtime(state_file) if _safe_exists(state_file) else 0.0
    except OSError:
        state_mtime = 0.0
    if cache_mtime < state_mtime:
        rebuild_pruned_cache(pool_id)


def _anchor_dedup(entries: Dict[str, dict], max_per_anchor: int = 4) -> Dict[str, dict]:
    """
    锚点分组去重：相同方向且共享同一"锚定条件"（该方向下出现频次最高的条件）
    的策略，按综合评分保留 Top max_per_anchor 条。

    互补保护：若两条策略的主导市场状态不同（如一条以空头市场为主、
    另一条以震荡市为主），则认为它们互补，不计入同一限额。
    """
    from collections import defaultdict

    def _main_state(v: dict) -> str:
        """返回该策略触发最多的市场状态，如无则返回空字符串。"""
        bd = v.get('market_state_breakdown') or {}
        best, best_t = '', 0
        for state, info in bd.items():
            t = info.get('total_triggers', 0) if isinstance(info, dict) else 0
            if t > best_t:
                best_t, best = t, state
        return best

    # 统计每个方向下条件出现频次，用于决定“最高频锚点”
    cond_freq: dict = defaultdict(lambda: defaultdict(int))
    for entry in entries.values():
        direction = entry.get('direction', 'long')
        for cond in entry.get('conditions', []) or []:
            cond_freq[direction][cond] += 1

    def _choose_anchor(conds: list, direction: str) -> str:
        """选择该方向下出现频次最高的条件作为锚点（频次并列取更短的条件）。"""
        if not conds:
            return '__none__'
        freq_map = cond_freq.get(direction, {})
        # 频次优先，其次条件长度（更短更通用），最后按字符串稳定排序
        return sorted(
            conds,
            key=lambda c: (-freq_map.get(c, 0), len(c), c)
        )[0]

    # 按 (方向, 锚点条件) 分组
    groups: dict = defaultdict(list)
    for key, entry in entries.items():
        conds = entry.get('conditions', []) or []
        direction = entry.get('direction', 'long')
        anchor = _choose_anchor(conds, direction)
        groups[(direction, anchor)].append((key, entry))

    result = {}
    for group in groups.values():
        if len(group) <= max_per_anchor:
            # 数量未超限，全部保留
            for key, entry in group:
                result[key] = entry
            continue

        # 按综合评分降序排列
        group.sort(key=lambda x: x[1].get('综合评分', 0.0), reverse=True)

        kept = []
        kept_states: list = []  # 已保留条目的主导状态列表

        for key, entry in group:
            ms = _main_state(entry)
            # 互补保护：主导状态与已保留的全部不同，直接保留（不占名额）
            if ms and ms not in kept_states:
                kept.append((key, entry))
                kept_states.append(ms)
                continue
            # 普通情况：按 Top-K 限额保留
            if len(kept) < max_per_anchor:
                kept.append((key, entry))
                if ms and ms not in kept_states:
                    kept_states.append(ms)

        for key, entry in kept:
            result[key] = entry

    return result


def rebuild_pruned_cache(pool_id: str = "pool1") -> None:
    data = _load_raw(pool_id)
    cumulative = data.get("cumulative", {})

    CANDIDATE_MAX_STD    = 0.05
    CANDIDATE_MIN_ROUNDS = 5

    def _keep_entry(v: dict) -> bool:
        rate = v.get('overall_rate', 0.0)
        direction = v.get('direction', 'long')

        # 负期望策略直接排除（先于层级判断）
        tp_pct, sl_pct = _get_tp_sl(direction, pool_id)
        ev = rate * tp_pct - (1.0 - rate) * sl_pct
        if ev <= 0:
            return False

        # pool2 两方向对称门槛：精品≥59%，优质≥55%，候选≥52%
        if pool_id == 'pool2':
            if rate >= 0.59: tier = '精品'
            elif rate >= 0.55: tier = '优质'
            elif rate >= 0.52: tier = '候选'
            else: tier = ''
        else:
            tier = _tier_from_rate(rate, direction)
        if tier in ('精品', '优质'):
            return True
        if tier == '候选':
            return (v.get('rate_std', 1.0) <= CANDIDATE_MAX_STD
                    and v.get('appear_rounds', 0) >= CANDIDATE_MIN_ROUNDS)
        return False

    filtered = {k: v for k, v in cumulative.items() if _keep_entry(v)}
    normalized = _normalize_all_entries(filtered)
    pruned = _prune_redundant_subsets(normalized)
    # 锚点分组去重：每个同方向同核心条件的组最多保留 4 条（互补状态不受限）
    pruned = _anchor_dedup(pruned, max_per_anchor=4)

    # 用正确的 pool_id TP/SL 参数重新计算估算盈亏字段（修正旧数据用错参数的问题）
    for entry in pruned.values():
        rate = entry.get('overall_rate', 0.0)
        direction = entry.get('direction', 'long')
        total_triggers = entry.get('total_triggers', 0)
        entry['estimated_pnl_pct'] = _estimated_pnl_pct(rate, total_triggers, direction, pool_id)
        entry['ev_per_trigger_pct'] = _ev_per_trigger_pct(rate, direction, pool_id)

    _save_cumulative_cache({**data, "cumulative": pruned}, pool_id)


def get_cumulative(direction: Optional[str] = None, pool_id: str = "pool1") -> Dict[str, dict]:
    _ensure_pruned_cache_fresh(pool_id)
    cumulative = _load_cumulative_fast(pool_id)
    
    if direction is not None:
        cumulative = {k: v for k, v in cumulative.items() if v.get('direction') == direction}
        
    return cumulative


# ═══════════════════════════════════════════════════════════════════════════
# 智能合并 & 多样性筛选辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _get_condition_family(cond: str) -> str:
    """提取条件的指标族，如 'boll_pos_loose' → 'boll_pos'"""
    for suffix in ('_loose', '_strict'):
        if cond.endswith(suffix):
            return cond[:-len(suffix)]
    return cond


def _get_family_set(conditions: List[str]) -> frozenset:
    """获取条件组合涉及的指标族集合"""
    return frozenset(_get_condition_family(c) for c in conditions)


def _family_overlap_ratio(families_a: frozenset, families_b: frozenset) -> float:
    """计算两个指标族集合的重叠度（Jaccard）"""
    if not families_a or not families_b:
        return 0.0
    intersection = len(families_a & families_b)
    union = len(families_a | families_b)
    return intersection / union if union > 0 else 0.0


def _is_loose_version(cond: str) -> bool:
    """判断是否为宽松版条件"""
    return cond.endswith('_loose')


def _merge_similar_combos(combos: List[dict]) -> List[dict]:
    """
    合并相似组合：
    1. 按指标族集合分组
    2. 同一组内，选取评分最高 + 宽松版条件多的作为代表
    3. 返回合并后的代表组合列表
    """
    from collections import defaultdict

    family_groups: Dict[frozenset, List[dict]] = defaultdict(list)
    for c in combos:
        families = _get_family_set(c.get("conditions", []))
        family_groups[families].append(c)

    merged: List[dict] = []
    for families, group in family_groups.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            # 多个组合共用相同指标族，选择最优代表
            # 优先选：综合评分最高 + 使用宽松版条件多的
            def score_combo(c):
                conds = c.get("conditions", [])
                loose_count = sum(1 for cd in conds if _is_loose_version(cd))
                return c.get("综合评分", 0.0) + loose_count * 0.5
            best = max(group, key=score_combo)
            merged_entry = dict(best)
            merged_entry["appear_rounds"] = max(g.get("appear_rounds", 0) for g in group)
            merged_entry["total_triggers"] = sum(int(g.get("total_triggers", 0) or 0) for g in group)
            merged_entry["total_hits"] = sum(int(g.get("total_hits", 0) or 0) for g in group)
            if merged_entry["total_triggers"] > 0:
                merged_entry["overall_rate"] = merged_entry["total_hits"] / merged_entry["total_triggers"]
            merged_entry["_merged_count"] = len(group)
            merged.append(merged_entry)

    return merged


def _select_high_freq_top(combos: List[dict], top_n: int = 6,
                          min_triggers: int = 20,
                          min_score: float = 70.0,
                          max_conditions: int = 3,
                          max_overlap: float = 0.5,
                          min_state_rate: Optional[float] = None) -> List[dict]:
    """
    高频策略筛选：
    1. 条件数量 2-3 个（简单策略，易于执行）
    2. 触发次数：
       - 普通模式（min_state_rate=None）：total_triggers >= min_triggers（全局）
       - 震荡市专项（min_state_rate 已设置）：state_triggers >= min_triggers（状态专项）
    3. 质量门槛：
       - 普通模式：综合评分 >= min_score（全局评分）
       - 震荡市专项：state_rate >= min_state_rate（状态专项命中率，避免全局低分过滤掉震荡市好策略）
    4. 排序：命中率 > 触发次数（强调胜率优先）
    5. 多样性约束放宽至 50%
    """
    if min_state_rate is not None:
        # 震荡市专项模式：用状态专项命中率替代全局综合评分
        qualified = [
            c for c in combos
            if 2 <= len(c.get("conditions", [])) <= max_conditions
            and c.get("state_triggers", 0) >= min_triggers
            and c.get("state_rate", 0.0) >= min_state_rate
        ]
    else:
        # 普通模式：全局触发次数和综合评分
        qualified = [
            c for c in combos
            if 2 <= len(c.get("conditions", [])) <= max_conditions
            and c.get("total_triggers", 0) >= min_triggers
            and c.get("综合评分", 0.0) >= min_score
        ]

    # 排序：命中率优先，触发次数次之
    sorted_combos = sorted(
        qualified,
        key=lambda c: (c.get("state_rate", c.get("overall_rate", 0.0)),
                       c.get("state_triggers", c.get("total_triggers", 0))),
        reverse=True
    )

    # 多样性筛选（放宽至 50%）
    selected: List[dict] = []
    selected_families: List[frozenset] = []

    for c in sorted_combos:
        if len(selected) >= top_n:
            break
        families = _get_family_set(c.get("conditions", []))
        max_current_overlap = 0.0
        for sf in selected_families:
            overlap = _family_overlap_ratio(families, sf)
            max_current_overlap = max(max_current_overlap, overlap)

        if max_current_overlap < max_overlap:
            selected.append(c)
            selected_families.append(families)

    return selected



def _select_diverse_top(combos: List[dict], top_n: int = 6,
                        max_overlap: float = 0.3,
                        min_score: float = 80.0) -> List[dict]:
    """
    多样性选择（质量优先 + 多样性保障）：
    1. 只考虑综合评分 >= min_score 的策略
    2. 按综合评分降序
    3. 逐个加入，如果与已选组合重叠度 < max_overlap 才加入
    4. 宁缺毋滥：不足 top_n 时不补充低质量策略
    """
    # 质量门槛：只考虑高分策略
    qualified = [c for c in combos if c.get("综合评分", 0.0) >= min_score]
    sorted_combos = sorted(qualified, key=lambda c: c.get("综合评分", 0.0), reverse=True)

    selected: List[dict] = []
    selected_families: List[frozenset] = []

    for c in sorted_combos:
        if len(selected) >= top_n:
            break
        families = _get_family_set(c.get("conditions", []))
        max_current_overlap = 0.0
        for sf in selected_families:
            overlap = _family_overlap_ratio(families, sf)
            max_current_overlap = max(max_current_overlap, overlap)

        # 严格多样性：重叠度必须 < 阈值才能入选
        if max_current_overlap < max_overlap:
            selected.append(c)
            selected_families.append(families)

    # 宁缺毋滥：不再补充低质量/高重叠策略
    return selected


def get_premium_pool(
    state: Optional[str] = None,
    direction: Optional[str] = None,
    top_n: int = 6,
    min_state_triggers: int = 5,
    include_high_freq: bool = True,
) -> List[dict]:
    """
    按市场状态与方向挑选精品池（两层：精品层 + 高频层）。

    规则：
      精品层（tier="精品"）：
        1. 评分 >= 80，多样性 < 30%
        2. 排序：评分 > 命中率
        3. 每组 Top N

      高频层（tier="高频"，可选）：
        震荡市：top10，评分 >= 65，触发次数 >= 5，重叠度 < 50%
        趋势市：top_n，评分 >= 70，触发次数 >= 10，重叠度 < 50%
        去重：排除已在精品层的组合

    Args:
        state:             目标市场状态（None 表示三状态都取）
        direction:         'long' / 'short' / None（两个方向都取）
        top_n:             每个状态+方向的数量上限
        min_state_triggers: 该状态最少触发次数
        include_high_freq:  是否包含高频层（默认 True）

    Returns:
        List[dict]，元素包含：
          combo_key, direction, conditions, market_state,
          score, state_rate, state_triggers, tier
    """
    cumulative = get_cumulative(direction, "pool1")
    if not cumulative:
        return []

    states = [state] if state else list(_ALL_STATES)
    directions = [direction] if direction else ["long", "short"]
    results: List[dict] = []

    for market_state in states:
        for dir_val in directions:
            # 1. 收集该状态+方向的候选组合
            candidates: List[dict] = []
            for key, entry in cumulative.items():
                if entry.get("direction") != dir_val:
                    continue
                rate, triggers = _get_state_rate(
                    entry.get("market_state_breakdown"), market_state
                )
                if triggers < min_state_triggers:
                    continue
                # 构建完整候选条目
                candidates.append({
                    "combo_key":      key,
                    "direction":      entry.get("direction"),
                    "conditions":     entry.get("conditions", []),
                    "market_state":   market_state,
                    "综合评分":        entry.get("综合评分", 0.0),
                    "score":          entry.get("综合评分", 0.0),
                    "state_rate":     rate,
                    "state_triggers": triggers,
                    "appear_rounds":  entry.get("appear_rounds", 0),
                    "total_triggers": entry.get("total_triggers", 0),
                    "total_hits":     entry.get("total_hits", 0),
                    "overall_rate":   entry.get("overall_rate", 0.0),
                    "market_state_breakdown": entry.get("market_state_breakdown", {}),
                })

            if not candidates:
                continue

            # 2. 智能合并（同指标族选宽松版代表）
            merged = _merge_similar_combos(candidates)

            # 3. 按该状态命中率排序（用于精品层筛选）
            merged.sort(
                key=lambda c: (
                    c.get("state_rate", 0.0),
                    c.get("综合评分", 0.0),
                ),
                reverse=True,
            )

            # ══════════════════════════════════════════════════════════════════
            # 精品层：质量门槛80分 + 重叠度<30%
            # ══════════════════════════════════════════════════════════════════
            premium_list = _select_diverse_top(merged, top_n=top_n)
            premium_keys = {c.get("combo_key") for c in premium_list}

            for c in premium_list:
                c["tier"] = "精品"
            results.extend(premium_list)

            # ══════════════════════════════════════════════════════════════════
            # 高频层：条件2-3个，重叠度<50%；震荡市扩容（top10，门槛降低）
            # ══════════════════════════════════════════════════════════════════
            if include_high_freq:
                high_freq_candidates = [
                    c for c in merged if c.get("combo_key") not in premium_keys
                ]
                if market_state == "震荡市":
                    # 震荡市：不排除精品层，从全量中找 2-3 条件的简单好策略
                    # 用 dict(c) 创建副本，避免覆盖精品层的 tier 标签
                    sideways_pool = [dict(c) for c in merged]
                    _min_state_rate = 0.64 if dir_val == "long" else 0.52
                    high_freq_list = _select_high_freq_top(
                        sideways_pool,
                        top_n=10,
                        min_triggers=5,
                        min_score=65.0,
                        max_conditions=3,
                        max_overlap=0.5,
                        min_state_rate=_min_state_rate,
                    )
                else:
                    # 趋势市保持原标准
                    high_freq_list = _select_high_freq_top(
                        high_freq_candidates,
                        top_n=top_n,
                        min_triggers=10,
                        min_score=70.0,
                        max_conditions=3,
                        max_overlap=0.5,
                    )
                for c in high_freq_list:
                    c["tier"] = "高频"
                results.extend(high_freq_list)

    return results


def get_rounds(pool_id: str = "pool1") -> List[dict]:
    rounds = _load_rounds_fast(pool_id)
    return sorted(rounds, key=lambda r: r.get('round_id', 0))


def get_round(round_id: int, pool_id: str = "pool1") -> Optional[dict]:
    for r in get_rounds(pool_id):
        if r.get('round_id') == round_id:
            return r
    return None


def clear_all(pool_id: str = "pool1") -> None:
    global _cumul_cache, _cumul_cache_mtime
    invalidate_cache()
    
    _, cumul_file = _get_files(pool_id)
    if _safe_exists(cumul_file):
        try:
            os.remove(cumul_file)
        except OSError:
            pass
            
    _save_raw(_empty_state(), pool_id)


def get_stable_combos(
    min_rounds: int = 5,
    min_avg_rate: float = 0.62,
    max_rate_std: float = 0.05,
    direction: Optional[str] = None,
    pool_id: str = "pool1",
) -> List[dict]:
    cumulative = get_cumulative(direction, pool_id)
    stable = []
    for key, entry in cumulative.items():
        if (
            entry.get('appear_rounds', 0) >= min_rounds
            and entry.get('avg_rate', 0.0)  >= min_avg_rate
            and entry.get('rate_std', 1.0)  <= max_rate_std
        ):
            stable.append({**entry, 'combo_key': key})
    stable.sort(key=lambda x: x.get('综合评分', 0.0), reverse=True)
    return stable


def summary(pool_id: str = "pool1") -> dict:
    data = _load_raw(pool_id)
    stable = get_stable_combos(pool_id=pool_id)
    return {
        'total_rounds':  len(data.get('rounds', [])),
        'total_combos':  len(data.get('cumulative', {})),
        'stable_combos': len(stable),
        'last_updated':  data.get('last_updated', ''),
    }


def record_live_result(combo_key: str, hit: bool, pool_id: str = "pool1") -> dict:
    data = _load_raw(pool_id)
    cumulative = data.get('cumulative', {})

    if combo_key not in cumulative:
        return {}

    entry = cumulative[combo_key]
    lt = entry.setdefault('live_tracking', _default_live_tracking())

    lt['total'] += 1
    if hit:
        lt['hits']        += 1
        lt['streak_loss']  = 0
    else:
        lt['miss']        += 1
        lt['streak_loss'] += 1

    lt['live_rate']    = round(lt['hits'] / lt['total'], 6) if lt['total'] > 0 else 0.0
    lt['last_updated'] = datetime.now().isoformat(timespec='seconds')

    avg_rate = entry.get('avg_rate', 0.0)
    lt['alert'] = (lt['total'] >= 10) and (lt['live_rate'] < avg_rate - 0.10)

    data['last_updated'] = lt['last_updated']
    _save_raw(data, pool_id)
    return dict(lt)


def get_live_alerts(pool_id: str = "pool1") -> List[dict]:
    cumulative = get_cumulative(None, pool_id)
    alerts: List[dict] = []

    for key, entry in cumulative.items():
        lt = entry.get('live_tracking') or {}
        total     = lt.get('total', 0)
        live_rate = lt.get('live_rate', 0.0)
        avg_rate  = entry.get('avg_rate', 0.0)

        if total >= 10 and live_rate < avg_rate - 0.10:
            alerts.append({
                'combo_key':    key,
                'direction':    entry.get('direction'),
                'conditions':   entry.get('conditions', []),
                'avg_rate':     avg_rate,
                'live_tracking': lt,
                'decay':        round(avg_rate - live_rate, 6),
            })

    alerts.sort(key=lambda x: x['decay'], reverse=True)
    return alerts


def get_cumulative_results(top_n: int = 1000, direction: Optional[str] = None, pool_id: str = "pool1") -> Tuple[List[dict], dict]:
    cumulative = get_cumulative(direction, pool_id)
    items = [{**v, 'combo_key': k} for k, v in cumulative.items()]
    items.sort(key=lambda x: x.get('综合评分', 0.0), reverse=True)
    return items[:top_n], cumulative


def clear(pool_id: str = "pool1") -> None:
    clear_all(pool_id)


def get_cumulative_both_pools() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """一次性返回两个池子的 cumulative dict"""
    return get_cumulative(pool_id='pool1'), get_cumulative(pool_id='pool2')

def get_ev_per_bar(combo: dict) -> float:
    """
    根据 combo 内的 pool_id 字段和 overall_rate 自动选择净 TP/SL 计算每根 K 线的期望收益。
    """
    from config import (
        POOL2_LONG_TP_PCT, POOL2_LONG_SL_PCT, 
        POOL2_SHORT_TP_PCT, POOL2_SHORT_SL_PCT
    )
    
    pool_id = combo.get('pool_id', 'pool1')
    direction = combo.get('direction', 'long')
    overall_rate = combo.get('overall_rate', 0.0)
    avg_hold_bars = max(combo.get('avg_hold_bars', 1), 1)  # 避免除以 0
    
    if pool_id == 'pool2':
        if direction == 'long':
            tp = POOL2_LONG_TP_PCT - 0.0006
            sl = POOL2_LONG_SL_PCT + 0.0006
        else:
            tp = POOL2_SHORT_TP_PCT - 0.0006
            sl = POOL2_SHORT_SL_PCT + 0.0006
    else:
        if direction == 'long':
            tp = 0.0054  # 0.6% - 0.06%
            sl = 0.0086  # 0.8% + 0.06%
        else:
            tp = 0.0074  # 0.8% - 0.06%
            sl = 0.0066  # 0.6% + 0.06%

    ev_per_trade = overall_rate * tp - (1.0 - overall_rate) * sl
    return ev_per_trade / avg_hold_bars


def backup_to_github(target_dir: Optional[str] = None) -> Dict[str, object]:
    """备份信号分析数据到可提交目录（支持 pool1/pool2）。"""
    if target_dir is None:
        target_dir = os.path.join(_DATA_DIR, "github_backup")
    _safe_makedirs(target_dir)

    def _write_json(path: str, payload: dict) -> None:
        if _USE_ORJSON:
            with _open_file(path, 'wb') as f:
                f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        else:
            with _open_file(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    result: Dict[str, object] = {"target_dir": target_dir, "files": []}

    for pool_id in ("pool1", "pool2"):
        data = _load_raw(pool_id)
        state_name = "signal_analysis_state.json" if pool_id == "pool1" else "signal_analysis_state_pool2.json"
        cumul_name = "signal_cumulative_cache.json" if pool_id == "pool1" else "signal_cumulative_cache_pool2.json"

        state_path = os.path.join(target_dir, state_name)
        cumul_path = os.path.join(target_dir, cumul_name)

        _write_json(state_path, data)
        _write_json(cumul_path, {
            "version": data.get("version", _SCHEMA_VERSION),
            "last_updated": data.get("last_updated", ""),
            "cumulative": data.get("cumulative", {}),
        })

        result["files"].extend([state_path, cumul_path])

    return result

