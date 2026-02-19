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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── 文件路径 ─────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(os.path.dirname(_THIS_DIR), 'data')
STATE_FILE = os.path.join(_DATA_DIR, 'signal_analysis_state.json')

_SCHEMA_VERSION = "3.0"


# ═══════════════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════════════

def _combo_key(direction: str, conditions: List[str]) -> str:
    """生成组合的唯一键（条件排序后拼接）。"""
    return f"{direction}|{'+'.join(sorted(conditions))}"


def _load_raw() -> dict:
    """加载 JSON 文件，不存在则返回空骨架。"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
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


def _save_raw(data: dict) -> None:
    """保存 JSON 文件（原子写：先写临时文件再重命名）。"""
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def _next_round_id(data: dict) -> int:
    rounds = data.get('rounds', [])
    if not rounds:
        return 1
    return max(r.get('round_id', 0) for r in rounds) + 1


# ═══════════════════════════════════════════════════════════════════════════
# 累计指标计算
# ═══════════════════════════════════════════════════════════════════════════

# 做多：含费后 TP净0.54%，SL净0.86%
_LONG_TP_PCT  = 0.0054
_LONG_SL_PCT  = 0.0086

# 做空：含费后 TP净0.74%，SL净0.66%
_SHORT_TP_PCT = 0.0074
_SHORT_SL_PCT = 0.0066

_ALL_STATES = ("多头趋势", "空头趋势", "震荡市")


def _estimated_pnl_pct(overall_rate: float, total_triggers: int,
                        direction: str = 'long') -> float:
    """
    估算累计盈亏百分比。

    做多：每笔期望 = overall_rate × 0.54% - (1 - overall_rate) × 0.86%
    做空：每笔期望 = overall_rate × 0.74% - (1 - overall_rate) × 0.66%
    估算总盈亏 = 每笔期望 × total_triggers
    """
    tp_pct = _LONG_TP_PCT  if direction == 'long' else _SHORT_TP_PCT
    sl_pct = _LONG_SL_PCT  if direction == 'long' else _SHORT_SL_PCT
    per_trade = overall_rate * tp_pct - (1.0 - overall_rate) * sl_pct
    return round(per_trade * total_triggers * 100, 4)   # 单位：%


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
    return float(info.get("rate", 0.0)), int(info.get("total_triggers", 0))


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


def _compute_cumulative_metrics(history: List[dict], direction: str = 'long') -> dict:
    """
    根据历次命中率记录计算累计指标。

    history 元素字段：round_id, hit_rate, trigger_count, hit_count,
                      market_state_breakdown (可选)
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
    """
    rates = [h['hit_rate'] for h in history]
    n = len(rates)

    total_triggers = sum(h['trigger_count'] for h in history)
    total_hits     = sum(h.get('hit_count', 0) for h in history)
    overall_rate   = total_hits / total_triggers if total_triggers > 0 else 0.0

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
    stab_score  = min(100.0, max(0.0, (1.0 - rate_std / 0.15) * 100.0))
    round_score = min(100.0, math.log(n + 1) / math.log(11) * 100.0)  # n=10 时满分
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
        'estimated_pnl_pct':      _estimated_pnl_pct(overall_rate, total_triggers, direction),
        'market_state_breakdown': _merge_market_state_breakdown(history),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 公开 API
# ═══════════════════════════════════════════════════════════════════════════

def merge_round(
    new_results: List[dict],
    direction: str,
    bar_count: int = 5000,
    timestamp: Optional[str] = None,
) -> dict:
    """
    将一轮分析结果合并到持久化状态中。

    Args:
        new_results: signal_analyzer.analyze() 的返回值
        direction:   'long' 或 'short'
        bar_count:   本轮使用的 K 线数量（用于记录）
        timestamp:   ISO 格式时间字符串，默认取当前时间

    Returns:
        本轮新增/更新的 cumulative 条目 dict（key 为 combo_key）
    """
    data = _load_raw()

    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec='seconds')

    round_id = _next_round_id(data)

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

        if key not in cumulative:
            cumulative[key] = {
                'direction':    item['direction'],
                'conditions':   sorted(item['conditions']),
                'history':      [],
                'live_tracking': _default_live_tracking(),
            }
        elif 'live_tracking' not in cumulative[key]:
            cumulative[key]['live_tracking'] = _default_live_tracking()

        entry = cumulative[key]
        entry['history'].append({
            'round_id':               round_id,
            'hit_rate':               item['hit_rate'],
            'trigger_count':          item['trigger_count'],
            'hit_count':              item['hit_count'],
            'tier':                   item['tier'],
            'market_state_breakdown': item.get('market_state_breakdown'),
        })

        metrics = _compute_cumulative_metrics(entry['history'], direction=item['direction'])
        entry.update(metrics)
        updated_keys[key] = entry

    data['version']      = _SCHEMA_VERSION
    data['last_updated'] = timestamp
    _save_raw(data)

    return updated_keys


def get_cumulative(direction: Optional[str] = None) -> Dict[str, dict]:
    """
    读取所有（或指定方向的）累计结果。

    Args:
        direction: 'long' / 'short' / None（返回全部）

    Returns:
        dict，key 为 combo_key，value 为累计条目。
    """
    data = _load_raw()
    cumulative = data.get('cumulative', {})
    if direction is None:
        return cumulative
    return {k: v for k, v in cumulative.items() if v.get('direction') == direction}


def get_premium_pool(
    state: Optional[str] = None,
    direction: Optional[str] = None,
    top_n: int = 6,
    min_state_triggers: int = 5,
) -> List[dict]:
    """
    按市场状态与方向挑选精品池（每组 TOP N）。

    规则：
      - 过滤该状态触发次数 < min_state_triggers 的组合
      - 主排序：该状态命中率（降序）
      - 次排序：综合评分（降序）

    Args:
        state:             目标市场状态（None 表示三状态都取）
        direction:         'long' / 'short' / None（两个方向都取）
        top_n:             每个状态+方向的数量上限
        min_state_triggers: 该状态最少触发次数

    Returns:
        List[dict]，元素包含：
          combo_key, direction, conditions, market_state,
          score, state_rate, state_triggers
    """
    cumulative = get_cumulative(direction)
    if not cumulative:
        return []

    states = [state] if state else list(_ALL_STATES)
    directions = [direction] if direction else ["long", "short"]
    results: List[dict] = []

    for market_state in states:
        for dir_val in directions:
            candidates: List[Tuple[str, dict, float, int]] = []
            for key, entry in cumulative.items():
                if entry.get("direction") != dir_val:
                    continue
                rate, triggers = _get_state_rate(
                    entry.get("market_state_breakdown"), market_state
                )
                if triggers < min_state_triggers:
                    continue
                candidates.append((key, entry, rate, triggers))

            if not candidates:
                continue

            candidates.sort(
                key=lambda item: (
                    item[2],  # state_rate
                    item[1].get("综合评分", 0.0),
                ),
                reverse=True,
            )

            for key, entry, rate, triggers in candidates[:top_n]:
                results.append({
                    "combo_key":      key,
                    "direction":      entry.get("direction"),
                    "conditions":     entry.get("conditions", []),
                    "market_state":   market_state,
                    "score":          entry.get("综合评分", 0.0),
                    "state_rate":     rate,
                    "state_triggers": triggers,
                })

    return results


def get_rounds() -> List[dict]:
    """返回所有历史轮次记录列表（按 round_id 升序）。"""
    data = _load_raw()
    rounds = data.get('rounds', [])
    return sorted(rounds, key=lambda r: r.get('round_id', 0))


def get_round(round_id: int) -> Optional[dict]:
    """返回指定轮次的记录，不存在返回 None。"""
    for r in get_rounds():
        if r.get('round_id') == round_id:
            return r
    return None


def clear_all() -> None:
    """清空所有历史记录（写入空骨架）。"""
    _save_raw(_empty_state())


def get_stable_combos(
    min_rounds: int = 3,
    min_avg_rate: float = 0.64,
    max_rate_std: float = 0.08,
    direction: Optional[str] = None,
) -> List[dict]:
    """
    返回通过多轮稳定性筛选的组合列表，按综合评分降序。

    筛选条件（默认值）：
      - appear_rounds >= 3
      - avg_rate      >= 0.64
      - rate_std      <= 0.08

    Args:
        min_rounds:   最少出现轮次
        min_avg_rate: 最低平均命中率
        max_rate_std: 最高命中率标准差
        direction:    过滤方向（None = 全部）

    Returns:
        List[dict]，每个元素即 cumulative 条目，附加字段 'combo_key'。
    """
    cumulative = get_cumulative(direction)
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


def summary() -> dict:
    """
    返回当前状态摘要（用于 UI 信息展示）。

    Returns:
        {
          "total_rounds"   : int,
          "total_combos"   : int,
          "stable_combos"  : int,
          "last_updated"   : str,
        }
    """
    data = _load_raw()
    stable = get_stable_combos()
    return {
        'total_rounds':  len(data.get('rounds', [])),
        'total_combos':  len(data.get('cumulative', {})),
        'stable_combos': len(stable),
        'last_updated':  data.get('last_updated', ''),
    }


def record_live_result(combo_key: str, hit: bool) -> dict:
    """
    记录一笔实盘/模拟盘结果，更新对应组合的 live_tracking，并触发衰减告警检查。

    衰减告警条件：total >= 10 AND live_rate < avg_rate - 0.10

    Args:
        combo_key: 组合唯一键（格式同 _combo_key）
        hit:       True 表示止盈命中，False 表示止损/超时

    Returns:
        更新后的 live_tracking dict；若 combo_key 不存在则返回空 dict。
    """
    data = _load_raw()
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
    _save_raw(data)
    return dict(lt)


def get_live_alerts() -> List[dict]:
    """
    返回所有触发衰减告警的组合列表。

    衰减定义：live_rate < avg_rate - 0.10（且 total >= 10）

    Returns:
        List[dict]，每个元素包含 combo_key、avg_rate、live_tracking 字段，
        按 (avg_rate - live_rate) 降序排列（衰减最严重的排前面）。
    """
    cumulative = get_cumulative()
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


def get_cumulative_results(top_n: int = 200, direction: Optional[str] = None) -> List[dict]:
    """UI 调用的接口，返回按综合评分降序的累计结果列表（附加 combo_key 字段）。"""
    cumulative = get_cumulative(direction)
    items = [{**v, 'combo_key': k} for k, v in cumulative.items()]
    items.sort(key=lambda x: x.get('综合评分', 0.0), reverse=True)
    return items[:top_n]


def clear() -> None:
    """UI 调用的清空接口，等同于 clear_all()。"""
    clear_all()
