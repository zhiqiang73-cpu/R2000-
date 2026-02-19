"""
core/signal_utils.py
信号分析通用工具函数，供引擎和 UI 共用。
"""
from typing import List, Dict

_COND_LABELS = {
    "rsi": {
        "label": "RSI",
        "long":  {"loose": "<40", "strict": "<30"},
        "short": {"loose": ">60", "strict": ">70"},
    },
    "k": {
        "label": "KDJ-K",
        "long":  {"loose": "<35", "strict": "<25"},
        "short": {"loose": ">65", "strict": ">75"},
    },
    "j": {
        "label": "KDJ-J",
        "long":  {"loose": "<25", "strict": "<15"},
        "short": {"loose": ">75", "strict": ">85"},
    },
    "boll_pos": {
        "label": "布林位置",
        "long":  {"loose": "<0.25", "strict": "<0.15"},
        "short": {"loose": ">0.75", "strict": ">0.85"},
    },
    "vol_ratio": {
        "label": "量比",
        "long":  {"loose": ">1.3", "strict": ">1.8"},
        "short": {"loose": ">1.3", "strict": ">1.8"},
    },
    "close_vs_ma5": {
        "label": "偏离MA5",
        "long":  {"loose": "<-1.0%", "strict": "<-1.5%"},
        "short": {"loose": ">1.0%", "strict": ">1.5%"},
    },
    "lower_shd": {
        "label": "下影线/实体",
        "long":  {"loose": ">1.5", "strict": ">2.5"},
        "short": {},
    },
    "upper_shd": {
        "label": "上影线/实体",
        "long":  {},
        "short": {"loose": ">1.5", "strict": ">2.5"},
    },
    "consec_bear": {
        "label": "连续阴线",
        "long":  {"loose": "≥2", "strict": "≥3"},
        "short": {},
    },
    "consec_bull": {
        "label": "连续阳线",
        "long":  {},
        "short": {"loose": "≥2", "strict": "≥3"},
    },
    "atr_ratio": {
        "label": "ATR波动率比",
        "long":  {"loose": ">1.2", "strict": ">1.8"},
        "short": {"loose": ">1.2", "strict": ">1.8"},
    },
}


def _cond_label(name: str, direction: str) -> str:
    """获取单个条件的中文描述"""
    strict = name.endswith("_strict")
    level = "strict" if strict else "loose"
    base = name.replace("_strict", "").replace("_loose", "")
    info = _COND_LABELS.get(base)
    if not info:
        return name
    label = info.get("label", base)
    rule = info.get(direction, {}).get(level)
    if not rule:
        return name
    suffix = "（严格）" if strict else "（宽松）"
    return f"{label}{rule}{suffix}"


def _format_conditions(conditions: List[str], direction: str) -> str:
    """格式化条件组合为中文描述字符串"""
    return " & ".join(_cond_label(c, direction) for c in conditions)
