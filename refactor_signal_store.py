import re

with open('core/signal_store.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace global cache definitions
content = re.sub(
    r'_cache: Optional\[dict\] = None\n_cache_mtime: float = 0\.0',
    r'_cache: Dict[str, dict] = {}\n_cache_mtime: Dict[str, float] = {}',
    content
)

content = re.sub(
    r'_cumul_cache: Optional\[dict\] = None\n_cumul_cache_mtime: float = 0\.0',
    r'_cumul_cache: Dict[str, dict] = {}\n_cumul_cache_mtime: Dict[str, float] = {}',
    content
)

content = re.sub(
    r'_rounds_cache: Optional\[list\] = None\n_rounds_cache_mtime: float = 0\.0',
    r'_rounds_cache: Dict[str, list] = {}\n_rounds_cache_mtime: Dict[str, float] = {}\n\ndef _get_files(pool_id: str) -> Tuple[str, str]:\n    if pool_id == "pool2":\n        return (os.path.join(_DATA_DIR, "signal_analysis_state_pool2.json"),\n                os.path.join(_DATA_DIR, "signal_cumulative_cache_pool2.json"))\n    return (STATE_FILE, CUMULATIVE_CACHE_FILE)\n',
    content
)

# _load_raw
content = re.sub(
    r'def _load_raw\(\) -> dict:\n.*?return _empty_state\(\)',
    r'''def _load_raw(pool_id: str = "pool1") -> dict:
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
    return _empty_state()''',
    content,
    flags=re.DOTALL
)

# _save_raw
content = re.sub(
    r'def _save_raw\(data: dict\) -> None:\n.*?_save_cumulative_cache\(data\)',
    r'''def _save_raw(data: dict, pool_id: str = "pool1") -> None:
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
    _save_cumulative_cache(data, pool_id)''',
    content,
    flags=re.DOTALL
)

# _save_cumulative_cache
content = re.sub(
    r'def _save_cumulative_cache\(data: dict\) -> None:\n.*?_cumul_cache_mtime = _safe_getmtime\(CUMULATIVE_CACHE_FILE\)',
    r'''def _save_cumulative_cache(data: dict, pool_id: str = "pool1") -> None:
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
        _cumul_cache_mtime[pool_id] = _safe_getmtime(cumul_file)''',
    content,
    flags=re.DOTALL
)

# _load_cumulative_fast
content = re.sub(
    r'def _load_cumulative_fast\(\) -> dict:\n.*?return \{\}',
    r'''def _load_cumulative_fast(pool_id: str = "pool1") -> dict:
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
    return {}''',
    content,
    flags=re.DOTALL
)

# _load_rounds_fast
content = re.sub(
    r'def _load_rounds_fast\(\) -> list:\n.*?return _rounds_cache',
    r'''def _load_rounds_fast(pool_id: str = "pool1") -> list:
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
    return _rounds_cache[pool_id]''',
    content,
    flags=re.DOTALL
)

# invalidate_cache
content = re.sub(
    r'def invalidate_cache\(\) -> None:\n.*?_rounds_cache_mtime = 0\.0',
    r'''def invalidate_cache() -> None:
    """强制清除所有内存缓存，下次加载将重新读取磁盘文件。"""
    global _cache, _cache_mtime, _cumul_cache, _cumul_cache_mtime, _rounds_cache, _rounds_cache_mtime
    _cache.clear()
    _cache_mtime.clear()
    _cumul_cache.clear()
    _cumul_cache_mtime.clear()
    _rounds_cache.clear()
    _rounds_cache_mtime.clear()''',
    content,
    flags=re.DOTALL
)

# merge_round
content = re.sub(
    r'def merge_round\(\n    new_results: List\[dict\],\n    direction: str,\n    bar_count: int = 5000,\n    timestamp: Optional\[str\] = None,\n\) -> dict:\n.*?return updated_keys',
    r'''def merge_round(
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

    return updated_keys''',
    content,
    flags=re.DOTALL
)

# merge_rounds -> merge_results
content = re.sub(
    r'def merge_rounds\(\n    long_results: Optional\[List\[dict\]\],\n    short_results: Optional\[List\[dict\]\],\n    bar_count: int = 5000,\n    timestamp: Optional\[str\] = None,\n\) -> dict:\n.*?return updated_keys',
    r'''def merge_results(
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

    return updated_keys''',
    content,
    flags=re.DOTALL
)

# _ensure_pruned_cache_fresh
content = re.sub(
    r'def _ensure_pruned_cache_fresh\(\) -> None:\n.*?rebuild_pruned_cache\(\)',
    r'''def _ensure_pruned_cache_fresh(pool_id: str = "pool1") -> None:
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
        rebuild_pruned_cache(pool_id)''',
    content,
    flags=re.DOTALL
)

# rebuild_pruned_cache
content = re.sub(
    r'def rebuild_pruned_cache\(\) -> None:\n.*?_save_cumulative_cache\(\{.*?"cumulative": pruned\}\)',
    r'''def rebuild_pruned_cache(pool_id: str = "pool1") -> None:
    data = _load_raw(pool_id)
    cumulative = data.get("cumulative", {})

    CANDIDATE_MAX_STD    = 0.05
    CANDIDATE_MIN_ROUNDS = 5

    def _keep_entry(v: dict) -> bool:
        tier = _tier_from_rate(v.get('overall_rate', 0.0), v.get('direction', 'long'))
        if tier in ('精品', '优质'):
            return True
        if tier == '候选':
            return (v.get('rate_std', 1.0) <= CANDIDATE_MAX_STD
                    and v.get('appear_rounds', 0) >= CANDIDATE_MIN_ROUNDS)
        return False

    filtered = {k: v for k, v in cumulative.items() if _keep_entry(v)}
    normalized = _normalize_all_entries(filtered)
    pruned = _prune_redundant_subsets(normalized)

    _save_cumulative_cache({**data, "cumulative": pruned}, pool_id)''',
    content,
    flags=re.DOTALL
)

# get_cumulative
content = re.sub(
    r'def get_cumulative\(direction: Optional\[str\] = None\) -> Dict\[str, dict\]:\n.*?return cumulative',
    r'''def get_cumulative(direction: Optional[str] = None, pool_id: str = "pool1") -> Dict[str, dict]:
    _ensure_pruned_cache_fresh(pool_id)
    cumulative = _load_cumulative_fast(pool_id)
    
    if direction is not None:
        cumulative = {k: v for k, v in cumulative.items() if v.get('direction') == direction}
        
    return cumulative''',
    content,
    flags=re.DOTALL
)

# get_premium_pool
content = re.sub(
    r'cumulative = get_cumulative\(direction\)',
    r'cumulative = get_cumulative(direction, "pool1")',
    content,
    count=1
)

# get_rounds
content = re.sub(
    r'def get_rounds\(\) -> List\[dict\]:\n.*?return sorted\(rounds, key=lambda r: r\.get\(\'round_id\', 0\)\)',
    r'''def get_rounds(pool_id: str = "pool1") -> List[dict]:
    rounds = _load_rounds_fast(pool_id)
    return sorted(rounds, key=lambda r: r.get('round_id', 0))''',
    content,
    flags=re.DOTALL
)

# get_round
content = re.sub(
    r'def get_round\(round_id: int\) -> Optional\[dict\]:\n.*?return None',
    r'''def get_round(round_id: int, pool_id: str = "pool1") -> Optional[dict]:
    for r in get_rounds(pool_id):
        if r.get('round_id') == round_id:
            return r
    return None''',
    content,
    flags=re.DOTALL
)

# clear_all
content = re.sub(
    r'def clear_all\(\) -> None:\n.*?_save_raw\(_empty_state\(\)\)',
    r'''def clear_all(pool_id: str = "pool1") -> None:
    global _cumul_cache, _cumul_cache_mtime
    invalidate_cache()
    
    _, cumul_file = _get_files(pool_id)
    if _safe_exists(cumul_file):
        try:
            os.remove(cumul_file)
        except OSError:
            pass
            
    _save_raw(_empty_state(), pool_id)''',
    content,
    flags=re.DOTALL
)

# get_stable_combos
content = re.sub(
    r'def get_stable_combos\(\n    min_rounds: int = 5,\n    min_avg_rate: float = 0\.62,\n    max_rate_std: float = 0\.05,\n    direction: Optional\[str\] = None,\n\) -> List\[dict\]:\n.*?cumulative = get_cumulative\(direction\)',
    r'''def get_stable_combos(
    min_rounds: int = 5,
    min_avg_rate: float = 0.62,
    max_rate_std: float = 0.05,
    direction: Optional[str] = None,
    pool_id: str = "pool1",
) -> List[dict]:
    cumulative = get_cumulative(direction, pool_id)''',
    content,
    flags=re.DOTALL
)

# summary
content = re.sub(
    r'def summary\(\) -> dict:\n.*?data = _load_raw\(\)\n    stable = get_stable_combos\(\)',
    r'''def summary(pool_id: str = "pool1") -> dict:
    data = _load_raw(pool_id)
    stable = get_stable_combos(pool_id=pool_id)''',
    content,
    flags=re.DOTALL
)

# record_live_result
content = re.sub(
    r'def record_live_result\(combo_key: str, hit: bool\) -> dict:\n.*?data = _load_raw\(\)',
    r'''def record_live_result(combo_key: str, hit: bool, pool_id: str = "pool1") -> dict:
    data = _load_raw(pool_id)''',
    content,
    flags=re.DOTALL
)
content = re.sub(
    r'_save_raw\(data\)',
    r'_save_raw(data, pool_id)',
    content
)

# get_live_alerts
content = re.sub(
    r'def get_live_alerts\(\) -> List\[dict\]:\n.*?cumulative = get_cumulative\(\)',
    r'''def get_live_alerts(pool_id: str = "pool1") -> List[dict]:
    cumulative = get_cumulative(None, pool_id)''',
    content,
    flags=re.DOTALL
)

# get_cumulative_results
content = re.sub(
    r'def get_cumulative_results\(top_n: int = 1000, direction: Optional\[str\] = None\) -> Tuple\[List\[dict\], dict\]:\n.*?cumulative = get_cumulative\(direction\)',
    r'''def get_cumulative_results(top_n: int = 1000, direction: Optional[str] = None, pool_id: str = "pool1") -> Tuple[List[dict], dict]:
    cumulative = get_cumulative(direction, pool_id)''',
    content,
    flags=re.DOTALL
)

# clear
content = re.sub(
    r'def clear\(\) -> None:\n.*?clear_all\(\)',
    r'''def clear(pool_id: str = "pool1") -> None:
    clear_all(pool_id)''',
    content,
    flags=re.DOTALL
)

# Add new functions for Pool 2 requirements
new_functions = """

def get_cumulative_both_pools() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    \"\"\"一次性返回两个池子的 cumulative dict\"\"\"
    return get_cumulative(pool_id='pool1'), get_cumulative(pool_id='pool2')

def get_ev_per_bar(combo: dict) -> float:
    \"\"\"
    根据 combo 内的 pool_id 字段和 overall_rate 自动选择净 TP/SL 计算每根 K 线的期望收益。
    \"\"\"
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

"""

content += new_functions

with open('core/signal_store.py', 'w', encoding='utf-8') as f:
    f.write(content)
