"""
从币安下载 BTCUSDT 合约 1 分钟 K 线数据

用法:
    python scripts/download_binance_klines.py

下载内容:
    1. 最近三个月 1M K 线 -> historydata/btcusdt_1m_recent_3m.parquet
    2. 2021-2022 年 1M K 线 -> historydata/btcusdt_1m_2021_2022.parquet

支持代理: 通过 .env 设置 HTTP_PROXY 或 SOCKS_PROXY
"""
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 添加项目根目录并加载 .env（通过 config）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import config  # 触发 .env 加载
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("请安装 requests: pip install requests")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("请安装 pandas: pip install pandas")
    sys.exit(1)

# 币安合约 K 线 API
FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
LIMIT_PER_REQUEST = 1000  # 单次最多 1000 根
MS_PER_MINUTE = 60_000
REQUEST_DELAY = 0.3  # 请求间隔，避免限流


def _get_proxies():
    """从环境变量读取代理"""
    proxies = {}
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    socks_proxy = os.environ.get("SOCKS_PROXY") or os.environ.get("socks_proxy")
    if http_proxy:
        proxies = {"http": http_proxy, "https": http_proxy}
    elif socks_proxy:
        proxies = {"http": socks_proxy, "https": socks_proxy}
    return proxies or None


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, proxies=None) -> list:
    """单次请求 K 线"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT_PER_REQUEST,
    }
    resp = requests.get(FUTURES_KLINES_URL, params=params, timeout=30, proxies=proxies)
    resp.raise_for_status()
    return resp.json()


def download_range(symbol: str, interval: str, start_dt: datetime, end_dt: datetime, proxies=None) -> pd.DataFrame:
    """
    下载指定时间范围的 K 线，分页请求并合并
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    all_rows = []
    current = start_ms

    while current < end_ms:
        batch = fetch_klines(symbol, interval, current, end_ms, proxies)
        if not batch:
            break
        all_rows.extend(batch)
        # 下一批从最后一根 K 线的收盘时间之后开始
        last_close = batch[-1][6]
        current = last_close + 1
        time.sleep(REQUEST_DELAY)
        print(f"  已获取 {len(all_rows):,} 根...", end="\r")

    if not all_rows:
        return pd.DataFrame()

    # Binance 返回格式: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"open_time": "timestamp"})
    return df


def main():
    os.chdir(ROOT)
    data_dir = ROOT / "historydata"
    data_dir.mkdir(exist_ok=True)
    proxies = _get_proxies()
    if proxies:
        print("使用代理:", list(proxies.keys())[0])

    symbol = "BTCUSDT"
    interval = "1m"
    tz_utc = timezone.utc

    # 1. 最近三个月
    now = datetime.now(tz_utc)
    start_3m = now - timedelta(days=90)
    print(f"\n[1/2] 下载最近三个月: {start_3m.date()} ~ {now.date()}")
    df_3m = download_range(symbol, interval, start_3m, now, proxies)
    if len(df_3m) > 0:
        out_3m = data_dir / "btcusdt_1m_recent_3m.parquet"
        df_3m.to_parquet(out_3m, index=False)
        print(f"\n  保存: {out_3m} ({len(df_3m):,} 行)")
    else:
        print("  未获取到数据")

    # 2. 2021-2022
    start_2021 = datetime(2021, 1, 1, tzinfo=tz_utc)
    end_2022 = datetime(2022, 12, 31, 23, 59, 59, tzinfo=tz_utc)
    print(f"\n[2/2] 下载 2021-2022: {start_2021.date()} ~ {end_2022.date()}")
    df_2122 = download_range(symbol, interval, start_2021, end_2022, proxies)
    if len(df_2122) > 0:
        out_2122 = data_dir / "btcusdt_1m_2021_2022.parquet"
        df_2122.to_parquet(out_2122, index=False)
        print(f"\n  保存: {out_2122} ({len(df_2122):,} 行)")
    else:
        print("  未获取到数据")

    print("\n完成。")


if __name__ == "__main__":
    main()
