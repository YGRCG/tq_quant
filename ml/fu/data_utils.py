from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CHINESE_COLUMN_MAP = {
    "时间": "datetime",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "持仓量": "open_interest",
}

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "open_interest"]


def read_csv_with_fallback(path: str | Path, encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gbk")) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load FU one-minute OHLCV data and normalize column names."""
    df = read_csv_with_fallback(path)
    df = df.rename(columns={col: CHINESE_COLUMN_MAP.get(col, col) for col in df.columns})

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.reset_index(drop=True)
    return df


def summarize_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    gaps = df["datetime"].diff().dropna()
    if gaps.empty:
        return pd.DataFrame(columns=["gap", "count"])

    counts = gaps.value_counts().reset_index()
    counts.columns = ["gap", "count"]
    counts = counts.sort_values(["gap"]).reset_index(drop=True)
    return counts
