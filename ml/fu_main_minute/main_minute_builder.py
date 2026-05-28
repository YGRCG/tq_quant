from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
FU_MODULE_DIR = ROOT_DIR / "ml" / "fu"
if str(FU_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FU_MODULE_DIR))

from data_utils import load_ohlcv_csv  # noqa: E402
from features import build_features  # noqa: E402
from labeling import BarrierConfig, make_barrier_labels  # noqa: E402


def parse_contract_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def contract_number(contract: str | None) -> int:
    if not contract:
        return -1
    digits = "".join(ch for ch in contract.upper() if ch.isdigit())
    return int(digits) if digits else -1


def select_files(
    *,
    input_dir: str | Path,
    contract_glob: str,
    start_contract: str | None,
    end_contract: str | None,
    include_contracts: str | None,
    exclude_contracts: str | None,
    max_contracts: int | None,
) -> list[Path]:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(contract_glob), key=lambda p: contract_number(p.stem))

    include = parse_contract_list(include_contracts)
    exclude = parse_contract_list(exclude_contracts)
    start_no = contract_number(start_contract) if start_contract else None
    end_no = contract_number(end_contract) if end_contract else None

    selected: list[Path] = []
    for path in files:
        contract = path.stem.upper()
        number = contract_number(contract)
        if include and contract not in include:
            continue
        if exclude and contract in exclude:
            continue
        if start_no is not None and number < start_no:
            continue
        if end_no is not None and number > end_no:
            continue
        selected.append(path)

    if max_contracts is not None:
        selected = selected[:max_contracts]
    return selected


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def compute_trade_date(datetime_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(datetime_series)
    return (dt.dt.normalize() + pd.to_timedelta((dt.dt.hour >= 21).astype(int), unit="D")).dt.strftime("%Y-%m-%d")


def build_one_contract(path: Path, args: Any, config: BarrierConfig) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    contract = path.stem.upper()
    stat: dict[str, Any] = {
        "contract": contract,
        "source_file": str(path),
        "status": "started",
    }

    df = load_ohlcv_csv(path)
    stat["raw_rows"] = int(len(df))
    if len(df) < args.min_rows:
        stat["status"] = "skipped_min_rows"
        return None, stat

    features = build_features(
        df,
        tick_size=args.tick_size,
        profit_ticks=args.profit_ticks,
        loss_ticks=args.loss_ticks,
    )
    labels = make_barrier_labels(df, config)
    stat["feature_count"] = int(len(features.columns))
    stat["labeled_rows"] = int(len(labels))
    if labels.empty:
        stat["status"] = "skipped_no_labels"
        return None, stat

    dataset = labels.join(features, how="left")
    dataset = dataset[dataset["decision_idx"] >= args.min_history_bars].copy()

    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)
    volume_ma20 = volume.rolling(20, min_periods=10).mean()
    decision_idx = dataset["decision_idx"].astype(int)
    dataset["filter_volume"] = volume.iloc[decision_idx].to_numpy()
    dataset["filter_volume_ma20"] = volume_ma20.iloc[decision_idx].to_numpy()

    if args.min_volume > 0:
        dataset = dataset[dataset["filter_volume"] >= args.min_volume].copy()
    if args.min_volume_ma20 > 0:
        dataset = dataset[dataset["filter_volume_ma20"] >= args.min_volume_ma20].copy()
    if args.drop_any_ambiguous:
        dataset = dataset[~(dataset["long_ambiguous"] | dataset["short_ambiguous"])].copy()

    stat["kept_rows"] = int(len(dataset))
    if dataset.empty:
        stat["status"] = "skipped_no_rows_after_filters"
        return None, stat

    dataset.insert(0, "source_file", path.name)
    dataset.insert(0, "contract", contract)
    dataset["trade_date"] = compute_trade_date(dataset["decision_time"])
    dataset = dataset.sort_values("decision_time").reset_index(drop=True)

    stat.update(
        {
            "status": "ok",
            "time_start": str(pd.to_datetime(dataset["decision_time"]).min()),
            "time_end": str(pd.to_datetime(dataset["decision_time"]).max()),
            "long_label_distribution": value_counts_dict(dataset["long_label"]),
            "short_label_distribution": value_counts_dict(dataset["short_label"]),
            "avg_filter_volume": float(dataset["filter_volume"].mean()),
            "avg_filter_volume_ma20": float(dataset["filter_volume_ma20"].mean()),
        }
    )
    return dataset, stat


def estimate_daily_contract_volume(path: Path, args: Any) -> pd.DataFrame:
    contract = path.stem.upper()
    df = load_ohlcv_csv(path)
    if len(df) < args.min_rows:
        return pd.DataFrame(columns=["trade_date", "contract", "daily_filter_volume", "daily_filter_volume_ma20"])

    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)
    volume_ma20 = volume.rolling(20, min_periods=10).mean()
    candidates = pd.DataFrame(
        {
            "trade_date": compute_trade_date(df["datetime"]),
            "decision_idx": df.index,
            "filter_volume": volume,
            "filter_volume_ma20": volume_ma20,
        }
    )
    candidates = candidates[candidates["decision_idx"] >= args.min_history_bars].copy()
    if args.min_volume > 0:
        candidates = candidates[candidates["filter_volume"] >= args.min_volume].copy()
    if args.min_volume_ma20 > 0:
        candidates = candidates[candidates["filter_volume_ma20"] >= args.min_volume_ma20].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["trade_date", "contract", "daily_filter_volume", "daily_filter_volume_ma20"])

    summary = (
        candidates.groupby("trade_date", observed=False)
        .agg(
            daily_filter_volume=("filter_volume", "sum"),
            daily_filter_volume_ma20=("filter_volume_ma20", "sum"),
            daily_rows=("filter_volume", "size"),
        )
        .reset_index()
    )
    summary["contract"] = contract
    return summary[["trade_date", "contract", "daily_filter_volume", "daily_filter_volume_ma20", "daily_rows"]]


def choose_daily_main_contract(daily_volume: pd.DataFrame) -> pd.DataFrame:
    if daily_volume.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "daily_main_contract",
                "daily_main_contract_count",
                "daily_total_filter_volume",
                "selected_daily_filter_volume",
                "selected_daily_volume_share",
            ]
        )

    totals = (
        daily_volume.groupby("trade_date", observed=False)
        .agg(
            daily_main_contract_count=("contract", "count"),
            daily_total_filter_volume=("daily_filter_volume", "sum"),
        )
        .reset_index()
    )
    selected = (
        daily_volume.sort_values(
            ["trade_date", "daily_filter_volume", "daily_filter_volume_ma20", "daily_rows", "contract"],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates(subset=["trade_date"], keep="first")
        .rename(
            columns={
                "contract": "daily_main_contract",
                "daily_filter_volume": "selected_daily_filter_volume",
                "daily_filter_volume_ma20": "selected_daily_filter_volume_ma20",
            }
        )
    )
    selected = selected.merge(totals, on="trade_date", how="left")
    selected["selected_daily_volume_share"] = (
        selected["selected_daily_filter_volume"] / selected["daily_total_filter_volume"].replace(0, pd.NA)
    )
    return selected[
        [
            "trade_date",
            "daily_main_contract",
            "daily_main_contract_count",
            "daily_total_filter_volume",
            "selected_daily_filter_volume",
            "selected_daily_volume_share",
        ]
    ].reset_index(drop=True)


def select_main_daily_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = ["contract", "decision_time", "filter_volume", "filter_volume_ma20"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["filter_volume"] = pd.to_numeric(df["filter_volume"], errors="coerce").fillna(0.0)
    df["filter_volume_ma20"] = pd.to_numeric(df["filter_volume_ma20"], errors="coerce").fillna(0.0)
    df["trade_date"] = compute_trade_date(df["decision_time"])

    daily_volume = (
        df.groupby(["trade_date", "contract"], observed=False)
        .agg(
            daily_filter_volume=("filter_volume", "sum"),
            daily_filter_volume_ma20=("filter_volume_ma20", "sum"),
            daily_rows=("filter_volume", "size"),
        )
        .reset_index()
    )
    daily_main = choose_daily_main_contract(daily_volume)
    selected = df.merge(
        daily_main,
        left_on=["trade_date", "contract"],
        right_on=["trade_date", "daily_main_contract"],
        how="inner",
    )
    return selected.sort_values("decision_time").reset_index(drop=True)


def select_main_minute_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = ["contract", "decision_time", "filter_volume", "filter_volume_ma20"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["filter_volume"] = pd.to_numeric(df["filter_volume"], errors="coerce").fillna(0.0)
    df["filter_volume_ma20"] = pd.to_numeric(df["filter_volume_ma20"], errors="coerce").fillna(0.0)

    group = df.groupby("decision_time", observed=False)
    df["same_minute_contract_count"] = group["contract"].transform("count")
    df["same_minute_total_filter_volume"] = group["filter_volume"].transform("sum")
    df["selected_filter_volume_share"] = (
        df["filter_volume"] / df["same_minute_total_filter_volume"].replace(0, pd.NA)
    )

    return (
        df.sort_values(
            ["decision_time", "filter_volume", "filter_volume_ma20", "contract"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(subset=["decision_time"], keep="first")
        .sort_values("decision_time")
        .reset_index(drop=True)
    )


def update_minute_aggregates(
    *,
    dataset: pd.DataFrame,
    minute_count: pd.Series | None,
    minute_volume_sum: pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    count = dataset.groupby("decision_time", observed=False)["contract"].count()
    volume_sum = dataset.groupby("decision_time", observed=False)["filter_volume"].sum()

    if minute_count is None:
        minute_count = count
    else:
        minute_count = minute_count.add(count, fill_value=0)

    if minute_volume_sum is None:
        minute_volume_sum = volume_sum
    else:
        minute_volume_sum = minute_volume_sum.add(volume_sum, fill_value=0)

    return minute_count, minute_volume_sum


def update_best_rows(best: pd.DataFrame | None, dataset: pd.DataFrame) -> pd.DataFrame:
    if best is None:
        combined = dataset
    else:
        combined = pd.concat([best, dataset], ignore_index=True)
    return (
        combined.sort_values(
            ["decision_time", "filter_volume", "filter_volume_ma20", "contract"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(subset=["decision_time"], keep="first")
        .sort_values("decision_time")
        .reset_index(drop=True)
    )


def finalize_selected_rows(
    *,
    selected: pd.DataFrame,
    minute_count: pd.Series,
    minute_volume_sum: pd.Series,
) -> pd.DataFrame:
    selected = selected.copy()
    selected["same_minute_contract_count"] = selected["decision_time"].map(minute_count).astype(float).astype(int)
    selected["same_minute_total_filter_volume"] = selected["decision_time"].map(minute_volume_sum).astype(float)
    selected["selected_filter_volume_share"] = (
        selected["filter_volume"] / selected["same_minute_total_filter_volume"].replace(0, pd.NA)
    )
    return selected.sort_values("decision_time").reset_index(drop=True)
