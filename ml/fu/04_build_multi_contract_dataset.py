from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_utils import load_ohlcv_csv
from features import build_features
from labeling import BarrierConfig, make_barrier_labels


ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "1min_FU"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "fu_multi_dataset.csv"


def parse_contract_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def contract_number(contract: str) -> int:
    digits = "".join(ch for ch in contract.upper() if ch.isdigit())
    return int(digits) if digits else -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-contract FU dataset with per-contract labels/features.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing FU*.csv files.")
    parser.add_argument("--contract-glob", default="FU*.csv", help="File glob under input-dir, for example FU260*.csv.")
    parser.add_argument("--start-contract", default=None, help="Inclusive lower contract bound, for example FU2301.")
    parser.add_argument("--end-contract", default=None, help="Inclusive upper contract bound, for example FU2609.")
    parser.add_argument("--include-contracts", default=None, help="Comma-separated contracts to include.")
    parser.add_argument("--exclude-contracts", default=None, help="Comma-separated contracts to exclude.")
    parser.add_argument("--max-contracts", type=int, default=None, help="Optional cap after sorting/filtering contracts.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output merged dataset CSV path.")
    parser.add_argument("--tick-size", type=float, default=1.0)
    parser.add_argument("--profit-ticks", type=float, default=10.0)
    parser.add_argument("--loss-ticks", type=float, default=5.0)
    parser.add_argument("--horizon-minutes", type=int, default=5)
    parser.add_argument("--same-bar-policy", choices=["loss", "win"], default="loss")
    parser.add_argument("--min-rows", type=int, default=5000, help="Skip contracts with fewer raw rows.")
    parser.add_argument("--min-history-bars", type=int, default=60, help="Drop early rows before enough history exists.")
    parser.add_argument("--min-volume", type=float, default=5.0, help="Filter decision bars by current volume.")
    parser.add_argument(
        "--min-volume-ma20",
        type=float,
        default=20.0,
        help="Filter decision bars by rolling 20-bar mean volume.",
    )
    parser.add_argument("--drop-any-ambiguous", action="store_true")
    return parser.parse_args()


def select_files(args: argparse.Namespace) -> list[Path]:
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.contract_glob), key=lambda p: contract_number(p.stem))

    include = parse_contract_list(args.include_contracts)
    exclude = parse_contract_list(args.exclude_contracts)
    start_no = contract_number(args.start_contract) if args.start_contract else None
    end_no = contract_number(args.end_contract) if args.end_contract else None

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

    if args.max_contracts is not None:
        selected = selected[: args.max_contracts]
    return selected


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def build_one_contract(path: Path, args: argparse.Namespace, config: BarrierConfig) -> tuple[pd.DataFrame | None, dict[str, Any]]:
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
    dataset = dataset.sort_values("decision_time").reset_index(drop=True)

    stat.update(
        {
            "status": "ok",
            "time_start": str(pd.to_datetime(dataset["decision_time"]).min()),
            "time_end": str(pd.to_datetime(dataset["decision_time"]).max()),
            "long_label_distribution": value_counts_dict(dataset["long_label"]),
            "short_label_distribution": value_counts_dict(dataset["short_label"]),
            "long_ambiguous_rate": float(dataset["long_ambiguous"].mean()),
            "short_ambiguous_rate": float(dataset["short_ambiguous"].mean()),
            "avg_filter_volume": float(dataset["filter_volume"].mean()),
            "avg_filter_volume_ma20": float(dataset["filter_volume_ma20"].mean()),
        }
    )
    return dataset, stat


def main() -> None:
    args = parse_args()
    config = BarrierConfig(
        tick_size=args.tick_size,
        profit_ticks=args.profit_ticks,
        loss_ticks=args.loss_ticks,
        horizon_minutes=args.horizon_minutes,
        same_bar_policy=args.same_bar_policy,
    )

    files = select_files(args)
    if not files:
        raise RuntimeError("No contract files matched the selection.")

    frames: list[pd.DataFrame] = []
    contract_stats: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        dataset, stat = build_one_contract(path, args, config)
        contract_stats.append(stat)
        if dataset is not None:
            frames.append(dataset)
        print(
            f"[{index}/{len(files)}] {path.stem}: {stat['status']}, "
            f"raw={stat.get('raw_rows', 0):,}, kept={stat.get('kept_rows', 0):,}"
        )

    if not frames:
        raise RuntimeError("No rows were kept after filters. Loosen min-volume/min-rows settings.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["decision_time", "contract"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    ok_stats = [item for item in contract_stats if item["status"] == "ok"]
    metadata = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "contract_glob": args.contract_glob,
        "output": str(output_path.resolve()),
        "selected_contract_files": len(files),
        "ok_contracts": len(ok_stats),
        "rows": int(len(merged)),
        "contracts": sorted(merged["contract"].unique().tolist(), key=contract_number),
        "tick_size": args.tick_size,
        "profit_ticks": args.profit_ticks,
        "loss_ticks": args.loss_ticks,
        "horizon_minutes": args.horizon_minutes,
        "same_bar_policy": args.same_bar_policy,
        "min_rows": args.min_rows,
        "min_history_bars": args.min_history_bars,
        "min_volume": args.min_volume,
        "min_volume_ma20": args.min_volume_ma20,
        "drop_any_ambiguous": bool(args.drop_any_ambiguous),
        "time_start": str(pd.to_datetime(merged["decision_time"]).min()),
        "time_end": str(pd.to_datetime(merged["decision_time"]).max()),
        "long_label_distribution": value_counts_dict(merged["long_label"]),
        "short_label_distribution": value_counts_dict(merged["short_label"]),
        "contract_stats": contract_stats,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dataset: {output_path.resolve()}")
    print(f"metadata: {metadata_path.resolve()}")
    print(f"rows: {len(merged):,}")
    print(f"contracts: {len(ok_stats)}")
    print("long_label_distribution:", metadata["long_label_distribution"])
    print("short_label_distribution:", metadata["short_label_distribution"])


if __name__ == "__main__":
    main()
