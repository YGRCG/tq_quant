from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from main_minute_builder import (
    build_one_contract,
    choose_daily_main_contract,
    contract_number,
    estimate_daily_contract_volume,
    finalize_selected_rows,
    select_files,
    select_main_daily_rows,
    select_main_minute_rows,
    update_best_rows,
    update_minute_aggregates,
    value_counts_dict,
)

import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
FU_MODULE_DIR = ROOT_DIR / "ml" / "fu"
if str(FU_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FU_MODULE_DIR))

from labeling import BarrierConfig  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "1min_FU"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "fu_main_minute_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dominant FU contract dataset with enhanced features.")
    parser.add_argument(
        "--input",
        default=None,
        help="Optional existing multi-contract dataset CSV. If set, only select highest-volume rows from it.",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing raw FU*.csv files.")
    parser.add_argument(
        "--main-mode",
        choices=["daily", "minute"],
        default="daily",
        help="daily: choose one main contract per trade date; minute: choose max-volume contract per minute.",
    )
    parser.add_argument("--contract-glob", default="FU*.csv", help="File glob under input-dir.")
    parser.add_argument("--start-contract", default="FU2401", help="Inclusive lower contract bound.")
    parser.add_argument("--end-contract", default="FU2609", help="Inclusive upper contract bound.")
    parser.add_argument("--include-contracts", default=None, help="Comma-separated contracts to include.")
    parser.add_argument("--exclude-contracts", default=None, help="Comma-separated contracts to exclude.")
    parser.add_argument("--max-contracts", type=int, default=None, help="Optional cap after sorting/filtering contracts.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output dataset CSV path.")
    parser.add_argument("--tick-size", type=float, default=1.0, help="Price value of one tick.")
    parser.add_argument("--profit-ticks", type=float, default=10.0, help="Profit barrier in ticks.")
    parser.add_argument("--loss-ticks", type=float, default=5.0, help="Loss barrier in ticks.")
    parser.add_argument("--horizon-minutes", type=int, default=5, help="Calendar minutes after entry to observe.")
    parser.add_argument("--same-bar-policy", choices=["loss", "win"], default="loss")
    parser.add_argument("--min-rows", type=int, default=5000, help="Skip contracts with fewer raw rows.")
    parser.add_argument("--min-history-bars", type=int, default=60, help="Drop early rows before enough history exists.")
    parser.add_argument("--min-volume", type=float, default=5.0, help="Filter decision bars by current volume.")
    parser.add_argument("--min-volume-ma20", type=float, default=20.0, help="Filter decision bars by MA20 volume.")
    parser.add_argument("--drop-any-ambiguous", action="store_true")
    return parser.parse_args()


def build_from_existing_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    if args.min_volume > 0:
        df = df[pd.to_numeric(df["filter_volume"], errors="coerce").fillna(0.0) >= args.min_volume].copy()
    if args.min_volume_ma20 > 0:
        df = df[pd.to_numeric(df["filter_volume_ma20"], errors="coerce").fillna(0.0) >= args.min_volume_ma20].copy()
    if df.empty:
        raise RuntimeError("No rows left after filters.")

    if args.main_mode == "daily":
        selected = select_main_daily_rows(df)
    else:
        selected = select_main_minute_rows(df)
    metadata = {
        "mode": "select_existing_dataset",
        "main_mode": args.main_mode,
        "input": str(input_path.resolve()),
        "candidate_rows": int(len(df)),
        "contract_stats": [],
    }
    return selected, metadata


def build_from_raw_contracts(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    config = BarrierConfig(
        tick_size=args.tick_size,
        profit_ticks=args.profit_ticks,
        loss_ticks=args.loss_ticks,
        horizon_minutes=args.horizon_minutes,
        same_bar_policy=args.same_bar_policy,
    )
    files = select_files(
        input_dir=args.input_dir,
        contract_glob=args.contract_glob,
        start_contract=args.start_contract,
        end_contract=args.end_contract,
        include_contracts=args.include_contracts,
        exclude_contracts=args.exclude_contracts,
        max_contracts=args.max_contracts,
    )
    if not files:
        raise RuntimeError("No contract files matched the selection.")

    best: pd.DataFrame | None = None
    minute_count: pd.Series | None = None
    minute_volume_sum: pd.Series | None = None
    contract_stats: list[dict] = []
    candidate_rows = 0
    daily_main = pd.DataFrame()

    if args.main_mode == "daily":
        daily_volume_frames: list[pd.DataFrame] = []
        for path in files:
            daily_volume = estimate_daily_contract_volume(path, args)
            if not daily_volume.empty:
                daily_volume_frames.append(daily_volume)
        if not daily_volume_frames:
            raise RuntimeError("No daily volume rows were available after filters.")
        daily_main = choose_daily_main_contract(pd.concat(daily_volume_frames, ignore_index=True))

    for index, path in enumerate(files, start=1):
        dataset, stat = build_one_contract(path, args, config)
        contract_stats.append(stat)
        if dataset is not None:
            candidate_rows += len(dataset)
            if args.main_mode == "daily":
                selected_contract_rows = dataset.merge(
                    daily_main,
                    left_on=["trade_date", "contract"],
                    right_on=["trade_date", "daily_main_contract"],
                    how="inner",
                )
                if not selected_contract_rows.empty:
                    best = (
                        selected_contract_rows
                        if best is None
                        else pd.concat([best, selected_contract_rows], ignore_index=True)
                    )
            else:
                minute_count, minute_volume_sum = update_minute_aggregates(
                    dataset=dataset,
                    minute_count=minute_count,
                    minute_volume_sum=minute_volume_sum,
                )
                best = update_best_rows(best, dataset)

        print(
            f"[{index}/{len(files)}] {path.stem}: {stat['status']}, "
            f"raw={stat.get('raw_rows', 0):,}, kept={stat.get('kept_rows', 0):,}"
        )

    if best is None:
        raise RuntimeError("No rows were kept after filters.")

    if args.main_mode == "daily":
        selected = best.sort_values("decision_time").reset_index(drop=True)
    else:
        if minute_count is None or minute_volume_sum is None:
            raise RuntimeError("No minute aggregates were available.")
        selected = finalize_selected_rows(
            selected=best,
            minute_count=minute_count,
            minute_volume_sum=minute_volume_sum,
        )
    metadata = {
        "mode": "build_from_raw_contracts",
        "main_mode": args.main_mode,
        "input_dir": str(Path(args.input_dir).resolve()),
        "contract_glob": args.contract_glob,
        "start_contract": args.start_contract,
        "end_contract": args.end_contract,
        "selected_contract_files": len(files),
        "ok_contracts": sum(1 for item in contract_stats if item["status"] == "ok"),
        "candidate_rows": int(candidate_rows),
        "daily_main_contracts": daily_main.to_dict(orient="records") if args.main_mode == "daily" else [],
        "contract_stats": contract_stats,
    }
    return selected, metadata


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.input:
        selected, build_metadata = build_from_existing_dataset(args)
    else:
        selected, build_metadata = build_from_raw_contracts(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")

    feature_columns = [
        col
        for col in selected.columns
        if col
        not in {
            "contract",
            "source_file",
            "trade_date",
            "decision_idx",
            "decision_time",
            "entry_time",
            "horizon_end_time",
            "entry_price",
            "bars_observed",
            "filter_volume",
            "filter_volume_ma20",
            "same_minute_contract_count",
            "same_minute_total_filter_volume",
            "selected_filter_volume_share",
            "daily_main_contract",
            "daily_main_contract_count",
            "daily_total_filter_volume",
            "selected_daily_filter_volume",
            "selected_daily_volume_share",
            "long_label",
            "short_label",
            "long_ambiguous",
            "short_ambiguous",
            "long_exit_ticks",
            "short_exit_ticks",
            "long_exit_bar_offset",
            "short_exit_bar_offset",
        }
        and pd.api.types.is_numeric_dtype(selected[col])
    ]
    metadata = {
        **build_metadata,
        "output": str(output_path.resolve()),
        "output_rows": int(len(selected)),
        "feature_count": int(len(feature_columns)),
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
        "main_mode": args.main_mode,
        "selection_rule": (
            "per trade_date choose contract with max daily sum(filter_volume)"
            if args.main_mode == "daily"
            else "per decision_time choose max filter_volume, then max filter_volume_ma20, then contract name"
        ),
        "time_start": str(pd.to_datetime(selected["decision_time"]).min()),
        "time_end": str(pd.to_datetime(selected["decision_time"]).max()),
        "contracts": sorted(selected["contract"].unique().tolist(), key=contract_number),
        "contract_distribution": value_counts_dict(selected["contract"]),
        "long_label_distribution": value_counts_dict(selected["long_label"]),
        "short_label_distribution": value_counts_dict(selected["short_label"]),
        "avg_selected_filter_volume_share": (
            float(selected["selected_filter_volume_share"].mean())
            if "selected_filter_volume_share" in selected.columns
            else None
        ),
        "avg_same_minute_contract_count": (
            float(selected["same_minute_contract_count"].mean())
            if "same_minute_contract_count" in selected.columns
            else None
        ),
        "avg_selected_daily_volume_share": (
            float(selected["selected_daily_volume_share"].mean())
            if "selected_daily_volume_share" in selected.columns
            else None
        ),
        "avg_daily_main_contract_count": (
            float(selected["daily_main_contract_count"].mean())
            if "daily_main_contract_count" in selected.columns
            else None
        ),
        "feature_columns": feature_columns,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dataset: {output_path.resolve()}")
    print(f"metadata: {metadata_path.resolve()}")
    print(f"candidate_rows: {metadata['candidate_rows']:,}")
    print(f"output_rows: {len(selected):,}")
    print(f"feature_count: {len(feature_columns)}")
    print("contract_distribution:", metadata["contract_distribution"])
    print("long_label_distribution:", metadata["long_label_distribution"])
    print("short_label_distribution:", metadata["short_label_distribution"])


if __name__ == "__main__":
    main()
