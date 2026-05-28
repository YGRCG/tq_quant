from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_utils import load_ohlcv_csv
from features import build_features
from labeling import BarrierConfig, make_barrier_labels


ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT_DIR / "data" / "1min_FU" / "FU2609.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "fu2609_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FU2609 ML dataset with triple-barrier labels.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input one-minute CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output dataset CSV path.")
    parser.add_argument("--tick-size", type=float, default=1.0, help="Price value of one tick.")
    parser.add_argument("--profit-ticks", type=float, default=10.0, help="Profit barrier in ticks.")
    parser.add_argument("--loss-ticks", type=float, default=5.0, help="Loss barrier in ticks.")
    parser.add_argument("--horizon-minutes", type=int, default=5, help="Calendar minutes after entry to observe.")
    parser.add_argument(
        "--same-bar-policy",
        choices=["loss", "win"],
        default="loss",
        help="When TP and SL are both touched inside one minute, assign this outcome.",
    )
    parser.add_argument("--min-history-bars", type=int, default=60, help="Drop early rows before enough history exists.")
    parser.add_argument(
        "--drop-any-ambiguous",
        action="store_true",
        help="Drop samples where either long or short TP/SL was ambiguous inside one minute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BarrierConfig(
        tick_size=args.tick_size,
        profit_ticks=args.profit_ticks,
        loss_ticks=args.loss_ticks,
        horizon_minutes=args.horizon_minutes,
        same_bar_policy=args.same_bar_policy,
    )

    df = load_ohlcv_csv(args.input)
    features = build_features(
        df,
        tick_size=args.tick_size,
        profit_ticks=args.profit_ticks,
        loss_ticks=args.loss_ticks,
    )
    labels = make_barrier_labels(df, config)
    if labels.empty:
        raise RuntimeError("No labels were generated. Check the input data and horizon settings.")

    dataset = labels.join(features, how="left")
    dataset = dataset[dataset["decision_idx"] >= args.min_history_bars].copy()
    if args.drop_any_ambiguous:
        dataset = dataset[~(dataset["long_ambiguous"] | dataset["short_ambiguous"])].copy()

    dataset = dataset.sort_values("decision_time").reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")

    feature_columns = list(features.columns)
    metadata = {
        "input": str(Path(args.input).resolve()),
        "output": str(output_path.resolve()),
        "rows": int(len(dataset)),
        "feature_count": int(len(feature_columns)),
        "tick_size": args.tick_size,
        "profit_ticks": args.profit_ticks,
        "loss_ticks": args.loss_ticks,
        "horizon_minutes": args.horizon_minutes,
        "same_bar_policy": args.same_bar_policy,
        "min_history_bars": args.min_history_bars,
        "drop_any_ambiguous": bool(args.drop_any_ambiguous),
        "time_start": str(pd.to_datetime(dataset["decision_time"]).min()),
        "time_end": str(pd.to_datetime(dataset["decision_time"]).max()),
        "long_label_distribution": dataset["long_label"].value_counts().to_dict(),
        "short_label_distribution": dataset["short_label"].value_counts().to_dict(),
        "long_ambiguous_rate": float(dataset["long_ambiguous"].mean()),
        "short_ambiguous_rate": float(dataset["short_ambiguous"].mean()),
        "bars_observed_distribution": dataset["bars_observed"].value_counts().sort_index().to_dict(),
        "feature_columns": feature_columns,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dataset: {output_path.resolve()}")
    print(f"metadata: {metadata_path.resolve()}")
    print(f"rows: {len(dataset):,}")
    print(f"feature_count: {len(feature_columns)}")
    print("long_label_distribution:", metadata["long_label_distribution"])
    print("short_label_distribution:", metadata["short_label_distribution"])
    print(f"long_ambiguous_rate: {metadata['long_ambiguous_rate']:.4%}")
    print(f"short_ambiguous_rate: {metadata['short_ambiguous_rate']:.4%}")
    print("bars_observed_distribution:", metadata["bars_observed_distribution"])


if __name__ == "__main__":
    main()
