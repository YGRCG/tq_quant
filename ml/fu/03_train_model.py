from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from labeling import LABEL_LOSS, LABEL_NONE, LABEL_WIN
from modeling import (
    chronological_split,
    evaluate_side,
    expected_ticks,
    infer_feature_columns,
    make_calibration_table,
    predict_proba_frame,
    save_model_bundle,
    summarize_directional_strategy,
    summarize_thresholds,
    train_side_model,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "output" / "fu2609_dataset.csv"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "output" / "models"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "output" / "reports"


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FU2609 long/short probability models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset CSV from 02_build_dataset.py.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Output model directory.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Output report directory.")
    parser.add_argument("--profit-ticks", type=float, default=10.0, help="Profit ticks used for EV calculation.")
    parser.add_argument("--loss-ticks", type=float, default=5.0, help="Loss ticks used for EV calculation.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological train ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Chronological validation ratio.")
    parser.add_argument("--purge-bars", type=int, default=5, help="Rows to skip around split boundaries.")
    parser.add_argument("--random-state", type=int, default=20260526)
    parser.add_argument("--thresholds", default="0,1,2,3", help="Comma-separated EV thresholds in ticks.")
    parser.add_argument(
        "--drop-any-ambiguous",
        action="store_true",
        help="Drop samples where either long or short TP/SL was ambiguous inside one minute.",
    )
    return parser.parse_args()


def _side_report(
    *,
    side: str,
    model: Any,
    split_name: str,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    return evaluate_side(
        model=model,
        data=data,
        feature_columns=feature_columns,
        target_column=f"{side}_label",
        prefix=side,
    )


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    dataset_path = Path(args.dataset)
    df = pd.read_csv(dataset_path)
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df = df.sort_values("decision_time").reset_index(drop=True)

    if args.drop_any_ambiguous:
        df = df[~(df["long_ambiguous"].astype(bool) | df["short_ambiguous"].astype(bool))].copy()
        df = df.reset_index(drop=True)

    feature_columns = infer_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("No numeric feature columns were found.")

    split = chronological_split(
        df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        purge_bars=args.purge_bars,
    )

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_name, long_model = train_side_model(
        train=split.train,
        feature_columns=feature_columns,
        target_column="long_label",
        random_state=args.random_state,
    )
    _, short_model = train_side_model(
        train=split.train,
        feature_columns=feature_columns,
        target_column="short_label",
        random_state=args.random_state + 1,
    )

    base_metadata = {
        "dataset": str(dataset_path.resolve()),
        "model_name": model_name,
        "feature_count": len(feature_columns),
        "profit_ticks": args.profit_ticks,
        "loss_ticks": args.loss_ticks,
        "train_rows": len(split.train),
        "valid_rows": len(split.valid),
        "test_rows": len(split.test),
        "purge_bars": args.purge_bars,
        "drop_any_ambiguous": bool(args.drop_any_ambiguous),
    }
    save_model_bundle(
        model_dir / "fu2609_long_model.joblib",
        model=long_model,
        feature_columns=feature_columns,
        metadata={**base_metadata, "side": "long"},
    )
    save_model_bundle(
        model_dir / "fu2609_short_model.joblib",
        model=short_model,
        feature_columns=feature_columns,
        metadata={**base_metadata, "side": "short"},
    )

    metrics: dict[str, Any] = {
        "base": base_metadata,
        "long": {
            "valid": _side_report(
                side="long",
                model=long_model,
                split_name="valid",
                data=split.valid,
                feature_columns=feature_columns,
            ),
            "test": _side_report(
                side="long",
                model=long_model,
                split_name="test",
                data=split.test,
                feature_columns=feature_columns,
            ),
        },
        "short": {
            "valid": _side_report(
                side="short",
                model=short_model,
                split_name="valid",
                data=split.valid,
                feature_columns=feature_columns,
            ),
            "test": _side_report(
                side="short",
                model=short_model,
                split_name="test",
                data=split.test,
                feature_columns=feature_columns,
            ),
        },
    }

    long_test_proba = predict_proba_frame(long_model, split.test[feature_columns], "long")
    short_test_proba = predict_proba_frame(short_model, split.test[feature_columns], "short")
    long_test_ev = expected_ticks(long_test_proba, "long", args.profit_ticks, args.loss_ticks)
    short_test_ev = expected_ticks(short_test_proba, "short", args.profit_ticks, args.loss_ticks)

    test_predictions = split.test[
        [
            "decision_time",
            "entry_time",
            "entry_price",
            "long_label",
            "short_label",
            "long_exit_ticks",
            "short_exit_ticks",
            "long_ambiguous",
            "short_ambiguous",
        ]
    ].copy()
    test_predictions = pd.concat([test_predictions, long_test_proba, short_test_proba], axis=1)
    test_predictions["long_ev_ticks"] = long_test_ev
    test_predictions["short_ev_ticks"] = short_test_ev
    test_predictions.to_csv(report_dir / "fu2609_test_predictions.csv", index=False, encoding="utf-8-sig")

    make_calibration_table(split.test["long_label"], long_test_proba, prefix="long", label=LABEL_WIN).to_csv(
        report_dir / "fu2609_long_win_calibration.csv", index=False, encoding="utf-8-sig"
    )
    make_calibration_table(split.test["short_label"], short_test_proba, prefix="short", label=LABEL_WIN).to_csv(
        report_dir / "fu2609_short_win_calibration.csv", index=False, encoding="utf-8-sig"
    )

    metrics["ev_thresholds"] = {
        "long": summarize_thresholds(
            data=split.test,
            ev=long_test_ev,
            actual_ticks_col="long_exit_ticks",
            label_col="long_label",
            thresholds=thresholds,
        ),
        "short": summarize_thresholds(
            data=split.test,
            ev=short_test_ev,
            actual_ticks_col="short_exit_ticks",
            label_col="short_label",
            thresholds=thresholds,
        ),
        "directional": summarize_directional_strategy(
            data=split.test,
            long_ev=long_test_ev,
            short_ev=short_test_ev,
            thresholds=thresholds,
        ),
    }

    metrics_path = report_dir / "fu2609_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model_name: {model_name}")
    print(f"features: {len(feature_columns)}")
    print(f"train/valid/test rows: {len(split.train):,}/{len(split.valid):,}/{len(split.test):,}")
    print(f"long_model: {(model_dir / 'fu2609_long_model.joblib').resolve()}")
    print(f"short_model: {(model_dir / 'fu2609_short_model.joblib').resolve()}")
    print(f"metrics: {metrics_path.resolve()}")
    print(f"test_predictions: {(report_dir / 'fu2609_test_predictions.csv').resolve()}")
    print(
        "test accuracy:",
        {
            "long": metrics["long"]["test"]["accuracy"],
            "short": metrics["short"]["test"]["accuracy"],
        },
    )
    print(
        "test log_loss:",
        {
            "long": metrics["long"]["test"].get("log_loss"),
            "short": metrics["short"]["test"].get("log_loss"),
        },
    )
    print("classes:", [LABEL_WIN, LABEL_LOSS, LABEL_NONE])


if __name__ == "__main__":
    main()
