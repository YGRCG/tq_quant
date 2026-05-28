from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
FU_MODULE_DIR = ROOT_DIR / "ml" / "fu"
if str(FU_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FU_MODULE_DIR))

from labeling import LABEL_WIN  # noqa: E402
from modeling import (  # noqa: E402
    chronological_split_by_time,
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


DEFAULT_DATASET = SCRIPT_DIR / "output" / "fu_main_minute_dataset.csv"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "output" / "models"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "output" / "reports"


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train highest-volume-per-minute FU models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--profit-ticks", type=float, default=10.0)
    parser.add_argument("--loss-ticks", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--purge-minutes", type=int, default=5)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--random-state", type=int, default=20260526)
    parser.add_argument("--thresholds", default="0,1,2,3")
    return parser.parse_args()


def split_summary(split: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, data in [("train", split.train), ("valid", split.valid), ("test", split.test)]:
        result[name] = {
            "rows": int(len(data)),
            "time_start": str(pd.to_datetime(data["decision_time"]).min()),
            "time_end": str(pd.to_datetime(data["decision_time"]).max()),
            "contract_distribution": {str(k): int(v) for k, v in data["contract"].value_counts().to_dict().items()},
            "long_label_distribution": {str(k): int(v) for k, v in data["long_label"].value_counts().to_dict().items()},
            "short_label_distribution": {str(k): int(v) for k, v in data["short_label"].value_counts().to_dict().items()},
        }
    return result


def save_feature_importance(model: Any, feature_columns: list[str], path: Path) -> None:
    classifier = model.named_steps["classifier"]
    importance = getattr(classifier, "feature_importances_", None)
    if importance is None:
        return
    table = pd.DataFrame({"feature": feature_columns, "importance": importance})
    table.sort_values("importance", ascending=False).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    df = pd.read_csv(args.dataset)
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df = df.sort_values(["decision_time", "contract"]).reset_index(drop=True)

    feature_columns = infer_feature_columns(df)
    split = chronological_split_by_time(
        df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        purge_minutes=args.purge_minutes,
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
        class_weight=args.class_weight,
    )
    _, short_model = train_side_model(
        train=split.train,
        feature_columns=feature_columns,
        target_column="short_label",
        random_state=args.random_state + 1,
        class_weight=args.class_weight,
    )

    base = {
        "dataset": str(Path(args.dataset).resolve()),
        "selection": "one max-volume contract per decision_time",
        "model_name": model_name,
        "feature_count": len(feature_columns),
        "profit_ticks": args.profit_ticks,
        "loss_ticks": args.loss_ticks,
        "split_mode": "time_ratio",
        "class_weight": args.class_weight,
        "split_summary": split_summary(split),
    }

    save_model_bundle(
        model_dir / "fu_main_minute_long_model.joblib",
        model=long_model,
        feature_columns=feature_columns,
        metadata={**base, "side": "long"},
    )
    save_model_bundle(
        model_dir / "fu_main_minute_short_model.joblib",
        model=short_model,
        feature_columns=feature_columns,
        metadata={**base, "side": "short"},
    )
    save_feature_importance(long_model, feature_columns, report_dir / "fu_main_minute_long_feature_importance.csv")
    save_feature_importance(short_model, feature_columns, report_dir / "fu_main_minute_short_feature_importance.csv")

    long_test_proba = predict_proba_frame(long_model, split.test[feature_columns], "long")
    short_test_proba = predict_proba_frame(short_model, split.test[feature_columns], "short")
    long_test_ev = expected_ticks(long_test_proba, "long", args.profit_ticks, args.loss_ticks)
    short_test_ev = expected_ticks(short_test_proba, "short", args.profit_ticks, args.loss_ticks)

    metrics = {
        "base": base,
        "long": {
            "valid": evaluate_side(
                model=long_model,
                data=split.valid,
                feature_columns=feature_columns,
                target_column="long_label",
                prefix="long",
            ),
            "test": evaluate_side(
                model=long_model,
                data=split.test,
                feature_columns=feature_columns,
                target_column="long_label",
                prefix="long",
            ),
        },
        "short": {
            "valid": evaluate_side(
                model=short_model,
                data=split.valid,
                feature_columns=feature_columns,
                target_column="short_label",
                prefix="short",
            ),
            "test": evaluate_side(
                model=short_model,
                data=split.test,
                feature_columns=feature_columns,
                target_column="short_label",
                prefix="short",
            ),
        },
        "ev_thresholds": {
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
        },
    }

    test_predictions = split.test[
        [
            "contract",
            "decision_time",
            "entry_time",
            "entry_price",
            "long_label",
            "short_label",
            "long_exit_ticks",
            "short_exit_ticks",
        ]
    ].copy()
    test_predictions = pd.concat([test_predictions, long_test_proba, short_test_proba], axis=1)
    test_predictions["long_ev_ticks"] = long_test_ev
    test_predictions["short_ev_ticks"] = short_test_ev
    test_predictions.to_csv(report_dir / "fu_main_minute_test_predictions.csv", index=False, encoding="utf-8-sig")

    make_calibration_table(split.test["long_label"], long_test_proba, prefix="long", label=LABEL_WIN).to_csv(
        report_dir / "fu_main_minute_long_win_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_calibration_table(split.test["short_label"], short_test_proba, prefix="short", label=LABEL_WIN).to_csv(
        report_dir / "fu_main_minute_short_win_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_path = report_dir / "fu_main_minute_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model_name: {model_name}")
    print(f"features: {len(feature_columns)}")
    print(f"train/valid/test rows: {len(split.train):,}/{len(split.valid):,}/{len(split.test):,}")
    print(f"metrics: {metrics_path.resolve()}")
    print("test accuracy:", {"long": metrics["long"]["test"]["accuracy"], "short": metrics["short"]["test"]["accuracy"]})


if __name__ == "__main__":
    main()
