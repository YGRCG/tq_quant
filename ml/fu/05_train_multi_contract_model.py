from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from labeling import LABEL_WIN
from modeling import (
    SplitData,
    apply_isotonic_calibrators,
    chronological_split_by_time,
    evaluate_side,
    evaluate_proba,
    expected_ticks,
    fit_isotonic_calibrators,
    infer_feature_columns,
    make_calibration_table,
    predict_proba_frame,
    save_model_bundle,
    summarize_directional_strategy,
    summarize_thresholds,
    train_side_model,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "output" / "fu_multi_dataset.csv"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "output" / "models_multi"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "output" / "reports_multi"


def parse_contract_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FU multi-contract long/short probability models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset CSV from 04_build_multi_contract_dataset.py.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--profit-ticks", type=float, default=10.0)
    parser.add_argument("--loss-ticks", type=float, default=5.0)
    parser.add_argument("--split-mode", choices=["time_ratio", "contract_holdout"], default="time_ratio")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--purge-minutes", type=int, default=5)
    parser.add_argument("--valid-contracts", default=None, help="For contract_holdout, comma-separated validation contracts.")
    parser.add_argument("--test-contracts", default=None, help="For contract_holdout, comma-separated test contracts.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--random-state", type=int, default=20260526)
    parser.add_argument("--thresholds", default="0,1,2,3")
    parser.add_argument("--drop-any-ambiguous", action="store_true")
    return parser.parse_args()


def split_by_contract_holdout(df: pd.DataFrame, valid_contracts: set[str], test_contracts: set[str]) -> SplitData:
    if not valid_contracts:
        raise ValueError("--valid-contracts is required for contract_holdout")
    if not test_contracts:
        raise ValueError("--test-contracts is required for contract_holdout")
    overlap = valid_contracts & test_contracts
    if overlap:
        raise ValueError(f"Contracts cannot be both validation and test: {sorted(overlap)}")

    contracts = set(df["contract"].astype(str).str.upper().unique())
    unknown = (valid_contracts | test_contracts) - contracts
    if unknown:
        raise ValueError(f"Unknown contracts in dataset: {sorted(unknown)}")

    contract_col = df["contract"].astype(str).str.upper()
    valid = df[contract_col.isin(valid_contracts)].copy()
    test = df[contract_col.isin(test_contracts)].copy()
    train = df[~contract_col.isin(valid_contracts | test_contracts)].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(
            f"Contract split produced an empty partition: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )
    return SplitData(train=train, valid=valid, test=test)


def side_report(side: str, model: Any, data: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    return evaluate_side(
        model=model,
        data=data,
        feature_columns=feature_columns,
        target_column=f"{side}_label",
        prefix=side,
    )


def split_summary(split: SplitData) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, data in [("train", split.train), ("valid", split.valid), ("test", split.test)]:
        summary[name] = {
            "rows": int(len(data)),
            "time_start": str(pd.to_datetime(data["decision_time"]).min()),
            "time_end": str(pd.to_datetime(data["decision_time"]).max()),
            "contracts": sorted(data["contract"].unique().tolist()) if "contract" in data.columns else [],
            "long_label_distribution": {str(k): int(v) for k, v in data["long_label"].value_counts().to_dict().items()},
            "short_label_distribution": {str(k): int(v) for k, v in data["short_label"].value_counts().to_dict().items()},
        }
    return summary


def by_contract_summary(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract, group in data.groupby("contract"):
        rows.append(
            {
                "contract": str(contract),
                "rows": int(len(group)),
                "time_start": str(pd.to_datetime(group["decision_time"]).min()),
                "time_end": str(pd.to_datetime(group["decision_time"]).max()),
                "long_label_distribution": {
                    str(k): int(v) for k, v in group["long_label"].value_counts().to_dict().items()
                },
                "short_label_distribution": {
                    str(k): int(v) for k, v in group["short_label"].value_counts().to_dict().items()
                },
            }
        )
    return rows


def save_feature_importance(model: Any, feature_columns: list[str], path: Path) -> None:
    classifier = model.named_steps["classifier"]
    importance = getattr(classifier, "feature_importances_", None)
    if importance is None:
        return
    table = pd.DataFrame({"feature": feature_columns, "importance": importance})
    table = table.sort_values("importance", ascending=False)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def make_directional_trades(
    data: pd.DataFrame,
    long_ev: pd.Series,
    short_ev: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    choose_long = (long_ev >= threshold) & (long_ev > short_ev)
    choose_short = (short_ev >= threshold) & (short_ev > long_ev)
    base_columns = ["contract", "decision_time"]

    long_trades = data.loc[choose_long, base_columns].copy()
    long_trades["side"] = "long"
    long_trades["label"] = data.loc[choose_long, "long_label"].to_numpy()
    long_trades["actual_ticks"] = data.loc[choose_long, "long_exit_ticks"].to_numpy()
    long_trades["predicted_ev_ticks"] = long_ev.loc[choose_long].to_numpy()

    short_trades = data.loc[choose_short, base_columns].copy()
    short_trades["side"] = "short"
    short_trades["label"] = data.loc[choose_short, "short_label"].to_numpy()
    short_trades["actual_ticks"] = data.loc[choose_short, "short_exit_ticks"].to_numpy()
    short_trades["predicted_ev_ticks"] = short_ev.loc[choose_short].to_numpy()

    trades = pd.concat([long_trades, short_trades], ignore_index=True)
    if trades.empty:
        trades["decision_month"] = pd.Series(dtype=str)
        return trades
    trades["decision_time"] = pd.to_datetime(trades["decision_time"])
    trades["decision_month"] = trades["decision_time"].dt.to_period("M").astype(str)
    return trades


def append_trade_summary(
    rows: list[dict[str, Any]],
    *,
    ev_source: str,
    threshold: float,
    group_type: str,
    group_value: str,
    denominator: int,
    trades: pd.DataFrame,
) -> None:
    count = int(len(trades))
    label_counts = trades["label"].value_counts(normalize=True).to_dict() if count else {}
    side_counts = trades["side"].value_counts().to_dict() if count else {}
    rows.append(
        {
            "ev_source": ev_source,
            "threshold": float(threshold),
            "group_type": group_type,
            "group_value": group_value,
            "base_rows": int(denominator),
            "count": count,
            "coverage": float(count / denominator) if denominator else 0.0,
            "long_count": int(side_counts.get("long", 0)),
            "short_count": int(side_counts.get("short", 0)),
            "avg_predicted_ev_ticks": float(trades["predicted_ev_ticks"].mean()) if count else None,
            "avg_actual_ticks": float(trades["actual_ticks"].mean()) if count else None,
            "total_actual_ticks": float(trades["actual_ticks"].sum()) if count else 0.0,
            "win_rate": float(label_counts.get("win", 0.0)),
            "loss_rate": float(label_counts.get("loss", 0.0)),
            "none_rate": float(label_counts.get("none", 0.0)),
        }
    )


def make_ev_breakdown_table(
    *,
    data: pd.DataFrame,
    long_ev: pd.Series,
    short_ev: pd.Series,
    thresholds: list[float],
    ev_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = data[["contract", "decision_time"]].copy()
    base["decision_time"] = pd.to_datetime(base["decision_time"])
    base["decision_month"] = base["decision_time"].dt.to_period("M").astype(str)
    month_sizes = base.groupby("decision_month", observed=False).size().to_dict()
    contract_sizes = base.groupby("contract", observed=False).size().to_dict()
    month_contract = base.groupby(["decision_month", "contract"], observed=False).size().to_dict()

    for threshold in thresholds:
        trades = make_directional_trades(data, long_ev, short_ev, threshold)
        append_trade_summary(
            rows,
            ev_source=ev_source,
            threshold=threshold,
            group_type="all",
            group_value="all",
            denominator=len(data),
            trades=trades,
        )

        for month, denominator in month_sizes.items():
            append_trade_summary(
                rows,
                ev_source=ev_source,
                threshold=threshold,
                group_type="month",
                group_value=str(month),
                denominator=int(denominator),
                trades=trades[trades["decision_month"] == month],
            )

        for contract, denominator in contract_sizes.items():
            append_trade_summary(
                rows,
                ev_source=ev_source,
                threshold=threshold,
                group_type="contract",
                group_value=str(contract),
                denominator=int(denominator),
                trades=trades[trades["contract"] == contract],
            )

        for side in ["long", "short"]:
            append_trade_summary(
                rows,
                ev_source=ev_source,
                threshold=threshold,
                group_type="side",
                group_value=side,
                denominator=len(data),
                trades=trades[trades["side"] == side],
            )

        for (month, contract), denominator in month_contract.items():
            group_trades = trades[(trades["decision_month"] == month) & (trades["contract"] == contract)]
            append_trade_summary(
                rows,
                ev_source=ev_source,
                threshold=threshold,
                group_type="month_contract",
                group_value=f"{month}|{contract}",
                denominator=int(denominator),
                trades=group_trades,
            )

        for month, denominator in month_sizes.items():
            for side in ["long", "short"]:
                group_trades = trades[(trades["decision_month"] == month) & (trades["side"] == side)]
                append_trade_summary(
                    rows,
                    ev_source=ev_source,
                    threshold=threshold,
                    group_type="month_side",
                    group_value=f"{month}|{side}",
                    denominator=int(denominator),
                    trades=group_trades,
                )

        for contract, denominator in contract_sizes.items():
            for side in ["long", "short"]:
                group_trades = trades[(trades["contract"] == contract) & (trades["side"] == side)]
                append_trade_summary(
                    rows,
                    ev_source=ev_source,
                    threshold=threshold,
                    group_type="contract_side",
                    group_value=f"{contract}|{side}",
                    denominator=int(denominator),
                    trades=group_trades,
                )

    return pd.DataFrame(rows)


def rename_calibrated_proba(proba: pd.DataFrame, side: str) -> pd.DataFrame:
    return proba.rename(columns={f"{side}_p_{label}": f"{side}_cal_p_{label}" for label in ["win", "loss", "none"]})


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    dataset_path = Path(args.dataset)
    df = pd.read_csv(dataset_path)
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df = df.sort_values(["decision_time", "contract"]).reset_index(drop=True)

    if args.drop_any_ambiguous:
        df = df[~(df["long_ambiguous"].astype(bool) | df["short_ambiguous"].astype(bool))].copy()
        df = df.reset_index(drop=True)

    feature_columns = infer_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("No numeric feature columns were found.")

    if args.split_mode == "time_ratio":
        split = chronological_split_by_time(
            df,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            purge_minutes=args.purge_minutes,
        )
    else:
        split = split_by_contract_holdout(
            df,
            valid_contracts=parse_contract_list(args.valid_contracts),
            test_contracts=parse_contract_list(args.test_contracts),
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

    base_metadata = {
        "dataset": str(dataset_path.resolve()),
        "model_name": model_name,
        "feature_count": len(feature_columns),
        "profit_ticks": args.profit_ticks,
        "loss_ticks": args.loss_ticks,
        "split_mode": args.split_mode,
        "class_weight": args.class_weight,
        "drop_any_ambiguous": bool(args.drop_any_ambiguous),
        "split_summary": split_summary(split),
    }

    save_model_bundle(
        model_dir / "fu_multi_long_model.joblib",
        model=long_model,
        feature_columns=feature_columns,
        metadata={**base_metadata, "side": "long"},
    )
    save_model_bundle(
        model_dir / "fu_multi_short_model.joblib",
        model=short_model,
        feature_columns=feature_columns,
        metadata={**base_metadata, "side": "short"},
    )
    save_feature_importance(long_model, feature_columns, report_dir / "fu_multi_long_feature_importance.csv")
    save_feature_importance(short_model, feature_columns, report_dir / "fu_multi_short_feature_importance.csv")

    metrics: dict[str, Any] = {
        "base": base_metadata,
        "test_by_contract": by_contract_summary(split.test),
        "long": {
            "valid": side_report("long", long_model, split.valid, feature_columns),
            "test": side_report("long", long_model, split.test, feature_columns),
        },
        "short": {
            "valid": side_report("short", short_model, split.valid, feature_columns),
            "test": side_report("short", short_model, split.test, feature_columns),
        },
    }

    long_test_proba = predict_proba_frame(long_model, split.test[feature_columns], "long")
    short_test_proba = predict_proba_frame(short_model, split.test[feature_columns], "short")
    long_valid_proba = predict_proba_frame(long_model, split.valid[feature_columns], "long")
    short_valid_proba = predict_proba_frame(short_model, split.valid[feature_columns], "short")

    long_calibrators = fit_isotonic_calibrators(
        y_true=split.valid["long_label"],
        proba=long_valid_proba,
        prefix="long",
    )
    short_calibrators = fit_isotonic_calibrators(
        y_true=split.valid["short_label"],
        proba=short_valid_proba,
        prefix="short",
    )
    long_valid_cal_proba = apply_isotonic_calibrators(
        proba=long_valid_proba,
        prefix="long",
        calibrators=long_calibrators,
    )
    short_valid_cal_proba = apply_isotonic_calibrators(
        proba=short_valid_proba,
        prefix="short",
        calibrators=short_calibrators,
    )
    long_test_cal_proba = apply_isotonic_calibrators(
        proba=long_test_proba,
        prefix="long",
        calibrators=long_calibrators,
    )
    short_test_cal_proba = apply_isotonic_calibrators(
        proba=short_test_proba,
        prefix="short",
        calibrators=short_calibrators,
    )

    long_test_ev = expected_ticks(long_test_proba, "long", args.profit_ticks, args.loss_ticks)
    short_test_ev = expected_ticks(short_test_proba, "short", args.profit_ticks, args.loss_ticks)
    long_test_cal_ev = expected_ticks(long_test_cal_proba, "long", args.profit_ticks, args.loss_ticks)
    short_test_cal_ev = expected_ticks(short_test_cal_proba, "short", args.profit_ticks, args.loss_ticks)

    joblib.dump(
        {
            "long": long_calibrators,
            "short": short_calibrators,
            "metadata": {**base_metadata, "calibration": "one_vs_rest_isotonic_on_validation"},
        },
        model_dir / "fu_multi_probability_calibrators.joblib",
    )

    test_columns = [
        "contract",
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
    test_predictions = split.test[test_columns].copy()
    test_predictions = pd.concat([test_predictions, long_test_proba, short_test_proba], axis=1)
    test_predictions = pd.concat(
        [
            test_predictions,
            rename_calibrated_proba(long_test_cal_proba, "long"),
            rename_calibrated_proba(short_test_cal_proba, "short"),
        ],
        axis=1,
    )
    test_predictions["long_ev_ticks"] = long_test_ev
    test_predictions["short_ev_ticks"] = short_test_ev
    test_predictions["long_cal_ev_ticks"] = long_test_cal_ev
    test_predictions["short_cal_ev_ticks"] = short_test_cal_ev
    test_predictions.to_csv(report_dir / "fu_multi_test_predictions.csv", index=False, encoding="utf-8-sig")

    make_calibration_table(split.test["long_label"], long_test_proba, prefix="long", label=LABEL_WIN).to_csv(
        report_dir / "fu_multi_long_win_calibration.csv", index=False, encoding="utf-8-sig"
    )
    make_calibration_table(split.test["short_label"], short_test_proba, prefix="short", label=LABEL_WIN).to_csv(
        report_dir / "fu_multi_short_win_calibration.csv", index=False, encoding="utf-8-sig"
    )
    make_calibration_table(split.test["long_label"], long_test_cal_proba, prefix="long", label=LABEL_WIN).to_csv(
        report_dir / "fu_multi_long_win_calibration_calibrated.csv", index=False, encoding="utf-8-sig"
    )
    make_calibration_table(split.test["short_label"], short_test_cal_proba, prefix="short", label=LABEL_WIN).to_csv(
        report_dir / "fu_multi_short_win_calibration_calibrated.csv", index=False, encoding="utf-8-sig"
    )

    raw_breakdown = make_ev_breakdown_table(
        data=split.test,
        long_ev=long_test_ev,
        short_ev=short_test_ev,
        thresholds=thresholds,
        ev_source="raw",
    )
    calibrated_breakdown = make_ev_breakdown_table(
        data=split.test,
        long_ev=long_test_cal_ev,
        short_ev=short_test_cal_ev,
        thresholds=thresholds,
        ev_source="calibrated",
    )
    raw_breakdown.to_csv(report_dir / "fu_multi_raw_ev_breakdown.csv", index=False, encoding="utf-8-sig")
    calibrated_breakdown.to_csv(
        report_dir / "fu_multi_calibrated_ev_breakdown.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat([raw_breakdown, calibrated_breakdown], ignore_index=True).to_csv(
        report_dir / "fu_multi_ev_breakdown.csv",
        index=False,
        encoding="utf-8-sig",
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
    metrics["calibration"] = {
        "method": "one_vs_rest_isotonic_on_validation",
        "long": {
            "valid_raw": evaluate_proba(y_true=split.valid["long_label"], proba=long_valid_proba, prefix="long"),
            "valid_calibrated": evaluate_proba(
                y_true=split.valid["long_label"],
                proba=long_valid_cal_proba,
                prefix="long",
            ),
            "test_raw": evaluate_proba(y_true=split.test["long_label"], proba=long_test_proba, prefix="long"),
            "test_calibrated": evaluate_proba(
                y_true=split.test["long_label"],
                proba=long_test_cal_proba,
                prefix="long",
            ),
        },
        "short": {
            "valid_raw": evaluate_proba(y_true=split.valid["short_label"], proba=short_valid_proba, prefix="short"),
            "valid_calibrated": evaluate_proba(
                y_true=split.valid["short_label"],
                proba=short_valid_cal_proba,
                prefix="short",
            ),
            "test_raw": evaluate_proba(y_true=split.test["short_label"], proba=short_test_proba, prefix="short"),
            "test_calibrated": evaluate_proba(
                y_true=split.test["short_label"],
                proba=short_test_cal_proba,
                prefix="short",
            ),
        },
    }
    metrics["calibrated_ev_thresholds"] = {
        "long": summarize_thresholds(
            data=split.test,
            ev=long_test_cal_ev,
            actual_ticks_col="long_exit_ticks",
            label_col="long_label",
            thresholds=thresholds,
        ),
        "short": summarize_thresholds(
            data=split.test,
            ev=short_test_cal_ev,
            actual_ticks_col="short_exit_ticks",
            label_col="short_label",
            thresholds=thresholds,
        ),
        "directional": summarize_directional_strategy(
            data=split.test,
            long_ev=long_test_cal_ev,
            short_ev=short_test_cal_ev,
            thresholds=thresholds,
        ),
    }

    metrics_path = report_dir / "fu_multi_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model_name: {model_name}")
    print(f"class_weight: {args.class_weight}")
    print(f"features: {len(feature_columns)}")
    print(
        "train/valid/test rows:",
        f"{len(split.train):,}/{len(split.valid):,}/{len(split.test):,}",
    )
    print(f"long_model: {(model_dir / 'fu_multi_long_model.joblib').resolve()}")
    print(f"short_model: {(model_dir / 'fu_multi_short_model.joblib').resolve()}")
    print(f"metrics: {metrics_path.resolve()}")
    print(f"test_predictions: {(report_dir / 'fu_multi_test_predictions.csv').resolve()}")
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


if __name__ == "__main__":
    main()
