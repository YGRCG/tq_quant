from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from labeling import LABEL_LOSS, LABEL_NONE, LABEL_WIN, LABELS


NON_FEATURE_COLUMNS = {
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


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return columns


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    purge_bars: int = 5,
) -> SplitData:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")

    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    gap = max(int(purge_bars), 0)

    train = df.iloc[: max(train_end - gap, 0)].copy()
    valid = df.iloc[min(train_end + gap, n) : max(valid_end - gap, train_end + gap)].copy()
    test = df.iloc[min(valid_end + gap, n) :].copy()

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(
            f"Split produced an empty partition: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )
    return SplitData(train=train, valid=valid, test=test)


def chronological_split_by_time(
    df: pd.DataFrame,
    *,
    time_column: str = "decision_time",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    purge_minutes: int = 5,
) -> SplitData:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")
    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")

    ordered = df.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column])
    ordered = ordered.sort_values([time_column, "contract"] if "contract" in ordered.columns else [time_column])

    unique_times = pd.Series(ordered[time_column].dropna().unique()).sort_values().reset_index(drop=True)
    if len(unique_times) < 10:
        raise ValueError(f"Not enough unique timestamps to split: {len(unique_times)}")

    train_cut = unique_times.iloc[int(len(unique_times) * train_ratio)]
    valid_cut = unique_times.iloc[int(len(unique_times) * (train_ratio + valid_ratio))]
    purge_delta = pd.Timedelta(minutes=max(int(purge_minutes), 0))

    train = ordered[ordered[time_column] < train_cut - purge_delta].copy()
    valid = ordered[
        (ordered[time_column] >= train_cut + purge_delta) & (ordered[time_column] < valid_cut - purge_delta)
    ].copy()
    test = ordered[ordered[time_column] >= valid_cut + purge_delta].copy()

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(
            f"Time split produced an empty partition: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )
    return SplitData(train=train, valid=valid, test=test)


def make_classifier(random_state: int = 20260526, class_weight: str | None = "balanced") -> tuple[str, Pipeline]:
    if class_weight == "none":
        class_weight = None

    try:
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=60,
            reg_alpha=0.1,
            reg_lambda=0.3,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        model_name = "lightgbm"
    except ImportError:
        rf_class_weight = "balanced_subsample" if class_weight == "balanced" else class_weight
        classifier = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=50,
            class_weight=rf_class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        model_name = "random_forest"

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )
    return model_name, pipeline


def train_side_model(
    *,
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int,
    class_weight: str | None = "balanced",
) -> tuple[str, Pipeline]:
    model_name, pipeline = make_classifier(random_state=random_state, class_weight=class_weight)
    pipeline.fit(train[feature_columns], train[target_column])
    return model_name, pipeline


def predict_proba_frame(model: Pipeline, x: pd.DataFrame, prefix: str) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        proba = model.predict_proba(x)
    result = pd.DataFrame(proba, columns=[f"{prefix}_p_{cls}" for cls in classes], index=x.index)
    for label in LABELS:
        col = f"{prefix}_p_{label}"
        if col not in result.columns:
            result[col] = 0.0
    return result[[f"{prefix}_p_{LABEL_WIN}", f"{prefix}_p_{LABEL_LOSS}", f"{prefix}_p_{LABEL_NONE}"]]


def multiclass_brier_score(y_true: pd.Series, proba: pd.DataFrame, prefix: str) -> float:
    probabilities = proba[[f"{prefix}_p_{label}" for label in LABELS]].to_numpy()
    truth = pd.get_dummies(pd.Categorical(y_true, categories=LABELS)).to_numpy()
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def ordered_log_loss_score(y_true: pd.Series, proba: pd.DataFrame, prefix: str) -> float:
    probabilities = proba[[f"{prefix}_p_{label}" for label in LABELS]].to_numpy()
    probabilities = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    truth = pd.get_dummies(pd.Categorical(y_true, categories=LABELS)).to_numpy()
    return float(-np.mean(np.sum(truth * np.log(probabilities), axis=1)))


def evaluate_proba(
    *,
    y_true: pd.Series,
    proba: pd.DataFrame,
    prefix: str,
) -> dict[str, Any]:
    return {
        "rows": int(len(y_true)),
        "brier": multiclass_brier_score(y_true, proba, prefix),
        "log_loss": ordered_log_loss_score(y_true, proba, prefix),
        "label_distribution": {str(k): int(v) for k, v in y_true.value_counts().to_dict().items()},
    }


def evaluate_side(
    *,
    model: Pipeline,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    prefix: str,
) -> dict[str, Any]:
    x = data[feature_columns]
    y = data[target_column]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        pred = model.predict(x)
    proba = predict_proba_frame(model, x, prefix)

    metrics: dict[str, Any] = {
        "rows": int(len(data)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": multiclass_brier_score(y, proba, prefix),
        "log_loss": ordered_log_loss_score(y, proba, prefix),
        "label_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }

    return metrics


def make_calibration_table(
    y_true: pd.Series,
    proba: pd.DataFrame,
    *,
    prefix: str,
    label: str = LABEL_WIN,
    bins: int = 10,
) -> pd.DataFrame:
    p_col = f"{prefix}_p_{label}"
    table = pd.DataFrame({"prob": proba[p_col], "hit": (y_true == label).astype(int)})
    table["bin"] = pd.cut(table["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = table.groupby("bin", observed=False)
    return grouped.agg(count=("hit", "size"), mean_prob=("prob", "mean"), actual_rate=("hit", "mean")).reset_index()


def fit_isotonic_calibrators(
    *,
    y_true: pd.Series,
    proba: pd.DataFrame,
    prefix: str,
) -> dict[str, IsotonicRegression]:
    calibrators: dict[str, IsotonicRegression] = {}
    for label in LABELS:
        col = f"{prefix}_p_{label}"
        target = (y_true == label).astype(int).to_numpy()
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(proba[col].to_numpy(), target)
        calibrators[label] = calibrator
    return calibrators


def apply_isotonic_calibrators(
    *,
    proba: pd.DataFrame,
    prefix: str,
    calibrators: dict[str, IsotonicRegression],
) -> pd.DataFrame:
    result = pd.DataFrame(index=proba.index)
    for label in LABELS:
        col = f"{prefix}_p_{label}"
        calibrated = calibrators[label].predict(proba[col].to_numpy())
        result[col] = np.clip(calibrated, 0.0, 1.0)

    total = result.sum(axis=1).replace(0, np.nan)
    result = result.div(total, axis=0).fillna(1.0 / len(LABELS))
    return result[[f"{prefix}_p_{LABEL_WIN}", f"{prefix}_p_{LABEL_LOSS}", f"{prefix}_p_{LABEL_NONE}"]]


def expected_ticks(proba: pd.DataFrame, prefix: str, profit_ticks: float, loss_ticks: float) -> pd.Series:
    return proba[f"{prefix}_p_{LABEL_WIN}"] * profit_ticks - proba[f"{prefix}_p_{LABEL_LOSS}"] * loss_ticks


def summarize_thresholds(
    *,
    data: pd.DataFrame,
    ev: pd.Series,
    actual_ticks_col: str,
    label_col: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        mask = ev >= threshold
        selected = data.loc[mask]
        if selected.empty:
            summaries.append(
                {
                    "threshold": float(threshold),
                    "count": 0,
                    "coverage": 0.0,
                    "avg_actual_ticks": None,
                    "total_actual_ticks": 0.0,
                }
            )
            continue

        label_counts = selected[label_col].value_counts(normalize=True).to_dict()
        summaries.append(
            {
                "threshold": float(threshold),
                "count": int(len(selected)),
                "coverage": float(len(selected) / len(data)),
                "avg_predicted_ev_ticks": float(ev.loc[mask].mean()),
                "avg_actual_ticks": float(selected[actual_ticks_col].mean()),
                "total_actual_ticks": float(selected[actual_ticks_col].sum()),
                "win_rate": float(label_counts.get(LABEL_WIN, 0.0)),
                "loss_rate": float(label_counts.get(LABEL_LOSS, 0.0)),
                "none_rate": float(label_counts.get(LABEL_NONE, 0.0)),
            }
        )
    return summaries


def summarize_directional_strategy(
    *,
    data: pd.DataFrame,
    long_ev: pd.Series,
    short_ev: pd.Series,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        choose_long = (long_ev >= threshold) & (long_ev > short_ev)
        choose_short = (short_ev >= threshold) & (short_ev > long_ev)
        selected = choose_long | choose_short

        if not selected.any():
            summaries.append(
                {
                    "threshold": float(threshold),
                    "count": 0,
                    "coverage": 0.0,
                    "avg_actual_ticks": None,
                    "total_actual_ticks": 0.0,
                }
            )
            continue

        actual_ticks = pd.Series(index=data.index, dtype=float)
        actual_ticks.loc[choose_long] = data.loc[choose_long, "long_exit_ticks"]
        actual_ticks.loc[choose_short] = data.loc[choose_short, "short_exit_ticks"]
        chosen_label = pd.Series(index=data.index, dtype=object)
        chosen_label.loc[choose_long] = data.loc[choose_long, "long_label"]
        chosen_label.loc[choose_short] = data.loc[choose_short, "short_label"]
        label_counts = chosen_label.loc[selected].value_counts(normalize=True).to_dict()

        summaries.append(
            {
                "threshold": float(threshold),
                "count": int(selected.sum()),
                "coverage": float(selected.mean()),
                "long_count": int(choose_long.sum()),
                "short_count": int(choose_short.sum()),
                "avg_actual_ticks": float(actual_ticks.loc[selected].mean()),
                "total_actual_ticks": float(actual_ticks.loc[selected].sum()),
                "win_rate": float(label_counts.get(LABEL_WIN, 0.0)),
                "loss_rate": float(label_counts.get(LABEL_LOSS, 0.0)),
                "none_rate": float(label_counts.get(LABEL_NONE, 0.0)),
            }
        )
    return summaries


def save_model_bundle(path: str | Path, *, model: Pipeline, feature_columns: list[str], metadata: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns, "metadata": metadata}, path)
