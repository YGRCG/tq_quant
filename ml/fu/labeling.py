from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_WIN = "win"
LABEL_LOSS = "loss"
LABEL_NONE = "none"
LABELS = [LABEL_WIN, LABEL_LOSS, LABEL_NONE]


@dataclass(frozen=True)
class BarrierConfig:
    tick_size: float = 1.0
    profit_ticks: float = 10.0
    loss_ticks: float = 5.0
    horizon_minutes: int = 5
    same_bar_policy: str = "loss"

    @property
    def profit_move(self) -> float:
        return self.profit_ticks * self.tick_size

    @property
    def loss_move(self) -> float:
        return self.loss_ticks * self.tick_size


def _choose_same_bar_label(policy: str) -> str:
    if policy == "loss":
        return LABEL_LOSS
    if policy == "win":
        return LABEL_WIN
    raise ValueError("same_bar_policy must be 'loss' or 'win'")


def _label_side(
    *,
    side: str,
    entry_price: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    config: BarrierConfig,
) -> tuple[str, bool, float, int]:
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")

    if side == "long":
        profit_level = entry_price + config.profit_move
        loss_level = entry_price - config.loss_move
    else:
        profit_level = entry_price - config.profit_move
        loss_level = entry_price + config.loss_move

    for offset, (high, low) in enumerate(zip(future_high, future_low), start=1):
        if side == "long":
            hit_profit = high >= profit_level
            hit_loss = low <= loss_level
        else:
            hit_profit = low <= profit_level
            hit_loss = high >= loss_level

        if hit_profit and hit_loss:
            label = _choose_same_bar_label(config.same_bar_policy)
            exit_ticks = config.profit_ticks if label == LABEL_WIN else -config.loss_ticks
            return label, True, exit_ticks, offset
        if hit_profit:
            return LABEL_WIN, False, config.profit_ticks, offset
        if hit_loss:
            return LABEL_LOSS, False, -config.loss_ticks, offset

    if len(future_close) == 0:
        return LABEL_NONE, False, np.nan, 0

    if side == "long":
        exit_ticks = (future_close[-1] - entry_price) / config.tick_size
    else:
        exit_ticks = (entry_price - future_close[-1]) / config.tick_size
    return LABEL_NONE, False, float(exit_ticks), len(future_close)


def make_barrier_labels(df: pd.DataFrame, config: BarrierConfig) -> pd.DataFrame:
    """Create long/short triple-barrier labels for decisions made at each bar close.

    Decision time is row i close. Entry is row i+1 open. The horizon is calendar
    time, not "next N rows", so sparse no-trade minutes do not extend the lookahead.
    """
    if config.same_bar_policy not in {"loss", "win"}:
        raise ValueError("same_bar_policy must be 'loss' or 'win'")

    times = df["datetime"].to_numpy(dtype="datetime64[ns]")
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    horizon_delta = np.timedelta64(config.horizon_minutes, "m")
    records: list[dict[str, object]] = []

    for decision_idx in range(len(df) - 1):
        entry_idx = decision_idx + 1
        entry_time = times[entry_idx]
        horizon_end = entry_time + horizon_delta
        end_idx = int(np.searchsorted(times, horizon_end, side="left"))

        if end_idx <= entry_idx:
            continue

        entry_price = float(opens[entry_idx])
        future_high = highs[entry_idx:end_idx]
        future_low = lows[entry_idx:end_idx]
        future_close = closes[entry_idx:end_idx]

        long_label, long_ambiguous, long_exit_ticks, long_exit_offset = _label_side(
            side="long",
            entry_price=entry_price,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            config=config,
        )
        short_label, short_ambiguous, short_exit_ticks, short_exit_offset = _label_side(
            side="short",
            entry_price=entry_price,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            config=config,
        )

        records.append(
            {
                "decision_idx": decision_idx,
                "decision_time": pd.Timestamp(times[decision_idx]),
                "entry_time": pd.Timestamp(entry_time),
                "horizon_end_time": pd.Timestamp(horizon_end),
                "entry_price": entry_price,
                "bars_observed": int(end_idx - entry_idx),
                "long_label": long_label,
                "short_label": short_label,
                "long_ambiguous": bool(long_ambiguous),
                "short_ambiguous": bool(short_ambiguous),
                "long_exit_ticks": long_exit_ticks,
                "short_exit_ticks": short_exit_ticks,
                "long_exit_bar_offset": int(long_exit_offset),
                "short_exit_bar_offset": int(short_exit_offset),
            }
        )

    labels = pd.DataFrame.from_records(records)
    if labels.empty:
        return labels
    return labels.set_index("decision_idx", drop=False)
