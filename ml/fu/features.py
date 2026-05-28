from __future__ import annotations

import numpy as np
import pandas as pd


LOOKBACK_WINDOWS = [3, 5, 10, 20, 60]
RETURN_WINDOWS = [1, 3, 5, 10, 20, 60]
AGG_WINDOWS = [3, 5, 15]
BREAKOUT_WINDOWS = [20, 60]
RANK_WINDOW = 240


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rolling_rank(series: pd.Series, window: int = RANK_WINDOW) -> pd.Series:
    min_periods = max(20, window // 4)
    return series.rolling(window, min_periods=min_periods).rank(pct=True)


def build_features(
    df: pd.DataFrame,
    tick_size: float = 1.0,
    profit_ticks: float = 10.0,
    loss_ticks: float = 5.0,
) -> pd.DataFrame:
    """Build point-in-time features available at each bar close."""
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)
    amount = df["amount"] if "amount" in df.columns else pd.Series(0, index=df.index)
    open_interest = df["open_interest"] if "open_interest" in df.columns else pd.Series(0, index=df.index)

    features = pd.DataFrame(index=df.index)

    bar_range = (high - low) / tick_size
    body = (close - open_) / tick_size
    upper_shadow = (high - np.maximum(open_, close)) / tick_size
    lower_shadow = (np.minimum(open_, close) - low) / tick_size

    features["bar_range_ticks"] = bar_range
    features["body_ticks"] = body
    features["abs_body_ticks"] = body.abs()
    features["upper_shadow_ticks"] = upper_shadow
    features["lower_shadow_ticks"] = lower_shadow
    features["close_location"] = _safe_ratio(close - low, high - low).fillna(0.5)

    dt = df["datetime"]
    trade_date = dt.dt.normalize() + pd.to_timedelta((dt.dt.hour >= 21).astype(int), unit="D")
    session_code = pd.Series(3, index=df.index)
    session_code[(dt.dt.hour >= 21) | (dt.dt.hour < 3)] = 0
    session_code[(dt.dt.hour >= 9) & (dt.dt.hour < 12)] = 1
    session_code[(dt.dt.hour >= 13) & (dt.dt.hour < 16)] = 2

    day_high_so_far = high.groupby(trade_date).cummax()
    day_low_so_far = low.groupby(trade_date).cummin()
    day_open = open_.groupby(trade_date).transform("first")
    session_high_so_far = high.groupby([trade_date, session_code]).cummax()
    session_low_so_far = low.groupby([trade_date, session_code]).cummin()
    session_open = open_.groupby([trade_date, session_code]).transform("first")

    features["day_range_so_far_ticks"] = (day_high_so_far - day_low_so_far) / tick_size
    features["dist_to_day_high_ticks"] = (day_high_so_far - close) / tick_size
    features["dist_to_day_low_ticks"] = (close - day_low_so_far) / tick_size
    features["day_position"] = _safe_ratio(close - day_low_so_far, day_high_so_far - day_low_so_far).fillna(0.5)
    features["ret_from_day_open_ticks"] = (close - day_open) / tick_size
    features["ret_from_day_open_pct"] = _safe_ratio(close - day_open, day_open)
    features["session_range_so_far_ticks"] = (session_high_so_far - session_low_so_far) / tick_size
    features["dist_to_session_high_ticks"] = (session_high_so_far - close) / tick_size
    features["dist_to_session_low_ticks"] = (close - session_low_so_far) / tick_size
    features["session_position"] = _safe_ratio(close - session_low_so_far, session_high_so_far - session_low_so_far).fillna(0.5)
    features["ret_from_session_open_ticks"] = (close - session_open) / tick_size
    features["ret_from_session_open_pct"] = _safe_ratio(close - session_open, session_open)

    for window in RETURN_WINDOWS:
        features[f"ret_{window}_ticks"] = (close - close.shift(window)) / tick_size
        features[f"ret_{window}_pct"] = close.pct_change(window)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    one_bar_return = close.pct_change()
    for window in LOOKBACK_WINDOWS:
        min_periods = max(2, window // 2)
        rolling_high = high.rolling(window, min_periods=min_periods).max()
        rolling_low = low.rolling(window, min_periods=min_periods).min()
        rolling_close_mean = close.rolling(window, min_periods=min_periods).mean()
        rolling_volume_mean = volume.rolling(window, min_periods=min_periods).mean()

        features[f"atr_{window}_ticks"] = true_range.rolling(window, min_periods=min_periods).mean() / tick_size
        features[f"range_mean_{window}_ticks"] = bar_range.rolling(window, min_periods=min_periods).mean()
        features[f"ret_std_{window}"] = one_bar_return.rolling(window, min_periods=min_periods).std()
        features[f"ma_dist_{window}_ticks"] = (close - rolling_close_mean) / tick_size
        features[f"volume_ma_ratio_{window}"] = _safe_ratio(volume, rolling_volume_mean)
        features[f"dist_to_high_{window}_ticks"] = (rolling_high - close) / tick_size
        features[f"dist_to_low_{window}_ticks"] = (close - rolling_low) / tick_size
        features[f"channel_pos_{window}"] = _safe_ratio(close - rolling_low, rolling_high - rolling_low).fillna(0.5)

    up_bar = (close > open_).astype(int)
    down_bar = (close < open_).astype(int)
    flat_bar = (close == open_).astype(int)
    higher_high = (high > high.shift(1)).astype(int)
    lower_low = (low < low.shift(1)).astype(int)
    higher_low = (low > low.shift(1)).astype(int)
    lower_high = (high < high.shift(1)).astype(int)
    for window in AGG_WINDOWS:
        min_periods = max(2, window // 2)
        agg_open = open_.shift(window - 1)
        agg_high = high.rolling(window, min_periods=min_periods).max()
        agg_low = low.rolling(window, min_periods=min_periods).min()
        agg_body = close - agg_open
        agg_range = agg_high - agg_low
        agg_upper_shadow = agg_high - np.maximum(agg_open, close)
        agg_lower_shadow = np.minimum(agg_open, close) - agg_low

        features[f"agg_{window}_return_ticks"] = agg_body / tick_size
        features[f"agg_{window}_return_pct"] = _safe_ratio(agg_body, agg_open)
        features[f"agg_{window}_range_ticks"] = agg_range / tick_size
        features[f"agg_{window}_upper_shadow_ticks"] = agg_upper_shadow / tick_size
        features[f"agg_{window}_lower_shadow_ticks"] = agg_lower_shadow / tick_size
        features[f"agg_{window}_close_position"] = _safe_ratio(close - agg_low, agg_range).fillna(0.5)
        features[f"agg_{window}_volume_sum"] = volume.rolling(window, min_periods=min_periods).sum()
        features[f"agg_{window}_amount_sum"] = amount.rolling(window, min_periods=min_periods).sum()
        features[f"agg_{window}_oi_change"] = open_interest - open_interest.shift(window)
        features[f"up_bar_count_{window}"] = up_bar.rolling(window, min_periods=min_periods).sum()
        features[f"down_bar_count_{window}"] = down_bar.rolling(window, min_periods=min_periods).sum()
        features[f"flat_bar_count_{window}"] = flat_bar.rolling(window, min_periods=min_periods).sum()
        features[f"higher_high_count_{window}"] = higher_high.rolling(window, min_periods=min_periods).sum()
        features[f"lower_low_count_{window}"] = lower_low.rolling(window, min_periods=min_periods).sum()
        features[f"higher_low_count_{window}"] = higher_low.rolling(window, min_periods=min_periods).sum()
        features[f"lower_high_count_{window}"] = lower_high.rolling(window, min_periods=min_periods).sum()
        features = features.copy()

    for window in BREAKOUT_WINDOWS:
        min_periods = max(5, window // 2)
        prev_high = high.shift(1).rolling(window, min_periods=min_periods).max()
        prev_low = low.shift(1).rolling(window, min_periods=min_periods).min()
        prev_range = prev_high - prev_low

        features[f"break_high_{window}"] = (close > prev_high).astype(int)
        features[f"break_low_{window}"] = (close < prev_low).astype(int)
        features[f"intrabar_break_high_{window}"] = (high > prev_high).astype(int)
        features[f"intrabar_break_low_{window}"] = (low < prev_low).astype(int)
        features[f"high_break_close_inside_{window}"] = ((high > prev_high) & (close <= prev_high)).astype(int)
        features[f"low_break_close_inside_{window}"] = ((low < prev_low) & (close >= prev_low)).astype(int)
        features[f"close_inside_prev_range_{window}"] = ((close <= prev_high) & (close >= prev_low)).astype(int)
        features[f"dist_to_prev_high_{window}_ticks"] = (prev_high - close) / tick_size
        features[f"dist_to_prev_low_{window}_ticks"] = (close - prev_low) / tick_size
        features[f"prev_range_{window}_ticks"] = prev_range / tick_size
        features[f"prev_channel_pos_{window}"] = _safe_ratio(close - prev_low, prev_range).fillna(0.5)
    features = features.copy()

    ma_5 = close.rolling(5, min_periods=3).mean()
    ma_10 = close.rolling(10, min_periods=5).mean()
    ma_20 = close.rolling(20, min_periods=10).mean()
    ma_60 = close.rolling(60, min_periods=30).mean()
    features["ma_5_20_spread_ticks"] = (ma_5 - ma_20) / tick_size
    features["ma_10_60_spread_ticks"] = (ma_10 - ma_60) / tick_size
    features["ma_5_slope_3_ticks"] = (ma_5 - ma_5.shift(3)) / tick_size
    features["ma_20_slope_5_ticks"] = (ma_20 - ma_20.shift(5)) / tick_size

    features["log_volume"] = np.log1p(volume.clip(lower=0))
    features["log_amount"] = np.log1p(amount.clip(lower=0))
    features["volume_change_1"] = volume.diff(1)
    features["volume_change_5"] = volume.diff(5)
    volume_ma20 = volume.rolling(20, min_periods=10).mean()
    amount_ma20 = amount.rolling(20, min_periods=10).mean()
    features["volume_ma20_rank_240"] = _rolling_rank(volume_ma20)
    features["volume_rank_240"] = _rolling_rank(volume)
    features["amount_rank_240"] = _rolling_rank(amount)
    features["amount_ma20_rank_240"] = _rolling_rank(amount_ma20)
    features["oi_change_1"] = open_interest.diff(1)
    features["oi_change_5"] = open_interest.diff(5)
    features["oi_change_20"] = open_interest.diff(20)

    features["atr_20_rank_240"] = _rolling_rank(features["atr_20_ticks"])
    features["atr_60_rank_240"] = _rolling_rank(features["atr_60_ticks"])
    features["range_mean_20_rank_240"] = _rolling_rank(features["range_mean_20_ticks"])
    features["ret_std_20_rank_240"] = _rolling_rank(features["ret_std_20"])
    features["atr_20_to_profit_ratio"] = features["atr_20_ticks"] / profit_ticks
    features["atr_20_to_loss_ratio"] = features["atr_20_ticks"] / loss_ticks
    features["atr_60_to_profit_ratio"] = features["atr_60_ticks"] / profit_ticks
    features["atr_60_to_loss_ratio"] = features["atr_60_ticks"] / loss_ticks

    for window in [5, 20]:
        price_change = close - close.shift(window)
        oi_change = open_interest - open_interest.shift(window)
        volume_change = volume_ma20 - volume_ma20.shift(window)
        price_up = price_change > 0
        price_down = price_change < 0
        oi_up = oi_change > 0
        oi_down = oi_change < 0
        volume_up = volume_change > 0
        volume_down = volume_change < 0

        features[f"price_up_oi_up_{window}"] = (price_up & oi_up).astype(int)
        features[f"price_up_oi_down_{window}"] = (price_up & oi_down).astype(int)
        features[f"price_down_oi_up_{window}"] = (price_down & oi_up).astype(int)
        features[f"price_down_oi_down_{window}"] = (price_down & oi_down).astype(int)
        features[f"price_up_volume_up_{window}"] = (price_up & volume_up).astype(int)
        features[f"price_up_volume_down_{window}"] = (price_up & volume_down).astype(int)
        features[f"price_down_volume_up_{window}"] = (price_down & volume_up).astype(int)
        features[f"price_down_volume_down_{window}"] = (price_down & volume_down).astype(int)

    features["ret_5_volume_interaction"] = features["ret_5_ticks"] * features["volume_ma_ratio_20"]
    features["ret_5_oi_interaction"] = features["ret_5_ticks"] * features["oi_change_5"]
    features["ret_20_volume_interaction"] = features["ret_20_ticks"] * features["volume_ma_ratio_20"]
    features["ret_20_oi_interaction"] = features["ret_20_ticks"] * features["oi_change_20"]

    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    features["hour"] = dt.dt.hour
    features["minute"] = dt.dt.minute
    features["day_of_week"] = dt.dt.dayofweek
    features["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["is_night_session"] = ((dt.dt.hour >= 21) | (dt.dt.hour < 3)).astype(int)
    features["is_morning_session"] = ((dt.dt.hour >= 9) & (dt.dt.hour < 12)).astype(int)
    features["is_afternoon_session"] = ((dt.dt.hour >= 13) & (dt.dt.hour < 16)).astype(int)

    features = features.replace([np.inf, -np.inf], np.nan)
    features.columns = [str(col) for col in features.columns]
    return features
