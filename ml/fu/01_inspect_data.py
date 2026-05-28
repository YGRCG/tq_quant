from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import load_ohlcv_csv, summarize_time_gaps


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "data" / "1min_FU" / "FU2609.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FU2609 one-minute CSV quality.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input one-minute CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ohlcv_csv(args.input)

    print(f"input: {Path(args.input).resolve()}")
    print(f"rows: {len(df):,}")
    print(f"time_start: {df['datetime'].iloc[0]}")
    print(f"time_end: {df['datetime'].iloc[-1]}")
    print("columns:", ", ".join(df.columns))
    print()

    print("missing values:")
    print(df.isna().sum().to_string())
    print()

    print("price summary:")
    print(df[["open", "high", "low", "close"]].describe().to_string())
    print()

    if "volume" in df.columns:
        zero_volume = int((df["volume"] <= 0).sum())
        print(f"zero_or_negative_volume_rows: {zero_volume:,}")
        print("volume summary:")
        print(df["volume"].describe().to_string())
        print()

    gaps = summarize_time_gaps(df)
    print("time gap counts, first 20:")
    print(gaps.head(20).to_string(index=False))
    print()

    over_one_minute = gaps[gaps["gap"] > "0 days 00:01:00"]["count"].sum() if not gaps.empty else 0
    print(f"gaps_over_one_minute: {int(over_one_minute):,}")


if __name__ == "__main__":
    main()
