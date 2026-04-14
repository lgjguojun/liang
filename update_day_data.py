from __future__ import annotations

import argparse
import re
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import akshare as ak
import pandas as pd

DAY_RECORD_STRUCT = struct.Struct("<IIIIIfII")
PRICE_SCALE = 100
RESERVED_VALUE = 65536
SYMBOL_PATTERN = re.compile(r"^(sh|sz|bj)(\d{6})$", re.IGNORECASE)


@dataclass(frozen=True)
class UpdateResult:
    symbol: str
    rows: int
    start_date: str
    end_date: str
    file_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh data/*.day files with the latest daily A-share data."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory that contains .day files.",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=2,
        help="Keep only the most recent N calendar months relative to each symbol's latest trading day.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional symbol list such as sh600012 sz000429. Defaults to all existing .day files.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count per symbol when the upstream request fails.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Pause seconds between successful symbol updates.",
    )
    return parser.parse_args()


def list_symbols(data_dir: Path, explicit_symbols: list[str] | None) -> list[str]:
    if explicit_symbols:
        symbols = sorted({symbol.strip().lower() for symbol in explicit_symbols if symbol.strip()})
    else:
        symbols = sorted(path.stem.lower() for path in data_dir.glob("*.day"))

    invalid = [symbol for symbol in symbols if not SYMBOL_PATTERN.fullmatch(symbol)]
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unsupported symbol format: {joined}")
    return symbols


def fetch_history(symbol: str, months: int) -> pd.DataFrame:
    match = SYMBOL_PATTERN.fullmatch(symbol)
    if not match:
        raise ValueError(f"Unsupported symbol format: {symbol}")

    today = pd.Timestamp.today().normalize()
    start_date = (today - pd.DateOffset(months=months + 2)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    raw_df = ak.stock_zh_a_daily(
        symbol=symbol.lower(),
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )
    if raw_df.empty:
        raise ValueError("Upstream returned no rows.")

    required_columns = ["date", "open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in required_columns if column not in raw_df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Upstream data is missing columns: {joined}")

    df = raw_df[required_columns].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume", "amount"])
    if df.empty:
        raise ValueError("No valid rows remained after cleanup.")

    latest_date = df["date"].max()
    cutoff_date = latest_date - pd.DateOffset(months=months)
    df = df.loc[df["date"] >= cutoff_date].copy()
    if df.empty:
        raise ValueError("No rows remained after applying the calendar-month window.")

    df["volume"] = df["volume"].round().astype(int)
    df["amount"] = df["amount"].astype(float)
    df = df[["date", "open", "high", "low", "close", "amount", "volume"]]
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def price_to_int(value: float) -> int:
    return int(round(float(value) * PRICE_SCALE))


def write_day_file(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as handle:
            temp_file = Path(handle.name)
            for row in df.itertuples(index=False):
                handle.write(
                    DAY_RECORD_STRUCT.pack(
                        int(row.date.strftime("%Y%m%d")),
                        price_to_int(row.open),
                        price_to_int(row.high),
                        price_to_int(row.low),
                        price_to_int(row.close),
                        float(row.amount),
                        int(row.volume),
                        RESERVED_VALUE,
                    )
                )
        temp_file.replace(path)
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)


def update_symbol(data_dir: Path, symbol: str, months: int, retries: int) -> UpdateResult:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = fetch_history(symbol, months=months)
            file_path = data_dir / f"{symbol}.day"
            write_day_file(file_path, df)
            return UpdateResult(
                symbol=symbol,
                rows=len(df),
                start_date=df.iloc[0]["date"].strftime("%Y-%m-%d"),
                end_date=df.iloc[-1]["date"].strftime("%Y-%m-%d"),
                file_path=file_path,
            )
        except Exception as exc:  # pragma: no cover - network and upstream errors vary.
            last_error = exc
            if attempt < retries:
                time.sleep(min(1.5 * attempt, 5.0))
    assert last_error is not None
    raise RuntimeError(f"{symbol} update failed after {retries} attempts: {last_error}") from last_error


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    symbols = list_symbols(data_dir, args.symbols)
    if not symbols:
        print(f"No symbols found under {data_dir}")
        return 1

    print(f"Updating {len(symbols)} symbols in {data_dir}")
    print(f"Window: latest available trade date minus {args.months} calendar month(s)")

    failures: list[str] = []
    for index, symbol in enumerate(symbols, start=1):
        try:
            result = update_symbol(data_dir, symbol, months=args.months, retries=args.retries)
            print(
                f"[{index}/{len(symbols)}] {result.symbol}: "
                f"{result.start_date} -> {result.end_date}, {result.rows} rows"
            )
            if args.pause > 0:
                time.sleep(args.pause)
        except Exception as exc:
            message = f"[{index}/{len(symbols)}] {symbol}: FAILED - {exc}"
            print(message)
            failures.append(message)

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(failure)
        return 1

    print("\nAll symbols updated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
