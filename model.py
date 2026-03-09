import pandas as pd
import numpy as np

# ── Stock Prices and Daily Movements ─────────────────────────────────────────

# Load raw CSV (Capital IQ format with metadata header rows)
raw = pd.read_csv("CAPIQ Stocks and Daily Movement.csv", header=None)

# Column 0 is empty (leading comma); labels are in column 1, data starts at column 2
tickers = raw.iloc[1, 2:].values  # Row 1 = Ticker row

# Rows 0-7 are metadata; row 8 onward is daily price data
date_start = 8

# Build the prices DataFrame
prices = raw.iloc[date_start:, 2:].copy()
prices.columns = tickers
prices.index = pd.to_datetime(raw.iloc[date_start:, 1], format="%m/%d/%Y")
prices.index.name = "Date"

# Remove commas from numbers and convert to float
prices = prices.replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")

print(f"Shape: {prices.shape[0]} trading days x {prices.shape[1]} stocks")
print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Null count: {prices.isnull().sum().sum()}")
print(prices.head())

# ── WMEC Stocks and Info ──────────────────────────────────────────────────────

# Load WMEC Stocks and Info
stocks_info = pd.read_csv("WMEC Stocks and Info.csv")

print(f"\nShape: {stocks_info.shape[0]} companies x {stocks_info.shape[1]} columns")
print(f"Columns: {list(stocks_info.columns)}")
print(stocks_info.head())
