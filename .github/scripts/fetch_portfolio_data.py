import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import json
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# --- Configuration ---
DEFAULT_SOFR_RATE_PCT = 4.41 # Fallback if FRED fetch fails (Update with a recent value if needed)
OUTPUT_DIR = Path("data")
OUTPUT_FILENAME = "portfolio-performance.json"
MAX_RETRIES = 3  # Maximum number of retries for failed fetches
BATCH_SIZE = 5   # Number of tickers to fetch at once (Adjust based on testing)
MIN_DELAY = 1.5  # Minimum delay between batches (seconds)
MAX_DELAY = 4.0  # Maximum delay between batches (seconds)
REQUEST_TIMEOUT = 30 # Timeout for yfinance requests

def fetch_ytd_sofr_data(start_date, end_date):
    """ Fetch SOFR rate data from FRED for the specified period. """
    print(f"Fetching SOFR rates from FRED for period: {start_date} to {end_date}...")
    try:
        fred_api_key = os.environ.get('FRED_API_KEY')
        if not fred_api_key: print("Warning: FRED_API_KEY environment variable not set.")

        # Add a small delay before hitting FRED API
        time.sleep(1)
        sofr_data = web.DataReader('SOFR', 'fred', start_date, end_date, api_key=fred_api_key) # Consider adding timeout if supported

        if sofr_data.empty: raise ValueError("FRED returned no data for SOFR.")

        sofr_data.index = pd.to_datetime(sofr_data.index)
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sofr_data = sofr_data.reindex(full_date_range).ffill()

        print(f"Successfully fetched and processed SOFR data. Shape: {sofr_data.shape}")
        return sofr_data

    except Exception as e:
        print(f"Warning: Could not fetch SOFR data: {e}")
        print(f"Using default SOFR rate: {DEFAULT_SOFR_RATE_PCT:.4f}%")
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        default_data = pd.DataFrame(index=full_date_range, data={'SOFR': DEFAULT_SOFR_RATE_PCT})
        return default_data

def fetch_stock_data_in_batches(tickers, start_date, end_date):
    """
    Fetch stock data in batches with delays and retries.

    Args:
        tickers (list): List of ticker symbols to fetch
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        tuple: (DataFrame of adj close prices, list of failed tickers)
    """
    print(f"Fetching stock/ETF data for {len(tickers)} tickers in batches of {BATCH_SIZE}...")

    all_ticker_data = {}
    failed_tickers = []
    ticker_batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    total_batches = len(ticker_batches)

    for batch_idx, batch in enumerate(ticker_batches):
        print(f"  Fetching batch {batch_idx+1}/{total_batches}: {batch}")
        batch_data = None # Reset batch data for each batch attempt cycle

        # --- Add Delay BEFORE fetching (except for the very first batch) ---
        if batch_idx > 0:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            print(f"  Waiting {delay:.1f} seconds before next batch...")
            time.sleep(delay)

        # --- Retry Loop for the current batch ---
        for attempt in range(MAX_RETRIES):
            try:
                # Fetch the batch
                batch_data = yf.download(
                    tickers=batch, # Pass list directly
                    start=start_date,
                    end=end_date,
                    auto_adjust=True, # Use adjusted prices
                    progress=False,   # Disable yfinance progress bar
                    timeout=REQUEST_TIMEOUT
                )

                # Check if data is empty even on success (e.g., invalid tickers in batch)
                if batch_data.empty and len(batch) > 0:
                     print(f"  Warning: Batch {batch_idx+1} returned empty DataFrame for tickers {batch}.")
                     # Don't necessarily retry if empty, could be invalid tickers
                     # Add all tickers in this batch to failed list for now
                     failed_tickers.extend(b for b in batch if b not in failed_tickers)
                     batch_data = None # Ensure it's None so we don't process it
                     break # Exit retry loop for this batch

                # If successful, break out of retry loop
                print(f"  Successfully fetched batch {batch_idx+1}.")
                break # Exit retry loop

            except Exception as e:
                print(f"  Error fetching batch {batch_idx+1} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                     print("  Rate limit error detected.")
                # Check if it's the last attempt
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff + jitter
                    wait_time = (2 ** attempt) * 3 + random.uniform(0, 2) # 3-5s, 7-9s, 17-19s...
                    print(f"  Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    # Add all tickers in batch to failed list if all retries failed
                    failed_tickers.extend(b for b in batch if b not in failed_tickers) # Avoid duplicates
                    print(f"  Failed to fetch batch {batch_idx+1} after {MAX_RETRIES} attempts.")
                    batch_data = None # Ensure it's None

        # --- Process successful batch data ---
        if batch_data is not None and not batch_data.empty:
            # Select 'Close' column. Handle MultiIndex vs single index.
            if isinstance(batch_data.columns, pd.MultiIndex):
                close_data = batch_data['Close']
            else: # Should be single ticker if not MultiIndex
                close_data = batch_data[['Close']] if 'Close' in batch_data.columns else pd.DataFrame() # Empty if 'Close' missing

            # Add successfully fetched data to our main dictionary
            for ticker in batch:
                if ticker in close_data.columns:
                    # Check for all NaN columns which yfinance sometimes returns on failure
                    if not close_data[ticker].isnull().all():
                         all_ticker_data[ticker] = close_data[ticker].rename(ticker)
                    else:
                         print(f"  Warning: Data for {ticker} in batch {batch_idx+1} was all NaN.")
                         if ticker not in failed_tickers: failed_tickers.append(ticker)
                elif ticker not in failed_tickers: # If ticker wasn't in columns and not already marked failed
                    print(f"  Warning: Ticker {ticker} not found in response for batch {batch_idx+1}.")
                    failed_tickers.append(ticker)

    # --- Combine and return ---
    if not all_ticker_data:
        print("Error: Failed to fetch data for all tickers after retries.")
        return None, tickers # Return None and all original tickers as failed

    # Use outer join to keep all dates from all fetched series
    combined_data = pd.concat(all_ticker_data.values(), axis=1, join='outer')
    combined_data.index = pd.to_datetime(combined_data.index)

    # Report final results
    success_count = len(tickers) - len(failed_tickers)
    print(f"\nFetch summary: Successfully fetched {success_count}/{len(tickers)} tickers.")
    if failed_tickers:
        print(f"Failed tickers: {list(set(failed_tickers))}") # Show unique failed tickers

    return combined_data, list(set(failed_tickers)) # Return unique failed tickers

def generate_portfolio_data(portfolio_holdings, benchmark_ticker='SPY'):
    """
    Generates the portfolio performance JSON using batched fetching.
    Metrics calculation is REMOVED - should be done client-side.
    """
    print("Starting data generation...")

    # --- 1. Define Tickers and Date Range ---
    portfolio_tickers = list(portfolio_holdings.keys())
    all_tickers = portfolio_tickers + [benchmark_ticker]
    weights = np.array(list(portfolio_holdings.values()))

    if not np.isclose(weights.sum(), 1.0):
        print(f"Warning: Portfolio weights sum to {weights.sum():.4f}, not 1.0. Normalizing weights.")
        weights = weights / weights.sum()

    today = datetime.now().date()
    start_of_year = datetime(today.year, 1, 1).date()
    start_date_str = start_of_year.strftime('%Y-%m-%d')
    end_date_str = (today + timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Date range: {start_date_str} to {today.strftime('%Y-%m-%d')}")

    # --- 2. Fetch YTD SOFR Data ---
    sofr_history_df = fetch_ytd_sofr_data(start_of_year, today)
    latest_sofr_pct = sofr_history_df['SOFR'].iloc[-1] if not sofr_history_df.empty else DEFAULT_SOFR_RATE_PCT

    # --- 3. Fetch Historical Stock/ETF Data (in batches) ---
    adj_close_data, failed_tickers = fetch_stock_data_in_batches(all_tickers, start_date_str, end_date_str)

    if adj_close_data is None or adj_close_data.empty:
        print("Error: No valid stock/ETF data after fetching.")
        error_json = {"error": "Failed to fetch any valid stock/ETF data.", "riskFree": {"name": "SOFR", "rate": latest_sofr_pct}}
        return json.dumps(error_json, indent=2)

    # --- Post-fetch Processing ---
    adj_close_data = adj_close_data.ffill().bfill()
    adj_close_data = adj_close_data[adj_close_data.index.date >= start_of_year]
    adj_close_data = adj_close_data[adj_close_data.index.date <= today]

    if adj_close_data.empty:
        print("Error: No valid stock/ETF data remaining after processing.")
        error_json = {"error": "No valid stock/ETF data remaining after processing.", "riskFree": {"name": "SOFR", "rate": latest_sofr_pct}}
        return json.dumps(error_json, indent=2)

    first_day_prices = adj_close_data.iloc[0]
    if adj_close_data.isnull().values.any(): print("Warning: Missing stock/ETF data found after fill/join.")

    print(f"Stock/ETF data processed successfully. Shape: {adj_close_data.shape}")
    latest_data_row = adj_close_data.iloc[-1]
    current_date_actual = adj_close_data.index[-1].date()

    # --- 4. Calculate Daily Returns ---
    daily_returns = adj_close_data.pct_change().dropna()
    if daily_returns.empty:
        print("Error: Not enough data points to calculate daily returns.")
        error_json = {"error": "Not enough data points to calculate daily returns.", "riskFree": {"name": "SOFR", "rate": latest_sofr_pct}}
        return json.dumps(error_json, indent=2)

    # --- 5. Calculate Portfolio Performance ---
    present_tickers = [t for t in portfolio_tickers if t in daily_returns.columns and t not in failed_tickers]
    present_weights = np.array([portfolio_holdings[t] for t in present_tickers])
    if not present_tickers:
         print("Error: Failed to fetch data for all portfolio tickers.")
         error_json = {"error": "Failed to fetch data for any portfolio tickers.", "riskFree": {"name": "SOFR", "rate": latest_sofr_pct}}
         return json.dumps(error_json, indent=2)
    if len(present_tickers) < len(portfolio_tickers):
        original_failed = set(portfolio_tickers) - set(present_tickers)
        print(f"Warning: Portfolio tickers missing data: {original_failed}. Renormalizing weights.")
        if present_weights.sum() > 0: present_weights = present_weights / present_weights.sum()
        else:
             print("Error: Portfolio ticker weights sum to zero after filtering.")
             error_json = {"error": "Portfolio weights sum to zero.", "riskFree": {"name": "SOFR", "rate": latest_sofr_pct}}
             return json.dumps(error_json, indent=2)
    weighted_returns = daily_returns[present_tickers] * present_weights[np.newaxis, :]
    portfolio_daily_performance = weighted_returns.sum(axis=1)
    cumulative_portfolio_performance = (1 + portfolio_daily_performance).cumprod() - 1

    # --- 6. Calculate Benchmark Performance ---
    if benchmark_ticker in failed_tickers or benchmark_ticker not in daily_returns.columns:
         print(f"Error: Benchmark ticker {benchmark_ticker} data missing or failed to fetch.")
         error_json = {"error": f"Benchmark ticker {benchmark_ticker} data missing/failed.", "riskFree": {"name": "SOFR", "rate": latest_sofr_pct}}
         return json.dumps(error_json, indent=2)
    benchmark_returns_daily = daily_returns[benchmark_ticker]
    cumulative_benchmark_performance = (1 + benchmark_returns_daily).cumprod() - 1

    # --- 7. Align SOFR Data with Portfolio Dates ---
    daily_returns.index = pd.to_datetime(daily_returns.index)
    sofr_history_df.index = pd.to_datetime(sofr_history_df.index)
    aligned_sofr_series = sofr_history_df['SOFR'].reindex(daily_returns.index, method='ffill').bfill()
    if aligned_sofr_series.isnull().any():
        print("Warning: Could not align SOFR rates for all trading dates. Filling remaining NaNs with latest rate.")
        aligned_sofr_series = aligned_sofr_series.fillna(latest_sofr_pct)
    sofr_rates_list_pct = aligned_sofr_series.round(4).tolist()

    # --- 8. Prepare Time Series Data ---
    portfolio_ytd_pct = (cumulative_portfolio_performance * 100).round(2).tolist()
    benchmark_ytd_pct = (cumulative_benchmark_performance * 100).round(2).tolist()
    dates_list = cumulative_portfolio_performance.index.strftime('%Y-%m-%d').tolist()
    if len(sofr_rates_list_pct) != len(dates_list):
         print(f"Warning: SOFR list length ({len(sofr_rates_list_pct)}) doesn't match dates list length ({len(dates_list)}). Adjusting...")
         # Adjusting SOFR rates to match dates
         if len(dates_list) > 0:
             if len(sofr_rates_list_pct) == 0: sofr_rates_list_pct = [latest_sofr_pct] * len(dates_list)
             else:
                 # Create a mapping of dates to rates, then fill in values
                 sofr_dict = dict(zip(sofr_history_df.index.strftime('%Y-%m-%d').tolist(), sofr_history_df['SOFR'].tolist()))
                 sofr_rates_list_pct = [sofr_dict.get(date, latest_sofr_pct) for date in dates_list]


    # --- 9. Prepare Daily Snapshot ---
    snapshot_portfolio_stocks = []
    for ticker in present_tickers:
        start_price = first_day_prices.get(ticker, 0)
        latest_price = latest_data_row.get(ticker, 0)
        ytd_change = ((latest_price / start_price) - 1) * 100 if start_price != 0 else 0
        snapshot_portfolio_stocks.append({"ticker": ticker, "weight": portfolio_holdings[ticker], "price": round(latest_price, 2), "ytdChange": round(ytd_change, 2)})
    latest_portfolio_perf = portfolio_ytd_pct[-1] if portfolio_ytd_pct else 0
    latest_benchmark_perf = benchmark_ytd_pct[-1] if benchmark_ytd_pct else 0
    latest_benchmark_price = round(latest_data_row.get(benchmark_ticker, 0), 2)
    daily_snapshot = {
        "portfolio": {"performance": latest_portfolio_perf, "stocks": snapshot_portfolio_stocks},
        "benchmark": {"ticker": benchmark_ticker, "performance": latest_benchmark_perf, "price": latest_benchmark_price},
        "riskFree": {"name": "SOFR", "rate": round(latest_sofr_pct, 4)}
    }

    # --- 10. Assemble Final JSON ---
    # REMOVED calculation of risk metrics here
    output_data = {
        "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        "lastUpdated": datetime.now().strftime('%-m/%-d/%Y, %-I:%M:%S %p'),
        "startDate": start_of_year.strftime('%Y-%m-%d'),
        "currentDate": current_date_actual.strftime('%Y-%m-%d'),
        "dailySnapshot": daily_snapshot,
        "timeSeriesData": {
            "dates": dates_list,
            "portfolio": portfolio_ytd_pct,
            "benchmark": benchmark_ytd_pct,
            "sofr_rates_pct": sofr_rates_list_pct # Use consistent key name
        }
        # REMOVED "riskMetrics" object
    }

    print("Data generation complete.")
    if failed_tickers: print(f"Warning: Failed to fetch data for: {list(set(failed_tickers))}")

    return json.dumps(output_data, indent=2)

# --- Main Execution ---
def main():
    portfolio = { "PGR": 0.2703, "IAU": 0.2394, "JPM": 0.0368, "CAT": 0.0293, "NVDA": 0.1229, "MSFT": 0.0554, "COST": 0.0414, "NOW": 0.0551, "WMT": 0.1109, "RDDT": 0.0385 }
    json_output = generate_portfolio_data(portfolio_holdings=portfolio, benchmark_ticker='SPY')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        output_path = OUTPUT_DIR / OUTPUT_FILENAME
        with open(output_path, "w") as f: f.write(json_output)
        print(f"\nSuccessfully saved data to {output_path}")
    except Exception as e: print(f"\nError saving file: {e}")

if __name__ == "__main__":
    main()
