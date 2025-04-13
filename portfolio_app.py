import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import pytz
from functools import lru_cache
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules
from timezone_handler import timezone_handler
from asset_classifier import asset_classifier
from visualization_helper import visualization_helper

# ==============================================
# Configuration
# ==============================================
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# Constants
MONEY_MARKET_TICKERS = ["WMPXX", "FNSXX", "VMFXX"]
COMPARISON_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E"
    # Removed 10Y Treasury as requested
}
PERIOD_MAP = {
    "Today": "1d",
    "1W": "1wk",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "5Y": "5y",
    "Max": "max"
}

# ==============================================
# Helper Functions
# ==============================================
def get_market_times():
    """Get proper market open/close times based on current time, accounting for weekends and holidays"""
    # Use the timezone_handler to get the market timezone
    market_tz = timezone_handler.get_market_timezone("US")
    now = timezone_handler.now().astimezone(market_tz)
    today = now.date()
    
    # Create base market hours for today
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Check if today is a weekend or holiday
    is_weekend = now.weekday() >= 5  # Saturday or Sunday
    is_holiday = today in timezone_handler.US_HOLIDAYS
    
    if is_weekend or is_holiday:
        # Find the most recent trading day
        days_back = 1
        check_date = today - datetime.timedelta(days=days_back)
        
        while check_date.weekday() >= 5 or check_date in timezone_handler.US_HOLIDAYS:
            days_back += 1
            check_date = today - datetime.timedelta(days=days_back)
        
        # Set market times to the most recent trading day
        market_open = now.replace(
            year=check_date.year,
            month=check_date.month,
            day=check_date.day,
            hour=9,
            minute=30,
            second=0,
            microsecond=0
        )
        market_close = market_open.replace(hour=16, minute=0)
    
    # Handle after hours on a trading day
    elif now > market_close:
        # Find the next trading day
        days_forward = 1
        check_date = today + datetime.timedelta(days=days_forward)
        
        while check_date.weekday() >= 5 or check_date in timezone_handler.US_HOLIDAYS:
            days_forward += 1
            check_date = today + datetime.timedelta(days=days_forward)
        
        # Set market times to the next trading day
        market_open = now.replace(
            year=check_date.year,
            month=check_date.month,
            day=check_date.day,
            hour=9,
            minute=30,
            second=0,
            microsecond=0
        )
        market_close = market_open.replace(hour=16, minute=0)
    
    return market_open, market_close

@lru_cache(maxsize=128)
def fetch_market_data(label, symbol, period):
    """Fetch market data with robust error handling and consistent timezone handling"""
    logger.info(f"Fetching market data for {label} ({symbol}) with period {period}")
    try:
        is_crypto = any(x in symbol for x in ["-USD", "-EUR"])
        is_european = any(x in symbol for x in ["^STOXX", ".PA", ".AS", ".DE", ".L", "BNP", "AMS:", "LON:", "FRA:", "EPA:"])
        is_otc = any(x in symbol for x in ["BNPQF", "BNPQY"]) or (len(symbol) == 5 and symbol.endswith("F"))
        is_etf = any(x in symbol for x in ["RSP", "EEM", "VTI", "CLOU", "KBE", "QQQ", "QQQJ", "SIXG"]) or symbol.startswith("^")
        ticker = yf.Ticker(symbol)
        hist = pd.DataFrame()

        # Special handling for intraday data
        if period == "1d" and not (is_crypto or is_otc):
            market_open, market_close = get_market_times()
            
            # European market adjustment
            if is_european:
                # For Euro Stoxx 50 and other European markets, use a different approach
                if "^STOXX" in symbol:
                    # Use a longer period for Euro Stoxx 50 to ensure we get data
                    try:
                        # Get data for the last 7 days to ensure we have enough data
                        hist = ticker.history(period="7d")
                        
                        # Filter to just today's data if available, using proper timezone
                        user_tz = timezone_handler.get_user_timezone()
                        today = timezone_handler.now().date()
                        
                        if not hist.empty and len(hist.index) > 0:
                            # Safely handle index conversion
                            try:
                                if isinstance(hist.index[0], pd.Timestamp):
                                    # Convert index dates to user timezone before comparing
                                    hist_dates = pd.Series([
                                        ts.astimezone(user_tz).date() if ts.tzinfo is not None
                                        else ts.replace(tzinfo=pytz.UTC).astimezone(user_tz).date()
                                        for ts in hist.index
                                    ])
                                    # Use boolean indexing safely
                                    valid_dates = hist_dates == today
                                    if any(valid_dates):
                                        hist = hist.loc[valid_dates]
                            except Exception as e:
                                logger.warning(f"Error filtering Euro Stoxx 50 data: {str(e)}")
                        
                        # If still empty, use all data
                        if hist.empty:
                            hist = ticker.history(period=period)
                    except Exception as e:
                        logger.warning(f"Error fetching Euro Stoxx 50 data: {str(e)}")
                        hist = ticker.history(period=period)
                else:
                    # Other European markets
                    try:
                        # Convert NY market times to European time using proper timezone conversion
                        ny_tz = timezone_handler.get_market_timezone("US")
                        eu_tz = timezone_handler.get_market_timezone("Europe")
                        
                        # Ensure market_open and market_close have timezone info
                        if market_open.tzinfo is None:
                            market_open = market_open.replace(tzinfo=ny_tz)
                        if market_close.tzinfo is None:
                            market_close = market_close.replace(tzinfo=ny_tz)
                            
                        # Convert to European timezone
                        euro_open = market_open.astimezone(eu_tz)
                        euro_close = market_close.astimezone(eu_tz)
                        
                        hist = ticker.history(
                            start=euro_open,
                            end=euro_close,
                            interval="5m",
                            prepost=False
                        )
                    except Exception as e:
                        # Fallback to daily data if 5-minute data fails
                        st.warning(f"5-minute data unavailable for {symbol}, using daily data instead: {str(e)}")
                        hist = ticker.history(period=period)
            else:
                try:
                    hist = ticker.history(
                        start=market_open,
                        end=market_close,
                        interval="5m",
                        prepost=False
                    )
                except Exception as e:
                    # Fallback to daily data if 5-minute data fails
                    st.warning(f"5-minute data unavailable for {symbol}, using daily data instead: {str(e)}")
                    hist = ticker.history(period=period)
        else:
            # For non-intraday periods, crypto, or OTC stocks
            hist = ticker.history(period=period)

        if hist.empty:
            return None

        # Create a copy to avoid SettingWithCopyWarning
        hist = hist.copy()
        
        # IMPORTANT: Standardize timezone info for all datetime indices for consistent merging
        if not hist.empty and hist.index.tz is not None:
            # Convert all timestamps to the user's timezone for consistent display
            user_tz = timezone_handler.get_user_timezone()
            hist.index = pd.DatetimeIndex([d.astimezone(user_tz) for d in hist.index])
            
        # Ensure we have the required columns
        if 'Close' not in hist.columns:
            return None
            
        # Standardize columns and reset index
        hist_reset = hist.reset_index().rename(columns={
            'index': 'Date',
            'Datetime': 'Date'
        })
        
        # Calculate normalized performance with improved type handling
        try:
            # Ensure Close column is numeric
            if 'Close' in hist.columns:
                hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
                
            base_price = float(hist['Close'].iloc[0])
            if base_price <= 0:
                return None
                
            hist_reset["Normalized"] = ((hist['Close'] / base_price) - 1) * 100
            
            # Calculate percentage change
            pct_change = (float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0]) - 1) * 100
        except (IndexError, ZeroDivisionError, TypeError, ValueError) as e:
            st.warning(f"Error calculating performance for {label}: {str(e)}")
            return None
        
        return {
            "data": hist_reset[["Date", "Normalized"]],
            "pct_change": pct_change,
            "label": label
        }
    except Exception as e:
        error_msg = f"Data error for {label}: {str(e)}"
        logger.error(error_msg)
        st.warning(error_msg)
        return None

def clean_portfolio(df):
    """Clean and standardize portfolio dataframe"""
    df.columns = [col.strip().capitalize() for col in df.columns]
    keep_cols = ['Ticker', 'Quantity']
    if 'Account' in df.columns:
        keep_cols.append('Account')
    return df[keep_cols].dropna(subset=['Ticker', 'Quantity'])

def is_money_market(ticker):
    """Check if ticker is a money market fund"""
    classification = asset_classifier.classify(ticker)
    return classification["asset_class"] == "Money Market" or ticker in MONEY_MARKET_TICKERS or "XX" in ticker

# ==============================================
# UI Components
# ==============================================
def render_header():
    st.title("📊 Portfolio Tracker")
    st.caption("Version 6.0 | Created by Rohan Potthoff")
    st.markdown("""
    <style>
    .social-icons { display: flex; gap: 15px; margin-top: -10px; margin-bottom: 10px; }
    .social-icons a { color: #9e9e9e !important; text-decoration: none; font-size: 14px; }
    .social-icons a:hover { color: #1DA1F2 !important; }
    </style>
    <div class="social-icons">
        <a href="mailto:rohanpotthoff@gmail.com">✉️ Email</a>
        <a href="https://www.linkedin.com/in/rohanpotthoff" target="_blank">🔗 LinkedIn</a>
    </div>
    <hr style='margin-bottom:20px'>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar.expander("📦 Version History", expanded=False):
        st.markdown("""
        - **v6.0**: Enhanced asset classification, global timezone support, improved visualizations
        - **v5.0**: Fixed performance calculation, removed 10Y Treasury, improved ETF handling
        - **v4.9**: Fixed 5-minute interval data processing and OTC stock handling
        - **v4.8**: Fixed DatetimeIndex.dt attribute error for multiple stocks
        """)
    
    with st.sidebar.expander("🔧 Filters & Settings", expanded=True):
        # Performance period selection
        selected_period = st.selectbox(
            "Performance Period",
            list(PERIOD_MAP.keys()),
            index=0
        )
        
        # Visualization options
        st.subheader("Visualization Options")
        show_absolute = st.checkbox("Show absolute values", value=False,
                                   help="Display portfolio value alongside percentage returns")
        
        # Timezone selection
        st.subheader("Timezone Settings")
        current_tz = timezone_handler.get_user_timezone().zone
        available_timezones = [
            "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
            "Europe/London", "Europe/Paris", "Europe/Berlin", "Asia/Tokyo",
            "Asia/Shanghai", "Asia/Singapore", "Australia/Sydney"
        ]
        
        if current_tz not in available_timezones:
            available_timezones.append(current_tz)
            available_timezones.sort()
            
        selected_timezone = st.selectbox(
            "Display Timezone",
            available_timezones,
            index=available_timezones.index(current_tz),
            help="Select timezone for displaying dates and times"
        )
        
        # Update timezone if changed
        if selected_timezone != current_tz:
            st.session_state['user_timezone'] = selected_timezone
            st.rerun()
        
        # Display market status
        st.info(timezone_handler.get_market_status_display())
        
    return selected_period, show_absolute

# ==============================================
# Main Processing
# ==============================================
def main():
    render_header()
    selected_period, show_absolute = render_sidebar()
    period = PERIOD_MAP[selected_period]

    # Fetch benchmark data with improved error handling
    benchmark_series = []
    benchmark_values = {}
    
    with st.spinner("Fetching market data..."):
        for label, symbol in COMPARISON_TICKERS.items():
            try:
                result = fetch_market_data(label, symbol, period)
                if result and "pct_change" in result:
                    # Ensure pct_change is a valid number
                    pct_change = result["pct_change"]
                    if pd.isna(pct_change) or not np.isfinite(pct_change):
                        pct_change = 0.0
                        st.warning(f"Invalid percentage change for {label}. Using 0% as default.")
                    
                    benchmark_series.append(result)
                    benchmark_values[label] = pct_change
                else:
                    st.warning(f"Could not fetch data for {label}")
                    # Add a default value to ensure the benchmark appears in the UI
                    benchmark_values[label] = 0.0
            except Exception as e:
                st.warning(f"Error fetching {label} data: {str(e)}")
                # Add a default value to ensure the benchmark appears in the UI
                benchmark_values[label] = 0.0

    # File upload handling
    uploaded_files = st.file_uploader(
        "Upload portfolio holdings (CSV/Excel)", 
        type=["csv", "xlsx"], 
        accept_multiple_files=True,
        help="Required columns: Ticker, Quantity | Optional: Account"
    )

    if uploaded_files:
        try:
            # Process uploaded files
            df_list = []
            seen_tickers = set()
            duplicates = set()
            
            for file in uploaded_files:
                file_df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                cleaned_df = clean_portfolio(file_df)
                
                new_dupes = set(cleaned_df.Ticker) & seen_tickers
                duplicates.update(new_dupes)
                seen_tickers.update(cleaned_df.Ticker)
                df_list.append(cleaned_df)
            
            df = pd.concat(df_list, ignore_index=True)
            
            if duplicates:
                with st.expander("⚠️ Duplicate Tickers Detected", expanded=True):
                    st.warning(f"Duplicate tickers: {', '.join(duplicates)}")

            # Process holdings
            price_data = []
            portfolio_history = pd.DataFrame()
            portfolio_start_value = 0
            portfolio_end_value = 0

            # First pass: Calculate portfolio value
            portfolio_start_value = 0
            portfolio_end_value = 0
            
            for ticker in df.Ticker.unique():
                try:
                    qty = df[df.Ticker == ticker].Quantity.sum()
                    
                    if is_money_market(ticker):
                        # Money market funds are always valued at $1.00
                        portfolio_start_value += qty * 1.0
                        portfolio_end_value += qty * 1.0
                        price_data.append({
                            "Ticker": ticker,
                            "Current Price": 1.0,
                            "Sector": "Cash",
                            "Asset Class": "Money Market"
                        })
                        continue

                    # Fetch market data with better error handling for European stocks
                    result = fetch_market_data(ticker, ticker, period)
                    if not result:
                        # Special handling for OTC stocks
                        if any(x in ticker for x in ["BNPQF", "BNPQY"]) or (len(ticker) == 5 and ticker.endswith("F")):
                            st.info(f"{ticker} appears to be an OTC stock. Using alternative data source.")
                        else:
                            st.warning(f"Could not fetch data for {ticker}. Using last known price.")
                        # Try to get basic info for price
                        try:
                            ticker_info = yf.Ticker(ticker).info
                            current_price = ticker_info.get('regularMarketPrice', 0)
                            if current_price and current_price > 0:
                                portfolio_start_value += qty * current_price
                                portfolio_end_value += qty * current_price
                                price_data.append({
                                    "Ticker": ticker,
                                    "Current Price": current_price,
                                    "Sector": ticker_info.get("sector", "Unknown"),
                                    "Asset Class": ticker_info.get("quoteType", "Stock").title()
                                })
                                continue
                        except Exception:
                            pass
                        
                        # If we still can't get data, skip this ticker
                        st.error(f"Skipping {ticker} due to data retrieval failure.")
                        continue
                        
                    hist = result["data"]
                    if hist.empty:
                        st.warning(f"Empty data for {ticker}. Using last known price.")
                        continue
                    
                    # Safely extract prices with improved type checking
                    try:
                        # For historical data, we need both start and current price
                        if 'Close' in hist.columns and len(hist) >= 2:
                            # Ensure Close column is numeric
                            if not pd.api.types.is_numeric_dtype(hist['Close']):
                                hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
                                
                            start_price = float(hist['Close'].iloc[0])
                            current_price = float(hist['Close'].iloc[-1])
                        else:
                            # Fallback to current price only
                            ticker_info = yf.Ticker(ticker).info
                            current_price = ticker_info.get('regularMarketPrice', 0)
                            # Handle non-numeric values
                            if not isinstance(current_price, (int, float)):
                                current_price = pd.to_numeric(current_price, errors='coerce')
                            start_price = current_price
                            
                        if pd.isna(start_price) or pd.isna(current_price) or start_price <= 0 or current_price <= 0:
                            raise ValueError("Invalid price data")
                    except Exception as e:
                        st.warning(f"Price data error for {ticker}: {str(e)}. Using default values.")
                        start_price = 1.0
                        current_price = 1.0
                    
                    # Update portfolio values
                    portfolio_start_value += qty * start_price
                    portfolio_end_value += qty * current_price
                    
                    # Store performance data with proper error handling
                    try:
                        if 'Date' in hist.columns and 'Normalized' in hist.columns:
                            # Ensure Normalized column is numeric
                            if not pd.api.types.is_numeric_dtype(hist['Normalized']):
                                hist['Normalized'] = pd.to_numeric(hist['Normalized'], errors='coerce')
                                
                            hist_copy = hist[["Date", "Normalized"]].copy()
                            
                            # Ensure Date is a proper datetime - handle both Series and DatetimeIndex
                            if isinstance(hist_copy["Date"], pd.DatetimeIndex):
                                # Convert DatetimeIndex to regular datetime Series
                                hist_copy["Date"] = pd.Series(hist_copy["Date"].to_pydatetime())
                            else:
                                # For regular Series, use pd.to_datetime with error handling
                                hist_copy["Date"] = pd.to_datetime(hist_copy["Date"], errors='coerce')
                                
                            # Drop any rows with NaT dates
                            hist_copy = hist_copy.dropna(subset=["Date"])
                            
                            # Standardize timezone info to user timezone
                            user_tz = timezone_handler.get_user_timezone()
                            hist_copy["Date"] = hist_copy["Date"].apply(
                                lambda x: x.astimezone(user_tz) if hasattr(x, 'tzinfo') and x.tzinfo is not None else
                                          x.replace(tzinfo=user_tz) if hasattr(x, 'tzinfo') else x
                            )
                            
                            if not portfolio_history.empty:
                                # Create a copy to avoid modifying original data
                                portfolio_copy = portfolio_history.copy()
                                
                                # Ensure both Date columns are proper datetime objects
                                if isinstance(portfolio_copy["Date"], pd.DatetimeIndex):
                                    # Convert DatetimeIndex to regular datetime Series
                                    portfolio_copy["Date"] = pd.Series(portfolio_copy["Date"].to_pydatetime())
                                else:
                                    # For regular Series, use pd.to_datetime with error handling
                                    portfolio_copy["Date"] = pd.to_datetime(portfolio_copy["Date"], errors='coerce')
                                    
                                # Drop any rows with NaT dates
                                portfolio_copy = portfolio_copy.dropna(subset=["Date"])
                                
                                # Standardize timezone info to user timezone
                                user_tz = timezone_handler.get_user_timezone()
                                portfolio_copy["Date"] = portfolio_copy["Date"].apply(
                                    lambda x: x.astimezone(user_tz) if hasattr(x, 'tzinfo') and x.tzinfo is not None else
                                              x.replace(tzinfo=user_tz) if hasattr(x, 'tzinfo') else x
                                )
                                
                                # Rename the Normalized column to the ticker symbol
                                hist_copy = hist_copy.rename(columns={"Normalized": ticker})
                                
                                # Use simple merge with better error handling
                                try:
                                    combined = pd.merge(
                                        portfolio_copy,
                                        hist_copy,
                                        on="Date",
                                        how="outer"
                                    )
                                    # Sort by date and remove duplicates
                                    combined = combined.sort_values("Date").drop_duplicates(subset=["Date"])
                                    
                                    portfolio_history = combined
                                except Exception as e:
                                    st.warning(f"Error merging data for {ticker}: {str(e)}")
                                    # Try an alternative approach
                                    try:
                                        # Convert dates to strings for more reliable merging
                                        # Handle both Series and DatetimeIndex types safely
                                        try:
                                            if hasattr(portfolio_copy["Date"], "dt"):
                                                portfolio_copy["Date_str"] = portfolio_copy["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
                                            else:
                                                portfolio_copy["Date_str"] = portfolio_copy["Date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                                                
                                            if hasattr(hist_copy["Date"], "dt"):
                                                hist_copy["Date_str"] = hist_copy["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
                                            else:
                                                hist_copy["Date_str"] = hist_copy["Date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                                        except Exception as e:
                                            st.warning(f"Date conversion error: {str(e)}. Using alternative approach.")
                                            # Fallback to simpler approach
                                            portfolio_copy["Date_str"] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in portfolio_copy["Date"]]
                                            hist_copy["Date_str"] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in hist_copy["Date"]]
                                        
                                        # Merge on string dates
                                        combined = pd.merge(
                                            portfolio_copy,
                                            hist_copy,
                                            on="Date_str",
                                            how="outer"
                                        )
                                        
                                        # Convert back to datetime
                                        combined["Date"] = pd.to_datetime(combined["Date_str"])
                                        combined = combined.drop("Date_str", axis=1)
                                        
                                        # Sort by date and remove duplicates
                                        combined = combined.sort_values("Date").drop_duplicates(subset=["Date"])
                                        
                                        portfolio_history = combined
                                    except Exception as e2:
                                        st.error(f"Failed alternative merge for {ticker}: {str(e2)}")
                            else:
                                portfolio_history = hist_copy.rename(columns={"Normalized": ticker})
                    except Exception as e:
                        st.warning(f"Error processing performance data for {ticker}: {str(e)}")
                        continue

                    # Get stock info with robust error handling
                    try:
                        info = yf.Ticker(ticker).info
                        
                        # Safely extract sector and asset class with type checking
                        sector = info.get("sector", "Unknown")
                        if not isinstance(sector, str):
                            sector = "Unknown"
                            
                        asset_class = info.get("quoteType", "Stock")
                        if not isinstance(asset_class, str):
                            asset_class = "Stock"
                            
                        price_data.append({
                            "Ticker": ticker,
                            "Current Price": current_price,
                            "Sector": sector,
                            "Asset Class": asset_class.title()
                        })
                    except Exception as e:
                        # Fallback if info retrieval fails
                        price_data.append({
                            "Ticker": ticker,
                            "Current Price": current_price,
                            "Sector": "Unknown",
                            "Asset Class": "Stock"
                        })
                        st.warning(f"Could not retrieve info for {ticker}: {str(e)}")

                except Exception as e:
                    st.error(f"Critical error processing {ticker}: {str(e)}")
                    continue

            # Create portfolio performance data
            portfolio_pct = ((portfolio_end_value / portfolio_start_value - 1) * 100
                           if portfolio_start_value > 0 else 0.0)
            
            # Ensure portfolio_pct is a valid number
            if pd.isna(portfolio_pct) or not np.isfinite(portfolio_pct):
                portfolio_pct = 0.0
                st.warning("Could not calculate portfolio percentage change. Using 0% as default.")
            
            # Performance Metrics
            st.subheader("📈 Performance Metrics")
            cols = st.columns([2] + [1]*len(COMPARISON_TICKERS))
            
            with cols[0]:
                delta_color = "normal" if portfolio_pct >= 0 else "inverse"
                st.metric(
                    "My Portfolio",
                    f"${portfolio_end_value:,.2f}",
                    f"{portfolio_pct:.2f}%",
                    delta_color=delta_color
                )

            for i, (label, value) in enumerate(benchmark_values.items()):
                with cols[i+1]:
                    # Ensure value is a valid number
                    if pd.isna(value) or not np.isfinite(value):
                        value = 0.0
                    
                    delta_color = "normal" if value >= 0 else "inverse"
                    st.metric(label, "", f"{value:.2f}%", delta_color=delta_color)

            # Performance Visualization with enhanced charts
            if not portfolio_history.empty:
                st.subheader("📊 Performance Comparison")
                
                # Handle portfolio normalization with proper error handling
                try:
                    # Clean up the portfolio history data
                    portfolio_history = portfolio_history.copy()
                    
                    # Ensure Date column is properly formatted
                    portfolio_history["Date"] = pd.to_datetime(portfolio_history["Date"])
                    
                    # Drop any rows with NaN Date values
                    portfolio_history = portfolio_history.dropna(subset=['Date'])
                    
                    # Get numeric columns (excluding Date)
                    numeric_cols = [col for col in portfolio_history.columns if col != 'Date']
                    
                    # Fill NaN values with forward fill then backward fill
                    for col in numeric_cols:
                        portfolio_history[col] = portfolio_history[col].ffill().bfill()
                    
                    # Set index and calculate mean, handling NaN values properly
                    portfolio_history_indexed = portfolio_history.set_index('Date')
                    
                    # Add validation to prevent division by zero
                    if not portfolio_history_indexed.empty and len(numeric_cols) > 0:
                        # Calculate mean performance
                        # First check if there are any numeric columns
                        if len(numeric_cols) == 0:
                            st.warning("No numeric columns found in portfolio history data.")
                            # Create a simple DataFrame with the portfolio percentage change
                            current_time = timezone_handler.now()
                            portfolio_mean = pd.Series([0.0, portfolio_pct],
                                                      index=[current_time - datetime.timedelta(hours=1), current_time])
                        else:
                            # Calculate mean performance with proper error handling
                            try:
                                portfolio_mean = portfolio_history_indexed[numeric_cols].mean(axis=1, skipna=True)
                                
                                # Ensure portfolio_mean is not empty and contains valid data
                                if portfolio_mean.empty or portfolio_mean.isna().all():
                                    st.warning("Portfolio performance data is empty or contains only NaN values.")
                                    # Create a simple DataFrame with the portfolio percentage change
                                    current_time = timezone_handler.now()
                                    portfolio_mean = pd.Series([0.0, portfolio_pct],
                                                              index=[current_time - datetime.timedelta(hours=1), current_time])
                            except Exception as e:
                                st.warning(f"Error calculating portfolio mean: {str(e)}")
                                # Create a simple DataFrame with the portfolio percentage change
                                current_time = timezone_handler.now()
                                portfolio_mean = pd.Series([0.0, portfolio_pct],
                                                          index=[current_time - datetime.timedelta(hours=1), current_time])
                        
                        # Add absolute portfolio value if available
                        if show_absolute:
                            # Calculate absolute values based on portfolio weights
                            try:
                                portfolio_history_indexed["Value"] = portfolio_end_value
                            except Exception as e:
                                st.warning(f"Error adding absolute value: {str(e)}")
                        
                        # Create enhanced visualization
                        fig = visualization_helper.create_performance_chart(
                            portfolio_mean,
                            benchmark_series,
                            period,
                            show_absolute=show_absolute
                        )
                        
                        # Render the chart
                        visualization_helper.render_chart(fig)
                    else:
                        st.warning("Insufficient data for performance visualization")
                except Exception as e:
                    st.error(f"Error calculating portfolio performance: {str(e)}")
                    st.error(f"Details: {str(e)}")
                
                # Asset Allocation Analysis
                st.subheader("📊 Asset Allocation Analysis")
                
                # Add asset class information to the dataframe
                df["Asset Class"] = df["Ticker"].apply(
                      lambda x: asset_classifier.classify(x)["asset_class"]
                )
                
                # Create asset allocation charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Asset Class Allocation
                    # Ensure df has the required columns
                    if "Asset Class" not in df.columns:
                        # Add Asset Class if missing
                        df["Asset Class"] = df["Ticker"].apply(
                            lambda x: asset_classifier.classify(x)["asset_class"]
                        )
                    
                    # Create the chart with error handling
                    try:
                        asset_class_fig = visualization_helper.create_allocation_chart(
                            df,
                            "Asset Class",
                            "Asset Class Allocation"
                        )
                        visualization_helper.render_chart(asset_class_fig)
                    except Exception as e:
                        st.error(f"Error creating Asset Class chart: {str(e)}")
                    
                with col2:
                    # Sector Allocation
                    # Ensure df has the required columns
                    if "Sector" not in df.columns:
                        # Add Sector if missing
                        df["Sector"] = "Unknown"
                        st.warning("Missing sector data. Using 'Unknown' as default.")
                    
                    # Create the chart with error handling
                    try:
                        sector_fig = visualization_helper.create_allocation_chart(
                            df,
                            "Sector",
                            "Sector Allocation"
                        )
                        visualization_helper.render_chart(sector_fig)
                    except Exception as e:
                        st.error(f"Error creating Sector chart: {str(e)}")

            # Portfolio Insights
            st.subheader("🧠 Portfolio Insights")
            insights = []
            
            if portfolio_end_value > 0:
                try:
                    # Merge price data with error handling
                    price_df = pd.DataFrame(price_data)
                    
                    # Check if price_df is empty
                    if price_df.empty:
                        st.warning("No price data available. Using default values.")
                        df["Current Price"] = 1.0  # Default price
                    else:
                        # Ensure both dataframes have the Ticker column
                        if "Ticker" not in df.columns:
                            st.error("Portfolio data missing Ticker column.")
                            df["Ticker"] = "Unknown"
                        
                        if "Ticker" not in price_df.columns:
                            st.error("Price data missing Ticker column.")
                            # Skip the merge if Ticker column is missing
                            pass
                        else:
                            # Perform the merge with error handling
                            df = df.merge(price_df, on="Ticker", how="left")
                        
                        # Fill missing values
                        if "Current Price" not in df.columns:
                            df["Current Price"] = 1.0
                        else:
                            df["Current Price"] = df["Current Price"].fillna(1.0)
                    
                    # Calculate Market Value and Weight
                    df["Market Value"] = df["Quantity"] * df["Current Price"]
                    df["Weight"] = df["Market Value"] / portfolio_end_value
                    
                except Exception as e:
                    st.error(f"Error processing portfolio data: {str(e)}")
                    # Create default columns to prevent further errors
                    if "Market Value" not in df.columns:
                        df["Market Value"] = df["Quantity"] * 1.0
                    if "Weight" not in df.columns:
                        df["Weight"] = 1.0 / len(df) if len(df) > 0 else 1.0

                # Concentration risk with proper error handling
                try:
                    # Ensure Weight column exists and is numeric
                    if "Weight" in df.columns:
                        # Convert to numeric with error handling
                        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
                        
                        # Use boolean indexing safely
                        weight_filter = df["Weight"] > 0.1
                        if not weight_filter.empty:
                            heavy = df[weight_filter]
                            for _, row in heavy.iterrows():
                                if "Ticker" in row and "Weight" in row and pd.notna(row["Weight"]):
                                    insights.append(f"⚠️ **{row.Ticker}** ({row.Weight:.1%}) exceeds 10% allocation")
                except Exception as e:
                    st.warning(f"Error analyzing concentration risk: {str(e)}")

                # Cash position with proper error handling
                try:
                    if "Asset Class" in df.columns and "Market Value" in df.columns:
                        # Use boolean indexing safely
                        cash_filter = df["Asset Class"] == "Money Market"
                        if not cash_filter.empty:
                            cash = df[cash_filter]["Market Value"].sum()
                            cash_pct = cash / portfolio_end_value if portfolio_end_value > 0 else 0
                            if cash_pct > 0.15:
                                insights.append(f"🪙 Cash allocation ({cash_pct:.1%}) may create drag")
                except Exception as e:
                    st.warning(f"Error analyzing cash position: {str(e)}")

                # Big movers with improved error handling
                try:
                    for _, row in df.iterrows():
                        # Skip money market funds and ensure Asset Class exists
                        if "Asset Class" not in row or row.get("Asset Class") == "Money Market":
                            continue
                        
                        # Ensure Ticker exists
                        if "Ticker" not in row:
                            continue
                            
                        try:
                            result = fetch_market_data(row.Ticker, row.Ticker, period)
                            if result and "pct_change" in result:
                                change = result["pct_change"]
                                # Ensure change is a valid number
                                if pd.isna(change) or not np.isfinite(change):
                                    continue
                                    
                                if change <= -10:
                                    insights.append(f"🔻 **{row.Ticker}** dropped {abs(change):.1f}%")
                                elif change >= 20:
                                    insights.append(f"🚀 **{row.Ticker}** gained {change:.1f}%")
                        except Exception as e:
                            logger.warning(f"Error processing big mover {row.get('Ticker', 'Unknown')}: {str(e)}")
                except Exception as e:
                    st.warning(f"Error analyzing big movers: {str(e)}")

                # Earnings alerts with improved error handling
                try:
                    # Ensure df has Ticker column
                    if "Ticker" not in df.columns:
                        st.warning("Missing Ticker column for earnings alerts")
                    else:
                        for ticker in df.Ticker.unique():
                            # Skip money market funds, ETFs, indices, and crypto
                            if is_money_market(ticker):
                                continue
                            
                            # Skip ETFs, indices, and crypto that don't have earnings
                            if any(x in ticker for x in ["RSP", "EEM", "VTI", "CLOU", "KBE", "QQQ", "QQQJ", "SIXG"]) or \
                               ticker.startswith("^") or "-USD" in ticker or "-EUR" in ticker:
                                continue
                                
                            try:
                                cal = yf.Ticker(ticker).calendar
                                
                                # Check if calendar exists and has the expected structure
                                if isinstance(cal, pd.DataFrame) and not cal.empty and 'EarningsDate' in cal.columns:
                                    # Handle different possible data types for earnings date
                                    earnings_date = cal.EarningsDate.max()
                                    
                                    if pd.notna(earnings_date):
                                        # Convert to date object safely
                                        if isinstance(earnings_date, pd.Timestamp):
                                            earnings_date = earnings_date.date()
                                        elif isinstance(earnings_date, datetime.datetime):
                                            earnings_date = earnings_date.date()
                                        elif isinstance(earnings_date, str):
                                            try:
                                                earnings_date = pd.to_datetime(earnings_date).date()
                                            except:
                                                continue
                                        
                                        # Calculate days difference
                                        days = (earnings_date - datetime.date.today()).days
                                        if 0 <= days <= 14:
                                            insights.append(f"📅 **{ticker}** earnings in {days} days")
                            except Exception as e:
                                # Silently ignore 404 errors for ETFs and crypto
                                if "404" in str(e):
                                    pass
                                else:
                                    logger.warning(f"Error fetching earnings for {ticker}: {str(e)}")
                except Exception as e:
                    st.warning(f"Error processing earnings alerts: {str(e)}")

            # Display insights
            if insights:
                priority = {"⚠️": 1, "🪙": 2, "🔻": 3, "🚀": 4, "📅": 5}
                insights.sort(key=lambda x: priority.get(x[:2], 6))
                
                with st.expander("📌 Active Alerts (Top 10)", expanded=True):
                    for note in insights[:10]:
                        st.markdown(f"- {note}")
                    if len(insights) > 10:
                        st.caption(f"+ {len(insights)-10} additional alerts")
                    
                    st.download_button(
                        "💾 Download Insights", 
                        "\n".join(insights), 
                        file_name=f"portfolio_alerts_{datetime.date.today()}.txt"
                    )
            else:
                st.success("🎉 No alerts - portfolio looks healthy!")

            # Portfolio Composition
            st.subheader("📊 Portfolio Composition")
            if portfolio_end_value > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Value", f"${portfolio_end_value:,.2f}")
                with col2:
                    st.metric("Holdings", len(df))
                with col3:
                    accounts = df.Account.nunique() if "Account" in df else 1
                    st.metric("Accounts", accounts)

                # Visualizations
                viz_cols = st.columns(3)
                with viz_cols[0]:
                    sector_group = df.groupby("Sector", as_index=False)["Market Value"].sum()
                    fig = px.pie(sector_group, values="Market Value", names="Sector", 
                               title="Sector Allocation", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)

                with viz_cols[1]:
                    asset_group = df.groupby("Asset Class", as_index=False)["Market Value"].sum()
                    if len(asset_group) > 1:
                        fig = px.pie(asset_group, values="Market Value", names="Asset Class", 
                                   title="Asset Classes", hole=0.3)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Diversify assets for breakdown")

                with viz_cols[2]:
                    if "Account" in df and df.Account.nunique() > 1:
                        account_group = df.groupby("Account", as_index=False)["Market Value"].sum()
                        fig = px.pie(account_group, values="Market Value", names="Account", 
                                   title="Account Distribution", hole=0.3)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Add 'Account' column for breakdown")

                # Raw data export
                with st.expander("🔍 Raw Portfolio Data"):
                    st.dataframe(df.sort_values("Market Value", ascending=False))
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV", 
                        data=csv,
                        file_name=f"portfolio_{datetime.date.today()}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.stop()

    else:
        st.info("📤 Upload portfolio files to begin analysis")

if __name__ == "__main__":
    main()
