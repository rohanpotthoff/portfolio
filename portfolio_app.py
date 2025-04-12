import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import pytz
from functools import lru_cache

# ==============================================
# Configuration
# ==============================================
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# Constants
MONEY_MARKET_TICKERS = ["WMPXX", "FNSXX", "VMFXX"]
COMPARISON_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E",
    "10Y US Treasury": "^TNX"
}
PERIOD_MAP = {
    "Today": "1d",
    "1W": "1wk",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "5Y": "5y"
}
MARKET_TIMEZONE = pytz.timezone("America/New_York")
US_HOLIDAYS = [
    # 2024 US Market Holidays
    datetime.date(2024, 1, 1),    # New Year's Day
    datetime.date(2024, 1, 15),   # Martin Luther King Jr. Day
    datetime.date(2024, 2, 19),   # Presidents' Day
    datetime.date(2024, 3, 29),   # Good Friday
    datetime.date(2024, 5, 27),   # Memorial Day
    datetime.date(2024, 6, 19),   # Juneteenth
    datetime.date(2024, 7, 4),    # Independence Day
    datetime.date(2024, 9, 2),    # Labor Day
    datetime.date(2024, 11, 28),  # Thanksgiving Day
    datetime.date(2024, 12, 25),  # Christmas Day
    # 2025 US Market Holidays
    datetime.date(2025, 1, 1),    # New Year's Day
    datetime.date(2025, 1, 20),   # Martin Luther King Jr. Day
    datetime.date(2025, 2, 17),   # Presidents' Day
    datetime.date(2025, 4, 18),   # Good Friday
    datetime.date(2025, 5, 26),   # Memorial Day
    datetime.date(2025, 6, 19),   # Juneteenth
    datetime.date(2025, 7, 4),    # Independence Day
    datetime.date(2025, 9, 1),    # Labor Day
    datetime.date(2025, 11, 27),  # Thanksgiving Day
    datetime.date(2025, 12, 25),  # Christmas Day
]

# ==============================================
# Helper Functions
# ==============================================
def get_market_times():
    """Get proper market open/close times based on current time, accounting for weekends and holidays"""
    now = datetime.datetime.now(MARKET_TIMEZONE)
    today = now.date()
    
    # Create base market hours for today
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Check if today is a weekend or holiday
    is_weekend = now.weekday() >= 5  # Saturday or Sunday
    is_holiday = today in US_HOLIDAYS
    
    if is_weekend or is_holiday:
        # Find the most recent trading day
        days_back = 1
        check_date = today - datetime.timedelta(days=days_back)
        
        while check_date.weekday() >= 5 or check_date in US_HOLIDAYS:
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
        
        while check_date.weekday() >= 5 or check_date in US_HOLIDAYS:
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
    try:
        is_crypto = any(x in symbol for x in ["-USD", "-EUR"])
        is_european = any(x in symbol for x in ["^STOXX", ".PA", ".AS", ".DE", ".L", "BNP", "AMS:", "LON:", "FRA:", "EPA:"])
        is_otc = any(x in symbol for x in ["BNPQF", "BNPQY"]) or (len(symbol) == 5 and symbol.endswith("F"))
        ticker = yf.Ticker(symbol)
        hist = pd.DataFrame()

        # Special handling for intraday data
        if period == "1d" and not (is_crypto or is_otc):
            market_open, market_close = get_market_times()
            
            # European market adjustment
            if is_european:
                # Convert NY market times to Berlin time (6 hours ahead)
                berlin_open = market_open - datetime.timedelta(hours=6)
                berlin_close = market_close - datetime.timedelta(hours=6)
                
                try:
                    hist = ticker.history(
                        start=berlin_open,
                        end=berlin_close,
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
        
        # IMPORTANT: Remove timezone info from all datetime indices for consistent merging
        if not hist.empty and hist.index.tz is not None:
            # Convert to timezone-naive datetime objects without using .dt accessor
            hist.index = pd.DatetimeIndex([d.replace(tzinfo=None) for d in hist.index])
            
        # Ensure we have the required columns
        if 'Close' not in hist.columns:
            return None
            
        # Standardize columns and reset index
        hist_reset = hist.reset_index().rename(columns={
            'index': 'Date',
            'Datetime': 'Date'
        })
        
        # Calculate normalized performance
        try:
            base_price = float(hist['Close'].iloc[0])
            if base_price <= 0:
                return None
                
            hist_reset["Normalized"] = ((hist['Close'] / base_price) - 1) * 100
            
            # Calculate percentage change
            pct_change = (float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0]) - 1) * 100
        except (IndexError, ZeroDivisionError, TypeError, ValueError):
            return None
        
        return {
            "data": hist_reset[["Date", "Normalized"]],
            "pct_change": pct_change,
            "label": label
        }
    except Exception as e:
        st.warning(f"Data error for {label}: {str(e)}")
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
    return ticker in MONEY_MARKET_TICKERS or "XX" in ticker

# ==============================================
# UI Components
# ==============================================
def render_header():
    st.title("📊 Portfolio Tracker")
    st.caption("Version 4.9 | Created by Rohan Potthoff")
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
        - **v4.9**: Fixed 5-minute interval data processing and OTC stock handling
        - **v4.8**: Fixed DatetimeIndex.dt attribute error for multiple stocks
        - **v4.7**: Fixed portfolio valuation, weekend handling, and European stocks
        - **v4.6**: Fixed cryptocurrency data handling and timezone consistency
        """)
    
    with st.sidebar.expander("🔧 Filters & Settings", expanded=True):
        selected_period = st.selectbox(
            "Performance Period",
            list(PERIOD_MAP.keys()),
            index=0
        )
        # Removed truncated tooltip
    return selected_period

# ==============================================
# Main Processing
# ==============================================
def main():
    render_header()
    selected_period = render_sidebar()
    period = PERIOD_MAP[selected_period]

    # Fetch benchmark data
    benchmark_series = []
    benchmark_values = {}
    for label, symbol in COMPARISON_TICKERS.items():
        result = fetch_market_data(label, symbol, period)
        if result:
            benchmark_series.append(result)
            benchmark_values[label] = result["pct_change"]

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
                    
                    # Safely extract prices with proper type checking
                    try:
                        # For historical data, we need both start and current price
                        if 'Close' in hist.columns and len(hist) >= 2:
                            start_price = float(hist['Close'].iloc[0])
                            current_price = float(hist['Close'].iloc[-1])
                        else:
                            # Fallback to current price only
                            ticker_info = yf.Ticker(ticker).info
                            current_price = ticker_info.get('regularMarketPrice', 0)
                            start_price = current_price
                            
                        if start_price <= 0 or current_price <= 0:
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
                            hist_copy = hist[["Date", "Normalized"]].copy()
                            
                            # Ensure Date is a proper datetime - handle both Series and DatetimeIndex
                            if isinstance(hist_copy["Date"], pd.DatetimeIndex):
                                # Convert DatetimeIndex to regular datetime Series
                                hist_copy["Date"] = pd.Series(hist_copy["Date"].to_pydatetime())
                            else:
                                # For regular Series, use pd.to_datetime
                                hist_copy["Date"] = pd.to_datetime(hist_copy["Date"])
                            
                            # Remove timezone info if present
                            hist_copy["Date"] = hist_copy["Date"].apply(
                                lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x
                            )
                            
                            if not portfolio_history.empty:
                                # Create a copy to avoid modifying original data
                                portfolio_copy = portfolio_history.copy()
                                
                                # Ensure both Date columns are timezone-naive datetime objects
                                if isinstance(portfolio_copy["Date"], pd.DatetimeIndex):
                                    # Convert DatetimeIndex to regular datetime Series
                                    portfolio_copy["Date"] = pd.Series(portfolio_copy["Date"].to_pydatetime())
                                else:
                                    # For regular Series, use pd.to_datetime
                                    portfolio_copy["Date"] = pd.to_datetime(portfolio_copy["Date"])
                                
                                # Remove timezone info if present
                                portfolio_copy["Date"] = portfolio_copy["Date"].apply(
                                    lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x
                                )
                                
                                # Use pd.concat instead of merge for more robust handling
                                hist_copy = hist_copy.rename(columns={"Normalized": ticker})
                                
                                # Use simple merge instead of complex mapping
                                combined = pd.merge(
                                    portfolio_copy,
                                    hist_copy,
                                    on="Date",
                                    how="outer"
                                )
                                # Sort by date and remove duplicates
                                combined = combined.sort_values("Date").drop_duplicates(subset=["Date"])
                                
                                portfolio_history = combined
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
                           if portfolio_start_value > 0 else 0)
            
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
                    st.metric(label, "", f"{value:.2f}%")

            # Performance Visualization
            if not portfolio_history.empty:
                st.subheader("📊 Performance Comparison")
                # Handle portfolio normalization with proper error handling
                try:
                    if not portfolio_history.empty:
                        # Clean up the portfolio history data
                        portfolio_history = portfolio_history.copy()
                        
                        # Ensure Date column is properly formatted - handle both Series and DatetimeIndex
                        if isinstance(portfolio_history["Date"], pd.DatetimeIndex):
                            # Already a DatetimeIndex, convert to Series
                            date_series = pd.Series(portfolio_history["Date"].to_pydatetime())
                            # Create a new DataFrame with the converted Date
                            portfolio_history = portfolio_history.reset_index(drop=True)
                            portfolio_history["Date"] = date_series
                        else:
                            # For regular Series, use pd.to_datetime
                            portfolio_history["Date"] = pd.to_datetime(portfolio_history["Date"])
                        
                        # Remove timezone info if present
                        portfolio_history["Date"] = portfolio_history["Date"].apply(
                            lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x
                        )
                        
                        # Drop any rows with NaN Date values
                        portfolio_history = portfolio_history.dropna(subset=['Date'])
                        
                        # Get numeric columns (excluding Date)
                        numeric_cols = [col for col in portfolio_history.columns if col != 'Date']
                        
                        # Fill NaN values with forward fill then backward fill
                        for col in numeric_cols:
                            portfolio_history[col] = portfolio_history[col].ffill().bfill()
                        
                        # Set index and calculate mean, handling NaN values properly
                        portfolio_history_indexed = portfolio_history.set_index('Date')
                        portfolio_mean = portfolio_history_indexed[numeric_cols].mean(axis=1, skipna=True)
                        
                        # Create a new dataframe with the results
                        portfolio_norm = pd.DataFrame({
                            'Date': portfolio_mean.index,
                            'Normalized': portfolio_mean.values
                        })
                        
                        portfolio_norm["Index"] = "My Portfolio"
                    else:
                        st.warning("No portfolio history data available for charting.")
                        portfolio_norm = pd.DataFrame(columns=["Date", "Normalized", "Index"])
                except Exception as e:
                    st.error(f"Error calculating portfolio performance: {str(e)}")
                    st.error(f"Details: {str(e)}")
                    st.info(f"Debug info - Date type: {type(portfolio_history['Date']).__name__ if not portfolio_history.empty else 'empty'}")
                    portfolio_norm = pd.DataFrame(columns=["Date", "Normalized", "Index"])

                # Process benchmark data
                bench_dfs = []
                for b in benchmark_series:
                    try:
                        bench_df = pd.DataFrame(b["data"])
                        
                        # Ensure Date is properly formatted - handle both Series and DatetimeIndex
                        if isinstance(bench_df["Date"], pd.DatetimeIndex):
                            # Convert DatetimeIndex to regular datetime Series
                            bench_df["Date"] = pd.Series(bench_df["Date"].to_pydatetime())
                        else:
                            # For regular Series, use pd.to_datetime
                            bench_df["Date"] = pd.to_datetime(bench_df["Date"])
                        
                        # Remove timezone info if present
                        bench_df["Date"] = bench_df["Date"].apply(
                            lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x
                        )
                        
                        bench_df["Index"] = b["label"]
                        bench_dfs.append(bench_df)
                    except Exception as e:
                        st.warning(f"Error processing benchmark {b['label']}: {str(e)}")
                
                # Combine portfolio and benchmark data
                try:
                    # Make sure portfolio_norm Date is not a DatetimeIndex
                    if isinstance(portfolio_norm["Date"], pd.DatetimeIndex):
                        portfolio_norm = portfolio_norm.reset_index(drop=True)
                        portfolio_norm["Date"] = pd.Series(portfolio_norm["Date"].to_pydatetime())
                    
                    # Use a safer approach to combine data
                    all_dfs = [portfolio_norm] + bench_dfs
                    combined = pd.concat(all_dfs, ignore_index=True)
                    
                    # Sort by Date for proper visualization
                    combined = combined.sort_values("Date")
                except Exception as e:
                    st.error(f"Error combining performance data: {str(e)}")
                    st.error(f"Details: {str(e)}")
                    combined = pd.DataFrame(columns=["Date", "Normalized", "Index"])

                fig = px.line(
                    combined,
                    x="Date",
                    y="Normalized",
                    color="Index",
                    height=500,
                    template="plotly_white"
                ).update_layout(
                    xaxis_title="Market Hours" if period == "1d" else "Date",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    yaxis_title="Normalized Performance (%)"
                )
                
                # Format x-axis based on period and ensure proper timezone display
                if period == "1d":
                    # For intraday, show hours and minutes
                    fig.update_xaxes(
                        tickformat="%H:%M",
                        title=f"Market Hours ({MARKET_TIMEZONE.zone})"
                    )
                elif period in ["1wk", "1mo", "3mo"]:
                    # For shorter periods, show month and day
                    fig.update_xaxes(tickformat="%b %d")
                else:
                    # For longer periods, show month and year
                    fig.update_xaxes(tickformat="%b %Y")
                
                st.plotly_chart(fig, use_container_width=True)

            # Portfolio Insights
            st.subheader("🧠 Portfolio Insights")
            insights = []
            
            if portfolio_end_value > 0:
                # Merge price data
                price_df = pd.DataFrame(price_data)
                df = df.merge(price_df, on="Ticker")
                df["Market Value"] = df["Quantity"] * df["Current Price"]
                df["Weight"] = df["Market Value"] / portfolio_end_value

                # Concentration risk
                heavy = df[df.Weight > 0.1]
                for _, row in heavy.iterrows():
                    insights.append(f"⚠️ **{row.Ticker}** ({row.Weight:.1%}) exceeds 10% allocation")

                # Cash position
                cash = df[df["Asset Class"] == "Money Market"]["Market Value"].sum()
                cash_pct = cash / portfolio_end_value
                if cash_pct > 0.15:
                    insights.append(f"🪙 Cash allocation ({cash_pct:.1%}) may create drag")

                # Big movers
                for _, row in df.iterrows():
                    if row["Asset Class"] == "Money Market":
                        continue
                    try:
                        result = fetch_market_data(row.Ticker, row.Ticker, period)
                        if result and "pct_change" in result:
                            change = result["pct_change"]
                            if change <= -10:
                                insights.append(f"🔻 **{row.Ticker}** dropped {abs(change):.1f}%")
                            elif change >= 20:
                                insights.append(f"🚀 **{row.Ticker}** gained {change:.1f}%")
                    except Exception:
                        pass

                # Earnings alerts
                for ticker in df.Ticker.unique():
                    if is_money_market(ticker):
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
                    except Exception:
                        pass

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
