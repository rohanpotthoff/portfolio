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

# ==============================================
# Helper Functions
# ==============================================
def get_market_times():
    """Get proper market open/close times based on current time"""
    now = datetime.datetime.now(MARKET_TIMEZONE)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Handle weekends
    if now.weekday() >= 5:  # Saturday or Sunday
        last_friday = now - datetime.timedelta(days=(now.weekday() - 4) % 7)
        market_open = last_friday.replace(hour=9, minute=30)
        market_close = last_friday.replace(hour=16, minute=0)
    # Handle after hours
    elif now > market_close:
        market_open = market_open + datetime.timedelta(days=1)
        market_close = market_close + datetime.timedelta(days=1)
        
    return market_open, market_close

@lru_cache(maxsize=128)
def fetch_market_data(label, symbol, period):
    """Fetch market data with proper column handling"""
    try:
        # Handle cryptocurrency 24/7 trading
        is_crypto = any(x in symbol for x in ["-USD", "-EUR"])
        
        if period == "1d" and not is_crypto:
            market_open, market_close = get_market_times()
            ticker = yf.Ticker(symbol)
            
            # Handle European indices
            if "^STOXX50E" in symbol:
                hist = ticker.history(
                    start=market_open - datetime.timedelta(hours=6),
                    end=market_close - datetime.timedelta(hours=6),
                    interval="5m",
                    prepost=False
                )
                hist.index = hist.index.tz_convert("Europe/Berlin").tz_localize(None)
            else:
                hist = ticker.history(
                    start=market_open,
                    end=market_close,
                    interval="5m",
                    prepost=False
                )
                hist.index = hist.index.tz_localize(None)
        else:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty and hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)

        # Standardize column names
        hist = hist.reset_index().rename(columns={
            'index': 'Date',
            'Datetime': 'Date'
        })
        
        if hist.empty or len(hist) < 2:
            return None
            
        # Calculate percentage change
        base_price = hist["Close"].iloc[0]
        hist["Normalized"] = (hist["Close"] / base_price - 1) * 100
        
        return {
            "data": hist[["Date", "Normalized"]],
            "pct_change": (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100,
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
    st.caption("Version 4.3 | Created by Rohan Potthoff")
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
        - **v4.3**: Fixed column handling, crypto support, timezone issues
        - **v4.2**: Enhanced error handling and data validation
        """)
    
    with st.sidebar.expander("🔧 Filters & Settings", expanded=True):
        selected_period = st.selectbox(
            "Performance Period", 
            list(PERIOD_MAP.keys()), 
            index=0
        )
        st.caption("Tip: Money market funds valued at $1.00")
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
                
                # Check for duplicates
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

            for ticker in df.Ticker.unique():
                try:
                    if is_money_market(ticker):
                        qty = df[df.Ticker == ticker].Quantity.sum()
                        portfolio_start_value += qty * 1.0
                        portfolio_end_value += qty * 1.0
                        price_data.append({
                            "Ticker": ticker,
                            "Current Price": 1.0,
                            "Sector": "Cash",
                            "Asset Class": "Money Market"
                        })
                        continue

                    result = fetch_market_data(ticker, ticker, period)
                    if not result:
                        raise ValueError(f"No data found for {ticker}")
                        
                    hist = result["data"]
                    qty = df[df.Ticker == ticker].Quantity.sum()
                    
                    # Calculate values
                    start_price = hist.Close.iloc[0] if "Close" in hist else 1.0
                    current_price = hist.Close.iloc[-1] if "Close" in hist else 1.0
                    
                    portfolio_start_value += qty * start_price
                    portfolio_end_value += qty * current_price

                    # Store performance data
                    if not hist.empty:
                        if portfolio_history.empty:
                            portfolio_history = hist[["Date"]].copy()
                        portfolio_history[ticker] = hist["Normalized"]

                    # Get stock info
                    info = yf.Ticker(ticker).info
                    price_data.append({
                        "Ticker": ticker,
                        "Current Price": current_price,
                        "Sector": info.get("sector", "Unknown"),
                        "Asset Class": info.get("quoteType", "Stock").title()
                    })

                except Exception as e:
                    st.warning(f"Skipping {ticker}: {str(e)}")
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
                portfolio_norm = portfolio_history.mean(axis=1).to_frame(name="Normalized")
                portfolio_norm["Date"] = portfolio_history["Date"]
                portfolio_norm["Index"] = "My Portfolio"

                # Combine benchmark data
                if benchmark_series:
                    bench_dfs = [pd.DataFrame(b["data"]).assign(Index=b["label"]) 
                               for b in benchmark_series]
                    combined = pd.concat([portfolio_norm] + bench_dfs)
                else:
                    combined = portfolio_norm

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
                
                if period == "1d":
                    fig.update_xaxes(tickformat="%H:%M")
                st.plotly_chart(fig, use_container_width=True)

            # Portfolio Insights
            st.subheader("🧠 Portfolio Insights")
            insights = []
            
            if portfolio_end_value > 0:
                # Merge price data
                price_df = pd.DataFrame(price_data)
                df = df.merge(price_df, on="Ticker")
                df["Market Value"] = df.Quantity * df["Current Price"]
                df["Weight"] = df["Market Value"] / portfolio_end_value

                # Concentration risk
                heavy = df[df.Weight > 0.1]
                for _, row in heavy.iterrows():
                    insights.append(f"⚠️ **{row.Ticker}** ({row.Weight:.1%}) exceeds 10% allocation")

                # Cash position
                cash = df[df["Asset Class"] == "Money Market"].MarketValue.sum()
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
                        if not cal.empty:
                            earnings_date = cal.EarningsDate.max()
                            if pd.notna(earnings_date):
                                days = (earnings_date.date() - datetime.date.today()).days
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
                    sector_group = df.groupby("Sector", as_index=False).MarketValue.sum()
                    fig = px.pie(sector_group, values="MarketValue", names="Sector", 
                               title="Sector Allocation", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)

                with viz_cols[1]:
                    asset_group = df.groupby("Asset Class", as_index=False).MarketValue.sum()
                    if len(asset_group) > 1:
                        fig = px.pie(asset_group, values="MarketValue", names="Asset Class", 
                                   title="Asset Classes", hole=0.3)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Diversify assets for breakdown")

                with viz_cols[2]:
                    if "Account" in df and df.Account.nunique() > 1:
                        account_group = df.groupby("Account", as_index=False).MarketValue.sum()
                        fig = px.pie(account_group, values="MarketValue", names="Account", 
                                   title="Account Distribution", hole=0.3)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Add 'Account' column for breakdown")

                # Raw data export
                with st.expander("🔍 Raw Portfolio Data"):
                    st.dataframe(df.sort_values("MarketValue", ascending=False))
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
