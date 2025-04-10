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
    
    # Handle after hours
    if now > market_close:
        market_open = market_open - datetime.timedelta(days=1)
        market_close = market_close - datetime.timedelta(days=1)
    elif now < market_open:
        market_open = market_open - datetime.timedelta(days=1)
        market_close = market_close - datetime.timedelta(days=1)
        
    return market_open, market_close

@lru_cache(maxsize=128)
def fetch_market_data(label, symbol, period):
    """Fetch market data with proper intraday handling"""
    try:
        if period == "1d":
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
                
            hist = hist[~hist.index.duplicated(keep='last')]
            
        else:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            else:
                hist.index = hist.index.tz_localize(None)
        
        if hist.empty or len(hist) < 2:
            return None
            
        # Calculate percentage change from first data point
        base_price = hist["Close"].iloc[0]
        hist["Normalized"] = (hist["Close"] / base_price - 1) * 100
        
        return {
            "data": hist[["Normalized"]].reset_index(),
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
    st.caption("Version 4.1 | Created by Rohan Potthoff")
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
        - **v4.1**: Fixed normalization, intraday handling, Euro Stoxx timing
        - **v4.0**: Enhanced metrics, treasury benchmark, raw data export
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
            dataframes = []
            tickers_seen = set()
            duplicate_tickers = set()

            # Process uploaded files
            for uploaded_file in uploaded_files:
                file_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                cleaned_df = clean_portfolio(file_df)
                
                if 'Ticker' in cleaned_df.columns:
                    current_tickers = set(cleaned_df['Ticker'])
                    duplicates = current_tickers & tickers_seen
                    duplicate_tickers.update(duplicates)
                    tickers_seen.update(current_tickers)
                
                dataframes.append(cleaned_df)

            if duplicate_tickers:
                with st.expander("⚠️ Duplicate Tickers Detected", expanded=True):
                    st.warning(f"Duplicate tickers: {', '.join(duplicate_tickers)}")

            df = pd.concat(dataframes, ignore_index=True)
            st.success("✅ Portfolio data loaded successfully")

            # Process holdings
            price_data = []
            portfolio_start_value = 0
            portfolio_end_value = 0
            portfolio_history = pd.DataFrame()

            for ticker in df["Ticker"].unique():
                try:
                    if is_money_market(ticker):
                        current_price = 1.0
                        start_price = 1.0
                        hist = None
                        sector = "Cash"
                        asset_class = "Money Market"
                    else:
                        stock = yf.Ticker(ticker)
                        result = fetch_market_data(ticker, ticker, period)
                        
                        if result:
                            hist = result["data"]
                            start_price = stock.history(period=period)["Close"].iloc[0]
                            current_price = stock.history(period=period)["Close"].iloc[-1]
                            sector = stock.info.get("sector", "Unknown")
                            asset_class = stock.info.get("quoteType", "Stock").title()
                            
                            # Store for portfolio performance
                            if portfolio_history.empty:
                                portfolio_history = hist[["Date"]].copy()
                            portfolio_history[ticker] = hist["Normalized"]
                        else:
                            start_price = 1.0
                            current_price = 1.0

                    qty = df[df["Ticker"] == ticker]["Quantity"].sum()
                    portfolio_start_value += qty * start_price
                    portfolio_end_value += qty * current_price

                    price_data.append({
                        "Ticker": ticker, 
                        "Current Price": current_price,
                        "Sector": sector,
                        "Asset Class": asset_class
                    })

                except Exception as e:
                    st.warning(f"Skipping {ticker}: {str(e)}")
                    continue

            # Calculate portfolio performance
            if not portfolio_history.empty:
                portfolio_history["Portfolio"] = portfolio_history.drop("Date", axis=1).mean(axis=1)
                portfolio_data = {
                    "data": portfolio_history[["Date", "Portfolio"]].rename(columns={"Portfolio": "Normalized"}),
                    "pct_change": (portfolio_end_value / portfolio_start_value - 1) * 100,
                    "label": "My Portfolio"
                }
            else:
                portfolio_data = None

            # Create unified chart
            all_data = benchmark_series.copy()
            if portfolio_data:
                all_data.append(portfolio_data)

            chart_data = []
            for item in all_data:
                df = item["data"].copy()
                df["Index"] = item["label"]
                chart_data.append(df)
            
            combined = pd.concat(chart_data).reset_index(drop=True)

            # Performance Metrics
            st.subheader("📈 Performance Metrics")
            cols = st.columns(1 + len(COMPARISON_TICKERS))
            
            # Portfolio metric
            with cols[0]:
                if portfolio_data:
                    delta_color = "normal" if portfolio_data["pct_change"] >= 0 else "inverse"
                    st.metric(
                        "My Portfolio",
                        f"${portfolio_end_value:,.2f}",
                        f"{portfolio_data['pct_change']:.2f}%",
                        delta_color=delta_color
                    )
            
            # Benchmark metrics
            for i, (label, value) in enumerate(benchmark_values.items()):
                with cols[i+1]:
                    st.metric(
                        label,
                        "",
                        f"{value:.2f}%" if value is not None else "N/A"
                    )

            st.markdown("---")

            # Performance Visualization
            st.subheader("📉 Performance Comparison")
            if not combined.empty:
                fig = px.line(
                    combined,
                    x="Date",
                    y="Normalized",
                    color="Index",
                    title="",
                    template="plotly_white",
                    height=500
                ).update_layout(
                    xaxis_title="Market Hours" if period == "1d" else "Date",
                    hovermode="x unified",
                    legend_title_text="",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    yaxis_title="Performance (%)"
                )
                
                if period == "1d":
                    fig.update_xaxes(
                        tickformat="%H:%M",
                        rangeslider_visible=True
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate performance chart")

            st.markdown("---")

            # Portfolio Insights
            st.subheader("🧠 Portfolio Insights")
            insights = []
            
            if portfolio_end_value > 0:
                # Concentration risk
                df["Weight"] = df["Market Value"] / portfolio_end_value
                heavy = df[df["Weight"] > 0.1]
                for _, row in heavy.iterrows():
                    insights.append(f"⚠️ **{row['Ticker']}** ({row['Weight']:.1%}) exceeds 10% allocation")

                # Cash position
                cash = df[df["Asset Class"] == "Money Market"]["Market Value"].sum()
                cash_pct = cash / portfolio_end_value
                if cash_pct > 0.15:
                    insights.append(f"🪙 Cash allocation ({cash_pct:.1%}) may create drag")

            # Price movements
            for ticker in df["Ticker"].unique():
                if is_money_market(ticker):
                    continue
                
                try:
                    result = fetch_market_data(ticker, ticker, period)
                    if result and len(result["data"]) > 1:
                        change = result["pct_change"]
                        if change <= -10:
                            insights.append(f"🔻 **{ticker}** dropped {abs(change):.1f}%")
                        elif change >= 20:
                            insights.append(f"🚀 **{ticker}** gained {change:.1f}%")
                except Exception:
                    pass

            # Earnings alerts
            for ticker in df["Ticker"].unique():
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
                insight_order = {"⚠️": 1, "🪙": 2, "🔻": 3, "🚀": 4, "📅": 5}
                insights.sort(key=lambda x: insight_order.get(x[:2], 6))
                
                with st.expander("📌 Active Alerts (Top 5)", expanded=True):
                    for note in insights[:5]:
                        st.markdown(f"- {note}")
                    
                    if len(insights) > 5:
                        st.markdown("**Additional alerts:**")
                        for note in insights[5:10]:  # Show max 10 total
                            st.markdown(f"- {note}")
                    
                    st.download_button(
                        "💾 Download Insights", 
                        "\n".join(insights), 
                        file_name=f"portfolio_alerts_{datetime.date.today()}.txt"
                    )
            else:
                st.success("🎉 No alerts - portfolio looks healthy!")

            st.markdown("---")

            # Portfolio Composition
            st.subheader("📊 Portfolio Composition")
            
            # Merge price data
            price_df = pd.DataFrame(price_data)
            df = df.merge(price_df, on="Ticker", how="left")
            df["Market Value"] = df["Quantity"] * df["Current Price"]
            total_value = df["Market Value"].sum()

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Holdings", len(df))
            with col3:
                accounts = df["Account"].nunique() if "Account" in df.columns else 1
                st.metric("Accounts", accounts)

            # Visualizations
            viz_cols = st.columns(3)
            with viz_cols[0]:
                sector_group = df.groupby("Sector")["Market Value"].sum().reset_index()
                fig = px.pie(sector_group, values="Market Value", names="Sector", 
                            title="Sector Allocation", hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

            with viz_cols[1]:
                asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
                if len(asset_group) > 1:
                    fig = px.pie(asset_group, values="Market Value", names="Asset Class", 
                                title="Asset Class Distribution", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Diversify assets for breakdown")

            with viz_cols[2]:
                if "Account" in df.columns and df["Account"].nunique() > 1:
                    account_group = df.groupby("Account")["Market Value"].sum().reset_index()
                    fig = px.pie(account_group, values="Market Value", names="Account", 
                                title="Account Distribution", hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Multiple accounts needed")

            # Raw data export
            with st.expander("🔍 View Raw Portfolio Data", expanded=False):
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
