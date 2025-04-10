import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import os

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
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
    "1W": "7d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "5Y": "5y"
}

# ────────────────────────────────────────────────────────────────────────────────
# Version History
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("📦 Version History", expanded=False):
    st.markdown("""
- **v4.0**  
  - Renamed to Portfolio Tracker  
  - Added 10Y Treasury benchmark  
  - Enhanced performance metrics display  
  - Individual stock performance comparison  
  - Improved insights prioritization  
  - Raw data download capability  

- **v3.0**  
  - Portfolio insights system  
  - Benchmark comparisons  
  - Sector/asset breakdowns  
  - Earnings alerts  
  - Money market fund support  
    """)

# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# UI Header
# ────────────────────────────────────────────────────────────────────────────────
st.title("📊 Portfolio Tracker")
st.caption("Version 4.0 | Created by Rohan Potthoff")

# Social icons and divider
st.markdown("""
<style>
.social-icons {
    display: flex;
    gap: 15px;
    margin-top: -10px;
    margin-bottom: 10px;
}
.social-icons a {
    color: #9e9e9e !important;
    text-decoration: none;
    font-size: 14px;
}
.social-icons a:hover {
    color: #1DA1F2 !important;
}
</style>
<div class="social-icons">
    <a href="mailto:rohanpotthoff@gmail.com">✉️ Email</a>
    <a href="https://www.linkedin.com/in/rohanpotthoff" target="_blank">🔗 LinkedIn</a>
</div>
<hr style='margin-bottom:20px'>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# File Upload
# ────────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload portfolio holdings (CSV/Excel)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True,
    help="Required columns: Ticker, Quantity | Optional: Account"
)

# ────────────────────────────────────────────────────────────────────────────────
# Filters
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🔧 Filters & Settings", expanded=True):
    selected_period = st.selectbox(
        "Performance Period", 
        list(PERIOD_MAP.keys()), 
        index=0  # Default to Today
    )
    st.caption("Tip: Money market funds are automatically valued at $1.00")

# ────────────────────────────────────────────────────────────────────────────────
# Main Processing
# ────────────────────────────────────────────────────────────────────────────────
if uploaded_files:
    # Data Loading & Cleaning
    dataframes = []
    tickers_seen = set()
    duplicate_tickers = set()

    for uploaded_file in uploaded_files:
        try:
            df = (pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") 
                 else pd.read_excel(uploaded_file))
            df = clean_portfolio(df)
            
            if 'Ticker' in df.columns:
                dup = set(df['Ticker'].unique()).intersection(tickers_seen)
                duplicate_tickers.update(dup)
                tickers_seen.update(df['Ticker'].unique())
                dataframes.append(df)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    if duplicate_tickers:
        with st.expander("⚠️ Duplicate Tickers Detected", expanded=True):
            st.warning(f"These tickers appear in multiple files: {', '.join(duplicate_tickers)}")

    if not dataframes:
        st.error("No valid portfolio data found in uploaded files")
        st.stop()

    df = pd.concat(dataframes, ignore_index=True)
    st.success("✅ Portfolio data loaded successfully")
    
    # Data Processing
    period = PERIOD_MAP[selected_period]
    benchmark_data = {}
    benchmark_series = []
    portfolio_change = None
    portfolio_normalized = None

    # Get benchmark data
    for label, symbol in COMPARISON_TICKERS.items():
        try:
            ticker_obj = yf.Ticker(symbol)
            
            if period == "1d":
                info = ticker_obj.info
                open_price = info.get("regularMarketOpen")
                price = info.get("regularMarketPrice")
                pct_change = (price / open_price - 1) * 100 if open_price else 0
                benchmark_data[label] = pct_change
                hist = ticker_obj.history(period="2d", interval="5m")
                hist = hist[hist.index.date == pd.Timestamp.today().date()]
            else:
                hist = ticker_obj.history(period=period)
                pct_change = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100 
                             if not hist.empty else 0)
                benchmark_data[label] = pct_change

            if not hist.empty:
                norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                benchmark_series.append(pd.DataFrame({
                    "Date": hist.index,
                    "Normalized Price": norm_price,
                    "Index": label
                }))
        except Exception as e:
            st.warning(f"Error fetching {label} data: {str(e)}")
            benchmark_data[label] = None

    # Process portfolio holdings
    tickers = df["Ticker"].unique().tolist()
    price_data = []
    portfolio_start_value = 0
    portfolio_end_value = 0

    for ticker in tickers:
        try:
            if is_money_market(ticker):
                current_price = 1.0
                start_price = 1.0
                end_price = 1.0
                hist = None
                sector = "Cash Equivalent"
                asset_class = "Money Market"
            else:
                stock = yf.Ticker(ticker)
                
                if period == "1d":
                    info = stock.info
                    open_price = info.get("regularMarketOpen")
                    price = info.get("regularMarketPrice")
                    hist = stock.history(period="2d", interval="5m")
                    hist = hist[hist.index.date == pd.Timestamp.today().date()]
                    start_price = open_price if open_price is not None else price
                    end_price = price if price is not None else open_price
                else:
                    hist = stock.history(period=period)
                    info = stock.info
                    price = info.get("regularMarketPrice") or info.get("currentPrice")
                    start_price = hist["Close"].iloc[0] if not hist.empty else price
                    end_price = hist["Close"].iloc[-1] if not hist.empty else price

                sector = info.get("sector", "Unknown")
                asset_class = info.get("quoteType", "Stock").title()
                asset_class = "ETF" if asset_class == "Etf" else asset_class

            df_ticker = df[df["Ticker"] == ticker]
            quantity = df_ticker["Quantity"].sum()
            portfolio_start_value += quantity * start_price
            portfolio_end_value += quantity * end_price

            if hist is not None and not hist.empty:
                norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                if portfolio_normalized is None:
                    portfolio_normalized = norm_price * quantity
                else:
                    portfolio_normalized += norm_price * quantity

            price_data.append({
                "Ticker": ticker, 
                "Current Price": end_price,
                "Sector": sector,
                "Asset Class": asset_class
            })

        except Exception as e:
            st.warning(f"Error processing {ticker}: {str(e)}")
            price_data.append({
                "Ticker": ticker, 
                "Current Price": 1.0,
                "Sector": "Unknown",
                "Asset Class": "Unknown"
            })

    # Calculate portfolio performance
    if portfolio_start_value > 0:
        portfolio_change = (portfolio_end_value / portfolio_start_value - 1) * 100
        if portfolio_normalized is not None and benchmark_series:
            portfolio_normalized = pd.DataFrame({
                "Date": benchmark_series[0]["Date"],
                "Normalized Price": portfolio_normalized / portfolio_start_value * 100,
                "Index": "My Portfolio"
            })

    # Merge price data with holdings
    price_df = pd.DataFrame(price_data)
    df = df.merge(price_df, on="Ticker", how="left")
    df["Market Value"] = df["Quantity"] * df["Current Price"]
    total_value = df["Market Value"].sum()

    # Portfolio Insights
    st.subheader("🧠 Portfolio Insights")
    insights = []

    # Overconcentration check
    top_holdings = df.sort_values("Market Value", ascending=False)
    overweights = top_holdings[top_holdings["Market Value"] / total_value > 0.10]
    for _, row in overweights.iterrows():
        pct = row["Market Value"] / total_value * 100
        insights.append(f"⚠️ **{row['Ticker']}** is {pct:.1f}% of portfolio (over 10%)")

    # Cash drag check
    cash_assets = df[df["Asset Class"].str.contains("Money Market|Cash", case=False, na=False)]
    cash_pct = cash_assets["Market Value"].sum() / total_value * 100
    if cash_pct > 15:
        insights.append(f"🪙 You have {cash_pct:.1f}% in cash/money markets (>15%)")

    # Big movers
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        try:
            if not is_money_market(ticker):
                hist = yf.Ticker(ticker).history(period=period)
                if not hist.empty:
                    change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    if change <= -10:
                        insights.append(f"🔻 **{ticker}** dropped {abs(change):.1f}% ({selected_period})")
                    if change >= 10:
                        insights.append(f"🚀 **{ticker}** gained {change:.1f}% ({selected_period})")
        except Exception:
            pass

    # Earnings alerts
    earnings_tickers = set()
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        try:
            if not is_money_market(ticker):
                cal = yf.Ticker(ticker).calendar
                if not cal.empty:
                    earnings_date = cal.loc["Earnings Date"].max()
                    if pd.notna(earnings_date):
                        days = (earnings_date - pd.Timestamp.today()).days
                        if 0 <= days <= 14:
                            earnings_tickers.add(ticker)
                            insights.append(f"📅 **{ticker}** earnings in {days} days (~{earnings_date.strftime('%b %d')})")
        except Exception:
            pass

    if insights:
        insight_order = {"⚠️": 1, "🪙": 2, "🔻": 3, "🚀": 4, "📅": 5}
        insights.sort(key=lambda x: insight_order.get(x.split()[0], 6))
        
        with st.expander("📌 Active Alerts (Top 5)", expanded=True):
            for note in insights[:5]:
                st.markdown(f"- {note}")
            
            if len(insights) > 5:
                st.markdown("**Additional alerts:**")
                for note in insights[5:]:
                    st.markdown(f"- {note}")
            
            insights_text = "\n".join(insights)
            st.download_button(
                "💾 Download Insights", 
                insights_text, 
                file_name=f"portfolio_alerts_{datetime.date.today()}.txt"
            )
    else:
        st.success("🎉 No alerts - portfolio looks healthy!")
    
    st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# Performance Summary (Corrected Placement)
# ────────────────────────────────────────────────────────────────────────────────
    st.subheader("📈 Performance Metrics")
    
    # Create two rows - top row for portfolio, bottom for benchmarks
    top_cols = st.columns(1)
    bench_cols = st.columns(len(COMPARISON_TICKERS))
    
    # Portfolio metric
    with top_cols[0]:
        if portfolio_change is not None:
            delta_arrow = "↑" if portfolio_change >= 0 else "↓" if portfolio_change < 0 else ""
            color = "#2ECC40" if portfolio_change > 0 else "#FF4136" if portfolio_change < 0 else "#AAAAAA"
            display_value = f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 24px; font-weight: bold;">{abs(portfolio_change):.2f}%</span>
                <span style="color: {color}; font-size: 24px;">{delta_arrow}</span>
            </div>
            """
            st.markdown(display_value, unsafe_allow_html=True)
            st.metric(
                label="My Portfolio",
                value="",
                delta=None,
                help=f"Portfolio {selected_period} performance"
            )
        else:
            st.metric("My Portfolio", "N/A")

    # Benchmarks
    for i, (label, value) in enumerate(benchmark_data.items()):
        with bench_cols[i]:
            if value is not None:
                delta_arrow = "↑" if value >= 0 else "↓" if value < 0 else ""
                color = "#2ECC40" if value > 0 else "#FF4136" if value < 0 else "#AAAAAA"
                display_value = f"""
                <div style="display: flex; align-items: center; gap: 4px;">
                    <span>{abs(value):.2f}%</span>
                    <span style="color: {color}; font-size: 1.1em;">{delta_arrow}</span>
                </div>
                """
                st.markdown(display_value, unsafe_allow_html=True)
                st.metric(
                    label=label,
                    value="",
                    delta=None,
                    help=f"{label} {selected_period} performance"
                )
            else:
                st.metric(label, "N/A")

    st.markdown("---")  # Keep this divider here

# ────────────────────────────────────────────────────────────────────────────────
# STRATEGIC PERFORMANCE VISUALIZATION
# Purpose: Align financial metrics with organizational objectives through 
#          temporal analysis of portfolio performance
# Key Features:
#   - Real-time intraday tracking for tactical decision making
#   - Benchmark correlation analysis for strategic positioning
#   - Resource allocation visualization for operational optimization
# ────────────────────────────────────────────────────────────────────────────────
if benchmark_series:
    # USER-CENTRIC DATA SELECTION INTERFACE
    selected_tickers = st.multiselect(
        "Add holdings to chart:", 
        options=df['Ticker'].unique(),
        default=[],
        help="Strategic comparison of assets against market indicators"
    )
    
    # OPERATIONAL EXECUTION FRAMEWORK
    for ticker in selected_tickers:
        try:
            if is_money_market(ticker):
                # LIQUIDITY POSITION MODELING
                hist = pd.DataFrame({'Close': [1.0]*len(benchmark_series[0])}, 
                                   index=benchmark_series[0]['Date'])
            else:
                # MARKET DATA ACQUISITION PIPELINE
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    period="1d", 
                    interval="5m" if period == "1d" else None,
                    prepost=False  # Focus on core trading hours
                ) if period == "1d" else stock.history(period=period)
            
            if not hist.empty:
                # TEMPORAL ALIGNMENT PROCESSING
                hist = hist.between_time('09:30', '16:00') if period == "1d" else hist
                norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                
                benchmark_series.append(pd.DataFrame({
                    "Date": hist.index,
                    "Normalized Price": norm_price,
                    "Index": ticker
                }))
        except Exception as e:
            # RISK MANAGEMENT PROTOCOL
            st.warning(f"Asset analysis limitation: {ticker} data unavailable")
            continue

    # STRATEGIC VISUALIZATION ENGINE
    try:
        # TEMPORAL NORMALIZATION FRAMEWORK
        if period == "1d":
            # INTRADAY OPERATIONS CALIBRATION
            market_hours = pd.date_range(
                start=pd.Timestamp.today().normalize() + pd.Timedelta(hours=9, minutes=30),
                end=pd.Timestamp.today().normalize() + pd.Timedelta(hours=16),
                freq='5T'
            )
            
            aligned_series = []
            for series in benchmark_series:
                aligned = series.set_index('Date').reindex(market_hours, method='ffill')
                aligned_series.append(aligned.reset_index().rename(columns={'index':'Date'}))
            
            all_series = pd.concat(aligned_series)
        else:
            # LONG-TERM STRATEGIC ALIGNMENT
            all_series = pd.concat([
                s[s['Date'] >= min(s['Date'] for s in benchmark_series)] 
                for s in benchmark_series
            ])

        # EXECUTIVE DECISION SUPPORT VISUALIZATION
        fig = px.line(
            all_series, 
            x="Date", 
            y="Normalized Price", 
            color="Index",
            title=f"Strategic Performance Alignment: {selected_period}",
            template="plotly_white",
            height=500
        ).update_layout(
            xaxis_title="Trading Hours" if period == "1d" else "Strategic Timeline",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                title_text="Performance Entities"
            )
        )
        
        # TEMPORAL FORMATTING GOVERNANCE
        if period == "1d":
            fig.update_xaxes(
                tickformat="%H:%M",
                tickvals=pd.date_range(start=market_hours[0], end=market_hours[-1], freq='1H'),
                range=[market_hours[0], market_hours[-1]]
            )
            
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        # BUSINESS CONTINUITY PROTECTION
        st.error(f"Strategic visualization system error: {str(e)}")
        st.stop()
    
    # Portfolio Overview
    st.subheader("📊 Portfolio Composition")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Holdings", len(df))
    with col3:
        st.metric("Accounts", df["Account"].nunique() if "Account" in df.columns else "1")

    # Visualizations
    viz_cols = st.columns([1.2, 1.2, 1])
    with viz_cols[0]:
        sector_group = df.groupby("Sector")["Market Value"].sum().reset_index()
        fig_sector = px.pie(sector_group, values="Market Value", names="Sector", title="By Sector", hole=0.3)
        st.plotly_chart(fig_sector, use_container_width=True)

    with viz_cols[1]:
        asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
        if len(asset_group) > 1:
            fig_asset = px.pie(asset_group, values="Market Value", names="Asset Class", title="By Asset Class", hole=0.3)
            st.plotly_chart(fig_asset, use_container_width=True)
        else:
            st.info("Add diverse assets for breakdown")

    with viz_cols[2]:
        if "Account" in df.columns and df["Account"].nunique() > 1:
            account_group = df.groupby("Account")["Market Value"].sum().reset_index()
            fig_account = px.pie(account_group, values="Market Value", names="Account", title="By Account", hole=0.3)
            st.plotly_chart(fig_account, use_container_width=True)
        else:
            st.info("Multiple accounts needed for breakdown")

    # Raw data export
    with st.expander("🔍 View Raw Portfolio Data", expanded=False):
        st.dataframe(df.sort_values("Market Value", ascending=False))
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Raw Data",
            data=csv,
            file_name=f"portfolio_data_{datetime.date.today()}.csv",
            mime="text/csv"
        )

else:
    st.info("📤 Upload portfolio files to begin analysis")
    st.markdown("""
    ### Expected format:
    - CSV/Excel with columns:  
      - **Ticker** (required)  
      - **Quantity** (required)  
      - **Account** (optional)
    """)
