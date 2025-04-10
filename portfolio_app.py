import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import os
import io

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Tracker Pro", layout="wide")

# Constants
MONEY_MARKET_TICKERS = ["WMPXX", "FNSXX", "VMFXX", "SPAXX"]  # Add others as needed
COMPARISON_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E"
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
- **v3.1**  
  - Added special handling for money market funds (XX tickers)  
  - Modernized performance summary display  
  - Improved error handling for missing price data  
  - Added visual spacing between sections  
  - Fixed normalization data length bugs  

- **v3.0**  
  - Portfolio insights with alerts system  
  - Benchmark comparison charts  
  - Sector/asset class breakdowns  
  - Historical snapshot tracking  
    """)

# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def clean_portfolio(df):
    """Clean and standardize portfolio dataframe"""
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df[df['Ticker'].str.upper() != "CUR:USD"]
    df['Ticker'] = df['Ticker'].str.replace("CUR:GE", "GE")
    return df

def save_daily_snapshot(df):
    """Save portfolio snapshot for historical tracking"""
    today = datetime.date.today().isoformat()
    snapshot_path = "portfolio_history.csv"
    df["Date"] = today
    
    if os.path.exists(snapshot_path):
        prev = pd.read_csv(snapshot_path)
        combined = pd.concat([prev, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(snapshot_path, index=False)

def is_money_market(ticker):
    """Check if ticker is a money market fund"""
    return ticker in MONEY_MARKET_TICKERS or "XX" in ticker

# ────────────────────────────────────────────────────────────────────────────────
# UI Header
# ────────────────────────────────────────────────────────────────────────────────
st.title("📊 Portfolio Tracker Pro")
st.caption("Version 3.1 | Tracking your investments with precision")

# ────────────────────────────────────────────────────────────────────────────────
# File Upload
# ────────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload your portfolio holdings (CSV/Excel)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True,
    help="Files should contain 'Ticker' and 'Quantity' columns"
)

# ────────────────────────────────────────────────────────────────────────────────
# Filters
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🔧 Filters & Settings", expanded=True):
    selected_period = st.selectbox(
        "Performance Period", 
        list(PERIOD_MAP.keys()), 
        index=5  # Default to YTD
    )
    st.markdown("---")
    st.caption("Tip: Money market funds are automatically valued at $1.00")

# ────────────────────────────────────────────────────────────────────────────────
# Main Processing
# ────────────────────────────────────────────────────────────────────────────────
if uploaded_files:
    # ────────────────────────────────────────────────────────────────────────────
    # Data Loading & Cleaning
    # ────────────────────────────────────────────────────────────────────────────
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
    save_daily_snapshot(df.copy())
    
    if not all(col in df.columns for col in ["Ticker", "Quantity"]):
        st.error("Uploaded files must contain at least 'Ticker' and 'Quantity' columns")
        st.stop()

    st.success("✅ Portfolio data loaded successfully")
    st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────────
    # Data Processing
    # ────────────────────────────────────────────────────────────────────────────
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
                # Special handling for money market funds
                current_price = 1.0
                start_price = 1.0
                end_price = 1.0
                hist = None
                sector = "Cash Equivalent"
            else:
                # Regular security processing
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

                # Determine sector/asset class
                sector = info.get("sector", "")
                if not sector:
                    if "-USD" in ticker:
                        sector = "Cryptocurrency"
                    elif info.get("quoteType") in ["ETF", "MUTUALFUND"]:
                        sector = "Fund"
                    else:
                        sector = "Unknown"

            # Handle cases where we still don't have prices
            if start_price is None or end_price is None:
                st.warning(f"⚠️ Missing price data for {ticker} - using $1.00")
                current_price = 1.0
                start_price = 1.0
                end_price = 1.0

            # Calculate position values
            df_ticker = df[df["Ticker"] == ticker]
            quantity = df_ticker["Quantity"].sum()
            portfolio_start_value += quantity * start_price
            portfolio_end_value += quantity * end_price

            # Normalization logic
            if hist is not None and not hist.empty:
                norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                if portfolio_normalized is None:
                    portfolio_normalized = norm_price * quantity
                else:
                    portfolio_normalized += norm_price * quantity
            elif benchmark_series:
                # For money market funds or missing data, assume flat performance
                flat_series = pd.Series([100] * len(benchmark_series[0])) * quantity
                if portfolio_normalized is None:
                    portfolio_normalized = flat_series
                else:
                    portfolio_normalized += flat_series

            price_data.append({
                "Ticker": ticker, 
                "Current Price": end_price,
                "Sector": sector
            })

        except Exception as e:
            st.warning(f"⚠️ Error processing {ticker}: {str(e)} - using $1.00")
            price_data.append({
                "Ticker": ticker, 
                "Current Price": 1.0,
                "Sector": "Unknown"
            })

    # Calculate portfolio performance
    if portfolio_start_value > 0:
        portfolio_change = (portfolio_end_value / portfolio_start_value - 1) * 100
        if portfolio_normalized is not None and benchmark_series:
            try:
                portfolio_normalized = pd.DataFrame({
                    "Date": benchmark_series[0]["Date"],
                    "Normalized Price": portfolio_normalized / portfolio_start_value * 100,
                    "Index": "My Portfolio"
                })
            except Exception as e:
                st.warning(f"⚠️ Could not normalize portfolio data: {str(e)}")
                portfolio_normalized = None

    # Merge price data with holdings
    price_df = pd.DataFrame(price_data)
    df = df.merge(price_df, on="Ticker", how="left")
    df["Market Value"] = df["Quantity"] * df["Current Price"]
    total_value = df["Market Value"].sum()

    # Account filtering
    if "Account" in df.columns:
        accounts = df["Account"].dropna().unique().tolist()
        selected_accounts = st.sidebar.multiselect(
            "Filter Accounts", 
            accounts, 
            default=accounts,
            help="Show only selected accounts"
        )
        df = df[df["Account"].isin(selected_accounts)]

    # ────────────────────────────────────────────────────────────────────────────
    # Portfolio Insights
    # ────────────────────────────────────────────────────────────────────────────
    st.subheader("🔍 Portfolio Insights")
    insights = []

    # Overconcentration check
    top_holdings = df.sort_values("Market Value", ascending=False)
    overweights = top_holdings[top_holdings["Market Value"] / total_value > 0.10]
    for _, row in overweights.iterrows():
        pct = row["Market Value"] / total_value * 100
        insights.append(f"⚠️ **{row['Ticker']}** is {pct:.1f}% of portfolio (over 10%)")

    # Cash drag check
    if "Asset Class" in df.columns:
        cash_assets = df[df["Asset Class"].str.contains(
            "Money Market|Cash", case=False, na=False
        )]
        cash_pct = cash_assets["Market Value"].sum() / total_value * 100
        if cash_pct > 15:
            insights.append(f"🪙 You have {cash_pct:.1f}% in cash/money markets (>15%)")

    # Big movers
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        try:
            if not is_money_market(ticker):  # Skip for money market funds
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
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        try:
            if not is_money_market(ticker):  # Skip for money market funds
                cal = yf.Ticker(ticker).calendar
                if not cal.empty:
                    earnings_date = cal.loc["Earnings Date"].max()
                    if pd.notna(earnings_date):
                        days = (earnings_date - pd.Timestamp.today()).days
                        if 0 <= days <= 14:
                            insights.append(
                                f"📅 **{ticker}** earnings in {days} days "
                                f"(~{earnings_date.strftime('%b %d')})"
                            )
        except Exception:
            pass

    if insights:
        with st.expander("📌 Active Alerts", expanded=True):
            for note in insights:
                st.markdown(f"- {note}")
        
        # Export insights
        insights_text = "\n".join(insights)
        st.download_button(
            "💾 Download Insights", 
            insights_text, 
            file_name=f"portfolio_alerts_{datetime.date.today()}.txt"
        )
    else:
        st.success("🎉 No alerts - your portfolio looks healthy!")
    
    st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────────
    # Performance Summary
    # ────────────────────────────────────────────────────────────────────────────
    st.subheader("📈 Performance Metrics")
    
    # Create metrics in a modern layout
    cols = st.columns(len(COMPARISON_TICKERS) + 1)
    metrics = [("My Portfolio", portfolio_change)] + [
        (label, benchmark_data.get(label)) 
        for label in COMPARISON_TICKERS
    ]
    
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            if value is not None:
                color = "green" if value >= 0 else "red"
                arrow = "↑" if value >= 0 else "↓"
                st.metric(
                    label,
                    f"{abs(value):.2f}%",
                    delta=f"{arrow} {abs(value):.2f}%",
                    delta_color="normal",
                    help=f"{label} {selected_period} performance"
                )
                st.markdown(
                    f"<style>div[data-testid='stMetricDelta'] svg {{color: {color};}}</style>",
                    unsafe_allow_html=True
                )
            else:
                st.metric(label, "N/A")

    st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────────
    # Performance Chart
    # ────────────────────────────────────────────────────────────────────────────
    if benchmark_series:
        try:
            all_series = pd.concat(benchmark_series)
            if portfolio_normalized is not None:
                # Align dates
                min_date = min(
                    all_series["Date"].min(),
                    portfolio_normalized["Date"].min()
                )
                max_date = max(
                    all_series["Date"].max(),
                    portfolio_normalized["Date"].max()
                )
                
                all_series = all_series[
                    (all_series["Date"] >= min_date) & 
                    (all_series["Date"] <= max_date)
                ]
                portfolio_normalized = portfolio_normalized[
                    (portfolio_normalized["Date"] >= min_date) & 
                    (portfolio_normalized["Date"] <= max_date)
                ]
                
                all_series = pd.concat([all_series, portfolio_normalized])
            
            fig = px.line(
                all_series, 
                x="Date", 
                y="Normalized Price", 
                color="Index", 
                title=f"Normalized Performance ({selected_period})",
                template="plotly_white",
                height=500
            )
            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Could not generate performance chart: {str(e)}")
    
    st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────────
    # Portfolio Overview
    # ────────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Portfolio Composition")
    
    # Summary metrics
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
        fig_sector = px.pie(
            sector_group, 
            values="Market Value", 
            names="Sector", 
            title="By Sector",
            hole=0.3
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with viz_cols[1]:
        if "Asset Class" in df.columns:
            asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
            fig_asset = px.pie(
                asset_group, 
                values="Market Value", 
                names="Asset Class", 
                title="By Asset Class",
                hole=0.3
            )
            st.plotly_chart(fig_asset, use_container_width=True)
        else:
            st.info("No asset class data available")

    with viz_cols[2]:
        if "Account" in df.columns:
            account_group = df.groupby("Account")["Market Value"].sum().reset_index()
            fig_account = px.pie(
                account_group, 
                values="Market Value", 
                names="Account", 
                title="By Account",
                hole=0.3
            )
            st.plotly_chart(fig_account, use_container_width=True)
        else:
            st.info("No account data available")

    # Raw data
    with st.expander("🔍 View Raw Portfolio Data", expanded=False):
        st.dataframe(df.sort_values("Market Value", ascending=False))

else:
    st.info("📤 Upload portfolio files to begin analysis")
    st.markdown("""
    ### Expected file format:
    - CSV or Excel format
    - Must contain columns: `Ticker` and `Quantity`
    - Optional columns: `Account`, `Asset Class`
    
    💡 *Tip: You can upload multiple files from different accounts*
    """)
