import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import io

# ── App Configuration ──
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

with st.sidebar.expander("📦 Version History", expanded=False):
    st.markdown("""
- **v1.0.0.3**
  - Portfolio insights section with overconcentration, cash drag, big movers, earnings alerts
  - Hover tooltips for earnings
  - Insights export to TXT
  - Fixed string literal bug
- **v1.0.0.2**
  - Open-to-current price logic
  - Benchmark overlay fix
  - Normalization cleanup
  - Summary grid
  - ETF/Mutual/Crypto fallback
  - Duplicate file detection
- **v1.0.0.1**
  - Initial stable version with performance summary
  - Benchmark normalization
  - Sector and asset class breakdowns
    """)

# Version header with tooltip
st.title("📈 Portfolio Tracker Dashboard")
st.caption("Version 1.0.0.3")

# ── Upload Holdings ──
uploaded_files = st.file_uploader("Upload your holdings CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=True)

# ── Sidebar Filters ──
with st.sidebar.expander("🔧 Filters", expanded=True):
    period_map = {
        "Today": "1d",
        "1W": "7d",
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "YTD": "ytd",
        "1Y": "1y",
        "5Y": "5y"
    }
    selected_period = st.selectbox("Select time range", list(period_map.keys()), index=0)  
   
comparison_tickers = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E"
}

def clean_portfolio(df):
    # Capitalize columns once after full concat
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df[df['Ticker'].str.upper() != "CUR:USD"]
    df['Ticker'] = df['Ticker'].str.replace("CUR:GE", "GE")
    return df

# ── Auto-Save Snapshot for Historical Tracking ──
today = datetime.date.today().isoformat()
snapshot_path = "portfolio_history.csv"

def save_daily_snapshot(df):
    df["Date"] = today
    if os.path.exists(snapshot_path):
        prev = pd.read_csv(snapshot_path)
        combined = pd.concat([prev, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(snapshot_path, index=False)

if uploaded_files:
    dataframes = []
    tickers_seen = set()
    duplicate_tickers = set()

    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = clean_portfolio(df)
        df.columns = [col.strip().capitalize() for col in df.columns]
        if 'Ticker' in df.columns:
            dup = set(df['Ticker'].unique()).intersection(tickers_seen)
            duplicate_tickers.update(dup)
            tickers_seen.update(df['Ticker'].unique())
            dataframes.append(df)

    if duplicate_tickers:
        with st.expander("⚠️ Potential duplicate tickers found", expanded=True):
            st.warning(f"The following tickers appear in multiple uploaded files: {', '.join(duplicate_tickers)}")

    df = pd.concat(dataframes, ignore_index=True)
    save_daily_snapshot(df.copy())
    df.columns = [col.strip().capitalize() for col in df.columns]
    required_columns = ["Ticker", "Quantity"]

    if all(col in df.columns for col in required_columns):
        st.success("Holdings file(s) loaded successfully!")

        st.subheader(f"📊 Portfolio vs Benchmarks ({selected_period})")
        period = period_map[selected_period]
        benchmark_data = {}
        benchmark_series = []
        portfolio_change = None
        portfolio_normalized = None

        for label, symbol in comparison_tickers.items():
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
                    pct_change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100 if not hist.empty else 0
                    benchmark_data[label] = pct_change

                if not hist.empty:
                    norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                    benchmark_series.append(pd.DataFrame({
                        "Date": hist.index,
                        "Normalized Price": norm_price,
                        "Index": label
                    }))
            except Exception:
                benchmark_data[label] = None

        tickers = df["Ticker"].unique().tolist()
        data = []
        portfolio_start_value = 0
        portfolio_end_value = 0

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                if period == "1d":
                    info = stock.info
                    open_price = info.get("regularMarketOpen")
                    price = info.get("regularMarketPrice")
                    hist = stock.history(period="2d", interval="5m")
                    hist = hist[hist.index.date == pd.Timestamp.today().date()]
                    start_price = open_price
                    end_price = price
                else:
                    hist = stock.history(period=period)
                    info = stock.info
                    price = info.get("regularMarketPrice") or info.get("currentPrice")
                    start_price = hist["Close"].iloc[0] if not hist.empty else price
                    end_price = hist["Close"].iloc[-1] if not hist.empty else price

                sector = info.get("sector")
                if not sector:
                    if "-USD" in ticker:
                        sector = "Cryptocurrency"
                    elif info.get("quoteType") in ["ETF", "MUTUALFUND"]:
                        sector = "Fund"
                    else:
                        sector = "Unknown"

                df_ticker = df[df["Ticker"] == ticker]
                quantity = df_ticker["Quantity"].sum()
                portfolio_start_value += quantity * start_price
                portfolio_end_value += quantity * end_price

                if not hist.empty:
                    norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                    if portfolio_normalized is None:
                        portfolio_normalized = norm_price * quantity
                    else:
                        portfolio_normalized += norm_price * quantity

                data.append({"Ticker": ticker, "Current Price": end_price, "Sector": sector})
            except Exception:
                data.append({"Ticker": ticker, "Current Price": None, "Sector": "Unknown"})

        if portfolio_start_value > 0 and portfolio_normalized is not None:
            portfolio_change = (portfolio_end_value / portfolio_start_value - 1) * 100
            portfolio_normalized = pd.DataFrame({
                "Date": hist.index,
                "Normalized Price": portfolio_normalized / portfolio_start_value * 100,
                "Index": "My Portfolio"
            })

        price_df = pd.DataFrame(data)
        df = df.merge(price_df, on="Ticker", how="left")
        df["Market Value"] = df["Quantity"] * df["Current Price"]

        if "Account" in df.columns:
            accounts = df["Account"].dropna().unique().tolist()
            selected_accounts = st.sidebar.multiselect("Filter by account(s):", accounts, default=accounts)
            df = df[df["Account"].isin(selected_accounts)]

        st.subheader("🔁 Performance Summary")

        # ── Insights & Alerts ──
        st.subheader("🧠 Portfolio Insights")
        insights = []

        top_holdings = df.sort_values("Market Value", ascending=False)
        total_value = df["Market Value"].sum()

        # Overconcentration
        overweights = top_holdings[top_holdings["Market Value"] / total_value > 0.10]
        for _, row in overweights.iterrows():
            pct = row["Market Value"] / total_value * 100
            insights.append(f"⚠️ **{row['Ticker']}** is {pct:.1f}% of your portfolio.")

        # High cash drag
        if "Asset Class" in df.columns:
            cash_assets = df[df["Asset Class"].str.contains("Money Market|Cash", case=False, na=False)]
            cash_pct = cash_assets["Market Value"].sum() / total_value * 100
            if cash_pct > 15:
                insights.append(f"🪙 You have {cash_pct:.1f}% in cash or money market funds.")

        # Sharp drops and big gains
        for i, row in df.iterrows():
            ticker = row["Ticker"]
            try:
                hist = yf.Ticker(ticker).history(period=period)
                if not hist.empty:
                    change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    if change <= -10:
                        insights.append(f"🔻 **{ticker}** dropped {change:.1f}% over selected period.")
                    if change >= 10:
                        insights.append(f"🚀 **{ticker}** gained {change:.1f}% over selected period.")
            except Exception as e:
                st.warning(f"Error processing {ticker}: {e}")

        # Upcoming earnings
        for i, row in df.iterrows():
            ticker = row["Ticker"]
            try:
                cal = yf.Ticker(ticker).calendar
                if not cal.empty:
                    earnings_date = cal.loc["Earnings Date"].max()
                    if pd.notna(earnings_date):
                        days = (earnings_date - pd.Timestamp.today()).days
                        if 0 <= days <= 14:
                            insights.append(f"📅 <span title='Earnings Date: {earnings_date.date()}'>**{ticker}** reports earnings in {days} days.</span>")
            except: pass

        if insights:
            insights.sort()
            with st.expander("View insights", expanded=True):
                for note in insights:
                    st.markdown(f"- {note}")

            # Export insights to download
            insights_text = "\n".join(insights)
            st.download_button("📥 Download Insights Report", insights_text, file_name="portfolio_insights.txt")
        else:
            st.success("No alerts. Portfolio looks healthy.")

        metric_cols = st.columns(2)
        perf_metrics = [
            ("📦 My Portfolio", portfolio_change),
            *[(f"📊 {label}", benchmark_data.get(label)) for label in comparison_tickers]
        ]

        for i, (label, value) in enumerate(perf_metrics):
            color = "green" if value is not None and value >= 0 else "red"
            arrow = "⬆️" if value is not None and value >= 0 else "⬇️"
            formatted_value = f"{value:+.2f}% {arrow}" if value is not None else "N/A"
            with metric_cols[i % 2]:
                st.markdown(f"<div style='font-size: 14px; color: {color};'>{label}: {formatted_value}</div>", unsafe_allow_html=True)

        if benchmark_series:
            all_series = pd.concat(benchmark_series + ([portfolio_normalized] if portfolio_normalized is not None else []))
            fig = px.line(all_series, x="Date", y="Normalized Price", color="Index", title="Normalized Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Portfolio Overview")
        st.dataframe(df)
        total_value = df["Market Value"].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            sector_group = df.groupby("Sector")["Market Value"].sum().reset_index()
            fig_sector = px.pie(sector_group, values="Market Value", names="Sector", title="By Sector")
            st.plotly_chart(fig_sector, use_container_width=True)

        with col2:
            if "Asset Class" in df.columns:
                asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
                fig_asset = px.pie(asset_group, values="Market Value", names="Asset Class", title="By Asset Class")
                st.plotly_chart(fig_asset, use_container_width=True)

        with col3:
            if "Account" in df.columns:
                account_group = df.groupby("Account")["Market Value"].sum().reset_index()
                fig_account = px.pie(account_group, values="Market Value", names="Account", title="By Account")
                st.plotly_chart(fig_account, use_container_width=True)
else:
    st.info("Upload at least one CSV or Excel portfolio file to get started.")
