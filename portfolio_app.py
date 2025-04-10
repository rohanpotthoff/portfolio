import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
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
  - Summary grid
  - ETF/Mutual/Crypto fallback
  - Duplicate file detection
- **v1.0.0.1**
  - Initial stable version with performance summary
  - Benchmark normalization
  - Sector and asset class breakdowns
  - Summary grid
  - ETF/Mutual/Crypto fallback
  - Duplicate file detection
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

        if portfolio_start_value > 0 and portfolio_normalized is not None and not hist.empty:
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
            import io
            insights_buffer = io.BytesIO(os.linesep.join(insights).encode("utf-8"))
            st.download_button("📥 Download Insights Report", insights_buffer, file_name="portfolio_insights.txt", mime="text/plain")

            # ── Performance Grid ──
            if portfolio_normalized is not None and benchmark_series:
                all_perf = pd.concat([portfolio_normalized] + benchmark_series)
                fig = px.line(all_perf, x="Date", y="Normalized Price", color="Index", title="Portfolio vs Benchmarks")
                fig.update_layout(height=400, legend_title="", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 💹 Summary Performance")
            cols = st.columns(4)
            perf_data = [
                ("My Portfolio", portfolio_change),
                ("S&P 500", benchmark_data.get("S&P 500")),
                ("Nasdaq 100", benchmark_data.get("Nasdaq 100")),
                ("Euro Stoxx 50", benchmark_data.get("Euro Stoxx 50"))
            ]
            for i, (label, change) in enumerate(perf_data):
                color = "green" if change and change > 0 else "red" if change and change < 0 else "gray"
                icon = "⬆️" if change and change > 0 else "⬇️" if change and change < 0 else "➖"
                cols[i].markdown(f"**{label}**

<span style='color:{color}; font-size: 24px;'>{icon} {change:.2f}%</span>", unsafe_allow_html=True)

            # Removed summary table section
            st.subheader("📊 Performance Summary Table")
            summary_data = {"My Portfolio": portfolio_change}
            summary_data.update(benchmark_data)
            perf_df = pd.DataFrame(list(summary_data.items()), columns=["Index", "% Change"])
            perf_df["Trend"] = perf_df["% Change"].apply(lambda x: "⬆️" if x > 0 else "⬇️" if x < 0 else "➖")
            perf_df["Color"] = perf_df["% Change"].apply(lambda x: "green" if x > 0 else "red" if x < 0 else "gray")
            perf_df_display = perf_df.drop(columns="Color")
            st.dataframe(perf_df_display.set_index("Index"))

            # ── Portfolio Holdings Grid ──
            st.subheader("📋 Holdings Overview")
            display_cols = ["Ticker", "Quantity", "Current Price", "Market Value"]
            if "Account" in df.columns:
                display_cols.insert(1, "Account")
            st.dataframe(df[display_cols].sort_values("Market Value", ascending=False).reset_index(drop=True))

            # ── Allocation Charts ──
            st.subheader("📎 Portfolio Allocation")
            alloc_by_sector = df.groupby("Sector")["Market Value"].sum().reset_index()
            fig_sector = px.pie(alloc_by_sector, values="Market Value", names="Sector", title="By Sector")
            st.plotly_chart(fig_sector, use_container_width=True)

            if "Account" in df.columns:
                alloc_by_acct = df.groupby("Account")["Market Value"].sum().reset_index()
                fig_acct = px.pie(alloc_by_acct, values="Market Value", names="Account", title="By Account")
                st.plotly_chart(fig_acct, use_container_width=True)

