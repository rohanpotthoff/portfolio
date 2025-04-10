import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
import os

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
    "1W": "7d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "5Y": "5y"
}

# ==============================================
# Helper Functions
# ==============================================
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

def fetch_market_data(label, symbol, period):
    """Fetch market data with proper labeling"""
    try:
        ticker = yf.Ticker(symbol)
        if period == "1d":
            hist = ticker.history(period="1d", interval="5m", prepost=False)
            hist = hist.between_time('09:30', '16:00')
            # Remove duplicate timestamps
            hist = hist[~hist.index.duplicated(keep='last')]
        else:
            hist = ticker.history(period=period)
        
        # Timezone handling
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        
        if not hist.empty:
            # Use proper display names
            norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
            return {
                "data": pd.DataFrame({
                    "Date": hist.index,
                    "Normalized Price": norm_price,
                    "Index": label  # Use friendly name instead of symbol
                }),
                "pct_change": (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            }
        return None
    except Exception as e:
        st.warning(f"Data fetch error for {label}: {str(e)}")
        return None

# ==============================================
# UI Components
# ==============================================
def render_header():
    st.title("📊 Portfolio Tracker")
    st.caption("Version 4.0 | Created by Rohan Potthoff")
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
        - **v4.0**: Enhanced metrics, treasury benchmark, raw data export
        - **v3.0**: Portfolio insights, money market support
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

    benchmark_series = []
    benchmark_data = {}
    portfolio_change = None
    df = pd.DataFrame()
    total_value = 0

    # Fetch benchmark data
    for label, symbol in COMPARISON_TICKERS.items():
        result = fetch_market_data(label, symbol, period)  # Pass both label and symbol
        if result:
            benchmark_series.append(result["data"])
            benchmark_data[label] = result["pct_change"]

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
                    dup = set(cleaned_df['Ticker']).intersection(tickers_seen)
                    duplicate_tickers.update(dup)
                    tickers_seen.update(cleaned_df['Ticker'])
                
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

            for ticker in df["Ticker"].unique():
                try:
                    if is_money_market(ticker):
                        current_price = 1.0
                        start_price = 1.0
                        end_price = 1.0
                        hist = None
                        sector = "Cash"
                        asset_class = "Money Market"
                    else:
                        stock = yf.Ticker(ticker)
                        
                        if period == "1d":
                            info = stock.info
                            open_price = info.get("regularMarketOpen", 1.0)
                            end_price = info.get("regularMarketPrice", 1.0)
                            hist = stock.history(period="1d", interval="5m", prepost=False)
                            hist = hist.between_time('09:30', '16:00')
                        else:
                            hist = stock.history(period=period)
                            info = stock.info
                            start_price = hist["Close"].iloc[0] if not hist.empty else 1.0
                            end_price = hist["Close"].iloc[-1] if not hist.empty else 1.0
                        
                        sector = info.get("sector", "Unknown")
                        asset_class = info.get("quoteType", "Stock").title()

                    qty = df[df["Ticker"] == ticker]["Quantity"].sum()
                    portfolio_start_value += qty * (start_price if period != "1d" else open_price)
                    portfolio_end_value += qty * end_price

                    price_data.append({
                        "Ticker": ticker, 
                        "Current Price": end_price,
                        "Sector": sector,
                        "Asset Class": asset_class
                    })

                except Exception as e:
                    st.warning(f"Skipping {ticker}: {str(e)}")
                    continue

            # Calculate performance
            if portfolio_start_value > 0:
                portfolio_change = (portfolio_end_value / portfolio_start_value - 1) * 100

            # Merge price data
            price_df = pd.DataFrame(price_data)
            df = df.merge(price_df, on="Ticker", how="left")
            df["Market Value"] = df["Quantity"] * df["Current Price"]
            total_value = df["Market Value"].sum()

            # Add portfolio to benchmarks
            if total_value > 0 and len(benchmark_series) > 0:
                portfolio_normalized = pd.DataFrame({
                    "Date": benchmark_series[0]["Date"],
                    "Normalized Price": (df["Market Value"] / portfolio_start_value * 100).cumsum(),
                    "Index": "My Portfolio"
                })
                benchmark_series.append(portfolio_normalized)

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
                    
                    st.download_button(
                        "💾 Download Insights", 
                        "\n".join(insights), 
                        file_name=f"portfolio_alerts_{datetime.date.today()}.txt"
                    )
            else:
                st.success("🎉 No alerts - portfolio looks healthy!")

            st.markdown("---")

            # Performance Metrics
            st.subheader("📈 Performance Metrics")
            
            # Portfolio metric row
            portfolio_cols = st.columns(1)
            with portfolio_cols[0]:
                if portfolio_change is not None:
                    delta_arrow = "↑" if portfolio_change >= 0 else "↓"
                    color = "#2ECC40" if portfolio_change > 0 else "#FF4136"
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 24px; font-weight: bold;">{portfolio_change:.2f}%</span>
                        <span style="color: {color}; font-size: 24px;">{delta_arrow}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="My Portfolio", value="", delta=None)
            
            # Benchmarks row
            bench_cols = st.columns(len(COMPARISON_TICKERS))
            for i, (label, value) in enumerate(benchmark_data.items()):
                with bench_cols[i]:
                    if value is not None:
                        delta_arrow = "↑" if value >= 0 else "↓"
                        color = "#2ECC40" if value > 0 else "#FF4136"
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 4px;">
                            <span>{value:.2f}%</span>
                            <span style="color: {color};">{delta_arrow}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.metric(label=label, value="", delta=None)

            st.markdown("---")

            # ==============================================
            # Performance Visualization
            # ==============================================
            st.subheader("📉 Intraday Performance Comparison" if period == "1d" else "📊 Historical Performance")
            if benchmark_series:
                 try:
                        # Remove duplicate indices and sort
                        clean_series = []
                        for series in benchmark_series:
                            # Ensure chronological order
                            deduped = series.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
                            clean_series.append(deduped)
                        
                        # Create unified timeline
                        if period == "1d":
                            market_hours = pd.date_range(
                                start=pd.Timestamp.today().tz_localize(None).normalize() + pd.Timedelta(hours=9, minutes=30),
                                end=pd.Timestamp.today().tz_localize(None).normalize() + pd.Timedelta(hours=16),
                                freq='5T'
                            )
                            
                            aligned_data = []
                            for series in clean_series:
                                # Sort before processing
                                sorted_series = series.sort_values('Date')
                                s = sorted_series.set_index('Date').reindex(market_hours, method='ffill').reset_index()
                                aligned_data.append(s.rename(columns={'index':'Date'}))
                            
                            chart_data = pd.concat(aligned_data).sort_values('Date')  # Final sort
                        else:
                            chart_data = pd.concat(clean_series).sort_values('Date')  # Sort combined data
                
                        # Create interactive chart
                        fig = px.line(
                            chart_data.dropna(), 
                            x="Date", 
                            y="Normalized Price", 
                            color="Index",
                            title="",
                            template="plotly_white",
                            height=500
                        ).update_layout(
                            xaxis_title="Market Hours" if period == "1d" else "Date",
                            hovermode="x unified",
                            legend_title_text="Benchmarks",
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
                            fig.update_xaxes(
                                tickformat="%H:%M",
                                tickvals=pd.date_range(
                                    start=market_hours[0], 
                                    end=market_hours[-1], 
                                    freq='1H'
                                ),
                                range=[market_hours[0], market_hours[-1]]
                            )
                
                        st.plotly_chart(fig, use_container_width=True)
                
                    except Exception as e:
                        st.error(f"Chart rendering error: {str(e)}")

            st.markdown("---")

            # Portfolio Composition
            st.subheader("📊 Portfolio Composition")
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
