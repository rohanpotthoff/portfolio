import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime
from functools import lru_cache

# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

MONEY_MARKET = {"WMPXX", "FNSXX", "VMFXX", "CASH**"}
BENCHMARKS = {
    "S&P 500": "^GSPC", "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E", "10Y Treasury": "^TNX"
}
PERIODS = {
    "Today": "1d", "1W": "7d", "1M": "1mo",
    "3M": "3mo", "6M": "6mo", "YTD": "ytd",
    "1Y": "1y", "5Y": "5y"
}

# =============================================================================
# Core Functions
# =============================================================================
@lru_cache(maxsize=128)
def fetch_market_data(symbol, period):
    """Cached market data fetcher with intraday handling"""
    try:
        is_intraday = period == "1d"
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=period,
            interval="5m" if is_intraday else "1d",
            prepost=False
        ).tz_localize(None)
        
        if is_intraday:
            data = data.between_time('09:30', '16:00').resample('5T').last()
            data = data[~data.index.duplicated()]
            
        return data[['Close']] if not data.empty else None
    except Exception as e:
        st.error(f"Failed to fetch {symbol}: {str(e)}")
        return None

def process_portfolio(uploaded_files, period):
    """Process uploaded files into portfolio analysis"""
    portfolio = pd.DataFrame()
    duplicates = set()
    
    # Process files
    for file in uploaded_files:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df = df.rename(columns=lambda x: x.strip().capitalize())
        df = df[['Ticker', 'Quantity'] + (['Account'] if 'Account' in df else [])].dropna()
        
        # Track duplicates across files
        dupes = set(df.Ticker) & set(portfolio.Ticker) if not portfolio.empty else set()
        duplicates.update(dupes)
        portfolio = pd.concat([portfolio, df], ignore_index=True)
    
    # Add financial data
    if portfolio.empty:
        return None, None, 0, 0, duplicates
    
    portfolio = portfolio.groupby('Ticker', as_index=False).agg({'Quantity': 'sum', **({'Account': 'first'} if 'Account' in portfolio else {})})
    
    # Calculate values
    start_val = current_val = 0
    records = []
    for _, row in portfolio.iterrows():
        ticker = row['Ticker']
        qty = row['Quantity']
        is_cash = ticker in MONEY_MARKET or "XX" in ticker
        
        data = fetch_market_data(ticker, PERIODS[period])
        info = yf.Ticker(ticker).info if not is_cash else {}
        
        open_price = 1.0 if is_cash else (data.Close.iloc[0] if data is not None else 1.0)
        current_price = 1.0 if is_cash else (data.Close.iloc[-1] if data is not None else 1.0)
        
        start_val += qty * open_price
        current_val += qty * current_price
        
        records.append({
            'Ticker': ticker,
            'Quantity': qty,
            'Price': current_price,
            'Sector': info.get('sector', 'Cash'),
            'Asset Class': 'Money Market' if is_cash else info.get('quoteType', 'Stock').title(),
            'Market Value': qty * current_price,
            **({'Account': row['Account']} if 'Account' in row else {})
        })
    
    return pd.DataFrame(records), start_val, current_val, duplicates

def generate_insights(portfolio, total_value, period):
    """Generate portfolio health insights"""
    insights = []
    
    # Concentration risk
    portfolio['Pct'] = portfolio.MarketValue / total_value
    heavy = portfolio[portfolio.Pct > 0.1]
    for ticker in heavy.Ticker:
        insights.append(f"⚠️ **{ticker}** ({heavy[heavy.Ticker == ticker].Pct.values[0]:.1%}) exceeds 10% allocation")
    
    # Cash position
    cash = portfolio[portfolio['Asset Class'] == 'Money Market'].MarketValue.sum()
    if cash / total_value > 0.15:
        insights.append(f"🪙 Cash allocation ({cash/total_value:.1%}) may create drag")
    
    # Security movements
    for ticker in portfolio[portfolio['Asset Class'] != 'Money Market'].Ticker:
        data = fetch_market_data(ticker, PERIODS[period])
        if data is None:
            continue
            
        change = (data.Close.iloc[-1] / data.Close.iloc[0] - 1) * 100
        if change <= -10:
            insights.append(f"🔻 **{ticker}** dropped {abs(change):.1f}% ({period})")
        elif change >= 20:
            insights.append(f"🚀 **{ticker}** surged {change:.1f}% ({period})")
        
        # Earnings check
        cal = yf.Ticker(ticker).calendar
        if not cal.empty:
            earnings = cal.EarningsDate.max()
            days = (earnings.date() - datetime.date.today()).days
            if 0 <= days <= 14:
                insights.append(f"📅 **{ticker}** reports earnings in {days} days")
    
    return insights

# =============================================================================
# UI Components
# =============================================================================
def render_header():
    """Page header with social links"""
    st.title("📊 Portfolio Tracker")
    st.caption("Version 4.0 | Data: Yahoo Finance | Created by Rohan Potthoff")
    st.markdown("""
    <style>
    .social {display:flex; gap:15px; margin:-10px 0 10px;}
    .social a {color:#9e9e9e!important; text-decoration:none; font-size:14px;}
    .social a:hover {color:#1DA1F2!important;}
    </style>
    <div class="social">
        <a href="mailto:rohanpotthoff@gmail.com">✉️ Email</a>
        <a href="https://linkedin.com/in/rohanpotthoff" target="_blank">🔗 LinkedIn</a>
    </div>
    <hr style='margin-bottom:20px'>
    """, unsafe_allow_html=True)

def performance_chart(bench_data, portfolio_data, period):
    """Interactive normalized performance comparison"""
    # Prepare portfolio series
    if portfolio_data['current'] > 0:
        portfolio_norm = pd.DataFrame({
            'Date': bench_data[0]['Date'] if bench_data else [],
            'Normalized': (portfolio_data['current'] / portfolio_data['start'] * 100) if portfolio_data['start'] else 100,
            'Index': 'My Portfolio'
        })
        full_data = pd.concat(bench_data + [portfolio_norm])
    else:
        full_data = pd.concat(bench_data)
    
    # Create visualization
    fig = px.line(
        full_data, x='Date', y='Normalized', 
        color='Index', height=500, 
        template='plotly_white'
    ).update_layout(
        hovermode='x unified',
        legend={'title': None, 'orientation': 'h', 'y': 1.1},
        yaxis_title='Normalized Performance (%)',
        xaxis_title='Market Hours' if period == 'Today' else 'Date'
    )
    
    if period == 'Today':
        fig.update_xaxes(tickformat="%H:%M", rangeslider_visible=True)
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Main Application Flow
# =============================================================================
def main():
    render_header()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        period = st.selectbox("Analysis Period", list(PERIODS.keys()), index=5)
        with st.expander("Version History"):
            st.markdown("""
            - v4.0: Treasury benchmarks, intraday tracking
            - v3.0: Multi-account support, cash management
            - v2.0: Portfolio insights, sector analysis
            """)
    
    # Benchmark data
    bench_series = []
    bench_values = {}
    for name, symbol in BENCHMARKS.items():
        data = fetch_market_data(symbol, PERIODS[period])
        if data is not None:
            norm = data.Close / data.Close.iloc[0] * 100
            bench_series.append(pd.DataFrame({
                'Date': data.index,
                'Normalized': norm,
                'Index': name
            }))
            bench_values[name] = (data.Close.iloc[-1]/data.Close.iloc[0]-1)*100
    
    # Portfolio input
    uploaded_files = st.file_uploader(
        "Upload portfolio files (CSV/Excel)", 
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="Required columns: Ticker, Quantity"
    )
    
    if uploaded_files:
        portfolio, start_val, current_val, duplicates = process_portfolio(uploaded_files, period)
        
        if duplicates:
            st.warning(f"Duplicate tickers found: {', '.join(duplicates)}")
        
        if portfolio is not None and not portfolio.empty:
            # Performance metrics
            pct_change = (current_val/start_val - 1)*100 if start_val else 0
            delta_type = "normal" if abs(pct_change) < 10 else "inverse"
            
            st.subheader("Performance Overview")
            cols = st.columns(len(BENCHMARKS)+1)
            with cols[0]:
                st.metric("My Portfolio", f"${current_val:,.2f}", 
                         f"{pct_change:.2f}%", delta_color=delta_type)
            for i, (name, val) in enumerate(bench_values.items()):
                cols[i+1].metric(name, "", f"{val:.2f}%")
            
            # Insights
            st.subheader("Portfolio Insights")
            insights = generate_insights(portfolio, current_val, period)
            if insights:
                with st.expander(f"Active Alerts ({len(insights)})", expanded=True):
                    for insight in insights[:5]:
                        st.markdown(f"- {insight}")
                    if len(insights) > 5:
                        st.caption(f"Plus {len(insights)-5} additional alerts")
            else:
                st.success("✅ No critical issues detected")
            
            # Performance chart
            st.subheader("Comparative Performance")
            portfolio_data = {'start': start_val, 'current': current_val}
            performance_chart(bench_series, portfolio_data, period)
            
            # Composition analysis
            st.subheader("Portfolio Composition")
            cols = st.columns(3)
            with cols[0]:
                fig = px.pie(portfolio, values='Market Value', names='Sector',
                            title="Sector Allocation", hole=0.4)
                st.plotly_chart(fig, True)
            with cols[1]:
                fig = px.pie(portfolio, values='Market Value', names='Asset Class',
                            title="Asset Classes", hole=0.4)
                st.plotly_chart(fig, True)
            with cols[2]:
                if 'Account' in portfolio:
                    fig = px.pie(portfolio, values='Market Value', names='Account',
                                title="Account Distribution", hole=0.4)
                    st.plotly_chart(fig, True)
                else:
                    st.info("Enable multi-account tracking by adding 'Account' column")
            
            # Data export
            with st.expander("Raw Portfolio Data"):
                st.dataframe(portfolio.sort_values('Market Value', ascending=False))
                st.download_button(
                    "Export CSV",
                    portfolio.to_csv(index=False),
                    f"portfolio_snapshot_{datetime.date.today()}.csv"
                )

if __name__ == "__main__":
    main()
