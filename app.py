import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="SAHASRANETRA - The Thousand Eyes",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .bullish { color: #00ff88; font-weight: bold; }
    .bearish { color: #ff4444; font-weight: bold; }
    .neutral { color: #ffaa00; font-weight: bold; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #3498DB 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Core Classes (from our previous batches)
class DataCollector:
    def __init__(self):
        self.cache = {}
    
    def get_yahoo_quote(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = info.get('regularMarketPreviousClose', current_price)
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close else 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
                'high': hist['High'].iloc[-1] if not hist.empty else current_price,
                'low': hist['Low'].iloc[-1] if not hist.empty else current_price,
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_yahoo_historical(self, symbol, period="1mo"):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching historical data for {symbol}: {e}")
            return None

class MarketAnalytics:
    def __init__(self, data_collector):
        self.data_collector = data_collector
    
    def calculate_sma(self, prices, window=20):
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window
    
    def calculate_rsi(self, prices, window=14):
        if len(prices) < window + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        if len(gains) < window:
            return None
        
        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def analyze_stock(self, symbol):
        # Get historical data
        hist_data = self.data_collector.get_yahoo_historical(symbol, period="3mo")
        if hist_data is None or hist_data.empty:
            return None
        
        prices = hist_data['Close'].tolist()
        
        # Calculate indicators
        sma_20 = self.calculate_sma(prices, 20)
        rsi = self.calculate_rsi(prices)
        
        # Current price
        current_price = prices[-1]
        
        # Generate signals
        signals = []
        if sma_20:
            if current_price > sma_20:
                signals.append("Bullish - Price above SMA")
            else:
                signals.append("Bearish - Price below SMA")
        
        if rsi:
            if rsi > 70:
                signals.append("Overbought - RSI > 70")
            elif rsi < 30:
                signals.append("Oversold - RSI < 30")
        
        # Calculate volatility
        returns = hist_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'sma_20': sma_20,
            'rsi': rsi,
            'volatility': volatility,
            'signals': signals if signals else ["Neutral"],
            'historical_data': hist_data
        }

# Initialize components
@st.cache_resource
def initialize_components():
    data_collector = DataCollector()
    analytics = MarketAnalytics(data_collector)
    return data_collector, analytics

data_collector, analytics = initialize_components()

# Header
st.markdown("""
<div class="main-header">
    <h1>👁️ SAHASRANETRA</h1>
    <h3>The Thousand Eyes Market Intelligence System</h3>
    <p>Real-time Global Market Analysis & Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Control Panel")

# Market Selection
default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
selected_symbols = st.sidebar.multiselect(
    "Select Stocks to Analyze",
    options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ'],
    default=default_symbols[:5]
)

# Custom symbol input
custom_symbol = st.sidebar.text_input("Add Custom Symbol", placeholder="e.g., IBM")
if custom_symbol and custom_symbol.upper() not in selected_symbols:
    selected_symbols.append(custom_symbol.upper())

# Refresh button
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.session_state.data_cache.clear()
    st.rerun()

# Auto refresh toggle
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")

# Time period selection
time_period = st.sidebar.selectbox(
    "Analysis Period",
    ["1mo", "3mo", "6mo", "1y"],
    index=1
)

# Main dashboard
if selected_symbols:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📈 Technical Analysis", "🌡️ Market Sentiment", "📰 Live Feed"])
    
    with tab1:
        st.header("📊 Real-time Market Overview")
        
        # Market metrics
        col1, col2, col3, col4 = st.columns(4)
        
        market_data = {}
        total_volume = 0
        gainers = 0
        losers = 0
        
        # Fetch data for all symbols
        progress_bar = st.progress(0)
        for i, symbol in enumerate(selected_symbols):
            quote = data_collector.get_yahoo_quote(symbol)
            if quote:
                market_data[symbol] = quote
                total_volume += quote.get('volume', 0)
                if quote.get('change_percent', 0) > 0:
                    gainers += 1
                else:
                    losers += 1
            progress_bar.progress((i + 1) / len(selected_symbols))
        
        # Display summary metrics
        with col1:
            st.metric("📈 Gainers", gainers, delta=f"{gainers}/{len(selected_symbols)}")
        with col2:
            st.metric("📉 Losers", losers, delta=f"{losers}/{len(selected_symbols)}")
        with col3:
            st.metric("📊 Total Volume", f"{total_volume/1000000:.1f}M")
        with col4:
            market_status = "🟢 Bullish" if gainers > losers else "🔴 Bearish" if losers > gainers else "🟡 Neutral"
            st.metric("Market Sentiment", market_status)
        
        # Price display
        if market_data:
            st.subheader("💰 Live Prices")
            
            # Create price chart
            symbols_list = list(market_data.keys())
            prices = [market_data[s]['price'] for s in symbols_list]
            changes = [market_data[s]['change_percent'] for s in symbols_list]
            
            # Color based on change
            colors = ['green' if change > 0 else 'red' if change < 0 else 'yellow' 
                     for change in changes]
            
            fig = go.Figure(data=[
                go.Bar(x=symbols_list, y=prices, 
                      marker_color=colors,
                      text=[f"${price:.2f}<br>{change:+.2f}%" 
                           for price, change in zip(prices, changes)],
                      textposition='auto')
            ])
            
            fig.update_layout(
                title="Current Stock Prices",
                xaxis_title="Symbols",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed Market Data")
            df = pd.DataFrame(market_data).T
            df = df[['price', 'change', 'change_percent', 'volume', 'high', 'low']]
            df.columns = ['Price ($)', 'Change ($)', 'Change (%)', 'Volume', 'High ($)', 'Low ($)']
            
            # Style the dataframe
            def color_change(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            
            styled_df = df.style.applymap(color_change, subset=['Change ($)', 'Change (%)'])
            st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.header("📈 Technical Analysis")
        
        if selected_symbols:
            # Symbol selector for detailed analysis
            analysis_symbol = st.selectbox("Select Symbol for Detailed Analysis", selected_symbols)
            
            if analysis_symbol:
                analysis = analytics.analyze_stock(analysis_symbol)
                
                if analysis:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Price chart with indicators
                        hist_data = analysis['historical_data']
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=['Price & Moving Average', 'RSI'],
                            vertical_spacing=0.1,
                            row_heights=[0.7, 0.3]
                        )
                        
                        # Candlestick chart
                        fig.add_trace(
                            go.Candlestick(
                                x=hist_data.index,
                                open=hist_data['Open'],
                                high=hist_data['High'],
                                low=hist_data['Low'],
                                close=hist_data['Close'],
                                name="Price"
                            ), row=1, col=1
                        )
                        
                        # SMA line
                        if analysis['sma_20']:
                            sma_line = [analysis['sma_20']] * len(hist_data)
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=sma_line,
                                    mode='lines',
                                    name='SMA(20)',
                                    line=dict(color='orange')
                                ), row=1, col=1
                            )
                        
                        # RSI
                        if analysis['rsi']:
                            rsi_line = [analysis['rsi']] * len(hist_data)
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=rsi_line,
                                    mode='lines',
                                    name='RSI',
                                    line=dict(color='purple')
                                ), row=2, col=1
                            )
                            
                            # RSI levels
                            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                        
                        fig.update_layout(
                            title=f"{analysis_symbol} Technical Analysis",
                            template="plotly_dark",
                            height=600,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Key Metrics")
                        
                        # Display metrics
                        st.metric("Current Price", f"${analysis['current_price']:.2f}")
                        
                        if analysis['sma_20']:
                            st.metric("SMA (20)", f"${analysis['sma_20']:.2f}")
                        
                        if analysis['rsi']:
                            rsi_color = "🔴" if analysis['rsi'] > 70 else "🟢" if analysis['rsi'] < 30 else "🟡"
                            st.metric("RSI", f"{analysis['rsi']:.1f} {rsi_color}")
                        
                        st.metric("Volatility", f"{analysis['volatility']:.2f}%")
                        
                        # Trading signals
                        st.subheader("🎯 Trading Signals")
                        for signal in analysis['signals']:
                            if "Bullish" in signal or "Oversold" in signal:
                                st.success(f"🟢 {signal}")
                            elif "Bearish" in signal or "Overbought" in signal:
                                st.error(f"🔴 {signal}")
                            else:
                                st.info(f"🟡 {signal}")
    
    with tab3:
        st.header("🌡️ Market Sentiment Analysis")
        
        # Analyze sentiment for all selected symbols
        sentiment_data = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'analyses': {}
        }
        
        progress_bar = st.progress(0)
        for i, symbol in enumerate(selected_symbols):
            analysis = analytics.analyze_stock(symbol)
            if analysis:
                sentiment_data['analyses'][symbol] = analysis
                
                # Categorize sentiment
                signals = analysis['signals']
                if any('Bullish' in s or 'Oversold' in s for s in signals):
                    sentiment_data['bullish'] += 1
                elif any('Bearish' in s or 'Overbought' in s for s in signals):
                    sentiment_data['bearish'] += 1
                else:
                    sentiment_data['neutral'] += 1
            
            progress_bar.progress((i + 1) / len(selected_symbols))
        
        # Sentiment overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 Bullish Stocks", sentiment_data['bullish'])
        with col2:
            st.metric("🔴 Bearish Stocks", sentiment_data['bearish'])
        with col3:
            st.metric("🟡 Neutral Stocks", sentiment_data['neutral'])
        
        # Sentiment pie chart
        if sum([sentiment_data['bullish'], sentiment_data['bearish'], sentiment_data['neutral']]) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Bullish', 'Bearish', 'Neutral'],
                values=[sentiment_data['bullish'], sentiment_data['bearish'], sentiment_data['neutral']],
                marker_colors=['green', 'red', 'orange']
            )])
            
            fig.update_layout(
                title="Overall Market Sentiment",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed sentiment breakdown
        if sentiment_data['analyses']:
            st.subheader("📋 Detailed Sentiment Analysis")
            
            sentiment_df = []
            for symbol, analysis in sentiment_data['analyses'].items():
                sentiment_df.append({
                    'Symbol': symbol,
                    'Price': f"${analysis['current_price']:.2f}",
                    'RSI': f"{analysis['rsi']:.1f}" if analysis['rsi'] else "N/A",
                    'Volatility': f"{analysis['volatility']:.2f}%",
                    'Primary Signal': analysis['signals'][0] if analysis['signals'] else "No Signal"
                })
            
            df = pd.DataFrame(sentiment_df)
            st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.header("📰 Live Market Feed")
        
        # Real-time updates
        st.subheader("🔴 Live Updates")
        
        # Last update time
        if st.session_state.last_update:
            st.text(f"Last updated: {st.session_state.last_update}")
        
        # Quick stats
        if market_data:
            st.subheader("⚡ Quick Stats")
            
            # Top gainer and loser
            sorted_by_change = sorted(market_data.items(), 
                                    key=lambda x: x[1].get('change_percent', 0), 
                                    reverse=True)
            
            if sorted_by_change:
                col1, col2 = st.columns(2)
                
                with col1:
                    top_gainer = sorted_by_change[0]
                    st.success(f"🚀 Top Gainer: {top_gainer[0]} ({top_gainer[1]['change_percent']:+.2f}%)")
                
                with col2:
                    top_loser = sorted_by_change[-1]
                    st.error(f"📉 Top Loser: {top_loser[0]} ({top_loser[1]['change_percent']:+.2f}%)")
        
        # Market alerts
        st.subheader("🚨 Market Alerts")
        
        alerts = []
        for symbol, data in market_data.items():
            change_pct = data.get('change_percent', 0)
            if abs(change_pct) > 5:  # Alert for >5% change
                alert_type = "🚀 Strong Move Up" if change_pct > 0 else "📉 Strong Move Down"
                alerts.append(f"{alert_type}: {symbol} {change_pct:+.2f}%")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.info("No significant alerts at this time")

# Auto refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>👁️ SAHASRANETRA - The Thousand Eyes Market Intelligence System</p>
    <p>Real-time market data powered by Yahoo Finance | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Update last refresh time
st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")