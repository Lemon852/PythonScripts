import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.title("Enhanced VWAP & SMA(50) Intraday Backtest with Multi-Period Analysis")

# --- User input
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")
profit_target = st.number_input("Profit Target (%)", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
stop_loss = st.number_input("Stop Loss (%)", value=0.3, min_value=0.1, max_value=5.0, step=0.1)
max_trades_per_day = st.number_input("Max Trades Per Day", value=3, min_value=1, max_value=10)
initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, max_value=100000, step=1000)

def calculate_indicators(df, sma_period=50):
    """Calculate technical indicators"""
    # Calculate SMA with adaptive period
    df[f'SMA{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
    
    # Calculate VWAP (reset daily)
    df['Date'] = df.index.date
    df['VWAP'] = 0.0
    
    for date in df['Date'].unique():
        mask = df['Date'] == date
        day_data = df[mask].copy()
        
        if len(day_data) > 0:
            typical_price = (day_data['High'] + day_data['Low'] + day_data['Close']) / 3
            cum_vol = day_data['Volume'].cumsum()
            cum_vol_x_price = (typical_price * day_data['Volume']).cumsum()
            vwap_values = cum_vol_x_price / cum_vol
            df.loc[mask, 'VWAP'] = vwap_values
    
    return df

def generate_signals(df, sma_period=50):
    """Generate trading signals"""
    sma_col = f'SMA{sma_period}'
    df['prev_vwap'] = df['VWAP'].shift(1)
    df['prev_sma'] = df[sma_col].shift(1)
    
    # Signal: VWAP crosses above SMA
    df['signal'] = np.where(
        (df['VWAP'] > df[sma_col]) & 
        (df['prev_vwap'] <= df['prev_sma']) & 
        (df['VWAP'].notna()) & 
        (df[sma_col].notna()),
        1, 0
    )
    
    return df

def simulate_trades(df, profit_target, stop_loss, max_trades_per_day, initial_capital):
    """Simulate trades with realistic exit conditions"""
    trades = []
    current_capital = initial_capital
    daily_trades = {}
    
    for idx, row in df[df['signal'] == 1].iterrows():
        # Check daily trade limit
        trade_date = row['Date']
        if trade_date not in daily_trades:
            daily_trades[trade_date] = 0
        
        if daily_trades[trade_date] >= max_trades_per_day:
            continue
            
        entry_price = row['Close']
        entry_time = idx
        
        # Calculate position size (risk 1% of capital per trade)
        risk_amount = current_capital * 0.01
        stop_loss_amount = entry_price * (stop_loss / 100)
        shares = int(risk_amount / stop_loss_amount)
        
        if shares <= 0:
            continue
            
        position_value = shares * entry_price
        
        # Find exit point
        exit_found = False
        remaining_data = df.loc[idx:].iloc[1:]  # Skip current candle
        
        for exit_idx, exit_row in remaining_data.iterrows():
            # Check if we're still in the same day
            if exit_row['Date'] != trade_date:
                # Force exit at end of day
                exit_price = exit_row['Close']
                exit_time = exit_idx
                exit_found = True
                break
                
            change_pct = (exit_row['Close'] - entry_price) / entry_price * 100
            
            if change_pct >= profit_target:
                exit_price = exit_row['Close']
                exit_time = exit_idx
                exit_found = True
                break
            elif change_pct <= -stop_loss:
                exit_price = exit_row['Close']
                exit_time = exit_idx
                exit_found = True
                break
        
        if not exit_found:
            # Force exit at last available price
            exit_price = remaining_data['Close'].iloc[-1] if len(remaining_data) > 0 else entry_price
            exit_time = remaining_data.index[-1] if len(remaining_data) > 0 else idx
        
        # Calculate P&L
        pnl_dollars = (exit_price - entry_price) * shares
        pnl_percent = (exit_price - entry_price) / entry_price * 100
        current_capital += pnl_dollars
        
        result = "Win" if pnl_dollars > 0 else "Loss"
        
        trades.append({
            'Entry Time': entry_time,
            'Entry Price': round(entry_price, 2),
            'Exit Time': exit_time,
            'Exit Price': round(exit_price, 2),
            'Shares': shares,
            'Position Value': round(position_value, 2),
            'P&L ($)': round(pnl_dollars, 2),
            'P&L (%)': round(pnl_percent, 2),
            'Capital': round(current_capital, 2),
            'Result': result
        })
        
        daily_trades[trade_date] += 1
    
    return pd.DataFrame(trades), current_capital

def get_performance_metrics(trades_df, initial_capital, final_capital):
    """Calculate performance metrics"""
    if len(trades_df) == 0:
        return {}
    
    wins = len(trades_df[trades_df['Result'] == 'Win'])
    losses = len(trades_df[trades_df['Result'] == 'Loss'])
    total_trades = len(trades_df)
    
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['Result'] == 'Win']['P&L (%)'].mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df['Result'] == 'Loss']['P&L (%)'].mean() if losses > 0 else 0
    
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    return {
        'Total Trades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Win Rate (%)': round(win_rate, 2),
        'Avg Win (%)': round(avg_win, 2),
        'Avg Loss (%)': round(avg_loss, 2),
        'Total Return (%)': round(total_return, 2),
        'Final Capital ($)': round(final_capital, 2)
    }

if st.button("Run Multi-Period Backtest"):
    st.write(f"Fetching data for {ticker}...")
    
    # Define multiple test periods with fallback options
    periods = [
        ("Recent 5 Days", "5d", "1h"),
        ("Recent 1 Month", "1mo", "1h"),
        ("Recent 3 Months", "3mo", "1h")
    ]
    
    all_results = []
    
    for period_name, period_code, interval in periods:
        try:
            st.write(f"Testing {period_name} with {interval} intervals...")
            
            # Download data with error handling
            try:
                df = yf.download(
                    tickers=ticker, 
                    period=period_code, 
                    interval=interval, 
                    progress=False,
                    auto_adjust=True,
                    prepost=False
                )
            except Exception as download_error:
                st.error(f"Download failed for {period_name}: {str(download_error)}")
                continue
            
            # Debug: Show what we got
            st.write(f"Downloaded {len(df)} rows for {period_name}")
            if not df.empty:
                st.write(f"Columns: {list(df.columns)}")
                st.write(f"Data range: {df.index[0]} to {df.index[-1]}")
            
            if df.empty:
                st.warning(f"No data available for {period_name}")
                continue
            
            # Handle MultiIndex columns more robustly
            if isinstance(df.columns, pd.MultiIndex):
                # Get the first level (should be the actual column names like 'Close', 'High', etc.)
                df.columns = df.columns.get_level_values(0)
            
            # Map possible column names to standard names
            column_mapping = {
                'Adj Close': 'Close',
                'close': 'Close',
                'Close': 'Close',
                'volume': 'Volume',
                'Volume': 'Volume',
                'high': 'High',
                'High': 'High',
                'low': 'Low',
                'Low': 'Low',
                'open': 'Open',
                'Open': 'Open'
            }
            
            # Rename columns to standardize
            df = df.rename(columns=column_mapping)
            
            # Debug: Show columns after processing
            st.write(f"Columns after processing: {list(df.columns)}")
            
            # Check required columns
            required_cols = ['Close', 'Volume', 'High', 'Low']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns for {period_name}: {missing_cols}")
                st.write(f"Available columns: {list(df.columns)}")
                continue
            
            # Clean data
            df = df.dropna(subset=required_cols)
            
            # Alternative: try different intervals if 1m fails
            if len(df) < 100 and interval == "1m":
                st.write(f"Insufficient 1m data, trying 5m interval...")
                try:
                    df = yf.download(
                        tickers=ticker, 
                        period=period_code, 
                        interval="1h", 
                        progress=False,
                        auto_adjust=True,
                        prepost=False
                    )
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(1)
                    df = df.rename(columns=column_mapping)
                    df = df.dropna(subset=required_cols)
                    st.write(f"Successfully got {len(df)} rows with 5m data")
                except:
                    pass
            
            if len(df) < 50:  # Minimum data requirement
                st.warning(f"Insufficient data for {period_name}: only {len(df)} rows")
                continue
            
            # Use shorter SMA for limited data
            sma_period = min(20, len(df) // 3)  # Adaptive SMA period
            
            # Calculate indicators and signals
            df = calculate_indicators(df, sma_period)
            df = generate_signals(df, sma_period)
            
            # Simulate trades
            trades_df, final_capital = simulate_trades(
                df, profit_target, stop_loss, max_trades_per_day, initial_capital
            )
            
            # Calculate metrics
            metrics = get_performance_metrics(trades_df, initial_capital, final_capital)
            metrics['Period'] = period_name
            metrics['Data Points'] = len(df)
            metrics['Signal Count'] = len(df[df['signal'] == 1])
            
            all_results.append(metrics)
            
            # Display individual results
            st.subheader(f"Results for {period_name}")
            
            if len(trades_df) > 0:
                st.write("**Trade Details:**")
                st.dataframe(trades_df)
                
                # Plot for this period
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # Price chart
                sma_col = f'SMA{sma_period}'
                ax1.plot(df.index, df['Close'], label='Close', color='blue', linewidth=1)
                ax1.plot(df.index, df['VWAP'], label='VWAP', color='orange', linewidth=1)
                ax1.plot(df.index, df[sma_col], label=f'SMA{sma_period}', color='green', linewidth=1)
                
                # Plot signals
                signal_df = df[df['signal'] == 1]
                if len(signal_df) > 0:
                    ax1.scatter(signal_df.index, signal_df['Close'], 
                              marker='^', color='magenta', s=100, label='Entry Signal')
                
                ax1.legend()
                ax1.set_title(f'{ticker} - {period_name}')
                ax1.set_ylabel("Price")
                
                # Equity curve
                if len(trades_df) > 0:
                    ax2.plot(range(len(trades_df)), trades_df['Capital'], 
                            color='green', linewidth=2, label='Portfolio Value')
                    ax2.axhline(y=initial_capital, color='red', linestyle='--', label='Initial Capital')
                    ax2.set_title('Portfolio Value Over Time')
                    ax2.set_xlabel('Trade Number')
                    ax2.set_ylabel('Capital ($)')
                    ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.write("No trades executed for this period")
                
        except Exception as e:
            st.error(f"Error processing {period_name}: {str(e)}")
    
    # Summary table
    if all_results:
        st.subheader("Multi-Period Performance Summary")
        summary_df = pd.DataFrame(all_results)
        
        # Reorder columns for better display
        col_order = ['Period', 'Total Trades', 'Win Rate (%)', 'Total Return (%)', 
                    'Final Capital ($)', 'Wins', 'Losses', 'Avg Win (%)', 'Avg Loss (%)',
                    'Signal Count', 'Data Points']
        summary_df = summary_df[col_order]
        
        st.dataframe(summary_df)
        
        # Overall statistics
        st.subheader("Strategy Analysis")
        
        total_trades = summary_df['Total Trades'].sum()
        avg_win_rate = summary_df['Win Rate (%)'].mean()
        avg_return = summary_df['Total Return (%)'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades Across All Periods", total_trades)
        with col2:
            st.metric("Average Win Rate", f"{avg_win_rate:.2f}%")
        with col3:
            st.metric("Average Return", f"{avg_return:.2f}%")
        
        # Strategy insights
        st.subheader("Strategy Insights")
        
        if avg_win_rate > 50:
            st.success("✅ Strategy shows positive win rate across test periods")
        else:
            st.warning("⚠️ Strategy shows below 50% win rate - consider adjustments")
            
        if avg_return > 0:
            st.success("✅ Strategy shows positive average returns")
        else:
            st.warning("⚠️ Strategy shows negative average returns - needs optimization")
            
        st.write("**Recommendations:**")
        st.write("• Test with different profit targets and stop losses")
        st.write("• Consider adding volume or volatility filters")
        st.write("• Analyze time-of-day performance patterns")
        st.write("• Consider market conditions and news events")
        
    else:
        st.error("No valid results obtained from any test period")
    
    st.success("Multi-period backtest completed!")