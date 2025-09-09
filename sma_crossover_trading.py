import pandas as pd
import numpy as np

def load_and_process_data():
    """Load and process the CSV data"""
    try:
        # Load the CSV file
        df = pd.read_csv('xbtusd_1h_8y.csv')
        
        # Convert time columns to datetime
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])
        
        # Set open_time as index for easier time-based calculations
        df.set_index('open_time', inplace=True)
        
        # Ensure data is sorted chronologically
        df.sort_index(inplace=True)
        
        return df
    
    except FileNotFoundError:
        print("Error: File 'xbtusd_1h_8y.csv' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_sma_and_trades(df):
    """Calculate SMA and simulate trades"""
    if df is None or len(df) < 200:
        print("Not enough data for 200-period SMA calculation")
        return [], []
    
    # Calculate 200-period Simple Moving Average
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Identify crossover points
    df['prev_close'] = df['close'].shift(1)
    df['prev_sma'] = df['sma_200'].shift(1)
    
    # Bullish crossover (price crosses above SMA)
    bullish_cross = (df['close'] > df['sma_200']) & (df['prev_close'] <= df['prev_sma'])
    
    # Bearish crossover (price crosses below SMA)
    bearish_cross = (df['close'] < df['sma_200']) & (df['prev_close'] >= df['prev_sma'])
    
    return bullish_cross, bearish_cross

def simulate_trading(df, bullish_signals, bearish_signals):
    """Simulate trading and calculate PnL"""
    trades = []
    current_position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    
    for i in range(len(df)):
        if bullish_signals.iloc[i] and current_position != 'long':
            # Close any existing short position
            if current_position == 'short':
                exit_price = df['close'].iloc[i]
                pnl = (entry_price - exit_price) / entry_price * 100  # % PnL for short
                trades.append({
                    'type': 'short_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl
                })
            
            # Open long position
            current_position = 'long'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
        
        elif bearish_signals.iloc[i] and current_position != 'short':
            # Close any existing long position
            if current_position == 'long':
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price * 100  # % PnL for long
                trades.append({
                    'type': 'long_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl
                })
            
            # Open short position
            current_position = 'short'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
    
    # Close final position if still open
    if current_position is not None:
        exit_price = df['close'].iloc[-1]
        if current_position == 'long':
            pnl = (exit_price - entry_price) / entry_price * 100
            trade_type = 'long_close'
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
            trade_type = 'short_close'
        
        trades.append({
            'type': trade_type,
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl
        })
    
    return trades

def print_trade_results(trades):
    """Print trade results and summary statistics"""
    if not trades:
        print("No trades were executed.")
        return
    
    print("=" * 80)
    print("TRADE RESULTS")
    print("=" * 80)
    
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    
    for i, trade in enumerate(trades, 1):
        pnl_sign = "+" if trade['pnl_pct'] >= 0 else ""
        trade_direction = "LONG" if trade['type'] == 'long_close' else "SHORT"
        
        print(f"Trade {i}: {trade_direction}")
        print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Exit:  {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"  PnL:   {pnl_sign}{trade['pnl_pct']:.2f}%")
        print("-" * 40)
        
        total_pnl += trade['pnl_pct']
        if trade['pnl_pct'] >= 0:
            winning_trades += 1
        else:
            losing_trades += 1
    
    # Summary statistics
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print("SUMMARY STATISTICS:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}%")
    print(f"Average PnL per Trade: {total_pnl/total_trades:.2f}%" if total_trades > 0 else "N/A")

def main():
    """Main function to run the trading simulation"""
    print("Loading data and calculating SMA crossover strategy...")
    
    # Load and process data
    df = load_and_process_data()
    if df is None:
        return
    
    print(f"Data loaded successfully. {len(df)} rows processed.")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate SMA and trading signals
    bullish_signals, bearish_signals = calculate_sma_and_trades(df)
    
    # Simulate trading
    trades = simulate_trading(df, bullish_signals, bearish_signals)
    
    # Print results
    print_trade_results(trades)

if __name__ == "__main__":
    main()
