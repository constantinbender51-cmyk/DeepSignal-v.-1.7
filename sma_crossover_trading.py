import pandas as pd
import numpy as np

def load_and_process_data():
    """Load and process the CSV data with proper ISO8601 parsing"""
    try:
        # Load the CSV file with proper datetime parsing
        df = pd.read_csv('xbtusd_1h_8y.csv')
        
        # Convert time columns to datetime using ISO8601 format
        # This handles both formats: with and without milliseconds
        df['open_time'] = pd.to_datetime(df['open_time'], format='ISO8601')
        df['close_time'] = pd.to_datetime(df['close_time'], format='ISO8601')
        
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

def calculate_200_day_sma(df):
    """Calculate 200-day SMA (4800 hours)"""
    if df is None or len(df) < 4800:
        print(f"Not enough data for 200-day SMA calculation. Need at least 4800 hours, have {len(df)}")
        return df, [], []
    
    # Calculate 200-day SMA (200 days * 24 hours/day = 4800 hours)
    df['sma_200_day'] = df['close'].rolling(window=4800).mean()
    
    # Identify crossover points
    df['prev_close'] = df['close'].shift(1)
    df['prev_sma'] = df['sma_200_day'].shift(1)
    
    # Bullish crossover (price crosses above SMA)
    bullish_cross = (df['close'] > df['sma_200_day']) & (df['prev_close'] <= df['prev_sma'])
    
    # Bearish crossover (price crosses below SMA)
    bearish_cross = (df['close'] < df['sma_200_day']) & (df['prev_close'] >= df['prev_sma'])
    
    return df, bullish_cross, bearish_cross

def simulate_trading(df, bullish_signals, bearish_signals):
    """Simulate trading and calculate PnL"""
    trades = []
    current_position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    entry_index = None
    
    # Only start trading after we have enough data for SMA calculation
    start_index = 4800  # Start after we have 200 days of data
    
    for i in range(start_index, len(df)):
        if bullish_signals.iloc[i] and current_position != 'long':
            # Close any existing short position
            if current_position == 'short':
                exit_price = df['close'].iloc[i]
                pnl = (entry_price - exit_price) / entry_price * 100  # % PnL for short
                trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)  # days
                
                trades.append({
                    'type': 'short_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'duration_days': trade_duration
                })
            
            # Open long position
            current_position = 'long'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
            entry_index = i
        
        elif bearish_signals.iloc[i] and current_position != 'short':
            # Close any existing long position
            if current_position == 'long':
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price * 100  # % PnL for long
                trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)  # days
                
                trades.append({
                    'type': 'long_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'duration_days': trade_duration
                })
            
            # Open short position
            current_position = 'short'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
            entry_index = i
    
    # Close final position if still open
    if current_position is not None:
        exit_price = df['close'].iloc[-1]
        trade_duration = (df.index[-1] - entry_time).total_seconds() / (24 * 3600)  # days
        
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
            'pnl_pct': pnl,
            'duration_days': trade_duration
        })
    
    return trades

def print_trade_results(trades):
    """Print trade results and summary statistics"""
    if not trades:
        print("No trades were executed.")
        return
    
    print("=" * 100)
    print("TRADE RESULTS - 200-DAY SMA CROSSOVER STRATEGY")
    print("=" * 100)
    
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    total_duration = 0
    
    for i, trade in enumerate(trades, 1):
        pnl_sign = "+" if trade['pnl_pct'] >= 0 else ""
        trade_direction = "LONG" if trade['type'] == 'long_close' else "SHORT"
        
        print(f"Trade {i}: {trade_direction}")
        print(f"  Entry:    {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Exit:     {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"  Duration: {trade['duration_days']:.1f} days")
        print(f"  PnL:      {pnl_sign}{trade['pnl_pct']:.2f}%")
        print("-" * 60)
        
        total_pnl += trade['pnl_pct']
        total_duration += trade['duration_days']
        if trade['pnl_pct'] >= 0:
            winning_trades += 1
        else:
            losing_trades += 1
    
    # Summary statistics
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_duration = total_duration / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    print("\nSUMMARY STATISTICS:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}%")
    print(f"Average PnL per Trade: {avg_pnl:.2f}%")
    print(f"Average Trade Duration: {avg_duration:.1f} days")

def main():
    """Main function to run the trading simulation"""
    print("Loading data and calculating 200-day SMA crossover strategy...")
    print("Note: 200-day SMA requires 4800 hours of data (200 days × 24 hours/day)")
    
    # Load and process data
    df = load_and_process_data()
    if df is None:
        return
    
    print(f"Data loaded successfully. {len(df)} hourly bars processed.")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total duration: {(df.index.max() - df.index.min()).days} days")
    
    # Calculate 200-day SMA and trading signals
    df, bullish_signals, bearish_signals = calculate_200_day_sma(df)
    
    if df is None:
        return
    
    # Count signals
    bullish_count = bullish_signals.sum()
    bearish_count = bearish_signals.sum()
    
    print(f"\nSignal Analysis:")
    print(f"Bullish crossovers (price > 200-day SMA): {bullish_count}")
    print(f"Bearish crossovers (price < 200-day SMA): {bearish_count}")
    
    # Simulate trading
    trades = simulate_trading(df, bullish_signals, bearish_signals)
    
    # Print results
    print_trade_results(trades)

if __name__ == "__main__":
    main()
