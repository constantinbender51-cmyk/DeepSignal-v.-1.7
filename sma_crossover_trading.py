import pandas as pd
import numpy as np
import time

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

def calculate_sma_crossovers(df):
    """Calculate 50-day and 200-day SMA and their crossover signals"""
    # Calculate window sizes in hours (50 days * 24 hours, 200 days * 24 hours)
    window_50d = 50 * 24  # 1200 hours
    window_200d = 200 * 24  # 4800 hours
    
    if df is None or len(df) < window_200d:
        print(f"Not enough data for SMA calculations. Need at least {window_200d} hours, have {len(df)}")
        return df, [], []
    
    # Calculate both SMAs
    df['sma_50_day'] = df['close'].rolling(window=window_50d).mean()
    df['sma_200_day'] = df['close'].rolling(window=window_200d).mean()
    
    # Identify crossover points (50-day SMA crossing 200-day SMA)
    df['prev_sma50'] = df['sma_50_day'].shift(1)
    df['prev_sma200'] = df['sma_200_day'].shift(1)
    
    # Golden cross (50-day SMA crosses above 200-day SMA) - BULLISH
    golden_cross = (df['sma_50_day'] > df['sma_200_day']) & (df['prev_sma50'] <= df['prev_sma200'])
    
    # Death cross (50-day SMA crosses below 200-day SMA) - BEARISH
    death_cross = (df['sma_50_day'] < df['sma_200_day']) & (df['prev_sma50'] >= df['prev_sma200'])
    
    return df, golden_cross, death_cross

def simulate_trading(df, golden_cross_signals, death_cross_signals):
    """Simulate trading based on 50-day/200-day SMA crossovers"""
    trades = []
    current_position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    entry_index = None
    
    # Only start trading after we have enough data for both SMA calculations
    start_index = 200 * 24  # 4800 hours for 200-day SMA
    
    for i in range(start_index, len(df)):
        if golden_cross_signals.iloc[i] and current_position != 'long':
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
                    'duration_days': trade_duration,
                    'signal': 'death_cross_exit'
                })
            
            # Open long position on golden cross
            current_position = 'long'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
            entry_index = i
        
        elif death_cross_signals.iloc[i] and current_position != 'short':
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
                    'duration_days': trade_duration,
                    'signal': 'golden_cross_exit'
                })
            
            # Open short position on death cross
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
            signal_type = 'final_close'
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
            trade_type = 'short_close'
            signal_type = 'final_close'
        
        trades.append({
            'type': trade_type,
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'duration_days': trade_duration,
            'signal': signal_type
        })
    
    return trades

def print_trade_results(trades):
    """Print trade results and summary statistics with 0.5 second delay between trades"""
    if not trades:
        print("No trades were executed.")
        return
    
    print("=" * 100)
    print("TRADE RESULTS - 50-DAY/200-DAY SMA CROSSOVER STRATEGY")
    print("=" * 100)
    print("Printing trades with 0.5 second delay...")
    print()
    
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    total_duration = 0
    
    for i, trade in enumerate(trades, 1):
        # Add 0.5 second delay before printing each trade
        time.sleep(0.5)
        
        pnl_sign = "+" if trade['pnl_pct'] >= 0 else ""
        trade_direction = "LONG" if trade['type'] == 'long_close' else "SHORT"
        signal_type = "GOLDEN CROSS" if 'golden' in trade.get('signal', '') else "DEATH CROSS"
        pnl_color = "\033[92m" if trade['pnl_pct'] >= 0 else "\033[91m"  # Green for profit, red for loss
        reset_color = "\033[0m"
        
        print(f"Trade {i}: {trade_direction} ({signal_type})")
        print(f"  Entry:    {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Exit:     {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"  Duration: {trade['duration_days']:.1f} days")
        print(f"  PnL:      {pnl_color}{pnl_sign}{trade['pnl_pct']:.2f}%{reset_color}")
        print("-" * 60)
        
        total_pnl += trade['pnl_pct']
        total_duration += trade['duration_days']
        if trade['pnl_pct'] >= 0:
            winning_trades += 1
        else:
            losing_trades += 1
    
    # Add delay before summary
    time.sleep(0.5)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)
    
    # Summary statistics
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_duration = total_duration / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}%")
    print(f"Average PnL per Trade: {avg_pnl:.2f}%")
    print(f"Average Trade Duration: {avg_duration:.1f} days")
    
    # Final performance assessment
    time.sleep(0.5)
    print("\n" + "=" * 60)
    print("PERFORMANCE ASSESSMENT:")
    print("=" * 60)
    if total_pnl > 0:
        print("✅ Strategy was PROFITABLE")
    else:
        print("❌ Strategy was NOT PROFITABLE")
    print(f"Final Return: {total_pnl:.2f}%")

def main():
    """Main function to run the trading simulation"""
    print("Loading data and calculating 50-day/200-day SMA crossover strategy...")
    print("Note: 200-day SMA requires 4800 hours of data (200 days × 24 hours/day)")
    print("Note: 50-day SMA requires 1200 hours of data (50 days × 24 hours/day)")
    
    # Load and process data
    df = load_and_process_data()
    if df is None:
        return
    
    print(f"Data loaded successfully. {len(df)} hourly bars processed.")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total duration: {(df.index.max() - df.index.min()).days} days")
    
    # Calculate SMA crossovers
    df, golden_cross_signals, death_cross_signals = calculate_sma_crossovers(df)
    
    if df is None:
        return
    
    # Count signals
    golden_count = golden_cross_signals.sum()
    death_count = death_cross_signals.sum()
    
    print(f"\nSignal Analysis:")
    print(f"Golden Cross signals (50-day SMA > 200-day SMA): {golden_count}")
    print(f"Death Cross signals (50-day SMA < 200-day SMA): {death_count}")
    
    # Simulate trading
    print("\nSimulating trades...")
    trades = simulate_trading(df, golden_cross_signals, death_cross_signals)
    
    # Print results with delay
    print_trade_results(trades)

if __name__ == "__main__":
    main()
