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
    """Calculate 5-day and 200-day SMA and their crossover signals"""
    # Calculate window sizes in hours (5 days * 24 hours, 200 days * 24 hours)
    window_5d = 5 * 24  # 120 hours
    window_200d = 200 * 24  # 4800 hours
    
    if df is None or len(df) < window_200d:
        print(f"Not enough data for SMA calculations. Need at least {window_200d} hours, have {len(df)}")
        return df, [], []
    
    # Calculate both SMAs
    df['sma_5_day'] = df['close'].rolling(window=window_5d).mean()
    df['sma_200_day'] = df['close'].rolling(window=window_200d).mean()
    
    # Identify crossover points (5-day SMA crossing 200-day SMA)
    df['prev_sma5'] = df['sma_5_day'].shift(1)
    df['prev_sma200'] = df['sma_200_day'].shift(1)
    
    # Bullish cross (5-day SMA crosses above 200-day SMA)
    bullish_cross = (df['sma_5_day'] > df['sma_200_day']) & (df['prev_sma5'] <= df['prev_sma200'])
    
    # Bearish cross (5-day SMA crosses below 200-day SMA)
    bearish_cross = (df['sma_5_day'] < df['sma_200_day']) & (df['prev_sma5'] >= df['prev_sma200'])
    
    return df, bullish_cross, bearish_cross

def calculate_rainbow_bubble_zones(df):
    """Calculate Rainbow Chart v2 bubble detection zones"""
    # Rainbow Chart v2 uses multiple exponential moving averages
    # For bubble detection, we'll use the 2x multiplier above 350WMA as bubble territory
    # (Based on typical Rainbow Chart bubble indicators)
    
    # Calculate 350-week moving average (350 * 7 * 24 = 58800 hours)
    window_350w = 350 * 7 * 24  # 58800 hours (350 weeks)
    
    if len(df) < window_350w:
        print(f"Warning: Not enough data for full Rainbow Chart calculation. Need {window_350w} hours, have {len(df)}")
        # Use available data but mark as insufficient for full analysis
        df['wma_350'] = df['close'].rolling(window=min(len(df), window_350w)).mean()
    else:
        df['wma_350'] = df['close'].rolling(window=window_350w).mean()
    
    # Calculate bubble thresholds (2x and 3x 350WMA - common bubble indicators)
    df['bubble_zone_2x'] = df['wma_350'] * 2
    df['bubble_zone_3x'] = df['wma_350'] * 3
    
    # Detect when price enters bubble territory
    df['in_bubble_2x'] = df['close'] > df['bubble_zone_2x']
    df['in_bubble_3x'] = df['close'] > df['bubble_zone_3x']
    
    # Extreme bubble (price > 3x 350WMA)
    df['extreme_bubble'] = df['in_bubble_3x']
    
    return df

def simulate_trading(df, bullish_cross_signals, bearish_cross_signals):
    """Simulate trading based on 5-day/200-day SMA crossovers with bubble reversal"""
    trades = []
    current_position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    entry_index = None
    
    # Only start trading after we have enough data for both SMA calculations
    start_index = 200 * 24  # 4800 hours for 200-day SMA
    
    for i in range(start_index, len(df)):
        current_price = df['close'].iloc[i]
        in_bubble_2x = df['in_bubble_2x'].iloc[i] if 'in_bubble_2x' in df.columns else False
        in_bubble_3x = df['in_bubble_3x'].iloc[i] if 'in_bubble_3x' in df.columns else False
        
        # BUBBLE REVERSAL LOGIC
        # If in extreme bubble territory (3x 350WMA), reverse position
        if in_bubble_3x and current_position == 'long':
            # Close long position due to extreme bubble
            exit_price = current_price
            pnl = (exit_price - entry_price) / entry_price * 100
            trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)
            
            trades.append({
                'type': 'long_close',
                'entry_time': entry_time,
                'exit_time': df.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'duration_days': trade_duration,
                'signal': 'bubble_reversal_exit',
                'bubble_level': 'extreme_3x'
            })
            
            # Open short position in extreme bubble
            current_position = 'short'
            entry_price = current_price
            entry_time = df.index[i]
            entry_index = i
            continue
        
        # REGULAR SMA CROSSOVER LOGIC
        if bullish_cross_signals.iloc[i] and current_position != 'long':
            # Close any existing short position (unless we're in bubble territory where we might want to hold)
            if current_position == 'short' and not in_bubble_2x:
                exit_price = current_price
                pnl = (entry_price - exit_price) / entry_price * 100
                trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)
                
                trades.append({
                    'type': 'short_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'duration_days': trade_duration,
                    'signal': 'bearish_cross_exit',
                    'bubble_level': 'normal' if not in_bubble_2x else 'bubble_2x'
                })
            
            # Open long position on bullish cross (unless in bubble territory)
            if not in_bubble_2x:
                current_position = 'long'
                entry_price = current_price
                entry_time = df.index[i]
                entry_index = i
        
        elif bearish_cross_signals.iloc[i] and current_position != 'short':
            # Close any existing long position
            if current_position == 'long':
                exit_price = current_price
                pnl = (exit_price - entry_price) / entry_price * 100
                trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)
                
                trades.append({
                    'type': 'long_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'duration_days': trade_duration,
                    'signal': 'bullish_cross_exit',
                    'bubble_level': 'normal' if not in_bubble_2x else 'bubble_2x'
                })
            
            # Open short position on bearish cross (or in bubble territory)
            current_position = 'short'
            entry_price = current_price
            entry_time = df.index[i]
            entry_index = i
    
    # Close final position if still open
    if current_position is not None:
        exit_price = df['close'].iloc[-1]
        trade_duration = (df.index[-1] - entry_time).total_seconds() / (24 * 3600)
        
        if current_position == 'long':
            pnl = (exit_price - entry_price) / entry_price * 100
            trade_type = 'long_close'
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
            trade_type = 'short_close'
        
        # Check if final close was in bubble territory
        final_bubble_2x = df['in_bubble_2x'].iloc[-1] if 'in_bubble_2x' in df.columns else False
        final_bubble_3x = df['in_bubble_3x'].iloc[-1] if 'in_bubble_3x' in df.columns else False
        
        bubble_level = 'normal'
        if final_bubble_3x:
            bubble_level = 'extreme_3x'
        elif final_bubble_2x:
            bubble_level = 'bubble_2x'
        
        trades.append({
            'type': trade_type,
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'duration_days': trade_duration,
            'signal': 'final_close',
            'bubble_level': bubble_level
        })
    
    return trades

def print_trade_results(trades):
    """Print trade results and summary statistics with 0.5 second delay between trades"""
    if not trades:
        print("No trades were executed.")
        return
    
    print("=" * 120)
    print("TRADE RESULTS - 5-DAY/200-DAY SMA CROSSOVER WITH BUBBLE REVERSAL STRATEGY")
    print("=" * 120)
    print("Printing trades with 0.5 second delay...")
    print()
    
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    total_duration = 0
    bubble_trades = 0
    
    for i, trade in enumerate(trades, 1):
        # Add 0.5 second delay before printing each trade
        time.sleep(0.5)
        
        pnl_sign = "+" if trade['pnl_pct'] >= 0 else ""
        trade_direction = "LONG" if trade['type'] == 'long_close' else "SHORT"
        signal_type = trade.get('signal', '').upper().replace('_', ' ')
        bubble_level = trade.get('bubble_level', 'normal').upper()
        
        pnl_color = "\033[92m" if trade['pnl_pct'] >= 0 else "\033[91m"
        reset_color = "\033[0m"
        
        bubble_indicator = ""
        if bubble_level != 'NORMAL':
            bubble_indicator = f" [{bubble_level}]"
            if 'EXTREME' in bubble_level:
                bubble_indicator = f" \033[91m⦿ {bubble_level}\033[0m"
            elif 'BUBBLE' in bubble_level:
                bubble_indicator = f" \033[93m⚠️  {bubble_level}\033[0m"
        
        print(f"Trade {i}: {trade_direction} {bubble_indicator}")
        print(f"  Signal:   {signal_type}")
        print(f"  Entry:    {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Exit:     {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"  Duration: {trade['duration_days']:.1f} days")
        print(f"  PnL:      {pnl_color}{pnl_sign}{trade['pnl_pct']:.2f}%{reset_color}")
        print("-" * 70)
        
        total_pnl += trade['pnl_pct']
        total_duration += trade['duration_days']
        if trade['pnl_pct'] >= 0:
            winning_trades += 1
        else:
            losing_trades += 1
        
        if bubble_level != 'NORMAL':
            bubble_trades += 1
    
    # Add delay before summary
    time.sleep(0.5)
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS:")
    print("=" * 70)
    
    # Summary statistics
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_duration = total_duration / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    bubble_percentage = (bubble_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Bubble Zone Trades: {bubble_trades} ({bubble_percentage:.1f}%)")
    print(f"Total PnL: {total_pnl:.2f}%")
    print(f"Average PnL per Trade: {avg_pnl:.2f}%")
    print(f"Average Trade Duration: {avg_duration:.1f} days")
    
    # Final performance assessment
    time.sleep(0.5)
    print("\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT:")
    print("=" * 70)
    if total_pnl > 0:
        print("✅ Strategy was PROFITABLE")
    else:
        print("❌ Strategy was NOT PROFITABLE")
    print(f"Final Return: {total_pnl:.2f}%")

def main():
    """Main function to run the trading simulation"""
    print("Loading data and calculating 5-day/200-day SMA crossover with bubble reversal strategy...")
    print("Note: 200-day SMA requires 4800 hours of data (200 days × 24 hours/day)")
    print("Note: 5-day SMA requires 120 hours of data (5 days × 24 hours/day)")
    print("Note: Rainbow Chart uses 350-week moving average for bubble detection")
    
    # Load and process data
    df = load_and_process_data()
    if df is None:
        return
    
    print(f"Data loaded successfully. {len(df)} hourly bars processed.")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total duration: {(df.index.max() - df.index.min()).days} days")
    
    # Calculate SMA crossovers
    df, bullish_cross_signals, bearish_cross_signals = calculate_sma_crossovers(df)
    
    if df is None:
        return
    
    # Calculate Rainbow Chart bubble zones
    df = calculate_rainbow_bubble_zones(df)
    
    # Count signals
    bullish_count = bullish_cross_signals.sum()
    bearish_count = bearish_cross_signals.sum()
    
    # Count bubble periods
    bubble_2x_count = df['in_bubble_2x'].sum() if 'in_bubble_2x' in df.columns else 0
    bubble_3x_count = df['in_bubble_3x'].sum() if 'in_bubble_3x' in df.columns else 0
    
    print(f"\nSignal Analysis:")
    print(f"Bullish Cross signals (5-day SMA > 200-day SMA): {bullish_count}")
    print(f"Bearish Cross signals (5-day SMA < 200-day SMA): {bearish_count}")
    print(f"Bubble Zone periods (Price > 2x 350WMA): {bubble_2x_count} hours")
    print(f"Extreme Bubble periods (Price > 3x 350WMA): {bubble_3x_count} hours")
    
    # Simulate trading
    print("\nSimulating trades with bubble reversal logic...")
    trades = simulate_trading(df, bullish_cross_signals, bearish_cross_signals)
    
    # Print results with delay
    print_trade_results(trades)

if __name__ == "__main__":
    main()
