import pandas as pd
import numpy as np
import time
from datetime import datetime
import openai
import os 

# Initialize DeepSeek client
client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def consult_deepseek_for_regime_change(df, current_index, signal_type):
    """
    Consult DeepSeek AI to determine if a signal represents a market regime change
    """
    # Get recent price action context (last 50 periods)
    start_idx = max(0, current_index - 50)
    recent_data = df.iloc[start_idx:current_index+1]
    
    # Prepare context for AI
    price_trend = "Bullish" if df['close'].iloc[current_index] > df['close'].iloc[start_idx] else "Bearish"
    volatility = recent_data['high'].max() - recent_data['low'].min()
    current_price = df['close'].iloc[current_index]
    sma_value = df['sma_200_day'].iloc[current_index] if not pd.isna(df['sma_200_day'].iloc[current_index]) else "N/A"
    
    # Create prompt for DeepSeek
    prompt = f"""
    As a financial market expert, analyze this trading situation and determine if it represents a genuine market regime change:
    
    Current Situation:
    - Signal Type: {signal_type} crossover
    - Current Price: ${current_price:.2f}
    - 200-Day SMA: ${sma_value:.2f} (if available)
    - Recent Price Trend: {price_trend}
    - Recent Volatility: {volatility:.2f} points
    
    Recent Price Action (last 50 periods):
    - Highest Price: ${recent_data['high'].max():.2f}
    - Lowest Price: ${recent_data['low'].min():.2f}
    - Average Range: ${(recent_data['high'] - recent_data['low']).mean():.2f}
    
    Based on your expertise in market regime changes and technical analysis, does this crossover signal represent:
    1. A genuine market regime change (respond with "YES")
    2. Just a temporary fluctuation (respond with "NO")
    
    Provide only a one-word response: "YES" or "NO"
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Lower temperature for more deterministic responses
        )
        print("Prompt: ", prompt)
        
        answer = response.choices[0].message.content.strip().upper()
        print("ANSWER: ", answer)
        return answer == "YES"
        
    except Exception as e:
        print(f"Error consulting DeepSeek: {e}")
        # Default to proceeding with trade if AI consultation fails
        return True

def load_and_process_data():
    """Load and process the CSV data with proper ISO8601 parsing"""
    try:
        # Load the CSV file with proper datetime parsing
        df = pd.read_csv('xbtusd_1h_8y.csv')
        
        # Convert time columns to datetime using ISO8601 format
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
    """Calculate 200-day SMA (4800 hours) and additional indicators"""
    if df is None or len(df) < 4800:
        print(f"Not enough data for 200-day SMA calculation. Need at least 4800 hours, have {len(df)}")
        return df, [], []
    
    # Calculate 200-day SMA (200 days * 24 hours/day = 4800 hours)
    df['sma_200_day'] = df['close'].rolling(window=4800).mean()
    
    # Calculate additional metrics for analysis
    df['daily_volatility'] = (df['high'] - df['low']) / df['close'] * 100  # % volatility
    df['sma_distance'] = (df['close'] - df['sma_200_day']) / df['sma_200_day'] * 100  # % from SMA
    
    # Identify crossover points
    df['prev_close'] = df['close'].shift(1)
    df['prev_sma'] = df['sma_200_day'].shift(1)
    
    # Bullish crossover (price crosses above SMA)
    bullish_cross = (df['close'] > df['sma_200_day']) & (df['prev_close'] <= df['prev_sma'])
    
    # Bearish crossover (price crosses below SMA)
    bearish_cross = (df['close'] < df['sma_200_day']) & (df['prev_close'] >= df['prev_sma'])
    
    return df, bullish_cross, bearish_cross

def simulate_trading(df, bullish_signals, bearish_signals):
    """Simulate trading with DeepSeek consultation for regime changes"""
    trades = []
    current_position = None
    entry_price = 0
    entry_time = None
    entry_index = None
    equity_curve = [10000]  # Start with $10,000 for equity curve
    max_drawdown = 0
    peak_equity = 10000
    ai_consultations = 0
    ai_rejections = 0
    
    # Only start trading after we have enough data for SMA calculation
    start_index = 4800
    
    for i in range(start_index, len(df)):
        current_equity = equity_curve[-1]
        should_trade = False
        signal_type = None
        
        # Check for bullish signal
        if bullish_signals.iloc[i] and current_position != 'long':
            signal_type = "BULLISH"
            print(f"\n🔍 Consulting DeepSeek about potential BULLISH regime change at {df.index[i]}...")
            should_trade = consult_deepseek_for_regime_change(df, i, signal_type)
            ai_consultations += 1
            if not should_trade:
                ai_rejections += 1
                print("❌ DeepSeek rejected this trade (not a regime change)")
        
        # Check for bearish signal    
        elif bearish_signals.iloc[i] and current_position != 'short':
            signal_type = "BEARISH"
            print(f"\n🔍 Consulting DeepSeek about potential BEARISH regime change at {df.index[i]}...")
            should_trade = consult_deepseek_for_regime_change(df, i, signal_type)
            ai_consultations += 1
            if not should_trade:
                ai_rejections += 1
                print("❌ DeepSeek rejected this trade (not a regime change)")
        
        # Execute trade if approved by AI
        if should_trade and signal_type == "BULLISH" and current_position != 'long':
            # Close any existing short position
            if current_position == 'short':
                exit_price = df['close'].iloc[i]
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                pnl_dollar = (entry_price - exit_price) * (current_equity / entry_price)
                trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)
                
                current_equity += pnl_dollar
                equity_curve.append(current_equity)
                
                # Update max drawdown
                if current_equity > peak_equity:
                    peak_equity = current_equity
                drawdown = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)
                
                trades.append({
                    'type': 'short_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_dollar': pnl_dollar,
                    'duration_days': trade_duration,
                    'equity_after': current_equity
                })
            
            # Open long position
            current_position = 'long'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
            entry_index = i
            print(f"✅ DeepSeek approved LONG entry at {entry_time}, price: ${entry_price:.2f}")
        
        elif should_trade and signal_type == "BEARISH" and current_position != 'short':
            # Close any existing long position
            if current_position == 'long':
                exit_price = df['close'].iloc[i]
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                pnl_dollar = (exit_price - entry_price) * (current_equity / entry_price)
                trade_duration = (df.index[i] - entry_time).total_seconds() / (24 * 3600)
                
                current_equity += pnl_dollar
                equity_curve.append(current_equity)
                
                # Update max drawdown
                if current_equity > peak_equity:
                    peak_equity = current_equity
                drawdown = (peak_equity - current_equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)
                
                trades.append({
                    'type': 'long_close',
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_dollar': pnl_dollar,
                    'duration_days': trade_duration,
                    'equity_after': current_equity
                })
            
            # Open short position
            current_position = 'short'
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
            entry_index = i
            print(f"✅ DeepSeek approved SHORT entry at {entry_time}, price: ${entry_price:.2f}")
        
        # Update equity curve even when no trade occurs
        if len(equity_curve) < i - start_index + 1:
            equity_curve.append(current_equity)
    
    # Close final position if still open
    if current_position is not None:
        exit_price = df['close'].iloc[-1]
        trade_duration = (df.index[-1] - entry_time).total_seconds() / (24 * 3600)
        
        if current_position == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_dollar = (exit_price - entry_price) * (equity_curve[-1] / entry_price)
            trade_type = 'long_close'
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl_dollar = (entry_price - exit_price) * (equity_curve[-1] / entry_price)
            trade_type = 'short_close'
        
        final_equity = equity_curve[-1] + pnl_dollar
        equity_curve.append(final_equity)
        
        trades.append({
            'type': trade_type,
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'duration_days': trade_duration,
            'equity_after': final_equity
        })
    
    # Add AI consultation stats to results
    ai_stats = {
        'consultations': ai_consultations,
        'rejections': ai_rejections,
        'approval_rate': (ai_consultations - ai_rejections) / ai_consultations * 100 if ai_consultations > 0 else 0
    }
    
    return trades, equity_curve, max_drawdown, ai_stats

def print_trade_results(trades, equity_curve, max_drawdown, ai_stats):
    """Print trade results and summary statistics with 0.5 second delay between trades"""
    if not trades:
        print("No trades were executed.")
        return
    
    print("=" * 100)
    print("TRADE RESULTS - AI-ENHANCED 200-DAY SMA CROSSOVER STRATEGY")
    print("=" * 100)
    print("Printing trades with 0.5 second delay...")
    print()
    
    total_pnl_pct = 0
    total_pnl_dollar = 0
    winning_trades = 0
    losing_trades = 0
    total_duration = 0
    best_trade = {'pnl_pct': -100}
    worst_trade = {'pnl_pct': 100}
    
    for i, trade in enumerate(trades, 1):
        time.sleep(0.5)
        
        pnl_sign = "+" if trade['pnl_pct'] >= 0 else ""
        trade_direction = "LONG" if trade['type'] == 'long_close' else "SHORT"
        pnl_color = "\033[92m" if trade['pnl_pct'] >= 0 else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"Trade {i}: {trade_direction}")
        print(f"  Entry:    {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Exit:     {trade['exit_time']} @ ${trade['exit_price']:.2f}")
        print(f"  Duration: {trade['duration_days']:.1f} days")
        print(f"  PnL:      {pnl_color}{pnl_sign}{trade['pnl_pct']:.2f}% ({pnl_sign}${trade['pnl_dollar']:.2f}){reset_color}")
        print(f"  Equity:   ${trade['equity_after']:.2f}")
        print("-" * 70)
        
        total_pnl_pct += trade['pnl_pct']
        total_pnl_dollar += trade['pnl_dollar']
        total_duration += trade['duration_days']
        
        if trade['pnl_pct'] >= 0:
            winning_trades += 1
        else:
            losing_trades += 1
        
        # Track best and worst trades
        if trade['pnl_pct'] > best_trade['pnl_pct']:
            best_trade = trade
        if trade['pnl_pct'] < worst_trade['pnl_pct']:
            worst_trade = trade
    
    # Summary statistics
    time.sleep(0.5)
    print("\n" + "=" * 70)
    print("🎯 PERFORMANCE SUMMARY")
    print("=" * 70)
    
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_duration = total_duration / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl_pct / total_trades if total_trades > 0 else 0
    
    initial_equity = 10000
    final_equity = equity_curve[-1]
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100
    
    print(f"💰 Total Return: {total_return_pct:+.2f}% (${final_equity:,.2f})")
    print(f"📊 Total Trades: {total_trades}")
    print(f"✅ Winning Trades: {winning_trades} ({win_rate:.1f}%)")
    print(f"❌ Losing Trades: {losing_trades}")
    print(f"📈 Average PnL per Trade: {avg_pnl:+.2f}%")
    print(f"⏰ Average Trade Duration: {avg_duration:.1f} days")
    print(f"📉 Maximum Drawdown: {max_drawdown:.2f}%")
    
    # AI Consultation Stats
    time.sleep(0.5)
    print(f"\n🤖 AI CONSULTATION STATS")
    print(f"   Consultations: {ai_stats['consultations']}")
    print(f"   Rejections: {ai_stats['rejections']}")
    print(f"   Approval Rate: {ai_stats['approval_rate']:.1f}%")
    
    # Best and worst trades
    time.sleep(0.5)
    print(f"\n🏆 Best Trade: {best_trade['pnl_pct']:+.2f}%")
    print(f"   {best_trade['entry_time']} → {best_trade['exit_time']}")
    print(f"   ${best_trade['entry_price']:.2f} → ${best_trade['exit_price']:.2f}")
    
    print(f"\n💥 Worst Trade: {worst_trade['pnl_pct']:+.2f}%")
    print(f"   {worst_trade['entry_time']} → {worst_trade['exit_time']}")
    print(f"   ${worst_trade['entry_price']:.2f} → ${worst_trade['exit_price']:.2f}")
    
    # Final assessment
    time.sleep(0.5)
    print("\n" + "=" * 70)
    print("📋 STRATEGY ASSESSMENT")
    print("=" * 70)
    
    if total_return_pct >= 100:
        print("🚀 EXTRAORDINARY PERFORMANCE! (>100% return)")
    elif total_return_pct >= 50:
        print("🎯 EXCELLENT PERFORMANCE! (>50% return)")
    elif total_return_pct >= 20:
        print("👍 GOOD PERFORMANCE! (>20% return)")
    elif total_return_pct >= 0:
        print("📗 PROFITABLE STRATEGY")
    else:
        print("📘 UNPROFITABLE STRATEGY")
    
    if win_rate >= 60:
        print("🎯 High win rate suggests consistent strategy")
    elif win_rate >= 40:
        print("📊 Moderate win rate")
    else:
        print("⚠️  Low win rate - strategy may be volatile")

def main():
    """Main function to run the trading simulation"""
    print("Loading data and calculating 200-day SMA crossover strategy...")
    print("Note: 200-day SMA requires 4800 hours of data (200 days × 24 hours/day)")
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  DEEPSEEK_API_KEY environment variable not set. AI consultation will not work.")
        print("You can set it with: export DEEPSEEK_API_KEY='your-api-key-here'")
    
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
    
    # Simulate trading with AI consultation
    print("\nSimulating trades with DeepSeek AI consultation...")
    trades, equity_curve, max_drawdown, ai_stats = simulate_trading(df, bullish_signals, bearish_signals)
    
    # Print results with delay
    print_trade_results(trades, equity_curve, max_drawdown, ai_stats)

if __name__ == "__main__":
    main()
