import pandas as pd
import numpy as np
import io
from typing import List, Dict

def check_sma_crossover(candles: List[Dict], short_period: int, long_period: int) -> bool:
    """
    Checks for a Simple Moving Average (SMA) crossover between a short and a long period.
    
    This function acts as a flexible preselection filter. It returns True if a
    crossover event (up or down) has occurred on the most recent candle,
    indicating that a signal should be generated. Otherwise, it returns False.
    
    Args:
        candles (List[Dict]): A list of candle dictionaries, with each dictionary
                              containing a 'c' (close) price. The list should
                              be at least as long as the 'long_period' to calculate
                              the SMAs correctly.
        short_period (int): The period for the shorter SMA (e.g., 20).
        long_period (int): The period for the longer SMA (e.g., 50).
    
    Returns:
        bool: True if an SMA crossover is detected on the last candle,
              False otherwise.
    """
    # Ensure we have enough data to calculate the longest SMA.
    if len(candles) < long_period:
        print(f"Warning: Not enough data for SMA calculation. Requires at least {long_period} candles.")
        return False

    # Extract the 'c' (close) prices into a pandas Series for easy calculation.
    close_prices = pd.Series([c['c'] for c in candles])

    # Calculate the SMAs for the entire series.
    sma_short = close_prices.rolling(window=short_period).mean()
    sma_long = close_prices.rolling(window=long_period).mean()

    # Get the current and previous values for both SMAs.
    # We use .iloc[-1] for the current value and .iloc[-2] for the previous value.
    sma_short_current = sma_short.iloc[-1]
    sma_long_current = sma_long.iloc[-1]
    sma_short_prev = sma_short.iloc[-2]
    sma_long_prev = sma_long.iloc[-2]

    # Check for a bullish crossover (short SMA crosses above long SMA).
    is_bullish_crossover = (sma_short_prev <= sma_long_prev) and (sma_short_current > sma_long_current)

    # Check for a bearish crossover (short SMA crosses below long SMA).
    is_bearish_crossover = (sma_short_prev >= sma_long_prev) and (sma_short_current < sma_long_current)

    # Return True if either crossover has occurred.
    return is_bullish_crossover or is_bearish_crossover

if __name__ == "__main__":
    # --- Example Usage with real CSV data from the file xbtusd_1h_8y.csv ---
    
    try:
        # Read the data directly from your specified file.
        df = pd.read_csv("xbtusd_1h_8y.csv")
        
        # Convert the DataFrame to the list of dictionaries format the function expects.
        # Note: 'close' becomes 'c' to match the function's internal logic.
        candles = df.rename(columns={'close': 'c'}).to_dict('records')
        
        print("--- Testing Crossover with data from xbtusd_1h_8y.csv ---")
        
        # Using a valid SMA combination for the data size
        short_period = 20
        long_period = 50
        
        if len(candles) >= long_period:
            crossover_detected = check_sma_crossover(candles[-long_period:], short_period=short_period, long_period=long_period)
            
            if crossover_detected:
                print(f"Crossover detected for SMA{short_period} and SMA{long_period}! This is a great preselection signal.")
            else:
                print(f"No crossover detected. The preselection filter suggests not to generate a signal right now.")
            
            # Show the SMAs to visualize the crossover
            close_prices = pd.Series([c['c'] for c in candles])
            sma_short_values = close_prices.rolling(window=short_period).mean().iloc[-5:]
            sma_long_values = close_prices.rolling(window=long_period).mean().iloc[-5:]
            
            print(f"\nVisualizing the last 5 SMA{short_period} and SMA{long_period} values:")
            print(f"SMA{short_period}:", [f"{v:.2f}" for v in sma_short_values.tolist()])
            print(f"SMA{long_period}:", [f"{v:.2f}" for v in sma_long_values.tolist()])
        else:
            print("Not enough data in the CSV to test the specified SMA periods.")
    
    except FileNotFoundError:
        print("Error: The file 'xbtusd_1h_8y.csv' was not found. Please make sure it's in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")
