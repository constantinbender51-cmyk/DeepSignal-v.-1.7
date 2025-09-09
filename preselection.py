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
        return False

    # Extract the 'c' (close) prices into a pandas Series for easy calculation.
    close_prices = pd.Series([c['c'] for c in candles])

    # Calculate the SMAs for the entire series.
    sma_short = close_prices.rolling(window=short_period).mean()
    sma_long = close_prices.rolling(window=long_period).mean()

    # Get the current and previous values for both SMAs.
    # We use .iloc[-1] for the current value and .iloc[-2] for the previous value.
    # The .iloc[-2] will be NaN if the Series has only one element, so we check for that.
    if len(sma_short) < 2 or len(sma_long) < 2:
        return False
        
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
    try:
        # Define the SMA periods you want to test
        short_period = 20
        long_period = 50
        
        # Read the data directly from your specified file.
        df = pd.read_csv("xbtusd_1h_8y.csv")
        
        # Rename the 'close' column to 'c' to match the function's expectations.
        df.rename(columns={'close': 'c'}, inplace=True)
        
        # Convert the DataFrame to a list of dictionaries.
        candles = df.to_dict('records')
        
        # Counter for candles that meet the preselection criteria
        matching_candles_count = 0
        
        print(f"--- Scanning 'xbtusd_1h_8y.csv' for SMA{short_period} / SMA{long_period} crossovers ---")
        
        # Iterate through the candles starting from a point where SMAs can be calculated.
        # We need at least 'long_period' candles to calculate both SMAs.
        for i in range(long_period, len(candles)):
            # Get the slice of candles needed for the SMA calculation.
            preselection_candles = candles[i - long_period : i]
            
            # Check if the crossover criteria is met for the current candle.
            if check_sma_crossover(preselection_candles, short_period=short_period, long_period=long_period):
                matching_candles_count += 1
                
        print(f"\nScan complete. Found {matching_candles_count} candles that matched the SMA crossover criteria.")
    
    except FileNotFoundError:
        print("Error: The file 'xbtusd_1h_8y.csv' was not found. Please make sure it's in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")
