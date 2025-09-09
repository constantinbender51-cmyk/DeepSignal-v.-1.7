import pandas as pd
import numpy as np
from typing import List, Dict

def check_sma_crossover(candles: List[Dict]) -> bool:
    """
    Checks for a Simple Moving Average (SMA) proximity condition on the last candle.

    This function defines a crossover not by a crossing of lines, but by their
    proximity. It returns True if the latest SMA20 value is within 5% of the
    latest SMA50 value, indicating a potential consolidation or volatility squeeze.
    
    Args:
        candles (List[Dict]): A list of candle dictionaries, with each dictionary
                              containing a 'c' (close) price. This list is expected
                              to contain at least 50 candles.
    
    Returns:
        bool: True if the SMA20 is within 5% of the SMA50 on the last candle,
              False otherwise.
    """
    # Ensure we have enough data to calculate both SMAs.
    if len(candles) < 50:
        return False

    # Extract the 'c' (close) prices into a pandas Series for easy calculation.
    close_prices = pd.Series([c['c'] for c in candles])

    # Calculate the SMAs for the entire series.
    sma_20 = close_prices.rolling(window=20).mean()
    sma_50 = close_prices.rolling(window=50).mean()

    # Get the latest values for both SMAs. These will be the last values in the series.
    sma_20_last = sma_20.iloc[-1]
    sma_50_last = sma_50.iloc[-1]

    # Check the condition: is the SMA20 value within 5% of the SMA50 value?
    # This is done by checking if the absolute difference is less than or equal to
    # 5% of the SMA50 value.
    if abs(sma_20_last - sma_50_last) / sma_50_last <= 0.05:
        return True
    
    return False

if __name__ == "__main__":
    try:
        # Define the preselection periods for clarity, although they are hardcoded in the function.
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
        
        print(f"--- Scanning 'xbtusd_1h_8y.csv' for SMA{short_period} within 5% of SMA{long_period} ---")
        
        # Iterate through the candles starting from a point where a 50-candle slice can be created.
        for i in range(long_period - 1, len(candles)):
            # Get the slice of candles needed for the SMA calculation.
            preselection_candles = candles[i - (long_period - 1) : i + 1]
            
            # Check if the proximity criteria is met for the current slice.
            if check_sma_crossover(preselection_candles):
                matching_candles_count += 1
                
        print(f"\nScan complete. Found {matching_candles_count} candles that matched the SMA proximity criteria.")
    
    except FileNotFoundError:
        print("Error: The file 'xbtusd_1h_8y.csv' was not found. Please make sure it's in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")
