# deepseek_signal.py
import json
import openai
from typing import Tuple, List, Dict
import os 
import numpy as np
import pandas as pd
from scipy import stats

client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

def calculate_technical_indicators(candles: List[Dict]) -> Dict:
    """Calculate comprehensive technical indicators from OHLCV data"""
    if len(candles) < 20:
        return {}
    
    df = pd.DataFrame(candles)
    closes = df['c'].values
    opens = df['o'].values
    highs = df['h'].values
    lows = df['l'].values
    volumes = df['v'].values
    
    indicators = {}
    
    # Current price and basic stats
    current_close = closes[-1]
    indicators['price'] = current_close
    indicators['price_change_1h'] = ((closes[-1] / closes[-2]) - 1) * 100 if len(closes) > 1 else 0
    indicators['price_change_20h'] = ((closes[-1] / closes[0]) - 1) * 100
    
    # 1. Trend Indicators
    indicators['sma_20'] = np.mean(closes)
    indicators['ema_20'] = pd.Series(closes).ewm(span=20).mean().iloc[-1]
    indicators['sma_10'] = np.mean(closes[-10:])
    indicators['ema_10'] = pd.Series(closes).ewm(span=10).mean().iloc[-1]
    
    # MACD (12,26,9) - using shorter periods for 20 candles
    ema_12 = pd.Series(closes).ewm(span=12).mean()
    ema_26 = pd.Series(closes).ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    indicators['macd_line'] = macd_line.iloc[-1]
    indicators['macd_signal'] = signal_line.iloc[-1]
    indicators['macd_histogram'] = indicators['macd_line'] - indicators['macd_signal']
    
    # 2. Momentum Indicators
    # RSI (14)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    
    if avg_loss == 0:
        indicators['rsi_14'] = 100
    else:
        rs = avg_gain / avg_loss
        indicators['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Stochastic (14,3)
    low_14 = np.min(lows[-14:])
    high_14 = np.max(highs[-14:])
    if high_14 - low_14 == 0:
        indicators['stoch_k'] = 50
    else:
        indicators['stoch_k'] = 100 * (closes[-1] - low_14) / (high_14 - low_14)
    
    # Simple stoch smoothing
    indicators['stoch_d'] = np.mean([indicators['stoch_k'], 
                                   (100 * (closes[-2] - np.min(lows[-15:-1])) / (np.max(highs[-15:-1]) - np.min(lows[-15:-1])) if len(closes) > 14 else 50)])
    
    # Rate of Change
    indicators['roc_14'] = ((closes[-1] / closes[-14]) - 1) * 100 if len(closes) > 14 else 0
    indicators['momentum_10'] = closes[-1] - closes[-10] if len(closes) > 10 else 0
    
    # 3. Volatility Indicators
    # Bollinger Bands
    sma_20 = indicators['sma_20']
    std_dev = np.std(closes)
    indicators['bb_upper'] = sma_20 + (std_dev * 2)
    indicators['bb_lower'] = sma_20 - (std_dev * 2)
    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20 * 100
    indicators['bb_position'] = (current_close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) * 100
    
    # ATR (14)
    true_ranges = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    indicators['atr_14'] = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)
    indicators['atr_pct'] = (indicators['atr_14'] / current_close) * 100
    
    # 4. Volume Indicators
    indicators['volume_avg_20'] = np.mean(volumes)
    indicators['volume_ratio'] = volumes[-1] / indicators['volume_avg_20'] if indicators['volume_avg_20'] > 0 else 1
    
    # OBV
    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
    indicators['obv'] = obv
    indicators['obv_change'] = ((obv - (obv - volumes[-1])) / abs(obv - volumes[-1])) * 100 if obv != volumes[-1] else 0
    
    # 5. Support/Resistance
    indicators['high_20'] = np.max(highs)
    indicators['low_20'] = np.min(lows)
    indicators['range_20'] = indicators['high_20'] - indicators['low_20']
    indicators['range_pct'] = (indicators['range_20'] / indicators['low_20']) * 100
    
    # Fibonacci levels
    indicators['fib_236'] = indicators['high_20'] - 0.236 * indicators['range_20']
    indicators['fib_382'] = indicators['high_20'] - 0.382 * indicators['range_20']
    indicators['fib_500'] = indicators['high_20'] - 0.5 * indicators['range_20']
    indicators['fib_618'] = indicators['high_20'] - 0.618 * indicators['range_20']
    
    # 6. Pattern Recognition (simplified)
    # Recent price action
    indicators['three_white_soldiers'] = int(
        len(closes) >= 3 and
        closes[-3] < closes[-2] < closes[-1] and
        opens[-3] < closes[-3] and opens[-2] < closes[-2] and opens[-1] < closes[-1]
    )
    
    indicators['three_black_crows'] = int(
        len(closes) >= 3 and
        closes[-3] > closes[-2] > closes[-1] and
        opens[-3] > closes[-3] and opens[-2] > closes[-2] and opens[-1] > closes[-1]
    )
    
    return indicators

def get_signal(last_50: list) -> Tuple[str, float, float]:
    # Use the most recent 20 candles for indicator calculation
    last_20 = last_50[-20:] if len(last_50) >= 20 else last_50
    
    # Calculate all technical indicators
    indicators = calculate_technical_indicators(last_20)
    
    if not indicators:
        return "FLAT", 0.0, 0.0
    
    # Create a comprehensive prompt with technical analysis
    prompt = f"""You are an expert crypto trading strategist. Analyze the following technical indicators derived from the last 20 hourly OHLCV candles:

CURRENT PRICE: ${indicators['price']:.2f}
Price Change (1h): {indicators['price_change_1h']:+.2f}%
Price Change (20h): {indicators['price_change_20h']:+.2f}%

TREND INDICATORS:
- SMA(20): ${indicators['sma_20']:.2f} ({'ABOVE' if indicators['price'] > indicators['sma_20'] else 'BELOW'} price)
- EMA(20): ${indicators['ema_20']:.2f}
- MACD Line: {indicators['macd_line']:.4f}
- MACD Signal: {indicators['macd_signal']:.4f}
- MACD Histogram: {indicators['macd_histogram']:+.4f} {'(BULLISH)' if indicators['macd_histogram'] > 0 else '(BEARISH)'}

MOMENTUM INDICATORS:
- RSI(14): {indicators['rsi_14']:.1f} {'(OVERSOLD <30)' if indicators['rsi_14'] < 30 else '(OVERBOUGHT >70)' if indicators['rsi_14'] > 70 else '(NEUTRAL)'}
- Stochastic %K: {indicators['stoch_k']:.1f} {'(OVERSOLD <20)' if indicators['stoch_k'] < 20 else '(OVERBOUGHT >80)' if indicators['stoch_k'] > 80 else '(NEUTRAL)'}
- ROC(14): {indicators['roc_14']:+.2f}%

VOLATILITY INDICATORS:
- Bollinger Upper: ${indicators['bb_upper']:.2f}
- Bollinger Lower: ${indicators['bb_lower']:.2f}
- BB Position: {indicators['bb_position']:.1f}% {'(NEAR UPPER)' if indicators['bb_position'] > 80 else '(NEAR LOWER)' if indicators['bb_position'] < 20 else '(MIDDLE)'}
- ATR: ${indicators['atr_14']:.2f} ({indicators['atr_pct']:.2f}% of price)

VOLUME:
- Current Volume: {indicators['volume_ratio']:.2f}x 20-period average
- OBV Trend: {'BULLISH' if indicators['obv_change'] > 0 else 'BEARISH'}

SUPPORT/RESISTANCE:
- 20-period High: ${indicators['high_20']:.2f}
- 20-period Low: ${indicators['low_20']:.2f}
- Fibonacci 61.8%: ${indicators['fib_618']:.2f}
- Fibonacci 38.2%: ${indicators['fib_382']:.2f}

PATTERNS:
- Three White Soldiers: {'YES' if indicators['three_white_soldiers'] else 'NO'}
- Three Black Crows: {'YES' if indicators['three_black_crows'] else 'NO'}

Based on this comprehensive technical analysis, generate a trading signal.
Consider trend direction, momentum, overbought/oversold conditions, volatility, and volume.

Reply with JSON only: {{"action":"BUY"|"SELL"|"FLAT","stop":<pct>,"target":<pct>}}
Set appropriate stop-loss (1-5%) and take-profit (2-10%) levels based on volatility (ATR) and current market conditions.
"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Slight temperature for minor creativity
            response_format={'type': 'json_object'}
        )
        
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        
        if not raw:
            return "FLAT", 0.0, 0.0
            
        obj = json.loads(raw)
        action = obj["action"].upper()
        stop = max(0.5, min(10.0, float(obj["stop"])))  # Clamp between 0.5% and 10%
        target = max(1.0, min(20.0, float(obj["target"])))  # Clamp between 1% and 20%
        
        return action, stop, target
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing AI response: {e}")
        return "FLAT", 0.0, 0.0
    except Exception as e:
        print(f"API error: {e}")
        return "FLAT", 0.0, 0.0
