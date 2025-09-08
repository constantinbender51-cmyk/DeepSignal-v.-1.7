# deepseek_signal.py
import json
import os
import openai
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Optional TA-Lib speed-up (comment out if you do not have it)
# ------------------------------------------------------------------
try:
    import talib as ta
    _HAS_TA = True
except ModuleNotFoundError:
    _HAS_TA = False
# ------------------------------------------------------------------

client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _sma(series: np.ndarray, length: int) -> np.ndarray:
    """Pure-numpy SMA."""
    out = np.full_like(series, np.nan)
    cumsum = np.nancumsum(series)
    out[length - 1:] = (cumsum[length - 1:] - np.concatenate(([0], cumsum[:-length]))) / length
    return out

def _ema(series: np.ndarray, length: int) -> np.ndarray:
    """Pure-numpy EMA (seed with SMA)."""
    alpha = 2.0 / (length + 1)
    out = np.full_like(series, np.nan)
    start = length - 1
    out[start] = _sma(series, length)[start]
    for i in range(start + 1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def _rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
    if _HAS_TA:
        return ta.RSI(close, timeperiod=length)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = _sma(gain, length)
    roll_down = _sma(loss, length)
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           fastk: int = 14, slowk: int = 3, slowd: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if _HAS_TA:
        k, d = ta.STOCH(high, low, close, fastk_period=fastk,
                        slowk_period=slowk, slowd_period=slowd)
        return k, d
    lowest_low = pd.Series(low).rolling(fastk).min().to_numpy()
    highest_high = pd.Series(high).rolling(fastk).max().to_numpy()
    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    slow_k = _sma(fast_k, slowk)
    slow_d = _sma(slow_k, slowd)
    return slow_k, slow_d

def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _HAS_TA:
        macd, signal_, hist = ta.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, signal_, hist
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    signal_ = _ema(macd, signal)
    hist = macd - signal_
    return macd, signal_, hist

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    if _HAS_TA:
        return ta.ATR(high, low, close, timeperiod=length)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    return _sma(tr, length)

def _bollinger(close: np.ndarray, length: int = 20, std: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _HAS_TA:
        upper, middle, lower = ta.BBANDS(close, timeperiod=length, nbdevup=std, nbdevdn=std, matype=0)
        return upper, middle, lower
    middle = _sma(close, length)
    stddev = pd.Series(close).rolling(length).std().to_numpy()
    upper = middle + std * stddev
    lower = middle - std * stddev
    return upper, middle, lower

def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    if _HAS_TA:
        return ta.ADX(high, low, close, timeperiod=length)
    # Simplified ADX (enough for 20-bar context)
    tr = _atr(high, low, close, 1)  # true range
    tr = pd.Series(tr).fillna(method='bfill').to_numpy()
    up = high - np.roll(high, 1)
    down = np.roll(low, 1) - low
    plus = np.where((up > down) & (up > 0), up, 0)
    minus = np.where((down > up) & (down > 0), down, 0)
    smooth_plus = _sma(plus, length)
    smooth_minus = _sma(minus, length)
    smooth_tr = _sma(tr, length)
    di_plus = 100 * smooth_plus / (smooth_tr + 1e-12)
    di_minus = 100 * smooth_minus / (smooth_tr + 1e-12)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-12)
    adx = _sma(dx, length)
    return adx

def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    if _HAS_TA:
        return ta.CCI(high, low, close, timeperiod=length)
    tp = (high + low + close) / 3.0
    sma_tp = _sma(tp, length)
    mad = pd.Series(tp).rolling(length).apply(lambda x: np.fabs(x - x.mean()).mean()).to_numpy()
    cci = (tp - sma_tp) / (0.015 * mad + 1e-12)
    return cci

def _roc(close: np.ndarray, length: int = 10) -> np.ndarray:
    if _HAS_TA:
        return ta.ROC(close, timeperiod=length)
    return (close - np.roll(close, length)) / (np.roll(close, length) + 1e-12) * 100.0

def _williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    if _HAS_TA:
        return ta.WILLR(high, low, close, timeperiod=length)
    hh = pd.Series(high).rolling(length).max().to_numpy()
    ll = pd.Series(low).rolling(length).min().to_numpy()
    return -100 * (hh - close) / (hh - ll + 1e-12)

def _chaikin_money_flow(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, length: int = 20) -> np.ndarray:
    if _HAS_TA:
        return ta.ADOSC(high, low, close, volume, fastperiod=length, slowperiod=length)
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-12)
    money_flow_volume = money_flow_multiplier * volume
    cmf = _sma(money_flow_volume, length) / (_sma(volume, length) + 1e-12)
    return cmf

def _ult_osc(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             len1: int = 7, len2: int = 14, len3: int = 28) -> np.ndarray:
    if _HAS_TA:
        return ta.ULTOSC(high, low, close, timeperiod1=len1, timeperiod2=len2, timeperiod3=len3)
    # Simplified version
    bp = close - np.minimum(low, np.roll(close, 1))
    tr_ = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    avg1 = _sma(bp, len1) / _sma(tr_, len1)
    avg2 = _sma(bp, len2) / _sma(tr_, len2)
    avg3 = _sma(bp, len3) / _sma(tr_, len3)
    ult = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return ult

def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    if _HAS_TA:
        return ta.OBV(close, volume)
    obv = np.zeros_like(close)
    obv[0] = 0.0
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv

# ------------------------------------------------------------------
# Master aggregator – compute everything
# ------------------------------------------------------------------
def _build_indicator_payload(ohlc: List[Dict[str, float]]) -> Dict[str, Any]:
    df = pd.DataFrame(ohlc)[-20:]  # ensure 20 bars
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    open_ = df['open'].to_numpy()
    volume = df.get('volume', np.ones_like(close)).astype(float)

    # Price-action helpers
    hl2 = (high + low) / 2
    hlc3 = (high + low + close) / 3

    # ------------------------------------------------------------------
    # 1. Moving averages
    # ------------------------------------------------------------------
    sma_10 = _sma(close, 10)[-1]
    sma_20 = _sma(close, 20)[-1]
    ema_10 = _ema(close, 10)[-1]
    ema_20 = _ema(close, 20)[-1]

    # ------------------------------------------------------------------
    # 2. Bands
    # ------------------------------------------------------------------
    bb_upper, bb_middle, bb_lower = _bollinger(close, 20, 2)
    bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100
    bb_percent = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-12) * 100

    # ------------------------------------------------------------------
    # 3. Momentum
    # ------------------------------------------------------------------
    rsi_14 = _rsi(close, 14)[-1]
    stoch_k, stoch_d = _stoch(high, low, close, 14, 3, 3)
    macd, macd_signal, macd_hist = _macd(close, 12, 26, 9)
    roc_10 = _roc(close, 10)[-1]
    willr_14 = _williams_r(high, low, close, 14)[-1]
    cci_14 = _cci(high, low, close, 14)[-1]

    # ------------------------------------------------------------------
    # 4. Volume
    # ------------------------------------------------------------------
    obv_line = _obv(close, volume)[-1]
    cmf_20 = _chaikin_money_flow(high, low, close, volume, 20)[-1]

    # ------------------------------------------------------------------
    # 5. Volatility
    # ------------------------------------------------------------------
    atr_14 = _atr(high, low, close, 14)[-1]
    stdev_20 = pd.Series(close).rolling(20).std().iloc[-1]

    # ------------------------------------------------------------------
    # 6. Trend strength
    # ------------------------------------------------------------------
    adx_14 = _adx(high, low, close, 14)[-1]

    # ------------------------------------------------------------------
    # 7. Ultimate oscillator (needs 28 – we still compute it)
    # ------------------------------------------------------------------
    ult_osc = _ult_osc(high, low, close, 7, 14, 28)[-1]

    # ------------------------------------------------------------------
    # 8. Simple price-action micro-metrics
    # ------------------------------------------------------------------
    body_ratio = abs(close[-1] - open_[-1]) / (high[-1] - low[-1] + 1e-12)
    upper_wick = high[-1] - max(open_[-1], close[-1])
    lower_wick = min(open_[-1], close[-1]) - low[-1]

    # ------------------------------------------------------------------
    # Build JSON-serialisable dict
    # ------------------------------------------------------------------
    payload = {
        "price": {
            "close": round(float(close[-1]), 6),
            "open": round(float(open_[-1]), 6),
            "high": round(float(high[-1]), 6),
            "low": round(float(low[-1]), 6),
        },
        "moving_averages": {
            "SMA_10": round(float(sma_10), 6),
            "SMA_20": round(float(sma_20), 6),
            "EMA_10": round(float(ema_10), 6),
            "EMA_20": round(float(ema_20), 6),
        },
        "bollinger": {
            "upper_20": round(float(bb_upper[-1]), 6),
            "middle_20": round(float(bb_middle[-1]), 6),
            "lower_20": round(float(bb_lower[-1]), 6),
            "band_width_pct": round(float(bb_width), 4),
            "percent_b": round(float(bb_percent), 4),
        },
        "momentum": {
            "RSI_14": round(float(rsi_14), 4),
            "STOCH_K_14_3_3": round(float(stoch_k[-1]), 4),
            "STOCH_D_14_3_3": round(float(stoch_d[-1]), 4),
            "MACD": round(float(macd[-1]), 6),
            "MACD_signal": round(float(macd_signal[-1]), 6),
            "MACD_hist": round(float(macd_hist[-1]), 6),
            "ROC_10": round(float(roc_10), 4),
            "WILLR_14": round(float(willr_14), 4),
            "CCI_14": round(float(cci_14), 4),
            "ULT_OSC": round(float(ult_osc), 4),
        },
        "volume": {
            "OBV": int(obv_line),
            "CMF_20": round(float(cmf_20), 4),
        },
        "volatility": {
            "ATR_14": round(float(atr_14), 6),
            "STDDEV_20": round(float(stdev_20), 6),
        },
        "trend": {
            "ADX_14": round(float(adx_14), 4),
        },
        "candle_metrics": {
            "body_ratio": round(float(body_ratio), 4),
            "upper_wick": round(float(upper_wick), 6),
            "lower_wick": round(float(lower_wick), 6),
        }
    }
    return payload

# ------------------------------------------------------------------
# Original API – unchanged signature
# ------------------------------------------------------------------
def get_signal(last_50: list) -> Tuple[str, float, float]:
    last_20 = last_50[-20:] if len(last_50) >= 20 else last_50
    indicators = _build_indicator_payload(last_20)

    prompt = (
        "You are a crypto strategist. Generate a signal based on the following OHLC data and technical indicators (last 20 1-h candles):\n"
        f"{json.dumps(indicators, indent=2)}\n"
        'Reply JSON only: {"action":"BUY"|"SELL"|"FLAT","stop":<pct>,"target":<pct>}'
    )

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    if not raw:
        return "FLAT", 0.0, 0.0
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "FLAT", 0.0, 0.0
    return obj["action"], float(obj["stop"]), float(obj["target"])
