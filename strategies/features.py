"""Feature engineering for ML trading strategy.

Computes technical and cross-asset features from OHLCV data.
"""

import numpy as np
import pandas as pd

def _drop_stale_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove forward-filled calendar days (weekends/holidays on a 24/7 grid).

    A forward-filled hour is a verbatim copy of the previous bar (identical
    OHLC *and* volume). A day where every hour is such a copy is a non-trading
    day and is dropped. Real trading days always contain fresh bars — even when
    the closing price repeats exactly — so genuine flat closes are kept.
    On a weekday-only grid (set_market('24/7') commented out) nothing matches,
    so this is a no-op.
    """
    cols = ["open", "high", "low", "close", "volume"]
    stale_hour = (df[cols] == df[cols].shift()).all(axis=1)
    stale_day = stale_hour.groupby(df.index.normalize()).transform("all")
    return df[~stale_day]

def compute_features(df: pd.DataFrame) -> pd.Series:
    """Compute ~30 features from the latest bar of an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: open, high, low, close, volume.

    Returns
    -------
    pd.Series
        Series of feature values derived from the latest bar.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    latest_close = close.iloc[-1]

    features = {}

    # -------------------------------------------------------------------------
    # Price returns (4): 1h, 4h, 12h, 24h pct_change
    # -------------------------------------------------------------------------
    for period, label in [(1, "ret_1h"), (12, "ret_12h")]: # COMMENTED OUT (4, "ret_4h"), (24, "ret_24h")
        if len(close) > period:
            prev = close.iloc[-1 - period]
            features[label] = (latest_close - prev) / prev if prev != 0 else 0.0
        else:
            features[label] = 0.0

    # -------------------------------------------------------------------------
    # MA distances (4): SMA10, SMA20, EMA12, EMA26 as (price - MA) / price
    # -------------------------------------------------------------------------
    sma10 = close.rolling(10).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    # ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
    # ema26 = close.ewm(span=26, adjust=False).mean().iloc[-1]

    features["dist_sma10"] = (latest_close - sma10) / latest_close if latest_close != 0 else 0.0
    # for ma_val, label in [
    #     (sma10, "dist_sma10"),
    #     (sma20, "dist_sma20"),
    #     (ema12, "dist_ema12"),
    #     (ema26, "dist_ema26"),
    # ]:
    #     features[label] = (latest_close - ma_val) / latest_close if latest_close != 0 else 0.0

    # -------------------------------------------------------------------------
    # Bollinger Bands (4): upper dist, lower dist, bandwidth, %B
    # Using SMA20 +/- 2*std20
    # -------------------------------------------------------------------------
    std20 = close.rolling(20).std().iloc[-1]
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_bandwidth = (bb_upper - bb_lower) / sma20 if sma20 != 0 else 0.0
    bb_range = bb_upper - bb_lower
    bb_pct_b = (latest_close - bb_lower) / bb_range if bb_range != 0 else 0.5

    features["bb_upper_dist"] = (bb_upper - latest_close) / latest_close if latest_close != 0 else 0.0
    features["bb_lower_dist"] = (latest_close - bb_lower) / latest_close if latest_close != 0 else 0.0
    features["bb_bandwidth"] = bb_bandwidth
    features["bb_pct_b"] = bb_pct_b

    # -------------------------------------------------------------------------
    # RSI (2): RSI(14), RSI(7) — manual computation using rolling gain/loss
    # -------------------------------------------------------------------------
    def _rsi(series: pd.Series, period: int) -> float:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    features["rsi_14"] = _rsi(close, 14)
    features["rsi_7"] = _rsi(close, 7)

    # -------------------------------------------------------------------------
    # MACD (3): MACD line (EMA12-EMA26), signal (EMA9 of MACD), histogram
    # -------------------------------------------------------------------------
    ema12_series = close.ewm(span=12, adjust=False).mean()
    ema26_series = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12_series - ema26_series
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    # macd_hist = macd_line - macd_signal

    features["macd_line"] = macd_line.iloc[-1]
    features["macd_signal"] = macd_signal.iloc[-1]
    # features["macd_histogram"] = macd_hist.iloc[-1]

    # -------------------------------------------------------------------------
    # ROC (1): rate of change over 12 periods
    # -------------------------------------------------------------------------
    if len(close) > 12:
        prev_12 = close.iloc[-1 - 12]
        features["roc_12"] = (latest_close - prev_12) / prev_12 if prev_12 != 0 else 0.0
    else:
        features["roc_12"] = 0.0

    # -------------------------------------------------------------------------
    # ATR (1): ATR(14) normalized by price
    # -------------------------------------------------------------------------
    # tr = pd.concat(
    #     [
    #         high - low,
    #         (high - close.shift(1)).abs(),
    #         (low - close.shift(1)).abs(),
    #     ],
    #     axis=1,
    # ).max(axis=1)
    # atr_14 = tr.rolling(14).mean().iloc[-1]
    # features["atr_14_norm"] = atr_14 / latest_close if latest_close != 0 else 0.0

    # -------------------------------------------------------------------------
    # Volatility (2): std of returns over 12h and 24h windows
    # -------------------------------------------------------------------------
    returns = close.pct_change()
    features["volatility_12h"] = returns.rolling(12).std().iloc[-1]
    # features["volatility_24h"] = returns.rolling(24).std().iloc[-1]

    # -------------------------------------------------------------------------
    # Range (1): (high - low) / close for latest bar
    # -------------------------------------------------------------------------
    features["range"] = (high.iloc[-1] - low.iloc[-1]) / latest_close if latest_close != 0 else 0.0

    # -------------------------------------------------------------------------
    # Volume ratio (1): current volume / 20-period avg volume
    # -------------------------------------------------------------------------
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    features["volume_ratio"] = volume.iloc[-1] / avg_vol_20 if avg_vol_20 != 0 else 0.0

    # -------------------------------------------------------------------------
    # VWAP deviation (1): (price - VWAP) / price using 20-period rolling VWAP
    # -------------------------------------------------------------------------
    # typical_price = (high + low + close) / 3.0
    # cum_tp_vol = (typical_price * volume).rolling(20).sum().iloc[-1]
    # cum_vol = volume.rolling(20).sum().iloc[-1]
    # vwap = cum_tp_vol / cum_vol if cum_vol != 0 else latest_close
    # features["vwap_deviation"] = (latest_close - vwap) / latest_close if latest_close != 0 else 0.0

    # -------------------------------------------------------------------------
    # OBV slope (1): slope of OBV over last 10 bars, normalized by avg volume
    # -------------------------------------------------------------------------
    obv_direction = np.sign(close.diff()).fillna(0)
    obv = (obv_direction * volume).cumsum()
    if len(obv) >= 10:
        obv_tail = obv.iloc[-10:].values
        x = np.arange(10, dtype=float)
        x_mean = x.mean()
        obv_mean = obv_tail.mean()
        denom = ((x - x_mean) ** 2).sum()
        slope = ((x - x_mean) * (obv_tail - obv_mean)).sum() / denom if denom != 0 else 0.0
        avg_vol_all = volume.mean()
        features["obv_slope"] = slope / avg_vol_all if avg_vol_all != 0 else 0.0
    else:
        features["obv_slope"] = 0.0

    # -------------------------------------------------------------------------
    # ADX (1): Average Directional Index, measures trend strength (0-100)
    # -------------------------------------------------------------------------
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / 14
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * smoothed_plus / smoothed_tr
    minus_di = 100 * smoothed_minus / smoothed_tr
    di_sum = plus_di + minus_di
    dx = ((plus_di - minus_di).abs() / di_sum.replace(0, 1)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    features["adx_14"] = adx.iloc[-1]

    # -------------------------------------------------------------------------
    # DEMA9-SMA50 distance (1): trend crossover signal (continuous)
    # -------------------------------------------------------------------------
    sma50 = close.rolling(50).mean().iloc[-1]
    ema9 = close.ewm(span=9, adjust=False).mean()
    dema9 = (2 * ema9 - ema9.ewm(span=9, adjust=False).mean()).iloc[-1]
    features["dema9_sma50_dist"] = (dema9 - sma50) / latest_close if latest_close != 0 else 0.0

    # -------------------------------------------------------------------------
    # TEMA7-SMA50 distance (1): fast trend crossover signal (continuous)
    # -------------------------------------------------------------------------
    ema7 = close.ewm(span=7, adjust=False).mean()
    ema7_ema = ema7.ewm(span=7, adjust=False).mean()
    tema7 = (3 * ema7 - 3 * ema7_ema + ema7_ema.ewm(span=7, adjust=False).mean()).iloc[-1]
    features["tema7_sma50_dist"] = (tema7 - sma50) / latest_close if latest_close != 0 else 0.0

    # -------------------------------------------------------------------------
    # Volume spike (1): 1.0 if volume > 2x 20-period avg, else 0.0
    # -------------------------------------------------------------------------
    # features["volume_spike"] = 1.0 if (avg_vol_20 > 0 and volume.iloc[-1] > 2 * avg_vol_20) else 0.0

    # -------------------------------------------------------------------------
    # Regime features
    # Daily-regime context features
    # -------------------------------------------------------------------------
    # daily_close = close.resample("D").last().dropna()
    daily_close = _drop_stale_days(df)["close"].resample("D").last().dropna()
    if len(daily_close) >= 20:
        dsma20 = daily_close.tail(20).mean()
        features["dist_dsma20"] = (latest_close - dsma20) / dsma20
    else:
        features["dist_dsma20"] = 0.0
    if len(daily_close) >= 21:
        features["vol_20d"] = daily_close.pct_change().tail(20).std()
    else:
        features["vol_20d"] = 0.0

    # -------------------------------------------------------------------------
    # Build Series, handle NaN by filling with 0
    # -------------------------------------------------------------------------
    result = pd.Series(features, dtype=float).fillna(0.0)
    return result

def compute_cross_asset_features(
    asset_returns: dict,
    asset_symbol: str,
    all_symbols: list,
) -> dict:
    """Compute cross-asset features for a given symbol.

    Parameters
    ----------
    asset_returns : dict
        Mapping of symbol -> pd.Series of returns.
    asset_symbol : str
        The symbol to compute features for.
    all_symbols : list
        List of all symbols in the universe.

    Returns
    -------
    dict
        Dictionary with keys: corr_btc, corr_spy, momentum_rank, volatility_rank.
    """
    features = {
        # "corr_btc": 0.0,
        # "corr_spy": 0.0,
        "momentum_rank": 0.5,
        # "volatility_rank": 0.5,
    }

    my_returns = asset_returns.get(asset_symbol)
    if my_returns is None or len(my_returns) < 24:
        return features

    my_tail = my_returns.iloc[-24:]

    # -------------------------------------------------------------------------
    # corr_btc: correlation with BTC returns over last 24 bars
    # -------------------------------------------------------------------------
    # btc_returns = asset_returns.get("BTC")
    # if btc_returns is not None and len(btc_returns) >= 24:
    #     corr = my_tail.corr(btc_returns.iloc[-24:])
    #     features["corr_btc"] = corr if not np.isnan(corr) else 0.0

    # -------------------------------------------------------------------------
    # corr_spy: correlation with AAPL returns over last 24 bars
    # -------------------------------------------------------------------------
    # aapl_returns = asset_returns.get("AAPL")
    # if aapl_returns is not None and len(aapl_returns) >= 24:
    #     corr = my_tail.corr(aapl_returns.iloc[-24:])
    #     features["corr_spy"] = corr if not np.isnan(corr) else 0.0

    # -------------------------------------------------------------------------
    # momentum_rank: rank of 12-bar momentum among all symbols (0 to 1)
    # -------------------------------------------------------------------------
    momentums = {}
    for sym in all_symbols:
        sym_ret = asset_returns.get(sym)
        if sym_ret is not None and len(sym_ret) >= 12:
            momentums[sym] = sym_ret.iloc[-12:].sum()

    if asset_symbol in momentums and len(momentums) > 1:
        sorted_vals = sorted(momentums.values())
        rank = sorted_vals.index(momentums[asset_symbol])
        features["momentum_rank"] = rank / (len(sorted_vals) - 1)
    elif asset_symbol in momentums:
        features["momentum_rank"] = 0.5

    # -------------------------------------------------------------------------
    # volatility_rank: rank of 24-bar volatility among all symbols (0 to 1)
    # -------------------------------------------------------------------------
    # volatilities = {}
    # for sym in all_symbols:
    #     sym_ret = asset_returns.get(sym)
    #     if sym_ret is not None and len(sym_ret) >= 24:
    #         volatilities[sym] = sym_ret.iloc[-24:].std()

    # if asset_symbol in volatilities and len(volatilities) > 1:
    #     sorted_vals = sorted(volatilities.values())
    #     rank = sorted_vals.index(volatilities[asset_symbol])
    #     features["volatility_rank"] = rank / (len(sorted_vals) - 1)
    # elif asset_symbol in volatilities:
    #     features["volatility_rank"] = 0.5

    return features

def compute_atr_pct(df: pd.DataFrame, period: int = 10) -> float:
    """ATR as a fraction of current price, computed on daily-resampled bars.

    Resamples hourly OHLCV to daily bars before computing ATR so the threshold
    reflects typical day-level moves, not hour-level noise.
    Returns 0.0 if data is insufficient (caller should use P.MAX_POSITION_LOSS as fallback).
    """
    try:
        # daily_high = df["high"].resample("D").max().dropna()
        # daily_low = df["low"].resample("D").min().dropna()
        # daily_close = df["close"].resample("D").last().dropna()
        clean = _drop_stale_days(df)
        daily_high = clean["high"].resample("D").max().dropna()
        daily_low = clean["low"].resample("D").min().dropna()
        daily_close = clean["close"].resample("D").last().dropna()
        if len(daily_close) < period + 1:
            print(f"[WARN] ATR inactive: only {len(daily_close)} daily closes, need {period + 1}")
            return 0.0
        prev_close = daily_close.shift(1)
        tr = pd.concat([
            daily_high - daily_low,
            (daily_high - prev_close).abs(),
            (daily_low - prev_close).abs(),
        ], axis=1).max(axis=1).dropna()
        atr = tr.tail(period).mean()
        latest_close = df["close"].iloc[-1]
        if latest_close <= 0 or atr <= 0:
            return 0.0
        return float(atr / latest_close)
    except Exception:
        return 0.0
    
def compute_regime_ok(df: pd.DataFrame, sma_days: int = 20, persist_days: int = 1) -> bool:
    """True if the latest daily close is above its daily SMA (uptrend regime).

    Resamples hourly bars to daily closes. Returns True (no filtering) when
    there is not enough history to compute the SMA.
    """
    try:
        # daily_close = df["close"].resample("D").last().dropna()
        daily_close = _drop_stale_days(df)["close"].resample("D").last().dropna()
        if len(daily_close) < sma_days:
            print(f"[WARN] regime filter inactive: only {len(daily_close)} daily closes, need {sma_days}")
            return True
        sma = daily_close.tail(sma_days).mean()
        recent = daily_close.tail(persist_days)
        return bool((recent > sma).all())
    except Exception:
        return True
    
def compute_regime_down(df: pd.DataFrame, sma_days: int = 20, persist_days: int = 2) -> bool:
    """True only if the last `persist_days` daily closes are ALL below the daily SMA
    (confirmed breakdown). Used to arm the tightened regime-flip stop."""
    try:
        daily_close = _drop_stale_days(df)["close"].resample("D").last().dropna()
        if len(daily_close) < sma_days:
            return False
        sma = daily_close.tail(sma_days).mean()
        recent = daily_close.tail(persist_days)
        return bool((recent < sma).all())
    except Exception:
        return False