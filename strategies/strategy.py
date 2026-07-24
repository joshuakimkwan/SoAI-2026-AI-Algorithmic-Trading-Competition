"""
SoAI 2026 AI Algorithmic Trading Competition - ML Strategy.

XGBoost signal generation + exponential decay adaptive position sizing.
Scores 15 equity/crypto candidates hourly, holds max 5 positions.

The class name MUST remain ``Strategy`` and this file MUST stay at
``strategies/strategy.py`` for the competition execution environment.
"""

import pandas as pd
import numpy as np

import os
import json

from lumibot.strategies import Strategy as _LumibotStrategy
from lumibot.entities import Asset

from strategies import params as P
from strategies.features import compute_features, compute_cross_asset_features, compute_atr_pct, compute_regime_ok, compute_regime_down
from strategies.model import TradingModel
from strategies.model_mean_reversion import MeanReversionModel
from strategies.risk import DecayTracker, compute_position_sizes


class Strategy(_LumibotStrategy):
    """ML trading strategy with adaptive position sizing."""

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = P.SLEEPTIME

        # For trading crypto

        self.set_market('24/7')
        
        # Build asset objects
        self.equity_assets = {
            s: Asset(s, asset_type=Asset.AssetType.STOCK)
            for s in P.STOCK_SLEEVE_SYMBOLS
        }
        self.crypto_assets = {
            s: Asset(s, asset_type=Asset.AssetType.CRYPTO)
            for s in P.CRYPTO_SLEEVE_SYMBOLS
        }

        self.usd_quote = Asset(symbol="USD", asset_type=Asset.AssetType.FOREX) # For trading crypto

        self.all_assets = {**self.equity_assets, **self.crypto_assets}
        self.all_symbols = list(self.all_assets.keys())

        # ML model and adaptive tracker
        # self.model = TradingModel()
        self.models = {s: TradingModel(symbol=s) for s in self.all_symbols} # CHANGED (18062026)
        self.tracker = DecayTracker(self.all_symbols)

        self.mr_models = {s: MeanReversionModel(symbol=s) for s in self.all_symbols}
        self.mr_tracker = DecayTracker(self.all_symbols)

        # State tracking
        self.iteration_count = 0
        self.peak_value = None
        self.prev_prices = {}
        self.last_predictions = {}  # {symbol: direction} from prior iteration
        self.paused = False
        self.asset_returns = {s: pd.Series(dtype=float) for s in self.all_symbols}
        self._prev_features = {}  # features from prior iteration for labeling
        self.last_stop_loss_time = {}
        self.last_stop_loss_time = {}
        self._yf_seed = {}  # symbol -> cached Yahoo lookback df (fetched once on cold start)

        self.target_streak = {} # CHANGED (12062026) {symbol: consecutive iterations with positive target weight}
        self.exit_streak = {} # CHANGED (12062026) {symbol: consecutive iterations a held position is out of targets}

        self.entry_prices = {}      # CHANGED (21062026) {symbol: price at which we last bought}
        # self.last_sold_prices = {}  # CHANGED (21062026) {symbol: price at which we last sold}
        self.accumulated_fees = {}
        self.peak_prices = {}       # {symbol: highest price seen since entry} for trailing stop

        self.failed_exit_streak = 0   # consecutive full exits below FAILED_EXIT_RETURN
        self.mr_mode_active = False
        self.mr_positions = set()     # symbols opened by the mean-reversion sleeve
        self.entry_times = {}         # {symbol: datetime of first entry}
        self.last_mr_predictions = {}
        self.mr_idle_iters = 0        # consecutive idle iterations while locked out, flat book

        self.log_message(
            f"ML Strategy initialized | universe={len(self.all_symbols)} assets | "
            f"max_positions={P.MAX_POSITIONS} | sleeptime={P.SLEEPTIME} | "
            f"regime_filter={P.REGIME_FILTER_ENABLED} (sma_days={P.REGIME_SMA_DAYS}) | "
            f"min_tracker_score={P.MIN_TRACKER_SCORE_FOR_ENTRY}"
        )


    # ------------------------------------------------------------------
    # Warmup model
    # ------------------------------------------------------------------
    def _warmup_models(self):
        """Pre-train models from historical bars so trading can start immediately."""
        for symbol in self.all_symbols:
            try:
                if symbol in P.CRYPTO_SYMBOLS:
                    ast = Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
                    crypto_q = self.usd_quote
                    bars = self.get_historical_prices(ast, length=P.WARMUP_LENGTH_HOURS, timestep="hour", quote=crypto_q)
                else:
                    bars = self.get_historical_prices(symbol, length=P.WARMUP_LENGTH_HOURS, timestep="hour")
                if bars is None or not hasattr(bars, "df") or len(bars.df) < P.MIN_HISTORY_BARS + 1:
                    continue
                df = bars.df
                closes = df["close"]
                mdl = self.models[symbol]
                added = 0
                cutoff = df.index[-1] - pd.Timedelta(hours=P.TRACKER_REPLAY_HOURS)  # end of Sample A
                for i in range(P.MIN_HISTORY_BARS, len(df) - 1):
                    if df.index[i] > cutoff:
                        break  # Sample B stays completely unseen by the model
                    cur = closes.iloc[i]
                    nxt = closes.iloc[i + 1]
                    if cur <= 0 or nxt == cur:
                        continue  # skip stale/forward-filled bars, same as the live loop
                    window = df.iloc[: i + 1]
                    feats = compute_features(window)
                    feats["momentum_rank"] = 0.5  # neutral placeholder; live loop supplies real rank
                    mdl.add_sample(feats, TradingModel.compute_target(cur, nxt))

                    # Mean-reversion strategy
                    if P.MR_SWITCH_ENABLED:
                        self.mr_models[symbol].add_sample(
                            feats, MeanReversionModel.compute_target(cur, nxt, float(feats.get("dist_dsma20", 0.0)))
                        )

                    added += 1
                self.log_message(f"WARMUP [{symbol}]: {added} samples from history")
            except Exception as e:
                self.log_message(f"WARMUP failed [{symbol}]: {e}")

    # ------------------------------------------------------------------
    # Warm-up tracker
    # ------------------------------------------------------------------
    def _warmup_tracker(self):
        """Replay recent history through the trained models to prime the DecayTracker."""
        crypto_q = self.usd_quote
        for symbol in self.all_symbols:
            mdl = self.models.get(symbol)
            if mdl is None or mdl.model is None:
                continue
            try:
                if symbol in P.CRYPTO_SYMBOLS:
                    ast = Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
                    bars = self.get_historical_prices(ast, length=P.WARMUP_LENGTH_HOURS, timestep="hour", quote=crypto_q)
                else:
                    bars = self.get_historical_prices(symbol, length=P.WARMUP_LENGTH_HOURS, timestep="hour")
                if bars is None or not hasattr(bars, "df"):
                    continue
                df = bars.df
                closes = df["close"]
                # collect the most recent usable bar indices, then replay chronologically
                cutoff = df.index[-1] - pd.Timedelta(hours=P.TRACKER_REPLAY_HOURS)
                idxs = []
                i = len(df) - 2
                while i >= P.MIN_HISTORY_BARS and len(idxs) < P.TRACKER_WARMUP_BARS:
                    if df.index[i] > cutoff and closes.iloc[i] > 0 and closes.iloc[i + 1] != closes.iloc[i]:
                        idxs.append(i)
                    i -= 1
                updated = 0
                for i in reversed(idxs):   # oldest -> newest so decay weighting is correct
                    feats = compute_features(df.iloc[: i + 1])
                    feats["momentum_rank"] = 0.5
                    direction, confidence = mdl.predict(feats)
                    ret = (closes.iloc[i + 1] - closes.iloc[i]) / closes.iloc[i]
                    if ret > P.RETURN_DEAD_ZONE:
                        actual = 1
                    elif ret < -P.RETURN_DEAD_ZONE:
                        actual = -1
                    else:
                        actual = 0
                    self.tracker.update(symbol, direction == actual)
                    mdl.add_sample(feats, actual)  # B joins training data AFTER being used as unseen test
                    if P.MR_SWITCH_ENABLED:
                        mr_dir, _ = self.mr_models[symbol].predict(feats)
                        self.mr_tracker.update(symbol, mr_dir == actual)
                        self.mr_models[symbol].add_sample(
                            feats, MeanReversionModel.compute_target(closes.iloc[i], closes.iloc[i + 1], float(feats.get("dist_dsma20", 0.0)))
                        )
                    updated += 1
                if updated > 0:
                    mdl.train()

                    # Mean-reversion model
                    if P.MR_SWITCH_ENABLED:
                        self.mr_models[symbol].train()

                self.log_message(f"TRACKER-WARMUP [{symbol}]: {updated} updates, score={self.tracker.get_score(symbol):.3f}")
            except Exception as e:
                self.log_message(f"TRACKER-WARMUP failed [{symbol}]: {e}")

    # ------------------------------------------------------------------
    # Competition startup: load pre-trained weights (no runtime training)
    # ------------------------------------------------------------------
    def _load_pretrained(self):
        """Load pre-trained models + primed tracker from WEIGHTS_DIR (built by
        pretrain.py). Per-symbol try/except: one bad file degrades that symbol
        only, never the whole strategy — the submission cannot be edited after
        the deadline, so we degrade gracefully and log loudly instead of crashing."""
        weights_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), P.WEIGHTS_DIR
        )
        loaded, failed = [], []
        for symbol in self.all_symbols:
            try:
                self.models[symbol].load(weights_dir)
                if self.models[symbol].model is not None:
                    loaded.append(symbol)
                else:
                    failed.append(symbol)
            except Exception as e:
                failed.append(symbol)
                self.log_message(f"[LOAD-FAIL] {symbol}: {e}")
        try:
            with open(os.path.join(weights_dir, "tracker.json"), encoding="utf-8") as fh:
                state = json.load(fh)
            for s, st in state.items():
                self.tracker.predictions[s] = [bool(x) for x in st.get("predictions", [])]
                self.tracker.accuracy_scores[s] = float(st.get("score", P.INITIAL_ACCURACY))
            self.log_message(f"[LOAD] tracker primed for {len(state)} symbols")
        except Exception as e:
            self.log_message(f"[LOAD-FAIL] tracker: {e} (starting at neutral {P.INITIAL_ACCURACY})")
        if failed:
            self.log_message(f"[LOAD] CRITICAL: {len(failed)} model(s) missing: {failed} — "
                             f"those symbols trade only after enough live samples accumulate")
        self.log_message(f"[LOAD] pre-trained models ready: {len(loaded)}/{len(self.all_symbols)} {loaded}")

    def _yf_lookback(self, symbol, is_crypto):
        """Fetch ~YF_BACKFILL_DAYS of trailing hourly bars from Yahoo, bounded by
        the strategy's current time (no lookahead). Returns an OHLCV DataFrame
        (UTC index, lowercase columns) or an empty frame on any failure — never
        raises. Cold-start backfill only; not used when the feed serves history."""
        cols = ["open", "high", "low", "close", "volume"]
        empty = pd.DataFrame(columns=cols)
        try:
            import yfinance as yf
            from datetime import timedelta

            ticker = P.YF_TICKER_MAP.get(symbol, symbol)  # stocks fall back to themselves
            end_ts = pd.Timestamp(self.get_datetime())
            end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
            start_ts = end_ts - timedelta(days=int(getattr(P, "YF_BACKFILL_DAYS", 90)))

            raw = yf.download(
                ticker,
                start=start_ts.tz_localize(None),
                end=end_ts.tz_localize(None),
                interval="1h",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or len(raw) == 0:
                return empty
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.lower)
            raw = raw[[c for c in cols if c in raw.columns]].copy()
            if raw.empty:
                return empty
            raw.index = pd.to_datetime(raw.index, utc=True)
            raw = raw[raw.index < end_ts]  # guard against lookahead
            for c in cols:
                if c not in raw.columns:
                    raw[c] = float("nan")
            out = raw[cols].dropna(how="all")
            self.log_message(f"[YF] backfilled {len(out)} bars for {symbol} ({ticker})")
            return out
        except Exception as e:
            try:
                self.log_message(f"[YF] backfill failed for {symbol}: {e}")
            except Exception:
                pass
            return empty

    def _history_for(self, symbol, is_crypto, feed_bars):
        """Return an hourly OHLCV DataFrame for `symbol`. When the live feed
        serves at least MIN_HISTORY_BARS, it is used as-is (Yahoo is never
        touched). On a cold feed, a one-time Yahoo lookback (cached in
        self._yf_seed) is merged UNDER the live bars, deduped by timestamp with
        the live bars taking precedence."""
        cols = ["open", "high", "low", "close", "volume"]
        feed_df = feed_bars.df if (feed_bars is not None and hasattr(feed_bars, "df")) else None
        have = 0 if feed_df is None else len(feed_df)

        # # Feed already served enough history, or backfill disabled: use feed as-is.
        # if have >= P.MIN_HISTORY_BARS or not getattr(P, "YF_BACKFILL_ENABLED", False):
        #     return feed_df
        
        # Never backfill on iteration 1: startup stays light (no heavy fetch or
        # train on hour 1, per the fairness rule). Cold-start warming begins hour 2.
        if (have >= P.MIN_HISTORY_BARS or not getattr(P, "YF_BACKFILL_ENABLED", False)
                or self.iteration_count <= 1):
            return feed_df
        
        # Cold feed: fetch the Yahoo seed once per symbol (cache the result — even
        # if empty — so we never re-hammer Yahoo), then merge under the live bars.
        if symbol not in self._yf_seed:
            self._yf_seed[symbol] = self._yf_lookback(symbol, is_crypto)
        seed = self._yf_seed[symbol]
        if seed is None or seed.empty:
            return feed_df
        if feed_df is None or feed_df.empty:
            return seed
        fdf = feed_df.copy()
        try:  # align timezones so concat/dedup is well-defined
            fdf.index = fdf.index.tz_localize("UTC") if fdf.index.tz is None else fdf.index.tz_convert("UTC")
        except Exception:
            pass
        merged = pd.concat([seed[cols], fdf[cols]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged

    # ------------------------------------------------------------------
    # Lifecycle: per-step decision making
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
        
        self.iteration_count += 1
        if self.iteration_count == 1 and P.LOAD_PRETRAINED:
            self._load_pretrained()  # before anything else: models predict from hour 1
        portfolio_value = self.get_portfolio_value()
        cash = self.get_cash()

        # -- Drawdown tracking --
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        current_drawdown = (
            (self.peak_value - portfolio_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )

        # -- Pause/resume on max drawdown --
        # if self.paused:
        #     if current_drawdown <= P.DRAWDOWN_RECOVERY:
        #         self.paused = False
        #         self.log_message("Drawdown recovered, resuming trading")
        #     else:
        #         self.log_message(
        #             f"[PAUSED] drawdown={current_drawdown:.2%} | "
        #             f"portfolio=${portfolio_value:,.2f}"
        #         )
        #         return

        if current_drawdown >= P.MAX_DRAWDOWN:
            self.paused = True
            self._sell_all()
            self.log_message(
                f"MAX DRAWDOWN HIT ({current_drawdown:.2%}), liquidating all positions"
            )
            return

        # -- Collect current prices and historical data --
        current_prices = {}
        historical = {}
        fresh = {}
        crypto_quote = self.usd_quote
        for symbol in self.all_symbols:
            try:
                is_crypto = symbol in P.CRYPTO_SYMBOLS
                is_stock = symbol in P.STOCK_SLEEVE_SYMBOLS
                if is_stock:
                    price = self.get_last_price(symbol)
                    if price is None or price <= 0:
                        continue
                    current_prices[symbol] = price
                    
                    bars = self.get_historical_prices(
                        symbol, length=P.HISTORY_LENGTH_HOURS, timestep="hour"
                    )

                    # if bars is not None and hasattr(bars, "df") and len(bars.df) >= P.MIN_HISTORY_BARS:
                    #     historical[symbol] = bars.df
                    #     # Stock bar is fresh only if it isn't a verbatim copy of the
                    #     # prior bar. (Lumibot forward-fills the 24/7 grid, so the last
                    #     # bar's timestamp is always current — a timestamp age test
                    #     # passes even on weekends. A copied row means market closed.)
                    #     _c = ["open", "high", "low", "close", "volume"]
                    #     fresh[symbol] = not bars.df[_c].iloc[-1].equals(bars.df[_c].iloc[-2])
                    df = self._history_for(symbol, False, bars)
                    if df is not None and len(df) >= P.MIN_HISTORY_BARS:
                        historical[symbol] = df
                        # Stock bar is fresh only if it isn't a verbatim copy of the
                        # prior bar. (Lumibot forward-fills the 24/7 grid, so the last
                        # bar's timestamp is always current — a timestamp age test
                        # passes even on weekends. A copied row means market closed.)
                        _c = ["open", "high", "low", "close", "volume"]
                        fresh[symbol] = not df[_c].iloc[-1].equals(df[_c].iloc[-2])
                if is_crypto:
                    ast = Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
                    price = self.get_last_price(ast, quote=crypto_quote)
                    
                    if price is None or price <= 0:
                        continue
                    current_prices[symbol] = price
                    
                    bars = self.get_historical_prices(
                        ast, length=P.HISTORY_LENGTH_HOURS, timestep="hour", quote=crypto_quote
                    )

                    # if bars is not None and hasattr(bars, "df") and len(bars.df) >= P.MIN_HISTORY_BARS:
                    #     historical[symbol] = bars.df
                    #     fresh[symbol] = True
                    df = self._history_for(symbol, True, bars)
                    if df is not None and len(df) >= P.MIN_HISTORY_BARS:
                        historical[symbol] = df
                        fresh[symbol] = True
            except Exception as e:
                self.log_message(f"Data error for {symbol}: {e}")
                continue

        if not current_prices:
            self.log_message("No price data available, skipping iteration")
            return

        # -- Update rolling returns for cross-asset features --
        for symbol, price in current_prices.items():
            if symbol in self.prev_prices and self.prev_prices[symbol] > 0:
                ret = (price - self.prev_prices[symbol]) / self.prev_prices[symbol]
                new_entry = pd.Series([ret])
                self.asset_returns[symbol] = pd.concat(
                    [self.asset_returns[symbol], new_entry], ignore_index=True
                ).tail(100)

        # -- Update decay tracker from last iteration's predictions --
        for symbol, predicted_dir in self.last_predictions.items():
            if symbol in current_prices and symbol in self.prev_prices:
                if not fresh.get(symbol, True):
                    continue  # frozen bar: no market outcome to grade against
                actual_return = (
                    (current_prices[symbol] - self.prev_prices[symbol])
                    / self.prev_prices[symbol]
                )
                if actual_return > P.RETURN_DEAD_ZONE:
                    actual_dir = 1
                elif actual_return < -P.RETURN_DEAD_ZONE:
                    actual_dir = -1
                else:
                    actual_dir = 0
                self.tracker.update(symbol, predicted_dir == actual_dir)
        # -- Update mean-reversion tracker from last iteration's MR predictions --
        if P.MR_SWITCH_ENABLED: # Not used for competition since MR does not mix well with the momentum approach
            for symbol, predicted_dir in self.last_mr_predictions.items():
                if symbol in current_prices and symbol in self.prev_prices:
                    if not fresh.get(symbol, True):
                        continue  # frozen bar: no market outcome to grade against
                    r = (current_prices[symbol] - self.prev_prices[symbol]) / self.prev_prices[symbol]
                    a = 1 if r > P.RETURN_DEAD_ZONE else (-1 if r < -P.RETURN_DEAD_ZONE else 0)
                    self.mr_tracker.update(symbol, predicted_dir == a)
        # -- Add training data from previous iteration --
        if self._prev_features and self.prev_prices:
            for symbol, feat in self._prev_features.items():
                if symbol in current_prices and symbol in self.prev_prices:

                    if self.prev_prices[symbol] == current_prices[symbol]:
                        continue  # CHANGED (12062026) stale bar (market closed, price unchanged) — skip

                    target = TradingModel.compute_target(
                        self.prev_prices[symbol], current_prices[symbol]
                    )
                    # self.model.add_sample(feat, target)
                    self.models[symbol].add_sample(feat, target)
                    if P.MR_SWITCH_ENABLED:
                        self.mr_models[symbol].add_sample(
                            feat, MeanReversionModel.compute_target(
                                self.prev_prices[symbol], current_prices[symbol], float(feat.get("dist_dsma20", 0.0))
                            )
                        )
        # -- Feature engineering + prediction --
        signals = {}
        mr_signals = {}
        current_features = {}
        for symbol, df in historical.items():
            try:
                features = compute_features(df)
                cross_feats = compute_cross_asset_features(
                    self.asset_returns, symbol, self.all_symbols
                )
                for k, v in cross_feats.items():
                    features[k] = v

                current_features[symbol] = features
                # direction, confidence = self.model.predict(features)
                # signals[symbol] = (direction, confidence)
                if self.models[symbol].model is not None: # CHANGED (18062026)
                    direction, confidence = self.models[symbol].predict(features)
                    signals[symbol] = (direction, confidence)
                if P.MR_SWITCH_ENABLED and self.mr_models[symbol].model is not None:
                    mr_signals[symbol] = self.mr_models[symbol].predict(features)
            except Exception as e:
                self.log_message(f"Feature/predict error for {symbol}: {e}")
                continue

        # -- Compute per-symbol ATR for dynamic stop-loss --
        atr_pcts = {}
        for symbol, df in historical.items():
            atr = compute_atr_pct(df, period=P.ATR_PERIOD)
            if atr > 0:
                atr_pcts[symbol] = min(P.ATR_MULTIPLIER * atr, P.ATR_STOP_LOSS_CAP)
                    # -- Compute per-symbol regime: daily close vs daily SMA --
        # regime_ok = {}
        # for symbol, df in historical.items():
        #     regime_ok[symbol] = compute_regime_ok(df, sma_days=P.REGIME_SMA_DAYS, persist_days=P.REGIME_PERSISTENCE_DAYS)
        # regime_down = {}
        # for symbol, df in historical.items():
        #     regime_down[symbol] = compute_regime_down(df, sma_days=P.REGIME_SMA_DAYS, persist_days=P.REGIME_PERSISTENCE_DAYS)

        regime_ok = {}
        regime_down = {}
        for symbol, df in historical.items():
            sma_days = P.REGIME_SMA_DAYS_CRYPTO if symbol in P.CRYPTO_SYMBOLS else P.REGIME_SMA_DAYS
            regime_ok[symbol] = compute_regime_ok(df, sma_days=sma_days, persist_days=P.REGIME_PERSISTENCE_DAYS)
            regime_down[symbol] = compute_regime_down(df, sma_days=sma_days, persist_days=P.REGIME_PERSISTENCE_DAYS)

        # -- Store state for next iteration --
        self._prev_features = current_features
        self.last_predictions = {s: sig[0] for s, sig in signals.items()}
        self.last_mr_predictions = {s: sig[0] for s, sig in mr_signals.items()}
        self.prev_prices = current_prices.copy()
        if self.iteration_count == 1 and P.WARMUP_ENABLED:
            self._warmup_models()
        # if self.iteration_count == 1 and P.WARMUP_ENABLED: # Not used since competition requires loading weights
        #     self._warmup_models()
        # -- Train / retrain model --
        # if self.iteration_count == 1 or self.iteration_count % P.RETRAIN_INTERVAL == 0:
        #     if self.model.has_enough_data():
        #         try:
        #             success = self.model.train()
        #             self.log_message(
        #                 f"Model retrain: {'updated' if success else 'kept previous'} | "
        #                 f"samples={len(self.model.feature_history)} | "
        #                 f"val_acc={self.model.last_val_accuracy:.3f}"
        #             )
        #         except Exception as e:
        #             self.log_message(f"Retrain failed: {e}")
        # if self.iteration_count == 1 or self.iteration_count % P.RETRAIN_INTERVAL == 0: # CHANGED (18062026) from above
        #     # self.last_sold_prices.clear()  # clear stale buyback memory each retrain cycle
        #     for symbol, mdl in self.models.items():
        #         if mdl.has_enough_data():
        #             try:
        #                 success = mdl.train()
        #                 self.log_message(
        #                     f"Model retrain [{symbol}]: {'updated' if success else 'kept previous'} | "
        #                     f"samples={len(mdl.feature_history)} | "
        #                     f"val_acc={mdl.last_val_accuracy:.3f}"
        #                 )
        #             except Exception as e:
        #                 self.log_message(f"Retrain failed [{symbol}]: {e}")

        if P.RETRAIN_STAGGER:
            if self.iteration_count == 1 and P.WARMUP_ENABLED:
                # Warm-up path (backtests / offline pretraining): train ALL models once at
                # startup so _warmup_tracker has fitted models to replay Sample B through.
                # Never runs in competition (there LOAD_PRETRAINED=True, WARMUP_ENABLED=False).
                for symbol, mdl in self.models.items():
                    if mdl.has_enough_data():
                        try:
                            success = mdl.train()
                            self.log_message(
                                f"Model retrain [{symbol}] (warm-up): {'updated' if success else 'kept previous'} | "
                                f"samples={len(mdl.feature_history)} | val_acc={mdl.last_val_accuracy:.3f}"
                            )
                        except Exception as e:
                            self.log_message(f"Retrain failed [{symbol}]: {e}")
            else:
                # Staggered: retrain ONE model per iteration, cycling through the universe,
                # so no single iteration runs all 12 fits (which risks the competition's
                # per-iteration time limit). Each model still retrains every ~RETRAIN_INTERVAL iters.
                n = len(self.all_symbols)
                spacing = max(1, P.RETRAIN_INTERVAL // n)  # e.g. 96 // 12 = 8 iters between retrains
                if self.iteration_count > 1 and self.iteration_count % spacing == 0:
                    symbol = self.all_symbols[(self.iteration_count // spacing) % n]
                    mdl = self.models[symbol]
                    if mdl.has_enough_data():
                        try:
                            success = mdl.train()
                            self.log_message(
                                f"Model retrain [{symbol}] (staggered): {'updated' if success else 'kept previous'} | "
                                f"train_rows={min(len(mdl.feature_history), P.MAX_TRAIN_SAMPLES)} | "
                                f"val_acc={mdl.last_val_accuracy:.3f}"
                            )
                        except Exception as e:
                            self.log_message(f"Retrain failed [{symbol}]: {e}")
        else:
            if self.iteration_count == 1 or self.iteration_count % P.RETRAIN_INTERVAL == 0: # original all-at-once path
                for symbol, mdl in self.models.items():
                    if mdl.has_enough_data():
                        try:
                            success = mdl.train()
                            self.log_message(
                                f"Model retrain [{symbol}]: {'updated' if success else 'kept previous'} | "
                                f"samples={len(mdl.feature_history)} | "
                                f"val_acc={mdl.last_val_accuracy:.3f}"
                            )
                        except Exception as e:
                            self.log_message(f"Retrain failed [{symbol}]: {e}")
            if P.MR_SWITCH_ENABLED: # For competition, MR is disabled since it does not work well with momentum strategy
                for symbol, mdl in self.mr_models.items():
                    if mdl.has_enough_data():
                        try:
                            mdl.train()
                        except Exception as e:
                            self.log_message(f"MR retrain failed [{symbol}]: {e}")
        if self.iteration_count == 1 and P.WARMUP_ENABLED:
            self._warmup_tracker()
                # -- Paused: keep learning and watching; resume when regime recovers --
        if self.paused:
            # Resume when a majority-capped number of universe symbols are regime-up,
            # regardless of asset class — works for stock-only, crypto-only, or mixed.
            regime_up_count = sum(1 for s in self.all_symbols if regime_ok.get(s, False))
            resume_needed = min(P.REGIME_RESUME_MIN_SYMBOLS, (len(self.all_symbols) + 1) // 2)
            if regime_up_count >= resume_needed or current_drawdown <= P.DRAWDOWN_RECOVERY:
                self.paused = False
                self.peak_value = portfolio_value  # reset baseline so drawdown scaling doesn't throttle the restart
                self.log_message(
                    f"RESUME: {regime_up_count} symbols regime-up (needed {resume_needed}), "
                    f"drawdown baseline reset @ ${portfolio_value:,.2f}"
                )
            else:
                self.log_message(
                    f"[PAUSED] drawdown={current_drawdown:.2%} | "
                    f"regime-up symbols={regime_up_count}/{resume_needed} needed | "
                    f"portfolio=${portfolio_value:,.2f}"
                )
                return
        # -- If model not ready yet, stay in cash --
        # if self.model.model is None:
        #     self.log_message(
        #         f"Collecting data: {len(self.model.feature_history)}/{P.MIN_TRAINING_SAMPLES} samples"
        #     )
        #     return
        ready_count = sum(1 for m in self.models.values() if m.model is not None) # CHANGED (18062026) from above
        if ready_count == 0:
            total_samples = sum(len(m.feature_history) for m in self.models.values())
            self.log_message(
                f"Collecting data: {total_samples} total samples, no models ready yet"
            )
            return

        # -- Position sizing --
        target_weights = compute_position_sizes(
            signals, self.tracker, portfolio_value, current_drawdown
        )
        if signals and self.iteration_count % 24 == 0:
            adj = {s: c * self.tracker.get_score(s) for s, (d, c) in signals.items() if d != 0}
            if adj:
                stock_adj = {s: v for s, v in adj.items() if s not in P.CRYPTO_SYMBOLS}
                crypto_adj = {s: v for s, v in adj.items() if s in P.CRYPTO_SYMBOLS}
                parts = []
                if stock_adj:
                    t = max(stock_adj, key=stock_adj.get)
                    parts.append(f"stock_best={stock_adj[t]:.3f}({t}) vs {P.MIN_CONFIDENCE}")
                if crypto_adj:
                    t = max(crypto_adj, key=crypto_adj.get)
                    parts.append(f"crypto_best={crypto_adj[t]:.3f}({t}) vs {P.MIN_CONFIDENCE_CRYPTO}")
                self.log_message("[GATE] " + " | ".join(parts))
        # -- Chop switch: consecutive failed exits -> pause momentum; MR only in non-bear chop --
        n_active = len(regime_down or {})
        n_down = sum(1 for v in (regime_down or {}).values() if v)
        broad_bear_now = n_active > 0 and n_down >= P.BEAR_MARKET_FRACTION * n_active
        self.mr_mode_active = P.MR_SWITCH_ENABLED and self.failed_exit_streak >= P.FAILED_STREAK_THRESHOLD
        if self.mr_mode_active:
            target_weights = {}  # momentum entries paused; held positions keep their exits
            if not broad_bear_now:  # cascade bear -> stay in cash; range chop -> MR candidates
                for s, (mr_dir, mr_conf) in mr_signals.items():
                    feats = current_features.get(s)
                    if feats is None or float(feats.get("dist_dsma20", 0.0)) > -P.MR_DIST_THRESHOLD:
                        continue  # not stretched below its mean -> no MR setup
                    if regime_down.get(s, False):
                        continue  # confirmed breakdown: never catch that knife
                    if s in self.entry_prices:
                        continue  # already held: don't convert or top-up
                    if mr_dir == 1 and mr_conf * self.mr_tracker.get_score(s) > P.MIN_CONFIDENCE_MR:
                        target_weights[s] = P.MR_WEIGHT_SCALE * min(
                            P.MAX_WEIGHT_PER_POSITION, (1 - P.CASH_BUFFER) / P.MAX_POSITIONS
                        )
            # -- Flat-book timeout: idle lockout must not become permanent --
            if not self.entry_prices and not target_weights:
                self.mr_idle_iters += 1
                if self.mr_idle_iters >= P.MR_IDLE_TIMEOUT_ITERS:
                    self.failed_exit_streak = 0
                    self.mr_idle_iters = 0
                    self.mr_mode_active = False
                    self.log_message("[MR-TIMEOUT] flat and idle, streak reset; momentum gates re-armed")
            else:
                self.mr_idle_iters = 0
            if self.mr_mode_active:
                self.log_message(
                    f"[MR-MODE] streak={self.failed_exit_streak} bear={broad_bear_now} candidates={sorted(target_weights)}"
                )

        # -- Entry persistence: update consecutive-target streaks -- CHANGED (12062026)
        for symbol in self.all_symbols:
            if not fresh.get(symbol, True):
                continue  # frozen bar: hold streak state, don't tick on stale signals
            if target_weights.get(symbol, 0) > 0:
                self.target_streak[symbol] = self.target_streak.get(symbol, 0) + 1
            else:
                self.target_streak[symbol] = 0

        # -- Block first-time buys until streak >= ENTRY_PERSISTENCE -- # CHANGED (12062026), now included this
        held = {p.asset.symbol for p in self.get_positions() if p.quantity > 0}
        filtered_weights = {}
        for s, w in target_weights.items():
            if s in held:
                filtered_weights[s] = w
                continue
            if self.target_streak.get(s, 0) < P.ENTRY_PERSISTENCE:
                continue
            if self.mr_mode_active:
                filtered_weights[s] = w
                continue  # MR entries: regime_ok and momentum-tracker gates don't apply
            if P.REGIME_FILTER_ENABLED and not regime_ok.get(s, True):
                sma_d = P.REGIME_SMA_DAYS_CRYPTO if s in P.CRYPTO_SYMBOLS else P.REGIME_SMA_DAYS
                self.log_message(f"REGIME-BLOCKED {s}: daily close below {sma_d}d SMA")
                continue
            if self.tracker.get_score(s) < P.MIN_TRACKER_SCORE_FOR_ENTRY:
                self.log_message(f"TRACKER-BLOCKED {s}: score={self.tracker.get_score(s):.3f}")
                continue
            filtered_weights[s] = w
        target_weights = filtered_weights

        # -- Exit persistence: count consecutive iterations held symbols are out of targets -- # CHANGED (12062026), now included this
        for symbol in held:
            if not fresh.get(symbol, True):
                continue  # frozen bar: hold streak state
            if target_weights.get(symbol, 0) <= 0:
                self.exit_streak[symbol] = self.exit_streak.get(symbol, 0) + 1
            else:
                self.exit_streak[symbol] = 0

        # -- Execute orders --
        self._rebalance(target_weights, current_prices, portfolio_value, fresh, atr_pcts, regime_ok, regime_down)

        # -- Logging --
        held_count = sum(
            1 for p in self.get_positions()
            if p.quantity > 0 and p.asset.symbol != "USD"
        )
        self.log_message(
            f"iter={self.iteration_count} | portfolio=${portfolio_value:,.2f} | "
            f"cash=${cash:,.2f} | drawdown={current_drawdown:.2%} | "
            f"current_holding={held_count} | targets={len(target_weights)} | signals={len(signals)}"
        )

        # -- Out of sample testing --
        if self.iteration_count == 1:
            self._oos_logged = False
        if not hasattr(self, '_oos_logged'):
            self._oos_logged = False
        if not self._oos_logged and str(self.get_datetime().date()) >= "2025-07-01":
            self.log_message(f"=== OOS START === portfolio=${portfolio_value:,.2f}")
            self._oos_logged = True

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def _make_order(self, symbol, quantity, side):
        """Create an order, using the crypto Asset + USD quote pair for crypto symbols."""
        if symbol in P.CRYPTO_SYMBOLS:
            return self.create_order(
                self.crypto_assets[symbol], quantity, side,
                quote=self.usd_quote,
            )
        return self.create_order(symbol, quantity, side)
    
    def _rebalance(self, target_weights, prices, portfolio_value, fresh=None, atr_pcts=None, regime_ok=None, regime_down=None):
        """Rebalance portfolio to match target weights. Sells first, then buys."""
        # keep ~2% slippage reserve
        available_cash = max(float(self.get_cash()) - 0.02 * portfolio_value, 0.0) # CHANGED (12062025) to check cash before buying to avoid negative cash

        current_positions = {}
        for p in self.get_positions():
            sym = p.asset.symbol
            current_positions[sym] = p
        
        # -- Independent stop-loss: fires regardless of model signal --
        stop_loss_exited = set()
        for symbol, position in list(current_positions.items()):
            if position.quantity <= 0:
                continue
            price = prices.get(symbol, 0)
            entry_price = self.entry_prices.get(symbol, 0)
            if entry_price <= 0 or price <= 0:
                continue
            position_return = (price - entry_price) / entry_price
            sl_threshold = max((atr_pcts or {}).get(symbol, 0.0), P.MAX_POSITION_LOSS)
            # if regime_ok is not None and not regime_ok.get(symbol, True):
            #     sl_threshold = min(sl_threshold, P.REGIME_FLIP_STOP)
            if regime_down is not None and regime_down.get(symbol, False):
                flip_stop = max(P.REGIME_FLIP_STOP, (atr_pcts or {}).get(symbol, 0.0) / P.ATR_MULTIPLIER)
                sl_threshold = min(sl_threshold, flip_stop)
            if position_return <= -sl_threshold:
                # order = self.create_order(symbol, position.quantity, "sell") # CHANGED for crypto
                order = self._make_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.entry_prices.pop(symbol, None)
                self.peak_prices.pop(symbol, None)
                self.accumulated_fees.pop(symbol, None)
                self.exit_streak.pop(symbol, None)
                self.target_streak.pop(symbol, None)
                del current_positions[symbol]
                self.log_message(f"STOP-LOSS {symbol}: return={position_return:.2%} threshold={sl_threshold:.2%} @ ${price:,.2f}")
                self.failed_exit_streak = 0 if position_return >= P.FAILED_EXIT_RETURN else self.failed_exit_streak + 1
                self.mr_positions.discard(symbol)
                self.entry_times.pop(symbol, None)
                stop_loss_exited.add(symbol)
                self.last_stop_loss_time[symbol] = self.get_datetime()
        
        # -- Mean-reversion exits: quick-bank or time-stop --
        for symbol in list(self.mr_positions):
            position = current_positions.get(symbol)
            if position is None or position.quantity <= 0:
                self.mr_positions.discard(symbol)
                continue
            price = prices.get(symbol, 0)
            entry_price = self.entry_prices.get(symbol, 0)
            if entry_price <= 0 or price <= 0:
                continue
            position_return = (price - entry_price) / entry_price
            held_h = (self.get_datetime() - self.entry_times.get(symbol, self.get_datetime())).total_seconds() / 3600
            if position_return >= P.MR_PROFIT_TARGET or held_h >= P.MR_TIME_STOP_HOURS:
                order = self._make_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.entry_prices.pop(symbol, None)
                self.peak_prices.pop(symbol, None)
                self.accumulated_fees.pop(symbol, None)
                self.exit_streak.pop(symbol, None)
                self.target_streak.pop(symbol, None)
                del current_positions[symbol]
                self.log_message(f"MR-EXIT {symbol}: return={position_return:.2%} held={held_h:.0f}h @ ${price:,.2f}")
                if position_return >= P.FAILED_EXIT_RETURN:
                    self.failed_exit_streak = max(0, self.failed_exit_streak - 1)  # MR win: evidence accrues, doesn't reset
                else:
                    self.failed_exit_streak += 1
                self.mr_positions.discard(symbol)
                self.entry_times.pop(symbol, None)

        # -- Trailing stop: arms once position peaks >= TRAILING_STOP_ACTIVATION,
        #    then exits if price falls TRAILING_STOP_PCT from that peak --
        trailing_stop_exited = set()

        # Broad bear-market gauge: fraction of the active universe in a confirmed
        # downtrend. A single weak name doesn't count as a bear market.
        n_active = len(regime_down or {})
        n_down = sum(1 for v in (regime_down or {}).values() if v)
        broad_bear = n_active > 0 and n_down >= P.BEAR_MARKET_FRACTION * n_active
        if broad_bear:
            self.log_message(f"BROAD_BEAR MARKET. DOWN: {[s for s,v in regime_down.items() if v]}")
        for symbol, position in list(current_positions.items()):
            if symbol in stop_loss_exited or position.quantity <= 0:
                continue
            if symbol in self.mr_positions:
                continue  # MR positions exit via their own profit-target/time-stop
            price = prices.get(symbol, 0)
            entry_price = self.entry_prices.get(symbol, 0)
            if entry_price <= 0 or price <= 0:
                continue
            peak = max(self.peak_prices.get(symbol, entry_price), price)
            self.peak_prices[symbol] = peak
            peak_return = (peak - entry_price) / entry_price
            activation = P.TRAILING_STOP_ACTIVATION_BEAR if broad_bear else P.TRAILING_STOP_ACTIVATION
            trail_pct = P.TRAILING_STOP_PCT_BEAR if broad_bear else P.TRAILING_STOP_PCT
            if peak_return < activation:
                continue
            if price <= peak * (1 - trail_pct):
                position_return = (price - entry_price) / entry_price
                # order = self.create_order(symbol, position.quantity, "sell") # CHANGED for crypto
                order = self._make_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.entry_prices.pop(symbol, None)
                self.peak_prices.pop(symbol, None)
                self.accumulated_fees.pop(symbol, None)
                self.exit_streak.pop(symbol, None)
                self.target_streak.pop(symbol, None)
                del current_positions[symbol]
                self.log_message(f"TRAILING-STOP {symbol}: return={position_return:.2%} peak={peak_return:.2%} @ ${price:,.2f}")
                self.failed_exit_streak = 0 if position_return >= P.FAILED_EXIT_RETURN else self.failed_exit_streak + 1
                self.mr_positions.discard(symbol)
                self.entry_times.pop(symbol, None)
                trailing_stop_exited.add(symbol)
        # -- Sell: exit positions not in targets or with negative weight -- 
        for symbol, position in current_positions.items():
            if symbol in self.mr_positions:
                continue  # MR positions exit via their own profit-target/time-stop
            target_w = target_weights.get(symbol, 0)
            if target_w <= 0 and position.quantity > 0:
                if self.exit_streak.get(symbol, 0) < P.EXIT_PERSISTENCE:
                    continue  # CHANGED (12062026) now included, signal must stay dead for N consecutive hours before exiting

                price = prices.get(symbol, 0) # CHANGED (21062026)
                entry_price = self.entry_prices.get(symbol, 0)

                # Stop-loss override: always allow sell if position is down too much
                if entry_price <= 0 or price <= 0:
                    continue
                position_return = (price - entry_price) / entry_price
                sl_threshold = max((atr_pcts or {}).get(symbol, 0.0), P.MAX_POSITION_LOSS)
                is_stop_loss = position_return <= -sl_threshold
                total_entry_value = position.quantity * entry_price
                total_buy_fees = self.accumulated_fees.get(symbol, 0)
                sell_fee = position.quantity * price * P.PERCENT_FEE_PER_SIDE
                # min_profit_needed = (total_buy_fees + sell_fee) * P.MIN_SELL_PROFIT_MULTIPLIER / total_entry_value
                min_profit_needed = P.MIN_MODEL_EXIT_RETURN
                meets_profit_target = position_return >= min_profit_needed
                # is_underwater = position_return < 0
                if not meets_profit_target and not is_stop_loss: # and not is_underwater
                    continue  # price hasn't moved enough to justify selling

                # order = self.create_order(symbol, position.quantity, "sell") # CHANGED for crypto
                order = self._make_order(symbol, position.quantity, "sell")
                self.submit_order(order)
                self.exit_streak.pop(symbol, None) # CHANGED (12062026)
                # self.last_sold_prices[symbol] = price  # CHANGED (21062026) track sold price
                self.entry_prices.pop(symbol, None)    # CHANGED (21062026) clear entry price
                self.peak_prices.pop(symbol, None)
                self.accumulated_fees.pop(symbol, None)
                
                self.log_message(f"SELL ALL {symbol}: qty={position.quantity}")
                self.failed_exit_streak = 0 if position_return >= P.FAILED_EXIT_RETURN else self.failed_exit_streak + 1
                self.mr_positions.discard(symbol)
                self.entry_times.pop(symbol, None)

        # -- Buy / adjust remaining positions --
        for symbol, weight in target_weights.items():
            if symbol in stop_loss_exited or symbol in trailing_stop_exited:
                continue
            last_sl = self.last_stop_loss_time.get(symbol)
            if last_sl is not None:
                hours_since = (self.get_datetime() - last_sl).total_seconds() / 3600
                if hours_since < P.STOP_LOSS_COOLDOWN_HOURS:
                    continue
            if weight <= 0:
                continue

            # Hard cap on holding 5 stocks (just testing)
            already_held = symbol in current_positions and current_positions[symbol].quantity > 0
            if not already_held:
                held_now = sum(1 for p in self.get_positions()
                               if p.quantity > 0 and p.asset.symbol != "USD")
                if held_now >= P.MAX_POSITIONS:
                    continue  # hard cap: no new symbols until an exit frees a slot

            if not (fresh or {}).get(symbol, True):
                continue  # stale off-hours bar: defer to next market-hours iteration
            price = prices.get(symbol)
            if not price or price <= 0:
                continue

            # CHANGED (21062026) Buy-back threshold: only buy if price has dropped enough from last sold price
            # last_sold = self.last_sold_prices.get(symbol)
            # if last_sold is not None: # and price > last_sold * (1 - P.MIN_BUYBACK_DROP):
            #     self.log_message(f"[BUYBACK BLOCKED] {symbol}: price=${price:.2f} > sold=${last_sold:.2f} * {1 - P.MIN_BUYBACK_DROP:.4f} = ${last_sold * (1 - P.MIN_BUYBACK_DROP):.2f}")
            #     continue

            target_value = abs(weight) * portfolio_value * 0.95 # CHANGED (12062026) from *1 to *0.6 to reduce amount transacted each trade

            current_value = 0
            current_qty = 0
            if symbol in current_positions:
                current_qty = current_positions[symbol].quantity
                current_value = current_qty * price

            tolerance = 0.05 * current_value # CHANGED (12062026) included now to ignore rebalance for small price movements

            diff_value = target_value - current_value
            cash_reserve = available_cash * (P.CASH_BUFFER) # CHANGED (06122026)
            max_order_notional = 350000 # CHANGED (06122026)
            spendable = available_cash - cash_reserve
            diff_value = min(diff_value, max_order_notional, spendable) # CHANGED (06122026)
            if abs(diff_value) < tolerance or (available_cash - abs(diff_value)) < cash_reserve or diff_value <= 0:  # skip tiny adjustments, also CHANGED from 10 to tolerance to ignore rebalance for small price movements
                continue
            # CHANGED (12062026) abs(diff_value) > cash_reserve to prevent overspending.

            # Crypto allows fractional, stocks need whole shares
            is_crypto = symbol in P.CRYPTO_SYMBOLS
            if is_crypto:
                quantity = round(abs(diff_value) / price, 6)
            else:
                quantity = int(abs(diff_value) / price)

            if quantity <= 0:
                continue

            if diff_value > 0:
                # order = self.create_order(symbol, quantity, "buy") # CHANGED for crypto
                order = self._make_order(symbol, quantity, "buy")
                self.submit_order(order)
                available_cash -= quantity * price * (1 + P.PERCENT_FEE_PER_SIDE) # To avoid negative cash
                # self.entry_prices[symbol] = price  # track entry price
                old_price = self.entry_prices.get(symbol, 0)
                old_qty = current_qty
                # self.last_sold_prices.pop(symbol, None)  # clear sold price
                if old_price > 0 and old_qty > 0:
                    self.entry_prices[symbol] = (old_qty * old_price + quantity * price) / (old_qty + quantity)
                else:
                    self.entry_prices[symbol] = price
                self.accumulated_fees[symbol] = self.accumulated_fees.get(symbol, 0) + quantity * price * P.PERCENT_FEE_PER_SIDE
                self.entry_times.setdefault(symbol, self.get_datetime())
                if self.mr_mode_active:
                    self.mr_positions.add(symbol)
                self.log_message(f"BUY {symbol}: qty={quantity} @ ${price:,.2f}")
            else:
                quantity = min(quantity, current_qty)
                if quantity > 0:
                    # order = self.create_order(symbol, quantity, "sell") # CHANGED for crypto
                    order = self._make_order(symbol, quantity, "sell")
                    self.submit_order(order)
                    self.log_message(f"SELL {symbol}: qty={quantity} @ ${price:,.2f}")
                else:
                    continue

    def _sell_all(self):
        """Liquidate all positions (used when max drawdown is hit)."""
        for position in self.get_positions():
            if position.quantity > 0:
                # order = self.create_order(position.asset.symbol, position.quantity, "sell")
                order = self._make_order(position.asset.symbol, position.quantity, "sell")
                self.submit_order(order)
                self.log_message(
                    f"LIQUIDATE {position.asset.symbol}: qty={position.quantity}"
                )
        self.entry_prices.clear()
        self.peak_prices.clear()
        self.accumulated_fees.clear()
        self.target_streak.clear()
        self.exit_streak.clear()
