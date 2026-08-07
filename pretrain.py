"""
Offline pre-training for the competition (run locally, never submitted as a
runtime step).

Why this exists
---------------
The organizers do not allow lengthy model training at competition startup
("15-minute startup training is not permissible ... upload the saved weights
instead"). This script therefore runs the strategy's OWN warm-up pipeline
(_warmup_models -> iteration-1 training -> _warmup_tracker) inside a tiny
local backtest, then saves every fitted model and the primed DecayTracker to
WEIGHTS_DIR. At competition time the strategy loads these files instead of
training (params: LOAD_PRETRAINED=True, WARMUP_ENABLED=False).

Running the real warm-up inside a real lumibot backtest (rather than
re-implementing it in pandas) guarantees the saved models are byte-identical
to what the validated strategy produces: same 24/7 grid, same forward-fill
behaviour, same feature pipeline.

Usage
-----
    python pretrain.py

All configuration lives in params.py (single source of truth):

    PRETRAIN_END_DATE   = None          -> anchor to the END of available data (production)
                        = "2025-05-31"  -> pretend the data ends there (historical rehearsal)
    PRETRAIN_WEIGHTS_DIR = None          -> save to WEIGHTS_DIR (production: weights/)
                         = "weights_test" -> save to a separate rehearsal folder

The end-date truncates every CSV before the warm-up runs, letting you build
weights "as of" a historical date with zero lookahead - e.g. pretrain as of
31 May 2025, then run rehearsal_backtest.py from 1 Jun 2025 in LOAD_PRETRAINED
mode to rehearse the full competition flow (pretrain July -> trade August).

- Uses the same CSVs as backtest.py (data/{SYMBOL}_1h_spot.csv).
- Trains on the LAST WARMUP_LENGTH_HOURS before the anchor date, so re-run
  right before submission (after refreshing the CSVs) so the weights reflect
  data through late July.
- Prints the on-disk footprint of the weights folder at the end.

Requires in params.py while running: WARMUP_ENABLED=True (asserted below;
LOAD_PRETRAINED is forced False in-process for the duration of the run).
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from strategies import params as P

# Pretraining must go through the warm-up path, never the load-weights path.
P.LOAD_PRETRAINED = False
assert P.WARMUP_ENABLED, "Set WARMUP_ENABLED=True in params.py before pretraining."

import backtest as B  # reuse the CSV loader + fee config (import only; no run)
from strategies.strategy import Strategy

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / P.WEIGHTS_DIR  # may be overridden by the 2nd CLI argument


def _load_data(end_cutoff=None):
    """Same as backtest._load_pandas_data, with an optional end-date truncation
    applied to every CSV so weights can be built 'as of' a historical date."""
    symbols = list(dict.fromkeys(
        P.STOCK_SLEEVE_SYMBOLS + P.CRYPTO_SLEEVE_SYMBOLS + [P.STOCK_BENCH, P.CRYPTO_BENCH]
    ))
    pandas_data, ends = {}, []
    crypto_quote = B.Asset(symbol="USD", asset_type=B.Asset.AssetType.FOREX)
    stock_quote = B.Asset(symbol="USD", asset_type=B.Asset.AssetType.FOREX)
    for symbol in symbols:
        path = B.DATA_DIR / f"{B._normalize_symbol(symbol)}_1h_spot.csv"
        if not path.exists():
            print(f"[WARN] missing CSV for {symbol}, skipped")
            continue
        df = B._read_raw_csv(path)
        if end_cutoff is not None:
            df = df.loc[:end_cutoff]
        if df.empty:
            print(f"[WARN] no data before cutoff for {symbol}, skipped")
            continue
        is_crypto = symbol in P.CRYPTO_SYMBOLS
        asset = B.Asset(symbol=symbol,
                        asset_type=B.Asset.AssetType.CRYPTO if is_crypto else B.Asset.AssetType.STOCK)
        pandas_data[asset] = B.Data(asset, df, timestep="hour",
                                    quote=crypto_quote if is_crypto else stock_quote)
        ends.append(df.index.max())
    return pandas_data, min(ends)


class PretrainStrategy(Strategy):
    """The real strategy, hijacked to save its state right after warm-up."""

    def on_trading_iteration(self):
        super().on_trading_iteration()
        if self.iteration_count != 1:
            return

        WEIGHTS.mkdir(parents=True, exist_ok=True)

        # -- Save every fitted model (momentum; MR only if enabled) --
        saved, skipped = [], []
        for symbol, mdl in self.models.items():
            if mdl.model is not None:
                mdl.save(str(WEIGHTS))
                saved.append(symbol)
            else:
                skipped.append(symbol)
        if P.MR_SWITCH_ENABLED:
            for symbol, mdl in self.mr_models.items():
                if mdl.model is not None:
                    mdl.save(str(WEIGHTS / "mr"))

        # -- Save the primed DecayTracker (tiny: JSON) --
        tracker_state = {
            s: {
                "score": self.tracker.accuracy_scores.get(s, P.INITIAL_ACCURACY),
                # keep only the recent window; older entries are decayed to irrelevance
                "predictions": [bool(x) for x in self.tracker.predictions.get(s, [])[-P.TRACKER_WARMUP_BARS:]],
            }
            for s in self.all_symbols
        }
        (WEIGHTS / "tracker.json").write_text(json.dumps(tracker_state, indent=1), encoding="utf-8")

        # -- Manifest for traceability --
        manifest = {
            "saved_models": saved,
            "skipped_no_model": skipped,
            "data_end": str(self.get_datetime()),
            "warmup_train_hours": P.WARMUP_TRAIN_HOURS,
            "tracker_replay_hours": P.TRACKER_REPLAY_HOURS,
            "max_train_samples": P.MAX_TRAIN_SAMPLES,
            "n_estimators": P.N_ESTIMATORS,
            "symbols": self.all_symbols,
        }
        (WEIGHTS / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        self.log_message(f"PRETRAIN: saved {len(saved)} models -> {WEIGHTS} | skipped: {skipped or 'none'}")


def main() -> None:
    global WEIGHTS
    end_cutoff = None
    if getattr(P, "PRETRAIN_END_DATE", None):
        end_cutoff = pd.Timestamp(P.PRETRAIN_END_DATE, tz="UTC")
        print(f"[INFO] PRETRAIN_END_DATE set: truncating all data to <= {end_cutoff} (historical rehearsal mode)")
    else:
        print("[INFO] PRETRAIN_END_DATE=None: anchoring to the end of available data (production mode)")
    out_dir = getattr(P, "PRETRAIN_WEIGHTS_DIR", None) or P.WEIGHTS_DIR
    WEIGHTS = ROOT / out_dir
    print(f"[INFO] saving weights to {WEIGHTS}")

    pandas_data, data_end_ts = _load_data(end_cutoff)
    data_end = data_end_ts.to_pydatetime()
    # Tiny trading window at the very END of the data: warm-up (which uses
    # WARMUP_LENGTH_HOURS of history before the start) sees the freshest bars.
    bt_start = data_end - timedelta(hours=12)
    print(f"[INFO] pretraining with warm-up on data ending {data_end} (window {bt_start} -> {data_end})")

    PretrainStrategy.run_backtest(
        B.PandasDataBacktesting,
        bt_start,
        data_end,
        pandas_data=pandas_data,
        budget=B.BUDGET,
        benchmark_asset=None,        # no benchmark: avoids the yfinance download entirely
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_indicators=False,
        save_logfile=False,
        **B._execution_cost_kwargs(),
    )

    # -- Footprint report --
    if WEIGHTS.exists():
        files = sorted(WEIGHTS.rglob("*"))
        total = 0
        print("\n=== weights/ footprint ===")
        for f in files:
            if f.is_file():
                kb = f.stat().st_size / 1024
                total += kb
                print(f"  {f.relative_to(ROOT)}  {kb:,.0f} KB")
        print(f"  TOTAL: {total / 1024:,.1f} MB")
        if total / 1024 > 150:
            print("  [WARN] large footprint - consider gzip or smaller N_ESTIMATORS before committing.")
    else:
        print("[ERROR] weights/ was not created - warm-up likely produced no trained models.")


if __name__ == "__main__":
    main()
