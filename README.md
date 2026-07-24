SoAI 2026 — ML Momentum Strategy (XGBoost, multi-asset, hourly)

----------

0. Overview (summary)

- A regime-filtered momentum strategy trading 9 US large-caps and 3 crypto majors
  on hourly bars. 
- Each asset has its own XGBoost direction classifier; an adaptive
  accuracy tracker sizes positions; a layered risk stack (regime filters, ATR and
  trailing stops, drawdown controls) governs exposure. 
- Models are pre-trained offline and loaded at startup — no training happens during the competition's
  startup window (as mentioned by the organisers in an email correspondence)

Remark: 
Named functions and parameters below point into the modular files under
`strategies/`; search for the name to jump to the definition.

1. Entry point & how it runs

- `strategies/strategy.py` defines `class Strategy(lumibot.strategies.Strategy)`,
  importable as `from strategies.strategy import Strategy`.
- `self.sleeptime = "60M"` — the strategy consumes the official minute feed
  resampled to hourly. Market set to 24/7 so crypto trades around the clock; 
  stock actions are gated to live bars only (§5).
- Startup loads pre-trained weights (`_load_pretrained` in
  `strategies/strategy.py`) from `weights/` at the first iteration — zero training
  at startup, as per the organizers' fairness rule (communicated via email)
  `params.py`:
  `LOAD_PRETRAINED = True`,
  `WARMUP_ENABLED = False`.

2. Signal generation — XGBoost per asset (`TradingModel`, `model.py`)

- One 3-class classifier per symbol: sell / hold / buy over the next hour,
  labeled with a plus minus 0.3% dead zone (`compute_target`).
- ~30 engineered features per bar (`compute_features`, `features.py`):
  returns, RSI/MACD/Bollinger, volume, ADX, daily-context
  (`dist_dsma20`, `vol_20d`), plus a cross-asset momentum rank.
- Two-pass training (`train`): rank feature importance on all
  features, retrain on the top 13
  (`NUM_SELECTED_FEATURES` in `params.py`).

3. Pre-trained weights (`weights/`)

- Built offline by `pretrain.py`, which runs the strategy's own warm-up pipeline
  (train on ~5500 grid-hours = Sample A, then grade an unseen
  ~2260-hour Sample B into the tracker) on historical data, then calls `TradingModel.save`. 
- Each model is stored as a gzipped XGBoost-3.x booster (`<SYMBOL>.ubj.gz`) plus 
  `_meta.npz` of the recent training rows and selected features; 
  `tracker.json` holds the primed accuracy scores; 
  `manifest.json` records the data cut-off. 
  `_load_pretrained` restores all of it via `TradingModel.load` with per-symbol
  error handling (a single bad file degrades that one symbol only, not the run).

From email correspondence with organisers, 
periodic online retraining is permitted and allowed to continue during the competition.
(`train` on a rolling 5000-row window). It is staggered
— one model per iteration — so no single iteration runs all
12 fits, keeping every step well inside the execution-time limit.
The standard execution time limit was not disclosed, so we estimated it.

4. Position sizing — adaptive tracker (`DecayTracker`, `risk.py`)

- Each model's live predictions are graded hourly; a decay-weighted accuracy score
  scales its signal. 
- `compute_position_sizes`: confidence x accuracy, governed 
  by per-class conviction (0.35 stocks / 0.17 crypto), top-5 by strength,
  35% weight cap, 8% cash buffer.
- Note that on top of the 8% cash buffer, _rebalance holds back a further 2% of portfolio 
  value as a slippage reserve before buying (approximately $20,000 at the $1M starting 
  capital, scaling with the portfolio value).

5. Risk management (`_rebalance`, `strategy.py`)

- Regime filter (entry gate): new longs only when
  the last two daily closes are above the daily SMA — 20-day
  stocks / 40-day crypto (`compute_regime_ok`, `features.py`). 
- Daily closes strip forward-filled non-trading days (`_drop_stale_days`) 
  so weekends never distort stock SMAs on the 24/7 grid.
- ATR stop-loss: max(2.0x daily ATR, 5%) capped at 12%
  (`compute_atr_pct`); a confirmed breakdown
  (`compute_regime_down`) tightens it.
- Trailing stop: arms at +3% peak, exits 2.5% off it. However, in a broad 
  bear regime (gauge), it switches to an early-arm tight trail to take profit
  before failed rallies (potentially) reverse.
- Freshness gate: a stock bar identical to its predecessor means the market 
  is closed — no orders, learning, or stop checks on frozen bars. 
  Crypto is always live.
- Portfolio brake: liquidate + pause at 30% drawdown, resume when 
  the universe turns broadly regime-up.

6. Experimental mean-reversion sleeve — disabled for competition

- The code contains a complete regime-switching mean-reversion sleeve
  (`# -- Chop switch` in `strategy.py`) that failed its pre-registered
  validation and ships **off** (`MR_SWITCH_ENABLED = False` in
  `params.py`). With `MR_SWITCH_ENABLED = False`, every MR path is inert, so
  the strategy is pure momentum which is what is submitted for this competition. 
  The mean-reversion code blocks are retained for post-competition research.

7. Local reproduction

pip install -r requirements.txt
python backtest.py # optional local dev harness — not used by the official run

- `backtest.py` is an optional local development harness — not used by the
  official evaluation, which feeds its own live data to the `Strategy` class 
  (we supply no data to the competition). 
- This submission ships code and pre-trained `weights/` only — no data CSVs — so 
  the harness will not run as-is without local OHLCV CSVs in `data/`; running it 
  is entirely optional. 
- The official run feeds live CCXT (crypto) and Massive (US equity) minute 
  bars into the same class.

`pretrain.py` regenerates `weights/` from local CSVs; re-run it after refreshing
the data so the committed weights reflect the latest pre-competition history.

Validation (local backtest, competition mode — weights loaded, no startup
training, staggered retrains, 8 bps/side):
- Trained on data through 30 May 2025, then traded 1 Jun 2025 -> 31 May 2026 with zero 
  lookahead: approximately +120.5%
  (max drawdown -10.0%, Sharpe 3.37, Calmar 12.1) versus SPY +29%. 
- The strategy is largely uncorrelated with the benchmark (R^2 0.26), 
  i.e. the return is alpha, not leveraged market beta. 
- The exit parameters were additionally validated on two out-of-sample windows 
  with independently pre-trained weights (trained through 31 Aug / 31 Oct 2025): 
  +45.6% over Sep 2025 -> Jun 2026, and
  +40.8% over Nov 2025 -> Jun 2026, 
  beating the prior parameter set on terminal value, Sharpe, and drawdown in 
  all three windows.

8. Data & execution notes

- Only OHLCV bars are used — no order book, news, or alternative data, as per 
  competition instructions.
- Orders are market orders sized under typical per-minute volume for these
  large-cap / major symbols; the model's volume-aware fill caps are respected
  by sizing, not assumed away.
- No hard-coded paths, no interactive prompts. The only runtime network call
  is a cold-start safeguard: if the live feed serves no pre-competition history
  at startup, the strategy fetches trailing hourly bars via `yf.download()`
  (permitted by the organizers, as replied by one of the organisers in an email). 
  It is skipped entirely when the feed provides history, and fails
  safe if Yahoo is unreachable. (The yfinance benchmark in the local harness
  remains optional and disabled for the official-style run.)

----------
End of README.
----------