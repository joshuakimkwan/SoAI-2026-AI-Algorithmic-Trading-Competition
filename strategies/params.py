"""Shared parameters for the ML trading strategy."""

# -- Asset universe -----------------------------------------
STOCK_SLEEVE_SYMBOLS = ["NVDA", "LLY", "VRTX", "AAPL", "MOD", "AMD", "ABBV", "COST", "GOOGL"]
CRYPTO_SLEEVE_SYMBOLS = ["BTC", "ETH", "SOL"]
CRYPTO_SYMBOLS = set(CRYPTO_SLEEVE_SYMBOLS)
ALL_SYMBOLS = STOCK_SLEEVE_SYMBOLS + CRYPTO_SLEEVE_SYMBOLS

STOCK_BENCH = "SPY"
CRYPTO_BENCH = "BTC"

# -- Strategy -----------------------------------------------
SLEEPTIME = "60M"
MAX_POSITIONS = 5
MAX_WEIGHT_PER_POSITION = 0.35

CASH_BUFFER = 0.08
MIN_CONFIDENCE = 0.35
MIN_CONFIDENCE_CRYPTO = 0.17
ENTRY_PERSISTENCE = 1
EXIT_PERSISTENCE = 2

# -- Transaction costs (used for threshold calculations) ----
PERCENT_FEE_PER_SIDE = 0.0008 # 2 bps per side. Assume 6 bps per side accounts for slippage

# -- Trade thresholds (as multiples of round-trip cost) -----
ROUND_TRIP_COST = 2 * PERCENT_FEE_PER_SIDE  # 0.0014 = 14 bps TOASK to import backtest??
MIN_SELL_PROFIT_MULTIPLIER = 1.5            # sell only if profit >= 2.5× round-trip cost
# MIN_BUYBACK_MULTIPLIER = 0.5              # buy back only if price dropped >= 0.6× round-trip cost

MIN_SELL_PROFIT = ROUND_TRIP_COST * MIN_SELL_PROFIT_MULTIPLIER      # CHANGED (21062026) only sell if price >= entry * (1 + this)
MIN_MODEL_EXIT_RETURN = 0.03    # model-driven exit only fires if position is up >= this (or stop-loss). Use 0.03
TRAILING_STOP_ACTIVATION = 0.03 # trailing stop arms once peak return since entry >= this. Use 0.03
TRAILING_STOP_PCT = 0.025        # exit if price falls this fraction from the peak since entry. Use 0.025
# MIN_BUYBACK_DROP = ROUND_TRIP_COST * MIN_BUYBACK_MULTIPLIER     # CHANGED (21062026) only buy back if price <= last_sold * (1 - this) 0.00084
MAX_POSITION_LOSS = 0.05      # CHANGED (21062026) stop-loss override: sell if down more than this (recommended). Original is 0.05 drop (5% drop of price bought)
ATR_PERIOD = 10              # trading dsays of daily bars used for ATR
ATR_MULTIPLIER = 2.0         # stop-loss threshold = ATR_MULTIPLIER × ATR%
ATR_STOP_LOSS_CAP = 0.12     # maximum threshold regardless of ATR (20%)

# -- Regime checking ----------------------------------------
REGIME_SMA_DAYS = 20            # daily SMA lookback for regime filter
REGIME_SMA_DAYS_CRYPTO = 40     # crypto needs a slower SMA: study shows 20d has ~zero edge, 40d has the best up/down separation
REGIME_FILTER_ENABLED = True    # only allow NEW entries when daily close > SMA
REGIME_PERSISTENCE_DAYS = 2     # consecutive daily closes above SMA required before entries reopen
REGIME_FLIP_STOP = 0.03         # tightened stop for held positions whose regime turns down
REGIME_RESUME_MIN_SYMBOLS = 3   # resume from max-drawdown pause when >= this many stocks are regime-up
TRAILING_STOP_ACTIVATION_BEAR = 0.011
TRAILING_STOP_PCT_BEAR = 0.003
BEAR_MARKET_FRACTION = 0.5      # bear regime when >= this *fraction* of the active universe is regime_down

# -- Chop switch + mean-reversion sleeve --------------------
MR_SWITCH_ENABLED = False         # master switch: False = pure momentum (frozen baseline behavior) NOT used for competition!
FAILED_EXIT_RETURN = 0.02
FAILED_STREAK_THRESHOLD = 3
MR_DIST_THRESHOLD = 0.02
MIN_CONFIDENCE_MR = 0.4
MR_PROFIT_TARGET = 0.02
MR_TIME_STOP_HOURS = 72
MR_WEIGHT_SCALE = 0.5
MR_IDLE_TIMEOUT_ITERS = 96        # flat + idle lockout resets streak after ~4 days (24/7 grid)

MIN_TRACKER_SCORE_FOR_ENTRY = 0.4   # block new entries when recent model accuracy below this
HISTORY_LENGTH_HOURS = 1250         # bars are on a 24/7 grid (crypto data): 700h ≈ 29 calendar days ≈ 20+ trading days
# WARMUP_ENABLED = True             # Moved to competition section below
# WARMUP_LENGTH_HOURS = 5500        # ~5-6 months of 24/7-grid hours used to pre-train models at startup

# For actual
WARMUP_TRAIN_HOURS = 5500           # Sample A: grid-hours of history used to pre-train models (~8 months)
TRACKER_REPLAY_HOURS = 2260         # Sample B: grid-hours after A, never used in training, replayed to grade models for the tracker (~3 months)

# Formula to decide WARMUP_TRAIN_HOURS and TRACKER_REPLAY_HOURS, where m is the number of months
# and m_train ≥ MIN_TRAINING_SAMPLES / 110 since Training months must yield enough samples:
# WARMUP_TRAIN_HOURS   = ceil(m_train  × 730 × 1.02 / 100) × 100
# TRACKER_REPLAY_HOURS = ceil(m_replay × 730 × 1.02 / 100) × 100

# For backtest only (next 3 lines WARMUP_TRAIN_HOURS, TRACKER_REPLAY_HOURS, MIN_TRAINING_SAMPLES)
# WARMUP_TRAIN_HOURS = 4500     # test only: data before Mar 2025 supports ~4500+1000
# TRACKER_REPLAY_HOURS = 1000   # test only
# MIN_TRAINING_SAMPLES = 600    # test only: 4500h yields ~650 stock samples

WARMUP_LENGTH_HOURS = WARMUP_TRAIN_HOURS + TRACKER_REPLAY_HOURS     # total history fetched at startup
TRACKER_WARMUP_BARS = 300                                           # cap on replayed bars within Sample B (decay makes older updates irrelevant)

# -- Risk ---------------------------------------------------
MAX_DRAWDOWN = 0.3
DRAWDOWN_SCALING_START = 0.2
DRAWDOWN_RECOVERY = 0.2
STOP_LOSS_COOLDOWN_HOURS = 36

# -- XGBoost model ------------------------------------------
N_ESTIMATORS = 2000
MAX_DEPTH = 10
LEARNING_RATE = 0.006758
SUBSAMPLE = 0.925368
COLSAMPLE_BYTREE = 0.857325
MIN_TRAINING_SAMPLES = 800
RETRAIN_INTERVAL = 96  # iterations (hours)

# -- Competition deployment: pre-trained weights + staggered retrain --
LOAD_PRETRAINED = True              # True in the competition (load weights/); False for backtests & pretraining (warm-up)
WARMUP_ENABLED = False              # Not allowed to pretrain upon start of competition
WEIGHTS_DIR = "weights"             # folder holding saved model boosters + metadata + tracker
MAX_TRAIN_SAMPLES = 5000            # cap per-model training rows to bound retrain time (crypto accumulates ~11k)
RETRAIN_STAGGER = True              # retrain one model per iteration instead of all 12 at once

# -- Offline pretraining (pretrain.py) ----------------------
PRETRAIN_END_DATE = None           # None = anchor to end of available data (production run);
                                   # a date string = historical rehearsal 'as of' that date
PRETRAIN_WEIGHTS_DIR = None        # None = WEIGHTS_DIR (production); separate folder for rehearsals

# -- Cold-start lookback backfill (yfinance) ----------------
# If the live feed serves fewer than MIN_HISTORY_BARS at startup (no
# pre-competition lookback), pull trailing hourly history from Yahoo once per
# symbol and merge it UNDER the live bars. Inert when the feed already serves
# enough history (Yahoo is never called); fails safe if Yahoo is unreachable.
YF_BACKFILL_ENABLED = True      # master switch for the cold-start backfill
YF_BACKFILL_DAYS = 90           # calendar days of trailing hourly bars to fetch (covers 40-day crypto regime SMA + features)
YF_TICKER_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}  # internal symbol -> Yahoo ticker (stocks map to themselves)

# -- Decay tracker ------------------------------------------
DECAY_RATE = 0.98
INITIAL_ACCURACY = 0.5
TRACKER_PRIOR_WEIGHT = 4.0

# -- Feature engineering ------------------------------------
MIN_HISTORY_BARS = 70
RETURN_DEAD_ZONE = 0.003
NUM_SELECTED_FEATURES = 13