"""Adaptive decay tracker and position sizing for the ML trading strategy."""

import numpy as np
from strategies import params as P


class DecayTracker:
    """
    Tracks per-asset prediction accuracy with exponential decay weighting.

    Recent predictions matter more than old ones.
    Used to adjust position sizes: assets where the model has been accurate
    recently get larger allocations.
    """

    def __init__(self, symbols: list):
        self.symbols = symbols
        self.predictions = {s: [] for s in symbols}
        self.accuracy_scores = {s: P.INITIAL_ACCURACY for s in symbols}

    def update(self, symbol: str, was_correct: bool):
        """Record whether the last prediction for this asset was correct."""
        if symbol not in self.predictions:
            self.predictions[symbol] = []
            self.accuracy_scores[symbol] = P.INITIAL_ACCURACY
        self.predictions[symbol].append(was_correct)
        self._recompute(symbol)

    def _recompute(self, symbol: str):
        """Recompute accuracy score using exponential decay weights."""
        preds = self.predictions[symbol]
        if not preds:
            return
        n = len(preds)
        weights = np.array([P.DECAY_RATE ** (n - 1 - i) for i in range(n)])
        values = np.array([float(p) for p in preds])

        prior_weight = P.TRACKER_PRIOR_WEIGHT
        total_weight = np.sum(weights) + prior_weight
        blended = np.sum(weights * values) + prior_weight * P.INITIAL_ACCURACY
        if total_weight > 0:
            self.accuracy_scores[symbol] = float(blended / total_weight)

    def get_score(self, symbol: str) -> float:
        """Get current accuracy score for an asset (0.0 to 1.0)."""
        return self.accuracy_scores.get(symbol, P.INITIAL_ACCURACY)


def compute_position_sizes(
    signals: dict,
    decay_tracker: DecayTracker,
    portfolio_value: float,
    current_drawdown: float,
) -> dict:
    """
    Compute target portfolio weights from model signals and tracker scores.

    Args:
        signals: {symbol: (direction, confidence)} from model predictions
        decay_tracker: DecayTracker instance with per-asset accuracy scores
        portfolio_value: current total portfolio value
        current_drawdown: current drawdown as fraction (0.0 to 1.0)

    Returns:
        {symbol: target_weight} where weight is fraction of portfolio (positive = long)
    """
    if current_drawdown >= P.MAX_DRAWDOWN:
        return {}

    # Compute adjusted signals
    adjusted = {}
    for symbol, (direction, confidence) in signals.items():
        if direction == 0:
            continue
        accuracy = decay_tracker.get_score(symbol)
        adj_signal = confidence * accuracy * direction
        min_conf = P.MIN_CONFIDENCE_CRYPTO if symbol in P.CRYPTO_SYMBOLS else P.MIN_CONFIDENCE
        if abs(adj_signal) > min_conf:
            adjusted[symbol] = adj_signal

    if not adjusted:
        return {}

    # Rank and select top N
    ranked = sorted(adjusted.items(), key=lambda x: abs(x[1]), reverse=True)
    selected = ranked[:P.MAX_POSITIONS]

    # Compute weights proportional to absolute signal strength
    total_signal = sum(abs(s) for _, s in selected)
    if total_signal == 0:
        return {}

    available = 1.0 - P.CASH_BUFFER
    weights = {}

    for symbol, signal in selected:
        weight = (abs(signal) / total_signal) * available
        weight = min(weight, P.MAX_WEIGHT_PER_POSITION)
        # Keep sign: positive = long, negative = short
        weights[symbol] = weight * np.sign(signal)

    # Drawdown scaling: linear reduction from DRAWDOWN_SCALING_START to MAX_DRAWDOWN
    if current_drawdown > P.DRAWDOWN_SCALING_START:
        scale = (P.MAX_DRAWDOWN - current_drawdown) / (P.MAX_DRAWDOWN - P.DRAWDOWN_SCALING_START)
        scale = max(scale, 0.0)
        weights = {s: w * scale for s, w in weights.items()}

    return weights
