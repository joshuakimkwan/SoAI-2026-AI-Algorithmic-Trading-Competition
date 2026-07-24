"""XGBoost model for mean-reversion signal prediction.

Same architecture as strategies/model.py (TradingModel), different objective:
instead of learning next-bar direction unconditionally, this model learns
P(bounce | price stretched below its 20d mean). Labels are non-zero ONLY when
the bar was a mean-reversion setup (dist_dsma20 <= -MR_DIST_THRESHOLD):

    +1  stretched below the mean AND the next bar rose  -> bounce (enter)
    -1  stretched below the mean AND the next bar fell  -> knife (avoid)
     0  everything else (no MR setup, or move inside the dead zone)

The strategy consults this model only while the failed-exit streak indicates a
choppy regime (see FAILED_STREAK_THRESHOLD in params.py), and only on symbols
currently stretched below their mean and not in a confirmed regime_down
breakdown. Exits are quick-bank (+MR_PROFIT_TARGET) or time-stop
(MR_TIME_STOP_HOURS), never the trend-following trailing/take-profit logic.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from strategies import params as P


class MeanReversionModel:
    """Wraps XGBoost for 3-class mean-reversion prediction: knife (-1), no-setup (0), bounce (+1)."""

    def __init__(self, symbol=None):
        self.model = None
        self.feature_history = []  # list of numpy arrays
        self.target_history = []   # list of ints (-1, 0, 1)
        self.last_val_accuracy = 0.0
        self.feature_names = None
        self.symbol = symbol
        self.selected_feature_indices = None

    def _build_model(self):
        """Create a fresh XGBoost classifier with params from config."""
        return xgb.XGBClassifier(
            n_estimators=P.N_ESTIMATORS,
            max_depth=P.MAX_DEPTH,
            learning_rate=P.LEARNING_RATE,
            subsample=P.SUBSAMPLE,
            colsample_bytree=P.COLSAMPLE_BYTREE,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )

    def add_sample(self, features: pd.Series, target: int):
        """Append one training sample (feature vector + target label)."""
        if self.feature_names is None:
            self.feature_names = list(features.index)
        self.feature_history.append(features.values.astype(float))
        self.target_history.append(target)

    def has_enough_data(self) -> bool:
        """Check if we have enough samples to train."""
        return len(self.feature_history) >= P.MIN_TRAINING_SAMPLES

    def train(self) -> bool:
        """Two-pass train: all features for importances, then top-K only.
        Identical procedure to TradingModel.train()."""
        if not self.has_enough_data():
            return False

        X = np.array(self.feature_history)
        y = np.array(self.target_history)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_mapped = y + 1

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y_mapped[:split], y_mapped[split:]

        if len(np.unique(y_train)) < 2:
            return False

        sample_weights = compute_sample_weight("balanced", y_train)

        # First pass: train on ALL features to determine importances
        model_full = self._build_model()
        model_full.fit(X_train, y_train, sample_weight=sample_weights)

        # Select top K features by importance
        importances = model_full.feature_importances_
        top_k = min(P.NUM_SELECTED_FEATURES, len(importances))
        top_indices = np.argsort(importances)[-top_k:]
        top_indices.sort()
        self.selected_feature_indices = top_indices

        # Second pass: retrain on selected features only
        X_train_sel = X_train[:, top_indices]
        X_val_sel = X_val[:, top_indices]

        model = self._build_model()
        model.fit(X_train_sel, y_train, sample_weight=sample_weights)

        val_preds = model.predict(X_val_sel)
        val_accuracy = (val_preds == y_val).mean()

        self.model = model
        self.last_val_accuracy = val_accuracy

        if self.feature_names is not None:
            selected_names = [self.feature_names[i] for i in top_indices]
            final_importances = model.feature_importances_
            ranked = sorted(zip(selected_names, final_importances), key=lambda x: x[1], reverse=True)
            imp_str = " | ".join(f"{n}={v:.4f}" for n, v in ranked[:10])
            print(f"[FEAT_IMP] [MR {self.symbol}] top: {imp_str}")

        return True

    def predict(self, features: pd.Series) -> tuple:
        """
        Predict MR action and confidence for given features.

        Returns:
            (direction, confidence) where:
            - direction: -1 (knife, avoid), 0 (no setup), or 1 (bounce, enter)
            - confidence: probability of predicted class (0.33 to 1.0)
        """
        if self.model is None:
            return 0, 0.0

        X = np.nan_to_num(features.values.astype(float).reshape(1, -1),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if self.selected_feature_indices is not None:
            X = X[:, self.selected_feature_indices]
        probs = self.model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        direction = predicted_class - 1  # Map back: 0->-1, 1->0, 2->1

        return direction, confidence

    @staticmethod
    def compute_target(current_price: float, next_price: float, dist_from_mean: float) -> int:
        """
        Mean-reversion training target.

        +1 = stretched below mean AND bounced; -1 = stretched below AND kept
        falling; 0 = everything else (no setup, or move inside the dead zone).
        `dist_from_mean` is the dist_dsma20 feature at the time the features
        were computed (negative = price below its 20d daily SMA).
        """
        if current_price <= 0 or dist_from_mean >= -P.MR_DIST_THRESHOLD:
            return 0  # not stretched below the mean -> not an MR setup
        ret = (next_price - current_price) / current_price
        if ret > P.RETURN_DEAD_ZONE:
            return 1
        elif ret < -P.RETURN_DEAD_ZONE:
            return -1
        return 0
