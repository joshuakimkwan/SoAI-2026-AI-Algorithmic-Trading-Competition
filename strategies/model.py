"""XGBoost model for trading signal prediction."""

import os
import gzip
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from strategies import params as P


class TradingModel:
    """Wraps XGBoost for 3-class direction prediction: sell (-1), hold (0), buy (+1)."""

    def __init__(self, symbol=None): # CHANGED (18062026)
        self.model = None
        self.feature_history = []  # list of numpy arrays
        self.target_history = []   # list of ints (-1, 0, 1)
        self.last_val_accuracy = 0.0
        self.feature_names = None
        self.symbol = symbol # CHANGED (18062026)
        self.selected_feature_indices = None # CHANGED (18062026)

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

    # def train(self) -> bool:
    #     """
    #     Train (or retrain) the model on all collected data.

    #     Uses 80/20 train/val split. Applies balanced class weights.
    #     Safety gate: only replaces current model if validation accuracy improves.

    #     Returns True if model was updated, False if previous model was kept.
    #     """
    #     if not self.has_enough_data():
    #         return False

    #     X = np.array(self.feature_history)
    #     y = np.array(self.target_history)

    #     # Replace any NaN/inf in features with 0
    #     X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    #     # Map labels: -1 -> 0, 0 -> 1, 1 -> 2 (XGBoost needs 0-indexed classes)
    #     y_mapped = y + 1

    #     split = int(len(X) * 0.8)
    #     X_train, X_val = X[:split], X[split:]
    #     y_train, y_val = y_mapped[:split], y_mapped[split:]

    #     # Need at least 2 classes in training set
    #     if len(np.unique(y_train)) < 2:
    #         return False

    #     sample_weights = compute_sample_weight("balanced", y_train)

    #     model = self._build_model()
    #     model.fit(X_train, y_train, sample_weight=sample_weights)

    #     val_preds = model.predict(X_val)
    #     val_accuracy = (val_preds == y_val).mean()

    #     # Safety gate: only update if improvement (or first model)
    #     # CHANGED REMOVED (15062026) if self.model is None or val_accuracy >= self.last_val_accuracy: to allow retraining.
    #     self.model = model
    #     self.last_val_accuracy = val_accuracy
    #     self._temp_feat_imp_ranking() # TO BE DELETD AFTER CHECKING TOP FEATURES

    #     if self.feature_names is not None:
    #         importances = self.model.feature_importances_
    #         ranked = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
    #         top = ranked[:10]
    #         imp_str = " | ".join(f"{n}={v:.4f}" for n, v in top)
    #         print(f"[FEAT_IMP] top10: {imp_str}")
    #     return True

    #     # return False # CHANGED REMOVED (15062026)
    def train(self) -> bool: # CHANGED (18062026) from above
        if not self.has_enough_data():
            return False

        # X = np.array(self.feature_history)
        # y = np.array(self.target_history)
        # Cap to the most recent MAX_TRAIN_SAMPLES rows to bound retrain time
        # (crypto accumulates ~11k samples on the 24/7 grid; an uncapped fit risks
        # exceeding the competition's per-iteration time limit).
        if len(self.feature_history) > P.MAX_TRAIN_SAMPLES:
            fh = self.feature_history[-P.MAX_TRAIN_SAMPLES:]
            th = self.target_history[-P.MAX_TRAIN_SAMPLES:]
        else:
            fh = self.feature_history
            th = self.target_history
        X = np.array(fh)
        y = np.array(th)
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

        # Log selected feature importances
        if self.feature_names is not None:
            selected_names = [self.feature_names[i] for i in top_indices]
            final_importances = model.feature_importances_
            ranked = sorted(zip(selected_names, final_importances), key=lambda x: x[1], reverse=True)
            imp_str = " | ".join(f"{n}={v:.4f}" for n, v in ranked[:10])
            print(f"[FEAT_IMP] [{self.symbol}] top: {imp_str}")

        # self._temp_feat_imp_ranking() REMOVED 23062026, parameters tuned using optuna.
        return True

    # def _temp_feat_imp_ranking(self): # TO BE DELETD AFTER CHECKING TOP FEATURES
    #     if not hasattr(self, '_feat_imp_records'): # CHANGED (18062026)
    #         self._feat_imp_records = []
    #     importances = self.model.feature_importances_
    #     paired = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
    #     top15 = dict(paired[:15])
    #     self._feat_imp_records.append(top15)
    #     df = pd.DataFrame(self._feat_imp_records)
    #     df.index = range(1, len(df) + 1)
    #     df.index.name = "retrain"
    #     df.to_csv("feat_imp_ranking.csv")
    # def _temp_feat_imp_ranking(self): # CHANGED (18062026) from above REMOVED 23062026, parameters tuned using optuna.
    #     if not hasattr(self, '_feat_imp_records'):
    #         self._feat_imp_records = []
    #     if self.selected_feature_indices is None or self.feature_names is None:
    #         return
    #     selected_names = [self.feature_names[i] for i in self.selected_feature_indices]
    #     importances = self.model.feature_importances_
    #     record = dict(zip(selected_names, importances))
    #     self._feat_imp_records.append(record)
    #     df = pd.DataFrame(self._feat_imp_records)
    #     df.index = range(1, len(df) + 1)
    #     df.index.name = "retrain"
    #     df.to_csv(f"feat_imp_ranking_{self.symbol}.csv")

    def predict(self, features: pd.Series) -> tuple:
        """
        Predict direction and confidence for given features.

        Returns:
            (direction, confidence) where:
            - direction: -1 (sell), 0 (hold), or 1 (buy)
            - confidence: probability of predicted class (0.33 to 1.0)
        """
        if self.model is None:
            return 0, 0.0

        X = np.nan_to_num(features.values.astype(float).reshape(1, -1),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if self.selected_feature_indices is not None: # CHANGED (18062026)
            X = X[:, self.selected_feature_indices] # CHANGED (18062026)
        probs = self.model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        direction = predicted_class - 1  # Map back: 0->-1, 1->0, 2->1

        return direction, confidence

    @staticmethod
    def compute_target(current_price: float, next_price: float) -> int:
        """
        Compute training target from price movement.
        Uses dead zone to filter noise: returns within +/-0.1% are labeled hold (0).
        """
        if current_price <= 0:
            return 0
        ret = (next_price - current_price) / current_price
        if ret > P.RETURN_DEAD_ZONE:
            return 1
        elif ret < -P.RETURN_DEAD_ZONE:
            return -1
        return 0

    def save(self, dir_path):
        """Persist the fitted booster + training state to dir_path/<symbol>.* so the
        model can be loaded at competition startup with no retraining (per the
        organizers' 'upload saved weights' guidance). Native XGBoost format, not
        pickle, so it survives library-version differences in their environment."""
        os.makedirs(dir_path, exist_ok=True)
        prefix = os.path.join(dir_path, str(self.symbol))
        if self.model is not None:
            self.model.save_model(prefix + ".ubj")
            with open(prefix + ".ubj", "rb") as fh_in:
                with gzip.open(prefix + ".ubj.gz", "wb", compresslevel=6) as fh_out:
                    fh_out.write(fh_in.read())
            os.remove(prefix + ".ubj")  # keep only the compressed copy (~40% of the size)
        keep = P.MAX_TRAIN_SAMPLES
        fh = self.feature_history[-keep:] if len(self.feature_history) > keep else self.feature_history
        th = self.target_history[-keep:] if len(self.target_history) > keep else self.target_history
        np.savez_compressed(
            prefix + "_meta.npz",
            feature_history=(np.array(fh, dtype=float) if fh else np.empty((0, 0))),
            target_history=np.array(th, dtype=int),
            selected_feature_indices=(self.selected_feature_indices
                                      if self.selected_feature_indices is not None else np.array([], dtype=int)),
            feature_names=np.array(self.feature_names if self.feature_names else [], dtype=object),
            last_val_accuracy=np.array([self.last_val_accuracy], dtype=float),
            has_model=np.array([1 if self.model is not None else 0], dtype=int),
        )

    def load(self, dir_path):
        """Restore booster + training state saved by save(). Returns self so the
        caller can do self.models[sym] = TradingModel(sym).load(dir)."""
        prefix = os.path.join(dir_path, str(self.symbol))
        meta = np.load(prefix + "_meta.npz", allow_pickle=True)
        if int(meta["has_model"][0]) == 1:
            self.model = self._build_model()
            if os.path.exists(prefix + ".ubj.gz"):
                # decompress to a temp file (XGBoost loads from a path), then clean up
                with gzip.open(prefix + ".ubj.gz", "rb") as fh:
                    raw = fh.read()
                tmp = tempfile.NamedTemporaryFile(suffix=".ubj", delete=False)
                try:
                    tmp.write(raw)
                    tmp.close()
                    self.model.load_model(tmp.name)
                finally:
                    os.unlink(tmp.name)
            elif os.path.exists(prefix + ".ubj"):
                self.model.load_model(prefix + ".ubj")
            else:
                self.model = None
        fh = meta["feature_history"]
        self.feature_history = [np.asarray(r, dtype=float) for r in fh] if fh.size else []
        self.target_history = [int(v) for v in meta["target_history"]]
        sfi = meta["selected_feature_indices"]
        self.selected_feature_indices = sfi.astype(int) if sfi.size else None
        fn = meta["feature_names"]
        self.feature_names = [str(v) for v in fn] if fn.size else None
        self.last_val_accuracy = float(meta["last_val_accuracy"][0])
        return self