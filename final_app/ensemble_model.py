# ==============================
# Author: Nathan Oliver
# ==============================
# Inference-only module for the California housing price stacked ensemble.
# Training code is not included. Load pre-trained artifacts with
# DeployableStackedEnsemble.load_native(artifacts_dir).
# ==============================

import gzip
import os

import numpy as np
import pandas as pd

import joblib
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import lightgbm as lgb
from xgboost import XGBRegressor


# ── Shared preprocessing transformers ─────────────────────────────────────────

class LevelsMultiLabelEncoder(BaseEstimator, TransformerMixin):
    LEVEL_CATEGORIES = ["One", "Two", "ThreeOrMore", "MultiSplit"]

    def __init__(self, column="Levels"):
        self.column = column

    def _parse_levels(self, value):
        if pd.isna(value) or str(value).lower() == "nan":
            return set()
        return {level.strip() for level in str(value).split(",") if level.strip()}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        levels = X[self.column] if isinstance(X, pd.DataFrame) else pd.Series(X.ravel())
        result_cols = []
        for level in self.LEVEL_CATEGORIES:
            indicator = levels.apply(
                lambda value, target=level: 1 if target in self._parse_levels(value) else 0
            ).fillna(0).astype(int).values
            result_cols.append(indicator)
        return np.column_stack(result_cols).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["is_OneStory", "is_TwoStory", "is_ThreeOrMoreStory", "is_MultiSplit"])


class DistributionImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        self.fill_values_ = {col: np.array(X[col].dropna()) for col in X.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("object")
            elif hasattr(X[col].dtype, "numpy_dtype"):
                X[col] = X[col].astype("float64")
        rng = np.random.default_rng(self.random_state)
        for col in X.columns:
            mask = X[col].isna()
            if mask.sum() > 0:
                fill_values = self.fill_values_[col]
                if fill_values.size == 0:
                    fill_values = np.array([0.0])
                X.loc[mask, col] = rng.choice(fill_values, size=mask.sum())
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=2026):
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "LivingArea" in X.columns:
            X["LivingArea_sq"] = X["LivingArea"] ** 2
        if "YearBuilt" in X.columns:
            X["YearBuilt_sq"] = X["YearBuilt"] ** 2
        if "LotSizeSquareFeet" in X.columns:
            X["LotSizeSquareFeet_sq"] = X["LotSizeSquareFeet"] ** 2
        if "LivingArea" in X.columns and "YearBuilt" in X.columns:
            X["LivingArea_x_YearBuilt"] = X["LivingArea"] * X["YearBuilt"]
        if "LivingArea" in X.columns and "BathroomsTotalInteger" in X.columns:
            X["LivingArea_x_Bathrooms"] = X["LivingArea"] * X["BathroomsTotalInteger"]
        if "LivingArea" in X.columns and "BedroomsTotal" in X.columns:
            X["SqFt_per_Bedroom"] = X["LivingArea"] / X["BedroomsTotal"].replace(0, 1).fillna(1)
        if "LivingArea" in X.columns and "BathroomsTotalInteger" in X.columns:
            X["SqFt_per_Bathroom"] = X["LivingArea"] / X["BathroomsTotalInteger"].replace(0, 1).fillna(1)
        if "BathroomsTotalInteger" in X.columns and "BedroomsTotal" in X.columns:
            X["Bath_to_Bed_Ratio"] = X["BathroomsTotalInteger"] / X["BedroomsTotal"].replace(0, 1).fillna(1)
        if "YearBuilt" in X.columns:
            X["Age"] = self.current_year - X["YearBuilt"]
            X["Age_sq"] = X["Age"] ** 2
        return X


class StackingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, postal_col="PostalCode", levels_col="Levels", random_state=42, current_year=2026):
        self.postal_col = postal_col
        self.levels_col = levels_col
        self.random_state = random_state
        self.current_year = current_year

    def transform(self, X):
        X_work = self._prepare_frame(X)
        X_work = self.feature_engineer_.transform(X_work)
        transformed = self._apply_non_ohe_steps(X_work)
        transformed = self._apply_one_hot(transformed)
        transformed = transformed.reindex(columns=self.feature_columns_, fill_value=0.0)
        imputed = self.imputer_.transform(transformed)
        scaled = self.scaler_.transform(imputed)
        return pd.DataFrame(scaled, columns=self.feature_columns_, index=transformed.index)

    def _prepare_frame(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        ordered_cols = self.raw_columns_ if hasattr(self, "raw_columns_") else X.columns.tolist()
        for col in ordered_cols:
            if col not in X.columns:
                X[col] = np.nan
        return X[ordered_cols].copy()

    def _apply_non_ohe_steps(self, X):
        X = X.copy()
        if self.postal_col in X.columns:
            if self.target_encoder_ is not None:
                X[self.postal_col + "_encoded"] = self.target_encoder_.transform(
                    X[[self.postal_col]].astype(str)
                ).flatten()
            else:
                X[self.postal_col + "_encoded"] = np.nan
            X = X.drop(columns=[self.postal_col])
        if self.levels_col in X.columns:
            if self.levels_encoder_ is not None:
                levels = self.levels_encoder_.transform(X)
                for i, name in enumerate(self.levels_encoder_.get_feature_names_out().tolist()):
                    X[name] = levels[:, i]
            X = X.drop(columns=[self.levels_col])
        return X

    def _apply_one_hot(self, X):
        X = X.copy()
        if self.onehot_encoder_ is None:
            return X.astype(float)
        base = X.drop(columns=self.categorical_columns_).copy()
        encoded = self.onehot_encoder_.transform(X[self.categorical_columns_].astype(str))
        encoded_cols = self.onehot_encoder_.get_feature_names_out(self.categorical_columns_)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
        return pd.concat([base, encoded_df], axis=1).astype(float)


# ── Native loader helpers ──────────────────────────────────────────────────────

class _NativeLGBMWrapper:
    """Wraps lgb.Booster to match LGBMRegressor.predict() interface."""

    def __init__(self, booster):
        self._booster = booster

    def predict(self, X, **kwargs):
        return self._booster.predict(X)


# ── Base model ─────────────────────────────────────────────────────────────────

class BaseHousingModel(BaseEstimator, RegressorMixin):
    def __init__(self, current_year=2026, random_state=42):
        self.current_year = current_year
        self.random_state = random_state

    def predict(self, X):
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model not fitted.")
        X_processed = self.preprocessor_.transform(X)
        return self._predict_processed(X_processed)

    def _predict_processed(self, X_processed):
        raise NotImplementedError


# ── Stacked ensemble ───────────────────────────────────────────────────────────

class DeployableStackedEnsemble(BaseHousingModel):
    """
    Stacked ensemble: XGBoost + LightGBM base models with GBR meta-model.
    Load pre-trained weights with load_native(artifacts_dir).
    """

    def __init__(self, meta_n_estimators=100, meta_max_depth=3, meta_learning_rate=0.05,
                 n_folds=5, random_state=42, device="cpu", current_year=2026):
        super().__init__(current_year=current_year, random_state=random_state)
        self.meta_n_estimators  = meta_n_estimators
        self.meta_max_depth     = meta_max_depth
        self.meta_learning_rate = meta_learning_rate
        self.n_folds            = n_folds
        self.device             = device
        self.meta_feature_names_ = ["XGB_pred", "LGBM_pred", "abs_diff", "avg_pred"]

    def _build_meta_features(self, xgb_preds, lgbm_preds):
        abs_diff = np.abs(xgb_preds - lgbm_preds)
        avg_pred = (xgb_preds + lgbm_preds) / 2
        return np.column_stack([xgb_preds, lgbm_preds, abs_diff, avg_pred])

    def _predict_processed(self, X_processed):
        xgb_preds  = np.mean([m.predict(X_processed) for m in self.xgb_models_], axis=0)
        lgbm_preds = np.mean([m.predict(X_processed) for m in self.lgbm_models_], axis=0)
        meta_features = self._build_meta_features(xgb_preds, lgbm_preds)
        return np.expm1(self.meta_model_.predict(meta_features))

    def predict_base_models(self, X):
        """Return (xgb_prices, lgbm_prices) as separate numpy arrays."""
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model not fitted.")
        X_processed = self.preprocessor_.transform(X)
        xgb_preds  = np.mean([m.predict(X_processed) for m in self.xgb_models_], axis=0)
        lgbm_preds = np.mean([m.predict(X_processed) for m in self.lgbm_models_], axis=0)
        return np.expm1(xgb_preds), np.expm1(lgbm_preds)

    @classmethod
    def load_native(cls, artifacts_dir):
        """Load from native format files produced by export_native.py."""
        inst = cls.__new__(cls)
        inst.is_fitted_          = True
        inst.meta_feature_names_ = ["XGB_pred", "LGBM_pred", "abs_diff", "avg_pred"]
        inst.preprocessor_       = joblib.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
        inst.meta_model_         = joblib.load(os.path.join(artifacts_dir, "meta_model.joblib"))

        inst.xgb_models_ = []
        i = 0
        while True:
            gz = os.path.join(artifacts_dir, f"xgb_fold_{i}.ubj.gz")
            if not os.path.exists(gz):
                break
            xgb = XGBRegressor()
            with gzip.open(gz, "rb") as f:
                xgb.load_model(bytearray(f.read()))
            inst.xgb_models_.append(xgb)
            i += 1

        inst.lgbm_models_ = []
        i = 0
        while True:
            gz = os.path.join(artifacts_dir, f"lgbm_fold_{i}.txt.gz")
            if not os.path.exists(gz):
                break
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                content = f.read()
            inst.lgbm_models_.append(_NativeLGBMWrapper(lgb.Booster(model_str=content)))
            i += 1

        return inst

    def get_artifact_metadata(self):
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model not fitted.")
        return {
            "n_folds":   self.n_folds,
            "device":    self.device,
            "feature_columns":        list(self.preprocessor_.feature_columns_),
            "meta_feature_names":     list(self.meta_feature_names_),
            "meta_feature_importances": {
                name: float(value)
                for name, value in zip(
                    self.meta_feature_names_, self.meta_model_.feature_importances_
                )
            },
        }
