"""LightGBM 학습/추론 루틴."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

from .config import ModelConfig


def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.mean(np.abs(y_true))
    if denominator == 0:
        return 0.0
    return float(mean_absolute_error(y_true, y_pred) / denominator)


@dataclass(frozen=True)
class TrainResult:
    model: LGBMRegressor
    feature_columns: list[str]
    val_nmae: float
    val_mae: float


def train_lightgbm(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    valid_X: pd.DataFrame,
    valid_y: pd.Series,
    config: ModelConfig,
) -> TrainResult:
    params = {**config.lightgbm, "random_state": config.random_state}
    model = LGBMRegressor(**params)
    model.fit(
        train_X,
        train_y,
        eval_set=[(valid_X, valid_y)],
        eval_metric="mae",
        callbacks=[
            early_stopping(stopping_rounds=config.early_stopping_rounds),
            log_evaluation(0),
        ],
    )
    valid_pred = model.predict(valid_X)
    return TrainResult(
        model=model,
        feature_columns=list(train_X.columns),
        val_nmae=normalized_mae(valid_y.to_numpy(), valid_pred),
        val_mae=float(mean_absolute_error(valid_y.to_numpy(), valid_pred)),
    )


def save_model(result: TrainResult, path) -> None:
    payload = {
        "model": result.model,
        "feature_columns": result.feature_columns,
        "val_nmae": result.val_nmae,
        "val_mae": result.val_mae,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_model(path):
    return joblib.load(path)
