"""End-to-end orchestration.

세 entry point 를 노출한다.

- `discover_pairs` : lead-lag 후보를 발굴하고 그래프 점수를 부여해 parquet 으로 저장
- `train` : 후보 페어 + 자체 lag 피처 위에서 LightGBM 학습 후 모델 저장
- `infer` : 저장된 모델로 forecast 시점 한 달치 예측을 생성하고 sample submission
  형식의 csv 로 저장
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .features import build_future_frame, build_training_frame
from .graph import annotate_with_graph_score
from .io import load_sample_submission, load_train, to_monthly_pivot
from .leadlag import build_candidate_pairs
from .models import TrainResult, load_model, save_model, train_lightgbm

LOGGER = logging.getLogger("lead_lag_forecasting")

NON_FEATURE_COLUMNS = {
    "date",
    "year",
    "month",
    "quarter",
    "value",
    "weight",
    "quantity",
}


def _aggregate_monthly(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    keys = [config.data.item_column, "date"]
    aggregations = {config.data.value_column: "sum"}
    for column in config.data.optional_columns:
        if column in frame.columns:
            aggregations[column] = "sum"
    monthly = frame.groupby(keys, as_index=False).agg(aggregations)
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    return monthly


def discover_pairs(config: PipelineConfig) -> pd.DataFrame:
    LOGGER.info("loading train data...")
    train = load_train(config.data)
    monthly = _aggregate_monthly(train, config)
    pivot = to_monthly_pivot(monthly, config.data)
    LOGGER.info("monthly pivot shape=%s", pivot.shape)

    LOGGER.info("scanning lead-lag candidates...")
    candidates = build_candidate_pairs(
        pivot,
        min_overlap_months=config.leadlag.min_overlap_months,
        max_lag=config.leadlag.max_lag,
        min_abs_corr=config.leadlag.min_abs_corr,
        max_pvalue=config.leadlag.max_pvalue,
    )
    LOGGER.info("candidate pairs found: %d", len(candidates))

    annotated = annotate_with_graph_score(candidates, config.leadlag.graph)
    LOGGER.info("pairs after graph filtering: %d", len(annotated))

    if config.leadlag.top_pairs_per_follower > 0:
        annotated = (
            annotated.sort_values("graph_score", ascending=False)
            .groupby("following_item_id")
            .head(config.leadlag.top_pairs_per_follower)
            .reset_index(drop=True)
        )
        LOGGER.info("pairs after top-%d-per-follower trim: %d", config.leadlag.top_pairs_per_follower, len(annotated))

    config.output.pairs_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_parquet(config.output.pairs_path, index=False)
    LOGGER.info("saved pairs to %s", config.output.pairs_path)
    return annotated


def _split_train_valid(featured: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_dates = sorted(featured["date"].unique())
    if len(sorted_dates) <= config.target.validation_months:
        raise ValueError("not enough months for validation split")
    val_start = sorted_dates[-config.target.validation_months]
    return featured[featured["date"] < val_start], featured[featured["date"] >= val_start]


def train(config: PipelineConfig) -> TrainResult:
    LOGGER.info("loading data and pairs...")
    train_df = load_train(config.data)
    monthly = _aggregate_monthly(train_df, config)
    pivot = to_monthly_pivot(monthly, config.data)
    pairs = pd.read_parquet(config.output.pairs_path)

    LOGGER.info("building leader-aware feature frame...")
    featured = build_training_frame(monthly, pivot, pairs, config.data, config.features)
    featured = featured.dropna(subset=[f"lag_{config.features.own_lags[-1]}"]).reset_index(drop=True)

    train_part, valid_part = _split_train_valid(featured, config)
    feature_columns = sorted(
        column
        for column in featured.columns
        if column not in NON_FEATURE_COLUMNS
        and column != config.data.item_column
    )

    LOGGER.info("training LightGBM on %d rows (validating on %d)...", len(train_part), len(valid_part))
    result = train_lightgbm(
        train_part[feature_columns].fillna(0.0),
        train_part[config.data.value_column],
        valid_part[feature_columns].fillna(0.0),
        valid_part[config.data.value_column],
        config.model,
    )
    LOGGER.info("validation NMAE=%.4f, MAE=%.4f", result.val_nmae, result.val_mae)
    save_model(result, config.output.model_path)
    LOGGER.info("saved model to %s", config.output.model_path)
    return result


def infer(config: PipelineConfig) -> pd.DataFrame:
    LOGGER.info("loading data, pairs, and model...")
    train_df = load_train(config.data)
    monthly = _aggregate_monthly(train_df, config)
    pivot = to_monthly_pivot(monthly, config.data)
    pairs = pd.read_parquet(config.output.pairs_path)
    payload = load_model(config.output.model_path)
    model = payload["model"]
    feature_columns: list[str] = payload["feature_columns"]

    forecast_date = pd.Timestamp(year=config.target.forecast_year, month=config.target.forecast_month, day=1)
    LOGGER.info("building future frame for %s...", forecast_date.date())
    future = build_future_frame(pivot, pairs, forecast_date, config.data, config.features)
    predictions = model.predict(future[feature_columns].fillna(0.0))

    forecast_frame = pd.DataFrame(
        {
            config.data.item_column: future[config.data.item_column].to_numpy(),
            "value": np.clip(predictions, a_min=0.0, a_max=None),
        }
    )
    config.output.forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_frame.to_csv(config.output.forecast_path, index=False)
    LOGGER.info("saved item-level forecast to %s", config.output.forecast_path)

    sample_submission = load_sample_submission(config.data)
    submission = _format_submission(sample_submission, pairs, forecast_frame, config)
    config.output.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.output.submission_path, index=False)
    LOGGER.info("saved submission to %s", config.output.submission_path)
    return submission


def _format_submission(
    sample_submission: pd.DataFrame,
    pairs: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """sample_submission 형식이 (leader, follower, value) 인 경우 follower 예측을 매핑.

    sample_submission 에 이미 `leading_item_id`, `following_item_id` 컬럼이 있으면
    그대로 사용하고, 없으면 학습된 페어를 그대로 제출한다.
    """

    forecast_lookup = dict(zip(forecast_frame[config.data.item_column], forecast_frame["value"]))

    if {"leading_item_id", "following_item_id"}.issubset(sample_submission.columns):
        submission = sample_submission.copy()
    else:
        submission = pairs[["leading_item_id", "following_item_id"]].copy()

    submission["value"] = submission["following_item_id"].map(forecast_lookup).fillna(0.0)
    return submission
