"""Leader 정보를 더한 follower 의 lag/rolling 피처."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig, FeatureConfig


def build_leader_lookup(pairs: pd.DataFrame) -> tuple[dict, dict]:
    """follower → 가장 강한 leader / lag 매핑."""

    if pairs.empty:
        return {}, {}
    sorted_pairs = pairs.sort_values("graph_score", ascending=False)
    best = sorted_pairs.groupby("following_item_id").first().reset_index()
    follower_to_leader = dict(zip(best["following_item_id"], best["leading_item_id"]))
    follower_to_lag = dict(zip(best["following_item_id"], best["best_lag"].astype(int)))
    return follower_to_leader, follower_to_lag


def build_training_frame(
    monthly: pd.DataFrame,
    pivot: pd.DataFrame,
    pairs: pd.DataFrame,
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """item × month long 프레임에 lag / rolling / leader 피처를 추가."""

    follower_to_leader, follower_to_lag = build_leader_lookup(pairs)

    enriched: list[pd.DataFrame] = []
    for item, group in monthly.groupby(data_config.item_column):
        frame = group.copy().sort_values("date").set_index("date")
        for lag in feature_config.own_lags:
            frame[f"lag_{lag}"] = frame[data_config.value_column].shift(lag)
        shifted = frame[data_config.value_column].shift(1)
        for window in feature_config.rolling_windows:
            frame[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
            frame[f"rolling_std_{window}"] = shifted.rolling(window).std()

        if feature_config.use_leader_feature and item in follower_to_leader:
            leader_id = follower_to_leader[item]
            lag = int(follower_to_lag[item])
            if leader_id in pivot.columns:
                frame["leader_value"] = pivot[leader_id].shift(lag).reindex(frame.index)
                frame["leader_lag"] = float(lag)
            else:
                frame["leader_value"] = np.nan
                frame["leader_lag"] = 0.0
        else:
            frame["leader_value"] = np.nan
            frame["leader_lag"] = 0.0

        enriched.append(frame.reset_index())

    full = pd.concat(enriched, ignore_index=True)
    full["month"] = full["date"].dt.month
    full["quarter"] = full["date"].dt.quarter
    return full


def build_future_frame(
    pivot: pd.DataFrame,
    pairs: pd.DataFrame,
    forecast_date: pd.Timestamp,
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """예측 시점에 대해 동일 컬럼 셋을 가진 한 줄짜리 frame 을 item 별로 만든다."""

    follower_to_leader, follower_to_lag = build_leader_lookup(pairs)
    rows: list[dict] = []

    for item in pivot.columns:
        history = pivot[item]
        row: dict = {data_config.item_column: item, "date": forecast_date}
        for lag in feature_config.own_lags:
            target_date = forecast_date - pd.DateOffset(months=lag)
            row[f"lag_{lag}"] = float(history.get(target_date, 0.0))
        recent = history.loc[history.index < forecast_date]
        for window in feature_config.rolling_windows:
            tail = recent.tail(window)
            row[f"rolling_mean_{window}"] = float(tail.mean()) if not tail.empty else 0.0
            row[f"rolling_std_{window}"] = float(tail.std()) if len(tail) > 1 else 0.0

        if feature_config.use_leader_feature and item in follower_to_leader:
            leader_id = follower_to_leader[item]
            lag = int(follower_to_lag[item])
            target_date = forecast_date - pd.DateOffset(months=lag)
            if leader_id in pivot.columns and target_date in pivot.index:
                row["leader_value"] = float(pivot.at[target_date, leader_id])
                row["leader_lag"] = float(lag)
            else:
                row["leader_value"] = np.nan
                row["leader_lag"] = 0.0
        else:
            row["leader_value"] = np.nan
            row["leader_lag"] = 0.0

        row["month"] = int(forecast_date.month)
        row["quarter"] = int((forecast_date.month - 1) // 3 + 1)
        rows.append(row)

    return pd.DataFrame(rows)
