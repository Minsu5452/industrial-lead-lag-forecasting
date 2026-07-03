"""대회 csv 로딩 + monthly pivot 헬퍼."""

from __future__ import annotations

import pandas as pd

from .config import DataConfig


def load_train(config: DataConfig) -> pd.DataFrame:
    """학습 데이터 로드 후 long 형식으로 정렬해 반환."""

    frame = pd.read_csv(config.data_dir / config.train_file)
    keep_columns = [config.item_column, config.year_column, config.month_column, config.value_column]
    keep_columns += [c for c in config.optional_columns if c in frame.columns]
    frame = frame[keep_columns].copy()
    frame[config.year_column] = frame[config.year_column].astype(int)
    frame[config.month_column] = frame[config.month_column].astype(int)
    frame["date"] = pd.to_datetime(
        frame[config.year_column].astype(str) + "-" + frame[config.month_column].astype(str).str.zfill(2) + "-01"
    )
    frame = frame.sort_values([config.item_column, "date"]).reset_index(drop=True)
    return frame


def load_sample_submission(config: DataConfig) -> pd.DataFrame:
    return pd.read_csv(config.data_dir / config.sample_submission_file)


def to_monthly_pivot(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """`(date × item_id)` 매트릭스 (값=value) — 결측은 0 으로 채운다."""

    monthly = (
        frame.groupby([config.item_column, "date"], as_index=False)[config.value_column]
        .sum()
        .pivot(index="date", columns=config.item_column, values=config.value_column)
        .sort_index()
        .fillna(0)
    )
    monthly.columns.name = None
    return monthly
