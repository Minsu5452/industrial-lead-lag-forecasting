"""파이프라인 설정 dataclass.

`configs/default.yaml` 을 그대로 dataclass 로 매핑한다. CLI 에서 `--config` 로
다른 yaml 을 넘기면 동일 인터페이스로 동작한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("data")
    train_file: str = "train.csv"
    sample_submission_file: str = "sample_submission.csv"
    item_column: str = "item_id"
    year_column: str = "year"
    month_column: str = "month"
    value_column: str = "value"
    optional_columns: tuple[str, ...] = ("weight", "quantity")


@dataclass(frozen=True)
class TargetConfig:
    forecast_year: int = 2025
    forecast_month: int = 8
    validation_months: int = 3


@dataclass(frozen=True)
class GraphConfig:
    score_threshold: float = 50.0
    community_resolution: float = 1.0


@dataclass(frozen=True)
class LeadLagConfig:
    min_overlap_months: int = 18
    max_lag: int = 9
    min_abs_corr: float = 0.30
    max_pvalue: float = 0.10
    top_pairs_per_follower: int = 3
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass(frozen=True)
class FeatureConfig:
    own_lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    rolling_windows: tuple[int, ...] = (3, 6)
    use_leader_feature: bool = True


@dataclass(frozen=True)
class ModelConfig:
    lightgbm: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 100
    random_state: int = 42


@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: Path = Path("artifacts")
    pairs_path: Path = Path("artifacts/leadlag_pairs.parquet")
    forecast_path: Path = Path("artifacts/forecast_2025_08.csv")
    submission_path: Path = Path("submissions/submission.csv")
    model_path: Path = Path("models/lightgbm.joblib")
    log_path: Path = Path("logs/run.log")


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    leadlag: LeadLagConfig = field(default_factory=LeadLagConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        data = DataConfig(
            data_dir=Path(raw["data"]["data_dir"]),
            train_file=raw["data"]["train_file"],
            sample_submission_file=raw["data"]["sample_submission_file"],
            item_column=raw["data"]["item_column"],
            year_column=raw["data"]["year_column"],
            month_column=raw["data"]["month_column"],
            value_column=raw["data"]["value_column"],
            optional_columns=tuple(raw["data"].get("optional_columns", []) or []),
        )
        target = TargetConfig(
            forecast_year=int(raw["target"]["forecast_year"]),
            forecast_month=int(raw["target"]["forecast_month"]),
            validation_months=int(raw["target"]["validation_months"]),
        )
        leadlag = LeadLagConfig(
            min_overlap_months=int(raw["leadlag"]["min_overlap_months"]),
            max_lag=int(raw["leadlag"]["max_lag"]),
            min_abs_corr=float(raw["leadlag"]["min_abs_corr"]),
            max_pvalue=float(raw["leadlag"]["max_pvalue"]),
            top_pairs_per_follower=int(raw["leadlag"]["top_pairs_per_follower"]),
            graph=GraphConfig(
                score_threshold=float(raw["leadlag"]["graph"]["score_threshold"]),
                community_resolution=float(raw["leadlag"]["graph"]["community_resolution"]),
            ),
        )
        features = FeatureConfig(
            own_lags=tuple(raw["features"]["own_lags"]),
            rolling_windows=tuple(raw["features"]["rolling_windows"]),
            use_leader_feature=bool(raw["features"]["use_leader_feature"]),
        )
        model = ModelConfig(
            lightgbm=dict(raw["model"]["lightgbm"]),
            early_stopping_rounds=int(raw["model"]["early_stopping_rounds"]),
            random_state=int(raw["model"]["random_state"]),
        )
        output = OutputConfig(
            artifacts_dir=Path(raw["output"]["artifacts_dir"]),
            pairs_path=Path(raw["output"]["pairs_path"]),
            forecast_path=Path(raw["output"]["forecast_path"]),
            submission_path=Path(raw["output"]["submission_path"]),
            model_path=Path(raw["output"]["model_path"]),
            log_path=Path(raw["output"]["log_path"]),
        )
        return cls(data=data, target=target, leadlag=leadlag, features=features, model=model, output=output)
