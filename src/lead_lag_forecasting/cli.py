"""CLI entry points.

```
python -m lead_lag_forecasting.cli discover --config configs/default.yaml
python -m lead_lag_forecasting.cli train    --config configs/default.yaml
python -m lead_lag_forecasting.cli infer    --config configs/default.yaml
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import PipelineConfig
from .pipeline import discover_pairs, infer, train


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lead-lag-forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (
        ("discover", "lead-lag 후보 발굴 + 그래프 점수 부여"),
        ("train", "follower 시계열에 leader 피처를 더한 LightGBM 학습"),
        ("infer", "예측 + 제출 파일 생성"),
    ):
        sub_parser = sub.add_parser(name, help=helptext)
        sub_parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    config = PipelineConfig.from_yaml(args.config)
    _setup_logging(config.output.log_path)

    if args.command == "discover":
        discover_pairs(config)
    elif args.command == "train":
        train(config)
    elif args.command == "infer":
        infer(config)


def discover_cli() -> None:
    config = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    _setup_logging(config.output.log_path)
    discover_pairs(config)


def train_cli() -> None:
    config = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    _setup_logging(config.output.log_path)
    train(config)


def infer_cli() -> None:
    config = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    _setup_logging(config.output.log_path)
    infer(config)


if __name__ == "__main__":
    main()
