"""편의 wrapper — `python scripts/infer.py --config configs/default.yaml`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_lag_forecasting.cli import _setup_logging  # noqa: E402
from lead_lag_forecasting.config import PipelineConfig  # noqa: E402
from lead_lag_forecasting.pipeline import infer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    config = PipelineConfig.from_yaml(args.config)
    _setup_logging(config.output.log_path)
    infer(config)


if __name__ == "__main__":
    main()
