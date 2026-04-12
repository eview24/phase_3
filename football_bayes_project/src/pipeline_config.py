"""
Shared run configuration for development and final Bayesian pipeline modes.

This module keeps the mode switch explicit and reproducible across the main
modeling script and the model-comparison script.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "dataset.ods"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
HOLDOUT_SEASON = "2024/25"
DEFAULT_MODE = "development"


@dataclass(frozen=True)
class RunConfig:
    mode: str
    draws: int
    tune: int
    chains: int
    target_accept: float
    random_seed: int
    progressbar: bool
    output_dir: Path
    holdout_season: str = HOLDOUT_SEASON
    coefficient_hdi_prob: float = 0.94
    prediction_hdi_prob: float = 0.94

    def to_serializable_dict(self):
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


MODE_SETTINGS = {
    "development": {
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.95,
        "random_seed": 42,
        "progressbar": True,
    },
    "final": {
        "draws": 2000,
        "tune": 2000,
        "chains": 4,
        "target_accept": 0.98,
        "random_seed": 42,
        "progressbar": True,
    },
}


def parse_mode_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_SETTINGS.keys()),
        default=DEFAULT_MODE,
        help="Run mode controlling sampling budget and output location.",
    )
    return parser.parse_args()


def build_run_config(mode):
    settings = MODE_SETTINGS[mode]
    output_dir = OUTPUT_ROOT / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunConfig(mode=mode, output_dir=output_dir, **settings)


def save_run_config(run_config, output_dir):
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config.to_serializable_dict(), f, indent=2)
