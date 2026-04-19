"""
Shared configuration for the elastic net regression pipeline.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT.parent / "dataset.ods"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
HOLDOUT_SEASON = "2024/25"
VAL_SEASON = "2023/24"
DEFAULT_MODE = "development"


@dataclass(frozen=True)
class RunConfig:
    mode: str
    output_dir: Path
    holdout_season: str = HOLDOUT_SEASON
    val_season: str = VAL_SEASON
    random_seed: int = 42
    n_trials: int = 100

    def to_serializable_dict(self):
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


MODE_SETTINGS = {
    "development": {
        "random_seed": 42,
        "n_trials": 50,
    },
    "final": {
        "random_seed": 42,
        "n_trials": 200,
    },
}


def parse_mode_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_SETTINGS.keys()),
        default=DEFAULT_MODE,
        help="Run mode controlling output location and number of Optuna trials.",
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
