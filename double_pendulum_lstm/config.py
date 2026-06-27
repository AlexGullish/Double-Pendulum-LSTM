from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REQUIRED_COLUMNS = ["time", "theta1_rad", "theta2_rad", "omega1", "omega2"]
FEATURE_COLUMNS = ["dt", "sin_theta1", "cos_theta1", "sin_theta2", "cos_theta2", "omega1", "omega2"]
TARGET_COLUMNS = ["theta1", "theta2", "omega1", "omega2"]
DEFAULT_HIDDEN_SIZES = [64, 128, 256]
DEFAULT_SEQUENCE_LENGTHS = [25, 50, 100]
DEFAULT_DROPOUTS = [0.1, 0.2, 0.3]
DEFAULT_FORECAST_REPORT_STEPS = [1, 10, 25, 50, 100, 500]


@dataclass
class AppConfig:
    experiments_dir: Path = Path("Experiments")
    output_dir: Path = Path("ML_Results")
    seed: int = 42
    device: str = "auto"
    hidden_sizes: List[int] = field(default_factory=lambda: DEFAULT_HIDDEN_SIZES.copy())
    sequence_lengths: List[int] = field(default_factory=lambda: DEFAULT_SEQUENCE_LENGTHS.copy())
    dropouts: List[float] = field(default_factory=lambda: DEFAULT_DROPOUTS.copy())
    num_layers: int = 2
    default_sequence_length: int = 50
    max_epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 20
    min_delta: float = 1.0e-5
    grad_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    forecast_steps: int = 500
    forecast_report_steps: List[int] = field(default_factory=lambda: DEFAULT_FORECAST_REPORT_STEPS.copy())
    horizon_threshold_degrees: float = 5.0
    plot_dpi: int = 300
    limit_runs: Optional[int] = None

    def resolved_device(self) -> str:
        if self.device == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if self.device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS was requested but is not available.")
            return "mps"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return self.device

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["experiments_dir"] = str(self.experiments_dir)
        data["output_dir"] = str(self.output_dir)
        return data

    def ensure_directories(self) -> Dict[str, Path]:
        dirs = {
            "root": self.output_dir,
            "models": self.output_dir / "models",
            "plots": self.output_dir / "plots",
            "reports": self.output_dir / "reports",
            "predictions": self.output_dir / "predictions",
            "diagnostics": self.output_dir / "diagnostics",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs
