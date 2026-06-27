from __future__ import annotations

import argparse
from pathlib import Path

from double_pendulum_lstm.config import AppConfig
from double_pendulum_lstm.manager import ExperimentManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM on experimental double-pendulum data and generate IB Extended Essay outputs."
    )
    parser.add_argument("--experiments", type=Path, default=Path("Experiments"), help="Root directory containing experiment folders.")
    parser.add_argument("--output", type=Path, default=Path("ML_Results"), help="Output directory for models, plots, reports, and predictions.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Device to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=None, help="Hidden sizes to search, e.g. 64 128 256.")
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=None, help="Sequence lengths to search, e.g. 25 50 100.")
    parser.add_argument("--dropouts", nargs="+", type=float, default=None, help="Dropout rates to search, e.g. 0.1 0.2 0.3.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--max-epochs", type=int, default=150, help="Maximum training epochs per hyperparameter run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0e-5, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    parser.add_argument("--min-delta", type=float, default=1.0e-5, help="Minimum validation-loss improvement for early stopping.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="ReduceLROnPlateau patience.")
    parser.add_argument("--forecast-steps", type=int, default=500, help="Maximum recursive forecast steps per test experiment.")
    parser.add_argument("--horizon-threshold-degrees", type=float, default=5.0, help="Angular error threshold for prediction horizon.")
    parser.add_argument("--plot-dpi", type=int, default=300, help="Plot DPI.")
    parser.add_argument("--limit-runs", type=int, default=None, help="Optional cap on hyperparameter runs for quick testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        experiments_dir=args.experiments,
        output_dir=args.output,
        device=args.device,
        seed=args.seed,
        hidden_sizes=args.hidden_sizes or [64, 128, 256],
        sequence_lengths=args.sequence_lengths or [25, 50, 100],
        dropouts=args.dropouts or [0.1, 0.2, 0.3],
        num_layers=args.num_layers,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        forecast_steps=args.forecast_steps,
        horizon_threshold_degrees=args.horizon_threshold_degrees,
        plot_dpi=args.plot_dpi,
        limit_runs=args.limit_runs,
    )
    manager = ExperimentManager(config)
    outputs = manager.run()
    print("Completed double-pendulum LSTM training.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
