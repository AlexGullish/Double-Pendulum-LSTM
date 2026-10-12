# Double-Pendulum LSTM

An LSTM deep learning application for modelling chaotic double-pendulum motion from experimental sensor data. Built as the analysis engine for an International Baccalaureate (IB) Extended Essay in Computer Science / Physics.

## Background

The double pendulum is a canonical chaotic system: small differences in initial conditions lead to exponentially diverging trajectories. This project trains a recurrent neural network (LSTM) on experimental time-series recordings and measures how far ahead the model can predict future angles and angular velocities before chaotic divergence exceeds a configurable threshold.

## Features

- Automated hyperparameter search over sequence length, hidden dimension, and dropout
- Recursive multi-step forecasting with angular error tracking
- Circular-statistics-aware evaluation for angle predictions (wrap-aware RMSE, MAE, R²)
- Publication-quality diagnostic plots: training curves, phase portraits, trajectory comparisons, error-vs-time, and prediction horizon distributions
- Device-aware training (CPU / CUDA / Apple MPS)
- Structured CLI with sensible defaults

## Installation

```bash
git clone https://github.com/AlexGullish/Double-Pendulum-LSTM.git
cd double-pendulum-lstm
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requirements:

- Python 3.9+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Data Format

Place experimental recordings in the `Experiments/` folder. Each subfolder must contain a `raw_data.csv` with the following columns:

| Column        | Description                          |
|---------------|--------------------------------------|
| `time`        | Timestamp (s)                        |
| `theta1_rad`  | Angle of the first mass (rad)        |
| `theta2_rad`  | Angle of the second mass (rad)       |
| `omega1`      | Angular velocity of the first mass   |
| `omega2`      | Angular velocity of the second mass  |

Optionally include a `dt` column (sampling interval). A `metadata.json` in the same folder is supported but optional.

## Usage

```bash
python main.py --help
```

Quick start:

```bash
python main.py
```

Key arguments:

```bash
python main.py \
  --experiments Experiments \
  --output ML_Results \
  --device auto \
  --seed 42 \
  --max-epochs 150 \
  --batch-size 64 \
  --forecast-steps 500 \
  --horizon-threshold-degrees 5.0
```

Hyperparameter search grids:

```bash
python main.py \
  --sequence-lengths 25 50 100 \
  --hidden-sizes 64 128 256 \
  --dropouts 0.1 0.2 0.3
```

## Project Structure

```
double-pendulum-lstm/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── double_pendulum_lstm/
│   ├── __init__.py                  # Package exports
│   ├── config.py                    # AppConfig dataclass and constants
│   ├── data.py                      # DatasetLoader and ExperimentData
│   ├── feature_engineering.py       # FeatureEngineer (sin/cos encoding)
│   ├── sequences.py                 # SequenceGenerator and train/val/test splitting
│   ├── model.py                     # LSTM architecture
│   ├── training.py                  # Trainer and TrainingRunResult
│   ├── evaluation.py                # Evaluator, metrics, recursive forecasting
│   ├── plotting.py                  # PlotGenerator
│   └── manager.py                   # ExperimentManager orchestrator
├── Experiments/                     # Input data (not tracked by Git)
└── ML_Results/                      # Generated outputs (not tracked by Git)
    ├── models/
    ├── plots/
    ├── predictions/
    ├── reports/
    └── diagnostics/
```

## Outputs

After a run, `ML_Results/` contains:

| File | Description |
|------|-------------|
| `best_lstm_model.pt` | PyTorch checkpoint with best hyperparameters |
| `scaler.pkl` | Pickled feature and target StandardScalers |
| `metrics.csv` | Per-target RMSE, MAE, R² on the test split |
| `prediction_horizons.csv` | Per-test-experiment horizon at the error threshold |
| `predictions.csv` | Full recursive forecast traces |
| `summary_report.txt` | Human-readable plain-text report |
| `plots/*.png` | Loss curves, phase portraits, trajectory and error plots |
| `diagnostics/*.csv` | Hyperparameter search history, training logs, dataset stats |

## Methodology

1. **Data loading** — CSVs are validated, cleaned, and assembled into `ExperimentData` objects.
2. **Feature engineering** — Angles are normalised to [-π, π] and encoded via `sin`/`cos`. Time-step `dt` is included as a feature.
3. **Sequencing** — Sliding windows of configurable length produce input sequences and single-step targets.
4. **Training** — Each hyperparameter combination trains a 2-layer LSTM with Adam, `MSELoss`, `ReduceLROnPlateau`, gradient clipping, and early stopping.
5. **Evaluation** — The best model is evaluated on held-out experiments. Recursive forecasting feeds predictions back into the window to extrapolate forward, while tracking cumulative angular error.
6. **Horizon analysis** — For each test experiment the first forecast step where angular error exceeds the threshold is recorded and summarised.
