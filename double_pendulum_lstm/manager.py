from __future__ import annotations

import pickle
import shutil
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import AppConfig, FEATURE_COLUMNS, TARGET_COLUMNS
from .data import DatasetLoader
from .evaluation import Evaluator
from .feature_engineering import FeatureEngineer
from .model import LSTMModel
from .plotting import PlotGenerator
from .sequences import SequenceGenerator
from .training import Trainer, TrainingRunResult


class ExperimentManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.directories: Dict[str, Path] = {}

    def run(self) -> Dict[str, Path]:
        self.directories = self.config.ensure_directories()
        experiments = DatasetLoader(self.config.experiments_dir).load_all()
        if not experiments:
            raise RuntimeError(f"No valid experiments with required columns were found in {self.config.experiments_dir}.")

        engineered = FeatureEngineer().engineer_all(experiments)
        engineered_frames = {experiment.experiment_id: frame for experiment, frame in zip(experiments, engineered.values())}
        experiment_ids = list(engineered_frames.keys())
        split_mode, train_ids, val_ids, test_ids = self._split_experiment_ids(experiment_ids)

        generator = SequenceGenerator()
        trainer = Trainer(self.config)
        history_rows: List[Dict[str, Any]] = []
        long_history_rows: List[Dict[str, Any]] = []
        best_run: TrainingRunResult | None = None

        combos = list(
            product(
                self.config.sequence_lengths,
                self.config.hidden_sizes,
                self.config.dropouts,
            )
        )
        if self.config.limit_runs is not None:
            combos = combos[: int(self.config.limit_runs)]

        for sequence_length, hidden_size, dropout in combos:
            hyperparameters = {
                "sequence_length": int(sequence_length),
                "hidden_size": int(hidden_size),
                "num_layers": int(self.config.num_layers),
                "dropout": float(dropout),
            }
            train_samples, val_samples, test_samples = self._samples_for_run(
                generator,
                engineered_frames,
                train_ids,
                val_ids,
                test_ids,
                split_mode,
                sequence_length,
            )
            if not train_samples or not val_samples or not test_samples:
                print(
                    f"Skipping seq={sequence_length}, hidden={hidden_size}, dropout={dropout}: "
                    f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}."
                )
                continue

            run_id = f"seq{sequence_length}_hidden{hidden_size}_layers{self.config.num_layers}_drop{dropout:.2f}"
            print(f"Training {run_id}...")
            run = trainer.train_one(run_id, train_samples, val_samples, hyperparameters)
            best_run = run if best_run is None or run.best_val_loss < best_run.best_val_loss else best_run
            history_rows.append(
                {
                    "run_id": run.run_id,
                    "sequence_length": sequence_length,
                    "hidden_size": hidden_size,
                    "num_layers": self.config.num_layers,
                    "dropout": dropout,
                    "train_samples": len(train_samples),
                    "validation_samples": len(val_samples),
                    "test_samples": len(test_samples),
                    "best_epoch": run.best_epoch,
                    "best_train_loss": run.best_train_loss,
                    "best_val_loss": run.best_val_loss,
                    "final_train_loss": run.final_train_loss,
                    "device": run.device,
                }
            )
            for epoch, train_loss, val_loss in zip(
                range(1, len(run.train_losses) + 1),
                run.train_losses,
                run.val_losses,
            ):
                long_history_rows.append(
                    {
                        "run_id": run.run_id,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )

        if best_run is None:
            raise RuntimeError("No hyperparameter combination produced train/validation/test samples successfully.")

        best_sequence_length = int(best_run.hyperparameters["sequence_length"])
        train_samples, val_samples, test_samples = self._samples_for_run(
            generator,
            engineered_frames,
            train_ids,
            val_ids,
            test_ids,
            split_mode,
            best_sequence_length,
        )
        if not test_samples:
            all_samples = generator.generate_for_experiments(engineered_frames, experiment_ids, best_sequence_length)
            train_samples, val_samples, test_samples = SequenceGenerator.split_by_time(
                all_samples,
                self.config.train_fraction,
                self.config.val_fraction,
            )
            split_mode = "time_fallback_for_final_test"
        if not test_samples:
            raise RuntimeError("No test samples are available after sequence generation.")

        model = self._build_model(best_run)
        evaluator = Evaluator(best_run.feature_scaler, best_run.target_scaler, self.config, device=torch.device(best_run.device))
        evaluation_result = evaluator.evaluate_samples(model, test_samples)

        forecast_frames = self._generate_forecasts(evaluator, model, engineered_frames, test_samples)
        horizons_df = self._prediction_horizons_dataframe(forecast_frames, evaluator)
        predictions_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()

        plot_generator = PlotGenerator(self.directories["plots"], dpi=self.config.plot_dpi)
        history_df = pd.DataFrame(history_rows).sort_values("best_val_loss").reset_index(drop=True)
        long_history_df = pd.DataFrame(long_history_rows)
        plot_generator.save_all(history_df, best_run.run_id, forecast_frames, horizons_df, long_history_df)

        dataset_stats_df = self._dataset_statistics_dataframe(experiments, split_mode, train_ids, val_ids, test_ids)
        metrics_df = evaluation_result.metrics_df
        horizon_stats_df = self._horizon_statistics_dataframe(horizons_df)

        self._save_outputs(
            best_run,
            history_df,
            long_history_df,
            dataset_stats_df,
            metrics_df,
            horizons_df,
            horizon_stats_df,
            predictions_df,
            evaluation_result.wide_metrics,
        )
        summary_path = self._write_summary_report(
            experiments,
            dataset_stats_df,
            split_mode,
            train_ids,
            val_ids,
            test_ids,
            best_run,
            history_df,
            metrics_df,
            horizons_df,
            horizon_stats_df,
            evaluation_result.wide_metrics,
        )

        return {
            "summary_report": summary_path,
            "metrics": self.directories["root"] / "metrics.csv",
            "prediction_horizons": self.directories["root"] / "prediction_horizons.csv",
            "predictions": self.directories["root"] / "predictions.csv",
            "best_model": self.directories["root"] / "best_lstm_model.pt",
            "scaler": self.directories["root"] / "scaler.pkl",
        }

    def _split_experiment_ids(self, experiment_ids: List[str]) -> Tuple[str, List[str], List[str], List[str]]:
        if len(experiment_ids) >= 3:
            rng = np.random.default_rng(self.config.seed)
            shuffled = list(experiment_ids)
            rng.shuffle(shuffled)
            n_train = max(1, int(np.floor(len(shuffled) * self.config.train_fraction)))
            n_val = max(1, int(np.floor(len(shuffled) * self.config.val_fraction)))
            if n_train + n_val >= len(shuffled):
                n_train = max(1, len(shuffled) - 2)
                n_val = 1
            train_ids = shuffled[:n_train]
            val_ids = shuffled[n_train : n_train + n_val]
            test_ids = shuffled[n_train + n_val :]
            return "experiment", train_ids, val_ids, test_ids

        return "time_fallback", [], [], []

    def _samples_for_run(
        self,
        generator: SequenceGenerator,
        engineered_frames: Dict[str, pd.DataFrame],
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        split_mode: str,
        sequence_length: int,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
        if split_mode == "experiment":
            return generator.split_by_experiment_ids(
                generator.generate_for_experiments(engineered_frames, list(engineered_frames.keys()), sequence_length),
                train_ids,
                val_ids,
                test_ids,
            )
        all_samples = generator.generate_for_experiments(engineered_frames, list(engineered_frames.keys()), sequence_length)
        return SequenceGenerator.split_by_time(all_samples, self.config.train_fraction, self.config.val_fraction)

    def _build_model(self, best_run: TrainingRunResult) -> LSTMModel:
        model = LSTMModel(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=int(best_run.hyperparameters["hidden_size"]),
            num_layers=int(best_run.hyperparameters["num_layers"]),
            dropout=float(best_run.hyperparameters["dropout"]),
            output_size=len(TARGET_COLUMNS),
        )
        model.load_state_dict(best_run.model_state)
        model.to(torch.device(best_run.device))
        model.eval()
        return model

    def _generate_forecasts(
        self,
        evaluator: Evaluator,
        model: LSTMModel,
        engineered_frames: Dict[str, pd.DataFrame],
        test_samples: List[Dict[str, object]],
    ) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for sample in test_samples:
            grouped.setdefault(str(sample["experiment_id"]), []).append(sample)

        for experiment_id, samples in grouped.items():
            if experiment_id not in engineered_frames or not samples:
                continue
            best_sample = max(samples, key=lambda item: int(item["experiment_length"]) - int(item["end_idx"]))
            forecast_df = evaluator.forecast_from_sample(
                model,
                best_sample,
                engineered_frames[experiment_id],
                self.config.forecast_steps,
            )
            if not forecast_df.empty:
                frames.append(forecast_df)
        return frames

    def _prediction_horizons_dataframe(
        self,
        forecast_frames: List[pd.DataFrame],
        evaluator: Evaluator,
    ) -> pd.DataFrame:
        horizons = [evaluator.compute_prediction_horizon(frame) for frame in forecast_frames if not frame.empty]
        if not horizons:
            return pd.DataFrame(
                columns=["experiment_id", "horizon_step", "horizon_seconds", "threshold_degrees", "exceeded_threshold"]
            )
        return pd.DataFrame(horizons)

    def _horizon_statistics_dataframe(self, horizons_df: pd.DataFrame) -> pd.DataFrame:
        if horizons_df.empty or horizons_df["horizon_seconds"].dropna().empty:
            return pd.DataFrame(
                [
                    {"statistic": "count", "value": 0, "unit": "experiments"},
                    {"statistic": "mean", "value": np.nan, "unit": "seconds"},
                    {"statistic": "median", "value": np.nan, "unit": "seconds"},
                    {"statistic": "std", "value": np.nan, "unit": "seconds"},
                    {"statistic": "max", "value": np.nan, "unit": "seconds"},
                    {"statistic": "min", "value": np.nan, "unit": "seconds"},
                ]
            )
        values = horizons_df["horizon_seconds"].dropna().to_numpy(dtype=float)
        return pd.DataFrame(
            [
                {"statistic": "count", "value": len(values), "unit": "experiments"},
                {"statistic": "mean", "value": float(np.mean(values)), "unit": "seconds"},
                {"statistic": "median", "value": float(np.median(values)), "unit": "seconds"},
                {"statistic": "std", "value": float(np.std(values)), "unit": "seconds"},
                {"statistic": "max", "value": float(np.max(values)), "unit": "seconds"},
                {"statistic": "min", "value": float(np.min(values)), "unit": "seconds"},
            ]
        )

    def _dataset_statistics_dataframe(
        self,
        experiments: List[Any],
        split_mode: str,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for experiment in experiments:
            df = experiment.df
            rows.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "rows": len(df),
                    "start_time": float(df["time"].iloc[0]),
                    "end_time": float(df["time"].iloc[-1]),
                    "duration_seconds": float(df["time"].iloc[-1] - df["time"].iloc[0]),
                    "median_dt": experiment.median_dt,
                    "split": "training" if experiment.experiment_id in train_ids else "validation" if experiment.experiment_id in val_ids else "testing" if experiment.experiment_id in test_ids else split_mode,
                }
            )
        return pd.DataFrame(rows)

    def _save_outputs(
        self,
        best_run: TrainingRunResult,
        history_df: pd.DataFrame,
        long_history_df: pd.DataFrame,
        dataset_stats_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        horizons_df: pd.DataFrame,
        horizon_stats_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        wide_metrics: Dict[str, float],
    ) -> None:
        history_df.to_csv(self.directories["diagnostics"] / "hyperparameter_search.csv", index=False)
        long_history_df.to_csv(self.directories["diagnostics"] / "training_history_long.csv", index=False)
        dataset_stats_df.to_csv(self.directories["diagnostics"] / "dataset_statistics.csv", index=False)
        metrics_df.to_csv(self.directories["root"] / "metrics.csv", index=False)
        metrics_df.to_csv(self.directories["reports"] / "metrics.csv", index=False)
        horizons_df.to_csv(self.directories["root"] / "prediction_horizons.csv", index=False)
        horizon_stats_df.to_csv(self.directories["diagnostics"] / "prediction_horizon_statistics.csv", index=False)
        if not predictions_df.empty:
            predictions_df.to_csv(self.directories["predictions"] / "predictions.csv", index=False)
            predictions_df.to_csv(self.directories["root"] / "predictions.csv", index=False)
        else:
            empty_predictions = pd.DataFrame(
                columns=[
                    "experiment_id",
                    "step",
                    "time",
                    "theta1_true",
                    "theta2_true",
                    "theta1_pred",
                    "theta2_pred",
                    "omega1_true",
                    "omega2_true",
                    "omega1_pred",
                    "omega2_pred",
                    "angular_error_deg",
                ]
            )
            empty_predictions.to_csv(self.directories["predictions"] / "predictions.csv", index=False)
            empty_predictions.to_csv(self.directories["root"] / "predictions.csv", index=False)

        scaler_bundle = {
            "feature_scaler": best_run.feature_scaler,
            "target_scaler": best_run.target_scaler,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
        }
        with (self.directories["root"] / "scaler.pkl").open("wb") as handle:
            pickle.dump(scaler_bundle, handle)

        checkpoint = {
            "model_state_dict": best_run.model_state,
            "hyperparameters": best_run.hyperparameters,
            "config": self.config.to_dict(),
            "feature_scaler": best_run.feature_scaler,
            "target_scaler": best_run.target_scaler,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "metrics": wide_metrics,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        model_path = self.directories["models"] / "best_lstm_model.pt"
        torch.save(checkpoint, model_path)
        shutil.copyfile(model_path, self.directories["root"] / "best_lstm_model.pt")

    def _write_summary_report(
        self,
        experiments: List[Any],
        dataset_stats_df: pd.DataFrame,
        split_mode: str,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        best_run: TrainingRunResult,
        history_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        horizons_df: pd.DataFrame,
        horizon_stats_df: pd.DataFrame,
        wide_metrics: Dict[str, float],
    ) -> Path:
        total_rows = int(dataset_stats_df["rows"].sum()) if not dataset_stats_df.empty else 0
        horizon_values = horizons_df["horizon_seconds"].dropna().to_numpy(dtype=float) if not horizons_df.empty else np.array([])
        lines: List[str] = []
        lines.append("Double-Pendulum LSTM Extended Essay Training Summary")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("Dataset statistics")
        lines.append(f"- Experiments discovered: {len(experiments)}")
        lines.append(f"- Total raw samples: {total_rows}")
        lines.append(f"- Split mode: {split_mode}")
        lines.append(f"- Training experiments: {len(train_ids)}")
        lines.append(f"- Validation experiments: {len(val_ids)}")
        lines.append(f"- Testing experiments: {len(test_ids)}")
        if not dataset_stats_df.empty:
            lines.append(f"- Median experiment duration: {dataset_stats_df['duration_seconds'].median():.3f} s")
            lines.append(f"- Median sampling interval: {dataset_stats_df['median_dt'].median():.6f} s")
        lines.append("")
        lines.append("Training results")
        lines.append(f"- Completed hyperparameter runs: {len(history_df)}")
        lines.append(f"- Best run: {best_run.run_id}")
        lines.append(f"- Best validation loss: {best_run.best_val_loss:.8f}")
        lines.append(f"- Best epoch: {best_run.best_epoch}")
        lines.append("")
        lines.append("Best hyperparameters")
        for key, value in best_run.hyperparameters.items():
            lines.append(f"- {key}: {value}")
        lines.append(f"- device: {best_run.device}")
        lines.append("")
        lines.append("Final model architecture")
        lines.append(
            f"- LSTM(input_size={len(FEATURE_COLUMNS)}, hidden_size={best_run.hyperparameters['hidden_size']}, "
            f"num_layers={best_run.hyperparameters['num_layers']}, dropout={best_run.hyperparameters['dropout']}) "
            "-> Linear(output_size=4)"
        )
        lines.append("- Input features: dt, sin(theta1), cos(theta1), sin(theta2), cos(theta2), omega1, omega2")
        lines.append("- Output targets: theta1, theta2, omega1, omega2")
        lines.append("")
        lines.append("Test performance metrics")
        for _, row in metrics_df.iterrows():
            lines.append(f"- {row['target']} {row['metric']}: {row['value']:.6g} {row['unit']}")
        lines.append("")
        lines.append("Wide test metrics")
        for key, value in wide_metrics.items():
            lines.append(f"- {key}: {value:.6g}")
        lines.append("")
        lines.append("Prediction horizon statistics")
        for _, row in horizon_stats_df.iterrows():
            value = int(row["value"]) if row["statistic"] == "count" else row["value"]
            lines.append(f"- {row['statistic']}: {value} {row['unit']}")
        lines.append("")
        lines.append("Observations about chaotic prediction degradation")
        if len(horizon_values) > 0:
            mean_horizon = float(np.mean(horizon_values))
            median_horizon = float(np.median(horizon_values))
            lines.append(
                f"- The mean 5° angular-error prediction horizon was {mean_horizon:.3f} s "
                f"(median {median_horizon:.3f} s)."
            )
            lines.append(
                "- Recursive forecasting feeds predictions back into the LSTM, so small one-step errors accumulate. "
                "This is expected for chaotic double-pendulum motion, where nearby trajectories separate rapidly."
            )
        else:
            lines.append(
                "- No prediction horizon could be calculated because no test forecast contained both true and predicted future states."
            )
        if split_mode != "experiment":
            lines.append(
                "- Fewer than three valid experiments were available, so the application used a time-based fallback split. "
                "For the Extended Essay, use multiple independent recordings to preserve experiment-level testing."
            )
        lines.append(
            "- Angles were converted to sin/cos representations before training to avoid artificial discontinuities at ±π."
        )
        lines.append("")
        lines.append("Output files")
        lines.append(f"- Metrics: {self.directories['root'] / 'metrics.csv'}")
        lines.append(f"- Prediction horizons: {self.directories['root'] / 'prediction_horizons.csv'}")
        lines.append(f"- Predictions: {self.directories['root'] / 'predictions.csv'}")
        lines.append(f"- Best model: {self.directories['root'] / 'best_lstm_model.pt'}")
        lines.append(f"- Scaler: {self.directories['root'] / 'scaler.pkl'}")
        lines.append(f"- Plots: {self.directories['plots']}")

        report_path = self.directories["root"] / "summary_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        (self.directories["reports"] / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
        return report_path
