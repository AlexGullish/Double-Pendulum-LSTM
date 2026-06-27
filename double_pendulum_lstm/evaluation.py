from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from .config import FEATURE_COLUMNS, TARGET_COLUMNS


@dataclass
class EvaluationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    samples: List[Dict[str, object]]
    metrics_df: pd.DataFrame
    wide_metrics: Dict[str, float]


class Evaluator:
    def __init__(
        self,
        feature_scaler: Any,
        target_scaler: Any,
        config: Any,
        device: torch.device | None = None,
    ) -> None:
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_samples(
        self,
        model: torch.nn.Module,
        samples: List[Dict[str, object]],
    ) -> EvaluationResult:
        if not samples:
            raise ValueError("Cannot evaluate an empty test set.")
        x_raw = np.stack([sample["X"] for sample in samples]).astype(np.float32)
        y_true = np.stack([sample["y"] for sample in samples]).astype(np.float32)
        x_scaled = self.transform_features(x_raw)
        model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_scaled).to(self.device)
            y_pred_scaled = model(x_tensor).detach().cpu().numpy()
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled).astype(np.float32)
        metrics_df, wide_metrics = self.compute_metrics(y_true, y_pred)
        return EvaluationResult(
            y_true=y_true,
            y_pred=y_pred,
            samples=samples,
            metrics_df=metrics_df,
            wide_metrics=wide_metrics,
        )

    def transform_features(self, x_raw: np.ndarray) -> np.ndarray:
        x_raw = np.asarray(x_raw, dtype=np.float32)
        original_shape = x_raw.shape
        if x_raw.ndim == 2:
            x_raw = x_raw.reshape(1, -1, len(FEATURE_COLUMNS))
        if x_raw.ndim != 3 or x_raw.shape[-1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected feature array shape (batch, sequence, {len(FEATURE_COLUMNS)}), got {x_raw.shape}.")
        x_flat = x_raw.reshape(-1, len(FEATURE_COLUMNS))
        x_scaled = self.feature_scaler.transform(x_flat).reshape(original_shape)
        return x_scaled.astype(np.float32)

    def forecast_from_sample(
        self,
        model: torch.nn.Module,
        sample: Dict[str, object],
        frame: pd.DataFrame,
        steps: int,
    ) -> pd.DataFrame:
        sequence = np.asarray(sample["X"], dtype=np.float32).copy()
        experiment_id = str(sample["experiment_id"])
        experiment_length = int(sample["experiment_length"])
        dt = float(frame["time"].diff().median()) if len(frame) > 1 else float("nan")
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        last_time = float(frame["time"].iloc[-1])

        rows: List[Dict[str, Any]] = []
        model.eval()
        with torch.no_grad():
            for step in range(1, int(steps) + 1):
                x_scaled = self.transform_features(sequence.reshape(1, sequence.shape[0], sequence.shape[1]))
                x_tensor = torch.from_numpy(x_scaled).to(self.device)
                prediction = model(x_tensor).detach().cpu().numpy()[0]
                theta1, theta2, omega1, omega2 = self.target_scaler.inverse_transform(prediction.reshape(1, -1))[0]
                theta1 = self.angle_to_pi(float(theta1))
                theta2 = self.angle_to_pi(float(theta2))
                omega1 = float(omega1)
                omega2 = float(omega2)

                true_index = int(sample["end_idx"]) + step - 1
                if true_index < experiment_length:
                    true_row = frame.iloc[true_index]
                    theta1_true = float(true_row["theta1"])
                    theta2_true = float(true_row["theta2"])
                    omega1_true = float(true_row["omega1"])
                    omega2_true = float(true_row["omega2"])
                    time = float(true_row["time"])
                else:
                    theta1_true = float("nan")
                    theta2_true = float("nan")
                    omega1_true = float("nan")
                    omega2_true = float("nan")
                    time = last_time + step * dt

                angular_error_deg = self.angular_error_degrees(
                    np.array([theta1_true, theta2_true]),
                    np.array([theta1, theta2]),
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "step": int(step),
                        "time": float(time),
                        "theta1_true": theta1_true,
                        "theta2_true": theta2_true,
                        "theta1_pred": theta1,
                        "theta2_pred": theta2,
                        "omega1_true": omega1_true,
                        "omega2_true": omega2_true,
                        "omega1_pred": omega1,
                        "omega2_pred": omega2,
                        "angular_error_deg": angular_error_deg,
                    }
                )

                new_feature = np.array(
                    [
                        dt,
                        np.sin(theta1),
                        np.cos(theta1),
                        np.sin(theta2),
                        np.cos(theta2),
                        omega1,
                        omega2,
                    ],
                    dtype=np.float32,
                )
                sequence = np.vstack([sequence[1:], new_feature])

        return pd.DataFrame(rows)

    def compute_prediction_horizon(self, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        if forecast_df.empty:
            return {
                "experiment_id": None,
                "horizon_step": np.nan,
                "horizon_seconds": np.nan,
                "threshold_degrees": float(self.config.horizon_threshold_degrees),
                "exceeded_threshold": False,
            }
        threshold = float(self.config.horizon_threshold_degrees)
        exceeded = forecast_df["angular_error_deg"] > threshold
        start_time = float(forecast_df["time"].iloc[0])
        if exceeded.any():
            row = forecast_df.loc[exceeded].iloc[0]
            return {
                "experiment_id": row["experiment_id"],
                "horizon_step": int(row["step"]),
                "horizon_seconds": float(row["time"] - start_time),
                "threshold_degrees": threshold,
                "exceeded_threshold": True,
            }
        row = forecast_df.iloc[-1]
        return {
            "experiment_id": row["experiment_id"],
            "horizon_step": int(row["step"]),
            "horizon_seconds": float(row["time"] - start_time),
            "threshold_degrees": threshold,
            "exceeded_threshold": False,
        }

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
        rows: List[Dict[str, Any]] = []
        wide: Dict[str, float] = {}
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            true = y_true[:, target_index]
            pred = y_pred[:, target_index]
            if target_name.startswith("theta"):
                diff = self.angular_difference_rad(true, pred)
                rmse_rad = float(np.sqrt(np.mean(diff**2)))
                mae_rad = float(np.mean(diff))
                r2 = self.circular_r2(true, pred, diff)
                rows.extend(
                    [
                        {"target": target_name, "metric": "RMSE_rad", "value": rmse_rad, "unit": "rad"},
                        {"target": target_name, "metric": "RMSE_deg", "value": np.degrees(rmse_rad), "unit": "deg"},
                        {"target": target_name, "metric": "MAE_rad", "value": mae_rad, "unit": "rad"},
                        {"target": target_name, "metric": "MAE_deg", "value": np.degrees(mae_rad), "unit": "deg"},
                        {"target": target_name, "metric": "R2", "value": r2, "unit": "dimensionless"},
                    ]
                )
                wide[f"{target_name}_rmse_rad"] = rmse_rad
                wide[f"{target_name}_rmse_deg"] = np.degrees(rmse_rad)
                wide[f"{target_name}_mae_rad"] = mae_rad
                wide[f"{target_name}_mae_deg"] = np.degrees(mae_rad)
                wide[f"{target_name}_r2"] = r2
            else:
                rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
                mae = float(np.mean(np.abs(true - pred)))
                r2 = float(r2_score(true, pred)) if true.size > 1 and np.std(true) > 0 else float("nan")
                rows.extend(
                    [
                        {"target": target_name, "metric": "RMSE", "value": rmse, "unit": "rad/s"},
                        {"target": target_name, "metric": "MAE", "value": mae, "unit": "rad/s"},
                        {"target": target_name, "metric": "R2", "value": r2, "unit": "dimensionless"},
                    ]
                )
                wide[f"{target_name}_rmse"] = rmse
                wide[f"{target_name}_mae"] = mae
                wide[f"{target_name}_r2"] = r2
        return pd.DataFrame(rows), wide

    @staticmethod
    def angular_error_degrees(true_angles: np.ndarray, pred_angles: np.ndarray) -> float:
        diff = Evaluator.angular_difference_rad(true_angles, pred_angles)
        return float(np.sqrt(np.mean(np.degrees(diff) ** 2)))

    @staticmethod
    def angular_difference_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def angle_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def circular_mean(angles: np.ndarray) -> float:
        return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))

    @staticmethod
    def circular_r2(true_angles: np.ndarray, pred_angles: np.ndarray, residual: np.ndarray) -> float:
        del pred_angles
        centered = Evaluator.angle_to_pi(true_angles - Evaluator.circular_mean(true_angles))
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            return float("nan")
        return float(1.0 - (float(np.sum(residual**2)) / denominator))
