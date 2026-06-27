from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotGenerator:
    def __init__(self, output_dir: str | Path, dpi: int = 300) -> None:
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        plt.rcParams.update(
            {
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "font.size": 11,
                "axes.grid": True,
                "grid.alpha": 0.25,
            }
        )

    def save_training_curves(
        self,
        history_df: pd.DataFrame,
        best_run_id: str,
        long_history_df: pd.DataFrame | None = None,
    ) -> None:
        if history_df.empty:
            return
        if long_history_df is None or long_history_df.empty:
            return

        for metric, filename in [("train_loss", "training_loss.png"), ("val_loss", "validation_loss.png")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for run_id, run_df in long_history_df.groupby("run_id"):
                alpha = 0.25 if run_id != best_run_id else 1.0
                linewidth = 1.0 if run_id != best_run_id else 2.5
                ax.plot(
                    run_df["epoch"],
                    run_df[metric],
                    label=run_id if run_id == best_run_id else None,
                    alpha=alpha,
                    linewidth=linewidth,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean squared error loss")
            ax.set_title(f"{'Training' if metric == 'train_loss' else 'Validation'} Loss by Hyperparameter Run")
            if best_run_id in set(long_history_df["run_id"]):
                ax.legend(title="Best run", loc="best")
            fig.tight_layout()
            fig.savefig(self.output_dir / filename, bbox_inches="tight")
            plt.close(fig)

    def save_time_series(self, forecast_df: pd.DataFrame) -> None:
        if forecast_df.empty:
            return
        forecast_df = forecast_df.copy()
        forecast_df["time_rel"] = forecast_df["time"] - forecast_df["time"].iloc[0]
        comparisons = [
            ("theta1", "Experimental vs Predicted Theta 1", "Angle (degrees)"),
            ("theta2", "Experimental vs Predicted Theta 2", "Angle (degrees)"),
            ("omega1", "Experimental vs Predicted Omega 1", "Angular velocity (rad/s)"),
            ("omega2", "Experimental vs Predicted Omega 2", "Angular velocity (rad/s)"),
        ]
        for target, title, ylabel in comparisons:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(forecast_df["time_rel"], forecast_df[f"{target}_true"], label="Experimental", linewidth=1.8)
            ax.plot(forecast_df["time_rel"], forecast_df[f"{target}_pred"], label="LSTM prediction", linewidth=1.8)
            ax.set_xlabel("Time from forecast start (s)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best")
            fig.tight_layout()
            filename = f"experimental_vs_predicted_{target}.png"
            fig.savefig(self.output_dir / filename, bbox_inches="tight")
            plt.close(fig)

    def save_error_vs_time(self, forecast_frames: List[pd.DataFrame]) -> None:
        if not forecast_frames:
            return
        frames = [frame.loc[:, ["step", "angular_error_deg"]].copy() for frame in forecast_frames if not frame.empty]
        if not frames:
            return
        combined = pd.concat(frames, ignore_index=True)
        grouped = combined.groupby("step")["angular_error_deg"].agg(["mean", "std", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grouped["step"], grouped["mean"], color="#1f77b4", linewidth=2.2, label="Mean error")
        if "std" in grouped and grouped["std"].notna().any():
            ax.fill_between(
                grouped["step"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                color="#1f77b4",
                alpha=0.25,
                label="±1 standard deviation",
            )
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Angular prediction error (degrees)")
        ax.set_title("Prediction Error vs Time for Recursive Forecasts")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_error_vs_time.png", bbox_inches="tight")
        plt.close(fig)

    def save_horizon_distribution(self, horizons_df: pd.DataFrame) -> None:
        if horizons_df.empty or horizons_df["horizon_seconds"].dropna().empty:
            return
        values = horizons_df["horizon_seconds"].dropna().to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=min(20, max(1, len(values))), color="#4c72b0", alpha=0.75, edgecolor="black")
        ax.axvline(values.mean(), color="#d62728", linestyle="--", linewidth=2.0, label=f"Mean = {values.mean():.2f} s")
        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel("Number of test experiments")
        ax.set_title("Prediction Horizon Distribution at 5° Angular Error")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_horizon_distribution.png", bbox_inches="tight")
        plt.close(fig)

    def save_phase_portraits(self, forecast_df: pd.DataFrame) -> None:
        if forecast_df.empty:
            return
        for target_index, target in enumerate(["theta1", "theta2"]):
            omega = "omega1" if target_index == 0 else "omega2"
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(
                np.degrees(forecast_df[f"{target}_true"]),
                forecast_df[f"{omega}_true"],
                s=12,
                alpha=0.55,
                label="Experimental",
            )
            ax.scatter(
                np.degrees(forecast_df[f"{target}_pred"]),
                forecast_df[f"{omega}_pred"],
                s=12,
                alpha=0.55,
                label="LSTM prediction",
            )
            ax.set_xlabel(f"{target} (degrees)")
            ax.set_ylabel(f"{omega} (rad/s)")
            ax.set_title(f"Phase Portrait Comparison: {target} vs {omega}")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"phase_portrait_{target}.png", bbox_inches="tight")
            plt.close(fig)

    def save_trajectory_comparison(self, forecast_df: pd.DataFrame) -> None:
        if forecast_df.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            np.degrees(forecast_df["theta1_true"]),
            np.degrees(forecast_df["theta2_true"]),
            s=12,
            alpha=0.55,
            label="Experimental trajectory",
        )
        ax.scatter(
            np.degrees(forecast_df["theta1_pred"]),
            np.degrees(forecast_df["theta2_pred"]),
            s=12,
            alpha=0.55,
            label="LSTM predicted trajectory",
        )
        ax.set_xlabel("Theta 1 (degrees)")
        ax.set_ylabel("Theta 2 (degrees)")
        ax.set_title("Double-Pendulum Trajectory Comparison")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_dir / "trajectory_comparison.png", bbox_inches="tight")
        plt.close(fig)

    def save_all(
        self,
        history_df: pd.DataFrame,
        best_run_id: str,
        forecast_frames: List[pd.DataFrame],
        horizons_df: pd.DataFrame,
        long_history_df: pd.DataFrame | None = None,
    ) -> None:
        self.save_training_curves(history_df, best_run_id, long_history_df)
        selected_forecast = self._select_forecast(forecast_frames)
        self.save_time_series(selected_forecast)
        self.save_error_vs_time(forecast_frames)
        self.save_horizon_distribution(horizons_df)
        self.save_phase_portraits(selected_forecast)
        self.save_trajectory_comparison(selected_forecast)

    @staticmethod
    def _select_forecast(forecast_frames: List[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [frame for frame in forecast_frames if not frame.empty]
        if not non_empty:
            return pd.DataFrame()
        return max(non_empty, key=lambda frame: len(frame)).copy()
