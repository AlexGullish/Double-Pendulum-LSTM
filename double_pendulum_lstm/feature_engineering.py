from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMNS
from .data import ExperimentData


class FeatureEngineer:
    feature_columns: List[str] = FEATURE_COLUMNS
    target_columns: List[str] = TARGET_COLUMNS

    def engineer_all(self, experiments: Iterable[ExperimentData]) -> Dict[str, pd.DataFrame]:
        return {experiment.experiment_id: self.engineer(experiment) for experiment in experiments}

    def engineer(self, experiment: ExperimentData) -> pd.DataFrame:
        df = experiment.df.copy()
        theta1 = self.angle_to_pi(df["theta1_rad"].to_numpy(dtype=float))
        theta2 = self.angle_to_pi(df["theta2_rad"].to_numpy(dtype=float))
        omega1 = df["omega1"].to_numpy(dtype=float)
        omega2 = df["omega2"].to_numpy(dtype=float)
        dt = df["dt"].to_numpy(dtype=float)

        engineered = pd.DataFrame(
            {
                "time": df["time"].to_numpy(dtype=float),
                "dt": dt,
                "theta1": theta1,
                "theta2": theta2,
                "omega1": omega1,
                "omega2": omega2,
                "sin_theta1": np.sin(theta1),
                "cos_theta1": np.cos(theta1),
                "sin_theta2": np.sin(theta2),
                "cos_theta2": np.cos(theta2),
            }
        )
        return engineered

    @staticmethod
    def angle_to_pi(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi
