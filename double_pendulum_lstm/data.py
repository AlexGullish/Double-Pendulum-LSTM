from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import REQUIRED_COLUMNS


@dataclass
class ExperimentData:
    experiment_id: str
    path: Path
    raw_path: Path
    df: pd.DataFrame
    metadata: Dict[str, Any]
    median_dt: float


class DatasetLoader:
    def __init__(self, root_dir: str | Path, required_columns: List[str] | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.required_columns = required_columns or REQUIRED_COLUMNS

    def discover_experiment_paths(self) -> List[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {self.root_dir}")
        paths = sorted(self.root_dir.glob("*/raw_data.csv"))
        if not paths and (self.root_dir / "raw_data.csv").exists():
            paths = [self.root_dir / "raw_data.csv"]
        return paths

    def load_all(self) -> List[ExperimentData]:
        experiments: List[ExperimentData] = []
        for raw_path in self.discover_experiment_paths():
            experiment = self.load_experiment(raw_path)
            if experiment is not None:
                experiments.append(experiment)
        return experiments

    def load_experiment(self, raw_path: str | Path) -> Optional[ExperimentData]:
        raw_path = Path(raw_path)
        experiment_id = raw_path.parent.name
        try:
            raw_df = pd.read_csv(raw_path)
        except Exception as exc:
            print(f"Ignoring {experiment_id}: could not read raw_data.csv ({exc}).")
            return None

        missing = sorted(set(self.required_columns) - set(raw_df.columns))
        if missing:
            print(f"Ignoring {experiment_id}: missing required columns {missing}.")
            return None

        columns = list(self.required_columns)
        if "dt" in raw_df.columns:
            columns.append("dt")

        df = raw_df.loc[:, columns].copy()
        for column in self.required_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if "dt" in df.columns:
            df["dt"] = pd.to_numeric(df["dt"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.required_columns)
        if df.empty:
            print(f"Ignoring {experiment_id}: no finite required-column rows.")
            return None

        df = df.sort_values("time").reset_index(drop=True)
        time_dt = np.diff(df["time"].to_numpy(dtype=float))
        time_dt = np.where(np.isfinite(time_dt) & (time_dt > 0), time_dt, np.nan)
        computed_dt = np.full(len(df), np.nan, dtype=float)
        if len(df) > 1:
            first_dt = time_dt[~np.isnan(time_dt)][0] if np.any(~np.isnan(time_dt)) else np.nan
            computed_dt[0] = first_dt
            computed_dt[1:] = time_dt
        else:
            computed_dt[0] = np.nan

        if "dt" in df.columns:
            df["dt"] = df["dt"].where(df["dt"].replace([np.inf, -np.inf], np.nan).notna(), computed_dt)
            df["dt"] = df["dt"].fillna(float(np.nanmedian(computed_dt)) if np.any(~np.isnan(computed_dt)) else np.nan)
        else:
            df["dt"] = computed_dt

        dt_values = df["dt"].to_numpy(dtype=float)
        dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
        median_dt = float(np.median(dt_values)) if dt_values.size else float("nan")

        metadata_path = raw_path.parent / "metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}

        return ExperimentData(
            experiment_id=experiment_id,
            path=raw_path.parent,
            raw_path=raw_path,
            df=df,
            metadata=metadata,
            median_dt=median_dt,
        )
