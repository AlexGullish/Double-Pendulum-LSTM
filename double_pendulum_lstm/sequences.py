from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMNS


class SequenceGenerator:
    def __init__(self, feature_columns: List[str] | None = None, target_columns: List[str] | None = None) -> None:
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.target_columns = target_columns or TARGET_COLUMNS

    def generate_for_experiments(
        self,
        engineered_frames: Dict[str, pd.DataFrame],
        experiment_ids: Iterable[str],
        sequence_length: int,
    ) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        for experiment_id in experiment_ids:
            frame = engineered_frames.get(experiment_id)
            if frame is None:
                continue
            samples.extend(self.generate(experiment_id, frame, sequence_length))
        return samples

    def generate(self, experiment_id: str, frame: pd.DataFrame, sequence_length: int) -> List[Dict[str, object]]:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        if len(frame) <= sequence_length:
            return []

        features = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        targets = frame.loc[:, self.target_columns].to_numpy(dtype=np.float32)
        times = frame["time"].to_numpy(dtype=float)
        experiment_length = len(frame)
        samples: List[Dict[str, object]] = []
        for start_idx in range(0, len(frame) - sequence_length):
            end_idx = start_idx + sequence_length
            samples.append(
                {
                    "X": features[start_idx:end_idx].astype(np.float32, copy=False),
                    "y": targets[end_idx].astype(np.float32, copy=False),
                    "time": float(times[end_idx]),
                    "experiment_id": experiment_id,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "experiment_length": int(experiment_length),
                }
            )
        return samples

    @staticmethod
    def split_by_experiment_ids(
        samples: Sequence[Dict[str, object]],
        train_ids: Iterable[str],
        val_ids: Iterable[str],
        test_ids: Iterable[str],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
        train_ids_set = set(train_ids)
        val_ids_set = set(val_ids)
        test_ids_set = set(test_ids)
        train_samples: List[Dict[str, object]] = []
        val_samples: List[Dict[str, object]] = []
        test_samples: List[Dict[str, object]] = []
        for sample in samples:
            experiment_id = str(sample["experiment_id"])
            if experiment_id in train_ids_set:
                train_samples.append(sample)
            elif experiment_id in val_ids_set:
                val_samples.append(sample)
            elif experiment_id in test_ids_set:
                test_samples.append(sample)
        return train_samples, val_samples, test_samples

    @staticmethod
    def split_by_time(
        samples: Sequence[Dict[str, object]],
        train_fraction: float = 0.70,
        val_fraction: float = 0.15,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for sample in samples:
            grouped[str(sample["experiment_id"])].append(sample)

        train_samples: List[Dict[str, object]] = []
        val_samples: List[Dict[str, object]] = []
        test_samples: List[Dict[str, object]] = []
        for experiment_samples in grouped.values():
            experiment_samples.sort(key=lambda item: (int(item["start_idx"]), int(item["end_idx"])))
            count = len(experiment_samples)
            if count == 0:
                continue

            train_end = int(np.floor(count * train_fraction))
            val_end = int(np.floor(count * (train_fraction + val_fraction)))

            if count >= 3:
                train_end = max(1, train_end)
                val_end = max(train_end + 1, min(count - 1, val_end))
            else:
                train_end = count
                val_end = count

            train_samples.extend(experiment_samples[:train_end])
            val_samples.extend(experiment_samples[train_end:val_end])
            test_samples.extend(experiment_samples[val_end:])
        return train_samples, val_samples, test_samples
