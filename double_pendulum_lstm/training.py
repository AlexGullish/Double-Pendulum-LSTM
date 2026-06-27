from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import AppConfig, FEATURE_COLUMNS, TARGET_COLUMNS
from .model import LSTMModel


@dataclass
class TrainingRunResult:
    run_id: str
    hyperparameters: Dict[str, Any]
    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    best_train_loss: float
    best_val_loss: float
    final_train_loss: float
    model_state: Dict[str, torch.Tensor]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: List[str]
    target_columns: List[str]
    config: Dict[str, Any]
    device: str


class Trainer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device(config.resolved_device())

    def train_one(
        self,
        run_id: str,
        train_samples: List[Dict[str, object]],
        val_samples: List[Dict[str, object]],
        hyperparameters: Dict[str, Any],
    ) -> TrainingRunResult:
        self._set_seed(self.config.seed)
        feature_scaler, target_scaler = self._fit_scalers(train_samples)
        model = LSTMModel(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=int(hyperparameters["hidden_size"]),
            num_layers=int(hyperparameters["num_layers"]),
            dropout=float(hyperparameters["dropout"]),
            output_size=len(TARGET_COLUMNS),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.config.scheduler_factor),
            patience=int(self.config.scheduler_patience),
        )
        loss_fn = torch.nn.MSELoss()

        train_loader = self._make_loader(
            train_samples,
            feature_scaler,
            target_scaler,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = self._make_loader(
            val_samples,
            feature_scaler,
            target_scaler,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_epoch = 0
        best_state: Dict[str, torch.Tensor] = {}
        train_losses: List[float] = []
        val_losses: List[float] = []
        patience_counter = 0

        for epoch in range(1, int(self.config.max_epochs) + 1):
            model.train()
            epoch_losses: List[float] = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.config.grad_clip))
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            val_loss = self._evaluate_loss(model, val_loader, loss_fn)
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(float(val_loss))

            if val_loss < best_val_loss - float(self.config.min_delta):
                best_val_loss = float(val_loss)
                best_train_loss = train_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= int(self.config.patience):
                break

        if not best_state:
            raise RuntimeError(f"Training failed to produce a best model state for run {run_id}.")

        return TrainingRunResult(
            run_id=run_id,
            hyperparameters=hyperparameters,
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
            final_train_loss=train_losses[-1] if train_losses else float("nan"),
            model_state=best_state,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            feature_columns=FEATURE_COLUMNS,
            target_columns=TARGET_COLUMNS,
            config=self.config.to_dict(),
            device=str(self.device),
        )

    def run_hyperparameter_search(
        self,
        train_samples: List[Dict[str, object]],
        val_samples: List[Dict[str, object]],
    ) -> Tuple[TrainingRunResult, pd.DataFrame]:
        runs: List[TrainingRunResult] = []
        rows: List[Dict[str, Any]] = []
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
            run_id = f"seq{sequence_length}_hidden{hidden_size}_layers{self.config.num_layers}_drop{dropout:.2f}"
            print(f"Training {run_id}...")
            run = self.train_one(run_id, train_samples, val_samples, hyperparameters)
            runs.append(run)
            rows.append(
                {
                    "run_id": run.run_id,
                    "sequence_length": sequence_length,
                    "hidden_size": hidden_size,
                    "num_layers": self.config.num_layers,
                    "dropout": dropout,
                    "best_epoch": run.best_epoch,
                    "best_train_loss": run.best_train_loss,
                    "best_val_loss": run.best_val_loss,
                    "final_train_loss": run.final_train_loss,
                    "device": run.device,
                }
            )

        if not runs:
            raise RuntimeError("No hyperparameter runs completed successfully.")
        history_df = pd.DataFrame(rows).sort_values("best_val_loss").reset_index(drop=True)
        best_run = min(runs, key=lambda run: run.best_val_loss)
        return best_run, history_df

    def _fit_scalers(
        self, train_samples: List[Dict[str, object]]
    ) -> Tuple[StandardScaler, StandardScaler]:
        x_raw = np.stack([sample["X"] for sample in train_samples]).astype(np.float32)
        y_raw = np.stack([sample["y"] for sample in train_samples]).astype(np.float32)
        x_flat = x_raw.reshape(-1, len(FEATURE_COLUMNS))
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        feature_scaler.fit(x_flat)
        target_scaler.fit(y_raw)
        return feature_scaler, target_scaler

    def _make_loader(
        self,
        samples: List[Dict[str, object]],
        feature_scaler: StandardScaler,
        target_scaler: StandardScaler,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        if not samples:
            raise ValueError("Cannot create a DataLoader from an empty sample list.")
        x_raw = np.stack([sample["X"] for sample in samples]).astype(np.float32)
        y_raw = np.stack([sample["y"] for sample in samples]).astype(np.float32)
        x_flat = x_raw.reshape(-1, len(FEATURE_COLUMNS))
        x_scaled = feature_scaler.transform(x_flat).reshape(x_raw.shape).astype(np.float32)
        y_scaled = self._transform_targets(y_raw, target_scaler)
        x_tensor = torch.from_numpy(x_scaled)
        y_tensor = torch.from_numpy(y_scaled.astype(np.float32))
        return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

    def _transform_targets(self, y_raw: np.ndarray, target_scaler: StandardScaler) -> np.ndarray:
        return target_scaler.transform(y_raw).astype(np.float32)

    def _evaluate_loss(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        loss_fn: torch.nn.Module,
    ) -> float:
        losses: List[float] = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                predictions = model(x_batch)
                loss = loss_fn(predictions, y_batch)
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
