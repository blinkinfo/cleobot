"""Temporal Convolutional Network (TCN) model for CleoBot.

Model B in the 3-layer ensemble. Captures sequential patterns in
time-series data that tabular models miss.

Input: Sequence of last 12-24 candles as multi-channel time series
       (OHLCV + orderbook snapshots)
Architecture: 3-4 residual blocks, dilations [1, 2, 4, 8], kernel size 3,
              dropout 0.2
Expected standalone accuracy: 53-56%
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils.logger import get_logger

logger = get_logger("models.tcn")

# Default TCN architecture hyperparameters
DEFAULT_SEQ_LENGTH = 24  # Last 24 candles (2 hours)
DEFAULT_NUM_CHANNELS = [32, 32, 32, 32]  # 4 residual blocks
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT = 0.2
DEFAULT_DILATIONS = [1, 2, 4, 8]

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 50
DEFAULT_WEIGHT_DECAY = 1e-4


# ------------------------------------------------------------------ #
# PyTorch modules
# ------------------------------------------------------------------ #

class Chomp1d(nn.Module):
    """Remove extra padding from causal convolution output."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single residual block in the TCN.

    Two causal dilated convolutions with ReLU, dropout, and a residual
    connection (with optional 1x1 conv for channel matching).
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2,
        )

        # 1x1 conv for residual connection if channel dims differ
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Initialise convolution weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNetwork(nn.Module):
    """Temporal Convolutional Network for binary classification.

    Architecture: stacked TemporalBlocks -> global average pool -> FC -> sigmoid.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilations: Optional[List[int]] = None,
    ):
        """Initialise the TCN.

        Args:
            num_inputs: Number of input channels (features per timestep).
            num_channels: List of channel sizes for each temporal block.
            kernel_size: Convolution kernel size.
            dropout: Dropout rate.
            dilations: Dilation factors per block. Defaults to [1, 2, 4, 8].
        """
        super().__init__()
        if dilations is None:
            dilations = DEFAULT_DILATIONS

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = dilations[i] if i < len(dilations) else 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_length).

        Returns:
            Probability tensor of shape (batch,).
        """
        # x: (batch, channels, seq_len)
        out = self.network(x)
        # Global average pooling over the time dimension
        out = out.mean(dim=2)  # (batch, channels)
        out = self.fc(out).squeeze(-1)  # (batch,)
        return torch.sigmoid(out)


class SequenceDataset(Dataset):
    """PyTorch dataset for TCN training.

    Creates sequences of fixed length from feature DataFrames.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        """Initialise the dataset.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            seq_length: Sequence length for each sample.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_length + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract sequence: (seq_length, n_features) -> transpose to (n_features, seq_length)
        x_seq = self.X[idx: idx + self.seq_length].T  # (features, seq_len)
        # Label is for the LAST timestep in the sequence
        y_val = self.y[idx + self.seq_length - 1]
        return x_seq, y_val


# ------------------------------------------------------------------ #
# TCN Model Wrapper
# ------------------------------------------------------------------ #

class TCNModel:
    """TCN wrapper with train/predict/save/load for CleoBot.

    Manages the PyTorch TCN model lifecycle including training with
    cosine annealing learning rate schedule.
    """

    def __init__(
        self,
        seq_length: int = DEFAULT_SEQ_LENGTH,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
        dilations: Optional[List[int]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
    ):
        self.seq_length = seq_length
        self.num_channels = num_channels or list(DEFAULT_NUM_CHANNELS)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilations = dilations or list(DEFAULT_DILATIONS)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay

        self.model: Optional[TCNNetwork] = None
        self.feature_names: List[str] = []
        self.num_inputs: int = 0
        self.version: int = 0
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self._is_trained: bool = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalisation parameters (fit on training data)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained and self.model is not None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Train the TCN model.

        Args:
            X_train: Training features (n_samples, n_features). Must be
                     temporally ordered (oldest first).
            y_train: Training labels (0=DOWN, 1=UP).
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Dict with training metrics.
        """
        self.feature_names = list(X_train.columns)
        self.num_inputs = len(self.feature_names)

        logger.info(
            f"Training TCN: {X_train.shape[0]} samples, "
            f"{self.num_inputs} features, seq_len={self.seq_length}"
        )

        # Fit normalisation on training data
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.float32)
        self._mean = X_train_np.mean(axis=0)
        self._std = X_train_np.std(axis=0)
        self._std[self._std < 1e-8] = 1.0  # Avoid div-by-zero

        X_train_norm = (X_train_np - self._mean) / self._std

        # Create dataset and loader
        train_dataset = SequenceDataset(X_train_norm, y_train_np, self.seq_length)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = (X_val.values.astype(np.float32) - self._mean) / self._std
            y_val_np = y_val.values.astype(np.float32)
            val_dataset = SequenceDataset(X_val_norm, y_val_np, self.seq_length)
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False,
                )

        # Build model
        self.model = TCNNetwork(
            num_inputs=self.num_inputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            dilations=self.dilations,
        ).to(self.device)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs,
        )
        criterion = nn.BCELoss()

        # Training loop
        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                preds = (output > 0.5).float()
                epoch_correct += (preds == batch_y).sum().item()
                epoch_total += batch_x.size(0)

            scheduler.step()

            train_acc = epoch_correct / max(epoch_total, 1)
            avg_loss = epoch_loss / max(epoch_total, 1)

            # Validation
            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"loss={avg_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_acc={val_acc:.4f}"
                )

        # Restore best model if validation was used
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.val_accuracy = best_val_acc
        else:
            self.val_accuracy = train_acc

        # Final training accuracy
        if len(train_dataset) > 0:
            final_train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False,
            )
            self.train_accuracy = self._evaluate(final_train_loader)
        else:
            self.train_accuracy = train_acc

        self._is_trained = True

        metrics = {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "epochs": self.epochs,
            "seq_length": self.seq_length,
            "num_features": self.num_inputs,
        }
        logger.info(
            f"TCN trained: train_acc={self.train_accuracy:.4f}, "
            f"val_acc={self.val_accuracy:.4f}"
        )
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict UP probability for each valid sequence.

        For a DataFrame with N rows, produces (N - seq_length + 1) predictions.
        The last prediction corresponds to the most recent data.

        Args:
            X: Feature DataFrame (temporally ordered, oldest first).

        Returns:
            Array of UP probabilities.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X_aligned = self._align_features(X)
        X_np = X_aligned.values.astype(np.float32)
        X_norm = (X_np - self._mean) / self._std

        self.model.eval()
        probas = []

        with torch.no_grad():
            for i in range(len(X_norm) - self.seq_length + 1):
                seq = X_norm[i: i + self.seq_length]  # (seq_len, features)
                x_tensor = torch.FloatTensor(seq.T).unsqueeze(0).to(self.device)
                prob = self.model(x_tensor).cpu().item()
                probas.append(prob)

        return np.array(probas)

    def predict_single(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict for the most recent sequence in the DataFrame.

        Args:
            X: Feature DataFrame with at least seq_length rows.

        Returns:
            Dict with 'direction', 'probability', 'confidence'.
        """
        if len(X) < self.seq_length:
            logger.warning(
                f"TCN needs {self.seq_length} rows but got {len(X)}. "
                "Returning neutral prediction."
            )
            return {"direction": "UP", "probability": 0.5, "confidence": 0.0}

        # Use only the last seq_length rows
        X_tail = X.tail(self.seq_length).copy()
        probas = self.predict_proba(X_tail)

        if len(probas) == 0:
            return {"direction": "UP", "probability": 0.5, "confidence": 0.0}

        proba = float(probas[-1])
        direction = "UP" if proba > 0.5 else "DOWN"
        confidence = abs(proba - 0.5) * 2
        return {
            "direction": direction,
            "probability": proba,
            "confidence": confidence,
        }

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: File path for the .pt file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "num_inputs": self.num_inputs,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "dilations": self.dilations,
            "seq_length": self.seq_length,
            "feature_names": self.feature_names,
            "mean": self._mean,
            "std": self._std,
            "version": self.version,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }
        torch.save(state, path)
        logger.info(f"TCN model saved to {path}")

    def load(self, path: str):
        """Load model from disk.

        Args:
            path: File path for the .pt file.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.num_inputs = state["num_inputs"]
        self.num_channels = state["num_channels"]
        self.kernel_size = state["kernel_size"]
        self.dropout = state["dropout"]
        self.dilations = state["dilations"]
        self.seq_length = state["seq_length"]
        self.feature_names = state["feature_names"]
        self._mean = state["mean"]
        self._std = state["std"]
        self.version = state["version"]
        self.train_accuracy = state["train_accuracy"]
        self.val_accuracy = state["val_accuracy"]

        self.model = TCNNetwork(
            num_inputs=self.num_inputs,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            dilations=self.dilations,
        ).to(self.device)
        if state["model_state_dict"]:
            self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self._is_trained = True
        logger.info(f"TCN model loaded from {path} (v{self.version})")

    def _evaluate(self, loader: DataLoader) -> float:
        """Evaluate accuracy on a DataLoader."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                preds = (output > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
        return correct / max(total, 1)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align input features to training feature order."""
        if not self.feature_names:
            return X
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            X = X.copy()
            for col in missing:
                X[col] = 0.0
        return X[self.feature_names]

    def get_optuna_param_space(self):
        """Return the Optuna hyperparameter search space."""

        def suggest_params(trial):
            n_blocks = trial.suggest_int("n_blocks", 2, 4)
            channel_size = trial.suggest_categorical(
                "channel_size", [16, 32, 64]
            )
            return {
                "seq_length": trial.suggest_categorical(
                    "seq_length", [12, 18, 24]
                ),
                "num_channels": [channel_size] * n_blocks,
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.4),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 5e-3, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128]
                ),
                "epochs": trial.suggest_int("epochs", 20, 60),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-5, 1e-3, log=True
                ),
            }

        return suggest_params
