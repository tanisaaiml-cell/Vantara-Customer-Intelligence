"""CPU PyTorch classifiers, purchase sequence model and spending autoencoder."""

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    """Fix random behavior and CPU thread count."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


class ANN(nn.Module):
    """Feed-forward churn classifier with batch normalization and dropout."""

    def __init__(self, inputs: int, hidden: int = 48, dropout: float = 0.25) -> None:
        """Construct the configured ANN."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a churn logit per customer."""
        return self.net(x).squeeze(-1)


class PurchaseLSTM(nn.Module):
    """Variable-length sequence encoder predicting normalized capped waiting time."""

    def __init__(self, inputs: int = 8, hidden: int = 32) -> None:
        """Construct the recurrent model."""
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Use packed sequences to ignore padding."""
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.head(hidden[-1]).squeeze(-1)


class Autoencoder(nn.Module):
    """Five-dimensional bottleneck for spending-pattern reconstruction."""

    def __init__(self, inputs: int) -> None:
        """Create encoder and decoder layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.ReLU(),
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, inputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a feature vector."""
        return self.net(x)


def train_network(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: dict[str, Any],
    task: str,
    train_lengths: np.ndarray | None = None,
    val_lengths: np.ndarray | None = None,
    epochs: int | None = None,
) -> list[dict[str, float]]:
    """Early-stop on validation loss and restore the best weights."""
    settings = cfg["dl"]
    tensors = [torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)]
    if train_lengths is not None:
        tensors.append(torch.tensor(train_lengths))
    loader = DataLoader(
        TensorDataset(*tensors), batch_size=settings["batch_size"], shuffle=True, drop_last=(task == "ann")
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    loss_fn = (
        nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1 - train_y).sum() / max(train_y.sum(), 1))))
        if task == "ann"
        else nn.MSELoss()
    )
    vx = torch.tensor(val_x, dtype=torch.float32)
    vy = torch.tensor(val_y, dtype=torch.float32)
    vl = None if val_lengths is None else torch.tensor(val_lengths)
    best, stale, state, curve = float("inf"), 0, copy.deepcopy(model.state_dict()), []
    for epoch in range(epochs or settings["epochs"]):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            prediction = model(batch[0], batch[2]) if train_lengths is not None else model(batch[0])
            loss = loss_fn(prediction, batch[1])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            prediction = model(vx, vl) if vl is not None else model(vx)
            val_loss = float(loss_fn(prediction, vy))
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best - 1e-5:
            best, stale, state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    return curve


def infer(model: nn.Module, x: np.ndarray, lengths: np.ndarray | None = None, sigmoid: bool = False) -> np.ndarray:
    """Run deterministic CPU inference with dropout disabled."""
    model.eval()
    with torch.no_grad():
        data = torch.tensor(x, dtype=torch.float32)
        output = model(data, torch.tensor(lengths)) if lengths is not None else model(data)
        if sigmoid:
            output = torch.sigmoid(output)
    return output.numpy()
