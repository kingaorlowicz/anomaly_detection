"""PyTorch chart-image classifier for visual anomaly detection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class VisualTrainingConfig:
    """Training parameters for the image classifier."""

    image_size: int = 224
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    random_state: int = 42


class ChartImageDataset:
    """Minimal PyTorch-compatible dataset backed by metadata rows."""

    def __init__(
        self,
        dataset_dir: Path,
        metadata_df: pd.DataFrame,
        image_size: int,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int):
        import torch

        row = self.metadata_df.iloc[index]
        image_path = self.dataset_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0

        # Normalization constants match common ImageNet preprocessing.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "visual_anomaly_detector requires PyTorch. Install torch to train "
            "or load the visual model."
        ) from exc
    return torch, nn, DataLoader


def build_simple_cnn(num_classes: int = 2):
    """Build a compact CNN that is adequate for chart-shape prototypes."""

    _, nn, _ = _require_torch()
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )


class VisualAnomalyDetector:
    """Supervised CNN classifier for normal vs anomalous chart images."""

    def __init__(
        self,
        config: VisualTrainingConfig | None = None,
        model=None,
    ) -> None:
        self.config = config or VisualTrainingConfig()
        torch, _, _ = _require_torch()
        self.torch = torch
        self.device = self._resolve_device(self.config.device)
        self.model = model if model is not None else build_simple_cnn()
        self.model.to(self.device)

    def fit(self, dataset_dir: Path) -> list[dict[str, float]]:
        """Train on train split and keep the best validation checkpoint."""

        torch, nn, DataLoader = _require_torch()
        torch.manual_seed(self.config.random_state)
        dataset_dir = Path(dataset_dir)
        metadata_df = pd.read_csv(dataset_dir / "window_metadata.csv")
        train_df = metadata_df[metadata_df["split"] == "train"]
        val_df = metadata_df[metadata_df["split"] == "val"]

        train_loader = DataLoader(
            ChartImageDataset(dataset_dir, train_df, self.config.image_size),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(
            ChartImageDataset(dataset_dir, val_df, self.config.image_size),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        class_weights = self._class_weights(train_df["label"].to_numpy())
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history = []
        best_state = None
        best_val_f1 = -1.0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(
                train_loader,
                criterion,
                optimizer,
            )
            val_metrics = self._evaluate_loader(val_loader)
            record = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                **val_metrics,
            }
            history.append(record)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict_proba(self, image_paths: list[Path]) -> np.ndarray:
        """Return anomaly probabilities for arbitrary chart image paths."""

        torch, _, DataLoader = _require_torch()
        metadata_df = pd.DataFrame(
            {
                "image_path": [str(path) for path in image_paths],
                "label": [0] * len(image_paths),
            }
        )
        dataset = _AbsoluteImageDataset(metadata_df, self.config.image_size)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probabilities.append(probs.detach().cpu().numpy())
        return np.concatenate(probabilities) if probabilities else np.array([])

    def predict_dataset_split(
        self,
        dataset_dir: Path,
        split: str = "test",
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Predict anomaly probabilities for one generated dataset split."""

        torch, _, DataLoader = _require_torch()
        dataset_dir = Path(dataset_dir)
        metadata_df = pd.read_csv(dataset_dir / "window_metadata.csv")
        split_df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
        loader = DataLoader(
            ChartImageDataset(dataset_dir, split_df, self.config.image_size),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        probabilities = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, target in loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probabilities.append(probs.detach().cpu().numpy())
                labels.append(target.numpy())

        y_true = np.concatenate(labels) if labels else np.array([], dtype=int)
        y_score = (
            np.concatenate(probabilities) if probabilities else np.array([])
        )
        return y_true, y_score, split_df

    def save(self, path: Path) -> None:
        """Save model weights and training config."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "VisualAnomalyDetector":
        """Load a saved detector."""

        torch, _, _ = _require_torch()
        checkpoint = torch.load(path, map_location="cpu")
        config = VisualTrainingConfig(**checkpoint["config"])
        config.device = device
        detector = cls(config=config)
        detector.model.load_state_dict(checkpoint["state_dict"])
        detector.model.to(detector.device)
        detector.model.eval()
        return detector

    def _resolve_device(self, device: str):
        if device != "auto":
            return self.torch.device(device)
        return self.torch.device(
            "cuda" if self.torch.cuda.is_available() else "cpu"
        )

    def _train_one_epoch(self, loader, criterion, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        return total_loss / max(total_items, 1)

    def _evaluate_loader(self, loader) -> dict[str, float]:
        self.model.eval()
        y_true = []
        y_pred = []

        with self.torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                predictions = self.torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(predictions.tolist())
                y_true.extend(labels.numpy().tolist())

        return _classification_metrics(
            np.asarray(y_true, dtype=int),
            np.asarray(y_pred, dtype=int),
        )

    def _class_weights(self, labels: np.ndarray):
        labels = np.asarray(labels, dtype=int)
        counts = np.bincount(labels, minlength=2).astype(float)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (2.0 * counts)
        return self.torch.tensor(weights, dtype=self.torch.float32)


class _AbsoluteImageDataset:
    def __init__(self, metadata_df: pd.DataFrame, image_size: int) -> None:
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int):
        import torch

        row = self.metadata_df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, torch.tensor(int(row["label"]), dtype=torch.long)


class LocalMultimodalJudge:
    """Thin adapter for a local multimodal model that returns JSON.

    Pass a callable that accepts ``image_path`` and ``prompt`` and returns a
    JSON string like: {"is_anomaly": true, "confidence": 0.91}.
    """

    prompt = (
        "Analyze the line chart image. Return only JSON with keys "
        "is_anomaly, confidence, and rationale. Classify visual anomalies "
        "such as spikes, sudden drops, contextual shifts, or variance shifts."
    )

    def __init__(self, analyzer: Callable[[Path, str], str]) -> None:
        self.analyzer = analyzer

    def predict_one(self, image_path: Path) -> dict[str, object]:
        raw_response = self.analyzer(Path(image_path), self.prompt)
        return json.loads(raw_response)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_visual_detector(
    dataset_dir: Path,
    model_out: Path,
    config: VisualTrainingConfig,
) -> VisualAnomalyDetector:
    """Train the CNN detector and save it to disk."""

    detector = VisualAnomalyDetector(config=config)
    history = detector.fit(dataset_dir)
    detector.save(model_out)
    history_path = model_out.with_suffix(".history.json")
    with history_path.open("w") as file:
        json.dump(history, file, indent=2)
    return detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/visual_detector.pt"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VisualTrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        random_state=args.random_state,
    )
    train_visual_detector(args.dataset_dir, args.model_out, config)
    print(f"Saved visual detector: {args.model_out}")


if __name__ == "__main__":
    main()
