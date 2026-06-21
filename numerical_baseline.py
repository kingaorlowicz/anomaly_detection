"""Numerical anomaly-detection baseline for generated time-series windows."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "q05",
    "q25",
    "q75",
    "q95",
    "iqr",
    "slope",
    "intercept",
    "residual_std",
    "residual_max_abs",
    "diff_mean",
    "diff_std",
    "diff_max_abs",
    "diff_abs_mean",
    "energy",
    "skew",
    "kurtosis",
    "zero_crossings",
    "max_zscore",
]


class WindowFeatureExtractor:
    """Extract compact statistical features from each time-series window."""

    feature_columns = FEATURE_COLUMNS

    def transform(
        self,
        numeric_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one feature row per image/window metadata row."""

        grouped = {
            int(series_id): frame.sort_values("step")["value"].to_numpy()
            for series_id, frame in numeric_df.groupby("series_id")
        }

        rows = []
        for record in metadata_df.itertuples(index=False):
            series_values = grouped[int(record.series_id)]
            window = series_values[int(record.start) : int(record.end)]
            rows.append(self.extract(window))
        return pd.DataFrame(rows, columns=self.feature_columns)

    def extract(self, values: np.ndarray) -> dict[str, float]:
        """Extract robust shape and distribution features for one window."""

        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Window values must be a 1D array with >= 2 items.")

        n = values.size
        diffs = np.diff(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        safe_std = max(std, 1e-8)
        centered = values - mean

        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, values, deg=1)
        trend = slope * x + intercept
        residuals = values - trend
        q05, q25, q75, q95 = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
        zscores = np.abs(centered / safe_std)

        return {
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "median": float(np.median(values)),
            "q05": float(q05),
            "q25": float(q25),
            "q75": float(q75),
            "q95": float(q95),
            "iqr": float(q75 - q25),
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_std": float(np.std(residuals, ddof=1)),
            "residual_max_abs": float(np.max(np.abs(residuals))),
            "diff_mean": float(np.mean(diffs)),
            "diff_std": _safe_std(diffs),
            "diff_max_abs": float(np.max(np.abs(diffs))),
            "diff_abs_mean": float(np.mean(np.abs(diffs))),
            "energy": float(np.mean(values**2)),
            "skew": float(np.mean((centered / safe_std) ** 3)),
            "kurtosis": float(np.mean((centered / safe_std) ** 4) - 3.0),
            "zero_crossings": float(np.sum(np.diff(np.signbit(centered)) != 0)),
            "max_zscore": float(np.max(zscores)),
        }


@dataclass
class NumericalAnomalyBaseline:
    """Window-level baseline using scikit-learn or a robust NumPy fallback."""

    model_type: str = "random_forest"
    random_state: int = 42
    threshold_quantile: float = 0.98
    estimator: Any | None = None
    feature_columns: list[str] | None = None
    robust_center_: np.ndarray | None = None
    robust_scale_: np.ndarray | None = None
    score_threshold_: float | None = None
    score_center_: float | None = None
    score_scale_: float | None = None

    def fit(self, features: pd.DataFrame, y: np.ndarray) -> "NumericalAnomalyBaseline":
        """Fit the configured baseline on window features."""

        self.feature_columns = list(features.columns)
        x_values = features.to_numpy(dtype=float)
        y = np.asarray(y, dtype=int)

        if self.model_type == "random_forest":
            self.estimator = self._fit_random_forest(x_values, y)
        elif self.model_type == "isolation_forest":
            self.estimator = self._fit_isolation_forest(x_values, y)
            self._calibrate_scores(x_values)
        elif self.model_type == "robust_threshold":
            self._fit_robust_threshold(x_values, y)
        else:
            raise ValueError(
                "model_type must be random_forest, isolation_forest, "
                "or robust_threshold."
            )
        return self

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels: 0 = normal, 1 = anomaly."""

        probabilities = self.predict_proba(features)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return anomaly probabilities or calibrated anomaly scores."""

        x_values = self._prepare_features(features)

        if self.model_type == "random_forest":
            if self.estimator is None:
                raise RuntimeError("Model has not been fitted.")
            probabilities = self.estimator.predict_proba(x_values)
            class_to_index = {
                int(label): idx for idx, label in enumerate(self.estimator.classes_)
            }
            if 1 not in class_to_index:
                return np.zeros(len(x_values), dtype=float)
            return probabilities[:, class_to_index.get(1, 0)]

        if self.model_type == "isolation_forest":
            if self.estimator is None:
                raise RuntimeError("Model has not been fitted.")
            raw_scores = -self.estimator.decision_function(x_values)
            return self._scores_to_probabilities(raw_scores)

        if self.model_type == "robust_threshold":
            raw_scores = self._robust_scores(x_values)
            return self._scores_to_probabilities(raw_scores)

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def save(self, path: Path) -> None:
        """Serialize the fitted baseline with pickle."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: Path) -> "NumericalAnomalyBaseline":
        """Load a fitted baseline from disk."""

        with path.open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected model type in {path}.")
        return model

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("Model has not been fitted.")
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        return features[self.feature_columns].to_numpy(dtype=float)

    def _fit_random_forest(self, x_values: np.ndarray, y: np.ndarray) -> Any:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise ImportError(
                "random_forest requires scikit-learn. Install scikit-learn "
                "or use --model-type robust_threshold."
            ) from exc

        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        estimator.fit(x_values, y)
        return estimator

    def _fit_isolation_forest(self, x_values: np.ndarray, y: np.ndarray) -> Any:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise ImportError(
                "isolation_forest requires scikit-learn. Install scikit-learn "
                "or use --model-type robust_threshold."
            ) from exc

        normal_values = x_values[y == 0]
        if len(normal_values) == 0:
            raise ValueError("Isolation Forest needs at least one normal window.")
        contamination = float(np.clip(np.mean(y), 0.01, 0.45))
        estimator = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        estimator.fit(normal_values)
        return estimator

    def _fit_robust_threshold(self, x_values: np.ndarray, y: np.ndarray) -> None:
        normal_values = x_values[y == 0]
        if len(normal_values) == 0:
            raise ValueError("robust_threshold needs at least one normal window.")

        self.robust_center_ = np.median(normal_values, axis=0)
        q25 = np.quantile(normal_values, 0.25, axis=0)
        q75 = np.quantile(normal_values, 0.75, axis=0)
        self.robust_scale_ = np.maximum(q75 - q25, 1e-6)
        self._calibrate_scores(normal_values)

    def _robust_scores(self, x_values: np.ndarray) -> np.ndarray:
        if self.robust_center_ is None or self.robust_scale_ is None:
            raise RuntimeError("Robust threshold model has not been fitted.")
        zscores = np.abs((x_values - self.robust_center_) / self.robust_scale_)
        top_k = min(5, zscores.shape[1])
        return np.mean(np.sort(zscores, axis=1)[:, -top_k:], axis=1)

    def _calibrate_scores(self, x_values: np.ndarray) -> None:
        if self.model_type == "isolation_forest":
            if self.estimator is None:
                raise RuntimeError("Isolation Forest model has not been fitted.")
            raw_scores = -self.estimator.decision_function(x_values)
        else:
            raw_scores = self._robust_scores(x_values)

        self.score_threshold_ = float(
            np.quantile(raw_scores, self.threshold_quantile)
        )
        self.score_center_ = float(np.median(raw_scores))
        mad = np.median(np.abs(raw_scores - self.score_center_))
        self.score_scale_ = float(max(mad * 1.4826, 1e-6))

    def _scores_to_probabilities(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.score_threshold_ is None:
            raise RuntimeError("Score calibration is missing.")
        if self.score_scale_ is None:
            raise RuntimeError("Score calibration is missing.")
        logits = (raw_scores - self.score_threshold_) / self.score_scale_
        logits = np.clip(logits, -40, 40)
        return 1.0 / (1.0 + np.exp(-logits))


def _safe_std(values: np.ndarray) -> float:
    """Return a finite sample standard deviation for very short arrays."""

    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def load_window_dataset(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load numeric data and transform all windows into feature vectors."""

    dataset_dir = Path(dataset_dir)
    numeric_path = dataset_dir / "numeric_data.csv"
    metadata_path = dataset_dir / "window_metadata.csv"

    if not numeric_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Expected numeric_data.csv and window_metadata.csv in "
            f"{dataset_dir}."
        )

    numeric_df = pd.read_csv(numeric_path)
    metadata_df = pd.read_csv(metadata_path)
    extractor = WindowFeatureExtractor()
    features = extractor.transform(numeric_df, metadata_df)
    y = metadata_df["label"].to_numpy(dtype=int)
    return features, y, metadata_df


def train_baseline(
    dataset_dir: Path,
    model_out: Path,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> NumericalAnomalyBaseline:
    """Train the baseline on the train split and save it."""

    features, y, metadata_df = load_window_dataset(dataset_dir)
    train_mask = metadata_df["split"].eq("train").to_numpy()
    model = NumericalAnomalyBaseline(
        model_type=model_type,
        random_state=random_state,
    )
    model.fit(features.loc[train_mask], y[train_mask])
    model.save(model_out)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/numerical_baseline.pkl"),
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "isolation_forest", "robust_threshold"],
        default="random_forest",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_baseline(
        dataset_dir=args.dataset_dir,
        model_out=args.model_out,
        model_type=args.model_type,
        random_state=args.random_state,
    )
    print(f"Saved numerical baseline: {args.model_out}")
    print(f"Model type: {model.model_type}")


if __name__ == "__main__":
    main()
