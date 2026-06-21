"""Evaluate and compare numerical and visual anomaly-detection approaches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from numerical_baseline import NumericalAnomalyBaseline, load_window_dataset
from visual_anomaly_detector import (
    VisualAnomalyDetector,
    VisualTrainingConfig,
    train_visual_detector,
)


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary metrics without requiring scikit-learn."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

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
        "f1_score": f1,
        "roc_auc": roc_auc(y_true, y_score),
    }


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC using average ranks with tie handling."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(y_score)
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0

    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def evaluate_numerical(
    dataset_dir: Path,
    model_path: Path,
    train_missing: bool,
    model_type: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load or train the numerical baseline and score the test split."""

    features, y, metadata_df = load_window_dataset(dataset_dir)
    if model_path.exists():
        model = NumericalAnomalyBaseline.load(model_path)
    elif train_missing:
        train_mask = metadata_df["split"].eq("train").to_numpy()
        model = NumericalAnomalyBaseline(
            model_type=model_type,
            random_state=random_state,
        )
        model.fit(features.loc[train_mask], y[train_mask])
        model.save(model_path)
    else:
        raise FileNotFoundError(
            f"Missing numerical model at {model_path}. Use --train-missing."
        )

    test_mask = metadata_df["split"].eq("test").to_numpy()
    test_features = features.loc[test_mask]
    test_meta = metadata_df.loc[test_mask].reset_index(drop=True)
    y_true = y[test_mask]
    y_score = model.predict_proba(test_features)
    return y_true, y_score, test_meta


def evaluate_visual(
    dataset_dir: Path,
    model_path: Path,
    train_missing: bool,
    epochs: int,
    batch_size: int,
    image_size: int,
    device: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load or train the visual detector and score the test split."""

    if model_path.exists():
        detector = VisualAnomalyDetector.load(model_path, device=device)
    elif train_missing:
        config = VisualTrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            random_state=random_state,
        )
        detector = train_visual_detector(dataset_dir, model_path, config)
    else:
        raise FileNotFoundError(
            f"Missing visual model at {model_path}. Use --train-missing."
        )

    return detector.predict_dataset_split(dataset_dir, split="test")


def build_report(
    numerical_result: tuple[np.ndarray, np.ndarray, pd.DataFrame],
    visual_result: tuple[np.ndarray, np.ndarray, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create metric table and row-level comparison table."""

    y_num, num_score, num_meta = numerical_result
    y_vis, vis_score, vis_meta = visual_result

    if not np.array_equal(y_num, y_vis):
        raise ValueError("Numerical and visual test labels are not aligned.")
    if not num_meta["image_path"].equals(vis_meta["image_path"]):
        raise ValueError("Numerical and visual test metadata are not aligned.")

    report = pd.DataFrame(
        [
            {"method": "numerical", **binary_metrics(y_num, num_score)},
            {"method": "visual", **binary_metrics(y_vis, vis_score)},
        ]
    )

    comparison = num_meta.copy()
    comparison["y_true"] = y_num
    comparison["numerical_score"] = num_score
    comparison["visual_score"] = vis_score
    comparison["numerical_pred"] = (num_score >= 0.5).astype(int)
    comparison["visual_pred"] = (vis_score >= 0.5).astype(int)
    comparison["numerical_correct"] = (
        comparison["numerical_pred"] == comparison["y_true"]
    )
    comparison["visual_correct"] = (
        comparison["visual_pred"] == comparison["y_true"]
    )
    return report, comparison


def save_case_sheets(
    dataset_dir: Path,
    comparison_df: pd.DataFrame,
    output_dir: Path,
    max_cases: int = 12,
) -> None:
    """Save contact sheets for cases where one method beats the other."""

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_better = comparison_df[
        comparison_df["visual_correct"] & ~comparison_df["numerical_correct"]
    ]
    numerical_better = comparison_df[
        comparison_df["numerical_correct"] & ~comparison_df["visual_correct"]
    ]

    _make_contact_sheet(
        dataset_dir,
        visual_better.head(max_cases),
        output_dir / "visual_better_than_numerical.png",
        "Visual correct, numerical wrong",
    )
    _make_contact_sheet(
        dataset_dir,
        numerical_better.head(max_cases),
        output_dir / "numerical_better_than_visual.png",
        "Numerical correct, visual wrong",
    )


def _make_contact_sheet(
    dataset_dir: Path,
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    tile_width = 300
    tile_height = 230
    header_height = 34
    columns = 3
    rows_count = max(int(np.ceil(len(rows) / columns)), 1)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows_count * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 10), title, fill="#111111", font=font)

    if rows.empty:
        draw.text((12, header_height + 16), "No cases found.", fill="#444444")
        sheet.save(output_path)
        return

    for idx, row in enumerate(rows.itertuples(index=False)):
        col = idx % columns
        row_idx = idx // columns
        x0 = col * tile_width
        y0 = header_height + row_idx * tile_height
        image_path = dataset_dir / row.image_path
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width - 24, tile_height - 60))
        x_img = x0 + (tile_width - image.width) // 2
        y_img = y0 + 8
        sheet.paste(image, (x_img, y_img))

        caption = (
            f"y={row.y_true} num={row.numerical_score:.2f} "
            f"vis={row.visual_score:.2f}"
        )
        draw.text((x0 + 12, y0 + tile_height - 36), caption, fill="#111111")

    sheet.save(output_path)


def run_evaluation(args: argparse.Namespace) -> pd.DataFrame:
    dataset_dir = Path(args.dataset_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    numerical_result = evaluate_numerical(
        dataset_dir=dataset_dir,
        model_path=Path(args.numerical_model),
        train_missing=args.train_missing,
        model_type=args.numerical_model_type,
        random_state=args.random_state,
    )
    visual_result = evaluate_visual(
        dataset_dir=dataset_dir,
        model_path=Path(args.visual_model),
        train_missing=args.train_missing,
        epochs=args.visual_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        random_state=args.random_state,
    )

    report, comparison = build_report(numerical_result, visual_result)
    report_path = artifacts_dir / "evaluation_report.csv"
    comparison_path = artifacts_dir / "prediction_comparison.csv"
    report.to_csv(report_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    save_case_sheets(dataset_dir, comparison, artifacts_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--numerical-model",
        type=Path,
        default=Path("artifacts/numerical_baseline.pkl"),
    )
    parser.add_argument(
        "--visual-model",
        type=Path,
        default=Path("artifacts/visual_detector.pt"),
    )
    parser.add_argument(
        "--numerical-model-type",
        choices=["random_forest", "isolation_forest", "robust_threshold"],
        default="random_forest",
    )
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--visual-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_evaluation(args)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
