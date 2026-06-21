"""Synthetic time-series and chart-image dataset generator.

The module creates numeric windows and matching PNG chart images for a
supervised visual anomaly-detection experiment. Images are stored in the
classic image-classification layout:

    dataset/train/normal
    dataset/train/anomaly
    dataset/val/normal
    dataset/val/anomaly
    dataset/test/normal
    dataset/test/anomaly
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


LABEL_TO_NAME = {0: "normal", 1: "anomaly"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for the synthetic dataset generation process."""

    output_dir: Path = Path("dataset")
    n_series: int = 120
    series_length: int = 512
    window_size: int = 100
    stride: int = 25
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    anomaly_series_fraction: float = 0.8
    point_anomalies_per_series: int = 4
    contextual_anomalies_per_series: int = 2
    variance_shifts_per_series: int = 2
    min_variance_shift_length: int = 24
    max_variance_shift_length: int = 72
    image_width: int = 640
    image_height: int = 360
    dpi: int = 120
    random_state: int = 42
    plot_backend: str = "auto"
    balance_classes: bool = True
    windows_per_class_per_split: int | None = None
    show_anomaly_markers: bool = False
    overwrite: bool = False


def validate_config(config: GeneratorConfig) -> None:
    """Validate the config early, before any files are written."""

    if config.series_length <= 0:
        raise ValueError("series_length must be positive.")
    if config.window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if config.window_size > config.series_length:
        raise ValueError("window_size cannot exceed series_length.")
    if config.stride <= 0:
        raise ValueError("stride must be positive.")
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= config.val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if config.train_ratio + config.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be lower than 1.")
    if config.plot_backend not in {"auto", "matplotlib", "plotly", "pil"}:
        raise ValueError("plot_backend must be auto, matplotlib, plotly, or pil.")
    if (
        config.windows_per_class_per_split is not None
        and config.windows_per_class_per_split <= 0
    ):
        raise ValueError("windows_per_class_per_split must be positive.")


def generate_base_signal(
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a normal signal: trend + seasonality + heteroscedastic noise."""

    t = np.arange(length, dtype=float)
    slope = rng.uniform(-0.015, 0.02)
    intercept = rng.uniform(-2.0, 2.0)
    trend = intercept + slope * t

    long_period = rng.integers(70, 150)
    short_period = rng.integers(18, 45)
    long_amp = rng.uniform(0.8, 2.4)
    short_amp = rng.uniform(0.15, 0.7)
    long_phase = rng.uniform(0, 2 * math.pi)
    short_phase = rng.uniform(0, 2 * math.pi)

    seasonality = (
        long_amp * np.sin(2 * math.pi * t / long_period + long_phase)
        + short_amp * np.cos(2 * math.pi * t / short_period + short_phase)
    )
    baseline = trend + seasonality

    noise_scale = rng.uniform(0.08, 0.35)
    noise_envelope = 1.0 + 0.25 * np.sin(2 * math.pi * t / length)
    noise = rng.normal(0.0, noise_scale * noise_envelope, length)

    return baseline + noise, baseline


def _mark_anomaly(
    flags: np.ndarray,
    anomaly_types: np.ndarray,
    start: int,
    end: int,
    anomaly_type: str,
) -> None:
    """Mark an anomalous interval in the label arrays."""

    flags[start:end] = 1
    anomaly_types[start:end] = anomaly_type


def inject_anomalies(
    values: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject point, contextual, and variance-shift anomalies."""

    values = values.copy()
    length = len(values)
    flags = np.zeros(length, dtype=int)
    anomaly_types = np.full(length, "normal", dtype=object)

    if rng.random() > config.anomaly_series_fraction:
        return values, flags, anomaly_types

    robust_scale = np.percentile(np.abs(values - np.median(values)), 75)
    robust_scale = max(float(robust_scale), float(np.std(values)), 0.2)
    margin = max(5, config.window_size // 10)

    for _ in range(config.point_anomalies_per_series):
        idx = int(rng.integers(margin, length - margin))
        direction = int(rng.choice([-1, 1]))
        magnitude = rng.uniform(4.0, 8.0) * robust_scale
        values[idx] += direction * magnitude
        _mark_anomaly(flags, anomaly_types, idx, idx + 1, "point_spike")

    for _ in range(config.contextual_anomalies_per_series):
        segment_length = int(rng.integers(6, max(7, config.window_size // 4)))
        start = int(rng.integers(margin, length - margin - segment_length))
        end = start + segment_length
        direction = int(rng.choice([-1, 1]))

        # The segment remains smooth, but is implausible in its local context.
        offset = direction * rng.uniform(2.5, 4.5) * robust_scale
        smooth_shape = np.hanning(segment_length)
        smooth_shape = 0.5 + 0.5 * smooth_shape
        values[start:end] = baseline[start:end] + offset * smooth_shape
        values[start:end] += rng.normal(0.0, 0.15 * robust_scale, segment_length)
        _mark_anomaly(flags, anomaly_types, start, end, "contextual_shift")

    for _ in range(config.variance_shifts_per_series):
        segment_length = int(
            rng.integers(
                config.min_variance_shift_length,
                config.max_variance_shift_length + 1,
            )
        )
        segment_length = min(segment_length, length - 2 * margin)
        start = int(rng.integers(margin, length - margin - segment_length))
        end = start + segment_length
        multiplier = rng.uniform(3.0, 6.0)
        values[start:end] += rng.normal(
            0.0,
            multiplier * robust_scale,
            segment_length,
        )
        _mark_anomaly(flags, anomaly_types, start, end, "variance_shift")

    return values, flags, anomaly_types


def generate_series(
    series_id: int,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create one labeled synthetic time series."""

    values, baseline = generate_base_signal(config.series_length, rng)
    values, flags, anomaly_types = inject_anomalies(values, baseline, rng, config)

    return pd.DataFrame(
        {
            "series_id": series_id,
            "step": np.arange(config.series_length),
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=config.series_length,
                freq="h",
            ),
            "value": values,
            "baseline": baseline,
            "is_anomaly": flags,
            "anomaly_type": anomaly_types,
        }
    )


def split_series_ids(config: GeneratorConfig) -> dict[int, str]:
    """Split by series id to reduce leakage between train/val/test."""

    rng = np.random.default_rng(config.random_state)
    ids = np.arange(config.n_series)
    rng.shuffle(ids)

    n_train = int(round(config.n_series * config.train_ratio))
    n_val = int(round(config.n_series * config.val_ratio))
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])

    split_map: dict[int, str] = {}
    for series_id in ids:
        if series_id in train_ids:
            split_map[int(series_id)] = "train"
        elif series_id in val_ids:
            split_map[int(series_id)] = "val"
        else:
            split_map[int(series_id)] = "test"
    return split_map


def _prepare_output_dir(config: GeneratorConfig) -> None:
    """Create the image folder structure."""

    output_dir = config.output_dir
    if output_dir.exists() and config.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for label_name in LABEL_TO_NAME.values():
            (output_dir / split / label_name).mkdir(parents=True, exist_ok=True)


def render_window_chart(
    window_df: pd.DataFrame,
    output_path: Path,
    config: GeneratorConfig,
) -> None:
    """Render one time-series window as a PNG chart."""

    backend = config.plot_backend
    if backend == "auto":
        backend = "matplotlib" if _can_import_matplotlib() else "pil"

    if backend == "matplotlib":
        _render_with_matplotlib(window_df, output_path, config)
    elif backend == "plotly":
        _render_with_plotly(window_df, output_path, config)
    elif backend == "pil":
        _render_with_pillow(window_df, output_path, config)
    else:
        raise ValueError(f"Unsupported plot backend: {backend}")


def _can_import_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _render_with_matplotlib(
    window_df: pd.DataFrame,
    output_path: Path,
    config: GeneratorConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = config.image_width / config.dpi
    height = config.image_height / config.dpi
    fig, ax = plt.subplots(figsize=(width, height), dpi=config.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        window_df["step"].to_numpy(),
        window_df["value"].to_numpy(),
        color="#1f77b4",
        linewidth=2.0,
    )

    if config.show_anomaly_markers:
        anomaly_df = window_df[window_df["is_anomaly"] == 1]
        ax.scatter(
            anomaly_df["step"],
            anomaly_df["value"],
            color="#d62728",
            s=16,
            zorder=3,
        )

    ax.grid(color="#d9dee7", linewidth=0.7, alpha=0.65)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0.02, y=0.12)
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_with_plotly(
    window_df: pd.DataFrame,
    output_path: Path,
    config: GeneratorConfig,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly backend requires plotly. Install it or use "
            "--plot-backend matplotlib."
        ) from exc

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window_df["step"],
            y=window_df["value"],
            mode="lines",
            line={"color": "#1f77b4", "width": 2},
            showlegend=False,
        )
    )
    fig.update_layout(
        width=config.image_width,
        height=config.image_height,
        margin={"l": 4, "r": 4, "t": 4, "b": 4},
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={
            "visible": False,
            "showgrid": True,
            "gridcolor": "#d9dee7",
        },
        yaxis={
            "visible": False,
            "showgrid": True,
            "gridcolor": "#d9dee7",
        },
    )
    try:
        fig.write_image(output_path)
    except ValueError as exc:
        raise RuntimeError(
            "Plotly PNG export requires kaleido. Install kaleido or use "
            "--plot-backend matplotlib."
        ) from exc


def _render_with_pillow(
    window_df: pd.DataFrame,
    output_path: Path,
    config: GeneratorConfig,
) -> None:
    """Small dependency-light chart renderer used as an emergency fallback."""

    width = config.image_width
    height = config.image_height
    margin = 18
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for i in range(1, 5):
        y = margin + i * (height - 2 * margin) // 5
        draw.line((margin, y, width - margin, y), fill="#d9dee7", width=1)
    for i in range(1, 7):
        x = margin + i * (width - 2 * margin) // 7
        draw.line((x, margin, x, height - margin), fill="#edf0f5", width=1)

    y_values = window_df["value"].to_numpy(dtype=float)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_pad = max((y_max - y_min) * 0.12, 1e-6)
    y_min -= y_pad
    y_max += y_pad

    points = []
    for idx, y_value in enumerate(y_values):
        x_norm = idx / max(len(y_values) - 1, 1)
        y_norm = (float(y_value) - y_min) / max(y_max - y_min, 1e-6)
        x_pixel = margin + x_norm * (width - 2 * margin)
        y_pixel = height - margin - y_norm * (height - 2 * margin)
        points.append((int(round(x_pixel)), int(round(y_pixel))))

    if len(points) > 1:
        draw.line(points, fill="#1f77b4", width=3, joint="curve")
    image.save(output_path)


def iter_window_starts(length: int, window_size: int, stride: int) -> Iterable[int]:
    """Yield deterministic sliding-window starts."""

    last_start = length - window_size
    start = 0
    while start <= last_start:
        yield start
        start += stride


def balance_window_metadata(
    metadata_df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample a 50/50 normal/anomaly window mix inside every data split."""

    if not config.balance_classes:
        return metadata_df.reset_index(drop=True)

    balanced_frames = []
    summary_rows = []

    for split in SPLITS:
        split_df = metadata_df[metadata_df["split"] == split]
        if split_df.empty:
            continue

        counts = split_df["label"].value_counts()
        normal_count = int(counts.get(0, 0))
        anomaly_count = int(counts.get(1, 0))

        if normal_count == 0 or anomaly_count == 0:
            raise ValueError(
                f"Cannot balance split '{split}'. Found {normal_count} normal "
                f"windows and {anomaly_count} anomaly windows. Increase "
                "n_series, reduce anomaly density, or disable balancing with "
                "--no-balance-classes."
            )

        if config.windows_per_class_per_split is None:
            target_count = min(normal_count, anomaly_count)
        else:
            target_count = config.windows_per_class_per_split
            if target_count > normal_count or target_count > anomaly_count:
                raise ValueError(
                    f"Cannot sample {target_count} windows per class for "
                    f"split '{split}'. Available: {normal_count} normal and "
                    f"{anomaly_count} anomaly."
                )

        for label in LABEL_TO_NAME:
            label_df = split_df[split_df["label"] == label]
            random_state = int(rng.integers(0, np.iinfo(np.int32).max))
            balanced_frames.append(
                label_df.sample(n=target_count, random_state=random_state)
            )

        summary_rows.append(
            {
                "split": split,
                "selected_normal": target_count,
                "selected_anomaly": target_count,
                "available_normal": normal_count,
                "available_anomaly": anomaly_count,
            }
        )

    balanced_df = pd.concat(balanced_frames, ignore_index=True)
    balanced_df = balanced_df.sort_values(
        ["split", "label", "series_id", "start"],
    ).reset_index(drop=True)
    balanced_df["is_balanced_sample"] = 1

    summary_df = pd.DataFrame(summary_rows)
    summary_path = config.output_dir / "balance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return balanced_df


def build_dataset(config: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate numeric data, chart images, and metadata files."""

    validate_config(config)
    _prepare_output_dir(config)

    rng = np.random.default_rng(config.random_state)
    split_map = split_series_ids(config)
    numeric_frames = []
    series_frames = {}
    metadata_records = []

    for series_id in range(config.n_series):
        series_df = generate_series(series_id, config, rng)
        split = split_map[series_id]
        series_df["split"] = split
        numeric_frames.append(series_df)
        series_frames[series_id] = series_df

        for window_index, start in enumerate(
            iter_window_starts(
                config.series_length,
                config.window_size,
                config.stride,
            )
        ):
            end = start + config.window_size
            window_df = series_df.iloc[start:end]
            anomaly_count = int(window_df["is_anomaly"].sum())
            label = int(anomaly_count > 0)
            label_name = LABEL_TO_NAME[label]
            image_name = (
                f"series_{series_id:04d}_window_{window_index:04d}_"
                f"start_{start:05d}.png"
            )
            relative_path = Path(split) / label_name / image_name
            metadata_records.append(
                {
                    "series_id": series_id,
                    "window_index": window_index,
                    "start": start,
                    "end": end,
                    "split": split,
                    "label": label,
                    "label_name": label_name,
                    "anomaly_count": anomaly_count,
                    "anomaly_ratio": anomaly_count / config.window_size,
                    "image_path": relative_path.as_posix(),
                }
            )

    numeric_df = pd.concat(numeric_frames, ignore_index=True)
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df = balance_window_metadata(metadata_df, config, rng)

    for row in metadata_df.itertuples(index=False):
        window_df = series_frames[int(row.series_id)].iloc[int(row.start) : int(row.end)]
        image_path = config.output_dir / row.image_path
        render_window_chart(window_df, image_path, config)

    numeric_df.to_csv(config.output_dir / "numeric_data.csv", index=False)
    metadata_df.to_csv(config.output_dir / "window_metadata.csv", index=False)
    _save_config(config)

    return numeric_df, metadata_df


def _save_config(config: GeneratorConfig) -> None:
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    with (config.output_dir / "generation_config.json").open("w") as file:
        json.dump(config_dict, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--n-series", type=int, default=120)
    parser.add_argument("--series-length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-balance-classes",
        action="store_true",
        help="Do not enforce a 50/50 normal/anomaly ratio in each split.",
    )
    parser.add_argument(
        "--windows-per-class-per-split",
        type=int,
        default=None,
        help=(
            "Optional exact number of normal and anomaly chart windows to keep "
            "inside each split."
        ),
    )
    parser.add_argument(
        "--plot-backend",
        choices=["auto", "matplotlib", "plotly", "pil"],
        default="auto",
    )
    parser.add_argument("--show-anomaly-markers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeneratorConfig(
        output_dir=args.output_dir,
        n_series=args.n_series,
        series_length=args.series_length,
        window_size=args.window_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        balance_classes=not args.no_balance_classes,
        windows_per_class_per_split=args.windows_per_class_per_split,
        plot_backend=args.plot_backend,
        show_anomaly_markers=args.show_anomaly_markers,
        overwrite=args.overwrite,
    )
    numeric_df, metadata_df = build_dataset(config)
    print(f"Saved {len(numeric_df):,} time points to {config.output_dir}.")
    print(f"Saved {len(metadata_df):,} chart windows to {config.output_dir}.")
    print(metadata_df.groupby(["split", "label_name"]).size().to_string())


if __name__ == "__main__":
    main()
