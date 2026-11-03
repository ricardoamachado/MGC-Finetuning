"""Plot the training results stored in a checkpoint folder as publication-ready PDF figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import MaxNLocator

# Categorical slots 1-3 of the reference palette: they stay distinguishable under
# color-vision deficiency for every pair, so they can share a plot.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# Distinct markers keep overlapping lines apart in grayscale prints.
SERIES_MARKERS = ("o", "s", "^")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#a3a29d"
GRID_COLOR = "#e4e3df"
SURFACE = "#ffffff"
# Single-hue blue ramp (light -> dark) for the confusion matrix.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update(
    {
        # TrueType fonts stay editable and avoid the Type 3 fonts rejected by many publishers.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.edgecolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def _best_epoch(history: pl.DataFrame, per_class: pl.DataFrame | None) -> int:
    """Epoch of the saved checkpoint: the last epoch with the highest validation accuracy."""
    if per_class is not None and "epoch" in per_class.columns:
        return int(per_class["epoch"][0])
    accuracy = history["validation_accuracy"].to_numpy()
    return int(history["epoch"][int(np.flatnonzero(accuracy == accuracy.max())[-1])])


def _mark_best_epoch(axis: plt.Axes, best_epoch: int) -> None:
    axis.axvline(best_epoch, color=TEXT_SECONDARY, linestyle="--", linewidth=0.9, zorder=1)
    axis.annotate(
        f"Best epoch ({best_epoch})",
        xy=(best_epoch, 1.0),
        xycoords=("data", "axes fraction"),
        # Placed just above the plot area so it never collides with the curves.
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color=TEXT_SECONDARY,
    )


def _epoch_axis(axis: plt.Axes, epochs: np.ndarray) -> None:
    axis.set_xlabel("Epoch")
    axis.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="x", visible=False)


def plot_curves(
    history: pl.DataFrame,
    columns: dict[str, str],
    ylabel: str,
    best_epoch: int,
    output_path: Path,
) -> None:
    """Line plot of several history columns against the epoch, sharing one y-axis."""
    epochs = history["epoch"].to_numpy()
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    for color, marker, (column, label) in zip(SERIES_COLORS, SERIES_MARKERS, columns.items()):
        axis.plot(epochs, history[column].to_numpy(), color=color, marker=marker, label=label, zorder=3)
    _epoch_axis(axis, epochs)
    axis.set_ylabel(ylabel)
    _mark_best_epoch(axis, best_epoch)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(columns))
    figure.savefig(output_path)
    plt.close(figure)


def plot_confusion_matrix(confusion: pl.DataFrame, output_path: Path) -> None:
    """Heatmap of the row-normalized confusion matrix with every cell annotated."""
    class_names = confusion["true_label"].to_list()
    matrix = confusion.select(class_names).to_numpy()
    size = max(4.5, 0.48 * len(class_names) + 1.8)
    figure, axis = plt.subplots(figsize=(size + 0.8, size))
    image = axis.imshow(matrix, cmap=SEQUENTIAL_BLUE, vmin=0.0, vmax=1.0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.tick_params(length=0)
    # White cell borders keep neighbouring cells apart.
    axis.set_xticks(np.arange(-0.5, len(class_names)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(class_names)), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.5)
    axis.tick_params(which="minor", length=0)
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column]
            red, green, blue = to_rgb(SEQUENTIAL_BLUE(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                # Zeros are de-emphasized so the actual confusions stand out.
                color=SURFACE if luminance < 0.5 else TEXT_PRIMARY if value > 0 else TEXT_MUTED,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Fraction of true-class samples")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    figure.savefig(output_path)
    plt.close(figure)


def plot_per_class_metrics(per_class: pl.DataFrame, output_path: Path) -> None:
    """Grouped bars of precision, recall and F1 for each class."""
    metrics = {"precision": "Precision", "recall": "Recall", "f1": "F1 score"}
    class_names = per_class["class"].to_list()
    positions = np.arange(len(class_names))
    bar_width = 0.8 / len(metrics)
    figure, axis = plt.subplots(figsize=(max(6.0, 0.62 * len(class_names) + 1.2), 3.6))
    for index, (color, (column, label)) in enumerate(zip(SERIES_COLORS, metrics.items())):
        offset = (index - (len(metrics) - 1) / 2) * bar_width
        axis.bar(
            positions + offset,
            per_class[column].to_numpy(),
            width=bar_width,
            color=color,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=label,
            zorder=3,
        )
    axis.set_xticks(positions, class_names, rotation=45, ha="right", rotation_mode="anchor")
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.6, len(class_names) - 0.4)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(metrics))
    figure.savefig(output_path)
    plt.close(figure)


def plot_results(checkpoint: Path, output_dir: Path) -> list[Path]:
    """Create every figure the checkpoint's CSV files allow and return their paths."""
    history_path = checkpoint / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"history.csv não encontrado em {checkpoint}.")
    history = pl.read_csv(history_path)
    per_class_path = checkpoint / "per_class_metrics.csv"
    confusion_path = checkpoint / "confusion_matrix.csv"
    per_class = pl.read_csv(per_class_path) if per_class_path.is_file() else None
    best_epoch = _best_epoch(history, per_class)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plot_curves(
        history,
        {"train_loss": "Training", "validation_loss": "Validation"},
        "Cross-entropy loss",
        best_epoch,
        output_dir / "loss.pdf",
    )
    saved.append(output_dir / "loss.pdf")
    plot_curves(
        history,
        {"train_accuracy": "Training", "validation_accuracy": "Validation"},
        "Accuracy",
        best_epoch,
        output_dir / "accuracy.pdf",
    )
    saved.append(output_dir / "accuracy.pdf")

    macro_columns = {
        "validation_precision_macro": "Precision (macro)",
        "validation_recall_macro": "Recall (macro)",
        "validation_f1_macro": "F1 score (macro)",
    }
    if set(macro_columns).issubset(history.columns):
        plot_curves(history, macro_columns, "Validation score", best_epoch, output_dir / "macro_metrics.pdf")
        saved.append(output_dir / "macro_metrics.pdf")
    else:
        print("history.csv sem métricas macro-averaged; macro_metrics.pdf não foi gerado.")

    if confusion_path.is_file():
        plot_confusion_matrix(pl.read_csv(confusion_path), output_dir / "confusion_matrix.pdf")
        saved.append(output_dir / "confusion_matrix.pdf")
    else:
        print("confusion_matrix.csv não encontrado; confusion_matrix.pdf não foi gerado.")

    if per_class is not None:
        plot_per_class_metrics(per_class, output_dir / "per_class_metrics.pdf")
        saved.append(output_dir / "per_class_metrics.pdf")
    else:
        print("per_class_metrics.csv não encontrado; per_class_metrics.pdf não foi gerado.")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera as figuras em PDF dos resultados de um checkpoint treinado.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pasta do checkpoint com history.csv, per_class_metrics.csv e confusion_matrix.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Pasta onde os PDFs são salvos. Padrão: {checkpoint}/figures.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.checkpoint / "figures"
    for path in plot_results(args.checkpoint, output_dir):
        print(f"figura salva: {path}")


if __name__ == "__main__":
    main()
