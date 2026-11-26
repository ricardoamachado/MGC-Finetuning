"""Predict the top-k genres of one audio file, or of a whole dataset split, using a fine-tuned checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset

from .data import (
    GTZAN_INDEX_URL,
    FMADataset,
    GTZANDataset,
    build_mel_transform,
    load_fma_track_info,
    load_gtzan_track_info,
)
from .metrics import macro_metrics
from .model import FINETUNER_CLASSES, autocast_context, checkpoint_model

DATASETS = ("fma_small", "fma_medium", "gtzan")
PREDICTIONS_DIR = Path("predictions")
# GTZAN names the training split "train", while the FMA metadata uses "training".
SPLIT_NAMES = {
    "gtzan": {"train": "train", "validation": "validation", "test": "test"},
    "fma": {"train": "training", "validation": "validation", "test": "test"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prevê os gêneros mais prováveis de um arquivo de áudio ou, com --dataset, "
            "de todas as faixas de um split, salvando o resultado em CSV."
        )
    )
    parser.add_argument("audio", type=Path, nargs="?", help="Arquivo de áudio a ser classificado (omitir com --dataset).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pasta do checkpoint, ex.: checkpoints/ast-gtzan-head ou checkpoints/bat-fma_small-lora.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Quantidade de gêneros mais prováveis a reportar.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Prevê em lote todas as faixas de um split do dataset em vez de um único arquivo.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Split do dataset usado com --dataset.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("datasets/fma_tracks.csv"),
        help="CSV de metadados do FMA (ignorado no GTZAN).",
    )
    parser.add_argument(
        "--gtzan-index",
        type=Path,
        default=None,
        help=f"Índice de artistas e títulos do GTZAN ({GTZAN_INDEX_URL}). Padrão: {{data-dir}}/GTZAN/index.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de saída do modo em lote. Padrão: predictions/{nome do checkpoint}_{dataset}_{split}.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do batch do modo em lote.")
    args = parser.parse_args()
    if (args.audio is None) == (args.dataset is None):
        parser.error("informe um arquivo de áudio ou --dataset, mas não ambos.")
    if args.top_k < 1:
        parser.error("--top-k deve ser pelo menos 1.")
    if args.output is None and args.dataset is not None:
        args.output = PREDICTIONS_DIR / f"{args.checkpoint.resolve().name}_{args.dataset}_{args.split}.csv"
    return args


def load_checkpoint(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, list[str]]:
    """Load the fine-tuned model, its evaluation Mel transform and its class names."""
    labels = json.loads((checkpoint / "labels.json").read_text(encoding="utf-8"))
    model_key = checkpoint_model(checkpoint)
    mel_transform = build_mel_transform(model_key, checkpoint, is_training=False)
    model = FINETUNER_CLASSES[model_key].from_pretrained(checkpoint, num_labels=len(labels)).to(device).eval()
    return model, mel_transform, labels


def predict_top_k(
    model: torch.nn.Module, mel_spectrograms: torch.Tensor, top_k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``top_k`` probabilities and class indices of each spectrogram, most likely first."""
    with torch.inference_mode(), autocast_context(device):
        probabilities = model(mel_spectrograms.to(device))["probabilities"]
    probabilities = probabilities.float()
    values, indices = probabilities.topk(min(top_k, probabilities.shape[-1]), dim=-1)
    return values.cpu(), indices.cpu()


def predict_file(args: argparse.Namespace, device: torch.device) -> None:
    model, mel_transform, labels = load_checkpoint(args.checkpoint, device)
    waveform, sample_rate = torchaudio.load(args.audio)
    mel_spectrogram = mel_transform(waveform, sample_rate).unsqueeze(0)
    values, indices = predict_top_k(model, mel_spectrogram, args.top_k, device)
    top_k = [
        {"label": labels[int(index)], "probability": float(value)}
        for value, index in zip(values[0], indices[0])
    ]
    result = {"label": top_k[0]["label"], "probability": top_k[0]["probability"], "top_k": top_k}
    print(json.dumps(result, ensure_ascii=False))


def build_dataset(
    args: argparse.Namespace, labels: list[str], mel_transform: torch.nn.Module
) -> tuple[Dataset, list[dict[str, object]]]:
    """Build the evaluation dataset and the identification of each of its tracks, in dataset order."""
    try:
        if args.dataset == "gtzan":
            dataset = GTZANDataset(
                args.data_dir,
                split=SPLIT_NAMES["gtzan"][args.split],
                class_names=labels,
                mel_transform=mel_transform,
            )
            tracks = [
                {"track_id": path.stem, "path": path.as_posix()} for path in dataset.audio_paths
            ]
        else:
            dataset = FMADataset(
                args.data_dir,
                args.metadata,
                subset=args.dataset.removeprefix("fma_"),
                split=SPLIT_NAMES["fma"][args.split],
                class_names=labels,
                mel_transform=mel_transform,
            )
            tracks = [
                {"track_id": str(track_id), "path": dataset.audio_path(track_id).as_posix()}
                for track_id in dataset.track_ids
            ]
    except ValueError as error:
        # The datasets raise ValueError when a track's genre is not among the checkpoint's classes.
        raise SystemExit(
            f"O checkpoint {args.checkpoint} não foi treinado com os gêneros de {args.dataset}: {error}"
        ) from error
    return dataset, tracks


def add_track_info(predictions: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    """Insert the ``artist`` and ``title`` of each track after ``track_id``; unknown values stay empty."""
    if args.dataset == "gtzan":
        index_path = args.gtzan_index or args.data_dir / "GTZAN" / "index.txt"
        if index_path.is_file():
            info = load_gtzan_track_info(index_path)
        else:
            print(f"aviso: {index_path} não encontrado; baixe-o de {GTZAN_INDEX_URL} para incluir artista e título.")
            info = pl.DataFrame(schema={"track_id": pl.String, "artist": pl.String, "title": pl.String})
    else:
        info = load_fma_track_info(args.metadata)
    info = info.unique("track_id", keep="first")  # Guards against repeated index entries.
    predictions = predictions.join(info, on="track_id", how="left", maintain_order="left")
    first = ["track_id", "artist", "title"]
    return predictions.select(*first, pl.exclude(first))


def predict_dataset(args: argparse.Namespace, device: torch.device) -> None:
    model, mel_transform, labels = load_checkpoint(args.checkpoint, device)
    dataset, tracks = build_dataset(args, labels, mel_transform)
    top_k = min(args.top_k, len(labels))
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    pending: list[tuple[dict[str, object], dict[str, torch.Tensor]]] = []

    def flush() -> None:
        if not pending:
            return
        mel_spectrograms = torch.stack([item["input_values"] for _, item in pending])
        values, indices = predict_top_k(model, mel_spectrograms, top_k, device)
        for (track, item), track_values, track_indices in zip(pending, values, indices):
            row = {**track, "genre": labels[int(item["labels"])]}
            for rank, (value, index) in enumerate(zip(track_values, track_indices), start=1):
                row[f"top{rank}_genre"] = labels[int(index)]
                row[f"top{rank}_probability"] = float(value)
            rows.append(row)
        pending.clear()

    for index, track in enumerate(tracks):
        # Unreadable files are reported and left out instead of aborting the whole split.
        try:
            item = dataset[index]
        except RuntimeError as error:
            skipped.append(track["path"])
            print(f"aviso: {error}", file=sys.stderr)
            continue
        pending.append((track, item))
        if len(pending) == args.batch_size:
            flush()
    flush()

    if not rows:
        raise SystemExit(
            f"Nenhuma faixa de {args.dataset} ({args.split}) pôde ser carregada ({len(skipped)} ignorada(s))."
        )
    predictions = add_track_info(pl.DataFrame(rows), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_csv(args.output)
    accuracies = {
        k: predictions.select(
            pl.any_horizontal(pl.col("genre") == pl.col(f"top{rank}_genre") for rank in range(1, k + 1)).mean()
        ).item()
        for k in sorted({1, min(2, top_k), top_k})
    }
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.array([label_to_index[genre] for genre in predictions["genre"]])
    y_pred = np.array([label_to_index[genre] for genre in predictions["top1_genre"]])
    macro = macro_metrics(y_true, y_pred, len(labels))  # Computed on the top-1 predictions.
    unidentified = predictions.filter(pl.col("artist").is_null() & pl.col("title").is_null()).height
    print(
        f"dataset={args.dataset} split={args.split} tracks={predictions.height} skipped={len(skipped)} "
        f"unidentified_tracks={unidentified} "
        + " ".join(f"top{k}_accuracy={accuracy:.4f}" for k, accuracy in accuracies.items())
    )
    print(" ".join(f"{name}={value:.4f}" for name, value in macro.items()))
    print(f"predições salvas: {args.output}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset is None:
        predict_file(args, device)
    else:
        predict_dataset(args, device)


if __name__ == "__main__":
    main()
