"""Predict the class of one audio file using a fine-tuned checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from .data import build_mel_transform
from .model import FINETUNER_CLASSES, checkpoint_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Pasta do checkpoint, ex.: checkpoints/ast-gtzan-head ou checkpoints/bat-fma_small-lora.",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads((args.checkpoint / "labels.json").read_text(encoding="utf-8"))
    model_key = checkpoint_model(args.checkpoint)
    mel_transform = build_mel_transform(model_key, args.checkpoint, is_training=False)
    model = FINETUNER_CLASSES[model_key].from_pretrained(args.checkpoint, num_labels=len(labels)).to(device).eval()
    waveform, sample_rate = torchaudio.load(args.audio)
    mel_spectrogram = mel_transform(waveform, sample_rate).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(mel_spectrogram.to(device))["probabilities"][0]
    index = int(probabilities.argmax())
    print(json.dumps({"label": labels[index], "probability": float(probabilities[index])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
