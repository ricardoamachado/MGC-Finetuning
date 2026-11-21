"""Plot the log-Mel spectrogram (dB) BAT computes for an audio file as a publication-ready PDF figure."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib.ticker import StrMethodFormatter

from .data import BATMelSpectrogram
from .plot import TEXT_SECONDARY  # Importing plot also applies its rcParams.

# Perceptually uniform and monotonic in lightness, so it also reads in grayscale prints.
DEFAULT_COLORMAP = "viridis"


def _hz_to_mel_htk(frequency: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def bat_mel_spectrogram(
    audio_path: Path, start_seconds: float = 0.0
) -> tuple[np.ndarray, BATMelSpectrogram, int]:
    """Return BAT's log-Mel power in dB, before min-max normalization, as [n_mels, frames].

    The steps of ``BATMelSpectrogram`` in evaluation mode are reproduced up to the dB conversion: mono
    16 kHz audio, Mel power spectrogram, the first 1024 frames after ``start_seconds`` and
    ``AmplitudeToDB`` with top_db=80. Only the per-clip min-max normalization to [0, 1] is left out, so
    the values keep their unit, dB relative to a power of 1 (a digital full-scale reference). Frames past
    the end of a short clip, which the model receives as zero padding, are returned as NaN.
    Also returns the transform and the number of frames that hold audio.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform[:, int(round(start_seconds * sample_rate)) :]
    if waveform.shape[-1] == 0:
        raise ValueError(f"{audio_path} tem menos de {start_seconds:g} s de áudio.")
    transform = BATMelSpectrogram(is_training=False)
    waveform = waveform.mean(dim=0)
    if sample_rate != transform.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, transform.sample_rate)
    with torch.no_grad():
        mel = transform.mel(waveform)[:, : transform.target_frames]  # [n_mels, frames]
        mel_db = transform.to_db(mel)
    audio_frames = mel_db.shape[1]
    padded = torch.full((mel_db.shape[0], transform.target_frames), float("nan"))
    padded[:, :audio_frames] = mel_db
    return padded.numpy(), transform, audio_frames


def plot_mel_spectrogram(
    audio_path: Path,
    output_path: Path,
    start_seconds: float = 0.0,
    colormap: str | None = None,
    figsize: tuple[float, float] = (6.0, 2.8),
) -> Path:
    """Save BAT's log-Mel spectrogram of ``audio_path`` as a PDF with time (s) and frequency (Mel) axes.

    Values are BAT's log-Mel power in dB, before its per-clip min-max normalization; the color scale
    spans the 80 dB kept by ``top_db``. When the clip is shorter than the model input, the zero-padded
    frames are left blank and their start is marked.
    """
    spectrogram, transform, audio_frames = bat_mel_spectrogram(audio_path, start_seconds)
    n_mels, frames = spectrogram.shape
    hop_seconds = transform.mel.hop_length / transform.sample_rate
    # The y axis is in Mel: bin i is centered on the (i + 1)-th of n_mels + 2 equally spaced points
    # of the HTK Mel scale between f_min and f_max, so each row spans one Mel step.
    f_min = transform.mel.f_min
    f_max = transform.mel.f_max or transform.sample_rate / 2
    mel_min, mel_max = _hz_to_mel_htk(f_min), _hz_to_mel_htk(f_max)
    mel_step = (mel_max - mel_min) / (n_mels + 1)
    peak = float(np.nanmax(spectrogram))
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        np.ma.masked_invalid(spectrogram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=colormap or DEFAULT_COLORMAP,
        vmin=peak - transform.to_db.top_db,
        vmax=peak,
        extent=(0.0, frames * hop_seconds, mel_min + 0.5 * mel_step, mel_min + (n_mels + 0.5) * mel_step),
        rasterized=True,  # Keeps the PDF small while axes and text stay vector.
    )
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.set_ylabel("Frequency (Mel)")
    axis.set_xlabel("Time (s)" if start_seconds == 0 else f"Time from {start_seconds:g} s (s)")

    if audio_frames < frames:
        padding_start = audio_frames * hop_seconds
        axis.axvline(padding_start, color=TEXT_SECONDARY, linestyle="--", linewidth=0.9)
        axis.annotate(
            "Zero padding",
            xy=(padding_start, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_label("Magnitude (dB)")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera, em PDF, o espectrograma Mel em dB de arquivos de áudio com o pré-processamento do BAT."
    )
    parser.add_argument("audio", type=Path, nargs="+", help="Arquivos de áudio (.mp3, .wav, ...).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/spectrograms"),
        help="Pasta dos PDFs; cada figura é salva como {nome do arquivo}_bat_mel.pdf.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Início do trecho de 10,24 s, em segundos. O padrão 0 corresponde ao trecho usado na inferência.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help=f"Colormap do matplotlib (ex.: magma, inferno). Padrão: {DEFAULT_COLORMAP}.",
    )
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start deve ser maior ou igual a 0.")
    for audio_path in args.audio:
        output_path = args.output_dir / f"{audio_path.stem}_bat_mel.pdf"
        print(f"figura salva: {plot_mel_spectrogram(audio_path, output_path, args.start, args.cmap)}")


if __name__ == "__main__":
    main()
