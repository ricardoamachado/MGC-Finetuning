"""Audio dataset discovery and preprocessing."""

from pathlib import Path

import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor


class ASTMelSpectrogram(torch.nn.Module):
    """Convert waveforms using the same Kaldi fbank preprocessing as AST."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        hop_length: int = 160,
        win_length: int = 400,
        target_frames: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
        is_training: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.mean = mean
        self.std = std
        self.is_training = is_training
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length

        expected_sample_rate = 16_000
        expected_hop_length = 160
        expected_win_length = 400
        if (
            sample_rate != expected_sample_rate
            or hop_length != expected_hop_length
            or win_length != expected_win_length
        ):
            raise ValueError(
                "A configuração do AST exige sample_rate=16000, "
                "hop_length=160 e win_length=400."
            )

    def forward(
        self, waveform: torch.Tensor, source_sample_rate: int
    ) -> torch.Tensor:
        # Garante entrada mono 1D [amostras]
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        # Reamostragem se necessário
        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_sample_rate, self.sample_rate
            )

        # This is the same preprocessing path used by ASTFeatureExtractor when
        # torchaudio is available: Hanning window, 25 ms frames, 10 ms shift,
        # pre-emphasis, DC-offset removal, and a 512-point FFT selected by Kaldi.
        log_mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            sample_frequency=self.sample_rate,
            num_mel_bins=self.n_mels,
            window_type="hanning",
            frame_length=self.win_length * 1000.0 / self.sample_rate,
            frame_shift=self.hop_length * 1000.0 / self.sample_rate,
            dither=0.0,
        )

        total_frames = log_mel.shape[0]

        # Random crop during training; AST-compatible initial crop otherwise.
        if total_frames > self.target_frames:
            if self.is_training:
                start = torch.randint(
                    0, total_frames - self.target_frames + 1, (1,)
                ).item()
            else:
                start = 0
            log_mel = log_mel[start : start + self.target_frames, :]
        elif total_frames < self.target_frames:
            log_mel = torch.nn.functional.pad(
                log_mel, (0, 0, 0, self.target_frames - total_frames)
            )

        # ASTFeatureExtractor normalizes after padding/truncation as
        # (log_mel - mean) / (std * 2).
        log_mel = (log_mel - self.mean) / (self.std * 2.0)

        # Return [target_frames, n_mels] -> [1024, 128].
        return log_mel.contiguous()


class BATMelSpectrogram(torch.nn.Module):
    """Convert waveforms using the log-mel preprocessing BAT was pretrained with.

    The pretrained weights expect a 1024-point Hann STFT, power-to-dB compression (top_db=80) and
    per-sample min-max normalization to [0, 1], unlike the 25 ms Kaldi fbank used by AST.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 160,
        n_mels: int = 128,
        top_db: float = 80.0,
        target_frames: int = 1024,
        is_training: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.is_training = is_training
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=top_db)

    def forward(
        self, waveform: torch.Tensor, source_sample_rate: int
    ) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_sample_rate, self.sample_rate
            )

        mel = self.mel(waveform)  # [n_mels, frames]

        # BAT was pretrained on 10 s clips normalized over the whole input, so longer tracks are
        # cropped before the dB reference and the min-max range are computed.
        total_frames = mel.shape[1]
        if total_frames > self.target_frames:
            start = (
                torch.randint(0, total_frames - self.target_frames + 1, (1,)).item()
                if self.is_training
                else 0
            )
            mel = mel[:, start : start + self.target_frames]

        log_mel = self.to_db(mel)
        minimum, maximum = log_mel.aminmax()
        log_mel = (log_mel - minimum) / (maximum - minimum + 1e-8)
        if total_frames < self.target_frames:
            log_mel = torch.nn.functional.pad(log_mel, (0, self.target_frames - total_frames))

        # Return [target_frames, n_mels] -> [1024, 128].
        return log_mel.T.contiguous()


def build_mel_transform(
    model: str, pretrained_path: str | Path, is_training: bool
) -> torch.nn.Module:
    """Create the waveform-to-spectrogram transform expected by a pretrained model."""
    if model == "bat":
        return BATMelSpectrogram(is_training=is_training)
    extractor = ASTFeatureExtractor.from_pretrained(pretrained_path)
    return ASTMelSpectrogram(mean=extractor.mean, std=extractor.std, is_training=is_training)


FMA_NESTED_SUBSETS = {
    "small": ("small",),
    "medium": ("small", "medium"),
    "large": ("small", "medium", "large"),
}


class FMADataset(Dataset):
    """Load FMA files and expose Mel spectrograms instead of raw waveforms."""

    def __init__(
        self,
        dataset_dir: Path,
        metadata_path: Path,
        subset: str = "small",
        split: str = "training",
        class_names: list[str] | None = None,
        mel_transform: torch.nn.Module | None = None,
    ):
        if subset not in FMA_NESTED_SUBSETS:
            raise ValueError(f"Subset must be one of {list(FMA_NESTED_SUBSETS)}.")
        if split not in {"training", "validation", "test"}:
            raise ValueError("Split must be one of 'training', 'validation', or 'test'.")
        self.data_dir = Path(dataset_dir)
        self.metadata_path = Path(metadata_path)
        self.subset = subset
        df = pl.read_csv(metadata_path, skip_rows=1, null_values=[""])
        df = df.rename({df.columns[0]: "track_id"})
        df = df.filter(pl.col("track_id") != "track_id").with_columns(
            pl.col("track_id").cast(pl.Int64)
        )
        # The metadata stores the smallest subset containing each track, while the subsets
        # are nested (small ⊂ medium ⊂ large), e.g. fma_medium also holds every small track.
        df_subset = df.filter(pl.col("subset").is_in(FMA_NESTED_SUBSETS[subset])).drop_nulls(
            subset=["genre_top"]
        )
        # The official training/validation/test splits are defined in the metadata file.
        df_subset = df_subset.filter(pl.col("split") == split)
        # Get the track_ids and genres as lists.
        self.track_ids = df_subset["track_id"].cast(pl.Int64).to_list()
        self.genres = df_subset["genre_top"].to_list()
        self.classes_ = sorted(class_names or set(self.genres))
        self.class_to_index = {name: index for index, name in enumerate(self.classes_)}
        unknown_classes = sorted(set(self.genres) - set(self.class_to_index))
        if unknown_classes:
            raise ValueError(f"Classes ausentes no conjunto de treino: {unknown_classes}")
        self.labels = [self.class_to_index[genre] for genre in self.genres]
        self.mel_transform = mel_transform or ASTMelSpectrogram()
        self.length = len(self.track_ids)
        audio_paths = [self.audio_path(track_id) for track_id in self.track_ids]
        missing_paths = [path for path in audio_paths if not path.is_file()]
        if missing_paths:
            examples = ", ".join(str(path) for path in missing_paths[:3])
            raise FileNotFoundError(
                f"{len(missing_paths)} arquivos de áudio do FMA não foram encontrados. "
                f"Exemplos: {examples}"
            )

    def audio_path(self, track_id: int) -> Path:
        # The FMA dataset organizes audio files in folders based on the first three digits of the track ID.
        track_id_str = f"{track_id:06d}"
        folder = track_id_str[:3]
        return self.data_dir / f"fma_{self.subset}" / folder / f"{track_id_str}.mp3"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track_id = self.track_ids[idx]
        audio_path = self.audio_path(track_id)

        # Do not replace failed reads with silence: that would make every failed
        # sample identical and can produce chance-level accuracy without errors.
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as error:
            raise RuntimeError(
                f"Não foi possível carregar o áudio FMA: {audio_path}"
            ) from error
        mel_spectrogram = self.mel_transform(waveform, sample_rate)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_values": mel_spectrogram, "labels": label}


# Column positions in the FMA tracks.csv, whose three header rows are (group, field, "track_id").
FMA_ARTIST_NAME_COLUMN = 26  # artist.name
FMA_TRACK_TITLE_COLUMN = 52  # track.title
GTZAN_INDEX_URL = "https://github.com/boblsturm/GTZAN/blob/main/index.txt"


def load_fma_track_info(metadata_path: Path) -> pl.DataFrame:
    """Return ``track_id``, ``artist`` and ``title`` for every track of the FMA metadata."""
    info = pl.read_csv(
        metadata_path,
        has_header=False,
        skip_rows=3,
        columns=[0, FMA_ARTIST_NAME_COLUMN, FMA_TRACK_TITLE_COLUMN],
        new_columns=["track_id", "artist", "title"],
        infer_schema=False,
    )
    # Integer round trip so the IDs match the unpadded ones written by predict (e.g. "182").
    return info.with_columns(
        pl.col("track_id").cast(pl.Int64).cast(pl.String),
        pl.col("artist", "title").str.strip_chars().replace("", None),
    )


def load_gtzan_track_info(index_path: Path) -> pl.DataFrame:
    """Return ``track_id`` (file stem), ``artist`` and ``title`` from Bob L. Sturm's GTZAN index.

    Each entry reads ``blues.00000.wav ::: John Lee Hooker ::: One Bourbon, One Scotch And One Beer``.
    Unidentified tracks have empty fields, which become nulls.
    """
    rows = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or ":::" not in line:
            continue
        file_name, artist, title = (part.strip() for part in (line.split(":::") + ["", ""])[:3])
        rows.append({"track_id": Path(file_name).stem, "artist": artist or None, "title": title or None})
    return pl.DataFrame(rows, schema={"track_id": pl.String, "artist": pl.String, "title": pl.String})


GTZAN_SPLIT_FILES = {
    "train": "train_stratified.txt",
    "validation": "valid_stratified.txt",
    "test": "test_stratified.txt",
}


def build_pitch_shift_cache(
    audio_paths: list[Path],
    source_dir: Path,
    cache_dir: Path,
    semitones: tuple[int, ...],
    sample_rate: int = 16_000,
    device: torch.device | None = None,
) -> None:
    """Write pitch-shifted copies of each file as 16 kHz WAVs, skipping the ones already cached.

    The phase vocoder is too slow to run on the CPU every epoch, so each shift is computed
    once (on ``device``, ideally a GPU) and reused across epochs and runs.
    """
    device = device or torch.device("cpu")
    pending = [
        (path, n_steps)
        for n_steps in semitones
        for path in audio_paths
        if not pitch_shift_cache_path(path, source_dir, cache_dir, n_steps).is_file()
    ]
    if not pending:
        return
    print(f"pitch_shift_cache: generating {len(pending)} files in {cache_dir} on {device}")
    for path, n_steps in pending:
        try:
            waveform, source_rate = torchaudio.load(path)
        except Exception as error:
            raise RuntimeError(f"Não foi possível carregar o áudio GTZAN: {path}") from error
        waveform = waveform.mean(dim=0)
        if source_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
        shifted = torchaudio.functional.pitch_shift(
            waveform.to(device), sample_rate, n_steps, n_fft=2048
        )
        target = pitch_shift_cache_path(path, source_dir, cache_dir, n_steps)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file first so an interrupted run never leaves a truncated WAV.
        temporary = target.with_suffix(".tmp.wav")
        torchaudio.save(temporary, shifted.clamp(-1.0, 1.0).unsqueeze(0).cpu(), sample_rate)
        temporary.replace(target)


def pitch_shift_cache_path(path: Path, source_dir: Path, cache_dir: Path, n_steps: int) -> Path:
    return cache_dir / f"semitones_{n_steps:+d}" / path.relative_to(source_dir)


class GTZANDataset(Dataset):
    """Load GTZAN files from the stratified split lists and expose Mel spectrograms.

    GTZAN has no metadata file: the genre is the name of the folder that contains each track.
    When ``pitch_shift_semitones`` is given, every track of the split is also included once per
    shift, so ``(-1, 1)`` triples the number of samples.
    """

    def __init__(
        self,
        dataset_dir: Path,
        split: str = "train",
        class_names: list[str] | None = None,
        mel_transform: torch.nn.Module | None = None,
        pitch_shift_semitones: tuple[int, ...] = (),
        cache_dir: Path | None = None,
        device: torch.device | None = None,
    ):
        if split not in GTZAN_SPLIT_FILES:
            raise ValueError(f"Split must be one of {sorted(GTZAN_SPLIT_FILES)}.")
        self.root = Path(dataset_dir) / "GTZAN"
        split_file = self.root / GTZAN_SPLIT_FILES[split]
        relative_paths = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [self.root / relative_path for relative_path in relative_paths]
        missing_paths = [path for path in paths if not path.is_file()]
        if missing_paths:
            examples = ", ".join(str(path) for path in missing_paths[:3])
            raise FileNotFoundError(
                f"{len(missing_paths)} arquivos de áudio do GTZAN não foram encontrados. Exemplos: {examples}"
            )

        genres = [path.parent.name for path in paths]
        self.classes_ = sorted(class_names or set(genres))
        self.class_to_index = {name: index for index, name in enumerate(self.classes_)}
        unknown_classes = sorted(set(genres) - set(self.class_to_index))
        if unknown_classes:
            raise ValueError(f"Classes ausentes no conjunto de treino: {unknown_classes}")

        self.audio_paths = list(paths)
        self.labels = [self.class_to_index[genre] for genre in genres]
        if pitch_shift_semitones:
            cache_dir = cache_dir or Path(dataset_dir) / "GTZAN_pitch_shifted"
            build_pitch_shift_cache(paths, self.root, cache_dir, pitch_shift_semitones, device=device)
            for n_steps in pitch_shift_semitones:
                self.audio_paths += [pitch_shift_cache_path(path, self.root, cache_dir, n_steps) for path in paths]
                self.labels += [self.class_to_index[genre] for genre in genres]
        self.mel_transform = mel_transform or ASTMelSpectrogram()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as error:
            raise RuntimeError(f"Não foi possível carregar o áudio GTZAN: {audio_path}") from error
        mel_spectrogram = self.mel_transform(waveform, sample_rate)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_values": mel_spectrogram, "labels": label}
