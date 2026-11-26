"""Interactively list the tracks a model assigns to a genre with at least a given probability.

The input is a CSV written by ``mgc_finetuning.predict --dataset ...``. A track matches when the
requested genre is among its top-k predictions with a probability greater than or equal to the
threshold.
"""

import argparse
import re
import sys
from pathlib import Path

import polars as pl

from .predict import PREDICTIONS_DIR

TOP_GENRE_COLUMN = re.compile(r"^top(\d+)_genre$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recomenda as faixas de um CSV de predições cujo gênero informado tem probabilidade "
            "maior ou igual a um threshold. Sem --genre e --threshold, pergunta os valores interativamente."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="CSV gerado pelo predict em lote. Padrão: escolher entre os CSVs de --predictions-dir.",
    )
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR, help="Pasta com os CSVs de predições.")
    parser.add_argument("--genre", default=None, help="Gênero musical buscado, ex.: blues.")
    parser.add_argument("--threshold", type=float, default=None, help="Probabilidade mínima, entre 0 e 1, ex.: 0.4.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Salva a lista de faixas neste CSV quando --genre e --threshold são informados. "
            "No modo interativo, o script pergunta se deve salvar em predictions/recommendations/."
        ),
    )
    args = parser.parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold deve estar entre 0 e 1.")
    return args


def ask(prompt: str) -> str:
    """Read one answer; an empty answer, Ctrl+C or Ctrl+Z/Ctrl+D ends the session."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def choose_predictions_file(predictions_dir: Path) -> Path | None:
    files = sorted(predictions_dir.glob("*.csv"))
    if not files:
        print(
            f"Nenhum CSV de predições em {predictions_dir}. Gere um com:\n"
            "  uv run python -m mgc_finetuning.predict --dataset gtzan --checkpoint <pasta do checkpoint>"
        )
        return None
    if len(files) == 1:
        print(f"Predições: {files[0]}")
        return files[0]
    print(f"CSVs de predições em {predictions_dir}:")
    for number, path in enumerate(files, start=1):
        print(f"  {number}) {path.name}")
    while True:
        answer = ask("Escolha o número do arquivo (Enter para sair): ")
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(files):
            return files[int(answer) - 1]
        print(f"Informe um número entre 1 e {len(files)}.")


def top_k_ranks(predictions: pl.DataFrame) -> list[int]:
    ranks = sorted(int(match.group(1)) for column in predictions.columns if (match := TOP_GENRE_COLUMN.match(column)))
    if not ranks:
        raise SystemExit("O CSV não tem colunas top{n}_genre; gere-o com mgc_finetuning.predict --dataset.")
    return ranks


def available_genres(predictions: pl.DataFrame, ranks: list[int]) -> list[str]:
    genres = {genre for rank in ranks for genre in predictions.get_column(f"top{rank}_genre").unique().to_list()}
    if "genre" in predictions.columns:
        genres |= set(predictions.get_column("genre").unique().to_list())
    return sorted(genres)


def recommend(predictions: pl.DataFrame, ranks: list[int], genre: str, threshold: float) -> pl.DataFrame:
    """Return the tracks whose top-k predictions give ``genre`` a probability >= ``threshold``."""
    identification = [column for column in ("track_id", "artist", "title") if column in predictions.columns]
    true_genre = [pl.col("genre").alias("true_genre")] if "genre" in predictions.columns else []
    path = ["path"] if "path" in predictions.columns else []
    # A genre appears at most once in each track's top-k, so stacking the ranks yields no duplicates.
    long = pl.concat(
        predictions.select(
            *identification,
            *true_genre,
            pl.lit(rank).alias("rank"),
            pl.col(f"top{rank}_genre").alias("predicted_genre"),
            pl.col(f"top{rank}_probability").alias("probability"),
            *path,
        )
        for rank in ranks
    )
    return (
        long.filter((pl.col("predicted_genre") == genre) & (pl.col("probability") >= threshold))
        .drop("predicted_genre")
        .sort("probability", descending=True)
    )


def show(result: pl.DataFrame, genre: str, threshold: float, top_k: int, num_genres: int) -> None:
    print(f"\n{result.height} faixa(s) com '{genre}' e probabilidade >= {threshold:g}:")
    if result.height:
        # ASCII borders: Unicode box characters fail when the Windows output is redirected (cp1252).
        with pl.Config(
            tbl_rows=-1,
            fmt_str_lengths=120,
            tbl_width_chars=200,
            tbl_formatting="ASCII_MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
        ):
            # Tracks missing from the FMA metadata or the GTZAN index have no artist or title.
            names = [column for column in ("artist", "title") if column in result.columns]
            print(result.with_columns(pl.col(names).fill_null("(desconhecido)")))
    # Any genre outside the top-k has probability <= 1 / (k + 1), since the k larger ones also count.
    if top_k < num_genres and threshold <= 1.0 / (top_k + 1):
        print(
            f"Aviso: o CSV guarda só o top-{top_k}; com threshold <= {1.0 / (top_k + 1):.2f}, faixas em que "
            f"'{genre}' ficou fora do top-{top_k} podem atender ao threshold e não aparecem na lista."
        )


def parse_genre(answer: str, genres: list[str]) -> str | None:
    if answer.isdigit() and 1 <= int(answer) <= len(genres):
        return genres[int(answer) - 1]
    by_name = {genre.lower(): genre for genre in genres}
    return by_name.get(answer.lower())


def parse_threshold(answer: str) -> float | None:
    try:
        threshold = float(answer.replace(",", "."))
    except ValueError:
        return None
    return threshold if 0.0 <= threshold <= 1.0 else None


def default_output(predictions_path: Path, genre: str, threshold: float) -> Path:
    safe_genre = re.sub(r"[^\w-]+", "_", genre)
    return predictions_path.parent / "recommendations" / f"{predictions_path.stem}_{safe_genre}_{threshold:g}.csv"


def save(result: pl.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output)
    print(f"Lista salva: {output}")


def interactive(predictions: pl.DataFrame, predictions_path: Path, ranks: list[int], args: argparse.Namespace) -> None:
    genres = available_genres(predictions, ranks)
    print(f"{predictions.height} faixas, top-{len(ranks)} gêneros por faixa.")
    print("Gêneros: " + ", ".join(f"{number}) {genre}" for number, genre in enumerate(genres, start=1)))
    while True:
        genre = parse_genre(args.genre, genres) if args.genre else None
        if args.genre and genre is None:
            print(f"Gênero desconhecido: {args.genre}")
        while genre is None:
            answer = ask("\nGênero (nome ou número; Enter para sair): ")
            if not answer:
                return
            genre = parse_genre(answer, genres)
            if genre is None:
                print(f"Gênero desconhecido: {answer}")
        threshold = args.threshold
        while threshold is None:
            answer = ask("Threshold de probabilidade entre 0 e 1 (ex.: 0.4; Enter para sair): ")
            if not answer:
                return
            threshold = parse_threshold(answer)
            if threshold is None:
                print(f"Threshold inválido: {answer}")
        args.genre = args.threshold = None  # Values from the command line only apply to the first query.

        result = recommend(predictions, ranks, genre, threshold)
        show(result, genre, threshold, len(ranks), len(genres))
        if result.height:
            output = default_output(predictions_path, genre, threshold)
            if ask(f"Salvar a lista em {output}? [s/N] ").lower() in {"s", "sim", "y", "yes"}:
                save(result, output)


def main() -> None:
    # Track titles may hold characters the Windows code page cannot encode when the output is redirected.
    sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    predictions_path = args.predictions or choose_predictions_file(args.predictions_dir)
    if predictions_path is None:
        return
    predictions = pl.read_csv(predictions_path, schema_overrides={"track_id": pl.String})
    ranks = top_k_ranks(predictions)

    if args.genre is not None and args.threshold is not None:
        genres = available_genres(predictions, ranks)
        genre = parse_genre(args.genre, genres)
        if genre is None:
            raise SystemExit(f"Gênero desconhecido: {args.genre}. Gêneros disponíveis: {', '.join(genres)}")
        result = recommend(predictions, ranks, genre, args.threshold)
        show(result, genre, args.threshold, len(ranks), len(genres))
        if args.output is not None:
            save(result, args.output)
        return
    interactive(predictions, predictions_path, ranks, args)


if __name__ == "__main__":
    main()
