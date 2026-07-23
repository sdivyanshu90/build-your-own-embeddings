"""Unified command-line interface for the complete local lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import uvicorn

from embedding_model.config import load_config
from embedding_model.data.preprocessing import pair_statistics
from embedding_model.data.readers import read_records
from embedding_model.data.samplers import lexical_hard_negatives
from embedding_model.data.schemas import PairRecord, ScoredPairRecord
from embedding_model.evaluation.diagnostics import embedding_diagnostics
from embedding_model.evaluation.similarity import similarity_metrics
from embedding_model.export.exporter import export_model, load_exported_model
from embedding_model.export.manifest import validate_manifest
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.logging import configure_logging
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.serving.app import create_app
from embedding_model.serving.settings import ServingSettings
from embedding_model.tokenization import BPETokenizer, LocalTokenizer
from embedding_model.training.trainer import Trainer

app = typer.Typer(
    name="embedding-project",
    help="Train, evaluate, export, index, search, and serve text embeddings.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Configure structured package logging for every command."""

    configure_logging()


@app.command("analyze-data")
def analyze_data(
    data: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Validate pair data and print safe aggregate statistics."""

    records = read_records(data, PairRecord)
    typer.echo(json.dumps(pair_statistics(records).__dict__, sort_keys=True))


@app.command("train-tokenizer")
def train_tokenizer(
    data: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    vocab_size: Annotated[int, typer.Option(min=8)] = 4096,
) -> None:
    """Train and save the deterministic local tokenizer."""

    records = read_records(data, PairRecord)
    texts = [text for record in records for text in (record.text_a, record.text_b)]
    tokenizer = BPETokenizer.train(texts, vocab_size=vocab_size)
    path = tokenizer.save(output_dir)
    typer.echo(str(path))


@app.command("mine-negatives")
def mine_negatives(
    queries: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    documents: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)],
    per_query: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Mine TF-IDF hard negatives while excluding all supplied positive IDs."""

    query_rows = _read_jsonl_objects(queries)
    document_rows = _read_jsonl_objects(documents)
    query_texts = {
        _required_string(row, "query_id"): _required_string(row, "query") for row in query_rows
    }
    document_texts = {
        _required_string(row, "id"): _required_string(row, "text") for row in document_rows
    }
    positives: dict[str, set[str]] = {}
    for row in query_rows:
        query_id = _required_string(row, "query_id")
        values = row.get("positive_ids")
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise typer.BadParameter("each query requires a non-empty positive_ids list")
        positives[query_id] = set(values)
    mined = lexical_hard_negatives(
        query_texts,
        document_texts,
        positives,
        per_query=per_query,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": item.query_id,
                    "document_id": item.document_id,
                    "score": item.score,
                },
                sort_keys=True,
            )
            + "\n"
            for item in mined
        ),
        encoding="utf-8",
    )
    typer.echo(str(output))


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    data: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    resume_from: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Train pair data on CPU/CUDA and export a checksum-verified artifact."""

    project = load_config(config)
    records = read_records(data, PairRecord)
    texts = [text for record in records for text in (record.text_a, record.text_b)]
    tokenizer = BPETokenizer.train(texts, vocab_size=project.model.vocabulary_size)
    model_config = project.model.model_copy(update={"vocabulary_size": tokenizer.vocab_size})
    model = EmbeddingModel(model_config)
    trainer = Trainer(model, tokenizer, project.training)
    if resume_from is not None:
        trainer.resume(resume_from)
    result = trainer.train_pairs(records)
    export_model(
        model,
        tokenizer,
        output_dir,
        training_metadata={
            "global_step": result.global_step,
            "elapsed_seconds": result.elapsed_seconds,
            "checkpoint": str(result.last_checkpoint),
        },
        evaluation={"training_loss": result.training_loss},
    )
    typer.echo(
        json.dumps(
            {
                "artifact": str(output_dir),
                "global_step": result.global_step,
                "training_loss": result.training_loss,
            },
            sort_keys=True,
        )
    )


@app.command()
def evaluate(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    data: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Evaluate scored text pairs and report embedding diagnostics."""

    embedder = TextEmbedder.from_pretrained(model_path)
    records = read_records(data, ScoredPairRecord)
    left = np.asarray(embedder.encode([record.text_a for record in records]))
    right = np.asarray(embedder.encode([record.text_b for record in records]))
    predictions = (left * right).sum(axis=1)
    report: dict[str, Any] = {
        **similarity_metrics(predictions, [record.score for record in records]),
        "diagnostics": embedding_diagnostics(np.concatenate((left, right), axis=0)),
    }
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered, nl=False)


@app.command()
def export(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    tokenizer_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    """Export an existing trusted local checkpoint into safe inference files."""

    project = load_config(config)
    tokenizer = LocalTokenizer.load(tokenizer_path)
    model_config = project.model.model_copy(update={"vocabulary_size": tokenizer.vocab_size})
    model = EmbeddingModel(model_config)
    trainer = Trainer(model, tokenizer, project.training)
    trainer.resume(checkpoint)
    export_model(
        model,
        tokenizer,
        output_dir,
        training_metadata={"global_step": trainer.global_step},
    )
    typer.echo(str(output_dir))


@app.command()
def index(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    documents: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
) -> None:
    """Encode JSONL ``id``/``text`` documents and build a cosine index."""

    rows = _read_jsonl_objects(documents)
    metadata: list[dict[str, Any]] = []
    texts: list[str] = []
    for row in rows:
        metadata.append(dict(row))
        _required_string(row, "id")
        texts.append(_required_string(row, "text"))
    embedder = TextEmbedder.from_pretrained(model_path)
    vectors = np.asarray(embedder.encode(texts, batch_size=batch_size))
    vector_index = VectorIndex(embedder.dimension)
    vector_index.add(vectors, metadata)
    vector_index.save(output_dir)
    typer.echo(json.dumps({"index": str(output_dir), "size": vector_index.size}))


@app.command()
def search(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    index_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    query: Annotated[str, typer.Option()],
    top_k: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Search a compatible model/index pair."""

    embedder = TextEmbedder.from_pretrained(model_path)
    vector_index = VectorIndex.load(index_path)
    if embedder.dimension != vector_index.dimension:
        raise typer.BadParameter("model and index dimensions do not match")
    hits = vector_index.search(np.asarray(embedder.encode(query)), top_k=top_k)
    typer.echo(
        json.dumps(
            [{"id": hit.document_id, "score": hit.score, "metadata": hit.metadata} for hit in hits],
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def serve(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    index_path: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Start one bounded inference process; scale with external process workers."""

    embedder = TextEmbedder.from_pretrained(model_path)
    vector_index = VectorIndex.load(index_path) if index_path is not None else None
    if vector_index is not None and vector_index.dimension != embedder.dimension:
        raise typer.BadParameter("model and index dimensions do not match")
    service = create_app(
        embedder=embedder,
        index=vector_index,
        settings=ServingSettings(),
    )
    uvicorn.run(service, host=host, port=port, access_log=False)


@app.command()
def benchmark(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    data: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    repeats: Annotated[int, typer.Option(min=1, max=100)] = 3,
) -> None:
    """Measure local encoding throughput without claiming portable thresholds."""

    records = read_records(data, PairRecord)
    texts = [record.text_a for record in records]
    embedder = TextEmbedder.from_pretrained(model_path)
    started = time.perf_counter()
    for _ in range(repeats):
        embedder.encode(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    typer.echo(
        json.dumps(
            {
                "batch_size": batch_size,
                "elapsed_seconds": elapsed,
                "texts": len(texts) * repeats,
                "texts_per_second": len(texts) * repeats / elapsed,
            },
            sort_keys=True,
        )
    )


@app.command("validate-artifacts")
def validate_artifacts(
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Verify checksums and load tensor shapes without performing inference."""

    manifest = validate_manifest(model_path)
    model, _ = load_exported_model(model_path)
    typer.echo(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "dimension": model.embedding_dimension,
                "valid": True,
            },
            sort_keys=True,
        )
    )


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise typer.BadParameter(f"blank JSONL line {line_number} in {path}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"invalid JSON on line {line_number} in {path}") from exc
        if not isinstance(row, dict):
            raise typer.BadParameter(f"line {line_number} in {path} must be an object")
        rows.append(row)
    if not rows:
        raise typer.BadParameter(f"JSONL file is empty: {path}")
    return rows


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise typer.BadParameter(f"every row requires a non-empty string {key!r}")
    return value


if __name__ == "__main__":
    app()
