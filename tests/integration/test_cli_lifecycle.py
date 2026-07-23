from __future__ import annotations

import json
from pathlib import Path

import pytest

import embedding_model.cli as cli

pytestmark = pytest.mark.integration


def test_cli_commands_run_a_real_local_lifecycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair_data = tmp_path / "pairs.jsonl"
    pair_data.write_text(
        '{"record_id":"1","text_a":"cat on mat","text_b":"kitten on rug"}\n'
        '{"record_id":"2","text_a":"bake bread","text_b":"bread recipe"}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    config.write_text(
        "model:\n"
        "  vocabulary_size: 32\n"
        "  hidden_size: 8\n"
        "  embedding_dimension: 4\n"
        "  num_attention_heads: 2\n"
        "  num_hidden_layers: 1\n"
        "  intermediate_size: 16\n"
        "  max_sequence_length: 12\n"
        "  dropout: 0.0\n"
        "training:\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  learning_rate: 0.01\n"
        "  warmup_ratio: 0.0\n"
        "  device: cpu\n"
        f"  checkpoint_dir: {checkpoint_dir}\n",
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "standalone-tokenizer"
    cli.train_tokenizer(pair_data, tokenizer_dir, vocab_size=32)
    assert (tokenizer_dir / "tokenizer.json").is_file()

    model_dir = tmp_path / "model"
    cli.train(config, pair_data, model_dir, resume_from=None)
    assert (model_dir / "manifest.json").is_file()

    exported_again = tmp_path / "exported-again"
    cli.export(
        config,
        model_dir / "tokenizer",
        checkpoint_dir / "last.pt",
        exported_again,
    )
    cli.validate_artifacts(exported_again)

    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        '{"id":"d1","text":"kitten on rug"}\n{"id":"d2","text":"bread recipe"}\n',
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    cli.index(model_dir, documents, index_dir, batch_size=2)
    cli.search(model_dir, index_dir, "bread", top_k=1)

    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        '{"record_id":"s1","text_a":"cat on mat","text_b":"kitten on rug","score":0.9}\n'
        '{"record_id":"s2","text_a":"cat on mat","text_b":"bread recipe","score":0.0}\n'
        '{"record_id":"s3","text_a":"bake bread","text_b":"bread recipe","score":0.8}\n',
        encoding="utf-8",
    )
    report = tmp_path / "evaluation.json"
    cli.evaluate(model_dir, scored, report)
    assert set(json.loads(report.read_text(encoding="utf-8"))) >= {
        "pearson",
        "spearman",
        "mse",
        "diagnostics",
    }
    cli.benchmark(model_dir, pair_data, batch_size=2, repeats=1)

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query_id":"q1","query":"cat","positive_ids":["d1"]}\n',
        encoding="utf-8",
    )
    mined = tmp_path / "mined.jsonl"
    cli.mine_negatives(queries, documents, mined, per_query=1)
    assert json.loads(mined.read_text(encoding="utf-8"))["document_id"] == "d2"
    assert "valid" in capsys.readouterr().out
