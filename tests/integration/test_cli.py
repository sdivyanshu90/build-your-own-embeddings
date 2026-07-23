from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from embedding_model.cli import app

pytestmark = pytest.mark.integration


def test_cli_help_and_analyze_cross_real_file_boundary(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "validate-artifacts" in help_result.stdout

    data = tmp_path / "pairs.jsonl"
    data.write_text(
        '{"record_id":"1","text_a":"alpha","text_b":"beta"}\n'
        '{"record_id":"2","text_a":"gamma","text_b":"delta"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["analyze-data", "--data", str(data)])
    assert result.exit_code == 0
    assert '"count": 2' in result.stdout
    assert '"unique_texts": 4' in result.stdout
