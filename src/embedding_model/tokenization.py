"""Small deterministic tokenizer used by offline tests and from-scratch training."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from embedding_model.exceptions import ArtifactValidationError, DataValidationError

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")


class LocalTokenizer:
    """A versioned lowercase word tokenizer with explicit special tokens."""

    schema_version = 1

    def __init__(self, vocabulary: dict[str, int]) -> None:
        expected = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        if any(vocabulary.get(token) != index for token, index in expected.items()):
            raise DataValidationError("tokenizer special-token IDs are invalid")
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise DataValidationError("tokenizer IDs must be contiguous from zero")
        self.vocabulary = dict(vocabulary)
        self.inverse_vocabulary = {index: token for token, index in vocabulary.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096) -> LocalTokenizer:
        """Fit a deterministic frequency vocabulary without network access."""

        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise DataValidationError(f"vocab_size must be at least {len(SPECIAL_TOKENS) + 1}")
        counts: Counter[str] = Counter()
        seen = 0
        for text in texts:
            if not isinstance(text, str):
                raise DataValidationError("tokenizer training texts must be strings")
            normalized = text.strip().lower()
            if not normalized:
                raise DataValidationError("tokenizer training texts must not be empty")
            counts.update(TOKEN_PATTERN.findall(normalized))
            seen += 1
        if seen == 0:
            raise DataValidationError("at least one tokenizer training text is required")
        ranked = sorted(counts, key=lambda token: (-counts[token], token))
        tokens = [*SPECIAL_TOKENS, *ranked[: vocab_size - len(SPECIAL_TOKENS)]]
        return cls({token: index for index, token in enumerate(tokens)})

    def tokenize(self, text: str) -> list[str]:
        """Normalize and split one non-empty input string."""

        if not isinstance(text, str):
            raise DataValidationError("text must be a string")
        normalized = text.strip().lower()
        if not normalized:
            raise DataValidationError("text must contain at least one non-whitespace character")
        return TOKEN_PATTERN.findall(normalized)

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        """Encode one string, adding CLS/SEP and truncating before SEP."""

        if max_length < 2:
            raise DataValidationError("max_length must be at least 2")
        token_ids = [
            self.vocabulary.get(token, self.vocabulary["[UNK]"]) for token in self.tokenize(text)
        ]
        ids = [
            self.vocabulary["[CLS]"],
            *token_ids[: max_length - 2],
            self.vocabulary["[SEP]"],
        ]
        return ids, [1] * len(ids)

    def batch_encode(
        self,
        texts: Sequence[str],
        max_length: int,
        *,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pad a non-empty batch and return integer tensors in stable input order."""

        if not texts:
            raise DataValidationError("cannot tokenize an empty batch")
        encoded = [self.encode(text, max_length) for text in texts]
        padded_length = max(len(item[0]) for item in encoded)
        input_ids = []
        attention_mask = []
        for ids, mask in encoded:
            padding = padded_length - len(ids)
            input_ids.append([*ids, *([self.vocabulary["[PAD]"]] * padding)])
            attention_mask.append([*mask, *([0] * padding)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        }

    def save(self, directory: str | Path) -> Path:
        """Write the tokenizer as inspectable JSON."""

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tokenizer.json"
        payload = {
            "schema_version": self.schema_version,
            "algorithm": "word",
            "vocabulary": self.vocabulary,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> LocalTokenizer:
        """Load and validate a tokenizer directory."""

        path = Path(directory) / "tokenizer.json"
        if not path.is_file():
            raise ArtifactValidationError(f"missing tokenizer file: {path}")
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != cls.schema_version:
                raise ArtifactValidationError("unsupported tokenizer schema version")
            vocabulary = payload["vocabulary"]
            if not isinstance(vocabulary, dict):
                raise ArtifactValidationError("tokenizer vocabulary must be an object")
            normalized_vocabulary = {str(token): int(index) for token, index in vocabulary.items()}
            algorithm = payload.get("algorithm", "word")
            if algorithm == "word":
                return cls(normalized_vocabulary)
            if algorithm == "bpe":
                raw_merges = payload.get("merges")
                if not isinstance(raw_merges, list) or not all(
                    isinstance(pair, list)
                    and len(pair) == 2
                    and all(isinstance(symbol, str) for symbol in pair)
                    for pair in raw_merges
                ):
                    raise ArtifactValidationError("BPE tokenizer merges are invalid")
                merges = [(str(pair[0]), str(pair[1])) for pair in raw_merges]
                return BPETokenizer(normalized_vocabulary, merges)
            raise ArtifactValidationError(f"unsupported tokenizer algorithm: {algorithm}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(f"invalid tokenizer file: {path}") from exc


class BPETokenizer(LocalTokenizer):
    """Deterministic byte-pair tokenizer trained from Unicode character symbols."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: list[tuple[str, str]],
    ) -> None:
        super().__init__(vocabulary)
        self.merges = list(merges)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096) -> BPETokenizer:
        """Learn frequent adjacent-symbol merges without a remote dependency."""

        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise DataValidationError(f"vocab_size must be at least {len(SPECIAL_TOKENS) + 1}")
        token_counts: Counter[tuple[str, ...]] = Counter()
        seen = 0
        for text in texts:
            if not isinstance(text, str):
                raise DataValidationError("tokenizer training texts must be strings")
            normalized = text.strip().lower()
            if not normalized:
                raise DataValidationError("tokenizer training texts must not be empty")
            for token in TOKEN_PATTERN.findall(normalized):
                token_counts[_initial_symbols(token)] += 1
            seen += 1
        if seen == 0:
            raise DataValidationError("at least one tokenizer training text is required")
        initial_symbols = sorted({symbol for symbols in token_counts for symbol in symbols})
        minimum_size = len(SPECIAL_TOKENS) + len(initial_symbols)
        if minimum_size > vocab_size:
            raise DataValidationError(
                "vocab_size is too small for the corpus character alphabet; "
                f"requires at least {minimum_size}"
            )
        vocabulary_tokens = [*SPECIAL_TOKENS, *initial_symbols]
        vocabulary_set = set(vocabulary_tokens)
        merges: list[tuple[str, str]] = []
        while len(vocabulary_tokens) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, count in token_counts.items():
                pair_counts.update({pair: count for pair in pairwise(symbols)})
            if not pair_counts:
                break
            selected = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )
            merged_symbol = "".join(selected)
            merges.append(selected)
            updated: Counter[tuple[str, ...]] = Counter()
            for symbols, count in token_counts.items():
                updated[_apply_merge(symbols, selected)] += count
            token_counts = updated
            if merged_symbol not in vocabulary_set:
                vocabulary_tokens.append(merged_symbol)
                vocabulary_set.add(merged_symbol)
        return cls(
            {token: index for index, token in enumerate(vocabulary_tokens)},
            merges,
        )

    def tokenize(self, text: str) -> list[str]:
        """Apply learned merges in training order to normalized input tokens."""

        output: list[str] = []
        for token in super().tokenize(text):
            symbols = _initial_symbols(token)
            for pair in self.merges:
                symbols = _apply_merge(symbols, pair)
            output.extend(symbols)
        return output

    def save(self, directory: str | Path) -> Path:
        """Write BPE vocabulary and ordered merge rules as versioned JSON."""

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tokenizer.json"
        payload = {
            "schema_version": self.schema_version,
            "algorithm": "bpe",
            "vocabulary": self.vocabulary,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def _initial_symbols(token: str) -> tuple[str, ...]:
    if not token:
        raise DataValidationError("BPE tokens must not be empty")
    return (f"▁{token[0]}", *token[1:])


def _apply_merge(
    symbols: tuple[str, ...],
    selected: tuple[str, str],
) -> tuple[str, ...]:
    output: list[str] = []
    index = 0
    while index < len(symbols):
        if (
            index + 1 < len(symbols)
            and symbols[index] == selected[0]
            and symbols[index + 1] == selected[1]
        ):
            output.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return tuple(output)
