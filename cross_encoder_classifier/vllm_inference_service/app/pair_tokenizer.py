from __future__ import annotations

from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer


class PairTokenizer:
    """Build the paired token sequence used by bge-reranker-v2-m3 training."""

    def __init__(self, tokenizer_path: Path, max_length: int) -> None:
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(
            max_length=max_length,
            strategy="longest_first",
        )
        probe = self._tokenizer.encode("первый текст", "второй текст")
        sequence_ids = set(value for value in probe.sequence_ids if value is not None)
        if sequence_ids != {0, 1}:
            raise ValueError(
                "tokenizer.json does not define proper pair post-processing."
            )
        if any(probe.type_ids):
            raise ValueError(
                "tokenizer.json is incompatible with BAAI/bge-reranker-v2-m3: "
                "pair tokenization unexpectedly uses token_type_ids."
            )

    def encode_batch(self, pairs: Sequence[tuple[str, str]]) -> list[list[int]]:
        if not pairs:
            return []
        encodings = self._tokenizer.encode_batch(list(pairs), add_special_tokens=True)
        if any(any(encoding.type_ids) for encoding in encodings):
            raise ValueError("Pair tokenization unexpectedly produced token_type_ids.")
        return [encoding.ids for encoding in encodings]
