from __future__ import annotations

from collections.abc import Sequence

from .codebook import CodeEntry, clean_text
from .pair_tokenizer import PairTokenizer
from .vllm_client import VLLMClient


UNKNOWN_CODE = "UNKNOWN"


def add_after_semicolon_prefix(value: object, prefix: str) -> str:
    text = clean_text(value)
    if not prefix or ";" not in text:
        return text
    before, after = text.split(";", 1)
    suffix = f" {after.strip()}" if after.strip() else ""
    return f"{before.strip()}; {prefix}{suffix}"


def decode_prediction(
    probabilities: Sequence[Sequence[float]],
    codebook: Sequence[CodeEntry],
    threshold: float,
    max_labels: int,
) -> str:
    candidates: list[tuple[float, str, int]] = []
    for values, entry in zip(probabilities, codebook, strict=True):
        if len(values) != 4:
            raise ValueError("Each pair must have four class probabilities.")
        presence = 1.0 - float(values[0])
        if presence >= threshold:
            sentiment = max(range(3), key=lambda index: float(values[index + 1]))
            candidates.append((presence, entry.code, sentiment))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_labels]
    return (
        ", ".join(f"{code}:{sentiment}" for _, code, sentiment in selected)
        or UNKNOWN_CODE
    )


class SurveyPredictor:
    def __init__(
        self,
        codebook: Sequence[CodeEntry],
        tokenizer: PairTokenizer,
        client: VLLMClient,
        threshold: float,
        max_labels: int,
        pair_batch_size: int,
        after_semicolon_prefix: str = "",
    ) -> None:
        self.codebook = list(codebook)
        self.tokenizer = tokenizer
        self.client = client
        self.threshold = threshold
        self.max_labels = max_labels
        self.pair_batch_size = pair_batch_size
        self.after_semicolon_prefix = after_semicolon_prefix

    async def predict(self, answers: Sequence[object]) -> list[str]:
        results = [UNKNOWN_CODE] * len(answers)
        rows_per_group = max(1, self.pair_batch_size // len(self.codebook))
        for group_start in range(0, len(answers), rows_per_group):
            group = answers[group_start : group_start + rows_per_group]
            pairs: list[tuple[str, str]] = []
            active_indices: list[int] = []
            for local_index, answer in enumerate(group):
                text = add_after_semicolon_prefix(
                    answer, self.after_semicolon_prefix
                )
                if not text:
                    continue
                active_indices.append(local_index)
                pairs.extend((text, entry.description) for entry in self.codebook)

            all_probabilities: list[list[float]] = []
            for pair_start in range(0, len(pairs), self.pair_batch_size):
                pair_batch = pairs[pair_start : pair_start + self.pair_batch_size]
                token_ids = self.tokenizer.encode_batch(pair_batch)
                all_probabilities.extend(await self.client.classify(token_ids))

            code_count = len(self.codebook)
            for active_position, local_index in enumerate(active_indices):
                start = active_position * code_count
                stop = start + code_count
                results[group_start + local_index] = decode_prediction(
                    all_probabilities[start:stop],
                    self.codebook,
                    self.threshold,
                    self.max_labels,
                )
        return results
