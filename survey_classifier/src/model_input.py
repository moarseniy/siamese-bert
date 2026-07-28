from __future__ import annotations

from typing import Any, Iterable


def resolve_input_prefix(
    model: Any,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
) -> str:
    if prompt_name and input_prefix is not None:
        raise ValueError("Use either prompt_name or input_prefix, not both.")
    if input_prefix is not None:
        return str(input_prefix)
    if not prompt_name:
        return ""

    prompts = getattr(model, "prompts", None) or {}
    if prompt_name not in prompts:
        available = ", ".join(sorted(map(str, prompts))) or "<none>"
        raise ValueError(
            f"Prompt {prompt_name!r} is absent in the SentenceTransformer model. "
            f"Available prompts: {available}"
        )
    prefix = prompts[prompt_name]
    if not isinstance(prefix, str):
        raise ValueError(f"Prompt {prompt_name!r} must resolve to a string.")
    return prefix


def configure_model_input(
    model: Any,
    prompt_name: str | None = None,
    input_prefix: str | None = None,
) -> str:
    prefix = resolve_input_prefix(
        model=model,
        prompt_name=prompt_name,
        input_prefix=input_prefix,
    )
    if hasattr(model, "default_prompt_name"):
        model.default_prompt_name = None
    return prefix


def prepare_model_text(text: str, input_prefix: str = "") -> str:
    return f"{input_prefix}{str(text)}"


def prepare_model_texts(texts: Iterable[str], input_prefix: str = "") -> list[str]:
    return [prepare_model_text(text, input_prefix=input_prefix) for text in texts]
