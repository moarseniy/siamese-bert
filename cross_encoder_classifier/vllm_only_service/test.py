from __future__ import annotations

import argparse
import json
import os
from pprint import pprint
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    url: str,
    *,
    payload: dict[str, object] | None = None,
    api_key: str | None = None,
) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Cannot reach {url}: {exc.reason}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the vLLM /classify API.")
    parser.add_argument(
        "--url",
        default=os.getenv("VLLM_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        default=(
            "Зарплата низкая, но коллектив отличный. "
            "Категория: Условия труда. Подкатегория: Уровень заработной платы"
        ),
        help="A raw single-sequence smoke-test input.",
    )
    input_group.add_argument(
        "--token-ids",
        help="A JSON array with an already tokenized answer/code pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.url.rstrip("/")
    models = request_json(f"{base_url}/v1/models", api_key=args.api_key)
    available = models.get("data", [])
    model = args.model
    if not model and isinstance(available, list) and available:
        model = available[0].get("id")
    if not model:
        raise SystemExit("No served model was returned by /v1/models.")

    model_input: str | list[int]
    if args.token_ids:
        model_input = json.loads(args.token_ids)
        if not isinstance(model_input, list) or not all(
            isinstance(value, int) for value in model_input
        ):
            raise SystemExit("--token-ids must be a JSON array of integers.")
    else:
        model_input = args.input

    result = request_json(
        f"{base_url}/classify",
        payload={"model": model, "input": model_input},
        api_key=args.api_key,
    )
    pprint(result)


if __name__ == "__main__":
    main()
