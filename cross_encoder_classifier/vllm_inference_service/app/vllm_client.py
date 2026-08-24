from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import httpx


class VLLMError(RuntimeError):
    pass


class VLLMClient:
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "unknown error"
        while time.monotonic() < deadline:
            try:
                response = await self._client.get("/v1/models")
                if response.is_success:
                    models = response.json().get("data", [])
                    ids = {str(item.get("id")) for item in models}
                    if self.model in ids:
                        return
                    last_error = f"model {self.model!r} is absent; available: {ids}"
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc)
            await asyncio.sleep(2)
        raise VLLMError(f"vLLM did not become ready: {last_error}")

    async def classify(self, token_ids: list[list[int]]) -> list[list[float]]:
        if not token_ids:
            return []
        payload = {
            "model": self.model,
            "input": token_ids,
            "use_activation": True,
        }
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = await self._client.post("/classify", json=payload)
                if response.status_code not in self.RETRYABLE_STATUS_CODES:
                    break
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise VLLMError(f"Cannot reach vLLM: {exc}") from exc
            await asyncio.sleep(2**attempt)

        if response is None:
            raise VLLMError("vLLM returned no response.")
        if not response.is_success:
            raise VLLMError(
                f"vLLM /classify failed with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        try:
            body: dict[str, Any] = response.json()
            items = sorted(body["data"], key=lambda item: int(item["index"]))
            probabilities = [list(map(float, item["probs"])) for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise VLLMError("Invalid response schema from vLLM /classify.") from exc
        if len(probabilities) != len(token_ids):
            raise VLLMError(
                f"vLLM returned {len(probabilities)} results for {len(token_ids)} pairs."
            )
        for row in probabilities:
            if len(row) != 4:
                raise VLLMError(
                    f"Expected four class probabilities, received {len(row)}."
                )
            if (
                any(not math.isfinite(value) or not 0 <= value <= 1 for value in row)
                or not math.isclose(sum(row), 1.0, abs_tol=1e-3)
            ):
                raise VLLMError(
                    "vLLM returned invalid probabilities. Start it with "
                    "--pooler-config '{\"use_activation\": true}'."
                )
        return probabilities
