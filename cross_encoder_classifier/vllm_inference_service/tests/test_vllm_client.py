import asyncio

import httpx
import pytest

from app.vllm_client import VLLMClient, VLLMError


def test_classify_validates_probabilities() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/classify"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "probs": [4.0, 1.0, -2.0, 0.0],
                    }
                ]
            },
        )

    client = VLLMClient("http://vllm", "model", timeout_seconds=1)
    old_client = client._client
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm"
    )

    async def run() -> None:
        await old_client.aclose()
        try:
            with pytest.raises(VLLMError, match="invalid probabilities"):
                await client.classify([[1, 2, 3]])
        finally:
            await client.close()

    asyncio.run(run())
