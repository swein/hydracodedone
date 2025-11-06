"""High-level API tests for FastAPI app routes."""

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

from src.main import app


@pytest.mark.asyncio
async def test_health_and_openai_health():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r1 = await ac.get("/health")
        r2 = await ac.get("/v1/health")
    assert r1.status_code == status.HTTP_200_OK and r2.status_code == status.HTTP_200_OK
    # /health returns detailed components, /v1/health returns simple status
    assert r1.json()["status"] == "ok" and r2.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_models_list():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/v1/models")
    data = resp.json()
    ids = {m["id"] for m in data["data"]}
    # Should return the proxy model name "hydracodedone"
    assert ids == {"hydracodedone"}


@pytest.mark.asyncio
async def test_chat_completion_non_stream(monkeypatch):
    from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage

    async def fake_call(messages, model_name, openai_base_url, temperature=None):
        return OpenAIChatCompletionMessage(role="assistant", content="hi")

    monkeypatch.setattr("src.main.call_openai_compatible_chat_model", fake_call)

    payload = {
        "model": "m1",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hi"


@pytest.mark.asyncio
async def test_chat_completion_stream(monkeypatch):
    async def fake_stream(messages, model_name, openai_base_url, temperature=None):
        yield {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {"content": "A"}, "finish_reason": None}],
        }
        yield {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": None}],
        }
        yield {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr("src.main.stream_openai_compatible_chat_model", fake_stream)

    payload = {
        "model": "m1",
        "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        text = await resp.aread()

    # joined SSE stream
    text = text.decode()
    assert "data: [DONE]" in text
    assert any('"content": "A"' in line for line in text.splitlines())
