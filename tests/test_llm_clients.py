"""Tests for llm_clients module (call_openai_compatible_chat_model & stream_openai_compatible_chat_model)."""

import httpx
import pytest

respx = pytest.importorskip("respx")
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage
from src.services.llm_clients import (
    call_openai_compatible_chat_model,
    stream_openai_compatible_chat_model,
)
from src.models.openai_types import CustomChatCompletionMessage

OPENAI_COMPATIBLE_URL = "http://127.0.0.1:1234/v1"


@pytest.mark.asyncio
@respx.mock
async def test_call_openai_compatible_chat_model_success():
    """call_openai_compatible_chat_model should return assistant message when OpenAI-compatible API responds 200."""
    messages = [CustomChatCompletionMessage(role="user", content="Ping?")]
    expected_reply = "Pong!"
    respx.post(f"{OPENAI_COMPATIBLE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": expected_reply,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )
    )

    result = await call_openai_compatible_chat_model(
        messages, "gpt-3.5-turbo", OPENAI_COMPATIBLE_URL
    )

    assert isinstance(result, OpenAIChatCompletionMessage)
    assert result.role == "assistant"
    assert result.content == expected_reply


@pytest.mark.asyncio
@respx.mock
async def test_call_openai_compatible_chat_model_http_error():
    """Should raise when OpenAI-compatible API returns non-200."""
    messages = [CustomChatCompletionMessage(role="user", content="hi")]
    respx.post(f"{OPENAI_COMPATIBLE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )

    with pytest.raises(Exception):
        await call_openai_compatible_chat_model(
            messages, "dummy", OPENAI_COMPATIBLE_URL
        )


@pytest.mark.asyncio
async def test_stream_openai_compatible_chat_model_iterates(monkeypatch):
    """stream_openai_compatible_chat_model should yield decoded JSON chunks."""
    messages = [CustomChatCompletionMessage(role="user", content="stream?")]

    # Build a fake AsyncClient.stream context manager
    class DummyStreamResponse:
        def __init__(self):
            self._lines = [
                'data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}',
                'data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]}',
                'data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
                "data: [DONE]",
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        def raise_for_status(self):
            pass

    # Patch httpx.AsyncClient to return our dummy response
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, method, url, **kwargs):
            return DummyStreamResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    chunks = []
    async for chunk in stream_openai_compatible_chat_model(
        messages, "gpt-3.5-turbo", OPENAI_COMPATIBLE_URL
    ):
        chunks.append(chunk)

    # Verify we got the expected chunks (excluding the [DONE] marker)
    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_openai_compatible_chat_model_ignores_empty_lines(monkeypatch):
    """stream_openai_compatible_chat_model should ignore empty lines from SSE."""
    messages = [CustomChatCompletionMessage(role="user", content="stream?")]

    class DummyStreamResponse:
        def __init__(self):
            self._lines = [
                "",  # Empty line should be ignored
                'data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}',
                "",  # Another empty line
                "data: [DONE]",
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        def raise_for_status(self):
            pass

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, method, url, **kwargs):
            return DummyStreamResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    chunks = []
    async for chunk in stream_openai_compatible_chat_model(
        messages, "gpt-3.5-turbo", OPENAI_COMPATIBLE_URL
    ):
        chunks.append(chunk)

    # Should only get one content chunk, ignoring empty lines
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_stream_openai_compatible_chat_model_handles_invalid_json(monkeypatch):
    """stream_openai_compatible_chat_model should skip invalid JSON lines."""
    messages = [CustomChatCompletionMessage(role="user", content="stream?")]

    class DummyStreamResponse:
        def __init__(self):
            self._lines = [
                'data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": "Valid"}, "finish_reason": null}]}',
                "data: {invalid json}",  # Invalid JSON should be skipped
                'data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": " again"}, "finish_reason": null}]}',
                "data: [DONE]",
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        def raise_for_status(self):
            pass

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, method, url, **kwargs):
            return DummyStreamResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    chunks = []
    async for chunk in stream_openai_compatible_chat_model(
        messages, "gpt-3.5-turbo", OPENAI_COMPATIBLE_URL
    ):
        chunks.append(chunk)

    # Should get two valid chunks, skipping the invalid JSON
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Valid"
    assert chunks[1]["choices"][0]["delta"]["content"] == " again"
