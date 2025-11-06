"""Test streaming critique pipeline functionality."""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import status
from unittest.mock import AsyncMock, patch
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage

from src.main import app


@pytest.mark.asyncio
async def test_streaming_with_critique_pipeline():
    """Test that streaming requests properly use critique pipeline."""
    
    # Mock Model 1 response
    model_1_response = OpenAIChatCompletionMessage(
        role="assistant", 
        content="Here is some basic code with a bug: print('hello'"
    )
    
    # Mock Model 2 (critique) response chunks
    critique_chunks = [
        {
            "id": "chatcmpl-critique-1",
            "object": "chat.completion.chunk", 
            "created": 1677652288,
            "model": "critique-model",
            "choices": [{"index": 0, "delta": {"content": "Here"}, "finish_reason": None}]
        },
        {
            "id": "chatcmpl-critique-2", 
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "critique-model",
            "choices": [{"index": 0, "delta": {"content": " is the corrected"}, "finish_reason": None}]
        },
        {
            "id": "chatcmpl-critique-3",
            "object": "chat.completion.chunk",
            "created": 1677652288, 
            "model": "critique-model",
            "choices": [{"index": 0, "delta": {"content": " code:"}, "finish_reason": None}]
        },
        {
            "id": "chatcmpl-critique-4",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "critique-model", 
            "choices": [{"index": 0, "delta": {"content": " print('hello')"}, "finish_reason": "stop"}]
        }
    ]
    
    payload = {
        "model": "hydracodedone",
        "stream": True,
        "messages": [{"role": "user", "content": "Write hello world code"}],
    }
    
    # Mock the LLM calls
    with patch('src.main.call_openai_compatible_chat_model', new_callable=AsyncMock) as mock_call, \
         patch('src.main.stream_openai_compatible_chat_model') as mock_stream, \
         patch('src.main.prepare_critique_messages') as mock_prepare:
        
        # Setup mocks
        mock_call.return_value = model_1_response
        mock_prepare.return_value = [
            {"role": "user", "content": "critique instruction"},
            {"role": "assistant", "content": model_1_response.content},
            {"role": "user", "content": "critique prompt"}
        ]
        
        # Make stream an async generator
        async def mock_stream_generator():
            for chunk in critique_chunks:
                yield chunk
        
        mock_stream.return_value = mock_stream_generator()
        
        # Test the streaming endpoint
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=payload)
            
        assert resp.status_code == 200
        
        # Read the streaming response
        text = await resp.aread()
        text = text.decode()
        
        # Verify streaming format
        assert "data: [DONE]" in text
        assert "Here" in text
        assert " is the corrected" in text
        assert " code:" in text
        assert "print('hello')" in text
        
        # Verify Model 1 was called first
        mock_call.assert_called_once()
        
        # Verify critique messages were prepared
        mock_prepare.assert_called_once()
        
        # Verify Model 2 streaming was called
        mock_stream.assert_called_once()


@pytest.mark.asyncio 
async def test_streaming_without_critique_model():
    """Test streaming when critique model is not configured."""
    
    # Mock Model 1 response
    model_1_response = OpenAIChatCompletionMessage(
        role="assistant",
        content="Simple response without critique"
    )
    
    payload = {
        "model": "hydracodedone", 
        "stream": True,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    
    # Mock settings to disable critique
    with patch('src.main.call_openai_compatible_chat_model', new_callable=AsyncMock) as mock_call, \
         patch('src.main.settings.CRITIQUE_MODEL_NAME', None), \
         patch('src.main.settings.CRITIQUE_SYSTEM_PROMPT', None):
        
        mock_call.return_value = model_1_response
        
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=payload)
            
        assert resp.status_code == 200
        
        # Read the streaming response
        text = await resp.aread()
        text = text.decode()
        
        # Should contain Model 1's response (chunked)
        assert "Simp" in text  # First chunk
        assert "le" in text  # Second chunk  
        assert "nse" in text  # Part of "response"
        assert "with" in text
        assert "out" in text
        assert "crit" in text  # Part of "critique"
        assert "ique" in text  # Part of "critique"
        assert "data: [DONE]" in text
        
        # Model 1 should be called
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_critique_error_handling():
    """Test error handling in streaming critique pipeline."""
    
    # Mock Model 1 response
    model_1_response = OpenAIChatCompletionMessage(
        role="assistant",
        content="Original response"
    )
    
    payload = {
        "model": "hydracodedone",
        "stream": True, 
        "messages": [{"role": "user", "content": "Test"}],
    }
    
    # Mock Model 1 success, Model 2 failure
    with patch('src.main.call_openai_compatible_chat_model', new_callable=AsyncMock) as mock_call, \
         patch('src.main.stream_openai_compatible_chat_model') as mock_stream, \
         patch('src.main.prepare_critique_messages'):
        
        mock_call.return_value = model_1_response
        
        # Make Model 2 stream raise an exception
        async def failing_stream():
            raise Exception("Critique model connection failed")
        
        mock_stream.return_value = failing_stream()
        
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post("/v1/chat/completions", json=payload)
            
        assert resp.status_code == 200
        
        # Read the streaming response
        text = await resp.aread()
        text = text.decode()
        
        # Should contain error information in stream
        assert "error" in text.lower()
        assert "data: [DONE]" in text