import httpx
import logging
import json
from typing import List, Optional, Dict, Any, AsyncIterator

from ..models.openai_types import CustomChatCompletionMessage  # For input
from openai.types.chat import (
    ChatCompletionMessage as OpenAIChatCompletionMessage,
)  # For output
from ..core.config import settings

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Custom exception for LLM connection issues."""

    pass


class LLMResponseError(Exception):
    """Custom exception for unexpected LLM response format."""

    pass


async def call_openai_compatible_chat_model(
    messages: List[CustomChatCompletionMessage],
    model_name: str,
    openai_base_url: str,
    temperature: Optional[float] = 0.7,
    # stream: bool = False, # OpenAI-compatible streaming support
    # options: Optional[Dict[str, Any]] = None
) -> OpenAIChatCompletionMessage:  # Return the official OpenAI response message type
    logger.debug(
        f"Calling OpenAI-compatible model '{model_name}' at {openai_base_url} with {len(messages)} messages."
    )
    """
    Calls an OpenAI-compatible /v1/chat/completions endpoint with the given messages and model.

    Args:
        messages: A list of CustomChatCompletionMessage objects representing the conversation history.
        model_name: The name of the model to use (e.g., "gpt-4", "llama3").
        openai_base_url: The base URL of the OpenAI-compatible server (e.g., "http://localhost:11434/v1").
        temperature: The temperature for sampling, passed to the model's options.
        # options: Optional dictionary of additional model options.

    Returns:
        A ChatMessage object containing the assistant's response.

    Raises:
        LLMConnectionError: If there's an issue connecting to the OpenAI-compatible server.
        LLMResponseError: If the OpenAI-compatible server returns an unexpected response format or an error.
    """
    # Use the passed openai_base_url parameter, applying Docker adjustment if needed
    effective_base_url = settings._adjust_for_docker(openai_base_url)
    api_url = f"{effective_base_url.rstrip('/')}/v1/chat/completions"

    # Convert CustomChatCompletionMessage Pydantic objects to dictionaries for OpenAI-compatible API
    # OpenAI-compatible APIs expect a list of {'role': 'user', 'content': '...'}, etc.
    api_messages = []
    for msg in messages:
        msg_dict = {"role": msg.role, "content": msg.content}
        # OpenAI-compatible APIs support 'name', 'tool_calls', 'tool_call_id' for advanced functionality
        # For basic chat, role and content are sufficient.
        api_messages.append(msg_dict)

    payload = {
        "model": model_name,
        "messages": api_messages,
        "stream": False,  # For now, we handle non-streamed responses
    }
    if temperature is not None:
        payload["temperature"] = temperature

    # if options:
    #     payload.setdefault("options", {}).update(options)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url, json=payload, timeout=120.0
            )  # Added a timeout
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses

    except httpx.RequestError as e:
        logger.error(
            f"httpx.RequestError calling OpenAI-compatible model '{model_name}' at {api_url}: {e}",
            exc_info=True,
        )
        # Handles connection errors, timeouts (excluding read timeouts if stream=True), etc.
        logger.error(
            f"Connection error using base URL: {effective_base_url}, full API URL: {api_url}"
        )
        raise LLMConnectionError(
            f"Error connecting to OpenAI-compatible API at {api_url}: {e}"
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error(
            f"httpx.HTTPStatusError calling OpenAI-compatible model '{model_name}' at {api_url}. Status: {e.response.status_code}, Response: {e.response.text}",
            exc_info=True,
        )
        # Handles HTTP error responses (4xx, 5xx)
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            error_detail = error_json.get("error", error_detail)
        except ValueError:
            pass  # Keep original text if not JSON
        raise LLMResponseError(
            f"OpenAI-compatible API request failed with status {e.response.status_code} at {api_url}: {error_detail}"
        ) from e

    response_data = None
    try:
        response_data = response.json()
        if response_data.get("error"):
            logger.error(
                f"OpenAI-compatible API returned an error for model '{model_name}': {response_data['error']}. Payload: {payload}"
            )
            raise LLMResponseError(
                f"OpenAI-compatible API returned an error: {response_data['error']}"
            )

        # OpenAI-compatible API response structure contains a 'choices' array with message objects
        choices = response_data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            logger.error(
                f"Unexpected response structure from OpenAI-compatible model '{model_name}': 'choices' field is missing or empty. Response: {response_data}"
            )
            raise LLMResponseError(
                f"Unexpected response structure from OpenAI-compatible API: 'choices' field is missing or empty. Response: {response_data}"
            )

        assistant_response_data = choices[0].get("message")
        if not assistant_response_data or not isinstance(assistant_response_data, dict):
            logger.error(
                f"Unexpected response structure from OpenAI-compatible model '{model_name}': 'message' field is missing or not a dict in first choice. Response: {response_data}"
            )
            raise LLMResponseError(
                f"Unexpected response structure from OpenAI-compatible API: 'message' field is missing or not a dict in first choice. Response: {response_data}"
            )

        role = assistant_response_data.get("role")
        content = assistant_response_data.get("content")

        if role != "assistant" or content is None:
            logger.error(
                f"Unexpected content in OpenAI-compatible response message for model '{model_name}'. Expected role 'assistant' and non-null content. Got role '{role}', content: '{content}'. Response: {response_data}"
            )
            raise LLMResponseError(
                f"Unexpected content in OpenAI-compatible response message. Expected role 'assistant' and non-null content. "
                f"Got role '{role}', content: '{content}'. Response: {response_data}"
            )

        logger.debug(
            f"Successfully received response from OpenAI-compatible model '{model_name}'. Role: {role}, Content length: {len(content) if content else 0}"
        )
        # Construct the official OpenAI ChatCompletionMessage for the response
        return OpenAIChatCompletionMessage(role=role, content=content)

    except ValueError as e:  # JSONDecodeError is a subclass of ValueError
        logger.error(
            f"Failed to decode JSON response from OpenAI-compatible model '{model_name}': {e}. Response text: {response.text}",
            exc_info=True,
        )
        raise LLMResponseError(
            f"Failed to decode JSON response from OpenAI-compatible API: {e}. Response text: {response.text}"
        ) from e
    except KeyError as e:
        response_data = response_data or {}
        logger.error(
            f"Missing expected key {e} in OpenAI-compatible response for model '{model_name}': {response_data}",
            exc_info=True,
        )
        raise LLMResponseError(
            f"Missing expected key {e} in OpenAI-compatible response: {response_data}"
        ) from e


# ---------------------------------------------------------------------------
# Streaming Support
# ---------------------------------------------------------------------------
async def stream_openai_compatible_chat_model(
    messages: List[CustomChatCompletionMessage],
    model_name: str,
    openai_base_url: str,
    temperature: Optional[float] = 0.7,
) -> AsyncIterator[dict]:
    """Stream responses from OpenAI-compatible API as they arrive.

    Yields each JSON chunk emitted by the API. Caller is responsible for
    converting these chunks to the desired wire format (e.g. OpenAI SSE).
    """
    # Use the passed openai_base_url parameter, applying Docker adjustment if needed
    effective_base_url = settings._adjust_for_docker(openai_base_url)
    api_url = f"{effective_base_url.rstrip('/')}/v1/chat/completions"

    api_messages = [{"role": m.role, "content": m.content} for m in messages]

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": api_messages,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", api_url, json=payload, timeout=120.0
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue  # skip keep-alive blanks

                    # Handle SSE format (data: {...}) and raw JSON lines
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break  # End of stream
                        json_line = line[6:]  # Remove "data: " prefix
                    else:
                        json_line = line  # Raw JSON line

                    try:
                        chunk = json.loads(json_line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to decode streaming line from OpenAI-compatible API: %s",
                            line,
                        )
                        continue
                    yield chunk
    except httpx.RequestError as e:
        logger.error(
            f"Connection error (stream) using base URL: {effective_base_url}, full API URL: {api_url}"
        )
        raise LLMConnectionError(
            f"Error connecting to OpenAI-compatible API at {api_url}: {e}"
        ) from e
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            error_detail = error_json.get("error", error_detail)
        except ValueError:
            pass
        logger.error(
            "httpx.HTTPStatusError (stream) calling OpenAI-compatible model '%s'. Status: %s, Response: %s",
            model_name,
            e.response.status_code,
            error_detail,
        )
        raise LLMResponseError(
            f"OpenAI-compatible API request failed with status {e.response.status_code} at {api_url}: {error_detail}"
        )
