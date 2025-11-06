# This file re-exports Pydantic models from the official OpenAI Python library
# to ensure our proxy uses the standard, up-to-date type definitions.

from typing import (
    List,
    Optional,
    Union,
    Dict,
)  # Added Dict, Any back for logit_bias and potential future use

from openai.types.chat import (
    ChatCompletion,  # For the overall response object. This is a Pydantic model.
    )
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types import CompletionUsage  # This is a Pydantic model.

from pydantic import BaseModel, Field  # Field is used by our custom response model


class CustomChatCompletionMessage(BaseModel):
    """
    Represents a message in a chat conversation for our request validation.
    This aligns with ChatCompletionMessageParam (which is a TypedDict) but is a Pydantic model.
    OpenAI's ChatCompletionMessage is for *responses*.
    """

    role: str  # Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = (
        None  # For assistant messages with tool calls
    )
    tool_call_id: Optional[str] = None  # For tool messages


class ChatCompletionRequest(BaseModel):
    """
    Represents a request body for our /v1/chat/completions endpoint.
    This aligns with the parameters for creating a chat completion in the OpenAI API.
    """

    model: str
    messages: List[CustomChatCompletionMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None  # Example: {"123": -100, "456": 100}
    user: Optional[str] = None
    api_key: Optional[str] = None  # Added for OpenAI client compatibility
    # tools: Optional[List[ChatCompletionToolParam]] = None
    # tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    # response_format: Optional[Dict[str, str]] = None # e.g. {"type": "json_object"}


# Re-exporting the main response model from OpenAI library
# This is the model our endpoint will return, matching OpenAI's structure.
# Type alias for clarity if needed, or use directly.
OpenAIChatCompletion = ChatCompletion


# Helper to create a default Usage object, as many OpenAI-compatible services don't provide token counts.
def create_default_usage() -> CompletionUsage:
    """Creates a CompletionUsage object with default (0) token counts."""
    return CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
