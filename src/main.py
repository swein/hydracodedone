import logging
import uuid
import json
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    HTTPException,
    Request as FastAPIRequest,
)  # Renamed to avoid conflict
from fastapi.responses import StreamingResponse
from .models.openai_types import (
    ChatCompletionRequest,
    OpenAIChatCompletion,
    create_default_usage,
)
from openai.types.chat.chat_completion import (
    Choice as OpenAIChatCompletionChoice,
)
from openai.types.chat import (
    ChatCompletionMessage as OpenAIChatCompletionMessage,
)  # For type hinting internal messages
from .services.llm_clients import (
    call_openai_compatible_chat_model,
    stream_openai_compatible_chat_model,
    LLMConnectionError,
    LLMResponseError,
)
from .services.critique_service import prepare_critique_messages
from .core.config import settings
from .core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with startup validation."""
    # Startup
    try:
        await validate_openai_base_url()
    except HTTPException as e:
        logger.error(f"Startup validation failed: {e.detail}")
        logger.warning(
            "Application will continue, but API functionality may be limited"
        )
    except Exception as e:
        logger.error(f"Unexpected error during startup validation: {e}", exc_info=True)
        logger.warning(
            "Application will continue, but API functionality may be limited"
        )
    
    yield
    
    # Shutdown (if needed in the future)
    pass


app = FastAPI(
    title="HydraCodeDone LLM Critique Proxy",
    description="A proxy server to enhance AI-assisted coding with a dual-model critique pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


async def validate_openai_base_url():
    """
    Validate that the configured OPENAI_BASE_URL has a working /v1/models endpoint.
    This runs on startup to ensure the downstream API is accessible and returns expected format.
    """
    try:
        api_url = f"{settings.effective_openai_base_url.rstrip('/')}/v1/models"
        logger.info(f"Validating OpenAI Base URL at: {api_url}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_url)
            response.raise_for_status()

            response_data = response.json()

            # Validate the expected OpenAI models endpoint structure
            if not isinstance(response_data, dict):
                logger.error(
                    f"OpenAI Base URL /v1/models endpoint did not return a dictionary. Response: {response_data}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"OpenAI Base URL /v1/models endpoint returned invalid format: expected dict, got {type(response_data)}",
                )

            if response_data.get("object") != "list":
                logger.error(
                    f"OpenAI Base URL /v1/models endpoint missing 'object': 'list' field. Response: {response_data}"
                )
                raise HTTPException(
                    status_code=503,
                    detail="OpenAI Base URL /v1/models endpoint missing required 'object': 'list' field",
                )

            data = response_data.get("data")
            if not isinstance(data, list):
                logger.error(
                    f"OpenAI Base URL /v1/models endpoint missing 'data' array. Response: {response_data}"
                )
                raise HTTPException(
                    status_code=503,
                    detail="OpenAI Base URL /v1/models endpoint missing required 'data' array",
                )

            # Check if any models are available
            if len(data) == 0:
                logger.warning(
                    f"OpenAI Base URL /v1/models endpoint returned empty models list. Response: {response_data}"
                )
            else:
                # Validate model format
                for i, model in enumerate(data):
                    if not isinstance(model, dict):
                        logger.error(
                            f"OpenAI Base URL /v1/models endpoint returned invalid model format at index {i}. Response: {response_data}"
                        )
                        raise HTTPException(
                            status_code=503,
                            detail=f"OpenAI Base URL /v1/models endpoint returned invalid model format at index {i}",
                        )

                    if not model.get("id"):
                        logger.error(
                            f"OpenAI Base URL /v1/models endpoint returned model missing 'id' field at index {i}. Response: {response_data}"
                        )
                        raise HTTPException(
                            status_code=503,
                            detail=f"OpenAI Base URL /v1/models endpoint returned model missing 'id' field at index {i}",
                        )

                    if model.get("object") != "model":
                        logger.error(
                            f"OpenAI Base URL /v1/models endpoint returned model missing 'object': 'model' field at index {i}. Response: {response_data}"
                        )
                        raise HTTPException(
                            status_code=503,
                            detail=f"OpenAI Base URL /v1/models endpoint returned model missing 'object': 'model' field at index {i}",
                        )

            # Check if primary model is available
            available_model_ids = [model.get("id") for model in data if model.get("id")]

            if settings.PRIMARY_MODEL_NAME not in available_model_ids:
                logger.error(
                    f"Primary model '{settings.PRIMARY_MODEL_NAME}' not found in available models: {available_model_ids}"
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Primary model '{settings.PRIMARY_MODEL_NAME}' not found in available models: {available_model_ids}",
                )

            logger.info(
                f"OpenAI Base URL validation successful. Found {len(data)} models: {available_model_ids}"
            )

            # Validate critique base URL if it's different from primary
            logger.info(
                f"OPENAI_CRITIQUE_BASE_URL setting: {settings.OPENAI_CRITIQUE_BASE_URL}"
            )
            logger.info(
                f"Effective critique base URL: {settings.effective_critique_base_url}"
            )
            if settings.OPENAI_CRITIQUE_BASE_URL:
                try:
                    critique_api_url = (
                        f"{settings.effective_critique_base_url.rstrip('/')}/v1/models"
                    )
                    logger.info(
                        f"Validating OpenAI Critique Base URL at: {critique_api_url}"
                    )

                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(critique_api_url)
                        response.raise_for_status()

                        critique_response_data = response.json()

                        # Validate basic structure
                        if (
                            not isinstance(critique_response_data, dict)
                            or critique_response_data.get("object") != "list"
                        ):
                            logger.error(
                                f"Invalid response from critique /v1/models endpoint: {critique_response_data}"
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Critique Base URL /v1/models endpoint returned invalid format",
                            )

                        critique_data = critique_response_data.get("data", [])
                        if not isinstance(critique_data, list):
                            logger.error(
                                f"Invalid 'data' field from critique /v1/models endpoint: {critique_response_data}"
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Critique Base URL /v1/models endpoint missing required 'data' array",
                            )

                        # Check if critique model is available in critique API
                        critique_model_ids = [
                            model.get("id")
                            for model in critique_data
                            if model.get("id")
                        ]
                        if (
                            settings.CRITIQUE_MODEL_NAME
                            and settings.CRITIQUE_MODEL_NAME not in critique_model_ids
                        ):
                            logger.error(
                                f"Critique model '{settings.CRITIQUE_MODEL_NAME}' not found in critique API models: {critique_model_ids}"
                            )
                            raise HTTPException(
                                status_code=503,
                                detail=f"Critique model '{settings.CRITIQUE_MODEL_NAME}' not found in critique API models: {critique_model_ids}",
                            )

                        logger.info(
                            f"OpenAI Critique Base URL validation successful. Found {len(critique_data)} models: {critique_model_ids}"
                        )

                except httpx.RequestError as e:
                    logger.error(
                        f"Failed to connect to OpenAI Critique Base URL at {settings.effective_critique_base_url}/v1/models: {e}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to connect to OpenAI Critique Base URL at {settings.effective_critique_base_url}: {e}",
                    )
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"OpenAI Critique Base URL /v1/models endpoint returned HTTP {e.response.status_code}: {e.response.text}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"OpenAI Critique Base URL /v1/models endpoint returned HTTP {e.response.status_code}: {e.response.text}",
                    )
                except ValueError as e:  # JSON decode error
                    logger.error(
                        f"OpenAI Critique Base URL /v1/models endpoint returned invalid JSON: {e}. Response text: {response.text}"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"OpenAI Critique Base URL /v1/models endpoint returned invalid JSON: {e}",
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error validating OpenAI Critique Base URL: {e}",
                        exc_info=True,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"Unexpected error validating OpenAI Critique Base URL: {e}",
                    )

    except httpx.RequestError as e:
        logger.error(
            f"Failed to connect to OpenAI Base URL at {settings.effective_openai_base_url}/v1/models: {e}"
        )
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to OpenAI Base URL at {settings.effective_openai_base_url}: {e}",
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            f"OpenAI Base URL /v1/models endpoint returned HTTP {e.response.status_code}: {e.response.text}"
        )
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI Base URL /v1/models endpoint returned HTTP {e.response.status_code}: {e.response.text}",
        )
    except ValueError as e:  # JSON decode error
        logger.error(
            f"OpenAI Base URL /v1/models endpoint returned invalid JSON: {e}"
        )
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI Base URL /v1/models endpoint returned invalid JSON: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error validating OpenAI Base URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"Unexpected error validating OpenAI Base URL: {e}"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint that validates downstream API connectivity.
    """
    health_status = {"status": "ok", "components": {}}

    # Check downstream API connectivity
    try:
        api_url = f"{settings.effective_openai_base_url.rstrip('/')}/v1/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            response_data = response.json()

            # Validate response structure
            if (
                isinstance(response_data, dict)
                and response_data.get("object") == "list"
                and isinstance(response_data.get("data"), list)
            ):
                health_status["components"]["downstream_api"] = {
                    "status": "healthy",
                    "url": settings.effective_openai_base_url,
                    "models_count": len(response_data.get("data", [])),
                }

                # Add critique API health if separate
                if settings.OPENAI_CRITIQUE_BASE_URL:
                    try:
                        critique_api_url = f"{settings.effective_critique_base_url.rstrip('/')}/v1/models"
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            critique_response = await client.get(critique_api_url)
                            critique_response.raise_for_status()
                            critique_data = critique_response.json()

                            health_status["components"]["critique_api"] = {
                                "status": "healthy",
                                "url": settings.effective_critique_base_url,
                                "models_count": len(critique_data.get("data", [])),
                            }
                    except Exception as e:
                        health_status["components"]["critique_api"] = {
                            "status": "unhealthy",
                            "url": settings.effective_critique_base_url,
                            "error": str(e),
                        }
                        if health_status["status"] == "ok":
                            health_status["status"] = "degraded"
            else:
                health_status["components"]["downstream_api"] = {
                    "status": "unhealthy",
                    "url": settings.effective_openai_base_url,
                    "error": "Invalid response format",
                }
                health_status["status"] = "degraded"

    except Exception as e:
        health_status["components"]["downstream_api"] = {
            "status": "unhealthy",
            "url": settings.effective_openai_base_url,
            "error": str(e),
        }
        health_status["status"] = "unhealthy"

    # Check configured models
    health_status["components"]["models"] = {
        "primary_model": settings.PRIMARY_MODEL_NAME,
        "critique_model": settings.CRITIQUE_MODEL_NAME,
    }

    return health_status


@app.get("/v1/health", tags=["Health"])
async def openai_health_check():
    """OpenAI-style health probe expected by some clients (Continue.dev)."""
    try:
        api_url = f"{settings.effective_openai_base_url.rstrip('/')}/v1/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url)
            response.raise_for_status()
            return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/v1/models", tags=["Models"])
async def list_models():
    """Return the HydraCodeDone proxy model that handles the dual-model critique pipeline."""
    logger.debug("Returning HydraCodeDone proxy model")
    return {
        "object": "list",
        "data": [
            {
                "id": "hydracodedone",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hydracodedone-proxy",
            }
        ],
    }





@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(
    request: ChatCompletionRequest, http_request: FastAPIRequest
):
    client_host = http_request.client.host if http_request.client else "unknown_client"
    logger.info(
        f"Received chat completion request from {client_host} for model {request.model}, stream={request.stream}"
    )
    """
    OpenAI-compatible chat completions endpoint.
    This endpoint takes a user request, calls a primary LLM (Model 1 via any OpenAI-compatible service),
    and returns its response. The optional critique pipeline (Model 2) refines the response.
    """

    # Streaming with critique pipeline
    if request.stream:

        async def event_stream():
            try:
                # Step 1: Get Model 1's complete response (non-streaming for critique context)
                logger.info(f"Getting Model 1 response for streaming request from {client_host}")
                model_1_response_message = await call_openai_compatible_chat_model(
                    messages=request.messages,
                    model_name=settings.PRIMARY_MODEL_NAME,
                    openai_base_url=settings.effective_openai_base_url,
                    temperature=request.temperature,
                )

                # Step 2: Prepare for Model 2 critique if configured
                if not settings.CRITIQUE_MODEL_NAME or not settings.CRITIQUE_SYSTEM_PROMPT:
                    # No critique model configured, stream Model 1's response directly
                    logger.info("No critique model configured, streaming Model 1 response directly")
                    # Convert Model 1's response to streaming format
                    content = model_1_response_message.content or ""
                    content_chunks = [content[i:i+4] for i in range(0, len(content), 4)]
                    
                    for i, chunk_content in enumerate(content_chunks):
                        if chunk_content:
                            data = {
                                "id": f"chatcmpl-{uuid.uuid4().hex}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk_content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    
                    # Send final chunk
                    finish = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Step 3: Get critique messages and stream Model 2's response
                logger.info("Preparing critique messages for Model 2")
                critique_messages = prepare_critique_messages(
                    original_request_messages=request.messages,
                    model1_response_message=model_1_response_message,
                    critique_task_instruction=settings.CRITIQUE_SYSTEM_PROMPT,
                )

                logger.info(f"Streaming Model 2 (critique) response for {client_host}")
                chunk_iter = stream_openai_compatible_chat_model(
                    messages=critique_messages,
                    model_name=settings.CRITIQUE_MODEL_NAME,
                    openai_base_url=settings.effective_critique_base_url,
                    temperature=request.temperature,
                )
                
                async for chunk in chunk_iter:
                    # OpenAI-compatible streaming responses have 'choices' array
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # If there's content in the delta, stream it
                    if delta.get("content"):
                        data = {
                            "id": chunk.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                            "object": "chat.completion.chunk",
                            "created": chunk.get("created", int(time.time())),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta.get("content")},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    # If this is the final chunk with finish_reason, send it and end
                    if finish_reason:
                        finish = {
                            "id": chunk.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                            "object": "chat.completion.chunk",
                            "created": chunk.get("created", int(time.time())),
                            "model": request.model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": finish_reason}
                            ],
                        }
                        yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

            except LLMConnectionError as e:
                logger.error(
                    f"LLMConnectionError during streaming for {client_host}: {e}", exc_info=True
                )
                error_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }
                    ],
                    "error": {"message": f"Connection error: {e}"}
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except LLMResponseError as e:
                logger.error(
                    f"LLMResponseError during streaming for {client_host}: {e}", exc_info=True
                )
                error_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }
                    ],
                    "error": {"message": f"Response error: {e}"}
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(
                    f"Unexpected error during streaming for {client_host}: {e}", exc_info=True
                )
                error_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }
                    ],
                    "error": {"message": f"Unexpected error: {e}"}
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    final_assistant_message: OpenAIChatCompletionMessage

    try:
        # Call Model 1 (Primary LLM)
        model_1_response_message = await call_openai_compatible_chat_model(
            messages=request.messages,
            model_name=settings.PRIMARY_MODEL_NAME,
            openai_base_url=settings.effective_openai_base_url,
            temperature=request.temperature,
        )

    except LLMConnectionError as e:
        logger.error(
            f"LLMConnectionError for Model 1 from {client_host}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=503, detail=f"Error connecting to Primary LLM (Model 1): {e}"
        )
    except LLMResponseError as e:
        logger.error(
            f"LLMResponseError for Model 1 from {client_host}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=502,
            detail=f"Error receiving response from Primary LLM (Model 1): {e}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during Model 1 call from {client_host}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during Model 1 call: {e}",
        )

    # Prepare messages for Model 2 (Critique Model)
    if not settings.CRITIQUE_MODEL_NAME or not settings.CRITIQUE_SYSTEM_PROMPT:
        # If critique model is not configured, just return Model 1's response
        final_assistant_message = model_1_response_message
    else:
        try:
            critique_messages = prepare_critique_messages(
                original_request_messages=request.messages,  # Ensure this matches the param name in critique_service
                model1_response_message=model_1_response_message,  # Ensure this matches the param name
                critique_task_instruction=settings.CRITIQUE_SYSTEM_PROMPT,
            )

            # Call Model 2 (Critique Model)
            model_2_response_message = await call_openai_compatible_chat_model(
                messages=critique_messages,
                model_name=settings.CRITIQUE_MODEL_NAME,
                openai_base_url=settings.effective_critique_base_url,  # Use critique-specific URL if configured
                temperature=request.temperature,  # Or a different temperature for critique
            )
            final_assistant_message = model_2_response_message

        except LLMConnectionError as e:
            logger.error(
                f"LLMConnectionError for Model 2 from {client_host}: {e}. Model 1 response was: {model_1_response_message.content}",
                exc_info=True,
            )
            # Consider if we should fallback to Model 1's response or error out
            # For now, error out, but log that Model 1 was successful.
            raise HTTPException(
                status_code=503,
                detail=f"Error connecting to Critique LLM (Model 2): {e}. Model 1 response was: {model_1_response_message.content}",
            )
        except LLMResponseError as e:
            logger.error(
                f"LLMResponseError for Model 2 from {client_host}: {e}. Model 1 response was: {model_1_response_message.content}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Error receiving response from Critique LLM (Model 2): {e}. Model 1 response was: {model_1_response_message.content}",
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during Model 2 call from {client_host}: {e}. Model 1 response was: {model_1_response_message.content}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred during Model 2 call: {e}. Model 1 response was: {model_1_response_message.content}",
            )

    # Create a choice containing the final assistant message (either from Model 1 or Model 2)
    choice = OpenAIChatCompletionChoice(
        index=0,
        message=final_assistant_message,
        finish_reason="stop",  # Assuming 'stop' is appropriate; backend might provide other reasons
        logprobs=None,  # Explicitly set to None if not available
    )

    # Construct the full OpenAI-compatible response object
    response = OpenAIChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,  # Echo back the requested model, or use the actual model name if it differs
        choices=[choice],
        usage=create_default_usage(),  # Use the helper for default token counts
        # system_fingerprint can be added if available/relevant
    )
    logger.info(
        f"Successfully processed chat completion request from {client_host} for model {request.model}. Returning refined response."
    )
    return response


# Alias without the /v1 prefix because some clients (e.g., Continue.dev) call it directly.
@app.post("/chat/completions", response_model=OpenAIChatCompletion, tags=["Chat"])
async def chat_completions_alias(
    request: ChatCompletionRequest, http_request: FastAPIRequest
):
    return await chat_completions(request, http_request)
