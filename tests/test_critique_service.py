# Placeholder for critique_service tests
# We will add tests for prepare_critique_messages

from src.models.openai_types import CustomChatCompletionMessage
from openai.types.chat import ChatCompletionMessage as OpenAIChatCompletionMessage
from src.services.critique_service import prepare_critique_messages

def test_prepare_critique_messages_basic():
    original_user_messages = [
        CustomChatCompletionMessage(role="user", content="What is Python?")
    ]
    model_1_response = OpenAIChatCompletionMessage(role="assistant", content="Python is a programming language.")
    critique_system_prompt_template = (
        "Critique the following: User: {original_request} Model1: {model_response}"
    )

    critique_messages = prepare_critique_messages(
        original_request_messages=original_user_messages, # Parameter name updated in service
        model1_response_message=model_1_response, # Parameter name updated in service
        critique_task_instruction=critique_system_prompt_template
    )

    assert len(critique_messages) == 3
    assert critique_messages[0].role == "user"
    assert "What is Python?" in critique_messages[0].content
    assert critique_messages[1].role == "assistant"
    assert critique_messages[1].content == "Python is a programming language."
    assert critique_messages[2].role == "user"
    assert critique_messages[2].content == critique_system_prompt_template

# Add more tests for edge cases, multiple user messages, etc.
