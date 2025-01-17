from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.agent import capture_run_messages

from pydantic_ai_bedrock.bedrock import (  # Replace from pydantic_ai.bedrock import BedrockModel when pydantic_ai support bedrock
    BedrockModel,
)


class TalkResponse(BaseModel):
    text: str


model = BedrockModel(
    model_name="us.amazon.nova-lite-v1:0",
    # model_name="anthropic.claude-3-5-haiku-20241022-v1:0",
)
agent = Agent(
    model,
    system_prompt="You are a helpful assistant. You muse use a tool.",
    result_type=TalkResponse,
)


@agent.result_validator
async def validate_json(response: TalkResponse) -> TalkResponse:
    if response.text != "Hello! :D":
        raise ModelRetry("You shloud say Hello! :D")
    return response


if __name__ == "__main__":
    with capture_run_messages() as messages:
        try:
            result = agent.run_sync("Say hello")
            print(result.data)
            print(result.usage())
        except Exception as e:
            print(e)
        finally:
            print(messages)
