import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent import capture_run_messages

from pydantic_ai_bedrock.bedrock import (  # Replace from pydantic_ai.bedrock import BedrockModel when pydantic_ai support bedrock
    BedrockModel,
)


class QuestionResponse(BaseModel):
    answer: str


model = BedrockModel(
    model_name="us.amazon.nova-lite-v1:0",
    # model_name="anthropic.claude-3-5-haiku-20241022-v1:0",
)
agent = Agent(
    model,
    system_prompt="You are a helpful assistant. You muse use a tool.",
    result_type=QuestionResponse,
)


async def main():
    async with agent.run_stream("What is the capital of the UK?") as response:
        print(await response.get_data())

    plain_agent = Agent(
        model,
        system_prompt="You are a helpful assistant.",
    )
    async with plain_agent.run_stream("hello") as response:
        print(await response.get_data())


if __name__ == "__main__":
    with capture_run_messages() as messages:
        try:
            asyncio.run(main())
        finally:
            print(messages)
