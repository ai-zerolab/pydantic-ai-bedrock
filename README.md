![](https://img.shields.io/github/license/wh1isper/pydantic-ai-bedrock)
![](https://img.shields.io/github/v/release/wh1isper/pydantic-ai-bedrock)
![](https://img.shields.io/pypi/dm/pydantic_ai_bedrock)
![](https://img.shields.io/github/last-commit/wh1isper/pydantic-ai-bedrock)
![](https://img.shields.io/pypi/pyversions/pydantic_ai_bedrock)

# pydantic_ai_bedrock

Pydantic has support for Bedrock. See: https://github.com/pydantic/pydantic-ai/pull/994

You don't need this package now. Unless you want to use claude 3.7's extended reasoning(see [0.2.2 release](https://github.com/ai-zerolab/pydantic-ai-bedrock/releases/tag/0.2.2)).

## Install

`pip install pydantic_ai_bedrock`

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_bedrock.bedrock import (
    BedrockModel,
)  # Replace with `pydantic_ai.bedrock import BedrockModel` when pydantic_ai support bedrock

model = BedrockModel(
    model_name="anthropic.claude-3-5-sonnet-20241022-v2:0",
)
agent = Agent(model, system_prompt="You are a helpful assistant.")


if __name__ == "__main__":
    result = agent.run_sync("Hello world!")
    print(result.data)
    print(result.usage())
```

## Develop

Install pre-commit before commit

```
pip install pre-commit
pre-commit install
```

Install package locally

```
pip install -e .[test]
```
