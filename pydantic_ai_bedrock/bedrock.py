from __future__ import annotations

import functools
import typing
from collections.abc import Hashable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Literal, TypedDict, overload

import anyio
import boto3
from pydantic_ai import UnexpectedModelBehavior, result
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition, ToolParams
from typing_extensions import ParamSpec, assert_never

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamMetadataEventTypeDef,
        ConverseStreamOutputTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        ReasoningContentBlockOutputTypeDef,
        ToolTypeDef,
        ToolUseBlockOutputTypeDef,
    )


P = ParamSpec("P")
T = typing.TypeVar("T")


class ReasoningTextBlock(TypedDict):
    text: str
    signature: str | None = None


@dataclass
class ReasoningPart(TextPart):
    content: str = ""
    reasoningText: ReasoningTextBlock | None = None
    redactedContent: bytes | None = None

    part_kind: Literal["reasoning"] = "reasoning"

    def has_content(self) -> bool:
        return bool(self.reasoningText or self.redactedContent)


@dataclass
class ReasoningPartDelta:
    reasoning_content: ReasoningContentBlockOutputTypeDef

    part_delta_kind: Literal["reasoning"] = "reasoning"

    def apply(self, part: ReasoningPart) -> ReasoningPart:
        if part.reasoningText:
            original_reasoning_text = part.reasoningText.get("text")
            original_signature = part.reasoningText.get("signature")
        else:
            original_reasoning_text = None
            original_signature = None

        original_redacted_content = part.redactedContent

        if "reasoningText" in self.reasoning_content:
            delta_reasoning_text = self.reasoning_content["reasoningText"].get("text")
            delta_signature = self.reasoning_content["reasoningText"].get("signature")
        else:
            delta_reasoning_text = None
            delta_signature = None

        if "redactedContent" in self.reasoning_content:
            delta_redacted_content = self.reasoning_content["redactedContent"]
        else:
            delta_redacted_content = None

        new_reasoning_text = (
            delta_reasoning_text + original_reasoning_text or ""
            if delta_reasoning_text is not None
            else original_reasoning_text
        )
        new_signature = (
            delta_signature + original_signature or ""
            if delta_signature is not None
            else original_signature
        )
        new_redacted_content = (
            delta_redacted_content + original_redacted_content or b""
            if delta_redacted_content is not None
            else original_redacted_content
        )

        return replace(
            part,
            reasoningText={
                "text": new_reasoning_text,
                "signature": new_signature,
            },
            redactedContent=new_redacted_content,
        )


async def run_in_threadpool(func: typing.Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Wrapper around `anyio.to_thread.run_sync`, copied from fastapi."""
    if kwargs:  # pragma: no cover
        # run_sync doesn't accept 'kwargs', so bind them in here
        func = functools.partial(func, **kwargs)
    return await anyio.to_thread.run_sync(func, *args)


def exclude_none(data):
    """Exclude None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}


class AsyncIteratorWrapper:
    def __init__(self, sync_iterator: Iterator[T]):
        self.sync_iterator = iter(sync_iterator)

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        try:
            # Run the synchronous next() call in a thread pool
            item = await anyio.to_thread.run_sync(next, self.sync_iterator)
            return item
        except RuntimeError as e:
            if type(e.__cause__) is StopIteration:
                raise StopAsyncIteration
            else:
                raise e


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def handle_reasoning_delta(
    self,
    *,
    vendor_part_id: Hashable | None,
    reasoning_content: ReasoningContentBlockOutputTypeDef,
):
    existing_reasoning_part_and_index: tuple[ReasoningPart, int] | None = None
    if vendor_part_id is None:
        # If the vendor_part_id is None, check if the latest part is a TextPart to update
        if self._parts:
            part_index = len(self._parts) - 1
            latest_part = self._parts[part_index]
            if isinstance(latest_part, TextPart):
                existing_reasoning_part_and_index = latest_part, part_index
    else:
        # Otherwise, attempt to look up an existing TextPart by vendor_part_id
        part_index = self._vendor_id_to_part_index.get(vendor_part_id)
        if part_index is not None:
            existing_part = self._parts[part_index]
            if not isinstance(existing_part, ReasoningPart):
                raise UnexpectedModelBehavior(
                    f"Cannot apply a reasoning content delta to {existing_part=}"
                )
            existing_reasoning_part_and_index = existing_part, part_index

    if existing_reasoning_part_and_index is None:
        # There is no existing ReasoningPart part that should be updated, so create a new one
        new_part_index = len(self._parts)
        part = ReasoningPart(
            reasoningText=reasoning_content.get("reasoningText"),
            redactedContent=reasoning_content.get("redactedContent"),
        )
        if vendor_part_id is not None:
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        self._parts.append(part)
        return PartStartEvent(index=new_part_index, part=part)
    else:
        # Update the existing ReasoningPart with the new content delta
        existing_reasoning_part, part_index = existing_reasoning_part_and_index
        part_delta = ReasoningPartDelta(reasoning_content=reasoning_content)
        self._parts[part_index] = part_delta.apply(existing_reasoning_part)
        return PartDeltaEvent(index=part_index, delta=part_delta)


def patch_reasoning_handler(parts_manager):
    if hasattr(parts_manager, "handle_reasoning_delta"):
        return parts_manager

    parts_manager.handle_reasoning_delta = functools.partial(handle_reasoning_delta, parts_manager)
    return parts_manager


@dataclass
class BedrockStreamedResponse(StreamedResponse):
    _model_name: str
    _event_stream: EventStream[ConverseStreamOutputTypeDef]
    _timestamp: datetime = field(default_factory=now_utc, init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.
        """
        self._parts_manager = patch_reasoning_handler(self._parts_manager)

        chunk: ConverseStreamOutputTypeDef
        async for chunk in AsyncIteratorWrapper(self._event_stream):
            # TODO: Switch this to `match` when we drop Python 3.9 support
            if "messageStart" in chunk:
                continue
            if "messageStop" in chunk:
                continue
            if "metadata" in chunk:
                if "usage" in chunk["metadata"]:
                    self._usage += self._map_usage(chunk["metadata"])
                continue
            if "contentBlockStart" in chunk:
                index = chunk["contentBlockStart"]["contentBlockIndex"]
                start = chunk["contentBlockStart"]["start"]
                if "toolUse" in start:
                    tool_use_start = start["toolUse"]
                    tool_id = tool_use_start["toolUseId"]
                    tool_name = tool_use_start["name"]
                    yield self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=index,
                        tool_name=tool_name,
                        args=None,
                        tool_call_id=tool_id,
                    )

            if "contentBlockDelta" in chunk:
                index = chunk["contentBlockDelta"]["contentBlockIndex"]
                delta = chunk["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield self._parts_manager.handle_text_delta(
                        vendor_part_id=index, content=delta["text"]
                    )
                if "toolUse" in delta:
                    tool_use = delta["toolUse"]
                    yield self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=index,
                        tool_name=tool_use.get("name"),
                        args=tool_use.get("input") or None,  # Fix for empty string
                        tool_call_id=tool_id,
                    )
                if "reasoningContent" in delta:
                    yield self._parts_manager.handle_reasoning_delta(
                        vendor_part_id=index,
                        reasoning_content=delta["reasoningContent"],
                    )

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    def _map_usage(self, metadata: ConverseStreamMetadataEventTypeDef) -> result.Usage:
        return result.Usage(
            request_tokens=metadata["usage"]["inputTokens"],
            response_tokens=metadata["usage"]["outputTokens"],
            total_tokens=metadata["usage"]["totalTokens"],
        )


@dataclass(init=False)
class BedrockModel(Model):
    """A model that uses the Bedrock-runtime API."""

    client: BedrockRuntimeClient

    _model_name: str = field(repr=False)
    _system: str | None = field(default="bedrock", repr=False)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider, ex: openai."""
        return self._system

    def __init__(
        self,
        model_name: str,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str | None = None,
        bedrock_client: BedrockRuntimeClient | None = None,
    ):
        self._model_name = model_name
        if bedrock_client:
            self.client = bedrock_client
        else:
            self.client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolParams]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
        return tools

    def name(self) -> str:
        return self.model_name

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
        return {
            "toolSpec": {
                "name": f.name,
                "description": f.description,
                "inputSchema": {"json": f.parameters_json_schema},
            }
        }

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, result.Usage]:
        response = await self._messages_create(
            messages, False, model_settings, model_request_parameters
        )
        return await self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        response = await self._messages_create(
            messages, True, model_settings, model_request_parameters
        )
        yield BedrockStreamedResponse(_model_name=self.model_name, _event_stream=response)

    async def _process_response(
        self,
        response: ConverseResponseTypeDef,
    ) -> tuple[ModelResponse, result.Usage]:
        items: list[ModelResponsePart] = []
        for item in response["output"]["message"]["content"]:
            # TODO: Switch this to `match` when we drop Python 3.9 support
            if (
                not item.get("text")
                and not item.get("toolUse")
                and not item.get("reasoningContent")
            ):
                # Unsupported content
                continue

            if item.get("text"):
                items.append(TextPart(item["text"]))
            if item.get("toolUse"):
                items.append(
                    ToolCallPart(
                        tool_name=item["toolUse"]["name"],
                        args=item["toolUse"]["input"],
                        tool_call_id=item["toolUse"]["toolUseId"],
                    ),
                )
            if item.get("reasoningContent"):
                items.append(
                    ReasoningPart(
                        reasoningText=item["reasoningContent"].get("reasoningText"),
                        redactedContent=item["reasoningContent"].get("redactedContent"),
                    )
                )

        usage = result.Usage(
            request_tokens=response["usage"]["inputTokens"],
            response_tokens=response["usage"]["outputTokens"],
            total_tokens=response["usage"]["totalTokens"],
        )
        return ModelResponse(items, model_name=self.model_name), usage

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> EventStream[ConverseStreamOutputTypeDef]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef | EventStream[ConverseStreamOutputTypeDef]:
        tools = self._get_tools(model_request_parameters)
        support_tools_choice = (
            self.model_name.startswith("anthropic") or "anthropic" in self.model_name
        )
        if not tools or not support_tools_choice:
            tool_choice: None = None
        elif not model_request_parameters.allow_text_result:
            tool_choice = {"any": {}}
        else:
            tool_choice = {"auto": {}}

        system_prompt, bedrock_messages = self._map_message(messages)
        inference_config = self._map_inference_config(model_settings)
        toolConfig = (
            exclude_none(
                {
                    "tools": tools,
                    "toolChoice": tool_choice,
                }
            )
            if tools
            else None
        )
        additional_settings = self._map_additional_settings(model_settings)

        params = exclude_none(
            dict(
                modelId=self.model_name,
                messages=bedrock_messages,
                system=[{"text": system_prompt}],
                inferenceConfig=inference_config,
                toolConfig=toolConfig,
                additionalModelRequestFields=additional_settings,
            )
        )
        if stream:
            model_response = await run_in_threadpool(self.client.converse_stream, **params)
            model_response = model_response["stream"]
        else:
            model_response = await run_in_threadpool(self.client.converse, **params)
        return model_response

    @staticmethod
    def _map_additional_settings(model_settings: ModelSettings | None):
        model_settings = model_settings or {}
        if model_settings.get("enable_reasoning"):
            return {
                "reasoning_config": {
                    "type": "enabled",
                    "budget_tokens": model_settings.get("budget_tokens", 1024),
                },
            }

    @staticmethod
    def _map_inference_config(
        model_settings: ModelSettings | None,
    ) -> InferenceConfigurationTypeDef:
        model_settings = model_settings or {}
        return exclude_none(
            {
                "maxTokens": model_settings.get("max_tokens"),
                "temperature": model_settings.get("temperature"),
                "topP": model_settings.get("top_p"),
                "stopSequences": model_settings.get(
                    "stop_sequences"
                ),  # TODO: This is not included in model_settings yet
            }
        )

    @staticmethod
    def _map_message(
        messages: list[ModelMessage],
    ) -> tuple[str, list[MessageUnionTypeDef]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt: str = ""
        bedrock_messages: list[MessageUnionTypeDef] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt += part.content
                    elif isinstance(part, UserPromptPart):
                        bedrock_messages.append(
                            {
                                "role": "user",
                                "content": [{"text": part.content}],
                            }
                        )
                    elif isinstance(part, ToolReturnPart):
                        bedrock_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "toolResult": {
                                            "toolUseId": part.tool_call_id,
                                            "content": [{"text": part.model_response_str()}],
                                            "status": "success",
                                        }
                                    }
                                ],
                            },
                        )

                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            bedrock_messages.append(
                                {
                                    "role": "user",
                                    "content": [{"text": part.content}],
                                }
                            )
                        else:
                            bedrock_messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "toolResult": {
                                                "toolUseId": part.tool_call_id,
                                                "content": [{"text": part.model_response()}],
                                                "status": "error",
                                            }
                                        }
                                    ],
                                }
                            )
            elif isinstance(m, ModelResponse):
                content: list[ContentBlockOutputTypeDef] = []
                for item in m.parts:
                    if isinstance(item, TextPart) and not isinstance(item, ReasoningPart):
                        content.append({"text": item.content})
                    elif isinstance(item, ReasoningPart):
                        content.append({"reasoningContent": _map_reasoning_content(item)})
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(_map_tool_call(item))  # FIXME: MISSING key
                bedrock_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
            else:
                assert_never(m)
        return system_prompt, bedrock_messages


def _map_reasoning_content(
    reasoning: ReasoningPart,
) -> ReasoningContentBlockOutputTypeDef:
    if reasoning.redactedContent:
        return {"redactedContent": reasoning.redactedContent}
    response = {
        "reasoningText": {},
    }
    if reasoning.reasoningText:
        response["reasoningText"]["text"] = reasoning.reasoningText["text"]
        if reasoning.reasoningText.get("signature"):
            response["reasoningText"]["signature"] = reasoning.reasoningText["signature"]

    return response


def _map_tool_call(t: ToolCallPart) -> ToolUseBlockOutputTypeDef:
    return {
        "toolUse": {
            "toolUseId": t.tool_call_id,
            "name": t.tool_name,
            "input": t.args_as_dict(),
        }
    }
