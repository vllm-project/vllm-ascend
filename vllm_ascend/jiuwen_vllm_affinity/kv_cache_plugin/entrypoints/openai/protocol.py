from typing import Any, Literal

from pydantic import Field
from pydantic.fields import FieldInfo
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    OpenAIBaseModel,
)


def register_chat_request():
    if "cache_sharing" not in ChatCompletionRequest.model_fields:
        ChatCompletionRequest.model_fields["cache_sharing"] = FieldInfo.from_annotation(bool | None)
    if "cache_salt" not in ChatCompletionRequest.model_fields:
        ChatCompletionRequest.model_fields["cache_salt"] = FieldInfo.from_annotation(str | None)
    ChatCompletionRequest.model_rebuild(force=True)


class ReleaseKvCacheRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    cache_sharing: bool | None
    cache_salt: str | None
    messages_released_index: int
    tools_released_index: int | None = None
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: Literal["none"] | Literal["auto"] | Literal["required"] | ChatCompletionNamedToolChoiceParam | None = (
        "none"
    )
    documents: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )


class ReleaseKvCacheResponse(OpenAIBaseModel):
    cache_salt: str
    block_released: int
