#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# OpenAI chat usage accounting: backport MiniMax reasoning token accounting.
#

from __future__ import annotations

import ast
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine import protocol as engine_protocol
from vllm.reasoning import minimax_m2_reasoning_parser as minimax_parser


def _extract_class_method_source(
    module_path: str,
    class_name: str,
    method_name: str,
) -> str:
    source = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    method_source = ast.get_source_segment(source, item)
                    if method_source is None:
                        break
                    return textwrap.dedent(method_source)

    raise RuntimeError(
        f"Unable to extract {class_name}.{method_name} from {module_path}."
    )


def _install_method(method_name: str, method_source: str) -> None:
    namespace: dict[str, Any] = {}
    exec(method_source, chat_serving.__dict__, namespace)
    method = namespace[method_name]
    method.__module__ = OpenAIServingChat.__module__
    method.__qualname__ = f"{OpenAIServingChat.__qualname__}.{method_name}"
    setattr(OpenAIServingChat, method_name, method)


def _replace_block(
    source: str,
    old: str,
    new: str,
    *,
    count: int = 1,
) -> str:
    if source.count(old) < count:
        raise RuntimeError(
            "Failed to locate expected block while patching "
            "OpenAIServingChat usage accounting."
        )
    return source.replace(old, new, count)


class CompletionTokenUsageInfo(engine_protocol.OpenAIBaseModel):
    reasoning_tokens: int | None = None
    audio_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class UsageInfo(engine_protocol.UsageInfo):
    completion_tokens_details: CompletionTokenUsageInfo | None = None


CompletionTokenUsageInfo.__module__ = engine_protocol.__name__
UsageInfo.__module__ = engine_protocol.__name__

engine_protocol.CompletionTokenUsageInfo = CompletionTokenUsageInfo
engine_protocol.UsageInfo = UsageInfo
chat_protocol.UsageInfo = UsageInfo
chat_serving.CompletionTokenUsageInfo = CompletionTokenUsageInfo
chat_serving.UsageInfo = UsageInfo


def _rebuild_model_field(model_cls, field_name: str, annotation) -> None:
    model_cls.__annotations__[field_name] = annotation
    model_cls.model_fields[field_name].annotation = annotation
    model_cls.model_rebuild(force=True)


_rebuild_model_field(chat_protocol.ChatCompletionResponse, "usage", UsageInfo)
_rebuild_model_field(
    chat_protocol.ChatCompletionStreamResponse, "usage", UsageInfo | None
)
_rebuild_model_field(
    engine_protocol.RequestResponseMetadata, "final_usage_info", UsageInfo | None
)


def _count_minimax_reasoning_tokens(
    token_ids: Sequence[int],
    end_token_id: int | None,
) -> int:
    if end_token_id is None:
        return 0

    for idx, token_id in enumerate(token_ids):
        if token_id == end_token_id:
            return idx
    return len(token_ids)


def _patched_count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
    return _count_minimax_reasoning_tokens(token_ids, self.end_token_id)


minimax_parser.MiniMaxM2ReasoningParser.count_reasoning_tokens = (
    _patched_count_reasoning_tokens
)
minimax_parser.MiniMaxM2AppendThinkReasoningParser.count_reasoning_tokens = (
    _patched_count_reasoning_tokens
)


def _count_reasoning_tokens_for_usage(
    token_ids: Sequence[int],
    reasoning_parser,
) -> int | None:
    if reasoning_parser is None:
        return None
    return reasoning_parser.count_reasoning_tokens(token_ids)


def _make_usage_info(
    self,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    num_cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> UsageInfo:
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    if reasoning_tokens is not None:
        usage.completion_tokens_details = CompletionTokenUsageInfo(
            reasoning_tokens=max(0, min(reasoning_tokens, completion_tokens))
        )
    if self.enable_prompt_tokens_details and num_cached_tokens:
        usage.prompt_tokens_details = chat_serving.PromptTokenUsageInfo(
            cached_tokens=num_cached_tokens
        )
    return usage


OpenAIServingChat._count_reasoning_tokens_for_usage = staticmethod(
    _count_reasoning_tokens_for_usage
)
OpenAIServingChat._make_usage_info = _make_usage_info


def _patch_chat_completion_stream_generator() -> None:
    method_source = _extract_class_method_source(
        chat_serving.__file__,
        "OpenAIServingChat",
        "chat_completion_stream_generator",
    )

    method_source = _replace_block(
        method_source,
        """\
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
""",
        """\
        previous_num_tokens = [0] * num_choices
        raw_output_token_ids = [[] for _ in range(num_choices)]
        finish_reason_sent = [False] * num_choices
""",
    )

    method_source = _replace_block(
        method_source,
        """\
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )
""",
        """\
                        if include_continuous_usage:
                            chunk.usage = self._make_usage_info(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                reasoning_tokens=self._count_reasoning_tokens_for_usage(
                                    raw_output_token_ids[i], reasoning_parser
                                ),
                            )
""",
    )

    method_source = _replace_block(
        method_source,
        """\
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )
""",
        """\
                                if include_continuous_usage:
                                    chunk.usage = self._make_usage_info(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        reasoning_tokens=self._count_reasoning_tokens_for_usage(
                                            raw_output_token_ids[i], reasoning_parser
                                        ),
                                    )
""",
    )

    method_source = _replace_block(
        method_source,
        """\
                    previous_num_tokens[i] += len(output.token_ids)
""",
        """\
                    previous_num_tokens[i] += len(output.token_ids)
                    raw_output_token_ids[i].extend(as_list(output.token_ids))
""",
    )

    method_source = _replace_block(
        method_source,
        """\
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )
""",
        """\
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = self._make_usage_info(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            reasoning_tokens=self._count_reasoning_tokens_for_usage(
                                raw_output_token_ids[i], reasoning_parser
                            ),
                        )
""",
    )

    method_source = _replace_block(
        method_source,
        """\
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )
""",
        """\
                reasoning_tokens = None
                if reasoning_parser is not None:
                    reasoning_tokens = sum(
                        self._count_reasoning_tokens_for_usage(
                            token_ids, reasoning_parser
                        )
                        or 0
                        for token_ids in raw_output_token_ids
                    )
                final_usage = self._make_usage_info(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    num_cached_tokens=num_cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
""",
    )

    method_source = _replace_block(
        method_source,
        """\
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )
""",
        """\
            reasoning_tokens = None
            if reasoning_parser is not None:
                reasoning_tokens = sum(
                    self._count_reasoning_tokens_for_usage(
                        token_ids, reasoning_parser
                    )
                    or 0
                    for token_ids in raw_output_token_ids
                )
            request_metadata.final_usage_info = self._make_usage_info(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                reasoning_tokens=reasoning_tokens,
            )
""",
    )

    _install_method("chat_completion_stream_generator", method_source)


def _patch_chat_completion_full_generator() -> None:
    method_source = _extract_class_method_source(
        chat_serving.__file__,
        "OpenAIServingChat",
        "chat_completion_full_generator",
    )

    method_source = _replace_block(
        method_source,
        """\
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )
""",
        """\
        reasoning_tokens = None
        if reasoning_parser is not None:
            reasoning_tokens = sum(
                self._count_reasoning_tokens_for_usage(
                    as_list(output.token_ids), reasoning_parser
                )
                or 0
                for output in final_res.outputs
            )
        usage = self._make_usage_info(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            num_cached_tokens=final_res.num_cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )
""",
    )

    _install_method("chat_completion_full_generator", method_source)


_patch_chat_completion_stream_generator()
_patch_chat_completion_full_generator()
