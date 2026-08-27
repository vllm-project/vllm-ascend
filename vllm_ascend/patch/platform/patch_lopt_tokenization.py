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

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import replace
from functools import partial
from typing import Any

from vllm_ascend import envs

# Rayon reads this setting when its global pool is initialized. Set conservative
# defaults before importing vLLM's HF renderer when LoPT is enabled, while still
# respecting explicit user tuning.
if envs.VLLM_ASCEND_LOPT_ENABLE:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm.inputs import TextPrompt, TokensPrompt  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer  # noqa: E402
from vllm.renderers.hf import HfRenderer  # noqa: E402
from vllm.renderers.params import ChatParams, TokenizeParams  # noqa: E402
from vllm.tokenizers.hf import maybe_make_thread_pool  # noqa: E402

from vllm_ascend.tokenization.lopt_tokenizer import (  # noqa: E402
    LoptConfig,
    LosslessParallelTokenizer,
)

logger = init_logger(__name__)

_PATCH_APPLIED = False
_original_hf_renderer_init = HfRenderer.__init__
_original_hf_renderer_shutdown = HfRenderer.shutdown
_original_hf_renderer_render_messages = HfRenderer.render_messages
_original_hf_renderer_render_messages_async = HfRenderer.render_messages_async
_original_hf_renderer_tokenize_prompt = HfRenderer._tokenize_prompt
_original_hf_renderer_tokenize_prompt_async = HfRenderer._tokenize_prompt_async
_original_deepseek_v4_renderer_init = DeepseekV4Renderer.__init__
_original_deepseek_v4_renderer_shutdown = DeepseekV4Renderer.shutdown
_original_deepseek_v4_renderer_render_messages = DeepseekV4Renderer.render_messages
_original_deepseek_v4_renderer_render_messages_async = DeepseekV4Renderer.render_messages_async
_original_deepseek_v4_renderer_tokenize_prompt = DeepseekV4Renderer._tokenize_prompt
_original_deepseek_v4_renderer_tokenize_prompt_async = DeepseekV4Renderer._tokenize_prompt_async

_LoptRenderer = HfRenderer | DeepseekV4Renderer


def _config_from_env() -> LoptConfig:
    return LoptConfig(
        enabled=envs.VLLM_ASCEND_LOPT_ENABLE,
        thread_workers=envs.VLLM_ASCEND_LOPT_THREAD_WORKERS,
        min_chars=envs.VLLM_ASCEND_LOPT_MIN_CHARS,
        chunk_chars=envs.VLLM_ASCEND_LOPT_CHUNK_CHARS,
        overlap_chars=envs.VLLM_ASCEND_LOPT_OVERLAP_CHARS,
        min_match_tokens=envs.VLLM_ASCEND_LOPT_MIN_MATCH_TOKENS,
        max_retries=envs.VLLM_ASCEND_LOPT_MAX_RETRIES,
        verify=envs.VLLM_ASCEND_LOPT_VERIFY,
    )


def _get_lopt(renderer: _LoptRenderer) -> LosslessParallelTokenizer | None:
    return getattr(renderer, "_ascend_lopt_tokenizer", None)


def _attach_lopt(renderer: _LoptRenderer) -> None:
    tokenizer = renderer.tokenizer
    is_fast = getattr(tokenizer, "is_fast", False) is True
    is_multimodal = renderer.mm_processor is not None

    if tokenizer is None:
        return
    if is_multimodal:
        return
    if not is_fast:
        return

    try:
        config = _config_from_env()
        maybe_make_thread_pool(tokenizer, config.thread_workers)
        renderer._ascend_lopt_tokenizer = LosslessParallelTokenizer(
            tokenizer,
            config,
        )
    except Exception as exc:
        logger.warning("LoPT initialization failed; standard tokenization will be used: %s", exc)


def _patched_hf_renderer_init(self: HfRenderer, *args: Any, **kwargs: Any) -> None:
    _original_hf_renderer_init(self, *args, **kwargs)
    _attach_lopt(self)


def _patched_deepseek_v4_renderer_init(self: DeepseekV4Renderer, *args: Any, **kwargs: Any) -> None:
    _original_deepseek_v4_renderer_init(self, *args, **kwargs)
    _attach_lopt(self)


def _chat_params_for_lopt(renderer: _LoptRenderer, params: ChatParams) -> ChatParams:
    if _get_lopt(renderer) is None:
        return params

    chat_template_kwargs = dict(params.chat_template_kwargs or {})
    chat_template_kwargs["tokenize"] = False
    return replace(params, chat_template_kwargs=chat_template_kwargs)


def _select_lopt(
    renderer: _LoptRenderer,
    text: str,
    encode_kwargs: dict[str, Any],
) -> LosslessParallelTokenizer | None:
    lopt = _get_lopt(renderer)
    return lopt if lopt is not None and lopt.can_use(text, encode_kwargs) else None


def _patched_hf_renderer_render_messages(self: HfRenderer, messages, params: ChatParams):
    return _original_hf_renderer_render_messages(self, messages, _chat_params_for_lopt(self, params))


async def _patched_hf_renderer_render_messages_async(self: HfRenderer, messages, params: ChatParams):
    return await _original_hf_renderer_render_messages_async(
        self,
        messages,
        _chat_params_for_lopt(self, params),
    )


def _patched_deepseek_v4_renderer_render_messages(
    self: DeepseekV4Renderer,
    messages,
    params: ChatParams,
):
    return _original_deepseek_v4_renderer_render_messages(self, messages, _chat_params_for_lopt(self, params))


async def _patched_deepseek_v4_renderer_render_messages_async(
    self: DeepseekV4Renderer,
    messages,
    params: ChatParams,
):
    return await _original_deepseek_v4_renderer_render_messages_async(
        self, messages, _chat_params_for_lopt(self, params)
    )


def _tokenize_prompt_with_lopt(
    renderer: _LoptRenderer,
    prompt: TextPrompt,
    params: TokenizeParams,
    standard_tokenize: Callable[..., TokensPrompt],
) -> TokensPrompt:
    encode_kwargs = params.get_encode_kwargs()
    text = prompt["prompt"]
    lopt = _select_lopt(renderer, text, encode_kwargs)
    if lopt is None:
        return standard_tokenize(renderer, prompt, params)

    prompt_token_ids = lopt.encode(text, **encode_kwargs)
    return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)


async def _tokenize_prompt_with_lopt_async(
    renderer: _LoptRenderer,
    prompt: TextPrompt,
    params: TokenizeParams,
    standard_tokenize: Callable[..., Awaitable[TokensPrompt]],
) -> TokensPrompt:
    encode_kwargs = params.get_encode_kwargs()
    text = prompt["prompt"]
    lopt = _select_lopt(renderer, text, encode_kwargs)
    if lopt is None:
        return await standard_tokenize(renderer, prompt, params)

    encode = partial(lopt.encode, text, **encode_kwargs)
    executor = getattr(renderer, "_executor", None)
    prompt_token_ids = await asyncio.get_running_loop().run_in_executor(executor, encode)
    return TokensPrompt(prompt_token_ids=prompt_token_ids, **prompt)


def _patched_hf_renderer_tokenize_prompt(
    self: HfRenderer,
    prompt: TextPrompt,
    params: TokenizeParams,
) -> TokensPrompt:
    return _tokenize_prompt_with_lopt(
        self,
        prompt,
        params,
        _original_hf_renderer_tokenize_prompt,
    )


async def _patched_hf_renderer_tokenize_prompt_async(
    self: HfRenderer,
    prompt: TextPrompt,
    params: TokenizeParams,
) -> TokensPrompt:
    return await _tokenize_prompt_with_lopt_async(
        self,
        prompt,
        params,
        _original_hf_renderer_tokenize_prompt_async,
    )


def _patched_deepseek_v4_renderer_tokenize_prompt(
    self: DeepseekV4Renderer,
    prompt: TextPrompt,
    params: TokenizeParams,
) -> TokensPrompt:
    return _tokenize_prompt_with_lopt(
        self,
        prompt,
        params,
        _original_deepseek_v4_renderer_tokenize_prompt,
    )


async def _patched_deepseek_v4_renderer_tokenize_prompt_async(
    self: DeepseekV4Renderer,
    prompt: TextPrompt,
    params: TokenizeParams,
) -> TokensPrompt:
    return await _tokenize_prompt_with_lopt_async(
        self, prompt, params, _original_deepseek_v4_renderer_tokenize_prompt_async
    )


def _shutdown_lopt(
    renderer: _LoptRenderer,
    standard_shutdown: Callable[..., None],
) -> None:
    lopt = _get_lopt(renderer)
    try:
        if lopt is not None:
            lopt.shutdown(wait=True)
    finally:
        standard_shutdown(renderer)


def _patched_hf_renderer_shutdown(self: HfRenderer) -> None:
    _shutdown_lopt(self, _original_hf_renderer_shutdown)


def _patched_deepseek_v4_renderer_shutdown(self: DeepseekV4Renderer) -> None:
    _shutdown_lopt(self, _original_deepseek_v4_renderer_shutdown)


def apply_lopt_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    HfRenderer.__init__ = _patched_hf_renderer_init
    HfRenderer.shutdown = _patched_hf_renderer_shutdown
    HfRenderer.render_messages = _patched_hf_renderer_render_messages
    HfRenderer.render_messages_async = _patched_hf_renderer_render_messages_async
    HfRenderer._tokenize_prompt = _patched_hf_renderer_tokenize_prompt
    HfRenderer._tokenize_prompt_async = _patched_hf_renderer_tokenize_prompt_async

    DeepseekV4Renderer.__init__ = _patched_deepseek_v4_renderer_init
    DeepseekV4Renderer.shutdown = _patched_deepseek_v4_renderer_shutdown
    DeepseekV4Renderer.render_messages = _patched_deepseek_v4_renderer_render_messages
    DeepseekV4Renderer.render_messages_async = _patched_deepseek_v4_renderer_render_messages_async
    DeepseekV4Renderer._tokenize_prompt = _patched_deepseek_v4_renderer_tokenize_prompt
    DeepseekV4Renderer._tokenize_prompt_async = _patched_deepseek_v4_renderer_tokenize_prompt_async
    _PATCH_APPLIED = True


if envs.VLLM_ASCEND_LOPT_ENABLE:
    apply_lopt_patch()
