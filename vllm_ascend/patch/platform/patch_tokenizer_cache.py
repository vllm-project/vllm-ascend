#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Segment-level tokenizer cache for high prefix-cache-hit serving.

Prefix caching removes the compute for a repeated conversation prefix, but
tokenization stays O(full prompt) on every turn: the token ids must be known
before the KV cache can be consulted. In agent workloads, where turn N re-sends
turns 1..N-1 verbatim and appends a short suffix, that leaves the tokenizer as a
large share of the frontend cost once the hit rate is high.

Enable with ``VLLM_ASCEND_TOKENIZER_CACHE_GB``; ``0`` (the default) is a no-op.

All four chat entry points bottom out in a *synchronous* function -
``safe_apply_chat_template`` for the HF renderer and ``_apply_chat_template``
for the DeepSeek ones - with the async variants built by ``make_async`` at
renderer construction. Platform patches run while the CLI args are parsed,
before any renderer exists, so wrapping the sync entry points covers the async
paths too.
"""

import weakref

from vllm.logger import init_logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.tokenizer_cache import IncrementalTokenizerCache

logger = init_logger(__name__)

# ``safe_apply_chat_template`` is a module-level function with no ``self``, so
# the cache is looked up by the tokenizer it is handed. Keyed weakly so a
# discarded renderer does not pin its tokenizer.
_CACHES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _arm_on_first_use(cache, render, chat_kwargs) -> bool:
    """Verify the chat identity once, using the first real request.

    ``apply_chat_template(tokenize=True)`` matching
    ``encode(apply_chat_template(tokenize=False))`` is a property of the
    template, not a guarantee, so it is checked rather than assumed. Using the
    live request as the probe makes the check stronger than a synthetic one; it
    costs one extra render+encode, once per process.
    """
    if cache.chat_arm_attempted:
        return cache.chat_path_enabled
    cache.chat_arm_attempted = True
    try:
        fused = render(**{**chat_kwargs, "tokenize": True})
        text = render(**{**chat_kwargs, "tokenize": False})
    except Exception:
        logger.debug("tokenizer cache: chat probe raised", exc_info=True)
        return False
    if not isinstance(text, str):
        return False
    return cache.arm_chat_path([(text, fused)])


def _chat_ids_via_cache(cache, render, chat_kwargs):
    """Token ids from the cache, or ``None`` to fall through to the original."""
    if cache is None or chat_kwargs.get("tokenize", True) is False:
        return None
    if not cache.chat_path_enabled and not _arm_on_first_use(cache, render, chat_kwargs):
        return None
    text = render(**{**chat_kwargs, "tokenize": False})
    if not isinstance(text, str):
        return None
    return cache.encode(text)


def _patch_renderer_init() -> None:
    from vllm.renderers.base import BaseRenderer

    original = BaseRenderer.__init__

    def patched(self, config, tokenizer):
        original(self, config, tokenizer)
        self._tokenizer_cache = None
        capacity_gb = envs_ascend.VLLM_ASCEND_TOKENIZER_CACHE_GB
        if capacity_gb <= 0 or tokenizer is None:
            return
        cache = IncrementalTokenizerCache(tokenizer, capacity_gb)
        if cache.enabled:
            cache.chat_arm_attempted = False
            self._tokenizer_cache = cache
            _CACHES[tokenizer] = cache

    BaseRenderer.__init__ = patched


def _patch_tokenize_prompt() -> None:
    """The plain ``/v1/completions`` path."""
    from vllm.renderers.base import BaseRenderer

    original = BaseRenderer._tokenize_prompt

    def patched(self, prompt, params):
        cache = getattr(self, "_tokenizer_cache", None)
        if cache is not None and not self._wants_offsets(prompt, params):
            kwargs = params.get_encode_kwargs()
            if cache.is_eligible(add_special_tokens=kwargs.get("add_special_tokens", True)):
                token_ids = cache.encode(prompt["prompt"])
                max_length = kwargs.get("max_length")
                truncated = kwargs.get("truncation") and max_length is not None and len(token_ids) > max_length
                if not truncated:
                    return self._build_tokens_prompt(token_ids, prompt)
        return original(self, prompt, params)

    BaseRenderer._tokenize_prompt = patched


def _patch_hf_chat() -> None:
    """The HF renderer chat path (sync and, via make_async, async)."""
    import vllm.renderers.hf as hf_mod

    original = hf_mod.safe_apply_chat_template

    def patched(model_config, tokenizer, conversation, **kwargs):
        cache = _CACHES.get(tokenizer)
        if cache is not None and not kwargs.get("return_assistant_tokens_mask"):

            def render(**kw):
                return original(model_config, tokenizer, conversation, **kw)

            token_ids = _chat_ids_via_cache(cache, render, kwargs)
            if token_ids is not None:
                return token_ids
        return original(model_config, tokenizer, conversation, **kwargs)

    hf_mod.safe_apply_chat_template = patched


def _patch_deepseek_chat() -> None:
    """The DeepSeek-V4 / V3.2 renderer chat paths."""
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

    renderers = [DeepseekV4Renderer]
    try:
        from vllm.renderers.deepseek_v32 import DeepseekV32Renderer

        renderers.append(DeepseekV32Renderer)
    except ImportError:  # pragma: no cover - older vLLM
        pass

    for renderer_cls in renderers:
        original = renderer_cls._apply_chat_template

        def make_patched(original=original):

            def patched(self, *args, **kwargs):
                cache = getattr(self, "_tokenizer_cache", None)

                def render(**kw):
                    return original(self, *args, **kw)

                token_ids = _chat_ids_via_cache(cache, render, kwargs)
                if token_ids is not None:
                    return token_ids
                return original(self, *args, **kwargs)

            return patched

        renderer_cls._apply_chat_template = make_patched()


if envs_ascend.VLLM_ASCEND_TOKENIZER_CACHE_GB > 0:
    _patch_renderer_init()
    _patch_tokenize_prompt()
    _patch_hf_chat()
    _patch_deepseek_chat()
    logger.info(
        "Incremental tokenizer cache patch installed (%.2f GiB per API server process).",
        envs_ascend.VLLM_ASCEND_TOKENIZER_CACHE_GB,
    )
