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
"""Segment-level (incremental) tokenizer cache for high prefix-cache-hit serving.

Prefix caching removes the *compute* for a repeated conversation prefix, but
tokenization stays ``O(full prompt)`` on every turn: the token ids must be known
before the KV cache can be consulted. In agent workloads, where turn ``N``
re-sends turns ``1..N-1`` verbatim and appends a short suffix, that leaves the
tokenizer as a large share of the frontend cost once the hit rate is high.

This module removes the redundant work by caching tokenization *per segment*
instead of per prompt. A whole-prompt cache is useless here - turn ``N``'s
prompt is never byte-identical to turn ``N-1``'s - but a segment cache hits on
every repeated turn.

Correctness rests on one identity: encoding ``A + B`` must give exactly the
concatenation of encoding ``A`` and encoding ``B``. That holds when the cut
falls immediately after an *added token* (a special token such as
``<|im_end|>``), because HF tokenizers split the input on the added vocabulary
**before** the pre-tokenizer and BPE ever run, so a merge can never span an
added token. Chat templates emit one at every message boundary, which is
exactly where this module cuts. It is still a property of the tokenizer rather
than a guarantee - a Metaspace/SentencePiece pre-tokenizer that prepends a
space to each segment, or a model with no added tokens at all, breaks it - so
the identity is verified empirically at renderer construction against a probe
corpus and the cache disables itself if the check does not pass.

Enable with ``VLLM_ASCEND_TOKENIZER_CACHE_GB``; ``0`` (the default) is a no-op.

Patched symbols:

* ``vllm.renderers.base.BaseRenderer.__init__`` - builds the cache once the
  renderer's tokenizer is known, and registers it for the module-level entry
  point below.
* ``vllm.renderers.base.BaseRenderer._tokenize_prompt`` - the plain
  ``/v1/completions`` path.
* ``vllm.renderers.hf.safe_apply_chat_template`` - the HF renderer chat path.
  It is a module-level function, and ``HfRenderer`` builds its async variant
  with ``make_async`` at construction, so wrapping this one covers both.
* ``DeepseekV4Renderer._apply_chat_template`` and
  ``DeepseekV32Renderer._apply_chat_template`` - the DeepSeek chat paths, whose
  async variants are likewise built from the sync method with ``make_async``.
"""

import sys
import threading
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import regex as re
from vllm.logger import logger
from vllm.tokenizers.protocol import TokenizerLike
from vllm.utils.cache import CacheInfo, LRUCache

import vllm_ascend.envs as envs_ascend

GiB = 1 << 30

# Rough per-entry accounting. Token ids are large enough to fall outside
# CPython's small-int cache, so each cached id is a distinct object.
_BYTES_PER_TOKEN_ID = 28


def _probe_corpus(special_tokens: Sequence[str]) -> list[str]:
    """Build probe strings that exercise the boundary cases that break the
    concatenation identity.

    The probes are assembled from the tokenizer's *own* added tokens so the
    check is meaningful for any model, not just chat models we know about.
    """
    if not special_tokens:
        return []

    # Deterministic pick so the check is reproducible across restarts.
    picks = sorted(special_tokens)[: min(3, len(special_tokens))]
    a, b = picks[0], picks[-1]

    code = 'def f(x: int) -> dict:\n    return {"a": x, "b": [1, 2, 3]}\n'
    return [
        # The common shape: text, boundary, text.
        f"hello world{a}second segment here{b}third",
        # Boundaries back to back, with nothing between them.
        f"{a}{b}{a}",
        # A boundary at the very start and the very end.
        f"{a}leading and trailing{b}",
        # Whitespace either side of a boundary - the classic Metaspace trap.
        f"trailing spaces   {a}   leading spaces",
        # Newlines, which chat templates always put next to boundaries.
        f"line one\n{a}\nline two\n\n{b}\n",
        # Multi-byte text straddling a boundary.
        f"你好世界\U0001f30f{a}再来一次{b}ok",
        # Code and JSON, i.e. what an agent transcript actually contains.
        f"{code}{a}{code}{b}{code}",
        # Long-ish body so at least one probe exercises a real BPE workload.
        (code * 40) + a + (code * 40),
    ]


class IncrementalTokenizerCache:
    """Caches tokenization results per added-token-delimited segment.

    Thread-safe: the renderer runs this from a `ThreadPoolExecutor` sized by
    ``--renderer-num-workers``. The lock only covers the dict operations, never
    the encode itself, so concurrent misses still tokenize in parallel (the HF
    fast tokenizer releases the GIL).
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        capacity_gb: float,
    ) -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()
        self._cache: LRUCache[str, list[int]] = LRUCache(
            capacity=max(1.0, capacity_gb * GiB),
            getsizeof=self._sizeof,
        )

        self._enabled = False
        # Whether ``add_special_tokens=True`` is a no-op for this tokenizer.
        # When it is (Qwen-style templates that emit their own specials), the
        # cache can serve those requests too.
        self._special_tokens_are_noop = False
        # Tri-state: None until the chat identity has been probed, then the
        # verdict. The chat fast path rests on a second identity on top of the
        # concatenation one, so it is armed separately.
        self._chat_ok: bool | None = None

        self._pattern = self._build_pattern()
        if self._pattern is None:
            logger.warning(
                "Incremental tokenizer cache disabled: this tokenizer exposes "
                "no added/special tokens, so there are no safe split points."
            )
            return

        self._enabled, self._special_tokens_are_noop = self._self_check()

    # ---------------------------------------------------------------- setup

    def _added_tokens(self) -> list[str]:
        """The exact set HF's ``AddedVocabulary`` splits on, when available."""
        get_added_vocab = getattr(self._tokenizer, "get_added_vocab", None)
        if callable(get_added_vocab):
            try:
                added = list(get_added_vocab().keys())
                if added:
                    return added
            except Exception:  # pragma: no cover - defensive
                logger.debug("get_added_vocab() failed", exc_info=True)

        try:
            return [t for t in self._tokenizer.all_special_tokens if t]
        except Exception:  # pragma: no cover - defensive
            logger.debug("all_special_tokens failed", exc_info=True)
            return []

    def _build_pattern(self) -> re.Pattern[str] | None:
        tokens = [t for t in self._added_tokens() if t]
        if not tokens:
            return None
        # Longest first so that a token which is a prefix of another cannot
        # shadow it (e.g. "<|im_end|>" vs "<|im_end|>\n" if both were added).
        tokens.sort(key=len, reverse=True)
        return re.compile("|".join(re.escape(t) for t in tokens))

    def _self_check(self) -> tuple[bool, bool]:
        """Verify the concatenation identity before trusting the cache.

        Returns ``(enabled, add_special_tokens_is_noop)``.
        """
        specials = self._added_tokens()
        probes = _probe_corpus(specials)
        if not probes:
            return False, False

        try:
            for text in probes:
                reference = list(self._tokenizer(text, add_special_tokens=False)["input_ids"])
                spliced: list[int] = []
                for segment in self._split(text):
                    spliced.extend(self._tokenizer(segment, add_special_tokens=False)["input_ids"])
                if spliced != reference:
                    logger.warning(
                        "Incremental tokenizer cache disabled: the segment "
                        "concatenation identity does not hold for this "
                        "tokenizer (%s). Tokenization will be unaffected.",
                        type(self._tokenizer).__name__,
                    )
                    return False, False
        except Exception:
            logger.warning(
                "Incremental tokenizer cache disabled: the startup self-check raised. Tokenization will be unaffected.",
                exc_info=True,
            )
            return False, False

        # Separately, find out whether add_special_tokens=True changes anything.
        # If it does not, requests that ask for it can also be served.
        noop = False
        try:
            noop = all(
                list(self._tokenizer(text, add_special_tokens=True)["input_ids"])
                == list(self._tokenizer(text, add_special_tokens=False)["input_ids"])
                for text in probes
            )
        except Exception:  # pragma: no cover - defensive
            logger.debug("add_special_tokens probe failed", exc_info=True)

        logger.info(
            "Incremental tokenizer cache enabled (capacity %.2f GiB, "
            "%d split tokens, add_special_tokens is %sa no-op).",
            self._cache.capacity / GiB,
            len(specials),
            "" if noop else "not ",
        )
        return True, noop

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _sizeof(ids: list[int]) -> int:
        return sys.getsizeof(ids) + _BYTES_PER_TOKEN_ID * len(ids)

    def _split(self, text: str) -> list[str]:
        """Cut ``text`` immediately after every added token.

        Every segment but the last therefore *ends* with an added token, which
        is the only cut point where the concatenation identity is guaranteed.
        """
        assert self._pattern is not None
        segments: list[str] = []
        last = 0
        for match in self._pattern.finditer(text):
            end = match.end()
            segments.append(text[last:end])
            last = end
        if last < len(text):
            segments.append(text[last:])
        return segments

    # ---------------------------------------------------------------- public

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stat(self) -> CacheInfo:
        """Segment-level hit/total counts."""
        return self._cache.stat()

    def is_eligible(self, *, add_special_tokens: bool) -> bool:
        if not self._enabled:
            return False
        return not add_special_tokens or self._special_tokens_are_noop

    def encode(self, text: str) -> list[int]:
        """Tokenize `text`, reusing cached segments where possible.

        Callers must have checked `is_eligible` first.

        Args:
            text: The text to tokenize.

        Returns:
            Token ids, bit-identical to
            `tokenizer(text, add_special_tokens=False)["input_ids"]`.
        """
        token_ids: list[int] = []
        for segment in self._split(text):
            with self._lock:
                cached = self._cache.get(segment)
            if cached is not None:
                token_ids.extend(cached)
                continue

            encoded = list(self._tokenizer(segment, add_special_tokens=False)["input_ids"])
            # A concurrent miss may have inserted this already; harmless, the
            # values are equal by construction. cachetools raises ValueError
            # when a single value exceeds the capacity.
            try:
                with self._lock:
                    self._cache[segment] = encoded
            except ValueError:
                pass
            token_ids.extend(encoded)

        return token_ids

    def verify_chat_path(self, render: Callable[..., Any]) -> bool:
        """Check, once, that render-then-encode reproduces the template's ids.

        ``render`` re-enters the patched chat entry point with ``tokenize``
        forced one way or the other. ``apply_chat_template(tokenize=True)``
        matching ``encode(apply_chat_template(tokenize=False))`` is a property
        of the template, not a guarantee, so the fast path is armed only after
        the first real request has confirmed it.

        Returns:
            True if the chat fast path is safe to use.
        """
        if self._chat_ok is None:
            try:
                text = render(tokenize=False)
                self._chat_ok = isinstance(text, str) and self.encode(text) == list(render(tokenize=True))
            except Exception:  # pragma: no cover - defensive
                logger.debug("Incremental tokenizer cache: chat probe raised", exc_info=True)
                self._chat_ok = False
            if self._chat_ok:
                logger.info("Incremental tokenizer cache: chat fast path armed.")
            else:
                logger.warning(
                    "Incremental tokenizer cache: chat fast path disabled - "
                    "apply_chat_template(tokenize=True) does not match "
                    "render-then-encode for this template."
                )
        return self._chat_ok


# ``safe_apply_chat_template`` and the DeepSeek renderers are reached without a
# renderer instance on hand, so the cache is looked up by the tokenizer it is
# handed. Entries are keyed weakly so a discarded renderer does not pin its
# tokenizer, and are written once per process during renderer construction,
# before any request is served.
_CACHES: weakref.WeakKeyDictionary[TokenizerLike, IncrementalTokenizerCache] = weakref.WeakKeyDictionary()


def _cache_for(tokenizer: TokenizerLike | None) -> IncrementalTokenizerCache | None:
    if tokenizer is None:
        return None
    return _CACHES.get(tokenizer)


def _chat_ids(
    cache: IncrementalTokenizerCache,
    render: Callable[..., Any],
    chat_kwargs: dict[str, Any],
) -> list[int] | None:
    """Token ids from the cache, or ``None`` to fall through to the original."""
    if chat_kwargs.get("tokenize", True) is False:
        return None
    if not cache.verify_chat_path(render):
        return None
    return cache.encode(render(tokenize=False))


def _patch_renderer_init() -> None:
    from vllm.renderers.base import BaseRenderer

    original = BaseRenderer.__init__

    def patched(self, config, tokenizer):
        original(self, config, tokenizer)
        capacity_gb = envs_ascend.VLLM_ASCEND_TOKENIZER_CACHE_GB
        if capacity_gb <= 0 or tokenizer is None:
            return
        cache = IncrementalTokenizerCache(tokenizer, capacity_gb)
        if cache.enabled:
            _CACHES[tokenizer] = cache

    BaseRenderer.__init__ = patched


def _patch_tokenize_prompt() -> None:
    """The plain ``/v1/completions`` path."""
    from vllm.renderers.base import BaseRenderer

    original = BaseRenderer._tokenize_prompt

    def patched(self, prompt, params):
        cache = _cache_for(self.tokenizer)
        if cache is None or self._wants_offsets(prompt, params):
            return original(self, prompt, params)

        kwargs = params.get_encode_kwargs()
        if not cache.is_eligible(add_special_tokens=kwargs.get("add_special_tokens", True)):
            return original(self, prompt, params)

        token_ids = cache.encode(prompt["prompt"])
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None and len(token_ids) > max_length:
            # Let the original path do the truncation.
            return original(self, prompt, params)
        return self._build_tokens_prompt(token_ids, prompt)

    BaseRenderer._tokenize_prompt = patched


def _patch_hf_chat() -> None:
    """The HF renderer chat path (sync and, via make_async, async)."""
    import vllm.renderers.hf as hf_mod

    original = hf_mod.safe_apply_chat_template

    def patched(model_config, tokenizer, conversation, **kwargs):
        cache = _cache_for(tokenizer)
        if cache is not None and not kwargs.get("return_assistant_tokens_mask"):

            def render(**kw):
                return original(model_config, tokenizer, conversation, **kw)

            token_ids = _chat_ids(cache, render, kwargs)
            if token_ids is not None:
                return token_ids
        return original(model_config, tokenizer, conversation, **kwargs)

    hf_mod.safe_apply_chat_template = patched


def _patch_renderer_chat(renderer_cls) -> None:
    original = renderer_cls._apply_chat_template

    def patched(self, *args, **kwargs):
        cache = _cache_for(self.tokenizer)
        if cache is not None:
            # ``kwargs`` carries the conversation (``messages``/``conversation``),
            # so the probe below has to keep it and only override ``tokenize``.
            def render(**kw):
                return original(self, *args, **{**kwargs, **kw})

            token_ids = _chat_ids(cache, render, kwargs)
            if token_ids is not None:
                return token_ids
        return original(self, *args, **kwargs)

    renderer_cls._apply_chat_template = patched


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
        _patch_renderer_chat(renderer_cls)


if envs_ascend.VLLM_ASCEND_TOKENIZER_CACHE_GB > 0:
    _patch_renderer_init()
    _patch_tokenize_prompt()
    _patch_hf_chat()
    _patch_deepseek_chat()
    logger.info(
        "Incremental tokenizer cache patch installed (%.2f GiB per API server process).",
        envs_ascend.VLLM_ASCEND_TOKENIZER_CACHE_GB,
    )
