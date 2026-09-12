# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the upstream vLLM change carried by
# patch/platform/patch_tokenizer_cache.py; see that file for the
# upstream contribution and removal plan.
"""Segment-level (incremental) tokenizer cache.

Prefix caching removes the *compute* for a repeated conversation prefix, but
tokenization is still `O(full prompt)` on every turn: the server has to know the
token ids before it can look anything up in the KV cache. In agent workloads,
where turn `N` re-sends turns `1..N-1` verbatim and appends a short suffix, that
makes the tokenizer the dominant per-request cost once the hit rate is high.

This module removes the redundant work by caching tokenization *per segment*
instead of per prompt. A whole-prompt cache is useless here — turn `N`'s prompt
is never byte-identical to turn `N-1`'s — but a segment cache hits on every
repeated turn.

Correctness rests on one identity: encoding `A + B` must give exactly the
concatenation of encoding `A` and encoding `B`. That holds when the cut falls
immediately after an *added token* (a special
token such as ``<|im_end|>``). HF tokenizers split the input on the added
vocabulary **before** the pre-tokenizer and BPE ever run, so a merge can never
span an added token. Chat templates emit one at every message boundary, which is
exactly where this module cuts.

That is a property of the *tokenizer*, not something that can be assumed: a
Metaspace/SentencePiece pre-tokenizer that prepends a space to each segment, or
a model with no added tokens at all, breaks it. So the identity is **verified
empirically at startup** against a probe corpus, and the cache disables itself
if the check does not pass. Anything not covered by the check falls through to
the plain tokenizer call, so an ineligible request is bit-identical to the
uncached path by construction.
"""

import contextlib
import re
import sys
import threading
from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.tokenizers.protocol import TokenizerLike
from vllm.utils.cache import CacheInfo, LRUCache

logger = init_logger(__name__)

GiB = 1 << 30

# Rough per-entry accounting. Token ids are large enough to fall outside
# CPython's small-int cache, so each one is a distinct object.
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
        # Whitespace either side of a boundary — the classic Metaspace trap.
        f"trailing spaces   {a}   leading spaces",
        f"no space{a}no space",
        # Newlines, which chat templates always put next to boundaries.
        f"line one\n{a}\nline two\n\n{b}\n",
        # Multi-byte text straddling a boundary.
        f"你好世界\U0001f30f{a}再来一次{b}ok",
        # Code and JSON, i.e. what an agent transcript actually contains.
        f"{code}{a}{code}{b}{code}",
        f'{{"name": "f", "arguments": {{"a": 1}}}}{a}result',
        # An added token appearing *inside* content rather than as a boundary.
        f"user said {a} literally{b}done",
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

        self._pattern: re.Pattern[str] | None = None
        self._enabled = False
        # Whether ``add_special_tokens=True`` is a no-op for this tokenizer.
        # When it is (Qwen-style templates that emit their own specials), the
        # cache can serve those requests too.
        self._special_tokens_are_noop = False
        # Armed separately: the chat fast path rests on a second identity.
        self._chat_path_ok = False
        # Set once the chat identity has been probed, pass or fail.
        self.chat_arm_attempted = False

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

    def stat(self, *, delta: bool = False) -> CacheInfo:
        """Segment-level hit/total counts."""
        with self._lock:
            return self._cache.stat(delta=delta)

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
            with self._lock, contextlib.suppress(ValueError):
                self._cache[segment] = encoded
            token_ids.extend(encoded)

        return token_ids

    @property
    def chat_path_enabled(self) -> bool:
        """Whether the render-then-encode chat fast path was verified."""
        return self._enabled and self._chat_path_ok

    def arm_chat_path(self, probes) -> bool:
        """Verify fused chat tokenization equals render-then-encode.

        `apply_chat_template(tokenize=True)` matching
        `encode(apply_chat_template(tokenize=False))` is a property of the
        template, not a guarantee, so it is checked rather than assumed.

        Args:
            probes: `(rendered_text, fused_token_ids)` pairs to verify.

        Returns:
            True if the chat fast path is safe to use.
        """
        if not self._enabled or not probes:
            return False
        for text, fused in probes:
            if self.encode(text) != list(fused):
                logger.warning(
                    "Incremental tokenizer cache: chat fast path disabled - "
                    "apply_chat_template(tokenize=True) does not match "
                    "render-then-encode for this template."
                )
                self._chat_path_ok = False
                return False
        self._chat_path_ok = True
        logger.info("Incremental tokenizer cache: chat fast path armed.")
        return True
