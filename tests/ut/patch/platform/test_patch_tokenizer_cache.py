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
"""Unit tests for the segment-level incremental tokenizer cache.

The cache is driven through a character-level fake tokenizer so the tests run
on CPU without loading a real model: the fake reproduces the only property the
cache relies on, namely that an added token is turned into a dedicated id and
therefore never participates in a BPE merge across the cut.
"""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import regex as re

from vllm_ascend.patch.platform.patch_tokenizer_cache import (
    _CACHE_ATTRIBUTE,
    IncrementalTokenizerCache,
    _cache_for,
    _chat_ids,
    _install_cache,
    _patch_hf_chat,
    _patch_renderer_chat,
    _patch_renderer_init,
    _probe_corpus,
)

_END = "<|im_end|>"
_START = "<|im_start|>"
_BOS_ID = 1


class _FakeTokenizer:
    """Minimal stand-in for a HF fast tokenizer with added tokens."""

    def __init__(
        self,
        added_tokens=(_END, _START),
        *,
        always_bos=False,
        add_special_bos=False,
        fail=False,
    ) -> None:
        self._added = {token: 900000 + i for i, token in enumerate(added_tokens)}
        self._pattern = (
            re.compile("|".join(re.escape(t) for t in sorted(added_tokens, key=len, reverse=True)))
            if added_tokens
            else None
        )
        self._always_bos = always_bos
        self._add_special_bos = add_special_bos
        self._fail = fail
        self.calls: list[str] = []

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._added)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        if self._fail:
            raise RuntimeError("tokenizer failure")

        self.calls.append(text)
        ids: list[int] = []
        cursor = 0
        if self._pattern is not None:
            for match in self._pattern.finditer(text):
                ids.extend(ord(ch) for ch in text[cursor : match.start()])
                ids.append(self._added[match.group()])
                cursor = match.end()
        ids.extend(ord(ch) for ch in text[cursor:])

        if self._always_bos or (self._add_special_bos and add_special_tokens):
            ids = [_BOS_ID, *ids]
        return {"input_ids": ids}

    def reference(self, text: str) -> list[int]:
        """Token ids the uncached path would produce."""
        return self(text, add_special_tokens=False)["input_ids"]


def _build(added_tokens=(_END, _START), capacity_gb=1, **kwargs):
    tokenizer = _FakeTokenizer(added_tokens, **kwargs)
    return tokenizer, IncrementalTokenizerCache(tokenizer, capacity_gb=capacity_gb)


def _chat_render(tokenizer, text):
    """Stand-in for a patched chat entry point: text out, ids in."""

    def render(**kwargs):
        return tokenizer.reference(text) if kwargs.get("tokenize", True) else text

    return render


def test_enabled_when_concatenation_identity_holds():
    _, cache = _build()
    assert cache.enabled
    assert cache.is_eligible(add_special_tokens=False)


@pytest.mark.parametrize(
    "text,expected",
    [
        (f"a{_END}b{_START}", [f"a{_END}", f"b{_START}"]),
        (f"a{_END}tail", [f"a{_END}", "tail"]),
        (f"{_END}{_START}", [_END, _START]),
        ("no boundary at all", ["no boundary at all"]),
    ],
)
def test_split_cuts_immediately_after_every_added_token(text, expected):
    _, cache = _build()
    assert cache._split(text) == expected


def test_longest_added_token_wins():
    _, cache = _build(added_tokens=("<|a|>", "<|a|>x"))
    assert cache._split("1<|a|>x2") == ["1<|a|>x", "2"]


@pytest.mark.parametrize(
    "text",
    [
        f"system{_END}user: hello{_END}",
        f"你好好{_END}world{_START}tail",
        f"line one\n{_END}\nline two\n\n{_START}\n",
        f'{{"a": 1}}{_END}result',
        "",
    ],
)
def test_encode_is_bit_identical_to_plain_tokenization(text):
    tokenizer, cache = _build()
    assert cache.encode(text) == tokenizer.reference(text)


def test_probe_corpus_round_trips_through_the_cache():
    tokenizer, cache = _build()
    corpus = _probe_corpus(tokenizer.all_special_tokens)
    # Every probe is distinct, so the corpus cannot silently degenerate into
    # repetitions of one shape.
    assert len(corpus) == len(set(corpus)) > 0
    for text in corpus:
        assert cache.encode(text) == tokenizer.reference(text)


def test_repeated_turns_only_tokenize_the_new_segment():
    tokenizer, cache = _build()
    turn1 = f"system prompt{_END}user: hi{_END}"
    turn2 = turn1 + f"assistant: hello{_END}"

    reference1 = tokenizer.reference(turn1)
    assert cache.encode(turn1) == reference1
    reference2 = tokenizer.reference(turn2)

    calls_after_turn1 = len(tokenizer.calls)
    assert cache.encode(turn2) == reference2

    # Only "assistant: hello<|im_end|>" is new; the two shared segments hit.
    assert len(tokenizer.calls) - calls_after_turn1 == 1


def test_stat_counts_segment_hits():
    tokenizer, cache = _build()
    text = f"a{_END}b{_END}"

    before = len(tokenizer.calls)
    cache.encode(text)
    # Two segments, both cold, so the miss half of the ratio has to show up.
    assert (cache.stat().hits, cache.stat().total) == (0, 2)
    cache.encode(text)

    assert cache.stat().hits == 2
    assert cache.stat().hit_ratio == 0.5
    assert len(tokenizer.calls) - before == 2


def test_disabled_when_tokenizer_has_no_added_tokens():
    _, cache = _build(added_tokens=())
    assert not cache.enabled
    assert not cache.is_eligible(add_special_tokens=False)
    assert cache.disabled_reason is not None


def test_disabled_when_concatenation_identity_breaks():
    # A tokenizer that unconditionally prepends BOS breaks the identity: the
    # spliced result gets one BOS per segment.
    tokenizer, cache = _build(always_bos=True)
    assert not cache.enabled
    assert not cache.is_eligible(add_special_tokens=False)
    assert cache.disabled_reason is not None
    assert tokenizer.reference(f"a{_END}b") != cache._split(f"a{_END}b")


def test_disabled_when_self_check_raises():
    _, cache = _build(fail=True)
    assert not cache.enabled
    assert cache.disabled_reason is not None


def test_add_special_tokens_is_served_only_when_it_is_a_noop():
    _, noop_cache = _build()
    assert noop_cache.is_eligible(add_special_tokens=True)

    _, bos_cache = _build(add_special_bos=True)
    assert bos_cache.enabled
    assert bos_cache.is_eligible(add_special_tokens=False)
    assert not bos_cache.is_eligible(add_special_tokens=True)


def test_chat_path_is_armed_only_by_a_matching_probe():
    tokenizer, cache = _build()
    text = f"system{_END}user: hi{_END}"

    assert cache.verify_chat_path(_chat_render(tokenizer, text))
    # The verdict is cached: a later mismatching probe cannot revoke it.
    assert cache.verify_chat_path(lambda **_: [123, 456])

    _, bad_cache = _build()

    def mismatched(**kwargs):
        return [123, 456] if kwargs.get("tokenize", True) else text

    assert not bad_cache.verify_chat_path(mismatched)
    assert not bad_cache.verify_chat_path(_chat_render(tokenizer, text))


def test_chat_path_is_not_armed_when_rendering_returns_no_text():
    _, cache = _build()
    assert not cache.verify_chat_path(lambda **_: [123, 456])


def test_tiny_capacity_still_returns_correct_ids():
    tokenizer, cache = _build(capacity_gb=1e-12)
    text = f"a{_END}b{_END}"
    expected = tokenizer.reference(text)

    assert cache.encode(text) == expected
    calls = len(tokenizer.calls)
    assert cache.encode(text) == expected
    # Nothing fits in the cache, so every segment is recomputed - and every
    # insert is rejected without raising.
    assert len(tokenizer.calls) - calls == 2


def test_concurrent_encodes_are_consistent():
    tokenizer, cache = _build()
    texts = [f"turn {i}{_END}payload {i}{_START}" for i in range(16)]
    expected = [tokenizer.reference(text) for text in texts]
    barrier = threading.Barrier(8)

    def encode(idx: int) -> list[int]:
        barrier.wait()
        return cache.encode(texts[idx % len(texts)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(encode, range(64)))

    for idx, result in enumerate(results):
        assert result == expected[idx % len(texts)]


def test_probe_corpus_is_deterministic_and_empty_without_specials():
    assert _probe_corpus([]) == []
    assert _probe_corpus([_START, _END]) == _probe_corpus([_END, _START])


def test_cache_for_returns_none_for_missing_or_unknown_tokenizers():
    assert _cache_for(None) is None
    assert _cache_for(_FakeTokenizer()) is None


def test_install_cache_is_reachable_through_the_tokenizer():
    """The cache is owned by the tokenizer; no module-level registry is used."""
    tokenizer = _FakeTokenizer()
    assert _cache_for(tokenizer) is None

    _, cache = _build()
    _install_cache(tokenizer, cache)
    assert _cache_for(tokenizer) is cache


def test_cache_for_ignores_foreign_attributes():
    tokenizer = _FakeTokenizer()
    setattr(tokenizer, _CACHE_ATTRIBUTE, object())
    assert _cache_for(tokenizer) is None


class _CopyingRenderer:
    """Mirrors ``HfRenderer``: ``BaseRenderer`` is handed a copy to keep."""

    def __init__(self, config, tokenizer) -> None:
        self.tokenizer = copy.copy(tokenizer)


def test_renderer_init_patch_attaches_the_cache_to_the_renderer_tokenizer(monkeypatch):
    """``_patch_renderer_init`` must install on the tokenizer the renderer keeps.

    ``HfRenderer`` hands ``BaseRenderer`` a copy of the tokenizer and looks the
    cache up through ``self.tokenizer``; installing it on the constructor
    argument instead would leave every lookup - and so the whole patch - dead
    with no error anywhere.
    """
    import vllm.renderers.base as base_mod

    original_init = _CopyingRenderer.__init__
    monkeypatch.setattr(base_mod, "BaseRenderer", _CopyingRenderer)
    _patch_renderer_init(capacity_gb=1)
    try:
        passed_in = _FakeTokenizer()
        renderer = _CopyingRenderer(None, passed_in)

        assert renderer.tokenizer is not passed_in
        assert _cache_for(renderer.tokenizer) is not None
        assert _cache_for(passed_in) is None
    finally:
        _CopyingRenderer.__init__ = original_init


def test_chat_ids_returns_none_for_requests_that_opt_out_of_tokenizing():
    tokenizer, cache = _build()
    assert _chat_ids(cache, _chat_render(tokenizer, f"a{_END}"), {"tokenize": False}) is None


def test_chat_ids_uses_the_cache_when_the_identity_holds():
    tokenizer, cache = _build()
    text = f"system{_END}user: hi{_END}"
    expected = tokenizer.reference(text)

    assert _chat_ids(cache, _chat_render(tokenizer, text), {}) == expected
    calls = len(tokenizer.calls)
    assert _chat_ids(cache, _chat_render(tokenizer, text), {}) == expected
    # Both segments of `text` are cached by now, so the second call is free.
    assert len(tokenizer.calls) == calls


def test_chat_ids_falls_through_when_the_template_identity_does_not_hold():
    tokenizer, cache = _build()
    text = f"system{_END}user: hi{_END}"

    def mismatched(**kwargs):
        return [123, 456] if kwargs.get("tokenize", True) else text

    assert _chat_ids(cache, mismatched, {}) is None


class _KeywordOnlyRenderer:
    """Mirrors ``DeepseekV4Renderer``: the conversation arrives as a keyword."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def _apply_chat_template(self, messages, *, tokenize=True, chat_template=None):
        text = f"{messages['content']}{_END}reply"
        if not tokenize:
            return text
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]


def test_renderer_chat_patch_forwards_the_callers_keyword_arguments():
    """The probe must keep the caller's kwargs; DeepSeek passes `messages` there.

    Dropping them made the probe raise, which silently kept the chat fast path
    switched off for every DeepSeek model even though its identity holds.
    """
    tokenizer, cache = _build()
    original = _KeywordOnlyRenderer._apply_chat_template
    _install_cache(tokenizer, cache)
    try:
        _patch_renderer_chat(_KeywordOnlyRenderer)
        renderer = _KeywordOnlyRenderer(tokenizer)

        first = renderer._apply_chat_template(messages={"content": "hi"}, chat_template=None)
        assert first == tokenizer.reference(f"hi{_END}reply")

        tokenized = len(tokenizer.calls)
        second = renderer._apply_chat_template(messages={"content": "hi"}, chat_template=None)
        assert second == first
        assert len(tokenizer.calls) == tokenized, "the repeat call must not re-tokenize"
    finally:
        _KeywordOnlyRenderer._apply_chat_template = original


def test_hf_chat_patch_forwards_the_callers_template_kwargs():
    """The HF renderer takes the conversation positionally, so this one is silent.

    OpenAI ``tools`` reach the renderer inside ``chat_template_kwargs``
    (``ChatParams.get_apply_chat_template_kwargs``), which is forwarded as
    keyword arguments. Dropping them from the probe - and from the render whose
    text is then cached - raises nothing: it quietly returns the token ids of a
    different prompt, with the tool definitions missing.
    """
    import vllm.renderers.hf as hf_mod

    tokenizer, cache = _build()
    seen: list = []

    def fake_safe_apply_chat_template(model_config, tok, conversation, *, tokenize=True, tools=None, **_ignored):
        seen.append(tools)
        text = f"{'tools' if tools else 'no tools'}{_END}user: hi{_END}"
        return text if not tokenize else tok(text, add_special_tokens=False)["input_ids"]

    original = hf_mod.safe_apply_chat_template
    _install_cache(tokenizer, cache)
    try:
        hf_mod.safe_apply_chat_template = fake_safe_apply_chat_template
        _patch_hf_chat()

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        got = hf_mod.safe_apply_chat_template(None, tokenizer, [], tools=tools)

        assert got == tokenizer.reference(f"tools{_END}user: hi{_END}")
        assert seen and all(render_tools is tools for render_tools in seen), (
            "every render on the cache path must keep the caller's kwargs"
        )
    finally:
        hf_mod.safe_apply_chat_template = original
