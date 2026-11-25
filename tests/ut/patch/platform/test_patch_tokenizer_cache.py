# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the segment-level incremental tokenizer cache."""

import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import regex as re

from vllm_ascend.patch.platform import patch_tokenizer_cache as tc

_END, _START = "<|im_end|>", "<|im_start|>"


class _FakeTokenizer:
    def __init__(self, added=(_END, _START), *, always_bos=False, add_special_bos=False, fail=False) -> None:
        self._added = {tok: 900000 + i for i, tok in enumerate(added)}
        pat = "|".join(re.escape(t) for t in sorted(added, key=len, reverse=True))
        self._pat = re.compile(pat) if pat else None
        self._always_bos, self._add_special_bos, self._fail = always_bos, add_special_bos, fail
        self.calls: list[str] = []

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        if self._fail:
            raise RuntimeError("tokenizer failure")
        self.calls.append(text)
        ids: list[int] = []
        cur = 0
        if self._pat is not None:
            for m in self._pat.finditer(text):
                ids.extend(ord(c) for c in text[cur : m.start()])
                ids.append(self._added[m.group()])
                cur = m.end()
        ids.extend(ord(c) for c in text[cur:])
        bos = self._always_bos or (self._add_special_bos and add_special_tokens)
        return {"input_ids": [1, *ids] if bos else ids}

    def ref(self, text: str) -> list[int]:
        return self(text, add_special_tokens=False)["input_ids"]


def _build(added=(_END, _START), capacity_gb=1, **kwargs):
    tok = _FakeTokenizer(added, **kwargs)
    return tok, tc.IncrementalTokenizerCache(tok, capacity_gb=capacity_gb)


def test_split_encode_reuse_eviction_and_concurrency():
    tok, cache = _build()
    assert cache.enabled and cache._split(f"a{_END}b{_START}tail") == [f"a{_END}", f"b{_START}", "tail"]
    assert _build(added=("<|a|>", "<|a|>x"))[1]._split("1<|a|>x2") == ["1<|a|>x", "2"]
    corpus = tc._probe_corpus(list(tok._added))
    assert tc._probe_corpus([]) == [] and len(corpus) == len(set(corpus)) > 0
    assert all(cache.encode(t) == tok.ref(t) for t in (f"你好{_END}world{_START}", "", *corpus))
    assert cache.encode(t1 := f"sys{_END}user: hi{_END}") == tok.ref(t1)
    ref2, n, h0 = tok.ref(t2 := f"{t1}assistant: hello{_END}"), len(tok.calls), cache.stat().hits
    assert cache.encode(t2) == ref2 and len(tok.calls) - n == 1 and cache.stat().hits - h0 == 2
    tok_l, lru = _build(capacity_gb=1500 / (1 << 30))
    for s in [f"e{i}{_END}" + ("long " * 80) for i in range(4)]:
        assert lru.encode(s) == tok_l.ref(s) and lru._cache.currsize <= lru._cache.capacity
    assert lru.encode(f"ok{_END}") == lru.encode(f"ok{_END}") and lru.stat().hits >= 1
    tok_t, tiny = _build(capacity_gb=1e-12)
    exp, n0 = tok_t.ref(ab := f"a{_END}b{_END}"), len(tok_t.calls)
    assert tiny.encode(ab) == exp and tiny.encode(ab) == exp and len(tok_t.calls) - n0 == 4
    texts = [f"turn {i}{_END}payload {i}{_START}" for i in range(4)]
    expected, barrier = [tok.ref(t) for t in texts], threading.Barrier(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        res = list(pool.map(lambda idx: (barrier.wait(), cache.encode(texts[idx % 4]))[1], range(16)))
    assert all(got == expected[i % 4] for i, got in enumerate(res))


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, False])
def test_disable_boundaries_and_invalid_budgets(bad):
    with pytest.raises(ValueError, match="finite number >= 0"):
        tc.IncrementalTokenizerCache(_FakeTokenizer(), bad)
    with pytest.raises(ValueError, match="finite number >= 0"):
        tc._renderer_capacity_gb(SimpleNamespace(additional_config={"tokenizer_cache_gb": bad}))
    for kw in ({"added": ()}, {"always_bos": True}, {"fail": True}):
        _, c = _build(**kw)
        assert not c.enabled and not c.is_eligible(add_special_tokens=False) and c.disabled_reason is not None
    _, bos = _build(add_special_bos=True)
    assert bos.is_eligible(add_special_tokens=False) and not bos.is_eligible(add_special_tokens=True)


def test_install_chat_ids_and_patch_wiring(monkeypatch):
    import vllm.renderers.base as base_mod

    tok, cache = _build()
    assert tc._cache_for(None) is None and tc._cache_for(tok) is None
    tc._install_cache(tok, cache)
    tc._install_cache(slotted := type("_Slotted", (), {"__slots__": ()})(), cache)
    assert tc._cache_for(tok) == cache and tc._cache_for(slotted) is None
    text, calls = f"system{_END}user: hi{_END}", []
    exp = tok.ref(text)

    def render(*, tokenize=True):
        calls.append(tokenize)
        return exp if tokenize else text

    assert tc._chat_ids(cache, render, {}) == exp and calls == [False, True]
    assert cache.verify_chat_path(exp, lambda **_: [123]) and not cache.chat_disabled
    for kw in ({"tokenize": False}, {"padding": True}, {"tokenizer_kwargs": {"return_offsets_mapping": True}}):
        assert tc._chat_ids(cache, render, kw) is None
    assert tc._chat_ids(cache, lambda **_: (text, [0]), {}) is None
    setattr(tok, tc._CACHE_ATTRIBUTE, object())
    assert tc._cache_for(tok) is None
    _, bad_c = _build()
    bad_fn = lambda **kw: [123] if kw.get("tokenize", True) else text
    assert tc._chat_ids(bad_c, bad_fn, {}) is None and bad_c.chat_disabled
    CopyR = type("_CopyR", (), {"__init__": lambda self, _c, t: setattr(self, "tokenizer", copy.copy(t))})
    cache_mod = sys.modules[tc._patch_renderer_init.__module__]
    monkeypatch.setattr(base_mod, "BaseRenderer", CopyR)
    for is_v0340, cap in ((True, 1), (False, 1), (False, 0)):
        monkeypatch.setattr(cache_mod, "vllm_version_is", lambda ver, f=is_v0340: f and ver == "0.34.0")
        monkeypatch.setattr(cache_mod, "_configured_capacity_gb", lambda c=cap: c)
        monkeypatch.setattr(cache_mod, "_legacy_tokenizer_supported", lambda _t: True)
        tc._patch_renderer_init()
        rend = CopyR(None, passed := _FakeTokenizer())
        assert rend.tokenizer is not passed and tc._cache_for(passed) is None
        assert (tc._cache_for(rend.tokenizer) is not None) if cap else (tc._cache_for(rend.tokenizer) is None)
