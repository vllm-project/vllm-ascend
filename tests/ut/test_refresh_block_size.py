# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.utils import refresh_block_size


def make_config(cache, *, hybrid=False, prefix=True, chunked=True):
    cache.enable_prefix_caching = prefix
    return SimpleNamespace(
        cache_config=cache,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3"), is_hybrid=hybrid),
    )


@pytest.mark.parametrize("prefix,chunked", [(True, False), (False, True), (True, True)])
def test_draft_refresh_preserves_explicit_target_block_size(prefix, chunked):
    cache = SimpleNamespace(block_size=768, user_specified_block_size=True)
    target = make_config(cache, hybrid=True, prefix=prefix, chunked=chunked)
    draft = make_config(cache, prefix=prefix, chunked=chunked)

    refresh_block_size(target)
    refresh_block_size(draft)

    assert draft.cache_config is target.cache_config
    assert target.cache_config.block_size == 768


@pytest.mark.parametrize("prefix,chunked", [(True, False), (False, True)])
def test_automatic_block_size_still_uses_prefix_or_chunked_default(prefix, chunked):
    cache = SimpleNamespace(block_size=768, user_specified_block_size=False)
    refresh_block_size(make_config(cache, prefix=prefix, chunked=chunked))
    assert cache.block_size == 128


def test_missing_cache_config_is_accepted():
    refresh_block_size(SimpleNamespace(cache_config=None, scheduler_config=None, model_config=None))
