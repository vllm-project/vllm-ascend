# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full prompt hits must restore a published checkpoint before replay."""

import pytest

from tests.ut.distributed.ascend_store.test_dsv4_prefix_offload import PrefixPool


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("local_tokens", [0, 12288])
@pytest.mark.parametrize("probe_local", [False, True])
def test_full_prompt_replay_restores_every_group_from_a_valid_checkpoint(model, local_tokens, probe_local):
    pool = PrefixPool(model)
    for start in range(0, 16384, 4096):
        pool.save(start, start + 4096)
    request = pool.request(16384)
    if probe_local:
        assert pool.scheduler.get_local_prefix_cache_limit(request) == 12288
    count, asynchronous = pool.scheduler.get_num_new_matched_tokens(request, local_tokens)
    assert (count, asynchronous) == (12288 - local_tokens, False)
    assert pool.scheduler.load_specs["b"].kvpool_cached_tokens == 12288
    # Exercise actual load ranges and tagged payloads with the original prompt.
    pool.restore(12288, prompt_tokens=16384)


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("missing_group", range(6))
def test_full_prompt_replay_reprobes_all_groups_at_the_earlier_checkpoint(model, missing_group):
    pool = PrefixPool(model)
    for start in range(0, 16384, 4096):
        pool.save(start, start + 4096)
    pool.worker.m_store.committed.remove(pool.page_key(missing_group, 12288 // pool.sizes[missing_group] - 1))
    assert pool.scheduler.get_local_prefix_cache_limit(pool.request(16384)) == 8192
    assert pool.scheduler.get_num_new_matched_tokens(pool.request(16384), 0) == (8192, False)
    pool.restore(8192, prompt_tokens=16384)


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("tokens,expected", [(4095, 0), (4096, 0), (4097, 4096)])
def test_first_checkpoint_replay_boundary(model, tokens, expected):
    pool = PrefixPool(model)
    pool.save(0, 4096)
    assert pool.scheduler.get_num_new_matched_tokens(pool.request(tokens), 0) == (expected, False)
