# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use real DSV4 layouts/managers and the existing in-memory pool transport."""

from math import ceil

import pytest

from tests.ut.distributed.ascend_store.test_dsv4_layerwise_runtime import reset_step
from tests.ut.distributed.ascend_store.test_dsv4_prefix_offload import PrefixPool
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("cache_format", ["a3", "a5_bf16", "a5_packed"])
def test_chunk_continuation_retains_exact_window_and_every_public_checkpoint(model, cache_format):
    pool = PrefixPool(model, cache_format=cache_format)
    previous = 0
    for end in (4096, 8192, 8193, 8197, 12288):
        request = pool.save(previous, end)
        masks = pool.worker._compute_reachable_load_masks(
            pool.meta(
                end,
                end + 1,
                load_spec=LoadSpec(
                    vllm_cached_tokens=0,
                    kvpool_cached_tokens=end,
                    can_load=True,
                    layerwise_previous_step=True,
                ),
            ),
            end,
        )
        public_masks = pool.coordinator.store_mask(request.save_end_token, request.num_prompt_tokens)
        for group, (spec, size) in enumerate(zip(pool.coordinator.group_effective_specs, pool.sizes)):
            window = getattr(spec, "sliding_window", None)
            start = 0 if window is None else max(0, end - window + 1 - spec.extra_retained_tokens) // size
            expected = set(range(start, ceil(end / size)))
            # Full-history masks are allocation masks; transfer ranges, not
            # the mask, exclude the newly allocated next-query page.
            relevant = masks[group] if window is not None else masks[group][: ceil(end / size)]
            assert {i for i, keep in enumerate(relevant) if keep} == expected
            for index, keep in enumerate(public_masks[group]):
                if keep and request.block_ids_by_group[group][index]:
                    assert pool.page_key(group, index) in pool.worker.m_store.committed
            if window is not None:
                assert len(expected) <= ceil((window - 1) / size) + 1
        # Exercise actual key selection, leases and transfer ranges, including
        # partial states at unaligned endpoints. Missing private data must fail.
        reset_step(pool.worker)
        continuation = pool.meta(
            end,
            end + 1,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=end,
                can_load=True,
                layerwise_previous_step=True,
            ),
        )
        assert pool.worker._prepare_load_gvas_for_request([continuation]) is not False
        for group in range(len(pool.groups)):
            pool.worker._process_load_for_layer_batch([continuation], 1, group)
            copied = pool.transfers(group, save=False)
            expected_ids = {
                continuation.block_ids_by_group[group][i]
                for i, keep in enumerate(masks[group][: ceil(end / pool.sizes[group])])
                if keep
            }
            assert set(copied) == expected_ids
            for gva in copied.values():
                for name in pool.groups[group].layer_names:
                    assert (gva, name) in pool.payloads
        previous = end


@pytest.mark.parametrize("model", ["flash", "pro"])
def test_window_reuse_preserves_non_reuse_prefix_hit_lengths(model):
    pools = [PrefixPool(model, buffers=buffers, cache_format="a5_bf16") for buffers in (3, 100)]
    for start in (0, 4096, 8192):
        for pool in pools:
            pool.save(start, start + 4096)
    # Exact-prompt replay has an existing separate checkpoint policy under
    # reuse, covered by test_dsv4_full_prefix_boundary.py. Compare reusable
    # prefixes here, with at least one new query token after each boundary.
    for length in (4095, 4097, 8193, 10000, 12289, 16383):
        assert pools[0].hit(length) == pools[1].hit(length)
