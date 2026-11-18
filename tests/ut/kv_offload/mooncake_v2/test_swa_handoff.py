# SPDX-License-Identifier: Apache-2.0
"""Regression for SWA page retirement at the P/D handoff."""

from dataclasses import replace
from types import SimpleNamespace

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeBaseConnectorScheduler

from .helpers import make_request, make_sliding_spec


def test_swa_handoff_retains_history_for_last_prompt_token() -> None:
    block_size, window_size, prompt_len = 128, 512, 1023
    spec = replace(make_sliding_spec(block_size), sliding_window=window_size)
    assert spec.extra_retained_tokens == 0
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    scheduler.group_unique_specs = [[spec]]
    scheduler.need_truncate = scheduler._needs_prefill_token_truncation()
    tokens = list(range(prompt_len))
    producer = make_request(
        prompt_token_ids=tokens.copy(),
        _all_token_ids=tokens.copy(),
        num_prompt_tokens=prompt_len,
        kv_transfer_params={"do_remote_decode": True},
    )
    consumer = make_request(
        prompt_token_ids=tokens.copy(),
        num_prompt_tokens=prompt_len,
        kv_transfer_params={"do_remote_prefill": True},
    )
    scheduler.on_new_request(producer)
    scheduler.on_new_request(consumer)
    assert consumer.prompt_token_ids == tokens
    assert consumer.num_prompt_tokens == prompt_len
    assert scheduler.get_num_new_matched_tokens(consumer, 0) == (producer.num_prompt_tokens, True)

    manager = SlidingWindowManager(
        kv_cache_spec=spec,
        block_pool=BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=block_size),
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    computed = producer.num_prompt_tokens
    manager.allocate_new_blocks("p", computed, computed)
    blocks = manager.req_to_blocks["p"]
    payload: dict[int, dict[int, int]] = {}
    for position in range(computed):
        block = blocks[position // block_size]
        payload.setdefault(block.block_id, {})[position % block_size] = position
    manager.remove_skipped_blocks("p", computed)

    # Reuse every free page so stale contents cannot hide premature retirement.
    for block in manager.block_pool.get_new_blocks(manager.block_pool.get_num_free_blocks()):
        payload[block.block_id] = dict.fromkeys(range(block_size), -1)

    # D computes position 1022; its 512-token window needs history 511..1021.
    query = prompt_len - 1
    for position in range(query - window_size + 1, query):
        block = blocks[position // block_size]
        assert not block.is_null, f"Missing history at position {position}"
        assert payload[block.block_id][position % block_size] == position
