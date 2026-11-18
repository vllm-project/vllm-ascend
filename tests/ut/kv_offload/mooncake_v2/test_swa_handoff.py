# SPDX-License-Identifier: Apache-2.0
"""SWA handoff uses one position for P retirement and D continuation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeBaseConnectorScheduler

from .helpers import make_circular_spec, make_full_spec, make_request, make_sliding_spec

BLOCK_SIZE = 128
WINDOW_SIZE = 512
CACHE_LAYOUTS = ["swa", "full_swa", "uniform_swa", "step5"]


def make_swa_scheduler(cache_layout: str, num_speculative_tokens: int = 0) -> MooncakeBaseConnectorScheduler:
    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    main_name = "model.layers.1.self_attn"
    summary_name = main_name + ".sparse_summary_cache"
    merged_spec = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE,
        kv_cache_specs={main_name: make_full_spec(BLOCK_SIZE), summary_name: make_full_spec(BLOCK_SIZE)},
    )
    swa_spec = replace(make_sliding_spec(BLOCK_SIZE), sliding_window=WINDOW_SIZE)
    swa_group = KVCacheGroupSpec(layer_names=["model.layers.0.self_attn"], kv_cache_spec=swa_spec)
    scheduler.kv_cache_groups = [swa_group]
    if cache_layout == "uniform_swa":
        names = ["model.layers.0.self_attn", "model.layers.2.self_attn"]
        scheduler.kv_cache_groups = [
            KVCacheGroupSpec(
                layer_names=names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=BLOCK_SIZE, kv_cache_specs=dict.fromkeys(names, swa_spec)
                ),
            )
        ]
    elif cache_layout == "full_swa":
        scheduler.kv_cache_groups.append(
            KVCacheGroupSpec(layer_names=[main_name], kv_cache_spec=make_full_spec(BLOCK_SIZE))
        )
    elif cache_layout == "step5":
        scheduler.kv_cache_groups.extend(
            [
                KVCacheGroupSpec(layer_names=[main_name, summary_name], kv_cache_spec=merged_spec),
                KVCacheGroupSpec(layer_names=[main_name + ".csa_tail_ring"], kv_cache_spec=make_circular_spec()),
            ]
        )
    scheduler.group_unique_specs = [scheduler._get_group_unique_specs(group) for group in scheduler.kv_cache_groups]
    scheduler.group_block_size = [group.kv_cache_spec.block_size for group in scheduler.kv_cache_groups]
    scheduler.num_speculative_tokens = num_speculative_tokens
    scheduler.pcp_size = scheduler.dcp_size = 1
    scheduler.need_truncate = scheduler._needs_prefill_token_truncation()
    return scheduler


@pytest.mark.parametrize("aux_suffix", [".sparse_summary_cache", ".csa_tail_ring", ""])
def test_layer_names_without_swa_do_not_enable_truncation(aux_suffix: str) -> None:
    scheduler = make_swa_scheduler("swa")
    scheduler.kv_cache_groups = [
        KVCacheGroupSpec(layer_names=["other.layer" + aux_suffix], kv_cache_spec=make_circular_spec()),
        KVCacheGroupSpec(layer_names=["other.swa"], kv_cache_spec=make_full_spec()),
    ]
    scheduler.group_unique_specs = [scheduler._get_group_unique_specs(group) for group in scheduler.kv_cache_groups]
    assert scheduler._needs_prefill_token_truncation() is False


@pytest.mark.parametrize("prompt_len", [1, 2, 8, 9, 128, 129, 639, 640, 1023, 1024, 1151, 1152])
@pytest.mark.parametrize("local_hit", [0, 128])
@pytest.mark.parametrize("cache_layout", CACHE_LAYOUTS)
def test_swa_prompt_and_remote_hits_share_the_handoff(prompt_len: int, local_hit: int, cache_layout: str) -> None:
    scheduler = make_swa_scheduler(cache_layout)
    tokens = list(range(prompt_len))
    producer = make_request(
        prompt_token_ids=tokens.copy(),
        _all_token_ids=tokens.copy(),
        num_prompt_tokens=prompt_len,
        block_hashes=["original-full-prompt-hash"],
        kv_transfer_params={"do_remote_decode": True},
    )
    consumer = make_request(
        prompt_token_ids=tokens.copy(),
        _all_token_ids=tokens.copy(),
        num_prompt_tokens=prompt_len,
        kv_transfer_params={"do_remote_prefill": True},
    )
    scheduler.on_new_request(producer)
    scheduler.on_new_request(producer)
    scheduler.on_new_request(consumer)

    handoff = max(1, prompt_len - 1)
    assert producer.num_prompt_tokens == handoff
    assert producer.prompt_token_ids == producer._all_token_ids == tokens[:handoff]
    if prompt_len > 1:
        assert producer.max_tokens == 1
        assert producer.block_hashes == [tuple(tokens[:handoff])]
    assert consumer.prompt_token_ids == consumer._all_token_ids == tokens
    assert consumer.num_prompt_tokens == prompt_len
    matched, asynchronous = scheduler.get_num_new_matched_tokens(consumer, local_hit)
    assert matched == max(handoff - local_hit, 0)
    assert asynchronous is (matched > 0)


@pytest.mark.parametrize(
    "prompt_len", [2, 7, 8, 9, 127, 128, 129, 511, 512, 513, 638, 639, 640, 1023, 1024, 1025, 1151, 1152]
)
@pytest.mark.parametrize("num_speculative_tokens", [0, 3])
@pytest.mark.parametrize("cache_layout", CACHE_LAYOUTS)
def test_swa_handoff_preserves_payload_without_extra_retention(
    prompt_len: int, num_speculative_tokens: int, cache_layout: str
) -> None:
    """Use real managers; recycle discarded pages before reading D's window."""
    scheduler = make_swa_scheduler(cache_layout, num_speculative_tokens)
    producer = make_request(
        prompt_token_ids=list(range(prompt_len)),
        _all_token_ids=list(range(prompt_len)),
        num_prompt_tokens=prompt_len,
        kv_transfer_params={"do_remote_decode": True},
    )
    consumer = make_request(
        prompt_token_ids=list(range(prompt_len)),
        num_prompt_tokens=prompt_len,
        kv_transfer_params={"do_remote_prefill": True},
    )
    scheduler.on_new_request(producer)
    external_tokens, _ = scheduler.get_num_new_matched_tokens(consumer, 0)
    spec = scheduler.group_unique_specs[0][0]
    assert spec.extra_retained_tokens == 0

    def make_manager() -> SlidingWindowManager:
        return SlidingWindowManager(
            kv_cache_spec=spec,
            block_pool=BlockPool(num_gpu_blocks=32, enable_caching=False, hash_block_size=BLOCK_SIZE),
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=BLOCK_SIZE,
        )

    p_manager, d_manager = make_manager(), make_manager()
    payload: dict[int, dict[int, int]] = {}
    processed = 0
    while processed < producer.num_prompt_tokens:
        end = min(processed + 256, producer.num_prompt_tokens)
        p_manager.allocate_new_blocks("p", end, end)
        for position in range(processed, end):
            block = p_manager.req_to_blocks["p"][position // BLOCK_SIZE]
            payload.setdefault(block.block_id, {})[position % BLOCK_SIZE] = position
        processed = end
        p_manager.remove_skipped_blocks("p", processed)

    # Simulate another request overwriting every free physical page.
    free_blocks = p_manager.block_pool.get_new_blocks(p_manager.block_pool.get_num_free_blocks())
    for block in free_blocks:
        payload[block.block_id] = {offset: -1 for offset in range(BLOCK_SIZE)}
    p_blocks = p_manager.req_to_blocks["p"]
    full_ids = list(range(100, 100 + len(p_blocks)))
    block_ids: tuple[list[int], ...] = ([block.block_id for block in p_blocks],)
    if cache_layout in ("full_swa", "step5"):
        block_ids += (full_ids,)
    if cache_layout == "step5":
        block_ids += ([200],)
    transfer_ids = scheduler._get_transfer_block_ids(block_ids, producer.num_prompt_tokens)
    if cache_layout in ("full_swa", "step5"):
        assert transfer_ids[1] == full_ids
    if cache_layout == "step5":
        assert transfer_ids[2] == [200]

    d_manager.add_local_computed_blocks("d", [], 0, external_tokens)
    d_manager.allocate_external_computed_blocks("d", 0, external_tokens)
    d_payload = {}
    for source_id, destination in zip(transfer_ids[0], d_manager.req_to_blocks["d"]):
        if source_id != p_manager.block_pool.null_block.block_id and not destination.is_null:
            d_payload[destination.block_id] = payload[source_id].copy()

    # D computes the last original prompt token; its own KV is written locally.
    query = prompt_len - 1
    d_manager.allocate_new_blocks("d", prompt_len + num_speculative_tokens, prompt_len)
    current_block = d_manager.req_to_blocks["d"][query // BLOCK_SIZE]
    assert not current_block.is_null
    d_payload.setdefault(current_block.block_id, {})[query % BLOCK_SIZE] = query
    for position in range(max(0, query - WINDOW_SIZE + 1), query + 1):
        block = d_manager.req_to_blocks["d"][position // BLOCK_SIZE]
        assert not block.is_null, (prompt_len, position)
        assert d_payload[block.block_id][position % BLOCK_SIZE] == position
