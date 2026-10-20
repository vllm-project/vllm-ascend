# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.quantization.methods.kv_cache.turboquant.final_prefill import (
    BF16_SLOT_BYTES,
    TQ_SLOT_BYTES,
    ColocatedFinalPrefill,
    FinalPrefillPlan,
    colocated_cache_views,
    plan_final_prefill,
    plan_incremental_slots,
    validate_colocated_final_prefill,
)


def request(computed, prompt=30000, generated=0, blocks=None):
    return SimpleNamespace(num_computed_tokens=computed,
                           num_prompt_tokens=prompt,
                           num_tokens=prompt + generated,
                           block_ids=(blocks or list(range(235)), ))


def config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=torch.bfloat16,
                                     enforce_eager=False,
                                     hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1,
                                        decode_context_parallel_size=1,
                                        pipeline_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        kv_transfer_config=None,
        speculative_config=None,
        use_v2_model_runner=False,
    )


def test_supported_graph_config():
    validate_colocated_final_prefill(config())


@pytest.mark.parametrize("field,value", [
    ("cache_config.cache_dtype", "turboquant_4bit_nc"),
    ("model_config.dtype", torch.float16),
    ("parallel_config.decode_context_parallel_size", 2),
    ("parallel_config.prefill_context_parallel_size", 2),
    ("parallel_config.pipeline_parallel_size", 2),
    ("compilation_config.cudagraph_mode", CUDAGraphMode.FULL),
    ("model_config.hf_text_config.kv_lora_rank", 256),
    ("use_v2_model_runner", True),
    ("speculative_config", object()),
    ("kv_transfer_config", object()),
])
def test_unsupported_configs_fail_before_cache_allocation(field, value):
    cfg = config()
    target = cfg
    *parents, name = field.split(".")
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, name, value)
    with pytest.raises(ValueError):
        validate_colocated_final_prefill(cfg)


def test_only_final_chunk_exports_complete_history():
    reqs = {"a": request(0)}
    plan = plan_final_prefill(reqs, {"a": 4096}, 0, 128)
    assert plan.has_prefill and not plan.block_ids
    reqs["a"] = request(28672)
    assert plan_final_prefill(reqs, {"a": 1328}, 0, 128).block_ids == tuple(range(235))


def test_full_prefix_hit_is_still_prefill():
    plan = plan_final_prefill({"a": request(29999)}, {"a": 1}, 0, 128)
    assert plan.has_prefill and len(plan.block_ids) == 235


def test_decode_and_resumed_recompute():
    plan = plan_final_prefill({"a": request(30000, generated=1)}, {"a": 1}, 0, 128)
    assert plan == FinalPrefillPlan()
    resumed = request(0, prompt=128, generated=10, blocks=[4, 2])
    plan = plan_final_prefill({"a": resumed}, {"a": 138}, 0, 128)
    assert plan == FinalPrefillPlan(True, (4, 2))


def test_shared_prefix_pages_are_deduplicated():
    reqs = {"a": request(127, 128, blocks=[4]), "b": request(127, 128, blocks=[4])}
    assert plan_final_prefill(reqs, {"a": 1, "b": 1}, 0, 128).block_ids == (4, )


def test_resumed_last_chunk_after_prompt_is_not_mistaken_for_decode():
    reqs = {"a": request(137, prompt=128, generated=10, blocks=[7, 3])}
    assert plan_final_prefill(reqs, {"a": 1}, 0, 128, {"a"}) == FinalPrefillPlan(True, (7, 3))
    assert plan_final_prefill(reqs, {"a": 1}, 0, 128, set()) == FinalPrefillPlan()


def test_both_cache_pools_are_charged_and_nonoverlapping():
    spec = AscendMLAAttentionSpec(block_size=128,
                                  num_kv_heads=1,
                                  head_size=576,
                                  dtype=torch.bfloat16,
                                  cache_final_prefill_turboquant=True)
    assert spec.page_size_bytes == 128 * (BF16_SLOT_BYTES + TQ_SLOT_BYTES)
    raw = torch.zeros(spec.page_size_bytes * 3, dtype=torch.int8)
    nope, rope, tq = colocated_cache_views(raw, 3, 128)
    nope.fill_(1)
    rope.fill_(2)
    tq.fill_(3)
    assert torch.all(nope == 1) and torch.all(rope == 2) and torch.all(tq == 3)


def test_update_changes_only_target_mirror_rows_preserving_bf16():
    raw = torch.zeros(3 * 128 * (BF16_SLOT_BYTES + TQ_SLOT_BYTES), dtype=torch.int8)
    cache = colocated_cache_views(raw, 3, 128)
    cache[0].fill_(1)
    cache[1].fill_(2)
    before = [t.clone() for t in cache[:2]]
    quantizer = MagicMock()

    def store(nope, rope, slots, output):
        assert torch.all(nope == 1) and torch.all(rope == 2)
        output[slots] = 7

    quantizer.compress_kv_into.side_effect = store
    state = ColocatedFinalPrefill(quantizer)
    state.bind(cache)
    state.update_decode_tokens(torch.tensor([256, 0]))
    assert torch.all(cache[2].view(-1, TQ_SLOT_BYTES)[[0, 256]] == 7)
    assert not cache[2][1].count_nonzero()
    for old, now in zip(before, cache[:2]):
        torch.testing.assert_close(old, now, rtol=0, atol=0)


def test_native_page_copy_preserves_all_three_cache_components():
    from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace

    raw = torch.zeros(3 * 128 * (BF16_SLOT_BYTES + TQ_SLOT_BYTES), dtype=torch.int8)
    cache = colocated_cache_views(raw, 3, 128)
    for index, tensor in enumerate(cache):
        tensor[0].fill_(index + 1)
    source = [tensor[0].clone() for tensor in cache]
    copy_kv_cache_blocks_inplace([cache], 3, [SimpleNamespace(src_block_id=0, dst_block_id=2)])
    for expected, tensor in zip(source, cache):
        torch.testing.assert_close(tensor[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(tensor[2], expected, rtol=0, atol=0)
        assert not tensor[1].count_nonzero()


def test_incremental_slots_cover_prefix_once_then_only_new_tokens():
    reqs = {"a": request(3, prompt=9, blocks=[7, 2, 5])}
    assert plan_incremental_slots(reqs, {"a": 3}, 0, 4, {"a"}, {"a"}) == (28, 29, 30, 31, 8, 9)
    reqs["a"].num_computed_tokens = 6
    assert plan_incremental_slots(reqs, {"a": 3}, 0, 4, {"a"}, set()) == (10, 11, 20)
    assert plan_incremental_slots(reqs, {"a": 3}, 0, 4, set(), set()) == ()


def test_incremental_slots_resume_and_shared_prefix():
    reqs = {"a": request(3, prompt=5, blocks=[7, 2]), "b": request(3, prompt=5, blocks=[7, 3])}
    assert plan_incremental_slots(reqs, {"a": 2, "b": 2}, 0, 4, set(reqs), set(reqs)) == (28, 29, 30, 31, 8, 12)
    reqs["a"] = request(0, prompt=5, generated=1, blocks=[4, 6])
    assert plan_incremental_slots(reqs, {"a": 6}, 0, 4, {"a"}, {"a"}) == (16, 17, 18, 19, 24, 25)
