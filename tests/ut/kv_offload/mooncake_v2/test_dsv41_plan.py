# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""V4.1 resource ownership and replay coverage, without a transfer engine."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.dsv41 import (
    DeepseekV41TransferPlan,
    is_deepseek_v41,
    validate_deepseek_v41_config,
)
from vllm_ascend.models.deepseek_v41.layerwise_cache_access import CacheResource, LayerCacheAccess


def make_plan(replay=True):
    swa = CacheResource("swa", 0, 0)
    shared = CacheResource("long_kv", 0, 1)
    borrowed = CacheResource("borrowed_swa", 1, 0)
    accesses = (
        LayerCacheAccess(0, (swa, shared), (swa, shared), 0, None),
        LayerCacheAccess(1, (borrowed, shared), (borrowed,), 0, None),
    )
    return DeepseekV41TransferPlan(accesses, {"swa": 4, "borrowed_swa": 4}, {"swa": 4, "borrowed_swa": 4}, replay)


def request(computed=0, scheduled=10, blocks=(0, 5, 6), prompt=10):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_scheduled_tokens=scheduled,
        num_prompt_tokens=prompt,
        local_block_ids=(blocks, (7, 8, 9)),
    )


def write_window(plan, slots):
    for name in plan.accesses:
        plan.record_write(name, torch.tensor(slots), cache_only=False)


def test_only_the_source_layer_publishes_shared_resources():
    plan = make_plan(replay=False)
    assert plan.record_write("swa", None, cache_only=False) == ("swa", "long_kv")
    assert plan.record_write("borrowed_swa", None, cache_only=False) == ("borrowed_swa",)
    assert plan.record_write("swa", None, cache_only=True) == ()
    assert plan.finish_step({"req": request()}) == {}


def test_cache_only_replay_does_not_publish_shared_resources():
    plan = make_plan()
    assert plan.record_write("swa", torch.tensor([23]), cache_only=True) == ()
    assert plan.record_write("swa", torch.tensor([24, 25]), cache_only=False) == ("long_kv",)
    plan.record_write("borrowed_swa", torch.tensor([23, 24, 25]), cache_only=False)
    assert plan.finish_step({"req": request()}) == {}


def test_replay_coverage_accumulates_across_chunks():
    plan = make_plan()
    write_window(plan, [23])
    assert plan.finish_step({"req": request(scheduled=8)}) == {}
    write_window(plan, [24, 25])
    assert plan.finish_step({"req": request(computed=8, scheduled=2)}) == {}
    # Finishing retires coverage, so a reused ID cannot inherit readiness.
    assert "req" in plan.finish_step({"req": request()})


def test_replay_snapshots_reused_slot_buffers():
    plan = make_plan()
    slots = torch.tensor([23])
    for name in plan.accesses:
        plan.record_write(name, slots, cache_only=True)
    slots.fill_(24)
    for name in plan.accesses:
        plan.record_write(name, slots, cache_only=True)
    slots.fill_(25)
    for name in plan.accesses:
        plan.record_write(name, slots, cache_only=False)
    slots.fill_(-1)
    assert plan.finish_step({"req": request()}) == {}


@pytest.mark.parametrize("slots", [[23, 24], [-1, 23, 24], []])
def test_missing_slots_fail_the_request(slots):
    plan = make_plan()
    write_window(plan, slots)
    assert "Incomplete" in plan.finish_step({"req": request()})["req"]


def test_each_layer_must_reconstruct_its_own_swa():
    plan = make_plan()
    plan.record_write("swa", torch.tensor([23, 24, 25]), cache_only=False)
    assert "borrowed_swa" in plan.finish_step({"req": request()})["req"]


@pytest.mark.parametrize("blocks", [(0, 5), (0, 5, 0)])
def test_missing_final_source_blocks_fail(blocks):
    plan = make_plan()
    write_window(plan, [23, 24, 25])
    assert "Missing" in plan.finish_step({"req": request(blocks=blocks)})["req"]


def test_discard_clears_partial_coverage():
    plan = make_plan()
    write_window(plan, [23])
    assert plan.finish_step({"req": request(scheduled=8)}) == {}
    plan.discard("req")
    write_window(plan, [24, 25])
    assert "req" in plan.finish_step({"req": request(computed=8, scheduled=2)})


def test_requests_do_not_share_coverage():
    plan = make_plan()
    write_window(plan, [23, 24, 25])
    errors = plan.finish_step({"ok": request(), "missing": request(blocks=(0, 8, 9))})
    assert set(errors) == {"missing"}


def test_short_prompt_and_empty_step():
    plan = make_plan()
    assert plan.finish_step({}) == {}
    write_window(plan, [4])
    assert plan.finish_step({"req": request(scheduled=1, prompt=1, blocks=(1,))}) == {}


@pytest.mark.parametrize("model_type", ["deepseek_v41", "deepseek_v41_text"])
def test_config_detection_and_limits(model_type):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        speculative_config=None,
    )
    assert is_deepseek_v41(config)
    validate_deepseek_v41_config(config)
    for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        setattr(config.parallel_config, field, 2)
        with pytest.raises(ValueError, match="PP=PCP=DCP=1"):
            validate_deepseek_v41_config(config)
        setattr(config.parallel_config, field, 1)
    config.speculative_config = object()
    with pytest.raises(ValueError, match="speculative"):
        validate_deepseek_v41_config(config)
    assert not is_deepseek_v41(SimpleNamespace())
