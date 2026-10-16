# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackendImpl,
    mxfp_packet_size_bytes,
    mxfp_paged_cache_views,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _runner(consumer=True, transfer=True, group_id=0):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    groups = [SimpleNamespace(layer_names=["mamba"]), SimpleNamespace(layer_names=["mamba.other"])]
    groups[group_id] = SimpleNamespace(layer_names=["attention"])
    runner.kv_cache_config = SimpleNamespace(num_blocks=4, kv_cache_groups=groups)
    scale = torch.arange(256, dtype=torch.int32).remainder(251).to(torch.uint8)
    layer = SimpleNamespace(
        impl=object.__new__(AscendC8MXFPAttentionBackendImpl),
        v_cache_scale=scale,
    )
    runner.compilation_config = SimpleNamespace(static_forward_context={"attention": layer})
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=consumer) if transfer else None
    )
    runner.input_batch = SimpleNamespace(req_ids=["request"])

    # Match the failing 3-to-5 kernel-block transfer, with padded packet strides
    # and guards so any accidental write into KV payloads or another page fails.
    ratio, offset, guard = 5, 137, 97
    pitch = mxfp_packet_size_bytes(1, 512, 256, 256) + 256
    raw = torch.full((offset + 4 * ratio * pitch + guard,), 69, dtype=torch.int8)
    cache = mxfp_paged_cache_views(raw[offset:-guard], 4 * ratio, 1, 256, 256, 512, page_stride_bytes=pitch)
    runner._fill_c8_mxfp_v_scale_caches({"attention": cache})
    loaded = raw.clone()
    raw[offset + ratio * pitch : offset + 2 * ratio * pitch].fill_(-1)
    raw[offset + ratio * pitch : offset + (ratio + 3) * pitch].copy_(
        loaded[offset + ratio * pitch : offset + (ratio + 3) * pitch]
    )
    return runner, raw, cache, scale, pitch, offset, guard


def _event(kind="new", group_id=0, pages=(1,), req_id="request", zeroed=()):
    block_ids = ([2], [2])
    block_ids[group_id][:] = pages
    cached = SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set())
    output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        new_block_ids_to_zero=list(zeroed),
        block_table_updates=None,
        kv_cache_block_copies=[],
    )
    if kind == "new":
        output.scheduled_new_reqs = [SimpleNamespace(req_id=req_id, block_ids=block_ids)]
    elif kind in ("cached", "resumed"):
        cached.req_ids = [req_id]
        cached.new_block_ids = [block_ids]
        cached.resumed_req_ids = {req_id} if kind == "resumed" else set()
    elif kind == "replacement":
        output.block_table_updates = {req_id: block_ids}
    return output


@pytest.mark.parametrize("kind", ["new", "cached", "resumed", "replacement"])
@pytest.mark.parametrize("group_id", [0, 1])
def test_partial_async_load_restores_tail_scales_without_changing_other_bytes(kind, group_id):
    runner, raw, cache, scale, pitch, offset, guard = _runner(group_id=group_id)
    expected = raw.clone()
    reference = mxfp_paged_cache_views(expected[offset:-guard], 20, 1, 256, 256, 512, page_stride_bytes=pitch)
    reference[3][5:10].copy_(scale.view(1, 16, 1, 16, 1))
    assert torch.all(cache[3][8:10] == 255)

    runner._queue_c8_mxfp_v_scale_pages(_event(kind, group_id=group_id))
    runner._restore_pending_c8_mxfp_v_scales()

    assert torch.equal(raw, expected)
    assert not runner._c8_mxfp_pending_v_scale_pages


@pytest.mark.parametrize("kind", ["new", "resumed", "replacement"])
def test_replaced_assignments_do_not_restore_pages_now_owned_by_another_group(kind):
    runner, raw, _, scale, pitch, offset, guard = _runner()
    runner._c8_mxfp_pending_v_scale_pages = {"request": {0: {2}}}
    runner._queue_c8_mxfp_v_scale_pages(_event(kind))
    assert runner._c8_mxfp_pending_v_scale_pages == {"request": {0: {1}}}
    expected = raw.clone()
    reference = mxfp_paged_cache_views(expected[offset:-guard], 20, 1, 256, 256, 512, page_stride_bytes=pitch)
    reference[3][5:10].copy_(scale.view(1, 16, 1, 16, 1))
    runner._restore_pending_c8_mxfp_v_scales()
    assert torch.equal(raw, expected)


def test_waiting_request_is_not_written_until_scheduled_for_forward():
    runner, raw, _, _, _, _, _ = _runner()
    before = raw.clone()
    runner._queue_c8_mxfp_v_scale_pages(_event())
    runner.input_batch.req_ids = []
    runner._restore_pending_c8_mxfp_v_scales()
    assert torch.equal(raw, before)
    assert runner._c8_mxfp_pending_v_scale_pages == {"request": {0: {1}}}
    runner.input_batch.req_ids = ["request"]
    runner._restore_pending_c8_mxfp_v_scales()
    assert not torch.equal(raw, before)
    assert not runner._c8_mxfp_pending_v_scale_pages


def test_finished_request_discards_pending_pages_without_writes():
    runner, raw, _, _, _, _, _ = _runner()
    before = raw.clone()
    runner._queue_c8_mxfp_v_scale_pages(_event())
    finished = _event("empty")
    finished.finished_req_ids = {"request"}
    runner._queue_c8_mxfp_v_scale_pages(finished)
    runner._restore_pending_c8_mxfp_v_scales()
    assert torch.equal(raw, before)
    assert not runner._c8_mxfp_pending_v_scale_pages


def test_already_zeroed_pages_and_null_block_are_not_queued():
    runner, raw, _, _, _, _, _ = _runner()
    before = raw.clone()
    runner._queue_c8_mxfp_v_scale_pages(_event(pages=(0, 1), zeroed=(1,)))
    runner._restore_pending_c8_mxfp_v_scales()
    assert torch.equal(raw, before)
    assert not runner._c8_mxfp_pending_v_scale_pages


@pytest.mark.parametrize("consumer,transfer", [(False, True), (False, False)])
def test_producer_and_non_pd_execution_do_not_prepare_pending_scales(consumer, transfer):
    runner, raw, _, _, _, _, _ = _runner(consumer=consumer, transfer=transfer)
    before = raw.clone()
    runner._queue_c8_mxfp_v_scale_pages(_event())
    runner._restore_pending_c8_mxfp_v_scales()
    assert torch.equal(raw, before)
    assert not runner._c8_mxfp_pending_v_scale_pages


def test_non_c8_execution_does_not_require_scale_state():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._queue_c8_mxfp_v_scale_pages(_event())
    runner._restore_pending_c8_mxfp_v_scales()


def test_preparation_is_consumed_once():
    runner, raw, cache, _, _, _, _ = _runner()
    runner._queue_c8_mxfp_v_scale_pages(_event())
    runner._restore_pending_c8_mxfp_v_scales()
    cache[3][8].fill_(64)
    before = raw.clone()
    runner._restore_pending_c8_mxfp_v_scales()
    assert torch.equal(raw, before)


def test_cached_appends_preserve_pending_pages_of_the_same_request():
    runner, _, _, _, _, _, _ = _runner()
    runner._queue_c8_mxfp_v_scale_pages(_event())
    runner._queue_c8_mxfp_v_scale_pages(_event("cached", pages=(2,)))
    assert runner._c8_mxfp_pending_v_scale_pages == {"request": {0: {1, 2}}}


def test_only_active_requests_consume_their_pending_pages():
    runner, _, _, _, _, _, _ = _runner()
    runner._queue_c8_mxfp_v_scale_pages(_event())
    runner._queue_c8_mxfp_v_scale_pages(_event(req_id="waiting", pages=(2,)))
    runner._restore_pending_c8_mxfp_v_scales()
    assert runner._c8_mxfp_pending_v_scale_pages == {"waiting": {0: {2}}}


def test_state_update_queues_pages_skipped_by_async_load_zeroing(monkeypatch):
    runner, _, _, _, _, _, _ = _runner()
    runner.use_async_scheduling = False
    if hasattr(runner, "_update_replay_requests"):
        monkeypatch.setattr(runner, "_update_replay_requests", lambda output: None)
    monkeypatch.setattr(runner, "_apply_pp_sampled_tokens_from_scheduler_output", lambda output: None)
    monkeypatch.setattr(runner, "_track_tmp_encoder_cache_refs", lambda output: None)
    monkeypatch.setattr(GPUModelRunner, "_update_states", lambda self, output: None)
    runner._update_states(_event())
    assert runner._c8_mxfp_pending_v_scale_pages == {"request": {0: {1}}}
