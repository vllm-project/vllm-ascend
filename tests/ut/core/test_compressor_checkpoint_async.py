# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real scheduler/allocator regressions with multiple outstanding worker steps."""

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.ascend_config import (
    DSV4_ASYNC_CHECKPOINT_SCHEDULER,
    DSV4_CHECKPOINT_SCHEDULER,
    get_dsv4_checkpoint_fallback_reasons,
)

spec = importlib.util.spec_from_file_location(
    "async_checkpoint_fixtures",
    Path(__file__).with_name("test_compressor_checkpoint_integration.py"),
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    fixtures.coordinator.__wrapped__()
    return fixtures.scheduler.__wrapped__(
        tmp_path, None, monkeypatch, SimpleNamespace(param={"async_scheduling": True})
    )


def complete(scheduler, output, tokens=None):
    req_ids = list(output.num_scheduled_tokens)
    return scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: index for index, rid in enumerate(req_ids)},
            sampled_token_ids=[(tokens or {}).get(rid, []) for rid in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_completed_range_never_publishes_later_scheduled_chunk(scheduler):
    req = fixtures.request("source", length=12289)
    scheduler.add_request(req)
    first = pickle.loads(pickle.dumps(scheduler.schedule()))
    second = scheduler.schedule()
    assert isinstance(scheduler, AsyncScheduler)
    assert req.num_computed_tokens == 8192
    coordinator = scheduler.checkpoint_coordinator
    assert coordinator.find_longest_cache_hit(req.block_hashes, 8192)[1] == 0
    with patch.object(coordinator, "cache_completed_blocks", wraps=coordinator.cache_completed_blocks) as cache:
        complete(scheduler, first)
        assert cache.call_args.args[1] == 4096
        assert coordinator.find_longest_cache_hit(req.block_hashes, 8192)[1] == 4096
        complete(scheduler, second)
        assert cache.call_args.args[1] == 8192
        assert coordinator.find_longest_cache_hit(req.block_hashes, 8192)[1] == 8192


def test_native_async_placeholders_and_single_cache_publication(scheduler):
    req = fixtures.request("source", length=4097)
    scheduler.add_request(req)
    first = scheduler.schedule()
    second = scheduler.schedule()
    third = scheduler.schedule()
    assert req.num_output_placeholders == 2
    assert third.num_scheduled_tokens == {"source": 1}
    complete(scheduler, first)
    coordinator = scheduler.checkpoint_coordinator
    with patch.object(coordinator, "cache_completed_blocks", wraps=coordinator.cache_completed_blocks) as cache:
        complete(scheduler, second, {"source": [20]})
        assert req.num_output_placeholders == 1
        assert cache.call_count == 1
        assert cache.call_args.args[1] == 4097
    complete(scheduler, third, {"source": [21]})
    assert req.num_output_placeholders == 0
    assert req.is_finished()


def test_two_inflight_saves_cancel_and_reset_drain_once(scheduler):
    req = fixtures.request("source", length=12289)
    scheduler.add_request(req)
    first, second = scheduler.schedule(), scheduler.schedule()
    scheduler.finish_requests("source", RequestStatus.FINISHED_ABORTED)
    blocks = scheduler.kv_cache_manager.block_pool
    rings = [operation.source for operation in first.compressor_saves[0].copies]
    assert all(blocks.blocks[index].ref_cnt > 0 for index in rings)
    assert not scheduler.reset_prefix_cache(reset_running_requests=True)
    complete(scheduler, first)
    assert all(blocks.blocks[index].ref_cnt > 0 for index in rings)
    assert not scheduler.reset_prefix_cache()
    complete(scheduler, second)
    assert scheduler.reset_prefix_cache()
    assert blocks.get_num_free_blocks() == blocks.num_gpu_blocks - 1


def test_preempted_output_cannot_publish_new_request_pages(scheduler):
    req = fixtures.request("source", length=12289)
    scheduler.add_request(req)
    first = scheduler.schedule()
    old_generation = first.compressor_completed_ranges["source"].num_preemptions
    scheduler.running.remove(req)
    scheduler._preempt_request(req, 0.0)
    # Current upstream retains in-flight pages and refuses readmission until
    # the old execution drains. Preserve that fence instead of assuming the
    # old version's immediate readmission/reallocation behavior.
    blocked = scheduler.schedule()
    assert "source" not in blocked.num_scheduled_tokens
    coordinator = scheduler.checkpoint_coordinator
    with patch.object(coordinator, "cache_completed_blocks", wraps=coordinator.cache_completed_blocks) as cache:
        complete(scheduler, first)
        cache.assert_not_called()
        assert coordinator.find_longest_cache_hit(req.block_hashes, 4096)[1] == 0
        second = scheduler.schedule()
        assert second.compressor_completed_ranges["source"].num_preemptions > old_generation
        complete(scheduler, second)
        assert cache.call_count == 1


def test_inflight_restore_and_destination_survive_cancel(scheduler):
    source = fixtures.request("source", length=4097)
    scheduler.add_request(source)
    complete(scheduler, scheduler.schedule())
    scheduler.finish_requests("source", RequestStatus.FINISHED_ABORTED)
    borrower = fixtures.request("borrower", length=8193, suffix=2)
    scheduler.add_request(borrower)
    first, second = scheduler.schedule(), scheduler.schedule()
    (restore,) = first.compressor_restores
    assert restore.position == 4096
    scheduler.finish_requests("borrower", RequestStatus.FINISHED_ABORTED)
    pool = scheduler.checkpoint_coordinator.checkpoints
    blocks = scheduler.kv_cache_manager.block_pool
    pool.reclaim(blocks.num_gpu_blocks)
    assert pool._entries[restore.handle].readers == 1
    assert all(blocks.blocks[copy.destination].ref_cnt > 0 for copy in restore.copies)
    complete(scheduler, first)
    assert pool._entries[restore.handle].readers == 0
    assert all(blocks.blocks[copy.destination].ref_cnt > 0 for copy in restore.copies)
    complete(scheduler, second)
    assert scheduler.reset_prefix_cache()
    assert blocks.get_num_free_blocks() == blocks.num_gpu_blocks - 1


def checkpoint_config(async_mode, selected):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4"), enforce_eager=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True, block_size=128),
        scheduler_config=SimpleNamespace(
            async_scheduling=async_mode, scheduler_cls=selected, enable_chunked_prefill=True
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        use_v2_model_runner=True,
        kv_events_config=None,
        kv_transfer_config=None,
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "async_mode,selected",
    [(False, DSV4_CHECKPOINT_SCHEDULER), (True, DSV4_ASYNC_CHECKPOINT_SCHEDULER)],
)
def test_gate_accepts_matching_scheduler_and_current_mrv2_layout(async_mode, selected):
    config = checkpoint_config(async_mode, selected)
    assert (
        get_dsv4_checkpoint_fallback_reasons(config, checkpoint_capable=True, multistream_dsv4_dsa_overlap=False) == []
    )
    config.scheduler_config.scheduler_cls = DSV4_CHECKPOINT_SCHEDULER if async_mode else DSV4_ASYNC_CHECKPOINT_SCHEDULER
    assert "compressor checkpoints require the checkpoint scheduler" in get_dsv4_checkpoint_fallback_reasons(
        config, checkpoint_capable=True, multistream_dsv4_dsa_overlap=False
    )


@pytest.mark.parametrize(
    "prefix,async_mode,expected",
    [(True, True, DSV4_ASYNC_CHECKPOINT_SCHEDULER), (True, False, DSV4_CHECKPOINT_SCHEDULER), (False, True, None)],
)
def test_platform_selects_checkpoint_scheduler_only_for_prefix(prefix, async_mode, expected):
    from vllm_ascend.platform import _setup_worker_and_scheduler

    config = checkpoint_config(async_mode, None)
    config.cache_config.enable_prefix_caching = prefix
    config.parallel_config.worker_cls = "test_worker"
    ascend = SimpleNamespace(
        enable_dsv4_shared_compressor_workspace=True,
        scheduler_config=SimpleNamespace(
            dyntra_lb_config=SimpleNamespace(enabled=False),
            recompute_scheduler_enable=False,
            profiling_chunk_config=SimpleNamespace(enabled=False),
            batch_job_sched_config=SimpleNamespace(enabled=False),
        ),
    )
    with (
        patch("vllm_ascend.platform.refresh_block_size"),
        patch("vllm_ascend.platform.get_current_hardware_profile") as profile,
    ):
        profile.return_value.supports.return_value = False
        _setup_worker_and_scheduler(config, ascend)
    assert config.scheduler_config.scheduler_cls == expected


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("use_v2_model_runner", False, "Model Runner V2"),
        ("model_config.enforce_eager", False, "eager Attention"),
        ("parallel_config.pipeline_parallel_size", 2, "pipeline_parallel_size=1"),
        ("scheduler_config.async_scheduling", None, "explicit scheduling mode"),
        ("scheduler_config.enable_chunked_prefill", False, "chunked prefill"),
    ],
)
def test_checkpoint_gate_rejects_unsupported_execution(field, value, reason):
    config = checkpoint_config(True, DSV4_ASYNC_CHECKPOINT_SCHEDULER)
    obj = config
    parts = field.split(".")
    for name in parts[:-1]:
        obj = getattr(obj, name)
    setattr(obj, parts[-1], value)
    reasons = get_dsv4_checkpoint_fallback_reasons(config, checkpoint_capable=True, multistream_dsv4_dsa_overlap=False)
    assert any(reason in item for item in reasons)
