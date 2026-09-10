# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CacheConfig, ObservabilityConfig, ParallelConfig, SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, KVCacheSpecRegistry
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.batch_invariant_config import (
    ASYNC_SCHEDULER,
    SCHEDULER,
    configure_batch_invariant,
    select_batch_invariant_scheduler,
)
from vllm_ascend.core.batch_invariant_scheduler import (
    BatchInvariantAsyncScheduler,
    BatchInvariantScheduler,
    reserve_generation_slots,
)


def make_manager(num_blocks=7, max_model_len=128):
    KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)],
    )
    return KVCacheManager(config, max_model_len, scheduler_block_size=16, hash_block_size=16, enable_caching=False)


def make_request(request_id, prompt=16, output=48):
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt,
        sampling_params=SamplingParams(max_tokens=output),
        pooling_params=None,
    )


def make_config(async_scheduling=False):
    return SimpleNamespace(
        kv_transfer_config=None,
        model_config=SimpleNamespace(max_model_len=128),
        cache_config=CacheConfig(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=8,
            max_num_batched_tokens=32,
            max_num_scheduled_tokens=24,
            max_num_encoder_input_tokens=32,
            encoder_cache_size=32,
            scheduler_cls=None,
            async_scheduling=async_scheduling,
        ),
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("block_size", [None, 16, 128, 256])
def test_config_disables_unsupported_features(monkeypatch, async_scheduling, block_size):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    logger = Mock()
    monkeypatch.setattr("vllm_ascend.batch_invariant_config.logger", logger)
    config = make_config(async_scheduling)
    config.cache_config = CacheConfig(block_size=block_size, enable_prefix_caching=True)
    configure_batch_invariant(config)
    select_batch_invariant_scheduler(config)
    scheduler = config.scheduler_config
    assert not scheduler.enable_chunked_prefill
    assert not config.cache_config.enable_prefix_caching
    assert config.cache_config.block_size == 128
    assert scheduler.long_prefill_token_threshold == 0
    assert scheduler.max_num_batched_tokens == scheduler.max_num_scheduled_tokens == 128
    assert scheduler.encoder_cache_size == scheduler.max_num_encoder_input_tokens == 128
    assert scheduler.scheduler_cls == (ASYNC_SCHEDULER if async_scheduling else SCHEDULER)
    logger.warning.assert_called_once_with(
        "VLLM_BATCH_INVARIANT=1: Batch invariance disables chunked prefill, prefix caching and request preemption. "
        "block_size=%d, max_num_batched_tokens=%d; reserving generation KV capacity may reduce concurrency.",
        128,
        128,
    )
    # Repeated platform initialization must be harmless.
    configure_batch_invariant(config)
    select_batch_invariant_scheduler(config)


def test_disabled_mode_preserves_config(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    logger = Mock()
    monkeypatch.setattr("vllm_ascend.batch_invariant_config.logger", logger)
    config = make_config()
    config.cache_config.block_size = 256
    config.scheduler_config.scheduler_cls = "custom.Scheduler"
    configure_batch_invariant(config)
    select_batch_invariant_scheduler(config)
    assert config.cache_config.enable_prefix_caching
    assert config.cache_config.block_size == 256
    assert config.scheduler_config.enable_chunked_prefill
    assert config.scheduler_config.max_num_batched_tokens == 32
    assert config.scheduler_config.scheduler_cls == "custom.Scheduler"
    logger.warning.assert_not_called()


def test_reject_unsupported_config(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config = make_config()
    config.kv_transfer_config = object()
    with pytest.raises(ValueError, match="KV transfer"):
        configure_batch_invariant(config)
    config.scheduler_config.scheduler_cls = "custom.Scheduler"
    with pytest.raises(ValueError, match="custom"):
        select_batch_invariant_scheduler(config)


def test_reservation_queues_until_running_request_frees_blocks():
    # Six usable blocks: each 16-token prompt plus 48-token output needs four.
    manager = make_manager()
    reserve_generation_slots(manager)
    first, second = make_request("first"), make_request("second")
    assert manager.allocate_slots(first, 16) is not None
    assert manager.block_pool.get_num_free_blocks() == 2
    assert manager.allocate_slots(second, 16) is None
    # Every decode step fits without taking any more blocks from the pool.
    first.num_computed_tokens = 16
    for _ in range(47):
        first.append_output_token_ids(2)
        assert manager.allocate_slots(first, 1) is not None
        first.num_computed_tokens += 1
        assert manager.block_pool.get_num_free_blocks() == 2
    assert first.num_preemptions == second.num_preemptions == 0
    manager.free(first)
    assert manager.allocate_slots(second, 16) is not None


def test_oversized_request_fails_without_allocating():
    manager = make_manager(num_blocks=4)
    reserve_generation_slots(manager)
    with pytest.raises(ValueError, match="total KV capacity"):
        manager.allocate_slots(make_request("oversized"), 16)
    assert manager.block_pool.get_num_free_blocks() == 3


def test_max_model_len_caps_reservation():
    manager = make_manager(num_blocks=5, max_model_len=64)
    reserve_generation_slots(manager)
    assert manager.allocate_slots(make_request("capped", output=1024), 16) is not None
    assert manager.block_pool.get_num_free_blocks() == 0


def test_speculative_headroom_reserved_at_admission():
    manager = make_manager(num_blocks=6)
    reserve_generation_slots(manager, num_spec_tokens=4)
    request = make_request("spec")
    assert manager.allocate_slots(request, num_new_tokens=16, num_lookahead_tokens=0) is not None
    assert manager.block_pool.get_num_free_blocks() == 0
    request.num_computed_tokens = 60
    assert manager.allocate_slots(request, num_new_tokens=4, num_lookahead_tokens=4) is not None


def test_reject_recycling_cache_layout():
    manager = make_manager()
    manager.kv_cache_config.kv_cache_groups[0].kv_cache_spec = SlidingWindowSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16, sliding_window=32
    )
    with pytest.raises(ValueError, match="full-attention"):
        reserve_generation_slots(manager)


@pytest.mark.parametrize("scheduler_cls", [BatchInvariantScheduler, BatchInvariantAsyncScheduler])
def test_explicit_eviction_refused_without_removing_running_request(scheduler_cls):
    scheduler = object.__new__(scheduler_cls)
    request = make_request("running")
    scheduler.running = [request]
    assert scheduler.reset_prefix_cache(reset_running_requests=True) is False
    assert scheduler.running == [request]
    with pytest.raises(RuntimeError, match="preemption"):
        scheduler._preempt_request(request, 0)


@pytest.mark.parametrize("scheduler_cls", [BatchInvariantScheduler, BatchInvariantAsyncScheduler])
def test_schedule_admission_and_cancellation(scheduler_cls):
    manager = make_manager()
    cache_config = CacheConfig(block_size=16, enable_prefix_caching=False)
    cache_config.num_gpu_blocks = 7
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=128,
            enable_return_routed_experts=False,
            uses_mrope=False,
            uses_xdrope=False,
            return_sampling_mask=False,
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=128,
            max_model_len=128,
            is_encoder_decoder=False,
            max_num_seqs=2,
            enable_chunked_prefill=False,
            watermark=0,
        ),
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        speculative_config=None,
        num_speculative_tokens=0,
        num_lookahead_tokens=0,
        is_mm_encoder_only=False,
        max_in_flight_tokens=128,
        use_v2_model_runner=False,
    )
    scheduler = scheduler_cls(
        vllm_config=config,
        kv_cache_config=manager.kv_cache_config,
        structured_output_manager=Mock(),
        block_size=16,
        mm_registry=Mock(supports_multimodal_inputs=Mock(return_value=False)),
    )
    first, second = make_request("first"), make_request("second")
    scheduler.add_request(first)
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"first": 16}
    assert second.status == RequestStatus.WAITING
    assert not output.preempted_req_ids
    # Cancellation uses the ordinary release path and admits the waiting request.
    scheduler.finish_requests("first", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"second": 16}
    assert not output.preempted_req_ids
    assert first.num_preemptions == second.num_preemptions == 0
