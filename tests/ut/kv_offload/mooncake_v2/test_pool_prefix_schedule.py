# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Actual scheduler/cache-manager admission with CPU pool transport only."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CacheConfig, KVTransferConfig, ObservabilityConfig, SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.ascend_config import SchedulerConfig as AscendSchedulerConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.layerwise_pool_connector import LayerwisePoolConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolSaveRequest
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_worker import LayerwisePoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.object_adapter import LayerwiseObjectAdapter
from vllm_ascend.distributed.kv_transfer.utils.request_identity import pool_request_digest

from .test_pool_object_adapter import layout, memory_backend, profile, sender


def scheduler_config(cache, failure_policy="fail"):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            uses_mrope=False,
            is_encoder_decoder=False,
            max_model_len=64,
            supports_multimodal_inputs=False,
            runner_type="generate",
            is_diffusion=False,
            return_sampling_mask=False,
        ),
        cache_config=cache,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=32,
            max_model_len=64,
            enable_chunked_prefill=True,
            async_scheduling=False,
            watermark=0.0,
            is_encoder_decoder=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
        ),
        observability_config=ObservabilityConfig(),
        lora_config=None,
        kv_events_config=None,
        is_mm_encoder_only=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LayerwisePoolConnector",
            kv_role="kv_producer",
            kv_load_failure_policy=failure_policy,
        ),
        ec_transfer_config=None,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        speculative_config=None,
        num_speculative_tokens=0,
        num_lookahead_tokens=0,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=64,
        max_concurrent_batches=1,
        use_v2_model_runner=True,
        aux_output_config=SimpleNamespace(enabled=False),
        additional_config={},
    )


def request(request_id):
    return Request(
        request_id,
        list(range(9)),
        SamplingParams(max_tokens=1, ignore_eos=True, extra_args={"kv_transfer_params": {"do_remote_decode": True}}),
        None,
        block_hasher=get_request_block_hasher(4, sha256),
    )


@pytest.mark.parametrize("local_tokens", [0, 4])
@pytest.mark.parametrize(
    "case,failure_policy", [("success", "fail"), ("failure", "fail"), ("failure", "recompute"), ("cancel", "fail")]
)
def test_pool_prefix_async_wait_then_actual_scheduler_resumes_only_valid_tokens(
    monkeypatch,
    local_tokens,
    case,
    failure_policy,
):
    read_fails, cancelled = case == "failure", case == "cancel"
    init_none_hash(sha256)
    backend, _, _ = memory_backend()
    source = torch.arange(16 * 2 * 4 * 2, dtype=torch.float16).reshape(16, 2, 4, 2)
    target = torch.full_like(source, -1)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=2, dtype=torch.float16)
    cache = KVCacheConfig(
        16,
        [
            KVCacheTensor(
                size=target.untyped_storage().nbytes(),
                layers=["layer.0"],
                layer_stride=target.untyped_storage().nbytes(),
                block_stride=target.stride(0) * target.element_size(),
            )
        ],
        [KVCacheGroupSpec(["layer.0"], spec)],
    )
    config = scheduler_config(CacheConfig(block_size=4, enable_prefix_caching=True), failure_policy)
    config.cache_config.num_gpu_blocks = 16
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "register_custom_kv_cache_specs", lambda config: None)
    # The Ascend wrapper reads hardware/model configuration after constructing
    # the real scheduler. Supply only its default scheduling policy here; all
    # admission, block allocation and completion handlers remain production code.
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config",
        lambda config: SimpleNamespace(scheduler_config=AscendSchedulerConfig()),
    )
    register_all_kvcache_specs(config)
    connector = LayerwisePoolConnector(config, KVConnectorRole.SCHEDULER, cache)
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.KVConnectorFactory.create_connector", lambda **kwargs: connector)
    engine = Scheduler(config, cache, MagicMock(), block_size=4, hash_block_size=4)
    req = request("request")
    hashes = tuple(value.hex() for value in req.block_hashes)
    thread = sender(backend, 1)
    # Both entries use the same production format; the ordinary writer seeds
    # the pool and the P connector discovers it through its worker endpoint.
    source_layout = layout({"layer.0": source})
    ordinary = LayerwiseObjectAdapter(backend, source_layout, profile(), lambda name: 0, sender=thread)
    worker = LayerwisePoolWorker(
        backend,
        cache,
        worker_id="pp0/pcp0/tp0",
        tp_rank=0,
        tp_size=1,
        dcp_rank=0,
        dcp_size=1,
        lease_ttl_ms=10000,
        total_num_kv_heads=2,
        base_layers=1,
    )
    entered, release = threading.Event(), threading.Event()
    try:
        worker.register_kv_caches({"layer.0": target})
        handshake = worker.enable_manifest_control(profile(), "127.0.0.1", retention_seconds=5)
        connector.set_xfer_handshake_metadata({0: handshake})
        ordinary.prepare_save(
            "seed", PoolSaveRequest("seed", "step", pool_request_digest(req), 0, 8, ((1, 2),), (hashes,))
        )
        ordinary.submit_save_layer(0, MagicMock(), MagicMock())
        assert ordinary.finish_step() == []
        if local_tokens:
            warm = request("warm")
            blocks = engine.kv_cache_manager.allocate_slots(warm, local_tokens)
            block = blocks.get_block_ids()[0][0]
            target[block].copy_(source[1])
            engine.kv_cache_manager.cache_blocks(warm, local_tokens)
            engine.kv_cache_manager.free(warm)
        original_copy = backend.batch_copy_gva.side_effect

        def copy(gvas, buffers, sizes, direction):
            if direction == 1:
                entered.set()
                assert release.wait(5)
                if read_fails:
                    return -7
            return original_copy(gvas, buffers, sizes, direction)

        backend.batch_copy_gva.side_effect = copy
        initial_free = engine.kv_cache_manager.block_pool.get_num_free_blocks()
        engine.add_request(req)
        first = engine.schedule()
        assert first.num_scheduled_tokens == {}
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert req.num_computed_tokens == 8
        assert first.kv_connector_metadata.loads[req.request_id].token_start == local_tokens
        worker.start_load_kv(first.kv_connector_metadata)
        assert entered.wait(5)
        pending = engine.schedule()
        assert pending.num_scheduled_tokens == {}  # No model work before I/O drains.
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert not worker.get_transfer_results().finished_recving
        if cancelled:
            before_cancel = engine.kv_cache_manager.block_pool.get_num_free_blocks()
            engine.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
            assert engine.kv_cache_manager.block_pool.get_num_free_blocks() == before_cancel
            assert req.request_id in engine.requests
            pending = engine.schedule()
            assert pending.kv_connector_metadata.cancelled == {req.request_id}
            worker.start_load_kv(pending.kv_connector_metadata)
            assert not worker.get_transfer_results().finished_recving
        release.set()
        if read_fails:
            with pytest.raises(RuntimeError, match="range copy failed"):
                worker.reader._active[req.request_id].future.result(timeout=5)
        else:
            worker.reader._active[req.request_id].future.result(timeout=5)
        result = worker.get_transfer_results()
        assert result.finished_recving == {req.request_id}
        assert result.failed_recving == ({req.request_id} if read_fails or cancelled else set())
        output = KVConnectorOutput(
            finished_recving=result.finished_recving,
            failed_recving=result.failed_recving,
            invalid_block_ids=worker.get_block_ids_with_load_errors(),
        )
        engine.update_from_output(pending, ModelRunnerOutput.with_kv_conn_output_only(output))
        if cancelled or (read_fails and failure_policy == "fail"):
            assert req.request_id not in engine.requests
            assert engine.kv_cache_manager.block_pool.get_num_free_blocks() == initial_free
            assert engine.schedule().num_scheduled_tokens == {}
            return
        resumed = engine.schedule()
        assert resumed.num_scheduled_tokens == {req.request_id: 9 - local_tokens if read_fails else 1}
        assert req.status == RequestStatus.RUNNING
        if read_fails:
            assert connector.connector_scheduler._source[req.request_id].prefix_failed
        if not read_fails:
            blocks = engine.kv_cache_manager.get_blocks(req.request_id).get_block_ids()[0]
            assert torch.equal(target[blocks[0]], source[1])
            assert torch.equal(target[blocks[1]], source[2])
    finally:
        release.set()
        worker.shutdown()
        ordinary.close()
        thread.stop()
        thread.join(5)
        engine.shutdown()
