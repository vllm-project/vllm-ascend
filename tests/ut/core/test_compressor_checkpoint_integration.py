# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real vLLM cache integration on host; no compressor numerics are emulated."""

import json

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.request import Request

from vllm_ascend.core.compressor_checkpoint import TailCheckpointKey
from vllm_ascend.core.compressor_checkpoint_coordinator import CompressorCheckpointCoordinator
from vllm_ascend.core.kv_cache_interface import (
    AscendCompressorTailSpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.worker.compressor_checkpoint import copy_compressor_tail_pages


def cache_config(num_blocks=4096):
    # A3 block_size=32: C4/C128 state pages hold 2/8 token rows.
    specs = {
        "attention.c4": AscendMLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            compress_ratio=4,
            model_version="deepseek_v4",
        ),
        "attention.c128": AscendMLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            compress_ratio=128,
            model_version="deepseek_v4",
        ),
        "swa": AscendSlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            sliding_window=128,
        ),
        "compressor.c4": AscendCompressorTailSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            sliding_window=8,
            compress_ratio=4,
            model_version="deepseek_v4",
            tail_tokens=8,
            ring_blocks_per_request=4,
            state_dim=4,
        ),
        "compressor.c128": AscendCompressorTailSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float32,
            sliding_window=128,
            compress_ratio=128,
            model_version="deepseek_v4",
            tail_tokens=128,
            ring_blocks_per_request=16,
            state_dim=4,
        ),
    }
    groups = [
        KVCacheGroupSpec([name], UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs={name: spec}))
        for name, spec in specs.items()
    ]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


@pytest.fixture
def coordinator():
    init_none_hash(sha256)
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    return CompressorCheckpointCoordinator(
        cache_config(),
        max_model_len=32768,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=2,
        scheduler_block_size=32,
        max_in_flight_tokens=4096,
    )


def request(name, length=8193, suffix=1):
    params = SamplingParams(max_tokens=2, temperature=0)
    params.update_from_generation_config({}, eos_token_id=255)
    return Request(
        request_id=name,
        prompt_token_ids=[10] * (length - 1) + [suffix],
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(2, sha256),
    )


def save_boundary(coordinator, req, position):
    coordinator.allocate_new_blocks(req.request_id, position, position)
    coordinator.cache_completed_blocks(req, position)
    key = TailCheckpointKey(req.block_hashes[position // 2 - 1], position)
    source = {group: manager.req_to_blocks[req.request_id] for group, manager in coordinator.tail_managers.items()}
    plan = coordinator.checkpoints.reserve(key, source)
    assert plan is not None
    coordinator.checkpoints.complete_save(plan.handle)
    return plan


def test_uniform_layout_uses_logical_compressed_granularity(coordinator):
    assert coordinator.lcm_block_size == 4096
    assert coordinator.scheduler_block_size == 4096
    assert coordinator.tail_managers[3].ring_blocks_per_request == 4
    assert coordinator.tail_managers[4].ring_blocks_per_request == 16
    assert len(coordinator.attention_groups) == 3


def test_base_coordinator_trims_both_dense_groups_after_swa_miss(coordinator):
    config = cache_config()
    config.kv_cache_groups = config.kv_cache_groups[:3]
    base = AscendHybridKVCacheCoordinator(
        config,
        max_model_len=32768,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=2,
        scheduler_block_size=4096,
        max_in_flight_tokens=4096,
    )
    req = request("base-source", length=12289)
    base.allocate_new_blocks(req.request_id, 8192, 8192)
    base.cache_blocks(req, 8192)
    later_window = base.single_type_managers[2].req_to_blocks[req.request_id][128:256]
    base.block_pool.evict_blocks({block.block_id for block in later_window if not block.is_null})
    blocks, position, _ = base.find_longest_cache_hit(req.block_hashes, 8192)
    assert position == 4096
    assert len(blocks[0]) == 32
    assert len(blocks[1]) == 1


@pytest.mark.parametrize("common_prefix_length", [4096, 6144, 8192])
def test_complete_restore_allocates_private_fixed_rings(coordinator, common_prefix_length):
    source = request("source")
    save = save_boundary(coordinator, source, 4096)
    borrower = request("borrower", length=common_prefix_length + 1, suffix=2)
    blocks, position, _ = coordinator.find_longest_cache_hit(borrower.block_hashes, common_prefix_length)
    assert position == 4096
    needed = coordinator.get_num_blocks_to_allocate(
        borrower.request_id,
        position + 1,
        blocks,
        0,
        position,
        position,
        position + 1,
    )
    assert needed >= 20
    coordinator.allocate_new_computed_blocks(borrower.request_id, blocks, position, 0)
    coordinator.allocate_new_blocks(borrower.request_id, position + 1, position + 1)
    handle = coordinator.pending_restores.pop(borrower.request_id)
    assert handle == save.handle
    private = {
        group: manager.req_to_blocks[borrower.request_id] for group, manager in coordinator.tail_managers.items()
    }
    restore = coordinator.checkpoints.restore_plan(handle, private)
    assert all(op.source != op.destination for op in restore.copies)
    assert [len(private[group]) for group in (3, 4)] == [4, 16]
    coordinator.checkpoints.release(handle)
    coordinator.free(source.request_id)
    coordinator.free(borrower.request_id)
    coordinator.checkpoints.reset()
    assert coordinator.block_pool.get_num_free_blocks() == coordinator.kv_cache_config.num_blocks - 1


def test_missing_swa_rechecks_earlier_checkpoint_and_trims_all_groups(coordinator):
    source = request("source", length=12289)
    save_boundary(coordinator, source, 4096)
    save_boundary(coordinator, source, 8192)
    # Remove only the later window from prefix lookup while keeping the tail.
    swa = coordinator.single_type_managers[2]
    later_window = swa.req_to_blocks[source.request_id][4096 // 32 : 8192 // 32]
    coordinator.block_pool.evict_blocks({block.block_id for block in later_window if not block.is_null})
    blocks, position, _ = coordinator.find_longest_cache_hit(source.block_hashes, 8192)
    assert position == 4096
    assert len(blocks[0]) == 4096 // (32 * 4)
    assert len(blocks[1]) == 4096 // (32 * 128)
    earlier_window = swa.req_to_blocks[source.request_id][: 4096 // 32]
    coordinator.block_pool.evict_blocks({block.block_id for block in earlier_window if not block.is_null})
    assert coordinator.find_longest_cache_hit(source.block_hashes, 8192)[1] == 0


def test_ordinary_kv_and_tail_are_not_published_before_completion(coordinator):
    req = request("source")
    coordinator.allocate_new_blocks(req.request_id, 4096, 4096)
    coordinator.cache_blocks(req, 4096)
    assert all(block.block_hash is None for block in coordinator.single_type_managers[0].req_to_blocks[req.request_id])
    sources = {group: manager.req_to_blocks[req.request_id] for group, manager in coordinator.tail_managers.items()}
    save = coordinator.checkpoints.reserve(TailCheckpointKey(req.block_hashes[2047], 4096), sources)
    coordinator.cache_completed_blocks(req, 4096)
    assert coordinator.find_longest_cache_hit(req.block_hashes, 4096)[1] == 0
    coordinator.checkpoints.complete_save(save.handle)
    assert coordinator.find_longest_cache_hit(req.block_hashes, 4096)[1] == 4096


def test_torch_page_copy_preserves_padding_slots_and_independent_borrowers(coordinator):
    req = request("source")
    coordinator.allocate_new_blocks(req.request_id, 4096, 4096)
    sources = {group: manager.req_to_blocks[req.request_id] for group, manager in coordinator.tail_managers.items()}
    pages = {group: [torch.arange(4096 * 17, dtype=torch.int32).view(4096, 17)] for group in sources}
    save = coordinator.checkpoints.reserve(TailCheckpointKey(req.block_hashes[2047], 4096), sources)
    expected = {op.destination: pages[op.group_id][0][op.source].clone() for op in save.copies}
    copy_compressor_tail_pages([save], pages)
    coordinator.checkpoints.complete_save(save.handle)
    coordinator.checkpoints.acquire(save.handle)
    for op in save.copies:
        pages[op.group_id][0][op.source].fill_(-9)
    destinations = {group: coordinator.block_pool.get_new_blocks(len(blocks)) for group, blocks in sources.items()}
    restore = coordinator.checkpoints.restore_plan(save.handle, destinations)
    copy_compressor_tail_pages([restore], pages)
    for op in restore.copies:
        assert torch.equal(pages[op.group_id][0][op.destination], expected[op.source])
    coordinator.checkpoints.release(save.handle)


@pytest.fixture
def scheduler(tmp_path, coordinator, monkeypatch, request):
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.structured_output import StructuredOutputManager

    from vllm_ascend.core.compressor_checkpoint_scheduler import CompressorCheckpointScheduler

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))

    options = getattr(request, "param", {})
    max_model_len = options.get("max_model_len", 32768)

    # Only local model metadata is needed for scheduler construction. The cache
    # groups above are the real compressor layouts, not this tiny OPT model's KV.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "hidden_size": 64,
                "ffn_dim": 256,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 256,
                "max_position_embeddings": max_model_len,
                "torch_dtype": "float32",
            }
        )
    )
    model = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True, dtype="float32")
    config = VllmConfig(
        model_config=model,
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=options.get("token_budget", 6000),
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            async_scheduling=False,
            watermark=0,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=32, enable_prefix_caching=True),
    )
    kv = cache_config(options.get("num_blocks", 4096))
    config.cache_config.num_gpu_blocks = kv.num_blocks
    return CompressorCheckpointScheduler(
        vllm_config=config,
        kv_cache_config=kv,
        block_size=32,
        hash_block_size=2,
        structured_output_manager=StructuredOutputManager(config),
        log_stats=False,
    )


def complete_step(scheduler, output):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(output.num_scheduled_tokens)
    sampled = [
        [20]
        if req_id in scheduler.requests
        and scheduler.requests[req_id].num_computed_tokens >= scheduler.requests[req_id].num_tokens
        else []
        for req_id in req_ids
    ]
    return scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_real_scheduler_chunks_publishes_restores_and_skips_prefix(scheduler):
    source = request("source", length=8193)
    scheduler.add_request(source)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"source": 4096}
    assert len(first.compressor_saves) == 1
    assert scheduler.checkpoint_coordinator.find_longest_cache_hit(source.block_hashes, 4096)[1] == 0
    complete_step(scheduler, first)
    assert scheduler.checkpoint_coordinator.find_longest_cache_hit(source.block_hashes, 4096)[1] == 4096
    borrower = request("borrower", length=4097, suffix=2)
    scheduler.add_request(borrower)
    second = scheduler.schedule()
    assert second.num_scheduled_tokens["borrower"] == 1
    assert len(second.compressor_restores) == 1
    assert second.compressor_restores[0].position == 4096
    assert next(req for req in second.scheduled_new_reqs if req.req_id == "borrower").num_computed_tokens == 4096
    complete_step(scheduler, second)


def test_real_scheduler_cancel_inflight_copy_and_reset(scheduler):
    from vllm.v1.request import RequestStatus

    source = request("source")
    scheduler.add_request(source)
    step = scheduler.schedule()
    scheduler.finish_requests("source", RequestStatus.FINISHED_ABORTED)
    assert scheduler.reset_prefix_cache() is False
    complete_step(scheduler, step)
    assert scheduler.reset_prefix_cache() is True
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def test_scheduler_does_not_split_when_snapshot_budget_is_zero(scheduler):
    scheduler.checkpoint_coordinator.checkpoints.max_blocks = 0
    source = request("source")
    scheduler.add_request(source)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens["source"] == 6000
    assert not step.compressor_saves
    complete_step(scheduler, step)


def test_cancel_inflight_restore_retains_snapshot_and_destination(scheduler):
    from vllm.v1.request import RequestStatus

    source = request("source")
    scheduler.add_request(source)
    complete_step(scheduler, scheduler.schedule())
    scheduler.finish_requests("source", RequestStatus.FINISHED_ABORTED)

    borrower = request("borrower", length=4097, suffix=2)
    scheduler.add_request(borrower)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"borrower": 1}
    (restore,) = step.compressor_restores
    pool = scheduler.checkpoint_coordinator.checkpoints
    blocks = scheduler.kv_cache_manager.block_pool
    scheduler.finish_requests("borrower", RequestStatus.FINISHED_ABORTED)

    # Reclaim may remove idle snapshots, but neither copy endpoint may be
    # reused while this worker step is still reading/writing its pages.
    pool.reclaim(blocks.num_gpu_blocks)
    assert pool.used_blocks == pool.blocks_per_checkpoint
    for operation in restore.copies:
        assert blocks.blocks[operation.source].ref_cnt > 0
        assert blocks.blocks[operation.destination].ref_cnt > 0

    complete_step(scheduler, step)
    assert scheduler.reset_prefix_cache()
    assert blocks.get_num_free_blocks() == blocks.num_gpu_blocks - 1


def test_scheduler_output_survives_worker_transport(scheduler):
    import pickle

    source = request("source")
    scheduler.add_request(source)
    step = scheduler.schedule()
    transported = pickle.loads(pickle.dumps(step))
    assert transported.compressor_saves == step.compressor_saves
    assert transported.compressor_restores == step.compressor_restores
    complete_step(scheduler, transported)


def test_copy_covers_main_and_indexer_layers_in_same_group(coordinator):
    req = request("source")
    coordinator.allocate_new_blocks(req.request_id, 4096, 4096)
    sources = {group: manager.req_to_blocks[req.request_id] for group, manager in coordinator.tail_managers.items()}
    save = coordinator.checkpoints.reserve(TailCheckpointKey(req.block_hashes[2047], 4096), sources)
    main = torch.arange(4096 * 16, dtype=torch.float32).view(4096, 16)
    indexer = -torch.arange(4096 * 8, dtype=torch.float32).view(4096, 8)
    c128 = torch.arange(4096 * 32, dtype=torch.float32).view(4096, 32)
    pages = {3: [main, indexer], 4: [c128]}
    copy_compressor_tail_pages([save], pages)
    for operation in save.copies:
        for tensor in pages[operation.group_id]:
            assert torch.equal(tensor[operation.source], tensor[operation.destination])


def test_preemption_before_dispatch_releases_pending_restore(coordinator):
    source = request("source")
    save_boundary(coordinator, source, 4096)
    borrower = request("borrower", length=4097, suffix=2)
    blocks, position, _ = coordinator.find_longest_cache_hit(borrower.block_hashes, 4096)
    coordinator.allocate_new_computed_blocks(borrower.request_id, blocks, position, 0)
    coordinator.allocate_new_blocks(borrower.request_id, position + 1, position + 1)
    assert borrower.request_id in coordinator.pending_restores
    # The scheduler's preemption/free path must release the undispatched reader.
    coordinator.free(borrower.request_id)
    assert borrower.request_id not in coordinator.pending_restores
    coordinator.checkpoints.reclaim(coordinator.kv_cache_config.num_blocks)
    assert coordinator.checkpoints.used_blocks == 0
    coordinator.free(source.request_id)
    assert coordinator.block_pool.get_num_free_blocks() == coordinator.kv_cache_config.num_blocks - 1


def test_failed_reset_preserves_checkpoint_and_kv(scheduler):
    source = request("source")
    scheduler.add_request(source)
    complete_step(scheduler, scheduler.schedule())
    coordinator = scheduler.checkpoint_coordinator
    before = coordinator.find_longest_cache_hit(source.block_hashes, 4096)
    assert before[1] == 4096
    assert not scheduler.reset_prefix_cache()
    assert coordinator.find_longest_cache_hit(source.block_hashes, 4096) == before
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert coordinator.checkpoints.used_blocks == 0
    assert coordinator.find_longest_cache_hit(source.block_hashes, 4096)[1] == 0
    step = scheduler.schedule()
    assert step.num_scheduled_tokens["source"] == 4096
    assert not step.compressor_restores
    complete_step(scheduler, step)


@pytest.mark.parametrize("scheduler", [{"num_blocks": 336, "token_budget": 128, "max_model_len": 65536}], indirect=True)
def test_failed_restore_admission_retries_as_miss(scheduler):
    from vllm.v1.request import RequestStatus

    source = request("source")
    scheduler.add_request(source)
    while source.num_computed_tokens < 4096:
        complete_step(scheduler, scheduler.schedule())
    scheduler.finish_requests("source", RequestStatus.FINISHED_ABORTED)
    coordinator = scheduler.checkpoint_coordinator
    assert coordinator.find_longest_cache_hit(source.block_hashes, 4096)[1] == 4096

    # With 336 pages and 128-token chunks, the full-sequence admission
    # needs 321 pages for a hit (315 free), or 326 for a miss (335 free).
    borrower = request("borrower", length=36865)
    scheduler.add_request(borrower)
    first = scheduler.schedule()
    assert not first.num_scheduled_tokens
    complete_step(scheduler, first)
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"borrower": 128}
    assert not second.compressor_restores
    assert second.scheduled_new_reqs[0].num_computed_tokens == 0
    complete_step(scheduler, second)
    # The miss must finish, including later checkpoint creation/reclamation,
    # rather than merely moving the admission stall to a later chunk.
    for _ in range((borrower.num_prompt_tokens + 127) // 128 + 2):
        if borrower.is_finished():
            break
        step = scheduler.schedule()
        assert step.num_scheduled_tokens
        complete_step(scheduler, step)
    assert borrower.is_finished()


@pytest.mark.parametrize("has_scheduled_reqs", [False, True])
def test_admission_reclaims_snapshots_for_watermark(scheduler, has_scheduled_reqs):
    coordinator = scheduler.checkpoint_coordinator
    source = request("source", length=12289)
    selected = save_boundary(coordinator, source, 4096)
    save_boundary(coordinator, source, 8192)
    save_boundary(coordinator, source, 12288)
    manager = scheduler.kv_cache_manager
    borrower = request("borrower", length=4097)
    blocks, hit, _ = manager.get_computed_blocks(borrower)
    assert hit == 4096
    needed = coordinator.get_num_blocks_to_allocate("borrower", 4097, blocks.blocks, 0, hit, hit, 4097)
    manager.watermark_blocks = 32
    pool = manager.block_pool
    # Hold real allocator pages to leave enough for the request itself but
    # not its watermark. Source still owns KV, so pressure cannot evict it.
    pressure = pool.get_new_blocks(pool.get_num_free_blocks() - needed - 2)
    allocated = manager.allocate_slots(
        borrower, 1, num_new_computed_tokens=hit, new_computed_blocks=blocks, has_scheduled_reqs=has_scheduled_reqs
    )
    assert allocated is not None
    if has_scheduled_reqs:
        assert pool.get_num_free_blocks() >= manager.watermark_blocks
    assert coordinator.pending_restores["borrower"] == selected.handle
    assert coordinator.checkpoints.used_blocks == coordinator.checkpoints.blocks_per_checkpoint * (
        1 if has_scheduled_reqs else 3
    )
    assert coordinator.admission_watermark == 0
    coordinator.free("borrower")
    coordinator.free("source")
    pool.free_blocks(pressure)
    coordinator.checkpoints.reset()
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
