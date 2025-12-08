# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile

import numpy as np
import pytest
import torch
from vllm.attention.backends.abstract import MultipleOf
from vllm.attention.layer import Attention
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.distributed.parallel_state import (cleanup_dist_env_and_memory,
                                             init_distributed_environment,
                                             initialize_model_parallel)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE = "cpu"


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def initialize_kv_cache(runner: NPUModelRunner):
    """
    Only perform necessary steps in NPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(
            runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
        kernel_block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
    )
    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="bfloat16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    model_config = vllm_config.model_config
    num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    vllm_config.compilation_config.static_forward_context[
        "layer.0"] = Attention(num_heads, head_size, 0.1)
    runner = NPUModelRunner(vllm_config, DEVICE)
    initialize_kv_cache(runner)
    return runner


model_runner_2 = model_runner


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0], ),
                num_computed_tokens=0,
                lora_request=None,
            ))
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(model_runner,
                                  sampling_metadata_before: SamplingMetadata):
    return model_runner.input_batch.sampling_metadata is not (
        sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(
            req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (block_table.block_table.np[req_index, :num_blocks] ==
            req_state.block_ids[0]).all()


def _make_mock_backend_for_kernel_block_size(
    supported_sizes: list[int | MultipleOf], ):

    class _MockBackend:

        @staticmethod
        def get_supported_kernel_block_sizes():
            return supported_sizes

    return _MockBackend()


def _make_kv_cache_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=1,
                             num_kv_heads=1,
                             head_size=1,
                             dtype="float16")


def test_select_common_block_size_prefers_manager_block_size():
    backend_a = _make_mock_backend_for_kernel_block_size([MultipleOf(32)])
    backend_b = _make_mock_backend_for_kernel_block_size([64, MultipleOf(16)])
    attn_groups = [
        AttentionGroup(backend_a, [], [], _make_kv_cache_spec(), 0),
        AttentionGroup(backend_b, [], [], _make_kv_cache_spec(), 0),
    ]

    selected_size = NPUModelRunner.select_common_block_size(128, attn_groups)
    assert selected_size == 128


def test_select_common_block_size_uses_largest_shared_int():
    backend_a = _make_mock_backend_for_kernel_block_size([128, 64])
    backend_b = _make_mock_backend_for_kernel_block_size([64, 32])
    attn_groups = [
        AttentionGroup(backend_a, [], [], _make_kv_cache_spec(), 0),
        AttentionGroup(backend_b, [], [], _make_kv_cache_spec(), 0),
    ]

    selected_size = NPUModelRunner.select_common_block_size(256, attn_groups)
    assert selected_size == 64


def test_select_common_block_size_no_valid_option():
    backend_a = _make_mock_backend_for_kernel_block_size([64])
    backend_b = _make_mock_backend_for_kernel_block_size([MultipleOf(16)])
    attn_groups = [
        AttentionGroup(backend_a, [], [], _make_kv_cache_spec(), 0),
        AttentionGroup(backend_b, [], [], _make_kv_cache_spec(), 0),
    ]

    with pytest.raises(ValueError):
        NPUModelRunner.select_common_block_size(48, attn_groups)


def test_update_states_new_request(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume req
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        resumed_req_ids=set(),
        new_token_ids=[[]],
        all_token_ids={},
        new_block_ids=[([0], )],
        num_computed_tokens=[0],
        num_output_tokens=[0],
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_get_nans_in_logits(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    scheduler_output = _schedule_new_request(*req_ids)
    model_runner._update_states(scheduler_output)

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
            [4.0, float("nan"), float("nan")],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 2}

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, float("nan"), float("nan")],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 2}

    result = model_runner._get_nans_in_logits(logits=None)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 0}

    logits = torch.tensor(
        [
            [float("nan"), float("nan"), 2.0],
            [1.0, 2.0, 3.0],
            [float("nan"), 2.0, 3.0],
        ],
        device=DEVICE,
    )
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 2, "req_1": 0}


def test_update_states_no_changes(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # schedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    # new reqs
    scheduler_output = _schedule_new_request(*req_ids)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert _is_req_scheduled(model_runner, req_ids[1])

    # unschedule req_1
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_ids[0]: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert not _is_req_scheduled(model_runner, req_ids[1])


def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})


def test_reload_weights_before_load_model(model_runner):
    with pytest.raises(AssertionError):
        model_runner.reload_weights()


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order():
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist():
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current():
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_hybrid_block_table_initialization():
    """Test hybrid block table with different kernel and kvcache_manager block
    sizes."""
    from vllm.v1.worker.block_table import BlockTable

    # Test configuration: kvcache_manager block size = 32,
    # kernel block size = 16
    block_size = 32
    kernel_block_sizes = [16]
    max_num_reqs = 10
    max_num_blocks_per_req = 20
    max_num_batched_tokens = 512
    cp_kv_cache_interleave_size = 8

    block_table = BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=False,
        device=torch.device(DEVICE),
        kernel_block_size=kernel_block_sizes[0],
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )

    # Verify hybrid block configuration
    assert block_table.use_hybrid_blocks is True
    assert block_table.block_size == kernel_block_sizes[0]
    assert block_table.blocks_per_kv_block == (
        block_size // kernel_block_sizes[0])  # Changed to use first element

    # Test block table conversion logic
    # One kvcache_manager block should map to multiple kernel blocks
    kvcache_manager_blocks = [0, 1, 2]

    # Verify that kvcache_manager blocks can be converted to kernel blocks
    # and that block table operations work correctly.
    req_index = 0
    block_table.append_row(kvcache_manager_blocks, req_index)
    # Get expected kernel blocks from the implementation for verification.
    expected_kernel_blocks = block_table.map_to_kernel_blocks(
        np.array(kvcache_manager_blocks),
        block_table.blocks_per_kv_block,
        block_table._kernel_block_arange,
    )
    # Verify block table state
    assert block_table.num_blocks_per_row[req_index] == len(
        expected_kernel_blocks)
    assert np.array_equal(
        block_table.block_table.np[req_index, :len(expected_kernel_blocks)],
        expected_kernel_blocks,
    )


def test_input_batch_with_kernel_block_sizes():
    """Test InputBatch initialization with kernel_block_sizes parameter."""
    max_num_reqs = 10
    max_model_len = 512
    max_num_batched_tokens = 512
    device = torch.device(DEVICE)
    pin_memory = False
    vocab_size = 50272

    # Test with different kernel block sizes
    block_sizes = [32, 64]
    kernel_block_sizes = [16, 32]

    input_batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        device=device,
        pin_memory=pin_memory,
        vocab_size=vocab_size,
        block_sizes=block_sizes,
        kernel_block_sizes=kernel_block_sizes,
    )

    # Verify that block tables were created with kernel block sizes
    assert len(input_batch.block_table.block_tables) == len(block_sizes)

    for i, (kv_size,
            kernel_size) in enumerate(zip(block_sizes, kernel_block_sizes)):
        block_table = input_batch.block_table.block_tables[i]
        if kv_size != kernel_size:
            assert block_table.use_hybrid_blocks is True
            assert block_table.block_size == kernel_size
        else:
            assert block_table.use_hybrid_blocks is False
            assert block_table.block_size == kernel_size


def test_hybrid_cache_integration(model_runner, dist_init):
    """Test hybrid cache architecture integration with NPUModelRunner."""
    # Create a new model runner with hybrid cache configuration
    vllm_config = get_vllm_config()

    # Configure hybrid cache with different kvcache_manager block size
    vllm_config.cache_config.block_size = 32

    model_config = vllm_config.model_config
    num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    vllm_config.compilation_config.static_forward_context[
        "layer.0"] = Attention(num_heads, head_size, 0.1)

    runner = NPUModelRunner(vllm_config, DEVICE)

    # Initialize KV cache with configuration
    attn_spec = FullAttentionSpec(
        block_size=16,  # Use kernel block size directly
        num_kv_heads=runner.model_config.get_num_kv_heads(
            runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config

    # Initialize input batch with kernel block sizes
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
        kernel_block_sizes=[16],
    )  # Use kernel block size

    runner.initialize_attn_backend(kv_cache_config)

    # Verify hybrid block table configuration
    block_table = runner.input_batch.block_table.block_tables[0]
    assert block_table.block_size == (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)

    # Test request processing with hybrid blocks
    req_id = "hybrid_req_0"
    scheduler_output = _schedule_new_request(req_id)

    # Update states should work with hybrid blocks
    runner._update_states(scheduler_output)
    assert _is_req_scheduled(runner, req_id)
    assert _is_req_state_block_table_match(runner, req_id)
