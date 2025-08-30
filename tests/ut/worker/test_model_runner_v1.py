# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import tempfile

import pytest
import torch
from vllm.attention import Attention
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.attention.attention_v1 import (AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import InputBatch

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.utils import AscendSocVersion
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE = 'cpu'

# yapf: disable
@pytest.mark.parametrize(
    "soc_version, enable_expert_parallel, world_size, num_tokens, mc2_tokens_capacity, expected_method",
    [
        # Case 1: Expert parallel is disabled, should always be 'allgather'
        (AscendSocVersion.A2, False, 8, 100, 256, "allgather"),
        (AscendSocVersion.A3, False, 16, 500, 256, "allgather"),

        # Case 2: A2 SOC
        # 2.1: MC2 conditions met (tokens <= capacity, world_size >= 16)
        (AscendSocVersion.A2, True, 16, 100, 256, "mc2"),
        (AscendSocVersion.A2, True, 32, 256, 256, "mc2"),
        # 2.2: MC2 token capacity exceeded
        (AscendSocVersion.A2, True, 16, 257, 256, "allgather"),
        # 2.3: MC2 world size not met
        (AscendSocVersion.A2, True, 8, 100, 256, "allgather"),
        (AscendSocVersion.A2, True, 15, 100, 256, "allgather"),

        # Case 3: A3 SOC
        # 3.1: MC2 condition met (tokens <= capacity)
        (AscendSocVersion.A3, True, 8, 100, 256, "mc2"),
        (AscendSocVersion.A3, True, 16, 256, 256, "mc2"),
        # 3.2: MC2 token capacity exceeded
        (AscendSocVersion.A3, True, 8, 257, 256, "alltoall"),
        (AscendSocVersion.A3, True, 16, 500, 256, "alltoall"),

    ])
# yapf: enable
def test_select_moe_comm_method(soc_version, enable_expert_parallel,
                                world_size, num_tokens, mc2_tokens_capacity,
                                expected_method):
    """
    Tests the _select_moe_comm_method with various configurations.
    """
    # Mock the NPUModelRunner instance and its dependencies
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = enable_expert_parallel
    mock_runner.parallel_config.world_size = world_size
    mock_runner.mc2_tokens_capacity = mc2_tokens_capacity

    # Patch the helper functions
    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=soc_version), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True):

        # Call the method under test
        method = NPUModelRunner._select_moe_comm_method(
            mock_runner, num_tokens)

        # Assert the result
        assert method == expected_method


def test_select_moe_comm_method_unsupported_soc():
    """
    Tests that _select_moe_comm_method raises ValueError for an unsupported SOC.
    """
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.parallel_config = MagicMock()
    mock_runner.parallel_config.enable_expert_parallel = True
    mock_runner.mc2_tokens_capacity = 256

    unsupported_soc = "UnsupportedSOC"

    with patch('vllm_ascend.worker.model_runner_v1.get_ascend_soc_version',
               return_value=unsupported_soc), \
         patch('vllm_ascend.worker.model_runner_v1.is_global_first_rank',
               return_value=True), \
         pytest.raises(ValueError, match=f"Unsupported soc_version: {unsupported_soc}"):

        NPUModelRunner._select_moe_comm_method(mock_runner, 100)


def get_vllm_config():
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
        enforce_eager=True,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    device_config = DeviceConfig(device=DEVICE, )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=device_config,
    )
    return vllm_config


def initialize_kv_cache(runner: NPUModelRunner):
    """
    Only perform necessary steps in GPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(
            runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
        use_mla=False,
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
    )
    runner.initialize_attn_backend(kv_cache_config)


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


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_kwargs=[],
                mm_hashes=[],
                mm_positions=[],
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
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


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
    return (block_table.block_table_np[req_index, :num_blocks] ==
            req_state.block_ids[0]).all()


def _is_attn_metadata_match(attn_metadata: AscendMetadata):
    assert attn_metadata.attn_mask.shape == (3, 3)
    assert attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
    assert attn_metadata.max_query_len == 3
    assert attn_metadata.num_actual_tokens == 3
    assert attn_metadata.query_lens.item() == 3
    assert all(attn_metadata.query_start_loc.to('cpu') == torch.tensor([0, 3]))
    assert attn_metadata.seq_lens.item() == 3
    assert all(attn_metadata.slot_mapping.to('cpu') == torch.tensor([0, 1, 2]))


def test_prepare_inputs(model_runner, dist_init):
    req_id = 'req_0'
    scheduler_output = _schedule_new_request(req_id)
    model_runner.load_model()
    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)

    (attn_metadata, positions, num_scheduled_tokens, num_input_tokens,
     num_tokens_across_dp, maybe_padded_num_tokens, logits_indices,
     spec_decode_metadata, input_ids, inputs_embeds,
     intermediate_tensors) = model_runner._prepare_inputs(scheduler_output)

    _is_attn_metadata_match(attn_metadata)
    assert all(input_ids.to('cpu') == torch.tensor([1, 2, 3]))
    assert all(positions.to('cpu') == torch.tensor([0, 1, 2]))

