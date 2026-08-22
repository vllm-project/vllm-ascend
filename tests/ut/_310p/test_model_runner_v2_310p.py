# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

import vllm_ascend._310p.worker.v2.model_runner as model_runner_module
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend._310p.worker.v2.model_state import Ascend310PModelState
from vllm_ascend._310p.worker.v2.sampler import Ascend310PSampler


def _make_vllm_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            is_hybrid=False,
            use_mla=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_config_accepts_tensor_parallelism() -> None:
    NPUModelRunner310V2._validate_config(_make_vllm_config())


@pytest.mark.parametrize(
    "setting",
    [
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ],
)
def test_config_rejects_non_tp_parallelism(setting: str) -> None:
    config = _make_vllm_config()
    setattr(config.parallel_config, setting, 2)
    with pytest.raises(NotImplementedError, match="only supports tensor parallelism"):
        NPUModelRunner310V2._validate_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("speculative_config", object(), "Speculative decoding"),
        ("kv_transfer_config", object(), "KV cache transfer"),
        ("lora_config", object(), "LoRA"),
    ],
)
def test_config_rejects_out_of_scope_features(field, value, message) -> None:
    with pytest.raises(NotImplementedError, match=message):
        NPUModelRunner310V2._validate_config(_make_vllm_config(**{field: value}))


def test_greedy_sampler_rejects_triton_sampling_features() -> None:
    sampler = Ascend310PSampler()
    sampler.add_request(0, 4, SamplingParams(temperature=0))
    with pytest.raises(NotImplementedError, match="greedy sampling only"):
        sampler.add_request(0, 4, SamplingParams(temperature=1))


def test_block_tables_use_cpu_metadata_for_gather_and_slot_mapping() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[4],
        max_num_reqs=3,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4],
        device=torch.device("cpu"),
        kernel_block_sizes=[4],
    )
    block_tables.append_block_ids(0, ([10, 11],), overwrite=True)
    block_tables.append_block_ids(1, ([20],), overwrite=True)

    gathered = block_tables.gather_block_tables(np.array([1, 0], dtype=np.int32), num_reqs_padded=3)
    torch.testing.assert_close(gathered[0][0, :2], torch.tensor([20, 0], dtype=torch.int32))
    torch.testing.assert_close(gathered[0][1, :2], torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(gathered[0][2], torch.zeros_like(gathered[0][2]))

    slots = block_tables.compute_slot_mappings(
        np.array([1, 0], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
        np.array([0, 1, 4, 5, 0, 0, 0, 0], dtype=np.int64),
        num_tokens_padded=8,
    )
    torch.testing.assert_close(
        slots,
        torch.tensor([[80, 81, 44, 45, -1, -1, -1, -1]], dtype=torch.int32),
    )


def test_block_table_expands_logical_blocks_to_310p_kernel_blocks() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[128],
        max_num_reqs=1,
        max_num_batched_tokens=2,
        max_num_blocks_per_group=[1],
        device=torch.device("cpu"),
        kernel_block_sizes=[64],
    )
    block_tables.append_block_ids(0, ([7],), overwrite=True)
    assert block_tables.block_tables_cpu[0][0, :2].tolist() == [14, 15]


def test_kv_cache_allocation_uses_separate_nz_k_and_v() -> None:
    class FakeAttentionSpec:
        block_size = 128
        storage_block_size = 128
        page_size_bytes = 4096
        num_kv_heads = 2
        head_size = 128
        head_size_v = 128
        dtype = torch.float16

    class FakeBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_type):
            del cache_type
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size, 16)

    spec = FakeAttentionSpec()
    kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["model.layers.0.self_attn"])],
        kv_cache_tensors=[SimpleNamespace(size=8192, shared_by=["model.layers.0.self_attn"])],
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.kernel_block_sizes = [64]
    runner.attn_groups = [[SimpleNamespace(backend=FakeBackend, layer_names=["model.layers.0.self_attn"])]]

    allocations = []

    def empty_with_format(*, size, dtype, device, acl_format):
        allocations.append((size, dtype, device, acl_format))
        return torch.zeros(size, dtype=dtype, device=device)

    with (
        patch.object(model_runner_module, "AttentionSpec", FakeAttentionSpec),
        patch.object(model_runner_module, "AscendAttentionBackend310", FakeBackend),
        patch.object(model_runner_module.torch_npu, "empty_with_format", empty_with_format),
    ):
        caches = runner._allocate_kv_cache_tensors(kv_cache_config, {})

    k_cache, v_cache = caches["model.layers.0.self_attn"]
    assert k_cache.data_ptr() != v_cache.data_ptr()
    assert len(allocations) == 2
    assert all(allocation[3] == model_runner_module.ACL_FORMAT_FRACTAL_NZ for allocation in allocations)


def test_model_state_uses_greedy_sampler() -> None:
    model_state = object.__new__(Ascend310PModelState)
    model_state.rope_state = None

    model_inputs = model_state.prepare_inputs(SimpleNamespace(), req_states=None)
    sampler, speculator = model_state.custom_sampler(object())

    assert model_inputs == {}
    assert isinstance(sampler, Ascend310PSampler)
    assert speculator is None


def test_aclgraph_query_lens_ignore_padded_request_entries() -> None:
    query_lens = NPUModelRunner310V2._get_valid_query_lens(
        torch.tensor([3, 7, -1, -1], dtype=torch.int32),
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )

    torch.testing.assert_close(query_lens, torch.tensor([2, 3], dtype=torch.int32))


def test_worker_selects_v2_runner_on_310p() -> None:
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.vllm_config = SimpleNamespace()
    worker.use_v2_model_runner = True
    worker.device = torch.device("cpu")
    with patch("vllm_ascend._310p.worker.v2.model_runner.NPUModelRunner310V2") as runner_cls:
        worker.model_runner = worker._create_model_runner()
    runner_cls.assert_called_once_with(worker.vllm_config, worker.device)
