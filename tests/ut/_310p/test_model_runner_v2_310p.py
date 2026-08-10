# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.triton_dispatcher import _get_kernel_impl
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend._310p.worker.v2.model_state import Ascend310PModelState
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler
from vllm_ascend.patch.platform import patch_use_v2_model_runner
from vllm_ascend.worker.v2.model_states.default import AscendModelState


def test_310p_slot_mapping_kernel_is_registered() -> None:
    kernel_name = "vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel"

    assert _get_kernel_impl(kernel_name) is not None


def test_310p_v2_config_validation_skips_upstream_triton_gate() -> None:
    config = SimpleNamespace(
        _get_v2_model_runner_unsupported_features=lambda: [],
        reasoning_config=None,
    )

    with (
        patch.object(patch_use_v2_model_runner, "is_310p", return_value=True),
        patch.object(
            patch_use_v2_model_runner,
            "_ORIGINAL_VALIDATE_V2_MODEL_RUNNER",
            side_effect=AssertionError("upstream Triton validation must not run"),
        ),
    ):
        patch_use_v2_model_runner._patched_validate_v2_model_runner(config)


def test_310p_v2_config_validation_keeps_unsupported_feature_gate() -> None:
    config = SimpleNamespace(
        _get_v2_model_runner_unsupported_features=lambda: ["unsupported feature"],
        reasoning_config=None,
    )

    with (
        patch.object(patch_use_v2_model_runner, "is_310p", return_value=True),
        pytest.raises(ValueError, match="unsupported feature"),
    ):
        patch_use_v2_model_runner._patched_validate_v2_model_runner(config)


def test_non_310p_v2_config_validation_uses_upstream_gate() -> None:
    config = MagicMock()
    original_validate = MagicMock()

    with (
        patch.object(patch_use_v2_model_runner, "is_310p", return_value=False),
        patch.object(
            patch_use_v2_model_runner,
            "_ORIGINAL_VALIDATE_V2_MODEL_RUNNER",
            original_validate,
        ),
    ):
        patch_use_v2_model_runner._patched_validate_v2_model_runner(config)

    original_validate.assert_called_once_with(config)


def _make_vllm_config(**parallel_overrides):
    parallel = {
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        lora_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(enable_sleep_mode=False),
    )


def test_first_release_config_accepts_tensor_parallelism() -> None:
    config = _make_vllm_config()
    config.parallel_config.tensor_parallel_size = 2

    NPUModelRunner310V2._validate_first_release_config(config)


@pytest.mark.parametrize(
    "setting",
    [
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ],
)
def test_first_release_config_rejects_non_tp_parallelism(setting: str) -> None:
    config = _make_vllm_config(**{setting: 2})

    with pytest.raises(NotImplementedError, match="only supports tensor parallelism"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_speculative_decoding() -> None:
    config = _make_vllm_config()
    config.speculative_config = SimpleNamespace(method="mtp")

    with pytest.raises(NotImplementedError, match="deferred to the second"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_prefix_caching() -> None:
    config = _make_vllm_config()
    config.cache_config.enable_prefix_caching = True

    with pytest.raises(NotImplementedError, match="deferred to the second"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_expert_parallelism() -> None:
    config = _make_vllm_config(enable_expert_parallel=True)

    with pytest.raises(NotImplementedError, match="Expert parallelism"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_uniform_decode_query_len_falls_back_to_decode_query_len() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    runner.decode_query_len = 1

    assert runner._get_uniform_decode_query_len() == 1

    runner.uniform_decode_query_len = 2
    assert runner._get_uniform_decode_query_len() == 2


def test_postprocess_query_lens_ignore_full_graph_request_padding() -> None:
    # Fifteen real decode requests replay a graph captured for a 16-request
    # bucket. query_start_loc has real request boundaries only, while the graph
    # idx_mapping carries one trailing -1 sentinel.
    idx_mapping = torch.tensor([*range(15), -1], dtype=torch.int32)
    query_start_loc = torch.arange(16, dtype=torch.int32)

    query_lens = NPUModelRunner310V2._get_valid_query_lens(idx_mapping, query_start_loc)

    torch.testing.assert_close(query_lens, torch.ones(15, dtype=torch.int32))


def test_postprocess_query_lens_keep_exact_graph_bucket() -> None:
    idx_mapping = torch.arange(4, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 3, 4, 7], dtype=torch.int32)

    query_lens = NPUModelRunner310V2._get_valid_query_lens(idx_mapping, query_start_loc)

    torch.testing.assert_close(query_lens, torch.tensor([1, 2, 1, 3], dtype=torch.int32))


def test_310p_model_state_refreshes_all_full_graph_seq_lens_buffers() -> None:
    model_state = object.__new__(Ascend310PModelState)
    model_state._capture_seq_lens_by_ptr = {}
    shared_capture_buffer = torch.full((4,), -1, dtype=torch.int32)
    second_capture_buffer = torch.full((3,), -1, dtype=torch.int32)

    # The same address may be seen through different bucket-sized views. Keep
    # the largest one even if capture order changes in a future vLLM version.
    model_state._record_capture_seq_lens(shared_capture_buffer[:2])
    model_state._record_capture_seq_lens(shared_capture_buffer)
    model_state._record_capture_seq_lens(second_capture_buffer)

    runtime_seq_lens = torch.tensor([17, 9], dtype=torch.int32)
    model_state._refresh_capture_seq_lens(runtime_seq_lens)

    torch.testing.assert_close(shared_capture_buffer, torch.tensor([17, 9, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(second_capture_buffer, torch.tensor([17, 9, 0], dtype=torch.int32))


def test_310p_model_state_only_refreshes_seq_lens_for_full_runtime() -> None:
    model_state = object.__new__(Ascend310PModelState)
    capture_seq_lens = torch.full((2,), -1, dtype=torch.int32)
    model_state._capture_seq_lens_by_ptr = {}
    capture_batch = SimpleNamespace(seq_lens=capture_seq_lens)
    input_batch = SimpleNamespace(seq_lens=torch.tensor([11, 12], dtype=torch.int32))

    with patch.object(AscendModelState, "prepare_attn", return_value={}) as prepare_attn:
        model_state.prepare_attn(
            capture_batch,
            CUDAGraphMode.NONE,
            (),
            MagicMock(),
            [],
            MagicMock(),
            for_capture=True,
        )
        model_state.prepare_attn(input_batch, CUDAGraphMode.NONE, (), MagicMock(), [], MagicMock())
        torch.testing.assert_close(capture_seq_lens, torch.full((2,), -1, dtype=torch.int32))

        model_state.prepare_attn(input_batch, CUDAGraphMode.PIECEWISE, (), MagicMock(), [], MagicMock())
        torch.testing.assert_close(capture_seq_lens, torch.full((2,), -1, dtype=torch.int32))

        model_state.prepare_attn(input_batch, CUDAGraphMode.FULL, (), MagicMock(), [], MagicMock())
        torch.testing.assert_close(capture_seq_lens, input_batch.seq_lens)

    assert prepare_attn.call_count == 4


def test_v2_allocates_attention_kv_cache_directly_as_nz() -> None:
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )
    num_blocks = 2
    layer_name = "model.layers.0.self_attn.attn"
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[layer_name])],
        kv_cache_tensors=[
            SimpleNamespace(
                size=num_blocks * spec.page_size_bytes,
                shared_by=[layer_name],
            )
        ],
        num_blocks=num_blocks,
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.attn_groups = [[SimpleNamespace(backend=AscendAttentionBackend310, layer_names=[layer_name])]]
    runner.kernel_block_sizes = [128]
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.device = torch.device("cpu")
    k_cache = MagicMock()
    v_cache = MagicMock()

    with patch("torch_npu.empty_with_format", side_effect=[k_cache, v_cache]) as empty_with_format:
        caches = runner._allocate_kv_cache_tensors_310p(kv_cache_config, {})

    expected_shape = (num_blocks, 64, 128, 16)
    assert caches[layer_name] == (k_cache, v_cache)
    assert empty_with_format.call_count == 2
    empty_with_format.assert_any_call(
        size=expected_shape,
        dtype=torch.float16,
        device=torch.device("cpu"),
        acl_format=29,
    )


def test_first_release_sampler_accepts_only_greedy() -> None:
    sampler = Ascend310PGreedySampler()
    sampler.add_request(0, 4, SamplingParams(temperature=0))

    with pytest.raises(NotImplementedError, match="only supports greedy"):
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
    block_tables.apply_staged_writes()

    gathered = block_tables.gather_block_tables(np.array([1, 0], dtype=np.int32), num_reqs_padded=3)
    torch.testing.assert_close(gathered[0][0, :2], torch.tensor([20, 0], dtype=torch.int32))
    torch.testing.assert_close(gathered[0][1, :2], torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(gathered[0][2], torch.zeros_like(gathered[0][2]))

    slots = block_tables.compute_slot_mappings(
        np.array([1, 0], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
        np.array([0, 1, 4, 5], dtype=np.int64),
        num_tokens_padded=8,
    )
    expected = torch.tensor([[80, 81, 44, 45, -1, -1, -1, -1]], dtype=torch.int32)
    torch.testing.assert_close(slots, expected)


def test_block_tables_reject_device_metadata() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[4],
        max_num_reqs=1,
        max_num_batched_tokens=4,
        max_num_blocks_per_group=[1],
        device=torch.device("cpu"),
        kernel_block_sizes=[4],
    )

    with pytest.raises(TypeError, match="CPU request-state mirror"):
        block_tables.gather_block_tables(torch.empty(1, device="meta", dtype=torch.int32), 1)


def test_worker_selects_v2_runner_when_enabled() -> None:
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.use_v2_model_runner = True
    worker.vllm_config = MagicMock()
    worker.device = torch.device("cpu")

    with patch(
        "vllm_ascend._310p.worker.v2.model_runner.NPUModelRunner310V2",
        return_value=MagicMock(),
    ) as runner_cls:
        worker._create_model_runner()

    runner_cls.assert_called_once_with(worker.vllm_config, worker.device)


def test_worker_keeps_v1_runner_when_v2_is_disabled() -> None:
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.use_v2_model_runner = False
    worker.vllm_config = MagicMock()
    worker.device = torch.device("cpu")

    with patch(
        "vllm_ascend._310p.worker_310p.NPUModelRunner310",
        return_value=MagicMock(),
    ) as runner_cls:
        worker._create_model_runner()

    runner_cls.assert_called_once_with(worker.vllm_config, worker.device)
