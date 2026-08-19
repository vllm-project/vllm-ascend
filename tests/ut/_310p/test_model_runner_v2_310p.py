# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
from vllm_ascend._310p.worker.v2 import block_table as block_table_module
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.feature_support import MRv2FeatureSupport
from vllm_ascend._310p.worker.v2.kernel_registry import (
    KERNEL_IMPLS,
    register_310p_kernels,
)
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend._310p.worker.v2.model_state import (
    Ascend310PMambaHybridModelState,
    Ascend310PModelState,
)
from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler
from vllm_ascend.patch.platform import patch_use_v2_model_runner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


def test_310p_v2_has_no_required_kernel_dispatcher_registration() -> None:
    assert KERNEL_IMPLS == {}
    assert register_310p_kernels() == ()


def test_310p_block_tables_do_not_import_triton_or_shared_v2_kernel() -> None:
    tree = ast.parse(inspect.getsource(block_table_module))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any("triton" in module for module in imported_modules)
    assert "vllm_ascend.worker.v2.block_table" not in imported_modules


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


def test_first_release_config_accepts_async_scheduling() -> None:
    config = _make_vllm_config()
    config.scheduler_config.async_scheduling = True

    NPUModelRunner310V2._validate_first_release_config(config)


def test_310p_hybrid_model_uses_dedicated_model_state() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True))
    model = MagicMock()
    encoder_cache = MagicMock()
    device = torch.device("cpu")
    expected = object()

    with (
        patch("vllm_ascend.worker.v2.model_states.is_310p", return_value=True),
        patch(
            "vllm_ascend._310p.worker.v2.model_state.Ascend310PMambaHybridModelState",
            return_value=expected,
        ) as model_state_cls,
    ):
        result = init_asecnd_model_state(config, model, encoder_cache, device)

    assert result is expected
    model_state_cls.assert_called_once_with(config, model, encoder_cache, device)


def test_310p_hybrid_model_state_keeps_ascend_hybrid_behavior() -> None:
    assert issubclass(Ascend310PMambaHybridModelState, AscendMambaHybridModelState)


def test_310p_hybrid_model_state_initializes_full_upstream_contract() -> None:
    state = object.__new__(Ascend310PMambaHybridModelState)
    config = MagicMock()
    model = MagicMock()
    encoder_cache = MagicMock()
    device = torch.device("cpu")

    with (
        patch.object(AscendMambaHybridModelState, "__init__") as parent_init,
        patch.object(state, "_replace_310p_rope_state") as replace_rope_state,
    ):
        Ascend310PMambaHybridModelState.__init__(state, config, model, encoder_cache, device)

    parent_init.assert_called_once_with(state, config, model, encoder_cache, device)
    replace_rope_state.assert_called_once_with(encoder_cache)
    assert state._capture_seq_lens_by_ptr == {}


def test_310p_mrope_state_prepares_cos_sin_before_model_forward() -> None:
    model_state = object.__new__(Ascend310PMambaHybridModelState)
    model_state.model_config = SimpleNamespace(uses_mrope=True)
    model_state.rope_state = MagicMock(spec=Ascend310PRopeState)
    positions = torch.zeros((3, 4), dtype=torch.int64)
    model_state.rope_state.get_positions.return_value = positions
    input_batch = SimpleNamespace(
        idx_mapping_np=np.array([0]),
        query_start_loc_np=np.array([0, 4]),
        num_tokens_after_padding=4,
    )
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(np=np.array([4])),
        num_computed_tokens_np=np.array([0]),
    )

    with patch("vllm_ascend._310p.worker.v2.model_state.prepare_mrope_cos_sin_slices_from_runner") as prepare_slices:
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    model_state.rope_state.prepare_positions_cpu.assert_called_once()
    prepare_slices.assert_called_once_with(model_state, positions)
    assert model_inputs["positions"] is positions


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

    with pytest.raises(NotImplementedError, match="Qwen3.5 MTP/speculative decoding"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_prefix_caching() -> None:
    config = _make_vllm_config()
    config.cache_config.enable_prefix_caching = True

    with pytest.raises(NotImplementedError, match="deferred to a later"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_future_feature_support_is_an_explicit_runner_extension_point() -> None:
    class FutureNPUModelRunner310V2(NPUModelRunner310V2):
        feature_support = MRv2FeatureSupport(prefix_caching=True, qwen3_5_mtp=True)

    config = _make_vllm_config()
    config.cache_config.enable_prefix_caching = True
    config.speculative_config = SimpleNamespace(method="mtp")

    FutureNPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_capability_properties_do_not_advertise_future_features() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    assert runner.supports_prefix_caching is False
    assert runner.supports_qwen3_5_mtp is False
    assert runner.supports_mtp is False


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


def test_310p_v2_restores_missing_linear_attention_kv_cache_specs() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    existing_spec = object()
    linear_spec = object()
    linear_layer = MagicMock()
    linear_layer.get_kv_cache_spec.return_value = linear_spec
    ignored_layer = MagicMock()
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "language_model.model.layers.0.linear_attn": linear_layer,
            "language_model.model.layers.3.self_attn.attn": ignored_layer,
        }
    )
    runner.vllm_config = MagicMock()

    with patch.object(
        NPUModelRunner,
        "get_kv_cache_spec",
        return_value={"language_model.model.layers.3.self_attn.attn": existing_spec},
    ):
        specs = runner.get_kv_cache_spec()

    assert specs["language_model.model.layers.0.linear_attn"] is linear_spec
    assert specs["language_model.model.layers.3.self_attn.attn"] is existing_spec
    linear_layer.get_kv_cache_spec.assert_called_once_with(runner.vllm_config)
    ignored_layer.get_kv_cache_spec.assert_not_called()


@pytest.mark.parametrize("needs_zeroing", [False, True])
def test_310p_v2_initializes_kv_zeroer_when_required(needs_zeroing: bool) -> None:
    runner = object.__new__(NPUModelRunner310V2)
    kv_cache_config = SimpleNamespace(needs_kv_cache_zeroing=needs_zeroing)

    with patch.object(runner, "_init_kv_zero_meta") as init_zero_meta:
        runner._init_kv_zero_meta_if_needed(kv_cache_config)

    assert init_zero_meta.call_count == int(needs_zeroing)


def test_310p_v2_kv_zeroer_uses_v2_pin_memory_capability() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.attn_groups = []
    runner.kernel_block_sizes = []
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    zeroer = MagicMock()

    with (
        patch(
            "vllm_ascend._310p.worker.v2.model_runner.is_pin_memory_available",
            return_value=True,
        ) as pin_memory_available,
        patch(
            "vllm_ascend._310p.worker.v2.model_runner.AscendKVBlockZeroer310V2",
            return_value=zeroer,
        ) as zeroer_cls,
    ):
        runner._init_kv_zero_meta()

    pin_memory_available.assert_called_once_with()
    zeroer_cls.assert_called_once_with(runner.device, True)
    zeroer.init_meta.assert_called_once()


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


def test_v2_selects_64_kernel_block_for_256_head_size() -> None:
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.float16,
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.attn_groups = [
        [SimpleNamespace(backend=SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64]))]
    ]
    runner.kernel_block_sizes = [128]
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])

    runner._adjust_kernel_block_sizes_310p(kv_cache_config)

    assert runner.kernel_block_sizes == [64]


def test_v2_separates_attention_and_mamba_shared_cache_slot() -> None:
    attention_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=128,
        shapes=((16,),),
        dtypes=(torch.float16,),
        page_size_padded=attention_spec.page_size_bytes,
        mamba_cache_mode="align",
    )
    num_blocks = 2
    attention_layer = "language_model.model.layers.3.self_attn.attn"
    mamba_layer = "language_model.model.layers.0.linear_attn"
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=attention_spec, layer_names=[attention_layer]),
            SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=[mamba_layer]),
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                size=num_blocks * attention_spec.page_size_bytes,
                shared_by=[attention_layer, mamba_layer],
            )
        ],
        num_blocks=num_blocks,
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.attn_groups = [[SimpleNamespace(backend=AscendAttentionBackend310, layer_names=[attention_layer])]]
    runner.kernel_block_sizes = [128, 0]
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.device = torch.device("cpu")
    k_cache = MagicMock()
    v_cache = MagicMock()

    with patch("torch_npu.empty_with_format", side_effect=[k_cache, v_cache]):
        caches = runner._allocate_kv_cache_tensors_310p(kv_cache_config, {})

    assert caches[attention_layer] == (k_cache, v_cache)
    assert isinstance(caches[mamba_layer], list)
    assert caches[mamba_layer][0].layout == torch.strided
    assert caches[mamba_layer] is not caches[attention_layer]


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


def test_block_tables_compute_slot_mappings_per_cache_group() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[4, 8],
        max_num_reqs=2,
        max_num_batched_tokens=4,
        max_num_blocks_per_group=[4, 2],
        device=torch.device("cpu"),
        kernel_block_sizes=[4, 8],
    )
    block_tables.append_block_ids(0, ([10, 11], [20]), overwrite=True)

    slots = block_tables.compute_slot_mappings(
        np.array([0], dtype=np.int32),
        np.array([0, 3], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int64),
        num_tokens_padded=4,
    )

    expected = torch.tensor(
        [
            [43, 44, 45, -1],
            [163, 164, 165, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(slots, expected)


def test_block_tables_pad_slot_mappings_when_no_tokens_are_scheduled() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[4],
        max_num_reqs=1,
        max_num_batched_tokens=4,
        max_num_blocks_per_group=[1],
        device=torch.device("cpu"),
        kernel_block_sizes=[4],
    )
    block_tables.append_block_ids(0, ([7],), overwrite=True)
    block_tables.compute_slot_mappings(
        np.array([0], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        np.array([0, 1], dtype=np.int64),
        num_tokens_padded=4,
    )

    slots = block_tables.compute_slot_mappings(
        np.array([], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([], dtype=np.int64),
        num_tokens_padded=4,
    )

    torch.testing.assert_close(slots, torch.full((1, 4), -1, dtype=torch.int32))


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
