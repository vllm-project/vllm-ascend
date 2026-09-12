# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams

import vllm_ascend._310p.worker.v2.model_runner as model_runner_module
from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend._310p.worker.v2.model_state import (
    Ascend310PMambaHybridModelState,
    Ascend310PModelState,
)
from vllm_ascend._310p.worker.v2.sampler import Ascend310PSampler
from vllm_ascend._310p.worker.v2.states import Ascend310PStagedWriteTensor
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


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


def test_config_accepts_prefix_caching() -> None:
    """310P MRv2 supports APC; gate must not reject enable_prefix_caching."""
    config = _make_vllm_config()
    config.cache_config.enable_prefix_caching = True
    NPUModelRunner310V2._validate_config(config)


def test_config_accepts_tensor_parallelism() -> None:
    NPUModelRunner310V2._validate_config(_make_vllm_config())


@pytest.mark.parametrize(
    ("has_mamba", "uses_eagle_block_drop", "num_spec_tokens", "expected"),
    [
        (True, True, 2, True),
        (True, True, 1, False),
        (True, False, 2, False),
        (False, True, 2, False),
        (True, True, 0, False),
    ],
)
def test_kv_zeroing_uses_narrow_310p_gate(
    has_mamba: bool,
    uses_eagle_block_drop: bool,
    num_spec_tokens: int,
    expected: bool,
) -> None:
    runner = object.__new__(NPUModelRunner310V2)
    runner.speculative_config = (
        SimpleNamespace(
            num_speculative_tokens=num_spec_tokens,
            use_eagle_block_drop=lambda: uses_eagle_block_drop,
        )
        if num_spec_tokens
        else None
    )
    kv_cache_config = SimpleNamespace(
        has_mamba_layers=has_mamba,
        kv_cache_groups=[SimpleNamespace(is_eagle_group=uses_eagle_block_drop)],
    )

    with patch.object(model_runner_module, "vllm_version_is", return_value=False):
        assert runner._needs_kv_cache_zeroing_310p(kv_cache_config) is expected


def test_kv_zeroing_matches_v028_gate() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    kv_cache_config = SimpleNamespace(
        has_mamba_layers=True,
        kv_cache_groups=[SimpleNamespace(is_eagle_group=True)],
    )

    with patch.object(model_runner_module, "vllm_version_is", return_value=True):
        assert runner._needs_kv_cache_zeroing_310p(kv_cache_config)


def test_update_requests_filters_unneeded_upstream_zeroing() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    runner.speculative_config = None
    runner.kv_cache_config = SimpleNamespace(
        has_mamba_layers=True,
        kv_cache_groups=[SimpleNamespace(is_eagle_group=False)],
    )
    scheduler_output = SimpleNamespace(
        kv_cache_block_copies=None,
        new_block_ids_to_zero=[1, 2, 3],
    )

    with patch.object(NPUModelRunner, "update_requests") as update_requests:
        runner.update_requests(scheduler_output)

    assert scheduler_output.new_block_ids_to_zero is None
    update_requests.assert_called_once_with(scheduler_output)


def test_config_accepts_qwen3_vl_multimodal_mrope() -> None:
    """Qwen3-VL is multimodal + MRoPE; 310P MRv2 must allow it."""
    config = _make_vllm_config()
    config.model_config.is_multimodal_model = True
    config.model_config.uses_mrope = True
    NPUModelRunner310V2._validate_config(config)


def test_config_accepts_qwen35_hybrid() -> None:
    """Qwen3.5 is hybrid + multimodal + MRoPE; 310P MRv2 must allow it."""
    config = _make_vllm_config()
    config.model_config.is_hybrid = True
    config.model_config.is_multimodal_model = True
    config.model_config.uses_mrope = True
    NPUModelRunner310V2._validate_config(config)


def test_310p_hybrid_model_state_keeps_ascend_hybrid_behavior() -> None:
    assert issubclass(Ascend310PMambaHybridModelState, AscendMambaHybridModelState)


def test_310p_v2_does_not_advertise_shared_kv_backing() -> None:
    assert NPUModelRunner310V2.supports_standardized_shared_kv_backing is False


def test_310p_hybrid_postprocess_filters_padding_indices() -> None:
    state = object.__new__(Ascend310PMambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.zeros(4, dtype=torch.int32)
    state._num_accepted_tokens_cpu = np.zeros(4, dtype=np.int32)
    state._align_mode = False
    state.recoverssm = None
    idx_mapping = torch.tensor([0, -1, 2], dtype=torch.int32)

    state.postprocess_state(idx_mapping, num_sampled=3)
    np.testing.assert_array_equal(state._num_accepted_tokens_cpu, np.array([3, 0, 3, 0], dtype=np.int32))

    num_sampled = torch.tensor([2, 9, 4], dtype=torch.int32)
    state.postprocess_state(idx_mapping, num_sampled=num_sampled)
    np.testing.assert_array_equal(state._num_accepted_tokens_cpu, np.array([2, 0, 4, 0], dtype=np.int32))


def test_310p_hybrid_model_state_initializes_full_upstream_contract() -> None:
    state = object.__new__(Ascend310PMambaHybridModelState)
    state.max_num_reqs = 4
    state._align_mode = False
    config = object()
    model = object()
    encoder_cache = object()
    device = torch.device("cpu")
    with (
        patch.object(AscendMambaHybridModelState, "__init__") as parent_init,
        patch.object(Ascend310PMambaHybridModelState, "_replace_310p_rope_state") as replace_rope,
        patch("vllm_ascend._310p.worker.v2.model_state.vllm_version_is", return_value=True),
    ):
        Ascend310PMambaHybridModelState.__init__(state, config, model, encoder_cache, device)
    parent_init.assert_called_once_with(state, config, model, encoder_cache, device)
    replace_rope.assert_called_once_with(encoder_cache)
    assert state.recoverssm is None
    assert isinstance(state._capture_seq_lens_by_ptr, dict)


def test_init_model_state_routes_qwen35_hybrid_to_310p() -> None:
    """Qwen3.5 is_hybrid must select Ascend310PMambaHybridModelState on 310P."""
    from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

    vllm_config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=True))
    model = MagicMock(spec=["forward"])  # no get_model_state_cls
    encoder_cache = object()
    device = torch.device("cpu")
    expected = object()

    with (
        patch("vllm_ascend.worker.v2.model_states.is_310p", return_value=True),
        patch(
            "vllm_ascend._310p.worker.v2.model_state.Ascend310PMambaHybridModelState",
            return_value=expected,
        ) as hybrid_cls,
    ):
        state = init_asecnd_model_state(vllm_config, model, encoder_cache, device)

    assert state is expected
    hybrid_cls.assert_called_once_with(vllm_config, model, encoder_cache, device)


def test_get_kv_cache_spec_restores_qwen35_linear_attn() -> None:
    """Qwen3.5 GDN layers may be omitted by upstream V2; 310P restores them."""
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.vllm_config = object()
    restored = object()
    linear_layer = SimpleNamespace(get_kv_cache_spec=lambda _cfg: restored)
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "model.layers.0.self_attn": object(),
            "model.layers.1.linear_attn": linear_layer,
            "model.layers.2.linear_attn": SimpleNamespace(get_kv_cache_spec=lambda _cfg: None),
        }
    )

    with patch.object(
        NPUModelRunner,
        "get_kv_cache_spec",
        return_value={"model.layers.0.self_attn": object()},
    ):
        specs = runner.get_kv_cache_spec()

    assert "model.layers.1.linear_attn" in specs
    assert specs["model.layers.1.linear_attn"] is restored
    assert "model.layers.2.linear_attn" not in specs


def test_kv_cache_allocation_qwen35_mamba_stays_nd() -> None:
    """Qwen3.5 hybrid Mamba/GDN state must stay ND (not FRACTAL_NZ)."""

    class FakeMambaSpec:
        block_size = 1
        page_size_bytes = 80
        shapes = [(4, 8), (2, 4)]
        dtypes = [torch.float16, torch.float16]

    spec = FakeMambaSpec()
    layer_name = "model.layers.1.linear_attn"
    kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[layer_name])],
        kv_cache_tensors=[
            SimpleNamespace(
                size=160,
                # Both supported versions use the #51718 layer descriptor.
                layers=[layer_name],
            )
        ],
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.kernel_block_sizes = [1]
    runner.attn_groups = [[SimpleNamespace(backend=object, layer_names=[layer_name])]]

    with patch.object(model_runner_module, "MambaSpec", FakeMambaSpec):
        caches = runner._allocate_kv_cache_tensors(kv_cache_config, {})

    states = caches[layer_name]
    assert isinstance(states, list)
    assert len(states) == 2
    assert states[0].shape == (2, 4, 8)
    assert states[1].shape == (2, 2, 4)
    assert states[0].dtype == torch.float16
    assert states[0].untyped_storage().nbytes() == 160


def test_main_mamba_descriptor_allocates_private_per_layer_pages() -> None:
    class FakeMambaSpec:
        block_size = 1
        page_size_bytes = 80
        shapes = [(4, 8), (2, 4)]
        dtypes = [torch.float16, torch.float16]

    spec = FakeMambaSpec()
    layer_names = [
        "model.layers.1.linear_attn",
        "model.layers.3.linear_attn",
    ]
    kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=layer_names,
            )
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                size=4096,
                layers=layer_names,
            )
        ],
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.kernel_block_sizes = [1]
    runner.attn_groups = [[SimpleNamespace(backend=object, layer_names=layer_names)]]

    with patch.object(model_runner_module, "MambaSpec", FakeMambaSpec):
        caches = runner._allocate_kv_cache_tensors(kv_cache_config, {})

    first_states = caches[layer_names[0]]
    second_states = caches[layer_names[1]]
    assert first_states[0].shape[0] == kv_cache_config.num_blocks
    assert second_states[0].shape[0] == kv_cache_config.num_blocks
    assert first_states[0].untyped_storage().data_ptr() != second_states[0].untyped_storage().data_ptr()
    assert first_states[0].untyped_storage().nbytes() == 160
    assert second_states[0].untyped_storage().nbytes() == 160


def test_runner_installs_310p_request_state() -> None:
    request_state = object()

    def init_common_runner(runner, vllm_config, device) -> None:
        del vllm_config
        runner.max_num_reqs = 4
        runner.max_model_len = 128
        runner.max_num_tokens = 32
        runner.num_speculative_steps = 0
        runner.vocab_size = 1024
        runner.device = device
        runner.input_buffers = SimpleNamespace()

    with (
        patch.object(
            NPUModelRunner,
            "__init__",
            autospec=True,
            side_effect=init_common_runner,
        ),
        patch.object(
            model_runner_module,
            "Ascend310PRequestState",
            return_value=request_state,
        ) as request_state_cls,
        patch.object(model_runner_module, "is_pin_memory_available", return_value=False),
    ):
        runner = NPUModelRunner310V2(_make_vllm_config(), torch.device("cpu"))

    assert runner.req_states is request_state
    request_state_cls.assert_called_once_with(
        max_num_reqs=4,
        max_model_len=128,
        max_num_batched_tokens=32,
        num_speculative_steps=0,
        vocab_size=1024,
        device=torch.device("cpu"),
    )


def test_prepare_inputs_dispatches_to_310p_implementation() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    scheduler_output = MagicMock()
    batch_desc = MagicMock()
    expected = object()

    with patch.object(runner, "_prepare_inputs_310p", return_value=expected) as prepare_inputs_310p:
        result = runner.prepare_inputs(scheduler_output, MagicMock(), batch_desc)

    assert result is expected
    prepare_inputs_310p.assert_called_once_with(scheduler_output, batch_desc)


def test_post_update_cpu_matches_upstream_bookkeeping() -> None:
    idx_mapping_np = np.array([1, 0], dtype=np.int32)
    query_start_loc_np = np.array([0, 2, 4], dtype=np.int32)
    total_len = Ascend310PStagedWriteTensor(2, dtype=torch.int32, device=torch.device("cpu"))
    num_computed_tokens = Ascend310PStagedWriteTensor(2, dtype=torch.int32, device=torch.device("cpu"))
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(cpu=torch.zeros((2, 8), dtype=torch.int32)),
        last_sampled_tokens_cpu=torch.zeros((2, 1), dtype=torch.int64),
        total_len=total_len,
        num_computed_tokens_np=np.zeros(2, dtype=np.int32),
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        num_computed_tokens=num_computed_tokens,
    )
    sampled_tokens = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
    num_sampled = torch.tensor([2, 1], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1], dtype=torch.int32)

    sampled_cpu = model_runner_module._post_update_cpu(
        idx_mapping_np,
        query_start_loc_np,
        req_states,
        sampled_tokens,
        num_sampled,
        num_rejected,
    )

    torch.testing.assert_close(sampled_cpu, num_sampled)
    np.testing.assert_array_equal(req_states.total_len.np, [1, 2])
    np.testing.assert_array_equal(req_states.num_computed_tokens_np, [1, 2])
    torch.testing.assert_close(req_states.last_sampled_tokens_cpu[:, 0], torch.tensor([20, 11]))
    torch.testing.assert_close(
        req_states.all_token_ids.cpu[1, :2],
        torch.tensor([10, 11], dtype=torch.int32),
    )
    req_states.total_len.apply_write()
    req_states.num_computed_tokens.apply_write()
    torch.testing.assert_close(req_states.total_len.gpu, torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(
        req_states.num_computed_tokens.gpu,
        torch.tensor([1, 2], dtype=torch.int32),
    )


def test_postprocess_sampled_keeps_last_token_on_device() -> None:
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.is_last_pp_rank = False
    runner._postprocess_idx_mapping_np = np.array([1, 0], dtype=np.int32)
    runner._postprocess_query_start_loc_np = np.array([0, 2, 4], dtype=np.int32)
    runner.req_states = SimpleNamespace(
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        last_sampled_tokens=torch.zeros((2, 1), dtype=torch.int64),
        last_sampled_tokens_cpu=torch.tensor([[20], [11]], dtype=torch.int64),
    )
    runner.model_state = MagicMock()
    runner.speculator = object()
    runner.rejection_sampler = MagicMock()
    runner._decode_req_indices = model_runner_module.CpuGpuBuffer(
        2, dtype=torch.int64, device=runner.device, pin_memory=False
    )
    runner._decode_input_indices = model_runner_module.CpuGpuBuffer(
        2, dtype=torch.int64, device=runner.device, pin_memory=False
    )
    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    sampled_tokens = torch.tensor([[10, 11], [20, -1]], dtype=torch.int32)
    num_sampled = torch.tensor([2, 1], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    runner._sampled_tokens_cpu = sampled_tokens
    runner._num_sampled_cpu = num_sampled
    runner._num_rejected_cpu = num_rejected

    with patch.object(model_runner_module, "_post_update_cpu", return_value=num_sampled.cpu()) as post_update:
        runner.postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

    post_update.assert_called_once()
    runner.model_state.postprocess_state.assert_called_once()
    postprocess_args = runner.model_state.postprocess_state.call_args.args
    torch.testing.assert_close(
        postprocess_args[0],
        torch.from_numpy(runner._postprocess_idx_mapping_np),
    )
    torch.testing.assert_close(postprocess_args[1], num_sampled.cpu())
    assert postprocess_args[2] is runner.req_states.num_computed_tokens_cpu
    torch.testing.assert_close(runner.req_states.last_sampled_tokens, runner.req_states.last_sampled_tokens_cpu)


@pytest.mark.parametrize(("finished_req_ids", "sync_count"), [({"finished"}, 1), (set(), 0)])
def test_finished_requests_synchronize_before_reusing_layout(finished_req_ids, sync_count) -> None:
    runner = object.__new__(NPUModelRunner310V2)
    scheduler_output = SimpleNamespace(finished_req_ids=finished_req_ids)

    with (
        patch.object(NPUModelRunner, "finish_requests") as finish_requests,
        patch.object(model_runner_module.torch.npu, "current_stream") as current_stream,
    ):
        runner.finish_requests(scheduler_output)

    finish_requests.assert_called_once_with(scheduler_output)
    assert current_stream.return_value.synchronize.call_count == sync_count


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


def test_config_accepts_mtp_and_rejects_non_mtp() -> None:
    """310P MRv2 allows method=mtp only."""
    NPUModelRunner310V2._validate_config(
        _make_vllm_config(speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1))
    )
    with pytest.raises(NotImplementedError, match="only supported via MTP"):
        NPUModelRunner310V2._validate_config(_make_vllm_config(speculative_config=SimpleNamespace(method="eagle")))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("speculative_config", object(), "only supported via MTP"),
        ("kv_transfer_config", object(), "KV cache transfer"),
        ("lora_config", object(), "LoRA"),
    ],
)
def test_config_rejects_out_of_scope_features(field, value, message) -> None:
    with pytest.raises(NotImplementedError, match=message):
        NPUModelRunner310V2._validate_config(_make_vllm_config(**{field: value}))


def test_copy_kv_cache_blocks_flattens_mamba_lists() -> None:
    """Prefix-cache CoW must flatten list[Tensor] mamba layers for upstream copy."""
    runner = object.__new__(NPUModelRunner310V2)
    runner._attn_kv_copy_params = []
    t0 = torch.zeros(4, 2)
    t1 = torch.zeros(4, 2)
    runner.kv_caches = [[t0, t1], torch.zeros(2)]  # hybrid: mamba list + other
    runner.kv_cache_config = SimpleNamespace(num_blocks=4)
    copies = [SimpleNamespace(src_block_id=0, dst_block_id=1)]

    with patch.object(model_runner_module, "copy_kv_cache_blocks_inplace") as mock_copy:
        NPUModelRunner310V2._copy_kv_cache_blocks_310p(runner, copies)

    mock_copy.assert_called_once()
    tensors_arg, num_blocks, copies_arg = mock_copy.call_args[0]
    assert tensors_arg == [t0, t1]
    assert num_blocks == 4
    assert copies_arg is copies


def test_sampler_accepts_temperature_and_rejects_penalties() -> None:
    sampler = Ascend310PSampler(max_num_reqs=4, device="cpu", vocab_size=16)
    sampler.add_request(0, 4, SamplingParams(temperature=0))
    sampler.add_request(1, 4, SamplingParams(temperature=0.8, top_p=0.9, top_k=8, seed=7))
    sampler.apply_staged_writes()
    assert sampler.sampling_states.temperature.gpu[1].item() == pytest.approx(0.8)
    assert sampler.sampling_states.top_p.gpu[1].item() == pytest.approx(0.9)
    assert int(sampler.sampling_states.top_k.gpu[1].item()) == 8
    with pytest.raises(NotImplementedError, match="Unsupported sampling parameters"):
        sampler.add_request(2, 4, SamplingParams(temperature=0, frequency_penalty=0.5))


def test_sampler_temperature_scales_logits_before_argmax() -> None:
    """Non-1 temperature must change relative logits before greedy/top paths."""
    from vllm_ascend._310p.worker.v2.sampler import _apply_temperature_pytorch

    logits = torch.tensor([[2.0, 4.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    expanded = torch.tensor([0, 1], dtype=torch.int64)
    temperature = torch.tensor([0.5, 1.0], dtype=torch.float32)
    _apply_temperature_pytorch(logits, expanded, temperature)
    torch.testing.assert_close(logits[0], torch.tensor([4.0, 8.0, 0.0]))
    torch.testing.assert_close(logits[1], torch.tensor([1.0, 1.0, 1.0]))


def test_sampler_greedy_call_returns_argmax() -> None:
    sampler = Ascend310PSampler(max_num_reqs=2, device="cpu", vocab_size=4)
    sampler.add_request(0, 2, SamplingParams(temperature=0))
    logits = torch.tensor([[0.1, 3.0, 0.2, 0.0]], dtype=torch.float32)
    input_batch = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_reqs=1,
        seq_lens=torch.ones(1, dtype=torch.int32),
    )
    out = sampler(logits, input_batch)
    assert int(out.sampled_token_ids.view(-1)[0].item()) == 1


def _make_sampler_batch(req_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        expanded_idx_mapping=torch.tensor([req_idx], dtype=torch.int32),
        idx_mapping_np=np.array([req_idx], dtype=np.int32),
        num_reqs=1,
        seq_lens=torch.ones(1, dtype=torch.int32),
    )


def test_310p_mrv2_apply_top_k_top_p_masks_logits() -> None:
    """Align with MRV1 ``tests/ut/sample/test_sampler.py`` mask checks.

    310P MRV2 calls the same Triton-free ``apply_top_k_top_p`` path; assert
    discarded logits become ``-inf`` and the kept set size matches top-k / top-p.
    """
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    # token0 > token1 >> rest
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
    k = torch.tensor([2], dtype=torch.int32)
    p = torch.tensor([0.5], dtype=torch.float32)

    top_k_only = apply_top_k_top_p(logits.clone(), k, None)
    finite_k = (top_k_only[0] > float("-inf")).nonzero(as_tuple=False).view(-1)
    assert finite_k.tolist() == [0, 1]
    assert top_k_only[0, 0] == pytest.approx(5.0)
    assert top_k_only[0, 1] == pytest.approx(4.0)
    assert not torch.isfinite(top_k_only[0, 2:]).any()

    # Mirror MRV1: combined filter keeps finite values and original shape.
    filtered = apply_top_k_top_p(logits.clone(), k, p)
    assert filtered.shape == logits.shape
    assert torch.isfinite(filtered).any()
    assert 0 in (filtered[0] > float("-inf")).nonzero(as_tuple=False).view(-1).tolist()

    top_p_only = apply_top_k_top_p(logits.clone(), None, p)
    finite_p = (top_p_only[0] > float("-inf")).nonzero(as_tuple=False).view(-1)
    assert 0 in finite_p.tolist()
    assert finite_p.numel() >= 1
    assert finite_p.numel() < logits.shape[-1]


def test_sampler_top_k_restricts_softmax_mass() -> None:
    """With temp>0, top-k must zero-out discarded tokens before inverse-CDF.

    vLLM clears top_k/top_p when temperature==0, so exercise the random path
    with ``_random_sample_310p`` mocked (CPU-safe; mirrors MRV1 CDF unit style).
    """
    import vllm_ascend._310p.worker.v2.sampler as sampler_mod

    sampler = Ascend310PSampler(max_num_reqs=1, device="cpu", vocab_size=5)
    sampler.add_request(0, 2, SamplingParams(temperature=0.8, top_k=2, top_p=1.0, seed=7))
    sampler.apply_staged_writes()
    # max at idx1, second at idx2; top_k=2 must keep only {1,2}
    logits = torch.tensor([[1.0, 5.0, 4.0, 0.0, -2.0]], dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def _capture_and_argmax(probs: torch.Tensor, generators):
        del generators
        captured["probs"] = probs.detach().cpu().clone()
        return probs.argmax(dim=-1)

    with patch.object(sampler_mod, "_random_sample_310p", side_effect=_capture_and_argmax):
        out = sampler(logits, _make_sampler_batch())

    probs = captured["probs"][0]
    assert probs[3:].sum().item() == pytest.approx(0.0, abs=1e-6)
    assert probs[0].item() == pytest.approx(0.0, abs=1e-6)
    assert probs[1:3].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert int(out.sampled_token_ids.view(-1)[0].item()) == 1


def test_sampler_top_k_one_matches_argmax_with_temperature() -> None:
    """top_k=1 leaves one candidate; sampled token equals global argmax."""
    import vllm_ascend._310p.worker.v2.sampler as sampler_mod

    sampler = Ascend310PSampler(max_num_reqs=1, device="cpu", vocab_size=5)
    sampler.add_request(0, 2, SamplingParams(temperature=0.9, top_k=1, top_p=1.0, seed=3))
    sampler.apply_staged_writes()
    logits = torch.tensor([[0.2, 0.1, 3.0, 1.5, -1.0]], dtype=torch.float32)

    def _argmax_sample(probs: torch.Tensor, generators):
        del generators
        return probs.argmax(dim=-1)

    with patch.object(sampler_mod, "_random_sample_310p", side_effect=_argmax_sample):
        out = sampler(logits, _make_sampler_batch())
    assert int(out.sampled_token_ids.view(-1)[0].item()) == int(logits.argmax(dim=-1).item())


def test_sampler_top_p_restricts_softmax_mass() -> None:
    """top_p must drop the long tail before sampling (MRV1-style mask contract)."""
    import vllm_ascend._310p.worker.v2.sampler as sampler_mod

    sampler = Ascend310PSampler(max_num_reqs=1, device="cpu", vocab_size=5)
    sampler.add_request(0, 2, SamplingParams(temperature=0.8, top_k=-1, top_p=0.5, seed=11))
    sampler.apply_staged_writes()
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def _capture_and_argmax(probs: torch.Tensor, generators):
        del generators
        captured["probs"] = probs.detach().cpu().clone()
        return probs.argmax(dim=-1)

    with patch.object(sampler_mod, "_random_sample_310p", side_effect=_capture_and_argmax):
        out = sampler(logits, _make_sampler_batch())

    probs = captured["probs"][0]
    assert probs[0].item() == pytest.approx(1.0, abs=1e-5) or probs[:2].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert probs[2:].sum().item() == pytest.approx(0.0, abs=1e-5)
    assert int(out.sampled_token_ids.view(-1)[0].item()) == 0


@patch("vllm_ascend._310p.worker.v2.block_table.is_pin_memory_available", return_value=False)
def test_block_tables_use_cpu_metadata_for_gather_and_slot_mapping(_pin_memory) -> None:
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


@patch("vllm_ascend._310p.worker.v2.block_table.is_pin_memory_available", return_value=False)
def test_block_table_expands_logical_blocks_to_310p_kernel_blocks(_pin_memory) -> None:
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
        kv_cache_tensors=[
            SimpleNamespace(
                size=8192,
                # Both supported versions use the #51718 layer descriptor.
                layers=["model.layers.0.self_attn"],
            )
        ],
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
        patch.object(model_runner_module.torch_npu, "empty_with_format", empty_with_format, create=True),
    ):
        caches = runner._allocate_kv_cache_tensors(kv_cache_config, {})

    k_cache, v_cache = caches["model.layers.0.self_attn"]
    assert k_cache.data_ptr() != v_cache.data_ptr()
    assert len(allocations) == 2
    assert all(allocation[3] == model_runner_module.ACL_FORMAT_FRACTAL_NZ for allocation in allocations)


def test_main_attention_descriptor_allocates_private_kv_per_layer() -> None:
    class FakeAttentionSpec:
        block_size = 128
        storage_block_size = 128
        page_size_bytes = 128 * 2 * (128 + 128) * 2
        num_kv_heads = 2
        head_size = 128
        head_size_v = 128
        dtype = torch.float16

    class FakeBackend:
        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_type,
        ):
            del cache_type
            return (
                2,
                num_blocks,
                num_kv_heads * head_size // 16,
                block_size,
                16,
            )

    spec = FakeAttentionSpec()
    layer_names = [
        "model.layers.0.self_attn",
        "model.layers.2.self_attn",
    ]
    kv_cache_config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=layer_names,
            )
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                size=spec.page_size_bytes * 100,
                layers=layer_names,
            )
        ],
    )
    runner = object.__new__(NPUModelRunner310V2)
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.kernel_block_sizes = [64]
    runner.attn_groups = [[SimpleNamespace(backend=FakeBackend, layer_names=layer_names)]]
    allocations = []

    def empty_with_format(*, size, dtype, device, acl_format):
        allocations.append((size, dtype, device, acl_format))
        return torch.zeros(size, dtype=dtype, device=device)

    with (
        patch.object(model_runner_module, "AttentionSpec", FakeAttentionSpec),
        patch.object(
            model_runner_module,
            "AscendAttentionBackend310",
            FakeBackend,
        ),
        patch.object(
            model_runner_module.torch_npu,
            "empty_with_format",
            empty_with_format,
            create=True,
        ),
    ):
        caches = runner._allocate_kv_cache_tensors(kv_cache_config, {})

    assert len(allocations) == 4
    assert caches[layer_names[0]][0].shape[0] == (kv_cache_config.num_blocks * 2)
    assert caches[layer_names[0]][0].data_ptr() != caches[layer_names[1]][0].data_ptr()
    for layer_name in layer_names:
        k_cache, v_cache = caches[layer_name]
        assert (k_cache.nbytes + v_cache.nbytes) == kv_cache_config.num_blocks * spec.page_size_bytes


def test_model_state_uses_greedy_sampler() -> None:
    model_state = object.__new__(Ascend310PModelState)
    model_state.rope_state = None

    model_inputs = model_state.prepare_inputs(SimpleNamespace(), req_states=None)
    sampler, speculator = model_state.custom_sampler(object())

    assert model_inputs == {}
    assert isinstance(sampler, Ascend310PSampler)
    assert speculator is None


def test_model_state_refreshes_full_graph_seq_lens_buffers() -> None:
    model_state = object.__new__(Ascend310PModelState)
    model_state._capture_seq_lens_by_ptr = {}
    shared_capture_buffer = torch.full((4,), -1, dtype=torch.int32)
    second_capture_buffer = torch.full((3,), -1, dtype=torch.int32)

    model_state._record_capture_seq_lens(shared_capture_buffer[:2])
    model_state._record_capture_seq_lens(shared_capture_buffer)
    model_state._record_capture_seq_lens(second_capture_buffer)
    model_state._refresh_capture_seq_lens(torch.tensor([17, 9], dtype=torch.int32))

    torch.testing.assert_close(shared_capture_buffer, torch.tensor([17, 9, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(second_capture_buffer, torch.tensor([17, 9, 0], dtype=torch.int32))


def test_model_state_only_refreshes_seq_lens_for_full_runtime() -> None:
    model_state = object.__new__(Ascend310PModelState)
    capture_seq_lens = torch.full((2,), -1, dtype=torch.int32)
    model_state._capture_seq_lens_by_ptr = {}
    capture_batch = SimpleNamespace(seq_lens=capture_seq_lens)
    input_batch = SimpleNamespace(seq_lens=torch.tensor([11, 12], dtype=torch.int32))

    with patch.object(AscendModelState, "prepare_attn", return_value={}):
        model_state.prepare_attn(
            capture_batch,
            CUDAGraphMode.NONE,
            (),
            object(),
            [],
            object(),
            for_capture=True,
        )
        model_state.prepare_attn(input_batch, CUDAGraphMode.PIECEWISE, (), object(), [], object())
        torch.testing.assert_close(capture_seq_lens, torch.full((2,), -1, dtype=torch.int32))

        model_state.prepare_attn(input_batch, CUDAGraphMode.FULL, (), object(), [], object())
        torch.testing.assert_close(capture_seq_lens, input_batch.seq_lens)


def test_worker_selects_v2_runner_on_310p() -> None:
    atb_ops = MagicMock()
    atb_ops._register_atb_extensions = MagicMock()
    profiler = MagicMock()
    profiler.dynamic_profile = MagicMock()
    with patch.dict(
        sys.modules,
        {
            "torch_npu.op_plugin": MagicMock(),
            "torch_npu.op_plugin.atb": MagicMock(),
            "torch_npu.op_plugin.atb._atb_ops": atb_ops,
            "torch_npu.profiler": profiler,
        },
    ):
        from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.vllm_config = SimpleNamespace()
    worker.use_v2_model_runner = True
    worker.device = torch.device("cpu")
    with patch("vllm_ascend._310p.worker.v2.model_runner.NPUModelRunner310V2") as runner_cls:
        worker.model_runner = worker._create_model_runner()
    runner_cls.assert_called_once_with(worker.vllm_config, worker.device)
