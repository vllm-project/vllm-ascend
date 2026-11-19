# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import vllm_ascend.ops.kimi_kda as projection_module
from vllm_ascend.models.glm5next import kda_input_projection_tp as common_projection
from vllm_ascend.ops.kda_pcp import KimiKDAHostMetadataBackend, KimiKDARecoverSSMMetadataBackend
from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiK3DeltaAttention,
    _KDAFusedBFGLinear,
    _prepare_beta,
)
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import vllm_version_is


def test_fused_bfg_head_parallel_keeps_unpadded_checkpoint_layout():
    with (
        patch("vllm_ascend.ops.kimi_kda.get_kda_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=2),
    ):
        linear = _KDAFusedBFGLinear(
            hidden_size=6,
            num_heads=4,
            head_dim=2,
            tp_size=2,
            quant_config=None,
            prefix="model.layers.0.self_attn.in_proj_gfab",
            head_parallel=True,
        )
    assert tuple(linear.weight.shape) == (10, 6)
    if not vllm_version_is("0.33.0"):
        # The inherited loader queries this property even on the custom
        # head-parallel path that bypasses the merged-layer constructor.
        assert linear._checkpoint_output_sizes == [4, 8, 8]
        assert linear.padding_fill_params() == []
        linear.weight.data.fill_(3)
        linear._zero_output_padding(linear.weight)
        torch.testing.assert_close(linear.weight, torch.full_like(linear.weight, 3))


class _RecordingLinear(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, _input: torch.Tensor):
        return self.output, None


@pytest.mark.parametrize("enabled", [False, True])
def test_kda_backend_selection_is_bound_to_layer_not_ambient_config(enabled):
    from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder

    attention = SimpleNamespace(use_recoverssm=enabled)
    backend = AscendKimiK3DeltaAttention.get_attn_backend(attention)
    assert backend is (KimiKDARecoverSSMMetadataBackend if enabled else KimiKDAHostMetadataBackend)
    if enabled:
        assert backend.get_builder_cls() is AscendKimiKDARecoverSSMMetadataBuilder


class _RecordingStream:
    def __init__(self, name: str, event_names: list[str], trace: list[str]) -> None:
        self.name = name
        self.event_names = iter(event_names)
        self.trace = trace

    def record_event(self) -> str:
        event = next(self.event_names)
        self.trace.append(f"{self.name}.record:{event}")
        return event

    def wait_event(self, event: str) -> None:
        self.trace.append(f"{self.name}.wait:{event}")


class _RecordingTensor:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace

    def record_stream(self, stream: _RecordingStream) -> None:
        self.trace.append(f"{self.name}.record_stream:{stream.name}")


class _RecordingStreamSwitch:
    def __init__(self, stream: _RecordingStream, trace: list[str]) -> None:
        self.stream = stream
        self.trace = trace

    def __enter__(self) -> None:
        self.trace.append(f"enter:{self.stream.name}")

    def __exit__(self, *args) -> None:
        self.trace.append(f"exit:{self.stream.name}")


def test_kda_output_norm_uses_checkpoint_epsilon():
    def fake_upstream_init(attention, _config, _vllm_config, _prefix):
        nn.Module.__init__(attention)
        attention.o_norm = SimpleNamespace(eps=1e-5)
        attention.conv_size = 4
        attention.local_projection_size = 2
        attention.model_config = SimpleNamespace(dtype=torch.bfloat16)
        attention.conv1d = nn.Module()
        attention.conv1d.weight = nn.Parameter(torch.empty(6, 1, 4))
        attention.conv1d.quant_method = SimpleNamespace(process_weights_after_loading=lambda: None)

    config = SimpleNamespace(rms_norm_eps=1e-6)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=None,
            enable_prompt_embeds=False,
        )
    )
    with patch(
        "vllm_ascend.ops.kimi_kda.KimiK3DeltaAttention.__init__",
        new=fake_upstream_init,
    ):
        attention = AscendKimiK3DeltaAttention(config, vllm_config)

    assert attention.o_norm.eps == config.rms_norm_eps


def test_prepare_beta_slices_and_applies_sigmoid_in_fp32():
    raw_beta = torch.tensor(
        [[[-20.0], [0.0], [20.0], [100.0]]],
        dtype=torch.bfloat16,
    )

    beta = _prepare_beta(raw_beta, num_actual_tokens=3)

    assert beta.dtype == torch.float32
    assert beta.shape == (1, 3, 1)
    torch.testing.assert_close(beta, raw_beta[:, :3].float().sigmoid())
    assert torch.all((beta >= 0.0) & (beta <= 1.0))


def test_prepare_beta_does_not_repeat_auxiliary_sigmoid():
    raw_beta = torch.tensor(
        [[[-20.0], [0.0], [20.0], [100.0]]],
        dtype=torch.bfloat16,
    )
    preprocessed_beta = raw_beta.float().sigmoid()

    beta = _prepare_beta(
        preprocessed_beta,
        num_actual_tokens=3,
        is_preprocessed=True,
    )

    assert beta.dtype == torch.float32
    assert beta.shape == (1, 3, 1)
    torch.testing.assert_close(beta, preprocessed_beta[:, :3])


@pytest.mark.parametrize("f_b_is_local", [False, True])
def test_fused_bfg_linear_composes_f_and_packs_bfg(f_b_is_local: bool):
    with (
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=4),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=2),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=2),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=4),
    ):
        linear = _KDAFusedBFGLinear(
            hidden_size=6,
            num_heads=8,
            head_dim=3,
            tp_size=4,
            quant_config=None,
            prefix="model.layers.0.self_attn.in_proj_gfab",
        )

    linear.weight.data.zero_()
    b_weight = torch.arange(8 * 6, dtype=linear.weight.dtype).reshape(8, 6)
    f_a_weight = torch.arange(3 * 6, dtype=linear.weight.dtype).reshape(3, 6) + 100
    global_f_b_weight = torch.arange(24 * 3, dtype=linear.weight.dtype).reshape(24, 3) + 200
    local_f_b_weight = global_f_b_weight[12:18]
    g_weight = torch.arange(24 * 6, dtype=linear.weight.dtype).reshape(24, 6) + 200

    linear.weight.weight_loader(linear.weight, b_weight, 0)
    linear.f_a_weight.weight_loader(linear.f_a_weight, f_a_weight)
    linear.f_b_weight.weight_loader(
        linear.f_b_weight,
        local_f_b_weight if f_b_is_local else global_f_b_weight,
    )
    linear.weight.weight_loader(linear.weight, g_weight, 2)

    expected_f = (local_f_b_weight.float() @ f_a_weight.float()).to(linear.weight.dtype)
    assert tuple(linear.weight.shape) == (14, 6)
    torch.testing.assert_close(linear.weight[:2], b_weight[4:6])
    torch.testing.assert_close(linear.weight[2:8], expected_f)
    torch.testing.assert_close(linear.weight[8:], g_weight[12:18])


def test_fused_bfg_linear_recomposes_f_after_source_reload():
    with (
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        linear = _KDAFusedBFGLinear(
            hidden_size=4,
            num_heads=2,
            head_dim=2,
            tp_size=1,
            quant_config=None,
            prefix="model.layers.0.self_attn.in_proj_gfab",
        )

    linear.weight.data.zero_()
    first_f_a = torch.arange(8, dtype=linear.weight.dtype).reshape(2, 4)
    first_f_b = torch.arange(8, dtype=linear.weight.dtype).reshape(4, 2)
    linear.f_a_weight.weight_loader(linear.f_a_weight, first_f_a)
    torch.testing.assert_close(linear.weight[2:6], torch.zeros_like(linear.weight[2:6]))
    linear.f_b_weight.weight_loader(linear.f_b_weight, first_f_b)
    torch.testing.assert_close(linear.weight[2:6], first_f_b.float() @ first_f_a.float())

    reloaded_f_a = first_f_a + 10
    reloaded_f_b = first_f_b + 20
    linear.f_a_weight.weight_loader(linear.f_a_weight, reloaded_f_a)
    linear.f_b_weight.weight_loader(linear.f_b_weight, reloaded_f_b)
    torch.testing.assert_close(linear.weight[2:6], reloaded_f_b.float() @ reloaded_f_a.float())


@pytest.mark.parametrize("recover", [False, True])
def test_fused_bfg_projection_preserves_staged_outputs(recover):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 3
    attention.use_recoverssm = recover
    attention._fused_bfg_output_sizes = (2, 6, 6)
    hidden_states = torch.randn(4, 5)
    fused_output = torch.arange(56, dtype=torch.float32).reshape(4, 14).to(torch.bfloat16)
    attention.fused_bfg_proj = _RecordingLinear(fused_output)

    projected_bfg = attention._project_bfg(hidden_states)
    assert projected_bfg is fused_output

    beta, raw_gate, output_gate = attention._postprocess_bfg(projected_bfg)
    expected_beta = fused_output[:, :2] if recover else fused_output[:, :2].float().sigmoid()
    torch.testing.assert_close(beta, expected_beta.unsqueeze(0))
    torch.testing.assert_close(raw_gate, fused_output[:, 2:8].reshape(4, 2, 3).unsqueeze(0))
    torch.testing.assert_close(output_gate, fused_output[:, 8:].reshape(4, 2, 3))


@pytest.mark.parametrize("recover", [False, True])
def test_mixed_forward_marks_auxiliary_beta_as_preprocessed(recover):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.uses_mixed_projection = True
    attention.qkv_layout = None
    attention.use_recoverssm = recover
    attention.pcp_size = 1
    attention.local_num_heads = 2
    attention.head_dim = 3
    hidden_states = torch.randn(4, 6)
    positions = torch.arange(4)
    mixed_qkv = torch.randn(4, 18)
    beta = torch.rand(1, 4, 2, dtype=torch.float32)
    raw_gate = torch.randn(1, 4, 2, 3)
    output_gate = torch.randn(4, 2, 3)
    projected = torch.randn(4, 6)
    attention._run_overlapped_qkv_bfg = MagicMock(return_value=(mixed_qkv, beta, raw_gate, output_gate))
    attention._forward = MagicMock()
    attention.o_proj = _RecordingLinear(projected)

    actual = attention.forward(hidden_states, positions)

    assert actual is projected
    assert attention._forward.call_args.kwargs["beta"] is beta
    assert attention._forward.call_args.kwargs["beta_is_preprocessed"] is (not recover)


def test_overlapped_qkv_bfg_keeps_two_stage_vector_cube_overlap():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    trace: list[str] = []
    main_stream = _RecordingStream("main", ["hidden_ready", "quant_ready"], trace)
    bfg_stream = _RecordingStream("bfg", ["bfg_projection_ready", "bfg_ready"], trace)
    hidden_states = _RecordingTensor("hidden", trace)
    fused_bfg = _RecordingTensor("fused_bfg", trace)
    processed_bfg = tuple(_RecordingTensor(name, trace) for name in ("beta", "raw_gate", "output_gate"))
    quantized_qkv = object()
    qkv = object()

    def record_project_bfg(_hidden_states: object) -> _RecordingTensor:
        trace.append("project_bfg")
        return fused_bfg

    def record_dynamic_quant(_hidden_states: object) -> object:
        trace.append("dynamic_quant")
        return quantized_qkv

    def record_qkv_matmul(_qkv_input: object) -> object:
        trace.append("qkv_matmul")
        return qkv

    def record_postprocess_bfg(*_args: object) -> tuple[_RecordingTensor, ...]:
        trace.append("postprocess_bfg")
        return processed_bfg

    attention._project_bfg = MagicMock(side_effect=record_project_bfg)
    attention._quantize_fused_qkv = MagicMock(side_effect=record_dynamic_quant)
    attention._matmul_fused_qkv = MagicMock(side_effect=record_qkv_matmul)
    attention._postprocess_bfg = MagicMock(side_effect=record_postprocess_bfg)

    with (
        patch("vllm_ascend.ops.kimi_kda.torch.npu.current_stream", return_value=main_stream),
        patch("vllm_ascend.ops.kimi_kda._kda_bfg_stream", return_value=bfg_stream),
        patch(
            "vllm_ascend.ops.kimi_kda.npu_stream_switch",
            side_effect=lambda stream: _RecordingStreamSwitch(stream, trace),
        ),
    ):
        actual = attention._run_overlapped_qkv_bfg(hidden_states)

    assert actual == (qkv, *processed_bfg)
    assert trace == [
        "main.record:hidden_ready",
        "hidden.record_stream:bfg",
        "enter:bfg",
        "bfg.wait:hidden_ready",
        "project_bfg",
        "bfg.record:bfg_projection_ready",
        "exit:bfg",
        "dynamic_quant",
        "main.record:quant_ready",
        "main.wait:bfg_projection_ready",
        "qkv_matmul",
        "enter:bfg",
        "bfg.wait:quant_ready",
        "postprocess_bfg",
        "bfg.record:bfg_ready",
        "exit:bfg",
        "beta.record_stream:main",
        "raw_gate.record_stream:main",
        "output_gate.record_stream:main",
        "main.wait:bfg_ready",
    ]


@pytest.mark.parametrize(
    "quant_method_type",
    [AscendW4A8MXFPDynamicLinearMethod, AscendW8A8MXFP8DynamicLinearMethod],
)
def test_fused_qkv_splits_mxfp_dynamic_quant_from_matmul(quant_method_type):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    inner_quant_method = quant_method_type.__new__(quant_method_type)
    if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
        inner_quant_method.dynamic_mx_quant_scale_alg = "floor"
    adapter = SimpleNamespace(
        quant_method=inner_quant_method,
        apply=MagicMock(return_value=torch.randn(4, 18)),
    )
    attention.in_proj_qkvgfab = SimpleNamespace(quant_method=adapter)
    hidden_states = torch.randn(4, 6, dtype=torch.bfloat16)
    quantized = torch.empty(4, 6, dtype=torch.float8_e4m3fn)
    dynamic_scale = torch.empty(4, 1, dtype=torch.uint8)

    with patch(
        "vllm_ascend.ops.kimi_kda.torch_npu.npu_dynamic_mx_quant",
        return_value=(quantized, dynamic_scale),
    ) as dynamic_quant:
        qkv_input = attention._quantize_fused_qkv(hidden_states)

    assert isinstance(qkv_input, tuple)
    assert qkv_input[0] is quantized
    assert qkv_input[1] is dynamic_scale
    if isinstance(inner_quant_method, AscendW8A8MXFP8DynamicLinearMethod):
        dynamic_quant.assert_called_once_with(
            hidden_states,
            dst_type=torch.float8_e4m3fn,
            scale_alg="floor",
        )
    else:
        dynamic_quant.assert_called_once_with(hidden_states, dst_type=torch.float8_e4m3fn)
    output = attention._matmul_fused_qkv(qkv_input)
    assert output is adapter.apply.return_value
    adapter.apply.assert_called_once_with(attention.in_proj_qkvgfab, qkv_input, bias=None)


def test_fused_qkv_keeps_non_mxfp_quantization_in_linear_apply():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    adapter = SimpleNamespace(
        quant_method=object(),
        apply=MagicMock(return_value=torch.randn(4, 18)),
    )
    attention.in_proj_qkvgfab = SimpleNamespace(quant_method=adapter)
    hidden_states = torch.randn(4, 6)

    with patch("vllm_ascend.ops.kimi_kda.torch_npu.npu_dynamic_mx_quant") as dynamic_quant:
        qkv_input = attention._quantize_fused_qkv(hidden_states)

    assert qkv_input is hidden_states
    dynamic_quant.assert_not_called()
    output = attention._matmul_fused_qkv(qkv_input)
    assert output is adapter.apply.return_value
    adapter.apply.assert_called_once_with(attention.in_proj_qkvgfab, hidden_states, bias=None)


@pytest.mark.parametrize("lower_bound", [None, -4.0])
def test_prefill_fuses_raw_gate_and_updates_v_first_state(lower_bound):
    """Prepared state copy preserves fused-gate arguments and cache updates."""
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 2
    attention.gate_lower_bound = lower_bound
    attention.A_log = nn.Parameter(torch.randn(1))
    attention.dt_bias = nn.Parameter(torch.randn(2))

    q = torch.randn(1, 2, 1, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_gate = torch.randn_like(q)
    beta = torch.randn(1, 2, 1)
    recurrent_state = torch.randn(1, 1, 2, 2)
    state_indices = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([True])
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 2),
        cu_seqlens_kern=None,
        keep_meta=None,
        chunk_indices_chunk64_host=(0, 0),
    )
    output = torch.randn_like(v)
    final_state = torch.randn(1, 1, 2, 2)
    # This unit test bypasses worker startup; supply the plan normally bound there.
    attention._ascend_kda_state_copy = SimpleNamespace(
        gather=lambda state, indices, flags: state[indices].contiguous(),
        scatter=lambda state, packed, indices: state.__setitem__(indices, packed.to(state.dtype)),
    )
    attention._kda_state_copy_ready = True
    chunk_kda_fwd = MagicMock(return_value=(output, final_state, *([None] * 10)))

    with (
        patch("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", side_effect=lambda x: x),
        patch("vllm_ascend.ops.kda.chunk_kda_fwd", new=chunk_kda_fwd),
    ):
        actual = attention._run_prefill(
            q,
            k,
            v,
            raw_gate,
            beta,
            recurrent_state,
            state_indices,
            has_initial_state,
            metadata,
        )

    assert actual is output
    assert chunk_kda_fwd.call_args.args[3] is raw_gate
    assert chunk_kda_fwd.call_args.kwargs["use_gate_in_kernel"] is True
    assert chunk_kda_fwd.call_args.kwargs["state_v_first"] is True
    assert chunk_kda_fwd.call_args.args[4] is beta
    assert chunk_kda_fwd.call_args.kwargs["safe_gate"] is (lower_bound is not None)
    assert chunk_kda_fwd.call_args.kwargs["lower_bound"] == (lower_bound if lower_bound is not None else -5.0)
    torch.testing.assert_close(recurrent_state[state_indices], final_state)


@pytest.mark.parametrize("lower_bound", [None, -4.0])
@pytest.mark.parametrize("qkv_padding", [0, 64])
def test_recurrent_preserves_preprocessed_beta_and_mtp_state(lower_bound, qkv_padding):
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.gate_lower_bound = lower_bound
    attention.A_log, attention.dt_bias = torch.zeros(1), torch.zeros(128)
    q, k, v = (torch.ones(1, 4, 1, 128 + qkv_padding * i, dtype=torch.bfloat16)[..., :128] for i in (1, 2, 3))
    beta = torch.full((1, 4, 1), 0.25)
    state = torch.zeros(8, 1, 128, 128)
    starts = torch.tensor([0, 4], dtype=torch.int32)
    slots = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
    accepted = torch.tensor([2], dtype=torch.int32)
    recurrent = MagicMock(return_value=(q, None))
    with patch("vllm_ascend.ops.kda.recurrent_kda", new=recurrent):
        output = attention._run_recurrent(q, k, v, q, beta, state, starts, slots, num_accepted_tokens=accepted)
    assert output is q
    assert recurrent.call_args.args[0] is q
    assert recurrent.call_args.args[1] is k
    assert recurrent.call_args.args[2] is v
    assert recurrent.call_args.args[4] is beta
    assert recurrent.call_args.args[5] is state
    assert recurrent.call_args.kwargs["cu_seqlens"] is starts
    assert recurrent.call_args.kwargs["ssm_state_indices"] is slots
    assert recurrent.call_args.kwargs["state_v_first"] is True
    assert recurrent.call_args.kwargs["inplace_final_state"] is True
    assert recurrent.call_args.kwargs["output_final_state"] is False
    assert recurrent.call_args.kwargs["num_accepted_tokens"] is accepted
    assert recurrent.call_args.kwargs["use_beta_sigmoid_in_kernel"] is False
    assert recurrent.call_args.kwargs["safe_gate"] is (lower_bound is not None)
    assert recurrent.call_args.kwargs["lower_bound"] == (lower_bound if lower_bound is not None else -5.0)


def test_kda_empty_forward_context_clears_preallocated_output():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    core_attn_out = torch.full((1, 4, 2, 3), torch.nan)

    with patch(
        "vllm_ascend.ops.kimi_kda.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=None),
    ):
        attention._forward(
            mixed_qkv=torch.empty(4, 18),
            g1=torch.empty(1, 4, 2, 3),
            g2=torch.empty(4, 2, 3),
            beta=torch.empty(1, 4, 2),
            core_attn_out=core_attn_out,
        )

    assert torch.equal(core_attn_out, torch.zeros_like(core_attn_out))


def test_kda_conv_weight_is_packed_once_in_kernel_layout():
    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.conv_size = 4
    attention.local_projection_size = 6
    attention.conv1d = nn.Module()
    source = torch.arange(18 * 4, dtype=torch.float32).reshape(18, 1, 4)
    attention.conv1d.weight = nn.Parameter(source)
    attention.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(torch.empty(4, 18, dtype=torch.bfloat16), requires_grad=False),
    )
    original = attention.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = original.data_ptr()

    attention._pack_conv_weights()

    packed = attention.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    assert packed.data_ptr() == original_ptr
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    torch.testing.assert_close(
        packed,
        source[:, 0, :].transpose(0, 1).to(torch.bfloat16),
    )


@pytest.mark.parametrize(
    "mode,source_layout",
    [(mode, layout) for mode in ("spec", "decode", "mixed", "mixed_sparse") for layout in ("contiguous", "strided")]
    + [("spec", "cast"), ("decode", "cast")],
)
def test_kda_forward_preserves_live_rows_with_nan_padding(mode, source_layout):
    state = torch.zeros(1)
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4]), cache_indices=torch.tensor([0]), num_accepted_tokens=None
    )
    metadata = SimpleNamespace(
        num_actual_tokens=6,
        num_prefills=0,
        num_decodes=0 if mode == "spec" else 1,
        num_decode_tokens=4,
        spec_sequence_masks=None if mode == "decode" else torch.tensor([True]),
        spec_token_indx=torch.tensor([0, 2] if mode == "mixed_sparse" else [0, 2, 4]),
        non_spec_token_indx=torch.tensor([1, 3] if mode == "mixed_sparse" else [1, 3, 5]),
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=conv_meta),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv_meta),
        spec_query_start_loc=torch.tensor([0, 2 if mode == "mixed" else 4]),
        non_spec_query_start_loc=torch.tensor([0, 2 if mode == "mixed" else 4]),
        spec_state_indices_tensor=torch.tensor([0]),
        non_spec_state_indices_tensor=torch.tensor([0]),
    )
    values = (torch.arange(1, 9, dtype=torch.float32) / 7).view(8, 1).repeat(1, 6)
    values[4:] = torch.nan
    gate = torch.zeros(8, 1, 2)
    gate[4:] = torch.nan

    def recurrent(q, k, v, raw_gate, beta, recurrent_state, query_start_loc, state_indices, **kwargs):
        recurrent_state.add_(1)
        if source_layout == "strided":
            storage = torch.empty(*v.shape[:-1], v.shape[-1] * 2)
            source = storage[..., ::2]
            source.copy_(v)
            return source
        if source_layout == "cast":
            return v.to(torch.bfloat16)
        return v.clone()

    norm_inputs = []

    def norm_into(x, g, *, out):
        norm_inputs.append(x)
        assert x.dtype == out.dtype
        if mode in ("spec", "decode") and source_layout == "contiguous":
            # The fast path must not clear or copy into the destination first.
            assert torch.isnan(out).all()
            assert x.data_ptr() != out.data_ptr()
        else:
            assert x.data_ptr() == out.data_ptr()
        if mode == "mixed_sparse":
            torch.testing.assert_close(x[:, 4:6], torch.zeros_like(x[:, 4:6]))
        out[:, : x.shape[1]].copy_(x * torch.sigmoid(g))
        out[:, x.shape[1] :].zero_()

    attention = SimpleNamespace(
        prefix="kda",
        head_dim=2,
        num_spec=3,
        use_recoverssm=False,
        kv_cache=(torch.zeros(1), state),
        get_parameter=lambda name: torch.empty(1),
        _run_causal_conv1d=lambda x, *args, **kwargs: x,
        _run_recurrent=recurrent,
        o_norm=SimpleNamespace(forward_oot=norm_into),
    )
    output = torch.full((1, 8, 1, 2), torch.nan)
    with (
        patch("vllm_ascend.ops.kimi_kda.GDNAttentionMetadata", SimpleNamespace),
        patch(
            "vllm_ascend.ops.kimi_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"kda": metadata}),
        ),
    ):
        attention._forward_core = AscendKimiK3DeltaAttention._forward_core.__get__(attention)
        AscendKimiK3DeltaAttention._forward(
            attention,
            values,
            torch.zeros(1, 8, 1, 2),
            gate,
            torch.zeros(1, 8, 1),
            output,
        )
    expected = values[:4, :2]
    if source_layout == "cast":
        expected = expected.to(torch.bfloat16).to(output.dtype)
    torch.testing.assert_close(output[0, :4, 0], expected * 0.5)
    torch.testing.assert_close(output[:, 6:], torch.zeros_like(output[:, 6:]))
    torch.testing.assert_close(state, torch.tensor([2.0 if mode.startswith("mixed") else 1.0]))
    assert len(norm_inputs) == 1


@pytest.mark.parametrize("mixed", [False, True])
def test_recoverssm_dispatches_without_spec_mask_and_sigmoids_beta_once(mixed):
    num_tokens, dim = 4, 2
    spec_tokens = torch.tensor([0, 2]) if mixed else None
    non_spec_tokens = torch.tensor([1, 3]) if mixed else None
    spec_count = 2 if mixed else num_tokens
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, spec_count]),
        cache_indices=torch.tensor([7]),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        num_prefills=0,
        num_decodes=int(mixed),
        num_decode_tokens=2 if mixed else 0,
        num_spec_decodes=1,
        spec_sequence_masks=None,
        spec_token_indx=spec_tokens,
        non_spec_token_indx=non_spec_tokens,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=conv_meta),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv_meta),
        spec_query_start_loc=torch.tensor([0, spec_count]),
        non_spec_query_start_loc=torch.tensor([0, 1, 2]),
        spec_state_indices_tensor=torch.tensor([[7]], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([1, 2], dtype=torch.int32),
    )
    qkv = torch.arange(num_tokens * dim * 3, dtype=torch.float32).reshape(num_tokens, dim * 3)
    raw_beta = torch.linspace(-2, 2, num_tokens).reshape(1, num_tokens, 1)
    records = SimpleNamespace(correction=torch.empty(1), key_gate=torch.empty(1))

    def verify(*args, **kwargs):
        expected_beta = raw_beta if spec_tokens is None else raw_beta.index_select(1, spec_tokens)
        torch.testing.assert_close(args[4], expected_beta)
        torch.testing.assert_close(args[12], torch.tensor([7], dtype=torch.int32))
        return args[2] * 2

    def norm_into(x, _gate, *, out):
        out.copy_(x)

    def native_fn(_q, _k, v, _gate, beta, *_args, **_kwargs):
        torch.testing.assert_close(beta, raw_beta.sigmoid().index_select(1, non_spec_tokens))
        return v * 3

    native = MagicMock(side_effect=native_fn)
    attention = SimpleNamespace(
        prefix="kda",
        head_dim=dim,
        num_spec=3,
        use_recoverssm=True,
        recoverssm_records=records,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(dim),
        gate_lower_bound=-5.0,
        kv_cache=(torch.empty(1), torch.empty(1)),
        get_parameter=lambda _name: torch.empty(1),
        _run_causal_conv1d=lambda x, *_args, **_kwargs: x,
        _run_recurrent=native,
        o_norm=SimpleNamespace(forward_oot=norm_into),
    )
    output = torch.empty(1, num_tokens, 1, dim)
    with (
        patch("vllm_ascend.ops.kimi_kda.GDNAttentionMetadata", SimpleNamespace),
        patch(
            "vllm_ascend.ops.kimi_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"kda": metadata}),
        ),
        patch("vllm_ascend.ops.kimi_kda.kda_recoverssm_verify", side_effect=verify) as recover,
    ):
        attention._forward_core = AscendKimiK3DeltaAttention._forward_core.__get__(attention)
        AscendKimiK3DeltaAttention._forward(
            attention, qkv, torch.zeros(1, num_tokens, 1, dim), torch.zeros(num_tokens, 1, dim), raw_beta, output
        )
    assert recover.call_count == 1
    assert native.call_count == int(mixed)
    v = qkv[:, 2 * dim :].reshape(1, num_tokens, 1, dim)
    expected = v * 2
    if mixed:
        expected[:, non_spec_tokens] = v[:, non_spec_tokens] * 3
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("mode", ["prefill", "idle"])
def test_kda_prefill_direct_norm_and_idle_output(mode, recover):
    tokens, capacity, heads, dim = 3, 4, 2, 128
    core = torch.randn(1, tokens, heads, dim)
    before = core.clone()
    output = torch.full((1, capacity, heads, dim), torch.nan)
    gate = torch.randn(capacity, heads, dim)
    conv_meta = SimpleNamespace(query_start_loc=None, cache_indices=None, initial_state_mode=None)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        num_prefills=int(mode == "prefill"),
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        spec_state_indices_tensor=None,
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv_meta, chunk=None),
        prefill_state_indices=torch.tensor([0]),
        prefill_has_initial_state=torch.tensor([False]),
    )
    norm = MagicMock()

    def norm_into(x, g, *, out):
        assert x is core
        assert out is output
        assert torch.isnan(out).all()
        out[:, :tokens].copy_(x * g.sigmoid())
        out[:, tokens:].zero_()

    norm.side_effect = norm_into
    attention = SimpleNamespace(
        prefix="kda",
        use_recoverssm=recover,
        head_dim=dim,
        kv_cache=(torch.empty(1), torch.empty(1)),
        get_parameter=lambda name: torch.empty(1),
        _run_causal_conv1d=lambda x, *args, **kwargs: x,
        _run_prefill=MagicMock(return_value=core),
        o_norm=SimpleNamespace(forward_oot=norm),
    )
    with (
        patch("vllm_ascend.ops.kimi_kda.GDNAttentionMetadata", SimpleNamespace),
        patch(
            "vllm_ascend.ops.kimi_kda.get_forward_context",
            return_value=SimpleNamespace(attn_metadata={"kda": metadata}),
        ),
    ):
        attention._forward_core = AscendKimiK3DeltaAttention._forward_core.__get__(attention)
        raw_beta = torch.linspace(-2, 2, tokens).reshape(1, tokens, 1).expand(-1, -1, heads)
        attention._forward_core = AscendKimiK3DeltaAttention._forward_core.__get__(attention)
        AscendKimiK3DeltaAttention._forward(
            attention,
            torch.randn(tokens, 3 * heads * dim),
            torch.randn(1, tokens, heads, dim),
            gate,
            raw_beta,
            output,
        )
    if mode == "idle":
        norm.assert_not_called()
        attention._run_prefill.assert_not_called()
        torch.testing.assert_close(output, torch.zeros_like(output))
    else:
        norm.assert_called_once()
        torch.testing.assert_close(attention._run_prefill.call_args.args[4], raw_beta.sigmoid())
        torch.testing.assert_close(output[:, :tokens], before * gate[:tokens].sigmoid())
        torch.testing.assert_close(output[:, tokens:], torch.zeros_like(output[:, tokens:]))
        torch.testing.assert_close(core, before)


@pytest.mark.parametrize("groups", ["shared", "distinct", "qkv_only", "gate_only"])
@pytest.mark.parametrize("mode", ["decode", "verify", "mixed", "idle", "no_metadata"])
@pytest.mark.parametrize("comm_mode", ["CCU_SCHED", "AIV", "AIV_ONLY"])
def test_local_tp_projection_stream_order(monkeypatch, groups, mode, comm_mode):
    """Trace stream submissions through the real KDA dispatch, including owners with no recurrent work."""
    trace = []
    main, aux = MagicMock(), MagicMock()
    active = main

    def mark(name):
        trace.append((name, "main" if active is main else "aux"))

    @contextmanager
    def switch(stream):
        nonlocal active
        previous, active = active, stream
        try:
            yield
        finally:
            active = previous

    main.wait_stream.side_effect = lambda stream: mark("join")
    aux.wait_stream.side_effect = lambda stream: mark("fork")
    pack_ready = object()

    def record_event():
        mark("pack_ready")
        return pack_ready

    def wait_event(event):
        assert event is pack_ready
        mark("wait_pack")

    main.record_event.side_effect = record_event
    aux.wait_event.side_effect = wait_event
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", comm_mode)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: active)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(projection_module, "_kda_bfg_stream", lambda: aux)
    monkeypatch.setattr(projection_module, "npu_stream_switch", switch)
    monkeypatch.setattr(common_projection, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=2))

    def gather(hidden, dim):
        mark("AG")
        return torch.cat([hidden, hidden], dim=dim)

    def gather_async(hidden, group):
        mark("AG_submit")
        return torch.cat([hidden, hidden]), SimpleNamespace(wait=lambda: mark("AG_wait"))

    def exchange(out, src, group):
        mark("QKV_A2A" if src.shape[-1] == 3 * q_layout.width + q_layout.heads else "G_A2A")
        out.copy_(src)

    monkeypatch.setattr(projection_module, "all_gather_async", gather_async)
    monkeypatch.setattr(common_projection.dist, "all_to_all_single", exchange)
    group = SimpleNamespace(world_size=2, device_group=object(), all_gather=gather)
    q_group = None if groups == "gate_only" else group
    g_group = None if groups == "qkv_only" else group
    if groups == "distinct":
        g_group = SimpleNamespace(world_size=2, device_group=object(), all_gather=gather)
    layout = common_projection.KDAInputProjectionTPLayout
    q_layout, g_layout = layout(q_group, 2, 2), layout(g_group, 2, 2)
    restore_gate = g_layout.restore_gate

    def restore(received, capacity):
        mark("restore_G")
        return restore_gate(received, capacity)

    monkeypatch.setattr(g_layout, "restore_gate", restore)

    def linear(name, width):
        def apply(x):
            mark(name)
            return x.new_ones((len(x), width)), None

        return apply

    def quantize(hidden):
        mark("quant")
        return hidden

    def conv(x, *args, **kwargs):
        mark("conv")
        return x

    def recurrent(q, k, v, *args, **kwargs):
        mark("recurrent")
        return v.contiguous()

    def verify(q, k, v, *args):
        mark("verify")
        return v.contiguous()

    def norm(x, gate, *, out):
        mark("norm")
        assert gate.shape == x.shape[1:] == (2, 2, 2)
        out.copy_(x * gate)

    speculative = mode in ("verify", "mixed")
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2]), cache_indices=torch.tensor([0]), num_accepted_tokens=None
    )
    ids = torch.tensor([0]) if mode == "mixed" else torch.tensor([0, 1])
    metadata = SimpleNamespace(
        num_actual_tokens=0 if mode == "idle" else 2,
        num_prefills=0,
        num_decodes=int(mode in ("decode", "mixed")),
        num_decode_tokens=int(mode == "mixed") or 2,
        num_spec_decodes=int(speculative),
        spec_sequence_masks=None,
        spec_token_indx=ids,
        non_spec_token_indx=torch.tensor([1]),
        spec_query_start_loc=conv_meta.query_start_loc,
        non_spec_query_start_loc=conv_meta.query_start_loc,
        spec_state_indices_tensor=torch.tensor([[0]]),
        non_spec_state_indices_tensor=torch.tensor([0]),
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=conv_meta),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv_meta),
    )
    monkeypatch.setattr(projection_module, "GDNAttentionMetadata", SimpleNamespace)
    monkeypatch.setattr(
        projection_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None if mode == "no_metadata" else {"layer": metadata}),
    )
    monkeypatch.setattr(projection_module, "kda_recoverssm_verify", verify)
    for name in (
        "maybe_save_kv_layer_to_connector",
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
    ):
        monkeypatch.setattr(projection_module, name, MagicMock())
    layer = SimpleNamespace(
        qkv_layout=q_layout,
        gate_layout=g_layout,
        num_heads=2,
        local_num_heads=2,
        head_dim=2,
        prefix="layer",
        use_recoverssm=speculative,
        num_spec=1,
        kv_cache=(None, None),
        A_log=torch.zeros(2),
        dt_bias=None,
        gate_lower_bound=-5.0,
        recoverssm_records=SimpleNamespace(correction=None, key_gate=None),
        get_parameter=lambda name: None,
        _run_causal_conv1d=conv,
        _run_recurrent=recurrent,
        f_a_proj=linear("F_a", 2),
        f_b_proj=linear("F_b", 4),
        b_proj=linear("beta", q_layout.heads),
        g_proj=linear("G", g_layout.width),
        _quantize_fused_qkv=quantize,
        _matmul_fused_qkv=lambda x: linear("QKV", 3 * q_layout.width)(x)[0],
        o_norm=SimpleNamespace(forward_oot=norm),
        o_proj=lambda x: (x, None),
    )
    layer._forward_core = AscendKimiK3DeltaAttention._forward_core.__get__(layer)
    hidden = torch.ones((0 if mode == "idle" else 2, 4))
    layer._forward_local_tp = AscendKimiK3DeltaAttention._forward_local_tp.__get__(layer)
    output = AscendKimiK3DeltaAttention.forward(layer, hidden, torch.arange(len(hidden)))
    assert trace.count(("AG", "main")) == int(g_group is not None and g_group is not q_group)
    assert trace.count(("AG_submit", "main")) == int(q_group is not None)
    qkv_exchange_stream = "aux" if comm_mode == "CCU_SCHED" else "main"
    assert trace.count(("QKV_A2A", qkv_exchange_stream)) == int(q_group is not None)
    assert trace.count(("G_A2A", "aux")) == int(g_group is not None)
    assert (
        trace.index(("F_a", "main"))
        < trace.index(("F_b", "main"))
        < trace.index(("beta", "aux"))
        < trace.index(("quant", "main"))
        < trace.index(("join", "main"))
        < trace.index(("QKV", "main"))
        < trace.index(("G", "main"))
    )
    gate_index = trace.index(("G", "main"))
    if q_group is not None:
        qkv_exchange_index = trace.index(("QKV_A2A", qkv_exchange_stream))
        assert gate_index < qkv_exchange_index
        assert (
            trace.index(("AG_submit", "main"))
            < trace.index(("F_a", "main"))
            < trace.index(("F_b", "main"))
            < trace.index(("AG_wait", "main"))
            < trace.index(("beta", "aux"))
        )
        if comm_mode == "CCU_SCHED":
            assert trace.index(("pack_ready", "main")) < gate_index < trace.index(("wait_pack", "aux"))
            assert trace[qkv_exchange_index + 1] == ("join", "main")
    if g_group is not None:
        gate_exchange_index = trace.index(("G_A2A", "aux"))
        assert gate_index < gate_exchange_index
        if q_group is not None:
            assert qkv_exchange_index < gate_exchange_index
    if mode in ("idle", "no_metadata"):
        assert ("restore_G", "main") not in trace
        assert ("norm", "main") not in trace
        torch.testing.assert_close(output, torch.zeros_like(hidden))
    else:
        first_kernel = "verify" if speculative else "recurrent"
        assert gate_index < trace.index(("conv", "main")) < trace.index((first_kernel, "main"))
        if g_group is not None:
            assert trace.index(("conv", "main")) < gate_exchange_index < trace.index((first_kernel, "main"))
            assert trace[-3] == ("join", "main")
        if mode == "mixed":
            assert trace.index(("recurrent", "main")) < trace.index(("restore_G", "main"))
        assert trace[-2:] == [("restore_G", "main"), ("norm", "main")]
        torch.testing.assert_close(output, torch.ones_like(hidden))


@pytest.mark.parametrize("dp_tokens", [None, 0])
def test_local_tp_empty_capacity_skips_both_streams(monkeypatch, dp_tokens):
    monkeypatch.setattr(common_projection, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=dp_tokens))
    stream = MagicMock()
    monkeypatch.setattr(projection_module, "_kda_bfg_stream", stream)
    layer = SimpleNamespace(qkv_layout=common_projection.KDAInputProjectionTPLayout(None, 2, 2))
    hidden = torch.empty(0, 3)
    assert AscendKimiK3DeltaAttention._forward_local_tp(layer, hidden).shape == hidden.shape
    stream.assert_not_called()


def test_local_tp_restores_head_order():
    receive = torch.tensor(
        [
            [[10, 11, 20, 21, 30, 31, 40], [110, 111, 120, 121, 130, 131, 140]],
            [[12, 13, 22, 23, 32, 33, 41], [112, 113, 122, 123, 132, 133, 141]],
        ]
    )
    layout = common_projection.KDAInputProjectionTPLayout(SimpleNamespace(world_size=2), 2, 2)
    qkv, beta = layout.restore_qkv_beta(receive, 2)
    torch.testing.assert_close(
        qkv,
        torch.tensor(
            [
                [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
                [110, 111, 112, 113, 120, 121, 122, 123, 130, 131, 132, 133],
            ]
        ),
    )
    torch.testing.assert_close(beta, torch.tensor([[[40, 41], [140, 141]]]))
    torch.testing.assert_close(layout.restore_gate(receive[..., :2], 2), qkv[:, :4].view(2, 2, 2))


@pytest.mark.parametrize(
    "qkv_size,gate_size,heads,full_rank,error",
    [(2, 1, 4, False, "full-rank"), (2, 1, 3, True, "KDA head count"), (1, 2, 3, True, "KDA head count")],
)
def test_kimi_projection_checks_its_own_head_contract(qkv_size, gate_size, heads, full_rank, error):
    def init(attention, config, vllm_config, prefix):
        nn.Module.__init__(attention)
        attention.num_heads = heads

    config = SimpleNamespace(linear_attn_config={"use_full_rank_gate": full_rank})
    vc = SimpleNamespace(
        quant_config=SimpleNamespace(uses_kimi_k3_mixed_kda_projection=lambda prefix: True),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        additional_config={
            "finegrained_tp_config": {
                "qkv_iproj_tensor_parallel_size": qkv_size,
                "gproj_tensor_parallel_size": gate_size,
            }
        },
    )
    with (
        patch("vllm_ascend.ops.kimi_kda.KimiK3DeltaAttention.__init__", init),
        pytest.raises(ValueError, match=error),
    ):
        AscendKimiK3DeltaAttention(config, vc)


def test_kda_forward_preserves_core_override():
    class CustomCore(AscendKimiK3DeltaAttention):
        def _forward_core(self, *args, **kwargs):
            return torch.ones(1, 2, 1, 2)

    attention = CustomCore.__new__(CustomCore)
    nn.Module.__init__(attention)
    attention.o_norm = SimpleNamespace(forward_oot=MagicMock())
    gate = torch.zeros(4, 1, 2)
    output = torch.empty(1, 4, 1, 2)
    with patch("vllm_ascend.ops.kimi_kda.maybe_save_kv_layer_to_connector"):
        attention._forward(torch.empty(0), torch.empty(0), gate, torch.empty(0), output)
    norm_input, owner_gate = attention.o_norm.forward_oot.call_args.args
    torch.testing.assert_close(norm_input, torch.ones(1, 2, 1, 2))
    torch.testing.assert_close(owner_gate, gate[:2])
    assert attention.o_norm.forward_oot.call_args.kwargs["out"] is output
