#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiGatedDeltaNetAttention,
    _KDAFusedBFGLinear,
    _load_a_log,
    _require_kimi_k3_full_rank_gate,
    _zero_padded_spec_output,
)
from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
    AscendW4A8MXFPDynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)


class _NoopQuantMethod(QuantizeMethodBase):
    def create_weights(self, layer: nn.Module, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class _RecordingLinear(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.input: torch.Tensor | None = None

    def forward(self, input_: torch.Tensor):
        self.input = input_
        return self.output, None


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


def _make_conv_pack_attention(
    *,
    local_num_heads: int = 2,
    head_dim: int = 3,
    conv_size: int = 4,
    model_dtype: torch.dtype = torch.bfloat16,
) -> AscendKimiGatedDeltaNetAttention:
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = local_num_heads
    attention.head_dim = head_dim
    attention.conv_size = conv_size
    attention.model_config = SimpleNamespace(dtype=model_dtype)

    local_channels = local_num_heads * head_dim
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = nn.Module()
        conv.weight = nn.Parameter(
            torch.empty(
                local_channels,
                1,
                conv_size,
                dtype=torch.float32,
            )
        )
        conv.weight.weight_loader = default_weight_loader
        conv.quant_method = _NoopQuantMethod()
        setattr(attention, name, conv)

    attention.q_conv1d.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(
            torch.empty(
                attention._packed_conv_shape(),
                dtype=model_dtype,
            ),
            requires_grad=False,
        ),
    )
    for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
        attention._wrap_conv_process_weights(conv)
    return attention


def _process_conv_weights(
    attention: AscendKimiGatedDeltaNetAttention,
) -> None:
    for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
        conv.quant_method.process_weights_after_loading(conv)


def _expected_packed_conv_weights(
    attention: AscendKimiGatedDeltaNetAttention,
) -> torch.Tensor:
    return torch.cat(
        [
            conv.weight[:, 0, :].transpose(0, 1)
            for conv in (
                attention.q_conv1d,
                attention.k_conv1d,
                attention.v_conv1d,
            )
        ],
        dim=1,
    ).to(attention.model_config.dtype)


def test_load_a_log_slices_padded_1d_checkpoint_by_tp_rank():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.arange(6, dtype=torch.float32)

    with patch("vllm_ascend.ops.kimi_kda.get_tensor_model_parallel_rank", return_value=1):
        _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, torch.tensor([[[[2.0], [3.0]]]]))


def test_load_a_log_preserves_exact_local_4d_checkpoint():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.tensor([[[[4.0], [5.0]]]])

    _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, loaded_weight)


def test_load_a_log_rejects_unsupported_layout():
    with pytest.raises(ValueError, match="must be 1-D or 4-D"):
        _load_a_log(torch.empty(1, 1, 2, 1), torch.empty(2, 2), num_heads=4)


def test_zero_padded_spec_output_clears_uninitialized_tail():
    output = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(1, 16, 2, 3)
    output[:, 8:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 8], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :8], output[:, :8])
    assert torch.equal(masked[:, 8:], torch.zeros_like(masked[:, 8:]))
    assert torch.isfinite(masked).all()


def test_zero_padded_spec_output_preserves_fully_covered_output():
    output = torch.randn(1, 16, 2, 3)
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked, output)


def test_zero_padded_spec_output_supports_multiple_real_and_dummy_rows():
    output = torch.randn(1, 32, 2, 3)
    expected = output[:, :16].clone()
    output[:, 16:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :16], expected)
    assert torch.equal(masked[:, 16:], torch.zeros_like(masked[:, 16:]))
    assert masked.shape == output.shape
    assert masked.dtype == output.dtype
    assert masked.device == output.device


def test_kimi_k3_kda_requires_full_rank_gate():
    _require_kimi_k3_full_rank_gate({"use_full_rank_gate": True})
    with pytest.raises(ValueError, match="requires use_full_rank_gate=true"):
        _require_kimi_k3_full_rank_gate({"use_full_rank_gate": False})


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
            prefix="model.layers.0.self_attn.fused_bfg_proj",
        )

    linear.weight.data.zero_()
    b_weight = torch.arange(8 * 6, dtype=linear.weight.dtype).reshape(8, 6)
    f_a_weight = torch.arange(3 * 6, dtype=linear.weight.dtype).reshape(3, 6) + 100
    global_f_b_weight = torch.arange(24 * 3, dtype=linear.weight.dtype).reshape(24, 3) + 200
    local_f_b_weight = global_f_b_weight[12:18]
    g_weight = torch.arange(24 * 6, dtype=linear.weight.dtype).reshape(24, 6) + 200

    linear.weight.weight_loader(linear.weight, b_weight, 0)
    linear.f_a_weight.weight_loader(linear.f_a_weight, f_a_weight)
    with patch("vllm_ascend.ops.kimi_kda.get_tensor_model_parallel_rank", return_value=2):
        linear.f_b_weight.weight_loader(
            linear.f_b_weight,
            local_f_b_weight if f_b_is_local else global_f_b_weight,
        )
    linear.weight.weight_loader(linear.weight, g_weight, 2)

    expected_f = torch.matmul(
        local_f_b_weight.float(),
        f_a_weight.float(),
    ).to(linear.weight.dtype)
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
            prefix="model.layers.0.self_attn.fused_bfg_proj",
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
    torch.testing.assert_close(
        linear.weight[2:6],
        reloaded_f_b.float() @ reloaded_f_a.float(),
    )


def test_fused_bfg_projection_preserves_staged_outputs():
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.head_dim = 3
    attention._fused_bfg_output_sizes = (2, 6, 6)

    hidden_states = torch.randn(4, 5)
    fused_output = torch.arange(56, dtype=torch.float32).reshape(4, 14)
    attention.fused_bfg_proj = _RecordingLinear(fused_output)

    beta, raw_gate, output_gate = attention._project_bfg(hidden_states)

    torch.testing.assert_close(beta, fused_output[:, :2])
    torch.testing.assert_close(raw_gate, fused_output[:, 2:8])
    torch.testing.assert_close(output_gate, fused_output[:, 8:])

    beta, raw_gate, output_gate = attention._postprocess_bfg(
        beta,
        raw_gate,
        output_gate,
    )
    torch.testing.assert_close(beta, fused_output[:, :2].sigmoid().unsqueeze(0))
    torch.testing.assert_close(
        raw_gate,
        fused_output[:, 2:8].reshape(4, 2, 3).unsqueeze(0),
    )
    torch.testing.assert_close(output_gate, fused_output[:, 8:].reshape(4, 2, 3))


def test_overlapped_qkv_bfg_has_two_bidirectional_event_joins():
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    trace: list[str] = []
    main_stream = _RecordingStream(
        "main",
        ["hidden_ready", "quant_ready", "qkv_ready"],
        trace,
    )
    bfg_stream = _RecordingStream(
        "bfg",
        ["bfg_projection_ready", "bfg_ready"],
        trace,
    )
    hidden_states = _RecordingTensor("hidden", trace)
    raw_bfg = tuple(_RecordingTensor(name, trace) for name in ("beta_raw", "raw_gate_raw", "gate_raw"))
    processed_bfg = tuple(_RecordingTensor(name, trace) for name in ("beta", "raw_gate", "output_gate"))
    quantized_qkv = object()
    qkv = object()

    attention._project_bfg = MagicMock(
        side_effect=lambda _: (trace.append("project_bfg"), raw_bfg)[1],
    )
    attention._quantize_fused_qkv = MagicMock(
        side_effect=lambda _: (trace.append("dynamic_quant"), quantized_qkv)[1],
    )
    attention._matmul_fused_qkv = MagicMock(
        side_effect=lambda _: (trace.append("qkv_matmul"), qkv)[1],
    )
    attention._postprocess_bfg = MagicMock(
        side_effect=lambda *_: (trace.append("postprocess_bfg"), processed_bfg)[1],
    )

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
        "main.record:qkv_ready",
        "enter:bfg",
        "bfg.wait:quant_ready",
        "postprocess_bfg",
        "bfg.record:bfg_ready",
        "bfg.wait:qkv_ready",
        "exit:bfg",
        "beta.record_stream:main",
        "raw_gate.record_stream:main",
        "output_gate.record_stream:main",
        "main.wait:bfg_ready",
    ]


@pytest.mark.parametrize(
    "quant_method_type",
    [
        AscendW4A8MXFPDynamicLinearMethod,
        AscendW8A8MXFP8DynamicLinearMethod,
    ],
)
def test_fused_qkv_splits_mxfp_dynamic_quant_from_matmul(quant_method_type):
    attention = AscendKimiGatedDeltaNetAttention.__new__(
        AscendKimiGatedDeltaNetAttention,
    )
    nn.Module.__init__(attention)
    inner_quant_method = quant_method_type.__new__(quant_method_type)
    adapter = SimpleNamespace(
        quant_method=inner_quant_method,
        apply=MagicMock(return_value=torch.randn(4, 18)),
    )
    attention.fused_qkv = SimpleNamespace(quant_method=adapter)
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
    dynamic_quant.assert_called_once_with(hidden_states, dst_type=torch.float8_e4m3fn)

    output = attention._matmul_fused_qkv(qkv_input)

    assert output is adapter.apply.return_value
    adapter.apply.assert_called_once_with(attention.fused_qkv, qkv_input, bias=None)


def test_fused_qkv_keeps_non_mxfp_quantization_in_linear_apply():
    attention = AscendKimiGatedDeltaNetAttention.__new__(
        AscendKimiGatedDeltaNetAttention,
    )
    nn.Module.__init__(attention)
    adapter = SimpleNamespace(
        quant_method=object(),
        apply=MagicMock(return_value=torch.randn(4, 18)),
    )
    attention.fused_qkv = SimpleNamespace(quant_method=adapter)
    hidden_states = torch.randn(4, 6)

    with patch(
        "vllm_ascend.ops.kimi_kda.torch_npu.npu_dynamic_mx_quant",
    ) as dynamic_quant:
        qkv_input = attention._quantize_fused_qkv(hidden_states)

    assert qkv_input is hidden_states
    dynamic_quant.assert_not_called()

    output = attention._matmul_fused_qkv(qkv_input)

    assert output is adapter.apply.return_value
    adapter.apply.assert_called_once_with(
        attention.fused_qkv,
        hidden_states,
        bias=None,
    )


def test_output_norm_gate_uses_kda_fused_triton_kernel():
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.o_norm = SimpleNamespace(
        weight=nn.Parameter(torch.randn(3)),
        eps=1e-6,
    )
    core_attn_out = torch.randn(1, 4, 2, 3)
    output_gate = torch.randn(4, 2, 3)
    expected = torch.randn_like(core_attn_out)

    with patch(
        "vllm_ascend.ops.kimi_kda.apply_kda_rms_norm_sigmoid_gate",
        return_value=expected,
    ) as fused_norm_gate:
        actual = attention._apply_output_norm_gate(core_attn_out, output_gate)

    assert actual is expected
    fused_norm_gate.assert_called_once_with(
        core_attn_out,
        output_gate,
        attention.o_norm.weight,
        attention.o_norm.eps,
    )


def test_conv_post_load_processing_packs_kernel_layout_in_place():
    attention = _make_conv_pack_attention()
    convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for shard_id, conv in enumerate(convs):
        conv.weight.data.copy_(
            torch.arange(
                conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv.weight)
            + shard_id * 100
        )

    packed = attention.q_conv1d.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = packed.data_ptr()
    _process_conv_weights(attention)

    assert packed.data_ptr() == original_ptr
    torch.testing.assert_close(packed, _expected_packed_conv_weights(attention))
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    assert attention._conv_weights_t().data_ptr() == original_ptr
    parameter_name = f"q_conv1d.{_PACKED_CONV_WEIGHT_NAME}"
    assert dict(attention.named_parameters())[parameter_name] is packed
    assert attention.state_dict()[parameter_name].data_ptr() == original_ptr

    convs[1].weight.data.fill_(777)
    convs[1].quant_method.process_weights_after_loading(convs[1])

    assert attention._conv_weights_t().data_ptr() == original_ptr
    torch.testing.assert_close(
        attention._conv_weights_t(),
        _expected_packed_conv_weights(attention),
    )


def test_full_checkpoint_reload_refreshes_packed_weight_in_place():
    attention = _make_conv_pack_attention()
    convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for shard_id, conv in enumerate(convs, start=1):
        conv.weight.data.fill_(shard_id)
    _process_conv_weights(attention)
    original_packed = attention._conv_weights_t()
    original_ptr = original_packed.data_ptr()

    record_metadata_for_reloading(attention)
    initialize_layerwise_reload(attention)
    for shard_id, conv in enumerate(convs, start=11):
        loaded_weight = torch.full(
            conv.weight.shape,
            shard_id,
            dtype=torch.float32,
        )
        conv.weight.weight_loader(conv.weight, loaded_weight)
    finalize_layerwise_reload(
        attention,
        SimpleNamespace(dtype=torch.bfloat16),
    )

    refreshed = attention._conv_weights_t()
    assert refreshed is original_packed
    assert refreshed.data_ptr() == original_ptr
    torch.testing.assert_close(
        refreshed,
        _expected_packed_conv_weights(attention),
    )


def test_repack_waits_until_all_source_weights_are_materialized():
    attention = _make_conv_pack_attention()
    source_convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for value, conv in enumerate(source_convs, start=1):
        conv.weight.data.fill_(value)
    packed = attention._conv_weights_t()
    packed.data.fill_(-1)
    before = packed.clone()
    v_shape = attention.v_conv1d.weight.shape
    attention.v_conv1d.weight = nn.Parameter(
        torch.empty(v_shape, device="meta"),
    )

    attention.q_conv1d.quant_method.process_weights_after_loading(attention.q_conv1d)

    torch.testing.assert_close(packed, before)

    attention.v_conv1d.weight = nn.Parameter(
        torch.full(v_shape, 3, dtype=torch.float32),
    )
    attention.v_conv1d.quant_method.process_weights_after_loading(attention.v_conv1d)

    torch.testing.assert_close(packed, _expected_packed_conv_weights(attention))


def test_kernel_format_reload_updates_named_packed_parameter():
    attention = _make_conv_pack_attention(model_dtype=torch.float16)
    for shard_id, conv in enumerate(
        (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d),
        start=1,
    ):
        conv.weight.data.fill_(shard_id)
    _process_conv_weights(attention)

    parameter_name = f"q_conv1d.{_PACKED_CONV_WEIGHT_NAME}"
    packed = attention.get_parameter(parameter_name)
    original_ptr = packed.data_ptr()
    kernel_weight = torch.arange(
        packed.numel(),
        dtype=packed.dtype,
    ).reshape_as(packed)

    with torch.no_grad():
        attention.get_parameter(parameter_name).copy_(kernel_weight)

    assert attention._conv_weights_t().data_ptr() == original_ptr
    torch.testing.assert_close(attention._conv_weights_t(), kernel_weight)
