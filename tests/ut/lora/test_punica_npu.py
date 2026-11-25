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
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.lora import lora_ops
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def _make_wrapper(
    *,
    is_prefill=False,
    no_lora=False,
    device_type=AscendDeviceType.A2,
    capturing=False,
) -> PunicaWrapperNPU:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = is_prefill
    wrapper.no_lora = no_lora
    wrapper.ascend_device_type = device_type
    wrapper._in_graph_mode = Mock(return_value=capturing)
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand = Mock()
    wrapper.bgmv_expand_slice = Mock()
    wrapper.sgmv_shrink = Mock()
    wrapper.sgmv_expand = Mock()
    wrapper.sgmv_expand_slice = Mock()
    # PunicaWrapperBase exposes these as read-only properties.
    wrapper.batch_size = 1
    wrapper.max_length = 1
    wrapper.token_nums = 1
    wrapper._seq_start_locs = torch.tensor([0])
    wrapper._seq_lengths = torch.tensor([1])
    wrapper._lora_indices_per_batch = torch.tensor([0])
    wrapper._token_lora_indices = torch.tensor([0, 1, 2, 3])
    wrapper._sampler_indices = torch.tensor([1, 0, 1, 0])
    wrapper.indices_len = [2, 2, 2, 2]
    return wrapper


@pytest.mark.parametrize(
    ("device_type", "max_lora_rank", "expect_torch_ops"),
    [
        (AscendDeviceType._310P, 8, True),
        (AscendDeviceType.A2, 128, True),
        (AscendDeviceType.A2, 16, False),
    ],
)
def test_punica_init_selects_kernel_backend(device_type, max_lora_rank, expect_torch_ops) -> None:
    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_current_hardware_profile",
            return_value=get_hardware_profile(device_type),
        ),
        patch(
            "vllm_ascend.lora.punica_npu.get_ascend_device_type",
            return_value=device_type,
        ),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes") as refresh,
    ):
        wrapper = PunicaWrapperNPU(
            8,
            2,
            torch.device("cpu"),
            lora_config=SimpleNamespace(
                max_lora_rank=max_lora_rank,
                max_loras=2,
                fully_sharded_loras=False,
            ),
        )
    refresh.assert_called_once()
    if expect_torch_ops:
        from vllm.lora.ops.torch_ops import bgmv_shrink

        assert wrapper.bgmv_shrink is bgmv_shrink
    else:
        assert wrapper.bgmv_shrink is lora_ops.bgmv_shrink


def test_prefill_calls_sgmv_when_lora_active() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 1.0)
    wrapper._apply_expand(y, x, weights, 0, 8, True)
    wrapper.sgmv_shrink.assert_called_once()
    wrapper.sgmv_expand_slice.assert_called_once()


def test_prefill_shrink_and_expand_skip_when_no_lora() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=True)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 1.0)
    wrapper._apply_expand(y, x, weights, 0, 8, True)
    wrapper.sgmv_shrink.assert_not_called()
    wrapper.sgmv_expand_slice.assert_not_called()


def test_decode_always_invokes_bgmv() -> None:
    wrapper = _make_wrapper(is_prefill=False, no_lora=True)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 0.5)
    wrapper._apply_expand(y, x, weights, 2, 4, False)
    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()
    assert torch.equal(wrapper.bgmv_shrink.call_args.args[3], torch.tensor([0, 1]))


@pytest.mark.parametrize(
    ("is_prefill", "capturing", "device_type", "expected"),
    [
        (False, True, AscendDeviceType._310P, False),
        (False, False, AscendDeviceType._310P, False),
        (True, True, AscendDeviceType._310P, False),
        (True, False, AscendDeviceType._310P, True),
        (True, True, AscendDeviceType.A2, True),
        (True, False, AscendDeviceType.A2, True),
    ],
)
def test_use_sgmv_selects_kernel_by_phase_device_and_capture(is_prefill, capturing, device_type, expected) -> None:
    wrapper = _make_wrapper(is_prefill=is_prefill, device_type=device_type, capturing=capturing)
    assert wrapper._use_sgmv() is expected


def test_prefill_falls_back_to_bgmv_during_graph_capture_on_310p() -> None:
    wrapper = _make_wrapper(is_prefill=True, device_type=AscendDeviceType._310P, capturing=True)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 1.0)
    wrapper._apply_expand(y, x, weights, 0, 8, True)
    wrapper.sgmv_shrink.assert_not_called()
    wrapper.sgmv_expand_slice.assert_not_called()
    wrapper.bgmv_shrink.assert_called_once()
    wrapper.bgmv_expand_slice.assert_called_once()
    assert torch.equal(wrapper.bgmv_shrink.call_args.args[3], torch.tensor([0, 1]))


def test_prefill_keeps_sgmv_when_eager_on_310p() -> None:
    wrapper = _make_wrapper(is_prefill=True, device_type=AscendDeviceType._310P, capturing=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    weights = torch.ones(2, 8, 4)
    wrapper._apply_shrink(y, x, weights, 1.0)
    wrapper._apply_expand(y, x, weights, 0, 8, True)
    wrapper.sgmv_shrink.assert_called_once()
    wrapper.sgmv_expand_slice.assert_called_once()
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


def test_add_lora_embedding_falls_back_to_bgmv_during_graph_capture_on_310p() -> None:
    wrapper = _make_wrapper(is_prefill=True, device_type=AscendDeviceType._310P, capturing=True)
    wrapper.add_lora_embedding(torch.zeros(2, 8), torch.ones(2, 8), torch.ones(2, 8, 8))
    wrapper.sgmv_expand.assert_not_called()
    wrapper.bgmv_expand.assert_called_once()


def test_in_graph_mode_reflects_stream_capture_state() -> None:
    wrapper = _make_wrapper()
    with patch.object(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), create=True):
        assert PunicaWrapperNPU._in_graph_mode(wrapper) is True
    with patch.object(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), create=True):
        assert PunicaWrapperNPU._in_graph_mode(wrapper) is False


def test_in_graph_mode_returns_false_when_stream_query_unavailable() -> None:
    wrapper = _make_wrapper()
    broken_query = Mock(side_effect=RuntimeError("stream query unsupported"))
    with patch.object(torch, "npu", SimpleNamespace(is_current_stream_capturing=broken_query), create=True):
        assert PunicaWrapperNPU._in_graph_mode(wrapper) is False


def test_add_shrink_and_expand_walk_slices_with_offsets() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    wrapper._apply_shrink = Mock()
    wrapper._apply_expand = Mock()
    x = torch.ones(2, 8)
    y = torch.zeros(2, 12)
    a = (torch.ones(1, 4, 8), torch.ones(1, 4, 8))
    b = (torch.ones(1, 4, 4), torch.ones(1, 8, 4))
    wrapper.add_shrink((torch.zeros(2, 4), torch.zeros(2, 4)), x, a, 0.25)
    wrapper.add_expand(y, (torch.ones(2, 4), torch.ones(2, 4)), b, (4, 8), offset_start=0)
    assert wrapper._apply_shrink.call_count == 2
    assert wrapper._apply_expand.call_args_list[0].args[3] == 0
    assert wrapper._apply_expand.call_args_list[1].args[3] == 4
    assert wrapper._apply_expand.call_args_list[1].args[4] == 8


def test_add_lora_embedding_casts_input_to_fp32() -> None:
    wrapper = _make_wrapper(is_prefill=False)
    y = torch.zeros(2, 8)
    x = torch.ones(2, 8, dtype=torch.float16)
    weights = torch.ones(2, 8, 8)
    wrapper.add_lora_embedding(y, x, weights)
    passed_x = wrapper.bgmv_expand.call_args.args[0]
    assert passed_x.dtype == torch.float32


def test_add_lora_embedding_prefill_uses_sgmv_expand() -> None:
    wrapper = _make_wrapper(is_prefill=True, no_lora=False)
    wrapper.add_lora_embedding(torch.zeros(2, 8), torch.ones(2, 8), torch.ones(2, 8, 8))
    wrapper.sgmv_expand.assert_called_once()
    wrapper.bgmv_expand.assert_not_called()


def test_lora_linear_kernel_allocates_fp32_buffer_when_missing() -> None:
    wrapper = _make_wrapper()
    wrapper._lora_shrink_buffers = {}
    wrapper._max_num_batched_tokens = 8
    wrapper.add_shrink = Mock()
    wrapper.add_expand = Mock()
    x = torch.ones(3, 8)
    y = torch.zeros(3, 16)
    a = (torch.ones(2, 4, 8),)
    b = (torch.ones(2, 16, 4),)
    wrapper._lora_linear_kernel(y, x, a, b, 1.0, (16,))
    buffer = wrapper.add_shrink.call_args.args[0]
    assert len(buffer) == 1
    assert buffer[0].shape == (3, 4)
    assert buffer[0].dtype == torch.float32
    wrapper.add_expand.assert_called_once()


def test_add_lora_fused_moe_builds_graph_safe_combined_index() -> None:
    wrapper = _make_wrapper()
    max_loras, num_experts, rank, in_f, out_f = 2, 4, 8, 16, 32
    a = torch.ones(max_loras, num_experts, rank, in_f)
    b = torch.ones(max_loras, num_experts, out_f, rank)
    x = torch.ones(3, in_f)
    y = torch.zeros(3, out_f)
    mapping = torch.tensor([0, -1, 1])
    expert_ids = torch.tensor([1, 2, 3])
    adapter_enabled = torch.tensor([1, 1])
    wrapper.add_lora_fused_moe(
        y,
        x,
        (a,),
        (b,),
        expert_ids=expert_ids,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=mapping,
        offset=5,
    )
    combined = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(combined, torch.tensor([1, -1, 7]))
    assert wrapper.bgmv_expand_slice.call_args.args[4:6] == (5, 32)
    assert wrapper.bgmv_expand_slice.call_args.kwargs["add_inputs"] is True


def test_add_lora_fused_moe_masks_disabled_adapter_rows() -> None:
    wrapper = _make_wrapper()
    a = torch.ones(2, 4, 2, 8)
    b = torch.ones(2, 4, 8, 2)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 8),
        torch.ones(2, 8),
        (a,),
        (b,),
        expert_ids=torch.tensor([1, 2]),
        adapter_enabled=torch.tensor([0, 1]),
        token_lora_mapping=torch.tensor([0, 1]),
    )
    combined = wrapper.bgmv_shrink.call_args.args[3]
    assert torch.equal(combined, torch.tensor([-1, 6]))


def test_add_lora_fused_moe_rejects_unexpanded_rows() -> None:
    wrapper = _make_wrapper()
    with pytest.raises(AssertionError, match="top_k_num=1"):
        wrapper.add_lora_fused_moe(
            torch.zeros(1, 4),
            torch.ones(1, 4),
            (torch.ones(1, 1, 2, 4),),
            (torch.ones(1, 1, 4, 2),),
            expert_ids=torch.tensor([0]),
            adapter_enabled=torch.tensor([1]),
            top_k_num=2,
        )


def test_add_lora_fused_moe_scales_shrink_buffer_by_routed_weight() -> None:
    wrapper = _make_wrapper()

    def _shrink(x, a, shrink_out, idx, scale):
        shrink_out.fill_(2.0)

    wrapper.bgmv_shrink.side_effect = _shrink
    a = torch.ones(1, 1, 2, 4)
    b = torch.ones(1, 1, 4, 2)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 4),
        torch.ones(2, 4),
        (a,),
        (b,),
        expert_ids=torch.tensor([0, 0]),
        adapter_enabled=torch.tensor([1]),
        token_lora_mapping=torch.tensor([0, 0]),
        mul_routed_weight=True,
        topk_weights=torch.tensor([0.5, 1.5]),
    )
    delta = wrapper.bgmv_expand_slice.call_args.args[0]
    torch.testing.assert_close(delta, torch.tensor([[1.0, 1.0], [3.0, 3.0]]))


def test_add_lora_fused_moe_shared_lora_w13_and_w2_math_matches_dense_reference() -> None:
    from vllm.lora.ops.torch_ops import bgmv_expand_slice, bgmv_shrink

    wrapper = _make_wrapper(is_prefill=False, no_lora=False)
    wrapper.bgmv_shrink = bgmv_shrink
    wrapper.bgmv_expand_slice = bgmv_expand_slice

    torch.manual_seed(0)
    max_loras, num_experts, rank = 2, 4, 3
    hidden, inter = 6, 4
    num_tokens = 6
    token_lora_mapping = torch.tensor([0, 1, -1, 1, 0, 1], dtype=torch.long)
    expert_ids = torch.tensor([2, 0, 1, 3, 4, 1], dtype=torch.long)
    adapter_enabled = torch.tensor([0, 1], dtype=torch.int32)

    # w13 shared LoRA: a_gate/a_up shape [max_loras, 1, rank, hidden], b shape [max_loras, num_experts, inter, rank]
    a_gate = torch.randn(max_loras, 1, rank, hidden, dtype=torch.float32)
    a_up = torch.randn(max_loras, 1, rank, hidden, dtype=torch.float32)
    b_gate = torch.randn(max_loras, num_experts, inter, rank, dtype=torch.float32)
    b_up = torch.randn(max_loras, num_experts, inter, rank, dtype=torch.float32)
    x_w13 = torch.randn(num_tokens, hidden, dtype=torch.float32)
    offset = 2
    y_w13_base = torch.randn(num_tokens, offset + 2 * inter, dtype=torch.float32)
    y_w13 = y_w13_base.clone()

    wrapper.add_lora_fused_moe(
        y_w13,
        x_w13,
        (a_gate, a_up),
        (b_gate, b_up),
        expert_ids=expert_ids,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=token_lora_mapping,
        offset=offset,
    )

    y_w13_ref = y_w13_base.clone()
    for row in range(num_tokens):
        lora_id = int(token_lora_mapping[row].item())
        exp_id = int(expert_ids[row].item())
        if lora_id < 0 or exp_id >= num_experts or not adapter_enabled[lora_id].item():
            continue
        gate_delta = (x_w13[row] @ a_gate[lora_id, 0].T) @ b_gate[lora_id, exp_id].T
        up_delta = (x_w13[row] @ a_up[lora_id, 0].T) @ b_up[lora_id, exp_id].T
        y_w13_ref[row, offset : offset + inter] += gate_delta
        y_w13_ref[row, offset + inter : offset + 2 * inter] += up_delta
    torch.testing.assert_close(y_w13, y_w13_ref, atol=1e-5, rtol=1e-5)

    # w2 shared LoRA: a shape [max_loras, num_experts, rank, inter], b shape [max_loras, 1, hidden, rank]
    a_down = torch.randn(max_loras, num_experts, rank, inter, dtype=torch.float32)
    b_down = torch.randn(max_loras, 1, hidden, rank, dtype=torch.float32)
    x_w2 = torch.randn(num_tokens, inter, dtype=torch.float32)
    topk_weights = torch.tensor([0.25, 0.5, 0.75, 1.25, 0.4, 0.9], dtype=torch.float32)
    y_w2_base = torch.randn(num_tokens, hidden, dtype=torch.float32)
    y_w2 = y_w2_base.clone()

    wrapper.add_lora_fused_moe(
        y_w2,
        x_w2,
        (a_down,),
        (b_down,),
        expert_ids=expert_ids,
        adapter_enabled=adapter_enabled,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=True,
        topk_weights=topk_weights,
    )

    y_w2_ref = y_w2_base.clone()
    for row in range(num_tokens):
        lora_id = int(token_lora_mapping[row].item())
        exp_id = int(expert_ids[row].item())
        if lora_id < 0 or exp_id >= num_experts or not adapter_enabled[lora_id].item():
            continue
        down_delta = ((x_w2[row] @ a_down[lora_id, exp_id].T) * topk_weights[row]) @ b_down[lora_id, 0].T
        y_w2_ref[row] += down_delta
    torch.testing.assert_close(y_w2, y_w2_ref, atol=1e-5, rtol=1e-5)


def test_add_lora_logits_uses_sampler_indices() -> None:
    wrapper = _make_wrapper()
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    a = torch.ones(3, 4, 4)
    b = torch.ones(3, 8, 4)
    wrapper.add_lora_logits(y, x, a, b, 0.5)
    assert torch.equal(wrapper.bgmv_shrink.call_args.args[3], torch.tensor([1, 0]))
    wrapper.bgmv_expand.assert_called_once()


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("add_inputs", [True, False])
def test_apply_expand_and_shrink_fp32_uses_weight_dtype_workspace(is_prefill: bool, add_inputs: bool) -> None:
    wrapper = _make_wrapper(is_prefill=is_prefill, no_lora=False, device_type=AscendDeviceType.A3)
    seen: dict[str, object] = {}

    def strict_shrink(inp, weight, out, *args) -> None:
        assert inp.dtype in (torch.half, torch.bfloat16)
        seen["shrink_x_dtype"] = inp.dtype
        out.copy_(torch.full_like(out, 2.0))

    def strict_expand_slice(inp, weight, out, *args) -> None:
        assert out.dtype in (torch.half, torch.bfloat16)
        slice_offset, slice_size, passed_add_inputs = args[-3:]
        seen["expand_out_dtype"] = out.dtype
        seen["expand_offset"] = slice_offset
        seen["expand_slice_size"] = slice_size
        seen["expand_add_inputs"] = passed_add_inputs
        out[:, slice_offset : slice_offset + slice_size] += 1.5

    if is_prefill:
        wrapper.sgmv_shrink = Mock(side_effect=strict_shrink)
        wrapper.sgmv_expand_slice = Mock(side_effect=strict_expand_slice)
    else:
        wrapper.bgmv_shrink = Mock(side_effect=strict_shrink)
        wrapper.bgmv_expand_slice = Mock(side_effect=strict_expand_slice)

    shrink_buf = torch.zeros(2, 4, dtype=torch.float32)
    x_fp32 = torch.ones(2, 8, dtype=torch.float32)
    w_a = torch.ones(2, 4, 8, dtype=torch.bfloat16)
    wrapper._apply_shrink(shrink_buf, x_fp32, w_a, 0.5)
    assert seen["shrink_x_dtype"] == torch.bfloat16

    y_base = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]], dtype=torch.float32)
    y = y_base.clone()
    w_b = torch.ones(2, 4, 4, dtype=torch.bfloat16)
    wrapper._apply_expand(y, shrink_buf, w_b, y_offset=2, y_slice_size=4, add_inputs=add_inputs)

    assert y.dtype == torch.float32
    assert seen["expand_out_dtype"] == torch.bfloat16
    assert seen["expand_offset"] == 0
    assert seen["expand_slice_size"] == 4
    assert seen["expand_add_inputs"] is True
    expected = y_base.clone()
    if add_inputs:
        expected[:, 2:6] += 1.5
    else:
        expected[:, 2:6] = 1.5
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("is_prefill", [False, True])
def test_add_lora_linear_max_loras_gt1_supports_fp32_gating_output(is_prefill: bool) -> None:
    from vllm.lora.ops import torch_ops

    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = is_prefill
    wrapper.no_lora = False
    wrapper.ascend_device_type = AscendDeviceType.A3
    wrapper._single_lora_slot = False
    wrapper._single_lora_mask = None
    wrapper._lora_shrink_buffers = {}
    wrapper._lora_triton_workspaces = {}
    wrapper._max_num_batched_tokens = 8
    wrapper.batch_size = 3
    wrapper.max_length = 2
    wrapper.token_nums = 4
    wrapper._seq_start_locs = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    wrapper._seq_lengths = torch.tensor([2, 1, 1], dtype=torch.int32)
    wrapper._lora_indices_per_batch = torch.tensor([0, -1, 1], dtype=torch.int32)
    wrapper._token_lora_indices = torch.tensor([0, 0, -1, 1], dtype=torch.int64)
    wrapper._sampler_indices = torch.tensor([0, 0, -1, 1], dtype=torch.int64)
    wrapper.indices_len = [4, 4, 3, 3]

    def checked_bgmv_shrink(x, w, y, indices, scale):
        assert x.dtype in (torch.half, torch.bfloat16)
        return torch_ops.bgmv_shrink(x, w, y, indices, scale)

    def checked_bgmv_expand_slice(x, w, y, indices, y_offset, y_slice_size, add_inputs=True):
        assert y.dtype in (torch.half, torch.bfloat16)
        return torch_ops.bgmv_expand_slice(x, w, y, indices, y_offset, y_slice_size, add_inputs=add_inputs)

    def checked_sgmv_shrink(x, w, y, *args):
        assert x.dtype in (torch.half, torch.bfloat16)
        return torch_ops.sgmv_shrink(x, w, y, *args)

    def checked_sgmv_expand_slice(x, w, y, *args):
        assert y.dtype in (torch.half, torch.bfloat16)
        return torch_ops.sgmv_expand_slice(x, w, y, *args)

    wrapper.bgmv_shrink = checked_bgmv_shrink
    wrapper.bgmv_expand_slice = checked_bgmv_expand_slice
    wrapper.sgmv_shrink = checked_sgmv_shrink
    wrapper.sgmv_expand_slice = checked_sgmv_expand_slice

    rank, in_features, out_features = 4, 8, 6
    torch.manual_seed(42)
    x = torch.randn(4, in_features, dtype=torch.bfloat16)
    lora_a = torch.randn(2, 1, rank, in_features, dtype=torch.bfloat16)
    lora_b = torch.randn(2, 1, out_features, rank, dtype=torch.bfloat16)
    y_base = torch.randn(4, out_features, dtype=torch.float32)
    y = y_base.clone()

    with patch.object(wrapper, "_in_graph_mode", return_value=False):
        wrapper._lora_linear_kernel(y, x, (lora_a,), (lora_b,), 0.5, (out_features,))

    assert y.dtype == torch.float32
    y_ref = y_base.clone()
    for row in range(4):
        slot = int(wrapper._token_lora_indices[row].item())
        if slot < 0:
            continue
        delta = (((x[row].float() @ lora_a[slot, 0].float().T) * 0.5) @ lora_b[slot, 0].float().T).to(torch.bfloat16)
        y_ref[row] += delta.float()
    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
