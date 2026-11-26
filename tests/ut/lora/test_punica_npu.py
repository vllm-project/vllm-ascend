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

import gc
import weakref
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
    wrapper.add_lora_embedding(y, x, weights)
    wrapper.sgmv_shrink.assert_not_called()
    wrapper.sgmv_expand_slice.assert_not_called()
    wrapper.sgmv_expand.assert_not_called()


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
    wrapper.bgmv_shrink, wrapper.bgmv_expand_slice = bgmv_shrink, bgmv_expand_slice
    torch.manual_seed(0)
    max_loras, num_experts, rank, hidden, inter, num_tokens, offset = 2, 4, 3, 6, 4, 6, 2
    slots = torch.tensor([0, 1, -1, 1, 0, 1], dtype=torch.long)
    exp_ids = torch.tensor([2, 0, 1, 3, 4, 1], dtype=torch.long)
    enabled = torch.tensor([0, 1], dtype=torch.int32)

    a_g, a_u = [torch.randn(max_loras, 1, rank, hidden) for _ in range(2)]
    b_g, b_u = [torch.randn(max_loras, num_experts, inter, rank) for _ in range(2)]
    x_w13 = torch.randn(num_tokens, hidden)
    y_w13_base = torch.randn(num_tokens, offset + 2 * inter)
    y_w13 = y_w13_base.clone()
    wrapper.add_lora_fused_moe(
        y_w13,
        x_w13,
        (a_g, a_u),
        (b_g, b_u),
        expert_ids=exp_ids,
        adapter_enabled=enabled,
        token_lora_mapping=slots,
        offset=offset,
    )

    a_d = torch.randn(max_loras, num_experts, rank, inter)
    b_d = torch.randn(max_loras, 1, hidden, rank)
    x_w2 = torch.randn(num_tokens, inter)
    weights = torch.tensor([0.25, 0.5, 0.75, 1.25, 0.4, 0.9])
    y_w2_base = torch.randn(num_tokens, hidden)
    y_w2 = y_w2_base.clone()
    wrapper.add_lora_fused_moe(
        y_w2,
        x_w2,
        (a_d,),
        (b_d,),
        expert_ids=exp_ids,
        adapter_enabled=enabled,
        token_lora_mapping=slots,
        mul_routed_weight=True,
        topk_weights=weights,
    )

    y_w13_ref, y_w2_ref = y_w13_base.clone(), y_w2_base.clone()
    for row in range(num_tokens):
        lid, eid = int(slots[row]), int(exp_ids[row])
        if lid >= 0 and eid < num_experts and enabled[lid]:
            y_w13_ref[row, offset : offset + inter] += (x_w13[row] @ a_g[lid, 0].T) @ b_g[lid, eid].T
            y_w13_ref[row, offset + inter : offset + 2 * inter] += (x_w13[row] @ a_u[lid, 0].T) @ b_u[lid, eid].T
            y_w2_ref[row] += ((x_w2[row] @ a_d[lid, eid].T) * weights[row]) @ b_d[lid, 0].T
    torch.testing.assert_close(y_w13, y_w13_ref, atol=1e-5, rtol=1e-5)
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


def test_punica_wrapper_is_collected_from_global_registry() -> None:
    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A2),
        ),
        patch(
            "vllm_ascend.lora.punica_npu.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes"),
    ):
        wrapper = PunicaWrapperNPU(
            8,
            2,
            torch.device("cpu"),
            lora_config=SimpleNamespace(
                max_lora_rank=16,
                max_loras=1,
                fully_sharded_loras=False,
                lora_dtype=torch.bfloat16,
            ),
        )

    wrapper_id = wrapper._wrapper_id
    wrapper_ref = weakref.ref(wrapper)
    mask_ref = weakref.ref(wrapper._single_lora_mask)
    assert lora_ops._LORA_WRAPPERS.get(wrapper_id) is wrapper

    del wrapper
    gc.collect()

    assert wrapper_ref() is None
    assert mask_ref() is None
    assert wrapper_id not in lora_ops._LORA_WRAPPERS


def test_lora_linear_raises_runtime_error_for_collected_wrapper() -> None:
    y = torch.zeros(2, 4, dtype=torch.bfloat16)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    a = [torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)]
    b = [torch.zeros(1, 1, 4, 2, dtype=torch.bfloat16)]
    with pytest.raises(RuntimeError, match="no longer registered"):
        lora_ops.lora_linear(-1, y, x, a, b, 1.0, [4], None, None, True)


def test_lora_linear_raises_runtime_error_for_stale_wrapper_id() -> None:
    with (
        patch(
            "vllm_ascend.lora.punica_npu.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A2),
        ),
        patch(
            "vllm_ascend.lora.punica_npu.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes"),
    ):
        wrapper = PunicaWrapperNPU(
            8,
            2,
            torch.device("cpu"),
            lora_config=SimpleNamespace(
                max_lora_rank=16,
                max_loras=1,
                fully_sharded_loras=False,
                lora_dtype=torch.bfloat16,
            ),
        )

    wrapper_id = wrapper._wrapper_id
    wrapper_ref = weakref.ref(wrapper)
    assert lora_ops._LORA_WRAPPERS.get(wrapper_id) is wrapper

    del wrapper
    gc.collect()

    assert wrapper_ref() is None
    assert wrapper_id not in lora_ops._LORA_WRAPPERS

    y = torch.zeros(2, 4, dtype=torch.bfloat16)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    a = [torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)]
    b = [torch.zeros(1, 1, 4, 2, dtype=torch.bfloat16)]
    with pytest.raises(RuntimeError, match=rf"wrapper id {wrapper_id} is no longer registered"):
        lora_ops.lora_linear(wrapper_id, y, x, a, b, 1.0, [4], None, None, True)


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("add_inputs", [True, False])
def test_apply_expand_and_shrink_fp32_uses_weight_dtype_workspace(is_prefill: bool, add_inputs: bool) -> None:
    wrapper, seen = _make_wrapper(is_prefill=is_prefill, no_lora=False, device_type=AscendDeviceType.A3), {}

    def strict_shrink(inp, _w, out, *args) -> None:
        assert inp.dtype in (torch.half, torch.bfloat16)
        seen["shrink_x_dtype"] = inp.dtype
        out.fill_(2.0)

    def strict_expand_slice(_inp, _w, out, *args) -> None:
        assert out.dtype in (torch.half, torch.bfloat16)
        off, size, add_in = args[-3:]
        seen.update(out_dtype=out.dtype, off=off, size=size, add_in=add_in)
        out[:, off : off + size] += 1.5

    setattr(wrapper, "sgmv_shrink" if is_prefill else "bgmv_shrink", Mock(side_effect=strict_shrink))
    setattr(wrapper, "sgmv_expand_slice" if is_prefill else "bgmv_expand_slice", Mock(side_effect=strict_expand_slice))

    shrink_buf = torch.zeros(2, 4, dtype=torch.float32)
    wrapper._apply_shrink(
        shrink_buf, torch.ones(2, 8, dtype=torch.float32), torch.ones(2, 4, 8, dtype=torch.bfloat16), 0.5
    )
    y_base = torch.arange(1, 13, dtype=torch.float32).view(2, 6)
    y = y_base.clone()
    wrapper._apply_expand(
        y, shrink_buf, torch.ones(2, 4, 4, dtype=torch.bfloat16), y_offset=2, y_slice_size=4, add_inputs=add_inputs
    )
    expected = y_base.clone()
    expected[:, 2:6] = expected[:, 2:6] + 1.5 if add_inputs else 1.5
    assert y.dtype == torch.float32 and seen == {
        "shrink_x_dtype": torch.bfloat16,
        "out_dtype": torch.bfloat16,
        "off": 0,
        "size": 4,
        "add_in": True,
    }
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("is_prefill", [False, True])
def test_add_lora_linear_max_loras_gt1_supports_fp32_gating_output(is_prefill: bool) -> None:
    from vllm.lora.ops import torch_ops

    wrapper = _make_wrapper(is_prefill=is_prefill, no_lora=False, device_type=AscendDeviceType.A3)
    (
        wrapper._single_lora_slot,
        wrapper._single_lora_mask,
        wrapper._lora_shrink_buffers,
        wrapper._lora_triton_workspaces,
        wrapper._max_num_batched_tokens,
    ) = False, None, {}, {}, 8
    wrapper.batch_size, wrapper.max_length, wrapper.token_nums, wrapper.indices_len = 3, 2, 4, [4, 4, 3, 3]
    wrapper._seq_start_locs = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    wrapper._seq_lengths = torch.tensor([2, 1, 1], dtype=torch.int32)
    wrapper._lora_indices_per_batch = torch.tensor([0, -1, 1], dtype=torch.int32)
    wrapper._token_lora_indices = torch.tensor([0, 0, -1, 1], dtype=torch.int64)

    def _checked(fn, idx: int):
        return lambda *args, **kwargs: (assert_dtype(args[idx]), fn(*args, **kwargs))[1]

    def assert_dtype(t):
        assert t.dtype in (torch.half, torch.bfloat16)

    for name, fn, idx in (
        ("bgmv_shrink", torch_ops.bgmv_shrink, 0),
        ("bgmv_expand_slice", torch_ops.bgmv_expand_slice, 2),
        ("sgmv_shrink", torch_ops.sgmv_shrink, 0),
        ("sgmv_expand_slice", torch_ops.sgmv_expand_slice, 2),
    ):
        setattr(wrapper, name, _checked(fn, idx))

    torch.manual_seed(42)
    x, lora_a, lora_b, y_base = (
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.randn(2, 1, 4, 8, dtype=torch.bfloat16),
        torch.randn(2, 1, 6, 4, dtype=torch.bfloat16),
        torch.randn(4, 6, dtype=torch.float32),
    )
    y = y_base.clone()
    wrapper._lora_linear_kernel(y, x, (lora_a,), (lora_b,), 0.5, (6,))
    y_ref = y_base.clone()
    for row, slot in enumerate(wrapper._token_lora_indices.tolist()):
        if slot >= 0:
            y_ref[row] += (
                (((x[row].float() @ lora_a[slot, 0].float().T) * 0.5) @ lora_b[slot, 0].float().T)
                .to(torch.bfloat16)
                .float()
            )
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("act_dtype", "weight_dtype"),
    [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)],
)
def test_add_lora_fused_moe_aligns_cross_dtype_inputs_and_outputs(
    act_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> None:
    from vllm.lora.ops import torch_ops

    wrapper = _make_wrapper(is_prefill=False, no_lora=False, device_type=AscendDeviceType.A3)
    seen_shrink: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
    seen_expand: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []

    def strict_shrink(x, w, y, indices, scale):
        seen_shrink.append((x.dtype, w.dtype, y.dtype))
        assert x.dtype == w.dtype and y.dtype == torch.float32
        return torch_ops.bgmv_shrink(x, w, y, indices, scale)

    def strict_expand_slice(x, w, y, indices, y_offset, y_slice_size, add_inputs=True):
        seen_expand.append((x.dtype, w.dtype, y.dtype))
        assert x.dtype == torch.float32 and y.dtype == w.dtype
        return torch_ops.bgmv_expand_slice(x, w, y, indices, y_offset, y_slice_size, add_inputs=add_inputs)

    wrapper.bgmv_shrink, wrapper.bgmv_expand_slice = strict_shrink, strict_expand_slice
    torch.manual_seed(7)
    a_down = torch.randn(2, 2, 4, 4, dtype=weight_dtype)
    b_down = torch.randn(2, 2, 6, 4, dtype=weight_dtype)
    x = torch.randn(4, 4, dtype=act_dtype)
    y_base = torch.randn(4, 6, dtype=act_dtype)
    y = y_base.clone()
    expert_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    mapping = torch.tensor([0, 1, -1, 0], dtype=torch.long)

    wrapper.add_lora_fused_moe(
        y,
        x,
        (a_down,),
        (b_down,),
        expert_ids=expert_ids,
        adapter_enabled=torch.tensor([1, 1], dtype=torch.int32),
        token_lora_mapping=mapping,
    )
    assert y.dtype == act_dtype
    assert seen_shrink == [(weight_dtype, weight_dtype, torch.float32)]
    assert seen_expand == [(torch.float32, weight_dtype, weight_dtype)]
    y_ref = y_base.float()
    for row in range(4):
        slot, exp = int(mapping[row]), int(expert_ids[row])
        if slot >= 0:
            y_ref[row] += (x[row].to(weight_dtype).float() @ a_down[slot, exp].float().T) @ b_down[slot, exp].float().T
    torch.testing.assert_close(y.float(), y_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize(
    ("act_dtype", "weight_dtype"),
    [(torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)],
)
def test_apply_shrink_expand_and_logits_align_cross_half_bf16_weights(
    is_prefill: bool,
    act_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> None:
    wrapper = _make_wrapper(is_prefill=is_prefill, no_lora=False, device_type=AscendDeviceType.A3)
    seen: list[tuple[str, torch.dtype, torch.dtype]] = []

    def strict_shrink(inp, weight, out, *args, **kwargs) -> None:
        assert inp.dtype == weight.dtype and out.dtype == torch.float32
        seen.append(("sgmv_shrink" if len(args) > 2 else "bgmv_shrink", inp.dtype, weight.dtype))
        out.fill_(2.0)

    def strict_expand(inp, weight, out, *args, **kwargs) -> None:
        assert inp.dtype == torch.float32 and out.dtype == weight.dtype
        seen.append(("sgmv_expand" if len(args) >= 3 else "bgmv_expand", out.dtype, weight.dtype))
        out += 1.5

    def strict_expand_slice(inp, weight, out, *args, **kwargs) -> None:
        assert inp.dtype == torch.float32 and out.dtype == weight.dtype
        off, size = args[-3], args[-2]
        seen.append(("sgmv_expand_slice" if len(args) >= 6 else "bgmv_expand_slice", out.dtype, weight.dtype))
        out[:, off : off + size] += 1.5

    for attr, fn in (
        ("sgmv_shrink", strict_shrink),
        ("sgmv_expand", strict_expand),
        ("sgmv_expand_slice", strict_expand_slice),
        ("bgmv_shrink", strict_shrink),
        ("bgmv_expand", strict_expand),
        ("bgmv_expand_slice", strict_expand_slice),
    ):
        setattr(wrapper, attr, Mock(side_effect=fn))

    shrink_buf = torch.zeros(2, 4, dtype=torch.float32)
    x_act = torch.ones(2, 8, dtype=act_dtype)
    w_a = torch.ones(2, 4, 8, dtype=weight_dtype)
    wrapper._apply_shrink(shrink_buf, x_act, w_a, 0.5)

    y_base = torch.ones(2, 6, dtype=act_dtype) * 2.0
    y = y_base.clone()
    wrapper._apply_expand(y, shrink_buf, torch.ones(2, 4, 4, dtype=weight_dtype), 1, 4, add_inputs=True)
    expected = y_base.clone()
    expected[:, 1:5] += 1.5
    assert y.dtype == act_dtype
    torch.testing.assert_close(y, expected)

    emb_y = torch.ones(2, 8, dtype=act_dtype)
    wrapper.add_lora_embedding(emb_y, torch.ones(2, 4, dtype=act_dtype), torch.ones(2, 8, 4, dtype=weight_dtype))
    torch.testing.assert_close(emb_y, torch.full((2, 8), 2.5, dtype=act_dtype))

    logits_y = torch.ones(2, 8, dtype=act_dtype)
    wrapper.add_lora_logits(logits_y, x_act, w_a, torch.ones(2, 8, 4, dtype=weight_dtype), 0.5)
    torch.testing.assert_close(logits_y, torch.full((2, 8), 2.5, dtype=act_dtype))

    prefix = "sgmv" if is_prefill else "bgmv"
    assert seen == [
        (f"{prefix}_shrink", weight_dtype, weight_dtype),
        (f"{prefix}_expand_slice", weight_dtype, weight_dtype),
        (f"{prefix}_expand", weight_dtype, weight_dtype),
        ("bgmv_shrink", weight_dtype, weight_dtype),
        ("bgmv_expand", weight_dtype, weight_dtype),
    ]
