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


def _make_wrapper(*, is_prefill=False, no_lora=False) -> PunicaWrapperNPU:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = is_prefill
    wrapper.no_lora = no_lora
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
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes") as refresh,
    ):
        wrapper = PunicaWrapperNPU(
            8,
            2,
            torch.device("cpu"),
            lora_config=SimpleNamespace(max_lora_rank=max_lora_rank),
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


def test_add_lora_linear_allocates_fp32_buffer_when_missing() -> None:
    wrapper = _make_wrapper()
    wrapper.add_shrink = Mock()
    wrapper.add_expand = Mock()
    x = torch.ones(3, 8)
    y = torch.zeros(3, 16)
    a = (torch.ones(2, 4, 8),)
    b = (torch.ones(2, 16, 4),)
    wrapper.add_lora_linear(y, x, a, b, 1.0, (16,))
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


def test_add_lora_logits_uses_sampler_indices() -> None:
    wrapper = _make_wrapper()
    y = torch.zeros(2, 8)
    x = torch.ones(2, 4)
    a = torch.ones(3, 4, 4)
    b = torch.ones(3, 8, 4)
    wrapper.add_lora_logits(y, x, a, b, 0.5)
    assert torch.equal(wrapper.bgmv_shrink.call_args.args[3], torch.tensor([1, 0]))
    wrapper.bgmv_expand.assert_called_once()
