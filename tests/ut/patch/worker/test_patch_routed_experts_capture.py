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
#

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.forward_context import BatchDescriptor, DPMetadata, ForwardContext
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.patch.worker import patch_routed_experts_capture

_CAPTURER_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"

_NUM_LAYERS = 4
_TOP_K = 2


def _forward_context(**kwargs) -> ForwardContext:
    return ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={}, **kwargs)


def _capturer(*, dp_rank: int = 0, tp_size: int = 1, max_tokens: int = 16) -> RoutedExpertsCapturer:
    # Bypass __init__: it allocates on the NPU and needs a full VllmConfig.
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.dp_rank = dp_rank
    capturer.tp_size = tp_size
    capturer.output_dtype = torch.uint8
    capturer.device_buffer = torch.full((max_tokens, _NUM_LAYERS, _TOP_K), -1, dtype=torch.int32)
    return capturer


def _prepare(num_tokens: int, *, replace_allreduce: bool = False) -> SimpleNamespace:
    return SimpleNamespace(num_tokens=num_tokens, replace_allreduce=replace_allreduce)


def _set_extras(
    ctx: ForwardContext,
    comm_type: MoECommType | None,
    *,
    prepare: SimpleNamespace | None = None,
    padded_num_tokens: int | None = None,
) -> ForwardContext:
    """Publish the Ascend MoE extras the way the V1 model runner does."""
    ctx.moe_comm_type = comm_type
    ctx.moe_comm_method = SimpleNamespace(prepare_finalize=prepare)
    ctx.padded_num_tokens = padded_num_tokens
    return ctx


def _all_gather_fills(full: torch.Tensor, tp_size: int):
    """``dist.all_gather`` stand-in: fill every split with its chunk of ``full``."""

    def impl(output_list, input_tensor, device_group):
        for chunk, part in zip(output_list, torch.tensor_split(full, tp_size, dim=0)):
            chunk.copy_(part)

    return impl


def _capture(
    capturer: RoutedExpertsCapturer,
    ctx: ForwardContext,
    topk_ids: torch.Tensor,
    *,
    layer_id: int = 0,
    gathered: torch.Tensor | None = None,
    tp_size: int = 1,
):
    """Call ``capture`` with the NPU collectives replaced by test doubles."""
    tp_group = Mock()
    tp_group.device_group = Mock()
    all_gather = Mock(side_effect=_all_gather_fills(gathered, tp_size) if gathered is not None else None)
    with (
        patch(f"{_CAPTURER_MODULE}.get_forward_context", return_value=ctx),
        patch.object(patch_routed_experts_capture, "get_tp_group", return_value=tp_group),
        patch.object(patch_routed_experts_capture, "dist", SimpleNamespace(all_gather=all_gather)),
    ):
        capturer.capture(layer_id=layer_id, topk_ids=topk_ids)
    return tp_group, all_gather


def test_capture_keeps_the_upstream_implementation():
    """Only the row restore is overridden, so upstream keeps owning the layout."""
    assert RoutedExpertsCapturer.capture.__module__ == "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
    assert RoutedExpertsCapturer._restore_local_topk_ids is patch_routed_experts_capture._restore_local_topk_ids


def test_single_dp_all2all_gathers_uneven_tp_shards():
    """All2All only pads to tp_size for small batches; tensor_split is uneven."""
    full = torch.arange(5 * _TOP_K, dtype=torch.int32).view(5, _TOP_K)
    shard = torch.tensor_split(full, 2, dim=0)[1]
    assert shard.shape[0] == 2

    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=5)),
        MoECommType.ALLTOALL,
        prepare=_prepare(5),
    )
    _, all_gather = _capture(capturer, ctx, shard, gathered=full, tp_size=2)

    all_gather.assert_called_once()
    torch.testing.assert_close(capturer.device_buffer[:5, 0], full)


def test_single_dp_mc2_gathers_padded_tp_shards():
    full = torch.arange(4 * _TOP_K, dtype=torch.int32).view(4, _TOP_K)
    shard = torch.tensor_split(full, 2, dim=0)[0]

    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=3)),
        MoECommType.MC2,
        prepare=_prepare(3),
        padded_num_tokens=4,
    )
    _capture(capturer, ctx, shard, gathered=full, tp_size=2)

    torch.testing.assert_close(capturer.device_buffer[:3, 0], full[:3])


def test_single_dp_sp_mc2_restores_the_padded_batch():
    """SP already sharded the sequence; MC2 pads each shard to padded_num_tokens / tp."""
    full = torch.arange(6 * _TOP_K, dtype=torch.int32).view(6, _TOP_K)
    shard = torch.tensor_split(full, 2, dim=0)[1]

    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=6)),
        MoECommType.MC2,
        prepare=_prepare(3, replace_allreduce=True),
        padded_num_tokens=6,
    )
    _capture(capturer, ctx, shard, gathered=full, tp_size=2)

    torch.testing.assert_close(capturer.device_buffer[:6, 0], full)


def test_single_dp_without_batch_descriptor_still_gathers():
    """Without a descriptor the pre-SP size is unknown; keep the padded rows."""
    full = torch.arange(4 * _TOP_K, dtype=torch.int32).view(4, _TOP_K)
    shard = torch.tensor_split(full, 2, dim=0)[1]

    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(),
        MoECommType.MC2,
        prepare=_prepare(3),
        padded_num_tokens=4,
    )
    _capture(capturer, ctx, shard, gathered=full, tp_size=2)

    torch.testing.assert_close(capturer.device_buffer[:4, 0], full)


def test_multi_dp_padded_all_gather_takes_this_rank_block():
    max_tokens = 7
    topk_ids = torch.arange(14 * _TOP_K, dtype=torch.int32).view(14, _TOP_K)

    capturer = _capturer(dp_rank=1, tp_size=2)
    ctx = _set_extras(
        _forward_context(dp_metadata=DPMetadata(num_tokens_across_dp_cpu=torch.tensor([7, 5]))),
        MoECommType.MC2,
        prepare=_prepare(5),
        padded_num_tokens=max_tokens,
    )
    _capture(capturer, ctx, topk_ids)

    # DP rank 1 owns rows [max_tokens, max_tokens + 5) of the padded gather.
    torch.testing.assert_close(capturer.device_buffer[:5, 0], topk_ids[7:12])


def test_multi_dp_all2all_gathers_this_ranks_uneven_shard():
    """DP metadata describes the DP batch; the local batch is still TP-sharded."""
    full = torch.arange(5 * _TOP_K, dtype=torch.int32).view(5, _TOP_K)
    shard = torch.tensor_split(full, 2, dim=0)[0]

    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(dp_metadata=DPMetadata(num_tokens_across_dp_cpu=torch.tensor([5, 7]))),
        MoECommType.ALLTOALL,
        prepare=_prepare(5),
    )
    _capture(capturer, ctx, shard, gathered=full, tp_size=2)

    torch.testing.assert_close(capturer.device_buffer[:5, 0], full)


def test_multi_dp_naive_dispatch_defers_to_upstream():
    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(dp_metadata=DPMetadata(num_tokens_across_dp_cpu=torch.tensor([5, 7]))),
        MoECommType.ALLTOALL,
        prepare=_prepare(5),
    )
    topk_ids = torch.arange(12 * _TOP_K, dtype=torch.int32).view(12, _TOP_K)
    _capture(capturer, ctx, topk_ids)

    torch.testing.assert_close(capturer.device_buffer[:5, 0], topk_ids[:5])


def test_multi_dp_modular_local_batch_defers_to_upstream():
    capturer = _capturer(dp_rank=1, tp_size=2)
    ctx = _set_extras(
        _forward_context(dp_metadata=DPMetadata(num_tokens_across_dp_cpu=torch.tensor([5, 7]))),
        MoECommType.MC2,
        prepare=_prepare(7),
        padded_num_tokens=7,
    )
    topk_ids = torch.arange(7 * _TOP_K, dtype=torch.int32).view(7, _TOP_K)
    _capture(capturer, ctx, topk_ids)

    torch.testing.assert_close(capturer.device_buffer[:7, 0], topk_ids)


def test_multi_dp_gathered_shards_defers_to_upstream():
    """Regression test for #21619: local_sizes picks the shard group, not max_tokens."""
    capturer = _capturer(dp_rank=1, tp_size=2)
    ctx = _set_extras(
        _forward_context(
            dp_metadata=DPMetadata(num_tokens_across_dp_cpu=torch.tensor([3, 2]), local_sizes=[2, 2, 1, 1])
        ),
        MoECommType.ALLTOALL,
        prepare=_prepare(2),
    )
    topk_ids = torch.arange(6 * _TOP_K, dtype=torch.int32).view(6, _TOP_K)
    _capture(capturer, ctx, topk_ids)

    torch.testing.assert_close(capturer.device_buffer[:2, 0], topk_ids[4:])


def test_single_dp_tp_shard_defers_to_upstream_without_a_context_lookup():
    """Regression test for #21619: no _EXTRA_CTX access, so no second lookup."""
    capturer = _capturer(tp_size=2)
    local_shard = torch.tensor([[10, 12], [11, 13]], dtype=torch.int32).t()
    gathered = torch.tensor([[10, 11], [12, 13], [14, 15], [-1, -1]], dtype=torch.int32)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=3)),
        None,
        prepare=_prepare(3),
    )
    tp_group = Mock()
    tp_group.all_gather = Mock(return_value=gathered)

    with (
        patch(f"{_CAPTURER_MODULE}.get_forward_context", return_value=ctx),
        patch(f"{_CAPTURER_MODULE}.get_tp_group", return_value=tp_group),
    ):
        capturer.capture(layer_id=0, topk_ids=local_shard)

    tp_group.all_gather.assert_called_once()
    torch.testing.assert_close(capturer.device_buffer[:3, 0], gathered[:3])


def test_allgather_comm_type_defers_to_upstream():
    """AllGather gathers the batch before routing, so the rows were never split."""
    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=6)),
        MoECommType.ALLGATHER,
        prepare=_prepare(6),
    )
    topk_ids = torch.arange(6 * _TOP_K, dtype=torch.int32).view(6, _TOP_K)
    _capture(capturer, ctx, topk_ids)

    torch.testing.assert_close(capturer.device_buffer[:6, 0], topk_ids)


def test_tp_size_one_defers_to_upstream():
    capturer = _capturer(tp_size=1)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=4)),
        MoECommType.ALLTOALL,
        prepare=_prepare(4),
    )
    topk_ids = torch.arange(4 * _TOP_K, dtype=torch.int32).view(4, _TOP_K)
    _capture(capturer, ctx, topk_ids)

    torch.testing.assert_close(capturer.device_buffer[:4, 0], topk_ids)


def test_v2_runner_publishes_the_extras_through_additional_kwargs():
    """The v2 model runner keeps the Ascend extras in ``additional_kwargs``."""
    full = torch.arange(5 * _TOP_K, dtype=torch.int32).view(5, _TOP_K)
    shard = torch.tensor_split(full, 2, dim=0)[0]

    capturer = _capturer(tp_size=2)
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=5))
    ctx.additional_kwargs["moe_comm_type"] = MoECommType.ALLTOALL
    ctx.additional_kwargs["moe_comm_method"] = SimpleNamespace(prepare_finalize=_prepare(5))
    ctx.additional_kwargs["padded_num_tokens"] = 5
    _capture(capturer, ctx, shard, gathered=full, tp_size=2)

    torch.testing.assert_close(capturer.device_buffer[:5, 0], full)


def test_missing_dispatch_metadata_fails_fast():
    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=5)),
        MoECommType.ALLTOALL,
        prepare=None,
    )
    topk_ids = torch.arange(3 * _TOP_K, dtype=torch.int32).view(3, _TOP_K)

    with pytest.raises(AssertionError, match="cannot rebuild the TP-sharded routing batch"):
        _capture(capturer, ctx, topk_ids)
    assert (capturer.device_buffer == -1).all()


def test_layer_out_of_range_is_rejected_before_gathering():
    capturer = _capturer(tp_size=2)
    ctx = _set_extras(
        _forward_context(batch_descriptor=BatchDescriptor(num_tokens=5)),
        MoECommType.ALLTOALL,
        prepare=_prepare(5),
    )
    topk_ids = torch.arange(3 * _TOP_K, dtype=torch.int32).view(3, _TOP_K)

    with pytest.raises(IndexError, match="exceeds capture buffer"):
        _capture(capturer, ctx, topk_ids, layer_id=_NUM_LAYERS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
