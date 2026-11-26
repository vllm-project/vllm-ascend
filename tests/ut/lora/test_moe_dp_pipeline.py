# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.lora.fused_moe import _recover_moe_lora_routing_allgather, all2all_lora_indices
from vllm_ascend.ops.fused_moe import moe_mlp, prepare_finalize, token_dispatcher
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All, PrepareAndFinalizeWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


def _context(indices, slot_ids=None, no_lora=False):
    return SimpleNamespace(
        top_k=1,
        punica_wrapper=SimpleNamespace(
            token_lora_indices=torch.tensor(indices, dtype=torch.long),
            moe_lora_slot_ids=slot_ids,
            moe_lora_active=True if slot_ids is not None else None,
            no_lora=no_lora,
        ),
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "rank_ids",
    [([11], [-1]), ([11, -1], [22]), ([], [22, 11]), ([11, -1], [22, -1])],
)
def test_allgather_maps_remote_tokens_and_padding_to_receiver_slots(rank, rank_ids):
    slots = [11, 22] if rank == 0 else [22, 11]
    expected_slot = {adapter_id: slot for slot, adapter_id in enumerate(slots)}
    local_ids = rank_ids[rank]
    max_rows = max(map(len, rank_ids))
    # A missing final -1 models graph padding absent from Punica's real-token map.
    indices = [expected_slot.get(adapter_id, -1) for adapter_id in local_ids]
    if indices and indices[-1] == -1:
        indices = indices[:-1]
    ctx = _context(indices, torch.tensor(slots))

    def gather(tensor, dim):
        assert dim == 0
        if tensor.ndim == 1:
            other = rank_ids[1 - rank] + [-1] * (max_rows - len(rank_ids[1 - rank]))
            peer = torch.tensor(other, dtype=torch.long)
        else:
            peer = torch.zeros_like(tensor)
        return torch.cat([tensor, peer] if rank == 0 else [peer, tensor])

    layer = object.__new__(PrepareAndFinalizeWithAllGather)
    layer.lora_context = ctx
    layer.moe_config = SimpleNamespace(
        dp_size=2, pcp_size=1, is_sequence_parallel=False, dp_group=SimpleNamespace(all_gather=gather)
    )
    with patch.object(prepare_finalize, "_EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=max_rows)):
        output = layer.prepare(torch.ones(len(local_ids), 3), torch.ones(len(local_ids), 2))

    global_ids = [value for ids in rank_ids for value in ids + [-1] * (max_rows - len(ids))]
    expected = torch.tensor([expected_slot.get(adapter_id, -1) for adapter_id in global_ids])
    assert output.hidden_states.shape[0] == expected.numel()
    torch.testing.assert_close(ctx.gathered_lora_indices, expected)
    _, routed = _recover_moe_lora_routing_allgather(
        ctx, torch.arange(expected.numel()), torch.zeros(expected.numel(), 1, dtype=torch.long)
    )
    torch.testing.assert_close(routed, expected)


def test_allgather_rejects_mismatched_routing_rows():
    ctx = _context([0], torch.tensor([11]))
    ctx.gathered_lora_indices = torch.tensor([0])
    with pytest.raises(AssertionError, match="metadata does not match"):
        _recover_moe_lora_routing_allgather(ctx, torch.arange(2), torch.zeros(2, 1, dtype=torch.long))


def test_allgather_sequence_parallel_keeps_inactive_padding_and_floating_inputs():
    layer = object.__new__(PrepareAndFinalizeWithAllGather)
    layer.lora_context = _context([0, 1, -1], torch.tensor([11, 22]), no_lora=True)
    layer.moe_config = SimpleNamespace(is_sequence_parallel=True)
    with (
        patch.object(prepare_finalize, "get_tensor_model_parallel_world_size", return_value=2),
        patch.object(prepare_finalize, "sequence_parallel_chunk", side_effect=lambda tensor: tensor.chunk(2)[1]),
        patch.object(
            torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", side_effect=lambda tensor: tensor, create=True
        ),
        patch.object(prepare_finalize.torch_npu, "npu_dynamic_quant", create=True) as quantize,
    ):
        output = layer.prepare(torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(2, 2), quant_type=QuantType.W8A8)
    torch.testing.assert_close(layer.lora_context.gathered_lora_indices, torch.tensor([-1, -1]))
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.pertoken_scale is None
    quantize.assert_not_called()


def test_alltoall_sequence_parallel_prepares_current_token_mapping():
    layer = object.__new__(PrepareAndFinalizeWithAll2All)
    layer.tp_rank = 1
    layer.tp_size = 2
    layer.lora_context = _context([0, 1, -1])
    layer.lora_context.split_lora_indices = torch.tensor([9, 9])
    layer.prepare(torch.ones(2, 3), torch.ones(2, 2), replace_allreduce=True)
    torch.testing.assert_close(layer.lora_context.split_lora_indices, torch.tensor([-1, -1]))


def test_alltoall_exchanges_adapter_ids_instead_of_local_slots():
    ctx = _context([0, 1], torch.tensor([22, 11]))
    ctx.permuted_lora_indices = torch.tensor([1, -1, 0])
    with patch(
        "vllm_ascend.lora.fused_moe.async_all_to_all",
        return_value=(None, torch.tensor([22, 11, -1]), Mock()),
    ) as exchange:
        all2all_lora_indices(ctx, output_splits=[1, 2], input_splits=[2, 1], ep_group="ep")
    torch.testing.assert_close(exchange.call_args.args[0], torch.tensor([11, -1, 22]))
    torch.testing.assert_close(ctx.exchanged_lora_indices, torch.tensor([0, 1, -1]))


@pytest.mark.parametrize(
    ("local_active", "cross_dp", "expected_dtypes"),
    [
        (False, False, [torch.float32, torch.int8, torch.int64]),
        (True, False, [torch.bfloat16, torch.int64]),
        (False, True, [torch.bfloat16, torch.int64]),
        (True, True, [torch.bfloat16, torch.int64]),
    ],
)
def test_quantized_alltoall_protocol_is_uniform_across_dp(local_active, cross_dp, expected_dtypes):
    ctx = _context([0 if local_active else -1], torch.tensor([11]) if cross_dp else None, no_lora=not local_active)
    ctx.permuted_lora_indices = ctx.punica_wrapper.token_lora_indices
    layer = object.__new__(token_dispatcher.TokenDispatcherWithAll2AllV)
    layer.lora_context = ctx
    layer._dispatch_preprocess = lambda hidden, ids: (
        hidden.clone(),
        torch.tensor([0]),
        torch.tensor([1]),
        [1],
        [1],
        torch.tensor([0]),
        hidden.shape,
        hidden.shape,
    )
    layer._dispatch_postprocess = lambda hidden, scale, *args: (hidden, scale, torch.tensor([0]))
    dtypes = []

    def exchange(tensor, *args):
        dtypes.append(tensor.dtype)
        return None, tensor.clone(), Mock()

    inputs = SimpleNamespace(
        hidden_states=torch.ones(1, 2, dtype=torch.bfloat16),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.long),
        quant=SimpleNamespace(
            is_mxfp=False, is_quant=True, dispatch_with_quant=True, quant_type=QuantType.W8A8, get_dst_type=torch.int8
        ),
        routing=SimpleNamespace(pertoken_scale=None),
    )
    with (
        patch.object(token_dispatcher, "get_ep_group", return_value=SimpleNamespace(device_group="ep")),
        patch.object(token_dispatcher, "async_all_to_all", side_effect=exchange),
        patch("vllm_ascend.lora.fused_moe.async_all_to_all", side_effect=exchange),
        patch.object(token_dispatcher, "validate_quant_moe_lora_activation_input"),
        patch.object(
            token_dispatcher.DeviceOperator,
            "npu_dynamic_quant",
            side_effect=lambda tensor, **kwargs: (tensor.to(torch.int8), torch.ones(tensor.shape[0])),
        ),
    ):
        layer.token_dispatch(inputs)
    assert dtypes == expected_dtypes


def test_quantized_expert_with_local_base_tokens_processes_remote_lora():
    ctx = _context([-1], torch.tensor([11]), no_lora=True)
    inputs = SimpleNamespace(quant=SimpleNamespace(is_quant=True), lora_context=ctx)
    output = torch.ones(1, 2)
    quant_method = Mock()
    with patch("vllm_ascend.lora.quant_moe.quant_apply_mlp_with_moe_lora", return_value=(output, None)) as apply:
        actual, _ = moe_mlp.apply_moe_mlp(inputs, quant_method)
    assert actual is output
    apply.assert_called_once_with(mlp_compute_input=inputs, quant_method=quant_method)
    quant_method.supports_fused_activation.assert_not_called()
