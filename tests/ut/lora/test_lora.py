from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import vllm
from torch import nn
from vllm.lora.layers import MergedColumnParallelLinearWithLoRA, MergedQKVParallelLinearWithLoRA
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoEWithLoRA,
    _assert_ascend_moe_lora_supported,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    all2all_lora_indices,
    has_lora,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
    postprocess_lora_indices,
    prepare_lora_indices,
    preprocess_lora_indices,
    reset_lora_indices,
    sync_lora_context,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA,
    AscendMergedQKVParallelLinearWithLoRA,
    _PackedLoRAAWeightsMixin,
    refresh_all_lora_classes,
)
from vllm_ascend.utils import AscendDeviceType


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_single_lora_slot_enabled_on_a2_and_a3(device_type: AscendDeviceType) -> None:
    lora_config = SimpleNamespace(
        max_lora_rank=8,
        max_loras=1,
        fully_sharded_loras=False,
        lora_dtype=torch.bfloat16,
    )

    with (
        patch.object(PunicaWrapperBase, "__init__", return_value=None),
        patch("vllm_ascend.lora.punica_npu.refresh_all_lora_classes"),
        patch("vllm_ascend.lora.punica_npu.get_ascend_device_type", return_value=device_type),
    ):
        wrapper = PunicaWrapperNPU(16, 4, "cpu", lora_config=lora_config)

    assert wrapper._single_lora_slot
    assert wrapper._single_lora_mask.shape == (16, 1)


def _make_base_layer(*, num_local_experts=256, is_act_and_mul=True, shared_experts=None, use_ep=False):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim=4096,
            num_local_experts=num_local_experts,
            num_experts=num_local_experts,
            intermediate_size_per_partition=256,
            experts_per_token=8,
            moe_parallel_config=SimpleNamespace(tp_size=8, tp_rank=3, ep_rank=0, use_ep=use_ep),
            is_act_and_mul=is_act_and_mul,
        ),
        _shared_experts=shared_experts,
    )


def test_ascend_fused_moe_lora_initializes_skipped_upstream_fields() -> None:
    shared_experts = torch.nn.Module()
    base_layer = _make_base_layer(shared_experts=shared_experts)

    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)

    assert wrapper._lora_stream is None
    assert wrapper._events is None
    assert wrapper.enable_moe_shared_loras is False
    assert wrapper._shared_experts is shared_experts
    assert wrapper.n_slices == 256 * 3


def test_ascend_fused_moe_lora_omits_shared_experts_attr_when_absent() -> None:
    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(_make_base_layer(is_act_and_mul=False))

    assert not hasattr(wrapper, "_shared_experts")
    assert wrapper._w13_slices == 1
    assert wrapper.n_slices == 256 * 2


def test_set_mapping_publishes_context_on_base_layer() -> None:
    base_layer = _make_base_layer()
    base_layer.set_lora_context = Mock()
    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)
    context = object()
    with (
        patch.object(BaseLayerWithLoRA, "set_mapping") as set_mapping,
        patch.object(wrapper, "_build_lora_context", return_value=context),
    ):
        wrapper.set_mapping("punica")

    set_mapping.assert_called_once_with(wrapper, "punica")
    base_layer.set_lora_context.assert_called_once_with(context)


def test_build_lora_context_propagates_ep_flag() -> None:
    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(_make_base_layer(use_ep=True))
    context = SimpleNamespace()
    with patch.object(FusedMoEWithLoRA, "_build_lora_context", return_value=context):
        out = wrapper._build_lora_context()
    assert out is context
    assert out.use_ep is True


def test_moe_lora_apply_uses_adapter_enabled() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(
        punica_wrapper=punica_wrapper,
        w13_lora_a_stacked="w13_a",
        w13_lora_b_stacked="w13_b",
        w2_lora_a_stacked="w2_a",
        w2_lora_b_stacked="w2_b",
        adapter_enabled="all_enabled",
        split_lora_indices=torch.tensor([0]),
        permuted_lora_indices=torch.tensor([0]),
        exchanged_lora_indices=torch.tensor([0]),
    )
    routing = (torch.tensor([0]), torch.tensor([0]))

    moe_lora_apply_w13(
        context,
        gate_up_out="gate_up_out",
        hidden_states="hidden_states",
        lora_routing=routing,
    )
    moe_lora_apply_w2(
        context,
        down_out="down_out",
        silu_out="silu_out",
        lora_routing=routing,
    )

    calls = punica_wrapper.add_lora_fused_moe.call_args_list
    assert calls[0].kwargs["adapter_enabled"] == "all_enabled"
    assert calls[1].kwargs["adapter_enabled"] == "all_enabled"
    assert not hasattr(context, "split_lora_indices")
    assert not hasattr(context, "permuted_lora_indices")
    assert not hasattr(context, "exchanged_lora_indices")


def test_moe_lora_apply_skips_empty_ep_rank() -> None:
    punica_wrapper = Mock()
    context = SimpleNamespace(punica_wrapper=punica_wrapper)
    empty = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))

    moe_lora_apply_w13(context, gate_up_out="g", hidden_states="h", lora_routing=empty)
    moe_lora_apply_w2(context, down_out="d", silu_out="s", lora_routing=empty)

    punica_wrapper.add_lora_fused_moe.assert_not_called()


def test_allgather_routing_preserves_multi_adapter_and_base_mapping() -> None:
    context = SimpleNamespace(
        top_k=2,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0, -1, 1])),
    )
    topk_ids = torch.tensor([[1, 0], [0, 1], [1, 1]])
    # Original flat rows [0..5] land at these expert-sorted positions.
    expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert torch.equal(expert_ids, torch.tensor([0, 0, 1, 1, 1, 1]))
    assert torch.equal(lora_slots, torch.tensor([0, -1, 0, -1, 1, 1]))


def test_all2all_routing_uses_local_experts_and_exchanged_adapters() -> None:
    context = SimpleNamespace(
        local_num_experts=3,
        exchanged_lora_indices=torch.tensor([1, -1, 0, 2]),
    )

    expert_ids, lora_slots = _recover_moe_lora_routing_all2all(
        context,
        group_list=torch.tensor([2, 0, 2]),
    )

    assert torch.equal(expert_ids, torch.tensor([0, 0, 2, 2]))
    assert torch.equal(lora_slots, torch.tensor([1, -1, 0, 2]))


def test_all2all_routing_requires_exchanged_indices() -> None:
    with pytest.raises(AssertionError, match="exchanged_lora_indices"):
        _recover_moe_lora_routing_all2all(SimpleNamespace(local_num_experts=1), group_list=torch.tensor([1]))


def test_all2all_routing_rejects_misaligned_metadata() -> None:
    context = SimpleNamespace(
        local_num_experts=2,
        exchanged_lora_indices=torch.tensor([0]),
    )
    with pytest.raises(AssertionError, match="misaligned"):
        _recover_moe_lora_routing_all2all(context, group_list=torch.tensor([1, 1]))


def test_has_lora_follows_batch_metadata() -> None:
    assert not has_lora(None)
    assert not has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True)))
    assert has_lora(SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False)))


def test_reset_lora_indices_drops_only_present_fields() -> None:
    context = SimpleNamespace(split_lora_indices=1, extra=2)
    reset_lora_indices(context)
    assert not hasattr(context, "split_lora_indices")
    assert context.extra == 2


def test_prepare_lora_indices_pads_inactive_slots() -> None:
    context = SimpleNamespace(punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([3, 1, 0, 9, 9])))
    prepare_lora_indices(context, num_tokens=3, pad_size=2, tp_size=1, tp_rank=0)
    assert torch.equal(context.split_lora_indices, torch.tensor([3, 1, 0, -1, -1]))


def test_prepare_lora_indices_tp_splits_local_rank() -> None:
    context = SimpleNamespace(punica_wrapper=SimpleNamespace(token_lora_indices=torch.arange(8)))
    prepare_lora_indices(context, num_tokens=8, pad_size=0, tp_size=4, tp_rank=2)
    assert torch.equal(context.split_lora_indices, torch.tensor([4, 5]))


def test_preprocess_lora_indices_skips_missing_split() -> None:
    context = SimpleNamespace()
    preprocess_lora_indices(
        context,
        topk_ids=torch.zeros(2, 2, dtype=torch.long),
        reversed_permutation_mapping=torch.arange(4),
    )
    assert not hasattr(context, "permuted_lora_indices")


def test_preprocess_lora_indices_repeats_and_permutes() -> None:
    context = SimpleNamespace(split_lora_indices=torch.tensor([0, 1]))
    reversed_mapping = torch.tensor([2, 0, 3, 1])
    preprocess_lora_indices(context, topk_ids=torch.zeros(2, 2), reversed_permutation_mapping=reversed_mapping)
    assert torch.equal(context.permuted_lora_indices, torch.tensor([0, 1, 0, 1]))


def test_postprocess_lora_indices_skips_missing_exchange() -> None:
    context = SimpleNamespace()
    postprocess_lora_indices(context, reversed_permutation_mapping=torch.arange(2))
    assert not hasattr(context, "exchanged_lora_indices")


def test_postprocess_lora_indices_reorders_exchanged_slots() -> None:
    context = SimpleNamespace(exchanged_lora_indices=torch.tensor([10, 20, 30, 40]))
    postprocess_lora_indices(context, reversed_permutation_mapping=torch.tensor([2, 0, 3, 1]))
    assert torch.equal(context.exchanged_lora_indices, torch.tensor([20, 40, 10, 30]))


def test_all2all_lora_indices_skips_missing_permuted() -> None:
    context = SimpleNamespace()
    all2all_lora_indices(context, output_splits=None, input_splits=None, ep_group=None)
    assert not hasattr(context, "exchanged_lora_indices")


def test_all2all_lora_indices_restores_dtype_after_exchange() -> None:
    context = SimpleNamespace(permuted_lora_indices=torch.tensor([1, 2], dtype=torch.int32))
    exchanged = torch.tensor([7, 8], dtype=torch.int64)
    handle = Mock()
    with patch(
        "vllm_ascend.lora.fused_moe.async_all_to_all",
        return_value=(None, exchanged, handle),
    ) as all_to_all:
        all2all_lora_indices(context, output_splits="out", input_splits="in", ep_group="ep")

    all_to_all.assert_called_once_with(context.permuted_lora_indices, "out", "in", "ep")
    handle.wait.assert_called_once()
    assert torch.equal(context.exchanged_lora_indices, torch.tensor([7, 8], dtype=torch.int32))
    assert context.exchanged_lora_indices.dtype == torch.int32


def test_sync_lora_context_updates_available_setters() -> None:
    comm = SimpleNamespace()
    comm.set_lora_context = Mock()
    quant = SimpleNamespace()
    quant.set_lora_context = Mock()
    extra = SimpleNamespace(moe_comm_method=object())
    with patch("vllm_ascend.lora.fused_moe._EXTRA_CTX", extra):
        sync_lora_context(object(), "ctx")
    extra.moe_comm_method = comm
    with patch("vllm_ascend.lora.fused_moe._EXTRA_CTX", extra):
        sync_lora_context(quant, "ctx")
    comm.set_lora_context.assert_called_once_with("ctx")
    quant.set_lora_context.assert_called_once_with("ctx")


def test_assert_rejects_dynamic_eplb_and_fused_mc2() -> None:
    with pytest.raises(AssertionError, match="dynamic EPLB"):
        _assert_ascend_moe_lora_supported(SimpleNamespace(dynamic_eplb=True, _shared_experts=None))
    with (
        patch(
            "vllm_ascend.lora.fused_moe.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=1),
        ),
        pytest.raises(AssertionError, match="FusedMC2"),
    ):
        _assert_ascend_moe_lora_supported(SimpleNamespace(dynamic_eplb=False, _shared_experts=None))


def test_assert_warns_once_for_shared_experts() -> None:
    with (
        patch(
            "vllm_ascend.lora.fused_moe.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=0),
        ),
        patch("vllm_ascend.lora.fused_moe.logger.warning_once") as warn,
    ):
        _assert_ascend_moe_lora_supported(SimpleNamespace(dynamic_eplb=False, _shared_experts=object()))
    warn.assert_called_once()
    assert "shared_experts" in warn.call_args.args[0]


@pytest.mark.parametrize(
    ("index_mapping", "expected_no_lora"),
    [((0, 0), True), ((0, 1), False), ((2, 0), False)],
)
def test_decode_metadata_refreshes_no_lora(index_mapping, expected_no_lora) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    mapping = SimpleNamespace(index_mapping=index_mapping)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(mapping, [], 2, 100)
    assert wrapper.no_lora is expected_no_lora


def test_single_lora_mask_is_refreshed_with_metadata() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._token_lora_indices = torch.tensor([0, -1, 0, -1])
    wrapper._single_lora_mask = torch.empty(4, 1, dtype=torch.bfloat16)
    wrapper.indices_len = [4, 0, 0, 0]

    with patch.object(PunicaWrapperBase, "_update_base_metadata"):
        PunicaWrapperNPU._update_base_metadata(wrapper, Mock(), [], 1, 100)

    torch.testing.assert_close(
        wrapper._single_lora_mask,
        torch.tensor([[1], [0], [1], [0]], dtype=torch.bfloat16),
    )


def test_single_lora_linear_masks_base_rows(add_inputs: bool) -> None:
    token_indices = torch.tensor([0, -1, 0, -1, 0])
    adapter_mask = token_indices.eq(0).unsqueeze(1).to(torch.bfloat16)
    wrapper: Any = SimpleNamespace(
        _single_lora_slot=True,
        _single_lora_mask=adapter_mask,
    )
    x = torch.randn(5, 6, dtype=torch.bfloat16)
    y = torch.randn(5, 7, dtype=torch.bfloat16)
    original_y = y.clone()
    lora_a = (torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),)
    lora_b = (torch.randn(1, 1, 7, 3, dtype=torch.bfloat16),)
    scale = 0.5

    applied = PunicaWrapperNPU._apply_single_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        scale,
        (7,),
        add_inputs=add_inputs,
    )

    delta = torch.matmul(
        torch.matmul(x, lora_a[0][0, 0].transpose(0, 1)),
        lora_b[0][0, 0].transpose(0, 1),
    )
    delta.mul_(token_indices.eq(0).unsqueeze(1))
    expected = original_y.add(delta, alpha=scale) if add_inputs else delta.mul(scale)
    assert applied
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_single_lora_linear_packed_slices(add_inputs: bool, scale: float) -> None:
    token_indices = torch.tensor([0, -1, 0, -1])
    adapter_mask = token_indices.eq(0).unsqueeze(1).to(torch.bfloat16)
    wrapper: Any = SimpleNamespace(
        _single_lora_slot=True,
        _single_lora_mask=adapter_mask,
    )
    x = torch.randn(4, 6, dtype=torch.bfloat16)
    y = torch.randn(4, 7, dtype=torch.bfloat16)
    original_y = y.clone()
    lora_a = (
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
    )
    lora_b = (
        torch.randn(1, 1, 4, 3, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 3, dtype=torch.bfloat16),
    )
    applied = PunicaWrapperNPU._apply_single_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        scale,
        (4, 3),
        add_inputs=add_inputs,
    )

    deltas = []
    for a_weight, b_weight in zip(lora_a, lora_b, strict=True):
        shrink = torch.matmul(x, a_weight[0, 0].transpose(0, 1))
        shrink.mul_(adapter_mask)
        deltas.append(torch.matmul(shrink, b_weight[0, 0].transpose(0, 1)))
    delta = torch.cat(deltas, dim=1)
    expected = original_y.add(delta, alpha=scale) if add_inputs else delta.mul(scale)
    assert applied
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_single_lora_linear_uses_prepacked_a(add_inputs: bool, scale: float) -> None:
    adapter_mask = torch.tensor([[1], [0], [1], [0]], dtype=torch.bfloat16)
    wrapper: Any = SimpleNamespace(
        _single_lora_slot=True,
        _single_lora_mask=adapter_mask,
    )
    x = torch.randn(4, 6, dtype=torch.bfloat16)
    y = torch.randn(4, 7, dtype=torch.bfloat16)
    original_y = y.clone()
    lora_a = (
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 6, dtype=torch.bfloat16),
    )
    packed_lora_a = torch.cat(lora_a, dim=2)
    lora_b = (
        torch.randn(1, 1, 4, 3, dtype=torch.bfloat16),
        torch.randn(1, 1, 3, 3, dtype=torch.bfloat16),
    )
    applied = PunicaWrapperNPU._apply_single_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        scale,
        (4, 3),
        packed_lora_a=packed_lora_a,
        add_inputs=add_inputs,
    )

    shrink = torch.matmul(x, packed_lora_a[0, 0].transpose(0, 1))
    shrink.mul_(adapter_mask)
    delta = torch.cat(
        (
            torch.matmul(shrink[:, :3], lora_b[0][0, 0].transpose(0, 1)),
            torch.matmul(shrink[:, 3:], lora_b[1][0, 0].transpose(0, 1)),
        ),
        dim=1,
    )
    expected = original_y.add(delta, alpha=scale) if add_inputs else delta.mul(scale)
    assert applied
    torch.testing.assert_close(y, expected)


def test_packed_lora_wrappers_extend_only_non_sharded_merged_layers() -> None:
    assert AscendMergedColumnParallelLinearWithLoRA.__mro__[:3] == (
        AscendMergedColumnParallelLinearWithLoRA,
        _PackedLoRAAWeightsMixin,
        MergedColumnParallelLinearWithLoRA,
    )
    assert MergedQKVParallelLinearWithLoRA in AscendMergedQKVParallelLinearWithLoRA.__mro__
    assert all("Sharded" not in base.__name__ for base in AscendMergedQKVParallelLinearWithLoRA.__mro__)


def test_merged_column_packed_wrapper_requires_single_adapter_slot(max_loras: int, expected: bool) -> None:
    lora_config: Any = SimpleNamespace(max_loras=max_loras, fully_sharded_loras=False)

    with patch("vllm_ascend.lora.utils.maybe_get_oot_by_class", return_value=nn.Linear):
        can_replace = AscendMergedColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=nn.Linear(2, 2),
            lora_config=lora_config,
            packed_modules_list=["gate", "up"],
            model_config=None,
        )

    assert can_replace is expected


def test_qkv_packed_wrapper_requires_single_adapter_slot(max_loras: int, expected: bool) -> None:
    class FakeAscendQKVParallelLinear(nn.Module):
        pass

    lora_config: Any = SimpleNamespace(max_loras=max_loras, fully_sharded_loras=False)
    source_layer = FakeAscendQKVParallelLinear()

    with patch("vllm_ascend.lora.utils.AscendQKVParallelLinear", FakeAscendQKVParallelLinear):
        can_replace = AscendMergedQKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=["q", "k", "v"],
            model_config=None,
        )

    assert can_replace is expected


def test_refresh_lora_classes_prioritizes_packed_wrappers() -> None:
    original_classes = vllm.lora.utils._all_lora_classes
    ascend_classes = (
        AscendMergedColumnParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    expected_count = len(ascend_classes) + sum(cls not in ascend_classes for cls in original_classes)
    with patch.object(vllm.lora.utils, "_all_lora_classes", original_classes):
        refresh_all_lora_classes()
        refresh_all_lora_classes()
        assert vllm.lora.utils._all_lora_classes[:2] == (
            AscendMergedColumnParallelLinearWithLoRA,
            AscendMergedQKVParallelLinearWithLoRA,
        )
        assert len(vllm.lora.utils._all_lora_classes) == expected_count


def test_packed_lora_a_weights_follow_set_and_reset_lifecycle() -> None:
    layer: Any = object.__new__(AscendMergedColumnParallelLinearWithLoRA)
    nn.Module.__init__(layer)
    layer.n_slices = 2
    layer.input_size = 4
    layer.device = torch.device("cpu")
    layer.lora_a_stacked = (
        torch.ones(1, 1, 2, 4),
        torch.full((1, 1, 2, 4), 2.0),
    )
    lora_config: Any = SimpleNamespace(lora_dtype=torch.float32)

    with patch.object(MergedColumnParallelLinearWithLoRA, "create_lora_weights"):
        layer.create_lora_weights(1, lora_config)

    assert layer.lora_a_packed.shape == (1, 1, 4, 4)
    with patch.object(MergedColumnParallelLinearWithLoRA, "set_lora"):
        layer.set_lora(0, [], [])
    torch.testing.assert_close(layer.lora_a_packed[0, 0, :2], layer.lora_a_stacked[0][0, 0])
    torch.testing.assert_close(layer.lora_a_packed[0, 0, 2:], layer.lora_a_stacked[1][0, 0])

    with patch.object(MergedColumnParallelLinearWithLoRA, "reset_lora"):
        layer.reset_lora(0)
    assert not torch.count_nonzero(layer.lora_a_packed)


def test_non_homogeneous_prefill_linear_falls_back() -> None:
    wrapper: Any = SimpleNamespace(
        no_lora=False,
        _single_lora_slot=False,
        add_shrink=Mock(),
        add_expand=Mock(),
    )
    wrapper._apply_single_lora_linear = MethodType(PunicaWrapperNPU._apply_single_lora_linear, wrapper)
    x = torch.randn(2, 4)
    y = torch.randn(2, 6)
    lora_a = (torch.randn(1, 1, 2, 4),)
    lora_b = (torch.randn(1, 1, 6, 2),)
    buffer = (torch.empty(2, 2),)

    PunicaWrapperNPU.add_lora_linear(
        wrapper,
        y,
        x,
        lora_a,
        lora_b,
        1.0,
        (6,),
        buffer=buffer,
    )

    wrapper.add_shrink.assert_called_once_with(buffer, x, lora_a, 1.0)
    wrapper.add_expand.assert_called_once_with(y, buffer, lora_b, (6,), add_inputs=True)
