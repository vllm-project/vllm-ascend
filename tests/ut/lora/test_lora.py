from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.lora.layers import LoRAMapping
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
    has_moe_lora,
    moe_lora_apply_w2,
    moe_lora_apply_w13,
    postprocess_lora_indices,
    prepare_lora_indices,
    preprocess_lora_indices,
    reset_lora_indices,
    should_disable_all2all_quant_for_lora,
    should_exchange_all2all_lora_indices,
    should_gather_allgather_lora_indices,
    should_use_unfused_quant_moe_for_lora,
    slice_local_lora_indices,
    sync_lora_context,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.quantization.quant_type import QuantType


def _make_base_layer(*, num_local_experts=256, is_act_and_mul=True, shared_experts=None, use_ep=False):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            dp_size=1,
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
    assert out.ep_rank == 0


def test_build_lora_context_propagates_nonzero_ep_rank() -> None:
    base_layer = _make_base_layer(use_ep=True)
    base_layer.moe_config.moe_parallel_config.ep_rank = 3
    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)
    context = SimpleNamespace()
    with patch.object(FusedMoEWithLoRA, "_build_lora_context", return_value=context):
        out = wrapper._build_lora_context()
    assert out.ep_rank == 3


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
    context = SimpleNamespace(split_lora_indices=1, gathered_lora_indices=2, extra=3)
    reset_lora_indices(context)
    assert not hasattr(context, "split_lora_indices")
    assert not hasattr(context, "gathered_lora_indices")
    assert context.extra == 3


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


def test_single_lora_mask_follows_token_mapping() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._token_lora_indices = torch.tensor([0, -1, 0, -1])
    wrapper._single_lora_mask = torch.empty(4, 1, dtype=torch.bfloat16)
    wrapper.indices_len = [4, 0, 0, 0]
    with patch.object(PunicaWrapperBase, "_update_base_metadata"):
        wrapper._update_base_metadata(Mock(), [], 1, 100)
    torch.testing.assert_close(
        wrapper._single_lora_mask,
        torch.tensor([[1], [0], [1], [0]], dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("use_buffer", [False, True])
def test_add_lora_linear_skips_no_lora_batch(use_buffer: bool) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.no_lora = True
    wrapper._wrapper_id = 0
    wrapper.add_shrink = Mock()
    wrapper.add_expand = Mock()
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = torch.randn(2, 5, dtype=torch.bfloat16)
    original = y.clone()
    lora_a = (torch.randn(1, 1, 2, 4, dtype=torch.bfloat16),)
    lora_b = (torch.randn(1, 1, 5, 2, dtype=torch.bfloat16),)
    buffer = (torch.empty(2, 2, dtype=torch.float32),) if use_buffer else None

    with patch("vllm_ascend.lora.punica_npu.lora_linear") as linear:
        wrapper.add_lora_linear(y, x, lora_a, lora_b, 1.0, (5,), buffer=buffer)

    linear.assert_not_called()
    wrapper.add_shrink.assert_not_called()
    wrapper.add_expand.assert_not_called()
    torch.testing.assert_close(y, original)


@pytest.mark.parametrize("add_inputs", [True, False])
def test_packed_single_lora_matmul(add_inputs: bool) -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper._single_lora_mask = torch.tensor([[1], [0], [1]], dtype=torch.bfloat16)
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    y = torch.randn(3, 5, dtype=torch.bfloat16)
    original = y.clone()
    lora_a = tuple(torch.randn(1, 1, 2, 4, dtype=torch.bfloat16) for _ in range(2))
    lora_b = (
        torch.randn(1, 1, 3, 2, dtype=torch.bfloat16),
        torch.randn(1, 1, 2, 2, dtype=torch.bfloat16),
    )
    packed_a = torch.cat(lora_a, dim=2)
    packed_b = torch.zeros(1, 1, 4, 5, dtype=torch.bfloat16)
    packed_b[0, 0, :2, :3] = lora_b[0][0, 0].T
    packed_b[0, 0, 2:, 3:] = lora_b[1][0, 0].T

    wrapper._lora_linear_matmul(y, x, lora_a, lora_b, 0.5, (3, 2), packed_a, packed_b, add_inputs)
    delta = (x @ packed_a[0, 0].T).mul(wrapper._single_lora_mask)
    expected = delta.mul(0.5) @ packed_b[0, 0]
    if add_inputs:
        expected.add_(original)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize(
    ("active", "loaded", "expected"),
    [({0}, {0}, True), ({1}, {0}, False), ({1}, set(), False)],
)
def test_moe_lora_intersects_active_and_loaded_slots(active, loaded, expected):
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=False, active_lora_slots=active),
        layer_lora_slots=loaded,
        use_ep=False,
    )
    assert has_moe_lora(context) is expected


def test_ep_lora_does_not_drop_remote_tokens_on_idle_rank():
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots={0},
        use_ep=True,
    )
    assert has_moe_lora(context)


def test_request_ids_resolve_to_slots_without_treating_base_as_adapter():
    wrapper = object.__new__(PunicaWrapperNPU)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        wrapper.update_metadata(SimpleNamespace(index_mapping=(0, 7, 0)), [7, 11], 2, 100)
        assert wrapper.active_lora_slots == {0}
        assert not wrapper.no_lora
        wrapper.update_metadata(SimpleNamespace(index_mapping=(0, 0)), [7, 11], 2, 100)
        assert wrapper.active_lora_slots == set()
        assert wrapper.no_lora


@pytest.mark.parametrize("is_prefill", [True, False])
def test_metadata_uses_upstream_id_to_slot_conversion(is_prefill):
    wrapper = object.__new__(PunicaWrapperNPU)
    PunicaWrapperBase.__init__(wrapper, 8, 4, torch.device("cpu"))
    wrapper._single_lora_mask = None
    mapping = LoRAMapping((0, 7, 0), (0, 7, 0), is_prefill=is_prefill)
    wrapper.update_metadata(mapping, [7, 11], 2, 100)
    assert wrapper.token_lora_indices.tolist() == [-1, 0, -1]
    assert wrapper.active_lora_slots == {0}
    assert not wrapper.no_lora
    mapping = LoRAMapping((0, 0), (0, 0), is_prefill=is_prefill)
    wrapper.update_metadata(mapping, [7, 11], 2, 100)
    assert wrapper.token_lora_indices.tolist() == [-1, -1]
    assert wrapper.no_lora


def test_ep_replicated_batch_ignores_inactive_loaded_adapter():
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=False, active_lora_slots={1}),
        layer_lora_slots={0},
        use_ep=True,
        replicated_batch=True,
    )
    assert not has_moe_lora(context)


def test_expert_bounds_cannot_address_another_adapter():
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    weights = (torch.ones(2, 4, 1, 1),)
    wrapper.add_lora_fused_moe(
        torch.zeros(4, 1),
        torch.ones(4, 1),
        weights,
        weights,
        expert_ids=torch.tensor([4, -1, 3, 0]),
        token_lora_mapping=torch.tensor([0, 1, 1, 2]),
        adapter_enabled=torch.ones(2, dtype=torch.bool),
    )
    assert wrapper.bgmv_shrink.call_args.args[3].tolist() == [-1, -1, 7, -1]


def test_add_lora_fused_moe_skips_when_adapter_slots_empty():
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.bgmv_shrink = Mock()
    wrapper.bgmv_expand_slice = Mock()
    weights = (torch.ones(1, 4, 1, 1),)
    wrapper.add_lora_fused_moe(
        torch.zeros(2, 1),
        torch.ones(2, 1),
        weights,
        weights,
        expert_ids=torch.tensor([0, 1]),
        token_lora_mapping=torch.tensor([0, -1]),
        adapter_enabled=torch.zeros(0, dtype=torch.bool),
    )
    wrapper.bgmv_shrink.assert_not_called()
    wrapper.bgmv_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("context", "quant_type", "expected_disable", "expected_exchange"),
    [
        (None, None, False, False),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=False, active_lora_slots={0}),
                layer_lora_slots=set(),
                replicated_batch=False,
            ),
            None,
            False,
            False,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
                layer_lora_slots={0},
                replicated_batch=False,
            ),
            None,
            True,
            True,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
                layer_lora_slots={0},
                replicated_batch=True,
            ),
            None,
            False,
            False,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
                layer_lora_slots={0},
                replicated_batch=True,
            ),
            QuantType.W4A8,
            True,
            False,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
                layer_lora_slots={0},
                replicated_batch=True,
            ),
            QuantType.W8A8,
            True,
            False,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
                layer_lora_slots={0},
                replicated_batch=True,
            ),
            QuantType.W4A8MXFP,
            False,
            False,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=False, active_lora_slots={1}),
                layer_lora_slots={0},
                replicated_batch=True,
            ),
            None,
            False,
            False,
        ),
        (
            SimpleNamespace(
                punica_wrapper=SimpleNamespace(no_lora=False, active_lora_slots={0}),
                layer_lora_slots={0},
                replicated_batch=True,
            ),
            None,
            True,
            True,
        ),
    ],
)
def test_should_disable_all2all_quant_for_lora(context, quant_type, expected_disable, expected_exchange):
    assert should_disable_all2all_quant_for_lora(context, quant_type=quant_type) is expected_disable
    assert should_exchange_all2all_lora_indices(context) is expected_exchange


def test_all2all_lora_index_helpers_handle_empty_and_short_tensors() -> None:
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0])),
    )
    prepare_lora_indices(context, num_tokens=3, pad_size=1, tp_size=1, tp_rank=0)
    assert context.split_lora_indices.tolist() == [0, -1, -1, -1]

    context.split_lora_indices = torch.empty((0,), dtype=torch.long)
    with patch("vllm_ascend.lora.fused_moe.torch.argsort") as mock_argsort:
        preprocess_lora_indices(
            context,
            topk_ids=torch.empty((0, 2), dtype=torch.long),
            reversed_permutation_mapping=torch.empty((0,), dtype=torch.int64),
        )
        assert context.permuted_lora_indices.numel() == 0

        context.exchanged_lora_indices = torch.empty((0,), dtype=torch.long)
        postprocess_lora_indices(
            context,
            reversed_permutation_mapping=torch.empty((0,), dtype=torch.int64),
        )
        assert context.exchanged_lora_indices.numel() == 0
    mock_argsort.assert_not_called()


def test_moe_lora_apply_w2_resets_indices_when_empty_rows() -> None:
    context = SimpleNamespace(
        punica_wrapper=Mock(),
        split_lora_indices=torch.tensor([0]),
        permuted_lora_indices=torch.tensor([0]),
        exchanged_lora_indices=torch.tensor([0]),
    )
    moe_lora_apply_w2(
        context,
        down_out=torch.empty((0, 4)),
        silu_out=torch.empty((0, 4)),
        lora_routing=(torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)),
    )
    context.punica_wrapper.add_lora_fused_moe.assert_not_called()
    assert not hasattr(context, "split_lora_indices")
    assert not hasattr(context, "permuted_lora_indices")
    assert not hasattr(context, "exchanged_lora_indices")


def test_should_use_unfused_quant_moe_for_lora() -> None:
    assert not should_use_unfused_quant_moe_for_lora(None, QuantType.W4A8)

    empty_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots=set(),
        use_ep=False,
    )
    assert not should_use_unfused_quant_moe_for_lora(empty_ctx, QuantType.W4A8)
    assert not should_use_unfused_quant_moe_for_lora(empty_ctx, QuantType.W4A8MXFP)

    loaded_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots={0},
        use_ep=False,
    )
    assert should_use_unfused_quant_moe_for_lora(loaded_ctx, QuantType.W4A8)
    assert should_use_unfused_quant_moe_for_lora(loaded_ctx, QuantType.W8A8)
    assert not should_use_unfused_quant_moe_for_lora(loaded_ctx, QuantType.W4A8MXFP)

    with patch("vllm_ascend.lora.fused_moe._EXTRA_CTX", SimpleNamespace(capturing=True)):
        assert should_use_unfused_quant_moe_for_lora(empty_ctx, QuantType.W4A8)


def test_ascend_fused_moe_set_lora_wraps_shape_runtime_error() -> None:
    base_layer = _make_base_layer()
    base_layer.prefix = "model.layers.2.mlp.experts"
    with (
        patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base_layer)
    wrapper.w13_lora_a_stacked = (torch.empty(1, 2, 8, 4096),)
    wrapper.w13_lora_b_stacked = (torch.empty(1, 2, 256, 8),)
    wrapper.w2_lora_a_stacked = (torch.empty(1, 2, 8, 256),)
    wrapper.w2_lora_b_stacked = (torch.empty(1, 2, 4096, 8),)
    lora_a = [torch.empty(2, 8, 5120), torch.empty(2, 8, 256), torch.empty(2, 8, 5120)]
    lora_b = [torch.empty(2, 256, 8), torch.empty(2, 5120, 8), torch.empty(2, 256, 8)]
    with (
        patch.object(FusedMoEWithLoRA, "set_lora", side_effect=RuntimeError("shape mismatch")),
        pytest.raises(
            ValueError,
            match="model.layers.2.mlp.experts.*slot 1.*\\(2, 8, 5120\\).*\\(2, 8, 4096\\).*shape mismatch",
        ),
    ):
        wrapper.set_lora(1, lora_a, lora_b)


def test_should_gather_allgather_lora_indices() -> None:
    assert not should_gather_allgather_lora_indices(None)

    empty_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots=set(),
    )
    assert not should_gather_allgather_lora_indices(empty_ctx)

    loaded_idle_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots={0},
    )
    assert should_gather_allgather_lora_indices(loaded_idle_ctx)

    with patch("vllm_ascend.lora.fused_moe._is_lora_graph_capturing", return_value=True):
        assert should_gather_allgather_lora_indices(empty_ctx)


def test_slice_local_lora_indices_truncates_and_pads_minus_one() -> None:
    context = SimpleNamespace(punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([2, -1, 0, 9])))
    assert slice_local_lora_indices(context, num_tokens=2, pad_size=2).tolist() == [2, -1, -1, -1]
    assert slice_local_lora_indices(context, num_tokens=6).tolist() == [2, -1, 0, 9, -1, -1]


def test_allgather_routing_dp2_ep_translates_experts_and_ignores_inactive_rows() -> None:
    context = SimpleNamespace(
        top_k=2,
        use_ep=True,
        ep_rank=1,
        local_num_experts=2,
        replicated_batch=False,
        gathered_lora_indices=torch.tensor([-1, -1, 0, 1]),
    )
    # EP rank 1 owns global experts [2, 3]. Global routing pairs across 4 gathered tokens:
    # (0,0)->0(-1), (0,1)->2(0), (1,0)->3(1), (1,1)->1(-1), (2,0)->2(2), (2,1)->0(-1), (3,0)->3(3), (3,1)->2(4).
    topk_ids = torch.tensor([[0, 2], [3, 1], [2, 0], [3, 2]])
    expanded_row_idx = torch.tensor([-1, 0, 1, -1, 2, -1, 3, 4], dtype=torch.int32)

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert expert_ids[:5].tolist() == [0, 1, 0, 1, 0]
    assert lora_slots.tolist() == [-1, -1, 0, 1, 1, -1, -1, -1]


def test_allgather_routing_dp_gt1_requires_gathered_indices() -> None:
    context = SimpleNamespace(
        top_k=1,
        use_ep=True,
        replicated_batch=False,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0])),
    )
    with pytest.raises(AssertionError, match="gathered_lora_indices"):
        _recover_moe_lora_routing_allgather(context, torch.tensor([0]), torch.tensor([[0]]))


@pytest.mark.parametrize(("base_rank", "adapter_rank"), [(0, 1), (1, 0)])
def test_graph_allgather_dp2_base_and_adapter_ranks(base_rank: int, adapter_rank: int) -> None:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig

    from vllm_ascend import ascend_forward_context as afc
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.device.hardware_profile import get_hardware_profile
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
    from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
    from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=False,
            hf_text_config=SimpleNamespace(top_k_experts=2, hidden_size=6),
            get_num_experts=lambda: 4,
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=2,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        lora_config=SimpleNamespace(max_loras=1),
    )
    with (
        patch.object(afc, "is_moe_model", return_value=True),
        patch.object(afc, "get_mc2_tokens_capacity", return_value=128),
        patch.object(afc, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3)),
        patch.object(afc, "get_ep_group", return_value=SimpleNamespace(world_size=2)),
        patch.object(afc, "get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=0)),
    ):
        assert afc.select_moe_comm_method(2, vllm_config) == MoECommType.ALLGATHER

    torch.manual_seed(0)
    hidden_size, inter_size, lora_rank = 6, 3, 2
    num_experts, num_local_experts, top_k = 4, 2, 2
    max_tokens_across_dp = 3

    w13_base = torch.randn(num_experts, hidden_size, inter_size * 2) * 0.2
    w2_base = torch.randn(num_experts, inter_size, hidden_size) * 0.2
    w13_lora_a = torch.randn(1, num_experts, lora_rank, hidden_size) * 0.3
    w13_lora_b = torch.randn(1, num_experts, inter_size * 2, lora_rank) * 0.3
    w2_lora_a = torch.randn(1, num_experts, lora_rank, inter_size) * 0.3
    w2_lora_b = torch.randn(1, num_experts, hidden_size, lora_rank) * 0.3

    per_rank_tokens = {
        base_rank: torch.randn(2, hidden_size),
        adapter_rank: torch.randn(3, hidden_size),
    }
    # Force routing across both local and remote experts for every token so
    # each DP rank hits both rank 0's experts [0, 1] and rank 1's experts [2, 3].
    per_rank_router = {
        0: torch.tensor([[5.0, 1.0, 4.0, 0.5], [0.5, 4.0, 5.0, 1.0]]),
        1: torch.tensor([[4.5, 0.2, 5.0, 0.1], [0.2, 4.0, 0.5, 5.0], [5.0, 3.0, 0.2, 0.1]]),
    }
    if base_rank == 1:
        per_rank_router = {
            0: torch.tensor([[4.5, 0.2, 5.0, 0.1], [0.2, 4.0, 0.5, 5.0], [5.0, 3.0, 0.2, 0.1]]),
            1: torch.tensor([[5.0, 1.0, 4.0, 0.5], [0.5, 4.0, 5.0, 1.0]]),
        }

    def fake_bgmv_shrink(x, a_flat, shrink_out, combined_idx, scale):
        for row, idx in enumerate(combined_idx.tolist()):
            if idx >= 0:
                shrink_out[row] = scale * (x[row].float() @ a_flat[idx].transpose(0, 1).float())

    def fake_bgmv_expand_slice(delta, b_flat, y, combined_idx, offset, out_size, add_inputs=True):
        del add_inputs
        for row, idx in enumerate(combined_idx.tolist()):
            if idx >= 0:
                y[row, offset : offset + out_size] += (delta[row] @ b_flat[idx].transpose(0, 1).float()).to(y.dtype)

    def make_punica(token_indices: list[int], *, num_tokens: int, no_lora: bool):
        wrapper = object.__new__(PunicaWrapperNPU)
        wrapper.no_lora = no_lora
        wrapper.active_lora_slots = set() if no_lora else {0}
        wrapper.indices_len = [num_tokens]
        wrapper._token_lora_indices = torch.tensor(token_indices, dtype=torch.long)
        wrapper.bgmv_shrink = fake_bgmv_shrink
        wrapper.bgmv_expand_slice = fake_bgmv_expand_slice
        return wrapper

    punica_by_rank = {
        base_rank: make_punica([-1, -1, 0, 0], num_tokens=2, no_lora=True),
        adapter_rank: make_punica([0, 0, 0, -1], num_tokens=3, no_lora=False),
    }
    contexts = {}
    prepares = {}
    dispatchers = {}
    for rank in range(2):
        moe_config = MagicMock(spec=FusedMoEConfig)
        moe_config.dp_size = 2
        moe_config.pcp_size = 1
        moe_config.tp_size = 1
        moe_config.ep_size = 2
        moe_config.num_experts = num_experts
        moe_config.num_local_experts = num_local_experts
        moe_config.experts_per_token = top_k
        moe_config.is_sequence_parallel = False
        contexts[rank] = SimpleNamespace(
            punica_wrapper=punica_by_rank[rank],
            use_ep=True,
            ep_rank=rank,
            local_num_experts=num_local_experts,
            top_k=top_k,
            layer_lora_slots={0},
            replicated_batch=False,
            adapter_enabled=torch.tensor([True]),
            w13_lora_a_stacked=(w13_lora_a[:, rank * num_local_experts : (rank + 1) * num_local_experts],),
            w13_lora_b_stacked=(w13_lora_b[:, rank * num_local_experts : (rank + 1) * num_local_experts],),
            w2_lora_a_stacked=(w2_lora_a[:, rank * num_local_experts : (rank + 1) * num_local_experts],),
            w2_lora_b_stacked=(w2_lora_b[:, rank * num_local_experts : (rank + 1) * num_local_experts],),
        )
        prepares[rank] = PrepareAndFinalizeWithAllGather(moe_config)
        prepares[rank].set_lora_context(contexts[rank])
        dispatchers[rank] = TokenDispatcherWithAllGather(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
        )
        dispatchers[rank].set_lora_context(contexts[rank])

    # Build the synchronized 2-rank all_gather outputs and assert each rank's prepare
    # passes the exact expected local padded tensors (with -1 padding on LoRA indices).
    padded_hidden = {}
    padded_router = {}
    padded_lora = {}
    for rank in range(2):
        h = per_rank_tokens[rank]
        r = per_rank_router[rank]
        pad_size = max_tokens_across_dp - h.shape[0]
        li = slice_local_lora_indices(contexts[rank], num_tokens=h.shape[0], pad_size=pad_size, device=h.device)
        padded_hidden[rank] = torch.nn.functional.pad(h, (0, 0, 0, pad_size)) if pad_size > 0 else h
        padded_router[rank] = torch.nn.functional.pad(r, (0, 0, 0, pad_size)) if pad_size > 0 else r
        padded_lora[rank] = li
    global_hidden = torch.cat([padded_hidden[0], padded_hidden[1]], dim=0)
    global_router = torch.cat([padded_router[0], padded_router[1]], dim=0)
    global_lora = torch.cat([padded_lora[0], padded_lora[1]], dim=0)

    def _dp_all_gather(rank: int, tensor: torch.Tensor, _dim: int) -> torch.Tensor:
        if tensor.ndim == 1:
            assert torch.equal(tensor, padded_lora[rank])
            return global_lora
        if tensor.shape[-1] == hidden_size:
            assert torch.equal(tensor, padded_hidden[rank])
            return global_hidden
        assert torch.equal(tensor, padded_router[rank])
        return global_router

    prepared = {}
    for rank in range(2):
        prepares[rank].moe_config.dp_group = SimpleNamespace(
            all_gather=lambda tensor, dim, rank=rank: _dp_all_gather(rank, tensor, dim)
        )
        with patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", SimpleNamespace(max_tokens_across_dp=3)):
            prepared[rank] = prepares[rank].prepare(per_rank_tokens[rank], per_rank_router[rank])

    assert contexts[0].gathered_lora_indices.tolist() == global_lora.tolist()
    assert contexts[1].gathered_lora_indices.tolist() == global_lora.tolist()

    def cpu_init_routing(hidden_states, topk_ids, *, active_expert_range, **_kwargs):
        first, last = active_expert_range
        num_tokens = hidden_states.shape[0]
        expanded = torch.full((num_tokens * top_k,), -1, dtype=torch.int32)
        rows_per_expert = []
        dst = 0
        for local_e in range(first, last):
            count = 0
            for t in range(num_tokens):
                for k in range(top_k):
                    if int(topk_ids[t, k]) == local_e:
                        expanded[t * top_k + k] = dst
                        rows_per_expert.append(hidden_states[t])
                        dst += 1
                        count += 1
            rows_per_expert.append(torch.empty((0, hidden_size))) if count == 0 else None
        active_rows = [r for r in rows_per_expert if r is not None and r.ndim == 1]
        out = torch.full((num_tokens * top_k, hidden_size), 999.0, dtype=hidden_states.dtype)
        if active_rows:
            stacked = torch.stack(active_rows, dim=0)
            out[: stacked.shape[0]] = stacked
        expert_counts = torch.tensor(
            [int((topk_ids == local_e).sum()) for local_e in range(first, last)],
            dtype=torch.int32,
        )
        return out, expanded, expert_counts, None

    def cpu_unpermute(*, permuted_tokens, sorted_indices, probs, **_kwargs):
        num_tokens = probs.shape[0]
        flat_idx = sorted_indices.reshape(num_tokens, top_k)
        out = torch.zeros((num_tokens, hidden_size), dtype=permuted_tokens.dtype)
        for t in range(num_tokens):
            for k in range(top_k):
                idx = int(flat_idx[t, k])
                if idx >= 0:
                    out[t] += probs[t, k] * permuted_tokens[idx]
        return out

    partial_outputs = {}
    expert_maps = {
        0: torch.tensor([0, 1, -1, -1]),
        1: torch.tensor([-1, -1, 0, 1]),
    }
    topk_weights, topk_ids = torch.topk(global_router, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    for rank in range(2):
        assert has_moe_lora(contexts[rank])
        td_input = MoETokenDispatchInput(
            hidden_states=prepared[rank].hidden_states,
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.to(torch.int32),
            routing=MoeRouterInput(
                expert_map=expert_maps[rank],
                global_redundant_expert_num=0,
                mc2_mask=None,
                apply_router_weight_on_input=False,
            ),
            quant=MoEQuantParams(),
        )
        with (
            patch(
                "vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group",
                return_value=SimpleNamespace(rank_in_group=rank),
            ),
            patch(
                "vllm_ascend.ops.fused_moe.token_dispatcher.DeviceOperator.npu_moe_init_routing",
                side_effect=cpu_init_routing,
            ),
            patch(
                "vllm_ascend.ops.fused_moe.token_dispatcher.DeviceOperator.npu_moe_token_unpermute",
                side_effect=cpu_unpermute,
            ),
        ):
            dispatched = dispatchers[rank].token_dispatch(td_input)
            lora_routing = _recover_moe_lora_routing_allgather(
                contexts[rank],
                dispatched.combine_metadata.expanded_row_idx,
                topk_ids,
            )
            expert_per_row, lora_per_row = lora_routing
            active_rows = int(dispatched.group_list.sum().item())
            sorted_h = dispatched.hidden_states
            gate_up_out = torch.zeros((sorted_h.shape[0], inter_size * 2), dtype=sorted_h.dtype)
            local_offset = rank * num_local_experts
            for row in range(active_rows):
                global_e = local_offset + int(expert_per_row[row])
                gate_up_out[row] = sorted_h[row] @ w13_base[global_e]

            moe_lora_apply_w13(
                contexts[rank],
                gate_up_out=gate_up_out,
                hidden_states=sorted_h,
                lora_routing=lora_routing,
            )
            gate, up = gate_up_out.chunk(2, dim=-1)
            silu_out = torch.nn.functional.silu(gate) * up
            down_out = torch.zeros_like(sorted_h)
            for row in range(active_rows):
                global_e = local_offset + int(expert_per_row[row])
                down_out[row] = silu_out[row] @ w2_base[global_e]

            moe_lora_apply_w2(
                contexts[rank],
                down_out=down_out,
                silu_out=silu_out,
                lora_routing=lora_routing,
            )
            partial_outputs[rank] = dispatchers[rank].token_combine(down_out, dispatched.combine_metadata)
            assert not hasattr(contexts[rank], "gathered_lora_indices")
            assert not (lora_per_row[active_rows:] >= 0).any()

    summed_global = partial_outputs[0] + partial_outputs[1]
    finalized = {}
    with patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dp_group") as mock_dp_group:
        for rank in range(2):
            mock_dp_group.return_value = SimpleNamespace(
                reduce_scatter=lambda _hidden, _dim, rank=rank: summed_global.chunk(2, dim=0)[rank]
            )
            finalized[rank] = prepares[rank].finalize(partial_outputs[rank], reduce_results=True)

    # Pure-PyTorch global reference: rank tokens routed with or without adapter 0.
    offset = 0
    for rank in range(2):
        tokens = per_rank_tokens[rank]
        is_adapter_rank = rank == adapter_rank
        expected = torch.zeros_like(tokens)
        for local_idx in range(tokens.shape[0]):
            g_idx = offset + local_idx
            x = tokens[local_idx]
            for k in range(top_k):
                e = int(topk_ids[g_idx, k])
                prob = topk_weights[g_idx, k]
                gu = x @ w13_base[e]
                if is_adapter_rank:
                    gu = gu + (x @ w13_lora_a[0, e].T) @ w13_lora_b[0, e].T
                g, u = gu.chunk(2, dim=-1)
                act = torch.nn.functional.silu(g) * u
                dn = act @ w2_base[e]
                if is_adapter_rank:
                    dn = dn + (act @ w2_lora_a[0, e].T) @ w2_lora_b[0, e].T
                expected[local_idx] += prob * dn
        torch.testing.assert_close(finalized[rank], expected, atol=1e-5, rtol=1e-5)
        offset += max_tokens_across_dp
