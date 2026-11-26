from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe import MoERunner

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
    _assert_ascend_moe_lora_supported,
    _recover_moe_lora_routing_all2all,
    _recover_moe_lora_routing_allgather,
    all2all_lora_indices,
    assert_moe_lora_dp1_supported,
    has_collective_moe_lora,
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


def _make_base_layer(*, num_local_experts=256, is_act_and_mul=True, shared_experts=None, use_ep=False, dp_size=1):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            dp_size=dp_size,
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


def test_assert_rejects_eplb_and_fused_mc2() -> None:
    with (
        patch(
            "vllm_ascend.lora.fused_moe.get_current_vllm_config_or_none",
            return_value=SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=True)),
        ),
        pytest.raises(AssertionError, match="incompatible with EPLB"),
    ):
        _assert_ascend_moe_lora_supported(SimpleNamespace(_shared_experts=None))
    with (
        patch("vllm_ascend.lora.fused_moe.get_current_vllm_config_or_none", return_value=None),
        patch(
            "vllm_ascend.lora.fused_moe.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=1),
        ),
        pytest.raises(AssertionError, match="FusedMC2"),
    ):
        _assert_ascend_moe_lora_supported(SimpleNamespace(_shared_experts=None))


def test_assert_warns_once_for_shared_experts() -> None:
    with (
        patch(
            "vllm_ascend.lora.fused_moe.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=0),
        ),
        patch("vllm_ascend.lora.fused_moe.logger.warning_once") as warn,
    ):
        _assert_ascend_moe_lora_supported(SimpleNamespace(_shared_experts=object()))
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


_FM = "vllm_ascend.lora.fused_moe"


def _lctx(no_lora=False, active=None, loaded=None, ep_slots=None, use_ep=True, rep=True, **kw):
    pw = SimpleNamespace(no_lora=no_lora, active_lora_slots=set() if active is None else active)
    if "token_lora_indices" in kw:
        pw.token_lora_indices = kw.pop("token_lora_indices")
    return SimpleNamespace(
        punica_wrapper=pw,
        layer_lora_slots=set() if loaded is None else loaded,
        ep_lora_slots=ep_slots,
        use_ep=use_ep,
        replicated_batch=rep,
        **kw,
    )


def test_moe_lora_predicates_dp_guards_and_can_replace_layer() -> None:
    """Verify MoE LoRA slot predicates, DP=1 guards, and layer replacement checks."""
    cases: tuple[tuple[set[int], set[int], bool, bool], ...] = (
        ({0}, {0}, False, True),
        ({1}, {0}, False, False),
        ({1}, set(), False, False),
        ({0}, {0}, True, True),
        ({1}, {0}, True, False),
    )
    for act, load, use_ep, exp in cases:
        assert has_moe_lora(_lctx(False, act, load, use_ep=use_ep)) is exp
    assert not has_moe_lora(_lctx(True, set(), {0}))
    assert_moe_lora_dp1_supported(1)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        assert_moe_lora_dp1_supported(2)
    with (
        patch(f"{_FM}.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=0)),
        patch(f"{_FM}._get_lora_device", return_value=torch.device("cpu")),
    ):
        for fn in (_assert_ascend_moe_lora_supported, AscendFusedMoEWithLoRA):
            with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
                fn(_make_base_layer(dp_size=2, use_ep=True))

    t1 = torch.tensor([0])
    dp2 = _lctx(
        False,
        {0},
        {0},
        rep=False,
        top_k=1,
        local_num_experts=1,
        token_lora_indices=t1,
        gathered_lora_indices=t1,
        exchanged_lora_indices=t1,
    )
    dp_guards: tuple[tuple[Any, tuple[Any, ...], dict[str, Any]], ...] = (
        (has_moe_lora, (dp2,), {}),
        (should_disable_all2all_quant_for_lora, (dp2, QuantType.W8A8), {}),
        (should_exchange_all2all_lora_indices, (dp2, QuantType.W8A8), {}),
        (should_gather_allgather_lora_indices, (dp2,), {}),
        (should_use_unfused_quant_moe_for_lora, (dp2, QuantType.W4A8), {}),
        (slice_local_lora_indices, (dp2,), {"num_tokens": 2}),
        (_recover_moe_lora_routing_allgather, (dp2, t1, torch.tensor([[0]])), {}),
        (_recover_moe_lora_routing_all2all, (dp2, torch.tensor([1])), {}),
    )
    for g_fn, args, kwargs in dp_guards:
        with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
            g_fn(*args, **kwargs)

    moe_cls = maybe_get_oot_by_class(MoERunner)
    l1, l2 = object.__new__(moe_cls), object.__new__(moe_cls)
    torch.nn.Module.__init__(l1)
    torch.nn.Module.__init__(l2)
    l1.moe_config, l2.moe_config = SimpleNamespace(dp_size=1), SimpleNamespace(dp_size=2)
    cfg = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)
    assert AscendFusedMoEWithLoRA.can_replace_layer(l1, cfg, ["w1", "w3"], None)
    assert AscendFusedMoE3DWithLoRA.can_replace_layer(l1, cfg, ["w13"], None)
    assert not AscendFusedMoE3DWithLoRA.can_replace_layer(l1, cfg, ["w1", "w3"], None)
    assert not AscendFusedMoEWithLoRA.can_replace_layer(l2, cfg, ["w1", "w3"], None)
    assert not AscendFusedMoE3DWithLoRA.can_replace_layer(l2, cfg, ["w13"], None)


def test_all2all_unfused_allgather_slicing_and_shape_error() -> None:
    """Verify AlltoAll/unfused/AllGather gating, empty-batch cleanup, slicing, and shape errors."""
    for ctx, qt, ed, ee, eu, eg in (
        (None, None, False, False, False, False),
        (_lctx(True, set(), set()), QuantType.W4A8, False, False, False, False),
        (_lctx(True, set(), {0}), QuantType.W4A8, True, False, True, True),
        (_lctx(True, set(), {0}), QuantType.W8A8, True, False, True, True),
        (_lctx(True, set(), {0}), QuantType.W4A8MXFP, False, False, False, True),
        (_lctx(False, {1}, {0}), None, False, False, False, True),
        (_lctx(False, {0}, {0}), None, True, True, False, True),
    ):
        assert should_disable_all2all_quant_for_lora(ctx, quant_type=qt) is ed
        assert should_exchange_all2all_lora_indices(ctx, quant_type=qt) is ee
        assert should_gather_allgather_lora_indices(ctx) is eg
        if qt is not None:
            assert should_use_unfused_quant_moe_for_lora(ctx, qt) is eu
    with (
        patch(f"{_FM}._EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch(f"{_FM}._is_lora_graph_capturing", return_value=True),
    ):
        empty_c = _lctx(True, set(), set())
        assert should_use_unfused_quant_moe_for_lora(empty_c, QuantType.W4A8)
        assert should_gather_allgather_lora_indices(empty_c)

    z = torch.empty((0,), dtype=torch.long)
    tok = SimpleNamespace(token_lora_indices=torch.tensor([0]))
    e_ctx = SimpleNamespace(punica_wrapper=tok, split_lora_indices=z, exchanged_lora_indices=z)
    prepare_lora_indices(e_ctx, num_tokens=3, pad_size=1, tp_size=1, tp_rank=0)
    assert e_ctx.split_lora_indices.tolist() == [0, -1, -1, -1]
    e_ctx.split_lora_indices = z
    with patch(f"{_FM}.torch.argsort") as m_sort:
        preprocess_lora_indices(e_ctx, topk_ids=torch.empty((0, 2), dtype=torch.long), reversed_permutation_mapping=z)
        postprocess_lora_indices(e_ctx, reversed_permutation_mapping=z)
    m_sort.assert_not_called()
    assert e_ctx.permuted_lora_indices.numel() == 0 and e_ctx.exchanged_lora_indices.numel() == 0

    w2_ctx = SimpleNamespace(
        punica_wrapper=Mock(),
        split_lora_indices=torch.tensor([0]),
        permuted_lora_indices=torch.tensor([0]),
        exchanged_lora_indices=torch.tensor([0]),
    )
    moe_lora_apply_w2(w2_ctx, down_out=torch.empty((0, 4)), silu_out=torch.empty((0, 4)), lora_routing=(z, z))
    w2_ctx.punica_wrapper.add_lora_fused_moe.assert_not_called()
    assert all(
        not hasattr(w2_ctx, k) for k in ("split_lora_indices", "permuted_lora_indices", "exchanged_lora_indices")
    )

    s_ctx = SimpleNamespace(punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([2, -1, 0, 9])))
    assert slice_local_lora_indices(s_ctx, num_tokens=2, pad_size=2).tolist() == [2, -1, -1, -1]
    assert slice_local_lora_indices(s_ctx, num_tokens=6).tolist() == [2, -1, 0, 9, -1, -1]
    ep_e, ep_l = _recover_moe_lora_routing_allgather(
        _lctx(ep_rank=1, local_num_experts=2, top_k=2, gathered_lora_indices=torch.tensor([-1, -1, 0, 1])),
        torch.tensor([-1, 0, 1, -1, 2, -1, 3, 4], dtype=torch.int32),
        torch.tensor([[0, 2], [3, 1], [2, 0], [3, 2]]),
    )
    assert ep_e[:5].tolist() == [0, 1, 0, 1, 0] and ep_l.tolist() == [-1, -1, 0, 1, 1, -1, -1, -1]

    base = _make_base_layer()
    base.prefix = "model.layers.2.mlp.experts"
    with (
        patch(f"{_FM}._assert_ascend_moe_lora_supported"),
        patch(f"{_FM}._get_lora_device", return_value=torch.device("cpu")),
    ):
        wrapper = AscendFusedMoEWithLoRA(base)
    wrapper.w13_lora_a_stacked, wrapper.w13_lora_b_stacked = (torch.empty(1, 2, 8, 4096),), (torch.empty(1, 2, 256, 8),)
    wrapper.w2_lora_a_stacked, wrapper.w2_lora_b_stacked = (torch.empty(1, 2, 8, 256),), (torch.empty(1, 2, 4096, 8),)
    pat = r"model.layers.2.mlp.experts.*slot 1.*\(2, 8, 5120\).*\(2, 8, 4096\).*shape mismatch"
    with (
        patch.object(FusedMoEWithLoRA, "set_lora", side_effect=RuntimeError("shape mismatch")),
        pytest.raises(ValueError, match=pat),
    ):
        wrapper.set_lora(1, [torch.empty(2, 8, 5120)] * 3, [torch.empty(2, 256, 8)] * 3)


def test_punica_metadata_bounds_and_fused_moe_clamping() -> None:
    """Verify PunicaWrapperNPU metadata refresh and fused MoE expert/slot bounds clamping."""
    w = object.__new__(PunicaWrapperNPU)
    with patch.object(PunicaWrapperBase, "update_metadata"):
        w.update_metadata(SimpleNamespace(index_mapping=(0, 7, 0)), [7, 11], 2, 100)
        assert w.active_lora_slots == {0} and not w.no_lora
        w.update_metadata(SimpleNamespace(index_mapping=(0, 0)), [7, 11], 2, 100)
        assert w.active_lora_slots == set() and w.no_lora
    for is_prefill in (True, False):
        pw = object.__new__(PunicaWrapperNPU)
        PunicaWrapperBase.__init__(pw, 8, 4, torch.device("cpu"))
        pw._single_lora_mask = None
        pw.update_metadata(LoRAMapping((0, 7, 0), (0, 7, 0), is_prefill=is_prefill), [7, 11], 2, 100)
        assert pw.token_lora_indices.tolist() == [-1, 0, -1] and pw.active_lora_slots == {0} and not pw.no_lora
        pw.update_metadata(LoRAMapping((0, 0), (0, 0), is_prefill=is_prefill), [7, 11], 2, 100)
        assert pw.token_lora_indices.tolist() == [-1, -1] and pw.no_lora

    w.bgmv_shrink, w.bgmv_expand_slice = Mock(), Mock()
    w.add_lora_fused_moe(
        torch.zeros(4, 1),
        torch.ones(4, 1),
        (torch.ones(2, 4, 1, 1),),
        (torch.ones(2, 4, 1, 1),),
        expert_ids=torch.tensor([4, -1, 3, 0]),
        token_lora_mapping=torch.tensor([0, 1, 1, 2]),
        adapter_enabled=torch.ones(2, dtype=torch.bool),
    )
    assert w.bgmv_shrink.call_args.args[3].tolist() == [-1, -1, 7, -1]
    w.bgmv_shrink.reset_mock()
    w.bgmv_expand_slice.reset_mock()
    w.add_lora_fused_moe(
        torch.zeros(2, 1),
        torch.ones(2, 1),
        (torch.ones(1, 4, 1, 1),),
        (torch.ones(1, 4, 1, 1),),
        expert_ids=torch.tensor([0, 1]),
        token_lora_mapping=torch.tensor([0, -1]),
        adapter_enabled=torch.zeros(0, dtype=torch.bool),
    )
    w.bgmv_shrink.assert_not_called()
    w.bgmv_expand_slice.assert_not_called()


def test_graph_allgather_dp1_ep2_mixed_base_and_adapter_tokens() -> None:
    """Verify DP=1 EP=2 graph AllGather MoE LoRA matches unsharded reference for mixed tokens."""
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig

    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
    from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
    from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather

    torch.manual_seed(0)
    h, inter, rank, num_e, local_e, top_k, n_tok, slots = 6, 3, 2, 4, 2, 2, 4, [-1, 0, 1, -1]
    w13, w2 = torch.randn(num_e, h, inter * 2) * 0.2, torch.randn(num_e, inter, h) * 0.2
    w13_a, w13_b = torch.randn(2, num_e, rank, h) * 0.3, torch.randn(2, num_e, inter * 2, rank) * 0.3
    w2_a, w2_b = torch.randn(2, num_e, rank, inter) * 0.3, torch.randn(2, num_e, h, rank) * 0.3
    tokens = torch.randn(n_tok, h)
    logits = torch.tensor([[5.0, 1.0, 4.0, 0.5], [0.5, 4.0, 5.0, 1.0], [4.5, 0.2, 5.0, 0.1], [0.2, 4.0, 0.5, 5.0]])

    def _shrink(x, a, s_out, c_idx, scale):
        for r, idx in enumerate(c_idx.tolist()):
            if idx >= 0:
                s_out[r] = scale * (x[r].float() @ a[idx].T.float())

    def _expand(delta, b, y, c_idx, off, sz, add_inputs=True):
        for r, idx in enumerate(c_idx.tolist()):
            if idx >= 0:
                y[r, off : off + sz] += (delta[r] @ b[idx].T.float()).to(y.dtype)

    punica = object.__new__(PunicaWrapperNPU)
    punica.no_lora, punica.active_lora_slots, punica.indices_len = False, {0, 1}, [n_tok]
    punica._token_lora_indices = torch.tensor(slots, dtype=torch.long)
    punica.bgmv_shrink, punica.bgmv_expand_slice = _shrink, _expand

    def _init_routing(hs, tk_ids, *, active_expert_range, **_kw):
        exp = torch.full((hs.shape[0] * top_k,), -1, dtype=torch.int32)
        rows: list[torch.Tensor] = []
        for le in range(*active_expert_range):
            for t in range(hs.shape[0]):
                for k in range(top_k):
                    if int(tk_ids[t, k]) == le:
                        exp[t * top_k + k] = len(rows)
                        rows.append(hs[t])
        out = torch.full((hs.shape[0] * top_k, h), 999.0, dtype=hs.dtype)
        if rows:
            out[: len(rows)] = torch.stack(rows)
        counts = torch.tensor([int((tk_ids == le).sum()) for le in range(*active_expert_range)], dtype=torch.int32)
        return out, exp, counts, None

    def _unpermute(*, permuted_tokens, sorted_indices, probs, **_kw):
        idx = sorted_indices.reshape(probs.shape[0], top_k)
        return sum(
            torch.where(idx[:, k : k + 1] >= 0, probs[:, k : k + 1] * permuted_tokens[idx[:, k].clamp(min=0)], 0.0)
            for k in range(top_k)
        )

    tk_w, tk_ids = torch.topk(logits, k=top_k, dim=-1)
    tk_w, e_maps, partials = (
        torch.softmax(tk_w, dim=-1),
        {0: torch.tensor([0, 1, -1, -1]), 1: torch.tensor([-1, -1, 0, 1])},
        [],
    )
    td_mod = "vllm_ascend.ops.fused_moe.token_dispatcher"
    mc_kw = dict(
        dp_size=1,
        pcp_size=1,
        tp_size=2,
        ep_size=2,
        num_experts=num_e,
        num_local_experts=local_e,
        experts_per_token=top_k,
        is_sequence_parallel=False,
    )
    for r in range(2):
        mc, sl = MagicMock(spec=FusedMoEConfig, **mc_kw), slice(r * local_e, (r + 1) * local_e)
        ctx = SimpleNamespace(
            punica_wrapper=punica,
            use_ep=True,
            ep_rank=r,
            local_num_experts=local_e,
            top_k=top_k,
            layer_lora_slots={0, 1},
            replicated_batch=True,
            adapter_enabled=torch.tensor([True, True]),
            w13_lora_a_stacked=(w13_a[:, sl].contiguous(),),
            w13_lora_b_stacked=(w13_b[:, sl].contiguous(),),
            w2_lora_a_stacked=(w2_a[:, sl].contiguous(),),
            w2_lora_b_stacked=(w2_b[:, sl].contiguous(),),
        )
        prep = PrepareAndFinalizeWithAllGather(mc)
        disp = TokenDispatcherWithAllGather(top_k=top_k, num_experts=num_e, num_local_experts=local_e)
        prep.set_lora_context(ctx)
        prepped = prep.prepare(tokens, logits)
        assert not hasattr(ctx, "gathered_lora_indices")
        disp.set_lora_context(ctx)
        td_in = MoETokenDispatchInput(
            prepped.hidden_states,
            tk_w.clone(),
            tk_ids.to(torch.int32),
            MoeRouterInput(e_maps[r], 0, None, False),
            MoEQuantParams(),
        )
        with (
            patch(f"{td_mod}.get_ep_group", return_value=SimpleNamespace(rank_in_group=r)),
            patch(f"{td_mod}.DeviceOperator.npu_moe_init_routing", side_effect=_init_routing),
            patch(f"{td_mod}.DeviceOperator.npu_moe_token_unpermute", side_effect=_unpermute),
        ):
            d_out = disp.token_dispatch(td_in)
            lr = _recover_moe_lora_routing_allgather(ctx, d_out.combine_metadata.expanded_row_idx, tk_ids)
            act_rows, s_h = int(d_out.group_list.sum()), d_out.hidden_states
            gu = torch.zeros((s_h.shape[0], inter * 2))
            gu[:act_rows] = torch.bmm(s_h[:act_rows, None], w13[r * local_e + lr[0][:act_rows]]).squeeze(1)
            moe_lora_apply_w13(ctx, gate_up_out=gu, hidden_states=s_h, lora_routing=lr)
            g, u = gu.chunk(2, dim=-1)
            silu = torch.nn.functional.silu(g) * u
            dn = torch.zeros_like(s_h)
            dn[:act_rows] = torch.bmm(silu[:act_rows, None], w2[r * local_e + lr[0][:act_rows]]).squeeze(1)
            moe_lora_apply_w2(ctx, down_out=dn, silu_out=silu, lora_routing=lr)
            partials.append(disp.token_combine(dn, d_out.combine_metadata))
            assert not (lr[1][act_rows:] >= 0).any()

    expected = torch.zeros_like(tokens)
    for t in range(n_tok):
        x, s = tokens[t], slots[t]
        for k in range(top_k):
            e = int(tk_ids[t, k])
            gu = x @ w13[e] + ((x @ w13_a[s, e].T) @ w13_b[s, e].T if s >= 0 else 0)
            g, u = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(g) * u
            expected[t] += tk_w[t, k] * (act @ w2[e] + ((act @ w2_a[s, e].T) @ w2_b[s, e].T if s >= 0 else 0))
    torch.testing.assert_close(partials[0] + partials[1], expected, atol=1e-5, rtol=1e-5)


def test_ep_partial_expert_lora_sync_and_collective_uniformity() -> None:
    """Verify EP-wide slot sync across partial-expert ranks and collective uniformity."""
    w0, w1 = (AscendFusedMoEWithLoRA.__new__(AscendFusedMoEWithLoRA) for _ in range(2))
    for rank, w in ((0, w0), (1, w1)):
        base = _make_base_layer(use_ep=True)
        base.moe_config.moe_parallel_config.ep_rank, base.set_lora_context = rank, Mock()
        with (
            patch(f"{_FM}._assert_ascend_moe_lora_supported"),
            patch(f"{_FM}._get_lora_device", return_value=torch.device("cpu")),
        ):
            AscendFusedMoEWithLoRA.__init__(w, base)

    reduced: list[int] = []

    def _ar(t, op=None, group=None):
        assert group == "ep_cpu"
        reduced.append(int(t.item()))
        t.fill_(1 if len(reduced) <= 2 else 0)

    ep_grp = SimpleNamespace(world_size=2, cpu_group="ep_cpu", device_group=None)
    with (
        patch.object(FusedMoEWithLoRA, "reset_lora", autospec=True),
        patch.object(FusedMoEWithLoRA, "set_lora", autospec=True, side_effect=lambda s, i, _a, _b: s.reset_lora(i)),
        patch(f"{_FM}.torch.distributed.is_initialized", return_value=True),
        patch(f"{_FM}.get_ep_group", return_value=ep_grp),
        patch(f"{_FM}.torch.distributed.all_reduce", side_effect=_ar) as m_ar,
    ):
        w0.set_lora(0, [], [])
        w1.reset_lora(0)
        assert m_ar.call_count == 2 and w0._lora_slots == {0} and w1._lora_slots == set()
        assert w0._ep_lora_slots == w1._ep_lora_slots == {0} and reduced == [1, 0]
        with patch.object(FusedMoEWithLoRA, "_build_lora_context", side_effect=[SimpleNamespace(), SimpleNamespace()]):
            b0, b1 = w0._build_lora_context(), w1._build_lora_context()
        assert b0.layer_lora_slots is w0._lora_slots and b0.ep_lora_slots is w0._ep_lora_slots
        assert b1.layer_lora_slots is w1._lora_slots and b1.ep_lora_slots is w1._ep_lora_slots
        w0.reset_lora(0)
        w1.reset_lora(0)
        assert m_ar.call_count == 4 and not w0._ep_lora_slots and not w1._ep_lora_slots and reduced == [1, 0, 0, 0]

    tok_idx = torch.tensor([0, -1], dtype=torch.long)
    for is_base in (False, True):
        act = set() if is_base else {0}
        r0 = _lctx(is_base, act, {0}, {0}, token_lora_indices=tok_idx)
        r1 = _lctx(is_base, act, set(), {0}, token_lora_indices=tok_idx)
        assert has_moe_lora(r0) is (not is_base) and not has_moe_lora(r1)
        assert has_collective_moe_lora(r0) is (not is_base) and has_collective_moe_lora(r1) is (not is_base)
        for ctx in (r0, r1):
            for qt in (QuantType.W4A8, QuantType.W8A8):
                assert should_use_unfused_quant_moe_for_lora(ctx, qt)
                assert should_disable_all2all_quant_for_lora(ctx, qt)
                assert should_exchange_all2all_lora_indices(ctx, qt) is (not is_base)
            assert not should_use_unfused_quant_moe_for_lora(ctx, QuantType.W4A8MXFP)
            assert should_disable_all2all_quant_for_lora(ctx, QuantType.NONE) is (not is_base)
            assert should_gather_allgather_lora_indices(ctx)
    assert slice_local_lora_indices(
        _lctx(False, {0}, set(), {0}, token_lora_indices=tok_idx), num_tokens=2
    ).tolist() == [0, -1]
