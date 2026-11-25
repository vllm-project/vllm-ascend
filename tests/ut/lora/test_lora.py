from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import vllm.lora.utils as upstream_lora_utils
from vllm.config import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.model_manager import LRUCacheLoRAModelManager
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.models import SupportsLoRA

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
from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
    AscendReplicatedLinearWithLoRA,
    AscendRowParallelLinearWithLoRA,
    install_dp_moe_lora_load_guard,
    refresh_all_lora_classes,
)
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


def test_ep_dp1_base_only_batch_skips_moe_lora():
    idle_context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots={0},
        use_ep=True,
        replicated_batch=True,
    )
    assert not has_moe_lora(idle_context)

    active_context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=False, active_lora_slots={0}),
        layer_lora_slots={0},
        use_ep=True,
        replicated_batch=True,
    )
    assert has_moe_lora(active_context)


def test_moe_lora_rejects_dp_gt1_at_init_and_runtime():
    assert_moe_lora_dp1_supported(1)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        assert_moe_lora_dp1_supported(2)

    base_dp2 = _make_base_layer(dp_size=2, use_ep=True)
    with (
        patch("vllm_ascend.lora.fused_moe.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=0)),
        patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
    ):
        with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
            _assert_ascend_moe_lora_supported(base_dp2)
        with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
            AscendFusedMoEWithLoRA(base_dp2)

    dp_gt1_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(
            no_lora=False,
            active_lora_slots={0},
            token_lora_indices=torch.tensor([0, -1], dtype=torch.long),
        ),
        layer_lora_slots={0},
        use_ep=True,
        replicated_batch=False,
        top_k=1,
        local_num_experts=1,
        gathered_lora_indices=torch.tensor([0], dtype=torch.long),
        exchanged_lora_indices=torch.tensor([0], dtype=torch.long),
    )
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        has_moe_lora(dp_gt1_ctx)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_disable_all2all_quant_for_lora(dp_gt1_ctx, quant_type=QuantType.W8A8)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_exchange_all2all_lora_indices(dp_gt1_ctx, quant_type=QuantType.W8A8)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_gather_allgather_lora_indices(dp_gt1_ctx)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_use_unfused_quant_moe_for_lora(dp_gt1_ctx, QuantType.W4A8)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        slice_local_lora_indices(dp_gt1_ctx, num_tokens=2)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        _recover_moe_lora_routing_allgather(dp_gt1_ctx, torch.tensor([0]), torch.tensor([[0]]))
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        _recover_moe_lora_routing_all2all(dp_gt1_ctx, torch.tensor([1]))


def test_ascend_moe_lora_can_replace_layer_respects_dp_size() -> None:
    moe_cls = maybe_get_oot_by_class(MoERunner)
    layer_dp1 = object.__new__(moe_cls)
    torch.nn.Module.__init__(layer_dp1)
    layer_dp1.moe_config = SimpleNamespace(dp_size=1)
    layer_dp2 = object.__new__(moe_cls)
    torch.nn.Module.__init__(layer_dp2)
    layer_dp2.moe_config = SimpleNamespace(dp_size=2)

    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)
    assert AscendFusedMoEWithLoRA.can_replace_layer(layer_dp1, lora_config, ["w1", "w3"], None) is True
    assert AscendFusedMoE3DWithLoRA.can_replace_layer(layer_dp1, lora_config, ["w1", "w3"], None) is False
    assert AscendFusedMoEWithLoRA.can_replace_layer(layer_dp1, lora_config, ["w13"], None) is False
    assert AscendFusedMoE3DWithLoRA.can_replace_layer(layer_dp1, lora_config, ["w13"], None) is True
    assert AscendFusedMoEWithLoRA.can_replace_layer(layer_dp2, lora_config, ["w1", "w3"], None) is False
    assert AscendFusedMoE3DWithLoRA.can_replace_layer(layer_dp2, lora_config, ["w13"], None) is False


def _make_dp_test_replicated_linear(prefix: str, in_dim: int = 4, out_dim: int = 6) -> torch.nn.Module:
    replicated_cls = maybe_get_oot_by_class(ReplicatedLinear)
    layer = object.__new__(replicated_cls)
    torch.nn.Module.__init__(layer)
    layer.input_size = in_dim
    layer.output_size = out_dim
    layer.output_partition_sizes = [out_dim]
    layer.input_size_per_partition = in_dim
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.skip_bias_add = False
    layer.return_bias = False
    layer.bias = None
    layer.prefix = prefix
    layer.quant_method = None
    layer.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim, dtype=torch.bfloat16))
    return layer


def _make_dp_test_moe_runner(dp_size: int, shared_experts: torch.nn.Module) -> torch.nn.Module:
    moe_cls = maybe_get_oot_by_class(MoERunner)
    runner = object.__new__(moe_cls)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(
        dp_size=dp_size,
        hidden_dim=4,
        num_local_experts=2,
        num_experts=2,
        intermediate_size_per_partition=4,
        experts_per_token=1,
        moe_parallel_config=SimpleNamespace(tp_size=1, tp_rank=0, ep_rank=0, use_ep=False),
        is_act_and_mul=True,
    )
    runner.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(is_monolithic=False))
    runner._shared_experts = shared_experts
    runner.set_lora_context = MagicMock()
    return runner


class _FakeDP2MoEModel(torch.nn.Module):
    supports_lora = SupportsLoRA.supports_lora
    config = None
    is_3d_moe_weight = False
    is_non_gated_moe = False
    embedding_modules: dict[str, str] = {}
    packed_modules_mapping: dict[str, list[str]] = {}
    lora_skip_prefixes: list[str] = []

    def __init__(self, *, dp_size: int = 2, multimodal_prefix: bool = False) -> None:
        super().__init__()
        self._multimodal_prefix = multimodal_prefix
        layer0 = torch.nn.Module()
        layer0.self_attn = torch.nn.Module()
        prefix = "language_model.model.layers.0" if multimodal_prefix else "model.layers.0"
        layer0.self_attn.wq_a = _make_dp_test_replicated_linear(f"{prefix}.self_attn.wq_a")
        layer0.mlp = torch.nn.Module()
        layer0.mlp.shared_experts = torch.nn.Module()
        layer0.mlp.shared_experts.down_proj = _make_dp_test_replicated_linear(f"{prefix}.mlp.shared_experts.down_proj")
        layer0.mlp.experts = _make_dp_test_moe_runner(dp_size, layer0.mlp.shared_experts)
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([layer0])
        if multimodal_prefix:
            language_model = torch.nn.Module()
            language_model.model = inner
            self.language_model = language_model
        else:
            self.model = inner

    def get_mm_mapping(self):
        return SimpleNamespace(language_model=("language_model.",), tower_model=(), connector=())

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return [
            ("experts.0.w1", "experts.0.gate_proj.", 0, "w1"),
            ("experts.0.w2", "experts.0.down_proj.", 0, "w2"),
            ("experts.0.w3", "experts.0.up_proj.", 0, "w3"),
        ]


def test_default_config_dp_gt1_allows_attention_only_startup_and_rejects_expert_loads() -> None:
    original_classes = upstream_lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        model = _FakeDP2MoEModel(dp_size=2)
        moe_runner = model.model.layers[0].mlp.experts
        lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16, max_cpu_loras=2)
        vllm_config = SimpleNamespace(
            lora_config=lora_config,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            model_config=None,
        )
        punica_wrapper = object.__new__(PunicaWrapperNPU)
        PunicaWrapperBase.__init__(punica_wrapper, 16, 2, "cpu")

        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica_wrapper):
            manager = LRUCacheLoRAModelManager(
                model=model,
                max_num_seqs=2,
                max_num_batched_tokens=16,
                vocab_size=32,
                lora_config=lora_config,
                device=torch.device("cpu"),
                vllm_config=vllm_config,
            )

        assert "model.layers.0.self_attn.wq_a" in manager.modules
        assert "model.layers.0.mlp.shared_experts.down_proj" in manager.modules
        assert isinstance(manager.modules["model.layers.0.self_attn.wq_a"], AscendReplicatedLinearWithLoRA)
        shared_expert_lora = manager.modules["model.layers.0.mlp.shared_experts.down_proj"]
        assert isinstance(shared_expert_lora, AscendReplicatedLinearWithLoRA)
        assert "model.layers.0.mlp.experts" not in manager.modules
        assert model.model.layers[0].mlp.experts is moe_runner
        moe_runner.set_lora_context.assert_not_called()

        dummy = manager.create_dummy_lora(lora_id=1, rank=4, embedding_modules=model.embedding_modules)
        assert set(dummy.loras.keys()) == {
            "model.layers.0.self_attn.wq_a",
            "model.layers.0.mlp.shared_experts.down_proj",
        }
        assert manager.add_adapter(dummy) is True
        assert manager.activate_adapter(dummy.id) is True

        registered: dict[int, object] = {}

        def _fake_load(lora_request):
            return lora_request.loaded_model

        def _fake_add_adapter(lora_request):
            lora_model = worker_manager._load_adapter(lora_request)
            registered[lora_model.id] = lora_model
            return True

        worker_manager = SimpleNamespace(_adapter_manager=manager, _load_adapter=_fake_load)
        install_dp_moe_lora_load_guard(worker_manager)

        attn_shared_adapter = SimpleNamespace(
            id=1,
            loras={
                "model.layers.0.self_attn.wq_a": object(),
                "model.layers.0.mlp.shared_experts.down_proj": object(),
            },
        )
        assert _fake_add_adapter(SimpleNamespace(loaded_model=attn_shared_adapter)) is True
        assert set(registered.keys()) == {1}

        for bad_id, bad_loras in (
            (2, {"model.layers.0.mlp.experts.0.gate_proj": object()}),
            (3, {"model.layers.0.mlp.experts": object()}),
            (4, {"model.layers.0.mlp.experts.base_layer": object()}),
        ):
            bad_adapter = SimpleNamespace(
                id=bad_id,
                loras={
                    "model.layers.0.self_attn.wq_a": object(),
                    **bad_loras,
                },
            )
            with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
                _fake_add_adapter(SimpleNamespace(loaded_model=bad_adapter))
        assert set(registered.keys()) == {1}

        mm_model = _FakeDP2MoEModel(dp_size=2, multimodal_prefix=True)
        mm_manager = SimpleNamespace(
            model=mm_model,
            modules={"language_model.model.layers.0.self_attn.wq_a": object()},
            _get_text_lora_names=lambda name: ["model.layers.0.mlp.experts"] if name.endswith("experts") else [],
        )
        mm_worker = SimpleNamespace(
            _adapter_manager=mm_manager,
            _load_adapter=lambda req: SimpleNamespace(id=5, loras={"model.layers.0.mlp.experts.0.gate_proj": object()}),
        )
        install_dp_moe_lora_load_guard(mm_worker)
        with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
            mm_worker._load_adapter(SimpleNamespace())

        explicit_expert_config = LoRAConfig(
            max_loras=1,
            max_lora_rank=8,
            lora_dtype=torch.bfloat16,
            max_cpu_loras=2,
            target_modules=["experts"],
        )
        with (
            patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica_wrapper),
            pytest.raises(ValueError, match="experts"),
        ):
            LRUCacheLoRAModelManager(
                model=_FakeDP2MoEModel(dp_size=2),
                max_num_seqs=2,
                max_num_batched_tokens=16,
                vocab_size=32,
                lora_config=explicit_expert_config,
                device=torch.device("cpu"),
                vllm_config=SimpleNamespace(
                    lora_config=explicit_expert_config,
                    parallel_config=SimpleNamespace(enable_expert_parallel=False),
                    model_config=None,
                ),
            )
    finally:
        upstream_lora_utils._all_lora_classes = original_classes


def test_model_runner_load_lora_model_installs_dp_moe_guard() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = object.__new__(NPUModelRunner)
    runner.lora_manager = SimpleNamespace(marker="worker_lora_manager")
    sentinel_model = object()
    with (
        patch.object(GPUModelRunner, "load_lora_model", return_value=sentinel_model) as super_load,
        patch("vllm_ascend.lora.utils.install_dp_moe_lora_load_guard") as install_guard,
    ):
        result = runner.load_lora_model(sentinel_model, SimpleNamespace(), torch.device("cpu"))
    assert result is sentinel_model
    super_load.assert_called_once()
    install_guard.assert_called_once_with(runner.lora_manager)


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
                punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
                layer_lora_slots=set(),
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
    assert should_exchange_all2all_lora_indices(context, quant_type=quant_type) is expected_exchange


@pytest.mark.parametrize("quant_type", [None, QuantType.W8A8, QuantType.W4A8MXFP])
def test_all2all_helpers_reject_dp_gt1(quant_type) -> None:
    context = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots={0},
        replicated_batch=False,
    )
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_disable_all2all_quant_for_lora(context, quant_type=quant_type)
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_exchange_all2all_lora_indices(context, quant_type=quant_type)


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

    replicated_empty_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots=set(),
        replicated_batch=True,
    )
    assert not should_gather_allgather_lora_indices(replicated_empty_ctx)

    loaded_idle_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots={0},
        replicated_batch=True,
    )
    assert should_gather_allgather_lora_indices(loaded_idle_ctx)

    dp_gt1_empty_ctx = SimpleNamespace(
        punica_wrapper=SimpleNamespace(no_lora=True, active_lora_slots=set()),
        layer_lora_slots=set(),
        replicated_batch=False,
    )
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        should_gather_allgather_lora_indices(dp_gt1_empty_ctx)

    with patch("vllm_ascend.lora.fused_moe._is_lora_graph_capturing", return_value=True):
        assert should_gather_allgather_lora_indices(replicated_empty_ctx)


def test_slice_local_lora_indices_truncates_and_pads_minus_one() -> None:
    context = SimpleNamespace(punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([2, -1, 0, 9])))
    assert slice_local_lora_indices(context, num_tokens=2, pad_size=2).tolist() == [2, -1, -1, -1]
    assert slice_local_lora_indices(context, num_tokens=6).tolist() == [2, -1, 0, 9, -1, -1]


def test_allgather_routing_ep_gathered_indices_translates_experts_and_ignores_inactive_rows() -> None:
    context = SimpleNamespace(
        top_k=2,
        use_ep=True,
        ep_rank=1,
        local_num_experts=2,
        replicated_batch=True,
        gathered_lora_indices=torch.tensor([-1, -1, 0, 1]),
    )
    # EP rank 1 owns global experts [2, 3]. Global routing pairs across 4 gathered tokens:
    # (0,0)->0(-1), (0,1)->2(0), (1,0)->3(1), (1,1)->1(-1), (2,0)->2(2), (2,1)->0(-1), (3,0)->3(3), (3,1)->2(4).
    topk_ids = torch.tensor([[0, 2], [3, 1], [2, 0], [3, 2]])
    expanded_row_idx = torch.tensor([-1, 0, 1, -1, 2, -1, 3, 4], dtype=torch.int32)

    expert_ids, lora_slots = _recover_moe_lora_routing_allgather(context, expanded_row_idx, topk_ids)

    assert expert_ids[:5].tolist() == [0, 1, 0, 1, 0]
    assert lora_slots.tolist() == [-1, -1, 0, 1, 1, -1, -1, -1]


def test_allgather_routing_rejects_dp_gt1() -> None:
    context = SimpleNamespace(
        top_k=1,
        use_ep=True,
        replicated_batch=False,
        punica_wrapper=SimpleNamespace(token_lora_indices=torch.tensor([0])),
    )
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        _recover_moe_lora_routing_allgather(context, torch.tensor([0]), torch.tensor([[0]]))

    context.gathered_lora_indices = torch.tensor([0])
    with pytest.raises(NotImplementedError, match="data_parallel_size > 1"):
        _recover_moe_lora_routing_allgather(context, torch.tensor([0]), torch.tensor([[0]]))


def test_graph_allgather_dp1_ep2_mixed_base_and_adapter_tokens() -> None:
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
            tensor_parallel_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        lora_config=SimpleNamespace(max_loras=2),
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
    num_tokens = 4

    w13_base = torch.randn(num_experts, hidden_size, inter_size * 2) * 0.2
    w2_base = torch.randn(num_experts, inter_size, hidden_size) * 0.2
    w13_lora_a = torch.randn(2, num_experts, lora_rank, hidden_size) * 0.3
    w13_lora_b = torch.randn(2, num_experts, inter_size * 2, lora_rank) * 0.3
    w2_lora_a = torch.randn(2, num_experts, lora_rank, inter_size) * 0.3
    w2_lora_b = torch.randn(2, num_experts, hidden_size, lora_rank) * 0.3

    tokens = torch.randn(num_tokens, hidden_size)
    router_logits = torch.tensor(
        [
            [5.0, 1.0, 4.0, 0.5],
            [0.5, 4.0, 5.0, 1.0],
            [4.5, 0.2, 5.0, 0.1],
            [0.2, 4.0, 0.5, 5.0],
        ]
    )
    token_slots = [-1, 0, 1, -1]

    def fake_bgmv_shrink(x, a_flat, shrink_out, combined_idx, scale):
        for row, idx in enumerate(combined_idx.tolist()):
            if idx >= 0:
                shrink_out[row] = scale * (x[row].float() @ a_flat[idx].transpose(0, 1).float())

    def fake_bgmv_expand_slice(delta, b_flat, y, combined_idx, offset, out_size, add_inputs=True):
        del add_inputs
        for row, idx in enumerate(combined_idx.tolist()):
            if idx >= 0:
                y[row, offset : offset + out_size] += (delta[row] @ b_flat[idx].transpose(0, 1).float()).to(y.dtype)

    punica = object.__new__(PunicaWrapperNPU)
    punica.no_lora = False
    punica.active_lora_slots = {0, 1}
    punica.indices_len = [num_tokens]
    punica._token_lora_indices = torch.tensor(token_slots, dtype=torch.long)
    punica.bgmv_shrink = fake_bgmv_shrink
    punica.bgmv_expand_slice = fake_bgmv_expand_slice

    def cpu_init_routing(hidden_states, topk_ids, *, active_expert_range, **_kwargs):
        first, last = active_expert_range
        n_tok = hidden_states.shape[0]
        expanded = torch.full((n_tok * top_k,), -1, dtype=torch.int32)
        rows = []
        dst = 0
        for local_e in range(first, last):
            for t in range(n_tok):
                for k in range(top_k):
                    if int(topk_ids[t, k]) == local_e:
                        expanded[t * top_k + k] = dst
                        rows.append(hidden_states[t])
                        dst += 1
        out = torch.full((n_tok * top_k, hidden_size), 999.0, dtype=hidden_states.dtype)
        if rows:
            stacked = torch.stack(rows, dim=0)
            out[: stacked.shape[0]] = stacked
        expert_counts = torch.tensor(
            [int((topk_ids == local_e).sum()) for local_e in range(first, last)],
            dtype=torch.int32,
        )
        return out, expanded, expert_counts, None

    def cpu_unpermute(*, permuted_tokens, sorted_indices, probs, **_kwargs):
        n_tok = probs.shape[0]
        flat_idx = sorted_indices.reshape(n_tok, top_k)
        out = torch.zeros((n_tok, hidden_size), dtype=permuted_tokens.dtype)
        for t in range(n_tok):
            for k in range(top_k):
                idx = int(flat_idx[t, k])
                if idx >= 0:
                    out[t] += probs[t, k] * permuted_tokens[idx]
        return out

    topk_weights, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    expert_maps = {
        0: torch.tensor([0, 1, -1, -1]),
        1: torch.tensor([-1, -1, 0, 1]),
    }

    partial_outputs = {}
    for rank in range(2):
        moe_config = MagicMock(spec=FusedMoEConfig)
        moe_config.dp_size = 1
        moe_config.pcp_size = 1
        moe_config.tp_size = 2
        moe_config.ep_size = 2
        moe_config.num_experts = num_experts
        moe_config.num_local_experts = num_local_experts
        moe_config.experts_per_token = top_k
        moe_config.is_sequence_parallel = False

        sl = slice(rank * num_local_experts, (rank + 1) * num_local_experts)
        context = SimpleNamespace(
            punica_wrapper=punica,
            use_ep=True,
            ep_rank=rank,
            local_num_experts=num_local_experts,
            top_k=top_k,
            layer_lora_slots={0, 1},
            replicated_batch=True,
            adapter_enabled=torch.tensor([True, True]),
            w13_lora_a_stacked=(w13_lora_a[:, sl].contiguous(),),
            w13_lora_b_stacked=(w13_lora_b[:, sl].contiguous(),),
            w2_lora_a_stacked=(w2_lora_a[:, sl].contiguous(),),
            w2_lora_b_stacked=(w2_lora_b[:, sl].contiguous(),),
        )
        prepare_layer = PrepareAndFinalizeWithAllGather(moe_config)
        prepare_layer.set_lora_context(context)
        prepared = prepare_layer.prepare(tokens, router_logits)
        assert not hasattr(context, "gathered_lora_indices")

        dispatcher = TokenDispatcherWithAllGather(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
        )
        dispatcher.set_lora_context(context)
        assert has_moe_lora(context)

        td_input = MoETokenDispatchInput(
            hidden_states=prepared.hidden_states,
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
            dispatched = dispatcher.token_dispatch(td_input)
            lora_routing = _recover_moe_lora_routing_allgather(
                context,
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
                context,
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
                context,
                down_out=down_out,
                silu_out=silu_out,
                lora_routing=lora_routing,
            )
            partial_outputs[rank] = dispatcher.token_combine(down_out, dispatched.combine_metadata)
            assert not (lora_per_row[active_rows:] >= 0).any()

    actual = partial_outputs[0] + partial_outputs[1]
    expected = torch.zeros_like(tokens)
    for t in range(num_tokens):
        x = tokens[t]
        slot = token_slots[t]
        for k in range(top_k):
            e = int(topk_ids[t, k])
            prob = topk_weights[t, k]
            gu = x @ w13_base[e]
            if slot >= 0:
                gu = gu + (x @ w13_lora_a[slot, e].T) @ w13_lora_b[slot, e].T
            g, u = gu.chunk(2, dim=-1)
            act = torch.nn.functional.silu(g) * u
            dn = act @ w2_base[e]
            if slot >= 0:
                dn = dn + (act @ w2_lora_a[slot, e].T) @ w2_lora_b[slot, e].T
            expected[t] += prob * dn
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_ep_partial_expert_lora_syncs_ep_slots_without_reentrant_deadlock() -> None:
    wrapper_rank0 = AscendFusedMoEWithLoRA.__new__(AscendFusedMoEWithLoRA)
    wrapper_rank1 = AscendFusedMoEWithLoRA.__new__(AscendFusedMoEWithLoRA)
    for rank, w in ((0, wrapper_rank0), (1, wrapper_rank1)):
        base_layer = _make_base_layer(use_ep=True)
        base_layer.moe_config.moe_parallel_config.ep_rank = rank
        base_layer.set_lora_context = Mock()
        with (
            patch("vllm_ascend.lora.fused_moe._assert_ascend_moe_lora_supported"),
            patch("vllm_ascend.lora.fused_moe._get_lora_device", return_value=torch.device("cpu")),
        ):
            AscendFusedMoEWithLoRA.__init__(w, base_layer)

    ep_group = SimpleNamespace(world_size=2, cpu_group="ep_cpu", device_group=None)
    inner_calls: list[str] = []

    def fake_super_reset(self, index: int) -> None:
        inner_calls.append(f"reset:{id(self)}:{index}")

    def fake_super_set(self, index: int, lora_a, lora_b) -> None:
        inner_calls.append(f"set_enter:{id(self)}:{index}")
        # Upstream FusedMoEWithLoRA.set_lora internally invokes self.reset_lora(index).
        self.reset_lora(index)
        inner_calls.append(f"set_done:{id(self)}:{index}")

    # Simulate rank 0 calling set_lora(0, ...) while rank 1 calls reset_lora(0),
    # verifying each rank performs exactly ONE all_reduce without deadlock.
    reduced_inputs: list[int] = []

    def count_all_reduce(tensor, op=None, group=None):
        assert group == "ep_cpu"
        reduced_inputs.append(int(tensor.item()))
        # First collective pair comes from set_lora (1) on rank 0 + reset_lora (0) on rank 1.
        # Second collective pair comes from deactivation reset_lora (0) on both ranks.
        tensor.fill_(1 if len(reduced_inputs) <= 2 else 0)

    with (
        patch.object(FusedMoEWithLoRA, "reset_lora", autospec=True, side_effect=fake_super_reset),
        patch.object(FusedMoEWithLoRA, "set_lora", autospec=True, side_effect=fake_super_set),
        patch("vllm_ascend.lora.fused_moe.torch.distributed.is_initialized", return_value=True),
        patch("vllm_ascend.lora.fused_moe.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.lora.fused_moe.torch.distributed.all_reduce", side_effect=count_all_reduce) as mock_ar,
    ):
        wrapper_rank0.set_lora(0, [], [])
        assert mock_ar.call_count == 1
        wrapper_rank1.reset_lora(0)
        assert mock_ar.call_count == 2

        assert wrapper_rank0._lora_slots == {0}
        assert wrapper_rank1._lora_slots == set()
        assert wrapper_rank0._ep_lora_slots == {0}
        assert wrapper_rank1._ep_lora_slots == {0}
        assert reduced_inputs == [1, 0]

        ctx0 = SimpleNamespace()
        ctx1 = SimpleNamespace()
        with patch.object(FusedMoEWithLoRA, "_build_lora_context", side_effect=[ctx0, ctx1]):
            built0 = wrapper_rank0._build_lora_context()
            built1 = wrapper_rank1._build_lora_context()
        assert built0.layer_lora_slots is wrapper_rank0._lora_slots
        assert built0.ep_lora_slots is wrapper_rank0._ep_lora_slots
        assert built1.layer_lora_slots is wrapper_rank1._lora_slots
        assert built1.ep_lora_slots is wrapper_rank1._ep_lora_slots

        # Deactivate slot 0 on both ranks: both clear both _lora_slots and _ep_lora_slots.
        wrapper_rank0.reset_lora(0)
        wrapper_rank1.reset_lora(0)
        assert mock_ar.call_count == 4
        assert wrapper_rank0._lora_slots == set()
        assert wrapper_rank1._lora_slots == set()
        assert wrapper_rank0._ep_lora_slots == set()
        assert wrapper_rank1._ep_lora_slots == set()
        assert reduced_inputs == [1, 0, 0, 0]


def test_ep_partial_expert_lora_keeps_collectives_and_quant_uniform_across_ranks() -> None:
    token_indices = torch.tensor([0, -1], dtype=torch.long)
    active_punica = SimpleNamespace(no_lora=False, active_lora_slots={0}, token_lora_indices=token_indices)
    base_punica = SimpleNamespace(no_lora=True, active_lora_slots=set(), token_lora_indices=token_indices)

    for no_lora_batch, punica in ((False, active_punica), (True, base_punica)):
        rank0 = SimpleNamespace(
            punica_wrapper=punica,
            layer_lora_slots={0},
            ep_lora_slots={0},
            use_ep=True,
            replicated_batch=True,
        )
        rank1 = SimpleNamespace(
            punica_wrapper=punica,
            layer_lora_slots=set(),
            ep_lora_slots={0},
            use_ep=True,
            replicated_batch=True,
        )

        assert has_moe_lora(rank0) is (not no_lora_batch)
        assert has_moe_lora(rank1) is False
        assert has_collective_moe_lora(rank0) is (not no_lora_batch)
        assert has_collective_moe_lora(rank1) is (not no_lora_batch)

        for qtype in (QuantType.W4A8, QuantType.W8A8):
            assert should_use_unfused_quant_moe_for_lora(rank0, qtype) is True
            assert should_use_unfused_quant_moe_for_lora(rank1, qtype) is True
            assert should_disable_all2all_quant_for_lora(rank0, qtype) is True
            assert should_disable_all2all_quant_for_lora(rank1, qtype) is True
            assert should_exchange_all2all_lora_indices(rank0, qtype) is (not no_lora_batch)
            assert should_exchange_all2all_lora_indices(rank1, qtype) is (not no_lora_batch)

        assert should_use_unfused_quant_moe_for_lora(rank0, QuantType.W4A8MXFP) is False
        assert should_use_unfused_quant_moe_for_lora(rank1, QuantType.W4A8MXFP) is False
        assert should_disable_all2all_quant_for_lora(rank0, QuantType.NONE) is (not no_lora_batch)
        assert should_disable_all2all_quant_for_lora(rank1, QuantType.NONE) is (not no_lora_batch)
        assert should_gather_allgather_lora_indices(rank0) is True
        assert should_gather_allgather_lora_indices(rank1) is True

    rank1_active = SimpleNamespace(
        punica_wrapper=active_punica,
        layer_lora_slots=set(),
        ep_lora_slots={0},
        use_ep=True,
        replicated_batch=True,
    )
    assert slice_local_lora_indices(rank1_active, num_tokens=2).tolist() == [0, -1]


def _make_dp_test_column_linear(prefix: str, in_dim: int = 4, out_dim: int = 6) -> torch.nn.Module:
    col_cls = maybe_get_oot_by_class(ColumnParallelLinear)
    layer = object.__new__(col_cls)
    torch.nn.Module.__init__(layer)
    layer.input_size = in_dim
    layer.output_size = out_dim
    layer.output_size_per_partition = out_dim
    layer.output_partition_sizes = [out_dim]
    layer.output_sizes = [out_dim]
    layer.input_size_per_partition = in_dim
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.gather_output = False
    layer.skip_bias_add = False
    layer.return_bias = False
    layer.bias = None
    layer.prefix = prefix
    layer.quant_method = None
    layer.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim, dtype=torch.bfloat16))
    return layer


def _make_dp_test_row_linear(prefix: str, in_dim: int = 6, out_dim: int = 4) -> torch.nn.Module:
    row_cls = maybe_get_oot_by_class(RowParallelLinear)
    layer = object.__new__(row_cls)
    torch.nn.Module.__init__(layer)
    layer.input_size = in_dim
    layer.output_size = out_dim
    layer.output_size_per_partition = out_dim
    layer.output_partition_sizes = [out_dim]
    layer.output_sizes = [out_dim]
    layer.input_size_per_partition = in_dim
    layer.tp_size = 1
    layer.tp_rank = 0
    layer.input_is_parallel = False
    layer.reduce_results = True
    layer.skip_bias_add = False
    layer.return_bias = False
    layer.bias = None
    layer.prefix = prefix
    layer.quant_method = None
    layer.weight = torch.nn.Parameter(torch.zeros(out_dim, in_dim, dtype=torch.bfloat16))
    return layer


class _FakeDSAWoaModel(torch.nn.Module):
    supports_lora = SupportsLoRA.supports_lora
    config = None
    is_3d_moe_weight = False
    is_non_gated_moe = False
    embedding_modules: dict[str, str] = {}
    packed_modules_mapping: dict[str, list[str]] = {}
    lora_skip_prefixes: list[str] = []

    def __init__(self, *, n_local_groups: int, multimodal_prefix: bool = False) -> None:
        super().__init__()
        self._multimodal_prefix = multimodal_prefix
        layer0 = torch.nn.Module()
        layer0.self_attn = torch.nn.Module()
        prefix = "language_model.model.layers.0" if multimodal_prefix else "model.layers.0"
        layer0.self_attn.wq_a = _make_dp_test_replicated_linear(f"{prefix}.self_attn.wq_a")
        layer0.self_attn.wo_a = _make_dp_test_column_linear(f"{prefix}.self_attn.wo_a")
        layer0.self_attn.wo_a._ascend_dsa_n_local_groups = n_local_groups
        layer0.self_attn.k_proj = _make_dp_test_column_linear(f"{prefix}.self_attn.k_proj")
        layer0.self_attn.wo_b = _make_dp_test_row_linear(f"{prefix}.self_attn.wo_b")
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([layer0])
        if multimodal_prefix:
            language_model = torch.nn.Module()
            language_model.model = inner
            self.language_model = language_model
        else:
            self.model = inner

    def get_mm_mapping(self):
        return SimpleNamespace(language_model=("language_model.",), tower_model=(), connector=())


@pytest.mark.parametrize(
    ("n_local_groups", "otp_enabled", "fully_sharded", "expect_wo_a_wrapped"),
    [
        (2, False, False, False),
        (1, True, False, False),
        (1, False, True, False),
        (1, False, False, True),
    ],
)
def test_unsupported_wo_a_topologies_skip_wrapping_and_reject_wo_a_adapters(
    n_local_groups: int,
    otp_enabled: bool,
    fully_sharded: bool,
    expect_wo_a_wrapped: bool,
) -> None:
    original_classes = upstream_lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        model = _FakeDSAWoaModel(n_local_groups=n_local_groups, multimodal_prefix=not expect_wo_a_wrapped)
        lora_config = LoRAConfig(
            max_loras=1,
            max_lora_rank=8,
            lora_dtype=torch.bfloat16,
            max_cpu_loras=2,
            fully_sharded_loras=fully_sharded,
        )
        vllm_config = SimpleNamespace(
            lora_config=lora_config,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            model_config=None,
        )
        punica_wrapper = object.__new__(PunicaWrapperNPU)
        PunicaWrapperBase.__init__(punica_wrapper, 16, 2, "cpu")

        with (
            patch("vllm_ascend.lora.utils._safe_oproj_tp_enable", return_value=otp_enabled),
            patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica_wrapper),
        ):
            manager = LRUCacheLoRAModelManager(
                model=model,
                max_num_seqs=2,
                max_num_batched_tokens=16,
                vocab_size=32,
                lora_config=lora_config,
                device=torch.device("cpu"),
                vllm_config=vllm_config,
            )

        prefix = "model.layers.0.self_attn" if expect_wo_a_wrapped else "language_model.model.layers.0.self_attn"
        assert f"{prefix}.wq_a" in manager.modules
        assert f"{prefix}.k_proj" in manager.modules
        assert f"{prefix}.wo_b" in manager.modules
        if not fully_sharded:
            assert isinstance(manager.modules[f"{prefix}.wq_a"], AscendReplicatedLinearWithLoRA)
            assert isinstance(manager.modules[f"{prefix}.wo_b"], AscendRowParallelLinearWithLoRA)
            assert isinstance(manager.modules[f"{prefix}.k_proj"], AscendColumnParallelLinearWithLoRA)
        else:
            # Other normal linear modules remain LoRA-wrapped while wo_a is skipped.
            assert isinstance(manager.modules[f"{prefix}.wq_a"], BaseLayerWithLoRA)
            assert isinstance(manager.modules[f"{prefix}.k_proj"], BaseLayerWithLoRA)

        assert (f"{prefix}.wo_a" in manager.modules) is expect_wo_a_wrapped
        if expect_wo_a_wrapped:
            assert isinstance(manager.modules[f"{prefix}.wo_a"], AscendColumnParallelLinearWithLoRA)

        dummy = manager.create_dummy_lora(lora_id=1, rank=4, embedding_modules=model.embedding_modules)
        assert (f"{prefix}.wo_a" in dummy.loras) is expect_wo_a_wrapped
        assert manager.add_adapter(dummy) is True
        assert manager.activate_adapter(dummy.id) is True

        registered: dict[int, object] = {}

        def _fake_load(lora_request):
            return lora_request.loaded_model

        def _fake_add_adapter(lora_request):
            lora_model = worker_manager._load_adapter(lora_request)
            registered[lora_model.id] = lora_model
            return True

        worker_manager = SimpleNamespace(_adapter_manager=manager, _load_adapter=_fake_load)
        with patch("vllm_ascend.lora.utils._safe_oproj_tp_enable", return_value=otp_enabled):
            install_dp_moe_lora_load_guard(worker_manager)

        valid_adapter = SimpleNamespace(
            id=1,
            loras={f"{prefix}.wq_a": object(), f"{prefix}.wo_b": object()},
        )
        assert _fake_add_adapter(SimpleNamespace(loaded_model=valid_adapter)) is True
        assert set(registered.keys()) == {1}

        for bad_id, bad_key in (
            (2, f"{prefix}.wo_a"),
            (3, "model.layers.0.self_attn.wo_a"),
        ):
            wo_a_adapter = SimpleNamespace(
                id=bad_id,
                loras={f"{prefix}.wq_a": object(), bad_key: object()},
            )
            if expect_wo_a_wrapped:
                assert _fake_add_adapter(SimpleNamespace(loaded_model=wo_a_adapter)) is True
            else:
                with pytest.raises(NotImplementedError, match="wo_a"):
                    _fake_add_adapter(SimpleNamespace(loaded_model=wo_a_adapter))
                assert set(registered.keys()) == {1}
    finally:
        upstream_lora_utils._all_lora_classes = original_classes
