# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList
from vllm_ascend.ops.fused_moe.router import AscendFusedMoERouter
from vllm_ascend.quantization.nz import NzCastSpec
from vllm_ascend.quantization.quant_type import QuantType


class _FakeNpuTensor:
    """Minimal NPU tensor stub for CPU-only unit tests."""

    def __init__(
        self,
        shape,
        npu_format,
        *,
        dtype=torch.float16,
        storage_offset=0,
        stride=None,
    ):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.layout = torch.strided
        self.device = torch.device("npu")
        self.npu_format = npu_format
        self._storage_offset = storage_offset
        self._stride = stride or torch.empty(shape).stride()

    def size(self):
        return self.shape

    def storage_offset(self):
        return self._storage_offset

    def stride(self):
        return self._stride


def _routed_experts(weight_views):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace(
        get_eplb_weight_views=lambda layer: weight_views,
    )
    return routed_experts


def test_get_expert_weights_flattens_layout_aware_views():
    weights = [torch.randn(2, 3, 4), torch.randn(2, 5)]

    views = list(_routed_experts(weights).get_expert_weights())

    assert [view.shape for view in views] == [torch.Size([2, 12]), torch.Size([2, 5])]
    assert views[0].untyped_storage().data_ptr() == weights[0].untyped_storage().data_ptr()


def test_get_expert_weights_preserves_independent_expert_tensors():
    expert_tensors = [torch.randn(3, 4), torch.randn(3, 4)]

    views = list(_routed_experts([expert_tensors]).get_expert_weights())

    assert len(views) == 1
    assert isinstance(views[0], EplbExpertTensorList)
    assert views[0].shape == torch.Size([2, 3, 4])
    assert all(actual is expected for actual, expected in zip(views[0], expert_tensors))

    buffer = torch.empty_like(views[0])
    assert isinstance(buffer, EplbExpertTensorList)
    assert buffer.shape == views[0].shape
    assert all(tensor.storage_offset() == 0 for tensor in buffer)


def test_eplb_expert_tensor_list_supports_upstream_uint8_contract():
    # vLLM main registers expert weights via
    # `w.view(torch.uint8) if w.element_size() == 1 else w` (interfaces.py,
    # #61786); the per-expert container must satisfy that contract.
    expert_tensors = [
        torch.ones(3, 4, dtype=torch.int8),
        torch.ones(3, 4, dtype=torch.int8),
    ]
    weights = EplbExpertTensorList(expert_tensors)

    assert weights.element_size() == 1

    viewed = weights.view(torch.uint8)
    assert isinstance(viewed, EplbExpertTensorList)
    assert all(tensor.dtype == torch.uint8 for tensor in viewed)
    assert all(
        actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
        for actual, expected in zip(viewed, expert_tensors)
    )

    registered = [w.view(torch.uint8) if w.element_size() == 1 else w for w in [weights, torch.randn(2, 5)]]
    assert isinstance(registered[0], EplbExpertTensorList)
    assert all(tensor.dtype == torch.uint8 for tensor in registered[0])
    assert registered[1].dtype == torch.float32

    buffer = torch.empty_like(registered[0])
    assert isinstance(buffer, EplbExpertTensorList)
    assert all(tensor.dtype == torch.uint8 for tensor in buffer)


def test_expert_weight_views_keep_the_cast_they_were_built_with():
    """FRACTAL_NZ is not one layout, so a view carries the cast behind it.

    Whatever rebuilds these weights elsewhere has to repeat that cast, and the
    Gloo receive buffers inherit it from here.
    """
    spec = NzCastSpec(customize_dtype=torch.float8_e4m3fn)
    expert_tensors = [torch.ones(3, 4, dtype=torch.int8), torch.ones(3, 4, dtype=torch.int8)]
    declared = EplbExpertTensorList(expert_tensors, spec)

    views = list(_routed_experts([declared, torch.randn(2, 5)]).get_expert_weights())

    assert views[0].nz_cast_spec is spec
    # The uint8 reinterpretation upstream registers weights through, and the
    # buffer allocation, both have to keep it.
    assert views[0].view(torch.uint8).nz_cast_spec is spec
    assert torch.empty_like(views[0]).nz_cast_spec is spec
    # A stacked view declares no cast.
    assert not hasattr(views[1], "nz_cast_spec")


def test_undeclared_expert_tensor_lists_still_become_eplb_containers():
    """Most quant methods cast without parameters and declare nothing."""
    expert_tensors = [torch.ones(3, 4, dtype=torch.int8), torch.ones(3, 4, dtype=torch.int8)]

    views = list(_routed_experts([expert_tensors]).get_expert_weights())

    assert isinstance(views[0], EplbExpertTensorList)
    assert views[0].nz_cast_spec is None


@pytest.mark.parametrize(
    "dtype,npu_formats",
    [
        pytest.param(torch.float16, (29, 50), id="nz-float-weight"),
        pytest.param(torch.float16, (2, 2), id="nd-float-weight"),
        pytest.param(torch.int8, (29, 50), id="nz-int8-weight"),
        pytest.param(torch.int32, (29, 50), id="nz-packed-weight"),
        pytest.param(torch.int64, (2, 2), id="nd-integer-scale"),
    ],
)
def test_eplb_expert_buffers_preserve_each_source_npu_format(monkeypatch, dtype, npu_formats):
    sources = EplbExpertTensorList(_FakeNpuTensor((3, 4), fmt, dtype=dtype) for fmt in npu_formats)
    format_queries = []
    allocations = []

    def fake_get_npu_format(tensor):
        format_queries.append(tensor)
        return tensor.npu_format

    def fake_empty_with_format(**kwargs):
        allocations.append(kwargs)
        # The CPU tensor stands in for the allocated NPU tensor.
        return torch.empty(kwargs["size"], dtype=kwargs["dtype"])

    monkeypatch.setattr(
        routed_experts_module.torch_npu,
        "get_npu_format",
        fake_get_npu_format,
        raising=False,
    )
    monkeypatch.setattr(
        routed_experts_module.torch_npu,
        "empty_with_format",
        fake_empty_with_format,
        raising=False,
    )

    buffers = torch.empty_like(sources)

    assert isinstance(buffers, EplbExpertTensorList)
    assert [buffer.shape for buffer in buffers] == [
        torch.Size((3, 4)),
        torch.Size((3, 4)),
    ]
    assert all(buffer.dtype == dtype and not buffer.requires_grad for buffer in buffers)
    assert format_queries == list(sources)
    assert allocations == [
        {
            "size": torch.Size((3, 4)),
            "dtype": dtype,
            "layout": torch.strided,
            "device": torch.device("npu"),
            "acl_format": fmt,
        }
        for fmt in npu_formats
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"dtype": torch.float32}, id="changed-dtype"),
        pytest.param({"device": 0}, id="changed-device"),
        pytest.param(
            {"memory_format": torch.contiguous_format},
            id="changed-memory-format",
        ),
        pytest.param({"requires_grad": True}, id="requires-grad"),
    ],
)
def test_eplb_expert_buffer_rejects_property_overrides(kwargs):
    sources = EplbExpertTensorList([_FakeNpuTensor((3, 4), 50)])

    with pytest.raises(TypeError):
        torch.empty_like(sources, **kwargs)


def test_eplb_expert_buffer_rejects_nonzero_storage_offset(monkeypatch):
    source = _FakeNpuTensor((3, 4), 50, storage_offset=1)

    def fail_empty_with_format(**kwargs):
        raise AssertionError("empty_with_format must not be called")

    monkeypatch.setattr(
        routed_experts_module.torch_npu,
        "empty_with_format",
        fail_empty_with_format,
        raising=False,
    )

    with pytest.raises(AssertionError, match="storage_offset=1"):
        routed_experts_module._empty_like_preserving_npu_format(
            source,
        )


def test_eplb_expert_buffer_rejects_stride_mismatch(monkeypatch):
    source = _FakeNpuTensor((3, 4), 50)

    def fake_get_npu_format(tensor):
        return tensor.npu_format

    def fake_empty_with_format(**kwargs):
        del kwargs
        return torch.empty((4, 3)).t()

    monkeypatch.setattr(
        routed_experts_module.torch_npu,
        "get_npu_format",
        fake_get_npu_format,
        raising=False,
    )
    monkeypatch.setattr(
        routed_experts_module.torch_npu,
        "empty_with_format",
        fake_empty_with_format,
        raising=False,
    )

    with pytest.raises(AssertionError, match="source=.*buffer="):
        routed_experts_module._empty_like_preserving_npu_format(
            source,
        )


def test_get_expert_weights_rejects_unsupported_quantization():
    with pytest.raises(NotImplementedError, match="weight views are not defined"):
        list(_routed_experts([]).get_expert_weights())


def test_get_expert_weights_rejects_missing_weight_view_contract():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="must implement get_eplb_weight_views"):
        list(routed_experts.get_expert_weights())


def test_get_expert_weights_rejects_non_expert_first_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        list(_routed_experts([torch.randn(3, 4)]).get_expert_weights())


def test_get_expert_weights_rejects_wrong_expert_tensor_list_length():
    with pytest.raises(ValueError, match="must contain local_num_experts"):
        list(_routed_experts([[torch.randn(3, 4)]]).get_expert_weights())


def test_get_expert_weights_rejects_non_contiguous_view():
    with pytest.raises(ValueError, match="flattenable without a copy"):
        list(_routed_experts([torch.randn(2, 3, 4).transpose(1, 2)]).get_expert_weights())


def test_ascend_expert_map_uses_upstream_map():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    # Both v0.28.0 and main read quant_method in RoutedExperts.expert_map.
    routed_experts.quant_method = SimpleNamespace(moe_kernel=None)
    object.__setattr__(routed_experts, "_expert_map", upstream_map)
    object.__setattr__(routed_experts, "rocm_aiter_fmoe_enabled", False)

    assert routed_experts.ascend_expert_map is upstream_map


def _make_pcp_routed_experts_fixture(*, scoring_func="sqrtsoftplus", tid2eid=None, pcp_size=2):
    moe_config = SimpleNamespace(
        dp_size=1,
        tp_size=1,
        pcp_size=pcp_size,
        ep_size=pcp_size,
        is_sequence_parallel=False,
        num_experts=8,
        original_num_experts=8,
    )
    router = AscendFusedMoERouter(
        top_k=2,
        global_num_experts=8,
        scoring_func=scoring_func,
        tid2eid=tid2eid,
    )
    captured = {}

    def apply_fn(*, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        del shared_experts, shared_experts_input
        captured["x"] = x
        captured["scale"] = layer.ascend_pertoken_scale
        captured["topk_weights"] = topk_weights
        captured["topk_ids"] = topk_ids
        return FusedExpertsResult(routed_out=torch.zeros((x.shape[0], 64), dtype=torch.bfloat16))

    experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    object.__setattr__(experts, "moe_config", moe_config)
    object.__setattr__(experts, "router", router)
    object.__setattr__(
        experts,
        "quant_method",
        SimpleNamespace(
            quant_method=SimpleNamespace(quant_type=QuantType.W4A8MXFP),
            apply=apply_fn,
        ),
    )
    experts.enable_npugraph_ex_static_kernel = False
    experts.apply_router_weight_on_input = False
    experts.n_shared_experts = 0
    experts.global_redundant_expert_num = 0
    experts.log2phy = None
    experts.mix_placement = False
    experts.dynamic_eplb = False
    experts.return_with_event = False
    experts.ascend_pertoken_scale = None
    experts.ascend_mc2_mask = None

    comm = AllGatherCommImpl.__new__(AllGatherCommImpl)
    comm.moe_config = moe_config
    with patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dynamic_mx_quant_scale_alg", return_value=1):
        comm.prepare_finalize = PrepareAndFinalizeWithAllGather(moe_config)
    return experts, comm, captured


def _run_two_rank_pcp_forward(experts, comm, *, capturing, rank0_rows=1):
    tid2eid = experts.router.tid2eid
    rank0_h = torch.full((rank0_rows, 64), 2.0, dtype=torch.bfloat16)
    rank0_logits = torch.arange(rank0_rows * 8, dtype=torch.float32).reshape(rank0_rows, 8)
    rank1_logits = torch.arange(100, 116, dtype=torch.float32).reshape(2, 8)
    rank0_ids = torch.arange(1, 1 + rank0_rows, dtype=torch.int64)
    rank1_ids = torch.tensor([3, 4], dtype=torch.int64)

    pad_r0 = 2 - rank0_rows
    r1_val = torch.arange(128, 256, dtype=torch.uint8).reshape(2, 64).view(torch.float8_e4m3fn)
    r1_scl = torch.tensor([[[122, 123]], [[124, 125]]], dtype=torch.uint8).view(torch.float8_e8m0fnu)

    def fake_quant(x, dst_type, scale_alg):
        del dst_type, scale_alg
        vals = (x.to(torch.int32).to(torch.uint8)).view(torch.float8_e4m3fn)
        scls = torch.full((x.shape[0], 1, 2), 120, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return vals, scls

    def fake_hash_topk(*, x, k, input_ids, tid2eid, **kwargs):
        del kwargs
        scores = torch.nn.functional.softplus(x).sqrt()
        ids = (
            tid2eid[input_ids].to(torch.int32)
            if tid2eid is not None
            else torch.topk(scores, k=k, dim=-1).indices.to(torch.int32)
        )
        weights = scores.gather(1, ids.to(torch.int64)).to(torch.float32)
        return weights, ids, None

    pcp_calls = []

    def fake_all_gather(tensor, dim=0):
        pcp_calls.append(tensor)
        if tensor.dtype == torch.uint8 and tensor.shape == (2, 64 + 2 + 16):
            r1_w, r1_i, _ = fake_hash_topk(x=rank1_logits, k=2, input_ids=rank1_ids, tid2eid=tid2eid)
            r1_router = torch.cat((r1_w.view(torch.int32), r1_i), dim=1).view(torch.uint8)
            remote = torch.cat((r1_val.view(torch.uint8), r1_scl.view(torch.uint8).flatten(1), r1_router), dim=-1)
            return torch.cat((tensor, remote), dim=dim)
        if tensor.dtype == torch.uint8 and tensor.shape == (2, 64 + 2):
            remote = torch.cat((r1_val.view(torch.uint8), r1_scl.view(torch.uint8).flatten(1)), dim=-1)
            return torch.cat((tensor, remote), dim=dim)
        if tensor.dtype == torch.float32 and tensor.shape == (2, 8):
            return torch.cat((tensor, rank1_logits), dim=dim)
        if tensor.ndim == 1 and tensor.shape == (2,):
            return torch.cat((tensor, rank1_ids), dim=dim)
        raise AssertionError(f"Unexpected PCP all_gather payload: shape={tensor.shape}, dtype={tensor.dtype}")

    pcp_group = MagicMock()
    pcp_group.all_gather.side_effect = fake_all_gather
    pcp_group.reduce_scatter.side_effect = lambda tensor, dim=0: tensor[:2]
    extra_ctx = SimpleNamespace(
        in_profile_run=False,
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=comm,
        max_tokens_across_pcp=2,
        capturing=capturing,
        eplb_heat_collection_status=False,
    )
    ascend_cfg = SimpleNamespace(enable_force_eplb=False)
    with (
        patch.object(routed_experts_module, "get_forward_context", return_value=SimpleNamespace(all_moe_layers=[])),
        patch.object(routed_experts_module, "get_moe_comm_method", return_value=comm),
        patch.object(routed_experts_module, "get_ascend_config", return_value=ascend_cfg),
        patch.object(routed_experts_module, "_EXTRA_CTX", extra_ctx),
        patch("vllm_ascend.ops.fused_moe.router.fused_topk_router._EXTRA_CTX", extra_ctx),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", extra_ctx),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_pcp_group", return_value=pcp_group),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_current_hardware_profile") as hw,
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.torch_npu.npu_dynamic_mx_quant", side_effect=fake_quant),
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", side_effect=fake_hash_topk, create=True) as topk_op,
    ):
        hw.return_value.supports.return_value = True
        out = experts.forward_impl(
            hidden_states=rank0_h,
            router_logits=rank0_logits,
            input_ids=rank0_ids if tid2eid is not None else None,
        )
    assert out.shape == (rank0_rows, 64)
    assert pad_r0 >= 0
    return pcp_calls, topk_op


def test_pcp_mxfp8_local_topk_matches_global_routing_in_single_collective():
    tid2eid = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [1, 6], [3, 5]], dtype=torch.int32)
    fast_experts, fast_comm, fast_captured = _make_pcp_routed_experts_fixture(tid2eid=tid2eid)
    fast_calls, fast_topk = _run_two_rank_pcp_forward(fast_experts, fast_comm, capturing=False, rank0_rows=1)

    ref_experts, ref_comm, ref_captured = _make_pcp_routed_experts_fixture(tid2eid=tid2eid)
    ref_calls, ref_topk = _run_two_rank_pcp_forward(ref_experts, ref_comm, capturing=True, rank0_rows=1)

    # Eager prefill packs MXFP8 activations, scales, and local TopK into 1 collective (local rows=2).
    assert len(fast_calls) == 1
    assert fast_calls[0].shape == (2, 64 + 2 + 16)
    assert fast_topk.call_args.kwargs["x"].shape == (2, 8)
    # Graph capture falls back to global TopK after separate collectives (global rows=4).
    assert len(ref_calls) == 3
    assert ref_topk.call_args.kwargs["x"].shape == (4, 8)

    torch.testing.assert_close(fast_captured["x"].view(torch.uint8), ref_captured["x"].view(torch.uint8))
    torch.testing.assert_close(fast_captured["scale"].view(torch.uint8), ref_captured["scale"].view(torch.uint8))
    torch.testing.assert_close(fast_captured["topk_weights"], ref_captured["topk_weights"])
    torch.testing.assert_close(fast_captured["topk_ids"], ref_captured["topk_ids"])


def test_pcp_mxfp8_local_topk_keeps_spmd_collective_shape_on_empty_local_shard():
    tid2eid = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [1, 6], [3, 5]], dtype=torch.int32)
    experts, comm, captured = _make_pcp_routed_experts_fixture(tid2eid=tid2eid)
    pcp_calls, topk_op = _run_two_rank_pcp_forward(experts, comm, capturing=False, rank0_rows=0)

    assert len(pcp_calls) == 1
    assert pcp_calls[0].shape == (2, 64 + 2 + 16)
    assert topk_op.call_args.kwargs["x"].shape == (2, 8)
    assert captured["topk_ids"].shape == (4, 2)


@pytest.mark.parametrize(
    "mutator,expect_mxfp8_prepare",
    [
        pytest.param(lambda e: setattr(e.moe_config, "pcp_size", 1), False, id="pcp-1"),
        pytest.param(lambda e: setattr(e.router, "scoring_func", "sigmoid"), False, id="non-sqrtsoftplus"),
        pytest.param(lambda e: setattr(e, "apply_router_weight_on_input", True), False, id="weight-on-input"),
        pytest.param(lambda e: setattr(e, "_ascend_moe_lora_context", object()), False, id="lora-active"),
        pytest.param(lambda e: setattr(e.router, "bias_vl", torch.ones(8)), True, id="vision-router"),
        pytest.param(lambda e: setattr(e.router, "is_routing_input_ids_producer", True), True, id="ids-producer"),
    ],
)
def test_pcp_mxfp8_and_local_routing_guards(mutator, expect_mxfp8_prepare):
    experts, comm, _ = _make_pcp_routed_experts_fixture()
    mutator(experts)
    lora_ctx = getattr(experts, "_ascend_moe_lora_context", None)
    hidden = torch.ones((2, 64), dtype=torch.bfloat16)
    extra_ctx = SimpleNamespace(max_tokens_across_pcp=2, capturing=False)
    ascend_cfg = SimpleNamespace(enable_force_eplb=False)
    with (
        patch.object(routed_experts_module, "_EXTRA_CTX", extra_ctx),
        patch.object(routed_experts_module, "get_ascend_config", return_value=ascend_cfg),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_current_hardware_profile") as hw,
    ):
        hw.return_value.supports.return_value = True
        supports_mxfp8 = experts._supports_pcp_mxfp8_prepare(comm, lora_ctx)
        can_pre_route = supports_mxfp8 and experts._can_pre_route_pcp(hidden, QuantType.W4A8MXFP, False)
    assert supports_mxfp8 is expect_mxfp8_prepare
    assert not can_pre_route
