# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList
from vllm_ascend.ops.fused_moe.router import AscendFusedMoERouter
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
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


def test_ascend_expert_map_uses_upstream_placement():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    routed_experts.quant_method = SimpleNamespace(moe_kernel=None)
    object.__setattr__(routed_experts, "_expert_map", upstream_map)

    assert routed_experts.ascend_expert_map is upstream_map


def _make_pcp_routed_experts_fixture(*, scoring_func="sqrtsoftplus", tid2eid=None, pcp_size=2):
    moe_config = SimpleNamespace(
        dp_size=1,
        tp_size=1,
        tp_group=SimpleNamespace(world_size=1),
        pcp_size=pcp_size,
        ep_size=pcp_size,
        is_sequence_parallel=False,
        num_experts=8,
        original_num_experts=8,
    )
    captured: dict[str, Any] = {}

    def apply_fn(*, layer, x, topk_weights, topk_ids, **_kwargs):
        captured.update(x=x, scale=layer.ascend_pertoken_scale, topk_weights=topk_weights, topk_ids=topk_ids)
        return FusedExpertsResult(routed_out=torch.zeros((x.shape[0], 64), dtype=torch.bfloat16))

    experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    for key, val in {
        "moe_config": moe_config,
        "router": AscendFusedMoERouter(top_k=2, global_num_experts=8, scoring_func=scoring_func, tid2eid=tid2eid),
        "quant_method": SimpleNamespace(quant_method=SimpleNamespace(quant_type=QuantType.W4A8MXFP), apply=apply_fn),
        "enable_npugraph_ex_static_kernel": False,
        "apply_router_weight_on_input": False,
        "n_shared_experts": 0,
        "global_redundant_expert_num": 0,
        "mix_placement": False,
        "return_with_event": False,
        "ascend_pertoken_scale": None,
        "ascend_mc2_mask": None,
    }.items():
        object.__setattr__(experts, key, val)

    comm = AllGatherCommImpl.__new__(AllGatherCommImpl)
    comm.moe_config = moe_config
    with patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_dynamic_mx_quant_scale_alg", return_value=1):
        comm.prepare_finalize = PrepareAndFinalizeWithAllGather(moe_config)
    return experts, comm, captured


@contextmanager
def _pcp_forward_env(comm, *, capturing=False):
    extra_ctx = SimpleNamespace(
        in_profile_run=False,
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=comm,
        max_tokens_across_pcp=2,
        capturing=capturing,
    )
    with (
        patch.object(routed_experts_module, "get_forward_context", return_value=SimpleNamespace(all_moe_layers=[])),
        patch.object(routed_experts_module, "get_moe_comm_method", return_value=comm),
        patch.object(routed_experts_module, "get_ascend_config", return_value=SimpleNamespace(enable_force_eplb=False)),
        patch.object(routed_experts_module, "sync_lora_context"),
        patch.object(routed_experts_module, "_EXTRA_CTX", extra_ctx),
    ):
        yield extra_ctx


def _run_two_rank_pcp_forward(experts, comm, *, capturing, rank0_rows=1):
    tid2eid = experts.router.tid2eid
    rank0_h = torch.full((rank0_rows, 64), 2.0, dtype=torch.bfloat16)
    rank0_logits = torch.arange(rank0_rows * 8, dtype=torch.float32).reshape(rank0_rows, 8)
    rank1_logits = torch.arange(100, 116, dtype=torch.float32).reshape(2, 8)
    rank0_ids = torch.arange(1, 1 + rank0_rows, dtype=torch.int64)
    rank1_ids = torch.tensor([3, 4], dtype=torch.int64)
    r1_val = torch.arange(128, 256, dtype=torch.uint8).reshape(2, 64)
    r1_scl = torch.tensor([[[122, 123]], [[124, 125]]], dtype=torch.uint8)

    def fake_quant(x, **_kwargs):
        return (
            x.to(torch.int32).to(torch.uint8).view(torch.float8_e4m3fn),
            torch.full((x.shape[0], 1, 2), 120, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        )

    def fake_hash_topk(*, x, k, input_ids, tid2eid, **_kwargs):
        scores = torch.nn.functional.softplus(x).sqrt()
        ids = (
            tid2eid[input_ids].to(torch.int32)
            if tid2eid is not None
            else torch.topk(scores, k=k, dim=-1).indices.to(torch.int32)
        )
        return scores.gather(1, ids.to(torch.int64)).to(torch.float32), ids, None

    pcp_calls: list[torch.Tensor] = []

    def fake_all_gather(tensor, dim=0):
        pcp_calls.append(tensor)
        if tensor.dtype == torch.float8_e4m3fn and tensor.shape == (2, 64):
            return torch.cat((tensor.view(torch.uint8), r1_val), dim=dim).view(torch.float8_e4m3fn)
        if tensor.dtype == torch.float8_e8m0fnu and tensor.shape == (2, 1, 2):
            return torch.cat((tensor.view(torch.uint8), r1_scl), dim=dim).view(torch.float8_e8m0fnu)
        if tensor.dtype == torch.int32 and tensor.shape == (2, 4):
            r1_w, r1_i, _ = fake_hash_topk(x=rank1_logits, k=2, input_ids=rank1_ids, tid2eid=tid2eid)
            remote = torch.cat((r1_w.view(torch.int32), r1_i), dim=1)
        elif tensor.dtype == torch.float32 and tensor.shape == (2, 8):
            remote = rank1_logits
        elif tensor.ndim == 1 and tensor.shape == (2,):
            remote = rank1_ids
        else:
            raise AssertionError(f"Unexpected PCP all_gather payload: shape={tensor.shape}, dtype={tensor.dtype}")
        return torch.cat((tensor, remote), dim=dim)

    pcp_group = MagicMock()
    pcp_group.all_gather.side_effect = fake_all_gather
    pcp_group.reduce_scatter.side_effect = lambda tensor, dim=0: tensor[:2]
    with (
        _pcp_forward_env(comm, capturing=capturing) as extra_ctx,
        patch("vllm_ascend.ops.fused_moe.router.fused_topk_router._EXTRA_CTX", extra_ctx),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", extra_ctx),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_pcp_group", return_value=pcp_group),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.torch_npu.npu_dynamic_mx_quant", side_effect=fake_quant),
        patch.object(torch.ops._C_ascend, "moe_gating_top_k_hash", side_effect=fake_hash_topk, create=True) as topk_op,
    ):
        out = experts.forward_impl(
            hidden_states=rank0_h, router_logits=rank0_logits, input_ids=rank0_ids if tid2eid is not None else None
        )
    assert out.shape == (rank0_rows, 64)
    return pcp_calls, topk_op


@pytest.mark.parametrize(
    "rank0_rows",
    [pytest.param(1, id="non-empty"), pytest.param(0, id="empty-local-shard")],
)
def test_pcp_mxfp8_local_topk_matches_global_routing(rank0_rows):
    tid2eid = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [1, 6], [3, 5]], dtype=torch.int32)
    fast_experts, fast_comm, fast_cap = _make_pcp_routed_experts_fixture(tid2eid=tid2eid)
    fast_calls, fast_topk = _run_two_rank_pcp_forward(fast_experts, fast_comm, capturing=False, rank0_rows=rank0_rows)
    ref_experts, ref_comm, ref_cap = _make_pcp_routed_experts_fixture(tid2eid=tid2eid)
    ref_calls, ref_topk = _run_two_rank_pcp_forward(ref_experts, ref_comm, capturing=True, rank0_rows=rank0_rows)

    assert [(t.shape, t.dtype) for t in fast_calls] == [
        ((2, 64), torch.float8_e4m3fn),
        ((2, 4), torch.int32),
        ((2, 1, 2), torch.float8_e8m0fnu),
    ]
    assert fast_topk.call_args.kwargs["x"].shape == (2, 8)
    assert len(ref_calls) == 4 and ref_topk.call_args.kwargs["x"].shape == (4, 8)
    for key in ("x", "scale"):
        torch.testing.assert_close(fast_cap[key].view(torch.uint8), ref_cap[key].view(torch.uint8))
    for key in ("topk_weights", "topk_ids"):
        torch.testing.assert_close(fast_cap[key], ref_cap[key])


class _CustomTopKRouter(AscendFusedMoERouter):
    pass


@pytest.mark.parametrize(
    "target,attr,value,expected_quant_type",
    [
        pytest.param("moe_config", "pcp_size", 1, QuantType.W4A8MXFP, id="pcp-1"),
        pytest.param(
            "moe_config",
            "tp_group",
            SimpleNamespace(world_size=2),
            QuantType.W4A8MXFP,
            id="physical-tp-gt-1",
        ),
        pytest.param("quant", "quant_type", QuantType.W8A8, QuantType.W8A8, id="non-w4a8mxfp"),
        pytest.param("experts", "router", _CustomTopKRouter, QuantType.NONE, id="router-subclass"),
        pytest.param("experts", "router", AscendGroupedTopKRouter, QuantType.W4A8MXFP, id="grouped-topk-router"),
        pytest.param("experts", "apply_router_weight_on_input", True, QuantType.NONE, id="weight-on-input"),
        pytest.param("experts", "_ascend_moe_lora_context", object(), QuantType.NONE, id="lora-active"),
        pytest.param("router", "scoring_func", "sigmoid", QuantType.W4A8MXFP, id="non-sqrtsoftplus"),
        pytest.param("router", "bias_vl", torch.ones(8), QuantType.W4A8MXFP, id="vision-router"),
        pytest.param("router", "is_routing_input_ids_producer", True, QuantType.W4A8MXFP, id="ids-producer"),
    ],
)
def test_pcp_mxfp8_and_local_routing_guards(target: str, attr: str, value: Any, expected_quant_type: QuantType):
    experts, comm, _ = _make_pcp_routed_experts_fixture()
    if value is _CustomTopKRouter:
        value = _CustomTopKRouter(top_k=2, global_num_experts=8, scoring_func="sqrtsoftplus")
    elif value is AscendGroupedTopKRouter:
        value = AscendGroupedTopKRouter(top_k=2, global_num_experts=8, num_expert_group=None, topk_group=None)
    targets = {
        "experts": experts,
        "router": experts.router,
        "moe_config": experts.moe_config,
        "quant": experts.quant_method.quant_method,
    }
    object.__setattr__(targets[target], attr, value)
    hidden = torch.ones((2, 64), dtype=torch.bfloat16)
    logits = torch.zeros((2, 8), dtype=torch.float32)
    prepare_out = MoEPrepareOutput(hidden, logits, None, None, None)
    with (
        _pcp_forward_env(comm),
        patch.object(comm, "prepare", return_value=prepare_out) as prepare_mock,
        patch.object(comm, "finalize", side_effect=lambda hidden_states, **_kw: hidden_states),
        patch.object(
            AscendRoutedExperts,
            "_select_experts",
            return_value=(torch.ones((2, 2)), torch.zeros((2, 2), dtype=torch.int32)),
        ) as select_experts,
    ):
        experts.forward_impl(hidden_states=hidden, router_logits=logits)
    assert prepare_mock.call_args.kwargs["quant_type"] == expected_quant_type
    assert prepare_mock.call_args.kwargs["router_logits"].dtype == torch.float32
    select_experts.assert_called_once()
