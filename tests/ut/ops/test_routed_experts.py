# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList


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


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_ascend_expert_map_follows_model_runner(use_v2_model_runner):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    upstream_map = torch.tensor([0, 1], dtype=torch.int32)
    object.__setattr__(routed_experts, "_use_v2_model_runner", use_v2_model_runner)
    # Both v0.28.0 and main read quant_method in RoutedExperts.expert_map.
    routed_experts.quant_method = SimpleNamespace(moe_kernel=None)
    routed_experts.ascend_expert_map = legacy_map
    object.__setattr__(routed_experts, "_expert_map", upstream_map)
    object.__setattr__(routed_experts, "rocm_aiter_fmoe_enabled", False)

    expected = upstream_map if use_v2_model_runner else legacy_map
    assert routed_experts.ascend_expert_map is expected


def test_update_expert_map_preserves_upstream_and_legacy_contracts(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    parent_update_calls = []

    def parent_update(instance):
        parent_update_calls.append(instance)

    monkeypatch.setattr(type(routed_experts).__mro__[1], "update_expert_map", parent_update)
    expert_map_manager = SimpleNamespace(_expert_map=None)
    object.__setattr__(routed_experts, "expert_map_manager", expert_map_manager)

    routed_experts.update_expert_map()

    assert parent_update_calls == [routed_experts]

    legacy_map = torch.tensor([1, 0], dtype=torch.int32)
    routed_experts.update_expert_map(legacy_map)

    assert routed_experts.ascend_expert_map is legacy_map
    assert expert_map_manager._expert_map is legacy_map


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("disabled_by", [None, "comm", "pcp", "sp", "router"])
def test_local_routing_capability_is_independent_of_dp_and_features(dp_size, disabled_by):
    layer = SimpleNamespace(
        router=SimpleNamespace(supports_local_routing=True),
        moe_config=SimpleNamespace(dp_size=dp_size, pcp_size=8, is_sequence_parallel=False),
        dynamic_eplb=True,
        mix_placement=True,
        _ascend_moe_lora_context=object(),
    )
    comm = MagicMock(spec=routed_experts_module.AllGatherCommImpl)
    if disabled_by == "comm":
        comm = object()
    elif disabled_by == "pcp":
        layer.moe_config.pcp_size = 1
    elif disabled_by == "sp":
        layer.moe_config.is_sequence_parallel = True
    elif disabled_by == "router":
        layer.router.supports_local_routing = False
    assert AscendRoutedExperts._can_select_experts_before_gather(layer, comm) == (disabled_by is None)


@pytest.mark.parametrize("tokens", [48, 1023, 1024, 2000])
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("feature", ["plain", "capture_mapping", "mix", "dynamic", "profile", "force", "lora"])
def test_forward_local_topk_preserves_global_postprocessing(monkeypatch, tokens, dp_size, feature):
    config = SimpleNamespace(
        dp_size=dp_size,
        pcp_size=2,
        tp_size=1,
        is_sequence_parallel=False,
        num_experts=12,
        num_logical_experts=12,
    )
    config.dp_group = SimpleNamespace(all_gather=lambda x, dim: torch.cat([x] * dp_size, dim=dim))
    prepare = object.__new__(prepare_module.PrepareAndFinalizeWithAllGather)
    prepare.moe_config = config
    group = SimpleNamespace(
        all_gather=lambda x, dim: torch.cat([x] * 2, dim=dim),
        reduce_scatter=lambda x, dim: x.chunk(2, dim=dim)[0].contiguous(),
    )
    context = SimpleNamespace(
        in_profile_run=feature == "profile",
        max_tokens_across_dp=tokens,
        max_tokens_across_pcp=tokens * dp_size,
        moe_comm_type=None,
        moe_comm_method=None,
        eplb_heat_collection_status=feature == "dynamic",
    )
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", context)
    monkeypatch.setattr(prepare_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr(prepare_module, "get_dp_group", lambda: group)
    monkeypatch.setattr(routed_experts_module, "_EXTRA_CTX", context)
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        routed_experts_module, "get_ascend_config", lambda: SimpleNamespace(enable_force_eplb=feature == "force")
    )
    monkeypatch.setattr(
        routed_experts_module,
        "get_force_eplb_topk",
        lambda ids, experts: (torch.arange(ids.numel()).reshape(ids.shape) % experts).to(ids.dtype),
    )
    sync_calls = []
    monkeypatch.setattr(routed_experts_module, "sync_lora_context", lambda q, c: sync_calls.append(c))
    comm = MagicMock(spec=routed_experts_module.AllGatherCommImpl)
    comm.prepare_finalize, comm.prepare, comm.finalize = prepare, prepare.prepare, prepare.finalize
    monkeypatch.setattr(routed_experts_module, "get_moe_comm_method", lambda *args: comm)
    selection_rows, capture_rows = [], []

    def compute(hidden_states, router_logits, **kwargs):
        selection_rows.append(len(router_logits))
        # Exact arithmetic isolates branch/transport semantics from CPU sigmoid SIMD.
        scores = router_logits.abs() + 1
        ids = scores.topk(4, dim=-1).indices.int()
        weights = scores.gather(1, ids.long())
        return weights / weights.sum(-1, keepdim=True), ids

    def finish(weights, ids):
        capture_rows.append(len(ids))
        if feature == "capture_mapping":
            # Global row-index dependence exposes accidentally local mapping.
            ids = (ids + torch.arange(len(ids)).unsqueeze(-1)) % 12
        return weights, ids

    router = SimpleNamespace(
        supports_local_routing=True,
        select_experts_locally=compute,
        finalize_local_routing=finish,
        _select_experts=lambda hidden_states, router_logits, **kw: finish(*compute(hidden_states, router_logits)),
    )
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer.router, layer.moe_config = router, config
    layer.dynamic_eplb, layer.mix_placement = feature == "dynamic", feature == "mix"
    layer.enable_npugraph_ex_static_kernel = layer._use_v2_model_runner = layer.return_with_event = False
    layer.global_redundant_expert_num, layer.n_shared_experts = 0, 2 if feature == "mix" else 0
    layer.log2phy = None
    layer.multi_stage, layer.moe_load = False, torch.zeros(14, dtype=torch.int64)
    layer._ascend_moe_lora_context = object() if feature == "lora" else None
    torch.manual_seed(17)
    hidden, logits = torch.randn(tokens, 16), torch.randn(tokens, 12)
    global_hidden, global_logits = torch.cat([hidden] * (2 * dp_size)), torch.cat([logits] * (2 * dp_size))
    torch.manual_seed(23)
    expected_w, expected_i = layer._select_experts(global_hidden, global_logits, feature == "profile")
    selection_rows.clear()
    capture_rows.clear()
    apply_calls = []

    def apply(**kwargs):
        apply_calls.append(kwargs)
        torch.testing.assert_close(kwargs["topk_weights"], expected_w, rtol=0, atol=0)
        assert torch.equal(kwargs["topk_ids"], expected_i)
        assert torch.equal(kwargs["x"], global_hidden)
        value = (kwargs["topk_weights"] * kwargs["topk_ids"]).sum(-1, keepdim=True)
        return SimpleNamespace(
            routed_out=value.expand(-1, 16).contiguous(),
            expert_tokens=torch.bincount(kwargs["topk_ids"].flatten().long(), minlength=14),
            group_list_type=1,
        )

    layer.quant_method = SimpleNamespace(
        apply=apply, quant_method=SimpleNamespace(quant_type=routed_experts_module.QuantType.W8A8)
    )
    torch.manual_seed(23)
    actual = layer.forward_impl(hidden_states=hidden, router_logits=logits)
    expected = (expected_w * expected_i).sum(-1, keepdim=True)[:tokens].expand(-1, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    optimized = tokens * dp_size >= 1024
    assert selection_rows == [tokens * dp_size if optimized else tokens * dp_size * 2]
    assert capture_rows == [tokens * dp_size * 2]
    assert len(apply_calls) == 1
    if feature == "dynamic":
        assert torch.equal(layer.moe_load, torch.bincount(expected_i.flatten().long(), minlength=14))
    if feature == "lora":
        assert sync_calls == [layer._ascend_moe_lora_context, None]
    assert layer.ascend_pertoken_scale is None and layer.ascend_mc2_mask is None
