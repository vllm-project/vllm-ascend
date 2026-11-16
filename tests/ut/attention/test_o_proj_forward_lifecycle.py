# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFAPCPDCPImpl,
    AscendSFAPCPImpl,
)
from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
from vllm_ascend.weight_switch import (
    OProjectionMode,
    ParallelOProjection,
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchMixin,
)


class _LinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)


def _make_projection(layer):
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    return ParallelOProjection(
        layer.input_size_per_partition,
        layer,
        config=WeightSwitchConfig.from_group(group, shard_axis="input"),
        tp_group=group,
        pool={},
        pool_key="lifecycle_test",
        project_full=lambda inputs, output: output,
        decode_sharded=False,
    )


@pytest.fixture(
    params=[AscendSFAPCPImpl, AscendSFAPCPDCPImpl],
    ids=["pcp", "pcp_dcp"],
)
def sfa_impl(request, monkeypatch):
    impl = request.param.__new__(request.param)
    impl.o_proj_full_pools = {}
    impl.tp_size = 2
    impl.enable_pcp_o_proj_weight_sharding = True
    impl.is_pcp_decode_sharded = False
    impl.o_proj = torch.nn.Module()
    impl.o_proj.input_size = 4
    impl.o_proj.input_size_per_partition = 2
    impl.o_proj.output_size = impl.o_proj.output_size_per_partition = 3
    impl.o_proj.weight = torch.nn.Parameter(torch.arange(6.0).reshape(3, 2), requires_grad=False)
    impl.o_proj.quant_method = _LinearMethod()
    impl._parallel_o_proj = _make_projection(impl.o_proj)
    impl._parallel_o_proj.enabled = True
    impl._parallel_o_proj.enable()
    impl.waits = Mock()

    def gather(state, config):
        assert not state.handles
        state.handles.append(SimpleNamespace(wait=impl.waits))
        state.gather_parts["weight"].full_tensor.fill_(1)

    monkeypatch.setattr(impl.o_proj.quant_method, "all_gather_weight", gather)
    return impl


def _metadata(attn_state, *, global_prefill=False):
    lengths = torch.tensor([1])
    slots = torch.tensor([0])
    metadata = object.__new__(AscendSFADCPMetadata)
    vars(metadata).update(
        attn_state=attn_state,
        pcp_has_global_prefill=global_prefill,
        num_prefills=int(global_prefill),
        num_input_tokens=1,
        cum_query_lens=lengths,
        seq_lens=lengths,
        slot_mapping=slots,
        pcp_slot_mapping=slots,
        dcp_context=SimpleNamespace(slot_mapping=slots),
    )

    return metadata


def _prepare_forward(impl, metadata, hidden):
    context = impl._get_parallel_forward_context(metadata, 1, hidden)
    assert not hasattr(context, "gather_full_o_proj")


def test_sfa_prefill_decode_and_profile_share_one_mode(sfa_impl, monkeypatch):
    entry_modes: list[bool] = []
    modes: list[bool] = []
    local_address = sfa_impl.o_proj.weight.data_ptr()
    weight_state = sfa_impl._parallel_o_proj._weight_switch_states[0][2]

    def forward(impl, name, hidden, cache, metadata, output):
        entry_modes.append(impl._parallel_o_proj.uses_full_weight)
        if metadata is None:
            return output.zero_()
        _prepare_forward(impl, metadata, hidden)
        modes.append(impl._parallel_o_proj.uses_full_weight)
        if impl._parallel_o_proj.uses_full_weight:
            with impl._parallel_o_proj.use_full_weights():
                assert impl.o_proj.weight.data_ptr() == weight_state.gather_parts["weight"].full_tensor.data_ptr()
        else:
            assert impl.o_proj.weight.data_ptr() == local_address
        return output

    monkeypatch.setattr(AscendSFAImpl, "forward", forward)
    hidden = torch.empty(1, 4)
    output = torch.empty_like(hidden)
    for state in (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    ):
        assert sfa_impl.forward("layer", hidden, (), _metadata(state), output) is output
        assert not sfa_impl._parallel_o_proj.uses_full_weight
        assert not weight_state.handles
        assert sfa_impl.o_proj.weight.data_ptr() == local_address
    assert modes == [True, False, False]
    sfa_impl.forward("layer", hidden, (), None, output)
    assert not sfa_impl._parallel_o_proj.uses_full_weight
    assert sfa_impl.waits.call_count == 1
    assert entry_modes == ([True, False, False, False] if isinstance(sfa_impl, AscendSFAPCPImpl) else [False] * 4)


@pytest.mark.parametrize("sfa_impl", [AscendSFAPCPImpl, AscendSFAPCPDCPImpl], indirect=True)
@pytest.mark.parametrize(
    "state,global_prefill,decode_sharded,switch_enabled,expected_mode",
    [
        (AscendAttentionState.DecodeOnly, False, False, True, OProjectionMode.LOCAL_WEIGHT),
        (AscendAttentionState.DecodeOnly, False, True, True, OProjectionMode.EXCHANGE_TOKENS),
        (AscendAttentionState.SpecDecoding, False, True, True, OProjectionMode.EXCHANGE_TOKENS),
        (AscendAttentionState.ChunkedPrefill, False, False, True, OProjectionMode.FULL_WEIGHT),
        (AscendAttentionState.ChunkedPrefill, False, True, True, OProjectionMode.FULL_WEIGHT),
        (AscendAttentionState.DecodeOnly, True, True, True, OProjectionMode.FULL_WEIGHT),
        (None, False, True, True, OProjectionMode.LOCAL_WEIGHT),
        (AscendAttentionState.ChunkedPrefill, False, True, False, OProjectionMode.LOCAL_WEIGHT),
    ],
    ids=[
        "replicated_decode",
        "sharded_decode",
        "sharded_spec_decode",
        "prefill",
        "prefill_with_sharded_decode",
        "short_global_prefill",
        "profile",
        "switch_disabled",
    ],
)
def test_sfa_pcp_forward_selects_batch_mode_before_attention(
    sfa_impl, monkeypatch, state, global_prefill, decode_sharded, switch_enabled, expected_mode
):
    sfa_impl.is_pcp_decode_sharded = decode_sharded
    sfa_impl.enable_pcp_o_proj_weight_sharding = switch_enabled
    if not switch_enabled:
        sfa_impl._parallel_o_proj = _make_projection(sfa_impl.o_proj)
    projection = sfa_impl._parallel_o_proj
    projection.decode_sharded = decode_sharded
    entry_modes: list[OProjectionMode] = []
    local_address = sfa_impl.o_proj.weight.data_ptr()

    def forward(impl, name, hidden, cache, metadata, output):
        entry_modes.append(projection.mode)
        if metadata is not None:
            _prepare_forward(impl, metadata, hidden)
            assert projection.mode is expected_mode
        return output.zero_()

    monkeypatch.setattr(AscendSFAImpl, "forward", forward)
    hidden = torch.empty(1, 4)
    output = torch.full((1, 3), float("nan"))
    metadata = None if state is None else _metadata(state, global_prefill=global_prefill)
    assert sfa_impl.forward("layer", hidden, (), metadata, output) is output
    assert torch.count_nonzero(output) == 0
    assert entry_modes == [expected_mode]
    assert projection.mode is OProjectionMode.LOCAL_WEIGHT
    assert all(not switch_state.handles for _, _, switch_state in projection._weight_switch_states or ())
    assert sfa_impl.o_proj.weight.data_ptr() == local_address
    assert sfa_impl.waits.call_count == int(expected_mode is OProjectionMode.FULL_WEIGHT)


@pytest.mark.parametrize("stage", ["before_projection", "projection"])
def test_failed_sfa_prefill_restores_local_weights_and_next_decode(sfa_impl, monkeypatch, stage):
    failure = RuntimeError("injected attention failure")
    local_address = sfa_impl.o_proj.weight.data_ptr()
    weight_state = sfa_impl._parallel_o_proj._weight_switch_states[0][2]

    def forward(impl, name, hidden, cache, metadata, output):
        _prepare_forward(impl, metadata, hidden)
        if metadata.attn_state == AscendAttentionState.DecodeOnly:
            assert not impl._parallel_o_proj.uses_full_weight
            assert not weight_state.handles
            return output
        assert impl._parallel_o_proj.uses_full_weight
        if stage == "before_projection":
            assert weight_state.handles
            raise failure
        with impl._parallel_o_proj.use_full_weights():
            assert impl.o_proj.weight.data_ptr() != local_address
            raise failure

    monkeypatch.setattr(AscendSFAImpl, "forward", forward)
    hidden = torch.empty(1, 4)
    output = torch.empty_like(hidden)
    with pytest.raises(RuntimeError) as caught:
        sfa_impl.forward("layer", hidden, (), _metadata(AscendAttentionState.ChunkedPrefill), output)
    assert caught.value is failure
    assert not sfa_impl._parallel_o_proj.uses_full_weight
    assert not weight_state.handles
    assert sfa_impl.o_proj.weight.data_ptr() == local_address
    assert sfa_impl.waits.call_count == 1
    sfa_impl.forward("layer", hidden, (), _metadata(AscendAttentionState.DecodeOnly), output)


def test_all_gather_requires_prepared_switches_without_enabling():
    layer = torch.nn.Module()
    layer.input_size_per_partition = 2
    layer.output_size_per_partition = 3
    projection = _make_projection(layer)
    projection.enable = Mock(side_effect=AssertionError("all-gather must not enable"))
    method = Mock()
    state = SimpleNamespace(handles=[])
    projection._weight_switch_states = ((layer, method, state),)
    projection._weight_switch_enabled = True
    projection.set_mode(use_full_weight=True, exchange_tokens=False)

    projection.all_gather_weights()

    projection.enable.assert_not_called()
    method.all_gather_weight.assert_called_once_with(state, projection.config)


def test_dsa_cleanup_waits_every_layer_without_hiding_attention_failure(monkeypatch):
    layer = torch.nn.Module()
    layer.input_size_per_partition = 2
    layer.output_size_per_partition = 3
    projection = _make_projection(layer)
    projection._weight_switch_enabled = True
    attention_failure = RuntimeError("attention failed")
    first = SimpleNamespace(handles=[SimpleNamespace(wait=Mock(side_effect=RuntimeError("wait failed")))])
    second_wait = Mock()
    second = SimpleNamespace(handles=[SimpleNamespace(wait=second_wait)])
    projection._weight_switch_states = tuple((torch.nn.Module(), _LinearMethod(), state) for state in (first, second))

    with pytest.raises(RuntimeError) as caught, projection.forward_context():
        projection.set_mode(use_full_weight=True, exchange_tokens=False)
        raise attention_failure
    assert caught.value is attention_failure
    second_wait.assert_called_once_with()
    assert not first.handles and not second.handles
    assert not projection.uses_full_weight
    assert projection.mode is OProjectionMode.LOCAL_WEIGHT


@pytest.mark.parametrize(
    "backend,implementation",
    [(AscendDSASWABackend, AscendDSAPCPImpl), (AscendSFABackend, AscendSFAPCPImpl)],
)
def test_upstream_backend_capability_accepts_the_selected_pcp_implementation(backend, implementation, monkeypatch):
    monkeypatch.setattr(backend, "get_impl_cls", staticmethod(lambda: implementation))
    assert backend.supports_pcp()


@pytest.mark.parametrize("fail", [False, True])
def test_native_sfa_dcp_forward_needs_no_projection_component(monkeypatch, fail):
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    failure = RuntimeError("native DCP attention failed")
    output = torch.empty(1, 3)

    def forward(owner, name, hidden, cache, metadata, result):
        assert owner is impl
        assert not hasattr(owner, "_parallel_o_proj")
        if fail:
            raise failure
        return result.zero_()

    monkeypatch.setattr(AscendSFAImpl, "forward", forward)
    if fail:
        with pytest.raises(RuntimeError) as caught:
            impl.forward("layer", torch.empty(1, 4), (), None, output)
        assert caught.value is failure
    else:
        assert impl.forward("layer", torch.empty(1, 4), (), None, output) is output


@pytest.mark.parametrize("switch_enabled", [False, True])
@pytest.mark.parametrize("impl_cls", [AscendSFADCPImpl, AscendSFAPCPDCPImpl])
def test_sfa_variant_constructors_and_postload_cooperate(impl_cls, switch_enabled, monkeypatch):
    from vllm_ascend.attention.context_parallel import common_cp, sfa_cp

    group = SimpleNamespace(world_size=2, rank_in_group=0, device_group=object())
    owner = torch.nn.Module()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=4), dtype=torch.float32),
        compilation_config=SimpleNamespace(static_forward_context={"layer": owner}),
    )
    base_calls = []
    loaded = []

    def initialize_base(impl, *args, **kwargs):
        base_calls.append(impl)
        impl.vllm_config = config
        impl.layer_name = "layer"
        impl.tp_size, impl.num_heads = 2, 2
        impl.enable_mlapo = True
        impl.is_pcp_decode_sharded = False
        impl.q_proj = torch.nn.Linear(4, 4, bias=False)
        impl.o_proj = torch.nn.Module()
        impl.o_proj.input_size, impl.o_proj.input_size_per_partition = 8, 4
        impl.o_proj.output_size = impl.o_proj.output_size_per_partition = 3
        impl.o_proj.weight = torch.nn.Parameter(torch.ones(3, 4), requires_grad=False)
        impl.o_proj.weight.input_dim = 1
        impl.o_proj.weight.weight_loader = Mock()
        impl.o_proj.quant_method = _LinearMethod()
        impl.o_proj_full_pools = {}

    def process_base(impl, dtype):
        loaded.append(impl)
        return "loaded"

    monkeypatch.setattr(AscendSFAImpl, "__init__", initialize_base)
    monkeypatch.setattr(AscendSFAImpl, "process_weights_after_loading", process_base)
    monkeypatch.setattr(sfa_cp, "enable_pcp_o_proj_weight_sharding", lambda: switch_enabled)
    for module in (common_cp, sfa_cp):
        for name in ("get_tp_group", "get_pcp_group", "get_dcp_group"):
            monkeypatch.setattr(module, name, lambda: group, raising=False)
    impl = impl_cls(2, 4, 1.0, 1, None, None, "auto", None, "decoder", None)
    assert base_calls == [impl]
    has_component = impl_cls is not AscendSFADCPImpl
    assert hasattr(impl, "_parallel_o_proj") == has_component
    if has_component:
        assert impl._parallel_o_proj.enabled == switch_enabled
        assert impl._parallel_o_proj.project_full.__self__ is impl
        assert impl._parallel_o_proj.project_full.__func__ is AscendSFAImpl._finalize_o_proj
    assert impl.process_weights_after_loading(torch.float32) == "loaded"
    states = impl._parallel_o_proj._weight_switch_states if has_component else None
    assert (states is not None) == (has_component and switch_enabled)
    assert impl.process_weights_after_loading(torch.float32) == "loaded"
    assert loaded == [impl, impl]
    if has_component:
        assert impl._parallel_o_proj._weight_switch_states is states
    if isinstance(impl, AscendSFADCPImpl):
        assert not impl.enable_mlapo
        assert owner._dcp_sfa_remap_order is impl._remap_order
        assert owner._dcp_sfa_remap_invalid_index is impl._remap_invalid_index
