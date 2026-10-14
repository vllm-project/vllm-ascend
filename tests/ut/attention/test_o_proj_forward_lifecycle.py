# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADSACPImpl,
    AscendSFAPCPImpl,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.weight_switch import WeightSwitchConfig, WeightSwitchGatherSpec, WeightSwitchMixin


class _LinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)


@pytest.fixture(
    params=[AscendSFAPCPImpl, AscendSFADSACPImpl],
    ids=["pcp", "legacy_cp"],
)
def sfa_impl(request, monkeypatch):
    impl = request.param.__new__(request.param)
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl._initialize_o_proj_weight_switch(WeightSwitchConfig.from_group(group, shard_axis="input"))
    impl.o_proj_full_pools = {}
    impl.tp_size = 2
    impl.enable_dsa_cp_full_o_proj = True
    impl.enable_pcp_o_proj_weight_sharding = True
    impl.o_proj = torch.nn.Module()
    impl.o_proj.input_size = 4
    impl.o_proj.input_size_per_partition = 2
    impl.o_proj.output_size = impl.o_proj.output_size_per_partition = 3
    impl.o_proj.weight = torch.nn.Parameter(torch.arange(6.0).reshape(3, 2), requires_grad=False)
    impl.o_proj.quant_method = _LinearMethod()
    impl._enable_o_proj_weight_switch()
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
    return SimpleNamespace(
        attn_state=attn_state,
        pcp_has_global_prefill=global_prefill,
        num_prefills=int(global_prefill),
        num_input_tokens=1,
        cum_query_lens=lengths,
        seq_lens=lengths,
        slot_mapping=slots,
        pcp_slot_mapping=slots,
        dcp_context=SimpleNamespace(slot_mapping=slots),
        dsa_cp_context=SimpleNamespace(
            actual_seq_lengths_query=lengths,
            actual_seq_lengths_key=lengths,
            slot_mapping_cp=slots,
            local_end_with_pad=1,
            local_start=0,
        ),
    )


def _prepare_forward(impl, metadata, hidden):
    context = impl._get_parallel_forward_context(metadata, 1, hidden)
    assert not hasattr(context, "gather_full_o_proj")
    if not isinstance(impl, AscendSFAPCPImpl) and impl._o_proj_uses_full_weight():
        # Legacy CP launches the weight gather after its KV gather.
        impl._store_parallel_kv(None, None, None, None, [], None, None, metadata)


def test_sfa_prefill_decode_and_profile_share_one_mode(sfa_impl, monkeypatch):
    modes = []
    local_address = sfa_impl.o_proj.weight.data_ptr()
    weight_state = sfa_impl._weight_switch_states[0][2]

    def forward(impl, name, hidden, cache, metadata, output):
        assert not impl._o_proj_uses_full_weight()
        if metadata is None:
            return output.zero_()
        _prepare_forward(impl, metadata, hidden)
        modes.append(impl._o_proj_uses_full_weight())
        if impl._o_proj_uses_full_weight():
            with impl._use_full_o_proj_weight_switches():
                assert impl.o_proj.weight.data_ptr() == weight_state.gather_parts["weight"].full_tensor.data_ptr()
        else:
            assert impl.o_proj.weight.data_ptr() == local_address
        return output

    monkeypatch.setattr(AscendSFAImpl, "_forward_impl", forward)
    hidden = torch.empty(1, 4)
    output = torch.empty_like(hidden)
    for state in (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    ):
        assert sfa_impl.forward("layer", hidden, (), _metadata(state), output) is output
        assert not sfa_impl._o_proj_uses_full_weight()
        assert not weight_state.handles
        assert sfa_impl.o_proj.weight.data_ptr() == local_address
    assert modes == [True, False, False]
    sfa_impl.forward("layer", hidden, (), None, output)
    assert not sfa_impl._o_proj_uses_full_weight()
    assert sfa_impl.waits.call_count == 1


@pytest.mark.parametrize("stage", ["before_projection", "projection"])
def test_failed_sfa_prefill_restores_local_weights_and_next_decode(sfa_impl, monkeypatch, stage):
    failure = RuntimeError("injected attention failure")
    local_address = sfa_impl.o_proj.weight.data_ptr()
    weight_state = sfa_impl._weight_switch_states[0][2]

    def forward(impl, name, hidden, cache, metadata, output):
        _prepare_forward(impl, metadata, hidden)
        if metadata.attn_state == AscendAttentionState.DecodeOnly:
            assert not impl._o_proj_uses_full_weight()
            assert not weight_state.handles
            return output
        assert impl._o_proj_uses_full_weight()
        if stage == "before_projection":
            assert weight_state.handles
            raise failure
        with impl._use_full_o_proj_weight_switches():
            assert impl.o_proj.weight.data_ptr() != local_address
            raise failure

    monkeypatch.setattr(AscendSFAImpl, "_forward_impl", forward)
    hidden = torch.empty(1, 4)
    output = torch.empty_like(hidden)
    with pytest.raises(RuntimeError) as caught:
        sfa_impl.forward("layer", hidden, (), _metadata(AscendAttentionState.ChunkedPrefill), output)
    assert caught.value is failure
    assert not sfa_impl._o_proj_uses_full_weight()
    assert not weight_state.handles
    assert sfa_impl.o_proj.weight.data_ptr() == local_address
    assert sfa_impl.waits.call_count == 1
    sfa_impl.forward("layer", hidden, (), _metadata(AscendAttentionState.DecodeOnly), output)


def test_all_gather_requires_prepared_switches_without_enabling():
    impl = AscendDSAPCPImpl.__new__(AscendDSAPCPImpl)
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    impl._initialize_o_proj_weight_switch(WeightSwitchConfig.from_group(group))
    impl._enable_o_proj_weight_switch = Mock(side_effect=AssertionError("all-gather must not enable"))

    method = Mock()
    state = SimpleNamespace(handles=[])
    impl._weight_switch_states = ((torch.nn.Module(), method, state),)

    impl._all_gather_o_proj_weight_switches()

    impl._enable_o_proj_weight_switch.assert_not_called()
    method.all_gather_weight.assert_called_once_with(state, impl._weight_switch_config)


def test_dsa_cleanup_waits_every_layer_without_hiding_attention_failure(monkeypatch):
    impl = AscendDSAPCPImpl.__new__(AscendDSAPCPImpl)
    impl._initialize_o_proj_weight_switch(None)
    attention_failure = RuntimeError("attention failed")
    first = SimpleNamespace(handles=[SimpleNamespace(wait=Mock(side_effect=RuntimeError("wait failed")))])
    second_wait = Mock()
    second = SimpleNamespace(handles=[SimpleNamespace(wait=second_wait)])
    impl._weight_switch_states = tuple((torch.nn.Module(), _LinearMethod(), state) for state in (first, second))

    with pytest.raises(RuntimeError) as caught, impl._o_proj_forward_context():
        impl._set_o_proj_use_full_weight(True)
        raise attention_failure
    assert caught.value is attention_failure
    second_wait.assert_called_once_with()
    assert not first.handles and not second.handles
    assert not impl._o_proj_uses_full_weight()
