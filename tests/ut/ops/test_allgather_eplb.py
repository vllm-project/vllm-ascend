# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.distributed.eplb import eplb_state as state_module
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe import eplb as eplb_ops
from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe.eplb import (
    build_expert_replica_routing_table,
    count_allgather_tokens,
    make_allgather_mask,
    map_to_physical,
    record_expert_tokens,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.patch.platform.patch_fused_moe import _ascend_apply_eplb_mapping


@pytest.fixture(autouse=True)
def v2_forward_context(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_allgather_routes_execute_and_count_each_token_once(monkeypatch, tp_size):
    profile = get_hardware_profile(AscendDeviceType.A5)
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.eplb.get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(
        afc,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={"moe_comm_type": afc.MoECommType.ALLGATHER}),
    )
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_map_to_physical", map_to_physical)
    # Default initial layout: four owners, eight logical experts, four replicas.
    # Expert 1's copies live on ranks 0 and 3, so rank-dependent choices
    # disagree on whether either owner should execute the same token.
    placement = torch.cat([torch.arange(8), torch.arange(4)]).reshape(4, 3)
    logical_map = torch.full((8, 2), -1)
    replicas = torch.ones(8, dtype=torch.int64)
    for logical in range(8):
        physical_slots = torch.where(placement.flatten() == logical)[0]
        logical_map[logical, : physical_slots.numel()] = physical_slots
        replicas[logical] = physical_slots.numel()
    table = build_expert_replica_routing_table(logical_map, replicas, ep_rank=0)
    dp_size = 4 // tp_size
    num_gathered_rows = 7 * dp_size
    logical_ids = torch.arange(8).expand(num_gathered_rows, -1)
    weights = torch.arange(1, 9, dtype=torch.float32).expand(num_gathered_rows, -1) / 36
    hidden = torch.arange(1, num_gathered_rows + 1, dtype=torch.float32)
    validity = torch.tensor([True, True, False, True, False, True, False]).repeat(dp_size)
    owners = torch.zeros_like(logical_ids)
    output = torch.zeros_like(hidden)
    old_output = torch.zeros_like(hidden)
    counts = []
    for rank in range(4):
        layer_state = SimpleNamespace(
            expert_replica_routing_table=build_expert_replica_routing_table(logical_map, replicas, rank),
            allgather_expert_replica_routing_table=table,
        )
        router = SimpleNamespace(eplb_state=layer_state, _validate_eplb_state=lambda: None)
        physical = _ascend_apply_eplb_mapping(router, logical_ids)
        local = (physical >= rank * 3) & (physical < (rank + 1) * 3)
        owners.add_(local)
        # TP ranks have the same inputs; ownership, rather than dividing load
        # after the fact, ensures each route contributes exactly once.
        logical_weights = placement.flatten()[physical.long()].float() + 1
        output += (hidden[:, None] * logical_weights * weights * local).sum(dim=1)
        mask = make_allgather_mask(
            num_gathered_rows, torch.tensor(4 * dp_size), torch.tensor(True), is_padding=~validity
        )
        counts.append(count_allgather_tokens(physical, mask, rank * 3, 3))
        old_physical = map_to_physical(logical_ids, build_expert_replica_routing_table(logical_map, replicas, rank))
        old_local = (old_physical >= rank * 3) & (old_physical < (rank + 1) * 3)
        old_output += (hidden[:, None] * (logical_ids + 1) * weights * old_local).sum(dim=1)
    reference = hidden * ((torch.arange(8).float() + 1) * weights[0]).sum()
    torch.testing.assert_close(owners, torch.ones_like(owners))
    torch.testing.assert_close(output, reference)
    assert not torch.allclose(old_output, reference)
    assert torch.stack(counts).sum() == validity.sum() * 8


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_sp_mask_tracks_padding_and_source_phase(tp_size):
    padding = torch.tensor([False, True, False, False, True, False, True])
    local_rows = (padding.numel() + tp_size - 1) // tp_size
    masks = [
        make_allgather_mask(
            local_rows, torch.tensor(4), torch.tensor(True), is_padding=padding, tp_size=tp_size, tp_rank=rank
        )
        for rank in range(tp_size)
    ]
    gathered = torch.cat(masks)[: padding.numel()]
    torch.testing.assert_close(gathered, ~padding)
    assert not make_allgather_mask(local_rows, torch.tensor(4), torch.tensor(False), tp_size=tp_size, tp_rank=0).any()
    fallback = torch.cat(
        [
            make_allgather_mask(local_rows, torch.tensor(4), torch.tensor(True), tp_size=tp_size, tp_rank=rank)
            for rank in range(tp_size)
        ]
    )
    assert fallback.sum() == 4


def test_dp_mask_uses_same_padded_rows_as_logits(monkeypatch):
    from vllm_ascend.ops.fused_moe import prepare_finalize as prepare_module

    monkeypatch.setattr(
        afc, "get_forward_context", lambda: SimpleNamespace(additional_kwargs={"max_tokens_across_dp": 4})
    )
    peer_logits = torch.tensor([[20.0], [21.0], [0.0], [0.0]])
    peer_mask = torch.tensor([1, 0, 0, 0], dtype=torch.int32)

    def gather(rows, dim):
        return torch.cat([rows, peer_logits if rows.ndim == 2 else peer_mask])

    config = SimpleNamespace(
        is_sequence_parallel=False, dp_size=2, pcp_size=1, dp_group=SimpleNamespace(all_gather=gather)
    )
    monkeypatch.setattr(prepare_module, "get_dynamic_mx_quant_scale_alg", lambda: None)
    prepare = PrepareAndFinalizeWithAllGather(config)
    logits = torch.tensor([[10.0], [11.0], [12.0]])
    prepared = prepare.prepare(logits, logits)
    mask = prepare.all_gather_eplb_mask(torch.tensor([True, True, False]))
    assert prepared.router_logits.flatten().tolist() == [10, 11, 12, 0, 20, 21, 0, 0]
    assert mask.tolist() == [True, True, False, False, True, False, False, False]


@pytest.mark.parametrize(
    "device,comm",
    [
        (AscendDeviceType.A5, afc.MoECommType.MC2),
        (AscendDeviceType.A5, afc.MoECommType.ALLTOALL),
        (AscendDeviceType.A3, afc.MoECommType.ALLGATHER),
    ],
)
def test_other_paths_keep_rank_dependent_mapping(monkeypatch, device, comm):
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.eplb.get_current_hardware_profile", lambda: get_hardware_profile(device)
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: SimpleNamespace(additional_kwargs={"moe_comm_type": comm}))
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_map_to_physical", map_to_physical)
    ranked = torch.full((1024, 2), 3, dtype=torch.int32)
    router = SimpleNamespace(
        _validate_eplb_state=lambda: None,
        eplb_state=SimpleNamespace(
            expert_replica_routing_table=ranked, allgather_expert_replica_routing_table=torch.zeros_like(ranked)
        ),
    )
    assert _ascend_apply_eplb_mapping(router, torch.tensor([[0, 1]])).tolist() == [[3, 3]]


@pytest.mark.parametrize(
    "device,comm",
    [
        (AscendDeviceType.A5, afc.MoECommType.MC2),
        (AscendDeviceType.A5, afc.MoECommType.ALLTOALL),
        (AscendDeviceType.A3, afc.MoECommType.ALLGATHER),
    ],
)
def test_source_mask_is_unused_outside_a5_allgather(monkeypatch, device, comm):
    state = AscendEplbLayerState()
    state.expert_load_view = torch.zeros(2, dtype=torch.int32)
    state.should_record_tensor = torch.tensor(True)
    state.local_expert_start = 0
    state.local_expert_count = 2
    context = SimpleNamespace(additional_kwargs={"moe_comm_type": comm})
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.eplb.get_current_hardware_profile", lambda: get_hardware_profile(device)
    )
    prepare = MagicMock(spec=PrepareAndFinalizeWithAllGather)
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    state.record_load(
        torch.tensor([2, 3]),
        1,
        topk_ids=torch.empty(0, 1, dtype=torch.int32),
        num_source_tokens=2,
        is_sequence_parallel=False,
        prepare_finalize=prepare,
    )
    assert state.expert_load_view.tolist() == [2, 3]
    prepare.all_gather_eplb_mask.assert_not_called()


@pytest.mark.parametrize("padding_source", ["draft", "context", "unpadded_count"])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_source_mask_preserves_ubatch_padding_and_tp_slice(monkeypatch, padding_source, sequence_parallel):
    state = AscendEplbLayerState()
    state.num_unpadded_tokens_tensors = [torch.tensor(0), torch.tensor(3)]
    state.should_record_tensor = torch.tensor(True)
    state.allgather_source_is_padding = torch.tensor([False, False, False, True]) if padding_source == "draft" else None
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    state.expert_load_view = torch.zeros(1, dtype=torch.int32)
    state.local_expert_start = 0
    state.local_expert_count = 1
    context = SimpleNamespace(
        is_padding=None if padding_source == "unpadded_count" else torch.tensor([False, True, True, False])
    )
    monkeypatch.setattr(eplb_ops, "use_allgather_eplb", lambda enabled: enabled)
    monkeypatch.setattr(state_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(state_module, "dbo_current_ubatch_id", lambda: 1)
    monkeypatch.setattr(state_module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(state_module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_expert_tokens)
    prepare = MagicMock(spec=PrepareAndFinalizeWithAllGather)
    prepare.all_gather_eplb_mask.side_effect = lambda mask: mask
    expected = {
        (False, "draft"): [True, True],
        (False, "context"): [True, False],
        (False, "unpadded_count"): [True, True],
        (True, "draft"): [True, False],
        (True, "context"): [False, True],
        (True, "unpadded_count"): [True, False],
    }[sequence_parallel, padding_source]

    def record():
        state.record_load(
            torch.tensor([2]),
            1,
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            num_source_tokens=2,
            is_sequence_parallel=sequence_parallel,
            prepare_finalize=prepare,
        )

    record()
    assert prepare.all_gather_eplb_mask.call_args.args[0].tolist() == expected
    assert state.expert_load_view.item() == sum(expected)
    state.should_record_tensor.fill_(False)
    record()
    assert prepare.all_gather_eplb_mask.call_args.args[0].tolist() == [False, False]
    assert state.expert_load_view.item() == sum(expected)
    assert prepare.all_gather_eplb_mask.call_count == 2


@pytest.mark.parametrize("source_padding,expected", [([False, False, False, True], 3), ([True] * 4, 0)])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_draft_source_mask_overrides_synthetic_model_padding(monkeypatch, source_padding, expected, dp_size):
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.prepare_finalize.get_dynamic_mx_quant_scale_alg", lambda: None)
    events = []
    state = AscendEplbLayerState()
    state.num_unpadded_tokens_tensors = [torch.tensor(4)]
    state.should_record_tensor = torch.tensor(True)
    state.allgather_source_is_padding = torch.tensor(source_padding)
    state.allgather_record_tensor = torch.tensor(-1, dtype=torch.int32)
    state.expert_load_view = torch.zeros(1, dtype=torch.int32)
    state.local_expert_count = 1
    state.local_expert_start = 0
    config = SimpleNamespace(
        is_sequence_parallel=False,
        dp_size=dp_size,
        pcp_size=1,
        ep_rank=0,
        num_local_experts=1,
        dp_group=SimpleNamespace(all_gather=lambda rows, dim: torch.cat([rows, torch.zeros_like(rows)])),
    )
    prepare = PrepareAndFinalizeWithAllGather(config)
    make_mask = eplb_ops.make_allgather_mask
    gather_mask = prepare.all_gather_eplb_mask

    def source_mask(*args, **kwargs):
        events.append("source_mask")
        assert args[0] == 4
        return make_mask(*args, **kwargs)

    def prepare_inputs(**kwargs):
        events.append("prepare")
        return prepare.prepare(**kwargs)

    def gather_source_mask(mask):
        events.append("gather_mask")
        return gather_mask(mask)

    def select_experts(**kwargs):
        events.append("route")
        assert kwargs["hidden_states"].shape[0] == 4 * dp_size
        return torch.ones(4 * dp_size, 1), torch.zeros(4 * dp_size, 1, dtype=torch.int32)

    def apply_experts(**kwargs):
        events.append("ffn")
        return FusedExpertsResult(
            routed_out=torch.ones(4 * dp_size, 2), expert_tokens=torch.tensor([4 * dp_size]), group_list_type=1
        )

    def record_load(*args):
        events.append("record_load")
        return record_expert_tokens(*args)

    def finalize(**kwargs):
        events.append("finalize")
        return kwargs["hidden_states"]

    comm = SimpleNamespace(
        prepare_finalize=prepare,
        prepare=prepare_inputs,
        finalize=finalize,
    )
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer.router = SimpleNamespace(eplb_state=state)
    layer._use_v2_model_runner = True
    layer.enable_npugraph_ex_static_kernel = False
    layer.moe_config = config
    layer._select_experts = select_experts
    object.__setattr__(layer, "quant_method", SimpleNamespace(apply=apply_experts))
    layer.dynamic_eplb = False
    layer.return_with_event = False
    context = SimpleNamespace(
        is_padding=torch.zeros(4, dtype=torch.bool), additional_kwargs={"max_tokens_across_dp": 4}
    )
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        routed_module, "_EXTRA_CTX", SimpleNamespace(in_profile_run=False, moe_comm_type=afc.MoECommType.ALLGATHER)
    )
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    monkeypatch.setattr(eplb_ops, "use_allgather_eplb", lambda enabled: enabled)
    monkeypatch.setattr(eplb_ops, "make_allgather_mask", source_mask)
    monkeypatch.setattr(prepare, "all_gather_eplb_mask", gather_source_mask)
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_record_expert_tokens", record_load)
    AscendRoutedExperts.forward_impl(layer, hidden_states=torch.ones(4, 2), router_logits=torch.ones(4, 1))
    assert events == ["prepare", "route", "ffn", "source_mask", "gather_mask", "record_load", "finalize"]
    assert state.expert_load_view.item() == expected
    assert state.allgather_record_tensor.item() == int(expected > 0)
