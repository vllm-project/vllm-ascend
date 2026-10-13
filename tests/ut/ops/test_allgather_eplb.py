# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.ut.ops.test_token_dispatcher import build_token_dispatch_input_fixture
from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe import token_dispatcher as dispatcher_module
from vllm_ascend.ops.fused_moe.eplb import (
    build_allgather_routing_table,
    build_expert_replica_routing_table,
    count_allgather_tokens,
    make_allgather_mask,
    map_to_physical,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.patch.platform.patch_fused_moe import _ascend_apply_eplb_mapping


@pytest.mark.parametrize("runner", ["mrv1", "mrv2"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_allgather_routes_execute_and_count_each_token_once(monkeypatch, runner, tp_size):
    profile = get_hardware_profile(AscendDeviceType.A5)
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.eplb.get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(afc, "get_forward_context", lambda: SimpleNamespace(moe_comm_type=afc.MoECommType.ALLGATHER))
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_map_to_physical", map_to_physical)
    # Default initial layout: four owners, eight logical experts, four replicas.
    # Expert 1's copies live on ranks 0 and 3, so rank-dependent choices
    # disagree on whether either owner should execute the same token.
    placement = torch.cat([torch.arange(8), torch.arange(4)]).reshape(4, 3)
    global_map = torch.full((4, 8), -1, dtype=torch.int32)
    global_map.scatter_(1, placement, torch.arange(3, dtype=torch.int32).expand(4, -1))
    table = build_allgather_routing_table(global_map, 3)
    logical_map = torch.full((8, 2), -1)
    replicas = torch.ones(8, dtype=torch.int64)
    for logical in range(8):
        physical_slots = torch.where(placement.flatten() == logical)[0]
        logical_map[logical, : physical_slots.numel()] = physical_slots
        replicas[logical] = physical_slots.numel()
    dp_size = 4 // tp_size
    num_gathered_rows = 7 * dp_size
    logical_ids = torch.arange(8).expand(num_gathered_rows, -1)
    weights = torch.arange(1, 9, dtype=torch.float32).expand(num_gathered_rows, -1) / 36
    hidden = torch.arange(1, num_gathered_rows + 1, dtype=torch.float32)
    validity = torch.tensor([True, True, False, True, False, True, False]).repeat(dp_size)
    output = torch.zeros_like(hidden)
    old_output = torch.zeros_like(hidden)
    counts = []
    for rank in range(4):
        if runner == "mrv1":
            layer = SimpleNamespace(
                router=SimpleNamespace(_select_experts=lambda **_: (weights, logical_ids)),
                log2phy=torch.arange(8),
                n_shared_experts=0,
                mix_placement=False,
                moe_config=SimpleNamespace(num_experts=12, num_logical_experts=8),
                global_redundant_expert_num=4,
                allgather_expert_replica_routing_table=table,
            )
            monkeypatch.setattr(routed_module, "get_ascend_config", lambda: SimpleNamespace(enable_force_eplb=False))
            _, physical = AscendRoutedExperts._select_experts(layer, hidden[:, None], weights, False)
        else:
            layer_state = SimpleNamespace(
                expert_replica_routing_table=build_expert_replica_routing_table(logical_map, replicas, rank),
                allgather_expert_replica_routing_table=table,
            )
            router = SimpleNamespace(eplb_state=layer_state, _validate_eplb_state=lambda: None)
            physical = _ascend_apply_eplb_mapping(router, logical_ids)
        local = (physical >= rank * 3) & (physical < (rank + 1) * 3)
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

    monkeypatch.setattr(afc, "get_forward_context", lambda: SimpleNamespace(max_tokens_across_dp=4))
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


def test_a5_dispatch_masks_physical_ids_beyond_logical_map(monkeypatch):
    monkeypatch.setattr(
        dispatcher_module, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    monkeypatch.setattr(dispatcher_module, "get_ep_group", lambda: SimpleNamespace(rank_in_group=3))
    dispatch = dispatcher_module.TokenDispatcherWithAllGather(top_k=2, num_experts=16, num_local_experts=4)
    ids = torch.tensor([[0, 12], [13, 7]], dtype=torch.int32)
    inputs = build_token_dispatch_input_fixture(
        hidden_states=torch.ones(2, 2),
        topk_ids=ids,
        topk_weights=torch.ones(2, 2),
        expert_map=torch.arange(8),
        global_redundant_expert_num=8,
    )
    op = MagicMock(return_value=(inputs.hidden_states, torch.arange(4), torch.ones(4), None))
    monkeypatch.setattr(dispatcher_module.DeviceOperator, "npu_moe_init_routing", op)
    result = dispatch.token_dispatch(inputs)
    assert op.call_args.kwargs["expert_num"] == 16
    assert op.call_args.kwargs["active_expert_range"] == [12, 16]
    torch.testing.assert_close(result.combine_metadata.topk_weights, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_legacy_migration_keeps_graph_table_and_drops_old_load():
    old_map = torch.tensor([[0, 1, -1], [0, -1, 1]], dtype=torch.int32)
    table = build_allgather_routing_table(old_map, 2)
    layer = SimpleNamespace(
        _allgather_placement=old_map.clone(),
        global_expert_map=old_map.clone(),
        allgather_expert_replica_routing_table=table,
        moe_config=SimpleNamespace(num_local_experts=2),
        moe_load=torch.ones(2),
        load_counter=torch.tensor(3),
    )
    AscendRoutedExperts.update_allgather_expert_map(layer, old_map)
    assert layer.moe_load.sum() == 2
    new_map = torch.tensor([[1, 0, -1], [1, -1, 0]], dtype=torch.int32)
    AscendRoutedExperts.update_allgather_expert_map(layer, new_map)
    assert layer.allgather_expert_replica_routing_table is table
    torch.testing.assert_close(table, build_allgather_routing_table(new_map, 2))
    assert layer.moe_load.sum() == 0 and layer.load_counter == 0


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
    monkeypatch.setattr(afc, "get_forward_context", lambda: SimpleNamespace(moe_comm_type=comm))
    monkeypatch.setattr(torch.ops.vllm, "ascend_eplb_map_to_physical", map_to_physical)
    ranked = torch.full((1024, 2), 3, dtype=torch.int32)
    router = SimpleNamespace(
        _validate_eplb_state=lambda: None,
        eplb_state=SimpleNamespace(
            expert_replica_routing_table=ranked, allgather_expert_replica_routing_table=torch.zeros_like(ranked)
        ),
    )
    assert _ascend_apply_eplb_mapping(router, torch.tensor([[0, 1]])).tolist() == [[3, 3]]
