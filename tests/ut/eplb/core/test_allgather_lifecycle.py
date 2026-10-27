# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.allgather import build_allgather_routing, route_allgather
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader, ExpertWeightUpdateState
from vllm_ascend.eplb.core.eplb_worker import EplbWorker
from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


def test_planner_transfers_and_publishes_new_hot_expert():
    worker = EplbWorker.__new__(EplbWorker)
    worker.rank_id, worker.tp_size = 0, 2
    old = torch.tensor([[[0, 1, 2], [3, 0, 1]], [[3, 2, 1], [0, 3, 2]]])
    new = torch.tensor([[[0, 1, 3], [3, 2, 1]], [[3, 2, 0], [0, 1, 2]]])
    worker.check_expert_placement(old, new)
    old_maps, new_maps = worker.local2global(old), worker.local2global(new)
    plans = list(worker.compose_expert_update_info_greedy(new_maps, old_maps))
    packed = worker.pack_update_info(iter(plans))
    for send, recv, local_map, rotated, allgather, layer_id in packed:
        target_expert = int(new[layer_id, 0, 2])
        assert (1, target_expert) in recv
        assert local_map[target_expert] == 2
        slots, counts, global_map = map(torch.tensor, allgather)
        assert torch.equal(global_map, new_maps[layer_id])
        ids = torch.full((8, 1), target_expert, dtype=torch.int32)
        physical = route_allgather(ids, slots, counts)
        assert set(physical.flatten().tolist()) == {2, 3}
        assert torch.equal(new[layer_id].flatten()[physical.long()], ids.long())


def test_route_publish_happens_after_all_weights_and_scales():
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader = D2DExpertWeightLoader()
    events = []
    adaptor = SimpleNamespace(
        num_moe_layers=1,
        do_update_expert_weight=lambda *args: events.append("weights-and-scales"),
        do_update_expert_map=lambda *args: events.append("planner"),
        do_update_log2phy_map=lambda *args: events.append("rotated"),
        do_update_allgather_maps=lambda *args: events.append("allgather"),
    )
    loader.set_adator(adaptor)
    loader.state = ExpertWeightUpdateState.TRANSFERRING
    loader.layer_id = 0
    loader.recv_expert_list = [(0, 0), (1, 1)]
    loader.updated_allgather_maps = (torch.zeros(1),)
    loader.update_expert_map_and_weight([SimpleNamespace(wait=lambda: events.append("wait"))])
    assert events == ["wait", "weights-and-scales", "weights-and-scales", "planner", "rotated", "allgather"]
    assert loader.state == ExpertWeightUpdateState.WAITING
    assert loader.updated_allgather_maps is None


def test_adaptor_refreshes_graph_buffers_without_replacing_them():
    worker = EplbWorker.__new__(EplbWorker)
    old = worker.local2global(torch.tensor([[[0, 1, 2], [3, 0, 1]]]))[0]
    new = worker.local2global(torch.tensor([[[0, 1, 3], [3, 2, 1]]]))[0]
    slots, counts = build_allgather_routing(old)
    layer = SimpleNamespace(
        ascend_expert_map=old[0].clone(),
        global_expert_map=old.clone(),
        allgather_slots=slots,
        allgather_replica_counts=counts,
    )
    adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
    adaptor.ep_rank, adaptor.moe_layers = 0, [layer]
    adaptor.expert_map_per_layer_cpu = {0: old[0].clone()}
    ptrs = (slots.data_ptr(), counts.data_ptr())
    adaptor.do_update_expert_map(0, new[0])
    adaptor.do_update_allgather_maps(0, (*build_allgather_routing(new), new))
    assert ptrs == (layer.allgather_slots.data_ptr(), layer.allgather_replica_counts.data_ptr())
    assert torch.equal(layer.ascend_expert_map, new[0])
    assert torch.equal(layer.global_expert_map, new)
    assert counts.tolist() == [1, 2, 1, 2]


def test_swiftbalance_reassigns_redundant_slots_when_hotspot_changes():
    policy = SwiftBalanceEplb()
    policy.num_die_per_host = 4
    worker = EplbWorker.__new__(EplbWorker)
    placement = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]]])
    for hot in (7, 5, 6):
        load = torch.ones_like(placement, dtype=torch.float32)
        load[placement == hot] = 1000 / int((placement == hot).sum())
        _, _, proposed = policy.rebalance_experts(placement, load)
        proposed = torch.tensor(proposed)
        worker.check_expert_placement(placement, proposed)
        assert int((proposed == hot).sum()) > int((placement == hot).sum())
        assert set(proposed.flatten().tolist()) == set(range(8))
        placement = proposed
