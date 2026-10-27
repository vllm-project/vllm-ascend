# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-card W8A8 ALLGATHER, real migration and NPU graph regression.

Run with pytest, or directly with torchrun --standalone --nproc-per-node=4.
Uses small distinguishable expert tensors; no model download is needed.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dynamic_eplb,aclgraph",
    parallel="EP",
    hardware="A2",
    quantization="W8A8",
    deploy="pd_mix",
    graph_mode="full_graph",
)
def test_allgather_eplb_w8a8_migration():
    if torch.npu.device_count() < 4:
        pytest.skip("Requires four NPU devices")
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="0", OMP_NUM_THREADS="1")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4", __file__],
        env=env,
        check=True,
        timeout=180,
    )


def _run_probe():
    # Only group lookups are bound to the standalone HCCL process group.
    # The production planner, transfer loader, adaptor and NPU operators run.
    import json
    import os
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    group = SimpleNamespace(ranks=list(range(4)), rank_in_group=rank, world_size=4, device_group=dist.group.WORLD)

    import vllm_ascend.eplb.core.eplb_device_transfer_loader as loader_mod
    import vllm_ascend.ops.fused_moe.token_dispatcher as dispatch_mod
    from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
    from vllm_ascend.eplb.core.allgather import build_allgather_routing, count_local_routes, route_allgather
    from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
    from vllm_ascend.eplb.core.eplb_worker import EplbWorker
    from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
    from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
    from vllm_ascend.quantization.quant_type import QuantType

    loader_mod.get_dynamic_eplb_group = lambda: group
    dispatch_mod.get_ep_group = lambda: group
    worker = EplbWorker.__new__(EplbWorker)
    worker.rank_id, worker.tp_size = rank, 4
    base = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]]])
    placements = [
        base,
        torch.tensor([[[0, 1, 3], [2, 4, 5], [6, 7, 0], [1, 2, 3]]]),
        torch.tensor([[[0, 1, 7], [2, 4, 5], [6, 7, 0], [7, 2, 3]]]),
        base,
    ]
    placements = [torch.cat([p, (p + 3) % 8]) for p in placements]
    maps = worker.local2global(placements[0])
    D, N, K = 128, 32, 2
    device = f"npu:{rank}"
    torch.manual_seed(12)
    x = (torch.randn(N, D) * 0.5).bfloat16().to(device)
    logical = torch.stack([torch.arange(N) % 8, torch.zeros(N, dtype=torch.int64)], 1).int().to(device)
    route_weights = torch.full((N, K), 0.5, dtype=torch.float32, device=device)
    valid = torch.arange(N, device=device) < 27
    eye = torch.eye(D, dtype=torch.int8)

    def params(expert):
        return [
            (eye * (expert + 1)).to(device),
            (eye * (9 - expert)).to(device),
            torch.full((D,), 0.02 * (expert + 1), dtype=torch.float32, device=device),
            torch.full((D,), 0.03 * (expert + 1), dtype=torch.bfloat16, device=device),
        ]

    layers = []
    adaptor = VllmEplbAdaptor.__new__(VllmEplbAdaptor)
    adaptor.ep_rank, adaptor.num_moe_layers = rank, 2
    adaptor.expert_map_per_layer_cpu = {}
    adaptor.expert_param_per_layer = {}
    adaptor.expert_weight_key_per_layer = {0: "w8", 1: "w8"}
    adaptor.buffer_tensor_list = {"w8": [[torch.empty_like(t) for t in params(0)] for _ in range(3)]}
    adaptor.log2phy_map_per_layer = {}
    for layer_id in range(2):
        slots, counts = build_allgather_routing(maps[layer_id])
        layer = SimpleNamespace(
            global_expert_map=maps[layer_id].clone(),
            ascend_expert_map=maps[layer_id, rank].clone(),
            allgather_slots=slots.to(device),
            allgather_replica_counts=counts.to(device),
        )
        layers.append(layer)
        adaptor.expert_map_per_layer_cpu[layer_id] = maps[layer_id, rank].clone()
        adaptor.log2phy_map_per_layer[layer_id] = generate_log2phy_map(maps[layer_id], rank, tp_size=4).to(device)
        adaptor.expert_param_per_layer[layer_id] = [params(int(e)) for e in placements[0][layer_id, rank]]
    adaptor.moe_layers = layers
    loader = loader_mod.D2DExpertWeightLoader()
    loader.set_adator(adaptor)
    dispatcher = dispatch_mod.TokenDispatcherWithAllGather(top_k=K, num_experts=12, num_local_experts=3)

    def forward(layer_id):
        layer = layers[layer_id]
        ids = route_allgather(logical, layer.allgather_slots, layer.allgather_replica_counts)
        dispatched = dispatcher.token_dispatch(
            MoETokenDispatchInput(
                x,
                route_weights,
                ids,
                MoeRouterInput(layer.ascend_expert_map, 4, valid, False),
                MoEQuantParams(QuantType.W8A8),
            )
        )
        tensors = adaptor.expert_param_per_layer[layer_id]
        w1, w2, s1, s2 = map(list, zip(*tensors))
        first = torch_npu.npu_grouped_matmul(
            x=[dispatched.hidden_states],
            weight=w1,
            scale=[s.to(torch.bfloat16) for s in s1],
            per_token_scale=[dispatched.dynamic_scale],
            split_item=2,
            group_type=0,
            group_list=dispatched.group_list,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]
        quant, scales = torch_npu.npu_dynamic_quant(torch.nn.functional.silu(first))
        second = torch_npu.npu_grouped_matmul(
            x=[quant],
            weight=w2,
            scale=s2,
            per_token_scale=[scales],
            split_item=2,
            group_type=0,
            group_list=dispatched.group_list,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]
        result = dispatcher.token_combine(second, dispatched.combine_metadata)
        dist.all_reduce(result)
        heat = count_local_routes(ids, valid, rank, 3)
        return result, heat

    graphs, outputs, baseline = [], [], []
    for layer_id in range(2):
        for _ in range(3):
            out, heat = forward(layer_id)
        baseline.append(out.clone())
        # Independent floating reference with distinct expert weights and scales.
        reference = torch.zeros_like(x, dtype=torch.float32)
        for k in range(K):
            e = logical[:, k].float().unsqueeze(1)
            first = x.float() * (e + 1) * (0.02 * (e + 1))
            reference += torch.nn.functional.silu(first) * (9 - e) * (0.03 * (e + 1)) * 0.5
        torch.testing.assert_close(out.float(), reference, atol=0.02, rtol=0.05)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = forward(layer_id)
        graphs.append(graph)
        outputs.append(captured)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured[0], baseline[layer_id], atol=0, rtol=0)

    ptrs = [[layer.allgather_slots.data_ptr(), layer.allgather_replica_counts.data_ptr()] for layer in layers]
    for cycle, placement in enumerate(placements):
        new_maps = worker.local2global(placement)
        if cycle:
            worker.check_expert_placement(placements[cycle - 1], placement)
            assert torch.equal(new_maps, worker.local2global(placement)), "placement unexpectedly rejected"
            updates = worker.pack_update_info(worker.compose_expert_update_info_greedy(new_maps, maps))
            for send, recv, updated, log2phy, allgather, layer_id in updates:
                loader.set_log2phy_map(torch.tensor(log2phy))
                loader.updated_allgather_maps = tuple(torch.tensor(v) for v in allgather)
                loader.generate_expert_d2d_transfer_task(send, recv, torch.tensor(updated), layer_id)
                requests = []
                loader.asyn_expert_weight_transfer(requests)
                loader.update_expert_map_and_weight(requests)
            maps = new_maps
        torch.npu.synchronize()
        for layer_id in range(2):
            for slot, expert in enumerate(placement[layer_id, rank]):
                for actual, expected in zip(adaptor.expert_param_per_layer[layer_id][slot], params(int(expert))):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert ptrs[layer_id] == [
                layers[layer_id].allgather_slots.data_ptr(),
                layers[layer_id].allgather_replica_counts.data_ptr(),
            ]
            eager, heat = forward(layer_id)
            graphs[layer_id].replay()
            torch.npu.synchronize()
            torch.testing.assert_close(eager, baseline[layer_id], atol=0, rtol=0)
            torch.testing.assert_close(outputs[layer_id][0], baseline[layer_id], atol=0, rtol=0)
            torch.testing.assert_close(outputs[layer_id][1], heat, atol=0, rtol=0)
            total_heat = heat.clone()
            dist.all_reduce(total_heat)
            assert int(total_heat.sum().cpu()) == 27 * K
            print(
                json.dumps(
                    {"rank": rank, "cycle": cycle, "layer": layer_id, "heat": heat.cpu().tolist(), "passed": True}
                ),
                flush=True,
            )
    policy = SwiftBalanceEplb()
    policy.num_die_per_host = 4
    placement = placements[-1]
    for hot in (7, 5, 6):
        # Change actual token routes, collect actual slot loads and use the real
        # policy to decide which logical expert gets redundant capacity next.
        logical[:, 0].copy_((torch.arange(N, device=device) % 7 + hot + 1).remainder(8).int())
        logical[:, 1].fill_(hot)
        local_heat = torch.stack([forward(i)[1] for i in range(2)])
        gathered = [torch.empty_like(local_heat) for _ in range(4)]
        dist.all_gather(gathered, local_heat)
        load = torch.stack(gathered, dim=1).cpu()
        _, _, proposed = policy.rebalance_experts(placement, load)
        proposed = torch.tensor(proposed)
        worker.check_expert_placement(placement, proposed)
        if rank == 0:
            print(
                json.dumps(
                    {"hot": hot, "load": load.tolist(), "old": placement.tolist(), "proposed": proposed.tolist()}
                ),
                flush=True,
            )
        # The policy can keep a layer if its candidate does not improve balance.
        # Require real redundant reassignment on every hotspot transition.
        assert any(int((proposed[i] == hot).sum()) > int((placement[i] == hot).sum()) for i in range(2))
        new_maps = worker.local2global(proposed)
        updates = worker.pack_update_info(worker.compose_expert_update_info_greedy(new_maps, maps))
        for send, recv, updated, log2phy, allgather, layer_id in updates:
            loader.set_log2phy_map(torch.tensor(log2phy))
            loader.updated_allgather_maps = tuple(torch.tensor(v) for v in allgather)
            loader.generate_expert_d2d_transfer_task(send, recv, torch.tensor(updated), layer_id)
            requests = []
            loader.asyn_expert_weight_transfer(requests)
            loader.update_expert_map_and_weight(requests)
        maps, placement = new_maps, proposed
        for valid_count in (N, 5, 0, 27):
            valid.copy_(torch.arange(N, device=device) < valid_count)
            for layer_id in range(2):
                for slot, expert in enumerate(placement[layer_id, rank]):
                    for actual, expected in zip(adaptor.expert_param_per_layer[layer_id][slot], params(int(expert))):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                eager, heat = forward(layer_id)
                reference = torch.zeros_like(x, dtype=torch.float32)
                for k in range(K):
                    e = logical[:, k].float().unsqueeze(1)
                    first = x.float() * (e + 1) * (0.02 * (e + 1))
                    reference += torch.nn.functional.silu(first) * (9 - e) * (0.03 * (e + 1)) * 0.5
                torch.testing.assert_close(eager.float(), reference, atol=0.02, rtol=0.05)
                graphs[layer_id].replay()
                torch.npu.synchronize()
                torch.testing.assert_close(outputs[layer_id][0], eager, atol=0, rtol=0)
                torch.testing.assert_close(outputs[layer_id][1], heat, atol=0, rtol=0)
                total_heat = heat.sum()
                dist.all_reduce(total_heat)
                assert int(total_heat.cpu()) == valid_count * K
        if rank == 0:
            print(
                json.dumps({"hot_expert": hot, "placement": placement.tolist(), "policy_graph_pass": True}), flush=True
            )
    dist.barrier()
    if rank == 0:
        print("DEVICE_MIGRATION_GRAPH_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_probe()
