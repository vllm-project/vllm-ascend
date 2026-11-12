# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.distributed.eplb import eplb_communicator
from vllm.distributed.eplb import eplb_state as upstream
from vllm.utils.network_utils import get_open_port

from vllm_ascend.distributed.eplb import eplb_state as ascend
from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer
from vllm_ascend.patch.platform import patch_eplb
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_RANKS = 2
_EXPERTS_PER_RANK = 2
_EXPERTS = _RANKS * _EXPERTS_PER_RANK
_WIDTH = 128
_TIMEOUT = 120


def _run_placement_export(rank: int, port: int, path: Path) -> None:
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    dist.init_process_group(
        "hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_RANKS, timeout=timedelta(seconds=_TIMEOUT)
    )
    group = SimpleNamespace(rank_in_group=rank, world_size=_RANKS, device_group=dist.group.WORLD)
    try:
        with patch.object(ascend, "get_ep_group", return_value=group):
            _check_commits(rank, device, path)
    finally:
        dist.destroy_process_group()


def _check_commits(rank: int, device: torch.device, path: Path) -> None:
    old = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32)
    target = torch.tensor([[0, 2, 1, 3], [2, 1, 0, 3]], dtype=old.dtype)
    slots = slice(rank * _EXPERTS_PER_RANK, (rank + 1) * _EXPERTS_PER_RANK)

    def weight_for(layout, layer):
        return (layout[layer, slots, None] * _WIDTH + torch.arange(_WIDTH) + 1000 * layer).to(torch.bfloat16)

    weights = [[weight_for(old, layer).to(device)] for layer in range(len(old))]
    logical_map, replica_count = upstream.compute_logical_maps(old, _EXPERTS)
    logical_map, replica_count = logical_map.to(device), replica_count.to(device)
    layers = []
    for layer in range(len(old)):
        routing = ascend.AscendEplbLayerState()
        routing.logical_to_physical_map = logical_map[layer]
        routing.logical_replica_count = replica_count[layer]
        routing.refresh_expert_replica_routing_table()
        layers.append(SimpleNamespace(eplb_state=routing))
    initial_routing = [layer.eplb_state.expert_replica_routing_table.clone() for layer in layers]
    communicator = eplb_communicator.TorchDistNcclEplbCommunicator(dist.group.WORLD)
    model_state = SimpleNamespace(
        model=SimpleNamespace(
            num_moe_layers=len(old),
            num_logical_experts=_EXPERTS,
            num_physical_experts=_EXPERTS,
            expert_weights=weights,
            moe_layers=layers,
        ),
        physical_to_logical_map=old.to(device),
        logical_to_physical_map=logical_map,
        logical_replica_count=replica_count,
        expert_buffer=[torch.empty_like(weights[0][0])],
        communicator=communicator,
        rebalanced=True,
    )
    state = ascend.AscendEplbState.__new__(ascend.AscendEplbState)
    state._placement_export = None
    state.model_states = {}
    state._placement_models = {"toy": {"model": "synthetic", "revision": "test-revision"}}
    state.enable_placement_export(str(path))
    state.model_states["toy"] = model_state
    assert (state._placement_export is not None) == (rank == 0)
    if state._placement_export is not None:
        model_state._export_eplb_placement_after_commit = state._export_committed_placement

    physical, logical, counts = patch_eplb._prepare_commit_maps(model_state, target)
    for layer in range(len(old)):
        sources = target[layer].numpy().reshape(_RANKS, _EXPERTS_PER_RANK)
        metadata = stage_explicit_layer_transfer(
            old[layer],
            target[layer],
            sources // _EXPERTS_PER_RANK,
            sources % _EXPERTS_PER_RANK,
            weights[layer],
            model_state.expert_buffer,
            dist.group.WORLD,
            communicator,
            layer_idx=layer,
        )
        last = layer == len(old) - 1
        model_state.pending_result = patch_eplb._AscendAsyncLayerResult(
            layer_idx=layer,
            new_physical_to_logical_map=physical[layer],
            new_logical_to_physical_map=logical[layer],
            new_logical_replica_count=counts[layer],
            transfer_metadata=metadata,
            consumed_event=torch.npu.Event(),
            is_last_result=last,
        )
        upstream._move_to_workspace(model_state=model_state, ep_rank=rank)
        torch.testing.assert_close(weights[layer][0].cpu(), weight_for(target, layer), rtol=0, atol=0)
        assert not torch.equal(layers[layer].eplb_state.expert_replica_routing_table, initial_routing[layer])
        dist.barrier()
        assert path.exists() == last
        dist.barrier()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "ep_size": _RANKS,
        "models": {
            "toy": {
                "model": "synthetic",
                "revision": "test-revision",
                "num_logical_experts": _EXPERTS,
                "num_redundant_experts": 0,
                "physical_to_logical_map": target.tolist(),
            }
        },
    }


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A3 or A5",
)
def test_mrv2_exports_after_final_commit(tmp_path):
    workers = mp.spawn(
        _run_placement_export, args=(get_open_port(), tmp_path / "placement.json"), nprocs=_RANKS, join=False
    )
    deadline = time.monotonic() + _TIMEOUT
    try:
        while not workers.join(timeout=5):
            assert time.monotonic() < deadline, "Placement export workers timed out"
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
