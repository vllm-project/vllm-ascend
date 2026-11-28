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
_SLOTS = 2
_EXPERTS = _RANKS * _SLOTS
_WIDTH = 128
_TIMEOUT = 120


def _worker(rank: int, port: int, rank0_path: Path, rank1_path: Path) -> None:
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    dist.init_process_group(
        "hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_RANKS, timeout=timedelta(seconds=_TIMEOUT)
    )
    ep_group = SimpleNamespace(rank_in_group=rank, world_size=_RANKS, device_group=dist.group.WORLD)
    eplb_group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=None)
    try:
        with (
            patch.object(ascend, "get_ep_group", return_value=ep_group),
            patch.object(ascend, "get_eplb_group", return_value=eplb_group),
        ):
            _verify_export(rank, device, rank0_path if rank == 0 else rank1_path, rank0_path, rank1_path)
    finally:
        dist.destroy_process_group()


def _verify_export(rank: int, device: torch.device, local_path: Path, rank0_path: Path, rank1_path: Path) -> None:
    old = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32)
    target = torch.tensor([[0, 2, 1, 3], [2, 1, 0, 3]], dtype=old.dtype)
    slots = slice(rank * _SLOTS, (rank + 1) * _SLOTS)

    def weight_for(layout, layer_idx):
        return (layout[layer_idx, slots, None] * _WIDTH + torch.arange(_WIDTH) + 1000 * layer_idx).bfloat16()

    weights = [[weight_for(old, idx).to(device)] for idx in range(len(old))]
    logical_map, replica_count = (t.to(device) for t in upstream.compute_logical_maps(old, _EXPERTS))
    layers = []
    for layer in range(len(old)):
        routing = ascend.AscendEplbLayerState()
        routing.logical_to_physical_map, routing.logical_replica_count = logical_map[layer], replica_count[layer]
        routing.refresh_expert_replica_routing_table()
        layers.append(SimpleNamespace(eplb_state=routing))
    communicator = eplb_communicator.TorchDistNcclEplbCommunicator(dist.group.WORLD)
    window = torch.zeros((4, len(old), _EXPERTS), dtype=torch.int64, device=device)
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
        expert_load_window=window,
        expert_buffer=[torch.empty_like(weights[0][0])],
        communicator=communicator,
        rebalanced=True,
    )
    state = ascend.AscendEplbState.__new__(ascend.AscendEplbState)
    state.model_states = {}
    state.enable_placement_export(str(local_path))
    assert state._exporter is not None
    state.model_states["t"] = model_state
    state._placement_model_roles = {"t": "target"}
    if state._exporter.is_stage_leader:
        model_state._placement_export_map = old.clone()
        model_state._on_placement_layer_committed = state._exporter.mark_committed

    # 1. Baseline before first rearrangement (ratio 1.8): no file written.
    window.zero_()
    window[:, :, slots] = (90 if rank == 0 else 10) // _SLOTS
    state._maybe_evaluate_and_export_placement()
    dist.barrier()
    assert not rank0_path.exists() and not rank1_path.exists()

    physical, logical, counts = patch_eplb._prepare_commit_maps(model_state, target)
    for layer in range(len(old)):
        sources = target[layer].numpy().reshape(_RANKS, _SLOTS)
        metadata = stage_explicit_layer_transfer(
            old[layer],
            target[layer],
            sources // _SLOTS,
            sources % _SLOTS,
            weights[layer],
            model_state.expert_buffer,
            dist.group.WORLD,
            communicator,
            layer_idx=layer,
        )
        model_state.pending_result = patch_eplb._AscendAsyncLayerResult(
            layer_idx=layer,
            new_physical_to_logical_map=physical[layer],
            new_logical_to_physical_map=logical[layer],
            new_logical_replica_count=counts[layer],
            transfer_metadata=metadata,
            consumed_event=upstream.CpuGpuEvent(),
            is_last_result=layer == len(old) - 1,
        )
        upstream._move_to_workspace(model_state=model_state, ep_rank=rank)
        torch.testing.assert_close(weights[layer][0].cpu(), weight_for(target, layer), rtol=0, atol=0)

    # 2. Improved ratio (1.04) before second rearrangement: rank 0 writes target, worse ratio (1.6) keeps it.
    window.zero_()
    window[:, :, slots] = (52 if rank == 0 else 48) // _SLOTS
    state._maybe_evaluate_and_export_placement()
    dist.barrier()
    assert rank0_path.exists() and not rank1_path.exists()
    expected = {
        "version": 1,
        "ep_size": _RANKS,
        "target": {
            "num_logical_experts": _EXPERTS,
            "num_redundant_experts": 0,
            "physical_to_logical_map": target.tolist(),
        },
    }
    assert json.loads(rank0_path.read_text(encoding="utf-8")) == expected

    if state._exporter.is_stage_leader:
        model_state._placement_export_map.copy_(old)
        state._exporter.mark_committed("target")
    window.zero_()
    window[:, :, slots] = (80 if rank == 0 else 20) // _SLOTS
    state._maybe_evaluate_and_export_placement()
    dist.barrier()
    assert json.loads(rank0_path.read_text(encoding="utf-8")) == expected


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A3 or A5",
)
def test_exports_committed_placement_when_peak_to_mean_improves(tmp_path: Path):
    workers = mp.spawn(
        _worker,
        args=(get_open_port(), tmp_path / "rank0.json", tmp_path / "rank1" / "out.json"),
        nprocs=_RANKS,
        join=False,
    )
    deadline = time.monotonic() + _TIMEOUT
    try:
        while not workers.join(timeout=5):
            assert time.monotonic() < deadline, "Placement export workers timed out"
    finally:
        for proc in workers.processes:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)
