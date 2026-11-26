# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.lora.moe_dp import MoELoRACoordinator, lora_ids_to_slots, lora_slots_to_ids


@dataclass(frozen=True)
class AdapterRequest:
    """Picklable request metadata; the coordinator does not load checkpoints."""

    lora_int_id: int
    lora_path: str
    __struct_fields__: ClassVar = ("lora_int_id", "lora_path")


class ExpertAdapterManager:
    """CPU expert shards with independent slot histories on the two workers."""

    def __init__(self, rank):
        self.rank = rank
        self.device = torch.device("cpu")
        self._adapter_manager = SimpleNamespace(lora_slots=2, lora_index_to_id=[11 if rank == 0 else 22, None])
        self.loaded = {11 if rank == 0 else 22}
        self.activations = 0
        self.fail_adapter = None

    def set_active_adapters(self, requests, mapping):
        assert mapping is None
        ids = sorted(request.lora_int_id for request in requests)
        for adapter_id in ids:
            if adapter_id == self.fail_adapter:
                raise FileNotFoundError(f"adapter {adapter_id} on rank {self.rank}")
            slots = self._adapter_manager.lora_index_to_id
            if adapter_id in slots:
                continue
            free = next((i for i, old_id in enumerate(slots) if old_id is None or old_id not in ids), None)
            assert free is not None
            slots[free] = adapter_id
            self.loaded.add(adapter_id)
            self.activations += 1

    def expert_delta(self, slots, inputs):
        # Different expert shards and adapters produce distinguishable deltas.
        # The expected values below are calculated from global adapter IDs.
        resident = self._adapter_manager.lora_index_to_id
        factors = torch.tensor([(adapter_id or 0) * (self.rank + 1) for adapter_id in resident], dtype=inputs.dtype)
        return torch.where(slots >= 0, inputs * factors[slots.clamp(min=0)], 0)


def test_slot_translation_preserves_padding_and_worker_local_layout():
    sender_slots = torch.tensor([11, 22, -1])
    receiver_slots = torch.tensor([22, -1, 11])
    adapter_ids = lora_slots_to_ids(torch.tensor([0, -1, 1, 0]), sender_slots)
    torch.testing.assert_close(adapter_ids, torch.tensor([11, -1, 22, 11]))
    torch.testing.assert_close(lora_ids_to_slots(adapter_ids, receiver_slots), torch.tensor([2, -1, 0, 2]))
    assert lora_ids_to_slots(torch.empty(0, dtype=torch.long), receiver_slots).numel() == 0


def test_single_dp_uses_existing_slot_indices():
    indices = torch.tensor([1, -1, 0])
    assert lora_slots_to_ids(indices, None) is indices
    assert lora_ids_to_slots(indices, None) is indices


def _dp_worker(rank, init_method):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=30))
    try:
        manager = ExpertAdapterManager(rank)
        wrapper = SimpleNamespace()
        coordinator = MoELoRACoordinator(manager, [wrapper], dist.group.WORLD)
        slot_buffer = wrapper.moe_lora_slot_ids.data_ptr()
        requests = [AdapterRequest(11, "/a"), AdapterRequest(22, "/b")]

        # Both ranks started with slot 0 occupied by different adapters. Each
        # receiver must load the remote adapter's expert shard before routing.
        coordinator.synchronize({requests[rank]})
        assert set(manager._adapter_manager.lora_index_to_id) == {11, 22}
        local_slots = torch.tensor([0, -1], dtype=torch.long)
        wire_ids = lora_slots_to_ids(local_slots, coordinator.slot_ids)
        gathered = [torch.empty_like(wire_ids) for _ in range(2)]
        dist.all_gather(gathered, wire_ids)
        global_ids = torch.cat(gathered)
        torch.testing.assert_close(global_ids, torch.tensor([11, -1, 22, -1]))
        recovered = lora_ids_to_slots(global_ids, coordinator.slot_ids)
        inputs = torch.arange(1, 5, dtype=torch.float32)
        reference = torch.tensor([11.0, 0.0, 66.0, 0.0]) * (rank + 1)
        torch.testing.assert_close(manager.expert_delta(recovered, inputs), reference)

        # Exercise the wire metadata with a real CPU AlltoAll as well.
        received_ids = torch.empty_like(wire_ids)
        dist.all_to_all_single(received_ids, wire_ids)
        received_slots = lora_ids_to_slots(received_ids, coordinator.slot_ids)
        expected = torch.tensor([11.0, 22.0]) * (rank + 1) if rank == 0 else torch.zeros(2)
        torch.testing.assert_close(manager.expert_delta(received_slots, torch.ones(2)), expected)

        # An idle DP rank still participates and loads the peer's adapter.
        # A third adapter reuses a cache slot; graph-captured pointers stay valid.
        coordinator.step_active = False
        coordinator.synchronize(set() if rank == 0 else {AdapterRequest(33, "/c")})
        assert 33 in manager._adapter_manager.lora_index_to_id
        assert wrapper.moe_lora_slot_ids.data_ptr() == slot_buffer
        recovered = lora_ids_to_slots(torch.tensor([33, -1]), coordinator.slot_ids)
        torch.testing.assert_close(
            manager.expert_delta(recovered, torch.ones(2)), torch.tensor([33.0, 0.0]) * (rank + 1)
        )

        # The union capacity, rather than only the local request count, matters.
        coordinator.step_active = False
        with pytest.raises(ValueError, match="--max-loras"):
            coordinator.synchronize(set(requests) if rank == 0 else {AdapterRequest(33, "/c")})
        assert not coordinator.step_active

        with pytest.raises(ValueError, match="Conflicting LoRA requests"):
            coordinator.synchronize({AdapterRequest(11, "/a" if rank == 0 else "/different")})

        # Only one worker fails to load. Both must report it before entering
        # model collectives; the successful peer must not hang in an AlltoAll.
        if rank == 1:
            manager.fail_adapter = 44
        with pytest.raises(RuntimeError, match="activation failed.*FileNotFoundError"):
            coordinator.synchronize({AdapterRequest(44, "/missing")} if rank == 0 else set())
        assert not coordinator.step_active
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_cross_dp_residency_routing_and_collective_errors(tmp_path):
    mp.spawn(_dp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
