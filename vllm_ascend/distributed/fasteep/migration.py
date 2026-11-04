# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Move individual experts while keeping model parameter addresses stable."""

import torch
import torch_npu

from vllm_ascend.distributed.fasteep.placement import plan_transfers


class ExpertMigration:
    def __init__(self, layers, rank, comms, *, verify=False):
        self.layers = layers
        self.rank = rank
        self.comms = comms
        self.verify = verify

    @staticmethod
    def tensors(layer, slot):
        # Both layouts are used by the unquantized Ascend MoE implementation.
        for name in ("w13_weight", "w2_weight", "w13_bias", "w2_bias"):
            weights = getattr(layer, name, None)
            if weights is None:
                weights = getattr(layer, name + "_list", None)
            if weights is None and name.endswith("_bias"):
                continue
            if weights is None:
                raise ValueError(f"Missing expert parameter {name}.")
            yield weights[slot]

    @torch.inference_mode()
    def execute(self, current, target):
        verified = 0
        for layer, before, after in zip(self.layers, current, target):
            transfers = plan_transfers(before, after)
            checks = []
            # Capture all outgoing values before any target slot is written.
            # A cyclic permutation can otherwise overwrite a later donor.
            source_slots = {move.source_slot for move in transfers if move.source_rank == self.rank}
            snapshots = {
                slot: [
                    torch_npu.npu_format_cast(tensor, 2).contiguous().clone() for tensor in self.tensors(layer, slot)
                ]
                for slot in source_slots
            }
            for move in transfers:
                if self.rank not in (move.source_rank, move.target_rank):
                    continue
                if move.source_rank == move.target_rank:
                    for destination, source in zip(self.tensors(layer, move.target_slot), snapshots[move.source_slot]):
                        destination.copy_(source)
                        if self.verify:
                            checks.append(torch.all(torch_npu.npu_format_cast(destination, 2) == source))
                    continue
                if self.rank == move.source_rank:
                    for source in snapshots[move.source_slot]:
                        self.comms.transfer(
                            source.data_ptr(), source.numel() * source.element_size(), move.target_rank, send=True
                        )
                else:
                    # Receive in ND layout; copy_ restores the parameter's
                    # physical format without replacing its captured address.
                    for destination in self.tensors(layer, move.target_slot):
                        received = torch.empty(destination.shape, dtype=destination.dtype, device=destination.device)
                        self.comms.transfer(
                            received.data_ptr(),
                            received.numel() * received.element_size(),
                            move.source_rank,
                            send=False,
                        )
                        destination.copy_(received)
                        if self.verify:
                            checks.append(torch.all(torch_npu.npu_format_cast(destination, 2) == received))
            torch.npu.synchronize()
            if self.verify:
                matched = not checks or torch.stack(checks).all().item()
                if not all(self.comms.gather_objects(matched)):
                    raise RuntimeError("Expert migration readback differs from the transferred tensor.")
                verified += len(checks)
            self.comms.barrier()
        return verified
