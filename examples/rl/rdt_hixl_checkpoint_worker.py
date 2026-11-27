# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank checkpoint oracle for the opt-in HIXL example verification."""

import hashlib
import json
from pathlib import Path

import torch

from vllm_ascend.worker.worker import NPUWorker


class CheckpointWorker(NPUWorker):
    def _parameter_hashes(self) -> dict[str, str]:
        torch.npu.synchronize()
        hashes = {}
        for name, param in self.model_runner.get_model().named_parameters():
            data = param.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            hashes[name] = hashlib.sha256(data).hexdigest()
        return hashes

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        self._reference = self._parameter_hashes()
        self._verified_round = 0
        super().init_weight_transfer_engine(init_info)
        # Each completed replay reads exactly pack_bytes once per planned chunk.
        # This counts bytes presented to HIXL, excluding link/protocol overhead.
        plan = self.weight_transfer_engine._cached_plan
        self._read_bytes = sum(chunk.pack_bytes for chunk in plan.chunks)

    def start_weight_update(self) -> None:
        if self._verified_round == 0:
            # Generation is paused. Make a skipped transfer observable even
            # though this worker and the trainer started from the same weights.
            with torch.no_grad():
                for param in self.model_runner.get_model().parameters():
                    param.zero_()
            torch.npu.synchronize()
        super().start_weight_update()

    def finish_weight_update(self) -> None:
        super().finish_weight_update()
        actual = self._parameter_hashes()
        mismatches = {
            name: (self._reference.get(name), actual.get(name))
            for name in self._reference.keys() | actual.keys()
            if actual.get(name) != self._reference.get(name)
        }
        self._verified_round += 1
        rank = self.parallel_config.data_parallel_index
        path = Path(self.vllm_config.additional_config["hixl_checkpoint_verify_dir"]) / f"rank-{rank}.json"
        path.write_text(
            json.dumps(
                {
                    "round": self._verified_round,
                    "parameters": len(actual),
                    "planned_read_bytes": self._read_bytes,
                    "mismatches": mismatches,
                }
            )
        )
        if mismatches:
            raise AssertionError(f"Rank {rank}, sync {self._verified_round}: checkpoint mismatch: {mismatches}")
