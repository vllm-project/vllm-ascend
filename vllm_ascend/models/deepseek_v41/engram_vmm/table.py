# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-shared host tables, with the existing checkpoint reader and quantizer."""

from pathlib import Path

import torch

from ..engram_hbm import NodeShardedEngram
from .mapping import VmmMapping


class VmmEngram(NodeShardedEngram):
    def __init__(self, rows, width, query_group, *, run_id, layer_id):
        if width != 256:
            raise ValueError("Engram VMM requires width 256, INT8 codes and FP32 group32 scales")
        super().__init__(rows, width, query_group, device="meta", storage_format="int8")
        root = Path("/dev/shm") / f"engram-vmm-{run_id}"
        self.closed = False
        self.loaded = False
        self.codes_map = VmmMapping(root / f"{layer_id}.codes", rows, width, torch.int8, query_group)
        try:
            self.scales_map = VmmMapping(root / f"{layer_id}.scales", rows, width // 32, torch.float32, query_group)
        except Exception as exc:
            self.codes_map._rollback(exc)
            raise
        try:
            self.weight = torch.nn.Parameter(self.codes_map.tensor[self.start : self.end], requires_grad=False)
            self.weight_scale = self.scales_map.tensor[self.start : self.end]
        except Exception as original:
            self.weight = None
            self.weight_scale = None
            errors = []
            for mapping in (self.scales_map, self.codes_map):
                try:
                    mapping._rollback(original)
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError("Engram construction rollback: " + "; ".join(errors)) from original
            raise

    def set_int8_rows(self, start, codes, scales):
        if start < 0 or start + len(codes) > self.end - self.start:
            raise ValueError("Engram publication exceeds this rank's checkpoint shard")
        self.codes_map.publish(self.start + start, codes)
        self.scales_map.publish(self.start + start, scales)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        super().load_checkpoint(model_path, key, chunk_rows)
        torch.npu.synchronize()
        self.loaded = True

    def forward(self, ids):
        raise RuntimeError("Engram VMM must use model preparation's direct lookup, not owner routing")

    def close(self):
        self.closed = True
        self.weight = None
        self.weight_scale = None
        errors = []
        for mapping in (self.scales_map, self.codes_map):
            try:
                mapping.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Engram VMM cleanup: " + "; ".join(errors))
