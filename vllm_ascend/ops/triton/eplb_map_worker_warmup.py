# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import triton


class WorkerMapKernel:
    """Collect layer geometry until the pre-registry worker reaches kernel_warmup."""

    def __init__(self, kernel, tiles: tuple[int, ...], routing_rows: int):
        self.kernel = kernel
        self.tiles = tiles
        self.routing_rows = routing_rows
        self.pending: set[tuple[int, int]] = set()

    def register_warmup(self, experts: int, topk: int) -> None:
        self.pending.add((experts, topk))

    def warmup(self) -> None:
        for experts, topk in sorted(self.pending):
            for tile in self.tiles:
                for dtype in (torch.int32, torch.int64):
                    self.kernel.warmup(
                        dtype,
                        torch.int32,
                        dtype,
                        experts,
                        tile,
                        self.routing_rows,
                        TOPK=topk,
                        BLOCK_SIZE=tile,
                        grid=(1,),
                    )
        self.pending.clear()

    def __call__(self, ids, table, output, experts, numel, rows, *, TOPK, BLOCK_SIZE):
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        self.kernel[grid](ids, table, output, experts, numel, rows, TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE)
