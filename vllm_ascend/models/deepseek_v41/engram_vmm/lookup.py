# SPDX-License-Identifier: Apache-2.0
"""Fused grouped gather/dequant, with final-buffer stores and padding."""

import torch
from vllm.triton_utils import tl, triton

CHUNK_ROWS = 1 << 22


@triton.jit
def engram_chunked(
    codes_ptrs,
    scales_ptrs,
    ids,
    count,
    valid_mask,
    out,
    mask_out,
    CAPACITY: tl.constexpr,
    K: tl.constexpr,
    MASK_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
    WRITE_MASK: tl.constexpr,
    TILE: tl.constexpr = 1,
):
    active = tl.load(count)
    col = tl.arange(0, 256)
    if TILE == 1:
        for scalar_row in range(tl.program_id(0), active, tl.num_programs(0)):
            index = tl.load(ids + scalar_row)
            chunk = index // CHUNK
            local = index % CHUNK
            weight = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
            scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
            value = tl.load(weight + local * 256 + col).to(tl.float32)
            scale = tl.load(scales + local * 8 + col // 32)
            tl.store(out + scalar_row * 256 + col, (value * scale).to(tl.bfloat16))
    else:
        # Tiled mode is only for contiguous VMM mappings, not separately
        # registered chunks. Their first pointer addresses the entire table.
        weight = tl.load(codes_ptrs).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs).to(tl.pointer_type(tl.float32))
        rr = tl.arange(0, TILE)
        groups = tl.arange(0, 8)
        for start in range(tl.program_id(0) * TILE, active, tl.num_programs(0) * TILE):
            row = start + rr
            index = tl.load(ids + row, row < active, 0)
            value = tl.load(weight + index[:, None] * 256 + col[None, :], row[:, None] < active, 0).to(tl.float32)
            scale = tl.load(scales + index[:, None] * 8 + groups[None, :], row[:, None] < active, 0)
            result = (value.reshape((TILE, 8, 32)) * scale[:, :, None]).reshape((TILE, 256))
            tl.store(out + row[:, None] * 256 + col[None, :], result.to(tl.bfloat16), row[:, None] < active)
    lane = tl.arange(0, 16384)
    for start in range(active * 256 + tl.program_id(0) * 16384, CAPACITY * 256, tl.num_programs(0) * 16384):
        off = start + lane
        tl.store(out + off, 0, off < CAPACITY * 256)
    if WRITE_MASK:
        if tl.program_id(0) == 0:
            token = tl.arange(0, MASK_SIZE)
            valid = tl.load(valid_mask + token, token * K < active, 0)
            tl.store(mask_out + token, valid, token < CAPACITY // K)


class Inputs:
    def __init__(self, tables, layer_ids, capacity, k, outputs=None, mask_output=None, tile=16):
        if tile not in (1, 16):
            raise ValueError("Supported lookup tiles are 1 and 16")
        if tile != 1 and any(not hasattr(t.codes_map, "tensor") or not hasattr(t.scales_map, "tensor") for t in tables):
            raise ValueError("Tiled lookup requires contiguous VMM tensor mappings")
        from triton.runtime import driver  # type: ignore[import-untyped]

        self.closed = False
        self.tables, self.layers, self.capacity, self.k = tables, layer_ids, capacity, k
        self.initial_counts = tables[0].codes_map.lifecycle_counts()
        self.calls = 0
        device = tables[0].codes_map.ptrs.device
        self.ids = [torch.empty(capacity * k, dtype=torch.int64, device=device) for _ in tables]
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.valid = torch.empty(capacity, dtype=torch.bool, device=device)
        self.mask = mask_output if mask_output is not None else torch.empty_like(self.valid)
        self.outputs = (
            outputs
            if outputs is not None
            else [torch.empty(capacity, k * 256, dtype=torch.bfloat16, device=device) for _ in tables]
        )
        self.grid = (driver.active.utils.get_device_properties("npu")["num_vectorcore"], 1, 1)
        self.kernels = []
        for i, t in enumerate(tables):
            self.kernels.append(
                engram_chunked.warmup(
                    t.codes_map.ptrs,
                    t.scales_map.ptrs,
                    self.ids[i],
                    self.count,
                    self.valid,
                    self.outputs[i],
                    self.mask,
                    CAPACITY=capacity * k,
                    K=k,
                    MASK_SIZE=triton.next_power_of_2(capacity),
                    CHUNK=CHUNK_ROWS,
                    WRITE_MASK=i == 0,
                    TILE=tile,
                    grid=self.grid,
                    num_warps=4,
                )
            )

    def prepare(self, hashes, mask):
        if self.closed or any(
            t.codes_map.closed or t.codes_map.tensor is None or t.scales_map.closed or t.scales_map.tensor is None
            for t in self.tables
        ):
            raise RuntimeError("Engram inputs/table are closed")
        self.calls += 1
        if self.calls % 100 == 1:
            counts = self.tables[0].codes_map.lifecycle_counts()
            if counts != self.initial_counts:
                raise RuntimeError("VMM mapping lifecycle changed during lookup")
            print(f"ENGRAM_VMM_STABLE calls={self.calls} allocations={counts[0]} frees={counts[1]}", flush=True)
        n = hashes.shape[0]
        if n > self.capacity:
            raise ValueError("Engram input exceeds static capacity")
        if hashes.device.type != "cpu" or hashes.dtype != torch.int64 or hashes.shape != (n, len(self.tables), self.k):
            raise ValueError("Engram hashes must be CPU INT64 [tokens, layers, heads]")
        if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != (n,):
            raise ValueError("Engram mask must be CPU BOOL [tokens]")
        # Hashes originate on CPU. Reject invalid addresses before launching
        # a direct mapped-host load, without synchronizing an NPU tensor.
        for i, table in enumerate(self.tables):
            if n and (hashes[:, i].min() < 0 or hashes[:, i].max() >= table.codes_map.rows):
                raise ValueError("Engram row ID outside table")
        for i in range(len(self.tables)):
            if n:
                self.ids[i][: n * self.k].copy_(hashes[:, i].reshape(-1))
        self.count.copy_(torch.tensor([n * self.k], dtype=torch.int32, device="cpu"))
        if n:
            self.valid[:n].copy_(mask)
        for i, t in enumerate(self.tables):
            self.kernels[i][self.grid](
                t.codes_map.ptrs, t.scales_map.ptrs, self.ids[i], self.count, self.valid, self.outputs[i], self.mask
            )
        return {"engram_lookups": dict(zip(self.layers, self.outputs)), "engram_mask": self.mask}

    def close(self):
        if self.closed:
            return
        torch.npu.synchronize()
        self.closed = True
        self.kernels.clear()
        self.ids.clear()
        self.outputs.clear()
        self.mask = self.count = self.valid = None
        self.tables = []
