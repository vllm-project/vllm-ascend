# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One PLE table backed by an independently owned MemFabric Offload instance."""

import ctypes

import torch
import torch.distributed as dist

GIB = 1 << 30
MAX_COPY_BYTES = 32 * 1024
COPY_ALIGNMENT = 32
BF16_BYTES = 2


def prepare_row_copies(ids, output, source, vocab_start, vocab_end, row_bytes):
    """Build device-only copy metadata; invalid rows remain zero in output.

    MemFabric's sparse-copy KV path pairs the first and second halves. Pad
    odd counts with a zero-length entry so its last real row is not dropped.
    """
    flat = ids.reshape(-1).to(torch.int64)
    valid = (flat >= vocab_start) & (flat < vocab_end)
    local = torch.where(valid, flat - vocab_start, 0)
    src = local * row_bytes + source
    dst = torch.arange(flat.numel(), dtype=torch.int64, device=ids.device) * row_bytes + output.data_ptr()
    lengths = valid.to(torch.int32) * row_bytes
    if flat.numel() % 2:
        src = torch.cat((src, src[:1]))
        dst = torch.cat((dst, dst[:1]))
        lengths = torch.cat((lengths, lengths.new_zeros(1)))
    count = torch.full((1,), src.numel(), dtype=torch.int32, device=ids.device)
    return src, dst, lengths, count


class PLEGvaTable:
    """CPU weight storage on the owner, GVA access on all participating ranks.

    Callers must stop graph replay before close, and close shared tables on
    every rank. No module-level Offload API is used: KV pools remain separate.
    """

    def __init__(self, shape, device, *, group=None, store_url=None):
        # Worker-only optional dependency; importing this module is CPU-safe.
        from memfabric_hybrid import offload

        if not callable(getattr(offload, "Offload", None)):
            raise RuntimeError("PLE GVA requires MemFabric with independent offload.Offload() instances")
        rows, width = shape
        if rows <= 0 or width <= 0 or width * BF16_BYTES > MAX_COPY_BYTES or width * BF16_BYTES % COPY_ALIGNMENT:
            raise ValueError("PLE GVA requires positive rows and BF16 row bytes divisible by 32 and <= 32768")
        if group is not None and (not isinstance(store_url, str) or not store_url.startswith("tcp://")):
            raise ValueError("Shared PLE GVA requires a dedicated tcp:// store URL for each ETP shard")
        self.device = device
        self.group = group
        self.owner = group is None or group.rank_in_group == 0
        self.tensor = None
        self._buffer = None
        self.pointer = 0
        self.source = 0
        self._initialized = False
        self._closed = False
        self._closing = False
        self._quiesced = False
        self.pool = offload.Offload()
        self.shape = shape
        size = rows * width * BF16_BYTES
        config = offload.OffloadConfig()
        config.device_id = device.index
        config.reserve_size = ((size + GIB - 1) // GIB) * GIB
        config.alloc_size = config.reserve_size if self.owner else 0
        config.scene = offload.Scene.LOCAL if group is None else offload.Scene.SHARED
        config.world_size = 1 if group is None else group.world_size
        config.rank_id = 0 if group is None else group.rank_in_group
        config.local_world_size = config.world_size
        if store_url is not None:
            config.store_url = store_url
        rc = self.pool.initialize(config)
        if rc != 0:
            raise RuntimeError(f"PLE GVA initialize failed: rc={rc}")
        self._initialized = True
        # Keep the partially constructed owner reachable if cleanup must retry.
        try:
            error = None
            if self.owner:
                try:
                    self.pointer = self.pool.malloc(size)
                    if not self.pointer:
                        raise RuntimeError("PLE GVA allocation failed")
                    self._buffer = (ctypes.c_ubyte * size).from_address(self.pointer)
                    self.tensor = torch.frombuffer(self._buffer, dtype=torch.bfloat16).reshape(shape)
                    self.tensor.zero_()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            if group is not None:
                payload = [self.pointer, error]
                dist.broadcast_object_list(payload, src=group.ranks[0], group=group.cpu_group)
                self.source, error = payload
            else:
                self.source = self.pointer
            if error is not None:
                raise RuntimeError(error)
            if not self.owner:
                # Only the owner loads checkpoint rows. Consumers need shape
                # and dtype metadata, not a duplicate CPU allocation.
                self.tensor = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup:
                raise PLEGvaCleanupError(self, exc, cleanup) from exc
            raise

    def lookup(self, ids, vocab_start, vocab_end):
        if self._closing or self._closed or not self._initialized:
            raise RuntimeError("PLE GVA table is closed")
        if ids.device != self.device:
            raise ValueError("PLE GVA IDs must be on the table's NPU")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("PLE GVA IDs must use int32 or int64")
        if not 0 <= vocab_end - vocab_start <= self.shape[0]:
            raise ValueError("PLE vocabulary range exceeds its backing allocation")
        output = torch.zeros((*ids.shape, self.shape[1]), dtype=torch.bfloat16, device=ids.device)
        if ids.numel() == 0:
            return output
        metadata = prepare_row_copies(ids, output, self.source, vocab_start, vocab_end, self.shape[1] * BF16_BYTES)
        rc = self.pool.sparse_copy(*metadata, self.device)
        if rc != 0:
            raise RuntimeError(f"PLE GVA sparse_copy failed: rc={rc}")
        return output

    def close(self):
        if self._closed:
            return
        self._closing = True
        if not self._quiesced:
            torch.npu.synchronize(self.device)
            if self.group is not None:
                dist.barrier(group=self.group.cpu_group)
            self._quiesced = True
        # Release tensor aliases before unmapping the allocation. The embedding
        # must also detach its Parameter before calling this method.
        self.tensor = None
        self._buffer = None
        if self.pointer:
            rc = self.pool.free(self.pointer)
            if rc not in (None, 0):
                raise RuntimeError(f"PLE GVA free failed: rc={rc}")
            self.pointer = 0
        if self._initialized:
            self.pool.uninitialize()
            self._initialized = False
        self._closed = True


class PLEGvaCleanupError(RuntimeError):
    def __init__(self, table, original, cleanup):
        super().__init__(f"PLE GVA construction failed ({original}); cleanup failed ({cleanup})")
        self.table = table
