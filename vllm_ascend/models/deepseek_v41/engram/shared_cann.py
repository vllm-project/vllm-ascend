# SPDX-License-Identifier: Apache-2.0
"""Fixed-membership A5 CANN Engram reader; private vendor APIs are not stable."""

import json
import threading
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu
from cann_ops_transformer.ops.mc2.common.elastic_buffer import ElasticBuffer

import vllm_ascend.ops  # noqa: F401

from .shared_rdma import checkpoint_metadata

_TIMEOUT = timedelta(seconds=1800)
_HELD_READERS = []

MAX_READERS = 8
MAX_SIDE_WORLD = MAX_READERS + 1
MAX_ROWS = 262144
BF16_ELEMS_PER_ROW = 132
PACKED_ROW_BYTES = 264
CODE_BYTES = 256


def _side_layout(model_rank, model_world, reader_count, rank_base):
    if not 0 <= model_rank < model_world or not 1 <= reader_count <= MAX_READERS:
        raise ValueError("invalid model rank or fixed reader count")
    if rank_base < 1 or rank_base + model_world - 1 > reader_count:
        raise ValueError("model ranks do not fit the assigned Engram reader slots")
    return model_rank + rank_base, {rank: rank + rank_base for rank in range(model_world)}


def _owner_only_ranges(rows, world):
    if not 0 < rows <= 2**31 - 1 or not 2 <= world <= MAX_SIDE_WORLD:
        raise ValueError("invalid owner-only Engram dimensions")
    return [(0, rows)]


def _group(store, name, side_rank, world, rank_map):
    substore = dist.PrefixStore(name, store)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.group_name = name.replace("-", "_")
    options.group_id = options.group_name
    options._timeout = _TIMEOUT
    backend = torch_npu._C._distributed_c10d.ProcessGroupHCCL(substore, side_rank, world, options)
    group = dist.ProcessGroup(substore, side_rank, world)
    group._register_backend(torch.device("npu"), dist.ProcessGroup.BackendType.CUSTOM, backend)
    # The side world has a different rank space from the model world.
    dist.distributed_c10d._world.pg_group_ranks[group] = rank_map
    return group


class CANNReader:
    """Reader that keeps IDs on NPU; all fixed group members start and close collectively."""

    direct = False
    local_fast = False
    device_only = True

    def __init__(self, model, endpoint, *, capacity, readers=None, rank_base=1, library_path=None):
        del library_path
        if not 0 < capacity <= MAX_ROWS:
            raise ValueError("invalid CANN Engram row capacity")
        self.closed = False
        self.failed = False
        self._lock = threading.RLock()
        self._uncertain = None
        self.capacity = capacity
        self.metadata = checkpoint_metadata(model)
        self.model_rank = dist.get_rank()
        model_world = dist.get_world_size()
        reader_count = model_world if readers is None else readers
        self.side_rank, rank_map = _side_layout(self.model_rank, model_world, reader_count, rank_base)
        side_world = reader_count + 1
        host, port = endpoint.rsplit(":", 1)
        self.store = dist.TCPStore(host, int(port), side_world, False, timeout=_TIMEOUT)
        if int(self.store.get("reader-count").decode()) != reader_count:
            raise ValueError("CANN owner and reader fixed membership differ")
        remote = json.loads(self.store.get("metadata").decode())
        if remote != self.metadata:
            raise ValueError("CANN owner and reader checkpoint metadata differ")
        if any(v["code_dtype"] != "F8_E4M3" or v["scale_dtype"] != "F8_E8M0" for v in self.metadata["tables"].values()):
            raise ValueError("CANN Engram reader requires FP8/E8M0 tables")
        self.groups = []
        self.buffers = {}
        if self.store.add(f"claim-{self.side_rank}", 1) != 1:
            raise ValueError("CANN reader slot already claimed")
        self.store.set(f"ready-{self.side_rank}", "1")
        try:
            for layer in self.metadata["tables"]:
                self.buffers[layer] = []
                for shard, (start, stop) in enumerate(
                    _owner_only_ranges(self.metadata["tables"][layer]["rows"], side_world)
                ):
                    self.store.wait([f"prepared-{layer}-{shard}"], _TIMEOUT)
                    group = _group(
                        self.store,
                        f"engram_{port}_{layer}_{shard}",
                        self.side_rank,
                        side_world,
                        rank_map,
                    )
                    self.groups.append(group)
                    storage = torch.zeros((1, BF16_ELEMS_PER_ROW), dtype=torch.bfloat16, device="cpu")
                    size = ElasticBuffer.get_engram_storage_size_hint(1, BF16_ELEMS_PER_ROW, torch.bfloat16)
                    buffer = ElasticBuffer(group, num_cpu_bytes=size, explicitly_destroy=True)
                    self.buffers[layer].append((start, stop, buffer))
                    buffer.engram_write(storage, owner_only_num_entries=stop - start)
        except BaseException:
            # A partial registration may still be visible to peers.
            self._hold()
            raise

    def _hold(self):
        if not self.failed:
            self.failed = True
            _HELD_READERS.append(self)

    def lookup_layers(self, tables, gathered, cpu_hashes, *, out_buffers=None):
        return None

    def lookup(self, table, indices, out, *, cpu_ids=None):
        with self._lock:
            return self._lookup(table, indices, out, cpu_ids=cpu_ids)

    def _lookup(self, table, indices, out, *, cpu_ids=None):
        del cpu_ids
        from vllm_ascend.ops.triton.engram_lookup import dequantize_packed_engram

        if self.closed or self.failed:
            raise RuntimeError("CANN Engram reader is closed or held after failure")
        layer = str(table._checkpoint_key.split(".")[1])
        rows = self.metadata["tables"][layer]["rows"]
        if table.num_embeddings != rows or table.vocab_start_idx != 0 or table.vocab_end_idx != rows:
            raise ValueError("CANN Engram reader requires a complete unsharded vocabulary")
        selected = indices[:, table.head_start : table.head_start + table.part_n_hash_cols].reshape(-1)
        if selected.numel() > self.capacity:
            raise ValueError("CANN Engram lookup exceeds configured row capacity")
        pending = [indices, out, selected]
        try:
            target = out.view(-1, CODE_BYTES)
            pending.append(target)
            target.zero_()
            for start, stop, buffer in self.buffers[layer]:
                valid = (selected >= start) & (selected < stop)
                safe = torch.where(valid, selected - start, 0).to(torch.int32)
                pending.extend((valid, safe))
                fetch_wait = buffer.engram_fetch(safe)
                pending.append(fetch_wait)
                packed = fetch_wait()
                pending.append(packed)
                decoded = torch.empty((selected.numel(), CODE_BYTES), dtype=torch.bfloat16, device=indices.device)
                pending.append(decoded)
                dequantize_packed_engram(
                    packed.view(torch.uint8), decoded, selected.numel(), PACKED_ROW_BYTES, "F8_E8M0", "F8_E4M3"
                )
                merged = torch.where(valid[:, None], decoded, target)
                pending.append(merged)
                target.copy_(merged)
        except BaseException:
            # A failed fetch/wait does not prove that remote reads have retired.
            self._uncertain = pending
            self._hold()
            raise

    def close(self):
        # The caller must prevent future graph replays before calling close.
        with self._lock:
            if self.closed:
                return
            if self.failed:
                raise RuntimeError("CANN Engram reader held after failure")
            try:
                torch.npu.synchronize()
                self.store.set(f"done-{self.side_rank}", "1")
                for shards in reversed(list(self.buffers.values())):
                    for _, _, buffer in reversed(shards):
                        buffer.destroy()
                for group in self.groups:
                    dist.distributed_c10d._world.pg_group_ranks.pop(group, None)
                self.closed = True
            except BaseException:
                self._hold()
                raise
