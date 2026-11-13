import base64
import contextlib
import pickle
import typing
import weakref
from dataclasses import dataclass
from multiprocessing import shared_memory
from zlib import adler32

import numpy as np
import torch
import torch_npu

with contextlib.suppress(Exception):
    # we should remove this after memfabric.offload is merged to master and available in ci machine.
    # Importing the package provisions an AICPU kernel on the fly, which raises its own
    # ProvisioningError rather than ImportError when the CANN build toolchain is incomplete.
    # Sparse KV offload is opt-in, so nothing here may abort worker startup.
    from memfabric_hybrid import offload  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_config import SparseKVOffloadConfig, get_ascend_config
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import copy_sfa_pool_capacity
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

# Main BF16 cache:
# [k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v].
# Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5
OFFLOAD_STORE_PORT_BASE = 8500

FSA_EXTERNAL_PLAN_READY_MARKER = 0x5A45
FSA_PAIRED_SELECTION_COPY_MARKER = 0x5A56
FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT = 16376
FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT = 16
FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT = 8
FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT = (
    (FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1)
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT = (
    (
        FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
        + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
        + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
        - 1
    )
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS = (
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
)


_SPARSE_KV_OFFLOAD_OPS = None
_SPARSE_KV_OFFLOAD_OP_NAMES = (
    "sparse_kv_warmup_lru_resident_threads",
    "sparse_kv_lru_resident_compact_with_plan_stable_rows",
    "sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows",
    "sparse_kv_restore_bfloat16_tensor",
    "sparse_kv_restore_int16_tensor",
)


def _sparse_kv_ops():
    global _SPARSE_KV_OFFLOAD_OPS
    if _SPARSE_KV_OFFLOAD_OPS is not None:
        return _SPARSE_KV_OFFLOAD_OPS
    if not enable_custom_op():
        raise RuntimeError("Sparse KV offload requires custom ops to be enabled.")

    missing_ops = [name for name in _SPARSE_KV_OFFLOAD_OP_NAMES if not hasattr(torch.ops._C_ascend, name)]
    if missing_ops:
        raise RuntimeError(
            "Sparse KV offload requires _C_ascend sparse KV ops, "
            f"missing={missing_ops}. Rebuild vllm_ascend_C with Atlas A3 support."
        )
    _SPARSE_KV_OFFLOAD_OPS = torch.ops._C_ascend
    return _SPARSE_KV_OFFLOAD_OPS


SELECTIVE_LIM_LAYER_FIELDS = 16


class HostGatherCoordinator:
    """Share only gathered rows; the complete KV pool remains owned by MemFabric."""

    def __init__(self, group, rank: int, world_size: int, capacity_bytes: int):
        header_bytes = ((6 + world_size) * 8 + 63) // 64 * 64
        if rank == 0:
            self.mapping = shared_memory.SharedMemory(create=True, size=header_bytes + capacity_bytes)
            self.mapping.buf[:header_bytes] = bytes(header_bytes)
            identity = self.mapping.name
        else:
            identity = None
        name = group.broadcast_object(identity, src=0)
        if rank != 0:
            self.mapping = shared_memory.SharedMemory(name=name)
        self.tensor = torch.frombuffer(self.mapping.buf, dtype=torch.uint8)
        self.control = self.tensor[:header_bytes].view(torch.int64)
        torch.ops._C_ascend.register_pinned_host_mmap(self.tensor)
        self.registration = weakref.finalize(self, torch.ops._C_ascend.unregister_pinned_host_mmap, self.tensor)
        self.payload = self.tensor[header_bytes:]
        self.epoch = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self.rank = rank
        self.world_size = world_size
        group.barrier()
        if rank == 0:
            self.mapping.unlink()


_SELECTIVE_NATIVE_TP_QUARANTINE = []


class SelectiveNativeTPCoordinator:
    """Pinned shared plan and single per-group K/V stage for the TP experiment."""

    def __init__(self, group, rank, world, rows, width, layers, block, k_bytes, v_bytes, layer_ids, shared_exchange=False):
        plan_width = 3 * width + width // block + 2
        sizes = [
            (28 + 2 * world) * 8, rows * plan_width * 4, rows * 12 * 8,
            rows * width * 4, layers * rows * width * 4, (layers + 1) * 8,
            rows * width * k_bytes, rows * width * v_bytes,
        ]
        if shared_exchange:
            sizes.append((1 + 3 * world) * 8 + world * ((1 << 20) + 8))
        offsets, total = [], 0
        for size in sizes:
            total = (total + 63) // 64 * 64
            offsets.append(total)
            total += size
        total = (total + 63) // 64 * 64
        self.total_bytes = total
        self.mapping = None
        self.owner = None
        create_error = None
        if rank == 0:
            try:
                self.mapping = shared_memory.SharedMemory(create=True, size=total)
            except Exception as error:
                create_error = str(error)
        try:
            creation = group.broadcast_object((self.mapping.name, None) if self.mapping is not None else
                                              (None, create_error) if rank == 0 else None, src=0)
        except BaseException:
            if self.mapping is not None:
                _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            raise
        if creation[1] is not None:
            raise RuntimeError(f"Native selective TP shared mapping creation failed: {creation[1]}")
        registered = False
        attempted_registration = False
        local_error = None
        try:
            if rank != 0:
                self.mapping = shared_memory.SharedMemory(name=creation[0])
            self.owner = torch.frombuffer(self.mapping.buf, dtype=torch.uint8)
            if rank == 0:
                self.owner.zero_()
            attempted_registration = True
            torch.ops._C_ascend.register_pinned_host_mmap(self.owner)
            registered = True

            def view(index, dtype, shape):
                return self.owner[offsets[index] : offsets[index] + sizes[index]].view(dtype).view(shape)

            self.stage_k = view(6, torch.uint8, (rows, width, k_bytes))
            self.stage_v = view(7, torch.uint8, (rows, width, v_bytes))
            self.exchange = view(8, torch.uint8, (sizes[8],)) if shared_exchange else None
            self.buffers = [view(0, torch.int64, (28 + 2 * world,)),
                            view(1, torch.int32, (rows, plan_width)),
                            view(2, torch.int64, (rows, 12)),
                            view(3, torch.int32, (rows, width)),
                            view(4, torch.int32, (layers, rows, width)), self.owner,
                            torch.tensor(layer_ids, dtype=torch.int64), view(5, torch.int64, (layers + 1,))]
        except Exception as error:
            local_error = str(error)
        errors = [None] * world
        try:
            torch.distributed.all_gather_object(errors, local_error, group=group.cpu_group)
        except BaseException:
            _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            raise
        if any(error is not None for error in errors):
            if rank == 0 and self.mapping is not None:
                try:
                    self.mapping.unlink()
                except Exception:
                    _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
                    raise
            if attempted_registration and not registered:
                # A failed runtime registration may have partially succeeded.
                _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            else:
                try:
                    if registered:
                        torch.ops._C_ascend.unregister_pinned_host_mmap(self.owner)
                    if hasattr(self, "buffers"):
                        self.buffers.clear()
                    self.stage_k = self.stage_v = None
                    self.exchange = None
                    self.owner = None
                    if self.mapping is not None:
                        self.mapping.close()
                except Exception:
                    _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            raise RuntimeError(f"Native selective TP shared mapping registration failed: {errors}")
        self.registration = weakref.finalize(self, torch.ops._C_ascend.unregister_pinned_host_mmap, self.owner)
        try:
            group.barrier()
        except BaseException:
            _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            raise
        unlink_error = None
        if rank == 0:
            try:
                self.mapping.unlink()
            except Exception as error:
                unlink_error = str(error)
        errors = [None] * world
        try:
            torch.distributed.all_gather_object(errors, unlink_error, group=group.cpu_group)
        except BaseException:
            _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            raise
        if any(error is not None for error in errors):
            _SELECTIVE_NATIVE_TP_QUARANTINE.append(self)
            raise RuntimeError(f"Native selective TP shared mapping unlink failed: {errors}")

    def close(self):
        self.registration()
        self.buffers.clear()
        self.stage_k = self.stage_v = None
        self.exchange = None
        self.owner = None
        self.mapping.close()


class SelectiveSDMATransfer:
    """Fixed-address Host gather, SDMA staging, and device scatter for graph replay."""

    @staticmethod
    def device_memory_bytes(entries: int, k_bytes: int, v_bytes: int) -> int:
        # K/V payloads, two address vectors, and the combined gather/scatter address vector.
        return entries * (k_bytes + v_bytes + 4 * torch.int64.itemsize)

    def __init__(self, entries: int, k_bytes: int, v_bytes: int, device, coordinator=None):
        self.coordinator = coordinator
        self.enqueue = torch.ops._C_ascend.enqueue_selective_host_gather
        self.complete = torch.ops._C_ascend.enqueue_selective_host_gather_done
        self.empty_control = torch.empty(0, dtype=torch.int64)
        self.sources = torch.empty(2 * entries, dtype=torch.int64, pin_memory=True)
        self.destinations = torch.empty_like(self.sources, pin_memory=True)
        self.lengths = torch.empty(2 * entries, dtype=torch.int32, pin_memory=True)
        if coordinator is None:
            self.host = [
                torch.zeros((entries, size), dtype=torch.uint8, pin_memory=True) for size in (k_bytes, v_bytes)
            ]
        else:
            self.host = [
                coordinator.payload[: entries * k_bytes].view(entries, k_bytes),
                coordinator.payload[entries * k_bytes : entries * (k_bytes + v_bytes)].view(entries, v_bytes),
            ]
        self.device = [torch.empty_like(tensor, device=device) for tensor in self.host]
        offsets = torch.arange(entries, dtype=torch.int64, device=device)
        self.addresses = [offsets * size + tensor.data_ptr() for tensor, size in zip(self.device, (k_bytes, v_bytes))]
        self.packed_addresses = torch.empty(2 * entries, dtype=torch.int64, device=device)
        self.payloads: dict[tuple[int, bool], tuple[torch.Tensor, torch.Tensor]] = {}

    def copy(self, destination, source, kind: int) -> None:
        size = source.numel() * source.element_size()
        self.copy_bytes(destination.data_ptr(), source.data_ptr(), size, kind)

    def copy_bytes(self, destination: int, source: int, size: int, kind: int) -> None:
        if size == 0:
            return
        torch.ops._C_ascend.swap_blocks_batch(
            torch.tensor([source], dtype=torch.int64),
            torch.tensor([destination], dtype=torch.int64),
            torch.tensor([size], dtype=torch.int64),
            kind - 1,
        )

    def transfer(self, manager, sources, destinations, lengths, count, *, to_device: bool) -> None:
        entries = sources.numel() // 2
        if sources.numel() != 2 * entries or entries > self.host[0].shape[0]:
            raise ValueError("Selective SDMA descriptors exceed staging capacity")
        if lengths.dtype != torch.int32 or lengths.numel() != sources.numel():
            raise ValueError("Selective SDMA requires one int32 length per descriptor")
        key = (entries, to_device)
        if key not in self.payloads:
            if self.coordinator is None:
                self.payloads[key] = tuple(host[:entries] for host in self.host)
            else:
                shared = self.coordinator.payload
                k_size, v_size = self.host[0].shape[1], self.host[1].shape[1]
                self.payloads[key] = (
                    shared[: entries * k_size].view(entries, k_size),
                    shared[entries * k_size : entries * (k_size + v_size)].view(entries, v_size),
                )
        packed = self.packed_addresses[: 2 * entries]
        packed[:entries].copy_(self.addresses[0][:entries])
        packed[entries:].copy_(self.addresses[1][:entries])
        self.copy(self.lengths[: 2 * entries], lengths, 2)
        if to_device:
            self.copy(self.sources[: 2 * entries], sources, 2)
            self._launch(key)
            payload = self.payloads[key]
            for host, device in zip(payload, self.device):
                self.copy_bytes(device.data_ptr(), host.data_ptr(), entries * device.shape[1], 1)
            if self.coordinator is not None:
                self.complete(self.device[0], self.coordinator.control, self.coordinator.epoch, self.coordinator.rank)
            manager.copy_sfa_kv(packed, destinations, lengths, count)
        else:
            manager.copy_sfa_kv(sources, packed, lengths, count)
            self.copy(self.destinations[: 2 * entries], destinations, 2)
            for host, device in zip(self.host, self.device):
                self.copy(host[:entries], device[:entries], 2)
            self._launch(key)

    def _launch(self, key) -> None:
        entries, to_device = key
        coordinator = self.coordinator
        host_k, host_v = self.payloads[key]
        self.enqueue(
            self.device[0],
            self.sources[: 2 * entries],
            self.destinations[: 2 * entries],
            self.lengths[: 2 * entries],
            host_k,
            host_v,
            coordinator.control if coordinator is not None else self.empty_control,
            coordinator.epoch if coordinator is not None else self.empty_control,
            entries,
            host_k.shape[1],
            host_v.shape[1],
            coordinator.rank if coordinator is not None else 0,
            coordinator.world_size if coordinator is not None else 1,
            to_device,
        )

    def restore(self, manager, sources, destinations, lengths) -> None:
        """Expand rollback blocks into token rows and reuse the same Host gather/SDMA path."""
        source_rows = sources.cpu().tolist()
        target_rows = destinations.cpu().tolist()
        sizes = lengths.cpu().tolist()
        blocks = len(source_rows) // 2
        k_bytes, v_bytes = self.host[0].shape[1], self.host[1].shape[1]
        k_sources, v_sources, k_targets, v_targets = [], [], [], []
        for block in range(blocks):
            tokens, k_remainder = divmod(sizes[block], k_bytes)
            v_tokens, v_remainder = divmod(sizes[blocks + block], v_bytes)
            if k_remainder or v_remainder or tokens != v_tokens:
                raise ValueError("Rollback K/V lengths must contain the same whole token rows")
            for token in range(tokens):
                k_sources.append(source_rows[block] + token * k_bytes)
                v_sources.append(source_rows[blocks + block] + token * v_bytes)
                k_targets.append(target_rows[block] + token * k_bytes)
                v_targets.append(target_rows[blocks + block] + token * v_bytes)
        capacity = self.host[0].shape[0]
        for start in range(0, len(k_sources), capacity):
            end = min(start + capacity, len(k_sources))
            rows = end - start
            src = torch.tensor(k_sources[start:end] + v_sources[start:end], dtype=torch.int64, device=sources.device)
            dst = torch.tensor(k_targets[start:end] + v_targets[start:end], dtype=torch.int64, device=sources.device)
            size = torch.tensor([k_bytes] * rows + [v_bytes] * rows, dtype=torch.int32, device=sources.device)
            count = torch.tensor([2 * rows], dtype=torch.int32, device=sources.device)
            self.transfer(manager, src, dst, size, count, to_device=True)
            # Rollback runs outside capture; retain temporary descriptors until their copies finish.
            torch_npu.npu.current_stream().synchronize()


def allocate_kv_offload_topk_buffer_pair(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens

    topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
    num_kv_heads = 1  # sparse kv offload only support sfa(mla) now.
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    max_num_topk_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        # The backend imports this manager; defer its layout constant import.
        from vllm_ascend.attention.sfa_kv_offload import COPY_SFA_TAIL_BLOCKS

        # LIM merges all query selections into one request-owned hot row.
        # Graph padding uses a second, private arena of the same capacity.
        # Query-row metadata still needs max_num_seqs * decode_width, but
        # multiplying the large hot K/V allocation by that width wastes HBM.
        request_rows = 2 * copy_sfa_pool_capacity(vllm_config.scheduler_config.max_num_seqs)
        max_num_topk_rows = request_rows
        topk_buffer_size += COPY_SFA_TAIL_BLOCKS * vllm_config.cache_config.block_size
    if sparse_kv_offload_config.selective_offload and vllm_config.cache_config.block_size == 128:
        topk_buffer_size += vllm_config.cache_config.block_size
    topk_buffer_k_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * k_dim * torch.bfloat16.itemsize
    topk_buffer_v_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * v_dim * torch.bfloat16.itemsize
    # NOTE make sure to allocate k+v together and split them after allocate.
    # Refer to the comment in empty_aligned_int8_cpu_tensors for reason.
    topk_buffer_raw = torch.empty([topk_buffer_k_size_bytes + topk_buffer_v_size_bytes], dtype=torch.int8, device="npu")
    topk_buffer_k = (
        topk_buffer_raw[:topk_buffer_k_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, k_dim])
    )
    topk_buffer_v = (
        topk_buffer_raw[topk_buffer_k_size_bytes : topk_buffer_k_size_bytes + topk_buffer_v_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, v_dim])
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        # Dummy copy-SFA reads its private hot rows during capture/replay.
        topk_buffer_raw.zero_()
    return (topk_buffer_k, topk_buffer_v)


def allocate_kv_offload_topk_profile_buffers(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_offload_layers = sum(bool(getattr(spec, "store_on_host", False)) for spec in kv_cache_spec.values())
    if num_offload_layers == 0:
        raise RuntimeError("Sparse KV offload profile did not find any host-resident SFA KV cache layers.")

    buffers = [
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config) for _ in range(num_offload_layers)
    ]
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for buffer_pair in buffers for tensor in buffer_pair)
    logger.info_once(
        "Sparse KV offload reserved %.2f GiB of KV offload topk buffers across %d layers for profile run.",
        buffer_bytes / (1 << 30),
        num_offload_layers,
    )
    return buffers


def get_selective_sdma_device_memory_bytes(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> int:
    if not sparse_kv_offload_config.selective_offload:
        return 0
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens
    rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        rows = 2 * copy_sfa_pool_capacity(vllm_config.scheduler_config.max_num_seqs)
    model = vllm_config.model_config.hf_text_config
    k_bytes = model.kv_lora_rank * torch.bfloat16.itemsize
    v_bytes = model.qk_rope_head_dim * torch.bfloat16.itemsize
    memory_bytes = SelectiveSDMATransfer.device_memory_bytes(rows * sparse_kv_offload_config.topk, k_bytes, v_bytes)
    sizes = getattr(getattr(vllm_config, "compilation_config", None), "cudagraph_capture_sizes", None) or ()
    shared_layers = getattr(sparse_kv_offload_config, "indexer_types", ()).count("shared")
    memory_bytes += shared_layers * SELECTIVE_LIM_LAYER_FIELDS * torch.int64.itemsize * len(sizes)
    memory_bytes += shared_layers * rows * sparse_kv_offload_config.topk_buffer_size * torch.int32.itemsize
    if get_tensor_model_parallel_rank() == 0:
        block_size = vllm_config.cache_config.block_size
        memory_bytes += SelectiveSDMATransfer.device_memory_bytes(rows, block_size * k_bytes, block_size * v_bytes)
        memory_bytes += SelectiveSDMATransfer.device_memory_bytes(
            vllm_config.scheduler_config.max_num_batched_tokens, k_bytes, v_bytes
        )
    return memory_bytes


_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
# Worst-case 2 MiB-alignment waste per host layer (see empty_aligned_int8_cpu_tensors):
# 1x for rounding the raw buffer base address up to the alignment boundary, plus 1x tail
# padding for each of the two aligned tensors (k_cache_cpu, v_cache_cpu) whose sizes are
# rounded up to whole alignment chunks.
_CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER = 3 * _CPU_CACHE_ALIGNMENT
_VLLM_NULL_BLOCK_COUNT = 1


def empty_aligned_int8_cpu_tensors(
    sizes: list[int],
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> list[torch.Tensor]:
    """
    Allocate multiple int8 tensors with specified sizes,
    each aligned to specified alignment,
    and minimize the gap between each tensor's data_ptr.
    Keep paired K/V allocations aligned within the shared host pool.
    """
    chunk_nums = [cdiv(size, alignment) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = offload.empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
    base_addr = raw_tensor.data_ptr()
    if base_addr % alignment:
        base_addr = (base_addr // alignment + 1) * alignment
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset : base_offset + size])
        base_offset += chunk_num * alignment
    return allocate_tensors


@dataclass(frozen=True)
class SparseKVOffloadMemoryBudget:
    npu_limit_blocks: int
    dram_limit_blocks: int
    workload_limit_blocks: int
    final_num_blocks: int
    final_planner_bytes: int
    planned_host_bytes: int
    planned_device_bytes: int
    host_alignment_reserve_bytes: int
    limiting_factor: str


def _split_host_device_kv_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[list[KVCacheSpec], list[KVCacheSpec]]:
    if not kv_cache_spec:
        raise ValueError("Sparse KV offload requires a non-empty kv_cache_spec")
    host_specs: list[KVCacheSpec] = []
    device_specs: list[KVCacheSpec] = []
    for spec in kv_cache_spec.values():
        if getattr(spec, "store_on_host", False):
            host_specs.append(spec)
        else:
            device_specs.append(spec)
    if not host_specs:
        raise ValueError("Sparse KV offload requires at least one host KV cache spec")
    if not device_specs:
        raise ValueError("Sparse KV offload requires at least one device KV cache spec")
    block_sizes = {spec.block_size for spec in host_specs + device_specs}
    if len(block_sizes) != 1:
        raise ValueError(f"Sparse KV offload memory planning requires one shared block size, got {sorted(block_sizes)}")
    return host_specs, device_specs


def plan_sparse_kv_offload_memory(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    available_device_memory_bytes: int,
    dram_limit_bytes: int,
    keep_device_kv_cache: bool,
    selective_offload: bool = False,
) -> SparseKVOffloadMemoryBudget:
    """Bound sparse KV offload blocks by NPU, DRAM, and active demand."""
    host_specs, device_specs = _split_host_device_kv_specs(kv_cache_spec)
    host_page_size_bytes = sum(spec.page_size_bytes for spec in host_specs)
    device_page_size_bytes = sum(spec.page_size_bytes for spec in device_specs)
    total_page_size_bytes = host_page_size_bytes + device_page_size_bytes

    host_alignment_reserve_bytes = len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    usable_dram_bytes = max(dram_limit_bytes - host_alignment_reserve_bytes, 0)
    dram_limit_blocks = usable_dram_bytes // host_page_size_bytes

    npu_page_size_bytes = total_page_size_bytes if keep_device_kv_cache else device_page_size_bytes
    npu_limit_blocks = max(available_device_memory_bytes, 0) // npu_page_size_bytes

    max_blocks_per_request = max(
        cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
        for spec in host_specs + device_specs
    )
    workload_limit_blocks = max_blocks_per_request * vllm_config.scheduler_config.max_num_seqs + _VLLM_NULL_BLOCK_COUNT
    planner_page_bytes = total_page_size_bytes
    if selective_offload:
        from vllm_ascend.utils import vllm_version_is

        if vllm_version_is("0.32.0"):
            from vllm.v1.core import kv_cache_utils as planning
        else:
            from vllm.v1.core import kv_cache_planning as planning
        groups = planning.get_kv_cache_groups(vllm_config, kv_cache_spec)
        # Groups claim distinct block IDs from one pool, including Host groups.
        pages = sum(
            group.kv_cache_spec.max_memory_usage_pages(vllm_config)
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), group.kv_cache_spec.page_size_bytes)
            for group in groups
        )
        workload_limit_blocks = pages * vllm_config.scheduler_config.max_num_seqs + _VLLM_NULL_BLOCK_COUNT
        planner_page_bytes = planning._pool_bytes_per_block(vllm_config, groups)

    limits = {
        "npu": npu_limit_blocks,
        "dram": dram_limit_blocks,
        "workload": workload_limit_blocks,
    }
    if dram_limit_blocks < npu_limit_blocks:
        logger.warning_once(
            "Sparse KV offload: host DRAM budget allows only %d KV blocks while NPU "
            "memory could hold %d, so batch size / max_model_len capacity is limited "
            "by host memory. Consider increasing sparse_kv_offload_config."
            "dram_size_per_dp_GB (current budget %.2f GiB) to use more of the NPU "
            "capacity.",
            dram_limit_blocks,
            npu_limit_blocks,
            dram_limit_bytes / (1 << 30),
        )
    limiting_factor = min(limits, key=lambda name: limits[name])
    final_num_blocks = limits[limiting_factor]
    final_planner_bytes = final_num_blocks * planner_page_bytes
    planned_host_bytes = final_num_blocks * host_page_size_bytes
    planned_device_bytes = final_num_blocks * npu_page_size_bytes
    return SparseKVOffloadMemoryBudget(
        npu_limit_blocks=npu_limit_blocks,
        dram_limit_blocks=dram_limit_blocks,
        workload_limit_blocks=workload_limit_blocks,
        final_num_blocks=final_num_blocks,
        final_planner_bytes=final_planner_bytes,
        planned_host_bytes=planned_host_bytes,
        planned_device_bytes=planned_device_bytes,
        host_alignment_reserve_bytes=host_alignment_reserve_bytes,
        limiting_factor=limiting_factor,
    )


def get_sparse_kv_offload_cpu_pool_size_bytes(
    kv_cache_config: KVCacheConfig,
) -> int:
    """Return a safe upper bound for aligned host KV allocations."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            layer_specs.update(group.kv_cache_spec.kv_cache_specs)
        else:
            layer_specs.update((layer_name, group.kv_cache_spec) for layer_name in group.layer_names)
    host_specs = [spec for spec in layer_specs.values() if getattr(spec, "store_on_host", False)]
    if not host_specs:
        raise ValueError("Sparse KV offload requires host-resident KV cache specs")
    raw_host_bytes = kv_cache_config.num_blocks * sum(spec.page_size_bytes for spec in host_specs)
    alignment_reserve_bytes = len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return raw_host_bytes + alignment_reserve_bytes


def allocate_kv_cache_tensors_for_sparse_kv_offload(
    k_tensor_size: int,
    v_tensor_size: int,
    alignment: int,
    tp_rank: int,
    keep_device_kv_cache: bool,
    npu_kv_cache_allocate_func: typing.Callable,
):
    if tp_rank == 0:
        [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors(
            [k_tensor_size, v_tensor_size],
            alignment,
        )
    else:
        k_tensor_cpu = None
        v_tensor_cpu = None

    if keep_device_kv_cache:
        k_tensor = npu_kv_cache_allocate_func(
            k_tensor_size,
            alignment,
        )
        v_tensor = npu_kv_cache_allocate_func(
            v_tensor_size,
            alignment,
        )
    else:
        k_tensor = None
        v_tensor = None

    return (k_tensor, v_tensor, k_tensor_cpu, v_tensor_cpu, k_tensor_size + v_tensor_size)


def reshape_kv_cache_tensors_for_sparse_kv_offload(
    raw_cache_tensors: tuple,
    current_kv_cache_spec: AttentionSpec,
    attn_backend: type[AttentionBackend],
    tp_rank: int,
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    raw_k_tensor, raw_v_tensor, raw_k_tensor_cpu, raw_v_tensor_cpu, sum_page_size_bytes = raw_cache_tensors
    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        current_kv_cache_spec.block_size,
        current_kv_cache_spec.num_kv_heads,
        current_kv_cache_spec.head_size,
    )
    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    k_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        k_dim,
    )
    v_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        v_dim,
    )
    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype

    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape) if raw_k_tensor is not None else None
    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape) if raw_v_tensor is not None else None

    if tp_rank == 0:
        k_cache_cpu = raw_k_tensor_cpu.view(k_cache_dtype).view(k_shape)
        v_cache_cpu = raw_v_tensor_cpu.view(v_cache_dtype).view(v_shape)
    else:
        k_cache_cpu = None
        v_cache_cpu = None

    topk_buffer_k, topk_buffer_v = allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
    return (k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v)


def update_sparse_kv_offload_metadata(
    num_tokens: int,
    num_reqs: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    req_ids: list[str],
    query_start_loc: CpuGpuBuffer | torch.Tensor | np.ndarray,
    offload_req_ids_tensor: CpuGpuBuffer,
    offload_token_to_req: CpuGpuBuffer,
    request_instance_ids: list[int] | None = None,
) -> None:
    """Populate per-request identity tensors for the Sparse KV offload LRU.

    Callers may supply unique request instance IDs; otherwise use the legacy
    scheduler ID hash. Rows are reset when the occupying request changes.
    """
    offload_req_ids_tensor.np[:num_reqs_padded].fill(0)
    effective_num_reqs = min(num_reqs, len(req_ids))
    if request_instance_ids is not None:
        if len(request_instance_ids) != effective_num_reqs or any(uid <= 0 for uid in request_instance_ids):
            raise ValueError("Request instance IDs must contain one positive ID per active request")
        req_id_values = np.asarray(request_instance_ids, dtype=np.int64)
    else:
        req_id_values = np.asarray(
            [
                adler32(req_id.encode("utf-8")) if isinstance(req_id, str) else row + 1
                for row, req_id in enumerate(req_ids[:effective_num_reqs])
            ],
            dtype=np.int64,
        )
    offload_req_ids_tensor.np[:effective_num_reqs] = req_id_values
    offload_req_ids_tensor.copy_to_gpu(num_reqs_padded)

    if isinstance(query_start_loc, CpuGpuBuffer):
        query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1].numpy()
    elif isinstance(query_start_loc, torch.Tensor):
        if query_start_loc.device.type != "cpu":
            raise ValueError("Sparse KV offload query boundaries must have a CPU mirror")
        query_start_loc_cpu = query_start_loc[: num_reqs + 1].numpy()
    else:
        query_start_loc_cpu = query_start_loc[: num_reqs + 1]
    query_lens = np.diff(query_start_loc_cpu).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        raise RuntimeError(
            "KV offload token_to_req metadata is shorter than the scheduled token batch: "
            f"metadata={token_to_req.shape[0]}, tokens={num_tokens}"
        )
    offload_token_to_req.np[:num_tokens] = token_to_req[:num_tokens]
    if num_tokens_padded > num_tokens:
        offload_token_to_req.np[num_tokens:num_tokens_padded].fill(0)
    offload_token_to_req.copy_to_gpu(num_tokens_padded)


def prepare_sparse_kv_offload_mtp_dummy_metadata(
    num_tokens: int,
    num_reqs: int,
    query_start_loc_cpu: torch.Tensor,
    req_ids_buffer: CpuGpuBuffer,
    token_to_req_buffer: CpuGpuBuffer,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not get_ascend_config().sparse_kv_offload_config.enabled:
        return None, None
    if req_ids_buffer is None or token_to_req_buffer is None:
        raise RuntimeError("Sparse KV offload metadata buffers are not initialized")

    query_lens = np.diff(query_start_loc_cpu[: num_reqs + 1].numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        token_to_req = np.pad(token_to_req, (0, num_tokens - token_to_req.shape[0]))

    req_ids_buffer.np[:num_reqs] = np.arange(1, num_reqs + 1, dtype=np.int64)
    req_ids_buffer.copy_to_gpu(num_reqs)
    token_to_req_buffer.np[:num_tokens] = token_to_req[:num_tokens]
    token_to_req_buffer.copy_to_gpu(num_tokens)
    return (
        req_ids_buffer.gpu[:num_reqs],
        token_to_req_buffer.gpu[:num_tokens],
    )


class SparseKVOffloadManager:
    """
    A manager responsible to the Sparse KV cache Offload.
    It enlarge the available memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sparse_kv_offload_config: SparseKVOffloadConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.sparse_kv_offload_config = sparse_kv_offload_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
        self.topk = sparse_kv_offload_config.topk
        self.use_fused_overlap = sparse_kv_offload_config.use_fused_overlap
        self.use_fused_copy_sfa = sparse_kv_offload_config.use_fused_copy_sfa

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        self.fused_overlap_membership_map: torch.Tensor | None = None
        self.fused_overlap_membership_map_rows = 0
        self.fused_overlap_plan_owner_layer_id: int | None = None
        self.fused_overlap_plan_topk: int | None = None
        self.fused_overlap_plan_num_tokens = 0
        self.fused_overlap_plan_membership_map: torch.Tensor | None = None
        if not getattr(sparse_kv_offload_config, "selective_offload", False):
            _sparse_kv_ops()

        dram_limit_bytes = int(sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30))
        planned_pool_size_bytes = get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)
        if planned_pool_size_bytes > dram_limit_bytes:
            raise ValueError(
                "Sparse KV offload planned CPU pool exceeds DRAM limit after "
                "alignment: "
                f"planned={planned_pool_size_bytes / (1 << 30):.2f} GiB, "
                f"limit={sparse_kv_offload_config.dram_size_per_dp_GB} GiB, "
                f"num_blocks={kv_cache_config.num_blocks}"
            )
        actual_pool_size_bytes = min(planned_pool_size_bytes, dram_limit_bytes)
        logger.info(
            "SparseKVOffloadManager starts CPU KV pool initialization: "
            "planned=%.2f GiB, configured_limit=%s GiB, num_blocks=%s.",
            actual_pool_size_bytes / (1 << 30),
            sparse_kv_offload_config.dram_size_per_dp_GB,
            kv_cache_config.num_blocks,
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.reserve_size = actual_pool_size_bytes
        config.alloc_size = actual_pool_size_bytes if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = offload.Scene.SHARED
        store_port = OFFLOAD_STORE_PORT_BASE + parallel_config.data_parallel_index
        config.store_url = f"tcp://127.0.0.1:{store_port}"
        assert offload.initialize(config) == 0, "Sparse KV offload offload.initialize failed."
        self.tp_group.barrier()

    def _warmup_external_lru_planner_threads(self) -> int:
        if not (self.use_fused_overlap and self.tp_rank == 0):
            return 0
        warmed_threads = _sparse_kv_ops().sparse_kv_warmup_lru_resident_threads(self.lru_workspace_threads)
        logger.info(
            "Warmed external LRU planner OpenMP team with %s threads",
            warmed_threads,
        )
        return warmed_threads

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> int:
        block_sizes = {group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups}
        if len(block_sizes) != 1:
            raise ValueError("Sparse KV offload requires one shared block size across target and draft groups.")
        return block_sizes.pop()

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        host_layer_names = set()
        main_layer_names = set()
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if "indexer" not in layer_name.lower():
                    main_layer_names.add(layer_name)
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[layer_name]
                if getattr(spec, "store_on_host", False):
                    host_layer_names.add(layer_name)
        if getattr(self.sparse_kv_offload_config, "selective_offload", False):
            expected = {name for name in main_layer_names if self.sparse_kv_offload_config.is_host_layer(name)}
            if host_layer_names != expected:
                raise ValueError(
                    "Selective KV offload cache placement disagrees with shared layer roles: "
                    f"missing_host={sorted(expected - host_layer_names)}, "
                    f"unexpected_host={sorted(host_layer_names - expected)}"
                )
            missing = expected - kv_caches.keys()
            if missing:
                raise ValueError(f"Selective KV offload shared cache tensors are missing: {sorted(missing)}")
        self.offload_layer_names = [layer_name for layer_name in kv_caches if layer_name in host_layer_names]
        if not self.offload_layer_names:
            raise ValueError("Sparse KV offload did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info_once(
            "Sparse KV offload registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        self.mtp_layer_id = (
            -1
            if getattr(self.sparse_kv_offload_config, "selective_offload", False)
            else self.num_layers - 1
            if self.num_layers != self.num_target_layers
            else -1
        )
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("Sparse KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "Sparse KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def _restore_bfloat16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = _sparse_kv_ops().sparse_kv_restore_bfloat16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_bfloat16_tensor returned a tensor with unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_bfloat16_tensor returned unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.bfloat16:
            raise RuntimeError(f"restore_bfloat16_tensor returned unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_bfloat16_tensor requires a contiguous view")
        return view

    def _restore_int16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = _sparse_kv_ops().sparse_kv_restore_int16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_int16_tensor returned a tensor with an unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_int16_tensor returned an unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.int16:
            raise RuntimeError(f"restore_int16_tensor returned an unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_int16_tensor requires a contiguous view")
        return view

    def allocate_fused_overlap_membership_map(
        self,
        row_capacity: int,
    ) -> torch.Tensor:
        if not self.use_fused_overlap:
            raise RuntimeError("mapped membership allocation requires kv_offload_decode_config.use_fused_overlap=true")
        if row_capacity <= 0:
            raise ValueError(f"mapped membership row capacity must be positive, got {row_capacity}")
        if self.topk >= FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT:
            raise ValueError(
                "SFA sparse topk exceeds external plan storage: "
                f"topk={self.topk} "
                f"control_offset={FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT}"
            )

        if getattr(self, "fused_overlap_membership_map", None) is not None:
            if row_capacity > self.fused_overlap_membership_map_rows:
                raise RuntimeError(
                    "fused_overlap membership storage was allocated for fewer "
                    f"rows: allocated={self.fused_overlap_membership_map_rows}, "
                    f"required={row_capacity}"
                )
            return self.fused_overlap_membership_map

        shape = [row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT]
        owner_ptr = torch.zeros(1, dtype=torch.int64, device="npu")
        membership_map = None
        if self.tp_rank == 0:
            membership_map = offload.empty(
                [row_capacity * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT],
                dtype=torch.int16,
                pin_memory=True,
            ).view(shape)
            membership_map.fill_(-1)
            control = membership_map[
                :,
                FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
                + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
            ]
            control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
            control[:, 2] = self.topk
            control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
            control[:, 7] = FSA_PAIRED_SELECTION_COPY_MARKER
            owner_ptr[0] = membership_map.data_ptr()
        self.tp_group.broadcast(owner_ptr, src=0)
        shared_ptr = int(owner_ptr.item())
        if self.tp_rank != 0:
            membership_map = self._restore_int16_tensor(shared_ptr, shape)
        if membership_map is None:
            raise RuntimeError("mapped membership storage was not initialized")
        self.tp_group.barrier()
        self.fused_overlap_membership_map = membership_map
        self.fused_overlap_membership_map_rows = row_capacity
        return membership_map

    def get_fused_overlap_cpu_kv_inputs(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU full KV tensors used by fused_overlap decode.

        On TP0 these are the owned CPU pools; on other ranks they are non-owning
        GVA views restored after broadcast.
        """
        if not self.use_fused_overlap:
            raise RuntimeError(
                "get_fused_overlap_cpu_kv_inputs requires kv_offload_decode_config.use_fused_overlap=true"
            )
        layer_id = self._get_offload_layer_id(layer_name)
        if layer_id >= len(self.k_caches_cpu) or layer_id >= len(self.v_caches_cpu):
            raise RuntimeError(
                "fused_overlap CPU KV views are not registered: "
                f"layer_id={layer_id}, k_len={len(self.k_caches_cpu)}, "
                f"v_len={len(self.v_caches_cpu)}"
            )
        return self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        self._register_offload_layers(kv_caches)
        selective = getattr(getattr(self, "sparse_kv_offload_config", None), "selective_offload", False)

        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"Sparse KV offload layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.tp_rank == 0:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "Sparse KV offload only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "Sparse KV offload requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if selective:
            device = self.topk_buffers_k[0].device
            rows = self.topk_buffers_k[0].shape[0]
            self.selective_lim_capture_buffers: list = []
            # MemFabric queues raw descriptor pointers, so keep their storage alive across replay.
            descriptor_shape = (self.num_layers, 2 * rows * self.topk)
            self.selective_copy_src = torch.empty(descriptor_shape, dtype=torch.int64, device=device)
            self.selective_copy_dst = torch.empty_like(self.selective_copy_src)
            self.selective_copy_lengths = torch.empty(descriptor_shape, dtype=torch.int32, device=device)
            self.selective_copy_counts = torch.arange(1, rows + 1, dtype=torch.int32, device=device) * (2 * self.topk)
            self.selective_physical = torch.empty(
                (self.num_layers, rows, 1, self.topk), dtype=torch.int32, device=device
            )
            self.selective_prefetch_topk = torch.empty((rows, self.topk), dtype=torch.int64, device=device)
            self.selective_prefetch_stream = torch_npu.npu.Stream(device=device)
            self.selective_prefetch_ready = [torch_npu.npu.Event() for _ in self.offload_layer_names]
            host_layers = {extract_layer_index(name): name for name in self.offload_layer_names}
            self.selective_followers: dict[int, list[str]] = {}
            leader = 0
            for index, role in enumerate(self.sparse_kv_offload_config.indexer_types):
                if role == "full":
                    leader = index
                else:
                    self.selective_followers.setdefault(leader, []).append(host_layers[index])
            self.selective_group_tails = {followers[-1] for followers in self.selective_followers.values()}
            self.selective_writeback_fence = torch.zeros(1, dtype=torch.int32, device=device)
            self.selective_row_stride = self.topk_buffers_k[0].shape[1]
            row_blocks = self.selective_row_stride // self.block_size
            hot_blocks = self.topk_buffer_size // self.block_size
            self.selective_hot_block_table = (
                torch.arange(rows * row_blocks, dtype=torch.int32, device=device)
                .view(rows, row_blocks)[:, :hot_blocks]
                .contiguous()
            )
            self.selective_tail_len = torch.full((rows,), self.selective_row_stride, dtype=torch.int32, device=device)
            self.selective_hot_keys = [
                torch.full((rows, self.topk_buffer_size + 1), -1, dtype=torch.int64, device=device)
                for _ in self.offload_layer_names
            ]
            self.selective_hot_requests = [
                torch.full((rows, 1), -1, dtype=torch.int64, device=device) for _ in self.offload_layer_names
            ]
            self.selective_lru_order = [
                torch.empty((rows, self.topk_buffer_size), dtype=torch.int32, device=device)
                for _ in self.offload_layer_names
            ]
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "Sparse KV offload topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H descriptors commit newly produced KV into the shared host pool.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_dst_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_lengths_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int32, device=device)
        if self.use_fused_overlap or (selective and self.tp_rank == 0):
            self.current_kv_save_stream = torch_npu.npu.Stream(device=device)
        if self.use_fused_overlap:
            self.fused_plan_stream = torch_npu.npu.Stream()
            self.fused_plan_metadata_npu = torch.zeros(
                1 + self.max_num_topk_rows,
                dtype=torch.int32,
                device=device,
            )
            self.fused_plan_status_npu = self.fused_plan_metadata_npu[:1]
            self.fused_plan_current_linear_slots_npu = self.fused_plan_metadata_npu[1:]
        self.current_kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(self.max_num_tokens, dtype=torch.int64, device=device)

        # Broadcast the shared host KV pool bases and shapes.
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        # RD2H consumers need byte strides for each TP-shared host KV block.
        self.cpu_block_lens: list[tuple[int, int]] = []
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        cpu_block_lens_tensor = torch.zeros([self.num_layers, 2], dtype=torch.int64, device="npu")
        shape_k_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        shape_v_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                cpu_block_lens_tensor[layer_id, 0] = (
                    k_cpu.numel() * k_cpu.element_size() // self.kv_cache_config.num_blocks
                )
                cpu_block_lens_tensor[layer_id, 1] = (
                    v_cpu.numel() * v_cpu.element_size() // self.kv_cache_config.num_blocks
                )
            shape_k_tensor.copy_(torch.tensor(self.k_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
            shape_v_tensor.copy_(torch.tensor(self.v_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
        self.tp_group.broadcast(shape_k_tensor, src=0)
        self.tp_group.broadcast(shape_v_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())

            self.cpu_block_lens.append(
                (
                    cpu_block_lens_tensor[layer_id, 0].item(),
                    cpu_block_lens_tensor[layer_id, 1].item(),
                )
            )

        if self.tp_rank != 0 and not selective:
            cpu_k_shape = [int(x) for x in shape_k_tensor.tolist()]
            cpu_v_shape = [int(x) for x in shape_v_tensor.tolist()]
            self.k_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_k_shape) for ptr in self.gvas_k_bases]
            self.v_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_v_shape) for ptr in self.gvas_v_bases]
            logger.info(
                "[fused_overlap_offload][init] restored shared CPU KV views on "
                "tp_rank=%s layer_count=%s k_shape=%s v_shape=%s",
                self.tp_rank,
                self.num_layers,
                cpu_k_shape,
                cpu_v_shape,
            )
        if self.use_fused_copy_sfa:
            self._bind_copy_sfa_bases()

        self.selective_sdma_prefetch = None
        self.selective_sdma_tail = None
        self.selective_sdma_prefill = None
        if selective:
            rows = self.topk_buffers_k[0].shape[0]
            self.selective_host_gather = HostGatherCoordinator(
                self.tp_group,
                self.tp_rank,
                self.tp_size,
                rows * self.topk * (self.token_size_bytes_k + self.token_size_bytes_v),
            )
            self.selective_sdma_prefetch = SelectiveSDMATransfer(
                rows * self.topk,
                self.token_size_bytes_k,
                self.token_size_bytes_v,
                device,
                coordinator=self.selective_host_gather,
            )
            if self.tp_rank == 0:
                self.selective_sdma_tail = SelectiveSDMATransfer(
                    rows,
                    self.block_size * self.token_size_bytes_k,
                    self.block_size * self.token_size_bytes_v,
                    device,
                )
                self.selective_sdma_prefill = SelectiveSDMATransfer(
                    self.max_num_tokens, self.token_size_bytes_k, self.token_size_bytes_v, device
                )
            logger.info("Selective KV uses Host gather, SDMA H2D/D2H, and device-local gather/scatter")

        if self.use_fused_overlap:
            self._init_fused_overlap_planner()

    def _init_fused_overlap_planner(self) -> None:
        """Allocate the CPU LRU workspace used by the overlap operator."""
        self.lru_workspace_threads = 8
        self._warmup_external_lru_planner_threads()
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [
            torch.full(
                [self.max_num_topk_rows, self.topk_buffer_size],
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_slots_cpu_list = [
            torch.arange(
                self.topk_buffer_size,
                dtype=torch.int32,
                device="cpu",
            )
            .view(1, -1)
            .repeat(self.max_num_topk_rows, 1)
            .pin_memory()
            for _ in range(self.num_layers)
        ]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_tokens_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_slots_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device="cpu", pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [
            torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_visible_seq_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_physical_row_workspace = torch.empty(
            [self.max_num_topk_rows * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [
            lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list
        ]
        self.lru_visible_seq_lens_ptr = self.lru_visible_seq_lens_cpu.data_ptr()
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [
            lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list
        ]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [
            lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list
        ]
        self.lru_miss_tokens_ptrs = [
            lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list
        ]
        self.lru_miss_slots_ptrs = [
            lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list
        ]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()
        self.lru_physical_row_workspace_ptr = self.lru_physical_row_workspace.data_ptr()

    def _bind_copy_sfa_bases(self) -> None:
        """Pin host/device row bases used by eager prefix-rollback tail restore."""
        if not self.k_caches_cpu or not self.topk_buffers_k:
            raise RuntimeError("fused_copy_sfa tail restore requires host and device KV bases")
        device = self.topk_buffers_k[0].device
        descriptor_rows = 4 * (self.max_num_reqs + 2)
        self.copy_sfa_copy_src = torch.empty(descriptor_rows, dtype=torch.int64, device=device)
        self.copy_sfa_copy_dst = torch.empty_like(self.copy_sfa_copy_src)
        self.copy_sfa_host_bases = []
        self.copy_sfa_device_bases = []
        for layer_id in range(len(self.topk_buffers_k)):
            self.copy_sfa_host_bases.append(
                torch.tensor(
                    [self.k_caches_cpu[layer_id].data_ptr(), self.v_caches_cpu[layer_id].data_ptr()],
                    dtype=torch.int64,
                    device=device,
                ).view(2, 1)
            )
            self.copy_sfa_device_bases.append(
                torch.tensor(
                    [self.topk_buffers_k[layer_id].data_ptr(), self.topk_buffers_v[layer_id].data_ptr()],
                    dtype=torch.int64,
                    device=device,
                ).view(2, 1)
            )

    def dense_fill_copy_sfa_rows(
        self,
        dense_fills: dict[int, tuple[int, int]],
        *,
        block_size: int,
        block_table: np.ndarray | torch.Tensor,
    ) -> None:
        """Restore whole short rows from the host pool after prefix rollback.

        A rollback across the hot boundary leaves a sparse-layout row under a
        dense (-3) reader. Restore it using the input batch's block table
        (a CPU mirror in V1 or a device tensor in V2) and block size, outside
        graph capture at metadata-build time. Host and device bases are bound
        during cache registration.
        """
        topk_k = self.topk_buffers_k[0]
        stride_tokens = topk_k.shape[1]
        token_bytes = torch.tensor(
            [
                [topk_k.element_size() * topk_k.shape[-1]],
                [self.topk_buffers_v[0].element_size() * self.topk_buffers_v[0].shape[-1]],
            ],
            dtype=torch.int64,
            device=topk_k.device,
        )
        block_bytes = token_bytes * block_size
        for slot, (row, computed) in dense_fills.items():
            nblocks = (computed + block_size - 1) // block_size
            if isinstance(block_table, torch.Tensor):
                if block_table.device != topk_k.device:
                    raise ValueError("Tensor block table must be on the same device as the top-k buffers")
                # block_bytes is a dimensioned int64 tensor, not a Python scalar.
                # The product is int64 before adding the host base, including offsets above 2 GiB.
                src_offsets = block_table[row, :nblocks].view(1, -1) * block_bytes
            else:
                src_tokens = torch.tensor(
                    [int(block_table[row, b]) * block_size for b in range(nblocks)], dtype=torch.int64
                ).to(topk_k.device)
                src_offsets = src_tokens.view(1, -1) * token_bytes
            lengths = torch.clamp(
                torch.arange(nblocks, dtype=torch.int64, device=topk_k.device) * (-block_size) + computed,
                min=0,
                max=block_size,
            )
            dst_block_tokens = torch.arange(nblocks, dtype=torch.int64, device=topk_k.device) * block_size
            dst_offsets = (slot * stride_tokens + dst_block_tokens.view(1, -1).expand(2, -1)) * token_bytes
            lengths_bytes = lengths.view(1, -1).expand(2, -1) * token_bytes
            count = torch.full((1,), 2 * nblocks, dtype=torch.int32, device=topk_k.device)
            for host_bases, device_bases in zip(self.copy_sfa_host_bases, self.copy_sfa_device_bases):
                sources = (src_offsets + host_bases).reshape(-1)
                destinations = (dst_offsets + device_bases).reshape(-1)
                if self.selective_sdma_prefetch is not None:
                    self.selective_sdma_prefetch.restore(self, sources, destinations, lengths_bytes.reshape(-1))
                else:
                    self.copy_sfa_kv(sources, destinations, lengths_bytes.reshape(-1), count)

    def restore_copy_sfa_tails(self, metadata) -> None:
        """Copy the current tail descriptors for every layer. Used on prefix rollback."""
        src_off = getattr(metadata, "copy_sfa_copy_src_offsets", None)
        dst_off = getattr(metadata, "copy_sfa_copy_dst_offsets", None)
        lengths = getattr(metadata, "copy_sfa_copy_lengths", None)
        count = getattr(metadata, "copy_sfa_copy_count", None)
        if src_off is None or dst_off is None or lengths is None or count is None:
            return
        if not getattr(self, "copy_sfa_host_bases", None):
            raise RuntimeError("fused_copy_sfa KV base addresses must be bound before tail restore")
        n = src_off.numel()
        for host_bases, device_bases in zip(self.copy_sfa_host_bases, self.copy_sfa_device_bases):
            torch.add(src_off.view(2, -1), host_bases, out=self.copy_sfa_copy_src[:n].view(2, -1))
            torch.add(dst_off.view(2, -1), device_bases, out=self.copy_sfa_copy_dst[:n].view(2, -1))
            if self.selective_sdma_prefetch is not None:
                self.selective_sdma_prefetch.restore(
                    self, self.copy_sfa_copy_src[:n], self.copy_sfa_copy_dst[:n], lengths
                )
            else:
                self.copy_sfa_kv(self.copy_sfa_copy_src[:n], self.copy_sfa_copy_dst[:n], lengths, count)

    def prepare_selective_kv(self, layers, topk_indices) -> None:
        if getattr(self, "_selective_native_experiment", False):
            if self._selective_native_failed:
                raise RuntimeError("Native selective experiment is quarantined after a graph failure")
            if torch_npu.npu.is_current_stream_capturing():
                self._prepare_selective_native_kv(layers, topk_indices)
                return
            if getattr(self, "_selective_native_started", False):
                raise RuntimeError("Native selective experiment cannot mix eager and captured hot-buffer paths")
        topk = topk_indices.reshape(-1, self.topk).to(torch.int64).contiguous()
        rows, width = topk.shape
        if not 0 < width <= self.topk_buffer_size <= 2048:
            raise ValueError("Precomputed-TopK LIM requires 0 < TopK <= hot capacity <= 2048")
        entries = 2 * rows * width
        descriptors = []
        inputs = []
        for layer_id, metadata, seq_lengths in layers:
            if metadata.req_ids_tensor is None or metadata.selective_tail_rows is None:
                raise RuntimeError("Precomputed-TopK LIM requires request identities and selective tail metadata")
            table = metadata.block_table[:rows].to(torch.int32).contiguous()
            tensors = (
                table,
                seq_lengths[:rows].to(torch.int32).contiguous(),
                metadata.req_ids_tensor[:rows].to(torch.int64).contiguous(),
                metadata.selective_tail_rows[:rows].to(torch.int32).contiguous(),
                self.selective_hot_keys[layer_id],
                self.selective_hot_requests[layer_id],
                self.selective_physical[layer_id],
                self.selective_copy_src[layer_id, :entries],
                self.selective_copy_dst[layer_id, :entries],
                self.selective_copy_lengths[layer_id, :entries],
            )
            lru_order = self.selective_lru_order[layer_id]
            inputs.append((*tensors, lru_order))
            descriptors.append(
                [tensor.data_ptr() for tensor in tensors]
                + [
                    self.gvas_k_bases[layer_id],
                    self.gvas_v_bases[layer_id],
                    self.topk_buffers_k[layer_id].data_ptr(),
                    self.topk_buffers_v[layer_id].data_ptr(),
                    table.shape[1],
                    lru_order.data_ptr(),
                ]
            )
        host = torch.tensor(descriptors, dtype=torch.int64, pin_memory=True)
        device = host.to(topk.device, non_blocking=True)
        if torch_npu.npu.is_current_stream_capturing():
            # Captured H2D keeps the same Host table and its pointed-to tensors on every replay.
            self.selective_lim_capture_buffers.append((host, device, inputs))
        torch.ops._C_ascend.npu_precomputed_topk_manage(
            topk,
            device,
            self.kv_cache_config.num_blocks,
            self.block_size,
            self.topk_buffer_size,
            self.selective_row_stride,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
        )
        logger.info_once("Selective KV offload launched the native precomputed-TopK LIM kernel (grouped).")
        for layer_id, _, _ in layers:
            self.prefetch_selective_kv(
                rows,
                self.selective_copy_src[layer_id, :entries],
                self.selective_copy_dst[layer_id, :entries],
                self.selective_copy_lengths[layer_id, :entries],
            )
            self.selective_prefetch_ready[layer_id].record()

    def enable_selective_native_experiment(self) -> None:
        """Enable the native group path for an isolated MRV2 integration test."""
        model_state = getattr(self, "_selective_model_state", None)
        local_error = None
        try:
            if self.topk != self.topk_buffer_size or self.use_fused_copy_sfa:
                raise RuntimeError("Native selective integration probe requires TopK=hot capacity")
            if self.max_model_len >= 1 << 31:
                raise RuntimeError("Native selective integration probe requires int32 token indices")
            if model_state is not None and (model_state._selective_request_ids or model_state._offload_live_req_ids):
                raise RuntimeError("Native selective experiment must be enabled before admitting requests")
            max_rows = self.topk_buffers_k[0].shape[0]
            device = self.topk_buffers_k[0].device
            self._selective_native_rows = torch.arange(max_rows, dtype=torch.int64, device=device)
            self._selective_native_offsets = torch.arange(max_rows * self.topk, dtype=torch.int64, device=device)
        except Exception as error:
            local_error = str(error)
        if self.tp_size > 1:
            gathered = [None] * self.tp_size
            torch.distributed.all_gather_object(gathered, local_error, group=self.tp_group.cpu_group)
            if any(value is not None for value in gathered):
                raise RuntimeError(f"Native selective TP activation failed: {gathered}")
        elif local_error is not None:
            raise RuntimeError(local_error)
        self._selective_native_experiment = True
        self._selective_native_groups = {}
        self._selective_native_graphs = {}
        self._selective_native_graph_layers = {}
        self._selective_native_graph_keys = {}
        self._selective_native_epoch = 0
        self._selective_native_started = False
        self._selective_native_failed = False
        self._selective_native_inflight = False
        self._selective_native_stopping = False
        self._selective_native_tp_groups = {}
        if self.tp_size > 1:
            try:
                signature_error = None
                groups, signature = [], ()
                try:
                    groups = [(leader, tuple(names)) for leader, names in sorted(self.selective_followers.items())
                              if names]
                    signature = tuple((leader, tuple((extract_layer_index(name), name) for name in names))
                                      for leader, names in groups)
                except Exception as error:
                    signature_error = str(error)
                use_shared_exchange = getattr(self, "_selective_native_shared_exchange_requested", False)
                signature = (signature, use_shared_exchange)
                gathered = [None] * self.tp_size
                torch.distributed.all_gather_object(gathered, (signature_error, signature), group=self.tp_group.cpu_group)
                if any(value[0] is not None or value[1] != signature for value in gathered):
                    raise RuntimeError("Native selective TP group definitions differ across ranks")
                for leader, names in groups:
                    layer_ids = tuple(self._get_offload_layer_id(name) for name in names)
                    if not layer_ids:
                        continue
                    stable_ids = [leader, *(extract_layer_index(name) for name in names)]
                    coordinator = SelectiveNativeTPCoordinator(
                        self.tp_group, self.tp_rank, self.tp_size, max_rows, self.topk, len(layer_ids),
                        self.block_size, self.token_size_bytes_k, self.token_size_bytes_v, stable_ids,
                        shared_exchange=use_shared_exchange and not self._selective_native_tp_groups,
                    )
                    self._selective_native_tp_groups[layer_ids] = coordinator
                    logger.info("Native selective TP group %s shared pinned region: %.2f MiB",
                                layer_ids, coordinator.total_bytes / (1 << 20))
                if use_shared_exchange and not self._selective_native_tp_groups:
                    raise RuntimeError("Native selective TP shared exchange requires a nonempty group")
                self._selective_tp_exchange_epoch = 0
            except BaseException:
                self._selective_native_failed = True
                raise
        if model_state is not None:
            model_state._selective_native_experiment = True

    def _prepare_selective_native_kv(self, layers, topk_indices) -> None:
        from vllm_ascend.compilation.updatable_graph import _ACTIVE_GRAPH

        graph = _ACTIVE_GRAPH.get()
        if graph is None:
            raise RuntimeError("Native selective capture requires an active UpdatableGraph")
        ops = torch.ops._C_ascend
        layer_ids = tuple(layer_id for layer_id, _, _ in layers)
        if not layer_ids:
            raise RuntimeError("Native selective capture requires at least one shared layer")
        captured_layers = self._selective_native_graph_layers.setdefault(graph, set())
        if len(set(layer_ids)) != len(layer_ids) or captured_layers.intersection(layer_ids):
            raise RuntimeError("Native selective groups overlap within the captured graph")
        captured_layers.update(layer_ids)
        topk = topk_indices.reshape(-1, self.topk)
        rows, width = topk.shape
        max_rows = self.topk_buffers_k[0].shape[0]
        if rows <= 0 or rows > max_rows:
            raise RuntimeError("Native selective capture exceeds hot-buffer rows")
        device = topk.device
        state = self._selective_native_groups.get(layer_ids)
        if state is None:
            state = [
                torch.full((max_rows, width), -1, dtype=torch.int32),
                torch.arange(width, dtype=torch.int32).repeat(max_rows, 1),
                torch.full((max_rows, len(layer_ids), width), -1, dtype=torch.int32),
                torch.full((max_rows, 9), -1, dtype=torch.int64),
                torch.zeros((max_rows, self.max_model_len), dtype=torch.int32),
                torch.full((max_rows, self.max_model_len), -1, dtype=torch.int32),
                torch.zeros(max_rows, dtype=torch.int32),
                torch.zeros((max_rows, 2), dtype=torch.int64),
            ]
            self._selective_native_groups[layer_ids] = state
            state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state)
            logger.info("Native selective experimental group %s CPU state: %.2f MiB", layer_ids, state_bytes / (1 << 20))

        def pair(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device), torch.empty(shape, dtype=dtype, pin_memory=True)

        topk_d, topk_h = pair((rows, width), torch.int32)
        topk_d.copy_(torch.where((topk >= 0) & (topk < self.max_model_len), topk, -1).to(torch.int32))
        meta_d, meta_h = pair((rows, 12), torch.int64)
        metadata = layers[0][1]
        seq_lengths = layers[0][2][:rows]
        if metadata.req_ids_tensor is None or metadata.selective_tail_rows is None:
            raise RuntimeError("Native selective capture requires request and tail metadata")
        meta_d.zero_()
        meta_d[:, 0].copy_(metadata.req_ids_tensor[:rows])
        meta_d[:, 4].copy_(seq_lengths)
        meta_d[:, 5:7].copy_(metadata.selective_tail_rows[:rows])
        meta_d[:, 7].fill_(self.block_size)
        host_blocks = self.kv_cache_config.num_blocks
        meta_d[:, 8].fill_(host_blocks)
        meta_d[:, 9].copy_((seq_lengths - 1).clamp_min(0))
        meta_d[:, 10].copy_(self._selective_native_rows[:rows])
        meta_d[:, 11].fill_(self.selective_row_stride)
        physical_d, physical_h = pair((rows, width), torch.int32)
        block_d, block_h = pair((rows, width // self.block_size + 1), torch.int32)
        shared = [topk_d, topk_h, meta_d, meta_h, physical_d, physical_h, block_d, block_h, *state]
        layer_buffers = []
        scatter = []
        coordinator = self._selective_native_tp_groups.get(layer_ids) if self.tp_size > 1 else None
        if self.tp_size > 1 and coordinator is None:
            raise RuntimeError("Native selective TP group does not match the registered layer roles")
        offsets = self._selective_native_offsets[: rows * width]
        count = self.selective_copy_counts[rows - 1]
        for layer_id, layer_metadata, _ in layers:
            pages_d = layer_metadata.block_table[:rows].to(torch.int32).contiguous()
            pages_h = torch.empty(pages_d.shape, dtype=torch.int32, pin_memory=True)
            if self.tp_rank == 0:
                host_k, host_v = self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]
                if host_k.shape[0] != host_blocks or host_v.shape[0] != host_blocks:
                    raise RuntimeError("Native selective group Host capacity disagrees with the cache plan")
            else:
                host_k = torch.empty(0, dtype=torch.bfloat16)
                host_v = torch.empty(0, dtype=torch.bfloat16)
            hot_k, hot_v = self.topk_buffers_k[layer_id], self.topk_buffers_v[layer_id]
            if coordinator is None:
                stage_k_d, stage_k_h = pair((rows, width, self.token_size_bytes_k), torch.uint8)
                stage_v_d, stage_v_h = pair((rows, width, self.token_size_bytes_v), torch.uint8)
            else:
                stage_k_d = torch.empty((rows, width, self.token_size_bytes_k), dtype=torch.uint8, device=device)
                stage_v_d = torch.empty((rows, width, self.token_size_bytes_v), dtype=torch.uint8, device=device)
                stage_k_h, stage_v_h = coordinator.stage_k[:rows], coordinator.stage_v[:rows]
            dest_d, dest_h = pair((2, rows, width), torch.int64)
            length_d, length_h = pair((2, rows, width), torch.int32)
            layer_buffers.extend([
                pages_d, pages_h, host_k, host_v, hot_k, hot_v, stage_k_h, stage_v_h,
                stage_k_d, stage_v_d, dest_h, dest_d, length_h, length_d,
            ])
            sources = torch.cat((offsets * self.token_size_bytes_k + stage_k_d.data_ptr(),
                                 offsets * self.token_size_bytes_v + stage_v_d.data_ptr()))
            scatter.append((layer_id, sources, dest_d, length_d))
        pinned = [topk_h, meta_h, physical_h, block_h]
        for offset in range(0, len(layer_buffers), 14):
            indices = (1, 6, 7, 10, 12) if coordinator is None else (1, 10, 12)
            pinned.extend(layer_buffers[offset + index] for index in indices)
        pinned_bytes = sum(tensor.numel() * tensor.element_size() for tensor in pinned)
        logger.info("Native selective experimental group %s per-graph pinned buffers: %.2f MiB",
                    layer_ids, pinned_bytes / (1 << 20))
        device_buffers = [topk_d, meta_d, physical_d, block_d, *(entry[1] for entry in scatter)]
        for offset in range(0, len(layer_buffers), 14):
            device_buffers.extend(layer_buffers[offset + index] for index in (8, 9, 11, 13))
        device_bytes = sum(tensor.numel() * tensor.element_size() for tensor in device_buffers)
        logger.info("Native selective experimental group %s per-graph named NPU buffers: %.2f MiB",
                    layer_ids, device_bytes / (1 << 20))
        if coordinator is None:
            handle = ops.selective_group_native_create(shared, layer_buffers, len(layer_ids))
        else:
            handle = ops.selective_group_tp_native_create(
                shared, layer_buffers, len(layer_ids), coordinator.buffers, self.tp_rank, self.tp_size
            )
        record = self._selective_native_graphs.setdefault(graph, [])
        record.append(handle)
        key = (tuple(int(value) for value in coordinator.buffers[6].tolist()) if coordinator is not None else
               tuple(extract_layer_index(self.offload_layer_names[layer_id]) for layer_id in layer_ids))
        self._selective_native_graph_keys.setdefault(graph, []).append(key)
        anchor = self.topk_buffers_k[layer_ids[0]]
        ops.selective_native_capture(handle, anchor)
        graph_identity = ops.selective_native_graph_identity(handle)
        if len(record) > 1 and ops.selective_native_graph_identity(record[0]) != graph_identity:
            raise RuntimeError("Native selective handles do not belong to the same captured graph")
        for index, (layer_id, sources, destinations, lengths) in enumerate(scatter):
            self.selective_physical[layer_id, :rows, 0, :].copy_(physical_d)
            self.copy_sfa_kv(sources, destinations.flatten(), lengths.flatten(), count)
            if index == len(scatter) - 1:
                # The last ready event also joins the completion marker to the main stream.
                ops.selective_native_scatter_done(handle, anchor)
            self.selective_prefetch_ready[layer_id].record()

    def _selective_native_ordered_handles(self, graph):
        handles = self._selective_native_graphs[graph]
        keys = self._selective_native_graph_keys[graph]
        if len(handles) != len(keys) or len(set(keys)) != len(keys):
            raise RuntimeError("Native selective graph group keys are incomplete or duplicated")
        return [handle for _, handle in sorted(zip(keys, handles))]

    @staticmethod
    def _selective_native_descriptor_key(desc):
        fields = ("cg_mode", "num_tokens", "num_reqs", "uniform_token_count", "max_query_len",
                  "num_active_loras", "num_ubatches")
        return tuple(str(getattr(desc, field, None)) for field in fields)

    def _selective_native_exchange_storage(self):
        groups = getattr(self, "_selective_native_tp_groups", {})
        if not getattr(self, "_selective_native_shared_exchange_requested", False) or not groups:
            return None
        return next(iter(groups.values())).exchange

    def mark_selective_native_failure(self) -> None:
        self._selective_native_failed = True
        groups = getattr(self, "_selective_native_tp_groups", {})
        storage = self._selective_native_exchange_storage()
        flags = [(storage, False)] if storage is not None else []
        for coordinator in groups.values():
            buffers = getattr(coordinator, "buffers", ())
            if buffers:
                flags.append((buffers[0], True))
        for flag, is_control in flags:
            try:
                if is_control:
                    flag = flag.view(torch.uint8)
                torch.ops._C_ascend.selective_tp_exchange_fail_cpu(flag)
            except Exception:
                logger.exception("Failed to publish selective TP failure; peer relies on timeout")

    def selective_native_all_gather(self, value, phase):
        storage = self._selective_native_exchange_storage()
        if storage is None:
            gathered = [None] * self.tp_size
            torch.distributed.all_gather_object(gathered, value, group=self.tp_group.cpu_group)
            return gathered
        try:
            payload = base64.b85encode(pickle.dumps((phase, value), protocol=5)).decode("ascii")
            self._selective_tp_exchange_epoch += 1
            encoded = torch.ops._C_ascend.selective_tp_exchange_cpu(
                storage, self.tp_rank, self.tp_size, self._selective_tp_exchange_epoch, payload
            )
            if not encoded:
                gathered = [None] * self.tp_size
                torch.distributed.all_gather_object(gathered, (phase, value), group=self.tp_group.cpu_group)
            else:
                gathered = [pickle.loads(base64.b85decode(item.encode("ascii"))) for item in encoded]
            if any(item[0] != phase for item in gathered):
                raise RuntimeError("Native selective TP exchange phases differ across ranks")
            result = [item[1] for item in gathered]
            torch.ops._C_ascend.selective_tp_exchange_confirm_cpu(
                storage, self.tp_rank, self.tp_size, self._selective_tp_exchange_epoch
            )
            return result
        except Exception:
            self.mark_selective_native_failure()
            raise

    def check_selective_native_graph_step(self, desc) -> None:
        if self.tp_size <= 1:
            return
        descriptor = self._selective_native_descriptor_key(desc)
        gathered = self.selective_native_all_gather(descriptor, "graph descriptor")
        if any(value != descriptor for value in gathered):
            self.mark_selective_native_failure()
            raise RuntimeError("Native selective TP replay descriptors differ across ranks")

    def check_selective_native_capture_status(self, error) -> None:
        self.check_selective_native_phase_status("graph capture", error)

    def check_selective_native_phase_status(self, phase, error) -> None:
        if self.tp_size > 1:
            try:
                if phase == "connector preemption":
                    gathered = self.selective_native_all_gather(
                        str(error) if error is not None else None, phase
                    )
                else:
                    gathered = [None] * self.tp_size
                    torch.distributed.all_gather_object(
                        gathered, str(error) if error is not None else None, group=self.tp_group.cpu_group
                    )
            except Exception:
                self.mark_selective_native_failure()
                raise
            if any(value is not None for value in gathered):
                self.mark_selective_native_failure()
                raise RuntimeError(f"Native selective TP {phase} failed: {gathered}")
        elif error is not None:
            self._selective_native_failed = True
            raise error

    def start_selective_native_graphs(self, graphs) -> None:
        graph_set = set(graphs.values())
        error = None
        signature = []
        try:
            if graph_set != set(self._selective_native_graphs):
                raise RuntimeError("Native selective graph-to-handle binding is incomplete")
            for desc, graph in graphs.items():
                if getattr(desc, "num_ubatches", 1) != 1:
                    raise RuntimeError("Native selective experiment does not support multi-ubatch capture")
                self._selective_native_ordered_handles(graph)
                descriptor = self._selective_native_descriptor_key(desc)
                signature.append((descriptor, tuple(sorted(self._selective_native_graph_keys[graph]))))
            signature.sort()
            if len({item[0] for item in signature}) != len(signature):
                raise RuntimeError("Native selective graph descriptors are ambiguous")
        except Exception as exc:
            error = str(exc)
        if self.tp_size > 1:
            gathered = [None] * self.tp_size
            torch.distributed.all_gather_object(gathered, (error, signature), group=self.tp_group.cpu_group)
            if any(value[0] is not None or value[1] != signature for value in gathered):
                raise RuntimeError("Native selective TP graph/group bindings differ across ranks")
        elif error is not None:
            raise RuntimeError(error)
        ops = torch.ops._C_ascend
        self._selective_native_started = True
        start_error = None
        try:
            for graph in graph_set:
                for handle in self._selective_native_ordered_handles(graph):
                    ops.selective_native_start(handle)
        except Exception as error:
            start_error = error
        if self.tp_size > 1:
            gathered = [None] * self.tp_size
            torch.distributed.all_gather_object(
                gathered, str(start_error) if start_error is not None else None, group=self.tp_group.cpu_group
            )
            if any(value is not None for value in gathered):
                raise RuntimeError(f"Native selective TP worker startup failed: {gathered}")
        elif start_error is not None:
            raise start_error
        self._selective_native_stopping = False

    def begin_selective_native_graph(self, graph, anchor):
        if self._selective_native_failed or self._selective_native_inflight or self._selective_native_stopping:
            raise RuntimeError("Native selective graph is unavailable or already in flight")
        handles = self._selective_native_ordered_handles(graph)
        self._selective_native_epoch += 1
        epoch = self._selective_native_epoch
        self._selective_native_inflight = True
        for handle in handles:
            torch.ops._C_ascend.selective_native_begin(handle, epoch, anchor)
        return epoch

    def finish_selective_native_graph(self, graph, epoch, anchor) -> None:
        handles = self._selective_native_ordered_handles(graph)
        ops = torch.ops._C_ascend
        for handle in handles:
            ops.selective_native_record_complete(handle, anchor)
        for handle in handles:
            ops.selective_native_retire(handle, epoch)
        self._selective_native_inflight = False

    def selective_native_cache_ids_for_modified_pages(self, block_ids, active_ids) -> set[int]:
        if self._selective_native_failed or self._selective_native_inflight:
            raise RuntimeError("Cannot inspect selective cache while a graph may be in flight")
        pages = np.asarray(sorted(set(block_ids)), dtype=np.int64)
        if pages.size == 0 or not active_ids:
            return set()
        found = set()
        for state in self._selective_native_groups.values():
            resident, identity = state[2].numpy(), state[3].numpy()
            for row, cache_id in enumerate(identity[:, 0]):
                if int(cache_id) in active_ids and np.isin(resident[row], pages).any():
                    found.add(int(cache_id))
        return found

    def stop_selective_native_graphs(self) -> None:
        self._selective_native_stopping = True
        error = "Native selective graph failure requires rank quarantine" if (
            self._selective_native_failed or self._selective_native_inflight
        ) else None
        if self.tp_size > 1:
            gathered = [None] * self.tp_size
            try:
                torch.distributed.all_gather_object(gathered, error, group=self.tp_group.cpu_group)
            except Exception:
                self._selective_native_failed = True
                raise
            if any(value is not None for value in gathered):
                self._selective_native_failed = True
                raise RuntimeError(f"Native selective TP stop preflight failed: {gathered}")
        elif error is not None:
            raise RuntimeError(error)
        stop_error = None
        try:
            for handles in self._selective_native_graphs.values():
                for handle in handles:
                    torch.ops._C_ascend.selective_native_stop(handle)
        except Exception as exc:
            stop_error = str(exc)
        if self.tp_size > 1:
            gathered = [None] * self.tp_size
            try:
                torch.distributed.all_gather_object(gathered, stop_error, group=self.tp_group.cpu_group)
            except Exception:
                self._selective_native_failed = True
                raise
            if any(value is not None for value in gathered):
                self._selective_native_failed = True
                raise RuntimeError(f"Native selective TP worker stop failed: {gathered}")
        elif stop_error is not None:
            self._selective_native_failed = True
            raise RuntimeError(stop_error)

    def release_selective_native_graphs(self, *, close_tp=False) -> None:
        ops = torch.ops._C_ascend
        release_error = None
        try:
            for graph, handles in self._selective_native_graphs.items():
                graph.reset()
                for handle in handles:
                    ops.selective_native_release(handle)
        except Exception as error:
            release_error = str(error)
        if self.tp_size > 1:
            gathered = [None] * self.tp_size
            try:
                torch.distributed.all_gather_object(gathered, release_error, group=self.tp_group.cpu_group)
            except Exception:
                self._selective_native_failed = True
                raise
            if any(value is not None for value in gathered):
                self._selective_native_failed = True
                raise RuntimeError(f"Native selective TP graph release failed: {gathered}")
        elif release_error is not None:
            self._selective_native_failed = True
            raise RuntimeError(release_error)
        self._selective_native_graphs.clear()
        self._selective_native_graph_layers.clear()
        self._selective_native_graph_keys.clear()
        self._selective_native_groups.clear()
        self._selective_native_started = False
        if close_tp and self.tp_size > 1:
            close_error = None
            try:
                for coordinator in self._selective_native_tp_groups.values():
                    coordinator.close()
            except Exception as error:
                close_error = str(error)
            gathered = [None] * self.tp_size
            try:
                torch.distributed.all_gather_object(gathered, close_error, group=self.tp_group.cpu_group)
            except Exception:
                self._selective_native_failed = True
                raise
            if any(value is not None for value in gathered):
                self._selective_native_failed = True
                raise RuntimeError(f"Native selective TP coordinator close failed: {gathered}")
            self._selective_native_tp_groups.clear()

    def prefetch_selective_kv(self, rows, sources, destinations, lengths) -> None:
        count = self.selective_copy_counts[rows - 1]
        assert self.selective_sdma_prefetch is not None
        self.selective_sdma_prefetch.transfer(self, sources, destinations, lengths, count, to_device=True)

    def copy_sfa_kv(self, sources, destinations, lengths, count) -> None:
        """Enqueue bounded descriptor copies on the current compute stream."""
        result = offload.sparse_copy(sources, destinations, lengths, count, sources.device)
        if result not in (None, 0):
            raise RuntimeError(f"memfabric fused_copy_sfa tail H2D failed with result={result}")

    def reset_selective_hot_cache(self) -> None:
        if not getattr(self.sparse_kv_offload_config, "selective_offload", False):
            return
        for keys, request in zip(self.selective_hot_keys, self.selective_hot_requests):
            keys.fill_(-1)
            request.fill_(-1)

    def write_selective_tail(self, layer_name, slots, tail_rows, k, v):
        layer_id = self._get_offload_layer_id(layer_name)
        k_cache, v_cache = self.topk_buffers_k[layer_id], self.topk_buffers_v[layer_id]
        rows = slots.numel()

        def tail_positions():
            host_slots = slots.reshape(-1).to(torch.int64)
            tail_slot, first = tail_rows[:rows].unbind(-1)
            valid = (tail_slot >= 0) & (host_slots >= 0)
            safe_slots = host_slots.clamp_min(0)
            offset = safe_slots % self.block_size
            tail_base = tail_slot.clamp_min(0) * self.selective_row_stride + self.topk_buffer_size
            return valid, safe_slots, offset, tail_base, first

        valid, _, offset, tail_base, _ = tail_positions()
        device_slots = torch.where(valid, tail_base + offset, -1)
        torch_npu.npu_scatter_pa_kv_cache(
            key=k.reshape(rows, 1, k_cache.shape[-1]).contiguous(),
            value=v.reshape(rows, 1, v_cache.shape[-1]).contiguous(),
            key_cache=k_cache.view(-1, self.block_size, 1, k_cache.shape[-1]),
            value_cache=v_cache.view(-1, self.block_size, 1, v_cache.shape[-1]),
            slot_mapping=device_slots,
            cache_mode="Norm",
        )
        if self.tp_rank != 0:
            return

        def submit():
            # Build stream-local temporaries from the persistent step metadata.
            valid, safe_slots, offset, tail_base, first = tail_positions()
            start = first.clamp_min(0)
            host_base = safe_slots - offset + start
            source = tail_base + start
            tokens = torch.where(valid & (offset == self.block_size - 1), self.block_size - start, 0)
            for i, (cache, gva, row_bytes) in enumerate(
                (
                    (k_cache, self.gvas_k_bases[layer_id], self.token_size_bytes_k),
                    (v_cache, self.gvas_v_bases[layer_id], self.token_size_bytes_v),
                )
            ):
                segment = slice(i * rows, (i + 1) * rows)
                self.d2h_src_ptrs_npu[segment].copy_(cache.data_ptr() + source * row_bytes)
                self.d2h_dst_ptrs_npu[segment].copy_(gva + host_base * row_bytes)
                self.d2h_lengths_npu[segment].copy_((tokens * row_bytes).to(torch.int32))
            self.d2h_size_npu.fill_(2 * rows)
            self._submit_current_kv_writeback(k_cache.device, rows, tail=True)

        context = get_forward_context()
        pending = getattr(context, "_selective_tail_writebacks", None)
        if pending is None:
            pending = context._selective_tail_writebacks = []
        pending.append(submit)

    def flush_selective_tail_group(self, layer_name: str, capturing: bool) -> None:
        if layer_name not in self.selective_group_tails:
            return
        context = get_forward_context()
        if capturing:
            context._selective_tail_fence_required = True
        if self.tp_rank != 0:
            return
        pending = context._selective_tail_writebacks
        if capturing:
            self.current_kv_save_stream.wait_stream(torch_npu.npu.current_stream())
            with torch_npu.npu.stream(self.current_kv_save_stream):
                for submit in pending:
                    submit()
        else:
            for submit in pending:
                submit()
        pending.clear()

    def finish_selective_tail_writeback(self) -> None:
        self.wait_for_current_kv_writeback(capturing=True)
        if get_tensor_model_parallel_world_size() > 1:
            # Every rank must observe TP0's Host commit before the next replay.
            get_tp_group().all_reduce(self.selective_writeback_fence)

    def offload_new_kv(
        self,
        layer_name: str,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,
        v_cache_npu: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        selective = getattr(self.sparse_kv_offload_config, "selective_offload", False)
        if selective or (self.use_fused_overlap and not has_prefill and k is not None and v is not None):
            layer_id = self._get_offload_layer_id(layer_name)
            if not has_prefill and k is not None and v is not None:
                self.current_kv_by_layer[layer_id] = (k, v)
        use_side_stream = self.tp_rank == 0 and (selective or self.use_fused_overlap) and capturing and not has_prefill
        if not use_side_stream:
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
            )
            return

        current_kv_ready = torch_npu.npu.current_stream().record_event()
        with torch_npu.npu.stream(self.current_kv_save_stream):
            self.current_kv_save_stream.wait_event(current_kv_ready)
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
            )

    def _offload_new_kv_on_current_stream(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        v: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError(
                "Sparse KV offload prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        if has_prefill:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("prefill offload requires NPU paged K/V caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device

        token_count = slot_mapping.numel()
        if token_count > self.max_num_tokens:
            raise ValueError(
                "Sparse KV offload rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        num_v_slots = v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                f"Sparse KV offload CPU K/V pools have incompatible token capacities: k={num_k_slots}, v={num_v_slots}"
            )
        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v

        dst_k = int(k_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(self.token_size_bytes_v)
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)
        self._submit_current_kv_writeback(device, token_count)

    def _submit_current_kv_writeback(self, device: torch.device, entries=None, *, tail=False) -> None:
        if self.selective_sdma_prefetch is not None:
            transfer = self.selective_sdma_tail if tail else self.selective_sdma_prefill
            assert transfer is not None
            transfer.transfer(
                self,
                self.d2h_src_ptrs_npu[: 2 * entries],
                self.d2h_dst_ptrs_npu[: 2 * entries],
                self.d2h_lengths_npu[: 2 * entries],
                self.d2h_size_npu,
                to_device=False,
            )
            return
        result = offload.sparse_copy(
            self.d2h_src_ptrs_npu,
            self.d2h_dst_ptrs_npu,
            self.d2h_lengths_npu,
            self.d2h_size_npu,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")

    def prepare_fused_overlap_external_plan(
        self,
        layer_name: str,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
        capturing: bool = False,
        skip_topk: bool = False,
    ) -> bool:
        self._validate_fused_overlap_external_plan_inputs(
            num_tokens,
            topk_indices_npu,
            req_ids_npu,
            stable_prefix_lens_npu,
            visible_seq_lens_npu,
            selection_membership_map,
        )
        layer_id = self._get_offload_layer_id(layer_name)
        plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
        plan_storage = selection_membership_map[
            :num_tokens,
            plan_start:FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS,
        ]
        encoded_plan_stride = selection_membership_map.stride(0)

        owner_layer_id = self.fused_overlap_plan_owner_layer_id
        owner_map = self.fused_overlap_plan_membership_map
        can_reuse_owner_plan = (
            skip_topk
            and owner_layer_id is not None
            and layer_id > owner_layer_id
            and self.fused_overlap_plan_topk == self.topk
            and self.fused_overlap_plan_num_tokens == num_tokens
            and owner_map is not None
        )
        if can_reuse_owner_plan:
            assert owner_map is not None
            if selection_membership_map.data_ptr() != owner_map.data_ptr():
                if capturing:
                    torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
                plan_storage.copy_(
                    owner_map[:num_tokens, plan_start:FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS],
                    non_blocking=capturing,
                )
            return True

        def run_planner(enqueue: bool) -> None:
            sparse_kv_ops = _sparse_kv_ops()
            planner = (
                sparse_kv_ops.sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows
                if enqueue
                else sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
            )
            planner(
                self.lru_req_ids_ptr,
                self.lru_last_req_ids_ptrs[layer_id],
                self.lru_topk_indices_ptr,
                self.lru_stable_prefix_lens_ptr,
                self.lru_slot_to_token_ptrs[layer_id],
                self.lru_slots_ptrs[layer_id],
                self.lru_current_slots_ptr,
                self.lru_miss_count_ptrs[layer_id],
                self.lru_miss_tokens_ptrs[layer_id],
                self.lru_miss_slots_ptrs[layer_id],
                self.lru_token_mark_workspace_ptr,
                self.lru_token_pos_workspace_ptr,
                self.lru_slot_workspace_ptr,
                self.lru_miss_position_workspace_ptr,
                self.lru_epochs_ptr,
                self.lru_physical_row_workspace_ptr,
                self.max_num_topk_rows,
                plan_storage.data_ptr(),
                encoded_plan_stride,
                num_tokens,
                self.topk,
                self.topk_buffer_size,
                self.max_model_len,
                self.lru_workspace_threads,
                self.lru_workspace_threads,
                self.lru_visible_seq_lens_ptr,
            )

        if capturing:
            plan_inputs_ready = torch_npu.npu.current_stream().record_event()
            with torch_npu.npu.stream(self.fused_plan_stream):
                self.fused_plan_stream.wait_event(plan_inputs_ready)
                if self.tp_rank == 0:
                    self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu, non_blocking=True)
                    self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu, non_blocking=True)
                    self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(stable_prefix_lens_npu, non_blocking=True)
                    self.lru_visible_seq_lens_cpu[:num_tokens].copy_(visible_seq_lens_npu, non_blocking=True)
                    run_planner(enqueue=True)
                    self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                        self.lru_physical_row_workspace[
                            self.max_num_topk_rows * 2 : self.max_num_topk_rows * 2 + num_tokens
                        ],
                        non_blocking=True,
                    )
                self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
            self.fused_overlap_plan_owner_layer_id = layer_id
            self.fused_overlap_plan_topk = self.topk
            self.fused_overlap_plan_num_tokens = num_tokens
            self.fused_overlap_plan_membership_map = selection_membership_map
            return True

        planner_error = None
        if self.tp_rank == 0:
            self.fused_plan_status_npu.zero_()
            try:
                self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu)
                self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu)
                self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(stable_prefix_lens_npu)
                self.lru_visible_seq_lens_cpu[:num_tokens].copy_(visible_seq_lens_npu)
                run_planner(enqueue=False)
                self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                    self.lru_physical_row_workspace[
                        self.max_num_topk_rows * 2 : self.max_num_topk_rows * 2 + num_tokens
                    ]
                )
            except Exception as exc:
                planner_error = exc
                self.fused_plan_status_npu.fill_(1)
        self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
        if int(self.fused_plan_status_npu.item()) != 0:
            detail = (
                f"{type(planner_error).__name__}: {planner_error}"
                if planner_error is not None
                else "TP0 external planner failed"
            )
            raise RuntimeError(
                f"SFA fused_overlap external planner failed: layer={layer_name}, tp_rank={self.tp_rank}, {detail}"
            ) from planner_error
        self.fused_overlap_plan_owner_layer_id = layer_id
        self.fused_overlap_plan_topk = self.topk
        self.fused_overlap_plan_num_tokens = num_tokens
        self.fused_overlap_plan_membership_map = selection_membership_map
        return True

    def _validate_fused_overlap_external_plan_inputs(
        self,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
    ) -> None:
        if not self.use_fused_overlap:
            raise RuntimeError("external FSA plan requires fused_overlap mode")
        if num_tokens <= 0 or num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "external FSA plan rows exceed configured workspace: "
                f"num_tokens={num_tokens}, max_rows={self.max_num_topk_rows}"
            )
        expected_topk_shape = (num_tokens, self.topk)
        if tuple(topk_indices_npu.shape) != expected_topk_shape:
            raise ValueError(
                "external FSA plan requires flattened single-head TopK input: "
                f"expected={expected_topk_shape}, actual={tuple(topk_indices_npu.shape)}"
            )
        if tuple(req_ids_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one request id per logical row: "
                f"expected=({num_tokens},), actual={tuple(req_ids_npu.shape)}"
            )
        if tuple(stable_prefix_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one stable prefix length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(stable_prefix_lens_npu.shape)}"
            )
        if tuple(visible_seq_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one visible KV length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(visible_seq_lens_npu.shape)}"
            )
        if (
            selection_membership_map.dim() != 2
            or selection_membership_map.shape[0] < num_tokens
            or selection_membership_map.shape[1] < FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS
            or selection_membership_map.dtype != torch.int16
            or selection_membership_map.device.type != "cpu"
        ):
            raise ValueError(
                "external FSA plan requires mapped CPU int16 membership storage: "
                f"min_shape=({num_tokens}, {FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS}), "
                f"actual_shape={tuple(selection_membership_map.shape)}, "
                f"dtype={selection_membership_map.dtype}, "
                f"device={selection_membership_map.device}"
            )

    def inject_current_kv_into_selection(
        self,
        layer_name: str,
        num_tokens: int,
        selection_kv_cache: torch.Tensor,
        selection_k_rope: torch.Tensor,
        capturing: bool = False,
    ) -> None:
        layer_id = self._get_offload_layer_id(layer_name)
        current_kv = self.current_kv_by_layer.get(layer_id)
        if current_kv is None:
            raise RuntimeError(f"current decode K/V is unavailable for fused layer {layer_name}")
        if capturing:
            torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
        current_k, current_rope = current_kv
        current_k = current_k.reshape(-1, selection_kv_cache.shape[-1])[:num_tokens]
        current_rope = current_rope.reshape(-1, selection_k_rope.shape[-1])[:num_tokens]
        linear_slots = self.fused_plan_current_linear_slots_npu[:num_tokens]
        indices = linear_slots.view(-1, 1)
        flat_kv = selection_kv_cache.reshape(-1, selection_kv_cache.shape[-1])
        flat_rope = selection_k_rope.reshape(-1, selection_k_rope.shape[-1])
        torch_npu.npu_scatter_nd_update_(flat_kv, indices, current_k)
        torch_npu.npu_scatter_nd_update_(flat_rope, indices, current_rope)

    def wait_for_current_kv_writeback(self, capturing: bool = False) -> None:
        selective = getattr(self.sparse_kv_offload_config, "selective_offload", False)
        if (selective or self.use_fused_overlap) and capturing and self.tp_rank == 0:
            torch_npu.npu.current_stream().wait_stream(self.current_kv_save_stream)


_SPARSE_KV_OFFLOAD_MANAGER: SparseKVOffloadManager | None = None


def init_sparse_kv_offload_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    global _SPARSE_KV_OFFLOAD_MANAGER
    if _SPARSE_KV_OFFLOAD_MANAGER is None:
        if (
            not getattr(sparse_kv_offload_config, "selective_offload", False)
            and get_ascend_device_type() != AscendDeviceType.A3
        ):
            raise RuntimeError("Sparse KV offload is only support on A3")
        _SPARSE_KV_OFFLOAD_MANAGER = SparseKVOffloadManager(
            vllm_config,
            kv_cache_config,
            sparse_kv_offload_config,
        )
    return _SPARSE_KV_OFFLOAD_MANAGER


def get_sparse_kv_offload_manager():
    assert _SPARSE_KV_OFFLOAD_MANAGER is not None, "KV offload manager is not initialized."
    return _SPARSE_KV_OFFLOAD_MANAGER
