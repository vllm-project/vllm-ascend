# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU-side Engram storage, routing and lookup.

Every layer is INT8 with group-32 FP32 scales, sharded into contiguous row
ranges so a lookup only crosses the node-local fabric for rows another rank
owns.  With ``EngramConfig.cpu_offload`` the shard stays in host memory:
``aclrtHostRegisterV2`` pins it and ``aclrtHostGetDevicePointer`` publishes the
address the NPU gather kernel reads, so an offloaded table needs neither an H2D
copy nor a host-side gather.
"""

import ctypes
import json
import socket
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.logger import logger
from vllm.triton_utils import tl, triton

SCALE_GROUP = 32
# A 384M row table overflows the 32 bit offset arithmetic a single Triton tile
# can express, so the device address of every group of rows is published
# separately.
CHUNK_ROWS = 1 << 22
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


def engram_cpu_offload(vllm_config) -> bool:
    """Whether the Engram table is offloaded to host memory (UVA lookup).

    vLLM only grew ``--engram-config`` after 0.28, and the option is what turns
    the offload on: without it the tables stay on the device, exactly like
    upstream.
    """

    engram_config = getattr(vllm_config, "engram_config", None)
    return bool(engram_config is not None and engram_config.cpu_offload)


def quantize_engram_rows(rows):
    """Group32 symmetric INT8 with FP32 power-of-two scales and ties-to-even."""
    grouped = rows.float().unflatten(-1, (-1, SCALE_GROUP))
    maximum = grouped.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    # NPU exp2 can return one ULP below an exact power of two, changing
    # ties-to-even codes. ldexp constructs the binary scale exactly.
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent), torch.ldexp(torch.ones_like(scale), exponent.int()), scale)
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).flatten(-2)
    return codes, scale.squeeze(-1)


def dequantize_engram_rows(codes, scale):
    # Keep one FP32 work buffer: in-place scaling avoids the extra FP32 result
    # allocation created by the broadcast multiply expression.
    decoded = codes.float().unflatten(-1, (-1, SCALE_GROUP))
    decoded.mul_(scale.unsqueeze(-1))
    return decoded.flatten(-2).bfloat16()


@triton.jit
def _engram_int8_gather_dequant_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    output_ptr,
    rows,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, WIDTH)
    source_row = tl.load(ids_ptr + row).to(tl.int64)
    codes = tl.load(weight_ptr + source_row * WIDTH + offsets).to(tl.float32)
    scales = tl.load(scale_ptr + source_row * (WIDTH // GROUP) + offsets // GROUP)
    tl.store(output_ptr + row * WIDTH + offsets, (codes * scales).to(tl.bfloat16))


def gather_dequantize_engram_int8(
    weight: torch.Tensor, scales: torch.Tensor, ids: torch.Tensor, width: int
) -> torch.Tensor:
    """Gather rows ``ids`` from a device table and dequantize in one kernel."""

    # Importing the ops package initializes the active Triton backend, so keep
    # it out of CPU-only routing and test workers.
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    if ids.numel() == 0:
        return torch.empty((0, width), dtype=torch.bfloat16, device=weight.device)
    init_device_properties_triton()
    output = torch.empty((ids.shape[0], width), dtype=torch.bfloat16, device=weight.device)
    _engram_int8_gather_dequant_kernel[(ids.shape[0],)](
        weight, scales, ids, output, ids.shape[0], WIDTH=width, GROUP=SCALE_GROUP, num_warps=4
    )
    return output


@cache
def _host_library() -> ctypes.CDLL:
    """The CANN runtime entry points that publish host memory to the device."""

    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class HostUvaBuffer:
    """Host memory the device gathers from directly."""

    def __init__(self, shape, dtype, device):
        self.lib = _host_library()
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        rc = self.lib.aclrtHostRegisterV2(self.pointer, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
        if rc:
            raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
        address = ctypes.c_void_p()
        rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
        if rc:
            raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
        self.ptrs = torch.tensor(
            [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
            dtype=torch.int64,
            device=device,
        )

    def close(self):
        self.lib.aclrtHostUnregister(self.pointer)
        self.tensor = None
        self.buffer = None
        self.lib.aclrtFreeHost(self.pointer)
        self.pointer = ctypes.c_void_p()


@triton.jit
def _engram_host_uva_gather_dequant_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        index = tl.load(ids + row).to(tl.int64)
        chunk = index // CHUNK
        local = index % CHUNK
        codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        scale = tl.load(scales + local * (WIDTH // GROUP) + col // GROUP)
        tl.store(output + row * WIDTH + col, (value * scale).to(tl.bfloat16))


def gather_dequantize_host_uva(codes: HostUvaBuffer, scales: HostUvaBuffer, ids: torch.Tensor) -> torch.Tensor:
    """Gather rows ``ids`` from a registered host table and dequantize on device."""

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    width = codes.tensor.shape[-1]
    rows = ids.numel()
    output = torch.empty((rows, width), dtype=torch.bfloat16, device=ids.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    _engram_host_uva_gather_dequant_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs,
        ids.reshape(-1).to(torch.int64),
        output,
        rows,
        CHUNK=CHUNK_ROWS,
        WIDTH=width,
        GROUP=SCALE_GROUP,
        num_warps=4,
    )
    return output


class EngramQueryGroup:
    """One node group shared by all Engram layers; TP leaders submit queries.

    All ranks (including idle DP replicas) must call lookup in the same order.
    Counts and all-to-all split sizes are eager metadata, not graph inputs.
    """

    def __init__(self, group, cpu_group, tp_group, tp_source):
        self.group = group
        self.cpu_group = cpu_group
        self.tp_group = tp_group
        self.tp_source = tp_source
        # HCCL metadata avoids the CPU/Gloo rendezvous; standalone Gloo/MPI
        # probes keep CPU metadata.
        backend = str(dist.get_backend(group)).lower()
        self.metadata_on_device = backend not in ("gloo", "mpi")
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.is_source = dist.get_rank() == tp_source

    @classmethod
    def from_vllm(cls, parallel):
        # Lazy imports keep the transport usable in standalone distributed probes.
        from vllm.distributed import get_ep_group, get_tp_group

        ep, tp = get_ep_group(), get_tp_group()
        hosts = [None] * ep.world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=ep.cpu_group)
        node_groups = [[ep.ranks[i] for i, host in enumerate(hosts) if host == name] for name in dict.fromkeys(hosts)]
        selected = None
        for ranks in node_groups:
            # Every world rank creates groups in the same order.
            cpu = dist.new_group(ranks, backend="gloo")
            device = dist.new_group(ranks, backend=dist.get_backend(ep.device_group))
            if dist.get_rank() in ranks:
                selected = cls(device, cpu, tp.device_group, tp.ranks[0])
        return selected


class NodeShardedEngram(nn.Module):
    """Contiguous INT8 row shards; only BF16 rows cross the node-local fabric."""

    def __init__(self, rows, width, query_group, device=None, cpu_offload=False):
        super().__init__()
        self.rows, self.width = rows, width
        self.query_group = query_group
        self._empty_flat = torch.empty(0, dtype=torch.int64, device="cpu")
        # Reuse fixed-size HCCL metadata buffers across requests.
        self._metadata_device_buffers = {}
        self._empty_metadata = torch.zeros(query_group.size + 1, dtype=torch.int64, device="cpu")
        # Ceil partition leaves at most size-1 unused rows, never a replica.
        self.shard_rows = (rows + query_group.size - 1) // query_group.size
        self.start = query_group.rank * self.shard_rows
        self.end = min(self.start + self.shard_rows, rows)
        # Host offload keeps the shard where it already is: host memory the
        # device reads through its registered address.  Everything else
        # (routing, collectives, loaders) is unchanged.
        self._host_uva = None
        if cpu_offload:
            if device is None:
                device = torch.device("npu", torch.npu.current_device())
            host_rows = self.end - self.start
            self._host_uva = (
                HostUvaBuffer((host_rows, width), torch.int8, device),
                HostUvaBuffer((host_rows, width // SCALE_GROUP), torch.float32, device),
            )
            registered = host_rows * (width + (width // SCALE_GROUP) * 4)
            logger.info(
                "Engram shard rows %d-%d offloaded to host memory (%.2f GiB registered)",
                self.start,
                self.end,
                registered / 1024**3,
            )
        codes = self._host_uva[0].tensor if self._host_uva is not None else None
        self.weight = nn.Parameter(
            codes
            if codes is not None
            else torch.empty(
                self.end - self.start,
                width,
                dtype=torch.int8,
                device=device,
                pin_memory=False,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            (
                self._host_uva[1].tensor
                if self._host_uva is not None
                else torch.empty(
                    self.end - self.start,
                    width // SCALE_GROUP,
                    dtype=torch.float32,
                    device=device,
                    pin_memory=False,
                )
            ),
        )

    def set_rows(self, start, rows):
        """Quantize BF16 rows into local storage without an INT8 table copy."""
        end = start + rows.shape[0]
        codes, scales = quantize_engram_rows(rows.to(self.weight.device))
        self.weight.data[start:end].copy_(codes)
        self.weight_scale[start:end].copy_(scales)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        """Load one INT8 Engram shard, quantizing a BF16 source when needed.

        Offloaded tables stay host resident; only requested BF16 rows enter the
        node-local all-to-all response buffer.
        """
        root = Path(model_path)
        scale_key = key.removesuffix(".weight") + ".scale"
        index = {}
        quant_index = root / "quant_model_weights.safetensors.index.json"
        if quant_index.is_file():
            index = json.loads(quant_index.read_text())["weight_map"]
        if key not in index:
            index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            quantized = tensor.get_dtype() in ("I8", "INT8")
            if quantized:
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    for start in range(self.start, self.end, chunk_rows):
                        stop = min(start + chunk_rows, self.end)
                        self.weight.data[start - self.start : stop - self.start].copy_(tensor[start:stop])
                        self.weight_scale[start - self.start : stop - self.start].copy_(scale[start:stop])
        if not quantized:
            for start in range(self.start, self.end, chunk_rows):
                stop = min(start + chunk_rows, self.end)
                self.set_rows(start - self.start, tensor[start:stop])
        logger.info("Engram shard rows %d-%d loaded from %s", self.start, self.end, index[key])

    def lookup_local(self, ids):
        # Idle DP replicas still enter routing collectives, but must not launch
        # gather/dequant kernels for an empty owner request.
        if ids.numel() == 0:
            return torch.empty((*ids.shape, self.width), dtype=torch.bfloat16, device=self.weight.device)
        flat_ids = ids.reshape(-1)
        if self._host_uva is not None:
            # The routing path hands local ids as CPU tensors: the registered
            # table is device-readable, so move them and gather on device.
            device = self._host_uva[0].ptrs.device
            rows = gather_dequantize_host_uva(self._host_uva[0], self._host_uva[1], flat_ids.to(device))
        elif self.weight.device.type == "npu":
            rows = gather_dequantize_engram_int8(self.weight, self.weight_scale, flat_ids, self.width)
        else:
            # index_select avoids the extra advanced-indexing wrapper on the
            # CPU-resident table and keeps row selection explicit.
            rows = dequantize_engram_rows(
                torch.index_select(self.weight, 0, flat_ids),
                torch.index_select(self.weight_scale, 0, flat_ids),
            )
        return rows.view(*ids.shape, self.width)

    def _metadata(self, ids):
        q = self.query_group
        flat = ids.reshape(-1) if q.is_source else ids.new_empty(0)
        if flat.numel() == 0:
            return self._empty_flat, self._empty_flat, self._empty_metadata
        invalid = bool(flat.min() < 0 or flat.max() >= self.rows)
        owners = flat.clamp(0, self.rows - 1) // self.shard_rows if invalid else flat // self.shard_rows
        # Only owner grouping is required; preserving equal-owner order adds
        # avoidable CPU sort work because the same permutation restores rows.
        order = owners.argsort(stable=False)
        counts = torch.bincount(owners, minlength=q.size)
        metadata = torch.empty(q.size + 1, dtype=torch.int64, device="cpu")
        metadata[:-1].copy_(counts)
        metadata[-1] = int(invalid)
        return flat, order, metadata

    @torch.inference_mode()
    def _forward_with_gathered(self, ids, gathered, routing=None, broadcast=True, output=None):
        """Return ids.shape + [width], bit-preserving, even when a DP is idle.

        IDs reside on CPU; hashing/history already runs at the eager boundary.
        Exactly one TP rank submits the DP's queries. All owners serve requests;
        reverse all-to-all restores requester order before the TP broadcast.
        """
        q = self.query_group
        if routing is None:
            flat, order, metadata = self._metadata(ids)
        else:
            flat, order, metadata = routing
        counts = [row.tolist() for row in gathered]
        send = metadata[:-1].tolist()
        recv = [row[q.rank] for row in counts]
        total_recv = sum(recv)
        total_requests = sum(send)
        total_global = sum(sum(row[:-1]) for row in counts)
        backend = str(dist.get_backend(q.group)).lower()
        # HCCL collectives require NPU tensors even when the table is host resident.
        device = self.weight.device
        if device.type == "cpu" and backend not in ("gloo", "mpi"):
            device = torch.device("npu")
        if device.type == "npu" and device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        # All-zero rounds are skipped identically on every rank (HCCL portability).
        if total_global:
            incoming = torch.empty(total_recv, dtype=torch.int64, device=device)
            ordered_ids = torch.index_select(flat, 0, order).to(device)
            dist.all_to_all_single(incoming, ordered_ids, recv, send, group=q.group)
            local_ids = incoming - self.start
            local_values = self.lookup_local(local_ids.cpu() if self.weight.device.type == "cpu" else local_ids)
            values = local_values.to(device=device, dtype=torch.bfloat16).contiguous()
            returned = torch.empty((total_requests, self.width), dtype=torch.bfloat16, device=device)
            dist.all_to_all_single(returned, values, send, recv, group=q.group)
        else:
            returned = torch.empty((0, self.width), dtype=torch.bfloat16, device=device)
        if output is None:
            result = torch.empty((ids.numel(), self.width), dtype=torch.bfloat16, device=device)
        else:
            result = output
        if q.is_source:
            result[order.to(device)] = returned
        if broadcast and result.numel():
            dist.broadcast(result, src=q.tp_source, group=q.tp_group)
        return result.view(*ids.shape, self.width)

    def _gather_metadata_device(self, metadata, device, group):
        """All-gather metadata through reusable device buffers."""
        key = (str(device), metadata.numel(), metadata.dtype)
        buffers = self._metadata_device_buffers.get(key)
        if buffers is None:
            buffers = (
                torch.empty(metadata.numel(), dtype=metadata.dtype, device=device),
                torch.empty(self.query_group.size * metadata.numel(), dtype=metadata.dtype, device=device),
            )
            self._metadata_device_buffers[key] = buffers
        metadata_device, gathered_device = buffers
        metadata_device.copy_(metadata, non_blocking=False)
        dist.all_gather_into_tensor(gathered_device, metadata_device, group=group)
        # The reshape/unbind views are copied to CPU before returning.  Keep
        # consumption local to this route so the reusable device buffer remains
        # safe for the next collective.
        return list(gathered_device.reshape(self.query_group.size, *metadata.shape).cpu().unbind(0))

    @torch.inference_mode()
    def forward(self, ids):
        q = self.query_group
        routing = self._metadata(ids)
        metadata = routing[2]
        if q.metadata_on_device:
            device = self.weight.device if self.weight.device.type == "npu" else torch.device("npu")
            gathered = self._gather_metadata_device(metadata, device, q.group)
        else:
            gathered = [torch.empty_like(metadata) for _ in range(q.size)]
            dist.all_gather(gathered, metadata, group=q.cpu_group)
        return self._forward_with_gathered(ids, gathered, routing)

    @torch.inference_mode()
    def forward_many(self, ids_list):
        """Route several Engram tables with one CPU metadata collective."""
        return self.route_many([self] * len(ids_list), ids_list)

    @torch.inference_mode()
    def route_many(self, tables, ids_list):
        """Route distinct tables while sharing their CPU metadata collective."""
        if not ids_list:
            return []
        q = self.query_group
        routing = [table._metadata(ids) for table, ids in zip(tables, ids_list)]
        metadata = [item[2] for item in routing]
        packed = torch.cat(metadata)
        if q.metadata_on_device:
            device = tables[0].weight.device if tables[0].weight.device.type == "npu" else torch.device("npu")
            gathered_packed = self._gather_metadata_device(packed, device, q.group)
        else:
            gathered_packed = [torch.empty_like(packed) for _ in range(q.size)]
            dist.all_gather(gathered_packed, packed, group=q.cpu_group)
        width = q.size + 1
        gathered = [
            [row[offset : offset + width] for row in gathered_packed]
            for offset in range(0, len(ids_list) * width, width)
        ]
        if len(tables) == 1:
            result = tables[0]._forward_with_gathered(ids_list[0], gathered[0], routing[0])
            return [result]
        total = sum(ids.numel() * table.width for table, ids in zip(tables, ids_list))
        device = tables[0].weight.device
        if device.type == "cpu" and str(dist.get_backend(q.group)).lower() not in ("gloo", "mpi"):
            device = torch.device("npu")
        if device.type == "npu" and device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        combined = torch.empty(total, dtype=torch.bfloat16, device=device)
        results = []
        offset = 0
        for table, ids, group, item in zip(tables, ids_list, gathered, routing):
            size = ids.numel() * table.width
            result = table._forward_with_gathered(
                ids,
                group,
                item,
                broadcast=False,
                output=combined[offset : offset + size].view(ids.numel(), table.width),
            )
            results.append(result)
            offset += size
        if total:
            dist.broadcast(combined, src=q.tp_source, group=q.tp_group)
        return results
