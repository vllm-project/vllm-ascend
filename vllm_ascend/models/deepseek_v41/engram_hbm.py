# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local BF16/INT8 Engram storage, independent of model TP."""

import json
import socket
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

_OFFLOAD_BUFFER_CACHE_SIZE = 8
_OFFLOAD_BUFFER_BYTES_LIMIT = 512 * 1024 * 1024
_BF16_BYTES = 2


def quantize_engram_rows(rows):
    """Group32 symmetric INT8 with FP32 power-of-two scales and ties-to-even."""
    grouped = rows.float().unflatten(-1, (-1, 32))
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
    decoded = codes.float().unflatten(-1, (-1, 32))
    decoded.mul_(scale.unsqueeze(-1))
    return decoded.flatten(-2).bfloat16()


def pack_engram_int8_rows(codes, scale):
    """Pack INT8 codes and FP32 group scales as one row-oriented wire buffer."""
    if codes.dtype != torch.int8 or scale.dtype != torch.float32:
        raise TypeError("Engram INT8 wire packing expects int8 codes and FP32 scales")
    return torch.cat((codes.view(torch.uint8), scale.view(torch.uint8)), dim=-1).contiguous()


def unpack_engram_int8_rows(payload, width):
    """Decode the packed INT8 wire buffer without changing BF16 lookup semantics."""
    groups = width // 32
    expected = width + groups * 4
    if payload.dtype != torch.uint8 or payload.shape[-1] != expected:
        raise ValueError(f"Invalid Engram INT8 wire payload: {tuple(payload.shape)}")
    codes = payload[..., :width].contiguous().view(torch.int8)
    scale = payload[..., width:].contiguous().view(torch.float32).reshape(*payload.shape[:-1], groups)
    return dequantize_engram_rows(codes, scale)


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
        # probes keep CPU metadata. Callers may override this for rollback.
        backend = str(dist.get_backend(group)).lower()
        self.metadata_on_device = backend not in ("gloo", "mpi")
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.is_source = dist.get_rank() == tp_source

    @classmethod
    def from_vllm(cls, parallel):
        # Lazy imports keep the transport usable in standalone distributed probes.
        from vllm.distributed import get_ep_group, get_tp_group

        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or not parallel.enable_expert_parallel
        ):
            raise ValueError("Engram HBM sharing requires EP and PP=PCP=DCP=1")
        ep, tp = get_ep_group(), get_tp_group()
        hosts = [None] * ep.world_size
        dist.all_gather_object(hosts, socket.gethostname(), group=ep.cpu_group)
        node_groups = [[ep.ranks[i] for i, host in enumerate(hosts) if host == name] for name in dict.fromkeys(hosts)]
        node_sizes = {len(ranks) for ranks in node_groups}
        if len(node_sizes) != 1:
            raise ValueError(f"Engram requires equal rank counts on every node: {node_groups}")
        selected = None
        for ranks in node_groups:
            # Every world rank creates groups in the same order.
            cpu = dist.new_group(ranks, backend="gloo")
            device = dist.new_group(ranks, backend=dist.get_backend(ep.device_group))
            if dist.get_rank() in ranks:
                if not set(tp.ranks).issubset(ranks):
                    raise ValueError("Engram requires each TP group to stay within one node")
                selected = cls(device, cpu, tp.device_group, tp.ranks[0])
        return selected


class NodeShardedEngram(nn.Module):
    """Contiguous row shards; only BF16 rows cross the node-local fabric."""

    def __init__(self, rows, width, query_group, device=None, storage_format="bf16"):
        super().__init__()
        if storage_format not in ("bf16", "int8", "fp8", "mxfp8"):
            raise ValueError("Engram storage_format must be bf16, int8, fp8, or mxfp8")
        if storage_format in ("int8", "fp8", "mxfp8") and width % 32:
            raise ValueError("INT8 Engram requires a width divisible by 32")
        self.storage_format = storage_format
        self.rows, self.width = rows, width
        self.query_group = query_group
        # Kept opt-in while the reduced-payload protocol is benchmarked.
        self.compressed_int8_wire = False
        # The fused gather/dequant kernel wins even for one row on A3; retain
        # the threshold as a local rollback knob for future kernel changes.
        self.use_triton_int8 = True
        self.triton_int8_min_rows = 1
        self.offload_pinned = storage_format in ("fp8", "mxfp8")
        self._offload_buffers = OrderedDict()
        self._offload_buffer_bytes = 0
        self._offload_buffer_bytes_limit = _OFFLOAD_BUFFER_BYTES_LIMIT
        self._offload_buffer_index = {}
        self._offload_events = {}
        # Ceil partition leaves at most size-1 unused rows, never a replica.
        self.shard_rows = (rows + query_group.size - 1) // query_group.size
        self.start = query_group.rank * self.shard_rows
        self.end = min(self.start + self.shard_rows, rows)
        if self.start >= rows:
            raise ValueError("Engram table must have at least one row per rank")
        self._empty_flat = torch.empty(0, dtype=torch.int64, device="cpu")
        # Reuse fixed-size HCCL metadata buffers across requests.
        self._metadata_device_buffers = {}
        self._empty_metadata = torch.zeros(query_group.size + 1, dtype=torch.int64, device="cpu")
        self.weight = nn.Parameter(
            torch.empty(
                self.end - self.start,
                width,
                dtype=(
                    torch.int8
                    if storage_format == "int8"
                    else (torch.float8_e4m3fn if storage_format in ("fp8", "mxfp8") else torch.bfloat16)
                ),
                device=(torch.device("cpu") if storage_format in ("fp8", "mxfp8") else device),
                pin_memory=False,
            ),
            requires_grad=False,
        )
        if storage_format == "int8":
            self.register_buffer(
                "weight_scale", torch.empty(self.end - self.start, width // 32, dtype=torch.float32, device=device)
            )
        elif storage_format in ("fp8", "mxfp8"):
            self.register_buffer(
                "weight_scale",
                torch.empty(
                    self.end - self.start,
                    width // 32,
                    dtype=torch.float8_e8m0fnu,
                    device="cpu",
                    pin_memory=False,
                ),
            )

    def set_rows(self, start, rows):
        """Load BF16 rows into local storage without allocating a BF16 table copy."""
        end = start + rows.shape[0]
        if self.storage_format == "int8":
            codes, scales = quantize_engram_rows(rows.to(self.weight.device))
            if not bool((torch.isfinite(scales) & (scales > 0)).all()):
                raise ValueError("INT8 Engram requires finite positive group scales")
            self.weight.data[start:end].copy_(codes)
            self.weight_scale[start:end].copy_(scales)
        else:
            self.weight.data[start:end].copy_(rows)

    def lookup_local(self, ids, *, pin_output=False):
        # Idle DP replicas still enter routing collectives, but must not launch
        # gather/dequant kernels for an empty owner request.
        if ids.numel() == 0:
            return torch.empty((*ids.shape, self.width), dtype=torch.bfloat16, device=self.weight.device)
        original_shape = ids.shape
        flat_ids = ids.reshape(-1)
        if self.storage_format == "int8":
            if (
                self.use_triton_int8
                and self.weight.device.type == "npu"
                and self.width == 256
                and flat_ids.device.type == "npu"
                and flat_ids.shape[0] >= self.triton_int8_min_rows
            ):
                # Importing the ops package initializes the active Triton
                # backend, so keep it out of CPU-only routing and test workers.
                from vllm_ascend.ops.triton.engram_int8 import gather_dequantize_engram_int8

                rows = gather_dequantize_engram_int8(self.weight, self.weight_scale, flat_ids)
            else:
                codes = torch.index_select(self.weight, 0, flat_ids)
                scales = torch.index_select(self.weight_scale, 0, flat_ids)
                rows = dequantize_engram_rows(codes, scales)
        elif self.storage_format in ("fp8", "mxfp8"):
            # index_select avoids the extra advanced-indexing wrapper on the
            # CPU-resident PLE table and keeps row selection explicit.
            rows = torch.index_select(self.weight, 0, flat_ids)
            decoded = rows.float().reshape(-1, self.width // 32, 32)
            scales = torch.index_select(self.weight_scale, 0, flat_ids)
            decoded.mul_(scales.float().unsqueeze(-1))
            decoded = decoded.reshape(-1, self.width)
            if self.offload_pinned and pin_output:
                key = decoded.shape[0]
                slots = self._offload_buffers.get(key)
                if slots is None:
                    slot_bytes = 2 * key * self.width * _BF16_BYTES
                    # A single shape may be larger than the configured cache
                    # budget.  It cannot be split without changing the lookup
                    # contract, so keep that one shape as an explicit bound
                    # exception after evicting all prior shapes.
                    while self._offload_buffers and (
                        len(self._offload_buffers) >= _OFFLOAD_BUFFER_CACHE_SIZE
                        or self._offload_buffer_bytes + slot_bytes > self._offload_buffer_bytes_limit
                    ):
                        evicted, evicted_slots = self._offload_buffers.popitem(last=False)
                        self._offload_buffer_bytes -= 2 * evicted * self.width * _BF16_BYTES
                        self._offload_buffer_index.pop(evicted, None)
                        for slot in evicted_slots:
                            event = self._offload_events.pop(slot.data_ptr(), None)
                            if event is not None:
                                event.synchronize()
                    slots = [torch.empty((key, self.width), dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
                    self._offload_buffers[key] = slots
                    self._offload_buffer_bytes += slot_bytes
                    self._offload_buffer_index[key] = 0
                else:
                    self._offload_buffers.move_to_end(key)
                index = self._offload_buffer_index[key]
                decoded_slot = slots[index]
                self._offload_buffer_index[key] = 1 - index
                event = self._offload_events.pop(decoded_slot.data_ptr(), None)
                if event is not None:
                    event.synchronize()
                # Convert directly into the reusable pinned destination.  The
                # previous expression materialized a separate BF16 tensor
                # before this copy, doubling the temporary decoded allocation.
                decoded_slot.copy_(decoded)
                rows = decoded_slot
            else:
                rows = decoded.bfloat16()
        else:
            rows = torch.index_select(self.weight, 0, flat_ids)
        return rows.view(*original_shape, self.width)

    def _record_offload_use(self, source_ptr, device):
        """Keep a pinned staging slot alive until the submitted device work ends."""
        if not self.offload_pinned or device.type != "npu":
            return
        event = torch.npu.Event()
        event.record(torch.npu.current_stream(device))
        self._offload_events[source_ptr] = event

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        """Load BF16, INT8, FP8, or MXFP8 Engram tensors with bounded IO.

        FP8/MXFP8 remain CPU resident (PLE_OFFLOAD); only decoded BF16 rows
        enter the node-local all-to-all response buffer.
        """
        root = Path(model_path)
        index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
        scale_key = key.removesuffix(".weight") + ".scale"
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            if tensor.get_shape() != [self.rows, self.width]:
                raise ValueError(f"{key}: expected BF16/FP8 [{self.rows}, {self.width}]")
            source_dtype = tensor.get_dtype()
            if self.storage_format == "int8" and source_dtype in ("I8", "INT8"):
                if scale_key not in index:
                    raise ValueError(f"{key}: INT8 source requires .scale")
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    if scale.get_shape() != [self.rows, self.width // 32] or scale.get_dtype() != "F32":
                        raise ValueError(f"{scale_key}: expected FP32 [{self.rows}, {self.width // 32}]")
                    for start in range(self.start, self.end, chunk_rows):
                        stop = min(start + chunk_rows, self.end)
                        self.weight.data[start - self.start : stop - self.start].copy_(tensor[start:stop])
                        self.weight_scale[start - self.start : stop - self.start].copy_(scale[start:stop])
                return
            if self.storage_format in ("fp8", "mxfp8"):
                if source_dtype not in ("F8_E4M3", "F8_E4M3FN") or scale_key not in index:
                    raise ValueError(f"{key}: {self.storage_format} requires FP8 weight and .scale")
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    if scale.get_shape() != [self.rows, self.width // 32]:
                        raise ValueError(f"{scale_key}: expected [{self.rows}, {self.width // 32}]")
                    for start in range(self.start, self.end, chunk_rows):
                        stop = min(start + chunk_rows, self.end)
                        self.weight.data[start - self.start : stop - self.start].copy_(tensor[start:stop])
                        self.weight_scale[start - self.start : stop - self.start].copy_(scale[start:stop])
                return
            if source_dtype != "BF16":
                raise ValueError(f"{key}: expected BF16 source for {self.storage_format}")
            for start in range(self.start, self.end, chunk_rows):
                stop = min(start + chunk_rows, self.end)
                self.set_rows(start - self.start, tensor[start:stop])

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
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("Engram routing expects CPU int64 IDs")
        if routing is None:
            flat, order, metadata = self._metadata(ids)
        else:
            flat, order, metadata = routing
        counts = [row.tolist() for row in gathered]
        if any(row[-1] for row in counts):
            raise IndexError("Engram hash ID outside table")
        send = metadata[:-1].tolist()
        recv = [row[q.rank] for row in counts]
        total_recv = sum(recv)
        total_requests = sum(send)
        total_global = sum(sum(row[:-1]) for row in counts)
        backend = str(dist.get_backend(q.group)).lower()
        # HCCL collectives require NPU tensors even when the compressed table is CPU resident.
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
            if self.storage_format == "int8" and self.compressed_int8_wire:
                lookup_ids = local_ids.cpu() if self.weight.device.type == "cpu" else local_ids
                codes = torch.index_select(self.weight, 0, lookup_ids)
                scales = torch.index_select(self.weight_scale, 0, lookup_ids)
                packed = pack_engram_int8_rows(codes.to(device), scales.to(device))
                wire_width = self.width + (self.width // 32) * 4
                returned = torch.empty(total_requests * wire_width, dtype=torch.uint8, device=device)
                wire_send = [count * wire_width for count in send]
                wire_recv = [count * wire_width for count in recv]
                dist.all_to_all_single(returned, packed.flatten(), wire_send, wire_recv, group=q.group)
                returned = unpack_engram_int8_rows(returned.reshape(total_requests, wire_width), self.width)
            else:
                local_values = self.lookup_local(
                    local_ids.cpu() if self.weight.device.type == "cpu" else local_ids,
                    pin_output=device.type == "npu",
                )
                source_ptr = local_values.data_ptr()
                values = local_values.to(
                    device=device, dtype=torch.bfloat16, non_blocking=self.offload_pinned
                ).contiguous()
                returned = torch.empty((total_requests, self.width), dtype=torch.bfloat16, device=device)
                dist.all_to_all_single(returned, values, send, recv, group=q.group)
                self._record_offload_use(source_ptr, device)
        else:
            returned = torch.empty((0, self.width), dtype=torch.bfloat16, device=device)
        if output is None:
            result = torch.empty((ids.numel(), self.width), dtype=torch.bfloat16, device=device)
        else:
            result = output
            if result.shape != (ids.numel(), self.width) or result.device != device:
                raise ValueError("Engram output buffer has an incompatible shape or device")
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
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("Engram routing expects CPU int64 IDs")
        _, _, metadata = self._metadata(ids)
        if q.metadata_on_device:
            device = self.weight.device if self.weight.device.type == "npu" else torch.device("npu")
            gathered = self._gather_metadata_device(metadata, device, q.group)
        else:
            gathered = [torch.empty_like(metadata) for _ in range(q.size)]
            dist.all_gather(gathered, metadata, group=q.cpu_group)
        return self._forward_with_gathered(ids, gathered)

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
        if len(tables) != len(ids_list):
            raise ValueError("tables and ids_list must have the same length")
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
        outputs = []
        offset = 0
        for result in results:
            size = result.numel()
            outputs.append(combined[offset : offset + size].view_as(result))
            offset += size
        return outputs
