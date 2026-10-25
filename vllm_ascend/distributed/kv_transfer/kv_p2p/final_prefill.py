# SPDX-License-Identifier: Apache-2.0
"""Bounded final or incremental export of ordinary C8 SFA caches.

The producer cache (including shared prefix pages) is read-only here. Only the
registered staging tensor is recycled, after each synchronous Mooncake write.
Logical block IDs are retained until this boundary: main and indexer components
can have different kernel page sizes and padded physical strides.
"""

import math
from dataclasses import dataclass, field

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE, get_sfa_qsfa_packed_head_dim

from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent, pack_c8_kv

C8_TO_TQ4 = "c8_to_tq4"
LATENT_DIM = 512
ROPE_DIM = 64
PACKED_DIM = LATENT_DIM // 2
TQ_SLOT_BYTES = PACKED_DIM + ROPE_DIM * 2 + 2
C8_SLOT_BYTES = get_sfa_qsfa_packed_head_dim(LATENT_DIM, ROPE_DIM)
C8_ABI = "glm_sfa_c8_v1"
TQ_ABI = "glm_sfa_tq4_nc_v1"
TRANSFER_ALIGNMENT = 2 * 1024**2


def final_prefill_mode(config):
    extra = config.kv_transfer_config.kv_connector_extra_config
    value = extra.get("final_prefill_export") if isinstance(extra, dict) else None
    if value is None:
        return None
    if value != C8_TO_TQ4:
        raise ValueError("final_prefill_export must be c8_to_tq4")
    parallel = config.parallel_config
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("Final-prefill export requires unsharded SFA KV pages (PCP=DCP=1)")
    if config.kv_transfer_config.kv_connector != "MooncakeConnectorV1" and (
            parallel.pipeline_parallel_size != 1 or config.speculative_config is not None):
        raise ValueError("PP or speculative final-prefill export requires MooncakeConnectorV1")
    transfer = config.kv_transfer_config
    if transfer.is_kv_producer == transfer.is_kv_consumer:
        raise ValueError("Final-prefill export requires separate kv_producer and kv_consumer roles")
    expected_dtype = "turboquant_4bit_nc" if transfer.is_kv_consumer else "int8"
    if config.cache_config.cache_dtype != expected_dtype or config.model_config.dtype != torch.bfloat16:
        raise ValueError(f"Final-prefill export requires BF16 model dtype and kv-cache-dtype={expected_dtype}")
    if not config.model_config.enforce_eager:
        if transfer.is_kv_producer or config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise ValueError("Final-prefill export requires eager P; D supports eager or FULL_DECODE_ONLY")
    return value


@dataclass
class ExportLayerMetadata:
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]
    export_mode: str | None = None
    cache_format: str | None = None
    logical_block_size: int = 0
    component_shapes: list[list[int]] = field(default_factory=list)
    component_dtypes: list[str] = field(default_factory=list)
    block_stride: list[int] = field(default_factory=list)
    num_blocks: list[int] = field(default_factory=list)


def describe_layout(meta, tensors, spec, mode, is_producer):
    """Publish the byte ABI, including stride, rather than infer it from size."""
    if getattr(spec, "store_on_host", False):
        raise ValueError("Final-prefill export does not support host-offloaded cache pages")
    if any(t.ndim != 4 or t.shape[2] != 1 or not t[0].is_contiguous() for t in tensors):
        raise ValueError("Final-prefill export supports ND SFA pages [blocks, tokens, 1, dim] only")
    dims = [t.shape[-1] for t in tensors]
    if getattr(spec, "cache_sparse_sfa_turboquant", False):
        if is_producer or mode != C8_TO_TQ4 or dims != [TQ_SLOT_BYTES] or tensors[0].element_size() != 1:
            raise ValueError("Invalid TQ4 destination layout")
        cache_format = TQ_ABI
    elif "Indexer" in type(spec).__name__:
        cache_format = "sfa_indexer_v1"
    elif getattr(spec, "cache_sparse_sfa_c8", False):
        if not is_producer or dims != [C8_SLOT_BYTES] or tensors[0].dtype != torch.int8:
            raise ValueError("Final-prefill export requires the SFA INT8 packed source layout")
        cache_format = C8_ABI
    else:
        # Auxiliary layers that are not quantized retain their original ABI.
        if dims != [LATENT_DIM, ROPE_DIM] or any(t.dtype != torch.bfloat16 for t in tensors):
            raise ValueError("Unsupported passthrough cache layout")
        cache_format = "sfa_float_v1"
    meta.export_mode = mode
    meta.cache_format = cache_format
    meta.logical_block_size = spec.block_size
    meta.component_shapes = [list(t.shape[1:]) for t in tensors]
    meta.component_dtypes = [str(t.dtype) for t in tensors]
    meta.block_stride = [t.stride(0) * t.element_size() for t in tensors]
    meta.num_blocks = [t.shape[0] for t in tensors]
    for i, tensor in enumerate(tensors):
        if tensor.shape[1] * meta.block_size_scale[i] != spec.block_size:
            raise ValueError("Unsupported kernel/logical block size relationship")
        if meta.block_stride[i] < meta.block_len[i]:
            raise ValueError("Overlapping cache pages are not supported")


def validate_remote_layout(local, remote, *, match_group=True):
    components = len(remote.kv_caches_base_addr)
    fields = (remote.tensor_group_idx, remote.block_len, remote.block_size_scale, remote.component_shapes,
              remote.component_dtypes, remote.block_stride, remote.num_blocks)
    if not components or any(len(field) != components for field in fields):
        raise ValueError("Incomplete destination cache layout")
    if any(scale <= 0 or remote.logical_block_size % scale for scale in remote.block_size_scale):
        raise ValueError("Invalid destination kernel page scale")
    if any(stride < length or length <= 0 for stride, length in zip(remote.block_stride, remote.block_len)):
        raise ValueError("Invalid destination page stride")
    if remote.export_mode != local.export_mode or remote.logical_block_size != local.logical_block_size:
        raise ValueError("P/D final-prefill mode and logical block sizes must match")
    if match_group and local.tensor_group_idx[0] != remote.tensor_group_idx[0]:
        raise ValueError("P/D cache group ordering must match")
    if (local.cache_format == C8_ABI and local.export_mode == C8_TO_TQ4
            and remote.cache_format != C8_ABI):
        shape = [remote.logical_block_size // remote.block_size_scale[0], 1, TQ_SLOT_BYTES]
        if (remote.cache_format != TQ_ABI or remote.component_shapes != [shape]
                or remote.component_dtypes not in (["torch.int8"], ["torch.uint8"])
                or remote.block_len != [math.prod(shape)]):
            raise ValueError("D must publish the GLM TQ4 ABI")
        return
    if (remote.cache_format != local.cache_format or remote.component_shapes != local.component_shapes
            or remote.component_dtypes != local.component_dtypes or remote.block_len != local.block_len
            or remote.block_size_scale != local.block_size_scale):
        raise ValueError("P/D passthrough cache layouts must match")


def kernel_ids(block_ids, scale):
    return [block * scale + offset for block in block_ids for offset in range(scale)]


def validate_block_ids(ids, meta):
    for scale, count in zip(meta.block_size_scale, meta.num_blocks):
        if any(block < 0 or (block + 1) * scale > count for block in ids):
            raise ValueError("Transfer block ID outside registered cache")


class FinalPrefillExporter:
    """One per sending worker; called only by its serial transfer thread."""

    def __init__(self, caches, metadata, mode, chunk_tokens, engine, *, fused_finalize=False, incremental=False):
        if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
            raise ValueError("final_prefill_chunk_tokens must be a positive integer")
        self.incremental = incremental
        self.caches = caches
        self.metadata = metadata
        self.mode = mode
        self.engine = engine
        self.device = next(iter(caches.values()))[0].device
        self.stream = torch.npu.Stream(device=self.device)
        self.quantizer = TurboQuantLatent(fused_finalize=fused_finalize)
        self.chunk_tokens = chunk_tokens
        self.staging = None
        if mode == C8_TO_TQ4:
            largest_block = max(m.logical_block_size for m in metadata.values())
            if chunk_tokens < largest_block:
                raise ValueError("final_prefill_chunk_tokens must fit at least one logical cache block")
            self.quantizer._initialize(self.device)
            # HIXL device registrations require an IPC-exportable aligned
            # address; a small allocator suballocation is not sufficient.
            size = chunk_tokens * TQ_SLOT_BYTES
            self._staging_storage = torch.empty(size + TRANSFER_ALIGNMENT, dtype=torch.uint8, device=self.device)
            offset = (-self._staging_storage.data_ptr()) % TRANSFER_ALIGNMENT
            self.staging = self._staging_storage[offset:offset + size].view(chunk_tokens, TQ_SLOT_BYTES)
            if engine.register_memory(self.staging.data_ptr(), self.staging.numel()) != 0:
                raise RuntimeError("Final-prefill staging memory registration failed")
        self.staging_buffers = [self.staging]
        if mode == C8_TO_TQ4 and incremental:
            self._second_storage = torch.empty(size + TRANSFER_ALIGNMENT, dtype=torch.uint8, device=self.device)
            offset = (-self._second_storage.data_ptr()) % TRANSFER_ALIGNMENT
            second = self._second_storage[offset:offset + size].view(chunk_tokens, TQ_SLOT_BYTES)
            if engine.register_memory(second.data_ptr(), second.numel()) != 0:
                raise RuntimeError("Final-prefill staging memory registration failed")
            self.staging_buffers.append(second)
        if mode == C8_TO_TQ4:
            # Load/compile the packer before serving requests. Doing this in
            # the serial export thread can block other D workers' handshakes.
            warmup = torch.zeros((largest_block, C8_SLOT_BYTES), dtype=torch.int8, device=self.device)
            warmup[:, :LATENT_DIM] = 1
            warmup[:, LATENT_DIM + ROPE_DIM * 2:] = torch.ones(
                (largest_block, LATENT_DIM // SFA_QSFA_TILE_SIZE), dtype=torch.float32, device=self.device
            ).view(torch.int8)
            pack_c8_kv(warmup, self.quantizer)
        # Transforms/staging are created on the model stream before worker use.
        self.ready = torch.npu.Event()
        self.ready.record()

    def _write(self, session, src, dst, lengths):
        if src and self.engine.batch_transfer_sync_write(session, src, dst, lengths) < 0:
            raise RuntimeError(f"Final-prefill transfer failed: {session}")

    @torch.inference_mode()
    def transfer(self, task):
        # Incremental mode uses the cache-write event; legacy final export
        # waits until Attention completes. Both protect main and indexer KV.
        task.wait_event.synchronize()
        self.ready.synchronize()
        for request in task.send_request.values():
            indices_cache = {}
            if not request.chunk_finish and not getattr(self, "incremental", False):
                raise ValueError("Final-prefill export received a non-final chunk")
            session = f"{request.remote_host}:{request.remote_te_rpc_port}"
            # Validate all components before writing any part of this layer.
            for name in task.layer_names:
                local = self.metadata[name]
                remote = request.remote_layer_metadata[name]
                validate_remote_layout(local, remote)
                group = local.tensor_group_idx[0]
                src_ids, dst_ids = request.local_block_ids[group], request.remote_block_ids[group]
                if len(src_ids) != len(dst_ids):
                    raise ValueError("Source and destination page counts differ")
                validate_block_ids(src_ids, local)
                validate_block_ids(dst_ids, remote)
            for name in task.layer_names:
                local = self.metadata[name]
                remote = request.remote_layer_metadata[name]
                group = local.tensor_group_idx[0]
                src_ids, dst_ids = request.local_block_ids[group], request.remote_block_ids[group]
                if remote.cache_format == TQ_ABI:
                    self._convert_and_write(name, local, remote, src_ids, dst_ids, session, indices_cache)
                else:
                    src, dst, lengths = [], [], []
                    for i, scale in enumerate(local.block_size_scale):
                        for source, destination in zip(kernel_ids(src_ids, scale), kernel_ids(dst_ids, scale)):
                            src.append(local.kv_caches_base_addr[i] + source * local.block_stride[i])
                            dst.append(remote.kv_caches_base_addr[i] + destination * remote.block_stride[i])
                            lengths.append(local.block_len[i])
                    self._write(session, src, dst, lengths)

    @torch.inference_mode()
    def export_layers(self, layers, session):
        """Export name-mapped logical pages supplied by a bulk connector.

        The caller owns the request's source pages until this returns. Validate
        the entire request first, including passthrough layers and MTP caches.
        Cache manager group numbers need not agree across PP stages or formats.
        """
        self.ready.synchronize()
        indices_cache = {}
        pending = None
        for name, (remote, source, destination) in layers.items():
            local = self.metadata[name]
            validate_remote_layout(local, remote, match_group=False)
            if len(source) != len(destination):
                raise ValueError("Source and destination page counts differ")
            validate_block_ids(source, local)
            validate_block_ids(destination, remote)
        try:
            for name, (remote, source, destination) in layers.items():
                local = self.metadata[name]
                if remote.cache_format == TQ_ABI:
                    if getattr(self, "incremental", False):
                        pending = self._pipeline_convert_and_write(
                            name, local, remote, source, destination, session, indices_cache,
                            pending=pending, finish=False,
                        )
                    else:
                        self._convert_and_write(name, local, remote, source, destination, session, indices_cache)
                else:
                    src, dst, lengths = [], [], []
                    for i, scale in enumerate(local.block_size_scale):
                        for s, d in zip(kernel_ids(source, scale), kernel_ids(destination, scale)):
                            src.append(local.kv_caches_base_addr[i] + s * local.block_stride[i])
                            dst.append(remote.kv_caches_base_addr[i] + d * remote.block_stride[i])
                            lengths.append(local.block_len[i])
                    self._write(session, src, dst, lengths)
            if pending is not None:
                self._send_converted(pending, session)
        finally:
            # A failed transfer must also retire any queued conversion before
            # the connector can release source pages or acknowledge failure.
            self.stream.synchronize()

    def _source_indices(self, src_ids, scales, indices_cache):
        # Request-local metadata only: never retain page IDs across requests.
        # Upload once per page mapping/scale and slice the same indices for each
        # layer/chunk. Keep these tensors alive until the export stream drains.
        if not src_ids:
            return []
        source = tuple(src_ids)
        with torch.npu.stream(self.stream):
            for scale in scales:
                key = (source, scale)
                if key not in indices_cache:
                    indices_cache[key] = torch.tensor(kernel_ids(source, scale), device=self.device, dtype=torch.int64)
        return [indices_cache[(source, scale)] for scale in scales]

    def _convert_and_write(self, name, local, remote, src_ids, dst_ids, session, indices_cache=None):
        if indices_cache is None:
            indices_cache = {}
        if getattr(self, "incremental", False):
            try:
                return self._pipeline_convert_and_write(name, local, remote, src_ids, dst_ids, session, indices_cache)
            except Exception:
                # A later buffer can still be reading source pages when an
                # earlier transfer fails. Retire it before ownership is released.
                self.stream.synchronize()
                raise
        pages_per_chunk = self.chunk_tokens // local.logical_block_size
        layer_indices = self._source_indices(src_ids, local.block_size_scale, indices_cache)
        for start in range(0, len(src_ids), pages_per_chunk):
            source = src_ids[start:start + pages_per_chunk]
            destination = dst_ids[start:start + pages_per_chunk]
            with torch.npu.stream(self.stream):
                parts = []
                for tensor, scale, all_indices in zip(self.caches[name], local.block_size_scale, layer_indices):
                    indices = all_indices[start * scale:(start + len(source)) * scale]
                    parts.append(tensor.index_select(0, indices).reshape(-1, tensor.shape[-1]))
                packed = pack_c8_kv(parts[0], self.quantizer)
                self.staging[:packed.shape[0]].copy_(packed)
            self.stream.synchronize()
            page_bytes = math.prod(remote.component_shapes[0])
            remote_ids = kernel_ids(destination, remote.block_size_scale[0])
            self._write(
                session,
                [self.staging.data_ptr() + i * page_bytes for i in range(len(remote_ids))],
                [remote.kv_caches_base_addr[0] + block * remote.block_stride[0] for block in remote_ids],
                [page_bytes] * len(remote_ids),
            )
            # No next conversion may overwrite staging until _write completes.

    def _send_converted(self, item, session):
        event, buffer_index, destinations, remote = item
        event.synchronize()
        buffer = self.staging_buffers[buffer_index]
        page_bytes = math.prod(remote.component_shapes[0])
        remote_ids = kernel_ids(destinations, remote.block_size_scale[0])
        self._write(session, [buffer.data_ptr() + i * page_bytes for i in range(len(remote_ids))],
                    [remote.kv_caches_base_addr[0] + block * remote.block_stride[0] for block in remote_ids],
                    [page_bytes] * len(remote_ids))

    def _pipeline_convert_and_write(self, name, local, remote, src_ids, dst_ids, session, indices_cache=None,
                                    *, pending=None, finish=True):
        """Queue conversion N+1 before the blocking transfer of buffer N.

        At most two registered buffers are live; a buffer is reused only after
        its synchronous transfer completed. Persistent C8 source pages remain
        owned until the model-step send queue is drained.
        """
        pages = self.chunk_tokens // local.logical_block_size
        if indices_cache is None:
            indices_cache = {}
        layer_indices = self._source_indices(src_ids, local.block_size_scale, indices_cache)
        for start in range(0, len(src_ids), pages):
            buffer_index = 0 if pending is None else (pending[1] + 1) % len(self.staging_buffers)
            buffer = self.staging_buffers[buffer_index]
            source = src_ids[start:start + pages]
            with torch.npu.stream(self.stream):
                parts = []
                for tensor, scale, all_indices in zip(self.caches[name], local.block_size_scale, layer_indices):
                    indices = all_indices[start * scale:(start + len(source)) * scale]
                    parts.append(tensor.index_select(0, indices).reshape(-1, tensor.shape[-1]))
                packed = pack_c8_kv(parts[0], self.quantizer)
                buffer[:packed.shape[0]].copy_(packed)
                event = torch.npu.Event()
                event.record()
            if pending is not None:
                self._send_converted(pending, session)
            pending = (event, buffer_index, dst_ids[start:start + pages], remote)
        if finish:
            if pending is not None:
                self._send_converted(pending, session)
            return None
        return pending
