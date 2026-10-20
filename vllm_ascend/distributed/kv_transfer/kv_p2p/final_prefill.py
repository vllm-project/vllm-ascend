# SPDX-License-Identifier: Apache-2.0
"""Bounded final or incremental export of ordinary BF16 SFA caches.

The producer cache (including shared prefix pages) is read-only here. Only the
registered staging tensor is recycled, after each synchronous Mooncake write.
Logical block IDs are retained until this boundary: main and indexer components
can have different kernel page sizes and padded physical strides.
"""

import math

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent

BF16_TO_TQ4 = "bf16_to_tq4"
BF16_TO_BF16 = "bf16_to_bf16"
LATENT_DIM = 512
ROPE_DIM = 64
PACKED_DIM = LATENT_DIM // 2
TQ_SLOT_BYTES = PACKED_DIM + ROPE_DIM * 2 + 2
TQ_ABI = "glm_sfa_tq4_nc_v1"
TRANSFER_ALIGNMENT = 2 * 1024**2


def final_prefill_mode(config):
    extra = config.kv_transfer_config.kv_connector_extra_config
    value = extra.get("final_prefill_export") if isinstance(extra, dict) else None
    if value is None:
        return None
    if value not in (BF16_TO_TQ4, BF16_TO_BF16):
        raise ValueError("final_prefill_export must be bf16_to_tq4 or bf16_to_bf16")
    parallel = config.parallel_config
    if (parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1
            or parallel.pipeline_parallel_size != 1 or config.speculative_config is not None):
        raise ValueError("Final-prefill export requires PCP=DCP=PP=1 and no speculative decoding")
    transfer = config.kv_transfer_config
    if transfer.is_kv_producer == transfer.is_kv_consumer:
        raise ValueError("Final-prefill export requires separate kv_producer and kv_consumer roles")
    expected_dtype = "turboquant_4bit_nc" if transfer.is_kv_consumer and value == BF16_TO_TQ4 else "auto"
    if config.cache_config.cache_dtype != expected_dtype or config.model_config.dtype != torch.bfloat16:
        raise ValueError(f"Final-prefill export requires BF16 model dtype and kv-cache-dtype={expected_dtype}")
    if not config.model_config.enforce_eager:
        if transfer.is_kv_producer or config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise ValueError("Final-prefill export requires eager P; D supports eager or FULL_DECODE_ONLY")
    return value


def pack_bf16_kv(nope, rope, quantizer):
    """Input is already RMS-normalized / RoPE-transformed, in original coordinates."""
    rows = quantizer.compress(nope.reshape(-1, LATENT_DIM)).reshape(-1, PACKED_DIM + 2)
    scale = rows[:, PACKED_DIM:].contiguous().view(torch.float16).reshape(-1, 1)
    rope = (rope.reshape(-1, ROPE_DIM).float() / (scale.float() + 1e-20)).to(torch.bfloat16)
    return torch.cat((rows[:, :PACKED_DIM], rope.contiguous().view(torch.uint8), rows[:, PACKED_DIM:]), -1)


def describe_layout(meta, tensors, spec, mode, is_producer):
    """Publish the byte ABI, including stride, rather than infer it from size."""
    if getattr(spec, "store_on_host", False):
        raise ValueError("Final-prefill export does not support host-offloaded cache pages")
    if any(t.ndim != 4 or t.shape[2] != 1 or not t[0].is_contiguous() for t in tensors):
        raise ValueError("Final-prefill export supports ND SFA pages [blocks, tokens, 1, dim] only")
    dims = [t.shape[-1] for t in tensors]
    if getattr(spec, "cache_sparse_sfa_turboquant", False):
        if is_producer or mode != BF16_TO_TQ4 or dims != [TQ_SLOT_BYTES] or tensors[0].element_size() != 1:
            raise ValueError("Invalid TQ4 destination layout")
        cache_format = TQ_ABI
    elif "Indexer" in type(spec).__name__:
        cache_format = "sfa_indexer_v1"
    else:
        if dims != [LATENT_DIM, ROPE_DIM] or any(t.dtype != torch.bfloat16 for t in tensors):
            raise ValueError("Final-prefill export requires GLM SFA BF16 latent512 + RoPE64")
        cache_format = "sfa_bf16_v1"
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


def validate_remote_layout(local, remote):
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
    if local.tensor_group_idx[0] != remote.tensor_group_idx[0]:
        raise ValueError("P/D cache group ordering must match")
    if local.cache_format == "sfa_bf16_v1" and local.export_mode == BF16_TO_TQ4:
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
        if mode == BF16_TO_TQ4:
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
        if mode == BF16_TO_TQ4 and incremental:
            self._second_storage = torch.empty(size + TRANSFER_ALIGNMENT, dtype=torch.uint8, device=self.device)
            offset = (-self._second_storage.data_ptr()) % TRANSFER_ALIGNMENT
            second = self._second_storage[offset:offset + size].view(chunk_tokens, TQ_SLOT_BYTES)
            if engine.register_memory(second.data_ptr(), second.numel()) != 0:
                raise RuntimeError("Final-prefill staging memory registration failed")
            self.staging_buffers.append(second)
            self.staging_slots = torch.arange(chunk_tokens, dtype=torch.int64, device=self.device)
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
                if local.cache_format == "sfa_bf16_v1" and self.mode == BF16_TO_TQ4:
                    self._convert_and_write(name, local, remote, src_ids, dst_ids, session)
                else:
                    src, dst, lengths = [], [], []
                    for i, scale in enumerate(local.block_size_scale):
                        for source, destination in zip(kernel_ids(src_ids, scale), kernel_ids(dst_ids, scale)):
                            src.append(local.kv_caches_base_addr[i] + source * local.block_stride[i])
                            dst.append(remote.kv_caches_base_addr[i] + destination * remote.block_stride[i])
                            lengths.append(local.block_len[i])
                    self._write(session, src, dst, lengths)

    def _convert_and_write(self, name, local, remote, src_ids, dst_ids, session):
        if getattr(self, "incremental", False):
            try:
                return self._pipeline_convert_and_write(name, local, remote, src_ids, dst_ids, session)
            except Exception:
                # A later buffer can still be reading source pages when an
                # earlier transfer fails. Retire it before ownership is released.
                self.stream.synchronize()
                raise
        pages_per_chunk = self.chunk_tokens // local.logical_block_size
        for start in range(0, len(src_ids), pages_per_chunk):
            source = src_ids[start:start + pages_per_chunk]
            destination = dst_ids[start:start + pages_per_chunk]
            with torch.npu.stream(self.stream):
                parts = []
                for tensor, scale in zip(self.caches[name], local.block_size_scale):
                    indices = torch.tensor(kernel_ids(source, scale), device=self.device, dtype=torch.int64)
                    parts.append(tensor.index_select(0, indices).reshape(-1, tensor.shape[-1]))
                packed = pack_bf16_kv(*parts, self.quantizer)
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

    def _pipeline_convert_and_write(self, name, local, remote, src_ids, dst_ids, session):
        """Queue conversion N+1 before the blocking transfer of buffer N.

        At most two registered buffers are live; a buffer is reused only after
        its synchronous transfer completed. Persistent BF16 source pages remain
        owned until the model-step send queue is drained.
        """
        pages = self.chunk_tokens // local.logical_block_size
        page_bytes = math.prod(remote.component_shapes[0])
        pending = None

        def send(item):
            event, buffer, destinations = item
            event.synchronize()
            remote_ids = kernel_ids(destinations, remote.block_size_scale[0])
            self._write(session, [buffer.data_ptr() + i * page_bytes for i in range(len(remote_ids))],
                        [remote.kv_caches_base_addr[0] + block * remote.block_stride[0] for block in remote_ids],
                        [page_bytes] * len(remote_ids))

        for index, start in enumerate(range(0, len(src_ids), pages)):
            buffer = self.staging_buffers[index % 2]
            source = src_ids[start:start + pages]
            with torch.npu.stream(self.stream):
                parts = []
                for tensor, scale in zip(self.caches[name], local.block_size_scale):
                    indices = torch.tensor(kernel_ids(source, scale), device=self.device, dtype=torch.int64)
                    parts.append(tensor.index_select(0, indices).reshape(-1, tensor.shape[-1]))
                self.quantizer.compress_kv_into(*parts, self.staging_slots[:parts[0].shape[0]], buffer)
                event = torch.npu.Event()
                event.record()
            if pending is not None:
                send(pending)
            pending = (event, buffer, dst_ids[start:start + pages])
        if pending is not None:
            send(pending)
