# SPDX-License-Identifier: Apache-2.0
"""NOVA-KV 2-bit attention with paged history and BF16 exact windows."""

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import regex as re
import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf

from vllm_ascend.ascend_config import NovaKVConfig
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.compilation.updatable_graph import get_capture_resource
from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.nova_kv import nova_exact_window_update, nova_prefill_attention, nova_quant_cache
from vllm_ascend.ops.triton.nova_kv.cache import ExactWindowPool, planar_cache_views
from vllm_ascend.ops.triton.nova_kv.encode import encode_decode_cache, prepare_encoder
from vllm_ascend.ops.triton.nova_kv.exact import (
    exact_partial,
    exact_partial_grouped,
    exact_physical,
    merge_partial_into,
)
from vllm_ascend.ops.triton.nova_kv.query import prepare_query_projection, project_query
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec, cache_views, packet_bytes

_EXACT_TILE = 64
_ENCODE_GROUP = 1
_ENCODE_OVERLAP = True
_EXACT_OVERLAP = True
_PREFILL_QUERY_TOKENS = 8192
_PREFILL_KV_TOKENS = 32768
_LONG_CONTEXT_SPLIT_TOKENS = 4096
_LONG_CONTEXT_MIN_TOKENS = 65536


@lru_cache(maxsize=8)
def _encode_stream(device):
    return torch.npu.Stream(device=device)


@triton.jit
def _batch_bounds(
    SEQ, SLOTS, LENGTHS, STARTS, ENDS, POSITIONS, LIMIT: tl.constexpr, SINK: tl.constexpr, RING: tl.constexpr
):
    batch = tl.program_id(0)
    slot = tl.load(SLOTS + batch)
    length = tl.where((slot >= 0) & (slot < LIMIT), tl.load(SEQ + batch), 0)
    start = tl.minimum(length, SINK)
    tl.store(LENGTHS + batch, length)
    tl.store(STARTS + batch, start)
    tl.store(ENDS + batch, tl.maximum(start, length - RING))
    tl.store(POSITIONS + batch, length - 1)


class NovaMetadataBuilder(AscendAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.slot_mapping = common_attn_metadata.slot_mapping
        return metadata


class NovaAttentionBackend(AscendAttentionBackend):
    uses_nova_kv_cache = True

    @staticmethod
    def get_impl_cls():
        return NovaAttentionImpl

    @staticmethod
    def reshape_cache(raw, spec, kernel_block_size):
        return cache_views(raw, spec, kernel_block_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(128)]

    @classmethod
    def customize_spec(cls, spec):
        return replace(spec, dtype=torch.uint8, state_content_bytes=packet_bytes(spec.head_size))

    @staticmethod
    def get_builder_cls():
        return NovaMetadataBuilder


class NovaAttentionImpl(AscendAttentionBackendImpl):
    """Encode KV history and combine compressed and exact attention segments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_nova_state(self.vllm_config)

    def _initialize_nova_state(self, config):
        if not is_950():
            raise ValueError("NOVA-KV currently requires Ascend 950")
        self._options = options = NovaKVConfig(**config.additional_config["nova_kv"])
        block_size = config.cache_config.block_size
        if block_size < 128 or block_size % 128:
            raise ValueError("NOVA-KV requires cache block_size divisible by 128")
        if self.head_size != 256 or not 1 <= self.num_queries_per_kv <= 6:
            raise ValueError("NOVA-KV requires head_size=256 and <=6 query groups")
        if self.sliding_window is not None or self.alibi_slopes is not None or self.sinks is not None:
            raise ValueError("NOVA-KV does not support sliding windows, ALiBi or attention sinks")
        if self.attn_type != AttentionType.DECODER or self.kv_sharing_target_layer_name is not None:
            raise ValueError("NOVA-KV requires full decoder attention without cross-layer KV sharing")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("NOVA exact-window ownership requires prefix caching disabled")
        self.nova_codec = self._window = self._causal_mask = None
        self._limit = config.scheduler_config.max_num_seqs
        self._sink, self._recent = options.sink_tokens, options.recent_tokens
        self._prefill_query_tile = _PREFILL_QUERY_TOKENS
        self._prefill_kv_tile = _PREFILL_KV_TOKENS

    def _codec(self, layer, device):
        if self.nova_codec is None:
            layer_id = int(re.search(r"layers\.(\d+)", layer.layer_name).group(1))
            self.nova_codec = NovaCodec.load(
                str(Path(self._options.artifact_path) / f"layer_{layer_id:02d}.pt"),
                device,
                get_tensor_model_parallel_rank(),
                get_tensor_model_parallel_world_size(),
                self.num_kv_heads,
            )
        codec = self.nova_codec
        if not hasattr(codec, "_decode_books"):
            codec._decode_books = codec._books_float.to(torch.float16)
        if _ENCODE_GROUP:
            prepare_encoder(codec)
        if _ENCODE_OVERLAP or _EXACT_OVERLAP:
            _encode_stream(device)
        prepare_query_projection(codec)
        return codec

    def _ensure_runtime(self, device, dtype):
        if self._window is None:
            self._window = ExactWindowPool(
                self._limit,
                self.num_kv_heads,
                self.head_size,
                self._sink,
                self._recent,
                device,
                dtype,
            )
            self._causal_mask = torch.ones((2048, 2048), device=device, dtype=torch.bool).triu_(1)

    def _request_slots(self, metadata, batch):
        if metadata.seq_lens_gpu is None or metadata.nova_request_slots is None:
            raise RuntimeError("NOVA needs persistent device sequence lengths and request slots")
        return metadata.nova_request_slots[:batch]

    def _prefill(self, query, key, value, views, metadata, output):
        batch = len(metadata.seq_lens_list)
        slots = metadata.nova_request_slots_cpu
        if slots is None:
            raise RuntimeError("NOVA needs host request slots for prefill")
        slots = slots + [-1] * (batch - len(slots))
        decode_requests, decode_rows = [], []
        first = 0
        for request, end in enumerate(metadata.actual_seq_lengths_q):
            if end - first == 1 and metadata.seq_lens_list[request] > 1 and 0 <= slots[request] < self._limit:
                decode_requests.append(request)
                decode_rows.append(first)
            first = end
        if decode_requests:
            requests = torch.tensor(decode_requests, device=query.device, dtype=torch.long)
            rows = torch.tensor(decode_rows, device=query.device, dtype=torch.long)
            decode_metadata = SimpleNamespace(
                seq_lens_list=[metadata.seq_lens_list[i] for i in decode_requests],
                seq_lens_gpu=metadata.seq_lens_gpu.index_select(0, requests),
                block_tables=metadata.block_tables.index_select(0, requests),
            )
            decoded = torch.empty_like(query.index_select(0, rows))
            self._decode(
                query.index_select(0, rows),
                key.index_select(0, rows),
                value.index_select(0, rows),
                views,
                decode_metadata,
                decoded,
                self.nova_codec,
                request_slots=metadata.nova_request_slots.index_select(0, requests),
            )
            output.index_copy_(0, rows, decoded)
        start = 0
        for request, end in enumerate(metadata.actual_seq_lengths_q):
            count, slot = end - start, slots[request]
            if count and 0 <= slot < self._limit and request not in decode_requests:
                old_length = metadata.seq_lens_list[request] - count
                k, v = key[start:end], value[start:end]
                result, _ = nova_prefill_attention(
                    query[start:end].reshape(count, self.num_kv_heads, -1, self.head_size),
                    self.nova_codec,
                    views,
                    metadata.block_tables[request],
                    self._window,
                    self.scale,
                    key=k,
                    value=v,
                    old_length=old_length,
                    request_slot=slot,
                    causal_mask=self._causal_mask,
                    query_tile=self._prefill_query_tile,
                    kv_tile=self._prefill_kv_tile,
                )
                output[start:end].copy_(result.reshape_as(output[start:end]).to(output.dtype))
                nova_exact_window_update(
                    self._window, k.contiguous(), v.contiguous(), request_slot=slot, old_length=old_length
                )
            start = end
        return output

    def _forward_general(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        if output is None:
            raise ValueError("NOVA requires an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("NOVA does not support output quantization")
        if getattr(layer, "head_size_v", self.head_size) != self.head_size:
            raise ValueError("NOVA-KV requires equal key and value head dimensions")
        if key is None or value is None:
            raise NotImplementedError("NOVA requires current-token K/V")
        codec = self._codec(layer, query.device)
        self._ensure_runtime(query.device, query.dtype)
        views = planar_cache_views(kv_cache)
        decoding = attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        count = len(attn_metadata.seq_lens_list) if decoding else attn_metadata.num_actual_tokens
        output.zero_()
        token_slots = attn_metadata.slot_mapping[:count]
        if decoding:
            token_slots = torch.where(attn_metadata.nova_request_slots[:count] >= 0, token_slots, -1)
        nova_quant_cache(
            key[:count], value[:count], codec, views, token_slots, chunk_size=self._options.codec_batch_size
        )
        if decoding:
            return self._decode(query, key, value, views, attn_metadata, output, codec)
        return self._prefill(query, key, value, views, attn_metadata, output)

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        if attn_metadata is not None:
            if (
                _ENCODE_GROUP
                and attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and len(attn_metadata.seq_lens_list) in (1, 4)
                and output is not None
                and output_scale is None
                and output_block_scale is None
                and key is not None
                and value is not None
                and getattr(layer, "head_size_v", self.head_size) == self.head_size
            ):
                codec = self._codec(layer, query.device)
                self._ensure_runtime(query.device, query.dtype)
                views = planar_cache_views(kv_cache)
                batch = len(attn_metadata.seq_lens_list)
                requests = attn_metadata.nova_request_slots[:batch].contiguous()
                if output.shape[0] != batch:
                    output.zero_()
                if _ENCODE_OVERLAP and self._window.ring >= 256:
                    stream = _encode_stream(query.device)
                    if batch == 4:
                        return self._decode(
                            query,
                            key,
                            value,
                            views,
                            attn_metadata,
                            output,
                            codec,
                            requests,
                            pending_encoder=stream,
                            encode_after_exact=True,
                        )
                    current = torch.npu.current_stream(query.device)
                    # The current token lies beyond even the final padded cold tile.
                    # Finish the writer before returning so allocator reuse is ordered.
                    stream.wait_stream(current)
                    with torch.npu.stream(stream):
                        encode_decode_cache(
                            key[:batch],
                            value[:batch],
                            codec,
                            views,
                            attn_metadata.slot_mapping[:batch],
                            requests,
                            self._limit,
                            _ENCODE_GROUP,
                        )
                    return self._decode(
                        query, key, value, views, attn_metadata, output, codec, requests, pending_encoder=stream
                    )
                encode_decode_cache(
                    key[:batch],
                    value[:batch],
                    codec,
                    views,
                    attn_metadata.slot_mapping[:batch],
                    requests,
                    self._limit,
                    _ENCODE_GROUP,
                )
                return self._decode(query, key, value, views, attn_metadata, output, codec, requests)
            return self._forward_general(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        if output is None or output_scale is not None or output_block_scale is not None:
            raise ValueError("NOVA-KV warmup requires an unquantized output buffer")
        codec = self._codec(layer, query.device)
        self._ensure_runtime(query.device, query.dtype)
        if key is not None and value is not None:
            codec.encode(key, value, chunk_size=self._options.codec_batch_size)
        return output.zero_()

    def _decode(
        self,
        query,
        key,
        value,
        views,
        metadata,
        output,
        codec,
        request_slots=None,
        pending_encoder=None,
        encode_after_exact=False,
    ):
        batch = len(metadata.seq_lens_list)
        if not 1 <= batch <= 4:
            raise ValueError("NOVA-KV requires 1..4 decode requests")
        max_tokens = self.vllm_config.model_config.max_model_len
        if max(metadata.seq_lens_list) > max_tokens:
            raise ValueError("decode sequence exceeds experimental capacity")
        if request_slots is None:
            request_slots = self._request_slots(metadata, batch)
        request_slots = request_slots.contiguous()
        lengths, starts, ends, positions = get_capture_resource(
            ("nova_batch_decode_bounds", batch, query.device),
            lambda: (
                torch.empty(batch, device=query.device, dtype=torch.int32),
                torch.empty(batch, device=query.device, dtype=torch.int32),
                torch.empty(batch, device=query.device, dtype=torch.int32),
                torch.empty(batch, device=query.device, dtype=torch.int64),
            ),
        )
        _batch_bounds[(batch,)](
            metadata.seq_lens_gpu,
            request_slots,
            lengths,
            starts,
            ends,
            positions,
            self._limit,
            self._window.sink,
            self._window.ring,
        )
        nova_exact_window_update(
            self._window,
            key[:batch].contiguous(),
            value[:batch].contiguous(),
            request_slots=request_slots,
            positions=positions,
        )
        grouped = query[:batch].reshape(batch, self.num_kv_heads, -1, self.head_size).float().contiguous()
        side = None
        # Exact attention shares the encoder stream; cold PV only reads older KV.
        if _EXACT_OVERLAP and batch in (1, 4) and _EXACT_TILE and output.is_contiguous():
            grouped = grouped.contiguous()
            side = _encode_stream(query.device)
            current = torch.npu.current_stream(query.device)
            parts = triton.cdiv(self._window.capacity, _EXACT_TILE)
            exact = get_capture_resource(
                ("nova_parallel_exact", batch, self.num_kv_heads, self.num_heads, query.device),
                lambda: (
                    torch.empty((*grouped.shape[:-1], parts, self.head_size), device=query.device, dtype=torch.float32),
                    torch.empty((*grouped.shape[:-1], parts), device=query.device, dtype=torch.float32),
                ),
            )
            side.wait_stream(current)
            with torch.npu.stream(side):
                # Grouped tiles help B4; B1 needs the larger per-group grid.
                exact_kernel = exact_partial_grouped if batch == 4 else exact_partial
                exact_kernel(self._window, grouped, request_slots, lengths, self.scale, _EXACT_TILE, exact)
        if encode_after_exact:
            # Launch the short exact reduction before encoding occupies the AIVs.
            # Current-token compressed KV is outside the padded cold read range.
            pending_encoder.wait_stream(torch.npu.current_stream(query.device))
            with torch.npu.stream(pending_encoder):
                encode_decode_cache(
                    key[:batch],
                    value[:batch],
                    codec,
                    views,
                    metadata.slot_mapping[:batch],
                    request_slots,
                    self._limit,
                    _ENCODE_GROUP,
                )
        cold, cold_lse = self._attend_cold(
            grouped, codec, views, metadata.block_tables[:batch], starts, ends, max_tokens
        )
        cold = torch.einsum("bhgd,hde->bhge", cold, codec.value_transform.transpose(-1, -2))
        if side is not None:
            current.wait_stream(side)
            merge_partial_into(cold, cold_lse, exact, output[:batch])
            return output
        if _EXACT_TILE and batch == 4:
            result, _, _ = exact_physical(
                self._window,
                grouped,
                request_slots,
                lengths,
                self.scale,
                cold,
                cold_lse,
                _EXACT_TILE,
            )
        else:
            result, _, _ = self._window.attend_and_merge(
                grouped,
                request_slots,
                lengths,
                self.scale,
                cold,
                cold_lse,
            )
        output[:batch].copy_(result.reshape_as(output[:batch]).to(output.dtype))
        if pending_encoder is not None:
            torch.npu.current_stream(query.device).wait_stream(pending_encoder)
        return output

    def _attend_cold(self, grouped, codec, views, table, starts, ends, max_tokens):
        from vllm_ascend.ops.tilelang.nova_kv_decode import NovaDecodeKernel

        batch = grouped.shape[0]
        transformed, mean = project_query(grouped, codec, fuse_outputs=True)
        work = get_capture_resource(
            (
                "nova_decode",
                batch,
                max_tokens,
                self.num_kv_heads,
                self.num_heads,
                views[0].shape,
                views[0].stride(),
                views[2].stride(),
                table.shape[1],
                grouped.device,
            ),
            lambda: NovaDecodeKernel(
                max_tokens,
                self.num_kv_heads,
                self.num_heads // self.num_kv_heads,
                views,
                table,
                grouped.device,
                batch=batch,
                split_tokens=_LONG_CONTEXT_SPLIT_TOKENS
                if batch == 4 and max_tokens >= _LONG_CONTEXT_MIN_TOKENS
                else 1024,
                scale=self.scale,
            ),
        )
        return work(transformed, mean, views, codec._decode_books, table, starts, ends)
