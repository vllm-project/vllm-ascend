# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-length FA3 attention for A2/A3 Qwen targets and DFlash2 drafts."""

from dataclasses import dataclass, field

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, notify_kv_cache_written
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window


def _get_window_size(sliding_window: int | None, causal: bool) -> tuple[int, int]:
    if sliding_window is None:
        return (-1, -1)
    return (sliding_window - 1, 0 if causal else sliding_window - 1)


class AscendFlashAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    forward_includes_kv_cache_update = True
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # The 910 FA3 ABI supports native widths through 256, including 192.
        return 1 <= head_size <= 256

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=""):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [128]

    @staticmethod
    def get_impl_cls():
        return AscendFlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        return AscendFlashAttentionMetadataBuilder


@dataclass
class AscendFA3QueryGroup:
    ranges: list[tuple[int, int]]
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)


@dataclass
class AscendFlashAttentionMetadata(AttentionMetadata):
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    causal: bool
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)
    full_attention_window: int | None = None
    query_offsets_cpu: tuple[int, ...] = ()
    eager_query_groups: list[AscendFA3QueryGroup] = field(default_factory=list)


class AscendFlashAttentionMetadataBuilder(AttentionMetadataBuilder[AscendFlashAttentionMetadata]):
    reorder_batch_threshold = None

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.use_graph_buffers = (
            not vllm_config.speculative_config.enforce_eager
            and vllm_config.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        self.scheduler_specs = set()
        for name in layer_names:
            impl = vllm_config.compilation_config.static_forward_context[name].impl
            self.scheduler_specs.add(
                (
                    impl.num_heads,
                    impl.num_kv_heads,
                    impl.head_size,
                    kv_cache_spec.dtype,
                    impl.scale,
                    impl.logits_soft_cap,
                    impl.sliding_window,
                )
            )
        self.scheduler_buffers: dict[tuple, dict[tuple, torch.Tensor]] = {}

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def _prepare_block_tables(self, block_tables, active_reqs):
        return block_tables[:active_reqs], None

    def _get_max_query_len(self, common, num_input_tokens):
        return common.max_query_len

    def build(self, common_prefix_len, common_attn_metadata: AscendCommonAttentionMetadata, fast_build=False):
        # The optional extension is loaded only by workers selecting this backend.
        from flash_attn_npu_3 import get_scheduler_metadata

        common = common_attn_metadata
        offsets_cpu = common.query_start_loc_cpu[: common.num_reqs + 1]
        # FULL draft metadata reports padded tokens while CPU offsets end at the
        # real query extent. Exclude repeated padding offsets from FA3 tiling.
        active_reqs = max(1, int(torch.searchsorted(offsets_cpu, offsets_cpu[-1])))
        query_start_loc = common.query_start_loc[: active_reqs + 1]
        seq_lens = common.seq_lens[:active_reqs]
        block_tables, full_attention_window = self._prepare_block_tables(common.block_table_tensor, active_reqs)
        num_input_tokens = common.num_input_tokens or common.num_actual_tokens
        max_query_len = self._get_max_query_len(common, num_input_tokens)
        source_key = (query_start_loc.data_ptr(), seq_lens.data_ptr(), block_tables.data_ptr())
        # DFlash rounds capture sizes to whole queries; target capture sizes do
        # not identify its graph buckets. Retain storage for every actual shape.
        graph_key = (source_key, num_input_tokens, max_query_len, common.causal)
        scheduler_metadata = self.scheduler_buffers.setdefault(graph_key, {}) if self.use_graph_buffers else {}
        for spec in self.scheduler_specs:
            num_heads, num_kv_heads, head_size, dtype, scale, softcap, sliding_window = spec
            # AICPU tiling stays outside the graph: its auxiliary stream events
            # are not capture-safe on every CANN version. This tiling call uses
            # device lengths and is shared by layers with identical parameters.
            tiling = get_scheduler_metadata(
                batch_size=active_reqs,
                max_seqlen_q=max_query_len,
                max_seqlen_k=block_tables.shape[1] * self.block_size,
                num_heads_q=num_heads,
                num_heads_kv=num_kv_heads,
                headdim=head_size,
                cache_seqlens=seq_lens,
                qkv_dtype=dtype,
                cu_seqlens_q=query_start_loc,
                page_size=self.block_size,
                causal=common.causal,
                window_size=_get_window_size(
                    sliding_window if sliding_window is not None else full_attention_window, common.causal
                ),
                has_softcap=softcap != 0.0,
                num_splits=0,
            )
            if spec in scheduler_metadata:
                buffer = scheduler_metadata[spec]
                # The 910 ABI uses a fixed byte layout per mask/window mode.
                # Reallocating here would leave a graph reading stale metadata.
                if buffer.shape != tiling.shape or buffer.dtype != tiling.dtype:
                    raise RuntimeError("FA3 scheduler metadata layout changed for a captured draft shape")
                buffer.copy_(tiling)
            else:
                scheduler_metadata[spec] = tiling
        return AscendFlashAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=block_tables,
            slot_mapping=common.slot_mapping[: common.num_actual_tokens],
            max_query_len=max_query_len,
            causal=common.causal,
            scheduler_metadata=scheduler_metadata,
            full_attention_window=full_attention_window,
            query_offsets_cpu=tuple(offsets_cpu[: active_reqs + 1].tolist())
            if full_attention_window is not None
            else (),
        )


class AscendQwenFlashAttentionMetadataBuilder(AscendFlashAttentionMetadataBuilder):
    """Keep full attention on the device-length path of the 910 FA3 ABI."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        mode = vllm_config.compilation_config.cudagraph_mode
        self.use_graph_buffers = mode.decode_mode() == CUDAGraphMode.FULL
        self.capture_query_len = (
            1 + vllm_config.speculative_config.num_speculative_tokens
            if mode == CUDAGraphMode.FULL_DECODE_ONLY
            else None
        )
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.block_table_buffers: dict[tuple, torch.Tensor] = {}

    def _prepare_block_tables(self, block_tables, active_reqs):
        # A window covering the original page-table capacity includes every
        # valid KV token, so it is exactly full attention. Add an unused column
        # to keep FA3 0.5.0 from normalizing that window back to global attention:
        # the global TND branch synchronously reads lengths to select FlashDecode.
        # Exact seq_lens are unchanged; the extra column is never accessed.
        key = (block_tables.data_ptr(), block_tables.shape[1])
        if key not in self.block_table_buffers:
            self.block_table_buffers[key] = block_tables.new_zeros((self.max_num_reqs, block_tables.shape[1] + 1))
        padded = self.block_table_buffers[key][:active_reqs]
        padded[:, :-1].copy_(block_tables[:active_reqs])
        return padded, block_tables.shape[1] * self.block_size

    def _get_max_query_len(self, common, num_input_tokens):
        # A captured decode bucket can be replayed with fewer valid query
        # tokens after rejection. Retain its scalar Q bound and tiling buffer.
        if self.use_graph_buffers:
            if self.capture_query_len is None:
                return num_input_tokens
            if common.max_query_len <= self.capture_query_len:
                return self.capture_query_len
        return common.max_query_len


class AscendQwenFlashAttentionBackend(AscendFlashAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return AscendQwenFlashAttentionMetadataBuilder


class AscendFlashAttentionImpl(AttentionImpl[AscendFlashAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ):
        if attn_type != AttentionType.DECODER or alibi_slopes is not None or sinks is not None:
            raise ValueError("DFlash2 FA3 requires decoder attention without ALiBi or sinks")
        logger.info_once("Using device-length Ascend FA3 attention.", scope="process")
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.logits_soft_cap = 0.0 if logits_soft_cap is None else logits_soft_cap

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # The builder has already refreshed the captured device tiling storage.
        pass

    def do_kv_cache_update(self, layer: AttentionLayer, key, value, kv_cache, slot_mapping) -> None:
        if self.kv_sharing_target_layer_name is not None:
            return
        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_mapping=slot_mapping,
        )
        notify_kv_cache_written(layer.layer_name)

    def _forward_grouped_queries(self, query, kv_cache, metadata, scheduler_key, output):
        from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata

        # The 910 TND eager wrapper reads cu_seqlens_q[-1].item() even when
        # device tiling is supplied. CPU query offsets are already exact, so
        # group equal-width requests into BSND calls without that validation
        # read. FULL graphs keep TND: capture skips the check and device tiling
        # handles changing query widths/active batches on every replay.
        if metadata.query_offsets_cpu[-1] > query.shape[0]:
            raise ValueError("Qwen FA3 query offsets exceed the packed query tensor")
        if not metadata.eager_query_groups:
            requests_by_width: dict[int, list[int]] = {}
            offsets = metadata.query_offsets_cpu
            for req, (start, end) in enumerate(zip(offsets, offsets[1:])):
                if end > start:
                    requests_by_width.setdefault(end - start, []).append(req)
            for requests in requests_by_width.values():
                start, end = requests[0], requests[-1] + 1
                if end - start == len(requests):
                    lengths = metadata.seq_lens[start:end]
                    tables = metadata.block_tables[start:end]
                else:
                    lengths = torch.cat([metadata.seq_lens[req : req + 1] for req in requests])
                    tables = torch.cat([metadata.block_tables[req : req + 1] for req in requests])
                metadata.eager_query_groups.append(
                    AscendFA3QueryGroup([(offsets[req], offsets[req + 1]) for req in requests], lengths, tables)
                )
        window = _get_window_size(
            self.sliding_window if self.sliding_window is not None else metadata.full_attention_window,
            metadata.causal,
        )
        for group in metadata.eager_query_groups:
            start, end = group.ranges[0][0], group.ranges[-1][1]
            query_len = group.ranges[0][1] - start
            contiguous = end - start == len(group.ranges) * query_len
            packed = query[start:end] if contiguous else torch.cat([query[a:b] for a, b in group.ranges])
            grouped = packed.view(len(group.ranges), query_len, self.num_heads, self.head_size)
            if scheduler_key not in group.scheduler_metadata:
                group.scheduler_metadata[scheduler_key] = get_scheduler_metadata(
                    batch_size=len(group.ranges),
                    max_seqlen_q=query_len,
                    max_seqlen_k=group.block_tables.shape[1] * kv_cache[0].shape[1],
                    num_heads_q=self.num_heads,
                    num_heads_kv=self.num_kv_heads,
                    headdim=self.head_size,
                    cache_seqlens=group.seq_lens,
                    qkv_dtype=query.dtype,
                    page_size=kv_cache[0].shape[1],
                    causal=metadata.causal,
                    window_size=window,
                    has_softcap=self.logits_soft_cap != 0.0,
                    num_splits=0,
                )
            result = flash_attn_with_kvcache(
                grouped,
                kv_cache[0],
                kv_cache[1],
                cache_seqlens=group.seq_lens,
                page_table=group.block_tables,
                softmax_scale=self.scale,
                causal=metadata.causal,
                window_size=window,
                softcap=self.logits_soft_cap,
                scheduler_metadata=group.scheduler_metadata[scheduler_key],
                num_splits=0,
            )
            if contiguous:
                output[start:end].copy_(result.flatten(0, 1))
            else:
                for row, (a, b) in enumerate(group.ranges):
                    output[a:b].copy_(result[row])
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata: AscendFlashAttentionMetadata | None,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        assert output is not None
        assert output_scale is None and output_block_scale is None
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        if attn_metadata is None:
            return output.fill_(0)

        from flash_attn_npu_3 import flash_attn_with_kvcache

        if key is not None and value is not None:
            num_tokens = attn_metadata.num_actual_tokens
            self.do_kv_cache_update(
                layer,
                key[:num_tokens],
                value[:num_tokens],
                kv_cache,
                attn_metadata.slot_mapping[:num_tokens],
            )
        scheduler_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            query.dtype,
            self.scale,
            self.logits_soft_cap,
            self.sliding_window,
        )
        with attention_transfer_window():
            if attn_metadata.full_attention_window is not None and not torch.npu.is_current_stream_capturing():
                return self._forward_grouped_queries(query, kv_cache, attn_metadata, scheduler_key, output)
            # Keep packed TND queries and the runner's graph-stable offsets.
            # Refreshed device tiling limits replay to the active requests.
            result = flash_attn_with_kvcache(
                query,
                kv_cache[0],
                kv_cache[1],
                cache_seqlens=attn_metadata.seq_lens,
                page_table=attn_metadata.block_tables,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=_get_window_size(
                    self.sliding_window if self.sliding_window is not None else attn_metadata.full_attention_window,
                    attn_metadata.causal,
                ),
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata[scheduler_key],
                num_splits=0,
            )
            output.copy_(result)
        return output
