# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)
from vllm_ascend.ops.triton.mamba.gdn_prepare import prepare_gdn

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18
# Accepted-token counts are one-based. The zero-length query segment makes
# graph-padding rows no-ops, while 1 keeps their count valid for GDN kernels.
_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        # Boolean mask values are exactly representable in float32. Integer
        # argsort falls back to AiCPU on Ascend; keep this stable partition on
        # AiCore without changing the ordering of tokens within either group.
        tensor = tensor.to(torch.float32)
    return torch.argsort(tensor, stable=True)


def _remove_spec_graph_padding_queries(
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
    *,
    num_actual_reqs: int | None = None,
) -> CommonAttentionMetadata:
    """Remove inactive FIA graph rows from the GDN-local query view.

    FIA represents every padded graph request with a full K+1 query span.
    Those rows must be zero-length for the target's recurrent GDN state update.
    The shared MLA metadata remains unchanged.
    """
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    if seq_lens is None or num_decode_draft_tokens_cpu is None:
        return common_attn_metadata
    num_reqs = common_attn_metadata.num_reqs
    draft_tokens = num_decode_draft_tokens_cpu[:num_reqs]
    if draft_tokens.numel() != num_reqs or not torch.any(draft_tokens >= 0).item():
        return common_attn_metadata
    inactive = (seq_lens[:num_reqs] == 0) & (draft_tokens < 0)
    if not torch.any(inactive).item():
        return common_attn_metadata
    first_inactive = int(torch.nonzero(inactive, as_tuple=True)[0][0])
    if not torch.all(inactive[first_inactive:]).item():
        raise ValueError("GDN speculative graph padding must be a request suffix")
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.clone()
    query_start_loc_cpu[first_inactive + 1 :] = query_start_loc_cpu[first_inactive]
    # Adaptive verification can redistribute the CPU's even split on device.
    # Remove only padding; copying CPU boundaries would undo that allocation.
    query_start_loc = common_attn_metadata.query_start_loc
    if num_actual_reqs != first_inactive:
        query_start_loc = query_start_loc.clone()
        query_start_loc[first_inactive + 1 :] = query_start_loc[first_inactive]
    return common_attn_metadata.replace(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc,
        num_actual_tokens=int(query_start_loc_cpu[-1]),
    )


def _treat_single_token_prefills_with_state_as_decodes(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Use decode metadata for one-token stateful prompt chunks.

    A final one-token prompt chunk can replay the same fixed graph as an
    ordinary decode. Once recurrent state exists, both paths must construct
    identical GDN metadata so the graph consumes the current state indices.
    First-token prefills remain on the prefill path because they have no state
    to update.
    """
    is_prefilling = common_attn_metadata.is_prefilling
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    if is_prefilling is None or seq_lens_cpu is None:
        return common_attn_metadata

    query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
    prefill_to_decode = is_prefilling & (query_lens_cpu == 1) & (seq_lens_cpu > 1)
    if not torch.any(prefill_to_decode).item():
        return common_attn_metadata

    is_prefilling = is_prefilling.clone()
    is_prefilling[prefill_to_decode] = False
    return common_attn_metadata.replace(is_prefilling=is_prefilling)


@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    chunk_indices_chunk64: torch.Tensor | None
    chunk_offsets_chunk64: torch.Tensor | None
    update_chunk_offsets_chunk64: torch.Tensor | None
    final_chunk_indices_chunk64: torch.Tensor | None
    chunk_indices_large_block: torch.Tensor | None
    block_indices_cumsum: torch.Tensor | None
    num_decodes: int = 0
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None


@dataclass
class GDNCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    initial_state_mode: torch.Tensor | None
    output_buffer: torch.Tensor | None = None


@dataclass
class GDNSpecCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    output_buffer: torch.Tensor | None = None


@dataclass
class GDNPrefillMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class AscendGDNAttentionMetadata(GDNAttentionMetadata):
    producer_chunk_metadata: GDNChunkedPrefillMetadata | None = None
    consumer_chunk_metadata: GDNChunkedPrefillMetadata | None = None


@dataclass
class GDNDecodeMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor


@dataclass
class GDNSpecDecodeMetadata:
    spec_causal_conv1d: GDNSpecCausalConv1dMetadata
    # Native common GDN consumes these lengths. Adaptive GDN uses query_start_loc
    # directly; 310P derives lengths in its recurrent wrapper.
    actual_seq_lengths: torch.Tensor | None
    # Adaptive graphs must retain the state-table row stride independently
    # of the current query length, including accepted_prev > query_len.
    use_upstream_recurrent: bool = False
    # Adaptive FULL graphs retain padded token tensors as query lengths vary.
    use_upstream_conv: bool = False


def _build_actual_seq_lengths(
    query_start_loc: torch.Tensor,
    num_sequences: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    actual_seq_lengths = (
        torch.empty_like(query_start_loc[: num_sequences + 1]) if out is None else out[: num_sequences + 1]
    )
    actual_seq_lengths[:1].copy_(query_start_loc[:1])
    torch.sub(
        query_start_loc[1 : num_sequences + 1],
        query_start_loc[:num_sequences],
        out=actual_seq_lengths[1:],
    )
    return actual_seq_lengths


def _materialize_graph_request_tensor(
    buffer: torch.Tensor,
    values: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
    padding_value: int | bool,
) -> torch.Tensor:
    """Write request-level values into a stable full-graph buffer."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count > buffer.size(0):
        raise ValueError(f"graph_request_count ({graph_request_count}) exceeds buffer capacity ({buffer.size(0)})")
    if actual_request_count > values.size(0):
        raise ValueError(f"actual_request_count ({actual_request_count}) exceeds source rows ({values.size(0)})")

    materialized = buffer[:graph_request_count]
    materialized[:actual_request_count].copy_(
        values[:actual_request_count],
        non_blocking=True,
    )
    materialized[actual_request_count:].fill_(padding_value)
    return materialized


def _materialize_graph_query_start_loc(
    buffer: torch.Tensor,
    query_start_loc: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
) -> torch.Tensor:
    """Pad request boundaries with zero-length dummy request segments."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count + 1 > buffer.size(0):
        raise ValueError(
            f"graph request boundaries ({graph_request_count + 1}) exceed buffer capacity ({buffer.size(0)})"
        )
    if actual_request_count + 1 > query_start_loc.size(0):
        raise ValueError(
            f"actual request boundaries ({actual_request_count + 1}) exceed source rows ({query_start_loc.size(0)})"
        )
    materialized = buffer[: graph_request_count + 1]
    materialized[: actual_request_count + 1].copy_(
        query_start_loc[: actual_request_count + 1],
        non_blocking=True,
    )
    padding = materialized[actual_request_count + 1 :]
    if padding.numel() > 0:
        padding.copy_(
            materialized[actual_request_count : actual_request_count + 1].expand_as(padding),
            non_blocking=True,
        )
    return materialized


def _compact_empty_segments(cu_seqlens_host, initial_state, device=None):
    """Drop zero-length segments so AscendC fwd_h/fwd_o indexing lines up.

    Returns ``(cu_seqlens_kern, initial_state_kern, keep_meta)``:
    cu_seqlens / initial_state with empty segments removed, plus a bool
    mask (None when nothing was removed).  The compacted ``final_state``
    must be scattered back via ``keep_meta`` (empty segments keep their
    initial state).

    When *device* is given, ``keep_meta`` is moved to that device so that
    callers can index NPU tensors without an extra host→device sync.
    """
    if cu_seqlens_host is None:
        return None, initial_state, None
    cu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    keep = (cu[1:] - cu[:-1]) > 0
    if bool(keep.all()):
        return cu_seqlens_host, initial_state, None
    # Compute compact cu_seqlens while keep is still on CPU (cu is CPU-only).
    cu_kern = torch.cat([cu[:1], cu[1:][keep]]).tolist()
    # Move keep to device only for indexing device-side tensors.
    if device is not None:
        keep = keep.to(device)
    st_kern = initial_state[keep] if initial_state is not None else None
    return cu_kern, st_kern, keep


def _build_non_spec_chunked_prefill_metadata(
    builder,
    cu_seqlens_cpu: torch.Tensor,
    device: torch.device,
    build_device_metadata: bool = True,
) -> GDNChunkedPrefillMetadata:
    chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    if build_device_metadata:
        hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
        linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
        if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
            gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
        elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
            gdn_num_heads = (
                hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
            )
        else:
            gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(
                builder.vllm_config.parallel_config
            )
        cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
        cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()
        chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        chunk_indices_large_block = prepare_chunk_indices(
            cu_seqlens_cpu,
            _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        )
        block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)

    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
    # Pre-compute compact cu_seqlens for AscendC kernels so each layer
    # can reuse them instead of calling _compact_empty_segments again.
    cu_seqlens_kern, _, keep_meta = _compact_empty_segments(cu_seqlens_host, None, device=device)
    if keep_meta is None:
        cu_seqlens_kern = None
    else:
        cu_seqlens_kern = tuple(cu_seqlens_kern)

    if build_device_metadata:
        device_chunk_metadata = {
            "chunk_indices_chunk64": chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_offsets_chunk64": chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "update_chunk_offsets_chunk64": update_chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "final_chunk_indices_chunk64": final_chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_indices_large_block": chunk_indices_large_block.to(device=device, non_blocking=True),
            "block_indices_cumsum": block_indices_cumsum.to(device=device, non_blocking=True),
        }
    else:
        device_chunk_metadata = {
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": None,
            "update_chunk_offsets_chunk64": None,
            "final_chunk_indices_chunk64": None,
            "chunk_indices_large_block": None,
            "block_indices_cumsum": None,
        }

    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),
        **device_chunk_metadata,
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=keep_meta,
    )


def _split_gdn_batch(
    m: CommonAttentionMetadata,
    spec_mask: torch.Tensor | None,
    num_spec: int,
) -> tuple[int, int, int, int, int, torch.Tensor | None]:
    if m.num_reqs == 0:
        return 0, 0, 0, 0, 0, m.query_start_loc_cpu
    if spec_mask is None:
        return (
            *split_decodes_and_prefills(m, decode_threshold=1, treat_short_extends_as_decodes=False),
            0,
            m.query_start_loc_cpu,
        )
    if num_spec == m.num_reqs:
        return 0, 0, 0, 0, m.num_actual_tokens, None
    lengths = torch.diff(m.query_start_loc_cpu)[~spec_mask]
    # Ordinary decode rows share the non-spec prefill branch when drafts exist.
    num_prefills, num_prefill_tokens = int((lengths > 0).sum()), int(lengths.sum())
    query = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
    return 0, num_prefills, 0, num_prefill_tokens, m.num_actual_tokens - num_prefill_tokens, query


class AscendGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    build_device_chunk_metadata = True

    @classmethod
    def get_varlen_cudagraph_max_query_len(cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec) -> int | None:
        # Only adaptive verification advertises variable-length graph replay.
        # Long fixed queries can share the strided recurrent kernel while
        # retaining uniform graph dispatch.
        spec_config = vllm_config.speculative_config
        if spec_config is None or not spec_config.enable_adaptive_verification:
            return None
        return 1 + spec_config.num_speculative_tokens

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sequence_index_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
        )

        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (sequence_index_capacity,), dtype=torch.bool, device=device
        )

        self.spec_sequence_masks_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        )

        self.spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.non_spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.non_spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.spec_actual_seq_lengths: torch.Tensor | None = None
        self.non_spec_actual_seq_lengths: torch.Tensor | None = None
        if self._USE_COMMON_KERNEL_METADATA:
            self.spec_actual_seq_lengths = torch.empty((sequence_index_capacity + 1,), dtype=torch.int32, device=device)
            self.non_spec_actual_seq_lengths = torch.empty(
                (sequence_index_capacity + 1,), dtype=torch.int32, device=device
            )

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        super()._init_reorder_batch_threshold(
            reorder_batch_threshold,
            supports_spec_as_decode,
            True,
        )
        if self.reorder_batch_threshold != 1:  # type: ignore
            speculative_config = self.vllm_config.speculative_config
            method = getattr(speculative_config, "method", None)
            num_spec = getattr(speculative_config, "num_speculative_tokens", None)
            if num_spec is not None and method in ("dflash", "dspark"):
                # The target-model verification query always contains the
                # base token plus N speculative tokens. DSpark's
                # sample_from_anchor only makes the draft-model forward use N
                # queries; it must not change target batch reordering.
                self.reorder_batch_threshold = 1 + num_spec

    def _copy_sequence_indices_to_device(
        self,
        spec_sequence_masks_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        spec_indices_cpu.copy_(
            torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        non_spec_indices_cpu.copy_(
            torch.nonzero(~spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        non_spec_indices = self.non_spec_sequence_indices[:num_non_spec_decodes]
        non_spec_indices.copy_(non_spec_indices_cpu, non_blocking=True)

        return spec_indices, non_spec_indices

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        """Make a captured speculative branch a no-op for this replay."""
        self.spec_state_indices_tensor[:graph_batch_size].fill_(PAD_SLOT_ID)
        self.spec_query_start_loc[: graph_batch_size + 1].zero_()
        self.num_accepted_tokens[:graph_batch_size].zero_()
        if self.spec_actual_seq_lengths is not None:
            self.spec_actual_seq_lengths[: graph_batch_size + 1].zero_()

    def _attach_non_spec_prefill_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        chunk_metadata: GDNChunkedPrefillMetadata | None,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_prefill_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA or attn_metadata.num_prefills <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for Ascend GDN non-spec prefill path.")
        if attn_metadata.prefill_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.prefill_query_start_loc for Ascend GDN non-spec prefill path.")
        if chunk_metadata is None:
            raise RuntimeError("Expected chunk metadata for Ascend GDN non-spec prefill path.")

        initial_state_mode = attn_metadata.has_initial_state
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN prefill conv1d path.")
        prefill_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        # MRV2 builds complete request metadata for recurrent attention.
        # Fresh requests must ignore reused convolution state on every rank.
        attn_metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:prefill_num_rows],
                initial_state_mode=initial_state_mode,
            ),
            chunk=chunk_metadata,
        )
        return attn_metadata

    def _attach_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        actual_seq_lengths: torch.Tensor | None = None,
        cache_indices: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        attn_metadata.spec_decode_metadata = None
        if attn_metadata.spec_sequence_masks is None:
            return attn_metadata

        if attn_metadata.spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.spec_query_start_loc for Ascend GDN speculative path.")
        if attn_metadata.spec_state_indices_tensor is None:
            raise RuntimeError("Expected spec_state_indices_tensor for Ascend GDN speculative conv1d path.")
        if attn_metadata.num_accepted_tokens is None:
            raise RuntimeError("Expected num_accepted_tokens for Ascend GDN speculative conv1d path.")

        num_sequences = attn_metadata.num_spec_decodes
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            num_sequences = attn_metadata.spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.spec_actual_seq_lengths
        spec_num_rows = attn_metadata.spec_query_start_loc.size(0) - 1

        use_upstream_recurrent = False
        if self._USE_COMMON_KERNEL_METADATA and self.speculative_config:
            if getattr(self.speculative_config, "enable_adaptive_verification", False):
                # Variable query lengths must retain the previous accepted
                # state offset on every supported adaptive backend.
                use_upstream_recurrent = True
            else:
                # Fixed-length verification only opts into the workaround
                # on profiles with an observed native numerical failure.
                fallback_min_query_len = get_current_hardware_profile().gdn_fixed_spec_recurrent_fallback_min_query_len
                use_upstream_recurrent = (
                    fallback_min_query_len is not None and self.num_spec + 1 >= fallback_min_query_len
                )
        attn_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            use_upstream_recurrent=use_upstream_recurrent,
            use_upstream_conv=bool(
                self._USE_COMMON_KERNEL_METADATA
                and self.speculative_config
                and getattr(self.speculative_config, "enable_adaptive_verification", False)
            ),
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                query_start_loc=attn_metadata.spec_query_start_loc,
                cache_indices=(attn_metadata.spec_state_indices_tensor if cache_indices is None else cache_indices)[
                    :spec_num_rows
                ],
                num_accepted_tokens=attn_metadata.num_accepted_tokens[:spec_num_rows],
            ),
            actual_seq_lengths=(
                (
                    actual_seq_lengths[: num_sequences + 1]
                    if actual_seq_lengths is not None
                    else _build_actual_seq_lengths(
                        attn_metadata.spec_query_start_loc,
                        num_sequences,
                        actual_seq_lengths_buffer,
                    )
                )
                if self._USE_COMMON_KERNEL_METADATA and not use_upstream_recurrent
                else None
            ),
        )
        return attn_metadata

    def _attach_non_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        non_spec_cache_indices: torch.Tensor | None,
        actual_seq_lengths: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_decode_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA:
            return attn_metadata
        if attn_metadata.num_decodes <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected non-spec query_start_loc for Ascend GDN non-spec decode path.")
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN decode conv1d path.")

        num_sequences = attn_metadata.num_decodes
        non_spec_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_spec_decodes == 0:
            num_sequences = attn_metadata.non_spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.non_spec_actual_seq_lengths

        attn_metadata.non_spec_decode_metadata = GDNDecodeMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:non_spec_num_rows],
                initial_state_mode=None,
            ),
            actual_seq_lengths=(
                actual_seq_lengths[: num_sequences + 1]
                if actual_seq_lengths is not None
                else _build_actual_seq_lengths(
                    attn_metadata.non_spec_query_start_loc, num_sequences, actual_seq_lengths_buffer
                )
            ),
        )
        return attn_metadata

    _SPEC_GRAPH_PAD_SLOT_ID = NULL_BLOCK_ID
    # Common GDN/KDA kernels consume prebuilt chunks and recurrent lengths.
    # 310P consumes top-level fields and only shares the spec conv wrapper.
    _USE_COMMON_KERNEL_METADATA = True

    def _can_pad_spec_decode(self, graph_request_count: int, num_spec_decode_tokens: int) -> bool:
        return (
            graph_request_count <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _pad_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize speculative decode metadata at graph request granularity."""
        actual_request_count = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_state_indices is not None
        assert spec_sequence_masks is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None

        # Request and token buffers have different capacities under MTP.
        # Check token indices separately before modifying any captured inputs.
        assert attn_metadata.non_spec_token_indx is not None
        assert attn_metadata.spec_token_indx is not None
        if attn_metadata.spec_token_indx.numel() > self.spec_token_indx.numel():
            raise ValueError("Spec token indices exceed the graph token buffer capacity")
        if attn_metadata.non_spec_token_indx.numel() > self.non_spec_token_indx.numel():
            raise ValueError("Non-spec token indices exceed the graph token buffer capacity")

        attn_metadata.spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.spec_state_indices_tensor,
            spec_state_indices,
            actual_request_count,
            graph_request_count,
            self._SPEC_GRAPH_PAD_SLOT_ID,
        )
        attn_metadata.spec_sequence_masks = _materialize_graph_request_tensor(
            self.spec_sequence_masks,
            spec_sequence_masks,
            actual_request_count,
            graph_request_count,
            False,
        )

        non_spec_token_count = attn_metadata.non_spec_token_indx.size(0)
        spec_token_count = attn_metadata.spec_token_indx.size(0)
        self.non_spec_token_indx[:non_spec_token_count].copy_(
            attn_metadata.non_spec_token_indx,
            non_blocking=True,
        )
        self.spec_token_indx[:spec_token_count].copy_(
            attn_metadata.spec_token_indx,
            non_blocking=True,
        )
        attn_metadata.non_spec_token_indx = self.non_spec_token_indx[:non_spec_token_count]
        attn_metadata.spec_token_indx = self.spec_token_indx[:spec_token_count]

        attn_metadata.spec_query_start_loc = _materialize_graph_query_start_loc(
            self.spec_query_start_loc,
            spec_query_start_loc,
            actual_request_count,
            graph_request_count,
        )
        attn_metadata.num_accepted_tokens = _materialize_graph_request_tensor(
            self.num_accepted_tokens,
            num_accepted_tokens,
            actual_request_count,
            graph_request_count,
            _GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT,
        )

    def _pad_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize non-speculative decode metadata at request granularity."""
        # Ordinary decode has one token per live request. Zero-length graph
        # rows (including idle DP batches) must not copy stale state indices.
        actual_request_count = attn_metadata.num_decode_tokens
        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None

        attn_metadata.non_spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.non_spec_state_indices_tensor,
            state_indices,
            actual_request_count,
            graph_request_count,
            NULL_BLOCK_ID,
        )
        attn_metadata.non_spec_query_start_loc = _materialize_graph_query_start_loc(
            self.non_spec_query_start_loc,
            query_start_loc,
            actual_request_count,
            graph_request_count,
        )

    def _build_fused_metadata(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        spec_sequence_masks_cpu: torch.Tensor | None,
        num_spec_decodes: int,
        graph_request_count: int,
        batch: tuple[int, int, int, int, int, torch.Tensor | None],
    ) -> GDNAttentionMetadata | None:
        return None

    def _build_grouped_prefill_metadata(
        self,
        m: CommonAttentionMetadata,
        spec_sequence_masks: torch.Tensor | None,
        num_decodes: int,
        num_prefills: int,
        prefill_state_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
    ) -> dict[str, object]:
        grouped: dict[str, object] = {
            "prefix_producer_ranges": None,
            "producer_token_indices": None,
            "producer_query_start_loc": None,
            "consumer_ranges": None,
            "consumer_token_indices": None,
            "consumer_query_start_loc": None,
            "shared_initial_state_source": None,
            "shared_state_destinations": None,
            "consumer_shared_state_sources": None,
            "private_final_state_destination": None,
            "checkpoint_source_block_ids": getattr(m, "mamba_checkpoint_source_block_ids", None),
            "producer_slice": None,
            "consumer_slice": None,
            "producer_chunk_metadata": None,
            "consumer_chunk_metadata": None,
        }
        producer_indices = getattr(m, "mamba_prefix_producer_indices", None)
        source_block_ids = getattr(m, "mamba_checkpoint_source_block_ids", None)
        has_producers = producer_indices is not None and (
            getattr(producer_indices, "_has_valid", None)
            if hasattr(producer_indices, "_has_valid")
            else bool(torch.any(producer_indices >= 0).item())
        )
        has_consumers = source_block_ids is not None and (
            getattr(source_block_ids, "_has_valid", None)
            if hasattr(source_block_ids, "_has_valid")
            else bool(torch.any(source_block_ids >= 0).item())
        )
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        enable_grouped = (
            pcp_size == 1
            and spec_sequence_masks is None
            and ((has_producers and num_decodes == 0) or (not has_producers and has_consumers and num_prefills > 0))
        )
        if not enable_grouped:
            return grouped

        device = m.query_start_loc.device
        if not has_producers:
            assert prefill_state_indices is not None
            assert source_block_ids is not None
            prefill_count = prefill_state_indices.shape[0]
            src_ids = source_block_ids[num_decodes : num_decodes + prefill_count].to(
                device=device, dtype=prefill_state_indices.dtype
            )
            grouped["consumer_shared_state_sources"] = torch.where(src_ids >= 0, src_ids, prefill_state_indices)
            grouped["private_final_state_destination"] = prefill_state_indices
            return grouped

        def cpu_list(tensor: torch.Tensor | None, default: list[int]) -> list[int]:
            if tensor is None:
                return default
            values = getattr(tensor, "_cpu_list", None)
            return values if values is not None else tensor.cpu().tolist()

        producer_indices_cpu = cpu_list(producer_indices, [-1] * m.num_reqs)
        source_block_ids_cpu = cpu_list(source_block_ids, [-1] * m.num_reqs)
        checkpoint_positions = m.mamba_checkpoint_positions
        assert checkpoint_positions is not None
        checkpoint_positions_cpu = cpu_list(checkpoint_positions, [-1] * m.num_reqs)
        query_starts = m.query_start_loc_cpu.tolist()
        producer_ranges_list: list[list[int]] = []
        consumer_ranges_list: list[list[int]] = []
        initial_sources_list: list[int] = []
        producer_row_indices: list[int] = []
        producer_block_indices: list[int] = []
        consumer_row_indices: list[int] = []
        consumer_producer_indices: list[int] = []
        producer_to_index: dict[int, int] = {}
        producer_rows: set[int] = set()
        consumer_rows: set[int] = set()
        assert prefill_state_indices is not None

        for consumer_idx, producer_idx in enumerate(producer_indices_cpu):
            source_block_id = source_block_ids_cpu[consumer_idx]
            if producer_idx < 0 and source_block_id < 0:
                continue
            checkpoint_position = checkpoint_positions_cpu[consumer_idx]
            if checkpoint_position <= 0:
                raise ValueError("grouped GDN checkpoint position must be positive")
            producer_phase_index = -1
            if producer_idx >= 0:
                producer_rows.add(producer_idx)
                if producer_idx not in producer_to_index:
                    producer_to_index[producer_idx] = len(producer_ranges_list)
                    producer_ranges_list.append([query_starts[producer_idx], query_starts[producer_idx + 1]])
                    initial_sources_list.append(NULL_BLOCK_ID)
                    producer_row_indices.append(producer_idx)
                    producer_block_indices.append((checkpoint_position - 1) // self.kv_cache_spec.block_size)
                if source_block_id < 0:
                    producer_phase_index = producer_to_index[producer_idx]
            consumer_rows.add(consumer_idx)
            consumer_row_indices.append(consumer_idx)
            consumer_producer_indices.append(producer_phase_index)
            consumer_ranges_list.append([query_starts[consumer_idx], query_starts[consumer_idx + 1]])

        if hasattr(m, "num_computed_tokens_cpu"):
            has_init_list = [tokens > 0 for tokens in m.num_computed_tokens_cpu.tolist()]
        else:
            has_init_list = has_initial_state.cpu().tolist() if has_initial_state is not None else [False] * m.num_reqs
        for request_idx in range(m.num_reqs):
            if request_idx in producer_rows or request_idx in consumer_rows:
                continue
            start, end = query_starts[request_idx : request_idx + 2]
            if start == end:
                continue
            consumer_row_indices.append(request_idx)
            consumer_producer_indices.append(-1)
            consumer_ranges_list.append([start, end])

        producer_rows_tensor = torch.tensor(producer_row_indices, dtype=torch.int64, device=device)
        producer_blocks_tensor = torch.tensor(producer_block_indices, dtype=torch.int64, device=device)
        shared_destinations = m.block_table_tensor[producer_rows_tensor, producer_blocks_tensor].to(torch.int32)
        consumer_rows_tensor = torch.tensor(consumer_row_indices, dtype=torch.int64, device=device)
        consumer_producers_tensor = torch.tensor(consumer_producer_indices, dtype=torch.int64, device=device)
        private_destinations = prefill_state_indices[consumer_rows_tensor].to(torch.int32)
        if source_block_ids is not None:
            consumer_sources = source_block_ids[consumer_rows_tensor].to(torch.int32).clone()
        else:
            consumer_sources = torch.full((len(consumer_row_indices),), -1, dtype=torch.int32, device=device)
        paired_mask = consumer_producers_tensor >= 0
        consumer_sources[paired_mask] = shared_destinations[consumer_producers_tensor[paired_mask]]
        has_init = torch.tensor(has_init_list, dtype=torch.bool, device=device)[consumer_rows_tensor]
        consumer_sources = torch.where(
            consumer_sources >= 0,
            consumer_sources,
            torch.where(has_init, private_destinations, NULL_BLOCK_ID),
        )

        is_contiguous = bool(
            producer_ranges_list
            and consumer_ranges_list
            and all(
                producer_ranges_list[idx][1] == producer_ranges_list[idx + 1][0]
                for idx in range(len(producer_ranges_list) - 1)
            )
            and all(
                consumer_ranges_list[idx][1] == consumer_ranges_list[idx + 1][0]
                for idx in range(len(consumer_ranges_list) - 1)
            )
        )
        if is_contiguous:
            grouped["producer_slice"] = slice(producer_ranges_list[0][0], producer_ranges_list[-1][1])
            grouped["consumer_slice"] = slice(consumer_ranges_list[0][0], consumer_ranges_list[-1][1])

        def build_packed_phase(
            ranges: list[list[int]],
        ) -> tuple[torch.Tensor, torch.Tensor, GDNChunkedPrefillMetadata]:
            if is_contiguous:
                token_indices = torch.empty(0, dtype=torch.int64, device=device)
            else:
                token_indices = torch.tensor(
                    [token for start, end in ranges for token in range(start, end)],
                    dtype=torch.int64,
                    device=device,
                )
            query_start_loc = [0]
            for start, end in ranges:
                query_start_loc.append(query_start_loc[-1] + end - start)
            query_start_loc_cpu = torch.tensor(query_start_loc, dtype=torch.int32)
            query_start_loc_gpu = query_start_loc_cpu.to(device=device, non_blocking=True)
            chunk_metadata = _build_non_spec_chunked_prefill_metadata(
                self,
                query_start_loc_cpu,
                device,
                build_device_metadata=self.build_device_chunk_metadata,
            )
            return token_indices, query_start_loc_gpu, chunk_metadata

        producer_tokens, producer_qsl, producer_chunk = build_packed_phase(producer_ranges_list)
        consumer_tokens, consumer_qsl, consumer_chunk = build_packed_phase(consumer_ranges_list)
        grouped.update(
            prefix_producer_ranges=torch.tensor(producer_ranges_list, dtype=torch.int32, device=device).reshape(-1, 2),
            producer_token_indices=producer_tokens,
            producer_query_start_loc=producer_qsl,
            consumer_ranges=torch.tensor(consumer_ranges_list, dtype=torch.int32, device=device).reshape(-1, 2),
            consumer_token_indices=consumer_tokens,
            consumer_query_start_loc=consumer_qsl,
            shared_initial_state_source=torch.tensor(initial_sources_list, dtype=torch.int32, device=device),
            shared_state_destinations=shared_destinations,
            consumer_shared_state_sources=consumer_sources,
            private_final_state_destination=private_destinations,
            producer_chunk_metadata=producer_chunk,
            consumer_chunk_metadata=consumer_chunk,
        )
        return grouped

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
    ) -> GDNAttentionMetadata:
        runtime_batch = num_actual_reqs is not None
        # Normalize FIA's full-width dummy queries before slicing requests:
        # its common token count may still include those inactive rows.
        if self.use_full_cuda_graph and self.use_spec_decode:
            common_attn_metadata = _remove_spec_graph_padding_queries(
                common_attn_metadata,
                num_decode_draft_tokens_cpu,
                num_actual_reqs=num_actual_reqs,
            )
        m = common_attn_metadata
        # Common metadata follows the padded graph shape. Build request-phase
        # metadata from the logical batch, then materialize graph-sized buffers.
        graph_request_count = m.num_reqs
        if num_actual_reqs is None:
            num_actual_reqs = graph_request_count
        elif not 0 <= num_actual_reqs <= graph_request_count:
            raise ValueError(
                f"num_actual_reqs ({num_actual_reqs}) must be between 0 and graph_request_count ({graph_request_count})"
            )

        if num_actual_reqs < graph_request_count:
            m = m.unpadded(m.num_actual_tokens, num_actual_reqs)
            if num_accepted_tokens is not None:
                num_accepted_tokens = num_accepted_tokens[:num_actual_reqs]
            if num_decode_draft_tokens_cpu is not None:
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu[:num_actual_reqs]
        m = _treat_single_token_prefills_with_state_as_decodes(m)

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_indices: torch.Tensor | None = None
        non_spec_sequence_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
            # Zero-draft adaptive rows still need the previous accepted state.
            torch.ge(num_decode_draft_tokens_cpu, 0, out=spec_sequence_masks_cpu)
            adaptive = bool(
                self.speculative_config and getattr(self.speculative_config, "enable_adaptive_verification", False)
            )
            if not adaptive and not torch.any(num_decode_draft_tokens_cpu > 0).item():
                # Static graphs retain the native no-draft decode route.
                spec_sequence_masks_cpu.zero_()
            num_spec_decodes = int(spec_sequence_masks_cpu.sum())
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]

        batch = _split_gdn_batch(m, spec_sequence_masks_cpu, num_spec_decodes)
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
            num_spec_decode_tokens,
            non_spec_query_start_loc_cpu,
        ) = batch
        if runtime_batch and num_actual_reqs > 0:
            fused = self._build_fused_metadata(
                m, num_accepted_tokens, spec_sequence_masks_cpu, num_spec_decodes, graph_request_count, batch
            )
            if fused is not None:
                return fused

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        mamba_block_table = m.block_table_tensor
        # Restrict recurrent state lookup to real request rows.
        if mamba_cache_mode != "none" and mamba_block_table.shape[0] != m.seq_lens.shape[0]:
            mamba_block_table = mamba_block_table[: m.seq_lens.shape[0]]
        block_table_tensor = mamba_get_block_table_tensor(
            mamba_block_table,
            m.seq_lens,
            self.kv_cache_spec,
            mamba_cache_mode,
        )

        non_spec_conv1d_cache_indices: torch.Tensor | None = None
        if spec_sequence_masks is not None:
            assert spec_sequence_masks_cpu is not None
            spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)

        if spec_sequence_masks is None:
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            non_spec_conv1d_cache_indices = block_table_tensor
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                # Reuse vLLM's pure-spec path: real requests precede padding,
                # so neither device sequence indices nor a gather are needed.
                spec_state_indices_tensor = block_table_tensor[:num_spec_decodes, : self.num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                )
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, : self.num_spec + 1],
                    0,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, 0],
                    0,
                    non_spec_sequence_indices,
                )
                non_spec_conv1d_cache_indices = non_spec_state_indices_tensor
                spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    spec_sequence_indices,
                )
                non_spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    non_spec_sequence_indices,
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    non_spec_query_lens,
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )

            assert num_accepted_tokens is not None
            if num_prefills == 0 and num_decodes == 0:
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                num_accepted_tokens = torch.index_select(num_accepted_tokens, 0, spec_sequence_indices)

        # A FULL graph retains captured speculative conv/recurrent tasks. Clear
        # their stable inputs on every no-spec replay so an idle or prefill
        # batch cannot mutate state belonging to the preceding request.
        if self.use_full_cuda_graph and self.use_spec_decode and num_spec_decodes == 0:
            self._reset_spec_decode_graph_inputs(graph_request_count)

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_query_start_loc_cpu: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        non_spec_chunked_prefill_metadata: GDNChunkedPrefillMetadata | None = None
        if num_prefills > 0:
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            assert prefill_query_start_loc_cpu is not None
            if self._USE_COMMON_KERNEL_METADATA:
                non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                    self,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                    build_device_metadata=self.build_device_chunk_metadata,
                )
                # Common GDN/KDA callers also consume the upstream chunk fields.
                chunk_indices = non_spec_chunked_prefill_metadata.chunk_indices_chunk64
                chunk_offsets = non_spec_chunked_prefill_metadata.chunk_offsets_chunk64

        if num_prefills > 0:
            has_initial_state = self._build_prefill_has_initial_state(
                context_lens_tensor=m.compute_num_computed_tokens(),
                non_spec_sequence_indices=non_spec_sequence_indices,
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        grouped_metadata = self._build_grouped_prefill_metadata(
            m,
            spec_sequence_masks,
            num_decodes,
            num_prefills,
            prefill_state_indices,
            has_initial_state,
        )
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        attn_metadata = AscendGDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            **grouped_metadata,
        )
        if (
            self.use_full_cuda_graph
            and num_spec_decodes > 0
            and num_prefills == 0
            and num_decodes == 0
            and self._can_pad_spec_decode(graph_request_count, num_spec_decode_tokens)
        ):
            self._pad_spec_decode_metadata(attn_metadata, graph_request_count)
        elif (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and graph_request_count <= self.decode_cudagraph_max_bs
        ):
            self._pad_decode_metadata(attn_metadata, graph_request_count)
            non_spec_conv1d_cache_indices = attn_metadata.non_spec_state_indices_tensor
        # Attach once, using the final graph-stable Tensor references.
        self._attach_non_spec_prefill_metadata(
            attn_metadata, non_spec_chunked_prefill_metadata, non_spec_conv1d_cache_indices
        )
        self._attach_spec_decode_metadata(attn_metadata)
        self._attach_non_spec_decode_metadata(attn_metadata, non_spec_conv1d_cache_indices)
        return self._finalize_metadata(attn_metadata)

    def _finalize_metadata(self, metadata: GDNAttentionMetadata, fused: bool = False) -> GDNAttentionMetadata:
        return metadata

    def _build_prefill_has_initial_state(
        self,
        *,
        context_lens_tensor: torch.Tensor,
        non_spec_sequence_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        has_initial_state = context_lens_tensor > 0
        if non_spec_sequence_indices is not None:
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
        return has_initial_state


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Decode kernels consume device boundaries; CPU lengths only size
        # the batch and classify request phases during adaptive verification.
        return True

    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder


class AscendGDNHostMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Build only the descriptors consumed by AscendC host-metadata paths."""

    build_device_chunk_metadata = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._conv_layer_name = layer_names[0] if layer_names else None
        self._spec_conv_indices: torch.Tensor | None = None
        self._non_spec_conv_indices: torch.Tensor | None = None
        self._conv_output_buffer: torch.Tensor | None = None
        self._fused_layout_cpu: torch.Tensor | None = None
        self._fused_layout: torch.Tensor | None = None

    def _ensure_conv_output_buffer(self) -> None:
        if self._conv_output_buffer is not None:
            return
        layers = getattr(self.compilation_config, "static_forward_context", {})
        layer = layers.get(self._conv_layer_name)
        if layer is None or not getattr(layer, "use_shared_conv_output", False):
            return
        # Decode needs at most K+1 tokens per request; prefill does not use this
        # buffer. Include graph padding without reserving the prefill token budget.
        token_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            self.decode_cudagraph_max_bs,
        )
        self._conv_output_buffer = torch.empty(
            (token_capacity, 3 * layer.local_projection_size),
            dtype=self.vllm_config.model_config.dtype,
            device=self.spec_sequence_masks.device,
        )

    def _ensure_conv_index_buffers(self) -> None:
        if self._spec_conv_indices is not None:
            return
        # Resolve the group's representative layer when building metadata.
        # Only opted-in models allocate the additional convolution buffers.
        layers = getattr(self.compilation_config, "static_forward_context", {})
        if not getattr(layers.get(self._conv_layer_name), "use_contiguous_conv_indices", False):
            return
        # Each group owns its buffers, with fixed addresses for graph replay.
        device = self.spec_sequence_masks.device
        self._spec_conv_indices = torch.empty(self.spec_sequence_masks.numel(), dtype=torch.int32, device=device)
        self._non_spec_conv_indices = torch.empty_like(self._spec_conv_indices)

    @staticmethod
    def _prepare_conv_indices(indices: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        # Mixed batches and ordinary graph decode already provide a contiguous
        # vector. Matrix sources are prepared once, including single-row MTP.
        if indices.ndim == 1 and indices.is_contiguous():
            return indices
        column = indices[:, 0] if indices.ndim == 2 else indices
        requests = column.numel()
        if requests > buffer.numel():
            raise ValueError("Convolution index count exceeds the cache-group buffer capacity")
        prepared = buffer[:requests]
        prepared.copy_(column, non_blocking=True)
        return prepared

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        super()._reset_spec_decode_graph_inputs(graph_batch_size)
        if self._spec_conv_indices is not None:
            self._spec_conv_indices[:graph_batch_size].fill_(PAD_SLOT_ID)

    def _build_fused_metadata(
        self,
        m: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        spec_sequence_masks_cpu: torch.Tensor | None,
        num_spec_decodes: int,
        graph_request_count: int,
        batch: tuple[int, int, int, int, int, torch.Tensor | None],
    ) -> GDNAttentionMetadata | None:
        # Request validation and phase classification have already run in build.
        if (
            self._spec_conv_indices is None
            or self.vllm_config.cache_config.mamba_cache_mode != "align"
            or (num_spec_decodes > 0 and self.kv_cache_spec.num_speculative_blocks < self.num_spec)
        ):
            return None

        mixed_spec = 0 < num_spec_decodes < m.num_reqs
        if (
            mixed_spec
            and self.speculative_config
            and getattr(self.speculative_config, "enable_adaptive_verification", False)
        ):
            # The fused mixed layout uses CPU group offsets. Adaptive drafts
            # can redistribute lengths on device; the generic path builds
            # group boundaries and token partitions from those device lengths.
            return None
        assert not num_spec_decodes or num_accepted_tokens is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens, num_spec_tokens, non_spec_query_cpu = batch
        spec_rows = num_spec_decodes
        non_spec_rows = m.num_reqs - num_spec_decodes
        if self.use_full_cuda_graph and num_prefills == 0:
            if num_spec_decodes and self._can_pad_spec_decode(graph_request_count, num_spec_tokens):
                spec_rows = graph_request_count
            elif not num_spec_decodes and graph_request_count <= self.decode_cudagraph_max_bs:
                non_spec_rows = graph_request_count
        if self.use_full_cuda_graph and self.use_spec_decode and not num_spec_decodes:
            self._reset_spec_decode_graph_inputs(graph_request_count)

        # Reuse captured addresses; larger eager batches need only fresh views.
        def buffer_view(buffer: torch.Tensor, rows: int) -> torch.Tensor:
            return buffer[:rows] if rows <= buffer.shape[0] else buffer.new_empty((rows, *buffer.shape[1:]))

        spec_states = buffer_view(self.spec_state_indices_tensor, spec_rows)
        non_spec_states = buffer_view(self.non_spec_state_indices_tensor, non_spec_rows)
        spec_query = buffer_view(self.spec_query_start_loc, spec_rows + 1)
        non_spec_query = buffer_view(self.non_spec_query_start_loc, non_spec_rows + 1)
        assert self.spec_actual_seq_lengths is not None and self.non_spec_actual_seq_lengths is not None
        spec_lengths = self.spec_actual_seq_lengths[: spec_rows + 1]
        non_spec_lengths = self.non_spec_actual_seq_lengths[: non_spec_rows + 1]
        accepted = buffer_view(self.num_accepted_tokens, spec_rows)
        has_initial_state = m.seq_lens.new_empty(non_spec_rows, dtype=torch.bool) if num_prefills else None
        prefill_query = non_spec_query.new_empty(non_spec_rows - num_decodes + 1) if num_prefills else None
        spec_tokens = buffer_view(self.spec_token_indx, num_spec_tokens)
        non_spec_tokens = self.non_spec_token_indx[:0]
        layout = None
        if mixed_spec:
            # One compact transfer replaces the mask and two sequence-index copies.
            # The kernel uses this stable CPU partition for both requests and tokens.
            if self._fused_layout_cpu is None:
                self._fused_layout_cpu = torch.empty(
                    (self.spec_sequence_masks.numel(), 2),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=m.seq_lens.device.type != "cpu",
                )
                self._fused_layout = torch.empty_like(self._fused_layout_cpu, device=m.seq_lens.device)
            assert self._fused_layout is not None and spec_sequence_masks_cpu is not None
            plan = self._fused_layout_cpu[: m.num_reqs]
            mask = spec_sequence_masks_cpu
            assert non_spec_query_cpu is not None
            query_lens_cpu = torch.diff(m.query_start_loc_cpu)
            plan[mask, 0] = torch.arange(num_spec_decodes, dtype=torch.int32)
            plan[~mask, 0] = -torch.arange(non_spec_rows, dtype=torch.int32) - 1
            spec_query_cpu = torch.cat((query_lens_cpu.new_zeros(1), query_lens_cpu[mask].cumsum(0)))
            plan[mask, 1] = spec_query_cpu[:-1].to(torch.int32)
            plan[~mask, 1] = non_spec_query_cpu[:-1].to(torch.int32)
            layout = self._fused_layout[: m.num_reqs]
            layout.copy_(plan, non_blocking=True)
            # Match the generic stable partition's token-index dtype.
            spec_tokens = m.seq_lens.new_empty(num_spec_tokens, dtype=torch.int64)
            non_spec_tokens = m.seq_lens.new_empty(num_prefill_tokens, dtype=torch.int64)

        prepare_gdn(
            m.block_table_tensor,
            m.seq_lens,
            m.query_start_loc,
            num_accepted_tokens,
            layout,
            spec_states,
            non_spec_states,
            self._spec_conv_indices,
            self.spec_sequence_masks,
            spec_query,
            non_spec_query,
            spec_lengths,
            non_spec_lengths,
            accepted,
            spec_tokens,
            non_spec_tokens,
            has_initial_state,
            prefill_query,
            self._conv_output_buffer,
            num_reqs=m.num_reqs,
            num_spec_decodes=num_spec_decodes,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            has_prefill=num_prefills > 0,
            spec_width=self.num_spec + 1,
            mixed_spec=mixed_spec,
            block_size=self.kv_cache_spec.block_size,
            pad_block_id=self._SPEC_GRAPH_PAD_SLOT_ID,
        )
        metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            prefill_query_start_loc=prefill_query,
            prefill_state_indices=non_spec_states[num_decodes:] if num_prefills else None,
            prefill_has_initial_state=has_initial_state[num_decodes:] if has_initial_state is not None else None,
            spec_state_indices_tensor=spec_states if num_spec_decodes else None,
            spec_sequence_masks=self.spec_sequence_masks[: max(m.num_reqs, spec_rows)] if num_spec_decodes else None,
            spec_query_start_loc=spec_query if num_spec_decodes else None,
            num_accepted_tokens=accepted if num_spec_decodes else None,
            spec_token_indx=spec_tokens if num_spec_decodes else None,
            non_spec_token_indx=non_spec_tokens if num_spec_decodes else None,
            non_spec_state_indices_tensor=non_spec_states if non_spec_rows else None,
            non_spec_query_start_loc=non_spec_query if non_spec_rows else None,
        )
        chunk = None
        if num_prefills:
            assert non_spec_query_cpu is not None
            chunk = _build_non_spec_chunked_prefill_metadata(
                self,
                non_spec_query_cpu[num_decodes:] - num_decode_tokens,
                m.seq_lens.device,
                build_device_metadata=False,
            )
        self._attach_non_spec_prefill_metadata(metadata, chunk, non_spec_states)
        self._attach_spec_decode_metadata(metadata, spec_lengths, self._spec_conv_indices)
        self._attach_non_spec_decode_metadata(metadata, non_spec_states, non_spec_lengths)
        return self._finalize_metadata(metadata, fused=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
    ) -> GDNAttentionMetadata:
        self._ensure_conv_index_buffers()
        self._ensure_conv_output_buffer()
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
        )
        metadata.pcp_context = getattr(common_attn_metadata, "pcp_context", None)
        return metadata

    def _finalize_metadata(self, metadata: GDNAttentionMetadata, fused: bool = False) -> GDNAttentionMetadata:
        if self._conv_output_buffer is not None:
            # Layers in this group consume the same decode rows in sequence.
            # Clear once per step, including inactive captured branches: rows
            # that were active on the previous replay can now be padding.
            if not fused:
                self._conv_output_buffer.zero_()
            if metadata.spec_decode_metadata is not None:
                metadata.spec_decode_metadata.spec_causal_conv1d.output_buffer = self._conv_output_buffer
            if metadata.non_spec_decode_metadata is not None:
                metadata.non_spec_decode_metadata.causal_conv1d.output_buffer = self._conv_output_buffer
        if fused or self._spec_conv_indices is None:
            return metadata
        assert self._non_spec_conv_indices is not None
        # Parent build has finalized request counts and graph padding. Keep
        # recurrent indices intact and only replace the convolution view.
        if metadata.spec_decode_metadata is not None:
            conv = metadata.spec_decode_metadata.spec_causal_conv1d
            conv.cache_indices = self._prepare_conv_indices(conv.cache_indices, self._spec_conv_indices)
        non_spec_phases = (metadata.non_spec_prefill_metadata, metadata.non_spec_decode_metadata)
        conv_metadata = [phase.causal_conv1d for phase in non_spec_phases if phase is not None]
        if conv_metadata:
            # Both phases reference prefixes of the same non-spec index table.
            source = max(conv_metadata, key=lambda conv: conv.cache_indices.shape[0])
            prepared = self._prepare_conv_indices(source.cache_indices, self._non_spec_conv_indices)
            for conv in conv_metadata:
                conv.cache_indices = prepared[: conv.cache_indices.shape[0]]
        return metadata


class AscendGDNHostMetadataBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[GDNAttentionMetadataBuilder]:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config.hf_text_config.model_type in (
            "glm5_next",
            "glm5_next_text",
        ):
            # The RecoverSSM builder imports these host descriptors. Resolve it
            # lazily. Selection only reads the opt-in validated at config init.
            from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm

            if use_glm_recoverssm(vllm_config):
                from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder

                return AscendKDARecoverSSMMetadataBuilder
        return AscendGDNHostMetadataBuilder


class AscendGDNFusedAttentionBackend(AscendGDNAttentionBackend):
    """Select host-only metadata when Qwen can use a fused prefill op."""

    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        # Import lazily to avoid the gdn <-> builder import cycle at module
        # initialization.  The probe result is cached by the attention class.
        from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

        if get_pcp_group().world_size == 1 and AscendGatedDeltaNetAttention._supports_host_metadata_prefill():
            return AscendGDNHostMetadataBuilder
        return AscendGDNAttentionMetadataBuilder
