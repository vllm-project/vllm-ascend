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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
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

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata
    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
        TreeVerifyPlan,
        TreeVerifyPlanBuffers,
    )

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18
# Accepted-token counts are one-based. The zero-length query segment makes
# graph-padding rows no-ops, while 1 keeps their count valid for GDN kernels.
_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1

# Per-step shared spec-split bookkeeping; see _get_shared_spec_setup.
_SHARED_SPEC_SETUP: tuple | None = None


def _get_shared_spec_setup(
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor,
    num_reqs: int,
) -> dict:
    """Host-side speculative-decode split, computed once per step from CPU inputs."""
    global _SHARED_SPEC_SETUP
    qsl_np = query_start_loc_cpu.numpy()
    draft_np = num_decode_draft_tokens_cpu.numpy()
    key = (qsl_np.tobytes(), draft_np.tobytes())
    cached = _SHARED_SPEC_SETUP
    if cached is not None and cached[0] == key:
        return cached[1]
    # Requests that carry at least one draft token take the speculative path.
    # A batch whose active rows all carry zero drafts keeps the plain
    # prefill/decode split (mirrors the previous ge/sum/item semantics).
    mask_np = draft_np >= 0 if np.any(draft_np > 0) else np.zeros(num_reqs, dtype=bool)
    num_spec_decodes = int(mask_np.sum())
    query_lens_np = np.diff(qsl_np)
    non_spec_lens_np = query_lens_np[~mask_np]
    setup = {
        "mask_np": mask_np,
        "num_spec_decodes": num_spec_decodes,
        "num_zero_len": int((non_spec_lens_np == 0).sum()),
        "spec_indices_np": np.nonzero(mask_np)[0].astype(np.int64) if num_spec_decodes else None,
        "non_spec_indices_np": np.nonzero(~mask_np)[0].astype(np.int64) if num_spec_decodes else None,
    }
    _SHARED_SPEC_SETUP = (key, setup)
    return setup


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
class GDNDecodeMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor


def _tree_conv_width(kv_cache_spec: AttentionSpec, mamba_type, state_columns: int) -> int | None:
    """Short-conv kernel width of a GDN state spec (``None`` when not GDN).

    ``state_columns`` is the number of speculative state columns the conv-state
    shape carries: ``num_speculative_tokens`` while the spec still has the
    linear-chain layout, ``slots`` once ``attn_utils`` widened it to the
    draft-tree width.
    """
    shapes = getattr(kv_cache_spec, "shapes", None)
    if not shapes:
        return None
    # Lazy import: the tree spec-decode package imports this module.
    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import tree_conv_width

    return tree_conv_width(mamba_type, tuple(shapes[0]), state_columns)


def _pack_tree_rows_to_tokens(
    table: torch.Tensor,
    row_widths: torch.Tensor,
    slots: int,
) -> torch.Tensor:
    """Pack a ``[rows, slots]`` tree table into token-major order.

    The recurrent kernel reads ``ssm_state_indices`` and ``init_state_indices``
    by global token offset (``ssmStateIndicesGm_.GetValue(seq_i)`` in
    ``csrc/attention/recurrent_gated_delta_rule``), while the tree tables are
    indexed as ``[row, node]``. Every row normally verifies the full
    ``1 + budget = slots`` tokens, but the scheduler truncates the draft of a
    request near ``max_model_len`` and the tree search can run out of candidates,
    so the rows have to be laid out back to back: token ``p`` of the step takes
    node ``p - start(row)`` of its row. The result keeps the ``[rows, slots]``
    shape because the op receives ``reshape(-1)``, and only the slots past the
    packed tokens hold ``NULL_BLOCK_ID`` -- the kernel never reaches them.
    """
    rows = table.shape[0]
    widths = row_widths.to(device=table.device, dtype=torch.long).reshape(-1)
    if widths.numel() != rows:
        raise ValueError(f"got {widths.numel()} row widths for {rows} tree rows.")
    starts = torch.cumsum(widths, dim=0) - widths
    ends = starts + widths
    positions = torch.arange(rows * slots, device=table.device)
    token_rows = (positions.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1).clamp(max=rows - 1)
    token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
    packed = table.reshape(-1).index_select(0, token_rows * slots + token_cols)
    packed = packed.masked_fill(positions >= ends[-1], NULL_BLOCK_ID)
    return packed.view(rows, slots)


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
    # Draft-tree only: per-token state page the recurrent kernel loads its
    # initial state from, see spec_decode/tree/state_rollback.py. ``None`` keeps
    # the linear-chain behaviour of the kernel.
    init_state_indices: torch.Tensor | None = None
    # Draft-tree only: conv tap rows and committed-history source columns.
    conv_window_rows: torch.Tensor | None = None
    conv_prefix_src: torch.Tensor | None = None
    # Draft-tree only: indexed verification over fixed-width token rows.
    verify_plan: "TreeVerifyPlan | None" = None


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
        context = getattr(vllm_config.compilation_config, "static_forward_context", {}) or {}
        self._ascend_gdn_checkpoints = (
            bool(layer_names)
            and all(getattr(context.get(name), "_ascend_internal_checkpoint_enabled", False) for name in layer_names)
            and bool(getattr(kv_cache_spec, "num_prefill_checkpoint_blocks", 0))
        )
        if getattr(kv_cache_spec, "num_prefill_checkpoint_blocks", 0) and not self._ascend_gdn_checkpoints:
            raise ValueError("Internal GDN checkpoint specs require Ascend-owned checkpoint-capable layers")
        # Draft-tree state layout, filled in by _resize_spec_buffers_for_tree;
        # ``None`` means the linear MTP chain (default).
        self.spec_tree_slots: int | None = None
        self.spec_init_state_indices: torch.Tensor | None = None
        self.spec_conv_width: int | None = None
        self.spec_conv_window_rows: torch.Tensor | None = None
        self.spec_conv_prefix_src: torch.Tensor | None = None
        # Stable tensors for the indexed Tree verifier under FULL replay.
        self.spec_verify_plan_buffers: TreeVerifyPlanBuffers | None = None
        self._spec_verify_plan_views: TreeVerifyPlan | None = None
        self._resize_spec_buffers_for_tree(device)
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

    def _resize_spec_buffers_for_tree(self, device: torch.device) -> None:
        """Follow the draft-tree state width in the spec-sized buffers.

        ``GDNAttentionMetadataBuilder.__init__`` sizes every spec-sized buffer
        with ``1 + num_speculative_tokens``, the number of queries of a linear
        draft chain, but a draft tree verifies ``1 + budget`` tokens per request
        and caches one state per node. ``MambaSpec`` is widened as needed by
        ``attn_utils.get_kv_cache_spec``. The tensors allocated below mirror the
        upstream constructor and have to stay in sync with it.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_state_slot_count,
            validate_tree_state_slots,
        )

        slots = tree_state_slot_count(self.vllm_config)
        if slots is None:
            return
        mamba_type = getattr(self.kv_cache_spec, "mamba_type", None)
        if mamba_type is not None:
            validate_tree_state_slots(slots, mamba_type)
        self.spec_tree_slots = slots
        self.spec_init_state_indices = torch.empty(
            (self.decode_cudagraph_max_bs, slots), dtype=torch.int32, device=device
        )
        # ``attn_utils.get_kv_cache_spec`` widens the conv-state shape from the
        # linear-chain ``num_spec`` state columns to the tree's ``slots`` ones as
        # soon as the tree is wider than the chain, so the kernel width has to be
        # recovered with the column count the shape actually carries.
        state_columns = max(self.num_spec, slots)
        conv_width = _tree_conv_width(self.kv_cache_spec, mamba_type, state_columns)
        self.spec_conv_width = conv_width
        if conv_width is not None:
            # Row tables of the tree short conv, indexed by the spec-sized batch
            # and, for the windows, by its tokens; see _build_tree_conv_rows.
            self.spec_conv_window_rows = torch.empty(
                (self.decode_cudagraph_max_bs * slots, conv_width), dtype=torch.int32, device=device
            )
            self.spec_conv_prefix_src = torch.empty(
                (self.decode_cudagraph_max_bs, conv_width - 1), dtype=torch.int32, device=device
            )
        if slots == self.num_spec + 1:
            # The tree is exactly as wide as the draft chain, so the buffers the
            # upstream constructor allocated already carry its width. The verify
            # may still outgrow one op call, hence the chunk buffers first.
            self._allocate_tree_verify_plan_buffers(device, slots)
            return

        self.num_spec = slots - 1
        self.decode_cudagraph_max_bs = self.vllm_config.scheduler_config.max_num_seqs * slots
        max_capture_size = self.compilation_config.max_cudagraph_capture_size
        if max_capture_size is not None:
            self.decode_cudagraph_max_bs = min(self.decode_cudagraph_max_bs, max_capture_size)

        max_bs = self.decode_cudagraph_max_bs
        self.spec_state_indices_tensor = torch.empty((max_bs, slots), dtype=torch.int32, device=device)
        self.non_spec_state_indices_tensor = torch.empty((max_bs,), dtype=torch.int32, device=device)
        self.spec_token_indx = torch.empty((max_bs * slots,), dtype=torch.int32, device=device)
        self.non_spec_token_indx = torch.empty((max_bs * slots,), dtype=torch.int32, device=device)
        self.spec_query_start_loc = torch.empty((max_bs + 1,), dtype=torch.int32, device=device)
        self.non_spec_query_start_loc = torch.empty((max_bs + 1,), dtype=torch.int32, device=device)
        self.num_accepted_tokens = torch.empty((max_bs,), dtype=torch.int32, device=device)
        if self.use_spec_decode and self.reorder_batch_threshold != 1:
            # The threshold counts the queries of one verification batch, and a
            # tree verification always runs the root plus every node.
            self.reorder_batch_threshold = slots

        self._allocate_tree_verify_plan_buffers(device, slots)

    def _allocate_tree_verify_plan_buffers(self, device: torch.device, slots: int) -> None:
        """Allocate stable indexed-plan tensors for wide-tree graph replay.

        Graphs retain these addresses while each step updates their values.
        Eager verification can allocate a fresh plan instead.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            GDN_TREE_DIRECT_VERIFY_MAX_SLOTS,
            allocate_tree_verify_plan_buffers,
        )

        if slots <= GDN_TREE_DIRECT_VERIFY_MAX_SLOTS or not self.use_full_cuda_graph:
            return
        self.spec_verify_plan_buffers = allocate_tree_verify_plan_buffers(slots, self.decode_cudagraph_max_bs, device)

    def _build_tree_verify_plan(
        self,
        row_widths: torch.Tensor,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor | None,
        scratch_base: int,
        shared_gdn_tree: "SharedGDNTreeMetadata | None" = None,
    ) -> "TreeVerifyPlan | None":
        """Build the indexed plan for trees above the direct-path threshold.

        ``None`` keeps direct verification for small trees. Wider trees use one
        whole-tree Triton call with token indirection and stable state tables.

        A full graph reads its plan tensors by address, so the live data is
        copied into the builder's stable buffers before replay.
        ``row_widths`` has one entry per op row, i.e. including the padded rows of
        a graph batch (they have length 0 and the kernel skips them).

        Both tables have to be in the packed token-major order of the step
        (``_pack_tree_rows_to_tokens``), where ``table.reshape(-1)[p]`` is token
        ``p`` of the step; the caller only builds a plan when that packing ran.
        """
        if init_state_indices is None or self.spec_tree_slots is None:
            return None
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            GDN_TREE_DIRECT_VERIFY_MAX_SLOTS,
            compute_tree_verify_plan,
        )

        if state_indices.shape[1] <= GDN_TREE_DIRECT_VERIFY_MAX_SLOTS:
            return None
        plan = (
            shared_gdn_tree.bind_plan(state_indices, init_state_indices)
            if shared_gdn_tree is not None
            else compute_tree_verify_plan(row_widths, state_indices, init_state_indices, scratch_base)
        )
        if not self.use_full_cuda_graph:
            return plan
        if self.spec_verify_plan_buffers is None:
            raise RuntimeError("FULL graph replay requires stable buffers for indexed GDN Tree verification.")

        graph_plan = self._spec_verify_plan_views
        if graph_plan is None or (graph_plan.rows, graph_plan.slots) != (plan.rows, plan.slots):
            graph_plan = self.spec_verify_plan_buffers.views(plan.rows, plan.slots)
            self._spec_verify_plan_views = graph_plan
        plan.copy_into(graph_plan)
        return graph_plan

    def _build_tree_init_state_indices(
        self,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        query_lens_cpu: torch.Tensor,
        spec_sequence_masks_cpu: torch.Tensor,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-token state page each draft-tree node starts from.

        The result keeps the ``[rows, slots]`` shape but is packed token-major,
        because the recurrent kernel indexes the table by global token offset.
        Returns ``None`` for the linear MTP chain, where the recurrent kernel
        carries the state of the previous token in registers.
        """
        if self.spec_tree_slots is None or state_indices is None:
            return None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN state rollback needs the tree layout; "
                "tree_parents/tree_num_nodes were not passed to build()."
            )
        slots = self.spec_tree_slots
        # A row may verify fewer than ``1 + budget`` tokens: the scheduler
        # truncates the draft of a request near ``max_model_len`` and the tree
        # search can run out of candidates. The tables are packed token-major
        # below, so only the per-row bounds matter here.
        spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
        if bool(((spec_query_lens_cpu < 1) | (spec_query_lens_cpu > slots)).any()):
            raise ValueError(
                "draft-tree GDN state rollback needs 1 <= verified tokens <= "
                f"1 + budget = {slots} per speculative request, got "
                f"{spec_query_lens_cpu.tolist()}."
            )
        # Lazy import: the tree spec-decode package imports this module.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_init_state_indices,
        )

        init_state_indices = compute_tree_init_state_indices(
            state_indices,
            num_accepted_tokens,
            torch.index_select(tree_parents, 0, spec_sequence_indices),
            torch.index_select(tree_num_nodes, 0, spec_sequence_indices),
        )
        return _pack_tree_rows_to_tokens(init_state_indices, spec_query_lens_cpu, slots)

    def _select_spec_state_indices(
        self,
        block_table_tensor: torch.Tensor,
        spec_sequence_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select the per-request GDN state pages for speculative rows.

        The chain path keeps its original ``num_spec + 1`` pages.  A tree
        verify owns one page per verified node, so its table must use the
        widened ``1 + budget`` capacity allocated by ``attn_utils``.  Using
        the chain width here silently aliases tree nodes and corrupts the
        recurrent state after the first branch.
        """
        if self.spec_tree_slots is None:
            return torch.index_select(
                block_table_tensor[:, : self.num_spec + 1],
                0,
                spec_sequence_indices,
            )
        return torch.index_select(
            block_table_tensor[:, : self.spec_tree_slots],
            0,
            spec_sequence_indices,
        )

    def _build_tree_conv_rows(
        self,
        num_accepted_tokens: torch.Tensor,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        row_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Conv tap table of every draft-tree node and the committed history.

        ``window_rows`` indexes ``cat([committed history, this step's
        activations])`` and ``prefix_src`` names the conv-state columns of the
        history the next step resumes from, both consumed by
        ``tree_causal_conv1d_triton``. ``(None, None)`` keeps the
        sliding-window custom operator of the linear MTP chain.
        """
        if self.spec_tree_slots is None or self.spec_conv_window_rows is None:
            return None, None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN state rollback needs the tree layout; "
                "tree_parents/tree_num_nodes were not passed to build()."
            )
        if prev_path_node_ids is None or prev_num_sampled is None:
            raise RuntimeError(
                "draft-tree GDN short conv needs the accepted path of the previous step; it was not passed to build()."
            )
        rows = num_accepted_tokens.shape[0]
        assert self.spec_conv_width is not None
        # Lazy import: the tree spec-decode package imports this module.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_conv_rows,
            compute_tree_prefix_source,
        )

        window_rows = compute_tree_conv_rows(
            torch.index_select(tree_parents, 0, spec_sequence_indices),
            torch.index_select(tree_num_nodes, 0, spec_sequence_indices),
            self.spec_conv_width,
            # The committed history of every request comes first, then this
            # step's activations, which are laid out request-major over the
            # tokens each request actually verifies.
            rows * (self.spec_conv_width - 1),
            row_widths=row_widths,
        )
        prefix_src = compute_tree_prefix_source(
            torch.index_select(prev_path_node_ids, 0, spec_sequence_indices),
            torch.index_select(prev_num_sampled, 0, spec_sequence_indices),
            self.spec_conv_width,
        )
        return window_rows, prefix_src

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
        spec_indices_np: np.ndarray | None = None,
        non_spec_indices_np: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        if spec_indices_np is not None:
            # Shared per-step numpy result (see _get_shared_spec_setup).
            spec_indices_cpu.numpy()[:] = spec_indices_np
        else:
            spec_indices_cpu.copy_(
                torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
            )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        if non_spec_indices_np is not None:
            non_spec_indices_cpu.numpy()[:] = non_spec_indices_np
        else:
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
        if self.spec_init_state_indices is not None:
            self.spec_init_state_indices[:graph_batch_size].fill_(NULL_BLOCK_ID)
        if self.spec_verify_plan_buffers is not None:
            self.spec_verify_plan_buffers.zero_lengths()

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
        init_state_indices: torch.Tensor | None = None,
        conv_window_rows: torch.Tensor | None = None,
        conv_prefix_src: torch.Tensor | None = None,
        shared_gdn_tree: "SharedGDNTreeMetadata | None" = None,
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
        if conv_window_rows is not None and actual_seq_lengths is None:
            actual_seq_lengths = _build_actual_seq_lengths(
                attn_metadata.spec_query_start_loc,
                num_sequences,
                actual_seq_lengths_buffer,
            )
        # The indexed plan reads packed state tables. Tree packing runs when
        # the convolution has tree rows. Padding has query length zero.
        verify_plan = None
        if conv_window_rows is not None:
            verify_plan = self._build_tree_verify_plan(
                actual_seq_lengths[1:],
                attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                None if init_state_indices is None else init_state_indices[:spec_num_rows],
                attn_metadata.num_actual_tokens,
                shared_gdn_tree,
            )

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
                if self._USE_COMMON_KERNEL_METADATA and (init_state_indices is not None or not use_upstream_recurrent)
                else None
            ),
            init_state_indices=init_state_indices,
            conv_window_rows=conv_window_rows,
            conv_prefix_src=conv_prefix_src,
            verify_plan=verify_plan,
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

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        shared_gdn_tree: "SharedGDNTreeMetadata | None" = None,
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
        spec_setup = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        elif self.spec_tree_slots is None:
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

        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            # One shared numpy pass per step instead of an aten dispatch
            # chain (ge/sum/item/nonzero) per cache group.
            spec_setup = _get_shared_spec_setup(
                m.query_start_loc_cpu,
                num_decode_draft_tokens_cpu,
                num_reqs,
            )
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
            mask_np = spec_setup["mask_np"]
            if bool(mask_np.any()):
                spec_sequence_masks_cpu.numpy()[:] = mask_np
            else:
                # Dynamic speculative decoding can be enabled while this batch
                # carries no draft tokens. Keep the normal prefill/decode split.
                spec_sequence_masks_cpu.numpy()[:].fill(False)
            num_spec_decodes = spec_setup["num_spec_decodes"]
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
                spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                    spec_setup["spec_indices_np"],
                    spec_setup["non_spec_indices_np"],
                )

        batch = _split_gdn_batch(m, spec_sequence_masks_cpu, num_spec_decodes)
        (
            num_decodes,
            num_prefills,
            num_decode_tokens,
            num_prefill_tokens,
            num_spec_decode_tokens,
            non_spec_query_start_loc_cpu,
        ) = batch
        if (
            self.spec_tree_slots is None
            and runtime_batch
            and num_actual_reqs > 0
            and not (self._ascend_gdn_checkpoints and num_prefills)
        ):
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
        if spec_sequence_masks is not None and self.spec_tree_slots is None:
            assert spec_sequence_masks_cpu is not None
            spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)

        spec_init_state_indices: torch.Tensor | None = None
        spec_conv_window_rows: torch.Tensor | None = None
        spec_conv_prefix_src: torch.Tensor | None = None
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
            assert spec_sequence_masks_cpu is not None
            total_tokens = m.num_actual_tokens if spec_setup is not None else query_start_loc_cpu[-1].item()

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    total_tokens,
                )
                if spec_setup is None:
                    spec_token_indx = torch.arange(spec_token_size, dtype=torch.int32, device=query_start_loc.device)
                else:
                    # Share only tree metadata; width and device distinguish
                    # builders that happen to see the same CPU request layout.
                    arange_key = (self.num_spec, query_start_loc.device)
                    ranges = spec_setup.setdefault("spec_token_aranges", {})
                    if arange_key not in ranges:
                        ranges[arange_key] = torch.arange(
                            num_spec_decodes * (self.num_spec + 1),
                            dtype=torch.int32,
                            device=query_start_loc.device,
                        )
                    spec_token_indx = ranges[arange_key][:spec_token_size]
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                if self.spec_tree_slots is None:
                    spec_state_indices_tensor = block_table_tensor[:num_spec_decodes, : self.num_spec + 1]
                else:
                    spec_state_indices_tensor = self._select_spec_state_indices(
                        block_table_tensor, spec_sequence_indices
                    )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                if spec_setup is None:
                    spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                        spec_sequence_masks_cpu, num_spec_decodes
                    )
                query_lens = query_start_loc[1:] - query_start_loc[:-1]
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=total_tokens,
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = self._select_spec_state_indices(
                    block_table_tensor,
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
            if num_prefills == 0 and num_decodes == 0 and (spec_setup is None or spec_setup["num_zero_len"] == 0):
                # Every row is a speculative row, so the spec selection is the
                # identity: the leading rows already are the spec rows. A view
                # keeps the current step's values (the accepted counts change
                # every step) without a dispatch.
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
            else:
                num_accepted_tokens = torch.index_select(
                    num_accepted_tokens,
                    0,
                    spec_sequence_indices,
                )
            if self.spec_tree_slots is not None:
                query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
                # Tokens each speculative request verifies; a row may be shorter
                # than the tree width, see _build_tree_init_state_indices.
                spec_row_widths = query_lens_cpu[spec_sequence_masks_cpu]
                if shared_gdn_tree is not None:
                    # A complete BS1 row is already token-major. Share logical
                    # columns/windows, but gather this group's state pages.
                    if (
                        self.spec_tree_slots != shared_gdn_tree.slots
                        or self.spec_conv_width != shared_gdn_tree.conv_width
                        or graph_request_count != 1
                        or num_spec_decodes != 1
                        or num_prefills != 0
                        or num_decodes != 0
                        or int(spec_row_widths[0]) != shared_gdn_tree.slots
                    ):
                        raise ValueError("shared GDN tree metadata requires one complete speculative tree row")
                    spec_init_state_indices = shared_gdn_tree.bind_init_state_indices(spec_state_indices_tensor)
                    spec_conv_window_rows = shared_gdn_tree.conv_window_rows
                    spec_conv_prefix_src = shared_gdn_tree.conv_prefix_src
                else:
                    spec_init_state_indices = self._build_tree_init_state_indices(
                        spec_state_indices_tensor,
                        num_accepted_tokens,
                        query_lens_cpu,
                        spec_sequence_masks_cpu,
                        tree_parents,
                        tree_num_nodes,
                        spec_sequence_indices,
                    )
                    spec_conv_window_rows, spec_conv_prefix_src = self._build_tree_conv_rows(
                        num_accepted_tokens,
                        tree_parents,
                        tree_num_nodes,
                        spec_sequence_indices,
                        prev_path_node_ids,
                        prev_num_sampled,
                        row_widths=spec_row_widths,
                    )
                    if spec_init_state_indices is not None and self.spec_conv_window_rows is not None:
                        # Same packing for the state pages: the recurrent op consumes
                        # them by global token offset, like the init pages and the conv
                        # windows. The conv path keeps reading the table per row, taking
                        # each request's running page at its token start
                        # (tree_causal_conv1d_triton).
                        spec_state_indices_tensor = _pack_tree_rows_to_tokens(
                            spec_state_indices_tensor,
                            spec_row_widths,
                            spec_state_indices_tensor.shape[1],
                        )
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
        checkpoint = None
        active_prefill_rows = None
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
            if self._ascend_gdn_checkpoints:
                assert prefill_state_indices is not None
                if spec_sequence_masks_cpu is None:
                    prefill_rows = list(range(num_decodes, m.num_reqs))
                else:
                    non_spec_rows = [i for i, is_spec in enumerate(spec_sequence_masks_cpu.tolist()) if not is_spec]
                    prefill_rows = non_spec_rows[num_decodes:]
                # Chunk indices use compact row numbers. Keep the planner and
                # recurrence aligned when padded requests have no tokens.
                live_rows = prefill_query_start_loc_cpu.diff() > 0
                if not live_rows.all():
                    active_rows_cpu = live_rows.nonzero(as_tuple=True)[0]
                    active_prefill_rows = active_rows_cpu.to(query_start_loc.device, non_blocking=True)
                    prefill_state_indices = prefill_state_indices.index_select(0, active_prefill_rows)
                    prefill_rows = [prefill_rows[row] for row in active_rows_cpu.tolist()]
                    prefill_query_start_loc_cpu = torch.cat(
                        (prefill_query_start_loc_cpu[:1], prefill_query_start_loc_cpu[1:][live_rows])
                    )
                    prefill_query_start_loc = prefill_query_start_loc_cpu.to(query_start_loc.device, non_blocking=True)
                checkpoint = self.checkpoint_builder.build(m, prefill_rows)
                if checkpoint is not None:
                    # Do not round at a boundary without a reserved cache page.
                    checkpoint.checkpoint_offsets.masked_fill_(checkpoint.state_indices == NULL_BLOCK_ID, 0)
            if self._USE_COMMON_KERNEL_METADATA:
                non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                    self,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                    build_device_metadata=self.build_device_chunk_metadata or self._ascend_gdn_checkpoints,
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
            if active_prefill_rows is not None:
                prefill_has_initial_state = prefill_has_initial_state.index_select(0, active_prefill_rows)
        else:
            has_initial_state = None

        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        attn_metadata = GDNAttentionMetadata(
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
            checkpoint=checkpoint,
        )
        if (
            self.use_full_cuda_graph
            and num_spec_decodes > 0
            and num_prefills == 0
            and num_decodes == 0
            and self._can_pad_spec_decode(graph_request_count, num_spec_decode_tokens)
        ):
            self._pad_spec_decode_metadata(attn_metadata, graph_request_count)
            if spec_init_state_indices is not None:
                assert self.spec_init_state_indices is not None
                self.spec_init_state_indices[:num_spec_decodes].copy_(spec_init_state_indices, non_blocking=True)
                spec_init_state_indices = self.spec_init_state_indices[:graph_request_count]
                spec_init_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)
            if spec_conv_window_rows is not None:
                assert spec_conv_prefix_src is not None
                assert self.spec_conv_window_rows is not None and self.spec_conv_prefix_src is not None
                self.spec_conv_window_rows[: spec_conv_window_rows.size(0)].copy_(
                    spec_conv_window_rows, non_blocking=True
                )
                self.spec_conv_prefix_src[: spec_conv_prefix_src.size(0)].copy_(spec_conv_prefix_src, non_blocking=True)
                spec_conv_window_rows = self.spec_conv_window_rows[: graph_request_count * self.spec_tree_slots]
                spec_conv_prefix_src = self.spec_conv_prefix_src[:graph_request_count]
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
        self._attach_spec_decode_metadata(
            attn_metadata,
            init_state_indices=spec_init_state_indices,
            conv_window_rows=spec_conv_window_rows,
            conv_prefix_src=spec_conv_prefix_src,
            shared_gdn_tree=shared_gdn_tree,
        )
        self._attach_non_spec_decode_metadata(attn_metadata, non_spec_conv1d_cache_indices)
        return self._finalize_metadata(attn_metadata)

    def _finalize_metadata(self, metadata: GDNAttentionMetadata, fused: bool = False) -> GDNAttentionMetadata:
        return metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        if self.spec_tree_slots is None:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        m = common_attn_metadata
        device = m.query_start_loc.device
        tree_parents = torch.zeros((m.num_reqs, self.spec_tree_slots - 1), dtype=torch.int32, device=device)
        tree_num_nodes = torch.zeros((m.num_reqs,), dtype=torch.int32, device=device)
        prev_path_node_ids = torch.zeros_like(tree_parents)
        prev_num_sampled = torch.zeros((m.num_reqs,), dtype=torch.int32, device=device)
        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        return self.build(
            0,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            prev_path_node_ids=prev_path_node_ids,
            prev_num_sampled=prev_num_sampled,
        )

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

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True


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
        self._full_tree_metadata: GDNAttentionMetadata | None = None
        self._full_tree_request_id: str | None = None
        self._using_full_tree_inputs = False
        self._tree_input_layouts = None
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

    def _can_reuse_full_tree_metadata(
        self,
        common: CommonAttentionMetadata,
        tree: "SharedGDNTreeMetadata",
        num_accepted_tokens: torch.Tensor | None,
        draft_widths_cpu: torch.Tensor | None,
        num_actual_reqs: int | None,
    ) -> bool:
        # The producer supplies shared topology only for a complete BS1 FULL
        # decode. Other users of build(), including capture, may call it directly.
        return (
            self.use_full_cuda_graph
            and self.use_spec_decode
            and self.vllm_config.parallel_config.pipeline_parallel_size == 1
            and (not self.vllm_config.scheduler_config.async_scheduling or tree.request_id is not None)
            and not getattr(self.speculative_config, "enable_adaptive_verification", False)
            and not getattr(common, "for_cudagraph_capture", False)
            and getattr(common, "pcp_context", None) is None
            and self.spec_tree_slots == tree.slots
            and self.spec_conv_width == tree.conv_width
            and num_actual_reqs == common.num_reqs == 1
            and common.num_actual_tokens == common.max_query_len == tree.slots
            and common.query_start_loc_cpu.shape == (2,)
            and tuple(common.query_start_loc_cpu.numpy()) == (0, tree.slots)
            and draft_widths_cpu is not None
            and draft_widths_cpu.device.type == "cpu"
            and draft_widths_cpu.shape == (1,)
            and int(draft_widths_cpu.numpy()[0]) == tree.slots - 1
            and num_accepted_tokens is not None
            and num_accepted_tokens.shape == (1,)
            and common.block_table_tensor.shape[0] == common.seq_lens.shape[0] == 1
        )

    def select_full_tree_inputs(self, active: bool, common_buffers: dict) -> None:
        """Bind stable destinations before capture or replay of the uniform BS1 descriptor."""
        if self._using_full_tree_inputs == active or self.spec_verify_plan_buffers is None:
            return
        if self._tree_input_layouts is None:
            original = (
                self.num_accepted_tokens,
                self.spec_conv_prefix_src,
                self.spec_conv_window_rows,
                self.spec_verify_plan_buffers,
            )
            if any(t is None for t in original) or self.spec_init_state_indices is None:
                return
            key = (
                self.spec_tree_slots,
                self.spec_conv_width,
                tuple((t.shape, t.stride(), t.device, t.dtype) for t in original[:3]),
            )
            if key not in common_buffers:
                common_buffers[key] = tuple(torch.empty_like(t) for t in original[:3])
            # One complete row already has the plan's exact packed token order.
            # Every group keeps its own two physical page tables.
            plan = replace(
                self.spec_verify_plan_buffers,
                state_indices=self.spec_state_indices_tensor[0],
                init_state_indices=self.spec_init_state_indices[0],
            )
            self._tree_input_layouts = original, (*common_buffers[key], plan)
        (
            self.num_accepted_tokens,
            self.spec_conv_prefix_src,
            self.spec_conv_window_rows,
            self.spec_verify_plan_buffers,
        ) = self._tree_input_layouts[int(active)]
        self._using_full_tree_inputs = active
        self._spec_verify_plan_views = None
        self._full_tree_metadata = None

    def try_refresh_tree_group(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tree: "SharedGDNTreeMetadata",
        accepted: torch.Tensor,
    ) -> GDNAttentionMetadata | None:
        """Refresh this group's warm descriptor for a prevalidated full row."""
        if (
            self._full_tree_metadata is None
            or not self._using_full_tree_inputs
            or self._full_tree_request_id != tree.request_id
            or self.spec_tree_slots != tree.slots
            or self.spec_conv_width != tree.conv_width
            or block_table.shape[0] != 1
            or seq_lens.shape[0] != 1
        ):
            return None
        return self._refresh_full_tree_metadata(self._full_tree_metadata, block_table, seq_lens, tree, accepted)

    def _refresh_full_tree_metadata(
        self,
        metadata: GDNAttentionMetadata,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        tree: "SharedGDNTreeMetadata",
        accepted: torch.Tensor,
    ) -> GDNAttentionMetadata:
        # All shape-only fields are still valid in the builder's capture
        # buffers. Refresh every data-dependent field from this group's pages.
        spec = metadata.spec_decode_metadata
        table = mamba_get_block_table_tensor(
            block_table,
            seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        pool = id(self.num_accepted_tokens)
        if not self._using_full_tree_inputs or pool not in tree._written_graph_inputs:
            metadata.num_accepted_tokens.copy_(accepted)
            spec.conv_window_rows.copy_(tree.conv_window_rows)
            spec.conv_prefix_src.copy_(tree.conv_prefix_src)
            if self._using_full_tree_inputs:
                tree._written_graph_inputs.add(pool)
        spec.spec_causal_conv1d.cache_indices = metadata.spec_state_indices_tensor
        if (
            self._using_full_tree_inputs
            and tree.page_batch is not None
            and tree.page_batch.add(tree.init_state_columns, table, self, metadata)
        ):
            return metadata
        pages = table[:1, : tree.slots]
        init_pages = tree.bind_init_state_indices(pages)
        metadata.spec_state_indices_tensor.copy_(pages)
        spec.init_state_indices.copy_(init_pages)
        if not self._using_full_tree_inputs:
            spec.verify_plan.state_indices.copy_(pages[0])
            spec.verify_plan.init_state_indices.copy_(init_pages[0])
        return self._finalize_metadata(metadata)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        shared_gdn_tree: "SharedGDNTreeMetadata | None" = None,
    ) -> GDNAttentionMetadata:
        self._ensure_conv_index_buffers()
        self._ensure_conv_output_buffer()
        complete = (
            common_prefix_len == 0
            and shared_gdn_tree is not None
            and self._can_reuse_full_tree_metadata(
                common_attn_metadata, shared_gdn_tree, num_accepted_tokens, num_decode_draft_tokens_cpu, num_actual_reqs
            )
        )
        # Worker dispatch enqueues the previous target, tree publication and
        # these dynamic writes in order on the same compute stream, including
        # when the scheduler has already queued the next 33-slot descriptor.
        if (
            complete
            and self._full_tree_metadata is not None
            and self._full_tree_request_id == shared_gdn_tree.request_id
        ):
            return self._refresh_full_tree_metadata(
                self._full_tree_metadata,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                shared_gdn_tree,
                num_accepted_tokens,
            )
        if self._full_tree_metadata is not None:
            # Other graph shapes can overwrite the stable Tree buffers.
            self._full_tree_metadata = None
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            prev_path_node_ids=prev_path_node_ids,
            prev_num_sampled=prev_num_sampled,
            shared_gdn_tree=shared_gdn_tree,
        )
        metadata.pcp_context = getattr(common_attn_metadata, "pcp_context", None)
        if complete and metadata.spec_decode_metadata is not None:
            spec = metadata.spec_decode_metadata
            plan = spec.verify_plan
            if (
                metadata.num_spec_decodes == 1
                and metadata.num_prefills == metadata.num_decodes == 0
                and metadata.spec_state_indices_tensor.shape == (1, shared_gdn_tree.slots)
                and spec.init_state_indices is not None
                and spec.conv_window_rows is not None
                and spec.conv_prefix_src is not None
                and plan is not None
                and plan.rows == 1
                and plan.slots == shared_gdn_tree.slots
            ):
                self._full_tree_metadata = metadata
                self._full_tree_request_id = shared_gdn_tree.request_id
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
