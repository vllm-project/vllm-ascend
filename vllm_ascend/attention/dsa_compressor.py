# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Shared DeepSeek-V4 Compressor metadata and execution.

The SP path has three stages: build a per-rank input/output plan, bind the
state-cache slots supplied by the separate state cache group, then execute the
local Compressor followed by fixed-shape KV/state synchronization.
"""

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.device.device_op import DeviceOperator

CompressorMetadataOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_COMPRESSOR_METADATA_CACHE_KEY = "dsv4_compressor_metadata_cache"


@dataclass
class CompressorSPMetadata:
    """Device tensors consumed by one sequence-parallel Compressor owner.

    Main and LI own separate instances. State fields are attached later from
    the matching state cache group so cache effects remain identical to non-SP.
    """

    ############################################################
    # Local Compressor Input
    ############################################################

    # Per-request offsets into the packed local compressor input.
    # Shape: [num_reqs + 1].
    packed_query_start_loc: torch.Tensor
    # Absolute sequence position of the first token in each packed request
    # segment. Empty local segments use 0. Shape: [num_reqs].
    packed_start_pos: torch.Tensor
    # Preallocated right-aligned local suffix sent by the suffix all-gather.
    # Shape: [window_size, hidden_dim].
    suffix_buffer: torch.Tensor
    # Indices that form request-major compressor input from the concatenated
    # all-rank suffixes and local hidden states. Shape: [input_count].
    pack_indices: torch.Tensor
    # Start of the valid suffix rows in this rank's local hidden states.
    local_suffix_start: int
    # Number of valid rows right-aligned in suffix_buffer. The remaining leading
    # rows are zero. Range: [0, window_size].
    local_suffix_valid_len: int
    # Number of packed input rows consumed by this rank's compressor.
    input_count: int
    # Maximum packed rows: one local shard plus window_size - 1 history rows.
    input_capacity: int
    # Local compressor output capacity, including per-request padding rows.
    num_compressed_tokens: int

    ############################################################
    # Compressed KV Aggregation
    ############################################################

    # Fixed output rows contributed by each rank to the compressed-KV gather.
    gathered_compressed_tokens: int
    # Global compressor output capacity used by the global slot mapping.
    global_num_compressed_tokens: int
    # Indices that reorder rank-major gathered outputs into global request order.
    # Shape: [global_num_compressed_tokens].
    gathered_kv_reorder_indices: torch.Tensor

    # Preallocated local send-buffer view for compressed-KV all-gather.
    # Shape: [gathered_compressed_tokens, output_dim].
    compressed_kv_send_buffer: torch.Tensor
    # Preallocated rank-major result-buffer view for compressed-KV all-gather.
    # Shape: [tp_size * gathered_compressed_tokens, output_dim].
    gathered_compressed_kv_buffer: torch.Tensor

    ############################################################
    # State Cache Synchronization
    ############################################################

    # Fixed number of original packed token rows assigned to each TP rank.
    tokens_per_rank: int
    # Global packed token rows after padding to an equal TP partition.
    num_tokens_pad: int
    # Preallocated local state rows contributed to the state all-gather.
    # Shape: [tokens_per_rank, 2 * coff * output_dim].
    state_send_buffer: torch.Tensor
    # Preallocated rank-major result buffer for the state all-gather.
    # Shape: [num_tokens_pad, 2 * coff * output_dim].
    gathered_state_buffer: torch.Tensor
    # Physical block indices for this rank's original token shard.
    # Invalid/null rows address the safe sink block 0. Shape: [tokens_per_rank].
    local_block_indices: torch.Tensor
    # Physical offsets paired with local_block_indices. Invalid/null rows use 0.
    # Shape: [tokens_per_rank].
    local_offset_indices: torch.Tensor
    # True for the non-SP state write-set. Shape: [num_tokens_pad].
    valid_slots: torch.Tensor
    # Global state slots with invalid/null rows mapped to the safe sink [0, 0].
    # Shape: [num_tokens_pad, 2].
    scatter_slot_mapping: torch.Tensor


class CompressorSPMetadataBuilder:
    """Build one Compressor output group's reusable SP plan and workspaces.

    Scheduler metadata is converted to CPU lists before entering this builder.
    Small index tensors use ``CpuGpuBuffer`` and communication tensors are
    allocated once. The separately built state cache group later supplies its
    physical slots through :meth:`bind_state_slots`.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        tp_size: int,
        compress_ratio: int,
        coff: int,
        hidden_dim: int,
        output_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if compress_ratio <= 1:
            raise ValueError(f"invalid compressor ratio: {compress_ratio}")
        if coff not in (1, 2):
            raise ValueError(f"invalid compressor coff: {coff}")
        if hidden_dim <= 0:
            raise ValueError(f"invalid compressor hidden dim: {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"invalid compressor output dim: {output_dim}")

        # Compressor configuration.
        self._tp_size = tp_size
        self._compress_ratio = compress_ratio
        self._coff = coff
        self._window_size = compress_ratio * coff

        # A rank owns at most one padded TP shard and may prepend at most one
        # full Compressor window minus the first owned token.
        max_pack_indices = (max_num_batched_tokens + tp_size - 1) // tp_size + self._window_size - 1
        # The kernel reserves at most one extra output row per request for
        # partial compression groups.
        max_gathered_compressed_tokens = min(
            max_pack_indices,
            max_pack_indices // compress_ratio + max_num_seqs,
        )
        max_reorder_indices = min(
            max_num_batched_tokens,
            max_num_batched_tokens // compress_ratio + max_num_seqs,
        )

        # Local compressor input metadata.
        self._packed_query_start_loc_buffer = CpuGpuBuffer(max_num_seqs + 1, dtype=torch.int32, device=device)
        self._packed_start_pos_buffer = CpuGpuBuffer(max_num_seqs, dtype=torch.int32, device=device)
        self._suffix_buffer = torch.empty((self._window_size, hidden_dim), dtype=dtype, device=device)
        self._pack_indices_buffer = CpuGpuBuffer(max_pack_indices, dtype=torch.int64, device=device)

        # Compressed KV aggregation metadata.
        self._gathered_kv_reorder_indices_buffer = CpuGpuBuffer(max_reorder_indices, dtype=torch.int64, device=device)
        self._compressed_kv_send_buffer = torch.empty(
            (max_gathered_compressed_tokens, output_dim), dtype=dtype, device=device
        )
        self._gathered_compressed_kv_buffer = torch.empty(
            (tp_size * max_gathered_compressed_tokens, output_dim), dtype=dtype, device=device
        )

        # State cache synchronization metadata.
        max_tokens_per_rank = (max_num_batched_tokens + tp_size - 1) // tp_size
        max_num_tokens_pad = tp_size * max_tokens_per_rank
        # State stores KV state and score state, each with coff output groups.
        state_dim = 2 * coff * output_dim
        self._state_send_buffer = torch.empty((max_tokens_per_rank, state_dim), dtype=torch.float32, device=device)
        self._gathered_state_buffer = torch.empty((max_num_tokens_pad, state_dim), dtype=torch.float32, device=device)
        self._local_block_indices_buffer = torch.empty(max_tokens_per_rank, dtype=torch.int64, device=device)
        self._local_offset_indices_buffer = torch.empty(max_tokens_per_rank, dtype=torch.int64, device=device)
        self._valid_slots_buffer = torch.empty(max_num_tokens_pad, dtype=torch.bool, device=device)
        self._scatter_slot_mapping_buffer = torch.empty((max_num_tokens_pad, 2), dtype=torch.int32, device=device)

    @staticmethod
    def _copy_to_gpu(
        buffer: CpuGpuBuffer,
        values: list[int],
    ) -> torch.Tensor:
        """Copy one variable-length CPU list into a reusable device view."""
        count = len(values)
        if count > buffer.cpu.shape[0]:
            raise ValueError(f"Compressor metadata length {count} exceeds buffer capacity " f"{buffer.cpu.shape[0]}")
        buffer.np[:count] = values
        return buffer.copy_to_gpu(count)

    ############################################################
    # State Slot Binding
    ############################################################

    @staticmethod
    def bind_state_slots(
        metadata: CompressorSPMetadata,
        state_slot_mapping: torch.Tensor,
        local_token_start: int,
        tokens_per_rank: int,
        num_tokens_pad: int,
    ) -> CompressorSPMetadata:
        """Attach the non-SP state write-set to an existing Compressor plan.

        The state cache and compressed output are separate vLLM cache groups,
        so their builders may run in either order. This method joins them after
        both describe the same padded TP token partition.
        """
        if state_slot_mapping.ndim != 2 or state_slot_mapping.shape[1] != 2:
            raise ValueError("Compressor SP state slot mapping must have shape [num_tokens, 2]")
        if state_slot_mapping.shape[0] > num_tokens_pad:
            raise ValueError(f"State slot rows {state_slot_mapping.shape[0]} exceed planned " f"rows {num_tokens_pad}")
        if tokens_per_rank != metadata.tokens_per_rank or num_tokens_pad != metadata.num_tokens_pad:
            raise ValueError("Compressor and state-cache SP partitions do not match")
        if local_token_start < 0 or tokens_per_rank < 0 or local_token_start + tokens_per_rank > num_tokens_pad:
            raise ValueError("Compressor SP local state partition is outside the global slots")

        global_slots = metadata.scatter_slot_mapping
        num_state_slots = state_slot_mapping.shape[0]
        global_slots[:num_state_slots].copy_(state_slot_mapping)
        global_slots[num_state_slots:].zero_()

        # Physical block 0 is the null block. Preserve the exact non-SP write
        # set and map null/padding rows to the safe sink slot [0, 0].
        torch.gt(global_slots[:, 0], 0, out=metadata.valid_slots)
        invalid_slots = ~metadata.valid_slots
        global_slots[:, 0].masked_fill_(invalid_slots, 0)
        global_slots[:, 1].masked_fill_(invalid_slots, 0)

        # Each rank reads only its original-token shard before the state
        # all-gather; all ranks later scatter the complete global write-set.
        local_slots = global_slots.narrow(
            0,
            local_token_start,
            tokens_per_rank,
        )
        metadata.local_block_indices.copy_(local_slots[:, 0])
        metadata.local_offset_indices.copy_(local_slots[:, 1])
        return metadata

    ############################################################
    # Request Plan Construction
    ############################################################

    def build_sp(
        self,
        query_start_loc: list[int],
        seq_lens: list[int],
        num_actual_tokens: int,
        num_input_tokens: int,
        tp_rank: int,
        num_reqs_actual: int,
    ) -> CompressorSPMetadata:
        """Build fixed-shape compressor metadata from flattened requests.

        Pack indices address ``[gathered_rank_suffixes, local_hidden]``.
        Each rank contributes a right-aligned compressor-window suffix. The
        packed input contains the local shard plus the history needed for group
        alignment and, when coff is 2, the preceding overlap group.
        """
        tp_size = self._tp_size
        compress_ratio = self._compress_ratio
        window_size = self._window_size
        overlap_tokens = compress_ratio * (self._coff - 1)
        num_reqs = len(query_start_loc) - 1
        if len(seq_lens) != num_reqs:
            raise ValueError("seq_lens and query_start_loc must describe the same requests")
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP topology: size={tp_size}, rank={tp_rank}")

        # Split the flattened scheduler batch into equal contiguous TP shards.
        # Only the final shard may contain padding beyond num_actual_tokens.
        num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
        tokens_per_rank = num_tokens_pad // tp_size
        local_start = tp_rank * tokens_per_rank
        local_end = local_start + tokens_per_rank
        actual_end = min(num_actual_tokens, query_start_loc[-1])
        local_valid_len = max(0, min(tokens_per_rank, actual_end - local_start))
        input_capacity = tokens_per_rank + window_size - 1
        local_suffix_valid_len = min(local_valid_len, window_size)
        local_suffix_start = local_valid_len - local_suffix_valid_len

        pack_indices: list[int] = []
        packed_query_start_loc = [0]
        packed_start_pos: list[int] = []
        local_hidden_base = tp_size * window_size

        def suffix_source_index(flat_position: int) -> int:
            """Map a global token row to its rank's right-aligned suffix."""
            src_rank = flat_position // tokens_per_rank
            src_start = src_rank * tokens_per_rank
            src_valid_len = max(0, min(tokens_per_rank, actual_end - src_start))
            src_offset = flat_position - src_start
            suffix_offset = window_size - src_valid_len + src_offset
            if not 0 <= src_rank < tp_size or not 0 <= suffix_offset < window_size:
                raise ValueError(f"Compressor boundary position {flat_position} is outside gathered suffixes")
            return src_rank * window_size + suffix_offset

        # Build this rank's request-major Compressor input. Boundary history is
        # addressed in gathered suffixes; owned tokens come from local hidden.
        for req_idx, (query_start, query_end, seq_len) in enumerate(
            zip(query_start_loc[:-1], query_start_loc[1:], seq_lens)
        ):
            query_len = query_end - query_start
            prefix_len = seq_len - query_len
            local_query_start = max(query_start, local_start)
            local_query_end = min(query_end, local_end, actual_end)
            local_query_len = (
                max(0, local_query_end - local_query_start) if req_idx < num_reqs_actual else 0
            )

            if local_query_len == 0:
                packed_start_pos.append(0)
                packed_query_start_loc.append(len(pack_indices))
                continue

            sequence_start_pos = prefix_len + local_query_start - query_start
            aligned_group_start = sequence_start_pos // compress_ratio * compress_ratio
            compressor_start_pos = max(prefix_len, aligned_group_start - overlap_tokens)
            history_start = query_start + compressor_start_pos - prefix_len

            pack_indices.extend(suffix_source_index(pos) for pos in range(history_start, local_query_start))
            pack_indices.extend(
                local_hidden_base + pos - local_start for pos in range(local_query_start, local_query_end)
            )
            packed_start_pos.append(compressor_start_pos)
            packed_query_start_loc.append(len(pack_indices))

        input_count = len(pack_indices)
        if input_count > input_capacity:
            raise ValueError(f"Compressor packed input {input_count} exceeds fixed capacity {input_capacity}")
        num_compressed_tokens = min(
            input_count,
            input_count // compress_ratio + num_reqs,
        )

        # Determine which raw output row owns each global compressed group. C4
        # overlap can replay a group on two ranks; ownership belongs to the rank
        # containing that group's final token.
        global_output_offsets: list[int] = []
        global_output_rows = 0
        for req_idx, (query_start, query_end, seq_len) in enumerate(
            zip(query_start_loc[:-1], query_start_loc[1:], seq_lens)
        ):
            global_output_offsets.append(global_output_rows)
            if req_idx < num_reqs_actual:
                prefix_len = seq_len - (query_end - query_start)
                global_output_rows += seq_len // compress_ratio - prefix_len // compress_ratio

        gathered_kv_reorder_indices = [-1] * global_output_rows
        rank_output_capacities: list[int] = []
        rank_owned_output_rows: list[list[tuple[int, int]]] = []
        for rank in range(tp_size):
            rank_start = rank * tokens_per_rank
            rank_end = rank_start + tokens_per_rank
            rank_input_count = 0
            raw_output_row = 0
            owned_output_rows: list[tuple[int, int]] = []
            for req_idx, (query_start, query_end, seq_len) in enumerate(
                zip(query_start_loc[:-1], query_start_loc[1:], seq_lens)
            ):
                if req_idx >= num_reqs_actual:
                    continue
                query_len = query_end - query_start
                prefix_len = seq_len - query_len
                rank_query_start = max(query_start, rank_start)
                rank_query_end = min(query_end, rank_end, actual_end)
                if rank_query_end <= rank_query_start:
                    continue
                sequence_start_pos = prefix_len + rank_query_start - query_start
                sequence_end_pos = prefix_len + rank_query_end - query_start
                aligned_group_start = sequence_start_pos // compress_ratio * compress_ratio
                compressor_start_pos = max(prefix_len, aligned_group_start - overlap_tokens)
                history_start = query_start + compressor_start_pos - prefix_len
                packed_len = rank_query_end - history_start
                output_rows = (
                    (compressor_start_pos + packed_len) // compress_ratio
                    - compressor_start_pos // compress_ratio
                )
                rank_input_count += packed_len
                for output_offset in range(output_rows):
                    group_idx = compressor_start_pos // compress_ratio + output_offset
                    group_end_pos = (group_idx + 1) * compress_ratio - 1
                    if not sequence_start_pos <= group_end_pos < sequence_end_pos:
                        raw_output_row += 1
                        continue
                    global_row = global_output_offsets[req_idx] + group_idx - prefix_len // compress_ratio
                    owned_output_rows.append((global_row, raw_output_row))
                    raw_output_row += 1
            rank_output_capacities.append(
                min(
                    rank_input_count,
                    rank_input_count // compress_ratio + num_reqs,
                )
            )
            rank_owned_output_rows.append(owned_output_rows)

        # Fixed-shape all-gather stores [rank, raw_output_row]. Convert each
        # global owner into one flat source index for the final index_select.
        gathered_compressed_tokens = max(rank_output_capacities)
        for rank, owned_output_rows in enumerate(rank_owned_output_rows):
            for global_row, local_output_row in owned_output_rows:
                if gathered_kv_reorder_indices[global_row] != -1:
                    raise ValueError(f"Compressor row {global_row} is produced by multiple TP ranks")
                gathered_kv_reorder_indices[global_row] = rank * gathered_compressed_tokens + local_output_row
        if any(source < 0 for source in gathered_kv_reorder_indices):
            raise ValueError("Compressor SP plan does not cover every global compressed row")

        global_num_compressed_tokens = min(
            num_actual_tokens,
            num_actual_tokens // compress_ratio + num_reqs,
        )
        if global_output_rows > global_num_compressed_tokens:
            raise ValueError(
                f"Compressor valid rows {global_output_rows} exceed global capacity "
                f"{global_num_compressed_tokens}"
            )
        # Capacity-only rows map to source 0 and later target null slots. No
        # per-rank padding row is selected by this reorder plan.
        gathered_kv_reorder_indices.extend([0] * (global_num_compressed_tokens - global_output_rows))
        return CompressorSPMetadata(
            # Local compressor input metadata.
            packed_query_start_loc=self._copy_to_gpu(self._packed_query_start_loc_buffer, packed_query_start_loc),
            packed_start_pos=self._copy_to_gpu(self._packed_start_pos_buffer, packed_start_pos),
            suffix_buffer=self._suffix_buffer,
            pack_indices=self._copy_to_gpu(self._pack_indices_buffer, pack_indices),
            local_suffix_start=local_suffix_start,
            local_suffix_valid_len=local_suffix_valid_len,
            input_count=input_count,
            input_capacity=input_capacity,
            num_compressed_tokens=num_compressed_tokens,
            # Compressed KV aggregation metadata.
            gathered_compressed_tokens=gathered_compressed_tokens,
            global_num_compressed_tokens=global_num_compressed_tokens,
            gathered_kv_reorder_indices=self._copy_to_gpu(
                self._gathered_kv_reorder_indices_buffer,
                gathered_kv_reorder_indices,
            ),
            compressed_kv_send_buffer=self._compressed_kv_send_buffer.narrow(0, 0, gathered_compressed_tokens),
            gathered_compressed_kv_buffer=self._gathered_compressed_kv_buffer.narrow(
                0, 0, tp_size * gathered_compressed_tokens
            ),
            # State cache synchronization metadata.
            tokens_per_rank=tokens_per_rank,
            num_tokens_pad=num_tokens_pad,
            state_send_buffer=self._state_send_buffer.narrow(0, 0, tokens_per_rank),
            gathered_state_buffer=self._gathered_state_buffer.narrow(0, 0, num_tokens_pad),
            local_block_indices=self._local_block_indices_buffer.narrow(0, 0, tokens_per_rank),
            local_offset_indices=self._local_offset_indices_buffer.narrow(0, 0, tokens_per_rank),
            valid_slots=self._valid_slots_buffer.narrow(0, 0, num_tokens_pad),
            scatter_slot_mapping=self._scatter_slot_mapping_buffer.narrow(0, 0, num_tokens_pad),
        )


def reset_compressor_metadata_cache() -> None:
    """Release metadata outputs before a composite forward changes substeps."""
    get_forward_context().additional_kwargs.pop(_COMPRESSOR_METADATA_CACHE_KEY, None)


def get_or_compute_compressor_metadata(
    metadata: Any,
    compress_ratio: int,
) -> CompressorMetadataOutput:
    """Return kernel RoPE/slot metadata once per cache group and substep.

    Both the non-SP path and the SP global write-back need the same metadata
    operator result. Caching it in ``ForwardContext`` avoids launching that
    operator twice while keeping prefill/decode phases isolated.
    """
    forward_context = get_forward_context()
    cache: dict[tuple[str, type], CompressorMetadataOutput] = forward_context.additional_kwargs.setdefault(
        _COMPRESSOR_METADATA_CACHE_KEY,
        {},
    )
    cache_group_key = metadata.cache_group_key
    if not cache_group_key:
        raise ValueError("DSV4 compressor metadata requires a cache-group key")
    # The pre-refactor v0.26 DSA path invokes prefill and decode separately
    # within one mixed-batch forward. They share a cache group but require
    # different compressor metadata, so keep the metadata phases isolated.
    cache_key = (cache_group_key, type(metadata))
    cached_metadata = cache.get(cache_key)
    if cached_metadata is not None:
        return cached_metadata

    assert metadata.full_compress_cos is not None
    assert metadata.full_compress_sin is not None
    assert metadata.num_compressed_tokens is not None
    assert metadata.start_pos is not None
    assert metadata.num_reqs_actual is not None
    full_compress_cos = metadata.full_compress_cos.view(
        metadata.full_compress_cos.shape[0],
        metadata.full_compress_cos.shape[-1],
    )
    full_compress_sin = metadata.full_compress_sin.view(
        metadata.full_compress_sin.shape[0],
        metadata.full_compress_sin.shape[-1],
    )
    computed_metadata = torch.ops._C_ascend.compressor_metadata(
        full_compress_cos,
        full_compress_sin,
        metadata.query_start_loc,
        metadata.start_pos,
        metadata.block_table,
        metadata.block_size,
        DeviceOperator.get_dsa_compressor_slot_mapping_format(),
        compress_ratio,
        metadata.num_compressed_tokens,
        metadata.num_reqs_actual,
    )
    cache[cache_key] = computed_metadata
    return computed_metadata


def hadamard_transform_ref(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply a padded Hadamard transform and restore the original shape."""
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(
    x: torch.Tensor,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    """Apply the normalized Hadamard rotation required by LI caches."""
    hidden_size = x.size(-1)
    return hadamard_transform_ref(
        x,
        hadamard=hadamard,
        scale=hidden_size**-0.5,
    )


class CompressorExecutor:
    """Execute one main or LI Compressor and preserve non-SP cache semantics.

    Attention prepares either the full non-SP input or the SP packed input,
    then :meth:`run` selects the matching execution path. The base class writes
    main compressed KV; ``IndexerCompressorExecutor`` overrides only the LI
    output epilogue.
    """

    def __init__(
        self,
        compressor: torch.nn.Module,
        rope_head_dim: int | None,
        tp_group: Any,
    ) -> None:
        self.compressor = compressor
        self.rope_head_dim = rope_head_dim
        self.tp_group = tp_group

    @property
    def compress_ratio(self) -> int:
        return self.compressor.compress_ratio

    ############################################################
    # Shared Kernel and Cache Operations
    ############################################################

    def _run_kernel(
        self,
        compressor_input: torch.Tensor,
        state_cache: torch.Tensor,
        *,
        metadata: Any,
        state_block_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the common Compressor kernel without choosing cache epilogue."""
        compress_cos, compress_sin, slot_mapping = get_or_compute_compressor_metadata(
            metadata,
            self.compress_ratio,
        )
        compressed_kv = torch.ops._C_ascend.compressor(
            compressor_input,
            self.compressor.wkv.weight,
            self.compressor.wgate.weight,
            state_cache.squeeze(-2),
            self.compressor.ape,
            self.compressor.norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=self.compressor.coff,
            norm_eps=self.compressor.norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        return compressed_kv, slot_mapping

    def _write_cache(
        self,
        compressed_kv: torch.Tensor,
        slot_mapping: torch.Tensor,
        output_cache: Any,
        *,
        hadamard: torch.Tensor | None = None,
    ) -> None:
        """Write globally ordered rows; subclasses may replace the epilogue."""
        if compressed_kv.shape[0] == 0:
            return
        DeviceOperator.dsa_kv_compress_scatter(
            output_cache,
            compressed_kv,
            slot_mapping,
        )

    ############################################################
    # Execution Dispatch
    ############################################################

    def run(
        self,
        compressor_input: torch.Tensor,
        state_cache: torch.Tensor,
        output_cache: Any,
        *,
        metadata: Any,
        state_block_table: torch.Tensor,
        sp_metadata: CompressorSPMetadata | None = None,
        hadamard: torch.Tensor | None = None,
    ) -> None:
        """Dispatch using the SP metadata that also defines cache side effects."""
        if sp_metadata is None:
            self._run_non_sp(
                compressor_input,
                state_cache,
                output_cache,
                metadata=metadata,
                state_block_table=state_block_table,
                hadamard=hadamard,
            )
        else:
            self._run_sp(
                compressor_input,
                state_cache,
                output_cache,
                metadata=metadata,
                state_block_table=state_block_table,
                sp_metadata=sp_metadata,
                hadamard=hadamard,
            )

    ############################################################
    # Non-SP Execution
    ############################################################

    def prepare_non_sp_input(
        self,
        hidden_states_local: torch.Tensor,
        num_actual_tokens: int,
        need_gather_q_kv: bool,
    ) -> torch.Tensor:
        """Restore the full scheduler batch expected by the non-SP kernel."""
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states_local,
            need_gather_q_kv,
        )
        return hidden_states[:num_actual_tokens]

    def _run_non_sp(
        self,
        compressor_input: torch.Tensor,
        state_cache: torch.Tensor,
        output_cache: Any,
        *,
        metadata: Any,
        state_block_table: torch.Tensor,
        hadamard: torch.Tensor | None = None,
    ) -> None:
        compressed_kv, slot_mapping = self._run_kernel(
            compressor_input,
            state_cache,
            metadata=metadata,
            state_block_table=state_block_table,
            cu_seqlens=metadata.query_start_loc,
            start_pos=metadata.start_pos,
        )
        self._write_cache(
            compressed_kv,
            slot_mapping,
            output_cache,
            hadamard=hadamard,
        )

    ############################################################
    # SP Execution
    ############################################################

    def prepare_sp_input(
        self,
        hidden_states_local: torch.Tensor,
        sp_metadata: CompressorSPMetadata,
    ) -> torch.Tensor:
        """Gather boundary history and form this rank's request-major input."""
        local_suffix = sp_metadata.suffix_buffer
        suffix_size = local_suffix.shape[0]
        valid_len = sp_metadata.local_suffix_valid_len
        left_pad = suffix_size - valid_len
        local_suffix.narrow(0, 0, left_pad).zero_()
        local_suffix.narrow(0, left_pad, valid_len).copy_(
            hidden_states_local.narrow(
                0,
                sp_metadata.local_suffix_start,
                valid_len,
            )
        )

        # The fixed window keeps all ranks on the fast equal-shape all-gather.
        # Every rank must participate, even when its valid suffix is empty.
        gathered_suffixes = torch.empty(
            (
                self.tp_group.world_size * suffix_size,
                hidden_states_local.shape[-1],
            ),
            dtype=hidden_states_local.dtype,
            device=hidden_states_local.device,
        )
        dist.all_gather_into_tensor(
            gathered_suffixes,
            local_suffix.contiguous(),
            group=self.tp_group.device_group,
            async_op=False,
        )

        # pack_indices address this exact concatenation: all-rank boundary
        # suffixes first, followed by the complete local hidden-state shard.
        pack_source = torch.cat([gathered_suffixes, hidden_states_local], dim=0)
        return pack_source.index_select(0, sp_metadata.pack_indices).contiguous()

    def _gather_sp_output(
        self,
        compressed_kv: torch.Tensor,
        sp_metadata: CompressorSPMetadata,
    ) -> torch.Tensor:
        """All-gather fixed-capacity raw rows before global owner reorder."""
        target_rows = sp_metadata.gathered_compressed_tokens
        pad_rows = target_rows - compressed_kv.shape[0]
        if pad_rows < 0:
            raise ValueError(f"Compressed rows {compressed_kv.shape[0]} exceed gather capacity " f"{target_rows}")

        send_buffer = sp_metadata.compressed_kv_send_buffer
        # The reorder plan never selects unused send-buffer rows, so only the
        # produced prefix must be refreshed on each invocation.
        send_buffer[: compressed_kv.shape[0]].copy_(compressed_kv)

        gathered_kv = sp_metadata.gathered_compressed_kv_buffer
        dist.all_gather_into_tensor(
            gathered_kv,
            send_buffer,
            group=self.tp_group.device_group,
            async_op=False,
        )
        return gathered_kv

    def _sync_sp_state(
        self,
        state_cache: torch.Tensor,
        sp_metadata: CompressorSPMetadata,
    ) -> None:
        """Replicate the complete non-SP state write-set on every TP rank.

        Synchronizing only request tails would make internal prefix-cache
        checkpoints invalid on ranks that did not compute those tokens.
        """
        if sp_metadata.local_block_indices.shape[0] != sp_metadata.tokens_per_rank:
            raise ValueError("Compressor SP local state slots do not match the token partition")
        if sp_metadata.scatter_slot_mapping.shape[0] != sp_metadata.num_tokens_pad:
            raise ValueError("Compressor SP global state slots do not match the padded tokens")

        state_view = state_cache.squeeze(-2)
        sp_metadata.state_send_buffer.copy_(
            state_view[
                sp_metadata.local_block_indices,
                sp_metadata.local_offset_indices,
            ]
        )

        gathered_state = sp_metadata.gathered_state_buffer
        dist.all_gather_into_tensor(
            gathered_state,
            sp_metadata.state_send_buffer,
            group=self.tp_group.device_group,
            async_op=False,
        )

        # Block 0 is the null block: the Compressor kernel does not update it.
        # Map every null/padding row to one safe sink slot and write back that
        # rank's original sink value so fixed-shape scatter has no side effect.
        sink_state = state_view[0, 0]
        torch.where(
            sp_metadata.valid_slots[:, None],
            gathered_state,
            sink_state,
            out=gathered_state,
        )
        DeviceOperator.dsa_kv_compress_scatter(
            state_view,
            gathered_state,
            sp_metadata.scatter_slot_mapping,
        )

    def _run_sp(
        self,
        compressor_input: torch.Tensor,
        state_cache: torch.Tensor,
        output_cache: Any,
        *,
        metadata: Any,
        state_block_table: torch.Tensor,
        sp_metadata: CompressorSPMetadata,
        hadamard: torch.Tensor | None = None,
    ) -> None:
        """Run local compute, aggregate owned KV rows, then replicate state."""
        if sp_metadata.input_count == 0:
            # Empty ranks still enter both fixed-shape collectives below.
            compressed_kv = compressor_input.new_empty((0, self.compressor.norm.weight.shape[0]))
        else:
            # The local packed input has different request offsets and output
            # capacity from the global scheduler batch. Give it a distinct
            # cache key so local and global metadata cannot alias.
            local_metadata = replace(
                metadata,
                query_start_loc=sp_metadata.packed_query_start_loc,
                start_pos=sp_metadata.packed_start_pos,
                num_compressed_tokens=sp_metadata.num_compressed_tokens,
                cache_group_key=(f"{metadata.cache_group_key}:c{self.compress_ratio}_sp"),
            )
            compressed_kv, _ = self._run_kernel(
                compressor_input,
                state_cache,
                metadata=local_metadata,
                state_block_table=state_block_table,
                cu_seqlens=sp_metadata.packed_query_start_loc,
                start_pos=sp_metadata.packed_start_pos,
            )

        gathered_kv = self._gather_sp_output(compressed_kv, sp_metadata)
        # C4 overlap may produce the same group on adjacent ranks. The builder
        # chooses the owner row whose final token lies in that rank's shard.
        reordered_kv = gathered_kv.index_select(
            0,
            sp_metadata.gathered_kv_reorder_indices,
        )
        _, _, global_slot_mapping = get_or_compute_compressor_metadata(
            metadata,
            self.compress_ratio,
        )
        if global_slot_mapping.shape[0] != sp_metadata.global_num_compressed_tokens:
            raise ValueError(
                f"Global compressor slot rows {global_slot_mapping.shape[0]} do not "
                f"match planned rows {sp_metadata.global_num_compressed_tokens}"
            )
        self._write_cache(
            reordered_kv,
            global_slot_mapping,
            output_cache,
            hadamard=hadamard,
        )
        self._sync_sp_state(
            state_cache,
            sp_metadata,
        )


class IndexerCompressorExecutor(CompressorExecutor):
    """Reuse Compressor execution and replace only the LI cache epilogue."""

    def _write_cache(
        self,
        compressed_kv: torch.Tensor,
        slot_mapping: torch.Tensor,
        output_cache: Any,
        *,
        hadamard: torch.Tensor | None = None,
    ) -> None:
        if compressed_kv.shape[0] == 0:
            return
        if self.compressor.rotate:
            assert hadamard is not None
            compressed_kv = rotate_activation(compressed_kv, hadamard)

        # LI stores quantized K, its scale, and the unquantized rotated value in
        # three coordinated caches instead of the main BF16 compressed cache.
        indexer_k_cache, indexer_scale_cache, indexer_full_cache = output_cache
        _, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
            compressed_kv,
            indexer_k_cache,
            indexer_full_cache,
            slot_mapping,
        )
        if kv_scale is not None:
            DeviceOperator.dsa_indexer_scatter_scale_part3(
                kv_scale,
                indexer_scale_cache,
                slot_mapping,
            )
