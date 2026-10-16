# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE request-phase and replay-chunk metadata for the Ascend runtime.

Keep the upstream metadata contract and graph buffers while distinguishing
normal multi-token decode chunks from prefill and speculative rollback.
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadata as UpstreamPleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder as UpstreamPleShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, mamba_get_block_table_tensor


@dataclass
class PleShortConvChunkMetadata:
    """A normal multi-token decode group, with no speculative rollback."""

    token_indices: torch.Tensor
    query_start_loc: torch.Tensor
    state_indices_tensor: torch.Tensor
    has_initial_states: torch.Tensor
    num_reqs: int
    num_tokens: int
    max_query_len: int


@dataclass
class PleShortConvAttentionMetadata(UpstreamPleShortConvAttentionMetadata):
    normal_chunk: PleShortConvChunkMetadata | None = None
    remaining_metadata: "PleShortConvAttentionMetadata | None" = None
    remaining_token_indices: torch.Tensor | None = None


def _ple_request_phase_masks(
    is_prefilling: torch.Tensor | None,
    num_reqs_actual: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return authoritative decode/prefill masks for real requests only."""
    if is_prefilling is None:
        raise ValueError("PLE requires authoritative per-request phase metadata")
    if not 0 <= num_reqs_actual <= is_prefilling.numel():
        raise ValueError(
            "PLE actual request count must fit the phase metadata: "
            f"actual={num_reqs_actual}, phase_rows={is_prefilling.numel()}"
        )

    prefill_mask = is_prefilling[:num_reqs_actual].to(dtype=torch.bool)
    decode_mask = ~prefill_mask
    if torch.any(decode_mask & prefill_mask).item():
        raise AssertionError("PLE decode/prefill phase masks must be disjoint")
    if int(decode_mask.sum().item()) + int(prefill_mask.sum().item()) != num_reqs_actual:
        raise AssertionError("PLE phase masks must cover every real request")
    return decode_mask, prefill_mask


def _ple_decode_execution_masks(
    decode_phase_mask: torch.Tensor,
    query_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split known decode rows into regular and variable-length execution."""
    if decode_phase_mask.shape != query_lens.shape:
        raise ValueError(
            "PLE decode phase and query lengths must have the same shape: "
            f"phase={tuple(decode_phase_mask.shape)}, "
            f"query_lens={tuple(query_lens.shape)}"
        )
    variable_decode_mask = decode_phase_mask & (query_lens > 1)
    regular_decode_mask = decode_phase_mask & ~variable_decode_mask
    return regular_decode_mask, variable_decode_mask


class PleShortConvAttentionMetadataBuilder(UpstreamPleShortConvAttentionMetadataBuilder):
    metadata_cls = PleShortConvAttentionMetadata
    # PLE uses the dilated tensor convolution rather than Triton causal_conv1d.
    needs_causal_conv1d_metadata = False

    def _build_normal_chunk_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        query_lens_cpu: torch.Tensor,
        decode_mask_cpu: torch.Tensor,
        variable_decode_mask_cpu: torch.Tensor,
        prefill_mask_cpu: torch.Tensor,
    ) -> PleShortConvAttentionMetadata:
        """Keep replayed decode tokens separate from real prefills and MTP."""
        num_reqs = query_lens_cpu.numel()
        num_tokens = int(query_lens_cpu.sum().item())
        m = common_attn_metadata.unpadded(num_tokens, num_reqs)
        device = m.query_start_loc.device

        # Build indices from authoritative CPU lengths, avoiding a device-side
        # repeat_interleave/nonzero whose output size would require a sync.
        def group_indices(
            req_indices_cpu: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            lengths_cpu = query_lens_cpu[req_indices_cpu]
            starts_cpu = torch.zeros(req_indices_cpu.numel() + 1, dtype=torch.int32)
            torch.cumsum(lengths_cpu, dim=0, out=starts_cpu[1:])
            tokens_cpu = torch.cat(
                [
                    torch.arange(int(m.query_start_loc_cpu[i]), int(m.query_start_loc_cpu[i + 1]))
                    for i in req_indices_cpu.tolist()
                ]
            )
            return (
                async_tensor_h2d(req_indices_cpu, device=device),
                async_tensor_h2d(tokens_cpu, device=device),
                starts_cpu,
                async_tensor_h2d(starts_cpu, device=device),
            )

        chunk_reqs_cpu = variable_decode_mask_cpu.nonzero(as_tuple=True)[0]
        chunk_reqs, chunk_tokens, chunk_starts_cpu, chunk_starts = group_indices(chunk_reqs_cpu)
        block_table = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        chunk = PleShortConvChunkMetadata(
            token_indices=chunk_tokens,
            query_start_loc=chunk_starts,
            state_indices_tensor=block_table[chunk_reqs, 0],
            has_initial_states=m.compute_num_computed_tokens()[chunk_reqs] > 0,
            num_reqs=chunk_reqs_cpu.numel(),
            num_tokens=int(chunk_starts_cpu[-1].item()),
            max_query_len=int(query_lens_cpu[chunk_reqs_cpu].max().item()),
        )

        # The existing non-spec dispatch expects decodes before prefills.
        # Keep a separate token permutation even when no MTP rows exist.
        remaining_reqs_cpu = torch.cat(
            (
                decode_mask_cpu.nonzero(as_tuple=True)[0],
                prefill_mask_cpu.nonzero(as_tuple=True)[0],
            )
        )
        remaining_metadata = None
        remaining_tokens = None
        if remaining_reqs_cpu.numel():
            remaining_reqs, remaining_tokens, starts_cpu, starts = group_indices(remaining_reqs_cpu)
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            selected_seq_lens_cpu = seq_lens_cpu[remaining_reqs_cpu] if seq_lens_cpu is not None else None
            # A fresh base metadata object prevents full-batch cached mappings
            # or Ascend graph padding from leaking into this private subgroup.
            remaining_common = CommonAttentionMetadata(
                query_start_loc=starts,
                query_start_loc_cpu=starts_cpu,
                seq_lens=m.seq_lens[remaining_reqs],
                num_reqs=remaining_reqs_cpu.numel(),
                num_actual_tokens=int(starts_cpu[-1].item()),
                max_query_len=int(query_lens_cpu[remaining_reqs_cpu].max().item()),
                max_seq_len=m.max_seq_len,
                block_table_tensor=m.block_table_tensor[remaining_reqs],
                slot_mapping=m.slot_mapping[remaining_tokens],
                is_prefilling=prefill_mask_cpu[remaining_reqs_cpu],
                seq_lens_cpu_upper_bound=selected_seq_lens_cpu,
            )
            remaining_metadata = self.build(0, remaining_common)

        return PleShortConvAttentionMetadata(
            num_prefills=int(prefill_mask_cpu.sum().item()),
            num_prefill_tokens=int(query_lens_cpu[prefill_mask_cpu].sum().item()),
            num_decodes=int((decode_mask_cpu | variable_decode_mask_cpu).sum().item()),
            num_decode_tokens=int(query_lens_cpu[~prefill_mask_cpu].sum().item()),
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            query_start_loc=m.query_start_loc,
            seq_lens=m.seq_lens,
            normal_chunk=chunk,
            remaining_metadata=remaining_metadata,
            remaining_token_indices=remaining_tokens,
            has_initial_states_p=None,
            query_start_loc_p=None,
            num_computed_tokens_p=None,
            state_indices_tensor_p=None,
            state_indices_tensor_d=None,
            query_start_loc_d=None,
            num_accepted_tokens=None,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        num_reqs_actual: int | None = None,
        **kwargs: Any,
    ) -> PleShortConvAttentionMetadata:
        m = common_attn_metadata
        num_reqs_padded = m.num_reqs
        if num_reqs_actual is None:
            num_reqs_actual = m.num_reqs
        decode_phase_mask_cpu, prefill_mask_cpu = _ple_request_phase_masks(
            m.is_prefilling,
            num_reqs_actual,
        )
        query_lens_cpu = torch.diff(m.query_start_loc_cpu[: num_reqs_actual + 1])
        if query_lens_cpu.numel() != num_reqs_actual:
            raise AssertionError("PLE query lengths must cover every real request")

        # Request phase comes exclusively from ``is_prefilling``. Multi-token
        # decodes can also be normal replay chunks without speculative tokens.
        decode_mask_cpu, spec_sequence_masks_cpu = _ple_decode_execution_masks(
            decode_phase_mask_cpu,
            query_lens_cpu,
        )
        num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
        if num_spec_decodes > 0 and not self.use_spec_decode:
            return self._build_normal_chunk_metadata(
                m,
                query_lens_cpu,
                decode_mask_cpu,
                spec_sequence_masks_cpu,
                prefill_mask_cpu,
            )

        # Preserve the existing graph-safe single-token decode path. Any
        # prefill (including a one-token chunked prefill) or multi-token decode
        # uses the existing explicit mixed/spec metadata path below.
        if num_spec_decodes == 0 and not bool(prefill_mask_cpu.any().item()):
            if not bool(torch.all(query_lens_cpu == 1).item()):
                raise AssertionError("PLE regular decode rows must contain one token")
            return self._build_non_spec_metadata(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                num_decode_draft_tokens_cpu,
                **kwargs,
            )

        del common_prefix_len, fast_build, kwargs
        m = m.unpadded(m.num_actual_tokens, num_reqs_actual)
        query_start_loc = m.query_start_loc
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        has_spec_decode = num_spec_decodes > 0
        spec_sequence_masks = None
        if has_spec_decode:
            if query_start_loc.device.type == "cpu":
                spec_sequence_masks = spec_sequence_masks_cpu
            else:
                spec_sequence_masks = async_tensor_h2d(spec_sequence_masks_cpu, device=query_start_loc.device)

        has_initial_states_p = None
        has_initial_states_d = None
        num_computed_tokens_p = None
        # Original request indices of the non-spec requests, ordered
        # [decodes, prefills]. Used to gather per-request data consistently.
        non_spec_req_idx_cpu: torch.Tensor | None = None

        query_lens = torch.diff(query_start_loc)
        # Per-request classification by mask, NOT by position. With
        # spec-decode, the front decode group can contain both spec-decode
        # requests and plain non-spec single-token decodes. Spec requests are
        # therefore not guaranteed to occupy the first num_spec_decodes slots.
        num_decodes = int(decode_mask_cpu.sum().item())
        num_prefills = int(prefill_mask_cpu.sum().item())
        num_decode_tokens = num_decodes
        num_prefill_tokens = int(query_lens_cpu[prefill_mask_cpu].sum().item())
        num_spec_decode_tokens = int(query_lens_cpu[spec_sequence_masks_cpu].sum().item())
        # Max prefill query length, CPU-side (no device-to-host sync) for the
        # PLE dilated short-conv packing buffer.
        max_prefill_query_len = int(query_lens_cpu[prefill_mask_cpu].max().item()) if num_prefills > 0 else 0

        # Original request indices grouped as
        # [spec | non-spec decode | non-spec prefill]; each group keeps the
        # original (already reordered) relative order via a stable nonzero.
        spec_req_idx_cpu = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0]
        decode_req_idx_cpu = decode_mask_cpu.nonzero(as_tuple=True)[0]
        prefill_req_idx_cpu = prefill_mask_cpu.nonzero(as_tuple=True)[0]
        non_spec_req_idx_cpu = torch.cat((decode_req_idx_cpu, prefill_req_idx_cpu))
        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)
        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)

        if num_decodes == 0 and num_prefills == 0:
            # Pure speculative-decode batch: all real tokens are spec tokens.
            spec_token_indx = torch.arange(
                num_spec_decode_tokens,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            non_spec_token_indx = torch.empty(0, dtype=torch.int32, device=query_start_loc.device)
            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
        else:
            # Mixed batch: build a per-token group key consistent with the
            # request grouping above (spec=0 | decode=1 | prefill=2) and a
            # stable sort, so tokens of each request stay contiguous and in
            # request order. This yields spec tokens first, then the non-spec
            # [decode, prefill] tokens.
            req_group = torch.full(
                (m.num_reqs,),
                2,
                dtype=torch.int64,
                device=query_start_loc.device,
            )
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1
            token_group = torch.repeat_interleave(req_group, query_lens, output_size=int(m.query_start_loc_cpu[-1]))
            token_perm = torch.argsort(token_group, stable=True)
            spec_token_indx = token_perm[:num_spec_decode_tokens]
            non_spec_token_indx = token_perm[num_spec_decode_tokens:]

            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = block_table_tensor[non_spec_req_idx, 0]
            spec_query_start_loc = torch.zeros(
                num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(query_lens[spec_req_idx], dim=0, out=spec_query_start_loc[1:])
            non_spec_query_start_loc = torch.zeros(
                num_decodes + num_prefills + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[non_spec_req_idx],
                dim=0,
                out=non_spec_query_start_loc[1:],
            )

        if num_spec_decodes > 0:
            if num_accepted_tokens is None:
                raise ValueError("PLE multi-token decode requires accepted-token metadata")
            # Accepted-token counts must follow the same request order as the
            # speculative state indices.
            num_accepted_tokens = num_accepted_tokens[spec_req_idx_cpu.to(num_accepted_tokens.device)]

        # Compute the conv-state slots for the non-spec decode/prefill split,
        # plus the initial-state masks and Triton causal_conv1d metadata.
        if non_spec_state_indices_tensor is None:
            state_indices_tensor = block_table_tensor[:0, 0]
        else:
            state_indices_tensor = non_spec_state_indices_tensor

        # Build the regular decode/prefill state metadata inherited from the
        # generic short-conv metadata contract.
        query_start_loc_p = None
        query_start_loc_d = None
        state_indices_tensor_p = None
        state_indices_tensor_d = None
        if num_decodes > 0 or num_prefills > 0:
            num_computed_tokens = m.compute_num_computed_tokens()
            if non_spec_req_idx_cpu is not None:
                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)
                num_computed_tokens = num_computed_tokens[non_spec_req_idx]

            state_indices_tensor_d = state_indices_tensor[:num_decodes]
            state_indices_tensor_p = state_indices_tensor[num_decodes : num_decodes + num_prefills]
            if num_decodes > 0:
                has_initial_states_d = num_computed_tokens[:num_decodes] > 0
                assert non_spec_query_start_loc is not None
                query_start_loc_d = non_spec_query_start_loc[: num_decodes + 1]
            if num_prefills > 0:
                num_computed_tokens_p = num_computed_tokens[num_decodes : num_decodes + num_prefills]
                has_initial_states_p = num_computed_tokens_p > 0
                assert non_spec_query_start_loc is not None
                query_start_loc_p = non_spec_query_start_loc[num_decodes:] - num_decode_tokens

        # Prepare persistent tensors for CUDA graph capture and replay.
        # ``m.num_actual_tokens`` is already padded by the model runner.
        # Request-level buffers use ``m.num_reqs`` while token-level buffers
        # use their independently bounded token count.
        batch_size = num_reqs_padded
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and spec_sequence_masks is not None
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_tokens
        ):
            assert spec_state_indices_tensor is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(spec_state_indices_tensor, non_blocking=True)
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(spec_sequence_masks, non_blocking=True)
            self.spec_sequence_masks[num_spec_decodes:batch_size].fill_(False)
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]

            assert spec_query_start_loc is not None
            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(spec_query_start_loc, non_blocking=True)
            spec_num_query_tokens = spec_query_start_loc[-1]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            assert num_accepted_tokens is not None
            self.num_accepted_tokens[:num_spec_decodes].copy_(num_accepted_tokens, non_blocking=True)
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        return PleShortConvAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_reqs=num_reqs_actual,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            spec_query_len=self.num_spec + 1,
            max_prefill_query_len=max_prefill_query_len,
            query_start_loc=query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_p=has_initial_states_p,
            has_initial_states_d=has_initial_states_d,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_query_start_loc=spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            query_start_loc_p=query_start_loc_p,
            query_start_loc_d=query_start_loc_d,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_computed_tokens_p=num_computed_tokens_p,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            seq_lens=m.seq_lens,
        )
