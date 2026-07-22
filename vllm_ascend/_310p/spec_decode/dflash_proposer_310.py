#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""310P dflash/dspark input construction.

310P has no Triton, so the dflash/dspark ``set_inputs_first_pass`` overrides
here build the draft-model inputs via the AscendC custom op
``npu_copy_and_expand_dflash_inputs`` instead of the Triton
``copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid`` used on other
platforms. These classes are wired onto the shared proposers by
``vllm_ascend.patch.worker.patch_idex_310`` so the generic spec-decode modules
stay free of any 310P coupling.
"""

import functools
from typing import Any

import torch
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer

# On 310P the target spec-verify forward miscomputes for a verify query length
# >= 9 (i.e. num_speculative_tokens >= 8): the per-token pre-attention pipeline
# gets corrupted by the batch token count, collapsing draft acceptance to ~0
# (verified: qlen<=8 healthy, qlen==9 collapses). The attention op, KV slots,
# RoPE positions and verify input ids were all proven correct, so the ceiling is
# a hardware/library limit of the 310P verify path, not this code. Keep
# num_speculative_tokens within the healthy regime; small values are also best
# for acceptance in practice (num_spec=3 gives the highest observed rate).
MAX_RELIABLE_NUM_SPEC_TOKENS_310P = 7


def _draft_cache_block_size_310(proposer: Any) -> int | None:
    """Read the block_size actually used by the allocated draft KV cache.

    The draft ``kv_cache_spec.block_size`` (e.g. 640) is split into kernel
    sub-blocks when the KV cache is allocated (310P NZ layout
    ``(2, num_blocks, (nkv*hd)//16, block_size, 16)``), so the real per-block
    size is the tensor's ``shape[-2]`` (e.g. 64) - NOT
    ``get_supported_kernel_block_sizes()[0]`` (128) that the base proposer used.
    The block_table is expressed in these kernel blocks too, so slot mapping
    must use this size or SplitFuse reads empty blocks (all-zero draft output).
    """
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(proposer.vllm_config, AttentionLayerBase)
    layer = layers[proposer.attn_layer_names[0]]
    cache = getattr(layer, "kv_cache", None)
    # kv_cache is list[per-virtual-engine]; each entry may be a (k, v) tuple or a
    # single stacked tensor. Unwrap to the first real tensor.
    while isinstance(cache, (list, tuple)) and len(cache) > 0:
        cache = cache[0]
    if cache is None or not hasattr(cache, "shape") or cache.dim() < 2:
        return None
    return int(cache.shape[-2])


def _recompute_context_slots_310(
    out_context_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    kbs: int,
    num_context: int,
    num_reqs: int,
) -> None:
    """Rebuild the context KV-cache slots with block_table + kernel_block_size.

    The AscendC op recomputes QUERY slots from ``block_table[pos // kbs] * kbs +
    pos % kbs`` (kbs = corrected 64), but CONTEXT slots are an identity
    passthrough of ``cad.slot_mapping`` which the model runner built for a
    128-block layout. In the allocated 64-block draft cache those context slots
    (e.g. 15*128=1920) point at an unreachable physical block (30) that is not in
    the request's block_table, so the draft cross-attention never sees the
    context K/V. Recompute context slots with the SAME scheme as the query slots
    so context and query land in the same physical blocks.
    """
    if num_context <= 0 or kbs <= 0:
        return
    dev = out_context_slot_mapping.device
    qsl = query_start_loc[: num_reqs + 1].to(device=dev, dtype=torch.long)
    counts = (qsl[1:] - qsl[:-1]).clamp(min=0)
    if int(counts.sum().item()) < num_context:
        return
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=dev), counts
    )[:num_context]
    cpos = context_positions[:num_context].to(device=dev, dtype=torch.long)
    block_num = cpos // kbs
    blk = block_table[req_ids, block_num].to(torch.long)
    slot = (blk * kbs + (cpos % kbs)).to(out_context_slot_mapping.dtype)
    out_context_slot_mapping[:num_context].copy_(slot)


def _ensure_kernel_block_size_matches_cache_310(proposer: Any) -> None:
    """Align the draft ``kernel_block_size`` with the allocated KV cache.

    The base proposer sets ``kernel_block_size = get_supported_kernel_block_sizes()[0]``
    (128), but on 310P the draft KV cache is allocated with a smaller kernel
    block size (e.g. 64, from splitting a 640-block spec). The mismatch makes the
    AscendC draft input builder compute slot mappings for a 128-block cache while
    the real cache/block_table use 64, so the SplitFuse cross-attention reads
    empty blocks and returns all-zero output (acceptance ~0).
    """
    if getattr(proposer, "_kernel_block_size_fixed_310", False):
        return
    proposer._kernel_block_size_fixed_310 = True

    num_spec = getattr(proposer, "num_speculative_tokens", None)
    if num_spec is not None and num_spec > MAX_RELIABLE_NUM_SPEC_TOKENS_310P:
        logger.warning(
            "dflash/dspark on 310P: num_speculative_tokens=%s exceeds the reliable "
            "limit %s. The 310P target spec-verify forward corrupts draft "
            "acceptance (drops to ~0) once the verify query length reaches %s "
            "(num_speculative_tokens>=%s); set num_speculative_tokens<=%s "
            "(num_speculative_tokens=3 gives the best observed acceptance).",
            num_spec,
            MAX_RELIABLE_NUM_SPEC_TOKENS_310P,
            MAX_RELIABLE_NUM_SPEC_TOKENS_310P + 2,
            MAX_RELIABLE_NUM_SPEC_TOKENS_310P + 1,
            MAX_RELIABLE_NUM_SPEC_TOKENS_310P,
        )

    current = getattr(proposer, "kernel_block_size", None)
    try:
        cache_block_size = _draft_cache_block_size_310(proposer)
        if cache_block_size and cache_block_size != current:
            proposer.kernel_block_size = cache_block_size
            logger.info(
                "Aligned dflash draft kernel_block_size %s -> %s to match allocated KV cache",
                current, cache_block_size,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("dflash draft kernel_block_size alignment skipped: %s", exc)


def wrap_dummy_run_with_draft_flag(original):
    """Wrap a proposer ``dummy_run`` so the draft-model forward runs with the
    310P drafting RoPE flag enabled.

    ``dummy_run``'s profile branch calls the draft model directly (not through
    ``_run_merged_draft``, which is where the real flow sets the flag). Without
    the flag, ``_rope_forward_oot`` falls back to the main model's global cos/sin
    slice, which is never populated for a VL main model (it uses MRoPE), leaving
    cos/sin as ``None`` and crashing ``npu_apply_rotary_pos_emb`` with
    ``cos != nullptr``. Enabling the flag makes the draft build cos/sin from its
    own ``cos_sin_cache``. The prior flag value is restored so nesting is safe.
    """

    @functools.wraps(original)
    def dummy_run(self, *args, **kwargs):
        prev_flag = AscendRotaryEmbedding310._is_drafting_update_enabled
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            return original(self, *args, **kwargs)
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(prev_flag)

    return dummy_run


def _copy_and_expand_inputs_ascendc(
    self,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    cad: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor | None,
    num_query_per_req: int,
    batch_size: int,
    num_context: int,
    sample_from_anchor: bool,
) -> torch.Tensor:
    """AscendC (310P) replacement for the Triton dflash/dspark input
    construction kernel.

    Writes the query/context buffers in place (mirroring the Triton path)
    and returns ``token_indices_to_sample``.
    """
    # MRoPE models feed positions as [3, num_context]; the op (like the Triton
    # kernel it replaces) treats target_positions as a flat [num_context] vector,
    # and both its tiling and infershape size the context outputs from
    # target_positions.shape[0]. The Triton kernel reads the first num_context
    # contiguous elements (row 0), so take that row to keep shape[0] equal to the
    # context token count instead of the mrope dim (which would size the context
    # outputs as 3 and mismatch the buffers).
    if target_positions.dim() > 1:
        target_positions = target_positions[0]

    # 310P: the draft KV cache is allocated by splitting the draft spec
    # block_size (e.g. 640) into smaller kernel blocks (e.g. 64), but the base
    # proposer set self.kernel_block_size to get_supported_kernel_block_sizes()[0]
    # (128). Passing 128 here makes the AscendC builder compute draft slot
    # mappings for a 128-block cache while the real cache/block_table use 64, so
    # SplitFuse reads empty blocks and returns all-zero output. Align
    # self.kernel_block_size with the allocated cache once, right before the op.
    _ensure_kernel_block_size_matches_cache_310(self)

    if num_rejected_tokens_gpu is not None:
        num_rejected = num_rejected_tokens_gpu.to(torch.int32)
    else:
        # The op always consumes a real [batch_size] tensor; when the caller
        # has no rejection info we feed an all-zero one.
        num_rejected = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

    (
        out_input_ids,
        out_query_positions,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_token_indices,
    ) = torch.ops._C_ascend.npu_copy_and_expand_dflash_inputs(
        next_token_ids.to(torch.int32),
        target_positions.to(torch.int32),
        cad.slot_mapping.to(torch.int32),
        cad.query_start_loc.to(torch.int32),
        cad.seq_lens.to(torch.int32),
        cad.block_table_tensor.to(torch.int32),
        num_rejected,
        int(self.parallel_drafting_token_id),
        int(self.kernel_block_size),
        int(num_query_per_req),
        int(self.num_speculative_tokens),
        bool(sample_from_anchor),
    )

    num_query_total = batch_size * num_query_per_req

    # The op passes context slots through from cad.slot_mapping (128-block
    # layout) while query slots are recomputed for the 64-block cache. Rebuild
    # context slots with the same block_table + kernel_block_size so context K/V
    # is written where the draft cross-attention actually reads it.
    _recompute_context_slots_310(
        out_context_slot_mapping,
        out_context_positions,
        cad.query_start_loc,
        cad.block_table_tensor,
        int(self.kernel_block_size),
        num_context,
        batch_size,
    )

    self.input_ids[:num_query_total].copy_(out_input_ids[:num_query_total])
    self.positions[:num_query_total].copy_(out_query_positions[:num_query_total])
    self._slot_mapping_buffer[:num_query_total].copy_(out_query_slot_mapping[:num_query_total])
    self._context_positions_buffer[:num_context].copy_(out_context_positions[:num_context])
    self._context_slot_mapping_buffer[:num_context].copy_(out_context_slot_mapping[:num_context])
    return out_token_indices


class AscendDflashProposer310(AscendDflashProposer):
    """310P dflash proposer: builds inputs with the AscendC op (no Triton)."""

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.num_reqs
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        has_num_rejected = num_rejected_tokens_gpu is not None

        token_indices_to_sample = _copy_and_expand_inputs_ascendc(
            self,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            num_query_per_req=num_query_per_req,
            batch_size=batch_size,
            num_context=num_context,
            sample_from_anchor=False,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        # DFlash draft cross-attention is non-causal: the query tokens (last
        # sampled token + parallel-drafting mask tokens) attend bidirectionally
        # to the full context and to each other. On 310P this maps to the
        # non-causal SplitFuse mask.
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None


class AscendDsparkProposer310(AscendDsparkProposer):
    """310P dspark proposer: builds inputs with the AscendC op (no Triton)."""

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # Dspark cross-attention: context K/V from target hidden states,
        # Q from query embeddings (next token + mask tokens).
        batch_size = cad.num_reqs

        # Query length of a single request and the whole batch
        num_query_per_req = self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Newly added hidden_states, need to convert to KV Cache
        num_context = target_token_ids.shape[0]
        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        if n < self._dspark_seed_buffer.shape[0]:
            self._dspark_seed_buffer[n:].fill_(0)

        has_num_rejected = num_rejected_tokens_gpu is not None

        # Remove the rejected token to avoid polluting cross-attention
        token_indices_to_sample = _copy_and_expand_inputs_ascendc(
            self,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            num_query_per_req=num_query_per_req,
            batch_size=batch_size,
            num_context=num_context,
            sample_from_anchor=True,
        )

        # Build attn_metadata
        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None
