from dataclasses import dataclass

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel


@triton.jit
def _prepare_tree_decode_inputs_kernel(
    # inputs (read-only)
    idx_mapping_ptr,  # [num_reqs] int64: batch_idx -> req_state_idx
    cu_num_logits_ptr,  # [num_reqs + 1] int32
    query_start_loc_ptr,  # [num_reqs + 1] int32
    num_computed_tokens_ptr,  # [num_req_states] req_states.num_computed_tokens.gpu
    prefill_len_ptr,  # [num_req_states] req_states.prefill_len.gpu
    last_sampled_tokens_ptr,  # [num_req_states] int32
    draft_tokens_ptr,  # [num_req_states, S] int32
    draft_tokens_stride,
    tree_depths_ptr,  # [num_req_states, S] int32
    tree_depths_stride,
    # outputs
    expanded_idx_mapping_ptr,  # [total_num_logits] int64
    expanded_local_pos_ptr,  # [total_num_logits] int32
    positions_ptr,  # [max_num_tokens] int64 (RoPE, depth-based)
    kv_cache_positions_ptr,  # [max_num_tokens] int64 (unique KV token index)
    seq_lens_ptr,  # [max_num_reqs] int32
    input_ids_ptr,  # [max_num_tokens] int32
    logits_indices_ptr,  # [total_num_logits] int64
    max_num_reqs,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs (same as the
        # standalone prepare kernels).
        for i in tl.range(num_reqs, max_num_reqs, 1024):
            block = i + tl.arange(0, 1024)
            tl.store(seq_lens_ptr + block, 0, mask=block < max_num_reqs)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    # ---- _prepare_tree_spec_pos_seq_lens_kernel body (decode branch) ----
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    # RoPE: root at num_computed; drafts use tree depth (siblings may share).
    # KV slots: unique token index so reshape_and_cache does not collide.
    tl.store(positions_ptr + start, num_computed_tokens)
    tl.store(kv_cache_positions_ptr + start, num_computed_tokens)
    for i in tl.range(1, query_len, BLOCK_T):
        block = i + tl.arange(0, BLOCK_T)
        mask = block < query_len
        depth = tl.load(tree_depths_ptr + req_state_idx * tree_depths_stride + block - 1, mask=mask)
        tl.store(positions_ptr + start + block, depth + num_computed_tokens, mask=mask)
        tl.store(kv_cache_positions_ptr + start + block, num_computed_tokens + block, mask=mask)

    # ---- _expand_idx_mapping_kernel body ----
    logits_start = tl.load(cu_num_logits_ptr + req_id)
    logits_end = tl.load(cu_num_logits_ptr + req_id + 1)
    num_logits = logits_end - logits_start
    for i in tl.range(0, num_logits, BLOCK_T):
        block = i + tl.arange(0, BLOCK_T)
        mask = block < num_logits
        tl.store(expanded_idx_mapping_ptr + logits_start + block, req_state_idx, mask=mask)
        tl.store(expanded_local_pos_ptr + logits_start + block, block, mask=mask)

    # ---- _combine_sampled_and_draft_tokens_kernel body ----
    # Logits indices: always written, even for prefill-like rows (the
    # standalone kernel writes them before the prefill early-out).
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    token_logits_start = end - num_logits
    for i in tl.range(0, num_logits, BLOCK_T):
        block = i + tl.arange(0, BLOCK_T)
        mask = block < num_logits
        tl.store(logits_indices_ptr + logits_start + block, token_logits_start + block, mask=mask)

    if seq_len > prefill_len:
        # Keep prompt-tail slots intact; only rewrite generated-token slots.
        first_logit_seq_pos = seq_len - num_logits
        if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
            # Write the last sampled token ID to input_ids.
            last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
            tl.store(input_ids_ptr + token_logits_start, last_token_id)

        num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS
        if num_draft_tokens > 0:
            for i in tl.range(0, num_draft_tokens, BLOCK_T):
                block = i + tl.arange(0, BLOCK_T)
                mask = block < num_draft_tokens
                draft_tokens = tl.load(draft_tokens_ptr + req_state_idx * draft_tokens_stride + block, mask=mask)
                tl.store(input_ids_ptr + end - num_draft_tokens + block, draft_tokens, mask=mask)


class PrepareTreeDecodeInputsKernel(AscendTritonJitKernel["PrepareTreeDecodeInputsKernel.CompileKey"]):
    kernel = _prepare_tree_decode_inputs_kernel

    @dataclass(frozen=True)
    class CompileKey:
        draft_tokens_stride: int
        tree_depths_stride: int
        max_num_reqs: int
        num_new_sampled_tokens: int
        block_t: int

    def get_warmup_keys(
        self, draft_tokens_stride, tree_depths_stride, max_num_reqs, num_new_sampled_tokens, block_t
    ) -> list[CompileKey]:
        return [self.CompileKey(draft_tokens_stride, tree_depths_stride, max_num_reqs, num_new_sampled_tokens, block_t)]

    def warmup_inputs(self, compile_key: CompileKey) -> dict:
        i32 = TritonWarmupTensor(torch.int32)
        i64 = TritonWarmupTensor(torch.int64)
        return dict(
            idx_mapping_ptr=i64,
            cu_num_logits_ptr=i32,
            query_start_loc_ptr=i32,
            num_computed_tokens_ptr=i32,
            prefill_len_ptr=i32,
            last_sampled_tokens_ptr=i32,
            draft_tokens_ptr=i32,
            draft_tokens_stride=compile_key.draft_tokens_stride,
            tree_depths_ptr=i32,
            tree_depths_stride=compile_key.tree_depths_stride,
            expanded_idx_mapping_ptr=i64,
            expanded_local_pos_ptr=i32,
            positions_ptr=i64,
            kv_cache_positions_ptr=i64,
            seq_lens_ptr=i32,
            input_ids_ptr=i32,
            logits_indices_ptr=i64,
            max_num_reqs=compile_key.max_num_reqs,
            NUM_NEW_SAMPLED_TOKENS=compile_key.num_new_sampled_tokens,
            BLOCK_T=compile_key.block_t,
        )

    @kernel_launcher
    def __call__(
        self,
        idx_mapping_ptr,
        cu_num_logits_ptr,
        query_start_loc_ptr,
        num_computed_tokens_ptr,
        prefill_len_ptr,
        last_sampled_tokens_ptr,
        draft_tokens_ptr,
        draft_tokens_stride: int,
        tree_depths_ptr,
        tree_depths_stride: int,
        expanded_idx_mapping_ptr,
        expanded_local_pos_ptr,
        positions_ptr,
        kv_cache_positions_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        logits_indices_ptr,
        max_num_reqs: int,
        NUM_NEW_SAMPLED_TOKENS: int,
        BLOCK_T: int,
    ) -> LaunchSpec:
        return (idx_mapping_ptr.shape[0] + 1,), {}


fused_tree_spec_prep = PrepareTreeDecodeInputsKernel()


def prepare_tree_decode_inputs(
    idx_mapping: torch.Tensor,
    cu_num_logits: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    prefill_len: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    tree_depths: torch.Tensor,
    positions: torch.Tensor,
    kv_cache_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    input_ids: torch.Tensor,
    total_num_logits: int,
    num_new_sampled_tokens: int,
    block_t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the fused prep kernel; return (expanded_idx_mapping,
    expanded_local_pos, logits_indices) — the same fresh outputs the
    standalone kernels would have produced."""
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32, device=idx_mapping.device)
    logits_indices = torch.empty(total_num_logits, dtype=torch.int64, device=input_ids.device)
    fused_tree_spec_prep(
        idx_mapping,
        cu_num_logits,
        query_start_loc,
        num_computed_tokens,
        prefill_len,
        last_sampled_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        tree_depths,
        tree_depths.stride(0),
        expanded_idx_mapping,
        expanded_local_pos,
        positions,
        kv_cache_positions,
        seq_lens,
        input_ids,
        logits_indices,
        seq_lens.shape[0],
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
        BLOCK_T=block_t,
    )
    return expanded_idx_mapping, expanded_local_pos, logits_indices
