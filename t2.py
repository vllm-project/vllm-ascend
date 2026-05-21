import torch
import numpy as np



dcp_mtp_attn_mask_buffer = torch.ones(1,4,16384, dtype=torch.bool)
def get_cp_local_seq_lens(
    seq_lens: torch.Tensor,
    pcp_world_size: int = 1,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using pcp or dcp, kv_cache size stored on each rank may be different,
    use this function to calculate split decode seq_lens of each (p/d)cp rank.
    """
    num_requests = seq_lens.size(0)
    total_world_size = pcp_world_size * dcp_world_size
    seq_lens_tiled = seq_lens.unsqueeze(-1).repeat(1, total_world_size)
    rank_offsets = torch.arange(total_world_size, dtype=torch.int32).unsqueeze(0).repeat(num_requests, 1)
    base = seq_lens_tiled // cp_kv_cache_interleave_size // total_world_size * cp_kv_cache_interleave_size
    remainder = seq_lens_tiled - base * total_world_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = (base + remainder).reshape([-1, pcp_world_size, dcp_world_size])
    return dcp_local_seq_lens

def generate_mtp_attention_mask_for_decode(
    decode_num_computed_tokens: list[int],
    decode_num_scheduled_tokens: np.ndarray,
) -> list[torch.Tensor | None]:
    """
    Generate MTP attention masks for decode requests in PCP mode.

    This function handles the case where decode requests with MTP (speculative decoding)
    need attention masks computed based on the local sequence after load balancing.

    New MTP token allocation logic (using position % cp_size):
    - History tokens are already split via DualChunkSwap
    - MTP tokens are allocated based on (history_len + mtp_idx) % cp_size
    - Each rank only computes mask for tokens assigned to itself

    Example:
        - pcp=1, dcp=2 (cp_size=2)
        - history_len=5: [a,b,c,d,e] split via DualChunkSwap
            - cp0: [a,b,c] (positions 0,1,2) -> 3 tokens
            - cp1: [d,e] (positions 3,4) -> 2 tokens
        - num_scheduled_tokens=4: [f,g,h,i] (positions 5,6,7,8)
        - MTP allocation by position % cp_size:
            - f: pos 5 % 2 = 1 -> rank1
            - g: pos 6 % 2 = 0 -> rank0
            - h: pos 7 % 2 = 1 -> rank1
            - i: pos 8 % 2 = 0 -> rank0
        - Final:
            - rank0: [a,b,c,g,i] positions [0,1,2,6,8] -> mask 4x5
            - rank1: [d,e,f,h] positions [3,4,5,7] -> mask 4x4

    Args:
        decode_num_computed_tokens: List of global history lengths for decode requests
        decode_num_scheduled_tokens: Array of scheduled token counts for decode requests

    Returns:
        List of attention mask tensors for decode requests, one per request
        Each mask has shape [num_mtp_tokens, num_local_tokens]
        Returns None for non-decode requests or when no decode requests exist
    """
    # Calculate combined CP rank and size (same as dcp_rank/dcp_size in reference)
    cp_rank = 0
    cp_size = 2
    assert cp_size > 1, "cp_size must be greater than 1"

    # Get local history tokens on each rank (similar to k_lens in reference)
    num_computed_tokens_tensor = torch.tensor(decode_num_computed_tokens, dtype=torch.int32)
    local_seq_lens = get_cp_local_seq_lens(
        num_computed_tokens_tensor,
        1,
        2,
        1,
    )
    # Shape: [num_decode_reqs, pcp_world_size, dcp_world_size]
    local_history_lens = local_seq_lens[:, 0, 0].int()

    # q_lens = mtp_token_len (num_scheduled_tokens)
    q_lens = torch.tensor(decode_num_scheduled_tokens[:1], dtype=torch.int32)
    # global_histories = decode_num_computed_tokens (global history lengths)
    global_histories = torch.tensor(decode_num_computed_tokens, dtype=torch.int32)
    # total_lens = global_history + mtp_token_len (global sequence length)
    total_lens = global_histories + q_lens
    # context_lens = total_lens - q_lens (consistent with reference)
    context_lens = total_lens - q_lens

    # max indices for global sequences
    max_indices = total_lens - 1

    # if max_indices are smaller than cp_rank, current rank has no cache, is invalid
    valid = (max_indices >= cp_rank)

    if not valid.any():
        return dcp_mtp_attn_mask_buffer[:1]

    # local kv lens on current cp_rank (similar to k_lens calculation in reference)
    # k_lens = floor((max_index - cp_rank) / cp_size) + 1
    k_lens = torch.div(max_indices - cp_rank, cp_size, rounding_mode="floor") + 1
    k_lens = torch.where(valid, k_lens, torch.zeros_like(k_lens))

    # obtain the max length of all prefill reqs (same as reference)
    max_q = int(q_lens[valid].max().item())
    max_k = int(k_lens[valid].max().item())

    # generate local q and k indices
    q_indices = torch.arange(max_q, dtype=torch.int32)
    k_indices = torch.arange(max_k, dtype=torch.int32)

    # valid q and k indices of each reqs
    valid_q = valid[:, None] & (q_indices[None, :] < q_lens[:, None])
    valid_k = valid[:, None] & (k_indices[None, :] < k_lens[:, None])

    # k_upper = floor((context_lens + q_indices - cp_rank) / cp_size)
    k_upper = torch.div(context_lens[:, None] + q_indices - cp_rank, cp_size, rounding_mode="floor")
    k_upper = torch.where(
        valid_q,
        torch.clamp(k_upper, min=-1),
        k_upper.new_full(k_upper.shape, -1)
    )

    # mask: k_idx > k_upper means k_idx is NOT in the masked region
    # (i.e., k_idx can be attended to)
    mask = (k_indices[None, None, :] > k_upper[:, :, None]) & (k_upper[:, :, None] >= 0)
    valid_positions = valid_q[:, :, None] & valid_k[:, None, :]

    # Get flattened valid mask (same as reference)
    custom_mask = torch.masked_select(mask, valid_positions)

    # Pre-allocate output buffer
    mtp_attn_mask = dcp_mtp_attn_mask_buffer[:1]
    mtp_attn_mask.zero_()

    # Create output indices for each position
    req_indices = torch.arange(1, dtype=torch.long)[:, None, None].expand(1, max_q, max_k)
    q_indices_expanded = q_indices[None, :, None].expand(1, max_q, max_k)
    k_indices_expanded = k_indices[None, None, :].expand(1, max_q, max_k)

    # Get indices for valid positions only
    valid_req_grid = req_indices[valid_positions]
    valid_q_grid = q_indices_expanded[valid_positions]
    valid_k_grid = k_indices_expanded[valid_positions]

    # Compute flat target indices for output buffer
    query_len = 4
    flat_target = valid_req_grid * query_len * 16384 + valid_q_grid * 16384 + valid_k_grid

    # Scatter to output buffer
    mtp_attn_mask_flat = mtp_attn_mask.view(-1)
    mtp_attn_mask_flat[flat_target] = custom_mask

    return mtp_attn_mask


decode_num_computed_tokens = [5]
decode_num_scheduled_tokens = np.array([4])

mask = generate_mtp_attention_mask_for_decode(decode_num_computed_tokens,decode_num_scheduled_tokens)

print(mask[:1, :4, :5].numpy())


