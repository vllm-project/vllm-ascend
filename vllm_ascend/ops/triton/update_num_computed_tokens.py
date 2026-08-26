

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

_DEFAULT_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["n_elements"])
def _update_num_computed_tokens_kernel(
    num_computed_tokens_ptr,  
    num_accepted_tokens_ptr,  
    prev_positions_ptr,  
    valid_sampled_token_count_ptr,  
    prev_num_draft_tokens_ptr,  
    cpu_num_computed_tokens_ptr, 
    n_elements,  
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    prev_pos = tl.load(prev_positions_ptr + offsets, mask=mask, other=0)
    gather_idx = tl.maximum(prev_pos, 0)

    valid_counts = tl.load(valid_sampled_token_count_ptr + gather_idx, mask=mask, other=0)
    prev_computed = tl.load(num_computed_tokens_ptr + gather_idx, mask=mask, other=0)
    prev_drafts = tl.load(prev_num_draft_tokens_ptr + gather_idx, mask=mask, other=0)
    cpu_computed = tl.load(cpu_num_computed_tokens_ptr + offsets, mask=mask, other=0)
    cur_accepted = tl.load(num_accepted_tokens_ptr + offsets, mask=mask, other=0)

    participating = (prev_pos >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.to(prev_computed.dtype)

    out_computed = tl.where(participating, corrected, cpu_computed.to(prev_computed.dtype))
    out_accepted = tl.where(participating, valid_counts.to(cur_accepted.dtype), cur_accepted)

    tl.store(num_computed_tokens_ptr + offsets, out_computed, mask=mask)
    tl.store(num_accepted_tokens_ptr + offsets, out_accepted, mask=mask)


def update_num_computed_tokens_triton(
    num_computed_tokens,
    num_accepted_tokens,
    prev_positions,
    valid_sampled_token_count,
    prev_num_draft_tokens,
    cpu_num_computed_tokens,
) -> None:
    n = prev_positions.shape[0]
    if n == 0:
        return
    block_size = max(_DEFAULT_BLOCK_SIZE, next_power_of_2(n))
    _update_num_computed_tokens_kernel[(1,)](
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
        n,
        BLOCK_SIZE=block_size,
    )
