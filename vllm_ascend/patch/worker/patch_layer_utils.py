import torch

from vllm.triton_utils import tl, triton
import vllm.model_executor.layers.utils as lu


@triton.jit
def _token_bin_counts_and_mask_kernel(
    tokens_ptr,
    tokens_batch_stride,
    tokens_seq_stride,
    batch_size,
    seq_len,
    vocab_size,
    bin_counts_ptr,
    counts_batch_stride,
    counts_vocab_stride,
    MAX_GRID_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    batches_per_thread = (batch_size + MAX_GRID_SIZE - 1) // MAX_GRID_SIZE
    start_batch = pid * batches_per_thread
    end_batch = min(start_batch + batches_per_thread, batch_size)
    
    for batch_idx in range(start_batch, end_batch):
        batch_tokens_start = tokens_ptr + batch_idx * tokens_batch_stride
        batch_counts_start = bin_counts_ptr + batch_idx * counts_batch_stride
        
        for pos in range(seq_len):
            token = tl.load(batch_tokens_start + pos * tokens_seq_stride)
            if token < vocab_size:
                count_ptr = batch_counts_start + token * counts_vocab_stride
                tl.atomic_add(count_ptr, 1)

def get_token_bin_counts_and_mask_triton(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = tokens.device
    
    if tokens.dim() == 2:
        batch_size, seq_len = tokens.shape
    else:
        if num_seqs is None:
            raise ValueError("num_seqs must be provided for 1D input")
        total_len = tokens.shape[0]
        seq_len = total_len // num_seqs
        tokens = tokens.view(num_seqs, seq_len)
        batch_size = num_seqs
    
    if num_seqs is not None and num_seqs > 0:
        batch_size = num_seqs
    
    bin_counts = torch.zeros(
        (batch_size, vocab_size), 
        dtype=torch.int32,
        device=device
    )
    if not tokens.is_contiguous():
        tokens = tokens.contiguous()
    if not bin_counts.is_contiguous():
        bin_counts = bin_counts.contiguous()
    
    grid_size = min(batch_size, 40)
    grid_size = max(1, grid_size)

    _token_bin_counts_and_mask_kernel[(grid_size,)](
        tokens,
        tokens.stride(0),
        tokens.stride(1),
        batch_size,
        seq_len,
        vocab_size,
        bin_counts,
        bin_counts.stride(0),
        bin_counts.stride(1),
        MAX_GRID_SIZE=grid_size
    )
    mask = bin_counts > 0
    return bin_counts, mask

# add HAS_TRITON check here
lu.get_token_bin_counts_and_mask = get_token_bin_counts_and_mask_triton