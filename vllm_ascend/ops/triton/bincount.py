# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def token_bin_counts_and_mask_kernel(
    tokens_ptr,
    tokens_batch_stride,
    tokens_seq_stride,
    batch_size,
    seq_len,
    vocab_size,
    bin_counts_ptr,
    counts_batch_stride,
    counts_vocab_stride,
    BIG_CORE_NUM: tl.constexpr,
    BIG_ROW_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    SMALL_ROW_BLOCK_SIZE = BIG_ROW_BLOCK_SIZE - 1
    if pid < BIG_CORE_NUM:
        row_block_size = BIG_ROW_BLOCK_SIZE
        row_start_idx = pid * BIG_ROW_BLOCK_SIZE
    else:
        row_block_size = SMALL_ROW_BLOCK_SIZE
        row_start_idx = BIG_CORE_NUM * BIG_ROW_BLOCK_SIZE + (pid - BIG_CORE_NUM) * SMALL_ROW_BLOCK_SIZE

    row_end_idx = min(row_start_idx + row_block_size, batch_size)
    for batch_idx in range(row_start_idx, row_end_idx):
        batch_tokens_start = tokens_ptr + batch_idx * tokens_batch_stride
        batch_counts_start = bin_counts_ptr + batch_idx * counts_batch_stride

        for pos in range(seq_len):
            token = tl.load(batch_tokens_start + pos * tokens_seq_stride)
            if token >= 0 and token < vocab_size:
                count_ptr = batch_counts_start + token * counts_vocab_stride
                # Each (batch_idx, token) pair is only written by one program.
                old = tl.load(count_ptr)
                tl.store(count_ptr, old + 1)


def get_token_bin_counts_and_mask_triton(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()
    n_rows, n_cols = tokens.shape
    if num_seqs is not None and num_seqs > 0:
        assert n_rows == num_seqs, (
            "tokens rows must match num_seqs to avoid silently skipping rows: "
            f"tokens.shape[0]={n_rows}, num_seqs={num_seqs}"
        )
        n_rows = num_seqs

    bin_counts = torch.zeros((n_rows, vocab_size), dtype=torch.int32, device=tokens.device)
    if not tokens.is_contiguous():
        tokens = tokens.contiguous()
    if not bin_counts.is_contiguous():
        bin_counts = bin_counts.contiguous()

    big_row_block_size = triton.cdiv(n_rows, core_num)
    # cdiv guarantees big_row_block_size * core_num >= n_rows.
    # big_core_num cores handle big_row_block_size rows each;
    # the remaining cores handle (big_row_block_size - 1) rows each.
    big_core_num = max(
        0,
        min(
            core_num - (big_row_block_size * core_num - n_rows),
            core_num,
        ),
    )
    grid = (min(n_rows, core_num),)

    token_bin_counts_and_mask_kernel[grid](
        tokens,
        tokens.stride(0),
        tokens.stride(1),
        n_rows,
        n_cols,
        vocab_size,
        bin_counts,
        bin_counts.stride(0),
        bin_counts.stride(1),
        BIG_CORE_NUM=big_core_num,
        BIG_ROW_BLOCK_SIZE=big_row_block_size,
    )
    return bin_counts, bin_counts > 0
