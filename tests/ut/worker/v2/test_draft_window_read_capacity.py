# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""A single Full prompt must not expand every concurrent window operand."""

import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import prepare_dense_draft_read
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource


def test_full_prompt_does_not_multiply_decode_scratch_by_full_capacity():
    n, block, full_end, full_width = 16, 128, 32767, 33
    ends = torch.tensor([1151] * (n - 1) + [full_end], dtype=torch.int32)
    offsets = torch.tensor([4 * row for row in range(n)] + [4 * (n - 1) + full_width], dtype=torch.int32)
    table = torch.arange(n * 256, dtype=torch.int32).reshape(n, 256)
    read = prepare_dense_draft_read(
        DirectPrefixSource(table, tuple(range(n)), (256,) * n, block),
        exact_device=ends,
        row_indices=torch.arange(n),
        upper_cpu=ends + 1,
        window_size=1024,
        anchor_span=block,
        query_width=full_width,
        causal=False,
        full_rows_cpu=torch.tensor([False] * (n - 1) + [True]),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.clone(),
        retain=lambda *objects: None,
    )
    private_reads = [part.read for part in read.parts] if hasattr(read, "parts") else [read]
    # Bound actual private allocation geometry, rather than a nominal estimate.
    # The original implementation allocated n*32768 KV tokens and n*33*32768
    # mask elements for this mixed request, even though 15 rows only need W.
    kv_tokens = sum(private.slots.numel() for private in private_reads)
    mask_elements = sum(private.mask.numel() for private in private_reads)
    assert kv_tokens == 15 * 1152 + 32768
    assert mask_elements == 15 * 4 * 1152 + full_width * 32768
    assert read.seq_lens.tolist() == [1151] * 15 + [full_end]
    assert read.starts.tolist() == [0] * n
