# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TileLang kernels for replicated GLM-5.3-Flash NoPE attention on Ascend 950."""

from functools import lru_cache


@lru_cache(maxsize=128)
def compile_mtp_decode(
    requests,
    index_pages,
    main_pages,
    ring_pages,
    *,
    ring_table_stride=1,
    index_width=2176,
    tokens=None,
    model_context=139264,
    table_columns=None,
    native_output=False,
    native_head_weights=False,
):
    """Compile the six-query C8 decode graph. Use the tensor layout in mtp.live_cache_abi.

    The main and indexer page tables retain their own physical row strides.
    The caller owns the workspace, output and writable cache buffers.
    """
    from .mtp import PAGE_SIZE, Q6_MAX_CONTEXT, score_context
    from .mtp.entry import build

    if not 0 < model_context <= Q6_MAX_CONTEXT:
        raise ValueError("Expected an MTP5 context between 1 and 1048576 tokens")
    required = (model_context + PAGE_SIZE - 1) // PAGE_SIZE
    main_cols, index_cols = table_columns or (required, required)
    if min(main_cols, index_cols) < required:
        raise ValueError("Page tables cannot cover the configured context")
    return build(
        requests,
        requests * 6 if tokens is None else tokens,
        index_pages,
        score_context(model_context),
        main_pages,
        main_cols,
        ring_pages,
        ring_table_stride,
        index_width,
        index_cols,
        native_output=native_output,
        native_head_weights=native_head_weights,
    )


def compile_single_decode(
    batch,
    pages,
    context,
    *,
    index_width=512,
    prepacked_heads=False,
    packed_output=False,
    index_pages=None,
    table_columns=None,
    tail_layout=None,
    index_norm_eps=1e-5,
    mask_padding=False,
    resident_workspace=False,
):
    """Compile a one-query C8 kernel. The default ABI accepts BF16 index head weights.

    With resident_workspace, QA has at least 32 padded rows per request for batches two to four.
    """
    from .single.entry import compile_decode

    return compile_decode(
        batch,
        pages,
        context,
        index_width=index_width,
        prepacked_heads=prepacked_heads,
        packed_output=packed_output,
        index_pages=index_pages,
        table_columns=None if table_columns is None else tuple(table_columns),
        tail_layout=None if tail_layout is None else tuple(tail_layout),
        index_norm_eps=index_norm_eps,
        mask_padding=mask_padding,
        resident_workspace=resident_workspace,
    )
