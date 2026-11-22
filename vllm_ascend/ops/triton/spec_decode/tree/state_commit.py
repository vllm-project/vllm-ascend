# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape accepted-path bookkeeping for Qwen GDN tree state."""

from vllm.triton_utils import tl, triton


@triton.jit
def _commit_tree_snapshot_kernel(
    mapping,
    path,
    sampled,
    tree_rows,
    previous_path,
    previous_sampled,
    mapping_stride: tl.constexpr,
    path_stride_r: tl.constexpr,
    path_stride_c: tl.constexpr,
    sampled_stride: tl.constexpr,
    tree_row_stride: tl.constexpr,
    previous_path_stride_r: tl.constexpr,
    previous_path_stride_c: tl.constexpr,
    previous_sampled_stride: tl.constexpr,
    CAPACITY: tl.constexpr,
    PATH_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PATH: tl.constexpr,
    HAS_TREE_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(mapping + row * mapping_stride).to(tl.int64)
    live = (request >= 0) & (request < CAPACITY)
    count = tl.full((), 0, tl.int32)
    if HAS_PATH:
        columns = tl.arange(0, BLOCK)
        values = tl.load(
            path + row * path_stride_r + columns * path_stride_c,
            mask=columns < PATH_WIDTH,
            other=0,
        ).to(tl.int32)
        tl.store(
            previous_path + request * previous_path_stride_r + columns * previous_path_stride_c,
            values,
            mask=live & (columns < PATH_WIDTH),
        )
        count = tl.load(sampled + row * sampled_stride).to(tl.int32)
        if HAS_TREE_ROWS:
            is_tree = tl.load(tree_rows + row * tree_row_stride).to(tl.int1)
            count = tl.where(is_tree, count, 0)
    tl.store(previous_sampled + request * previous_sampled_stride, count, mask=live)


def update_tree_acceptance_history_(
    previous_path,
    previous_sampled,
    idx_mapping,
    path=None,
    num_sampled=None,
    is_tree_row=None,
):
    """Commit all rows without materializing a device-dependent row count.

    Negative request IDs are padding and never write a request slot. A valid
    linear/prefill row clears its saved sampled count, so stale tree history
    cannot survive a step which did not verify a tree. The upstream state
    postprocessor still owns the accepted recurrent-state cursor.
    """
    if idx_mapping.ndim != 1 or previous_path.ndim != 2 or previous_sampled.ndim != 1:
        raise ValueError("Tree snapshot bookkeeping requires vector IDs/counts and a path matrix")
    rows = idx_mapping.shape[0]
    if previous_path.shape[0] != previous_sampled.shape[0]:
        raise ValueError("Tree snapshot path/count capacities differ")
    has_path = path is not None and num_sampled is not None
    if has_path and path.ndim != 2:
        raise ValueError("Accepted tree paths must be a matrix")
    width = path.shape[1] if has_path else 0
    if has_path and (
        path.shape[0] != rows or num_sampled.ndim != 1 or num_sampled.shape[0] != rows or width > previous_path.shape[1]
    ):
        raise ValueError("Accepted tree path/count shape exceeds the snapshot buffers")
    if is_tree_row is not None and (is_tree_row.ndim != 1 or is_tree_row.shape[0] != rows):
        raise ValueError("Tree-row flags must match the request batch")
    if rows == 0:
        return
    path_input = path if has_path else previous_path
    sampled_input = num_sampled if has_path else previous_sampled
    tree_input = is_tree_row if is_tree_row is not None else previous_sampled
    _commit_tree_snapshot_kernel[(rows,)](
        idx_mapping,
        path_input,
        sampled_input,
        tree_input,
        previous_path,
        previous_sampled,
        idx_mapping.stride(0),
        path_input.stride(0),
        path_input.stride(1),
        sampled_input.stride(0),
        tree_input.stride(0),
        previous_path.stride(0),
        previous_path.stride(1),
        previous_sampled.stride(0),
        CAPACITY=previous_sampled.shape[0],
        PATH_WIDTH=width,
        BLOCK=triton.next_power_of_2(max(width, 1)),
        HAS_PATH=has_path,
        HAS_TREE_ROWS=is_tree_row is not None,
        num_warps=1,
    )
