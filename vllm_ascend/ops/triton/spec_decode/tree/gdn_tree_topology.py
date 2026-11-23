# SPDX-License-Identifier: Apache-2.0
"""Initial-state columns and short-convolution indices for uniform verification rows."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gdn_tree_topology_kernel(
    parents,
    node_count,
    depths,
    accepted,
    previous_path,
    sampled,
    init_columns,
    window_rows,
    prefix_src,
    PARENT_ROW_STRIDE: tl.constexpr,
    DEPTH_ROW_STRIDE: tl.constexpr,
    PATH_ROW_STRIDE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    SAMPLED_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    PARENT_STRIDE: tl.constexpr,
    DEPTH_STRIDE: tl.constexpr,
    PATH_STRIDE: tl.constexpr,
    PATH_CAPACITY: tl.constexpr,
    SLOTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    parents += row * PARENT_ROW_STRIDE
    depths += row * DEPTH_ROW_STRIDE
    previous_path += row * PATH_ROW_STRIDE
    node_count += row * COUNT_STRIDE
    accepted += row * ACCEPTED_STRIDE
    sampled += row * SAMPLED_STRIDE
    init_columns += row * SLOTS
    window_rows += row * SLOTS * WIDTH
    prefix_src += row * (WIDTH - 1)
    history_base = row * (WIDTH - 1)
    activation_base = ROWS * (WIDTH - 1) + row * SLOTS
    node = tl.arange(0, BLOCK)
    live_child = (node > 0) & (node < SLOTS)
    child = tl.maximum(0, node - 1)
    is_tree = tl.load(node_count) > 0
    parent = tl.load(parents + child * PARENT_STRIDE, mask=live_child, other=0)
    parent = tl.where(is_tree, parent, node - 1)
    root_column = tl.maximum(tl.load(accepted) - 1, 0)
    tl.store(init_columns + node, tl.where(node == 0, root_column, parent).to(tl.int64), mask=node < SLOTS)

    depth = tl.load(depths + child * DEPTH_STRIDE, mask=live_child, other=0)
    depth = tl.where(is_tree, depth, node)
    # All requests' committed history precedes the token-major activations.
    tl.store(window_rows + node * WIDTH + WIDTH - 1, activation_base + node, mask=node < SLOTS)
    ancestor = parent
    for distance in tl.static_range(1, WIDTH):
        tap = WIDTH - 1 - distance
        history = tl.minimum(WIDTH - 2, depth + tap)
        src = tl.where(depth >= distance, activation_base + ancestor, history_base + history)
        tl.store(window_rows + node * WIDTH + tap, src, mask=node < SLOTS)
        grandparent = tl.load(
            parents + tl.maximum(0, ancestor - 1) * PARENT_STRIDE, mask=(ancestor > 0) & (ancestor < SLOTS), other=0
        )
        ancestor = tl.where(is_tree, grandparent, ancestor - 1)

    last_depth = tl.maximum(-1, tl.load(sampled) - 1)
    tap_depth = last_depth - (WIDTH - 2) + node
    path_node = tl.load(
        previous_path + tl.maximum(0, tap_depth - 1) * PATH_STRIDE,
        mask=(node < WIDTH - 1) & (tap_depth >= 1) & (tap_depth <= PATH_CAPACITY),
        other=0,
    )
    conv_prefix_source_rows = WIDTH - 1 + tl.where(tap_depth >= 1, path_node, tap_depth)
    tl.store(prefix_src + node, conv_prefix_source_rows, mask=node < WIDTH - 1)


def build_gdn_tree_state_and_conv_indices(parents, num_nodes, depths, accepted, previous_path, sampled, width):
    """Produce physical-page-independent inputs once on the caller's stream."""
    rows, budget = parents.shape
    slots = budget + 1
    init_state_columns = torch.empty((rows, slots), device=parents.device, dtype=torch.int64)
    conv_window_rows = torch.empty((rows * slots, width), device=parents.device, dtype=torch.int32)
    conv_prefix_source_rows = torch.empty((rows, width - 1), device=parents.device, dtype=torch.int32)
    _gdn_tree_topology_kernel[(rows,)](
        parents,
        num_nodes,
        depths,
        accepted,
        previous_path,
        sampled,
        init_state_columns,
        conv_window_rows,
        conv_prefix_source_rows,
        parents.stride(0),
        depths.stride(0),
        previous_path.stride(0),
        num_nodes.stride(0),
        accepted.stride(0),
        sampled.stride(0),
        rows,
        parents.stride(1),
        depths.stride(1),
        previous_path.stride(1),
        previous_path.shape[1],
        slots,
        width,
        triton.next_power_of_2(max(slots, width)),
        num_warps=1,
    )
    return init_state_columns, conv_window_rows, conv_prefix_source_rows
