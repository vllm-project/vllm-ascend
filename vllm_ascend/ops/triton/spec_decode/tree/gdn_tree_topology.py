# SPDX-License-Identifier: Apache-2.0
"""Initial-state columns and short-convolution indices for one complete tree."""

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
    PARENT_STRIDE: tl.constexpr,
    DEPTH_STRIDE: tl.constexpr,
    PATH_STRIDE: tl.constexpr,
    PATH_CAPACITY: tl.constexpr,
    SLOTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
    # History occupies rows [0, WIDTH - 1); each current node follows it.
    tl.store(window_rows + node * WIDTH + WIDTH - 1, WIDTH - 1 + node, mask=node < SLOTS)
    ancestor = parent
    for distance in tl.static_range(1, WIDTH):
        tap = WIDTH - 1 - distance
        history = tl.minimum(WIDTH - 2, depth + tap)
        src = tl.where(depth >= distance, WIDTH - 1 + ancestor, history)
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
    slots = parents.shape[1] + 1
    init_state_columns = torch.empty((1, slots), device=parents.device, dtype=torch.int64)
    conv_window_rows = torch.empty((slots, width), device=parents.device, dtype=torch.int32)
    conv_prefix_source_rows = torch.empty((1, width - 1), device=parents.device, dtype=torch.int32)
    _gdn_tree_topology_kernel[(1,)](
        parents,
        num_nodes,
        depths,
        accepted,
        previous_path,
        sampled,
        init_state_columns,
        conv_window_rows,
        conv_prefix_source_rows,
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
