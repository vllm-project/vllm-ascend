"""310P DFlash graph compatibility for scalar ``Tensor.size(dim)`` nodes."""

from __future__ import annotations

from typing import Any

import torch
from torch import fx
from vllm.logger import logger

_PATCH_MARKER = "_vllm_ascend_310p_dflash_piecewise_size_compat"


def _replace_nested(value: Any, target: fx.Node, replacement: fx.Node | int) -> Any:
    if value is target:
        return replacement
    if isinstance(value, slice):
        return slice(
            _replace_nested(value.start, target, replacement),
            _replace_nested(value.stop, target, replacement),
            _replace_nested(value.step, target, replacement),
        )
    if isinstance(value, tuple):
        return tuple(_replace_nested(item, target, replacement) for item in value)
    if isinstance(value, list):
        return [_replace_nested(item, target, replacement) for item in value]
    if isinstance(value, dict):
        return {key: _replace_nested(item, target, replacement) for key, item in value.items()}
    return value


def decompose_dim_size_nodes(graph: fx.GraphModule) -> None:
    """Replace scalar ``x.size(dim)`` nodes before vLLM decomposes ``x.size()``.

    vLLM 0.24's Piecewise splitter treats every ``call_method('size')`` as a
    full size tuple. A scalar ``size(dim)`` can therefore remain referenced
    from nested allocation shapes and slices when the splitter erases it.
    """

    size_nodes = list(graph.graph.find_nodes(op="call_method", target="size"))
    for node in size_nodes:
        if len(node.args) != 2 or node.kwargs:
            continue

        tensor_node, raw_dim = node.args
        if not isinstance(tensor_node, fx.Node) or not isinstance(raw_dim, int):
            continue
        example_value = tensor_node.meta.get("example_value")
        if example_value is None:
            raise AssertionError(
                f"Tensor node '{tensor_node.name}' has no example_value metadata. "
                f"Cannot decompose scalar size node '{node.name}'."
            )

        rank = example_value.dim()
        dim = raw_dim + rank if raw_dim < 0 else raw_dim
        if dim < 0 or dim >= rank:
            raise IndexError(f"Dimension out of range for scalar size node '{node.name}': dim={raw_dim}, rank={rank}.")

        dim_value = example_value.shape[dim]
        replacement: fx.Node | int
        if isinstance(dim_value, torch.SymInt):
            with graph.graph.inserting_after(tensor_node):
                replacement = graph.graph.call_function(
                    torch.ops.aten.sym_size.int,
                    args=(tensor_node, dim),
                )
            replacement.meta["example_value"] = dim_value
        elif isinstance(dim_value, int):
            replacement = dim_value
        else:
            raise AssertionError(
                f"Expected SymInt or int for dim {dim} of '{node.name}', got {type(dim_value).__name__}."
            )

        for user in list(node.users):
            user.args = _replace_nested(user.args, node, replacement)
            user.kwargs = _replace_nested(user.kwargs, node, replacement)
        graph.graph.erase_node(node)


def _install_size_node_compat(scope: str) -> None:
    """Install the vLLM 0.24 splitter guard once in the current worker."""

    from vllm.compilation import backends

    original = backends._decompose_size_nodes
    if getattr(original, _PATCH_MARKER, False):
        return

    def decompose_size_nodes_with_scalar_guard(graph: fx.GraphModule) -> None:
        decompose_dim_size_nodes(graph)
        original(graph)

    setattr(decompose_size_nodes_with_scalar_guard, _PATCH_MARKER, True)
    backends._decompose_size_nodes = decompose_size_nodes_with_scalar_guard
    logger.debug(
        "[310p-dflash-graph/compile] scope=%s installed scalar size-node compatibility guard",
        scope,
    )


def install_piecewise_size_node_compat() -> None:
    """Install the guard for the existing 310P DFlash Piecewise path."""
    _install_size_node_compat("piecewise")


def install_full_decode_size_node_compat() -> None:
    """Install the same independently valid guard for FULL_DECODE_ONLY."""
    _install_size_node_compat("full_decode_only")


def install_full_and_piecewise_size_node_compat() -> None:
    """Install the idempotent compile guard for the hybrid graph capability."""
    _install_size_node_compat("full_and_piecewise")
