"""Byte-preserving accepted-row transport shared by KV and hidden states."""

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.ops.triton.spec_decode.tree.kv_transport import gather_tree_cache_rows, scatter_tree_cache_rows_


@dataclass
class TreeRowMove:
    """Device row indices and optional persistent gather storage.

    ``source`` and ``destination`` are [rows, index_dims]. A negative
    destination disables the write. Indices address the leading dimensions;
    each remaining row is transported as bytes, including quantized payloads.
    """

    tensor: torch.Tensor
    source: torch.Tensor
    destination: torch.Tensor
    scratch: torch.Tensor | None = None


def relocate_tree_rows(moves: list[TreeRowMove], *, deduplicate: bool) -> None:
    """Gather every source before writing any aliased destination.

    Paged caches can have several bindings to the same physical view. Their
    destination slots are combined and deduplicated before scatter. Query
    rows occupy disjoint request intervals and do not need that sorting step.
    All data and indices stay on device; only shapes and view identities are
    inspected by the host.
    """
    transfers = {}
    for move in moves:
        tensor = move.tensor
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        if not deduplicate:
            key = (id(move),)
        transfers.setdefault(key, []).append(move)

    plans, writes = {}, []
    for bindings in transfers.values():
        first = bindings[0]
        view = first.tensor.view(torch.int8)
        dimensions = first.source.shape[1]
        # Index tensors are shared across layers. Sort/reorder the small index
        # plan once, before gathering; never sort or reorder each KV payload.
        key = (tuple((id(move.source), id(move.destination)) for move in bindings), tuple(view.shape[:dimensions]))
        if key not in plans:
            source = first.source if len(bindings) == 1 else torch.cat([move.source for move in bindings])
            destination = (
                first.destination if len(bindings) == 1 else torch.cat([move.destination for move in bindings])
            )
            if deduplicate:
                keep = (destination >= 0).all(dim=1)
                flat, capacity = destination[:, 0], view.shape[0]
                for axis in range(1, dimensions):
                    flat = flat * view.shape[axis] + destination[:, axis]
                    capacity *= view.shape[axis]
                flat = torch.where(keep, flat, capacity)
                flat, order = flat.sort(stable=True)
                keep = keep.index_select(0, order)
                duplicate = torch.zeros_like(keep)
                duplicate[1:] = flat[1:] == flat[:-1]
                source = source.index_select(0, order)
                destination = destination.index_select(0, order)
                destination = torch.where((keep & ~duplicate).unsqueeze(1), destination, -1)
            linear_source = source[:, 0]
            for axis in range(1, dimensions):
                linear_source = linear_source * view.shape[axis] + source[:, axis]
            plans[key] = source, destination, linear_source
        source, destination, linear_source = plans[key]
        scratch = (
            first.scratch.view(torch.int8)
            if len(bindings) == 1 and first.scratch is not None
            else view.new_empty((source.shape[0], *view.shape[dimensions:]))
        )
        # Flatten only when it is a view. Padded physical pages must retain
        # their strides; reshape could silently copy an entire KV cache.
        if all(view.stride(axis) == view.shape[axis + 1] * view.stride(axis + 1) for axis in range(dimensions - 1)):
            rows = view.flatten(0, dimensions - 1)
            torch.index_select(rows, 0, linear_source, out=scratch)
        else:
            # Read only selected rows; aten.index can materialize the entire
            # padded hybrid cache before gathering its small accepted path.
            gather_tree_cache_rows(view, source, scratch)
        writes.append((view, destination, scratch))

    for view, destination, data in writes:
        if view.is_contiguous():
            torch_npu.npu_scatter_nd_update_(view, destination, data)
        else:
            scatter_tree_cache_rows_(view, destination, data)
