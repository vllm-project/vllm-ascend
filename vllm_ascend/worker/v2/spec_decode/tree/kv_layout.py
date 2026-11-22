import torch
import torch_npu

from vllm_ascend.worker.v2.spec_decode.tree.transport import TreeRowMove, relocate_tree_rows

# Cached arange / column buffers keyed by device (grow on demand).
_dst_off_arange: dict[torch.device, torch.Tensor] = {}
_root_col: dict[torch.device, torch.Tensor] = {}
_valid_root_col: dict[torch.device, torch.Tensor] = {}


def _cached_arange(cache: dict, n: int, device: torch.device, dtype=torch.long) -> torch.Tensor:
    buf = cache.get(device)
    if buf is None or buf.numel() < n or buf.dtype != dtype:
        buf = torch.arange(n, device=device, dtype=dtype)
        cache[device] = buf
    return buf[:n]


def _cached_col(
    cache: dict,
    num_reqs: int,
    device: torch.device,
    dtype,
    fill_value,
) -> torch.Tensor:
    buf = cache.get(device)
    if buf is None or buf.shape[0] < num_reqs or buf.dtype != dtype:
        # These columns are constants. Captured graphs may retain older views,
        # so initialize once and never mutate shared storage during later calls.
        buf = torch.full((num_reqs, 1), fill_value, dtype=dtype, device=device)
        cache[device] = buf
    return buf[:num_reqs]


def iter_unique_kv_cache_tensors(kv_cache) -> list[torch.Tensor]:
    """Yield slot-major K/V tensors from a layer ``kv_cache`` binding."""
    if kv_cache is None:
        return []
    if isinstance(kv_cache, (tuple, list)):
        out: list[torch.Tensor] = []
        for item in kv_cache:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                out.append(item)
            elif isinstance(item, (tuple, list)):
                out.extend(iter_unique_kv_cache_tensors(item))
        return out
    if isinstance(kv_cache, torch.Tensor) and kv_cache.ndim >= 2:
        if kv_cache.shape[0] == 2 and kv_cache.ndim >= 3:
            return [kv_cache[0], kv_cache[1]]
        return [kv_cache]
    return []


def compact_accepted_query_rows(
    tensors: list[torch.Tensor],
    query_start_loc: torch.Tensor,
    path_node_ids: torch.Tensor,
    active: torch.Tensor,
    query_positions: torch.Tensor,
) -> None:
    """Move accepted-path query rows onto the linear prefix of each request.

    ``path_node_ids`` is [num_reqs, spec_len], with -1 for unused columns,
    and ``active`` is [num_reqs]. ``query_start_loc`` is [num_reqs + 1].
    Each tensor in ``tensors`` is token-major [num_tokens, ...] (hidden / aux).
    The root is query offset 0 and each draft node is at its node ID offset.
    Invalid path slots use negative scatter indices. Gather every view before
    scatter so overlapping rows and aliased hidden-state views stay correct.

    ``query_positions`` is ``[num_tokens]`` (device). Destination rows
    become ``root_pos + 0..k``. Rejected rows keep their packed positions, which
    may overlap accepted positions. The DFlash input preparation kernel pads
    the corresponding context slots to suppress draft KV writes.
    """
    num_reqs, spec_len = path_node_ids.shape
    if num_reqs == 0 or not tensors or tensors[0].shape[0] == 0:
        return
    node = path_node_ids.to(dtype=torch.long)
    qsl = query_start_loc[:num_reqs].to(dtype=torch.long)
    root = _cached_col(_root_col, num_reqs, node.device, torch.long, 0)
    src_off = torch.cat([root, node.clamp(min=0)], dim=1)
    dst_off = (
        _cached_arange(_dst_off_arange, spec_len + 1, node.device, dtype=torch.long).unsqueeze(0).expand(num_reqs, -1)
    )
    valid_root = _cached_col(_valid_root_col, num_reqs, node.device, torch.bool, True)
    valid = torch.cat([valid_root, node >= 0], dim=1)
    valid &= active.unsqueeze(1)
    src_off = torch.where(valid, src_off, dst_off)
    src_idx = qsl.unsqueeze(1) + src_off
    dst_idx = qsl.unsqueeze(1) + dst_off
    # Each request can have fewer scheduled rows near its output limit.
    n_tok = tensors[0].shape[0]
    last = n_tok - 1
    end = query_start_loc[1 : num_reqs + 1].to(torch.long).unsqueeze(1)
    in_bound = (
        (src_idx >= qsl.unsqueeze(1))
        & (src_idx < end)
        & (dst_idx >= qsl.unsqueeze(1))
        & (dst_idx < end)
        & (src_idx <= last)
        & (dst_idx <= last)
    )
    valid = valid & in_bound
    src_idx = src_idx.clamp(0, last).reshape(-1)
    # Valid destinations occupy disjoint request intervals. Padding cannot
    # overwrite another request's root or share a writable sentinel row.
    destination = torch.where(valid, dst_idx, -1).reshape(-1, 1)
    base = query_positions[qsl.clamp(min=0, max=last)].unsqueeze(1)
    positions = (base + dst_off.to(query_positions.dtype)).reshape(-1)
    relocate_tree_rows(
        [TreeRowMove(tensor, src_idx.unsqueeze(1), destination) for tensor in tensors],
        deduplicate=False,
    )
    torch_npu.npu_scatter_nd_update_(query_positions, destination, positions)
