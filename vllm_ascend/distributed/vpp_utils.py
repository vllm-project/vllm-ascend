#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Virtual Pipeline Parallelism (VPP) utilities.

Implements a V-shaped fold-back layer assignment topology:
- Even virtual stages flow forward  (rank 0 -> rank N-1)
- Odd  virtual stages flow backward (rank N-1 -> rank 0)

Example with pp_size=2, vp_size=2, 8 layers (2 per chunk):
  Rank 0: chunks [0, 3] -> layers [0,1] + [6,7]  (first + last)
  Rank 1: chunks [1, 2] -> layers [2,3] + [4,5]  (middle, fold point)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


def get_vpp_indices(
    num_hidden_layers: int,
    pp_rank: int,
    pp_size: int,
    vp_size: int,
) -> list[tuple[int, int]]:
    """Return layer ranges for each virtual stage on a given PP rank.

    Uses V-shaped fold-back assignment:
      even vp: chunk_idx = vp * pp_size + pp_rank        (forward)
      odd  vp: chunk_idx = (vp+1) * pp_size - 1 - pp_rank (backward)

    Supports uneven distribution: when ``num_hidden_layers`` is not
    divisible by ``pp_size * vp_size``, the first *remainder* chunks
    each receive one extra layer.

    Returns:
        List of (start_layer, end_layer) tuples, one per virtual stage.
        Layer indices are 0-based, end is exclusive.

    Examples:
        >>> get_vpp_indices(8, 0, 2, 2)
        [(0, 2), (6, 8)]
        >>> get_vpp_indices(8, 1, 2, 2)
        [(2, 4), (4, 6)]
        >>> get_vpp_indices(61, 0, 4, 2)
        [(0, 8), (54, 61)]
        >>> get_vpp_indices(61, 3, 4, 2)
        [(24, 32), (32, 39)]
    """
    total_chunks = pp_size * vp_size
    base = num_hidden_layers // total_chunks
    remainder = num_hidden_layers % total_chunks

    def _chunk_range(chunk_idx: int) -> tuple[int, int]:
        if chunk_idx < remainder:
            start = chunk_idx * (base + 1)
            end = start + base + 1
        else:
            start = remainder * (base + 1) + (chunk_idx - remainder) * base
            end = start + base
        return (start, end)

    ranges: list[tuple[int, int]] = []
    for vp in range(vp_size):
        if vp % 2 == 0:
            chunk_idx = vp * pp_size + pp_rank
        else:
            chunk_idx = (vp + 1) * pp_size - 1 - pp_rank
        ranges.append(_chunk_range(chunk_idx))
    return ranges


def validate_vpp_layer_ranges(
    layer_ranges: list[list[list[int]]],
    num_hidden_layers: int,
    pp_size: int,
    vp_size: int,
) -> list[list[tuple[int, int]]]:
    """Validate and normalize user-provided ``vpp_layer_ranges``.

    Args:
        layer_ranges: ``pp_size`` entries, each containing ``vp_size``
            ``[start, end)`` pairs.  Example for pp=4, vp=2, 61 layers::

                [
                    [[0, 8],  [53, 61]],   # rank 0
                    [[8, 16], [45, 53]],    # rank 1
                    [[16, 24],[37, 45]],    # rank 2
                    [[24, 37]],             # rank 3 (fold-point, may merge)
                ]

        num_hidden_layers: total layers in the model.
        pp_size: pipeline parallel size.
        vp_size: virtual pipeline parallel size.

    Returns:
        Normalized list of ``list[tuple[int, int]]`` per rank.

    Raises:
        ValueError on any validation failure.
    """
    if len(layer_ranges) != pp_size:
        raise ValueError(
            f"vpp_layer_ranges must have {pp_size} entries (one per PP rank), "
            f"got {len(layer_ranges)}.")

    normalized: list[list[tuple[int, int]]] = []
    all_indices: set[int] = set()

    for rank, rank_ranges in enumerate(layer_ranges):
        if len(rank_ranges) != vp_size:
            raise ValueError(
                f"Rank {rank}: expected {vp_size} layer ranges "
                f"(one per virtual stage), got {len(rank_ranges)}.")
        rank_tuples: list[tuple[int, int]] = []
        for stage_idx, pair in enumerate(rank_ranges):
            if len(pair) != 2:
                raise ValueError(
                    f"Rank {rank}, stage {stage_idx}: expected [start, end], "
                    f"got {pair}.")
            start, end = int(pair[0]), int(pair[1])
            if start < 0 or end > num_hidden_layers or start >= end:
                raise ValueError(
                    f"Rank {rank}, stage {stage_idx}: invalid range "
                    f"[{start}, {end}) for {num_hidden_layers} layers.")
            for i in range(start, end):
                if i in all_indices:
                    raise ValueError(
                        f"Layer {i} is assigned to multiple ranks/stages.")
                all_indices.add(i)
            rank_tuples.append((start, end))
        normalized.append(rank_tuples)

    if all_indices != set(range(num_hidden_layers)):
        missing = set(range(num_hidden_layers)) - all_indices
        raise ValueError(
            f"Not all layers are covered.  Missing layers: "
            f"{sorted(missing)}")

    return normalized


def is_vpp_first_stage(pp_rank: int, vp_stage: int) -> bool:
    """Whether this is the very first stage in the VPP pipeline."""
    return vp_stage == 0 and pp_rank == 0


def is_vpp_last_stage(
    pp_rank: int,
    pp_size: int,
    vp_stage: int,
    vp_size: int,
) -> bool:
    """Whether this is the very last stage in the VPP pipeline.

    For even vp_size the last sweep is backward, ending at rank 0.
    For odd  vp_size the last sweep is forward,  ending at rank pp_size-1.
    """
    if vp_stage != vp_size - 1:
        return False
    if vp_size % 2 == 0:
        return pp_rank == 0
    else:
        return pp_rank == pp_size - 1


def is_vpp_fold_point(
    pp_rank: int,
    pp_size: int,
    vp_stage: int,
    vp_size: int,
) -> bool:
    """Whether this rank is a fold point after the given virtual stage.

    At a fold point the same GPU continues to the next virtual stage
    without any inter-rank communication.
    """
    if vp_stage >= vp_size - 1:
        return False
    is_forward = (vp_stage % 2 == 0)
    if is_forward:
        return pp_rank == pp_size - 1
    else:
        return pp_rank == 0


@dataclass
class VPPCommInfo:
    need_recv: bool
    recv_src: int
    need_send: bool
    send_dst: int


def get_vpp_comm_info(
    pp_rank: int,
    pp_size: int,
    vp_stage: int,
    vp_size: int,
) -> VPPCommInfo:
    """Compute send/recv info for a given rank and virtual stage.

    Returns a VPPCommInfo with:
      - need_recv / recv_src: whether to recv and from which PP rank
      - need_send / send_dst: whether to send and to which PP rank
    """
    is_forward = (vp_stage % 2 == 0)

    # --- Recv logic ---
    is_first = is_vpp_first_stage(pp_rank, vp_stage)
    is_fold_from_prev = False
    if vp_stage > 0:
        prev_forward = ((vp_stage - 1) % 2 == 0)
        if prev_forward and pp_rank == pp_size - 1:
            is_fold_from_prev = True
        elif not prev_forward and pp_rank == 0:
            is_fold_from_prev = True

    need_recv = not is_first and not is_fold_from_prev
    recv_src = (pp_rank - 1) if is_forward else (pp_rank + 1)

    # --- Send logic ---
    is_last = is_vpp_last_stage(pp_rank, pp_size, vp_stage, vp_size)
    is_fold_to_next = is_vpp_fold_point(pp_rank, pp_size, vp_stage, vp_size)
    need_send = not is_last and not is_fold_to_next
    send_dst = (pp_rank + 1) if is_forward else (pp_rank - 1)

    return VPPCommInfo(
        need_recv=need_recv,
        recv_src=recv_src,
        need_send=need_send,
        send_dst=send_dst,
    )


# ---------------------------------------------------------------------------
# Model-level VPP utilities
# ---------------------------------------------------------------------------

LayerFn = Callable[..., "nn.Module"]


def make_vpp_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
    vp_size: int,
    custom_layer_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], "nn.ModuleList"]:
    """Build a ModuleList with VPP layer assignment.

    Layers not owned by this rank are replaced with ``PPMissingLayer``.

    Args:
        custom_layer_ranges: If provided, used directly instead of
            auto-computing via ``get_vpp_indices``.  Must contain
            ``vp_size`` entries of ``(start, end)`` for this rank.

    Returns:
        (layer_ranges, modules) where *layer_ranges* is a list of
        ``(start, end)`` for each virtual stage on this rank.
    """
    import torch
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm.model_executor.offloader import get_offloader

    if custom_layer_ranges is not None:
        layer_ranges = custom_layer_ranges
    else:
        pp_rank = get_pp_group().rank_in_group
        pp_size = get_pp_group().world_size
        layer_ranges = get_vpp_indices(
            num_hidden_layers, pp_rank, pp_size, vp_size
        )

    local_indices: set[int] = set()
    for start, end in layer_ranges:
        local_indices.update(range(start, end))

    modules = torch.nn.ModuleList(
        get_offloader().wrap_modules(
            layer_fn(prefix=f"{prefix}.{i}") if i in local_indices
            else PPMissingLayer()
            for i in range(num_hidden_layers)
        )
    )
    return layer_ranges, modules


def get_vp_size() -> int:
    """Return the configured virtual pipeline parallel size (1 if unset)."""
    from vllm_ascend.ascend_config import get_ascend_config
    try:
        return get_ascend_config().virtual_pipeline_parallel_size
    except RuntimeError:
        return 1


def get_custom_layer_ranges_for_rank() -> list[tuple[int, int]] | None:
    """Return manually specified VPP layer ranges for the current PP rank, or None."""
    from vllm_ascend.ascend_config import get_ascend_config
    try:
        all_ranges = get_ascend_config().vpp_layer_ranges
    except RuntimeError:
        return None
    if all_ranges is None:
        return None
    from vllm.distributed import get_pp_group
    return all_ranges[get_pp_group().rank_in_group]


def setup_vpp_layers(
    self,
    num_hidden_layers: int,
    layer_fn: "LayerFn",
    prefix: str,
) -> None:
    """Attach VPP-aware layer attributes to ``self`` (model instance).

    Sets:
        * ``self.vpp_layer_ranges`` — list of ``(start, end)`` per vp_stage
        * ``self.layers``           — ``nn.ModuleList`` for this pp_rank
        * ``self.start_layer`` / ``self.end_layer`` — boundaries of the
          first/last stage on this rank (for compatibility with non-VPP
          code paths that read these attributes).

    Reads the VPP size and custom layer ranges from ``get_ascend_config()``.
    No-op (besides reading config) when ``vp_size <= 1`` is not possible
    here — callers must check ``vp_size`` themselves if they want to
    fall back to the upstream ``make_layers`` path.
    """
    custom_ranges = get_custom_layer_ranges_for_rank()
    self.vpp_layer_ranges, self.layers = make_vpp_layers(
        num_hidden_layers,
        layer_fn,
        prefix,
        get_vp_size(),
        custom_layer_ranges=custom_ranges,
    )
    self.start_layer = self.vpp_layer_ranges[0][0]
    self.end_layer = self.vpp_layer_ranges[-1][1]
