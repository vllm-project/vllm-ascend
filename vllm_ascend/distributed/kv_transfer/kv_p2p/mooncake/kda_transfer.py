# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""GLM KDA prefill state geometry, independent of the transfer engine.

Only committed conv/recurrent state crosses the prefill boundary. RecoverSSM
correction and key/gate records are written by the next D-side verify before
its commit, and are deliberately excluded from this transfer.
"""

from dataclasses import dataclass
from math import prod

from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import MambaConvSplitInfo


@dataclass(frozen=True)
class MooncakeKDAStateMetadata:
    """Versioned GLM-only contract advertised by each cache owner."""

    head_shards: int
    pcp_head_sharded: bool
    conv_layout: str
    dtypes: tuple[str, ...]
    num_speculative_tokens: int
    recoverssm: bool
    cache_mode: str
    version: int = 1

    def head_rank(self, tp_rank: int, pcp_rank: int, tp_size: int, pcp_size: int) -> int:
        expected = pcp_size if tp_size == 1 else tp_size
        if (
            self.version != 1
            or self.head_shards != expected
            or self.pcp_head_sharded != (tp_size == 1 and pcp_size > 1)
            or not 0 <= tp_rank < tp_size
            or not 0 <= pcp_rank < pcp_size
        ):
            raise ValueError("Mooncake GLM KDA head-shard metadata does not match TP/PCP topology")
        return pcp_rank if self.pcp_head_sharded else tp_rank


def kda_remote_rank_groups(
    local: MooncakeKDAStateMetadata,
    remote: MooncakeKDAStateMetadata,
    *,
    tp_rank: int,
    pcp_rank: int,
    tp_size: int,
    pcp_size: int,
    remote_tp_size: int,
    remote_pcp_size: int,
) -> list[list[int]]:
    """Return head owners and interchangeable PCP replicas, in worker IDs."""
    local_rank = local.head_rank(tp_rank, pcp_rank, tp_size, pcp_size)
    remote.head_rank(0, 0, remote_tp_size, remote_pcp_size)
    larger, smaller = max(local.head_shards, remote.head_shards), min(local.head_shards, remote.head_shards)
    if smaller <= 0 or larger % smaller:
        raise ValueError("Mooncake GLM KDA head-shard counts must have an integer ratio")
    groups: dict[int, list[int]] = {}
    for remote_tp_rank in range(remote_tp_size):
        for remote_pcp_rank in range(remote_pcp_size):
            rank = remote.head_rank(remote_tp_rank, remote_pcp_rank, remote_tp_size, remote_pcp_size)
            if (
                rank * local.head_shards < (local_rank + 1) * remote.head_shards
                and local_rank * remote.head_shards < (rank + 1) * local.head_shards
            ):
                groups.setdefault(rank, []).append(remote_tp_rank * remote_pcp_size + remote_pcp_rank)
    return [groups[rank] for rank in sorted(groups)]


def kda_transfer_slices(
    local: MooncakeKDAStateMetadata,
    remote: MooncakeKDAStateMetadata,
    local_shapes: list[tuple[int, ...]],
    remote_shapes: list[tuple[int, ...]],
    local_lens: list[int],
    remote_lens: list[int],
    local_strides: list[int],
    remote_strides: list[int],
    *,
    local_head_rank: int,
    remote_head_rank: int,
) -> list[tuple[int, int, int, int]]:
    """Return (cache index, local offset, remote offset, bytes) for one peer.

    DS uses upstream NIXL's three-sub-projection slicing scheme: each Q/K/V
    channel interval includes all of its history contiguously. SD retains
    row-wise slicing. No packing buffers or NPU kernels are added here.
    """
    for metadata, shapes, lengths, strides in (
        (local, local_shapes, local_lens, local_strides),
        (remote, remote_shapes, remote_lens, remote_strides),
    ):
        count = 4 if metadata.recoverssm else 2
        if (
            metadata.version != 1
            or metadata.conv_layout not in ("DS", "SD")
            or not (len(shapes) == len(lengths) == len(strides) == len(metadata.dtypes) == count)
            or any(any(dim <= 0 for dim in shape) for shape in shapes)
            or any(length <= 0 or stride < length for length, stride in zip(lengths, strides))
        ):
            raise ValueError("Mooncake GLM KDA cache metadata is invalid")
    if (
        local.conv_layout != remote.conv_layout
        or local.dtypes[:2] != remote.dtypes[:2]
        or local.num_speculative_tokens != remote.num_speculative_tokens
        or local.cache_mode != remote.cache_mode
    ):
        raise ValueError(
            "Mooncake GLM KDA requires matching conv layout, base dtypes, speculative window and cache mode"
        )
    # This first version supports native -> RecoverSSM and equal execution
    # modes. Reverse handoff would need native speculative-slot preparation.
    if remote.recoverssm and not local.recoverssm:
        raise ValueError("Mooncake GLM KDA RecoverSSM producer to native consumer is not supported")

    local_state, remote_state = local_shapes[1], remote_shapes[1]
    if (
        len(local_shapes[0]) != 2
        or len(remote_shapes[0]) != 2
        or len(local_state) != 3
        or len(remote_state) != 3
        or local_state[1:] != remote_state[1:]
        or local_state[1] != local_state[2]
        or local_state[0] * local.head_shards != remote_state[0] * remote.head_shards
    ):
        raise ValueError("Mooncake GLM KDA state shapes do not describe the same global heads")
    total_heads = local_state[0] * local.head_shards
    if (
        not 0 <= local_head_rank < local.head_shards
        or not 0 <= remote_head_rank < remote.head_shards
        or max(local.head_shards, remote.head_shards) % min(local.head_shards, remote.head_shards)
    ):
        raise ValueError("Mooncake GLM KDA invalid head interval")
    start = max(local_head_rank * local_state[0], remote_head_rank * remote_state[0])
    end = min((local_head_rank + 1) * local_state[0], (remote_head_rank + 1) * remote_state[0])
    if not 0 <= start < end <= total_heads:
        raise ValueError("Mooncake GLM KDA peer head intervals do not overlap")
    local_offset = start - local_head_rank * local_state[0]
    remote_offset = start - remote_head_rank * remote_state[0]
    heads = end - start
    sizes: list[list[int]] = []
    for shapes, lengths in ((local_shapes, local_lens), (remote_shapes, remote_lens)):
        if any(lengths[i] % prod(shapes[i]) for i in range(2)):
            raise ValueError("Mooncake GLM KDA cache byte size does not match its shape")
        sizes.append([lengths[i] // prod(shapes[i]) for i in range(2)])
    if sizes[0] != sizes[1]:
        raise ValueError("Mooncake GLM KDA element sizes differ")
    conv_bytes, state_bytes = sizes[0]

    history_axis = 1 if local.conv_layout == "DS" else 0
    channel_axis = 1 - history_axis
    history = local_shapes[0][history_axis]
    if (
        history != remote_shapes[0][history_axis]
        or local_shapes[0][channel_axis] != 3 * local_state[0] * local_state[1]
        or remote_shapes[0][channel_axis] != 3 * remote_state[0] * remote_state[1]
    ):
        raise ValueError("Mooncake GLM KDA convolution history or Q/K/V dimensions differ")
    slices: list[tuple[int, int, int, int]] = []
    if local.conv_layout == "DS":
        # Reuse the upstream NIXL sub-projection geometry. Unlike the generic
        # TP path, GLM's shard rank may belong to PCP or to a TP replica group.
        local_conv = MambaConvSplitInfo(
            conv_rows=history,
            local_proj_dims=(local_state[0] * local_state[1],) * 3,
            conv_dtype_size=conv_bytes,
            ssm_sizes=tuple(local_lens[:2]),
        )
        remote_conv = MambaConvSplitInfo(
            conv_rows=history,
            local_proj_dims=(remote_state[0] * remote_state[1],) * 3,
            conv_dtype_size=conv_bytes,
            ssm_sizes=tuple(remote_lens[:2]),
        )
        for (local_base, _), (remote_base, _) in zip(local_conv.local_conv_offsets, remote_conv.local_conv_offsets):
            slices.append(
                (
                    0,
                    local_base + local_offset * local_state[1] * history * conv_bytes,
                    remote_base + remote_offset * remote_state[1] * history * conv_bytes,
                    heads * local_state[1] * history * conv_bytes,
                )
            )
        head_bytes = prod(local_state[1:]) * state_bytes
        slices.append((1, local_offset * head_bytes, remote_offset * head_bytes, heads * head_bytes))
        return slices
    for projection in range(3):
        local_channel = (projection * local_state[0] + local_offset) * local_state[1]
        remote_channel = (projection * remote_state[0] + remote_offset) * remote_state[1]
        width = heads * local_state[1]
        for row in range(history):
            slices.append(
                (
                    0,
                    (row * local_shapes[0][1] + local_channel) * conv_bytes,
                    (row * remote_shapes[0][1] + remote_channel) * conv_bytes,
                    width * conv_bytes,
                )
            )
    head_bytes = prod(local_state[1:]) * state_bytes
    slices.append((1, local_offset * head_bytes, remote_offset * head_bytes, heads * head_bytes))
    return slices
