# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small state-layout calculations shared by direct and pooled P/D."""

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


def mamba_conv_projection_widths(
    mamba_type: MambaAttentionBackendEnum,
    conv_width: int,
    temporal_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return projection widths in the existing V2 convolution storage order.

    Mamba2 stores x/b/c; GDN (including GLM KDA) stores q/k/v. Splitting the
    entire merged channel axis as one TP shard would mix those projections.
    Temporal rows, byte strides, checkpoint selection and I/O are caller-owned.
    """
    if mamba_type == MambaAttentionBackendEnum.MAMBA1:
        return (conv_width,)
    if mamba_type not in (MambaAttentionBackendEnum.MAMBA2, MambaAttentionBackendEnum.GDN_ATTN):
        raise NotImplementedError(f"Mooncake unequal-TP conv transfer does not support Mamba type {mamba_type!r}")
    if temporal_shape is None or len(temporal_shape) < 2:
        raise ValueError(f"Mooncake {mamba_type.name} conv transfer requires a temporal state shape")
    state_width = temporal_shape[0] * temporal_shape[1]
    remainder = conv_width - state_width
    if remainder <= 0 or remainder % 2:
        raise ValueError(
            f"Mooncake {mamba_type.name} conv shape cannot be decomposed: "
            f"conv_width={conv_width}, temporal_shape={temporal_shape}"
        )
    if mamba_type == MambaAttentionBackendEnum.MAMBA2:
        return state_width, remainder // 2, remainder // 2
    return remainder // 2, remainder // 2, state_width


def mamba_component_copy_slices(
    *,
    mamba_type: MambaAttentionBackendEnum,
    dtype_size: int,
    component: int,
    local_shapes: list[tuple[int, ...]],
    remote_shapes: list[tuple[int, ...]],
    local_block_len: int,
    remote_block_len: int,
    local_size: int,
    remote_size: int,
    rank_offset: int,
    tp_replicated: bool,
) -> list[tuple[int, int, int]]:
    """V2's component address calculation, without transport or base pointers.

    Returns (local offset, remote offset, bytes). Preserve V2's MTP/layout
    compatibility checks: this does not convert speculative workspaces.
    """
    local_shape, remote_shape = local_shapes[component], remote_shapes[component]
    tp_ratio = max(local_size, remote_size) // min(local_size, remote_size)
    if tp_replicated:
        if local_shape != remote_shape or local_block_len != remote_block_len:
            raise ValueError("Mooncake replicated Mamba metadata mismatch")
        return [(0, 0, local_block_len)]
    if local_size == remote_size:
        if local_block_len != remote_block_len:
            raise ValueError("Mooncake Mamba equal-TP metadata mismatch")
        return [(0, 0, local_block_len)]
    if component == 0 and len(local_shape) == len(remote_shape) == 2:
        if local_shape[0] != remote_shape[0]:
            raise ValueError(f"Mooncake Mamba conv row mismatch: local={local_shape}, remote={remote_shape}")
        local_widths = mamba_conv_projection_widths(
            mamba_type, local_shape[1], local_shapes[1] if len(local_shapes) > 1 else None
        )
        remote_widths = mamba_conv_projection_widths(
            mamba_type, remote_shape[1], remote_shapes[1] if len(remote_shapes) > 1 else None
        )
        slices = []
        local_offset = remote_offset = 0
        for local_width, remote_width in zip(local_widths, remote_widths, strict=True):
            if remote_size > local_size:
                if local_width != remote_width * tp_ratio:
                    raise ValueError("Mooncake Mamba conv projection TP ratio mismatch")
                local_start, remote_start, width = (
                    local_offset + rank_offset * remote_width,
                    remote_offset,
                    remote_width,
                )
            else:
                if remote_width != local_width * tp_ratio:
                    raise ValueError("Mooncake Mamba conv projection TP ratio mismatch")
                local_start, remote_start, width = local_offset, remote_offset + rank_offset * local_width, local_width
            for row in range(local_shape[0]):
                slices.append(
                    (
                        (row * local_shape[1] + local_start) * dtype_size,
                        (row * remote_shape[1] + remote_start) * dtype_size,
                        width * dtype_size,
                    )
                )
            local_offset += local_width
            remote_offset += remote_width
        return slices
    if remote_size > local_size:
        if local_block_len != remote_block_len * tp_ratio:
            raise ValueError("Mooncake Mamba state TP ratio mismatch")
        return [(rank_offset * remote_block_len, 0, remote_block_len)]
    if remote_block_len != local_block_len * tp_ratio:
        raise ValueError("Mooncake Mamba state TP ratio mismatch")
    return [(0, rank_offset * local_block_len, local_block_len)]
