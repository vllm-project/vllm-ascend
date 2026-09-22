# SPDX-License-Identifier: Apache-2.0
"""Precompiled gather/clear and scatter for KDA's layer-strided state cache."""

import torch

from vllm_ascend.utils import is_950


def supports_kda_state_copy(state: torch.Tensor) -> bool:
    if (
        state.device.type != "npu"
        or not is_950()
        or not hasattr(torch.ops._C_ascend, "kda_state_copy")
        or state.dtype not in (torch.float32, torch.bfloat16)
        or state.ndim != 4
        or state.shape[0] == 0
    ):
        return False
    payload = 1
    for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
        if size <= 0 or (size > 1 and stride != payload):
            return False
        payload *= size
    return state.stride(0) >= payload


def copy_kda_states(state, packed_states, indices, *, to_cache=False, has_initial_state=None):
    """Copy only selected dense [H,V,K] payloads, preserving cache page gaps.

    Gather writes zero for invalid indices and false initial-state flags.
    Scatter skips invalid indices; valid scatter indices must be unique, as
    with the preceding parallel byte-copy implementation. Index arithmetic
    stays on the device and uses 64-bit byte offsets.
    """
    if has_initial_state is not None:
        # Match clear_ssm_states' accepted metadata. The normal builder already
        # supplies same-device BOOL vectors, requiring no conversion or copy.
        if has_initial_state.device != state.device or has_initial_state.dtype != torch.bool:
            has_initial_state = has_initial_state.to(device=state.device, dtype=torch.bool, non_blocking=True)
        if has_initial_state.ndim != 1:
            has_initial_state = has_initial_state.reshape(-1)
    torch.ops._C_ascend.kda_state_copy(state, packed_states, indices, has_initial_state, to_cache)
