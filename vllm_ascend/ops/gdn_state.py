# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


def canonicalize_folded_prefill_state(
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_tokens: int,
) -> None:
    """Commit the final folded prompt state to the canonical input location.

    GDN/KDA write one recurrent slot per speculative token, whereas conv
    history lives at token offset ``num_tokens - 1`` within the first block.
    Run outside the captured model graph, after every real forward/replay.
    Cloning the history also handles overlapping source/destination windows
    and non-contiguous DS layouts safely.
    """
    indices = state_indices.to(dtype=torch.int64)
    first = indices[:1]
    last = indices[num_tokens - 1 : num_tokens]
    # Device indices remain authoritative when asynchronous sequence-length
    # corrections differ from the CPU upper bound used during metadata build.
    final_state = recurrent_state.index_select(0, last)
    conv = conv_state.index_select(0, first)
    history_axis = 2 if is_conv_state_dim_first() else 1
    offset = num_tokens - 1
    history_len = conv.shape[history_axis] - offset
    if history_len <= 0:
        raise ValueError("Folded prefill exceeds the convolution state capacity")
    history = conv.narrow(history_axis, offset, history_len).clone()
    conv.narrow(history_axis, 0, history_len).copy_(history)
    recurrent_state.index_copy_(0, first, final_state)
    conv_state.index_copy_(0, first, conv)
