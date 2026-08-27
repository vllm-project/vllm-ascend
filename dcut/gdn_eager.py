# SPDX-License-Identifier: Apache-2.0
"""Per-request conv1d fallback for variable query_len spec-decode."""
from __future__ import annotations

import torch
from torch.nn import functional as _F

# === GDN D-Cut monkeypatch additions ===
# Appended to monkeypatch.py. These changes were previously applied directly
# to /vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py in vllm_dcut.
# Moved here so vllm-ascend stays at vllm_src baseline.
#
# Changes:
# 1. _conv1d_spec_varlen_eager — per-request F.conv1d fallback for variable
#    query_len (D-Cut truncation on hybrid Mamba/GDN)
# 2. _patch_gdn_dcut — patches AscendGatedDeltaNetAttention._forward_core to:
#    a. Use _conv1d_spec_varlen_eager in the spec Conv1D eager path
#    b. Align ssm_state_indices with actual token positions (boolean mask)
#    c. Clamp num_accepted_tokens to actual seq lengths

from torch.nn import functional as _F


def _conv1d_spec_varlen_eager(
    output_spec,
    mixed_qkv_spec,
    conv_weights,
    conv_state,
    bias,
    activation,
    num_spec,
    spec_query_start_loc,
    spec_state_indices_tensor,
    num_accepted_tokens,
    num_spec_decodes,
):
    """Per-request conv1d for variable query_len spec-decode (D-Cut).

    When D-Cut truncates draft tokens, spec-decode requests have variable
    query_len.  The CANN operator npu_causal_conv1d_custom with run_mode=1
    requires uniform q_per_seq = num_spec + 1 and crashes on variable-length
    input.  This fallback processes each request independently using F.conv1d
    and updates the conv_state following the kernel's spec-decode state-update
    semantics (shift=1, offset from num_accepted_tokens).

    conv_state layout is SD: (num_cache_lines, state_len, dim).
    """
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    width = conv_weights.size(1)  # conv_kernel_size
    state_len = width - 1 + num_spec  # conv_kernel_size - 1 + num_spec
    dim = conv_weights.size(0)
    # Depthwise conv weight: (dim, 1, width)
    dw_weight = conv_weights.unsqueeze(1)

    spec_total = mixed_qkv_spec.size(0)
    out_total = output_spec.size(0)
    for b in range(num_spec_decodes):
        qs = int(spec_query_start_loc[b])
        qe = int(spec_query_start_loc[b + 1])
        # Clamp to actual tensor size — D-Cut RANDOM_CUT may trim tokens,
        # making spec_query_start_loc offsets exceed the tensor length.
        qe = min(qe, spec_total, out_total)
        ql = qe - qs
        ci = int(spec_state_indices_tensor[b, 0])
        nat_b = min(int(num_accepted_tokens[b]), ql)

        if ci == PAD_SLOT_ID or ql <= 0:
            continue

        offset = nat_b - 1  # conv_state_token_offset in kernel

        # --- Conv1d computation ---
        initial_state = conv_state[ci, offset:offset + width - 1, :].t()
        x_b = mixed_qkv_spec[qs:qe].t()
        x_concat = torch.cat([initial_state, x_b], dim=1)
        out = _F.conv1d(
            x_concat.unsqueeze(0),
            dw_weight,
            bias,
            padding=0,
            groups=dim,
        )

        if activation:
            out = _F.silu(out)

        output_spec[qs:qe] = out.squeeze(0).t()

        # --- Conv-state update (kernel spec-decode semantics) ---
        state_len_run = width - 2 + ql
        keep = state_len_run - ql  # = width - 2
        old_state = conv_state[ci, offset:offset + state_len_run, :].clone()
        if keep > 0:
            conv_state[ci, 0:keep, :] = old_state[1:1 + keep, :]
        conv_state[ci, keep:keep + ql, :] = x_b.t()
