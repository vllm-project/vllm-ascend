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

"""310P GDN recurrent operator wrappers."""

import torch

from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p
from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def _flatten_state_indices(
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    if ssm_state_indices.ndim == 1:
        return ssm_state_indices[:total_tokens].to(torch.int32).contiguous()

    num_seqs = (cu_seqlens[1:] - cu_seqlens[:-1]).shape[0]
    q_per_seq = ssm_state_indices.shape[1]

    # Uniform spec-decode ACL graph uses fixed q_len per request; reshape avoids
    # NPU masked_select and seq_lens scalar reads which break stream capture.
    if _EXTRA_CTX.capturing or total_tokens == num_seqs * q_per_seq:
        flat = ssm_state_indices[:num_seqs, :q_per_seq].reshape(-1)
        return flat[:total_tokens].to(torch.int32).contiguous()

    seq_lens = cu_seqlens[1 : num_seqs + 1] - cu_seqlens[:num_seqs]
    ssm_state_indices = ssm_state_indices[:num_seqs]

    # Eager mixed batches with variable seq_lens: compact on CPU, copy back async.
    ssm_cpu = ssm_state_indices.cpu()
    seq_lens_cpu = seq_lens.cpu()
    q_per_seq = ssm_cpu.shape[1]
    positions = torch.arange(q_per_seq)
    valid = positions.unsqueeze(0) < seq_lens_cpu.unsqueeze(1)
    flat_cpu = ssm_cpu.masked_select(valid).to(torch.int32).contiguous()[:total_tokens]
    if not flat_cpu.is_pinned:
        flat_cpu = flat_cpu.pin_memory()
    flat_dev = torch.empty(flat_cpu.numel(), dtype=torch.int32, device=ssm_state_indices.device)
    flat_dev.copy_(flat_cpu, non_blocking=True)
    return flat_dev.contiguous()


def _mask_padded_recurrent_accepted_tokens(
    num_accepted_tokens: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
) -> torch.Tensor:
    accepted_tokens = num_accepted_tokens[: actual_seq_lengths.shape[0]].to(torch.int32).contiguous()
    return torch.where(
        actual_seq_lengths > 0,
        accepted_tokens,
        torch.zeros_like(accepted_tokens),
    ).contiguous()


def npu_recurrent_gated_delta_rule_310(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if use_qk_l2norm_in_kernel:
        q = l2norm_310p(q)
        k = l2norm_310p(k)

    total_tokens = v.shape[1]
    flat_state_indices = _flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)
    actual_seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32).contiguous()
    # Do not clamp PAD_SLOT_ID (-1) to 0 — that would write mamba block 0.
    # Callers must slice away padded requests before invoking this helper.
    accepted_tokens = None
    if num_accepted_tokens is not None:
        accepted_tokens = _mask_padded_recurrent_accepted_tokens(
            num_accepted_tokens,
            actual_seq_lengths,
        )

    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=q.squeeze(0).to(torch.float16).contiguous(),
        key=k.squeeze(0).to(torch.float16).contiguous(),
        value=v.squeeze(0).to(torch.float16).contiguous(),
        g=None if g is None else g.squeeze(0).to(torch.float32).contiguous(),
        gk=None,
        beta=beta.squeeze(0).to(torch.float16).contiguous(),
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=flat_state_indices,
        num_accepted_tokens=accepted_tokens,
        scale_value=k.shape[-1] ** -0.5,
    ).unsqueeze(0)
    return out
