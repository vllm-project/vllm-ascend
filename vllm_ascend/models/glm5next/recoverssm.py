# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental synchronous RecoverSSM bridge using the existing KDA cache.

Each target layer owns one pending verification, consumed immediately after
sampling. Native cache slots and convolution history remain allocated as before;
mirroring the committed state preserves MRV1's next-step state-copy contract.
This bridge provides an executable prototype, not serving-cache memory savings.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_commit, kda_recoverssm_verify


def validate_recoverssm_config(config) -> None:
    """Fail at initialization for execution paths this prototype cannot handle."""
    if not config.model_config.enforce_eager or config.scheduler_config.async_scheduling:
        raise ValueError("Experimental KDA RecoverSSM requires eager, synchronous scheduling")
    if config.use_v2_model_runner:
        raise ValueError("Experimental KDA RecoverSSM requires Model Runner V1")
    if config.cache_config.enable_prefix_caching or config.kv_transfer_config is not None:
        raise ValueError("Experimental KDA RecoverSSM does not support prefix caching or KV transfer")
    parallel = config.parallel_config
    if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
        raise ValueError("Experimental KDA RecoverSSM requires PP=1 and DP=1")
    spec = config.speculative_config
    # Current vLLM normalizes model-specific aliases to "mtp" during config
    # validation; retain the legacy alias for builds that expose it unchanged.
    if spec is None or spec.method not in ("mtp", "deepseek_mtp") or spec.num_speculative_tokens < 1:
        raise ValueError("Experimental KDA RecoverSSM requires MTP with at least one speculative token")


@dataclass
class _PendingVerification:
    checkpoint: torch.Tensor
    correction: torch.Tensor
    kg: torch.Tensor
    a_log: torch.Tensor
    bias: torch.Tensor
    lower_bound: float
    starts: torch.Tensor
    indices: torch.Tensor
    native_state: torch.Tensor
    native_indices: torch.Tensor
    batch_rows: torch.Tensor


class KDARecoverSSMAdapter:
    """Layer-owned scratch records, valid from verify until the same step's sample."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.pending: _PendingVerification | None = None

    def verify(
        self,
        q,
        k,
        v,
        gate,
        beta,
        state,
        starts,
        native_indices,
        a_log,
        bias,
        lower_bound,
        previous_accepted,
        spec_sequence_masks,
    ):
        if self.pending is not None:
            raise RuntimeError("Previous KDA verification has not been committed")
        batch = starts.numel() - 1
        native_indices = native_indices[:batch]
        previous_accepted = previous_accepted[:batch]
        if native_indices.ndim != 2 or native_indices.shape[1] != self.capacity:
            raise ValueError("Native KDA slot width must match verification capacity")
        # Read the same checkpoint selected by the original AscendC recurrence.
        rows = torch.arange(batch, device=q.device)
        columns = (previous_accepted.long() - 1).clamp(0, self.capacity - 1)
        selected = native_indices[rows, columns].long().clamp_min(0)
        initial = state.index_select(0, selected)
        checkpoint = torch.cat([torch.zeros_like(initial[:1]), initial], dim=0).contiguous()
        heads, value_dim, key_dim = state.shape[1:]
        correction = torch.empty(batch + 1, heads, self.capacity, value_dim, device=q.device, dtype=torch.float32)
        kg = torch.empty(batch + 1, heads, self.capacity, 2 * key_dim, device=q.device, dtype=q.dtype)
        indices = torch.arange(1, batch + 1, device=q.device, dtype=torch.int32)
        indices = torch.where(native_indices[:, 0] > 0, indices, 0)
        starts = starts.contiguous()
        a_log, bias = a_log.reshape(-1).contiguous(), bias.contiguous()
        batch_rows = torch.nonzero(spec_sequence_masks, as_tuple=False).flatten()
        if batch_rows.numel() != batch:
            raise ValueError("Speculative request mask does not match packed KDA rows")
        output = kda_recoverssm_verify(
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            bias,
            lower_bound,
            checkpoint,
            correction,
            kg,
            starts,
            indices,
            self.capacity,
        )
        self.pending = _PendingVerification(
            checkpoint,
            correction,
            kg,
            a_log,
            bias,
            lower_bound,
            starts,
            indices,
            state,
            native_indices,
            batch_rows,
        )
        return output

    def commit(self, sampled_ids: torch.Tensor) -> None:
        entry = self.pending
        if entry is None:
            return
        if sampled_ids.ndim != 2:
            raise ValueError("Sampled IDs must have shape [requests, verified positions]")
        # RejectionSampler pads with -1; include the leading target token in counts.
        counts = (sampled_ids >= 0).sum(dim=-1).to(torch.int32)
        accepted = counts.index_select(0, entry.batch_rows).contiguous()
        kda_recoverssm_commit(
            entry.checkpoint,
            entry.correction,
            entry.kg,
            entry.a_log,
            entry.bias,
            entry.lower_bound,
            entry.starts,
            entry.indices,
            accepted,
        )
        # Preserve the old allocator and next-forward copy/index selection. All
        # native positions now contain the same logical committed checkpoint.
        width = entry.native_indices.shape[1]
        destinations = entry.native_indices.reshape(-1).long()
        source_rows = entry.indices.repeat_interleave(width).long()
        valid = destinations > 0
        entry.native_state.index_copy_(0, destinations[valid], entry.checkpoint.index_select(0, source_rows[valid]))
        self.pending = None
