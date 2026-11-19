# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-owned roles and routing workspace for L on single-device offload."""

from __future__ import annotations

import numpy as np
import torch

TOP_K = 6

UNRESTRICTED = 0
CONTRIBUTOR = 1
RESTRICTED = 2


def build_roles(
    *, query_start_loc, draft_counts, is_prefilling, padded_rows: int, protected_positions: int
) -> np.ndarray:
    roles = np.full(padded_rows, UNRESTRICTED, dtype=np.int32)
    if draft_counts is None:
        return roles
    for req, drafts in enumerate(draft_counts):
        if not drafts or is_prefilling[req]:
            continue
        begin, end = map(int, query_start_loc[req : req + 2])
        start = end - int(drafts) - 1
        if not 0 <= begin <= start < end <= padded_rows:
            raise ValueError("DSpark verification segment exceeds the scheduled query")
        split = min(start + protected_positions, end)
        roles[start:split] = CONTRIBUTOR
        roles[split:end] = RESTRICTED
    return roles


def verify_graph_eligible(batch_state, *, num_speculative_tokens: int, structured: bool) -> bool:
    rows = batch_state.num_scheduled_tokens
    drafts = batch_state.num_draft_tokens_np
    verify_width = num_speculative_tokens + 1
    return (
        not structured
        and drafts is not None
        and len(rows) == len(drafts) > 0
        and not any(batch_state.is_prefilling_np)
        and all(count == num_speculative_tokens for count in drafts)
        and all(count == verify_width for count in rows)
    )


def supports_router(layer) -> bool:
    from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter

    router = layer.router
    return (
        isinstance(router, AscendFusedTopKRouter)
        and router.scoring_func == "sqrtsoftplus"
        and router.top_k == TOP_K
        and router.num_logical_experts >= TOP_K
        and router.renormalize
        and router.tid2eid is None
        and router.bias_vl is None
        and router.custom_routing_function is None
        and router.num_expert_group in (None, 1)
        and router.topk_group in (None, 1)
        and layer.log2phy is None
        and not getattr(layer, "mix_placement", False)
    )


class DSparkExpertPool:
    def __init__(
        self,
        device: torch.device,
        capacity: int,
        protected_positions: int,
        suffix_experts: int,
        *,
        num_speculative_tokens: int,
        num_device_experts: int,
    ):
        self.protected_positions = protected_positions
        self.suffix_experts = suffix_experts
        self.num_speculative_tokens = num_speculative_tokens
        # Size only the reusable scratch. Graph admission remains with expert offload.
        self.workspace_rows = min(capacity, num_device_experts // TOP_K)
        self.roles = torch.empty(capacity, dtype=torch.int32, device=device)
        self.workspaces: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.active = False
        self.has_native_rows = False
        self.num_rows = 0

    def _prepare(self, batch, draft_counts, prefilling) -> None:
        self.num_rows = batch.num_tokens_after_padding
        if not batch.num_tokens <= self.num_rows <= self.roles.numel():
            raise ValueError("DSpark routing exceeds the token-role buffer")
        values = build_roles(
            query_start_loc=batch.query_start_loc_np[: batch.num_reqs + 1],
            draft_counts=draft_counts,
            is_prefilling=prefilling,
            padded_rows=self.num_rows,
            protected_positions=self.protected_positions,
        )
        self.active = bool(np.any(values == RESTRICTED))
        self.has_native_rows = self.active and bool(np.any(values[: batch.num_tokens] == UNRESTRICTED))
        if self.active:
            source = torch.from_numpy(values)
            # Stage each submission independently so a later step cannot
            # overwrite host memory while an asynchronous copy is pending.
            if self.roles.device.type == "npu":
                source = source.pin_memory()
            self.roles[: self.num_rows].copy_(source, non_blocking=True)

    def prepare(self, batch, *, invalid_drafts: bool = False) -> None:
        drafts = None if batch.has_structured_output_reqs or invalid_drafts else batch.num_draft_tokens_per_req
        self._prepare(batch, drafts, batch.is_prefilling_np)

    def prepare_capture(self, batch) -> None:
        boundaries = batch.query_start_loc_np[: batch.num_reqs + 1]
        verify_width = self.num_speculative_tokens + 1
        full_verify = batch.num_reqs > 0 and np.all(np.diff(boundaries) == verify_width)
        drafts = [self.num_speculative_tokens] * batch.num_reqs if full_verify else None
        self._prepare(batch, drafts, [False] * batch.num_reqs)

    def route(self, router, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_ascend.ops.triton.deepseek_expert_pool import route_experts

        rows, experts = logits.shape
        if rows != self.num_rows or experts != router.num_logical_experts or logits.dtype != torch.float32:
            raise ValueError("DSpark routing requires aligned FP32 logits and one row per prepared token role")
        if rows > self.workspace_rows:
            # Large mixed-prefill eager batches can have many shapes.
            raw_scores = torch.empty_like(logits, memory_format=torch.contiguous_format)
            votes = torch.empty(experts, dtype=torch.int32, device=logits.device)
        else:
            if experts not in self.workspaces:
                self.workspaces[experts] = (
                    torch.empty((self.workspace_rows, experts), dtype=logits.dtype, device=logits.device),
                    torch.empty(experts, dtype=torch.int32, device=logits.device),
                )
            raw_scores, votes = self.workspaces[experts]
            raw_scores = raw_scores[:rows]
        bias = router.e_score_correction_bias
        if bias is None:
            bias = torch.zeros(experts, dtype=logits.dtype, device=logits.device)
        return route_experts(
            logits,
            self.roles[:rows],
            bias.to(dtype=logits.dtype).contiguous(),
            raw_scores,
            votes,
            self.suffix_experts,
            router.routed_scaling_factor,
        )

    def merge_native(
        self,
        routed: tuple[torch.Tensor, torch.Tensor],
        native: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unrestricted = self.roles[: self.num_rows, None] == UNRESTRICTED
        weights, ids = routed
        native_weights, native_ids = native
        return (
            torch.where(unrestricted, native_weights, weights.to(native_weights.dtype)),
            torch.where(unrestricted, native_ids, ids.to(native_ids.dtype)),
        )
