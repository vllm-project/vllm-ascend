# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Graph-owned proposal inputs for the MRV2 Draft Window read plane.

This module joins real proposal inputs to capture-owned Window buffers and
the normal Ascend metadata builder. Positions, query layout, slot mappings
and Target metadata remain owned by the existing runner.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.draft_window_graph import (
    DraftWindowGraphBindingProvider,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


class DraftWindowGraphInputMixin:
    """Proposal input ownership for DFlash, DFlash2 and eligible DSpark Graph."""

    _speculator_name: str
    _group_causal: bool | Mapping[int, bool]
    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _draft_window_dp_dummy_active: bool
    input_batch: InputBatch
    input_buffers: Any
    block_tables: Any
    vllm_config: Any
    draft_attn_layer_names: set[str] | None
    attn_groups: Any
    kv_cache_config: Any
    num_query_per_req: int

    @contextmanager
    def _draft_window_proposal_scope(
        self,
        input_batch: InputBatch,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> Iterator[None]:
        provider = getattr(self, "_draft_window_graph_binding_provider", None)
        if provider is None:
            yield
            return
        rows = (
            ()
            if dummy_run or is_profile
            else tuple(map(bool, continuing_prefill_rows(input_batch, input_batch.num_reqs)))
        )
        with provider.proposal_scope(rows):
            # Empty rows identify startup/profile/idle; only a real idle DP
            # replica must replay the canonical dummy communication Graph.
            self._draft_window_dp_dummy_active = (
                dummy_run and not is_profile and getattr(self, "dp_size", 1) > 1 and bool(provider._carriers)
            )
            try:
                yield
            finally:
                self._draft_window_dp_dummy_active = False

    def _build_uniform_attn_metadata(
        self,
        batch_desc: Any,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        provider = getattr(self, "_draft_window_graph_binding_provider", None)
        rows = None if provider is None else provider.proposal
        if provider is not None and rows is None:
            raise RuntimeError("Draft Window metadata build has no active proposal")
        if not rows or all(rows):
            return super()._build_uniform_attn_metadata(  # type: ignore[misc]
                batch_desc,
                num_reqs,
                num_query_per_req,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
            )
        if not self.draft_attn_layer_names:
            return None
        padded = batch_desc.num_reqs or num_reqs
        tokens = batch_desc.num_tokens
        if (
            num_query_per_req != self.num_query_per_req
            or step != self.num_query_per_req
            or batch_desc.cg_mode.name != "FULL"
            or tokens != padded * self.num_query_per_req
            or (getattr(self, "_draft_graph_review_limit", None) is not None and padded != num_reqs)
        ):
            raise RuntimeError("Draft Window requires its actual uniform FULL dispatch; no eager fallback")
        assert provider is not None
        input_batch = self.input_batch
        if num_reqs != input_batch.num_reqs or num_reqs != len(rows):
            raise RuntimeError("metadata request count does not match proposal epoch")
        if padded < num_reqs:
            raise ValueError("padded request count must be at least actual requests")
        selected = provider.capture_identity_for(descriptor=batch_desc)
        # The private bank initializes padded rows; snapshot only actual requests.
        cpu_full_exact = self.input_buffers.seq_lens[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
        provider.prepare(selected, cpu_full_exact=cpu_full_exact, idx_mapping_np=input_batch.idx_mapping_np)
        # The fixed manager binds this bank's FIA parameters immediately before replay.
        return None
