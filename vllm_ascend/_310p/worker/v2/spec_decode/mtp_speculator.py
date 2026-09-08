# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P MTP speculator: CPU block-table slot mappings + RoPE flag + draft quant."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend._310p.worker.v2.spec_decode.aclgraph import AutoRegressiveAclGraphManager310
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendMTPSpeculator310(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator for 310P MRv2 (Triton-free draft loop)."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_model_config = self.speculative_config.draft_model_config
        if draft_model_config.hf_overrides is None:
            draft_model_config.hf_overrides = {}

        # Keep PP=1 for draft execution (same as AscendAutoRegressiveSpeculator).
        parallel_config = replace(
            self.vllm_config.parallel_config,
            pipeline_parallel_size=1,
        )
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=draft_model_config,
            parallel_config=parallel_config,
        )

        target_path = os.path.realpath(self.vllm_config.model_config.model)
        draft_path = os.path.realpath(draft_model_config.model)
        if target_path == draft_path and self.vllm_config.quant_config is not None:
            draft_vllm_config = replace(
                draft_vllm_config,
                quant_config=self.vllm_config.quant_config,
            )
        return draft_vllm_config

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.draft_vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = spec_config.draft_model_config.hf_config if spec_config is not None else None
        self.share_mtp_topk_indices = (
            getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Use 310P draft graph managers (seq_lens refresh; no FIA update)."""
        super().init_cudagraph_manager(cudagraph_mode)
        assert self.prefill_cudagraph_manager is not None
        # Rebind class so run_fullgraph uses 310P refresh path; keep instance state.
        self.prefill_cudagraph_manager.__class__ = AutoRegressiveAclGraphManager310
        if self.decode_cudagraph_manager is not None:
            self.decode_cudagraph_manager.__class__ = AutoRegressiveAclGraphManager310
        self.prefill_cudagraph_manager.speculator = self
        self.prefill_cudagraph_manager.update_stream = self.update_stream
        if self.decode_cudagraph_manager is not None:
            self.decode_cudagraph_manager.speculator = self
            self.decode_cudagraph_manager.update_stream = self.update_stream

    def _compute_draft_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> dict[str, torch.Tensor]:
        idx_mapping_np = idx_mapping.detach().cpu().numpy()
        query_start_loc_np = query_start_loc.detach().cpu().numpy()
        positions_np = positions.detach().cpu().numpy()
        # 310P BlockTables path accepts host ndarrays (CPU slot-map builder).
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping_np,  # type: ignore[arg-type]
            query_start_loc_np,  # type: ignore[arg-type]
            positions_np,  # type: ignore[arg-type]
            num_tokens_padded=num_tokens_padded,
        )
        return build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)

    @contextmanager
    def _rope_position_flag_310p(self):
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            yield
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)

    def capture(self) -> None:
        """Skip draft ACLGraph on 310P MTP — eager draft for correctness.

        Draft-decode graphs are illegal (host D2H / slot-map between steps).
        Draft-prefill FULL also garbles under concurrent multi-req batches
        (seen as acceptance collapse + Chinese/ASCII junk mid-answer). Keep
        draft eager until SpecDecoding pad/replay is proven for batch>1.
        """
        self.last_token_indices.zero_()
        logger.info(
            "Skipping draft ACLGraph capture on 310P MTP (eager draft-prefill/decode for concurrent correctness)."
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._rope_position_flag_310p():
            return super()._run_model(
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
            )

    def _calc_next_seq_lens_cpu(self, seq_lens_cpu, num_reqs, num_reqs_padded, step):
        """Match prepare_decode_inputs: seq = target - rejected + step."""
        next_seqs_cpu = seq_lens_cpu[:num_reqs_padded].clone()
        rejected = getattr(self, "_last_num_rejected_cpu", None)
        if rejected is not None and rejected.numel() >= num_reqs:
            next_seqs_cpu[:num_reqs] = next_seqs_cpu[:num_reqs] - rejected[:num_reqs].to(next_seqs_cpu.dtype)
        next_seqs_cpu = torch.clamp(next_seqs_cpu + step, max=self.max_model_len)
        next_seqs_cpu[num_reqs:].fill_(0)
        return next_seqs_cpu

    @torch.inference_mode()
    def propose(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Cache num_rejected for K>1 draft attn seq_lens (mainline leaves this TODO).
        num_rejected = kwargs.get("num_rejected")
        if num_rejected is None and len(args) >= 7:
            num_rejected = args[6]
        if isinstance(num_rejected, torch.Tensor):
            self._last_num_rejected_cpu = num_rejected.detach().to("cpu")
        return super().propose(*args, **kwargs)

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    ) -> None:
        """Eager non-fused multi-step with 310P CPU slot mappings.

        Concurrent (num_reqs>1) multi-step draft still garbles on 310P even in
        pure eager mode; skip extra steps so batch>1 behaves like K=1 while
        single-request keeps full K.
        """
        if num_reqs > 1 and self.num_speculative_steps > 1:
            logger.warning_once(
                "310P MTP: skipping K>1 multi-step draft for concurrent batch "
                "(num_reqs=%s); using first draft token only for correctness.",
                num_reqs,
            )
            # Invalidate remaining slots — stale draft ids would poison verify.
            self.draft_tokens[:num_reqs, 1:].fill_(0)
            return
        assert seq_lens_cpu_upper_bound is not None
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        # Align draft attn seq_lens with prepare_decode (subtract rejected).
        seq_ub = seq_lens_cpu_upper_bound
        rejected = getattr(self, "_last_num_rejected_cpu", None)
        if rejected is not None and rejected.numel() >= num_reqs:
            seq_ub = seq_lens_cpu_upper_bound.clone()
            seq_ub[:num_reqs] = seq_ub[:num_reqs] - rejected[:num_reqs].to(seq_ub.dtype)

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings_by_layer = self._compute_draft_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens,
                    seq_lens_cpu_upper_bound=seq_ub,
                    step=step,
                )

            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                batch_desc.num_tokens,
                attn_metadata,
                slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
