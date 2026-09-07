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
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import disable_target_pcp_for_replicated_draft


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
        """Capture draft-prefill FULL with SpecDecoding; skip draft-decode graphs.

        Default ``AscendInputBatch.make_dummy`` forces DecodeOnly → PA, but MTP
        draft-prefill uses q_len=1+K (SpecDecoding/splitfuse). Draft-decode
        capture is illegal on 310P (host D2H / slot-map updates between steps).
        """
        self.last_token_indices.zero_()
        orig_make_dummy = AscendInputBatch.make_dummy

        @classmethod
        def make_dummy_spec_decode(
            cls,
            num_reqs: int,
            num_tokens: int,
            input_buffers: Any,
            max_query_len: int | None = None,
        ) -> AscendInputBatch:
            kwargs: dict[str, Any] = {}
            if max_query_len is not None:
                kwargs["max_query_len"] = max_query_len
            batch = orig_make_dummy(num_reqs, num_tokens, input_buffers, **kwargs)
            if num_reqs > 0 and (num_tokens // num_reqs) > 1:
                batch.attn_state = AscendAttentionState.SpecDecoding
            return batch

        AscendInputBatch.make_dummy = make_dummy_spec_decode  # type: ignore[method-assign]
        try:
            logger.info("Capturing draft-prefill ACLGraph for 310P MTP (SpecDecoding)...")
            assert self.prefill_cudagraph_manager is not None
            if self.prefill_cudagraph_manager.use_breakable_cg:
                self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
            with disable_target_pcp_for_replicated_draft(self):
                self.prefill_cudagraph_manager.capture(
                    self._prefill,
                    self.model_state,
                    self.target_input_buffers,
                    self.block_tables,
                    self.draft_prefill_attn_groups,
                    self.kv_cache_config,
                    progress_bar_desc="Capturing prefill CUDA graphs",
                )
            if self.num_speculative_steps > 1:
                logger.info(
                    "Skipping draft-decode ACLGraph capture on 310P MTP "
                    "(host slot-map / draft-input updates between steps)."
                )
        finally:
            AscendInputBatch.make_dummy = orig_make_dummy  # type: ignore[method-assign]

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

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    ) -> None:
        """Eager non-fused multi-step with 310P CPU slot mappings."""
        assert seq_lens_cpu_upper_bound is not None
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

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
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
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
