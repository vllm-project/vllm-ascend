# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P MTP speculator: CPU block-table slot mappings + RoPE flag + draft quant.

Target path aligns with MRv1 concurrent uniform SpecDecoding FULL (hybrid
``prepare_attn`` actual/pad split). Draft-prefill FULL (K=1) uses
``AutoRegressiveAclGraphManager310`` with SpecDecoding capture (splitfuse).
K>1 draft-decode FULL uses per-step graphs: host slot_mapping between steps.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import numpy as np
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
from vllm_ascend._310p.worker.v2.spec_utils import set_draft_step_host
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendMTPSpeculator310(AscendAutoRegressiveSpeculator, MTPSpeculator):
    """Ascend MTP speculator for 310P MRv2 (Triton-free draft loop)."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_model_config = self.speculative_config.draft_model_config
        if draft_model_config.hf_overrides is None:
            draft_model_config.hf_overrides = {}

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

    def _as_numpy_host(self, value: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False)
        if value.device.type == "cpu":
            return value.detach().numpy().astype(np.int64, copy=False)
        # Sync D2H is illegal while an NPU stream is capturing (GLOBAL mode).
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError(
                "310P draft slot_mapping cannot D2H while the NPU stream is capturing; "
                "prepare host mirrors outside ACLGraph capture."
            )
        return value.detach().cpu().numpy().astype(np.int64, copy=False)

    def _compute_draft_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> dict[str, torch.Tensor]:
        idx_mapping_np = self._as_numpy_host(idx_mapping)
        query_start_loc_np = self._as_numpy_host(query_start_loc)
        positions_np = self._as_numpy_host(positions)
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
        """Capture draft-prefill FULL + per-step draft-decode FULL on 310P."""
        self.last_token_indices.zero_()
        logger.info(
            "Capturing 310P MTP draft ACLGraph (draft-prefill FULL + SpecDecoding; draft-decode per-step FULL)."
        )
        super().capture()

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
        num_rejected = kwargs.get("num_rejected")
        if num_rejected is None and len(args) >= 7:
            num_rejected = args[6]
        if isinstance(num_rejected, torch.Tensor):
            self._last_num_rejected_cpu = num_rejected.detach().to("cpu")
        return super().propose(*args, **kwargs)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Skip Ascend metadata H2D under ACLGraph capture (pageable memcpy ban)."""
        # Call GPU AR generate_draft (sample + update_draft_inputs) without the
        # Ascend post-step ``seq_lens_cpu.copy_`` which is illegal while capturing.
        from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
            AutoRegressiveSpeculator,
        )

        AutoRegressiveSpeculator._generate_draft(
            self,
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        if attn_metadata is None or torch.npu.is_current_stream_capturing():
            return
        self._update_decode_attn_metadata(attn_metadata, 1, num_reqs)

    def _set_draft_step(self, step: int) -> None:
        self.current_draft_step.fill_(step)
        set_draft_step_host(step)

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    ) -> None:
        """K>1 draft decode: per-step FULL replay or eager CPU slot mappings."""
        assert seq_lens_cpu_upper_bound is not None
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]
        seq_ub = seq_lens_cpu_upper_bound
        rejected = getattr(self, "_last_num_rejected_cpu", None)
        if rejected is not None and rejected.numel() >= num_reqs:
            seq_ub = seq_lens_cpu_upper_bound.clone()
            seq_ub[:num_reqs] = seq_ub[:num_reqs] - rejected[:num_reqs].to(seq_ub.dtype)

        use_full = batch_desc.cg_mode == CUDAGraphMode.FULL
        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            if not skip_attn and (self.advance_draft_positions or step == 1):
                # Host slot_mapping + attn metadata must run outside capture.
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
                if attn_metadata is not None:
                    for meta in attn_metadata.values():
                        if meta is None:
                            continue
                        seq_lens = getattr(meta, "seq_lens", None)
                        if seq_lens is not None and seq_lens.device != self.device:
                            meta.seq_lens = seq_lens.to(device=self.device, non_blocking=False)

            self._set_draft_step(step)
            if use_full:
                assert self.decode_cudagraph_manager is not None
                self._pending_draft_attn_metadata = attn_metadata
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
                if attn_metadata is not None:
                    self._update_decode_attn_metadata(attn_metadata, 1, num_reqs)
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                )
