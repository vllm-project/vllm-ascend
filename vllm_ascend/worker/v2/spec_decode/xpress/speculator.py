# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.xpress.speculator import XPressSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.xpress.refiner import jacobi_refine_greedy


class AscendXPressSpeculator(XPressSpeculator, AscendDFlashSpeculator):
    """Reuse XPress loading/draft forward and Ascend DFlash attention/KV handling."""

    # XPress calls model.compute_draft_logits directly on whole query blocks;
    # it bypasses the mixin's group-aligned sample_draft padding.
    _lmhead_tp_sample_draft_supported = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        config = vllm_config.speculative_config
        assert config is not None
        if config.draft_sample_method != "greedy":
            raise NotImplementedError("XPress on Ascend currently requires draft_sample_method='greedy'.")
        if config.use_local_argmax_reduction:
            raise NotImplementedError("XPress requires full-vocabulary logits; disable use_local_argmax_reduction.")
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError("XPress on Ascend does not yet support prefill context parallelism.")
        super().__init__(vllm_config, device)
        if self.num_jacobi_passes < 1:
            raise ValueError("xpress_num_passes must be positive.")

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # XPress's uniform draft queries can use FULL graphs even when the
        # target uses PIECEWISE graphs. Keep eager and backend capability checks
        # in the parent implementation.
        if cudagraph_mode == CUDAGraphMode.PIECEWISE:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        super().init_cudagraph_manager(cudagraph_mode)
        if self.query_cudagraph_manager.cudagraph_mode.has_full_cudagraphs() and self.update_stream is None:
            # A PIECEWISE-only target does not create an attention update stream.
            self.update_stream = torch.npu.Stream()
            self.query_cudagraph_manager.update_stream = self.update_stream

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        # XPress's loader reuses load_dspark_model and does not delegate to
        # AscendDFlashSpeculator.load_draft_model in the MRO. Mirror its
        # post-load hook after upstream has shared target layers.
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if hasattr(model, "post_process"):
            model.post_process(self.vllm_config)
        head = model.model.xpress_head
        if head.block_size != self.num_query_per_req:
            raise ValueError(
                f"XPress checkpoint block size {head.block_size} requires num_speculative_tokens={head.block_size - 1}."
            )
        return model

    def _jacobi_refine(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if num_reqs == 0:
            return
        block_size = self.num_query_per_req
        num_rows = num_reqs * block_size
        # Ascend LM-head operators consume token rows. Restore the block axis
        # only after projection, without including padded request rows.
        hidden_rows = head_hidden[:num_rows]
        hidden = hidden_rows.reshape(num_reqs, block_size, -1)
        base_logits = self.model.compute_draft_logits(hidden_rows).reshape(num_reqs, block_size, -1)
        anchor_ids = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        # Match upstream's fill-in convention: slot 0 is the verified anchor
        # and also stands in for the inaccessible token preceding that anchor.
        draft, logits = jacobi_refine_greedy(
            self.model.model.xpress_head,
            base_logits,
            hidden,
            anchor_ids,
            anchor_ids,
            self.num_jacobi_passes,
            return_logits=self.acceptance_estimator is not None,
        )
        self.draft_tokens[:num_reqs, : self.num_speculative_steps] = draft
        if logits is not None:
            # Feed the logits that actually selected the LAST pass's tokens.
            # sample_idx_mapping addresses persistent request state, whereas
            # sample_col addresses draft slots; the estimator writes the
            # batch-ordered confidence buffer before adaptive budgeting runs.
            num_sample = num_reqs * self.num_speculative_steps
            self._maybe_predict_acceptance(
                logits,
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )
