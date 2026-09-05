# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


@triton.jit
def _selector_walk_kernel_ascend(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0

    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        if not SAMPLE_PROBABILISTIC or temperature == 0.0:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, BLOCK_K), axis=0)
        else:
            # Triton Ascend does not support the uint64/float64 path used by
            # upstream. Request seeds, token ids, and positions use the NPU's
            # int32 Philox path; realized FP32 logits are kept for verification.
            position = (tl.load(sample_pos_ptr + flat) - 1).to(tl.int32)
            gumbel_seed = tl.randint(seed.to(tl.int32), position)
            # Token ids key the noise, preserving the upstream DFlash2
            # candidate-sampling contract.
            uniform = tl_rand32(
                gumbel_seed,
                candidates.to(tl.int32),
                includes_zero=False,
            )
            noise = -tl.log(-tl.log(1.0 - uniform))
            sampled_scores = tl.where(mask, scores / temperature + noise, float("-inf"))
            best = tl.max(sampled_scores, axis=0)
            index = tl.min(tl.where(sampled_scores == best, offsets, BLOCK_K), axis=0)

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


class AscendDFlash2Speculator(DFlash2Speculator, AscendDFlashSpeculator):
    """DFlash2 speculator with Ascend attention and sampling support."""

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The V2 runner passes the target model's graph mode here without
        # consulting the draft-specific eager override.  Honor that override
        # without disabling graphs for the target model.
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # DFlash2's draft RoPE and CandidateSelector still hit unsupported NPU
        # FakeTensor/broadcast paths under torch.compile. Disable torch.compile
        # on this draft instance only; ACLGraph capture remains controlled by
        # init_cudagraph_manager, and the target model is left untouched.
        target_module_ids = {id(module) for module in target_model.modules()}
        for module in model.modules():
            if id(module) not in target_module_ids and hasattr(module, "do_not_compile"):
                module.do_not_compile = True
        return model

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if self.use_fp64_gumbel:
            raise NotImplementedError("FP64 DFlash2 candidate sampling is not supported on NPU.")

        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel_ascend[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            num_warps=1,
        )
