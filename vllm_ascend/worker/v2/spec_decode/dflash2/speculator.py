# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""DFlash2 speculative decoding for the Ascend V2 model runner."""

import torch
import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    load_dflash_model as upstream_load_dflash_model,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

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
    USE_FP64: tl.constexpr,
):
    """Walk DFlash2 candidates without the unsupported ``tldevice.log1p``.

    Candidate ids key the Gumbel draws exactly as in vLLM.  Ascend uses the
    direct ``-log(-log(u))`` transform, matching its target sampler and
    avoiding the precision loss of forming ``1 - u`` in fp32.
    """
    tl.static_assert(not USE_FP64, "fp64 Gumbel sampling is not supported on NPU")
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    effective_temperature = temperature if SAMPLE_PROBABILISTIC else 0.0
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

        if effective_temperature == 0.0:
            sampled_scores = scores
        else:
            position = tl.load(sample_pos_ptr + flat) - 1
            # Triton Ascend's Philox path requires int32 seed/offset operands.
            gumbel_seed = tl.randint(seed.to(tl.int32), position.to(tl.int32))
            uniform = tl_rand32(
                gumbel_seed,
                candidates.to(tl.int32),
                includes_zero=False,
            )
            noise = -tl.log(-tl.log(uniform))
            sampled_scores = scores / effective_temperature + noise

        sampled_scores = tl.where(mask & valid, sampled_scores, -float("inf"))
        best = tl.max(sampled_scores, axis=0)
        # Preserve upstream's deterministic lowest-index tie break.
        index = tl.min(tl.where(sampled_scores == best, offsets, BLOCK_K), axis=0)
        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(
            candidate_ptr + candidate_base + index,
            mask=valid,
            other=0,
        )
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


def _load_dflash_model_with_draft_rope(
    target_model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module:
    """Load an old-vLLM DFlash model without target-derived RoPE layout.

    vLLM #54373 removed target RoPE inference: DFlash rotates its own Q/K, so
    the layout is a property of the draft checkpoint. vLLM revisions predating
    that fix still infer it from the target. This is their loader with only
    that inference removed; sharing and attention configuration stay aligned
    with upstream.
    """
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_model_config,
        )

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(target_inner, "embedding", None)
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model,
            "has_own_embed_tokens",
            draft_embed,
            target_embed,
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model,
        "has_own_lm_head",
        draft_lm_head,
        target_lm_head,
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model


class AscendDFlash2Speculator(DFlash2Speculator, AscendDFlashSpeculator):
    """DFlash2 speculator with Ascend attention, graph, and sampling paths."""

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        # The helper removed by vLLM #54373 is a reliable feature probe.  Use
        # upstream directly once it contains the fix; support older revisions
        # without relying on version-string comparisons.
        from vllm.model_executor.models import qwen3_dflash

        if not hasattr(qwen3_dflash, "dflash_target_rope_is_neox_style"):
            return upstream_load_dflash_model(target_model, self.vllm_config)
        return _load_dflash_model_with_draft_rope(
            target_model,
            self.vllm_config,
        )

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
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
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )
