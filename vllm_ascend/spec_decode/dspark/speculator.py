# SPDX-License-Identifier: Apache-2.0
"""DSpark speculator (parallel backbone + serial Markov sample loop) on Ascend NPU.

Mirrors upstream vllm PR #46995 `DSparkSpeculator(DFlashSpeculator)` design:

* Inherits AscendDflashProposer to reuse the parallel context-KV precompute +
  query-block forward + attention metadata machinery.
* num_query_per_req = num_speculative_steps  (anchor-as-first-prediction;
  DFlash uses 1 + N).
* `_sample_sequential`: N-step Markov sample loop in place of DFlash's single
  parallel sample.
* `_generate_draft`: backbone forward + sequential Markov loop, captured under
  a single CUDA Graph / ACLGraph (FULL_DECODE_ONLY).

NOTE: This is the M1 framework — non-causal sparse attention (NPU equivalent
of upstream's `sparse_swa`), the full sequential Markov capture under
ACLGraph, and the `prepare_dspark_inputs` Triton kernel land in follow-up
sprints. The current implementation falls back to dense attention + eager
mode so the dispatch path can be exercised end-to-end while those pieces
are validated.
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark.load import load_dspark_model

if TYPE_CHECKING:
    from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM


class AscendDsparkSpeculator(AscendDflashProposer):
    """DSpark speculator on NPU.

    Inherits DFlash machinery; overrides:
      * draft model load to share target embed_tokens / lm_head.
      * sample path to insert Markov bias step by step.
      * (TODO) attention to use non-causal sparse SWA.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)

        # Anchor-as-first-prediction: each request emits N query tokens, NOT 1 + N.
        # Matches upstream PR #46995 speculator.py L48.
        self.num_query_per_req = self.num_speculative_tokens

        # Cached reference to the loaded DSpark draft model — resolved lazily
        # so the speculator can be instantiated before the model loads.
        self._dspark_model: DSparkDeepseekV4ForCausalLM | None = None

        # Per-block Markov sample state. Reset on every `propose()` call.
        self._markov_embeds_buffer: list[torch.Tensor] = []
        self.last_confidence: torch.Tensor | None = None

        logger.info_once(
            "AscendDsparkSpeculator initialised (num_query_per_req=%d). "
            "Non-causal sparse SWA + full CUDA/ACLGraph capture are follow-up work; "
            "current path runs dense attention in eager mode.",
            self.num_query_per_req,
        )

    # --- Model loading ----------------------------------------------------

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str] | None = None,
    ) -> torch.nn.Module:
        """Use the dspark loader so embed_tokens / lm_head get aliased to target."""
        model = load_dspark_model(target_model, self.vllm_config)
        self._dspark_model = model
        return model

    # --- Sample path (Markov-biased; replaces DFlash's parallel sample) ----

    def _sample_with_markov(
        self,
        head_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """N-step Markov sample loop over backbone head_hidden.

        Args:
          head_hidden: pre-norm head hidden, shape [num_reqs * N, dim].
          anchor_tokens: per-request bonus token, shape [num_reqs].

        Returns:
          draft_token_ids: shape [num_reqs, N].

        Mirrors upstream PR #46995 speculator.py L143-183. For each step i:
          base_logits[i] = compute_logits(head_hidden[i])
          markov_embed_i = markov_embed(prev_token)
          biased_i        = base_logits[i] + markov_bias(markov_embed_i)
          next_token_i    = sample(biased_i)
          prev_token      = next_token_i
        """
        assert self._dspark_model is not None, "_sample_with_markov called before model load."
        model = self._dspark_model
        n_spec = self.num_speculative_tokens
        num_reqs = anchor_tokens.shape[0]
        num_sample = num_reqs * n_spec

        sample_hidden = head_hidden[:num_sample]
        base_logits = model.compute_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        draft_tokens = torch.empty((num_reqs, n_spec), dtype=torch.int64, device=head_hidden.device)
        prev = anchor_tokens
        self._markov_embeds_buffer = []

        for i in range(n_spec):
            markov_embed = model.markov_embed(prev)
            self._markov_embeds_buffer.append(markov_embed)
            bias = model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            next_tok = logits_i.argmax(dim=-1)  # M1: greedy; probabilistic later.
            draft_tokens[:, i] = next_tok
            prev = next_tok

        # Compute confidence for the M2/M3 prefix scheduler hook-up.
        markov_embeds = torch.stack(self._markov_embeds_buffer, dim=1)  # [num_reqs, n_spec, rank]
        hidden_for_conf = sample_hidden.view(num_reqs, n_spec, -1)
        try:
            self.last_confidence = model.compute_confidence(hidden_for_conf, markov_embeds)
        except Exception as exc:  # pragma: no cover — best-effort, log + skip
            logger.warning_once("DSpark confidence head failed: %s", exc)
            self.last_confidence = None
        return draft_tokens

    # --- Public propose entry --------------------------------------------

    @torch.inference_mode()
    def propose(self, *args, **kwargs):
        """Reset per-block state then delegate to base.

        TODO (next sprint): override the full propose() to honour the
        anchor-as-first-prediction layout + drive _sample_with_markov in
        place of DFlash's parallel sample.
        """
        self._markov_embeds_buffer = []
        self.last_confidence = None
        return super().propose(*args, **kwargs)
