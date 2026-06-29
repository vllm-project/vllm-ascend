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

    # --- Markov serial sample loop (port of vllm #46995 speculator.py L143) -

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        sample_indices: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """N-step Markov sample loop over backbone head_hidden.

        Direct port of upstream vllm #46995 ``DSparkSpeculator._sample_sequential``
        with two NPU-side adjustments:

          1. No ``gumbel_sample``/``draft_logits`` plumbing — probabilistic
             drafting uses ``torch.multinomial`` (the upstream version's
             Gumbel-based path has a known correctness issue per #46995 Known
             Issues; greedy is the safer default for M1).
          2. ``sample_indices`` is optional. When provided, ``head_hidden`` is
             treated as a flat ``[num_tokens, dim]`` buffer and indexed; when
             omitted, ``head_hidden`` is assumed to be already
             ``[num_reqs * N, dim]`` in (req, step) order.

        Args:
          num_reqs:      number of active requests in this draft step.
          head_hidden:   pre-norm head hidden, either ``[T, dim]`` (with
                         ``sample_indices``) or ``[num_reqs*N, dim]``.
          anchor_tokens: per-request bonus token from the previous verification
                         round, shape ``[num_reqs]``. Mirrors the upstream
                         persistent ``self._anchor_idx`` lookup.
          sample_indices: optional flat indices into ``head_hidden`` selecting
                         the ``[num_reqs * N]`` sample positions in (req, step)
                         order.
          temperature:   per-request sampling temperature, shape ``[num_reqs]``.
                         If None or all-greedy, this path uses ``argmax``.
          seeds:         per-request RNG seed for probabilistic sampling.

        Returns:
          draft_token_ids of shape ``[num_reqs, N]``.

        Implements paper §3.1 (Markov bias loop), equation 5:
          B(x_{k-1}, ·) = W1[x_{k-1}] · W2
          logits_k      = U_k + B(x_{k-1}, ·)
          x_k           = sample(softmax(logits_k))

        Sets ``self.last_confidence`` (per-position acceptance scores) for the
        M2/M3 prefix scheduler hook-up after the block is complete.
        """
        assert self._dspark_model is not None, "_sample_sequential called before model load."
        model = self._dspark_model
        n_spec = self.num_speculative_tokens
        num_sample = num_reqs * n_spec

        # Gather the per-(req, step) hidden states.
        if sample_indices is not None:
            sample_hidden = head_hidden[sample_indices[:num_sample]]
        else:
            sample_hidden = head_hidden[:num_sample]

        # Backbone logits U_k for all N positions, computed in one matmul.
        base_logits = model.compute_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        # Greedy path when no temperature provided or all temperatures are 0.
        is_greedy = temperature is None or temperature.numel() == 0 or bool((temperature == 0).all().item())

        draft_tokens = torch.empty(
            (num_reqs, n_spec),
            dtype=torch.int64,
            device=head_hidden.device,
        )
        prev = anchor_tokens
        self._markov_embeds_buffer = []

        for i in range(n_spec):
            # Sequential Markov bias: the bias for position i depends on the
            # token sampled at position i-1.
            markov_embed = model.markov_embed(prev)
            self._markov_embeds_buffer.append(markov_embed)
            bias = model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias

            if is_greedy:
                next_tok = logits_i.argmax(dim=-1)
            else:
                # Probabilistic sampling — apply per-request temperature and
                # multinomial-sample. Gumbel is intentionally skipped per
                # upstream #46995 Known Issues (probabilistic correctness bug).
                temp = temperature.view(-1, 1).to(logits_i.dtype)
                scaled = logits_i / temp.clamp_min(1e-5)
                probs = torch.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            draft_tokens[:, i] = next_tok
            prev = next_tok

        # Per-block confidence for the M2/M3 prefix scheduler. Empty / failure
        # paths log and set ``last_confidence = None`` so downstream callers
        # can treat None as "no truncation".
        markov_embeds = torch.stack(self._markov_embeds_buffer, dim=1)  # [num_reqs, N, rank]
        hidden_for_conf = sample_hidden.view(num_reqs, n_spec, -1)
        try:
            self.last_confidence = model.compute_confidence(hidden_for_conf, markov_embeds)
        except Exception as exc:  # pragma: no cover — best-effort, log + skip
            logger.warning_once("DSpark confidence head failed: %s", exc)
            self.last_confidence = None
        return draft_tokens

    # --- Public propose entry (model_runner calls _propose, not propose) -

    @torch.inference_mode()
    def _propose(self, *args, **kwargs):
        """Reset per-block state, then run the parent's parallel backbone.

        M1 framework: DFlash parallel backbone borrowed verbatim from
        ``AscendDflashProposer._propose``. The Markov bias / serial sample
        loop hook (``_sample_with_markov``) is still **not wired** into the
        actual sample call site inside the parent's _propose. The follow-up
        sprint will replace the parent's ``_sample_from_logits`` invocation
        with the Markov loop so the returned tokens carry intra-block
        dependency.

        For now this method is identical to DFlash's propose path with the
        DSpark draft model loaded — the architecture is in place but the
        signature Markov-bias step does not yet fire.
        """
        self._markov_embeds_buffer = []
        self.last_confidence = None
        draft_token_ids = super()._propose(*args, **kwargs)
        logger.warning_once(
            "AscendDsparkSpeculator._propose: Markov sample loop NOT wired "
            "yet (sprint follow-up). Returned draft tokens match DFlash "
            "parallel sampling without the intra-block Markov bias; "
            "confidence head inactive."
        )
        return draft_token_ids
