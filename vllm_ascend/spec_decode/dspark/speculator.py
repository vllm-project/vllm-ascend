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

        # NOTE on query layout: the DFlash base hardcodes
        # ``num_query_per_req = 1 + num_speculative_tokens`` locally inside its
        # prepare-inputs helpers (dflash_proposer.py L81/L172); it does NOT read
        # any instance attribute. So DSpark rides on the same ``1 + N`` layout —
        # the extra leading position is the verified anchor and the N trailing
        # positions are the draft slots. Only those N positions are gathered for
        # sampling (llm_base_proposer.py token_indices_to_sample), which is why
        # ``compute_logits`` receives exactly ``num_reqs * N`` rows and the
        # Markov wire in ``_propose`` lines up. We intentionally do NOT set
        # ``self.num_query_per_req`` here — it would be dead, misleading state.

        # Target model handle, captured in load_model so _get_model can route
        # draft creation through load_dspark_model (embed_tokens / lm_head alias).
        self._target_model: torch.nn.Module | None = None
        # Cached reference to the loaded DSpark draft model. Set by _get_model.
        self._dspark_model: DSparkDeepseekV4ForCausalLM | None = None

        # Per-block Markov sample state. Reset on every `propose()` call.
        self._markov_embeds_buffer: list[torch.Tensor] = []
        self.last_confidence: torch.Tensor | None = None

        logger.info_once(
            "AscendDsparkSpeculator initialised (num_speculative_tokens=%d). "
            "Non-causal sparse SWA + full CUDA/ACLGraph capture are follow-up work; "
            "current path runs dense attention in eager mode.",
            self.num_speculative_tokens,
        )

    # --- Model loading ----------------------------------------------------

    def load_model(self, target_model: torch.nn.Module) -> None:
        """Capture the target model so _get_model can alias its embeddings.

        The base ``load_model`` calls ``self._get_model()`` (which we override
        below) and then finishes draft attention-layer discovery. We only need
        to stash the target reference before delegating.
        """
        self._target_model = target_model
        super().load_model(target_model)

    def _get_model(self) -> torch.nn.Module:
        """Create the DSpark draft model via load_dspark_model.

        This is the real framework hook (base ``load_model`` does
        ``self.model = self._get_model()``). Routing through
        ``load_dspark_model`` both aliases the target's embed_tokens / lm_head
        into the draft (checkpoint stores draft weights in the target's
        ``mtp.*`` namespace with no private copies) and caches the concrete
        DSpark model so ``_propose`` / ``_sample_sequential`` can reach the
        Markov + confidence heads.
        """
        assert self._target_model is not None, "_get_model called before load_model captured target."
        model = load_dspark_model(self._target_model, self.vllm_config)
        self._dspark_model = model
        logger.info_once("DSpark draft model loaded and cached for Markov sampling.")
        return model

    # --- Method-string behaviour overrides -------------------------------

    def model_returns_tuple(self) -> bool:
        """DSpark's draft model returns a single hidden tensor, not a tuple.

        The base predicate (llm_base_proposer.py) excludes mtp/draft_model/dflash
        but not dspark, so it would try ``last, hidden = ret`` and raise
        ``ValueError: too many values to unpack``. DSpark is a DFlash variant and
        returns one tensor, so mirror dflash here.
        """
        return False

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

    # --- One-hot logits trick (wire helper) ------------------------------

    @staticmethod
    def _onehot_logits(
        flat: torch.Tensor,
        num_sample: int,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a ``[num_sample, vocab]`` logits tensor whose per-row argmax
        recovers ``flat``.

        ``flat`` has ``real = flat.numel()`` entries (the Markov-sampled token
        ids in row order). When ``num_sample > real`` (lmhead_tp padding), the
        trailing ``num_sample - real`` rows are left uniformly ``-1e30`` — the
        base proposer slices ``logits[:num_indices]`` so those rows are dropped,
        and their argmax (index 0) is never consumed.
        """
        real = flat.numel()
        assert num_sample >= real, f"num_sample {num_sample} < real {real}"
        fake = torch.full(
            (num_sample, vocab_size),
            -1.0e30,
            dtype=torch.float32,
            device=device,
        )
        fake[:real].scatter_(1, flat.unsqueeze(1), 1.0)
        return fake

    # --- Public propose entry (model_runner calls _propose, not propose) -

    @torch.inference_mode()
    def _propose(self, *args, **kwargs):
        """Reset per-block state, run DFlash backbone, replace argmax sample
        with the Markov serial loop.

        Implementation strategy (alternative to fully copy-pasting fork's
        700-line ``_propose``): we monkey-patch ``self.model.compute_logits``
        on the way in so the parent proposer's argmax step receives one-hot
        logits over our Markov-sampled token ids. The base proposer's call
        site (llm_base_proposer.py L1065-1074, parallel_drafting=True branch)
        is::

            logits = self.model.compute_logits(sample_hidden_states)
            draft_token_ids = logits.argmax(dim=-1)
            if ... or self.parallel_drafting:
                return draft_token_ids.view(-1, self.num_speculative_tokens)

        The argmax on a one-hot row recovers the original sampled token id,
        so this trick lets us swap parallel argmax for the Markov serial
        loop without rewriting the entire propose path.

        ``next_token_ids`` (per-request bonus token from the previous
        verification round) is the anchor for step 0 — pulled from kwargs
        (or args[3] under the documented positional signature).
        """
        self._markov_embeds_buffer = []
        self.last_confidence = None

        # Pull next_token_ids (positional index 3 in fork _propose signature
        # or kw alias) to seed the Markov anchor for step 0.
        anchor_tokens = kwargs.get("next_token_ids")
        if anchor_tokens is None and len(args) >= 4:
            anchor_tokens = args[3]

        if self._dspark_model is None or anchor_tokens is None:
            # Not yet a DSpark setup (e.g. profile run before load) — fall
            # back to parent behaviour with a warning.
            logger.warning_once(
                "AscendDsparkSpeculator._propose: skipping Markov serial "
                "sample (dspark_model=%s, anchor=%s); falling back to DFlash "
                "parallel sample.",
                "loaded" if self._dspark_model is not None else "none",
                "set" if anchor_tokens is not None else "none",
            )
            return super()._propose(*args, **kwargs)

        original_compute_logits = self.model.compute_logits
        markov_state = {"anchor": anchor_tokens}

        def _markov_compute_logits(sample_hidden_states, *cl_args, **cl_kwargs):
            """Replace argmax-on-logits with Markov serial sample.

            Returns a fake one-hot logits tensor so the caller's
            ``.argmax(dim=-1)`` recovers the Markov-sampled token id.
            """
            # Original logits computation is still needed once — we feed it
            # through _sample_sequential which calls model.compute_logits
            # itself. Make sure we don't recurse: temporarily restore the
            # original binding inside the call.
            self.model.compute_logits = original_compute_logits
            try:
                num_sample = sample_hidden_states.shape[0]
                num_reqs = markov_state["anchor"].shape[0]
                n_spec = self.num_speculative_tokens
                real = num_reqs * n_spec
                if num_sample < real:
                    # Genuinely mismatched call (e.g. profiling / a single
                    # anchor position) — fall back to raw logits so argmax
                    # still works. num_sample >= real is the normal / padded
                    # case handled below.
                    return original_compute_logits(sample_hidden_states, *cl_args, **cl_kwargs)
                # num_sample may exceed `real` when lmhead_tp pads the sample
                # buffer to max_num_reqs (llm_base_proposer.py L1036); the base
                # then slices logits[:num_indices], so only the first `real`
                # rows matter. Markov-sample exactly those.
                draft_tokens = self._sample_sequential(
                    num_reqs=num_reqs,
                    head_hidden=sample_hidden_states[:real],
                    anchor_tokens=markov_state["anchor"],
                )  # [num_reqs, N] int64
                flat = draft_tokens.reshape(-1)
                # Build a fake logits where argmax returns flat[i] for row i.
                # vocab_size comes from the config — no extra compute_logits call.
                vocab_size = self._dspark_model.config.vocab_size
                return self._onehot_logits(
                    flat, num_sample, vocab_size, sample_hidden_states.device
                )
            finally:
                # Restore the patched compute_logits so any nested DFlash
                # iterations later in _propose still go through it.
                self.model.compute_logits = _markov_compute_logits

        self.model.compute_logits = _markov_compute_logits
        try:
            draft_token_ids = super()._propose(*args, **kwargs)
        finally:
            self.model.compute_logits = original_compute_logits

        return draft_token_ids
