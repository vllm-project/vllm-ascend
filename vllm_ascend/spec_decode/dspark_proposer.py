# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative-decoding proposer (paper arxiv:2606.19348).

Extends ``AscendEagleProposer`` with the two DSpark add-ons:

  * Per-step Markov bias: between the backbone-produced logits and the
    sampler, add ``markov_w2(markov_w1(prev_token))`` so the next sampled
    token is conditioned on the previous draft token. This is the
    "semi-autoregressive" piece — the heavy parallel backbone stays parallel,
    only the lightweight Markov head runs in a serial loop, paying ~one
    Embedding+Linear per draft position.
  * Per-block Confidence: after the full draft block is sampled, project the
    backbone hidden + stacked Markov embeddings through the confidence head
    to produce one acceptance-probability score per draft position. The M2.4
    truncation step (TODO) will use these scores to chop low-confidence
    suffix tokens off the draft block before verification.

State management:
  * Each ``propose()`` call is one independent draft block.
  * ``_dspark_prev_tokens`` carries the previous step's sampled token across
    successive ``_sample_draft_tokens`` calls within one ``propose``.
    Initialised to ``next_token_ids`` (the target's bonus token) on entry.
  * ``_markov_embeds_buffer`` accumulates the per-step Markov embedding,
    consumed once at the end of ``propose`` by the confidence head.
  * ``last_confidence`` (output) is stored on the proposer so the model
    runner can pull it for truncation after ``propose`` returns.
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend import envs
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

if TYPE_CHECKING:
    from vllm_ascend.models.deepseek_v4_mtp import DeepSeekMultiTokenPredictorLayer


class AscendDsparkProposer(AscendEagleProposer):
    """DSpark proposer: parallel backbone + serial Markov sample loop.

    The backbone draft forward (parallel; cost ≈ MTP) is inherited from
    ``AscendEagleProposer``. Only the per-step Markov bias and per-block
    Confidence projection are added on top, plumbed through two overrides:

      * ``propose``: resets DSpark per-block state on entry and computes
        the confidence scores on exit (stored at ``self.last_confidence``).
      * ``_sample_draft_tokens``: injects Markov bias onto compute_logits
        output before sampling.

    See ``DeepSeekMultiTokenPredictorLayer.apply_dspark_markov_bias`` and
    ``compute_dspark_confidence`` for the helper implementations on the
    model side.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert envs.VLLM_ASCEND_ENABLE_DSPARK, "AscendDsparkProposer requires VLLM_ASCEND_ENABLE_DSPARK=1."

        # M3.3 ACLGraph compat warning. DSpark's per-step Markov sample loop
        # mutates draft state across spec steps, which is not yet validated
        # against ACLGraph capture. Recommend enforce_eager until M3 graph
        # support lands; only warn (don't abort) so users can experiment.
        comp_cfg = getattr(vllm_config, "compilation_config", None)
        cudagraph_mode = getattr(comp_cfg, "cudagraph_mode", None)
        if cudagraph_mode is not None and str(cudagraph_mode) not in ("CUDAGraphMode.NONE", "NONE", "None"):
            logger.warning(
                "DSpark proposer is running with cudagraph_mode=%s; the per-step Markov bias is not yet "
                'validated under graph capture. Recommend --compilation-config \'{"cudagraph_mode":"NONE"}\' '
                "or --enforce-eager for the M1/M2 release; tracking ACLGraph support as M3.3.",
                cudagraph_mode,
            )

        # Per-block accumulator for markov_embed; the confidence head consumes
        # the full stack after the block-level sample loop completes.
        self._markov_embeds_buffer: list[torch.Tensor] = []
        # Most-recently-sampled draft tokens, indexed by spec_step. Used as the
        # ``prev_token_ids`` argument to the next Markov step.
        self._dspark_prev_tokens: torch.Tensor | None = None
        # Cached reference to the DSpark last-stage MTP layer (the one that
        # owns the markov + confidence heads). Resolved lazily so the model is
        # fully loaded by then.
        self._dspark_layer: DeepSeekMultiTokenPredictorLayer | None = None
        # Per-position confidence scores from the most recent ``propose`` call,
        # shape ``[batch_size, num_speculative_tokens]``. The model runner /
        # scheduler reads this to decide how many draft tokens to truncate
        # before verification (M2.4).
        self.last_confidence: torch.Tensor | None = None
        # Per-request kept-prefix length from the most recent ``propose`` call,
        # shape ``[batch_size]`` int64. The verify-batch shrink (M3.1 hard
        # truncate) consumes this to drop rows from the verify batch.
        self.last_truncated_lengths: torch.Tensor | None = None
        # Hidden state from the last spec step's sample point, used as the
        # input to the confidence head at block end. Captured inside
        # ``_sample_draft_tokens``.
        self._dspark_last_hidden: torch.Tensor | None = None

    def _resolve_dspark_layer(self) -> "DeepSeekMultiTokenPredictorLayer | None":
        """Locate the last MTP stage that carries the DSpark heads.

        Returns ``None`` when the loaded model is not a DSpark variant, in
        which case the proposer behaves identically to ``AscendEagleProposer``.
        """
        if self._dspark_layer is not None:
            return self._dspark_layer
        model = getattr(self, "model", None)
        if model is None:
            return None
        # DSv4 MTP nests: model.model.layers["<idx>"] — find the last layer
        # with ``is_dspark_last_layer == True``.
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            return None
        for layer in layers.values():
            if getattr(layer, "is_dspark_last_layer", False):
                self._dspark_layer = layer
                return layer
        return None

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample one draft step with Markov bias on top of backbone logits.

        On a non-DSpark model (no DSpark layer found in the model tree), this
        is a pure pass-through to ``super()._sample_draft_tokens``, so the
        proposer is safe even if it gets routed for a non-DSpark request.

        Greedy sampling still benefits from the Markov bias (the bias shifts
        the argmax target), so we skip the base-class shortcut that bypasses
        ``compute_logits`` for greedy.
        """
        layer = self._resolve_dspark_layer()
        if layer is None:
            return super()._sample_draft_tokens(hidden_states, sampling_metadata)

        # Always go through compute_logits + sample_from_logits (don't use the
        # greedy shortcut) so the Markov bias is on the path used by both
        # greedy and probabilistic sampling.
        logits = self.model.compute_logits(hidden_states)

        prev_tokens = self._dspark_prev_tokens
        if prev_tokens is not None:
            biased_logits, markov_embed = layer.apply_dspark_markov_bias(logits, prev_tokens)
            if markov_embed is not None:
                self._markov_embeds_buffer.append(markov_embed)
        else:
            # First step: no previous draft token yet. Sample from raw logits
            # and let _sample_from_logits handle greedy vs probabilistic.
            biased_logits = logits

        # _sample_from_logits handles greedy / probabilistic uniformly and
        # returns (next_tokens, draft_probs).
        next_tokens, draft_probs = self._sample_from_logits(biased_logits, sampling_metadata)

        # Capture for next Markov step and the per-block confidence call.
        self._dspark_prev_tokens = next_tokens
        self._dspark_last_hidden = hidden_states
        return next_tokens, draft_probs

    def propose(
        self,
        num_speculative_tokens,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        """Reset DSpark state on entry, then delegate to base propose.

        After ``super().propose`` returns, ``_sample_draft_tokens`` has been
        called ``num_speculative_tokens`` times and ``_markov_embeds_buffer``
        contains the per-step Markov embeddings. Project through the
        confidence head and stash the result at ``self.last_confidence`` for
        the model runner (M2.4 truncation).
        """
        # Reset per-block state. Use the target's bonus token as the initial
        # "previous draft token" for step 0's Markov bias.
        self._dspark_prev_tokens = next_token_ids
        self._markov_embeds_buffer = []
        self._dspark_last_hidden = None
        self.last_confidence = None

        draft_token_ids = super().propose(
            num_speculative_tokens=num_speculative_tokens,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            mm_embed_inputs=mm_embed_inputs,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        )

        # End-of-block: compute confidence scores. Skipped silently if no
        # markov embeddings were collected (e.g. K=1 path) — the model runner
        # treats ``last_confidence is None`` as "do not truncate".
        layer = self._resolve_dspark_layer()
        if layer is not None and self._markov_embeds_buffer and self._dspark_last_hidden is not None:
            markov_embeds = torch.stack(self._markov_embeds_buffer, dim=1)
            # hidden_states needs the same [bsz, block_size, dim] shape — the
            # last sample's hidden is [bsz, dim], so broadcast across positions.
            hidden = self._dspark_last_hidden.unsqueeze(1).expand_as(
                markov_embeds[..., : self._dspark_last_hidden.size(-1)]
            )
            self.last_confidence = layer.compute_dspark_confidence(hidden, markov_embeds)

            # M3.4 debug logger: when DEBUG logging is on, dump per-position
            # mean confidence so users can pick a sensible threshold. Cheap
            # enough to leave gated only by logger level (no hot-path cost
            # when DEBUG is off).
            if logger.isEnabledFor(10):  # logging.DEBUG = 10
                with torch.no_grad():
                    per_pos = self.last_confidence.float().mean(dim=0).cpu().tolist()
                logger.debug(
                    "DSpark per-position confidence (mean across batch): %s",
                    [f"{v:.3f}" for v in per_pos],
                )

            # M2.4 soft truncation: when a confidence threshold is configured,
            # replace draft tokens whose confidence falls below it with -1 so
            # the verify pass treats them as rejected. This does NOT shrink
            # the verify batch (so doesn't save verify compute — that needs
            # M3 scheduler integration), but it lets us validate end-to-end
            # that the confidence head is producing usable scores and that
            # truncation improves acceptance rates on data where the paper's
            # prefix scheduler would have helped.
            threshold = envs.VLLM_ASCEND_DSPARK_CONFIDENCE_THRESHOLD
            if threshold > 0.0:
                # M3.1: compute per-request kept-prefix length and expose it
                # to the model runner. The hard-truncate (shrinking the verify
                # batch) lives in the runner because that's where the
                # spec_decode_metadata + attention metadata are owned. For now
                # we still apply the soft mask here so the rejection sampler
                # treats the dropped suffix as -1.
                self.last_truncated_lengths = self._kept_prefix_lengths(self.last_confidence, threshold)
                draft_token_ids = self._soft_truncate_by_confidence(draft_token_ids, self.last_confidence, threshold)

        # Always clear per-block buffers before returning so a re-entrant
        # caller doesn't leak across propose calls.
        self._markov_embeds_buffer = []
        self._dspark_prev_tokens = None
        self._dspark_last_hidden = None
        return draft_token_ids

    @staticmethod
    def _kept_prefix_lengths(
        confidence: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Per-request length of the prefix that meets the confidence threshold.

        Used by the M3.1 verify-batch shrink path: ``out[b]`` is the number of
        leading draft positions in row ``b`` whose confidence is ``>= threshold``.
        Once one position drops below, the rest of the row is rejected (paper §3.2
        first-rejection rule).

        ``confidence`` shape ``[bsz, K]``; returns ``[bsz]`` int64 on the same
        device.
        """
        keep = (confidence >= threshold).to(torch.int64)
        # cumprod of int turns into a left-aligned 1..0 step, summing it
        # gives the contiguous-prefix length.
        return keep.cumprod(dim=-1).sum(dim=-1)

    @staticmethod
    def _soft_truncate_by_confidence(
        draft_token_ids: torch.Tensor,
        confidence: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Soft-truncate draft tokens whose confidence is below threshold.

        Left-to-right contiguous: once a position falls below threshold the
        rest of the suffix is also marked rejected (matching the paper's §3.2
        "first rejection rejects the suffix" rule for the confidence-scheduled
        prefix scheduler).

        Returns a fresh tensor with rejected positions set to -1.
        """
        # confidence: [bsz, K], draft_token_ids: [bsz, K]
        keep = (confidence >= threshold).to(draft_token_ids.device)
        # cumulative AND so a single drop breaks the rest of the row.
        keep = keep.cumprod(dim=-1).bool()
        return torch.where(keep, draft_token_ids, draft_token_ids.new_full((), -1))
