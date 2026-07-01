# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendDSparkProposer — DeepSeek-V4 DSpark speculative-decode proposer.

DSpark's parallel backbone IS DFlash (parallel block construction + target-KV
injection + ACL graph), so this subclasses AscendDflashProposer and inherits
nearly everything:
  * set_inputs_first_pass        : gamma-block query (anchor + mask) construction
  * build_model_inputs_first_pass: calls model.precompute_and_store_context_kv
                                   (DSpark = MLA context-latent injection from main_x)
  * dummy_run / ACL graph capture, DP sync, parallel_drafting

The DSpark-specific pieces live in the MODEL (deepseek_v4_dspark.DeepSeekV4DSpark):
  * combine_hidden_states  = main_norm(main_proj(aux))   (called by base _propose)
  * precompute_and_store_context_kv = per-stage MLA latent -> paged cache
  * forward                = 3-deep V4 stack over the gamma block
  * compute_logits         = base draft logits U_k

Staging:
  * MVP (this class as-is)  = "DSpark-minus-Markov" == DFlash over the V4 backbone.
    Reuses the base parallel-drafting token path (compute_logits -> sample). Gives
    the parallel-backbone speculative speedup; measured vs MTP-1 == first benefit.
  * Phase 2 (compute_block override below) = semi-AR Markov head + confidence.
    The base samples gamma tokens from U_k independently; DSpark instead runs
    model.compute_block(hidden, anchor_ids, temp) -> (output_ids, confidence) which
    adds the per-step Markov bias (Eq.5) before sampling and emits the confidence
    logits (Eq.7) for the Algorithm-1 scheduler. Wire by overriding the
    post-forward token generation (base llm_base_proposer _run_merged_draft, the
    `compute_logits(sample_hidden_states)` site) to call compute_block instead.
"""
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

logger = init_logger(__name__)


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark proposer. MVP inherits the full DFlash flow; the semi-AR Markov head
    + confidence-scheduled verification are layered on at phase 2 (see module doc)."""

    def __init__(self, vllm_config: VllmConfig, device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        spec = vllm_config.speculative_config
        cfg = spec.draft_model_config.hf_config
        # DSpark draft block (gamma) must equal num_speculative_tokens so the
        # parallel block construction lines up with the verify window.
        self.dspark_block_size = getattr(cfg, "dspark_block_size", None)
        self.use_markov_head = False  # phase-2 switch (compute_block path)
        logger.info(
            "AscendDSparkProposer: gamma=%s num_spec=%s markov_head=%s (MVP=DFlash-backbone)",
            self.dspark_block_size, self.num_speculative_tokens, self.use_markov_head,
        )

    # --- Phase 2 hook (semi-AR Markov + confidence) -------------------------
    # def _draft_tokens_from_hidden(self, sample_hidden_states, anchor_ids):
    #     """Override the base U_k -> sample path with DSpark's compute_block:
    #     output_ids, confidence = self.model.compute_block(
    #         sample_hidden_states, anchor_ids, self.temperature)
    #     Stash `confidence` for the Algorithm-1 scheduler (schedule_prefix_lengths).
    #     """
