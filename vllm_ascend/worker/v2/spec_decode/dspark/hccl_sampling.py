# SPDX-License-Identifier: Apache-2.0
"""Experimental full-vocabulary DSpark request sharding for Ascend V2.

Opt in with VLLM_ASCEND_DSPARK_BATCH_SHARDED=1. This environment variable is
introduced by this patch, not an upstream vLLM configuration option.

The vLLM 0.33.0 branch supports GREEDY drafting only, without a proposal-logits
cache. Newer vLLM keeps the existing DraftLogitsCache integration.
"""
import os

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from vllm_ascend.ops.triton.sample.categorical_sample import categorical_sample
from vllm_ascend.utils import lmhead_tp_enable, vllm_version_is

logger = init_logger(__name__)


def exchange_logits(local_logits, tp_size, local_rows, device_group):
    """Equal-sized HCCL transpose: [P*L,V/P] -> [L,V_padded]."""
    if local_logits.shape[0] != tp_size * local_rows:
        raise ValueError("The owner-padded logits layout has the wrong row count")
    width = local_logits.shape[-1]
    recv = torch.empty(local_logits.shape, dtype=local_logits.dtype, device=local_logits.device)
    dist.all_to_all_single(recv, local_logits.contiguous(), group=device_group)
    return recv.view(tp_size, local_rows, width).permute(1, 0, 2).reshape(local_rows, tp_size * width)


class DSparkHCCLSamplingMixin:
    """Use before the upstream/Ascend DSpark bases in the MRO."""

    def __init__(self, vllm_config, device):
        setting = os.getenv("VLLM_ASCEND_DSPARK_BATCH_SHARDED", "0")
        if setting not in ("0", "1"):
            raise ValueError("VLLM_ASCEND_DSPARK_BATCH_SHARDED must be 0 or 1")
        self._hccl_draft_requested = setting == "1"
        self._hccl_draft_enabled = False
        self._hccl_legacy_033 = False
        super().__init__(vllm_config, device)
        if not self._hccl_draft_requested:
            return
        parallel = vllm_config.parallel_config
        self._hccl_legacy_033 = vllm_version_is("0.33.0")
        if self._hccl_legacy_033 and self.draft_logits is not None:
            raise ValueError(
                "The 0.33.0 HCCL adapter supports draft_sample_method='greedy' only; "
                "probabilistic tensor-cache drafting needs a separate cache implementation."
            )
        if not parallel.enable_batch_sharded_sampling:
            raise ValueError("DSpark HCCL sharding requires --enable-batch-sharded-sampling")
        if parallel.tensor_parallel_size <= 1:
            raise ValueError("DSpark HCCL sharding requires TP > 1")
        if (parallel.data_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1):
            raise ValueError("This experimental patch supports DP=PP=PCP=DCP=1")
        if lmhead_tp_enable():
            raise ValueError("Fine-grained LM-head TP is not supported by this patch")
        if self.enable_adaptive_verification or self.use_acceptance_estimator:
            raise ValueError("This first implementation requires adaptive verification/online estimation disabled")
        if self.draft_watermarker is not None or self._draft_topk is not None:
            raise ValueError("Watermarking and truncated top-k drafting are outside this patch's scope")
        self._build_hccl_layouts()

    def supports_batch_sharded_draft_logits(self):
        # Returning True is safe only together with the categorical cache-write
        # and rejection-input adapters shipped in the same patch.
        return self._hccl_draft_requested and not self._hccl_legacy_033

    def _build_hccl_layouts(self):
        group = get_tp_group()
        p, k = group.world_size, self.num_speculative_steps
        self._hccl_min_reqs = int(os.getenv("VLLM_ASCEND_DSPARK_SHARD_MIN_REQS", str(p)))
        if not 1 <= self._hccl_min_reqs <= self.max_num_reqs:
            raise ValueError("VLLM_ASCEND_DSPARK_SHARD_MIN_REQS must be between 1 and max_num_seqs")
        self._hccl_layouts = {}
        for n in range(1, self.max_num_reqs + 1):
            capacity = (n + p - 1) // p
            requests = (torch.arange(p, device=self.device)[:, None]
                        + p * torch.arange(capacity, device=self.device)[None, :])
            requests = torch.where(requests < n, requests, -1)
            rows = requests.clamp(min=0)[..., None] * k + self._step_cols
            batch_indices = torch.arange(n, device=self.device)
            restore = (batch_indices % p) * capacity + batch_indices // p
            tokens = self.draft_tokens.new_empty(capacity, k)
            gathered = self.draft_tokens.new_empty(p * capacity, k)
            self._hccl_layouts[n] = (capacity, requests, rows, restore, tokens, gathered)

    def load_model(self, target_model):
        super().load_model(target_model)
        if self._hccl_draft_requested:
            if not hasattr(target_model, "compute_logits_local"):
                raise ValueError("The target model must expose compute_logits_local")
            if self._hccl_legacy_033:
                # v0.33.0 DSpark.load_model never invokes the newer callback.
                # Activate only after the real draft model/head are loaded.
                self._try_enable_sharded_sequential_sampling()
            if not self._hccl_draft_enabled:
                raise ValueError("DSpark HCCL sharding did not activate; verify target compute_logits_local and V2 runner")
            # Ascend categorical_sample does not accept the upstream fused-bias
            # kwarg. Preserve logits+bias rounding before caching and sampling.
            if not self._hccl_legacy_033:
                self._fuse_markov_bias = False

    def _try_enable_sharded_sequential_sampling(self):
        # Called by upstream DSpark.load_model after model/cache creation.
        # Do not route an NPU through PyNCCL.
        if not self._hccl_draft_requested:
            return
        model = self.model
        group = get_tp_group()
        if type(model).__name__ != "DSparkDeepseekV41ForCausalLM":
            raise ValueError("This patch has been written for Ascend DSparkDeepseekV41ForCausalLM")
        if not hasattr(model, "compute_draft_logits_local") or model.lm_head.tp_size != group.world_size:
            raise ValueError("Draft LM head must expose the full TP group's padded vocabulary shard")
        if model.draft_id_to_target_id is not None or self._d2t_scatter_index is not None:
            raise ValueError("Reduced draft vocabulary is not supported")
        if hasattr(model, "sample_sequential") or hasattr(model, "sample_markov_chain"):
            raise ValueError("Model-specific fused Markov sampling needs a separate adapter")
        if getattr(model.model.markov_head.markov_w2, "tp_size", None) != 1:
            raise ValueError("Markov W2 must remain replicated; a sharded W2 would need per-step collectives")
        if self.use_confidence_head:
            raise ValueError("Adaptive confidence-head gathering is outside this patch's scope")
        if self._hccl_legacy_033:
            if self.draft_logits is not None:
                raise ValueError("0.33.0 request-sharded drafting supports only a greedy draft with no q cache")
            if getattr(model, "has_recurrent_markov_head", False):
                raise ValueError("The 0.33.0 adapter supports the vanilla non-recurrent Markov head")
        else:
            if self.draft_logits is not None and self.draft_logits.row_mapping is None:
                raise ValueError("Probabilistic drafting needs the sharded DraftLogitsCache")
            if not hasattr(self, "_sample_markov_chain"):
                raise ValueError("The installed upstream DSpark sampler has an incompatible API")
        self._hccl_draft_enabled = True
        logger.info("Ascend DSpark HCCL request-sharded Markov sampling enabled (TP=%d).", group.world_size)
        if self._hccl_legacy_033:
            logger.info("Ascend DSpark HCCL adapter: vLLM 0.33.0, greedy draft, no proposal-logits cache.")

    def _sample_logits(self, logits, idx_map, sample_pos, step, bias=None):
        if not self._hccl_draft_enabled:
            if bias is None:
                return super()._sample_logits(logits, idx_map, sample_pos, step)
            return super()._sample_logits(logits, idx_map, sample_pos, step, bias=bias)
        if bias is not None:
            raise RuntimeError("The HCCL path requires the Markov bias to be materialized before sampling")
        if self._hccl_legacy_033 and self.draft_logits is not None:
            raise RuntimeError("The 0.33.0 greedy adapter must not write or consume a proposal-logits cache")
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))
        return categorical_sample(
            logits, idx_map, self.temperature, self.seeds, sample_pos - 1,
            apply_temperature=True, is_drafting=True,
            logits_cache=self.draft_logits.values,
            logits_cache_col=self._step_cols[step],
            logits_cache_row_mapping=self.draft_logits.row_mapping,
            use_fp64=self.use_fp64_gumbel,
        )

    def _sample_markov_chain_033(self, base_logits, anchor, idx_map, sample_pos, out_tokens):
        """v0.33.0's vanilla greedy Markov loop, on this rank's requests.

        No cache, RNG, or rejection-acceptance decision is introduced here.
        Synthetic acceptance remains entirely in the target verifier.
        """
        prev = anchor
        for step in range(self.num_speculative_steps):
            markov_embed = self.model.markov_embed(prev)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, step] + bias
            if hasattr(self.model, "apply_output_logit_transform"):
                logits_i = self.model.apply_output_logit_transform(logits_i)
            selected = self._sample_logits(logits_i, idx_map[:, step], sample_pos[:, step], step)
            out_tokens[:, step] = selected
            prev = selected

    def _sample_sequential(self, num_reqs, head_hidden):
        if not self._hccl_draft_enabled:
            return super()._sample_sequential(num_reqs, head_hidden)
        if num_reqs == 0:
            return
        group = get_tp_group()
        p, rank, k = group.world_size, group.rank_in_group, self.num_speculative_steps
        if num_reqs < self._hccl_min_reqs:
            # The original replicated chain also uses our mapped cache writer.
            # Avoid padding a tiny batch to P requests just for all-to-all.
            return super()._sample_sequential(num_reqs, head_hidden)
        capacity, requests, rows, restore, tokens, gathered = self._hccl_layouts[num_reqs]
        # Every rank computes its vocabulary shard for the same owner-sorted,
        # padded list of request/step rows. All-to-all splits are equal.
        send_hidden = head_hidden[self.sample_indices[rows.flatten()]]
        partial = self.model.compute_draft_logits_local(send_hidden)
        base = exchange_logits(partial, p, capacity * k, group.device_group)
        base = base.view(capacity, k, -1)[..., :self.model.logits_processor.org_vocab_size]

        local_requests, local_rows = requests[rank], rows[rank]
        mapping = self.sample_idx_mapping[local_rows]
        mapping = torch.where(local_requests[:, None] >= 0, mapping, -1)
        local_hidden = send_hidden.view(p, capacity * k, -1)[rank]
        anchor = self.input_buffers.input_ids[self._anchor_idx[local_requests.clamp(min=0)]]
        if self._hccl_legacy_033:
            self._sample_markov_chain_033(base, anchor, mapping, self.sample_pos[local_rows], tokens)
        else:
            self._sample_markov_chain(
                base, local_hidden, anchor, mapping, self.sample_pos[local_rows], tokens,
            )
        # Keep the Markov chain local; communicate only the finished token block.
        dist.all_gather_into_tensor(gathered, tokens.contiguous(), group=group.device_group)
        self.draft_tokens[:num_reqs] = gathered[restore]
