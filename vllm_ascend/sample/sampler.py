import torch
import torch_npu
import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, global_stream, npu_stream_switch

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

_SAMPLING_EPS = 1e-5


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    q.record_stream(torch.npu.current_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendSampler(Sampler):
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            logger.warning_once(
                "[sample/sampler] Triton not available, falling back to vLLM default "
                "penalty implementation. Penalty performance may be degraded on NPU. "
            )
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler(logprobs_mode=logprobs_mode)
        logger.debug(
            "[sample/sampler] AscendSampler initialized. logprobs_mode=%s, triton_available=%s",
            logprobs_mode,
            HAS_TRITON,
        )

    def prepare_sampling(self, top_k):
        self.topk_topp_sampler.prepare_sampling(top_k)

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        if not get_ascend_config().enable_reduce_sample:
            return super().apply_logits_processors(logits, sampling_metadata, predict_bonus_token)

        # When enable_reduce_sample is active, temporarily change the class
        # of MinTokensLogitsProcessor / LogitBiasLogitsProcessor instances
        # to their Ascend variants. This routes apply() through the Ascend
        # override while preserving all instance state (logits_slice, etc.).
        # The parent apply_logits_processors is called via super(), so any
        # upstream changes to that method are automatically picked up.
        procs = sampling_metadata.logitsprocs
        swaps = []
        for p in procs.non_argmax_invariant + procs.argmax_invariant:
            if isinstance(p, MinTokensLogitsProcessor) and not isinstance(p, AscendMinTokensLogitsProcessor):
                swaps.append((p, p.__class__))
                p.__class__ = AscendMinTokensLogitsProcessor
            elif isinstance(p, LogitBiasLogitsProcessor) and not isinstance(p, AscendLogitBiasLogitsProcessor):
                swaps.append((p, p.__class__))
                p.__class__ = AscendLogitBiasLogitsProcessor

        try:
            return super().apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
        finally:
            for p, orig_cls in swaps:
                p.__class__ = orig_cls

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        if get_ascend_config().enable_reduce_sample:
            logger.debug_once(
                "[sample/sampler] Using reduce-sample greedy sampling. "
                "TP all-gather will be performed to find global argmax.",
            )
            tp_group = get_tp_group()
            B, V_local = logits.shape
            rank = tp_group.rank_in_group

            local_max_logits, local_max_indices = logits.max(dim=-1)
            local_global_idx = local_max_indices + rank * V_local  # [B]
            # [B, world_size]
            gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
            gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1)  # [B, world_size]
            global_max_rank = gathered_logits.argmax(dim=-1)  # [B]
            target_argmax = gathered_global_idx.gather(dim=-1, index=global_max_rank.unsqueeze(-1)).squeeze(-1)  # [B]
            return target_argmax
        else:
            return logits.argmax(dim=-1).view(-1)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p
        self.top_k = None

    def prepare_sampling(self, top_k):
        if top_k is not None:
            self.top_k = top_k
        else:
            self.top_k = None

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if envs.VLLM_BATCH_INVARIANT:
            logger.debug_once(
                "[sample/sampler] BATCH_INVARIANT mode enabled, "
                "falling back to vLLM native top-k/top-p implementation.",
            )
            return super().forward_native(logits, generators, k, p)

        if get_ascend_config().enable_reduce_sample:
            logger.debug_once(
                "[sample/sampler] Using reduce-sample path in forward_native. "
                "top-k/top-p with TP all-gather for distributed sampling.",
            )
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = cand_logits.softmax(dim=-1, dtype=torch.float32)
            pos = random_sample(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B]
            return next_token, logits_to_return
        else:
            logits = self.apply_top_k_top_p(logits, k, p)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, generators), logits_to_return


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
    top_k: int | None = None,
) -> torch.Tensor:
    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        B, V_local = logits.shape
        rank = tp_group.rank_in_group

        if top_k is None or (p is None and k is None):
            k_for_topk = V_local
        else:
            k_for_topk = min(top_k, V_local)

        local_vals, local_idx = torch.topk(logits, k=k_for_topk, dim=-1)
        local_global_idx = local_idx + rank * V_local
        gathered_vals = tp_group.all_gather(local_vals, dim=-1)
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)

        if p is None and k is None:
            return gathered_vals, gathered_idx

        probs = gathered_vals.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)
        if k is not None:
            kk = k.to(torch.long).clamp(min=1, max=V_local)
            top_k_count = (probs_sort.size(1) - kk).unsqueeze(1)  # [B,1]
            top_k_cutoff = probs_sort.gather(-1, top_k_count)
            no_top_k_mask = (kk == V_local).unsqueeze(1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))
            elements_to_discard = probs < top_k_cutoff
            gathered_vals.masked_fill_(elements_to_discard, -float("inf"))
        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= (1 - p.unsqueeze(1))
            top_p_mask[:, -1] = False  # at least one
            top_p_count = top_p_mask.sum(dim=-1, keepdim=True)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            elements_to_discard = probs < top_p_cutoff
            gathered_vals.masked_fill_(elements_to_discard, -float("inf"))
        return gathered_vals, gathered_idx
    else:
        if p is None and k is None:
            return logits

        probs = logits.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)

        if k is not None:
            top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
            top_k_count = top_k_count.unsqueeze(dim=1)
            top_k_cutoff = probs_sort.gather(-1, top_k_count)

            # Make sure the no top-k rows are no-op.
            no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

            elements_to_discard = probs < top_k_cutoff
            logits.masked_fill_(elements_to_discard, -float("inf"))

        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False  # at least one

            top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            elements_to_discard = probs < top_p_cutoff
            logits.masked_fill_(elements_to_discard, -float("inf"))

        return logits


def _apply_top_k_top_p_torch_npu(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        B, V_local = logits.shape
        rank = tp_group.rank_in_group

        if top_k is None or (p is None and k is None):
            k_for_topk = V_local
        else:
            k_for_topk = min(top_k, V_local)

        local_vals, local_idx = torch.topk(logits, k=k_for_topk, dim=-1)
        local_global_idx = local_idx + rank * V_local
        gathered_vals = tp_group.all_gather(local_vals, dim=-1)
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)

        if not (p is None and k is None):
            gathered_vals = torch_npu.npu_top_k_top_p(gathered_vals, k=k, p=p)
        return gathered_vals, gathered_idx

    if p is None and k is None:
        return logits
    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_torch_npu
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)


def _convert_logits_slice_to_local(
    logits_slice: tuple[torch.Tensor, torch.Tensor],
    V_local: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert global token IDs in logits_slice to local shard indices.

    When enable_reduce_sample is active, logits are TP-partitioned
    (shape [B, V_local]) but logits_slice contains global token IDs.
    This method converts them to local shard indices on device without
    CPU synchronization. Tokens outside this shard are filtered out
    via boolean indexing.

    Returns (req_indices, local_tok_ids, in_shard_mask).
    """
    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    vocab_start = tp_rank * V_local
    vocab_end = vocab_start + V_local

    req_indices, tok_ids = logits_slice
    in_shard_mask = (tok_ids >= vocab_start) & (tok_ids < vocab_end)
    local_tok_ids = (tok_ids - vocab_start)[in_shard_mask]
    local_req_indices = req_indices[in_shard_mask]
    return (local_req_indices, local_tok_ids, in_shard_mask)


class AscendMinTokensLogitsProcessor(MinTokensLogitsProcessor):
    """Ascend variant that handles TP-partitioned logits when
    enable_reduce_sample is active."""

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_toks:
            return logits
        if get_ascend_config().enable_reduce_sample:
            V_local = logits.shape[-1]
            local_req, local_tok, _ = _convert_logits_slice_to_local(self.logits_slice, V_local)
            logits.index_put_((local_req, local_tok), self.neg_inf_tensor)
            return logits
        return super().apply(logits)


class AscendLogitBiasLogitsProcessor(LogitBiasLogitsProcessor):
    """Ascend variant that handles TP-partitioned logits when
    enable_reduce_sample is active."""

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.biases:
            return logits
        if get_ascend_config().enable_reduce_sample:
            V_local = logits.shape[-1]
            local_req, local_tok, in_shard = _convert_logits_slice_to_local(self.logits_slice, V_local)
            logits[local_req, local_tok] += self.bias_tensor[in_shard]
            return logits
        return super().apply(logits)
