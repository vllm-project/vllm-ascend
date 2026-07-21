import torch
import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.force_topk_sample import build_compact_for_logprobs, force_topk_sample
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.sample.topk_map import CompactDist
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
        self.async_exponential_event = torch.npu.Event()
        self.force_topk = ascend_envs.VLLM_ASCEND_SAMPLER_FORCE_TOPK
        if self.force_topk > 0:
            logger.info(
                "[sample/sampler] force_topk enabled, k=%d. "
                "See FORCE_TOPK_DESIGN.md for semantics and fallback conditions.",
                self.force_topk,
            )
        logger.debug(
            "[sample/sampler] AscendSampler initialized. logprobs_mode=%s, triton_available=%s",
            logprobs_mode,
            HAS_TRITON,
        )

    def set_q_event(self, q, event):
        self.topk_topp_sampler.set_q_event(q, event)

    def prepare_sampling(self, top_k):
        self.topk_topp_sampler.prepare_sampling(top_k)

    def do_async_exponential(self, b_s, head_dim, generators):
        # Calculating exponential randoms in a different stream
        # and overlapping with model executing.
        with torch.npu.stream(global_stream()):
            global_stream().wait_stream(torch.npu.current_stream())
            q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
            # Goes to async exponential with AI-CPU exponential or default exponential.
            if len(generators) != q.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)

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

    # ------------------------------------------------------------------ #
    # force_topk: override sample() and gather_logprobs()                #
    # See FORCE_TOPK_DESIGN.md §4.2 / §4.5                               #
    # ------------------------------------------------------------------ #

    def _force_topk_enabled(self, sampling_metadata: SamplingMetadata) -> bool:
        """Check whether force_topk is safe to use for this batch.

        Returns False (→ fall back to super().sample()) when any of the
        design's safety conditions are violated. See FORCE_TOPK_DESIGN.md
        §4.6 and the decision log (D1-D8).
        """
        if self.force_topk <= 0:
            return False
        # D2: batch_invariant debug mode requires bit-exact upstream path
        if envs.VLLM_BATCH_INVARIANT:
            logger.warning_once("[force_topk] fallback: VLLM_BATCH_INVARIANT enabled")  # TODO: remove after debugging
            return False
        # D3: reduce_sample uses TP-sharded logits; force_topk on local shard
        # would not be a global top-k. Needs separate analysis (design TODO).
        if get_ascend_config().enable_reduce_sample:
            logger.warning_once("[force_topk] fallback: enable_reduce_sample=True")  # TODO: remove after debugging
            return False
        # D7: CompactDist semantics (topv - LSE(z)) only match raw_logprobs
        if self.logprobs_mode != "raw_logprobs":
            logger.warning_once("[force_topk] fallback: logprobs_mode=%s != raw_logprobs", self.logprobs_mode)  # TODO: remove after debugging
            return False
        # D4/D5: top-n must fit in k; -1 means full-vocab logprobs request
        num_lp = sampling_metadata.max_num_logprobs
        if num_lp is not None and (num_lp == -1 or num_lp > self.force_topk):
            logger.warning_once("[force_topk] fallback: num_logprobs=%d, k=%d", num_lp, self.force_topk)  # TODO: remove after debugging
            return False
        # D6: logprob_token_ids (generative_scoring) may query tokens outside
        # top-k → CompactDist.gather would return -inf (incorrect)
        if sampling_metadata.logprob_token_ids:
            logger.warning_once("[force_topk] fallback: logprob_token_ids non-empty")  # TODO: remove after debugging
            return False
        # D8: penalties change LSE, breaking I3 (raw vs post-penalty logprobs).
        # Only matters when logprobs are actually requested (design §4.6).
        wants_logprobs = num_lp is not None or bool(
            sampling_metadata.logprob_token_ids
        )
        if not sampling_metadata.no_penalties and wants_logprobs:
            logger.warning_once("[force_topk] fallback: penalties + logprobs requested")  # TODO: remove after debugging
            return False
        # Decision B: if there are argmax_invariant processors other than
        # MinP, we cannot cover their logic in force_topk_sample → fall back.
        for proc in sampling_metadata.logitsprocs.argmax_invariant:
            if not hasattr(proc, "min_p_count"):
                logger.warning_once("[force_topk] fallback: non-MinP argmax_invariant processor %s", type(proc).__name__)  # TODO: remove after debugging
                return False
        return True

    @staticmethod
    def _extract_min_p(
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        """Extract the min_p tensor from MinPLogitsProcessor without running
        its apply() (which would do a [B,V] softmax — the very thing force_topk
        aims to eliminate).

        Uses duck-typing on the ``min_p_count`` / ``min_p`` attributes.
        Returns None if no request uses min_p.
        """
        for proc in sampling_metadata.logitsprocs.argmax_invariant:
            if hasattr(proc, "min_p_count") and proc.min_p_count > 0:
                # min_p is stored as [B, 1]; squeeze to [B]
                return proc.min_p.squeeze(-1)
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Override sample() to add the force_topk branch.

        When force_topk is disabled or unsafe, delegates to super().sample().
        Otherwise, performs sampling entirely in [B, k] compact space and
        returns a CompactDist (instead of None) as the second element.
        The CompactDist flows through forward()'s ``raw_logprobs = cdist``
        assignment (sampler.py:103-104) into our gather_logprobs() override.
        """
        if not self._force_topk_enabled(sampling_metadata):
            return super().sample(logits, sampling_metadata, logprobs_mode_override)

        k = self.force_topk
        logger.warning(  # TODO: remove after debugging
            "[force_topk] sample() enter: B=%d, V=%d, k=%d, all_greedy=%s, all_random=%s",
            logits.shape[0], logits.shape[1], k,
            sampling_metadata.all_greedy, sampling_metadata.all_random,
        )
        # I1: all-greedy batch → full-vocab argmax, skip sampling path.
        # Still build CompactDist for logprobs reporting if requested.
        if sampling_metadata.all_greedy:
            logger.warning("[force_topk] all_greedy branch")  # TODO: remove after debugging
            greedy = self.greedy_sample(logits)
            cdist = None
            if (
                sampling_metadata.max_num_logprobs is not None
                or sampling_metadata.logprob_token_ids
            ):
                cdist = build_compact_for_logprobs(logits, k)
            return greedy, cdist

        assert sampling_metadata.temperature is not None

        # Greedy for mixed batches (computed but only used via torch.where)
        greedy_sampled = None
        if not sampling_metadata.all_random:
            greedy_sampled = self.greedy_sample(logits)

        # Prepare per-request params for force_topk_sample.
        # top_p / top_k may be None when no request uses them; provide defaults.
        device = logits.device
        B = logits.shape[0]
        top_p = sampling_metadata.top_p
        if top_p is None:
            top_p = torch.ones(B, dtype=torch.float32, device=device)
        top_k = sampling_metadata.top_k
        if top_k is None:
            top_k = torch.full((B,), -1, dtype=torch.int32, device=device)
        min_p = self._extract_min_p(sampling_metadata)

        logger.warning(  # TODO: remove after debugging
            "[force_topk] random branch: top_p=%s, top_k=%s, min_p=%s, generators=%d",
            "None" if sampling_metadata.top_p is None else f"{top_p.shape}",
            "None" if sampling_metadata.top_k is None else f"{top_k.shape}",
            "None" if min_p is None else f"{min_p.shape}",
            len(sampling_metadata.generators),
        )
        sampled, cdist = force_topk_sample(
            logits,
            sampling_metadata.temperature,
            top_p,
            top_k,
            min_p,
            sampling_metadata.generators,
            k,
        )

        # Mixed batch: select greedy for temperature < eps rows
        if greedy_sampled is not None:
            sampled = torch.where(
                sampling_metadata.temperature < _SAMPLING_EPS,
                greedy_sampled,
                sampled,
                out=greedy_sampled,
            )
        logger.warning(  # TODO: remove after debugging
            "[force_topk] sample() done: sampled.shape=%s, cdist=%s",
            sampled.shape, "CompactDist" if cdist is not None else "None",
        )
        return sampled, cdist

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor | CompactDist,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """Override gather_logprobs() to handle CompactDist.

        When logprobs is a CompactDist (force_topk path), use O(k) compact
        queries. Otherwise delegate to the upstream full-vocab implementation.
        """
        if isinstance(logprobs, CompactDist):
            logger.warning("[force_topk] gather_logprobs: compact path, n=%d", num_logprobs)  # TODO: remove after debugging
            return AscendSampler._gather_logprobs_compact(
                logprobs, num_logprobs, token_ids
            )
        logger.warning("[force_topk] gather_logprobs: standard path, n=%d", num_logprobs)  # TODO: remove after debugging
        return Sampler.gather_logprobs(logprobs, num_logprobs, token_ids)

    @staticmethod
    def _gather_logprobs_compact(
        cdist: CompactDist,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """Gather logprobs from a CompactDist (force_topk path).

        Replaces the upstream [B,V] topk+gather+rank with O(k) operations:
          - topn(n): zero-cost slice (already sorted)
          - gather(sampled): O(k) linear scan
          - rank: O(k) position lookup
        """
        assert token_ids.dtype == torch.int64
        logger.warning(  # TODO: remove after debugging
            "[force_topk] _gather_logprobs_compact: n=%d, cdist.k=%d",
            num_logprobs, cdist.token_index.shape[1],
        )
        # top-n logprobs and token ids (already descending → slice)
        topk_logprobs, topk_indices = cdist.topn(num_logprobs)   # [B, n]

        # Sampled token's logprob
        token_logprobs = cdist.gather(token_ids)                 # [B]

        # Rank: position of sampled token in token_index (0-based).
        # Miss (not in top-k) → rank = k (approximate, design §4.5).
        hit = cdist.token_index == token_ids.long().unsqueeze(-1)  # [B, k]
        token_ranks = hit.long().argmax(dim=-1)                   # [B]
        k = cdist.token_index.shape[1]
        token_ranks = torch.where(
            hit.any(dim=-1),
            token_ranks,
            torch.full_like(token_ranks, k),
        )

        # Concatenate: sampled token (col 0) + top-k (cols 1..n)
        indices = torch.cat((token_ids.unsqueeze(-1), topk_indices), dim=1)
        logprobs_cat = torch.cat(
            (token_logprobs.unsqueeze(-1), topk_logprobs), dim=1
        )
        indices = indices.to(torch.int32)
        return LogprobsTensors(indices, logprobs_cat, token_ranks)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p
        self.top_k = None

    def set_q_event(self, q, event):
        # Pass in async exponential results.
        # Also pass in event to prevent synchronize errors.
        self.q = q
        self.async_event = event

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
            if get_ascend_config().enable_async_exponential:
                # Add synchronize to prevent synchronize error.
                logger.debug_once(
                    "[sample/sampler] Using async-exponential sampling path. "
                    "Pre-computed exponential randoms from separate stream will be used.",
                )
                self.async_event.synchronize()
                return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
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


def _apply_top_k_top_p_ascendc(
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
            gathered_vals = torch.ops._C_ascend.npu_apply_top_k_top_p(gathered_vals, k=k, p=p)
        return gathered_vals, gathered_idx

    if p is None and k is None:
        return logits
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)
