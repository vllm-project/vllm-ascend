import torch
import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import HAS_TRITON
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
        total_seq = probs.shape[0]
        if len(generators) == total_seq:
            # 全序列都有独立随机生成器：批量执行，无Python循环
            sorted_gens = [gen for _, gen in sorted(generators.items())]
            q.exponential_(generator=sorted_gens)
        else:
            # 全局批量采样打底
            global_batch_seed = torch.randint(1, 2**31 - 1, size=(1,))
            torch.npu.manual_seed(global_batch_seed)
            q.exponential_()
            logger.debug("[sample/sampler] random_sample batch_random_seed=%d", global_batch_seed)

            if len(generators):
                logger.warning_once(
                    "[sample/sampler] Partial sequences(%d/%d) have custom seed. "
                    "Global RNG used for remaining sequences, deterministic cannot be fully guaranteed.",
                    len(generators),
                    total_seq,
                )
                # 优化：排序后批量切片采样，干掉Python逐行循环
                sorted_items = sorted(generators.items())
                indices = [idx for idx, _ in sorted_items]
                gen_list = [gen for _, gen in sorted_items]
                q[indices].exponential_(generator=gen_list)
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
            if len(generators) == b_s:
                # 全序列都有独立随机生成器：批量执行，无Python循环
                sorted_gens = [gen for _, gen in sorted(generators.items())]
                q.exponential_(generator=sorted_gens)
            else:
                # 全局批量采样打底
                global_batch_seed = torch.randint(1, 2**31 - 1, size=(1,))
                torch.npu.manual_seed(global_batch_seed)
                q.exponential_()
                logger.debug(
                    "[sample/sampler] do_async_exponential random_sample batch_random_seed=%d", global_batch_seed
                )

                if len(generators):
                    # 混合场景一次性告警，避免刷屏
                    logger.warning_once(
                        "[sample/sampler] do_async_exponential Partial sequences(%d/%d) have custom seed. "
                        "Global RNG used for remaining sequences, deterministic cannot be fully guaranteed.",
                        len(generators),
                        b_s,
                    )
                    # 优化：排序后批量切片采样，干掉Python逐行循环
                    sorted_items = sorted(generators.items())
                    indices = [idx for idx, _ in sorted_items]
                    gen_list = [gen for _, gen in sorted_items]
                    q[indices].exponential_(generator=gen_list)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        if get_ascend_config().enable_reduce_sample:
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
        # logger.info(
        #     "[sample/sampler] k=%s, p=%s",
        #     k,
        #     p,
        # )
        if k is not None and (k < 1).all():
            logger.error("[sample/sampler] Invalid top-k value. Must be a positive integer.")
        if p is not None and ((p <= 0).any() or (p > 1).any()):
            logger.error("[sample/sampler] Invalid top-p value. Must be between 0 and 1.")
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)

        if get_ascend_config().enable_reduce_sample:
            if (
                self.top_k is not None
                and k is not None
                and bool((k > self.top_k).any().item())
            ):
                logger.warning_once(
                    "[sample/sampler] local top_k=%d < requested max_top_k=%d, potential Top-K error may occur.",
                    self.top_k,
                    int(k.max().item()),
                )
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = torch.softmax(cand_logits, dim=-1)
            pos = random_sample(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B] global token id
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
        world_size = tp_group.world_size
        rank = tp_group.rank_in_group
        V_global = V_local * world_size

        local_vals, local_idx = torch.topk(logits, k=top_k, dim=-1)  # [B, top_k], [B, top_k]
        local_global_idx = local_idx + rank * V_local  # [B, top_k]

        gathered_vals = tp_group.all_gather(local_vals, dim=-1)  # k_for_topk * g_n
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)  # [B, top_k*tp]

        full_logits = logits.new_full((B, V_global), -float("inf"))
        full_logits.scatter_(dim=-1, index=gathered_idx, src=gathered_vals)

        if p is None and k is None:
            return full_logits
        probs = full_logits.softmax(dim=-1)
        probs_sort, _ = probs.sort(dim=-1, descending=False)
        if k is not None:
            num_gathered_vocabs = gathered_vals.shape[1]  # 等价 probs_sort.size(1)
            kk = k.to(torch.long).clamp(min=1, max=num_gathered_vocabs)
            top_k_count = (num_gathered_vocabs - kk).unsqueeze(1)  # [B,1]
            top_k_cutoff = probs_sort.gather(-1, top_k_count)
            no_top_k_mask = (kk == V_global).unsqueeze(1)
            top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))
            elements_to_discard = probs < top_k_cutoff
            full_logits.masked_fill_(elements_to_discard, -float("inf"))
        if p is not None:
            cumprob = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = cumprob <= (1 - p.unsqueeze(1))
            top_p_mask[:, -1] = False  # at least one
            top_p_count = top_p_mask.sum(dim=-1, keepdim=True)
            top_p_cutoff = probs_sort.gather(-1, top_p_count)
            elements_to_discard = probs < top_p_cutoff
            full_logits.masked_fill_(elements_to_discard, -float("inf"))
        return full_logits
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

        local_vals, local_idx = torch.topk(logits, k=top_k, dim=-1)  # [B, top_k], [B, top_k]

        local_global_idx = local_idx + rank * V_local  # [B, top_k]

        gathered_vals = tp_group.all_gather(local_vals, dim=-1)  # [B, top_k*tp]
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)  # [B, top_k*tp]

        if p is None and k is None:
            return logits
        gathered_vals = torch.ops._C_ascend.npu_apply_top_k_top_p(gathered_vals, k=k, p=p)
        return gathered_vals, gathered_idx
    else:
        if p is None and k is None:
            return logits
        return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)
