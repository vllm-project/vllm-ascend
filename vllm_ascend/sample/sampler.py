import torch
import torch_npu
from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, global_stream

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

_SAMPLING_EPS = 1e-5


def generate_random_sequence(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    stream: torch.npu.Stream,
) -> torch.Tensor:
    with torch_npu.npu.stream(stream):
        q = torch.empty_like(logits, dtype=torch.float32)
        if len(generators) != logits.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(stream)
    return q


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
        self.dsa_stream = torch_npu.npu.Stream()

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
        logits_to_return = None
        res = None

        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        if p is not None:
            p = p.to(device=logits.device, dtype=torch.float32)
        else:
            p = torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)

        if k is not None:
            k = k.to(device=logits.device, dtype=torch.int32)
        else:
            k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]

        q = generate_random_sequence(logits, generators, self.dsa_stream).type(torch.float32)
        if get_ascend_config().enable_reduce_sample:
            local_vals, _ = torch.topk(logits, k=self.top_k, dim=-1)
            res = torch_npu.npu_top_k_top_p_sample(local_vals, k, p, q)
        else:
            res = torch_npu.npu_top_k_top_p_sample(logits, k, p, q)
        return res[0], logits_to_return


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
    if p is None and k is None:
        return logits

    logits = logits.type(torch.float32)
    if p is not None:
        p = p.type(torch.float32)
    else:
        p = torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)

    if k is not None:
        k = k.type(torch.int32)
    else:
        k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]

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
        gathered_vals = tp_group.all_gather(local_vals, dim=-1)  # [B, top_k*tp]
        gathered_idx = tp_group.all_gather(local_global_idx, dim=-1)  # [B, top_k*tp]

        gathered_vals = torch_npu.npu_top_k_top_p_sample(gathered_vals, k, p, is_need_logits=True)[1]
        return gathered_vals, gathered_idx
    else:
        return torch_npu.npu_top_k_top_p_sample(logits, k, p, is_need_logits=True)[1]


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)
