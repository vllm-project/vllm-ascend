# SPDX-License-Identifier: Apache-2.0
"""Two-pass FP32 greedy sampling for eager HY4 Prefill."""

import numpy as np
import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@triton.jit
def _hy4_prefill_argmax_parts(
    logits,
    mapping,
    indices,
    positions,
    min_lens,
    stop_counts,
    stop_ids,
    partial_values,
    partial_indices,
    VOCAB: tl.constexpr,
    STRIDE: tl.constexpr,
    STOP_STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    MASK_STOP: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(logits + row * STRIDE + col, col < VOCAB, -float("inf"))
    if MASK_STOP:
        req = tl.load(mapping + row)
        index = tl.load(indices + row)
        pos = tl.load(positions + index)
        min_len = tl.load(min_lens + req)
        if pos + 1 < min_len:
            count = tl.load(stop_counts + req)
            for i in range(count):
                stop = tl.load(stop_ids + req * STOP_STRIDE + i)
                value = tl.where(col == stop, -float("inf"), value)
    best, index = tl.max(value, axis=0, return_indices=True)
    tl.store(partial_values + row * PARTS + part, best)
    tl.store(partial_indices + row * PARTS + part, part * BLOCK + index)


@triton.jit
def _hy4_prefill_argmax_finish(
    partial_values,
    partial_indices,
    sampled,
    mapping,
    seq_lens,
    prefill_lens,
    num_sampled,
    num_rejected,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FUSE_COUNTS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(partial_values + row * PARTS + col, col < PARTS, -float("inf"))
    part = tl.argmax(value, axis=0)
    token = tl.load(partial_indices + row * PARTS + part)
    tl.store(sampled + row, token.to(tl.int64))
    if FUSE_COUNTS:
        req = tl.load(mapping + row)
        seq_len = tl.load(seq_lens + row)
        prefill_len = tl.load(prefill_lens + req)
        tl.store(num_sampled + row, (seq_len >= prefill_len).to(tl.int32))
        tl.store(num_rejected + row, 0)


def sample_prefill(logits, batch, bias, prefill_lens, mask_stop, fuse_counts):
    rows, vocab = logits.shape
    block = 8192
    parts = triton.cdiv(vocab, block)
    values = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, parts), dtype=torch.int32, device=logits.device)
    sampled = torch.empty(rows, dtype=torch.int64, device=logits.device)
    num_sampled = torch.empty_like(batch.seq_lens[:rows]) if fuse_counts else None
    num_rejected = torch.empty_like(num_sampled) if fuse_counts else None
    _hy4_prefill_argmax_parts[(rows, parts)](
        logits,
        batch.idx_mapping,
        batch.logits_indices,
        batch.positions,
        bias.min_lens.gpu,
        bias.num_stop_token_ids.gpu,
        bias.stop_token_ids.gpu,
        values,
        indices,
        vocab,
        logits.stride(0),
        bias.stop_token_ids.gpu.stride(0),
        parts,
        block,
        mask_stop,
    )
    _hy4_prefill_argmax_finish[(rows,)](
        values,
        indices,
        sampled,
        batch.idx_mapping,
        batch.seq_lens,
        prefill_lens,
        num_sampled,
        num_rejected,
        parts,
        triton.next_power_of_2(parts),
        fuse_counts,
    )
    return sampled, num_sampled, num_rejected


class PrefillSampler:
    """Keep upstream sampler state and fuse eligible greedy Prefill batches."""

    def __init__(self, sampler, *, fuse_counts):
        self._sampler = sampler
        self._fuse_counts = fuse_counts

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def __call__(self, logits, input_batch):
        reqs = input_batch.idx_mapping_np
        dims = self.get_logprobs_dims(reqs)
        if not self._can_sample_hy4_prefill(logits, input_batch, dims):
            return self._sampler(logits, input_batch)

        bias = self.logits_processors[0]
        sampled, num_sampled, num_rejected = sample_prefill(
            logits,
            input_batch,
            bias,
            self.req_states.prefill_len.gpu,
            bool(np.any(bias.num_stop_token_ids.np[reqs])),
            self._fuse_counts,
        )
        logprobs = None
        if dims is not None:
            num_logprobs, max_per_req_token_ids = dims
            logprobs = compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                None,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_per_req_token_ids,
                logits_mode=self.logprobs_mode == "raw_logits",
            )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=None,
        )

    def _can_sample_hy4_prefill(self, logits, batch, logprobs_dims):
        if (
            not getattr(batch, "hy4_pure_prefill", False)
            or self.compute_nans
            or self.trace_replay_state is not None
            or self.return_sampling_mask
            or (logprobs_dims is not None and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
            or len(self.logits_processors) != 3
            or logits.device.type != "npu"
            or logits.dtype != torch.float32
            or logits.stride(1) != 1
            or logits.shape[0] != batch.num_reqs
        ):
            return False

        req = batch.idx_mapping_np
        bias, penalties, bad_words = self.logits_processors
        return (
            np.all(self.sampling_states.temperature.np[req] == 0.0)
            and not np.any(self.sampling_states.min_p.np[req])
            and np.all(self.sampling_states.top_p.np[req] == 1.0)
            and np.all(self.sampling_states.top_k.np[req] == self.sampling_states.vocab_size)
            and not np.any(bias.num_allowed_token_ids.np[req])
            and not np.any(bias.num_logit_bias.np[req])
            and not np.any(bias.restore_when_all_masked.np[req])
            and not np.any(penalties.use_penalty[req])
            and not np.any(bad_words.num_bad_words.np[req])
            and (
                not self.thinking_budget_state.enabled
                or not np.any(self.thinking_budget_state.use_thinking_budget[req])
            )
        )
