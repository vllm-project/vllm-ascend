# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Ascend-compatible DiffusionGemma model state.

``DiffusionGemmaModelState`` (in vllm/model_executor/models/diffusion_gemma.py)
was authored against upstream vLLM's attention-metadata API. On Ascend the V2
runner uses ``vllm_ascend.worker.v2.attn_utils.build_attn_metadata`` (different
signature, produces ``AscendMetadata``). This subclass keeps all of the
diffusion canvas/denoising logic and only re-implements ``prepare_attn`` so it:
  * builds Ascend attention metadata via the Ascend builder, and
  * stamps the per-request encoder(causal)/denoise(bidirectional) flag onto the
    resulting ``AscendMetadata.causal_per_req`` field, which the manual
    attention paths consume per request.
"""

from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaModelState,
    DiffusionSampler,
)
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


def _tp_sample_step(
    logits: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_idx: torch.Tensor,
    all_slots: torch.Tensor,
    valid_canvas_len: torch.Tensor,
    canvas: torch.Tensor,
    argmax_canvas: torch.Tensor,
    step_tensor: torch.Tensor,
    is_encoder_phase: torch.Tensor,
    confident_tensor: torch.Tensor,
    embed_weight: torch.Tensor,
    normalizer: torch.Tensor,
    history: torch.Tensor,
    history_len_tensor: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    draft_tokens: torch.Tensor,
    max_denoising_steps: float,
    t_min: float,
    t_max: float,
    confidence_threshold: float,
    vocab_size: int,
    canvas_length: int,
    stability_threshold: int,
    entropy_bound: float,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    added_weight_start_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    sampled.zero_()
    num_sampled.zero_()

    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    logits_3d = logits.reshape(num_decode, canvas_length, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)

    u = torch.rand_like(scaled).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    noisy = scaled + gumbel * (temp[:, None, None] > 0).float()
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, canvas_length)
    argmax_tokens = scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, canvas_length)

    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()

    token_entropy = -(probs * log_probs).sum(dim=-1)
    mean_entropy = token_entropy.mean(dim=-1)
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    is_commit = is_encoder_phase[decode_slots]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()
    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    random_tokens = torch.randint(
        0,
        vocab_size,
        (num_decode, canvas_length),
        device=device,
        dtype=canvas.dtype,
    )
    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)
    canvas[decode_slots] = torch.where(is_commit.unsqueeze(1), random_tokens, denoise_canvas)

    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % stability_threshold
    for i in range(stability_threshold):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(write_here, argmax_tokens, history[decode_slots, i])

    argmax_canvas[decode_slots] = torch.where(is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots])

    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    sampled[decode_idx] = argmax_canvas[decode_slots].to(sampled.dtype) * is_commit.unsqueeze(1).to(sampled.dtype)
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(num_sampled.dtype)

    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, stability_threshold):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= stability_threshold)) | (
        step_after >= max_denoising_steps
    )
    is_encoder_phase[decode_slots] = torch.where(is_commit, is_commit.new_zeros(num_decode), converged)

    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    org_probs = probs[..., org_vocab_start_index:org_vocab_end_index]
    org_weight = embed_weight[: org_vocab_end_index - org_vocab_start_index]
    local_soft_embeds = torch.matmul(org_probs.to(embed_weight.dtype), org_weight)

    if added_vocab_end_index > added_vocab_start_index:
        added_probs = probs[..., added_vocab_start_index:added_vocab_end_index]
        added_weight_end = added_weight_start_index + (added_vocab_end_index - added_vocab_start_index)
        added_weight = embed_weight[added_weight_start_index:added_weight_end]
        local_soft_embeds = local_soft_embeds + torch.matmul(added_probs.to(embed_weight.dtype), added_weight)
    local_soft_embeds = local_soft_embeds * normalizer

    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(newly_converged, argmax_canvas[decode_slots], canvas[decode_slots])

    draft_tokens[all_slots, :canvas_length] = canvas[all_slots]

    return scaled, local_soft_embeds, sc_keep


class AscendDiffusionSampler(DiffusionSampler):
    """DiffusionGemma sampler with TP-sharded soft embedding support."""

    def __init__(self, *args, embed_tokens: Any, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shard_indices = embed_tokens.shard_indices
        self.org_vocab_start_index = shard_indices.org_vocab_start_index
        self.org_vocab_end_index = shard_indices.org_vocab_end_index
        self.added_vocab_start_index = shard_indices.added_vocab_start_index
        self.added_vocab_end_index = shard_indices.added_vocab_end_index
        self.added_weight_start_index = shard_indices.num_org_elements_padded

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ):
        del draft_logits
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        states = self.diffusion_states
        canvas_length = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_copy_to_gpu(valid_canvas_len_np.astype(np.int64), device=device)

        if num_decode > 0 and valid_canvas_len_np.min() < canvas_length:
            ar = torch.arange(canvas_length, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)] * valid.reshape(-1, 1).to(logits.dtype)

        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        all_slots = input_batch.idx_mapping[:num_reqs]
        is_committing = states.is_encoder_phase[decode_slots].clone()

        scaled, local_soft_embeds, sc_keep = _tp_sample_step(
            logits,
            decode_slots,
            decode_idx,
            all_slots,
            valid_canvas_len,
            states.canvas,
            states.argmax_canvas,
            states.step,
            states.is_encoder_phase,
            states.confident,
            self.embed_weight,
            self.normalizer,
            states.accepted_canvas_history,
            states.accepted_canvas_history_len,
            sampled,
            num_sampled,
            self.req_states.draft_tokens,
            max_denoising_steps=float(states.max_denoising_steps),
            t_min=self.t_min,
            t_max=self.t_max,
            confidence_threshold=self.confidence_threshold,
            vocab_size=self.vocab_size,
            canvas_length=canvas_length,
            stability_threshold=states.stability_threshold,
            entropy_bound=self.entropy_bound,
            org_vocab_start_index=self.org_vocab_start_index,
            org_vocab_end_index=self.org_vocab_end_index,
            added_vocab_start_index=self.added_vocab_start_index,
            added_vocab_end_index=self.added_vocab_end_index,
            added_weight_start_index=self.added_weight_start_index,
        )

        soft_embeds = tensor_model_parallel_all_reduce(local_soft_embeds)
        states.self_conditioning_embeds[decode_slots] = (soft_embeds * sc_keep).to(
            states.self_conditioning_embeds.dtype
        )

        is_decode_np = per_req_nlogits_np > 0
        logprobs_tensors = None
        max_num_logprobs = self.sampling_states.max_num_logprobs(slots_np)
        if max_num_logprobs >= 0:
            converged_mask = states.is_encoder_phase[decode_slots]
            just_converged = converged_mask & ~is_committing
            if just_converged.any():
                flat_logits = scaled.reshape(-1, scaled.shape[-1])
                argmax_tokens = scaled.argmax(dim=-1)
                for local_idx in just_converged.nonzero(as_tuple=True)[0]:
                    li = local_idx.item()
                    slot = decode_slots[local_idx]
                    num_logits = int(valid_canvas_len_np[li])
                    start = li * canvas_length
                    self._pending_logprobs[slot.item()] = compute_topk_logprobs(
                        flat_logits[start : start + num_logits],
                        max_num_logprobs,
                        argmax_tokens[local_idx][:num_logits],
                    )

            if is_committing.any() and self._pending_logprobs:
                parts_ids, parts_lp, parts_ranks = [], [], []
                cu_gen: list[int] = []
                flat_offset = 0
                for i in range(num_reqs):
                    cu_gen.append(flat_offset)
                    slot = int(slots_np[i])
                    if is_decode_np[i] and slot in self._pending_logprobs:
                        lp = self._pending_logprobs.pop(slot)
                        parts_ids.append(lp.logprob_token_ids)
                        parts_lp.append(lp.logprobs)
                        parts_ranks.append(lp.selected_token_ranks)
                        flat_offset += lp.logprobs.shape[0]
                if parts_ids:
                    logprobs_tensors = LogprobsTensors(
                        logprob_token_ids=torch.cat(parts_ids),
                        logprobs=torch.cat(parts_lp),
                        selected_token_ranks=torch.cat(parts_ranks),
                        cu_num_generated_tokens=cu_gen,
                    )
        return self._build_output(input_batch, sampled, num_sampled, per_req_nlogits_np, device, logprobs_tensors)


class AscendDiffusionGemmaModelState(DiffusionGemmaModelState):
    """DiffusionGemma model state adapted to the Ascend V2 metadata API."""

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        diffusion_config = self.vllm_config.diffusion_config
        gen = self.gen_config
        sampler_cfg = gen.get("sampler_config") or {}
        if "EntropyBound" not in sampler_cfg.get("_cls_name", ""):
            raise ValueError("DiffusionGemma requires an EntropyBound sampler_config")
        entropy_bound = sampler_cfg.get("entropy_bound")
        if entropy_bound is None or entropy_bound <= 0:
            raise ValueError(f"entropy_bound must be a positive float (got {entropy_bound})")

        embed_tokens = self.model.model.embed_tokens
        return AscendDiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=gen["t_min"],
            t_max=gen["t_max"],
            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
            embed_weight=embed_tokens.weight,
            normalizer=self.model.model.normalizer,
            embed_tokens=embed_tokens,
        ), None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = int(input_batch.num_scheduled_tokens.max().item())

        # Per-request causal mode: encoder (commit) = causal, denoise =
        # bidirectional. Mirror DiffusionGemmaModelState.prepare_attn.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[slots]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal = self._causal_buf[:num_reqs]

        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=getattr(input_batch, "dcp_local_seq_lens", None),
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            for_cudagraph_capture=for_capture,
        )
        self.attn_metadata = attn_metadata

        # Stamp the per-request bidirectional/causal flag onto each layer's
        # AscendMetadata.
        #
        # `causal` is a view of the PERSISTENT, fixed-address `_causal_buf`
        # (updated in-place above via slice assignment), so a captured FULL
        # graph that reads it at replay sees the current step's per-request
        # phases -- the same persistence contract as `_persist_seqlens` /
        # `seq_lens_device`. We stamp it onto the new `causal_per_req` field,
        # which both the eager and capturable manual-attention paths consume
        # per request (encoder/commit == causal, denoise == bidirectional).
        #
        # `md.causal` (scalar) is kept as a coarse fallback for any consumer
        # that does not read `causal_per_req`: mark causal only if ALL requests
        # are in the encoder phase (any bidirectional request otherwise
        # dominates the batch-wide value). Correctness for mixed batches comes
        # from `causal_per_req`, not this scalar.
        batch_causal = bool(causal.all().item()) if causal.numel() > 0 else True
        for md in attn_metadata.values():
            md.causal = batch_causal
            md.causal_per_req = causal

        return attn_metadata
