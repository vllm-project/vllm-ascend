# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.sample.sampler import AscendSampler310, _random_sample_310p
from vllm_ascend.sample.sampler import apply_top_k_top_p


class Ascend310PSampler:
    """Triton-free sampler for 310P MRV2."""

    def __init__(self) -> None:
        self.penalties_state = SimpleNamespace(output_bin_counts=None)
        self.sampling_params: dict[int, SamplingParams] = {}
        self.generators: dict[int, torch.Generator] = {}

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        del prompt_len
        unsupported = []
        if sampling_params.repetition_penalty != 1.0:
            unsupported.append("repetition_penalty")
        if sampling_params.presence_penalty != 0.0 or sampling_params.frequency_penalty != 0.0:
            unsupported.append("presence/frequency penalty")
        if sampling_params.logprobs is not None or sampling_params.prompt_logprobs is not None:
            unsupported.append("logprobs")
        if (
            getattr(sampling_params, "bad_words", None)
            or getattr(sampling_params, "logit_bias", None)
            or getattr(sampling_params, "allowed_token_ids", None)
        ):
            unsupported.append("logits processors")
        if unsupported:
            # TODO: Support additional sampling features in the next 310P MRV2 iteration.
            raise NotImplementedError(
                f"Unsupported sampling parameters on model runner v2 for 310P: {', '.join(unsupported)}."
            )
        self.sampling_params[req_idx] = sampling_params
        if sampling_params.seed is None:
            self.generators.pop(req_idx, None)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sampling_params.seed)
            self.generators[req_idx] = generator

    def apply_staged_writes(self) -> None:
        pass

    def __call__(self, logits: torch.Tensor, input_batch) -> SamplerOutput:
        idx_mapping_np = input_batch.idx_mapping_np[: input_batch.num_reqs]
        params = [self.sampling_params[int(req_idx)] for req_idx in idx_mapping_np]

        temperatures = torch.tensor(
            [param.temperature for param in params],
            dtype=torch.float32,
            device=logits.device,
        )
        greedy_mask = temperatures == 0
        if all(param.temperature == 0 for param in params):
            sampled = AscendSampler310.greedy_sample(logits[: input_batch.num_reqs]).to(torch.int32)
            return self._build_output(sampled, input_batch)

        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        processed_logits = logits[: input_batch.num_reqs].to(torch.float32).clone()
        processed_logits = processed_logits / safe_temperatures.unsqueeze(-1)

        min_p = torch.tensor(
            [param.min_p for param in params],
            dtype=torch.float32,
            device=logits.device,
        )
        if any(param.min_p != 0.0 for param in params):
            probs = processed_logits.softmax(dim=-1)
            min_p_thresholds = probs.max(dim=-1, keepdim=True).values * min_p.unsqueeze(-1)
            processed_logits.masked_fill_(probs < min_p_thresholds, -float("inf"))

        vocab_size = processed_logits.shape[-1]
        top_k = torch.tensor(
            [param.top_k if 0 < param.top_k < vocab_size else vocab_size for param in params],
            dtype=torch.int32,
            device=logits.device,
        )
        top_p = torch.tensor(
            [param.top_p for param in params],
            dtype=torch.float32,
            device=logits.device,
        )
        k = top_k if any(param.top_k not in (-1, 0) and param.top_k < vocab_size for param in params) else None
        p = top_p if any(param.top_p != 1.0 for param in params) else None
        filtered = apply_top_k_top_p(processed_logits, k, p)
        candidate_ids = None
        if isinstance(filtered, tuple):
            processed_logits, candidate_ids = filtered
        else:
            processed_logits = filtered

        generators = {
            batch_idx: self.generators[int(req_idx)]
            for batch_idx, req_idx in enumerate(idx_mapping_np)
            if int(req_idx) in self.generators
        }
        sampled = _random_sample_310p(processed_logits.softmax(dim=-1), generators)
        sampled = torch.where(greedy_mask, processed_logits.argmax(dim=-1), sampled)
        if candidate_ids is not None:
            sampled = candidate_ids.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
        sampled = sampled.to(torch.int32)
        return self._build_output(sampled, input_batch)

    @staticmethod
    def _build_output(sampled: torch.Tensor, input_batch) -> SamplerOutput:
        num_sampled = input_batch.seq_lens.new_ones(input_batch.num_reqs)
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=torch.zeros_like(num_sampled),
        )
