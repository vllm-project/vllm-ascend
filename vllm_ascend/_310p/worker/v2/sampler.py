# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class Ascend310PSampler:
    """Triton-free sampler for 310P MRV2.

    Exposes a minimal ``sampling_states`` surface so MTP draft ``propose()``
    (and ``_dummy_run``) can read ``temperature.gpu`` / ``seeds.gpu`` without
    pulling in UVA-backed SamplingStates.
    """

    # TODO: Refactor this sampler to register 310P implementations through
    # Triton Dispatcher after vLLM RFC #45133 lands.

    def __init__(
        self,
        max_num_reqs: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        self.penalties_state = SimpleNamespace(output_bin_counts=None)
        self.max_num_reqs = max_num_reqs
        if device is None:
            device = torch.device("cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        # Greedy-only: temperature stays 0; seeds unused but must exist for MTP.
        temperature_gpu = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        seeds_gpu = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature_gpu),
            seeds=SimpleNamespace(gpu=seeds_gpu),
        )

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        del prompt_len
        unsupported = []
        if sampling_params.temperature != 0:
            unsupported.append("temperature")
        if sampling_params.top_p != 1.0:
            unsupported.append("top_p")
        if sampling_params.top_k not in (-1, 0):
            unsupported.append("top_k")
        if sampling_params.min_p != 0.0:
            unsupported.append("min_p")
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
        if 0 <= req_idx < self.max_num_reqs:
            self.sampling_states.temperature.gpu[req_idx] = 0.0
            seed = getattr(sampling_params, "seed", None)
            self.sampling_states.seeds.gpu[req_idx] = 0 if seed is None else int(seed)

    def apply_staged_writes(self) -> None:
        pass

    def __call__(self, logits: torch.Tensor, input_batch) -> SamplerOutput:
        sampled = logits.argmax(dim=-1).to(torch.int32)
        num_sampled = input_batch.seq_lens.new_ones(input_batch.num_reqs)
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=torch.zeros_like(num_sampled),
        )
