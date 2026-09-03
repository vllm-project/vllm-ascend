#
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
# See the License for the specific language govserning permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import vllm.envs as envs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)
from vllm_ascend.utils import global_stream, npu_stream_switch

_CPU_GENERATOR_CACHE_310P: dict[int, tuple[torch.Generator, torch.Generator]] = {}


def _prepare_cpu_generators_310p(
    generators: dict[int, torch.Generator],
) -> dict[int, torch.Generator]:
    """Return CPU RNGs while preserving requests across batch reordering."""
    cached_by_source = {
        id(source): (source, cpu_generator) for source, cpu_generator in _CPU_GENERATOR_CACHE_310P.values()
    }
    prepared: dict[int, torch.Generator] = {}
    next_cache: dict[int, tuple[torch.Generator, torch.Generator]] = {}

    for request_index, source in generators.items():
        cache_entry = cached_by_source.get(id(source))
        if cache_entry is None or cache_entry[0] is not source:
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(source.initial_seed())
        else:
            cpu_generator = cache_entry[1]

        prepared[request_index] = cpu_generator
        next_cache[request_index] = (source, cpu_generator)

    _CPU_GENERATOR_CACHE_310P.clear()
    _CPU_GENERATOR_CACHE_310P.update(next_cache)
    return prepared


def _generate_request_uniforms_310p(
    batch_size: int,
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Generate one uniform value per request on pinned CPU memory."""
    uniforms = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device="cpu",
        pin_memory=True,
    )
    for request_index, cpu_generator in _prepare_cpu_generators_310p(generators).items():
        uniforms[request_index] = torch.rand((), dtype=torch.float32, generator=cpu_generator)

    # Exact zero would select a zero-probability prefix in inverse CDF.
    uniforms.clamp_min_(torch.finfo(torch.float32).tiny)
    return uniforms.to(device, non_blocking=True)


def _sample_from_cdf_310p(
    weights: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Sample rows of non-negative weights with 310P-supported NPU ops."""
    cdf = weights.cumsum(dim=-1, dtype=torch.float32)
    thresholds = uniforms.unsqueeze(-1) * cdf[..., -1:]
    return torch.searchsorted(cdf, thresholds, right=True).squeeze(-1)


def fill_exponential_310p(
    reference: torch.Tensor,
    generators: dict[int, torch.Generator],
    active_mask: list[bool] | None = None,
) -> torch.Tensor:
    """Generate exponential values on CPU and transfer them to NPU."""
    batch_size = reference.shape[0]
    cpu_generators = _prepare_cpu_generators_310p(generators)
    needs_default_values = active_mask is not None or len(generators) != batch_size

    if needs_default_values:
        uniforms = torch.rand(
            reference.shape,
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
    else:
        uniforms = torch.empty(
            reference.shape,
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

    for request_index, cpu_generator in cpu_generators.items():
        if active_mask is not None and not active_mask[request_index]:
            continue
        uniforms[request_index] = torch.rand(
            reference.shape[1:],
            dtype=torch.float32,
            generator=cpu_generator,
        )

    uniforms.clamp_min_(torch.finfo(torch.float32).tiny)
    exponential = -torch.log(uniforms)
    return exponential.to(
        device=reference.device,
        dtype=reference.dtype,
        non_blocking=True,
    )


def _random_sample_310p(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """
    310P does not support the required NPU random-generation path.
    The previous implementation generated [batch, vocab] random values
    on the CPU and copied them to the NPU, causing performance degradation on
    small models and RC devices. This implementation generates only one CPU
    random value per request and performs inverse-CDF sampling on the NPU,
    reducing CPU computation and H2D transfer overhead.
    """
    with npu_stream_switch(global_stream()):
        uniforms = _generate_request_uniforms_310p(
            probs.shape[0],
            generators,
            probs.device,
        )

    torch.npu.current_stream().wait_stream(global_stream())
    sampled = _sample_from_cdf_310p(probs, uniforms)
    return sampled.view(-1)


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)
        if get_ascend_config().enable_reduce_sample:
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = torch.softmax(cand_logits, dim=-1)
            pos = _random_sample_310p(probs, generators)  # [B]

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
            return _random_sample_310p(probs, generators), logits_to_return


class AscendSampler310(AscendSampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)
