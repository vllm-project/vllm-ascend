#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from contextlib import contextmanager

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from vllm_ascend._310p.sample.sampler import fill_exponential_310p
from vllm_ascend.sample.rejection_sampler import (
    AscendRejectionSampler,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)


@contextmanager
def _bind_sample_recovered_tokens(fn):
    original = rejection_sampler_module.sample_recovered_tokens
    rejection_sampler_module.sample_recovered_tokens = fn
    try:
        yield
    finally:
        rejection_sampler_module.sample_recovered_tokens = original


class AscendRejectionSampler310(AscendRejectionSampler):
    """310P rejection sampler: PyTorch recovered-token path with CPU RNG (no Triton)."""

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        with _bind_sample_recovered_tokens(self.sample_recovered_tokens):
            return super().forward(metadata, draft_probs, logits, sampling_metadata)

    def sample_recovered_tokens(
        self,
        max_spec_len: int,
        num_draft_tokens: list[int],
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        device: torch.device,
        use_block_verify: bool = False,
        target_indices: torch.Tensor | None = None,
        global_vocab_size: int | None = None,
        enable_reduce_sampling: bool = False,
    ) -> torch.Tensor:
        batch_size = len(num_draft_tokens)
        vocab_size = target_probs.shape[-1]

        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
        has_draft_mask = num_draft_tensor > 0
        fill_exponential_310p(q, sampling_metadata.generators, has_draft_mask)

        recovered_token_ids = torch.empty_like(draft_token_ids)
        if use_block_verify:
            sample_recovered_tokens_blockwise_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        else:
            sample_recovered_tokens_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        return recovered_token_ids
