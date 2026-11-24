# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/async_utils.py
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

import torch
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import async_utils as _vllm_async_utils
from vllm.v1.worker.gpu.async_utils import AsyncOutput, async_copy_to_np, stream
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.utils import raise_if_nan_logits

# ---------------------------------------------------------------------------
# Pinned copy cache for AsyncOutput D2H staging
# ---------------------------------------------------------------------------


class _AsyncCopyCache:
    def __init__(self, cap_rows: int = 512, cap_cols: int = 32):
        self.cap_rows = cap_rows
        self.cap_cols = cap_cols
        self.copy_event = torch.Event(blocking=True)
        self.sampled_flat = torch.empty(
            cap_rows * cap_cols,
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        self.num_sampled_pinned = torch.empty(
            cap_rows,
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        self.in_flight = False


_ASYNC_OUTPUT_CACHES: dict[int, _AsyncCopyCache] = {}
_orig_async_output_init = AsyncOutput.__init__
_orig_async_output_get_output = AsyncOutput.get_output


def _patched_async_output_init(
    self: AsyncOutput,
    model_runner_output: ModelRunnerOutput,
    sampler_output: SamplerOutput,
    num_sampled_tokens: torch.Tensor,
    main_stream: torch.Stream,
    copy_stream: torch.Stream,
    check_ep_fault: bool,
    pending_aux_output,
    num_drafted_tokens_per_req: torch.Tensor | None = None,
    num_verified_tokens_per_req: torch.Tensor | None = None,
) -> None:
    self.model_runner_output = model_runner_output
    self.sampler_output = sampler_output
    self.num_sampled_tokens = num_sampled_tokens
    self.pending_aux_output = pending_aux_output
    self.num_drafted_tokens_per_req = num_drafted_tokens_per_req
    self.num_verified_tokens_per_req = num_verified_tokens_per_req
    self._has_fault = None
    self._has_fault_gpu = None
    if check_ep_fault:
        self._has_fault_gpu = get_ep_all2all_manager().query_fault()

    src_sampled = sampler_output.sampled_token_ids
    stream_key = id(copy_stream)
    cache = _ASYNC_OUTPUT_CACHES.get(stream_key)
    if cache is None:
        cache = _AsyncCopyCache()
        _ASYNC_OUTPUT_CACHES[stream_key] = cache

    use_pinned_cache = (
        not cache.in_flight
        and src_sampled.ndim == 2
        and num_sampled_tokens.ndim == 1
        and src_sampled.shape[0] > 0
        and src_sampled.shape[0] <= cache.cap_rows
        and src_sampled.shape[1] <= cache.cap_cols
        and src_sampled.dtype == torch.int64
        and num_sampled_tokens.dtype == torch.int32
        and src_sampled.is_contiguous()
        and num_sampled_tokens.is_contiguous()
    )
    if use_pinned_cache:
        cache.in_flight = True
        self._copy_cache = cache
        self.copy_event = cache.copy_event
    else:
        self._copy_cache = None
        self.copy_event = torch.Event(blocking=True)

    try:
        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            if use_pinned_cache:
                n_rows, n_cols = src_sampled.shape
                dst_sampled = cache.sampled_flat[: n_rows * n_cols].view(n_rows, n_cols)
                dst_sampled.copy_(src_sampled, non_blocking=True)
                self.sampled_token_ids = dst_sampled.numpy()
            else:
                self.sampled_token_ids = async_copy_to_np(src_sampled)
            self.logprobs_tensors = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = sampler_output.logprobs_tensors.to_cpu_nonblocking()
            self.num_nans = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            if use_pinned_cache:
                n_rows = src_sampled.shape[0]
                dst_num = cache.num_sampled_pinned[:n_rows]
                dst_num.copy_(num_sampled_tokens, non_blocking=True)
                self.num_sampled_tokens_np = dst_num.numpy()
            else:
                self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.num_drafted_tokens_per_req_np = (
                async_copy_to_np(num_drafted_tokens_per_req) if num_drafted_tokens_per_req is not None else None
            )
            self.num_verified_tokens_per_req_np = (
                async_copy_to_np(num_verified_tokens_per_req) if num_verified_tokens_per_req is not None else None
            )
            self.sampling_mask_tensors = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_tensors = sampler_output.sampling_mask_tensors.to_cpu_nonblocking()
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            token_id_logprobs = self.model_runner_output.prompt_token_id_logprobs_dict
            self.prompt_token_id_logprobs_dict = {
                k: v.to("cpu", non_blocking=True) for k, v in token_id_logprobs.items()
            }
            if self.pending_aux_output is not None:
                self.pending_aux_output.enqueue_cpu_copy(
                    num_sampled=self.num_sampled_tokens_np,
                    num_rejected=async_copy_to_np(sampler_output.num_rejected),
                )
            if self._has_fault_gpu is not None:
                self._has_fault = self._has_fault_gpu.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)
    except Exception:
        if self._copy_cache is not None:
            self._copy_cache.in_flight = False
            self._copy_cache = None
        raise


def _patched_async_output_get_output(self: AsyncOutput) -> ModelRunnerOutput:
    try:
        self.copy_event.synchronize()

        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids
        if self.num_drafted_tokens_per_req_np is not None:
            assert self.num_verified_tokens_per_req_np is not None
            self.model_runner_output.num_drafted_tokens_per_req = self.num_drafted_tokens_per_req_np.tolist()
            self.model_runner_output.num_verified_tokens_per_req = self.num_verified_tokens_per_req_np.tolist()

        if self.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = self.sampling_mask_tensors.tolists(self.num_sampled_tokens_np)

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        self.model_runner_output.prompt_token_id_logprobs_dict = self.prompt_token_id_logprobs_dict
        if self.pending_aux_output is not None:
            self.model_runner_output.aux_output_connector_output = self.pending_aux_output.process_output()

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output
    finally:
        if self._copy_cache is not None:
            self._copy_cache.in_flight = False
            self._copy_cache = None


if not hasattr(_vllm_async_utils, "_ASYNC_OUTPUT_CACHES") and not getattr(
    AsyncOutput.__init__, "_ascend_async_cache_patched", False
):
    _patched_async_output_init._ascend_async_cache_patched = True  # type: ignore[attr-defined]
    AsyncOutput.__init__ = _patched_async_output_init  # type: ignore[method-assign]
    AsyncOutput.get_output = _patched_async_output_get_output  # type: ignore[method-assign]
