# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor, UvaBuffer

from vllm_ascend.worker.v2.states import AscendRequestState


class Ascend310PStagedWriteTensor:
    """CPU-owned replacement for vLLM's Triton-backed staged writes."""

    # TODO: Refactor this 310P implementation to use Triton Dispatcher after
    # vLLM RFC #45133 lands.

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ) -> None:
        del max_concurrency
        self.dtype = dtype
        self.device = device
        self.uva_instead_of_gpu = uva_instead_of_gpu
        self._dirty_indices: set[int] = set()
        if uva_instead_of_gpu:
            self._uva_buffer = UvaBuffer(size, dtype)
            self.cpu = self._uva_buffer.cpu
            self.np = self._uva_buffer.np
            self.gpu = self._uva_buffer.uva
        else:
            self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
            self.np = self.cpu.numpy()
            self.gpu = torch.zeros(size, dtype=dtype, device=device)

    def stage_write(
        self,
        index: int,
        start: int,
        values: Iterable[int] | Iterable[float],
    ) -> None:
        values = list(values)
        if values:
            self.np[index, start : start + len(values)] = values
            self._dirty_indices.add(index)

    def stage_write_elem(self, index: int, value: int | float) -> None:
        self.np[index] = value
        self._dirty_indices.add(index)

    def apply_write(self) -> None:
        if not self._dirty_indices:
            return
        if self.uva_instead_of_gpu:
            self.gpu = self._uva_buffer.uva
            self._dirty_indices.clear()
            return
        indices = sorted(self._dirty_indices)
        indices_device = torch.tensor(indices, dtype=torch.int64, device=self.device)
        values = self.cpu[indices].to(self.device, non_blocking=True)
        self.gpu.index_copy_(0, indices_device, values)
        self._dirty_indices.clear()


class Ascend310PRequestState(AscendRequestState):
    """MRV2 request state using the same CPU-owner model as MRV1 310P."""

    # TODO: Refactor staged writes to use Triton Dispatcher after vLLM RFC
    # #45133 lands while retaining the 310P implementation.

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))
        self.all_token_ids = Ascend310PStagedWriteTensor(
            (max_num_reqs, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.prompt_len = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.total_len = Ascend310PStagedWriteTensor(max_num_reqs, dtype=torch.int32, device=device)
        self.num_computed_prefill_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = Ascend310PStagedWriteTensor(max_num_reqs, dtype=torch.int32, device=device)
        self.num_computed_tokens_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32, device="cpu")
        self.last_sampled_tokens = torch.zeros(max_num_reqs, 1, dtype=torch.int64, device=device)
        self.max_seq_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.draft_tokens = torch.zeros(
            max_num_reqs,
            num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int | None = None,
    ) -> None:
        super().add_request(
            req_id,
            prompt_len,
            all_token_ids,
            num_computed_tokens,
            max_tokens=max_tokens,
        )
        req_idx = self.req_id_to_index[req_id]
        self.num_computed_tokens_cpu[req_idx] = num_computed_tokens
        if num_computed_tokens > 0:
            self.last_sampled_tokens[req_idx, 0] = all_token_ids[num_computed_tokens - 1]
