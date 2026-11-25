# SPDX-License-Identifier: Apache-2.0
"""Bind NOVA window ownership to V2's per-step request order."""

import torch
from vllm.utils.torch_utils import async_tensor_h2d

from vllm_ascend.attention.nova_request_state import NovaRequestState


class NovaKVRuntime:
    def __init__(self, capacity: int, device: torch.device):
        self.state = NovaRequestState(capacity)
        self.request_slots = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        self._pending_request_ids: list[str] = []

    def prepare(self, input_batch, num_reqs: int, *, for_capture: bool = False) -> dict:
        if not input_batch.num_reqs <= num_reqs <= self.state.capacity:
            raise ValueError("NOVA request metadata exceeds the window capacity")
        dummy = for_capture or input_batch.is_dummy
        slots = []
        if not dummy:
            request_ids = input_batch.req_ids[: input_batch.num_reqs]
            self._pending_request_ids = list(request_ids)
            slots = self.state.prepare(
                request_ids,
                input_batch.num_computed_tokens_np[: input_batch.num_reqs],
                input_batch.num_scheduled_tokens[: input_batch.num_reqs],
            )
        slots = slots + [-1] * (num_reqs - len(slots))
        # Stage a fresh host snapshot; graph replay always reads the same device
        # allocation, even after a batch shrinks or changes request order.
        async_tensor_h2d(slots + [-1] * (self.state.capacity - num_reqs), out=self.request_slots)
        return {"nova_request_slots": self.request_slots[:num_reqs], "nova_request_slots_cpu": slots}

    def finish_execution(self, *, failed: bool) -> None:
        if failed:
            self.state.release(self._pending_request_ids)
        self._pending_request_ids = []
