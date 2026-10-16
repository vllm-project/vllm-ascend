# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side token history for Qwen4Exp PLE."""

from collections.abc import Sequence

import torch


class Qwen4ExpPLEHistory:
    """Keep PLE history independent of asynchronous CPU token placeholders.

    Slots belong to request IDs rather than persistent-batch row indices. The
    runner seeds newly scheduled or resumed requests with an authoritative CPU
    prefix, then records the corrected device input IDs before every forward.
    ``valid_ends`` describes a contiguous, real token prefix; positions outside
    that prefix must never be used to form context.

    Updates run on the runner's current device stream, before model execution.
    No generated token is copied back to the CPU, and no device scalar is read
    while preparing an input batch.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_tokens: int,
        context_len: int,
        eos_token_id: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        if min(max_num_reqs, max_model_len, max_num_tokens, context_len) <= 0:
            raise ValueError("Qwen4Exp PLE history dimensions must be positive")
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_tokens = max_num_tokens
        self.context_len = context_len
        self.eos_token_id = eos_token_id
        self.device = torch.device(device)
        self.pin_memory = pin_memory

        self.token_ids = torch.empty((max_num_reqs, max_model_len), dtype=torch.int32, device=self.device)
        self.device = self.token_ids.device
        # Separate from the model runner's captured output buffer: its graph
        # padding helper fills that buffer before copying the runtime context.
        self.context_buffer = torch.empty((max_num_reqs, context_len), dtype=torch.int32, device=self.device)
        self.req_slots: dict[str, int] = {}
        self.valid_ends: dict[str, int] = {}
        self._free_slots = list(reversed(range(max_num_reqs)))

        self._slots = torch.empty(max_num_reqs, dtype=torch.int64, device=self.device)
        self._context_ends = torch.empty(max_num_reqs, dtype=torch.int64, device=self.device)
        self._token_offsets = torch.arange(max_num_tokens, dtype=torch.int64, device=self.device)
        self._context_offsets = torch.arange(-context_len, 0, dtype=torch.int64, device=self.device)

    def has_request(self, req_id: str) -> bool:
        return req_id in self.req_slots

    def retain_requests(self, req_ids: Sequence[str]) -> None:
        """Release slots for finished, preempted, or unscheduled requests.

        A request returning after release must be initialized from the
        scheduler's real token snapshot, rather than the cached CPU placeholders.
        """
        active_req_ids = set(req_ids)
        if len(active_req_ids) != len(req_ids):
            raise ValueError("Qwen4Exp PLE request IDs must be unique")
        if len(active_req_ids) > self.max_num_reqs:
            raise ValueError("Qwen4Exp PLE batch exceeds history capacity")
        for req_id in list(self.req_slots):
            if req_id not in active_req_ids:
                self._free_slots.append(self.req_slots.pop(req_id))
                self.valid_ends.pop(req_id)

    def initialize_request(
        self,
        req_id: str,
        token_ids: Sequence[int],
        num_computed_tokens: int,
    ) -> None:
        """Seed the prefix preceding this request's next model input.

        Only ``token_ids[:num_computed_tokens]`` is consumed: the current input
        and later output positions are recorded from the device instead. Calling
        this for an existing ID explicitly replaces that request's valid prefix.
        """
        if not 0 <= num_computed_tokens <= self.max_model_len:
            raise ValueError("Qwen4Exp PLE prefix exceeds history bounds")
        if len(token_ids) < num_computed_tokens:
            raise ValueError("Qwen4Exp PLE initialization requires a complete prefix")
        prefix = token_ids[:num_computed_tokens]
        if any(token_id < 0 for token_id in prefix):
            raise ValueError("Qwen4Exp PLE initialization contains a placeholder token")

        slot = self.req_slots.get(req_id)
        if slot is None:
            if not self._free_slots:
                raise ValueError("Qwen4Exp PLE history has no free request slot")
            slot = self._free_slots.pop()
            self.req_slots[req_id] = slot
        if num_computed_tokens:
            prefix_cpu = torch.tensor(prefix, dtype=torch.int32, pin_memory=self.pin_memory)
            self.token_ids[slot, :num_computed_tokens].copy_(prefix_cpu, non_blocking=True)
        self.valid_ends[req_id] = num_computed_tokens

    def prepare_inputs(
        self,
        req_ids: Sequence[str],
        context_ends: Sequence[int],
        query_lens: Sequence[int],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        context_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record real model inputs and gather their preceding n-gram context.

        ``context_ends`` and ``query_lens`` are authoritative CPU scheduling
        metadata. ``positions`` contains absolute sequence positions and
        ``input_ids`` must already include the device-side asynchronous token
        correction. Padding beyond the real requests or tokens is ignored.

        A rewind is valid when its starting position is inside the retained
        prefix. Its old suffix becomes invalid and is overwritten by the next
        real inputs. A jump beyond the prefix is rejected so stale slot data
        cannot silently become PLE context.
        """
        if context_buffer is None:
            context_buffer = self.context_buffer
        num_reqs = len(req_ids)
        if num_reqs > self.max_num_reqs or len(set(req_ids)) != num_reqs:
            raise ValueError("Qwen4Exp PLE batch has invalid request IDs or size")
        if len(context_ends) != num_reqs or len(query_lens) != num_reqs:
            raise ValueError("Qwen4Exp PLE request metadata lengths do not match")
        if context_buffer.shape != (self.max_num_reqs, self.context_len):
            raise ValueError("Qwen4Exp PLE context buffer has an invalid shape")
        if context_buffer.dtype != torch.int32 or context_buffer.device != self.device:
            raise ValueError("Qwen4Exp PLE context buffer must use device int32 tokens")
        if input_ids.ndim != 1 or input_ids.dtype != torch.int32:
            raise ValueError("Qwen4Exp PLE input IDs must be a one-dimensional int32 tensor")
        if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("Qwen4Exp PLE positions must be a one-dimensional integer tensor")
        if query_start_loc.ndim != 1 or query_start_loc.numel() < num_reqs + 1:
            raise ValueError("Qwen4Exp PLE query start offsets have an invalid shape")
        if query_start_loc.dtype not in (torch.int32, torch.int64):
            raise ValueError("Qwen4Exp PLE query start offsets must be integer tokens")
        if any(tensor.device != self.device for tensor in (input_ids, positions, query_start_loc)):
            raise ValueError("Qwen4Exp PLE inputs and history must share a device")
        num_tokens = sum(query_lens)
        if not 0 <= num_tokens <= self.max_num_tokens:
            raise ValueError("Qwen4Exp PLE token batch exceeds history capacity")
        if min(input_ids.numel(), positions.numel()) < num_tokens:
            raise ValueError("Qwen4Exp PLE input tensors are smaller than the scheduled token batch")

        for req_idx, req_id in enumerate(req_ids):
            if req_id not in self.req_slots:
                raise ValueError(f"Qwen4Exp PLE request {req_id!r} was not initialized")
            start, query_len = context_ends[req_idx], query_lens[req_idx]
            if start < 0 or query_len <= 0 or start + query_len > self.max_model_len:
                raise ValueError("Qwen4Exp PLE request token range is outside history bounds")
            if start > self.valid_ends[req_id]:
                raise ValueError(f"Qwen4Exp PLE request {req_id!r} has a gap in its token history")

        context_buffer.fill_(self.eos_token_id)
        if num_reqs == 0:
            return context_buffer[:0]

        slots = self._slots[:num_reqs]
        context_ends_device = self._context_ends[:num_reqs]
        # Each pinned source is immutable while its asynchronous H2D copy is
        # pending. Reusing a mutable CPU staging buffer across iterations could
        # overwrite metadata before the preceding copy reads it.
        metadata_cpu = torch.tensor(
            [[self.req_slots[req_id] for req_id in req_ids], list(context_ends)],
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        slots.copy_(metadata_cpu[0], non_blocking=True)
        context_ends_device.copy_(metadata_cpu[1], non_blocking=True)

        token_request_indices = torch.bucketize(
            self._token_offsets[:num_tokens],
            query_start_loc[1:num_reqs].long(),
            right=True,
        )
        token_slots = slots[token_request_indices]
        history_indices = token_slots * self.max_model_len + positions[:num_tokens].long()
        self.token_ids.view(-1).scatter_(0, history_indices, input_ids[:num_tokens])

        context_positions = context_ends_device.unsqueeze(1) + self._context_offsets
        valid_positions = context_positions >= 0
        context_indices = slots.unsqueeze(1) * self.max_model_len + context_positions.clamp_min(0)
        context_tokens = self.token_ids.view(-1)[context_indices]
        context_buffer[:num_reqs].copy_(torch.where(valid_positions, context_tokens, self.eos_token_id))

        for req_id, start, query_len in zip(req_ids, context_ends, query_lens):
            self.valid_ends[req_id] = start + query_len
        return context_buffer[:num_reqs]
