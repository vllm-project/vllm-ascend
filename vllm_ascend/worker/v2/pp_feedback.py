# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host transport for small PP feedback messages on Ascend AIV.

The upstream handler still owns request generations, sample widths and auxiliary
hidden states. Only sampled-token/count/draft feedback moves off the device
communicator; forward hidden states continue to use HCCL.
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from time import monotonic

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import PPHandler, compute_need_sampled_mask

from vllm_ascend import envs

HOST_FEEDBACK_TIMEOUT_SECONDS = 120
# One token/count job and one optional draft job per in-flight microbatch.
FEEDBACK_JOBS_PER_STEP = 2


def use_host_pp_feedback(config: VllmConfig, device: torch.device) -> bool:
    # Keep the initial rollout within the two-stage asynchronous AIV scenario.
    # Other PP depths and synchronous scheduling retain the existing transport.
    return (
        device.type == "npu"
        and config.parallel_config.pipeline_parallel_size == 2
        and config.scheduler_config.async_scheduling
        and envs.HCCL_OP_EXPANSION_MODE == "AIV"
    )


class HostFeedbackTransport:
    """A bounded FIFO; failures remain observable even for discarded requests."""

    def __init__(self, group, source_rank: int, max_pending: int, device: torch.device):
        self.group = group
        self.source_rank = source_rank
        self.device = device
        self.max_pending = max_pending
        self.pending: deque[Future] = deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pp-feedback")
        self.closed = False
        self.error: Exception | None = None

    def _exchange(self, payload: list[torch.Tensor], ready) -> list[torch.Tensor]:
        if self.error is not None:
            raise RuntimeError("PP host feedback previously failed") from self.error
        try:
            if ready is not None:
                # Thread-local device selection is required for NPU events.
                with torch.npu.device(self.device):
                    ready.synchronize()
            for tensor in payload:
                torch.distributed.broadcast(tensor, src=self.source_rank, group=self.group)
            return payload
        except Exception as exc:
            self.error = exc
            raise

    def check(self) -> None:
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if self.error is not None:
            raise RuntimeError("PP host feedback previously failed") from self.error

    def submit(self, payload: list[torch.Tensor], ready=None) -> Future:
        if self.closed:
            raise RuntimeError("PP host feedback is closed")
        self.check()
        if len(self.pending) >= self.max_pending:
            self.pending[0].result(timeout=HOST_FEEDBACK_TIMEOUT_SECONDS)
            self.pending.popleft()
        future = self.executor.submit(self._exchange, payload, ready)
        self.pending.append(future)
        return future

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        deadline = monotonic() + HOST_FEEDBACK_TIMEOUT_SECONDS
        try:
            for future in self.pending:
                future.result(timeout=max(0, deadline - monotonic()))
            self.check()
        finally:
            # Gloo has the same finite timeout. Cancel jobs which have not begun
            # after a failure, and join the active job before destroying its PG.
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()


@dataclass
class PendingHostRecv:
    future: Future
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    need_sampled_mask: np.ndarray
    gen_at_receive_np: np.ndarray


class AscendPPHandler(PPHandler):
    """Replace an unused upstream handler at runner initialization.

    Adopt its initialized state so new upstream auxiliary-state fields and
    inherited methods are preserved without duplicating the upstream constructor.
    No upstream module/class is patched. This hook can be removed when upstream
    exposes a feedback transport factory supporting host collectives.
    """

    def __init__(self, handler: PPHandler):
        assert all(slot is None for slot in handler.queue), "PP feedback replacement must precede execution"
        assert not handler.sampled_tokens_broadcast
        self.__dict__.update(vars(handler))
        group = None
        # All world ranks must create all PP subgroups in identical order.
        for ranks in get_pp_group().group_ranks:
            candidate = torch.distributed.new_group(
                ranks,
                backend="gloo",
                timeout=timedelta(seconds=HOST_FEEDBACK_TIMEOUT_SECONDS),
                group_desc="pp_host_feedback",
            )
            if get_pp_group().rank in ranks:
                group = candidate
        assert group is not None
        self.broadcast_group = group
        self._host_feedback = HostFeedbackTransport(
            group, self.last_rank, get_pp_group().world_size * FEEDBACK_JOBS_PER_STEP, self.device
        )
        self._feedback_closed = False
        torch.distributed.destroy_process_group(handler.broadcast_group)

    def shutdown(self) -> None:
        if self._feedback_closed:
            return
        self._feedback_closed = True
        try:
            self._host_feedback.close()
        finally:
            self.queue.clear()
            torch.distributed.destroy_process_group(self.broadcast_group)

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor | None] | None:
        self._host_feedback.check()
        if not self.queue:
            return None
        slot = self.queue.popleft()
        self.queue.append(None)
        if slot is None:
            return None
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        if exclude_mask.all():
            # The transport retains this future until completion, including
            # failures, even though no request will consume the result.
            return None
        payload = slot.future.result(timeout=HOST_FEEDBACK_TIMEOUT_SECONDS)
        sampled_tokens = payload[0].to(self.device, non_blocking=True)
        combined = payload[1].to(self.device, non_blocking=True)
        draft_tokens = payload[2].to(self.device, non_blocking=True) if len(payload) == 3 else None
        num_sampled, num_rejected = combined.unbind(dim=0)
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            idx_mapping = async_tensor_h2d(np.where(exclude_mask, -1, slot.idx_mapping_np), device=self.device)
        return dict(
            sampled_tokens=sampled_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=draft_tokens,
        )

    def receive(self, input_batch: InputBatch, recv_drafts: bool) -> bool:
        assert not self.is_last_rank
        self._host_feedback.check()
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            return False
        num_reqs = input_batch.num_reqs
        payload = [
            torch.empty((num_reqs, self._sample_len(input_batch)), dtype=torch.int64, pin_memory=True),
            torch.empty((2, num_reqs), dtype=torch.int32, pin_memory=True),
        ]
        if recv_drafts:
            payload.append(torch.empty((num_reqs, self.num_speculative_steps), dtype=torch.int64, pin_memory=True))
        future = self._host_feedback.submit(payload)
        self.queue[-1] = PendingHostRecv(
            future,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np.copy(),
            need_sampled_mask.copy(),
            self.req_idx_gen_np[input_batch.idx_mapping_np].copy(),
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        self._host_feedback.check()
        if compute_need_sampled_mask(input_batch) is None:
            self.sampled_tokens_broadcast = False
            return
        assert sampled_token_ids.dtype == torch.int64
        assert sampled_token_ids.shape[-1] == self._sample_len(input_batch)
        self.sampled_tokens_broadcast = True
        combined = torch.stack((num_sampled, num_rejected), dim=0)
        self._enqueue_host_feedback((sampled_token_ids, combined))

    def broadcast_drafts(self, draft_tokens: torch.Tensor) -> None:
        assert self.is_last_rank
        if self.sampled_tokens_broadcast:
            self._enqueue_host_feedback((draft_tokens,))

    def _enqueue_host_feedback(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self._host_feedback.check()
        with torch.npu.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            snapshots = []
            for tensor in tensors:
                host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                host.copy_(tensor, non_blocking=True)
                tensor.record_stream(self.broadcast_stream)
                snapshots.append(host)
            ready = self.broadcast_stream.record_event()
        # Protect reused graph/output storage only until the local D2H snapshot.
        # Neither this dependency nor the broadcast stream waits for a peer.
        self.main_stream.wait_event(ready)
        self._host_feedback.submit(snapshots, ready)
