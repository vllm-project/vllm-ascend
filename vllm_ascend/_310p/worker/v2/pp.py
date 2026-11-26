# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
from queue import SimpleQueue
from threading import Thread
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_distributed_timeout_or_none
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pp_utils import PPHandler, compute_need_sampled_mask

_RELAY_WAIT_TIMEOUT = timedelta(seconds=45)


@dataclass
class _PendingSample:
    work: dist.Work
    payload: torch.Tensor
    indices: np.ndarray
    generation: np.ndarray
    needed: np.ndarray


class PPHandler310(PPHandler):
    """Keep sampled IDs on the host for the 310P CPU-owned request state."""

    def __init__(self, max_num_reqs: int):
        pp_group = get_pp_group()
        self.is_last_rank = pp_group.is_last_rank
        self.last_rank = pp_group.last_rank
        self.peers = pp_group.ranks[:-1]
        self.depth = pp_group.world_size
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)
        self.checked_received_tensors = False
        self.broadcast_group, _ = dist.new_subgroups_by_enumeration(
            pp_group.group_ranks,
            timeout=get_distributed_timeout_or_none(),
            backend="gloo",
            group_desc="pp_310p_samples",
        )
        assert self.broadcast_group is not None
        self._receives: deque[_PendingSample | None] = deque([None] * self.depth)
        self._sends: deque[tuple[torch.Tensor, list[dist.Work]]] = deque()
        self._relay_this_step = False
        self._sync_fallback: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pending_relays: deque[Future[None]] = deque()
        self._relay_queue: SimpleQueue[tuple[Any, Future[None]] | None] | None = None
        self._relay_thread: Thread | None = None
        if self.is_last_rank:
            self._relay_queue = SimpleQueue()
            self._relay_thread = Thread(target=self._relay_loop, daemon=True, name="pp-310p-sample-relay")
            self._relay_thread.start()

    def relay_aux_hidden_states(self, intermediate_tensors, output_intermediate_tensors):
        # 310P PP supports generation without speculative auxiliary states.
        return output_intermediate_tensors

    def begin_sample_step(self) -> None:
        self._relay_this_step = False
        self._sync_fallback = None

    def receive(self, input_batch: InputBatch, recv_drafts: bool) -> bool:
        assert not self.is_last_rank and not recv_drafts
        needed = compute_need_sampled_mask(input_batch)
        if needed is None:
            return False
        indices = input_batch.idx_mapping_np.copy()
        payload = torch.empty((2, input_batch.num_reqs), dtype=torch.int64, device="cpu")
        work = dist.irecv(payload, src=self.last_rank, group=self.broadcast_group)
        assert work is not None
        self._receives[-1] = _PendingSample(work, payload, indices, self.req_idx_gen_np[indices], needed)
        return bool(needed.all())

    def broadcast(
        self,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        del num_rejected
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            self._relay_this_step = False
            self._sync_fallback = None
            return
        # Defer the host relay until AsyncOutput has recorded its non-blocking
        # D2H copy event, avoiding a synchronous stream stall on the main thread.
        self._relay_this_step = True
        self._sync_fallback = (sampled_tokens, num_sampled)

    def _send_payload(self, payload: torch.Tensor) -> list[dist.Work]:
        sends = []
        for peer in self.peers:
            work = dist.isend(payload, dst=peer, group=self.broadcast_group)
            assert work is not None
            sends.append(work)
        return sends

    def _send_sync_fallback(self, sampled_tokens: torch.Tensor, num_sampled: torch.Tensor) -> None:
        if len(self._sends) >= self.depth:
            _, sends = self._sends.popleft()
            for work in sends:
                work.wait()
        # Without speculation each row produces at most one token.
        payload = torch.stack((sampled_tokens[:, 0], num_sampled)).to(device="cpu", dtype=torch.int64)
        self._sends.append((payload, self._send_payload(payload)))

    def submit_sample_output(self, output: Any) -> None:
        if not self._relay_this_step:
            return
        self._relay_this_step = False
        fallback = self._sync_fallback
        self._sync_fallback = None

        inner = getattr(output, "_inner", output)
        if (
            self._relay_queue is not None
            and inner is not None
            and hasattr(inner, "copy_event")
            and hasattr(inner, "sampled_token_ids")
            and hasattr(inner, "num_sampled_tokens_np")
        ):
            if len(self._pending_relays) >= self.depth:
                self._pending_relays.popleft().result(timeout=60)
            completion: Future[None] = Future()
            self._pending_relays.append(completion)
            self._relay_queue.put((inner, completion))
            return

        if fallback is not None:
            self._send_sync_fallback(*fallback)

    @torch.inference_mode()
    def _relay_loop(self) -> None:
        assert self._relay_queue is not None
        while (item := self._relay_queue.get()) is not None:
            output, completion = item
            try:
                output.copy_event.synchronize()
                tokens = np.asarray(output.sampled_token_ids)[:, 0]
                counts = np.asarray(output.num_sampled_tokens_np).reshape(-1)
                payload = torch.from_numpy(np.stack((tokens, counts)).astype(np.int64, copy=False))
                for work in self._send_payload(payload):
                    work.wait(_RELAY_WAIT_TIMEOUT)
            except Exception as exc:
                completion.set_exception(exc)
                return
            completion.set_result(None)

    def close(self) -> None:
        if self._relay_queue is not None and self._relay_thread is not None:
            self._relay_queue.put(None)
            self._relay_thread.join(timeout=5)
            self._relay_queue = None
            self._relay_thread = None
        while self._sends:
            _, sends = self._sends.popleft()
            for work in sends:
                work.wait()

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        if self.is_last_rank:
            return None
        pending = self._receives.popleft()
        self._receives.append(None)
        if pending is None:
            return None
        # Finish the receive even when every request in the batch was cancelled.
        pending.work.wait()
        valid = pending.needed & (self.req_idx_gen_np[pending.indices] == pending.generation)
        if not valid.any():
            return None
        sampled_tokens, num_sampled = pending.payload
        return dict(
            idx_mapping=torch.from_numpy(np.where(valid, pending.indices, -1)),
            sampled_tokens=sampled_tokens.unsqueeze(1),
            num_sampled=num_sampled,
            num_rejected=torch.zeros_like(num_sampled),
        )
