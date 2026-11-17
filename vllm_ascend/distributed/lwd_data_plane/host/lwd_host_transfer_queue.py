# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdHostTransferQueue: the host (UCX) implementation of one (dp, channel, op)
FIFO.

Subclasses the HCCL ``LwdTransferQueue`` and swaps only the execution
layer: seqno reorder buffering, skip/abort bookkeeping, snapshot and
reap semantics are inherited unchanged (they are transport-agnostic).
The HCCL-specific device-stream FIFO bridging is replaced by the UCX
link's own ordering (seqno IS the wire tag), so no predecessor event
chaining is needed.

Send path: the base snapshot clone is D2H-copied into a library-pool
staging block by the IO's sender thread (synchronous copy, host-readable
on return) and posted with tag=seqno; the future is bound to a host
completion event set from the lib IO thread.

Recv path: the buffer allocation/validation contract matches the HCCL
queue exactly (exact-size device buffer or the caller's recv_buffer,
aux only on UP); the posted recv's IO-thread callback submits H2D +
descramble onto the channel stream, records the NPU event and binds the
future — so ``done()``/``wait()`` semantics are identical to HCCL.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from vllm_ascend.distributed.lwd_data_plane.host.lwd_host_channels import (
    lwd_get_host_io,
)
from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import LwdNpuUcxChannelIo
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import (
    LwdTransferFuture,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import (
    LwdTransferQueue,
    lwd_scramble_seed_for,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LwdChannelType,
    LwdTransferRequest,
)


class LwdHostTransferQueue(LwdTransferQueue):
    """One FIFO of pending requests for one DP connection, channel and op,
    backed by a UCX host-RoCE link."""

    def skip_seqno(self, seqno: int) -> None:
        # Inherited bookkeeping first (held-future failing, hole advance),
        # then arm the wire-level discard recv so the frame (which the
        # peer may already have sent) is dropped instead of occupying a
        # receive block forever.
        super().skip_seqno(seqno)
        if self.op == "recv":
            io = self._io()
            if io is not None:
                io.add_skipped(seqno)

    def shutdown(self, timeout: float | None = None) -> list[LwdTransferFuture]:
        finalized = super().shutdown(timeout=timeout)
        io = getattr(self, "_ucx_io", None)
        if io is not None:
            io.shutdown(timeout=2.0 if timeout is None else float(timeout))
            self._ucx_io = None
        return finalized

    # ------------------------------------------------------------------ #
    # Execution                                                          #
    # ------------------------------------------------------------------ #

    def _io(self) -> LwdNpuUcxChannelIo | None:
        io = getattr(self, "_ucx_io", None)
        if io is None:
            try:
                # Resolve the link by the full connection identity: one process
                # holds several (edge, cloud) pairs on multi-edge/multi-cloud
                # topologies, and the bare dp_idx lookup is ambiguous there.
                io = lwd_get_host_io(
                    self.channel_type,
                    self.connection.dp_idx,
                    self.op,
                    edge_id=self.connection.edge_id,
                    cloud_id=self.connection.cloud_id,
                )
            except RuntimeError:
                # Not an endpoint rank of this connection (or channels not
                # initialized): leave it unset; _execute would fail loudly.
                return None
            self._ucx_io = io
        return io

    def _execute(
        self,
        req: LwdTransferRequest,
        predecessor: LwdTransferFuture | None,
        into: LwdTransferFuture | None = None,
    ) -> LwdTransferFuture:
        """Issue the channel op on the UCX link (ordering is by wire tag,
        so ``predecessor`` carries no device-ordering duty here)."""
        io = self._io()
        if io is None:
            future = into or LwdTransferFuture(req, [], None)
            future._finalize(
                error=RuntimeError(
                    f"no host {self.op} IO for channel {self.channel_type.value} connection={self.connection}"
                )
            )
            return future
        if req.op == "send":
            return self._execute_send(req, io, into)
        return self._execute_recv(req, io, into)

    def _execute_send(
        self,
        req: LwdTransferRequest,
        io: Any,
        into: LwdTransferFuture | None,
    ) -> LwdTransferFuture:
        assert req.tensor is not None, "send requires tensor"
        if self.channel_type is not LwdChannelType.UP and req.aux_tensor is not None:
            raise RuntimeError("aux payload is only supported on the UP channel")
        # keepalive: the sender thread's task tuple holds the snapshot
        # tensors until its synchronous D2H completes; the future itself
        # needs no device-side keepalive (the staging block is owned by
        # the transport until the send completes).
        future = into or LwdTransferFuture(replace(req, tensor=None), [], None)
        io.submit_send(req.seqno, req.tensor, req.aux_tensor, req, future)
        return future

    def _execute_recv(
        self,
        req: LwdTransferRequest,
        io: Any,
        into: LwdTransferFuture | None,
    ) -> LwdTransferFuture:
        # Exact-size buffer, same contract as the HCCL queue (the sender's
        # frame length must match; the IO callback validates on arrival).
        buffer = req.recv_buffer
        if buffer is None:
            buffer = torch.empty(req.num_elements, dtype=torch.bfloat16, device="npu")
        else:
            buffer_elements = buffer.numel()  # symbolic-meta-ok: buffer size check, not a meta shape
            if buffer_elements != req.num_elements:
                raise RuntimeError(
                    f"recv_buffer numel {buffer_elements} != requested "
                    f"num_elements {req.num_elements} on channel "
                    f"{self.channel_type.value} seqno={req.seqno}"
                )
        self._validate_stage_segments(req)
        aux_buffer = None
        if req.aux_num_elements > 0:
            if self.channel_type is not LwdChannelType.UP:
                raise RuntimeError("aux payload is only supported on the UP channel")
            aux_buffer = torch.empty(
                req.aux_num_elements,
                dtype=req.aux_dtype or torch.int64,
                device="npu",
            )
        # Descramble runs on the channel stream after H2D and before the
        # completion event is recorded — the same point the HCCL queue
        # applies it.
        seed = lwd_scramble_seed_for(req)
        post_h2d = None
        if seed is not None:
            from vllm_ascend.distributed.lwd_data_plane.lwd_payload_scrambler import (
                lwd_get_payload_scrambler,
            )

            def post_h2d(tensor: Any, _req: Any, _seed: int = seed) -> None:
                lwd_get_payload_scrambler().descramble(tensor, seed=_seed)

        future = into or LwdTransferFuture(req, [], None)
        io.submit_recv(req.seqno, req, buffer, aux_buffer, future, post_h2d=post_h2d)
        return future
