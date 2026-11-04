# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reclaim idle HCCL buffers while retaining CPU rendezvous and workers."""

import socket

import torch
from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.fasteep.hccl import RetainedCommunicators, RetainedGroup


class ReclaimableCommunicators(RetainedCommunicators):
    """Separate short-lived migration and active serving communicators.

    HCCL suspend/resume does not free its CCL buffers. Destroying unused
    native communicators releases those buffers; CPU groups remain available
    to restore them. Captured compute graphs must not embed these handles.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.active_group = None
        self.active_native = None
        self.cpu_groups = {}
        self.last_memory_delta = 0

    def begin_transition(self):
        if self.root is None:
            self.root = PyHcclCommunicator(self.control or self.parent.cpu_group, self.device)
            if not self.root.available:
                raise RuntimeError("Failed to restore the migration communicator.")

    def _cpu_group(self, size):
        if size in self.cpu_groups:
            return self.cpu_groups[size]
        listener = None
        endpoint = None
        if self.rank == 0:
            listener = socket.socket()
            listener.bind(("", 0))
            listener.listen()
            endpoint = (get_ip(), listener.getsockname()[1])
        host, port = self.gather_objects(endpoint)[0]
        cpu = None
        try:
            if self.rank < size:
                cpu = stateless_init_torch_distributed_process_group(
                    host=host,
                    port=port,
                    rank=self.rank,
                    world_size=size,
                    backend="gloo",
                    group_name=f"fasteep_reclaimable_{size}",
                    listen_socket=listener,
                )
        finally:
            if listener is not None:
                listener.close()
        self.cpu_groups[size] = cpu
        return cpu

    def activate(self, size):
        torch.npu.synchronize()
        free_before = torch.npu.mem_get_info()[0]
        # Every startup rank participates, including ranks becoming inactive.
        if self.active_native is not None:
            self.active_native.close()
            self.active_native = None
            self.active_group = None
        self.barrier()
        cpu = self._cpu_group(size)
        if self.rank < size:
            native = PyHcclCommunicator(cpu, self.device, group_local=True)
            if not native.available:
                raise RuntimeError("Failed to restore the serving communicator.")
            self.active_native = native
            self.active_group = RetainedGroup(self, native.comm, cpu, size)
        self.barrier()
        # No child native communicator references this migration root.
        if self.root is not None:
            self.root.close()
            self.root = None
        self.barrier()
        self.last_memory_delta = torch.npu.mem_get_info()[0] - free_before
        return self.active_group

    def close(self):
        torch.npu.synchronize()
        if self.active_native is not None:
            self.active_native.close()
            self.active_native = None
            self.active_group = None
        if self.root is not None:
            self.root.close()
            self.root = None
        for cpu in self.cpu_groups.values():
            if cpu is not None:
                stateless_destroy_torch_distributed_process_group(cpu)
        self.cpu_groups.clear()
