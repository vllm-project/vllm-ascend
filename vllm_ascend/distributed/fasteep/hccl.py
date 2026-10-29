# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""HCCL children with explicit ownership and no process-group reinitialization."""

import ctypes
import socket

import torch
import torch.distributed as dist
from vllm.distributed.utils import stateless_init_torch_distributed_process_group

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataTypeEnum,
)
from vllm_ascend.utils import current_stream


class RetainedGroup:
    """Minimal eager DP/EP group; the parent's device group is never borrowed."""

    def __init__(self, owner, comm, cpu_group, size):
        self.owner = owner
        self.comm = comm
        self.cpu_group = cpu_group
        self.world_size = size
        self.rank_in_group = owner.rank
        self.local_rank = owner.parent.local_rank
        self.ranks = owner.parent.ranks[:size]
        self.device = owner.device
        self.device_group = None
        self.device_communicator = self

    def all_reduce(self, tensor):
        if tensor.device.type == "cpu":
            dist.all_reduce(tensor, group=self.cpu_group)
            return tensor
        output = torch.empty_like(tensor)
        self.owner.library.hcclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(output.data_ptr()),
            tensor.numel(),
            hcclDataTypeEnum.from_torch(tensor.dtype),
            0,
            self.comm,
            aclrtStream_t(current_stream().npu_stream),
        )
        return output

    def all_gather(self, tensor, dim=-1):
        dim %= tensor.ndim
        source = tensor.movedim(dim, 0).contiguous()
        output = torch.empty(
            (source.shape[0] * self.world_size,) + source.shape[1:], device=source.device, dtype=source.dtype
        )
        self.owner.collective("HcclAllGather", source, output, source.numel(), self.comm)
        return output.movedim(0, dim).contiguous()

    def reduce_scatter(self, tensor, dim=-1):
        dim %= tensor.ndim
        source = tensor.movedim(dim, 0).contiguous()
        if source.shape[0] % self.world_size:
            raise ValueError("Reduce-scatter dimension must be divisible by active EP size.")
        output = torch.empty(
            (source.shape[0] // self.world_size,) + source.shape[1:], device=source.device, dtype=source.dtype
        )
        self.owner.collective("HcclReduceScatter", source, output, output.numel(), self.comm)
        return output.movedim(0, dim).contiguous()


class RetainedCommunicators:
    def __init__(self, parent):
        self.parent = parent
        self.rank = parent.rank_in_group
        self.device = torch.device("npu", torch.npu.current_device())
        self.control = getattr(parent, "tcp_store_group", None)
        self.root = PyHcclCommunicator(self.control or parent.cpu_group, self.device)
        if not self.root.available:
            raise RuntimeError("A live HCCL parent is required for FastEEP.")
        self.library = self.root.hccl
        self.children = {}
        self.sequence = 0
        self.functions = {}
        signatures = {
            "HcclAllGather": [buffer_type, buffer_type, ctypes.c_uint64, ctypes.c_int, hcclComm_t, aclrtStream_t],
            "HcclReduceScatter": [
                buffer_type,
                buffer_type,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_int,
                hcclComm_t,
                aclrtStream_t,
            ],
            "HcclSend": [buffer_type, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint32, hcclComm_t, aclrtStream_t],
            "HcclRecv": [buffer_type, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint32, hcclComm_t, aclrtStream_t],
        }
        for name, args in signatures.items():
            function = getattr(self.library.lib, name)
            function.argtypes = args
            function.restype = ctypes.c_int
            self.functions[name] = function

    def gather_objects(self, value):
        if self.control is not None:
            return self.control.all_gather_obj(value)
        values = [None] * self.parent.world_size
        dist.all_gather_object(values, value, group=self.parent.cpu_group)
        return values

    def barrier(self):
        if self.control is not None:
            self.control.barrier()
        else:
            dist.barrier(group=self.parent.cpu_group)

    def collective(self, name, source, output, count, comm):
        args = [
            buffer_type(source.data_ptr()),
            buffer_type(output.data_ptr()),
            count,
            hcclDataTypeEnum.from_torch(source.dtype),
        ]
        if name == "HcclReduceScatter":
            args.append(0)
        args += [comm, aclrtStream_t(current_stream().npu_stream)]
        self.library.HCCL_CHECK(self.functions[name](*args))

    def transfer(self, pointer, size, peer, send):
        self.library.HCCL_CHECK(
            self.functions["HcclSend" if send else "HcclRecv"](
                buffer_type(pointer),
                size,
                hcclDataTypeEnum.hcclUint8,
                peer,
                self.root.comm,
                aclrtStream_t(current_stream().npu_stream),
            )
        )

    def child(self, size):
        if size in self.children:
            return self.children[size]
        # Lazy import keeps normal serving usable with prebuilt older extensions.
        from vllm_ascend.vllm_ascend_C import python_hccl_subcomm

        function = self.library.lib.HcclCreateSubCommConfig
        self.sequence += 1
        handle = python_hccl_subcomm(
            ctypes.cast(function, ctypes.c_void_p).value,
            self.root.comm.value,
            list(range(size)),
            self.sequence,
            self.rank if self.rank < size else size,
            64,
        )
        listener = None
        port = None
        if self.rank == 0:
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
        ports = self.gather_objects(port)
        group = None
        try:
            if self.rank < size:
                cpu = stateless_init_torch_distributed_process_group(
                    host="127.0.0.1",
                    port=ports[0],
                    rank=self.rank,
                    world_size=size,
                    backend="gloo",
                    group_name=f"fasteep_{self.sequence}",
                    listen_socket=listener,
                )
                group = RetainedGroup(self, hcclComm_t(handle), cpu, size)
        except BaseException:
            if handle:
                self.library.hcclCommDestroy(hcclComm_t(handle))
            raise
        finally:
            if listener is not None:
                listener.close()
        self.children[size] = group
        return group

    def close(self):
        from vllm.distributed.utils import stateless_destroy_torch_distributed_process_group

        torch.npu.synchronize()
        for group in self.children.values():
            if group is not None:
                self.library.hcclCommDestroy(group.comm)
                stateless_destroy_torch_distributed_process_group(group.cpu_group)
        self.children.clear()
        self.root.close()
