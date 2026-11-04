# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""HCCL children with explicit ownership and no process-group reinitialization."""

import ctypes
import math
import socket
from importlib import import_module

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

    fasteep = True

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
        if self.world_size == 1:
            return tensor
        if tensor.device.type == "cpu":
            return self.cpu_all_reduce_(tensor)
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

    def cpu_all_reduce_(self, tensor):
        """Synchronize V2 scheduling metadata over the active CPU group."""
        if tensor.device.type != "cpu":
            raise ValueError("CPU all-reduce requires a CPU tensor.")
        if self.world_size > 1:
            options = dist.AllreduceOptions()
            options.reduceOp = dist.ReduceOp.SUM
            self.cpu_group.allreduce([tensor], options).wait()
        return tensor

    def all_gather(self, tensor, dim=-1):
        if self.world_size == 1:
            return tensor
        dim %= tensor.ndim
        source = tensor.movedim(dim, 0).contiguous()
        output = torch.empty(
            (source.shape[0] * self.world_size,) + source.shape[1:], device=source.device, dtype=source.dtype
        )
        self.owner.collective("HcclAllGather", source, output, source.numel(), self.comm)
        return output.movedim(0, dim).contiguous()

    def reduce_scatter(self, tensor, dim=-1):
        if self.world_size == 1:
            return tensor
        dim %= tensor.ndim
        source = tensor.movedim(dim, 0).contiguous()
        if source.shape[0] % self.world_size:
            raise ValueError("Reduce-scatter dimension must be divisible by active EP size.")
        output = torch.empty(
            (source.shape[0] // self.world_size,) + source.shape[1:], device=source.device, dtype=source.dtype
        )
        self.owner.collective("HcclReduceScatter", source, output, output.numel(), self.comm)
        return output.movedim(0, dim).contiguous()

    def all_to_all(self, output, source, output_splits, input_splits):
        if output.dtype != source.dtype or not source.is_contiguous() or not output.is_contiguous():
            raise ValueError("All-to-all requires contiguous tensors with matching dtypes.")
        if self.world_size == 1:
            output.copy_(source)
            return
        width = math.prod(source.shape[1:])
        if input_splits is None:
            if source.shape[0] % self.world_size:
                raise ValueError("All-to-all input must divide evenly across ranks.")
            input_splits = [source.shape[0] // self.world_size] * self.world_size
        if output_splits is None:
            output_splits = [output.shape[0] // self.world_size] * self.world_size
        if len(input_splits) != self.world_size or len(output_splits) != self.world_size:
            raise ValueError("All-to-all split counts must match the active group.")
        if any(n < 0 for n in input_splits) or any(n < 0 for n in output_splits):
            raise ValueError("All-to-all split counts must be nonnegative.")
        if sum(input_splits) != source.shape[0] or sum(output_splits) != output.shape[0]:
            raise ValueError("All-to-all split counts must cover the tensors.")
        counts_type = ctypes.c_uint64 * self.world_size
        send_counts = counts_type(*(int(n) * width for n in input_splits))
        recv_counts = counts_type(*(int(n) * width for n in output_splits))
        send_offsets = counts_type(*(sum(send_counts[:i]) for i in range(self.world_size)))
        recv_offsets = counts_type(*(sum(recv_counts[:i]) for i in range(self.world_size)))
        function = self.owner.library.lib.HcclAlltoAllV
        pointer = ctypes.POINTER(ctypes.c_uint64)
        function.argtypes = [
            buffer_type,
            pointer,
            pointer,
            ctypes.c_int,
            buffer_type,
            pointer,
            pointer,
            ctypes.c_int,
            hcclComm_t,
            aclrtStream_t,
        ]
        function.restype = ctypes.c_int
        dtype = hcclDataTypeEnum.from_torch(source.dtype)
        # HCCL still requires valid buffer pointers when every split is zero.
        send_buffer = source if source.numel() else source.new_empty(1)
        receive_buffer = output if output.numel() else output.new_empty(1)
        self.owner.library.HCCL_CHECK(
            function(
                buffer_type(send_buffer.data_ptr()),
                send_counts,
                send_offsets,
                dtype,
                buffer_type(receive_buffer.data_ptr()),
                recv_counts,
                recv_offsets,
                dtype,
                self.comm,
                aclrtStream_t(current_stream().npu_stream),
            )
        )
        # The dispatcher consumes this result immediately. Synchronizing also
        # retains all input and host count buffers until HCCL has consumed them.
        current_stream().synchronize()


class RetainedCommunicators:
    def __init__(self, parent):
        self.parent = parent
        self.rank = parent.rank_in_group
        self.device = torch.device("npu", torch.npu.current_device())
        self.control = getattr(parent, "tcp_store_group", None)
        self.root: PyHcclCommunicator | None = PyHcclCommunicator(self.control or parent.cpu_group, self.device)
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
        if self.root is None:
            raise RuntimeError("FastEEP migration communicator is inactive.")
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
        if self.root is None:
            raise RuntimeError("FastEEP parent communicator is inactive.")
        # Lazy import keeps normal serving usable with prebuilt older extensions.
        python_hccl_subcomm = import_module("vllm_ascend.vllm_ascend_C").python_hccl_subcomm

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
        if self.root is not None:
            self.root.close()
