# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import weakref
from multiprocessing import shared_memory

import torch
import torch_npu


class _GatherParticipant(ctypes.Structure):
    _fields_ = [
        ("control", ctypes.c_void_p),
        ("packed", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("world_size", ctypes.c_int),
        ("epoch", ctypes.c_uint64),
    ]


class HostGatherCoordinator:
    """Share only gathered rows; the complete KV pool remains owned by MemFabric."""

    def __init__(self, group, rank: int, world_size: int, capacity_bytes: int):
        header_bytes = ((6 + world_size) * 8 + 63) // 64 * 64
        if rank == 0:
            self.mapping = shared_memory.SharedMemory(create=True, size=header_bytes + capacity_bytes)
            self.mapping.buf[:header_bytes] = bytes(header_bytes)
            identity = self.mapping.name
        else:
            identity = None
        name = group.broadcast_object(identity, src=0)
        if rank != 0:
            self.mapping = shared_memory.SharedMemory(name=name)
        self.tensor = torch.frombuffer(self.mapping.buf, dtype=torch.uint8)
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
        self.acl.aclrtHostRegisterV2.restype = ctypes.c_int
        self.acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.acl.aclrtHostUnregister.restype = ctypes.c_int
        host_register_pinned = 0x10000000
        result = self.acl.aclrtHostRegisterV2(self.tensor.data_ptr(), self.tensor.numel(), host_register_pinned)
        if result != 0:
            raise RuntimeError(f"Selective shared Host buffer registration failed with result={result}")
        self.registration = weakref.finalize(self, self.acl.aclrtHostUnregister, self.tensor.data_ptr())
        self.payload = self.tensor[header_bytes:]
        self.participant = _GatherParticipant(
            self.tensor.data_ptr(), self.tensor.data_ptr() + header_bytes, rank, world_size, 0
        )
        group.barrier()
        if rank == 0:
            self.mapping.unlink()


class _HostTransfer(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_void_p) for name in ("sources", "destinations", "lengths", "packed_k", "packed_v")]
        + [(name, ctypes.c_size_t) for name in ("entries", "k_bytes", "v_bytes")]
        + [("to_device", ctypes.c_bool), ("participant", ctypes.c_void_p)]
    )


class SelectiveSDMATransfer:
    """Fixed-address Host gather, SDMA staging, and device scatter for graph replay."""

    @staticmethod
    def device_memory_bytes(entries: int, k_bytes: int, v_bytes: int) -> int:
        # K/V payloads, two address vectors, and the combined gather/scatter address vector.
        return entries * (k_bytes + v_bytes + 4 * torch.int64.itemsize)

    def __init__(self, library, entries: int, k_bytes: int, v_bytes: int, device, coordinator=None):
        self.coordinator = coordinator
        self.enqueue = library.selective_enqueue_host_gather
        self.enqueue.argtypes = [ctypes.c_void_p, ctypes.POINTER(_HostTransfer)]
        self.enqueue.restype = ctypes.c_int
        self.complete = library.selective_enqueue_host_gather_done
        self.complete.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GatherParticipant)]
        self.complete.restype = ctypes.c_int
        self.acl = ctypes.CDLL("libascendcl.so")
        self.acl.aclrtMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.acl.aclrtMemcpyAsync.restype = ctypes.c_int
        self.sources = torch.empty(2 * entries, dtype=torch.int64, pin_memory=True)
        self.destinations = torch.empty_like(self.sources, pin_memory=True)
        self.lengths = torch.empty(2 * entries, dtype=torch.int32, pin_memory=True)
        if coordinator is None:
            self.host = [
                torch.zeros((entries, size), dtype=torch.uint8, pin_memory=True) for size in (k_bytes, v_bytes)
            ]
        else:
            self.host = [
                coordinator.payload[: entries * k_bytes].view(entries, k_bytes),
                coordinator.payload[entries * k_bytes : entries * (k_bytes + v_bytes)].view(entries, v_bytes),
            ]
        self.device = [torch.empty_like(tensor, device=device) for tensor in self.host]
        offsets = torch.arange(entries, dtype=torch.int64, device=device)
        self.addresses = [offsets * size + tensor.data_ptr() for tensor, size in zip(self.device, (k_bytes, v_bytes))]
        self.packed_addresses = torch.empty(2 * entries, dtype=torch.int64, device=device)
        self.payloads: dict[tuple[int, bool], _HostTransfer] = {}

    def copy(self, destination, source, kind: int) -> None:
        size = source.numel() * source.element_size()
        self.copy_bytes(destination.data_ptr(), source.data_ptr(), size, kind)

    def copy_bytes(self, destination: int, source: int, size: int, kind: int) -> None:
        if size == 0:
            return
        stream = torch_npu.npu.current_stream().npu_stream
        result = self.acl.aclrtMemcpyAsync(destination, size, source, size, kind, stream)
        if result != 0:
            raise RuntimeError(f"Selective KV SDMA enqueue failed with result={result}")

    def transfer(self, manager, sources, destinations, lengths, count, *, to_device: bool) -> None:
        entries = sources.numel() // 2
        if sources.numel() != 2 * entries or entries > self.host[0].shape[0]:
            raise ValueError("Selective SDMA descriptors exceed staging capacity")
        if lengths.dtype != torch.int32 or lengths.numel() != sources.numel():
            raise ValueError("Selective SDMA requires one int32 length per descriptor")
        key = (entries, to_device)
        if key not in self.payloads:
            # Capture retains this payload and its pinned buffers until the manager is destroyed.
            packed_k, packed_v = (host.data_ptr() for host in self.host)
            if self.coordinator is not None:
                packed_v = packed_k + entries * self.host[0].shape[1]
            self.payloads[key] = _HostTransfer(
                self.sources.data_ptr(),
                self.destinations.data_ptr(),
                self.lengths.data_ptr(),
                packed_k,
                packed_v,
                entries,
                self.host[0].shape[1],
                self.host[1].shape[1],
                to_device,
                ctypes.addressof(self.coordinator.participant) if self.coordinator is not None else None,
            )
        packed = self.packed_addresses[: 2 * entries]
        packed[:entries].copy_(self.addresses[0][:entries])
        packed[entries:].copy_(self.addresses[1][:entries])
        self.copy(self.lengths[: 2 * entries], lengths, 2)
        if to_device:
            self.copy(self.sources[: 2 * entries], sources, 2)
            self._launch(key)
            payload = self.payloads[key]
            for source, device in zip((payload.packed_k, payload.packed_v), self.device):
                self.copy_bytes(device.data_ptr(), source, entries * device.shape[1], 1)
            if self.coordinator is not None:
                result = self.complete(
                    torch_npu.npu.current_stream().npu_stream, ctypes.byref(self.coordinator.participant)
                )
                if result != 0:
                    raise RuntimeError(f"Selective Host KV completion enqueue failed with result={result}")
            manager.copy_sfa_kv(packed, destinations, lengths, count)
        else:
            manager.copy_sfa_kv(sources, packed, lengths, count)
            self.copy(self.destinations[: 2 * entries], destinations, 2)
            for host, device in zip(self.host, self.device):
                self.copy(host[:entries], device[:entries], 2)
            self._launch(key)

    def _launch(self, key) -> None:
        result = self.enqueue(torch_npu.npu.current_stream().npu_stream, ctypes.byref(self.payloads[key]))
        if result != 0:
            raise RuntimeError(f"Selective Host KV callback enqueue failed with result={result}")

    def restore(self, manager, sources, destinations, lengths) -> None:
        """Expand rollback blocks into token rows and reuse the same Host gather/SDMA path."""
        source_rows = sources.cpu().tolist()
        target_rows = destinations.cpu().tolist()
        sizes = lengths.cpu().tolist()
        blocks = len(source_rows) // 2
        k_bytes, v_bytes = self.host[0].shape[1], self.host[1].shape[1]
        k_sources, v_sources, k_targets, v_targets = [], [], [], []
        for block in range(blocks):
            tokens, k_remainder = divmod(sizes[block], k_bytes)
            v_tokens, v_remainder = divmod(sizes[blocks + block], v_bytes)
            if k_remainder or v_remainder or tokens != v_tokens:
                raise ValueError("Rollback K/V lengths must contain the same whole token rows")
            for token in range(tokens):
                k_sources.append(source_rows[block] + token * k_bytes)
                v_sources.append(source_rows[blocks + block] + token * v_bytes)
                k_targets.append(target_rows[block] + token * k_bytes)
                v_targets.append(target_rows[blocks + block] + token * v_bytes)
        capacity = self.host[0].shape[0]
        for start in range(0, len(k_sources), capacity):
            end = min(start + capacity, len(k_sources))
            rows = end - start
            src = torch.tensor(k_sources[start:end] + v_sources[start:end], dtype=torch.int64, device=sources.device)
            dst = torch.tensor(k_targets[start:end] + v_targets[start:end], dtype=torch.int64, device=sources.device)
            size = torch.tensor([k_bytes] * rows + [v_bytes] * rows, dtype=torch.int32, device=sources.device)
            count = torch.tensor([2 * rows], dtype=torch.int32, device=sources.device)
            self.transfer(manager, src, dst, size, count, to_device=True)
            # Rollback runs outside capture; retain temporary descriptors until their copies finish.
            torch_npu.npu.current_stream().synchronize()
