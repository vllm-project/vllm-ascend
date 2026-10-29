# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Discard idle weights/KV and refill identical shards through retained HCCL."""

import gc

import torch
import torch_npu

from vllm_ascend.device_allocator.camem import CaMemAllocator


class PeerMemory:
    def __init__(self, model):
        self.model = model
        self.allocator = CaMemAllocator.get_instance()
        self.saved = []
        self.sleeping = False
        self.addresses = [(name, param.data_ptr()) for name, param in model.named_parameters()]

    def storages(self):
        ranges = [
            (ptr, ptr + data.handle[1]) for ptr, data in self.allocator.pointer_to_data.items() if data.tag == "weights"
        ]
        buckets = {}
        # Buffers are rank-local metadata and are backed up separately below.
        for name, tensor in self.model.named_parameters():
            if tensor.device.type != "npu" or not tensor.numel():
                continue
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            if not any(start <= pointer and pointer + storage.nbytes() <= end for start, end in ranges):
                raise ValueError(f"Model parameter {name} is outside the weights pool.")
            item = buckets.setdefault(pointer, {"pointer": pointer, "bytes": storage.nbytes(), "tensors": []})
            item["tensors"].append(
                (
                    name,
                    tensor.data_ptr() - pointer,
                    list(tensor.shape),
                    list(tensor.stride()),
                    str(tensor.dtype),
                    torch_npu.get_npu_format(tensor),
                )
            )
        if not buckets:
            raise ValueError("FastEEP found no pooled model weights.")
        return sorted(buckets.values(), key=lambda item: item["tensors"][0][0])

    def metadata(self):
        return [{key: value for key, value in item.items() if key != "pointer"} for item in self.storages()]

    def sleep(self):
        if self.sleeping:
            return
        storages = self.storages()
        known = {item["pointer"] for item in storages}
        ranges = [
            (ptr, ptr + data.handle[1]) for ptr, data in self.allocator.pointer_to_data.items() if data.tag == "weights"
        ]
        # Preserve live non-parameter state in this pool. Unlike copying whole
        # allocations, this does not copy allocator holes or model weights.
        saved, seen, total = [], set(), 0
        for tensor in gc.get_objects():
            try:
                if not isinstance(tensor, torch.Tensor) or getattr(tensor, "fake_mode", None) is not None:
                    continue
                if tensor.device.type != "npu" or not tensor.numel():
                    continue
                pointer = tensor.untyped_storage().data_ptr()
                if pointer in known or not any(start <= pointer < end for start, end in ranges):
                    continue
                key = (pointer, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
                if key in seen:
                    continue
                seen.add(key)
                total += tensor.numel() * tensor.element_size()
                if total > 64 * 1024 * 1024:
                    raise ValueError("Unregistered weight-pool state exceeds the 64 MiB metadata limit.")
                saved.append((tensor, tensor.detach().cpu().clone()))
            except (ReferenceError, TypeError):
                continue
        self.saved = saved
        self.allocator.discard(("weights", "kv_cache"))
        self.sleeping = True

    def map(self):
        if not self.sleeping:
            raise RuntimeError("Only sleeping ranks can be refilled.")
        self.allocator.wake_up(tags=["weights", "kv_cache"])

    def finish(self):
        for tensor, backup in self.saved:
            tensor.copy_(backup)
        torch.npu.synchronize()
        self.saved.clear()
        self.sleeping = False
        if self.addresses != [(name, param.data_ptr()) for name, param in self.model.named_parameters()]:
            raise RuntimeError("FastEEP changed a model parameter address.")
