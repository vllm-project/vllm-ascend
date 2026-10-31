# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from .transfer import TorchExpertTransfer


class PrefillExpertPool:
    """One reusable full layer; phase one requires homogeneous expert tensor specs."""

    def __init__(self):
        self.signature = None
        self.tensors: dict[str, torch.Tensor] = {}

    def initialize(self, binding) -> None:
        signature = binding.adapter.signature(binding.host_store)
        if self.signature is not None:
            if signature != self.signature:
                raise ValueError(
                    "expert offload requires homogeneous expert layouts for its single shared prefill pool"
                )
            return
        # Allocate in the scheme's native format, including scale strides.
        self.tensors = binding.adapter.allocate_pool(binding.host_store, binding.num_experts, binding.device)
        self.signature = signature

    def fill(self, binding):
        if self.signature != binding.adapter.signature(binding.host_store):
            raise RuntimeError("prefill pool layout mismatch")
        TorchExpertTransfer.copy_experts(
            binding.adapter,
            binding.host_store,
            self.tensors,
            tuple((eid, eid) for eid in range(binding.num_experts)),
            binding.device,
        )
        return self.tensors

    def close(self) -> None:
        self.tensors.clear()
        self.signature = None
