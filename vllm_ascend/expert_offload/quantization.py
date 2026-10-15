# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch


class MXFP4ExpertAdapter:
    """Reuse the supported scheme's exact post-processing on bounded expert chunks."""

    tensor_names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")

    def __init__(self, layer, scheme):
        if scheme.group_size != 32:
            raise ValueError("expert offload PR01 requires MXFP group_size=32")
        self.scheme = scheme
        self.activation = layer.activation

    def materialize(self, host_store, expert_ids, device) -> dict[str, torch.Tensor]:
        if host_store.prepared:
            raise RuntimeError("prepared experts must be copied directly from native host storage")
        indices = torch.tensor(expert_ids, dtype=torch.long, device="cpu")
        view = SimpleNamespace(activation=self.activation)
        for name in self.tensor_names:
            # One bounded chunk, never all layers' experts on the NPU.
            raw = host_store.tensors[name].index_select(0, indices).to(device)
            setattr(view, name, torch.nn.Parameter(raw, requires_grad=False))
        self.scheme.process_weights_after_loading(view)
        return {name: getattr(view, name).detach() for name in self.tensor_names}

    def allocate_pool(self, host_store, count, device) -> dict[str, torch.Tensor]:
        """Allocate once through the native scheme using a bounded raw prototype."""
        if not host_store.prepared:
            return self.materialize(host_store, range(count), device)
        view = SimpleNamespace(activation=self.activation)
        for name in self.tensor_names:
            prototype = host_store.prototypes[name]
            raw = prototype.expand(count, *prototype.shape[1:]).contiguous().to(device)
            setattr(view, name, torch.nn.Parameter(raw, requires_grad=False))
        self.scheme.process_weights_after_loading(view)
        return {name: getattr(view, name).detach() for name in self.tensor_names}

    def signature(self, host_store):
        return (str(self.activation), self.scheme.group_size, host_store.specs)
