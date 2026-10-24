# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch


def maybe_offload_quant_method(layer, method):
    if not getattr(layer, "_expert_offload_enabled", False):
        return method

    # Called during RoutedExperts construction, after Ascend config is ready.
    from vllm_ascend.ascend_config import get_ascend_config

    config = getattr(get_ascend_config(), "expert_offload_config", None)
    if config is None or not config.enabled:
        return method

    from .method import ExpertOffloadMoEMethod

    return ExpertOffloadMoEMethod(method, layer.moe_config, config)


class MXFP4ExpertAdapter:
    """Reuse the supported scheme's exact post-processing on bounded expert chunks."""

    tensor_names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")

    @staticmethod
    def validate_scheme(scheme) -> None:
        if getattr(scheme, "group_size", None) != 32:
            raise ValueError("expert offload requires MXFP group_size=32 for packed weight scales")
        if not callable(getattr(scheme, "process_weights_after_loading", None)):
            raise ValueError("expert offload requires native quantization post-processing")

    def __init__(self, layer, scheme):
        self.validate_scheme(scheme)
        self.scheme = scheme
        self.activation = layer.activation

    def validate_layer(self, layer) -> None:
        """Check the raw checkpoint contract before allocating owned host storage.

        Model names do not establish these capabilities. All routed layers must
        expose the same packed gate/up/down tensors and the expert/shard loader.
        Native NZ storage is validated separately when preparing each expert.
        """
        prefix = f"expert offload layer {layer.layer_name}"
        if not callable(getattr(layer, "weight_loader", None)):
            raise ValueError(f"{prefix} requires an expert/shard weight_loader")
        activation = getattr(self.activation, "value", self.activation)
        if activation not in ("silu", "situ"):
            raise ValueError(f"{prefix} requires a gated silu or situ activation; got {activation}")
        tensors: dict[str, torch.Tensor] = {}
        for name in self.tensor_names:
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError(f"{prefix} requires a rank-3 {name} expert tensor")
            tensors[name] = tensor
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        experts, hidden = layer.local_num_experts, w2.shape[1]
        intermediate = w13.shape[1] // 2
        alignment = 2 * self.scheme.group_size
        if hidden <= 0 or intermediate <= 0 or hidden % alignment or intermediate % alignment:
            raise ValueError(f"{prefix} requires hidden and intermediate sizes divisible by {alignment}")
        expected_shapes = {
            "w13_weight": (experts, 2 * intermediate, hidden // 2),
            "w2_weight": (experts, hidden, intermediate // 2),
            "w13_weight_scale": (experts, 2 * intermediate, hidden // self.scheme.group_size),
            "w2_weight_scale": (experts, hidden, intermediate // self.scheme.group_size),
        }
        for name, tensor in tensors.items():
            allowed_dtypes: tuple[torch.dtype, ...] = (torch.uint8,)
            scale_dtype = getattr(torch, "float8_e8m0fnu", None)
            if name.endswith("_scale") and scale_dtype is not None:
                allowed_dtypes += (scale_dtype,)
            if tensor.dtype not in allowed_dtypes or tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(
                    f"{prefix} has unsupported {name}: expected shape={expected_shapes[name]} "
                    f"with packed byte weights/E8M0 scales; got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                )

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
