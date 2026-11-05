# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm_ascend.quantization.quant_type import QuantType


def expert_offload_quant_method(layer, method, config):
    from vllm.compilation import backends

    # Native draft loaders switch away from the target's default backbone tag.
    if backends.model_tag != "backbone":
        return method

    from .method import ExpertOffloadMoEMethod

    return ExpertOffloadMoEMethod(method, layer.moe_config, config)


class MXFP4ExpertAdapter:
    """Reuse the supported scheme's exact post-processing on bounded expert chunks."""

    tensor_names: tuple[str, ...] = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    output_names: tuple[str, ...] = tensor_names

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
        """Check packed-scale alignment."""
        prefix = f"expert offload layer {layer.layer_name}"
        hidden = layer.w2_weight.shape[1]
        intermediate = layer.w13_weight.shape[1] // 2
        alignment = 2 * self.scheme.group_size
        if hidden <= 0 or intermediate <= 0 or hidden % alignment or intermediate % alignment:
            raise ValueError(f"{prefix} requires hidden and intermediate sizes divisible by {alignment}")

    def materialize(self, host_store, expert_ids, device) -> dict[str, torch.Tensor]:
        indices = torch.tensor(expert_ids, dtype=torch.long, device="cpu")
        view = torch.nn.Module()
        view.activation = self.activation
        for name in self.tensor_names:
            # One bounded chunk, never all layers' experts on the NPU.
            raw = host_store.tensors[name].index_select(0, indices).to(device)
            setattr(view, name, torch.nn.Parameter(raw, requires_grad=False))
        self.scheme.process_weights_after_loading(view)
        return {name: getattr(view, name).detach() for name in self.output_names}

    def allocate_pool(self, host_store, count, device) -> dict[str, torch.Tensor]:
        """Allocate once through the native scheme using a bounded raw prototype."""
        if not host_store.prepared:
            return self.materialize(host_store, range(count), device)
        view = torch.nn.Module()
        view.activation = self.activation
        for name in self.tensor_names:
            prototype = host_store.prototypes[name]
            raw = prototype.expand(count, *prototype.shape[1:]).contiguous().to(device)
            setattr(view, name, torch.nn.Parameter(raw, requires_grad=False))
        self.scheme.process_weights_after_loading(view)
        return {name: getattr(view, name).detach() for name in self.output_names}


class MXFP8ExpertAdapter(MXFP4ExpertAdapter):
    """Native FP8 weights and E8M0 group scales, using the existing MXFP8 scheme."""


def expert_adapter_type(scheme):
    """Select by native storage/execution contract, independently of model names."""
    quant_type = getattr(scheme, "quant_type", None)
    if quant_type == QuantType.W4A8MXFP:
        return MXFP4ExpertAdapter
    if quant_type == QuantType.W8A8MXFP:
        return MXFP8ExpertAdapter
    raise ValueError("expert offload requires W4A8 MXFP or W8A8 MXFP8 routed expert weights")
