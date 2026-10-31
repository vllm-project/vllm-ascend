# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from vllm_ascend.quantization.quant_type import QuantType

from .utils import is_offload_target_loading


def maybe_offload_quant_method(layer, method):
    if not is_offload_target_loading():
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

    tensor_names: tuple[str, ...] = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    output_names: tuple[str, ...] = tensor_names
    weight_dtypes = (torch.uint8,)
    packing_factor = 2

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
            "w13_weight": (experts, 2 * intermediate, hidden // self.packing_factor),
            "w2_weight": (experts, hidden, intermediate // self.packing_factor),
            "w13_weight_scale": (experts, 2 * intermediate, hidden // self.scheme.group_size),
            "w2_weight_scale": (experts, hidden, intermediate // self.scheme.group_size),
        }
        for name, tensor in tensors.items():
            allowed_dtypes: tuple[torch.dtype, ...] = self.weight_dtypes if name.endswith("_weight") else (torch.uint8,)
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

    def signature(self, host_store):
        return (type(self), str(self.activation), self.scheme.group_size, host_store.specs)


class MXFP8ExpertAdapter(MXFP4ExpertAdapter):
    """Native FP8 weights and E8M0 group scales, using the existing MXFP8 scheme."""

    weight_dtypes = (torch.float8_e4m3fn,)
    packing_factor = 1

    def validate_layer(self, layer) -> None:
        if getattr(self.activation, "value", self.activation) != "silu":
            raise ValueError("MXFP8 expert offload requires a gated silu activation")
        super().validate_layer(layer)


class W8A8ExpertAdapter(MXFP4ExpertAdapter):
    """Keep the native INT8 channel scales, offsets and derived fp32 scale together."""

    tensor_names = MXFP4ExpertAdapter.tensor_names + ("w13_weight_offset", "w2_weight_offset")
    output_names = tensor_names + ("w13_weight_scale_fp32",)

    @staticmethod
    def validate_scheme(scheme) -> None:
        if not callable(getattr(scheme, "process_weights_after_loading", None)):
            raise ValueError("expert offload requires native quantization post-processing")
        if getattr(scheme, "use_expert_weight_list", False):
            raise ValueError("expert offload requires tensor weights, not EPLB expert lists")

    def validate_layer(self, layer) -> None:
        if not callable(getattr(layer, "weight_loader", None)):
            raise ValueError("W8A8 expert offload requires an expert/shard weight_loader")
        if getattr(self.activation, "value", self.activation) != "silu":
            raise ValueError("W8A8 expert offload requires a gated silu activation")
        experts = layer.local_num_experts
        w13, w2 = layer.w13_weight, layer.w2_weight
        if w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("W8A8 expert offload requires rank-3 expert weights")
        hidden, intermediate = w2.shape[1], w13.shape[1] // 2
        shapes = {
            "w13_weight": (experts, 2 * intermediate, hidden),
            "w2_weight": (experts, hidden, intermediate),
        }
        for name in self.tensor_names:
            tensor = getattr(layer, name, None)
            shape = (
                shapes[name] if name in shapes else (*shapes[name.replace("_scale", "").replace("_offset", "")][:-1], 1)
            )
            dtypes = (torch.int8,) if name in shapes else (torch.bfloat16, torch.float16, torch.float32)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype not in dtypes:
                raise ValueError(f"unsupported W8A8 expert tensor {name}; expected shape={shape}, dtype={dtypes}")
        if hidden <= 0 or intermediate <= 0:
            raise ValueError("W8A8 expert offload requires positive hidden and intermediate sizes")

    def signature(self, host_store):
        return (type(self), str(self.activation), host_store.specs)


class BlockFP8ExpertAdapter(MXFP8ExpertAdapter):
    """Resolve checkpoint block scales through the native scheme in bounded chunks."""

    tensor_names = ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv")
    output_names = MXFP8ExpertAdapter.output_names

    @staticmethod
    def validate_scheme(scheme) -> None:
        native = getattr(scheme, "mxfp8_method", None)
        if native is None or getattr(native, "group_size", None) != 32:
            raise ValueError("block FP8 expert offload requires the native MXFP8 execution scheme")
        for dimension in ("block_n", "block_k"):
            if type(getattr(scheme, dimension, None)) is not int or getattr(scheme, dimension) <= 0:
                raise ValueError("block FP8 expert offload requires positive block dimensions")
        if not callable(getattr(scheme, "process_weights_after_loading", None)):
            raise ValueError("expert offload requires native quantization post-processing")

    def validate_layer(self, layer) -> None:
        if not callable(getattr(layer, "weight_loader", None)):
            raise ValueError("block FP8 expert offload requires an expert/shard weight_loader")
        if getattr(self.activation, "value", self.activation) != "silu":
            raise ValueError("block FP8 expert offload requires a gated silu activation")
        w13, w2 = layer.w13_weight, layer.w2_weight
        if w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("block FP8 expert offload requires rank-3 expert weights")
        experts, hidden, intermediate = layer.local_num_experts, w2.shape[1], w13.shape[1] // 2
        # Preserve separate gate/up scale boundaries used by the native shard loader.
        if hidden <= 0 or intermediate <= 0 or hidden % 64 or intermediate % 64 or intermediate % self.scheme.block_n:
            raise ValueError("block FP8 expert offload requires aligned expert reduction and gate/up block dimensions")
        shapes = {
            "w13_weight": (experts, 2 * intermediate, hidden),
            "w2_weight": (experts, hidden, intermediate),
        }
        for name in self.tensor_names:
            tensor = getattr(layer, name, None)
            if name.endswith("_scale_inv"):
                e, n, k = shapes[name.removesuffix("_scale_inv")]
                shape = (
                    e,
                    (n + self.scheme.block_n - 1) // self.scheme.block_n,
                    (k + self.scheme.block_k - 1) // self.scheme.block_k,
                )
                dtype = torch.float32
            else:
                shape, dtype = shapes[name], torch.float8_e4m3fn
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"unsupported block FP8 expert tensor {name}; expected shape={shape}, dtype={dtype}")

    def signature(self, host_store):
        return (type(self), str(self.activation), self.scheme.block_n, self.scheme.block_k, host_store.specs)


def expert_adapter_type(scheme):
    """Select by native storage/execution contract, independently of model names."""
    # The block FP8 scheme switches quant_type after its first post-processing.
    if hasattr(scheme, "block_n") and hasattr(scheme, "mxfp8_method"):
        return BlockFP8ExpertAdapter
    quant_type = getattr(scheme, "quant_type", None)
    if quant_type == QuantType.W4A8MXFP:
        return MXFP4ExpertAdapter
    if quant_type == QuantType.W8A8MXFP:
        return MXFP8ExpertAdapter
    if quant_type == QuantType.W8A8:
        return W8A8ExpertAdapter
    raise ValueError("expert offload requires W4A8 MXFP, W8A8 MXFP8, W8A8 INT8 or block FP8 routed expert weights")
