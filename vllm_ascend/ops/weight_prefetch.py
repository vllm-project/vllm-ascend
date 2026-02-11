from dataclasses import dataclass, field

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context

from vllm_ascend.ascend_config import WeightPrefetchConfig
from vllm_ascend.ops.linear import AscendQKVParallelLinear, AscendRowParallelLinear
from vllm_ascend.utils import is_moe_model

SUPPORTED_MODULES = ["attn", "mlp", "moe"]
MOE_PREFETCH_TOKEN_THRESHOLD = 96
MAX_PREFETCH_WEIGHT_SIZE = 18 * 1024 * 1024


@dataclass
class ModuleWeightPrefetchConfig:
    module_name: str
    enable: bool = False
    is_active_this_forward: bool = False
    prefetch_ratio: dict = field(default_factory=dict)
    linear_prefix_map: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prefetch_ratio = {prefix: ratio for prefix, ratio in self.prefetch_ratio.items() if 0 <= ratio <= 1}

        assert self.module_name in SUPPORTED_MODULES, (
            f"Invalid module name {self.module_name}, should be one of {SUPPORTED_MODULES}"
        )

        if self.module_name in SUPPORTED_MODULES:
            self.enable = self.enable and any(self.prefetch_ratio.values()) > 0


class WeightPrefetchMethod:
    """
    Unified weight prefetch method.
    """

    is_moe: bool = True
    MLP_GATE_UP: str = "gate_up"
    MLP_DOWN: str = "down"

    # backward compatibility: delete in future versions
    mlp_pre_version_compatibale_config: dict = {}

    def __init__(self, weight_prefetch_config: WeightPrefetchConfig) -> None:
        self.is_moe = is_moe_model(get_current_vllm_config())
        self.mla_sfa_prefetch_enable = weight_prefetch_config.enabled

        self.attn = ModuleWeightPrefetchConfig(
            module_name="attn",
            enable=weight_prefetch_config.enabled,
            prefetch_ratio=weight_prefetch_config.prefetch_ratio.get("attn", {}) or {"qkv": 1.0, "o": 1.0},
            linear_prefix_map={
                AscendQKVParallelLinear.__name__: "qkv",
                AscendRowParallelLinear.__name__: "o",
            },
        )
        self.moe = ModuleWeightPrefetchConfig(
            module_name="moe",
            enable=weight_prefetch_config.enabled and self.is_moe,
            prefetch_ratio=weight_prefetch_config.prefetch_ratio.get("moe", {}) or {"gate_up": 0.8},
        )

        self.mlp = ModuleWeightPrefetchConfig(
            module_name="mlp",
            enable=weight_prefetch_config.enabled and not self.is_moe,
            prefetch_ratio=weight_prefetch_config.prefetch_ratio.get("mlp", {}) or {"gate_up": 1.0, "down": 1.0},
        )
        self.mlp_pre_version_compatibale_config = weight_prefetch_config.mlp_pre_version_compatibale_config

    def maybe_prefetch_attn_weight_preprocess(
        self, layer_cls_name: str, weight: torch.Tensor, start_flag: torch.Tensor
    ) -> None:
        if not self.attn.enable or layer_cls_name not in self.attn.linear_prefix_map:
            return

        prefix = self.attn.linear_prefix_map.get(layer_cls_name, "")
        weight_size = weight.data.element_size() * weight.data.numel() * self.attn.prefetch_ratio.get(prefix, 0)

        torch.ops.vllm.prefetch_preprocess(weight=weight, start_flag=start_flag, max_weight_size=int(weight_size))

    def maybe_prefetch_attn_weight_postprocess(self, layer_cls_name: str, stop_flag: torch.Tensor) -> None:
        if not self.attn.enable or layer_cls_name not in self.attn.linear_prefix_map:
            return

        torch.ops.vllm.prefetch_postprocess(stop_flag)

    def maybe_prefetch_moe_weight_preprocess(self, hidden_states, prefix):
        self.moe.is_active_this_forward = (
            hidden_states.shape[0] >= MOE_PREFETCH_TOKEN_THRESHOLD if self.moe.enable else False
        )
        if not self.moe.is_active_this_forward:
            return
        forward_context = get_forward_context()
        if not forward_context or forward_context.model_instance is None:
            return

        # layer_idx is subtracted by 1 because layer_idx was incremented by 1 at layernorm.
        weight = forward_context.model_instance.model.layers[forward_context.layer_idx - 1].mlp.experts.w13_weight
        weight_size = weight.data.element_size() * weight.data.numel() * self.moe.prefetch_ratio.get(prefix, 0)
        torch.ops.vllm.prefetch_preprocess(weight=weight, start_flag=None, max_weight_size=int(weight_size))

    def maybe_prefetch_moe_weight_postprocess(self, stop_flag: torch.Tensor):
        if not self.moe.is_active_this_forward:
            return

        torch.ops.vllm.prefetch_postprocess(stop_flag)

    # x_dependency only eager mode can pass None
    def maybe_prefetch_mlp_weight_preprocess(
        self, prefetch_layer_name: str, x_dependency: torch.Tensor | None, curr_layer_prefix: str | None = None
    ):
        if not self.mlp.enable and not self.mlp_pre_version_compatibale_config:
            self.mlp.is_active_this_forward = False
            return

        try:
            forward_context = get_forward_context()
        except AssertionError:
            return
        self.mlp.is_active_this_forward = (
            forward_context.layer_idx is not None
            and forward_context.num_tokens is not None
            and forward_context.num_tokens < 500
        )
        if not self.mlp.is_active_this_forward:
            return

        if prefetch_layer_name == self.MLP_GATE_UP:
            self._maybe_prefetch_mlp_gate_up_weight_preprocess(x_dependency, forward_context, curr_layer_prefix)
        elif prefetch_layer_name == self.MLP_DOWN:
            self._maybe_prefetch_mlp_down_weight_preprocess(x_dependency, forward_context)
        else:
            raise ValueError(f"Unsupported prefetch weight name: {prefetch_layer_name}")

    def _maybe_prefetch_mlp_gate_up_weight_preprocess(
        self, x_dependency: torch.Tensor, forward_context: ForwardContext, curr_layer_prefix: str | None
    ):
        if not curr_layer_prefix:
            raise ValueError("curr_layer_prefix must been specified when prefetching mlp gate_up_proj weight")

        # start point of gate_up_proj weight prefetch
        if curr_layer_prefix.split(".")[-2] == "self_attn":
            model_instance = forward_context.model_instance
            layer_idx = int(curr_layer_prefix.split(".")[2])
            weight = model_instance.model.layers[layer_idx].mlp.gate_up_proj.weight
            if self.mlp_pre_version_compatibale_config:
                weight_size = self.mlp_pre_version_compatibale_config.get(self.MLP_GATE_UP, 0)
            else:
                weight_size = (
                    weight.data.element_size() * weight.data.numel() * self.mlp.prefetch_ratio.get(self.MLP_GATE_UP, 0)
                )
            if weight_size > MAX_PREFETCH_WEIGHT_SIZE:
                weight_size = MAX_PREFETCH_WEIGHT_SIZE
            torch.ops.vllm.prefetch_preprocess(weight=weight, start_flag=x_dependency, max_weight_size=int(weight_size))
            forward_context.prefetch_mlp_gate_up_proj = True

    def _maybe_prefetch_mlp_down_weight_preprocess(self, x_dependency: torch.Tensor, forward_context: ForwardContext):
        layer_idx = forward_context.layer_idx
        model_instance = forward_context.model_instance
        weight = model_instance.model.layers[layer_idx].mlp.down_proj.weight
        if self.mlp_pre_version_compatibale_config:
            weight_size = self.mlp_pre_version_compatibale_config.get(self.MLP_DOWN, 0)
        else:
            weight_size = (
                weight.data.element_size() * weight.data.numel() * self.mlp.prefetch_ratio.get(self.MLP_DOWN, 0)
            )
        if weight_size > MAX_PREFETCH_WEIGHT_SIZE:
            weight_size = MAX_PREFETCH_WEIGHT_SIZE
        torch.ops.vllm.prefetch_preprocess(weight=weight, start_flag=x_dependency, max_weight_size=int(weight_size))
        forward_context.prefetch_mlp_down_proj = True
        forward_context.layer_idx += 1

    def maybe_prefetch_mlp_weight_postprocess(self, stop_flag: torch.Tensor):
        if not self.mlp.is_active_this_forward:
            return

        try:
            forward_context = get_forward_context()
        except AssertionError:
            return

        if forward_context.prefetch_mlp_gate_up_proj or forward_context.prefetch_mlp_down_proj:
            torch.ops.vllm.prefetch_postprocess(stop_flag)
            forward_context.prefetch_mlp_gate_up_proj = False
            forward_context.prefetch_mlp_down_proj = False

    def maybe_prefetch_mla_or_sla_weight_in_current_stream(
        self,
        inputs: torch.Tensor,
        dependency: torch.Tensor,
        max_size: int = 0,
        linear_layer: torch.nn.Module | None = None,
    ) -> None:
        if not self.mla_sfa_prefetch_enable:
            return

        # The prefetching of the weights of the o_proj matrix in the W8A8
        # scene is already performed once in AscendW8A8LinearMethod, so it
        # is not needed here.
        if linear_layer is not None:
            from vllm_ascend.quantization.methods import AscendW8A8LinearMethod

            if isinstance(
                getattr(linear_layer.quant_method, "quant_method", None),
                AscendW8A8LinearMethod,
            ):
                return

        input_size = inputs.element_size() * inputs.numel()
        if max_size <= 0 or max_size > input_size:
            max_size = input_size
        torch.ops.vllm.prefetch_preprocess(weight=inputs, start_flag=dependency, max_weight_size=int(max_size))


def maybe_npu_prefetch(
    inputs: torch.Tensor, dependency: torch.Tensor, max_size: int = 0, offset: int = 0, *, enabled: bool = True
) -> None:
    if not enabled:
        return
    input_size = inputs.element_size() * inputs.numel()
    if max_size <= 0 or max_size > input_size:
        max_size = input_size
    torch_npu.npu_prefetch(inputs, dependency, max_size, offset)
