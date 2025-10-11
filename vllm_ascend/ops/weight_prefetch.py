from dataclasses import dataclass, field

import torch
import torch_npu

from vllm_ascend.ascend_config import WeightPrefetchConfig
from vllm_ascend.ops.linear import (AscendQKVParallelLinear,
                                    AscendRowParallelLinear)

SUPPORTED_MODULES = ["attn", "mlp", "moe"]


@dataclass
class ModuleWeightPrefetchConfig:
    module_name: str
    enable: bool = False
    prefetch_ratio: dict = field(default_factory=dict)
    linear_prefix_map: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prefetch_ratio = {
            prefix: ratio
            for prefix, ratio in self.prefetch_ratio.items() if 0 <= ratio <= 1
        }

        assert self.module_name in SUPPORTED_MODULES, (
            f"Invalid module name {self.module_name}, should be one of {SUPPORTED_MODULES}"
        )

        if self.module_name in SUPPORTED_MODULES:
            self.enable = self.enable and any(self.prefetch_ratio.values()) > 0


class WeightPrefetchMethod:
    """
    Unified weight prefetch method.
    """

    def __init__(self, weight_prefetch_config: WeightPrefetchConfig) -> None:
        self.attn = ModuleWeightPrefetchConfig(
            module_name="attn",
            enable=weight_prefetch_config.enabled,
            prefetch_ratio=weight_prefetch_config.prefetch_ratio.get(
                "attn", {}),
            linear_prefix_map={
                AscendQKVParallelLinear.__name__: "qkv",
                AscendRowParallelLinear.__name__: "o",
            })

    def maybe_prefetch_attn_weight_preprocess(
            self, layer_cls_name: str, weight: torch.Tensor,
            start_flag: torch.Tensor) -> None:
        if not self.attn.enable or layer_cls_name not in self.attn.linear_prefix_map:
            return

        prefix = self.attn.linear_prefix_map.get(layer_cls_name, "")
        weight_size = weight.data.element_size() * weight.data.numel(
        ) * self.attn.prefetch_ratio.get(prefix, 0)

        torch.ops.vllm.prefetch_preprocess(weight=weight,
                                           start_flag=start_flag,
                                           max_weight_size=int(weight_size))

    def maybe_prefetch_attn_weight_postprocess(
            self, layer_cls_name: str, stop_flag: torch.Tensor) -> None:
        if not self.attn.enable or layer_cls_name not in self.attn.linear_prefix_map:
            return

        torch.ops.vllm.prefetch_postprocess(stop_flag)


def maybe_npu_prefetch(inputs: torch.Tensor,
                       dependency: torch.Tensor,
                       max_size: int = 0,
                       offset: int = 0,
                       *,
                       enabled: bool = True) -> None:
    if not enabled:
        return
    input_size = inputs.element_size() * inputs.numel()
    if max_size <= 0 or max_size > input_size:
        max_size = input_size
    torch_npu.npu_prefetch(inputs, dependency, max_size, offset)
