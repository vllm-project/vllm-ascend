from abc import ABC, abstractmethod
from typing import Optional

import math
import torch
import torch_npu

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op
from vllm_ascend.ascend_config import WeightPrefetchConfig, get_ascend_config

from vllm_ascend.ascend_config import WeightPrefetchConfig


PREFETCH_STREAM: Optional[torch_npu.npu.Stream] = None


def get_prefetch_stream() -> torch_npu.npu.Stream:
    global PREFETCH_STREAM
    if PREFETCH_STREAM is None:
        PREFETCH_STREAM = torch_npu.npu.Stream()
    return PREFETCH_STREAM


class WeightPrefetchMethod(ABC):
    """
    Base weight prefetch method.
    """

    def __init__(self, weight_prefetch_config: WeightPrefetchConfig):
        self.weight_prefetch_config = weight_prefetch_config
        self.main_stream = torch_npu.npu.current_stream()
        self.prefetch_stream = get_prefetch_stream()

    @classmethod
    @abstractmethod
    def create(cls, weight_prefetch_config: WeightPrefetchConfig, vllm_config: VllmConfig) -> "WeightPrefetchMethod":
        raise NotImplementedError

    @abstractmethod
    def maybe_weight_prefetch_preprocess(self, weight: torch.Tensor, start_flag: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def maybe_weight_prefetch_postprocess(self):
        raise NotImplementedError


class AttentionWeightPrefetchMethod(WeightPrefetchMethod):
    """
    Weight prefetch method for Attention layer.
    """

    def __init__(self, weight_prefetch_config: WeightPrefetchConfig):
        super().__init__(weight_prefetch_config)

    @classmethod
    def create(cls, weight_prefetch_config: Optional[WeightPrefetchConfig]) -> "AttentionWeightPrefetchMethod":
        weight_prefetch_config = weight_prefetch_config if weight_prefetch_config else WeightPrefetchConfig({})
        return cls(weight_prefetch_config)

    def maybe_weight_prefetch_preprocess(self, weight: torch.Tensor, start_flag: torch.Tensor):
        if not self.weight_prefetch_config.attn_weight_prefetch_config.enabled:
            return

        weight_size = weight.data.element_size() * weight.data.numel()

        self.weight_prefetch_impl(weight=weight,
                                  start_flag=start_flag,
                                  max_weight_size=weight_size)

    def maybe_weight_prefetch_postprocess(self):
        if self.weight_prefetch_config.attn_weight_prefetch_config.enabled and self.prefetch_stream is not None:
            self.main_stream.wait_stream(self.prefetch_stream)

    def weight_prefetch_impl(self,
                             weight: torch.Tensor,
                             start_flag: torch.Tensor,
                             max_weight_size: int) -> None:
        self.prefetch_stream.wait_stream(self.main_stream)
        with torch_npu.npu.stream(stream=self.prefetch_stream):
            torch.ops.vllm.npu_prefetch(inputs=weight,
                                        dependency=start_flag,
                                        max_size=max_weight_size)

class MoEWeightPrefetchMethod(WeightPrefetchMethod):
    
    def __init__(self, weight_prefetch_config: WeightPrefetchConfig, vllm_config: VllmConfig):
        super().__init__(weight_prefetch_config)
        self.activated = False
        self._update_activate_param(vllm_config)
        self.prefetch_instruction_delivered = False

    @classmethod
    def create(cls, weight_prefetch_config: WeightPrefetchConfig, vllm_config: VllmConfig) -> "MoEWeightPrefetchMethod":
        return cls(weight_prefetch_config, vllm_config)

    def _update_activate_param(self, vllm_config: VllmConfig):
        parallel_config = vllm_config.parallel_config
        forward_context = get_forward_context()
        if not self.weight_prefetch_config.moe_weight_prefetch_config.enabled:
            self.activated = False
            return
        if vllm_config.model_config.hf_config.model_type == "qwen3_moe":
            if (not forward_context.with_prefill and forward_context.num_tokens >= 96 and 
                parallel_config.world_size in (16, 32)):
                self.activated = True
                prefetch_ratio = self.weight_prefetch_config.moe_weight_prefetch_config.prefetch_ratio
                expert_total_num = vllm_config.model_config.hf_config.num_experts
                self.prefetch_experts_per_npu = math.floor(expert_total_num / forward_context.ep_size * prefetch_ratio) \
                    if parallel_config.enable_expert_parallel \
                    else math.floor(prefetch_ratio * expert_total_num)
        else:
            self.activated = False
            return

    def maybe_weight_prefetch_preprocess(self, weight: torch.Tensor, start_flag: torch.Tensor):
        if not self.activated or self.prefetch_instruction_delivered:
            return
        forward_context = get_forward_context()
        self.prefetch_stream.wait_stream(self.main_stream)
        with torch.npu.stream(self.prefetch_stream):
            chunk = forward_context.model_instance.model.layers[forward_context.layer_idx].mlp.experts.w13_weight[:self.prefetch_experts_per_npu]
            torch.ops.vllm.npu_prefetch(inputs = chunk, dependency = None, max_size = chunk.element_size() * chunk.numel())
        self.prefetch_instruction_delivered = True
        forward_context.layer_idx += 1

    def maybe_weight_prefetch_postprocess(self):
        if not self.activated or self.prefetch_instruction_delivered:
            return
        self.main_stream.wait_stream(self.prefetch_stream)
        self.prefetch_instruction_delivered = False
    
class PrefetchManager:
    prefetch_map = {MoEWeightPrefetchMethod: None}

    @classmethod
    def init_forward_prefetch(cls, vllm_config: VllmConfig):
        weight_prefetch_config = get_ascend_config().weight_prefetch_config
        for prefetch_cls in PrefetchManager.prefetch_map.keys():
            PrefetchManager.prefetch_map[prefetch_cls] = prefetch_cls.create(weight_prefetch_config, vllm_config)

    @classmethod
    def get_prefetch_obj(cls, prefetch_cls: type[WeightPrefetchMethod]) -> WeightPrefetchMethod:
        return PrefetchManager.prefetch_map.get(prefetch_cls)

def npu_prefetch(inputs: torch.Tensor,
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


def npu_prefetch_fake(inputs: torch.Tensor,
                      dependency: torch.Tensor,
                      max_size: int = 0,
                      offset: int = 0,
                      *,
                      enabled: bool = True) -> None:
    return


direct_register_custom_op(
    op_name="npu_prefetch",
    op_func=npu_prefetch,
    mutates_args=[],
    fake_impl=npu_prefetch_fake,
    dispatch_key=current_platform.dispatch_key,
)
