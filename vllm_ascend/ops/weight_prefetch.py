from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch_npu
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

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

    def __init__(self,
                 weight_prefetch_config: WeightPrefetchConfig) -> None:
        self.weight_prefetch_config = weight_prefetch_config
        self.main_stream = torch_npu.npu.current_stream()
        self.prefetch_stream = get_prefetch_stream()

    @abstractmethod
    def create(cls,
               weight_prefetch_config: Optional[WeightPrefetchConfig] = None) -> "WeightPrefetchMethod":
        raise NotImplementedError
    
    @abstractmethod
    def maybe_weight_prefetch_preprocess(self,
                                         weight: torch.Tensor,
                                         start_flag: torch.Tensor) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def maybe_weight_prefetch_postprocess(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def weight_prefetch_impl(self,
                             weight: torch.Tensor,
                             start_flag: torch.Tensor,
                             max_weight_size: int) -> None:
        raise NotImplementedError


class AttentionWeightPrefetchMethod(WeightPrefetchMethod):
    """
    Weight prefetch method for Attention layer.
    """

    def __init__(self,
                 weight_prefetch_config: WeightPrefetchConfig) -> None:
        super().__init__(weight_prefetch_config)

    @classmethod
    def create(cls,
               weight_prefetch_config: Optional[WeightPrefetchConfig]) -> "AttentionWeightPrefetchMethod":
        weight_prefetch_config = weight_prefetch_config if weight_prefetch_config else WeightPrefetchConfig({})
        return cls(weight_prefetch_config)

    def maybe_weight_prefetch_preprocess(self,
                                         weight: torch.Tensor,
                                         start_flag: torch.Tensor) -> None:
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
