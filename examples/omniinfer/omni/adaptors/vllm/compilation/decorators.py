from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
from omni.adaptors.vllm.compilation.wrapper import \
    TorchNpuCompilerWrapperWithCustomDispatcher
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_T = TypeVar('_T', bound=type[nn.Module])


def _support_torch_compile(
    cls: _T,
    dynamic_arg_dims: dict[str, Union[int, list[int]]],
) -> _T:
    if TorchNpuCompilerWrapperWithCustomDispatcher in cls.__bases__:
        return cls

    cls.__bases__ = cls.__bases__ + (
        TorchNpuCompilerWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        compilation_counter.num_models_seen += 1
        TorchNpuCompilerWrapperWithCustomDispatcher.__init__(
            self, vllm_config, dynamic_arg_dims)

    cls.__init__ = __init__
    cls.__call__ = TorchNpuCompilerWrapperWithCustomDispatcher.__call__

    return cls
