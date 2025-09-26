from abc import abstractmethod
import types
from typing import Union

import torch
import torch.nn as nn
import torchair
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
import vllm.envs as envs
from vllm.utils import supports_dynamo

from omni.adaptors.vllm.compilation.compile_config import get_torchair_config

logger = init_logger(__name__)


class TorchNpuCompilerWrapperWithCustomDispatcher:

    def __init__(self, vllm_config: VllmConfig, dynamic_arg_dims: dict[str, Union[int, list[int]]]):
        self.compiled_model = None
        self.cached_compiled_models = {}
        self.vllm_config = vllm_config
        self.dynamic_arg_dims = dynamic_arg_dims
        self.do_not_compile = vllm_config.npu_compilation_config.level == CompilationLevel.NO_COMPILATION or not supports_dynamo()
        if self.do_not_compile:
            return
        self.compile_dispatcher()

    def compile_dispatcher(self):
        backend = self.vllm_config.npu_compilation_config.init_backend(self.vllm_config)
        if not self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            logger.debug(
                f"[not use cache npu graph], VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE = {envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE}")
            self.compiled_model = torch.compile(
                self.forward,
                dynamic=True,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend)

        elif self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            logger.debug(
                f"[use cache npu graph], VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE = {envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE}")
            for gear_size in self.vllm_config.npu_compilation_config.decode_gear_list:
                new_forward_proxy_name = f"{self.__class__.__name__}_forward_with_gear_size_{gear_size}"
                code = self.forward.__code__
                new_code = code.replace(co_name=new_forward_proxy_name, )
                new_func = types.FunctionType(new_code, self.forward.__globals__,
                                              name=new_forward_proxy_name,
                                              argdefs=self.forward.__defaults__)
                self.__dict__[new_forward_proxy_name] = new_func.__get__(self, nn.Module)
                self.cached_compiled_models[gear_size] = torchair.inference.cache_compile(
                    self.__dict__[new_forward_proxy_name],
                    config=get_torchair_config(),
                    dynamic=True,
                    ge_cache=True,
                    fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                )
                logger.debug(f"[use cache npu graph], the method name = {new_forward_proxy_name}")

    def call_dispatcher(self, *args, **kwargs):
        use_eager_model = self.should_use_eager_mode(*args, **kwargs)
        if self.do_not_compile or use_eager_model:
            return self.forward(*args, **kwargs)

        if not self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            return self.compiled_model(*args, **kwargs)

        elif self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            if len(args) == 0 and len(kwargs) == 0:
                raise ValueError(
                    "If you use the compile cache function, you must input a tensor or directly input the gear size")

            input_ids = 0
            if len(args) > 0:
                input_ids = args[0]
            elif len(kwargs) > 0:
                input_ids = kwargs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                gear_size = input_ids.shape[0]
                return self.cached_compiled_models[gear_size](*args, **kwargs)
            if isinstance(input_ids, int):
                gear_size = input_ids
                return self.cached_compiled_models[gear_size](*args, **kwargs)

        logger.error(f"encountered a missed scene")
        return None

    def __call__(self, *args, **kwargs):
        return self.call_dispatcher(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def should_use_eager_mode(self, *args, **kwargs):
        ...