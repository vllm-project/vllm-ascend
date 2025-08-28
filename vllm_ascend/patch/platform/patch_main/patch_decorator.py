import inspect
from typing import Callable, Optional, TypeVar, Union
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._dynamo.symbolic_convert import InliningInstructionTranslator
from vllm.compilation import decorators
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import _should_ignore_torch_compile
from vllm.compilation.monitor import start_monitoring_torch_compile
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import CompilationLevel, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo

from vllm_ascend.attention.attention_v1 import AscendAttentionState

logger = init_logger(__name__)

IGNORE_COMPILE_KEY = "_ignore_compile_vllm"

_T = TypeVar("_T", bound=type[nn.Module])


def _ascend_support_torch_compile(
    cls: _T,
    dynamic_arg_dims: dict[str, Union[int, list[int]]],
    enable_if: Optional[Callable[[VllmConfig], bool]] = None,
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWrapperWithCustomDispatcher in cls.__bases__:
        # support decorating multiple times
        return cls

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    setattr(cls, IGNORE_COMPILE_KEY, False)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        enable_compile = enable_if is None or enable_if(vllm_config)
        # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = \
            vllm_config.compilation_config.level in [
            CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS
        ] or not supports_dynamo() or _should_ignore_torch_compile(
            self.__class__) or not enable_compile
        if self.do_not_compile:
            return

        compilation_counter.num_models_seen += 1
        TorchCompileWrapperWithCustomDispatcher.__init__(
            self, compilation_level=vllm_config.compilation_config.level)

    cls.__init__ = __init__

    def __call__(self, *args, **kwargs):
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and \
            hasattr(attn_metadata, 'attn_state') and \
            attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            return self.forward(*args, **kwargs)

        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # the first compilation needs to have dynamic shapes marked
        if len(self.compiled_codes) < 1:
            sig = inspect.signature(self.__class__.forward)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for k, dims in dynamic_arg_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        dims = [
                            arg.ndim + dim if dim < 0 else dim for dim in dims
                        ]
                        torch._dynamo.mark_dynamic(arg, dims)
                    elif isinstance(arg, IntermediateTensors):
                        for tensor in arg.tensors.values():
                            # In case dims is specified with negative indexing
                            dims = [
                                tensor.ndim + dim if dim < 0 else dim
                                for dim in dims
                            ]
                            torch._dynamo.mark_dynamic(tensor, dims)
                    else:
                        raise ValueError(
                            "Unsupported dynamic dimensions"
                            f" {dims} for argument {k} with type {type(arg)}.")
            # here, it is the starting point of the `torch.compile` process
            start_monitoring_torch_compile(self.vllm_config)
            logger.debug("Start compiling function %s",
                         self.original_code_object)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            # it seems Dynamo reuse the compilation across instances,
            # while we need to make sure the compiled code is not reused.
            # we need to control all the compilation of the model.
            torch._dynamo.eval_frame.remove_from_cache(
                self.original_code_object)

            # collect all relevant files traced by Dynamo,
            # so that the compilation cache can trigger re-compilation
            # properly when any of these files change.

            # 1. the file containing the top-level forward function
            self.vllm_config.compilation_config.traced_files.add(
                self.original_code_object.co_filename)

            # 2. every time Dynamo sees a function call, it will inline
            # the function by calling InliningInstructionTranslator.inline_call
            # we hijack this function to know all the functions called
            # during Dynamo tracing, and their corresponding files
            inline_call = InliningInstructionTranslator.inline_call

            def patched_inline_call(parent, func, args, kwargs):
                code = func.get_code()
                self.vllm_config.compilation_config.traced_files.add(
                    code.co_filename)
                return inline_call(parent, func, args, kwargs)

            # Disable the C++ compilation of symbolic shape guards. C++-fication
            # of symbolic shape guards can improve guard overhead. But, since
            # vllm skip guards anyways, setting this flag to False can improve
            # compile time.
            dynamo_config_patches = {}
            try:
                _ = torch._dynamo.config.enable_cpp_symbolic_shape_guards
                dynamo_config_patches[
                    "enable_cpp_symbolic_shape_guards"] = False
            except AttributeError:
                # Note: this config is not available in torch 2.6, we can skip
                # if the config doesn't exist
                logger.debug(
                    "enable_cpp_symbolic_shape_guards config not available")

            with patch.object(InliningInstructionTranslator, 'inline_call',
                              patched_inline_call), torch._dynamo.config.patch(
                                  **dynamo_config_patches):
                output = self.compiled_callable(*args, **kwargs)
            return output

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output

    cls.__call__ = __call__
    return cls


decorators._support_torch_compile = _ascend_support_torch_compile