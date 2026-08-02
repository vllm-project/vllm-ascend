#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep base and LoRA full graphs in separate vLLM compile variants.

vLLM e5588e49 drops non-shape Dynamo guards after the first trace. With LoRA
specialization enabled, that makes base and adapter requests reuse one compiled
callable. Ascend needs independent callables because each variant owns a
different ACL Graph and graph-task resource set.

This worker patch keeps the vLLM wheel unchanged and extends its compile wrapper
before model loading. Its source hash guard binds it to the verified vLLM commit.
"""

import hashlib
import inspect
from contextlib import contextmanager, nullcontext
from types import FunctionType, MethodType
from typing import Any

import torch
import vllm.envs as envs
from vllm.compilation import wrapper as wrapper_module
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

_SUPPORTED_VLLM_COMMIT = "e5588e49bc2642670116664a7fc4096e27adb179"
_SUPPORTED_WRAPPER_SHA256 = "c1b1fca679ea16aa07a696831a810c0d531da2bb0ea32dd6f1a95cdaef36de07"
_PATCH_MARKER = "_vllm_ascend_lora_dual_graph_patch"
_ORIGINAL_INIT_ATTR = "_vllm_ascend_lora_dual_graph_original_init"
_ORIGINAL_INIT = getattr(
    TorchCompileWithNoGuardsWrapper,
    _ORIGINAL_INIT_ATTR,
    TorchCompileWithNoGuardsWrapper.__init__,
)
_BASELINE_VERIFIED = False


def _verify_vllm_wrapper_baseline() -> None:
    global _BASELINE_VERIFIED
    if _BASELINE_VERIFIED:
        return

    wrapper_path = inspect.getsourcefile(wrapper_module)
    if wrapper_path is None:
        raise RuntimeError("Cannot locate vLLM compilation wrapper source")

    with open(wrapper_path, "rb") as wrapper_file:
        actual_sha256 = hashlib.sha256(wrapper_file.read()).hexdigest()
    if actual_sha256 != _SUPPORTED_WRAPPER_SHA256:
        raise RuntimeError(
            "The Ascend LoRA dual-graph patch requires the unmodified vLLM "
            f"{_SUPPORTED_VLLM_COMMIT} compilation wrapper. Expected SHA256 "
            f"{_SUPPORTED_WRAPPER_SHA256}, got {actual_sha256} from {wrapper_path}."
        )
    _BASELINE_VERIFIED = True


@contextmanager
def _without_bytecode_hook():
    """Disable the global hook only while upstream initializes this wrapper."""

    had_override = "VLLM_USE_BYTECODE_HOOK" in vars(envs)
    previous_override = vars(envs).get("VLLM_USE_BYTECODE_HOOK")
    envs.VLLM_USE_BYTECODE_HOOK = False
    try:
        yield
    finally:
        if had_override:
            envs.VLLM_USE_BYTECODE_HOOK = previous_override
        else:
            delattr(envs, "VLLM_USE_BYTECODE_HOOK")


def _clone_forward(self: Any, suffix: str) -> MethodType:
    """Return a bound forward with an independent Dynamo code identity."""

    forward_func = self.forward.__func__
    code_name = f"{forward_func.__code__.co_name}_{suffix}"
    cloned_func = FunctionType(
        forward_func.__code__.replace(
            co_name=code_name,
            co_qualname=f"{forward_func.__code__.co_qualname}_{suffix}",
        ),
        forward_func.__globals__,
        name=code_name,
        argdefs=forward_func.__defaults__,
        closure=forward_func.__closure__,
    )
    cloned_func.__kwdefaults__ = forward_func.__kwdefaults__
    cloned_func.__annotations__ = forward_func.__annotations__
    return MethodType(cloned_func, self)


def _patched_init(
    self: Any,
    compile_prefix: str = "",
    is_encoder: bool = False,
) -> None:
    vllm_config = get_current_vllm_config()
    specialize_lora = bool(
        vllm_config.lora_config is not None and vllm_config.compilation_config.cudagraph_specialize_lora
    )
    use_bytecode_hook = envs.VLLM_USE_BYTECODE_HOOK and not specialize_lora

    if specialize_lora and envs.VLLM_USE_AOT_COMPILE:
        raise RuntimeError("Ascend base/LoRA dual-graph specialization requires VLLM_USE_AOT_COMPILE=0.")

    if specialize_lora:
        _verify_vllm_wrapper_baseline()
        with _without_bytecode_hook():
            _ORIGINAL_INIT(self, compile_prefix=compile_prefix, is_encoder=is_encoder)
    else:
        _ORIGINAL_INIT(self, compile_prefix=compile_prefix, is_encoder=is_encoder)

    self.specialize_lora = specialize_lora
    self.use_bytecode_hook = use_bytecode_hook
    self._base_dynamic_inputs_marked = False
    self._punica_wrappers = None

    if not specialize_lora:
        return

    compilation_config = vllm_config.compilation_config
    base_prefix = f"{compile_prefix}.base" if compile_prefix else "base"
    base_one_prefix = f"{compile_prefix}.base_one" if compile_prefix else "base_one"
    base_backend = compilation_config.init_backend(
        vllm_config,
        prefix=base_prefix,
        is_encoder=is_encoder,
    )
    base_one_backend = compilation_config.init_backend(
        vllm_config,
        prefix=base_one_prefix,
        is_encoder=is_encoder,
    )
    options = {}
    if isinstance(base_backend, str) and base_backend == "inductor":
        options = compilation_config.inductor_compile_config

    self._base_compiled_callable = torch.compile(
        _clone_forward(self, "base"),
        fullgraph=True,
        dynamic=False,
        backend=base_backend,
        options=options,
    )
    self._base_one_compiled_callable = torch.compile(
        _clone_forward(self, "base_one"),
        fullgraph=True,
        dynamic=False,
        backend=base_one_backend,
        options=options,
    )


def _punica_has_lora(self: Any) -> bool | None:
    if self._punica_wrappers is None:
        wrappers: list[Any] = []
        seen: set[int] = set()
        for module in self.modules():
            punica_wrapper = getattr(module, "punica_wrapper", None)
            if punica_wrapper is not None and id(punica_wrapper) not in seen:
                wrappers.append(punica_wrapper)
                seen.add(id(punica_wrapper))
        self._punica_wrappers = wrappers

    if not self._punica_wrappers:
        return None
    return any(not wrapper.no_lora for wrapper in self._punica_wrappers)


def _mark_variant_dynamic_inputs(self: Any, *args: Any, **kwargs: Any) -> None:
    dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", {})
    if not dynamic_arg_dims:
        return

    signature = inspect.signature(self.__class__.forward)
    bound_args = signature.bind(self, *args, **kwargs)
    bound_args.apply_defaults()

    for name, dims_value in dynamic_arg_dims.items():
        arg = bound_args.arguments.get(name)
        if arg is None:
            continue

        if isinstance(dims_value, dict):
            dims = list(dims_value)
        elif isinstance(dims_value, int):
            dims = [dims_value]
        else:
            dims = list(dims_value)

        tensors = [arg] if isinstance(arg, torch.Tensor) else []
        if hasattr(arg, "tensors"):
            tensors.extend(arg.tensors.values())

        for tensor in tensors:
            normalized_dims = [tensor.ndim + dim if dim < 0 else dim for dim in dims]
            torch._dynamo.mark_dynamic(tensor, normalized_dims)


def _patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
    compiled_callable = self._compiled_callable
    use_base_callable = False
    if self.specialize_lora and is_forward_context_available():
        batch_descriptor = get_forward_context().batch_descriptor
        has_lora = batch_descriptor.has_lora if batch_descriptor is not None else True
        punica_has_lora = self._punica_has_lora()
        if punica_has_lora is not None:
            has_lora = punica_has_lora
        if not has_lora:
            use_base_callable = True
            if batch_descriptor is not None and batch_descriptor.num_tokens == 1:
                compiled_callable = self._base_one_compiled_callable
            else:
                compiled_callable = self._base_compiled_callable

    if use_base_callable and compiled_callable is self._base_compiled_callable and not self._base_dynamic_inputs_marked:
        self._mark_variant_dynamic_inputs(*args, **kwargs)
        self._base_dynamic_inputs_marked = True

    if self.use_bytecode_hook:
        if self.vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            return compiled_callable(*args, **kwargs)

        if not self._compiled_bytecode:
            torch._dynamo.eval_frame.remove_from_cache(self.original_code_object())
            return self._call_with_optional_nvtx_range(compiled_callable, *args, **kwargs)
        with self._dispatch_to_compiled_code():
            return self._call_with_optional_nvtx_range(self.forward, *args, **kwargs)

    ctx = (
        nullcontext()
        if self.first_compile or not self.evaluate_guards
        else torch.compiler.set_stance("fail_on_recompile")
    )
    self.first_compile = False
    with wrapper_module._compilation_context(), ctx:
        return self._call_with_optional_nvtx_range(compiled_callable, *args, **kwargs)


def apply_patch() -> None:
    if getattr(TorchCompileWithNoGuardsWrapper, _PATCH_MARKER, False):
        return

    setattr(TorchCompileWithNoGuardsWrapper, _ORIGINAL_INIT_ATTR, _ORIGINAL_INIT)
    TorchCompileWithNoGuardsWrapper.__init__ = _patched_init
    TorchCompileWithNoGuardsWrapper.__call__ = _patched_call
    TorchCompileWithNoGuardsWrapper._punica_has_lora = _punica_has_lora
    TorchCompileWithNoGuardsWrapper._mark_variant_dynamic_inputs = _mark_variant_dynamic_inputs
    setattr(TorchCompileWithNoGuardsWrapper, _PATCH_MARKER, True)
    logger.info_once(
        "Applied Ascend vLLM %s base/LoRA dual compile-graph patch",
        _SUPPORTED_VLLM_COMMIT,
    )


apply_patch()
