import importlib
import sys
import types
from typing import Any, Dict, Optional, Type

from vllm.logger import logger

from .func_wrapper import (wrapper_rmsnorm_forward_oot, wrapper_rmsnorm_init,
                           wrapper_vocab_parallel_embedding_init)
from .w4a8_dynamic import (AscendW4A8DynamicFusedMoEMethod,
                           AscendW4A8DynamicLinearMethod)
from .w8a8 import (AscendC8KVCacheMethod, AscendW8A8FusedMoEMethod,
                   AscendW8A8LinearMethod)
from .w8a8_dynamic import (AscendW8A8DynamicFusedMoEMethod,
                           AscendW8A8DynamicLinearMethod)

patched = False

ASCEND_QUANTIZATION_METHOD_MAP: Dict[str, Dict[str, Type[Any]]] = {
    "W4A8_DYNAMIC": {
        "linear": AscendW4A8DynamicLinearMethod,
        "moe": AscendW4A8DynamicFusedMoEMethod,
    },
    "W8A8": {
        "linear": AscendW8A8LinearMethod,
        "moe": AscendW8A8FusedMoEMethod,
        "attention": AscendC8KVCacheMethod,
    },
    "W8A8_DYNAMIC": {
        "linear": AscendW8A8DynamicLinearMethod,
        "moe": AscendW8A8DynamicFusedMoEMethod,
    },
    "C8": {
        "attention": AscendC8KVCacheMethod,
    },
}


def get_linear_quant_type(quant_description: Dict[str, Any], prefix: str,
                          packed_modules_mapping: Dict[str, Any]):
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            shard_quant_type = quant_description[shard_prefix + '.weight']

            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                raise ValueError(
                    f"Not all shards of {prefix} are quantized with same quant type."
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                    f"use {quant_type}. Please check quantization config.")
    else:
        quant_type = quant_description[prefix + '.weight']
    return quant_type


def get_quant_method(quant_description: Dict[str, Any],
                     prefix: str,
                     layer_type: str,
                     packed_modules_mapping: Optional[Dict[str, Any]] = None):
    apply_quantization_patch(quant_description)
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()
    # Attention
    if '.attn' in prefix and 'fa_quant_type' in quant_description.keys():
        quant_type = quant_description['fa_quant_type']
    # Use KVCache int8
    elif '.attn' in prefix and 'kv_quant_type' in quant_description.keys():
        quant_type = quant_description['kv_quant_type']
    # Linear
    else:
        quant_type = get_linear_quant_type(quant_description, prefix,
                                           packed_modules_mapping)
    if quant_type in ASCEND_QUANTIZATION_METHOD_MAP.keys():
        method_map = ASCEND_QUANTIZATION_METHOD_MAP[quant_type]
        if layer_type in method_map.keys():
            method_cls = method_map[layer_type]
            return method_cls()
        else:
            raise NotImplementedError(
                f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}."
            )
    raise NotImplementedError("Currently, vLLM Ascend only supports following quant types:" \
                                f"{list(ASCEND_QUANTIZATION_METHOD_MAP.keys())}")


def apply_quantization_patch(quant_description):
    global patched
    if patched:
        return
    for name in quant_description.keys():
        if "norm.bias" in name:
            apply_patch("vllm.model_executor.layers.layernorm.RMSNorm",
                        "__init__", [wrapper_rmsnorm_init])
            apply_patch("vllm_ascend.ops.layernorm.AscendRMSNorm",
                        "forward_oot", [wrapper_rmsnorm_forward_oot])
            apply_patch(
                "vllm_ascend.ops.vocab_parallel_embedding.AscendVocabParallelEmbedding",
                "__init__", [wrapper_vocab_parallel_embedding_init])
            break
    patched = True
    logger.info("Using the vLLM Ascend Quantization now!")


def apply_patch(target_module, target_function, wrappers):

    original_module, original_function = parse_path(target_module,
                                                    target_function, False)

    original_function_id = id(original_function)

    candidate = original_function
    for wrapper in wrappers:
        candidate = wrapper(candidate)
    if target_function is not None:
        setattr(original_module, target_function, candidate)

    for _, value in sys.modules.copy().items():
        if target_function is None:
            continue
        try:
            attr = getattr(value, target_function, None)
            if attr is not None and id(attr) == original_function_id:
                setattr(value, target_function, candidate)
        except ImportError:
            continue


def parse_path(module_path, function_name, create_dummy):
    """
    Parse module path and resolve/create modules as needed.

    Args:
        module_path: Dot-separated module path
        function_name: Target function name (None for module only)
        create_dummy: Create dummy modules/functions when missing

    Returns:
        Tuple of (resolved module, target function/none)

    Raises:
        ModuleNotFoundError: If module path is invalid and create_dummy=False
        AttributeError: If function is missing and create_dummy=False
    """
    from importlib.machinery import ModuleSpec

    def create_dummy_module(full_path, parent=None):
        """Create and register a placeholder module"""
        dummy = types.ModuleType(full_path)
        dummy.__file__ = "vllm_ascend.dummy_module.py"
        dummy.__spec__ = ModuleSpec(full_path, None)
        sys.modules[full_path] = dummy
        if parent:
            setattr(parent, full_path.split(".")[-1], dummy)
        return dummy

    def create_placeholder_function(func_name):
        """Create dummy function that raises when called"""

        def placeholder(*args, **kwargs):
            raise NotImplementedError(f"Function {func_name} is a placeholder")

        placeholder.__name__ = func_name
        return placeholder

    modules = module_path.split(".")
    current_module = None
    processed_path = []

    for idx, part in enumerate(modules):
        current_path = ".".join(modules[:idx + 1])
        parent_path = ".".join(modules[:idx]) if idx > 0 else None

        try:
            current_module = importlib.import_module(current_path)
        except ModuleNotFoundError:
            # Handle missing module
            parent = importlib.import_module(
                parent_path) if parent_path else None
            if parent and hasattr(parent, part):
                # Use existing attribute from parent
                current_module = getattr(parent, part)
                # Check for early function resolution
                if function_name and hasattr(current_module, function_name):
                    return current_module, getattr(current_module,
                                                   function_name)
                if function_name and create_dummy:
                    ph_func = create_placeholder_function(function_name)
                    setattr(current_module, function_name, ph_func)
                    return current_module, ph_func
                if function_name:
                    raise AttributeError(
                        f"Function {function_name} missing in {current_path}")
            else:
                if not create_dummy:
                    raise
                # Create and register dummy module
                current_module = create_dummy_module(
                    current_path,
                    parent=importlib.import_module(parent_path)
                    if parent_path else None)

        processed_path.append(part)

    # Final function handling
    final_module = sys.modules[module_path]
    if function_name is not None:
        if not hasattr(final_module, function_name):
            if create_dummy:
                ph_func = create_placeholder_function(function_name)
                setattr(final_module, function_name, ph_func)
            else:
                setattr(final_module, function_name, None)
        return final_module, getattr(final_module, function_name)

    return final_module, None
