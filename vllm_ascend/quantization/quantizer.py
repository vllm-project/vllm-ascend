#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import importlib
import sys
import types
from typing import Any, Dict, List, Optional

from vllm.logger import logger

from .func_wrapper import (wrapper_rmsnorm_forward_oot, wrapper_rmsnorm_init,
                           wrapper_vocab_parallel_embedding_init)
from .w4a8_dynamic import (AscendW4A8DynamicFusedMoEMethod,
                           AscendW4A8DynamicLinearMethod)
from .w8a8 import (AscendC8KVCacheMethod, AscendW8A8FusedMoEMethod,
                   AscendW8A8LinearMethod)
from .w8a8_dynamic import (AscendW8A8DynamicFusedMoEMethod,
                           AscendW8A8DynamicLinearMethod)

CUSTOMIZED_QUANTIZER_TYPE: List[str] = []


class AscendQuantizer:
    """An interface to different quantization implementations for ascend hardwares."""

    @classmethod
    def get_quantizer(cls,
                      quant_config: Dict[str, Any],
                      prefix: str,
                      packed_modules_mapping: Optional[Dict[str,
                                                            Any]] = dict()):
        # TODO: Need a param to choose quantization algorithms.
        quantization_algorithm = ''

        if quantization_algorithm in CUSTOMIZED_QUANTIZER_TYPE:
            return

        return VLLMAscendQuantizer.get_quantizer(quant_config, prefix,
                                                 packed_modules_mapping)

    def build_linear_method(self):
        raise NotImplementedError

    def build_moe_method(self):
        raise NotImplementedError

    def build_attention_method(self):
        raise NotImplementedError


class VLLMAscendQuantizer:
    _instance: Optional[object] = None
    patched = False

    def __init__(self, quant_description):
        if VLLMAscendQuantizer.patched:
            return
        for name in quant_description.keys():
            if "norm.bias" in name:
                VLLMAscendQuantizer.apply_patch(
                    "vllm.model_executor.layers.layernorm.RMSNorm", "__init__",
                    [wrapper_rmsnorm_init])
                VLLMAscendQuantizer.apply_patch(
                    "vllm_ascend.ops.layernorm.AscendRMSNorm", "forward_oot",
                    [wrapper_rmsnorm_forward_oot])
                VLLMAscendQuantizer.apply_patch(
                    "vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding",
                    "__init__", [wrapper_vocab_parallel_embedding_init])
                break
        VLLMAscendQuantizer.patched = True
        logger.info("Using the vLLM Ascend Quantizer version now!")

    @staticmethod
    def apply_patch(target_module, target_function, wrappers):

        original_module, original_function = VLLMAscendQuantizer.parse_path(
            target_module, target_function, False)

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

    @staticmethod
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
                raise NotImplementedError(
                    f"Function {func_name} is a placeholder")

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
                    if function_name and hasattr(current_module,
                                                 function_name):
                        return current_module, getattr(current_module,
                                                       function_name)
                    if function_name and create_dummy:
                        ph_func = create_placeholder_function(function_name)
                        setattr(current_module, function_name, ph_func)
                        return current_module, ph_func
                    if function_name:
                        raise AttributeError(
                            f"Function {function_name} missing in {current_path}"
                        )
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

    @staticmethod
    def build_linear_method():
        raise NotImplementedError(
            "Linear method is not implemented for the current quant type.")

    @staticmethod
    def build_moe_method():
        raise NotImplementedError(
            "MoE method is not implemented for the current quant type.")

    @staticmethod
    def build_attention_method():
        raise NotImplementedError(
            "Attention method is not implemented for the current quant type.")

    @staticmethod
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

    @classmethod
    def get_quantizer(cls,
                      quant_description: Dict[str, Any],
                      prefix: str,
                      packed_modules_mapping: Optional[Dict[str, Any]] = None):
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
            quant_type = cls.get_linear_quant_type(quant_description, prefix,
                                                   packed_modules_mapping)
        if quant_type in SUPPORT_ASCEND_QUANTIZER_TYPE.keys():
            cls = SUPPORT_ASCEND_QUANTIZER_TYPE[quant_type]
            if not cls._instance:
                cls._instance = cls(quant_description)
            return cls._instance
        raise NotImplementedError("Currently, vLLM Ascend only supports following quant types:" \
                                  f"{list(SUPPORT_ASCEND_QUANTIZER_TYPE.keys())}")


class W4A8DYNAMICQuantizer(VLLMAscendQuantizer):

    @staticmethod
    def build_linear_method():
        return AscendW4A8DynamicLinearMethod()

    @staticmethod
    def build_moe_method():
        return AscendW4A8DynamicFusedMoEMethod()


class W8A8Quantizer(VLLMAscendQuantizer):

    @staticmethod
    def build_linear_method():
        return AscendW8A8LinearMethod()

    @staticmethod
    def build_moe_method():
        return AscendW8A8FusedMoEMethod()

    @staticmethod
    def build_attention_method():
        return AscendC8KVCacheMethod()


class W8A8DYNAMICQuantizer(VLLMAscendQuantizer):

    @staticmethod
    def build_linear_method():
        return AscendW8A8DynamicLinearMethod()

    @staticmethod
    def build_moe_method():
        return AscendW8A8DynamicFusedMoEMethod()


SUPPORT_ASCEND_QUANTIZER_TYPE = {
    "W4A8_DYNAMIC": W4A8DYNAMICQuantizer,
    "W8A8": W8A8Quantizer,
    "W8A8_DYNAMIC": W8A8DYNAMICQuantizer,
    "C8": W8A8Quantizer,
}
