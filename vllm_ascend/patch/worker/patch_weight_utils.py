import sys
from typing import Any

from vllm.logger import logger
from vllm.model_executor.model_loader.weight_utils import maybe_remap_kv_scale_name


class ImportPatchDecorator:
    """Import patch decorator"""

    _patches: dict[str, Any] = {}

    @classmethod
    def register(cls, module_name):
        """Decorator for registering module patches"""

        def decorator(func):
            cls._patches[module_name] = func
            return func

        return decorator

    @classmethod
    def apply_patches(cls):
        """Apply all patches"""
        for module_name, patch_func in cls._patches.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                try:
                    patch_func(module)
                except Exception as e:
                    logger.error(f"Patch application failed {module_name}: {e}")


@ImportPatchDecorator.register("vllm.model_executor.models.deepseek_v2")
def patch_deepseek(module):
    ori_maybe_remap_kv_scale_name = maybe_remap_kv_scale_name

    def new_remap(name: str, params_dict: dict):
        name = ori_maybe_remap_kv_scale_name(name, params_dict)

        replace_scale_names = [
            "fa_q.scale",
            "fa_k.scale",
            "fa_v.scale",
            "fa_q.offset",
            "fa_k.offset",
            "fa_v.offset",
            "indexer.q_rot",
            "indexer.k_rot",
        ]

        for scale_name in replace_scale_names:
            if name.endswith(scale_name):
                remap_name = name.replace(scale_name, f"mla_attn.mla_attn.{scale_name}")
                if remap_name in params_dict:
                    return remap_name
                else:
                    return remap_name.replace(".mla_attn", "")

        return name

    if hasattr(module, "maybe_remap_kv_scale_name"):
        module._original_maybe_remap_kv_scale_name = module.maybe_remap_kv_scale_name
        module.maybe_remap_kv_scale_name = new_remap


@ImportPatchDecorator.register("vllm.model_executor.model_loader.weight_utils")
def patch_weight_utils(module):
    if "vllm.model_executor.models.deepseek_v2" in sys.modules:
        deepseek = sys.modules["vllm.model_executor.models.deepseek_v2"]
        if hasattr(deepseek, "maybe_remap_kv_scale_name"):
            module.maybe_remap_kv_scale_name = deepseek.maybe_remap_kv_scale_name


original_import = __builtins__["__import__"]  # type: ignore


def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)

    if name in ImportPatchDecorator._patches:
        try:
            ImportPatchDecorator._patches[name](module)
        except Exception as e:
            logger.error(f"Patch application failed during import {name}: {e}")

    return module


__builtins__["__import__"] = patched_import

ImportPatchDecorator.apply_patches()
