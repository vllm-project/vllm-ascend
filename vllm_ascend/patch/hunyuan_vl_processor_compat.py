# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

from vllm_ascend.utils import vllm_version_is


def install_hunyuan_vl_processor_compat() -> bool:
    """Expose the 0706 Hunyuan processor on the pinned Transformers version."""
    if vllm_version_is("0.23.0"):
        return False

    import transformers

    try:
        _native_processor = transformers.HunYuanVLProcessor
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass
    else:
        return False

    from vllm_ascend.patch.transformers_compat import (
        HunYuanVLImageProcessor,
        HunYuanVLProcessor,
        smart_resize,
    )

    transformers.HunYuanVLProcessor = HunYuanVLProcessor

    parent_name = "transformers.models.hunyuan_vl"
    image_module_name = f"{parent_name}.image_processing_hunyuan_vl"

    parent_module = sys.modules.get(parent_name)
    if parent_module is None:
        parent_module = ModuleType(parent_name)
        parent_module.__package__ = "transformers.models"
        parent_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[parent_name] = parent_module

    image_module = ModuleType(image_module_name)
    image_module.__package__ = parent_name
    image_module.HunYuanVLImageProcessor = HunYuanVLImageProcessor  # type: ignore[attr-defined]
    image_module.smart_resize = smart_resize  # type: ignore[attr-defined]
    image_module.__all__ = ["HunYuanVLImageProcessor", "smart_resize"]  # type: ignore[attr-defined]
    sys.modules[image_module_name] = image_module

    parent_module.HunYuanVLProcessor = HunYuanVLProcessor  # type: ignore[attr-defined]
    parent_module.image_processing_hunyuan_vl = image_module  # type: ignore[attr-defined]

    import transformers.models as transformers_models

    transformers_models.hunyuan_vl = parent_module
    return True
