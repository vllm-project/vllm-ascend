# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

import vllm.model_executor.layers.rotary_embedding.mrope_vit_setup as mrope_vit_setup

from vllm_ascend.models.mrope_vit_setup import (
    _vit_mrope_setup_kernel as _vit_mrope_setup_kernel,
)
from vllm_ascend.models.mrope_vit_setup import (
    vit_mrope_setup,
)

mrope_vit_setup.vit_mrope_setup = vit_mrope_setup


# Rebind preloaded aliases, including Ascend's directly loaded vision tower.
# Later imports receive the patched wrapper from the source module above.
def _patch_loaded_vision_towers():
    for module_name in (
        "vllm.models.minimax_m3.common.vision_tower",
        "vllm_ascend.models._vllm_minimax_m3_common_vision_tower",
    ):
        if module_name in sys.modules:
            vars(sys.modules[module_name])["vit_mrope_setup"] = vit_mrope_setup


_patch_loaded_vision_towers()
