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
# ---------------------------------------------------------------------------
# Fix: LoRA is a no-op (delta == 0) for VLM-wrapped hybrid dense models.
#
# Applies to models whose text tower is wrapped under a submodule such as
# `language_model` (e.g. Qwen3.5 dense variants).
#
# Root cause:
#   * The adapter tensors load correctly, but are keyed by the adapter's bare
#     module names, e.g.  model.layers.N.mlp.down_proj
#   * The model registers its LoRA modules under the wrapped name, e.g.
#     language_model.model.layers.N.mlp.down_proj
#   The weight lookup at activation uses the wrapped model name, misses the
#   bare-keyed adapter weights, and set_lora receives a zero buffer, so the
#   LoRA delta is zero (LoRA output is bit-identical to the base model).
#
# Fix: after WorkerLoRAManager._load_adapter loads the LoRAModel (but before it
# is registered / packed-merged), rename the loras dict keys so they line up
# with the model's actual module names. The prefix is auto-detected from the
# manager's own module table (LoRAModelManager.modules): find a model module
# whose name ends with a (non-packed) adapter key and take the leading
# difference as the prefix. This works unchanged for any wrapper depth.
#
# Once the keys line up:
#   * plain modules (down_proj/o_proj) match directly, and
#   * _create_merged_loras_inplace finds q/k/v & gate/up sub-loras and packs the
#     non-zero qkv_proj / gate_up_proj automatically.
#
# Safety: only touches the real-adapter load path (dummy LoRAs already use
# model names); leaves keys untouched if no prefix is detected (already-aligned
# plain text models); never raises into the serving path. Disable with
# VLLM_ASCEND_LORA_VLM_PREFIX=0.
# ---------------------------------------------------------------------------
import os

from vllm.logger import init_logger
from vllm.lora.worker_manager import WorkerLoRAManager

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_ASCEND_LORA_VLM_PREFIX", "1") not in (
    "0", "", "false", "False")

_orig_load_adapter = WorkerLoRAManager._load_adapter


def _detect_prefix(lora_keys, model_module_names):
    """Return the prefix P such that some model module == P + <adapter_key>.

    Uses only keys that are likely NON-packed (down_proj / o_proj / *_proj that
    already appear verbatim as a model module suffix), so packed q/k/v vs
    qkv_proj naming does not confuse detection. Returns "" if the adapter keys
    already match model modules (nothing to do), or None if no prefix aligns.
    """
    model_set = set(model_module_names)
    for lk in lora_keys:
        # Already aligned? then no prefix needed.
        if lk in model_set:
            return ""
        for mm in model_module_names:
            if mm.endswith("." + lk):
                return mm[:-len(lk)]
    return None


def _load_adapter(self, lora_request):
    lora = _orig_load_adapter(self, lora_request)
    try:
        mgr = self._adapter_manager
        model_module_names = list(getattr(mgr, "modules", {}).keys())
        lora_keys = list(lora.loras.keys())
        if not model_module_names or not lora_keys:
            return lora

        prefix = _detect_prefix(lora_keys, model_module_names)
        if not prefix:
            # None (no alignment) or "" (already aligned): leave keys unchanged.
            return lora

        remapped = {}
        changed = 0
        for key, weights in lora.loras.items():
            new_key = key
            if not key.startswith(prefix):
                new_key = prefix + key
                changed += 1
            remapped[new_key] = weights
        lora.loras = remapped

        logger.debug(
            "[lora-vlm-prefix] detected prefix=%r, remapped %d/%d adapter keys "
            "to align with model modules", prefix, changed, len(lora_keys))
    except Exception as e:  # noqa: BLE001 - never break serving on remap
        logger.warning("[lora-vlm-prefix] remap skipped due to %s: %s",
                       type(e).__name__, e)
    return lora


if _ENABLED:
    WorkerLoRAManager._load_adapter = _load_adapter
