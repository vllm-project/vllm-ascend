#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import json
import os

from vllm.logger import init_logger

from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD

logger = init_logger(__name__)


def detect_quantization_method(model_path: str) -> str | None:
    """Auto-detect the quantization method from model directory files.

    This function performs a lightweight check (JSON files and file existence
    only — no .safetensors or .bin inspection) to determine which quantization
    method was used to produce the weights in *model_path*.

    Detection priority:
        1. **ModelSlim (Ascend)** – ``quant_model_description.json`` exists
           in the model directory.
        2. **LLM-Compressor (compressed-tensors)** – ``config.json`` contains
           a ``quantization_config`` section with
           ``"quant_method": "compressed-tensors"``.
        3. **None** – neither condition is met; the caller should fall back to
           the default (float) behaviour.

    Args:
        model_path: Path to the local model directory.

    Returns:
        ``"ascend"`` for ModelSlim models,
        ``"compressed-tensors"`` for LLM-Compressor models,
        or ``None`` if no quantization signature is found.
    """
    if not os.path.isdir(model_path):
        return None

    # Case 1: ModelSlim — look for quant_model_description.json
    modelslim_config_path = os.path.join(model_path, MODELSLIM_CONFIG_FILENAME)
    if os.path.isfile(modelslim_config_path):
        return ASCEND_QUANTIZATION_METHOD

    # Case 2: LLM-Compressor — look for compressed-tensors in config.json
    config_json_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_json_path):
        try:
            with open(config_json_path) as f:
                config = json.load(f)
            quant_cfg = config.get("quantization_config")
            if isinstance(quant_cfg, dict):
                quant_method = quant_cfg.get("quant_method", "")
                if quant_method == COMPRESSED_TENSORS_METHOD:
                    return COMPRESSED_TENSORS_METHOD
        except (json.JSONDecodeError, OSError):
            # Malformed or unreadable config.json — skip silently.
            pass

    # Case 3: No quantization signature found.
    return None


def maybe_auto_detect_quantization(vllm_config) -> None:
    """Auto-detect and apply the quantization method on *vllm_config*.

    This should be called during engine initialisation (from
    ``NPUPlatform.check_and_update_config``) **after** ``VllmConfig`` has been
    created but **before** heavy weights are loaded.

    Because ``check_and_update_config`` runs *after*
    ``VllmConfig.__post_init__`` has already evaluated
    ``_get_quantization_config`` (which returned ``None`` when
    ``model_config.quantization`` was not set), we must:

    1. Set ``model_config.quantization`` to the detected value.
    2. Recreate ``vllm_config.quant_config`` so that the quantization
       pipeline (``get_quant_config`` → ``QuantizationConfig`` →
       ``get_quant_method`` for every layer) is properly initialised.

    Rules:
        * If the user explicitly set ``--quantization``, that value is
          respected.  A warning is emitted when the detected method differs.
        * If no ``--quantization`` was given, the detected method (if any) is
          applied automatically.

    Args:
        vllm_config: A ``vllm.config.VllmConfig`` instance (mutable).
    """
    model_config = vllm_config.model_config
    model_path = model_config.model
    user_quant = model_config.quantization
    detected = detect_quantization_method(model_path)

    if detected is None:
        # No quantization signature found — nothing to do.
        return

    if user_quant is not None:
        # User explicitly specified a quantization method.
        if user_quant != detected:
            logger.warning(
                "Auto-detected quantization method '%s' from model "
                "files at '%s', but user explicitly specified "
                "'--quantization %s'. Respecting the user-specified "
                "value. If you encounter errors during model loading, "
                "consider using '--quantization %s' instead.",
                detected,
                model_path,
                user_quant,
                detected,
            )
        return

    # No user-specified quantization — apply auto-detected value.
    model_config.quantization = detected
    logger.info(
        "Auto-detected quantization method '%s' from model files "
        "at '%s'. To override, pass '--quantization <method>' explicitly.",
        detected,
        model_path,
    )

    # Recreate quant_config on VllmConfig.  The original __post_init__
    # already ran _get_quantization_config(), but at that point
    # model_config.quantization was None so it returned None.  Now that
    # we've set it, we need to build the actual QuantizationConfig so the
    # downstream model-loading code can use it.
    from vllm.config import VllmConfig as _VllmConfig

    vllm_config.quant_config = _VllmConfig._get_quantization_config(model_config, vllm_config.load_config)
