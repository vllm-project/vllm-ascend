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

_GLOBAL_PATCH_APPLIED = False
_DSPARK_SPECULATORS_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def _ensure_dspark_speculators_patch():
    """Register GLM-5.2 DSpark speculators config through vLLM-Ascend.

    vLLM 0.23 knows DFlash but does not know the speculators
    ``speculators_model_type=dspark`` config used by GLM-5.2. Keep the source
    change in vLLM-Ascend by extending vLLM's runtime registry from the plugin
    entrypoint, instead of patching files under the vLLM package.
    """
    global _DSPARK_SPECULATORS_PATCH_APPLIED
    if _DSPARK_SPECULATORS_PATCH_APPLIED:
        return

    from vllm.transformers_utils.configs.speculators import algos
    from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

    if "dspark" not in algos.SUPPORTED_SPECULATORS_TYPES:

        def update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
            algos.update_dflash(config_dict, pre_trained_config)
            pre_trained_config["architectures"] = ["DSparkDraftModel"]
            aux_offset = int(config_dict.get("aux_hidden_state_layer_offset", 0) or 0)
            if aux_offset:
                aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
                pre_trained_config["eagle_aux_hidden_state_layer_ids"] = [
                    i + aux_offset for i in aux_layer_ids
                ]

            dflash_config = pre_trained_config.setdefault("dflash_config", {})
            dflash_config["source_speculators_model_type"] = "dspark"
            dflash_config["markov_rank"] = config_dict.get("markov_rank", 0)
            dflash_config["enable_confidence_head"] = config_dict.get(
                "enable_confidence_head", False
            )
            dflash_config["confidence_head_with_markov"] = config_dict.get(
                "confidence_head_with_markov", False
            )

        algos.SUPPORTED_SPECULATORS_TYPES["dspark"] = update_dspark

    original = SpeculatorsConfig.build_vllm_speculative_config
    if not getattr(original, "_ascend_dspark_patched", False):

        def build_vllm_speculative_config(cls, config_dict: dict) -> dict:
            result = original(config_dict)
            if result.get("method") == "dspark":
                result["method"] = "dflash"
                result["parallel_drafting"] = True
            return result

        build_vllm_speculative_config._ascend_dspark_patched = True
        SpeculatorsConfig.build_vllm_speculative_config = classmethod(
            build_vllm_speculative_config
        )

    _DSPARK_SPECULATORS_PATCH_APPLIED = True


def register():
    """Register the NPU platform."""

    _ensure_dspark_speculators_patch()
    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_connector()
    register_engine()


def register_model_loader():
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    _ensure_dspark_speculators_patch()

    from .models import register_model

    register_model()


import vllm_ascend.logger  # noqa: E402, F401
