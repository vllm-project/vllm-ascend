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
import vllm.envs as envs
from vllm.logger import logger


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config=None):
        if vllm_config is None:
            self._is_initialized = False
            return

        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        torchair_graph_config = additional_config.get("torchair_graph_config",
                                                      {})
        self.torchair_graph_config = TorchairGraphConfig(torchair_graph_config)

        ascend_scheduler_config = additional_config.get(
            "ascend_scheduler_config", {})
        self.ascend_scheduler_config = AscendSchedulerConfig(
            ascend_scheduler_config)

        self.expert_tensor_parallel_size = int(
            additional_config.get("expert_tensor_parallel_size", 1))

        self._is_initialized = True

    @property
    def is_initialized(self):
        return self._is_initialized


class TorchairGraphConfig:
    """
    Configuration Object for torchair_graph_config from additional_config
    """

    def __init__(self, torchair_graph_config):
        self.enabled = torchair_graph_config.get("enabled", False)
        self.use_cached_graph = torchair_graph_config.get(
            "use_cached_graph", False)


class AscendSchedulerConfig:
    """
    Configuration Object for ascend_scheduler_config from additional_config
    """

    def __init__(self, ascend_scheduler_config):
        self.enabled = ascend_scheduler_config.get("enabled", False)


ASCEND_CONFIG = AscendConfig()


def init_ascend_config(vllm_config):
    global ASCEND_CONFIG
    if not ASCEND_CONFIG.is_initialized:
        ASCEND_CONFIG = AscendConfig(vllm_config)


def check_ascend_config(vllm_config, enforce_eager):
    global ASCEND_CONFIG

    # Both for V0 and V1 Engine, torchair_graph cannot be enabled with eager mode.
    if ASCEND_CONFIG.torchair_graph_config.enabled and not enforce_eager:
        raise RuntimeError(
            "Can't enable graph mode and eager mode at the same time. Please set `enforce_eager=False` if you attempt to enable NPU graph mode."
        )

    # torchair_graph only work with deepseek model and mla enabled.
    if ASCEND_CONFIG.torchair_graph_config.enabled:
        if envs.VLLM_MLA_DISABLE:
            logger.warning(
                "Torchair graph mode is still experimental and not supported for V1 without mla currently, "
                "it has been disabled automatically.")
            ASCEND_CONFIG.ascend_scheduler_config.enabled = False
        if vllm_config.model_config:
            model_type = vllm_config.model_config.hf_config.model_type
            if "deepseek" not in model_type:
                raise NotImplementedError(
                    "Torchair graph mode only works with deepseek model.")

    # for V1 Engine, aclgraph doesn't work with deepseek model and only qwen model is well tested.
    if envs.VLLM_USE_V1 and vllm_config.model_config is not None and not enforce_eager:
        model_type = vllm_config.model_config.hf_config.model_type
        if "deepseek" in model_type:
            raise NotImplementedError(
                "ACL Graph does not support deepseek. Please "
                "try torchair graph mode to serve deepseek models on vllm-ascend."
                " Or set `enforce_eager=True` to use eager mode.")
        if "qwen" not in model_type:
            logger.warning(
                "ACL Graph is currently experimental. Please "
                "raise an issue on https://github.com/vllm-project/vllm-ascend/issues"
                " if you encourage any Error")
