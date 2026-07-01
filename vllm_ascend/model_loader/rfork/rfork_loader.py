#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import gc
import time

import torch
import torch.nn as nn
from torch.nn import Module
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.model_loader.rfork.rfork_worker import RForkWorker


@register_model_loader("rfork")
class RForkModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        config = load_config.model_loader_extra_config
        if not isinstance(config, dict):
            raise RuntimeError("RFork requires --model-loader-extra-config to be a JSON object.")

        def _get_extra_config(key: str, default: str = "") -> str:
            value = config.get(key)
            return value if isinstance(value, str) and value else default

        def _get_extra_config_float(key: str, default: float) -> float:
            value = config.get(key)
            parsed_value = default
            if isinstance(value, (int, float)):
                parsed_value = float(value)
            elif isinstance(value, str) and value:
                try:
                    parsed_value = float(value)
                except ValueError:
                    return default

            if parsed_value <= 0:
                return default

            return parsed_value

        self.model_url = _get_extra_config("model_url", "")
        self.model_deploy_strategy_name = _get_extra_config("model_deploy_strategy_name", "")
        self.scheduler_url = _get_extra_config("rfork_scheduler_url", "")
        self.seed_timeout_sec = _get_extra_config_float("rfork_seed_timeout_sec", 5.0)
        self.seed_key_separator = _get_extra_config("rfork_seed_key_separator", "$")

        logger.info(
            "Initializing rfork with config: "
            "MODEL_URL=%s, MODEL_DEPLOY_STRATEGY_NAME=%s, "
            "SCHEDULER_URL=%s, SEED_TIMEOUT_SEC=%s, "
            "SEED_KEY_SEPARATOR=%s",
            self.model_url,
            self.model_deploy_strategy_name,
            self.scheduler_url,
            self.seed_timeout_sec,
            self.seed_key_separator,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def _ensure_rfork_worker(self, vllm_config: VllmConfig) -> RForkWorker:
        rfork_worker = getattr(self.load_config, "rfork_worker", None)
        if rfork_worker is None:
            kv_transfer_config = vllm_config.kv_transfer_config
            disaggregation_mode = "kv_both" if kv_transfer_config is None else str(kv_transfer_config.kv_role)
            is_draft_model = (
                getattr(vllm_config.model_config, "runner_type", None) == "draft"
                or getattr(vllm_config.scheduler_config, "runner_type", None) == "draft"
            )
            device_id = torch.distributed.get_rank()
            self.load_config.rfork_worker = RForkWorker(
                disaggregation_mode=disaggregation_mode,
                node_rank=vllm_config.parallel_config.node_rank,
                tp_rank=get_tensor_model_parallel_rank(),
                device_id=device_id,
                scheduler_url=self.scheduler_url,
                model_url=self.model_url,
                model_deploy_strategy_name=self.model_deploy_strategy_name,
                seed_timeout_sec=self.seed_timeout_sec,
                seed_key_separator=self.seed_key_separator,
                is_draft_model=is_draft_model,
            )
            logger.info("RFork worker initialized, load_format=rfork")
            rfork_worker = self.load_config.rfork_worker
        return rfork_worker

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> Module | None:
        device_config = vllm_config.device_config
        load_config = self.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            need_del = False
            rfork_worker = self._ensure_rfork_worker(vllm_config)
            try:
                if not rfork_worker.is_seed_available():
                    raise RuntimeError("seed is not available.")

                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
                    need_del = True

                weight_load_start_time = time.time()
                if not rfork_worker.pre_transfer(model):
                    raise RuntimeError("pre_transfer failed.")
                if not rfork_worker.transfer(model):
                    raise RuntimeError("transfer failed.")
                if not rfork_worker.post_transfer():
                    raise RuntimeError("post_transfer failed.")
                logger.info(
                    "Loading model weights took %.2f seconds",
                    time.time() - weight_load_start_time,
                )

                rfork_worker.start_seed_service(model)
                process_weights_after_loading(model, model_config, target_device)

                return model.eval()
            except Exception as e:
                logger.warning(f"RFork transfer failed: {e}, clean up and fall back to default loader")

                rfork_worker.post_transfer()

                if need_del:
                    del model
                    gc.collect()
                    torch.npu.empty_cache()
                    for _ in range(3):
                        gc.collect()
                        torch.npu.empty_cache()

                self.load_config.load_format = "auto"
                self.load_config.model_loader_extra_config = {}

                from vllm.model_executor.model_loader import get_model

                model = get_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                try:
                    rfork_worker.start_seed_service(model)
                except Exception as e:
                    logger.warning(
                        "Fallback model loaded, but start_seed_service failed: %s",
                        e,
                    )
                return model
