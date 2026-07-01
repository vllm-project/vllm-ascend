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
import os
import time
from copy import copy

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


def _is_mtp_hf_config(hf_config: object | None) -> bool:
    if hf_config is None:
        return False

    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower().endswith("_mtp"):
        return True

    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, (list, tuple)):
        return False

    return any(isinstance(architecture, str) and architecture.endswith("MTPModel") for architecture in architectures)


def _is_draft_model_config(model_config: object | None) -> bool:
    if model_config is None:
        return False
    if getattr(model_config, "runner_type", None) == "draft":
        return True

    return any(
        _is_mtp_hf_config(getattr(model_config, hf_config_attr, None))
        for hf_config_attr in ("hf_config", "hf_text_config")
    )


def _is_draft_model(vllm_config: VllmConfig, model_config: ModelConfig | None = None) -> bool:
    return (
        _is_draft_model_config(model_config)
        or _is_draft_model_config(getattr(vllm_config, "model_config", None))
        or _is_draft_model_config(getattr(vllm_config, "scheduler_config", None))
    )


def _get_rfork_worker_attr(vllm_config: VllmConfig, model_config: ModelConfig) -> str:
    return "rfork_draft_worker" if _is_draft_model(vllm_config, model_config) else "rfork_worker"


def _make_fallback_load_config(load_config: LoadConfig) -> LoadConfig:
    fallback_load_config = copy(load_config)
    fallback_load_config.load_format = "auto"
    fallback_load_config.model_loader_extra_config = {}
    return fallback_load_config


def _is_layer_sharding_enabled(vllm_config: VllmConfig) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("layer_sharding"))


@register_model_loader("rfork")
class RForkModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        config = load_config.model_loader_extra_config
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            err_msg = "RFork requires --model-loader-extra-config to be a JSON object."
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        def _get_extra_config(key: str, default: str = "") -> str:
            value = config.get(key)
            if value is None or not isinstance(value, str):
                value = os.environ.get(key.upper())
            return value if isinstance(value, str) and value else default

        def _get_extra_config_float(key: str, default: float) -> float:
            value = config.get(key)
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
                value = os.environ.get(key.upper())
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

    def _ensure_rfork_worker(self, vllm_config: VllmConfig, model_config: ModelConfig) -> RForkWorker:
        worker_attr = _get_rfork_worker_attr(vllm_config, model_config)
        rfork_worker = getattr(self.load_config, worker_attr, None)
        if rfork_worker is None:
            kv_transfer_config = vllm_config.kv_transfer_config
            disaggregation_mode = "kv_both" if kv_transfer_config is None else str(kv_transfer_config.kv_role)
            is_draft_model = _is_draft_model(vllm_config, model_config)
            device_id = torch.distributed.get_rank()
            rfork_worker = RForkWorker(
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
            setattr(self.load_config, worker_attr, rfork_worker)
            logger.info(
                "RFork worker initialized, load_format=rfork, is_draft_model=%s, worker_attr=%s",
                is_draft_model,
                worker_attr,
            )
        return rfork_worker

    def _requires_processed_layout_transfer(self, model_config: ModelConfig) -> bool:
        return getattr(model_config, "quantization", None) is not None

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
            if _is_layer_sharding_enabled(vllm_config):
                logger.warning(
                    "RFork transfer is disabled when additional_config.layer_sharding "
                    "is enabled; using the default model loader."
                )
                fallback_load_config = _make_fallback_load_config(self.load_config)

                from vllm.model_executor.model_loader import get_model

                try:
                    return get_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        load_config=fallback_load_config,
                        prefix=prefix,
                    )
                except Exception:
                    logger.exception("RFork disabled for layer_sharding, but default loader failed.")
                    raise

            rfork_worker = self._ensure_rfork_worker(vllm_config, model_config)
            processed_layout_transfer = self._requires_processed_layout_transfer(model_config)
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

                if processed_layout_transfer:
                    logger.info("RFork uses post-load tensor layout transfer for quantized model.")
                    process_weights_after_loading(model, model_config, target_device)

                weight_load_start_time = time.perf_counter()
                if not rfork_worker.pre_transfer(model):
                    raise RuntimeError("pre_transfer failed.")
                if not rfork_worker.transfer(model):
                    raise RuntimeError("transfer failed.")
                if not rfork_worker.post_transfer():
                    raise RuntimeError("post_transfer failed.")
                logger.info(
                    "Loading model weights took %.2f seconds",
                    time.perf_counter() - weight_load_start_time,
                )

                rfork_worker.start_seed_service(model)
                if not processed_layout_transfer:
                    process_weights_after_loading(model, model_config, target_device)

                return model.eval()
            except Exception as e:
                logger.warning("RFork transfer failed: %s, clean up and fall back to default loader", e)

                rfork_worker.post_transfer()
                rfork_worker.reset_transfer_state()

                if need_del:
                    del model
                    gc.collect()
                    torch.npu.empty_cache()
                    for _ in range(3):
                        gc.collect()
                        torch.npu.empty_cache()

                fallback_load_config = _make_fallback_load_config(self.load_config)

                from vllm.model_executor.model_loader import get_model

                try:
                    model = get_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        load_config=fallback_load_config,
                        prefix=prefix,
                    )
                except Exception:
                    logger.exception("RFork fallback default loader failed.")
                    raise

                try:
                    rfork_worker.reset_transfer_state()
                    rfork_worker.start_seed_service(model)
                except Exception as e:
                    logger.warning(
                        "Fallback model loaded, but start_seed_service failed: %s",
                        e,
                    )
                return model
