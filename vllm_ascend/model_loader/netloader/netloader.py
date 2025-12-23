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
#

import gc
import json
import time
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torch import nn
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from .interaction.elastic import ElasticServer
from .load import elastic_load
from .utils import find_free_port, is_valid_path_prefix


@register_model_loader("netloader")
class ModelNetLoaderElastic(BaseModelLoader):
    """
    A model loader that uses elastic loading for loading weights.
    """
    source: Optional[List[dict]]
    model_path: Optional[str]
    listen_port: Optional[int]
    int8_cache: str
    int8_cache_name: Optional[List[str]]
    output_prefix: Optional[str]

    def __init__(self, load_config: LoadConfig):
        """
        Initializes the ModelNetLoaderElastic with configuration.

        Parameters:
        - load_config: Configuration for loading the model.
        """
        super().__init__(load_config)

        config = None

        # Try to read config file at first
        extra = load_config.model_loader_extra_config
        if extra and "CONFIG_FILE" in extra:
            try:
                logger.info(
                    f"Reading configs in file {load_config.model_loader_extra_config['CONFIG_FILE']} ..."
                )
                with open(extra["CONFIG_FILE"], 'r') as f:
                    config = json.load(f)
            except FileNotFoundError:
                logger.error("CONFIG_FILE not found")
            except json.JSONDecodeError:
                logger.error("CONFIG_FILE is not a valid JSON file")
            except Exception as e:
                logger.error(
                    f"Unexpected error while reading CONFIG_FILE: {e}")

        if config is None and extra:
            logger.info("Reading configs in model_loader_extra_config ...")
            config = extra
        config = config or {}

        for key, attr, checker, caster, default in [
            ("SOURCE", "source", lambda v: isinstance(v, list), lambda v: v,
             None),
            ("MODEL", "model_path", lambda v: isinstance(v, str), lambda v: v,
             None),
            ("LISTEN_PORT", "listen_port", lambda v: isinstance(v, int) or
             (isinstance(v, str) and v.isdigit()), lambda v: int(v), None),
            ("INT8_CACHE", "int8_cache", lambda v: isinstance(v, str) and v.
             lower() in ['hbm', 'dram', 'no'], lambda v: v.lower(), 'no'),
            ("INT8_CACHE_NAME", "int8_cache_name",
             lambda v: isinstance(v, list), lambda v: v, None),
            ("OUTPUT_PREFIX", "output_prefix",
             lambda v: isinstance(v, str) and is_valid_path_prefix(v),
             lambda v: v, None),
        ]:
            v = config.get(key, default)
            if not checker(v):
                v = default
            else:
                v = caster(v)
            setattr(self, attr, v)

        logger.info(
            "Initializing elastic Netloader with config: "
            "MODEL=%s, LISTEN_PORT=%s,"
            "SOURCE=%s, INT8_CACHE=%s, INT8_CACHE_NAME=%s,"
            "OUTPUT_PREFIX=%s)",
            self.model_path,
            self.listen_port,
            self.source,
            self.int8_cache,
            self.int8_cache_name,
            self.output_prefix,
        )

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """
        Loads the model using the specified configuration.

        Parameters:
        - vllm_config: Configuration for the VLLM.
        - model_config: Configuration for the model.

        Returns:
        - The loaded model.
        """

        device_config = vllm_config.device_config
        parallel_config = vllm_config.parallel_config

        need_process_weights_after_loading = False

        if self.model_path is None:
            self.model_path = model_config.model
            logger.info(f"model_path is set to {self.model_path}")

        device_id = torch.distributed.get_rank()

        if (self.source is None or not isinstance(self.source, list)
                or device_id not in [
                    one_device["device_id"] for one_device in self.source if
                    isinstance(one_device, dict) and "device_id" in one_device
                ]):
            logger.warning(
                "Did not get valid source info, use DefaultModelLoader")
            model, need_process_weights_after_loading = self.revert_to_default(
                model_config, vllm_config, device_config)

        else:
            target_device = torch.device(device_config.device)

            vllm_config_backup = deepcopy(vllm_config)
            model_config_backup = deepcopy(model_config)

            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config,
                                             model_config=model_config)

                start_elastic_load = time.perf_counter()
                model = elastic_load(
                    model=model,
                    device_id=device_id,
                    model_path=self.model_path,
                    sources=self.source,
                    tp=parallel_config.tensor_parallel_size,
                    pp=parallel_config.pipeline_parallel_size,
                )
                end_elastic_load = time.perf_counter()
                logger.info(
                    f"Elastic load time: {end_elastic_load - start_elastic_load}, rank: {device_id}"
                )
                need_process_weights_after_loading = True

                if model is None:
                    logger.warning(
                        "Netloader elastic loading fails, use load format DefaultModelLoader"
                    )

                    vllm_config = vllm_config_backup
                    model_config = model_config_backup

                    del model
                    gc.collect()
                    if device_config.device_type == 'npu':
                        logger.info("Empty NPU cache")
                        torch.npu.empty_cache()
                    elif device_config.device_type == 'cuda':
                        logger.info("Empty CUDA cache")
                        torch.cuda.empty_cache()

                    model, need_process_weights_after_loading = self.revert_to_default(
                        model_config, vllm_config, device_config)

        start_elastic_server = time.perf_counter()
        # start elastic server
        if model is not None and (
            (self.listen_port and self.listen_port in range(1024, 65535)) or
            (self.listen_port is None)):

            from vllm.utils.network_utils import get_ip
            driver_ip = get_ip()

            if driver_ip == '0.0.0.0':
                logger.error(
                    "Driver IP is not set, skip to start Netloader server")
            else:
                if self.listen_port is None:
                    self.listen_port = find_free_port()
                else:
                    self.listen_port += device_id

                logger.info(
                    f"Start elastic Netloader server, rank: {device_id}, listen port: {driver_ip}:{self.listen_port}"
                )

                if self.output_prefix is not None:
                    try:
                        with open(self.output_prefix + str(device_id) + '.txt',
                                  'w') as file:
                            file.write(f"{driver_ip}:{self.listen_port}")
                        logger.info(
                            f"Successfully wrote server address to file: {self.output_prefix + str(device_id)}"
                        )
                    except FileNotFoundError:
                        logger.error(
                            f"File path {self.output_prefix + str(device_id)} does not exist."
                        )
                    except PermissionError:
                        logger.error(
                            f"No permission to write to file {self.output_prefix + str(device_id)}."
                        )
                    except IOError as e:
                        logger.error(
                            f"I/O error occurred while writing to file {self.output_prefix + str(device_id)}: {e}"
                        )
                    except Exception as e:
                        logger.error(f"Unknown error: {e}")

                try:
                    assert isinstance(
                        self.listen_port, int
                    ), f"listen port should be int but get {self.listen_port}"

                    elastic_server = ElasticServer(
                        driver_ip, self.listen_port, model, device_id,
                        self.model_path, parallel_config.tensor_parallel_size,
                        parallel_config.pipeline_parallel_size,
                        self.int8_cache, self.int8_cache_name)
                    elastic_server.start()
                except Exception as e:
                    logger.error(
                        f"Failed to start Netloader server for rank: {device_id}, details: {e}"
                    )
        else:
            logger.info("Skip to start Netloader server")

        end_elastic_server = time.perf_counter()
        logger.info(
            f"Elastic server start time: {end_elastic_server - start_elastic_server}, rank: {device_id}"
        )

        if need_process_weights_after_loading:
            process_weights_after_loading(model, model_config,
                                          torch.device(device_config.device))

        if model is None:
            logger.error("NetLoader elastic loads model fails")
            return None

        return model.eval()

    def revert_to_default(self, model_config, vllm_config,
                          device_config) -> Tuple[nn.Module, bool]:
        """
        Reverts to the default model loading logic when elastic loading fails or is not applicable.

        This method resets the loader's extra config and load format to defaults,
        then delegates model loading to a DefaultModelLoader.
        If quantization is enabled, it will load the model and then run the
        processing of weights (i.e. applying quantization adjustments) before returning.

        Parameters:
        - model_config: Configuration describing model architecture, quantization, etc.
        - vllm_config: Configuration for vLLM (device, parallelism, dtype, etc).
        - device_config: Configuration for the target device (device type, device id, etc).

        Returns:
        - A tuple (model, need_process_weights_after_loading):
            * model: The loaded `nn.Module` under default loading logic.
            * need_process_weights_after_loading: A boolean flag indicating whether
              weights post-processing (e.g. quantization adjustments) still needs to be applied.
        """
        self.load_config.model_loader_extra_config = {}
        self.load_config.load_format = "auto"
        default_model_loader = DefaultModelLoader(self.load_config)

        if model_config.quantization is None:
            model = default_model_loader.load_model(vllm_config=vllm_config,
                                                    model_config=model_config)
            need_process_weights_after_loading = False
        else:
            logger.warning(
                "Quantization is set, netloader use DefaultModelLoader with process_weights_after_loading "
            )
            need_process_weights_after_loading = True
            target_device = torch.device(device_config.device)
            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = initialize_model(vllm_config=vllm_config,
                                             model_config=model_config)
                default_model_loader.load_weights(model, model_config)
            model = model.eval()

        return model, need_process_weights_after_loading

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        pass
