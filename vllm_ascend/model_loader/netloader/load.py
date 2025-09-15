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

import time

from vllm.logger import logger

from .executor.elastic_load import P2PLoad
from .interaction.elastic import ElasticClient


def elastic_load(
    model,
    device_id: int,
    model_path: str,
    sources: list,
    tp: int,
    pp: int,
):
    """
    Loads a model using elastic loading across multiple devices.

    Parameters:
    - model: The model instance to be loaded.
    - device_id: The ID of the current device (i.e. global rank).
    - model_path: The path to the model file.
    - sources: A list of source configurations, each containing device_id and sources.
    - tp: Tensor parallel size, indicating the number of devices for tensor parallelism.
    - pp: Pipeline parallel size, indicating the number of devices for pipeline parallelism.

    Returns:
    - The loaded model if successful, otherwise None.
    """

    # Filter sources for the current device
    sources_this_device = []
    for s in sources:
        if isinstance(
                s, dict
        ) and "device_id" in s and s["device_id"] == device_id and isinstance(
                s["sources"], list):
            sources_this_device += s["sources"]
    if len(sources_this_device) == 0:
        return None

    try:
        # Initialize the interaction layer with the ElasticClient
        with ElasticClient(sources_this_device, device_id, model_path, tp,
                           pp) as client_interaction_layer:
            if client_interaction_layer.s is None or client_interaction_layer.server_addr is None:
                raise RuntimeError(
                    "Failed to initialize ElasticClient: socket or server_addr is None"
                )
            ack = client_interaction_layer.ack
            if ack is None:
                raise RuntimeError("ElasticClient.register did not return ack")

            t0 = time.perf_counter()
            elastic_loader = P2PLoad(ack[0],
                                     client_interaction_layer.server_addr,
                                     ack[1])
            model_loaded = elastic_loader.load(model=model)
            if model_loaded is None:
                logger.error("Failed to load model")
                return None
            logger.info("Finish elastic load (duration: {}s)".format(
                time.perf_counter() - t0))
            return model_loaded
    except Exception as e:
        logger.error(f"elastic_load error: {e}")
        return None
