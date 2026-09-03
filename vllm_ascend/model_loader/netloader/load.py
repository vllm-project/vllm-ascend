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
    group_name: str = "netloader",
    int8_cache: str = "no",
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
    - group_name: Name of the HCCL process group.
    - int8_cache: The type of caching for int8 parameters (HBM, DRAM, or no).

    Returns:
    - The loaded model if successful, otherwise None.
    """

    # Filter sources for the current device
    sources_this_device = []
    for s in sources:
        if isinstance(s, dict) and "device_id" in s and s["device_id"] == device_id and isinstance(s["sources"], list):
            sources_this_device += s["sources"]
    if len(sources_this_device) == 0:
        return None

    try:
        start_elastic_client_join = time.perf_counter()
        # Initialize the interaction layer with the ElasticClient
        with ElasticClient(
            sources_this_device,
            device_id,
            model_path,
            tp,
            pp,
            group_name=group_name,
            int8_cache=int8_cache,
        ) as client_interaction_layer:
            elastic_client_join_time = time.perf_counter() - start_elastic_client_join
            logger.info(
                "Netloader elastic client join time: %s, device_id: %s, group: %s",
                elastic_client_join_time,
                device_id,
                group_name,
            )
            if client_interaction_layer.s is None or client_interaction_layer.server_addr is None:
                raise RuntimeError("Failed to initialize ElasticClient: socket or server_addr is None")
            ack = client_interaction_layer.ack
            if ack is None:
                raise RuntimeError("ElasticClient.register did not return ack")

            start_p2p_load = time.perf_counter()
            elastic_loader = P2PLoad(
                ack[0],
                client_interaction_layer.server_addr,
                ack[1],
                group_name,
                transfer_processed_layout=int8_cache == "no",
                transfer_shape_manifest=client_interaction_layer.transfer_shape_manifest,
            )
            model_loaded = elastic_loader.load(model=model)
            if model_loaded is None:
                logger.error("Failed to load model")
                return None
            logger.info(
                "Netloader P2P load time: %s, device_id: %s, group: %s",
                time.perf_counter() - start_p2p_load,
                device_id,
                group_name,
            )
            return model_loaded
    except Exception as e:
        logger.info("elastic_load error: %s", e)
        return None
