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
    sources_this_device = []
    for s in sources:
        if isinstance(
                s, dict
        ) and "device_id" in s and s["device_id"] == device_id and isinstance(
                s["sources"], list):
            sources_this_device += s["sources"]
    if len(sources_this_device) == 0:
        return None

    client_interaction_layer = ElasticClient(sources_this_device, device_id,
                                             model_path, tp, pp)
    if client_interaction_layer.s is None or client_interaction_layer.server_addr is None:
        return None
    ack = client_interaction_layer.ack
    if ack is None:
        return None

    t0 = time.perf_counter()
    elastic_loader = P2PLoad(ack[0], client_interaction_layer.server_addr,
                             ack[1])
    model = elastic_loader.load(model=model)
    if model is None:
        logger.error("Failed to load model")
    else:
        logger.info(
            "Finish elastic load (duration: {}s)".format(time.perf_counter() -
                                                         t0))

    del client_interaction_layer
    return model
