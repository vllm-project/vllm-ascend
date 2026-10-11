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
# Patch vLLM's ECExampleConnector.start_load_caches so that safetensors
# loads are pinned to the correct NPU device index in TP>1 deployments.
#
# Background:
# The upstream method only qualifies the load device with an index when
# current_platform.is_cuda_alike() returns True. Ascend NPU returns False,
# so in TP>1 every rank loads the EC cache to the default device (npu:0),
# causing rank 1 to access cross-device tensors and trigger NPU AI Core
# errcode=95 (MTE DDR address out of range).
#
# The patch only takes effect when the connector is acting as an EC
# consumer. Non-EC deployments and EC producer nodes fall through to the
# original upstream implementation unchanged.

from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnector

_original_start_load_caches = ECExampleConnector.start_load_caches


def _patched_start_load_caches(self, encoder_cache, **kwargs):
    # Only apply the NPU device-index fix on EC consumer nodes. Producers
    # and non-EC deployments keep the original upstream behavior.
    if not getattr(self, "is_consumer", False):
        return _original_start_load_caches(self, encoder_cache, **kwargs)

    import logging

    import safetensors.torch
    from vllm.distributed.ec_transfer.ec_connector.example_connector import ECExampleConnectorMetadata
    from vllm.platforms import current_platform
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    logger = logging.getLogger(__name__)

    metadata = self._get_connector_metadata()
    assert isinstance(metadata, ECExampleConnectorMetadata)
    assert encoder_cache is not None
    if metadata is None:
        logger.warning("In connector.start_load_caches, but the connector metadata is None")
        return

    # Pin the load to the selected device so tensor-parallel workers load
    # the cache locally. On Ascend NPU, is_cuda_alike() returns False but
    # we still need the device index.
    device = current_platform.device_type
    if current_platform.is_cuda_alike() or device == "npu":
        device = f"{device}:{current_platform.current_device()}"

    for mm_data in metadata.mm_datas:
        if mm_data.mm_hash in encoder_cache:
            continue
        filename = self._generate_filename_debug(mm_data.mm_hash)
        with gpu_sync_allowed():
            ec_cache = safetensors.torch.load_file(filename, device=device)["ec_cache"]
        encoder_cache[mm_data.mm_hash] = ec_cache
        logger.debug("Success load encoder cache for hash %s", mm_data.mm_hash)


ECExampleConnector.start_load_caches = _patched_start_load_caches
