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

from vllm.distributed.weight_transfer.factory import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)


def register_engine():
    """Register Ascend weight transfer engines as vLLM plugins."""
    _register_sharded_rdt()
    WeightTransferEngineFactory.register_engine(
        "sparse_hccl",
        "vllm_ascend.distributed.weight_transfer.sparse_hccl_engine",
        "SparseHCCLWeightTransferEngine",
    )
    WeightTransferTrainerFactory.register_engine(
        "sparse_hccl",
        "vllm_ascend.distributed.weight_transfer.sparse_hccl_engine",
        "SparseHCCLTrainerWeightTransferEngine",
    )
    WeightTransferEngineFactory.register_engine(
        "hccl",
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "HCCLWeightTransferEngine",
    )
    WeightTransferEngineFactory.register_engine(
        "npu_ipc",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "NPUIPCWeightTransferEngine",
    )
    WeightTransferTrainerFactory.register_engine(
        "npu_ipc",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "NPUIPCTrainerWeightTransferEngine",
    )
    WeightTransferTrainerFactory.register_engine(
        "hccl",
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "HCCLTrainerWeightTransferEngine",
    )


def _register_sharded_rdt():
    """Keep the public backend key and select the transport at lazy load time.

    The pinned upstream factory has no replace/platform-registration API.
    Remove this registry adapter once it supports platform-specific loaders.
    Other platforms retain the original upstream loader.
    """
    import importlib

    for factory, module_name, class_name in (
        (WeightTransferEngineFactory, "sharded_rdt_engine", "ShardedRDTHixlWeightTransferEngine"),
        (WeightTransferTrainerFactory, "sharded_rdt_trainer", "ShardedRDTHixlTrainerWeightTransferEngine"),
    ):
        original = factory._registry.get("sharded_rdt")
        if original is None or getattr(original, "_ascend_sharded_rdt", False):
            continue

        def loader(original=original, module_name=module_name, class_name=class_name):
            from vllm.platforms import current_platform

            if current_platform.device_type != "npu":
                return original()
            module = importlib.import_module(f"vllm_ascend.distributed.weight_transfer.{module_name}")
            return getattr(module, class_name)

        loader._ascend_sharded_rdt = True
        factory._registry["sharded_rdt"] = loader
