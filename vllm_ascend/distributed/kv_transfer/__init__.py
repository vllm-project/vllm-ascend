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

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory


def register_connector():
    # override multi_connector as ascend_multi_connector
    if "MultiConnector" in KVConnectorFactory._registry:
        KVConnectorFactory._registry.pop("MultiConnector")
    KVConnectorFactory.register_connector(
        "MultiConnector", "vllm_ascend.distributed.kv_transfer.ascend_multi_connector", "AscendMultiConnector"
    )

    KVConnectorFactory.register_connector(
        "MooncakeConnectorV1", "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector", "MooncakeConnector"
    )

    KVConnectorFactory.register_connector(
        "MooncakeConnectorV2",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector",
        "MooncakeConnector",
    )

    KVConnectorFactory.register_connector(
        "MooncakePullConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector",
        "MooncakePullConnector",
    )

    KVConnectorFactory.register_connector(
        "MooncakeHybridConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
        "MooncakeConnector",
    )

    KVConnectorFactory.register_connector(
        "MooncakeConnectorStoreV1",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    )

    KVConnectorFactory.register_connector(
        "AscendStoreConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    )

    KVConnectorFactory.register_connector(
        "MooncakeLayerwiseConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector",
        "MooncakeLayerwiseConnector",
    )

    KVConnectorFactory.register_connector(
        "UCMConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector.connector",
        "UCMConnectorV1",
    )

    # vLLM's native offloading worker assumes attention KV caches are packed
    # into one Tensor. Ascend keeps K/V as separate tensors, so replace only
    # the connector's worker-side canonicalization boundary while reusing the
    # upstream scheduler, manager, metrics, and transfer lifecycle.
    if "OffloadingConnector" in KVConnectorFactory._registry:
        KVConnectorFactory._registry.pop("OffloadingConnector")
    KVConnectorFactory.register_connector(
        "OffloadingConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector",
        "AscendOffloadingConnector",
    )

    # Override the upstream SimpleCPUOffloadConnector with the NPU
    # adaptation that uses aclrtMemcpyBatchAsync + torch.npu streams.
    # Only override if the upstream module exists in this vLLM version.
    try:
        import vllm.v1.simple_kv_offload  # noqa: F401
    except ImportError:
        pass
    else:
        if "SimpleCPUOffloadConnector" in KVConnectorFactory._registry:
            KVConnectorFactory._registry.pop("SimpleCPUOffloadConnector")
        KVConnectorFactory.register_connector(
            "SimpleCPUOffloadConnector",
            "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.simple_cpu_offload_connector",
            "AscendSimpleCPUOffloadConnector",
        )

    KVConnectorFactory.register_connector(
        "RecomputeCPUOffloadConnector",
        "vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.recompute_cpu_offload_connector",
        "RecomputeCPUOffloadConnectorV1",
    )

    KVConnectorFactory.register_connector(
        "SfaRemoteD2HConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector",
        "SfaRemoteD2HConnector",
    )
