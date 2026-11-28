# SPDX-License-Identifier: Apache-2.0
"""Mooncake/SFA startup policy without connector or worker imports.

Connector entry points call this policy after parsing Ascend configuration.
Keep runtime imports independent of AscendConfig to avoid re-entering its
initialization through the KV-transfer package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.ascend_config import SparseKVOffloadConfig


def validate_mooncake_sfa_config(vllm_config: VllmConfig, sparse_config: SparseKVOffloadConfig) -> None:
    """Check transport combinations while the outer connector is still visible."""
    if not sparse_config.enabled:
        return
    transfer = vllm_config.kv_transfer_config
    connector = getattr(transfer, "kv_connector", None)
    if connector == "MultiConnector" and (
        transfer.has_connector("MooncakeConnectorV2") or transfer.has_connector("MooncakePullConnector")
    ):
        raise ValueError("Mooncake SFA KV offload currently requires a direct connector, not MultiConnector")
    if connector == "MooncakePullConnector":
        raise ValueError("Mooncake SFA KV offload requires kv_connector='MooncakeConnectorV2'")
    if connector != "MooncakeConnectorV2":
        return
    if not vllm_config.use_v2_model_runner:
        raise ValueError("Mooncake SFA KV offload requires ModelRunner V2")
    if sparse_config.use_fused_overlap:
        raise ValueError("Mooncake SFA KV offload currently requires LIM + Copy-SFA")
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is not None and speculative.method == "dspark":
        raise ValueError("Mooncake SFA KV offload has not adapted the DSpark remote prompt protocol")
    # Device main KV alone does not reset a failed remote PD handoff. Require
    # fail until the connector supports recovery to local prefill.
    if transfer.is_kv_consumer and getattr(transfer, "kv_load_failure_policy", "fail") != "fail":
        raise ValueError(
            "Mooncake SFA KV offload consumers require kv_load_failure_policy='fail'; "
            "failed PD recovery to local prefill is not supported, including keep_device_kv_cache=True"
        )
