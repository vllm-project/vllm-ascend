# SPDX-License-Identifier: Apache-2.0
"""Ascend QoS configuration from additional_config.ai_qos; KV only."""

import copy

LEVEL_TO_QOS = {"low": 0, "medium": 3, "high": 7}


def normalize_kv_priority(value):
    """One normalization at ingress; internal priority == final QoS number."""
    if isinstance(value, str):
        name = value.strip().lower()
        if name in LEVEL_TO_QOS:
            return LEVEL_TO_QOS[name]
    # Needed for the P -> D handoff and serialized worker metadata.
    if type(value) is int and value in LEVEL_TO_QOS.values():
        return value
    raise ValueError("kv_priority must be low/medium/high (case-insensitive), or canonical 0/3/7")


def validate_ai_qos(config):
    if not isinstance(config, dict) or set(config) - {"op_submit", "collective_communication", "kv_transfer"}:
        raise ValueError("ai_qos supports op_submit, collective_communication and kv_transfer sections")
    # Never accept a configuration and silently claim an unimplemented domain.
    for domain in ("op_submit", "collective_communication"):
        if domain in config:
            raise ValueError(
                f"{domain} is not wired in this KV-only update; "
                "enable it after integrating that domain's implementation"
            )
    if "kv_transfer" not in config:
        return copy.deepcopy(config)
    value = config["kv_transfer"]
    if not isinstance(value, dict) or set(value) - {"enabled", "request_priority", "default_priority", "log"}:
        raise ValueError("invalid ai_qos.kv_transfer configuration")
    out = dict(enabled=True, request_priority=False, default_priority=0, log=False)
    out.update(value)
    for name in ("enabled", "request_priority", "log"):
        if type(out[name]) is not bool:
            raise ValueError(f"kv_transfer.{name} must be a JSON boolean")
    out["default_priority"] = normalize_kv_priority(out["default_priority"])
    return {"kv_transfer": out}


def apply_config(vllm_config):
    additional = vllm_config.additional_config or {}
    if "ai_qos" not in additional:
        return
    normalized = validate_ai_qos(additional["ai_qos"])
    kv = normalized.get("kv_transfer")
    if kv is None:
        return
    transfer = vllm_config.kv_transfer_config
    if not kv["enabled"]:
        if transfer is not None and transfer.kv_connector_extra_config.get("kv_qos") is not None:
            raise ValueError("kv_transfer.enabled=false conflicts with legacy kv_qos")
        return
    if transfer is None or transfer.kv_connector not in (
        "AscendStoreConnector",
        "MooncakeConnectorV1",
        "MooncakeLayerwiseConnector",
    ):
        raise ValueError(
            "KV QoS requires --kv-transfer-config with AscendStoreConnector, "
            "MooncakeConnectorV1 or MooncakeLayerwiseConnector"
        )
    extra = transfer.kv_connector_extra_config
    if transfer.kv_connector == "AscendStoreConnector":
        if extra.get("backend", "mooncake").lower() != "mooncake" or extra.get("use_layerwise", False):
            raise ValueError("KV QoS requires mooncake and use_layerwise=false")
    previous = extra.get("kv_qos")
    desired = dict(
        priority_to_qos={"0": 0, "3": 3, "7": 7},
        default_priority=kv["default_priority"],
        level_names=True,
        request_priority=kv["request_priority"],
        log=kv["log"],
    )
    if previous is not None:
        if not isinstance(previous, dict):
            raise ValueError("legacy kv_qos must be an object")
        # The transport protocol is orthogonal and may be kept from the old config.
        if "resource_config" in previous:
            desired["resource_config"] = copy.deepcopy(previous["resource_config"])
        previous_policy = {k: v for k, v in previous.items() if k != "resource_config"}
        if previous_policy and previous_policy != {k: v for k, v in desired.items() if k != "resource_config"}:
            raise ValueError(
                "additional_config.ai_qos conflicts with old kv_qos mapping; remove the old mapping (including swapped)"
            )
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import validate_request_qos_policy

    validate_request_qos_policy(dict(extra, kv_qos=desired))
    extra["kv_qos"] = desired
    # Public API inputs are checked before engine scheduling, so bad labels do
    # not first fail inside the scheduler process.
    from vllm_ascend.patch.platform.patch_ai_qos_request import install_request_validation

    install_request_validation()
