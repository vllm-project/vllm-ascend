# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""RFork loader guards for configurations with mutable model storage."""

from typing import Any

# Sessions live on the process-lifetime LoadConfig; see RForkModelLoader._ensure_rfork_session.
_SESSION_ATTRS = ("rfork_session", "rfork_draft_session")


def mutable_weights_bypass_reason(vllm_config: Any, model_config: Any) -> str | None:
    """Return why RFork must not register weights that can later be replaced."""
    if any(
        bool(getattr(config, "enable_sleep_mode", False))
        for config in (model_config, getattr(vllm_config, "model_config", None))
    ):
        return "sleep mode"
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        return "online weight transfer (weight_transfer_config)"
    return None


def _sessions_holding_registered_weights(vllm_config: Any) -> list[str]:
    """Name the RFork sessions that currently have model weight storage registered."""
    load_config = getattr(vllm_config, "load_config", None)
    if load_config is None:
        return []

    holders: list[str] = []
    for attr in _SESSION_ATTRS:
        transfer_backend = getattr(getattr(load_config, attr, None), "transfer_backend", None)
        snapshot = getattr(transfer_backend, "snapshot_registered_weight_blocks", None)
        if callable(snapshot) and snapshot():
            holders.append("draft" if attr == "rfork_draft_session" else "main")
    return holders


def ensure_no_registered_rfork_weights(vllm_config: Any, operation: str) -> None:
    """Reject a runtime operation that would replace weight storage RFork has registered.

    ``mutable_weights_bypass_reason`` keeps RFork away from configurations whose
    weights are replaced later, but it can only read configuration at load time.
    Operations reached purely over RPC carry no such signal, so check the live
    registration instead: while RFork holds a model's storage registered, peers
    may read those addresses at any time, and replacing the tensors underneath
    would serve them a mix of old and new weights.
    """
    holders = _sessions_holding_registered_weights(vllm_config)
    if not holders:
        return
    raise RuntimeError(
        f"{operation} is not supported while RFork holds registered weight storage "
        f"({', '.join(holders)} model). Peers can read those addresses at any time, so the "
        "weights must not be replaced underneath them."
    )
