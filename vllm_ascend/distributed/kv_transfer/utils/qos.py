# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
# [Role] Validate KV transfer QoS during connector initialization.
# [Invariants] QoS is opt-in and uses integer priorities from 0 through 7.
# [Timeouts & Locks] No blocking operations or shared state.
# [Side-Effect Policy] Emit a warning for priorities above 4.

from typing import Any

from vllm.logger import logger

QOS_MAX = 7
QOS_WARNING_THRESHOLD = 4


def validate_qos(qos: Any, source: str) -> int:
    if isinstance(qos, bool) or not isinstance(qos, int) or not 0 <= qos <= QOS_MAX:
        raise ValueError(f"Invalid QoS {qos!r} in {source}. QoS must be an integer in [0, {QOS_MAX}].")
    return qos


def parse_qos_from_extra_config(extra_config: dict[str, Any] | None) -> int | None:
    if not extra_config or "qos_priority" not in extra_config:
        return None
    return validate_qos(extra_config["qos_priority"], "kv_connector_extra_config.qos_priority")


def warn_high_qos(qos: int, source: str) -> None:
    if qos > QOS_WARNING_THRESHOLD:
        logger.warning(
            "QoS %d in %s exceeds %d. Verify that the transfer backend and switch are configured for this priority.",
            qos,
            source,
            QOS_WARNING_THRESHOLD,
        )
