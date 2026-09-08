# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import json
import os

from vllm_ascend import envs

QOS_KEY = "comm_resource_config.qos"
PD_QOS_DEFAULT = 1
STORE_QOS_DEFAULT = 0
QOS_MAX = 4


def inject_qos(qos: int, *, store: bool = False) -> None:
    """Merge QoS before engine initialization (connector construction is serial).

    A Store override replaces, rather than inherits, the top-level HIXL
    config. Preserve legacy inherited settings when creating that override.
    """
    if type(qos) is not int or not 0 <= qos <= QOS_MAX:
        raise ValueError("kv_connector_extra_config.qos must be an integer in [0, 4]")
    raw = envs.ASCEND_GLOBAL_RESOURCE_CONFIG
    try:
        config = json.loads(raw) if raw else {}
    except (ValueError, TypeError) as exc:
        raise ValueError("ASCEND_GLOBAL_RESOURCE_CONFIG must be a JSON object") from exc
    if not isinstance(config, dict):
        raise ValueError("ASCEND_GLOBAL_RESOURCE_CONFIG must be a JSON object")
    if "store" in config and not isinstance(config["store"], dict):
        raise ValueError("ASCEND_GLOBAL_RESOURCE_CONFIG.store must be a JSON object")
    if store:
        if "store" not in config:
            config["store"] = dict(config)
        config["store"][QOS_KEY] = qos
    else:
        config[QOS_KEY] = qos
    os.environ[envs.ASCEND_GLOBAL_RESOURCE_CONFIG_ENV] = json.dumps(config)
