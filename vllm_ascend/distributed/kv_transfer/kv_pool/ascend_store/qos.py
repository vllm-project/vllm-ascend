# SPDX-License-Identifier: Apache-2.0
"""Request KV priority policy. Labels are independent of scheduler priority."""

import logging
from dataclasses import dataclass
from typing import Any

from vllm_ascend.ai_qos import normalize_kv_priority


@dataclass(frozen=True)
class KvQosPolicy:
    priority_to_qos: dict[int, int]
    default_priority: int
    resource_config: dict[str, Any] | None
    level_names: bool = False
    request_priority_enabled: bool = True
    log_enabled: bool = False

    @classmethod
    def from_config(cls, config):
        if config is None:
            return None
        if not isinstance(config, dict) or set(config) - {
            "priority_to_qos",
            "default_priority",
            "resource_config",
            "level_names",
            "request_priority",
            "log",
        }:
            raise ValueError("invalid kv_qos configuration")
        source = config.get("priority_to_qos")
        if not isinstance(source, dict) or not source:
            raise ValueError("kv_qos.priority_to_qos must be a nonempty mapping")
        mapping = {}
        for key, qos in source.items():
            if not (type(key) is int or isinstance(key, str) and key.isdecimal()):
                raise ValueError("priority labels must be integers in 0..255")
            priority = int(key)
            if not 0 <= priority <= 255 or priority in mapping:
                raise ValueError("invalid or duplicate priority label")
            if type(qos) is not int or not 0 <= qos <= 7:
                raise ValueError("QoS must be an integer in 0..7")
            mapping[priority] = qos
        default = config.get("default_priority", 0)
        if type(default) is not int or default not in mapping:
            raise ValueError("default_priority must be a configured integer label")
        base = config.get("resource_config")
        if base is not None and (not isinstance(base, dict) or "store" in base or "comm_resource_config" in base):
            raise ValueError("resource_config must use flat literal keys")
        flags = [config.get("level_names", False), config.get("request_priority", True), config.get("log", False)]
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("level_names/request_priority/log must be booleans")
        if flags[0] and mapping != {0: 0, 3: 3, 7: 7}:
            raise ValueError("named KV levels require identity mapping 0/3/7")
        return cls(mapping, default, base, *flags)

    def select(self, priority):
        if type(priority) is not int or priority not in self.priority_to_qos:
            raise ValueError(f"unmapped KV priority: {priority!r}")
        return self.priority_to_qos[priority]

    def resolve_priority(self, params):
        if params is not None and not isinstance(params, dict):
            raise ValueError("kv_transfer_params must be an object")
        supplied = (params or {}).get("kv_priority", self.default_priority)
        priority = supplied if self.request_priority_enabled else self.default_priority
        if self.level_names:
            priority = normalize_kv_priority(priority)
        self.select(priority)
        return priority

    def request_priority(self, request):
        params = request.kv_transfer_params
        priority = self.resolve_priority(params)
        qos = self.select(priority)
        if self.log_enabled:
            logging.getLogger(__name__).info(
                "KV_QOS_MAP request=%s supplied=%r priority=%d lane_qos=%d request_priority=%s",
                getattr(request, "request_id", "input-validation"),
                (params or {}).get("kv_priority", "omitted"),
                priority,
                qos,
                self.request_priority_enabled,
            )
        return priority


def validate_request_qos_policy(extra_config):
    extra_config = extra_config or {}
    policy = KvQosPolicy.from_config(extra_config.get("kv_qos"))
    if policy is not None and "qos_priority" in extra_config:
        raise ValueError("kv_qos conflicts with explicit qos_priority")
    return policy


def validate_qos_mode(extra_config):
    policy = validate_request_qos_policy(extra_config)
    if policy is not None:
        if extra_config.get("backend", "mooncake").lower() != "mooncake":
            raise ValueError("kv_qos requires backend=mooncake")
        if extra_config.get("use_layerwise", False):
            raise ValueError("kv_qos currently requires use_layerwise=false")
    return policy
