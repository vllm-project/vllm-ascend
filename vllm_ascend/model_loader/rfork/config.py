# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_RFORK_SEED_TIMEOUT_SEC = 5.0
DEFAULT_RFORK_REQUEST_TIMEOUT_SEC = 10.0
DEFAULT_RFORK_HEARTBEAT_INTERVAL_SEC = 30.0
DEFAULT_RFORK_LEASE_RELEASE_MAX_ATTEMPTS = 3
DEFAULT_RFORK_LEASE_RELEASE_RETRY_INTERVAL_SEC = 30.0

# Keys read by RForkConfig.from_extra_config. Keep in sync with that method.
KNOWN_EXTRA_CONFIG_KEYS = frozenset(
    {
        "model_url",
        "model_deploy_strategy_name",
        "rfork_scheduler_url",
        "rfork_seed_timeout_sec",
        "rfork_request_timeout_sec",
        "rfork_model_identity_headers",
        "rfork_heartbeat_interval_sec",
        "rfork_lease_release_max_attempts",
        "rfork_lease_release_retry_interval_sec",
        "rfork_seed_removal_supported",
        "rfork_seed_bind_host",
        "rfork_seed_port_base",
        "rfork_seed_advertise_host",
    }
)


class RForkConfigurationError(Exception):
    """RFork configuration is missing or invalid.

    Deliberately not a ``ValueError``/``RuntimeError`` subclass: the loader treats
    those as recoverable and falls back to the default model loader, while a
    configuration error must abort startup so the operator sees it.
    """


def unknown_extra_config_keys(raw_config: object) -> list[str]:
    """Return extra-config keys this RFork version does not read, such as typos."""
    if not isinstance(raw_config, dict):
        return []
    return sorted(str(key) for key in raw_config if key not in KNOWN_EXTRA_CONFIG_KEYS)


def _string_value(
    config: dict[str, Any],
    key: str,
    env_name: str,
    default: str | None = "",
) -> str | None:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        value = os.getenv(env_name)
    return value if isinstance(value, str) and value else default


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _float_value(
    config: dict[str, Any],
    key: str,
    env_name: str,
    default: float,
) -> float:
    if key in config:
        parsed = _positive_float(config[key])
        if parsed is not None:
            return parsed
    return _positive_float(os.getenv(env_name)) or default


@dataclass(frozen=True, slots=True)
class RForkConfig:
    model_url: str
    model_deploy_strategy_name: str
    planner_url: str
    seed_timeout_sec: float = DEFAULT_RFORK_SEED_TIMEOUT_SEC
    request_timeout_sec: float = DEFAULT_RFORK_REQUEST_TIMEOUT_SEC
    seed_bind_host: str = "0.0.0.0"
    seed_port_base: int = 0
    seed_advertise_host: str | None = None
    heartbeat_interval_sec: float = DEFAULT_RFORK_HEARTBEAT_INTERVAL_SEC
    lease_release_max_attempts: int = DEFAULT_RFORK_LEASE_RELEASE_MAX_ATTEMPTS
    lease_release_retry_interval_sec: float = DEFAULT_RFORK_LEASE_RELEASE_RETRY_INTERVAL_SEC
    send_model_identity_headers: bool = False
    seed_removal_supported: bool | None = False

    def __post_init__(self) -> None:
        for name in ("heartbeat_interval_sec", "lease_release_retry_interval_sec"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or _positive_float(value) is None:
                raise RForkConfigurationError(f"rfork_{name} must be a finite positive JSON number")
        attempts = self.lease_release_max_attempts
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
            raise RForkConfigurationError("rfork_lease_release_max_attempts must be a positive JSON integer")
        port_base = self.seed_port_base
        if isinstance(port_base, bool) or not isinstance(port_base, int) or port_base < 0 or port_base > 65535:
            raise RForkConfigurationError("rfork_seed_port_base must be a JSON integer in range [0, 65535]")
        if not isinstance(self.send_model_identity_headers, bool):
            raise RForkConfigurationError("rfork_model_identity_headers must be a JSON boolean")
        if self.seed_removal_supported is not None and not isinstance(self.seed_removal_supported, bool):
            raise RForkConfigurationError("rfork_seed_removal_supported must be a JSON boolean or null")
        if self.send_model_identity_headers:
            for name in ("model_url", "model_deploy_strategy_name"):
                if not getattr(self, name).isascii():
                    raise RForkConfigurationError(
                        f"{name} must contain only ASCII characters when rfork_model_identity_headers is enabled"
                    )

    @classmethod
    def from_extra_config(cls, raw_config: object) -> "RForkConfig":
        if raw_config is None:
            config: dict[str, Any] = {}
        elif isinstance(raw_config, dict):
            config = raw_config
        else:
            raise RForkConfigurationError("RFork requires --model-loader-extra-config to be a JSON object.")

        if "rfork_model_identity_headers" in config:
            send_model_identity_headers = config["rfork_model_identity_headers"]
        else:
            model_identity_headers = os.getenv("RFORK_MODEL_IDENTITY_HEADERS", "0").strip().lower()
            if model_identity_headers not in {"0", "1", "true", "false"}:
                raise RForkConfigurationError("RFORK_MODEL_IDENTITY_HEADERS must be '0', '1', 'true' or 'false'")
            send_model_identity_headers = model_identity_headers in {"1", "true"}

        return cls(
            send_model_identity_headers=send_model_identity_headers,
            seed_removal_supported=config.get("rfork_seed_removal_supported", False),
            heartbeat_interval_sec=config.get("rfork_heartbeat_interval_sec", DEFAULT_RFORK_HEARTBEAT_INTERVAL_SEC),
            lease_release_max_attempts=config.get(
                "rfork_lease_release_max_attempts", DEFAULT_RFORK_LEASE_RELEASE_MAX_ATTEMPTS
            ),
            lease_release_retry_interval_sec=config.get(
                "rfork_lease_release_retry_interval_sec", DEFAULT_RFORK_LEASE_RELEASE_RETRY_INTERVAL_SEC
            ),
            model_url=_string_value(config, "model_url", "MODEL_URL", "") or "",
            model_deploy_strategy_name=(
                _string_value(
                    config,
                    "model_deploy_strategy_name",
                    "MODEL_DEPLOY_STRATEGY_NAME",
                    "",
                )
                or ""
            ),
            planner_url=(_string_value(config, "rfork_scheduler_url", "RFORK_SCHEDULER_URL", "") or ""),
            seed_timeout_sec=_float_value(
                config,
                "rfork_seed_timeout_sec",
                "RFORK_SEED_TIMEOUT_SEC",
                DEFAULT_RFORK_SEED_TIMEOUT_SEC,
            ),
            request_timeout_sec=_float_value(
                config,
                "rfork_request_timeout_sec",
                "RFORK_REQUEST_TIMEOUT_SEC",
                DEFAULT_RFORK_REQUEST_TIMEOUT_SEC,
            ),
            seed_bind_host=(
                _string_value(
                    config,
                    "rfork_seed_bind_host",
                    "RFORK_SEED_BIND_HOST",
                    "0.0.0.0",
                )
                or "0.0.0.0"
            ),
            seed_port_base=config.get("rfork_seed_port_base", 0),
            seed_advertise_host=_string_value(
                config,
                "rfork_seed_advertise_host",
                "RFORK_SEED_ADVERTISE_HOST",
                None,
            ),
        )
