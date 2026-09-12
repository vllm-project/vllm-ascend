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


def _string_value(
    config: dict[str, Any],
    keys: tuple[str, ...],
    env_name: str,
    default: str | None = "",
) -> str | None:
    value = next((config[key] for key in keys if key in config), None)
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
    keys: tuple[str, ...],
    env_name: str,
    default: float,
) -> float:
    for key in keys:
        if key in config:
            parsed = _positive_float(config[key])
            if parsed is not None:
                return parsed
            break
    return _positive_float(os.getenv(env_name)) or default


@dataclass(frozen=True, slots=True)
class RForkConfig:
    model_url: str
    model_deploy_strategy_name: str
    planner_url: str
    seed_timeout_sec: float = DEFAULT_RFORK_SEED_TIMEOUT_SEC
    request_timeout_sec: float = DEFAULT_RFORK_REQUEST_TIMEOUT_SEC
    seed_bind_host: str = "0.0.0.0"
    seed_advertise_host: str | None = None
    heartbeat_interval_sec: float = DEFAULT_RFORK_HEARTBEAT_INTERVAL_SEC
    lease_release_max_attempts: int = DEFAULT_RFORK_LEASE_RELEASE_MAX_ATTEMPTS
    lease_release_retry_interval_sec: float = DEFAULT_RFORK_LEASE_RELEASE_RETRY_INTERVAL_SEC

    def __post_init__(self) -> None:
        # These operational settings are JSON-only: reject typos instead of silently
        # falling back to defaults or reading similarly named environment variables.
        for name in ("heartbeat_interval_sec", "lease_release_retry_interval_sec"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or _positive_float(value) is None:
                raise ValueError(f"rfork_{name} must be a finite positive JSON number")
        attempts = self.lease_release_max_attempts
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
            raise ValueError("rfork_lease_release_max_attempts must be a positive JSON integer")

    @classmethod
    def from_extra_config(cls, raw_config: object) -> "RForkConfig":
        if raw_config is None:
            config: dict[str, Any] = {}
        elif isinstance(raw_config, dict):
            config = raw_config
        else:
            raise RuntimeError("RFork requires --model-loader-extra-config to be a JSON object.")

        return cls(
            heartbeat_interval_sec=config.get("rfork_heartbeat_interval_sec", DEFAULT_RFORK_HEARTBEAT_INTERVAL_SEC),
            lease_release_max_attempts=config.get(
                "rfork_lease_release_max_attempts", DEFAULT_RFORK_LEASE_RELEASE_MAX_ATTEMPTS
            ),
            lease_release_retry_interval_sec=config.get(
                "rfork_lease_release_retry_interval_sec", DEFAULT_RFORK_LEASE_RELEASE_RETRY_INTERVAL_SEC
            ),
            model_url=_string_value(config, ("model_url",), "MODEL_URL", "") or "",
            model_deploy_strategy_name=(
                _string_value(
                    config,
                    ("model_deploy_strategy_name",),
                    "MODEL_DEPLOY_STRATEGY_NAME",
                    "",
                )
                or ""
            ),
            planner_url=(_string_value(config, ("rfork_scheduler_url",), "RFORK_SCHEDULER_URL", "") or ""),
            seed_timeout_sec=_float_value(
                config,
                ("rfork_seed_timeout_sec",),
                "RFORK_SEED_TIMEOUT_SEC",
                DEFAULT_RFORK_SEED_TIMEOUT_SEC,
            ),
            request_timeout_sec=_float_value(
                config,
                ("rfork_request_timeout_sec", "request_timeout_sec"),
                "RFORK_REQUEST_TIMEOUT_SEC",
                DEFAULT_RFORK_REQUEST_TIMEOUT_SEC,
            ),
            seed_bind_host=(
                _string_value(
                    config,
                    ("rfork_seed_bind_host", "seed_bind_host", "bind_host"),
                    "RFORK_SEED_BIND_HOST",
                    "0.0.0.0",
                )
                or "0.0.0.0"
            ),
            seed_advertise_host=_string_value(
                config,
                ("rfork_seed_advertise_host", "seed_advertise_host", "advertise_host"),
                "RFORK_SEED_ADVERTISE_HOST",
                None,
            ),
        )
