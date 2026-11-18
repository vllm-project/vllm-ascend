# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed ``--model-loader-extra-config`` for the bigtensorloader loader.

Same channel and validation philosophy as the D2D loaders (netloader,
rfork): the JSON object passed via ``--model-loader-extra-config`` is
parsed once into a frozen dataclass, unknown keys are reported so typos
do not pass silently, and invalid values abort startup through a
dedicated error type instead of degrading to a configuration the
operator did not ask for.
"""

from dataclasses import dataclass
from typing import Any

DEFAULT_VERIFY_MODE = "size"

VERIFY_MODES = ("size", "sha256", "none")

# Keys read by BigTensorConfig.from_extra_config. Keep in sync with that method.
KNOWN_EXTRA_CONFIG_KEYS = frozenset(
    {
        "fast_restore",
        "checkpoint_path",
        "verify",
        "bulk_h2d",
        "bulk_chunk_mb",
    }
)


class BigTensorConfigurationError(Exception):
    """BigTensorLoader configuration is missing or invalid.

    Deliberately not a ``ValueError``/``RuntimeError`` subclass: the
    snapshot loader treats those as recoverable (restore failures fall
    back to a fresh load), while a configuration error must abort
    startup so the operator sees it.
    """


def unknown_extra_config_keys(raw_config: object) -> list[str]:
    """Return extra-config keys this loader version does not read, such as typos."""
    if not isinstance(raw_config, dict):
        return []
    return sorted(str(key) for key in raw_config if key not in KNOWN_EXTRA_CONFIG_KEYS)


def _bool_value(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    raise BigTensorConfigurationError(f"{key} must be a JSON boolean (true/false), got {value!r}")


def _positive_int_or_none(config: dict[str, Any], key: str) -> int | None:
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BigTensorConfigurationError(f"{key} must be a positive JSON integer, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class BigTensorConfig:
    """Snapshot fast-restore knobs, all opt-in with safe defaults."""

    fast_restore: bool = False
    checkpoint_path: str | None = None
    verify: str = DEFAULT_VERIFY_MODE
    bulk_h2d: bool = False
    bulk_chunk_mb: int | None = None  # None: dynamic chunk sizing from free NPU memory

    def __post_init__(self) -> None:
        if self.verify not in VERIFY_MODES:
            raise BigTensorConfigurationError(f"verify must be one of {VERIFY_MODES}, got {self.verify!r}")
        if self.fast_restore and not self.checkpoint_path:
            raise BigTensorConfigurationError(
                "fast_restore=true requires checkpoint_path, a writable directory where "
                "the per-rank snapshots ({rank}.json + {rank}.snapshot) are stored"
            )

    @classmethod
    def from_extra_config(cls, raw_config: object) -> "BigTensorConfig":
        if raw_config is None:
            config: dict[str, Any] = {}
        elif isinstance(raw_config, dict):
            config = raw_config
        else:
            raise BigTensorConfigurationError(
                "bigtensorloader requires --model-loader-extra-config to be a JSON object."
            )

        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path is not None and not isinstance(checkpoint_path, str):
            raise BigTensorConfigurationError(f"checkpoint_path must be a JSON string, got {checkpoint_path!r}")

        return cls(
            fast_restore=_bool_value(config, "fast_restore", False),
            checkpoint_path=checkpoint_path,
            verify=config.get("verify", DEFAULT_VERIFY_MODE),
            bulk_h2d=_bool_value(config, "bulk_h2d", False),
            bulk_chunk_mb=_positive_int_or_none(config, "bulk_chunk_mb"),
        )
