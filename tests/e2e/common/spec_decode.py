"""Shared helpers for speculative-decoding acceptance sampling in nightly harnesses.

Single-node, internal-DP and external-DP harnesses all need to parse the
speculative configuration from server commands, pick the accuracy benchmark
that owns the acceptance sampling window, and prime the draft counters with a
warmup request. This module keeps those mechanics in one place; deployment
specifics (which command is the decode side, which endpoints expose metrics)
stay in each harness.
"""

import json
import logging
import shlex
from collections.abc import Callable
from typing import Any

import requests

logger = logging.getLogger(__name__)

SPECULATIVE_CONFIG_FLAGS = ("--speculative-config", "--speculative_config")


def split_server_cmd(server_cmd: list[str] | str) -> list[str]:
    """Normalize a server command given as a list or a shell string."""
    if isinstance(server_cmd, str):
        try:
            return shlex.split(server_cmd)
        except ValueError:
            return server_cmd.split()
    return list(server_cmd)


def extract_flag_value(server_cmd: list[str] | str, flag: str) -> str | None:
    """Return the value following `flag` in a server command, or None."""
    cmd_list = split_server_cmd(server_cmd)
    try:
        idx = cmd_list.index(flag)
        return cmd_list[idx + 1]
    except (ValueError, IndexError):
        return None


def parse_json_flag(server_cmd: list[str] | str, flag: str) -> dict[str, Any]:
    """Extract and JSON-parse the value following `flag` in a server command."""
    val = extract_flag_value(server_cmd, flag)
    if not val:
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return {}


def get_speculative_config(server_cmd: list[str] | str) -> dict[str, Any]:
    """Parse the speculative config using either supported flag spelling."""
    for flag in SPECULATIVE_CONFIG_FLAGS:
        spec_config = parse_json_flag(server_cmd, flag)
        if spec_config:
            return spec_config
    return {}


def get_speculative_num_tokens(server_cmd: list[str] | str, default: int = 1) -> int:
    """Return the draft position count derived from the (decode-side) command."""
    spec_config = get_speculative_config(server_cmd)
    return int(spec_config.get("num_speculative_tokens", default))


def select_acceptance_case(
    cases: list[tuple[str, dict[str, Any]]],
    requested: str | None,
) -> int | None:
    """Return the index of the accuracy benchmark that owns the acceptance window.

    ``cases`` is a list of ``(name, case_config)`` pairs in execution order. An
    explicit ``requested`` name must reference an accuracy case. Without one,
    the sole accuracy case is auto-selected only when no performance case is
    configured; otherwise None is returned and the caller keeps the legacy
    whole-session sampling window.
    """
    if requested:
        for index, (name, case_config) in enumerate(cases):
            if name == requested:
                if case_config.get("case_type") != "accuracy":
                    raise ValueError(f"acceptance_rate.benchmark {requested!r} must reference an accuracy benchmark")
                return index
        names = [name for name, _ in cases]
        raise ValueError(f"acceptance_rate.benchmark {requested!r} is not configured in benchmarks: {names}")

    accuracy_indices = [
        index for index, (_, case_config) in enumerate(cases) if case_config.get("case_type") == "accuracy"
    ]
    performance_indices = [
        index for index, (_, case_config) in enumerate(cases) if case_config.get("case_type") == "performance"
    ]
    if len(accuracy_indices) == 1 and not performance_indices:
        return accuracy_indices[0]
    return None


def make_chat_warmup_fn(chat_url: str, model: str) -> Callable[[], None]:
    """Build a warmup request that initializes spec-decode counters before sampling."""

    def warmup_fn() -> None:
        response = requests.post(
            chat_url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 16,
            },
            timeout=120,
        )
        if not response.ok:
            logger.warning(
                "Spec-decode warmup request returned HTTP %s; the acceptance sampling window may not be primed",
                response.status_code,
            )

    return warmup_fn
