# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from collections.abc import Mapping
from http import HTTPStatus
from urllib.parse import urlsplit, urlunsplit

import requests
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import RFORK_PROTOCOL_VERSION, SeedTransferInfo


def _join_host_port(host: str, port: int) -> str:
    """Format a host and an independently supplied port for an HTTP URL."""
    host = host.strip("[]")
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def build_seed_url(host: object, port: int) -> str:
    seed_host = str(host).strip()
    if isinstance(port, bool) or not isinstance(port, int) or port <= 0 or port > 65535:
        raise ValueError(f"RFork seed port must be in 1..65535, got {port!r}")

    if seed_host.startswith(("http://", "https://")):
        try:
            parsed = urlsplit(seed_host)
            explicit_port = parsed.port
        except ValueError as exc:
            raise ValueError(f"RFork seed URL is malformed: {seed_host!r}") from exc
        if parsed.hostname is None:
            raise ValueError(f"RFork seed URL has no host: {seed_host!r}")
        if explicit_port is not None and explicit_port != port:
            raise ValueError(f"RFork seed URL port conflicts with lease port: url={explicit_port}, lease={port}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("RFork seed URL must not include user credentials")
        netloc = _join_host_port(parsed.hostname, explicit_port or port)
        return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), parsed.query, parsed.fragment)).rstrip("/")

    if seed_host.startswith("[") and "]" in seed_host:
        closing = seed_host.find("]")
        host_part = seed_host[1:closing]
        suffix = seed_host[closing + 1 :]
        if suffix:
            if not suffix.startswith(":") or not suffix[1:].isdigit():
                raise ValueError(f"RFork seed host has an invalid port: {seed_host!r}")
            explicit_port = int(suffix[1:])
            if explicit_port != port:
                raise ValueError(f"RFork seed host port conflicts with lease port: host={explicit_port}, lease={port}")
        seed_host = host_part
    return f"http://{_join_host_port(seed_host, port)}"


def fetch_seed_transfer_info(
    seed_url: str,
    seed_key: str,
    request_timeout_sec: float,
) -> SeedTransferInfo | None:
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": seed_key},
            timeout=request_timeout_sec,
        )
        if response.status_code != HTTPStatus.OK:
            logger.error("RFork seed metadata request returned status=%s", response.status_code)
            return None
        response_payload = response.json()
        if not isinstance(response_payload, Mapping):
            logger.error("RFork seed returned malformed transfer metadata response.")
            return None
        protocol_version = response_payload.get("rfork_protocol_version")
        if (
            not isinstance(protocol_version, int)
            or isinstance(protocol_version, bool)
            or protocol_version != RFORK_PROTOCOL_VERSION
        ):
            logger.error(
                "RFork seed protocol mismatch: expected=%s got=%r",
                RFORK_PROTOCOL_VERSION,
                protocol_version,
            )
            return None
        info = response_payload.get("rfork_transfer_engine_info")
        if not isinstance(info, (list, tuple)) or len(info) != 2:
            logger.error("RFork seed returned malformed transfer metadata.")
            return None
        session_id, weights = info
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, Mapping):
            logger.error("RFork seed returned an invalid session or weight manifest.")
            return None

        formats_candidate = response_payload.get("rfork_transfer_engine_format_info")
        if not isinstance(formats_candidate, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, int) or isinstance(value, bool)
            for name, value in formats_candidate.items()
        ):
            logger.error("RFork seed returned malformed format metadata.")
            return None
        formats = dict(formats_candidate)

        return SeedTransferInfo(session_id=session_id, weights=weights, formats=formats)
    except Exception as exc:
        logger.error("RFork seed metadata request failed for %s: %s", seed_url, exc)
        return None
