# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import requests
from vllm.logger import logger
from vllm.utils.network_utils import join_host_port

from vllm_ascend.model_loader.rfork.types import SeedTransferInfo


def build_seed_url(host: object, port: int) -> str:
    seed_host = str(host).strip()
    if seed_host.startswith(("http://", "https://")):
        return seed_host.rstrip("/")
    if seed_host.startswith("[") and "]" in seed_host:
        closing = seed_host.find("]")
        if closing == len(seed_host) - 1:
            seed_host = seed_host[1:-1]
        elif seed_host[closing + 1 :] == f":{port}":
            return f"http://{seed_host}"
        else:
            seed_host = seed_host[1:closing]
    return f"http://{join_host_port(seed_host, port)}"


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
        info = response_payload.get("rfork_transfer_engine_info")
        if not isinstance(info, (list, tuple)) or len(info) != 2:
            logger.error("RFork seed returned malformed transfer metadata.")
            return None
        session_id, weights = info
        if not isinstance(session_id, str) or not session_id or not isinstance(weights, Mapping):
            logger.error("RFork seed returned an invalid session or weight manifest.")
            return None

        shapes: Mapping[str, Any] | None = None
        if "rfork_transfer_engine_shape_info" in response_payload:
            candidate = response_payload.get("rfork_transfer_engine_shape_info")
            if candidate is not None and not isinstance(candidate, Mapping):
                logger.error("RFork seed returned a malformed shape manifest.")
                return None
            shapes = candidate
        else:
            shape_response = requests.get(
                f"{seed_url}/get_rfork_transfer_engine_shape_info",
                params={"seed_key": seed_key},
                timeout=request_timeout_sec,
            )
            if shape_response.status_code == HTTPStatus.OK:
                shape_payload = shape_response.json()
                if not isinstance(shape_payload, Mapping):
                    logger.error("RFork seed returned malformed shape metadata response.")
                    return None
                candidate = shape_payload.get("rfork_transfer_engine_shape_info")
                if candidate is not None and not isinstance(candidate, Mapping):
                    logger.error("RFork seed returned a malformed shape manifest.")
                    return None
                shapes = candidate
        return SeedTransferInfo(session_id=session_id, weights=weights, shapes=shapes)
    except Exception as exc:
        logger.error("RFork seed metadata request failed for %s: %s", seed_url, exc)
        return None
