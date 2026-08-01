# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP helpers for weight transfer examples."""

from typing import Any

import requests


def weight_transfer_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def post_weight_transfer_endpoint(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> None:
    url = weight_transfer_url(base_url, endpoint)
    response = requests.post(url, json=payload, timeout=timeout) if payload is not None else requests.post(url, timeout=timeout)
    response.raise_for_status()


def pause_generation(base_url: str) -> None:
    post_weight_transfer_endpoint(base_url, "pause")


def resume_generation(base_url: str) -> None:
    post_weight_transfer_endpoint(base_url, "resume")


def init_weight_transfer_engine(base_url: str, init_info: dict[str, Any]) -> None:
    post_weight_transfer_endpoint(base_url, "init_weight_transfer_engine", {"init_info": init_info})


def start_weight_update(base_url: str, is_checkpoint_format: bool = True) -> None:
    post_weight_transfer_endpoint(
        base_url,
        "start_weight_update",
        {"is_checkpoint_format": is_checkpoint_format},
    )


def update_weights(base_url: str, update_info: dict[str, Any]) -> None:
    post_weight_transfer_endpoint(base_url, "update_weights", {"update_info": update_info}, timeout=300)


def finish_weight_update(base_url: str) -> None:
    post_weight_transfer_endpoint(base_url, "finish_weight_update")


def get_world_size(base_url: str) -> int:
    response = requests.get(weight_transfer_url(base_url, "get_world_size"), timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]
