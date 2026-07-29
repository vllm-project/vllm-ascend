# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for weight transfer e2e tests."""

import os
import threading
from typing import Any

import requests
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tests.e2e.conftest import RemoteOpenAIServer

CONTROL_TIMEOUT = 60
UPDATE_TIMEOUT = 300
INIT_TIMEOUT = 120


def log(message: str) -> None:
    """Flushed log so step markers show up immediately even when stdout is piped."""
    print(f"[trainer] {message}", flush=True)


def build_trainer_model(model_name: str, device_index: int, *, eval_mode: bool = False):
    """Build the trainer-side model for weight transfer e2e tests.

    By default the model is instantiated from the architecture config with random
    weights (no ``model.safetensors`` download required). Set
    ``WEIGHT_TRANSFER_TEST_MODEL=/path/to/checkpoint`` to broadcast real weights
    from a local directory instead.
    """
    device = f"npu:{device_index}"
    override_path = os.getenv("WEIGHT_TRANSFER_TEST_MODEL")
    if override_path:
        log(f"loading real trainer weights from {override_path}")
        model = AutoModelForCausalLM.from_pretrained(override_path, dtype=torch.bfloat16)
    else:
        log("building trainer model from config with random weights (download-free)")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
    model = model.to(device=device, dtype=torch.bfloat16)
    if eval_mode:
        model.eval()
    return model


def post(server: RemoteOpenAIServer, route: str, *, json: dict[str, Any] | None = None, timeout: int = CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


class BackgroundPost(threading.Thread):
    """Run an HTTP POST in a thread while keeping its exception visible.

    Some trainer-side data transfers block until the matching server-side RPC is
    running. If that RPC fails, swallowing the exception would leave the trainer
    waiting forever; instead we record it and surface it on join().
    """

    def __init__(
        self,
        server: RemoteOpenAIServer,
        route: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: int = CONTROL_TIMEOUT,
    ) -> None:
        super().__init__(daemon=True)
        self._server = server
        self._route = route
        self._json = json
        self._timeout = timeout
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            post(self._server, self._route, json=self._json, timeout=self._timeout)
            log(f"background POST /{self._route} done")
        except BaseException as exc:  # noqa: BLE001 - re-raised on join via raise_if_failed
            self.error = exc
            log(f"background POST /{self._route} FAILED: {exc!r}")

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"server-side /{self._route} failed") from self.error


def generate(client, model: str, prompts: list[str]) -> list[str]:
    completions = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=16,
            temperature=0,
        )
        completions.append(response.choices[0].text)
    return completions


def collect_weight_metadata(train_model) -> tuple[list[str], list[str], list[list[int]], int]:
    """Collect parameter metadata and size the packed buffer for broadcasting."""
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    max_tensor_bytes = 0
    for name, parameter in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(parameter.dtype).split(".")[-1])
        shapes.append(list(parameter.shape))
        tensor_bytes = parameter.numel() * parameter.element_size()
        max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)

    # Keep the 1 GiB default unless a single tensor needs more (+128 MiB headroom).
    packed_buffer_size_bytes = max(max_tensor_bytes + 128 * 2**20, 2**30)
    return names, dtype_names, shapes, packed_buffer_size_bytes


def has_lifecycle_endpoints(server: RemoteOpenAIServer) -> bool:
    """Probe ``/start_weight_update``; also performs the actual call when present."""
    try:
        response = requests.post(
            server.url_for("start_weight_update"),
            json={"is_checkpoint_format": True},
            timeout=CONTROL_TIMEOUT,
        )
    except requests.RequestException:
        return False
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True
