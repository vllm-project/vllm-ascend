# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Startup-load reference and two-update equivalence checks."""

import json
from contextlib import contextmanager

from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import rlhf_server, wait_for_free_device_memory
from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    FixedRandomWeightSource,
    fixed_startup_checkpoint,
)

UPDATE_ROUNDS = 2


FULL_DECODE_ONLY_CONFIG = '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}'


def _common_serve_args(
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> list[str]:
    """Flags a live-update lane and its reference share.

    Holding them in one place is what makes the two signatures comparable: the
    reference may differ from the lane in exactly one way — it loads the payload
    at startup instead of starting from dummy weights and receiving it live.
    dtype, graph mode, NZ layout, memory budget, max length, tokenizer/executor
    backend and config overrides all stay identical.
    """
    return [
        "--dtype",
        "bfloat16",
        "--compilation-config",
        FULL_DECODE_ONLY_CONFIG,
        # Each NPU IPC trainer rank shares a chip with a rollout worker: the co-located
        # trainer only gets the HBM it needs once the rollout engine sleeps, so
        # the lane releases the engine with a level-2 sleep before every
        # transfer. That requires the CaMem pool, and both builders enable it so
        # the reference stays a plain startup load of the same payload under the
        # very same allocation pool; the reference itself never sleeps.
        "--enable-sleep-mode",
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--port",
        str(port),
        "--trust-remote-code",
        "--additional-config",
        '{"weight_nz_mode": 0}',
        *case.server_args(),
    ]


def live_update_serve_args(
    case: WeightUpdateModelCase,
    *,
    backend: str,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """A lane that starts from dummy weights and receives W over ``backend``."""
    return [
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        json.dumps({"backend": backend}),
        *_common_serve_args(
            case,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
    ]


def reference_serve_args(
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """A server that loads the transfer's payload at normal startup.

    No ``--load-format dummy`` and no transfer backend: this server must be
    correct on its own. It serves under the case's model name and tokenizer so
    both servers answer to the same ``model`` argument in the client calls.
    """
    return [
        "--tokenizer",
        case.model,
        "--served-model-name",
        case.model,
        *_common_serve_args(
            case,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
    ]


# Reuse a startup reference only for the same model, memory, TP, logical device
# and effective child-server environment. Transport and packing do not affect it.
_REFERENCE_SIGNATURES: dict[
    tuple[str, float, int, int, tuple[tuple[str, str], ...]], list[tuple[str, tuple[float, ...]]]
] = {}


def reference_signature(
    source: FixedRandomWeightSource,
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    device_index: int,
    env_dict: dict[str, str] | None = None,
    server_host: str = "127.0.0.1",
) -> list[tuple[str, tuple[float, ...]]]:
    """Signature of a server that loaded the payload at normal startup.

    This is the correctness oracle and it never runs the live-update path, so a
    transfer that silently drops, renames, reshapes or recasts part of the
    payload cannot make the comparison pass. Independent server processes also
    keep the baseline free of whatever the live path does to the model (graph
    capture, layerwise reload state, derived representations).
    """
    cache_key = (
        case.id,
        gpu_memory_utilization,
        tensor_parallel_size,
        device_index,
        tuple(sorted((env_dict or {}).items())),
    )
    cached = _REFERENCE_SIGNATURES.get(cache_key)
    if cached is not None:
        return cached

    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    with (
        fixed_startup_checkpoint(source) as checkpoint_dir,
        rlhf_server(
            checkpoint_dir,
            vllm_serve_args=reference_serve_args(
                case,
                port=port,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            ),
            server_host=server_host,
            server_port=port,
            env_dict=env_dict,
        ) as server,
    ):
        signature = generation_signature(server.get_client(), case.model)

    _REFERENCE_SIGNATURES[cache_key] = signature
    return signature


PROMPTS = [
    "The capital of France is",
    "Explain why the sky is blue in one sentence:",
]


def generation_signature(client, model: str) -> list[tuple[str, tuple[float, ...]]]:
    """Capture deterministic text and logprobs for exact reload comparison."""
    signature = []
    for prompt in PROMPTS:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=8,
            temperature=0,
            logprobs=1,
            seed=0,
        )
        choice = response.choices[0]
        token_logprobs = tuple(choice.logprobs.token_logprobs or ())
        assert token_logprobs, f"{model}: generation returned no token logprobs"
        signature.append((choice.text, token_logprobs))
    return signature


@contextmanager
def weight_transfer_servers(
    source: FixedRandomWeightSource,
    case: WeightUpdateModelCase,
    *,
    backend: str,
    tensor_parallel_size: int,
    device_index: int,
    gpu_memory_utilization: float,
    env_dict: dict[str, str],
):
    """Load W independently, then start the dummy lane under identical settings.

    The reference server must release every TP device before the live-update
    server starts. Both use the case's environment and the same generated W.
    """
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        device_index=device_index,
        env_dict=env_dict,
    )
    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    port = get_open_port()
    with rlhf_server(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend=backend,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
    ) as server:
        dummy_signature = generation_signature(server.get_client(), case.model)
        yield server, reference, dummy_signature


def assert_weight_update_matches_reference(
    dummy_signature, reference_signature, updated_signatures, case: WeightUpdateModelCase
) -> None:
    """Compare every live update against independent startup loading.

    Dummy output must differ to prove the payload is observable. The first
    update proves that transfer takes effect; repeated reload checks that
    layerwise reload and graph-captured storage preserve the same behavior.
    Comparing updates only to each other could hide consistently wrong loads.
    """
    assert dummy_signature != reference_signature, (
        f"{case.id}: a dummy-loaded model already matches a normally loaded one; "
        "the reference cannot observe the payload"
    )
    assert len(updated_signatures) == UPDATE_ROUNDS, (
        f"{case.id}: expected {UPDATE_ROUNDS} updates, got {len(updated_signatures)}"
    )
    for update_round, signature in enumerate(updated_signatures, start=1):
        phase = "first live weight update" if update_round == 1 else "repeated live reload"
        assert signature == reference_signature, (
            f"{case.id}: {phase} (round {update_round}) does not match normal startup loading of the same weights"
        )
