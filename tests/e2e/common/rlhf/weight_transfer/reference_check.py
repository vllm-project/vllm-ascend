# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent cold-load references and repeated-update equivalence checks."""

import json
from contextlib import ExitStack, contextmanager

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
    execution_mode: str = "graph",
    enable_sleep: bool = True,
) -> list[str]:
    """Flags a live-update lane and its reference share.

    Holding them in one place is what makes the two signatures comparable: the
    reference loads its payload at startup instead of receiving a live update.
    The live lane may start from dummy weights or a generated checkpoint A.
    dtype, graph mode, NZ layout, memory budget, max length, tokenizer/executor
    backend and config overrides all stay identical.
    """
    if execution_mode not in ("eager", "graph"):
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    execution_args = (
        ["--enforce-eager"] if execution_mode == "eager" else ["--compilation-config", FULL_DECODE_ONLY_CONFIG]
    )
    return [
        "--dtype",
        "bfloat16",
        *execution_args,
        # Match allocation-pool settings in the live and cold-reference servers.
        # The reference never sleeps; no-sleep Direct lanes do not enable the pool.
        *(["--enable-sleep-mode"] if enable_sleep else []),
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
    execution_mode: str = "graph",
    enable_sleep: bool = True,
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
            execution_mode=execution_mode,
            enable_sleep=enable_sleep,
        ),
    ]


def reference_serve_args(
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
    execution_mode: str = "graph",
    enable_sleep: bool = True,
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
            execution_mode=execution_mode,
            enable_sleep=enable_sleep,
        ),
    ]


# Reuse a startup reference only for the same model, memory, TP, logical device
# and effective child-server environment. Transport and packing do not affect it.
_REFERENCE_SIGNATURES: dict[tuple, list[tuple[str, tuple[float, ...]]]] = {}


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
    execution_mode: str = "graph",
    enable_sleep: bool = True,
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
        source.seed,
        gpu_memory_utilization,
        tensor_parallel_size,
        device_index,
        execution_mode,
        enable_sleep,
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
                execution_mode=execution_mode,
                enable_sleep=enable_sleep,
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
    execution_mode: str = "graph",
    enable_sleep: bool = True,
    cold_start: bool = False,
):
    """Load A independently, then start the live lane under identical settings.

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
        execution_mode=execution_mode,
        enable_sleep=enable_sleep,
    )
    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    port = get_open_port()
    with ExitStack() as stack:
        options = dict(
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            execution_mode=execution_mode,
            enable_sleep=enable_sleep,
        )
        if cold_start:
            model_path = stack.enter_context(fixed_startup_checkpoint(source))
            args = reference_serve_args(case, **options)
            args += ["--weight-transfer-config", json.dumps({"backend": backend})]
            args += [
                "--worker-extension-cls",
                "tests.e2e.common.rlhf.weight_transfer.reload_probe.ReloadProbeWorkerExtension",
            ]
        else:
            model_path = case.model
            args = live_update_serve_args(case, backend=backend, **options)
        server = stack.enter_context(
            rlhf_server(
                model_path,
                vllm_serve_args=args,
                server_host="127.0.0.1",
                server_port=port,
                env_dict=env_dict,
            )
        )
        initial_signature = generation_signature(server.get_client(), case.model)
        if cold_start:
            assert initial_signature == reference, f"{case.id}: cold A differs from the independent reference"
        yield server, reference, initial_signature


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
