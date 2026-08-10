# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense and sparse HCCL weight updates through vLLM's HTTP API.

Both phases start a fresh real-checkpoint vLLM server and apply the same
deterministic trainer patch. Dense HCCL sends all parameters; sparse HCCL sends
only flat ``indices`` and ``values``. Two NPUs are required (server 0, trainer 1).

Current sparse MVP limitations: TP=1, PP=1, runtime-format parameter names,
and no checkpoint-format or packed sparse updates.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.utils.network_utils import get_ip, get_open_port

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerSendWeightsArgs,
    HCCLWeightTransferEngine,
)
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLWeightTransferEngine,
    SparseWeightPatch,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
PATCHED_PARAM_NAME = "model.embed_tokens.weight"
MAX_PATCH_ROWS = 32
MAX_TOKENS = 100
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@dataclass
class TrainerState:
    model: torch.nn.Module
    tokenizer: Any
    patched_param: torch.Tensor
    patches: list[SparseWeightPatch] | None = None


def load_trainer(model_name: str, device: str) -> TrainerState:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    try:
        parameter = model.get_parameter(PATCHED_PARAM_NAME)
    except AttributeError as exc:
        raise RuntimeError(f"Missing runtime parameter {PATCHED_PARAM_NAME}") from exc
    return TrainerState(model, tokenizer, parameter)


def select_patch_rows(
    tokenizer: Any,
    parameter: torch.Tensor,
    prompts: Sequence[str] = PROMPTS,
    max_rows: int = MAX_PATCH_ROWS,
) -> torch.Tensor:
    selected: list[int] = []
    special_ids = set(tokenizer.all_special_ids)
    for prompt in prompts:
        for token_id in tokenizer(prompt, add_special_tokens=False)["input_ids"]:
            if token_id not in special_ids and token_id not in selected:
                selected.append(token_id)
            if len(selected) == max_rows:
                break
        if len(selected) == max_rows:
            break
    if not selected:
        raise ValueError("Could not derive any non-special token IDs to patch")
    token_id = selected[-1]
    while len(selected) < max_rows:
        token_id = (token_id + 1) % parameter.shape[0]
        if token_id not in special_ids and token_id not in selected:
            selected.append(token_id)
    return torch.tensor(selected, device=parameter.device, dtype=torch.long)


def prepare_sparse_patch(
    state: TrainerState,
    prompts: Sequence[str] = PROMPTS,
    max_rows: int = MAX_PATCH_ROWS,
) -> tuple[dict[str, object], list[int], str, int]:
    rows = select_patch_rows(state.tokenizer, state.patched_param, prompts, max_rows)
    width = state.patched_param.shape[1]
    columns = torch.arange(width, device=state.patched_param.device)
    with torch.no_grad():
        state.patched_param[rows] = state.patched_param[rows].roll(1, dims=0)
    indices = (rows[:, None] * width + columns).reshape(-1).to(torch.int32)
    values = state.patched_param[rows].reshape(-1).contiguous()
    patch = SparseWeightPatch(PATCHED_PARAM_NAME, indices, values)
    state.patches = [patch]
    digest = hashlib.sha256(
        indices.cpu().numpy().tobytes()
        + values.detach().float().cpu().numpy().tobytes()
    ).hexdigest()
    payload = indices.numel() * indices.element_size()
    payload += values.numel() * values.element_size()
    info = {
        "names": [PATCHED_PARAM_NAME],
        "dtype_names": [str(values.dtype).split(".")[-1]],
        "shapes": [list(state.patched_param.shape)],
        "num_updates_list": [indices.numel()],
    }
    return info, rows.tolist(), digest, payload


def dense_update_info(model: torch.nn.Module) -> tuple[dict[str, object], int]:
    names, dtypes, shapes, payload = [], [], [], 0
    for name, parameter in model.named_parameters():
        names.append(name)
        dtypes.append(str(parameter.dtype).split(".")[-1])
        shapes.append(list(parameter.shape))
        payload += parameter.numel() * parameter.element_size()
    return {
        "names": names,
        "dtype_names": dtypes,
        "shapes": shapes,
        "packed": False,
    }, payload


def post(base_url: str, endpoint: str, payload: dict | None = None) -> None:
    response = requests.post(
        f"{base_url}/{endpoint}", json=payload, timeout=300
    )
    response.raise_for_status()


def generate(
    base_url: str,
    model: str,
    tokenizer: Any,
) -> list[dict[str, object]]:
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": PROMPTS,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    choices = sorted(response.json()["choices"], key=lambda item: item["index"])
    results = []
    for choice in choices:
        text = choice["text"]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        results.append({"token_ids": token_ids, "text": text})
    return results


def build_server_command(args: argparse.Namespace, backend: str) -> list[str]:
    executable = shutil.which("vllm")
    if executable is None:
        raise RuntimeError("`vllm` executable was not found in PATH")
    return [
        executable,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--weight-transfer-config",
        json.dumps({"backend": backend}, separators=(",", ":")),
    ]


def wait_for_server(url: str, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with status {process.returncode}")
        try:
            if requests.get(f"{url}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM server did not become ready within {timeout}s")


def wait_for_port_release(host: str, port: int, timeout: int = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex((host, port)) != 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"Port {host}:{port} was not released")


@contextmanager
def running_server(args: argparse.Namespace, backend: str):
    env = os.environ.copy()
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=str(args.server_device),
        VLLM_ASCEND_ENABLE_NZ="0",
        VLLM_SERVER_DEV_MODE="1",
    )
    print(f"Starting vLLM server with backend={backend}")
    process = subprocess.Popen(build_server_command(args, backend), env=env)
    url = f"http://{args.host}:{args.port}"
    try:
        wait_for_server(url, process, args.server_start_timeout)
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        wait_for_port_release(args.host, args.port)


def initialize_transfer(base_url: str) -> Any:
    response = requests.get(f"{base_url}/get_world_size", timeout=10)
    response.raise_for_status()
    world_size = response.json()["world_size"] + 1
    address, port, errors = get_ip(), get_open_port(), []

    def server_init() -> None:
        try:
            post(base_url, "init_weight_transfer_engine", {"init_info": {
                "master_address": address,
                "master_port": port,
                "rank_offset": 1,
                "world_size": world_size,
            }})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=server_init)
    thread.start()
    group = HCCLWeightTransferEngine.trainer_init(
        {"master_address": address, "master_port": port, "world_size": world_size}
    )
    thread.join()
    if errors:
        raise errors[0]
    return group


def send_update(
    base_url: str, info: dict[str, object], sender: Callable[[], None]
) -> float:
    errors = []

    def receiver() -> None:
        try:
            post(base_url, "update_weights", {"update_info": info})
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=receiver)
    thread.start()
    start = time.perf_counter()
    sender()
    torch.accelerator.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    thread.join()
    if errors:
        raise errors[0]
    return elapsed


def run_phase(args: argparse.Namespace, backend: str) -> dict[str, object]:
    device = f"npu:{args.trainer_device}"
    torch.accelerator.set_device_index(device)
    state = load_trainer(args.model, device)
    with running_server(args, backend) as base_url:
        before = generate(base_url, args.model, state.tokenizer)
        group = initialize_transfer(base_url)
        post(base_url, "pause")
        post(base_url, "start_weight_update")
        if backend == "nccl":
            info, payload = dense_update_info(state.model)
            _, selected, digest, _ = prepare_sparse_patch(state)

            def sender() -> None:
                HCCLWeightTransferEngine.trainer_send_weights(
                    state.model.named_parameters(), HCCLTrainerSendWeightsArgs(group)
                )
        else:
            info, selected, digest, payload = prepare_sparse_patch(state)

            def sender() -> None:
                SparseHCCLWeightTransferEngine.trainer_send_weights(
                    iter(state.patches or []), HCCLTrainerSendWeightsArgs(group)
                )
        send_ms = send_update(base_url, info, sender)
        state.patches = None
        post(base_url, "finish_weight_update")
        post(base_url, "resume")
        after = generate(base_url, args.model, state.tokenizer)
    return {
        "before": before,
        "after": after,
        "selected_token_ids": selected,
        "patch_digest": digest,
        "payload_bytes": payload,
        "send_ms": send_ms,
    }


def token_sequences_match(left: Sequence[dict], right: Sequence[dict]) -> bool:
    return [item["token_ids"] for item in left] == [item["token_ids"] for item in right]


def validate_results(dense: dict, sparse: dict) -> dict[str, bool]:
    checks = {
        "baseline_equal": token_sequences_match(dense["before"], sparse["before"]),
        "patch_selection_equal": (
            dense["selected_token_ids"] == sparse["selected_token_ids"]
        ),
        "patch_digest_equal": dense["patch_digest"] == sparse["patch_digest"],
        "after_equal": token_sequences_match(dense["after"], sparse["after"]),
        "any_output_changed": any(
            x["token_ids"] != y["token_ids"]
            for x, y in zip(dense["before"], dense["after"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Sparse HCCL validation failed: " + ", ".join(failed))
    return checks


def print_generations(label: str, generations: Sequence[dict]) -> None:
    print(f"\n{label}")
    print("-" * 50)
    for prompt, generation in zip(PROMPTS, generations):
        print(f"Prompt: {prompt!r}")
        print(f"Token IDs: {generation['token_ids']}")
        print(f"Text: {generation['text']!r}")
        print("-" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-device", type=int, default=0)
    parser.add_argument("--trainer-device", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--server-start-timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.server_device == args.trainer_device:
        raise ValueError("Server and trainer must use different physical NPUs")
    dense = run_phase(args, "nccl")
    sparse = run_phase(args, "sparse_nccl")
    checks = validate_results(dense, sparse)
    print_generations("Dense baseline outputs", dense["before"])
    print_generations("Sparse baseline outputs", sparse["before"])
    print_generations("Dense outputs after update", dense["after"])
    print_generations("Sparse outputs after update", sparse["after"])
    print(f"patched_token_ids = {dense['selected_token_ids']}")
    print(f"dense_patch_digest = {dense['patch_digest']}")
    print(f"sparse_patch_digest = {sparse['patch_digest']}")
    for name, passed in checks.items():
        print(f"{name} = {passed}")
    print(f"dense_payload_mb = {dense['payload_bytes'] / 2**20:.2f}")
    print(f"sparse_payload_mb = {sparse['payload_bytes'] / 2**20:.2f}")
    print(f"dense_send_ms = {dense['send_ms']:.2f}")
    print(f"sparse_send_ms = {sparse['send_ms']:.2f}")


if __name__ == "__main__":
    main()
