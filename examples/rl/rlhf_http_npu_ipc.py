# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with NPU IPC-based weight syncing APIs.

Unlike rlhf_http_hccl.py which uses HCCL and can use separate NPUs, this script
uses Ascend NPU IPC which requires the training model and vLLM server to be on
the same physical NPU. Memory must be carefully managed to fit both models.

Prerequisites:
    Start a vLLM server with weight transfer enabled and reduced NPU memory
    utilization to leave room for the training model:

    $ VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        vllm serve Qwen/Qwen3-0.6b --enforce-eager \
        --weight-transfer-config '{"backend": "ipc"}' \
        --load-format dummy \
        --gpu-memory-utilization 0.5

    Then run this script:

    $ python rlhf_http_npu_ipc.py

The example performs the following steps:

* Load the training model on NPU 0 (same NPU as the vLLM server).
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Initialize weight transfer via HTTP endpoint (no-op for NPU IPC).
* Pause generation and broadcast the real weights from the training model to
  the vLLM server using NPU IPC handles (via HTTP). The pause/resume is
  handled by ``trainer_send_weights`` — it calls ``update_weights`` internally.
* Generate text again to show normal output after the weight update.
"""

import os

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerSendWeightsArgs,
    NPUIPCWeightTransferEngine,
)

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Enable insecure serialization for IPC handle serialization over HTTP
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


def generate_completions(client: OpenAI, model: str, prompts: list[str]) -> list[str]:
    """Generate completions using the OpenAI-compatible API."""
    results = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=32,
            temperature=0,
        )
        results.append(response.choices[0].text)
    return results


def init_weight_transfer_engine(base_url: str) -> None:
    """Initialize weight transfer via HTTP endpoint (no-op for NPU IPC)."""
    url = f"{base_url}/init_weight_transfer_engine"
    payload: dict[str, dict] = {"init_info": {}}
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()


def start_weight_update(base_url: str, is_checkpoint_format: bool = True) -> None:
    """Start weight update via HTTP endpoint.

    Prepares the model for layerwise reload on the vLLM server side.
    Must be called before update_weights.
    """
    url = f"{base_url}/start_weight_update"
    payload = {"is_checkpoint_format": is_checkpoint_format}
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()


def finish_weight_update(base_url: str) -> None:
    """Finish weight update via HTTP endpoint.

    Finalizes layerwise reload on the vLLM server side.
    Must be called after all update_weights calls are complete.
    """
    url = f"{base_url}/finish_weight_update"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def pause_generation(base_url: str) -> None:
    """Pause generation via HTTP endpoint."""
    url = f"{base_url}/pause"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def resume_generation(base_url: str) -> None:
    """Resume generation via HTTP endpoint."""
    url = f"{base_url}/resume"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def main():
    # NPU IPC requires the training model to be on the same NPU as the vLLM server.
    # The server should be started on NPU 0 with reduced memory utilization.
    device = "npu:0"
    torch.accelerator.set_device_index(device)

    # Load the training model on the same NPU as the server.
    # Use bfloat16 to reduce memory footprint.
    print(f"Loading training model: {MODEL_NAME} on {device}")
    print(
        "Note: Ensure the vLLM server was started with --gpu-memory-utilization 0.5 "
        "or lower to leave room for the training model."
    )
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)
    train_model.eval()

    # Create OpenAI client pointing to the vLLM server
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="EMPTY",
    )

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Generate text before weight update. The output is expected to be nonsense
    # because the server is initialized with dummy weights.
    print("-" * 50)
    print("Generating text BEFORE weight update (expect nonsense):")
    print("-" * 50)
    outputs = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Initialize weight transfer on vLLM server (no-op for NPU IPC)
    print("Initializing weight transfer (NPU IPC backend)...")
    init_weight_transfer_engine(BASE_URL)

    # Pause generation before weight sync
    pause_generation(BASE_URL)

    # Start weight update (prepares layerwise reload on the vLLM server)
    start_weight_update(BASE_URL)

    # Send weights via NPU IPC handles using HTTP mode.
    # trainer_send_weights internally collects all parameters,
    # creates IPC handles, and POSTs them to /update_weights.
    print("Broadcasting weights via NPU IPC (HTTP)...")
    trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode="http", url=BASE_URL)
    NPUIPCWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=trainer_args,
    )

    # Finish weight update (finalizes layerwise reload on the vLLM server)
    finish_weight_update(BASE_URL)

    # Resume generation after weight sync
    resume_generation(BASE_URL)

    # Generate text after weight update. The output is expected to be normal
    # because the real weights are now loaded.
    print("-" * 50)
    print("Generating text AFTER weight update:")
    print("-" * 50)
    outputs_updated = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs_updated):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Note: The training model and IPC handles remain in memory.
    # In a real RLHF training loop, you would update the training model
    # and create new IPC handles for each weight update.


if __name__ == "__main__":
    main()
