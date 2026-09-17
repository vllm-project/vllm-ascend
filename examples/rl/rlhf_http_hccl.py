#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with native weight syncing APIs.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- HCCL for actual weight data transfer

Prerequisites:
    Start a vLLM server with weight transfer enabled:

    $ VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6b \
        --enforce-eager \
        --weight-transfer-config '{"backend": "hccl"}' \
        --load-format dummy

    Then run this script:

    $ python rlhf_http_hccl.py

The example performs the following steps:

* Load the training model on NPU 0.
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Build the trainer-side HCCL engine, which opens the HCCL group (trainer is
  rank 0) and drives the whole weight-update transaction over HTTP.
* Broadcast the real weights from the training model to the vLLM server
  using HCCL.
* Generate text again to show normal output after the weight update.
"""

import logging

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen3-0.6B"


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


def get_world_size(base_url: str) -> int:
    """Get world size from the vLLM server."""
    url = f"{base_url}/get_world_size"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Get the inference world size from the vLLM server
    inference_world_size = get_world_size(BASE_URL)
    world_size = inference_world_size + 1  # +1 for the trainer
    device = f"npu:{inference_world_size}"
    torch.accelerator.set_device_index(device)

    # Load the training model
    logger.info("Loading training model: %s", MODEL_NAME)
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)

    # Create OpenAI client pointing to the vLLM server
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="EMPTY",  # vLLM doesn't require an API key by default
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
    logger.info("-" * 50)
    logger.info("Generating text BEFORE weight update (expect nonsense):")
    logger.info("-" * 50)
    outputs = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs):
        logger.info("Prompt: %r\nGenerated text: %r", prompt, generated_text)
        logger.info("-" * 50)

    # Set up the communication channel between the training process and the
    # vLLM server. The trainer is HCCL rank 0 and the vLLM worker(s) follow at
    # rank 1, so the group is the inference workers plus the trainer.
    master_address = get_ip()
    master_port = get_open_port()

    # Size the packed buffer to fit the largest tensor with 128 MB headroom,
    # but keep the default 1 GB when the largest tensor is smaller than that.
    max_tensor_bytes = max(p.numel() * p.element_size() for _, p in train_model.named_parameters())
    packed_buffer_size_bytes = max(max_tensor_bytes + 128 * 2**20, 2**30)
    logger.info(
        "Largest tensor: %.2f GiB, packed buffer: %.2f GiB",
        max_tensor_bytes / 2**30,
        packed_buffer_size_bytes / 2**30,
    )

    # Pause generation before weight sync
    pause_generation(BASE_URL)

    logger.info("Initializing weight transfer: master=%s:%s", master_address, master_port)

    # The trainer-side engine owns the whole transaction: ``trainer_init`` opens
    # the rank-0 HCCL endpoint and hands the worker its matching init info over
    # HTTP, then ``send_weights`` runs START -> broadcast -> FINISH. packed=True
    # enables efficient batched tensor broadcasting.
    from vllm.distributed.weight_transfer.base import ModuleSource
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

    engine = WeightTransferTrainerFactory.trainer_init(
        HCCLTrainerInitInfo(
            rank=0,
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            packed=True,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
        ),
        client=HTTPVLLMWeightSyncClient(base_url=BASE_URL),
        source=ModuleSource(train_model),
    )

    # Broadcast all weights from trainer to vLLM workers
    logger.info("Broadcasting weights via HCCL...")
    engine.send_weights()

    # Resume generation after weight sync
    resume_generation(BASE_URL)

    # Generate text after weight update. The output is expected to be normal
    # because the real weights are now loaded.
    logger.info("-" * 50)
    logger.info("Generating text AFTER weight update:")
    logger.info("-" * 50)
    outputs_updated = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs_updated):
        logger.info("Prompt: %r\nGenerated text: %r", prompt, generated_text)
        logger.info("-" * 50)


if __name__ == "__main__":
    main()
