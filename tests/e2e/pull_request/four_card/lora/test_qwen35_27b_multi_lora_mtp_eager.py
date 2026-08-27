# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Qwen3.5-27B named multi-LoRA + built-in MTP, eager (no ACL graph).
# PR CI sets HF_HUB_OFFLINE=1 and VLLM_USE_MODELSCOPE=true. Public 27B
# PEFT adapters are not on ModelScope, so this test synthesizes two
# rank-8 adapters with Qwen3.5-27B MLP shapes. Override with MIX_LORA_A /
# MIX_LORA_B to use local checkpoints.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests
import torch
from safetensors.torch import save_file
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

HUB_MODEL = "Qwen/Qwen3.5-27B"
MODEL_PATH = os.environ.get("QWEN35_27B", HUB_MODEL)
# Qwen3.5-27B text_config: hidden=5120, intermediate=17408, 64 layers.
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
NUM_LAYERS = 64
LORA_RANK = 8
PROMPT = "你是谁？"


def _resolve_adapter_dir(root: str) -> str:
    path = Path(root)
    matches = sorted(path.rglob("adapter_config.json"), key=lambda p: len(p.parts))
    if not matches:
        raise FileNotFoundError(f"No adapter_config.json under {root}")
    return str(matches[0].parent)


def _write_dummy_lora(root: Path, seed: int) -> str:
    """Write a PEFT adapter that vLLM can load onto Qwen3.5-27B.

    Keys follow the Qwen3.5 PEFT layout
    ``base_model.model.model.language_model.layers.*.mlp.down_proj``.
    Seed 0 is zeros (near-identity); a nonzero seed changes greedy output.
    """
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": LORA_RANK,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["down_proj"],
        "base_model_name_or_path": HUB_MODEL,
        "inference_mode": True,
    }
    (root / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tensors: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(seed)
    for layer_idx in range(NUM_LAYERS):
        prefix = f"base_model.model.model.language_model.layers.{layer_idx}.mlp.down_proj"
        if seed == 0:
            lora_a = torch.zeros(LORA_RANK, INTERMEDIATE_SIZE)
            lora_b = torch.zeros(HIDDEN_SIZE, LORA_RANK)
        else:
            lora_a = torch.randn(LORA_RANK, INTERMEDIATE_SIZE, generator=generator) * 0.05
            lora_b = torch.randn(HIDDEN_SIZE, LORA_RANK, generator=generator) * 0.05
        tensors[f"{prefix}.lora_A.weight"] = lora_a
        tensors[f"{prefix}.lora_B.weight"] = lora_b
    save_file(tensors, str(root / "adapter_model.safetensors"))
    return str(root)


def _lora_dir(env_key: str, tmp_root: Path, seed: int) -> str:
    local = os.environ.get(env_key)
    if local:
        return _resolve_adapter_dir(local)
    return _write_dummy_lora(tmp_root, seed)


@pytest.fixture(scope="session")
def qwen35_27b_lora_a_files(tmp_path_factory) -> str:
    return _lora_dir("MIX_LORA_A", tmp_path_factory.mktemp("mix-lora-a"), seed=0)


@pytest.fixture(scope="session")
def qwen35_27b_lora_b_files(tmp_path_factory) -> str:
    return _lora_dir("MIX_LORA_B", tmp_path_factory.mktemp("mix-lora-b"), seed=1)


def _chat(server: RemoteOpenAIServer, model_name: str) -> str:
    resp = requests.post(
        server.url_for("v1", "chat", "completions"),
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    assert content, f"empty response for model={model_name}"
    return content.strip()


@pytest.mark.e2e_model("Qwen/Qwen3.5-27B")
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="lora,multi_lora,fully_sharded_lora,mtp",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_qwen35_27b_named_multi_lora_mtp_eager(
    qwen35_27b_lora_a_files,
    qwen35_27b_lora_b_files,
):
    port = get_open_port()
    server_args = [
        "--trust-remote-code",
        "--enforce-eager",
        "--enable-lora",
        "--max-loras",
        "2",
        "--max-lora-rank",
        str(LORA_RANK),
        "--fully-sharded-loras",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        "4",
        "--distributed-executor-backend",
        "mp",
        "--port",
        str(port),
        "--speculative-config",
        json.dumps(
            {
                "method": "qwen3_5_mtp",
                "num_speculative_tokens": 3,
                "enforce_eager": True,
            }
        ),
        "--lora-modules",
        json.dumps(
            {
                "name": "mix-lora-a",
                "path": qwen35_27b_lora_a_files,
                "base_model_name": HUB_MODEL,
            }
        ),
        json.dumps(
            {
                "name": "mix-lora-b",
                "path": qwen35_27b_lora_b_files,
                "base_model_name": HUB_MODEL,
            }
        ),
    ]

    with RemoteOpenAIServer(
        MODEL_PATH,
        server_args,
        server_port=port,
        auto_port=False,
        env_dict={"HCCL_BUFFSIZE": "1024"},
    ) as server:
        models = requests.get(server.url_for("v1", "models"), timeout=60)
        models.raise_for_status()
        served = {item["id"] for item in models.json()["data"]}
        assert "mix-lora-a" in served and "mix-lora-b" in served, served
        text_a = _chat(server, "mix-lora-a")
        text_b = _chat(server, "mix-lora-b")
        print(f"mix-lora-a: {text_a!r}")
        print(f"mix-lora-b: {text_b!r}")
        assert text_a != text_b, f"named LoRA routing collapsed: {text_a!r}"
