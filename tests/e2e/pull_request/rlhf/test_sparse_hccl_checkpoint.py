# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Qwen3 MoE loader oracle for checkpoint sparse HCCL (5/9 NPUs).

The checkpoint is deliberately small and synthetic. This checks native TP/EP
mapping and generation equivalence, not pretrained-model accuracy/performance.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3MoeConfig
from vllm.distributed import get_dp_group
from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient, WeightTransferTrainerFactory

from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import register_engines_once
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerInitInfo,
    checkpoint_loader_views,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch

TP_SIZE = 4
NUM_EXPERTS = 8
HIDDEN_SIZE = 1024
HEAD_SIZE = 128
MOE_SIZE = 512
VOCAB_SIZE = 512


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_checkpoint(path):
    path.mkdir()
    config = Qwen3MoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MOE_SIZE,
        moe_intermediate_size=MOE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=HEAD_SIZE,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        architectures=["Qwen3MoeForCausalLM"],
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.save_pretrained(path)
    shapes = {
        "model.embed_tokens.weight": (VOCAB_SIZE, HIDDEN_SIZE),
        "lm_head.weight": (VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": (HIDDEN_SIZE,),
        "model.layers.0.input_layernorm.weight": (HIDDEN_SIZE,),
        "model.layers.0.post_attention_layernorm.weight": (HIDDEN_SIZE,),
        "model.layers.0.self_attn.q_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        "model.layers.0.self_attn.k_proj.weight": (HIDDEN_SIZE // 2, HIDDEN_SIZE),
        "model.layers.0.self_attn.v_proj.weight": (HIDDEN_SIZE // 2, HIDDEN_SIZE),
        "model.layers.0.self_attn.o_proj.weight": (HIDDEN_SIZE, HIDDEN_SIZE),
        "model.layers.0.self_attn.q_norm.weight": (HEAD_SIZE,),
        "model.layers.0.self_attn.k_norm.weight": (HEAD_SIZE,),
        "model.layers.0.mlp.gate.weight": (NUM_EXPERTS, HIDDEN_SIZE),
    }
    for expert in range(NUM_EXPERTS):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        shapes[f"{prefix}.gate_proj.weight"] = (MOE_SIZE, HIDDEN_SIZE)
        shapes[f"{prefix}.up_proj.weight"] = (MOE_SIZE, HIDDEN_SIZE)
        shapes[f"{prefix}.down_proj.weight"] = (HIDDEN_SIZE, MOE_SIZE)
    generator = torch.Generator().manual_seed(1234)
    weights = {
        name: torch.ones(shape, dtype=torch.bfloat16)
        if "norm" in name
        else (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16)
        for name, shape in shapes.items()
    }
    save_file(weights, path / "model.safetensors")
    records = []
    for name, weight in weights.items():
        # Touch each quarter and both ends, including every global expert owner.
        indices = torch.tensor(
            sorted({0, weight.numel() - 1, *(weight.numel() * n // TP_SIZE for n in range(TP_SIZE))}), dtype=torch.int32
        )
        records.append(
            dict(
                name=name,
                full_shape=tuple(weight.shape),
                indices=indices,
                values=weight.flatten()[indices.long()] + 0.0625,
            )
        )
    repeated = next(record for record in records if record["name"].endswith("q_proj.weight"))
    records.append({**repeated, "values": repeated["values"] + 0.0625})
    torch.save(records, path / "patches.pt")


class SparseCheckpointOracle:
    """Test-only worker extension comparing complete runtime parameter state."""

    def sparse_digest(self):
        hashes = {}
        for name, parameter in self.get_model().named_parameters():
            raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
        # The HTTP utility API broadcasts to all DP engines but returns only
        # DP0's results. Gather matching TP ranks across DP to verify every owner.
        group = get_dp_group()
        gathered = [None] * group.world_size
        torch.distributed.all_gather_object(gathered, hashes, group=group.cpu_group)
        return gathered

    def sparse_capture_baseline(self):
        self.sparse_baseline = {name: p.detach().cpu().clone() for name, p in self.get_model().named_parameters()}
        return self.sparse_digest()

    @torch.no_grad()
    def sparse_restore_baseline(self):
        for name, parameter in self.get_model().named_parameters():
            parameter.copy_(self.sparse_baseline[name])
        return self.sparse_digest()

    @torch.no_grad()
    def sparse_dense_reference(self, model_path):
        weights = load_file(str(Path(model_path) / "model.safetensors"))
        records = torch.load(Path(model_path) / "patches.pt", weights_only=True)
        for record in records:
            weights[record["name"]].view(-1).index_copy_(0, record["indices"].long(), record["values"])
        with checkpoint_loader_views(self.get_model()):
            self.get_model().load_weights(weights.items())
        return self.sparse_digest()


def rpc(url, method, args=()):
    response = requests.post(f"{url}/collective_rpc", json={"method": method, "args": list(args)}, timeout=180)
    response.raise_for_status()
    return [hashes for tp_results in response.json()["results"] for hashes in tp_results]


def generate(url):
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "sparse-oracle",
            "prompt": [1, 13, 42, 77],
            "max_tokens": 4,
            "temperature": 0,
            "seed": 1234,
            "return_token_ids": True,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"]


@pytest.mark.parametrize("dp_size", [1, 2], ids=["tp4-ep4", "tp4-dp2-ep8"])
def test_sparse_checkpoint_matches_native_dense(tmp_path, dp_size):
    """Compare every worker parameter and generated token with dense loading."""
    workers = TP_SIZE * dp_size
    if torch.npu.device_count() < workers + 1:
        pytest.skip(f"requires {workers + 1} NPUs (including a distinct trainer)")
    model_path = tmp_path / "checkpoint"
    write_checkpoint(model_path)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(1, workers + 1))
    environment["VLLM_SERVER_DEV_MODE"] = "1"
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_path),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--served-model-name",
        "sparse-oracle",
        "--skip-tokenizer-init",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--data-parallel-size",
        str(dp_size),
        "--enable-expert-parallel",
        "--distributed-executor-backend",
        "mp",
        "--gpu-memory-utilization",
        "0.15",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "128",
        "--no-enable-prefix-caching",
        "--additional-config",
        '{"weight_nz_mode":0}',
        "--weight-transfer-config",
        '{"backend":"sparse_hccl"}',
        "--worker-extension-cls",
        __name__ + ".SparseCheckpointOracle",
    ]
    engine = None
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(args, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "server.log").read_text()[-10000:])
                try:
                    if requests.get(f"{url}/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                pytest.fail("server did not become ready")
            baseline = rpc(url, "sparse_capture_baseline")
            reference = rpc(url, "sparse_dense_reference", [str(model_path)])
            assert len(reference) == workers
            assert all(before != after for before, after in zip(baseline, reference, strict=True))
            expected = generate(url)
            assert rpc(url, "sparse_restore_baseline") == baseline
            torch.npu.set_device(0)
            register_engines_once()
            client = HTTPVLLMWeightSyncClient(url)
            requests.post(f"{url}/pause", timeout=60).raise_for_status()
            engine = WeightTransferTrainerFactory.trainer_init(
                init_info=SparseHCCLTrainerInitInfo("127.0.0.1", free_port(), workers + 1, rank=0), client=client
            )
            records = torch.load(model_path / "patches.pt", weights_only=True)
            client.start_weight_update()
            for offset in range(0, len(records), 4):
                engine.send_weight_chunk(SparseWeightPatch(**r) for r in records[offset : offset + 4])
            client.finish_weight_update()
            assert rpc(url, "sparse_digest") == reference
            requests.post(f"{url}/resume", timeout=60).raise_for_status()
            actual = generate(url)
            assert all(len(c["token_ids"]) == 4 for c in actual)
            assert [(c["token_ids"], c["finish_reason"]) for c in actual] == [
                (c["token_ids"], c["finish_reason"]) for c in expected
            ]
            (tmp_path / "validation.json").write_text(
                json.dumps(
                    {
                        "tp": TP_SIZE,
                        "dp": dp_size,
                        "ep": workers,
                        "checkpoint_tensors": len(records),
                        "runtime_hashes": reference,
                        "generation": [(c["token_ids"], c["finish_reason"]) for c in actual],
                        "all_rank_hashes_equal": True,
                    },
                    indent=2,
                )
            )
        finally:
            if engine is not None:
                engine.shutdown()
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
