# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Draft HCCL client, model snapshots and two-draft/one-target transaction."""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.rlhf.common import post_control as _post
from tests.e2e.rlhf.common import (
    register_engines_once,
    rlhf_server,
    server_visible_devices,
    wait_for_free_device_memory,
)
from tests.e2e.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.rlhf.weight_transfer.oracle import generation_signature
from tests.e2e.rlhf.weight_transfer.weight_source import (
    FIXED_WEIGHT_SEED,
    FixedRandomWeightSource,
    packed_buffer_size_for,
)

INFERENCE_WORLD_SIZE = 1
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
TRAINER_DEVICE_INDEX = INFERENCE_DEVICE_INDEX + 1
UPDATE_TIMEOUT = 300


def _log(message):
    print(f"[trainer] {message}", flush=True)


class WeightUpdateClient(HTTPVLLMWeightSyncClient):
    """Select the draft START endpoint without changing the trainer protocol."""

    def __init__(self, base_url: str):
        super().__init__(base_url=base_url, timeout=UPDATE_TIMEOUT)
        self.is_draft = False

    def start_weight_update(self) -> None:
        self._post("start_draft_weight_update" if self.is_draft else "start_weight_update")


def send_update(server, engine, client: WeightUpdateClient, source, *, is_draft: bool = False):
    client.is_draft = is_draft
    engine.source = source

    _post(server, "pause")
    engine.send_weights()
    _post(server, "resume")


class WeightUpdateTestWorkerExtension:
    """Read real worker parameters through the development collective RPC."""

    def snapshot_weight_update_models(self):
        result = {}
        for name, model in (("target", self.get_model()), ("draft", self.get_draft_model())):
            assert model is not None, f"{name} model is missing"
            result[name] = {
                key: {
                    "digest": hashlib.sha256(
                        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "data_ptr": value.data_ptr(),
                }
                for key, value in model.named_parameters()
            }
        return result


def _snapshot_models(server):
    response = _post(server, "collective_rpc", json={"method": "snapshot_weight_update_models"})
    results = response.json()["results"]
    assert len(results) == INFERENCE_WORLD_SIZE
    return results[0]


def _parameter_digests(snapshot):
    return {model: {name: value["digest"] for name, value in params.items()} for model, params in snapshot.items()}


def _parameter_addresses(snapshot):
    return {model: {name: value["data_ptr"] for name, value in params.items()} for model, params in snapshot.items()}


def _write_tiny_draft_config(directory: Path, *, num_hidden_layers: int = 2) -> WeightUpdateModelCase:
    """Use PR #20130's local Qwen3 topology, without model downloads."""
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=128,
        bos_token_id=1,
        eos_token_id=127,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    ).save_pretrained(directory)
    vocabulary = {"<unk>": 0, "<bos>": 1, **{f"token_{i}": i for i in range(2, 127)}, "<eos>": 127}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>"
    ).save_pretrained(directory)
    return WeightUpdateModelCase(id="tiny-qwen3-draft", model=str(directory), hf_overrides={})


def _draft_serve_args(checkpoint: Path, case, port: int, *, enforce_eager: bool, live: bool) -> list[str]:
    args = [
        "--tokenizer",
        case.model,
        "--served-model-name",
        case.id,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "64",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "64",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.2",
        "--distributed-executor-backend",
        "mp",
        "--port",
        str(port),
        "--additional-config",
        '{"rl_config": {"enabled": true}}',
        "--speculative-config",
        json.dumps(
            {
                "method": "draft_model",
                "model": str(checkpoint / "draft"),
                "num_speculative_tokens": 2,
            }
        ),
        "--worker-extension-cls",
        "tests.e2e.rlhf.weight_transfer.draft_hccl.WeightUpdateTestWorkerExtension",
    ]
    if enforce_eager:
        args.append("--enforce-eager")
    else:
        args += [
            "--compilation-config",
            '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]}',
        ]
    if live:
        args += ["--weight-transfer-config", '{"backend": "hccl"}']
    return args


def run_draft_hccl_weight_transfer(tmp_path, monkeypatch, packed: bool, enforce_eager: bool):
    """Full draft payload twice, then target payload, against independent loads.

    Greedy output alone cannot observe a changed draft. Match every parameter
    digest to a startup-load reference as well, and preserve runtime addresses.
    """
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    device = torch.device("npu", TRAINER_DEVICE_INDEX)
    case = _write_tiny_draft_config(tmp_path / "config")
    draft_case = _write_tiny_draft_config(tmp_path / "draft-config", num_hidden_layers=1)
    initial = FixedRandomWeightSource(case, device, draft_case=draft_case)
    draft_updated = FixedRandomWeightSource(case, device, draft_case=draft_case, draft_seed=FIXED_WEIGHT_SEED + 2)
    target_updated = FixedRandomWeightSource(
        case, device, seed=FIXED_WEIGHT_SEED + 3, draft_case=draft_case, draft_seed=FIXED_WEIGHT_SEED + 2
    )
    initial_path = tmp_path / "initial"
    initial.write_checkpoint(initial_path)
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(INFERENCE_DEVICE_INDEX, INFERENCE_WORLD_SIZE),
    }
    references = []
    for name, source in (("draft-reference", draft_updated), ("target-reference", target_updated)):
        checkpoint = tmp_path / name
        source.write_checkpoint(checkpoint)
        wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
        port = get_open_port()
        with rlhf_server(
            str(checkpoint),
            vllm_serve_args=_draft_serve_args(checkpoint, case, port, enforce_eager=enforce_eager, live=False),
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
        ) as server:
            signature = generation_signature(server.get_client(), case.id)
            references.append((_parameter_digests(_snapshot_models(server)), signature))

    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
    port = get_open_port()
    with rlhf_server(
        str(initial_path),
        vllm_serve_args=_draft_serve_args(initial_path, case, port, enforce_eager=enforce_eager, live=True),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
    ) as server:
        client = server.get_client()
        generation_signature(client, case.id)  # Capture/warm up graphs before checking addresses.
        before = _snapshot_models(server)
        addresses = _parameter_addresses(before)
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        register_engines_once()
        sync_client = WeightUpdateClient(server.url_root)
        assert draft_updated.draft_source is not None
        engine = WeightTransferTrainerFactory.trainer_init(
            HCCLTrainerInitInfo(
                rank=0,
                master_address=get_ip(),
                master_port=get_open_port(),
                world_size=INFERENCE_WORLD_SIZE + 1,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(target_updated),
            ),
            client=sync_client,
            source=draft_updated.draft_source,
        )
        try:
            assert _parameter_digests(before)["draft"] != references[0][0]["draft"], "Draft payload is not observable"
            for update_round in range(2):
                _log(f"draft round={update_round + 1}")
                send_update(server, engine, sync_client, draft_updated.draft_source, is_draft=True)
                assert generation_signature(client, case.id) == references[0][1]
                after = _snapshot_models(server)
                assert _parameter_digests(after) == references[0][0]
                assert _parameter_digests(after)["target"] == _parameter_digests(before)["target"]
                assert _parameter_addresses(after) == addresses

            send_update(server, engine, sync_client, target_updated, is_draft=False)
            assert generation_signature(client, case.id) == references[1][1]
            after = _snapshot_models(server)
            assert _parameter_digests(after) == references[1][0]
            assert _parameter_addresses(after) == addresses
        finally:
            engine.shutdown()
