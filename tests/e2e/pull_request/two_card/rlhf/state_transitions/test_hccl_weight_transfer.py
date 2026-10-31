# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-aware two-card HCCL live weight-update regression tests.

NPU 0 hosts the inference worker and NPU 1 deterministically generates every
parameter of the layer-reduced model. Names and shapes come from a meta-device
HF model; values are fixed-random BF16 and require no checkpoint weights.

The correctness oracle is *normal startup loading of that same payload*: the
generator also writes a temporary checkpoint, a reference server loads it at
startup, and the live-update lane has to reproduce its signature exactly —

    normal startup load(W) == first live update(W) == second live update(W)

The dummy-started lane's pre-update signature must differ from the reference, so
a transfer that never ran cannot pass, and the second update covers the layerwise
reload lifecycle, runtime/destructive representations, derived state and the
graph-captured storage of the first update. The packed lane is currently skipped
on this branch — see ``PACKED_LANE_SKIP_REASON`` — so only the unpacked lane runs.

The trainer factory drives the handshake and both update rounds through
``HCCLTrainerWeightTransferEngine`` and ``HTTPVLLMWeightSyncClient``, including
the START -> broadcast -> FINISH transaction and propagation of RPC errors.

The draft lane uses local Qwen3 target/draft checkpoints, compares all parameter
digests with independent startup loads and verifies storage identity across two
draft updates followed by a target update. It covers the basic autoregressive
draft path, not the EAGLE/MTP/DFLASH/DSpark architecture matrix.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from tests.e2e.conftest import RemoteOpenAIServer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    FIXED_WEIGHT_SEED,
    PACKED_LANE_SKIP_REASON,
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    assert_weight_update_matches_reference,
    generation_signature,
    live_update_serve_args,
    packed_buffer_size_for,
    pytest_model_cases,
    reference_signature,
    register_engines_once,
    wait_for_free_device_memory,
)

INFERENCE_WORLD_SIZE = 1
# Card the inference server runs on, as an absolute device index inside the
# container: the harness hands the server `ASCEND_RT_VISIBLE_DEVICES` explicitly,
# so this is a physical chip. The trainer sits next to it. The default is the
# first card pair CI uses; a shared host can point a lane at idle cards by
# exporting `VLLM_RL_TEST_DEVICE_INDEX` (leaving `ASCEND_RT_VISIBLE_DEVICES`
# unset, so the pytest process's logical indices equal the physical ones).
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
TRAINER_DEVICE_INDEX = INFERENCE_DEVICE_INDEX + 1
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60
GPU_MEMORY_UTILIZATION = 0.75


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="HCCL weight transfer e2e test requires at least 2 NPUs.",
)
@pytest.mark.parametrize("case", pytest_model_cases())
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    if case.skip_reason is not None:
        pytest.skip(f"{case.id}: {case.skip_reason}")
    if packed:
        pytest.skip(f"{case.id}: {PACKED_LANE_SKIP_REASON}")

    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = FixedRandomWeightSource(case, torch.device("npu", TRAINER_DEVICE_INDEX))
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }

    # Independent oracle first: a server that loads exactly this payload at
    # startup, on its own lifecycle, before the live path touches anything.
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=INFERENCE_WORLD_SIZE,
        device_index=INFERENCE_DEVICE_INDEX,
        env_dict=env_dict,
    )

    # Dummy sanity, first live update and the reload regression share one lane.
    # Its pre-update signature is the dummy-sanity check; the *reference* is the
    # part that must come from a separate server. The reference and the lane share
    # one card, so wait for its process tree to hand the HBM back.
    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend="hccl",
            port=port,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=INFERENCE_WORLD_SIZE,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()
        dummy_signature = generation_signature(client, case.model)

        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            HCCLTrainerInitInfo(
                rank=0,
                master_address=get_ip(),
                master_port=get_open_port(),
                world_size=INFERENCE_WORLD_SIZE + 1,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root, timeout=UPDATE_TIMEOUT),
            source=source,
        )
        signatures = []
        try:
            for update_round in range(2):
                _log(
                    f"{case.id}: sending {('packed' if packed else 'unpacked')} "
                    f"fixed-random update round={update_round + 1}"
                )
                _post(server, "pause")
                try:
                    engine.send_weights()
                finally:
                    _post(server, "resume")
                signatures.append(generation_signature(client, case.model))
        finally:
            engine.shutdown()

    updated_signature, reloaded_signature = signatures
    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        updated_signature,
        reloaded_signature,
        case,
    )


class _WeightUpdateClient(HTTPVLLMWeightSyncClient):
    """Select the draft START endpoint without changing the trainer protocol."""

    def __init__(self, base_url: str):
        super().__init__(base_url=base_url, timeout=UPDATE_TIMEOUT)
        self.is_draft = False

    def start_weight_update(self) -> None:
        self._post("start_draft_weight_update" if self.is_draft else "start_weight_update")


def _send_update(server, engine, client: _WeightUpdateClient, source, *, is_draft: bool = False):
    client.is_draft = is_draft
    engine.source = source
    _post(server, "pause")
    try:
        engine.send_weights()
    finally:
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
        "tests.e2e.pull_request.two_card.rlhf.state_transitions.test_hccl_weight_transfer.WeightUpdateTestWorkerExtension",
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


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="HCCL draft updates require two NPUs.")
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "aclgraph"])
def test_hccl_draft_then_target_update(tmp_path, monkeypatch, packed: bool, enforce_eager: bool):
    """Full draft payload twice, then target payload, against independent loads.

    Greedy output alone cannot observe a changed draft. Match every parameter
    digest to a startup-load reference as well, and preserve runtime addresses.
    """
    if packed:
        pytest.skip(PACKED_LANE_SKIP_REASON)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
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
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }
    references = []
    for name, source in (("draft-reference", draft_updated), ("target-reference", target_updated)):
        checkpoint = tmp_path / name
        source.write_checkpoint(checkpoint)
        wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
        port = get_open_port()
        with RemoteOpenAIServer(
            str(checkpoint),
            vllm_serve_args=_draft_serve_args(checkpoint, case, port, enforce_eager=enforce_eager, live=False),
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
            auto_port=False,
        ) as server:
            signature = generation_signature(server.get_client(), case.id)
            references.append((_parameter_digests(_snapshot_models(server)), signature))

    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
    port = get_open_port()
    with RemoteOpenAIServer(
        str(initial_path),
        vllm_serve_args=_draft_serve_args(initial_path, case, port, enforce_eager=enforce_eager, live=True),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()
        generation_signature(client, case.id)  # Capture/warm up graphs before checking addresses.
        before = _snapshot_models(server)
        addresses = _parameter_addresses(before)
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        register_engines_once()
        sync_client = _WeightUpdateClient(server.url_root)
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
                _send_update(server, engine, sync_client, draft_updated.draft_source, is_draft=True)
                assert generation_signature(client, case.id) == references[0][1]
                after = _snapshot_models(server)
                assert _parameter_digests(after) == references[0][0]
                assert _parameter_digests(after)["target"] == _parameter_digests(before)["target"]
                assert _parameter_addresses(after) == addresses

            _send_update(server, engine, sync_client, target_updated, is_draft=False)
            assert generation_signature(client, case.id) == references[1][1]
            after = _snapshot_models(server)
            assert _parameter_digests(after) == references[1][0]
            assert _parameter_addresses(after) == addresses
        finally:
            engine.shutdown()
