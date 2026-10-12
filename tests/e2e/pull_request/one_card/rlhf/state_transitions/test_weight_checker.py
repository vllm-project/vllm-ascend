# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Single-NPU end-to-end tests for the Weight Checker endpoint.

The Weight Checker (upstream vLLM PR #51350, requires a paired vLLM that
includes merge commit 42d9f8ccdb) exposes three dev-mode actions over
POST /weight_checker — checksum, reset and compare — so an RL stack can prove
that a weight update landed on every rank. These lanes cover single-card
behavior: request validation, digest stability, key coverage, the
reset -> checkpoint-reload recovery, the NPU IPC transfer round trip, and the
weight_nz_mode guardrail (transfer is rejected, reload still recovers).

The transaction machinery (layer-reduced model, fixed-random payload,
sleep/wake brackets) is shared with the NPU IPC weight-transfer tests in
tests/e2e/pull_request/rlhf/weight_transfer_test_utils.py.
"""

import json
import os

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    live_update_serve_args,
    packed_buffer_size_for,
    register_engines_once,
    wait_for_free_device_memory,
)

CONTROL_TIMEOUT = 60
RELOAD_TIMEOUT = 300
GPU_MEMORY_UTILIZATION = 0.45
MODEL = "Qwen/Qwen3-0.6B"
HF_OVERRIDES = {"num_hidden_layers": 4}
# Physical card the lane runs on; see the NPU IPC weight-transfer tests.
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))

CHECKER_CASE = WeightUpdateModelCase(
    id="qwen3-0.6b-checker",
    model=MODEL,
    hf_overrides=HF_OVERRIDES,
)

pytestmark = pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="Weight Checker e2e tests require at least 1 NPU.",
)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, params=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def weight_checker(server: RemoteOpenAIServer, action: str, baseline=None) -> requests.Response:
    body = {"action": action}
    if baseline is not None:
        body["baseline"] = baseline
    return requests.post(server.url_for("weight_checker"), json=body, timeout=RELOAD_TIMEOUT)


def checksums(server: RemoteOpenAIServer) -> dict[str, str]:
    response = weight_checker(server, "checksum")
    response.raise_for_status()
    return response.json()["checksums"]


def ranks(keys) -> set[str]:
    return {key.rsplit(":", 1)[0] for key in keys}


def _basic_serve_args(port: int, *, nz_mode: int, extra_args=()) -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--distributed-executor-backend",
        "mp",
        "--hf-overrides",
        json.dumps(HF_OVERRIDES),
        "--additional-config",
        json.dumps({"weight_nz_mode": nz_mode}),
        "--port",
        str(port),
        *extra_args,
    ]


def _env_dict() -> dict[str, str]:
    return {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }


@pytest.fixture(scope="module")
def checker_server():
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=_basic_serve_args(port, nz_mode=0),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=_env_dict(),
        auto_port=False,
    ) as server:
        yield server


@pytest.mark.parametrize("payload", [{}, {"action": "frobnicate"}, {"action": "compare"}])
def test_invalid_request_returns_400(checker_server, payload):
    response = requests.post(
        checker_server.url_for("weight_checker"),
        json=payload,
        timeout=CONTROL_TIMEOUT,
    )
    assert response.status_code == 400, response.text
    health = requests.get(checker_server.url_for("health"), timeout=CONTROL_TIMEOUT)
    assert health.status_code == 200


def test_checksum_is_stable_and_compare_reports_differences(checker_server):
    baseline = checksums(checker_server)
    assert baseline, "checksum must cover at least one parameter"
    hexdigits = set("0123456789abcdef")
    assert all(len(digest) == 64 and set(digest) <= hexdigits for digest in baseline.values())
    assert checksums(checker_server) == baseline

    changed, missing, *_ = baseline
    altered = {**baseline, changed: "0" * 64}
    del altered[missing]
    response = weight_checker(checker_server, "compare", altered)
    response.raise_for_status()
    assert response.json() == {"match": False, "mismatches": sorted([changed, missing])}


def test_checksum_keys_cover_single_rank_with_parameter_names(checker_server):
    # A full inventory against model.named_parameters() needs worker-side
    # access; client-side we assert the shape of the coverage: exactly one
    # rank prefix on a single card, dotted parameter names, and a sane count.
    baseline = checksums(checker_server)
    assert ranks(baseline) == {"dp0:pp0:pcp0:tp0"}
    names = {key.split(":", 4)[-1] for key in baseline}
    assert all("." in name for name in names)
    assert len(names) > 10


def test_reset_and_reload_restores(checker_server):
    baseline = checksums(checker_server)
    try:
        weight_checker(checker_server, "reset").raise_for_status()
        comparison = weight_checker(checker_server, "compare", baseline).json()
        assert comparison["match"] is False
        assert set(comparison["mismatches"]) == set(baseline)
    finally:
        _post(
            checker_server,
            "collective_rpc",
            json={"method": "reload_weights"},
            timeout=RELOAD_TIMEOUT,
        )
    assert weight_checker(checker_server, "compare", baseline).json() == {
        "match": True,
        "mismatches": [],
    }


def test_nz_mode_blocks_transfer_but_reload_recovers():
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=_basic_serve_args(
            port,
            nz_mode=1,
            extra_args=("--weight-transfer-config", json.dumps({"backend": "hccl"})),
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=_env_dict(),
        auto_port=False,
    ) as server:
        baseline = checksums(server)
        try:
            # reset succeeds under NZ (a warning is logged): the weights can
            # still be restored from the checkpoint via reload_weights.
            weight_checker(server, "reset").raise_for_status()
            # The transfer transaction is what NZ blocks: start_weight_update
            # fails its _check_nz_disabled guard on the worker.
            response = requests.post(
                server.url_for("collective_rpc"),
                json={"method": "start_weight_update"},
                timeout=CONTROL_TIMEOUT,
            )
            assert response.status_code >= 400, response.text
        finally:
            _post(server, "collective_rpc", json={"method": "reload_weights"}, timeout=RELOAD_TIMEOUT)
        assert weight_checker(server, "compare", baseline).json() == {
            "match": True,
            "mismatches": [],
        }


@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_npu_ipc_transfer_restores_reset_weights(packed: bool):
    """The documented RL flow over NPU IPC: transfer W, baseline, reset,
    transfer the same W again, and expect an exact match."""
    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    source = FixedRandomWeightSource(CHECKER_CASE, torch.device("npu", INFERENCE_DEVICE_INDEX))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env_dict = {**_env_dict(), "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}

    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=live_update_serve_args(
            CHECKER_CASE,
            backend="npu_ipc",
            port=port,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(
                rank=0,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
            source=source,
        )

        def transfer() -> None:
            # Same-chip bracket, mirrored from the NPU IPC weight-transfer
            # tests: release the engine's HBM to the co-located trainer, map
            # the weights allocation back for the layerwise reload, and wake
            # the KV cache last.
            _post(server, "pause")
            _post(server, "sleep", params={"level": 2})
            _post(server, "wake_up", params={"tags": ["weights"]})
            engine.send_weights()
            _post(server, "wake_up", params={"tags": ["kv_cache"]})

        transfer()
        baseline = checksums(server)

        weight_checker(server, "reset").raise_for_status()
        comparison = weight_checker(server, "compare", baseline).json()
        assert comparison["match"] is False
        assert ranks(comparison["mismatches"]) == ranks(baseline)

        transfer()
        assert weight_checker(server, "compare", baseline).json() == {
            "match": True,
            "mismatches": [],
        }
