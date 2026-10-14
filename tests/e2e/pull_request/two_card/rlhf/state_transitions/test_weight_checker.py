# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Weight Checker end-to-end tests on Ascend NPU.

Single-card scenarios run on the inference device alone; multi-rank
scenarios use the inference and trainer devices together. This file lives
in the two_card lane so the CI runner always provides both cards. Covered:
request validation, digest stability, key coverage, the reset ->
checkpoint-reload recovery, the NPU IPC transfer round trip, TP2/DP2 rank
coverage with missing-rank detection, and the cross-NPU HCCL round trip.

The Weight Checker (upstream vLLM PR #51350, requires a paired vLLM that
includes merge commit 42d9f8ccdb) exposes three dev-mode actions over
POST /weight_checker — checksum, reset and compare — so an RL stack can
prove that a weight update landed on every rank. The transaction machinery
(layer-reduced model, fixed-random payload, sleep/wake brackets) is shared
with the NPU weight-transfer tests in
tests/e2e/pull_request/rlhf/weight_transfer_test_utils.py, and the HCCL
trainer handshake is mirrored from test_hccl_weight_transfer.py.
"""

import json
import math
import os
import threading

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    CONTROL_TIMEOUT,
    RELOAD_TIMEOUT,
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    checksums,
    live_update_serve_args,
    packed_buffer_size_for,
    post_route,
    ranks,
    register_engines_once,
    wait_for_free_device_memory,
    weight_checker,
)

# The server endpoint ships with the paired vLLM (PR #51350); skip the whole
# module while the CI-pinned vLLM predates it.
pytest.importorskip(
    "vllm.model_executor.model_loader.weight_checksum",
    reason="Weight Checker requires a paired vLLM with PR #51350 (>= v0.31.1rc0).",
)

INIT_TIMEOUT = 120
# Matches the validated acceptance setup; leaves HBM headroom for the
# co-located trainer in the same-chip transfer lanes.
GPU_MEMORY_UTILIZATION = 0.25
MODEL = "Qwen/Qwen3-0.6B"
HF_OVERRIDES = {"num_hidden_layers": 4}
# Physical cards; see the NPU weight-transfer tests.
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
TRAINER_DEVICE_INDEX = INFERENCE_DEVICE_INDEX + 1

CHECKER_CASE = WeightUpdateModelCase(
    id="qwen3-0.6b-checker",
    model=MODEL,
    hf_overrides=HF_OVERRIDES,
)

requires_one_npu = pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="Single-card Weight Checker scenarios require at least 1 NPU.",
)
requires_two_npus = pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="Multi-rank Weight Checker scenarios require at least 2 NPUs.",
)


def _basic_serve_args(port: int, *, extra_args=()) -> list[str]:
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
        json.dumps({"weight_nz_mode": 0}),
        "--port",
        str(port),
        "--enforce-eager",
        *extra_args,
    ]


def _env_dict(*device_indexes: int) -> dict[str, str]:
    return {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(index) for index in device_indexes),
    }


@pytest.fixture(scope="module")
def checker_server():
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=_basic_serve_args(port),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=_env_dict(INFERENCE_DEVICE_INDEX),
        auto_port=False,
    ) as server:
        yield server


@requires_one_npu
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


@requires_one_npu
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


@requires_one_npu
def test_checksum_keys_cover_single_rank_with_parameter_names(checker_server):
    # Single card: one rank prefix, dotted parameter names, and a sane count.
    baseline = checksums(checker_server)
    assert ranks(baseline) == {"dp0:pp0:pcp0:tp0"}
    names = {key.split(":", 4)[-1] for key in baseline}
    assert all("." in name for name in names)
    assert len(names) > 10


@requires_one_npu
def test_reset_and_reload_restores(checker_server):
    baseline = checksums(checker_server)
    try:
        weight_checker(checker_server, "reset").raise_for_status()
        comparison = weight_checker(checker_server, "compare", baseline).json()
        assert comparison["match"] is False
        assert set(comparison["mismatches"]) == set(baseline)
    finally:
        post_route(
            checker_server,
            "collective_rpc",
            json={"method": "reload_weights"},
            timeout=RELOAD_TIMEOUT,
        )
    assert weight_checker(checker_server, "compare", baseline).json() == {
        "match": True,
        "mismatches": [],
    }


@requires_one_npu
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_npu_ipc_transfer_restores_reset_weights(packed: bool):
    """The documented RL flow over NPU IPC: transfer W, baseline, reset,
    transfer the same W again, and expect an exact match."""
    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    source = FixedRandomWeightSource(CHECKER_CASE, torch.device("npu", INFERENCE_DEVICE_INDEX))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env_dict = {**_env_dict(INFERENCE_DEVICE_INDEX), "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}

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
            post_route(server, "pause")
            post_route(server, "sleep", params={"level": 2})
            post_route(server, "wake_up", params={"tags": ["weights"]})
            engine.send_weights()
            post_route(server, "wake_up", params={"tags": ["kv_cache"]})

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


@requires_two_npus
@pytest.mark.parametrize(
    "parallel_args, expected_ranks",
    [
        (["--tensor-parallel-size", "2"], {"dp0:pp0:pcp0:tp0", "dp0:pp0:pcp0:tp1"}),
        (["--data-parallel-size", "2"], {"dp0:pp0:pcp0:tp0", "dp1:pp0:pcp0:tp0"}),
    ],
    ids=["tp2", "dp2"],
)
def test_checksum_and_reset_cover_every_rank(parallel_args, expected_ranks):
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=_basic_serve_args(port, extra_args=parallel_args),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=_env_dict(INFERENCE_DEVICE_INDEX, TRAINER_DEVICE_INDEX),
        auto_port=False,
    ) as server:
        baseline = checksums(server)
        assert ranks(baseline) == expected_ranks

        # A rank dropped from the baseline must be reported exactly.
        victim = sorted(expected_ranks)[-1]
        partial = {key: digest for key, digest in baseline.items() if not key.startswith(victim + ":")}
        comparison = weight_checker(server, "compare", partial).json()
        assert comparison["match"] is False
        assert comparison["mismatches"] == sorted(key for key in baseline if key.startswith(victim + ":"))

        # A reset that skipped an engine would leave its keys matching.
        try:
            weight_checker(server, "reset").raise_for_status()
            comparison = weight_checker(server, "compare", baseline).json()
            assert comparison["match"] is False
            assert ranks(comparison["mismatches"]) == expected_ranks
        finally:
            post_route(server, "collective_rpc", json={"method": "reload_weights"}, timeout=RELOAD_TIMEOUT)
        assert weight_checker(server, "compare", baseline).json() == {
            "match": True,
            "mismatches": [],
        }


class _BackgroundPost(threading.Thread):
    """Run a blocking server-side HCCL RPC and surface its exception."""

    def __init__(self, server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
        super().__init__(daemon=True)
        self._server = server
        self._route = route
        self._json = json
        self._timeout = timeout
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            post_route(self._server, self._route, json=self._json, timeout=self._timeout)
        except BaseException as exc:  # noqa: BLE001 - re-raised by raise_if_failed
            self.error = exc

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"server-side /{self._route} failed") from self.error


def _collect_weight_metadata(source: FixedRandomWeightSource):
    """Parameter metadata plus a packed buffer that fits the largest tensor."""
    metadata = source.metadata()
    max_tensor_bytes = max(math.prod(meta.shape) * meta.dtype.itemsize for meta in metadata)
    return (
        [meta.name for meta in metadata],
        [str(meta.dtype).split(".")[-1] for meta in metadata],
        [list(meta.shape) for meta in metadata],
        max(max_tensor_bytes + 128 * 2**20, 2**30),
    )


def _send_update(server, source, model_update_group, *, packed: bool) -> None:
    """Run one START -> broadcast -> FINISH transaction over the server's RPCs."""
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    names, dtype_names, shapes, packed_buffer_size_bytes = _collect_weight_metadata(source)
    post_route(server, "pause")
    post_route(server, "start_weight_update")

    # update_weights blocks on the server while it waits for the HCCL broadcasts,
    # so run it in a thread while the trainer produces the data.
    update_thread = _BackgroundPost(
        server,
        "update_weights",
        json={
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": packed,
                "packed_buffer_size_bytes": packed_buffer_size_bytes,
            }
        },
        timeout=RELOAD_TIMEOUT,
    )
    update_thread.start()
    HCCLWeightTransferEngine.trainer_send_weights(
        iterator=iter(source),
        trainer_args=HCCLTrainerSendWeightsArgs(
            group=model_update_group,
            packed=packed,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
        ),
    )
    update_thread.join()
    update_thread.raise_if_failed()
    post_route(server, "finish_weight_update")
    post_route(server, "resume")


@requires_two_npus
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_hccl_transfer_restores_reset_weights(packed: bool):
    """The documented RL flow over HCCL: transfer W, baseline, reset,
    transfer the same W again, and expect an exact match on every rank."""
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = FixedRandomWeightSource(CHECKER_CASE, torch.device("npu", TRAINER_DEVICE_INDEX))

    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=live_update_serve_args(
            CHECKER_CASE,
            backend="hccl",
            port=port,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=_env_dict(INFERENCE_DEVICE_INDEX),
        auto_port=False,
    ) as server:
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

        master_address = get_ip()
        master_port = get_open_port()
        # The trainer is HCCL rank 0; the single inference worker is rank 1.
        world_size = 2
        # The server side blocks until the trainer connects, so kick its init
        # RPC off in a background thread while the trainer opens rank 0.
        init_thread = _BackgroundPost(
            server,
            "init_weight_transfer_engine",
            json={
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": world_size,
                }
            },
            timeout=INIT_TIMEOUT,
        )
        init_thread.start()
        model_update_group = HCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_thread.join()
        init_thread.raise_if_failed()

        _send_update(server, source, model_update_group, packed=packed)
        baseline = checksums(server)

        weight_checker(server, "reset").raise_for_status()
        comparison = weight_checker(server, "compare", baseline).json()
        assert comparison["match"] is False
        assert ranks(comparison["mismatches"]) == ranks(baseline)

        _send_update(server, source, model_update_group, packed=packed)
        assert weight_checker(server, "compare", baseline).json() == {
            "match": True,
            "mismatches": [],
        }
