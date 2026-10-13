# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Two-NPU Weight Checker e2e tests: multi-rank coverage and the HCCL round trip.

The first lane starts TP=2 and DP=2 servers and asserts that the checksum keys
cover exactly the expected rank coordinates, that a rank dropped from the
baseline is reported exactly, and that reset reaches every rank (a reset that
skipped an engine would leave its keys matching). The second lane drives the
cross-NPU HCCL deployment's main path: transfer W, baseline, reset, transfer
the same W again, and expect an exact match.

The HCCL trainer handshake is mirrored from test_hccl_weight_transfer.py.
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
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    live_update_serve_args,
    wait_for_free_device_memory,
)

CONTROL_TIMEOUT = 60
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
GPU_MEMORY_UTILIZATION = 0.45
MODEL = "Qwen/Qwen3-0.6B"
HF_OVERRIDES = {"num_hidden_layers": 4}
# Physical cards; see the HCCL weight-transfer tests.
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
TRAINER_DEVICE_INDEX = INFERENCE_DEVICE_INDEX + 1

CHECKER_CASE = WeightUpdateModelCase(
    id="qwen3-0.6b-checker",
    model=MODEL,
    hf_overrides=HF_OVERRIDES,
)

# The server endpoint ships with the paired vLLM (PR #51350); skip the whole
# module while the CI-pinned vLLM predates it.
pytest.importorskip(
    "vllm.model_executor.model_loader.weight_checksum",
    reason="Weight Checker requires a paired vLLM with PR #51350 (>= v0.31.1rc0).",
)

pytestmark = pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="Weight Checker multi-rank e2e tests require at least 2 NPUs.",
)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


def weight_checker(server: RemoteOpenAIServer, action: str, baseline=None) -> requests.Response:
    body = {"action": action}
    if baseline is not None:
        body["baseline"] = baseline
    return requests.post(server.url_for("weight_checker"), json=body, timeout=UPDATE_TIMEOUT)


def checksums(server: RemoteOpenAIServer) -> dict[str, str]:
    response = weight_checker(server, "checksum")
    response.raise_for_status()
    return response.json()["checksums"]


def ranks(keys) -> set[str]:
    return {key.rsplit(":", 1)[0] for key in keys}


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
    serve_args = [
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
        *parallel_args,
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": f"{INFERENCE_DEVICE_INDEX},{TRAINER_DEVICE_INDEX}",
    }
    with RemoteOpenAIServer(
        MODEL,
        vllm_serve_args=serve_args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
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
            _post(server, "collective_rpc", json={"method": "reload_weights"}, timeout=UPDATE_TIMEOUT)
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
            _post(self._server, self._route, json=self._json, timeout=self._timeout)
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
    _post(server, "pause")
    _post(server, "start_weight_update")

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
        timeout=UPDATE_TIMEOUT,
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
    _post(server, "finish_weight_update")
    _post(server, "resume")


@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_hccl_transfer_restores_reset_weights(packed: bool):
    """The documented RL flow over HCCL: transfer W, baseline, reset,
    transfer the same W again, and expect an exact match on every rank."""
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = FixedRandomWeightSource(CHECKER_CASE, torch.device("npu", TRAINER_DEVICE_INDEX))
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }

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
        env_dict=env_dict,
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
