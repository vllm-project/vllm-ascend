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
graph-captured storage of the first update. Both packed modes exercise the same
transaction.

The trainer side keeps this backend's engine handshake: the trainer opens the
rank-0 HCCL endpoint with ``HCCLWeightTransferEngine.trainer_init`` and the
server-side init/update RPCs are driven explicitly, so the transaction under
test is START -> broadcast -> FINISH.
"""

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
    assert_weight_update_matches_reference,
    generation_signature,
    live_update_serve_args,
    pytest_model_cases,
    reference_signature,
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
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60
GPU_MEMORY_UTILIZATION = 0.75


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


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


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="HCCL weight transfer e2e test requires at least 2 NPUs.",
)
@pytest.mark.parametrize("case", pytest_model_cases())
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    if (skip_reason := case.skip_reasons.get("hccl")) is not None:
        pytest.skip(f"{case.id}: {skip_reason}")

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

        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

        master_address = get_ip()
        master_port = get_open_port()
        # The trainer is HCCL rank 0; the single inference worker is rank 1, so
        # the group is the workers plus the sender.
        world_size = INFERENCE_WORLD_SIZE + 1

        # The server side blocks until the trainer connects, so kick its init RPC
        # off in a background thread while the trainer opens rank 0.
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

        signatures = []
        for update_round in range(2):
            _log(
                f"{case.id}: sending {('packed' if packed else 'unpacked')} "
                f"fixed-random update round={update_round + 1}"
            )
            # START -> broadcast -> FINISH against the inference server.
            _send_update(server, source, model_update_group, packed=packed)
            signatures.append(generation_signature(client, case.model))

    updated_signature, reloaded_signature = signatures
    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        updated_signature,
        reloaded_signature,
        case,
    )
