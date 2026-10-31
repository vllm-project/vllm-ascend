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
graph-captured storage of the first update. General packed model lanes remain
skipped; the dedicated cross-chunk regression enables the validated four-layer
Qwen3.5 MoE case with an explicit service memory budget.

The trainer factory drives the handshake and both update rounds through
``HCCLTrainerWeightTransferEngine`` and ``HTTPVLLMWeightSyncClient``, including
the START -> broadcast -> FINISH transaction and propagation of RPC errors.
"""

import math
import os
from dataclasses import replace

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    MODEL_CASES,
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
PACKED_STRESS_UPDATE_TIMEOUT = 900
# Bound KV and graph-capture allocations for this functional regression.
PACKED_REGRESSION_SERVE_ARGS = (
    "--kv-cache-memory-bytes",
    str(2**30),
    "--max-num-seqs",
    "4",
    "--max-num-batched-tokens",
    "1024",
)
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
    _run_weight_transfer_transaction(case, packed=packed)


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="HCCL weight transfer e2e test requires at least 2 NPUs.")
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            replace(
                case,
                id=f"{case.id}-packed-cross-chunk",
                extra_server_args=(*case.extra_server_args, *PACKED_REGRESSION_SERVE_ARGS),
            ),
            id=case.id,
            marks=pytest.mark.e2e_model(case.model),
        )
        for case in MODEL_CASES
        if case.id == "qwen3.5-35b-a3b-moe-layout"
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_hccl_packed_cross_chunk_reload(case: WeightUpdateModelCase, num_buffers: int):
    """Force one nonempty tensor per chunk and repeat reload after generation.

    The threshold is a flush trigger, not an allocation limit for HCCL. A
    one-byte threshold splits layer weights across streams irrespective of
    model size, exercising retained inputs and slot rotation on actual NPU.
    """
    _run_weight_transfer_transaction(
        case,
        packed=True,
        num_buffers=num_buffers,
        buffer_size_bytes=1,
        update_rounds=4,
        update_timeout=PACKED_STRESS_UPDATE_TIMEOUT,
        packed_regression=True,
    )


def _run_weight_transfer_transaction(
    case: WeightUpdateModelCase,
    *,
    packed: bool,
    num_buffers: int = 2,
    buffer_size_bytes: int | None = None,
    update_rounds: int = 2,
    update_timeout: int = UPDATE_TIMEOUT,
    packed_regression: bool = False,
) -> None:
    if update_rounds < 2:
        raise ValueError("weight-transfer regression requires at least two update rounds")
    if case.skip_reason is not None:
        pytest.skip(f"{case.id}: {case.skip_reason}")
    if packed and not packed_regression:
        pytest.skip(f"{case.id}: {PACKED_LANE_SKIP_REASON}")

    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = FixedRandomWeightSource(case, torch.device("npu", TRAINER_DEVICE_INDEX))
    if buffer_size_bytes == 1:
        # All BF16 weights exceed the threshold, so each is a separate chunk.
        # Require enough chunks to rotate through every slot more than twice.
        chunk_count = sum(math.prod(meta.shape) * meta.dtype.itemsize > 1 for meta in source.metadata())
        assert chunk_count > 2 * num_buffers, (
            f"{case.id}: {chunk_count} chunks must rotate through {num_buffers} slots more than twice"
        )
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
                packed_buffer_size_bytes=(
                    packed_buffer_size_for(source) if buffer_size_bytes is None else buffer_size_bytes
                ),
                packed_num_buffers=num_buffers,
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root, timeout=update_timeout),
            source=source,
        )
        signatures = []
        try:
            for update_round in range(update_rounds):
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

    for reloaded_signature in signatures[1:]:
        assert_weight_update_matches_reference(
            dummy_signature,
            reference,
            signatures[0],
            reloaded_signature,
            case,
        )
