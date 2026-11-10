# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Four-card E2E: E+P+D with MooncakeLayerwise KV transfer through the proxy."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import create_single_node_kv_pool_manager
from tests.e2e.conftest import RemoteEPDServer, wait_until_npu_memory_free
from tests.e2e.pull_request.two_card.epd.epd_utils import (
    assert_outputs_equal,
    chat_completions_url,
    clean_shared_storage,
    collect_baseline,
    encoder_server_args,
    image_payload,
    image_urls,
    post_chat,
    unique_ec_path,
    wait_until,
)

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_MODEL_LEN = 10000
MOONCAKE_SEGMENT_SIZE = "1GB"
LAYERWISE_PROXY_SCRIPT = (
    Path(__file__).parents[5] / "examples" / "epd_disaggregated" / "epd_load_balance_proxy_layerwise_server_example.py"
)


def _ec_transfer_config(role: str, shared_storage_path: str) -> str:
    return json.dumps(
        {
            "ec_connector": "ECExampleConnector",
            "ec_role": role,
            "ec_connector_extra_config": {"shared_storage_path": shared_storage_path},
        }
    )


def _kv_transfer_config(role: str, kv_port: int) -> str:
    return json.dumps(
        {
            "kv_connector": "MooncakeLayerwiseConnector",
            "kv_role": role,
            "kv_port": str(kv_port),
            "kv_connector_extra_config": {
                "use_ascend_direct": True,
                "prefill": {"dp_size": 1, "tp_size": 1},
                "decode": {"dp_size": 1, "tp_size": 1},
            },
        }
    )


def _encoder_args(encode_port: int, shared_storage_path: str) -> list[str]:
    return encoder_server_args(MODEL, encode_port, shared_storage_path) + [
        "--trust-remote-code",
        "--enable-request-id-headers",
    ]


def _prefill_args(prefill_port: int, shared_storage_path: str, kv_port: int) -> list[str]:
    return [
        "--port",
        str(prefill_port),
        "--model",
        MODEL,
        "--trust-remote-code",
        "--enable-request-id-headers",
        "--gpu-memory-utilization",
        "0.7",
        "--enforce-eager",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        json.dumps({"image": 2, "video": 0}),
        "--ec-transfer-config",
        _ec_transfer_config("ec_consumer", shared_storage_path),
        "--kv-transfer-config",
        _kv_transfer_config("kv_producer", kv_port),
    ]


def _decode_args(decode_port: int, kv_port: int) -> list[str]:
    return [
        "--port",
        str(decode_port),
        "--model",
        MODEL,
        "--trust-remote-code",
        "--enable-request-id-headers",
        "--gpu-memory-utilization",
        "0.7",
        "--enforce-eager",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        json.dumps({"image": 2, "video": 0}),
        "--kv-transfer-config",
        _kv_transfer_config("kv_consumer", kv_port),
    ]


def _proxy_healthcheck(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/healthcheck", timeout=2).status_code == 200
    except requests.RequestException:
        return False


@contextmanager
def _layerwise_proxy(
    encode_port: int,
    prefill_port: int,
    decode_port: int,
    env: dict[str, str],
) -> Iterator[int]:
    """Start the official MooncakeLayerwise proxy and always stop it."""
    if not LAYERWISE_PROXY_SCRIPT.exists():
        pytest.fail(f"layerwise proxy script not found: {LAYERWISE_PROXY_SCRIPT}")
    proxy_port = get_open_port()
    cmd = [
        sys.executable,
        str(LAYERWISE_PROXY_SCRIPT),
        "--host",
        "127.0.0.1",
        "--port",
        str(proxy_port),
        "--encoder-hosts",
        "127.0.0.1",
        "--encoder-ports",
        str(encode_port),
        "--prefiller-hosts",
        "127.0.0.1",
        "--prefiller-ports",
        str(prefill_port),
        "--decoder-hosts",
        "127.0.0.1",
        "--decoder-ports",
        str(decode_port),
    ]
    proc = subprocess.Popen(cmd, env={**os.environ, **env})
    try:
        wait_until(
            lambda: _proxy_healthcheck(proxy_port),
            timeout=120.0,
            description=f"layerwise proxy /healthcheck on port {proxy_port}",
        )
        yield proxy_port
    finally:
        if proc.poll() is None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def _mooncake_manager():
    config = MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "device_name": "",
            "global_segment_size": MOONCAKE_SEGMENT_SIZE,
            "local_buffer_size": MOONCAKE_SEGMENT_SIZE,
            "preferred_segment": False,
            "prefer_alloc_in_same_node": True,
            "enable_ssd_offload": False,
        },
        master_port=get_open_port(),
        metrics_port=get_open_port(),
    )
    return create_single_node_kv_pool_manager(config, "n6b_epd_1e1p1d")


@pytest.mark.e2e_model("Qwen/Qwen3-VL-8B-Instruct")
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="",
    parallel="",
    deploy="epd",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_1e1p1d_matches_baseline() -> None:
    payload = image_payload([image_urls()[0]], "Describe this image briefly.", max_tokens=16)
    payload["model"] = MODEL
    baseline = collect_baseline([payload], model=MODEL)[0]

    shared_storage_path = unique_ec_path("1e1p1d")
    clean_shared_storage(shared_storage_path)
    manager = _mooncake_manager()
    with manager:
        env = {
            **manager.server_envs,
            "PYTHONHASHSEED": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        }
        encode_port = get_open_port()
        prefill_port = get_open_port()
        decode_port = get_open_port()
        kv_port = get_open_port()
        server_args = [
            _encoder_args(encode_port, shared_storage_path),
            _prefill_args(prefill_port, shared_storage_path, kv_port),
            _decode_args(decode_port, kv_port),
        ]
        with (
            RemoteEPDServer(vllm_serve_args=server_args, env_dict=env),
            _layerwise_proxy(encode_port, prefill_port, decode_port, env) as proxy_port,
        ):
            output = post_chat(chat_completions_url(proxy_port), payload)

    assert_outputs_equal(output, baseline, context="E+P+D 1E1P1D")
