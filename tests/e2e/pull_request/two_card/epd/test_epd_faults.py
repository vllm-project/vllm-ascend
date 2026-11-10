# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Two-card EPD fault injection: encoder disconnect, restart and graceful shutdown."""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer, wait_until_npu_memory_free
from tests.e2e.pull_request.two_card.epd.epd_utils import (
    MODEL,
    REQUEST_TIMEOUT_SECONDS,
    EpdCluster,
    assert_outputs_equal,
    clean_shared_storage,
    collect_baseline,
    encoder_server_args,
    image_payload,
    image_urls,
    pd_server_args,
    post_chat,
    post_chat_request,
    reference_proxy_args,
    unique_ec_path,
)

# Proxy CLI value used by the disconnect case; a refused connection must fail
# long before this budget, so the test can assert a much smaller real bound.
ENCODE_STAGE_TIMEOUT_SECONDS = 30.0
# Measured upper bound for one image request after the E process was killed.
DISCONNECT_ERROR_MAX_SECONDS = 10.0
# Task requirement: every EPD process must exit within 60 s of SIGTERM.
GRACEFUL_SHUTDOWN_MAX_SECONDS = 60.0


def _metrics_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/metrics"


@contextmanager
def _fault_cluster(shared_storage_path: str) -> Iterator[EpdCluster]:
    """Start E + PD + reference proxy with an explicit encode-stage timeout.

    RemoteEPDServer is entered before DisaggEpdProxy on purpose: if the proxy
    process fails to start or its startup health probe fails, the proxy
    constructor raises and the already-entered E/PD context still runs its
    teardown. This is the proxy-startup-failure cleanup design handed over by
    the N6a review; no extra test function is added for it here.
    """
    clean_shared_storage(shared_storage_path)
    encode_port = get_open_port()
    decode_port = get_open_port()
    proxy_port = get_open_port()
    server_args = [
        encoder_server_args(MODEL, encode_port, shared_storage_path),
        pd_server_args(MODEL, decode_port, shared_storage_path),
    ]
    proxy_args = reference_proxy_args(encode_port, decode_port, proxy_port)
    proxy_args += ["--encode-timeout-seconds", str(ENCODE_STAGE_TIMEOUT_SECONDS)]
    with (
        RemoteEPDServer(vllm_serve_args=server_args) as servers,
        DisaggEpdProxy(proxy_args=proxy_args) as proxy,
    ):
        yield EpdCluster(
            servers=servers,
            proxy=proxy,
            encode_port=encode_port,
            decode_port=decode_port,
            proxy_port=proxy_port,
            shared_storage_path=shared_storage_path,
        )


def _terminate_instance(server: RemoteEPDServer, index: int) -> None:
    """Kill one EPD instance through the same tree-reaping path the teardown uses."""
    proc = server._proc_list[index]
    server._terminate_process_tree(proc)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


@pytest.mark.e2e_model("Qwen/Qwen2.5-VL-7B-Instruct")
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
def test_encoder_disconnect_returns_error() -> None:
    with _fault_cluster(unique_ec_path("fault_disconnect")) as cluster:
        _terminate_instance(cluster.servers, 0)
        request_body = image_payload([image_urls()[0]], "Describe this image.", max_tokens=8)

        started = time.monotonic()
        response = post_chat_request(cluster.proxy_chat_url, request_body, timeout=REQUEST_TIMEOUT_SECONDS)
        elapsed = time.monotonic() - started

        assert 500 <= response.status_code < 600, (
            f"dead encoder request returned {response.status_code}, expected a 5xx: {response.text}"
        )
        assert elapsed < DISCONNECT_ERROR_MAX_SECONDS, (
            f"dead encoder request took {elapsed:.3f}s, expected less than {DISCONNECT_ERROR_MAX_SECONDS}s"
        )

        metrics = requests.get(_metrics_url(cluster.proxy_port), timeout=30)
        assert metrics.status_code == 200, metrics.text
        assert 'epd_requests_total{stage="encode",status="connection"}' in metrics.text, metrics.text
        assert 'epd_retries_total{stage="encode"}' in metrics.text, metrics.text


@pytest.mark.e2e_model("Qwen/Qwen2.5-VL-7B-Instruct")
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
def test_encoder_restart_recovers() -> None:
    payload = image_payload([image_urls()[0]], "Describe this image.", max_tokens=16)
    baseline = collect_baseline([payload])[0]
    shared_storage_path = unique_ec_path("fault_restart")

    with _fault_cluster(shared_storage_path) as cluster:
        _terminate_instance(cluster.servers, 0)
        with RemoteEPDServer(
            vllm_serve_args=[encoder_server_args(MODEL, cluster.encode_port, shared_storage_path)]
        ) as replacement:
            recovered = post_chat(cluster.proxy_chat_url, payload)
        assert replacement.exit_codes == [0], f"restarted encoder exit codes: {replacement.exit_codes}"

    assert_outputs_equal(recovered, baseline, context="encoder restart")


@pytest.mark.e2e_model("Qwen/Qwen2.5-VL-7B-Instruct")
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
def test_graceful_shutdown_exit_codes() -> None:
    shared_storage_path = unique_ec_path("fault_shutdown")
    with _fault_cluster(shared_storage_path) as cluster:
        post_chat(cluster.proxy_chat_url, image_payload([image_urls()[0]], "Describe this image.", max_tokens=4))
        pids = [proc.pid for proc in cluster.servers._proc_list]
        pids += [proc.pid for proc in cluster.proxy._proc_list]

        started = time.monotonic()
        # SIGTERM the proxy first, then E, then PD (the order required by the node).
        cluster.proxy._terminate_server()
        cluster.servers._terminate_server()
        elapsed = time.monotonic() - started

        assert cluster.proxy.exit_codes == [0], f"proxy exit codes: {cluster.proxy.exit_codes}"
        assert cluster.servers.exit_codes == [0, 0], f"E/PD exit codes: {cluster.servers.exit_codes}"
        assert elapsed < GRACEFUL_SHUTDOWN_MAX_SECONDS, (
            f"graceful shutdown took {elapsed:.3f}s, expected less than {GRACEFUL_SHUTDOWN_MAX_SECONDS}s"
        )

    assert not Path(shared_storage_path).exists(), f"EC shared storage was not cleaned up: {shared_storage_path}"
    alive = [pid for pid in pids if psutil.pid_exists(pid)]
    assert not alive, f"EPD processes still alive after shutdown: {alive}"
