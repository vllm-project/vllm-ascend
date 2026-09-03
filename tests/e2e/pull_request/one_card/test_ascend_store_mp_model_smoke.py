"""Real-model smoke tests for the AscendStore MP path.

Boots real vLLM engines with the MP connector and a real Mooncake backend in
bulk and layerwise modes. The first request stores KV through KVCacheServer,
the local prefix cache is then reset, and the second identical request must be
served from an external lookup hit with identical greedy output. A separate
test proves lookup degrades instead of failing when no server is available.
"""

import contextlib
import json
import logging
import multiprocessing
import os
import threading
import time
from multiprocessing.connection import Connection

import pytest
import regex as re

from tests.e2e.pull_request.one_card.test_ascend_store_mp_ipc import (
    _SERVER_URL,
    _receive,
    _request_server_stop,
    _stop_process,
    _wait_for_mooncake_master,
)

_MODEL_ENV = "ASCEND_STORE_MP_SMOKE_MODEL"
_HIT_LOG_PATTERN = re.compile(r"kvpool hit tokens: (\d+)")
_STORE_DRAIN_TIMEOUT_S = 30.0

# Repeat the context so it spans multiple whole Ascend hash blocks. The test
# also verifies this against the engine's actual block size before generation.
_PROMPT_CONTEXT = (
    "The history of computing machinery is a story of abstraction layers. "
    "Early machines were programmed by rewiring panels; then came stored programs, "
    "assemblers, compilers, and operating systems. Each layer hid the one below it "
    "and let programmers think in larger units. Modern inference stacks continue the "
    "same tradition: kernels hide accelerators, runtimes hide schedulers, and cache "
    "layers hide storage boundaries. A well-designed system lets each layer evolve "
    "without renegotiating its contracts."
)
_PROMPT = " ".join([_PROMPT_CONTEXT] * 4) + " Summarize the guiding principle in one sentence:"


def _model_path() -> str | None:
    path = os.getenv(_MODEL_ENV)
    if path is None:
        return None
    if not os.path.isdir(path):
        # Fail loudly instead of letting transformers misreport the path as an
        # invalid hub repo id: a configured path must be a visible directory.
        pytest.fail(f"{_MODEL_ENV}={path!r} is not a directory visible to this process")
    return path


def _run_smoke_server(endpoint_connection: Connection, control_connection: Connection, log_path: str) -> None:
    server = None
    control_thread = None
    file_handler = None
    vllm_logger = None
    try:
        from vllm.logger import logger as vllm_logger

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp import KVCacheServer

        # pool_scheduler reports lookup hits at DEBUG through the vllm logger;
        # capture it to a file so the test can assert the hit really happened.
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        vllm_logger.setLevel(logging.DEBUG)
        vllm_logger.addHandler(file_handler)
        vllm_logger.debug("KV cache smoke server log capture ready")

        server = KVCacheServer(_SERVER_URL, scheduler_threads=2, worker_threads=2)
        control_thread = threading.Thread(
            target=_request_server_stop,
            args=(server, control_connection),
            daemon=True,
            name="ascend-store-mp-smoke-stop",
        )
        control_thread.start()
        endpoint_connection.send(("ready", server.endpoint))
        endpoint_connection.close()
        server.run()
    finally:
        if server is not None and not server.close():
            server.abort()
        if control_thread is not None:
            control_thread.join(10.0)
        endpoint_connection.close()
        control_connection.close()
        if file_handler is not None:
            if vllm_logger is not None:
                vllm_logger.removeHandler(file_handler)
            file_handler.close()


def _build_llm(model_path: str, server_url: str, monkeypatch, use_layerwise: bool = False, backend: str = "mooncake"):
    monkeypatch.setenv("VLLM_ASCEND_STORE_MULTIPROCESS", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM
    from vllm.config import KVTransferConfig

    # vLLM forks an EngineCore subprocess by default, and a forked child
    # cannot re-initialize NPU once this process has touched it. Run the
    # engine in-process instead.
    gpu_memory_utilization = float(os.getenv("ASCEND_STORE_MP_SMOKE_GPU_MEM", "0.5"))

    kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "backend": backend,
            "kv_cache_server_url": server_url,
            "use_layerwise": use_layerwise,
        },
    )
    return LLM(
        model=model_path,
        kv_transfer_config=kv_transfer_config,
        max_model_len=1024,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )


def _generate_once(llm, prompt: str = _PROMPT) -> str:
    from vllm import SamplingParams

    outputs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=32))
    return outputs[0].outputs[0].text


def _wait_for_prefix_cache_reset(llm) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    deadline = time.monotonic() + _STORE_DRAIN_TIMEOUT_S
    dummy_params = SamplingParams(max_tokens=1)
    while not llm.reset_prefix_cache():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"External KV Store did not drain within {_STORE_DRAIN_TIMEOUT_S:.0f} seconds")
        # A completed async Store is released when the scheduler next polls
        # get_finished. This one-token request advances that scheduler step
        # without creating another external Store operation.
        llm.generate(
            [TokensPrompt(prompt_token_ids=[0])],
            dummy_params,
            use_tqdm=False,
        )


@pytest.mark.parametrize("use_layerwise", [False, True], ids=["bulk", "layerwise"])
def test_real_model_lookup_hit_and_retrieve(tmp_path, monkeypatch, use_layerwise: bool) -> None:
    import torch
    import torch_npu  # noqa: F401
    from vllm.utils.network_utils import get_open_port

    from tests.e2e.conftest import MooncakeLauncher, cleanup_dist_env_and_memory
    from vllm_ascend.ascend_config import clear_ascend_config

    model_path = _model_path()
    if model_path is None:
        pytest.skip(f"Set {_MODEL_ENV} to a local model path to run this smoke test")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")

    context = multiprocessing.get_context("spawn")
    endpoint_connection, endpoint_child_connection = context.Pipe()
    control_connection, control_child_connection = context.Pipe()
    server_log = tmp_path / "kv_cache_server.log"
    server = None
    llm = None
    failure: BaseException | None = None
    server_exitcode = None
    server_forced = False

    master_port = get_open_port()
    metrics_port = get_open_port()
    with MooncakeLauncher(master_port, metrics_port) as launcher:
        try:
            _wait_for_mooncake_master(launcher.process, master_port)
            config_path = tmp_path / "mooncake.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metadata_server": "P2PHANDSHAKE",
                        "protocol": "ascend",
                        "device_name": "",
                        "master_server_address": f"127.0.0.1:{master_port}",
                        "global_segment_size": "1GB",
                        "local_buffer_size": "64MB",
                        "preferred_segment": False,
                        "prefer_alloc_in_same_node": True,
                    }
                )
            )
            monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
            monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
            monkeypatch.delenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", raising=False)

            server = context.Process(
                target=_run_smoke_server,
                args=(endpoint_child_connection, control_child_connection, str(server_log)),
                name="kv-cache-smoke-server",
            )
            server.start()
            endpoint_child_connection.close()
            control_child_connection.close()
            server_status, server_result = _receive(endpoint_connection, "KV cache server")
            if server_status != "ready":
                raise RuntimeError(f"KV cache server failed to start:\n{server_result}")

            torch.npu.set_device(0)
            llm = _build_llm(model_path, server_result, monkeypatch, use_layerwise)
            block_size = llm.llm_engine.vllm_config.cache_config.block_size
            prompt_token_count = len(llm.get_tokenizer().encode(_PROMPT))
            assert prompt_token_count >= 2 * block_size, (
                f"Smoke prompt has {prompt_token_count} tokens, but at least {2 * block_size} are required"
            )
            first_output = _generate_once(llm)
            assert first_output.strip(), "First generation produced empty output"

            # Drop the engine's own prefix cache so the second request must be
            # served by the external pool instead of local HBM blocks.
            _wait_for_prefix_cache_reset(llm)
            second_output = _generate_once(llm)

            assert second_output == first_output, "Retrieved KV changed the greedy output"

            hits = [int(value) for value in _HIT_LOG_PATTERN.findall(server_log.read_text())]
            assert hits and max(hits) > 0, "No external KV pool hit was recorded by the server"
        except BaseException as exc:
            failure = exc
        finally:
            if llm is not None:
                try:
                    # Worker connectors are process-global in vLLM and are
                    # cleared only by EngineCore shutdown. Keep the server
                    # alive until both connector roles have unregistered.
                    llm.llm_engine.engine_core.shutdown()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                llm = None
                clear_ascend_config()
                cleanup_dist_env_and_memory()
            endpoint_connection.close()
            endpoint_child_connection.close()
            control_child_connection.close()
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                control_connection.send("stop")
            control_connection.close()
            server_exitcode, server_forced = _stop_process(server)

    if failure is not None:
        raise failure
    if server_forced:
        pytest.fail("KV cache server did not stop gracefully after the smoke run")
    assert server_exitcode == 0


def test_real_model_degrades_when_server_unavailable(monkeypatch) -> None:
    import torch
    import torch_npu  # noqa: F401

    from tests.e2e.conftest import cleanup_dist_env_and_memory
    from vllm_ascend.ascend_config import clear_ascend_config

    model_path = _model_path()
    if model_path is None:
        pytest.skip(f"Set {_MODEL_ENV} to a local model path to run this smoke test")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")

    # Port 1 never accepts: registration and lookup must degrade to misses
    # instead of failing engine startup or generation.
    torch.npu.set_device(0)
    llm = _build_llm(model_path, "tcp://127.0.0.1:1", monkeypatch)
    try:
        output = _generate_once(llm)
        assert output.strip(), "Generation produced empty output while degraded"
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        clear_ascend_config()
        cleanup_dist_env_and_memory()
