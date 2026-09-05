"""Real-model YuanRong E2E for the AscendStore MP backend."""

import contextlib
import json
import multiprocessing
import os
import uuid
from pathlib import Path

import pytest

from tests.e2e.pull_request.one_card.test_ascend_store_mp_ipc import _receive, _stop_process
from tests.e2e.pull_request.one_card.test_ascend_store_mp_model_smoke import (
    _HIT_LOG_PATTERN,
    _PROMPT,
    _build_llm,
    _generate_once,
    _model_path,
    _run_smoke_server,
    _wait_for_prefix_cache_reset,
)

_YUANRONG_CONFIG_ENV = "YR_CONFIG_PATH"


def _require_yuanrong() -> None:
    try:
        from yr.datasystem.hetero_client import HeteroClient  # type: ignore[import-not-found]  # noqa: F401
        from yr.datasystem.kv_client import SetParam  # type: ignore[import-not-found]  # noqa: F401
        from yr.datasystem.object_client import WriteMode  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("openyuanrong-datasystem is required for this E2E test") from exc


def _validate_yuanrong_config() -> None:
    config_path_value = os.getenv(_YUANRONG_CONFIG_ENV)
    if not config_path_value:
        pytest.skip(f"Set {_YUANRONG_CONFIG_ENV} to run the YuanRong E2E test")
    assert config_path_value is not None

    config_path = Path(config_path_value)
    if not config_path.is_file():
        pytest.fail(f"{_YUANRONG_CONFIG_ENV}={config_path_value!r} is not a readable file")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        pytest.fail(f"{_YUANRONG_CONFIG_ENV} must contain a JSON object")
    if (
        config.get("enable_remote_h2d", False)
        and config.get("remote_h2d_transport_backend", "HIXL") == "HIXL"
        and not config.get("enable_fabric_mem", False)
        and config.get("enable_dev_mem_pregister", False)
    ):
        pytest.fail("AscendStore MP requires enable_dev_mem_pregister=false because YuanRong cannot unregister it")


@pytest.mark.parametrize("use_layerwise", [False, True], ids=["bulk", "layerwise"])
def test_real_model_yuanrong_lookup_hit_and_retrieve(tmp_path, monkeypatch, use_layerwise: bool) -> None:
    import torch
    import torch_npu  # noqa: F401

    from tests.e2e.conftest import cleanup_dist_env_and_memory
    from vllm_ascend.ascend_config import clear_ascend_config

    model_path = _model_path()
    if model_path is None:
        pytest.skip("Set ASCEND_STORE_MP_SMOKE_MODEL to a local model path to run this smoke test")
    assert model_path is not None
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")
    _validate_yuanrong_config()
    if os.getenv("PYTHONHASHSEED") != "0":
        pytest.fail("PYTHONHASHSEED=0 must be set before pytest starts for YuanRong key consistency")
    _require_yuanrong()

    context = multiprocessing.get_context("spawn")
    endpoint_connection, endpoint_child_connection = context.Pipe()
    control_connection, control_child_connection = context.Pipe()
    server_log = tmp_path / "kv_cache_server.log"
    server = None
    llm = None
    npu_initialized = False
    failure: BaseException | None = None
    server_exitcode = None
    server_forced = False

    try:
        server = context.Process(
            target=_run_smoke_server,
            args=(endpoint_child_connection, control_child_connection, str(server_log)),
            name=f"kv-cache-yuanrong-{'layerwise' if use_layerwise else 'bulk'}-server",
        )
        server.start()
        endpoint_child_connection.close()
        control_child_connection.close()
        server_status, server_result = _receive(endpoint_connection, "KV cache server")
        if server_status != "ready":
            raise RuntimeError(f"KV cache server failed to start:\n{server_result}")

        torch.npu.set_device(0)
        npu_initialized = True
        llm = _build_llm(model_path, server_result, monkeypatch, use_layerwise=use_layerwise, backend="yuanrong")
        prompt = f"Cache namespace {uuid.uuid4().hex}. {_PROMPT}"
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
        prompt_token_count = len(llm.get_tokenizer().encode(prompt))
        assert prompt_token_count >= 2 * block_size, (
            f"Smoke prompt has {prompt_token_count} tokens, but at least {2 * block_size} are required"
        )

        first_output = _generate_once(llm, prompt)
        assert first_output.strip(), "First generation produced empty output"
        _wait_for_prefix_cache_reset(llm)
        second_output = _generate_once(llm, prompt)

        assert second_output == first_output, "Retrieved KV changed the greedy output"
        hits = [int(value) for value in _HIT_LOG_PATTERN.findall(server_log.read_text())]
        assert hits and max(hits) > 0, "No external YuanRong KV pool hit was recorded by the server"
    except BaseException as exc:
        failure = exc
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except BaseException as exc:
                if failure is None:
                    failure = exc
            llm = None
        if npu_initialized:
            clear_ascend_config()
            cleanup_dist_env_and_memory()
        endpoint_connection.close()
        endpoint_child_connection.close()
        control_child_connection.close()
        if server is not None:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                control_connection.send("stop")
        control_connection.close()
        server_exitcode, server_forced = _stop_process(server)

    if failure is not None:
        raise failure
    if server_forced:
        pytest.fail("KV cache server did not stop gracefully after the YuanRong smoke run")
    assert server_exitcode == 0
