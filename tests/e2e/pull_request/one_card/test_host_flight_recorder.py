# SPDX-License-Identifier: Apache-2.0
"""NPU validation for opt-in MRV1/MRV2 DFX (not a throughput test)."""

import time
from pathlib import Path

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from vllm_ascend.dfx.inspect_trace import read_trace


@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("device_capture_interval", [0, 1])
@pytest.mark.parametrize("use_v2", [False, True])
def test_host_recorder_preserves_greedy_outputs_and_exports(
    monkeypatch, tmp_path, enforce_eager, async_scheduling, device_capture_interval, use_v2
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2)))
    prompts = ["The capital of France is", "A short explanation of gravity:"]
    params = SamplingParams(temperature=0, max_tokens=8)
    outputs = []
    for enabled in (False, True):
        with VllmRunner(
            "Qwen/Qwen3-0.6B",
            max_model_len=512,
            enforce_eager=enforce_eager,
            async_scheduling=async_scheduling,
            seed=0,
            additional_config={
                "dfx_config": {
                    "enabled": enabled,
                    "output_dir": str(tmp_path),
                    "run_id": "host-dfx-e2e",
                    "max_dumps": 100,
                    "device_capture_interval": device_capture_interval,
                    "token_patterns": [[outputs[0][0][0]]] if enabled else [],
                }
            },
        ) as runner:
            generated = runner.model.generate(prompts, params)
            outputs.append([list(result.outputs[0].token_ids) for result in generated])
            if enabled and device_capture_interval:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    stats = runner.model.collective_rpc("get_dfx_stats")
                    if all(item.get("device_completed", 0) > 0 for item in stats):
                        break
                    time.sleep(0.05)
                assert all(item.get("device_completed", 0) > 0 for item in stats)
                assert all(item.get("device_errors", 0) == 0 for item in stats)
            responses = runner.model.collective_rpc("dump_dfx_trace", args=("e2e-complete",))
            if not enabled:
                assert all(response["status"] == "disabled_or_unavailable" for response in responses)
                assert not list(tmp_path.glob("worker-*"))
                continue
            assert all(response["status"] == "queued" for response in responses)
            deadline = time.monotonic() + 10
            paths = [Path(response["path"]) for response in responses]
            while not all(path.exists() for path in paths) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert all(path.exists() for path in paths)
            for path in paths:
                trace = read_trace(path)
                assert trace["identity"]["run_id"] == "host-dfx-e2e"
                assert trace["stats"]["capture_errors"] == 0
                batches = [record for record in trace["records"] if record["kind"] == "host_batch_prepared"]
                assert batches
                assert all(set(record["violations"]) <= {"output_token_pattern"} for record in trace["records"])
                resolved = [record for record in trace["records"] if record["kind"] == "resolved_output"]
                assert resolved
                assert any("output_token_pattern" in record["violations"] for record in resolved)
                assert trace["stats"]["dumps_submitted"] >= 1  # Automatic detection preceded this manual dump.
                assert all(record["payload"]["actual_device_inputs_verified"] is False for record in batches)
                if use_v2:
                    assert trace["identity"]["runner"] == "mrv2"
                    assert all(record["payload"]["input_token_ids"] is None for record in batches)
                else:
                    assert all(len(record["payload"]["input_token_ids"]) > 0 for record in batches)
                if device_capture_interval:
                    device = [record for record in trace["records"] if record["kind"] == "device_snapshot"]
                    assert device and any(record["payload"] for record in device)
                    assert trace["coverage"]["extra_dummy_forward"] is False
    assert outputs[0] == outputs[1]
