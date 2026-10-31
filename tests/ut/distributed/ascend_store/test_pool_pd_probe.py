# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import httpx
import pytest

from tests.e2e.common.kv_pool.pool_pd import D_COUNTERS, P_COUNTERS, compare_case, observe_transfers, read_metrics


@pytest.mark.parametrize("outcome", ["success", "different_tokens", "fallback", "failed_restore"])
@pytest.mark.parametrize("chat", [False, True])
def test_probe_requires_real_transfer_counters_and_exact_tokens(outcome, chat):
    completed = False
    case = (
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,test-image"}},
                    ],
                }
            ]
        }
        if chat
        else {"prompt": [5, 6]}
    )

    def handle(request):
        nonlocal completed
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload["return_token_ids"] is True
            assert payload["temperature"] == 0
            assert "kv_transfer_params" not in payload
            assert request.url.path == ("/v1/chat/completions" if chat else "/v1/completions")
            for key, value in case.items():
                assert payload[key] == value
            candidate = request.url.host == "candidate"
            completed |= candidate
            tokens = [2, 3] if candidate and outcome == "different_tokens" else [1, 2]
            return httpx.Response(200, json={"choices": [{"token_ids": tokens, "finish_reason": "length"}]})
        counters = P_COUNTERS if request.url.host == "p" else D_COUNTERS
        values = dict.fromkeys(counters, 0)
        if completed:
            if request.url.host == "p":
                values.update(save_committed=1, save_bytes=16)
            elif outcome == "failed_restore":
                values["restore_failed"] = 1
            elif outcome != "fallback":
                values.update(restore_completed=1, restore_bytes=16)
        return httpx.Response(200, text="\n".join(f"vllm:pool_pd_{key}_total {value}" for key, value in values.items()))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        arguments = dict(
            model="test",
            baseline_url="http://baseline",
            candidate_url="http://candidate",
            prefill_metrics=["http://p/metrics"],
            decode_metrics=["http://d/metrics"],
            metrics_timeout=0,
        )
        if outcome in ("fallback", "failed_restore"):
            with pytest.raises((TimeoutError, ValueError), match="fallback|failure"):
                compare_case(client, case, **arguments)
        else:
            result = compare_case(client, case, **arguments)
            assert result["passed"] == (outcome == "success")
            assert result["counter_deltas"]["restore_bytes"] == 16


@pytest.mark.parametrize("case", [{}, {"prompt": "hi", "messages": []}, {"messages": [], "kv_transfer_params": {}}])
def test_probe_rejects_ambiguous_input_or_manufactured_handoff_before_sending(case):
    with pytest.raises(ValueError, match="either prompt or messages"):
        compare_case(
            None,
            case,
            model="test",
            baseline_url="",
            candidate_url="",
            prefill_metrics=["p"],
            decode_metrics=["d"],
            metrics_timeout=0,
        )


@pytest.mark.parametrize("body", ["", "vllm:pool_pd_restore_bytes_total NaN", "vllm:pool_pd_restore_bytes_total -1"])
def test_probe_rejects_missing_or_invalid_metrics(body):
    with (
        httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))) as client,
        pytest.raises(ValueError, match="Missing|Invalid"),
    ):
        read_metrics(client, ["http://d/metrics"], ("restore_bytes",))


@pytest.mark.parametrize("outcome", ["success", "fallback", "failed_restore", "benchmark_error"])
def test_benchmark_observer_keeps_evidence_and_first_failure(tmp_path, outcome):
    ran = False
    seen = []

    def handle(request):
        seen.append(request.url.host)
        counters = P_COUNTERS if request.url.host.startswith("p") else D_COUNTERS
        values = dict.fromkeys(counters, 0)
        if ran:
            # Traffic on the second DP engines must also count.
            if request.url.host == "p1":
                values.update(save_committed=1, save_bytes=16)
            elif request.url.host == "d1" and outcome == "failed_restore":
                values["restore_failed"] = 1
            elif request.url.host == "d1" and outcome == "success":
                values.update(restore_completed=1, restore_bytes=16)
            elif outcome == "benchmark_error":
                raise AssertionError("Do not mask the benchmark failure with a subsequent metrics failure")
        return httpx.Response(200, text="\n".join(f"vllm:pool_pd_{key}_total {value}" for key, value in values.items()))

    output = tmp_path / "transfer.json"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:

        def run():
            nonlocal ran
            with observe_transfers(
                client,
                ["http://p0/metrics", "http://p1/metrics"],
                ["http://d0/metrics", "http://d1/metrics"],
                output,
                metrics_timeout=0,
            ):
                ran = True
                if outcome == "benchmark_error":
                    raise RuntimeError("benchmark first failure")

        if outcome == "success":
            run()
        else:
            error, message = {
                "fallback": (TimeoutError, "fallback"),
                "failed_restore": (ValueError, "failure"),
                "benchmark_error": (RuntimeError, "benchmark first failure"),
            }[outcome]
            with pytest.raises(error, match=message):
                run()
    evidence = json.loads(output.read_text())
    assert evidence["status"] == ("passed" if outcome == "success" else "failed")
    assert set(seen) == {"p0", "p1", "d0", "d1"}
    if outcome == "benchmark_error":
        assert len(seen) == 4
        assert "benchmark first failure" in evidence["error"]
    elif outcome == "success":
        assert evidence["counter_deltas"]["restore_bytes"] == 16
    with pytest.raises(FileExistsError), observe_transfers(None, ["p"], ["d"], output):
        pytest.fail("Existing evidence must be reserved before executing the benchmark")


def test_layerwise_pool_glm_case_uses_original_proxy_and_independent_connector():
    from tests.e2e.common.multi_node.external_dp.external_dp_config import ExternalDPConfigLoader, RankResolver
    from tests.e2e.common.multi_node.external_dp.runtime import build_all_server_commands

    root = Path(__file__).resolve().parents[4]
    path = root / "tests/e2e/cases/features/kv_pool/GLM-5.2-W8A8C8-A3-LayerwisePool-PD.yaml"
    config = ExternalDPConfigLoader.from_yaml(str(path), cluster_ips=[f"192.0.2.{i}" for i in range(1, 5)])
    ranks = RankResolver(config).resolve()
    commands = build_all_server_commands(config, ranks)
    assert config.routing.proxy_script == "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
    assert "pool_pd_transfer" in config.test_content
    assert len(ranks) == len(commands) == 4
    for rank, command in zip(ranks, commands):

        def flag(name, command=command):
            return command.cmd[command.cmd.index(name) + 1]

        is_p = rank.role == "prefiller"
        transfer = json.loads(flag("--kv-transfer-config"))
        assert transfer["kv_connector"] == "LayerwisePoolConnector"
        assert transfer["kv_role"] == ("kv_producer" if is_p else "kv_consumer")
        assert transfer["kv_connector_extra_config"]["backend"] == "memcache"
        assert "connectors" not in transfer["kv_connector_extra_config"]
        assert json.loads(flag("--speculative-config"))["num_speculative_tokens"] == (1 if is_p else 5)
        assert (rank.tp_size, rank.cp_size) == ((1, 16) if is_p else (16, 1))
        assert command.env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        assert not rank.is_headless
