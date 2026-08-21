import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_310p_dflash_full_decode_only_acceptance import (
    C10_CAPTURE_SIZES,
    CAPTURE_SIZES,
    GRAPH_PROOF_OUTPUT_LENGTHS,
    MODEL_PRESETS,
    PairComparisonError,
    build_server_command,
    capture_sizes_for_scenario,
    collect_graph_evidence,
    compare_pair,
    require_full_decode_graphs,
)


def _valid_graph_log(tp: int) -> str:
    records = [
        "[310p-dflash-full-decode-only/dispatch] "
        "state=EXPECTED_NONE_PREFILL expected=NONE selected=NONE reason=prefill",
        "[310p-dflash-full-decode-only/dispatch] "
        "state=FULL_ELIGIBLE_UNIFORM_DECODE expected=FULL selected=FULL "
        "reason=uniform_dflash_decode",
    ]
    for rank in range(tp):
        for descriptor in CAPTURE_SIZES:
            for component in ("target", "draft"):
                records.append(
                    "[310p-dflash-full-decode-only/manifest] event=record "
                    f"component={component} rank={rank} mode=FULL "
                    f"descriptor=BatchDescriptor(num_tokens={descriptor}, "
                    "num_reqs=1, uniform=True) capture_count=1 "
                    "warmup_replay_count=1"
                )
                records.append(
                    "[310p-dflash-graph] event=replay "
                    f"component={component} rank={rank} mode=FULL "
                    f"descriptor=BatchDescriptor(num_tokens={descriptor}, "
                    "num_reqs=1, uniform=True) capture_count=1 "
                    "replay_count=1 contract=stable"
                )
                records.append(
                    "[310p-dflash-graph] event=native-graph-dump "
                    f"component={component} rank={rank} "
                    f"descriptor=BatchDescriptor(num_tokens={descriptor}, "
                    "num_reqs=1, uniform=True) path=/evidence/graph.json"
                )
        records.append(
            "[310p-dflash-full-decode-only/manifest] event=complete "
            f"rank={rank} entries=4 components=target,draft descriptors=[16, 160]"
        )
    return "\n".join(records)


def _summary(
    *,
    request_throughput: float = 10.0,
    output_throughput: float = 100.0,
    acceptance_length: float = 6.0,
    acceptance_rate: float = 40.0,
) -> dict:
    return {
        "generated_token_ids": [[1, 2], [3, 4]],
        "benchmark": {
            "request_throughput": request_throughput,
            "output_throughput": output_throughput,
            "spec_decode_acceptance_length": acceptance_length,
            "spec_decode_acceptance_rate": acceptance_rate,
        },
    }


def test_server_commands_keep_frozen_flags_and_zero_internal_warmup() -> None:
    eager = build_server_command(MODEL_PRESETS["35b"], "eager", 2, 8112)
    full_decode = build_server_command(MODEL_PRESETS["35b"], "full_decode_only", 2, 8112)

    assert eager[:-1] == full_decode[:-1]
    assert eager[-1] == '{"cudagraph_mode":"NONE"}'
    assert json.loads(full_decode[-1]) == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [160, 16],
        "cudagraph_num_of_warmups": 0,
    }
    assert any('"num_speculative_tokens":15' in argument for argument in full_decode)


@pytest.mark.parametrize(
    ("model", "tp"),
    (("4b", 1), ("4b", 2), ("35b", 2), ("35b", 4)),
)
def test_final_matrix_uses_frozen_capture_sizes(model: str, tp: int) -> None:
    capture_sizes = capture_sizes_for_scenario(model, tp, 10)
    full_decode = build_server_command(
        MODEL_PRESETS[model],
        "full_decode_only",
        tp,
        8111,
        capture_sizes,
    )

    assert capture_sizes == C10_CAPTURE_SIZES == (160, 16)
    assert json.loads(full_decode[-1])["cudagraph_capture_sizes"] == [160, 16]


def test_rejects_scenarios_outside_final_matrix() -> None:
    with pytest.raises(ValueError, match="outside the frozen acceptance matrix"):
        capture_sizes_for_scenario("35b", 1, 10)


def test_four_graph_proof_requests_force_64_to_32_to_16_contraction() -> None:
    assert GRAPH_PROOF_OUTPUT_LENGTHS == (128, 80, 16, 16)
    assert len(GRAPH_PROOF_OUTPUT_LENGTHS) == 4
    assert len(set(GRAPH_PROOF_OUTPUT_LENGTHS)) == 3


def test_direct_script_entry_does_not_shadow_python_standard_library() -> None:
    repo = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, str(repo / "tools/run_310p_dflash_full_decode_only_acceptance.py"), "--help"],
        cwd=repo,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_graph_evidence_requires_real_target_and_draft_graphs_on_every_rank() -> None:
    evidence = collect_graph_evidence(_valid_graph_log(tp=2), tp=2)

    require_full_decode_graphs(evidence)
    assert evidence.expected_none_dispatches == 1
    assert evidence.runtime_full_dispatches == 1
    assert evidence.safety_errors == ()


@pytest.mark.parametrize(
    "removed_line,error",
    [
        ("event=record component=draft rank=1 mode=FULL descriptor=BatchDescriptor(num_tokens=160", "manifest"),
        ("event=replay component=target rank=0 mode=FULL descriptor=BatchDescriptor(num_tokens=160", "replay"),
        ("event=native-graph-dump component=target rank=0 descriptor=BatchDescriptor(num_tokens=16", "native"),
        ("event=complete rank=1", "complete"),
        ("expected=NONE selected=NONE reason=prefill", "NONE"),
        ("expected=FULL selected=FULL reason=uniform_dflash_decode", "FULL"),
    ],
)
def test_graph_evidence_rejects_missing_or_false_positive_proof(removed_line: str, error: str) -> None:
    log_text = "\n".join(line for line in _valid_graph_log(tp=2).splitlines() if removed_line not in line)

    with pytest.raises(RuntimeError, match=error):
        require_full_decode_graphs(collect_graph_evidence(log_text, tp=2))


@pytest.mark.parametrize(
    "marker",
    [
        "GraphInputContractError: address changed",
        "graph input contract failed for target",
        "eligible uniform decode selected NONE",
        "eligible uniform decode has no validated FULL descriptor",
    ],
)
def test_graph_evidence_rejects_safety_failures(marker: str) -> None:
    evidence = collect_graph_evidence(_valid_graph_log(tp=1) + "\n" + marker, tp=1)

    with pytest.raises(RuntimeError, match="safety"):
        require_full_decode_graphs(evidence)


def test_graph_evidence_ignores_only_traceback_after_controlled_sigterm() -> None:
    log_text = (
        _valid_graph_log(tp=1)
        + "\n[shutdown] EngineCore: trigger received signal=SIGTERM"
        + "\nTraceback (most recent call last): controlled shutdown"
    )

    evidence = collect_graph_evidence(log_text, tp=1)

    require_full_decode_graphs(evidence)
    assert evidence.safety_errors == ()


def test_graph_evidence_rejects_traceback_before_controlled_sigterm() -> None:
    log_text = (
        _valid_graph_log(tp=1)
        + "\nTraceback (most recent call last): runtime failure"
        + "\n[shutdown] EngineCore: trigger received signal=SIGTERM"
    )

    with pytest.raises(RuntimeError, match="safety"):
        require_full_decode_graphs(collect_graph_evidence(log_text, tp=1))


def test_pair_comparison_enforces_all_frozen_thresholds(tmp_path) -> None:
    eager_path = tmp_path / "eager.json"
    eager_path.write_text(json.dumps(_summary()), encoding="utf-8")
    current = _summary(
        request_throughput=8.5,
        output_throughput=85.0,
        acceptance_length=5.4,
        acceptance_rate=35.0,
    )

    comparison = compare_pair(current, eager_path)

    assert comparison["request_throughput_ratio"] == pytest.approx(0.85)
    assert comparison["output_throughput_ratio"] == pytest.approx(0.85)
    assert comparison["acceptance_length_ratio"] == pytest.approx(0.9)
    assert comparison["acceptance_rate_delta_pp"] == pytest.approx(-5.0)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"request_throughput": 8.49}, "request throughput"),
        ({"output_throughput": 84.9}, "output throughput"),
        ({"acceptance_length": 4.99}, "accepted length"),
        ({"acceptance_length": 5.39}, "accepted length ratio"),
        ({"acceptance_rate": 34.9}, "acceptance rate"),
    ],
)
def test_pair_comparison_rejects_each_failed_gate(tmp_path, overrides: dict, error: str) -> None:
    eager_path = tmp_path / "eager.json"
    eager_path.write_text(json.dumps(_summary()), encoding="utf-8")

    with pytest.raises(PairComparisonError, match=error):
        compare_pair(_summary(**overrides), eager_path)


def test_pair_comparison_reports_exact_token_mismatch(tmp_path) -> None:
    eager_path = tmp_path / "eager.json"
    eager_path.write_text(json.dumps(_summary()), encoding="utf-8")
    current = _summary()
    current["generated_token_ids"][1][0] = 9

    with pytest.raises(PairComparisonError, match="request 1 token 0"):
        compare_pair(current, eager_path)
