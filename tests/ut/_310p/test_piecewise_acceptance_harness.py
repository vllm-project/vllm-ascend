import json

import pytest

from tools.run_310p_dflash_piecewise_acceptance import (
    MODEL_PRESETS,
    PairComparisonError,
    build_server_command,
    collect_graph_evidence,
    compare_pair,
    parse_devices,
    require_piecewise_graphs,
)


def test_server_commands_keep_eager_and_piecewise_flags_identical_except_mode() -> None:
    eager = build_server_command(MODEL_PRESETS["9b"], "eager", 2, 8112)
    piecewise = build_server_command(MODEL_PRESETS["9b"], "piecewise", 2, 8112)

    assert eager[:-1] == piecewise[:-1]
    assert eager[-1] == '{"cudagraph_mode":"NONE"}'
    assert piecewise[-1] == ('{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[64,32]}')
    assert "--enable-prefix-caching" in piecewise
    assert "--enable-chunked-prefill" in piecewise
    assert "--async-scheduling" in piecewise
    assert any('"num_speculative_tokens":15' in argument for argument in piecewise)


def test_graph_evidence_requires_target_and_draft_on_every_rank() -> None:
    records = []
    for rank in (0, 1):
        for event in ("capture", "replay"):
            for component in ("target", "draft"):
                records.append(
                    f"event={event} component={component} rank={rank} "
                    "mode=PIECEWISE descriptor=BatchDescriptor(num_tokens=64)"
                )
    records.append("cudagraph_capture_sizes': [32, 64]")
    evidence = collect_graph_evidence("\n".join(records), tp=2)

    require_piecewise_graphs(evidence)
    assert evidence.contract_errors == 0


def test_graph_evidence_rejects_missing_rank_replay() -> None:
    evidence = collect_graph_evidence(
        "event=capture component=target rank=0\n"
        "event=capture component=draft rank=0\n"
        "event=replay component=target rank=0\n"
        "cudagraph_capture_sizes': [32, 64]",
        tp=1,
    )

    with pytest.raises(RuntimeError, match="missing target/draft"):
        require_piecewise_graphs(evidence)


def test_pair_comparison_requires_exact_tokens_and_thresholds(tmp_path) -> None:
    eager = {
        "generated_token_ids": [[1, 2], [3, 4]],
        "benchmark": {
            "output_throughput": 100.0,
            "spec_decode_acceptance_length": 6.0,
            "spec_decode_acceptance_rate": 40.0,
        },
    }
    eager_path = tmp_path / "eager.json"
    eager_path.write_text(json.dumps(eager), encoding="utf-8")
    piecewise = {
        "generated_token_ids": [[1, 2], [3, 4]],
        "benchmark": {
            "output_throughput": 90.0,
            "spec_decode_acceptance_length": 5.7,
            "spec_decode_acceptance_rate": 37.0,
        },
    }

    comparison = compare_pair(piecewise, eager_path)

    assert comparison["token_mismatch_indices"] == []
    assert comparison["throughput_delta"] == pytest.approx(-0.10)
    assert parse_devices("6,7", 2) == (6, 7)


def test_pair_comparison_reports_first_differing_token(tmp_path) -> None:
    eager = {
        "generated_token_ids": [[1, 2, 3]],
        "benchmark": {
            "output_throughput": 100.0,
            "spec_decode_acceptance_length": 6.0,
            "spec_decode_acceptance_rate": 40.0,
        },
    }
    eager_path = tmp_path / "eager.json"
    eager_path.write_text(json.dumps(eager), encoding="utf-8")
    piecewise = {
        "generated_token_ids": [[1, 9, 3]],
        "benchmark": eager["benchmark"],
    }

    with pytest.raises(PairComparisonError, match="request 0 token 1") as raised:
        compare_pair(piecewise, eager_path)

    assert raised.value.comparison["token_mismatch_details"] == [
        {
            "request_index": 0,
            "first_differing_token_index": 1,
            "eager_token": 2,
            "piecewise_token": 9,
            "eager_length": 3,
            "piecewise_length": 3,
        }
    ]
