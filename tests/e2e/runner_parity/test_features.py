# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json

from compare import compare_precision, read_records
from evidence_checks import assert_control_runtime_and_ownership


def feature_arm(root, runner):
    return root / runner / "phase-diagnostic"


def test_glm51_mtp_paired_precision(feature_root, upstream_utils):
    paths = [feature_arm(feature_root, arm) / "controlled-precision.jsonl" for arm in ("v1", "v2")]
    for path in paths:
        assert all(row["request"]["speculative_tokens"] == 3 for row in read_records(path).values())
    report = compare_precision(*paths, upstream_utils)
    assert report["case_count"] == report["passed"] == 32
    assert report["failed"] == 0


def test_glm51_mtp_runtime_and_execution(feature_root):
    assert_control_runtime_and_ownership(feature_root, extra_source_keys=("parser_source_sha256",))
    reference = None
    for arm in ("v1", "v2"):
        root = feature_arm(feature_root, arm)
        options = json.loads((root / "effective-options.json").read_text())
        if reference is None:
            reference = options
        assert options == reference
        assert options["enforce_eager"] and options["compilation_config"]["cudagraph_mode"] == "NONE"
        assert options["speculative_config"]["method"] == "deepseek_mtp"
        assert options["speculative_config"]["num_speculative_tokens"] == 3
        assert options["tensor_parallel_size"] == 8 and options["enable_expert_parallel"]
        assert options["quantization"] == "ascend" and options["kv_cache_dtype"] == "fp8"
        workers = json.loads((root / "worker-feature-fingerprints.json").read_text())
        assert {worker["rank"] for worker in workers} == set(range(8))
        for worker in workers:
            assert worker["runner"] and worker["draft"]
            assert any(
                name.endswith("AscendFusedMoEMethod") and count > 0 for name, count in worker["quant_methods"].items()
            )
        stats = json.loads((root / "speculative-evidence.json").read_text())
        assert stats["vllm:spec_decode_num_drafts"] > 0
        assert 0 < stats["vllm:spec_decode_num_accepted_tokens"] <= stats["vllm:spec_decode_num_draft_tokens"]


def test_glm51_stop_cancel_and_context_boundaries(feature_root):
    names = {
        "max-output-one",
        "checkpoint-eos-stop",
        "stop-token",
        "cancel-and-recover",
        "max-context-accepted",
        "over-context-rejected",
        "recovery-after-invalid",
    }
    for arm in ("v1", "v2"):
        rows = json.loads((feature_arm(feature_root, arm) / "feature-checks.json").read_text())
        assert len(rows) == 7 and {row["name"] for row in rows} == names
        assert all(row["passed"] for row in rows), rows
        checks = {row["name"]: row["result"] for row in rows}
        assert len(checks["max-output-one"]["token_ids"]) == 1
        eos = checks["checkpoint-eos-stop"]
        assert eos["finish_reason"] == "stop" and eos["prompt_token_ids"]
        # The primary tokenizer EOS may be represented by a null stop_reason;
        # additional checkpoint stop IDs carry their explicit token ID.
        assert eos["stop_reason"] is None or eos["stop_reason"] in eos["configured_stop_ids"]
        stop = checks["stop-token"]
        assert stop["finish_reason"] == "stop" and stop["stop_reason"] == stop["stop_id"]
        assert len(stop["token_ids"]) == 1
        assert checks["cancel-and-recover"]["cancelled_internal_id"]
        assert len(checks["cancel-and-recover"]["recovery"]["token_ids"]) == 4
        accepted = checks["max-context-accepted"]
        assert accepted["prompt_tokens"] == 40956 and len(accepted["token_ids"]) == 4
        rejected = checks["over-context-rejected"]
        assert rejected["prompt_tokens"] == 40961 and rejected["error_type"] == "VLLMValidationError"
        assert "40960" in rejected["error"] and "40961" in rejected["error"]
        assert len(checks["recovery-after-invalid"]["token_ids"]) == 4


def test_glm51_actual_model_reasoning_and_tools(feature_root):
    for arm in ("v1", "v2"):
        rows = json.loads((feature_arm(feature_root, arm) / "model-parser-checks.json").read_text())
        assert len(rows) == 2 and {row["name"] for row in rows} == {"reasoning", "tool"}
        assert all(row["passed"] and row["finish_reason"] == "stop" for row in rows)
        for row in rows:
            assert row["raw_text"] and row["generated_token_ids"]
            assert row["prompt_token_ids"] and all(isinstance(token, int) for token in row["prompt_token_ids"])
        values = {row["name"]: row for row in rows}
        assert values["reasoning"]["reasoning"].strip() and values["reasoning"]["content"].strip()
        tool = values["tool"]["parsed"]
        assert tool["tools_called"] and len(tool["tool_calls"]) == 1
        call = tool["tool_calls"][0]["function"]
        assert call["name"] == "weather" and json.loads(call["arguments"]) == {"city": "Beijing"}
