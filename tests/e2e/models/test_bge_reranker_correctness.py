# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import huggingface_hub
import pytest
import torch
import yaml
from modelscope import snapshot_download  # type: ignore[import-untyped]
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm.exceptions import VLLMValidationError
from vllm.platforms import current_platform

from tests.e2e.conftest import VllmRunner

RTOL = 0.01
BGE_CONFIG_PATH = Path(__file__).with_name("configs") / "bge-reranker-v2-m3.yaml"
QUERY = "What is the capital of France?"
DOCUMENT = "Paris is the capital of France."


def _text_pairs(task):
    queries = task["queries"]
    documents = task["documents"]
    queries = [queries] if isinstance(queries, str) else queries
    documents = [documents] if isinstance(documents, str) else documents
    if len(queries) == 1:
        queries = queries * len(documents)
    assert queries and len(queries) == len(documents)
    return [[query, document] for query, document in zip(queries, documents)]


def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _token_digest(token_ids):
    return sha256(json.dumps(token_ids).encode()).hexdigest()


def _compare_scores(task, case):
    pairs = _text_pairs(task)
    reference_scores = case["reference_scores"]
    scores = case["ascend_scores"]
    assert len(scores) == len(reference_scores) == len(pairs)
    assert all(isinstance(score, float) and math.isfinite(score) for score in scores)
    assert all(isinstance(score, float) and math.isfinite(score) for score in reference_scores)
    comparisons = []
    for reference, score in zip(reference_scores, scores, strict=True):
        absolute_error = abs(score - reference)
        comparisons.append(
            {
                "absolute_error": absolute_error,
                "relative_error": absolute_error / abs(reference) if reference else None,
                "matches": score == pytest.approx(reference, rel=RTOL),
            }
        )
    case["score_comparisons"] = comparisons
    metrics = case["metrics"] = {
        "score_match_rate": sum(comparison["matches"] for comparison in comparisons) / len(scores),
    }
    expected_metrics = {metric["name"]: metric["value"] for metric in task["metrics"]}
    if "ranking_agreement" in expected_metrics or "expected_top_index" in task:
        assert len(pairs) > 1 and len({query for query, _ in pairs}) == 1, (
            "Ranking requires one query and multiple documents"
        )
        reference_ranking = sorted(range(len(reference_scores)), key=reference_scores.__getitem__, reverse=True)
        ranking = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        case["reference_ranking"] = reference_ranking
        case["ascend_ranking"] = ranking
        metrics["ranking_agreement"] = float(ranking == reference_ranking)
        if "expected_top_index" in task:
            assert ranking[0] == reference_ranking[0] == task["expected_top_index"]
    for name, expected in expected_metrics.items():
        assert metrics[name] == expected, f"{task['name']}: {name}={metrics[name]}, expected {expected}"


def _record_rejection(llm, queries, documents, message, result):
    try:
        with pytest.raises(VLLMValidationError, match=message) as error:
            llm.score(queries, documents, use_tqdm=False)
    except (AssertionError, TypeError, ValueError, pytest.fail.Exception) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    else:
        result.update(exception=type(error.value).__name__, message=str(error.value))


def _check_invalid_inputs(llm, max_model_len, results):
    cases = [
        ("empty_batch", [], [], "At least one text element"),
        ("empty_queries", [], [DOCUMENT], "At least one text element"),
        ("empty_documents", QUERY, [], "At least one text_pair element"),
        ("mismatched_batch", [QUERY, QUERY], [DOCUMENT], "Input lengths"),
        ("invalid_text_type", {"content": []}, DOCUMENT, "MultiModalParam is not supported"),
    ]
    for name, queries, documents, message in cases:
        result = {"name": name}
        results.append(result)
        _record_rejection(llm, queries, documents, message, result)
    overlong_document = "Paris " * max_model_len
    tokenizer = llm.get_tokenizer()
    token_count = len(tokenizer(QUERY, text_pair=overlong_document, truncation=False)["input_ids"])
    assert token_count > max_model_len
    result = {"name": "overlong_pair", "pair_tokens": token_count, "max_model_len": max_model_len}
    results.append(result)
    _record_rejection(llm, QUERY, overlong_document, r"maximum (?:context|model) length", result)
    outputs = llm.score(QUERY, DOCUMENT, use_tqdm=False)
    assert len(outputs) == 1 and math.isfinite(outputs[0].outputs.score)
    results.append({"name": "valid_pair_after_rejections", "score": outputs[0].outputs.score})


def run_bge_reranker_check(config_filename, tp_size, report_dir, *, skip_other_models=False):
    config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    is_reranker = config.get("model_type") == "vllm-reranker"
    if skip_other_models and not is_reranker:
        pytest.skip("Use a reranker configuration for this test")
    assert is_reranker, f"Expected a reranker configuration: {config_filename}"

    model_id = config["model_name"]
    serve = dict(config["serve"])
    trust_remote_code = serve.pop("trust_remote_code", True)
    serve["tensor_parallel_size"] = int(tp_size)
    tasks = config["tasks"]
    reference_device = current_platform.device_type
    report = {
        "status": "failed",
        "stage": "model_setup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_id,
        "config_path": str(config_filename),
        "serve": serve,
        "score_relative_tolerance": RTOL,
        "reference": {"device": reference_device, "dtype": "float16", "activation": "sigmoid"},
        "versions": {
            name: _package_version(name) for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers")
        },
        "cases": [],
        "invalid_inputs": [],
    }
    report_path = Path(report_dir) / f"{Path(model_id).name}.json"
    try:
        assert serve["tensor_parallel_size"] == 1, "This task validates a single-card reranker"
        assert reference_device == "npu" and serve["dtype"] == "half", "Expected single-card Ascend FP16 scoring"
        model_path = (
            str(Path(model_id).resolve())
            if Path(model_id).is_dir()
            else snapshot_download(model_id, local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE)
        )
        report["model_path"] = model_path
        report["stage"] = "ascend_scoring"
        failures = []
        engine_start = perf_counter()
        with VllmRunner(model_path, **serve) as runner:
            report["engine_startup_seconds"] = perf_counter() - engine_start
            for task in tasks:
                case = {"name": task["name"], "queries": task["queries"], "documents": task["documents"]}
                report["cases"].append(case)
                score_start = perf_counter()
                outputs = runner.model.score(task["queries"], task["documents"], use_tqdm=False)
                case["score_seconds"] = perf_counter() - score_start
                case["ascend_scores"] = [output.outputs.score for output in outputs]
                case["ascend_score_shape"] = [len(case["ascend_scores"])]
                case["ascend_pair_tokens"] = [len(output.prompt_token_ids) for output in outputs]
                case["ascend_pair_token_sha256"] = [_token_digest(output.prompt_token_ids) for output in outputs]
            report["stage"] = "invalid_inputs"
            _check_invalid_inputs(runner.model, serve["max_model_len"], report["invalid_inputs"])
        failures.extend(f"invalid input: {case['name']}" for case in report["invalid_inputs"] if "error" in case)

        # Release the vLLM engine before allocating NPU memory for Transformers.
        report["stage"] = "reference"
        reference_start = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=trust_remote_code
        )
        reference_model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_path, dtype=torch.float16, local_files_only=True, trust_remote_code=trust_remote_code
            )
            .to(reference_device)
            .eval()
        )
        try:
            with torch.inference_mode():
                for task, case in zip(tasks, report["cases"], strict=True):
                    pairs = _text_pairs(task)
                    queries, documents = zip(*pairs, strict=True)
                    encoded = tokenizer(
                        list(queries), list(documents), padding=True, truncation=False, return_tensors="pt"
                    )
                    counts = encoded["attention_mask"].sum(dim=1).tolist()
                    case["pair_tokens"] = counts
                    case["reference_pair_token_sha256"] = [
                        _token_digest(ids[mask.bool()].tolist())
                        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"], strict=True)
                    ]
                    if "min_pair_tokens" in task:
                        assert min(counts) >= task["min_pair_tokens"]
                    assert max(counts) <= serve["max_model_len"]
                    logits = reference_model(**encoded.to(reference_device)).logits.float().cpu()
                    assert logits.shape == (len(pairs), 1)
                    case["reference_logit_shape"] = list(logits.shape)
                    case["reference_scores"] = logits[:, 0].sigmoid().tolist()
                    try:
                        _compare_scores(task, case)
                        assert case["reference_pair_token_sha256"] == case["ascend_pair_token_sha256"], (
                            "Transformers and vLLM received different token sequences"
                        )
                    except AssertionError as error:
                        case["error"] = str(error)
                        failures.append(f"score or tokens: {task['name']}")
                    print(json.dumps(case, ensure_ascii=False))
        finally:
            del reference_model
            torch.npu.empty_cache()
        report["reference_seconds"] = perf_counter() - reference_start
        if failures:
            report["stage"] = "assertions"
            pytest.fail(f"Reranker assertions failed: {', '.join(failures)}")
        report["stage"] = "complete"
        report["status"] = "passed"
    except (Exception, pytest.fail.Exception) as error:
        report["error"] = repr(error)
        raise
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Reranker report: {report_path}")


def test_bge_reranker_correctness(config_filename, tp_size, report_dir):
    run_bge_reranker_check(config_filename, tp_size, report_dir, skip_other_models=True)
