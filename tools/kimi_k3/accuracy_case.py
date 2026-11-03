# SPDX-License-Identifier: Apache-2.0
"""CPU-only input and result validation for the EP256 accuracy case."""

import hashlib
import json
import math
import shlex
from pathlib import Path

TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def validate_inputs(model, dataset, baseline, tolerance):
    baseline, tolerance = float(baseline), float(tolerance)
    if not math.isfinite(baseline) or not math.isfinite(tolerance):
        raise ValueError("Baseline and tolerance must be finite")
    if not 0 <= baseline <= 100 or not 0 <= tolerance <= 100:
        raise ValueError("Baseline and tolerance must be percentages in [0,100]")
    config = json.loads((Path(model) / "config.json").read_text())
    text = config.get("text_config", config)
    for name, expected in {"num_experts": 256, "num_hidden_layers": 93, "num_experts_per_tok": 16}.items():
        if text.get(name) != expected:
            raise ValueError(f"Expected {name}={expected}, got {text.get(name)}")
    if not (Path(dataset) / "train.jsonl").is_file():
        raise ValueError("GSM8K train.jsonl is required by the dataset loader")
    test = Path(dataset) / "test.jsonl"
    if hashlib.sha256(test.read_bytes()).hexdigest() != TEST_SHA256:
        raise ValueError("GSM8K test.jsonl does not match the recorded dataset")
    return baseline, tolerance


def validate_results(work, dataset):
    work = Path(work)
    results = [p for p in work.rglob("gsm8k.json") if "results" in p.relative_to(work).parts]
    predictions = [
        p
        for p in work.rglob("gsm8k.jsonl")
        if "predictions" in p.relative_to(work).parts and "tmp" not in p.relative_to(work).parts
    ]
    if len(results) != 1 or len(predictions) != 1:
        raise ValueError("Expected one final result and one final prediction file")
    records = [json.loads(line) for line in predictions[0].read_text().splitlines()]
    ids = [int(row["id"]) for row in records]
    if len(ids) != 500 or set(ids) != set(range(500)):
        raise ValueError("Expected exactly 500 unique question IDs 0..499")
    source = [json.loads(line) for line in (Path(dataset) / "test.jsonl").read_text().splitlines()]
    from tools.kimi_k3.final_answer import gsm8k_final_postprocess

    correct = invalid = failures = 0
    for row in records:
        original = source[int(row["id"])]
        if original["question"] not in row["origin_prompt"]:
            raise ValueError("Prediction prompt does not match the original question")
        pred = gsm8k_final_postprocess(row.get("prediction") or "")
        reference = original["answer"].split("#### ")[1].replace(",", "").strip()
        try:
            match = pred == reference or abs(float(pred) - int(reference)) < 1e-6
        except (TypeError, ValueError):
            match = False
        correct += bool(match)
        invalid += pred == "NULL"
        failures += not row.get("success", False)
    result = json.loads(results[0].read_text())
    details = result["details"]
    if len(details) != 500 or (isinstance(details, dict) and set(map(int, details)) != set(range(500))):
        raise ValueError("Expected 500 evaluation details")
    accuracy = correct / 5
    if not math.isclose(float(result["accuracy"]), accuracy, abs_tol=1e-8):
        raise ValueError("Independent accuracy does not match AISBench")
    return {
        "count": 500,
        "correct": correct,
        "accuracy": accuracy,
        "invalid_final_answers": invalid,
        "request_failures": failures,
        "result_path": str(results[0]),
        "prediction_path": str(predictions[0]),
    }


def service_arguments(script):
    """Reuse serving flags, leaving the shared fixture to allocate a free port."""
    command = script[script.index("exec vllm serve") :].replace("\\\n", " ")
    args = shlex.split(command)[4:]
    port = args.index("--port")
    del args[port : port + 2]
    return args
