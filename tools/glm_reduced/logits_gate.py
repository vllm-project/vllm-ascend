# SPDX-License-Identifier: Apache-2.0
"""Fixed-position GLM5.x (non-Flash) intermediate-logits regression gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SAMPLE_COUNT = 200
POSITION_COUNT = 20693
VOCAB_SIZE = 154880
MAIN_LAYERS = 11
CAPTURE_LAYER = 10
TP_SIZE = 8
DATA = Path(__file__).parent / "data" / "glm52" / "logits"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model(config: dict, model_id: str) -> None:
    """Do not apply a DSA baseline to Flash/hybrid models or another family."""
    text = config.get("text_config", config)
    if "flash" in model_id.lower() or "flash" in json.dumps(config).lower():
        raise ValueError("Flash models are excluded from this gate")
    if not model_id.lower().split("/")[-1].startswith("glm-5") or text.get("model_type") != "glm_moe_dsa":
        raise ValueError("Only registered non-Flash GLM5.x DSA models are supported")
    if any(kind != "deepseek_sparse_attention" for kind in text.get("layer_types", [])):
        raise ValueError("Hybrid/linear attention is excluded")
    if text.get("num_hidden_layers") != MAIN_LAYERS or text.get("num_nextn_predict_layers") != 1:
        raise ValueError("Expected 11 main layers and one retained MTP layer")
    if text.get("vocab_size") != VOCAB_SIZE or text.get("hidden_size") != 6144:
        raise ValueError("Model dimensions differ from the registered baseline")


def load_case(directory: Path = DATA) -> tuple[dict, dict, dict]:
    case = read_json(directory / "case.json")
    for name in ("dataset.json", "A1.json"):
        if sha256(directory / name) != case["files"][name]:
            raise ValueError(f"Pinned fixture checksum mismatch: {name}")
    dataset, reference = read_json(directory / "dataset.json"), read_json(directory / "A1.json")
    samples = dataset["samples"]
    if len(samples) != SAMPLE_COUNT or len({s["id"] for s in samples}) != SAMPLE_COUNT:
        raise ValueError("Expected exactly 200 unique fixed samples")
    if samples[0]["id"] != "gsm8k-test-0015":
        raise ValueError("The first sample must not be excluded")
    for sample in samples:
        tokens = sample["input_token_ids"]
        prompt, answer = sample["prompt_token_ids"], sample["answer_token_ids"]
        if not prompt or not answer or tokens != prompt + answer[:-1]:
            raise ValueError("Invalid teacher-forcing input")
        if sample["prediction_positions"] != list(range(len(prompt) - 1, len(tokens))):
            raise ValueError("Missing/reordered prediction positions")
        if hashlib.sha256(json.dumps(tokens).encode()).hexdigest() != sample["input_sha256"]:
            raise ValueError("Input token checksum mismatch")
        if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in tokens + answer):
            raise ValueError("Invalid token ID")
    if sum(len(s["answer_token_ids"]) for s in samples) != POSITION_COUNT:
        raise ValueError("Expected all 20,693 prediction positions")
    if reference["run"] != "A1" or reference["dataset_sha256"] != case["files"]["dataset.json"]:
        raise ValueError("Reference must be A1 from the pinned dataset")
    compare(reference["samples"], reference["samples"], samples)
    return case, dataset, reference


def compare(reference: list[dict], candidate: list[dict], samples: list[dict]) -> dict:
    """Compare every position; >=90% is inclusive and uses integer arithmetic."""
    if not samples or len(reference) != len(samples) or len(candidate) != len(samples):
        raise ValueError("Missing or extra samples")
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate samples")
    matched = total = 0
    details = []
    for expected, actual, sample in zip(reference, candidate, samples):
        for record in (expected, actual):
            if record["id"] != sample["id"] or record["input_sha256"] != sample["input_sha256"]:
                raise ValueError("Sample identity/order/input mismatch")
            ids = record["argmax"]
            if len(ids) != len(sample["answer_token_ids"]) or not ids:
                raise ValueError("Missing or extra prediction positions")
            if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids):
                raise ValueError("Invalid argmax token ID")
        hits = sum(a == b for a, b in zip(expected["argmax"], actual["argmax"]))
        count = len(expected["argmax"])
        matched += hits
        total += count
        details.append({"id": sample["id"], "matched": hits, "total": count})
    return {
        "passed": matched * 10 >= total * 9,
        "metric": "A1 intermediate-logits argmax agreement (not GSM8K answer accuracy)",
        "minimum_agreement": 0.90,
        "matched": matched,
        "total": total,
        "agreement": matched / total,
        "samples": details,
    }
