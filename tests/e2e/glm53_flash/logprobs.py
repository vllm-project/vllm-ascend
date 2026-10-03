# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict curl/Completion protocol checks, shared by capture and CI."""

import json
import math
import re
import subprocess
from pathlib import Path

from tests.e2e.glm53_flash.checkpoint import write_json

OUTPUT_TOKENS = 32
TOP_LOGPROBS = 5
REPLAY_ROUNDS = 2
LOGPROB_ATOL = 1e-4
CURL_TIMEOUT_SECONDS = 180
SERVED_MODEL = "glm53-flash-ci"
TOKEN_KEY = re.compile(r"token_id:(\d+)\Z")


def completion_request(prompt: list[int]) -> dict:
    return {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "temperature": 0,
        "seed": 0,
        "top_p": 1,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": False,
        "logprobs": TOP_LOGPROBS,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }


def curl_completion(url: str, prompt: list[int], artifact: Path) -> dict:
    artifact.mkdir(parents=True, exist_ok=False)
    request_path = artifact / "request.json"
    response_path = artifact / "response.json"
    write_json(request_path, completion_request(prompt))
    command = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--noproxy",
        "*",
        "--connect-timeout",
        "10",
        "--max-time",
        str(CURL_TIMEOUT_SECONDS),
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        f"@{request_path.resolve()}",
        "--output",
        str(response_path.resolve()),
        url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=CURL_TIMEOUT_SECONDS + 10)
    except subprocess.TimeoutExpired as error:
        (artifact / "curl.stderr").write_text(str(error), encoding="utf-8")
        raise AssertionError(f"curl timed out; artifacts: {artifact}") from error
    (artifact / "curl.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise AssertionError(f"curl failed ({result.returncode}); response/stderr: {artifact}")
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response.get("model") != SERVED_MODEL:
            raise AssertionError("Response came from a different served model")
        usage = response["usage"]
        if usage["prompt_tokens"] != len(prompt) or usage["completion_tokens"] != OUTPUT_TOKENS:
            raise AssertionError("Server token counts differ from the fixed request")
        return normalize_completion(response)
    except (KeyError, TypeError, ValueError, AssertionError) as error:
        raise AssertionError(f"Invalid completion; artifacts: {artifact}: {error}") from error


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError(f"Missing/non-finite logprob at {context}: {value!r}")
    return float(value)


def normalize_completion(response: dict) -> dict:
    if "error" in response or len(response["choices"]) != 1:
        raise AssertionError("Expected exactly one successful completion")
    choice = response["choices"][0]
    if choice["finish_reason"] != "length":
        raise AssertionError(f"Unexpected finish reason: {choice['finish_reason']}")
    ids = choice["token_ids"]
    if not isinstance(ids, list) or len(ids) != OUTPUT_TOKENS:
        raise AssertionError(f"Expected {OUTPUT_TOKENS} output token IDs")
    if any(type(token) is not int or token < 0 for token in ids):
        raise AssertionError("Invalid output token IDs")
    logprobs = choice["logprobs"]
    for key in ("tokens", "token_logprobs", "top_logprobs"):
        if len(logprobs[key]) != OUTPUT_TOKENS:
            raise AssertionError(f"Invalid logprobs length: {key}")
    steps = []
    for step, token in enumerate(ids):
        if logprobs["tokens"][step] != f"token_id:{token}":
            raise AssertionError(f"Token placeholder/ID mismatch at step {step}")
        selected = _finite(logprobs["token_logprobs"][step], f"step {step}")
        candidates = logprobs["top_logprobs"][step]
        if not isinstance(candidates, dict) or len(candidates) not in (TOP_LOGPROBS, TOP_LOGPROBS + 1):
            raise AssertionError(f"Expected top-{TOP_LOGPROBS} candidates (plus optional selected token)")
        normalized = {}
        for key, value in candidates.items():
            match = TOKEN_KEY.fullmatch(key)
            if not match or key != f"token_id:{int(match[1])}":
                raise AssertionError(f"Candidate is not represented by token ID: {key}")
            normalized[str(int(match[1]))] = _finite(value, f"step {step}, token {key}")
        if str(token) not in normalized or normalized[str(token)] != selected:
            raise AssertionError(f"Selected token missing/inconsistent in top logprobs at step {step}")
        steps.append({"token_logprob": selected, "top_logprobs": normalized})
    return {"token_ids": ids, "steps": steps}


def compare_completion(expected: dict, actual: dict, case: str) -> float:
    """Stop at the first token divergence: later conditional histories differ."""
    for result in (expected, actual):
        if len(result["token_ids"]) != OUTPUT_TOKENS or len(result["steps"]) != OUTPUT_TOKENS:
            raise AssertionError(f"{case}: expected {OUTPUT_TOKENS} generated steps")
    max_error = 0.0
    for step in range(OUTPUT_TOKENS):
        phase = "prefill" if step == 0 else "decode"
        context = f"{case}: {phase} step={step}"
        expected_id, actual_id = expected["token_ids"][step], actual["token_ids"][step]
        if expected_id != actual_id:
            raise AssertionError(f"{context}: token mismatch: golden={expected_id}, actual={actual_id}")
        baseline, replay = expected["steps"][step], actual["steps"][step]
        if baseline["top_logprobs"].keys() != replay["top_logprobs"].keys():
            raise AssertionError(f"{context}: candidate token-ID set mismatch")
        pairs = [("selected", baseline["token_logprob"], replay["token_logprob"])]
        pairs.extend(
            (key, baseline["top_logprobs"][key], replay["top_logprobs"][key])
            for key in sorted(baseline["top_logprobs"], key=int)
        )
        for token, base, value in pairs:
            error = abs(_finite(value, context) - _finite(base, context))
            max_error = max(max_error, error)
            if error > LOGPROB_ATOL:
                raise AssertionError(
                    f"{context}: token={token}, golden={base}, actual={value}, "
                    f"abs_error={error} > atol={LOGPROB_ATOL} (rtol=0)"
                )
    return max_error


def validate_golden(golden: dict, expected_contract: dict, case_names: set[str]) -> None:
    if golden.get("schema_version") != 1 or golden.get("contract") != expected_contract:
        raise AssertionError("Checkpoint/input/config differs from the reviewed golden")
    if set(golden["cases"]) != case_names:
        raise AssertionError("Golden cases differ from the reviewed input corpus")
    validation = golden["validation"]
    if len(validation) != 3:
        raise AssertionError("Golden must pass three independent cold starts")
    for report in validation:
        error = _finite(report["max_abs_error"], "golden validation")
        if report.get("errors") or report.get("fatal_error") or not 0 <= error <= LOGPROB_ATOL:
            raise AssertionError("Golden includes an unsuccessful cold-start validation")
        expected_requests = REPLAY_ROUNDS * len(case_names)
        if report.get("completed_rounds") != REPLAY_ROUNDS or report.get("completed_requests") != expected_requests:
            raise AssertionError("Golden has incomplete replay rounds/requests")
    if sorted(worker["rank"] for worker in golden["workers"]) != [0, 1]:
        raise AssertionError("Golden must contain both workers' effective configuration")
    for name, result in golden["cases"].items():
        compare_completion(result, result, f"golden/{name}")
