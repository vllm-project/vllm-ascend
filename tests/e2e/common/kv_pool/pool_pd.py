# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare isolated, already-running pool PD and baseline completion services.

This probe does not launch/stop engines. It requires exclusive test traffic and
metrics from every candidate P/D API engine. A local-recompute fallback cannot
pass solely because its generated text matches the baseline.
"""

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

PREFIX = "vllm:pool_pd_"
P_COUNTERS = ("save_committed", "save_bytes", "save_failed")
D_COUNTERS = ("restore_completed", "restore_bytes", "restore_failed", "restore_cancelled")


def wait_for_transfers(client, prefill_metrics, decode_metrics, before, metrics_timeout, evidence):
    """Require some real pool traffic and no reported transfer failures.

    Engine-wide counters cannot prove that every benchmark request transferred.
    Callers must isolate traffic and include all P/D API engines.
    """
    deadline = time.monotonic() + metrics_timeout
    while True:
        after = read_metrics(client, prefill_metrics, P_COUNTERS) | read_metrics(client, decode_metrics, D_COUNTERS)
        delta = {key: after[key] - before[key] for key in before}
        evidence["counter_deltas"] = delta
        if any(value < 0 for value in delta.values()):
            raise ValueError("Pool PD counters reset during acceptance; repeat on stable isolated engines")
        if any(delta[key] for key in ("save_failed", "restore_failed", "restore_cancelled")):
            raise ValueError(f"Pool PD failure or cancellation during acceptance: {delta}")
        if all(delta[key] > 0 for key in ("save_committed", "save_bytes", "restore_completed", "restore_bytes")):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No completed P write and D pool restore observed; local fallback is not a pass: {delta}"
            )
        time.sleep(0.2)


@contextmanager
def observe_transfers(client, prefill_metrics, decode_metrics, output: Path, *, metrics_timeout=30):
    """Bracket an existing benchmark, preserving its first failure and evidence."""
    if not prefill_metrics or not decode_metrics:
        raise ValueError("Both P and D metrics endpoints are required")
    with output.open("x", encoding="utf-8") as stream:
        evidence = {"status": "running", "prefill_metrics": prefill_metrics, "decode_metrics": decode_metrics}
        try:
            before = read_metrics(client, prefill_metrics, P_COUNTERS) | read_metrics(
                client, decode_metrics, D_COUNTERS
            )
            evidence["before"] = before
            yield
            wait_for_transfers(client, prefill_metrics, decode_metrics, before, metrics_timeout, evidence)
            evidence["status"] = "passed"
        except BaseException as exc:
            evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")


def read_metrics(client: httpx.Client, urls: list[str], counters: tuple[str, ...]) -> dict[str, float]:
    totals = dict.fromkeys(counters, 0.0)
    for url in urls:
        response = client.get(url)
        response.raise_for_status()
        values = {}
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                for key in counters:
                    if sample.name == f"{PREFIX}{key}_total":
                        if not math.isfinite(sample.value) or sample.value < 0:
                            raise ValueError(f"Invalid counter at {url}: {sample.name}")
                        values[key] = values.get(key, 0.0) + sample.value
        if values.keys() != totals.keys():
            raise ValueError(f"Missing pool PD counters at {url}; check connector and metrics configuration")
        for key in counters:
            totals[key] += values[key]
    return totals


def completion(client: httpx.Client, url: str, payload: dict) -> dict:
    endpoint = "/v1/chat/completions" if "messages" in payload else "/v1/completions"
    response = client.post(url.rstrip("/") + endpoint, json=payload)
    if response.is_error:
        raise ValueError(f"Completion failed at {url}: HTTP {response.status_code}: {response.text[:8192]}")
    result = response.json()
    choices = result.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Acceptance requires one completion choice")
    tokens = choices[0].get("token_ids")
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("Response must contain generated token_ids; text alone is insufficient")
    return result


def compare_case(
    client: httpx.Client,
    case: dict,
    *,
    model: str,
    baseline_url: str,
    candidate_url: str,
    prefill_metrics: list[str],
    decode_metrics: list[str],
    metrics_timeout: float,
    evidence: dict | None = None,
) -> dict:
    if ("prompt" in case) == ("messages" in case) or "kv_transfer_params" in case:
        raise ValueError("Each case must supply either prompt or messages and must not manufacture transfer metadata")
    if not prefill_metrics or not decode_metrics:
        raise ValueError("Both P and D metrics endpoints are required")
    payload = {key: value for key, value in case.items() if key != "id"}
    payload.update(model=model, temperature=0, seed=0, stream=False, n=1, return_token_ids=True)
    payload.setdefault("max_tokens", 32)
    evidence = {} if evidence is None else evidence
    evidence.update(id=case.get("id"), passed=False, request=payload)
    baseline = completion(client, baseline_url, payload)
    evidence["baseline"] = baseline
    before = read_metrics(client, prefill_metrics, P_COUNTERS) | read_metrics(client, decode_metrics, D_COUNTERS)
    candidate = completion(client, candidate_url, payload)
    evidence["candidate"] = candidate
    wait_for_transfers(client, prefill_metrics, decode_metrics, before, metrics_timeout, evidence)
    expected, actual = baseline["choices"][0], candidate["choices"][0]
    matched = all(expected.get(key) == actual.get(key) for key in ("token_ids", "finish_reason", "stop_reason"))
    evidence["passed"] = matched
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--prefill-metrics-url", action="append", required=True)
    parser.add_argument("--decode-metrics-url", action="append", required=True)
    parser.add_argument("--cases", type=Path, required=True, help="JSONL completion/chat requests; optional id per row")
    parser.add_argument("--output", type=Path, required=True, help="New result JSON file; never overwritten")
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--metrics-timeout", type=float, default=30)
    args = parser.parse_args()
    if (
        not math.isfinite(args.request_timeout)
        or not math.isfinite(args.metrics_timeout)
        or args.request_timeout <= 0
        or args.metrics_timeout < 0
    ):
        parser.error("Timeouts must be positive (metrics timeout may be zero)")
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if not cases or any(not isinstance(case, dict) for case in cases):
        parser.error("Cases must contain at least one JSON object")
    # Reserve the evidence file before sending traffic; preserve partial results
    # on any failure and never overwrite an earlier run.
    with args.output.open("x", encoding="utf-8") as output:
        results = []
        report = {"status": "running", "cases": results}
        try:
            with httpx.Client(timeout=args.request_timeout, trust_env=False) as client:
                for case in cases:
                    evidence = {"id": case.get("id"), "passed": False}
                    results.append(evidence)
                    result = compare_case(
                        client,
                        case,
                        model=args.model,
                        baseline_url=args.baseline_url,
                        candidate_url=args.candidate_url,
                        prefill_metrics=args.prefill_metrics_url,
                        decode_metrics=args.decode_metrics_url,
                        metrics_timeout=args.metrics_timeout,
                        evidence=evidence,
                    )
                    if not result["passed"]:
                        raise ValueError(f"Generated tokens differ for case {case.get('id')!r}")
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
