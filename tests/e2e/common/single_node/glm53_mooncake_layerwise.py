# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in smoke against a dedicated, running GLM-5.3/Mooncake server.

Resets local prefix caching (never the external pool), compares greedy tokens,
and requires actual connector GETs. See the Mooncake hybrid attention guide.
"""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path


def call(base_url, path, payload=None):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode()
    return json.loads(body) if body and path != "/metrics" else body


def metric(base_url, name):
    rows = call(base_url, "/metrics").splitlines()
    samples = [row for row in rows if row.startswith((name + "{", name + " "))]
    if not samples:
        raise RuntimeError(f"Missing {name}; enable server/connector metrics")
    return sum(float(row.split()[-1]) for row in samples)


def reset_local_cache(base_url):
    for _ in range(30):
        result = call(base_url, "/reset_prefix_cache?reset_external=false", {})
        if isinstance(result, dict) and result.get("success") is False:
            time.sleep(1)
            continue
        # Older vLLM versions return an empty successful HTTP response.
        return
    raise RuntimeError("Local prefix cache still has pinned/in-flight blocks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="glm")
    parser.add_argument("--repetitions", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("glm53-layerwise-smoke.json"))
    args = parser.parse_args()
    # A unique first token prefix keeps the cold request out of existing pools.
    prompt = f"Run {uuid.uuid4().hex}.\n" + "The river passes the village and reaches the sea.\n" * args.repetitions
    prompt += "Summarize the repeated sentence in one sentence."
    payload = dict(model=args.model, prompt=prompt, max_tokens=64, temperature=0, seed=1024, return_token_ids=True)
    cold = call(args.base_url, "/v1/completions", payload)
    if cold["usage"]["prompt_tokens"] <= args.chunk_size:
        raise RuntimeError("Increase --repetitions to exercise multiple prefill chunks")
    reset_local_cache(args.base_url)
    # Let the periodic metrics exporter account for the cold request first.
    time.sleep(15)
    get_metric = "vllm:ascend_store_load_get_keys_total"
    before = metric(args.base_url, get_metric)
    warm = call(args.base_url, "/v1/completions", payload)
    for _ in range(30):
        after = metric(args.base_url, get_metric)
        if after > before:
            break
        time.sleep(1)
    cold_ids = cold["choices"][0].get("token_ids")
    warm_ids = warm["choices"][0].get("token_ids")
    report = dict(cold=cold, warm=warm, remote_get_keys=after - before)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not cold_ids or cold_ids != warm_ids:
        raise RuntimeError(f"Cold/warm token mismatch (or missing token IDs); see {args.output}")
    if after <= before:
        raise RuntimeError(f"No remote GETs observed; a local-cache hit is insufficient; see {args.output}")
    print(f"PASS: {len(cold_ids)} equal tokens, {after - before:g} remote GET keys; report: {args.output}")


if __name__ == "__main__":
    main()
