#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


COMMON_PREFIX = """You are evaluating a DeepSeek V4 sparse-attention KV-cache experiment.
Keep the following synthetic notes in mind and answer the final question.
"""

FACT_BLOCK = """Section {idx}: request locality matters for sparse attention. Adjacent decode
steps often revisit a similar set of historical tokens, but the model may still
occasionally jump to distant evidence. The experiment should preserve exact
outputs while measuring cache reuse and miss pressure.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send OpenAI-compatible probe requests to a DSV4 vLLM server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8900")
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", choices=["smoke", "shared-prefix", "long-decode", "mixed"], default="mixed")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-repeat", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    return parser.parse_args()


def request_json(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            request_json("GET", f"{base_url.rstrip('/')}/v1/models", None, timeout=10.0)
            return
        except Exception as exc:  # noqa: BLE001 - report the final connection error.
            last_error = exc
            time.sleep(5)
    raise TimeoutError(f"server did not become ready within {timeout_s}s; last_error={last_error}")


def repeated_context(prompt_repeat: int) -> str:
    return "\n".join(FACT_BLOCK.format(idx=i) for i in range(prompt_repeat))


def build_prompts(scenario: str, rounds: int, prompt_repeat: int) -> list[tuple[int, int, str]]:
    prompts: list[tuple[int, int, str]] = []
    shared_context = COMMON_PREFIX + repeated_context(prompt_repeat)

    smoke_prompts = [
        COMMON_PREFIX + "Question: answer with one sentence about why sparse attention can help long context.",
        COMMON_PREFIX + "Question: list two risks of remote KV-cache loading.",
    ]
    shared_suffixes = [
        "Question: summarize the cache-locality hypothesis in three bullets.",
        "Question: identify the most important correctness risk in this experiment.",
        "Question: explain why a small on-device LRU might still work.",
        "Question: propose one metric for deciding whether to implement this feature.",
    ]
    long_decode_prompt = (
        shared_context
        + "\nQuestion: write a careful experimental analysis with observations, risks, and next actions."
    )

    for round_idx in range(rounds):
        if scenario == "smoke":
            selected = smoke_prompts
        elif scenario == "shared-prefix":
            selected = [shared_context + "\n" + suffix for suffix in shared_suffixes]
        elif scenario == "long-decode":
            selected = [long_decode_prompt]
        else:
            selected = smoke_prompts + [shared_context + "\n" + suffix for suffix in shared_suffixes[:2]]
            selected.append(long_decode_prompt)

        for request_idx, prompt in enumerate(selected):
            prompts.append((round_idx, request_idx, prompt))

    return prompts


def call_chat(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    try:
        response = request_json("POST", f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout)
        latency_s = time.perf_counter() - started
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        return {
            "ok": True,
            "latency_s": latency_s,
            "usage": response.get("usage", {}),
            "response_preview": content[:240],
        }
    except urllib.error.HTTPError as exc:
        latency_s = time.perf_counter() - started
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "latency_s": latency_s, "error": f"HTTP {exc.code}: {body[:1000]}"}
    except Exception as exc:  # noqa: BLE001 - keep probe script self-contained.
        latency_s = time.perf_counter() - started
        return {"ok": False, "latency_s": latency_s, "error": repr(exc)}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def write_summary(out_dir: Path, results: list[dict]) -> None:
    latencies = [item["latency_s"] for item in results if item.get("ok")]
    summary = {
        "total": len(results),
        "ok": sum(1 for item in results if item.get("ok")),
        "failed": sum(1 for item in results if not item.get("ok")),
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_max_s": max(latencies) if latencies else None,
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wait_for_server(args.base_url, args.wait_timeout)
    prompts = build_prompts(args.scenario, args.rounds, args.prompt_repeat)
    results_path = out_dir / "probe_results.jsonl"
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = {}
        for round_idx, request_idx, prompt in prompts:
            future = executor.submit(call_chat, args.base_url, args.model, prompt, args.max_tokens, args.timeout)
            futures[future] = (round_idx, request_idx, len(prompt))

        with results_path.open("w", encoding="utf-8") as results_file:
            for future in as_completed(futures):
                round_idx, request_idx, prompt_chars = futures[future]
                result = future.result()
                result.update(
                    {
                        "scenario": args.scenario,
                        "round": round_idx,
                        "request_idx": request_idx,
                        "prompt_chars": prompt_chars,
                    }
                )
                results.append(result)
                results_file.write(json.dumps(result, separators=(",", ":")) + "\n")
                results_file.flush()
                status = "ok" if result.get("ok") else "failed"
                print(
                    f"[{status}] round={round_idx} request={request_idx} latency={result['latency_s']:.3f}s",
                    flush=True,
                )

    write_summary(out_dir, results)
    print(f"Wrote {results_path} and {out_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
