# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare API token IDs and collect named, bounded same-batch diagnostics."""

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path

import httpx


async def check(args, report):
    """Compare solo/mixed tokens, then exercise slot eviction and API reload."""
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=180) as client:
        deadline = time.monotonic() + 600
        while True:
            try:
                ready = await client.get("/v1/models")
                if ready.status_code == 200:
                    report["models"] = ready.json()
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("The HTTP model service did not become ready")
            await asyncio.sleep(1)
        prompts = [[1, 3, 4], [1] + list(range(100, 116)), [1] + list(range(100, 228))]

        async def generate(index, variant, gate=None):
            if gate is not None:
                await gate.wait()
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "m31-base" if variant == "base" else variant,
                    "prompt": prompts[index],
                    "temperature": 0,
                    "max_tokens": 8,
                    "ignore_eos": True,
                    "seed": 20261124,
                    "return_token_ids": True,
                },
            )
            response.raise_for_status()
            value = response.json()
            tokens = value["choices"][0].get("token_ids")
            if not isinstance(tokens, list) or not tokens:
                raise ValueError("The API did not return raw token_ids")
            return {"request_id": value["id"], "tokens": tokens, "index": index, "variant": variant}

        async def rpc(method, args=None):
            response = await client.post("/collective_rpc", json={"method": method, "args": args or []})
            response.raise_for_status()
            return response.json()["results"]

        baselines = {}
        for i in range(3):
            for v in ("base", "a", "b"):
                baselines[(i, v)] = (await generate(i, v))["tokens"]
        report["baselines"] = {f"{i}:{v}": tokens for (i, v), tokens in baselines.items()}
        witnessed = False
        for batch in range(3):
            await rpc("m31_trace_start", ["batch"])
            gate = asyncio.Event()
            requests = [(i % 3, ("base", "a", "b")[(i + batch) % 3]) for i in range(12)]
            tasks = [asyncio.create_task(generate(i, v, gate)) for i, v in requests]
            gate.set()
            outputs = await asyncio.gather(*tasks)
            trace = await rpc("m31_trace_stop")

            def events(value):
                if isinstance(value, dict):
                    if value.get("kind") == "batch":
                        yield value
                elif isinstance(value, list):
                    for member in value:
                        yield from events(member)

            same_batch = any(event["slots"] == [-1, 0, 1] for event in events(trace))
            exact = [o["tokens"] == baselines[(o["index"], o["variant"])] for o in outputs]
            report["batches"].append(
                {"tokens_exact": exact, "outputs": outputs, "trace": trace, "same_batch": same_batch}
            )
            witnessed |= same_batch
            if not all(exact):
                raise AssertionError("API mixed requests differ from their individual baselines")
        if not witnessed:
            raise AssertionError("No actual execution batch contained base/A/B together")
        report["same_batch_witnessed"] = True
        if args.third_adapter is not None:
            response = await client.post(
                "/v1/load_lora_adapter",
                json={
                    "lora_name": "c",
                    "lora_path": str(args.third_adapter.resolve()),
                },
            )
            response.raise_for_status()
            c_baseline = (await generate(0, "c"))["tokens"]
            gate = asyncio.Event()
            variants = ("base", "b", "c") * 4
            tasks = [asyncio.create_task(generate(0, variant, gate)) for variant in variants]
            gate.set()
            outputs = await asyncio.gather(*tasks)
            expected = {"base": baselines[(0, "base")], "b": baselines[(0, "b")], "c": c_baseline}
            exact = [value["tokens"] == expected[value["variant"]] for value in outputs]
            assert all(exact), "API output changed after adapter C reassigned GPU slots"
            assert (await generate(0, "a"))["tokens"] == baselines[(0, "a")], "Adapter A did not recover"
            response = await client.post("/v1/unload_lora_adapter", json={"lora_name": "a"})
            response.raise_for_status()
            response = await client.post(
                "/v1/load_lora_adapter",
                json={
                    "lora_name": "a",
                    "lora_path": str(args.adapter_a.resolve()),
                },
            )
            response.raise_for_status()
            assert (await generate(0, "a"))["tokens"] == baselines[(0, "a")], "Reloaded A changed token IDs"
            report["lifecycle"] = {"c_mixed_tokens_exact": exact, "a_restored": True, "a_reloaded": True}
        report["status"] = "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--third-adapter", type=Path)
    parser.add_argument("--adapter-a", type=Path)
    args = parser.parse_args()
    if args.third_adapter is not None and args.adapter_a is None:
        parser.error("--third-adapter requires --adapter-a for the unload/reload control")
    report = {"started": time.time(), "batches": []}
    try:
        asyncio.run(check(args, report))
    except Exception as exc:
        report.update(status="FAIL", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    report["ended"] = time.time()
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"traceback", "batches", "baselines", "models"}}))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
