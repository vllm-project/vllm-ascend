# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Capture real reloads, then compare every rank with an independent cold engine."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def capture(args):
    import torch
    from bf16_scope import CORE_TARGETS
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    torch.set_num_threads(8)
    options = dict(
        model=str(args.replacement if args.cold else args.model),
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        enable_expert_parallel=args.ep,
        language_model_only=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        skip_tokenizer_init=True,
        max_num_seqs=3,
        max_model_len=256,
        max_num_batched_tokens=512,
        block_size=128,
        gpu_memory_utilization=0.6,
        kv_cache_memory_bytes=536870912,
        seed=20261124,
        generation_config="vllm",
        worker_extension_cls="reload_observation.ReloadWorker",
        additional_config={"enable_fused_mc2": 0},
    )
    if args.lora:
        options.update(
            enable_lora=True,
            max_loras=2,
            max_cpu_loras=3,
            max_lora_rank=8,
            lora_dtype="bfloat16",
            lora_target_modules=list(CORE_TARGETS),
        )
    report = {"tp": args.tp, "ep": args.ep, "lora": args.lora, "cold": args.cold, "stages": []}
    llm = None
    try:
        llm = LLM(**options)
        adapters = {
            name: LoRARequest(name, index, str(path.resolve()))
            for index, (name, path) in enumerate(zip(("a", "b"), args.adapters, strict=False), 1)
        }
        for request in adapters.values():
            assert llm.llm_engine.add_lora(request)
        prompts = [[1, 3, 4], [1] + list(range(100, 116)), [1] + list(range(100, 228))]
        sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5)

        def observe(stage):
            outputs = {}
            for index, prompt in enumerate(prompts):
                for variant in ("base", *adapters):
                    value = llm.generate(
                        [{"prompt_token_ids": prompt}], sampling, lora_request=adapters.get(variant), use_tqdm=False
                    )[0].outputs[0]
                    outputs[f"{index}:{variant}"] = {
                        "tokens": list(value.token_ids),
                        "logprobs": [
                            {str(key): float(prob.logprob) for key, prob in step.items()} for step in value.logprobs
                        ],
                    }
            report["stages"].append(
                {"stage": stage, "outputs": outputs, "snapshot": llm.collective_rpc("m31_reload_snapshot")}
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")

        observe("cold")
        if not args.cold:
            for stage, path in (("changed", args.replacement), ("restored", args.model)):
                llm.collective_rpc(
                    "reload_weights", timeout=300, kwargs={"weights_path": str(path), "is_checkpoint_format": True}
                )
                observe(stage)
        report["status"] = "CAPTURED"
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def compare(args):
    actual = json.loads(args.actual.read_text())
    reference = json.loads(args.reference.read_text())
    assert actual.get("status") == reference.get("status") == "CAPTURED"
    for key in ("tp", "ep", "lora"):
        assert actual[key] == reference[key], key
    assert actual.get("dp", 1) == reference.get("dp", 1)
    before, changed, restored = actual["stages"]
    target = reference["stages"][0]
    assert [s["stage"] for s in actual["stages"]] == ["cold", "changed", "restored"]
    assert len(before["snapshot"]) == len(changed["snapshot"]) == len(restored["snapshot"]) == len(target["snapshot"])
    ranks = []
    for old, new, undo, cold in zip(
        before["snapshot"], changed["snapshot"], restored["snapshot"], target["snapshot"], strict=True
    ):
        assert old.get("rank") == new.get("rank") == undo.get("rank") == cold.get("rank")
        expected = cold["parameters"]
        differences = [name for name in expected if old["parameters"].get(name) != expected[name]]
        ranks.append(
            {
                "parameter_count": len(expected),
                "expected_changes": len(differences),
                "parameters_exact": new["parameters"] == expected,
                "restored_parameters_exact": undo["parameters"] == old["parameters"],
                "adapter_buffers_unchanged": new["adapter_buffers"]
                == undo["adapter_buffers"]
                == old["adapter_buffers"],
                "no_meta_parameters": all(not param["meta"] for param in new["parameters"].values()),
            }
        )
    keys = list(before["outputs"])
    outputs_exact = {key: changed["outputs"][key] == target["outputs"][key] for key in keys}
    tokens_exact = {key: changed["outputs"][key]["tokens"] == target["outputs"][key]["tokens"] for key in keys}
    probe_valid = all(rank["expected_changes"] > 0 for rank in ranks) and any(
        before["outputs"][key] != target["outputs"][key] for key in keys
    )
    restored_outputs_exact = restored["outputs"] == before["outputs"]
    passed = (
        probe_valid
        and restored_outputs_exact
        and all(outputs_exact.values())
        and all(
            all(
                rank[key]
                for key in (
                    "parameters_exact",
                    "restored_parameters_exact",
                    "adapter_buffers_unchanged",
                    "no_meta_parameters",
                )
            )
            for rank in ranks
        )
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "actual_sha256": hashlib.sha256(args.actual.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "ranks": ranks,
        "probe_valid": probe_valid,
        "outputs_exact": outputs_exact,
        "tokens_exact": tokens_exact,
        "restored_outputs_exact": restored_outputs_exact,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not passed:
        raise SystemExit(1)


def capture_http(args):
    import httpx

    report = {"tp": args.tp, "dp": args.dp, "ep": args.ep, "lora": True, "cold": args.cold, "stages": []}
    with httpx.Client(base_url=args.url, timeout=300) as client:
        deadline = time.monotonic() + 600
        while True:
            try:
                if client.get("/v1/models").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("HTTP service did not become ready")
            time.sleep(1)

        def rpc(method, **kwargs):
            response = client.post("/collective_rpc", json={"method": method, "kwargs": kwargs, "timeout": 300})
            response.raise_for_status()
            return response.json()["results"]

        def ranks(value):
            if isinstance(value, dict) and "parameters" in value:
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from ranks(item)

        for stage, path in (
            (("cold", None),)
            if args.cold
            else (("cold", None), ("changed", args.replacement), ("restored", args.model))
        ):
            if path is not None:
                rpc("reload_weights", weights_path=str(path), is_checkpoint_format=True)
            outputs = {}
            for index, prompt in enumerate(([1, 3, 4], [1] + list(range(100, 116)), [1] + list(range(100, 228)))):
                for variant in ("base", "a", "b"):
                    response = client.post(
                        "/v1/completions",
                        json={
                            "model": "m31-base" if variant == "base" else variant,
                            "prompt": prompt,
                            "temperature": 0,
                            "max_tokens": 8,
                            "ignore_eos": True,
                            "seed": 20261124,
                            "return_token_ids": True,
                            "logprobs": 5,
                            "return_tokens_as_token_ids": True,
                        },
                    )
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    assert choice["token_ids"] and choice["logprobs"] is not None
                    outputs[f"{index}:{variant}"] = {"tokens": choice["token_ids"], "logprobs": choice["logprobs"]}
            snapshot = sorted(ranks(rpc("m31_reload_snapshot_all_dp")), key=lambda rank: rank["rank"])
            assert len(snapshot) == args.tp * args.dp
            report["stages"].append({"stage": stage, "outputs": outputs, "snapshot": snapshot})
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "CAPTURED"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--replacement", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--lora", action="store_true")
    run.add_argument("--adapters", type=Path, nargs="*", default=[])
    run.add_argument("--cold", action="store_true")
    run.add_argument("--tp", type=int, default=1)
    run.add_argument("--ep", action="store_true")
    http = commands.add_parser("capture-http")
    http.add_argument("--model", type=Path, required=True)
    http.add_argument("--replacement", type=Path, required=True)
    http.add_argument("--output", type=Path, required=True)
    http.add_argument("--url", default="http://127.0.0.1:8000")
    http.add_argument("--tp", type=int, default=2)
    http.add_argument("--dp", type=int, default=2)
    http.add_argument("--ep", action="store_true")
    http.add_argument("--cold", action="store_true")
    check = commands.add_parser("compare")
    check.add_argument("--actual", type=Path, required=True)
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("capture", "capture-http"):
        args.model = args.model.resolve()
        args.replacement = args.replacement.resolve()
    if args.command == "capture":
        if len(args.adapters) > 2 or (args.adapters and not args.lora):
            parser.error("Use --lora with at most two adapter paths")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        capture(args)
    elif args.command == "capture-http":
        capture_http(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
