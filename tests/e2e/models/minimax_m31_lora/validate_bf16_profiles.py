# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run real-model public LoRA loading/generation for each BF16 core profile."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--profiles", nargs="+", default=["indexer", "mlp", "shared", "experts", "all"])
    parser.add_argument("--max-loras", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--rank", type=int, choices=(8, 16, 128), default=8)
    parser.add_argument("--alpha", type=int)
    parser.add_argument("--ep", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch_npu  # noqa: F401
    from bf16_scope import CORE_TARGETS, build_adapter, select_targets
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    torch.set_num_threads(8)
    llm = None
    report = {
        "cases": [],
        "scope": "public-LoRA/BF16-layer-oracle",
        "started": time.time(),
        "config": {
            "tp": args.tp,
            "ep": args.ep,
            "max_loras": args.max_loras,
            "rank": args.rank,
            "alpha": args.alpha if args.alpha is not None else 2 * args.rank,
        },
    }

    def persist():
        (args.directory / "report.json").write_text(json.dumps(report, indent=2))

    try:
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            enable_lora=True,
            max_loras=args.max_loras,
            max_cpu_loras=max(2, args.max_loras),
            max_lora_rank=args.rank,
            lora_dtype="bfloat16",
            lora_target_modules=list(CORE_TARGETS),
            enforce_eager=True,
            tensor_parallel_size=args.tp,
            data_parallel_size=1,
            enable_expert_parallel=args.ep,
            language_model_only=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            skip_tokenizer_init=True,
            max_num_seqs=max(1, args.max_loras + 1),
            max_model_len=256,
            max_num_batched_tokens=256,
            block_size=128,
            gpu_memory_utilization=0.60,
            seed=20261124,
            generation_config="vllm",
            worker_extension_cls="bf16_scope.ScopeWorker",
            additional_config={"enable_fused_mc2": 0},
        )
        inventories = llm.collective_rpc("m31_scope_inventory")
        (args.directory / "inventory.json").write_text(json.dumps(inventories, indent=2))
        inventory = inventories[0]["targets"]
        report["target_count"] = len(inventory)
        prompt = {"prompt_token_ids": [1, 3, 4]}
        sampling = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, logprobs=5)

        def generate(request=None):
            output = llm.generate([prompt], sampling, lora_request=request, use_tqdm=False)[0].outputs[0]
            return {
                "tokens": list(output.token_ids),
                "logprobs": [{str(k): float(v.logprob) for k, v in step.items()} for step in output.logprobs],
            }

        base = generate()
        report["base"] = base
        persist()
        for profile in args.profiles:
            targets = select_targets(inventory, profile)
            for variant in ("zero", "a", "b"):
                case = {"profile": profile, "variant": variant, "targets": len(targets), "status": "STARTED"}
                report["cases"].append(case)
                persist()
                try:
                    for identity in llm.llm_engine.list_loras():
                        llm.llm_engine.remove_lora(identity)
                    path = build_adapter(
                        args.directory / f"{profile}-{variant}",
                        targets,
                        variant=variant,
                        rank=args.rank,
                        alpha=args.alpha if args.alpha is not None else 2 * args.rank,
                    )
                    request = LoRARequest(f"{profile}-{variant}", 1, str(path))
                    assert llm.llm_engine.add_lora(request)
                    case["output"] = generate(request)
                    if variant == "zero":
                        assert case["output"] == base, "Zero adapter changed base tokens/logprobs"
                    case["layer_checks"] = llm.collective_rpc("m31_scope_check_layers", args=(str(path),))
                    metrics = [m for device in case["layer_checks"] for values in device.values() for m in values]
                    assert all(m["loaded_exact"] and m["inactive_exact"] and m["violations"] == 0 for m in metrics), (
                        case["layer_checks"]
                    )
                    if variant != "zero":
                        assert max(m["signal_margin"] for m in metrics) >= 10, (
                            "INVALID_PROBE: delta below 10x tolerance"
                        )
                    case["status"] = "SMOKE_PASS"
                except Exception as exc:
                    case.update(
                        status="FAIL", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
                    )
                persist()
                print(
                    json.dumps({k: v for k, v in case.items() if k not in {"traceback", "layer_checks", "output"}}),
                    flush=True,
                )
                if case["status"] == "FAIL":
                    report["status"] = "FAIL"
                    return 1
        report["status"] = "SMOKE_PASS"
        return 0
    except Exception as exc:
        report.update(status="FAIL", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["ended"] = time.time()
        persist()
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
