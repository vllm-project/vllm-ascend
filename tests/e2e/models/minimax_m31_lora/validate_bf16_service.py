# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise real base/A/B mixed batches, slot reuse, and long-context indexer."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--profile", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indexer-long", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch_npu  # noqa: F401
    from bf16_scope import CORE_TARGETS, build_adapter
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    torch.set_num_threads(8)
    report = {
        "profile": args.profile,
        "started": time.time(),
        "cases": [],
        "mode": "indexer-long" if args.indexer_long else "mixed-LLM",
    }
    llm = None
    try:
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            enable_lora=True,
            max_loras=2,
            max_cpu_loras=3,
            max_lora_rank=8,
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
            max_num_seqs=3,
            max_model_len=4352 if args.indexer_long else 256,
            max_num_batched_tokens=4352 if args.indexer_long else 256,
            block_size=128,
            gpu_memory_utilization=0.6,
            seed=20261124,
            generation_config="vllm",
            worker_extension_cls="bf16_scope.ScopeWorker",
            additional_config={"enable_fused_mc2": 0},
        )
        adapters = {v: LoRARequest(v, i, str(args.online / f"{args.profile}-{v}")) for i, v in enumerate(("a", "b"), 1)}
        for request in adapters.values():
            assert llm.llm_engine.add_lora(request)
        sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5)

        def generate(tokens, variants):
            outputs = llm.generate(
                [{"prompt_token_ids": t} for t in tokens],
                sampling,
                lora_request=[adapters.get(v) for v in variants],
                use_tqdm=False,
            )
            return [{"tokens": list(o.outputs[0].token_ids), "request_id": o.request_id} for o in outputs]

        if args.indexer_long:
            prompt = [1] + [100 + (i * 7919) % 20000 for i in range(4096)]
            outputs = {}
            for variant in ("base", "a", "b", "a"):
                llm.collective_rpc("m31_trace_start", args=("indexer",))
                output = generate([prompt], [variant])[0]
                trace = llm.collective_rpc("m31_trace_stop")
                report["cases"].append({"variant": variant, "output": output, "trace": trace})
                if variant in outputs:
                    assert output["tokens"] == outputs[variant]["tokens"], "Indexer A did not recover"
                outputs[variant] = output
            base_trace = report["cases"][0]["trace"][0]
            a_trace = report["cases"][1]["trace"][0]
            b_trace = report["cases"][2]["trace"][0]
            report["query_changed_a"] = base_trace[0]["query"] != a_trace[0]["query"]
            report["query_changed_b"] = base_trace[0]["query"] != b_trace[0]["query"]
            report["selection_changed_a"] = base_trace[0]["selection"] != a_trace[0]["selection"]
            report["selection_changed_b"] = base_trace[0]["selection"] != b_trace[0]["selection"]
            assert all(
                report[k] for k in ("query_changed_a", "query_changed_b", "selection_changed_a", "selection_changed_b")
            )
            control = generate([prompt], ["a"])[0]
            assert control["tokens"] == outputs["a"]["tokens"], "Unobserved long-context control differs"
            report["prompt_length"] = len(prompt)
        else:
            prompts = [[1, 3, 4], [1] + list(range(100, 116)), [1] + list(range(100, 228))]
            baselines = {}
            for index, prompt in enumerate(prompts):
                for variant in ("base", "a", "b"):
                    baselines[(index, variant)] = generate([prompt], [variant])[0]["tokens"]
            batches = [[(i, v) for v in ("base", "a", "b")] for i in range(3)]
            batches += [[(0, "a"), (1, "base"), (2, "b"), (1, "a")], [(2, "b"), (0, "a"), (1, "base")]]
            witnessed = False
            for batch in batches:
                llm.collective_rpc("m31_trace_start", args=("batch",))
                outputs = generate([prompts[i] for i, v in batch], [v for i, v in batch])
                trace = llm.collective_rpc("m31_trace_stop")
                exact = [o["tokens"] == baselines[(i, v)] for o, (i, v) in zip(outputs, batch, strict=True)]
                witnessed |= any(r["slots"] == [-1, 0, 1] for device in trace for r in device)
                report["cases"].append({"batch": batch, "outputs": outputs, "tokens_exact": exact, "trace": trace})
                assert all(exact), "Mixed requests differ from their individual baselines"
            assert witnessed, "No actual execution batch contained base/A/B together"
            report["mixed_batch_witnessed"] = witnessed
            target_rows = json.loads((args.online / f"{args.profile}-a" / "targets.json").read_text())
            c_dir = build_adapter(args.output.parent / f"{args.output.stem}-c", target_rows, variant="c")
            adapters["c"] = LoRARequest("c", 3, str(c_dir))
            assert llm.llm_engine.add_lora(adapters["c"])
            generate([prompts[0]] * 3, ["base", "b", "c"])
            restored = generate([prompts[0]], ["a"])[0]["tokens"]
            assert restored == baselines[(0, "a")], "A differs after GPU slot reassignment"
            report["slot_reassignment_recovered"] = True
            report["single_baselines"] = {f"{i}:{v}": t for (i, v), t in baselines.items()}
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        report["ended"] = time.time()
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in {"traceback", "cases", "single_baselines"}}))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
