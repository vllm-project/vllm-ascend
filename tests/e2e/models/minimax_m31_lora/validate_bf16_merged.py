# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare an independently merged BF16 model against saved public LoRA outputs."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--variant", choices=("zero", "a", "b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # The parent may also need the custom loader registry during configuration.
    import bf16_merge_loader  # noqa: F401
    import torch
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams

    torch.set_num_threads(8)
    online = json.loads((args.online / "report.json").read_text())
    reference = next(
        c["output"] for c in online["cases"] if c["profile"] == args.profile and c["variant"] == args.variant
    )
    adapter = args.online / f"{args.profile}-{args.variant}"
    llm = None
    report = {"profile": args.profile, "variant": args.variant, "started": time.time()}
    try:
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            enable_lora=False,
            load_format="m31_bf16_merged",
            model_loader_extra_config={"adapter": str(adapter)},
            enforce_eager=True,
            tensor_parallel_size=args.tp,
            data_parallel_size=1,
            language_model_only=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            skip_tokenizer_init=True,
            max_num_seqs=2,
            max_model_len=256,
            max_num_batched_tokens=256,
            block_size=128,
            gpu_memory_utilization=0.60,
            seed=20261124,
            generation_config="vllm",
            worker_extension_cls="bf16_merge_loader.MergedWorker",
            additional_config={"enable_fused_mc2": 0},
        )
        report["identity"] = llm.collective_rpc("m31_merged_identity")
        assert all(not item["lora_enabled"] for item in report["identity"])
        result = llm.generate(
            [{"prompt_token_ids": [1, 3, 4]}],
            SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, logprobs=5),
            use_tqdm=False,
        )[0].outputs[0]
        report["merged"] = {
            "tokens": list(result.token_ids),
            "logprobs": [{str(k): float(v.logprob) for k, v in s.items()} for s in result.logprobs],
        }
        report["online"] = reference
        report["tokens_exact"] = report["merged"]["tokens"] == reference["tokens"]
        report["status"] = "PASS" if report["tokens_exact"] else "FAIL"
    except Exception as exc:
        report.update(status="ERROR", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        report["ended"] = time.time()
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in {"traceback", "merged", "online"}}))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
