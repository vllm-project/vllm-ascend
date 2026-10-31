#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path


def main() -> None:
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000")

    from vllm import LLM, SamplingParams

    model = "Qwen/Qwen2.5-0.5B-Instruct"
    out_path = Path("/vllm-workspace/vllm-ascend/deliverables/results/sleep_tp2_suspend.json")
    print("MODEL", model, flush=True)
    t0 = time.time()
    llm = LLM(
        model=model,
        tensor_parallel_size=2,
        max_model_len=512,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        enable_sleep_mode=True,
        additional_config={
            "enable_sleep_mode_extra_cleanup": True,
            "sleep_hccl_policy": "lightweight",
            "sleep_acl_graph_policy": "inplace",
            "sleep_allow_fallback": True,
            "weight_nz_mode": 0,
        },
    )
    sp = SamplingParams(max_tokens=16, temperature=0)
    prompts = ["How are you?"]
    print("GEN1", flush=True)
    o1 = llm.generate(prompts, sp)
    text1 = o1[0].outputs[0].text
    print("TEXT1", repr(text1), flush=True)
    t_sleep0 = time.time()
    llm.sleep(level=1)
    sleep_s = time.time() - t_sleep0
    print("SLEEP_OK", sleep_s, flush=True)
    t_wake0 = time.time()
    llm.wake_up()
    wake_s = time.time() - t_wake0
    print("WAKE_OK", wake_s, flush=True)
    print("GEN2", flush=True)
    o2 = llm.generate(prompts, sp)
    text2 = o2[0].outputs[0].text
    print("TEXT2", repr(text2), flush=True)
    result = {
        "case": "tp2_suspend_resume",
        "ok": bool(text1) and bool(text2),
        "sleep_s": round(sleep_s, 4),
        "wake_s": round(wake_s, 4),
        "text1": text1,
        "text2": text2,
        "elapsed_s": round(time.time() - t0, 2),
        "hccl_policy": "lightweight",
        "mode": "suspend_via_pg_resume",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result), flush=True)
    print("E2E_TP2_DONE", flush=True)


if __name__ == "__main__":
    main()
