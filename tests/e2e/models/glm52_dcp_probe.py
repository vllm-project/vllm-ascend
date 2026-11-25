# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-checkpoint GLM-5.2 DCP probe; compare reports from identical TP/DP/PP.

Example on 16 A3 dies (run DCP1 first, then DCP8 without changing other flags)::

    python tests/e2e/models/glm52_dcp_probe.py --model /models/GLM-5.2-w8a8c8 \
        --tp 8 --dp 2 --dcp 1 --output /results/tp8-dcp1.json
    python tests/e2e/models/glm52_dcp_probe.py --model /models/GLM-5.2-w8a8c8 \
        --tp 8 --dp 2 --dcp 8 --baseline /results/tp8-dcp1.json \
        --output /results/tp8-dcp8.json

Build the matching vLLM Ascend native operators before starting the probe.
Use run_job.py or another external monitor to record device memory separately.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

RUNTIME_ENV_KEYS = (
    "LCCL_DETERMINISTIC",
    "HCCL_DETERMINISTIC",
    "HCCL_ALGO",
    "HCCL_BUFFSIZE",
    "PYTORCH_NPU_ALLOC_CONF",
    "VLLM_ENGINE_READY_TIMEOUT_S",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
    "ASCEND_LAUNCH_BLOCKING",
)


def build_cases(model, lengths):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    samples = {
        "english": ("A distributed system preserves the order of its events. ", "Explain tensor parallelism briefly."),
        "chinese": ("分布式推理需要保持每个请求的顺序和上下文。", "用一句话解释张量并行。"),
        "code": (
            "def identity(value):\n    return value\n\n",
            "Complete this function:\ndef binary_search(arr, target):",
        ),
    }
    cases = {}
    for name, (unit, question) in samples.items():
        short = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        assert isinstance(short, list), "Tokenizer must return a flat token ID list"
        cases[f"{name}-short"] = {"prompt": list(short), "kind": name, "input_tokens": len(short)}
        unit_length = len(tokenizer.encode(unit, add_special_tokens=False))
        for length in lengths:
            repetitions = length // unit_length + 100
            # Adjacent copies can merge into fewer tokens than isolated units.
            # Size from actual chat tokenization, not a linear estimate.
            for _ in range(8):
                text = unit * repetitions + "\n" + question
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                assert isinstance(tokens, list), "Tokenizer must return a flat token ID list"
                if len(tokens) >= length:
                    break
                repetitions *= 2
            else:
                raise ValueError(f"Tokenizer did not produce {length} tokens after repeated expansion")
            # Remove repeated context in the middle, preserving the chat prefix,
            # question, and assistant-generation suffix at an exact token length.
            tokens = list(tokens[: length - 128]) + list(tokens[-128:])
            assert len(tokens) == length
            cases[f"{name}-{length}"] = {"prompt": tokens, "kind": name, "input_tokens": len(tokens)}
    for case in cases.values():
        case["prompt_sha256"] = hashlib.sha256(json.dumps(case["prompt"]).encode()).hexdigest()
    return cases


def command_for(args):
    command = [
        "vllm",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "glm52",
        "--trust-remote-code",
        "--quantization",
        "ascend",
        "--tensor-parallel-size",
        str(args.tp),
        "--data-parallel-size",
        str(args.dp),
        "--pipeline-parallel-size",
        str(args.pp),
        "--prefill-context-parallel-size",
        str(args.pcp),
        "--decode-context-parallel-size",
        str(args.dcp),
        "--enable-expert-parallel",
        "--max-model-len",
        str(max(args.lengths) + 512),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_bytes),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--attention_config.indexer_kv_dtype",
        args.indexer_kv_dtype,
        "--seed",
        "42",
        "--additional-config",
        json.dumps({"enable_fused_mc2": 0, "enable_mlapo": not args.disable_mlapo}),
    ]
    if args.graph:
        command += ["--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}']
    else:
        command += ["--enforce-eager"]
    if args.hf_overrides is not None:
        command += ["--hf-overrides", args.hf_overrides]
    if args.mtp:
        command += ["--speculative-config", '{"method":"deepseek_mtp","num_speculative_tokens":3,"enforce_eager":true}']
    return command


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness with status {process.returncode}")
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            response.raise_for_status()
            models = response.json()
            if not any(item["id"] == "glm52" for item in models["data"]):
                raise RuntimeError("Port is serving an unrelated model")
            return models
        except requests.RequestException:
            time.sleep(3)
    raise TimeoutError("Server readiness timed out")


def probe_requests(cases, args, report):
    with requests.Session() as session:
        session.headers["X-data-parallel-rank"] = str(args.request_dp_rank)
        for name, case in cases.items():
            for sampling, temperature in (("greedy", 0), ("sampled", 0.6)):
                start = time.perf_counter()
                response = session.post(
                    f"http://127.0.0.1:{args.port}/v1/completions",
                    json={
                        "model": "glm52",
                        "prompt": case["prompt"],
                        "max_tokens": 32,
                        "temperature": temperature,
                        "top_p": 1 if temperature == 0 else 0.9,
                        "seed": 42,
                        "logprobs": 5,
                        "return_tokens_as_token_ids": True,
                    },
                    timeout=600,
                )
                response.raise_for_status()
                result = response.json()
                assert result["choices"][0]["logprobs"]["tokens"], "Empty generation"
                report["results"][f"{name}-{sampling}"] = {
                    "response": result,
                    "seconds": time.perf_counter() - start,
                    "input_tokens": case["input_tokens"],
                    "prompt_sha256": case["prompt_sha256"],
                }
                args.output.write_text(json.dumps(report, indent=2))
                print(name, sampling, "PASS", flush=True)


def compare_baseline(report, path):
    baseline = json.loads(path.read_text())
    for field in (
        "model",
        "tp",
        "dp",
        "pp",
        "pp_layer_partition",
        "pcp",
        "mtp",
        "graph",
        "max_num_batched_tokens",
        "kv_cache_bytes",
        "request_dp_rank",
        "kv_cache_dtype",
        "indexer_kv_dtype",
        "disable_mlapo",
        "hf_overrides",
        "runtime_env",
    ):
        assert field in baseline, f"Baseline lacks {field}; rerun with the same probe protocol"
        assert report[field] == baseline[field], f"Baseline differs in {field}"
    assert baseline["dcp"] == 1 and baseline["passed"], "Baseline must be a successful DCP1 run"
    assert report["results"].keys() == baseline["results"].keys(), "Baseline case set differs"
    comparisons = {}
    for name, actual in report["results"].items():
        reference = baseline["results"][name]
        assert actual["prompt_sha256"] == reference["prompt_sha256"], f"Prompt differs: {name}"
        actual_probs = actual["response"]["choices"][0]["logprobs"]
        reference_probs = reference["response"]["choices"][0]["logprobs"]
        actual_tokens, reference_tokens = actual_probs["tokens"], reference_probs["tokens"]
        exact = actual_tokens == reference_tokens
        if name.endswith("-greedy"):
            passed = exact
        else:
            # Use the repository's first-divergence mutual top-k rule. Later
            # positions no longer share a prefix and cannot compare logits.
            passed = exact
            for index, (actual_token, reference_token) in enumerate(zip(actual_tokens, reference_tokens)):
                if actual_token != reference_token:
                    passed = (
                        actual_token in reference_probs["top_logprobs"][index]
                        and reference_token in actual_probs["top_logprobs"][index]
                    )
                    break
        comparisons[name] = {"passed": passed, "exact_tokens": exact}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--dcp", type=int, default=1)
    parser.add_argument("--request-dp-rank", type=int, default=0)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--kv-cache-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--kv-cache-dtype", choices=("int8", "auto"), default="int8")
    parser.add_argument("--indexer-kv-dtype", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--disable-mlapo", action="store_true")
    parser.add_argument(
        "--hf-overrides",
        type=json.loads,
        help="JSON model-config overrides for a labeled diagnostic run; require the same value in both runs.",
    )
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.request_dp_rank < args.dp:
        parser.error("--request-dp-rank must be within the configured DP group")
    if args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if args.hf_overrides is not None:
        if not isinstance(args.hf_overrides, dict):
            parser.error("--hf-overrides must decode to a JSON object")
        args.hf_overrides = json.dumps(args.hf_overrides, sort_keys=True, separators=(",", ":"))
    if args.dcp > 1 and args.baseline is None and not args.prepare_only:
        parser.error("A DCP>1 run requires --baseline from the same TP/DP/PP with DCP1")
    cases = build_cases(args.model, args.lengths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".prompts.json").write_text(json.dumps(cases, indent=2))
    command = command_for(args)
    report = {
        "command": command,
        "results": {},
        "model": args.model,
        "tp": args.tp,
        "dp": args.dp,
        "pp": args.pp,
        "pp_layer_partition": os.environ.get("VLLM_PP_LAYER_PARTITION"),
        "pcp": args.pcp,
        "dcp": args.dcp,
        "mtp": args.mtp,
        "graph": args.graph,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_bytes": args.kv_cache_bytes,
        "kv_cache_dtype": args.kv_cache_dtype,
        "indexer_kv_dtype": args.indexer_kv_dtype,
        "disable_mlapo": args.disable_mlapo,
        "hf_overrides": args.hf_overrides,
        "request_dp_rank": args.request_dp_rank,
        "runtime_env": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": False,
    }
    if args.prepare_only:
        args.output.write_text(json.dumps(report, indent=2))
        return
    # Never send validation requests to an existing service on a shared host.
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    with args.output.with_suffix(".serve.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            start = time.monotonic()
            report["models"] = wait_ready(process, args.port, args.startup_timeout)
            report["startup_seconds"] = time.monotonic() - start
            probe_requests(cases, args, report)
            report["passed"] = True
            if args.baseline:
                report["baseline"] = str(args.baseline)
                report["comparisons"] = compare_baseline(report, args.baseline)
                report["passed"] = all(case["passed"] for case in report["comparisons"].values())
                if not report["passed"]:
                    raise AssertionError("DCP output comparison failed; see per-case comparisons")
        except Exception as error:
            report["error"] = repr(error)
            report["passed"] = False
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2))
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
