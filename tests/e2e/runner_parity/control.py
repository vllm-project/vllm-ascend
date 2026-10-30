# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Collect one full-model, fixed-group precision arm with optional MTP boundaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def install_schedule_recorder(worker):
    import functools

    runner = worker.model_runner
    original = runner.execute_model
    runner._parity_step_records = []
    runner._parity_case = None

    @functools.wraps(original)
    def measured(scheduler_output, *args, **kwargs):
        counts = list(scheduler_output.num_scheduled_tokens.values())
        if not counts:
            return original(scheduler_output, *args, **kwargs)
        result = original(scheduler_output, *args, **kwargs)
        runner._parity_step_records.append(
            {
                "case": runner._parity_case,
                "scheduled_tokens": sum(counts),
                "scheduled_requests": len(counts),
                "scheduled": list(scheduler_output.num_scheduled_tokens.items()),
            }
        )
        return result

    runner.execute_model = measured
    return {"rank": worker.rank}


def select_case(worker, case, timed=False):
    assert not timed, "Offline precision capture does not enable device timers"
    worker.model_runner._parity_case = case


def collect_records(worker):
    return {"rank": worker.rank, "steps": worker.model_runner._parity_step_records}


def inspect_feature_worker(worker):
    from collections import Counter

    runner = worker.model_runner
    model = runner.get_model()
    quant_methods = Counter(
        type(module.quant_method).__module__ + "." + type(module.quant_method).__name__
        for module in model.modules()
        if getattr(module, "quant_method", None) is not None
    )
    draft = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
    return {
        "rank": worker.rank,
        "runner": type(runner).__module__ + "." + type(runner).__name__,
        "draft": None if draft is None else type(draft).__module__ + "." + type(draft).__name__,
        "quant_methods": dict(quant_methods),
    }


def run_feature_boundaries(llm, inputs, output):
    """Persist each stop/cancel/context result, including failures and recovery."""
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError

    checks = []
    tokens = inputs["cases"][0]["tokens"]

    def record(name, action):
        llm.collective_rpc(select_case, args=("feature/" + name, False))
        try:
            result = action()
            row = {"name": name, "passed": True, "result": result}
        except Exception as error:
            row = {"name": name, "passed": False, "error": type(error).__name__ + ": " + str(error)}
        checks.append(row)
        output.write_text(json.dumps(checks, indent=2))
        print(json.dumps({"feature": name, "passed": row["passed"]}), flush=True)

    def bounded_generate(prompt, count):
        item = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(temperature=0, max_tokens=count, ignore_eos=True),
            use_tqdm=False,
        )[0]
        assert len(item.outputs[0].token_ids) == count
        return {
            "prompt_tokens": len(prompt),
            "token_ids": list(item.outputs[0].token_ids),
            "finish_reason": item.outputs[0].finish_reason,
        }

    record("max-output-one", lambda: bounded_generate(tokens, 1))

    def checkpoint_eos():
        eos = llm.llm_engine.model_config.hf_text_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        # Raw fixed-length token fixtures need not reach EOS. Use a complete
        # chat request to check natural termination without forcing its logits.
        prompt = llm.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": "What is 1+1? Give only the answer."}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(prompt, list) and all(isinstance(token, int) for token in prompt)
        sample = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(temperature=0, max_tokens=32, ignore_eos=False, stop_token_ids=eos_ids),
            use_tqdm=False,
        )[0].outputs[0]
        result = {
            "configured_stop_ids": eos_ids,
            "prompt_token_ids": prompt,
            "token_ids": list(sample.token_ids),
            "text": sample.text,
            "finish_reason": sample.finish_reason,
            "stop_reason": sample.stop_reason,
        }
        assert sample.finish_reason == "stop", result
        return result

    record("checkpoint-eos-stop", checkpoint_eos)

    def stop_probe():
        baseline = bounded_generate(tokens, 1)
        stop_id = baseline["token_ids"][0]
        sample = llm.generate(
            [{"prompt_token_ids": tokens}],
            SamplingParams(temperature=0, max_tokens=16, stop_token_ids=[stop_id], include_stop_str_in_output=True),
            use_tqdm=False,
        )[0].outputs[0]
        result = {
            "stop_id": stop_id,
            "token_ids": list(sample.token_ids),
            "finish_reason": sample.finish_reason,
            "stop_reason": sample.stop_reason,
        }
        assert len(sample.token_ids) == 1 and sample.finish_reason == "stop" and sample.stop_reason == stop_id, result
        return result

    record("stop-token", stop_probe)

    def cancelled():
        engine = llm.llm_engine
        request_id = engine.add_request(
            "parity-cancel", {"prompt_token_ids": tokens}, SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        )
        engine.step()
        engine.abort_request([request_id], internal=True)
        for _ in range(4):
            if not engine.has_unfinished_requests():
                break
            engine.step()
        assert not engine.has_unfinished_requests()
        recovery = bounded_generate(tokens, 4)
        return {"cancelled_internal_id": request_id, "recovery": recovery}

    record("cancel-and-recover", cancelled)
    long_tokens = inputs["cases"][4]["tokens"]
    boundary = (long_tokens * 2)[:40956]
    record("max-context-accepted", lambda: bounded_generate(boundary, 4))

    def over_limit():
        invalid = (long_tokens * 2)[:40961]
        try:
            llm.generate([{"prompt_token_ids": invalid}], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        except VLLMValidationError as error:
            message = str(error)
            assert "40961" in message and "40960" in message and "context" in message.lower(), message
            return {"prompt_tokens": len(invalid), "error_type": type(error).__name__, "error": message}
        raise AssertionError("An over-limit prompt was not rejected")

    record("over-context-rejected", over_limit)
    record("recovery-after-invalid", lambda: bounded_generate(tokens, 4))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/glm51_a5_tp8.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--features", action="store_true", help="Also exercise stop/cancel/context boundaries")
    parser.add_argument("--mtp-tokens", type=int, default=0)
    parser.add_argument("--model-parser-checks", action="store_true", help="Parse two actual chat generations")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(config["env"])
    os.environ.update(config["runner_env"][args.runner])
    assert 0 <= args.mtp_tokens <= 5
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.with_name("controlled-precision.jsonl").exists(), "Use a fresh output directory"
    # Keep engine core in the caller so the entire group is enqueued before
    # stepping. TP8 workers still use the configured multiprocessing executor.
    # Callable RPCs then stay on the executor's trusted local worker queues.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == ("0" if args.runner == "v1" else "1")
    # Runtime imports are deliberately lazy: only the isolated NPU container loads them.
    from vllm import LLM, SamplingParams

    options = {key.replace("-", "_"): value for key, value in config["args"].items()}
    for key in ("host", "port", "served_model_name", "attention_config.indexer_kv_dtype"):
        options.pop(key, None)
    options.update(
        model=args.model,
        max_model_len=40960,
        max_num_seqs=32,
        max_num_batched_tokens=args.token_budget,
        attention_config={"indexer_kv_dtype": "fp8"},
    )
    if args.mode == "graph":
        options.pop("enforce_eager", None)
        options["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
        }
    else:
        options["enforce_eager"] = True
        options["compilation_config"] = {"cudagraph_mode": "NONE"}
    (args.output.parent / "effective-options.json").write_text(json.dumps(options, indent=2))
    if args.mtp_tokens:
        options["speculative_config"] = {
            "method": "deepseek_mtp",
            "num_speculative_tokens": args.mtp_tokens,
            "enforce_eager": True,
        }
    options["disable_log_stats"] = False
    (args.output.parent / "effective-options.json").write_text(json.dumps(options, indent=2))
    llm = LLM(**options)
    assert type(llm.llm_engine.engine_core).__name__ == "InprocClient"
    print(json.dumps({"control_engine_client": "InprocClient", "tp_workers": 8}), flush=True)
    llm.collective_rpc(install_schedule_recorder)
    inputs = json.loads(args.inputs.read_text())
    (args.output.parent / "worker-feature-fingerprints.json").write_text(
        json.dumps(llm.collective_rpc(inspect_feature_worker), indent=2)
    )
    if args.model_parser_checks:
        from tests.e2e.runner_parity.model_parser_checks import run_model_parser_checks

        llm.collective_rpc(select_case, args=("parser/actual-generation", False))
        run_model_parser_checks(llm, args.output.parent / "model-parser-checks.json")
    # Enqueue the full group before stepping: remove HTTP request arrival races.
    controls = []
    for length in (128, 2048, 32768):
        cases = [case for case in inputs["cases"] if case["length"] == length]
        controls.append((f"uniform-{length}", cases + cases[:2]))
    controls.append(("mixed", [inputs["cases"][i] for i in (0, 4, 8, 12, 16, 1, 5, 9)]))
    control_path = args.output.with_name("controlled-precision.jsonl")
    for group, cases in controls:
        prompts = [{"prompt_token_ids": case["tokens"]} for case in cases]
        sampling = SamplingParams(temperature=0, seed=1024, max_tokens=32, ignore_eos=True, logprobs=20)
        llm.collective_rpc(select_case, args=("precision/" + group, False))
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        assert len(outputs) == 8
        hashes = [hashlib.sha256(json.dumps(case["tokens"], sort_keys=True).encode()).hexdigest() for case in cases]
        with control_path.open("a") as stream:
            for index, (case, output) in enumerate(zip(cases, outputs)):
                assert list(output.prompt_token_ids) == case["tokens"], "Output/input request alignment changed"
                sample = output.outputs[0]
                assert len(sample.token_ids) == 32 and sample.logprobs is not None
                row = {
                    "key": f"{group}/{index}/{case['id']}",
                    "runner": args.runner,
                    "output_request_id": output.request_id,
                    "input_sha256": hashes[index],
                    "request": {
                        "temperature": 0,
                        "seed": 1024,
                        "max_tokens": 32,
                        "logprobs": 20,
                        "ignore_eos": True,
                        "speculative_tokens": args.mtp_tokens,
                        "declared_group_size": 8,
                        "group": group,
                        "batch_input_hashes": hashes,
                        "index": index,
                    },
                    "response": {
                        "usage": {"prompt_tokens": len(case["tokens"]), "completion_tokens": 32},
                        "choices": [
                            {
                                "text": sample.text,
                                "token_ids": list(sample.token_ids),
                                "logprobs": {
                                    "top_logprobs": [
                                        {f"token_id:{token}": value.logprob for token, value in step.items()}
                                        for step in sample.logprobs
                                    ]
                                },
                            }
                        ],
                    },
                }
                stream.write(json.dumps(row) + "\n")
        print(json.dumps({"controlled_precision_completed": group}), flush=True)
    checks = run_feature_boundaries(llm, inputs, args.output.parent / "feature-checks.json") if args.features else []
    import dataclasses

    metrics = [dataclasses.asdict(value) for value in llm.get_metrics()]
    (args.output.parent / "metrics-final.json").write_text(json.dumps(metrics, indent=2))
    spec = {value["name"]: value["value"] for value in metrics if "spec_decode" in value["name"] and "value" in value}
    (args.output.parent / "speculative-evidence.json").write_text(json.dumps(spec, indent=2))
    records = llm.collective_rpc(collect_records)
    args.output.write_text(
        json.dumps(
            {
                "runner": args.runner,
                "records": records,
                "scope": "Control precision metadata only; no NPU timing events.",
            },
            indent=2,
        )
    )
    assert all(row["passed"] for row in checks), "Feature boundary failures; inspect preserved feature-checks.json"
    if args.mtp_tokens:
        assert spec.get("vllm:spec_decode_num_draft_tokens", 0) > 0, spec
        assert spec.get("vllm:spec_decode_num_accepted_tokens", 0) > 0, spec
    return


if __name__ == "__main__":
    main()
