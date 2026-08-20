import multiprocessing
import os
import queue
import traceback
from contextlib import nullcontext

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.pull_request.accuracy_probe import (
    DETERMINISTIC_SEED,
    apply_deterministic_settings,
    deterministic_test_scope,
    print_runtime_state,
)
from tests.e2e.pull_request.four_card import test_graph_mode as baseline


def _make_sampling_params(seed: int | None) -> SamplingParams:
    kwargs = {
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "logprobs": 20,
    }
    if seed is not None:
        kwargs["seed"] = seed
    return SamplingParams(**kwargs)


def _extract_outputs(outputs) -> list[dict]:
    extracted = []
    for output in outputs:
        sequence = output.outputs[0]
        extracted.append(
            {
                "text": sequence.text,
                "token_ids": list(sequence.token_ids),
                "logprobs": [
                    {token_id: logprob.logprob for token_id, logprob in step.items()} for step in sequence.logprobs
                ]
                if sequence.logprobs
                else None,
            }
        )
    return extracted


def _run_worker_process(
    rank: int,
    world_size: int,
    cur_case: dict,
    execution_mode: str,
    deterministic: bool,
    master_port: int,
    result_queue: multiprocessing.Queue,
) -> None:
    os.environ.update(
        {
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_SIZE": str(world_size),
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(master_port),
        }
    )
    for key, value in cur_case.get("env_vars", {}).items():
        os.environ[key] = str(value)
    if deterministic:
        os.environ["HCCL_DETERMINISTIC"] = "true"
        apply_deterministic_settings()

    try:
        print_runtime_state(
            "graph-worker-start",
            {
                "rank": rank,
                "world_size": world_size,
                "execution_mode": execution_mode,
                "deterministic_probe": deterministic,
                "model": cur_case["model"],
                "quantization": cur_case["quantization"],
                "tensor_parallel_size": cur_case["tensor_parallel_size"],
            },
        )
        llm_kwargs = {
            "model": cur_case["model"],
            "max_model_len": 1024,
            "quantization": cur_case["quantization"],
            "tensor_parallel_size": cur_case["tensor_parallel_size"],
            "enable_expert_parallel": cur_case["enable_expert_parallel"],
            "trust_remote_code": True,
        }
        seed = DETERMINISTIC_SEED if deterministic else None
        if seed is not None:
            llm_kwargs["seed"] = seed
        if execution_mode == "eager":
            llm_kwargs["enforce_eager"] = True
        elif execution_mode == "graph":
            llm_kwargs["compilation_config"] = cur_case["compilation_config"]
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

        print(
            f"[accuracy-probe][graph-phase] rank={rank} mode={execution_mode} phase=llm-start",
            flush=True,
        )
        llm = LLM(**llm_kwargs)
        print(
            f"[accuracy-probe][graph-phase] rank={rank} mode={execution_mode} phase=llm-ready",
            flush=True,
        )

        sampling_params = _make_sampling_params(seed)
        prompts_short = cur_case["prompts"]["short"]
        short_chunk_size = len(prompts_short) // world_size
        short_start = rank * short_chunk_size
        short_end = short_start + short_chunk_size if rank < world_size - 1 else len(prompts_short)

        prompts_long = cur_case["prompts"]["long"]
        long_chunk_size = len(prompts_long) // world_size
        long_start = rank * long_chunk_size
        long_end = long_start + long_chunk_size if rank < world_size - 1 else len(prompts_long)

        print(
            f"[accuracy-probe][graph-phase] rank={rank} mode={execution_mode} phase=generate-short-start",
            flush=True,
        )
        outputs_short = llm.generate(prompts_short[short_start:short_end], sampling_params)
        print(
            f"[accuracy-probe][graph-phase] rank={rank} mode={execution_mode} phase=generate-long-start",
            flush=True,
        )
        outputs_long = llm.generate(prompts_long[long_start:long_end], sampling_params)
        print(
            f"[accuracy-probe][graph-phase] rank={rank} mode={execution_mode} phase=generate-finished",
            flush=True,
        )
        result_queue.put(
            {
                "rank": rank,
                "mode": execution_mode,
                "short": {"prompt_idx": short_start, "outputs": _extract_outputs(outputs_short)},
                "long": {"prompt_idx": long_start, "outputs": _extract_outputs(outputs_long)},
            }
        )
    except BaseException:
        result_queue.put(
            {
                "rank": rank,
                "mode": execution_mode,
                "error": traceback.format_exc(),
            }
        )
        raise


def _run_worker_group(cur_case: dict, execution_mode: str, deterministic: bool) -> list[dict]:
    world_size = cur_case["data_parallel_size"]
    result_queue: multiprocessing.Queue[dict] = multiprocessing.Queue()
    master_port = get_open_port()
    workers = []
    for rank in range(world_size):
        process = multiprocessing.Process(
            target=_run_worker_process,
            args=(rank, world_size, cur_case, execution_mode, deterministic, master_port, result_queue),
        )
        process.start()
        workers.append(process)

    results = []
    timed_out = False
    for _ in range(world_size):
        try:
            results.append(result_queue.get(timeout=360))
        except queue.Empty:
            timed_out = True
            break

    failed_workers = []
    for process in workers:
        process.join(timeout=60)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.exitcode != 0:
            failed_workers.append((process.pid, process.exitcode))

    errors = [result for result in results if "error" in result]
    if timed_out or failed_workers or errors:
        details = "\n".join(result["error"] for result in errors)
        raise RuntimeError(
            f"{execution_mode} worker phase failed: timed_out={timed_out}, failed_workers={failed_workers}\n{details}"
        )
    assert len(results) == world_size
    return sorted(results, key=lambda result: result["rank"])


def _top_values(step_logprobs: dict[int, float], limit: int = 5) -> list[tuple[int, float]]:
    return sorted(step_logprobs.items(), key=lambda item: item[1], reverse=True)[:limit]


def _dump_results(label: str, results: list[dict]) -> None:
    for result in results:
        for prompt_group in ("short", "long"):
            grouped_result = result[prompt_group]
            for local_idx, output in enumerate(grouped_result["outputs"]):
                prompt_idx = grouped_result["prompt_idx"] + local_idx
                print(
                    f"[accuracy-probe][graph-output] label={label} rank={result['rank']} "
                    f"group={prompt_group} prompt={prompt_idx} token_ids={output['token_ids']} "
                    f"text={output['text']!r}",
                    flush=True,
                )
                if output["logprobs"] is None:
                    continue
                for token_idx, step_logprobs in enumerate(output["logprobs"]):
                    print(
                        f"[accuracy-probe][graph-topk] label={label} rank={result['rank']} "
                        f"group={prompt_group} prompt={prompt_idx} token={token_idx} "
                        f"values={_top_values(step_logprobs)}",
                        flush=True,
                    )


def _check_dynamic_output(baseline_output: dict, candidate_output: dict, prompt_idx: int) -> None:
    baseline_ids = baseline_output["token_ids"]
    candidate_ids = candidate_output["token_ids"]
    baseline_logprobs = baseline_output["logprobs"]
    candidate_logprobs = candidate_output["logprobs"]
    assert baseline_logprobs is not None and candidate_logprobs is not None
    assert len(baseline_ids) == len(candidate_ids) == 16

    decode_atol = 2 * baseline.ATOL
    for token_idx, (baseline_id, candidate_id) in enumerate(zip(baseline_ids, candidate_ids, strict=True)):
        baseline_topk = baseline_logprobs[token_idx]
        candidate_topk = candidate_logprobs[token_idx]
        baseline_value = baseline_topk[baseline_id]
        tolerance = baseline.ATOL if token_idx == 0 else decode_atol

        if token_idx == 0:
            assert baseline_id == candidate_id, (
                f"Graph prefill token mismatch at prompt {prompt_idx}: "
                f"eager_token={baseline_id}, graph_token={candidate_id}, "
                f"eager_topk={_top_values(baseline_topk)}, "
                f"graph_topk={_top_values(candidate_topk)}"
            )
            candidate_value = candidate_topk[candidate_id]
            assert abs(baseline_value - candidate_value) <= tolerance, (
                f"Graph prefill logprob mismatch at prompt {prompt_idx}: "
                f"eager={baseline_value:.4f}, graph={candidate_value:.4f}, "
                f"diff={abs(baseline_value - candidate_value):.4f} > tolerance={tolerance}"
            )
            continue

        if baseline_id != candidate_id:
            print(
                f"[accuracy-probe][graph-divergence] prompt={prompt_idx} token={token_idx} "
                f"eager_token={baseline_id} graph_token={candidate_id} "
                f"eager_topk={_top_values(baseline_topk)} graph_topk={_top_values(candidate_topk)}",
                flush=True,
            )

        if baseline_id == candidate_id:
            candidate_value = candidate_topk[candidate_id]
            assert abs(baseline_value - candidate_value) <= tolerance, (
                f"Graph logprob mismatch at prompt {prompt_idx}, token {token_idx}: "
                f"eager={baseline_value:.4f}, graph={candidate_value:.4f}, "
                f"diff={abs(baseline_value - candidate_value):.4f} > tolerance={tolerance}"
            )
        else:
            assert baseline_id in candidate_topk, (
                f"Eager token {baseline_id} missing from graph top-20 at prompt {prompt_idx}, token {token_idx}"
            )
            candidate_value = candidate_topk[baseline_id]
            assert abs(baseline_value - candidate_value) <= tolerance, (
                f"Graph distribution mismatch at prompt {prompt_idx}, token {token_idx}: "
                f"eager_token={baseline_id}, eager={baseline_value:.4f}, "
                f"graph={candidate_value:.4f}, "
                f"diff={abs(baseline_value - candidate_value):.4f} > tolerance={tolerance}"
            )
            # Later autoregressive tokens no longer share the same context.
            # Stop after validating the first distribution-level divergence.
            break


def _compare_dynamic_results(eager_results: list[dict], graph_results: list[dict]) -> None:
    for eager_result, graph_result in zip(eager_results, graph_results, strict=True):
        assert eager_result["rank"] == graph_result["rank"]
        for prompt_group in ("short", "long"):
            eager_group = eager_result[prompt_group]
            graph_group = graph_result[prompt_group]
            assert eager_group["prompt_idx"] == graph_group["prompt_idx"]
            for local_idx, (eager_output, graph_output) in enumerate(
                zip(eager_group["outputs"], graph_group["outputs"], strict=True)
            ):
                prompt_idx = eager_group["prompt_idx"] + local_idx
                _check_dynamic_output(eager_output, graph_output, prompt_idx)


def run_static_golden_deterministic_probe(cur_case: dict, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with deterministic_test_scope(monkeypatch):
        results = _run_worker_group(cur_case, "graph", deterministic=True)
        baseline._exit()

    _dump_results("graph-deterministic-vs-static-golden", results)
    decode_atol = 2 * baseline.ATOL
    for result in results:
        baseline.check_accuracy(cur_case["golden_answers"]["short"], result["short"], baseline.ATOL, decode_atol)
        baseline.check_accuracy(cur_case["golden_answers"]["long"], result["long"], baseline.ATOL, decode_atol)


def run_same_runner_eager_graph_probe(cur_case: dict, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with nullcontext():
        eager_results = _run_worker_group(cur_case, "eager", deterministic=False)
        baseline._exit()
        graph_results = _run_worker_group(cur_case, "graph", deterministic=False)
        baseline._exit()

    _dump_results("graph-diagnostics-eager", eager_results)
    _dump_results("graph-diagnostics-graph", graph_results)
    _compare_dynamic_results(eager_results, graph_results)
