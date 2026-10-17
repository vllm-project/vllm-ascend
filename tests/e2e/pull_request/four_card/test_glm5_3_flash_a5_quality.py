# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 DP4/TP1/EP synthetic numerical and steady-state performance guard.

Keep production tensor widths and experts in the nine-layer MXFP8 smoke model.
Frozen raw probabilities guard numerical regressions, not checkpoint accuracy.
Time 3500 input / 128 output tokens with global batch four (one per DP rank),
including prefill and DP dispatch but excluding startup and graph capture.
"""

import contextlib
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner
from tests.e2e.pull_request.four_card.glm53flash_quality import (
    assert_performance,
    compare_snapshots,
    summarize_outputs,
    summarize_performance,
)
from tests.e2e.pull_request.four_card.test_glm5_3_flash import _write_model
from tests.e2e.pull_request.four_card.test_glm5_3_flash_a5 import (
    DP_SIZE,
    INIT_CHUNK_SIZE,
    LOADER,
    MXFP8_SCALE_EXPONENT,
    MXFP8_SCHEME,
    FlashA5DummyLoader,
    FlashA5Worker,
    _parameter_seed,
    _write_a5_model,
)

BASELINE_FILE = Path(__file__).with_name("glm53flash_assets") / "quality_a5_baseline.json"
ACCURACY_TOKENS = 16
ACCURACY_CONTEXT_TOKEN = 42
ACCURACY_LOGPROBS = 64
NUMERICAL_SCENARIOS = ((127, 0), (128, 137), (129, 274), (513, 0))
PERFORMANCE_INPUT_TOKENS = 3500
PERFORMANCE_OUTPUT_TOKENS = 128
PERFORMANCE_BATCH_SIZE = 4
WARMUP_RUNS = 2
MEASURED_RUNS = 5
# These existing runtime options match the A5 smoke and are part of the frozen
# protocol. Calibration callers must also set them before engine construction.
QUALITY_ENVIRONMENT = (
    ("VLLM_USE_V2_MODEL_RUNNER", "0"),
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
    ("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"),
    ("HCCL_OP_EXPANSION_MODE", "AIV"),
    ("HCCL_BUFFSIZE", "1024"),
    ("OMP_NUM_THREADS", "1"),
    ("TASK_QUEUE_ENABLE", "1"),
)


class FlashA5QualityWorker(FlashA5Worker):
    """Count real graph replays only during the numerical scenarios."""

    def start_replay_probe(self):
        self._flash_replays = 0
        self._flash_original_replay = torch.npu.NPUGraph.replay

        def replay(graph, *args, **kwargs):
            self._flash_replays += 1
            return self._flash_original_replay(graph, *args, **kwargs)

        torch.npu.NPUGraph.replay = replay
        return True

    def stop_replay_probe(self):
        torch.npu.NPUGraph.replay = self._flash_original_replay
        return self._flash_replays

    def read_replay_probe(self):
        return self._flash_replays

    def check_raw_logprobs(self):
        assert self.model_runner.sampler.logprobs_mode == "raw_logprobs"
        return True


def _prompt(length, salt=0):
    return [10 + (index * 17 + salt) % 1000 for index in range(length)]


def _parameters(max_tokens):
    return SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, detokenize=False)


def _probe_token_ids(vocab_size):
    return [ACCURACY_CONTEXT_TOKEN] + [
        index * (vocab_size - 1) // (ACCURACY_LOGPROBS - 2) for index in range(ACCURACY_LOGPROBS - 1)
    ]


def _accuracy_parameters(vocab_size):
    # Fix the decode context even when synthetic logits have near-tied argmaxes.
    # raw_logprobs must precede the allowed-token mask applied for sampling.
    return SamplingParams(
        temperature=0,
        max_tokens=ACCURACY_TOKENS,
        ignore_eos=True,
        detokenize=False,
        logprobs=ACCURACY_LOGPROBS,
        logprob_token_ids=_probe_token_ids(vocab_size),
        allowed_token_ids=[ACCURACY_CONTEXT_TOKEN],
    )


def _check_completions(outputs, prompts, output_tokens, vocab_size):
    assert len(outputs) == len(prompts)
    for output, prompt in zip(outputs, prompts):
        assert output.finished and len(output.outputs) == 1
        assert list(output.prompt_token_ids) == prompt
        tokens = output.outputs[0].token_ids
        assert len(tokens) == output_tokens
        assert all(0 <= token < vocab_size for token in tokens)


def _rank_counts(results):
    assert len(results) == DP_SIZE, "Missing DP rank replay results"
    assert all(isinstance(rank, list) and len(rank) == 1 for rank in results), "Expected TP1 for every DP rank"
    counts = [rank[0] for rank in results]
    assert all(type(count) is int and count >= 0 for count in counts)
    return counts


def _engine_settings(enforce_eager):
    settings = dict(
        load_format=LOADER,
        worker_extension_cls=f"{__name__}.FlashA5QualityWorker",
        dtype="bfloat16",
        data_parallel_size=DP_SIZE,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        quantization="ascend",
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_logprobs=ACCURACY_LOGPROBS,
        logprobs_mode="raw_logprobs",
        limit_mm_per_prompt={"image": 0, "video": 0},
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=1024**3,
        seed=1024,
        enforce_eager=enforce_eager,
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0},
    )
    if not enforce_eager:
        settings["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        }
    return settings


def quality_protocol():
    """Bind the A5 baseline to its initializer, writers, assets and workload."""
    assets = BASELINE_FILE.parent
    names = (
        "config.json",
        "multimodal.json",
        "quant.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    data = {name: json.loads((assets / name).read_text(encoding="utf-8")) for name in names}
    initializers = (
        FlashA5DummyLoader.load_weights,
        FlashA5DummyLoader.initialize_parameter,
        _parameter_seed,
        _write_model,
        _write_a5_model,
        _prompt,
    )
    data["source"] = [inspect.getsource(function).replace("\r\n", "\n") for function in initializers]
    data["mxfp8"] = {"scheme": MXFP8_SCHEME, "scale_exponent": MXFP8_SCALE_EXPONENT, "init_chunk_size": INIT_CHUNK_SIZE}
    return {
        "schema_version": 1,
        "platform": "A5",
        "model_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        "sampling_sha256": hashlib.sha256(
            (inspect.getsource(_parameters) + inspect.getsource(_accuracy_parameters)).replace("\r\n", "\n").encode()
        ).hexdigest(),
        "engines": {mode: _engine_settings(mode == "eager") for mode in ("eager", "graph")},
        "environment": dict(QUALITY_ENVIRONMENT),
        "numerical_output_tokens": ACCURACY_TOKENS,
        "numerical_context_token": ACCURACY_CONTEXT_TOKEN,
        "numerical_logprobs": ACCURACY_LOGPROBS,
        "numerical_probe_token_ids": _probe_token_ids(data["config.json"]["vocab_size"]),
        "numerical_prompts": [list(scenario) for scenario in NUMERICAL_SCENARIOS],
        "numerical_requests_per_scenario": DP_SIZE,
        "numerical_result_order": "scenario_then_dp_rank",
        "performance_input_tokens": PERFORMANCE_INPUT_TOKENS,
        "performance_output_tokens": PERFORMANCE_OUTPUT_TOKENS,
        "performance_batch_size": PERFORMANCE_BATCH_SIZE,
        "performance_requests_per_dp_rank": PERFORMANCE_BATCH_SIZE // DP_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
    }


def collect_numerical(runner, vocab_size, enforce_eager):
    assert runner.collective_rpc("check_raw_logprobs") == [[True]] * DP_SIZE
    snapshots = []
    scenario_replays = []
    assert runner.collective_rpc("start_replay_probe") == [[True]] * DP_SIZE
    try:
        for length, salt in NUMERICAL_SCENARIOS:
            # One identical request per DP rank fixes quantized kernel shapes
            # and retains every rank's numerical result. Do not rely on dummy
            # requests used internally by DPVllmRunner for otherwise idle ranks.
            prompts = [_prompt(length, salt=salt) for _ in range(DP_SIZE)]
            outputs = runner.generate_raw(prompts, _accuracy_parameters(vocab_size), use_tqdm=False)
            _check_completions(outputs, prompts, ACCURACY_TOKENS, vocab_size)
            snapshot = summarize_outputs(outputs)
            for request in snapshot:
                assert request["token_ids"] == [ACCURACY_CONTEXT_TOKEN] * ACCURACY_TOKENS
                for step in request["logprobs"]:
                    assert set(step) == {str(token) for token in _probe_token_ids(vocab_size)}
                    assert step[str(ACCURACY_CONTEXT_TOKEN)] < 0, "Masked logprobs cannot be a numerical reference"
            snapshots.extend(snapshot)
            scenario_replays.append(_rank_counts(runner.collective_rpc("read_replay_probe")))
    except BaseException:
        # A failed DP request can already have stopped all workers and cleared
        # its connections. Cleanup must not replace the original engine error
        # with an empty replay-count assertion or a second transport failure.
        with contextlib.suppress(Exception):
            runner.collective_rpc("stop_replay_probe")
        raise
    else:
        replay_counts = _rank_counts(runner.collective_rpc("stop_replay_probe"))
    if enforce_eager:
        assert replay_counts == [0] * DP_SIZE, replay_counts
    else:
        previous = [0] * DP_SIZE
        for counts in scenario_replays:
            assert all(current > old for current, old in zip(counts, previous)), (
                "A numerical scenario did not replay graphs on every DP rank"
            )
            previous = counts
    return {"numerical": snapshots, "replay_counts": replay_counts, "scenario_replays": scenario_replays}


def collect_quality(model: Path, enforce_eager: bool):
    """Collect independently without reading or writing a frozen baseline.

    Set ``os.environ.update(dict(QUALITY_ENVIRONMENT))`` before calling. Use a
    fresh model directory and a separate engine invocation per calibration run.
    The pytest entry below always requires a separately reviewed A5 baseline.
    """
    for name, value in QUALITY_ENVIRONMENT:
        assert os.environ.get(name) == value, f"Set {name}={value} before A5 calibration"
    config = _write_a5_model(model)
    vocab_size = config["text_config"]["vocab_size"]
    assert PERFORMANCE_BATCH_SIZE == DP_SIZE, "This workload requires exactly one request per DP rank"
    with DPVllmRunner(str(model), **_engine_settings(enforce_eager)) as runner:
        assert runner.collective_rpc("check_flash_a5_paths") == [[True]] * DP_SIZE
        numerical = collect_numerical(runner, vocab_size, enforce_eager)
        prompts = [_prompt(PERFORMANCE_INPUT_TOKENS, salt=index * 137) for index in range(PERFORMANCE_BATCH_SIZE)]
        params = _parameters(PERFORMANCE_OUTPUT_TOKENS)
        elapsed = []
        for run in range(WARMUP_RUNS + MEASURED_RUNS):
            start = time.perf_counter()
            outputs = runner.generate_raw(prompts, params, use_tqdm=False)
            seconds = time.perf_counter() - start
            _check_completions(outputs, prompts, PERFORMANCE_OUTPUT_TOKENS, vocab_size)
            print(
                f"Flash A5 quality {'eager' if enforce_eager else 'graph'} run={run} seconds={seconds:.6f}", flush=True
            )
            if run >= WARMUP_RUNS:
                elapsed.append(seconds)
        performance = summarize_performance(
            elapsed, output_tokens_per_run=PERFORMANCE_BATCH_SIZE * PERFORMANCE_OUTPUT_TOKENS
        )
    return {"protocol": quality_protocol(), **numerical, "performance": performance}


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
def test_glm53flash_a5_dp4_quality(tmp_path, monkeypatch, enforce_eager, record_property):
    for name, value in QUALITY_ENVIRONMENT:
        monkeypatch.setenv(name, value)
    # A missing baseline is a failure, never a skip or automatic calibration.
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    assert baseline["protocol"] == quality_protocol(), "A5 model/workload changed: independently recalibrate and review"
    mode = "eager" if enforce_eager else "graph"
    observed = collect_quality(tmp_path / "model", enforce_eager)
    observed_json = json.dumps(observed, indent=2)
    (tmp_path / "observed.json").write_text(observed_json, encoding="utf-8")
    if runner_temp := os.environ.get("RUNNER_TEMP"):
        artifacts = Path(runner_temp) / "selected-tests-glm53flash-a5-quality"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / f"{mode}.json").write_text(observed_json, encoding="utf-8")
    record_property("glm53flash_a5_performance", json.dumps(observed["performance"]))
    print(json.dumps({"mode": mode, "performance": observed["performance"], "replays": observed["replay_counts"]}))
    compare_snapshots(observed["numerical"], baseline["numerical"], atol=baseline["logprob_atol"])
    assert_performance(observed["performance"], minimum_output_tokens_s=baseline["minimum_output_tokens_s"][mode])
