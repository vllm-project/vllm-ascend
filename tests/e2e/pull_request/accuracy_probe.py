import os
import platform
import random
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.utils import _check_decode_token, _check_prefill_token

DETERMINISTIC_SEED = 0
_COMPILATION_KEYS = {"compilation_config", "additional_config", "cudagraph_capture_sizes"}
_DECODE_TOPK = 20


def apply_deterministic_settings(seed: int = DETERMINISTIC_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


@contextmanager
def deterministic_test_scope(monkeypatch, seed: int = DETERMINISTIC_SEED):
    """Apply deterministic settings to pytest and future Python workers."""
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    bootstrap_dir = Path(__file__).resolve().parent / "deterministic_bootstrap"
    existing_pythonpath = os.getenv("PYTHONPATH")
    pythonpath = str(bootstrap_dir)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"

    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("PYTHONHASHSEED", str(seed))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    apply_deterministic_settings(seed)
    print_runtime_state("deterministic-scope-enabled")
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)


def print_runtime_state(label: str, extra: dict | None = None) -> None:
    state = {
        "label": label,
        "pid": os.getpid(),
        "runner_name": os.getenv("RUNNER_NAME"),
        "runner_arch": os.getenv("RUNNER_ARCH"),
        "platform": platform.platform(),
        "hccl_deterministic": os.getenv("HCCL_DETERMINISTIC"),
        "python_hash_seed": os.getenv("PYTHONHASHSEED"),
        "enpu_enable": os.getenv("ENPU_ENABLE"),
        "worker_multiproc_method": os.getenv("VLLM_WORKER_MULTIPROC_METHOD"),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "torch_initial_seed": torch.initial_seed(),
    }
    if extra:
        state.update(extra)
    print(f"[accuracy-probe][runtime] {pformat(state, sort_dicts=True)}", flush=True)


def _make_sampling_params(seed: int | None) -> SamplingParams:
    kwargs = {
        "max_tokens": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "logprobs": _DECODE_TOPK,
    }
    if seed is not None:
        kwargs["seed"] = seed
    return SamplingParams(**kwargs)


def _top_logprobs(step_logprobs, limit: int = 5) -> list[tuple[int, float]]:
    return sorted(
        ((token_id, logprob.logprob) for token_id, logprob in step_logprobs.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]


def dump_request_outputs(label: str, outputs) -> None:
    for prompt_idx, output in enumerate(outputs):
        sequence = output.outputs[0]
        print(
            f"[accuracy-probe][output] label={label} prompt={prompt_idx} "
            f"token_ids={list(sequence.token_ids)} text={sequence.text!r}",
            flush=True,
        )
        if sequence.logprobs is None:
            continue
        for token_idx, step_logprobs in enumerate(sequence.logprobs):
            print(
                f"[accuracy-probe][topk] label={label} prompt={prompt_idx} "
                f"token={token_idx} values={_top_logprobs(step_logprobs)}",
                flush=True,
            )


def _dump_first_divergence(label: str, baseline_outputs, candidate_outputs) -> None:
    for prompt_idx, (baseline_output, candidate_output) in enumerate(
        zip(baseline_outputs, candidate_outputs, strict=True)
    ):
        baseline_ids = list(baseline_output.outputs[0].token_ids)
        candidate_ids = list(candidate_output.outputs[0].token_ids)
        for token_idx, (baseline_id, candidate_id) in enumerate(zip(baseline_ids, candidate_ids, strict=True)):
            if baseline_id != candidate_id:
                print(
                    f"[accuracy-probe][first-divergence] label={label} "
                    f"prompt={prompt_idx} token={token_idx} "
                    f"baseline_token={baseline_id} candidate_token={candidate_id}",
                    flush=True,
                )
                break


def compare_logprobs_probe(
    *,
    label: str,
    runner_kwargs: dict,
    prompts: list[str],
    seed: int | None = None,
    atol: float = 0.0689,
    decode_atol: float | None = None,
) -> None:
    """Compare eager baseline and candidate while emitting reproducible diagnostics."""
    if decode_atol is None:
        decode_atol = 2 * atol

    candidate_kwargs = dict(runner_kwargs)
    if seed is not None:
        candidate_kwargs.setdefault("seed", seed)
    baseline_kwargs = {key: value for key, value in candidate_kwargs.items() if key not in _COMPILATION_KEYS}
    baseline_kwargs["enforce_eager"] = True
    sampling_params = _make_sampling_params(seed)

    print_runtime_state(
        f"{label}-start",
        {
            "baseline_kwargs": baseline_kwargs,
            "candidate_kwargs": candidate_kwargs,
            "sampling_seed": seed,
        },
    )
    print(f"[accuracy-probe][phase] label={label} phase=baseline-start", flush=True)
    with VllmRunner(**baseline_kwargs) as runner:
        baseline_outputs = runner.model.generate(prompts=prompts, sampling_params=sampling_params)
    print(f"[accuracy-probe][phase] label={label} phase=baseline-finished", flush=True)

    print(f"[accuracy-probe][phase] label={label} phase=candidate-start", flush=True)
    with VllmRunner(**candidate_kwargs) as runner:
        candidate_outputs = runner.model.generate(prompts=prompts, sampling_params=sampling_params)
    print(f"[accuracy-probe][phase] label={label} phase=candidate-finished", flush=True)

    dump_request_outputs(f"{label}-baseline", baseline_outputs)
    dump_request_outputs(f"{label}-candidate", candidate_outputs)
    _dump_first_divergence(label, baseline_outputs, candidate_outputs)

    for prompt_idx, (baseline_output, candidate_output) in enumerate(
        zip(baseline_outputs, candidate_outputs, strict=True)
    ):
        baseline_sequence = baseline_output.outputs[0]
        candidate_sequence = candidate_output.outputs[0]
        assert baseline_sequence.logprobs is not None and candidate_sequence.logprobs is not None
        assert len(baseline_sequence.token_ids) == len(candidate_sequence.token_ids) == 3
        _check_prefill_token(baseline_sequence, candidate_sequence, prompt_idx, atol)
        for token_idx in range(1, 3):
            _check_decode_token(
                baseline_sequence,
                candidate_sequence,
                token_idx,
                prompt_idx,
                decode_atol,
            )
