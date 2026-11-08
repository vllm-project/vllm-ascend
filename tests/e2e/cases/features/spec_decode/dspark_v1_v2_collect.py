#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Collect one DSpark speculative-decoding run for a v1/v2 model-runner comparison.

The model runner is selected by ``VLLM_USE_V2_MODEL_RUNNER``, which vLLM reads
from the environment when the config is built. It therefore cannot be switched
inside a live process: run this script once per runner and compare the two JSON
dumps with ``dspark_v1_v2_compare.py``.

The dump holds everything the comparison needs:

* ``outputs``   - per prompt: sampled token ids, text, and per-position top-k
  logprobs (the only distribution view the public API exposes).
* ``acceptance`` - per-position acceptance rates plus the raw spec-decode
  counters, read from ``LLM.get_metrics()`` (local mode) or the ``/metrics``
  Prometheus endpoint (``--endpoint`` / remote mode).
* ``timing``    - wall-clock generation time and output-token throughput.

Two run modes, same output schema (so ``dspark_v1_v2_compare.py`` never needs
to know which one produced a given JSON file):

* **Local** (default, unchanged behaviour): constructs ``vllm.LLM`` in this
  process. Only works for deployments that one local Python process can
  construct; use remote mode for an independently deployed service.
* **Remote** (``--endpoint http://<host>:<port>``): talks to an
  already-running ``vllm serve`` instance over its OpenAI-compatible API
  instead of constructing a
  model in-process. This is the only way to drive a real multi-node DP/TP
  deployment, since ``vllm serve`` is what actually knows how to spread
  ranks across machines - a single ``vllm.LLM()`` call cannot.
  Acceptance counters come from scraping ``<endpoint>/metrics`` before and
  after the batch and taking the delta (the counters are cumulative across
  the server's lifetime, see ``_scrape_spec_decode_counters``), using the
  exact same Prometheus metric names vLLM's in-process ``get_metrics()``
  reads from under the hood (``vllm/v1/spec_decode/metrics.py``), so the
  numbers are equivalent, not an approximation.

Example (greedy parity, v1 then v2), local mode (single-process, e.g. the
Qwen3-8B proxy smoke test)::

    VLLM_USE_V2_MODEL_RUNNER=0 python dspark_v1_v2_collect.py \\
        --model Qwen/Qwen3-8B --draft-model deepseek-ai/dspark_qwen3_8b_block7 \\
        --num-speculative-tokens 7 --temperature 0 --out v1_greedy.json
    VLLM_USE_V2_MODEL_RUNNER=1 python dspark_v1_v2_collect.py \\
        ... --out v2_greedy.json

Example, remote mode against an independently deployed multi-node service::

    python dspark_v1_v2_collect.py \\
        --endpoint http://<host>:<port> --served-model-name <served-model-name> \\
        --model <target-model> --draft-model <draft-model> \\
        --num-speculative-tokens 7 --temperature 0 --out v1_greedy.json
    # Run again against the V2 deployment (a separately started server) for v2_greedy.json.
    # --model/--draft-model/--tensor-parallel-size/--data-parallel-size are only
    # recorded into the output JSON's "config" block in remote mode; they are not
    # used to build anything, since the server was already started separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

import regex as re

PROMPT_SUITE: dict[str, str] = {
    # Short English.
    "short_en": "Explain what speculative decoding is in two sentences.",
    # Short Chinese.
    "short_zh": "用两句话解释什么是投机推理。",
    # Long English: long enough to span several KV blocks, which is where the
    # v2 MLA draft recomputes valid KV lengths instead of reusing the target's
    # optimistic upper bound.
    "long_en": (
        "Read the following notes and then summarise the key risks.\n\n"
        + " ".join(
            f"Note {i}: the inference engine batches requests and reuses the "
            f"paged key-value cache across decoding steps."
            for i in range(220)
        )
        + "\n\nSummarise the key risks in a short paragraph."
    ),
    # Long Chinese.
    "long_zh": (
        "请阅读以下材料，然后总结其中的主要风险。\n\n"
        + "".join(f"材料{i}：推理引擎会对请求做批处理，并在解码步骤之间复用分页键值缓存。" for i in range(220))
        + "\n\n请用一段话总结主要风险。"
    ),
    # Code: a different token distribution from prose, and the one most likely
    # to expose draft/target vocabulary handling differences.
    "code": (
        "Write a Python function that merges two sorted lists into one sorted "
        "list without using sorted(). Include a short docstring."
    ),
}


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Target model path or id.")
    parser.add_argument("--draft-model", required=True, help="DSpark draft model path or id.")
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument(
        "--draft-sample-method",
        choices=["greedy", "probabilistic"],
        default="greedy",
        help="Draft-side sampling. Orthogonal to --temperature, which only "
        "controls target-side verification. vLLM defaults to greedy.",
    )
    parser.add_argument(
        "--enforce-eager-draft",
        action="store_true",
        help="Set speculative_config.enforce_eager.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=20,
        help="Top-k logprobs kept per position, used for the top-k overlap and KL comparison.",
    )
    parser.add_argument(
        "--return-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remote mode: request generated token ids when the server supports them (default: enabled).",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--quantization",
        default=None,
        help="e.g. 'ascend' for the W4A8 Kimi-K3 checkpoint.",
    )
    parser.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    parser.add_argument(
        "--prompts",
        default="all",
        help=f"Comma-separated subset of {sorted(PROMPT_SUITE)}, or 'all'.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Duplicate each prompt to build a larger batch.",
    )
    parser.add_argument(
        "--additional-config",
        default=None,
        help="JSON passed through as additional_config.",
    )
    parser.add_argument("--out", required=True, help="Destination JSON path.")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Remote mode: base URL of an already-running `vllm serve` instance "
        "(e.g. http://<host>:8000). When set, talks to the OpenAI-compatible "
        "API and scrapes /metrics for acceptance counters instead of "
        "constructing a local vllm.LLM. Required for real multi-node "
        "deployments (a single process cannot drive a 4-node DP4/TP16 "
        "cluster). --model/--draft-model/--tensor-parallel-size/"
        "--data-parallel-size are only recorded into the output JSON's "
        "config block in this mode, not used to build anything.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Remote mode only: the --served-model-name the server was started "
        "with (what goes in the API request's `model` field). Defaults to "
        "--model if unset.",
    )
    parser.add_argument(
        "--runner-evidence-file",
        default=None,
        help="Remote mode only: server startup log copied to the collector host. "
        "The file must contain the expected 'using Model Runner V1/V2' marker; "
        "only its SHA-256 and verification result are written to the dump. "
        "If it also contains the server's speculative_config (rank-0 log), "
        "--num-speculative-tokens/--draft-sample-method/--enforce-eager-draft must match it.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Remote mode only: per-request HTTP timeout in seconds. Long "
        "prompts / large batches on a big MoE model can legitimately take a "
        "while; this is deliberately generous rather than clipping a slow "
        "but otherwise-successful run.",
    )
    return parser.parse_args()


def _select_prompts(selection: str, repeat: int) -> list[tuple[str, str]]:
    names = sorted(PROMPT_SUITE) if selection == "all" else [n.strip() for n in selection.split(",")]
    unknown = [n for n in names if n not in PROMPT_SUITE]
    if unknown:
        raise SystemExit(f"unknown prompt ids {unknown}; available: {sorted(PROMPT_SUITE)}")
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    return [
        (name if repeat == 1 else f"{name}#{index + 1}", PROMPT_SUITE[name])
        for name in names
        for index in range(repeat)
    ]


_PROM_LINE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)\s*$")
_PROM_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# The exact four Prometheus counter names vLLM's spec-decode path exports,
# per vllm/v1/spec_decode/metrics.py::SpecDecodingProm. Scraping these over
# HTTP and diffing before/after a batch is mathematically the same quantity
# LLM.get_metrics() reads in-process - not an approximation, just a different
# transport.
_SPEC_COUNTER_NAMES = frozenset(
    {
        # Prometheus exposition appends `_total` to counters; this is the actual
        # name emitted by vLLM's /metrics endpoint.
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    }
)


def _parse_prometheus_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Minimal Prometheus exposition-format parser: name -> [(labels, value), ...].

    Deliberately hand-rolled instead of depending on `prometheus_client`'s
    parser (not guaranteed to be importable client-side; the server has it,
    the collector script may be run from a different, lighter environment).
    Exposition format is simple line-oriented text, a regex is sufficient:
    https://github.com/prometheus/docs/blob/main/content/docs/instrumenting/exposition_formats.md
    """
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m or m.group("name") not in _SPEC_COUNTER_NAMES:
            continue
        labels = dict(_PROM_LABEL_RE.findall(m.group("labels") or ""))
        out.setdefault(m.group("name"), []).append((labels, float(m.group("value"))))
    return out


def _scrape_spec_decode_counters(endpoint: str, num_speculative_tokens: int, timeout: float) -> dict[str, Any]:
    """GET <endpoint>/metrics and extract the four spec-decode counters.

    Returns the same shape `_acceptance_per_pos` expects to diff against:
    raw totals plus a per-position list. Multiple engine/model label sets
    (e.g. data-parallel ranks each exporting their own series) are summed -
    we want the aggregate acceptance across the whole deployment, matching
    what a single-process `LLM.get_metrics()` would report for a one-engine
    local run.
    """
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/metrics")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    parsed = _parse_prometheus_text(text)

    num_drafts = sum(v for _, v in parsed.get("vllm:spec_decode_num_drafts_total", []))
    num_draft_tokens = sum(v for _, v in parsed.get("vllm:spec_decode_num_draft_tokens_total", []))
    num_accepted = sum(v for _, v in parsed.get("vllm:spec_decode_num_accepted_tokens_total", []))
    per_pos = [0.0] * num_speculative_tokens
    for labels, value in parsed.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", []):
        pos = int(labels.get("position", -1))
        if 0 <= pos < num_speculative_tokens:
            per_pos[pos] += value
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "accepted_tokens_per_pos": per_pos,
    }


def _diff_spec_decode_counters(
    before: dict[str, Any], after: dict[str, Any], num_speculative_tokens: int
) -> dict[str, Any]:
    """Same output shape as `_acceptance_per_pos` (the local-mode function), computed
    from a before/after snapshot diff instead of a single in-process reading."""
    num_drafts = after["num_drafts"] - before["num_drafts"]
    num_accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    per_pos = [
        after["accepted_tokens_per_pos"][i] - before["accepted_tokens_per_pos"][i]
        for i in range(num_speculative_tokens)
    ]
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": after["num_draft_tokens"] - before["num_draft_tokens"],
        "num_accepted_tokens": num_accepted,
        "acceptance_per_pos": [p / num_drafts if num_drafts else 0.0 for p in per_pos],
        "mean_acceptance_length": (num_accepted / num_drafts + 1.0) if num_drafts else 0.0,
    }


class _RemoteSamplingParams:
    """Plain stand-in for the vllm.SamplingParams fields _remote_completion needs.
    Not a dataclass import from vllm on purpose - see `_run_remote` docstring:
    remote mode must not require vllm to be installed on the machine issuing
    the HTTP requests."""

    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_tokens: int,
        logprobs: int,
        seed: int,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.seed = seed


def _first_remote_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("remote completion response contains no usable choice")
    return choices[0]


def _remote_completion(
    endpoint: str,
    model: str,
    prompt: str,
    sampling_params: Any,
    timeout: float,
    return_token_ids: bool = True,
) -> dict[str, Any]:
    """One /v1/completions call against an already-running vllm serve instance.

    Uses the legacy (non-chat) completions endpoint deliberately: it is the
    one that returns per-token `logprobs.tokens` / `top_logprobs` in a shape
    close enough to the local-mode `CompletionOutput.logprobs` to reuse the
    same downstream comparison logic, and it accepts a bare prompt string
    without needing a chat template (closer to what `llm.generate()` does
    locally with a raw prompt).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "seed": sampling_params.seed,
    }
    # A value of zero means distribution comparison is disabled. Omit the API
    # field rather than sending `logprobs: 0`, which is not portable across
    # OpenAI-compatible servers.
    if sampling_params.logprobs > 0:
        payload["logprobs"] = sampling_params.logprobs
    if return_token_ids:
        payload["return_token_ids"] = True
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response = json.loads(resp.read().decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("remote completion response is not a JSON object")
        _first_remote_choice(response)
        return response
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {endpoint}/v1/completions failed ({e.code}): {body[:500]}") from e


def _extract_remote_token_ids(choice: dict[str, Any]) -> list[int] | None:
    token_ids = choice.get("token_ids")
    if token_ids is None:
        return None
    try:
        return [int(token_id) for token_id in token_ids]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"remote response token_ids is not an integer sequence: {token_ids!r}") from exc


def _serialise_logprobs_remote(choice: dict[str, Any], keep: int) -> list[dict[str, float]]:
    """Same output shape as `_serialise_logprobs` (local mode), built from the
    OpenAI-compatible completions response's `logprobs.top_logprobs` field
    (a list of {token_string: logprob} dicts, one per generated position).

    Local mode keys top-k dicts by token *id* (str(int)); remote mode can
    only see token *strings* here (the public API does not expose ids for
    legacy completions), so this intentionally keys by token string instead.
    Comparisons that rely on exact id equality across v1/v2 local-mode dumps
    won't directly apply byte-for-byte to a local-vs-remote comparison - not
    expected to matter here since v1 and v2 are both collected the same way
    (both remote, or both local), never mixed.
    """
    lp = choice.get("logprobs") or {}
    top_logprobs = lp.get("top_logprobs") or []
    out: list[dict[str, float]] = []
    for step in top_logprobs:
        if not step:
            out.append({})
            continue
        ranked = sorted(step.items(), key=lambda kv: kv[1], reverse=True)[:keep]
        out.append({str(tok): float(val) for tok, val in ranked})
    return out


def _acceptance_per_pos(metrics: list[Any], num_speculative_tokens: int) -> dict[str, Any]:
    """Per-position acceptance plus the raw counters behind it.

    Mirrors ``tests/e2e/pull_request/one_card/spec_decode/utils.py`` so the
    numbers line up with the existing DSpark acceptance baselines.
    """
    from vllm.v1.metrics.reader import Counter, Vector

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted = 0
    per_pos = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, value in enumerate(metric.values):
                if pos < num_speculative_tokens:
                    per_pos[pos] += value

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "acceptance_per_pos": [p / num_drafts if num_drafts else 0.0 for p in per_pos],
        # Mean accepted length per draft, counting the always-accepted bonus token.
        "mean_acceptance_length": (num_accepted / num_drafts + 1.0) if num_drafts else 0.0,
    }


def _serialise_logprobs(output: Any, keep: int) -> list[dict[str, float]]:
    """Top-k logprobs per generated position, as {token_id: logprob}.

    ``logprobs`` is None when the request did not ask for them; callers treat an
    empty list as "distribution comparison unavailable".
    """
    if not output.logprobs:
        return []
    per_pos = []
    for step in output.logprobs:
        ranked = sorted(step.items(), key=lambda kv: kv[1].logprob, reverse=True)[:keep]
        per_pos.append({str(token_id): float(lp.logprob) for token_id, lp in ranked})
    return per_pos


def _run_local(args: argparse.Namespace, runner_env: str, prompts: list[tuple[str, str]]) -> dict[str, Any]:
    """Original single-process mode: construct `vllm.LLM` in this script."""
    from vllm import LLM, SamplingParams

    speculative_config: dict[str, Any] = {
        "method": "dspark",
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_sample_method": args.draft_sample_method,
    }
    if args.enforce_eager_draft:
        speculative_config["enforce_eager"] = True

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "speculative_config": speculative_config,
        "compilation_config": {"cudagraph_mode": args.cudagraph_mode},
        "disable_log_stats": False,
        "seed": args.seed,
        "enable_prefix_caching": False,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.additional_config:
        llm_kwargs["additional_config"] = json.loads(args.additional_config)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        seed=args.seed,
        ignore_eos=False,
    )

    llm = LLM(**llm_kwargs)
    start = time.perf_counter()
    request_outputs = llm.generate([text for _, text in prompts], sampling_params)
    elapsed = time.perf_counter() - start
    metrics = llm.get_metrics()

    outputs = []
    total_output_tokens = 0
    for (name, prompt_text), request_output in zip(prompts, request_outputs, strict=True):
        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids)
        total_output_tokens += len(token_ids)
        outputs.append(
            {
                "prompt_id": name,
                "prompt_num_tokens": len(request_output.prompt_token_ids or []),
                "prompt_preview": prompt_text[:120],
                "token_ids": token_ids,
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "top_logprobs": _serialise_logprobs(completion, args.logprobs),
            }
        )

    return {
        "acceptance": _acceptance_per_pos(metrics, args.num_speculative_tokens),
        "timing": {
            "generate_wall_seconds": elapsed,
            "total_output_tokens": total_output_tokens,
            "output_tokens_per_second": total_output_tokens / elapsed if elapsed > 0 else 0.0,
        },
        "outputs": outputs,
    }


def _run_remote(args: argparse.Namespace, prompts: list[tuple[str, str]]) -> dict[str, Any]:
    """Talk to an already-running `vllm serve` instance over HTTP.

    This is the supported collector path for a multi-node DP/TP deployment,
    because one local `vllm.LLM` process cannot construct that remote cluster.
    Deliberately does NOT import vllm: the client only needs plain HTTP access
    to the service and does not need vllm, torch_npu, or NPU drivers installed.
    """
    served_name = args.served_model_name or args.model
    # A tiny stand-in for the handful of SamplingParams fields _remote_completion
    # needs - kept as a plain object (not vllm.SamplingParams) specifically so
    # this function has zero vllm import requirement. See class docstring above.
    sampling_params = _RemoteSamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        seed=args.seed,
    )

    before = _scrape_spec_decode_counters(args.endpoint, args.num_speculative_tokens, args.request_timeout)
    start = time.perf_counter()
    responses = []
    for name, text in prompts:
        print(f"[collect] requesting prompt={name}", flush=True)
        responses.append(
            _remote_completion(
                args.endpoint,
                served_name,
                text,
                sampling_params,
                args.request_timeout,
                args.return_token_ids,
            )
        )
        print(f"[collect] completed prompt={name}", flush=True)
    elapsed = time.perf_counter() - start
    after = _scrape_spec_decode_counters(args.endpoint, args.num_speculative_tokens, args.request_timeout)

    outputs = []
    total_output_tokens = 0
    for (name, prompt_text), resp in zip(prompts, responses, strict=True):
        choice = _first_remote_choice(resp)
        text_out = choice.get("text", "")
        usage = resp.get("usage") or {}
        n_out = usage.get("completion_tokens", 0)
        total_output_tokens += n_out
        outputs.append(
            {
                "prompt_id": name,
                "prompt_num_tokens": usage.get("prompt_tokens"),
                "prompt_preview": prompt_text[:120],
                # Newer vLLM servers return token_ids when requested. Older
                # servers may omit them; compare.py then falls back to text.
                "token_ids": _extract_remote_token_ids(choice),
                "text": text_out,
                "finish_reason": choice.get("finish_reason"),
                "top_logprobs": _serialise_logprobs_remote(choice, args.logprobs),
            }
        )

    return {
        "acceptance": _diff_spec_decode_counters(before, after, args.num_speculative_tokens),
        "timing": {
            "generate_wall_seconds": elapsed,
            "total_output_tokens": total_output_tokens,
            "output_tokens_per_second": total_output_tokens / elapsed if elapsed > 0 else 0.0,
            # Remote-mode timing includes network/queueing overhead that local
            # mode's in-process `elapsed` does not. Not expected to bias a
            # v1-vs-v2 comparison since both runs pay the same overhead, but
            # don't compare a remote-mode number against a local-mode one.
            "remote_mode": True,
        },
        "outputs": outputs,
    }


def _server_speculative_config(text: str) -> dict[str, Any] | None:
    """Extract speculative_config from a rank-0 ``non-default args`` line."""
    for line in text.splitlines():
        if "non-default args:" not in line:
            continue
        raw_args = line.split("non-default args:", 1)[1].strip()
        try:
            parsed_args = ast.literal_eval(raw_args)
        except (ValueError, SyntaxError) as exc:
            if "speculative_config" in raw_args:
                raise SystemExit("runner evidence contains an unparsable speculative_config") from exc
            continue
        if not isinstance(parsed_args, dict):
            continue
        parsed = parsed_args.get("speculative_config")
        if parsed is None:
            return None
        if not isinstance(parsed, dict):
            raise SystemExit("runner evidence speculative_config is not a dictionary")
        return {
            "num_speculative_tokens": parsed.get("num_speculative_tokens"),
            "draft_sample_method": parsed.get("draft_sample_method"),
            "enforce_eager": bool(parsed.get("enforce_eager", False)),
        }
    return None


def _runner_verification(
    endpoint: str | None,
    runner_env: str,
    evidence_file: str | None,
    claimed_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner = "v1" if runner_env in ("0", "false", "False") else "v2"
    if not endpoint:
        return {
            "status": "local_process_environment",
            "verified": True,
            "runner": runner,
        }
    if not evidence_file:
        return {
            "status": "unverified_client_claim",
            "verified": False,
            "runner": runner,
        }

    with open(evidence_file, "rb") as handle:
        content = handle.read()
    text = content.decode(errors="replace")
    expected = f"using Model Runner {runner.upper()}"
    observed_runners = set(re.findall(r"using Model Runner (V[12])", text))
    if observed_runners != {runner.upper()}:
        raise SystemExit(f"runner evidence must contain only {runner.upper()}, observed {sorted(observed_runners)}")

    # In remote mode the CLI flags describe the *client's* belief about the server; they do not
    # configure it. When the evidence log carries the server's own speculative_config, require the
    # recorded configuration to match it so a client default can never be written into the dump as
    # if it were a server fact.
    server_config = _server_speculative_config(text)
    if server_config is not None and claimed_config is not None:
        mismatches = [
            f"{key}: recorded {claimed_config[key]!r}, server {server_config[key]!r}"
            for key in server_config
            if key in claimed_config and claimed_config[key] != server_config[key]
        ]
        if mismatches:
            raise SystemExit(
                "recorded configuration disagrees with the server startup log: "
                + "; ".join(mismatches)
                + ". Pass flags that match the running service (e.g. --enforce-eager-draft)."
            )
    return {
        "status": "verified_server_startup_log",
        "verified": True,
        "runner": runner,
        "evidence_sha256": hashlib.sha256(content).hexdigest(),
        "matched_marker": expected,
        "server_config": server_config,
    }


def main() -> None:
    args = _build_args()

    # Report the runner that will actually be used. vLLM defaults to V2 when the
    # variable is unset, so an unset value here means "V2", not "V1" - a very
    # easy way to silently compare V2 against itself.
    runner_env = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if runner_env is None:
        raise SystemExit(
            "VLLM_USE_V2_MODEL_RUNNER is unset. Set it explicitly to 0 (V1) or 1 (V2); "
            "leaving it unset silently selects V2, the current default."
        )

    prompts = _select_prompts(args.prompts, args.repeat)
    runner_verification = _runner_verification(
        args.endpoint,
        runner_env,
        args.runner_evidence_file,
        {
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": args.draft_sample_method,
            "enforce_eager": args.enforce_eager_draft,
        },
    )

    if args.endpoint:
        result = _run_remote(args, prompts)
    else:
        result = _run_local(args, runner_env, prompts)

    record = {
        "runner": "v2" if runner_env not in ("0", "false", "False") else "v1",
        "runner_env": runner_env,
        "runner_verification": runner_verification,
        "mode": "remote" if args.endpoint else "local",
        "config": {
            "model": args.model,
            "draft_model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": args.draft_sample_method,
            "enforce_eager_draft": args.enforce_eager_draft,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "logprobs": args.logprobs,
            "return_token_ids": args.return_token_ids,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "cudagraph_mode": args.cudagraph_mode,
            "batch_size": len(prompts),
            "endpoint": args.endpoint,
        },
        **result,
    }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print(
        f"[collect] runner={record['runner']} mode={record['mode']} "
        f"runner_verification={record['runner_verification']['status']} "
        f"acceptance_per_pos={[round(x, 4) for x in record['acceptance']['acceptance_per_pos']]} "
        f"mean_acceptance_length={record['acceptance']['mean_acceptance_length']:.4f} "
        f"throughput={record['timing']['output_tokens_per_second']:.2f} tok/s -> {args.out}"
    )


if __name__ == "__main__":
    main()
