from __future__ import annotations

import difflib

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.tokenizers.registry import resolve_tokenizer_args
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import MODELS, calculate_acceptance_per_pos

# Window size for the draft KV cache. Deliberately small enough that the
# prompts below slide the window and trigger eviction of out-of-window draft
# blocks (the null-block-table path), not just the no-op short-prompt path.
DRAFT_KV_WINDOW = 2048
NUM_SPECULATIVE_TOKENS = 3
MAX_MODEL_LEN = 8448

# Greedy speculative decoding is target-driven: every emitted token is
# verified by the target model against its full KV cache. Evicting *draft*
# context can therefore only shift outputs through numerics — once acceptance
# boundaries move, the target verification batch shapes change and bf16
# near-ties can flip, and a single early flip cascades into a completely
# different greedy continuation. Token-level similarity can therefore only be
# a smoke guard against shredded/garbage output; the real detectors for a
# block-table/FIA-mask bug are (a) the acceptance rate — a poisoned draft
# collapses it — and (b) run-to-run determinism — reading an evicted-then-
# reallocated block is data-dependent and cannot replay bitwise.
MIN_SIMILARITY = 0.15


def _assert_similar_ids(baseline_ids, windowed_ids, context: str):
    failures = []
    for idx, (ref, got) in enumerate(zip(baseline_ids, windowed_ids, strict=True)):
        common = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
        ratio = difflib.SequenceMatcher(None, ref, got).ratio()
        print(f"{context} prompt {idx}: common_prefix={common} similarity={ratio:.3f}")
        if ratio < MIN_SIMILARITY:
            failures.append(f"prompt {idx} (common_prefix={common}, similarity={ratio:.3f})")
    assert not failures, (
        f"{context}: greedy output shredded beyond a numerics cascade: {failures}; "
        "the windowed block table and the FIA band mask may disagree"
    )


def _templated_prompts(tokenizer, contents: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for content in contents
    ]


def _long_contents(tokenizer, num_prompts: int, approx_tokens: int) -> list[str]:
    """Build distinct filler prompts of at least ``approx_tokens``.

    Distinct filler per prompt keeps prefix caching from sharing blocks across
    requests, so the concurrent test really exercises block reuse.
    """
    chunk = (
        "The committee reviewed the quarterly engineering report on speculative "
        "decoding, KV cache management, and long-context serving efficiency. "
    )
    chunk_tokens = len(tokenizer(chunk).input_ids)
    # BPE merges tokens across repetition boundaries, so request a generous
    # repeat count and then top up per content to guarantee the length.
    repeats = approx_tokens // chunk_tokens + 2
    contents = [f"Document {idx} begins. {chunk * repeats} Summarize the document." for idx in range(num_prompts)]
    for idx, content in enumerate(contents):
        while len(tokenizer(content).input_ids) < approx_tokens:
            content = content.removesuffix(" Summarize the document.") + chunk + " Summarize the document."
        contents[idx] = content
    return contents


def _run_eagle3(
    monkeypatch: pytest.MonkeyPatch,
    prompts: list[str],
    draft_kv_window: int | None,
    max_num_seqs: int,
    max_tokens: int,
) -> tuple[list[list[int]], list[float]]:
    """Run EAGLE3 spec decode on MRV2 with or without the draft KV window.

    Returns the generated token IDs per prompt and the per-position acceptance.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    main_model_name = MODELS["eagle3"]["main"]
    spec_model_name = MODELS["eagle3"]["spec"]

    additional_config = {} if draft_kv_window is None else {"draft_kv_window": draft_kv_window}
    compilation_config = CompilationConfig(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[12])

    with VllmRunner(
        main_model_name,
        max_model_len=MAX_MODEL_LEN,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        speculative_config={
            "method": "eagle3",
            "model": spec_model_name,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        compilation_config=compilation_config,
        enable_prefix_caching=True,
        additional_config=additional_config,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=max_tokens)
        outputs = llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    acceptance = calculate_acceptance_per_pos(metrics, NUM_SPECULATIVE_TOKENS, Counter, Vector)
    return token_ids, acceptance


def test_draft_kv_window_short_prompt_noop(monkeypatch: pytest.MonkeyPatch):
    """Prompts shorter than the window: the window never binds.

    No draft blocks may be evicted and acceptance must stay close to the
    full-length draft KV run. Output is checked against shredding rather than
    bitwise equality because compact KV grouping can change BF16 near-ties.
    """
    main_model_name = MODELS["eagle3"]["main"]
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_args(main_model_name)[1],
        trust_remote_code=True,
    )
    prompts = _templated_prompts(
        tokenizer,
        [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ],
    )

    baseline_ids, baseline_acceptance = _run_eagle3(monkeypatch, prompts, None, max_num_seqs=4, max_tokens=64)
    windowed_ids, windowed_acceptance = _run_eagle3(
        monkeypatch, prompts, DRAFT_KV_WINDOW, max_num_seqs=4, max_tokens=64
    )

    _assert_similar_ids(baseline_ids, windowed_ids, "short prompts")
    assert all(abs(a - b) < 0.05 for a, b in zip(windowed_acceptance, baseline_acceptance, strict=True)), (
        f"acceptance moved on short prompts: {windowed_acceptance} vs {baseline_acceptance}"
    )


def test_draft_kv_window_long_prompt_eviction(monkeypatch: pytest.MonkeyPatch):
    """Prompt longer than the window: the window slides from the first decode step.

    Out-of-window draft blocks are nulled in the block table and returned to
    the pool while generation continues. The FIA band mask (pre_tokens=W) must
    never read those nulled entries; if it does, the greedy output diverges
    from the full-length draft KV reference.
    """
    main_model_name = MODELS["eagle3"]["main"]
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_args(main_model_name)[1],
        trust_remote_code=True,
    )
    # ~6000-token prompts: > DRAFT_KV_WINDOW from the start, and
    # prompt + generation stays within MAX_MODEL_LEN.
    prompts = _templated_prompts(tokenizer, _long_contents(tokenizer, num_prompts=2, approx_tokens=6000))
    for prompt in prompts:
        num_prompt_tokens = len(tokenizer(prompt).input_ids)
        assert num_prompt_tokens > DRAFT_KV_WINDOW, "prompt must exceed the window so eviction triggers"
        assert num_prompt_tokens + 256 <= MAX_MODEL_LEN

    baseline_ids, baseline_acceptance = _run_eagle3(monkeypatch, prompts, None, max_num_seqs=2, max_tokens=256)
    windowed_ids, windowed_acceptance = _run_eagle3(
        monkeypatch, prompts, DRAFT_KV_WINDOW, max_num_seqs=2, max_tokens=256
    )

    _assert_similar_ids(baseline_ids, windowed_ids, "eviction")
    # The draft sees less context, so acceptance may drop slightly, but a large
    # drop would make the feature useless on long contexts.
    print(f"acceptance baseline={baseline_acceptance} windowed={windowed_acceptance}")
    assert all(b - a < 0.15 for a, b in zip(windowed_acceptance, baseline_acceptance, strict=True)), (
        f"acceptance dropped too much with the window: {windowed_acceptance} vs {baseline_acceptance}"
    )


def test_draft_kv_window_concurrent_block_reuse(monkeypatch: pytest.MonkeyPatch):
    """Concurrent long requests in waves: evicted blocks are reallocated.

    With max_num_seqs=4 and 12 requests, finished requests free their blocks
    (including the bounded draft window blocks) and later waves reuse them. A
    dangling read of an evicted-then-reallocated draft block would poison the
    draft, collapsing acceptance and shredding the greedy output; outputs must
    stay close to the full-length draft KV reference and acceptance must hold.
    """
    main_model_name = MODELS["eagle3"]["main"]
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_args(main_model_name)[1],
        trust_remote_code=True,
    )
    prompts = _templated_prompts(tokenizer, _long_contents(tokenizer, num_prompts=12, approx_tokens=3000))
    for prompt in prompts:
        assert len(tokenizer(prompt).input_ids) > DRAFT_KV_WINDOW

    baseline_ids, baseline_acceptance = _run_eagle3(monkeypatch, prompts, None, max_num_seqs=4, max_tokens=128)
    windowed_ids, windowed_acceptance = _run_eagle3(
        monkeypatch, prompts, DRAFT_KV_WINDOW, max_num_seqs=4, max_tokens=128
    )
    # Reading an evicted-then-reallocated block is data-dependent on whatever
    # the previous owner left behind, so such a bug cannot replay bitwise. A
    # second identical windowed run must reproduce the first run token for
    # token; nondeterminism here means a dangling read of a reused block.
    rerun_ids, _ = _run_eagle3(monkeypatch, prompts, DRAFT_KV_WINDOW, max_num_seqs=4, max_tokens=128)
    assert rerun_ids == windowed_ids, (
        "windowed run is not deterministic across identical replays; an evicted "
        "draft block was likely read after being reallocated to another request"
    )

    _assert_similar_ids(baseline_ids, windowed_ids, "concurrent")
    print(f"acceptance baseline={baseline_acceptance} windowed={windowed_acceptance}")
    assert all(b - a < 0.15 for a, b in zip(windowed_acceptance, baseline_acceptance, strict=True)), (
        f"acceptance dropped too much with the window: {windowed_acceptance} vs {baseline_acceptance}"
    )
