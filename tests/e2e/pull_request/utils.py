from dataclasses import dataclass

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

PROMPTS_SHORT = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# NOTE: Randomly fill the prompt with the requested amount for
# the specified capture shape to prevent accuracy issues caused by padding
PROMPTS_LONG = [
    (
        "Solve the following math problem step by step."
        "The last line of your response should be of the form Answer: "
        "$Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$"
        "be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$,"
        "$\\angle BDC = 90^\\circ$. Suppose $AD = 1$ and $\\frac{BD}{CD} = \\frac{3}{2}$."
        "If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$,"
        "where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$."
    ),
    (
        "Solve the following math problem step by step."
        "The last line of your response should be of the form Answer: "
        "$Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        "Let $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen"
        "independently and uniformly at random on the perimeter of $ABCD$."
        "If the expected value of the area of triangle $\\triangle AXY$"
        "can be expressed as $\\frac{m}{n}$, for relatively prime positive"
        "integers $m$ and $n$, compute $m+n$."
    ),
    (
        "Solve the following math problem step by step."
        "The last line of your response should be of the form Answer: "
        "$Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        "Let $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$"
        "and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$"
        "and $x^2 + cx + b = 0$ also have a common real root."
        "Compute the sum $a + b + c$."
    ),
]


@dataclass(frozen=True)
class LLMTestCase:
    model: str
    prompts: list[str]
    golden_answers: list[str] | None = None
    quantization: str | None = None


# Keys that are specific to compilation/graph capture and should not be passed
# to the eager baseline runner.
_COMPILATION_KEYS = {"compilation_config", "additional_config", "cudagraph_capture_sizes"}

# Top-K logprobs to fetch per token; used for decode-phase cross-lookup.
_DECODE_TOPK = 20

_LOGPROB_SAMPLING_PARAMS = SamplingParams(
    max_tokens=3,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    logprobs=_DECODE_TOPK,
)


def _check_prefill_token(
    base_seq,
    comp_seq,
    prompt_idx: int,
    atol: float,
) -> None:
    """Token 0 is produced by the prefill pass; both models see identical input,
    so the chosen token *must* be the same and its logprob must match within atol."""
    base_token_id = base_seq.token_ids[0]
    comp_token_id = comp_seq.token_ids[0]
    assert base_token_id == comp_token_id, (
        f"Prefill token mismatch at prompt {prompt_idx}: baseline={base_token_id}, compiled={comp_token_id}"
    )
    base_logprob = base_seq.logprobs[0][base_token_id].logprob
    comp_logprob = comp_seq.logprobs[0][comp_token_id].logprob
    assert abs(base_logprob - comp_logprob) <= atol, (
        f"Prefill logprob mismatch at prompt {prompt_idx}: "
        f"baseline={base_logprob:.4f}, compiled={comp_logprob:.4f}, "
        f"diff={abs(base_logprob - comp_logprob):.4f} > atol={atol}"
    )


def _check_decode_token(
    base_seq,
    comp_seq,
    token_idx: int,
    prompt_idx: int,
    decode_atol: float,
) -> None:
    """Tokens 1-2 come from decode passes.  When the two models pick different
    tokens the context has already diverged, so we cannot compare logprobs of
    the chosen tokens directly.  Instead we do a cross-lookup: find the
    baseline's chosen token inside compiled's top-K distribution (and vice
    versa) and assert that the assigned log-probability is close.  This
    confirms that the compiled model's distribution is numerically consistent
    with the baseline's even when the argmax differs by a tiny margin.
    """
    base_token_id = base_seq.token_ids[token_idx]
    comp_token_id = comp_seq.token_ids[token_idx]
    base_topk = base_seq.logprobs[token_idx]  # dict[token_id, Logprob]
    comp_topk = comp_seq.logprobs[token_idx]

    if base_token_id == comp_token_id:
        # Happy path: same token, direct logprob comparison.
        diff = abs(base_topk[base_token_id].logprob - comp_topk[comp_token_id].logprob)
        assert diff <= decode_atol, (
            f"Decode logprob mismatch at prompt {prompt_idx}, token {token_idx}: "
            f"baseline={base_topk[base_token_id].logprob:.4f}, "
            f"compiled={comp_topk[comp_token_id].logprob:.4f}, "
            f"diff={diff:.4f} > decode_atol={decode_atol}"
        )
        return

    # Tokens differ – cross-lookup in each model's top-K distribution.
    base_logprob = base_topk[base_token_id].logprob
    comp_logprob = comp_topk[comp_token_id].logprob

    # Check: what log-probability did compiled assign to baseline's token?
    assert base_token_id in comp_topk, (
        f"Decode token mismatch at prompt {prompt_idx}, token {token_idx}: "
        f"baseline chose token {base_token_id} (logprob={base_logprob:.4f}) but "
        f"compiled chose token {comp_token_id} (logprob={comp_logprob:.4f}) and "
        f"baseline's token does not appear in compiled's top-{_DECODE_TOPK} distribution"
    )
    comp_logprob_of_base_token = comp_topk[base_token_id].logprob
    diff = abs(base_logprob - comp_logprob_of_base_token)
    assert diff <= decode_atol, (
        f"Decode distribution mismatch at prompt {prompt_idx}, token {token_idx}: "
        f"baseline chose token {base_token_id} with logprob={base_logprob:.4f}; "
        f"compiled assigned logprob={comp_logprob_of_base_token:.4f} to that token, "
        f"diff={diff:.4f} > decode_atol={decode_atol} "
        f"(compiled chose token {comp_token_id} with logprob={comp_logprob:.4f})"
    )


def compare_logprobs(
    runner_kwargs: dict,
    prompts: list[str],
    atol: float = 0.0689,
    decode_atol: float | None = None,
) -> None:
    """Run the model in eager baseline mode and in the configured compilation
    mode, generate 3 tokens per prompt, then verify numerical accuracy:

    * Token 0 (prefill pass): chosen token must be identical; logprob must
      match within *atol*.
    * Tokens 1-2 (decode passes): if chosen tokens match, logprob must be
      within *decode_atol*; if they differ, the baseline token must appear in
      the compiled model's top-K distribution with a logprob within
      *decode_atol* of the baseline value.

    *decode_atol* defaults to ``2 * atol`` when not supplied.
    """
    if decode_atol is None:
        decode_atol = 2 * atol

    baseline_kwargs = {k: v for k, v in runner_kwargs.items() if k not in _COMPILATION_KEYS}
    baseline_kwargs["enforce_eager"] = True

    with VllmRunner(**baseline_kwargs) as runner:
        baseline_outputs = runner.model.generate(prompts=prompts, sampling_params=_LOGPROB_SAMPLING_PARAMS)

    with VllmRunner(**runner_kwargs) as runner:
        compiled_outputs = runner.model.generate(prompts=prompts, sampling_params=_LOGPROB_SAMPLING_PARAMS)

    for prompt_idx, (base_out, comp_out) in enumerate(zip(baseline_outputs, compiled_outputs)):
        base_seq = base_out.outputs[0]
        comp_seq = comp_out.outputs[0]

        assert base_seq.logprobs is not None and comp_seq.logprobs is not None, (
            f"logprobs not returned for prompt {prompt_idx}"
        )
        assert len(base_seq.token_ids) == len(comp_seq.token_ids) == 3, (
            f"Expected 3 tokens for prompt {prompt_idx}, "
            f"got baseline={len(base_seq.token_ids)}, compiled={len(comp_seq.token_ids)}"
        )

        _check_prefill_token(base_seq, comp_seq, prompt_idx, atol)
        for token_idx in range(1, 3):
            _check_decode_token(base_seq, comp_seq, token_idx, prompt_idx, decode_atol)
