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

_LOGPROB_SAMPLING_PARAMS = SamplingParams(
    max_tokens=3,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    logprobs=1,
)


def compare_logprobs(
    runner_kwargs: dict,
    prompts: list[str],
    atol: float = 0.02,
) -> None:
    """Run the model in eager baseline mode and in the configured compilation
    mode, generate 3 tokens per prompt, then assert that the selected-token
    logprobs match within *atol*.

    Token 0 comes from the prefill forward pass; tokens 1-2 come from decode
    forward passes, so all three pipeline stages are exercised.
    """
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

        for token_idx in range(3):
            base_token_id = base_seq.token_ids[token_idx]
            comp_token_id = comp_seq.token_ids[token_idx]
            assert base_token_id == comp_token_id, (
                f"Token ID mismatch at prompt {prompt_idx}, token {token_idx}: "
                f"baseline={base_token_id}, compiled={comp_token_id}"
            )

            base_logprob = base_seq.logprobs[token_idx][base_token_id].logprob
            comp_logprob = comp_seq.logprobs[token_idx][comp_token_id].logprob
            assert abs(base_logprob - comp_logprob) <= atol, (
                f"Logprob mismatch at prompt {prompt_idx}, token {token_idx}: "
                f"baseline={base_logprob:.4f}, compiled={comp_logprob:.4f}, "
                f"diff={abs(base_logprob - comp_logprob):.4f} > atol={atol}"
            )
