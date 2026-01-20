from dataclasses import dataclass, field
from typing import Optional

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

PROMPTS_SHORT = [
    "Hello, my name is", "The president of the United States is",
    "The capital of France is", "The future of AI is"
]

# NOTE: Randomly fill the prompt with the requested amount for
# the specified capture shape to prevent accuracy issues caused by padding
PROMPTS_LONG = [
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$'
     'be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$,'
     '$\\angle BDC = 90^\\circ$. Suppose $AD = 1$ and $\\frac{BD}{CD} = \\frac{3}{2}$.'
     'If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$,'
     'where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.'
     ),
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'Let $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen'
     'independently and uniformly at random on the perimeter of $ABCD$.'
     'If the expected value of the area of triangle $\\triangle AXY$'
     'can be expressed as $\\frac{m}{n}$, for relatively prime positive'
     'integers $m$ and $n$, compute $m+n$.'),
    ('Solve the following math problem step by step.'
     'The last line of your response should be of the form Answer: '
     '$Answer (without quotes) where $Answer is the answer to the problem.\n\n'
     'Let $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$'
     'and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$'
     'and $x^2 + cx + b = 0$ also have a common real root.'
     'Compute the sum $a + b + c$.')
]


@dataclass(frozen=True)
class LLMTestCase:
    model: str
    prompts: list[str]
    golden_answers: list[str]
    quantization: Optional[str] = None
    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            n=1,
        ))


def gen_and_valid(runner_kwargs: dict, prompts: list[str],
                  sampling_params: SamplingParams, golden_answers: list[str]):
    with VllmRunner(**runner_kwargs) as runner:
        vllm_aclgraph_outputs = runner.model.generate(
            prompts=prompts, sampling_params=sampling_params)
    outputs_gen = []
    for output in vllm_aclgraph_outputs:
        outputs_gen.append(([output.outputs[0].index], output.outputs[0].text))

    output_origin = [([0], answer) for answer in golden_answers]

    check_outputs_equal(
        outputs_0_lst=output_origin,
        outputs_1_lst=outputs_gen,
        name_0="output_origin",
        name_1="outputs_gen",
    )
