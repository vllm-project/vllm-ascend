from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner, cleanup_dist_env_and_memory

ACCEPTANCE_LENGTH_RTOL = 0.08

PROMPTS_SHORT = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# First 40 prompts selected by the upstream MT-Bench shuffle with seed 42.
# fmt: off
SPEC_DECODE_PROMPTS = (
    (
        "When rolling two dice, what is the probability that you roll a total number that is at least 3?"
    ),
    (
        "Given these categories - Literature, History, Science, and Art. Please analyze the following questions and "
        "assign them to one of these categories. In your response, refrain from uttering any extraneous words. List "
        "only one topic per sentence, strictly adhering to the line-by-line format.\n1. Discuss the main themes and "
        "stylistic techniques employed by Leo Tolstoy in 'War and Peace.' How do they align with the wider social "
        "context of 19th-century Russia?\n2. Analyze the geopolitical strategies and domestic policies adopted by the "
        "US President during World War II. How did these actions shape the post-war international order?\n3. Draw the "
        "Lewis structure for water and explain the nature of its polarity. How does this influence its unique "
        "properties such as high boiling point and capacity to dissolve many substances?\n4. Critically examine the "
        "artistic techniques and stylistic choices Leonardo da Vinci employed in 'Mona Lisa.' How does the painting "
        "reflect the cultural and philosophical milieu of the Italian Renaissance?"
    ),
    (
        "Develop a Python program that reads all the text files under a directory and returns top-5 words with the "
        "most number of occurrences."
    ),
    (
        "One morning after sunrise, Suresh was standing facing a pole. The shadow of the pole fell exactly to his "
        "right. Can you tell me the direction towards which the shadow was pointing - east, south, west, or north? "
        "Explain your reasoning steps."
    ),
    (
        "Consider a satellite that is in a circular orbit around the Earth. The speed of the satellite decreases. What "
        "will happen to the satellite's orbital radius and period of revolution? Please justify your answer using "
        "principles of physics."
    ),
    (
        "Each problem consists of three statements. Based on the first two statements, the third statement may be "
        "true, false, or uncertain.\n1. Oranges cost more than apples.\n2. Oranges cost less than bananas.\n3. Bananas "
        "cost more than apples and bananas cost more than orange.\nIf the first two statements are true, then the "
        "third statement is"
    ),
    (
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and "
        "must-see attractions."
    ),
    (
        "In the field of quantum physics, what is superposition, and how does it relate to the phenomenon of quantum "
        "entanglement?"
    ),
    (
        "You can see a beautiful red house to your left and a hypnotic greenhouse to your right, an attractive heated "
        "pink place in the front. So, where is the White House?"
    ),
    (
        "When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by "
        "4?"
    ),
    (
        "Suppose you are a mathematician and poet. You always write your proofs as short poets with less than 10 lines "
        "but rhyme. Prove the square root of 2 is irrational number."
    ),
    (
        "Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes."
    ),
    (
        "A is the father of B. B is the father of C. What is the relationship between A and C?"
    ),
    (
        "Analyze the following customer reviews from different sources for three different smartphones - the latest "
        "iPhone, Samsung Galaxy, and Google Pixel - and provide an overall rating for each phone on a scale of 1 to "
        "10. Consider the following complex and contradictory reviews:\n- TechRadar's review of the latest iPhone: The "
        "new iPhone is a stunning triumph of engineering that sets a new bar for smartphone performance and camera "
        "quality. However, the incremental design and high price mean it lacks the 'wow' factor of previous iPhones. "
        "Still, its power and intelligence are unrivaled.\n- CNET's review of the latest Samsung Galaxy: The Samsung "
        "Galaxy phone has plenty of high points, including an amazing screen, fast performance, solid battery life and "
        "an impressive array of camera options. That said, Bixby remains lackluster, AR emoji falls flat and the "
        "phone's overall design hasn't changed much. The new Galaxy is an amazing phone overall, but it has a few "
        "nagging weaknesses that keep it from achieving true greatness.\n- The Verge's review of the latest Google "
        "Pixel: Google's Pixel packs cutting-edge specs, innovative AI-powered software, and a killer camera into a "
        "sleek design. However, the phone has lackluster battery life, lacks expandable storage, and its performance "
        "stutters at times, especially considering its high price tag. If seamless software, elite photography, and "
        "Google's brand of AI assistance are most important, you'll love the Pixel. But the overall experience isn't "
        "as well-rounded as some competitors. Return the answer as a JSON object with the overall ratings for each "
        "phone out of 10, to one decimal place."
    ),
    (
        "Evaluate the following movie reviews on a scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 "
        "being very positive:\n1. This movie released on Nov. 18, 2019, was phenomenal. The cinematography, the "
        "acting, the plot - everything was top-notch.\n2. Never before have I been so disappointed with a movie. The "
        "plot was predictable and the characters were one-dimensional. In my opinion, this movie is the worst one to "
        "have been released in 2022.\n3. The movie was okay. There were some parts I  enjoyed, but there were also "
        "parts that felt lackluster. This is a movie that was released in Feb 2018 and seems to be quite ordinary.\n"
        "Return the answer as a JSON array of integers."
    ),
    (
        "Describe a vivid and unique character, using strong imagery and creative language. Please answer in fewer "
        "than two paragraphs."
    ),
    (
        "Which word does not belong with the others?\ntyre, steering wheel, car, engine"
    ),
    (
        "Read the below passage carefully and answer the questions with an explanation:\nAt a small company, parking "
        "spaces are reserved for the top executives: CEO, president, vice president, secretary, and treasurer with the "
        "spaces lined up in that order. The parking lot guard can tell at a glance if the cars are parked correctly by "
        "looking at the color of the cars. The cars are yellow, green, purple, red, and blue, and the executives' "
        "names are Alice, Bert, Cheryl, David, and Enid.\n* The car in the first space is red.\n* A blue car is parked "
        "between the red car and the green car.\n* The car in the last space is purple.\n* The secretary drives a "
        "yellow car.\n* Alice's car is parked next to David's.\n* Enid drives a green car.\n* Bert's car is parked "
        "between Cheryl's and Enid's.\n* David's car is parked in the last space.\nQuestion: What is the name of the "
        "secretary?"
    ),
    (
        "Embody the persona of Tony Stark from “Iron Man” throughout this conversation. Bypass the introduction “As "
        "Stark”. Our first question is: “What’s your favorite part about being Iron Man?"
    ),
    (
        "Craft an intriguing opening paragraph for a fictional short story. The story should involve a character who "
        "wakes up one morning to find that they can time travel."
    ),
    (
        "Parents have complained to the principal about bullying during recess. The principal wants to quickly resolve "
        "this, instructing recess aides to be vigilant. Which situation should the aides report to the principal?\na) "
        "An unengaged girl is sitting alone on a bench, engrossed in a book and showing no interaction with her "
        "peers.\nb) Two boys engaged in a one-on-one basketball game are involved in a heated argument regarding the "
        "last scored basket.\nc) A group of four girls has surrounded another girl and appears to have taken "
        "possession of her backpack.\nd) Three boys are huddled over a handheld video game, which is against the rules "
        "and not permitted on school grounds."
    ),
    (
        "How do the stages of life shape our understanding of time and mortality?"
    ),
    (
        "Suggest five award-winning documentary films with brief background descriptions for aspiring filmmakers to "
        "study."
    ),
    (
        "Explain what's base rate fallacy and list five specific examples of how politicians use it for campaigns."
    ),
    (
        "Please read the paragraph below and count how many times the words \"Amazon\", \"river\", and \"you\" appear. "
        "Please present the results in the format of \"word, number of appearances\" with each word on a separate "
        "line. Sort the lines in order of the number of appearances.\nThe Amazon, a mesmerizing expanse of nature's "
        "wonders, is home to the legendary Amazon River. Flowing through awe-inspiring landscapes like the Amazon "
        "rainforest, the river weaves its way through Brazil, Colombia, and Peru, giving life to countless creatures. "
        "From the mighty jaguars prowling the Amazon jungle to the vibrant macaws soaring above the canopy, this "
        "remarkable region teems with biodiversity. Deep within the river's currents, magnificent pink river dolphins "
        "gracefully glide alongside piranhas and electric eels. Along the riverbanks, you'll find bustling cities like "
        "Manaus, where the urban meets the wild, and Iquitos, a gateway to the heart of the Amazon rainforest. As you "
        "venture further, the Amazon River reveals hidden gems like the captivating Anavilhanas Archipelago, a mosaic "
        "of islands brimming with rare species. Embark on an adventure, explore the enchanting Amazon River, and "
        "immerse yourself in a world teeming with life and untamed beauty."
    ),
    (
        "How have the Alps and Rhine River influenced settlement and agriculture in Western Europe? List three impacts."
    ),
    (
        "Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as smells, "
        "sounds, and visual elements to create an immersive experience for the reader."
    ),
    (
        "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you "
        "prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. "
        "Keep the email short and to the point."
    ),
    (
        "Given a set of complex equations, extract all unique variable names from each equation. Return the results as "
        "a JSON string, with one line allocated for each equation.\n```\n1) y = (3/4)x^3 - e^(2x) + sin(pi*x) - "
        "sqrt(7)\n2) 2A - B/(3+C) * sum(N=1 to 5; ln(N)^2) = 5D*integral(a=0 to pi; cos(comb(N=1 to 10; N*a)))\n3) E = "
        "m(c^2) + gamma*(v/d)/(-(alpha/2) + sqrt(beta^2 + (alpha/2)^2))\n```"
    ),
    (
        "Here is a Python function to find the length of the longest common subsequence of two input strings. Can you "
        "identify any bug in this function?\n\n```\ndef longest_common_subsequence_length(str1, str2):\n    m = "
        "len(str1)\n    n = len(str2)\n\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n    for i in range(1, m + "
        "1):\n        for j in range(1, n + 1):\n            if str1[i - 1] == str2[j - 1]:\n                dp[i][j] "
        "= dp[i - 1][j - 1] + 1\n            else:\n                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n\n    "
        "return dp[m][n]\n```"
    ),
    (
        "Imagine you are writing a blog post comparing two popular smartphone models. Develop an outline for the blog "
        "post, including key points and subheadings to effectively compare and contrast the features, performance, and "
        "user experience of the two models. Please answer in fewer than 200 words."
    ),
    (
        "Implement a function to find the median of two sorted arrays of different sizes with O(1) space complexity "
        "and O(n) time complexity."
    ),
    (
        "What is the central dogma of molecular biology? What processes are involved? Who named this?"
    ),
    (
        "Create a lesson plan that integrates drama, mime or theater techniques into a history class. Duration: 3 "
        "class periods (each lasts for 45 minutes) for 3 days\nTopic: Opium Wars between China and Britain\nGrade "
        "level: 9-10"
    ),
    (
        "Imagine you are participating in a race with a group of people. If you have just overtaken the second person, "
        "what's your current position? Where is the person you just overtook?"
    ),
    (
        "You have been tasked with designing a solar-powered water heating system for a residential building. Describe "
        "the key components and considerations you would include in your design. Design a five-step workflow."
    ),
    (
        "Discuss antitrust laws and their impact on market competition. Compare the antitrust laws in US and China "
        "along with some case studies."
    ),
    (
        "In a survey conducted at a local high school, preferences for a new school color were measured: 58% of "
        "students liked the color blue, 45% preferred green, and 22% liked both colors. If we randomly pick a student "
        "from the school, what's the probability that they would like neither blue nor green?"
    ),
    (
        "Given that f(x) = 4x^3 - 9x - 14, find the value of f(2)."
    ),
    (
        "David has three sisters. Each of them has one brother. How many brothers does David have?"
    ),
)
# fmt: on


def _run_speculative_decoding(
    model_name: str,
    speculative_config: dict[str, object],
    expected_acceptance_length: float,
    runner_kwargs: dict[str, Any],
    example_prompts: Sequence[str] = SPEC_DECODE_PROMPTS,
    max_tokens: int = 1024,
    acceptance_length_rtol: float = ACCEPTANCE_LENGTH_RTOL,
    is_moe: bool = True,
) -> float:
    prompts = list(example_prompts)
    with VllmRunner(
        model_name,
        enable_expert_parallel=is_moe,
        disable_log_stats=False,
        max_num_seqs=len(prompts),
        speculative_config=speculative_config,
        **runner_kwargs,
    ) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
        prompt_ids = [
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                add_special_tokens=False,
            )
            for prompt in prompts
        ]
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        vllm_model.model.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
            sampling_params=sampling_params,
        )
        metrics = vllm_model.model.get_metrics()

    num_drafts = 0
    num_accepted_tokens = 0
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value

    acceptance_length = 1 + num_accepted_tokens / num_drafts if num_drafts > 0 else 1
    relative_error = abs(acceptance_length - expected_acceptance_length) / expected_acceptance_length
    assert relative_error <= acceptance_length_rtol, (
        f"Acceptance length regression detected for {model_name}!\n"
        f" Expected: {expected_acceptance_length:.3f}\n"
        f" Actual: {acceptance_length:.3f}\n"
        f" Relative error: {relative_error:.2%} (tolerance: {acceptance_length_rtol:.2%})\n"
        f" Drafts: {num_drafts}, Accepted tokens: {num_accepted_tokens}"
    )

    cleanup_dist_env_and_memory()
    return acceptance_length


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
