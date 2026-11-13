# Thinking token budget

Model Runner V2 supports a per-request `thinking_token_budget` on Ascend.
It limits the reasoning phase identified by the configured token markers.
When the budget is reached, sampling forces the reasoning end sequence and
generation continues into the answer. `max_tokens` still limits the entire
output: leave room for the end sequence and the final answer.

## Parameters

| Value | Behavior |
| --- | --- |
| Omitted, `None` / JSON `null`, or `-1` | Disable the reasoning budget. |
| `0` | End an active reasoning phase immediately. |
| Nonnegative integer | Limit reasoning to that many tokens. |
| Integer above `2147483647` | Accepted; V2 clamps device storage to the int32 maximum. |
| Below `-1`, boolean, float, string, list, or object | Rejected during request validation. |

Budgets are independent within a batch. V2 tracks committed token history by
request slot, including when a request resumes after preemption. Rejected
speculative tokens do not become committed reasoning history.

Configure the reasoning parser and markers for the selected model. A finite
budget without the required reasoning configuration is rejected during input
validation; it does not enable reasoning in an ordinary model. Without
reasoning configuration the V2 budget state stays disabled. After the natural
reasoning end, the answer is unaffected by the budget.

The budget counts token IDs after the reasoning start marker, including any
reasoning already present at the end of the prompt. The forced end sequence is
additional output. To audit a run, save prompt and output token IDs and locate
the marker boundaries; retokenizing displayed text can give a different count.

## Offline generation

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=4,
    reasoning_config={
        "reasoning_parser": "qwen3",
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
    },
)
tokenizer = llm.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 3?"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
params = [
    SamplingParams(temperature=0, max_tokens=512, thinking_token_budget=budget)
    for budget in [None, 0, 16, 64]
]
for output in llm.generate([prompt] * len(params), params):
    print(output.request_id, output.outputs[0].text)
```

## OpenAI serving

```bash
vllm serve Qwen/Qwen3-1.7B \
    --dtype bfloat16 --max-model-len 4096 --max-num-seqs 4 \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    temperature=0,
    max_tokens=512,
    extra_body={
        "thinking_token_budget": 16,
        "chat_template_kwargs": {"enable_thinking": True},
        "return_token_ids": True,
    },
)
print(response.choices[0].message.content)
```

For streaming, concatenate returned token IDs in chunk order. Invalid budgets
produce HTTP 400 with `thinking_token_budget` in the error.

## Execution modes and speculative decoding

Use `--enforce-eager` for eager execution, or the default compilation settings
for ACL Graph. Save the graph capture/replay logs with validation results.
Both greedy (`temperature=0`) and random sampling support budgets.

For n-gram speculative decoding, add:

```bash
--speculative-config '{"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

For a compatible draft model with the same tokenizer, configure the draft
model explicitly. For example, with the Qwen3-1.7B target above:

```bash
--speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

Verify the draft and accepted-token counters: enabling a speculative method
does not prove that a particular prompt actually exercises rejection sampling.
Other speculative methods require their own compatible model/configuration
and validation.
Upstream speculative-decoding restrictions still apply: for example, combining
speculation with `min_p` or `logit_bias` is rejected during request validation.

## Validation

```bash
python -m pytest -sv tests/ut/worker/v2/test_thinking_budget.py \
    tests/ut/worker/v2/test_thinking_budget_validation.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget.py \
    tests/e2e/pull_request/one_card/test_thinking_budget_addressing.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_generation.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_serving.py
```

V2 request lifecycle coverage targets `AscendRequestState` and upstream
`ThinkingBudgetState` in `tests/ut/worker/v2/`; the removed V1 input batch is
not required. The host tests check lifecycle and sampler contracts with device
storage mocked. NPU tests exercise the upstream budget state with Ascend kernels,
including rejected drafts, mixed budgets, slot reuse, and resumed end prefixes.
Rejection tests exercise absent, tensor, and cached draft logits with standard,
entropy, and block verification. Large-address tests require at least 8 GiB of
free NPU memory plus headroom and cover cold and incremental history scans.
Model tests cover Qwen3-1.7B and ordinary-model regression with
Qwen2.5-0.5B-Instruct. The constrained-cache test asserts that scheduler
preemption actually occurs and compares output against an unconstrained run.

For reproducible results record both source commits, model revision, CANN,
driver, torch/torch_npu and Triton versions, device type, commands, seeds,
batch settings, actual token IDs, finish reasons, and test logs.
Compare baseline and changed code on the same device with identical prompts
and output limits. Report TTFT, TPOT, throughput, and output lengths for
disabled, nonbinding, and binding budgets. A shortened reasoning response
alone is not evidence of faster sampling.
