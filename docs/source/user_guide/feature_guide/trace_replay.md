# Trace Replay

Trace replay forces decoding to follow a supplied token sequence while the
model still runs every forward pass and computes logits normally. The returned
logprobs and ranks therefore describe the model's real distribution for each
forced token.

This is useful for locating the first numerical divergence between Ascend and
another backend, or between eager and ACL Graph execution, without allowing an
earlier sampling difference to change the rest of the decode path.

## Requirements

Trace replay is provided by upstream vLLM and works on Ascend with Model Runner
V2. It reserves a per-request trace buffer, so it is disabled by default.

- Use a vLLM version that includes trace replay support.
- Enable Model Runner V2 with `VLLM_USE_V2_MODEL_RUNNER=1`.
- Construct the engine with `enable_trace_replay=True`.
- Use the offline Python API. The OpenAI-compatible request schemas do not
  currently expose `trace_decode_token_ids`.

## Offline example

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_trace_replay=True,
)

trace_token_ids = [9707, 11, 358, 1097]
params = SamplingParams(
    trace_decode_token_ids=trace_token_ids,
    logprobs=5,
)

output = llm.generate(["Hello, my name is"], params)[0].outputs[0]
assert list(output.token_ids) == trace_token_ids

for step, (token_id, step_logprobs) in enumerate(
    zip(output.token_ids, output.logprobs, strict=True)
):
    token_logprob = step_logprobs[token_id]
    print(
        f"step={step} token={token_id} "
        f"logprob={token_logprob.logprob} rank={token_logprob.rank}"
    )
```

`max_tokens` is set to the effective trace length. The trace is truncated when
it does not fit in the remaining model context. EOS tokens and other stop
conditions inside the trace do not stop replay early.

## Comparing eager and ACL Graph

Capture one trace and replay that same list for both configurations. Keeping
the token history identical makes the first logprob difference directly
comparable.

Eager configuration:

```python
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_trace_replay=True,
    enforce_eager=True,
)
```

ACL Graph decode configuration:

```python
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_trace_replay=True,
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
)
```

For each step, record at least the token ID, logprob, and rank. Compare the
records in order and report the first step that exceeds the tolerance selected
for the model and data type. Token IDs must match the trace exactly; logprobs
need not be bitwise identical across devices or execution modes.

## Dynamic batching

Each request can use a different trace and trace length. Trace replay follows
the request-state mapping as batches grow and shrink, and is compatible with
asynchronous scheduling on Ascend. Pass one `SamplingParams` instance per
prompt when traces differ:

```python
params = [
    SamplingParams(trace_decode_token_ids=[9707, 11], logprobs=5),
    SamplingParams(trace_decode_token_ids=[785, 374, 264], logprobs=5),
]
outputs = llm.generate(["Prompt A", "Prompt B"], params)
```

## Limitations

Trace replay requires `n=1` and cannot be combined with:

- prompt logprobs;
- speculative decoding;
- structured outputs;
- repetition detection;
- thinking token budgets;
- bad-word logit masking.

These combinations raise `ValueError` rather than silently disabling replay.

## Ascend UVA fallback

The upstream implementation stores trace state in a host-backed buffer. On
Ascend software stacks where Triton UVA is unavailable, vLLM Ascend's existing
UVA compatibility path transparently stages this state on the NPU. No Trace
Replay-specific CANN operator or environment variable is required.

For the generic API contract and additional examples, see the upstream
[Trace Replay documentation](https://docs.vllm.ai/en/latest/serving/online_serving/trace_replay/).
