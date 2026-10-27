# EAGLE-4 chain speculative decoding

This integration runs dense EAGLE-4 draft checkpoints on Model Runner V2. It
uses the official `eagle-harness` configuration validator, auxiliary-layer
mapping and low-rank scoring. The draft transformer, weight loader, chain
rejection sampler, attention metadata, KV cache and ACL Graph loop come from
vLLM and vLLM Ascend. NPU candidate rescoring uses a Triton Ascend kernel.

The initial implementation supports a fixed chain. Tree drafting, automatic
cost calibration and native-OFF scheduling are not implemented on Ascend.
The CUDA performance numbers from EAGLE-4 do not describe this implementation.
NPU model accuracy, graph execution and throughput require hardware validation
before production deployment.

## Requirements

- An Ascend environment matching this checkout's CANN, PyTorch, torch-npu and
  Triton Ascend dependencies. Ascend 310P is not supported.
- The companion vLLM changes containing `vllm.v1.spec_decode.registry`,
  `SpeculativeConfig.method_config` and registered auxiliary-hidden-state
  support. The local `../vllm` EAGLE-4 branch contains these hooks; stock vLLM
  v0.31.0 does not. Both engine and worker processes must use that build.
- `eagle-harness==0.1.1`, installed as an optional dependency. Install the base
  package without the `[vllm]` extra, which selects CUDA dependencies. Follow
  the upstream package's license terms; its source remains an external
  dependency.
- A dense, single-layer `LlamaForCausalLMEagle4` checkpoint with the trained full
  LM head and low-rank candidate factors. The official
  `EAGLE-Corp/Qwen3.8-27B-EAGLE4` structure is handled, including its three
  auxiliary states, feature norms, output norm and 4096-token sliding window.

Install the base harness into an already configured Ascend environment:

```bash
uv pip install "eagle-harness==0.1.1"
```

Do not install the CUDA engine wheels in the
[official CUDA installation recipe](https://github.com/eagle-hq/EAGLE-4/blob/main/docs/install.md#vllm-cuda)
into an Ascend environment. Install the two local engine checkouts using the
usual Ascend build instructions.

The Ascend wheel also exposes an optional `eagle4` extra with the same harness
pin (`vllm-ascend[eagle4]`); the default installation does not require it.

## Run

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --speculative-config '{"method":"eagle4","model":"EAGLE-Corp/Qwen3.8-27B-EAGLE4","num_speculative_tokens":3}'
```

For initial debugging, add `--enforce-eager`. Remove it to exercise the Ascend
ACL Graph path. Use the regular Ascend attention backend; the CUDA `CUSTOM`
tree-attention backend is not needed.

The `eagle4` method name selects the Ascend adapter. The official CUDA plugin
uses the separate name `eagle_harness`. Their model registrations cannot
overwrite the Ascend draft: the checkpoint architecture is normalized to
`AscendEagle4ForCausalLM` before vLLM inspects it.

Offline usage:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.8-27B",
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=8192,
    speculative_config={
        "method": "eagle4",
        "model": "EAGLE-Corp/Qwen3.8-27B-EAGLE4",
        "num_speculative_tokens": 3,
    },
)
outputs = llm.generate(["Explain speculative decoding."], SamplingParams(temperature=0, max_tokens=128))
```

## Supported configuration

`num_speculative_tokens` specifies the chain length, from 1 to 15. Alternatively,
set `method_config.max_verify_width` to that length plus one. Setting both
requires agreement. The harness parser validates the width and shape.

The adapter defaults `method_config` to `mode="static"`,
`intensity="chain_intensity"`, `node_top_k=1`. These are also the supported
explicit values. `tree_depth` is accepted by the harness shape validator;
the actual chain length is `num_speculative_tokens`.

Target and draft use the same TP group. PP, DP, context parallelism, DBO,
separate LM-head TP, LoRA and multimodal targets are rejected. Parallel
drafting, local-argmax reduction, adaptive verification, quantized draft
weights, dynamic chain schedules, heterogeneous vocabularies, disaggregated
KV transfer, quantized KV caches and nonstandard rejection sampling are rejected. Nemotron MoE,
NoPE, attention-normalization and post-MLP-normalization draft structures
are also rejected until their Ascend paths are implemented.

Candidate selection is per vocabulary shard, as in the official implementation.
Each rank re-scores its candidates with its own trained full head, accumulates
in FP32 and rounds to the draft dtype before gathering. Padded vocabulary rows
are excluded. Only the target input embedding is shared; the draft LM head is
preserved. Checkpoint output-layer IDs are converted once to vLLM input-layer
IDs (`[3, 31, 59]` becomes `[4, 32, 60]`).

The draft uses the existing sliding-window KV cache specification. The FIA
left-band width is `window - 1`, so the checkpoint's window includes the
current token. The same boundary is used during prefill, decoding and graph
capture.

## Validation

On a CPU development host, the following runs real configuration and numerical
helpers without importing the NPU test harness:

```bash
.venv/bin/python -m pytest --noconftest tests/ut/spec_decode/test_eagle4.py -q
```

On an Ascend machine with the companion vLLM build and harness installed:

```bash
pytest tests/e2e/pull_request/one_card/spec_decode/test_eagle4_head.py -v
pytest tests/e2e/pull_request/two_card/model_runner_v2/test_eagle4.py -v
```

The first suite checks NPU candidate rescoring, changed-input graph replay and
the sliding-window boundary against a CPU attention reference. The model suite
compares greedy token IDs and logprobs with the target alone in eager and graph
modes. It includes chunked prefill beyond the 4096-token draft window, prefix
cache hits, recycled request slots, sampled logprobs scored by the target on
the sampled tokens' own prefixes, and actual accepted draft tokens. These tests are provided for hardware validation; their presence
does not mean they have passed on NPU.

Measure performance with the same target, TP, precision, cache settings, input
lengths and output lengths for both servers. First start the target without
`--speculative-config`, then repeat with the command above. For each server,
warm it up and run identical benchmark commands, for example:

```bash
vllm bench serve --backend vllm --model Qwen/Qwen3.8-27B \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 128 --max-concurrency 4 --request-rate inf \
  --save-result --result-dir results/eagle4-chain
```

Use separate result directories for the baseline and chain. Repeat at
concurrency 1, 4, 16 and 32, with both short and long prompts. Record output
tokens/s, TPOT, P50/P95 latency, acceptance length and peak NPU memory. Keep
chain speculation enabled only for workloads where measured end-to-end
latency or throughput improves. Automatic tuning must use Ascend measurements,
not the bundled CUDA cost profiles.
