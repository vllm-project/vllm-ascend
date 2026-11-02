# DeepSeek-R1-Distill on Atlas 300I DUO

DeepSeek-R1-Distill Qwen and Llama checkpoints are dense text models. They use
the upstream `Qwen2ForCausalLM` and `LlamaForCausalLM` implementations. They do
not require DeepSeek MoE or MLA model patches.

## Validation scope

Qwen-1.5B and Llama-8B validation used one 310P3 device, FP16, tensor parallelism
of one, and a 1024-token context limit. Eager and `FULL_DECODE_ONLY` produced
identical generated token IDs over three rounds of alternating batch sizes
4, 1, 3, and 2. Every request generated 32 tokens with EOS stopping disabled.
The graph logs also confirmed real replay. Llama used the separate verified
tokenizer directory described below. This short comparison establishes
mode consistency for those inputs; it does not establish benchmark accuracy
or long-running memory stability.

That measurement used vLLM 0.23.0 and vLLM-Ascend 0.23.0.post1 with CANN 9.1.0,
PyTorch 2.10.0, and TorchNPU 2.10.0.post4. It is release-stack evidence, rather
than a hardware validation of current main. Select a complete compatible stack
from the [version matrix](../../community/versioning_policy.md#release-compatibility-matrix)
when installing a different release.

Do not extrapolate this result to Qwen-7B/14B/32B or Llama-70B. Each checkpoint
and tensor-parallel topology needs its own real-weight validation.

With the query-prescale fix below, Qwen-1.5B additionally completed 100 graph
rounds, 400 alternating-size batches, and 1000 requests. All 32000 generated
graph tokens, and the 960 tokens from three eager rounds, exactly matched an
independent CPU FP32 Transformers reference. Actual graph replay was logged.
After excluding ten warmup rounds, 90 device-wide memory samples ranged from
21140 to 21141 MB, with the first and last both 21141 MB. This fixed-workload
observation does not establish leak freedom or general model accuracy.

## Deployment

### Check tokenizer compatibility

Some Transformers 5 releases select `LlamaTokenizer` for a local copy of
DeepSeek-R1-Distill-Llama-8B even though its `tokenizer.json` describes byte-level
BPE. This can drop whitespace and encode Chinese text as an empty sequence.
An eager/graph comparison cannot detect a common tokenizer error in both modes.

Check encoding against the checkpoint's `tokenizer.json` before evaluating
quality. If affected, create a separate tokenizer directory with the supplied
compatibility tool:

```bash
python examples/prepare_deepseek_distill_tokenizer.py \
  /path/to/DeepSeek-R1-Distill-Llama-8B \
  /path/to/llama-distill-tokenizer
```

The tool copies tokenizer assets, changes only the tokenizer class metadata to
the generic fast backend, and verifies token IDs and whitespace/Chinese
round-trips against the original definition. It also checks that the chat
template is preserved and writes a hash manifest. It refuses to overwrite an
existing destination. Add `--tokenizer /path/to/llama-distill-tokenizer` to
the service command when using this workaround. Do not edit the weight files
or remove Chinese prompts to bypass the failure.

This is a checkpoint/tokenizer compatibility workaround, rather than an NPU
kernel patch. See the [upstream tokenizer issue](https://github.com/huggingface/transformers/issues/45488).

### Start the service

Load the CANN and NNAL environments provided by your installation. On 310P,
use FP16 execution for these BF16 checkpoints, and explicitly bound context
and graph capture sizes. Triton and Triton Ascend are not used on 310P.

Download and pin the checkpoint revision before starting the service. Set
`MODEL_PATH` to the actual directory returned by your downloader; do not assume
a particular ModelScope cache layout. Record checkpoint hashes and tokenizer
files with the environment and source commits used for validation.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B

vllm serve "${MODEL_PATH}" \
  --served-model-name distill \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

For the eager comparison, replace `--compilation-config` with `--enforce-eager`
and keep the other arguments unchanged. Run the two services sequentially so
that their weights and KV caches do not coexist on the same device.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"distill","prompt":"Calculate 17 + 25. Explain your answer.","max_tokens":32,"temperature":0}'
```

A nonempty response proves only that generation ran. Compare generated token
IDs for mode consistency, and evaluate answers against a separate reference
dataset for accuracy. The displayed context limit is a tested resource bound,
rather than the checkpoint's theoretical maximum context length.

## Regression and replay checks

For unquantized Qwen2 FP16, the 310P backend pre-scales queries by 1/8 and
compensates the attention scale. This addresses intermediate QK overflow seen
with Distill Qwen-1.5B: FP32 profiling found unscaled scores around 310k,
although the final scaled scores fit FP16. The cached keys and values are
unchanged. This does not establish a numerical bound for larger checkpoints
or longer contexts.

Check against an independent finite reference as well as comparing execution
modes. The initial uncorrected Qwen eager and graph outputs agreed while both
were incorrect. CPU FP16 Transformers also produced nonfinite logits for this
checkpoint; the diagnostic reference therefore used CPU FP32. Preserve this
distinction when interpreting an eager/graph parity result.

Run the real-weight regression explicitly on 310P:

```bash
VLLM_USE_MODELSCOPE=True python -m pytest -v --capture=fd \
  tests/e2e/pull_request/one_card/_310p/test_deepseek_distill_replay_parity_310p.py \
  -o junit_family=legacy --junitxml=distill-parity.xml
```

The regression creates eager and graph engines sequentially, disables prefix
caching, and forces multiple decode steps. The three-request batch exercises
padding into the four-request capture bucket; alternating batch sizes and
repeated requests exercise updates to persistent graph inputs. The output
comparison uses generated tokens, excluding the prompt prefix.

Capture completion alone is insufficient evidence of replay. Current main
logs `ACL graph replay is active`; the 0.23.0.post1 release logs
`Replaying aclgraph`. Check the source for the version being tested and retain
the worker logs. CPU mocks and static checks cannot establish this behavior.

## Performance and memory measurements

### Bounded accuracy regression

The dedicated `tests/e2e/models/configs/accuracy_310p_distill.txt` suite records
independent 32-sample GSM8K baselines using lm-eval 0.4.12, five chat examples,
and at most 512 generated tokens per question. With the Qwen query-prescale
fix, flexible answer extraction measured 0.21875 (7/32); the initial uncorrected
Qwen run measured zero. Llama with the verified tokenizer measured 0.40625
(13/32). Strict-format matching was zero for both. These are bounded regression
observations, not full-dataset model accuracy or the model author's benchmarks.

The configs pin Hugging Face revisions verified against the measured weight
hashes. For affected Llama tokenizers, make a local copy of its config and set
`serve.tokenizer` to the verified directory prepared above. A local checkpoint
may similarly replace `model_name` after verifying its hashes. Run the common
evaluation entry point on 310P with the selected config:

```bash
VLLM_USE_MODELSCOPE=False python -m pytest -v \
  tests/e2e/models/test_lm_eval_correctness.py \
  --config /path/to/verified-local-config.yaml --tp-size 1
```

Keep this suite separate from default A2/A3 accuracy jobs. When fetching a
revision-pinned Hugging Face checkpoint, use the Hugging Face downloader;
ModelScope revision identifiers need not equal Hugging Face commit hashes.

### Serving measurements

Use streaming requests to measure TTFT and TPOT. Offline batch wall time
includes prefill, decode, and client overhead and must not be labeled TTFT.
Use the same checkpoint, input lengths, output lengths, concurrency, seed,
and common engine arguments for eager and graph. Warm up before measuring,
run multiple rounds, and retain failures along with successful results.

For vLLM 0.23.0, a fixed-length random workload uses
`--random-range-ratio 0.0`; a ratio of one permits zero-length inputs. Verify
the parameter semantics in the installed version before running benchmarks.

Record capture time separately from process startup and weight loading. The
initial Qwen capture log reported approximately two seconds and 0.15 GiB of
NPU graph memory; neither number is a measurement of total device memory.
Sample `npu-smi` throughout sustained traffic and report the actual field
names. On the tested 310P3 installation, `npu-smi` exposes `Memory-Usage`,
rather than a separate HBM counter.

See the [performance guide](../../developer_guide/performance_and_debug/performance_benchmark.md)
for the serving benchmark workflow. Report request failures, generated token
counts, TTFT and TPOT percentiles, throughput, and post-warmup memory trends.
Short smoke tests do not establish leak freedom or general model quality.
