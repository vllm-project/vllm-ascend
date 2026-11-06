# GLM-5.3-Flash MTP adaptive verification

This opt-in Model Runner V2 path adapts [vLLM #60229](https://github.com/vllm-project/vllm/pull/60229)
to Ascend's KDA and CANN KPool implementation. It chooses verification lengths
on the device using vLLM's existing acceptance estimator and cost model.
`num_speculative_tokens` remains the maximum draft depth.

Use a vLLM main build containing #60229 and the matching Ascend source tree.
The release-specific Model Runner V1 deployment commands in the
[GLM-5.3-Flash tutorial](../../tutorials/models/GLM5.3-Flash.md) do not enable this path.

## Configuration

The initial scope is a colocated Model Runner V2 instance with PP=1, PCP=1,
DCP=1 and RecoverSSM disabled. Tensor parallelism can shard the model.
Use the model's normal BF16 activations and the quantized checkpoint appropriate
for the hardware. The current KDA limit is seven speculative tokens.
Upstream adaptive-verification constraints, including full decode graphs,
no LoRA and no local-argmax reduction, still apply.

Add these options to the checkpoint's working colocated serving command:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve /path/to/GLM-5.3-Flash \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 16 \
  --quantization ascend \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"enable_adaptive_verification":true}'
```

Select tensor parallelism and memory settings for the actual checkpoint and
available devices. The configuration above is an example, not a measured sizing
recommendation. Set `enable_adaptive_verification` to `false` for fixed MTP.
The feature remains disabled by default.

## Correctness and performance validation

Hardware validation of this adaptation is pending. CPU source execution and
mocked tests cannot establish NPU compilation, full-model accuracy or speedup.
Run the following tests in the matching NPU environment:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/models/test_glm5next_adaptive.py \
  tests/ut/models/test_glm5next_kda_contracts.py \
  tests/ut/models/test_glm5next_kda_dispatch.py \
  tests/ut/models/test_glm5next_conv_dispatch.py \
  tests/ut/models/test_glm5next_kpool_model_backend.py \
  tests/ut/ops/test_gdn_attn_builder.py

pytest -q \
  tests/e2e/pull_request/one_card/test_glm53_adaptive_kda.py \
  tests/e2e/pull_request/one_card/test_glm53_adaptive_kpool.py

pytest -q tests/e2e/pull_request/four_card/test_glm5_3_flash.py -k adaptive_graph
```

The operator tests compare recurrence and convolution against independent CPU
references, including accepted history longer than the current query, zero-length
requests, cache preservation, stale graph-padding slots and changing request
partitions during ACL graph replay. The four-card model uses synthetic weights;
it is a functional test, not an accuracy or throughput benchmark.

For the full checkpoint, run the comparison harness in a fresh process for each
mode. Supply checkpoint-specific deployment arguments through a JSON file,
for example `{"quantization":"ascend","enable_expert_parallel":true}`.

```bash
python -m benchmarks.glm53_adaptive_verification \
  --model /path/to/GLM-5.3-Flash --tensor-parallel-size 16 \
  --engine-args /path/to/engine-args.json \
  --mode plain --output /tmp/glm53-plain.json

python -m benchmarks.glm53_adaptive_verification \
  --model /path/to/GLM-5.3-Flash --tensor-parallel-size 16 \
  --engine-args /path/to/engine-args.json \
  --mode fixed --output /tmp/glm53-fixed.json --reference /tmp/glm53-plain.json

python -m benchmarks.glm53_adaptive_verification \
  --model /path/to/GLM-5.3-Flash --tensor-parallel-size 16 \
  --engine-args /path/to/engine-args.json \
  --mode adaptive --output /tmp/glm53-adaptive.json --reference /tmp/glm53-plain.json
```

The harness records software versions, revision, hardware, per-rank peak device
memory, repeated generation throughput and separate greedy tokens/logprobs.
It fails on greedy token mismatches; inspect shared-prefix logprob differences
before attributing a mismatch to numerical sensitivity. Do not raise tolerances
to hide a divergence. Generation throughput includes prefill and decode; measure
serving TTFT and TPOT with `vllm bench serve` as well for the deployment workload.
Repeat at the same draft depths, batch sizes and context lengths on the baseline
and candidate. Adaptive verification is workload-dependent and does not imply a
speedup at every concurrency. Run the deployment's established AISBench accuracy
suite in plain, fixed and adaptive modes before accepting the change.
