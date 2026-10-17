# Experimental KDA RecoverSSM

GLM-5.3-Flash can replace its speculative KDA recurrence with Ascend Triton
verify and commit kernels. Existing AscendC prefill, ordinary decode and causal
convolution remain in use. No upstream vLLM source changes are required.

This is an experimental bridge for numerical and performance investigation.
It retains the original recurrent cache allocation and mirrors each committed
checkpoint into the native speculative slots. It does **not** reduce the serving
KV pool. Scratch allocation, gathers, dynamic request selection and writeback
add overhead and are included in end-to-end measurements.

## Enable the experiment

The non-sensitive environment variable
`VLLM_ASCEND_ENABLE_EXPERIMENTAL_KDA_RECOVERSSM` accepts `0` (default, original
AscendC verification) or `1` (experimental RecoverSSM). Unsupported configurations
fail during KDA layer initialization.

```bash
export VLLM_ASCEND_ENABLE_EXPERIMENTAL_KDA_RECOVERSSM=1
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve /path/to/GLM-5.3-Flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"enforce_eager":true}'
```

Use the model's usual Ascend quantization and deployment options. The current
experiment supports MRV1, eager execution, synchronous scheduling, MTP, PP=1
and DP=1. Prefix caching and KV transfer are unsupported. Graph execution,
asynchronous scheduling and cache allocation savings require further work.

## State lifecycle

1. Each target KDA layer owns one adapter. Verify reads the checkpoint selected
   by the previous accepted length, generates outputs and records correction,
   key and raw-gate vectors. Native recurrent state remains unchanged.
2. Immediately after rejection sampling, MRV1 passes sampled token IDs to each
   adapter. IDs padded with `-1` are excluded; the count includes the leading
   target token. The speculative request mask maps packed KDA rows to sampler
   rows when speculative and ordinary requests share a batch.
3. Commit reconstructs only the accepted prefix and mirrors the resulting state
   into the existing native slots. The next MRV1 copy and index selection therefore
   retain their original contract. The adapter releases the step's records.
4. Causal convolution continues using its existing accepted-length history
   handling. The adapter neither rewrites convolution history nor saves prefix
   checkpoints. A second verification before commit fails instead of silently
   replacing pending records.

The adapter has no process-global mutable queue. Its pending tensors live only
between target forward and sampling in one synchronous step. The runner collects
target adapters after loading the model and commits them before MTP drafting.
The bridge can be replaced when compact-cache allocation and metadata have a
validated production implementation.

## Validate and profile

```bash
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_recoverssm_npu.py
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_recoverssm_adapter_npu.py
python benchmarks/benchmark_kda_recoverssm.py --output benchmark_kda_recoverssm.json
```

For service profiling, use the official `--profiler-config` and
`/start_profile` / `/stop_profile` workflow in the
[service profiling guide](service_profiling_guide.md). Run throughput and latency
benchmarks outside the profiling window. Compare identical workloads with the
experimental flag disabled and enabled. Measure speculative counters before and
after each workload; report accepted draft counts and rates for every position.

Small-norm operator tests do not establish identical generated token sequences
or model accuracy. Serving-level accuracy and performance validation are required
before treating this draft as a production optimization.
