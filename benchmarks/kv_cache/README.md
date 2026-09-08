# Typed heterogeneous KV-cache allocator prototype

This directory contains the reproducible code used to explore two-level,
typed-page allocation for hybrid Attention/Mamba-style caches on Ascend.  The
runtime prototype keeps scheduler block IDs local to each cache group and uses
a group-specific address table to translate them into a shared byte arena.

## Status and evidence boundary

The implementation is an experimental, default-off MVP for Ascend A3. Ascend
310P is explicitly rejected until its CPU fallback can translate logical block
IDs to physical addresses. It currently covers
cache sizing, no-prefix-cache scheduling, atomic multi-group admission,
logical-to-physical BlockTable translation, group-specific worker views, and
page clearing before reuse.

The separate `jenga_lcm_prefix` mode adds a tested policy reproduction for
cache-group/layer-type combinations of attention and recurrent state. It
covers prefix legality, five-tier allocation, small/large-page LRU eviction,
stale-hash invalidation, and in-place ownership rebind after metadata
invalidation; no payload migration is implemented. Its first runtime adapter
supports whole-page prefix hits on an exact-LCM layout only, clips scheduler
chunks at configured recurrent-state checkpoint boundaries, and rolls back
partly completed multi-group allocation/touch operations consistently. Correct
NPU state materialization at those boundaries has not yet been validated.
See [PREFIX_CACHE_REPRODUCTION.md](PREFIX_CACHE_REPRODUCTION.md) for commands
and the precise evidence boundary.

The framework-side allocator and address-translation tests pass, and offline
trace replay shows workload-dependent capacity potential.  The source tree now
contains two remedies for the GDN leading-stride blocker: a selectable
`triton-strided` decode backend for early correctness work and stride plumbing
through the AscendC Torch adapter, ACLNN/L0 interface, tiling data, and both
generic and arch35 kernels.  End-to-end NPU serving performance is **still not
a validated result** until the device-side tests and exact-token gates below
have been run on the rebuilt custom operator.

Generated datasets, prompts, model outputs, server logs, profiles, plots, and
result archives must be written under `benchmarks/kv_cache/datasets/` or
`benchmarks/kv_cache/results/`. Those two trees and `*.log` are ignored by
Git; review any output written elsewhere before sharing.

## Compared allocation policies

- `uniform-shared-page`: current uniform physical-page accounting baseline.
- `jenga-exact-lcm`: historical exact-LCM superpage experiment.
- `typed-superpage-*`: bounded-superpage design-space simulations.
- `static_partition`: fixed per-group regions using the same runtime address
  translation path as the dynamic experiment.
- `address_table`: current runtime experiment; released byte intervals can be
  retyped and reused by another cache group.

Exact LCM is not the final design. In a local cache-profile capture it worked
for the small-model geometry but grew to 391.5 MiB for the Qwen3.5-27B cache
specification, larger than the 374.5 MiB managed budget per shared layer
slice. The raw profile contains environment metadata and is not published.

## 1. Run the dependency-free allocator smoke test

From the repository root:

```shell
python -m benchmarks.kv_cache.run_jenga_experiment \
  benchmarks/kv_cache/scenarios/hybrid_demo.json
```

The bundled scenario is synthetic and must not be reported as a Qwen3.5 or
serving result.

## 2. Capture a model cache profile on an NPU

Install the matching vLLM revision and provide a model path explicitly:

```shell
python -m benchmarks.kv_cache.capture_kv_cache_profile \
  --model /path/to/model \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.8 \
  --output /tmp/kv-profile.json
```

The profile records cache-group page sizes, unpadded state sizes, scheduler and
kernel block sizes, virtual-splitting ratios, layer counts, and the raw cache
memory budget.  It contains environment-derived metadata, so inspect it before
publishing.

## 3. Build and replay an allocation scenario

```shell
python -m benchmarks.kv_cache.build_scenario_from_profile \
  /tmp/kv-profile.json \
  --token-lengths 4096,8192,16384,32768,49152 \
  --arrival-gap 1 \
  --duration 8 \
  --output /tmp/kv-scenario.json

python -m benchmarks.kv_cache.run_jenga_experiment \
  /tmp/kv-scenario.json \
  --json-output /tmp/allocator-result.json
```

For Mamba-style groups, the scenario builder models the `align` lifecycle:
Attention demand grows with sequence length, while only the active recurrent
states remain resident.

## 4. Run the local NPU microbenchmark

```shell
python -m benchmarks.kv_cache.run_typed_runtime_mvp \
  --profile /tmp/kv-profile.json \
  --device npu:0 \
  --num-superpages 8 \
  --iterations 1000 \
  --output /tmp/typed-runtime-mvp.json
```

This command checks allocation and release, group-local IDs, physical byte
offsets, Attention slot mapping, Mamba views, and the Ascend cache-scatter
kernel.  Its latency is a narrow kernel microbenchmark and must not be
extrapolated to end-to-end serving.

## 5. Runtime feature flags

The normal path is unchanged unless the experiment is explicitly enabled:

```shell
export VLLM_ASCEND_ENABLE_TYPED_KV_CACHE=1
export VLLM_ASCEND_TYPED_KV_CACHE_MODE=address_table
export VLLM_ASCEND_GDN_DECODE_BACKEND=triton-strided
```

`VLLM_ASCEND_GDN_DECODE_BACKEND` accepts `ascendc` (the default) and
`triton-strided`.  Ordinary decode uses the packed Triton decode kernel under
`triton-strided`; MTP/speculative and mixed decode use the general fused
recurrent kernel.  Select the Triton backend first for eager correctness
validation.  Rebuild the custom operators and select `ascendc` for the formal
performance run.

Use `static_partition` for the fixed-region comparison.  These two no-prefix
modes admit one fail-closed speculative subset for device validation: serial,
fixed-width Qwen3-Next or Qwen3.5 MTP with matching target/draft families,
`mamba_cache_mode=none`, and
`1 <= num_speculative_tokens <= 15`.  Every
Mamba group must reserve exactly `num_speculative_tokens` additional state
pages.  Parallel drafting, dynamic speculative widths, other speculative
methods/models, and `align`/`all` Mamba modes remain rejected.  The MVP also
rejects cache events, KV transfer/offload, DCP/PCP/PP, packed KV tensors, and
cross attention.

To exercise the experimental exact-LCM prefix path instead, enable vLLM prefix
caching and select:

```shell
export VLLM_ASCEND_TYPED_KV_CACHE_MODE=jenga_lcm_prefix
```

This path still rejects cache events, speculative decoding, KV
transfer/offload, DCP/PCP/PP, packed KV tensors, cross attention, and
fine-grained partial-block copy-on-write. It also requires vLLM's scheduler
watermark to be configured as zero. It is not usable for the measured
Qwen3.5-27B geometry because one exact-LCM large page exceeds that layer
slice's managed budget. The recurrent checkpoint interval defaults to 512
tokens and is coarsened to its least common multiple with the scheduler block
size when whole-page alignment requires it.

## 6. Correctness gate before performance testing

Do not publish serving performance unless all of the following pass:

1. Static and address-table modes produce identical token IDs or logits against
   an independent trusted baseline.
2. Tests cover padded leading stride, nonzero storage offset, and low/high GDN
   physical state indices.
3. All requests succeed and cache canaries/state checksums match.
4. Each workload runs after warm-up for at least three independent repetitions.
5. The report records exact source commits, image/CANN/custom-op versions,
   model and dataset identities, commands, seeds, and machine-readable outputs.

Raw experiment artifacts belong under `benchmarks/kv_cache/results/`; source
datasets belong under `benchmarks/kv_cache/datasets/`.  Both locations are
ignored so that code review cannot accidentally publish logs or prompt data.

## 7. GDN stride validation sequence

Run both direct-kernel regressions after rebuilding vLLM Ascend in an NPU
environment.  They use a nonzero storage offset, a padded leading stride, low
and high nonzero state indices, graph-padding rows, and canaries around every
state page.  Both ordinary decode and MTP-style state-table cases are covered:

```shell
pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_recurrent_gdn_strided_state.py

pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_recurrent_gated_delta_rule_strided.py
```

Then run the deterministic eager gate once per allocation mode with the
stride-aware Triton backend.  `precision.json` records the selected backend and
the generated token IDs, and `analyze_jenga_precision.py` reports the first
mismatching token rather than comparing only final text:

```shell
MODE=static_partition RUN_LABEL=static-triton-a \
MODEL_PATH=/path/to/model WORKLOAD_DIR=/path/to/workloads \
GDN_DECODE_BACKEND=triton-strided \
  bash benchmarks/kv_cache/run_jenga_precision.sh

MODE=static_partition RUN_LABEL=static-triton-b \
MODEL_PATH=/path/to/model WORKLOAD_DIR=/path/to/workloads \
GDN_DECODE_BACKEND=triton-strided \
  bash benchmarks/kv_cache/run_jenga_precision.sh

MODE=address_table RUN_LABEL=address-triton-a \
MODEL_PATH=/path/to/model WORKLOAD_DIR=/path/to/workloads \
GDN_DECODE_BACKEND=triton-strided \
  bash benchmarks/kv_cache/run_jenga_precision.sh

python benchmarks/kv_cache/analyze_jenga_precision.py \
  --static-a benchmarks/kv_cache/results/jenga-precision/static-triton-a/precision.json \
  --static-b benchmarks/kv_cache/results/jenga-precision/static-triton-b/precision.json \
  --address benchmarks/kv_cache/results/jenga-precision/address-triton-a/precision.json \
  --output benchmarks/kv_cache/results/jenga-precision/triton-comparison.json
```

Run a separate three-run comparison for MTP by supplying the same speculative
configuration to every invocation (do not mix ordinary and MTP JSON files in
one analysis):

```shell
SPECULATIVE_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
MODE=address_table RUN_LABEL=address-triton-mtp-a \
MODEL_PATH=/path/to/model WORKLOAD_DIR=/path/to/workloads \
GDN_DECODE_BACKEND=triton-strided \
  bash benchmarks/kv_cache/run_jenga_precision.sh
```

Repeat that command for `static_partition` twice and `address_table` once, then
pass those three `precision.json` files to the same analyzer.  The script
records the speculative configuration, and the analyzer rejects comparisons
whose backend, seed, output length, or speculative configuration differs.

If ACL Graph capture is unstable, keep this gate in eager mode as the script
does by default.  Once exact token IDs match, rebuild and repeat it with
`GDN_DECODE_BACKEND=ascendc`.  Only after that AscendC gate passes should
`run_jenga_paper_style.sh` be used for reportable performance measurements; its
GDN backend defaults explicitly to `ascendc`.
