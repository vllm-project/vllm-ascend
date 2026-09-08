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

The framework-side allocator and address-translation tests pass, and offline
trace replay shows workload-dependent capacity potential.  End-to-end serving
performance is **not** a validated result: a correctness gate found that the
custom GDN decode operator treats a page-strided recurrent-state view as dense.
Until that operator accepts and applies the leading state stride, throughput,
TTFT, TPOT, and generated-text comparisons from this prototype are diagnostic
only.

Generated datasets, prompts, model outputs, server logs, profiles, plots, and
result archives are intentionally ignored by Git.  Recreate them locally and
review them before sharing.

## Compared allocation policies

- `uniform-shared-page`: current uniform physical-page accounting baseline.
- `jenga-exact-lcm`: historical exact-LCM superpage experiment.
- `typed-superpage-*`: bounded-superpage design-space simulations.
- `static_partition`: fixed per-group regions using the same runtime address
  translation path as the dynamic experiment.
- `address_table`: current runtime experiment; released byte intervals can be
  retyped and reused by another cache group.

Exact LCM is not the final design.  It worked for the small-model geometry but
grew to 391.5 MiB for the Qwen3.5-27B cache specification, larger than the
374.5 MiB managed budget per shared layer slice.

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
```

Use `static_partition` for the fixed-region comparison.  The MVP rejects
prefix caching, cache events, speculative decoding, KV transfer/offload,
DCP/PCP/PP, packed KV tensors, and cross attention.

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
