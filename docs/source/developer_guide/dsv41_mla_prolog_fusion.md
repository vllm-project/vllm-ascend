# DeepSeek-V4.1 fused MLA prologue

## Background

DeepSeek-V4.1 runs a nine-step MLA prologue for every layer and every decode
step: `wq_a`, RMSNorm, dynamic quantization, the INT8 `wq_b` GEMM, partial RoPE
on q, `wkv`, the KV RMSNorm, partial RoPE on kv and the SWA cache scatter.
On the `enable_dsa_cp` path the prologue is replicated across the TP group while
only a rank-local slice of the queries is used, so the per-step launch and
Python overhead is paid for work that is almost entirely bandwidth/vector bound.

`dsv41_mla_prolog` fuses all nine steps into one AscendC kernel. The kernel
returns `q_output` and `qr` (the `q_norm(wq_a(x))` intermediate consumed by the
indexer) and publishes KV into the SWA cache in place.

## Operator

```
dsv41_mla_prolog(
    Tensor x,                 # [t, 5120]  bf16, contiguous rows
    Tensor wq_a,              # [1280, 5120] bf16
    Tensor gamma_q_a,         # [1280] bf16
    Tensor wq_b,              # [1280, 32768] int8, FRACTAL_NZ
    Tensor wq_b_scale,        # [32768] bf16 (per-channel)
    Tensor wkv,               # [512, 5120] bf16
    Tensor gamma_kv,          # [512] bf16
    Tensor rope_cos,          # [t, 1, 1, 64] f32 (interleaved)
    Tensor rope_sin,          # [t, 1, 1, 64] f32 (interleaved)
    Tensor slot_mapping,      # [t] int32 flat slots, < 0 skips the row
    Tensor(a!) kv_cache,      # [blocks, bs, 1, 512] bf16
    float epsilon             # both RMSNorms
) -> (Tensor q_output, Tensor qr)
```

* `q_output`: `[t, 64, 512]` bf16, query after the Q-B GEMM and partial RoPE.
* `qr`: `[t, 1280]` bf16, `q_norm(wq_a(x))`; required by the indexer.
* Rows with `slot_mapping < 0` are still projected but never written.
* 1 <= t <= 4096.

## Build

The kernel is compiled by the AscendC toolchain as an isolated CMake project
(`csrc/direct_kernels/dsv41_mla_prolog`) and linked into the C extension as
`libvllm_ascend_mla_prolog_direct.so`. Keeping the ASC language enablement in
its own project avoids interfering with the `ascendc_library` precompile
targets of the main build.

CATLASS **must be >= v1.6.1** (`csrc/third_party/catlass` is pinned to v1.6.3).
v1.5.0 brace-initializes `layout::zN` in
`QuantMatmulMultiStageWorkspace::ToUnderlyingArguments`, leaving every stride
zero so the Q-B GEMM reads only the first 32 weight columns while still
producing plausible-looking output. See cann/ops-transformer#16036.

## Integration

`AscendDSAV41Impl._preprocess` calls the op when `VLLM_ASCEND_DSV41_MLA_PROLOG=1`
and every layout assumption holds:

* contiguous bf16 2-D hidden rows; the query rows are a row-slice of the cache
  rows (checked by pointer/stride arithmetic);
* one rope table: the query cos/sin are the pointer-slice of the cache table;
* bf16, non-packed SWA cache with `[blocks, bs, 1, 512]` and inner stride 1;
* INT8 `wq_b` `[q_lora_rank, n_heads * head_dim]` with per-channel scale;
* equal `q_norm` / `kv_norm` epsilon;
* `flat_slot_mapping` present.

The gate falls back to the existing path when `weight_nz_mode=2` (that mode
converts the bf16 `wq_a`/`wkv` weights to FRACTAL_NZ, which the kernel does not
read) and during graph capture unless `VLLM_ASCEND_DSV41_MLA_PROLOG_GRAPH=1`.
The op is off by default, so existing deployments are unchanged.

Prefill chunks larger than 4096 tokens fall back to the unfused path; decode
(`t` = running sequences, or `sequences * (1 + num_speculative_tokens)` with
speculative decoding) uses the fused op.

## Verification

Operator level against the nine-op reference (`npu_quant_matmul` and the same
RMSNorm/RoPE/scatter semantics), single A3 chip:

| t | q_output matched / max_abs | qr | kv rows | sentinel |
|---:|---|---|---|---|
| 1 | 100.000% / 0.0 | 100% | 100% | exact |
| 96 | 99.988% / 1.0 | 100% | 100% | exact |

Whole network on 16 chips (DP4 x TP4 + EP, `enable_dsa_cp`), deterministic
prompts: fused and unfused outputs match; graph capture was exercised with
`VLLM_ASCEND_DSV41_MLA_PROLOG_GRAPH=1`.

## Performance

A3, 16 chips, DP4 x TP4 + EP, `enable_dsa_cp`, `--enforce-eager`,
`max-model-len 1048576`, gpu-memory-utilization 0.97, random 131072/1024
prompts, concurrency 16, MTP=5 (`dspark`), seed 1234:

| metric | baseline | fused | delta |
|---|---:|---:|---:|
| duration (s) | 439.18 | 376.95 | **-14.2%** |
| TTFT mean / median (s) | 67.81 / 67.77 | 53.78 / 53.48 | **-20.7% / -21.1%** |
| TPOT mean / median (ms) | 315.54 / 316.51 | 286.22 / 288.51 | **-9.3% / -8.8%** |
| ITL mean / median (ms) | 370.18 / 341.43 | 343.92 / 314.74 | **-7.1% / -7.8%** |
| output throughput (tok/s) | 37.31 | 43.47 | **+16.5%** |

`vllm bench` reports ITL per streamed chunk. With MTP the acceptance length on
these random prompts is only ~1.2 tokens per step, so ITL is close to the
per-step latency and TPOT tracks the per-token latency.

Single-operator latency (idle chip, includes the torch adapter allocations):

| t | steady (ms/call) | per-call mean (ms) | throughput |
|---:|---:|---:|---:|
| 1 | 0.054 | 0.077 | 18.6 K tok/s |
| 24 | 0.061 | 0.085 | 396 K tok/s |
| 96 | 0.089-0.096 | 0.112-0.124 | ~1.0 M tok/s |

msprof at t=96: 88.5 us per launch, 20 AIC / 40 AIV (whole chip); AIV busy
84.6 us (critical path), AIC MAC 22.7 us (~390 TOPS for the Q-B GEMM) and AIC
MTE2 30.9 us for the 41.9 MB weight stream (~1.36 TB/s). The nine-op chain at
the same shape spends 131.6 us of device time (44.2 us in the NZ Q-B GEMM) and
nine launches; eager wall clock is ~0.29-0.39 ms per call versus ~0.09-0.11 ms
for the fused op.

## Limitations

* `weight_nz_mode=2` is not supported by the kernel (bf16 weights are NZ); the
  gate falls back. Supporting it needs an NZ bf16 `wq_a`/`wkv` reader.
* Prefill chunks above 4096 tokens use the unfused path.
* The op is enabled per process through the environment variable; a
  configuration flag can replace it once the layout coverage grows.
