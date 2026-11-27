# CatCCOS: DeepSeek-V4-Flash integration

The `catccos` FusedMC2 backend accepts official DeepSeek-V4-Flash MXFP4
expert weights, one shared expert and SwiGLU clamp=10 on Ascend 950DT.
The shared expert runs outside the routed kernel. Official E8M0 scales
are preserved; expert weights are not requantized.

Use the matching CatCCOS kernel branch
`codex/deepseek-v4-flash-catccos`. CANN's MX MMAD interface requires FP8
weights, so its wrapper exactly expands E2M1 values into reusable E4M3
scratch on each invocation. Every E2M1 value is exactly representable in
E4M3. The scratch costs 1.5 GiB per rank for this model and its expansion
time is included in inference latency. Expanded weights are not cached
per layer.

## Configuration

The validated experiment uses NPUs 0–3, DP4/TP1/EP4, BF16, eager,
max-model-len=8192, max-num-batched-tokens=4096, max-num-seqs=32,
no MTP, no prefix cache and seed42. Additional configuration:

```json
{
  "enable_fused_mc2": 1,
  "fused_mc2_backend": "catccos",
  "catccos_library_path": "/absolute/path/libcatccos_torch.so",
  "catccos_store_url": "tcp://127.0.0.1:27041",
  "catccos_local_mem_size": 1073741824,
  "catccos_min_tokens": 1,
  "catccos_max_tokens_per_rank": 4096,
  "catccos_sync_after_launch": false,
  "enable_prefill_mc2": true,
  "multistream_dsv4_dsa_overlap": false
}
```

Source CANN and SHMEM environment scripts before launching. The custom
OPP package must contain the Ascend950 InplacePartialRotaryMul binary
metadata; `csrc/build_aclnn.sh` generates and verifies that index.

The common comparison uses GPU memory utilization0.80 and
HCCL_BUFFSIZE1536. Native MC2 startup needs at least2225MB actual
communication space for the profiling shape; this configuration supplies
3072MB. Both settings apply equally to all three backends.

## Validation

The CatCCOS adapter/configuration regression suite passed62 tests. Four-rank
synthetic tests include independent CPU E2M1/E8M0 decoding, activation
MXFP8 rounding, clamp10, top-k6, M1/32/4096, model-size routing and repeated
asynchronous calls. Packed FP4 and independently expanded FP8 outputs in
the same kernel were identical. See the kernel branch's
`tests/torch_binding/MXFP4_VALIDATION.md` for details and runnable tests.

The real checkpoint passed140/140 full-model smoke/stress requests with
diagnostic synchronization disabled. They cover ISL128/2048/4096,
concurrency8/32, OSL256 decode and OSL128 mixed requests, numeric finite
log probabilities and an arithmetic sanity check. These checks do not
establish task accuracy or provide a formal performance score.

Long-input tests previously encountered device address faults at a0.90
memory budget. They passed at0.80 with DSA attention overlap disabled.
The exact origin of the0.90 fault has not been established.

The server retains the launch harness, fixed data hashes, synthetic tests
and full-model logs under `work/deepseek-v4-test/results/full-20261011`.
