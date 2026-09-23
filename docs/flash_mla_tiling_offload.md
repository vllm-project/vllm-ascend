# Minimal MRV2 FlashMLA tiling offload

This opt-in integration connects the external FlashMLA metadata operator to
the attention operator. It does not implement a new kernel or cache allocator.
The dependency baseline supplies token-fused, page-strided MLA cache views.

## Data flow

1. MRV2 supplies device sequence lengths, query boundaries, slots and block tables.
2. `flash_mla_with_kvcache_metadata` generates a fresh schedule per batch on the
   current stream. No device length is converted to a CPU list by this adapter.
3. Ordinary projections form absorbed Q. The stride-aware writer updates the
   existing cache through its latent/RoPE slices, preserving page and token strides.
4. `flash_mla_with_kvcache` consumes Q, the original cache and the same batch's
   lengths, query boundaries and schedule.
5. Head-major latent output feeds the existing V-up, optional gate and O projection.
   Physical padding is excluded from cache writes and masked before projection.

The runtime still performs ordinary scheduling and may have other CPU length
users. This change does not claim removal of all engine synchronization.

## Contract and scope

- Default off: `VLLM_ASCEND_ENABLE_FLASH_MLA=0` (valid values `0` and `1`).
- Enable with `VLLM_ASCEND_ENABLE_FLASH_MLA=1`, `VLLM_USE_V2_MODEL_RUNNER=1` and
  `--enforce-eager`. Requires A5, BF16 dense MLA, unquantized BF16 KV, PCP/DCP=1.
- Both external operators must be installed in `cann_ops_transformer.ops`.
  Missing operators fail; there is no silent FIA fallback.
- Q: `TND [T,H,576]`, actual local heads 8/12/64/96; no replicated/fake heads.
- KV: `PA_BBND [P,128,1,576]`, original storage offset, non-overlapping page/token
  strides and contiguous channels. The cache is never made contiguous or repacked.
- Both calls pass `max_seqlen_q=max_seqlen_kv=-1`. Actual lengths come from
  device tensors; these attributes are not buffer capacities.
- Pass cumulative `cu_seqlens_q` and actual `seqused_q` to both calls. Physical
  padding belongs to a final zero-used row, with cache slots set to -1.
- Causal mode 3 passes the 2048-square upper-triangular int8 mask (ones above
  the diagonal); mode 0 passes no mask. Main consumes metadata's schedule.
- Output: `NTD [H,T,512]`; LSE is not requested because this route has no merge.
- RoPE layers rotate their 64-channel operands; NoPE layers retain those channels
  without rotation. Models with a zero-dimensional RoPE operand are not supported.
- No C8, speculative/DSpark integration, DCP merge, PD/KV transfer or graph
  lifecycle is added. Enabling unsupported execution modes fails explicitly.
  Other backends in a hybrid model are unchanged.

This increment is extracted from the earlier public FlashMLA handoff; it does
not import that snapshot's DSpark, graph, DCP helpers or fused output kernels.
Cache layout/lifetime changes belong to the dependency branch, not this increment.

## Validation and reproduction

Local CPU tests exercise real tensor metadata operations and a mocked operator
boundary. They do not establish binary compatibility, NPU numerical correctness,
model correctness or speedup:

```bash
python tests/ut/attention/test_flash_mla_host_contract.py
```

On an authorized A5 environment with a compatible vLLM/Ascend installation and
the external package, run the real operator smoke (no mocking):

```bash
python tests/e2e/nightly/single_node/ops/flash_mla_tiling_smoke.py --heads 8
```

Repeat with local heads 12/64/96. It checks causal attention against CPU float32
reference, 127/129/257 KV lengths, Q=1/2/16, permuted pages, page/token strides,
nonzero offset, real cache writes and exact guard preservation. Output tolerances
are fixed at `atol=rtol=0.02`; cache and guards require exact equality.

Next run a supported dense MLA model with the switches above, comparing against
the same dependency baseline with the switch off. Use identical weights, prompts,
sampling seed, token budget and parallel configuration. Cover cold/chunked prefill,
prefix reuse, mixed batches, multiple decode steps and cache COW/zeroing regressions.
Record both repository SHAs, package/CANN/torch_npu versions and commands with logs.

Real-package smoke, model eager, long contexts (100K/128K), Batch 29-32 and
performance remain **pending** until run on the candidate SHA. FP16/PA_Nz package
capability tests are outside this BF16/BBND integration, not declared unsupported
by the operator. Graph and DSpark are not claimed by this Draft.
