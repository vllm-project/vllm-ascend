# Reuse host sequence boundaries in 310P GDN prefill

The chunk-prefill Python wrapper consumes cumulative sequence boundaries on
the CPU. Passing `non_spec_query_start_loc` from the NPU causes an int64
conversion and a device-to-host synchronization in each GDN layer, even though
the attention builder already retains the same boundaries in
`non_spec_prefill_metadata.chunk.cu_seqlens_host`.

The 310P model wrapper now materializes an int64 CPU tensor from that immutable
tuple. This changes only metadata routing, not tensor arithmetic, sequence
order, accepted-path handling or native operator implementations.

## Scope and fallback

- Reuse host metadata only when there are no ordinary decode rows in the batch.
  A mixed speculative-decode/prefill batch may still use its non-spec host data.
- Retain empty sequence segments; do not substitute compacted kernel metadata.
- Fall back to the original device path if host metadata is missing or its
  number of boundaries does not match the device tensor's extent.
- Mixed ordinary decode/prefill remains on the original path: its host chunk
  metadata excludes ordinary decode rows and cannot replace the full non-spec
  boundaries.
- Causal convolution and recurrent decode continue receiving device metadata.
  No graph policy, native ABI, tiling, buffers or compute_wy code is changed.

## Validation

Nine CPU-only selector tests cover immutable ownership, 1280/768/2048-token
chunks, empty segments, missing/mismatched metadata and absence of device-value
reads. Together with existing GDN, graph-capability and chunk-reference tests,
27 targeted CPU tests pass.
The integration regression suite also passes (141 tests), as do seven benchmark
helper tests. Ruff/pre-commit are unavailable in this environment; this is not a
claim of passing the full repository lint pipeline.

One-prompt 310P3 full-model comparison: Qwen3.5-9B W8A8, FP16 DFlash body,
shared pruned W8A8 head, custom QBMM/compute_wy, eager, async scheduling, batch 1,
greedy, 2048 input and 1024 output, block 16, prefill budget 1280.
Three repeats after warmup on a monitored NPU without a foreign NPU process:

| Version | TTFT mean, ms | TPOT mean, ms/token |
| --- | ---: | ---: |
| Before (`0e284fd28`) | 1438.635 | 32.351 |
| Host boundaries | 1405.542 | 32.672 |

All 1024 output token IDs match the original control in all three repeats.
Acceptance remains 782/3720 (21.0215%), with 248 verification cycles and 4.125
emitted decode tokens per verification. TTFT falls 2.30%; the approximately
1% TPOT difference is reported rather than hidden. The decode code is unchanged.
This is a narrow repeated workload result, not a broad performance guarantee
or a proof of full-model AR equivalence.

With the same model/prompt and a 4096-token budget (one full prefill), three
additional repeats have TTFT 1313.268 ms versus 1344.548 ms before. All 1024
output IDs again match that configuration's original control, with unchanged
707/4770 accepted tokens and 318 verification cycles. The 1280-budget run is
still preferable for this long-output prompt: its decode TPOT is about 32.7 ms
versus 42.6 ms with the larger budget. Changing chunk boundaries changes this
model's generation trajectory, so budget 4096 is not promoted as a default.

The historical sub-second HTTP benchmark used 1442 tokens, not 2048: its input
length was specified in characters. Its ordinary-MTP/vLLM-version/streaming
configuration also differed. This patch does not claim sub-second TTFT at
2048 input tokens and does not constitute a new native-kernel validation gate.
