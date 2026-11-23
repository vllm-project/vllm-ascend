# Experimental A3 short-query FIA workaround

Set `VLLM_ASCEND_FIA_SHORT_QUERY_WORKAROUND=1` only on the affected,
validated A3 runtime. The default is `0`. The switch contains a numerical
workaround; it is not a general attention performance option.

## Scope

The device adaptor handles BF16 paged TND attention with head dimension 256,
eight query heads per KV head, sparse mode 3, and no LSE output. CPU cumulative
query lengths must describe a batch whose largest query length is at most 16,
with at least one request of length 9 through 15. KV lengths must be CPU
integers of at least 32. Calls with `query_rope` or `pse_shift` bypass the workaround.

Every nonempty request in a qualifying batch is front-padded to 32 queries.
For cached length K, original query length Q and query index j, the causal
position remains `K - (Q + padding) + (j + padding) = K - Q + j`.
Padded output rows are discarded; K/V and recurrent state are unchanged.
Padding adds temporary tensors and attention work, especially in mixed batches.

This does not intercept direct native `.out` calls used by full graph replay.
Batches containing query lengths above 16 bypass it. Other shapes, short caches,
quantized caches and software versions need separate numerical validation.
Remove the workaround after the native operator is fixed and verified.

## Validation and limitations

The reproduced environment was A3, CANN 9.2.1, torch 2.10.0 and
torch-npu 2.10.0.post7, with 128-token cache blocks. Independent BF16 native
FIA tests used FP32 attention as reference, without FLA, GDN or P/D transfer.
One 10-query case had maximum absolute errors 0.01207 / 0.00004267 / 0.10850
for native / front-padded / native execution. A model probe found nonfinite
values first in FIA output, after finite Q/K/V.

The accompanying W4A8 change uses the routed-expert TP size before weight
allocation. ModelSlim divides 16 scale-bias partitions by that TP size, matching
the loader. NaN-sentinel tests exercise the actual loader to detect unwritten
columns. Compressed-tensors weights do not have this 16-partition restriction.

Historical 3P1D validation used the same A3 resources, Qwen3.6 W4A8 weights,
64K/1K, concurrency 24, 96 fixed requests, no prefix cache and MTP3.
The optimized environment with both candidate fixes completed 96/96 requests:
TTFT 9.789 s, TPOT 12.029 ms, output throughput 1013.381 tokens/s and MTP
acceptance 81.046%. Blank and dominant-word repetition screens found zero cases.
Six role-word flags still need semantic interpretation because EOS was ignored
to force 1024 output tokens. These are screening results, not accuracy scores.

The matched clean candidate completed 96/96 with TTFT 9.466 s, TPOT 13.139 ms,
974.344 tokens/s and MTP acceptance 80.871%. These end-to-end measurements include
other environment/code differences and do not isolate the benefit of either fix.
The original optimized environment yielded 466.627 tokens/s and 18.711% MTP
acceptance; the FIA-only candidate yielded 944.540 tokens/s and 78.687%.

Residual DP1/DP2 routed-MoE numerical differences remain. Reverting the W4A8
fix did not reliably reproduce model repetition, so the allocation defect is
confirmed independently but is not proven to explain every model anomaly.
The hardware runs predate the additional short-KV/optional-input bypass guards;
CPU dispatch tests cover those guards. CI and broader hardware validation remain
required before enabling the workaround by default.
