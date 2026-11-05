# Owner-local Kimi Top-K output

On the native sharded Kimi indexer path with physical PCP size one, candidate
merge can return SFA owner-local indices. PCP and TurboQuant retain global
output: their query exchange or full-history prefill needs global coordinates.
The indexer and SFA use one static layout predicate, including shared-index
layers, so `topk_indices_buffer` never mixes coordinate systems across steps.

Candidate exchange packs IDs and score bits for one AllGather per query tile,
bounded at 64 rows independently of prompt history. Merge writes each tile
directly into the final output buffer without a concatenation. Global
selection still uses the native comparator score, descending, with ascending
global token ID for ties. Only after selecting the global K candidates does
the final kernel filter by owner, remap, and compact in selected order. It
writes INT32 indices with the existing shape and pads the tail with -1.
The SFA consumer skips its separate remap only for this local-output layout.
The final cross-rank attention output merge remains required.

The packed path fuses score masking, local remap and compaction into the
existing selection kernel. All batch sizes use the same packed path.
Compaction sorts unique FP32 position keys,
reusing the A5 remap algorithm; index values are gathered as INT32 payload
bits. Neither scores nor the global selection comparator are approximated.

CPU tests exercise merge selection and routing with a test-only reference
finalizer, including ties, nonfinite scores, empty owners and query tiles.
Production local finalization always calls Triton and propagates failures.
NPU tests additionally compare the fused packed path with the existing remap
and cover changed graph inputs. To validate the local output path on an
allocated A5 device:

```bash
pytest -q tests/e2e/pull_request/one_card/test_dcp_pre_post.py
```

No whole-model speedup or precision acceptance follows from the operator tests alone.
