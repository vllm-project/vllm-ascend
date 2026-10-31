# MiniMax-M3 GBSA decode preparation

MiniMax-M3 retains separate decode index-score implementations: AscendC on
A2/A3 and Triton on A5. Both model paths request ordered block IDs from their
TopK postprocessor and pass `indices_are_sorted=True` to GBSA. The attention
wrapper retains its normalizer for direct callers that do not opt into this
contract.

The postprocessor filters invalid IDs, computes per-query valid counts, sorts
FP32 keys with invalid slots placed after valid blocks, and stores int32 IDs
with trailing `-1`. Block IDs must be exactly representable in FP32. The A5
wrapper preserves preallocated output addresses and strides; its TopK and
sequence-length inputs are normalized to contiguous storage before launching
the kernel. Normal model inputs already satisfy this layout.

TopK=16 uses four query rows per program and 32 internal sorting slots. Extra
slots are masked and never written to the output. Other TopK widths use one
query row per program. This is a launch policy, not a change to the selected
block count or the attention computation.

GBSA metadata reuse and uniform decode query prefixes are shared with A5.
Metadata remains forward-local, with persistent buffers refreshed before the
captured layer consumers. Both metadata and attention use `softmax_precision=1`
for E4M3 FP8 KV and `0` for BF16 KV.

## Validation

CPU wiring, output-buffer contracts, and metadata regression checks:

```bash
pytest -q tests/ut/models/minimax_m3/test_msa_m3.py
```

Exact fused block ordering, non-contiguous storage, and changing graph inputs:

```bash
pytest -q tests/e2e/pull_request/one_card/test_minimax_m3_decode_index_sort.py
```

A5 BF16/FP8 indexer and prepared GBSA graph execution:

```bash
pytest -q tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py \
    -k 'prepared_gbsa_counts_match_a5_indexer or decode_gbsa_prepared_metadata_replay'
```

Kernel correctness and performance must be measured on each hardware family.
Compare SAS and fused GBSA with matching model, KV dtype, speculative settings,
cache policy, and workload. Profiled kernel windows do not establish clean
end-to-end TPOT or throughput improvements.
