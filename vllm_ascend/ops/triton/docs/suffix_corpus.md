# Suffix corpus lookup and ingestion

## Description

- **Function**: Index completed responses and retrieve a continuation using `_corpus_draft`.
- **Formula**: Index entries pack `(document_sequence << 32) | corpus_position`. A continuation's score is its sum of prefix-support counts divided by the initial candidate count.
- **Algorithm flow**: `launch_suffix_ingest` writes documents to the ring and indexes 2-token and 4-token keys. Lookup verifies hits, deduplicates positions and votes on prefix-consistent continuations.
- **Supported modes**: Ascend A3 and A5; eager and graph execution.

## Parameters

`R` is request capacity, `L` history capacity, `F` finished-document count and `C` corpus capacity.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `token_ids_ptr`, `token_ids_stride`, `total_len_ptr` | Input / Attribute | Histories, actual row stride and valid lengths | int32, integer | `[R, L]`, scalar, `[R]` |
| `docs_ptr`, `num_docs` | Input / Attribute | Finished request slots and history starts, sorted by request ID | int32, integer | `[F, 2]`, scalar |
| `corpus_ptr`, `corpus_len`, `table_ptr` | Input / Output / Attribute | Token ring, capacity and hash indexes | int32, integer, int64 | `[C]`, scalar, `[2, NUM_BUCKETS, SLOTS]` |
| `head_ptr`, `doc_seq_ptr`, `head_out_ptr`, `doc_seq_out_ptr` | Input / Output | Current and next ring cursor / document sequence | int64 | `[1]` each |
| `HASH_MUL`, `NUM_BUCKETS`, `SLOTS`, `BLOCK`, `launch_options` | Attribute | Upstream hash and tile configuration | Integers / mapping | Host values |
| `row_ptr`, `seq_len`, `MAX_DEPTH`, `DEPTH_PO2`, `K`, `K_PO2` | Input / Attribute | Lookup history, valid length and padded match/draft limits | int32 pointer, integers | History row, scalars |

## Constraints

All buffers reside on one device. Histories have contiguous columns; other buffers use the contiguous layouts above.
Lookup needs depth at least two, power-of-two bucket count and eight slots per bucket.
Token IDs are nonnegative; `-1` separates documents. Documents wrap without straddling the ring and oversized responses retain their newest `C - 1` tokens.
Lookup stops at document boundaries and retains the newest valid index entry for a repeated corpus position.
The caller copies ingestion cursor outputs back and owns request lifecycle.

## Origin and Differences

- **Origin**: [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339).
- **Differences**: Count later-document overlaps before stores for deterministic ingestion. Find the first continuation separator with a minimum reduction to avoid the A5 narrow-row cumsum issue.

## Test Cases

Corpus/index state and draft tokens require exact equality against CPU references and deterministic expected continuations. Cases include wraparound, overwrite, voting and document boundaries.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_suffix_corpus.py
```

## Example

N/A for standalone use; ingestion is dispatched from the speculator and finalize calls the lookup helper.
