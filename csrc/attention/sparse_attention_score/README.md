# SparseAttentionScore

Sparse attention operator that gathers the selected KV blocks from a blocked KV cache according to externally supplied Top-K block indices (`select_idx`) and a logical-to-physical block mapping (`block_table`), then performs the FlashAttention computation.

## API

```python
torch_npu.npu_sparse_attention_score(
    query,          # [T, N, D], fp16/bf16/fp8
    key,            # [blockNum, blockSize, KVHead, D]
    value,          # [blockNum, blockSize, KVHead, D]
    select_idx,     # [KVHead, maxQSeqlen, TopK], int32
    block_table,    # [batch, maxBlocksPerBatch], int32
    *,
    select_num_idx=None,        # [KVHead, maxQSeqlen], int32
    actual_seq_lengths=None,    # list[int]
    actual_seq_lengths_kv=None, # list[int]
    num_key_value_heads=1,
    scale_value=1.0,
    block_size=128,
    top_k=16,
    attention_out_dtype=torch.bfloat16,
) -> Tensor  # [T, N, D]
```

## Constraints

- Platform: Ascend 950
- `block_size = 128`
- GQA is supported (`num_heads` must be divisible by `num_key_value_heads`).
- FP8 input requires a dequantization scale.
