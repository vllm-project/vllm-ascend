# MsaIndexScore Test Guide

## 1. End-to-End Accuracy Self-Check (Primary Test)

`examples/test_aclnn_msa_index_score.cpp` contains both the aclnn invocation and a CPU golden implementation.

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

## 2. Test Matrix

The matrix follows the golden cases in the design document. `start_loc` is a **logical-block index**, and `sparse_mode=3` provides causal masking.

| Test Case | Scenario | Coverage |
| --------- | -------- | -------- |
| `L0-debug-trace` | Minimal dimensions | Main path/TRACE |
| `L0-int8-dequant-trace` | int8 + scale | Prefused dequantization |
| `L0-prefill-aligned` | Aligned chunked prefill | rightDownCausal + local_mask |
| `L1-prefill-unaligned` | Multi-batch variable length | Boundary-block mask |
| `L1-prefill-multi-mtile` | Row count > M-tile | M-tile partitioning |
| `L1-decode-lq1` | Decode with q_len=1 | Multiple sequence lengths |
| `L1-decode-speculative` | q_len>1 | Speculative decoding |
| `L1-long-seq-multi-stile` | kv=4096 | Multiple S-tiles |
| `L1-bf16` / `L1-int8-dequant` | dtype | Non-quantized/quantized |
| `L2-tiny-kv` | Minimal KV length | Tail padding |
| `L1-bnbd` / `L1-bnbd-int8` | PageAttention BNBD | `[NP, N2, P, D]` |
| `L1-tnd-unaligned` / `L1-tnd-int8` / `L0-tnd-tiny` | Packed TND | No block_table; klen prefix sum |

The complete test matrix, including TND and BNBD, runs by default. The key layout is specified by `layout_key` (`layoutKeyOptional` in aclnn) and is no longer inferred from the shape.

## 3. Python Reference

`tests/golden/msa_index_score_golden.py`:

```python
golden = msa_index_score_golden(
    query, key, block_table, actual_seq_qlen, actual_seq_klen, start_loc,
    sparse_mode=3, scale=None)
```

## 4. Acceptance Criteria

- Padded positions (invisible blocks) are `-inf` on both sides.
- Blocks forced to high scores by `local_mask` are `≥1e28` on both sides.
- Valid positions use `atol/rtol=1e-3` and `error_ratio≤1e-3`.
