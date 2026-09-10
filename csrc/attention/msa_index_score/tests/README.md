# MsaIndexScore Test Guide

## 1. End-to-End Accuracy Self-Check

`examples/test_aclnn_msa_index_score.cpp` contains the aclnn invocation and a
CPU golden implementation.

For Atlas A2/A3:

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust \
  --vendor_name=custom --soc=ascend910b
```

For Ascend 950, pass `--soc=ascend950` explicitly and source the installed
environment before running the example:

```bash
bash build.sh --pkg --soc=ascend950 --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path=/tmp/msa_opp
source /tmp/msa_opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust \
  --vendor_name=custom --soc=ascend950
```

The expected summary is 40/40 passing cases on Ascend 950. A2/A3 skip four
FP8 cases and run 36 cases.

## 2. Test Matrix

`start_loc` is a logical-block index, and `sparse_mode=3` applies
right-down-causal masking.

| Test Case | Scenario | Coverage |
| --------- | -------- | -------- |
| `L0-debug-trace` | Minimal dimensions | Main path and trace |
| `L0-int8-dequant-trace` | INT8 key with scale | Fused dequantization |
| `L0-prefill-aligned` | Aligned chunked prefill | Causal and local masks |
| `L1-prefill-unaligned` | Variable-length batch | Boundary-block mask |
| `L1-prefill-multi-mtile` | Row count greater than M-tile | M-tile partitioning |
| `L1-decode-lq1` | Decode with `q_len=1` | Multiple sequence lengths |
| `L1-decode-speculative` | Decode with `q_len>1` | Speculative decoding |
| `L1-long-seq-multi-stile` | `kv_len=4096` | Multiple S-tiles |
| `L1-bf16` / `L1-int8-dequant` | Data type | Non-quantized and quantized paths |
| `L2-tiny-kv` | Minimal KV length | Tail padding |
| `L1-bnbd` / `L1-bnbd-int8` | PageAttention BNBD | `[NP, N2, P, D]` layout |
| `L1-tnd-unaligned` / `L1-tnd-int8` / `L0-tnd-tiny` | Packed TND | No block table and key-length prefix sums |
| `L0-fp8-e4m3fn` / `L0-fp8-e5m2` / `L1-fp8-e4m3fn-prefill` | Ascend 950 FP8 | Native E4M3FN/E5M2 Cube paths; HIFLOAT8 is kernel-only |
| `L1-pad-q0` / `L1-pad-kv0` | Empty request in a mixed batch | Skip empty query or fill empty KV scores |
| `L1-pad-q0-kv0` / `L1-tnd-pad-q0-kv0` / `L1-pad-mid-q0` | Empty request at an edge or in the middle | PageAttention and TND padding |
| `L0-all-q0` / `L1-all-q0` | Entire batch has `q_len=0` | Host acceptance and skipped computation |
| `L0-all-kv0` / `L0-all-q0-kv0` | Entire batch has empty KV | Fill scores and fully empty input |
| `L0-tnd-all-q0` / `L0-tnd-all-kv0` / `L0-tnd-all-q0-kv0` | Empty packed TND batch | Empty query and key tensors |
| `L0-stride-bbnd` / `L1-stride-bbnd` / `L1-stride-bnbd` | Page axis has a gap of two | Non-contiguous physical-page addressing |
| `L1-stride-int8` | INT8 page axis has a gap of two | Quantized page copy with a stride |
| `L0-wide-table-257` / `L1-wide-table-257-bf16` | Block-table width 257 | Ascend 950 C2UB windowed flush |
| `L0-fp8-wide-table-257` | Width 257 with FP8 | Aligned score width and fill positions |

The full matrix runs by default. The key layout is selected by `layout_key`
(`layoutKeyOptional` in aclnn) and is not inferred from tensor rank.

## 3. Python Reference and Unit Tests

The CPU reference is in `tests/golden/msa_index_score_golden.py`:

```python
inputs = MsaIndexScoreGoldenInputs(
    query=query,
    key=key,
    block_table=block_table,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_klen=actual_seq_klen,
    start_loc=start_loc,
    sparse_mode=3,
    scale=None,
)
score = msa_index_score_golden(inputs)
```

The repository unit test
`tests/ut/ops/test_msa_index_score_golden.py` executes this imported reference
for non-quantized PageAttention, INT8 dequantization, and a non-contiguous
physical-page axis.

## 4. Acceptance Criteria

- Fill positions use the negative fill value on both sides.
- Blocks forced by `local_mask` are at least `1e28` on both sides.
- FLOAT16, BFLOAT16, and INT8 use `atol=rtol=1e-3` and an error ratio no
  greater than `1e-3`.
- Ascend 950 FP8 uses `atol=rtol=2e-2`.
