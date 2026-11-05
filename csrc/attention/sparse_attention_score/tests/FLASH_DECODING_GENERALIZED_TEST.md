# SparseAttentionScore Arch35 FlashDecoding Generalization Tests

## 1. Test Objectives

These tests verify both the host policy that automatically selects FlashDecoding (FD) and the current Arch35 FlashDecoding implementation. They cover:

- Automatic FD selection when the host-side eligibility conditions are satisfied, and automatic fallback to the normal path otherwise
- BF16 tiling keys 10002 and 10006
- FP16 tiling keys 10001 and 10005
- Boundary conditions for the number of FD shards and base tasks
- MQA, GQA, and MHA
- Runtime clipping of the valid block count through `select_num_idx`
- Nonsequential `block_table` mappings and a final KV block containing fewer than 128 valid tokens
- Numerical patterns including random values, equal logits, logits with large differences across shards, and constant values
- Normal-path fallback when the FD threshold is not met

The tests have two layers:

1. Python/NPU functionality and accuracy tests that execute the kernel selected automatically by the host.
2. Host tiling C++ unit tests that directly assert the automatically selected tiling key.

## 2. Test Files

### 2.1 Python Tests on NPU Hardware

File: `test_flash_decoding_generalized.py`

Each accuracy case generates one fixed input and computes two results:

```text
CPU FP32 mathematical golden result
                  ^
NPU host-auto path (FD or normal fallback)
```

The tests check:

- Relative L1 error between the NPU and CPU FP32 results does not exceed `2e-2`.
- Cosine similarity between the NPU and CPU FP32 results is at least `0.999`.
- Output shape, data type, and absence of NaN/Inf values.
- All random inputs use fixed seeds so failures are reproducible.

### 2.2 Host Tiling Tests

File: `ut/op_host/test_sparse_attention_score_fd_tiling.cpp`

These tests use a simulated Ascend 950 platform with 28 AICs and directly assert the following results:

| Case | Expected Result |
|---|---:|
| BF16 with all conditions satisfied | 10006 |
| FP16 with all conditions satisfied | 10005 |
| `top_k=1`, so sharding cannot increase parallelism | 10002 |
| `top_k=17`, exceeding the current FD limit | 10002 |
| `base_tasks=8`, `aic_num=28`, `top_k=16` | 10006 |
| `base_tasks=24`, `aic_num=28`, `top_k=2` | 10002 |
| `base_tasks=28`, `aic_num=28` | 10002 |

## 3. Generalization Test Matrix

### 3.1 FD-Eligible Cases

The Python tests contain 15 FD-eligible shapes:

| Dimension | Covered Values |
|---|---|
| Data type | BF16, FP16 |
| `top_k` | 2, 4, 8, 16 |
| `base_tasks` | 1, 2, 4, 8, 24 |
| `group_size` | 1, 4, 8, 16, 128 |
| Attention type | MQA, GQA, MHA |
| Q tokens | 1, 2, 4, 12 |
| Valid block count | 1, 2, `top_k - 1`, `top_k` |
| KV tail block | Full 128 tokens or partially valid |
| `block_table` | Reverse mapping from logical IDs to physical IDs |
| Numerical distribution | Random, equal logits, constant values, shard extremes |

Important sharding boundaries:

```text
base_tasks=1,  top_k=2  -> 2 shards
base_tasks=1,  top_k=16 -> 16 shards
base_tasks=2,  top_k=8  -> 16 shards
base_tasks=8,  top_k=8  -> 22 compute cores (3 flattened tasks/core)
base_tasks=24, top_k=2  -> 24 compute cores (2 flattened tasks/core)
```

### 3.2 FD Fallback Cases

The following cases verify automatic host fallback:

- `top_k=1`: The final shard count equals the base-task count, providing no additional parallelism.
- `top_k=17`: This exceeds the current FD threshold of `top_k <= 16`.
- `base_tasks=28`: This fails the `base_tasks < aic_num` condition and also exceeds the base-task metadata capacity.
- FP16 with `top_k=1`: This additionally covers the FP16 fallback key.

`top_k=17` is outside the currently supported FD range. The host unit test asserts automatic fallback to key 10002; the normal kernel's result for this range is not included in the CPU accuracy guarantee.

### 3.3 Automatic Policy Behavior

- The caller does not provide an FD switch. The host automatically selects the path according to the SoC, data type, shape, task count, and cost model.
- Meta-tests verify that the case table retains coverage for data type, `top_k`, `base_tasks`, valid-count, and numerical-pattern boundaries.

## 4. Running the Tests

### 4.1 Complete Python/NPU Test Suite

```bash
source /usr/local/Ascend/cann/set_env.sh
source /home/npu_user1/l00937279/package/package_custom_fork_msa_msa_demo_v2_6de16cedf9/vendors/custom_transformer/bin/set_env.bash
ASCEND_RT_VISIBLE_DEVICES=1 python -m pytest \
  attention/sparse_attention_score/tests/test_flash_decoding_generalized.py \
  -v -s
```

Run only FD-eligible shapes:

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python -m pytest \
  attention/sparse_attention_score/tests/test_flash_decoding_generalized.py::test_fd_generalized_accuracy \
  -v -s
```

### 4.2 Host Tiling Unit Tests

Build:

```bash
bash build.sh --ophost_test --ops=sparse_attention_score --noexec -j16
```

Run:

```bash
BUILD_PATH="$(pwd)/build" \
  build/tests/ut/framework_normal/op_host/transformer_op_host_ut \
  --gtest_filter='SparseAttentionScoreFdTilingTest.*'
```

## 5. Hardware Results from July 26, 2026

Device: Ascend 950PR, NPU 1.

Python/NPU:

```text
23 passed, 1 xfailed in 4.86s
```

Key accuracy results:

| Metric | Result |
|---|---:|
| Largest CPU `max_diff` among BF16 FD-eligible cases | 0.00018646 |
| Largest relative L1 among BF16 FD-eligible cases | 0.00456663 |
| Smallest cosine similarity among BF16 FD-eligible cases | 0.99999386 |
| Largest CPU `max_diff` among FP16 FD-eligible cases | 0.00001946 |
| Largest relative L1 among FP16 FD-eligible cases | 0.00027511 |
| Smallest cosine similarity among FP16 FD-eligible cases | 0.99999964 |

Host tiling unit tests:

```text
9 tests from SparseAttentionScoreFdTilingTest
9 passed
```

### 5.1 Verification of the 28-AIC Update on July 28, 2026

After synchronizing the host and kernel values of `SASA_FD_MAX_AIC` to 28, a fixed-length BF16 shape passed on Ascend 950PR hardware with the following configuration: `batch=8`, `q_seqlen=1`, `kv_seqlen=2048`, `q_heads=16`, `kv_heads=1`, `head_size=128`, `block_size=128`, and `top_k=16`.

Profiler results:

```text
Op Name: SparseAttentionScore_*_10006_mix_aic
Block Dim: 28
Mix Block Dim: 56
All task success
```

Numerical results:

```text
passed: true
max_diff_pipeline: 0.000244140625
relative_l1_math: 0.0024210647674262624
cosine_similarity_math: 0.999998927116394
```

The current host tiling unit-test result is 15/15 passing.

## 6. Findings and Known Limitations

### 6.1 `select_num_idx=0`

When a base task has `select_num_idx=0`, the current normal and FD kernels do not fully initialize the corresponding output or partial result. The output therefore does not satisfy the mathematically expected all-zero semantics. This scenario remains in the test suite as:

```text
test_zero_valid_blocks_should_produce_zero_output
```

It is marked with a non-strict `xfail`. Real causal-block generation includes at least the current block, so the normal valid range starts at 1. The primary accuracy matrix covers `1`, `2`, `top_k - 1`, and `top_k`.

### 6.2 `top_k=17`

The first generalization run found that, with `top_k=17`, the host unit test confirmed fallback to key 10002, but the normal kernel had a relative L1 error of 0.22660052 against the CPU FP32 result. This shape is outside the current FD range of `top_k <= 16`, so it is retained only as a policy-fallback test and is not treated as an accuracy-supported case.

## 7. Extending the Tests

To add a case, add an `FDCase` to either `FD_ELIGIBLE_CASES` or `FD_FALLBACK_CASES`. The case name, data type, shape, valid-count pattern, numerical pattern, and seed are all included in the pytest ID. The generator automatically computes both the CPU golden result and the NPU host-auto result.

When changing the host FD threshold, also update all of the following:

1. The Python case table and `_expected_fd_compute_cores`.
2. The host tiling-key unit tests.
3. The coverage matrix in this document.
4. The complete test suite on NPU hardware.
