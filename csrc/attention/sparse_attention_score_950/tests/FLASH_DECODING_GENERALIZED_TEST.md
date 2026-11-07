# SparseAttentionScore Arch35 FlashDecoding 泛化测试说明

## 1. 测试目标

本测试用于验证 Host 自动选择 FlashDecoding 的策略和当前 Arch35 FlashDecoding 实现，覆盖：

- 满足 Host FD 条件时自动进入 FD，不满足时自动回退普通路径；
- BF16 tiling key 10002/10006；
- FP16 tiling key 10001/10005；
- FD shard 数量和 base task 数量的边界；
- MQA、GQA、MHA；
- `select_num_idx` 运行时有效 block 数裁剪；
- 非顺序 `block_table` 映射和最后一个不满 128 token 的 KV block；
- 随机、相同 logits、跨 shard 大小 logits、常量 value 等数值模式；
- 不满足 FD 门槛时的普通路径回退。

测试分为两层：

1. Python/NPU 功能和精度测试：执行 Host 自动选择的 kernel。
2. Host tiling C++ 单元测试：直接断言自动选择的 tiling key。

## 2. 测试文件

### 2.1 Python 真机测试

文件：`test_flash_decoding_generalized.py`

每个精度用例只生成一份固定输入，并执行两份结果：

```text
CPU FP32 mathematical golden
              ↑
NPU Host-auto (FD or normal fallback)
```

检查项：

- NPU 和 CPU FP32：相对 L1 误差不超过 `2e-2`；
- NPU 和 CPU FP32：余弦相似度不低于 `0.999`；
- 输出 shape、dtype、NaN/Inf；
- 所有随机输入都使用固定 seed，失败可复现。

### 2.2 Host tiling 测试

文件：`ut/op_host/test_sparse_attention_score_fd_tiling.cpp`

该测试使用 28-AIC Ascend950 fake platform，直接断言：

| 用例 | 预期结果 |
|---|---:|
| BF16、条件满足 | 10006 |
| FP16、条件满足 | 10005 |
| `topK=1`，不能增加 shard | 10002 |
| `topK=17`，超过 FD 上限 | 10002 |
| baseTasks=8、AIC=28、topK=16 | 10006 |
| baseTasks=24、AIC=28、topK=2 | 10002 |
| baseTasks=28、AIC=28 | 10002 |

## 3. 泛化用例矩阵

### 3.1 FD 有效用例

Python 测试包含 15 个 FD 有效 shape：

| 维度 | 覆盖值 |
|---|---|
| dtype | BF16、FP16 |
| topK | 2、4、8、16 |
| baseTasks | 1、2、4、8、24 |
| groupSize | 1、4、8、16、128 |
| Attention 类型 | MQA、GQA、MHA |
| Q token | 1、2、4、12 |
| valid block 数 | 1、2、topK-1、topK |
| KV 尾块 | 完整 128、部分有效 token |
| blockTable | logical ID 与 physical ID 逆序映射 |
| 数值分布 | random、equal logits、constant value、shard extremes |

重要 shard 边界：

```text
baseTasks=1,  topK=2  -> 2 shards
baseTasks=1,  topK=16 -> 16 shards
baseTasks=2,  topK=8  -> 16 shards
baseTasks=8,  topK=8  -> 22 compute cores (3 flattened tasks/core)
baseTasks=24, topK=2  -> 24 compute cores (2 flattened tasks/core)
```

### 3.2 FD 回退用例

以下用例验证 Host 自动回退：

- `topK=1`：最终 shard 数等于 base task 数，没有并行收益；
- `topK=17`：超过当前 FD 的 `topK<=16` 门槛；
- baseTasks=28：不满足 `baseTasks<aicNum`，同时超过 base-task 元数据容量；
- FP16 `topK=1`：同时覆盖 FP16 回退 key。

`topK=17` 超出当前 FD 支持范围。Host UT 断言其自动回退到 key 10002；不把普通 kernel 在该范围的结果纳入 CPU 精度承诺。

### 3.3 自动策略行为

- 调用方不提供 FD 开关，Host 根据 SoC、dtype、shape、任务量和 cost model 自动选择路径；
- 元测试会检查 case 表本身没有丢失 dtype、topK、baseTasks、valid count 和数值模式边界。

## 4. 运行方法

### 4.1 Python/NPU 全量测试

```bash
source /usr/local/Ascend/cann/set_env.sh
source /home/npu_user1/l00937279/package/package_custom_fork_msa_msa_demo_v2_6de16cedf9/vendors/custom_transformer/bin/set_env.bash
ASCEND_RT_VISIBLE_DEVICES=1 python -m pytest \
  attention/sparse_attention_score/tests/test_flash_decoding_generalized.py \
  -v -s
```

只运行 FD 有效 shape：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python -m pytest \
  attention/sparse_attention_score/tests/test_flash_decoding_generalized.py::test_fd_generalized_accuracy \
  -v -s
```

### 4.2 Host tiling UT

编译：

```bash
bash build.sh --ophost_test --ops=sparse_attention_score --noexec -j16
```

运行：

```bash
BUILD_PATH="$(pwd)/build" \
  build/tests/ut/framework_normal/op_host/transformer_op_host_ut \
  --gtest_filter='SparseAttentionScoreFdTilingTest.*'
```

## 5. 2026-07-26 真机结果

设备：Ascend950PR，NPU 1。

Python/NPU：

```text
23 passed, 1 xfailed in 4.86s
```

关键精度范围：

| 项目 | 结果 |
|---|---:|
| BF16 FD 有效用例最大 CPU max diff | 0.00018646 |
| BF16 FD 有效用例最大 relative L1 | 0.00456663 |
| BF16 FD 有效用例最小 cosine | 0.99999386 |
| FP16 FD 有效用例最大 CPU max diff | 0.00001946 |
| FP16 FD 有效用例最大 relative L1 | 0.00027511 |
| FP16 FD 有效用例最小 cosine | 0.99999964 |

Host tiling UT：

```text
9 tests from SparseAttentionScoreFdTilingTest
9 passed
```

### 5.1 2026-07-28 28-AIC 更新验证

Host 与 kernel 的 `SASA_FD_MAX_AIC` 同步更新为 28 后，使用
BF16、batch=8、qSeqlen=1、kvSeqlen=2048、qHeads=16、kvHeads=1、
headSize=128、blockSize=128、topK=16 的固定长度 shape 在 Ascend950PR
真机通过。Profiler 结果：

```text
Op Name: SparseAttentionScore_*_10006_mix_aic
Block Dim: 28
Mix Block Dim: 56
All task success
```

数值结果：

```text
passed: true
max_diff_pipeline: 0.000244140625
relative_l1_math: 0.0024210647674262624
cosine_similarity_math: 0.999998927116394
```

当前 Host tiling UT 为 15/15 通过。

## 6. 测试发现和已知限制

### 6.1 `select_num_idx=0`

当某个 base task 的 `select_num_idx=0` 时，当前普通/FD kernel 没有完整初始化对应输出或 partial result，因此输出不满足数学语义上的全零。测试将该场景保留为：

```text
test_zero_valid_blocks_should_produce_zero_output
```

并使用非严格 `xfail` 标记。真实 causal block 生成逻辑至少会包含当前 block，正常有效范围从 1 开始；主精度矩阵覆盖 `1、2、topK-1、topK`。

### 6.2 `topK=17`

首次泛化运行发现，`topK=17` 下 Host UT 确认回退到 10002，但普通 kernel 相对 CPU FP32 的 relative L1 为 0.22660052。该 shape 超出当前 FD 的 `topK<=16` 范围，因此仅作为策略回退测试，不作为精度支持用例。

## 7. 后续扩展方式

新增 case 时只需要向 `FD_ELIGIBLE_CASES` 或 `FD_FALLBACK_CASES` 添加一个 `FDCase`。case 名、dtype、shape、valid-count 模式、数值模式和 seed 都会进入 pytest ID；生成器会自动完成 CPU golden 和 NPU Host-auto 两方计算。

如果修改 Host FD 门槛，必须同时更新：

1. Python case 表和 `_expected_fd_compute_cores`；
2. Host tiling key 单元测试；
3. 本文档的覆盖矩阵；
4. 重新执行真机全量测试。
