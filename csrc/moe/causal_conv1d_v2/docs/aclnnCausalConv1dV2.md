# aclnnCausalConv1dV2

## 函数原型

```cpp
aclnnStatus aclnnCausalConv1dV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    const aclTensor *convStatesOptional,
    const aclTensor *queryStartLocOptional,
    const aclTensor *cacheIndicesOptional,
    const aclTensor *hasInitialStateOptional,
    const aclTensor *numAcceptedTokensOptional,
    const aclIntArray *queryStartLocCpuOptional,
    const aclIntArray *cacheIndicesCpuOptional,
    const aclIntArray *hasInitialStateCpuOptional,
    const aclIntArray *numAcceptedTokensCpuOptional,
    char *activationOptional,
    int64_t padSlotId,
    int64_t nullBlockId,
    int64_t runMode,
    int64_t headNum,
    int64_t maxQueryLen,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnCausalConv1dV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数

| 参数 | 约束 |
| --- | --- |
| `x` | FP16/BF16，dim-last。FN 支持 `(T,D)`、`(B,S,D)`；UPDATE 支持 `(B,D)`、`(B,S,D)` 及带 `queryStartLoc` 的 `(T,D)`。 |
| `weight` | `(W,D)`，`W` 为 2、3、4，`D % 16 == 0`。 |
| `biasOptional` | 可选 `(D,)`，dtype/device 与 `x` 一致。 |
| `convStatesOptional` | `(N,L,D)`，`L >= W-1`。FN 可为空，UPDATE 必须为非空；执行时可能原地更新。 |
| `queryStartLocOptional` | 可选 INT32 device Tensor，长度为 `batch+1`。 |
| `cacheIndicesOptional` | 可选 INT32 device Tensor，长度为 `batch`。 |
| `hasInitialStateOptional` | 可选 BOOL/INT32 device Tensor，仅 FN 使用。 |
| `numAcceptedTokensOptional` | 可选 INT32 device Tensor，UPDATE speculative 路径使用。 |
| 四个 `*CpuOptional` | 对应 metadata 的 INT64 `aclIntArray` 兼容输入，与 device Tensor 输入互斥。 |
| `activationOptional` | `"none"`、`"silu"` 或 `"swish"`。 |
| `padSlotId` | 跳过 FN padding 索引。 |
| `nullBlockId` | 非负时跳过保留 null 状态块，负值禁用。 |
| `runMode` | `0` 为 FN，`1` 为 UPDATE。 |
| `headNum` | `0` 返回与 `x` 相同 shape；正值仅用于 FN，要求整除 `D` 且 head dim 为 16 的倍数。 |
| `maxQueryLen` | UPDATE 变长模式下各分段长度的上界，取值 `>= 0`；其他模式传 `-1`。该属性追加在 `headNum` 之后。 |
| `out` | 调用方分配。flat 输出与 `x` 同 shape；分头输出为 `(N,T,D/N)` 或 `(B,N,S,D/N)`。 |

## 行为

- FN 的 `cacheIndices` 同时受 `padSlotId` 和启用的 `nullBlockId` 过滤。
- Python 高层 FN 的 3D `(B,S,D)` 输入不接受 `query_start_loc`；序列边界由固定的 `B`、`S` 确定。
- UPDATE 通过 `cacheIndices` 选择状态行；Python 高层接口对外命名为 `conv_state_indices`。
- UPDATE 传入 `queryStartLoc` 时，`maxQueryLen` 对齐 vLLM 的 `max_query_len`，用于声明最大分段长度，并为 speculative 状态容量校验提供上界；实际分段窗口仍由 `queryStartLoc` 决定。
- 未提供 `hasInitialState` 时，FN 使用零历史并按有效 cache index 写回最终状态。
- device metadata 必须使用与数据 Tensor 相同的设备。
- Host-array 输入仅用于兼容，计划在 2026 年 12 月后删除。
- 算子默认使用确定性实现。

## 调用

先调用 `aclnnCausalConv1dV2GetWorkspaceSize`，按返回大小分配 workspace，再在同一 stream 上调用 `aclnnCausalConv1dV2`。完整最小示例见 [test_aclnn_causal_conv1d_v2.cpp](../examples/test_aclnn_causal_conv1d_v2.cpp)。
