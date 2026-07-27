# RecurrentKda API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Kimi K3 单算子入口 | `torch.ops._C_ascend.recurrent_kda` | 支持（不接整网） |
| aclnn | `aclnnRecurrentKdaGetWorkspaceSize` / `aclnnRecurrentKda` | 支持 |
| Ascend C `<<<>>>` | `recurrent_kda<<<blockDim, nullptr, stream>>>` | 支持 |

各入口表达同一个 fused recurrent KDA 前向语义。算子计算本身在一个 AICore kernel 内完成，不把
`KdaGateCumsum + recurrent` 暴露为两段式公共实现。

## 2. 公共参数与约束

Shape 符号统一引用 [KDA 模型符号表](../../README.md#model-shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `BSND=[B,T,H_k,K]` 或 `TND=[T,H_k,K]` | BF16 | BSND/TND | query |
| `k` | 必选 | 与 `q` 相同 | BF16 | BSND/TND | key |
| `v` | 必选 | `BSND=[B,T,H_v,V]` 或 `TND=[T,H_v,V]` | BF16 | BSND/TND | value |
| `g` | 必选 | `BSND=[B,T,H_v,K]` 或 `TND=[T,H_v,K]` | FP32/BF16/FP16 | BSND/TND | 预计算 step log gate 或 raw gate |
| `beta` | 必选 | `BSND=[B,T,H_v]` 或 `TND=[T,H_v]` | FP32/BF16/FP16 | BSND/TND | delta 更新系数 |
| `initial_state` | 必选 | `[state_capacity,H_v,V,K]` | FP32/BF16 | ND | 可变 state pool；命中槽原位更新 |
| `cu_seqlens` | 必选 | `[seq_num+1]` | INT32/INT64 | ND | fla-org 累积 offset；首项为 0，末项为有效 packed token 数且不超过输入 token capacity，相邻差值为各序列长度 |
| `ssm_state_indices` | 必选 | packed `[>=T]` 或 speculative `[seq_num,max_step]` | INT32/INT64 | ND | 每个 token 对应的 state pool 槽索引 |
| `A_log` | 条件必选 | `[H_v]` | FP32 | ND | `use_gate_in_kernel=True` 时必选 |
| `dt_bias` | 可选 | `[H_v*K]` 或 `[H_v,K]` | FP32 | ND | raw gate 偏置 |
| `num_accepted_tokens` | 可选 | `[seq_num]` | INT32/INT64 | ND | 必须与 `ssm_state_indices` 一起传 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | 与 `v` 相同 | BF16 | recurrent 输出；当有效 token 数小于 capacity 时，padding tail 不作保证 |

`initial_state` 是可变输入：命中槽位原地更新，未命中槽位保持不变，不作为独立 alias 输出返回。

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `BSND` | `{"BSND", "TND"}` | 输入布局 |
| `scale` | float? | Python/legacy 为 `K ** -0.5` | 任意有限浮点 | 乘到 query 上 |
| `output_final_state` | bool | `false` | `{false, true}` | 兼容属性；state 始终通过 `stateRef` 原地更新，不产生独立输出 |
| `use_qk_l2norm_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内对 q/k 做 L2 normalize |
| `use_gate_in_kernel` | bool | `false` | `{false, true}` | 是否把 `g` 解释为 raw gate |
| `use_beta_sigmoid_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内计算 `sigmoid(beta)` |
| `allow_neg_eigval` | bool | `false` | `{false, true}` | beta sigmoid 后是否乘 2 |
| `safe_gate` | bool | `false` | `{false, true}` | raw gate 的 safe 分支 |
| `lower_bound` | float? | `-5.0` | `[-5,0)` when `safe_gate=True` | safe gate 下界 |
| `state_v_first` | bool | `true` | 当前必须为 `true` | 状态布局为 `[state_capacity,H_v,V,K]` |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *stateRef,
    const aclTensor *cuSeqlens,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成参数校验、连续化/cast 预处理和 executor 创建；第二段在传入 stream 上异步执行。
`stateRef` 同时是输入和输出。非连续 state 使用临时连续 tensor 执行，再通过 `ViewCopy` 回写原 view。
`cuSeqlens` 仅在 host 检查 rank/dtype，具体 offset 值由 device kernel 读取，兼容 ACLGraph replay。
首项必须为 0，offset 必须单调不减，末项表示有效 token 数且不得超过输入 token capacity，相邻差值为各序列长度。
末项小于 capacity 时，kernel 仅处理有效前缀并逐行跳过零长度序列；padding tail 输出不作保证。
输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

### 3.2 调用示例

```cpp
// 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
ACLNN_CHECK(aclnnRecurrentKdaGetWorkspaceSize(
    q, k, v, g, beta, state, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, "BSND", scale, true, true, true,
    true, false, false, -5.0, true, out,
    &workspaceSize, &executor));
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnRecurrentKda(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
```

## 4. `torch.ops._C_ascend` API

### 4.1 接口签名

```python
torch.ops._C_ascend.recurrent_kda(
    query, key, value, gate, beta, initial_state,
    cu_seqlens, ssm_state_indices, A_log, dt_bias,
    *, num_accepted_tokens=None, scale=128**-0.5,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    safe_gate=True, lower_bound=-5.0)
```

输入 rank 为 3 时自动选择 TND，rank 为 4 时自动选择 BSND。Torch API 只返回 `out`；
`initial_state` 是显式 mutable input，命中槽位原地更新，未命中的容量槽保持不变。Torch schema
不返回 aliased state output，因此可以被 AOT functionalization 和 NPU 图编译安全处理。

### 4.2 调用示例

```python
import torch
T, H, K = 5, 6, 128
q = torch.randn(T, H, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn(T, H, K, device="npu", dtype=torch.float32)
beta = torch.randn(T, H, device="npu", dtype=torch.float32)
state = torch.zeros(13, H, K, K, device="npu", dtype=torch.float32)
cu_seqlens = torch.tensor([0, 1, 4, 4, 5], device="npu", dtype=torch.int32)
slots = torch.tensor(
    [[8, 8, 8, 8, 8], [2, 11, 6, 2, 2], [4, 4, 4, 4, 4], [1, 3, 5, 7, 9]],
    device="npu")
A_log = torch.randn(H, device="npu", dtype=torch.float32)
dt_bias = torch.randn(H, K, device="npu", dtype=torch.float32)

out = torch.ops._C_ascend.recurrent_kda(
    q, k, v, g, beta, state, cu_seqlens, slots, A_log, dt_bias)
torch.npu.synchronize()
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果。参数顺序与 kernel 定义保持一致：

```cpp
recurrent_kda<<<blockDim, nullptr, stream>>>(
    q, k, v, g, beta, initialState, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, out, finalState, workspace, tiling);
```

直调通路只作为 route/诊断入口；公开 Python 和 aclnn API 负责完整参数校验。直调通路按连续物理
布局解释 GM 地址，非连续 state 需要先由调用侧连续化。

## 6. 已知限制

- `q/k/v/out` 当前仅支持 BF16。
- `K/V` 当前仅支持 `K=128,V=128` 或 `K=128,V=256` 两档枚举。
- `_C_ascend` 入口支持非连续 `initial_state`，并保持原 tensor 的原位更新与 alias 语义。
- 所有活跃 slot 必须位于 `[0,state_capacity)`，且不同活跃序列不得共享正在写入的槽。
- 空序列不读取 `ssm_state_indices/num_accepted_tokens`，也不读写 state pool。
- `cu_seqlens` 必传，首项必须为 0，offset 必须单调不减，末项为有效 token 数且不得超过输入
  token capacity；相邻差值为序列长度。值约束由 device kernel 检查。
- Ascend C `<<<>>>` 直调入口要求 state 为连续物理布局。
- 每条 recurrent 有效序列长度必须 `<=8`。
- 仅支持 `layout="BSND"` 和 `layout="TND"`。
- 仅支持 `state_v_first=True`。
- `use_gate_in_kernel=false` 时 `A_log/dt_bias/safe_gate` 必须为空或 false。

## 7. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 必选 tensor、workspaceSize 或 executor 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| rank/shape/dtype/layout 或属性组合非法 | `ACLNN_ERR_PARAM_INVALID` |
| 内部 tensor 创建或 L0 调用失败 | `ACLNN_ERR_INNER_NULLPTR` |
| torch 输入不在同一 NPU device，或 shape/dtype 非法 | `RuntimeError` |
