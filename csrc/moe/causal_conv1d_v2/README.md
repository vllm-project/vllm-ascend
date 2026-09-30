# CausalConv1dV2

## 支持范围

| 产品 | 支持 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列 | √ |
| Atlas A2 训练/推理系列 | √ |
| Atlas 200I/500 A2、Atlas 推理/训练系列 | × |

## 功能

算子使用同一个 `CausalConv1dV2` L0 路径完成两种模式：

- `runMode=0`：FN/prefill，输入为 dim-last `(T,D)` 或 `(B,S,D)`。
- `runMode=1`：UPDATE/decode，更新 `convStates` 并输出当前 token 的卷积结果。

计算为：

```text
y[t] = activation(sum(weight[j] * x[t-j]) + bias)
```

卷积宽度支持 2、3、4，`D` 必须是 16 的倍数，数据 dtype 支持 FP16 和 BF16。

## 输入与属性

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `x` | Tensor | dim-last 数据。 |
| `weight` | Tensor | `(W,D)`。 |
| `bias` | Optional Tensor | `(D,)`。 |
| `convStates` | Optional mutable Tensor | `(N,L,D)`；FN 可省略或传空 Tensor，UPDATE 必须提供。 |
| `queryStartLoc` | Optional INT32 Tensor | 变长序列边界。 |
| `cacheIndices` | Optional INT32 Tensor | 序列到状态行的映射。 |
| `hasInitialState` | Optional BOOL/INT32 Tensor | FN 是否读取已有状态。 |
| `numAcceptedTokens` | Optional INT32 Tensor | speculative UPDATE 的接受 token 数。 |
| `*Cpu` | Optional INT64 aclIntArray | Host-array metadata 输入；FN/UPDATE 保留该形式且不产生 warning。 |
| `activation` | string | `"none"`、`"silu"` 或 `"swish"`。 |
| `padSlotId` | int64 | FN padding 索引；raw 默认 `-1`。 |
| `nullBlockId` | int64 | 保留空状态块；负值禁用。 |
| `runMode` | int64 | `0` 为 FN，`1` 为 UPDATE。 |
| `headNum` | int64 | FN 输出分头；`0` 保持扁平输出。 |
| `maxQueryLen` | int64 | UPDATE 变长模式的最大分段长度上界；其他模式为 `-1`。 |

device Tensor metadata 与对应 `*Cpu` 输入互斥。device metadata 必须与数据 Tensor 位于同一设备；`queryStartLoc/cacheIndices/numAcceptedTokens` 仅支持 INT32，`hasInitialState` 支持 BOOL 或 INT32。

## Python API

推荐使用解耦的 ctypes 入口：

```python
from fla_npu.ops.ascendc import causal_conv1d_v2_fn, causal_conv1d_v2_update

y = causal_conv1d_v2_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=cache_indices,
    has_initial_state=has_initial_state,
    head_num=0,
)

causal_conv1d_v2_update(
    x,
    conv_states,
    weight,
    bias,
    conv_state_indices=conv_state_indices,
)

# 变长 UPDATE：max_query_len 必须为非负值，且不小于最长分段。
causal_conv1d_v2_update(
    x_varlen,
    conv_states,
    weight,
    bias,
    conv_state_indices=conv_state_indices,
    query_start_loc=query_start_loc,
    max_query_len=3,
)
```

FN 使用 `cache_indices`；UPDATE 使用 `conv_state_indices`。两个高层接口既接受与数据位于同一设备的 Tensor metadata，也保留对应的 `*_cpu` Host metadata 输入；同一 metadata 的 device 与 Host 形式互斥。使用 `*_cpu` Host 输入不会产生 warning。高层接口的 `activation` 接受 `None`、`"silu"` 或 `"swish"`，其中 `None` 表示不启用激活。两个接口均保持 dim-last，不执行 transpose、permute 或隐式 dtype 转换。UPDATE 默认原地写回 `x`；传入 `out` 时写入 `out`，并始终原地更新 `conv_states`。

FN 的 `head_num` 默认为 `0`，表示输出保持 flat dim-last；正值要求整除 `D` 且 head dim 为 16 的倍数。FN 的 3D `(B,S,D)` 输入不接受 `query_start_loc` 或 `query_start_loc_cpu`；序列边界由 `B`、`S` 直接确定。

UPDATE 传入 `query_start_loc` 或 `query_start_loc_cpu` 时，必须同时提供非负的 `max_query_len`。它只声明最大分段长度上界；每个序列的实际范围仍由所传序列边界 metadata 决定。`validate_data=True` 时会拒绝小于实际最长分段的值。

高层 FN 默认 `pad_slot_id=-1`、`null_block_id=0`；高层 UPDATE 不公开 pad sentinel，默认 `null_block_id=0`。`null_block_id=None` 可禁用 null block 过滤。保留在签名中的 APC/block scheduling 参数当前不支持，传入非默认值会抛出 `NotImplementedError`。

`causal_conv1d_v2` Python 接口作为废弃兼容入口保留，调用时会发出 `FutureWarning`，并将在 2027 年 2 月移除；新代码应使用 FN/UPDATE 接口。该入口保持原有签名以及 prefill 场景的 autograd 绑定行为。

 `npu_causal_conv1d_v2` Python 接口仅为兼容保留，调用时会发出 `FutureWarning`，并将在 2026 年 12 月废弃；新代码应使用 FN/UPDATE 接口。该接口保持原有参数不变，`query_start_loc`、`cache_indices`、`initial_state_mode` 和 `num_accepted_tokens` 仍按 Host int-array metadata 传递，不接受接口使用的 device metadata 参数。该入口继续发出既有的 `FutureWarning`。

## ACLNN

C 接口原型见 [aclnnCausalConv1dV2](./docs/aclnnCausalConv1dV2.md)，最小 C++ 调用见 [test_aclnn_causal_conv1d_v2.cpp](./examples/test_aclnn_causal_conv1d_v2.cpp)。
