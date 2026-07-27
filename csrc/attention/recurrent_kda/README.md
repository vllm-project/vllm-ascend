# RecurrentKda

> Kimi K3 integration note: this adapter accepts device-tensor `cu_seqlens` and a
> capacity-sized KDA state pool addressed by packed or 2D speculative `ssm_state_indices`.
> These differences are required for Kimi K3 ACLGraph and speculative decode.

`RecurrentKda` 是 KDA 的 fused recurrent 前向算子。算子在一个 AICore kernel 内完成 recurrent state decay、delta 更新和输出计算；`raw gate -> log gate` 和 `beta sigmoid` 可以在 kernel 内完成，不依赖 `KdaGateCumsum` 或 GDN recurrent 的接口。

完整接口和调用示例见 [API 文档](docs/api.md)，实现方案见 [设计文档](docs/design.md)。Shape 符号统一引用
[KDA 模型符号表](../README.md#model-shape-symbols)。

## Kimi K3 单算子验证接口

```python
out = torch.ops._C_ascend.recurrent_kda(
    q, k, v, raw_gate, beta, state, cu_seqlens, state_indices,
    A_log, dt_bias,
    num_accepted_tokens=num_accepted_tokens,
)
```

## 语义

- `layout="BSND"`：`q/k=[B,T,H,K]`，`v=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `layout="TND"`：`q/k=[T,H,K]`，`v=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- 私仓 Kimi K3 入口要求传入 `[state_capacity,HV,V,K]` 的容量型 state pool，并按 packed `[T]` 或 speculative `[seq_num,max_step]` 索引原地更新；未命中槽位保持不变。
- rank-3 q/k/v/g 使用 TND，rank-4 输入使用 BSND；`cu_seqlens` 和索引均留在 device，host 不读取其值。
- `cu_seqlens` 使用与 fla-org 一致的累积 offset 语义：首项为 0，末项等于有效 packed token 数，
  且可小于图捕获的 token capacity；相邻差值为各序列长度。
- 空序列不读取 state 索引或 state pool；`state` 是显式 mutable input，未命中槽位保持不变。
- `scale=None` 时，Python wrapper 使用 `K ** -0.5`。
- `use_qk_l2norm_in_kernel=True` 时，kernel 内对每个 token 的 `q/k` 做 L2 normalize，然后对 `q` 乘 `scale`。
- `use_gate_in_kernel=False` 时，`g` 被视为已经预计算好的 step log gate，kernel 使用 `exp(g)` 做 state decay。
- `use_gate_in_kernel=True` 时，`g` 是 raw gate，必须传 `A_log`；可选 `dt_bias`。
    - `safe_gate=False`：`gate = -exp(A_log) * softplus(g + dt_bias)`。
    - `safe_gate=True`：`gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- `use_beta_sigmoid_in_kernel=True` 时，kernel 使用 `sigmoid(beta)`；若 `allow_neg_eigval=True`，再乘 2。
- `_C_ascend`/aclnn 入口支持非连续 state，并通过临时连续 tensor 回写原 view。

每个 token 的 recurrent 更新为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ (q_t * scale)
```

## 当前限制

- `q/k/v/out` 仅支持 `BF16`。
- `g/beta` Python 入口支持 `FP32/BF16/FP16`，aclnn 预处理后以 `FP32` 输入 kernel。
- `A_log/dt_bias` 支持 `FP32`。
- Kimi K3 torch/aclnn 入口的 `cu_seqlens` 为必传的设备 INT32/INT64 Tensor；offset 必须单调不减，
  末项不得超过输入 token capacity，各相邻差值必须不超过 8。末项小于 capacity 时，仅有效 token
  对应的输出和 state 更新有定义，padding tail 输出不作保证。
- 仅支持 `layout="BSND"` 和 `layout="TND"`。
- 仅支持 `state_v_first=True`，state layout 为 `[state_capacity,HV,V,K]`；底层 aclnn 接口要求显式传入可变 state。
- `HV` 必须能被 `H` 整除；`H/HV <= 256`。底层 kernel 支持 `K=128,V=128/256`，当前 Kimi K3 torch 入口收窄为 `K=V=128`。
