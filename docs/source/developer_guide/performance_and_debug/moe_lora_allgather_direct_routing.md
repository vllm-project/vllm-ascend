# AllGather 场景下量化 MoE LoRA 直接路由优化设计

## 1. 背景与目标

Ascend AllGather MoE 路径通过 `npu_moe_init_routing_v2` 将 token 展开并
重排为 expert-sorted 顺序。量化 W13/W2 使用重排后的 tensor 执行 GMM，
而 LoRA 还需要知道每一行对应的 adapter 和 expert。

当前 `_recover_moe_lora_routing_allgather` 通过以下逻辑恢复路由：

```python
expanded = torch.abs(expanded_row_idx)
inv_perm = torch.argsort(expanded)
expert_per_row = topk_ids.reshape(-1)[inv_perm]
orig_token = inv_perm // top_k
lora_per_row = token_lora_indices[orig_token]
```

随后 `add_lora_fused_moe` 再构造：

```python
combined_idx = lora_id * num_experts + expert_id
```

这会引入全量 `argsort`、多个索引临时张量、W13/W2 重复 metadata 构造，
以及多个 BGMV kernel launch。本设计目标是：

1. 保存 routing 前的 BF16 hidden states；
2. W13 直接消费 `expanded_row_idx` 并累加到 base W13；
3. W13 同时生成一次 dispatched-order `combined_idx`；
4. W2 连续访问并复用该 `combined_idx`；
5. 从 AllGather 热路径删除路由恢复的 `argsort`；
6. 保持 AlltoAll/EP 路径不变。

## 2. 路由语义

定义：

```text
T = token 数量
K = top_k
P = T * K
p = 原始 (token, top-k slot) pair 编号
r = expert-sorted dispatched row 编号
```

`expanded_row_idx` 的方向为：

```text
expanded_row_idx[p] = r
```

对于每个原始 pair：

```python
token = p // K
expert = topk_ids.reshape(-1)[p]
lora = token_lora_indices[token]
row = abs(expanded_row_idx[p])
combined = lora * num_experts + expert
```

正常 AllGather 路径中，`abs(expanded_row_idx)` 应为 `[0, P)` 的一个
置换，因此每个 pair 都有唯一目标 row，按 pair 写入 dispatched layout
不会产生多个 core 写同一行的冲突。

该置换性质只能在单测或离线调试中检查，不应在推理热路径执行
`sort/equal`。

## 3. 最终数据流

```text
routing 前 BF16 hidden [T, H]
           |
           | W13 LoRA 按原始 pair p 计算
           v
gate_up_out[expanded_row_idx[p]] += W13 LoRA delta
           |
           | 同时生成一次
           v
combined_idx[expanded_row_idx[p]] = lora * E + expert
           |
           | activation + quantized base W2
           v
W2 LoRA 按连续 dispatched row 使用 combined_idx
           |
           v
down_out[row] += W2 LoRA delta
```

W13 使用 routing 前输入，避免恢复 inverse permutation；W2 不按原始 pair
随机读取 activation，而是复用 W13 生成的 dispatched-order index。

## 4. 保存 routing 前的 BF16 hidden states

### 4.1 保存位置

在 `TokenDispatcherWithAllGather.token_dispatch` 中，调用
`DeviceOperator.npu_moe_init_routing` 之前保存。

若启用 `apply_router_weight_on_input`，必须先应用 routed weight：

```python
if apply_router_weight_on_input:
    hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

original_hidden_states = hidden_states

sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = (
    DeviceOperator.npu_moe_init_routing(
        hidden_states,
        topk_ids,
        ...
    )
)
```

保存的 tensor 必须与真正传入 routing 算子的 tensor 语义一致，否则 base
W13 和 LoRA W13 会使用不同输入。

### 4.2 传递与生命周期

优先仅保存引用，不执行 `clone()`：

```python
original_hidden_states = hidden_states
```

建议在 `MoEMlpComputeInput` 增加：

```python
original_hidden_states: torch.Tensor | None = None
```

仅在 AllGather 且 batch 存在 LoRA 时设置。W2 完成后清空引用，避免延长
大 tensor 生命周期。实现前必须确认后续没有原地修改该 tensor。

## 5. W13 直接路由计算

### 5.1 数学关系

对每个原始 pair `p`：

```python
token = p // top_k
expert = flat_topk_ids[p]
lora = token_lora_indices[token]
row = abs(expanded_row_idx[p])

tmp = original_hidden_states[token] @ lora_a[lora, expert].T
delta = tmp @ lora_b[lora, expert].T
gate_up_out[row] += delta
```

W13 通常有 gate/up 两个 slice：

```text
gate_up_out[row, 0:I]   += gate LoRA delta
gate_up_out[row, I:2*I] += up LoRA delta
```

必须在 activation 前累加：

```text
正确：activation(base_w13 + lora_w13)
错误：activation(base_w13) + lora_w13
```

### 5.2 第一版 AscendC 接口

第一版可保留 shrink/expand 两阶段，增加 AllGather 专用算子：

```python
torch.ops._C_ascend.bgmv_shrink_moe_allgather(
    original_hidden_states,
    lora_a_weights,
    flat_topk_ids,
    token_lora_indices,
    adapter_enabled,
    shrink_out,
    top_k,
    num_experts,
)

torch.ops._C_ascend.bgmv_expand_moe_allgather(
    shrink_out,
    lora_b_weights,
    expanded_row_idx,
    flat_topk_ids,
    token_lora_indices,
    adapter_enabled,
    gate_up_out,
    combined_idx,
    top_k,
    num_experts,
    slice_offset,
    slice_size,
    write_combined_idx,
)
```

`shrink_out` 按原始 pair 存放：

```text
shrink_out[p, rank]
```

Shrink kernel 伪代码：

```cpp
for (int64_t p = start; p < end; ++p) {
    int64_t token = p / topK;
    int64_t expert = topkIds[p];
    int64_t lora = tokenLoraIndices[token];

    if (lora < 0 || adapterEnabled[lora] == 0) {
        ZeroShrinkOutput(p);
        continue;
    }

    int64_t weightIdx = lora * numExperts + expert;
    LoadX(originalHiddenStates[token]);
    LoadWeightA(weightIdx);
    ComputeShrink();
    StoreShrinkOutput(p);
}
```

Expand kernel 伪代码：

```cpp
for (int64_t p = start; p < end; ++p) {
    int64_t row = Abs(expandedRowIdx[p]);
    int64_t token = p / topK;
    int64_t expert = topkIds[p];
    int64_t lora = tokenLoraIndices[token];

    if (lora < 0 || adapterEnabled[lora] == 0) {
        if (writeCombinedIdx) {
            combinedIdx[row] = -1;
        }
        continue;
    }

    int64_t weightIdx = lora * numExperts + expert;
    if (writeCombinedIdx) {
        combinedIdx[row] = weightIdx;
    }

    LoadShrinkOutput(p);
    LoadWeightB(weightIdx);
    ComputeExpand();
    AddToOutput(gateUpOut[row], sliceOffset);
}
```

只有一个 W13 slice 写 `combined_idx`，另一个 slice 关闭该输出，避免重复
写 metadata。

### 5.3 Python 接入

AllGather 使用独立入口：

```python
moe_lora_apply_w13_allgather(
    lora_context,
    gate_up_out=gate_up_out,
    original_hidden_states=mlp_compute_input.original_hidden_states,
    expanded_row_idx=mlp_compute_input.expanded_row_idx,
    topk_ids=mlp_compute_input.topk_ids,
)
```

建议在 `PunicaWrapperNPU` 增加：

```python
add_lora_fused_moe_allgather_w13(...)
```

避免向通用 `add_lora_fused_moe` 继续加入大量可选参数，以免影响
AlltoAll 和 dense LoRA 的可维护性。

## 6. W2 复用路由

### 6.1 上游 vLLM 思路

上游 Triton MoE 的中间缓存可按原始 flat pair 编号寻址。W13 LoRA 生成：

```text
sorted_token_ids_lora
expert_ids_lora
num_tokens_post_padded_lora
token_lora_mapping
```

W2 直接复用这些 metadata，并使用未量化 activation 计算 LoRA。参考：

- [vLLM Punica GPU implementation](https://github.com/vllm-project/vllm/blob/main/vllm/lora/punica_wrapper/punica_gpu.py)
- [vLLM Triton MoE experts](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/triton_moe.py)
- [vLLM fused MoE LoRA kernel](https://github.com/vllm-project/vllm/blob/main/vllm/lora/ops/triton_ops/fused_moe_lora_op.py)

Ascend activation 是 expert-sorted layout，不能原样照搬 flat pair 索引，
但应复用“W13 只生成一次路由、W2 直接复用”的原则。

### 6.2 Ascend 实现

W13 已生成：

```text
combined_idx[row] = lora * num_experts + expert
```

W2 继续使用现有 BGMV，直接传入 dispatched-order index：

```python
bgmv_shrink(
    activated,
    w2_a_flat,
    shrink_out,
    combined_idx,
    1.0,
)

bgmv_expand_slice(
    shrink_out,
    w2_b_flat,
    down_out,
    combined_idx,
    offset,
    out_size,
    add_inputs=True,
)
```

这样 W2 连续读取 `activated[row]` 和 `combined_idx[row]`，连续写入
`down_out[row]`，不再执行 `argsort`，也不需要按 pair 随机读取
`activated[expanded_row_idx[p]]`。

### 6.3 Routed weight

上游 W2 LoRA kernel 通常在 kernel 内乘 `topk_weights`。当前 Ascend
量化路径可能已经执行：

```python
activated *= topk_scales
```

如果 W2 LoRA 使用该 `activated`，不得再次乘 routed weight，否则得到
`topk_weight ** 2`。必须分别验证：

- `apply_router_weight_on_input=False/True`；
- `topk_scales is None/not None`。

## 7. 正确性原型

在实现 AscendC kernel 前，可以用 PyTorch/BGMV 验证数学关系。该版本
允许临时 materialize 和 scatter，但不能作为最终性能实现：

```python
flat_topk_ids = topk_ids.reshape(-1).long()
num_pairs = flat_topk_ids.numel()
pair_ids = torch.arange(num_pairs, device=topk_ids.device)
token_ids = pair_ids // top_k

lora_ids = token_lora_indices[token_ids]
safe_lora_ids = lora_ids.clamp(min=0)
enabled = (lora_ids >= 0) & adapter_enabled[safe_lora_ids].bool()

combined_per_pair = safe_lora_ids * num_experts + flat_topk_ids
combined_per_pair = torch.where(
    enabled,
    combined_per_pair,
    torch.full_like(combined_per_pair, -1),
)

pair_hidden_states = original_hidden_states[token_ids]
delta_pair = run_existing_bgmv(pair_hidden_states, combined_per_pair)

dispatched_rows = expanded_row_idx.abs().long()
delta_sorted = torch.empty_like(delta_pair)
delta_sorted[dispatched_rows] = delta_pair
gate_up_out.add_(delta_sorted)
```

该原型用于验证保存时机、权重选择、路由方向和 W13 layout 对齐。完整
delta buffer、scatter 和额外 `add_` 会影响性能，验证通过后必须替换成
直接写 `gate_up_out[row]` 的 AscendC kernel。

## 8. 分阶段实施

### 阶段一：基线与观测

1. 分别采集 Prefill 和 Decode profile。
2. 记录 `T`、`top_k`、rank、active LoRA 数和 expert 数。
3. 分离 routing recovery、W13 LoRA、W2 LoRA 和 base GMM 时间。
4. 调试打印不得用于最终性能数据。

### 阶段二：正确性原型

1. 保存 routing 前 BF16 hidden states。
2. 使用 pair-order BGMV 计算 W13 LoRA。
3. 根据 `expanded_row_idx` scatter 后与 base W13 相加。
4. 与当前 argsort 实现逐层、逐元素对比。

### 阶段三：直接路由 W13 kernel

1. 新增 AllGather routed shrink/expand custom op。
2. Expand 直接写 `gate_up_out[row]`。
3. 同时输出 dispatched-order `combined_idx[row]`。
4. W13 gate/up slice 共用 metadata。

### 阶段四：W2 路由复用

1. W2 删除 `_recover_moe_lora_routing_allgather`。
2. W2 直接使用 W13 生成的 `combined_idx`。
3. 预分配或复用 FP32 shrink workspace。
4. 验证 routed weight 只应用一次。

### 阶段五：kernel 融合

参考上游 one-shot/small-batch 路径：

- Decode、小 batch、低 rank 时融合 `A + B`；
- W13 gate/up slice 在同一 kernel 内处理；
- Prefill 大 batch 按 `(LoRA, expert)` block 化并使用 Cube/GMM；
- 按 `T * top_k`、expert 数和 active LoRA 数选择 naive/block 路径。

### 阶段六：重评双流

仅在 LoRA kernel 完成融合后重新测试双流。必须计入 delta workspace、
event 同步、最终 `add_`、AIV/Cube 竞争及 ACLGraph 稳定性。如果额外
开销大于被隐藏的 LoRA 时间，应保留单流。

## 9. 测试要求

### 9.1 单元测试

至少覆盖：

- 单 LoRA 和 Multi-LoRA；
- `lora_id=-1` 与 adapter disabled；
- `top_k=1` 和 `top_k>1`；
- 非平凡 `expanded_row_idx` 置换；
- 一个 token 的多个 slot 选择相同 expert；
- W13 两个 slice；
- fully-sharded offset/通信语义；
- FP16/BF16；
- 空 dispatched rows。

比较新旧实现：

```python
torch.testing.assert_close(new_gate_up_out, ref_gate_up_out)
torch.testing.assert_close(new_down_out, ref_down_out)
torch.testing.assert_close(new_final_out, ref_final_out)
```

### 9.2 NPU 测试

真实 NPU 上验证：

- Eager Prefill/Decode；
- ACLGraph capture/replay；
- AllGather TP；
- W8A8 dynamic；
- 不同 batch、prompt length、LoRA rank；
- 连续请求无 workspace 污染；
- 输出精度相对当前实现无回退；
- AlltoAll/EP 回归无变化。

## 10. 性能验收

分别报告 Prefill 和 Decode：

| 指标 | 当前实现 | 新实现 | 目标 |
| --- | ---: | ---: | ---: |
| routing recovery 时间 | 待测 | 待测 | 基本消除 |
| W13 LoRA 时间 | 待测 | 待测 | 明显下降 |
| W2 LoRA 时间 | 待测 | 待测 | 不回退 |
| 单层 MoE 时间 | 待测 | 待测 | 下降 |
| TTFT | 待测 | 待测 | 不回退 |
| TPOT | 待测 | 待测 | 不回退 |
| 吞吐 | 待测 | 待测 | 提升 |

测试必须使用相同模型、权重、请求、TP、ACLGraph 和 warmup 次数。性能
profile 中不得保留 `print`、tensor dump 或设备同步断言。

## 11. 风险与约束

- 必须用真实 NPU 验证 `expanded_row_idx` 的符号和置换语义。
- expert ID 必须与本 rank LoRA 权重 expert 维度一致。
- original hidden 不得被原地修改或跨层长期持有。
- inactive row 使用 `empty` workspace 时，shrink 必须显式写零。
- Expand 对 inactive row 只能跳过，不能清空 base GMM 输出。
- routed weight 必须与 base MoE 保持一致且只应用一次。
- 设备 tensor 的 `.item()` 不得进入推理热路径。
- custom op 必须保持静态输出 shape，以兼容 ACLGraph。

## 12. 推荐最小闭环

1. 保存 routing 前 BF16 hidden states；
2. W13 routed shrink/expand 直接写 `gate_up_out[row]`；
3. W13 输出 `combined_idx[row]`；
4. W2 复用 `combined_idx`；
5. 删除 AllGather 热路径的 argsort recovery；
6. 完成正确性、ACLGraph、Prefill/Decode 性能验证；
7. 再评估 one-shot、block assignment 和双流。

该顺序保持量化 base GMM、AlltoAll/EP 和现有权重布局不变，便于独立
评估每一阶段的正确性和性能收益。
