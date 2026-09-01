# MoE LoRA AllGather 路由数据说明

本文解释 vLLM Ascend 在 TP + AllGather（不开 EP）场景中，MoE dispatch 前后各类行号如何变化，以及
`expanded_row_idx`、`inv_perm`、`expert_per_row` 和 `lora_per_row` 是如何生成的。

对应实现位于：

- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm_ascend/lora/fused_moe.py`
- `vllm_ascend/lora/punica_npu.py`

## 1. 先区分四种“行”

假设当前 MoE 层收到：

```text
hidden_states.shape = [T, H]
topk_ids.shape       = [T, K]
```

其中：

- `T` 是当前批次进入该 MoE 层的 token 数量。
- `H` 是 hidden size。
- `K` 是每个 token 选择的 expert 数量，即 `top_k`。

需要区分以下概念：

| 名称 | 含义 | 取值范围 |
| --- | --- | --- |
| token row | dispatch 前 `hidden_states` 中的 token 行 | `0..T-1` |
| top-k slot | 一个 token 的第几个路由分支 | `0..K-1` |
| original pair row | 将 `(token, top-k slot)` 展平后的行，本文记为 `p` | `0..T*K-1` |
| dispatched row | 按 expert 排序后 `sorted_hidden_states` 中的行，本文记为 `r` | `0..T*K-1` |

原始 pair 行号的计算公式是：

```text
p = token_idx * K + topk_slot
```

因此，一个 token 会展开成 `K` 个 pair。它们的数据内容相同，但目标 expert 可能不同。

## 2. AllGather dispatch 做了什么

AllGather dispatcher 调用 `DeviceOperator.npu_moe_init_routing`：

```python
sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = (
    DeviceOperator.npu_moe_init_routing(
        hidden_states,
        topk_ids,
        active_num=num_tokens * top_k,
        ...
    )
)
```

该过程可以理解为两步：

1. 按 `topk_ids` 将每个 token 展开成 `K` 个 pair。
2. 将所有 pair 按 expert 分组，生成供 grouped matmul 使用的 `sorted_hidden_states`。

输出中：

- `sorted_hidden_states[r]` 是 dispatch 后第 `r` 行的数据。
- `expert_tokens[e]` 表示 expert `e` 分到了多少行，当前代码以 count 模式作为 GMM 的 `group_list`。
- `expanded_row_idx[p]` 表示原始 pair `p` 被移动到了哪个 dispatched row。

最重要的是最后一点：

```text
expanded_row_idx 的方向：original pair p -> dispatched row r
```

它不是“dispatch 后第 `r` 行来自哪个 pair”。后者是恢复 LoRA 路由时真正需要的方向。

## 3. 一个完整例子

假设有 3 个 token、2 个 expert，并且 `top_k=2`：

```text
hidden_states = [x0, x1, x2]

topk_ids =
[
  [1, 0],  # x0 选择 expert1 和 expert0
  [0, 1],  # x1 选择 expert0 和 expert1
  [1, 1],  # x2 的两个分支都选择 expert1
]
```

### 3.1 生成原始 pair

将 `topk_ids` 按行展平：

```python
flat_expert_ids = topk_ids.reshape(-1)
# [1, 0, 0, 1, 1, 1]
```

原始 pair 如下：

| pair `p` | token | top-k slot | `flat_expert_ids[p]` | 携带的数据 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 1 | `x0` |
| 1 | 0 | 1 | 0 | `x0` |
| 2 | 1 | 0 | 0 | `x1` |
| 3 | 1 | 1 | 1 | `x1` |
| 4 | 2 | 0 | 1 | `x2` |
| 5 | 2 | 1 | 1 | `x2` |

如果暂时不考虑 expert 排序，展开后的数据顺序是：

```text
pair order = [p0, p1, p2, p3, p4, p5]
data       = [x0, x0, x1, x1, x2, x2]
expert     = [ 1,  0,  0,  1,  1,  1]
```

### 3.2 Dispatch 按 expert 排序

expert0 的 pair 是 `p1、p2`，expert1 的 pair 是 `p0、p3、p4、p5`。因此排序结果可以表示为：

```text
dispatched row = [r0, r1, r2, r3, r4, r5]
original pair  = [p1, p2, p0, p3, p4, p5]
data           = [x0, x1, x0, x1, x2, x2]
expert         = [ 0,  0,  1,  1,  1,  1]
```

所以 grouped matmul 实际接收的 `sorted_hidden_states` 是：

```text
sorted_hidden_states = [x0, x1, x0, x1, x2, x2]
expert_tokens        = [2, 4]
```

这里 `expert_tokens=[2, 4]` 表示前 2 行交给 expert0，后 4 行交给 expert1。

### 3.3 生成 `expanded_row_idx`

`expanded_row_idx` 以原始 pair `p` 为下标，记录它的新位置 `r`：

| original pair `p` | 目标 expert | dispatched row `r` |
| ---: | ---: | ---: |
| 0 | 1 | 2 |
| 1 | 0 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 3 |
| 4 | 1 | 4 |
| 5 | 1 | 5 |

所以：

```python
expanded_row_idx = torch.tensor([2, 0, 1, 3, 4, 5])
```

其含义是：

```text
p0 -> r2
p1 -> r0
p2 -> r1
p3 -> r3
p4 -> r4
p5 -> r5
```

## 4. `expert_per_row` 是怎么恢复的

LoRA BGMV 输入已经是 dispatch 后的顺序，因此需要回答：

> dispatched row `r` 来自哪个原始 pair，它应该使用哪个 expert 的 LoRA 权重？

`expanded_row_idx` 的方向正好相反，所以先通过 `argsort` 构造逆置换：

```python
expanded = torch.abs(expanded_row_idx)
inv_perm = torch.argsort(expanded)
# [1, 2, 0, 3, 4, 5]
```

`inv_perm[r]` 表示 dispatched row `r` 来自哪个原始 pair `p`：

```text
r0 <- p1
r1 <- p2
r2 <- p0
r3 <- p3
r4 <- p4
r5 <- p5
```

随后，用这个原始 pair 编号去索引展平后的 expert ID：

```python
expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.long)
# [0, 0, 1, 1, 1, 1]
```

最终对应关系是：

| dispatched row `r` | `inv_perm[r]` | 数据 | `expert_per_row[r]` |
| ---: | ---: | --- | ---: |
| 0 | 1 | `x0` | 0 |
| 1 | 2 | `x1` | 0 |
| 2 | 0 | `x0` | 1 |
| 3 | 3 | `x1` | 1 |
| 4 | 4 | `x2` | 1 |
| 5 | 5 | `x2` | 1 |

因此，`expert_per_row` 的准确含义是：

```text
dispatch 后每一行应该使用哪个 expert 的权重
```

它的下标是 dispatched row，值是 expert ID。它不是 row index。

## 5. 为什么不能直接使用 `expanded_row_idx`

下面的写法是错误的：

```python
wrong_expert_per_row = topk_ids.reshape(-1)[expanded_row_idx]
```

在本例中会得到：

```text
flat_expert_ids                    = [1, 0, 0, 1, 1, 1]
flat_expert_ids[expanded_row_idx] = [0, 1, 0, 1, 1, 1]  # 错误
```

正确结果应该是：

```text
flat_expert_ids[argsort(expanded_row_idx)]
= [0, 0, 1, 1, 1, 1]
```

错误的根本原因是把两个方向混淆了：

```text
expanded_row_idx[p] = r   # 原始 pair -> dispatched row
inv_perm[r]         = p   # dispatched row -> 原始 pair
```

## 6. `lora_per_row` 是怎么恢复的

LoRA 路由还需要知道每个 dispatched row 属于哪个请求、应该选择哪个 LoRA adapter。

`inv_perm[r]` 给出原始 pair 编号。因为每个 token 有 `K` 个连续 pair，所以原始 token 行号是：

```python
orig_token = inv_perm // top_k
```

在本例中：

```text
inv_perm  = [1, 2, 0, 3, 4, 5]
top_k     = 2
orig_token = [0, 1, 0, 1, 2, 2]
```

假设三个 token 的 LoRA slot 是：

```python
token_lora_indices = torch.tensor([0, -1, 1])
```

其中 `-1` 表示该 token 不使用 LoRA。按照 dispatched row 顺序恢复：

```python
lora_per_row = token_lora_indices[orig_token]
# [0, -1, 0, -1, 1, 1]
```

现在 `expert_per_row`、`lora_per_row` 和 `sorted_hidden_states` 完全对齐：

| dispatched row | 数据 | expert ID | LoRA slot |
| ---: | --- | ---: | ---: |
| 0 | `x0` | 0 | 0 |
| 1 | `x1` | 0 | -1 |
| 2 | `x0` | 1 | 0 |
| 3 | `x1` | 1 | -1 |
| 4 | `x2` | 1 | 1 |
| 5 | `x2` | 1 | 1 |

## 7. BGMV 最终使用的索引

`moe_lora_apply_w13` 和 `moe_lora_apply_w2` 将下面两个张量传给 `add_lora_fused_moe`：

```text
expert_ids        = expert_per_row
token_lora_mapping = lora_per_row
```

Punica 将 LoRA slot 和 expert ID 合并成一个权重索引：

```python
combined_idx = lora_slot * num_experts + expert_id
```

不使用 LoRA、adapter 未启用的行会被设置为 `-1`。本例中 `num_experts=2`，因此：

```text
expert_per_row = [ 0,  0, 1,  1, 1, 1]
lora_per_row   = [ 0, -1, 0, -1, 1, 1]
combined_idx   = [ 0, -1, 1, -1, 3, 3]
```

`combined_idx=3` 的含义是选择 `LoRA slot 1 + expert 1` 对应的权重块：

```text
3 = 1 * 2 + 1
```

## 8. 当前实现的完整恢复代码

当前 AllGather 恢复逻辑可以概括为：

```python
def recover_allgather_routing(
    expanded_row_idx,
    topk_ids,
    token_lora_indices,
    top_k,
):
    # original pair -> dispatched row
    expanded = torch.abs(expanded_row_idx)

    # dispatched row -> original pair
    inv_perm = torch.argsort(expanded)

    # dispatched row -> expert ID
    expert_per_row = topk_ids.reshape(-1)[inv_perm].to(torch.long)

    # dispatched row -> original token -> LoRA slot
    orig_token = inv_perm // top_k
    orig_token = orig_token.clamp_(max=token_lora_indices.numel() - 1)
    lora_per_row = token_lora_indices[orig_token]

    return expert_per_row, lora_per_row
```

实际实现中的 `torch.abs` 用于按现有算子契约规范化 `expanded_row_idx`；恢复逻辑只使用其绝对位置。
`clamp_` 是图安全的防御处理，在正常输入下不会改变 `orig_token`。

## 9. Prefill 和 Decode 是否不同

路由算法本身没有区别，变化的只是 `T`：

- Prefill：`T` 通常是本轮被调度的所有 prompt token 数量之和。
- Decode：普通解码时，每个活跃请求通常贡献一个 token，因此 `T` 通常接近活跃请求数。

无论是 Prefill 还是 Decode，均遵循：

```text
T 个 token
  -> T*K 个 original pair
  -> 按 expert 排序成 T*K 个 dispatched row
  -> 通过 inv_perm 恢复每行的 expert 和 LoRA slot
```

## 10. 名称速查

仓库实现中并没有一个统一命名为 `expert_row_idx` 的变量。口头讨论“expert row idx”时，可能指的是不同对象：

| 变量 | 下标是什么 | 值是什么 | 方向 |
| --- | --- | --- | --- |
| `expanded_row_idx` | original pair `p` | dispatched row `r` | `p -> r` |
| `inv_perm` | dispatched row `r` | original pair `p` | `r -> p` |
| `expert_per_row` | dispatched row `r` | expert ID | `r -> expert` |
| `orig_token` | dispatched row `r` | 原始 token 行号 | `r -> token` |
| `lora_per_row` | dispatched row `r` | LoRA slot | `r -> LoRA` |
| `expert_tokens` | expert ID | 该 expert 的 dispatched row 数量 | `expert -> count` |

最简单的记忆方式是：

```text
expanded_row_idx：它去了哪里
inv_perm：这一行从哪里来
expert_per_row：这一行给谁算
lora_per_row：这一行用哪个 LoRA
```

## 11. 调试时如何检查

在离线单测或 eager 调试中，可以检查 `expanded_row_idx` 是否构成有效置换，以及恢复结果是否与 dispatch
后的行顺序一致：

```python
num_pairs = topk_ids.numel()
expanded = torch.abs(expanded_row_idx).to(torch.long)
expected = torch.arange(num_pairs, device=expanded.device)

assert torch.equal(torch.sort(expanded).values, expected)
assert expert_per_row.shape == (num_pairs,)
assert lora_per_row.shape == (num_pairs,)
```

这些断言仅适合测试和调试，不应直接加入 NPU 推理热路径，因为设备张量的 Python 条件判断可能造成
NPU 与 CPU 同步。
