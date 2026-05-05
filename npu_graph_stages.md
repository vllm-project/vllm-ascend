# NPU Graph 模式三阶段总结

## 概述

NPU Graph 模式通过分段式（piecewise）CUDA Graph 捕获技术，将模型前向计算录制为可重放的数据流图，以提升推理性能。该模式涉及 **3 个阶段**：

```
Warmup → Capture → Replay
  ↓         ↓         ↓
False    True      False
```

## 1. Warmup 阶段（预热阶段）

**文件位置**：`vllm_ascend/worker/v2/aclgraph_utils.py:142-143`

```python
# In warmup phase, capturing=False by default.
```

### 目的
- 验证模型前向逻辑正确性
- 预热 kernel、分配显存
- 不进行任何 graph 捕获

### capturing 状态
`False`

### 执行路径
```
_forward_decode_pcp_dcp → else 分支 (line 650-654)
```

### 代码特点
```python
# 直接调用，capturing=False 时走这个分支
attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(query, k_nope, value, **common_kwargs)
```

---

## 2. Capture 阶段（捕获阶段）

**文件位置**：`vllm_ascend/compilation/acl_graph.py:156`

```python
forward_context.capturing = True
with torch.npu.graph(aclgraph, pool=self.graph_pool):
    output = self.runnable(*args, **kwargs)
```

### 目的
- 将计算操作录制到 NPUGraph 中
- 收集所有 tensor 地址和参数信息
- 构建可重放的 graph

### capturing 状态
`True`

### 执行路径
```
_forward_decode_pcp_dcp → if _EXTRA_CTX.capturing 分支 (line 605-649)
```

### 代码特点
```python
# 1. 等待当前 stream 完成
event = torch.npu.ExternalEvent()
event.wait(stream)
event.reset(stream)
graph_params.events[num_tokens].append(event)

# 2. 获取/分配 workspace
workspace = graph_params.workspaces.get(num_tokens)
if workspace is None:
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)

# 3. 预分配输出 tensor
attn_out = torch.empty_like(query)
attn_lse = torch.empty((num_tokens, num_heads, 1), dtype=torch.float, device=query.device)

# 4. 收集 graph 参数供后续 replay
graph_params.attn_params[num_tokens].append((
    weak_ref_tensors(query),
    weak_ref_tensors(k_nope),
    weak_ref_tensors(value),
    ...
))

# 5. 使用 out= 语法进行 graph 捕获
torch.npu.graph_task_group_begin(stream)
torch_npu.npu_fused_infer_attention_score.out(
    query, k_nope, value, **common_kwargs, workspace=workspace, out=[attn_out, attn_lse]
)
handle = torch.npu.graph_task_group_end(stream)
graph_params.handles[num_tokens].append(handle)
```

---

## 3. Replay 阶段（回放阶段）

### 目的
- 复现已捕获的 graph
- 获得最优推理性能

### capturing 状态
`False`

### 执行路径
```
_forward_decode_pcp_dcp → else 分支 (line 650-654)
```

### 代码特点
```python
# 与 warmup 相同的调用方式，但 tensor 地址已被 graph 记录
attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(query, k_nope, value, **common_kwargs)
```

### 与 Warmup 的区别
| 维度 | Warmup | Replay |
|------|--------|--------|
| 目的 | 验证功能正确性 | 性能优化 |
| tensor 地址 | 新分配 | 与 Capture 相同 |
| kernel 编译 | JIT 编译 | 直接加载预编译版本 |
| 性能 | 慢 | 最优 |

---

## 关键变量说明

### `graph_params` 结构
```python
graph_params = {
    "events": {},        # NPU events per num_tokens
    "workspaces": {},    # 预分配的 workspace per num_tokens
    "attn_params": {},   # attention 参数（query/k/v 弱引用等）
    "handles": {},       # graph_task_group handles
}
```

### `_EXTRA_CTX.capturing` 判断逻辑
```python
# aclgraph_utils.py
if torch.npu.is_current_stream_capturing():
    _EXTRA_CTX.capturing = True
```

---

## 总结

| 阶段 | capturing | 主要操作 | 性能 |
|------|-----------|----------|------|
| Warmup | `False` | 验证正确性、预热 | 慢 |
| Capture | `True` | 录制 graph、收集参数 | 中 |
| Replay | `False` | 重放 graph | 最优 |
