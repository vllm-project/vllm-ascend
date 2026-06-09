# Operator Compatibility Baseline

本文档定义 **模型新增算子 / 自定义 kernel / Ascend 算子接口兼容性** 这一层的当前能力基线。

## 1. 这一层解决什么问题

这一层关注：

- 新模型是否引入了 Torch / Triton / CUDA / `torch_npu` / `aclnn` 特定算子；
- 当前 Ascend 环境是否具备功能支持；
- 是功能不支持、精度不稳定，还是只需要 layout / dtype / shape 调整。

## 2. 当前能力基线

当前 skill 的默认决策表是：

- Torch native op：通常可运行，性能待验证；
- Triton：功能与精度都需要显式验证；
- CUDA kernel：Ascend 不支持，必须有 fallback；
- Ascend-specific op：必须参考官方 HiAscend 文档约束。

## 3. 当前实现倾向

### 3.1 先分类，再决定是否继续

不要在未分类前直接改代码。先确认：

- 这是不是纯 CUDA 路径；
- Triton 是否有 Torch fallback；
- `torch_npu` / `aclnn` 调用是否违反了文档限制。

### 3.2 对 Ascend-specific op 必须查官方文档

以下类型一旦失败，不要只靠盲试：

- `torch_npu.*`
- `torch.ops.npu.*`
- `aclnn*`

至少提取：

- dtype 支持
- shape 约束
- layout / contiguous 约束
- graph-mode 限制
- fallback / replacement 建议

## 4. 典型输入证据

- `modeling_*.py`
- `processing_*.py`
- vLLM 新增 model adapter 文件
- 栈里出现的 operator symbol
- HiAscend operator 文档

## 5. 固定输出模板

```markdown
## Operator Compatibility Gap Analysis

### 1. Current Capability
- Existing supported operator class:
- Existing fallback expectations:
- Existing Ascend doc-backed constraints:

### 2. Model Requirement
- New operators introduced:
- Operator type per item:
- Required dtype/layout/shape:
- Expected fallback path:

### 3. Gap
- Unsupported operator:
- Missing fallback:
- Constraint mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal fallback or call-site change:
- Validation focus:
- Stop / escalate condition:
```

## 6. 什么时候必须停止

- 纯 CUDA kernel 且无 fallback；
- Triton 在 Ascend 上验证失败且没有合理替代；
- Ascend-specific operator 的官方约束与模型要求根本冲突。
