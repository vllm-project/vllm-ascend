# Operator Compatibility Baseline

This document defines **model new operator/custom kernel/Ascend operator interface compatibility** the current capability baseline of this layer.

## 1. What problem does this layer solve?

This layer focuses on:

- Whether the new model introduces Torch / Triton / CUDA / `torch_npu` / `aclnn` specific operators;
- Whether the current Ascend environment has functional support;
- Is the function not supported, the accuracy is unstable, or it just needs layout/dtype/shape adjustment.

## 2. Current capability baseline

The default decision table for the current skill is:

- Torch native op: usually runnable, performance to be verified;
- Triton: Both functionality and accuracy require explicit verification;
- CUDA kernel: Ascend does not support it and must have fallback;
- Ascend-specific op: Must refer to the official HiAscend document constraints.

## 3. Current implementation trends

### 3.1 Classify first, then decide whether to continue

Do not change the code directly before classification. Confirm first:

- Is this a pure CUDA path;
- Does Triton have Torch fallback;
- `torch_npu` / `aclnn` whether the call violates documentation restrictions.

### 3.2 You must check the official documentation for Ascend-specific ops

If the following types fail, don’t just rely on blind testing:

- `torch_npu.*`
- `torch.ops.npu.*`
- `aclnn*`

At least extract:

- dtype support
- shape constraints
- layout / contiguous constraints
- graph-mode restrictions
- fallback / replacement suggestions

## 4. Typical input evidence

- `modeling_*.py`
- `processing_*.py`
- vLLM adds model adapter file
- operator symbol appearing on the stack
- HiAscend operator documentation

## 5. Fixed output template

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

## 6. When must you stop?

- Pure CUDA kernel without fallback;
- Triton fails validation on Ascend and has no reasonable replacement;
- The official constraints of the Ascend-specific operator fundamentally conflict with the model requirements.
