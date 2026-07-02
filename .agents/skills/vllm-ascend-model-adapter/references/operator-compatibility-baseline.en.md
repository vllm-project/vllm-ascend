# Operator Compatibility Baseline

This document defines the current baseline for newly introduced operators, custom kernels, and Ascend-operator compatibility.

## What this layer focuses on

It focuses on:

- whether a new model introduces Torch, Triton, CUDA, `torch_npu`, or `aclnn` operators
- whether the current Ascend environment can support them functionally
- whether a failure is caused by lack of support, unstable accuracy, or only layout, dtype, or shape mismatch

## Current decision baseline

- Torch native ops are usually functionally runnable, but performance still needs verification.
- Triton kernels require explicit functional and numerical verification.
- CUDA kernels are unsupported on Ascend unless a fallback exists.
- Ascend-specific ops must be checked against official HiAscend documentation.
