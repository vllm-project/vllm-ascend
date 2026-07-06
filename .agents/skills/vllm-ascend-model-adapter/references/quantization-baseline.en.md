# Quantization Baseline

This document defines the current baseline for quantization-related adaptation, especially fp8 checkpoints, KV quantization, W8A8, and C8.

## What this layer focuses on

It focuses on:

- whether the checkpoint is fp8, int8, compressed-tensors, or another quant format
- whether the problem is a weight-loading quant issue or a runtime kernel or operator issue
- which quantized execution paths already exist on Ascend
- whether a safer bf16 or fallback path should be preferred

## Current baseline

- fp8 on NPU often prefers load-time dequant to bf16
- KV quant, especially C8 KV cache, already has dedicated attention support
- W8A8 and compressed-tensors are supported in several Ascend-specific paths, but not every format can be assumed to work automatically
- dummy runs do not replace real-weight verification for quantized models
