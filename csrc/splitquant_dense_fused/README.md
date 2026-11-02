# SplitQuant unified operator interface

## Scope

The library is enabled by CMake on supported Ascend 910B systems and is
installed as `vllm_ascend/lib/libvllm_ascend_splitquant_dense.so`. It provides
the dense, multi-projection, state-cache, router, and MoE operators used by
the SplitQuant Qwen3 integration. The Python loader opens this bundled library
in each vLLM worker.

## 调用签名

```python
torch.ops.npu.splitquant_dense_fused(x, packed_weight, weight_scale, weight_offset, activation_clip_max, activation_clip_min, bias=None, group_size=128, group_trans=None, right_trans=None, left_trans=None)
torch.ops.npu.splitquant_dense_fused_multi(x, packed_weight, weight_scale, weight_offset, clip_max, clip_min, rotation_transposed, splits)
torch.ops.npu.splitquant_state_gather(cache, indices)
torch.ops.npu.splitquant_state_scatter(cache, source, indices)
torch.ops.npu.splitquant_moe_fused(x, up_weight, up_scale, down_weight_bf16, up_clip_max, up_clip_min, hidden_clip_max, hidden_clip_min, down_rotation, expert_ids, routing_weights, silu_lut)
torch.ops.npu.splitquant_merged_build_info()  # metadata only
```

## Tensor contract

Dense inputs use FP16 or BF16 `[M,K]` tensors with `K=G*128`; packed INT4
weights have shape `[N,K/2]`, scales and offsets `[N,G]`, and clips `[G]`.
The multi-projection operator uses BF16 activations and group-major packed
weights. State-cache operators support FP16, BF16, and FP32, including the
documented paged strides. MoE uses packed INT4 gate/up weights and BF16 down
weights. Inputs must be finite, on the same NPU, and use group size 128.
Autograd is not supported.

## Returns

Dense returns `[M,N]` in the activation dtype; gather returns compact selected
rows; scatter updates the cache; and MoE returns the shape and dtype of `x`.

## 加载与使用
The operator is built as `libvllm_ascend_splitquant_dense.so` and installed under
`vllm_ascend/lib` by the vLLM-Ascend CMake build on supported Atlas A2 systems.
The Python quantization methods load this bundled library in each worker process.
MoE calls use `torch.ops.npu.splitquant_moe_fused`; state helpers are registered
in the same library. Set the model-specific variables documented by the vLLM
deployment configuration before starting workers.

## Validation

Run `csrc/splitquant_dense_fused/test/run_equivalence.py` after building on an
NPU host. Model-level reports should include model and dataset checksums, the
native-library checksum, runtime flags, and router/MoE call counters.
