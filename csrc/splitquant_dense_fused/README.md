# SplitQuant 统一算子接口

## 功能与验证边界
14B、27B、30B使用同一份已验证原生库。通用Dense、state和MoE接口均由当前唯一 V9 原生库提供；76项算子检查及三个模型全量精度全部零差。当前部署和验证均使用此统一库。

## 调用签名
```python
torch.ops.npu.splitquant_dense_fused(x, packed_weight, weight_scale, weight_offset, activation_clip_max, activation_clip_min, bias=None, group_size=128, group_trans=None, right_trans=None, left_trans=None)
torch.ops.npu.splitquant_dense_fused_multi(x, packed_weight, weight_scale, weight_offset, clip_max, clip_min, rotation_transposed, splits)
torch.ops.npu.splitquant_state_gather(cache, indices)
torch.ops.npu.splitquant_state_scatter(cache, source, indices)
torch.ops.npu.splitquant_moe_fused(x, up_weight, up_scale, down_weight_bf16, up_clip_max, up_clip_min, hidden_clip_max, hidden_clip_min, down_rotation, expert_ids, routing_weights, silu_lut)
torch.ops.npu.splitquant_merged_build_info()  # metadata only
```

## 参数、dtype和形状
Dense激活FP16/BF16 [M,K]、K=G*128，权重INT4打包字节[N,K/2]，single scale/offset [N,G]、clip [G]。multi激活BF16，权重[G,N,64]，scale/offset[G,N]，clip[P,G]、rotation[G,128,128]，splits给出每个投影宽度。state支持FP16/BF16/FP32，页面可带非紧凑首维stride，indices为同设备INT32/INT64。MoE使用当前统一库接口，gate/up INT4、down BF16缓存；输入及约束见ops.h和原README。参数应同设备、有限，group128；不支持autograd。

## 返回
Dense返回同激活dtype的[M,N]；gather返回紧凑选中行，scatter写回原cache并返回别名；MoE返回与x同shape/dtype。metadata返回构建标识字符串。

## 加载与使用
The operator is built as `libvllm_ascend_splitquant_dense.so` and installed under
`vllm_ascend/lib` by the vLLM-Ascend CMake build on supported Atlas A2 systems.
The Python quantization methods load this bundled library in each worker process.
MoE calls use `torch.ops.npu.splitquant_moe_fused`; state helpers are registered
in the same library. Set the model-specific variables documented by the vLLM
deployment configuration before starting workers.

## 验证
原始算子对照76/76通过，完整ARC_c/ARC_e/PIQA每模型5386题17864候选，相对各自原量化实现分数及预测全部零差。测评结果和回归脚本均针对当前统一库。当前单库回归入口为test/run_equivalence.py，可对照保留的原版outputs.pt。完整性能结果见实验目录REPORT.md。
