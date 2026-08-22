class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        # File: /mnt/share/l00889328/10t/vllm-ascend/tests/ut/ops/test_gdn_layerwise_kv.py:85 in split_ba, code: return ba[..., :midpoint], ba[..., midpoint:]
        slice_1: "f32[2, 1]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 1)
        
        # File: /mnt/share/l00889328/10t/vllm-ascend/vllm_ascend/ops/gdn.py:85 in forward, code: b = b.contiguous()
        clone: "f32[2, 1]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        
        # File: /mnt/share/l00889328/10t/vllm-ascend/tests/ut/ops/test_gdn_layerwise_kv.py:85 in split_ba, code: return ba[..., :midpoint], ba[..., midpoint:]
        slice_2: "f32[2, 1]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807)
        
        # File: /mnt/share/l00889328/10t/vllm-ascend/vllm_ascend/ops/gdn.py:86 in forward, code: a = a.contiguous()
        clone_1: "f32[2, 1]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        
        # File: /mnt/share/l00889328/10t/vllm-ascend/vllm_ascend/ops/gdn.py:119 in forward, code: core_attn_out = torch.zeros(
        full_default: "f32[2, 1, 2]" = torch.ops.aten.full.default([2, 1, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        
        # File: /mnt/share/l00889328/10t/vllm-ascend/vllm_ascend/ops/gdn.py:125 in forward, code: torch.ops.vllm.qwen_gdn_attention_core(
        qwen_gdn_attention_core_default = torch.ops.vllm.qwen_gdn_attention_core.default(arg0_1, clone, clone_1, full_default, 'layers.0.linear_attn');  arg0_1 = clone = clone_1 = qwen_gdn_attention_core_default = None
        
        # No stacktrace found for following nodes
        view_8: "f32[2, 2]" = torch.ops.aten.reshape.default(full_default, [-1, 2]);  full_default = None
        return (view_8,)
        