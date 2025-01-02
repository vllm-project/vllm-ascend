from vllm.model_executor.custom_op import CustomOp

import vllm_ascend_plugin.ops.layernorm

def forward_npu(self, *args, **kwargs):
    # By default, we assume that NPU ops are compatible with the
    # PyTorch-native implementation.
    return self.forward_native(*args, **kwargs)

CustomOp.set_foward_method(forward_npu)
