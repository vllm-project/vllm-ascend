import vllm.model_executor.layers.fla.ops.fused_recurrent
import vllm.model_executor.layers.fla.ops.layernorm_guard
import vllm.model_executor.layers.mamba.ops.causal_conv1d

from vllm_ascend.ops.casual_conv1d import (causal_conv1d_fn,
                                           causal_conv1d_update_npu)
from vllm_ascend.ops.fla import LayerNormFn
from vllm_ascend.ops.sigmoid_gating import \
    fused_recurrent_gated_delta_rule_fwd_kernel

vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = fused_recurrent_gated_delta_rule_fwd_kernel
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn