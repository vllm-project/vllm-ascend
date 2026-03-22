import vllm.model_executor.layers.mamba.ops.causal_conv1d
import vllm.v1.worker.gpu.sample.gumbel

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_recurrent_gated_delta_rule_fwd_kernel
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample as ascend_gumbel_sample

vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule
vllm.v1.worker.gpu.sample.gumbel.gumbel_sample = ascend_gumbel_sample
