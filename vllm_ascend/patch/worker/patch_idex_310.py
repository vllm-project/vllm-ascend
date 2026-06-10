import vllm

from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310
from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)
from vllm_ascend.utils import vllm_version_is

vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices = prepare_chunk_indices_310

vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets = prepare_chunk_offsets_310

# Patch _warmup_prefill_kernels to no-op on 310P: triton.next_power_of_2 does
# not exist in the triton version used on 310P CI, and NPU does not use these
# CUDA warmup kernel anyway.
if vllm_version_is("0.21.0"):
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (  # type: ignore[import-not-found]
        GatedDeltaNetAttention,
    )

    GatedDeltaNetAttention._warmup_prefill_kernels = lambda self, qkv_or_qkvz, v_dim: None  # type: ignore[method-assign]
    GatedDeltaNetAttention._forward_core = AscendGatedDeltaNetAttention310._forward_core
    GatedDeltaNetAttention.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype
else:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    QwenGatedDeltaNetAttention._warmup_prefill_kernels = lambda self, qkv_or_qkvz, v_dim: None  # type: ignore[method-assign]
    QwenGatedDeltaNetAttention._forward_core = AscendGatedDeltaNetAttention310._forward_core
    QwenGatedDeltaNetAttention.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype
