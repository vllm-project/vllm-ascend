import vllm
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

import vllm_ascend.ops.gdn as gdn_ops
from vllm_ascend._310p.ops.fla.gdn_310 import (
    AscendGatedDeltaNetAttention310,
    update_conv1d_graph_params_310p,
)
from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import is_rc_device

vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices = prepare_chunk_indices_310

vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets = prepare_chunk_offsets_310

# 310P GDN causal conv1d uses buffer_replay; keep shared gdn.py unchanged.
gdn_ops.update_conv1d_graph_params = update_conv1d_graph_params_310p

# 310P: protect tail slot during MTP input_ids shift to avoid GatherV2 corruption
# caused by the NPU slice-assign writing one element past the intended range
# on the persistent drafter input_ids buffer.
AscendSpecDecodeBaseProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.set_inputs_first_pass
)

# Patch _warmup_prefill_kernels to no-op on 310P: triton.next_power_of_2 does
# not exist in the triton version used on 310P CI, and NPU does not use these
# CUDA warmup kernel anyway.
QwenGatedDeltaNetAttention._warmup_prefill_kernels = lambda self, qkv_or_qkvz, v_dim: None  # type: ignore[method-assign]
QwenGatedDeltaNetAttention._forward_core = AscendGatedDeltaNetAttention310._forward_core
QwenGatedDeltaNetAttention.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype

if is_rc_device():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    from vllm_ascend._310p.ops.gdn_attn_builder_310 import GDNAttentionMetadataBuilder310

    # Qwen3.5 on 310P RC uses upstream GDNAttentionBackend via MambaBase.get_attn_backend().
    GDNAttentionBackend.get_builder_cls = staticmethod(  # type: ignore[method-assign]
        lambda: GDNAttentionMetadataBuilder310
    )
