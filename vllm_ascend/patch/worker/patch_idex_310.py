from vllm.third_party.flash_linear_attention.ops import index as fla_index

from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import is_rc_device

fla_index.prepare_chunk_indices = prepare_chunk_indices_310
fla_index.prepare_chunk_offsets = prepare_chunk_offsets_310

# 310P: protect tail slot during MTP input_ids shift to avoid GatherV2 corruption
# caused by the NPU slice-assign writing one element past the intended range
# on the persistent drafter input_ids buffer.
AscendSpecDecodeBaseProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.set_inputs_first_pass
)
AscendSpecDecodeBaseProposer._run_merged_draft = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._run_merged_draft
)

if is_rc_device():
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    from vllm_ascend._310p.ops.gdn_attn_builder_310 import GDNAttentionMetadataBuilder310
    from vllm_ascend._310p.ops.qwen3vl_310 import rot_pos_emb_310

    # 310P RC: use blocking H2D in rot_pos_emb to avoid race with subsequent indexing.
    Qwen3_VisionTransformer.rot_pos_emb = rot_pos_emb_310  # type: ignore[method-assign]

    # Qwen3.5 on 310P RC uses upstream GDNAttentionBackend via MambaBase.get_attn_backend().
    GDNAttentionBackend.get_builder_cls = staticmethod(  # type: ignore[method-assign]
        lambda: GDNAttentionMetadataBuilder310
    )
