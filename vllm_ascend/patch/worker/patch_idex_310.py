import vllm

from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import is_rc_device

vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices = prepare_chunk_indices_310

vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets = prepare_chunk_offsets_310

# 310P: protect tail slot during MTP input_ids shift to avoid GatherV2 corruption
# caused by the NPU slice-assign writing one element past the intended range
# on the persistent drafter input_ids buffer.
AscendSpecDecodeBaseProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.set_inputs_first_pass
)

if is_rc_device():
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    from vllm_ascend._310p.ops.qwen3vl_310 import rot_pos_emb_310

    # 310P RC: use blocking H2D in rot_pos_emb to avoid race with subsequent indexing.
    Qwen3_VisionTransformer.rot_pos_emb = rot_pos_emb_310  # type: ignore[method-assign]
