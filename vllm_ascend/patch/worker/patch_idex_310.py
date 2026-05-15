import vllm

from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)

vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices = prepare_chunk_indices_310

vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets = prepare_chunk_offsets_310
