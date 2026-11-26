# SPDX-License-Identifier: Apache-2.0
"""Graph-external TP1 decode embedding preparation.

No module/model forward is replaced. Caller must supply valid vocabulary IDs,
apart from the negative speculative holes whose original semantics are row 0.
"""

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding


@triton.jit
def _target_embedding_prepare_kernel(
    ids,
    weight,
    output,
    ID_STRIDE: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    WEIGHT_COL: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    OUTPUT_COL: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(ids + row * ID_STRIDE).to(tl.int64)
    token = tl.maximum(token, 0)
    values = tl.load(weight + token * WEIGHT_ROW + col * WEIGHT_COL, col < HIDDEN, other=0)
    tl.store(output + row * OUTPUT_ROW + col * OUTPUT_COL, values, col < HIDDEN)


def _embedding_eligible(embedding, ids, output):
    """Metadata-only gate; no device data reads, reductions or uploads."""
    if type(embedding) is not AscendVocabParallelEmbedding:
        return False
    if type(embedding.quant_method) is not UnquantizedEmbeddingMethod:
        return False
    if embedding.tp_size != 1 or embedding.token_exchange_group is not None:
        return False
    reduce_group = getattr(embedding, "output_reduce_group", None)
    if reduce_group is not None and reduce_group.world_size != 1:
        return False
    weight = embedding.weight
    return (
        ids.ndim == 1
        and output.ndim == weight.ndim == 2
        and ids.dtype in (torch.int32, torch.int64)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.device.type == "npu"
        and ids.device == weight.device == output.device
        and output.shape[0] >= ids.numel()
        and output.shape[1] == weight.shape[1]
        and weight.shape[0] >= embedding.num_embeddings
        and ids.stride(0) > 0
        and min(weight.stride()) > 0
        and min(output.stride()) > 0
    )


def try_prepare_target_embeddings(state, scheduled_encoder_inputs, input_batch, req_states):
    """Return preallocated buffer view, or None without touching device state.

    Covers pure decode with no media work, no prompt embeddings, no pruning,
    no PCP/TP exchange, no quantized/LoRA/custom embedding implementation.
    Padding rows remain untouched, matching the original prepare_inputs_embeds.
    """
    if (
        scheduled_encoder_inputs
        or getattr(input_batch, "has_prefill", True)
        or input_batch.is_dummy
        or state.prompt_embeds_state is not None
        or getattr(state, "pcp_context", None) is not None
    ):
        return None
    core = getattr(state.model, "language_model", state.model).model
    if not getattr(core, "uses_dfly", False):
        return None
    embedding = core.embed_tokens
    if state.supports_mm_inputs:
        runner = state.encoder_runner
        if runner.is_realtime or state.mm_pruner is not None:
            return None
        output = runner.inputs_embeds
    else:
        output = state.inputs_embeds
    ids = input_batch.input_ids[: input_batch.num_tokens]
    if not _embedding_eligible(embedding, ids, output):
        return None
    # Match AscendModelState.prepare_inputs_embeds's host metadata update.
    setter = getattr(state.model, "set_embedding_num_tokens_padded", None)
    if setter is not None:
        setter(input_batch.num_tokens_after_padding)
    if ids.numel():
        _target_embedding_prepare_kernel[(ids.numel(), triton.cdiv(output.shape[1], 256))](
            ids,
            embedding.weight,
            output,
            ids.stride(0),
            *embedding.weight.stride(),
            *output.stride(),
            output.shape[1],
            256,
        )
    return output[: input_batch.num_tokens_after_padding]
