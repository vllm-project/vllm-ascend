from vllm.forward_context import get_forward_context

from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker.ubatching import UBatchEventKey, dbo_record_current_stream, dbo_wait_current_stream_and_yield


class GlmMoeDsaAllgatherTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A2 (MLA+MoE model, same architecture as DeepSeek V2/V3):
    #
    # GlmMoeDsaForCausalLM inherits from DeepseekV2ForCausalLM with identical
    # forward flow: MLA attention (MultiHeadLatentAttention) + SharedFusedMoE.
    #
    # Per-layer hook call sequence:
    #   dbo_mla_preprocess_hook(True)   <- MLA AllGather start
    #   [MLA AllGather]
    #   dbo_mla_preprocess_hook(False)  <- MLA AllGather done
    #   [MLA compute]
    #   dbo_linear_row_hook(True)       <- o_proj ReduceScatter start
    #   [o_proj ReduceScatter]
    #   dbo_linear_row_hook(False)      <- o_proj ReduceScatter done (pass in A2)
    #   dbo_moe_prepare_hook(True)      <- MoE EP AllGather start (pass)
    #   [EP AllGather]
    #   dbo_moe_prepare_hook(False)     <- MoE EP AllGather done
    #   [Expert compute]
    #   dbo_moe_finalize_hook(True)     <- MoE EP ReduceScatter start
    #   [EP ReduceScatter]
    #   dbo_moe_finalize_hook(False)    <- MoE EP ReduceScatter done (pass)
    #
    # Overlap strategy (3 rules):
    #   1. ubatch0 attn post comm (o_proj RS + MoE Prepare AG) overlaps with ubatch1 attn compute
    #   2. ubatch1 attn post comm overlaps with ubatch0 MoE Expert compute
    #   3. ubatch0 MoE finalize comm (EP RS) overlaps with ubatch1 MoE Expert compute
    #
    # Note: o_proj RS and MoE Prepare AG are merged as a continuous comm block
    # (only LayerNorm + MoE routing between them, < 200μs).

    # MLA AllGather: overlap with prev layer's MoE finalize comm
    def dbo_mla_preprocess_hook(self, is_record):
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # o_proj ReduceScatter: record ATTN_POST before comm starts
    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)

    # Shared expert MLP column parallel (gate/up AllGather): wait for ATTN_POST
    def dbo_linear_column_hook(self, is_record):
        if not is_record:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)

    # MoE prepare (EP AllGather): wait for ATTN_POST after comm done
    def dbo_moe_prepare_hook(self, is_record):
        if not is_record:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)

    # MoE finalize (EP ReduceScatter): record ATTN_PRE before comm starts
    def dbo_moe_finalize_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)


class GlmMoeDsaAlltoallTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A3 (MLA+MoE model, same architecture as DeepSeek V2/V3):
    #
    # Per-layer hook call sequence:
    #   dbo_mla_preprocess_hook(True)   <- MLA AllGather start
    #   [MLA AllGather]
    #   dbo_mla_preprocess_hook(False)  <- MLA AllGather done
    #   [MLA compute]
    #   dbo_linear_row_hook(True)       <- o_proj comm start
    #   [o_proj comm]
    #   dbo_linear_row_hook(False)      <- o_proj comm done
    #   dbo_moe_prepare_hook(True)      <- Dispatch AlltoAll start
    #   [Dispatch AlltoAll + handle.wait()]
    #   dbo_moe_prepare_hook(False)     <- Dispatch AlltoAll done
    #   [Expert compute]
    #   dbo_moe_finalize_hook(True)     <- Combine AlltoAll start
    #   [Combine AlltoAll + handle.wait()]
    #   dbo_moe_finalize_hook(False)    <- Combine AlltoAll done
    #
    # Overlap strategy (5 rules):
    #   1. ubatch0 o_proj comm overlaps with ubatch1 MLA compute
    #   2. ubatch0 Dispatch AlltoAll overlaps with ubatch1 o_proj comm
    #   3. ubatch0 Finalize AlltoAll overlaps with ubatch1 Expert compute
    #   4. ubatch1 Dispatch AlltoAll overlaps with ubatch0 Expert compute
    #   5. ubatch1 Finalize AlltoAll overlaps with ubatch0 next layer MLA compute

    # MLA AllGather: overlap with prev layer's MoE finalize comm
    def dbo_mla_preprocess_hook(self, is_record):
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # o_proj comm: record ATTN_POST before, yield (no wait) after
    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST, wait=False)

    # Shared expert column hook: pass (no overlap needed in A3)
    def dbo_linear_column_hook(self, is_record):
        pass

    # MoE Dispatch AlltoAll: record MOE_DISPATCH before, wait MOE_DISPATCH after
    def dbo_moe_prepare_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.MOE_DISPATCH)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.MOE_DISPATCH)

    # MoE Combine AlltoAll: record ATTN_PRE before finalize comm
    def dbo_moe_finalize_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
