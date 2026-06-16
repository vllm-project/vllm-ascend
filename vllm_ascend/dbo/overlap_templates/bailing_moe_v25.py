from vllm.forward_context import get_forward_context

from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker.ubatching import UBatchEventKey, dbo_record_current_stream, dbo_wait_current_stream_and_yield


class BailingMoEV25AllgatherTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A2 (MLA+MoE hybrid attention model):
    #
    # BailingMoeV25 is a hybrid attention model:
    #   - Full attention layers use MLA (MultiHeadLatentAttention)
    #   - Linear attention layers use standard QKV (QKVParallelLinear)
    #   - Both layer types share the same MoE block (SharedFusedMoE)
    #
    # Full attention layer (MLA) hook call sequence:
    #   dbo_mla_preprocess_hook(True)  <- MLA AllGather start
    #   [MLA AllGather]
    #   dbo_mla_preprocess_hook(False) <- MLA AllGather done
    #   [MLA compute]
    #   dbo_linear_row_hook(True)      <- o_proj ReduceScatter start
    #   [o_proj ReduceScatter]
    #   dbo_moe_prepare_hook(True)     <- MoE EP AllGather start
    #   [EP AllGather]
    #   dbo_moe_prepare_hook(False)    <- MoE EP AllGather done
    #   [Expert compute]
    #   dbo_moe_finalize_hook(True)    <- MoE EP ReduceScatter start
    #   [EP ReduceScatter]
    #   dbo_moe_finalize_hook(False)   <- MoE EP ReduceScatter done
    #
    # Linear attention layer hook call sequence:
    #   dbo_linear_column_hook(True)   <- QKV AllGather start
    #   dbo_linear_column_hook(False)  <- QKV AllGather done
    #   [Linear Attention compute]
    #   dbo_linear_row_hook(True)      <- dense ReduceScatter start
    #   [dense ReduceScatter]
    #   dbo_moe_prepare_hook(True)     <- MoE EP AllGather start
    #   [EP AllGather]
    #   dbo_moe_prepare_hook(False)    <- MoE EP AllGather done
    #   [Expert compute]
    #   dbo_moe_finalize_hook(True)    <- MoE EP ReduceScatter start
    #   [EP ReduceScatter]
    #   dbo_moe_finalize_hook(False)   <- MoE EP ReduceScatter done
    #
    # Overlap strategy (same for both layer types):
    #   1. ubatch0 attn post comm (o_proj/dense RS) overlaps with ubatch1 attn compute
    #   2. ubatch1 attn post comm overlaps with ubatch0 MoE compute
    #   3. ubatch0 MoE finalize comm overlaps with ubatch1 MoE compute

    # MLA AllGather (full attention layers): overlap with prev layer's MoE finalize comm
    def dbo_mla_preprocess_hook(self, is_record):
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # QKV AllGather (linear attention layers): overlap with prev layer's MoE finalize comm
    def dbo_linear_column_hook(self, is_record):
        if is_record:
            # First layer has no previous MoE finalize event to wait on,
            # so we proactively record ATTN_PRE to unblock ubatch1's wait.
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # o_proj/dense ReduceScatter: record ATTN_POST before comm starts
    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)

    # MoE prepare (EP AllGather): wait for ATTN_POST after comm done
    def dbo_moe_prepare_hook(self, is_record):
        if not is_record:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)

    # MoE finalize (EP ReduceScatter): record ATTN_PRE before comm starts
    def dbo_moe_finalize_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)


class BailingMoEV25AlltoallTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A3 (MLA+MoE hybrid attention model):
    #
    # BailingMoeV25 is a hybrid attention model:
    #   - Full attention layers use MLA (MultiHeadLatentAttention)
    #   - Linear attention layers use standard QKV (QKVParallelLinear)
    #   - Both layer types share the same MoE block (SharedFusedMoE)
    #
    # Full attention layer (MLA) hook call sequence:
    #   dbo_mla_preprocess_hook(True)  <- MLA AllGather start
    #   [MLA AllGather]
    #   dbo_mla_preprocess_hook(False) <- MLA AllGather done
    #   [MLA compute]
    #   dbo_linear_row_hook(True)      <- o_proj comm start
    #   [o_proj comm]
    #   dbo_linear_row_hook(False)     <- o_proj comm done
    #   dbo_moe_prepare_hook(True)     <- Dispatch AlltoAll start
    #   [Dispatch AlltoAll + handle.wait()]
    #   dbo_moe_prepare_hook(False)    <- Dispatch AlltoAll done
    #   [Expert compute]
    #   dbo_moe_finalize_hook(True)    <- Combine AlltoAll start
    #   [Combine AlltoAll + handle.wait()]
    #   dbo_moe_finalize_hook(False)   <- Combine AlltoAll done
    #
    # Linear attention layer hook call sequence:
    #   dbo_linear_column_hook(True)   <- QKV AllGather start
    #   dbo_linear_column_hook(False)  <- QKV AllGather done
    #   [Linear Attention compute]
    #   dbo_linear_row_hook(True)      <- dense comm start
    #   [dense comm]
    #   dbo_linear_row_hook(False)     <- dense comm done
    #   dbo_moe_prepare_hook(True)     <- Dispatch AlltoAll start
    #   [Dispatch AlltoAll + handle.wait()]
    #   dbo_moe_prepare_hook(False)    <- Dispatch AlltoAll done
    #   [Expert compute]
    #   dbo_moe_finalize_hook(True)    <- Combine AlltoAll start
    #   [Combine AlltoAll + handle.wait()]
    #   dbo_moe_finalize_hook(False)   <- Combine AlltoAll done
    #
    # Overlap strategy (same for both layer types):
    #   1. ubatch0 o_proj/dense comm overlaps with ubatch1 attn compute
    #   2. ubatch0 Dispatch AlltoAll overlaps with ubatch1 o_proj/dense comm
    #   3. ubatch0 Finalize AlltoAll overlaps with ubatch1 Expert compute
    #   4. ubatch1 Dispatch AlltoAll overlaps with ubatch0 Expert compute
    #   5. ubatch1 Finalize AlltoAll overlaps with ubatch0 next layer attn compute
    #
    # Timeline (full attention / MLA layer):
    #   ubatch0: [MLA_AG]-[MLA]-[o_proj]-record(ATTN_POST)-[o_proj_comm]
    #            -wait(ATTN_POST,wait=False)-record(MOE_DISPATCH)-[Dispatch_A2A]
    #            -wait(MOE_DISPATCH,wait=False)-[Expert]-record(ATTN_PRE)-[Finalize_A2A]
    #   ubatch1: wait(ATTN_PRE)-[MLA_AG]-[MLA]-[o_proj]-record(ATTN_POST)-[o_proj_comm]
    #            -wait(ATTN_POST,wait=False)-wait(MOE_DISPATCH)-[Dispatch_A2A]
    #            -[Expert]-record(ATTN_PRE)-[Finalize_A2A]
    #
    # Timeline (linear attention layer):
    #   ubatch0: [QKV_AG]-[LinAttn]-[dense]-record(ATTN_POST)-[dense_comm]
    #            -wait(ATTN_POST,wait=False)-record(MOE_DISPATCH)-[Dispatch_A2A]
    #            -wait(MOE_DISPATCH,wait=False)-[Expert]-record(ATTN_PRE)-[Finalize_A2A]
    #   ubatch1: wait(ATTN_PRE)-[QKV_AG]-[LinAttn]-[dense]-record(ATTN_POST)-[dense_comm]
    #            -wait(ATTN_POST,wait=False)-wait(MOE_DISPATCH)-[Dispatch_A2A]
    #            -[Expert]-record(ATTN_PRE)-[Finalize_A2A]

    # MLA AllGather (full attention layers): overlap with prev layer's MoE finalize comm
    def dbo_mla_preprocess_hook(self, is_record):
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # QKV AllGather (linear attention layers): overlap with prev layer's MoE finalize comm
    def dbo_linear_column_hook(self, is_record):
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # o_proj/dense comm: record ATTN_POST before, yield (no wait) after
    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST, wait=False)

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
