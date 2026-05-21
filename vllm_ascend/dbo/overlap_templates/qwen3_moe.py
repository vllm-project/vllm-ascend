from vllm.forward_context import get_forward_context

from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker.ubatching import UBatchEventKey, dbo_record_current_stream, dbo_wait_current_stream_and_yield


class QwenMoEAllgatherTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A2 (MoE model without MLA):
    #
    # Per-layer hook call sequence:
    #   dbo_linear_column_hook(True)   <- QKV AllGather start
    #   dbo_linear_column_hook(False)  <- QKV AllGather done
    #   [Attention compute]
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
    # Overlap strategy:
    #   1. ubatch0 attn post comm (o_proj RS) overlaps with ubatch1 attn compute
    #   2. ubatch1 attn post comm overlaps with ubatch0 MoE compute
    #   3. ubatch0 MoE finalize comm overlaps with ubatch1 MoE compute

    # QKV AllGather: overlap with prev layer's MoE finalize comm
    def dbo_linear_column_hook(self, is_record):
        if is_record:
            # First layer has no previous MoE finalize event to wait on,
            # so we proactively record ATTN_PRE to unblock ubatch1's wait.
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # o_proj ReduceScatter: record ATTN_POST before comm starts
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


class QwenMoEAlltoallTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A3 (MoE model without MLA):
    #
    # Per-layer hook call sequence:
    #   dbo_linear_column_hook(True)   <- QKV AllGather start
    #   dbo_linear_column_hook(False)  <- QKV AllGather done
    #   [Attention compute]
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
    # Overlap strategy:
    #   1. ubatch0 o_proj comm overlaps with ubatch1 attn compute
    #   2. ubatch0 Dispatch AlltoAll overlaps with ubatch1 o_proj comm
    #   3. ubatch0 Finalize AlltoAll overlaps with ubatch1 Expert compute
    #   4. ubatch1 Dispatch AlltoAll overlaps with ubatch0 Expert compute
    #   5. ubatch1 Finalize AlltoAll overlaps with ubatch0 next layer attn compute

    # QKV AllGather: overlap with prev layer's MoE finalize comm
    def dbo_linear_column_hook(self, is_record):
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
