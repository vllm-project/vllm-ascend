from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker.ubatching import UBatchEventKey, dbo_record_current_stream, dbo_wait_current_stream_and_yield


class QwenDenseAllgatherTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A2 (Dense model):
    #
    # Per-layer hook call sequence:
    #   dbo_linear_column_hook(True)   <- QKV AllGather start
    #   dbo_linear_column_hook(False)  <- QKV AllGather done
    #   [Attention compute]
    #   dbo_linear_row_hook(True)      <- o_proj ReduceScatter start
    #   dbo_linear_row_hook(False)     <- o_proj ReduceScatter done
    #   dbo_linear_column_hook(True)   <- MLP gate/up AllGather start
    #   dbo_linear_column_hook(False)  <- MLP gate/up AllGather done
    #   [MLP compute]
    #   dbo_linear_row_hook(True)      <- MLP down ReduceScatter start
    #   dbo_linear_row_hook(False)     <- MLP down ReduceScatter done
    #
    # Overlap strategy (using DEFAULT event, no cross-layer dependency):
    #   1. ubatch0 attn post comm overlaps with ubatch1 attn compute
    #   2. ubatch1 attn post comm overlaps with ubatch0 MLP compute
    #   3. ubatch0 MLP post comm overlaps with ubatch1 MLP compute

    def dbo_linear_column_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.DEFAULT)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.DEFAULT)

    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.DEFAULT)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.DEFAULT)


class QwenDenseAlltoallTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A3 (Dense model):
    # Dense model has no MoE, so A3 strategy is identical to A2.
    # TP communication (AllGather/ReduceScatter) is the same on both A2 and A3.

    def dbo_linear_column_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.DEFAULT)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.DEFAULT)

    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.DEFAULT)
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.DEFAULT)
