from vllm.forward_context import get_forward_context

from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker.ubatching import UBatchEventKey, dbo_record_current_stream, dbo_wait_current_stream_and_yield


class QwenMoEAllgatherTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A2:
    # qwen3 do not use mla preprocess
    def dbo_mla_preprocess_hook(self, is_record):
        pass

    # post mla, overlap together with moe prepare comm/mlp comm
    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)

    # QKV layer comm, overlap together with prev layer's moe finalize comm
    def dbo_linear_column_hook(self, is_record):
        if is_record:
            if get_forward_context().dbo_first_layer_sync:
                dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
                get_forward_context().dbo_first_layer_sync = False
        else:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_PRE)

    # moe prepare comm, overlap together with post mla comm
    def dbo_moe_prepare_hook(self, is_record):
        if not is_record:
            dbo_wait_current_stream_and_yield(event=UBatchEventKey.ATTN_POST)

    # moe finalize comm, overlap together with mla preprocess comm
    def dbo_moe_finalize_hook(self, is_record):
        if is_record:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)


class QwenMoEAlltoallTemplate(UbatchOverlapBaseTemplate):
    pass
