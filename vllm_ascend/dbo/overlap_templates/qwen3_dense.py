from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.worker.ubatching import dbo_record_current_stream, dbo_wait_current_stream_and_yield


class QwenDenseAllgatherTemplate(UbatchOverlapBaseTemplate):
    # DBO overlap policy for A2:

    # post attn / post mlp comm
    def dbo_linear_row_hook(self, is_record):
        if is_record:
            dbo_record_current_stream()
        else:
            dbo_wait_current_stream_and_yield()

    # QKV layer comm and MLP layer comm
    def dbo_linear_column_hook(self, is_record):
        if is_record:
            dbo_record_current_stream()
        else:
            dbo_wait_current_stream_and_yield()


class QwenDenseAlltoallTemplate(UbatchOverlapBaseTemplate):
    pass
