from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
) -> BatchDescriptor:
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

    # FULL mode should not be treated as uniform decode
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)

    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
