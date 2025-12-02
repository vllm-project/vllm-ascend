from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

def warp(func):
    def abc(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if get_forward_context().sp_enabled:
            deepstack_input_embeds.tensors = {k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()}
        return deepstack_input_embeds
    
    return abc

Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = warp(Qwen3VLForConditionalGeneration._get_deepstack_input_embeds)