from .context import (get_multistream_layer_context,
                      get_multistream_microbatch_context)


# vllm v1 use get_forward_context to get the attn_metadata,
# we can use this decorator to update the attn metadata
def set_multistream_support():

    def decorator(func):

        def wrapper():
            context = func()
            layer_index, ms_metadata, attn_metadata = get_multistream_layer_context(
            )
            micro_batch_num = get_multistream_microbatch_context()
            if layer_index != -1 and micro_batch_num != -1:
                context.attn_metadata = attn_metadata[micro_batch_num]
            return context

        return wrapper

    return decorator
