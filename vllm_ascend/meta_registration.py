import torch
from torch.library import Library

# This file provides a template and registration utilities for writing "meta" implementations
# of custom operators in Python for the vllm_ascend project.
#
# We offer two ways to implement meta implementations for custom ops:
#   1. Python meta implementation (as shown in this file): Write a Python function that
#      takes the same arguments as your operator and returns empty tensors with the correct
#      shapes and dtypes. This is useful for rapid prototyping and for ops that are only
#      used in Python.
#   2. C++ meta implementation: You can also implement the meta function in C++ for better
#      performance or to match the C++ op logic more closely. See `torch_binding_meta.cpp`
#      for examples of C++ meta implementations and how to register them.
#
# Both approaches enable tracing, export, and shape inference in PyTorch and vLLM, which
# is essential for supporting `torch.compile` and aclgraph.

# How to add a new meta implementation in Python:
# -------------------------------------
# 1. Write a Python function that takes the same arguments as your operator, and returns
#    empty tensors (using torch.empty_like, torch.empty, etc.) with the correct shapes and dtypes.
#    Do NOT perform any real computation or allocate device memory.
#
# 2. Register your meta function using `register_meta_if_necessary`, providing:
#    - The namespace (usually "_C_ascend" for custom ops)
#    - The operator name (as registered in C++)
#    - The Python meta function
#    - (Optional) The overload name, if your op has overloads
#
# 3. The registration utility will check if a meta implementation already exists for your op,
#    and only register if necessary. This avoids duplicate registrations.
#
# 4. Example meta implementations are provided below for rotary_embedding and get_masked_input_and_mask.
#
# 5. When developing new custom ops, always provide a meta implementation to enable tracing,
#    export, and shape inference in PyTorch and vLLM to enable the capture of `torch.compile`
#    and aclgraph.
#
# For more details, see: https://pytorch.org/docs/stable/notes/extending.html#meta-tensors

lib = Library("_C_ascend", "IMPL")


def register_meta_if_necessary(ns: str, op_name: str, fn, overload: str = ""):
    if overload != "":
        op_name = op_name + "." + overload
    schema_to_find = ns + "::" + op_name
    meta_impl_list = torch._C._dispatch_get_registrations_for_dispatch_key(
        "Meta")
    if schema_to_find in meta_impl_list:
        return
    lib.impl(op_name, fn, "Meta")


def rotary_embedding_meta(positions: torch.Tensor, query: torch.Tensor,
                          key: torch.Tensor, head_size: int,
                          cos_sin_cache: torch.Tensor, is_neox: bool):

    num_tokens = positions.numel()
    query_hidden_size = query.numel() // num_tokens
    key_hidden_size = key.numel() // num_tokens
    num_heads = query_hidden_size // head_size
    num_kv_heads = key_hidden_size // head_size

    query_dst = torch.empty_like(query).view(num_tokens, num_heads, head_size)
    key_dst = torch.empty_like(key).view(num_tokens, num_kv_heads, head_size)
    return query_dst, key_dst


def get_masked_input_and_mask_meta(input: torch.Tensor,
                                   org_vocab_start_index: int,
                                   org_vocab_end_index: int,
                                   num_org_vocab_padding: int,
                                   added_vocab_start_index: int,
                                   added_vocab_end_index: int):

    masked_input = torch.empty_like(input)
    mask = torch.empty_like(input).to(torch.bool)

    return masked_input, mask


def bgmv_expand_meta(x: torch.Tensor, weight: torch.Tensor,
                     indices: torch.Tensor, y: torch.Tensor, slice_offset: int,
                     slice_size: int):

    y_out = torch.empty_like(y)
    return y_out


def sgmv_expand_meta(x: torch.Tensor, weight: torch.Tensor,
                     lora_indices: torch.Tensor, seq_len: torch.Tensor,
                     y: torch.Tensor, slice_offset: int, slice_size: int):

    y_out = torch.empty_like(y)
    return y_out


register_meta_if_necessary("_C_ascend", "rotary_embedding",
                           rotary_embedding_meta)
register_meta_if_necessary("_C_ascend", "get_masked_input_and_mask",
                           get_masked_input_and_mask_meta)
register_meta_if_necessary("_C_ascend", "bgmv_expand", bgmv_expand_meta)
register_meta_if_necessary("_C_ascend", "sgmv_expand", sgmv_expand_meta)
