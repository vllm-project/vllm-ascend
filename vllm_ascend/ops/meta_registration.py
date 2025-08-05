import torch
from torch.library import Library

lib = Library("_C", "IMPL")

def register_meta_if_necessary(ns:str, op_name: str, fn, overload: str = ""):
  if overload != "":
    op_name = op_name + "." + overload
  schema_to_find = ns + "::" + op_name
  meta_impl_list = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
  if schema_to_find in meta_impl_list:
    return 
  lib.impl(op_name, fn, "Meta")

def rotary_embedding_meta(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool):

  num_tokens = positions.numel()
  query_hidden_size = query.numel() / num_tokens
  key_hidden_size = key.numel() / num_tokens
  num_heads = query_hidden_size / head_size
  num_kv_heads = key_hidden_size / head_size

  query_dst = torch.empty_like(query).view(num_tokens, num_heads, head_size)
  key_dst = torch.empty_like(key).view(num_tokens, num_kv_heads, head_size)
  return query_dst, key_dst


def get_masked_input_and_mask_meta(
    input: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int):

  masked_input = torch.empty_like(input)
  mask = torch.empty_like(input).to(torch.bool)

  return masked_input, mask



register_meta_if_necessary("_C", "rotary_embedding", rotary_embedding_meta)
register_meta_if_necessary("_C", "get_masked_input_and_mask", get_masked_input_and_mask_meta)