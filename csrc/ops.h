#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

extern void rotary_embedding(
  turbo_types type,
  bool is_neox,
  void* stream,
  int64_t* positions,
  void* query_dst,
  void* key_dst,
  void* query,
  void* key,
  void* cos_sin_cache,
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int64_t dst_query_stride,
  const int64_t dst_key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size,
  const int64_t num_tokens,
  const int64_t loop_cnt,
  int aivNum);