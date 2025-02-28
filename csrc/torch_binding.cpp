#include "cache.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kPrivateUse1, &rotary_embedding);
}


REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
