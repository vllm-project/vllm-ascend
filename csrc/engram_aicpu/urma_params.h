// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
struct EngramUrmaParams {
    uint64_t metadata, state, ids, codes;
    int64_t tokens, width, ids_stride, head_start, local_heads, vocab_start, vocab_end, id_bytes;
    uint64_t endpoint_chip, endpoint_die, close;
};
