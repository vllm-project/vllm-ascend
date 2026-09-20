# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 A5 production operator adapters."""

from .attention import build_window_indices, qsmla
from .compressor import compressor_v2
from .hyper_connection import hc_post, hc_pre
from .indexer import run_a5_indexer
from .quantization import mxfp4_quantize_e8m0
from .rotary import apply_partial_rotary_inplace
from .writers import write_attention_cache, write_index_cache

__all__ = [
    "apply_partial_rotary_inplace",
    "build_window_indices",
    "compressor_v2",
    "hc_post",
    "hc_pre",
    "mxfp4_quantize_e8m0",
    "qsmla",
    "run_a5_indexer",
    "write_attention_cache",
    "write_index_cache",
]
