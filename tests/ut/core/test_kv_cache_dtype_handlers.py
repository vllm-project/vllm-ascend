# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hif8 KV cache dtype registration must expose uint8 storage.

The whole HIF8 C8 cache layout contract rides on ``--kv-cache-dtype hif8``
resolving to ``torch.uint8`` (raw HIFLOAT8 bytes, one byte per latent
element, no per-token scale packed into the page). If the registration ever
falls back to an upstream mapping, the cache would silently allocate the
wrong dtype and the attention kernel would read garbage.
"""

import torch
from vllm.config.cache import KV_CACHE_DTYPES
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import KVQuantMode

# Importing registers every Ascend handler (including hif8) in the
# process-global tables the config layer consults.
import vllm_ascend.core.kv_cache_dtype_handlers  # noqa: F401


def test_hif8_kv_cache_dtype_resolves_to_uint8_storage():
    assert "hif8" in KV_CACHE_DTYPES
    assert STR_DTYPE_TO_TORCH_DTYPE["hif8"] is torch.uint8


def test_hif8_handler_reports_custom_quant_mode():
    # CUSTOM keeps the Ascend backend in full control of kernel dispatch:
    # no upstream generic fp8/per-token-head kernel path may claim it.
    from vllm_ascend.core.kv_cache_dtype_handlers import Hif8AscendHandler

    handler = Hif8AscendHandler()
    assert handler.name == "hif8"
    assert handler.torch_dtype() is torch.uint8
    assert handler.quant_mode() == KVQuantMode.CUSTOM
