# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-quant V4.1 cache operations selected by the device adaptor."""

from vllm_ascend.ops.triton.a5_slot_mapping import build_a5_slot_mapping
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton

from .attention import build_smla_metadata, qsmla
from .indexer import run_a5_indexer
from .writers import write_attention_cache, write_index_cache


class DeepseekV41PackedCacheOps:
    """Operations sharing the packed mixed-quant cache and metadata ABI.

    Generic normalization, RoPE and mHC stay in DeviceOperator or their
    existing model paths. Selecting this class does not change those APIs.
    """

    build_a5_slot_mapping = staticmethod(build_a5_slot_mapping)
    build_window_indices = staticmethod(build_window_indices_triton)
    build_smla_metadata = staticmethod(build_smla_metadata)
    qsmla = staticmethod(qsmla)
    run_a5_indexer = staticmethod(run_a5_indexer)
    write_attention_cache = staticmethod(write_attention_cache)
    write_index_cache = staticmethod(write_index_cache)
