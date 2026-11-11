# SPDX-License-Identifier: Apache-2.0
"""Relative cache names eligible for temporal offload buffer reuse."""

REUSE_CONFIG: dict[str, bool] = {
    "linear_attn": True,
    "self_attn.attn": True,
    "self_attn.indexer.raw_key_cache": False,
    "self_attn.indexer.compressed_key_cache": False,
}
