# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce paired-SFA/DSpark spec geometry without constructing model weights.

Run in the approved runtime container, e.g.::

    python tests/ut/core/check_paired_sfa_dspark_geometry.py --draft-cache-dtype auto

The default geometry is the delivered K3.1 TP16 checkpoint and five-layer
DSpark draft after platform Mamba alignment to 768 tokens. Dtype is explicit:
select the value assigned by the installed draft-loading context. This uses
the real MLA spec method and worker conversion; only attention-layer discovery
and non-geometric global feature switches are supplied by the harness.
"""

import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--draft-cache-dtype", choices=("auto", "bfloat16", "int8"), required=True)
    args = parser.parse_args()

    import torch
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    from vllm_ascend.models.kimi_k3 import paired_sfa_cache
    from vllm_ascend.worker.v2 import attn_utils

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=16),
        cache_config=SimpleNamespace(
            block_size=args.block_size, cache_dtype="int8", mamba_page_size_padded=args.block_size * 656
        ),
        attention_config=SimpleNamespace(indexer_kv_dtype="int8"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                model_type="kimi_linear", kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128
            ),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    ascend_config = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        is_sparse_li_c8_layer=lambda name: True,
        is_sparse_li_c4_layer=lambda name: False,
    )

    def make_mla(name, cache_dtype, *, sparse):
        layer = attn_utils.MLAAttention.__new__(attn_utils.MLAAttention)
        torch.nn.Module.__init__(layer)
        layer.layer_name = name
        layer.is_draft_layer = not sparse
        layer.num_heads = 4
        layer.num_kv_heads = 1
        layer.head_size = 576
        layer.kv_lora_rank = 512
        layer.qk_rope_head_dim = 64
        layer.kv_cache_dtype = cache_dtype
        layer.sliding_window = None
        layer.indexer = object() if sparse else None
        # The delivered draft has neither dflash_config.causal nor
        # full_attention_causal, so K3 DSpark enables its non-causal window.
        layer.non_causal_multi_token_decode = not sparse
        layer.impl = SimpleNamespace(
            fa_quant_layer=False,
            dtype=torch.bfloat16,
            enable_sparse_sfa_c8=sparse,
            enable_sparse_sfa_turboquant=False,
        )
        return layer

    target = "language_model.model.layers.3.self_attn.attn"
    indexer_name = "language_model.model.layers.3.self_attn.indexer.k_cache"
    indexer = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
    torch.nn.Module.__init__(indexer)
    indexer.is_draft_layer = False
    indexer.get_kv_cache_spec = lambda cfg: MLAAttentionSpec(
        block_size=cfg.cache_config.block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        cache_dtype_str="int8",
    )
    mamba = MambaSpec(
        block_size=args.block_size,
        shapes=((3, 2304), (6, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
        page_size_padded=config.cache_config.mamba_page_size_padded,
    )
    layers = {
        target: make_mla(target, "int8", sparse=True),
        indexer_name: indexer,
        "language_model.model.layers.0.linear_attn": SimpleNamespace(
            get_kv_cache_spec=lambda cfg: mamba, is_draft_layer=False
        ),
    }
    for index in range(5):
        name = f"model.layers.{93 + index}.self_attn.attn"
        layers[name] = make_mla(name, args.draft_cache_dtype, sparse=False)

    with (
        patch.object(attn_utils, "get_layers_from_vllm_config", return_value=layers),
        patch.object(attn_utils, "enable_sfa_dcp_replicated_indexer", return_value=False),
        patch.object(attn_utils, "enable_sfa", return_value=True),
        patch.object(attn_utils, "get_ascend_config", return_value=ascend_config),
    ):
        captured = attn_utils.get_kv_cache_spec(config)

    assert captured, "Worker conversion did not produce cache specs"
    summary = {
        name: {
            "type": type(spec).__name__,
            "block_size": spec.block_size,
            "head_size": getattr(spec, "head_size", None),
            "dtype": str(getattr(spec, "dtype", None)),
            "raw_page_bytes": getattr(spec, "real_page_size_bytes", spec.page_size_bytes),
            "page_bytes": spec.page_size_bytes,
            "packed_sfa": getattr(spec, "cache_sparse_sfa_c8", False),
            "non_causal": getattr(spec, "non_causal_multi_token_decode", False),
        }
        for name, spec in captured.items()
    }
    target_only = {name: spec for name, spec in captured.items() if not name.startswith("model.layers.")}
    groups = paired_sfa_cache.paired_sfa_cache_groups(config, target_only)
    print(
        json.dumps(
            {
                "draft_cache_dtype": args.draft_cache_dtype,
                "specs": summary,
                "target_only_plan": "passed",
                "target_only_group_count": len(groups),
                "target_only_pool_bytes": paired_sfa_cache.paired_sfa_pool_bytes(groups),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
