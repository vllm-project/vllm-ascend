# SPDX-License-Identifier: Apache-2.0
"""Exercise the upstream indexer pipeline without an NPU or model load."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "variant", ["native", "prefill", "mixed", "pcp", "dsa", "dcp1", "large", "subclass", "reshape"]
)
def test_native_quantization_and_cache_pipeline_routing(variant):
    """Keep quantization exact and finish parallel-layout transforms before writing."""
    source = ROOT / "vllm_ascend/attention/indexer.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAIndexerBackend"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("forward", "forward_k", "_gather_cache_inputs")
    ]
    events = []
    rows = 13 if variant == "large" else 6
    quant = torch.arange(rows * 128).remainder(127).to(torch.int8).view(rows, 128)
    scales = torch.full((rows,), 0.12345, dtype=torch.float32)
    caches = (torch.zeros(256, 128, dtype=torch.int8), torch.zeros(256, 1, dtype=torch.float16))

    class Backend:
        k_hadamard = torch.eye(128, dtype=torch.bfloat16)

    class OtherBackend(Backend):
        pass

    def native_quant(*args, **kwargs):
        events.append("quant")
        return quant, scales

    def store(keys, stats, slots, key_cache, scale_cache):
        assert events[-1] == "gather_done" and stats.dtype == torch.float32
        events.append("fused")
        key_cache.index_copy_(0, slots, keys)
        scale_cache.index_copy_(0, slots, stats.to(torch.float16))

    namespace = {
        "torch": torch,
        "HAS_TRITON": True,
        "AscendSFAIndexerBackend": Backend,
        "torch_npu": SimpleNamespace(npu_dynamic_quant=native_quant),
        "rope_forward_triton_siso": lambda x, *args, **kwargs: x,
        "can_fuse_store": lambda k, s, slots, n: 0 < n <= 12,
        "store_indexer_key_scale": store,
        "INDEXER_K_CACHE_SLOT": 0,
        "INDEXER_SCALE_CACHE_SLOT": 1,
        "_gather_prefill_cache_inputs": lambda tensors, slots, count: (tensors, slots),
        "all_gather_async": lambda value, *args, **kwargs: (value, None),
        "get_tp_group": lambda: None,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + methods, type_ignores=[]), str(source), "exec"), namespace)
    state = OtherBackend() if variant == "subclass" else Backend()
    state._dcp_size = 1 if variant == "dcp1" else 8
    state._pcp_active, state._dsa_cp_active = variant == "pcp", variant == "dsa"
    state.enable_sparse_li_c8 = True
    state._use_c8_reshape_optim = lambda: variant == "reshape"
    state.k_cache = SimpleNamespace(kv_cache=caches)
    state.wk_weights_proj = lambda x: (torch.zeros(rows, 160, dtype=torch.bfloat16), None)
    state.k_norm = lambda x: x
    state.head_dim, state.qk_rope_head_dim, state.is_rope_neox_style = 128, 64, False
    state.c8_k_cache_dtype, state.c8_k_scale_cache_dtype = torch.int8, torch.float16
    state.forward_k = MethodType(namespace["forward_k"], state)

    def gather(*args):
        result = namespace["_gather_cache_inputs"](state, *args)
        events.append("gather_done")
        return result

    def legacy(keys, stats, slots, **kwargs):
        assert events[-1] == "gather_done" and stats.dtype == torch.float16
        events.append("legacy")
        caches[0].index_copy_(0, slots, keys)
        caches[1].index_copy_(0, slots, stats)

    state._gather_cache_inputs, state.write_cache = gather, legacy
    decode = 0 if variant == "prefill" else (1 if variant == "mixed" else rows)
    metadata = SimpleNamespace(
        num_actual_tokens=rows, num_decode_tokens=decode, slot_mapping=torch.arange(rows, dtype=torch.int64) + 5
    )
    x = torch.zeros(rows, 4)
    result = namespace["forward"](
        state, x, None, torch.ones(rows, 64), torch.zeros(rows, 64), x, metadata, compute_topk=False
    )
    assert result is None and events == ["quant", "gather_done", "fused" if variant == "native" else "legacy"]
    assert torch.equal(caches[0][metadata.slot_mapping], quant)
    assert torch.equal(caches[1][metadata.slot_mapping], scales.to(torch.float16).view(rows, 1))
