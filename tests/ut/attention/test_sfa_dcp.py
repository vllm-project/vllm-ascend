# SPDX-License-Identifier: Apache-2.0
"""CPU routing, dependency, integer-mapping and specialization contracts for SFA."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
TRITON_ROOT = ROOT / "vllm_ascend/ops/triton"


def load_function(path, name, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "tokens,ranks,scatter,supported,pcp",
    [
        (1, 8, 1, True, False),
        (6, 8, 1, True, False),
        (12, 8, 1, True, False),
        (13, 8, 1, True, False),
        (48, 8, 1, True, False),
        (192, 8, 1, True, False),
        (193, 8, 1, True, False),
        (6, 2, 1, True, False),
        (6, 4, 1, True, False),
        (6, 16, 1, True, False),
        (8, 8, 0, True, False),
        (6, 8, 1, True, True),
        (6, 1, 1, False, True),
        (6, 1, 1, False, False),
    ],
)
@pytest.mark.parametrize("a5", [False, True])
def test_registered_op_keeps_upstream_fallback_and_output_dtype(tokens, ranks, scatter, supported, pcp, a5):
    # Device generation must not change the tensor/mode eligibility contract.
    calls = []
    group = object()
    output = torch.zeros(tokens, 64, 512, dtype=torch.bfloat16)
    lse = torch.zeros(tokens, 64, 1)
    shape = (tokens, 64 // ranks, 512) if scatter else (tokens // ranks, 64, 512)

    def exchange(out, stats, selected_group):
        assert selected_group is group and out is output and stats is lse
        calls.append("exchange")
        return torch.ones(shape, dtype=torch.float32)

    def transfer(recv, send, *, group):
        calls.append("upstream_collective")

    namespace = {
        "torch": torch,
        "dist": SimpleNamespace(all_to_all_single=transfer),
        "is_950": lambda: a5,
        "can_exchange": lambda *_: supported,
        "exchange": exchange,
        "pack_dcp_output_lse": lambda *_, **kwargs: torch.empty(8, 16),
        "fused_dcp_lse_combine": lambda *_, **kwargs: torch.ones(shape, dtype=output.dtype),
    }
    run = load_function("vllm_ascend/ops/triton/dcp/dcp_a2a.py", "dcp_a2a_fused_combine", namespace)

    def pcp_gather(value, dim):
        assert dim == 0
        calls.append("pcp_gather")
        return value

    pcp_group = SimpleNamespace(all_gather=pcp_gather) if pcp else None
    result = run(output, lse, ranks, scatter, group, pcp_group)
    assert result.dtype == output.dtype and result.shape == shape
    if scatter == 1 and supported and not pcp:
        assert calls == ["exchange"]
    else:
        assert calls == (["upstream_collective"] if ranks > 1 else []) + (["pcp_gather"] if pcp else [])


@pytest.mark.parametrize(
    "tokens,expected", [(0, False), (1, True), (6, True), (12, True), (13, True), (48, True), (192, True), (193, True)]
)
def test_exchange_gate(tokens, expected):
    namespace = {"torch": torch}
    run = load_function("vllm_ascend/ops/triton/dcp/sfa_dcp_exchange.py", "can_exchange", namespace)
    device = SimpleNamespace(type="npu")
    output = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=3, shape=(tokens, 64, 512))
    lse = SimpleNamespace(device=device, dtype=torch.float32, shape=(tokens, 64, 1))
    assert run(output, lse, 8) is expected


@pytest.mark.parametrize("ranks", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("local_heads", [1, 3, 8, 16, 32, 33, 64])
@pytest.mark.parametrize(
    "head_dim,vector_compatible",
    [(0, False), (1, False), (2, True), (128, True), (496, False), (512, True), (513, False), (1024, True)],
)
def test_exchange_gate_derives_head_shards_from_group(ranks, local_heads, head_dim, vector_compatible):
    namespace = {"torch": torch}
    run = load_function("vllm_ascend/ops/triton/dcp/sfa_dcp_exchange.py", "can_exchange", namespace)
    device = SimpleNamespace(type="npu")
    heads = ranks * local_heads
    output = SimpleNamespace(device=device, dtype=torch.bfloat16, ndim=3, shape=(13, heads, head_dim))
    lse = SimpleNamespace(device=device, dtype=torch.float32, shape=(13, heads, 1))
    assert run(output, lse, ranks) is (vector_compatible and local_heads & (local_heads - 1) == 0)
    output.shape = (13, heads + 1, head_dim)
    assert not run(output, lse, ranks)


@pytest.mark.parametrize(
    "tokens,heads,head_dim,ranks,expected",
    [
        (192, 64, 512, 8, True),
        (512, 64, 512, 2, True),
        (513, 64, 512, 2, True),
        (1024, 64, 512, 4, True),
        (1025, 64, 512, 4, True),
        (256, 128, 512, 2, True),
        (257, 128, 512, 2, True),
        (1, 128, 512, 2, True),
        (1, 128, 1024, 2, True),
    ],
)
def test_exchange_gate_accepts_tiled_sizes(tokens, heads, head_dim, ranks, expected):
    run = load_function("vllm_ascend/ops/triton/dcp/sfa_dcp_exchange.py", "can_exchange", {"torch": torch})
    output = torch.empty((tokens, heads, head_dim), dtype=torch.bfloat16, device="meta")
    lse = torch.empty((tokens, heads, 1), device="meta")
    assert run(output, lse, ranks) is expected


def test_custom_op_fake_retains_shape_dtype_and_does_not_collect():
    run = load_function("vllm_ascend/ops/triton/dcp/dcp_a2a.py", "dcp_a2a_fused_fake", {"torch": torch})
    output = torch.empty(6, 64, 512, dtype=torch.bfloat16)
    result = run(output, torch.empty(6, 64, 1), 8, 1, "unused")
    assert result.shape == (6, 8, 512) and result.dtype == output.dtype


@pytest.mark.parametrize("supported", [False, True])
def test_missing_scatter_group_raises_without_communication(supported):
    def forbidden(*args, **kwargs):
        raise AssertionError("communication attempted without a group")

    namespace = {
        "torch": torch,
        "can_exchange": lambda *_: supported,
        "exchange": forbidden,
        "dist": SimpleNamespace(all_to_all_single=forbidden),
        "pack_dcp_output_lse": lambda *args, **kwargs: torch.empty(8, 16),
    }
    run = load_function("vllm_ascend/ops/triton/dcp/dcp_a2a.py", "dcp_a2a_fused_combine", namespace)
    with pytest.raises(ValueError, match="explicit All2All group"):
        run(torch.empty(6, 64, 512, dtype=torch.bfloat16), torch.empty(6, 64, 1), 8, 1, None)


@pytest.mark.parametrize("defer", [False, True])
def test_new_main_merge_modes_never_take_output_only_fast_path(defer):
    """Sparse-offload callers need packed contributions or the merged LSE."""
    packed = torch.empty(8, 8, 6, 516)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("output-only fast path loses upstream merge state")

    def combine(recv, head_dim, *, scatter_dim, return_lse):
        assert recv.shape == packed.shape and head_dim == 512 and scatter_dim == 1
        assert return_lse
        calls.append("merge_lse")
        return torch.empty(6, 8, 513)

    namespace = {
        "torch": torch,
        "can_exchange": lambda *_: True,
        "exchange": forbidden,
        "dist": SimpleNamespace(all_to_all_single=lambda *args, **kwargs: calls.append("collective")),
        "pack_dcp_output_lse": lambda *args, **kwargs: packed,
        "fused_dcp_lse_combine": combine,
    }
    run = load_function("vllm_ascend/ops/triton/dcp/dcp_a2a.py", "dcp_a2a_fused_combine", namespace)
    result = run(
        torch.empty(6, 64, 512),
        torch.empty(6, 64, 1),
        8,
        1,
        object(),
        return_lse=not defer,
        defer_combine=defer,
    )
    assert result.shape == (packed.shape if defer else (6, 8, 513))
    assert calls == (["collective"] if defer else ["collective", "merge_lse"])


def test_registered_operator_forwards_group_and_fake_keeps_shape():
    group = SimpleNamespace(world_size=8, device_group=object())
    calls = []

    def combine(*args, **kwargs):
        calls.append(args[2])
        return args[0]

    namespace = {"torch": torch, "_groups": {"dcp": lambda: group}, "dcp_a2a_fused_combine": combine}
    run = load_function("vllm_ascend/ops/triton/dcp/dcp_a2a.py", "dcp_a2a_fused", namespace)
    fake = load_function("vllm_ascend/ops/triton/dcp/dcp_a2a.py", "dcp_a2a_fused_fake", {"torch": torch})
    output = torch.empty(48, 64, 512, dtype=torch.bfloat16)
    lse = torch.empty(48, 64, 1)
    for ranks in (2, 4, 8):
        group.world_size = ranks
        run(output, lse, ranks, 1, "dcp")
        result = fake(output, lse, ranks, 1, "dcp")
        assert result.shape == (48, 64 // ranks, 512) and result.dtype == output.dtype
    assert calls == [2, 4, 8]


def test_attention_forwards_its_group_without_scheduler_config():
    source = ROOT / "vllm_ascend/attention/context_parallel/sfa_cp.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AscendSFADCPImpl"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_merge_dcp_outputs")
    calls = []

    def invoke(*args, **kwargs):
        calls.append((args[2], args[4]))
        return args[0]

    namespace: dict[str, Any] = {
        "torch": SimpleNamespace(ops=SimpleNamespace(vllm=SimpleNamespace(dcp_a2a_fused=invoke))),
        "get_pcp_group": lambda: SimpleNamespace(world_size=1),
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), namespace)
    state = SimpleNamespace(
        dcp_size=8,
        dcp_group=SimpleNamespace(unique_name="dcp"),
    )
    namespace["_merge_dcp_outputs"](state, object(), object())
    assert calls == [(8, "dcp")]


@pytest.mark.parametrize(
    "tokens,decode,prepared,pcp_size",
    [
        (1, True, True, 1),
        (6, True, True, 1),
        (12, True, True, 1),
        (13, True, True, 1),
        (6, False, False, 1),
        (6, True, False, 2),
        (6, True, False, 8),
    ],
)
@pytest.mark.parametrize("keep_view", [False, True])
@pytest.mark.parametrize("layout", ["native", "fp16", "fp32", "other_width", "non_power_two"])
@pytest.mark.parametrize("ranks", [2, 4, 8, 16])
def test_query_gather_waits_before_final_unpack_and_preserves_fallback(
    tokens, decode, prepared, pcp_size, ranks, layout, keep_view
):
    if pcp_size > ranks:
        pytest.skip("PCP must divide the DCP group")
    source = ROOT / "vllm_ascend/attention/context_parallel/sfa_cp.py"
    tree = ast.parse(source.read_text())
    context_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DCPGatherContext")
    impl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFADCPImpl")
    methods = [
        n
        for n in impl.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_start_dcp_query_gather", "_finish_dcp_gather", "_record_query_gather_context")
    ]
    for method in methods:
        method.decorator_list = []
    events = []
    group = object()
    tp_group = object()
    replicas = ranks // pcp_size
    handle = SimpleNamespace(wait=lambda: events.append("wait"))

    def gather(value, selected_group):
        assert selected_group is (tp_group if pcp_size > 1 else group)
        events.append("gather")
        return torch.cat([value] * replicas, dim=0), handle

    def prep(qn, qr):
        events.append("prepare")
        return torch.cat((qn, qr), dim=-1).permute(1, 0, 2).contiguous()

    def unpack(value, split_sizes):
        assert events[-1] == "wait"
        events.append("unpack")
        return tuple(part.contiguous() for part in value.permute(1, 0, 2).split(split_sizes, dim=-1))

    namespace = {
        "torch": torch,
        "M": object,
        "AscendSFADCPMetadata": SimpleNamespace,
        "NamedTuple": NamedTuple,
        "all_gather_async": gather,
        "get_pcp_group": lambda: SimpleNamespace(world_size=pcp_size),
        "get_tp_group": lambda: tp_group,
        "prepare_query_head_major": prep,
        "unpack_query": unpack,
    }
    exec(compile(ast.Module(body=[context_class, *methods], type_ignores=[]), str(source), "exec"), namespace)
    context_type = namespace["DCPGatherContext"]

    def legacy(value, dim, split_sizes):
        send = value if dim == 0 else value.permute(1, 0, 2).contiguous()
        gathered, work = gather(send, group)
        # Four-argument construction must keep older KV/query callers working.
        return context_type(gathered, work, None if dim == 0 else (1, 0, 2), split_sizes)

    state = SimpleNamespace(
        dcp_size=ranks,
        dcp_group=group,
        _start_dcp_gather=legacy,
        _has_prefill=lambda metadata: metadata.num_prefills > 0,
    )
    state._start_dcp_query_gather = lambda *args, **kwargs: namespace["_start_dcp_query_gather"](state, *args, **kwargs)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(layout, torch.bfloat16)
    nope_dim = {"other_width": 256, "non_power_two": 496}.get(layout, 512)
    prepared = prepared and layout != "non_power_two"
    qn = torch.arange(tokens * 8 * nope_dim).to(dtype).view(8, tokens, nope_dim).transpose(0, 1)
    qr = torch.ones(tokens, 8, 64, dtype=dtype)
    metadata = SimpleNamespace(num_prefills=0 if decode else 1, dcp_context=SimpleNamespace(gather_context=None))
    namespace["_record_query_gather_context"](state, qn, qr, metadata)
    context = metadata.dcp_context.gather_context
    if not decode:
        assert context is None and not events
        return
    assert context.query_prepared is prepared
    assert "wait" not in events
    result = namespace["_finish_dcp_gather"](context, keep_view=keep_view)
    assert torch.equal(result[0], torch.cat([qn] * replicas, dim=1))
    assert torch.equal(result[1], torch.cat([qr] * replicas, dim=1))
    assert events.count("gather") == events.count("wait") == (0 if pcp_size == ranks else 1)
    assert ("unpack" in events) is (prepared and not keep_view)
    if prepared and not keep_view:
        assert all(part.is_contiguous() for part in result)
    elif prepared:
        assert all(
            part.untyped_storage().data_ptr() == context.gathered.untyped_storage().data_ptr() for part in result
        )


@pytest.mark.parametrize(
    "key_dtype,width,expected",
    [
        (torch.int8, 128, True),
        (torch.float8_e4m3fn, 128, False),
        (torch.int8, 64, True),
        (torch.int8, 256, True),
        (torch.int8, 127, False),
    ],
)
def test_static_store_selection_respects_native_cache_format(key_dtype, width, expected):
    source = ROOT / "vllm_ascend/attention/indexer.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendSFAIndexerBackend"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    policy = next(
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_use_fused_cache_store" for t in n.targets)
    )
    state = SimpleNamespace(
        use_pcp=False,
        enable_sparse_li_c8=True,
        c8_k_cache_dtype=key_dtype,
        head_dim=width,
    )
    namespace = {"torch": torch, "self": state, "AscendSFAIndexerBackend": SimpleNamespace}
    exec(compile(ast.Module(body=[policy], type_ignores=[]), str(source), "exec"), namespace)
    assert state._use_fused_cache_store is expected


@pytest.mark.parametrize(
    "variant", ["native", "prefill", "mixed", "pcp", "dcp1", "dcp2", "dcp4", "large", "subclass", "c4"]
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
        if isinstance(n, ast.FunctionDef)
        and n.name in ("forward", "forward_k", "_gather_cache_inputs", "_quantize_li_tensor")
    ]
    events = []
    rows = 13 if variant == "large" else 6
    quant = torch.arange(rows * 128).remainder(127).to(torch.int8).view(rows, 128)
    scales = torch.full((rows,), 0.12345, dtype=torch.float32)
    caches = (torch.zeros(256, 128, dtype=torch.int8), torch.zeros(256, 1, dtype=torch.float16))
    if variant == "c4":
        quant = torch.arange(rows * 64).remainder(127).to(torch.uint8).view(rows, 64)
        scales = torch.arange(rows * 4).to(torch.uint8).view(rows, 2, 2)
        caches = (torch.zeros(256, 64, dtype=torch.uint8), torch.zeros(256, 1, 2, 2, dtype=torch.uint8))

    class Backend(SimpleNamespace):
        q_hadamard = torch.eye(128, dtype=torch.bfloat16)

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
        "torch_npu": SimpleNamespace(npu_dynamic_quant=native_quant, npu_dynamic_mx_quant=native_quant),
        "rope_forward_triton_siso": lambda x, *args, **kwargs: x,
        "store_indexer_key_scale": store,
        "INDEXER_K_CACHE_SLOT": 0,
        "INDEXER_SCALE_CACHE_SLOT": 1,
        "_gather_prefill_cache_inputs": lambda tensors, slots, count: (tensors, slots),
        "get_pcp_num_replicated_tokens": lambda count, sharded: 0 if sharded else count,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=[*future, *methods], type_ignores=[]), str(source), "exec"), namespace)
    state = OtherBackend() if variant == "subclass" else Backend()
    state._dcp_size = int(variant[-1]) if variant in ("dcp1", "dcp2", "dcp4") else 8
    state.use_pcp = variant == "pcp"
    state._is_pcp_decode_sharded = False
    state.enable_sparse_li_c8 = variant != "c4"
    state.enable_sparse_li_c4 = variant == "c4"
    state.enable_sparse_li_quant = True
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    policy = next(
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_use_fused_cache_store" for t in n.targets)
    )
    state.k_cache = SimpleNamespace(kv_cache=caches)
    projected = torch.arange(rows * 160).reshape(rows, 160).to(torch.bfloat16)
    state.wk_weights_proj = lambda x: (projected, None)
    state.k_norm = lambda x: x
    state.head_dim, state.qk_rope_head_dim, state.is_rope_neox_style = 128, 64, False
    state.c8_k_cache_dtype, state.c8_k_scale_cache_dtype = torch.int8, torch.float16
    exec(compile(ast.Module(body=[policy], type_ignores=[]), str(source), "exec"), {**namespace, "self": state})
    state.c4_k_op_dtype, state.c4_k_scale_cache_dtype = torch.uint8, torch.uint8
    state._quantize_li_tensor = MethodType(namespace["_quantize_li_tensor"], state)
    state.forward_k = MethodType(namespace["forward_k"], state)

    def gather(*args):
        result = namespace["_gather_cache_inputs"](state, *args)
        events.append("gather_done")
        return result

    def legacy(keys, stats, slots, **kwargs):
        assert events[-1] == "gather_done" and stats.dtype == (torch.uint8 if variant == "c4" else torch.float16)
        events.append("legacy")
        caches[0].index_copy_(0, slots, keys)
        caches[1].index_copy_(0, slots, stats)

    state._gather_cache_inputs, state.write_cache = gather, legacy
    decode = 0 if variant == "prefill" else (1 if variant == "mixed" else rows)
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        num_decode_tokens=decode,
        uses_pcp_sharded_layout=variant == "pcp",
        slot_mapping=torch.arange(rows, dtype=torch.int64) + 5,
        cos=torch.ones(rows, 64),
        sin=torch.zeros(rows, 64),
    )
    x = torch.zeros(rows, 4)
    result = namespace["forward"](state, x, None, x, metadata, compute_topk=False)
    assert result is None and events == (
        ["quant", "gather_done", "fused" if variant in ("native", "dcp1", "dcp2", "dcp4", "large") else "legacy"]
    )
    assert torch.equal(caches[0][metadata.slot_mapping], quant)
    expected_scales = scales.view(rows, 1, 2, 2) if variant == "c4" else scales.to(torch.float16).view(rows, 1)
    assert torch.equal(caches[1][metadata.slot_mapping], expected_scales)
    # Current main reuses this tail for top-k instead of repeating the GEMM.
    _, _, indexer_weights = state.forward_k(x, metadata.cos, metadata.sin)
    assert torch.equal(indexer_weights, projected[:, 128:])


ALIGNMENT_METADATA = {
    "_pack_token_peer": {
        "output_stride_token_words",
        "output_stride_head_words",
        "lse_stride_token",
        "lse_stride_head",
    },
    "_fused_merge_kernel": {"out_stride_rank", "out_stride_axis1", "out_stride_axis2", "head_dim"},
    "_prepare_query_head_major": {
        "nope_stride_token",
        "nope_stride_head",
        "nope_stride_feature",
        "rope_stride_token",
        "rope_stride_head",
        "rope_stride_feature",
    },
}


CASES: list[tuple[str, str, set[str], set[str]]] = [
    ("dcp/sfa_dcp_exchange.py", "_pack_t1_peer", {"LOCAL_HEADS", "HEAD_WORDS"}, set()),
    (
        "dcp/sfa_dcp_exchange.py",
        "_pack_token_peer",
        {"LOCAL_HEADS", "HEAD_WORDS", "STATS_TILE"},
        {"output_stride_token_words", "output_stride_head_words", "lse_stride_token", "lse_stride_head", "tokens"},
    ),
    (
        "dcp/sfa_dcp_merge.py",
        "_fused_merge_kernel",
        {"RANKS", "NATIVE_LAYOUT", "BLOCK_D", "LSE_RANK_STRIDE"},
        {
            "out_stride_rank",
            "out_stride_axis1",
            "out_stride_axis2",
            "lse_stride_axis1",
            "head_count",
            "head_dim",
            "total_tiles",
            "num_programs",
        },
    ),
    (
        "dcp/sfa_dcp_query.py",
        "_prepare_query_head_major",
        {"NOPE_DIM", "ROPE_DIM"},
        {
            "nope_stride_token",
            "nope_stride_head",
            "nope_stride_feature",
            "rope_stride_token",
            "rope_stride_head",
            "rope_stride_feature",
            "tokens",
            "programs",
            "heads",
        },
    ),
    ("dcp/sfa_dcp_query.py", "_unpack_query_fragments", {"NOPE_DIM", "ROPE_DIM"}, {"tokens", "programs", "heads"}),
    ("sfa_indexer_store.py", "_store_indexer_key_scale", {"KEY_WIDTH"}, {"capacity"}),
]


@pytest.mark.parametrize("filename,kernel,configuration,runtime", CASES)
def test_kernel_parameter_classification(filename, kernel, configuration, runtime):
    tree = ast.parse((TRITON_ROOT / filename).read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == kernel)
    constants = {
        a.arg
        for a in function.args.args
        if isinstance(a.annotation, ast.Attribute) and a.annotation.attr == "constexpr"
    }
    assert constants == configuration
    assert all(name.isupper() for name in constants)
    if runtime:
        jit = next(d for d in function.decorator_list if isinstance(d, ast.Call))
        excluded = ast.literal_eval(next(k.value for k in jit.keywords if k.arg == "do_not_specialize"))
        alignment = ALIGNMENT_METADATA.get(kernel, set())
        assert runtime - alignment <= set(excluded)
        assert not alignment & set(excluded), "do not discard useful runtime layout metadata"
    used = {
        n.id
        for statement in function.body
        for n in ast.walk(statement)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assert {a.arg for a in function.args.args} <= used, "unused arguments must not create specialization keys"


@pytest.mark.parametrize("shape", [(0, 2048), (1, 0), (2, 3, 0)])
def test_empty_remap_preserves_shape_dtype_and_avoids_launch(shape):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "remap_sparse_indices_triton")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    value = torch.empty(shape, dtype=torch.int32)
    assert namespace["remap_sparse_indices_triton"](value, 8, 0, 128) is value


@pytest.mark.parametrize("ranks", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("interleave", [1, 2, 64, 128, 256])
def test_bitwise_mapping_matches_python_integer_contract(ranks, interleave):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    function = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "remap_sparse_indices_fused_kernel"
    )
    statements: list[ast.stmt] = [
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("owner", "remapped", "valid")
    ]
    indices = [
        -1,
        0,
        interleave - 1,
        interleave,
        ranks * interleave - 1,
        ranks * interleave,
        2**24 - 1,
        2**24 + 127,
        2**31 - 1,
        2**40 + 255,
        2**62 - 1,
    ]
    for rank in range(ranks):
        namespace = dict(
            idx=torch.tensor(indices, dtype=torch.int64),
            dcp_rank=rank,
            DCP_BITS=ranks.bit_length() - 1,
            INTERLEAVE_BITS=interleave.bit_length() - 1,
        )
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
        valid = [i >= 0 and (i // interleave) % ranks == rank for i in indices]
        mapped = [(i // (ranks * interleave)) * interleave + i % interleave for i in indices]
        assert namespace["valid"].tolist() == valid
        assert namespace["remapped"][valid].tolist() == [m for m, v in zip(mapped, valid) if v]


@pytest.mark.parametrize("ranks,interleave", [(1, 3), (3, 128), (8, 65), (2, 3), (3, 127), (5, 129)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_generic_wrapper_routes_to_exact_integer_fallback(ranks, interleave, dtype):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sparse_index_remap.py"
    tree = ast.parse(source.read_text())
    functions: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_remap_sparse_indices_integer", "remap_sparse_indices_triton")
    ]
    # No Triton launch objects: a generic call must reach the actual fallback.
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    value = torch.tensor([-1, 0, 1, 127, 128, 2**24 - 1, 2**24, 2**24 + 1, 2**31 - 2, 2**31 - 1], dtype=dtype)
    value = value.repeat(2, 3, 1).transpose(0, 1)
    for rank in range(ranks):
        expected = torch.full_like(value, -1)
        for i in range(3):
            for j in range(2):
                owned = [int(x) for x in value[i, j] if int(x) >= 0 and (int(x) // interleave) % ranks == rank]
                mapped = [(x // (ranks * interleave)) * interleave + x % interleave for x in owned]
                expected[i, j, : len(mapped)] = torch.tensor(mapped, dtype=dtype)
        actual = namespace["remap_sparse_indices_triton"](value, ranks, rank, interleave)
        assert torch.equal(actual, expected)
