# SPDX-License-Identifier: Apache-2.0
"""DSA Host contracts; actual constructor/CP/indices bodies execute on CPU."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import (
    DSABuilderOwners,
    freeze_dsa_rotary_reads,
    validate_dsa_read_subset,
)

ROOT = Path(__file__).resolve().parents[4]


def extract(path, *, functions=(), class_name=None, methods=(), globals_=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    if class_name:
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        nodes.extend(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "math": math, "F": F, **(globals_ or {})}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("start", [0, 4, 8, 12, 16])
def test_native_subset_preserves_baseline_absolute_reads(causal, start):
    # Positive stale worker IDs before the native window are immaterial: the
    # consumer must never address them. The certificate is interval algebra,
    # not an assertion that canonical table capacity proves cache residency.
    result = validate_dsa_read_subset(
        window_starts=(start,),
        exact_ends=(20,),
        query_lengths=(2,),
        kernel_block_span=4,
        native_window=4,
        causal=causal,
    )[0]
    baseline = set(range(max(0, 20 - 2 - 4 + int(causal)), 20))
    projected = set(range(result.first_token, result.end_token))
    assert projected == baseline.intersection(range(start, 20))


def test_native_subset_rejects_cut_query_and_unaligned_start():
    for start in (20, 15):
        with pytest.raises(ValueError, match="complete query"):
            validate_dsa_read_subset(
                window_starts=(start,),
                exact_ends=(20,),
                query_lengths=(2,),
                kernel_block_span=4,
                native_window=4,
                causal=False,
            )


@pytest.mark.parametrize("cp", [False, True])
def test_real_dsa_constructor_owns_complete_lane_and_step_buffers(cp):
    path = "vllm_ascend/attention/context_parallel/dsa_cp.py" if cp else "vllm_ascend/attention/dsa_v1.py"
    name = "AscendDSACPMetadataBuilder" if cp else "AscendDSAMetadataBuilder"
    namespace = extract(
        path,
        functions=("_aligned_dspark_index_width", "build_dspark_swa_indices")
        + (() if cp else ("_draft_uses_sparse_flash_mla",)),
        class_name=name,
        methods=(
            "__init__",
            "_init_hadamard",
            "_get_dspark_swa_indices_buffer",
            "_build_dspark_swa_indices_eager",
            "dspark_num_query_per_req",
        ),
        globals_={
            "build_dspark_swa_indices_triton": None,
            "get_storage_block_size": lambda spec: spec.block_size,
            "get_dsa_attn_kv_plan": lambda config: SimpleNamespace(
                requires_block_offset_slots=True, uses_sparse_flash_mla=True
            ),
            "DSA_METADATA_BUFFER_SIZE": 1024,
            "SAS_METADATA_SIZE": 1024,
            "AscendDSAMetadata": SimpleNamespace,
            "CUDAGraphMode": SimpleNamespace(NONE="none"),
        },
    )
    attrs = {"__init__": namespace["__init__"], "_request_capacity_factor": 1}
    if "_get_dspark_swa_indices_buffer" in namespace:
        for method in ("_get_dspark_swa_indices_buffer", "_build_dspark_swa_indices_eager", "dspark_num_query_per_req"):
            attrs[method] = namespace[method]
        attrs["_dspark_swa_indices_fast_path_eligible"] = lambda self, count: False
    if "_init_hadamard" in namespace:
        attrs["_init_hadamard"] = namespace["_init_hadamard"]
    actual = type(name, (), attrs)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="independent-draft", sliding_window=4)),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        compilation_config=SimpleNamespace(cudagraph_mode="none", cudagraph_capture_sizes=[]),
    )
    spec = SimpleNamespace(block_size=4, compress_ratio=1)
    original = actual(spec, ["draft.layer"], config, torch.device("cpu"))
    owners = DSABuilderOwners()
    first = owners.select(original, layer_names=["draft.layer"], lane=0, step=0)
    second = owners.select(original, layer_names=["draft.layer"], lane=1, step=0)
    next_step = owners.select(original, layer_names=["draft.layer"], lane=0, step=1)
    assert first is owners.select(original, layer_names=["draft.layer"], lane=0, step=0)
    assert first.vllm_config is original.vllm_config
    assert first.kv_cache_spec is original.kv_cache_spec
    assert len({builder.slot_mapping.data_ptr() for builder in (original, first, second, next_step)}) == 4
    first.spec_sas_metadata[0].fill_(37)
    assert not original.spec_sas_metadata[0].any()
    assert not second.spec_sas_metadata[0].any()
    if cp:
        assert first.spec_local_seq_lens[0].data_ptr() != second.spec_local_seq_lens[0].data_ptr()
    else:
        # The main lane preallocates these buffers; release builds may still
        # allocate them on first use. Both paths must keep lane/step ownership.
        initial = tuple(builder.dspark_swa_indices_buffer for builder in (original, first, second, next_step))
        if all(buffer is not None for buffer in initial):
            assert len({buffer.data_ptr() for buffer in initial}) == 4
        args = (torch.tensor([[5, 6]]), 3, 4, 4, torch.tensor([0, 4]), torch.tensor([8]), 4)
        result, lengths = first._build_dspark_swa_indices_eager(args)
        buffer = first.dspark_swa_indices_buffer
        assert result.data_ptr() == buffer.data_ptr()
        assert lengths.tolist() == [8] * 4
        repeated, _ = first._build_dspark_swa_indices_eager(args)
        assert repeated.data_ptr() == buffer.data_ptr()
        assert buffer is first._get_dspark_swa_indices_buffer()
        assert buffer.data_ptr() != second._get_dspark_swa_indices_buffer().data_ptr()
        for builder, previous in zip((original, first, second, next_step), initial):
            if previous is not None:
                assert builder.dspark_swa_indices_buffer is previous
    original._device_metadata_enabled = True
    with pytest.raises(NotImplementedError, match="executor completion"):
        owners.select(original, layer_names=["draft.layer"], lane=0, step=0)


@pytest.mark.parametrize("noncausal", [False, True])
def test_actual_cp_query_projection_rebases_end_instead_of_dividing_by_tp(noncausal):
    namespace = extract(
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
        class_name="AscendDSACPMetadataBuilder",
        methods=("_build_local_token_metadata",),
        globals_={"HAS_TRITON": False},
    )
    builder = SimpleNamespace(_local_token_range=lambda count: (3, 6, 3, 6))
    query = torch.tensor([0, 4, 6], dtype=torch.int32)
    # Rank 1 gets the last token of request 0 and both tokens of request 1.
    result = namespace["_build_local_token_metadata"](
        builder, 2, 6, query, torch.tensor([10, 12]), is_noncausal=noncausal
    )
    assert result[4].tolist() == [0, 1, 3]
    assert result[5].tolist() == [10, 12]
    # Rank 0 receives only the first 3 tokens of request 0. Causal reads omit
    # the query token on the later rank; noncausal reads include it.
    builder._local_token_range = lambda count: (0, 3, 3, 6)
    result = namespace["_build_local_token_metadata"](
        builder, 2, 6, query, torch.tensor([10, 12]), is_noncausal=noncausal
    )
    assert result[5].tolist() == ([10, 0] if noncausal else [9, 0])


def test_actual_dspark_sparse_indices_preserve_physical_slots_after_rebase():
    namespace = extract(
        "vllm_ascend/attention/dsa_v1.py",
        functions=("build_dspark_swa_indices", "_aligned_dspark_index_width"),
        globals_={"build_dspark_swa_indices_triton": None},
    )
    build = namespace["build_dspark_swa_indices"]
    full = torch.tensor([[10, 21, 32, 43, 54]], dtype=torch.int32)
    query = torch.tensor([0, 2])
    original, _ = build(full, 2, 4, 4, query, torch.tensor([18]), num_decode_tokens=2)
    projected, _ = build(full[:, 2:].clone(), 2, 4, 4, query, torch.tensor([10]), num_decode_tokens=2)
    assert torch.equal(original, projected)


def test_actual_rope_proxy_snapshot_survives_global_cache_overwrite():
    tree = ast.parse((ROOT / "vllm_ascend/ops/rope_dsv4.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RopeDataProxy")
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"torch": torch, "F": F}
    exec(compile(module, "actual_rope_proxy", "exec"), namespace)
    proxy = namespace["RopeDataProxy"]
    cos, sin = torch.ones(3, 1), torch.ones(3, 1) * 2
    data = {"config": {"group": (cos, sin)}}
    req = SimpleNamespace(
        cos=proxy(data),
        sin=proxy(data, is_cos=False),
        cp_metadata=SimpleNamespace(local_cos=proxy(data)[:2], local_sin=proxy(data, is_cos=False)[:2]),
    )
    metadata = SimpleNamespace(req_metadata=req)
    retained = freeze_dsa_rotary_reads({"draft.layer": metadata}, ["draft.layer"])
    cos.fill_(9)
    sin.fill_(8)
    assert req.cos._data["config"]["group"][0].tolist() == [[1.0]] * 3
    assert req.sin._data["config"]["group"][1].tolist() == [[2.0]] * 3
    assert req.cp_metadata.local_cos._data["config"]["group"][0].tolist() == [[1.0]] * 2
    assert req.cp_metadata.local_sin._data["config"]["group"][1].tolist() == [[2.0]] * 2
    assert retained


@pytest.mark.parametrize("query_width", [8, 9])
def test_actual_dspark_indices_cover_anchor_and_nonanchor_at_alignment_boundary(query_width):
    ns = extract(
        "vllm_ascend/attention/dsa_v1.py",
        functions=("build_dspark_swa_indices", "_aligned_dspark_index_width"),
        globals_={"build_dspark_swa_indices_triton": None},
    )
    table = torch.tensor([[10, 21, 32, 43, 54]], dtype=torch.int32)
    indices, lengths = ns["build_dspark_swa_indices"](
        table,
        8,
        120,
        32,
        torch.tensor([0, query_width]),
        torch.tensor([128 + query_width]),
        num_decode_tokens=query_width,
    )
    assert lengths.tolist() == [120 + query_width] * query_width
    assert (indices >= 0).sum(2).flatten().tolist() == lengths.tolist()


@pytest.mark.parametrize("start", [8, 16])
def test_actual_physical_indices_native_intersection_for_s_inside_or_before_native(start):
    ns = extract(
        "vllm_ascend/attention/dsa_v1.py",
        functions=("build_dspark_swa_indices", "_aligned_dspark_index_width"),
        globals_={"build_dspark_swa_indices_triton": None},
    )
    table = torch.tensor([[10, 21, 32, 43, 54]], dtype=torch.int32)
    query = torch.tensor([0, 2])
    projected, lens = ns["build_dspark_swa_indices"](
        table[:, start // 4 :].clone(),
        2,
        4,
        4,
        query,
        torch.tensor([18 - start]),
        num_decode_tokens=2,
    )
    logical_read = list(range(max(start, 18 - 2 - 4), 18))
    physical_read = [int(table[0, token // 4]) * 4 + token % 4 for token in logical_read]
    assert projected[0, 0][projected[0, 0] >= 0].tolist() == physical_read
    assert lens.tolist() == [len(physical_read)] * 2


def test_actual_native_call_receives_window_read_and_original_full_write():
    from contextlib import nullcontext

    captured = []
    full_slots = torch.tensor([187, 192])
    kwargs = {}

    def sparse_attention(query, **values):
        captured.append(values)
        return (query,)

    plan = SimpleNamespace(
        get_dsa_sparse_attn_op=lambda: sparse_attention,
        get_dsa_sparse_attn_base_kwargs=lambda: {},
    )
    ns = extract(
        "vllm_ascend/attention/dsa_v1.py",
        functions=("_require_req_metadata",),
        class_name="AscendDSAImpl",
        methods=("_forward_attention",),
        globals_={
            "DeviceOperator": SimpleNamespace(
                unpack_dsa_forward_kv_cache=lambda kv, ratio: (None, kv, None, None, None, None)
            ),
            "DeviceMetadataStage": SimpleNamespace(ATTENTION="attention"),
            "notify_kv_cache_written": lambda layer: None,
            "record_attention_compute_start": lambda: None,
            "wait_for_device_metadata": lambda *args: None,
            "_dsa_swa_only_cmp_ratio": lambda *args: 1,
            "_dsa_layout_kv": lambda config: "PA_BSND",
            "attention_transfer_window": nullcontext,
        },
    )
    req = SimpleNamespace(
        block_table=torch.tensor([[32, 43, 54]]),
        seq_lens=torch.tensor([10]),
        query_start_loc=torch.tensor([0, 2]),
        slot_mapping=full_slots,
        cos={"draft": torch.ones(2, 1)},
        sin={"draft": torch.zeros(2, 1)},
        ori_win_left=5,
        ori_win_right=0,
        sas_metadata=torch.zeros(4),
        dspark_swa_indices=torch.tensor([[[172, 173, 174, 175, 216, 217]]] * 2),
        vision_swa_indices=None,
        use_sparse_flash_mla_for_draft=False,
    )
    item = SimpleNamespace(req_metadata=req, num_prefills=0)

    def prolog(hidden, cos, sin, cache, slots, **values):
        kwargs["write_slots"] = slots
        return hidden, hidden, None

    impl = SimpleNamespace(
        compress_ratio=1,
        multistream_dsv4_dsa_overlap=False,
        _mla_prolog_single_stream=prolog,
        window_size=4,
        get_kv_plan=lambda: plan,
        attn_sink=None,
        softmax_scale=1.0,
        vllm_config=None,
        turboquant=None,
    )
    ns["_forward_attention"](impl, "draft", torch.ones(2, 1), torch.zeros(8), SimpleNamespace(attention=None, swa=item))
    assert captured[0]["ori_block_table"] is req.block_table
    assert captured[0]["seqused_kv"].tolist() == [10]
    assert captured[0]["cu_seqlens_q"].tolist() == [0, 2]
    assert captured[0]["ori_sparse_indices"] is req.dspark_swa_indices
    assert kwargs["write_slots"] is full_slots


def test_actual_cp_native_call_preserves_global_writes_and_local_reads():
    captured, writes = [], []

    class Unquantized:
        pass

    class Linear:
        quant_method = Unquantized()

        def __call__(self, value):
            return value

    def sparse_attention(query, **values):
        captured.append(values)
        return (query,)

    plan = SimpleNamespace(
        dsa_kv_compress_scatter=lambda cache, values, slots: writes.append(slots),
        get_dsa_sparse_attn_op=lambda: sparse_attention,
        get_dsa_sparse_attn_base_kwargs=lambda: {},
    )
    fake_torch = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(inplace_partial_rotary_mul=lambda *args, **kwargs: None))
    )
    ns = extract(
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
        class_name="AscendDSACPImpl",
        methods=("_forward",),
        globals_={
            "torch": fake_torch,
            "DeviceOperator": SimpleNamespace(
                unpack_dsa_forward_kv_cache=lambda kv, ratio: (None, kv, None, None, None, None),
                apply_dsa_q_rms=lambda query, *args: query,
            ),
            "AscendUnquantizedLinearMethod": Unquantized,
            "DeviceMetadataStage": SimpleNamespace(ATTENTION="attention"),
            "notify_kv_cache_written": lambda layer: None,
            "wait_for_device_metadata": lambda *args: None,
            "record_attention_compute_start": lambda: None,
            "get_dsa_attn_kv_plan": lambda cfg: plan,
            "_dsa_swa_only_cmp_ratio": lambda *args: 1,
            "_dsa_layout_kv": lambda config: "PA_BSND",
            "tensor_model_parallel_all_gather": lambda value, dim: torch.ones(6, 2),
        },
    )
    cp = SimpleNamespace(
        local_query_start_loc=torch.tensor([0, 3, 3]),
        local_seq_lens=torch.tensor([10, 0]),
        local_cos={"draft": torch.ones(3, 1)},
        local_sin={"draft": torch.zeros(3, 1)},
    )
    full_slots = torch.arange(6) + 200
    req = SimpleNamespace(
        block_table=torch.tensor([[32, 43, 54], [33, 44, 55]]),
        seq_lens=torch.tensor([10, 16]),
        query_start_loc=torch.tensor([0, 4, 6]),
        slot_mapping=full_slots,
        cos={"draft": torch.ones(6, 1)},
        sin={"draft": torch.zeros(6, 1)},
        ori_win_left=7,
        ori_win_right=0,
        sas_metadata=torch.zeros(4),
        cp_metadata=cp,
        dspark_swa_indices=torch.tensor([[[128, 129, 130, 131, 172, 173, 174, 175, 216, 217]]] * 3),
    )
    item = SimpleNamespace(req_metadata=req, num_prefills=0, num_actual_tokens=6)
    impl = SimpleNamespace(
        compress_ratio=1,
        wq_b=Linear(),
        wq_a=Linear(),
        q_norm=Linear(),
        wkv=Linear(),
        kv_norm=Linear(),
        num_heads=1,
        head_dim=2,
        nope_head_dim=1,
        rope_head_dim=1,
        eps=1e-6,
        q_norm_without_weight=None,
        _maybe_all_gather_o_proj_full_weight=lambda flag: None,
        window_size=4,
        attn_sink=None,
        softmax_scale=1.0,
        vllm_config=None,
    )
    ns["_forward"](
        impl,
        "draft",
        torch.ones(3, 2),
        torch.zeros(8),
        SimpleNamespace(compressor_cache=None, swa=item),
        need_gather_q_kv=True,
    )
    assert captured[0]["seqused_kv"].tolist() == [10, 0]
    assert captured[0]["cu_seqlens_q"].tolist() == [0, 3, 3]
    assert captured[0]["ori_block_table"] is req.block_table
    assert captured[0]["ori_sparse_indices"] is req.dspark_swa_indices
    assert writes[0] is full_slots


def test_actual_dspark_device_buffers_allocate_before_deferred_work_and_reuse():
    namespace = extract(
        "vllm_ascend/attention/dsa_v1.py",
        functions=("_aligned_dspark_index_width",),
        class_name="AscendDSAMetadataBuilder",
        methods=(
            "enable_device_metadata",
            "enable_dspark_device_metadata",
            "max_num_reqs_for_dspark",
            "dspark_num_query_per_req",
        ),
        globals_={"CUDAGraphMode": SimpleNamespace(NONE="none", FULL="full")},
    )
    builder = SimpleNamespace(
        compressor_ratio=1,
        _request_capacity_factor=1,
        _dspark_num_query_per_req=3,
        dspark_swa_indices_buffer=None,
        dspark_lens_scratch=None,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=4)),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            compilation_config=SimpleNamespace(cudagraph_mode="none", cudagraph_capture_sizes=[]),
        ),
        device=torch.device("cpu"),
    )
    for name in (
        "enable_device_metadata",
        "enable_dspark_device_metadata",
        "max_num_reqs_for_dspark",
        "dspark_num_query_per_req",
    ):
        setattr(builder, name, namespace[name].__get__(builder))
    builder.enable_dspark_device_metadata(16)
    indices, lengths = builder.dspark_swa_indices_buffer, builder.dspark_lens_scratch
    assert builder._device_metadata_enabled
    assert indices.shape == (16, 1, 128)
    assert lengths.shape == (16,)
    builder.enable_dspark_device_metadata(16)
    assert builder.dspark_swa_indices_buffer is indices
    assert builder.dspark_lens_scratch is lengths
