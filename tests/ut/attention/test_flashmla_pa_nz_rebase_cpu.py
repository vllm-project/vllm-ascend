# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the BF16 PA_NZ port alongside native FlashMLA.

Load production definitions without vLLM or torch_npu. Only vendor kernels,
device streams and collectives are emulated; these checks are not NPU evidence.
"""

import ast
import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"
OPS = "vllm_ascend/ops/flash_mla_pa_nz.py"


def method(name, scope):
    tree = ast.parse((ROOT / MLA).read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), MLA, "exec"), scope)
    return scope[name]


def nz_view(cache):
    return cache.view(cache.shape[0], 1, 36, 128, 16)


def write_rows(cache, slots, rows):
    physical = nz_view(cache)
    for slot, row in zip(slots.tolist(), rows):
        if slot >= 0:
            physical[slot // 128, 0, :, slot % 128, :].copy_(row.reshape(36, 16))


def read_row(cache, slot):
    return nz_view(cache)[slot // 128, 0, :, slot % 128, :].reshape(576)


def dense_pa_nz(query, physical_cache, **kwargs):
    """Emulate the vendor kernel while checking its physical layout and ABI."""
    assert kwargs["layout_q"] == "TND" and kwargs["layout_kv"] == "PA_NZ"
    assert kwargs["layout_out"] == "NTD"
    assert physical_cache.shape[1:] == (1, 36, 128, 16)
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_kv"] == -1
    heads, tokens = query.shape[1], query.shape[0]
    schedule = kwargs["metadata"]
    assert schedule[:2].tolist() == [heads, kwargs["mask_mode"]]
    lengths, cu, used = kwargs["cache_seqlens"], kwargs["cu_seqlens_q"], kwargs["seqused_q"]
    torch.testing.assert_close(schedule[3:], torch.cat((lengths, cu, used)))
    output = torch.zeros(heads, tokens, 512, dtype=query.dtype)
    lse = torch.full((heads, tokens), torch.inf, dtype=torch.float32)
    for request, length in enumerate(lengths.tolist()):
        if not length or not used[request]:
            continue
        keys = torch.stack(
            [
                physical_cache[int(kwargs["block_table"][request, index // 128]), 0, :, index % 128].reshape(576)
                for index in range(length)
            ]
        ).double()
        for offset in range(int(used[request])):
            token = int(cu[request]) + offset
            visible = length - int(used[request]) + offset + 1 if kwargs["mask_mode"] == 3 else length
            scores = query[token].double() @ keys[:visible].T * kwargs["softmax_scale"]
            output[:, token] = (scores.softmax(-1) @ keys[:visible, :512]).to(query.dtype)
            lse[:, token] = scores.logsumexp(-1).float()
    return output, lse


@pytest.fixture
def pa_runtime(monkeypatch):
    shared = runpy.run_path(str(ROOT / "tests/ut/attention/test_flashmla_dcp.py"))
    runtime = shared["runtime"].__wrapped__(monkeypatch)
    spec = importlib.util.spec_from_file_location("pa_nz_cpu_ops", ROOT / OPS)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    parent = ModuleType("vllm_ascend.ops")
    parent.flash_mla_pa_nz = ops
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", parent)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.flash_mla_pa_nz", ops)

    def schedule(lengths, *, num_heads_q, mask_mode, max_seqlen_q, cu_seqlens_q, seqused_q, **kwargs):
        assert kwargs == dict(num_heads_kv=1, max_seqlen_kv=-1, head_dim_qk=576, head_dim_v=512, layout_q="TND")
        runtime.events.append(("metadata", runtime.active_stream[0]))
        return torch.cat((lengths.new_tensor([num_heads_q, mask_mode, max_seqlen_q]), lengths, cu_seqlens_q, seqused_q))

    metadata_op, attention_op = Mock(side_effect=schedule), Mock(side_effect=dense_pa_nz)
    monkeypatch.setattr(ops, "_get_pa_nz_flash_mla_ops", lambda: (attention_op, metadata_op))
    operator = Mock()
    api = SimpleNamespace(
        mla_prolog_v2=operator,
        QkvQuantMode=SimpleNamespace(A16C16=object()),
        CacheMode=SimpleNamespace(PA_NZ=object()),
        RopeMode=SimpleNamespace(NO_ROPE=object()),
    )

    def import_prolog(name):
        assert name == "ops.mla_prolog_v4"
        return api

    monkeypatch.setattr(ops, "import_module", import_prolog)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1)

    def builder(*, heads=6, size=8, rank=0, interleave=128, speculative=True, asynchronous=False):
        impl = SimpleNamespace(
            num_heads=heads,
            scale=0.125,
            dtype=torch.bfloat16,
            use_mla_rope=False,
            flash_mla_layout="pa_nz",
            dcp_size=size,
            dcp_rank=rank,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave),
                speculative_config=SimpleNamespace(method="dspark") if speculative else None,
            ),
        )
        return runtime.api.FlashMLAMetadataBuilder(impl, torch.device("cpu"), 8, mask, enable_async=asynchronous)

    def inputs(lengths=(389, 2), queries=(3, 2), *, causal=True, tokens=8):
        common = shared["common"](list(lengths), list(queries), causal=causal, tokens=tokens)
        common.max_query_len = max(queries)
        common.block_table_tensor.copy_(torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.int32))
        common.slot_mapping.copy_(torch.tensor([128, -1, 130, 256, -1][: sum(queries)], dtype=torch.int64))
        return common

    return SimpleNamespace(
        runtime=runtime,
        ops=ops,
        prolog_api=api,
        prolog_op=operator,
        metadata_op=metadata_op,
        attention_op=attention_op,
        builder=builder,
        inputs=inputs,
        Metadata=shared["Metadata"],
    )


@pytest.mark.parametrize("heads", [6, 12])
def test_current_bf16_prolog_uses_nine_tensors_and_local_heads(pa_runtime, heads):
    ops, api, operator = pa_runtime.ops, pa_runtime.prolog_api, pa_runtime.prolog_op
    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    dq = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    qb = torch.randn(8, heads * 192, dtype=torch.bfloat16, requires_grad=True)
    uk = torch.randn(heads, 128, 512, dtype=torch.bfloat16, requires_grad=True)
    dkv = torch.randn(16, 576, dtype=torch.bfloat16, requires_grad=True)
    gamma_q, gamma_kv = torch.ones(8, dtype=torch.bfloat16), torch.ones(512, dtype=torch.bfloat16)
    cache = torch.zeros(3, 128, 1, 576, dtype=torch.bfloat16)
    slots = torch.tensor([128, 129, 130], dtype=torch.int32)
    expected = (torch.zeros(3, heads, 576, dtype=torch.bfloat16), None, None)
    operator.return_value = expected
    assert ops.mla_prolog_no_rope(hidden, dq, qb, uk, dkv, gamma_q, gamma_kv, cache, slots, 1e-6) is expected
    args, kwargs = operator.call_args
    assert len(args) == 9
    for actual, source in zip(args[:5], (hidden, dq, qb, dkv, uk)):
        assert actual.data_ptr() == source.data_ptr() and not actual.requires_grad
    torch.testing.assert_close(args[5], gamma_q.float())
    torch.testing.assert_close(args[6], gamma_kv.float())
    assert args[7] is cache and args[8].dtype == torch.int64
    torch.testing.assert_close(args[8], slots.long())
    assert kwargs == dict(
        qkv_quant_mode=api.QkvQuantMode.A16C16,
        cache_mode=api.CacheMode.PA_NZ,
        rope_mode=api.RopeMode.NO_ROPE,
        return_qr=False,
        norm_eps=1e-6,
    )


@pytest.mark.parametrize("heads", [6, 12])
def test_bf16_weight_preparation_keeps_unpadded_tp_local_heads(heads):
    replacements = []

    def replace(owner, name, value, *, prefer_copy):
        replacements.append((name, prefer_copy))
        setattr(owner, name, value)

    cast = Mock(side_effect=lambda value, fmt: value)
    prepare = method(
        "_prepare_flash_mla_bf16_weights",
        {"torch_npu": SimpleNamespace(npu_format_cast=cast), "ACL_FORMAT_FRACTAL_NZ": 29, "replace_parameter": replace},
    )
    qb = torch.randn(heads * 192, 8, dtype=torch.bfloat16)
    uk = torch.randn(heads, 128, 512, dtype=torch.bfloat16)
    impl = SimpleNamespace(
        q_proj=SimpleNamespace(weight=torch.nn.Parameter(qb, requires_grad=False)),
        num_heads=heads,
        qk_head_dim=192,
        q_lora_rank=8,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        W_UK_T=uk,
    )
    prepare(impl)
    assert impl.flash_weight_uq_qr.shape == (8, heads * 192)
    torch.testing.assert_close(impl.flash_weight_uq_qr, qb.T)
    assert impl.flash_weight_uk.shape == (heads, 128, 512)
    assert impl.flash_weight_uk.data_ptr() == uk.data_ptr()
    assert replacements == [("flash_weight_uq_qr", True), ("flash_weight_uk", True)]
    assert cast.call_args.args[1] == 29
    impl.q_proj.weight = torch.nn.Parameter(torch.zeros(heads * 2 * 192, 8), requires_grad=False)
    with pytest.raises(ValueError, match="TP-local Q-B"):
        prepare(impl)


@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("q_len,mode", [(1, 3), (3, 3), (3, 0)])
def test_pa_nz_adapter_preserves_mask_schedule_layout_and_dense_output(pa_runtime, heads, q_len, mode):
    torch.manual_seed(11)
    config = pa_runtime.runtime.api.FlashMLAConfig(
        heads, 0.125, layout_kv="PA_NZ", mask_mode=mode, return_softmax_lse=True
    )
    adapter = pa_runtime.runtime.api.PaNzFlashMLAImpl.load(config)
    query = torch.randn(q_len, heads, 576, dtype=torch.bfloat16) * 0.1
    keys = torch.randn(5, 576, dtype=torch.bfloat16) * 0.1
    cache = torch.zeros(2, 128, 1, 576, dtype=torch.bfloat16)
    write_rows(cache, torch.arange(128, 133), keys)
    lengths, cu, used = (
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([0, q_len], dtype=torch.int32),
        torch.tensor([q_len], dtype=torch.int32),
    )
    schedule = adapter.build_metadata(lengths, cu, used, q_len=q_len)
    mask_mode = 0 if q_len == 1 else mode
    assert schedule[:3].tolist() == [heads, mask_mode, q_len]
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1) if q_len > 1 and mode == 3 else None
    output, lse = adapter.attention(
        query,
        cache,
        block_table=torch.tensor([[1]], dtype=torch.int32),
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        seqused_q=used,
        metadata=schedule,
        attn_mask=mask,
        q_len=q_len,
    )
    expected, expected_lse = [], []
    for offset in range(q_len):
        visible = 5 - q_len + offset + 1 if mask_mode == 3 else 5
        scores = torch.einsum("hd,kd->hk", query[offset].double(), keys[:visible].double()) * 0.125
        expected.append(torch.einsum("hk,kd->hd", scores.softmax(-1), keys[:visible, :512].double()))
        expected_lse.append(scores.logsumexp(-1))
    torch.testing.assert_close(output.double(), torch.stack(expected, dim=1), atol=0.002, rtol=0.01)
    torch.testing.assert_close(lse.double(), torch.stack(expected_lse, dim=1), atol=1e-6, rtol=1e-6)
    assert pa_runtime.attention_op.call_args.kwargs["mask_mode"] == mask_mode
    if q_len == 1:
        assert pa_runtime.attention_op.call_args.kwargs["attn_mask"] is None


@pytest.mark.parametrize("invalid", ["page", "dtype", "stride"])
def test_pa_nz_adapter_rejects_incompatible_cache(pa_runtime, invalid):
    config = pa_runtime.runtime.api.FlashMLAConfig(6, 0.125, layout_kv="PA_NZ")
    adapter = pa_runtime.runtime.api.PaNzFlashMLAImpl.load(config)
    if invalid == "page":
        cache = torch.zeros(2, 256, 1, 576, dtype=torch.bfloat16)
    elif invalid == "dtype":
        cache = torch.zeros(2, 128, 1, 576, dtype=torch.float16)
    else:
        cache = torch.zeros(2, 128, 1, 1152, dtype=torch.bfloat16)[..., :576]
    with pytest.raises(ValueError, match="PA_NZ FlashMLA cache"):
        adapter.validate_cache(cache)
    pa_runtime.attention_op.assert_not_called()


def test_pa_nz_prolog_reports_missing_current_dsl_api(pa_runtime, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(pa_runtime.ops, "import_module", missing)
    with pytest.raises(RuntimeError, match="ops.mla_prolog_v4.mla_prolog_v2"):
        pa_runtime.ops._get_bf16_mla_prolog_op()


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("causal,speculative", [(True, True), (True, False), (False, True)])
def test_pa_nz_dcp_history_current_lengths_and_precomputed_copy_indices(pa_runtime, rank, causal, speculative):
    builder = pa_runtime.builder(rank=rank, speculative=speculative)
    inputs = pa_runtime.inputs(causal=causal)
    flash = builder.build(inputs, 2, 5, False, retain_for_graph=True)
    split = causal and speculative
    histories = [389 - 3, 2 - 2] if split else [389, 2]
    expected_lengths = [sum(position // 128 % 8 == rank for position in range(length)) for length in histories]
    assert flash.cache_lens[:2].tolist() == expected_lengths
    assert not flash.cache_lens[2:].any()
    assert flash.global_cache_lens is None
    assert flash.schedule[:3].tolist() == [6 * 8, 0, 3]
    assert (flash.current is not None) == split
    if not split:
        return
    assert flash.current.schedule[:3].tolist() == [6, 3, 3]
    assert flash.current.slots.tolist() == [0, 1, 2, 128, 129, -1, -1, -1]
    scratch = (flash.current.cache.shape[0] - 1) * 128
    assert flash.pa_nz_current_slots.tolist() == [0, 1, 2, 128, 129, scratch, scratch, scratch]
    assert flash.pa_nz_history_slots.tolist() == [128, 0, 130, 256, 0, 0, 0, 0]
    for slots, indices in (
        (flash.pa_nz_current_slots, flash.pa_nz_copy_src_indices),
        (flash.pa_nz_history_slots, flash.pa_nz_copy_dst_indices),
    ):
        expected = [((slot // 128 * 36 + tile) * 128 + slot % 128) for slot in slots.tolist() for tile in range(36)]
        assert indices.tolist() == expected
    addresses = tuple(
        value.data_ptr() for value in (flash.pa_nz_current_slots, flash.pa_nz_copy_src_indices, flash.schedule)
    )
    inputs.seq_lens[0] = 259
    replay = builder.build(inputs, 2, 5, False)
    assert replay is flash
    assert addresses == tuple(
        value.data_ptr() for value in (replay.pa_nz_current_slots, replay.pa_nz_copy_src_indices, replay.schedule)
    )


@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("dcp", [False, True])
def test_decode_composes_prolog_local_query_and_current_to_history_copy(pa_runtime, heads, dcp):
    torch.manual_seed(31)
    builder = pa_runtime.builder(heads=heads, size=8 if dcp else 1)
    inputs = pa_runtime.inputs(lengths=(3, 2))
    inputs.slot_mapping.copy_(torch.tensor([128, 129, 130, 256, 257]))
    flash = builder.build(inputs, 2, 5, False)
    metadata = pa_runtime.Metadata(external_flashmla=flash, num_decode_tokens=8)
    history = torch.zeros(7, 128, 1, 576, dtype=torch.bfloat16)
    query = torch.randn(8, heads, 576, dtype=torch.bfloat16) * 0.1
    keys = torch.randn(8, 576, dtype=torch.bfloat16) * 0.1

    def prolog(*args, **kwargs):
        assert len(args) == 9 and kwargs["norm_eps"] == 1e-6
        assert args[2].shape == (8, heads * 192) and args[4].shape == (heads, 128, 512)
        write_rows(args[7], args[8], keys)
        return query, None, None

    pa_runtime.prolog_op.side_effect = prolog

    def scatter(cache, indices, values):
        cache.index_copy_(0, indices.flatten(), values)

    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_scatter_nd_update_=scatter),
        "PA_NZ_TILE_WIDTH": 16,
        "PA_NZ_TILE_COUNT": 36,
        "FLASHMLA_BLOCK_SIZE": 128,
        "FLASHMLA_QK_DIM": 576,
        "record_attention_compute_start": Mock(),
    }
    copy = method("_copy_pa_nz_rows", scope)
    decode = method("_forward_decode_flash_bf16", scope)
    impl = SimpleNamespace(
        flash_mla_layout="pa_nz",
        q_a_layernorm=SimpleNamespace(variance_epsilon=1e-6),
        kv_a_layernorm=object(),
        flash_weight_uq_qr=torch.zeros(8, heads * 192, dtype=torch.bfloat16),
        flash_weight_uk=torch.zeros(heads, 128, 512, dtype=torch.bfloat16),
        flash_gamma_qa=torch.ones(8),
        flash_gamma_kva=torch.ones(512),
        weight_dq=torch.zeros(16, 8, dtype=torch.bfloat16),
        weight_dkv_kr=torch.zeros(16, 576, dtype=torch.bfloat16),
        num_heads=heads,
        kv_lora_rank=512,
        dcp_enabled=dcp,
        wait_flash_mla_metadata=Mock(),
        _copy_pa_nz_rows=Mock(side_effect=copy),
        _v_up_proj=Mock(side_effect=lambda value: value.transpose(0, 1)),
        reorg_decode_q=Mock(
            side_effect=lambda nope, rope: (torch.cat([nope] * 8, dim=1), torch.cat([rope] * 8, dim=1))
        ),
        _forward_decode_split_attention=Mock(return_value=torch.ones(8, heads, 512)),
    )
    hidden = torch.randn(8, 16, dtype=torch.bfloat16)
    result = decode(impl, hidden, history, metadata)
    impl.wait_flash_mla_metadata.assert_called_once_with(metadata)
    if dcp:
        assert result is impl._forward_decode_split_attention.return_value
        assert pa_runtime.prolog_op.call_args.args[7] is flash.current.cache
        impl._copy_pa_nz_rows.assert_called_once()
        args, kwargs = impl._forward_decode_split_attention.call_args
        assert args[0].shape == (8, heads * 8, 512) and args[1].shape == (8, heads * 8, 64)
        assert args[2] is history and args[3:6] == (None, None, None)
        assert kwargs == {"current_kv_staged": True}
        pa_runtime.attention_op.assert_not_called()
    else:
        assert pa_runtime.prolog_op.call_args.args[7] is history
        impl._copy_pa_nz_rows.assert_not_called()
        impl._forward_decode_split_attention.assert_not_called()
        for offset, start in ((0, 0), (1, 0), (2, 0), (0, 3), (1, 3)):
            token = start + offset
            scores = query[token].double() @ keys[start : token + 1].double().T * 0.125
            expected = scores.softmax(-1) @ keys[start : token + 1, :512].double()
            torch.testing.assert_close(result[token].double(), expected, atol=0.002, rtol=0.01)
        assert not result[5:].any()
    for row, slot in enumerate(inputs.slot_mapping.tolist()):
        torch.testing.assert_close(read_row(history, slot), keys[row])


@pytest.mark.parametrize("capture", [False, True])
def test_pa_nz_metadata_stream_builds_both_schedules_before_consumer(pa_runtime, monkeypatch, capture):
    runtime = pa_runtime.runtime
    main = torch.npu.current_stream()
    metadata_stream = type(main)("metadata")
    monkeypatch.setattr(torch.npu, "Stream", lambda: metadata_stream, raising=False)
    monkeypatch.setattr(
        torch.npu, "current_stream", lambda: metadata_stream if runtime.active_stream[0] == "metadata" else main
    )
    builder = pa_runtime.builder(asynchronous=True)
    flash = builder.build(pa_runtime.inputs(), 2, 5, False, retain_for_graph=capture)
    if capture:
        assert not flash.metadata_started
    builder.start_metadata(flash)
    count = pa_runtime.metadata_op.call_count
    builder.start_metadata(flash)
    assert pa_runtime.metadata_op.call_count == count
    assert flash.metadata_started and flash.metadata_ready == "metadata"
    assert runtime.events[-3:] == [("metadata", "metadata"), ("metadata", "metadata"), ("record", "metadata")]
    recorded = {id(tensor) for tensor, stream in runtime.recorded_streams if stream is metadata_stream}
    assert {
        id(value) for value in (flash.schedule, flash.cache_lens, flash.cu, flash.used_q, flash.current.schedule)
    } <= recorded
    builder.wait_metadata(flash)
    assert runtime.events[-1] == ("wait", "main", "metadata")
