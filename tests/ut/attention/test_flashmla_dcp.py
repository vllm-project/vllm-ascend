# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the production FlashMLA DCP orchestration.

Run with --confcutdir=tests/ut/attention without vLLM/torch_npu installed.
Only dependencies, attention kernels and collectives are emulated. Dense
attention is the independent reference; this is not NPU or ACL graph evidence.
"""

import ast
import runpy
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_method(path, class_name, method, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[method]


def local_lengths(lengths, *, dcp_size, dcp_rank, cp_kv_cache_interleave_size):
    # Stand-in for vLLM's public helper; expectations below enumerate owners.
    tile = cp_kv_cache_interleave_size
    base = lengths // (tile * dcp_size) * tile
    return base + (lengths % (tile * dcp_size) - dcp_rank * tile).clamp(0, tile)


def scatter_current(*, key, value, key_cache, value_cache, slot_mapping, cache_mode):
    assert cache_mode == "Norm"
    for token, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            key_cache[slot // 128, slot % 128].copy_(key[token])
            value_cache[slot // 128, slot % 128].copy_(value[token])


def paged_attention(query, cache, **kwargs):
    """CPU replacement for the external op, with its NTD/NT return contract."""
    heads, tokens = query.shape[1], query.shape[0]
    output = torch.zeros(heads, tokens, 512, dtype=query.dtype)
    lse = torch.full((heads, tokens), torch.inf, dtype=torch.float32)
    cu, used = kwargs["cu_seqlens_q"], kwargs["seqused_q"]
    for request, length in enumerate(kwargs["cache_seqlens"].tolist()):
        qlen = int(used[request])
        if not length or not qlen:
            continue
        indices = torch.arange(length)
        blocks = kwargs["block_table"][request, indices // 128].long()
        keys = cache[blocks, indices % 128, 0].double()
        for offset in range(qlen):
            token = int(cu[request]) + offset
            visible = length - qlen + offset + 1 if kwargs["mask_mode"] == 3 else length
            scores = query[token].double() @ keys[:visible].T * kwargs["softmax_scale"]
            output[:, token] = (scores.softmax(-1) @ keys[:visible, :512]).to(query.dtype)
            lse[:, token] = scores.logsumexp(-1).float()
    return output, lse


def combine_outputs(recv, head_dim, *, scatter_dim, local_output=None, local_lse=None):
    assert head_dim == 512 and scatter_dim == 1
    outputs, lses = recv
    if local_output is not None:
        outputs = [*outputs, local_output]
        lses = [*lses, local_lse]
    outputs = torch.stack(outputs).double()
    stats = torch.stack(lses).double()
    valid = torch.isfinite(stats)
    stats = torch.where(valid, stats, -torch.inf)
    maximum = stats.amax(0)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.exp(stats - maximum)
    outputs = torch.where(valid, outputs, 0)
    result = (outputs * weights).sum(0) / weights.sum(0).clamp_min(1e-300)
    return result.to(recv[0][0].dtype)


@pytest.fixture
def runtime(monkeypatch):
    """Bind production DCP methods to CPU attention, stream and collective emulators."""
    events: list[tuple[str, ...]] = []
    recorded_streams = []
    active_stream = ["main"]
    current_query_storages: set[int] = set()
    original_copy = torch.Tensor.copy_

    def copy_tensor(destination, *args, **kwargs):
        if destination.untyped_storage().data_ptr() in current_query_storages:
            events.append(("query_copy", active_stream[0]))
        return original_copy(destination, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)

    def scatter(**kwargs):
        events.append(("scatter", active_stream[0]))
        return scatter_current(**kwargs)

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            events.append(("record", self.name))
            return self.name

        def wait_event(self, event):
            events.append(("wait", self.name, event))

    main_stream, attn_stream = Stream("main"), Stream("attn")

    @contextmanager
    def on_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream.name
        try:
            yield
        finally:
            active_stream[0] = previous

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: main_stream, stream=on_stream), raising=False
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: recorded_streams.append((tensor, stream)))

    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, relative):
        return install(name, **runpy.run_path(str(ROOT / relative)))

    install("vllm_ascend")
    utils = install("vllm_ascend.attention.utils", get_flashmla_ops=lambda: None)
    tree = ast.parse((ROOT / "vllm_ascend/attention/utils.py").read_text(encoding="utf-8"))
    constant = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MLA_FLASH_SUPPORTED_Q_HEADS" for target in node.targets)
    )
    exec(compile(ast.Module(body=[constant], type_ignores=[]), "<heads>", "exec"), utils.__dict__)
    install("vllm.v1.attention.backends.utils", get_dcp_local_seq_lens=local_lengths)
    install(
        "vllm.forward_context", BatchDescriptor=object, get_forward_context=Mock(), is_forward_context_available=Mock()
    )

    def rope(positions):
        values = positions[:, None, None, None].expand(-1, 1, 1, 64).to(torch.bfloat16)
        return values, -values

    install("vllm_ascend.ops.rotary_embedding", get_cos_and_sin_mla=rope)
    api = load("vllm_ascend.attention.flashmla", "vllm_ascend/attention/flashmla.py")

    def schedule(lengths, *, cu_seqlens_q, seqused_q, **kwargs):
        return torch.cat((lengths, cu_seqlens_q, seqused_q))

    metadata_op = Mock(side_effect=schedule)

    def attention(query, cache, **kwargs):
        kind = "current" if kwargs["mask_mode"] == 3 else "history"
        events.append((kind, active_stream[0]))
        return paged_attention(query, cache, **kwargs)

    attention_op = Mock(side_effect=attention)
    monkeypatch.setattr(
        api.FlashMLAImpl,
        "load",
        classmethod(lambda cls, config: cls(config, attention_op, metadata_op)),
    )

    def builder(*, size=8, rank=0, interleave=1, heads=8, dtype=torch.float16, use_rope=False, speculative=True):
        impl = SimpleNamespace(
            num_heads=heads,
            scale=0.125,
            dtype=dtype,
            use_mla_rope=use_rope,
            dcp_size=size,
            dcp_rank=rank,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave),
                speculative_config=SimpleNamespace(method="dspark") if speculative else None,
            ),
        )
        return api.FlashMLAMetadataBuilder(
            impl, torch.device("cpu"), 8, torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1)
        )

    class AttentionKind(Enum):
        HISTORY = "history"
        CURRENT = "current"

    context = SimpleNamespace(capturing=False, is_draft_model=False, is_draft_model_prefill=False)
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_scatter_pa_kv_cache=scatter),
        "record_attention_compute_start": Mock(),
        "fused_dcp_lse_combine": combine_outputs,
        "AscendMLAMetadata": object,
        "AscendMLADCPDecodeMetadata": SimpleNamespace,
        "CPKVScope": AttentionKind,
        "_EXTRA_CTX": context,
        "cp_decode_comm_stream": lambda: attn_stream,
        "get_graph_params": Mock(side_effect=AssertionError("FlashMLA must not prepare FIA graph tasks")),
    }
    scope["torch_npu"]._npu_fused_infer_attention_score_get_max_workspace = Mock(
        side_effect=AssertionError("FlashMLA must not query FIA workspace")
    )
    methods = {
        name: load_method("vllm_ascend/attention/context_parallel/mla_cp.py", "AscendMlaDCPImpl", name, scope)
        for name in (
            "_forward_external_flashmla",
            "_prepare_dcp_flashmla_current",
            "_run_dcp_flashmla_attention",
            "_forward_decode_split_attention",
        )
    }

    methods.update(
        {
            name: load_method("vllm_ascend/attention/context_parallel/common_cp.py", "DCPImplMixin", name, scope)
            for name in ("_local_decode_query", "_merge_dcp_attention_output")
        }
    )

    def bind(impl):
        for name, method in methods.items():
            setattr(impl, name, MethodType(method, impl))
        impl._forward_decode_split_attention = Mock(wraps=impl._forward_decode_split_attention)
        impl._run_dcp_mtp_split_attention_op = Mock(side_effect=AssertionError("FlashMLA must not call FIA"))
        return impl

    return SimpleNamespace(
        builder=builder,
        api=api,
        metadata_op=metadata_op,
        attention_op=attention_op,
        forward=methods["_forward_external_flashmla"],
        bind=bind,
        scope=scope,
        context=context,
        events=events,
        active_stream=active_stream,
        recorded_streams=recorded_streams,
        current_query_storages=current_query_storages,
    )


def common(lengths, queries, *, tokens=None, causal=True):
    boundaries = [0]
    for length in queries:
        boundaries.append(boundaries[-1] + length)
    actual = boundaries[-1]
    return SimpleNamespace(
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        query_start_loc=torch.tensor(boundaries, dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1, 2]] * len(lengths), dtype=torch.int32),
        slot_mapping=torch.full((actual,), -1, dtype=torch.int64),
        positions=torch.arange(actual),
        num_actual_tokens=actual,
        num_input_tokens=tokens or actual,
        causal=causal,
    )


@pytest.mark.parametrize(
    "causal,speculative,interleave,rank",
    [
        (True, True, 1, 0),
        (True, True, 4, 1),
        (True, True, 128, 7),
        (True, False, 4, 1),
        (False, True, 128, 7),
        (False, False, 1, 0),
    ],
)
def test_visibility_uses_global_history_before_sharding(runtime, causal, speculative, interleave, rank):
    builder = runtime.builder(rank=rank, interleave=interleave, speculative=speculative)
    split = causal and speculative
    lengths = [1031, 3, 132, 0]
    queries = [4, 3, 2, 1] if speculative else [1, 1, 1, 1]
    inputs = common(lengths, queries, tokens=16, causal=causal)
    flash = builder.build(inputs, 4, sum(queries), False)
    expected = []
    for length, query in zip(lengths, queries):
        visible = max(0, length - query) if split else length
        expected.append(sum(token // interleave % 8 == rank for token in range(visible)))
    assert flash.cache_lens[:4].tolist() == expected
    assert not flash.cache_lens[4:].any()
    assert flash.query.shape == (16, 64, 576)
    assert flash.adapter.config.mask_mode == 0
    assert flash.adapter.config.return_softmax_lse
    torch.testing.assert_close(flash.schedule, torch.cat((flash.cache_lens, flash.cu, flash.used_q)))
    if split:
        assert flash.current is not None
        assert flash.current.query.shape == (16, 8, 576)
        assert flash.current.slots.tolist() == [0, 1, 2, 3, 128, 129, 130, 256, 257] + [-1] * 7
        assert flash.current.adapter.config.mask_mode == 3
        torch.testing.assert_close(flash.current.schedule, torch.cat((flash.used_q, flash.cu, flash.used_q)))
    else:
        assert flash.current is None


def addresses(flash):
    tensors = {key: value for key, value in vars(flash).items() if isinstance(value, torch.Tensor)}
    if flash.current is not None:
        tensors.update(
            {f"current.{key}": value for key, value in vars(flash.current).items() if isinstance(value, torch.Tensor)}
        )
    return {key: value.data_ptr() for key, value in tensors.items()}


@pytest.mark.parametrize("causal,rank", [(True, 3), (False, 7)])
def test_metadata_refresh_reuses_device_rank(runtime, monkeypatch, causal, rank):
    builder = runtime.builder(rank=rank, interleave=4)
    rank_tensors = []

    def partition(lengths, *, dcp_size, dcp_rank, cp_kv_cache_interleave_size):
        # vLLM's int branch constructs a device tensor and synchronizes H2D.
        assert isinstance(dcp_rank, torch.Tensor)
        assert dcp_rank.dtype == torch.int32
        assert dcp_rank.device == lengths.device
        assert dcp_rank.ndim == 0
        assert dcp_rank.item() == rank
        rank_tensors.append(dcp_rank)
        return local_lengths(
            lengths,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

    monkeypatch.setitem(builder.build.__globals__, "get_dcp_local_seq_lens", partition)
    for step, capacity in enumerate((4, 8, 4)):
        lengths = [35 + step, 130 + step]
        inputs = common(lengths, [2, 2], tokens=capacity, causal=causal)
        # All host inputs and builder constants already exist before refresh.
        with monkeypatch.context() as guard:
            guard.setattr(torch, "tensor", Mock(side_effect=AssertionError("tensor construction during refresh")))
            flash = builder.build(inputs, 2, 4, False, retain_for_graph=True)
        visible = [length - 2 if causal else length for length in lengths]
        expected = [sum(token // 4 % 8 == rank for token in range(length)) for length in visible]
        assert flash.cache_lens[:2].tolist() == expected
    assert len(rank_tensors) == 3
    assert all(tensor is rank_tensors[0] for tensor in rank_tensors)


@pytest.mark.parametrize("speculative,rank", [(False, 0), (True, 7)])
def test_single_token_replay_keeps_configured_split_and_buffer_addresses(runtime, monkeypatch, speculative, rank):
    builder = runtime.builder(rank=rank, speculative=speculative, use_rope=True)
    inputs = common([9, 1], [1, 1], tokens=4)
    flash = builder.build(inputs, 2, 2, False, retain_for_graph=True)
    assert (flash.current is not None) is speculative
    pointers = addresses(flash)
    schedule = flash.schedule.clone()
    runtime.metadata_op.reset_mock()
    inputs.seq_lens.copy_(torch.tensor([17, 0]))
    inputs.positions.add_(8)
    inputs.slot_mapping.copy_(torch.arange(2))
    torch.testing.assert_close(flash.schedule, schedule)

    def forbidden(*args, **kwargs):
        pytest.fail("device metadata refresh must not read back tensors")

    with monkeypatch.context() as patch:
        for method in ("item", "tolist", "cpu", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        same = builder.build(inputs, 2, 2, False)

    assert same is flash
    visible = 16 if speculative else 17
    expected_length = sum(token % 8 == rank for token in range(visible))
    assert flash.cache_lens.tolist() == [expected_length, 0, 0, 0, 0]
    assert flash.used_q.tolist() == [1, 0, 0, 0, 0]
    assert flash.token_live.tolist() == [True, False, False, False]
    assert flash.slots.tolist() == [0, -1, -1, -1]
    assert flash.positions.tolist() == [8, 9, 0, 0]
    torch.testing.assert_close(flash.cos[:, 0, 0, 0], flash.positions.to(flash.cos.dtype))
    torch.testing.assert_close(flash.sin, -flash.cos)
    if flash.current is not None:
        assert flash.current.slots.tolist() == [0, -1, -1, -1]
        torch.testing.assert_close(flash.current.schedule, torch.cat((flash.used_q, flash.cu, flash.used_q)))
    assert flash.adapter.config.num_heads == 64
    assert flash.adapter.config.mask_mode == 0
    assert flash.adapter.config.return_softmax_lse
    assert runtime.metadata_op.call_count == (2 if speculative else 1)
    assert addresses(flash) == pointers
    assert (flash.current is not None) is speculative
    torch.testing.assert_close(flash.schedule, torch.cat((flash.cache_lens, flash.cu, flash.used_q)))


def test_current_pages_cross_physical_blocks_and_eager_does_not_retain(runtime):
    builder = runtime.builder()
    flash = builder.build(common([260, 133], [130, 129]), 2, 259, True)
    assert flash.current.slots.tolist() == list(range(130)) + list(range(256, 385))
    assert flash.current.block_table[:2, :2].tolist() == [[0, 1], [2, 3]]
    assert builder.buffers == {}


@pytest.mark.parametrize("causal", [True, False])
def test_dcp_one_keeps_original_single_attention_contract(runtime, causal):
    builder = runtime.builder(size=1)
    flash = builder.build(common([9], [2], tokens=4, causal=causal), 1, 2, False)
    assert flash.current is None
    assert flash.query.shape == (4, 8, 576)
    assert flash.cache_lens.tolist() == [9, 0, 0, 0, 0]
    assert flash.adapter.config.mask_mode == (3 if causal else 0)
    assert not flash.adapter.config.return_softmax_lse


@pytest.mark.parametrize(
    "causal,speculative,interleave,dtype,capturing",
    [
        (True, False, 1, torch.float16, False),
        (True, False, 128, torch.bfloat16, True),
        (True, True, 4, torch.float16, False),
        (True, True, 128, torch.bfloat16, True),
        (False, True, 1, torch.bfloat16, False),
        (False, True, 4, torch.float16, True),
    ],
)
def test_forward_matches_dense_with_strided_history_and_empty_ranks(
    runtime, monkeypatch, causal, speculative, interleave, dtype, capturing
):
    runtime.context.capturing = capturing
    torch.manual_seed(41)
    size, local_heads, tokens = 8, 8, 5
    split = causal and speculative
    queries = [3, 1] if speculative else [1, 1]
    histories = [257, 0]
    lengths = [history + query for history, query in zip(histories, queries)]
    keys = [torch.randn(length, 576, dtype=dtype) * 0.2 for length in lengths]
    query = torch.randn(tokens, size * local_heads, 576, dtype=dtype) * 0.2
    current_kv = torch.zeros(tokens, 1, 576, dtype=dtype)
    start = 0
    for sequence, qlen in zip(keys, queries):
        current_kv[start : start + qlen, 0] = sequence[-qlen:]
        start += qlen
    flashes, caches, partials = [], [], []
    for rank in range(size):
        builder = runtime.builder(rank=rank, interleave=interleave, dtype=dtype, speculative=speculative)
        inputs = common(lengths, queries, tokens=tokens, causal=causal)
        inputs.block_table_tensor.copy_(torch.tensor([[1, 3, 5], [2, 4, 6]]))
        flash = builder.build(inputs, 2, sum(queries), False)
        # Nonzero storage offset and padding between physical pages.
        stride = 128 * 576 + 32
        backing = torch.full((8 * stride + 64,), 17.0, dtype=dtype)
        cache = backing.as_strided((8, 128, 1, 576), (stride, 576, 576, 1), 32)
        for request, sequence in enumerate(keys):
            owners = torch.arange(sequence.shape[0]) // interleave % size
            local = sequence[owners == rank]
            for index, row in enumerate(local):
                block = int(inputs.block_table_tensor[request, index // 128])
                cache[block, index % 128, 0].copy_(row)
        flash.query.copy_(query)
        partials.append(
            flash.adapter.attention(
                flash.query,
                cache,
                block_table=flash.block_table,
                cache_seqlens=flash.cache_lens,
                cu_seqlens_q=flash.cu,
                seqused_q=flash.used_q,
                metadata=flash.schedule,
                attn_mask=None,
            )
        )
        flashes.append(flash)
        caches.append((cache, backing, backing.clone()))

    expected = torch.zeros(tokens, size * local_heads, 512, dtype=torch.float64)
    start = 0
    for sequence, history, qlen in zip(keys, histories, queries):
        for offset in range(qlen):
            visible = history + offset + 1 if causal else sequence.shape[0]
            scores = query[start + offset].double() @ sequence[:visible].double().T * 0.125
            expected[start + offset] = scores.softmax(-1) @ sequence[:visible, :512].double()
        start += qlen
    for rank, flash in enumerate(flashes):
        owned = slice(rank * local_heads, (rank + 1) * local_heads)

        def exchange(
            output, lse, world_size, scatter_dim, group, *, defer_combine, owned=owned, rank=rank, flash=flash
        ):
            assert (world_size, scatter_dim, group, defer_combine) == (size, 1, "cpu-dcp", True)
            runtime.events.append(("exchange", runtime.active_stream[0]))
            assert output.shape == (tokens, 64, 512) and lse.shape == (tokens, 64, 1)
            expected_output, expected_lse = partials[rank]
            torch.testing.assert_close(output, expected_output.transpose(0, 1))
            torch.testing.assert_close(lse, expected_lse.transpose(0, 1).unsqueeze(-1))
            return (
                [out[owned].transpose(0, 1) for out, _ in partials],
                [stats[owned].transpose(0, 1).unsqueeze(-1) for _, stats in partials],
            )

        monkeypatch.setattr(torch.ops.vllm, "dcp_a2a_fused", exchange, raising=False)

        def combine(recv, head_dim, *, scatter_dim, local_output=None, local_lse=None):
            assert runtime.events[-1] == (("wait", "main", "attn") if split else ("exchange", "main"))
            assert (local_output is not None) == split
            assert (local_lse is not None) == split
            runtime.events.append(("merge", runtime.active_stream[0]))
            return combine_outputs(
                recv, head_dim, scatter_dim=scatter_dim, local_output=local_output, local_lse=local_lse
            )

        combined = Mock(side_effect=combine)
        runtime.scope["fused_dcp_lse_combine"] = combined

        def project(latent):
            runtime.events.append(("project", runtime.active_stream[0]))
            return latent

        impl = runtime.bind(
            SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                num_kv_heads=1,
                num_heads=local_heads,
                dcp_size=size,
                dcp_rank=rank,
                dcp_group=SimpleNamespace(unique_name="cpu-dcp"),
                _v_up_proj_batch_major=Mock(side_effect=project),
            )
        )
        preprocessed = SimpleNamespace(
            ql_nope=query[..., :512],
            q_pe=query[..., 512:],
            current_k_nope=current_kv[..., :512] if split else None,
            current_k_pe=current_kv[..., 512:] if split else None,
        )
        cache, backing, snapshot = caches[rank]
        runtime.events.clear()
        runtime.recorded_streams.clear()
        runtime.attention_op.reset_mock()
        runtime.current_query_storages.clear()
        if split:
            runtime.current_query_storages.add(flash.current.query.untyped_storage().data_ptr())
        # Production external metadata has no FIA CPU length mirrors.
        decode = SimpleNamespace(cp_history_seq_len=None, actual_seq_lengths_q=None)
        result = runtime.forward(impl, preprocessed, cache, SimpleNamespace(external_flashmla=flash, decode=decode))
        impl._forward_decode_split_attention.assert_called_once()
        impl._run_dcp_mtp_split_attention_op.assert_not_called()
        combined.assert_called_once()
        impl._v_up_proj_batch_major.assert_called_once()
        assert runtime.attention_op.call_count == (2 if split else 1)
        if split:
            assert runtime.events == [
                ("history", "main"),
                ("scatter", "main"),
                ("query_copy", "main"),
                ("query_copy", "main"),
                ("record", "main"),
                ("wait", "attn", "main"),
                ("current", "attn"),
                ("record", "attn"),
                ("exchange", "main"),
                ("wait", "main", "attn"),
                ("merge", "main"),
                ("project", "main"),
            ]
            for tensor in (
                flash.current.query,
                flash.current.cache,
                flash.current.schedule,
                flash.current.block_table,
                flash.current.slots,
                flash.current.attn_mask,
                flash.cu,
                flash.used_q,
                flash.token_live,
            ):
                assert any(held is tensor and stream.name == "attn" for held, stream in runtime.recorded_streams)
            outputs_on_main = [tensor for tensor, stream in runtime.recorded_streams if stream.name == "main"]
            assert [tuple(tensor.shape) for tensor in outputs_on_main] == [
                (tokens, local_heads, 512),
                (tokens, local_heads, 1),
            ]
        else:
            assert runtime.events == [("history", "main"), ("exchange", "main"), ("merge", "main"), ("project", "main")]
            assert not runtime.recorded_streams
        tolerance = 8e-4 if dtype == torch.bfloat16 else 1e-4
        # Unused query outputs are outside the attention result contract.
        torch.testing.assert_close(
            result[flash.token_live].double(), expected[flash.token_live, owned], atol=tolerance, rtol=0.02
        )
        assert torch.equal(backing, snapshot)  # Forward never repacks or writes persistent history.
        cache_call = runtime.attention_op.call_args_list[-2 if split else -1]
        assert cache_call.args[1] is cache
    runtime.scope["get_graph_params"].assert_not_called()
    runtime.scope["torch_npu"]._npu_fused_infer_attention_score_get_max_workspace.assert_not_called()


@pytest.mark.parametrize("draft", [False, True])
def test_external_dcp_graph_skips_fia_parameter_updates(draft):
    metadata = {"flash": SimpleNamespace(decode=object(), external_flashmla=object())}
    scope = {
        "_EXTRA_CTX": SimpleNamespace(is_draft_model=draft, is_draft_model_prefill=False),
        "get_graph_params": lambda: object(),
        "get_draft_graph_params": lambda: object(),
    }
    update = load_method(
        "vllm_ascend/attention/context_parallel/mla_cp.py", "AscendMlaDCPImpl", "update_graph_params", scope
    )
    # No torch.npu stream or FIA graph registry access is possible in this scope.
    update(None, SimpleNamespace(attn_metadata=metadata), 4, draft_attn_metadatas=[metadata])


@pytest.mark.parametrize("bad_lse", ["layout", "dtype"])
def test_dcp_rejects_incompatible_lse_contract(runtime, bad_lse):
    flash = runtime.builder().build(common([1], [1]), 1, 1, False)
    output = torch.zeros(64, 1, 512, dtype=torch.float16)
    lse = torch.zeros(1, 64) if bad_lse == "layout" else torch.zeros(64, 1, dtype=torch.float16)
    runtime.attention_op.side_effect = None
    runtime.attention_op.return_value = output, lse
    with pytest.raises(ValueError, match="softmax_lse"):
        flash.adapter.attention(
            flash.query,
            torch.zeros(1, 128, 1, 576, dtype=torch.float16),
            block_table=flash.block_table,
            cache_seqlens=flash.cache_lens,
            cu_seqlens_q=flash.cu,
            seqused_q=flash.used_q,
            metadata=flash.schedule,
        )


def test_causal_target_and_noncausal_draft_do_not_share_schedules(runtime):
    builder = runtime.builder()
    target = builder.build(common([9], [2], tokens=4), 1, 2, False, retain_for_graph=True)
    draft = builder.build(common([9], [2], tokens=4, causal=False), 1, 2, False, retain_for_graph=True)
    assert target is not draft
    assert target.schedule.data_ptr() != draft.schedule.data_ptr()
    assert target.current is not None and draft.current is None
    assert target.cache_lens[0] == 1 and draft.cache_lens[0] == 2
