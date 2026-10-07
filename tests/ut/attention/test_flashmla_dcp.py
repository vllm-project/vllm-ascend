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
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, relative):
        return install(name, **runpy.run_path(str(ROOT / relative)))

    install("vllm_ascend")
    utils = install("vllm_ascend.attention.utils")
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
    tasks = load("vllm_ascend.worker.device_metadata", "vllm_ascend/worker/device_metadata.py")
    api = load("vllm_ascend.attention.flashmla", "vllm_ascend/attention/flashmla.py")

    def schedule(lengths, *, cu_seqlens_q, seqused_q, **kwargs):
        return torch.cat((lengths, cu_seqlens_q, seqused_q))

    metadata_op = Mock(side_effect=schedule)
    attention_op = Mock(side_effect=paged_attention)
    monkeypatch.setattr(
        api.FlashMLAAdapter,
        "load",
        classmethod(lambda cls, config: cls(config, attention_op, metadata_op)),
    )

    def rope(positions):
        values = positions[:, None, None, None].expand(-1, 1, 1, 64).to(torch.bfloat16)
        return values, -values

    install("vllm_ascend.ops.rotary_embedding", get_cos_and_sin_mla=rope)
    metadata = load("flashmla_dcp_metadata_under_test", "vllm_ascend/attention/flashmla_metadata.py")

    def builder(*, size=8, rank=0, interleave=1, heads=8, dtype=torch.float16, use_rope=False):
        impl = SimpleNamespace(
            num_heads=heads,
            scale=0.125,
            dtype=dtype,
            use_mla_rope=use_rope,
            dcp_size=size,
            dcp_rank=rank,
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave)),
        )
        return metadata.FlashMLAMetadataBuilder(
            impl, torch.device("cpu"), 8, torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1)
        )

    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_scatter_pa_kv_cache=scatter_current),
        "record_attention_compute_start": Mock(),
        "fused_dcp_lse_combine": combine_outputs,
    }
    forward = load_method(
        "vllm_ascend/attention/context_parallel/mla_cp.py", "AscendMlaDCPImpl", "_forward_external_flashmla", scope
    )
    return SimpleNamespace(
        builder=builder, api=api, metadata_op=metadata_op, attention_op=attention_op, forward=forward, tasks=tasks
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


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("interleave", [1, 4, 128])
@pytest.mark.parametrize("rank", [0, 1, 7])
def test_visibility_uses_global_history_before_sharding(runtime, causal, interleave, rank):
    builder = runtime.builder(rank=rank, interleave=interleave)
    lengths, queries = [1031, 3, 132, 0], [4, 3, 2, 1]
    inputs = common(lengths, queries, tokens=16, causal=causal)
    flash = builder.build(inputs, 4, 10, False)
    expected = []
    for length, query in zip(lengths, queries):
        visible = max(0, length - query) if causal else length
        expected.append(sum(token // interleave % 8 == rank for token in range(visible)))
    assert flash.cache_lens[:4].tolist() == expected
    assert not flash.cache_lens[4:].any()
    assert flash.query.shape == (16, 64, 576)
    assert flash.adapter.config.mask_mode == 0
    assert flash.adapter.config.return_softmax_lse
    torch.testing.assert_close(flash.schedule, torch.cat((flash.cache_lens, flash.cu, flash.used_q)))
    if causal:
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


def test_two_schedules_refresh_after_rejection_without_changing_graph_addresses(runtime, monkeypatch):
    builder = runtime.builder(rank=3, interleave=4, use_rope=True)
    inputs = common([37, 15], [4, 3], tokens=16)
    flash = builder.build(inputs, 2, 7, False, retain_for_graph=True)
    pointers = addresses(flash)
    previous = flash.schedule.clone()
    previous_current = flash.current.schedule.clone()
    builder.defer = True
    same = builder.build(inputs, 2, 7, False)
    assert same is flash
    (task,) = builder.take_tasks()
    assert builder.take_tasks() == ()
    # Simulate device rejection correction after metadata construction.
    inputs.seq_lens.copy_(torch.tensor([35, 0]))
    inputs.query_start_loc.copy_(torch.tensor([0, 2, 3]))
    torch.testing.assert_close(flash.schedule, previous)
    torch.testing.assert_close(flash.current.schedule, previous_current)

    def forbidden(*args, **kwargs):
        pytest.fail("device lengths must not be copied to the host")

    with monkeypatch.context() as patch:
        for method in ("item", "tolist", "cpu", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        task.run()
    assert addresses(flash) == pointers
    assert flash.cache_lens[:2].tolist() == [4, 0]
    assert flash.current.slots.tolist() == [0, 1] + [-1] * 14
    assert not torch.equal(flash.schedule, previous)
    assert not torch.equal(flash.current.schedule, previous_current)


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


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("interleave", [1, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_matches_dense_with_strided_history_and_empty_ranks(runtime, monkeypatch, causal, interleave, dtype):
    torch.manual_seed(41)
    size, local_heads, tokens = 8, 8, 5
    queries = [3, 1]
    histories = [257, 0]
    lengths = [history + query for history, query in zip(histories, queries)]
    keys = [torch.randn(length, 576, dtype=dtype) * 0.2 for length in lengths]
    query = torch.randn(tokens, size * local_heads, 576, dtype=dtype) * 0.2
    current_kv = torch.zeros(tokens, 1, 576, dtype=dtype)
    current_kv[:3, 0] = keys[0][-3:]
    current_kv[3, 0] = keys[1][-1]
    flashes, caches, partials = [], [], []
    for rank in range(size):
        builder = runtime.builder(rank=rank, interleave=interleave, dtype=dtype)
        inputs = common(lengths, queries, tokens=tokens, causal=causal)
        inputs.block_table_tensor.copy_(torch.tensor([[1, 3, 5], [2, 4, 6]]))
        flash = builder.build(inputs, 2, 4, False)
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
            assert output.shape == (tokens, 64, 512) and lse.shape == (tokens, 64, 1)
            expected_output, expected_lse = partials[rank]
            expected_output = expected_output.masked_fill(~flash.token_live[None, :, None], 0)
            expected_lse = expected_lse.masked_fill(~flash.token_live[None, :], -torch.inf)
            torch.testing.assert_close(output, expected_output.transpose(0, 1))
            torch.testing.assert_close(lse, expected_lse.transpose(0, 1).unsqueeze(-1))
            return (
                [out[owned].transpose(0, 1) for out, _ in partials],
                [stats[owned].transpose(0, 1).unsqueeze(-1) for _, stats in partials],
            )

        monkeypatch.setattr(torch.ops.vllm, "dcp_a2a_fused", exchange, raising=False)
        impl = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_heads=local_heads,
            dcp_size=size,
            dcp_rank=rank,
            dcp_group=SimpleNamespace(unique_name="cpu-dcp"),
            _v_up_proj_batch_major=lambda latent: latent,
        )
        preprocessed = SimpleNamespace(
            ql_nope=query[..., :512],
            q_pe=query[..., 512:],
            current_k_nope=current_kv[..., :512] if causal else None,
            current_k_pe=current_kv[..., 512:] if causal else None,
        )
        cache, backing, snapshot = caches[rank]
        result = runtime.forward(impl, preprocessed, cache, SimpleNamespace(external_flashmla=flash))
        tolerance = 8e-4 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(result.double(), expected[:, owned], atol=tolerance, rtol=0.02)
        assert torch.equal(result[-1], torch.zeros_like(result[-1]))
        assert torch.equal(backing, snapshot)  # Forward never repacks or writes persistent history.
        cache_call = runtime.attention_op.call_args_list[-2 if causal else -1]
        assert cache_call.args[1] is cache


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


@pytest.mark.parametrize(
    "heads,size,expected", [(8, 1, True), (8, 8, True), (12, 8, True), (8, 2, False), (12, 4, False)]
)
def test_selection_checks_both_local_and_gathered_heads(runtime, heads, size, expected):
    scope = {
        "torch": torch,
        "HardwareCapability": SimpleNamespace(MLA_FLASH="mla"),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "MLA_FLASH_SUPPORTED_Q_HEADS": runtime.api.MLA_FLASH_SUPPORTED_Q_HEADS,
        "FLASHMLA_V_DIM": 512,
        "FLASHMLA_QK_DIM": 576,
        "supports_component_major_mla_pd": lambda _: True,
        "enable_sfa": lambda _: False,
        "get_flashmla_ops": lambda: (Mock(), Mock()),
    }
    select = load_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "_can_use_flashmla", scope)
    impl = SimpleNamespace(
        num_heads=heads,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=size),
        ),
    )
    assert select(impl) is expected
    impl.pcp_enabled = True
    assert not select(impl)


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


def test_current_kv_requirement_uses_external_split_metadata(runtime):
    scope = {"AscendMLAMetadata": object}
    requires = load_method(
        "vllm_ascend/attention/context_parallel/mla_cp.py", "AscendMlaDCPImpl", "_decode_requires_current_kv", scope
    )
    fallback = Mock(return_value=True)
    impl = SimpleNamespace(_use_history_current_split_decode=fallback)
    for causal in (True, False):
        flash = runtime.builder().build(common([9], [2], causal=causal), 1, 2, False)
        assert requires(impl, SimpleNamespace(external_flashmla=flash)) is causal
    fallback.assert_not_called()
    assert requires(impl, SimpleNamespace(external_flashmla=None))
    fallback.assert_called_once()
