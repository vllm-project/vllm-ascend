# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from vllm.config import ParallelConfig
from vllm.v1.attention.ops import metadata as upstream_metadata_ops
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker import cp_utils
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.models.qwen4_exp import model, qsa
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


def test_replicated_qsa_draft_passes_pcp_compatibility(monkeypatch) -> None:
    impl = qsa.AscendQSAImpl(2, 256, 0.0625, 1, "auto")
    layers = {"mtp.layers.0.self_attn": SimpleNamespace(impl=impl)}
    monkeypatch.setattr(cp_utils, "get_layers_from_vllm_config", lambda *_: layers)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2, decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        speculative_config=object(),
    )

    cp_utils.check_attention_cp_compatibility(config, target_layer_names=set())


def test_main_and_draft_qkv_projection_use_upstream_path() -> None:
    assert "_project_qkv_gate" not in qsa.AscendQwen4ExpQSAAttention.__dict__
    assert qsa.AscendQwen4ExpQSAAttention._project_qkv_gate is qsa.upstream_qsa.Qwen4ExpQSAAttention._project_qkv_gate


def test_indexer_accepts_main_packed_inputs_without_fused_preparation() -> None:
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    metadata = SimpleNamespace(num_actual_tokens=2)
    layer._metadata = Mock(return_value=(metadata, metadata))
    layer.skip_topk = False
    query = torch.empty((2, 4, 128))
    keys = torch.empty((2, 1, 128))
    layer.project_qk = Mock(return_value=(query, keys))
    layer._update_and_compress = Mock()
    selected = torch.zeros((2, 3), dtype=torch.int32)
    layer._select = Mock(return_value=selected)
    # Deterministic input: assert_close treats NaN as unequal to itself, so
    # uninitialized memory would make the slice comparison below flaky.
    hidden_states = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    positions = torch.arange(4)
    gate = torch.full((2, 2, 4), 7.0)

    indices, main_outputs = layer(
        hidden_states,
        positions,
        selected,
        attn=SimpleNamespace(use_fused_qsa_prepare=True),
        qkv=torch.empty((2, 24)),
        slot_mapping=torch.arange(2),
        gate_out=gate,
    )

    assert indices is selected and main_outputs is None
    projected_hidden, projected_positions = layer.project_qk.call_args.args
    torch.testing.assert_close(projected_hidden, hidden_states[:2])
    torch.testing.assert_close(projected_positions, positions[:2])
    layer._update_and_compress.assert_called_once_with(keys, projected_positions, metadata, metadata)
    layer._select.assert_called_once_with(query, metadata, selected)
    torch.testing.assert_close(gate, torch.full_like(gate, 7.0))


@pytest.mark.parametrize("axes", [1, 3])
def test_indexer_rotary_gathers_clamped_cache_before_kernel(monkeypatch, axes) -> None:
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    cache = torch.arange(5 * 64, dtype=torch.float32).reshape(5, 64)
    layer.rotary_emb = SimpleNamespace(
        _match_cos_sin_cache_dtype=Mock(return_value=cache),
        mrope_section=[11, 11, 10],
        rotary_dim=64,
    )
    positions = torch.tensor([-1, 2, 8])
    if axes == 3:
        positions = torch.stack((positions, positions + 1, positions - 1))
    tensor = torch.empty((3, 4, 128), dtype=torch.bfloat16)
    norm = SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6)
    kernel = Mock(return_value=tensor)
    monkeypatch.setattr(torch.ops.vllm, "triton_qsa_rmsnorm_mrope", kernel)

    assert layer._fused_norm_rope(tensor, norm, positions) is tensor

    arguments = kernel.call_args.kwargs
    torch.testing.assert_close(arguments["cos_sin"], cache[positions.clamp(0, 4)])
    assert arguments["tensor"] is tensor
    assert arguments["weight"] is norm.weight
    assert "positions" not in arguments


def test_main_falls_back_when_fused_projection_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value=("q", "k", "v", "gate"))
    monkeypatch.setattr(
        qsa.upstream_qsa.Qwen4ExpQSAAttention,
        "_project_qkv_gate",
        upstream_projection,
    )
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer._main_fused_norm_rope_eligible = Mock(return_value=False)
    qkv = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer._project_qkv_gate(qkv, positions) == ("q", "k", "v", "gate")
    upstream_projection.assert_called_once_with(qkv, positions)


def test_indexer_query_falls_back_when_fused_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value=("query", "key"))
    monkeypatch.setattr(
        qsa.upstream_indexer.QSAIndexer,
        "project_qk",
        upstream_projection,
    )
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_n_heads = 2
    layer.index_head_dim = 4
    layer.index_kv_heads = 1
    layer.index_qk_proj = Mock(return_value=(torch.empty((2, 12)), None))
    layer._fused_norm_rope_eligible = Mock(return_value=False)
    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer.project_qk(hidden_states, positions) == ("query", "key")
    upstream_projection.assert_called_once_with(hidden_states, positions)


def test_indexer_compressed_key_falls_back_when_fused_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value="keys")
    monkeypatch.setattr(
        qsa.upstream_indexer.QSAIndexer,
        "normalize_compressed_keys",
        upstream_projection,
    )
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_head_dim = 4
    layer.rotary_emb = SimpleNamespace(mrope_section=[1, 1, 0])
    layer._fused_norm_rope_eligible = Mock(return_value=False)
    pooled = torch.empty((2, 4))
    first_positions = torch.arange(6).reshape(2, 3)

    assert layer.normalize_compressed_keys(pooled, first_positions) == "keys"
    upstream_projection.assert_called_once_with(pooled, first_positions)


def test_text_only_indexer_uses_fused_rope_without_mrope_sections(monkeypatch) -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_n_heads = 4
    indexer.index_kv_heads = 1
    indexer.index_head_dim = 128
    norm = SimpleNamespace(weight=torch.zeros(128, dtype=torch.bfloat16), variance_epsilon=1e-6)
    indexer.q_layernorm = norm
    indexer.k_layernorm = norm
    cache = torch.arange(256, dtype=torch.bfloat16).reshape(4, 64)
    indexer.rotary_emb = SimpleNamespace(rotary_dim=64, is_neox_style=True, _match_cos_sin_cache_dtype=lambda _: cache)
    projection = torch.ones(2, 5 * 128, dtype=torch.bfloat16)
    indexer.index_qk_proj = lambda _: (projection, None)
    fused = Mock(side_effect=lambda **kwargs: kwargs["tensor"])
    monkeypatch.setattr(torch.ops.vllm, "triton_qsa_rmsnorm_mrope", fused, raising=False)
    positions = torch.tensor([1, 3])

    queries, keys = indexer.project_qk(torch.empty(2, 16), positions)
    indexer.normalize_compressed_keys(keys, positions[:, None])

    assert queries.shape == (2, 4, 128)
    assert keys.shape == (2, 1, 128)
    assert fused.call_count == 2
    for call in fused.call_args_list:
        assert call.kwargs["mrope_section"] == []
        torch.testing.assert_close(call.kwargs["cos_sin"], cache[positions])


@pytest.mark.parametrize("pcp", [False, True])
def test_fused_prepare_upstream_abi_materializes_qkv_and_gate(monkeypatch, pcp) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.head_dim = 4
    layer.kv_cache = (torch.empty(1), torch.empty(1))
    layer.topk_indices_buffer = torch.empty((2, 3), dtype=torch.int32)

    main_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.tensor([0, 1]),
    )
    side_prefix = "model.layers.0.self_attn.indexer.raw_key_cache"
    side_metadata = SimpleNamespace(
        num_actual_tokens=2,
        token_to_req=torch.tensor([0, 0]),
        logical_positions=torch.tensor([0, 1]),
        seq_lens=torch.tensor([2]),
    )
    if pcp:
        side_metadata.pcp_context = SimpleNamespace(
            local_num_tokens=2,
            global_batch=SimpleNamespace(num_tokens=4),
            hidden_restore_idx=torch.tensor([0, 2, 3, 1]),
        )
        monkeypatch.setattr(qsa, "get_pcp_group", lambda: object())
        gather = Mock(side_effect=lambda rows, *args, **kwargs: (torch.cat((rows, rows + 100)), None))
        monkeypatch.setattr(qsa, "all_gather_async", gather)
        main_metadata.slot_mapping = torch.tensor([4, 5, 6, 7])
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                layer.layer_name: main_metadata,
                side_prefix: side_metadata,
            }
        ),
    )

    selected = torch.zeros((2, 3), dtype=torch.int32)
    # Main's indexer returns (selection, prepared main outputs).
    indexer = Mock(return_value=(selected, None))
    indexer.output_width = 3
    indexer.raw_key_cache = SimpleNamespace(prefix=side_prefix)
    layer.indexer = indexer
    layer.impl = Mock()

    projected_q = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    projected_k = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    projected_v = projected_k + 10
    projected_gate = projected_q + 20
    layer._project_qkv_gate = Mock(return_value=(projected_q, projected_k, projected_v, projected_gate))

    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)
    qkv = torch.empty((2, 24))
    output = torch.empty((2, 2, 4))
    gate = torch.empty_like(output)

    layer._run_qsa(
        hidden_states,
        positions,
        None,
        None,
        None,
        output,
        qkv=qkv,
        gate=gate,
    )

    layer._project_qkv_gate.assert_called_once_with(qkv, positions)
    torch.testing.assert_close(gate, projected_gate.view_as(gate))
    layer.impl.do_kv_cache_update.assert_called_once()
    update_args = layer.impl.do_kv_cache_update.call_args.args
    expected_key = projected_k.view(2, 1, 4)
    expected_value = projected_v.view(2, 1, 4)
    if pcp:
        order = [0, 2, 3, 1]
        expected_key = torch.cat((expected_key, expected_key + 100))[order]
        expected_value = torch.cat((expected_value, expected_value + 100))[order]
    torch.testing.assert_close(update_args[1], expected_key)
    torch.testing.assert_close(update_args[2], expected_value)
    torch.testing.assert_close(update_args[4], main_metadata.slot_mapping)
    assert layer.impl.forward_qsa.call_count + layer.impl.forward.call_count == 1
    forward_call = layer.impl.forward
    forward_args = forward_call.call_args.args
    torch.testing.assert_close(forward_args[1], projected_q.view(2, 2, 4))


def test_fused_prepare_zeros_gate_without_runtime_metadata(monkeypatch) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    output = torch.ones((2, 2, 4))
    gate = torch.ones_like(output)

    layer._run_qsa(
        torch.empty((2, 8)),
        torch.arange(2),
        None,
        None,
        None,
        output,
        qkv=torch.empty((2, 24)),
        gate=gate,
    )

    assert not output.any()
    assert not gate.any()


def test_pcp_indexer_compresses_complete_groups_across_ranks(monkeypatch) -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.compress_ratio = 4
    indexer.raw_key_cache = SimpleNamespace(key_cache=torch.empty(1), rope_position_cache=None)
    indexer.compressed_key_cache = SimpleNamespace(kv_cache=torch.empty(1))
    indexer.normalize_compressed_keys = lambda keys, positions: keys
    global_keys = torch.tensor([10, 11, 12, 13]).view(4, 1, 1)
    raw_write = SimpleNamespace(
        num_actual_tokens=4,
        block_table=torch.tensor([[1]]),
        token_to_req=torch.tensor([0, 0, 0, 0]),
        query_start_loc=torch.tensor([0, 4]),
        logical_positions=torch.arange(4),
        slot_mapping=torch.tensor([4, 5, 6, 7]),
    )
    compressed_write = SimpleNamespace(slot_mapping=torch.tensor([-1, -1, -1, 8]))
    local_keys = global_keys[[0, 3, 0]]
    gathered = torch.cat((local_keys, global_keys[[1, 2, 0]]))
    context = SimpleNamespace(
        local_num_tokens=3, global_batch=SimpleNamespace(num_tokens=4), hidden_restore_idx=torch.tensor([0, 3, 4, 1, 0])
    )
    raw = SimpleNamespace(pcp_context=context, write_metadata=raw_write)
    compressed = SimpleNamespace(write_metadata=compressed_write)
    compress = Mock(return_value=(torch.empty(1), torch.empty(1)))
    store = Mock()
    group = object()
    all_gather = Mock(return_value=(gathered, None))
    monkeypatch.setattr(qsa, "get_pcp_group", lambda: group)
    monkeypatch.setattr(qsa, "all_gather_async", all_gather)
    monkeypatch.setattr(qsa, "_compress_qsa_groups", compress)
    monkeypatch.setattr(qsa, "_store_qsa_cache_rows", store)

    indexer._update_and_compress(local_keys, torch.tensor([0, 3, 0]), raw, compressed)

    all_gather.assert_called_once_with(local_keys, group, async_op=False)
    compression_args = compress.call_args.args
    torch.testing.assert_close(compression_args[0], global_keys)
    torch.testing.assert_close(compression_args[6], raw_write.logical_positions)
    torch.testing.assert_close(compression_args[7], compressed_write.slot_mapping)
    torch.testing.assert_close(store.call_args_list[1].args[1], raw_write.slot_mapping)


def test_pcp_indexer_replaces_registered_cache_owners() -> None:
    static_context: dict[str, object] = {}
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=static_context))
    cache_args = dict(
        vllm_config=config,
        cache_config=SimpleNamespace(block_size=128),
        dtype=torch.bfloat16,
        head_size=128,
        compress_ratio=4,
    )
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.raw_key_cache = qsa.QSARawKeyCache(**cache_args, prefix="raw", cache_rope_positions=True)
    indexer.compressed_key_cache = qsa.qsa_indexer.QSACompressedKeyCache(**cache_args, prefix="compressed")
    raw, compressed = indexer.raw_key_cache, indexer.compressed_key_cache

    indexer.configure_pcp_cache_backends(config)

    assert static_context == {"raw": indexer.raw_key_cache, "compressed": indexer.compressed_key_cache}
    assert indexer.raw_key_cache is not raw
    assert indexer.compressed_key_cache is not compressed
    assert indexer.raw_key_cache.cache_rope_positions
    for original, cache in ((raw, indexer.raw_key_cache), (compressed, indexer.compressed_key_cache)):
        assert cache.head_size == original.head_size
        assert cache.compress_ratio == original.compress_ratio
        assert cache.get_attn_backend().supports_pcp()
        assert cache.get_attn_backend().get_builder_cls() is qsa.AscendPCPQSAIndexerMetadataBuilder


def test_pcp_indexer_global_write_invalidates_the_local_layout(monkeypatch) -> None:
    local_starts = torch.tensor([0, 2], dtype=torch.int32)
    global_starts = torch.tensor([0, 5], dtype=torch.int32)
    table = torch.tensor([[0]], dtype=torch.int32)
    context = SimpleNamespace(
        local_token_indices=torch.arange(2),
        global_batch=SimpleNamespace(
            num_reqs=1,
            num_tokens=5,
            query_start_loc=global_starts,
            query_start_loc_np=global_starts.numpy(),
            seq_lens=global_starts[1:],
            positions=torch.arange(5),
            num_scheduled_tokens=global_starts[1:].numpy(),
        ),
        global_block_tables=table.unsqueeze(0),
        global_slot_mappings=torch.arange(5).unsqueeze(0),
    )
    common = AscendCommonAttentionMetadata(
        query_start_loc=local_starts,
        query_start_loc_cpu=local_starts,
        seq_lens=local_starts[1:],
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=5,
        block_table_tensor=table,
        slot_mapping=torch.arange(2),
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    local_mapping = torch.zeros(2, dtype=torch.int32)
    common._token_to_req_indices_cache = local_mapping
    common._num_computed_tokens_cache = torch.tensor([0])
    monkeypatch.setattr(qsa.QSAIndexerMetadataBuilder, "build", lambda *_: SimpleNamespace())
    builder = qsa.AscendPCPQSAIndexerMetadataBuilder.__new__(qsa.AscendPCPQSAIndexerMetadataBuilder)
    builder.write_builder = Mock()

    builder.build(0, common)

    global_common = builder.write_builder.build.call_args.args[1]
    assert global_common.num_actual_tokens == 5
    assert global_common._token_to_req_indices_cache is None
    assert global_common._num_computed_tokens_cache is None
    assert common._token_to_req_indices_cache is local_mapping


@pytest.mark.parametrize("local_live_tokens", [0, 1, 2])
def test_pcp_qsa_full_graph_keeps_capture_and_replay_extents(monkeypatch, local_live_tokens) -> None:
    local_table = torch.tensor([[3], [0]], dtype=torch.int32)
    global_table = torch.tensor([[3], [4], [0], [0]], dtype=torch.int32)
    global_slots = torch.full((1, 4), -1, dtype=torch.long)
    restore = torch.arange(4)
    main_builder = object.__new__(qsa.AscendQSAMetadataBuilder)
    kv_cache_spec = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, tokens_per_state=4
    )
    indexer_builder = qsa.AscendPCPQSAIndexerMetadataBuilder(
        kv_cache_spec,
        ["indexer"],
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=4),
            speculative_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        torch.device("cpu"),
    )

    def token_to_req(query_starts, out, num_mapped, num_tokens):
        tokens = torch.arange(num_tokens, dtype=query_starts.dtype)
        requests = torch.searchsorted(query_starts[1:], tokens, right=True)
        out[:num_tokens].copy_(torch.where(tokens < num_mapped, requests, 0))
        return out[:num_tokens]

    monkeypatch.setattr(upstream_metadata_ops, "compute_token_to_req_indices", token_to_req)
    monkeypatch.setattr(
        qsa.upstream_qsa.Qwen4ExpQSAMetadataBuilder,
        "build",
        lambda _, prefix_len, common, fast_build=False: SimpleNamespace(num_actual_tokens=common.num_actual_tokens),
    )
    monkeypatch.setattr(qsa, "get_pcp_group", lambda: object())
    monkeypatch.setattr(qsa, "all_gather_async", lambda rows, *_args, **_kwargs: (torch.cat((rows, rows + 2)), None))
    groups = [
        AttentionGroup(
            backend=backend,
            layer_names=[name],
            kv_cache_spec=kv_cache_spec,
            kv_cache_group_id=0,
            metadata_builders=[builder],
        )
        for name, backend, builder in (
            ("main", qsa.AscendQSABackend, main_builder),
            ("indexer", qsa.AscendPCPQSAIndexerBackend, indexer_builder),
        )
    ]
    cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=None)])

    for capture in (True, False):
        global_live_tokens = 4 if capture else max(local_live_tokens, 1)
        local_live = 2 if capture else local_live_tokens
        global_starts = torch.arange(5, dtype=torch.int32)
        local_starts = torch.arange(3, dtype=torch.int32)
        if not capture:
            global_slots[0, :global_live_tokens] = torch.arange(10, 10 + global_live_tokens)
            restore[:global_live_tokens] = torch.arange(2, 2 + global_live_tokens)
            restore[global_live_tokens:] = 0
        global_batch = SimpleNamespace(
            num_reqs=global_live_tokens,
            num_reqs_after_padding=4,
            num_tokens=global_live_tokens,
            num_tokens_after_padding=4,
            query_start_loc=global_starts,
            query_start_loc_np=global_starts.numpy(),
            seq_lens=(torch.arange(4) < global_live_tokens).to(torch.int32) * 8,
            positions=torch.arange(4),
            num_scheduled_tokens=torch.ones(global_live_tokens, dtype=torch.int32).numpy(),
        )
        context = AscendPCPAttentionContext(
            local_num_tokens=2,
            local_token_indices=None if capture else torch.arange(local_live),
            hidden_restore_idx=restore,
            global_batch=cast(AscendInputBatch, global_batch),
            global_block_tables=(global_table,),
            global_slot_mappings=global_slots,
        )
        metadata = build_attn_metadata(
            attn_groups=[groups],
            query_start_loc_gpu=local_starts,
            query_start_loc_cpu=local_starts,
            seq_lens=(torch.arange(2) < local_live).to(torch.int32) * 8,
            num_reqs=2,
            num_actual_reqs=max(local_live, 1),
            num_tokens=2,
            num_actual_tokens=local_live,
            max_query_len=1,
            max_seq_len=8,
            block_tables=(local_table,),
            slot_mappings=torch.full((1, 4), -1, dtype=torch.long),
            is_prefilling=torch.zeros(2, dtype=torch.bool),
            kv_cache_config=cache_config,
            pcp_context=context,
            full_graph_mode=not capture,
            for_cudagraph_capture=capture,
            skip_ring_state_update=False,
        )
        main, local = metadata["main"], metadata["indexer"]
        write = local.write_metadata
        assert main.num_actual_tokens == local.num_actual_tokens == 2
        assert main.qsa_request_plan is None
        assert local.pcp_context is context
        assert local.logical_positions.tolist() == [7] * local_live + [-1] * (2 - local_live)
        assert write.num_actual_tokens == 4
        assert write.query_start_loc.shape == (5,)
        assert write.logical_positions.tolist() == [7] * global_live_tokens + [-1] * (4 - global_live_tokens)
        assert main.slot_mapping.data_ptr() == global_slots.data_ptr()
        assert (write.slot_mapping[global_live_tokens:] == -1).all()
        assert (write.slot_mapping[:global_live_tokens] >= 0).tolist() == [not capture] * global_live_tokens
        if capture:
            capture_buffers = (local.logical_positions, write.slot_mapping)
        else:
            assert local.logical_positions.data_ptr() == capture_buffers[0].data_ptr()
            assert write.slot_mapping.data_ptr() == capture_buffers[1].data_ptr()

        gathered = qsa._gather_pcp_cache_rows(torch.arange(2), context, write.num_actual_tokens)
        expected = (
            list(range(4)) if capture else list(range(2, 2 + global_live_tokens)) + [0] * (4 - global_live_tokens)
        )
        assert gathered.tolist() == expected


@pytest.mark.parametrize("num_local_tokens", [0, 2])
def test_pcp_indexer_projects_padding_for_collective_but_selects_only_live_queries(num_local_tokens) -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.skip_topk = False
    raw = SimpleNamespace(num_actual_tokens=num_local_tokens, pcp_context=object())
    indexer._metadata = lambda: (raw, raw)
    indexer.project_qk = lambda hidden, positions: (hidden, hidden)
    indexer._update_and_compress = Mock()
    indexer._select = lambda query, metadata, out: query
    hidden = torch.arange(3).view(3, 1)
    positions = torch.arange(3)

    selected, _ = indexer(hidden, positions, attn=object())

    key_rows, write_positions = indexer._update_and_compress.call_args.args[:2]
    torch.testing.assert_close(key_rows, hidden)
    torch.testing.assert_close(write_positions, positions)
    torch.testing.assert_close(selected, hidden[:num_local_tokens])


def test_qsa_kv_store_keeps_page_stride_with_multiple_heads(monkeypatch) -> None:
    backing = torch.zeros(2, 2, 4, 2, 3)
    cache = backing[:, 0]
    slots = torch.tensor([5])
    rows = torch.arange(1, 7, dtype=cache.dtype).view(1, 2, 3)

    def store(cache_view, slot_mapping, flat_rows):
        cache_view[slot_mapping // 4, slot_mapping % 4, 0] = flat_rows

    monkeypatch.setattr(qsa, "qsa_store_cache_rows_fused", store)
    qsa._store_qsa_cache_rows(cache, slots, rows)

    torch.testing.assert_close(cache[1, 1], rows[0])
    assert not backing[:, 1].any()


@pytest.mark.parametrize(
    "enable_dbo,ubatch_size,pcp_size,expected_sharing",
    [
        pytest.param(False, 0, 1, True, id="ordinary-decode"),
        pytest.param(False, 2, 1, False, id="explicit-ubatching"),
        pytest.param(True, 0, 1, False, id="dbo"),
        pytest.param(False, 0, 2, False, id="pcp"),
    ],
)
def test_target_topk_sharing_respects_parallel_modes(monkeypatch, enable_dbo, ubatch_size, pcp_size, expected_sharing):
    # Bypass unrelated distributed config validation while retaining the real
    # upstream use_ubatching property used by the target constructor.
    parallel_config = object.__new__(ParallelConfig)
    parallel_config.enable_dbo = enable_dbo
    parallel_config.ubatch_size = ubatch_size
    parallel_config.prefill_context_parallel_size = pcp_size
    parallel_config.decode_context_parallel_size = 1
    parallel_config.tensor_parallel_size = 1
    config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(linear_num_key_heads=4, linear_num_value_heads=4)),
    )

    def init_attention(self, *, vllm_config, **kwargs):
        torch.nn.Module.__init__(self)
        assert vllm_config.parallel_config.prefill_context_parallel_size == 1
        self.indexer = SimpleNamespace(configure_pcp_cache_backends=Mock())
        self.register_buffer("topk_indices_buffer", torch.empty(8, 4, dtype=torch.int32), persistent=False)

    def init_backbone(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.ModuleList(qsa.AscendQwen4ExpQSAAttention(vllm_config=config) for _ in range(3))
        self.start_layer, self.end_layer = 0, len(self.layers)

    monkeypatch.setattr(qsa.upstream_qsa.Qwen4ExpQSAAttention, "__init__", init_attention)
    # model.py replaces the upstream module global with its subclass.
    monkeypatch.setattr(model.AscendQwen4ExpModel.__bases__[0], "__init__", init_backbone)
    first = model.AscendQwen4ExpModel(vllm_config=config)
    buffers = [attention.topk_indices_buffer for attention in first.layers]
    assert config.parallel_config.prefill_context_parallel_size == pcp_size
    for layer in first.layers:
        if pcp_size > 1:
            layer.indexer.configure_pcp_cache_backends.assert_called_once_with(config)
        else:
            layer.indexer.configure_pcp_cache_backends.assert_not_called()

    if not expected_sharing:
        assert len({buffer.data_ptr() for buffer in buffers}) == len(buffers)
        first.to_empty(device="cpu")
        assert len({attention.topk_indices_buffer.data_ptr() for attention in first.layers}) == len(buffers)
        return

    assert all(buffer is buffers[0] for buffer in buffers)
    assert not hasattr(first, "topk_indices_buffer")
    second = model.AscendQwen4ExpModel(vllm_config=config)
    draft = [qsa.AscendQwen4ExpQSAAttention(vllm_config=config) for _ in range(2)]
    assert all(attention.topk_indices_buffer is not buffers[0] for attention in [*second.layers, *draft])
    assert draft[0].topk_indices_buffer is not draft[1].topk_indices_buffer

    # Parent conversion must preserve Top-K aliasing and still move ordinary buffers.
    first.register_buffer("ordinary_buffer", torch.zeros(1), persistent=False)
    parent = torch.nn.ModuleList([first, second, *draft])
    parent.to(device="meta")
    assert first.ordinary_buffer.device.type == "meta"
    parent.to_empty(device="cpu")
    assert first.ordinary_buffer.device.type == "cpu"
    moved = [attention.topk_indices_buffer for attention in first.layers]
    assert all(buffer.device.type == "cpu" and buffer is moved[0] for buffer in moved)
    assert moved[0] is not buffers[0]
    assert all(
        attention.topk_indices_buffer.data_ptr() != moved[0].data_ptr() for attention in [*second.layers, *draft]
    )
    assert draft[0].topk_indices_buffer is not draft[1].topk_indices_buffer
    assert not parent.state_dict()
