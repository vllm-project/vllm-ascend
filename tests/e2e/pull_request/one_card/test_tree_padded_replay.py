# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.ops.gdn_tree_triton import npu_recurrent_gated_delta_rule_tree_triton
from vllm_ascend.ops.triton.spec_decode.tree.tree_conv import tree_causal_conv1d_triton
from vllm_ascend.worker.v2.model_states.tree_mamba_hybrid import AscendTreeMambaHybridModelState
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.commit import TreeCommitCoordinator
from vllm_ascend.worker.v2.spec_decode.tree.kv_compaction import TreeKVCacheCompactor
from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats

_DRAFT_DEPTH = 15
_LIVE_ROWS = 2
_GRAPH_ROWS = 3
_TREE_WIDTH = 4  # Root and three sibling candidates.
_CONV_WIDTH = 2
_HEAD_DIM = 64
_REUSED_PAGE = 3


def _padded_metadata(builder, state_pages, kv_slots, model_state, prefix):
    device = state_pages.device
    starts = torch.tensor([0, _TREE_WIDTH, _LIVE_ROWS * _TREE_WIDTH, _LIVE_ROWS * _TREE_WIDTH], dtype=torch.int32)
    seq_lens = torch.tensor([prefix + _TREE_WIDTH, prefix + _TREE_WIDTH, 0], dtype=torch.int32)
    computed = torch.tensor([prefix, prefix, 0], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=starts.to(device),
        query_start_loc_cpu=starts,
        seq_lens=seq_lens.to(device),
        _seq_lens_cpu=seq_lens,
        seq_lens_cpu=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        _num_computed_tokens_cpu=computed,
        num_computed_tokens_cpu=computed,
        num_reqs=_GRAPH_ROWS,
        num_actual_tokens=_LIVE_ROWS * _TREE_WIDTH,
        max_query_len=_TREE_WIDTH,
        max_seq_len=prefix + _TREE_WIDTH,
        block_table_tensor=state_pages,
        slot_mapping=kv_slots.flatten(),
        causal=True,
        is_prefilling=torch.zeros(_GRAPH_ROWS, dtype=torch.bool),
    )
    prev_path, prev_sampled = model_state._tree_buffers(device)
    metadata = builder.build(
        0,
        common,
        num_actual_reqs=_LIVE_ROWS,
        num_accepted_tokens=model_state.num_accepted_tokens_gpu,
        num_decode_draft_tokens_cpu=torch.tensor([_TREE_WIDTH - 1, _TREE_WIDTH - 1, -1]),
        tree_parents=torch.zeros((_GRAPH_ROWS, _TREE_WIDTH - 1), dtype=torch.int32, device=device),
        tree_num_nodes=torch.tensor([_TREE_WIDTH - 1, _TREE_WIDTH - 1, 0], dtype=torch.int32, device=device),
        prev_path_node_ids=prev_path,
        prev_num_sampled=prev_sampled,
    )
    assert metadata.num_spec_decodes == _LIVE_ROWS
    spec = metadata.spec_decode_metadata
    assert spec.spec_causal_conv1d.query_start_loc.cpu().tolist() == starts.tolist()
    assert spec.actual_seq_lengths.cpu().tolist() == [0, _TREE_WIDTH, _TREE_WIDTH, 0]
    return spec


def _greedy_token(keys, values, recurrent_output):
    # The small deterministic readout uses both the attention cache and GDN.
    return (keys.float().sum(0) * 0.25 + values.float().sum(0) + recurrent_output.float() * 4).argmax().item()


@pytest.mark.e2e_coverage(
    arch="mamba_ssm",
    feature="spec_decode,aclgraph",
    parallel="",
    deploy="",
    hardware="A2",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@torch.inference_mode()
def test_padded_full_decode_tree_commit_survives_cross_request_slot_reuse():
    device = torch.device("npu")
    dtype = torch.bfloat16
    width = _TREE_WIDTH
    tokens_per_replay = _LIVE_ROWS * width
    config = SimpleNamespace(
        additional_config={"tree_spec_config": {"budget": width - 1}},
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=_GRAPH_ROWS * width,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(), enforce_eager=False),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1, tensor_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=_GRAPH_ROWS, max_num_batched_tokens=64),
        speculative_config=SimpleNamespace(
            method="dflash",
            num_speculative_tokens=_DRAFT_DEPTH,
            max_num_new_slots_for_drafting=0,
            parallel_drafting=False,
            enable_adaptive_verification=False,
            enforce_eager=False,
            use_eagle_block_drop=lambda: False,
        ),
        num_speculative_tokens=_DRAFT_DEPTH,
        num_hash_lookahead_tokens=0,
        use_v2_model_runner=True,
    )
    conv_columns = _DRAFT_DEPTH + _CONV_WIDTH - 1
    conv_shape = (_HEAD_DIM, conv_columns) if is_conv_state_dim_first() else (conv_columns, _HEAD_DIM)
    cache_spec = MambaSpec(
        block_size=16,
        shapes=(conv_shape, (1, _HEAD_DIM, _HEAD_DIM)),
        dtypes=(dtype, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=width - 1,
    )
    builder = AscendGDNAttentionMetadataBuilder(cache_spec, ["gdn"], config, device)

    # Provide the persistent buffers normally initialized by the full model runner.
    model_state = AscendTreeMambaHybridModelState.__new__(AscendTreeMambaHybridModelState)
    model_state.vllm_config = config
    model_state._tree_slots = width
    model_state._align_mode = False
    model_state._use_replayssm_trackers = False
    model_state.recoverssm = None
    model_state.num_accepted_tokens_gpu = torch.ones(_GRAPH_ROWS, dtype=torch.int32, device=device)
    computed = torch.zeros(_LIVE_ROWS, dtype=torch.int32, device=device)
    block_table = torch.zeros((_LIVE_ROWS, 2 + width), dtype=torch.int32, device=device)
    keys = torch.zeros((16, 1, _HEAD_DIM), dtype=dtype, device=device)
    values = torch.zeros_like(keys)
    gathers = tuple(cache.new_empty((_LIVE_ROWS * _DRAFT_DEPTH, _HEAD_DIM)) for cache in (keys, values))
    # One token per physical KV page lets the test force exact reuse.
    bound_groups = [((keys, values), block_table, 1, gathers)]
    runner = SimpleNamespace(
        device=device,
        num_speculative_steps=_DRAFT_DEPTH,
        max_num_reqs=_LIVE_ROWS,
        max_model_len=block_table.shape[1],
        compilation_config=config.compilation_config,
        model_config=config.model_config,
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=computed)),
        speculator=SimpleNamespace(
            tree_graph_stats=TreeGraphStats(),
            tree_shape=SimpleNamespace(target_verification_width=width),
        ),
    )
    runner.tree_kv_compact = TreeKVCacheCompactor(runner, bind_groups=lambda *_: bound_groups)
    runner.tree_kv_compact.capture([_LIVE_ROWS])
    assert _LIVE_ROWS in runner.tree_kv_compact._graphs
    committer = TreeCommitCoordinator(runner, model_state)

    conv_state = torch.zeros((16, *conv_shape), dtype=dtype, device=device)
    state = torch.zeros((16, 1, _HEAD_DIM, _HEAD_DIM), dtype=torch.float32, device=device)
    embedding = torch.eye(_HEAD_DIM, dtype=dtype)
    x = torch.zeros((tokens_per_replay, _HEAD_DIM), dtype=dtype, device=device)
    weight = torch.ones((_HEAD_DIM, _CONV_WIDTH), dtype=dtype, device=device)
    query = x.new_zeros((tokens_per_replay, 1, _HEAD_DIM))
    query[..., 0] = 1
    beta = x.new_full((tokens_per_replay, 1), 0.5)
    kv_slots = torch.zeros((tokens_per_replay, 1), dtype=torch.long, device=device)

    def verify(spec):
        conv = spec.spec_causal_conv1d
        activations = tree_causal_conv1d_triton(
            x,
            weight,
            conv_state,
            conv.cache_indices,
            conv.query_start_loc,
            spec.conv_window_rows,
            spec.conv_prefix_src,
            bias=None,
            activation=False,
        )
        torch_npu.npu_scatter_nd_update_(keys, kv_slots, x.unsqueeze(1))
        torch_npu.npu_scatter_nd_update_(values, kv_slots, activations.unsqueeze(1))
        return npu_recurrent_gated_delta_rule_tree_triton(
            query,
            query,
            activations.unsqueeze(1),
            state,
            beta=beta,
            scale=1.0,
            actual_seq_lengths=spec.actual_seq_lengths,
            ssm_state_indices=conv.cache_indices.flatten(),
            init_state_indices=spec.init_state_indices.flatten(),
        )

    idx_mapping = torch.arange(_LIVE_ROWS, dtype=torch.int32, device=device)
    next_tokens = torch.tensor([7, 12])
    previous = embedding[torch.tensor([17, 19])].float()
    linear_state = torch.zeros_like(previous)
    linear_keys = [[], []]
    linear_values = [[], []]
    graph = torch.npu.NPUGraph()
    captured_metadata = None
    for step, (kv_pages, gdn_pages, candidates, accepted) in enumerate(
        (
            (
                [[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 0, 0]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[8, 9, 10], [13, 14, 15]],
                [3, 1],
            ),
            (
                [[1, 2, 9, 10, 11, 12], [5, 6, 13, 14, 3, 15]],
                [[1, 2, 9, 4], [5, 6, 3, 8]],
                [[21, 22, 23], [25, 26, 27]],
                [3, 2],
            ),
        )
    ):
        prefix = 2 * step
        block_table.copy_(torch.tensor(kv_pages, dtype=torch.int32))
        computed.fill_(prefix)
        state_pages = torch.tensor([*gdn_pages, [0] * width], dtype=torch.int32, device=device)
        # Only the physical page tables change. The graph keeps its metadata addresses.
        slot_table = block_table[:, prefix : prefix + width]
        kv_slots.copy_(slot_table.reshape(-1, 1))
        token_ids = torch.cat((next_tokens[:, None], torch.tensor(candidates)), dim=1)
        x.copy_(embedding[token_ids.flatten()])
        spec = _padded_metadata(builder, state_pages, kv_slots, model_state, prefix)
        metadata_addresses = tuple(
            tensor.data_ptr()
            for tensor in (
                spec.spec_causal_conv1d.query_start_loc,
                spec.spec_causal_conv1d.cache_indices,
                spec.init_state_indices,
                spec.actual_seq_lengths,
                spec.conv_window_rows,
                spec.conv_prefix_src,
            )
        )
        if step == 0:
            captured_metadata = metadata_addresses
            verify(spec)  # Compile the kernels before capture.
            torch.npu.synchronize()
            with torch.npu.graph(graph):
                output = verify(spec)
            conv_state.zero_()
            state.zero_()
            keys.zero_()
            values.zero_()
            conv_history = conv_state[:, :, 0] if is_conv_state_dim_first() else conv_state[:, 0, :]
            conv_history[state_pages[:_LIVE_ROWS, 0].long()] = previous.to(device=device, dtype=dtype)
        else:
            assert metadata_addresses == captured_metadata
            assert slot_table[1, accepted[1]].item() == state_pages[1, accepted[1]].item() == _REUSED_PAGE
            assert spec.init_state_indices[:_LIVE_ROWS, 0].cpu().tolist() == [4, 6]

        graph.replay()
        if step == 0:
            assert slot_table[0, 2].item() == state_pages[0, 2].item() == _REUSED_PAGE
            assert not torch.equal(keys[_REUSED_PAGE], keys[4])
            assert not torch.equal(values[_REUSED_PAGE], values[4])
            assert not torch.equal(state[_REUSED_PAGE], state[4])

        last_node = torch.tensor(accepted, dtype=torch.int32, device=device)
        path = torch.full((_LIVE_ROWS, _DRAFT_DEPTH), -1, dtype=torch.int32, device=device)
        path[:, 0] = last_node
        sampled = torch.full_like(last_node, 2)
        acceptance = TreeAcceptance(path, sampled, last_node, torch.ones_like(last_node, dtype=torch.bool))
        committer.commit(idx_mapping, acceptance, num_rejected=width - sampled)
        model_state.postprocess_state(idx_mapping, sampled)
        cursor = model_state.num_accepted_tokens_gpu[:_LIVE_ROWS].cpu()
        assert cursor.tolist() == [node + 1 for node in accepted]

        for request, node in enumerate(accepted):
            # Independent linear decode forced through the root and accepted sibling.
            for token in token_ids[request, [0, node]]:
                current = embedding[token].float()
                value = previous[request] + current
                linear_state[request] += 0.5 * (value - linear_state[request])
                previous[request] = current
                linear_keys[request].append(current.to(dtype))
                linear_values[request].append(value.to(dtype))

            pages = block_table[request, : prefix + 2].long()
            actual_k = keys[pages, 0].cpu()
            actual_v = values[pages, 0].cpu()
            expected_k = torch.stack(linear_keys[request])
            expected_v = torch.stack(linear_values[request])
            torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)

            page = gdn_pages[request][cursor[request].item() - 1]
            expected_state = torch.zeros((1, _HEAD_DIM, _HEAD_DIM))
            expected_state[0, :, 0] = linear_state[request]
            torch.testing.assert_close(state[page].cpu(), expected_state, rtol=0, atol=0)
            next_token = _greedy_token(actual_k, actual_v, output[request * width + node, 0].cpu())
            assert next_token == _greedy_token(expected_k, expected_v, linear_state[request])
            next_tokens[request] = next_token

    assert model_state.tree_graph_stats.snapshot()["post"]["replays"] == 2
