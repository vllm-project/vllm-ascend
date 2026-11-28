# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.gdn_attn_builder import AscendGDNAttentionBackend, AscendGDNHostMetadataBackend
from vllm_ascend.ops.triton.mamba.cached_decode import materialize_cached_gdn_decode
from vllm_ascend.ops.triton.mamba.graph_state import GDNGraphStateUpdater
from vllm_ascend.utils import adapt_patch
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


@pytest.mark.parametrize("width", [1, 3, 8, 17])
@pytest.mark.parametrize("reset_spec", [False, True])
def test_batched_graph_state_replay_reads_current_group_ids(width, reset_spec):
    torch.npu.set_device(0)
    groups, capacity = 23, 8
    tables = (
        torch.arange(groups * capacity * (width + 2), dtype=torch.int32, device="npu").view(groups, capacity, width + 2)
        + 2**24
        + 1
    )
    # Exercise a prefix view alongside full source tables in the same launch.
    sources = list(tables.unbind())
    sources[-1] = sources[-1][:, :width]
    outputs = [torch.full((capacity, width), -99, dtype=torch.int32, device="npu") for _ in range(groups)]
    resets = [
        torch.full((capacity, 4), 99, dtype=torch.int32, device="npu") if reset_spec else None for _ in range(groups)
    ]
    addresses = [dst.data_ptr() for dst in outputs]
    updater = GDNGraphStateUpdater()

    def prepare(count):
        updater.apply([(src, dst, count, capacity, 0, reset) for src, dst, reset in zip(sources, outputs, resets)])

    main_stream = torch.npu.current_stream()
    stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
    snapshots = []
    with torch.npu.stream(stream):
        stream.wait_stream(main_stream)
        prepare(5)
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            captured = torch.stack(outputs)
            captured_resets = torch.stack(resets) if reset_spec else None
        for step, count in enumerate([5, 1, 0, 8], 1):
            tables.add_(100)
            prepare(count)
            graph.replay()
            snapshots.append(
                (step, count, captured.clone(), None if captured_resets is None else captured_resets.clone())
            )
    torch.npu.synchronize()
    original = torch.arange(groups * capacity * (width + 2), dtype=torch.int32).view(groups, capacity, width + 2)
    for step, count, current, cleared in snapshots:
        expected = original[:, :, :width].clone() + 2**24 + 1 + step * 100
        expected[:, count:] = 0
        torch.testing.assert_close(current.cpu(), expected, rtol=0, atol=0)
        if cleared is not None:
            assert torch.all(cleared.cpu() == -1)
    assert [dst.data_ptr() for dst in outputs] == addresses


@pytest.mark.parametrize(
    "num_groups,max_reqs,num_reqs,slots,seq_dtype",
    [(3, 33, 1, 3, torch.int64), (24, 64, 17, 4, torch.int32)],
)
def test_aligned_state_indices_aclgraph_replay(num_groups, max_reqs, num_reqs, slots, seq_dtype):
    torch.npu.set_device(0)
    adapt_patch()
    columns = 32
    tables = torch.arange(num_groups * max_reqs * columns, dtype=torch.int32, device="npu").view(
        num_groups, max_reqs, columns
    )
    seq_lens = torch.ones(max_reqs * 2, dtype=seq_dtype, device="npu")[::2]
    ctx = MambaSpecDecodeGPUContext.__new__(MambaSpecDecodeGPUContext)
    ctx.is_initialized = True
    ctx.block_table_ptrs = torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu")
    ctx.aligned_index_block_table_ptrs = ctx.block_table_ptrs
    ctx.block_table_stride_req = tables.stride(1)
    ctx.block_size = 16
    ctx.num_groups = num_groups
    ctx.aligned_state_indices = torch.full((num_groups, max_reqs, slots), -99, dtype=torch.int32, device="npu")
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    snapshots = []
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        ctx.compute_aligned_state_indices(seq_lens, num_reqs)
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            ctx.compute_aligned_state_indices(seq_lens, num_reqs)
        for step in (1, 2):
            lengths = ((torch.arange(max_reqs, dtype=seq_dtype) + step) % (columns - slots + 1)) * 16 + 1
            seq_lens.copy_(lengths)
            tables.add_(100)
            graph.replay()
            snapshots.append((step, lengths, ctx.aligned_state_indices.clone()))
    torch.npu.synchronize()
    original = torch.arange(num_groups * max_reqs * columns, dtype=torch.int32).view(num_groups, max_reqs, columns)
    for step, lengths, output in snapshots:
        cols = (lengths[:num_reqs, None] - 1) // 16 + torch.arange(slots)
        expected = torch.stack([(table + step * 100).gather(1, cols.long()) for table in original])
        torch.testing.assert_close(output[:, :num_reqs].cpu(), expected, rtol=0, atol=0)
        assert torch.all(output[:, num_reqs:].cpu() == -99)


@pytest.mark.parametrize(
    ("backend", "mode"), [(AscendGDNAttentionBackend, "none"), (AscendGDNHostMetadataBackend, "align")]
)
def test_gdn_shared_metadata_aclgraph_reads_updated_group_states(backend, mode):
    torch.npu.set_device(0)
    graph_reqs, width = 4, 4
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,), num_speculative_blocks=3)
    config = SimpleNamespace(
        num_speculative_tokens=3,
        additional_config=None,
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_text_config=SimpleNamespace(model_type="qwen3_5", linear_attn_config={"num_heads": 8}),
            get_num_attention_heads=lambda _parallel_config: 8,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=None
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=1024),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            max_num_new_slots_for_drafting=0,
            parallel_drafting=False,
            enable_adaptive_verification=False,
            use_eagle_block_drop=lambda: False,
        ),
        num_hash_lookahead_tokens=0,
    )
    builders = [backend.get_builder_cls()(spec, [f"layer{i}"], config, torch.device("npu")) for i in range(24)]
    table = torch.arange(graph_reqs * (width + 2), dtype=torch.int32, device="npu").view(graph_reqs, width + 2)
    group_tables = [table + i * 1000 for i in range(24)]

    groups = [
        [AttentionGroup(backend, [f"layer{i}"], spec, i, metadata_builders=[builder])]
        for i, builder in enumerate(builders)
    ]
    state = SimpleNamespace(vllm_config=config, max_model_len=1024, _align_mode=False)
    slot_mappings = torch.empty((len(groups), graph_reqs * width), dtype=torch.int64, device="npu")
    kv_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)] * len(groups))

    def prepare(count, step):
        query = np.minimum(np.arange(graph_reqs + 1, dtype=np.int32), count) * width
        state.num_accepted_tokens_gpu = torch.full((count,), (step - 1) % width + 1, dtype=torch.int32, device="npu")
        seq_lens = np.where(np.arange(graph_reqs) < count, 32, 0).astype(np.int32)
        batch = SimpleNamespace(
            num_reqs=count,
            num_reqs_after_padding=graph_reqs,
            num_tokens=count * width,
            num_tokens_after_padding=graph_reqs * width,
            is_prefilling_np=np.zeros(count, dtype=bool),
            prefill_runs_as_decode_np=None,
            idx_mapping=torch.arange(count, device="npu"),
            num_draft_tokens_per_req=np.full(count, width - 1, dtype=np.int32),
            num_scheduled_tokens=np.diff(query),
            max_query_len=width,
            query_start_loc=torch.from_numpy(query).to("npu"),
            query_start_loc_np=query,
            seq_lens=torch.from_numpy(seq_lens).to("npu"),
            seq_lens_np=seq_lens,
            dcp_local_seq_lens=None,
            positions=None,
            attn_state=None,
        )
        metadata = AscendMambaHybridModelState._build_attn_metadata_for_batch(
            state,
            batch,
            CUDAGraphMode.FULL,
            tuple(group_tables),
            slot_mappings,
            groups,
            kv_config,
            for_capture=step == 1,
            use_pcp_padding=False,
        )
        return [metadata[f"layer{i}"] for i in range(len(groups))]

    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    state_outputs = torch.empty((24, graph_reqs, width), dtype=torch.int32, device="npu")
    accepted_output = torch.empty(graph_reqs, dtype=torch.int32, device="npu")
    query_output = torch.empty(graph_reqs + 1, dtype=torch.int32, device="npu")
    mask_output = torch.empty(graph_reqs, dtype=torch.bool, device="npu")
    lengths_output = torch.empty(graph_reqs + 1, dtype=torch.int32, device="npu")
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        captured = prepare(graph_reqs, 1)
        assert len({m.spec_query_start_loc.data_ptr() for m in captured}) == 1
        assert len({m.spec_state_indices_tensor.data_ptr() for m in captured}) == len(groups)

        def consume():
            for i, metadata in enumerate(captured):
                state_outputs[i].copy_(metadata.spec_state_indices_tensor)
            accepted_output.copy_(captured[0].num_accepted_tokens)
            query_output.copy_(captured[0].spec_query_start_loc)
            mask_output.copy_(captured[0].spec_sequence_masks)
            lengths_output.copy_(captured[0].spec_decode_metadata.actual_seq_lengths)

        consume()
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            consume()
        snapshots = []
        # Repeat layouts while refreshing accepted offsets and group-local IDs.
        for step, count in enumerate([1, 1, 3, 3, 2, 2], start=2):
            for bt in group_tables:
                bt.add_(100)
            if step == 5:
                # Replace the source allocations while graph destinations stay
                # stable. Pointer-table caching must bind the new tables.
                group_tables[:] = [bt.clone() for bt in group_tables]
            # A layout cache must restore graph inputs, not assume the last
            # execution left their contents intact.
            builders[0].spec_query_start_loc.fill_(-1)
            builders[0].spec_sequence_masks.fill_(False)
            builders[0].spec_actual_seq_lengths.fill_(-1)
            current = prepare(count, step)
            for before, after in zip(captured, current):
                assert before.spec_state_indices_tensor.data_ptr() == after.spec_state_indices_tensor.data_ptr()
                assert before.num_accepted_tokens.data_ptr() == after.num_accepted_tokens.data_ptr()
            graph.replay()
            snapshots.append(
                (
                    count,
                    step,
                    state_outputs.clone(),
                    accepted_output.clone(),
                    query_output.clone(),
                    mask_output.clone(),
                    lengths_output.clone(),
                )
            )
    torch.npu.synchronize()
    for count, step, states, accepted, query, masks, lengths in snapshots:
        col = 1 if mode == "align" else 0  # seq_len=32 and block_size=16
        expected = torch.stack([table.cpu()[:, col : col + width] + i * 1000 + (step - 1) * 100 for i in range(24)])
        expected[:, count:].zero_()
        torch.testing.assert_close(states.cpu(), expected, rtol=0, atol=0)
        assert accepted.cpu().tolist() == [(step - 1) % width + 1] * count + [1] * (graph_reqs - count)
        assert query.cpu().tolist() == [min(i, count) * width for i in range(graph_reqs + 1)]
        assert masks.cpu().tolist() == [True] * count + [False] * (graph_reqs - count)
        assert lengths.cpu().tolist() == [0] + [width] * count + [0] * (graph_reqs - count)


def test_cached_decode_replays_strided_inputs_and_preserves_tail():
    width, graph_rows, actual_rows, stride = 8, 33, 17, 2
    torch.npu.set_device(0)
    capacity = graph_rows + 3
    state_values = (torch.arange(capacity * (width + 4), dtype=torch.int32) + 2**24 + 1).view(capacity, width + 4)
    states = state_values.to("npu")[:, :width]
    query_values = torch.arange((actual_rows + 1) * stride, dtype=torch.int32)
    query = (query_values * 5).to("npu")[::stride]
    accepted = torch.arange(max(1, actual_rows) * stride, dtype=torch.int32, device="npu")[::stride]
    tokens = torch.arange(actual_rows * 5, dtype=torch.int32, device="npu")
    outs = [
        torch.full(shape, -99, dtype=torch.int32, device="npu")
        for shape in (
            (capacity, width),
            (capacity + 1,),
            (capacity,),
            (max(1, actual_rows * 5) + 5,),
            (capacity + 1,),
        )
    ]
    out_state, out_query, out_accepted, out_token, out_lengths = outs
    out_masks = torch.ones(capacity, dtype=torch.bool, device="npu")
    stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()

    def prepare():
        materialize_cached_gdn_decode(
            states,
            query,
            accepted,
            tokens,
            out_state,
            out_query,
            out_accepted,
            out_token,
            out_masks,
            out_lengths,
            actual_rows,
            graph_rows,
        )

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        prepare()
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            prepare()
        snapshots = []
        for step in (1, 2, 3):
            states.add_(1)
            query.add_(1)
            accepted.add_(1)
            tokens.add_(1)
            graph.replay()
            snapshots.append([t.clone() for t in (*outs, out_masks)])
    torch.npu.synchronize()
    for step, snapshot in enumerate(snapshots, 1):
        (
            current_state,
            current_query,
            current_accepted,
            current_tokens,
            lengths,
            masks,
        ) = [t.cpu() for t in snapshot]
        expected_state = state_values[:graph_rows, :width].clone() + step
        expected_state[actual_rows:].zero_()
        torch.testing.assert_close(current_state[:graph_rows], expected_state, rtol=0, atol=0)
        offsets = torch.arange(graph_rows + 1).clamp_max(actual_rows)
        expected_query = offsets.to(torch.int32) * stride * 5 + step
        torch.testing.assert_close(current_query[: graph_rows + 1], expected_query, rtol=0, atol=0)
        expected_lengths = torch.cat([expected_query[:1], expected_query[1:] - expected_query[:-1]])
        torch.testing.assert_close(lengths[: graph_rows + 1], expected_lengths, rtol=0, atol=0)
        assert current_accepted[:graph_rows].tolist() == [i * stride + step for i in range(actual_rows)] + [1] * (
            graph_rows - actual_rows
        )
        assert masks[:graph_rows].tolist() == [True] * actual_rows + [False] * (graph_rows - actual_rows)
        assert current_tokens[: tokens.numel()].tolist() == [i + step for i in range(tokens.numel())]
        for tensor, tail in (
            (current_state, graph_rows),
            (current_query, graph_rows + 1),
            (current_accepted, graph_rows),
            (current_tokens, tokens.numel()),
            (lengths, graph_rows + 1),
        ):
            assert torch.all(tensor[tail:] == -99)
        assert torch.all(masks[graph_rows:])
