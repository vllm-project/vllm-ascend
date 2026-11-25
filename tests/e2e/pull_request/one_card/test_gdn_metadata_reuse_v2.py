# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.ops.triton.mamba.graph_state import GDNGraphStateUpdater
from vllm_ascend.utils import adapt_patch


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


def test_gdn_shared_metadata_aclgraph_reads_updated_group_states():
    torch.npu.set_device(0)
    graph_reqs, width = 4, 4
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,), num_speculative_blocks=3)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        additional_config=None,
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_text_config=SimpleNamespace(model_type="qwen3_5", linear_attn_config={"num_heads": 8}),
            get_num_attention_heads=lambda _parallel_config: 8,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
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
    builders = [AscendGDNAttentionMetadataBuilder(spec, [f"layer{i}"], config, torch.device("npu")) for i in range(24)]
    table = torch.arange(graph_reqs * (width + 2), dtype=torch.int32, device="npu").view(graph_reqs, width + 2)
    group_tables = [table + i * 1000 for i in range(24)]

    def prepare(count, step):
        query_cpu = torch.tensor([min(i, count) * width for i in range(graph_reqs + 1)], dtype=torch.int32)
        common = AscendCommonAttentionMetadata(
            query_start_loc=query_cpu.to("npu"),
            query_start_loc_cpu=query_cpu,
            seq_lens=torch.full((graph_reqs,), 32, dtype=torch.int32, device="npu"),
            seq_lens_cpu_upper_bound=torch.full((graph_reqs,), 32, dtype=torch.int32),
            num_reqs=graph_reqs,
            num_actual_tokens=count * width,
            max_query_len=width,
            max_seq_len=1024,
            block_table_tensor=group_tables[0],
            slot_mapping=torch.empty(graph_reqs * width, dtype=torch.int64, device="npu"),
            is_prefilling=torch.zeros(graph_reqs, dtype=torch.bool),
        )
        accepted = torch.full((graph_reqs,), (step - 1) % width + 1, dtype=torch.int32, device="npu")
        drafts = torch.tensor([3] * count + [-1] * (graph_reqs - count), dtype=torch.int32)
        first = builders[0].build(0, common, accepted, drafts, num_actual_reqs=count)
        updates: list = []
        metadata = [first] + [
            builder.update_block_table(first, bt, graph_state_updates=updates)
            for builder, bt in zip(builders[1:], group_tables[1:])
        ]
        builders[0].graph_state_updater.apply(updates)
        return metadata

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
        expected = torch.stack([table.cpu()[:, :width] + i * 1000 + (step - 1) * 100 for i in range(24)])
        expected[:, count:].zero_()
        torch.testing.assert_close(states.cpu(), expected, rtol=0, atol=0)
        assert accepted.cpu().tolist() == [(step - 1) % width + 1] * count + [1] * (graph_reqs - count)
        assert query.cpu().tolist() == [min(i, count) * width for i in range(graph_reqs + 1)]
        assert masks.cpu().tolist() == [True] * count + [False] * (graph_reqs - count)
        assert lengths.cpu().tolist() == [0] + [width] * count + [0] * (graph_reqs - count)
