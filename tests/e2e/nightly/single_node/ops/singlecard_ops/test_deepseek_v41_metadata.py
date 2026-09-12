# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise shared V4.1 metadata with native operators and changed-input replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.forward_context import BatchDescriptor

from tests.deepseek_v41_cache_utils import make_cache_config
from vllm_ascend.attention.dsa_v41 import DeepseekV41MetadataBuilder
from vllm_ascend.ops import rope_dsv4
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor

# SmlaMetadata and QliV2Metadata define the initialized payload. The remainder
# of each 1024-word allocation is reserved and is not read by the kernels.
SMLA_METADATA_WORDS = 36 * 9 + 72 * 8
QLI_METADATA_WORDS = 36 * 8 + 72 * 8

METADATA_TENSORS = (
    "seq_lens",
    "cache_seq_lens",
    "cmp_residual",
    "slot_mapping",
    "smla_metadata",
    "qli_metadata",
    "c2_ring_metadata",
    "c2_complete_mask",
    "c2_source_positions",
    "c2_source_cos",
    "c2_source_sin",
)


def _builders(runtime, device, deferred):
    result = []
    for group in make_cache_config(17).kv_cache_groups:
        layers_by_spec = {}
        for name, spec in group.kv_cache_spec.kv_cache_specs.items():
            layers_by_spec.setdefault(spec, []).append(name)
        group_builders = []
        for spec, names in layers_by_spec.items():
            builder = DeepseekV41MetadataBuilder(spec, names, runtime, device)
            # Resolve C2's exact source RoPE table for either execution mode.
            builder.enable_device_metadata()
            builder._device_metadata_enabled = deferred
            group_builders.append(builder)
        result.append(group_builders)
    return result


def _build(builders, lengths, query_len, *, shared, block_offset=0, idle=False):
    device = builders[0][0]._seq_lens.device
    positions = torch.tensor(
        [*range(lengths[0] - query_len, lengths[0]), *range(lengths[1] - query_len, lengths[1]), 0],
        device=device,
        dtype=torch.int64,
    )
    query_start_loc_cpu = torch.tensor([0, query_len, 2 * query_len, 2 * query_len + 1], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([*lengths, 0], dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens = seq_lens_cpu.to(device)
    batch_shared, metadata, tasks = {}, [], []
    for gid, group_builders in enumerate(builders):
        group_shared = {}
        blocks = torch.tensor([gid + 1 + block_offset, gid + 2 + block_offset, 0], device=device, dtype=torch.int32)
        slots = torch.cat(
            (
                blocks[0] * 128 + positions[:query_len] % 128,
                blocks[1] * 128 + positions[query_len : 2 * query_len] % 128,
                torch.full((1,), -1, device=device, dtype=torch.int64),
            )
        )
        common = SimpleNamespace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            slot_mapping=slots,
            block_table_tensor=blocks[:, None],
            num_reqs=3,
            num_input_tokens=positions.numel(),
            num_actual_tokens=2 * query_len,
            max_query_len=query_len,
            max_seq_len=max(lengths),
            is_prefilling=torch.tensor([query_len > 1, query_len > 1, False]),
        )
        for builder in group_builders:
            metadata.append(
                builder.build(
                    0,
                    common,
                    num_actual_reqs=2,
                    skip_ring_state_update=idle,
                    common_v41_metadata=group_shared,
                    common_v41_batch_metadata=batch_shared if shared else None,
                )
            )
            tasks.extend(builder.take_device_metadata_tasks())
    return metadata, tasks


def _consume(metadata):
    outputs = {}
    for index, entry in enumerate(metadata):
        for field in METADATA_TENSORS:
            value = getattr(entry, field)
            if value is not None:
                if field == "smla_metadata":
                    value = value[:SMLA_METADATA_WORDS]
                elif field == "qli_metadata":
                    value = value[:QLI_METADATA_WORDS]
                outputs[index, field] = value.clone()
        if entry.cos is not None:
            for layer in (0, 2):
                name = f"model.layers.{layer}.self_attn.attn"
                outputs[index, f"cos:{layer}"] = entry.cos[name].clone()
                outputs[index, f"sin:{layer}"] = entry.sin[name].clone()
    return outputs


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("query_len", [1, 3])
@torch.inference_mode()
def test_grouped_metadata_matches_independent_builds_and_replays(monkeypatch, deferred, query_len):
    assert enable_custom_op(), "V4.1 native metadata operators are required"
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", rope_dsv4.RopeGlobalState())
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=dict(
                sliding_window=128,
                num_attention_heads=32,
                head_dim=512,
                qk_rope_head_dim=64,
                index_topk=512,
                index_n_heads=64,
                index_head_dim=128,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=4),
        speculative_config=None,
    )
    with device:
        for layer in (0, 2, 8, 14):
            rope_dsv4.ComplexExpRotaryEmbedding(
                runtime,
                f"model.layers.{layer}.self_attn.attn",
                head_size=512,
                rotary_dim=64,
                max_position_embeddings=512,
                base=10000 if layer == 0 else 1000000,
                scaling_factor=1,
            )
    reference_builders = _builders(runtime, device, False)
    shared_builders = _builders(runtime, device, deferred)
    executor = DeviceMetadataExecutor() if deferred else None
    descriptor = BatchDescriptor(num_tokens=3, num_reqs=3) if query_len == 1 else None
    graph = None
    pointers = None
    for iteration, (lengths, idle) in enumerate([([127, 128], False), ([130, 131], False), ([130, 131], True)]):
        reference, _ = _build(reference_builders, lengths, query_len, shared=False, block_offset=iteration, idle=idle)
        expected = {key: value.cpu() for key, value in _consume(reference).items()}
        metadata, tasks = _build(shared_builders, lengths, query_len, shared=True, block_offset=iteration, idle=idle)
        if executor is not None:
            assert len(tasks) == 6
            executor.submit(tasks, descriptor)
        current_pointers = tuple(
            tuple(getattr(entry, field).data_ptr() for field in METADATA_TENSORS if getattr(entry, field) is not None)
            for entry in metadata
        )
        if pointers is not None:
            assert current_pointers == pointers
        pointers = current_pointers
        if query_len == 1 and graph is not None:
            graph.replay()
        elif query_len == 1:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                if executor is not None:
                    for task in tasks:
                        executor.wait(task.stage, task.group_id)
                outputs = _consume(metadata)
            graph.replay()
        else:
            if executor is not None:
                for task in tasks:
                    executor.wait(task.stage, task.group_id)
            outputs = _consume(metadata)
        if executor is not None:
            executor.release()
        torch.npu.synchronize()
        assert outputs.keys() == expected.keys()
        for key, value in outputs.items():
            torch.testing.assert_close(
                value.cpu(),
                expected[key],
                rtol=0,
                atol=0,
                msg=lambda detail, iteration=iteration, key=key: f"iteration={iteration}, {key}: {detail}",
            )
