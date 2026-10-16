# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for PLE request grouping and causal replay convolution."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, fields, make_dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "vllm_ascend" / "models" / "qwen4_exp"
NULL_BLOCK_ID = 0


@dataclass
class _CommonMetadata:
    """The pinned upstream constructor, without removed private CPU caches."""

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor
    is_prefilling: torch.Tensor | None = None
    seq_lens_cpu_upper_bound: torch.Tensor | None = None

    def compute_num_computed_tokens(self):
        return self.seq_lens - torch.diff(self.query_start_loc)

    def unpadded(self, num_tokens, num_reqs):
        return replace(
            self,
            query_start_loc=self.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            block_table_tensor=self.block_table_tensor[:num_reqs],
            slot_mapping=self.slot_mapping[:num_tokens],
            is_prefilling=self.is_prefilling[:num_reqs],
        )


def _source_module(name, filename, namespace, monkeypatch):
    # Isolate the tensor code from vLLM's NPU/distributed import side effects.
    tree = ast.parse(filename.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module.startswith(("vllm.", "vllm_ascend.")))
    ]
    module = ModuleType(name)
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(tree, str(filename), "exec"), module.__dict__)
    return module


@pytest.fixture
def runtime(monkeypatch):
    # Match the upstream required and optional metadata fields. A subclass must
    # remain acceptable to Qwen4ExpPLELayer._short_conv's isinstance guard.
    required = [
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "num_reqs",
        "has_initial_states_p",
        "query_start_loc_p",
        "num_computed_tokens_p",
        "state_indices_tensor_p",
        "state_indices_tensor_d",
        "query_start_loc_d",
        "num_accepted_tokens",
        "block_idx_last_scheduled_token",
        "block_idx_first_scheduled_token_p",
        "block_idx_last_computed_token",
        "block_idx_last_scheduled_token_prev_step",
        "seq_lens",
    ]
    optional = [
        "num_spec_decodes",
        "num_spec_decode_tokens",
        "num_actual_tokens",
        "spec_query_len",
        "max_prefill_query_len",
        "query_start_loc",
        "state_indices_tensor",
        "has_initial_states_d",
        "non_spec_query_start_loc",
        "spec_query_start_loc",
        "spec_state_indices_tensor",
        "spec_sequence_masks",
        "spec_token_indx",
        "non_spec_token_indx",
        "num_decode_draft_tokens_cpu",
    ]
    upstream_metadata = make_dataclass(
        "UpstreamMetadata",
        [(name, Any) for name in required]
        + [
            (
                name,
                Any,
                1
                if name == "spec_query_len"
                else 0
                if name in {"num_spec_decodes", "num_spec_decode_tokens", "num_actual_tokens", "max_prefill_query_len"}
                else None,
            )
            for name in optional
        ],
    )

    class UpstreamBuilder:
        pass

    metadata = _source_module(
        "qwen4_exp_test_short_conv_metadata",
        MODEL_DIR / "short_conv_attn.py",
        {
            "UpstreamPleShortConvAttentionMetadata": upstream_metadata,
            "UpstreamPleShortConvAttentionMetadataBuilder": UpstreamBuilder,
            "CommonAttentionMetadata": _CommonMetadata,
            "NULL_BLOCK_ID": NULL_BLOCK_ID,
            "async_tensor_h2d": lambda tensor, device: tensor.to(device),
            "mamba_get_block_table_tensor": lambda table, *_args: table,
        },
        monkeypatch,
    )
    conv = _source_module(
        "qwen4_exp_test_short_conv_execution",
        MODEL_DIR / "ple_short_conv.py",
        {"PleShortConvAttentionMetadata": metadata.PleShortConvAttentionMetadata, "NULL_BLOCK_ID": NULL_BLOCK_ID},
        monkeypatch,
    )
    return SimpleNamespace(metadata=metadata, conv=conv, upstream_metadata=upstream_metadata)


def _builder(runtime, spec_tokens=0, full_graph=False):
    builder = runtime.metadata.PleShortConvAttentionMetadataBuilder()
    builder.num_spec = spec_tokens
    builder.use_spec_decode = spec_tokens > 0
    builder.use_full_cuda_graph = full_graph
    builder.kv_cache_spec = None
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
    builder.decode_cudagraph_max_bs = 8
    builder.decode_cudagraph_max_tokens = 32
    builder.spec_state_indices_tensor = torch.empty(8, dtype=torch.int32)
    builder.spec_sequence_masks = torch.empty(8, dtype=torch.bool)
    builder.spec_query_start_loc = torch.empty(9, dtype=torch.int32)
    builder.num_accepted_tokens = torch.empty(8, dtype=torch.int32)
    return builder


def _common(lengths, phases, computed=None, slots=None):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)))
    computed = torch.tensor(computed or [5] * len(lengths), dtype=torch.int32)
    seq_lens = computed + lengths
    return _CommonMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=seq_lens,
        num_reqs=len(lengths),
        num_actual_tokens=int(starts[-1]),
        max_query_len=int(lengths.max()),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=torch.tensor(slots or list(range(1, len(lengths) + 1)), dtype=torch.int32).view(-1, 1),
        slot_mapping=torch.arange(int(starts[-1])),
        is_prefilling=torch.tensor(phases, dtype=torch.bool),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
    )


def _layer(runtime, dilation=2, kernel=3, num_spec=0):
    class Layer:
        conv_state_len = dilation * (kernel - 1)
        short_conv_dilation = dilation
        num_spec_tokens = num_spec

    Layer._short_conv_dilated_chunk_batched = runtime.conv._short_conv_dilated_chunk_batched
    Layer._short_conv_dilated_prefill_batched = runtime.conv._short_conv_dilated_prefill_batched
    return Layer()


def _reference_chunk(layer, inputs, initial, weights):
    history = initial.clone()
    outputs = []
    for token in inputs:
        window = torch.cat((history, token.view(-1, 1)), dim=1)
        value = (window[:, :: layer.short_conv_dilation] * weights).sum(-1)
        outputs.append(F.silu(value))
        if layer.conv_state_len:
            history = window[:, -layer.conv_state_len :]
    return torch.stack(outputs), history


def test_normal_chunk_group_retains_real_phase_counts(runtime):
    common = _common([1, 3, 1, 2], [True, False, False, True], computed=[0, 7, 9, 4], slots=[4, 3, 2, 1])
    result = _builder(runtime).build(0, common)

    assert isinstance(result, runtime.upstream_metadata)
    assert (result.num_decodes, result.num_decode_tokens) == (2, 4)
    assert (result.num_prefills, result.num_prefill_tokens) == (2, 3)
    assert result.num_spec_decodes == 0
    assert result.normal_chunk.token_indices.tolist() == [1, 2, 3]
    assert result.normal_chunk.state_indices_tensor.tolist() == [3]
    assert result.normal_chunk.query_start_loc.tolist() == [0, 3]
    assert result.remaining_token_indices.tolist() == [4, 0, 5, 6]
    assert result.remaining_metadata.query_start_loc_p.tolist() == [0, 1, 3]
    assert result.remaining_metadata.state_indices_tensor.tolist() == [2, 4, 1]
    assert result.remaining_metadata.has_initial_states_p.tolist() == [False, True]
    assert runtime.metadata.PleShortConvAttentionMetadataBuilder.needs_causal_conv1d_metadata is False


@pytest.mark.parametrize("dilation,kernel", [(1, 1), (1, 3), (2, 3), (4, 4)])
@pytest.mark.parametrize("state_first", [False, True])
def test_replay_chunks_match_sequential_decode_and_commit_all_tokens(runtime, dilation, kernel, state_first):
    torch.manual_seed(8)
    layer = _layer(runtime, dilation, kernel)
    state = torch.randn(4, 3, layer.conv_state_len)
    if not state_first:
        state = state.transpose(1, 2).contiguous().transpose(1, 2)
    before = state.clone()
    inputs = torch.randn(5, 3)
    weights = torch.randn(3, kernel)
    expected_a, expected_state_a = _reference_chunk(layer, inputs[:3], before[2], weights)
    expected_b, expected_state_b = _reference_chunk(layer, inputs[3:], torch.zeros_like(before[1]), weights)

    result = layer._short_conv_dilated_chunk_batched(
        inputs, state, weights, torch.tensor([0, 3, 5]), torch.tensor([2, 1]), torch.tensor([True, False]), 2, 5, 3
    )

    torch.testing.assert_close(result, torch.cat((expected_a, expected_b)))
    torch.testing.assert_close(state[2], expected_state_a)
    torch.testing.assert_close(state[1], expected_state_b)
    torch.testing.assert_close(state[[0, 3]], before[[0, 3]])


def test_null_state_slot_has_zero_output_and_keeps_reserved_state(runtime):
    torch.manual_seed(10)
    layer = _layer(runtime)
    state = torch.randn(3, 2, layer.conv_state_len)
    before = state.clone()
    inputs = torch.randn(5, 2)
    weights = torch.randn(2, 3)
    expected_output, expected_state = _reference_chunk(layer, inputs[3:], before[1], weights)

    result = layer._short_conv_dilated_chunk_batched(
        inputs,
        state,
        weights,
        torch.tensor([0, 3, 5]),
        torch.tensor([NULL_BLOCK_ID, 1]),
        torch.tensor([True, True]),
        2,
        5,
        3,
    )

    torch.testing.assert_close(result[:3], torch.zeros_like(result[:3]))
    torch.testing.assert_close(result[3:], expected_output)
    torch.testing.assert_close(state[1], expected_state)
    torch.testing.assert_close(state[[0, 2]], before[[0, 2]])


@pytest.mark.parametrize(
    "lengths,phases,computed,slots",
    [
        ([1, 3, 2], [True, False, True], [0, 8, 3], [3, 1, 2]),
        ([1, 3, 1, 2], [True, False, False, True], [0, 8, 11, 3], [3, 1, 4, 2]),
        ([1, 1, 2], [True, False, True], [0, 11, 3], [3, 1, 2]),
    ],
)
def test_dispatch_scatter_preserves_interleaved_request_order_and_state(runtime, lengths, phases, computed, slots):
    common = _common(lengths, phases, computed=computed, slots=slots)
    metadata = _builder(runtime).build(0, common)
    layer = _layer(runtime)
    torch.manual_seed(9)
    inputs = torch.randn(sum(lengths), 2)
    state = torch.randn(len(lengths) + 1, 2, layer.conv_state_len)
    before = state.clone()
    weights = torch.randn(2, 3)
    expected = []
    expected_state = before.clone()
    for index, (start, stop) in enumerate(zip(common.query_start_loc[:-1], common.query_start_loc[1:])):
        slot = int(common.block_table_tensor[index, 0])
        initial = before[slot] if common.compute_num_computed_tokens()[index] > 0 else torch.zeros_like(before[slot])
        output, history = _reference_chunk(layer, inputs[start:stop], initial, weights)
        expected.append(output)
        expected_state[slot] = history

    def upstream_dispatch(owner, x, item, cache, conv_weights):
        parts = []
        if item.num_decodes:
            # Independent single-token decode reference for the upstream
            # dispatcher, whose PLE metadata contract is unchanged.
            for index in range(item.num_decodes):
                slot = int(item.state_indices_tensor[index])
                initial = cache[slot] if item.has_initial_states_d[index] else torch.zeros_like(cache[slot])
                output, history = _reference_chunk(owner, x[index : index + 1], initial, conv_weights)
                cache[slot] = history
                parts.append(output)
        parts.append(
            owner._short_conv_dilated_prefill_batched(
                x[item.num_decode_tokens :],
                item,
                cache,
                conv_weights,
                item.state_indices_tensor[item.num_decodes :],
                item.num_prefills,
                item.num_decode_tokens,
                item.num_prefill_tokens,
            )
        )
        return torch.cat(parts)

    result = runtime.conv._wrap_ple_short_conv_dispatch(upstream_dispatch)(layer, inputs, metadata, state, weights)

    torch.testing.assert_close(result, torch.cat(expected))
    torch.testing.assert_close(state, expected_state)


def test_single_token_prefill_uses_upstream_prefill_offsets(runtime):
    metadata = _builder(runtime).build(0, _common([1, 1], [True, False], computed=[0, 5]))
    assert metadata.non_spec_token_indx.tolist() == [1, 0]
    assert metadata.query_start_loc_p.tolist() == [0, 1]
    assert (metadata.num_prefills, metadata.num_decodes) == (1, 1)
    layer = _layer(runtime)
    inputs = torch.tensor([[10.0], [20.0]])

    def upstream_dispatch(_owner, tokens, _metadata, _state, _weights):
        # Decode receives the second input first; this marker distinguishes
        # execution groups and verifies the result's reverse permutation.
        return tokens + torch.tensor([[100.0], [1000.0]])

    result = runtime.conv._wrap_ple_short_conv_dispatch(upstream_dispatch)(layer, inputs, metadata, None, None)
    assert result.tolist() == [[1010.0], [120.0]]


def test_mixed_spec_group_gathers_accepted_counts_by_request(runtime):
    common = _common([1, 3, 1, 2], [False, False, True, False], slots=[4, 2, 1, 3])
    result = _builder(runtime, spec_tokens=2).build(0, common, num_accepted_tokens=torch.tensor([1, 2, 1, 3]))
    assert result.spec_token_indx.tolist() == [1, 2, 3, 5, 6]
    assert result.non_spec_token_indx.tolist() == [0, 4]
    assert result.spec_state_indices_tensor.tolist() == [2, 3]
    assert result.num_accepted_tokens.tolist() == [2, 3]
    assert result.query_start_loc_p.tolist() == [0, 1]
    assert result.normal_chunk is None


def test_spec_full_graph_padding_excluded_from_phase_masks(runtime):
    common = _common([3, 0, 0], [False, True, True], computed=[5, 0, 0])
    common.num_actual_tokens = 8
    builder = _builder(runtime, spec_tokens=2, full_graph=True)
    result = builder.build(0, common, num_reqs_actual=1, num_accepted_tokens=torch.tensor([2, 99, 99]))
    assert result.num_reqs == 1
    assert result.num_actual_tokens == 8
    assert result.spec_state_indices_tensor.tolist() == [1, 0, 0]
    assert result.spec_query_start_loc.tolist() == [0, 3, 3, 3]
    assert result.num_accepted_tokens.tolist() == [2, 1, 1]

    # Replay uses the same builder, so persistent tensors retain their storage.
    first = builder.build(0, common, num_reqs_actual=1, num_accepted_tokens=torch.tensor([1, 99, 99]))
    second = builder.build(0, common, num_reqs_actual=1, num_accepted_tokens=torch.tensor([2, 99, 99]))
    assert first.spec_state_indices_tensor.data_ptr() == second.spec_state_indices_tensor.data_ptr()
    assert first.spec_query_start_loc.data_ptr() == second.spec_query_start_loc.data_ptr()
    assert first.num_accepted_tokens.data_ptr() == second.num_accepted_tokens.data_ptr()
    assert first.num_accepted_tokens.tolist() == [2, 1, 1]


def test_pure_replay_chunks_exclude_graph_padding(runtime):
    common = _common([3, 2, 0], [False, False, True], computed=[7, 0, 0], slots=[3, 1, 2])
    common.num_actual_tokens = 8
    result = _builder(runtime).build(0, common, num_reqs_actual=2)
    assert result.num_actual_tokens == 5
    assert result.num_reqs == 2
    assert result.remaining_metadata is None
    assert result.remaining_token_indices is None
    assert result.normal_chunk.token_indices.tolist() == [0, 1, 2, 3, 4]
    assert result.normal_chunk.has_initial_states.tolist() == [True, False]


def test_installation_keeps_quantization_and_existing_wrappers(runtime, monkeypatch):
    class Layer:
        def _short_conv(self, inputs):
            return inputs

        def _short_conv_dilated_dispatch(self, inputs, *_args):
            return inputs

    class EmbeddingMethod:
        from_quant_config = staticmethod(lambda *_args: "upstream quantization")

    class UnquantizedMethod:
        pass

    class ModelSlimConfig:
        pass

    backend = SimpleNamespace(get_builder_cls=lambda: None)
    dependencies = {
        "vllm.models.qwen4_exp.amd.ple_layer": {"Qwen4ExpPLELayer": Layer},
        "vllm.models.qwen4_exp.common.ngram_embedding": {
            "Qwen4ExpPLEEmbeddingMethod": EmbeddingMethod,
            "Qwen4ExpPLEUnquantizedEmbeddingMethod": UnquantizedMethod,
        },
        "vllm.v1.attention.backends.short_conv_attn": {"PleShortConvAttentionBackend": backend},
        "vllm_ascend.models.qwen4_exp.ple_short_conv": {
            "patch_ple_chunk_execution": runtime.conv.patch_ple_chunk_execution
        },
        "vllm_ascend.models.qwen4_exp.short_conv_attn": {
            "PleShortConvAttentionMetadataBuilder": runtime.metadata.PleShortConvAttentionMetadataBuilder
        },
        "vllm_ascend.quantization.configs.modelslim_config": {"AscendModelSlimConfig": ModelSlimConfig},
    }
    for name, attributes in dependencies.items():
        dependency = ModuleType(name)
        dependency.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, dependency)
    source = MODEL_DIR / "ple.py"
    tree = ast.parse(source.read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "patch_upstream_ple_short_conv"
    ]
    namespace = {"Any": Any, "_wrap_ple_short_conv": lambda original: lambda self, inputs: original(self, inputs)}
    exec(compile(tree, str(source), "exec"), namespace)
    namespace["patch_upstream_ple_short_conv"]()
    first_conv = Layer._short_conv
    first_dispatch = Layer._short_conv_dilated_dispatch
    namespace["patch_upstream_ple_short_conv"]()
    assert Layer._short_conv is first_conv
    assert Layer._short_conv_dilated_dispatch is first_dispatch
    assert backend.get_builder_cls() is runtime.metadata.PleShortConvAttentionMetadataBuilder
    assert isinstance(EmbeddingMethod.from_quant_config(ModelSlimConfig(), "embedding"), UnquantizedMethod)
    assert EmbeddingMethod.from_quant_config(object(), "embedding") == "upstream quantization"


@pytest.mark.parametrize("phases,actual", [(None, 1), (torch.tensor([False]), 2)])
def test_missing_or_short_phase_metadata_fails(runtime, phases, actual):
    with pytest.raises(ValueError):
        runtime.metadata._ple_request_phase_masks(phases, actual)


def test_metadata_extends_only_the_upstream_contract(runtime):
    upstream_fields = {item.name for item in fields(runtime.upstream_metadata)}
    new_fields = {item.name for item in fields(runtime.metadata.PleShortConvAttentionMetadata)}
    assert new_fields - upstream_fields == {"normal_chunk", "remaining_metadata", "remaining_token_indices"}
