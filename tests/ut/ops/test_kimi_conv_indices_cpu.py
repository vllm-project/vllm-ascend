# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Kimi's metadata preparation without an NPU or vLLM import."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PAD_SLOT_ID = -1


@pytest.fixture
def runtime():
    class BaseBuilder:
        def __init__(self, spec, names, config, device):
            self.vllm_config = config
            self.compilation_config = config.compilation_config
            self.decode_cudagraph_max_bs = config.capture_capacity
            self.spec_sequence_masks = torch.empty(max(config.scheduler_config.max_num_seqs, config.capture_capacity))
            self.spec_state_indices_tensor = torch.empty(self.decode_cudagraph_max_bs, 4, dtype=torch.int32)
            self.non_spec_state_indices_tensor = torch.empty(self.decode_cudagraph_max_bs, dtype=torch.int32)

        def build(self, *args, **kwargs):
            return self._finalize_metadata(args[1])

        def build_for_cudagraph_capture(self, metadata):
            # Installed upstream capture dispatches through self.build().
            return self.build(0, metadata)

        def _reset_spec_decode_graph_inputs(self, count):
            self.spec_state_indices_tensor[:count].fill_(PAD_SLOT_ID)

    scope = dict(torch=torch, copy=copy, AscendGDNAttentionMetadataBuilder=BaseBuilder, PAD_SLOT_ID=PAD_SLOT_ID)
    sources = [
        ("vllm_ascend/ops/gdn_attn_builder.py", "AscendGDNHostMetadataBuilder"),
        ("vllm_ascend/ops/kda_pcp.py", "KimiKDAHostMetadataBuilder"),
        ("vllm_ascend/ops/kimi_kda.py", "_normalize_causal_cache_indices"),
    ]
    for path, name in sources:
        node = next(n for n in ast.parse((ROOT / path).read_text()).body if getattr(n, "name", None) == name)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), path, "exec"), scope)

    def make(capture_capacity=128):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=128, max_num_seqs=32),
            compilation_config=SimpleNamespace(
                static_forward_context={"kda": SimpleNamespace(use_contiguous_conv_indices=True)}
            ),
            capture_capacity=capture_capacity,
        )
        return scope["KimiKDAHostMetadataBuilder"](None, ["kda"], config, "cpu")

    return SimpleNamespace(
        make=make,
        normalize=scope["_normalize_causal_cache_indices"],
    )


def metadata(*, spec_indices=None, non_spec_indices=None, prefill=False):
    spec = None
    if spec_indices is not None:
        spec = SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                cache_indices=spec_indices,
                num_accepted_tokens=torch.ones(spec_indices.shape[0], dtype=torch.int32),
            )
        )
    non_spec = None
    if non_spec_indices is not None:
        non_spec = SimpleNamespace(causal_conv1d=SimpleNamespace(cache_indices=non_spec_indices))
    return SimpleNamespace(
        spec_decode_metadata=spec,
        spec_state_indices_tensor=spec_indices,
        non_spec_prefill_metadata=non_spec if prefill else None,
        non_spec_decode_metadata=None if prefill else non_spec,
    )


@pytest.mark.parametrize("rows", [0, 1, 4, 32])
def test_spec_first_column_is_prepared_once_and_recurrent_slots_survive(runtime, rows):
    builder = runtime.make()
    source = torch.arange(512, dtype=torch.int32).reshape(128, 4)[:rows]
    if rows > 1:
        source[-1].fill_(0)  # Existing graph padding uses the null state slot.
    original = source.clone()
    meta = metadata(spec_indices=source)
    result = builder.build(0, meta)
    prepared = result.spec_decode_metadata.spec_causal_conv1d.cache_indices
    assert prepared.ndim == 1 and prepared.is_contiguous()
    torch.testing.assert_close(prepared, original[:, 0])
    assert result.spec_state_indices_tensor is source
    torch.testing.assert_close(source, original)
    for _ in range(69):
        assert runtime.normalize(prepared) is prepared


def test_capture_replay_refreshes_stable_conv_pointer_and_reset_clears_it(runtime):
    builder = runtime.make()
    source = builder.spec_state_indices_tensor[:32]
    source.copy_(torch.arange(128, dtype=torch.int32).reshape(32, 4))
    captured = builder.build_for_cudagraph_capture(metadata(spec_indices=source))
    conv = captured.spec_decode_metadata.spec_causal_conv1d.cache_indices
    pointer = conv.data_ptr()
    source.add_(1000)
    replay = builder.build(0, metadata(spec_indices=source))
    assert replay.spec_decode_metadata.spec_causal_conv1d.cache_indices.data_ptr() == pointer
    torch.testing.assert_close(conv, source[:, 0])
    builder._reset_spec_decode_graph_inputs(32)
    torch.testing.assert_close(conv, torch.full((32,), PAD_SLOT_ID, dtype=torch.int32))
    assert torch.all(source == PAD_SLOT_ID)


@pytest.mark.parametrize("prefill", [False, True])
def test_non_spec_strided_vector_and_spec_use_independent_buffers(runtime, prefill):
    builder = runtime.make()
    spec_source = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    non_spec_source = torch.arange(40, 56, dtype=torch.int32).reshape(4, 4)[:, 0]
    meta = builder.build(0, metadata(spec_indices=spec_source, non_spec_indices=non_spec_source, prefill=prefill))
    spec = meta.spec_decode_metadata.spec_causal_conv1d.cache_indices
    non_spec = (meta.non_spec_prefill_metadata or meta.non_spec_decode_metadata).causal_conv1d.cache_indices
    assert spec.data_ptr() != non_spec.data_ptr()
    torch.testing.assert_close(spec, spec_source[:, 0])
    torch.testing.assert_close(non_spec, non_spec_source)
    assert runtime.normalize(non_spec) is non_spec


def test_contiguous_decode_stays_zero_copy_and_groups_do_not_alias(runtime):
    source = torch.arange(8, dtype=torch.int32)
    builders = [runtime.make(), runtime.make()]
    result = builders[0].build(0, metadata(non_spec_indices=source))
    prepared = result.non_spec_decode_metadata.causal_conv1d.cache_indices
    assert prepared.data_ptr() == source.data_ptr()
    torch.testing.assert_close(prepared, source)
    spec = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    first = builders[0].build(0, metadata(spec_indices=spec)).spec_decode_metadata.spec_causal_conv1d.cache_indices
    second = (
        builders[1].build(0, metadata(spec_indices=spec + 100)).spec_decode_metadata.spec_causal_conv1d.cache_indices
    )
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, spec[:, 0])
    torch.testing.assert_close(second, spec[:, 0] + 100)


def test_eager_batch_can_exceed_graph_request_capacity(runtime):
    builder = runtime.make(capture_capacity=16)
    source = torch.arange(128, dtype=torch.int32).reshape(32, 4)
    meta = builder.build(0, metadata(spec_indices=source))
    prepared = meta.spec_decode_metadata.spec_causal_conv1d.cache_indices
    assert prepared.shape == (32,)
    torch.testing.assert_close(prepared, source[:, 0])
