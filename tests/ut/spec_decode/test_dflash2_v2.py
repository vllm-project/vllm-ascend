#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the V2 DFlash2 speculator (worker/v2/spec_decode/dflash2)."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as upstream_dflash2

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
    _selector_walk_kernel_ascend,
)


@pytest.mark.parametrize("fail", [False, True])
def test_dflash_loader_temporarily_disables_profiling_chunk(monkeypatch, fail):
    additional_config = {"profiling_chunk_config": {"enabled": "true", "min_chunk": 64}}
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        additional_config=additional_config,
    )
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.vllm_config = config
    draft = object()

    def load(self, target_model, target_attn_layer_names):
        del target_model, target_attn_layer_names
        assert self.vllm_config is config
        assert self.vllm_config.additional_config["profiling_chunk_config"]["enabled"] is False
        if fail:
            raise RuntimeError("draft load failed")
        return draft

    monkeypatch.setattr(DFlashSpeculator, "load_draft_model", load)
    expected_context = pytest.raises(RuntimeError, match="draft load failed") if fail else nullcontext()
    with expected_context:
        assert speculator.load_draft_model(object(), set()) is draft

    assert config.additional_config is additional_config
    assert additional_config["profiling_chunk_config"]["enabled"] == "true"


def test_patch_swaps_upstream_selector_walk_kernel():
    import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as upstream

    import vllm_ascend.patch.worker.patch_v2.patch_dflash_speculator  # noqa: F401

    assert upstream._selector_walk_kernel is _selector_walk_kernel_ascend


def _spec_config(arch: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="dflash",
        enforce_eager=None,
        use_dspark=lambda: False,
        use_dflash=lambda: True,
        draft_model_config=SimpleNamespace(architectures=[arch]),
    )


def test_init_speculator_routes_dflash2_draft_model():
    cfg = SimpleNamespace(speculative_config=_spec_config("DFlash2DraftModel"))
    with patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as d2:
        assert init_speculator(cfg, torch.device("cpu")) is d2.return_value
        d2.assert_called_once_with(cfg, torch.device("cpu"))

    cfg = SimpleNamespace(speculative_config=_spec_config("DFlashDraftModel"))
    with (
        patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as d2,
        patch("vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator") as d1,
    ):
        assert init_speculator(cfg, torch.device("cpu")) is d1.return_value
        d2.assert_not_called()


@pytest.mark.parametrize("enforce_eager", [False, None, True])
@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE])
def test_init_cudagraph_manager_preserves_graph_mode_and_manager_references(monkeypatch, enforce_eager, mode):
    calls: list[CUDAGraphMode] = []
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.speculative_config = SimpleNamespace(enforce_eager=enforce_eager)
    speculator.update_stream = object()

    def initialize_parent(self, mode):
        calls.append(mode)
        self.query_cudagraph_manager = SimpleNamespace()

    # Keep AscendDFlashSpeculator's real delegation and graph-update wiring.
    monkeypatch.setattr(DFlashSpeculator, "init_cudagraph_manager", initialize_parent)

    speculator.init_cudagraph_manager(mode)

    assert calls == [CUDAGraphMode.NONE if enforce_eager else mode]
    assert speculator.query_cudagraph_manager.speculator is speculator
    assert speculator.query_cudagraph_manager.update_stream is speculator.update_stream


@pytest.mark.skipif(vllm_version_is("0.31.0"), reason="main-only context-KV capture callback")
def test_capture_resets_sampling_state_and_captures_dflash2_pipeline():
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.sample_indices = torch.full((12,), 11, dtype=torch.int64)
    speculator.sample_pos = torch.full((12,), 41, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.full((12,), 2, dtype=torch.int64)
    speculator._context_slot_mappings = torch.full((2, 16), 123, dtype=torch.int32)
    speculator.query_cudagraph_manager = SimpleNamespace(capture=MagicMock())
    speculator.input_buffers = object()
    speculator.block_tables = object()
    speculator.attn_groups = object()
    speculator.kv_cache_config = object()
    speculator.max_model_len = 2048
    speculator.max_num_tokens = 16
    speculator.num_speculative_steps = 3
    speculator._group_causal = {0: False, 1: True}
    speculator._precompute_context_kv = MagicMock()

    speculator.capture()

    torch.testing.assert_close(speculator.sample_indices, torch.zeros(12, dtype=torch.int64))
    torch.testing.assert_close(speculator.sample_pos, torch.zeros(12, dtype=torch.int64))
    torch.testing.assert_close(speculator.sample_idx_mapping, torch.full((12,), -1, dtype=torch.int64))
    torch.testing.assert_close(speculator._context_slot_mappings, torch.full((2, 16), -1, dtype=torch.int32))
    call = speculator.query_cudagraph_manager.capture.call_args
    forward = call.args[0]
    assert forward.__self__ is speculator
    assert forward.__func__ is upstream_dflash2.DFlash2Speculator._generate_draft
    assert call.args[1:] == (
        speculator.input_buffers,
        speculator.block_tables,
        speculator.attn_groups,
        speculator.kv_cache_config,
        2048,
    )
    assert call.kwargs["causal"] is speculator._group_causal
    call.kwargs["precompute_context_kv"](2)
    speculator._precompute_context_kv.assert_called_once_with(0, 8)


@pytest.mark.parametrize("probabilistic", [False, True])
def test_candidate_sampler_uses_ascend_walk_and_proposal_cache(monkeypatch, probabilistic):
    sampler_cls = getattr(upstream_dflash2, "CandidateSampler", None)
    if sampler_cls is None:
        pytest.skip("The shared candidate sampler is available on vLLM main")
    sampler = sampler_cls(max_num_reqs=4, num_steps=3, top_k=3, device=torch.device("cpu"))
    candidate_ids = torch.arange(36, dtype=torch.int64).view(2, 3, 6)[..., ::2]
    scores = torch.arange(54, dtype=torch.float32).view(2, 3, 3, 3).transpose(-1, -2)
    sample_pos = torch.arange(24, dtype=torch.int64)[::2]
    idx_mapping = torch.tensor([2, 2, 2, -1, -1, -1], dtype=torch.int64).repeat_interleave(2)[::2]
    temperatures = torch.tensor([0.0, 0.5, 1.0, 0.0])
    seeds = torch.arange(4, dtype=torch.int64)
    draft_tokens = torch.full((4, 3), -123, dtype=torch.int64)
    draft_logits = torch.full((4, 3, 64), -float("inf")) if probabilistic else None
    walk = MagicMock()
    cache = MagicMock()
    monkeypatch.setattr(upstream_dflash2, "_selector_walk_kernel", walk)
    monkeypatch.setattr(upstream_dflash2, "_cache_draft_logits_kernel", cache)

    sampler.sample(
        candidate_ids, scores, 2, sample_pos, idx_mapping, temperatures, seeds, draft_tokens, draft_logits, False
    )

    walk.__getitem__.assert_called_once_with((2,))
    launch = walk.__getitem__.return_value.call_args
    for actual, expected in zip(launch.args[:4], (scores, candidate_ids, sample_pos, idx_mapping)):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected)
    assert launch.args[4] is temperatures
    assert launch.args[5] is seeds
    assert launch.args[6] is draft_tokens
    assert launch.args[7] is sampler.scores
    assert launch.kwargs == {
        "num_steps": 3,
        "top_k": 3,
        "BLOCK_K": 4,
        "SAMPLE_PROBABILISTIC": probabilistic,
        "USE_FP64": False,
        "num_warps": 1,
    }
    if probabilistic:
        cache.__getitem__.assert_called_once_with((6,))
        cache_launch = cache.__getitem__.return_value.call_args
        assert cache_launch.args[0] is draft_logits
        assert cache_launch.args[1] is sampler.cached_candidate_ids
        torch.testing.assert_close(cache_launch.args[2], candidate_ids)
        assert cache_launch.args[3] is sampler.scores
        torch.testing.assert_close(cache_launch.args[4], idx_mapping)
        assert cache_launch.args[5:] == (draft_logits.stride(0), draft_logits.stride(1))
    else:
        cache.__getitem__.assert_not_called()


def test_greedy_walk_contract_reference():
    """Pure-Python mirror of ``_selector_walk_kernel_ascend``'s greedy walk
    (the kernel cannot launch on CPU): argmax per step, lowest index wins
    ties, chained via the previous step's winner, and padding rows
    (req_state < 0) never write."""
    num_steps, top_k = 2, 3
    candidates = torch.tensor(
        [[100, 200, 300], [400, 500, 600], [7, 8, 9], [70, 80, 90]],
        dtype=torch.int64,
    )
    scores = torch.full((4, top_k, top_k), 7.0)
    scores[0, 0] = torch.tensor([0.5, 0.9, 0.9])  # tie between 1 and 2
    scores[1, 1] = torch.tensor([-1.0, -2.0, 3.0])  # continues from candidate 1
    req_state = [0, 0, -1, -1]  # row 1 is padding

    tokens = [-123] * 4
    realized = [[-777.0] * top_k for _ in range(4)]
    for row in range(2):
        if req_state[row * num_steps] < 0:
            continue
        previous = 0
        for step in range(num_steps):
            flat = row * num_steps + step
            row_scores = scores[flat, previous].tolist()
            index = row_scores.index(max(row_scores))
            tokens[flat] = candidates[flat, index].item()
            realized[flat] = row_scores
            previous = index

    assert tokens == [200, 600, -123, -123]
    # The fixture stores fp32 scores, so compare with tolerance.
    assert realized[0] == pytest.approx([0.5, 0.9, 0.9])
    assert realized[1] == pytest.approx([-1.0, -2.0, 3.0])
    assert realized[2] == [-777.0] * top_k and realized[3] == [-777.0] * top_k


@pytest.mark.parametrize("num_reqs_padded", [2, 4])
def test_dflash_metadata_spans_padded_queries(monkeypatch, num_reqs_padded):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.input_batch = SimpleNamespace(num_reqs=2)
    speculator.num_query_per_req = 3
    speculator._group_causal = {0: False}
    seq_lens = torch.tensor([10, 20])
    metadata = {name: SimpleNamespace(actual_seq_lengths_q=[3, 6]) for name in ("draft.0", "draft.1")}
    speculator._build_uniform_attn_metadata = MagicMock(return_value=metadata)
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", nullcontext)

    assert speculator.build_draft_attn_metadatas(num_reqs_padded, seq_lens) == [metadata]

    kwargs = speculator._build_uniform_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == 2
    assert kwargs["seq_lens_cpu_upper_bound"] is seq_lens
    assert kwargs["causal"] is speculator._group_causal
    assert kwargs["batch_desc"].cg_mode == CUDAGraphMode.FULL
    assert kwargs["batch_desc"].num_tokens == num_reqs_padded * 3
    assert kwargs["batch_desc"].num_reqs == num_reqs_padded
    for layer in metadata.values():
        assert layer.actual_seq_lengths_q == list(range(3, num_reqs_padded * 3 + 1, 3))


@pytest.mark.parametrize("dummy_run,skip_attn", [(False, False), (False, True), (True, False), (True, True)])
def test_dflash_profiling_does_not_reuse_target_dp_counts(monkeypatch, dummy_run, skip_attn):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    batch, sync_state, result = object(), object(), object()
    received = []

    def propose(self, *args, **kwargs):
        assert self.input_batch is batch
        received.append(args[11])
        return result

    monkeypatch.setattr(DFlashSpeculator, "propose", propose)
    monkeypatch.setattr(dflash_module, "build_attn_metadata_wrapper", nullcontext)
    # vLLM #60754 renamed `dp_sync` -> `dp_sync_state` on main.
    dp_sync_kwarg = "dp_sync" if vllm_version_is("0.31.0") else "dp_sync_state"
    assert (
        speculator.propose(
            batch, *[None] * 10, **{dp_sync_kwarg: sync_state}, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn
        )
        is result
    )
    assert received == [None if dummy_run and skip_attn else sync_state]


@pytest.mark.parametrize("active_layers", [None, {"draft.1"}])
def test_dflash_attention_uses_only_draft_layers_and_int32_slots(monkeypatch, active_layers):
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.device = torch.device("cpu")
    speculator.max_num_tokens = 8
    speculator.draft_kv_cache_group_ids = [0, 1]
    speculator.draft_attn_layer_names = active_layers
    speculator.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    backends = {name: object() for name in ("draft.0", "draft.1")}
    layers = {name: SimpleNamespace(get_attn_backend=lambda name=name: backends[name]) for name in backends}
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=[name]) for name in backends])
    monkeypatch.setattr(DFlashSpeculator, "set_attn", MagicMock())
    monkeypatch.setattr(dflash_module, "get_layers_from_vllm_config", lambda *args: layers)
    # set_attn installs an upstream callback; restore it after this test.
    monkeypatch.setattr(dflash_module.dflash_speculator, "prepare_dflash_inputs", object())

    speculator.set_attn(None, cache, None, None, None)

    assert speculator.attn_backends == {
        name: backend for name, backend in backends.items() if active_layers is None or name in active_layers
    }
    assert speculator._context_slot_mappings.shape == (2, 8)
    assert speculator._context_slot_mappings.dtype == torch.int32
    assert torch.count_nonzero(speculator._context_slot_mappings) == 0


def test_dflash_input_preparation_keeps_physical_and_kernel_block_sizes_separate(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs_triton", kernel)
    inputs = [object() for _ in range(28)]
    inputs[17] = 16  # Kernel block size, distinct from physical KV blocks.
    inputs[-1] = True  # sample_from_anchor

    dflash_module.prepare_dflash_inputs_factory(128)(*inputs)

    kernel.assert_called_once_with(*inputs, kv_cache_block_size=128)


def test_dflash_input_preparation_restores_kernel_block_size_on_main(monkeypatch):
    """vLLM main forwards the physical block size, but the Ascend block table is
    kernel-block addressed, so the factory restores the kernel size per group."""
    kernel = MagicMock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs_triton", kernel)
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    block_tables = SimpleNamespace(
        kernel_block_sizes=[128],
        input_block_tables=[block_table],
    )
    inputs = [object() for _ in range(28)]
    inputs[16] = block_table
    inputs[17] = 384  # Physical KV block size forwarded by the DFlash caller.
    inputs[-1] = True  # sample_from_anchor

    if vllm_version_is("0.31.0"):
        dflash_module.prepare_dflash_inputs_factory(384)(*inputs)
    else:
        dflash_module.prepare_dflash_inputs_factory(384, block_tables)(*inputs)

    restored = kernel.call_args.args[17]
    if vllm_version_is("0.31.0"):
        # The release lane already forwards the kernel block size.
        assert restored == 384
    else:
        assert restored == 128
    assert kernel.call_args.kwargs["kv_cache_block_size"] == 384


@pytest.mark.skipif(vllm_version_is("0.31.0"), reason="main-only kernel block size restore")
def test_dflash_input_preparation_rejects_unknown_block_table(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs_triton", kernel)
    block_tables = SimpleNamespace(
        kernel_block_sizes=[128],
        input_block_tables=[torch.zeros(2, 4, dtype=torch.int32)],
    )
    inputs = [object() for _ in range(28)]
    # A block table that AscendBlockTables never handed out.
    inputs[16] = torch.zeros(2, 4, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="unknown to AscendBlockTables"):
        dflash_module.prepare_dflash_inputs_factory(384, block_tables)(*inputs)

    kernel.assert_not_called()
