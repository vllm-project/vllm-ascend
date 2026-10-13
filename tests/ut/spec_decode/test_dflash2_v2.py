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

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
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


def test_init_cudagraph_manager_requires_enforce_eager(monkeypatch):
    calls: list[CUDAGraphMode] = []
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator.init_cudagraph_manager",
        lambda self, mode: calls.append(mode),
    )
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)

    # Eager drafting forces the draft aclgraph manager to NONE.
    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert calls == [CUDAGraphMode.NONE]

    # Graph mode is rejected until the Ascend walk kernel is capturable, and
    # must fail before delegating to the parent manager.
    speculator.speculative_config = SimpleNamespace(enforce_eager=False)
    with pytest.raises(NotImplementedError, match="graph mode"):
        speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert calls == [CUDAGraphMode.NONE]


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
    assert (
        speculator.propose(
            batch, *[None] * 10, dp_sync=sync_state, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn
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
