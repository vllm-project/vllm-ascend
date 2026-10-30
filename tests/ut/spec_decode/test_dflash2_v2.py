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
from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
    _selector_walk_kernel_ascend,
)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_v2_loader_selects_flash_attn_and_restores_target_context(monkeypatch, causal, fail):
    from vllm.compilation import backends
    from vllm.config import AttentionConfig, CacheConfig, set_current_vllm_config
    from vllm.v1.attention import selector as selector_module
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as loader

    from vllm_ascend import platform
    from vllm_ascend.device.hardware_profile import AttentionBackendFamily

    target_config = SimpleNamespace(runner_type="generate", hf_text_config=SimpleNamespace(model_type="glm_moe_dsa"))
    draft_config = SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        hf_config=SimpleNamespace(num_hidden_layers=6, is_causal=causal),
    )
    spec = SimpleNamespace(
        method="dflash",
        target_model_config=target_config,
        draft_model_config=draft_config,
        enforce_eager=True,
        attention_backend=None,
        kv_cache_dtype="auto",
    )
    # Keep the real V2 loader's config cloning, causality and model-tag scopes.
    fields = {
        "use_v2_model_runner": True,
        "model_config": target_config,
        "speculative_config": spec,
        "attention_config": AttentionConfig(),
        "cache_config": CacheConfig(cache_dtype="fp8"),
        "load_config": object(),
        "parallel_config": SimpleNamespace(pipeline_parallel_size=1),
    }
    config_cls = make_dataclass("LoaderConfig", [(name, object) for name in fields])
    config = config_cls(**fields)
    draft_model = SimpleNamespace(model=object(), config=draft_config.hf_config)
    target_model = SimpleNamespace(model=object())
    monkeypatch.setattr(loader, "get_draft_load_config", lambda _: config.load_config)
    monkeypatch.setattr(loader, "get_pp_safe_draft_load_config", lambda value: value)
    monkeypatch.setattr(loader, "maybe_share_target_embed", lambda *_: None)
    monkeypatch.setattr(loader, "get_target_lm_head", lambda *_: None)
    monkeypatch.setattr(
        platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda _: True, attention_backend_family=AttentionBackendFamily.STANDARD),
    )
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *_: False)
    monkeypatch.setattr("vllm.platforms.current_platform", platform.NPUPlatform)
    monkeypatch.setattr(selector_module, "resolve_obj_by_qualname", lambda path: path)
    select = selector_module._cached_get_attn_backend
    target_selector = AttentionSelectorConfig(
        head_size=128, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="fp8", use_mla=True, use_sparse=True
    )
    initial_tag = backends.model_tag

    def get_model(*, vllm_config, model_config):
        assert model_config is draft_config
        assert vllm_config.model_config is target_config
        assert vllm_config.cache_config.cache_dtype == "auto"
        assert vllm_config.attention_config.use_non_causal is (not causal)
        assert backends.model_tag == "dflash_head"
        selector = target_selector._replace(
            use_mla=False, use_sparse=False, kv_cache_dtype="auto", use_non_causal=not causal
        )
        with set_current_vllm_config(vllm_config):
            assert select(None, selector) == "vllm_ascend.attention.fa_v1.AscendFlashAttentionBackend"
        if fail:
            raise RuntimeError("draft load failed")
        return draft_model

    monkeypatch.setattr(loader, "get_model", get_model)
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.vllm_config = config
    select.cache_clear()
    try:
        with set_current_vllm_config(config):
            assert select(None, target_selector) == "vllm_ascend.attention.sfa_v1.AscendSFABackend"
            expected = pytest.raises(RuntimeError, match="draft load failed") if fail else nullcontext()
            with expected:
                assert speculator.load_draft_model(target_model, set()) is draft_model
            assert backends.model_tag == initial_tag
            assert select(None, target_selector) == "vllm_ascend.attention.sfa_v1.AscendSFABackend"
        assert config.model_config is target_config
        assert config.cache_config.cache_dtype == "fp8"
        assert not config.attention_config.use_non_causal
        assert select.cache_info().currsize == 2
    finally:
        select.cache_clear()


@pytest.mark.parametrize("causal", [False, True])
def test_v2_flash_attn_uses_exact_lengths_after_rejection_and_reordering(monkeypatch, causal):
    import sys

    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    from vllm_ascend.attention.fa_v1 import AscendFlashAttentionImpl, AscendFlashAttentionMetadataBuilder

    num_query = 6
    layer_names = [f"draft.layers.{i}" for i in range(6)]
    builder = AscendFlashAttentionMetadataBuilder.__new__(AscendFlashAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
    group = SimpleNamespace(layer_names=layer_names, get_metadata_builder=lambda _: builder)
    kv_spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.arange_np = torch.arange(5, dtype=torch.int32).numpy()
    speculator.max_model_len = speculator.draft_max_seq_len = 1024
    speculator.dcp_size = 1
    speculator.input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        positions=torch.zeros(4 * num_query, dtype=torch.int64),
    )
    speculator.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    speculator.attn_groups = [[group]]
    speculator.kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=kv_spec)],
    )
    speculator.block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(4, 8, dtype=torch.int32)],
        slot_mappings=torch.arange(4 * num_query, dtype=torch.int32).view(1, -1),
    )
    ops = SimpleNamespace(
        flash_attn_metadata=MagicMock(side_effect=lambda *_args, **_kwargs: torch.empty(16)),
        flash_attn=MagicMock(side_effect=lambda query, *_args, **_kwargs: (query, None)),
    )
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", ops)
    impl = AscendFlashAttentionImpl.__new__(AscendFlashAttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.sliding_window = 8, 1, None
    impl.use_bnsd_kv_cache, impl.sinks, impl.scale = True, None, 128**-0.5
    impl.key_cache = impl.value_cache = torch.empty(0)
    upper_bounds = torch.tensor([110, 210, 0, 0], dtype=torch.int32)
    previous = None
    # Reuse buffer addresses while requests are reordered, rejected and removed.
    for step, (lengths, order, rejected, padded) in enumerate(
        [([100, 200], [0, 1], [0, 2], 3), ([210, 110], [1, 0], [3, 1], 2), ([220], [1], [4], 2)], start=1
    ):
        num_reqs = len(lengths)
        upper_bounds.zero_()
        upper_bounds[:num_reqs] = torch.tensor(lengths, dtype=torch.int32) + 20
        seq_lens = speculator.input_buffers.seq_lens
        seq_lens.zero_()
        exact_lengths = torch.tensor(lengths, dtype=torch.int32) - torch.tensor(rejected, dtype=torch.int32)
        seq_lens[:num_reqs].copy_(exact_lengths)
        boundaries = speculator.input_buffers.query_start_loc
        boundaries.fill_(num_reqs * num_query)
        boundaries[: num_reqs + 1].copy_(torch.arange(num_reqs + 1, dtype=torch.int32) * num_query)
        speculator.block_tables.input_block_tables[0][:num_reqs, 0] = torch.tensor(order, dtype=torch.int32)
        desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=padded * num_query, num_reqs=padded)

        def forbid_host_lengths(*_args, **_kwargs):
            raise AssertionError("exact draft lengths must remain on device")

        with monkeypatch.context() as no_host:
            for operation in ("tolist", "item", "cpu"):
                no_host.setattr(torch.Tensor, operation, forbid_host_lengths)
            with dflash_module.build_attn_metadata_wrapper():
                metadata_by_layer = speculator._build_uniform_attn_metadata(
                    batch_desc=desc,
                    num_reqs=num_reqs,
                    num_query_per_req=num_query,
                    seq_lens_cpu_upper_bound=upper_bounds,
                    step=num_query,
                    causal={0: causal},
                )
            metadata = metadata_by_layer[layer_names[0]]
            query = torch.zeros(padded * num_query, 8, 128, dtype=torch.bfloat16)
            output = torch.empty_like(query)
            for name in layer_names:
                assert metadata_by_layer[name] is metadata
                impl.forward_impl(query, None, None, None, metadata, output)
        assert metadata is not previous
        previous = metadata
        assert metadata.num_actual_tokens == num_reqs * num_query
        assert metadata.causal is causal
        assert metadata.seq_lens_gpu.data_ptr() == seq_lens.data_ptr()
        assert ops.flash_attn_metadata.call_count == step
        assert ops.flash_attn.call_count == step * len(layer_names)
        attrs = ops.flash_attn_metadata.call_args.kwargs
        expected_lengths = torch.ones(padded, dtype=torch.int32)
        expected_lengths[:num_reqs] = exact_lengths
        torch.testing.assert_close(attrs["seqused_kv"], expected_lengths)
        torch.testing.assert_close(attrs["cu_seqlens_q"], boundaries[: padded + 1])
        torch.testing.assert_close(metadata.block_tables[:num_reqs, 0], torch.tensor(order, dtype=torch.int32))


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


def test_eager_speculator_keeps_target_compilation_and_graph_config():
    spec = _spec_config("DFlash2DraftModel")
    spec.enforce_eager = True
    config = SimpleNamespace(
        speculative_config=spec,
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
    )
    with patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as draft:
        init_speculator(config, torch.device("cpu"))
    draft_config = draft.call_args.args[0]
    assert draft_config is not config
    assert draft_config.compilation_config is not config.compilation_config
    assert draft_config.compilation_config.mode == CompilationMode.NONE
    assert config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY


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
    # vLLM #60754 renamed `dp_sync` -> `dp_sync_state` on main.
    dp_sync_kwarg = "dp_sync_state"
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

    kernel.assert_called_once_with(*inputs, num_reqs_override=None, kv_cache_block_size=128)


@pytest.mark.skipif(vllm_version_is("0.32.0"), reason="main-only num_reqs_override")
def test_dflash_input_preparation_forwards_num_reqs_override(monkeypatch):
    """vLLM main (#58984) forwards the DP-synced request count positionally."""
    kernel = MagicMock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs_triton", kernel)
    inputs = [object() for _ in range(28)]
    inputs[-1] = True  # sample_from_anchor

    dflash_module.prepare_dflash_inputs_factory(128)(*inputs, 5)

    kernel.assert_called_once_with(*inputs, num_reqs_override=5, kv_cache_block_size=128)


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
    dflash_module.prepare_dflash_inputs_factory(384, block_tables)(*inputs)

    restored = kernel.call_args.args[17]
    assert restored == 128
    assert kernel.call_args.kwargs["kv_cache_block_size"] == 384


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
