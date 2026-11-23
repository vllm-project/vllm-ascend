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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.attention.large_head_paged_verify import AscendLargeHeadPagedMetadata
from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, SharedSource
from vllm_ascend.worker.v2 import spec_decode as v2_spec_decode
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import (
    AscendGemma4Speculator,
)
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import (
    AscendTargetDependentARSpeculator,
)


def _speculative_config():
    config = MagicMock()
    config.method = "mtp"
    # Falsy: init_speculator's enforce_eager branch copies compilation_config,
    # which the stub vllm_config does not carry.
    config.enforce_eager = None
    config.use_gemma4_mtp.return_value = True
    config.use_step3p5_mtp.return_value = False
    config.use_dspark.return_value = False
    config.use_dflash.return_value = False
    config.use_eagle.return_value = True
    return config


def test_routes_gemma4_mtp_to_ascend_gemma4_speculator():
    vllm_config = SimpleNamespace(speculative_config=_speculative_config())
    expected = object()
    with patch(
        "vllm_ascend.worker.v2.spec_decode.gemma4.speculator.AscendGemma4Speculator",
        return_value=expected,
    ) as speculator_cls:
        result = v2_spec_decode.init_speculator(vllm_config, torch.device("cpu"))

    assert result is expected
    speculator_cls.assert_called_once()


def test_eagle_speculator_is_not_reached_for_gemma4():
    vllm_config = SimpleNamespace(speculative_config=_speculative_config())
    with patch(
        "vllm_ascend.worker.v2.spec_decode.gemma4.speculator.AscendGemma4Speculator",
        return_value=object(),
    ):
        v2_spec_decode.init_speculator(vllm_config, torch.device("cpu"))

    vllm_config.speculative_config.use_eagle.assert_not_called()


def test_gemma4_speculator_mro():
    # Order invariants, not a fixed tuple: main may insert extra bases between
    # ours (e.g. LmheadTPDraftSamplingMixin) without breaking this contract.
    mro = AscendGemma4Speculator.__mro__
    order = (
        AscendGemma4Speculator,
        AscendTargetDependentARSpeculator,
        Gemma4Speculator,
        TargetDependentARSpeculator,
    )
    indices = [mro.index(cls) for cls in order]
    assert indices == sorted(indices), f"MRO order broken: {mro}"
    assert AscendGemma4Speculator.advance_draft_positions is Gemma4Speculator.advance_draft_positions
    assert AscendGemma4Speculator._share_embeddings is Gemma4Speculator._share_embeddings


def test_sync_kv_sharing_target_to_impl():
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    synced_impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    untouched_impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    draft_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(
                            impl=synced_impl,
                            kv_sharing_target_layer_name="target.attn",
                        )
                    )
                ),
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(
                            impl=untouched_impl,
                            kv_sharing_target_layer_name=None,
                        )
                    )
                ),
            ]
        )
    )

    speculator._sync_kv_sharing_target_to_impl(draft_model)

    assert synced_impl.kv_sharing_target_layer_name == "target.attn"
    assert untouched_impl.kv_sharing_target_layer_name is None


def test_sync_kv_sharing_target_to_impl_without_layers():
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    speculator._sync_kv_sharing_target_to_impl(SimpleNamespace(model=SimpleNamespace(layers=[])))


def test_prefill_rebinds_prefill_nocache_state():
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    original = {
        "shared": SimpleNamespace(attn_state=AscendAttentionState.PrefillNoCache),
        "other": SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly),
    }
    with patch.object(AscendTargetDependentARSpeculator, "_prefill") as parent_prefill:
        speculator._prefill(2, 8, original, None)

    passed = parent_prefill.call_args[0][2]
    assert passed is not original
    assert passed["shared"].attn_state is AscendAttentionState.PrefillCacheHit
    assert passed["other"].attn_state is AscendAttentionState.DecodeOnly
    assert original["shared"].attn_state is AscendAttentionState.PrefillNoCache


def test_prefill_keeps_cache_reading_states():
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    original = {"shared": SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)}
    with patch.object(AscendTargetDependentARSpeculator, "_prefill") as parent_prefill:
        speculator._prefill(2, 8, original, None)

    assert parent_prefill.call_args[0][2] is original


def test_build_fia_params_keeps_per_layer_block_tables_across_kv_groups():
    """Sliding (256-dim) and global (512-dim) groups must each receive their own block_table."""
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    speculator.use_dcp = False
    # vLLM v0.33.0 / main expose max_model_len as a read-only view of vllm_config.
    speculator.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    speculator.num_speculative_steps = 3
    speculator.draft_attn_layer_names = ["draft.swa.0", "draft.global.0"]
    speculator.input_batch = SimpleNamespace(num_reqs=1, seq_lens_np=[10])

    bt_swa = torch.tensor([[1, 2], [0, 0]], dtype=torch.int32)
    bt_global = torch.tensor([[7, 8], [0, 0]], dtype=torch.int32)

    swa_meta = AscendMetadata(
        block_tables=bt_swa,
        actual_seq_lengths_q=[4, 8],
        seq_lens_list=[10, 1],
    )
    global_meta = AscendLargeHeadPagedMetadata(
        block_tables=bt_global,
        actual_seq_lengths_q=[4, 8],
        seq_lens_list=[10, 1],
        use_pa_graph=False,
    )
    draft_meta = {"draft.swa.0": swa_meta, "draft.global.0": global_meta}

    prefill_params = speculator.build_fia_params(2, draft_meta, is_draft_model_prefill=True)
    by_layer_prefill = {p["layer_name"]: p for p in prefill_params}
    assert by_layer_prefill["draft.swa.0"]["block_table"] is bt_swa
    assert by_layer_prefill["draft.global.0"]["block_table"] is bt_global
    assert "atten_mask" not in by_layer_prefill["draft.global.0"]
    assert by_layer_prefill["draft.global.0"]["actual_seq_kvlen"] == [10, 4]

    decode_params = speculator.build_fia_params(2, draft_meta, is_draft_model_prefill=False)
    swa_decode = [p for p in decode_params if p["layer_name"] == "draft.swa.0"]
    global_decode = [p for p in decode_params if p["layer_name"] == "draft.global.0"]
    assert len(swa_decode) == 2 and len(global_decode) == 2
    assert all(p["block_table"] is bt_swa for p in swa_decode)
    # Per-layer slicing produces a view sharing storage with bt_global, never bt_swa.
    assert all(p["block_table"].data_ptr() == bt_global.data_ptr() for p in global_decode)
    # Gemma4 advance_draft_positions is False: KV length stays at 10 across steps.
    assert all(p["actual_seq_lengths_kv"] == [10, 0] for p in swa_decode)
    assert all(p["actual_seq_kvlen"] == [10, 0] for p in global_decode)


def test_update_decode_attn_metadata_respects_advance_draft_positions():
    """Gemma4 (Q-only) freezes draft KV length and clamps warmup zeros; Eagle still advances."""
    target_buffers = AscendInputBuffers.__new__(AscendInputBuffers)
    target_buffers.seq_lens_cpu = torch.tensor([0, 0, 0, 0], dtype=torch.int32)

    gemma_spec = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    gemma_spec.use_dcp = False
    gemma_spec.attn_architecture = "GQA"
    gemma_spec.max_num_reqs = 4
    gemma_spec.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    gemma_spec.target_input_buffers = target_buffers

    gemma_meta = {"draft.0": AscendMetadata(seq_lens_cpu=torch.zeros(4, dtype=torch.int32))}
    gemma_spec._update_decode_attn_metadata(gemma_meta, step=2, num_reqs=2)
    # Warmup active rows [0, 1] clamp to 1; padded rows [2, 3] remain 0.
    assert gemma_meta["draft.0"].seq_lens_list == [1, 1, 0, 0]
    assert gemma_meta["draft.0"].seq_lens_cpu.tolist() == [1, 1, 0, 0]

    # With populated seq_lens, Gemma4 stays fixed while Eagle advances +step.
    target_buffers.seq_lens_cpu.copy_(torch.tensor([12, 20, 0, 0], dtype=torch.int32))
    gemma_spec._update_decode_attn_metadata(gemma_meta, step=2, num_reqs=2)
    assert gemma_meta["draft.0"].seq_lens_list == [12, 20, 0, 0]

    eagle_spec = AscendTargetDependentARSpeculator.__new__(AscendTargetDependentARSpeculator)
    eagle_spec.use_dcp = False
    eagle_spec.attn_architecture = "GQA"
    eagle_spec.max_num_reqs = 4
    eagle_spec.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    eagle_spec.target_input_buffers = target_buffers

    eagle_meta = {"draft.0": AscendMetadata(seq_lens_cpu=torch.zeros(4, dtype=torch.int32))}
    eagle_spec._update_decode_attn_metadata(eagle_meta, step=2, num_reqs=2)
    assert eagle_meta["draft.0"].seq_lens_list == [14, 22, 0, 0]


@pytest.mark.parametrize("padded_reqs", [2, 4])
def test_build_draft_attn_metadatas_gqa_updatable_fast_path(padded_reqs):
    """GQA + UpdatableGraph draft decode reuses or slices block_tables without rebuilding full metadata."""
    speculator = AscendGemma4Speculator.__new__(AscendGemma4Speculator)
    speculator.attn_architecture = "GQA"
    speculator.use_dcp = False
    speculator.attn_backend = object()
    speculator.draft_attn_layer_names = {"draft.swa.0", "draft.global.0"}
    bt_swa_target = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    bt_global_target = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
    bt_swa_pool = torch.tensor([[1, 2], [3, 4], [0, 0], [0, 0]], dtype=torch.int32)
    bt_global_pool = torch.tensor([[5, 6], [7, 8], [0, 0], [0, 0]], dtype=torch.int32)

    swa_meta = AscendMetadata(block_tables=bt_swa_target, seq_lens_list=[10, 8])
    global_meta = AscendLargeHeadPagedMetadata(
        block_tables=bt_global_target,
        seq_lens_list=[10, 8],
        use_pa_graph=False,
    )
    target_meta = AscendMetadata(block_tables=bt_swa_target)
    speculator.model_state = SimpleNamespace(
        attn_metadata={"draft.swa.0": swa_meta, "draft.global.0": global_meta, "target.0": target_meta}
    )
    speculator.attn_groups = [
        [SimpleNamespace(layer_names=["draft.swa.0"])],
        [SimpleNamespace(layer_names=["draft.global.0"])],
    ]
    speculator.block_tables = SimpleNamespace(input_block_tables=(bt_swa_pool, bt_global_pool))
    speculator._init_decode_draft_attn_metadatas = MagicMock()
    speculator._update_decode_attn_metadata = MagicMock()

    with patch(
        "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator.use_updatable_graph",
        return_value=True,
    ):
        result = speculator.build_draft_attn_metadatas(padded_reqs, padded_reqs, is_draft_model_prefill=False)

    assert len(result) == 1
    assert set(result[0]) == {"draft.swa.0", "draft.global.0"}
    speculator._init_decode_draft_attn_metadatas.assert_not_called()
    speculator._update_decode_attn_metadata.assert_not_called()
    if padded_reqs == 2:
        assert result[0]["draft.swa.0"] is swa_meta
        assert result[0]["draft.global.0"] is global_meta
    else:
        assert result[0]["draft.swa.0"] is not swa_meta
        assert result[0]["draft.swa.0"].block_tables.shape == (4, 2)
        assert result[0]["draft.global.0"].block_tables.shape == (4, 2)
        assert swa_meta.block_tables is bt_swa_target

    # Fallback cases: use_pa_graph, use_dcp, or non-updatable backend.
    for setup_fn, updatable in (
        (lambda: setattr(global_meta, "use_pa_graph", True), True),
        (lambda: setattr(speculator, "use_dcp", True), True),
        (lambda: None, False),
    ):
        global_meta.use_pa_graph = False
        speculator.use_dcp = False
        speculator.input_batch = SimpleNamespace(num_reqs=2)
        setup_fn()
        speculator._init_decode_draft_attn_metadatas = MagicMock(return_value=[{"draft.swa.0": swa_meta}])
        speculator._update_decode_attn_metadata = MagicMock()
        with patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator.use_updatable_graph",
            return_value=updatable,
        ):
            speculator.build_draft_attn_metadatas(4, 4, is_draft_model_prefill=False)
        speculator._init_decode_draft_attn_metadatas.assert_called_once()
        speculator._update_decode_attn_metadata.assert_called_once()


def test_updatable_graph_bind_and_shared_source_lookup():
    """SharedSource indexes by layer_name once and GraphUpdateTask.bind updates matching keys cleanly."""
    source = SharedSource(
        [
            {"layer_name": "l0", "block_table": "bt0", "actual_seq_kvlen": [5]},
            {"layer_name": "l1", "block_table": "bt1", "actual_seq_kvlen": [7]},
            {"layer_name": "l0", "block_table": "bt0_s2", "actual_seq_kvlen": [6]},
        ]
    )
    l0_entries = source.get(SimpleNamespace(layer_name="l0"))
    assert l0_entries == [
        {"block_table": "bt0", "actual_seq_kvlen": [5]},
        {"block_table": "bt0_s2", "actual_seq_kvlen": [6]},
    ]
    assert source.get(SimpleNamespace(layer_name="missing")) == ()

    provider = SimpleNamespace(layer_name="l0")
    task = GraphUpdateTask(
        operation=lambda **_: None,
        kwargs={"query": "q_orig", "block_table": "bt_orig", "actual_seq_kvlen": [1], "sparse_mode": 3},
        provider=provider,
        provider_index=0,
        handle="h0",
        event="e0",
    )
    bound = task.bind({"block_table": "bt_new", "actual_seq_kvlen": [9, 10], "unused_key": "ignored"})
    assert bound.kwargs == {
        "query": "q_orig",
        "block_table": "bt_new",
        "actual_seq_kvlen": [9, 10],
        "sparse_mode": 3,
    }
    assert "unused_key" not in bound.kwargs
    assert task.kwargs["block_table"] == "bt_orig"


def _make_gemma4_test_speculator(cls, *, target_seq_lens):
    speculator = cls.__new__(cls)
    speculator.use_dcp = False
    speculator.pcp_manager = None
    speculator.dp_size = 1
    speculator.enable_adaptive_verification = False
    speculator.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    speculator.num_speculative_steps = 3
    speculator.draft_attn_layer_names = ["draft.swa.0", "draft.global.0"]
    speculator._num_rejected = None
    speculator._num_rejected_cpu = None
    speculator.input_batch = SimpleNamespace(num_reqs=len(target_seq_lens), seq_lens_np=target_seq_lens)
    return speculator


def _call_propose_with_hook(speculator, num_rejected, hook):
    dummy = torch.zeros(1)
    with (
        patch("vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils.lmhead_tp_enable", return_value=False),
        patch.object(TargetDependentARSpeculator, "propose", side_effect=hook),
    ):
        return speculator.propose(
            speculator.input_batch,
            {},
            {},
            dummy,
            None,
            dummy,
            num_rejected,
            dummy,
            dummy,
            dummy,
            dummy,
        )


@pytest.mark.parametrize("use_pa_graph", [False, True])
def test_gemma4_partial_rejection_updates_both_pa_and_fia_graph_params(use_pa_graph: bool):
    """Query-only Gemma4 graph replay must pass target - num_rejected to both FIA and PA."""
    speculator = _make_gemma4_test_speculator(
        AscendGemma4Speculator,
        target_seq_lens=np.array([11, 20], dtype=np.int32),
    )
    swa_meta = AscendMetadata(block_tables=torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32))
    global_meta = AscendLargeHeadPagedMetadata(
        block_tables=torch.tensor([[5, 6], [7, 8], [0, 0]], dtype=torch.int32),
        seq_lens=torch.zeros(3, dtype=torch.int32),
        use_pa_graph=use_pa_graph,
    )
    draft_meta = {"draft.swa.0": swa_meta, "draft.global.0": global_meta}
    captured: list[list[dict]] = []

    def _run_hook(*_args, **_kwargs):
        captured.append(speculator.build_fia_params(3, draft_meta, is_draft_model_prefill=False))
        return torch.zeros((2, 3), dtype=torch.int32)

    _call_propose_with_hook(speculator, torch.tensor([2, 0], dtype=torch.int32), _run_hook)

    assert len(captured) == 1
    decode_params = captured[0]
    swa_params = [p for p in decode_params if p["layer_name"] == "draft.swa.0"]
    global_params = [p for p in decode_params if p["layer_name"] == "draft.global.0"]
    assert len(swa_params) == 2 and len(global_params) == 2
    assert all(p["actual_seq_lengths_kv"] == [9, 20, 0] for p in swa_params)
    assert all(p["actual_seq_kvlen"] == [9, 20, 0] for p in swa_params)
    if use_pa_graph:
        assert all(
            p["context_lens"].dtype == torch.int32 and p["context_lens"].tolist() == [9, 20, 0] for p in global_params
        )
        assert all("actual_seq_kvlen" not in p for p in global_params)
    else:
        assert all(p["actual_seq_kvlen"] == [9, 20, 0] for p in global_params)
        assert all("context_lens" not in p for p in global_params)
    assert speculator._num_rejected is None and speculator._num_rejected_cpu is None

    def _raise_hook(*_args, **_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _call_propose_with_hook(speculator, torch.tensor([2, 0], dtype=torch.int32), _raise_hook)
    assert speculator._num_rejected is None and speculator._num_rejected_cpu is None


def test_gemma4_partial_rejection_metadata_update_clamp_and_eagle_non_regression():
    """Verify eager metadata update, lower clamp >=1, and unchanged Eagle +step behavior."""
    target_buffers = AscendInputBuffers.__new__(AscendInputBuffers)
    target_buffers.seq_lens_cpu = torch.tensor([11, 20, 0, 0], dtype=torch.int32)

    gemma_spec = _make_gemma4_test_speculator(
        AscendGemma4Speculator,
        target_seq_lens=np.array([11, 20], dtype=np.int32),
    )
    gemma_spec.attn_architecture = "GQA"
    gemma_spec.max_num_reqs = 4
    gemma_spec.target_input_buffers = target_buffers
    pa_lens = torch.zeros(4, dtype=torch.int32)
    gemma_meta = {
        "draft.swa.0": AscendMetadata(seq_lens_cpu=torch.zeros(4, dtype=torch.int32)),
        "draft.global.0": AscendLargeHeadPagedMetadata(
            seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
            pa_context_lens=pa_lens,
            use_pa_graph=True,
        ),
    }

    _call_propose_with_hook(
        gemma_spec,
        torch.tensor([2, 0], dtype=torch.int32),
        lambda *_a, **_kw: gemma_spec._update_decode_attn_metadata(gemma_meta, step=2, num_reqs=2),
    )
    assert gemma_meta["draft.swa.0"].seq_lens_list == [9, 20, 0, 0]
    assert gemma_meta["draft.swa.0"].seq_lens_cpu.tolist() == [9, 20, 0, 0]
    assert gemma_meta["draft.global.0"].seq_lens_list == [9, 20, 0, 0]
    assert pa_lens.tolist() == [9, 20, 0, 0]

    clamp_spec = _make_gemma4_test_speculator(AscendGemma4Speculator, target_seq_lens=np.array([3], dtype=np.int32))
    clamp_meta = {"draft.swa.0": AscendMetadata(block_tables=torch.zeros((2, 1), dtype=torch.int32))}
    clamped: list[list[dict]] = []
    _call_propose_with_hook(
        clamp_spec,
        torch.tensor([3], dtype=torch.int32),
        lambda *_a, **_kw: clamped.append(clamp_spec.build_fia_params(2, clamp_meta, is_draft_model_prefill=False)),
    )
    assert all(p["actual_seq_lengths_kv"] == [1, 0] for p in clamped[0])

    eagle_spec = _make_gemma4_test_speculator(
        AscendTargetDependentARSpeculator,
        target_seq_lens=np.array([11, 20], dtype=np.int32),
    )
    eagle_meta = {"draft.swa.0": AscendMetadata(block_tables=torch.zeros((3, 1), dtype=torch.int32))}
    eagle_captured: list[list[dict]] = []
    _call_propose_with_hook(
        eagle_spec,
        torch.tensor([2, 0], dtype=torch.int32),
        lambda *_a, **_kw: eagle_captured.append(
            eagle_spec.build_fia_params(3, eagle_meta, is_draft_model_prefill=False)
        ),
    )
    assert eagle_captured[0][0]["actual_seq_lengths_kv"] == [12, 21, 0]
    assert eagle_captured[0][1]["actual_seq_lengths_kv"] == [13, 22, 0]
    assert eagle_spec._num_rejected is None
