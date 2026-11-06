# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
    _validate_qualified_graph_config,
    validate_draft_graph_request_bounds,
    validate_draft_graph_review_requests,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy


def full8_config(dp=1, rows=2):
    parallel = NS(
        data_parallel_size=dp,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        all2all_backend="allgather_reducescatter",
        use_sequence_parallel_moe=True,
    )
    draft = NS(dtype="torch.bfloat16", hf_text_config=NS(max_position_embeddings=131072))
    speculative = NS(
        method="dflash",
        draft_model_config=draft,
        draft_parallel_config=NS(tensor_parallel_size=4, data_parallel_size=1),
        num_speculative_tokens=7,
        enable_adaptive_verification=False,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    return NS(
        speculative_config=speculative,
        parallel_config=parallel,
        scheduler_config=NS(max_num_seqs=rows),
        additional_config={"draft_window_size": 32768},
        model_config=NS(max_model_len=40960, hf_text_config=NS(max_position_embeddings=131072)),
        compilation_config=NS(
            cudagraph_mode=NS(name="FULL_DECODE_ONLY"),
            cudagraph_capture_sizes=[8 * n for n in range(1, rows + 1)],
            max_cudagraph_capture_size=8 * rows,
        ),
    )


@pytest.mark.parametrize("dp,rows,accepted", [(1, 1, True), (1, 2, True), (1, 3, False), (2, 1, True), (2, 2, False)])
def test_known_failed_full8_capacities_remain_closed(dp, rows, accepted):
    config = full8_config(dp, rows)
    if accepted:
        assert _validate_qualified_graph_config(config, "dflash_full8") is True
    else:
        with pytest.raises(NotImplementedError):
            _validate_qualified_graph_config(config, "dflash_full8")


def test_model_ceiling_and_actual_rows_are_independent():
    config = full8_config()
    config.model_config.max_model_len = 131073
    with pytest.raises(NotImplementedError, match="capacity"):
        _validate_qualified_graph_config(config, "dflash_full8")
    owner = NS(_draft_graph_review_limit=True, method="dflash", max_num_reqs=2)
    batch = NS(num_reqs=3, num_reqs_after_padding=3, req_ids=["a", "b", "c"], idx_mapping_np=np.arange(3))
    with pytest.raises(NotImplementedError, match="actual rows"):
        validate_draft_graph_review_requests(owner, batch)


def test_request_bounds_use_real_mapping_and_preemption():
    state = NS(
        prompt_len=NS(np=np.array([50, 100])), prefill_len=NS(np=np.array([50, 100])), max_seq_len=np.array([200, 300])
    )
    owner = NS(_draft_graph_request_state=state, method="dflash", num_query_per_req=8, max_model_len=512)
    batch = NS(req_ids=["second", "first"], idx_mapping_np=np.array([1, 0]))
    validate_draft_graph_request_bounds(owner, batch)
    owner._draft_graph_review_preempted_request_ids = {"first"}
    with pytest.raises(NotImplementedError, match="preemption"):
        validate_draft_graph_request_bounds(owner, batch)
    owner._draft_graph_review_preempted_request_ids.clear()
    state.max_seq_len[1] = 505
    with pytest.raises(NotImplementedError, match="lookahead"):
        validate_draft_graph_request_bounds(owner, batch)


def test_idle_context_does_not_open_unknown_active_shapes():
    owner = NS(_draft_graph_review_limit=True, method="dflash", max_num_reqs=1)
    batch = NS(num_reqs=0, num_reqs_after_padding=1, req_ids=[], idx_mapping_np=np.array([], dtype=np.int32))
    validate_draft_graph_review_requests(owner, batch, context_kv_only=True)


@pytest.mark.parametrize("method", ["eagle3", "dflash", "dspark"])
def test_static_window_policy_does_not_enable_dynamic_k(method):
    with pytest.raises(NotImplementedError):
        resolve_draft_window_policy(
            requested_window_size=1024,
            method=method,
            target_model_type="qwen3",
            builtin_v4_dspark=False,
            dynamic_k=True,
            is_dflash2=False,
        )


@pytest.mark.parametrize("windows,kind", [((None, None), "FULL"), ((2048, 2048), "SWA"), ((None, 2048), "MIXED")])
def test_native_attention_uses_consumers_not_full_kv_storage(windows, kind):
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.worker.v2.spec_decode.draft_window_native_attention import inspect_draft_native_attention

    layers = {}
    for index, window in enumerate(windows):
        impl = object.__new__(AscendAttentionBackendImpl)
        impl.sliding_window = window
        layers[f"draft.{index}"] = NS(impl=impl)
    owner = NS(
        draft_model_config=NS(hf_config=NS(architectures=["DFlashDraftModel"], num_hidden_layers=2)),
        vllm_config=NS(compilation_config=NS(static_forward_context=layers)),
        _group_causal=True,
        draft_attn_layer_names=set(layers),
        attn_groups=[[NS(layer_names=list(layers))]],
    )
    actual, native, issues = inspect_draft_native_attention(owner)
    assert actual == kind
    assert native == (() if kind == "FULL" else (2048,))
    assert issues == ()


def test_unknown_attention_does_not_reject_legal_window_off():
    from vllm_ascend.worker.v2.spec_decode.draft_window_native_attention import report_owner_draft_native_attention

    owner = NS(
        draft_model_config=NS(hf_config=NS(architectures=["UnknownDraft"], num_hidden_layers=1), enforce_eager=True),
        speculative_config=NS(enforce_eager=True),
        vllm_config=NS(compilation_config=NS(static_forward_context={})),
        _group_causal=True,
        draft_attn_layer_names={"draft.attn"},
        attn_groups=[],
    )
    report_owner_draft_native_attention(owner)
    owner.draft_window_policy = 1024
    with pytest.raises(NotImplementedError, match="known native attention"):
        report_owner_draft_native_attention(owner)


@pytest.mark.parametrize("mode_name,eager,gate", [("NONE", False, False), ("NONE", False, True), ("FULL", True, False)])
def test_dflash2_actual_eager_manager_does_not_require_graph_gate(monkeypatch, mode_name, eager, gate):
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backend import AttentionCGSupport

    from vllm_ascend import envs
    from vllm_ascend.attention import attention_v1  # noqa: F401
    from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    owner = object.__new__(AscendDFlash2Speculator)
    owner.speculative_config = NS(enforce_eager=eager)
    owner.attn_cg_support = NS(min_cg_support=AttentionCGSupport.ALWAYS)
    owner.vllm_config, owner.device, owner.update_stream = object(), torch.device("cpu"), object()
    owner.num_query_per_req = 8
    calls = []

    def manager(config, device, mode, *, decode_query_len):
        assert config is owner.vllm_config and device == owner.device and decode_query_len == 8
        calls.append(mode)
        return NS(cudagraph_mode=mode)

    monkeypatch.setattr(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", gate)
    monkeypatch.setattr(dflash.dflash_speculator, "DFlashCudaGraphManager", manager)
    owner.init_cudagraph_manager(getattr(CUDAGraphMode, mode_name))
    assert calls == [CUDAGraphMode.NONE]
    assert owner.query_cudagraph_manager.speculator is owner
    assert owner.query_cudagraph_manager.update_stream is owner.update_stream


@pytest.mark.parametrize(
    "mode_name,gate,error", [("FULL", False, ValueError), ("PIECEWISE", True, NotImplementedError)]
)
def test_dflash2_graph_gate_and_unsupported_mode_remain_closed(monkeypatch, mode_name, gate, error):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend import envs
    from vllm_ascend.attention import attention_v1  # noqa: F401
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    owner = object.__new__(AscendDFlash2Speculator)
    owner.speculative_config = NS(enforce_eager=False)
    monkeypatch.setattr(envs, "VLLM_ASCEND_EXPERIMENTAL_DRAFT_WINDOW_GRAPH", gate)
    with pytest.raises(error):
        owner.init_cudagraph_manager(getattr(CUDAGraphMode, mode_name))
