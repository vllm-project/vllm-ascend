# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window.config import (
    _validate_graph_consumer_config,
    resolve_draft_window_policy,
    validate_graph_request_bounds,
    validate_graph_requests,
)


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


@pytest.mark.parametrize("profile", ["dflash_full8", "eagle3"])
@pytest.mark.parametrize("dp,rows,accepted", [(1, 1, True), (1, 2, True), (1, 3, False), (2, 1, True), (2, 2, False)])
def test_known_failed_graph_capacities_remain_closed(profile, dp, rows, accepted):
    config = full8_config(dp, rows)
    if profile == "eagle3":
        config.speculative_config.method = "eagle3"
        config.speculative_config.num_speculative_tokens = 3
        config.speculative_config.draft_parallel_config.enable_expert_parallel = False
        config.additional_config["draft_window_size"] = 1024
        config.compilation_config.cudagraph_capture_sizes = [4 * n for n in range(1, rows + 1)]
    if accepted:
        assert _validate_graph_consumer_config(config, profile) is True
    else:
        with pytest.raises(NotImplementedError):
            _validate_graph_consumer_config(config, profile)


def test_model_ceiling_and_actual_rows_are_independent():
    config = full8_config()
    config.model_config.max_model_len = 131073
    with pytest.raises(NotImplementedError, match="capacity"):
        _validate_graph_consumer_config(config, "dflash_full8")
    owner = NS(_draft_window_graph_support=True, method="dflash", max_num_reqs=2)
    batch = NS(num_reqs=3, num_reqs_after_padding=3, req_ids=["a", "b", "c"], idx_mapping_np=np.arange(3))
    with pytest.raises(NotImplementedError, match="actual rows"):
        validate_graph_requests(owner, batch)


def test_request_bounds_use_real_mapping_and_preemption():
    state = NS(
        prompt_len=NS(np=np.array([50, 100])), prefill_len=NS(np=np.array([50, 100])), max_seq_len=np.array([200, 300])
    )
    owner = NS(_draft_window_request_state=state, method="dflash", num_query_per_req=8, max_model_len=512)
    batch = NS(req_ids=["second", "first"], idx_mapping_np=np.array([1, 0]))
    validate_graph_request_bounds(owner, batch)
    owner._draft_window_preempted_request_ids = {"first"}
    with pytest.raises(NotImplementedError, match="preemption"):
        validate_graph_request_bounds(owner, batch)
    owner._draft_window_preempted_request_ids.clear()
    state.max_seq_len[1] = 505
    with pytest.raises(NotImplementedError, match="lookahead"):
        validate_graph_request_bounds(owner, batch)


def test_idle_context_does_not_open_unknown_active_shapes():
    owner = NS(_draft_window_graph_support=True, method="dflash", max_num_reqs=1)
    batch = NS(num_reqs=0, num_reqs_after_padding=1, req_ids=[], idx_mapping_np=np.array([], dtype=np.int32))
    validate_graph_requests(owner, batch, context_kv_only=True)


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
    from vllm_ascend.worker.v2.spec_decode.draft_window.backends import inspect_draft_native_attention

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
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.spec_decode.draft_window.backends import report_owner_draft_native_attention

    speculative = NS(enforce_eager=True, draft_model_config=NS(enforce_eager=True))
    owner = NS(
        draft_model_config=NS(hf_config=NS(architectures=["UnknownDraft"], num_hidden_layers=1), enforce_eager=True),
        speculative_config=speculative,
        vllm_config=NS(
            speculative_config=speculative,
            compilation_config=NS(static_forward_context={}, cudagraph_mode=CUDAGraphMode.NONE),
        ),
        _group_causal=True,
        draft_attn_layer_names={"draft.attn"},
        attn_groups=[],
    )
    report_owner_draft_native_attention(owner)
    owner.draft_window_policy = 1024
    with pytest.raises(NotImplementedError, match="known native attention"):
        report_owner_draft_native_attention(owner)


@pytest.mark.parametrize("mode_name,eager", [("NONE", False), ("FULL", True)])
def test_dflash2_actual_eager_manager_keeps_original_path(monkeypatch, mode_name, eager):
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backend import AttentionCGSupport

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

    monkeypatch.setattr(dflash.dflash_speculator, "DFlashCudaGraphManager", manager)
    owner.init_cudagraph_manager(getattr(CUDAGraphMode, mode_name))
    assert calls == [CUDAGraphMode.NONE]
    assert owner.query_cudagraph_manager.speculator is owner
    assert owner.query_cudagraph_manager.update_stream is owner.update_stream


def test_dflash2_unsupported_graph_mode_remains_closed():
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    owner = object.__new__(AscendDFlash2Speculator)
    owner.speculative_config = NS(enforce_eager=False)
    owner._draft_window_policy = 1024
    with pytest.raises(NotImplementedError):
        owner.init_cudagraph_manager(CUDAGraphMode.PIECEWISE)


@pytest.mark.parametrize(
    "window,spec_eager,model_eager,mode,expected",
    [
        (None, False, False, "FULL", False),
        (1024, True, False, "FULL", False),
        (1024, False, True, "FULL", False),
        (1024, False, False, "NONE", False),
        (1024, False, False, "FULL", True),
    ],
)
def test_existing_execution_settings_select_the_window_owner(
    monkeypatch, window, spec_eager, model_eager, mode, expected
):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.spec_decode.draft_window import config

    calls = []
    monkeypatch.setattr(config, "validate_draft_window_graph_config", lambda *a, **kw: calls.append(kw["support"]))
    vllm_config = NS(
        speculative_config=NS(enforce_eager=spec_eager, draft_model_config=NS(enforce_eager=model_eager)),
        compilation_config=NS(cudagraph_mode=getattr(CUDAGraphMode, mode)),
    )
    assert config._use_graph_window_owner(vllm_config, window, True) is expected
    assert calls == ([True] if expected else [])


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        ("method", "dflash"),
        ("model_type", "qwen3"),
        ("use_mla", True),
        ("tensor_parallel_size", 16),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 4),
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("enable_expert_parallel", False),
        ("num_speculative_tokens", 7),
        ("enable_adaptive_verification", True),
        ("draft_pcp", 4),
        ("draft_dcp", 4),
    ],
)
def test_graph_pcp_keeps_replicated_dspark_boundary(monkeypatch, field, value):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.device import hardware_profile
    from vllm_ascend.worker.v2.spec_decode.draft_window.config import validate_draft_window_graph_config

    monkeypatch.setattr(
        hardware_profile,
        "get_current_hardware_profile",
        lambda: NS(attention_backend_family=hardware_profile.AttentionBackendFamily.STANDARD),
    )
    config = full8_config()
    config.parallel_config.prefill_context_parallel_size = 4
    speculative = config.speculative_config
    speculative.method = "dspark"
    speculative.num_speculative_tokens = 8
    speculative.draft_model_config.use_mla = False
    speculative.draft_parallel_config.prefill_context_parallel_size = 1
    speculative.draft_parallel_config.decode_context_parallel_size = 1
    config.model_config.hf_text_config.model_type = "glm_moe_dsa"
    config.additional_config["draft_window_size"] = 1024
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    if field in ("method", "num_speculative_tokens", "enable_adaptive_verification"):
        setattr(speculative, field, value)
    elif field == "model_type":
        config.model_config.hf_text_config.model_type = value
    elif field == "use_mla":
        speculative.draft_model_config.use_mla = value
    elif field in ("draft_pcp", "draft_dcp"):
        setattr(
            speculative.draft_parallel_config,
            "prefill_context_parallel_size" if field == "draft_pcp" else "decode_context_parallel_size",
            value,
        )
    elif field is not None:
        setattr(config.parallel_config, field, value)
    if field is None:
        validate_draft_window_graph_config(config, support=None)
    else:
        with pytest.raises((NotImplementedError, RuntimeError)):
            validate_draft_window_graph_config(config, support=None)


@pytest.mark.parametrize("additional", [None, {}, {"draft_window_size": None}])
def test_window_off_skips_window_admission_and_owner_creation(monkeypatch, additional):
    import torch

    import vllm_ascend.worker.v2.spec_decode as factory
    from vllm_ascend.worker.v2.spec_decode.draft_window import config

    monkeypatch.setattr(config, "get_ascend_config", lambda: NS(draft_window_size=None))
    actual = object()
    calls = []
    original = NS(speculative_config=NS(), additional_config=additional)

    def original_factory(vllm_config, device, req_states):
        assert vllm_config is original and req_states is None
        calls.append(device)
        return actual

    def unexpected(*args, **kwargs):
        pytest.fail("Window OFF must not inspect Window-only model/topology/capacity")

    monkeypatch.setattr(factory, "_init_speculator", original_factory)
    monkeypatch.setattr(factory, "resolve_graph_support", unexpected)
    monkeypatch.setattr(factory, "_use_graph_window_owner", unexpected)
    assert factory.init_speculator(original, torch.device("cpu")) is actual
    assert calls == [torch.device("cpu")]


@pytest.mark.parametrize("mode_name", ["NONE", "FULL_DECODE_ONLY", "PIECEWISE"])
def test_dflash2_window_off_preserves_the_official_manager_choice(monkeypatch, mode_name):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    owner = object.__new__(AscendDFlash2Speculator)
    calls = []
    monkeypatch.setattr(
        dflash.AscendDFlashSpeculator, "init_cudagraph_manager", lambda self, mode: calls.append((self, mode))
    )
    mode = getattr(CUDAGraphMode, mode_name)
    owner.init_cudagraph_manager(mode)
    assert calls == [(owner, mode)]


def test_window_off_does_not_inspect_draft_native_consumers(monkeypatch):
    from vllm_ascend.worker.v2.spec_decode.draft_window import backends, runtime

    def unexpected(*args, **kwargs):
        pytest.fail("Window OFF must not scan or qualify native consumer inventory")

    monkeypatch.setattr(backends, "report_owner_draft_native_attention", unexpected)
    owner = NS()
    runtime.initialize_draft_window_owner(owner)
    assert vars(owner) == {}


def test_window_off_preserves_eagle_ar_capture_config(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph import _eagle_ar_capture_config

    config = NS(additional_config={}, speculative_config=NS(method="eagle3"))
    assert _eagle_ar_capture_config(config, CUDAGraphMode.FULL_DECODE_ONLY, 1) is config
