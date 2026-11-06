# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolTailBackend,
    AscendIndexerKPoolTailMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAMetadataBuilder
from vllm_ascend.models.glm5next.adaptive import glm_mtp_adaptive_query_len, validate_glm_mtp_adaptive
from vllm_ascend.models.glm5next.ops import kda as model_kda_ops
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend, AscendGDNHostMetadataBuilder
from vllm_ascend.ops.triton.kda import fused_recurrent_kda as recurrent_ops

conv_ops = importlib.import_module("vllm_ascend.models.glm5next.ops.causal_conv1d")


def _config(method="mtp", adaptive=True, model_type="glm5_next_text"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        speculative_config=SimpleNamespace(
            method=method, enable_adaptive_verification=adaptive, num_speculative_tokens=5
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(use_replayssm=False),
    )


def test_adaptive_manager_accepts_all_glm_target_cache_planes(monkeypatch):
    groups = []
    for backend, builder in (
        (AscendGDNAttentionBackend, AscendGDNHostMetadataBuilder),
        (AscendSFABackend, AscendSFAMetadataBuilder),
        (AscendIndexerKPoolBackend, AscendIndexerKPoolMetadataBuilder),
        (AscendIndexerKPoolTailBackend, AscendIndexerKPoolTailMetadataBuilder),
    ):
        groups.append(
            SimpleNamespace(
                backend=backend,
                kv_cache_spec=None,
                layer_names=[backend.__name__],
                get_metadata_builder=lambda index, cls=builder: cls,
            )
        )
    manager = Mock()
    monkeypatch.setattr(adaptive_verification, "AdaptiveVerificationManager", manager)
    kwargs = dict(
        enable_adaptive_verification=True,
        attn_groups=[groups],
        req_states=SimpleNamespace(num_speculative_steps=5),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        num_bonus_tokens=1,
        max_total_logits=32,
        vllm_config=_config(),
    )
    assert adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs) is manager.return_value
    kwargs["req_states"].num_speculative_steps = 6
    with pytest.raises(ValueError, match="per-request query lengths vary"):
        adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs)
    manager.assert_called_once()


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text", "qwen3_next"])
@pytest.mark.parametrize("method", ["mtp", "deepseek_mtp", "dspark", "dflash"])
@pytest.mark.parametrize("enabled", [False, True])
def test_varlen_capabilities_are_limited_to_glm_mtp(model_type, method, enabled):
    config = _config(method, enabled, model_type)
    expected = 6 if model_type.startswith("glm5_next") and method in ("mtp", "deepseek_mtp") and enabled else None
    assert glm_mtp_adaptive_query_len(config) == expected
    for builder in (AscendIndexerKPoolMetadataBuilder, AscendIndexerKPoolTailMetadataBuilder, AscendSFAMetadataBuilder):
        assert builder.get_varlen_cudagraph_max_query_len(config, None) == expected


@pytest.mark.parametrize("setting", ["prefill_context_parallel_size", "decode_context_parallel_size", "use_replayssm"])
def test_adaptive_rejects_unadapted_partition_and_recovery(setting):
    config = _config()
    target = config.cache_config if setting == "use_replayssm" else config.parallel_config
    setattr(target, setting, True if setting == "use_replayssm" else 2)
    with pytest.raises(ValueError, match="GLM MTP adaptive verification"):
        validate_glm_mtp_adaptive(config)
    config.speculative_config.enable_adaptive_verification = False
    validate_glm_mtp_adaptive(config)


def test_kpool_state_table_capture_and_replay_keep_maximum_width():
    builder = AscendIndexerKPoolMetadataBuilder.__new__(AscendIndexerKPoolMetadataBuilder)
    builder.vllm_config = _config()
    builder.device = torch.device("cpu")
    builder.compress_ratio = 4
    builder._captured_state_tables = {}
    builder._state_table_scratch = {}
    builder._state_table_shapes = {}
    common = SimpleNamespace(num_reqs=4, num_input_tokens=8, max_query_len=1)
    captured = builder._get_state_table(common, True, 100)
    assert captured.shape == (4, 3)
    torch.testing.assert_close(captured, torch.tensor([[1, 5, 5], [2, 6, 6], [3, 7, 7], [4, 8, 8]], dtype=torch.int32))
    for width in (6, 1, 4):
        common.max_query_len = width
        replay = builder._get_state_table(common, False, 100)
        assert replay.data_ptr() == captured.data_ptr()
    # Enabling AV must not cap long prompt chunks at the decode width.
    common.max_query_len = 513
    assert builder._get_state_table(common, False, 100).shape == (4, 130)


def test_kpool_tail_retains_full_rollback_window_at_one_token_capture(monkeypatch):
    builder = AscendIndexerKPoolTailMetadataBuilder.__new__(AscendIndexerKPoolTailMetadataBuilder)
    builder.vllm_config = _config()
    builder.device = torch.device("cpu")
    builder._tail_inputs = {}
    builder._max_num_metadata_reqs = 4
    builder._pool_size, builder.block_size, builder._page_rows = 4, 9, 16
    builder._lookahead = True
    common = SimpleNamespace(
        num_reqs=4,
        num_input_tokens=8,
        max_query_len=1,
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
        block_table_tensor=torch.ones(4, 1, dtype=torch.int32),
        pcp_context=None,
    )
    adapter = importlib.import_module("vllm_ascend.ops.triton.glm5_next_cann_keypool")
    build = Mock(return_value=object())
    monkeypatch.setattr(adapter, "build_tail_inputs", build)
    builder.build(0, common)
    assert build.call_args.args[4] == 6
    original = builder._tail_inputs[common.slot_mapping.data_ptr()]
    common.max_query_len = 6
    builder.build(0, common)
    assert build.call_args.args[4] == 6
    assert build.call_args.args[5] is original
    common.max_query_len = 513
    builder.build(0, common)
    assert build.call_args.args[4] == builder.block_size


def test_adaptive_conv_preserves_accepted_history_and_device_queries(monkeypatch):
    x = torch.zeros(8, 384, dtype=torch.bfloat16)
    weight = torch.ones(4, 384, dtype=torch.bfloat16)
    state = torch.zeros(16, 8, 384, dtype=torch.bfloat16)
    starts = torch.tensor([0, 1, 7, 7, 7], dtype=torch.int32)
    indices = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    accepted = torch.tensor([6, 2, 1, 1], dtype=torch.int32)
    out = torch.empty_like(x)
    call = Mock(return_value=out)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update_varlen", call)
    monkeypatch.setattr(conv_ops, "causal_conv1d_update", Mock(side_effect=AssertionError("native padded path")))
    result = conv_ops.causal_conv1d(
        x,
        weight,
        state,
        starts,
        indices,
        run_mode=1,
        num_accepted_tokens=accepted,
        max_query_len=6,
        output_buffer=out,
    )
    assert result is out
    assert call.call_args.args == (x, state, weight)
    kwargs = call.call_args.kwargs
    assert kwargs["query_start_loc"] is starts and kwargs["num_accepted_tokens"] is accepted
    assert kwargs["max_query_len"] == 6 and kwargs["out"].data_ptr() == out.data_ptr()


def test_adaptive_recurrent_dispatch_keeps_wide_state_rows(monkeypatch):
    x = torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16)
    state = torch.zeros(32, 1, 128, 128)
    starts = torch.tensor([0, 1, 7, 7, 7], dtype=torch.int32)
    indices = torch.arange(1, 25, dtype=torch.int32).view(4, 6)
    accepted = torch.tensor([6, 2, 1, 1], dtype=torch.int32)
    call = Mock(return_value=x)
    monkeypatch.setattr(model_kda_ops, "fused_recurrent_kda_varlen", call)
    monkeypatch.setattr(
        model_kda_ops, "run_recurrent_kda", Mock(side_effect=AssertionError("native fixed-length path"))
    )
    result = model_kda_ops.recurrent_kda(
        x,
        x,
        x,
        x,
        torch.zeros(1, 8, 1),
        state,
        starts,
        indices,
        torch.zeros(1),
        torch.zeros(128),
        -4.0,
        accepted,
        use_triton=True,
    )
    assert result is x
    args = call.call_args.args
    assert args[6] is starts and args[7].shape == (4, 6)
    torch.testing.assert_close(args[7][0, args[8][0] - 1], indices[0, 5])


def test_fused_launch_preserves_projection_strides(monkeypatch):
    # Production merged QKV is [T,3*H*D], with head-contiguous views.
    merged = torch.zeros(1, 8, 3 * 2 * 128, dtype=torch.bfloat16)
    q, k, v = (part.view(1, 8, 2, 128) for part in merged.split(2 * 128, dim=-1))
    raw_beta = torch.zeros(1, 8, 10)[..., :2]
    state = torch.zeros(32, 2, 128, 128)
    starts = torch.tensor([0, 1, 7, 7, 7], dtype=torch.int32)
    indices = torch.arange(1, 25, dtype=torch.int32).view(4, 6)
    accepted = torch.tensor([6, 2, 1, 1], dtype=torch.int32)
    launch = Mock()
    kernel = Mock()
    kernel.__getitem__ = Mock(return_value=launch)
    monkeypatch.setattr(recurrent_ops, "fused_recurrent_gated_delta_rule_fwd_kernel", kernel)
    out = recurrent_ops.fused_recurrent_kda_varlen(
        q,
        k,
        v,
        torch.zeros_like(q),
        raw_beta,
        state,
        starts,
        indices,
        accepted,
        torch.zeros(2),
        torch.zeros(256),
        -4.0,
    )
    assert out.shape == q.shape and out.is_contiguous()
    assert all(actual is expected for actual, expected in zip(launch.call_args.args[:3], (q, k, v)))
    assert launch.call_args.kwargs["Q_TOKEN_STRIDE"] == 768
    assert launch.call_args.kwargs["BETA_TOKEN_STRIDE"] == 10
    assert launch.call_args.kwargs["COMPUTE_GATE"] and launch.call_args.kwargs["SIGMOID_BETA"]
