# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from contextlib import nullcontext
from dataclasses import replace as dataclass_replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    replace,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

from vllm_ascend.worker.v2.spec_decode.eagle import (
    speculator as eagle_speculator_module,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
    AscendMTPSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.pcp_utils import build_dspark_draft_config, disable_profiling_chunk_for_draft
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import (
    speculator as speculator_module,
)


@pytest.fixture(autouse=True)
def _stub_ascend_config(monkeypatch):
    # Speculators built through the real __init__ read the global ascend
    # config during the lmhead TP construction-time validation; the stub
    # reads as lmhead-off so validation no-ops.
    from vllm_ascend import ascend_config as _ascend_config_module

    monkeypatch.setattr(
        _ascend_config_module,
        "_ASCEND_CONFIG",
        SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(lmhead_tensor_parallel_size=0),
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            ascend_compilation_config=object(),
            eplb_config=object(),
        ),
    )


def _fake_config_replace(config, **changes):
    values = vars(config).copy()
    values.update(changes)
    return SimpleNamespace(**values)


def _config(additional_config, pp_size=2):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        additional_config=additional_config,
    )


@pytest.fixture
def eplb_worker_config(tmp_path):
    # Exercise real config validation without downloading metadata or weights.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    # Static capabilities avoid registry inspection in an NPU-dependent subprocess.
    model_info = SimpleNamespace(
        architecture="DeepseekV3ForCausalLM",
        is_text_generation_model=True,
        is_pooling_model=False,
        attn_type="decoder",
        default_seq_pooling_type=None,
        default_tok_pooling_type=None,
        score_type=None,
        supports_multimodal=False,
        supports_multimodal_raw_input_only=False,
        requires_raw_input_tokens=False,
        supports_multimodal_encoder_tp_data=False,
        supports_pp=True,
        has_inner_state=False,
        is_attention_free=False,
        is_hybrid=False,
        has_noops=False,
        supports_mamba_prefix_caching=False,
        supports_replayssm=False,
        supports_transcription=False,
        supports_transcription_only=False,
        supported_video_pruning_methods=(),
    )
    # Validate MRV2 PCP/EPLB config without requiring accelerator kernels on CPU.
    with (
        patch(
            "vllm.model_executor.models.ModelRegistry.inspect_model_cls",
            return_value=(model_info, model_info.architecture),
        ),
        patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", True),
        patch("vllm.config.vllm.HAS_TRITON", True),
        patch("vllm.config.parallel.current_platform.is_cuda_alike", return_value=True),
    ):
        # Keep the model dtype float-compatible with the bfloat16 KV cache the
        # draft loader override below sets: vLLM main rejects a mismatched
        # explicit float KV dtype.
        model_config = ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, dtype="bfloat16", enforce_eager=True, max_model_len=128
        )
        parallel_config = ParallelConfig(
            prefill_context_parallel_size=8,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="mp",
        )
        # Mirror the runtime marker added by communicator auto-selection.
        parallel_config.eplb_config.__dict__["_vllm_ascend_eplb_auto_selected"] = True
        yield VllmConfig(
            model_config=model_config,
            parallel_config=parallel_config,
            speculative_config=SpeculativeConfig(
                method="mtp",
                num_speculative_tokens=2,
                target_model_config=model_config,
                target_parallel_config=parallel_config,
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=128, max_num_batched_tokens=128, max_num_seqs=4, is_encoder_decoder=False
            ),
            device_config=DeviceConfig(device="cpu"),
        )


@pytest.mark.parametrize(
    ("tp_size", "pcp_size", "dp_size", "enabled", "redundant", "expected"),
    [
        (1, 8, 1, True, 0, False),
        (1, 8, 1, True, 8, False),
        (2, 8, 1, True, 0, True),
        (1, 8, 2, True, 0, True),
        (2, 1, 1, True, 0, True),
        (1, 8, 1, False, 0, False),
    ],
)
def test_dspark_draft_eplb_policy(eplb_worker_config, tp_size, pcp_size, dp_size, enabled, redundant, expected):
    config = replace(
        eplb_worker_config,
        parallel_config=replace(
            eplb_worker_config.parallel_config,
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=pcp_size,
            data_parallel_size=dp_size,
            enable_eplb=enabled,
            eplb_config=dataclass_replace(
                eplb_worker_config.parallel_config.eplb_config, num_redundant_experts=redundant
            ),
        ),
    )
    draft = build_dspark_draft_config(config)
    # Upstream DFlashSpeculator.__init__ forces PCP=1 on the copied config.
    assert draft.parallel_config.prefill_context_parallel_size == pcp_size
    assert (draft is config) == (pcp_size == 1)
    assert draft.parallel_config.enable_expert_parallel
    assert draft.parallel_config.enable_eplb == expected
    assert draft.parallel_config.eplb_config.num_redundant_experts == (redundant if expected else 0)
    assert config.parallel_config.prefill_context_parallel_size == pcp_size
    assert config.parallel_config.enable_eplb == enabled
    assert config.parallel_config.eplb_config.num_redundant_experts == redundant


def test_dspark_draft_config_leaves_dcp_to_upstream(eplb_worker_config):
    """PCP=2/DCP=2 on TP=1 is valid for the target; the draft copy keeps it.

    Re-validating PCP=1 with DCP=2 on TP=1 would fail, so the Ascend builder
    must not change the PCP/DCP sizes; upstream DFlash derives them.
    """
    config = replace(
        eplb_worker_config,
        parallel_config=replace(
            eplb_worker_config.parallel_config,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=2,
        ),
    )
    draft = build_dspark_draft_config(config)
    assert draft is not config
    assert draft.parallel_config.prefill_context_parallel_size == 2
    assert draft.parallel_config.decode_context_parallel_size == 2
    assert not draft.parallel_config.enable_eplb
    assert config.parallel_config.enable_eplb


def test_mtp_draft_config_keeps_target_pcp_topology(eplb_worker_config):
    """Sharded MTP drafts attend with the target's PCP partition (upstream #56107)."""
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.vllm_config = eplb_worker_config
    speculator.draft_model_config = eplb_worker_config.speculative_config.draft_model_config
    speculator.speculative_config = eplb_worker_config.speculative_config
    draft_config = speculator._create_draft_vllm_config()
    assert draft_config.parallel_config.prefill_context_parallel_size == 8
    assert draft_config.parallel_config.decode_context_parallel_size == 1
    assert draft_config.parallel_config.pipeline_parallel_size == 1
    assert draft_config.parallel_config.enable_eplb


def test_propose_passes_target_dp_sync_unchanged() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    input_batch = object()
    dp_sync = object()
    with (
        patch.object(speculator_module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(speculator_module, "torch_gather_wrapper", return_value=nullcontext()),
        patch.object(MTPSpeculator, "propose") as parent,
    ):
        speculator.propose(input_batch, *[MagicMock() for _ in range(10)], dp_sync)
    assert parent.call_args.args[0] is input_batch
    assert parent.call_args.args[11] is dp_sync
    assert speculator.input_batch is input_batch


def test_capture_follows_upstream_contract() -> None:
    speculator = object.__new__(AscendMTPSpeculator)
    events: list[tuple[str, object]] = []
    speculator.num_speculative_steps = 3
    speculator.max_num_reqs = 4
    speculator.last_token_indices = torch.full((4,), 7)
    speculator.idx_mapping = torch.full((4,), 9, dtype=torch.int32)
    speculator.model = object()
    speculator.model_state = object()
    speculator.target_input_buffers = object()
    speculator.input_buffers = object()
    speculator.block_tables = object()
    speculator.target_attn_groups = [["target"]]
    speculator.attn_groups = [["draft"]]
    speculator.kv_cache_config = object()
    speculator.pcp_manager = object()
    for hook in ("on_prefill_begin", "on_prefill_end", "on_multi_step_decode_begin", "on_multi_step_decode_end"):
        setattr(speculator, hook, lambda n, hook=hook: events.append((hook, n)))

    def manager(name):
        mgr = MagicMock(use_breakable_cg=False)
        mgr.capture.side_effect = lambda fn, *args, **kwargs: events.append((name, (fn, args, kwargs)))
        return mgr

    speculator.prefill_cudagraph_manager = manager("prefill")
    speculator.decode_cudagraph_manager = manager("decode")
    with patch.object(speculator_module, "build_attn_metadata_wrapper", return_value=nullcontext()):
        speculator.capture()

    assert torch.all(speculator.last_token_indices == 0)
    assert torch.all(speculator.idx_mapping == 0)
    names = [name for name, _ in events]
    assert names == [
        "on_prefill_begin",
        "prefill",
        "on_prefill_end",
        "on_multi_step_decode_begin",
        "decode",
        "on_multi_step_decode_end",
    ]
    prefill_fn, prefill_args, prefill_kwargs = events[1][1]
    # Prefill captures the routine the runtime replays, with target metadata.
    assert prefill_fn == speculator._prefill
    assert prefill_args[0] is speculator.model_state
    assert prefill_args[1] is speculator.target_input_buffers
    assert prefill_args[3] is speculator.target_attn_groups
    # Its dummy is staged PCP-locally, like the target capture.
    assert prefill_kwargs["pcp_manager"] is speculator.pcp_manager
    decode_fn, decode_args, decode_kwargs = events[4][1]
    assert "pcp_manager" not in decode_kwargs
    assert decode_fn == speculator._multi_step_decode
    assert decode_args[1] is speculator.input_buffers
    assert decode_args[3] is speculator.attn_groups
    assert all(n == 4 for name, n in events if name.startswith("on_"))


def test_eagle_draft_config_disables_profiling_chunk() -> None:
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": "yes"}}}
    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            prefill_context_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
        ),
        additional_config=additional_config,
    )
    speculator = object.__new__(AscendEagleSpeculator)
    # Delegation to the base's _create_draft_vllm_config reads these too.
    speculator.vllm_config = target_config
    target_config.cache_config = SimpleNamespace()
    target_config.parallel_config.decode_context_parallel_size = 1
    speculator.draft_model_config = SimpleNamespace(is_moe=False)

    with (
        patch.object(eagle_speculator_module, "replace", side_effect=_fake_config_replace),
        patch.object(speculator_module, "replace", side_effect=_fake_config_replace),
    ):
        draft_config = speculator._create_draft_vllm_config()

    assert draft_config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
    assert target_config.additional_config is additional_config
    assert additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] == "yes"
    assert draft_config.parallel_config.pipeline_parallel_size == 1
    assert draft_config.parallel_config.is_moe_model is False
    # Eagle drafts shard prefill with the target, so they keep its PCP topology.
    assert draft_config.parallel_config.prefill_context_parallel_size == 2


@pytest.mark.parametrize(("enabled", "legacy"), [(True, False), ("yes", True)])
def test_disable_profiling_chunk_for_draft_accepts_pydantic_true_values(enabled, legacy):
    profiling_chunk = {"enabled": enabled, "min_chunk": 128}
    if legacy:
        additional_config = {"profiling_chunk_config": profiling_chunk, "enable_cpu_binding": True}
    else:
        additional_config = {
            "scheduler_config": {"profiling_chunk_config": profiling_chunk},
            "enable_cpu_binding": True,
        }
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        draft_additional_config = config.additional_config
        draft_profiling_chunk = (
            draft_additional_config["profiling_chunk_config"]
            if legacy
            else draft_additional_config["scheduler_config"]["profiling_chunk_config"]
        )
        assert draft_profiling_chunk == {"enabled": False, "min_chunk": 128}
        assert draft_additional_config is not additional_config
        assert draft_profiling_chunk is not profiling_chunk

    assert config.additional_config is additional_config
    assert profiling_chunk["enabled"] == enabled


@pytest.mark.parametrize(("pp_size", "enabled"), [(1, True), (2, "off")])
def test_disable_profiling_chunk_for_draft_noop(pp_size, enabled):
    additional_config = {"profiling_chunk_config": {"enabled": enabled}}
    config = _config(additional_config, pp_size=pp_size)

    with disable_profiling_chunk_for_draft(config):
        assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_uses_nested_precedence():
    additional_config = {
        "scheduler_config": {"profiling_chunk_config": {"enabled": False}},
        "profiling_chunk_config": {"enabled": True},
    }
    config = _config(additional_config)

    with disable_profiling_chunk_for_draft(config):
        assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_restores_after_failure():
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": True}}}
    config = _config(additional_config)
    expected_context = pytest.raises(RuntimeError, match="draft failed")

    with expected_context, disable_profiling_chunk_for_draft(config):
        assert config.additional_config is not additional_config
        raise RuntimeError("draft failed")

    assert config.additional_config is additional_config


def test_disable_profiling_chunk_for_draft_rejects_invalid_boolean():
    config = _config({"profiling_chunk_config": {"enabled": "sometimes"}})

    with (
        pytest.raises(ValueError, match="additional_config.profiling_chunk_config.enabled must be a boolean"),
        disable_profiling_chunk_for_draft(config),
    ):
        pass
