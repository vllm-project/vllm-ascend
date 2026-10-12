# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.attention.sfa_kv_offload import _validate_fused_copy_sfa_config


def make_config(v2, mtp):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        use_v2_model_runner=v2,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=mtp) if mtp else None,
    )


def make_fused_config(vllm_config, hot_tokens):
    config = SparseKVOffloadConfig.from_additional_config(
        vllm_config,
        {"enabled": True, "fused_op_type": "fused_copy_sfa", "topk_buffer_size": hot_tokens},
    )
    # Kernel/layout constraints are checked during backend setup, not config import.
    _validate_fused_copy_sfa_config(vllm_config, config)
    return config


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("mtp", [0, 1, 2, 3])
def test_both_runners_accept_sparse_fused_mtp(v2, mtp):
    config = make_fused_config(make_config(v2, mtp), 8192)
    assert config.enabled
    assert config.use_fused_copy_sfa


@pytest.mark.parametrize("v2", [False, True])
def test_decode_pp_remains_rejected(v2):
    config = make_config(v2, 2)
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError):
        SparseKVOffloadConfig.from_additional_config(config, {"enabled": True})


@pytest.mark.parametrize("v2", [False, True])
def test_producer_offload_remains_rejected(v2):
    config = make_config(v2, 2)
    config.kv_transfer_config.is_kv_consumer = False
    with pytest.raises(AssertionError):
        SparseKVOffloadConfig.from_additional_config(config, {"enabled": True})


def make_dspark_config():
    config = make_config(True, 0)
    config.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=8,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5DSparkForCausalLM"],
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_size=8,
                sample_from_anchor=True,
            )
        ),
    )
    return config


@pytest.mark.parametrize("hot_tokens", [18432, 18688, 32512])
def test_v2_glm_mla_dspark_accepts_nine_row_kernel_budget(hot_tokens):
    config = make_fused_config(make_dspark_config(), hot_tokens)
    assert config.use_fused_copy_sfa


@pytest.mark.parametrize("hot_tokens", [8192, 16384, 18433, 32768])
def test_dspark_rejects_undersized_unaligned_or_kernel_overflow_budget(hot_tokens):
    with pytest.raises(ValueError, match="hot budget"):
        make_fused_config(make_dspark_config(), hot_tokens)


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", ["OtherMLADraft"]),
        ("block_size", 7),
        ("sample_from_anchor", False),
        ("kv_lora_rank", 256),
        ("qk_rope_head_dim", 32),
    ],
)
def test_fused_config_does_not_whitelist_draft_checkpoint_metadata(field, value):
    # Model/cache compatibility belongs to the loader/runner, not fused SFA config.
    config = make_dspark_config()
    setattr(config.speculative_config.draft_model_config.hf_config, field, value)
    offload = make_fused_config(config, 18432)
    assert offload.use_fused_copy_sfa


@pytest.mark.parametrize("draft_tokens", [1, 3, 7, 8, 9, 13])
def test_dspark_fused_budget_uses_configured_draft_width(draft_tokens):
    config = make_dspark_config()
    config.speculative_config.num_speculative_tokens = draft_tokens
    # No HF config is needed here; upstream/model validation owns block semantics.
    del config.speculative_config.draft_model_config
    offload = make_fused_config(config, (draft_tokens + 1) * 2048)
    assert offload.use_fused_copy_sfa


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("draft_tokens", [6, 7, 8, 13])
def test_mtp_fused_config_uses_same_kernel_limits(v2, draft_tokens):
    offload = make_fused_config(make_config(v2, draft_tokens), 32512)
    assert offload.use_fused_copy_sfa


@pytest.mark.parametrize("method", ["dspark", "mtp"])
@pytest.mark.parametrize("draft_tokens", [-1, 14])
def test_fused_config_rejects_query_width_outside_kernel_contract(method, draft_tokens):
    config = make_dspark_config()
    config.speculative_config.method = method
    config.speculative_config.num_speculative_tokens = draft_tokens
    with pytest.raises(ValueError, match="query rows"):
        make_fused_config(config, 32512)


def test_dspark_rejects_hot_budget_below_configured_width():
    config = make_dspark_config()
    config.speculative_config.num_speculative_tokens = 13
    with pytest.raises(ValueError, match="hot budget"):
        make_fused_config(config, 26624)


@pytest.mark.parametrize("fused_op_type", ["none", "fused_copy_sfa"])
def test_remote_dspark_requires_v2_context_initialization(fused_op_type):
    config = make_dspark_config()
    config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="V2 remote prompt-context initialization"):
        SparseKVOffloadConfig.from_additional_config(
            config, {"enabled": True, "fused_op_type": fused_op_type, "topk_buffer_size": 18432}
        )
