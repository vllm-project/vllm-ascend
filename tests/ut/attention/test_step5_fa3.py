# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests for A5 Step5 ecosystem attention."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.attention import test_minimax_fa3

fa3 = test_minimax_fa3.fa3
profiles = test_minimax_fa3.profiles
selection = test_minimax_fa3.selection


@pytest.mark.parametrize("arch,method", [("DSparkMiniMaxDraftModel", "dspark"), ("Step5MTP", "mtp")])
def test_fa3_profile_can_enable_multiple_model_families(selection, monkeypatch, arch, method):
    r = selection
    r.device = r.profiles.AscendDeviceType.A5
    model = r.profiles.FlashAttentionNpu3Model
    # Exercise a future profile without changing the registered A5 support.
    profile = replace(
        r.profiles.get_hardware_profile(r.device),
        flash_attn_npu_3_models=frozenset({model.MINIMAX, model.STEP5}),
    )
    monkeypatch.setattr(r.profiles, "get_hardware_profile", lambda _: profile)
    r.config.model_config = NS(hf_config=NS(architectures=[arch]))
    r.config.speculative_config.method = method
    assert "flash_attention_v3" in r.select()


@pytest.mark.parametrize("device", ["A2", "A3", "A5", "_310P"])
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("arch", ["Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP"])
@pytest.mark.parametrize("spec_state", ["none", "missing_draft", "mtp"])
def test_step5_hardware_and_model_routing(selection, device, installed, arch, spec_state):
    r = selection
    r.device = getattr(r.profiles.AscendDeviceType, device)
    r.installed = installed
    r.config.model_config = NS(hf_config=NS(architectures=[arch]))
    r.config.speculative_config = (
        None if spec_state == "none" else NS(method="mtp", draft_model_config=NS() if spec_state == "mtp" else None)
    )
    r.config.parallel_config.tensor_parallel_size = 4
    r.config.parallel_config.data_parallel_size = 4
    assert ("flash_attention_v3" in r.select()) == (device == "A5" and installed and spec_state == "mtp")


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "kv_cache_dtype",
        "cache_dtype",
        "c8",
        "has_sink",
        "use_pcp",
        "use_dcp",
        "use_rswa",
        "use_batch_invariant",
        "use_mm_prefix",
        "use_per_head_quant_scales",
        "use_adaptive_verification",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "use_compress",
        "use_sparse",
        "use_mla",
    ],
)
def test_step5_unsupported_configurations_fall_back(selection, fault):
    r = selection
    r.device = r.profiles.AscendDeviceType.A5
    r.config.model_config = NS(hf_config=NS(architectures=["Step5ForConditionalGeneration"]))
    r.config.speculative_config = NS(method="mtp", draft_model_config=NS())
    if fault.endswith("parallel_size"):
        setattr(r.config.parallel_config, fault, 2)
    elif fault == "cache_dtype":
        r.config.cache_config.cache_dtype = "fp8"
    elif fault == "c8":
        r.config.quant_config = NS(enable_c8_quant=True)
    else:
        setattr(r.selector, fault, {"dtype": torch.float32, "kv_cache_dtype": "fp8"}.get(fault, True))
    try:
        result = r.select()
    except KeyError:
        assert fault in ("use_sparse", "use_compress")
    else:
        assert "flash_attention_v3" not in result


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
def test_step5_mtp_graph_padding_bounds_scheduler_batch(fa3, num_reqs):
    builder = object.__new__(fa3.AscendFlashAttentionMetadataBuilder)
    builder.capture_sizes, builder.capture_query_len = {16}, 4
    builder.max_num_reqs, builder.block_size = 16, 128
    spec = (16, 1, 128, torch.bfloat16, 0.1, 0.0, None)
    builder.scheduler_specs, builder.scheduler_buffers = {spec}, {}
    common = NS(
        num_reqs=num_reqs,
        num_actual_tokens=16,
        num_input_tokens=16,
        max_query_len=1,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(num_reqs + 1, dtype=torch.int32),
        seq_lens=torch.ones(num_reqs, dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.arange(16),
        causal=True,
    )

    def scheduler(**kwargs):
        assert kwargs["batch_size"] == num_reqs
        assert kwargs["cu_seqlens_q"].numel() == num_reqs + 1
        assert kwargs["cache_seqlens"].numel() == num_reqs
        return torch.zeros(8, dtype=torch.uint8)

    fa3.get_scheduler_metadata.side_effect = scheduler
    builder.build(0, common)
