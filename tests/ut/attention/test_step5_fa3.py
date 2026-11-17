# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests for A5 Step5 ecosystem attention."""

from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.attention import test_minimax_fa3

profiles = test_minimax_fa3.profiles
selection = test_minimax_fa3.selection


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
