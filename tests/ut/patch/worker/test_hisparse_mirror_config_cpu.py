# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold policy selection must not turn specialization bounds into startup errors."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def policy_module():
    path = Path(__file__).parents[4] / "vllm_ascend/patch/worker/patch_hisparse_lazy_mtp_mirror.py"
    spec = importlib.util.spec_from_file_location("hisparse_mirror_config_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tokens=3):
    return NS(
        kv_transfer_config=NS(kv_connector="HiSparseConnector", kv_role="kv_both", kv_connector_extra_config={}),
        speculative_config=NS(method="mtp", num_speculative_tokens=tokens),
        model_config=NS(enforce_eager=False),
        compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY")),
        attention_config=NS(hisparse_config=NS(eager_host_mirror=False)),
    )


@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 7])
def test_other_draft_lengths_retain_original_core_without_importing_workers(tokens):
    policy = policy_module()
    current = config(tokens)
    assert policy._validate_consumer(current) is False

    def no_specialized_import(name):
        raise AssertionError(f"Fallback must leave Core/worker globals untouched: {name}")

    policy.import_module = no_specialized_import
    assert policy.install(current) is None


@pytest.mark.parametrize("kind", ["no_speculation", "other_method", "eager", "other_graph", "adaptive", "parallel"])
def test_unqualified_layout_falls_back_without_patch_installation(kind):
    policy = policy_module()
    current = config()
    if kind == "no_speculation":
        current.speculative_config = None
    elif kind == "other_method":
        current.speculative_config.method = "dflash"
    elif kind == "eager":
        current.model_config.enforce_eager = True
    elif kind == "other_graph":
        current.compilation_config.cudagraph_mode.name = "NONE"
    elif kind == "adaptive":
        current.speculative_config.enable_adaptive_verification = True
    else:
        current.speculative_config.parallel_drafting = True
    assert policy._validate_consumer(current) is False
    assert policy.install(current) is None


def test_current_qualified_policy_is_unchanged():
    assert policy_module()._validate_consumer(config()) is True


def test_producer_tail_durability_is_not_waived_by_layout_fallback():
    current = config(tokens=2)
    current.kv_transfer_config = NS(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "HiSparseConnector", "kv_role": "kv_both"},
                {"kv_connector": "HiSparsePDConnector", "kv_role": "kv_producer"},
            ]
        },
    )
    with pytest.raises(RuntimeError, match="producer/outbound-export"):
        policy_module().install(current)
