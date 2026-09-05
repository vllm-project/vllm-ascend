# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Registration and validation of the UNO speculative-decoding method.

``SpeculativeConfig`` is a pydantic dataclass, so ``method="uno"`` is rejected
during *field validation* -- before ``__post_init__`` runs. Admitting it means
rebuilding the field schema, which is the kind of thing that keeps working right
up until a pydantic upgrade; these tests are the tripwire.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config.speculative import SpeculativeConfig

# The conftest applies the platform patches (adapt_patch(True)), so importing
# this module is enough for the registration to be in place.
from vllm_ascend.models.sdar import SDARConfig
from vllm_ascend.patch.platform.patch_uno_speculative_config import UNO_METHOD
from vllm_ascend.platform import (
    UNO_LORA_RANK,
    UNO_LORA_TARGET_MODULES,
    _validate_and_update_uno_config,
)


def _make_speculative_config(**overrides):
    return SpeculativeConfig(
        method=UNO_METHOD,
        model="/path/to/uno/adapter",
        num_speculative_tokens=8,
        **overrides,
    )


def test_uno_method_passes_field_validation():
    """The schema must accept "uno" and then fail on UNO's own check.

    Reaching the ``target_model_config`` error proves the pydantic Literal was
    extended; a ``literal_error`` would mean the rebuild silently did not take.
    """
    with pytest.raises(Exception) as excinfo:
        _make_speculative_config()
    message = str(excinfo.value)
    assert "target_model_config must be present for uno" in message, message
    assert "literal_error" not in message, message


def test_unknown_methods_are_still_rejected():
    with pytest.raises(Exception) as excinfo:
        SpeculativeConfig(method="definitely-not-a-method", num_speculative_tokens=8)
    assert "definitely-not-a-method" in str(excinfo.value)


def test_uno_requires_the_adapter_path():
    with pytest.raises(Exception, match="requires the draft LoRA path"):
        SpeculativeConfig(method=UNO_METHOD, num_speculative_tokens=8)


_ASCEND_CONFIG = SimpleNamespace(enable_reduce_sample=False)


def _vllm_config(lora_config=None, **parallel):
    parallel_defaults = {
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "data_parallel_size": 1,
    }
    parallel_defaults.update(parallel)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=UNO_METHOD,
            num_speculative_tokens=8,
            num_speculative_tokens_per_batch_size=None,
            disable_padded_drafter_batch=False,
        ),
        parallel_config=SimpleNamespace(**parallel_defaults),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=8192),
        lora_config=lora_config,
        model_config=SimpleNamespace(dtype="bfloat16"),
    )


def test_lora_config_is_synthesised_for_uno():
    config = _vllm_config()
    _validate_and_update_uno_config(config, _ASCEND_CONFIG)

    assert config.lora_config is not None
    assert config.lora_config.max_lora_rank == UNO_LORA_RANK
    assert config.lora_config.max_loras == 1
    # Leaving target_modules unset would wrap lm_head, which then runs a
    # rank-128 LoRA op over the whole vocabulary on every forward for a delta
    # that is identically zero.
    assert set(config.lora_config.target_modules) == set(UNO_LORA_TARGET_MODULES)


def test_the_hook_is_idempotent():
    """`check_and_update_config` runs once in the front end and again inside the
    EngineCore process (`VllmConfig.__post_init__`). The second pass sees the
    config the first pass synthesised; rejecting it there killed every UNO run
    at engine start on hardware.
    """
    config = _vllm_config()
    _validate_and_update_uno_config(config, _ASCEND_CONFIG)
    synthesised = config.lora_config
    assert synthesised is not None

    # Second pass over the same VllmConfig: must be a no-op, not a rejection.
    _validate_and_update_uno_config(config, _ASCEND_CONFIG)
    assert config.lora_config is synthesised

    # ... and a third, because the config is pickled between processes and the
    # recognition has to be by value, not by object identity.
    from vllm.config.lora import LoRAConfig

    config.lora_config = LoRAConfig(
        max_lora_rank=UNO_LORA_RANK,
        max_loras=1,
        target_modules=list(UNO_LORA_TARGET_MODULES),
    )
    _validate_and_update_uno_config(config, _ASCEND_CONFIG)


def test_user_supplied_lora_config_is_rejected():
    """--enable-lora means request-selectable adapters, which UNO cannot share.

    A request adapter would also be applied to the verify forward, which has to
    run on unadapted base weights, and at max_loras=1 the two adapters would
    swap in and out of the single slot every step.
    """
    from vllm.config.lora import LoRAConfig

    # Deliberately not UNO's own shape: `--enable-lora` with several slots is
    # what a user asking for request-selectable adapters actually produces.
    config = _vllm_config(lora_config=LoRAConfig(max_lora_rank=16, max_loras=4))
    with pytest.raises(NotImplementedError, match="request-selectable"):
        _validate_and_update_uno_config(config, _ASCEND_CONFIG)


@pytest.mark.parametrize(
    "parallel_override",
    [
        {"pipeline_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"data_parallel_size": 2},
    ],
)
def test_unsupported_parallelism_is_rejected_at_startup(parallel_override):
    config = _vllm_config(**parallel_override)
    with pytest.raises(NotImplementedError):
        _validate_and_update_uno_config(config, _ASCEND_CONFIG)


def test_dynamic_speculative_lengths_are_rejected_at_startup():
    config = _vllm_config()
    config.speculative_config.num_speculative_tokens_per_batch_size = [[1, 32, 4]]
    with pytest.raises(NotImplementedError, match="dynamic speculative"):
        _validate_and_update_uno_config(config, _ASCEND_CONFIG)


def test_unpadded_drafter_batch_is_rejected_at_startup():
    config = _vllm_config()
    config.speculative_config.disable_padded_drafter_batch = True
    with pytest.raises(NotImplementedError, match="padded drafter batch"):
        _validate_and_update_uno_config(config, _ASCEND_CONFIG)


def test_reduce_sample_mode_is_rejected_at_startup():
    # In reduce-sample mode the shared `apply_sampling_constraints` returns a
    # compact top-k support, and UNO's draft probabilities must be full-vocab
    # for the rejection kernel's stride arithmetic to hold.
    config = _vllm_config()
    with pytest.raises(NotImplementedError, match="enable_reduce_sample"):
        _validate_and_update_uno_config(config, SimpleNamespace(enable_reduce_sample=True))


def test_non_uno_configs_are_left_alone():
    config = _vllm_config()
    config.speculative_config.method = "eagle"
    config.parallel_config.pipeline_parallel_size = 4
    _validate_and_update_uno_config(config, _ASCEND_CONFIG)
    assert config.lora_config is None


def test_sdar_config_presents_itself_as_qwen3():
    config = SDARConfig(
        architectures=["SDARForCausalLM"],
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        block_size=8,
        mask_token_id=151669,
    )
    assert config.model_type == "sdar"
    assert config.architectures == ["Qwen3ForCausalLM"]
    # Extra keys the bundle carries must survive: block_size is the diffusion
    # block the adapter was trained on, and the proposer warns against a wider
    # forward width.
    assert config.block_size == 8
    assert config.mask_token_id == 151669


def test_sdar_config_does_not_rewrite_a_foreign_architecture():
    config = SDARConfig(architectures=["SomethingElseForCausalLM"])
    assert config.architectures == ["SomethingElseForCausalLM"]


def test_sdar_registration_does_not_leak_into_qwen3():
    from transformers import Qwen3Config

    # Registering a bare Qwen3Config under "sdar" would set
    # Qwen3Config.model_type process-wide and break every real Qwen3 checkpoint
    # served by the same engine.
    assert Qwen3Config.model_type == "qwen3"
    assert issubclass(SDARConfig, Qwen3Config)


def test_the_two_method_constants_agree():
    """`uno_proposer` duplicates the method string to avoid a patch-order cycle."""
    from vllm_ascend.spec_decode.uno_proposer import UNO_METHOD as PROPOSER_UNO_METHOD

    assert PROPOSER_UNO_METHOD == UNO_METHOD


def test_only_the_base_cudagraph_case_is_captured_for_uno():
    """UNO's adapter is engine-internal, so no captured graph may contain it.

    Whether the LoRA ops end up in a graph is decided by Python branches at
    capture time, so a graph captured with the adapter live would keep applying
    it on replay -- including on the verify forward, which must run on base
    weights.
    """
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    dispatcher = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method=UNO_METHOD),
            lora_config=SimpleNamespace(max_loras=1),
        ),
        compilation_config=SimpleNamespace(cudagraph_specialize_lora=False),
        specialize_lora_count=False,
    )

    assert CudagraphDispatcher._get_lora_cases(dispatcher) == [0]

    # Anything else keeps upstream's behaviour: without specialization vLLM
    # captures only the LoRA-active case.
    dispatcher.vllm_config.speculative_config = SimpleNamespace(method="eagle")
    assert CudagraphDispatcher._get_lora_cases(dispatcher) == [2]


def test_a_failed_vllm_config_rebuild_leaves_the_schema_usable():
    """`rebuild_dataclass` deletes the cached validator before regenerating it.

    Swallowing an exception without restoring it would leave every
    ``VllmConfig(...)`` in the process raising.
    """
    from vllm.config import VllmConfig

    from vllm_ascend.patch.platform import patch_uno_speculative_config as mod

    before = {attr: VllmConfig.__dict__.get(attr) for attr in mod._PYDANTIC_SCHEMA_ATTRS}

    def _boom(*args, **kwargs):
        for attr in ("__pydantic_core_schema__", "__pydantic_validator__"):
            if attr in VllmConfig.__dict__:
                delattr(VllmConfig, attr)
        raise RuntimeError("simulated pydantic failure")

    with patch.object(mod, "rebuild_dataclass", _boom):
        mod._rebuild_vllm_config_schema()

    after = {attr: VllmConfig.__dict__.get(attr) for attr in mod._PYDANTIC_SCHEMA_ATTRS}
    assert after == before
    # And it still builds.
    assert VllmConfig() is not None


def test_a_draft_batch_wider_than_the_lora_index_buffers_is_rejected():
    """The punica index buffers are sized from max_num_batched_tokens.

    A draft forward of `max_num_seqs * F` rows wider than that fails inside
    `_update_base_metadata`'s `copy_` at the first decode step instead.
    """
    config = _vllm_config()
    config.scheduler_config.max_num_seqs = 256
    config.scheduler_config.max_num_batched_tokens = 1024  # 256 * 8 = 2048 rows
    with pytest.raises(ValueError, match="max-num-batched-tokens"):
        _validate_and_update_uno_config(config, _ASCEND_CONFIG)

    config.scheduler_config.max_num_batched_tokens = 2048
    _validate_and_update_uno_config(config, _ASCEND_CONFIG)
    assert config.lora_config is not None


def test_runtime_lora_requests_are_refused_while_uno_owns_the_slot():
    """Rejecting `--enable-lora` at config time does not cover the runtime API.

    `add_lora` is reachable through `collective_rpc("add_lora")` and
    `LLM.generate(lora_request=...)`. An adapter added that way would be applied
    to the verify forward, evict UNO's own adapter at `max_loras=1`, and -- since
    UNO's graphs are captured with LoRA switched off -- would silently not be
    applied at all on a captured decode step.
    """
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = SimpleNamespace(speculative_config=SimpleNamespace(method=UNO_METHOD))
    with pytest.raises(NotImplementedError, match="request-selectable"):
        NPUModelRunner.add_lora(runner, SimpleNamespace(lora_int_id=2))


def test_the_worker_reload_hook_targets_only_the_uno_drafter():
    from vllm_ascend.spec_decode.uno_proposer import AscendUnoProposer
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    # No drafter at all (no speculative decoding): must not raise.
    NPUModelRunner.reload_uno_draft_adapter(SimpleNamespace())
    # A different proposer: must not touch it.
    NPUModelRunner.reload_uno_draft_adapter(SimpleNamespace(drafter=SimpleNamespace()))

    # The UNO proposer: must re-register, because every warmup dummy run ends in
    # `remove_all_adapters()` and no request will ever bring the adapter back.
    drafter = object.__new__(AscendUnoProposer)
    calls = []
    drafter.load_lora_adapter = lambda: calls.append("reloaded")
    NPUModelRunner.reload_uno_draft_adapter(SimpleNamespace(drafter=drafter))
    assert calls == ["reloaded"]


def test_the_parent_runner_dispatch_is_survivable():
    """vLLM's own `GPUModelRunner.__init__` has a second drafter dispatch.

    It runs from `NPUModelRunner.__init__`'s `super().__init__()`, before
    vllm-ascend builds its own drafter, and ends in `raise ValueError("Unknown
    speculative decoding method: ...")`. On hardware that is exactly where UNO
    died. The patch presents `method="medusa"` for the duration of that
    constructor only -- inside it `method` is read nowhere else -- and restores
    it on the shared config object afterwards.
    """
    from vllm_ascend.patch.platform import patch_uno_speculative_config as mod

    seen = []
    speculative_config = SimpleNamespace(method=UNO_METHOD)
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    def _fake_parent_init(self, cfg, *args, **kwargs):
        # What the parent's dispatch chain would see.
        seen.append(cfg.speculative_config.method)

    with patch.object(mod, "_orig_gpu_runner_init", _fake_parent_init):
        mod._uno_gpu_runner_init(object(), vllm_config)

    assert seen == ["medusa"], seen
    # ... and the method the Ascend runner reads right afterwards is UNO's.
    assert speculative_config.method == UNO_METHOD


def test_the_parent_dispatch_patch_leaves_other_methods_alone():
    from vllm_ascend.patch.platform import patch_uno_speculative_config as mod

    seen = []
    vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(method="eagle"))

    def _fake_parent_init(self, cfg, *args, **kwargs):
        seen.append(cfg.speculative_config.method)

    with patch.object(mod, "_orig_gpu_runner_init", _fake_parent_init):
        mod._uno_gpu_runner_init(object(), vllm_config)

    assert seen == ["eagle"], seen


def test_the_parent_dispatch_restores_uno_after_initialization_failure():
    from vllm_ascend.patch.platform import patch_uno_speculative_config as mod

    config = SimpleNamespace(speculative_config=SimpleNamespace(method=UNO_METHOD))
    with (
        patch.object(mod, "_orig_gpu_runner_init", side_effect=RuntimeError("parent failed")),
        pytest.raises(RuntimeError, match="parent failed"),
    ):
        mod._uno_gpu_runner_init(object(), config)
    assert config.speculative_config.method == UNO_METHOD
