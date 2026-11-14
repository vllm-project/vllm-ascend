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

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.ut.test_ascend_config import VllmConfig
from vllm_ascend import ascend_config as ac


def make_config(pcp=16, mode=2):
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        config = VllmConfig()
    config.model_config.is_moe = True
    config.model_config.architectures = ["GlmMoeDsaForCausalLM"]
    config.model_config.hf_text_config = SimpleNamespace(
        hidden_size=6144, moe_intermediate_size=2048, num_experts_per_tok=8
    )
    config.model_config.get_num_experts = lambda: 256
    config.parallel_config.world_size = pcp * 2
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.enable_expert_parallel = True
    config.additional_config = {"enable_fused_mc2": mode}
    return config


@pytest.fixture(autouse=True)
def isolated_runtime():
    ac.clear_ascend_config()
    with patch.object(ac, "_MEGA_MOE_SUPPORTED", True):
        yield
    ac.clear_ascend_config()


def test_draft_validation_does_not_publish_or_disable_target():
    target = make_config()
    active = ac.init_ascend_config(target)
    assert active.megamoe_enabled
    draft = copy.copy(target)
    draft.parallel_config = copy.copy(target.parallel_config)
    draft.parallel_config.prefill_context_parallel_size = 1
    draft.parallel_config.world_size = 2
    validated = ac.build_ascend_config(draft)
    assert not validated.megamoe_enabled
    assert ac._ASCEND_CONFIG is active
    assert ac._INIT_VLLM_CONFIG is target
    assert active.megamoe_enabled
    assert ac.is_mega_moe_supported()


def test_context_switch_and_exception_restore_target():
    target = make_config()
    active = ac.init_ascend_config(target)
    draft = copy.copy(target)
    draft.parallel_config = copy.copy(target.parallel_config)
    draft.parallel_config.world_size = 2
    draft.parallel_config.prefill_context_parallel_size = 1
    with patch("vllm.config.get_current_vllm_config_or_none", return_value=target):
        assert ac.get_ascend_config() is active
        with (
            pytest.raises(RuntimeError, match="draft failed"),
            patch("vllm.config.get_current_vllm_config_or_none", return_value=draft),
        ):
            assert not ac.get_ascend_config().megamoe_enabled
            raise RuntimeError("draft failed")
        assert ac.get_ascend_config() is active
        assert ac.get_ascend_config().megamoe_enabled


def test_plain_mc2_validation_keeps_package_capability():
    target = make_config()
    active = ac.init_ascend_config(target)
    plain = ac.build_ascend_config(make_config(mode=1))
    assert not plain.megamoe_enabled
    assert ac.is_mega_moe_supported()
    assert active.megamoe_enabled


def test_failed_build_keeps_previous_runtime():
    target = make_config()
    active = ac.init_ascend_config(target)
    invalid = make_config()
    invalid.additional_config = {"enable_fused_mc2": 99}
    with pytest.raises(ValueError):
        ac.build_ascend_config(invalid)
    assert ac._ASCEND_CONFIG is active
    assert active.megamoe_enabled


def test_runtime_binding_preserves_vllm_replace():
    from vllm.config import replace

    target = make_config()
    ac.init_ascend_config(target)
    # Isolate the constructor from this metadata-only fixture's stub model.
    # Exercise replace's actual field extraction used by load_dflash_model.
    with patch.object(type(target), "__init__", return_value=None):
        replaced = replace(target, attention_config=copy.copy(target.attention_config))
    assert isinstance(replaced, type(target))
    assert "_ascend_runtime_config" not in target.__dict__
