# SPDX-License-Identifier: Apache-2.0

from inspect import signature

from vllm.config.model import ModelConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput

import vllm_ascend.patch.platform  # noqa: F401
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler


def test_pp_mtp_relies_on_vllm_implementations():
    scheduler_cls = BalanceScheduler.__bases__[0]

    assert "spec_token_ids" not in signature(ModelRunnerOutput.__init__).parameters
    assert not getattr(
        ModelConfig.verify_with_parallel_config,
        "_vllm_ascend_pp_mtp_patched",
        False,
    )
    assert not getattr(
        scheduler_cls._update_after_schedule,
        "_vllm_ascend_pp_mtp_inflight_patched",
        False,
    )
    assert not getattr(
        scheduler_cls._make_cached_request_data,
        "_vllm_ascend_pp_mtp_cached_data_patched",
        False,
    )
    assert not getattr(
        scheduler_cls.update_from_output,
        "_vllm_ascend_pp_mtp_patched",
        False,
    )
    assert SupportsPP in DeepSeekMTP.__mro__
    assert not getattr(
        EngineCore.post_step,
        "_vllm_ascend_pp_mtp_patched",
        False,
    )
