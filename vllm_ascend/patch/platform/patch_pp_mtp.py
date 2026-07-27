#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Backport vLLM PP + MTP runtime support.

The local Eagle/MTP drafter returns the draft tokens that belong to the model
output being processed. With PP batch_queue, EngineCore schedules a newer batch
before consuming the older output, so updating ``request.spec_token_ids`` from
``post_step`` observes live Request state from the newer schedule step.
"""

from __future__ import annotations

import copy
from functools import wraps
from itertools import chain

from vllm.logger import logger

_PATCHED = False
_PP_IN_FLIGHT_STEP = 1 << 60


def _is_pd_prefill_node(vllm_config) -> bool:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return False

    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_role == "kv_producer":
        return True

    is_kv_producer = getattr(kv_transfer_config, "is_kv_producer", False)
    is_kv_consumer = getattr(kv_transfer_config, "is_kv_consumer", False)
    return is_kv_producer and not is_kv_consumer


def _use_pp_mtp_runtime_patch(vllm_config, use_pp: bool) -> bool:
    if not _use_pp_ipc_runtime_patch(vllm_config, use_pp):
        return False
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return speculative_config is not None


def _use_pp_ipc_runtime_patch(vllm_config, use_pp: bool) -> bool:
    if not use_pp or _is_pd_prefill_node(vllm_config):
        return False
    return not getattr(vllm_config, "use_v2_model_runner", False)


def _patch_model_runner_output() -> None:
    from vllm.v1 import outputs as outputs_mod

    model_runner_output_cls = outputs_mod.ModelRunnerOutput
    fields = getattr(model_runner_output_cls, "__dataclass_fields__", {})
    if "spec_token_ids" not in fields:
        model_runner_output_cls.spec_token_ids = None
        original_init = model_runner_output_cls.__init__
        if getattr(original_init, "_vllm_ascend_pp_mtp_patched", False):
            return

        @wraps(original_init)
        def _patched_init(self, *args, spec_token_ids=None, **kwargs):
            original_init(self, *args, **kwargs)
            self.spec_token_ids = spec_token_ids

        _patched_init._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
        model_runner_output_cls.__init__ = _patched_init

    empty_output = outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
    if not hasattr(empty_output, "spec_token_ids"):
        empty_output.spec_token_ids = None


def _patch_engine_core() -> None:
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.post_step, "_vllm_ascend_pp_mtp_patched", False):
        return

    original_post_step = EngineCore.post_step

    @wraps(original_post_step)
    def _patched_post_step(self, model_executed: bool) -> None:
        scheduler = getattr(self, "scheduler", None)
        use_pp_mtp_runtime_patch = _use_pp_mtp_runtime_patch(
            getattr(scheduler, "vllm_config", None),
            getattr(scheduler, "use_pp", False),
        )
        if (
            use_pp_mtp_runtime_patch
            and getattr(self, "batch_queue", None) is not None
            and not getattr(self, "async_scheduling", False)
            and getattr(self, "use_spec_decode", False)
            and model_executed
        ):
            return
        return original_post_step(self, model_executed)

    _patched_post_step._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    EngineCore.post_step = _patched_post_step


def _patch_scheduler_update_after_schedule() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(
        Scheduler._update_after_schedule,
        "_vllm_ascend_pp_mtp_inflight_patched",
        False,
    ):
        return

    original_update_after_schedule = Scheduler._update_after_schedule

    @wraps(original_update_after_schedule)
    def _patched_update_after_schedule(self, scheduler_output):
        original_update_after_schedule(self, scheduler_output)
        if not _use_pp_ipc_runtime_patch(
            getattr(self, "vllm_config", None),
            getattr(self, "use_pp", False),
        ):
            return

        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            # Intermediate prefill chunks do not depend on sampled/spec token
            # writeback, so keep them schedulable to fill the PP pipeline.
            # Fence only chunks that can produce autoregressive output: the
            # final prefill chunk (after it flips is_prefill_chunk to False)
            # and decode chunks.
            if request is not None and not request.is_prefill_chunk:
                request.next_decode_eligible_step = _PP_IN_FLIGHT_STEP

    _patched_update_after_schedule._vllm_ascend_pp_mtp_inflight_patched = True  # type: ignore[attr-defined]
    Scheduler._update_after_schedule = _patched_update_after_schedule


def _patch_scheduler_make_cached_request_data() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(
        Scheduler._make_cached_request_data,
        "_vllm_ascend_pp_mtp_cached_data_patched",
        False,
    ):
        return

    original_make_cached = Scheduler._make_cached_request_data

    @wraps(original_make_cached)
    def _patched_make_cached_request_data(
        self,
        running_reqs,
        resumed_reqs,
        num_scheduled_tokens,
        spec_decode_tokens,
        req_to_new_blocks,
    ):
        saved_async = self.scheduler_config.async_scheduling
        use_pp_ipc_runtime_patch = _use_pp_ipc_runtime_patch(
            getattr(self, "vllm_config", None),
            getattr(self, "use_pp", False),
        )
        try:
            if use_pp_ipc_runtime_patch:
                self.scheduler_config.async_scheduling = False
            cached_reqs_data = original_make_cached(
                self,
                running_reqs,
                resumed_reqs,
                num_scheduled_tokens,
                spec_decode_tokens,
                req_to_new_blocks,
            )
        finally:
            self.scheduler_config.async_scheduling = saved_async

        if not saved_async or not use_pp_ipc_runtime_patch or not cached_reqs_data.new_token_ids:
            return cached_reqs_data

        for req_index, req in enumerate(chain(running_reqs, resumed_reqs)):
            if req_index >= len(cached_reqs_data.new_token_ids):
                break
            if cached_reqs_data.new_token_ids[req_index]:
                continue
            if req.num_output_tokens <= 0 or not req.all_token_ids:
                continue
            cached_reqs_data.new_token_ids[req_index] = [req.all_token_ids[-1]]
        return cached_reqs_data

    _patched_make_cached_request_data._vllm_ascend_pp_mtp_cached_data_patched = True  # type: ignore[attr-defined]
    Scheduler._make_cached_request_data = _patched_make_cached_request_data


def _update_pp_mtp_spec_token_ids(scheduler, scheduler_output, model_runner_output) -> None:
    spec_token_ids = getattr(model_runner_output, "spec_token_ids", None)
    if spec_token_ids is None:
        return

    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    for req_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(req_id)
        if request is None or request.is_finished():
            continue

        req_index = model_runner_output.req_id_to_index.get(req_id)
        if req_index is None:
            continue

        new_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
        if not new_token_ids or req_index >= len(spec_token_ids):
            request.spec_token_ids = []
            continue

        next_spec_token_ids = spec_token_ids[req_index]
        if scheduler.structured_output_manager.should_advance(request):
            metadata = request.structured_output_request
            assert metadata is not None and metadata.grammar is not None
            next_spec_token_ids = metadata.grammar.validate_tokens(next_spec_token_ids)
        request.spec_token_ids = next_spec_token_ids


def _patch_scheduler_update_from_output() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.update_from_output, "_vllm_ascend_pp_mtp_patched", False):
        return

    original_update_from_output = Scheduler.update_from_output

    @wraps(original_update_from_output)
    def _patched_update_from_output(self, scheduler_output, model_runner_output):
        use_pp_ipc_runtime_patch = _use_pp_ipc_runtime_patch(
            getattr(self, "vllm_config", None),
            getattr(self, "use_pp", False),
        )
        use_pp_mtp_runtime_patch = (
            use_pp_ipc_runtime_patch
            and getattr(getattr(self, "vllm_config", None), "speculative_config", None) is not None
        )
        if use_pp_mtp_runtime_patch and any(
            num_tokens <= 0 for num_tokens in scheduler_output.num_scheduled_tokens.values()
        ):
            scheduler_output = copy.copy(scheduler_output)
            scheduler_output.num_scheduled_tokens = {
                req_id: num_tokens
                for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items()
                if num_tokens > 0
            }
            scheduler_output.total_num_scheduled_tokens = sum(scheduler_output.num_scheduled_tokens.values())
            scheduler_output.scheduled_spec_decode_tokens = {
                req_id: token_ids
                for req_id, token_ids in scheduler_output.scheduled_spec_decode_tokens.items()
                if req_id in scheduler_output.num_scheduled_tokens
            }

        engine_core_outputs = original_update_from_output(
            self,
            scheduler_output,
            model_runner_output,
        )

        if use_pp_ipc_runtime_patch:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request is not None:
                    request.next_decode_eligible_step = 0

        if not use_pp_mtp_runtime_patch:
            return engine_core_outputs

        _update_pp_mtp_spec_token_ids(self, scheduler_output, model_runner_output)
        return engine_core_outputs

    _patched_update_from_output._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    Scheduler.update_from_output = _patched_update_from_output


def _patch_model_config_validation() -> None:
    from typing import get_args

    from vllm.config.model import ModelConfig
    from vllm.config.speculative import MTPModelTypes

    original_verify = ModelConfig.verify_with_parallel_config
    if getattr(original_verify, "_vllm_ascend_pp_mtp_patched", False):
        return

    mtp_model_types = set(get_args(MTPModelTypes))

    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        hf_config = getattr(self, "hf_config", None)
        model_type = getattr(hf_config, "model_type", None)
        is_eagle_drafter = (model_type == "eagle" or model_type == "speculators") and any(
            arch.startswith("Eagle") or arch.endswith("Eagle3") for arch in getattr(self, "architectures", ())
        )
        is_mtp_drafter = model_type in mtp_model_types
        if (
            getattr(self, "runner", None) == "draft"
            and (is_eagle_drafter or is_mtp_drafter)
            and getattr(parallel_config, "pipeline_parallel_size", 1) > 1
        ):
            # Local Eagle/MTP drafters are loaded on the last PP stage rather
            # than partitioned across all PP stages. Keep normal target-model
            # validation intact, but validate these draft models as PP=1.
            logger.warning(
                "Validating local Eagle/MTP drafter with pipeline_parallel_size=1 "
                "because it is loaded locally on the last pipeline stage."
            )
            patched_config = copy.copy(parallel_config)
            patched_config.pipeline_parallel_size = 1
            return original_verify(self, patched_config)
        return original_verify(self, parallel_config)

    _patched_verify_with_parallel_config._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    ModelConfig.verify_with_parallel_config = _patched_verify_with_parallel_config


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_model_runner_output()
    _patch_engine_core()
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_make_cached_request_data()
    _patch_scheduler_update_from_output()
    _patch_model_config_validation()


_apply_patch()
