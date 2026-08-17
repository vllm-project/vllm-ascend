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

    def _patch_optional_field(cls, field_name: str) -> None:
        """Backport an optional dataclass field without touching vLLM files.

        The Ascend worker can run against a vLLM checkout that predates the
        speculative-decoding side-channel fields.  Adding the field at runtime
        keeps the editable vllm-ascend package self-contained and preserves the
        upstream class layout/serialization contract.
        """

        fields = getattr(cls, "__dataclass_fields__", {})
        if field_name in fields:
            return

        setattr(cls, field_name, None)
        original_init = cls.__init__
        marker = f"_vllm_ascend_optional_{field_name}_patched"
        if getattr(original_init, marker, False):
            return

        @wraps(original_init)
        def _patched_init(self, *args, **kwargs):
            value = kwargs.pop(field_name, None)
            original_init(self, *args, **kwargs)
            setattr(self, field_name, value)

        setattr(_patched_init, marker, True)
        cls.__init__ = _patched_init

    # ``spec_token_ids`` is required by the PP/MTP compatibility path, while
    # ``proposal_lengths`` carries the logical dynamic-draft width.  Both are
    # optional on newer upstream vLLM and are installed here only when absent.
    _patch_optional_field(outputs_mod.ModelRunnerOutput, "spec_token_ids")
    _patch_optional_field(outputs_mod.ModelRunnerOutput, "proposal_lengths")
    _patch_optional_field(outputs_mod.DraftTokenIds, "proposal_lengths")

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


def _patch_scheduler_dynamic_gate_compat() -> None:
    """Backport the Ascend proposal gate to older vLLM schedulers.

    The hardware-aware scheduler lives in vllm-ascend's copied scheduler
    classes (BalanceScheduler/RecomputeScheduler).  They call the helper that
    was added to upstream vLLM in 6ec76df8.  Install the same helper and the
    minimal constructor state on the upstream base class when that commit is
    not present, keeping all compatibility code in this repository.
    """

    from vllm.v1.core.sched.scheduler import Scheduler

    if not hasattr(Scheduler, "_apply_ascend_proposal_gate"):

        def _apply_ascend_proposal_gate(
            self,
            configured_k: int,
            *,
            total_num_scheduled_tokens: int,
            num_scheduled_requests: int,
            prefill_scheduled: bool = False,
        ) -> int:
            gate = getattr(self, "_ascend_proposal_gate", None)
            if gate is None:
                return configured_k
            streaming_waiting = getattr(self, "num_waiting_for_streaming_input", 0)
            if not isinstance(streaming_waiting, int):
                streaming_waiting = len(streaming_waiting)
            return gate.select_k(
                configured_k,
                num_running=len(getattr(self, "running", ()))
                + streaming_waiting,
                num_waiting=len(getattr(self, "waiting", ()))
                + len(getattr(self, "skipped_waiting", ())),
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                num_scheduled_requests=num_scheduled_requests,
                prefill_scheduled=prefill_scheduled,
            )

        Scheduler._apply_ascend_proposal_gate = _apply_ascend_proposal_gate

    original_init = Scheduler.__init__
    if getattr(original_init, "_vllm_ascend_dynamic_gate_patched", False):
        return

    @wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._latest_proposal_lengths = {}
        self._ascend_proposal_gate = None

        vllm_config = args[0] if args else kwargs.get("vllm_config")
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        dynamic_spec_config = additional_config.get("dynamic_spec_config", {})
        if not isinstance(dynamic_spec_config, dict):
            dynamic_spec_config = {}
        if not dynamic_spec_config.get("proposal_gate_enabled", False):
            return

        try:
            from vllm_ascend.spec_decode.dynamic.proposal_gate import ProposalGate

            gate_params = dynamic_spec_config.get("proposal_gate_params", {})
            if not isinstance(gate_params, dict):
                raise TypeError("proposal_gate_params must be a dict")
            accepted = {
                key: value
                for key, value in gate_params.items()
                if key
                in {
                    "enter_ratio",
                    "exit_ratio",
                    "max_avg_scheduled_tokens",
                    "enter_steps",
                    "exit_steps",
                }
            }
            self._ascend_proposal_gate = ProposalGate(
                max_num_seqs=getattr(self, "max_num_running_reqs", 1),
                **accepted,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize Ascend proposal gate compatibility patch: %s",
                exc,
            )

    _patched_init._vllm_ascend_dynamic_gate_patched = True  # type: ignore[attr-defined]
    Scheduler.__init__ = _patched_init


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

    # vLLM commit 6ec76df8 added this field and the corresponding scheduler
    # plumbing.  Older vLLM checkouts can still execute the Ascend worker if we
    # install the small side-channel update here instead of modifying vLLM.
    from vllm.v1 import outputs as outputs_mod

    has_native_proposal_lengths = "proposal_lengths" in getattr(
        outputs_mod.ModelRunnerOutput, "__dataclass_fields__", {}
    )
    if has_native_proposal_lengths:
        return

    def _update_proposal_lengths(self, model_runner_output) -> None:
        lengths = getattr(model_runner_output, "proposal_lengths", None)
        if lengths is None:
            return
        req_ids = getattr(model_runner_output, "req_ids", ())
        if len(req_ids) != len(lengths):
            logger.warning(
                "Ignoring malformed proposal_lengths: %d request ids vs %d lengths",
                len(req_ids),
                len(lengths),
            )
            return
        latest = getattr(self, "_latest_proposal_lengths", None)
        if latest is None:
            latest = self._latest_proposal_lengths = {}
        for req_id, length in zip(req_ids, lengths):
            length = max(int(length), 0)
            latest[req_id] = length
            request = getattr(self, "requests", {}).get(req_id)
            if request is None or request.is_finished():
                continue
            if request.is_prefill_chunk:
                request.spec_token_ids = []
            else:
                request.spec_token_ids = [-1] * length

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

        # Apply the newly proposed logical width for the *next* schedule.  The
        # current output still uses the width that was scheduled before the
        # worker executed, so this must happen after upstream bookkeeping.
        _update_proposal_lengths(self, model_runner_output)

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
    _patch_scheduler_dynamic_gate_compat()
    _patch_scheduler_update_after_schedule()
    _patch_scheduler_make_cached_request_data()
    _patch_scheduler_update_from_output()
    _patch_model_config_validation()


_apply_patch()
