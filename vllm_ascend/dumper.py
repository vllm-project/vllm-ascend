#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import fcntl
import json
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2


class Dumper:
    """Manages observability triggers and debugger lifecycle.

    This class keeps observability-related mutable state out of the model
    runner and centralizes request-scoped trigger logic.
    """

    def __init__(self, runner: NPUModelRunner | NPUModelRunnerV2, dynamic_dump_config: Any):
        self.runner = runner
        self.full_log_requests_this_step: dict[str, bool] = {}
        self._debugger: Any | None = None

        self._mtp_acceptance_history: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        self._mtp_acceptance_window = dynamic_dump_config.mtp_acceptance_window
        self._mtp_acceptance_low_threshold = dynamic_dump_config.mtp_acceptance_low_threshold
        self._mtp_acceptance_len_low_threshold = dynamic_dump_config.mtp_acceptance_len_low_threshold
        self._mtp_acceptance_high_threshold = dynamic_dump_config.mtp_acceptance_high_threshold
        self._mtp_acceptance_len_high_threshold = dynamic_dump_config.mtp_acceptance_len_high_threshold
        self._msprobe_dump_cooldown_seconds = dynamic_dump_config.msprobe_dump_cooldown_seconds
        self._msprobe_dump_max_times = dynamic_dump_config.msprobe_dump_max_times

        self._msprobe_dump_total_count = 0
        self._msprobe_dumped_req_ids: set[str] = set()
        self._msprobe_last_dump_ts: float | None = None
        self._msprobe_dump_disable_delay_rounds = 0
        self._msprobe_dump_active = False
        self._debugger_started = False
        # Keep an internal alias so all debug-log-full writes are centralized.
        self._debug_log_full_by_req_id: dict[str, bool] = self.full_log_requests_this_step

        logger.info_once(
            "Dynamic dump config applied: mtp_acceptance_window=%d "
            "mtp_acceptance_low_threshold=%.4f "
            "mtp_acceptance_len_low_threshold=%.4f "
            "mtp_acceptance_high_threshold=%.4f "
            "mtp_acceptance_len_high_threshold=%.4f "
            "msprobe_dump_cooldown_seconds=%d msprobe_dump_max_times=%d",
            self._mtp_acceptance_window,
            self._mtp_acceptance_low_threshold,
            self._mtp_acceptance_len_low_threshold,
            self._mtp_acceptance_high_threshold,
            self._mtp_acceptance_len_high_threshold,
            self._msprobe_dump_cooldown_seconds,
            self._msprobe_dump_max_times,
        )

        # Keep debugger lifecycle fully encapsulated in Dumper.
        self._init_debugger(self.runner.compilation_config.cudagraph_mode)

    def _init_debugger(self, cudagraph_mode: CUDAGraphMode):
        dump_cfg = self.runner.ascend_config.dump_config_path
        if dump_cfg is None:
            self._debugger = None
            return None
        if cudagraph_mode == CUDAGraphMode.NONE:
            from msprobe.pytorch import PrecisionDebugger

            self._debugger = PrecisionDebugger(dump_cfg)
            return self._debugger

        try:
            from msprobe.pytorch import AclGraphDumper
        except Exception as exc:
            raise RuntimeError(
                "Failed to import AclGraphDumper from msprobe. "
                "Please install/rebuild msprobe with aclgraph_dump enabled."
            ) from exc
        self._debugger = AclGraphDumper(dump_cfg)
        return self._debugger

    def start_dump_data(self) -> None:
        # Always clear per-step flags, even when debugger is inactive.
        self.full_log_requests_this_step.clear()
        if self._debugger is None or self._debugger_started:
            return
        self._debugger.start(self.runner.model)
        self._debugger_started = True

    def finalize_dump_data(self, **kwargs) -> None:
        if self._debugger is None or not self._debugger_started:
            return
        if hasattr(self._debugger, "stop"):
            self._debugger.stop()
            self._debugger_started = False

        self._debugger.step(**kwargs)
        self.disable_msprobe_dump_if_needed()

    def check_acceptance_anomaly(
        self,
        req_idx: int,
        req_id: str,
        req_state: Any,
        accepted_token_num: int,
        sampled_ids: list[int] | torch.Tensor | None = None,
    ) -> None:
        if self._msprobe_dump_max_times == 0:
            return
        if not req_id or not getattr(self.runner, "need_accepted_tokens", False):
            return
        if not get_pp_group().is_last_rank:
            logger.warning("[Anomaly MTP] req_id=%s not last pp rank", req_id)
            return
        if not self.is_related_local_request(req_id, req_idx):
            return
        log_leader = self.runner.tp_rank == 0
        draft_len = getattr(req_state, "prev_num_draft_len", 0) or 0
        if draft_len <= 0:
            if log_leader:
                logger.warning("[Anomaly MTP] req_id=%s draft_len=%d", req_id, draft_len)
            return
        self._debug_log_full_by_req_id.pop(req_id, None)
        accepted_draft_tokens = max(0, accepted_token_num - 1)
        invalid_spec_tokens = 0
        effective_draft_len = max(0, draft_len - invalid_spec_tokens)
        history = self._mtp_acceptance_history[req_id]
        history.append((accepted_draft_tokens, effective_draft_len))
        if len(history) > self._mtp_acceptance_window:
            history.popleft()

        prompt_token_ids_raw = getattr(req_state, "prompt_token_ids", None)
        req_output_token_ids = getattr(self.runner.input_batch, "req_output_token_ids", None)
        if req_output_token_ids is not None and 0 <= req_idx < len(req_output_token_ids):
            output_token_ids_raw = req_output_token_ids[req_idx]
        else:
            output_token_ids_raw = getattr(req_state, "output_token_ids", None)
        prompt_token_count = len(prompt_token_ids_raw) if prompt_token_ids_raw is not None else 0
        output_token_count = len(output_token_ids_raw) if output_token_ids_raw is not None else 0
        accepted_sum = sum(accepted for accepted, _ in history)
        draft_sum = sum(draft for _, draft in history)
        acceptance_rate = accepted_sum / draft_sum if draft_sum > 0 else 0.0
        acceptance_len = accepted_sum / len(history) if history else 0.0

        if log_leader:
            logger.info(
                "[Anomaly MTP short] req_id=%s draft_len=%d effective_draft_len=%d "
                "invalid_spec_tokens=%d accepted_count=%d accepted_draft_count=%d "
                "accept_rate=%.4f accept_len=%.4f window=%d accepted=%d drafted=%d "
                "prompt_tokens=%d output_tokens=%d",
                req_id,
                draft_len,
                effective_draft_len,
                invalid_spec_tokens,
                accepted_token_num,
                accepted_draft_tokens,
                acceptance_rate,
                acceptance_len,
                len(history),
                accepted_sum,
                draft_sum,
                prompt_token_count,
                output_token_count,
            )

        if len(history) < self._mtp_acceptance_window:
            return

        should_log_full = bool(
            (
                acceptance_rate < self._mtp_acceptance_low_threshold
                and acceptance_len < self._mtp_acceptance_len_low_threshold
            )
            or (
                acceptance_rate > self._mtp_acceptance_high_threshold
                and acceptance_len > self._mtp_acceptance_len_high_threshold
            )
        )
        if not should_log_full:
            return
        if not self.enable_msprobe_dump_if_needed(req_id, req_idx=req_idx):
            return

        if log_leader:
            self._debug_log_full_by_req_id[req_id] = True
        if log_leader:
            sampled_token_ids = self._normalize_token_ids(sampled_ids)
            self.log_mtp_token_details(
                req_id=req_id,
                sampled_ids=sampled_token_ids,
                accepted_token_num=accepted_token_num,
                prompt_token_ids_raw=prompt_token_ids_raw,
                output_token_ids_raw=output_token_ids_raw,
            )

    def log_mtp_token_details(
        self,
        req_id: str,
        sampled_ids: list[int],
        accepted_token_num: int,
        prompt_token_ids_raw: Any,
        output_token_ids_raw: Any,
    ) -> None:
        accepted_token_ids = sampled_ids[:accepted_token_num] if accepted_token_num > 0 else []
        prompt_token_ids = list(prompt_token_ids_raw) if prompt_token_ids_raw is not None else []
        output_token_ids = list(output_token_ids_raw) if output_token_ids_raw is not None else []
        output_token_ids = [
            output_token_id.item() if isinstance(output_token_id, torch.Tensor) else output_token_id
            for output_token_id in output_token_ids
        ]

        logger.info("[Anomaly MTP] req_id=%s sampled_token_ids=%s", req_id, sampled_ids)
        logger.info("[Anomaly MTP] req_id=%s accepted_token_ids=%s", req_id, accepted_token_ids)
        logger.info(
            "[Anomaly MTP] req_id=%s prompt_token_count=%d prompt_token_ids=%s",
            req_id,
            len(prompt_token_ids),
            prompt_token_ids,
        )
        logger.info(
            "[Anomaly MTP] req_id=%s output_token_count=%d output_token_ids=%s",
            req_id,
            len(output_token_ids),
            output_token_ids,
        )

    @contextmanager
    def lock_msprobe_config(self, config_path: Path):
        lock_path = Path(f"{config_path}.lock")
        os.makedirs(lock_path.parent, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def disable_msprobe_dump_if_needed(self) -> None:
        if not self._msprobe_dump_active:
            return
        if self._debugger is None:
            return
        if self._msprobe_dump_disable_delay_rounds > 0:
            self._msprobe_dump_disable_delay_rounds -= 1
            logger.info(
                "[Anomaly msprobe] defer rollback dump_enable for one round. remaining_delay_rounds=%d",
                self._msprobe_dump_disable_delay_rounds,
            )
            return

        if not self.set_msprobe_dump_state(False):
            return
        self._msprobe_dump_active = False
        self._msprobe_dump_disable_delay_rounds = 0
        logger.info("[Anomaly msprobe] disable msprobe dump succeeded.")
        self._debugger._maybe_reload_config(force=True)

    def set_msprobe_dump_state(self, dump_state: bool) -> bool:
        dump_cfg = self.runner.ascend_config.dump_config_path
        if not dump_cfg:
            logger.error("[Anomaly msprobe] set msprobe dump state failed, because dump_config_path is empty")
            return False

        config_path = Path(dump_cfg)
        if not config_path.exists():
            logger.error(
                "[Anomaly msprobe] set msprobe dump state failed, because config file not found. path=%s",
                str(config_path),
            )
            return False

        try:
            with self.lock_msprobe_config(config_path):
                with config_path.open("r", encoding="utf-8") as f:
                    config_obj = json.load(f)

                if not isinstance(config_obj, dict):
                    logger.error(
                        "[Anomaly msprobe] set msprobe dump state failed, because json root is not object. type=%s",
                        type(config_obj).__name__,
                    )
                    return False

                ori_value = config_obj.get("dump_enable")
                if ori_value != dump_state:
                    config_obj["dump_enable"] = dump_state
                    with config_path.open("w", encoding="utf-8") as f:
                        json.dump(config_obj, f, ensure_ascii=False, indent=2)
                        f.write("\n")
            return True
        except Exception as e:
            logger.error(
                "[Anomaly msprobe] set msprobe dump state failed, path=%s error=%s",
                str(config_path),
                e,
            )
            return False

    def is_related_local_request(self, req_id: str, req_idx: int | None = None) -> bool:
        input_batch = getattr(self.runner, "input_batch", None)
        req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None

        # v2 (and batch-local) path: req_idx is the position in input_batch.req_ids.
        if req_ids is not None and req_idx is not None:
            if req_idx < 0 or req_idx >= len(req_ids) or req_ids[req_idx] != req_id:
                return False
            requests = getattr(self.runner, "requests", None)
            if requests is not None and req_id not in requests:
                return False
            req_states = getattr(self.runner, "req_states", None)
            req_id_to_index = getattr(req_states, "req_id_to_index", None)
            if req_id_to_index is not None and req_id not in req_id_to_index:
                return False
            discard_request_mask = getattr(self.runner, "discard_request_mask", None)
            if discard_request_mask is not None and hasattr(discard_request_mask, "np"):
                if req_idx < len(discard_request_mask.np) and discard_request_mask.np[req_idx]:
                    return False
            return True

        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if req_id_to_index is None:
            req_states = getattr(self.runner, "req_states", None)
            req_id_to_index = getattr(req_states, "req_id_to_index", None)
        if req_id_to_index is None:
            return False

        mapped_idx = req_id_to_index.get(req_id)
        if mapped_idx is None:
            return False

        if req_idx is not None and mapped_idx != req_idx:
            if self.runner.tp_rank == 0:
                logger.warning(
                    "[Anomaly msprobe] req_id=%s skip dump: req_idx mismatch input=%d mapped=%d",
                    req_id,
                    req_idx,
                    mapped_idx,
                )
            return False

        num_reqs = getattr(input_batch, "num_reqs", None)
        if num_reqs is None:
            req_states = getattr(self.runner, "req_states", None)
            num_reqs_np = getattr(req_states, "num_reqs_np", None)
            if num_reqs_np is not None:
                num_reqs = int(num_reqs_np[0])
        if num_reqs is None:
            return False

        if mapped_idx < 0 or mapped_idx >= num_reqs:
            return False

        if req_ids is not None and mapped_idx < len(req_ids) and req_ids[mapped_idx] != req_id:
            return False

        discard_request_mask = getattr(self.runner, "discard_request_mask", None)
        if discard_request_mask is not None and hasattr(discard_request_mask, "np"):
            if mapped_idx < len(discard_request_mask.np) and discard_request_mask.np[mapped_idx]:
                return False

        requests = getattr(self.runner, "requests", None)
        if requests is not None:
            return req_id in requests
        return True

    def enable_msprobe_dump_if_needed(self, req_id: str, req_idx: int | None = None) -> bool:
        if self._debugger is None:
            return False
        if not get_pp_group().is_last_rank:
            return False
        if not self.is_related_local_request(req_id, req_idx):
            if self.runner.tp_rank == 0:
                logger.info_once(
                    "[Anomaly msprobe] req_id=%s skip dump: not a related local request",
                    req_id,
                )
            return False
        if req_id in self._msprobe_dumped_req_ids:
            logger.info_once("[Anomaly msprobe] req_id=%s skip dump: request already dumped once", req_id)
            return False
        if self._msprobe_dump_total_count >= self._msprobe_dump_max_times:
            logger.info_once(
                "[Anomaly msprobe] req_id=%s skip dump: reached local max dump times=%d",
                req_id,
                self._msprobe_dump_max_times,
            )
            return False

        now_ts = time.time()
        elapsed = None if self._msprobe_last_dump_ts is None else now_ts - self._msprobe_last_dump_ts
        if elapsed is not None and elapsed < self._msprobe_dump_cooldown_seconds:
            logger.info_once(
                "[Anomaly msprobe] req_id=%s skip dump: cooldown active elapsed=%.2fs remaining=%.2fs",
                req_id,
                elapsed,
                self._msprobe_dump_cooldown_seconds - elapsed,
            )
            return False

        if not self.set_msprobe_dump_state(True):
            return False
        self._msprobe_dump_active = True
        self._msprobe_dumped_req_ids.add(req_id)
        self._msprobe_dump_total_count += 1
        self._msprobe_last_dump_ts = now_ts
        self._msprobe_dump_disable_delay_rounds = 1

        if self.runner.tp_rank == 0 and get_pp_group().is_last_rank:
            self.save_sample_param(target_req_id=req_id)

        logger.info(
            "[Anomaly msprobe] req_id=%s set msprobe dump state succeeded. local_dump_count=%d/%d",
            req_id,
            self._msprobe_dump_total_count,
            self._msprobe_dump_max_times,
        )
        self._debugger._maybe_reload_config(force=True)
        return True

    def check_all_acceptance(
        self,
        sampled_tokens: torch.Tensor,
        accepted_token_nums: Any,
    ) -> None:
        """Batch entry for v1/v2: fan out into check_acceptance_anomaly per request.

        Args:
            sampled_tokens: [num_reqs, max_tokens] sampled token ids.
            accepted_token_nums: per-request accepted token counts (tensor/ndarray/list).
        """
        if self._msprobe_dump_max_times == 0:
            return
        if not getattr(self.runner, "need_accepted_tokens", False):
            return
        input_batch = getattr(self.runner, "input_batch", None)
        if input_batch is None or not getattr(input_batch, "req_ids", None):
            return

        num_reqs = len(input_batch.req_ids)
        if torch.is_tensor(accepted_token_nums):
            accepted_list = accepted_token_nums[:num_reqs].tolist()
        else:
            accepted_list = [int(x) for x in accepted_token_nums[:num_reqs]]

        sampled_token_rows = sampled_tokens[:num_reqs]
        requests = getattr(self.runner, "requests", None)
        draft_lens = getattr(input_batch, "num_draft_tokens_per_req", None)

        for batch_idx, req_id in enumerate(input_batch.req_ids):
            accepted_token_num = int(accepted_list[batch_idx])
            sampled_ids = sampled_token_rows[batch_idx]

            if requests is not None and req_id in requests:
                req_state = requests[req_id]
            else:
                draft_len = int(draft_lens[batch_idx]) if draft_lens is not None else 0
                req_state = SimpleNamespace(
                    prev_num_draft_len=draft_len,
                    prompt_token_ids=None,
                    output_token_ids=None,
                )

            self.check_acceptance_anomaly(
                req_idx=batch_idx,
                req_id=req_id,
                req_state=req_state,
                accepted_token_num=accepted_token_num,
                sampled_ids=sampled_ids,
            )

    def save_sample_param(self, target_req_id: str) -> None:
        input_batch = getattr(self.runner, "input_batch", None)
        if input_batch is None:
            return
        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        if sampling_metadata is None:
            return
        req_ids = input_batch.req_ids
        for req_idx, req_id in enumerate(req_ids):
            if target_req_id and req_id != target_req_id:
                continue

            temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
            topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
            topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

            freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
            pres_pen = sampling_metadata.presence_penalties[req_idx].item()
            rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

            req_bad_words = sampling_metadata.bad_words_token_ids.get(req_idx, [])
            req_output_tokens = (
                sampling_metadata.output_token_ids[req_idx]
                if sampling_metadata.output_token_ids and req_idx < len(sampling_metadata.output_token_ids)
                else []
            )
            req_spec_tokens = (
                sampling_metadata.spec_token_ids[req_idx]
                if sampling_metadata.spec_token_ids and req_idx < len(sampling_metadata.spec_token_ids)
                else None
            )
            if sampling_metadata.logprob_token_ids:
                req_logprob_tokens = sampling_metadata.logprob_token_ids.get(req_idx, [])
            else:
                req_logprob_tokens = None

            logger.info(
                "[SamplingMeta] req_id=%s req_idx=%d "
                "dp_rank=%d tp_rank=%d "
                "temperature=%.4f top_k=%s top_p=%.4f "
                "freq_pen=%.4f pres_pen=%.4f rep_pen=%.4f "
                "bad_words_group_num=%d output_tokens_len=%d spec_tokens_len=%s logprob_target_tokens_len=%s "
                "all_greedy=%s all_random=%s max_num_logprobs=%s",
                req_id,
                req_idx,
                self.runner.dp_rank,
                self.runner.tp_rank,
                temp if temp is not None else -1,
                topk,
                topp if topp is not None else 1.0,
                freq_pen,
                pres_pen,
                rep_pen,
                len(req_bad_words),
                len(req_output_tokens),
                len(req_spec_tokens) if req_spec_tokens else None,
                len(req_logprob_tokens) if req_logprob_tokens else None,
                sampling_metadata.all_greedy,
                sampling_metadata.all_random,
                sampling_metadata.max_num_logprobs,
            )

    @staticmethod
    def _normalize_token_ids(token_ids: Any) -> list[int]:
        if token_ids is None:
            return []
        if torch.is_tensor(token_ids):
            return token_ids.tolist()
        return list(token_ids)
