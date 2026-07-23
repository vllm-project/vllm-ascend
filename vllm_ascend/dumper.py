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

        self._enable_spec_acceptance_check = bool(dynamic_dump_config.enable_spec_acceptance_check)
        self._enable_token_logprob_check = bool(dynamic_dump_config.enable_token_logprob_check)

        self._spec_acceptance_history: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        self._spec_acceptance_window = dynamic_dump_config.spec_acceptance_window
        self._spec_acceptance_low_threshold = dynamic_dump_config.spec_acceptance_low_threshold
        self._spec_acceptance_len_low_threshold = dynamic_dump_config.spec_acceptance_len_low_threshold
        self._spec_acceptance_high_threshold = dynamic_dump_config.spec_acceptance_high_threshold
        self._spec_acceptance_len_high_threshold = dynamic_dump_config.spec_acceptance_len_high_threshold
        self._dynamic_dump_cooldown_seconds = dynamic_dump_config.dynamic_dump_cooldown_seconds
        self._dynamic_dump_max_times = dynamic_dump_config.dynamic_dump_max_times

        self._token_logprob_window = dynamic_dump_config.token_logprob_window
        self._token_logprob_stride = dynamic_dump_config.token_logprob_stride
        self._token_logprob_topk = dynamic_dump_config.token_logprob_topk
        self._ill_window_thresh = {
            1: dynamic_dump_config.ill_rare_window_thresh,
            2: dynamic_dump_config.ill_garbled_window_thresh,
            3: dynamic_dump_config.ill_repet_window_thresh,
            4: dynamic_dump_config.ill_nan_window_thresh,
        }
        # Per-request buffers: token ids + topk logprob dicts; hit counts by ill_type.
        self._token_logprob_buf: dict[str, deque[tuple[int, dict[int, float]]]] = {}
        self._token_logprob_since_check: dict[str, int] = defaultdict(int)
        self._token_logprob_checked: set[str] = set()
        self._ill_window_hits: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._ill_detector: Any | None = None
        self._ill_detector_init_failed = False

        self._msprobe_dump_total_count = 0
        self._msprobe_dumped_req_ids: set[str] = set()
        self._msprobe_last_dump_ts: float | None = None
        self._msprobe_dump_active = False
        self._debugger_started = False
        # Keep an internal alias so all debug-log-full writes are centralized.
        self._debug_log_full_by_req_id: dict[str, bool] = self.full_log_requests_this_step

        logger.info_once(
            "Dynamic dump config applied: enable_spec_acceptance_check=%s "
            "enable_token_logprob_check=%s spec_acceptance_window=%d "
            "spec_acceptance_low_threshold=%.4f "
            "spec_acceptance_len_low_threshold=%.4f "
            "spec_acceptance_high_threshold=%.4f "
            "spec_acceptance_len_high_threshold=%.4f "
            "token_logprob_window=%d token_logprob_stride=%d token_logprob_topk=%d "
            "ill_thresh(rare/garbled/repeat/nan)=%d/%d/%d/%d "
            "dynamic_dump_cooldown_seconds=%d dynamic_dump_max_times=%d",
            self._enable_spec_acceptance_check,
            self._enable_token_logprob_check,
            self._spec_acceptance_window,
            self._spec_acceptance_low_threshold,
            self._spec_acceptance_len_low_threshold,
            self._spec_acceptance_high_threshold,
            self._spec_acceptance_len_high_threshold,
            self._token_logprob_window,
            self._token_logprob_stride,
            self._token_logprob_topk,
            self._ill_window_thresh[1],
            self._ill_window_thresh[2],
            self._ill_window_thresh[3],
            self._ill_window_thresh[4],
            self._dynamic_dump_cooldown_seconds,
            self._dynamic_dump_max_times,
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

    def check_spec_acceptance_anomaly(
        self,
        req_idx: int,
        req_id: str,
        req_state: Any,
        accepted_token_num: int,
        sampled_ids: list[int] | torch.Tensor | None = None,
    ) -> None:
        if not req_id:
            return
        if not get_pp_group().is_last_rank:
            logger.warning("[Anomaly spec] req_id=%s not last pp rank", req_id)
            return
        if not self.is_related_local_request(req_id, req_idx):
            return
        log_leader = self.runner.tp_rank == 0
        draft_len = getattr(req_state, "prev_num_draft_len", 0) or 0
        if draft_len <= 0:
            if log_leader:
                logger.warning("[Anomaly spec] req_id=%s draft_len=%d", req_id, draft_len)
            return
        self._debug_log_full_by_req_id.pop(req_id, None)
        accepted_draft_tokens = max(0, accepted_token_num - 1)
        history = self._spec_acceptance_history[req_id]
        history.append((accepted_draft_tokens, draft_len))
        if len(history) > self._spec_acceptance_window:
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
                "[Anomaly spec short] req_id=%s draft_len=%d "
                "accepted_count=%d accepted_draft_count=%d "
                "accept_rate=%.4f accept_len=%.4f window=%d accepted=%d drafted=%d "
                "prompt_tokens=%d output_tokens=%d",
                req_id,
                draft_len,
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

        if len(history) < self._spec_acceptance_window:
            return

        should_log_full = bool(
            (
                acceptance_rate < self._spec_acceptance_low_threshold
                and acceptance_len < self._spec_acceptance_len_low_threshold
            )
            or (
                acceptance_rate > self._spec_acceptance_high_threshold
                and acceptance_len > self._spec_acceptance_len_high_threshold
            )
        )
        if not should_log_full:
            return
        if not self.enable_msprobe_dump_if_needed(req_id, req_idx=req_idx):
            return

        if log_leader:
            self._debug_log_full_by_req_id[req_id] = True
            self.log_spec_token_details(
                req_id=req_id,
                sampled_ids=self._normalize_token_ids(sampled_ids),
                accepted_token_num=accepted_token_num,
                prompt_token_ids_raw=prompt_token_ids_raw,
                output_token_ids_raw=output_token_ids_raw,
            )

    def log_spec_token_details(
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

        logger.info("[Anomaly spec] req_id=%s sampled_token_ids=%s", req_id, sampled_ids)
        logger.info("[Anomaly spec] req_id=%s accepted_token_ids=%s", req_id, accepted_token_ids)
        logger.info(
            "[Anomaly spec] req_id=%s prompt_token_count=%d prompt_token_ids=%s",
            req_id,
            len(prompt_token_ids),
            prompt_token_ids,
        )
        logger.info(
            "[Anomaly spec] req_id=%s output_token_count=%d output_token_ids=%s",
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
        if not self.set_msprobe_dump_state(False):
            return
        self._msprobe_dump_active = False
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
        if self._msprobe_dump_total_count >= self._dynamic_dump_max_times:
            logger.info_once(
                "[Anomaly msprobe] req_id=%s skip dump: reached local max dump times=%d",
                req_id,
                self._dynamic_dump_max_times,
            )
            return False

        now_ts = time.time()
        elapsed = None if self._msprobe_last_dump_ts is None else now_ts - self._msprobe_last_dump_ts
        if elapsed is not None and elapsed < self._dynamic_dump_cooldown_seconds:
            logger.info_once(
                "[Anomaly msprobe] req_id=%s skip dump: cooldown active elapsed=%.2fs remaining=%.2fs",
                req_id,
                elapsed,
                self._dynamic_dump_cooldown_seconds - elapsed,
            )
            return False

        if not self.set_msprobe_dump_state(True):
            return False
        self._msprobe_dump_active = True
        self._msprobe_dumped_req_ids.add(req_id)
        self._msprobe_dump_total_count += 1
        self._msprobe_last_dump_ts = now_ts

        if self.runner.tp_rank == 0 and get_pp_group().is_last_rank:
            self.save_sample_param(target_req_id=req_id)

        logger.info(
            "[Anomaly msprobe] req_id=%s set msprobe dump state succeeded. local_dump_count=%d/%d",
            req_id,
            self._msprobe_dump_total_count,
            self._dynamic_dump_max_times,
        )
        self._debugger._maybe_reload_config(force=True)
        return True

    def check_all_spec_acceptance(
        self,
        sampled_tokens: torch.Tensor,
        accepted_token_nums: Any,
    ) -> None:
        """Batch entry for v1/v2: fan out into check_spec_acceptance_anomaly per request.

        Args:
            sampled_tokens: [num_reqs, max_tokens] sampled token ids.
            accepted_token_nums: per-request accepted token counts (tensor/ndarray/list).
        """
        if not self._enable_spec_acceptance_check:
            return
        if self._dynamic_dump_max_times == 0:
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

            self.check_spec_acceptance_anomaly(
                req_idx=batch_idx,
                req_id=req_id,
                req_state=req_state,
                accepted_token_num=accepted_token_num,
                sampled_ids=sampled_ids,
            )

    def clear_finished_requests(self, finished_req_ids: Any) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self._spec_acceptance_history.pop(req_id, None)
            self._token_logprob_buf.pop(req_id, None)
            self._token_logprob_since_check.pop(req_id, None)
            self._token_logprob_checked.discard(req_id)
            self._ill_window_hits.pop(req_id, None)

    def check_all_token_logprobs(
        self,
        sampled_token_ids: list[list[int]] | None,
        logprobs_lists: Any | None,
    ) -> None:
        """Batch entry: append accepted tokens/logprobs and run ILLDetector when due.

        Finished-request cleanup is the caller's responsibility via
        ``clear_finished_requests`` (done once before MTP + token checks in v1).
        """
        if not self._enable_token_logprob_check:
            return
        if self._dynamic_dump_max_times == 0:
            return
        if not get_pp_group().is_last_rank:
            return
        if sampled_token_ids is None or logprobs_lists is None:
            return

        input_batch = getattr(self.runner, "input_batch", None)
        if input_batch is None or not getattr(input_batch, "req_ids", None):
            return

        detector = self._get_ill_detector()
        if detector is None:
            return

        log_leader = self.runner.tp_rank == 0
        model_config = self._model_config_for_detector()
        for batch_idx, req_id in enumerate(input_batch.req_ids):
            if batch_idx >= len(sampled_token_ids):
                break
            if not self.is_related_local_request(req_id, batch_idx):
                continue
            token_ids = sampled_token_ids[batch_idx]
            if not token_ids:
                continue
            topk_rows = self._extract_req_topk_logprobs(logprobs_lists, batch_idx, len(token_ids))
            if topk_rows is None:
                continue
            self.check_token_logprob_anomaly(
                req_idx=batch_idx,
                req_id=req_id,
                token_ids=token_ids,
                topk_logprobs=topk_rows,
                model_config=model_config,
                detector=detector,
                log_leader=log_leader,
            )

    def check_token_logprob_anomaly(
        self,
        req_idx: int,
        req_id: str,
        token_ids: list[int],
        topk_logprobs: list[dict[int, float]],
        model_config: Any,
        detector: Any,
        log_leader: bool,
    ) -> None:
        if not token_ids or not topk_logprobs:
            return
        n = min(len(token_ids), len(topk_logprobs))
        buf = self._token_logprob_buf.get(req_id)
        if buf is None:
            buf = deque(maxlen=self._token_logprob_window)
            self._token_logprob_buf[req_id] = buf

        for i in range(n):
            buf.append((int(token_ids[i]), topk_logprobs[i]))
        self._token_logprob_since_check[req_id] += n

        if len(buf) < self._token_logprob_window:
            return
        already_checked = req_id in self._token_logprob_checked
        if already_checked and self._token_logprob_since_check[req_id] < self._token_logprob_stride:
            return

        self._token_logprob_since_check[req_id] = 0
        self._token_logprob_checked.add(req_id)
        tokens = [tid for tid, _ in buf]
        topk_dicts = [lp for _, lp in buf]

        if log_leader:
            logger.info(
                "[Anomaly token_logprob] detect req_id=%s window=%d active_reqs=%d",
                req_id,
                len(buf),
                len(self._token_logprob_buf),
            )

        try:
            result = detector.detector(topk_dicts, tokens, model_config)
        except Exception as e:
            logger.error("[Anomaly token_logprob] detector failed req_id=%s error=%s", req_id, e)
            return

        if not getattr(result, "is_ill", False):
            return

        ill_type = int(getattr(result, "ill_type", 0) or 0)
        thresh = self._ill_window_thresh.get(ill_type)
        if thresh is None:
            return
        hits = self._ill_window_hits[req_id]
        hits[ill_type] += 1
        if log_leader:
            logger.info(
                "[Anomaly token_logprob] hit req_id=%s ill_type=%d hits=%d/%d active_reqs=%d",
                req_id,
                ill_type,
                hits[ill_type],
                thresh,
                len(self._token_logprob_buf),
            )
        if hits[ill_type] < thresh:
            return
        if not self.enable_msprobe_dump_if_needed(req_id, req_idx=req_idx):
            return
        if log_leader:
            self._debug_log_full_by_req_id[req_id] = True

    def _get_ill_detector(self) -> Any | None:
        if self._ill_detector is not None:
            return self._ill_detector
        if self._ill_detector_init_failed:
            return None
        try:
            import msprobe.response_anomaly as response_anomaly
            from msprobe.response_anomaly.detector import ILLDetector

            base = Path(response_anomaly.__file__).resolve().parent
            detector = ILLDetector(
                str(base / "configs" / "config.yaml"),
                str(base / "configs" / "mtype_config.json"),
                str(base / "token2category"),
            )
            # One call == one window: outer dumper owns sliding.
            detector.window_size = self._token_logprob_window
            detector.stride = self._token_logprob_window
            detector.garbled_window_thresh = 0
            detector.single_window_thresh = 0
            detector.multi_window_thresh = 0
            self._ill_detector = detector
            logger.info_once(
                "[Anomaly token_logprob] ILLDetector ready window=%d stride=%d topk=%d",
                self._token_logprob_window,
                self._token_logprob_stride,
                self._token_logprob_topk,
            )
            return self._ill_detector
        except Exception as e:
            self._ill_detector_init_failed = True
            logger.error("[Anomaly token_logprob] failed to init ILLDetector: %s", e)
            return None

    def _model_config_for_detector(self) -> dict[str, str]:
        model_config = getattr(self.runner.vllm_config, "model_config", None)
        raw_name = ""
        if model_config is not None:
            raw_name = str(getattr(model_config, "model", None) or getattr(model_config, "model_id", "") or "")
        return {"model_name": Path(raw_name).name if raw_name else ""}

    def _extract_req_topk_logprobs(
        self,
        logprobs_lists: Any,
        req_idx: int,
        num_tokens: int,
    ) -> list[dict[int, float]] | None:
        """Convert LogprobsLists rows for one request into detector topk dicts."""
        try:
            token_ids_arr = logprobs_lists.logprob_token_ids
            logprobs_arr = logprobs_lists.logprobs
            cu = getattr(logprobs_lists, "cu_num_generated_tokens", None)
            if cu is not None:
                start = cu[req_idx]
                end = cu[req_idx + 1] if req_idx + 1 < len(cu) else start + num_tokens
            else:
                # Flat layout without cu: assume one row per request when num_tokens==1,
                # otherwise slice contiguous blocks of num_tokens (best-effort).
                if num_tokens == 1:
                    start = req_idx
                    end = req_idx + 1
                else:
                    start = req_idx * num_tokens
                    end = start + num_tokens
            end = min(end, start + num_tokens, len(token_ids_arr))
            if end <= start:
                return None
            rows: list[dict[int, float]] = []
            for row_i in range(start, end):
                rows.append(
                    self._row_to_topk_dict(
                        token_ids_arr[row_i],
                        logprobs_arr[row_i],
                        self._token_logprob_topk,
                    )
                )
            return rows
        except Exception as e:
            logger.error(
                "[Anomaly token_logprob] extract logprobs failed req_idx=%d error=%s",
                req_idx,
                e,
            )
            return None

    @staticmethod
    def _row_to_topk_dict(token_ids_row: Any, logprobs_row: Any, topk: int) -> dict[int, float]:
        tids = token_ids_row.tolist() if hasattr(token_ids_row, "tolist") else list(token_ids_row)
        lps = logprobs_row.tolist() if hasattr(logprobs_row, "tolist") else list(logprobs_row)
        pairs = []
        for tid, lp in zip(tids, lps):
            tid_i = int(tid)
            if tid_i < 0:
                continue
            pairs.append((tid_i, float(lp)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        # Keep insertion order of sorted pairs for detector.
        out: dict[int, float] = {}
        for tid, lp in pairs[:topk]:
            out[tid] = lp
        return out

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
